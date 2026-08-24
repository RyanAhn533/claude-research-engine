# smoke_task_vector_patch.py
# Purpose:
#   Step 0 smoke test for task-vector extraction + residual patching.
#   Uses a simple antonym task with forced-choice logprob scoring.
#
# Example:
#   python smoke_task_vector_patch.py --model Qwen/Qwen2.5-7B-Instruct --load-4bit

import argparse
import json
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


ANTONYM_PAIRS = [
    ("hot", "cold"),
    ("big", "small"),
    ("up", "down"),
    ("left", "right"),
    ("open", "closed"),
    ("fast", "slow"),
    ("light", "dark"),
    ("happy", "sad"),
    ("young", "old"),
    ("early", "late"),
    ("strong", "weak"),
    ("clean", "dirty"),
    ("empty", "full"),
    ("hard", "soft"),
    ("high", "low"),
    ("near", "far"),
    ("inside", "outside"),
    ("start", "finish"),
    ("push", "pull"),
    ("accept", "reject"),
]


@dataclass
class EvalItem:
    x: str
    y: str


def build_instruction() -> str:
    return (
        "Complete the antonym mapping.\n"
        "Given an input word, output its antonym.\n\n"
    )


def build_demo_block(pairs: List[Tuple[str, str]]) -> str:
    s = ""
    for x, y in pairs:
        s += f"Input: {x}\nOutput: {y}\n\n"
    return s


def build_prompt(query: str, demos: Optional[List[Tuple[str, str]]] = None) -> str:
    prompt = build_instruction()
    if demos:
        prompt += build_demo_block(demos)
    prompt += f"Input: {query}\nOutput:"
    return prompt


def build_task_vector_prompt(demos: List[Tuple[str, str]]) -> str:
    # Query-independent prompt for extracting the task vector.
    # Ends at "Input:" so the final token is after seeing the demo set but before any held-out query.
    return build_instruction() + build_demo_block(demos) + "Input:"


def build_control_vector_prompt() -> str:
    # Same high-level instruction, no examples.
    return build_instruction() + "Input:"


def get_layers(model):
    # Qwen/Yi/Mistral/Llama-like HuggingFace models usually expose decoder blocks here.
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise RuntimeError("Could not locate transformer layers. Inspect model architecture.")


@torch.no_grad()
def get_hidden_at_last_token(model, tokenizer, prompt: str, layer_idx: int, device: str):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    out = model(
        **inputs,
        output_hidden_states=True,
        use_cache=False,
    )
    # hidden_states[0] = embedding output, hidden_states[L] = input to layer L
    # For patching into layer L pre-hook, use hidden_states[layer_idx].
    h = out.hidden_states[layer_idx][0, -1, :].detach()
    return h


@torch.no_grad()
def extract_task_vector(model, tokenizer, demos, layer_idx: int, device: str):
    task_prompt = build_task_vector_prompt(demos)
    control_prompt = build_control_vector_prompt()

    h_task = get_hidden_at_last_token(model, tokenizer, task_prompt, layer_idx, device)
    h_ctrl = get_hidden_at_last_token(model, tokenizer, control_prompt, layer_idx, device)

    # Delta vector is safer than raw hidden state.
    theta = h_task - h_ctrl
    return theta


def make_patch_hook(theta: torch.Tensor, alpha: float, patch_pos: int):
    def hook(module, inputs):
        hidden = inputs[0]
        hidden = hidden.clone()
        vec = theta.to(device=hidden.device, dtype=hidden.dtype)
        hidden[:, patch_pos, :] += alpha * vec
        return (hidden,) + tuple(inputs[1:])
    return hook


@torch.no_grad()
def candidate_logprob(
    model,
    tokenizer,
    prompt: str,
    candidate: str,
    device: str,
    layer_idx: Optional[int] = None,
    theta: Optional[torch.Tensor] = None,
    alpha: float = 1.0,
) -> float:
    # Candidate gets a leading space because prompt ends with "Output:"
    cand_text = " " + candidate

    prompt_ids = tokenizer(prompt, add_special_tokens=True).input_ids
    cand_ids = tokenizer(cand_text, add_special_tokens=False).input_ids

    input_ids = torch.tensor([prompt_ids + cand_ids], device=device)
    prompt_len = len(prompt_ids)
    patch_pos = prompt_len - 1

    handle = None
    if theta is not None and layer_idx is not None:
        layers = get_layers(model)
        handle = layers[layer_idx].register_forward_pre_hook(
            make_patch_hook(theta, alpha, patch_pos)
        )

    try:
        out = model(input_ids=input_ids, use_cache=False)
    finally:
        if handle is not None:
            handle.remove()

    logits = out.logits[0]

    # Token at absolute position prompt_len + i is predicted by logits at prompt_len + i - 1.
    total = 0.0
    for i, tok_id in enumerate(cand_ids):
        logit_pos = prompt_len + i - 1
        log_probs = F.log_softmax(logits[logit_pos], dim=-1)
        total += float(log_probs[tok_id].item())

    return total


@torch.no_grad()
def predict_forced_choice(
    model,
    tokenizer,
    prompt: str,
    candidates: List[str],
    device: str,
    layer_idx: Optional[int] = None,
    theta: Optional[torch.Tensor] = None,
    alpha: float = 1.0,
) -> str:
    scores = {}
    for cand in candidates:
        scores[cand] = candidate_logprob(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            candidate=cand,
            device=device,
            layer_idx=layer_idx,
            theta=theta,
            alpha=alpha,
        )
    return max(scores.items(), key=lambda kv: kv[1])[0]


@torch.no_grad()
def evaluate_condition(
    model,
    tokenizer,
    eval_items: List[EvalItem],
    candidates: List[str],
    device: str,
    demos: Optional[List[Tuple[str, str]]] = None,
    layer_idx: Optional[int] = None,
    theta: Optional[torch.Tensor] = None,
    alpha: float = 1.0,
) -> float:
    correct = 0
    total = 0

    for item in eval_items:
        prompt = build_prompt(item.x, demos=demos)

        pred = predict_forced_choice(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            candidates=candidates,
            device=device,
            layer_idx=layer_idx,
            theta=theta,
            alpha=alpha,
        )

        correct += int(pred == item.y)
        total += 1

    return correct / max(total, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--n-eval", type=int, default=12)
    parser.add_argument("--alphas", type=str, default="0.25,0.5,1.0,2.0")
    parser.add_argument("--out", type=str, default="smoke_task_vector_results.json")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    quant_config = None
    if args.load_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
        use_fast=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        device_map="auto" if device == "cuda" else None,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        quantization_config=quant_config,
    )
    model.eval()

    if device == "cpu":
        model.to(device)

    pairs = ANTONYM_PAIRS.copy()
    random.shuffle(pairs)

    demos = pairs[: args.k]
    eval_pairs = pairs[args.k : args.k + args.n_eval]
    eval_items = [EvalItem(x=x, y=y) for x, y in eval_pairs]

    # Forced-choice candidates: all eval targets + demo targets.
    # This makes the task harder than binary choice but still stable.
    candidates = sorted(set([y for _, y in pairs]))

    layers = get_layers(model)
    n_layers = len(layers)
    alphas = [float(x) for x in args.alphas.split(",")]

    zero_acc = evaluate_condition(
        model=model,
        tokenizer=tokenizer,
        eval_items=eval_items,
        candidates=candidates,
        device=device,
        demos=None,
    )

    icl_acc = evaluate_condition(
        model=model,
        tokenizer=tokenizer,
        eval_items=eval_items,
        candidates=candidates,
        device=device,
        demos=demos,
    )

    results = {
        "model": args.model,
        "seed": args.seed,
        "k": args.k,
        "n_eval": len(eval_items),
        "demos": demos,
        "eval": [(x.x, x.y) for x in eval_items],
        "zero_acc": zero_acc,
        "icl_acc": icl_acc,
        "icl_gain": icl_acc - zero_acc,
        "patched": [],
    }

    print(f"[zero] {zero_acc:.3f}")
    print(f"[icl ] {icl_acc:.3f}")
    print(f"[gain] {icl_acc - zero_acc:.3f}")

    for layer_idx in range(0, n_layers):
        theta = extract_task_vector(
            model=model,
            tokenizer=tokenizer,
            demos=demos,
            layer_idx=layer_idx,
            device=device,
        )

        for alpha in alphas:
            patched_acc = evaluate_condition(
                model=model,
                tokenizer=tokenizer,
                eval_items=eval_items,
                candidates=candidates,
                device=device,
                demos=None,
                layer_idx=layer_idx,
                theta=theta,
                alpha=alpha,
            )

            icl_gain = icl_acc - zero_acc
            abs_gain = patched_acc - zero_acc
            recovery = None
            if abs(icl_gain) >= 0.03:
                recovery = abs_gain / icl_gain

            row = {
                "layer": layer_idx,
                "alpha": alpha,
                "patched_acc": patched_acc,
                "absolute_patch_gain": abs_gain,
                "recovery_ratio": recovery,
            }
            results["patched"].append(row)

            rec_str = "NA" if recovery is None else f"{recovery:.3f}"
            print(
                f"[L={layer_idx:02d} alpha={alpha}] "
                f"patched={patched_acc:.3f} "
                f"abs_gain={abs_gain:.3f} "
                f"recovery={rec_str}"
            )

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    best = max(
        results["patched"],
        key=lambda r: r["absolute_patch_gain"],
    )

    print("\n=== BEST BY ABSOLUTE PATCH GAIN ===")
    print(json.dumps(best, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
