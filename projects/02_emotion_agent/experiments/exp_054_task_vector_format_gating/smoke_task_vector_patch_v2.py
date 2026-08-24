# smoke_task_vector_patch_v2.py
# Step 0 smoke (v2): task-vector extraction + residual patching, forced-choice logprob.
#
# v1 failed because antonym + explicit instruction => zero-shot already 1.000 (no ICL gain
# to recover). v2 fix (Hendel/Todd-faithful): NEUTRAL instruction (task conveyed ONLY by
# demos) + algorithmic task with low LM prior (last-letter extraction) so zero-shot is
# uninformed and ICL produces a real, recoverable gain.
#
# Example:
#   python smoke_task_vector_patch_v2.py --model Qwen/Qwen2.5-7B-Instruct --load-4bit --task last_letter

import argparse
import json
import random
import string
from dataclasses import dataclass
from typing import List, Tuple, Optional

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


# Words chosen so first letter != last letter (avoids "first letter" default winning).
WORD_POOL = [
    "apple", "tiger", "lemon", "cloud", "plant", "brush", "zebra", "light",
    "grape", "snake", "river", "stone", "flame", "bread", "chair", "dream",
    "frost", "globe", "honey", "ivory", "joker", "knife", "maple", "novel",
    "ocean", "pearl", "queen", "robot", "sugar", "table", "ultra", "vapor",
    "wagon", "xenon", "yacht", "zonal", "amber", "blaze", "crane", "drift",
]

ANTONYM_PAIRS = [
    ("hot", "cold"), ("big", "small"), ("up", "down"), ("left", "right"),
    ("open", "closed"), ("fast", "slow"), ("light", "dark"), ("happy", "sad"),
    ("young", "old"), ("early", "late"), ("strong", "weak"), ("clean", "dirty"),
    ("empty", "full"), ("hard", "soft"), ("high", "low"), ("near", "far"),
]

# English -> French single-word translation (canonical task-vector task; knowledge is in
# the model but the *task* is only revealed by demos under a neutral instruction).
EN_FR_PAIRS = [
    ("dog", "chien"), ("cat", "chat"), ("house", "maison"), ("water", "eau"),
    ("bread", "pain"), ("book", "livre"), ("tree", "arbre"), ("fish", "poisson"),
    ("milk", "lait"), ("sun", "soleil"), ("moon", "lune"), ("king", "roi"),
    ("queen", "reine"), ("horse", "cheval"), ("flower", "fleur"), ("school", "ecole"),
    ("road", "route"), ("night", "nuit"), ("day", "jour"), ("snow", "neige"),
]


@dataclass
class EvalItem:
    x: str
    y: str


def build_task(task: str, seed: int) -> Tuple[List[Tuple[str, str]], List[str]]:
    """Return (pairs, candidates)."""
    rng = random.Random(seed)
    if task == "last_letter":
        words = WORD_POOL.copy()
        rng.shuffle(words)
        pairs = [(w, w[-1]) for w in words]
        candidates = sorted(set(y for _, y in pairs))
        return pairs, candidates
    elif task == "antonym":
        pairs = ANTONYM_PAIRS.copy()
        rng.shuffle(pairs)
        candidates = sorted(set(y for _, y in pairs))
        return pairs, candidates
    elif task == "en_fr":
        pairs = EN_FR_PAIRS.copy()
        rng.shuffle(pairs)
        candidates = sorted(set(y for _, y in pairs))
        return pairs, candidates
    raise ValueError(f"unknown task {task}")


def build_instruction(neutral: bool) -> str:
    if neutral:
        # Task identity is NOT given; only demos can reveal it.
        return "Follow the pattern shown by the examples.\n\n"
    return "Complete the antonym mapping.\nGiven an input word, output its antonym.\n\n"


def build_demo_block(pairs: List[Tuple[str, str]]) -> str:
    return "".join(f"Input: {x}\nOutput: {y}\n\n" for x, y in pairs)


def build_prompt(query: str, neutral: bool, demos: Optional[List[Tuple[str, str]]] = None) -> str:
    p = build_instruction(neutral)
    if demos:
        p += build_demo_block(demos)
    p += f"Input: {query}\nOutput:"
    return p


def build_task_vector_prompt(neutral: bool, demos: List[Tuple[str, str]]) -> str:
    return build_instruction(neutral) + build_demo_block(demos) + "Input:"


def build_control_vector_prompt(neutral: bool) -> str:
    return build_instruction(neutral) + "Input:"


def get_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise RuntimeError("Could not locate transformer layers.")


@torch.no_grad()
def get_hidden_at_last_token(model, tokenizer, prompt, layer_idx, device):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    out = model(**inputs, output_hidden_states=True, use_cache=False)
    return out.hidden_states[layer_idx][0, -1, :].detach()


@torch.no_grad()
def extract_task_vector(model, tokenizer, neutral, demos, layer_idx, device):
    h_task = get_hidden_at_last_token(model, tokenizer, build_task_vector_prompt(neutral, demos), layer_idx, device)
    h_ctrl = get_hidden_at_last_token(model, tokenizer, build_control_vector_prompt(neutral), layer_idx, device)
    return h_task - h_ctrl


def make_patch_hook(theta, alpha, patch_pos):
    def hook(module, inputs):
        hidden = inputs[0].clone()
        vec = theta.to(device=hidden.device, dtype=hidden.dtype)
        hidden[:, patch_pos, :] += alpha * vec
        return (hidden,) + tuple(inputs[1:])
    return hook


@torch.no_grad()
def candidate_logprob(model, tokenizer, prompt, candidate, device,
                      layer_idx=None, theta=None, alpha=1.0):
    cand_text = " " + candidate
    prompt_ids = tokenizer(prompt, add_special_tokens=True).input_ids
    cand_ids = tokenizer(cand_text, add_special_tokens=False).input_ids
    input_ids = torch.tensor([prompt_ids + cand_ids], device=device)
    prompt_len = len(prompt_ids)
    patch_pos = prompt_len - 1

    handle = None
    if theta is not None and layer_idx is not None:
        handle = get_layers(model)[layer_idx].register_forward_pre_hook(
            make_patch_hook(theta, alpha, patch_pos))
    try:
        out = model(input_ids=input_ids, use_cache=False)
    finally:
        if handle is not None:
            handle.remove()

    logits = out.logits[0]
    total = 0.0
    for i, tok_id in enumerate(cand_ids):
        log_probs = F.log_softmax(logits[prompt_len + i - 1], dim=-1)
        total += float(log_probs[tok_id].item())
    return total


@torch.no_grad()
def predict(model, tokenizer, prompt, candidates, device, layer_idx=None, theta=None, alpha=1.0):
    scores = {c: candidate_logprob(model, tokenizer, prompt, c, device, layer_idx, theta, alpha)
              for c in candidates}
    return max(scores.items(), key=lambda kv: kv[1])[0]


@torch.no_grad()
def evaluate(model, tokenizer, items, candidates, device, neutral,
             demos=None, layer_idx=None, theta=None, alpha=1.0):
    correct = 0
    for it in items:
        prompt = build_prompt(it.x, neutral, demos=demos)
        pred = predict(model, tokenizer, prompt, candidates, device, layer_idx, theta, alpha)
        correct += int(pred == it.y)
    return correct / max(len(items), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--task", default="last_letter", choices=["last_letter", "antonym", "en_fr"])
    ap.add_argument("--neutral-instruct", action="store_true", default=True)
    ap.add_argument("--explicit-instruct", dest="neutral_instruct", action="store_false")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--n-eval", type=int, default=12)
    ap.add_argument("--alphas", default="0.25,0.5,1.0,2.0")
    ap.add_argument("--out", default="smoke_v2_results.json")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    neutral = args.neutral_instruct

    quant = None
    if args.load_4bit:
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                                   bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True,
        device_map="auto" if device == "cuda" else None,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        quantization_config=quant)
    model.eval()
    if device == "cpu":
        model.to(device)

    pairs, candidates = build_task(args.task, args.seed)
    demos = pairs[: args.k]
    eval_items = [EvalItem(x, y) for x, y in pairs[args.k: args.k + args.n_eval]]

    layers = get_layers(model)
    n_layers = len(layers)
    alphas = [float(x) for x in args.alphas.split(",")]

    zero_acc = evaluate(model, tokenizer, eval_items, candidates, device, neutral, demos=None)
    icl_acc = evaluate(model, tokenizer, eval_items, candidates, device, neutral, demos=demos)
    icl_gain = icl_acc - zero_acc

    results = {"model": args.model, "task": args.task, "neutral_instruct": neutral,
               "seed": args.seed, "k": args.k, "n_eval": len(eval_items),
               "n_candidates": len(candidates),
               "demos": demos, "eval": [(x.x, x.y) for x in eval_items],
               "zero_acc": zero_acc, "icl_acc": icl_acc, "icl_gain": icl_gain, "patched": []}

    print(f"[task] {args.task} neutral={neutral} n_cand={len(candidates)}")
    print(f"[zero] {zero_acc:.3f}")
    print(f"[icl ] {icl_acc:.3f}")
    print(f"[gain] {icl_gain:.3f}")

    for L in range(n_layers):
        theta = extract_task_vector(model, tokenizer, neutral, demos, L, device)
        for alpha in alphas:
            patched = evaluate(model, tokenizer, eval_items, candidates, device, neutral,
                               demos=None, layer_idx=L, theta=theta, alpha=alpha)
            abs_gain = patched - zero_acc
            rec = abs_gain / icl_gain if abs(icl_gain) >= 0.03 else None
            results["patched"].append({"layer": L, "alpha": alpha, "patched_acc": patched,
                                       "absolute_patch_gain": abs_gain, "recovery_ratio": rec})
            print(f"[L={L:02d} a={alpha}] patched={patched:.3f} abs_gain={abs_gain:+.3f} "
                  f"recovery={'NA' if rec is None else f'{rec:.3f}'}")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    best = max(results["patched"], key=lambda r: r["absolute_patch_gain"])
    print("\n=== BEST BY ABSOLUTE PATCH GAIN ===")
    print(json.dumps(best, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
