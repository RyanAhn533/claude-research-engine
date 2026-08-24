# smoke_task_vector_patch_v3.py
# Step 0 smoke (v3): Hendel-faithful task-vector extraction + REPLACEMENT patching.
#
# Why v3:
#   v2 used  hidden += alpha * (h_task - h_ctrl)  (additive, scaled delta).
#   That created scale artifacts: alpha=2.0 destroyed accuracy at most layers and the
#   single positive (L7) was noise. Hendel et al. (2023) instead REPLACE the hidden
#   state at the query's last token with the RAW task vector theta = h(last demo token)
#   at layer L. No alpha. v3 implements that, and bumps n_eval to cut noise.
#
# Example:
#   python smoke_task_vector_patch_v3.py --model Qwen/Qwen2.5-7B-Instruct --load-4bit --task en_fr

import argparse
import json
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


EN_FR_PAIRS = [
    ("dog", "chien"), ("cat", "chat"), ("house", "maison"), ("water", "eau"),
    ("bread", "pain"), ("book", "livre"), ("tree", "arbre"), ("fish", "poisson"),
    ("milk", "lait"), ("sun", "soleil"), ("moon", "lune"), ("king", "roi"),
    ("queen", "reine"), ("horse", "cheval"), ("flower", "fleur"), ("school", "ecole"),
    ("road", "route"), ("night", "nuit"), ("day", "jour"), ("snow", "neige"),
    ("apple", "pomme"), ("car", "voiture"), ("bird", "oiseau"), ("hand", "main"),
    ("eye", "oeil"), ("door", "porte"), ("city", "ville"), ("friend", "ami"),
    ("love", "amour"), ("time", "temps"), ("year", "annee"), ("name", "nom"),
    ("song", "chanson"), ("rain", "pluie"), ("wind", "vent"), ("star", "etoile"),
]


@dataclass
class EvalItem:
    x: str
    y: str


def build_task(seed: int) -> Tuple[List[Tuple[str, str]], List[str]]:
    rng = random.Random(seed)
    pairs = EN_FR_PAIRS.copy()
    rng.shuffle(pairs)
    candidates = sorted(set(y for _, y in pairs))
    return pairs, candidates


def instruction() -> str:
    # Neutral: task identity comes only from demos.
    return "Follow the pattern shown by the examples.\n\n"


def demo_block(pairs: List[Tuple[str, str]]) -> str:
    return "".join(f"Input: {x}\nOutput: {y}\n\n" for x, y in pairs)


def build_prompt(query: str, demos: Optional[List[Tuple[str, str]]] = None) -> str:
    p = instruction()
    if demos:
        p += demo_block(demos)
    p += f"Input: {query}\nOutput:"
    return p


def task_vector_prompt(demos: List[Tuple[str, str]]) -> str:
    # Hendel: a dummy/query-free prompt ending right after the demos so the last token's
    # hidden state summarizes the task. We end at "Output:" of a neutral dummy query so
    # the extraction position mirrors the patch position (token before the answer).
    return instruction() + demo_block(demos) + "Input: x\nOutput:"


def get_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise RuntimeError("Could not locate transformer layers.")


@torch.no_grad()
def extract_raw_theta(model, tokenizer, demos, layer_idx, device):
    # Raw hidden state at the last token of the demo prompt, at the INPUT to layer_idx.
    prompt = task_vector_prompt(demos)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    out = model(**inputs, output_hidden_states=True, use_cache=False)
    return out.hidden_states[layer_idx][0, -1, :].detach()


def make_replace_hook(theta, patch_pos):
    # Hendel-faithful: REPLACE the hidden state at patch_pos (no scaling).
    def hook(module, inputs):
        hidden = inputs[0].clone()
        hidden[:, patch_pos, :] = theta.to(device=hidden.device, dtype=hidden.dtype)
        return (hidden,) + tuple(inputs[1:])
    return hook


@torch.no_grad()
def candidate_logprob(model, tokenizer, prompt, candidate, device,
                      layer_idx=None, theta=None):
    cand_text = " " + candidate
    prompt_ids = tokenizer(prompt, add_special_tokens=True).input_ids
    cand_ids = tokenizer(cand_text, add_special_tokens=False).input_ids
    input_ids = torch.tensor([prompt_ids + cand_ids], device=device)
    prompt_len = len(prompt_ids)
    patch_pos = prompt_len - 1  # the "Output:" colon token, mirrors extraction

    handle = None
    if theta is not None and layer_idx is not None:
        handle = get_layers(model)[layer_idx].register_forward_pre_hook(
            make_replace_hook(theta, patch_pos))
    try:
        out = model(input_ids=input_ids, use_cache=False)
    finally:
        if handle is not None:
            handle.remove()

    logits = out.logits[0]
    total = 0.0
    for i, tok_id in enumerate(cand_ids):
        lp = F.log_softmax(logits[prompt_len + i - 1], dim=-1)
        total += float(lp[tok_id].item())
    return total


@torch.no_grad()
def predict(model, tokenizer, prompt, candidates, device, layer_idx=None, theta=None):
    scores = {c: candidate_logprob(model, tokenizer, prompt, c, device, layer_idx, theta)
              for c in candidates}
    return max(scores.items(), key=lambda kv: kv[1])[0]


@torch.no_grad()
def evaluate(model, tokenizer, items, candidates, device, demos=None, layer_idx=None, theta=None):
    correct = 0
    for it in items:
        prompt = build_prompt(it.x, demos=demos)
        pred = predict(model, tokenizer, prompt, candidates, device, layer_idx, theta)
        correct += int(pred == it.y)
    return correct / max(len(items), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--task", default="en_fr")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--n-eval", type=int, default=24)
    ap.add_argument("--out", default="smoke_v3_results.json")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    quant = None
    if args.load_4bit:
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                                   bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True,
        device_map="auto" if device == "cuda" else None,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        quantization_config=quant)
    model.eval()
    if device == "cpu":
        model.to(device)

    pairs, candidates = build_task(args.seed)
    demos = pairs[: args.k]
    eval_items = [EvalItem(x, y) for x, y in pairs[args.k: args.k + args.n_eval]]
    n_layers = len(get_layers(model))

    zero_acc = evaluate(model, tok, eval_items, candidates, device, demos=None)
    icl_acc = evaluate(model, tok, eval_items, candidates, device, demos=demos)
    icl_gain = icl_acc - zero_acc

    results = {"model": args.model, "task": args.task, "method": "raw_replace",
               "seed": args.seed, "k": args.k, "n_eval": len(eval_items),
               "n_candidates": len(candidates), "zero_acc": zero_acc, "icl_acc": icl_acc,
               "icl_gain": icl_gain, "patched": []}

    print(f"[task] {args.task} method=raw_replace n_eval={len(eval_items)} n_cand={len(candidates)}")
    print(f"[zero] {zero_acc:.3f}")
    print(f"[icl ] {icl_acc:.3f}")
    print(f"[gain] {icl_gain:.3f}", flush=True)

    def save():
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    try:
        for L in range(n_layers):
            theta = extract_raw_theta(model, tok, demos, L, device)
            patched = evaluate(model, tok, eval_items, candidates, device,
                               demos=None, layer_idx=L, theta=theta)
            abs_gain = patched - zero_acc
            rec = abs_gain / icl_gain if abs(icl_gain) >= 0.03 else None
            results["patched"].append({"layer": L, "patched_acc": patched,
                                       "absolute_patch_gain": abs_gain, "recovery_ratio": rec})
            print(f"[L={L:02d}] patched={patched:.3f} abs_gain={abs_gain:+.3f} "
                  f"recovery={'NA' if rec is None else f'{rec:.3f}'}", flush=True)
            save()
            if device == "cuda":
                torch.cuda.empty_cache()
    finally:
        save()

    if results["patched"]:
        best = max(results["patched"], key=lambda r: r["absolute_patch_gain"])
        print("\n=== BEST BY ABSOLUTE PATCH GAIN ===")
        print(json.dumps(best, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
