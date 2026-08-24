"""
exp_035 batched: same gating experiment as run_gating.py but batched generation.
Keeps 4-bit nf4 (comparable to published Qwen numbers) — batching changes throughput,
NOT outputs (greedy decode, deterministic). ~10x faster than 1-sample-at-a-time.

Usage:
  python run_gating_batched.py --model mistral --batch_size 32
"""
import os, sys, json, time, glob, argparse
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# reuse loaders/prompts/labels from run_gating (guarded by __main__, safe to import)
sys.path.insert(0, str(Path(__file__).parent))
import run_gating as G

HUB = G.HUB

def gen_batch(prompts, model, tok, max_new_tokens=48):
    ps = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in prompts]
    enc = tok(ps, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(model.device)
    with torch.inference_mode():
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.pad_token_id)
    gen = out[:, enc.input_ids.shape[1]:]
    return [tok.decode(g, skip_special_tokens=True) for g in gen]

def run_cfg(items, exemplars, prompt_fn, model, tok, use_icl, bs):
    y_true = [g for _, g in items]
    msgs = [prompt_fn(x if x else "neutral.", exemplars if use_icl else None) for x, _ in items]
    preds = []
    t0 = time.time()
    for b in range(0, len(msgs), bs):
        raws = gen_batch(msgs[b:b+bs], model, tok)
        preds.extend(G.parse_output(r) for r in raws)
        if (b // bs) % 4 == 0:
            done = min(b+bs, len(msgs))
            print(f"      [{done}/{len(msgs)}] {time.time()-t0:.0f}s", flush=True)
    acc = float(accuracy_score(y_true, preds))
    f1 = float(f1_score(y_true, preds, labels=G.LABELS, average="macro", zero_division=0))
    return acc, f1

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(G.MODELS))
    ap.add_argument("--datasets", default="au,iemocap,meld")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--load", choices=["4bit", "fp16"], default="4bit",
                    help="fp16 = single-device load (fixes Phi3 RoPE device bug with bnb)")
    args = ap.parse_args()

    snap = glob.glob(f"{HUB}/models--{G.MODELS[args.model]}/snapshots/*")[0]
    print(f"[load] {args.model} bs={args.batch_size} mode={args.load} <- {snap}", flush=True)
    tok = AutoTokenizer.from_pretrained(snap)
    tok.padding_side = "left"               # decoder-only → left pad
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    if args.load == "fp16":
        model = AutoModelForCausalLM.from_pretrained(snap, torch_dtype=torch.float16).to("cuda")
    else:
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
        model = AutoModelForCausalLM.from_pretrained(snap, quantization_config=bnb,
                                                     device_map={"": 0}, torch_dtype=torch.float16)
    model.eval()
    print(f"[load] VRAM {torch.cuda.memory_allocated()/1024**2:.0f} MB", flush=True)

    results = {}
    for ds in args.datasets.split(","):
        novelty, loader, prompt_fn = G.DATASETS[ds]
        per = {"zero_shot": {"acc": [], "f1": []}, "icl_k4": {"acc": [], "f1": []}}
        for seed in G.SEEDS:
            items, exemplars = loader(seed)
            print(f"\n#### {args.model} / {ds} ({novelty}) / seed {seed}  N={len(items)} {dict(Counter(g for _,g in items))}", flush=True)
            for cfg, use_icl in [("zero_shot", False), ("icl_k4", True)]:
                acc, f1 = run_cfg(items, exemplars, prompt_fn, model, tok, use_icl, args.batch_size)
                per[cfg]["acc"].append(acc); per[cfg]["f1"].append(f1)
                print(f"    {cfg:10s} acc={acc*100:.2f}% f1={f1:.3f}", flush=True)
        agg = {}
        for cfg, d in per.items():
            a = np.array(d["acc"])
            agg[cfg] = {"acc_mean": float(a.mean()), "acc_std": float(a.std(ddof=1)), "accs": d["acc"]}
        agg["icl_gain_pp"] = (agg["icl_k4"]["acc_mean"] - agg["zero_shot"]["acc_mean"]) * 100
        results[ds] = {"novelty": novelty, **agg}
        print(f"  >>> {args.model}/{ds}: zero={agg['zero_shot']['acc_mean']*100:.2f}% "
              f"icl={agg['icl_k4']['acc_mean']*100:.2f}%  GAIN={agg['icl_gain_pp']:+.2f}pp", flush=True)

    outpath = G.OUT / f"gating_{args.model}.json"
    with open(outpath, "w") as f:
        json.dump({"model": args.model, "model_id": G.MODELS[args.model], "seeds": G.SEEDS,
                   "batched": True, "batch_size": args.batch_size, "results": results}, f, indent=2, ensure_ascii=False)
    print(f"\n[SAVED] {outpath}", flush=True)
    print("[SUMMARY] " + " | ".join(f"{ds}:{r['icl_gain_pp']:+.1f}pp({r['novelty']})" for ds, r in results.items()), flush=True)
