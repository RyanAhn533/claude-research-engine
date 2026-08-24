"""
exp_036: Operationalize "input novelty to the LLM" as the model's own per-token NLL
(perplexity) over the raw input representation. Grounds the central assumption that
AU-intensity text is NOVEL to LLMs while English dialog is FAMILIAR.

Hypothesis (pre-registered, falsifiable):
  H1: mean per-token NLL(AU-text) > NLL(IEMOCAP) ~ NLL(MELD), for every model.
  H2: across the 3 input types, NLL rank-correlates POSITIVELY with the ICL gain
      measured in exp_035 (more novel input -> larger ICL gain).
  NULL: NLL(AU) <= NLL(text)  OR  no positive NLL<->gain association.

Forward-pass only (no generation) -> fast. 200 inputs/dataset, 3 models.
"""
import os, sys, json, glob
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "exp_035_multimodel_gating"))
import run_gating as G

OUT = Path(__file__).parent / "cache"; OUT.mkdir(exist_ok=True, parents=True)
MODELS = {"qwen7b": G.MODELS["qwen7b"], "mistral": G.MODELS["mistral"], "qwen14b": G.MODELS["qwen14b"]}
# exp_035 measured ICL gains (pp) for H2 correlation
GAIN = {"au": {"qwen7b": 10.08, "mistral": 15.17, "qwen14b": 16.67},
        "iemocap": {"qwen7b": 1.00, "mistral": 2.08, "qwen14b": 5.50},
        "meld": {"qwen7b": 0.67, "mistral": 0.00, "qwen14b": 4.67}}
N = 200

def input_strings(ds, seed=42):
    items, _ = G.DATASETS[ds][1](seed)
    return [x for x, _ in items if x][:N]

@torch.inference_mode()
def mean_nll(model, tok, text):
    enc = tok(text, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    ids = enc.input_ids
    if ids.shape[1] < 2:
        return None
    out = model(ids, labels=ids)
    return float(out.loss)   # mean per-token CE over the sequence

if __name__ == "__main__":
    strings = {ds: input_strings(ds) for ds in ["au", "iemocap", "meld"]}
    results = {}
    for mname, mid in MODELS.items():
        snap = glob.glob(f"{G.HUB}/models--{mid}/snapshots/*")[0]
        print(f"[load] {mname}", flush=True)
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
        tok = AutoTokenizer.from_pretrained(snap)
        if tok.pad_token_id is None: tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(snap, quantization_config=bnb,
                                                     device_map={"": 0}, torch_dtype=torch.float16).eval()
        results[mname] = {}
        for ds in ["au", "iemocap", "meld"]:
            nlls = [v for v in (mean_nll(model, tok, s) for s in strings[ds]) if v is not None]
            a = np.array(nlls)
            results[mname][ds] = {"nll_mean": float(a.mean()), "nll_std": float(a.std()),
                                  "ppl": float(np.exp(a.mean())), "n": len(a), "gain_pp": GAIN[ds][mname]}
            print(f"  {mname}/{ds}: NLL={a.mean():.3f}  ppl={np.exp(a.mean()):.1f}  gain={GAIN[ds][mname]:+.1f}pp", flush=True)
        del model; torch.cuda.empty_cache()

    # H2: per-model Spearman-ish check via simple rank concordance over 3 points
    from scipy.stats import spearmanr, pearsonr
    pooled_nll, pooled_gain = [], []
    for m in results:
        for ds in ["au", "iemocap", "meld"]:
            pooled_nll.append(results[m][ds]["nll_mean"]); pooled_gain.append(results[m][ds]["gain_pp"])
    rho, p = spearmanr(pooled_nll, pooled_gain)
    r, pr = pearsonr(pooled_nll, pooled_gain)
    summary = {"per_model": results,
               "H2_pooled_spearman": {"rho": float(rho), "p": float(p), "n": len(pooled_nll)},
               "H2_pooled_pearson": {"r": float(r), "p": float(pr)}}
    with open(OUT / "nll_novelty_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[H2] pooled Spearman(NLL, ICL-gain) rho={rho:.3f} p={p:.4f} (n={len(pooled_nll)})", flush=True)
    print(f"[H2] pooled Pearson r={r:.3f} p={pr:.4f}", flush=True)
    print(f"[SAVED] {OUT}/nll_novelty_results.json", flush=True)
