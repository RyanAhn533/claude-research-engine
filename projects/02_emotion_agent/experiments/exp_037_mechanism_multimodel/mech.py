"""
exp_037: Does ICL COMPACT hidden representations exactly where it helps? Multi-model.

For each (model, dataset, config) extract last-layer last-token hidden state over
80 samples (20/class, seed 42), compute mean within-class L2 distance to class centroid.
Compaction = 1 - within_icl/within_zero  (scale-free ratio -> comparable across models).

Pre-registered hypothesis:
  H1: compaction(AU) > 0 (ICL tightens) AND compaction(IEMOCAP) <= compaction(AU).
  H2: across (model x dataset) cells, compaction POSITIVELY correlates with exp_035 ICL gain.
  NULL: compaction unrelated to / opposite of gain.
This is the POSITIVE predictor replacing the refuted perplexity story (exp_036: r=-0.83).
"""
import os, sys, json, glob
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "exp_035_multimodel_gating"))
import run_gating as G

OUT = Path(__file__).parent / "cache"; OUT.mkdir(exist_ok=True, parents=True)
MODELS = {"qwen7b": G.MODELS["qwen7b"], "mistral": G.MODELS["mistral"], "qwen14b": G.MODELS["qwen14b"]}
GAIN = {("qwen7b","au"):10.08,("qwen7b","iemocap"):1.00,
        ("mistral","au"):15.17,("mistral","iemocap"):2.08,
        ("qwen14b","au"):16.67,("qwen14b","iemocap"):5.50}
DATASETS = ["au", "iemocap"]
PER_CLASS = 20

def subsample(items):
    by = defaultdict(list)
    for x, g in items:
        if x and len(by[g]) < PER_CLASS:
            by[g].append((x, g))
    out = []
    for g in G.LABELS:
        out.extend(by[g])
    return out

@torch.inference_mode()
def within_class(model, tok, items, exemplars, prompt_fn, use_icl):
    by = defaultdict(list)
    for x, g in items:
        msgs = prompt_fn(x, exemplars if use_icl else None)
        ps = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = tok(ps, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
        h = model(**enc, output_hidden_states=True).hidden_states[-1][0, -1].float().cpu().numpy()
        by[g].append(h)
    dists = []
    for g, vs in by.items():
        if len(vs) < 2: continue
        c = np.mean(vs, axis=0)
        dists.extend(np.linalg.norm(v - c) for v in vs)
    return float(np.mean(dists))

if __name__ == "__main__":
    cells = []
    for mname, mid in MODELS.items():
        snap = glob.glob(f"{G.HUB}/models--{mid}/snapshots/*")[0]
        print(f"[load] {mname}", flush=True)
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
        tok = AutoTokenizer.from_pretrained(snap)
        if tok.pad_token_id is None: tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(snap, quantization_config=bnb,
                                                     device_map={"": 0}, torch_dtype=torch.float16).eval()
        for ds in DATASETS:
            novelty, loader, prompt_fn = G.DATASETS[ds]
            items, exemplars = loader(42)
            items = subsample(items)
            w_zero = within_class(model, tok, items, exemplars, prompt_fn, False)
            w_icl  = within_class(model, tok, items, exemplars, prompt_fn, True)
            compaction = 1.0 - (w_icl / w_zero)
            gain = GAIN[(mname, ds)]
            cells.append({"model": mname, "ds": ds, "novelty": novelty,
                          "within_zero": w_zero, "within_icl": w_icl,
                          "compaction": compaction, "gain_pp": gain})
            print(f"  {mname}/{ds}: within zero={w_zero:.2f} icl={w_icl:.2f} "
                  f"compaction={compaction*100:+.1f}%  gain={gain:+.1f}pp", flush=True)
        del model; torch.cuda.empty_cache()

    from scipy.stats import spearmanr, pearsonr
    comp = [c["compaction"] for c in cells]; gns = [c["gain_pp"] for c in cells]
    rho, p = spearmanr(comp, gns); r, pr = pearsonr(comp, gns)
    summary = {"cells": cells,
               "H2_spearman": {"rho": float(rho), "p": float(p), "n": len(cells)},
               "H2_pearson": {"r": float(r), "p": float(pr)}}
    with open(OUT / "mechanism_multimodel.json", "w") as f:
        json.dump(summary, f, indent=2)
    au = [c["compaction"] for c in cells if c["ds"]=="au"]
    ie = [c["compaction"] for c in cells if c["ds"]=="iemocap"]
    print(f"\n[H1] AU compaction mean={np.mean(au)*100:+.1f}%  IEMOCAP mean={np.mean(ie)*100:+.1f}%", flush=True)
    print(f"[H2] Spearman(compaction, gain) rho={rho:.3f} p={p:.4f}  Pearson r={r:.3f} p={pr:.4f}", flush=True)
    print(f"[SAVED] {OUT}/mechanism_multimodel.json", flush=True)
