"""exp_036b: perplexity negative-control, EXPANDED to 6 models + no-FACS AU (consistency).
Original used 3 models + with-FACS AU gains (mistral au=15.17) while the paper's Table 1
uses no-FACS 6-model gains (mistral au=+1.2). This recomputes the NLL<->gain correlation
on the SAME no-FACS AU format and the SAME clean-panel gains -> internally consistent.
Forward-pass only (no generation). Writes nll_novelty_6model.json for figB regen.
"""
import sys, json, glob
from pathlib import Path
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
P = Path("/home/ajy/CLAUDE_RESEARCH_ENGINE/projects/02_emotion_agent")
sys.path.insert(0, str(P/"experiments/exp_035_multimodel_gating"))
sys.path.insert(0, str(P/"experiments/exp_038_format_intervention"))
import run_gating as G, intervene as IV
OUT = Path(__file__).parent/"cache"; OUT.mkdir(exist_ok=True, parents=True)
MODELS = ["qwen3b", "qwen7b", "qwen14b", "yi6b", "falcon7b", "mistral"]
CLEAN = P/"experiments/exp_039_clean_gating_panel/cache"
N = 200

# no-FACS AU numeric (same format as the gains); iemocap/meld unchanged
def au_strings(seed=42):   return [x for x, _ in IV.load(seed, IV.numeric_fmt)[0] if x][:N]
def text_strings(ds, seed=42): return [x for x, _ in G.DATASETS[ds][1](seed)[0] if x][:N]
STR = {"au": au_strings(), "iemocap": text_strings("iemocap"), "meld": text_strings("meld")}

# clean-panel no-FACS gains
GAIN = {}
for m in MODELS:
    d = json.load(open(CLEAN/f"clean_{m}.json"))["results"]
    GAIN[m] = {"au": d["au_nofacs"]["gain_pp"], "iemocap": d["iemocap"]["gain_pp"], "meld": d["meld"]["gain_pp"]}

@torch.inference_mode()
def mean_nll(model, tok, text):
    enc = tok(text, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    ids = enc.input_ids
    if ids.shape[1] < 2:
        return None
    return float(model(ids, labels=ids).loss)

results = {}
for m in MODELS:
    snap = glob.glob(f"{G.HUB}/models--{G.MODELS[m]}/snapshots/*")[0]
    print(f"[load] {m}", flush=True)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    tok = AutoTokenizer.from_pretrained(snap, trust_remote_code=True)
    if tok.pad_token_id is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(snap, quantization_config=bnb, device_map={"": 0},
                                                 torch_dtype=torch.float16, trust_remote_code=True).eval()
    results[m] = {}
    for ds in ["au", "iemocap", "meld"]:
        nlls = [v for v in (mean_nll(model, tok, s) for s in STR[ds]) if v is not None]
        a = np.array(nlls)
        results[m][ds] = {"nll_mean": float(a.mean()), "nll_std": float(a.std()),
                          "ppl": float(np.exp(a.mean())), "n": len(a), "gain_pp": GAIN[m][ds]}
        print(f"  {m}/{ds}: NLL={a.mean():.3f} ppl={np.exp(a.mean()):.1f} gain={GAIN[m][ds]:+.1f}pp", flush=True)
    del model; torch.cuda.empty_cache()

from scipy.stats import spearmanr, pearsonr
pn, pg = [], []
for m in results:
    for ds in ["au", "iemocap", "meld"]:
        pn.append(results[m][ds]["nll_mean"]); pg.append(results[m][ds]["gain_pp"])
rho, p = spearmanr(pn, pg); r, pr = pearsonr(pn, pg)
summary = {"per_model": results,
           "H2_pooled_spearman": {"rho": float(rho), "p": float(p), "n": len(pn)},
           "H2_pooled_pearson": {"r": float(r), "p": float(pr)}}
json.dump(summary, open(OUT/"nll_novelty_6model.json", "w"), indent=2)
print(f"\n[H2 6-model no-FACS] Spearman rho={rho:.3f} p={p:.4f} | Pearson r={r:.3f} p={pr:.4f} (n={len(pn)})", flush=True)
print(f"[SAVED] {OUT}/nll_novelty_6model.json", flush=True)
