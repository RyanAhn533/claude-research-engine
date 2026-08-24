"""exp_042b: fertility vs ICL-gain, EXPANDED to all 6 clean-panel models (n=18).
Motivation: original n=9 (3 models) gave Pearson r=0.88 but Spearman rho=0.52 (NS).
Reviewer risk: Pearson inflated by AU-vs-familiar 2-cluster separation, not a graded
predictor. Fix: use the 6-model clean panel (5-seed, no-FACS AU) gains -> n=18 with
4 distinct tokenizer families (qwen/mistral/yi/falcon). Same numeric AU format as gains.
CPU only (tokenizers + loaders). Adds bootstrap CI + within-familiar rank check.
"""
import sys, json, glob
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer
P = Path("/home/ajy/CLAUDE_RESEARCH_ENGINE/projects/02_emotion_agent")
sys.path.insert(0, str(P/"experiments/exp_035_multimodel_gating"))
sys.path.insert(0, str(P/"experiments/exp_038_format_intervention"))
import run_gating as G, intervene as IV
HUB = G.HUB
MODELS = ["qwen3b", "qwen7b", "qwen14b", "mistral", "yi6b", "falcon7b"]
CLEAN = P/"experiments/exp_039_clean_gating_panel/cache"

# ---- gains from the CLEAN 5-seed panel (no-FACS AU) ----
DS_KEY = {"au": "au_nofacs", "iemocap": "iemocap", "meld": "meld"}
gain = {}          # (model, ds) -> mean gain_pp
gain_seeds = {}    # (model, ds) -> per-seed gain list (icl - zero)
for m in MODELS:
    d = json.load(open(CLEAN/f"clean_{m}.json"))["results"]
    for ds, key in DS_KEY.items():
        cell = d[key]
        gain[(m, ds)] = cell["gain_pp"]
        gain_seeds[(m, ds)] = [(i - z) * 100 for z, i in zip(cell["zero"], cell["icl"])]

# ---- fertility (tokens per whitespace-word), SAME items/format as fertility.py ----
def au_items(s):   return [x for x, _ in IV.load(s, IV.numeric_fmt)[0] if x][:200]
def text_items(ds, s): return [x for x, _ in G.DATASETS[ds][1](s)[0] if x][:200]
STR = {"au": au_items(42), "iemocap": text_items("iemocap", 42), "meld": text_items("meld", 42)}
def words(s): return max(1, len(s.split()))

rows = []
for m in MODELS:
    snap = glob.glob(f"{HUB}/models--{G.MODELS[m]}/snapshots/*")[0]
    tok = AutoTokenizer.from_pretrained(snap, trust_remote_code=True)
    for ds in ["au", "iemocap", "meld"]:
        toks = [len(tok(s).input_ids) for s in STR[ds]]
        wds = [words(s) for s in STR[ds]]
        fert = float(np.mean([t/w for t, w in zip(toks, wds)]))
        rows.append({"model": m, "ds": ds, "fertility": fert,
                     "gain": gain[(m, ds)], "novelty": "novel" if ds == "au" else "familiar"})
        print(f"{m:9s}/{ds:8s} fertility={fert:.3f}  gain={gain[(m,ds)]:+.2f}", flush=True)

from scipy.stats import spearmanr, pearsonr
F = np.array([r["fertility"] for r in rows]); Y = np.array([r["gain"] for r in rows])
r_p, p_p = pearsonr(F, Y); rho, p_s = spearmanr(F, Y)
print(f"\n[n={len(rows)}] Pearson r={r_p:.3f} p={p_p:.5f} | Spearman rho={rho:.3f} p={p_s:.5f}")

# bootstrap 95% CI on Pearson r (resample the 18 cells) -- deterministic seed
rng = np.random.default_rng(42)
boot = []
for _ in range(10000):
    idx = rng.integers(0, len(F), len(F))
    if np.std(F[idx]) == 0 or np.std(Y[idx]) == 0:
        continue
    boot.append(pearsonr(F[idx], Y[idx])[0])
lo, hi = np.percentile(boot, [2.5, 97.5])
print(f"[bootstrap] Pearson r 95% CI = [{lo:.3f}, {hi:.3f}]  (10k resamples)")

# within-familiar rank check: does fertility track gain among the 12 familiar cells?
fam = [r for r in rows if r["novelty"] == "familiar"]
Ff = [r["fertility"] for r in fam]; Yf = [r["gain"] for r in fam]
rho_f, p_f = spearmanr(Ff, Yf)
print(f"[within-familiar n={len(fam)}] Spearman rho={rho_f:.3f} p={p_f:.4f}  (graded, not cluster)")

out = {"rows": rows, "n": len(rows),
       "pearson": {"r": r_p, "p": p_p, "ci95": [lo, hi]},
       "spearman": {"rho": rho, "p": p_s},
       "within_familiar_spearman": {"rho": rho_f, "p": p_f, "n": len(fam)}}
json.dump(out, open(P/"experiments/exp_042_fertility/cache/fertility_6model.json", "w"), indent=2)
print("\nwrote cache/fertility_6model.json")
