"""Publication figures for PAPER_v1 (data already final, no GPU)."""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})

EXP = Path(__file__).resolve().parents[1] / "experiments"
OUT = Path(__file__).parent; OUT.mkdir(exist_ok=True)
MODELS = ["qwen7b", "mistral", "qwen14b"]
DISP = {"qwen7b": "Qwen2.5-7B", "mistral": "Mistral-7B", "qwen14b": "Qwen2.5-14B"}
DS = ["au", "iemocap", "meld"]
DSDISP = {"au": "AU (novel)", "iemocap": "IEMOCAP", "meld": "MELD"}

# ---- Fig A: gating bar chart (ICL gain per model x dataset) ----
gain = {}
for m in MODELS:
    d = json.load(open(EXP / "exp_035_multimodel_gating" / "cache" / f"gating_{m}.json"))
    gain[m] = {ds: d["results"][ds]["icl_gain_pp"] for ds in DS}

fig, ax = plt.subplots(figsize=(7.2, 4.0))
x = np.arange(len(MODELS)); w = 0.26
colors = {"au": "#d1495b", "iemocap": "#5b8e7d", "meld": "#8aa1b1"}
for j, ds in enumerate(DS):
    vals = [gain[m][ds] for m in MODELS]
    bars = ax.bar(x + (j-1)*w, vals, w, label=DSDISP[ds], color=colors[ds], edgecolor="black", linewidth=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, v+0.2, f"{v:+.1f}", ha="center", va="bottom", fontsize=8)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xticks(x); ax.set_xticklabels([DISP[m] for m in MODELS])
ax.set_ylabel("ICL gain (pp): ICL k=4  −  zero-shot")
ax.set_title("Few-shot helps the unfamiliar AU input, not familiar dialog text")
ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))
plt.tight_layout()
plt.savefig(OUT / "figA_gating.png", dpi=300, bbox_inches="tight")
plt.close()
print("[saved] figA_gating.png")

# ---- Fig B: perplexity NOT the cause (NLL vs gain, negative control) ----
nll = json.load(open(EXP / "exp_036_novelty_nll" / "cache" / "nll_novelty_results.json"))
pts = []
for m in MODELS:
    for ds in DS:
        c = nll["per_model"][m][ds]
        pts.append((c["nll_mean"], c["gain_pp"], m, ds))
fig, ax = plt.subplots(figsize=(6.2, 4.2))
for nllv, g, m, ds in pts:
    ax.scatter(nllv, g, s=70, color=colors[ds], edgecolor="black", linewidth=0.6, zorder=3)
    ax.annotate(DISP[m].split("-")[0][:6], (nllv, g), fontsize=7, xytext=(3,3), textcoords="offset points")
# trend
xs = np.array([p[0] for p in pts]); ys = np.array([p[1] for p in pts])
b1, b0 = np.polyfit(xs, ys, 1)
xx = np.linspace(xs.min(), xs.max(), 50)
ax.plot(xx, b1*xx+b0, "--", color="gray", linewidth=1.2,
        label=f"Spearman ρ={nll['H2_pooled_spearman']['rho']:.2f} (p={nll['H2_pooled_spearman']['p']:.3f})")
handles = [plt.Line2D([0],[0], marker="o", color="w", markerfacecolor=colors[ds], markeredgecolor="k", label=DSDISP[ds]) for ds in DS]
ax.legend(handles=handles + [plt.Line2D([0],[0], ls="--", color="gray", label=f"trend (ρ={nll['H2_pooled_spearman']['rho']:.2f})")], frameon=False, fontsize=8)
ax.set_xlabel("input perplexity (mean per-token NLL)  →  more 'surprising'")
ax.set_ylabel("ICL gain (pp)")
ax.set_title("Negative control: ICL gain is NOT explained by input surprise\n(AU is the lowest-perplexity input yet gains most)")
plt.tight_layout()
plt.savefig(OUT / "figB_perplexity_control.png", dpi=300, bbox_inches="tight")
plt.close()
print("[saved] figB_perplexity_control.png")
print("[DONE] figures in", OUT)
