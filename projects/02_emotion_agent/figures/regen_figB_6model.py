"""Regenerate figB (perplexity negative control) on the 6-model no-FACS data,
consistent with Table 1 gains. Reads exp_036 nll_novelty_6model.json. No GPU."""
import json
from pathlib import Path
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
EXP = Path(__file__).resolve().parents[1] / "experiments"; OUT = Path(__file__).parent
MODELS = ["qwen3b", "qwen7b", "qwen14b", "yi6b", "falcon7b", "mistral"]
DISP = {"qwen3b": "Qwen-3B", "qwen7b": "Qwen-7B", "qwen14b": "Qwen-14B",
        "yi6b": "Yi-6B", "falcon7b": "Falcon-7B", "mistral": "Mistral"}
DS = ["au", "iemocap", "meld"]; DSDISP = {"au": "AU (novel)", "iemocap": "IEMOCAP", "meld": "MELD"}
colors = {"au": "#d1495b", "iemocap": "#5b8e7d", "meld": "#8aa1b1"}
nll = json.load(open(EXP / "exp_036_novelty_nll" / "cache" / "nll_novelty_6model.json"))
pts = [(nll["per_model"][m][ds]["nll_mean"], nll["per_model"][m][ds]["gain_pp"], m, ds)
       for m in MODELS for ds in DS]
fig, ax = plt.subplots(figsize=(6.2, 4.2))
for nllv, g, m, ds in pts:
    ax.scatter(nllv, g, s=70, color=colors[ds], edgecolor="black", linewidth=0.6, zorder=3)
    ax.annotate(DISP[m].split("-")[0][:6], (nllv, g), fontsize=7, xytext=(3, 3), textcoords="offset points")
xs = np.array([p[0] for p in pts]); ys = np.array([p[1] for p in pts])
b1, b0 = np.polyfit(xs, ys, 1); xx = np.linspace(xs.min(), xs.max(), 50)
rho = nll["H2_pooled_spearman"]["rho"]; pv = nll["H2_pooled_spearman"]["p"]
ax.plot(xx, b1*xx + b0, "--", color="gray", linewidth=1.2,
        label=f"Spearman ρ={rho:.2f} (p={pv:.4f}, n=18)")
handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=colors[ds], markeredgecolor="k",
           label=DSDISP[ds]) for ds in DS]
ax.legend(handles=handles + [plt.Line2D([0], [0], ls="--", color="gray", label=f"trend (ρ={rho:.2f})")],
          frameon=False, fontsize=8)
ax.set_xlabel("input perplexity (mean per-token NLL)  →  more 'surprising'")
ax.set_ylabel("ICL gain (pp)")
ax.set_title("Negative control: ICL gain is NOT explained by input surprise\n(AU is the lowest-perplexity input yet gains most)")
plt.tight_layout(); plt.savefig(OUT / "figB_perplexity_control.png", dpi=300, bbox_inches="tight"); plt.close()
print(f"[saved] figB_perplexity_control.png  (6 models, ρ={rho:.3f}, p={pv:.4f}, n=18)")
