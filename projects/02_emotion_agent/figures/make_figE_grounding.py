"""figE: 3-way decomposition of ICL gain (GROUNDING / RECOG / TL) from exp_056.
Left: AU (novel) — GROUNDING dominates. Right: familiar dialog (IEMOCAP+MELD avg) — all ~0.
Message: on a novel format, label-free exemplars (GROUNDING) recover most of the gain."""
import json, glob
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXP = Path("/home/ajy/CLAUDE_RESEARCH_ENGINE/projects/02_emotion_agent/experiments/exp_056_unlabeled_control/cache")
ORDER = ["qwen3b", "qwen7b", "qwen14b", "yi6b", "falcon7b", "mistral"]
NICE = {"qwen3b": "Qwen-3B", "qwen7b": "Qwen-7B", "qwen14b": "Qwen-14B",
        "yi6b": "Yi-6B", "falcon7b": "Falcon-7B", "mistral": "Mistral-7B"}
C = {"GR": "#2563eb", "RC": "#d97706", "TL": "#059669"}  # grounding / recog / mapping

data = {}
for f in glob.glob(str(EXP / "unlabeled_*.json")):
    d = json.load(open(f)); data[d["model"]] = d["results"]

def comp(m, ds):
    r = data[m][ds]
    return r["GROUNDING_pp"], r["RECOG_pp"], r["TL_pp"]

fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), sharey=True)
x = np.arange(len(ORDER)); w = 0.26

for ax, (title, dsel) in zip(axes, [("AU intensities (novel format)", "au"),
                                     ("English dialog (familiar)", "fam")]):
    GR, RC, TL = [], [], []
    for m in ORDER:
        if dsel == "au":
            g, r, t = comp(m, "au_nofacs")
        else:  # average iemocap + meld
            gi, ri, ti = comp(m, "iemocap"); gm, rm, tm = comp(m, "meld")
            g, r, t = (gi+gm)/2, (ri+rm)/2, (ti+tm)/2
        GR.append(g); RC.append(r); TL.append(t)
    ax.bar(x - w, GR, w, label="Format grounding\n(unlabeled$-$zero)", color=C["GR"])
    ax.bar(x,     RC, w, label="Label-space\n(random$-$unlabeled)", color=C["RC"])
    ax.bar(x + w, TL, w, label="Mapping learning\n(gold$-$random)", color=C["TL"])
    ax.axhline(0, color="#444", lw=0.8)
    ax.set_title(title, fontsize=11)
    ax.set_xticks(x); ax.set_xticklabels([NICE[m] for m in ORDER], rotation=35, ha="right", fontsize=8.5)
    ax.grid(axis="y", alpha=0.25)

axes[0].set_ylabel("Contribution to ICL gain (pp)", fontsize=10)
axes[0].legend(fontsize=7.5, loc="upper right", framealpha=0.9)
fig.suptitle("ICL gain decomposes into format grounding, not label learning",
             fontsize=12, y=1.02)
fig.tight_layout()
for out in ["figE_grounding.png"]:
    fig.savefig(Path(__file__).parent / out, dpi=200, bbox_inches="tight")
print("[saved] figE_grounding.png")
# print the numbers going into the figure for the caption/text
print("\nAU decomposition (GROUNDING / RECOG / TL, pp):")
for m in ORDER:
    g, r, t = comp(m, "au_nofacs"); gain = data[m]["au_nofacs"]["gain_pp"]
    print(f"  {NICE[m]:11} GR={g:+5.1f}  RC={r:+5.1f}  TL={t:+5.1f}  gain={gain:+5.1f}  GR%={100*g/gain if gain>0.3 else float('nan'):4.0f}")
