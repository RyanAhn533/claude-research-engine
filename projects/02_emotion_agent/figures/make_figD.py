import json
from pathlib import Path
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
EXP=Path(__file__).resolve().parents[1]/"experiments"; OUT=Path(__file__).parent
M=["qwen3b","qwen7b","qwen14b","yi6b","falcon7b","mistral"]; DISP={"qwen3b":"Qwen-3B","qwen7b":"Qwen-7B","qwen14b":"Qwen-14B","yi6b":"Yi-6B","falcon7b":"Falcon-7B","mistral":"Mistral-7B"}
# AU domain: gain(AU) - max(familiar)
au={}
for m in M:
    r=json.load(open(EXP/"exp_039_clean_gating_panel"/"cache"/f"clean_{m}.json"))["results"]
    au[m]=r["au_nofacs"]["gain_pp"]-max(r["iemocap"]["gain_pp"],r["meld"]["gain_pp"])
# sentiment domain: leet - natural (4 models only)
leet={}
for m in ["qwen7b","qwen14b","falcon7b","mistral"]:
    try:
        r=json.load(open(EXP/"exp_045_sentiment_leet"/"cache"/f"sent_{m}.json"))["results"]; leet[m]=r["leet"]["gain_pp"]-r["natural"]["gain_pp"]
    except: pass
fig,ax=plt.subplots(figsize=(8,4)); x=np.arange(len(M)); w=0.38
au_v=[au[m] for m in M]
b1=ax.bar(x-w/2,au_v,w,label="Facial-AU domain  (AU − dialog)",color="#d1495b",edgecolor="k",lw=0.6)
leet_v=[leet.get(m,np.nan) for m in M]
b2=ax.bar(x+w/2,leet_v,w,label="Sentiment domain  (leetspeak − plain)",color="#3a7ca5",edgecolor="k",lw=0.6)
for b,v in zip(b1,au_v): ax.text(b.get_x()+b.get_width()/2, v+(0.2 if v>=0 else -0.6), f"{v:+.0f}", ha="center", fontsize=7)
for b,v in zip(b2,leet_v):
    if not np.isnan(v): ax.text(b.get_x()+b.get_width()/2, v+(0.2 if v>=0 else -0.6), f"{v:+.0f}", ha="center", fontsize=7)
ax.axhline(0,color="k",lw=0.9); ax.set_xticks(x); ax.set_xticklabels([DISP[m] for m in M],rotation=12)
ax.set_ylabel("format-novelty ICL effect (pp):\nnovel-format gain − familiar-format gain")
ax.set_title("Format-novelty gates ICL across TWO domains — Mistral the consistent exception")
ax.legend(frameon=False,loc="upper right")
plt.tight_layout(); plt.savefig(OUT/"figD_cross_domain.png",dpi=300,bbox_inches="tight"); plt.close()
print("[saved] figD_cross_domain.png"); print("AU effect:",{m:round(au[m],1) for m in M}); print("leet effect:",{m:round(leet[m],1) for m in leet})
