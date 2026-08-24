import json
from pathlib import Path
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
EXP=Path(__file__).resolve().parents[1]/"experiments"; OUT=Path(__file__).parent
M=["qwen3b","qwen7b","qwen14b","yi6b","falcon7b","mistral"]
DISP={"qwen3b":"Qwen-3B","qwen7b":"Qwen-7B","qwen14b":"Qwen-14B","yi6b":"Yi-6B","falcon7b":"Falcon-7B","mistral":"Mistral-7B"}
# Fig A: 6-model clean gating
gain={}
for m in M:
    d=json.load(open(EXP/"exp_039_clean_gating_panel"/"cache"/f"clean_{m}.json"))
    gain[m]={k:d["results"][k]["gain_pp"] for k in ["au_nofacs","iemocap","meld"]}
fig,ax=plt.subplots(figsize=(8,4)); x=np.arange(len(M)); w=0.26
cols={"au_nofacs":"#d1495b","iemocap":"#5b8e7d","meld":"#8aa1b1"}
lab={"au_nofacs":"AU (novel)","iemocap":"IEMOCAP","meld":"MELD"}
for j,k in enumerate(["au_nofacs","iemocap","meld"]):
    v=[gain[m][k] for m in M]; bars=ax.bar(x+(j-1)*w,v,w,label=lab[k],color=cols[k],edgecolor="black",lw=0.5)
    for b,val in zip(bars,v): ax.text(b.get_x()+b.get_width()/2,val+0.15,f"{val:+.0f}",ha="center",fontsize=7)
ax.axhline(0,color="k",lw=0.8); ax.set_xticks(x); ax.set_xticklabels([DISP[m] for m in M],rotation=12)
ax.set_ylabel("ICL gain (pp)"); ax.set_title("Format-novelty gates ICL: 5/6 models, 3 families (Mistral = exception)")
ax.legend(frameon=False,ncol=3,loc="upper right")
plt.tight_layout(); plt.savefig(OUT/"figA_gating_6model.png",dpi=300,bbox_inches="tight"); plt.close()
# Fig C: TR/TL mechanism on AU (gating models)
GM=["qwen7b","qwen14b","yi6b","falcon7b"]; TR=[];TL=[]
for m in GM:
    d=json.load(open(EXP/"exp_040_tr_tl"/"cache"/f"tr_tl_{m}.json"))["results"]["au_nofacs"]
    TR.append(d["TR_pp"]); TL.append(d["TL_pp"])
fig,ax=plt.subplots(figsize=(6.5,4)); x=np.arange(len(GM))
ax.bar(x,TR,0.5,label="Task Recognition (random-label OK)",color="#e08e45",edgecolor="k",lw=0.6)
ax.bar(x,TL,0.5,bottom=TR,label="Task Learning (needs correct labels)",color="#3a5a78",edgecolor="k",lw=0.6)
for i,(tr,tl) in enumerate(zip(TR,TL)):
    ax.text(i,tr/2,f"{tr:+.0f}",ha="center",va="center",color="white",fontsize=8,weight="bold")
    ax.text(i,tr+tl/2,f"{tl:+.0f}",ha="center",va="center",color="white",fontsize=8)
ax.set_xticks(x); ax.set_xticklabels([DISP[m] for m in GM]); ax.set_ylabel("AU ICL gain decomposition (pp)")
ax.set_title("Mechanism: AU gain is ~75% Task RECOGNITION, not Task Learning")
ax.legend(frameon=False,fontsize=8)
plt.tight_layout(); plt.savefig(OUT/"figC_mechanism_TRTL.png",dpi=300,bbox_inches="tight"); plt.close()
print("[saved] figA_gating_6model.png + figC_mechanism_TRTL.png")
