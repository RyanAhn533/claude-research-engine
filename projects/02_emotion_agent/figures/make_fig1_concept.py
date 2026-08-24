import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path
OUT=Path(__file__).parent
fig,(ax,ax2)=plt.subplots(1,2,figsize=(12,4.2),gridspec_kw={"width_ratios":[1.75,1]})
# ---- Left: mechanism flow ----
ax.set_xlim(0,10); ax.set_ylim(0,10); ax.axis("off")
def box(a,x,y,w,h,t,fc,fs=9,tc="black"):
    a.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.08",fc=fc,ec="black",lw=1.1))
    a.text(x+w/2,y+h/2,t,ha="center",va="center",fontsize=fs,color=tc,wrap=True)
def arr(a,x1,y1,x2,y2,t="",c="black"):
    a.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="-|>",mutation_scale=14,lw=1.4,color=c))
    if t: a.text((x1+x2)/2,(y1+y2)/2+0.25,t,ha="center",fontsize=8,color=c)
ax.text(5,9.5,"Same latent task, two input formats",ha="center",fontsize=11,weight="bold")
# familiar branch
box(ax,0.2,7.0,2.4,1.3,'FAMILIAR format\n"The glucose is 148."',"#cfe8df")
box(ax,4.0,7.0,2.0,1.3,"LLM\nzero-shot HIGH","#eaeaea")
box(ax,7.0,7.0,2.6,1.3,"+few-shot:\nsmall gain (~0)","#cfe8df")
arr(ax,2.6,7.65,4.0,7.65); arr(ax,6.0,7.65,7.0,7.65)
# novel branch
box(ax,0.2,1.5,2.4,1.3,'NOVEL/alien format\n"plas=148, mass=33.6"',"#f4cccc")
box(ax,4.0,1.5,2.0,1.3,"LLM\nzero-shot LOW\n(format blocks)","#eaeaea")
box(ax,7.0,1.5,2.6,1.3,"+few-shot:\nLARGE gain\n(task recognition)","#f4cccc")
arr(ax,2.6,2.15,4.0,2.15); arr(ax,6.0,2.15,7.0,2.15,"demos restore\nthe task frame","#b03030")
ax.text(5,4.7,"high tokenizer fertility (r=0.88)  →  novel format  →  ICL recovers latent ability",
        ha="center",fontsize=9,style="italic",color="#555")
# ---- Right: 2x2 ----
ax2.set_xlim(0,10); ax2.set_ylim(0,10); ax2.axis("off")
ax2.text(5,9.4,"When does ICL help? (2×2)",ha="center",fontsize=11,weight="bold")
cells=[(1.2,5.0,"#cfe8df","small\ngain"),(5.6,5.0,"#f4cccc","LARGE gain\nAU, leet-sentiment"),
       (1.2,1.2,"#eeeeee","—"),(5.6,1.2,"#fde9d9","~0 / hurts\nalien diabetes")]
for x,y,c,t in cells:
    box(ax2,x,y,3.2,3.0,t,c,fs=9)
ax2.text(2.8,8.4,"familiar\nformat",ha="center",fontsize=9,weight="bold")
ax2.text(7.2,8.4,"novel\nformat",ha="center",fontsize=9,weight="bold")
ax2.text(0.5,6.5,"task\nKNOWN",ha="center",fontsize=9,weight="bold",rotation=90)
ax2.text(0.5,2.7,"task\nUNKNOWN",ha="center",fontsize=9,weight="bold",rotation=90)
plt.tight_layout(); plt.savefig(OUT/"fig1_concept.png",dpi=300,bbox_inches="tight"); plt.close()
print("[saved] fig1_concept.png")
