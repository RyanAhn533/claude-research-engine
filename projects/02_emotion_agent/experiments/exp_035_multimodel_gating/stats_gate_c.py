import json, glob
from pathlib import Path
import numpy as np
from scipy import stats
OUT = Path("cache")
rows = []
for f in sorted(glob.glob("cache/gating_*.json")):
    d = json.load(open(f)); m = d["model"]
    for ds, r in d["results"].items():
        z = np.array(r["zero_shot"]["accs"]); i = np.array(r["icl_k4"]["accs"])
        delta = i - z
        md = float(delta.mean()); sd = float(delta.std(ddof=1))
        n = len(delta); se = sd/np.sqrt(n)
        tcrit = stats.t.ppf(0.975, n-1)
        ci = (md - tcrit*se, md + tcrit*se)
        dz = md/sd if sd > 0 else float('inf')   # Cohen's dz (paired)
        try: w_p = stats.wilcoxon(i, z).pvalue
        except Exception: w_p = None
        rows.append((m, ds, r["novelty"], md*100, ci[0]*100, ci[1]*100, dz, w_p))
print(f"{'model':9s} {'ds':8s} {'nov':9s} {'Δ pp':>7s} {'CI95':>16s} {'dz':>7s} {'wilcox_p':>9s}")
for m,ds,nov,md,lo,hi,dz,wp in rows:
    ci_excl = "✓" if lo>0 else " "
    print(f"{m:9s} {ds:8s} {nov:9s} {md:+7.2f} [{lo:+6.2f},{hi:+6.2f}]{ci_excl} {dz:+7.2f} {('%.3f'%wp) if wp else '   -':>9s}")
# novel vs familiar contrast
nov_g = [r[3] for r in rows if r[2]=="novel"]
fam_g = [r[3] for r in rows if r[2]=="familiar"]
print(f"\nnovel gain  mean={np.mean(nov_g):+.2f}pp (n={len(nov_g)})  [{min(nov_g):+.1f},{max(nov_g):+.1f}]")
print(f"familiar    mean={np.mean(fam_g):+.2f}pp (n={len(fam_g)})  [{min(fam_g):+.1f},{max(fam_g):+.1f}]")
u,pu = stats.mannwhitneyu(nov_g, fam_g, alternative="greater")
print(f"Mann-Whitney novel>familiar: U={u} p={pu:.4f}")
json.dump({"rows":[{"model":m,"ds":ds,"novelty":nov,"delta_pp":md,"ci95":[lo,hi],"cohens_dz":dz,"wilcoxon_p":wp} for m,ds,nov,md,lo,hi,dz,wp in rows],
           "novel_mean_pp":float(np.mean(nov_g)),"familiar_mean_pp":float(np.mean(fam_g)),
           "mannwhitney_novel_gt_familiar_p":float(pu)}, open(OUT/"gate_c_stats.json","w"), indent=2)
print(f"\n[SAVED] cache/gate_c_stats.json")
