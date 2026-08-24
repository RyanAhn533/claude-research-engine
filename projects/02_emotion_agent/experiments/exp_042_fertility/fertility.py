"""exp_042: tokenizer FERTILITY as corpus-free format-novelty proxy (lit-scout iter2).
perplexity failed (AU lowest-ppl, NEG corr). Hypothesis: AU numeric format has HIGH
fertility (fragments into many tokens) -> POSITIVE corr with ICL gain. CPU only."""
import sys, json, glob
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer
P = Path("/home/ajy/CLAUDE_RESEARCH_ENGINE/projects/02_emotion_agent")
sys.path.insert(0, str(P/"experiments/exp_035_multimodel_gating"))
sys.path.insert(0, str(P/"experiments/exp_038_format_intervention"))
import run_gating as G, intervene as IV
HUB=G.HUB
MODELS={"qwen7b":G.MODELS["qwen7b"],"mistral":G.MODELS["mistral"],"qwen14b":G.MODELS["qwen14b"]}
# exp_035 measured ICL gains (with FACS); fertility is format-level so still informative
GAIN={("qwen7b","au"):10.08,("qwen7b","iemocap"):1.00,("qwen7b","meld"):0.67,
      ("mistral","au"):15.17,("mistral","iemocap"):2.08,("mistral","meld"):0.00,
      ("qwen14b","au"):16.67,("qwen14b","iemocap"):5.50,("qwen14b","meld"):4.67}
def au_items(s): return [x for x,_ in IV.load(s,IV.numeric_fmt)[0] if x][:200]
def text_items(ds,s): return [x for x,_ in G.DATASETS[ds][1](s)[0] if x][:200]
STR={"au":au_items(42),"iemocap":text_items("iemocap",42),"meld":text_items("meld",42)}
# words per sample (semantic units ~ whitespace tokens) for fertility = tokens/word
def words(s): return max(1,len(s.split()))
rows=[]; 
for m,mid in MODELS.items():
    snap=glob.glob(f"{HUB}/models--{mid}/snapshots/*")[0]
    tok=AutoTokenizer.from_pretrained(snap,trust_remote_code=True)
    for ds in ["au","iemocap","meld"]:
        toks=[len(tok(s).input_ids) for s in STR[ds]]
        wds=[words(s) for s in STR[ds]]
        fert=np.mean([t/w for t,w in zip(toks,wds)])
        tps=np.mean(toks)
        rows.append((m,ds,float(fert),float(tps),GAIN[(m,ds)]))
        print(f"{m}/{ds}: fertility(tok/word)={fert:.2f} tok/sample={tps:.0f} gain={GAIN[(m,ds)]:+.1f}",flush=True)
from scipy.stats import spearmanr,pearsonr
F=[r[2] for r in rows]; Y=[r[4] for r in rows]
rho,p=spearmanr(F,Y); r,pr=pearsonr(F,Y)
print(f"\n[CORR] fertility vs ICL-gain: Spearman rho={rho:.3f} p={p:.4f} | Pearson r={r:.3f} p={pr:.4f}  (n={len(rows)})",flush=True)
print(f"[COMPARE] perplexity was rho=-0.75 (NEG). fertility {'POSITIVE -> SUCCEEDS' if rho>0.4 else 'weak'}",flush=True)
json.dump({"rows":[{"model":m,"ds":d,"fertility":f,"tok_per_sample":t,"gain":g} for m,d,f,t,g in rows],
           "spearman":{"rho":rho,"p":p},"pearson":{"r":r,"p":pr}},
          open(Path(__file__).parent/"cache/fertility.json","w"),indent=2)
