"""
exp_043: 2nd MODALITY (lit-scout iter3 graft, TabLLM AISTATS2023).
Serialize tabular rows to text ('The Glucose is 148. The BMI is 33.6.') and test whether
our gating + TR/TL mechanism + fertility generalize to a NON-emotion, NON-AU novel format.

Binary UCI tasks (diabetes, adult). zero-shot / gold-ICL k=4 / random-label-ICL k=4.
  gain = gold-zero ; TR = random-zero ; TL = gold-random
Prediction (if phenomenon is general, not emotion-specific):
  serialized-numeric-text (novel format, high fertility) -> sizable ICL gain, TR-dominant.
This kills the single-domain objection.
"""
import os, sys, json, glob, argparse
from pathlib import Path
import numpy as np
import torch
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "exp_035_multimodel_gating"))
import run_gating as G
OUT = Path(__file__).parent / "cache"; OUT.mkdir(exist_ok=True, parents=True)
SEEDS = [42, 123, 777]

DSETS = {  # openml id, label map -> clean words, task noun
  "diabetes": (37, {"tested_positive":"yes","tested_negative":"no"}, "diabetes", ["yes","no"]),
  "adult":    (1590, {">50K":"high","<=50K":"low"}, "income over $50K", ["high","low"]),
}

def serialize(row, cols):
    return " ".join(f"The {c.replace('-',' ')} is {row[c]}." for c in cols)

def load(name, seed):
    oid, lmap, noun, labels = DSETS[name]
    d = fetch_openml(data_id=oid, as_frame=True, parser="auto")
    df = d.data.copy(); df["_y"] = d.target.astype(str).map(lmap)
    df = df[df["_y"].notna()].dropna()
    cols = [c for c in d.data.columns]
    rng = np.random.default_rng(seed)
    # exemplars: 2 per class
    ex_idx = []
    for lab in labels:
        sub = df[df["_y"]==lab]
        ex_idx += sub.iloc[rng.choice(len(sub),size=2,replace=False)].index.tolist()
    ex_df = df.loc[ex_idx]; rem = df.drop(index=ex_idx)
    # test: 100 per class
    test_idx=[]
    for lab in labels:
        sub = rem[rem["_y"]==lab]; n=min(100,len(sub))
        test_idx += sub.iloc[rng.choice(len(sub),size=n,replace=False)].index.tolist()
    test = rem.loc[test_idx]
    ex = [(serialize(r,cols), r["_y"]) for _,r in ex_df.iterrows()]
    items = [(serialize(r,cols), r["_y"]) for _,r in test.iterrows()]
    return items, ex, noun, labels

def prompt(x, ex, noun, labels):
    lab_str = " or ".join(labels)
    sys_p = f"You are a classifier. Given a description, predict {noun}: answer {lab_str}.\n"
    if ex:
        sys_p += f"Here are {len(ex)} labeled examples:\n"
        for i,(t,l) in enumerate(ex): sys_p += f"\nExample {i+1}:\n  {t}\n  LABEL: {l}\n"
        sys_p += "\nNow classify the target.\n"
    sys_p += "Output format:\nLABEL: <"+lab_str+">\nREASON: <one sentence>"
    u = f"Target:\n{x}\n\nClassify." if ex else f"{x}\n\nClassify."
    return [{"role":"system","content":sys_p},{"role":"user","content":u}]

def parse_bin(raw, labels):
    for line in raw.strip().splitlines():
        s=line.strip()
        if s.upper().startswith("LABEL:"):
            v=s.split(":",1)[1].strip().lower()
            for l in labels:
                if l in v: return l
    low=raw.lower()
    for l in labels:
        if l in low: return l
    return labels[-1]

def rand_labels(ex, labels, seed):
    rng=np.random.default_rng(seed+9999)
    return [(t, labels[rng.integers(0,len(labels))]) for t,_ in ex]

@torch.inference_mode()
def run(items, ex, noun, labels, model, tok, use_icl, bs=32):
    y=[g for _,g in items]; msgs=[prompt(x, ex if use_icl else None, noun, labels) for x,_ in items]; pr=[]
    for b in range(0,len(msgs),bs):
        ps=[tok.apply_chat_template(m,tokenize=False,add_generation_prompt=True) for m in msgs[b:b+bs]]
        enc=tok(ps,return_tensors="pt",padding=True,truncation=True,max_length=2048).to(model.device)
        out=model.generate(**enc,max_new_tokens=40,do_sample=False,pad_token_id=tok.pad_token_id)
        pr.extend(parse_bin(tok.decode(g,skip_special_tokens=True),labels) for g in out[:,enc.input_ids.shape[1]:])
    return float(accuracy_score(y,pr))

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--model",required=True); ap.add_argument("--datasets",default="diabetes,adult"); args=ap.parse_args()
    snap=glob.glob(f"{G.HUB}/models--{G.MODELS[args.model]}/snapshots/*")[0]
    print(f"[load] {args.model}",flush=True)
    bnb=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",bnb_4bit_compute_dtype=torch.float16,bnb_4bit_use_double_quant=True)
    tok=AutoTokenizer.from_pretrained(snap,trust_remote_code=True); tok.padding_side="left"
    if tok.pad_token_id is None: tok.pad_token=tok.eos_token
    model=AutoModelForCausalLM.from_pretrained(snap,quantization_config=bnb,device_map={"":0},torch_dtype=torch.float16,trust_remote_code=True).eval()
    res={}
    for name in args.datasets.split(","):
        z=[];gold=[];rand=[];fert=[]
        for s in SEEDS:
            items,ex,noun,labels=load(name,s)
            z.append(run(items,ex,noun,labels,model,tok,False))
            gold.append(run(items,ex,noun,labels,model,tok,True))
            rand.append(run(items,rand_labels(ex,labels,s),noun,labels,model,tok,True))
            if s==42:
                toks=[len(tok(x).input_ids) for x,_ in items[:100]]; wds=[max(1,len(x.split())) for x,_ in items[:100]]
                fert=float(np.mean([t/w for t,w in zip(toks,wds)]))
        z,gold,rand=np.array(z),np.array(gold),np.array(rand)
        gain=(gold.mean()-z.mean())*100; TR=(rand.mean()-z.mean())*100; TL=(gold.mean()-rand.mean())*100
        res[name]={"zero":float(z.mean()*100),"gold":float(gold.mean()*100),"random":float(rand.mean()*100),
                   "gain_pp":float(gain),"TR_pp":float(TR),"TL_pp":float(TL),"fertility":fert}
        print(f"  {args.model}/{name}: zero={z.mean()*100:.1f} gold={gold.mean()*100:.1f} rand={rand.mean()*100:.1f} | gain={gain:+.1f} TR={TR:+.1f} TL={TL:+.1f} fert={fert:.2f}",flush=True)
    json.dump({"model":args.model,"results":res}, open(OUT/f"tab_{args.model}.json","w"), indent=2)
    print(f"[SAVED] tab_{args.model}.json",flush=True)
