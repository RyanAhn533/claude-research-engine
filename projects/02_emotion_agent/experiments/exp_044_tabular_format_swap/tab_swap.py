"""
exp_044: WITHIN-DOMAIN format-swap on tabular data — the cleanest causal test of the
format-novelty thesis, OUTSIDE emotion.
Same UCI rows, two serializations:
  natural (TabLLM): "The plasma glucose is 148. The BMI is 33.6."   -> LOW fertility
  alien   (coded) : "plas=148, mass=33.6, age=50, ..."             -> HIGH fertility
zero / gold-ICL / random-ICL on each. Prediction (format-novelty causal):
  alien  -> sizable ICL gain, TR-dominant, high fertility
  natural-> ~0 gain (already confirmed exp_043)
=> proves format novelty (not domain, not content) drives ICL gain, in a non-emotion task.
"""
import os, sys, json, glob, argparse
from pathlib import Path
import numpy as np, torch
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "exp_035_multimodel_gating"))
import run_gating as G
OUT = Path(__file__).parent / "cache"; OUT.mkdir(exist_ok=True, parents=True)
SEEDS = [42, 123, 777]
OID, LMAP, NOUN, LABELS = 37, {"tested_positive":"yes","tested_negative":"no"}, "diabetes", ["yes","no"]
FULL = {"preg":"number of times pregnant","plas":"plasma glucose concentration","pres":"diastolic blood pressure",
        "skin":"triceps skin fold thickness","insu":"serum insulin","mass":"body mass index",
        "pedi":"diabetes pedigree function","age":"age"}

def ser_natural(row, cols): return " ".join(f"The {FULL.get(c,c)} is {row[c]}." for c in cols)
def ser_alien(row, cols):   return ", ".join(f"{c}={row[c]}" for c in cols)
FMT = {"natural": ser_natural, "alien": ser_alien}

def load(seed, ser):
    d=fetch_openml(data_id=OID,as_frame=True,parser="auto"); df=d.data.copy(); df["_y"]=d.target.astype(str).map(LMAP)
    df=df[df["_y"].notna()].dropna(); cols=list(d.data.columns); rng=np.random.default_rng(seed)
    ex_idx=[]
    for lab in LABELS:
        sub=df[df["_y"]==lab]; ex_idx+=sub.iloc[rng.choice(len(sub),size=2,replace=False)].index.tolist()
    ex_df=df.loc[ex_idx]; rem=df.drop(index=ex_idx); test_idx=[]
    for lab in LABELS:
        sub=rem[rem["_y"]==lab]; n=min(100,len(sub)); test_idx+=sub.iloc[rng.choice(len(sub),size=n,replace=False)].index.tolist()
    test=rem.loc[test_idx]
    return [(ser(r,cols),r["_y"]) for _,r in test.iterrows()], [(ser(r,cols),r["_y"]) for _,r in ex_df.iterrows()]

def prompt(x, ex):
    sys_p=f"You are a classifier. Given a description, predict {NOUN}: answer yes or no.\n"
    if ex:
        sys_p+=f"Here are {len(ex)} labeled examples:\n"
        for i,(t,l) in enumerate(ex): sys_p+=f"\nExample {i+1}:\n  {t}\n  LABEL: {l}\n"
        sys_p+="\nNow classify the target.\n"
    sys_p+="Output format:\nLABEL: <yes or no>\nREASON: <one sentence>"
    return [{"role":"system","content":sys_p},{"role":"user","content":(f"Target:\n{x}\n\nClassify." if ex else f"{x}\n\nClassify.")}]

def parse_bin(raw):
    for line in raw.strip().splitlines():
        s=line.strip()
        if s.upper().startswith("LABEL:"):
            v=s.split(":",1)[1].strip().lower()
            if "yes" in v: return "yes"
            if "no" in v: return "no"
    low=raw.lower()
    return "yes" if ("yes" in low and "no" not in low) else "no"

def rand_labels(ex,seed):
    rng=np.random.default_rng(seed+9999); return [(t, LABELS[rng.integers(0,2)]) for t,_ in ex]

@torch.inference_mode()
def run(items, ex, model, tok, use_icl, bs=32):
    y=[g for _,g in items]; msgs=[prompt(x, ex if use_icl else None) for x,_ in items]; pr=[]
    for b in range(0,len(msgs),bs):
        ps=[tok.apply_chat_template(m,tokenize=False,add_generation_prompt=True) for m in msgs[b:b+bs]]
        enc=tok(ps,return_tensors="pt",padding=True,truncation=True,max_length=2048).to(model.device)
        out=model.generate(**enc,max_new_tokens=40,do_sample=False,pad_token_id=tok.pad_token_id)
        pr.extend(parse_bin(tok.decode(g,skip_special_tokens=True)) for g in out[:,enc.input_ids.shape[1]:])
    return float(accuracy_score(y,pr))

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--model",required=True); args=ap.parse_args()
    snap=glob.glob(f"{G.HUB}/models--{G.MODELS[args.model]}/snapshots/*")[0]; print(f"[load] {args.model}",flush=True)
    bnb=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",bnb_4bit_compute_dtype=torch.float16,bnb_4bit_use_double_quant=True)
    tok=AutoTokenizer.from_pretrained(snap,trust_remote_code=True); tok.padding_side="left"
    if tok.pad_token_id is None: tok.pad_token=tok.eos_token
    model=AutoModelForCausalLM.from_pretrained(snap,quantization_config=bnb,device_map={"":0},torch_dtype=torch.float16,trust_remote_code=True).eval()
    res={}
    for fmt,ser in FMT.items():
        z=[];gold=[];rand=[];fert=0
        for s in SEEDS:
            items,ex=load(s,ser)
            z.append(run(items,ex,model,tok,False)); gold.append(run(items,ex,model,tok,True)); rand.append(run(items,rand_labels(ex,s),model,tok,True))
            if s==42:
                toks=[len(tok(x).input_ids) for x,_ in items[:100]]; wds=[max(1,len(x.split())) for x,_ in items[:100]]; fert=float(np.mean([t/w for t,w in zip(toks,wds)]))
        z,gold,rand=np.array(z),np.array(gold),np.array(rand)
        res[fmt]={"zero":float(z.mean()*100),"gold":float(gold.mean()*100),"random":float(rand.mean()*100),
                  "gain_pp":float((gold.mean()-z.mean())*100),"TR_pp":float((rand.mean()-z.mean())*100),"TL_pp":float((gold.mean()-rand.mean())*100),"fertility":fert}
        r=res[fmt]; print(f"  {args.model}/{fmt}: zero={r['zero']:.1f} gold={r['gold']:.1f} rand={r['random']:.1f} | gain={r['gain_pp']:+.1f} TR={r['TR_pp']:+.1f} TL={r['TL_pp']:+.1f} fert={r['fertility']:.2f}",flush=True)
    res["delta_gain_alien_minus_natural"]=res["alien"]["gain_pp"]-res["natural"]["gain_pp"]
    json.dump({"model":args.model,"results":res}, open(OUT/f"swap_{args.model}.json","w"), indent=2)
    print(f"[H] alien_gain - natural_gain = {res['delta_gain_alien_minus_natural']:+.1f}pp (>0 = format novelty causes gain, in NON-emotion domain)",flush=True)
    print(f"[SAVED] swap_{args.model}.json",flush=True)
