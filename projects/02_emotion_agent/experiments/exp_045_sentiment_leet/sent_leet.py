"""
exp_045: 2nd-DOMAIN test of format-novelty gating, OUTSIDE emotion (lit-scout iter4 graft).
Sentiment (SST-2) = a task LLMs nail zero-shot (>90%) = guaranteed latent-known.
Two formats of the SAME sentence:
  natural : plain English        -> low fertility, familiar
  leet    : deterministic leetspeak ('th15 m0v13 15 4m4z1ng') -> high fertility, alien
zero / gold-ICL / random-ICL. Prediction (format-novelty gates task-recognition):
  natural -> ~0 ICL gain (sentiment already accessible cold)
  leet    -> sizable ICL gain, TR-dominant (demos restore recognition of the known task)
delta = leet_gain - natural_gain > 0  => format novelty causes ICL gain, in a NON-emotion task.
"""
import os, sys, json, glob, argparse
from pathlib import Path
import numpy as np, torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "exp_035_multimodel_gating"))
import run_gating as G
os.environ.setdefault("HF_HOME","/mnt/hdd/ajy/caches/huggingface")
OUT = Path(__file__).parent / "cache"; OUT.mkdir(exist_ok=True, parents=True)
SEEDS=[42,123,777]; LABELS=["positive","negative"]
LEET=str.maketrans({"a":"4","e":"3","i":"1","o":"0","s":"5","t":"7","b":"8","g":"9","l":"1"})
def natural(s): return s.strip()
def leet(s): return s.strip().lower().translate(LEET)
FMT={"natural":natural,"leet":leet}

_DS=None
def get_ds():
    global _DS
    if _DS is None:
        d=load_dataset("stanfordnlp/sst2",split="train")
        _DS=[(r["sentence"], "positive" if r["label"]==1 else "negative") for r in d if len(r["sentence"].split())>=4]
    return _DS

def load(seed, fmt):
    data=get_ds(); rng=np.random.default_rng(seed)
    pos=[x for x in data if x[1]=="positive"]; neg=[x for x in data if x[1]=="negative"]
    def pick(pool,n): return [pool[i] for i in rng.choice(len(pool),size=n,replace=False)]
    ex=[(fmt(s),l) for s,l in pick(pos,2)+pick(neg,2)]
    test=[(fmt(s),l) for s,l in pick(pos,100)+pick(neg,100)]
    return test,ex

def prompt(x, ex):
    sys_p="You are a sentiment classifier. Given a movie-review snippet, answer positive or negative.\n"
    if ex:
        sys_p+=f"Here are {len(ex)} labeled examples:\n"
        for i,(t,l) in enumerate(ex): sys_p+=f"\nExample {i+1}:\n  \"{t}\"\n  LABEL: {l}\n"
        sys_p+="\nNow classify the target.\n"
    sys_p+="Output format:\nLABEL: <positive or negative>\nREASON: <one sentence>"
    return [{"role":"system","content":sys_p},{"role":"user","content":(f"Target:\n\"{x}\"\n\nClassify." if ex else f"\"{x}\"\n\nClassify.")}]

def parse_bin(raw):
    for line in raw.strip().splitlines():
        s=line.strip()
        if s.upper().startswith("LABEL:"):
            v=s.split(":",1)[1].strip().lower()
            if "positive" in v or "pos" in v: return "positive"
            if "negative" in v or "neg" in v: return "negative"
    low=raw.lower()
    return "positive" if (low.count("positive")>=low.count("negative")) else "negative"

def rand_labels(ex,seed):
    rng=np.random.default_rng(seed+9999); return [(t,LABELS[rng.integers(0,2)]) for t,_ in ex]

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
    for fmt,fn in FMT.items():
        z=[];gold=[];rand=[];fert=0
        for s in SEEDS:
            items,ex=load(s,fn)
            z.append(run(items,ex,model,tok,False)); gold.append(run(items,ex,model,tok,True)); rand.append(run(items,rand_labels(ex,s),model,tok,True))
            if s==42:
                toks=[len(tok(x).input_ids) for x,_ in items[:100]]; wds=[max(1,len(x.split())) for x,_ in items[:100]]; fert=float(np.mean([t/w for t,w in zip(toks,wds)]))
        z,gold,rand=np.array(z),np.array(gold),np.array(rand)
        res[fmt]={"zero":float(z.mean()*100),"gold":float(gold.mean()*100),"random":float(rand.mean()*100),
                  "gain_pp":float((gold.mean()-z.mean())*100),"TR_pp":float((rand.mean()-z.mean())*100),"TL_pp":float((gold.mean()-rand.mean())*100),"fertility":fert}
        r=res[fmt]; print(f"  {args.model}/{fmt}: zero={r['zero']:.1f} gold={r['gold']:.1f} rand={r['random']:.1f} | gain={r['gain_pp']:+.1f} TR={r['TR_pp']:+.1f} TL={r['TL_pp']:+.1f} fert={r['fertility']:.2f}",flush=True)
    res["delta_gain_leet_minus_natural"]=res["leet"]["gain_pp"]-res["natural"]["gain_pp"]
    json.dump({"model":args.model,"results":res}, open(OUT/f"sent_{args.model}.json","w"), indent=2)
    print(f"[H] leet_gain - natural_gain = {res['delta_gain_leet_minus_natural']:+.1f}pp (>0 = format novelty causes gain in NON-emotion sentiment task)",flush=True)
    print(f"[SAVED] sent_{args.model}.json",flush=True)
