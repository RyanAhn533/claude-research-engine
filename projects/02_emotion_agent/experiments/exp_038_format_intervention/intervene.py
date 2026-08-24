"""
exp_038: CAUSAL intervention — does input FORMAT novelty (not content) drive ICL benefit?

Same facial signal (which AUs active + intensity), two surface formats:
  NUMERIC (alien, LLM never mapped to emotion): "AU6 (cheek raiser)=75, AU12 (lip corner puller)=82"
  PROSE   (familiar natural description):        "strong cheek raising, strong lip-corner pulling"
Content identical; only surface form differs.

Pre-registered hypothesis:
  H1: ICL gain(NUMERIC) > ICL gain(PROSE)  (format novelty -> ICL benefit)
  H1b: zero-shot(PROSE) > zero-shot(NUMERIC) (familiar format easier to map cold)
  NULL: ICL gain same across formats (=> content, not format, drives gating)
Models: qwen7b, mistral. 3 seeds, N=400. Batched.
"""
import os, sys, json, glob, argparse
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "exp_035_multimodel_gating"))
import run_gating as G

OUT = Path(__file__).parent / "cache"; OUT.mkdir(exist_ok=True, parents=True)
AU_PARQUET = G.AU_PARQUET
LABELS = G.LABELS; KR_TO_EN = G.KR_TO_EN; AU_DESC = G.AU_DESC; KEY_AUS = G.KEY_AUS
SEEDS = [42, 123, 777, 2024, 31337]

def intensity_word(v):
    return "strong" if v >= 70 else ("moderate" if v >= 50 else "slight")

# action gerunds for prose (same AU semantics as AU_DESC, natural-language surface)
def numeric_fmt(row, thr=50):
    active = [f"{au} ({AU_DESC[au]})={row[au]:.0f}" for au in KEY_AUS
              if au in row and not pd.isna(row[au]) and row[au] >= thr]
    if not active:
        sc = sorted(((au, float(row[au])) for au in KEY_AUS if au in row and not pd.isna(row[au])), key=lambda x:-x[1])[:3]
        active = [f"{a} ({AU_DESC[a]})={v:.0f}" for a,v in sc]
    return ", ".join(active)

def prose_fmt(row, thr=50):
    active = [f"{intensity_word(row[au])} {AU_DESC[au].replace('raiser','raising').replace('lowerer','lowering').replace('puller','pulling').replace('depressor','depressing').replace('tightener','tightening').replace('stretcher','stretching').replace('wrinkler','wrinkling').replace('drop','dropping').replace('part','parting').replace('dimpler','dimpling')}"
              for au in KEY_AUS if au in row and not pd.isna(row[au]) and row[au] >= thr]
    if not active:
        sc = sorted(((au, float(row[au])) for au in KEY_AUS if au in row and not pd.isna(row[au])), key=lambda x:-x[1])[:3]
        active = [f"{intensity_word(v)} {AU_DESC[a]}" for a,v in sc]
    return "The face shows " + ", ".join(active) + "."

def prompt_numeric(x, ex):
    sys_p = ("You are a facial emotion classifier using FACS action units (AUs). "
             "Given detected AU intensities (0-100) of a Korean subject, classify into: angry, happy, neutral, sad.\n")
    if ex:
        sys_p += f"Here are {len(ex)} Korean examples with correct labels:\n"
        for i,(t,l) in enumerate(ex): sys_p += f"\nExample {i+1}:\n  AUs: {t}\n  LABEL: {l}\n"
        sys_p += "\nNow classify the target.\n"
    sys_p += "Output format:\nLABEL: <label>\nREASON: <one sentence>"
    u = f"Target AUs:\n{x}\n\nClassify emotion." if ex else f"AUs: {x}\n\nClassify emotion."
    return [{"role":"system","content":sys_p},{"role":"user","content":u}]

def prompt_prose(x, ex):
    sys_p = ("You are a facial emotion classifier. Given a description of a Korean subject's "
             "facial movements, classify into: angry, happy, neutral, sad.\n")
    if ex:
        sys_p += f"Here are {len(ex)} Korean examples with correct labels:\n"
        for i,(t,l) in enumerate(ex): sys_p += f"\nExample {i+1}:\n  Face: {t}\n  LABEL: {l}\n"
        sys_p += "\nNow classify the target.\n"
    sys_p += "Output format:\nLABEL: <label>\nREASON: <one sentence>"
    u = f"Target face:\n{x}\n\nClassify emotion." if ex else f"{x}\n\nClassify emotion."
    return [{"role":"system","content":sys_p},{"role":"user","content":u}]

FORMATS = {"numeric": (numeric_fmt, prompt_numeric), "prose": (prose_fmt, prompt_prose)}

def load(seed, fmt_fn):
    df = pd.read_parquet(AU_PARQUET)
    df = df[df["is_selected"]==0].copy(); df["emotion_en"]=df["emotion"].map(KR_TO_EN)
    df = df[df["emotion_en"].notna()].copy()
    rng = np.random.default_rng(seed)
    ex_idx=[df[df["emotion_en"]==l].iloc[rng.integers(0,len(df[df["emotion_en"]==l]))].name for l in LABELS]
    ex_df=df.loc[ex_idx]; rem=df.drop(index=ex_idx); rng2=np.random.default_rng(seed)
    pieces=[]
    for l in LABELS:
        sub=rem[rem["emotion_en"]==l]
        pieces.append(sub.iloc[rng2.choice(len(sub),size=min(100,len(sub)),replace=False)])
    test=pd.concat(pieces).reset_index(drop=True)
    ex=[(fmt_fn(r), r["emotion_en"]) for _,r in ex_df.iterrows()]
    items=[(fmt_fn(r), r["emotion_en"]) for _,r in test.iterrows()]
    return items, ex

@torch.inference_mode()
def run_cfg(items, ex, prompt_fn, model, tok, use_icl, bs=32):
    y=[g for _,g in items]; msgs=[prompt_fn(x, ex if use_icl else None) for x,_ in items]; preds=[]
    for b in range(0,len(msgs),bs):
        ps=[tok.apply_chat_template(m,tokenize=False,add_generation_prompt=True) for m in msgs[b:b+bs]]
        enc=tok(ps,return_tensors="pt",padding=True,truncation=True,max_length=2048).to(model.device)
        out=model.generate(**enc,max_new_tokens=48,do_sample=False,pad_token_id=tok.pad_token_id)
        preds.extend(G.parse_output(tok.decode(g,skip_special_tokens=True)) for g in out[:,enc.input_ids.shape[1]:])
    return float(accuracy_score(y,preds))

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--model",default="qwen7b"); args=ap.parse_args()
    snap=glob.glob(f"{G.HUB}/models--{G.MODELS[args.model]}/snapshots/*")[0]
    print(f"[load] {args.model}",flush=True)
    bnb=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",bnb_4bit_compute_dtype=torch.float16,bnb_4bit_use_double_quant=True)
    tok=AutoTokenizer.from_pretrained(snap); tok.padding_side="left"
    if tok.pad_token_id is None: tok.pad_token=tok.eos_token
    model=AutoModelForCausalLM.from_pretrained(snap,quantization_config=bnb,device_map={"":0},torch_dtype=torch.float16).eval()
    res={}
    for fmt,(fmt_fn,prompt_fn) in FORMATS.items():
        z=[]; ic=[]
        for s in SEEDS:
            items,ex=load(s,fmt_fn)
            z.append(run_cfg(items,ex,prompt_fn,model,tok,False))
            ic.append(run_cfg(items,ex,prompt_fn,model,tok,True))
        zc=np.array(z); icc=np.array(ic)
        res[fmt]={"zero_mean":float(zc.mean()),"icl_mean":float(icc.mean()),
                  "gain_pp":float((icc.mean()-zc.mean())*100),"zero":z,"icl":ic}
        print(f"  {args.model}/{fmt}: zero={zc.mean()*100:.2f}% icl={icc.mean()*100:.2f}% GAIN={ (icc.mean()-zc.mean())*100:+.2f}pp",flush=True)
    res["delta_gain_numeric_minus_prose_pp"]=res["numeric"]["gain_pp"]-res["prose"]["gain_pp"]
    with open(OUT/f"intervention_{args.model}.json","w") as f: json.dump({"model":args.model,"results":res},f,indent=2)
    print(f"[H1] numeric_gain - prose_gain = {res['delta_gain_numeric_minus_prose_pp']:+.2f}pp (>0 supports format-novelty cause)",flush=True)
    print(f"[SAVED] {OUT}/intervention_{args.model}.json",flush=True)
