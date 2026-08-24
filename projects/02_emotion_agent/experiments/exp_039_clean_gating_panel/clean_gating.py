"""
exp_039: CLEAN gating (no FACS-prototype scaffolding) across an EXPANDED model panel.
Settles two things at once:
  (1) FACS-confound: AU uses the no-FACS numeric prompt (exp_038), so any gating is
      from few-shot examples, not prompt hints.
  (2) Generality: is novel>>familiar gating Qwen-specific or general? Test 6 models /
      4+ families / multiple scales.

Per model: AU(novel, no-FACS) / IEMOCAP / MELD (familiar) x 5 seeds x {zero,icl_k4}, batched 4-bit.
"""
import os, sys, json, glob, argparse
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from sklearn.metrics import accuracy_score
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "exp_035_multimodel_gating"))
sys.path.insert(0, str(HERE.parents[1] / "exp_038_format_intervention"))
import run_gating as G
import intervene as IV   # provides load() (AU numeric, no FACS) + prompt_numeric

OUT = HERE.parent / "cache"; OUT.mkdir(exist_ok=True, parents=True)
SEEDS = [42, 123, 777, 2024, 31337]

# AU(no-FACS numeric) uses IV.load + IV.prompt_numeric; text uses run_gating loaders+prompt
def au_loader(seed):
    return IV.load(seed, IV.numeric_fmt)
DATASETS = {
    "au_nofacs": ("novel", au_loader, IV.prompt_numeric),
    "iemocap":   G.DATASETS["iemocap"],
    "meld":      G.DATASETS["meld"],
}

@torch.inference_mode()
def run_cfg(items, ex, prompt_fn, model, tok, use_icl, bs=32):
    y=[g for _,g in items]; msgs=[prompt_fn(x if x else "neutral.", ex if use_icl else None) for x,_ in items]; preds=[]
    for b in range(0,len(msgs),bs):
        ps=[tok.apply_chat_template(m,tokenize=False,add_generation_prompt=True) for m in msgs[b:b+bs]]
        enc=tok(ps,return_tensors="pt",padding=True,truncation=True,max_length=2048).to(model.device)
        out=model.generate(**enc,max_new_tokens=48,do_sample=False,pad_token_id=tok.pad_token_id)
        preds.extend(G.parse_output(tok.decode(g,skip_special_tokens=True)) for g in out[:,enc.input_ids.shape[1]:])
    return float(accuracy_score(y,preds))

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--model",required=True)
    ap.add_argument("--load",choices=["4bit","fp16"],default="4bit"); args=ap.parse_args()
    snaps=glob.glob(f"{G.HUB}/models--{G.MODELS[args.model]}/snapshots/*")
    assert snaps, f"{args.model} not cached"
    snap=snaps[0]; print(f"[load] {args.model} mode={args.load}",flush=True)
    tok=AutoTokenizer.from_pretrained(snap,trust_remote_code=True); tok.padding_side="left"
    if tok.pad_token_id is None: tok.pad_token=tok.eos_token
    if args.load=="fp16":
        model=AutoModelForCausalLM.from_pretrained(snap,torch_dtype=torch.float16,trust_remote_code=True).to("cuda").eval()
    else:
        bnb=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",bnb_4bit_compute_dtype=torch.float16,bnb_4bit_use_double_quant=True)
        model=AutoModelForCausalLM.from_pretrained(snap,quantization_config=bnb,device_map={"":0},torch_dtype=torch.float16,trust_remote_code=True).eval()
    print(f"[load] VRAM {torch.cuda.memory_allocated()/1024**2:.0f}MB",flush=True)
    results={}
    for ds,(nov,loader,prompt_fn) in DATASETS.items():
        z=[]; ic=[]
        for s in SEEDS:
            items,ex=loader(s)
            z.append(run_cfg(items,ex,prompt_fn,model,tok,False))
            ic.append(run_cfg(items,ex,prompt_fn,model,tok,True))
        zc=np.array(z); icc=np.array(ic)
        results[ds]={"novelty":nov,"zero_mean":float(zc.mean()),"icl_mean":float(icc.mean()),
                     "gain_pp":float((icc.mean()-zc.mean())*100),"zero":z,"icl":ic}
        print(f"  {args.model}/{ds}({nov}): zero={zc.mean()*100:.1f} icl={icc.mean()*100:.1f} GAIN={(icc.mean()-zc.mean())*100:+.2f}pp",flush=True)
    json.dump({"model":args.model,"seeds":SEEDS,"results":results}, open(OUT/f"clean_{args.model}.json","w"), indent=2)
    g=results
    print(f"  >>> {args.model}: AU={g['au_nofacs']['gain_pp']:+.1f} | ie={g['iemocap']['gain_pp']:+.1f} | meld={g['meld']['gain_pp']:+.1f}",flush=True)
    print(f"[SAVED] clean_{args.model}.json",flush=True)
