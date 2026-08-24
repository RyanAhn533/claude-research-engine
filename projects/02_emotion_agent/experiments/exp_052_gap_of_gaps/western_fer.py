"""exp_052 — gap-of-gaps (JY 지적 킬러): 같은 trpakov FER를 서양 얼굴(AffectNet)에.
한국(exp_051)서 happy≫neg gap 0.446였는데, 서양선 gap이 작으면 = 한국 부정감정 천장은 문화 특이.
"""
import json, glob, random
from pathlib import Path
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
MODEL="trpakov/vit-face-expression"
ROOT="/mnt/ssd2/AffectNet/archive"
EMO_DIR={"happy":"happy","angry":"anger","sad":"sad","neutral":"neutral"}  # AffectNet 폴더명
N=300; random.seed(42); HERE=Path(__file__).resolve()
proc=AutoImageProcessor.from_pretrained(MODEL)
model=AutoModelForImageClassification.from_pretrained(MODEL); model.eval()
id2label={int(k):v.lower() for k,v in model.config.id2label.items()}
MAP={"happy":"happy","happiness":"happy","angry":"angry","anger":"angry","sad":"sad","sadness":"sad","neutral":"neutral"}
out={"model":MODEL,"dataset":"AffectNet(Western)","per_emotion":{},"confusion":{}}
for emo,d in EMO_DIR.items():
    files=glob.glob(f"{ROOT}/{d}/*.jpg")+glob.glob(f"{ROOT}/{d}/*.png"); random.shuffle(files); files=files[:N]
    correct=0; conf={}
    for fp in files:
        try: img=Image.open(fp).convert("RGB")
        except: continue
        with torch.no_grad(): logits=model(**proc(img,return_tensors="pt")).logits
        pred=MAP.get(id2label[int(logits.argmax(-1))],"other"); conf[pred]=conf.get(pred,0)+1
        if pred==emo: correct+=1
    n=len(files); out["per_emotion"][emo]={"n":n,"acc":round(correct/max(n,1),3)}; out["confusion"][emo]=conf
    print(f">>> {emo}: acc={correct/max(n,1):.3f} n={n} conf={conf}",flush=True)
neg=(out["per_emotion"]["angry"]["acc"]+out["per_emotion"]["sad"]["acc"])/2
out["gap_happy_minus_neg"]=round(out["per_emotion"]["happy"]["acc"]-neg,3)
HERE.parent.joinpath("result.json").write_text(json.dumps(out,indent=2,ensure_ascii=False))
print("\n=== EXP_052 DONE === Western gap=",out["gap_happy_minus_neg"])
