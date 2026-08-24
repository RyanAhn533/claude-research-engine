"""exp_051 — 진짜 SOTA FER(서양 학습)를 한국 얼굴에 직접 (JY 지적: 우리 LogReg 말고 학습된 FER baseline).
모델: trpakov/vit-face-expression (FER2013=서양 학습, 7감정 ViT).
한국 crop {happy,angry,sad,neutral} 각 N장 → per-emotion accuracy. CPU."""
import json, glob, random
from pathlib import Path
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

MODEL="trpakov/vit-face-expression"
ROOT="/home/ajy/FER_03_aihub_au_vit/data2/data_processed_korea"
KOR_EMOS=["happy","angry","sad","neutral"]   # 공유 4클래스
N=300; random.seed(42)
HERE=Path(__file__).resolve()

proc=AutoImageProcessor.from_pretrained(MODEL)
model=AutoModelForImageClassification.from_pretrained(MODEL); model.eval()
id2label={int(k):v.lower() for k,v in model.config.id2label.items()}
print("FER labels:",id2label, flush=True)
# FER2013 라벨 → 우리 4클래스 매핑 (없는 disgust/fear/surprise = other)
MAP={"happy":"happy","happiness":"happy","angry":"angry","anger":"angry",
     "sad":"sad","sadness":"sad","neutral":"neutral"}

out={"model":MODEL,"per_emotion":{}, "confusion":{}}
for emo in KOR_EMOS:
    files=glob.glob(f"{ROOT}/{emo}/*.jpg")+glob.glob(f"{ROOT}/{emo}/*.png")
    random.shuffle(files); files=files[:N]
    correct=0; conf={}
    for fp in files:
        try: img=Image.open(fp).convert("RGB")
        except: continue
        with torch.no_grad():
            logits=model(**proc(img,return_tensors="pt")).logits
        pred_raw=id2label[int(logits.argmax(-1))]
        pred=MAP.get(pred_raw,"other")
        conf[pred]=conf.get(pred,0)+1
        if pred==emo: correct+=1
    n=len([1 for fp in files])
    out["per_emotion"][emo]={"n":n,"acc":round(correct/max(n,1),3)}
    out["confusion"][emo]=conf
    print(f">>> {emo}: acc={correct/max(n,1):.3f} n={n} conf={conf}", flush=True)
neg=(out["per_emotion"]["angry"]["acc"]+out["per_emotion"]["sad"]["acc"])/2
out["gap_happy_minus_neg"]=round(out["per_emotion"]["happy"]["acc"]-neg,3)
HERE.parent.joinpath("result.json").write_text(json.dumps(out,indent=2,ensure_ascii=False))
print("\n=== EXP_051 DONE ===  gap=",out["gap_happy_minus_neg"])
print(json.dumps(out["per_emotion"],ensure_ascii=False))
