"""exp_053 — 문화 taxonomy 불일치 (B 승부수): 서양 FER 7-class를 한국 7감정 전부에.
한국 hurt(상처)/anxious(불안)는 서양 taxonomy에 없음 → 어디로 collapse하나?
trpakov(FER2013 7class: angry,disgust,fear,happy,sad,surprise,neutral)."""
import json, glob, random
from pathlib import Path
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
MODEL="trpakov/vit-face-expression"
ROOT="/home/ajy/FER_03_aihub_au_vit/data2/data_processed_korea"
KOR=["happy","angry","sad","neutral","anxious","hurt","surprised"]  # 한국 7감정
N=250; random.seed(42); HERE=Path(__file__).resolve()
proc=AutoImageProcessor.from_pretrained(MODEL)
model=AutoModelForImageClassification.from_pretrained(MODEL); model.eval()
id2label={int(k):v.lower() for k,v in model.config.id2label.items()}
print("Western 7-class:",sorted(id2label.values()),flush=True)
out={"model":MODEL,"korean_to_western_dist":{}}
for emo in KOR:
    files=glob.glob(f"{ROOT}/{emo}/*.jpg")+glob.glob(f"{ROOT}/{emo}/*.png"); random.shuffle(files); files=files[:N]
    dist={}
    for fp in files:
        try: img=Image.open(fp).convert("RGB")
        except: continue
        with torch.no_grad(): logits=model(**proc(img,return_tensors="pt")).logits
        p=id2label[int(logits.argmax(-1))]; dist[p]=dist.get(p,0)+1
    n=sum(dist.values()); top=sorted(dist.items(),key=lambda x:-x[1])
    out["korean_to_western_dist"][emo]={"n":n,"dist_pct":{k:round(v/n,3) for k,v in top}}
    print(f">>> 한국 {emo} (n={n}) -> 서양: "+", ".join(f"{k}:{v/n:.2f}" for k,v in top[:4]),flush=True)
HERE.parent.joinpath("result.json").write_text(json.dumps(out,indent=2,ensure_ascii=False))
print("\n=== EXP_053 DONE ===")
