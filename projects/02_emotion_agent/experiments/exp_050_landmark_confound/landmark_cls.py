"""exp_050 — confound 방어 #2: landmark geometric feature(17) per-emotion classifier.
OpenGraphAU(41 AU)와 완전히 다른 추출기. 여기서도 happy>>negative면 천장은 추출기 아닌 신호."""
import json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
CSV="/home/ajy/AU-RegionFormer/data/label_quality/face_features.csv"
KR={"기쁨":"happy","분노":"angry","슬픔":"sad","중립":"neutral"}
feat=["ear_left","ear_right","ear_avg","mar","mouth_width","brow_height_left","brow_height_right",
      "brow_height_avg","brow_furrow","nose_bridge","cheek_raise_left","cheek_raise_right",
      "cheek_raise_avg","lip_corner_angle","face_aspect_ratio","chin_length","forehead_height"]
df=pd.read_csv(CSV); df["y"]=df["emotion"].map(KR); df=df[df.y.notna()].copy()
X=df[feat].replace([np.inf,-np.inf],np.nan).fillna(df[feat].median()).values; y=df["y"].values
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.2,stratify=y,random_state=42)
sc=StandardScaler().fit(Xtr); clf=LogisticRegression(max_iter=2000,multi_class="multinomial").fit(sc.transform(Xtr),ytr)
pred=clf.predict(sc.transform(Xte))
out={"extractor":"landmark_geometric_17","overall_acc":float(accuracy_score(yte,pred)),
     "macro_f1":float(f1_score(yte,pred,average="macro")),"per_emotion_acc":{}}
for e in ["happy","angry","sad","neutral"]:
    m=yte==e; out["per_emotion_acc"][e]=float(accuracy_score(yte[m],pred[m])) if m.sum() else None
neg=np.mean([out["per_emotion_acc"]["angry"],out["per_emotion_acc"]["sad"]])
out["happy_minus_neg_gap"]=round(out["per_emotion_acc"]["happy"]-neg,3)
Path(__file__).parent.joinpath("result.json").write_text(json.dumps(out,indent=2,ensure_ascii=False))
print("=== EXP_050 landmark DONE ==="); print(json.dumps(out,ensure_ascii=False))
