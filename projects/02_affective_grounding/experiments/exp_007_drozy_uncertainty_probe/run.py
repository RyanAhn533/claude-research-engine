"""exp_007 DROZY drowsiness uncertainty probe (Track B survival gate, pivoted).
Subject-wise GroupKFold + small 1D-CNN deep-ensemble on EEG windows.
Primary metric: window-level error-vs-uncertainty AUROC.
"""
from __future__ import annotations
import json, re, glob
from pathlib import Path
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, log_loss

HERE = Path(__file__).parent
SESS_DIR = Path("/home/ajy/04_driver_monitoring/03_DMS_sleep_classification/251219/data/sessions")
SEEDS = [42, 123, 777]
N_SPLITS = 5
N_ENSEMBLE = 5
EPOCHS = 8           # quick — ~21k windows × 5 channels is plenty
BATCH = 256


def load_all():
    files = sorted(glob.glob(str(SESS_DIR / "*.npz")))
    Xs, ys, sids, sessids, win_sess = [], [], [], [], []
    for f in files:
        d = np.load(f)
        X = d["X"]; y = int(d["y"]); sess = str(d["session"])
        subj = int(re.match(r"(\d+)-", sess).group(1))
        n = X.shape[0]
        Xs.append(X); ys.append(np.full(n, y, dtype=np.int64))
        sids.append(np.full(n, subj, dtype=np.int64))
        sessids.append(np.full(n, len(Xs) - 1, dtype=np.int64))  # session index
        win_sess.append(sess)
    X = np.concatenate(Xs, axis=0)               # (W, T=512, C=5)
    y = np.concatenate(ys); subj = np.concatenate(sids); sess_idx = np.concatenate(sessids)
    print(f"loaded {len(files)} sessions, {len(X)} windows, {len(set(subj))} subjects, "
          f"classes={sorted(set(y))}, per-class {np.bincount(y).tolist()}")
    return X.astype(np.float32), y, subj, sess_idx, win_sess


def standardize(Xtr, Xte):
    # per-channel standardize using train stats
    mu = Xtr.mean(axis=(0, 1), keepdims=True); sd = Xtr.std(axis=(0, 1), keepdims=True) + 1e-6
    return (Xtr - mu) / sd, (Xte - mu) / sd


def train_ensemble_predict(Xtr, ytr, Xte, n_cls, seed):
    import torch, torch.nn as nn
    torch.manual_seed(seed); np.random.seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # (W, T, C) -> (W, C, T) for Conv1d
    def to_torch(X):
        return torch.tensor(X.transpose(0, 2, 1), dtype=torch.float32)

    Xtr_t = to_torch(Xtr); ytr_t = torch.tensor(ytr, dtype=torch.long)
    Xte_t = to_torch(Xte).to(dev)

    class CNN(nn.Module):
        def __init__(self, c_in=5, n_cls=3):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(c_in, 16, 9, padding=4), nn.ReLU(), nn.MaxPool1d(4),
                nn.Conv1d(16, 32, 7, padding=3), nn.ReLU(), nn.MaxPool1d(4),
                nn.Conv1d(32, 64, 5, padding=2), nn.ReLU(), nn.AdaptiveAvgPool1d(1),
                nn.Flatten(), nn.Dropout(0.5), nn.Linear(64, n_cls))
        def forward(self, x): return self.net(x)

    preds = []
    for m in range(N_ENSEMBLE):
        torch.manual_seed(seed * 100 + m)
        net = CNN(c_in=Xtr.shape[2], n_cls=n_cls).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
        lossf = nn.CrossEntropyLoss()
        ds = torch.utils.data.TensorDataset(Xtr_t, ytr_t)
        dl = torch.utils.data.DataLoader(ds, batch_size=BATCH, shuffle=True, num_workers=0)
        net.train()
        for ep in range(EPOCHS):
            for xb, yb in dl:
                xb, yb = xb.to(dev), yb.to(dev)
                opt.zero_grad(); loss = lossf(net(xb), yb); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            out = []
            for i in range(0, len(Xte_t), 1024):
                out.append(torch.softmax(net(Xte_t[i:i+1024]), 1).cpu().numpy())
            preds.append(np.concatenate(out, 0))
    return np.mean(preds, axis=0)


def ece(probs, y, n_bins=15):
    conf = probs.max(1); pred = probs.argmax(1); correct = (pred == y).astype(float)
    bins = np.linspace(0, 1, n_bins + 1); e = 0.0
    for i in range(n_bins):
        m = (conf > bins[i]) & (conf <= bins[i + 1])
        if m.sum() > 0:
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def brier_multi(probs, y, n_cls):
    return float(((probs - np.eye(n_cls)[y]) ** 2).sum(1).mean())


def run_seed(X, y, subj, n_cls, seed):
    gkf = GroupKFold(n_splits=N_SPLITS)
    oof = np.zeros((len(y), n_cls), dtype=np.float32)
    for tr, te in gkf.split(X, y, subj):
        Xtr_s, Xte_s = standardize(X[tr], X[te])
        oof[te] = train_ensemble_predict(Xtr_s, y[tr], Xte_s, n_cls, seed)
    pred = oof.argmax(1); ent = -(oof * np.log(oof + 1e-12)).sum(1)
    mis = (pred != y).astype(int)
    auroc = float(roc_auc_score(mis, ent)) if mis.sum() not in (0, len(mis)) else None
    return {
        "oof": oof, "pred": pred, "ent": ent, "mis": mis,
        "bal_acc": float(balanced_accuracy_score(y, pred)),
        "ece": ece(oof, y), "nll": float(log_loss(y, oof, labels=list(range(n_cls)))),
        "brier": brier_multi(oof, y, n_cls), "err_unc_auroc": auroc,
    }


def main():
    X, y, subj, sess_idx, win_sess = load_all()
    n_cls = int(y.max()) + 1
    print(f"running {len(SEEDS)} seeds × {N_SPLITS} folds × {N_ENSEMBLE} ensemble (~{len(SEEDS)*N_SPLITS*N_ENSEMBLE} small CNN trainings)")
    per = [run_seed(X, y, subj, n_cls, s) for s in SEEDS]

    # session-level: vote within session (mean prob), use seed-0 OOF
    p0 = per[0]; sess_pred, sess_true = [], []
    for sid in np.unique(sess_idx):
        m = (sess_idx == sid); sess_pred.append(p0["oof"][m].mean(0).argmax()); sess_true.append(int(y[m][0]))
    sess_bal = float(balanced_accuracy_score(sess_true, sess_pred))

    # bootstrap CI on seed-0 (window-level)
    rng = np.random.default_rng(0); n = len(y); B = 2000
    boot_auroc, boot_bal = [], []
    for _ in range(B):
        idx = rng.integers(0, n, n); m = p0["mis"][idx]
        if m.sum() not in (0, len(m)):
            boot_auroc.append(roc_auc_score(m, p0["ent"][idx]))
        boot_bal.append(balanced_accuracy_score(y[idx], p0["pred"][idx]))
    auroc_ci = [float(np.percentile(boot_auroc, 2.5)), float(np.percentile(boot_auroc, 97.5))]
    bal_ci = [float(np.percentile(boot_bal, 2.5)), float(np.percentile(boot_bal, 97.5))]

    # risk-coverage seed-0
    order = np.argsort(p0["ent"])
    def sel(cov):
        k = max(2, int(cov * n)); idx = order[:k]
        return float(balanced_accuracy_score(y[idx], p0["pred"][idx]))
    rc = {"full": p0["bal_acc"], "cov80": sel(0.8), "cov50": sel(0.5)}

    aucs = [p["err_unc_auroc"] for p in per if p["err_unc_auroc"] is not None]
    bals = [p["bal_acc"] for p in per]
    auroc_mean = float(np.mean(aucs)) if aucs else None
    THRESH = 0.60
    supported = bool(auroc_mean is not None and auroc_mean >= THRESH and auroc_ci[0] > 0.50)

    res = {
        "exp_id": "exp_007_drozy_uncertainty_probe", "n_windows": int(n), "n_classes": int(n_cls),
        "n_subjects": int(len(set(subj))), "n_sessions": int(len(set(sess_idx))),
        "chance": 1 / n_cls, "majority": float(np.bincount(y).max() / n),
        "seeds": SEEDS, "bal_acc_per_seed": bals, "bal_acc_mean": float(np.mean(bals)),
        "bal_acc_ci95": bal_ci, "session_bal_acc": sess_bal,
        "err_unc_auroc_per_seed": [p["err_unc_auroc"] for p in per],
        "err_unc_auroc_mean": auroc_mean, "err_unc_auroc_ci95": auroc_ci,
        "ece_mean": float(np.mean([p["ece"] for p in per])),
        "nll_mean": float(np.mean([p["nll"] for p in per])),
        "brier_mean": float(np.mean([p["brier"] for p in per])),
        "risk_coverage_balacc": rc, "threshold": THRESH,
        "supported": supported,
        "verdict": "uncertainty_informative" if supported else "uncertainty_uninformative",
    }
    (HERE / "results.json").write_text(json.dumps(res, indent=2))
    print(f"\nwindow bal_acc={res['bal_acc_mean']:.3f} (chance {res['chance']:.3f}, majority {res['majority']:.3f}) CI{bal_ci}")
    print(f"session bal_acc={sess_bal:.3f}  ECE={res['ece_mean']:.3f}  NLL={res['nll_mean']:.3f}")
    print(f"*** err-vs-unc AUROC = {auroc_mean} CI{auroc_ci} (>=0.60 & CI_lo>0.5 => informative) ***")
    print(f"risk-cov bal_acc: full={rc['full']:.3f} cov80={rc['cov80']:.3f} cov50={rc['cov50']:.3f}")
    print(f"VERDICT = {res['verdict']}")


if __name__ == "__main__":
    main()
