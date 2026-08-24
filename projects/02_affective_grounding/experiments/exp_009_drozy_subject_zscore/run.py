"""exp_009 DROZY binary drowsiness — band-power + PER-SUBJECT z-score (skeptic's biggest miss).
Goal: push from exp_008's 0.709 toward literature LOSO ceiling 0.75-0.82.
"""
from __future__ import annotations
import json, re, glob, sys
from pathlib import Path
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, log_loss, f1_score

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "exp_008_drozy_bandpower_probe"))
from run import features, load_all, ece  # reuse band-power extraction

SEEDS = [42, 123, 777]
N_SPLITS = 5


def subject_zscore(F, subj):
    """Unsupervised per-subject z-score (uses each subject's own feature distribution, no labels)."""
    Fz = F.copy().astype(np.float32)
    for s in np.unique(subj):
        m = (subj == s)
        mu = Fz[m].mean(0); sd = Fz[m].std(0) + 1e-6
        Fz[m] = (Fz[m] - mu) / sd
    return Fz


def run_seed(F, y, subj, sess_idx, n_cls, seed, do_subj_z):
    Fwork = subject_zscore(F, subj) if do_subj_z else F
    gkf = GroupKFold(n_splits=N_SPLITS)
    oof = np.zeros((len(y), n_cls))
    for tr, te in gkf.split(Fwork, y, subj):
        sc = StandardScaler().fit(Fwork[tr])
        Xtr = sc.transform(Fwork[tr]); Xte = sc.transform(Fwork[te])
        m = LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced", random_state=seed)
        m.fit(Xtr, y[tr]); oof[te] = m.predict_proba(Xte)
    pred = oof.argmax(1); ent = -(oof * np.log(oof + 1e-12)).sum(1)
    mis = (pred != y).astype(int)
    auroc = float(roc_auc_score(mis, ent)) if mis.sum() not in (0, len(mis)) else None
    # session voting
    sess_pred, sess_true = [], []
    for sid in np.unique(sess_idx):
        mm = (sess_idx == sid); sess_pred.append(oof[mm].mean(0).argmax()); sess_true.append(int(y[mm][0]))
    sess_bal = float(balanced_accuracy_score(sess_true, sess_pred))
    sess_f1 = float(f1_score(sess_true, sess_pred, average="macro"))
    return {"oof": oof, "pred": pred, "ent": ent, "mis": mis, "y": y,
            "win_bal_acc": float(balanced_accuracy_score(y, pred)),
            "sess_bal_acc": sess_bal, "sess_f1_macro": sess_f1,
            "ece": ece(oof, y), "nll": float(log_loss(y, oof, labels=list(range(n_cls)))),
            "err_unc_auroc": auroc}


def main():
    X, y3, subj, sess_idx = load_all()
    print(f"loaded {len(X)} windows, {len(set(subj))} subjects")
    print("extracting band-power features ...")
    F = features(X)
    print(f"features shape: {F.shape}")

    # binary KSS-derived drowsy: 0=alert, 1=drowsy (combine moderate+sleepy)
    y_bin = (y3 > 0).astype(np.int64)
    # also: extreme binary (alert vs sleepy only, drop moderate)
    keep = (y3 != 1)
    F_ext, y_ext, subj_ext, sess_ext = F[keep], (y3[keep] > 0).astype(np.int64), subj[keep], sess_idx[keep]

    targets = [
        ("binary_alert_vs_drowsy", F, y_bin, subj, sess_idx, 2),
        ("binary_alert_vs_sleepy_only", F_ext, y_ext, subj_ext, sess_ext, 2),
    ]
    methods = [("subj_z+logreg", True), ("logreg_only", False)]

    out = {"exp_id": "exp_009_drozy_subject_zscore", "n_features": int(F.shape[1])}
    for tname, Ft, yt, subjt, sessit, n_cls in targets:
        out[tname] = {}
        for mname, do_z in methods:
            per = [run_seed(Ft, yt, subjt, sessit, n_cls, s, do_z) for s in SEEDS]
            # bootstrap CI on seed-0
            p0 = per[0]; rng = np.random.default_rng(0); n = len(yt); B = 2000
            boot_auroc = []
            for _ in range(B):
                idx = rng.integers(0, n, n); mm = p0["mis"][idx]
                if mm.sum() not in (0, len(mm)):
                    boot_auroc.append(roc_auc_score(mm, p0["ent"][idx]))
            ci = [float(np.percentile(boot_auroc, 2.5)), float(np.percentile(boot_auroc, 97.5))]
            # risk-coverage on seed-0
            order = np.argsort(p0["ent"])
            def sel(cov):
                k = max(2, int(cov * n)); idx = order[:k]
                return float(balanced_accuracy_score(yt[idx], p0["pred"][idx]))
            out[tname][mname] = {
                "win_bal_acc_per_seed": [p["win_bal_acc"] for p in per],
                "win_bal_acc_mean": float(np.mean([p["win_bal_acc"] for p in per])),
                "sess_bal_acc_per_seed": [p["sess_bal_acc"] for p in per],
                "sess_bal_acc_mean": float(np.mean([p["sess_bal_acc"] for p in per])),
                "sess_f1_macro_mean": float(np.mean([p["sess_f1_macro"] for p in per])),
                "err_unc_auroc_per_seed": [p["err_unc_auroc"] for p in per],
                "err_unc_auroc_mean": float(np.mean([p["err_unc_auroc"] for p in per if p["err_unc_auroc"]])),
                "err_unc_auroc_ci95": ci,
                "ece_mean": float(np.mean([p["ece"] for p in per])),
                "nll_mean": float(np.mean([p["nll"] for p in per])),
                "risk_coverage": {"full": float(np.mean([p["win_bal_acc"] for p in per])),
                                  "cov80": sel(0.8), "cov50": sel(0.5), "cov25": sel(0.25)},
            }
    # decisive call
    best_sess = max((out[t][m]["sess_bal_acc_mean"], t, m) for t in [tg[0] for tg in targets] for m in [me[0] for me in methods])
    out["best_session_bal_acc"] = best_sess
    out["lit_loso_ceiling"] = [0.75, 0.82]
    out["verdict"] = "matches_lit_ceiling" if best_sess[0] >= 0.75 else (
        "improved_over_exp008" if best_sess[0] > 0.709 else "no_improvement")
    (HERE / "results.json").write_text(json.dumps(out, indent=2))

    print()
    for tname, _, _, _, _, _ in targets:
        print(f"=== {tname} ===")
        for mname, _ in methods:
            r = out[tname][mname]
            print(f"  {mname:18} win={r['win_bal_acc_mean']:.3f} sess={r['sess_bal_acc_mean']:.3f} F1={r['sess_f1_macro_mean']:.3f}  "
                  f"AUROC={r['err_unc_auroc_mean']:.3f} CI{r['err_unc_auroc_ci95']}  "
                  f"risk-cov: full={r['risk_coverage']['full']:.3f} c50={r['risk_coverage']['cov50']:.3f} c25={r['risk_coverage']['cov25']:.3f}  "
                  f"ECE={r['ece_mean']:.3f}")
    print(f"\nbest session bal_acc = {best_sess[0]:.3f} on ({best_sess[1]}, {best_sess[2]})")
    print(f"lit LOSO ceiling = 0.75~0.82 / exp_008 baseline = 0.709 / VERDICT = {out['verdict']}")


if __name__ == "__main__":
    main()
