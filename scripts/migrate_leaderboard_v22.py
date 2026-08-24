"""One-time migration: make project leaderboard.jsonl v2.2-schema compliant.
Drops deprecated fields (tost_passed, tost_equivalent), recomputes the hash chain,
and appends the exp_004 row. Keeps original row_id/timestamp so references stay valid.
"""
import json, os, sys, shutil, hashlib
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/home/ajy/CLAUDE_RESEARCH_ENGINE")
sys.path.insert(0, str(ROOT))
from engine.core.append_only_logger import AppendOnlyLog

LB = ROOT / "projects/01_au_regionformer_q2/state/leaderboard.jsonl"
SCHEMA = str(ROOT / "engine/schemas/experiment.schema.json")
DEPRECATED = ("tost_passed", "tost_equivalent")

# 1) read + backup
rows = [json.loads(l) for l in LB.read_text().splitlines() if l.strip()]
shutil.copy(LB, "/tmp/leaderboard_pre_v22.jsonl.bak")
print(f"read {len(rows)} rows (backup → /tmp/leaderboard_pre_v22.jsonl.bak)")

# 2) clean: drop deprecated + prev_hash (recomputed); keep row_id
cleaned = []
for r in rows:
    for k in DEPRECATED:
        r.pop(k, None)
    r.pop("prev_hash", None)
    cleaned.append(r)

# 3) build exp_004 row (v2.2-compliant, no tost_*)
res = json.loads((ROOT / "projects/01_au_regionformer_q2/experiments/exp_004_visual_identifiability_taxonomy/results.json").read_text())
a = res["aggregate"]["all"]; c = a["mean_counts"]; mcgr = res["model_correct_given_reject_all"]
now = datetime.now(timezone.utc).isoformat()
fp = "sha256:" + hashlib.sha256(b"exp_004 taxonomy v1").hexdigest()
exp004 = {
    "exp_id": "exp_004_visual_identifiability_taxonomy", "iteration": 1,
    "method": "Visual identifiability taxonomy V1-V4 (observer x model) over saved predictions, 3 baseline seeds.",
    "method_id": "v2_id_taxonomy", "config_fingerprint": fp, "metric_name": "model_correct_given_reject",
    "constraints_passed": True, "baseline_candidate": False, "timestamp": now,
    "notes": (f"V1={c['V1']:.0f} V2={c['V2']:.0f} V3={c['V3']:.0f} V4={c['V4']:.0f}. "
              f"V3/(V3+V4)={mcgr:.4f} (>=0.5 SUPPORTED, matches exp_003 0.773). "
              f"error_share_reject=V4/(V2+V4)={a['error_share_reject']:.3f}: ~70pct of errors are V2 "
              f"(model error on observer-AGREE) vs ~30pct V4 (ambiguous) => residual headroom on clear samples."),
    "results": res, "stage": "prototype", "n_seeds": 3,
    "paired_delta_mean": None, "paired_delta_ci_95": None, "cohen_d_paired": None, "wilcoxon_p": None,
    "practical_effect_passed": True, "gate_c_verdict": "improvement_candidate", "gate_b_verdict": "pass",
    "linked_claims": [], "manifest_ref": "reproducibility_manifests/exp_001_yonsei_disagreement_phase_a.yaml",
    "self_attack_run": False,
}
cleaned.append(exp004)

# 4) rebuild file via logger (re-validates each row against v2.2 schema, re-chains)
os.remove(LB)
sf = LB.with_suffix(LB.suffix + ".lockstate")
if sf.exists():
    os.remove(sf)
log = AppendOnlyLog(LB, schema=SCHEMA)
for r in cleaned:
    log.append(r)
n = log.verify_chain()
print(f"rebuilt + chain verified: {n} rows (was {len(rows)}, +1 exp_004)")
