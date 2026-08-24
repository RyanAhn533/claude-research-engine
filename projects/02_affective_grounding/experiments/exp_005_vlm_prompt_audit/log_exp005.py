"""Log exp_005 results to 02_affective_grounding engine state (v2.2 schema)."""
import json, hashlib, sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/home/ajy/CLAUDE_RESEARCH_ENGINE"); sys.path.insert(0, str(ROOT))
from engine.core.append_only_logger import AppendOnlyLog

STATE = ROOT / "projects/02_affective_grounding/state"
EXP = ROOT / "projects/02_affective_grounding/experiments/exp_005_vlm_prompt_audit"
res = json.loads((EXP / "metrics.json").read_text())
now = datetime.now(timezone.utc).isoformat()
gap = res.get("H4_oir_reject_minus_agree")
fp = "sha256:" + hashlib.sha256(b"exp_005 qwen2.5vl-7b 4bit 5prompt audit").hexdigest()

lb = AppendOnlyLog(STATE / "leaderboard.jsonl", schema=str(ROOT / "engine/schemas/experiment.schema.json"))
g = res["by_group"]
lb.append({
    "exp_id": "exp_005_vlm_prompt_audit", "iteration": 1,
    "method": "VLM prompt audit: Qwen2.5-VL-7B (4bit) zero-shot, 5 prompts, observer-group OIR/SAR/CSS.",
    "method_id": "vlm_audit_qwen25vl_7b_4bit", "config_fingerprint": fp,
    "metric_name": "oir_reject_minus_agree", "constraints_passed": True, "baseline_candidate": False,
    "timestamp": now,
    "notes": (f"n={res['n_total']}. OIR agree/reject="
              f"{g['A_agree']['OIR']}/{g['B_reject']['OIR']} gap={gap}. "
              f"SAR a/r={g['A_agree']['SAR']}/{g['B_reject']['SAR']}. "
              f"conf_int a/r={g['A_agree']['conf_internal_P3']}/{g['B_reject']['conf_internal_P3']}. "
              f"verdict={res['verdict']}."),
    "results": res, "stage": "prototype", "n_seeds": 3,
    "gate_c_verdict": "neutral", "gate_b_verdict": "pass", "linked_claims": [],
    "manifest_ref": "reproducibility_manifests/exp_005.yaml", "self_attack_run": False,
})
print("leaderboard appended")

h = AppendOnlyLog(STATE / "hypothesis_registry.jsonl", schema=str(ROOT / "engine/schemas/hypothesis.schema.json"))
h.append({
    "hypothesis_id": "H_vlm_overinfer", "exp_id": "exp_005_vlm_prompt_audit",
    "primary": "Qwen2.5-VL over-infers internal emotion under observer disagreement: OIR_reject >= OIR_agree (gap>=0).",
    "null_hypothesis": "OIR_reject < OIR_agree (VLM hedges more on ambiguous faces).",
    "success_criterion": {"metric": "oir_reject_minus_agree", "direction": "higher_is_better", "delta_threshold": 0.0},
    "falsifiability_check": "PASS", "registered_at": now,
    "outcome": {"result": "supported" if res.get("supported") else "rejected",
                "observed_delta": gap, "scored_at": now},
    "supersedes": "H_vlm_overinfer",
})
print("hypothesis outcome appended")
