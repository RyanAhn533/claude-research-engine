"""HT-DAR MOSEI stripped 사다리 결과 집계 → 엔진 leaderboard append.

각 run 로그(logs_run/<EXP>_s<SEED>.log)에서 val Loss 최소 epoch의 TEST acc_2를
선택값으로 뽑아 변형(variant)별로 seed를 모은다.
- n_seeds >= 3 변형 → leaderboard.jsonl 에 집계 row 1개 (per_seed/mean/std는 results)
- n_seeds < 3 변형 → exploratory_single_seed.json 에만 기록 (leaderboard 금지; v2 invariant)

KeyEval=Loss → best model = lowest val loss (DLF trainer 동작과 일치).
멱등: 이미 같은 (variant, n_seeds) 집계가 있으면 건너뜀.

Usage: python log_results.py
"""
import json, re, sys, statistics
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/home/ajy/CLAUDE_RESEARCH_ENGINE")
SD = ROOT / "projects/03_ht_dar/state"
SCH = ROOT / "engine/schemas"
LOGS = Path("/home/ajy/HT-DAR/code/logs_run")
EXP = "exp_001_mosei_stripped_ladder"
FP = json.loads((ROOT / "projects/03_ht_dar/artifacts/CONFIG_FINGERPRINTS.json").read_text())
EXPLORE = ROOT / "projects/03_ht_dar/artifacts/exploratory_single_seed.json"
sys.path.insert(0, str(ROOT))
from engine.core.append_only_logger import AppendOnlyLog

VAL_RE = re.compile(r"VAL-\(.*?\).*?acc_2:\s*([\d.]+).*?Loss:\s*([\d.]+)")
TEST_RE = re.compile(r"TEST-\(.*?\).*?acc_2:\s*([\d.]+).*?F1_score:\s*([\d.]+).*?Loss:\s*([\d.]+)")

VARIANT_DESC = {
    "M0": "encoder + meanpool (baseline)",
    "M1": "+ Hierarchical pyramid L=3",
    "M2": "+ DAR only (L=1)",
    "M3": "+ Hierarchical + DAR",
    "M4": "+ Hier + DAR + Evidential Beta gate (full)",
}
REFERENCE = {"M0": 0.8504, "M1": 0.8559, "M2": 0.8539, "M3": 0.8550, "M4": 0.8528}


def parse_log(path: Path):
    val, test = [], []
    for line in path.read_text(errors="ignore").splitlines():
        m = VAL_RE.search(line)
        if m:
            val.append((float(m.group(2)), float(m.group(1))))   # (loss, acc2)
        m = TEST_RE.search(line)
        if m:
            test.append((float(m.group(1)), float(m.group(2))))  # (acc2, f1)
    n = min(len(val), len(test))
    if n < 2:
        return None
    best_i = min(range(n), key=lambda i: val[i][0])
    return dict(best_epoch=best_i + 1, n_epochs=n, val_loss=val[best_i][0],
                test_acc2=test[best_i][0], test_f1=test[best_i][1])


def collect():
    """variant -> {seed: result}"""
    out = {}
    for log in sorted(LOGS.glob("M[0-9]_s*.log")):
        m = re.match(r"(M\d)_s(\d+)\.log", log.name)
        if not m:
            continue
        # 아직 학습 중인 로그는 건너뜀 (프로세스 살아있으면 skip)
        res = parse_log(log)
        if res is None:
            continue
        out.setdefault(m.group(1), {})[int(m.group(2))] = res
    return out


def existing_variant_nseeds():
    seen = set()
    p = SD / "leaderboard.jsonl"
    if p.exists():
        for line in p.read_text().splitlines():
            try:
                r = json.loads(line)
                seen.add((r["method_id"], r.get("n_seeds")))
            except Exception:
                pass
    return seen


def main():
    data = collect()
    if not data:
        print("[none] 완료 로그 없음")
        return
    lb = AppendOnlyLog(SD / "leaderboard.jsonl", SCH / "experiment.schema.json")
    seen = existing_variant_nseeds()
    now = datetime.now(timezone.utc).isoformat()
    explore = json.loads(EXPLORE.read_text()) if EXPLORE.exists() else {}

    print(f"{'var':4} {'n':>2} {'mean':>8} {'std':>7} {'ref':>7}  seeds")
    for v in sorted(data):
        seeds = sorted(data[v])
        accs = [data[v][s]["test_acc2"] for s in seeds]
        f1s = [data[v][s]["test_f1"] for s in seeds]
        mean = statistics.mean(accs)
        std = statistics.pstdev(accs) if len(accs) > 1 else 0.0
        print(f"{v:4} {len(seeds):>2} {mean:8.4f} {std:7.4f} {REFERENCE[v]:7.4f}  {seeds} {[round(a,4) for a in accs]}")
        method_id = f"strip_{v.lower()}"
        if len(seeds) >= 3:
            if (method_id, len(seeds)) in seen:
                continue
            row = dict(
                exp_id=EXP, iteration=len(seeds),
                method=f"DLF_HTCMA_Stripped {VARIANT_DESC[v]}", method_id=method_id,
                config_fingerprint=FP[v], metric_name="mosei_acc2",
                constraints_passed=True, baseline_candidate=(v == "M0"), timestamp=now,
                notes=f"{v} {len(seeds)}-seed mean={mean:.4f}±{std:.4f} (ref {REFERENCE[v]:.4f}) seeds={seeds}",
                results=dict(mosei_acc2=mean, mosei_acc2_std=std, per_seed=accs,
                             per_seed_f1=f1s, seeds=seeds, reference_acc2=REFERENCE[v]),
                stage="prototype", n_seeds=len(seeds),
                gate_c_verdict="pending", gate_b_verdict="pass",
                linked_claims=[], manifest_ref=f"reproducibility_manifests/{EXP}.pip_freeze.txt",
                self_attack_run=False,
            )
            lb.append(row)
            print(f"     -> leaderboard appended ({len(seeds)} seeds)")
        else:
            explore[v] = dict(variant=v, desc=VARIANT_DESC[v], n_seeds=len(seeds),
                              seeds=seeds, test_acc2=accs, mean=mean,
                              reference_acc2=REFERENCE[v], note="single-seed exploratory; n<3 so NOT on leaderboard")
    EXPLORE.write_text(json.dumps(explore, indent=2, ensure_ascii=False))
    print(f"[explore] {sorted(explore)} → {EXPLORE.name}")


if __name__ == "__main__":
    main()
