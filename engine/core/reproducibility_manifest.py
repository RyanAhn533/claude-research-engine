"""
reproducibility_manifest.py — generate the v2 manifest before each run.py.

The manifest is the v2 hard rule: missing manifest blocks LOGGING (METHODOLOGY §0.4).
This module collects everything mechanically so Claude doesn't forget.

Usage:
    from engine.core.reproducibility_manifest import generate

    mf = generate(
        exp_id="exp_039_normwear_resnet_ablation",
        random_seeds={"python": 42, "numpy": 42, "torch": 42},
        env_lock_path="env.lock",
        data_hash="sha256:b91a...",   # caller computes; data shape varies per project
    )
    mf.write_yaml(path="projects/02_emotion_agent/reproducibility_manifests/exp_039.yaml")
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _run(cmd: list[str], cwd: Path | None = None) -> str:
    try:
        return subprocess.check_output(cmd, cwd=cwd, stderr=subprocess.DEVNULL).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def _git_sha(cwd: Path | None = None) -> str:
    return _run(["git", "rev-parse", "HEAD"], cwd=cwd)


def _git_dirty(cwd: Path | None = None) -> bool:
    out = _run(["git", "status", "--porcelain"], cwd=cwd)
    return bool(out)


def _detect_gpu() -> str:
    # Try nvidia-smi; fall back to "unknown".
    out = _run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"])
    if out:
        # e.g. "NVIDIA RTX A6000, 49140 MiB"
        return out.split("\n")[0].strip()
    return "unknown"


def _detect_cuda() -> str:
    out = _run(["nvcc", "--version"])
    for line in out.splitlines():
        if "release" in line.lower():
            return line.strip()
    return _run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]) or "unknown"


def _file_hash(path: Path) -> str:
    if not path.exists():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return f"sha256:{h.hexdigest()}"


@dataclass
class Manifest:
    exp_id: str
    created_at: str
    git_sha: str
    git_dirty: bool
    data_hash: str
    env_lock: str
    env_lock_hash: str
    random_state: dict[str, Any]
    hardware: dict[str, str]
    container_digest: str | None = None
    row_id: str | None = None
    supersedes: str | None = None
    prev_hash: str | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        # drop row_id/supersedes/prev_hash if None to keep manifest concise
        return {k: v for k, v in d.items() if v is not None or k in {"container_digest"}}

    def write_yaml(self, path: str | Path) -> Path:
        # Minimal YAML emit. Avoids pyyaml dep.
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        lines = []
        for k, v in self.to_dict().items():
            if isinstance(v, dict):
                lines.append(f"{k}:")
                for kk, vv in v.items():
                    lines.append(f"  {kk}: {json.dumps(vv) if not isinstance(vv, str) else vv}")
            elif isinstance(v, bool):
                lines.append(f"{k}: {'true' if v else 'false'}")
            elif v is None:
                lines.append(f"{k}: null")
            else:
                lines.append(f"{k}: {v}")
        p.write_text("\n".join(lines) + "\n")
        return p

    def write_json(self, path: str | Path) -> Path:
        """For schema validation (against reproducibility_manifest.schema.json)."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=False))
        return p


def generate(
    *,
    exp_id: str,
    random_seeds: dict[str, int],
    env_lock_path: str | Path = "env.lock",
    data_hash: str = "",
    container_digest: str | None = None,
    repo_root: Path | None = None,
) -> Manifest:
    if not data_hash:
        raise ValueError(
            "data_hash is required. Compute it from the exact data slice used "
            "(train/val/test indices), not raw files."
        )
    if "python" not in random_seeds:
        raise ValueError("random_seeds must include 'python', 'numpy', 'torch' minimum")

    repo_root = repo_root or Path.cwd()
    env_lock_p = Path(env_lock_path)
    git_sha = _git_sha(repo_root)
    if not git_sha:
        raise ValueError(
            f"git_sha could not be resolved at {repo_root}. "
            "reproducibility_manifest requires a git repository; non-git contexts are "
            "not allowed (would silently produce schema-invalid manifests with "
            "git_sha='unknown'). Initialize the repo or pass a repo_root that is one."
        )
    return Manifest(
        exp_id=exp_id,
        created_at=datetime.now(timezone.utc).isoformat(),
        git_sha=git_sha,
        git_dirty=_git_dirty(repo_root),
        data_hash=data_hash,
        env_lock=str(env_lock_p),
        env_lock_hash=_file_hash(env_lock_p),
        random_state={**random_seeds, "cuda_deterministic": random_seeds.get("cuda_deterministic", True)},
        hardware={
            "gpu": _detect_gpu(),
            "driver": _run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]) or "unknown",
            "cuda": _detect_cuda(),
            "host": _run(["hostname"]) or "unknown",
        },
        container_digest=container_digest,
    )
