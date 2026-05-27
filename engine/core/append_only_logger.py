"""
append_only_logger.py — the spine of v2 enforcement.

Guarantees:
1. Every write is APPEND only. No row is ever modified.
2. Every row is schema-validated before write.
3. Every row carries prev_hash → integrity verifiable.
4. In-place edits (mtime/size mismatch vs last-recorded state) raise on next open.
5. Concurrent writers serialize via fcntl flock.

Corrections happen by writing a NEW row with `supersedes` set to the bad row_id.
Never edit. Never delete.

Usage:
    from engine.core.append_only_logger import AppendOnlyLog

    log = AppendOnlyLog(
        path="projects/02_emotion_agent/state/leaderboard.jsonl",
        schema="engine/schemas/experiment.schema.json",
    )
    log.append({"exp_id": "exp_039_...", ...})
    log.verify_chain()   # walk file and re-check every prev_hash
"""
from __future__ import annotations

import fcntl
import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from jsonschema import Draft7Validator

from .hashing import row_hash


class AppendOnlyViolation(RuntimeError):
    """Raised on in-place edit / hash chain break / schema reject."""


class AppendOnlyLog:
    def __init__(self, path: str | Path, schema: str | Path):
        self.path = Path(path)
        self.schema_path = Path(schema)
        self._schema = json.loads(self.schema_path.read_text())
        Draft7Validator.check_schema(self._schema)
        self._validator = Draft7Validator(self._schema)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._state_file = self.path.with_suffix(self.path.suffix + ".lockstate")

    # ------ Public ------

    def append(self, row: dict[str, Any]) -> dict[str, Any]:
        """Validate, chain, write atomically. Returns the written row (with row_id + prev_hash).

        Concurrency: the ENTIRE critical section (tamper-detect → last_row → prev_hash
        → schema validate → write → record_state) executes under exclusive flock.
        This prevents the race where two processes read the same last_row and write
        two rows with identical prev_hash.
        """
        # Open in append mode just to obtain a file descriptor for flock.
        # The file is created if missing (parent already mkdir'd in __init__).
        with self.path.open("a", encoding="utf-8") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                # All hash-chain-relevant reads MUST happen inside the lock.
                self._verify_not_tampered()

                prev = self._last_row()
                row = dict(row)
                row.setdefault("row_id", self._make_row_id())
                row["prev_hash"] = row_hash(prev) if prev is not None else None

                # schema check (still inside lock — reject must not leak partial state)
                errors = list(self._validator.iter_errors(row))
                if errors:
                    msg = "; ".join(f"{list(e.absolute_path)}:{e.message}" for e in errors[:5])
                    raise AppendOnlyViolation(f"schema reject for {self.path.name}: {msg}")

                line = json.dumps(row, sort_keys=False, ensure_ascii=False)
                f.write(line + "\n")
                f.flush()
                os.fsync(f.fileno())

                # record_state must also be inside lock — otherwise a concurrent
                # writer could append between our write and our state record,
                # and our state record would silently cover their bytes too.
                self._record_state()
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

        return row

    def iter_rows(self) -> Iterator[dict[str, Any]]:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as f:
            for n, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as e:
                    raise AppendOnlyViolation(
                        f"{self.path.name}:{n}: malformed JSON line"
                    ) from e

    def verify_chain(self) -> int:
        """Walk file end-to-end, re-checking schema and prev_hash. Returns row count."""
        prev = None
        count = 0
        for row in self.iter_rows():
            errs = list(self._validator.iter_errors(row))
            if errs:
                raise AppendOnlyViolation(
                    f"{self.path.name} row {row.get('row_id')}: schema reject"
                )
            expected = row_hash(prev) if prev is not None else None
            if row.get("prev_hash") != expected:
                raise AppendOnlyViolation(
                    f"{self.path.name} row {row.get('row_id')}: prev_hash mismatch"
                )
            prev = row
            count += 1
        return count

    # ------ Tamper detection ------

    def _verify_not_tampered(self) -> None:
        """If the file shrunk or an inner byte changed since last write → halt."""
        if not self.path.exists():
            return
        if not self._state_file.exists():
            # first run after a manual creation — record once and trust
            self._record_state()
            return
        recorded = json.loads(self._state_file.read_text())
        cur_size = self.path.stat().st_size
        if cur_size < recorded["size"]:
            raise AppendOnlyViolation(
                f"{self.path.name} SHRANK from {recorded['size']} → {cur_size}. "
                f"Append-only violation. Halt."
            )
        # bytes up to recorded_size must be unchanged
        with self.path.open("rb") as f:
            head = f.read(recorded["size"])
        import hashlib
        if hashlib.sha256(head).hexdigest() != recorded["head_sha256"]:
            raise AppendOnlyViolation(
                f"{self.path.name} HEAD BYTES MUTATED. In-place edit detected. Halt."
            )

    def _record_state(self) -> None:
        size = self.path.stat().st_size if self.path.exists() else 0
        import hashlib
        if size:
            with self.path.open("rb") as f:
                payload = f.read()
            sha = hashlib.sha256(payload).hexdigest()
        else:
            sha = hashlib.sha256(b"").hexdigest()
        self._state_file.write_text(json.dumps({"size": size, "head_sha256": sha}))

    # ------ Helpers ------

    def _last_row(self) -> dict[str, Any] | None:
        last = None
        for row in self.iter_rows():
            last = row
        return last

    def _make_row_id(self) -> str:
        stem = self.path.stem  # e.g. "leaderboard"
        return f"{stem}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:6]}"
