#!/usr/bin/env bash
# install_v2.sh — apply v2 layer to claude-research-engine
#
# Idempotent. Safe to re-run.
# 1. Backs up v1 README.md, CLAUDE.md to .v1_backup/ (if not already)
# 2. Lays down engine/, .claude/, install/, docs/ from this package
# 3. Patches CLAUDE.md with v2 prelude (or replaces)
# 4. Makes hooks executable
# 5. Installs Python deps (jsonschema)
# 6. Runs schema validator

set -euo pipefail

# Resolve self location. The package lives wherever install_v2.sh sits.
# Convention: this script and the engine/.claude/docs/ subdirs are in a `v2_pkg/`
# folder dropped at the repo root. We install INTO the repo root (parent of v2_pkg/).
PKG_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$PKG_DIR/.." && pwd)"
cd "$ROOT"

echo "════════════════════════════════════════════════════════════════"
echo "  claude-research-engine — v2 install"
echo "  package: $PKG_DIR"
echo "  repo   : $ROOT"
echo "════════════════════════════════════════════════════════════════"
echo

# Sanity: confirm we're at a git repo (warn but don't abort)
if ! git rev-parse --git-dir >/dev/null 2>&1; then
  echo "⚠ not a git repo. continuing anyway."
fi

# ─── 1. Backup v1 ──────────────────────────────────────────
mkdir -p .v1_backup
for f in README.md CLAUDE.md; do
  if [[ -f "$f" && ! -f ".v1_backup/$f" ]]; then
    cp "$f" ".v1_backup/$f"
    echo "  ✓ backed up $f → .v1_backup/$f"
  fi
done

# ─── 2. Lay down engine/, .claude/, docs/ ──────────────────
echo
echo "→ laying down v2 files…"

# engine/
mkdir -p engine/core engine/cli engine/schemas engine/tests/fixtures engine/agents
cp -n $PKG_DIR/engine/__init__.py        engine/__init__.py        2>/dev/null || true
cp -n $PKG_DIR/engine/core/__init__.py   engine/core/__init__.py   2>/dev/null || true
cp -n $PKG_DIR/engine/cli/__init__.py    engine/cli/__init__.py    2>/dev/null || true
cp    $PKG_DIR/engine/core/hashing.py             engine/core/hashing.py
cp    $PKG_DIR/engine/core/append_only_logger.py  engine/core/append_only_logger.py
cp    $PKG_DIR/engine/core/permission_policy.py   engine/core/permission_policy.py
cp    $PKG_DIR/engine/core/state_machine.py       engine/core/state_machine.py
cp    $PKG_DIR/engine/core/compute_budget.py      engine/core/compute_budget.py
cp    $PKG_DIR/engine/core/reproducibility_manifest.py engine/core/reproducibility_manifest.py
cp    $PKG_DIR/engine/core/leakage_auditor.py     engine/core/leakage_auditor.py
cp    $PKG_DIR/engine/cli/jy.py                   engine/cli/jy.py
cp -r $PKG_DIR/engine/schemas/*                   engine/schemas/
cp    $PKG_DIR/engine/tests/validate_schemas.py   engine/tests/validate_schemas.py
cp    $PKG_DIR/engine/tests/test_tier2.py         engine/tests/test_tier2.py
cp -r $PKG_DIR/engine/tests/fixtures/*            engine/tests/fixtures/

# gates/
mkdir -p engine/gates
touch engine/gates/__init__.py
cp $PKG_DIR/engine/gates/gate_c.py engine/gates/gate_c.py

# agents/
mkdir -p engine/agents
cp $PKG_DIR/engine/agents/*.md engine/agents/

echo "  ✓ engine/ laid down (core + gates + agents)"

# .claude/
mkdir -p .claude/commands .claude/hooks
cp $PKG_DIR/.claude/settings.json               .claude/settings.json
cp -r $PKG_DIR/.claude/commands/*               .claude/commands/
cp $PKG_DIR/.claude/hooks/block_jsonl_in_place.sh .claude/hooks/
cp $PKG_DIR/.claude/hooks/require_human_gate.sh   .claude/hooks/
chmod +x .claude/hooks/*.sh
echo "  ✓ .claude/ laid down (hooks executable)"

# docs/
mkdir -p docs
cp $PKG_DIR/docs/CHANGELOG_v1_to_v2.md docs/CHANGELOG_v1_to_v2.md
echo "  ✓ docs/CHANGELOG_v1_to_v2.md laid down"

# Root files
cp $PKG_DIR/README.md README.md
echo "  ✓ README.md replaced (v1 in .v1_backup/)"

# CLAUDE.md handling: prepend v2 patch, preserve v1 content
if [[ -f .v1_backup/CLAUDE.md ]]; then
  {
    cat $PKG_DIR/docs/CLAUDE.md.v2_patch
    echo
    echo "# --- v1 CONTENT BELOW (preserved) ---"
    echo
    cat .v1_backup/CLAUDE.md
  } > CLAUDE.md
  echo "  ✓ CLAUDE.md = v2 patch + v1 content (preserved)"
else
  cp $PKG_DIR/docs/CLAUDE.md.v2_patch CLAUDE.md
  echo "  ✓ CLAUDE.md = v2 patch only (no v1 found)"
fi

# ─── 3. Deps ───────────────────────────────────────────────
echo
echo "→ installing python deps…"
if python3 -c "import jsonschema, numpy" 2>/dev/null; then
  echo "  ✓ jsonschema, numpy already installed"
else
  pip install --quiet jsonschema numpy || pip install --break-system-packages --quiet jsonschema numpy
  echo "  ✓ jsonschema + numpy installed"
fi
# scipy is needed only for Gate C paper_ready. Install on demand later.
python3 -c "import scipy" 2>/dev/null && echo "  ✓ scipy present (Gate C paper_ready ready)" || \
  echo "  ℹ scipy not installed — needed only for paper_ready Gate C. Install when you're ready to ship."

# ─── 4. Validate ───────────────────────────────────────────
echo
echo "→ running schema validator…"
if python3 engine/tests/validate_schemas.py; then
  echo
  echo "════════════════════════════════════════════════════════════════"
  echo "  ✅ v2 install complete"
  echo "════════════════════════════════════════════════════════════════"
  echo
  echo "  Next steps:"
  echo "    1. Bootstrap (or re-bootstrap) a project:"
  echo "       python -m engine.cli.jy bootstrap --project 02_emotion_agent --prefix EMA"
  echo
  echo "    2. Check status:"
  echo "       python -m engine.cli.jy status --project 02_emotion_agent"
  echo
  echo "    3. Start a Claude Code session:"
  echo "       claude"
  echo "       /status 02_emotion_agent"
  echo "       /validate 02_emotion_agent"
  echo
  echo "  Read:"
  echo "    docs/CHANGELOG_v1_to_v2.md   — what changed and why"
  echo "    README.md                    — current usage"
  echo "    engine/schemas/README.md     — schema invariants"
  echo
  echo "  v1 files preserved in .v1_backup/"
  echo
else
  echo
  echo "  ✗ schema validator FAILED"
  echo "  See output above. Fix the failing schema/fixture before proceeding."
  exit 1
fi
