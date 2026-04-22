#!/bin/bash
# Usage: bash scripts/init_project.sh NN_project_name
# Creates a new project skeleton under projects/.

set -e
if [ -z "$1" ]; then
  echo "Usage: $0 NN_project_name (e.g. 02_s_pace_multimodal)"
  exit 1
fi

NAME=$1
ROOT=$(dirname "$0")/..
DST="$ROOT/projects/$NAME"

if [ -d "$DST" ]; then
  echo "Error: $DST already exists"
  exit 1
fi

mkdir -p "$DST"/{rules,state,experiments,research_log/{sessions,experiments,references,claude_evaluation/templates},results,src,baseline}

# Copy templates
cp "$ROOT/.global/rules_template/target_metrics_template.md" "$DST/rules/target_metrics.md"
cp "$ROOT/.global/rules_template/constraints_template.md" "$DST/rules/constraints.md"
cp "$ROOT/.global/templates/"*.md "$DST/research_log/claude_evaluation/templates/"

# Initial state files
touch "$DST/state/leaderboard.jsonl"
touch "$DST/state/paper_tried.jsonl"
echo "# Insights — $NAME" > "$DST/state/insights.md"

# Project README skeleton
cat > "$DST/README.md" <<EOF
# $NAME

## Objective
(Q1 target venue, thesis, etc.)

## Status
- Phase: 0
- Best metric: —
- Iterations: 0

## Quick links
- Target metrics: \`rules/target_metrics.md\`
- Leaderboard: \`state/leaderboard.jsonl\`
- Research log: \`research_log/\`

## Direction ID prefix
Use \`<PROJECT_PREFIX>-D###\` (choose 3-letter abbreviation).
EOF

echo "✓ Created $DST"
echo "Next steps:"
echo "  1. Edit $DST/rules/target_metrics.md"
echo "  2. Edit $DST/rules/constraints.md"
echo "  3. Populate baseline/ or src/"
echo "  4. Start iteration 1"
