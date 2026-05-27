#!/usr/bin/env bash
# require_human_gate.sh
# PreToolUse hook for Bash tool calls matching destructive patterns.
# Resolves auto_mode <-> HUMAN_APPROVAL conflict (external review §3 problem 2).
#
# Exit 0 = allow. Exit 2 = block. JY must run the command manually OR explicitly say
# "confirmed, proceed with <command>" to grant consent for that specific action.

set -euo pipefail

input=$(cat)
cmd=$(echo "$input" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('tool_input',{}).get('command',''))" 2>/dev/null || echo "")

# Patterns that require JY's explicit consent in *this exact message*.
deny_patterns=(
  'git push'
  'git push --force'
  'git push -f'
  'git branch -D'
  'git reset --hard'
  'rm -rf'
  'rm -r '
  ' > /dev/null && rm'
)

for p in "${deny_patterns[@]}"; do
  if [[ "$cmd" == *"$p"* ]]; then
    cat >&2 <<EOF
BLOCKED: command "$cmd" maps to action_class with default_policy=human_gate_required.

To proceed, JY must either:
  (a) run this command manually in another terminal, or
  (b) reply to Claude with explicit consent for this exact command.

See engine/core/permission_policy.py and projects/<PROJECT>/state/permission_policy.jsonl.
EOF
    exit 2
  fi
done

exit 0
