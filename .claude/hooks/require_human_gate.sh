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

# ── jsonl mutation block ───────────────────────────────────────────────────
# External review §3 problem 5: Edit/Write hook only catches Edit/Write tool calls.
# A Bash one-liner can append to *.jsonl, bypassing the append-only invariant and
# the hash chain. Block any Bash route that writes to a *.jsonl file. Legitimate
# appends must go through `python -m engine.cli.jy log ...` (or AppendOnlyLog API).
#
# We use grep -E with anchored alternatives. The patterns intentionally match the
# tail-end of the file path (`.jsonl` followed by EOL or whitespace) to reduce
# false positives on paths that merely *contain* "jsonl" as a substring.
jsonl_patterns=(
  '(>>|>)[[:space:]]*[^[:space:]|;&]*\.jsonl([[:space:]]|$)'    # echo ... >> foo.jsonl   or  > foo.jsonl
  'tee[[:space:]]+(-[aA][[:space:]]+)?[^|;&]*\.jsonl([[:space:]]|$)'  # tee -a foo.jsonl
  'sed[[:space:]]+-[iI][^|;&]*\.jsonl([[:space:]]|$)'           # sed -i ... foo.jsonl
  'perl[[:space:]]+-pi[^|;&]*\.jsonl([[:space:]]|$)'            # perl -pi -e ... foo.jsonl
  "open\([^)]*\.jsonl[^)]*['\"]a['\"]"                          # python open("...jsonl", "a")
  "open\([^)]*\.jsonl[^)]*['\"]w['\"]"                          # python open("...jsonl", "w") — truncate
)

for p in "${jsonl_patterns[@]}"; do
  if echo "$cmd" | grep -qE "$p"; then
    cat >&2 <<EOF
BLOCKED: command "$cmd" attempts to mutate a *.jsonl file outside the engine.

v2 invariant: jsonl files are append-only AND must go through the engine
(engine.cli.jy log, or engine.core.append_only_logger.AppendOnlyLog.append).
Direct Bash mutation bypasses both schema validation and the hash chain.

Action: rewrite as a Python call using AppendOnlyLog. If you actually need to
inspect, use the Read tool — not a shell redirect.
EOF
    exit 2
  fi
done

exit 0
