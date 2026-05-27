#!/usr/bin/env bash
# block_jsonl_in_place.sh
# PreToolUse hook. Fires on Edit/Write/Bash tool calls.
#
# Catches:
#   - Edit / Write on *.jsonl paths
#   - Bash commands that append-via-shell OR in-place-edit *.jsonl
#     (>>, tee -a, sed -i, perl -pi, awk -i, python ...open("a")..., truncate)
#
# All jsonl writes must go through engine/core/append_only_logger.py
# (the only path that holds the flock and maintains the hash chain).
#
# Exit 0 = allow. Exit 2 = block (Claude sees the message). Other = error.

set -euo pipefail

input=$(cat)

# --- Pull both Edit/Write fields AND Bash command ---
tool_name=$(echo "$input" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('tool_name',''))" 2>/dev/null || echo "")
file_path=$(echo "$input" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('tool_input',{}).get('file_path','') or d.get('tool_input',{}).get('path',''))" 2>/dev/null || echo "")
bash_cmd=$(echo "$input"  | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('tool_input',{}).get('command',''))" 2>/dev/null || echo "")

block() {
  cat >&2 <<EOF
BLOCKED: jsonl write-via-bypass detected.
  reason : $1
  target : $2

v2 invariant: state/*.jsonl files are append-only and MUST flow through
engine/core/append_only_logger.py (which holds flock + chains prev_hash).

Allowed paths:
  python -m engine.cli.jy <cmd>            # bootstrap, validate, seed-policy
  python -m engine.core.append_only_logger # if you really must, but prefer CLI

Use Edit/Write tool on jsonl is also forbidden — same lock + chain reasoning.
Corrections happen as NEW rows with \`supersedes\` set.
See METHODOLOGY §0.3 / permission_policy.action_class=in_place_jsonl_edit (hard_block).
EOF
  exit 2
}

# --- Rule 1: Edit/Write on *.jsonl ---
case "$file_path" in
  *.jsonl) block "Edit/Write tool on jsonl" "$file_path" ;;
esac

# --- Rule 2: Bash command touching *.jsonl via bypass route ---
if [ -n "$bash_cmd" ]; then
  # In-place edit operators
  if echo "$bash_cmd" | grep -Eq 'sed[[:space:]]+(-[a-zA-Z]*i|--in-place)([[:space:]]|=)[^|]*\.jsonl(\b|$)'; then
    block "sed -i on jsonl" "$bash_cmd"
  fi
  if echo "$bash_cmd" | grep -Eq 'perl[[:space:]]+-[a-zA-Z]*i[[:space:]]+[^|]*\.jsonl(\b|$)'; then
    block "perl -pi on jsonl" "$bash_cmd"
  fi
  if echo "$bash_cmd" | grep -Eq 'awk[[:space:]]+-i[[:space:]]+inplace[^|]*\.jsonl(\b|$)'; then
    block "awk -i inplace on jsonl" "$bash_cmd"
  fi

  # Bash-level append redirections
  if echo "$bash_cmd" | grep -Eq '(>>|>\|)[[:space:]]*[^|]*\.jsonl(\b|$)'; then
    block "shell append (>>) to jsonl" "$bash_cmd"
  fi
  if echo "$bash_cmd" | grep -Eq 'tee[[:space:]]+-a[^|]*\.jsonl(\b|$)'; then
    block "tee -a on jsonl" "$bash_cmd"
  fi

  # Truncation / overwrite
  if echo "$bash_cmd" | grep -Eq '(^|[[:space:]])(>|truncate[[:space:]])[[:space:]]*[^|<]*\.jsonl(\b|$)'; then
    block "shell overwrite/truncate of jsonl" "$bash_cmd"
  fi

  # Python open(..., "a"|"w"|"r+"|"a+") on a jsonl literal
  if echo "$bash_cmd" | grep -Eq 'python[0-9]*[[:space:]]+-c.*open\(.*\.jsonl.*,[[:space:]]*["'\''](a|w|r\+|a\+)["'\'']'; then
    block "python open(..., a/w/r+) on jsonl literal" "$bash_cmd"
  fi
fi

# Allow anything else
exit 0
