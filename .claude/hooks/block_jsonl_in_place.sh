#!/usr/bin/env bash
# block_jsonl_in_place.sh
# PreToolUse hook. Fires on Edit/Write tool calls.
# If the target file is *.jsonl and the operation is not pure-append, block.
#
# The hook receives JSON on stdin via tool_input.
# Exit 0 = allow. Exit 2 = block (Claude sees the message). Other = error.

set -euo pipefail

input=$(cat)

# Quick deny: any tool call touching a *.jsonl path with Edit
file_path=$(echo "$input" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('tool_input',{}).get('file_path','') or d.get('tool_input',{}).get('path',''))" 2>/dev/null || echo "")

case "$file_path" in
  *.jsonl)
    cat >&2 <<EOF
BLOCKED: in-place edit of $file_path is forbidden.
v2 invariant: jsonl files are append-only.
Corrections must be written as a NEW row with \`supersedes\` set to the bad row_id.
See METHODOLOGY §0.3 and permission_policy.action_class=in_place_jsonl_edit (hard_block).
EOF
    exit 2
    ;;
esac

# Allow anything else
exit 0
