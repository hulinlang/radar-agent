#!/usr/bin/env bash
# Inspect where llama-cli was stuck before the SIGKILL cascade in /tmp/strace.log
set -u
L=/tmp/strace.log
echo "--- total lines ---"
wc -l "$L"
echo "--- first SIGKILL line number ---"
N=$(grep -n 'killed by SIGKILL' "$L" | head -1 | cut -d: -f1)
echo "N=$N"
echo "--- 40 lines before the kill cascade ---"
head -n $((N-1)) "$L" | tail -40
echo "--- main thread (lowest pid) activity sample ---"
MAINPID=$(head -1 "$L" | awk '{print $1}')
echo "first-line pid field: $MAINPID"
grep -E '^(9[0-9]|1[0-4][0-9]) ' "$L" | grep -vE 'futex|killed' | tail -25
