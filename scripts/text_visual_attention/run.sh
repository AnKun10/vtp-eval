#!/usr/bin/env bash
# Run the Figure 3 reproduction on candidate #<idx> tracking the given words.
#
# Usage:
#     bash scripts/text_visual_attention/run.sh <chosen-index> <word> [word...]
#
# Examples:
#     bash scripts/text_visual_attention/run.sh 0 person bicycle
#     bash scripts/text_visual_attention/run.sh 3 person "sports ball"
set -euo pipefail

IDX=${1:?usage: $0 <chosen-index> <word> [word...]}
shift
if [ $# -lt 1 ]; then
    echo "usage: $0 <chosen-index> <word> [word...]" >&2
    exit 1
fi

python -m vtp_eval.insight.text_visual_attention --chosen-index "$IDX" --words "$@"
