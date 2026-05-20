#!/usr/bin/env bash
# Show candidate COCO images with their full annotation category labels.
# Use this BEFORE figure3_run.sh to decide --chosen-index and which words to track.
#
#     bash scripts/figure3_list_samples.sh
#
# Forwards any extra args to the underlying CLI (e.g. --min-cats 4 --num-candidates 12).
set -euo pipefail
python -m vtp_eval.figure3 --list-samples "$@"
