#!/usr/bin/env bash
# Show candidate COCO images with their full annotation category labels.
# Use this BEFORE text_visual_attention/run.sh to decide --chosen-index and which words to track.
#
#     bash scripts/text_visual_attention/list_samples.sh
#
# Forwards any extra args to the underlying CLI (e.g. --min-cats 4 --num-candidates 12).
set -euo pipefail
python -m vtp_eval.insight.text_visual_attention --list-samples "$@"
