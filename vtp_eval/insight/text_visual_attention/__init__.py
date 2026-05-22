"""Reproduction of Figure 3 of LearnPruner (ICLR 2026).

Sub-modules:
    coco       - download COCO val2017 annotations, select candidate images.
    tokens     - locate target word positions (single + multi-token) in a prompt.
    attention  - load LLaVA-1.5-7B, run forward, extract per-word attention.
    visualize  - sink-masked, percentile-clipped heatmap overlay + bar charts.
    metrics    - entropy / max-share / top-5% mass per (word, layer).

CLI:
    python -m vtp_eval.insight.text_visual_attention --list-samples
    python -m vtp_eval.insight.text_visual_attention --chosen-index N --words "person" "sports ball"
"""
