"""Visualize the proposed two-stage token pruning (Exp 2 & 3).

Exp 2: original | attention-only R1 | diversity-only R1.
Exp 3: original | after-R1 prune | after-R2 prune.

CLI:
    python -m vtp_eval.insight.prune_viz --list-samples
    python -m vtp_eval.insight.prune_viz --dataset gqa --index 2 --mode exp2 --r1 64
"""
