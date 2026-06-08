# tests/test_demo_record.py
import torch
from vtp_eval.proposed_method import stage1_vision, stage2_llm
from vtp_eval.demo import record


def test_recorders_capture_r1_and_r2_local():
    record.install_recorders()
    try:
        # R1: drive select_stage1 with random attn/hidden (B=1, P=20).
        B, H, P, D = 1, 4, 20, 16
        attn = torch.rand(B, H, 1 + P, 1 + P)
        hidden = torch.randn(B, 1 + P, D)
        keep = stage1_vision.select_stage1(attn, hidden, dominant_k=5, diversity_m=3)
        r1 = record.last_r1_idx()
        assert r1 is not None and r1.shape == (8,)               # R1 = 5 + 3
        assert torch.equal(r1, keep[0].cpu())

        # R2: drive build_keep_index with fake scores over Nv=8 vision tokens.
        scores = torch.rand(1, 8)
        vision_start = torch.tensor([2])
        _ = stage2_llm.build_keep_index(scores, r2=3, vision_start=vision_start, seq_len=12)
        r2 = record.last_r2_local()
        assert r2 is not None and r2.shape == (3,)
        assert torch.equal(r2, scores.topk(3, dim=1).indices[0].cpu())
    finally:
        record.uninstall_recorders()


def test_uninstall_restores_originals():
    orig_s1 = stage1_vision.select_stage1
    orig_bk = stage2_llm.build_keep_index
    record.install_recorders()
    record.uninstall_recorders()
    assert stage1_vision.select_stage1 is orig_s1
    assert stage2_llm.build_keep_index is orig_bk


def test_reset_clears_buffer():
    record.install_recorders()
    try:
        attn = torch.rand(1, 2, 21, 21); hidden = torch.randn(1, 21, 8)
        stage1_vision.select_stage1(attn, hidden, 5, 3)
        record.reset()
        assert record.last_r1_idx() is None
        assert record.last_r2_local() is None
    finally:
        record.uninstall_recorders()
