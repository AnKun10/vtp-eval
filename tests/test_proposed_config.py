import pytest
from vtp_eval.proposed_method.config import ProposedConfig


def test_r1_is_dominant_plus_diversity():
    cfg = ProposedConfig(dominant_k=54, diversity_m=10, pruned_layer=12, llm_keep_r2=37)
    assert cfg.r1 == 64


def test_avg_tokens_fastv_convention():
    # layers 0..k at R1 (k+1 layers), layers k+1..L-1 at R2
    cfg = ProposedConfig(dominant_k=54, diversity_m=10, pruned_layer=12, llm_keep_r2=37)
    expected = ((12 + 1) * 64 + (32 - 12 - 1) * 37) / 32
    assert cfg.avg_tokens(32) == pytest.approx(expected)


def test_validate_accepts_defaults():
    ProposedConfig().validate(num_llm_layers=32)  # no raise


def test_validate_rejects_r2_gt_r1():
    cfg = ProposedConfig(dominant_k=10, diversity_m=0, pruned_layer=12, llm_keep_r2=11)
    with pytest.raises(ValueError, match="llm_keep_r2"):
        cfg.validate(num_llm_layers=32)


def test_validate_rejects_r1_over_patches():
    cfg = ProposedConfig(dominant_k=600, diversity_m=0, pruned_layer=12, llm_keep_r2=10)
    with pytest.raises(ValueError, match="576"):
        cfg.validate(num_llm_layers=32)


def test_validate_rejects_bad_layer():
    cfg = ProposedConfig(pruned_layer=99)
    with pytest.raises(ValueError, match="pruned_layer"):
        cfg.validate(num_llm_layers=32)


def test_from_args_merges_overrides_over_defaults():
    cfg = ProposedConfig.from_args(
        defaults={"dominant_k": 54, "diversity_m": 10, "pruned_layer": 12, "llm_keep_r2": 37},
        overrides={"pruned_layer": 8},
    )
    assert (cfg.dominant_k, cfg.pruned_layer) == (54, 8)
