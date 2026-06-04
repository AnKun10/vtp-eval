# tests/test_fastv_config.py
from vtp_eval.baselines.fastv.patch import FastVConfig
from vtp_eval.baselines.fastv.score import last_token_to_vision_scores


def test_fastvconfig_exposes_fields_prune_after_layer_k_reads():
    cfg = FastVConfig(pruned_layer=2, llm_keep_r2=30)
    assert cfg.r1 == 576                      # no Stage-1: full image tokens
    assert cfg.pruned_layer == 2
    assert cfg.llm_keep_r2 == 30
    assert cfg.score_fn is last_token_to_vision_scores
    assert cfg.stage2_enabled is True


from vtp_eval.baselines.budgets import fastv_knobs


def test_fastv_budget_round_trip_avg64_avg128():
    # fastv_knobs gives (K=full-layers, R); adapter maps pruned_layer = K-1.
    # The realized avg_tokens must round to the requested budget.
    for avg in (64, 128):
        k, r = fastv_knobs(avg)
        cfg = FastVConfig(pruned_layer=k - 1, llm_keep_r2=r)
        assert round(cfg.avg_tokens(32)) == avg


def test_fastv_avg32_unreachable():
    import pytest
    with pytest.raises(ValueError, match="floor"):
        fastv_knobs(32)
