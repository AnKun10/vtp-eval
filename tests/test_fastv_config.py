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
