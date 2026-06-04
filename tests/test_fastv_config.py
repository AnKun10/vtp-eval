# tests/test_fastv_config.py
from vtp_eval.baselines.fastv.patch import FastVConfig
from vtp_eval.baselines.fastv.score import last_token_to_vision_scores


def test_fastvconfig_exposes_fields_prune_after_layer_k_reads():
    cfg = FastVConfig(pruned_layer=1, llm_keep_r2=64)
    assert cfg.r1 == 576                      # no Stage-1: full image tokens
    assert cfg.pruned_layer == 1
    assert cfg.llm_keep_r2 == 64
    assert cfg.score_fn is last_token_to_vision_scores
    assert cfg.stage2_enabled is True


def test_fastv_retain_true_avg():
    # FastV retains R tokens after K=2 full layers (pruned_layer = K-1 = 1).
    # true avg = ((pruned_layer+1)*576 + (32-pruned_layer-1)*R)/32 = (2*576 + 30*R)/32
    for retain, expected in [(32, 66), (64, 96), (128, 156)]:
        cfg = FastVConfig(pruned_layer=1, llm_keep_r2=retain)
        assert round(cfg.avg_tokens(32)) == expected
