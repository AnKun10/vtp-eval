import pytest
from vtp_eval.utils.tflops import (
    LLaMAConfig, LLAVA_15_7B, layer_flops, total_llm_flops,
    shape_baseline, shape_fastv, shape_sparsevlm,
    shape_visionzip, shape_divprune, shape_sparsevila,
    clip_encoder_flops, compute_method_tflops,
)


def test_llama_config_defaults_match_llava_15_7b():
    assert LLAVA_15_7B.n_layers == 32
    assert LLAVA_15_7B.d_model == 4096
    assert LLAVA_15_7B.d_ffn == 11008


def test_layer_flops_positive():
    assert layer_flops(622) > 0  # 35 sys + 576 visual + 10 text_q + 1 text_a


def test_total_llm_flops_baseline_in_expected_range():
    """Baseline LLaVA-1.5 7B prefill on POPE (~622 token sequence).

    attn_proj dominates at this seq_len: 32 * 129 GFLOPs ≈ 4.1 TFLOPs LLM-only.
    """
    total = total_llm_flops(shape_baseline())
    tflops = total / 1e12
    assert 3.0 < tflops < 12.0, f"baseline LLM tflops = {tflops:.2f}"


def test_shape_baseline_length():
    assert len(shape_baseline()) == 32
    assert all(v == 576 for v in shape_baseline())


def test_shape_fastv_K2_R128():
    shape = shape_fastv(K=2, R=128)
    assert len(shape) == 32
    assert shape[:3] == [576, 576, 576]
    assert shape[3:] == [128] * 29


def test_shape_sparsevlm_192_matches_source_schedule():
    """Reference: SparseVLMs/llava/model/language_model/score.py:11 comment.
       '2*576 4*300 10*200 16*110' for retain=192."""
    shape = shape_sparsevlm(192)
    assert len(shape) == 32
    assert shape[:2] == [576, 576]
    assert shape[2:6] == [300, 300, 300, 300]
    assert shape[6:16] == [200] * 10
    assert shape[16:] == [110] * 16


def test_shape_visionzip_pre_llm():
    shape = shape_visionzip(dominant=54, contextual=10)
    assert all(v == 64 for v in shape)


def test_shape_divprune_pre_llm():
    shape = shape_divprune(ratio=0.098)
    expected = int(576 * 0.098)
    assert all(v == expected for v in shape)


def test_shape_sparsevila_encoder_prune():
    shape = shape_sparsevila(enc_ratio=0.5, dec_ratio=0.5)
    expected = int(576 * 0.5)
    assert all(v == expected for v in shape)


def test_clip_encoder_flops_positive():
    assert clip_encoder_flops() > 0


def test_pruning_reduces_total_flops():
    base = compute_method_tflops("baseline")["tflops_total"]
    for method, kw in [
        ("fastv", dict(K=2, R=128)),
        ("sparsevlm", dict(retain=192)),
        ("visionzip", dict(dominant=54, contextual=10)),
        ("divprune", dict(ratio=0.098)),
        ("sparsevila", dict(enc_ratio=0.5, dec_ratio=0.5)),
    ]:
        r = compute_method_tflops(method, **kw)
        assert r["tflops_total"] < base, (
            f"{method} did not reduce flops: {r['tflops_total']} vs {base}"
        )


def test_visionzip_more_aggressive_than_fastv():
    vz = compute_method_tflops("visionzip", dominant=54, contextual=10)
    fv = compute_method_tflops("fastv", K=2, R=128)
    assert vz["tflops_total"] < fv["tflops_total"]


def test_compute_method_tflops_unknown_method_raises():
    with pytest.raises(KeyError):
        compute_method_tflops("not_a_method")
