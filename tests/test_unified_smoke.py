# tests/test_unified_smoke.py
import pytest

pytestmark = pytest.mark.slow  # CUDA + LLaVA weights + transformers 4.37.2 (Vast.ai)


def test_one_model_serves_demo_pruneviz_and_tva():
    from PIL import Image
    from vtp_eval.demo.engine import load_engine

    engine = load_engine()
    img = Image.new("RGB", (336, 336), (120, 130, 140))
    q = "What is in the image?"

    # 1. demo inference (patched model)
    res = engine.run_turn(img, q, history=[])
    assert res.answer and res.tokens == (576, 384, 128)

    # 2. prune_viz figures (original model, under vanilla mode)
    pv = engine.prune_viz_figures(img, q, R1=128, R2=64,
                                  dominant_k=96, diversity_m=32)
    assert pv["combined"].shape[0] == 128      # dominant_k + diversity_m
    assert pv["r2"].shape[0] == 64             # R2

    # 3. TVA attention ("image" appears verbatim in the query)
    tva = engine.tva_attention(img, q, ["image"], layers=(2, 12, 30))
    assert tva["pwl"]["image"]["shallow"].shape == (576,)
    assert tva["grid"] == 24 and len(tva["sinks"]) >= 1

    # After the insight calls, the demo turn still prunes (patches restored).
    # Reset the marker first so the assertion proves THIS turn re-pruned (not a
    # stale value left from warmup).
    engine.model.model._proposed_last_prune = None
    res2 = engine.run_turn(img, q, history=[])
    assert res2.answer
    assert engine.model.model._proposed_last_prune is not None
