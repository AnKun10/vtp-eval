# tests/test_demo_engine_smoke.py
import pytest

pytestmark = pytest.mark.slow  # CUDA + LLaVA weights + transformers 4.37.2 (Vast.ai)


def test_engine_two_turns_second_is_cache_hit():
    from PIL import Image
    from vtp_eval.demo.engine import load_engine

    engine = load_engine()
    img = Image.new("RGB", (336, 336), (120, 130, 140))

    r1 = engine.run_turn(img, "What colour is this image?", history=[])
    assert r1.answer, "turn 1 produced empty output"
    assert r1.cache_hit is False
    assert r1.tokens == (576, 384, 128)
    assert r1.overlay_r1.size == img.size
    assert r1.overlay_r2 is not None

    hist = [("What colour is this image?", r1.answer)]
    r2 = engine.run_turn(img, "Is it bright or dark?", history=hist)
    assert r2.answer
    assert r2.cache_hit is True, "turn 2 on same image must hit the cache"
    # R2 overlay is query-dependent: indices may differ from turn 1.
    assert r2.overlay_r2 is not None
