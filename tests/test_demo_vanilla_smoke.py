# tests/test_demo_vanilla_smoke.py
import pytest

pytestmark = pytest.mark.slow  # CUDA + LLaVA weights + transformers 4.37.2 (Vast.ai)


def test_vanilla_and_proposed_isolated_and_restored():
    from PIL import Image
    from llava.model.multimodal_encoder.clip_encoder import CLIPVisionTower
    from vtp_eval.demo.engine import load_engine

    engine = load_engine()                     # includes per-mode warmup
    img = Image.new("RGB", (336, 336), (90, 140, 110))

    # Vanilla path produces an answer and a positive latency.
    v_ans, v_lat = engine.vanilla_generate(img, "What is in this image?", history=[])
    assert v_ans, "vanilla produced empty output"
    assert v_lat > 0

    # After a vanilla turn the patched forward is restored (identity check).
    assert CLIPVisionTower.forward is not engine.originals["vision"], \
        "vanilla mode left the vision tower un-patched"

    # And the proposed turn still prunes (Stage-2 fires) after vanilla ran.
    res = engine.run_turn(img, "What is in this image?", history=[])
    assert res.answer
    assert res.tokens == (576, 384, 128)
    assert engine.model.model._proposed_last_prune is not None, \
        "Stage-2 prune did not fire after a vanilla turn (patches not restored)"
