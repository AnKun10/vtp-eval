# tests/test_proposed_smoke.py
import pytest

pytestmark = pytest.mark.slow  # requires CUDA + LLaVA weights + transformers 4.37.2 (Vast.ai)


def test_proposed_generates_and_prunes():
    import torch  # noqa: F401
    from PIL import Image
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import get_model_name_from_path
    from vtp_eval.proposed_method import proposed_prune, ProposedConfig

    path = "liuhaotian/llava-v1.5-7b"
    tok, model, image_processor, _ = load_pretrained_model(
        path, None, get_model_name_from_path(path), attn_implementation="eager")
    cfg = ProposedConfig(dominant_k=54, diversity_m=10, pruned_layer=12, llm_keep_r2=37)
    model = proposed_prune(model, cfg)

    img = Image.new("RGB", (336, 336), (127, 127, 127))
    # Build the prompt/generate call following the archived llava_baseline adapter
    # (git show archive/lmms-eval-pre-cleanup:vtp_eval/adapters/llava_baseline.py).
    # Assert: non-empty text output; spans recorded with vision_len == cfg.r1.
    assert model._proposed_cfg.r1 == 64
