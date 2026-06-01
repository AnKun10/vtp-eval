# tests/test_proposed_smoke.py
import pytest

pytestmark = pytest.mark.slow  # requires CUDA + LLaVA weights + transformers 4.37.2 (Vast.ai)


def test_proposed_generates_and_prunes():
    """End-to-end: load LLaVA-1.5, apply proposed_prune, run one generation,
    and assert (a) non-empty output and (b) Stage-2 saw the Stage-1 R1 vision
    tokens (spans recorded with vision_len == cfg.r1).

    Vast.ai only. The exact llava generate API may need a minor tweak against
    the installed liuhaotian/llava-v1.5-7b build; adjust here if so.
    """
    import torch
    from PIL import Image
    from llava.constants import IMAGE_TOKEN_INDEX
    from llava.mm_utils import (get_model_name_from_path, process_images,
                                tokenizer_image_token)
    from llava.model.builder import load_pretrained_model
    from vtp_eval.proposed_method import proposed_prune, ProposedConfig

    path = "liuhaotian/llava-v1.5-7b"
    tok, model, image_processor, _ = load_pretrained_model(
        path, None, get_model_name_from_path(path), attn_implementation="eager")
    cfg = ProposedConfig(dominant_k=54, diversity_m=10, pruned_layer=12, llm_keep_r2=37)
    model = proposed_prune(model, cfg)
    assert model._proposed_cfg.r1 == 64

    img = Image.new("RGB", (336, 336), (127, 127, 127))
    prompt = "USER: <image>\nWhat is in this image? ASSISTANT:"
    input_ids = tokenizer_image_token(
        prompt, tok, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device)
    image_tensor = process_images([img], image_processor, model.config)[0]

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor.unsqueeze(0).half().to(model.device),
            max_new_tokens=16, do_sample=False)
    text = tok.decode(output_ids[0], skip_special_tokens=True)

    assert text.strip(), "model produced empty output"
    # Stage-2 span recorder fired during prefill with the Stage-1 token budget.
    assert model._proposed_spans is not None
    assert model._proposed_spans["vision_len"] == cfg.r1
