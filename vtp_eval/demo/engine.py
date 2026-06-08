# vtp_eval/demo/engine.py
"""Demo inference engine: one LLaVA model patched with the proposed method,
wrapped with the retain-token cache + index recorders. One turn = one generate().
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import torch
from PIL import Image

from vtp_eval.proposed_method import ProposedConfig, proposed_prune
from vtp_eval.demo import record
from vtp_eval.demo.cache import RetainTokenCache, image_hash, install_cache
from vtp_eval.demo.overlay import overlay_pil, r2_local_to_patches

NUM_PATCHES = 576  # LLaVA-1.5 CLIP ViT-L/14-336

# Demo defaults: R1 = 384 (div 50%), R2 = 128, prune at LLM layer 12.
DEMO_CONFIG = dict(dominant_k=192, diversity_m=192, pruned_layer=12, llm_keep_r2=128)


@dataclass
class TurnResult:
    answer: str
    latency_s: float
    cache_hit: bool
    tokens: tuple            # (576, R1, R2)
    overlay_r1: Image.Image
    overlay_r2: Image.Image | None


class Engine:
    def __init__(self, tok, model, image_processor, cfg: ProposedConfig,
                 cache: RetainTokenCache):
        self.tok = tok
        self.model = model
        self.image_processor = image_processor
        self.cfg = cfg
        self.cache = cache

    # --- prompt building -------------------------------------------------
    def _build_prompt(self, question: str, history: list) -> str:
        """Single-image-per-turn prompt: prior turns folded in as text context so
        the <image> token (and thus vision merge + R2) is present every turn."""
        from llava.constants import DEFAULT_IMAGE_TOKEN
        from llava.conversation import conv_templates
        ctx = "".join(f"Previous question: {u}\nPrevious answer: {a}\n"
                      for u, a in history)
        conv = conv_templates["vicuna_v1"].copy()
        conv.append_message(conv.roles[0],
                            f"{DEFAULT_IMAGE_TOKEN}\n{ctx}{question}")
        conv.append_message(conv.roles[1], None)
        return conv.get_prompt()

    def _preprocess(self, image: Image.Image) -> torch.Tensor:
        from llava.mm_utils import process_images
        return process_images([image.convert("RGB")], self.image_processor,
                              self.model.config)[0]   # [3, 336, 336]

    # --- one turn --------------------------------------------------------
    @torch.inference_mode()
    def run_turn(self, image: Image.Image, question: str, history: list,
                 use_cache: bool = True, max_new_tokens: int = 128) -> TurnResult:
        from llava.constants import IMAGE_TOKEN_INDEX
        from llava.mm_utils import tokenizer_image_token

        self.cache.enabled = use_cache
        record.reset()

        image_tensor = self._preprocess(image)
        prompt = self._build_prompt(question, history)
        input_ids = tokenizer_image_token(
            prompt, self.tok, IMAGE_TOKEN_INDEX,
            return_tensors="pt").unsqueeze(0).to(self.model.device)
        images = image_tensor.unsqueeze(0).half().to(self.model.device)
        key = image_hash(images)   # hash the exact tensor encode_images receives

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out_ids = self.model.generate(
            input_ids, images=images, image_sizes=[image.size],
            do_sample=False, max_new_tokens=max_new_tokens, use_cache=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latency = time.perf_counter() - t0

        answer = self.tok.decode(out_ids[0], skip_special_tokens=True).strip()
        hit = self.cache.last_was_hit

        # R1 indices: from the cache entry on a hit; from the recorder on a miss
        # (then attach to the entry for future hits).
        entry = self.cache.get(key)
        if not hit and entry is not None and entry["r1_idx"] is None:
            self.cache.attach_r1(key, record.last_r1_idx())
            entry = self.cache.get(key)
        r1_idx = entry["r1_idx"] if (entry and entry["r1_idx"] is not None) \
            else record.last_r1_idx()

        overlay_r1 = overlay_pil(image, r1_idx)
        overlay_r2 = None
        r2_local = record.last_r2_local()
        if r2_local is not None and r1_idx is not None:
            overlay_r2 = overlay_pil(image, r2_local_to_patches(r1_idx, r2_local))

        r2_count = self.cfg.llm_keep_r2 if self.cfg.stage2_enabled else self.cfg.r1
        return TurnResult(answer, latency, hit,
                          (NUM_PATCHES, self.cfg.r1, r2_count),
                          overlay_r1, overlay_r2)

    @torch.inference_mode()
    def compare_latency(self, image: Image.Image, question: str, history: list,
                        max_new_tokens: int = 128) -> float:
        """Re-run the same turn with the cache disabled; return latency only.
        Does not mutate the cache or the chat."""
        res = self.run_turn(image, question, history, use_cache=False,
                            max_new_tokens=max_new_tokens)
        return res.latency_s


def load_engine(model_path: str = "liuhaotian/llava-v1.5-7b",
                stage2_enabled: bool = True, **cfg_overrides) -> Engine:
    """Load LLaVA (eager attention), apply proposed_prune, install cache +
    recorders. Vast.ai only (needs CUDA + weights + transformers 4.37.2)."""
    from llava.mm_utils import get_model_name_from_path
    from llava.model.builder import load_pretrained_model

    tok, model, image_processor, _ = load_pretrained_model(
        model_path, None, get_model_name_from_path(model_path),
        attn_implementation="eager", device_map="cuda")

    params = {**DEMO_CONFIG, **cfg_overrides, "stage2_enabled": stage2_enabled}
    cfg = ProposedConfig(**params)
    cfg.validate(num_llm_layers=model.config.num_hidden_layers)
    model = proposed_prune(model, cfg)

    record.install_recorders()
    cache = RetainTokenCache(maxsize=4)
    install_cache(model, cache)
    return Engine(tok, model, image_processor, cfg, cache)
