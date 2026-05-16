"""SparseVILA adapter — encoder-side query-agnostic prune + decode-time
query-aware retrieval.

Reference: SparseVILA (https://github.com/AnKun10/sparsevila-implementation).
Cannot inherit Llava.__init__ because SparseVILA loads the model via its own
load_sparse_llava() and generates via sparse_generate_packed(), not
model.generate(). We bypass Llava.__init__ and reimplement the minimal lmms
contract: __init__ + generate_until.
"""
from __future__ import annotations

import torch
from pathlib import Path
from typing import Optional

from lmms_eval.api.registry import register_model
from lmms_eval.models.simple.llava import Llava  # noqa: F401 (sibling reference)
from lmms_eval.api.model import lmms

from vtp_eval.utils.timing import TimingHook


@register_model("llava_sparsevila")
class LlavaSparseVILA(lmms):
    """Skip Llava.__init__ — load via sparsevila API directly."""

    def __init__(
        self,
        pretrained: str = "liuhaotian/llava-v1.5-7b",
        encoder_prune_ratio: float = 0.5,
        decode_retrieval_ratio: float = 0.5,
        use_flash_kernel: bool = True,
        quantize_llm: str = "none",
        device: str = "cuda",
        dtype: str = "float16",
        batch_size: int = 1,
        log_timing: bool = True,
        timing_sidecar: Optional[str] = None,
        **kw,
    ):
        super().__init__()  # lmms.__init__ (bypasses Llava)

        from sparsevila import load_sparse_llava, SparseVILAConfig
        from transformers import AutoTokenizer

        torch_dtype = {"float16": torch.float16,
                       "bfloat16": torch.bfloat16,
                       "float32": torch.float32}[dtype]

        cfg = SparseVILAConfig(
            encoder_prune_ratio=float(encoder_prune_ratio),
            decode_retrieval_ratio=float(decode_retrieval_ratio),
            use_flash_kernel=bool(use_flash_kernel),
            quantize_llm=str(quantize_llm),
        )
        self._model, self._image_processor = load_sparse_llava(
            pretrained, config=cfg, dtype=torch_dtype, device=device,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(pretrained, use_fast=False)
        self._sparse_config = cfg
        self._device = device
        self._pretrained = pretrained

        # lmms-eval contract attributes
        self.batch_size_per_gpu = int(batch_size)

        # Timing
        self.timing_hook: Optional[TimingHook] = (
            TimingHook() if log_timing else None
        )
        self.timing_sidecar_path = (
            Path(timing_sidecar) if timing_sidecar else None
        )

        self.pruning_meta = {
            "method": "sparsevila",
            "enc_ratio": float(encoder_prune_ratio),
            "dec_ratio": float(decode_retrieval_ratio),
        }

    # --- lmms required interface ----------------------------------------

    def loglikelihood(self, requests):
        raise NotImplementedError("SparseVILA only supports generate_until")

    def generate_until_multi_round(self, requests):
        raise NotImplementedError("Multi-round not supported in this phase")

    def generate_until(self, requests):
        """Run sparse_generate_packed for each request."""
        from sparsevila import sparse_generate_packed
        from llava.conversation import conv_templates
        from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
        from llava.mm_utils import tokenizer_image_token, process_images

        from contextlib import nullcontext
        results = []
        ctx_mgr = (self.timing_hook.measure()
                   if self.timing_hook is not None else nullcontext())
        with ctx_mgr:
            for r in requests:
                ctx, gen_kw, doc_to_visual, doc_id, task, split = r.args
                images = doc_to_visual(self.task_dict[task][split][doc_id])
                image_tensor = process_images(
                    images, self._image_processor, self._model.config,
                )[0].unsqueeze(0).to(self._device, dtype=torch.float16)

                # Build LLaVA-1.5 conv prompt
                conv = conv_templates["llava_v1"].copy()
                conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + ctx)
                conv.append_message(conv.roles[1], None)
                input_ids = tokenizer_image_token(
                    conv.get_prompt(), self._tokenizer, IMAGE_TOKEN_INDEX,
                    return_tensors="pt",
                ).unsqueeze(0).to(self._device)
                attn_mask = torch.ones_like(input_ids)

                out_ids = sparse_generate_packed(
                    self._model, self._tokenizer, input_ids,
                    image_tensor, attn_mask,
                    decode_retrieval_ratio=self._sparse_config.decode_retrieval_ratio,
                    max_new_tokens=int(gen_kw.get("max_new_tokens", 16)),
                )
                text = self._tokenizer.batch_decode(
                    out_ids, skip_special_tokens=True,
                )[0]
                # Strip prompt prefix if echoed
                if conv.sep in text:
                    text = text.split(conv.sep)[-1].strip()
                results.append(text)

        if self.timing_hook is not None and self.timing_sidecar_path is not None:
            self.timing_hook.pruning_meta = self.pruning_meta
            self.timing_hook.dump(self.timing_sidecar_path)
        return results
