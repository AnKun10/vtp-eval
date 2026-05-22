"""Shared base class for all visual-token-pruning adapters.

Inherits from lmms-eval's `Llava` adapter — overrides only generate_until to
inject the timing hook. Subclasses add pruning-specific config in their own
__init__ AFTER calling super().__init__.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from lmms_eval.models.simple.llava import Llava

from vtp_eval.utils.timing import TimingHook


class LlavaPruningBase(Llava):
    """Llava adapter + per-batch timing + pruning_meta plumbing.

    Timing strategy: wrap `self._model.generate` so TimingHook fires once per
    `model.generate(...)` call (= once per batch) rather than once per
    `generate_until()` (= once per whole eval run). This makes per-sample
    latency = latency_ms / batch_size meaningful for FLOPs/latency reporting.
    """

    def __init__(
        self,
        pretrained: str = "liuhaotian/llava-v1.5-7b",
        log_timing: bool = True,
        timing_sidecar: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(pretrained=pretrained, **kwargs)
        self.timing_hook: Optional[TimingHook] = (
            TimingHook() if log_timing else None
        )
        self.timing_sidecar_path = (
            Path(timing_sidecar) if timing_sidecar else None
        )
        self.pruning_meta: Dict = {}

        if self.timing_hook is not None:
            self._install_generate_hook()

    def _install_generate_hook(self) -> None:
        """Wrap `self._model.generate` so each call is timed via TimingHook.

        Reads batch size from the `inputs`/`input_ids` first dim (or `images`
        first dim for multimodal calls with no input_ids).
        """
        orig_generate = self._model.generate
        hook = self.timing_hook

        def _infer_batch_size(args, kwargs) -> int:
            for key in ("inputs", "input_ids", "images"):
                val = kwargs.get(key)
                if val is not None and hasattr(val, "shape") and val.dim() >= 1:
                    return int(val.shape[0])
            if args and hasattr(args[0], "shape") and args[0].dim() >= 1:
                return int(args[0].shape[0])
            return 1

        def timed_generate(*args, **kwargs):
            bs = _infer_batch_size(args, kwargs)
            with hook.measure(batch_size=bs):
                return orig_generate(*args, **kwargs)

        self._model.generate = timed_generate

    def generate_until(self, requests):
        out = super().generate_until(requests)
        # Dump sidecar after the whole eval (per-batch records already
        # captured by the wrapped generate).
        if self.timing_hook is not None and self.timing_sidecar_path is not None:
            self.timing_hook.pruning_meta = self.pruning_meta
            self.timing_hook.dump(self.timing_sidecar_path)
        return out
