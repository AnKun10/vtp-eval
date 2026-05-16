"""Shared base class for all visual-token-pruning adapters.

Inherits from lmms-eval's `Llava` adapter — overrides only generate_until to
inject the timing hook. Subclasses add pruning-specific config in their own
__init__ AFTER calling super().__init__.
"""
from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Optional

from lmms_eval.models.simple.llava import Llava

from vtp_eval.utils.timing import TimingHook


class LlavaPruningBase(Llava):
    """Llava adapter + per-batch timing + pruning_meta plumbing."""

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

    def generate_until(self, requests):
        ctx = (self.timing_hook.measure()
               if self.timing_hook is not None else nullcontext())
        with ctx:
            out = super().generate_until(requests)
        # Dump sidecar after each batch (cheap, ensures partial results survive crash)
        if self.timing_hook is not None and self.timing_sidecar_path is not None:
            self.timing_hook.pruning_meta = self.pruning_meta
            self.timing_hook.dump(self.timing_sidecar_path)
        return out
