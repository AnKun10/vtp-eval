# vtp_eval/adapters/_base.py
"""Shared base for pruning adapters: lmms-eval Llava + 3-stage timing plumbing.

Subclasses call super().__init__ first, then set self.pruning_meta and (for the
proposed method) apply their patching. The stage timer is installed lazily on
the first generate() call, so it wraps the FINAL (post-patch) forwards.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from lmms_eval.models.simple.llava import Llava

from vtp_eval.eval.stage_timer import StageTimer


class LlavaPruningBase(Llava):
    def __init__(self, pretrained: str = "liuhaotian/llava-v1.5-7b",
                 log_timing: bool = True, timing_sidecar: Optional[str] = None,
                 **kwargs) -> None:
        super().__init__(pretrained=pretrained, **kwargs)
        self.timer: Optional[StageTimer] = StageTimer() if log_timing else None
        self.timing_sidecar_path = Path(timing_sidecar) if timing_sidecar else None
        self.pruning_meta: Dict = {}
        if self.timer is not None:
            self._install_generate_hook()

    def _install_generate_hook(self) -> None:
        orig_generate = self._model.generate
        timer = self.timer
        model = self._model

        def _infer_bs(args, kwargs) -> int:
            for key in ("inputs", "input_ids", "images"):
                val = kwargs.get(key)
                if val is not None and hasattr(val, "shape") and val.dim() >= 1:
                    return int(val.shape[0])
            if args and hasattr(args[0], "shape") and args[0].dim() >= 1:
                return int(args[0].shape[0])
            return 1

        def timed_generate(*args, **kwargs):
            timer.install(model)            # lazy: forwards are final by now
            timer.start_batch(_infer_bs(args, kwargs))
            try:
                return orig_generate(*args, **kwargs)
            finally:
                timer.end_batch()

        self._model.generate = timed_generate

    def generate_until(self, requests):
        out = super().generate_until(requests)
        if self.timer is not None and self.timing_sidecar_path is not None:
            self.timer.dump(self.timing_sidecar_path, self.pruning_meta)
        return out
