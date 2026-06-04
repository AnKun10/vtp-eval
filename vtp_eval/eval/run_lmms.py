# vtp_eval/eval/run_lmms.py
"""CLI entry: register vtp-eval adapters into lmms-eval, then run its CLI.

lmms-eval resolves --model via lmms_eval.models.AVAILABLE_SIMPLE_MODELS (a dict
of name -> "module.Class"), NOT via @register_model. We import the adapter
modules (so the classes exist / @register_model fires for legacy paths) and
update that dict before handing off.

Usage:
    python -m vtp_eval.eval.run_lmms --model llava_proposed --tasks pope ...
"""
import vtp_eval.adapters.llava_baseline   # noqa: F401  (defines LlavaBaseline)
import vtp_eval.proposed_method.adapter   # noqa: F401  (defines LlavaProposed)
import vtp_eval.baselines.divprune.adapter    # noqa: F401
import vtp_eval.baselines.fastv.adapter        # noqa: F401
import vtp_eval.baselines.visionzip.adapter    # noqa: F401

import lmms_eval.models as _lm

_lm.AVAILABLE_SIMPLE_MODELS.update({
    "llava_baseline": "vtp_eval.adapters.llava_baseline.LlavaBaseline",
    "llava_proposed": "vtp_eval.proposed_method.adapter.LlavaProposed",
    "llava_divprune": "vtp_eval.baselines.divprune.adapter.LlavaDivPrune",
    "llava_fastv": "vtp_eval.baselines.fastv.adapter.LlavaFastV",
    "llava_visionzip": "vtp_eval.baselines.visionzip.adapter.LlavaVisionZip",
})

from lmms_eval.__main__ import cli_evaluate  # noqa: E402

if __name__ == "__main__":
    cli_evaluate()
