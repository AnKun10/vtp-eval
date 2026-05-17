"""lmms-eval adapter modules for visual token pruning methods.

Two registration steps fire on import:
  1. Each submodule's @register_model decorator populates the legacy
     MODEL_REGISTRY dict (still used by some test code).
  2. We mutate `lmms_eval.models.AVAILABLE_SIMPLE_MODELS` so that
     `lmms_eval.models.get_model(name)` resolves our adapter by import path.

Step 2 is the one that actually makes the adapters callable from the CLI.
"""
from vtp_eval.adapters import llava_baseline   # noqa: F401  (registers adapter)
from vtp_eval.adapters import llava_fastv      # noqa: F401
from vtp_eval.adapters import llava_sparsevlm  # noqa: F401
from vtp_eval.adapters import llava_visionzip  # noqa: F401
from vtp_eval.adapters import llava_divprune   # noqa: F401
from vtp_eval.adapters import llava_sparsevila # noqa: F401

# Wire adapters into lmms-eval's hardcoded simple-model registry.
# Required because @register_model populates a different dict that
# lmms_eval.models.get_model() doesn't consult.
import lmms_eval.models as _lm

_lm.AVAILABLE_SIMPLE_MODELS.update({
    "llava_baseline":   "vtp_eval.adapters.llava_baseline.LlavaBaseline",
    "llava_fastv":      "vtp_eval.adapters.llava_fastv.LlavaFastV",
    "llava_sparsevlm":  "vtp_eval.adapters.llava_sparsevlm.LlavaSparseVLM",
    "llava_visionzip":  "vtp_eval.adapters.llava_visionzip.LlavaVisionZip",
    "llava_divprune":   "vtp_eval.adapters.llava_divprune.LlavaDivPrune",
    "llava_sparsevila": "vtp_eval.adapters.llava_sparsevila.LlavaSparseVILA",
})
