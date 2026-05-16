"""lmms-eval adapter modules for visual token pruning methods.

Each submodule registers a model adapter via @register_model decorator.
Importing this package registers all adapters in lmms-eval's MODEL_REGISTRY.
"""
from vtp_eval.adapters import llava_baseline   # noqa: F401  (registers adapter)
from vtp_eval.adapters import llava_fastv      # noqa: F401
from vtp_eval.adapters import llava_sparsevlm  # noqa: F401
from vtp_eval.adapters import llava_visionzip  # noqa: F401
from vtp_eval.adapters import llava_divprune   # noqa: F401
from vtp_eval.adapters import llava_sparsevila # noqa: F401
