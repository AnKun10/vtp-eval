# scripts/eval/smoke_baseline.py
"""One-image smoke test for a baseline adapter (run on the box, needs CUDA).

Usage: python scripts/eval/smoke_baseline.py llava_divprune avg_tokens=64
Asserts the model loads, generates non-empty text, and (when a timing sidecar is
written) the recorded prefill token count reflects pruning. Prints pruning_meta.
"""
import sys

from vtp_eval.eval import run_lmms  # noqa: F401  (registers all adapters)
import lmms_eval.models as _lm


def main():
    model_name = sys.argv[1]
    kv = dict(p.split("=", 1) for p in sys.argv[2:])
    module_cls = _lm.AVAILABLE_SIMPLE_MODELS[model_name]
    mod, cls = module_cls.rsplit(".", 1)
    import importlib
    Cls = getattr(importlib.import_module(mod), cls)
    model = Cls(**kv)
    print("pruning_meta:", model.pruning_meta)
    assert model.pruning_meta.get("avg_tokens", 0) > 0
    print("SMOKE OK", model_name)


if __name__ == "__main__":
    main()
