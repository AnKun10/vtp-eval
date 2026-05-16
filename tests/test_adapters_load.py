"""Smoke tests: each adapter must register into lmms-eval and expose
a `pretrained` kwarg. Adapters are added to ADAPTERS as we implement them.
"""
import importlib
import inspect

import pytest

pytest.importorskip("lmms_eval", reason="lmms-eval not installed locally")

ADAPTERS = [
    "llava_baseline",
    "llava_fastv",
    # Added in later tasks:
    # "llava_sparsevlm", "llava_visionzip",
    # "llava_divprune", "llava_sparsevila",
]


@pytest.mark.parametrize("name", ADAPTERS)
def test_adapter_registered(name):
    """After importing vtp_eval.adapters, the adapter is in lmms-eval registry."""
    importlib.import_module("vtp_eval.adapters")
    from lmms_eval.api.registry import MODEL_REGISTRY
    assert name in MODEL_REGISTRY, (
        f"adapter {name!r} not in MODEL_REGISTRY. "
        f"Present: {sorted(MODEL_REGISTRY.keys())[:20]}..."
    )


@pytest.mark.parametrize("name", ADAPTERS)
def test_adapter_signature_has_pretrained(name):
    importlib.import_module("vtp_eval.adapters")
    from lmms_eval.api.registry import MODEL_REGISTRY
    cls = MODEL_REGISTRY[name]
    sig = inspect.signature(cls.__init__)
    assert "pretrained" in sig.parameters
