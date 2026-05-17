"""CLI wrapper that imports vtp-eval adapters before launching lmms-eval.

Why this exists:
- lmms-eval (v0.5 and HEAD) resolves model names via the hard-coded
  `lmms_eval.models.AVAILABLE_SIMPLE_MODELS` dict, not via the legacy
  `@register_model` decorator (which only populates an unused `MODEL_REGISTRY`).
- The `--include_path` CLI flag doesn't run our adapter package's __init__,
  so the `AVAILABLE_SIMPLE_MODELS` mutation in `vtp_eval/adapters/__init__.py`
  never fires when lmms-eval is invoked directly.
- This wrapper imports the adapters package first, which mutates the dict,
  then hands off to lmms-eval's CLI.

Usage:
    python -m vtp_eval.run_lmms --model llava_baseline --tasks pope ...
"""
import vtp_eval.adapters  # noqa: F401  — side effect: register adapters

from lmms_eval.__main__ import cli_evaluate

if __name__ == "__main__":
    cli_evaluate()
