# vtp_eval/demo/sections/__init__.py
"""Unified-UI tab builders (one tool per module), sharing one Engine.

OUT is where the Pruning/TVA tabs save their figure PNGs (served via
gr.Image(type="filepath")). Created lazily by the tabs, not at import time.
"""
from pathlib import Path

_ROOT = Path("/workspace") if Path("/workspace").is_dir() else Path.cwd()
OUT = _ROOT / "outputs/unified_ui"
