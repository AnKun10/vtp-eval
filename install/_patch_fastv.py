"""Copy FastV's patched modeling_llama.py into the active transformers install.

Idempotent: writes a .py.orig backup once, then overwrites every call.
"""
import shutil
import sys
from pathlib import Path

import transformers


def main() -> int:
    tf_dir = Path(transformers.__file__).parent
    target = tf_dir / "models" / "llama" / "modeling_llama.py"
    fastv_src = Path("/content/FastV/src/FastV/inference/transformers_replace"
                     "/models/llama/modeling_llama.py")
    if not fastv_src.exists():
        print(f"ERROR: FastV source not found at {fastv_src}", file=sys.stderr)
        return 1
    backup = target.with_suffix(".py.orig")
    if not backup.exists():
        backup.write_text(target.read_text())
        print(f"Backup written to {backup}")
    shutil.copy(fastv_src, target)
    print(f"FastV patch applied to {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
