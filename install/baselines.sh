# install/baselines.sh
#!/usr/bin/env bash
# Extra setup for the published-baseline adapters (run AFTER install/proposed.sh).
# Adds VisionZip to the environment (its visionzip package monkeypatches standard
# LLaVA). DivPrune + FastV are pure ports needing no extra install.
set -euo pipefail
PARENT="${PARENT:-$(cd "$(dirname "$0")/../.." && pwd)}"   # .../DATN
VZ="$PARENT/VisionZip"
if [ -d "$VZ" ]; then
    pip install -e "$VZ" --no-deps || \
        echo "[install/baselines] editable VisionZip failed; adapter falls back to sys.path"
    echo "[install/baselines] VisionZip wired ($VZ)"
else
    echo "[install/baselines] VisionZip repo not found at $VZ — clone it there."
fi
