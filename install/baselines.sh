# install/baselines.sh
#!/usr/bin/env bash
# Extra setup for the published-baseline adapters (run AFTER install/proposed.sh).
# Adds VisionZip to the environment (its visionzip package monkeypatches standard
# LLaVA). DivPrune + FastV are pure ports needing no extra install.
set -euo pipefail
PARENT="${PARENT:-$(cd "$(dirname "$0")/../.." && pwd)}"   # .../DATN
VZ="$PARENT/VisionZip"
if [ -d "$VZ" ]; then
    SP=$(python -c "import site; print(site.getsitepackages()[0])")
    echo "$VZ" > "$SP/visionzip_repo.pth"
    python -c "import visionzip; print('[install/baselines] visionzip importable from', visionzip.__file__)" \
        || echo "[install/baselines] WARNING: visionzip still not importable"
    echo "[install/baselines] VisionZip wired via .pth ($VZ -> $SP)"
else
    echo "[install/baselines] VisionZip repo not found at $VZ — clone it there."
fi
