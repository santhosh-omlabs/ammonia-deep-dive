#!/bin/bash
set -euo pipefail
cd /workspace/ammonia-deep-dive
# shellcheck disable=SC1091
source .venv/bin/activate
export AMMONIA_USE_GPU=1
if [[ -f all.log ]]; then
  mv -f all.log "all.log.$(date +%s)"
fi
nohup python -u main.py all > all.log 2>&1 < /dev/null &
echo "STARTED $!"
sleep 2
head -n 20 all.log || true
