#!/usr/bin/env bash
# GPU pod only. From this repo directory, after the GPU venv exists.
set -euo pipefail
export AMMONIA_USE_GPU=1
python gpu_simulator.py doctor
python main.py bench
python main.py all
