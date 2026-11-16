#!/usr/bin/env bash
# Pass --vision_only_ckpt /path/to/user-trained/model.pth plus run_eval.py options.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
exec bash ./vgl_python.sh src/evaluation/run_eval.py --vision_only_baseline "$@"
