#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
exec .venv-colibry-linux/bin/python scripts/view_colibry_camera.py --load-sram "$@"
