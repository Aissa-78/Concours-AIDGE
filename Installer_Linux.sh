#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
profile="${1:-app}"
flow_python="${FLOWSENSE_PYTHON:-python3}"
exec "$flow_python" scripts/install_platform.py --profile "$profile" "${@:2}"
