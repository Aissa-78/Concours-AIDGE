#!/bin/zsh
cd "${0:A:h}" || exit 1
kit=".local/colibry_examples/FlowSense_Camera_Mac"
if [[ -f "$kit/Demarrer_Camera_Mac.command" ]]; then
    export FLOWSENSE_CAMERA_PROJECT="$PWD"
    exec /bin/bash "$kit/Demarrer_Camera_Mac.command" --project "$PWD" "$@"
fi
exec ./.venv-colibry-hw/bin/python scripts/view_colibry_camera.py --load-sram "$@"
