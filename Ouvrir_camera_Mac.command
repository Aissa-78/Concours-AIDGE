#!/bin/zsh
cd "${0:A:h}" || exit 1
./.venv-colibry-hw/bin/python scripts/view_colibry_camera.py --load-sram
