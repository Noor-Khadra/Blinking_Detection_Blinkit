#!/bin/zsh
set -e
cd -- "${0:A:h}/../.."
export MPLBACKEND=Agg
export MPLCONFIGDIR="${MPLCONFIGDIR:-$PWD/.venv/live-webcam/matplotlib}"
mkdir -p "$MPLCONFIGDIR"
exec .venv/live-webcam/bin/python -u src/acquisition/record_dataset.py "$@"
