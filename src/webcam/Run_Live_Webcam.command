#!/bin/zsh
set -e
cd -- "${0:A:h}/../.."
if [[ ! -x .venv/live-webcam/bin/python ]]; then
  print -u2 'Set up the live webcam environment first. See src/webcam/README.md.'
  exit 1
fi
export MPLBACKEND=Agg
export MPLCONFIGDIR="${MPLCONFIGDIR:-$PWD/.venv/live-webcam/matplotlib}"
mkdir -p "$MPLCONFIGDIR"
exec .venv/live-webcam/bin/python src/webcam/live_webcam.py "$@"
