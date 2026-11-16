#!/bin/zsh
set -e
cd -- "${0:A:h}/../.."
exec .venv/rana-training/bin/python src/live_blink/app.py
