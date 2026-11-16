#!/bin/zsh
set -e
cd -- "${0:A:h}/../.."
PYTHON="$PWD/.venv/rana-training/bin/python"
TORCH_LIB="$($PYTHON -c 'import importlib.util; from pathlib import Path; print(Path(importlib.util.find_spec("torch").origin).parent / "lib")')"
export DYLD_LIBRARY_PATH="$TORCH_LIB${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2
exec "$PYTHON" src/model/train_noor4_addition.py
