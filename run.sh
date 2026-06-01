#!/bin/bash
# Run the full pipeline. Source this or use: bash run.sh [args]
# Sets PYTHONPATH so all modules resolve correctly from dev/.

export PYTHONPATH="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$(dirname "$0")/.venv/bin/python"

"$PYTHON" "$@"
