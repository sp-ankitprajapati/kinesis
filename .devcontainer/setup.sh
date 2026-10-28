#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
KINESIS_VENV=${KINESIS_VENV:-.venv}
python3 -m venv "$KINESIS_VENV"
"$KINESIS_VENV/bin/python" -m pip install -r requirements-cpu.txt --index-url https://download.pytorch.org/whl/cpu
"$KINESIS_VENV/bin/python" -m pip install -r requirements.txt
"$KINESIS_VENV/bin/python" -m pip check
