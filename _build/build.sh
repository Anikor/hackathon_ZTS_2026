#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python3}
export SUB="${1:-../submissions_mock}"
rm -rf figuri && "$PY" figures.py "$SUB" figuri
"$PY" report.py "${@:2}"
