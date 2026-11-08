#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python3}
[ -f data_mock/level3_train.csv ] || $PY src/make_mock.py
for L in 1 2 3; do
  $PY src/solve.py --level $L --data data_mock --out submissions_mock --mock-labels data_mock/level3_test_labels.csv 2>&1 | grep -v "fold "
done
$PY src/solve.py --level 3 --data data_mock --out submissions_mock --l3-weights entropy=1 --mock-labels data_mock/level3_test_labels.csv | tail -2
for f in submissions_mock/level*_submission*.csv; do
  L=$(basename "$f" | cut -c6); $PY src/score.py "$L" "$f" "data_mock/level${L}_test_labels.csv"
done
$PY src/build_report.py submissions_mock report/raport_tehnic_preview.pdf --preview
