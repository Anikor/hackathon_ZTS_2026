#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-$HOME/.venvs/zts2026/bin/python}
L=$1
for f in train test; do
  [ -f data/level${L}_$f.csv ] || { [ -f ~/Downloads/level${L}_$f.csv ] && cp ~/Downloads/level${L}_$f.csv data/; }
  [ -f data/level${L}_$f.csv ] || { echo "missing data/level${L}_$f.csv"; exit 1; }
done
echo "== baseline (solve.py)"
$PY src/solve.py --level $L --data data --out submissions 2>&1 | grep -E "CV|wrote" || true
echo "== multi-model OOF"
$PY src/multi.py train --level $L --data data --out submissions_l${L}x
$PY src/multi.py blend --level $L --data data --out submissions_l${L}x | grep -E "chosen|tuned|raw"
if [ "$L" = 3 ]; then
  for w in "" "--l3-weights entropy=1"; do
    $PY src/solve.py --level 3 --data data --out submissions_l3x/sub --probs submissions_l3x $w 2>&1 | grep -E "wrote|loco_blend_auroc" || true
  done
fi
ls -la submissions submissions_l${L}x | grep -E "level${L}_"
