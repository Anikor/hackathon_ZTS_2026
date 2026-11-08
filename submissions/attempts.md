# Attempts log

| Time | Level | File | Settings | CV | Live score |
|---|---|---|---|---|---|
| 10:19 | 1 | level1_submission.csv | default, full, th=0.47 | 0.9562 | 0.9576 (1/4) |
| 10:30 | 1 | submissions_l1x/level1_big_et.csv | src/l1_plus.py --variants big_et (LGBM big + ExtraTrees 50/50, th=0.475) | 0.9576 | 0.9584 (2/4) |
| 10:47 | 1 | submissions_l1x/level1_blend.csv | src/l1_more.py + src/l1_blend.py (greedy: ExtraTrees v2 x2 + LGBM big, th=0.475) | 0.9579 | 0.9588 (3/4) BEST |
| 11:21 | 1 | submissions_l1x/level1_blend_v2.csv | l1_more.py (+et3,et4) + l1_blend.py greedy 9 picks (blend_v2.json), th=0.48 | 0.95815 | 0.9587 (4/4); best stays 0.9588 (3/4) |
