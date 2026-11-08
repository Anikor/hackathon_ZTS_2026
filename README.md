# AI Cyber Defense Challenge (ZTS XIII, 8 Nov 2026), team Eunoia

Drangoi Karina, Secrieru Maxim. The technical report is `Raport_tehnic_ZTS2026_Eunoia.pdf`.
All submissions with their files, settings and scores are listed in `submissions/attempts.md`.

## Setup
```
python -m pip install -r requirements.txt
```
Put `level{N}_train.csv` and `level{N}_test.csv` (from the platform) in `data/`.

## Level 1 (best 95.88%)
```
python src/l1_plus.py --data data --out submissions_l1x --variants base,big
python src/l1_more.py --data data --out submissions_l1x --models et2,rf,hgb,cat,xgb
python src/l1_blend.py --data data --dir submissions_l1x      # greedy OOF blend -> level1_blend.csv
```

## Level 2 (best 61.17%)
```
python src/multi.py train  --level 2 --data data --out submissions_l2x --models lgb,et,rf,lgb2,lgb3,xgb,et2
python src/stack.py        --level 2 --data data --out submissions_l2x --models lgb,et,rf,lgb2,lgb3,xgb,et2
python src/multi.py combos --level 2 --data data --out submissions_l2x --combos "stack+stack+lgb+lgb+et"   # -> combo_0.csv
```

## Level 3 (best 83.94%)
```
python src/multi.py train  --level 3 --data data --out submissions_l3x --models lgb,et,rf,xgbg,catg   # xgbg/catg run on an NVIDIA GPU
python src/stack.py        --level 3 --data data --out submissions_l3x --models lgb,et,rf,xgbg,catg
python src/multi.py blend  --level 3 --data data --out submissions_l3x --fixed "stack+stack+lgb+lgb+et"
python src/solve.py --level 3 --data data --out submissions_l3x/sub --probs submissions_l3x --refine --refine-frac 0.12
python src/l3_eight.py --data data --probs submissions_l3x --sub submissions_l3x/sub/level3_submission_refine12x1.csv \
       --out submissions_l3x/sub/level3_submission_eight.csv
```
`src/check_l3.py` checks the CSV format before upload. `src/solve.py --level N` is the single-model baseline
for any level.

## Infrastructure
Training ran on a lab PC (28 cores, RTX 4070) over SSH. `zts.py` (status / run / fetch / submit) drives it and
talks to the platform API. The token is read from a local `.token` file, which is not part of the archive.
