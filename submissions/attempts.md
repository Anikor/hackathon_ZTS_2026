# Attempts log

| Time | Level | File | Settings | CV | Live score |
|---|---|---|---|---|---|
| 10:19 | 1 | level1_submission.csv | default, full, th=0.47 | 0.9562 | 0.9576 (1/4) |
| 10:30 | 1 | submissions_l1x/level1_big_et.csv | src/l1_plus.py --variants big_et (LGBM big + ExtraTrees 50/50, th=0.475) | 0.9576 | 0.9584 (2/4) |
| 10:47 | 1 | submissions_l1x/level1_blend.csv | src/l1_more.py + src/l1_blend.py (greedy: ExtraTrees v2 x2 + LGBM big, th=0.475) | 0.9579 | 0.9588 (3/4) BEST |
| 11:21 | 1 | submissions_l1x/level1_blend_v2.csv | l1_more.py (+et3,et4) + l1_blend.py greedy 9 picks (blend_v2.json), th=0.48 | 0.95815 | 0.9587 (4/4); best stays 0.9588 (3/4) |
| 13:04 | 2 | l2/level2_submission.csv | API upload | | 0.5966 |
| 13:11 | 2 | l2/submissions_l2x__level2_blend.csv | API upload | | 0.6048 |
| 14:08 | 2 | l2/combo_3_lgb2_lgb3_et_rf.csv | API upload | | 0.5996 |
| 14:24 | 2 | l2/combo_stack2_lgb2_et.csv | API upload | | 0.6117 |
| 14:32 | 2 | l2/combo_stack11x2_stackx2_lgb2_et.csv | API upload | | 0.6085 |
| 15:09 | 3 | l3/level3_submission.csv | API upload | | 0.7583 |
| 15:22 | 3 | l3/stack_refine_level3.csv | API upload | | 0.834 |
| 15:09 | 3 | submissions/level3_submission.csv | solve.py baseline (single LGBM, rank blend entropy/knn/adversarial) | known F1 0.6857, LOCO AUROC 0.8845 | 0.7583 |
| 15:22 | 3 | submissions_l3s/sub/level3_submission_refine.csv | stack(lgb,et,rf,xgbg,catg)x2 + lgb x2 + et, --refine top 4% x1 | known F1 0.6958 | 0.8340 |
| 15:55 | 3 | submissions_l3s/sub/level3_submission_refine8x1.csv | same, --refine-frac 0.08 | | 0.8385 BEST |
| 15:59 | 3 | submissions_l3s/sub/level3_submission_refine4x2.csv | same, --refine-frac 0.04 --refine-iters 2 | | 0.8275 |
| 16:01 | 3 | submissions_l3s/sub/level3_submission_refine12x1.csv | --refine-frac 0.12 | | 0.8393 |
| 16:03 | 3 | submissions_l3s/sub/level3_submission_refine16x1.csv | --refine-frac 0.16 | | 0.8393 (plateau) |
| 16:12 | 3 | submissions_l3s/sub/level3_submission_eight.csv | src/l3_eight.py: 8-class LGBM with pseudo-Unknown (2x650 test rows), 30% of unknown_score | | 0.8394 BEST |
| 17:35 | 3 | submissions_l3s/sub/level3_submission_ens.csv | src/l3_ensemble.py: rank mean of refine8/12/16 + eight, cutoff 657 | | 0.8393 (best stays 0.8394) |
