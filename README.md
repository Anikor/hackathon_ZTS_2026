# AI Cyber Defense Challenge (ZTS XIII, 8 Nov 2026): dry run

One script solves all three levels. It was rehearsed on public UNSW-NB15 data, reshaped into the contest format.
The stand-in data has the same 39 features under the guide's names, ~141.7k/38.65k rows, and
L3 train = 139,283 rows (the guide says ~139.3k).

## Layout
```
src/make_mock.py     UNSW-NB15 -> data_mock/level{N}_train/test(.csv) + hidden test_labels (practice only)
src/solve.py         --level 1|2|3 --data DIR --out DIR [--fast] [--l3-weights entropy=1] [--unknown-frac 0.017]
src/score.py         local copy of the official metrics
src/build_report.py  1-3 page PDF report from the *_cv.json files
dry_run.sh           the whole rehearsal in one command
_build/              UTM-style report: figures.py + report.py (reportkit, Lab1_AI_Braccio.docx template), build.sh
Raport_tehnic_ZTS2026_Eunoia.docx/.pdf/.pages          UTM-style report, 3 pages (dry-run preview)
report/raport_tehnic_preview.pdf                          compact 1-page report (fits the contest's 1-3 page rule)
downloads/           starter kit, guide page, UNSW-NB15 source CSVs
```

## Dry-run results (stand-in data, not the contest)
| Level | Out-of-fold validation | Score on hidden test | Runtime (M-series Mac) |
|---|---|---|---|
| 1 binary F1 | 0.960 | **0.962** | 35 s |
| 2 macro F1 (10) | 0.614 → 0.624 after class scaling | **0.630** | 2 min |
| 3 0.6·F1 + 0.4·AUROC | simulated AUROC 0.88 | **0.710** blend / **0.716** entropy only | ~4 min |

Baseline LogisticRegression from the starter kit is far lower; the gap is the point of the report.

## Contest-day plan
Files from /submit are named `level{N}_train.csv` / `level{N}_test.csv`. Put them in `data/`.
```
python -m pip install pandas numpy scikit-learn lightgbm scipy reportlab
python src/solve.py --level 1 --data data --out submissions        # 10:00 – 12:00, 4 attempts
python src/solve.py --level 2 --data data --out submissions        # 13:00 – 15:00, 6 attempts
python src/solve.py --level 3 --data data --out submissions        # 15:00 – 18:00, 8 attempts
python src/solve.py --level 3 --data data --out submissions --l3-weights entropy=1
python src/build_report.py submissions report/raport_tehnic.pdf    # then zip PDF + src/ (<15 MB) by 18:15
```
- **L1**: submit the default first. Use a spare attempt on a threshold nudge only if the score is far below CV.
- **L2**: default first. Scaling is tuned out-of-fold, so the leaderboard should land near CV.
- **L3**: the live AUROC is computed on the full test set, so it is a real A/B. Submit the blend, then `entropy=1`, then
  vary `--unknown-frac` (0.01 / 0.02) on the better one. Leave 2 min between submissions.
- If the machine is slow, `--fast` cuts trees from 1200 to 400. Most of L3's time is the simulation.
- Score = max per level. A format reject costs no attempt (`solve.py` checks the format before writing).

## UTM-style report
`_build/build.sh ../submissions` (Mac only: needs Pages + ~/Projects/utm-report-tools). `PREVIEW=0` drops the
dry-run note. The contest requires a 1–3 page PDF: a normal UTM cover page + 2 content pages, with no Cuprins and no
figure (`FIG=1` adds the LOCO figure, but then the report is 4 pages). Team "Eunoia": Drangoi Karina, Secrieru Maxim.
The prose has no ";".

## Notes
- Prediction labels never come from outside the level's own train.csv. UNSW-NB15 here is only for the rehearsal.
- The report has blanks for the team names and the AI-use sentence. Fill them in truthfully.
- The virtualenv lives outside this (synced) folder: `~/.venvs/zts2026` (`dry_run.sh` uses it; override with `PY=`).
- On macOS, LightGBM in that venv was pointed at sklearn's bundled libomp (`install_name_tool -add_rpath`).
  On Windows the pip wheel works as-is.
- Rules page vs starter-kit README disagree on attempts (4/6/8 vs "10 per level"). Plan for 4/6/8.
