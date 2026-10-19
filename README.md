# Cascading Delay Prediction on a Simulated Bus Route

Motivation: In "Modeling and Predicting the Cascading Effects of Delay in Transit Systems" (2019), delays at one stop propagate downstream and to following vehicles through headway recovery, so predicting a vehicle's future delays benefits from information about upstream stops and the vehicles ahead of it. This experiment checks whether a modern gradient-boosting model that uses that information beats a standard per-vehicle travel-time regression.

Question: Does a gradient-boosting model that uses upstream stop delays and the leading vehicle's delay predict downstream transit delays better than a standard per-vehicle travel-time regression on a simulated bus line?

Dataset: A numpy simulation of one bus route generated inside `analysis.py`; no download. It is not the paper's own dataset (same_as_paper = false).

Baseline vs improved model:
- Baseline: per-vehicle linear regression of downstream arrival delay on the vehicle's own last observed delay and scheduled time.
- Improved: LightGBM regressor, with hyperparameters chosen by grouped cross-validation on training days only. It adds delays at all upstream observed stops, the leading vehicle's delays and headways, stop index, and time of day.
- Split: time-ordered by trip date; the last days are held out as the test set. Main metric is MAE in minutes; RMSE and F1 for a late-arrival flag are secondary.

How to run:
```
pip install -r requirements.txt
python analysis.py
```

Results: results are written to `results.json` (and `figures/mae_comparison.png`).

## Results (latest run)

On a simulated single-bus-route dataset (172,800 rows, time-ordered day split), the per-vehicle linear regression baseline reached 1.51 minutes MAE for downstream arrival delay, while the LightGBM model using upstream stop and leader delays reached 1.30 minutes on the same held-out test set.

| Metric | Value |
|---|---|
| baseline_score | 1.507 |
| improved_score | 1.302 |
| improvement | 0.2054 |
| baseline_score_std | 0 |
| improved_score_std | 0.0008665 |
| baseline_rmse | 1.981 |
| improved_rmse | 1.693 |
| baseline_f1 | 0.2233 |
| improved_f1 | 0.4229 |
| cv_mae_best | 1.315 |
| best_num_leaves | 15 |
| best_n_estimators | 200 |

Run on 2026-10-19. Full output: [results.json](results.json) · [run.log](run.log)

Notes: Simulated data, so results reflect the simulator's assumptions, not a real transit agency. Leader delays are used only at observed stops. Baseline is deterministic; improved is averaged over LightGBM seeds 42-44. Late flag = delay above 3 minutes. Single simulated dataset (seed 42).

## Reference

This project is motivated by the following paper:

> A Oruganti, S Basak, F Sun, H Baroud (2019). *Modeling and Predicting the Cascading Effects of Delay in Transit Systems*. … Research Board 98th. https://par.nsf.gov/biblio/10117235

Paper author: Prof. Abhishek Dubey, Vanderbilt University (abhishek.dubey@vanderbilt.edu)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
