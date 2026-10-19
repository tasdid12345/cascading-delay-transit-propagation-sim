"""Simulated bus route: does a LightGBM model with upstream-stop and leader-vehicle features
predict downstream arrival delays better than a per-vehicle linear travel-time regression?
Seed 42 everywhere; LightGBM is also run with seeds 42, 43, 44."""
import json
import os
import sys

import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, f1_score
from sklearn.model_selection import GroupKFold

SEED = 42
MODEL_SEEDS = [42, 43, 44]
N_STOPS = 12          # stops on the route
N_OBS = 3             # stops 0..2 are observed in real time; stops 3..11 are predicted
N_DAYS = 60
TEST_FRAC = 0.2       # last 20% of days held out for testing
LATE_MIN = 3.0        # late-arrival flag threshold (minutes)
MAX_ROWS = 200_000    # runtime cap (subsample if exceeded)
OUT = "results.json"

rng = np.random.default_rng(SEED)
SEG = rng.uniform(2, 4, size=N_STOPS - 1)            # scheduled segment run times (min)
OFF = np.concatenate([[0], np.cumsum(SEG)])           # scheduled offset of each stop from departure
DEPS = np.arange(6 * 60, 22 * 60, 3.0)                # 3-min headway, 06:00-22:00 each day


def simulate():
    """Simulate delays: propagation along the route, headway recovery from the leader, peaks, noise."""
    D, P, H, DEP, DAY = [], [], [], [], []
    for d in range(N_DAYS):
        dayeff = rng.normal(0, 0.6)                    # shared day-level shock
        prev = np.zeros(N_STOPS)                       # leader's delays at each stop
        prev2 = np.zeros(N_STOPS)                      # leader's leader's delays
        for dep in DEPS:
            peak = 0.8 if (420 <= dep <= 540 or 960 <= dep <= 1140) else 0.0  # AM/PM peaks
            delay = np.zeros(N_STOPS)
            delay[0] = 0.5 * dayeff + 0.5 * peak + rng.normal(0, 1.5) - 0.2 * prev[0]
            for k in range(1, N_STOPS):
                over = max(0.0, delay[k - 1] - 3)     # congestion nonlinearity above 3 min
                delay[k] = (0.85 * delay[k - 1] + 0.1 * over + 0.1 * dayeff
                            + peak * SEG[k - 1] / 3 * 0.3 + rng.normal(0, 0.8)
                            - 0.2 * prev[k])           # headway recovery from leader
            delay = np.clip(delay, -5, 30)
            D.append(delay)
            P.append(prev.copy())                      # leader delays
            H.append(3 + prev - prev2)                 # leader's headway at each stop
            DEP.append(dep)
            DAY.append(d)
            prev2, prev = prev, delay
    return np.array(D), np.array(P), np.array(H), np.array(DEP), np.array(DAY)


def build_table(D, P, H, DEP, DAY):
    """One row per (trip, downstream stop). Features only use observed stops 0..N_OBS-1."""
    frames = []
    for k in range(N_OBS, N_STOPS):
        f = pd.DataFrame({"day": DAY, "k": k, "sched_k": DEP + OFF[k], "dep": DEP,
                          "last_obs": D[:, N_OBS - 1], "y": D[:, k]})
        for j in range(N_OBS):
            f[f"d{j}"] = D[:, j]        # own upstream delays
            f[f"ld{j}"] = P[:, j]       # leader delays
            f[f"lh{j}"] = H[:, j]       # leader headways
        frames.append(f)
    return pd.concat(frames, ignore_index=True)


def evaluate(y, pred):
    """Shared evaluation: MAE (main), RMSE, and F1 for the late-arrival flag."""
    return {"mae": mean_absolute_error(y, pred),
            "rmse": float(np.sqrt(mean_squared_error(y, pred))),
            "f1": f1_score(y > LATE_MIN, pred > LATE_MIN)}


def lgb_model(params, seed):
    return lgb.LGBMRegressor(learning_rate=0.05, subsample=0.8, subsample_freq=1,
                             colsample_bytree=0.9, random_state=seed, verbose=-1, **params)


print("Simulating bus route...")
df = build_table(*simulate())
print(f"Rows: {len(df)}")
if len(df) > MAX_ROWS:
    df = df.sample(MAX_ROWS, random_state=SEED)

# Time-ordered split by trip date: last 20% of days are the test set
test_start = N_DAYS - int(round(TEST_FRAC * N_DAYS))
train, test = df[df.day < test_start].copy(), df[df.day >= test_start].copy()
print(f"Train rows: {len(train)}  Test rows: {len(test)}")

BASE = ["last_obs", "sched_k"]                                  # baseline features
IMP = ["k", "sched_k", "dep"] + [f"{p}{j}" for p in ("d", "ld", "lh") for j in range(N_OBS)]

# Baseline: per-vehicle linear regression (no tuning)
base = LinearRegression().fit(train[BASE], train.y)
rb = evaluate(test.y, base.predict(test[BASE]))
print(f"Baseline test MAE: {rb['mae']:.4f}")

# Improved: small grid tuned by group CV (grouped by day) on training days only
GRID = [dict(num_leaves=nl, n_estimators=ne) for nl in (15, 63) for ne in (200, 500)]
gkf = GroupKFold(n_splits=3)
best, best_cv = None, np.inf
for p in GRID:
    maes = []
    for tr_i, va_i in gkf.split(train, groups=train.day):
        m = lgb_model(p, SEED).fit(train.iloc[tr_i][IMP], train.iloc[tr_i].y)
        maes.append(mean_absolute_error(train.iloc[va_i].y, m.predict(train.iloc[va_i][IMP])))
    print(f"CV {p}: MAE={np.mean(maes):.4f}")
    if np.mean(maes) < best_cv:
        best, best_cv = p, np.mean(maes)
print(f"Selected params: {best}")

# Final improved model: test set used once, for each model seed
res = [evaluate(test.y, lgb_model(best, s).fit(train[IMP], train.y).predict(test[IMP]))
       for s in MODEL_SEEDS]
imp = {m: float(np.mean([r[m] for r in res])) for m in res[0]}
imp_std = {m: float(np.std([r[m] for r in res])) for m in res[0]}
print(f"Improved test MAE (mean over seeds): {imp['mae']:.4f} +/- {imp_std['mae']:.4f}")

if not (np.isfinite(imp["mae"]) and np.isfinite(rb["mae"]) and len(test) > 0):
    print("Could not produce meaningful metrics")
    sys.exit(1)

os.makedirs("figures", exist_ok=True)
plt.figure(figsize=(5, 4))
plt.bar(["baseline (linear)", "improved (LightGBM)"], [rb["mae"], imp["mae"]], color=["grey", "steelblue"])
plt.ylabel("Test MAE of downstream arrival delay (min)")
plt.title("Baseline vs improved")
plt.tight_layout()
plt.savefig("figures/mae_comparison.png", dpi=120)
plt.close()

results = {
    "question": "Does a gradient-boosting model using upstream stop delays and the leading vehicle's delay "
                "predict downstream transit delays better than a standard per-vehicle travel-time regression "
                "on a simulated bus line?",
    "method": "Numpy simulation of one bus route with delay propagation, headway recovery, peaks and noise; "
              "time-ordered train/test split by day; baseline linear regression vs grouped-CV-tuned LightGBM.",
    "dataset": {"name": "Simulated single bus route (numpy, generated in analysis.py)",
                "source_url": "generated in analysis.py (no download)",
                "n_rows": int(len(df)), "n_train": int(len(train)), "n_test": int(len(test)),
                "same_as_paper": False, "baseline_is_papers_method": False},
    "metric_name": "MAE of downstream arrival delay (minutes)",
    "higher_is_better": False,
    "baseline": "Per-vehicle linear regression of downstream arrival delay on the vehicle's own last observed "
                "delay and scheduled time at the target stop.",
    "improved": "LightGBM regressor (CV-tuned on training days) adding delays at all upstream observed stops, "
                "leader delays and headways at observed stops, stop index and time of day.",
    "metrics": {
        "baseline_score": float(rb["mae"]), "improved_score": imp["mae"],
        "improvement": float(rb["mae"] - imp["mae"]),
        "baseline_score_std": 0.0, "improved_score_std": imp_std["mae"],
        "baseline_rmse": rb["rmse"], "improved_rmse": imp["rmse"],
        "baseline_f1": float(rb["f1"]), "improved_f1": imp["f1"],
        "cv_mae_best": float(best_cv), "best_num_leaves": best["num_leaves"],
        "best_n_estimators": best["n_estimators"],
    },
    "notes": "Simulated data, so results reflect the simulator's assumptions, not a real transit agency. "
             "Leader delays are used only at observed stops. Baseline is deterministic; improved is averaged "
             "over LightGBM seeds 42-44. Late flag = delay above 3 minutes. Single simulated dataset (seed 42).",
}
with open(OUT, "w", encoding="utf-8") as fh:
    json.dump(results, fh, indent=2)
print(f"Wrote {OUT}")
