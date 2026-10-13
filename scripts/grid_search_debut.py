"""Debut strength forecaster: exhaustive search over feature groups x models x settings.

Rules (fixed before looking at results)
- Validation: each year 2019-2023 predicted from all earlier years. 2024-2026 untouched.
- Main metric: log loss on "beats the leader" (strength > 1), pooled over the five years.
  P(beat) = P(predicted log strength + normal error > 0); predicted log strength =
  predicted log day-one streams - predicted log streams of the day's best existing song.
  The error spread for year y comes from the same candidate's errors on year y-1
  (predicted from the years before it), so no future information is used.
- Tiebreak / pick: simplest candidate within one standard error of the best
  (standard error of the mean of the five yearly log losses).

Run from the repo root:  python scripts/grid_search_debut.py
"""
import itertools
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PATH = sys.argv[1] if len(sys.argv) > 1 else "data/release_features.csv"
OUT = sys.argv[2] if len(sys.argv) > 2 else "data/debut_grid_search.csv"

F = pd.read_csv(PATH, parse_dates=["first_chart_date"])
F = F[F.no1_d2.notna()].reset_index(drop=True)
F["beat"] = F.strength > 1
F["l_n_prev"] = np.log1p(F.n_prev)
F["l_days_since"] = np.log1p(F.days_since_last)
# popularity, split into "on the chart at all" and "how much"
for k in ["pop7_max", "pop28_max"]:
    F[f"on_{k}"] = (F[k] > 0).astype(int)
    F[f"lpos_{k}"] = np.where(F[k] > 0, np.log(F[k].clip(lower=1)), 0.0)

GROUPS = {
    "track_core": ["last_ls", "last_same_ls", "has_hist"],
    "track_extra": ["best3_ls", "best2y_ls", "best_ever_ls", "ewm_ls", "l_n_prev", "n_no1_debuts", "l_days_since"],
    "popularity": ["on_pop7_max", "lpos_pop7_max", "on_pop28_max", "lpos_pop28_max", "share_pop7_max"],
    "release": ["is_album", "n_credited"],
    "timing_market": ["friday", "l_mkt"],
}
STREAM_COLS = ["last_ls", "last_same_ls", "best3_ls", "best2y_ls", "best_ever_ls", "ewm_ls"]

MODELS = [("OLS", {}, 0)]
MODELS += [("Ridge", {"alpha": a}, 1) for a in [0.1, 1, 10, 100]]
MODELS += [("GBM", {"learning_rate": lr, "max_leaf_nodes": lv, "min_samples_leaf": ms}, 2)
           for lr in [0.03, 0.1] for lv in [7, 15, 31] for ms in [20, 60]]


def make(kind, p):
    if kind == "OLS":
        return LinearRegression()
    if kind == "Ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=p["alpha"]))
    return HistGradientBoostingRegressor(max_iter=300, l2_regularization=1.0, random_state=0, **p)


def prep(X, tr, kind):
    if kind == "GBM":
        return X
    X = X.copy()
    med = tr.lstreams.median()
    for c in X.columns:
        if c in STREAM_COLS:
            X[c] = X[c].fillna(med)
    if "l_days_since" in X:
        X["l_days_since"] = X["l_days_since"].fillna(tr.l_days_since.median())
    return X.fillna(0)


def predict(kind, p, feats, tr, va):
    m = make(kind, p).fit(prep(tr[feats], tr, kind), tr.lstreams)
    return m.predict(prep(va[feats], tr, kind))


def den_pred(tr, va):
    m = LinearRegression().fit(tr[["l_no1_d2", "friday"]], tr.l_best_existing)
    return m.predict(va[["l_no1_d2", "friday"]])


YEARS = list(range(2019, 2024))
splits = {y: (F[F.year < y], F[F.year == y]) for y in [2018] + YEARS}
dens = {y: den_pred(*splits[y]) for y in splits}


def evaluate(kind, p, feats):
    preds = {y: predict(kind, p, feats, *splits[y]) for y in splits}
    rows = []
    for y in YEARS:
        prev_tr, prev_va = splits[y - 1]
        sig = np.std(preds[y - 1] - dens[y - 1] - prev_va.lstrength.values)
        va = splits[y][1]
        pr = preds[y] - dens[y]
        pb = np.clip(norm.sf(0, loc=pr, scale=sig), 1e-4, 1 - 1e-4)
        b = va.beat.values
        rows.append(pd.DataFrame({"year": y, "beat": b, "pr": pr, "pb": pb,
                                  "e": np.abs(preds[y] - va.lstreams.values),
                                  "top": (va.top_song_rank <= 10).values}))
    P = pd.concat(rows)
    P["ll"] = -(P.beat * np.log(P.pb) + (~P.beat) * np.log(1 - P.pb))
    yearly = P.groupby("year").ll.mean()
    return {"logloss": P.ll.mean(), "logloss_se": yearly.std(ddof=1) / np.sqrt(len(yearly)),
            "worst_year_ll": yearly.max(), "auc": roc_auc_score(P.beat, P.pr),
            "err_all": np.exp(P.e.mean()), "err_top10": np.exp(P.e[P.top].mean())}


combos = [c for r in range(1, len(GROUPS) + 1) for c in itertools.combinations(GROUPS, r)]
print(f"{len(combos)} feature-group combinations x {len(MODELS)} model settings = "
      f"{len(combos) * len(MODELS)} candidates", flush=True)
res, t0 = [], time.time()
for i, combo in enumerate(combos, 1):
    feats = [f for g in combo for f in GROUPS[g]]
    for kind, p, cx in MODELS:
        r = evaluate(kind, p, feats)
        r.update(groups="+".join(combo), n_groups=len(combo), n_feats=len(feats), model=kind,
                 settings=str(p), complexity=cx)
        res.append(r)
    print(f"  {i}/{len(combos)} combos done ({time.time() - t0:.0f}s)", flush=True)

R = pd.DataFrame(res).sort_values("logloss").reset_index(drop=True)
R.to_csv(OUT, index=False)

best = R.iloc[0]
within = R[R.logloss <= best.logloss + best.logloss_se].copy()
leaves = within.settings.str.extract(r"'max_leaf_nodes': (\d+)")[0].astype(float).fillna(0)
within["leaves"] = leaves
pick = within.sort_values(["complexity", "n_groups", "leaves", "logloss"]).iloc[0]

pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", 70)
cols = ["model", "settings", "groups", "logloss", "logloss_se", "auc", "err_all", "err_top10"]
print("\nTop 15 by log loss")
print(R[cols].head(15).round(4).to_string(index=False))
print(f"\n{len(within)} candidates within one standard error of the best ({best.logloss:.4f} + {best.logloss_se:.4f})")
print("\nPicked (simplest within one SE):")
print(pick[cols].to_string())
print("\nBest of each model type")
print(R.groupby("model").head(1)[cols].round(4).to_string(index=False))
