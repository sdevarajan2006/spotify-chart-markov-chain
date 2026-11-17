"""Debut strength forecaster, v2 (the configuration picked by grid_search_debut.py).

Target: log day-one streams of a release's top song. Gradient boosting (learning rate 0.1,
7 leaves, min 60 per leaf) on 14 features known before release. Strength = predicted streams
over a forecast of the day-D leader (linear on the D-2 #1 streams plus a Friday flag).
P(beats leader) = normal tail of predicted log strength, spread from the previous year's residuals.

No look-ahead:
- 2018-2023 releases are predicted by models trained on earlier years only.
- 2024-2026 releases are predicted by the model trained on 2017-2023 (scored once).

Surprise drops (not announced by D-2, see check_announcements.py) are left out; --keep-surprise
restores them.

Run from the repo root after build_release_features.py and check_announcements.py:
    python scripts/debut_forecaster.py [--keep-surprise] [--features PATH] [--announcements PATH] [--outdir DIR]
Writes (to --outdir, default data/):
    debut_forecast_release_predictions.csv   per-release predictions, all years
    debut_forecast_test_metrics.csv          2024-2026 forecaster score and leader-forecast error
Other scripts import load_releases() and forecast_releases() (see part1_overlay.py).
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score

FEATS = (["last_ls", "last_same_ls", "has_hist"]
         + ["best3_ls", "best2y_ls", "best_ever_ls", "ewm_ls", "l_n_prev", "n_no1_debuts", "l_days_since"]
         + ["is_album", "n_credited"] + ["friday", "l_mkt"])
LEADER_FEATS = ["l_no1_d2", "friday"]


def gbm():
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.1, max_leaf_nodes=7,
                                         min_samples_leaf=60, l2_regularization=1.0, random_state=0)


def load_releases(feat="data/release_features.csv", ann="data/release_announcements.csv",
                  exclude_surprise=True):
    """Releases with a D-2 chart. Surprise drops are removed unless exclude_surprise is False."""
    R = pd.read_csv(feat, parse_dates=["first_chart_date"])
    R = R[R.no1_d2.notna()].reset_index(drop=True)
    R["known_ahead"] = 1.0
    if os.path.exists(ann):
        A = pd.read_csv(ann, parse_dates=["first_chart_date"])[["first_chart_date", "artist", "top_song", "known_ahead"]]
        R = R.drop(columns="known_ahead").merge(A, on=["first_chart_date", "artist", "top_song"], how="left")
        n_s = int((R.known_ahead == 0).sum())
        print(f"{n_s} surprise releases ({int(((R.known_ahead == 0) & (R.year >= 2024)).sum())} in 2024-2026), "
              f"{'left out' if exclude_surprise else 'kept'}")
        if exclude_surprise:
            R = R[R.known_ahead.fillna(1) == 1].reset_index(drop=True)
    R["l_n_prev"] = np.log1p(R.n_prev)
    R["l_days_since"] = np.log1p(R.days_since_last)
    return R


def _fit_predict(tr, va):
    """Predicted log streams and log strength (streams minus the forecast leader)."""
    s = gbm().fit(tr[FEATS], tr.lstreams).predict(va[FEATS])
    den = LinearRegression().fit(tr[LEADER_FEATS], tr.l_best_existing)
    l_den = den.predict(va[LEADER_FEATS])
    return s, s - l_den, l_den


def forecast_releases(R):
    """Adds pred_lstreams, pred_l_leader, pred_lstrength, sigma and p_beat to a copy of R."""
    R = R.copy()
    for c in ["pred_lstreams", "pred_l_leader", "pred_lstrength"]:
        R[c] = np.nan
    for y in range(2018, 2024):                      # rolling: each year from earlier years
        idx = R.index[R.year == y]
        s, st, ld = _fit_predict(R[R.year < y], R.loc[idx])
        R.loc[idx, ["pred_lstreams", "pred_lstrength", "pred_l_leader"]] = np.column_stack([s, st, ld])
    idx = R.index[R.year >= 2024]                    # final: trained on 2017-2023
    s, st, ld = _fit_predict(R[R.year <= 2023], R.loc[idx])
    R.loc[idx, ["pred_lstreams", "pred_lstrength", "pred_l_leader"]] = np.column_stack([s, st, ld])

    R["resid"] = R.pred_lstrength - R.lstrength
    sig_by_year = R.groupby("year").resid.std()
    R["sigma"] = np.nan
    for y in range(2019, 2024):
        R.loc[R.year == y, "sigma"] = sig_by_year[y - 1]          # spread from the previous year
    R.loc[R.year >= 2024, "sigma"] = R[R.year.between(2019, 2023)].resid.std()
    R["p_beat"] = norm.sf(0, loc=R.pred_lstrength, scale=R.sigma)
    return R


def test_metrics(R):
    """One-time 2024-2026 score of the forecaster and of the leader forecast."""
    T = R[R.year >= 2024]
    beat = T.strength > 1
    pb = T.p_beat.clip(1e-4, 1 - 1e-4)
    err = np.abs(T.pred_lstreams - T.lstreams)
    base_pr = T.last_same_lstr.fillna(T.last_lstr).fillna(np.log(0.18))
    return {
        "releases": len(T), "beat_leader": int(beat.sum()),
        "err_all": float(np.exp(err.mean())),
        "err_top10": float(np.exp(err[T.top_song_rank <= 10].mean())),
        "auc": float(roc_auc_score(beat, T.pred_lstrength)),
        "logloss": float(-np.mean(beat * np.log(pb) + (~beat) * np.log(1 - pb))),
        "baseline_auc": float(roc_auc_score(beat, base_pr)),
        "leader_err": float(np.exp(np.abs(T.pred_l_leader - T.l_best_existing).mean())),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep-surprise", action="store_true")
    ap.add_argument("--features", default="data/release_features.csv")
    ap.add_argument("--announcements", default="data/release_announcements.csv")
    ap.add_argument("--outdir", default="data")
    a = ap.parse_args()
    R = forecast_releases(load_releases(a.features, a.announcements, not a.keep_surprise))
    m = test_metrics(R)
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in m.items()})
    R.to_csv(f"{a.outdir}/debut_forecast_release_predictions.csv", index=False)
    pd.DataFrame([m]).to_csv(f"{a.outdir}/debut_forecast_test_metrics.csv", index=False)


if __name__ == "__main__":
    main()
