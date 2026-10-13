"""Debut strength forecaster, step 2 (run from the repo root). Rolling validation on 2019-2023 (each year predicted from all earlier years).
2024-2026 is not touched. Target: log day-one streams of the release's top song.
"""
import sys
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PATH = sys.argv[1] if len(sys.argv) > 1 else "data/release_features.csv"
F = pd.read_csv(PATH, parse_dates=["first_chart_date"])
F = F[F.no1_d2.notna()].reset_index(drop=True)
F["l_n_prev"] = np.log1p(F.n_prev)
F["l_days_since"] = np.log1p(F.days_since_last)
F["beat"] = F.strength > 1

HIST = ["last_ls", "last_same_ls", "best3_ls", "best2y_ls", "best_ever_ls", "ewm_ls",
        "l_n_prev", "n_no1_debuts", "l_days_since"]
POP = ["l_pop7_max", "l_pop28_max", "share_pop7_max"]
CTX = ["is_album", "friday", "l_mkt", "has_hist", "n_credited"]
ALL = HIST + POP + CTX
SMALL = ["best3_ls", "l_pop7_max", "is_album", "friday", "l_mkt", "has_hist"]
LOG_HIST = [c for c in HIST if c.endswith("_ls")]


def fill(X, tr):
    """For linear models: no history -> typical day-one streams for that release type."""
    X = X.copy()
    med = tr.groupby("is_album").lstreams.median()
    fb = tr.loc[X.index, "is_album"].map(med) if False else X.get("is_album", pd.Series(0, index=X.index)).map(med)
    for c in X.columns:
        if c in LOG_HIST:
            X[c] = X[c].fillna(fb)
    if "l_days_since" in X:
        X["l_days_since"] = X["l_days_since"].fillna(tr.l_days_since.median())
    return X.fillna(0)


MODELS = {
    "OLS, 6 features": (SMALL, lambda: LinearRegression(), True),
    "OLS, all 17": (ALL, lambda: LinearRegression(), True),
    "Ridge, all 17": (ALL, lambda: make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 20))), True),
    "Gradient boosting, all 17": (ALL, lambda: HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.04, max_leaf_nodes=15, min_samples_leaf=40,
        l2_regularization=1.0, random_state=0), False),
}


def fit_pred(name, tr, va):
    feats, make, linear = MODELS[name]
    Xtr, Xva = tr[feats], va[feats]
    if linear:
        Xtr, Xva = fill(Xtr, tr), fill(Xva, tr)
    m = make().fit(Xtr, tr.lstreams)
    return m.predict(Xva)


def baseline(tr, va):
    med = tr.groupby("is_album").lstreams.median()
    return va.last_same_ls.fillna(va.last_ls).fillna(va.is_album.map(med)).values


def den_model(tr):
    from sklearn.linear_model import LinearRegression as LR
    m = LR().fit(tr[["l_no1_d2", "friday"]], tr.l_best_existing)
    return m


def predict_all(name, tr, va):
    return baseline(tr, va) if name == "Last release (baseline)" else fit_pred(name, tr, va)


names = ["Last release (baseline)"] + list(MODELS)
rows, pooled = [], {n: [] for n in names}
for y in range(2019, 2024):
    tr, va = F[F.year < y], F[F.year == y]
    den = den_model(tr)
    l_den = den.predict(va[["l_no1_d2", "friday"]])
    # spread for P(beat): residuals of the last training year, predicted from the years before it
    itr, iva = F[F.year < y - 1], F[F.year == y - 1]
    iden = den_model(itr)
    for n in names:
        ps = predict_all(n, tr, va)
        ips = predict_all(n, itr, iva)
        sig = np.std((ips - iden.predict(iva[["l_no1_d2", "friday"]])) - iva.lstrength)
        pr = ps - l_den
        pb = norm.sf(0, loc=pr, scale=sig)
        e = np.abs(ps - va.lstreams.values)
        top = (va.top_song_rank <= 10).values
        rows.append([y, n, len(va), int(va.beat.sum()), np.exp(e.mean()), np.exp(e[top].mean())])
        pooled[n].append(pd.DataFrame({"year": y, "e": e, "top": top, "beat": va.beat.values,
                                       "pr": pr, "pb": pb}))

R = pd.DataFrame(rows, columns=["year", "model", "n", "beat", "err_all", "err_top10"])
pd.set_option("display.width", 200)
print("Typical error (x) on day-one streams, by validation year")
print(R.pivot(index="model", columns="year", values="err_all").reindex(names).round(2).to_string())
print()
out = []
for n in names:
    P = pd.concat(pooled[n])
    pb = np.clip(P.pb, 1e-4, 1 - 1e-4)
    ll = -np.mean(P.beat * np.log(pb) + (~P.beat) * np.log(1 - pb))
    o = np.argsort(-P.pr.values); b = P.beat.values[o]
    fa = np.cumsum(~b); fa_budget = int(round(P.beat.sum() * 0.65))
    caught = int(b[: np.searchsorted(fa, fa_budget, side="right")].sum())
    out.append([n, np.exp(P.e.mean()), np.exp(P.e[P.top].mean()), roc_auc_score(P.beat, P.pr), ll,
                caught, int(P.beat.sum()), fa_budget])
O = pd.DataFrame(out, columns=["model", "err_all", "err_top10", "beat_AUC", "beat_logloss",
                               "caught", "of_beats", "at_false_alarms"])
print("Pooled over 2019-2023 validation years")
print(O.round(3).to_string(index=False))
O.to_csv("data/debut_validation_results.csv", index=False)
R.to_csv("data/debut_validation_by_year.csv", index=False)
