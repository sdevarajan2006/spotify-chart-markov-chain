"""Debut strength forecaster, step 2 (run from the repo root after build_release_features.py).

Target: log day-one streams of a release's top song. Train 2017-2023, test 2024-2026.
Compares ridge and gradient boosting against the typical-release and last-release baselines,
and converts the stream forecast to strength (vs a forecast of the day-D leader) for P(beats leader).
"""
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

F = pd.read_csv("data/release_features.csv", parse_dates=["first_chart_date"])
F["l_n_prev"] = np.log1p(F.n_prev)
F["l_days_since"] = np.log1p(F.days_since_last)
F["album_x_last"] = F.album * F.last_ls
F = F[F.l_no1_d2.notna()].reset_index(drop=True)

HIST = ["last_ls", "last_same_ls", "best3_ls", "best2y_ls", "best_ever_ls", "ewm_ls",
        "l_n_prev", "n_no1_debuts", "l_days_since"]
POP = ["l_pop7_max", "l_pop28_max", "share_pop7_max"]
CTX = ["album", "friday", "l_mkt", "has_hist", "n_credited"]
FEATS = HIST + POP + CTX

train = F[F.year <= 2023]
test = F[F.year >= 2024].copy()
top10 = test.top_song_rank <= 10
beat = test.strength > 1


def fill_linear(X, ref):
    X = X.copy()
    med = ref.groupby("album").lstreams.median()
    fallback = X["album"].map(med) if "album" in X else med.mean()
    for c in HIST:
        if c.endswith("_ls"):
            X[c] = X[c].fillna(fallback)
    return X.fillna({"l_days_since": ref.l_days_since.median() if "l_days_since" in ref else 0,
                     "share_pop7_max": 0}).fillna(0)


def models():
    return {
        "ridge": lambda: make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 20))),
        "gbm": lambda: HistGradientBoostingRegressor(max_iter=300, learning_rate=0.04, max_leaf_nodes=15,
                                                     min_samples_leaf=40, l2_regularization=1.0, random_state=0),
    }


def fit_predict(name, feats, target, tr, te):
    m = models()[name]()
    Xtr, Xte = tr[feats], te[feats]
    if name == "ridge":
        Xtr, Xte = fill_linear(Xtr, tr), fill_linear(Xte, tr)
    m.fit(Xtr, tr[target])
    return m.predict(Xte), m


def oof_sigma(name, feats, target):
    """Residual SD from expanding-window predictions on 2019-2023 (train years only)."""
    res = []
    for y in range(2019, 2024):
        tr, va = F[F.year < y], F[F.year == y]
        p, _ = fit_predict(name, feats, target, tr, va)
        res.append(va[target].values - p)
    return np.std(np.concatenate(res))


def err(pred, actual, mask=None):
    e = np.abs(pred - actual)
    if mask is not None:
        e = e[mask]
    return float(np.exp(e.mean()))


def beat_stats(p_beat):
    flag = p_beat > 0.5
    pb = np.clip(p_beat, 1e-4, 1 - 1e-4)
    ll = -np.mean(beat * np.log(pb) + (~beat) * np.log(1 - pb))
    return int((flag & beat).sum()), int((flag & ~beat).sum()), round(ll, 4), round(float(np.mean((pb - beat) ** 2)), 4)


rows = []
F["lstreams"] = np.log(F.top_song_streams)

# ---- denominator model: day-D best existing streams from the D-2 chart ----
den = RidgeCV().fit(train[["l_no1_d2", "friday"]], train.l_best_existing)
test["l_den_hat"] = den.predict(test[["l_no1_d2", "friday"]])
den_err = err(test.l_den_hat, test.l_best_existing)

# ---- baselines, in streams and strength ----
med_s = train.groupby("album").lstreams.median()
med_r = train.groupby("album").lstrength.median()
base = {
    "Typical release": (test.album.map(med_s), test.album.map(med_r)),
    "Last release, same type": (test.last_same_ls.fillna(test.last_ls).fillna(test.album.map(med_s)),
                                test.last_same_lstr.fillna(test.last_lstr).fillna(test.album.map(med_r))),
}
sig_b = {}
for k, (ps, pr) in base.items():
    tr_ps = {"Typical release": train.album.map(med_s),
             "Last release, same type": train.last_same_ls.fillna(train.last_ls).fillna(train.album.map(med_s))}[k]
    tr_pr = {"Typical release": train.album.map(med_r),
             "Last release, same type": train.last_same_lstr.fillna(train.last_lstr).fillna(train.album.map(med_r))}[k]
    s = np.std(train.lstrength - tr_pr)
    pb = norm.sf(0, loc=pr, scale=s)
    rows.append([k, "baseline", err(ps, test.lstreams), err(ps, test.lstreams, top10),
                 err(pr, test.lstrength), err(pr, test.lstrength, top10), *beat_stats(pb),
                 int(((pr > 0) & beat).sum()), int(((pr > 0) & ~beat).sum())])

# ---- models ----
preds = {}
for name in ["ridge", "gbm"]:
    # A: predict day-one streams, then divide by forecast denominator
    ps, _ = fit_predict(name, FEATS, "lstreams", train, test)
    pr = ps - test.l_den_hat
    # sigma for strength from OOF streams residual combined with denominator residual
    s_st = oof_sigma(name, FEATS, "lstreams")
    s_den = np.std(train.l_best_existing - den.predict(train[["l_no1_d2", "friday"]]))
    s = np.sqrt(s_st ** 2 + s_den ** 2)
    pb = norm.sf(0, loc=pr, scale=s)
    preds[f"{name}_streams"] = (ps, pr, pb)
    rows.append([f"{name}: target = day-one streams", "model", err(ps, test.lstreams), err(ps, test.lstreams, top10),
                 err(pr, test.lstrength), err(pr, test.lstrength, top10), *beat_stats(pb),
                 int(((pr > 0) & beat).sum()), int(((pr > 0) & ~beat).sum())])
    # B: predict strength directly (adds D-2 #1 streams as an input)
    pr2, _ = fit_predict(name, FEATS + ["l_no1_d2"], "lstrength", train, test)
    s2 = oof_sigma(name, FEATS + ["l_no1_d2"], "lstrength")
    pb2 = norm.sf(0, loc=pr2, scale=s2)
    ps2 = pr2 + test.l_den_hat
    preds[f"{name}_strength"] = (ps2, pr2, pb2)
    rows.append([f"{name}: target = strength", "model", err(ps2, test.lstreams), err(ps2, test.lstreams, top10),
                 err(pr2, test.lstrength), err(pr2, test.lstrength, top10), *beat_stats(pb2),
                 int(((pr2 > 0) & beat).sum()), int(((pr2 > 0) & ~beat).sum())])

cols = ["model", "kind", "err_streams_all", "err_streams_top10", "err_strength_all", "err_strength_top10",
        "caught_p50", "fa_p50", "beat_logloss", "beat_brier", "caught_pt", "fa_pt"]
out = pd.DataFrame(rows, columns=cols)
pd.set_option("display.width", 250)
print(f"test releases {len(test)}, top-10 debuts {top10.sum()}, beat leader {beat.sum()}, base rate {beat.mean():.3f}")
print(f"denominator forecast error {den_err:.2f}x")
print(out.round(2).to_string(index=False))
out.to_csv("data/debut_forecast_results.csv", index=False)

# save test predictions of the best streams model for inspection
ps, pr, pb = preds["gbm_streams"]
t = test[["first_chart_date", "artist", "artist_used", "top_song", "album", "top_song_streams", "strength",
          "last_same_ls", "best3_ls", "l_pop7_max"]].copy()
t["pred_streams"] = np.exp(ps).round(); t["pred_strength"] = np.exp(pr).round(3); t["p_beat"] = pb.round(3)
t.to_csv("data/debut_forecast_test_predictions_gbm.csv", index=False)
ps, pr, pb = preds["ridge_streams"]
t["pred_streams"] = np.exp(ps).round(); t["pred_strength"] = np.exp(pr).round(3); t["p_beat"] = pb.round(3)
t.to_csv("data/debut_forecast_test_predictions_ridge.csv", index=False)
