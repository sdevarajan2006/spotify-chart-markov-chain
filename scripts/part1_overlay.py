"""Overlay the debut strength forecaster on the Part 1 hold model.

Part 1: does the #1 on the D-2 chart still hold #1 on day D? (logistic regression)
New input: P(a release first charting on day D out-streams every existing song),
from the debut forecaster (scripts/debut_forecaster.py, gradient boosting picked by the grid search).

No look-ahead:
- Release predictions for 2019-2023 come from forecasters trained on earlier years only.
- Predictions for 2024-2026 come from the forecaster trained on 2017-2023.
- Part 1 is trained on 2019-2023 and tested on 2024-2026.
Assumes the release calendar is known by D-2 (releases come from the chart itself).
"""
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from debut_forecaster import forecast_releases, load_releases, test_metrics

CHART = sys.argv[1] if len(sys.argv) > 1 else "data/spotify_us_daily_top200_2017_2026.csv.gz"
FEAT = sys.argv[2] if len(sys.argv) > 2 else "data/release_features.csv"
OUTDIR = sys.argv[3] if len(sys.argv) > 3 else "data"
ANN = sys.argv[4] if len(sys.argv) > 4 else "data/release_announcements.csv"
EXCLUDE_SURPRISE = "--keep-surprise" not in sys.argv

# ---------------- debut forecaster (v2, scripts/debut_forecaster.py) ----------------
R = forecast_releases(load_releases(FEAT, ANN, EXCLUDE_SURPRISE))
fc_test = test_metrics(R)

# ---------------- Part 1 days ----------------
c = pd.read_csv(CHART, parse_dates=["chart_date"])
c["key"] = c.track_name.str.lower().str.strip() + "|" + c.artists.str.lower()
t1 = c[c["rank"] == 1].set_index("chart_date")
t2 = c[c["rank"] == 2].set_index("chart_date")
d = pd.DataFrame({"key1": t1.key, "s1": t1.streams.astype(float), "s2": t2.streams.astype(float),
                  "doc1": t1.days_on_chart.astype(float)})
prev = d.shift(2, freq="D").reindex(d.index)
D = pd.DataFrame(index=d.index)
D["hold"] = (d.key1 == prev.key1).astype(int)
D["log_lead"] = np.log(prev.s1 / prev.s2 - 1)
D["debut_inc"] = (prev.doc1 == 1).astype(int)
D = D[prev.key1.notna()].copy()
D["year"] = D.index.year
D["friday"] = (D.index.dayofweek == 4).astype(int)


def any_beat(day_shift):
    g = R.assign(day=R.first_chart_date + pd.Timedelta(days=day_shift)).groupby("day")
    p_any = 1 - g.p_beat.apply(lambda p: np.prod(1 - p))
    return p_any.reindex(D.index).fillna(0.0)


D["p_threat"] = any_beat(0)       # releases first charting on day D
D["p_threat_prev"] = any_beat(1)  # releases first charting on D-1 (also invisible at the D-2 cutoff)
eps = 1e-3
D["logit_threat"] = np.log((D.p_threat + eps) / (1 - D.p_threat + eps))
D["logit_threat_prev"] = np.log((D.p_threat_prev + eps) / (1 - D.p_threat_prev + eps))

tr = D[D.year.between(2019, 2023)]
te = D[D.year >= 2024]
tr_full = D[D.year <= 2023]

SPECS = [
    ("Constant", []),
    ("Log lead", ["log_lead"]),
    ("Log lead + debut flag (current Part 1)", ["log_lead", "debut_inc"]),
    ("+ Friday flag", ["log_lead", "debut_inc", "friday"]),
    ("+ debut threat (day D releases)", ["log_lead", "debut_inc", "logit_threat"]),
    ("+ debut threat + Friday flag", ["log_lead", "debut_inc", "logit_threat", "friday"]),
    ("+ debut threat, day D and D-1 releases", ["log_lead", "debut_inc", "logit_threat", "logit_threat_prev"]),
]
rows, coefs = [], {}
for name, f in SPECS:
    for label, T_ in [("2019-2023", tr), ("2017-2023", tr_full)]:
        if f and label == "2017-2023" and any("threat" in x for x in f):
            continue                                   # no out-of-sample threat before 2018
        if f:
            m = LogisticRegression(C=1e6, max_iter=2000).fit(T_[f], T_.hold)
            p = m.predict_proba(te[f])[:, 1]
            if label == "2019-2023":
                coefs[name] = dict(zip(f, m.coef_[0].round(3)))
        else:
            p = np.full(len(te), T_.hold.mean())
        chg = te.hold == 0
        rows.append([name, label, log_loss(te.hold, p), ((p > 0.5) == te.hold).mean(),
                     int(((p < 0.5) & chg).sum()), int(((p < 0.5) & ~chg).sum())])
P = pd.DataFrame(rows, columns=["model", "trained_on", "test_logloss", "test_accuracy", "changes_caught", "false_alarms"])

pd.set_option("display.width", 220)
print("Debut forecaster, one-time test on 2024-2026:")
print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in fc_test.items()})
print(f"\nPart 1 test set: {len(te)} days, {int((te.hold == 0).sum())} change days")
print(P.round(3).to_string(index=False))
print("\nCoefficients (trained 2019-2023):")
for k, v in coefs.items():
    print(f"  {k}: {v}")

# where the threat feature matters
te = te.assign(p_hold_base=LogisticRegression(C=1e6, max_iter=2000).fit(tr[SPECS[2][1]], tr.hold).predict_proba(te[SPECS[2][1]])[:, 1],
               p_hold_new=LogisticRegression(C=1e6, max_iter=2000).fit(tr[SPECS[4][1]], tr.hold).predict_proba(te[SPECS[4][1]])[:, 1])
te.to_csv(f"{OUTDIR}/part1_overlay_test_days.csv")
R.to_csv(f"{OUTDIR}/debut_forecast_release_predictions.csv", index=False)
P.to_csv(f"{OUTDIR}/part1_overlay_results.csv", index=False)
