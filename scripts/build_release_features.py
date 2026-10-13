"""Debut strength forecaster, step 1 (run from the repo root): build one row per new release with features known at the D-2 cutoff.

Target: log day-one streams of the release's top song (top_song_streams).
History and popularity only use charts dated D-2 or earlier.
"""
import sys
import numpy as np
import pandas as pd

DATA = sys.argv[1] if len(sys.argv) > 1 else "data"
OUT = sys.argv[2] if len(sys.argv) > 2 else "data/release_features.csv"

rel = pd.read_csv(f"{DATA}/release_groups.csv", parse_dates=["first_chart_date"])
ali = pd.read_csv(f"{DATA}/artist_aliases.csv")
chart = pd.read_csv(f"{DATA}/spotify_us_daily_top200_2017_2026.csv.gz", parse_dates=["chart_date"])

# Alias table: alias -> one name; joint projects / combined credits -> all members
alias_map = {}
for _, a in ali.iterrows():
    alias_map[a["name"]] = a["credit_to"].split("|")


def canon(name):
    return alias_map.get(name, [name])


# ---- Daily artist streams (any credit) and chart-level series ----
ch = chart[["chart_date", "rank", "streams", "artists", "track_name"]].copy()
ch["artist_list"] = ch["artists"].fillna("").str.split("|")
ex = ch.explode("artist_list")
ex = ex[ex["artist_list"] != ""]
ex["canon"] = ex["artist_list"].map(canon)
ex = ex.explode("canon")
art_day = ex.groupby(["canon", "chart_date"])["streams"].sum()

dates = pd.date_range(ch.chart_date.min(), ch.chart_date.max())
total = ch.groupby("chart_date")["streams"].sum().reindex(dates)
no1 = ch[ch["rank"] == 1].set_index("chart_date")["streams"].reindex(dates)

# 7-day artist stream sums ending on each date
art_wide = art_day.unstack("canon").reindex(dates).fillna(0)
art7 = art_wide.rolling(7, min_periods=1).sum()
art28 = art_wide.rolling(28, min_periods=1).sum()
tot7 = total.rolling(7, min_periods=1).sum()

# Credited artists on the release's top song (first chart date)
top_artists = (ch.drop_duplicates(["chart_date", "track_name"])
                 .set_index(["chart_date", "track_name"])["artists"])

# ---- Release history (first-credited artist, alias-mapped) ----
r = rel[rel.new_release].sort_values(["first_chart_date", "artist"]).reset_index(drop=True)
r["album_charted10"] = (r.songs_charted >= 10).astype(int)   # known only after release; kept for reference
r["album"] = r.wikipedia_album.notna().astype(int)            # is_album: an album came out that day (pre-release)
r["is_album"] = r["album"]
r["lstreams"] = np.log(r.top_song_streams)
r["lstrength"] = np.log(r.strength)
r["hist_artist"] = r.artist.map(lambda a: canon(a)[0])

hist = {}
for i, row in r.iterrows():
    hist.setdefault(row.hist_artist, []).append(i)


def artist_feats(a, D, album):
    cutoff = D - pd.Timedelta(days=2)
    idx = [j for j in hist.get(a, []) if r.at[j, "first_chart_date"] <= cutoff]
    f = {"n_prev": len(idx)}
    if not idx:
        return f
    h = r.loc[idx]
    f["last_ls"] = h.lstreams.iloc[-1]
    f["last_lstr"] = h.lstrength.iloc[-1]
    same = h[h.album == album]
    f["last_same_ls"] = same.lstreams.iloc[-1] if len(same) else np.nan
    f["last_same_lstr"] = same.lstrength.iloc[-1] if len(same) else np.nan
    f["best3_ls"] = h.lstreams.iloc[-3:].max()
    f["best3_lstr"] = h.lstrength.iloc[-3:].max()
    yr = h[h.first_chart_date > D - pd.Timedelta(days=730)]
    f["best2y_ls"] = yr.lstreams.max() if len(yr) else np.nan
    f["best_ever_ls"] = h.lstreams.max()
    w = 0.5 ** (np.arange(len(h))[::-1])            # weight halves each release back
    f["ewm_ls"] = np.average(h.lstreams, weights=w)
    f["n_no1_debuts"] = int(h.debuted_at_1.sum())
    f["days_since_last"] = (D - h.first_chart_date.iloc[-1]).days
    return f


def pop_feats(a, D):
    c = D - pd.Timedelta(days=2)
    if a not in art7.columns or c < dates[0]:
        return {"pop7": 0.0, "pop28": 0.0}
    return {"pop7": art7.at[c, a], "pop28": art28.at[c, a]}


rows = []
for i, row in r.iterrows():
    D = row.first_chart_date
    credited = [row.hist_artist]
    key = (D, row.top_song)
    if key in top_artists.index:
        v = top_artists.loc[key]
        v = v.iloc[0] if isinstance(v, pd.Series) else v
        for n in str(v).split("|"):
            for c in canon(n):
                if c not in credited:
                    credited.append(c)
    # strongest credited artist: highest best-of-last-3 day-one streams
    cand = []
    for a in credited:
        f = artist_feats(a, D, row.album)
        f.update(pop_feats(a, D))
        f["artist_used"] = a
        cand.append(f)
    with_hist = [f for f in cand if f["n_prev"] > 0]
    best = max(with_hist, key=lambda f: f["best3_ls"]) if with_hist else cand[0]
    best = dict(best)
    best["pop7_max"] = max(f["pop7"] for f in cand)
    best["pop28_max"] = max(f["pop28"] for f in cand)
    c = D - pd.Timedelta(days=2)
    best["mkt_tot7"] = tot7.get(c, np.nan)
    best["no1_d2"] = no1.get(c, np.nan)
    best["n_credited"] = len(credited)
    rows.append(best)

F = pd.concat([r.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
F["year"] = F.first_chart_date.dt.year
F["friday"] = (F.weekday == "Friday").astype(int)
F["has_hist"] = (F.n_prev > 0).astype(int)
F["l_no1_d2"] = np.log(F.no1_d2)
F["l_best_existing"] = np.log(F.best_existing_streams)
F["l_mkt"] = np.log(F.mkt_tot7)
for p in ["pop7_max", "pop28_max"]:
    F["l_" + p] = np.log1p(F[p])
    F["share_" + p] = F[p] / F.mkt_tot7
F.to_csv(OUT, index=False)
print(F.shape, F.has_hist.mean().round(3), F.artist_used.ne(F.hist_artist).sum(), "releases judged on a featured artist")
