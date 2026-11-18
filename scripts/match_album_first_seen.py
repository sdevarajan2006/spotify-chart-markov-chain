"""Match each album release to its first-seen date on Wikipedia's "List of YYYY albums" pages.

Run from the repo root after pull_wikipedia_album_history.py --parse and check_announcements.py:
    python scripts/match_album_first_seen.py
Writes data/release_album_list_status.csv, one row per release with a Wikipedia album match:
    first_chart_date, artist, hist_artist, top_song, wikipedia_album, page_view_status (from
    check_announcements.py), list_year, list_release_date, list_first_seen, lead_days
    (first_chart_date minus first_seen), known_list (1 if first seen on or before D-2, blank if
    no list match).
Match: normalized artist + album, list year within 1 of the first chart year, nearest list release date.
"""
import re
import unicodedata

import pandas as pd


def norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]", "", s)


H = pd.read_csv("data/wikipedia_album_first_seen.csv.gz", parse_dates=["first_seen", "release_date"])
H["ka"], H["kb"] = H.artist.map(norm), H.album.map(norm)
A = pd.read_csv("data/release_announcements.csv", parse_dates=["first_chart_date"])
X = A[A.wikipedia_album.notna()].copy()
X["ka"], X["kb"] = X.hist_artist.map(norm), X.wikipedia_album.map(norm)
X["yr"] = X.first_chart_date.dt.year
m = X.reset_index().merge(
    H[["list_year", "ka", "kb", "release_date", "first_seen"]].rename(
        columns={"release_date": "list_release_date", "first_seen": "list_first_seen"}),
    on=["ka", "kb"], how="left")
m = m[m.list_year.isna() | (m.list_year.sub(m.yr).abs() <= 1)]
m["gap"] = (m.list_release_date - m.first_chart_date).dt.days.abs()
m = m.sort_values(["index", "gap"]).drop_duplicates("index")
m["lead_days"] = (m.first_chart_date - m.list_first_seen).dt.days
m["known_list"] = (m.list_first_seen <= m.first_chart_date - pd.Timedelta(days=2)).astype(float).where(m.list_first_seen.notna())
m = m.rename(columns={"status": "page_view_status"})
cols = ["first_chart_date", "artist", "hist_artist", "top_song", "wikipedia_album", "page_view_status",
        "list_year", "list_release_date", "list_first_seen", "lead_days", "known_list"]
m[cols].to_csv("data/release_album_list_status.csv", index=False)
print(f"{len(m)} album releases, {int(m.list_first_seen.notna().sum())} matched to the list")
print(pd.crosstab(m.page_view_status, m.known_list.fillna(-1).rename("known_list (-1 = no match)")))
