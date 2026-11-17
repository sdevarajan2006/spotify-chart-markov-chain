"""Label each new release as announced ahead of time or not (standard rule).

Rule
- Albums (is_album = 1): announced ahead if the album's Wikipedia page had any
  page views on or before D-2, where D is the release's first chart date.
  An album page usually goes up when the album is announced.
    status = "known"     page had views by D-2
    status = "surprise"  page first viewed on D-1 or later (includes pages created after release)
    status = "no_page"   no Wikipedia page matched yet; treated as known, flagged for checking
- Singles: assumed known ahead (status = "single_assumed").
- Manual corrections in data/announcement_overrides.csv win
  (columns: artist, first_chart_date, status, note), e.g. for a page that was renamed.

Run from the repo root after pull_wikipedia_pageviews.py:
    python scripts/check_announcements.py
Writes data/release_announcements.csv and prints the top-10 debuts that need a look.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CUTOFF_DAYS = 2

F = pd.read_csv(DATA / "release_features.csv", parse_dates=["first_chart_date"])
B = pd.read_csv(DATA / "wikipedia_album_titles.csv", parse_dates=["first_chart_date"])
P = pd.read_csv(DATA / "wikipedia_pageviews_daily.csv.gz", usecols=["kind", "enwiki_title", "date"],
                parse_dates=["date"])
first_view = P[P.kind == "album"].groupby("enwiki_title").date.min()

B = B.drop_duplicates(["artist", "album"])
B["first_view"] = B.enwiki_title.map(first_view)
pages = B.set_index(["artist", "album"])[["enwiki_title", "first_view"]]

R = F[["first_chart_date", "artist", "hist_artist", "top_song", "top_song_rank", "strength",
       "is_album", "wikipedia_album"]].copy()
key = list(zip(R.hist_artist, R.wikipedia_album))
R["album_page"] = [pages.enwiki_title.get(k) if k in pages.index else None for k in key]
R["page_first_view"] = [pages.first_view.get(k) if k in pages.index else pd.NaT for k in key]
R["page_lead_days"] = (R.first_chart_date - pd.to_datetime(R.page_first_view)).dt.days


def status(r):
    if not r.is_album:
        return "single_assumed"
    if pd.isna(r.page_first_view):
        return "no_page"
    return "known" if r.page_lead_days >= CUTOFF_DAYS else "surprise"


R["status_rule"] = R.apply(status, axis=1)
R["status"] = R.status_rule
R["override_note"] = ""
ov_path = DATA / "announcement_overrides.csv"
if ov_path.exists():
    ov = pd.read_csv(ov_path, parse_dates=["first_chart_date"])
    for o in ov.itertuples():
        m = (R.artist == o.artist) & (R.first_chart_date == o.first_chart_date)
        R.loc[m, "status"] = o.status
        R.loc[m, "override_note"] = o.note if isinstance(o.note, str) else ""
R["known_ahead"] = R.status.isin(["known", "no_page", "single_assumed"]).astype(int)
R.to_csv(DATA / "release_announcements.csv", index=False)

print(R.status.value_counts().to_string())
top = R[(R.top_song_rank <= 10) & R.status.isin(["surprise", "no_page"]) & (R.override_note == "")]
print(f"\nTop-10 album debuts to look at ({len(top)}): surprise = page first viewed after D-2, "
      f"no_page = page not matched")
cols = ["first_chart_date", "artist", "wikipedia_album", "album_page", "page_lead_days", "top_song_rank", "status"]
print(top[cols].sort_values(["status", "first_chart_date"]).to_string(index=False))
