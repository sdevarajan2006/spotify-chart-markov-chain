"""Build the Spotify US Daily Top 200 dataset (2017-01-01 onward).

Source: Kaggle "Spotify Charts Daily Updated" (gonzalopezgil):
  charts_songs_daily.csv  (all countries, ~11 GB)
  songs.csv               (track metadata, used to fill release_date/labels gaps)

Usage:
  python scripts/build_us_top200.py --charts ~/Downloads/charts_songs_daily.csv \
      --songs ~/Downloads/songs.csv --end 2026-09-17

Writes data/spotify_us_daily_top200_2017_2026.csv.gz and data/coverage_gap_log.txt.
"""
import argparse
import gzip
import os

import numpy as np
import pandas as pd

KAGGLE_COLS = (
    "date,country,rank,uri,artist_names,track_name,label,peak_rank,previous_rank,"
    "days_on_chart,streams,consecutive_days,entry_status,peak_date,entry_rank,"
    "entry_date,release_date,artist_uris"
).split(",")

OUT_COLS = [
    "chart_date", "rank", "prev_rank", "peak_rank", "streak", "days_on_chart",
    "streams", "track_name", "artists", "track_uri", "release_date", "labels",
]


def filter_us(charts_path, us_path):
    """Stream the all-country file and keep US rows (date is always 10 chars)."""
    n = 0
    with open(charts_path, "r", encoding="utf-8") as src, open(us_path, "w", encoding="utf-8") as dst:
        header = src.readline().strip().split(",")
        assert header == KAGGLE_COLS, f"unexpected columns: {header}"
        for line in src:
            if line[11:14] == "us,":
                dst.write(line)
                n += 1
    print(f"US rows: {n:,}")


def build(us_path, songs_path, start, end):
    k = pd.read_csv(us_path, names=KAGGLE_COLS, header=None, dtype=str, keep_default_na=False)
    k = k[(k.date >= start) & (k.date <= end)]

    def num(s):
        return pd.to_numeric(s.replace("", np.nan)).astype("Int64")

    consecutive = num(k.consecutive_days)
    out = pd.DataFrame({
        "chart_date": k.date,
        "rank": num(k["rank"]),
        "prev_rank": num(k.previous_rank.replace("-1", "")),  # -1 = not on yesterday's chart
        "peak_rank": num(k.peak_rank),
        "streak": (consecutive - 1).where(consecutive > 1),    # debut/re-entry = null, day 2 = 1
        "days_on_chart": num(k.days_on_chart),
        "streams": num(k.streams),
        "track_name": k.track_name,
        "artists": k.artist_names,                             # all artists, "|"-separated
        "track_uri": k.uri,
        "release_date": k.release_date.replace("", np.nan),
        "labels": k.label.replace("", np.nan),
    })
    assert out.track_uri.str.match(r"^spotify:track:[A-Za-z0-9]{22}$").all()
    assert not out.duplicated(["chart_date", "rank"]).any()

    # Fill metadata gaps from songs.csv, matching on track_uri or any alias in all_uris.
    s = pd.read_csv(songs_path, dtype=str, usecols=["track_uri", "label", "release_date", "all_uris"])
    aliases = s.assign(u=s.all_uris.fillna(s.track_uri).str.split("|")).explode("u")
    lookup = pd.concat([s.assign(u=s.track_uri), aliases]).drop_duplicates("u").set_index("u")
    out["release_date"] = out.release_date.fillna(out.track_uri.map(lookup.release_date))
    out["labels"] = out.labels.fillna(out.track_uri.map(lookup.label))

    return out.sort_values(["chart_date", "rank"])[OUT_COLS]


def gap_log(out, start, end):
    per_day = out.groupby("chart_date").size()
    expected = pd.date_range(start, end).strftime("%Y-%m-%d")
    missing = sorted(set(expected) - set(per_day.index))
    lines = [
        "Source: Kaggle gonzalopezgil charts_songs_daily (country=us)",
        f"Range {start} to {end}: {len(expected)} expected days, {per_day.size} present, {len(out)} rows",
        "Missing days: " + (", ".join(missing) or "none"),
        "Days with != 200 rows:",
        *[f"  {d}: {n}" for d, n in per_day[per_day != 200].items()],
        f"release_date missing rows: {out.release_date.isna().sum()}; "
        f"labels missing rows: {out.labels.isna().sum()}",
    ]
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--charts", help="Kaggle charts_songs_daily.csv (skip if --us-raw exists)")
    p.add_argument("--us-raw", default="us_raw.csv", help="intermediate US-only rows (no header)")
    p.add_argument("--songs", required=True)
    p.add_argument("--start", default="2017-01-01")
    p.add_argument("--end", required=True)
    p.add_argument("--out-dir", default="data")
    a = p.parse_args()

    if a.charts:
        filter_us(os.path.expanduser(a.charts), a.us_raw)
    out = build(a.us_raw, os.path.expanduser(a.songs), a.start, a.end)

    os.makedirs(a.out_dir, exist_ok=True)
    csv_path = os.path.join(a.out_dir, f"spotify_us_daily_top200_{a.start[:4]}_{a.end[:4]}.csv.gz")
    with gzip.GzipFile(csv_path, "wb", mtime=0) as f:
        out.to_csv(f, index=False)
    log = gap_log(out, a.start, a.end)
    with open(os.path.join(a.out_dir, "coverage_gap_log.txt"), "w") as f:
        f.write(log)
    print(log)
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
