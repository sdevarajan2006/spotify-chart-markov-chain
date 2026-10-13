"""Pull daily counts of news articles mentioning each charting artist, from the
GDELT DOC 2.0 API (free, worldwide online news, 2017 onward).

Run from the repo root (needs internet; standard library + pandas only):
    python scripts/pull_gdelt_news.py --limit 3      # quick test: prints what came back
    python scripts/pull_gdelt_news.py                # full run, about 1,000 artists
    python scripts/pull_gdelt_news.py --parse-only   # rebuild the CSV from saved raw files

GDELT asks for at most one request every 5 seconds, so a full run takes
roughly 1.5 hours if GDELT returns a whole decade at daily resolution. If it
doesn't, the script finds the longest window that comes back daily (tested once,
on Taylor Swift) and only fetches the windows covering the 120 days before each
of the artist's releases. The plan and its estimated run time are printed first.
It saves every response under data/raw/gdelt/ and skips finished requests, so
you can stop it (Ctrl-C) and rerun it any time.

Query per artist:  "<artist name>" (album OR song OR single OR music OR rapper OR singer OR band)
The music words keep "Future" or "Drake" from matching unrelated news.
To change the query for an artist, add a row to data/gdelt_query_overrides.csv
(columns: name, query). Names with "$" have it replaced by "s".

Output: data/gdelt_news_daily.csv.gz with columns
    name, date, articles (matching articles that day), all_articles (all articles
    GDELT monitored that day, for scaling), query
"""
import argparse
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw" / "gdelt"
FEATURES = DATA / "release_features.csv"
ALIASES = DATA / "artist_aliases.csv"
OVERRIDES = DATA / "gdelt_query_overrides.csv"
OUT = DATA / "gdelt_news_daily.csv.gz"
LOG = DATA / "gdelt_query_log.csv"

API = "https://api.gdeltproject.org/api/v2/doc/doc"
UA = "spotify-chart-markov-chain/0.1 (https://github.com/sdevarajan2006/spotify-chart-markov-chain)"
START = date(2017, 1, 1)
END = date.today()
WAIT = 5.5
CONTEXT = "(album OR song OR single OR music OR rapper OR singer OR band)"
_last = [0.0]


def fetch(params):
    """One GDELT request, paced to one per WAIT seconds. Returns (json or None, raw text)."""
    url = API + "?" + urllib.parse.urlencode(params)
    for k in range(5):
        gap = time.time() - _last[0]
        if gap < WAIT:
            time.sleep(WAIT - gap)
        _last[0] = time.time()
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=90) as r:
                text = r.read().decode("utf-8", "replace")
        except (urllib.error.URLError, TimeoutError) as e:
            print(f"    network error ({e}); retrying")
            time.sleep(15 * (k + 1))
            continue
        if "limit requests" in text.lower():
            print("    rate-limited; waiting 30s")
            time.sleep(30)
            continue
        try:
            return json.loads(text), text
        except json.JSONDecodeError:
            return None, text.strip()[:300]
    return None, "gave up after retries"


def query_for(name, ov):
    if name in ov:
        return ov[name]
    clean = re.sub(r"[^\w\s'.&-]", "", name.replace("$", "s")).strip()
    return f'"{clean}" {CONTEXT}'


def points(js):
    """Daily points from a timelinevolraw response: list of (date, value, norm)."""
    out = []
    for series in (js or {}).get("timeline", []):
        for d in series.get("data", []):
            out.append((d["date"][:8], d.get("value", 0), d.get("norm")))
    return out


def is_daily(pts):
    if len(pts) < 3:
        return True
    ds = sorted(datetime.strptime(p[0], "%Y%m%d") for p in pts)
    gaps = [(b - a).days for a, b in zip(ds, ds[1:])]
    return sorted(gaps)[len(gaps) // 2] <= 1


def raw_path(name, a, b):
    safe = re.sub(r"[^\w\-]+", "_", name)[:100]
    return RAW / safe / f"{a:%Y%m%d}_{b:%Y%m%d}.json"


def get_window(name, q, a, b):
    """Fetch [a, b] once (saved to disk; a saved file is reused)."""
    p = raw_path(name, a, b)
    if p.exists():
        return json.loads(p.read_text())
    params = {"query": q, "mode": "timelinevolraw", "format": "json",
              "startdatetime": f"{a:%Y%m%d}000000", "enddatetime": f"{b:%Y%m%d}235959"}
    js, text = fetch(params)
    pts = points(js)
    saved = {"name": name, "query": q, "start": f"{a}", "end": f"{b}", "daily": is_daily(pts),
             "points": pts, "error": None if js is not None else text}
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(saved))
    return saved


SPAN_FILE = RAW / "_daily_span_days.txt"
SPANS = [4000, 730, 365, 180, 90, 45]   # candidate window lengths, longest first


def find_span(name, q):
    """Longest window GDELT still returns at daily resolution, found once and remembered."""
    if SPAN_FILE.exists():
        return int(SPAN_FILE.read_text())
    for span in SPANS:
        b = END
        a = max(START, b - timedelta(days=span - 1))
        s = get_window(name, q, a, b)
        if s["error"]:
            raise SystemExit(f"GDELT error while testing window length: {s['error']}")
        if s["daily"] and len(s["points"]) > 0:
            SPAN_FILE.parent.mkdir(parents=True, exist_ok=True)
            SPAN_FILE.write_text(str(span))
            print(f"GDELT returns daily data for windows up to {span} days")
            return span
    raise SystemExit("GDELT never returned daily data; check data/raw/gdelt for the responses")


def needed_windows(dates, span, before=120):
    """Fixed span-length chunks from START that overlap [D-before, D-1] for any release date D."""
    chunks = []
    a = START
    while a <= END:
        b = min(END, a + timedelta(days=span - 1))
        if any(a <= d - timedelta(days=1) and d - timedelta(days=before) <= b for d in dates):
            chunks.append((a, b))
        a = b + timedelta(days=1)
    return chunks


def load_names():
    """Artist -> release dates (as the first-credited artist or the artist whose record is used)."""
    F = pd.read_csv(FEATURES, parse_dates=["first_chart_date"])
    dates = {}
    for col in ["hist_artist", "artist_used"]:
        for n, d in zip(F[col], F.first_chart_date):
            if isinstance(n, str) and sum(ch.isalnum() for ch in n) >= 2:
                dates.setdefault(n, set()).add(d.date())
    return dict(sorted(dates.items()))


def parse():
    rows, log = [], []
    for f in sorted(RAW.glob("*/*.json")):
        s = json.loads(f.read_text())
        log.append((s["name"], s["start"], s["end"], s["daily"], len(s["points"]), s["error"]))
        if s["daily"]:
            rows += [(s["name"], d, v, n, s["query"]) for d, v, n in s["points"]]
    P = pd.DataFrame(rows, columns=["name", "date", "articles", "all_articles", "query"])
    P["date"] = pd.to_datetime(P.date, format="%Y%m%d").dt.date
    P = P.drop_duplicates(["name", "date"]).sort_values(["name", "date"])
    P.to_csv(OUT, index=False, compression="gzip")
    L = pd.DataFrame(log, columns=["name", "start", "end", "daily", "n_points", "error"])
    L.to_csv(LOG, index=False)
    print(f"wrote {len(P):,} daily rows for {P.name.nunique()} artists to {OUT.relative_to(ROOT)}")
    print(f"errors: {L.error.notna().sum()} (see {LOG.relative_to(ROOT)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, help="only this many artists (for testing)")
    ap.add_argument("--parse-only", action="store_true")
    a = ap.parse_args()
    if a.parse_only:
        parse()
        return
    ov = {}
    if OVERRIDES.exists():
        o = pd.read_csv(OVERRIDES)
        ov = dict(zip(o.name, o.query))
    rel = load_names()
    names = list(rel)[: a.limit] if a.limit else list(rel)
    span = find_span("Taylor Swift", query_for("Taylor Swift", ov))
    plan = {n: needed_windows(rel[n], span) for n in names}
    total = sum(len(v) for v in plan.values())
    print(f"{len(names)} artists, {total} requests, about {total * WAIT / 3600:.1f} hours")
    for i, n in enumerate(names, 1):
        q = query_for(n, ov)
        for w0, w1 in plan[n]:
            s = get_window(n, q, w0, w1)
        if a.limit:
            pts = s["points"]
            print(f"  {n}: {len(plan[n])} windows; last window {len(pts)} points, daily={s['daily']}, "
                  f"first={pts[0] if pts else None}, error={s['error']}")
        elif i % 25 == 0:
            print(f"  {i}/{len(names)} artists done")
    parse()


if __name__ == "__main__":
    main()
