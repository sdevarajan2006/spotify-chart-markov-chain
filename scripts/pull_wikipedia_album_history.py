"""When did each album first appear on Wikipedia's "List of YYYY albums" pages?

Backtest release calendar, albums part (see CLAUDE.md, Release calendar plan). For each list
page we take one revision per UTC day (the last revision before midnight), parse the artist,
album and release date from the wikitext, and record the first day each album was on the list.
An album is "known by D-2" if its first-seen day is on or before D-2.

Run from the repo root (needs internet; standard library + pandas only):
    python scripts/pull_wikipedia_album_history.py --list            # revision lists (cached)
    python scripts/pull_wikipedia_album_history.py --fetch           # download one revision per day, saved gzipped
    python scripts/pull_wikipedia_album_history.py --parse           # daily snapshots -> first-seen table
    python scripts/pull_wikipedia_album_history.py --fetch --years 2024 2025   # limit to some list years
Stop and rerun any time: finished downloads are skipped.

Raw revisions: data/raw/wikipedia_history/<page>/<revid>.txt.gz (not tracked by git)
Output: data/wikipedia_album_first_seen.csv.gz with one row per list year, artist and album:
    list_year, artist, album, release_date (latest listed), first_seen (first day on the list,
    needing two consecutive daily snapshots or the last snapshot), last_seen, days_seen, page
"""
import argparse
import gzip
import json
import re
import time
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

API = "https://en.wikipedia.org/w/api.php"
UA = "spotify-chart-markov-chain/0.1 (https://github.com/sdevarajan2006/spotify-chart-markov-chain)"
ROOT = Path(__file__).resolve().parents[1]
HIST = ROOT / "data" / "raw" / "wikipedia_history"
OUT = ROOT / "data" / "wikipedia_album_first_seen.csv.gz"
WAIT = 0.6
BATCH = 5            # revisions per content request (pages are large)
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September",
          "October", "November", "December"]


def pages_for(year):
    if year in (2021, 2026):   # split into half-year pages; the history of the early pages sits under the halves
        return [f"List of {year} albums", f"List of {year} albums (January–June)", f"List of {year} albums (July–December)"]
    return [f"List of {year} albums"]


def slug(title):
    return re.sub(r"[^A-Za-z0-9]+", "_", title).strip("_")


def api(**params):
    params.update(format="json", formatversion="2")
    url = API + "?" + urllib.parse.urlencode(params)
    for k in range(5):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept-Encoding": "gzip"})
            with urllib.request.urlopen(req, timeout=120) as r:
                raw = r.read()
                if r.headers.get("Content-Encoding") == "gzip":
                    raw = gzip.decompress(raw)
            return json.loads(raw)
        except Exception as e:
            print(f"    request failed ({e}); retrying")
            time.sleep(10 * (k + 1))
    raise SystemExit("gave up after retries")


def parse_ts(s):
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def list_revisions(title, year):
    """All revisions (id, timestamp), oldest first. Cached, except for the current year."""
    p = HIST / slug(title) / "revisions.json"
    if p.exists() and year < date.today().year:
        return json.loads(p.read_text())
    revs, cont = [], {}
    while True:
        r = api(action="query", prop="revisions", titles=title, rvprop="ids|timestamp", rvlimit="500",
                rvdir="newer", **cont)
        pg = r["query"]["pages"][0]
        if pg.get("missing"):
            return []
        revs += [(x["revid"], x["timestamp"]) for x in pg.get("revisions", [])]
        if "continue" not in r:
            break
        cont = r["continue"]
        time.sleep(WAIT)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(revs))
    print(f"  {title}: {len(revs)} revisions, {revs[0][1][:10] if revs else ''} to {revs[-1][1][:10] if revs else ''}")
    return revs


def daily_snapshots(revs, year):
    """{day: revid} = last revision before the end of each UTC day, from creation to Jan 31 of year+1."""
    if not revs:
        return {}
    times = [parse_ts(t) for _, t in revs]
    end = min(date.today() - timedelta(days=1), date(year + 1, 1, 31))
    out, i, day = {}, 0, times[0].date()
    while day <= end:
        cutoff = datetime(day.year, day.month, day.day, tzinfo=timezone.utc) + timedelta(days=1)
        while i + 1 < len(times) and times[i + 1] < cutoff:
            i += 1
        if times[i] < cutoff:
            out[day] = revs[i][0]
        day += timedelta(days=1)
    return out


def rev_path(title, revid):
    return HIST / slug(title) / f"{revid}.txt.gz"


def fetch_all(years):
    for year in years:
        for title in pages_for(year):
            revs = list_revisions(title, year)
            snaps = daily_snapshots(revs, year)
            need = sorted({r for r in snaps.values() if not rev_path(title, r).exists()})
            print(f"{title}: {len(set(snaps.values()))} snapshots needed, {len(need)} to download")
            for k in range(0, len(need), BATCH):
                ids = need[k:k + BATCH]
                r = api(action="query", prop="revisions", revids="|".join(map(str, ids)), rvprop="ids|content",
                        rvslots="main")
                got = {}
                for pg in r.get("query", {}).get("pages", []):
                    for rv in pg.get("revisions", []):
                        got[rv["revid"]] = rv["slots"]["main"].get("content", "")
                for rid in ids:
                    if rid in got:
                        with gzip.open(rev_path(title, rid), "wt", encoding="utf-8") as f:
                            f.write(got[rid])
                    else:
                        print(f"    revision {rid} not returned")
                if (k // BATCH) % 20 == 19:
                    print(f"    {k + len(ids)}/{len(need)}")
                time.sleep(WAIT)


# ---------------- wikitext parsing ----------------

def clean(text):
    t = re.sub(r"<ref[^>]*/>", "", text)
    t = re.sub(r"<ref[^>]*>.*?</ref>", "", t, flags=re.S)
    t = re.sub(r"<!--.*?-->", "", t, flags=re.S)
    t = re.sub(r"<br\s*/?>", " ", t)
    for _ in range(3):
        t = re.sub(r"\{\{[^{}]*\}\}", "", t)
    t = re.sub(r"\[\[(?:[^\]|]*\|)?([^\]]*)\]\]", r"\1", t)
    t = re.sub(r"<[^>]+>", "", t)
    t = t.replace("'''", "").replace("''", "").replace("&nbsp;", " ").replace("&amp;", "&")
    return re.sub(r"\s+", " ", t).strip()


def split_attrs(cell):
    """'rowspan="2" style="..." | text' -> (attrs, text). Pipes inside links or templates are not separators."""
    m = re.match(r"^([^\[\{|]*?)\|(?!\|)(.*)$", cell, flags=re.S)
    if m and "=" in m.group(1):
        return m.group(1), m.group(2)
    return "", cell


def span(attrs, name):
    m = re.search(name + r'\s*=\s*"?(\d+)', attrs)
    return int(m.group(1)) if m else 1


def parse_wikitext(text, year):
    """List of (release_date or None, artist, album) from the album tables in a wikitext page."""
    rows_out = []
    for block in re.split(r"(?m)^\{\|", text)[1:]:
        block = block.split("\n|}")[0]
        rows = re.split(r"(?m)^\|-.*$", block)
        cols, carry, cur = None, {}, None
        for row in rows:
            cells = []
            for line in row.split("\n"):
                if line.startswith("|+") or line.startswith("{|"):
                    continue
                if line[:1] in "!|" and line.strip():
                    head = line[0] == "!"
                    for part in re.split(r"!!" if head else r"\|\|", line[1:]):
                        cells.append([head, part])
                elif cells and line.strip():
                    cells[-1][1] += "\n" + line
            if not cells:
                continue
            if cols is None:
                names = [clean(split_attrs(c)[1]).lower() for h, c in cells]
                if "artist" in names and "album" in names:
                    cols = names
                continue
            vals, free, ci = {}, [c for c in range(len(cols)) if c not in carry], 0
            for c, (rem, v) in list(carry.items()):
                vals[c] = v
                carry[c] = (rem - 1, v)
                if rem - 1 <= 0:
                    del carry[c]
            for head, cell in cells:
                if ci >= len(free):
                    break
                attrs, body = split_attrs(cell)
                v = clean(body)
                for k in range(span(attrs, "colspan")):
                    if ci + k < len(free):
                        vals[free[ci + k]] = v
                        if span(attrs, "rowspan") > 1:
                            carry[free[ci + k]] = (span(attrs, "rowspan") - 1, v)
                ci += span(attrs, "colspan")
            rec = {cols[c]: v for c, v in vals.items() if c < len(cols)}
            artist, album = rec.get("artist"), rec.get("album")
            if not artist or not album or artist.lower() == "artist":
                continue
            rows_out.append((to_date(rec.get("release date"), year), artist, album))
    return rows_out


def to_date(s, year):
    if not s:
        return None
    m = re.match(r"([A-Za-z]+)\.?\s*(\d{1,2})\b", s)
    if not m or m.group(1).capitalize() not in MONTHS:
        return None
    try:
        return date(year, MONTHS.index(m.group(1).capitalize()) + 1, int(m.group(2)))
    except ValueError:
        return None


def parse_all(years):
    rows = []
    for year in years:
        snaps_by_day = {}
        for title in pages_for(year):
            revs = list_revisions(title, year)
            for day, rid in daily_snapshots(revs, year).items():
                p = rev_path(title, rid)
                if p.exists():
                    snaps_by_day.setdefault(day, []).append((title, rid, p))
        seen, last_day = {}, max(snaps_by_day) if snaps_by_day else None
        cache = {}
        for day in sorted(snaps_by_day):
            present = {}
            for title, rid, p in snaps_by_day[day]:
                if rid not in cache:
                    with gzip.open(p, "rt", encoding="utf-8") as f:
                        cache[rid] = parse_wikitext(f.read(), year)
                for d, a, al in cache[rid]:
                    present[(a, al)] = (d, title)
            for k, (d, title) in present.items():
                s = seen.setdefault(k, {"days": [], "date": d, "page": title})
                s["days"].append(day)
                s["date"], s["page"] = d, title
            if len(cache) > 400:
                cache.clear()
        for (a, al), s in seen.items():
            days = s["days"]
            # first day that starts a run of at least two consecutive snapshots (or the last snapshot)
            first = days[0]
            for i, d in enumerate(days):
                nxt = days[i + 1] if i + 1 < len(days) else None
                if d == last_day or (nxt is not None and (nxt - d).days == 1):
                    first = d
                    break
            rows.append((year, a, al, s["date"], first, days[-1], len(days), s["page"]))
        print(f"{year}: {sum(1 for r in rows if r[0] == year)} albums")
    df = pd.DataFrame(rows, columns=["list_year", "artist", "album", "release_date", "first_seen", "last_seen",
                                     "days_seen", "page"])
    df.sort_values(["list_year", "first_seen", "artist"]).to_csv(OUT, index=False, compression="gzip")
    print(f"wrote {len(df):,} rows to {OUT.relative_to(ROOT)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--parse", action="store_true")
    ap.add_argument("--years", nargs="+", type=int, default=list(range(2017, 2027)))
    a = ap.parse_args()
    if a.list:
        for y in a.years:
            for t in pages_for(y):
                list_revisions(t, y)
    if a.fetch:
        fetch_all(a.years)
    if a.parse:
        parse_all(a.years)


if __name__ == "__main__":
    main()
