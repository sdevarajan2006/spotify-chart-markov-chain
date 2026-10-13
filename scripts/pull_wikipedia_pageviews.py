"""Pull daily English Wikipedia page views for every charting artist, and for
album pages that match a release.

Run from the repo root (needs internet; standard library + pandas only):
    python scripts/pull_wikipedia_pageviews.py --limit 5     # quick test on 5 artists
    python scripts/pull_wikipedia_pageviews.py               # full run (~1,000 artists + albums)
    python scripts/pull_wikipedia_pageviews.py --parse-only  # rebuild the CSVs from saved raw files

Steps
1. Find each artist's Wikipedia article through Wikidata: search the name and keep
   the first match that has a MusicBrainz artist ID (property P434), so "Drake"
   resolves to the musician, not the duck. Results go to
   data/wikipedia_artist_titles.csv. Check the rows with match = "none" or an odd
   description, and put fixes in data/wikipedia_title_overrides.csv
   (columns: name, enwiki_title). Overrides always win.
2. Find each album page the same way: an album-type item whose performer is the
   artist, or whose description names the artist. Results go to
   data/wikipedia_album_titles.csv.
3. Download daily views (users only, all devices) from 2016-07-01 on.
   Raw responses are saved under data/raw/pageviews/, so reruns skip finished pages.

Output: data/wikipedia_pageviews_daily.csv.gz with columns
    kind (artist/album), name, enwiki_title, date, views
A page with no views on a day (including days before it was created) has no row.
"""
import argparse
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw" / "pageviews"
FEATURES = DATA / "release_features.csv"
ALIASES = DATA / "artist_aliases.csv"
OVERRIDES = DATA / "wikipedia_title_overrides.csv"
ARTIST_TITLES = DATA / "wikipedia_artist_titles.csv"
ALBUM_TITLES = DATA / "wikipedia_album_titles.csv"
OUT = DATA / "wikipedia_pageviews_daily.csv.gz"

UA = "spotify-chart-markov-chain/0.1 (https://github.com/sdevarajan2006/spotify-chart-markov-chain)"
WD_API = "https://www.wikidata.org/w/api.php"
PV_API = "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/user"
START, END = "20160701", date.today().strftime("%Y%m%d")

# Wikidata item types that count as an album-like release
ALBUM_TYPES = {"Q482994", "Q208569", "Q169930", "Q217199", "Q222910", "Q209939", "Q1242743", "Q23880345"}


def get_json(url, tries=4):
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if e.code == 429 or e.code >= 500:
                time.sleep(5 * (k + 1))
                continue
            raise
        except (urllib.error.URLError, TimeoutError):
            time.sleep(5 * (k + 1))
    raise RuntimeError(f"failed: {url}")


def wd(**params):
    params.update(format="json", formatversion="2")
    return get_json(WD_API + "?" + urllib.parse.urlencode(params))


def wd_entities(ids):
    if not ids:
        return {}
    out = wd(action="wbgetentities", ids="|".join(ids), props="claims|sitelinks|descriptions|labels",
             sitefilter="enwiki", languages="en")
    return (out or {}).get("entities", {})


def claim_ids(ent, prop):
    vals = []
    for c in ent.get("claims", {}).get(prop, []):
        v = c.get("mainsnak", {}).get("datavalue", {}).get("value")
        if isinstance(v, dict) and "id" in v:
            vals.append(v["id"])
        elif v is not None:
            vals.append(v)
    return vals


def enwiki(ent):
    return ent.get("sitelinks", {}).get("enwiki", {}).get("title")


def desc(ent):
    return ent.get("descriptions", {}).get("en", {}).get("value", "")


def resolve_artist(name):
    hits = (wd(action="wbsearchentities", search=name, language="en", type="item", limit=10) or {}).get("search", [])
    ents = wd_entities([h["id"] for h in hits])
    for h in hits:  # keep search order
        e = ents.get(h["id"], {})
        if claim_ids(e, "P434") and enwiki(e):
            return {"qid": h["id"], "enwiki_title": enwiki(e), "description": desc(e), "match": "musicbrainz"}
    for h in hits:
        e = ents.get(h["id"], {})
        if enwiki(e) and re.search(r"singer|rapper|musician|band|group|duo|DJ|producer|songwriter", desc(e), re.I):
            return {"qid": h["id"], "enwiki_title": enwiki(e), "description": desc(e), "match": "description"}
    return {"qid": None, "enwiki_title": None, "description": None, "match": "none"}


def resolve_album(album, artist, artist_qid):
    hits = (wd(action="wbsearchentities", search=album, language="en", type="item", limit=10) or {}).get("search", [])
    ents = wd_entities([h["id"] for h in hits])
    for h in hits:
        e = ents.get(h["id"], {})
        if not enwiki(e) or not (set(claim_ids(e, "P31")) & ALBUM_TYPES):
            continue
        if artist_qid and artist_qid in claim_ids(e, "P175"):
            return {"qid": h["id"], "enwiki_title": enwiki(e), "description": desc(e), "match": "performer"}
        if artist.lower() in desc(e).lower():
            return {"qid": h["id"], "enwiki_title": enwiki(e), "description": desc(e), "match": "description"}
    return {"qid": None, "enwiki_title": None, "description": None, "match": "none"}


def load_inputs():
    F = pd.read_csv(FEATURES, parse_dates=["first_chart_date"])
    names = set(F.hist_artist.dropna()) | set(F.artist_used.dropna())
    if ALIASES.exists():  # members of joint projects are already canonical names; add them too
        for v in pd.read_csv(ALIASES).credit_to:
            names |= set(v.split("|"))
    albums = (F[F.wikipedia_album.notna()][["hist_artist", "wikipedia_album", "first_chart_date"]]
              .drop_duplicates(["hist_artist", "wikipedia_album"]))
    return sorted(names), albums


def overrides():
    if OVERRIDES.exists():
        o = pd.read_csv(OVERRIDES)
        return dict(zip(o.name, o.enwiki_title))
    return {}


def resolve_all(names, albums, limit=None):
    ov = overrides()
    done = pd.read_csv(ARTIST_TITLES) if ARTIST_TITLES.exists() else pd.DataFrame(columns=["name"])
    rows = done.to_dict("records")
    have = set(done.name)
    todo = [n for n in names if n not in have][: limit if limit else None]
    for i, n in enumerate(todo, 1):
        r = resolve_artist(n)
        r["name"] = n
        rows.append(r)
        if i % 25 == 0:
            print(f"  artists resolved: {i}/{len(todo)}")
            pd.DataFrame(rows).to_csv(ARTIST_TITLES, index=False)
        time.sleep(0.2)
    A = pd.DataFrame(rows)
    A["enwiki_title"] = [ov.get(n, t) for n, t in zip(A.name, A.enwiki_title)]
    A.loc[A.name.isin(ov), "match"] = "override"
    A.to_csv(ARTIST_TITLES, index=False)
    print(f"artists: {len(A)} total, {A.enwiki_title.isna().sum()} without a page "
          f"(see {ARTIST_TITLES.name}; add fixes to {OVERRIDES.name})")

    qid = dict(zip(A.name, A.qid))
    done = pd.read_csv(ALBUM_TITLES) if ALBUM_TITLES.exists() else pd.DataFrame(columns=["artist", "album"])
    rows = done.to_dict("records")
    have = set(zip(done.artist, done.album))
    todo = [r for r in albums.itertuples() if (r.hist_artist, r.wikipedia_album) not in have]
    if limit:
        todo = [r for r in todo if r.hist_artist in set(A.name)][:limit]
    for i, r in enumerate(todo, 1):
        res = resolve_album(r.wikipedia_album, r.hist_artist, qid.get(r.hist_artist))
        res.update(artist=r.hist_artist, album=r.wikipedia_album, first_chart_date=r.first_chart_date.date())
        rows.append(res)
        if i % 25 == 0:
            print(f"  albums resolved: {i}/{len(todo)}")
            pd.DataFrame(rows).to_csv(ALBUM_TITLES, index=False)
        time.sleep(0.2)
    B = pd.DataFrame(rows)
    B.to_csv(ALBUM_TITLES, index=False)
    print(f"albums: {len(B)} total, {B.enwiki_title.isna().sum()} without a page")
    return A, B


def raw_path(kind, title):
    safe = re.sub(r"[^\w\-]+", "_", title)[:150]
    return RAW / kind / f"{safe}.json"


def download(kind, titles):
    (RAW / kind).mkdir(parents=True, exist_ok=True)
    titles = [t for t in dict.fromkeys(titles) if isinstance(t, str)]
    for i, t in enumerate(titles, 1):
        p = raw_path(kind, t)
        if p.exists():
            continue
        art = urllib.parse.quote(t.replace(" ", "_"), safe="")
        js = get_json(f"{PV_API}/{art}/daily/{START}/{END}")
        p.write_text(json.dumps({"title": t, "response": js}))
        if i % 50 == 0:
            print(f"  {kind} pages downloaded: {i}/{len(titles)}")
        time.sleep(0.1)


def parse(A, B):
    out = []
    for kind, df, name_col in [("artist", A, "name"), ("album", B, "album")]:
        for name, t in zip(df[name_col], df.enwiki_title):
            if not isinstance(t, str):
                continue
            p = raw_path(kind, t)
            if not p.exists():
                continue
            js = json.loads(p.read_text())["response"] or {}
            for it in js.get("items", []):
                out.append((kind, name, t, it["timestamp"][:8], it["views"]))
    P = pd.DataFrame(out, columns=["kind", "name", "enwiki_title", "date", "views"])
    P["date"] = pd.to_datetime(P.date, format="%Y%m%d").dt.date
    P.to_csv(OUT, index=False, compression="gzip")
    print(f"wrote {len(P):,} rows for {P[P.kind == 'artist'].name.nunique()} artists and "
          f"{P[P.kind == 'album'].name.nunique()} albums to {OUT.relative_to(ROOT)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, help="only resolve/download this many new artists (for testing)")
    ap.add_argument("--parse-only", action="store_true")
    a = ap.parse_args()
    names, albums = load_inputs()
    if a.parse_only:
        parse(pd.read_csv(ARTIST_TITLES), pd.read_csv(ALBUM_TITLES))
        return
    A, B = resolve_all(names, albums, a.limit)
    download("artist", A.enwiki_title.tolist())
    download("album", B.enwiki_title.tolist())
    parse(A, B)


if __name__ == "__main__":
    main()
