"""Pull album release dates from Wikipedia's yearly "List of YYYY albums" pages.

Step 1 downloads each page from the Wikipedia API and saves the raw response
under data/raw/wikipedia/. Step 2 parses the saved pages into one CSV. Saving
the raw pages means the parsing can be rerun or fixed without downloading again.

Usage (from the repo root):
    python scripts/pull_wikipedia_albums.py                # 2017 to the current year
    python scripts/pull_wikipedia_albums.py 2024 2026      # a range of years
    python scripts/pull_wikipedia_albums.py --parse-only   # reparse saved pages, no download

Writes data/wikipedia_album_releases.csv with one row per album:
    release_date, date_text, artist, album, genre, label, list_year, source_page, pulled_on
release_date is blank for releases listed without a confirmed day.
"""
import io
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path

import pandas as pd

API = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "spotify-chart-markov-chain/0.1 (https://github.com/sdevarajan2006/spotify-chart-markov-chain)"
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "wikipedia"
OUT = ROOT / "data" / "wikipedia_album_releases.csv"


def api(**params):
    params.update(format="json", formatversion="2")
    url = API + "?" + urllib.parse.urlencode(params)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def raw_path(title):
    return RAW / (re.sub(r"[^A-Za-z0-9]+", "_", title).strip("_") + ".json")


def download(year):
    """Save the main list page for a year, plus any linked sub-pages for the same year."""
    try:
        hits = api(action="query", list="prefixsearch", pssearch=f"List of {year} albums", pslimit="20")
        queue = [h["title"] for h in hits["query"]["prefixsearch"] if h["title"].startswith(f"List of {year} albums")]
    except Exception as error:
        print(f"  search failed for {year}: {error}")
        queue = []
    if f"List of {year} albums" not in queue:
        queue.insert(0, f"List of {year} albums")
    seen = set()
    while queue:
        title = queue.pop(0)
        if title in seen:
            continue
        seen.add(title)
        try:
            page = api(action="parse", page=title, prop="text|wikitext|links", redirects="1")
        except Exception as error:
            print(f"  could not download {title}: {error}")
            continue
        if "parse" not in page:
            print(f"  no page content for {title}: {str(page)[:200]}")
            continue
        page["list_year"] = year
        page["requested_title"] = title
        raw_path(title).write_text(json.dumps(page), encoding="utf-8")
        html = page["parse"].get("text", "")
        print(f"  saved {title} ({len(html):,} characters of HTML)")
        for link in page["parse"].get("links", []):
            linked = link.get("title", "")
            if f"{year} albums" in linked and linked not in seen:
                queue.append(linked)
        time.sleep(1)   # be polite to the API


def clean(value):
    if pd.isna(value):
        return None
    text = re.sub(r"\[[^\]]*\]", "", str(value)).strip()   # drop footnote markers such as [12]
    return text or None


def to_date(text, year):
    if not text:
        return pd.NaT
    match = re.match(r"([A-Za-z]+)\s*(\d{1,2})", text)
    if not match:
        return pd.NaT
    return pd.to_datetime(f"{match.group(1)} {match.group(2)} {year}", format="%B %d %Y", errors="coerce")


def read_tables(html):
    try:
        return pd.read_html(io.StringIO(html))
    except Exception:           # pandas raises when the page has no tables
        return []


def find_column(columns, word):
    for key, original in columns.items():
        if key == word or key.startswith(word):
            return original
    return None


def promote_header(table):
    """The list tables start with a navigation row, so the real header sits a row or two down."""
    names = {str(c).strip().lower() for c in table.columns}
    if "artist" in names and "album" in names:
        return table
    for i in range(min(5, len(table))):
        values = [str(v).strip() for v in table.iloc[i].tolist()]
        lowered = [v.lower() for v in values]
        if "artist" in lowered and "album" in lowered:
            fixed = table.iloc[i + 1:].copy()
            fixed.columns = [v if v.lower() != "nan" else f"extra_{n}" for n, v in enumerate(values)]
            return fixed
    return table


def parse_tables(html, year, title):
    rows = []
    for table in read_tables(html):
        if isinstance(table.columns, pd.MultiIndex):
            table.columns = [str(col[-1]) for col in table.columns]   # keep the lowest header row
        table = promote_header(table)
        cols = {str(c).strip().lower().rstrip("."): c for c in table.columns}
        artist_col, album_col = find_column(cols, "artist"), find_column(cols, "album")
        if artist_col is None or album_col is None:
            continue
        date_col = next((cols[k] for k in cols if "date" in k), None)
        genre_col, label_col = find_column(cols, "genre"), find_column(cols, "label")
        for _, r in table.iterrows():
            rows.append({
                "date_text": clean(r[date_col]) if date_col is not None else None,
                "artist": clean(r[artist_col]),
                "album": clean(r[album_col]),
                "genre": clean(r[genre_col]) if genre_col is not None else None,
                "label": clean(r[label_col]) if label_col is not None else None,
                "list_year": year,
                "source_page": title,
            })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out[out["artist"].notna() & out["album"].notna() & (out["artist"] != "Artist")
              & ~out["artist"].fillna("").str.startswith("Go to:")].copy()
    out["release_date"] = [to_date(t, year) for t in out["date_text"]]
    return out


def describe_problem(page):
    """Print what the page contains when no album tables were found."""
    parsed = page["parse"]
    tables = read_tables(parsed.get("text", ""))
    print(f"    tables on the page: {len(tables)}")
    for table in tables[:6]:
        print(f"    table columns: {[str(c) for c in table.columns][:8]}  rows: {len(table)}")
    print(f"    wikitext starts: {parsed.get('wikitext', '')[:400]!r}")
    related = [l.get("title", "") for l in parsed.get("links", []) if "albums" in l.get("title", "").lower()]
    print(f"    links mentioning albums: {related[:15]}")


def parse_saved(first, last):
    frames = []
    for path in sorted(RAW.glob("*.json")):
        page = json.loads(path.read_text(encoding="utf-8"))
        year = page.get("list_year")
        if year is None or not first <= year <= last:
            continue
        title = page["parse"].get("title", page.get("requested_title", path.stem))
        frame = parse_tables(page["parse"].get("text", ""), year, title)
        dated = int(frame["release_date"].notna().sum()) if not frame.empty else 0
        print(f"{title}: {len(frame):,} albums ({dated:,} with a date)")
        if frame.empty:
            describe_problem(page)
        else:
            frames.append(frame)
    if not frames:
        print("No albums parsed. Share the lines above so the parser can be fixed.")
        return
    albums = pd.concat(frames, ignore_index=True).drop_duplicates(["artist", "album", "list_year"])
    albums["pulled_on"] = date.today().isoformat()
    albums = albums.sort_values(["list_year", "release_date", "artist"])
    cols = ["release_date", "date_text", "artist", "album", "genre", "label", "list_year", "source_page", "pulled_on"]
    albums[cols].to_csv(OUT, index=False)
    print(f"Wrote {len(albums):,} albums to {OUT}")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    parse_only = "--parse-only" in sys.argv
    first = int(args[0]) if len(args) > 0 else 2017
    last = int(args[1]) if len(args) > 1 else date.today().year
    RAW.mkdir(parents=True, exist_ok=True)
    if not parse_only:
        for year in range(first, last + 1):
            print(f"Downloading {year}")
            download(year)
    parse_saved(first, last)


if __name__ == "__main__":
    main()
