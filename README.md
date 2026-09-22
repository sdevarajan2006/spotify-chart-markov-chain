# spotify-chart-markov-chain

Modeling daily movement between chart positions on Spotify's US Daily Top 200 with a Markov chain.

## Project overview

- **States:** 1-10, 11-20, 21-30, 31-40, 41-50, 51-100, 101-150, 151-200, off-chart
- **Frequency:** daily transitions, 2017-01-01 to 2026-09-17
- **Goal:** understand how songs move through the chart, as groundwork for predicting the daily #1

## Repo layout

```
data/spotify_us_daily_top200_2017_2026.csv.gz   dataset (709,397 rows, 13 MB)
data/coverage_gap_log.txt                       coverage check for the build
scripts/build_us_top200.py                      rebuilds the dataset from the Kaggle source
notebooks/analysis.ipynb                        transition matrix, heatmap, chain graphs
```

## Data

Source: Kaggle [Spotify Charts Daily Updated](https://www.kaggle.com/) (gonzalopezgil), filtered to `country = us`.
Every day in the range is present. Only 2024-05-20, 21 and 22 have 199 rows instead of 200.

| Column | Type | Nulls | Description |
|---|---|---|---|
| chart_date | date | none | Chart day |
| rank | int 1-200 | none | Position that day |
| prev_rank | int | not on yesterday's chart | Rank the previous day |
| peak_rank | int | none | Best rank so far, including today |
| streak | int | debut and re-entry days | Consecutive days on chart minus 1 (second straight day = 1) |
| days_on_chart | int | none | Total days on chart so far, counting gaps |
| streams | int | none | US streams that day |
| track_name | string | none | Track title |
| artists | string | none | All credited artists, separated by `\|` |
| track_uri | string | none | `spotify:track:<id>`, the song ID |
| release_date | date | 9,677 rows | Release date of the track's current release |
| labels | string | 26 rows | Record label(s) |

`(chart_date, rank)` is unique. peak_rank, streak and days_on_chart come from Spotify and are tracked per track_uri, so different versions of a song count separately.

**Caveat:** release_date belongs to the release the track currently sits on, often an album that came out after the single. 45,872 rows have a release_date later than chart_date, so don't compute days since release from it directly.

### Rebuilding

Download `charts_songs_daily.csv` and `songs.csv` from the Kaggle dataset, then:

```
python scripts/build_us_top200.py --charts ~/Downloads/charts_songs_daily.csv \
    --songs ~/Downloads/songs.csv --end 2026-09-17
```

## Method

Each song is tracked from its first chart day to the end of the data. Days it isn't on the chart are off-chart.
The notebook counts transitions directly instead of building the roughly 17M-row song-by-day panel. The counts are identical.

## Next steps

- Add a "not yet entered" state for songs before their first chart day
- Compare weekly transitions with daily ones
- Handle songs already on the chart on 2017-01-01, which enter directly at their rank
- Account for regime shift: the 2026 chart is far more catalog-heavy than 2017-2021
