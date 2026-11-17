# CLAUDE.md

Context for Claude Code sessions in this repo. Last updated 2026-11-17.

The README describes the original Markov chain work and is out of date. This file covers the current goal: predicting the daily US #1 on Spotify.

## Goal

Predict which song is #1 on Spotify's US Daily Top 200 on day D, using only information available from the chart dated D-2. This is a standalone prediction model; Kalshi contract structure is set aside.

## Ground rules

- **Information cutoff: the chart dated D-2.** Spotify's charting day is UTC and the chart is usually published by about 6 PM Eastern the next day, so D-1 is not available when predicting D. Every feature must be computable from data through D-2.
- **Free data only.** No paid data sources.
- **Do not use the live Spotify charts API** (`charts-spotify-com-service.spotify.com/...`). It needs a replayed personal bearer token, which is against Spotify's terms and risks the account.
- **Feature selection on validation years only.** Train/validate on 2017-2023 with rolling years (each of 2019-2023 predicted from earlier years), compare models with paired comparisons on log loss, and score the 2024-2026 test set once at the end. Earlier Part 1 choices were made on the test set, so those test scores are optimistic. Chart data after 2026-09-17 has never been looked at and could serve as an untouched check.
- **release_date is unreliable for timing.** It is the date of the release a track currently sits on (often a later album); 45,872 rows have release_date after chart_date. Do not compute days since release from it.
- Match songs on track name + artists, not track_uri, when checking whether "the same song" held #1 (re-uploads change URIs).
- Writing style for notes and docs: concise, no em dashes.

## Repo layout

```
data/spotify_us_daily_top200_2017_2026.csv.gz   main dataset (709,397 rows, 2017-01-01 to 2026-09-17, full top 200)
data/coverage_gap_log.txt                       coverage check for the build
notebooks/number1_source_distribution.ipynb     #1 source analysis (sections 1-9)
notebooks/analysis.ipynb                        original binned-state Markov chain
archive/                                        older material
scripts/build_us_top200.py                      rebuilds the dataset from the Kaggle source
scripts/build_release_features.py               -> data/release_features.csv
scripts/debut_forecaster.py                     -> data/debut_forecast_results.csv + test predictions (older 17-feature config, see Debut strength forecaster)
scripts/grid_search_debut.py                    -> data/debut_grid_search.csv
scripts/validate_debut_models.py                rolling validation on 2019-2023 of last-release baseline, OLS, ridge and gradient boosting (17 features) -> data/debut_validation_results.csv, data/debut_validation_by_year.csv
scripts/part1_overlay.py                        Part 1 with debut threat; also fits the final 14-feature forecaster -> data/part1_overlay_results.csv, part1_overlay_test_days.csv, debut_forecast_release_predictions.csv
scripts/check_announcements.py                  announced-ahead check -> data/release_announcements.csv
scripts/pull_wikipedia_albums.py                Wikipedia "List of YYYY albums" pull
scripts/pull_wikipedia_pageviews.py             album page views (--retry-unmatched also searches Wikidata)
scripts/pull_gdelt_news.py                      daily counts of news articles mentioning each charting artist (GDELT DOC 2.0, 1 request per 5.5 s), only for the 120 days before each of the artist's releases -> data/gdelt_news_daily.csv.gz. Not run yet (no output or data/raw/gdelt), and not used by any model
```

Other data files: `release_groups.csv` (one row per release; `new_release == True` for 4,647 new releases), `baseline_count_table_d2.csv`, `number1_prev_rank_distribution_h1_h10.csv`, `announcement_overrides.csv` and `wikipedia_album_title_overrides.csv` (manual fixes; neither file exists in data/ yet, the scripts read them if present), `tracks_with_release_dates.csv` (manual release-date fill, not merged yet).

Dataset columns: chart_date, rank, prev_rank, peak_rank, streak, days_on_chart, streams, track_name, artists (pipe-separated, all credited), track_uri, release_date, labels. `(chart_date, rank)` is unique. 11 tracks removed by Spotify are labelled `removed_1` to `removed_11`.

## Git

Updated 2026-11-17: 2ea7776 and 63b8a2d are on origin, and PRs #1 and #2 merged `number1-source-analysis` into `main`. Older note below is kept for history.


Branch `number1-source-analysis` holds the #1 work. As of 2026-11-08, commit 288e84b was pushed and 2ea7776 (notebook sections 7-9, requirements) was committed locally but not pushed. Not merged into `main` yet. Check `git status` before assuming.

## Model structure

Two parts, no off-chart state:
1. **Part 1:** does the D-2 #1 still hold #1 on day D? (logistic regression)
2. **If not, who takes over.** Replaced by the debut strength forecaster, since debuts are the main threat that history can't see on the chart.

**Baseline:** "the #1 on D-2 is #1 on D" is right 77.7% of the time. Of takeovers: #2 41%, debut 34%, #3 or lower 25%.

## Part 1 (logistic regression)

Train 2017-2023 (2,554 days, 79.0% hold), test 2024-2026 (991 days, 74.3% hold, 255 change days). Results so far (test set, to be redone on validation years):

| Model | Test log loss | Changes caught |
|---|---|---|
| Constant 79% | 0.577 | 0 |
| Log lead | 0.520 | 27 |
| Log lead + debut flag | 0.507 | 38 |
| + Friday flag | 0.494 | 61 |
| + debut threat, day D and D-1 releases (surprise drops excluded) | 0.484 | 65 |

### Feature list (agreed 2026-11-17)

All measured on the D-2 chart unless noted.

| # | Feature | Definition | Previous status |
|---|---|---|---|
| 1 | Lead | #1 streams / #2 streams - 1 | Replaced by log lead |
| 2 | Log lead | Lead on a log scale | In model |
| 3 | Gap to #3 | Try all three: log(#2/#3), log(#1/#3), #1/#3 - 1 | Queued |
| 4, 8, 9 | How the #1's run stands (one categorical) | Debuted at #1 that day (= old debut flag, days_on_chart == 1) / climbed to #1 that day / regained #1 that day / already #1 the day before (reference) | Debut flag in model; climbed and regained rejected |
| 5 | Day 2 flag | #1 has days_on_chart = 2 | Rejected |
| 6 | Day 3 flag | #1 has days_on_chart = 3 | Rejected |
| 10 | Growth gap, k days | #2's stream growth minus #1's over the last k days (k = 1 to 10), capped at +/-50% | Rejected (strongest at k = 7 in training) |
| 11 | Friday flag | Day D is a Friday | Helped; mostly covered by debut threat |
| 12 | Debut threat, day D | 1 - prod(1 - p_i) over releases first charting on day D | On hold until release calendar |
| 13 | Debut threat, D and D-1 | Same, including releases charting on D-1 | On hold until release calendar |
| 14 | Artist top-10 count | Number of top-10 songs by the #1's artist | Queued |
| 15 | Dethroned #1 nearby | A song at #2 or #3 that was #1 at some point in the previous 7 days; separately, 14 days. Window chosen on validation years | Queued |

- #7 ("first day at #1 of any kind") is dropped; the categorical implies it.
- #3: with log lead in the model, the two log versions give identical fits (log(#1/#3) = log lead + log(#2/#3)).
- #12-13 use releases not visible at the D-2 cutoff, and the current release list only contains releases that charted. They need a release calendar.

## Debut strength forecaster

Predicts a new release's log day-one streams (its top song) before release; strength = predicted streams / forecast of the day-D leader. Feeds P(beats leader) into Part 1 as the debut threat.

- Gradient boosting (learning rate 0.1, 7 leaves, min 60 per leaf), 14 features: artist track record (last release, last same-type release, best of last 3, best in 2 years, best ever, weighted average, release count, prior #1 debuts, days since last release), is_album (matched to Wikipedia "List of YYYY albums", known before release), number of credited artists, Friday, market size. Chosen by grid search over 527 candidates on validation years 2019-2023. This final config is implemented in `part1_overlay.py`; `debut_forecaster.py` still runs the earlier 17-feature version (adds page-view features, learning rate 0.04, 15 leaves, min 40 per leaf), and its saved results (gbm log loss 0.116, error 1.38x, 9 of 42 caught) differ from the numbers below.
- Test 2024-2026 (825 releases, 42 beat the leader): AUC 0.90 (last-release baseline 0.82), log loss 0.133, typical error 1.43x. After excluding surprise drops: AUC 0.904.
- Leader forecast = D-2 #1 streams plus a Friday adjustment (error 1.10x). ARIMA and seasonal ARIMA did not beat it.
- Surprise releases (album Wikipedia page first viewed on D-1 or later) are excluded from the forecaster and the Part 1 threat input (`--keep-surprise` restores them). Singles are currently assumed to be announced ahead.

## Next steps

1. Push commit 2ea7776 and merge `number1-source-analysis` into `main` via pull request.
2. **Build a release calendar (next major task, before more Part 1 modeling).** For each day D, list every release publicly known by D-2: artist, release date, album or single. Needs a historical version (2017-2026) for backtesting and a live version for upcoming releases. Must include releases that did not chart. Singles are the gap: they are most of the 4,647 releases and currently have no announcement data. Free sources only.
3. Redo Part 1 feature selection on validation years using the feature list above.
4. Check suspect surprise-drop flags (renamed or late-created Wikipedia pages such as I Am > I Was and Hozier's EPs; short-notice announcements such as JackBoys) and the 30 top-10 album debuts with no Wikipedia match.
5. Merge `data/tracks_with_release_dates.csv` (needs a rule for year-only and month-only dates).
