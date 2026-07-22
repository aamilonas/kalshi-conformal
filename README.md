# Kalshi Conformal Recalibration — Data Foundation (Phase 0–1)

Data engineering for the paper *"Are Prediction Market Prices Valid Probabilities?
Distribution-Free Recalibration and Prospective Coverage on Kalshi and Polymarket."*

## Provenance

| Item | Value |
| --- | --- |
| Raw archive | `E:\pm\becker-data\data.tar.zst` (36,020,641,508 bytes, verified == server Content-Length) |
| Archive date | 2026-02-05 (server Last-Modified) — **newer** than Le's 2025-12-31 cutoff |
| Source | https://s3.jbecker.dev/data.tar.zst (Becker, prediction-market-analysis) |
| Becker repo | github.com/Jon-Becker/prediction-market-analysis (reference only) |
| Le repo | github.com/**namanhzz**/prediction-market-calibration @ `143ca873` (note: double-z username) |
| Drive | E: is **exFAT** (not NTFS). Kept as-is — reformatting would have destroyed the completed download. Works fine for this workload; git needs `safe.directory` exceptions (already added). |

## Key reference-repo facts (load-bearing for Steps 7–8)

- Le's 216-cell calibration matrix: `prediction-market-calibration/supplementary/calibration_matrix_216.csv`
  (216 = 6 domains × 9 time bins × 4 size bins; see their `src/config.py`).
- Le's domain rules: `prediction-market-calibration/src/classify.py` (571 patterns).
  Vendored **verbatim** into our `src/le_classify.py`; wrapped by `src/classify_domains.py`.
- Le's Table 1 semantics (`src/pipeline.py::load_kalshi_market_stats`):
  `n_markets` = markets with `status='finalized' AND result IN ('yes','no')` having
  **≥10 trades with `created_time <= 2025-12-31T23:59:59Z`**; `n_trades` = count of those
  trades; `base_rate` = % of those markets resolving `yes`. Le does **not** dedup
  Becker's parquets before these queries.

### Classifier semantics (and a quirk you must not "fix")

Domain = `get_group(prefix)` where `prefix` = leading `[A-Z0-9]+` of `event_ticker`
(`'independent'` if empty). Matching is **ordered, case-insensitive substring**
containment over the 571-pattern list — not prefix equality. Consequences:

- `KXNFLGAME…` → Sports (contains `NFLGAME`) — Kalshi's newer `KX` tickers work automatically.
- **`FEDDECISION` → Politics**, not Finance: the short pattern `"EC"` (Electoral College)
  is listed earlier and substring-matches FEDD**EC**ISION. Same for anything containing `EC`.
- `SBADS` (Super Bowl ads) → Sports via the early `"SB"` pattern.

These quirks are embedded in Le's published Table 1; we reproduce them exactly
(verified by `tests/test_classify.py::test_equivalence_with_le_original`, which
asserts our vendored copy agrees with Le's repo on all 571 patterns).

Our 7-way `classify()` collapses Le's non-analysis groups (Esports, Science/Tech,
World Events, Media, Other) into `Other`; the 6 analysis domains are
Sports, Crypto, Politics, Finance, Weather, Entertainment.

## Raw schema (Becker `docs/SCHEMAS.md` — verified against data in Step 5)

**kalshi/markets** (one row per contract snapshot): `ticker`, `event_ticker`,
`market_type`, `title`, `yes_sub_title`, `no_sub_title`, `status`
(open/closed/finalized), `yes_bid`, `yes_ask`, `no_bid`, `no_ask`, `last_price`,
`volume`, `volume_24h`, `open_interest`, `result` (yes/no/empty), `created_time`,
`open_time`, `close_time`, `_fetched_at`.

**kalshi/trades** (one row per execution): `trade_id`, `ticker`, `count`,
`yes_price` (cents 1–99), `no_price` (= 100 − yes_price), `taker_side`,
`created_time`, `_fetched_at`.

Prices are **cents**; probability = `yes_price / 100.0`.

## Step 4 (extraction)

Streamed `zstd -dc --long=31 | tar -xf -` (no intermediate `data.tar`; the
`.tar.zst` original is retained as backup). Checkpoint counts, raw archive:

- `kalshi/trades/*.parquet`: **72,134,741** rows (Le's snapshot: ~64.7M; ours
  extends ~5 weeks past their 2025-12-31 cutoff — overshoot expected).
- `kalshi/markets/*.parquet`: **7,682,445** rows — and Step 5 shows this is
  **7,682,445 distinct tickers** (max 1 row/ticker). Kalshi genuinely listed
  ~7.7M contracts (2025 hourly crypto/sports explosion); an earlier draft of
  this note wrongly guessed "30 snapshots per ticker". Dedup in Step 6 is a
  formal no-op on this archive but is still run as a guard.

## Step 5 findings (schema inspection)

Columns match Becker's `docs/SCHEMAS.md` exactly (markets: 20 cols incl.
`no_bid`/`no_ask`/`volume_24h`; trades: 8 cols). Dtypes: prices/volumes BIGINT;
`created_time`/`open_time`/`close_time` are **`TIMESTAMP WITH TIME ZONE`**
(UTC instants; DuckDB renders them in the session tz), `_fetched_at` is naive
`TIMESTAMP_NS`.

**Silent killer 1 — duplicates: NONE.** 7,682,445 market rows = 7,682,445
distinct tickers; 72,134,741 trade rows = 72,134,741 distinct `trade_id`s.
Step 6 dedup is a no-op guard.

**Silent killer 2 — timezone: verified UTC instants.** PRES-2024 (election
winner) closes at `2025-01-20 12:03 ET` = inauguration noon, and its final
trades print DJT at 99c minutes before close — externally correct. Trades span
`2021-06-30` → **`2025-11-25`** (content end). NB: content ends BEFORE Le's
2025-12-31 cutoff, so if Le pulled this same archive vintage their cutoff
excludes nothing and Step 8 should reconcile near-exactly.

**Silent killer 3 — price units: cents (0–99), confirmed.** `p = yes_price/100`.
Caveat: 267,291 trades (0.37%) have `yes_price + no_price != 100` (incl.
`yes_price = 0` rows); the primary `[0.05, 0.95]` filter drops these edges.

**Status taxonomy** is richer than the docs: `finalized` (7,320,904; of which
7,314,375 have result yes/no), `active` (328,865), `initialized` (20,536),
`closed` (11,788), `inactive` (342), `determined` (9), `disputed` (1).
Le's resolved-market filter is `status='finalized' AND result IN ('yes','no')`.
Max `close_time` is 2099-08-01 (far-future placeholder on unresolved markets).

## Step 10 findings (sanity suite — 33 tests green)

- **DST bug found and fixed by test 1.** The original build used
  `INTERVAL '7 days'/'30 days'`; on TIMESTAMPTZ that is calendar arithmetic in
  the session timezone, so across spring-forward "7 days" = 167 real hours —
  85 forecasts (1w/1mo only, gaps 167.0–167.96h / 719.1–719.9h) leaked past
  the cutoff. Fix: horizons in absolute hours (`168 hours`, `720 hours`) +
  `SET TimeZone='UTC'`. τ is defined as absolute duration everywhere.
- **Base rates match to 0.000pp in all 7 domains** against the matched
  population (resolved, ≥10 trades, first trade ≥1h pre-close) recomputed
  independently of the ASOF join. NB: Step 8/Le base rates are NOT the right
  comparator for τ=1h in hourly-heavy domains — only 13,126 of 76,181
  ≥10-trade Crypto markets have any trade 1h pre-close (37.3% vs 40.7% yes).
  This selection effect is structural, not a bug.
- **Another substring quirk found:** `ARGINFLATIONM` (Argentina inflation) is
  labeled **Sports** by Le's classifier (ARGI"NFL"ATIONM ⊃ NFL). Kept as-is —
  reconciliation requires Le's exact labels.
- τ=1h price mass in [0,0.1)∪(0.9,1] is 69.5% (short horizons pile at the
  extremes, as expected); see `results/fig_price_hists.png`.
- `results/spotcheck.txt`: PRES-2024-DJT, KXNFLGAME-25AUG23HOUDET-HOU,
  HIGHAUS-23AUG01-B104.5, BTC-24AUG0917-B60500, AAAGASM-23DEC31-US-3.246 —
  the selected row is programmatically verified to be the genuinely last trade
  before cutoff at every τ (or correctly absent). Dump timestamps render in
  ET (−04/−05); the "OK:" lines render UTC (+00:00) — same instants.

## Pipeline

```
src/inspect_data.py      Step 5  — schema + silent-killer checks
src/build_dataset.py     Step 6  — markets_dedup.parquet, trades_dedup.parquet
src/classify_domains.py  Step 7  — + domain -> markets_classified.parquet, Becker cross-check
src/reconcile.py         Step 8  — HARD GATE vs Le Table 1 -> results/table_reconciliation.csv
src/build_forecasts.py   Step 9  — forecasts_unfiltered.parquet, forecasts.parquet
tests/                   Steps 7+10 — classification tests, 5-part sanity suite
```

`data/derived/forecasts.parquet` (frozen at tag `phase1-done`) is the only input
to all downstream phases. Columns: `ticker, domain, tau, price, outcome,
close_time, trade_time, n_trades_market, trade_size`;
τ ∈ {1h, 6h, 24h, 1w, 1mo}; primary filters `0.05 ≤ price ≤ 0.95`,
`n_trades_market ≥ 10` (unfiltered variant retained for robustness sweeps).
