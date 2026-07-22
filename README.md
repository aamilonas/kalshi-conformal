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
- `kalshi/markets/*.parquet`: **7,682,445** rows — ~30 snapshots per ticker.
  The archive appends repeated market snapshots (`_fetched_at` varies);
  Le's pull evidently had ~1 row/ticker. **Dedup (Step 6) is mandatory** and
  reproduces Le's effective single-snapshot semantics.

## Step 5 findings (schema inspection)

_To be filled from `src/inspect_data.py` output: exact dtypes, duplicate-snapshot
count, duplicate-trade count, timezone verification, price-unit verification._

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
