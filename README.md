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

## Phase 1 closed — 2026-08-18

- Data frozen at tag **`phase1-done`** = commit `4f5d08b` (Step 10). Instruction
  files tracked in `542986a`; this closing note is the last Phase 1 commit.
- Full suite re-verified green 2026-08-18: **33/33 passed**; regenerated
  spot-check dump and histograms byte-identical to committed versions.
- Canonical row counts (post-DST-fix files on disk; these supersede the stale
  Step 9 commit-message figures 241,376 / 947,225, which describe the pre-fix
  build):

| τ | `forecasts.parquet` | `forecasts_unfiltered.parquet` |
| --- | ---: | ---: |
| 1h | 68,111 | 428,582 |
| 6h | 87,544 | 327,421 |
| 24h | 67,282 | 157,177 |
| 1w | 12,141 | 22,965 |
| 1mo | 6,264 | 11,089 |
| **total** | **241,342** | **947,234** |

  (Filtered 1h < 6h is the price filter at work — 69.5% of τ=1h mass lies
  outside [0.05, 0.95]; the unfiltered set is monotone decreasing in τ.)
- Backup copies at **`C:\pm-backup\`** (2026-08-18, SHA256 verified == source):
  - `forecasts.parquet` — 7,589,481 bytes —
    `644C7CA4B020E174A86BA0EFB9A624F9C63C9551033C6BAC9C109FAE5AFD8DFF`
  - `forecasts_unfiltered.parquet` — 35,485,974 bytes —
    `D504C1CBDEF4A250D3159B151D9E98522059EAE1FFA2C8618BD474D7FF043FCC`
- Remote: https://github.com/aamilonas/kalshi-conformal.git (`master` + tag pushed).
- Downstream phases (see `PHASE2_3_METHODS.md`) read **only**
  `data/derived/forecasts.parquet`, plus `forecasts_unfiltered.parquet` for
  robustness sweeps. Raw archive retained at `E:\pm\becker-data\data.tar.zst`.

## Phase 4–5 closed — 2026-08-18

Prospective walk-forward, significance, robustness and the released table.
Tagged **`phase5-done`**. Steps 15–20 of `PHASE4_5_WALKFORWARD.md`.

### Protocol and horizons

Rolling quarterly walk-forward on market `close_time`: test quarter `Q`,
calibration = the trailing 12 months closing strictly before `Q`, refit every
quarter. The leakage assertion runs on every window and is fatal; a test
poisons a window to prove the guard fires.

| τ | quarters | span | test rows |
| --- | ---: | --- | ---: |
| 24h | 14 | 2022Q2 – 2025Q3 | 38,346 |
| 6h | 13 | 2022Q3 – 2025Q3 | 48,360 |
| 1w | 4 | 2024Q4 – 2025Q3 | 4,414 |
| 1mo | 2 | 2025Q2 – 2025Q3 | 1,297 |

2025Q4 is partial (data ends 2025-11-23) and excluded everywhere. τ=1mo is
excluded from H3; τ=1h was never run.

**Start-quarter deviation from 15a** (approved at gate 1). The instruction
file starts at the first quarter where *every* domain has ≥200 test rows at
τ=24h. Applied literally that yields 3 quarters at 24h and **zero** at 1w,
which makes its own mandatory 1w analysis impossible — Sports, Crypto and
Politics only launched in 2024–25. The pooled clause (trailing-12-month
calibration ≥1,000) sets the start quarter instead, and the per-domain clause
became the `meets_200_test` flag in `table_wf_counts.csv`, so the strict
schedule stays recoverable as a filter. Sanity check at gate 1 — rolling
Mondrian coverage at α=0.1, band [0.88, 0.92] — passed at every horizon
(24h 0.9065, 1w 0.9014, 6h 0.8992, 1mo 0.9144; 0.9026 pooled over 92,417
test rows in 33 windows).

### Headline results

**H1 — recalibration does not beat the raw Kalshi price.** Only 25 of 216
method × domain × horizon × scope cells improve on it out of sample, and
histogram binning improves in **0 of 54**. Market-clustered bootstrap at the
headline cell (domain ALL, pooled fit), ΔBrier vs raw:

| method | τ=24h | τ=1w |
| --- | --- | --- |
| Platt | +0.000119 [−0.000004, +0.000244] | +0.000370 [−0.000213, +0.001014] |
| Isotonic | +0.000574 [+0.000340, +0.000823] | +0.001941 [+0.000930, +0.003076] |
| Binning(10) | +0.002636 [+0.002227, +0.003058] | +0.001948 [+0.000753, +0.003139] |
| Venn–Abers | +0.000553 [+0.000323, +0.000772] | +0.001852 [+0.000775, +0.002838] |

Platt is indistinguishable from the raw price; the three flexible methods are
significantly **worse**. Of the 106 H1 intervals excluding zero, 103 are on
the worse side. This is the paper's central null and is reported as one.

**H2 — Mondrian reallocates coverage; it does not add any.** Pooled Δcoverage
(Mondrian − marginal) is indistinguishable from zero (24h −0.000026
[−0.001148, +0.001096]), which is the point: per-domain thresholds move
coverage *between* domains. At τ=24h the per-domain shifts are significant in
5 of 6 domains (Politics +0.0271, Finance +0.0237, Crypto +0.0204, Weather
−0.0083, Sports −0.0050). Mean |coverage − 0.9| across domains falls from
0.0142 to 0.0101 at 24h and 0.0246 to 0.0126 at 1w.

**H3 — ACI holds the level best at every horizon.** Mean absolute deviation
from nominal, pooled:

| τ | static-once | rolling refit | ACI (γ=0.005) |
| --- | ---: | ---: | ---: |
| 24h | 0.0076 | 0.0130 | **0.0038** |
| 6h | 0.0162 | 0.0126 | **0.0017** |
| 1w | 0.0343 | 0.0119 | **0.0094** |

ACI beats static-once significantly at both bootstrapped horizons
(24h −0.0038 [−0.0047, −0.0004]; 1w −0.0249 [−0.0304, −0.0127]). Rolling
refit beats static-once at 1w (−0.0223) but is significantly *worse* at 24h
(+0.0054). γ ∈ {0.01, 0.02} are also reported so the step size is not
cherry-picked.

### Robustness verdict

**H1's sign survives; H2's direction does not.** No recalibrator beats raw
under `price_02_98`, `price_10_90`, `bins5` or `bins20`. The single flip is
Platt under `trades_ge_100` at τ=24h (beats raw by 0.000287) — the same size
as the primary's own CI width, from the one method that ties everywhere else.
H2 **reverses** under the wide `price_02_98` band at both horizons (MAD
0.0123 → 0.0144 at 24h), so the H2 claim must be stated as conditional on the
price band. Full paragraph in `results/notes.md`.

### Two bugs found and fixed in this phase

1. **Degenerate Mondrian thresholds** (Step 16). A group whose calibration set
   is too small to resolve α has no `ceil((n+1)(1−α))` order statistic, so
   `q̂ = +inf` and every set became `{0,1}` — coverage 1.000, set size 2.000.
   25.3% of τ=24h test rows were "covered" by construction. This invalidated
   the gate-1 claim of an upward drift at 24h (2025Q3 static-once 0.9612 →
   0.9132 once fixed) and had also corrupted Phase 3's
   `table_M2_robustness_spec_split.csv`, which is regenerated. **No Phase 3
   exit-test verdict changed** — those run on the primary split, whose
   smallest per-domain calibration set is 739.
2. **A vacuous robustness verdict** (Step 18). `alpha` is the string `"NA"` in
   memory but NaN after a CSV round-trip, so filtering on `alpha.isna()`
   silently dropped every probability-metric row; the empty H1 loop then
   reported "HOLDS in every variant". Both are covered by regression tests,
   and `verdict()` now raises rather than passing on an empty comparison.

### Limitation that must reach the paper

Sports, Crypto and Politics per-domain results rest on **2025 alone** — at
most three test quarters at τ=24h and τ=6h, and no quarter clearing the
200-row bar at τ=1w or 1mo. Weather and Finance are the only domains spanning
the full range. Every per-domain row in `table_H1/H2/H3.csv` carries its own
`n_test` and `n_quarters`; the verbatim Limitations paragraph is in
`results/notes.md`.

### Reproduction

```
py run_all.py            # full pipeline, ~37 s
py run_all.py --fast     # skips the bootstrap
py run_all.py --list     # show stages
```

Order: `reproduce_le` → `run_single_split` → `walk_forward` →
`hypothesis_tables` → `bootstrap` → `robustness` → `build_recal_table`.
Venn–Abers is never subsampled: it fits once per unique test price (≤91 cent
values), so full test folds run in seconds and every `n_test` is the whole
fold.

**Clean-clone result (2026-08-18).** Cloned from origin into
`<pm>/tmp/clone-test`, copied in `forecasts.parquet` and
`forecasts_unfiltered.parquet`, ran `run_all.py --fast` plus the bootstrap
stage: **15 of 20 committed CSVs regenerated and matched exactly** (max
|Δ| 1.9e-16, float round-trip only). Adding `le_time_bins.parquet` (365 KB)
un-skips `reproduce_le` and reproduces 3 more (max |Δ| 4.4e-16). The last
two — `table_reconciliation.csv` and
`classify_crosscheck_disagreements.csv` — are Phase 1 outputs needing the
3.7 GB dedup tables and are frozen at `phase1-done`, outside `run_all`'s
scope by design.

If the sibling reference repos are not in the parent of the repo (as in a
scratch clone), set `KC_PM_ROOT` to the real `pm` directory. On macOS the
Windows `.venv` is unusable; `src/paths.py` resolves every path relative to
the repo root, so the code itself is cross-platform.

**Scope audit:** all 37 files in `results/` are mapped in
`results_needed.md`, and every file it names exists. Six pre-Phase-4 files
still carry an **UNMAPPED** paper slot awaiting a decision:
`table_slopes_ours_5h.csv`, `table_slopes_domain_time_9bin.csv`,
`table_reliability_bins_M2.csv`, `table_binning_ci_M2.csv`,
`table_M2_robustness_spec_split.csv`, `classify_crosscheck_disagreements.csv`.
