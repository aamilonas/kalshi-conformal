# DATA GUIDE — read this first when picking the project up cold

Project: **"Are Prediction Market Prices Valid Probabilities? Distribution-Free
Recalibration and Prospective Coverage on Kalshi and Polymarket"** — an academic
paper testing whether Kalshi prices work as probability forecasts, and fixing
them when they don't (Platt/isotonic/histogram binning/Venn–Abers/conformal).
This file explains every data artifact in the repo, how to load it, and the
findings that constrain how you may interpret it. Written 2026-08-18, at the
end of Phase 2–3 (single-split benchmark), gate-2 review pending.

---

## 1. The one file that matters

**`data/derived/forecasts.parquet`** — frozen at git tag `phase1-done`. Every
downstream analysis reads ONLY this file (plus its unfiltered sibling for
robustness). Never rebuild it; never commit it (gitignored).

One row = one (market, horizon) forecast:

| column | type | meaning |
|---|---|---|
| `ticker` | str | Kalshi market ticker (one resolved yes/no contract) |
| `domain` | str | Sports, Crypto, Politics, Finance, Weather, Entertainment, Other |
| `tau` | str | horizon: `1h`, `6h`, `24h`, `1w`, `1mo` |
| `price` | float | forecast = **last trade price strictly before `close_time − τ`**, in (0,1); = `yes_price/100`, so it lives on the cents grid (≤91 distinct values) |
| `outcome` | int | 1 if the market resolved `yes`, 0 if `no` |
| `close_time` | timestamptz (UTC) | market close |
| `trade_time` | timestamptz (UTC) | timestamp of the selected trade (always `< close_time − τ`; test-enforced) |
| `n_trades_market` | int | total trades on this market (liquidity proxy) |
| `trade_size` | int | contracts in the selected trade |

Semantics you must not forget:

- **τ is an absolute duration** (1w = 168 h, 1mo = 720 h), not calendar days.
  A DST bug (calendar `INTERVAL '7 days'` ≠ 168 real hours) was caught by the
  test suite and fixed; do not reintroduce calendar arithmetic on TIMESTAMPTZ.
- **Filters already applied**: `0.05 ≤ price ≤ 0.95` AND `n_trades_market ≥ 10`.
- **`forecasts_unfiltered.parquet`** = same construction, no filters. Use it for
  robustness sweeps ([0.02,0.98], [0.10,0.90], ≥100 trades) as a one-line filter,
  and for anything needing base rates uncontaminated by the price filter.
- A market appears at a given τ only if it had ≥1 trade before `close − τ`, so
  long-τ cells select for long-lived markets (unfiltered counts are monotone
  decreasing in τ; the filtered set is NOT — the price filter bites hardest at
  τ=1h where 69.5% of mass sits outside [0.05,0.95]).

Canonical row counts (post-DST-fix; these supersede the Step 9 commit message):

| τ | forecasts.parquet | forecasts_unfiltered.parquet |
|---|---:|---:|
| 1h | 68,111 | 428,582 |
| 6h | 87,544 | 327,421 |
| 24h | 67,282 | 157,177 |
| 1w | 12,141 | 22,965 |
| 1mo | 6,264 | 11,089 |
| **total** | **241,342** | **947,234** |

Loading:

```python
import pandas as pd
fc = pd.read_parquet("E:/pm/kalshi-conformal/data/derived/forecasts.parquet")
```

```sql
-- DuckDB: forward slashes ALWAYS, even on Windows
SELECT domain, tau, COUNT(*) FROM
  'E:/pm/kalshi-conformal/data/derived/forecasts.parquet'
GROUP BY domain, tau;
```

Backups: `C:\pm-backup\` holds byte-identical copies (SHA256 in README §"Phase
1 closed"). The raw Becker archive (`E:\pm\becker-data\data.tar.zst`, 36 GB) is
the rebuild-of-last-resort; never modify anything under `E:\pm\becker-data`.

## 2. Supporting data files (`data/derived/`, all gitignored)

| file | built by | contents / when to use |
|---|---|---|
| `le_time_bins.parquet` | `src/le_time_bins.py` | **Trade-level** dataset on Le's 9 time bins × 4 size bins, pre-aggregated by (domain, tbin, sbin, yes_price, is_yes) with `n_trades` and `total_contracts`. Used only for reproducing Le; verified to match their published matrix trade-for-trade. |
| `markets_classified.parquet` | Phase 1 | All 7.68M market snapshots + `domain`. Upstream input; don't reread in analyses. |
| `markets_dedup.parquet`, `trades_dedup.parquet` | Phase 1 | Deduped raw tables (dedup was a no-op on this archive). Phase 2+ code may not read `trades_dedup` except via `le_time_bins.py`. |

Prices in the raw/trade-level data are **cents** (1–99, BIGINT); only
`forecasts*.parquet` is already divided by 100.

## 3. Domains — Le's classifier, quirks included on purpose

`domain` comes from a verbatim port of Le's 571-pattern classifier
(`src/le_classify.py`, wrapped by `src/classify_domains.py`), which matches
**ordered case-insensitive substrings** on the event-ticker prefix. Known
quirks that are deliberately preserved because Le's published Table 1/matrix
embeds them (do NOT "fix"):

- `FEDDECISION` → Politics (matches early pattern `EC`), not Finance.
- `ARGINFLATIONM` → Sports (contains `NFL`).
- `SBADS` (Super Bowl ads) → Sports via `SB`.

The 6 analysis domains are Sports, Crypto, Politics, Finance, Weather,
Entertainment; everything else collapses to `Other` (present in
`forecasts*.parquet`, excluded from benchmarks).

## 4. Results inventory (`results/`, all committed)

| file | what it is |
|---|---|
| `table_reconciliation.csv` | Phase 1 gate: our counts vs Le Table 1 — markets match exactly (210,608), trades within 0.08%. |
| `table_reproduction.csv` | 216 rows: our per-(domain × time-bin × size-bin) slope refits vs Le's published matrix. **Identical to machine precision** (r = 1.000000, max Δ = 4e-16); per-cell trade counts equal. |
| `fig_reproduction.png/.pdf` | The 216-cell scatter on y=x. |
| `table_slopes_domain_time_9bin.csv` | Pooled contract-weighted slopes per (domain × 9 time bins) — Le's Table-3 estimator. Politics 1w-1mo = 1.833 (the headline underconfidence). |
| `table_slopes_ours_5h.csv` | **Market-level** slopes on our (domain × τ) grid, unweighted. NB: tells a different Politics story (see §6.1). |
| `table_M2_main.csv` | Step 14 benchmark, long tidy: `method, fit_scope (pooled/per_domain), domain (or ALL), tau, alpha (or NA), metric, value, n_test`. Metrics: brier, log_loss, ece, coverage, avg_set_size; α ∈ {0.1 primary, 0.05, 0.2}. Note: read with pandas → the `alpha` "NA" strings become NaN; filter with `t['alpha'].isna()`. |
| `table_M2_robustness_spec_split.csv` | Same benchmark at the original 2024-01-01 boundary, pooled fits only (see §6.2). |
| `fig_reliability_M2.png/.pdf` + `table_reliability_bins_M2.csv` | Reliability diagrams, Politics/Sports × pooled/per-domain, Raw vs Platt vs Venn–Abers, with underlying bin tables. |
| `table_binning_ci_M2.csv` | HistogramBinning per-domain bin tables with Clopper–Pearson 95% CIs — seeds the paper's released recalibration table. |
| `spotcheck.txt`, `fig_price_hists.png` | Phase 1 human-review artifacts (both reviewed/approved). |
| `notes.md` | Parked findings — read it; §6 below summarizes. |

## 5. Code map

```
src/metrics.py           brier, log_loss, equal-mass ECE, coverage,
                         avg_set_size, reliability_diagram (returns bin table)
src/recalibrators.py     Raw, Platt, Isotonic, HistogramBinning(+bin_table/CIs),
                         VennAbers (exact via unique cents prices),
                         SplitConformal(marginal|mondrian), Adaptive (Phase 4 stub)
src/run_single_split.py  Step 14 benchmark (chronological split, leakage asserted)
src/reproduce_le.py      Step 12 reproduction vs Le's published matrix
src/le_time_bins.py      the one permitted trades_dedup read (Le's extraction)
tests/                   46 tests, all green: metrics(6), recalibrators(7, incl.
                         conformal coverage canary), classify(28), forecasts(5)
```

Run everything with the venv: `.\.venv\Scripts\python.exe -m pytest -q` from
`E:\pm\kalshi-conformal` (Windows; `py`, never `python`; PowerShell).

## 6. Findings that constrain interpretation (the "what it all means")

### 6.1 Two estimands, two Politics stories — decide before writing

- **Trade-level, contract-weighted** (Le's estimand; `le_time_bins` grid):
  Politics slope at 1w-1mo = **1.833** — strong underconfidence, whale trades
  dominate the weight.
- **Market-level, unweighted** (our `forecasts.parquet` estimand): Politics at
  τ=1w slope = **0.871**, τ=24h = 0.981 — approximately calibrated.

Both are exactly computed from the same data (the reproduction is
machine-precision exact, so this is not a bug). "Politics is underconfident"
is substantially a trade/contract-weighting phenomenon. Any recalibration
claim must state which estimand it is about.

### 6.2 Kalshi is violently non-stationary — the split boundary decision

Most of Kalshi's product line launched mid-stream: at τ=24h, pre-2024 markets
number Sports **3**, Crypto **0**, Politics **14**. The instruction file's
original split (cal < 2024-01-01) was therefore infeasible per-domain. Per
user decision (2026-08-18): **primary boundary = 2025-07-01** (cal 28,642 /
test 31,465; every domain ≥ 739 cal markets), original boundary retained as a
pooled-only robustness table. Test fold = H2 2025 only.

### 6.3 Conformal overcoverage = temporal shift, not a bug (exit tests 1 & 3)

Marginal split-conformal pooled coverage at α=0.1 is **0.9223** vs nominal
[0.885, 0.915]. Decomposition: in-sample calibration-fold coverage is 0.9053
(= 0.90 + tie mass from the discrete cents grid — exactly nominal; the
implementation passes its i.i.d. canary), and the remaining **+0.017 is
H1→H2 2025 distribution shift**. Same pattern per-domain for mondrian
(0.918–0.935; spread narrowed from 0.077 to 0.052, and Crypto lifted
0.857→0.884, so the mechanism works). Direction is conservative — the ≥1−α
guarantee holds. This is the empirical motivation for Phase 4's adaptive
conformal (ACI). Exit tests 1 and 3 "fail as specified" for this documented
reason; gate-2 acceptance pending.

### 6.4 Politics recalibration null at market level (exit test 2)

Politics 24h test fold: base rate 0.460, mean price 0.493, Raw already
calibrated. Venn–Abers/HistogramBinning tie Raw on Brier within 1% of its SE
(0.18021/0.18045 vs 0.18017) while slightly improving ECE; Platt improves all
three metrics (Brier 0.17876). Investigated for leakage/wrong-fold/wrong-label:
clean. A genuine null, and consistent with §6.1.

### 6.5 Other data facts you'll eventually need

- Trades span 2021-06-30 → **2025-11-25** (content end). Archive vintage
  2026-02-05; identical to the vintage behind Le's published numbers.
- Timestamps are UTC instants (TIMESTAMPTZ); `_fetched_at` is naive. Verified
  against the 2024 election market's known close.
- 0.37% of raw trades have `yes_price + no_price ≠ 100`; the [0.05,0.95]
  filter drops these edges.
- Base rates (resolved, Le's population): overall 38.1%; per-domain in
  `table_reconciliation.csv`.
- Prices cluster at extremes at short τ — equal-mass (quantile) bins, never
  equal-width, for anything binned.

## 7. Project state as of 2026-08-18

- **Done & pushed** (`origin = github.com/aamilonas/kalshi-conformal`,
  branch `master`): Phases 0–1 (tag `phase1-done`), Steps 11–14.
- **Pending**: gate-2 review of the Step 14 exit-test verdicts (§6.3, §6.4);
  the `phase3-done` tag is withheld until the user accepts them.
- **Next**: Phase 4 — walk-forward evaluation + Adaptive Conformal Inference
  (the `Adaptive` stub), no instruction file written yet. Then the paper:
  §6.1's estimand choice is the first writing decision.
- Environment cautions: E: is exFAT (never reformat); disable AC sleep before
  heavy DuckDB work (`powercfg /change standby-timeout-ac 0`, restore to 60
  after); DuckDB spills to `E:/pm/tmp`; never commit anything under `data/`
  or any `*.parquet`.
