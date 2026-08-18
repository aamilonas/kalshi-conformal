# Phase 0–1: Data Engineering & Testing — Kalshi Conformal Recalibration Paper

## Context for Claude Code

You are helping build the data foundation for an academic paper: **"Are Prediction Market Prices Valid Probabilities? Distribution-Free Recalibration and Prospective Coverage on Kalshi and Polymarket."**

This document covers **only** the data engineering and testing phase: downloading the Becker dataset, building a clean forecast dataset, and validating it against Le (2026). Downstream phases (recalibration methods, walk-forward, writing) read from the output of this phase and are out of scope here.

**Environment facts:**
- Windows machine. Use `py` for Python, never `python` or `python3`. PowerShell is the shell.
- A 1 TB external drive is mounted (assume `E:`; confirm the actual letter with the user before writing anything, and substitute throughout).
- DuckDB paths use forward slashes even on Windows: `E:/pm/...`
- No WSL. Everything runs native Windows. Do not use `make` from the Becker repo; the download is done manually.

**Working layout (create this):**
```
E:\pm\
├── becker-data\          # raw archive + extracted data/ (never modified after extraction)
├── prediction-market-analysis\      # Becker's repo (reference only)
├── prediction-market-calibration\   # Le's repo (reference only)
├── tmp\                  # DuckDB spill directory
└── kalshi-conformal\     # OUR repo — all work happens here
    ├── data\derived\     # gitignored parquet outputs
    ├── src\
    ├── tests\
    ├── results\
    ├── notebooks\
    ├── paper\
    └── lit\
```

**Non-negotiable rules:**
1. Commit after every numbered step below. Commit messages reference the step number.
2. Every result lands as a CSV in `results/` before it becomes anything else.
3. When a number surprises you, suspect the code before the world. Coverage-of-join errors, duplicate snapshots, and timezone/unit mistakes are the three known silent killers in this dataset.
4. The reconciliation gate (Step 8) is a hard stop. Do not proceed past it on unreconciled data. Ask the user to review the reconciliation table before continuing.
5. Never commit raw Becker data to git. `.gitignore` must cover `data/`, `*.parquet`, `.venv/`, `lit/*.pdf`.

---

## Step 1 — Prep the drive

1. Confirm drive letter with the user (assumed `E:`).
2. Verify it is NTFS: `Get-Volume` in PowerShell. If exFAT/FAT32, ask the user before reformatting (formatting erases the drive).
3. Create directories:
```powershell
mkdir E:\pm\becker-data, E:\pm\tmp
```

## Step 2 — Start the dataset download FIRST (long-running; run in background)

```powershell
cd E:\pm\becker-data
curl.exe -L -C - -o data.tar.zst https://s3.jbecker.dev/data.tar.zst
```

- `-C -` resumes on interruption; the same command can be re-run safely.
- Expected final size: ~36 GiB (~38.7 GB as Windows reports). If the file is tiny, it's an error page — delete and retry.
- Run this in a separate terminal/background job and proceed to Step 3 while it downloads.

## Step 3 — Tooling, repos, and project scaffold

Install tools:
```powershell
winget install Facebook.Zstandard
winget install DuckDB.cli
```

Clone reference repos (code only; do not run their Makefiles):
```powershell
cd E:\pm
git clone https://github.com/Jon-Becker/prediction-market-analysis.git
git clone https://github.com/namanhz/prediction-market-calibration.git
```

From `prediction-market-calibration`, locate and note the paths to:
- (a) the 216-cell calibration matrix CSV
- (b) the domain classification rules (~560+ ticker-prefix rules)

Both are load-bearing for Steps 7–8.

Scaffold our repo:
```powershell
cd E:\pm
mkdir kalshi-conformal
cd kalshi-conformal
git init
mkdir src, tests, notebooks, results, paper, lit, data\derived
py -m venv .venv
.venv\Scripts\activate
pip install duckdb pandas numpy scikit-learn matplotlib scipy pyarrow pytest
pip freeze > requirements.txt
```

Create `.gitignore`:
```
data/
.venv/
*.parquet
lit/*.pdf
__pycache__/
E:/pm/tmp/
```

First commit: `git add -A && git commit -m "Step 3: scaffold repo and env"`

## Step 4 — Extract the archive

Only after the Step 2 download completes:
```powershell
cd E:\pm\becker-data
zstd -d data.tar.zst
# If zstd errors about window size, retry with: zstd -d --long=31 data.tar.zst
tar -xf data.tar
del data.tar
```

- Keep `data.tar.zst` permanently as the backup copy. Delete only the intermediate `data.tar`.
- Expected result: `E:\pm\becker-data\data\kalshi\{markets,trades}\` and `data\polymarket\{blocks,markets,trades}\` containing parquet files.

**Checkpoint:**
```powershell
duckdb -c "SELECT COUNT(*) FROM 'E:/pm/becker-data/data/kalshi/trades/*.parquet'"
```
Expect tens of millions of rows (~64.7M ± a few percent; the archive may be newer than Le's 2025-12-31 cutoff, so slightly more is fine).

## Step 5 — Schema inspection (`src/inspect_data.py`)

Known schema (from Becker's `docs/SCHEMAS.md`) — verify it matches reality:

**Kalshi markets** (one row per contract, possibly multiple snapshots per ticker):
`ticker`, `event_ticker`, `market_type`, `title`, `status` (open/closed/finalized), `yes_bid`, `yes_ask`, `last_price`, `volume`, `open_interest`, `result` (`yes`/`no`/empty), `created_time`, `open_time`, `close_time`, `_fetched_at`

**Kalshi trades** (one row per execution):
`trade_id`, `ticker`, `count`, `yes_price` (cents, 1–99), `no_price` (= 100 − yes_price), `taker_side`, `created_time`, `_fetched_at`

**Prices are in cents.** Probability conversion: `p = yes_price / 100.0`.

Write and run `src/inspect_data.py`:
```python
import duckdb
DATA = "E:/pm/becker-data/data"
con = duckdb.connect()

con.sql(f"DESCRIBE SELECT * FROM '{DATA}/kalshi/markets/*.parquet'").show()
con.sql(f"SELECT * FROM '{DATA}/kalshi/markets/*.parquet' LIMIT 3").show()
con.sql(f"DESCRIBE SELECT * FROM '{DATA}/kalshi/trades/*.parquet'").show()
con.sql(f"SELECT * FROM '{DATA}/kalshi/trades/*.parquet' LIMIT 3").show()

# Duplicate-snapshot check: if max_snapshots > 1, dedup in Step 6 is mandatory
con.sql(f"""
  SELECT MAX(cnt) AS max_snapshots FROM (
    SELECT ticker, COUNT(*) AS cnt
    FROM '{DATA}/kalshi/markets/*.parquet' GROUP BY ticker)
""").show()

# Trade duplicate check
con.sql(f"""
  SELECT COUNT(*) - COUNT(DISTINCT trade_id) AS dup_trades
  FROM '{DATA}/kalshi/trades/*.parquet'
""").show()
```

Record in `README.md`: exact column names, dtypes, timestamp timezone (verify UTC by spot-checking one externally verifiable market, e.g. a 2024 election contract), price units, and the duplicate counts found.

Three known silent killers to explicitly verify here:
1. **Duplicate snapshots** — markets fetched repeatedly with different `_fetched_at`.
2. **Timezone consistency** — `created_time` (trades) and `close_time` (markets) must be in the same tz (expect UTC).
3. **Price units** — cents, not dollars.

Commit: `"Step 5: schema inspection + README notes"`

## Step 6 — Deduped base tables (`src/build_dataset.py`, part 1)

```python
import duckdb
DATA = "E:/pm/becker-data/data"
OUT  = "E:/pm/kalshi-conformal/data/derived"

con = duckdb.connect()
con.sql("SET memory_limit='8GB'")
con.sql("SET temp_directory='E:/pm/tmp'")

con.sql(f"""
COPY (
  SELECT * FROM '{DATA}/kalshi/markets/*.parquet'
  QUALIFY ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY _fetched_at DESC) = 1
) TO '{OUT}/markets_dedup.parquet' (FORMAT PARQUET)
""")

con.sql(f"""
COPY (
  SELECT * FROM '{DATA}/kalshi/trades/*.parquet'
  QUALIFY ROW_NUMBER() OVER (PARTITION BY trade_id ORDER BY _fetched_at DESC) = 1
) TO '{OUT}/trades_dedup.parquet' (FORMAT PARQUET)
""")
```

Notes:
- The trades dedup over ~64M rows takes minutes; that's normal.
- `temp_directory` must point at the external drive so spills don't fill C:.
- Print row counts before/after dedup and record them in README.

Commit: `"Step 6: deduped base tables"`

## Step 7 — Domain classification (`src/classify_domains.py`)

Port **Le's** ticker-prefix rules (from `prediction-market-calibration`) into:
```python
def classify(ticker: str, event_ticker: str) -> str:
    """Returns one of: Sports, Crypto, Politics, Finance, Weather, Entertainment, Other"""
```

Why Le's rules and not our own: the reconciliation target in Step 8 is Le's Table 1, so we must match their domain definitions exactly.

Cross-check: Becker's repo has its own `get_group()` in `src/analysis/kalshi/util/categories.py`. After classifying, compare our labels against `get_group()` on the same tickers. Disagreement above a few percent in any major domain means the port is wrong — investigate before proceeding.

Apply to `markets_dedup.parquet` → save `data/derived/markets_classified.parquet` (all market columns + `domain`).

Also write unit tests now (`tests/test_classify.py`): ~20 known tickers with expected domains (e.g., `PRES-2024-DJT` → Politics; NFL game prefixes → Sports; BTC prefixes → Crypto).

Commit: `"Step 7: domain classification + unit tests"`

## Step 8 — RECONCILIATION GATE (hard stop; M0 exit test)

Compute per-domain: market counts, resolved counts, trade counts, base rates.

```python
con.sql(f"""
SELECT domain,
       COUNT(*) AS n_markets,
       SUM(CASE WHEN result IN ('yes','no') THEN 1 ELSE 0 END) AS n_resolved,
       AVG(CASE WHEN result='yes' THEN 1.0
                WHEN result='no'  THEN 0.0 END) AS base_rate
FROM '{OUT}/markets_classified.parquet'
GROUP BY domain ORDER BY n_markets DESC
""").show()
```

Plus trade counts per domain (join `trades_dedup` to `markets_classified` on ticker).

**Compare against Le (2026) Table 1:**
| Quantity | Le's value |
|---|---|
| Politics markets / trades | ≈6,609 / 4.9M |
| Sports markets / trades | ≈55,637 / 43.2M |
| Total markets / trades | ≈210,608 / 64.7M |
| Overall base rate | ≈38.1% |

**PASS:** within a few percent, with overshoots only (Le's cutoff was 2025-12-31; a newer archive should have *more* markets, never fewer). Document the exact deltas.
**FAIL:** anything wildly off (e.g., Politics at 3,000 or 20,000) means the classifier port or extraction is broken. STOP. Fix. Do not proceed.

Save the comparison as `results/table_reconciliation.csv` with columns: `domain, ours_markets, le_markets, ours_trades, le_trades, ours_base_rate, le_base_rate, delta_pct`.

**Present the table to the user and wait for their go-ahead before Step 9.**

Commit: `"Step 8: reconciliation vs Le Table 1 — PASS"`

## Step 9 — Build the forecast dataset (the paper's core object)

For horizons τ ∈ {1h, 6h, 24h, 1w, 1mo}: for each resolved market, the forecast is the price of the **last trade strictly before** `close_time − τ`; the label is the resolution outcome.

```python
HORIZONS = {"1h": "1 hour", "6h": "6 hours", "24h": "24 hours",
            "1w": "7 days", "1mo": "30 days"}

frames = []
for tau_name, tau_sql in HORIZONS.items():
    df = con.sql(f"""
      WITH resolved AS (
        SELECT ticker, domain, close_time,
               CASE WHEN result='yes' THEN 1 ELSE 0 END AS outcome,
               close_time - INTERVAL '{tau_sql}' AS cutoff
        FROM '{OUT}/markets_classified.parquet'
        WHERE result IN ('yes','no') AND close_time IS NOT NULL
      ),
      tcounts AS (
        SELECT ticker, COUNT(*) AS n_trades_market
        FROM '{OUT}/trades_dedup.parquet' GROUP BY ticker
      )
      SELECT r.ticker, r.domain, '{tau_name}' AS tau,
             t.yes_price / 100.0 AS price,
             r.outcome, r.close_time, t.created_time AS trade_time,
             tc.n_trades_market, t.count AS trade_size
      FROM resolved r
      ASOF JOIN '{OUT}/trades_dedup.parquet' t
        ON r.ticker = t.ticker AND t.created_time < r.cutoff
      JOIN tcounts tc ON tc.ticker = r.ticker
    """).df()
    frames.append(df)
```

Save two files:
1. `data/derived/forecasts_unfiltered.parquet` — everything.
2. `data/derived/forecasts.parquet` — primary filters applied: `0.05 <= price <= 0.95` AND `n_trades_market >= 10`.

Keeping the unfiltered version makes Phase 5 robustness ([0.02, 0.98], [0.10, 0.90], ≥100 trades) a one-line filter change, not a rebuild.

Final columns: `ticker, domain, tau, price, outcome, close_time, trade_time, n_trades_market, trade_size`.

Commit: `"Step 9: forecasts.parquet built"`

## Step 10 — Sanity test suite (`tests/test_forecasts.py`, run with `py -m pytest`)

All five tests must pass. Each is a real pytest test, not a notebook cell.

1. **Join correctness:** for every row, assert `trade_time < close_time − τ`. (An ASOF JOIN with a flipped inequality passes eyeball inspection and fails this.)
2. **Base rates:** `mean(outcome)` per domain on the unfiltered set matches Step 8's base rates within noise (assert |delta| < ~1 percentage point).
3. **Cell sizes:** print the (domain × τ) row-count grid. Assert no near-zero cells for domains Le reports as large, and assert counts are non-increasing as τ grows (fewer markets have a trade 1mo before close than 1h before).
4. **Price distributions:** generate per-domain histograms of `price` for τ=1h and τ=1mo, saved to `results/fig_price_hists.png`. Short horizons should pile mass near 0 and 1; long horizons flatter. Flag (don't hard-fail) if 1h looks uniform.
5. **Hand spot-check, five markets:** pick 5 tickers spanning domains (include one election market, one sports game, one weather market). For each, dump the full trade history sorted by time and programmatically verify the selected row is genuinely the last trade before cutoff at every τ. Save the dump to `results/spotcheck.txt` for the user to review by eye — this is the test that catches timezone and unit errors nothing else catches.

When green:
```powershell
git add -A
git commit -m "Step 10: sanity suite green — forecasts.parquet frozen"
git tag phase1-done
```

From this point, `forecasts.parquet` is frozen. All downstream analysis (reproduction of Le's slopes, the seven recalibration methods, walk-forward) reads ONLY this file.

---

## Operational notes

- Run heavy DuckDB steps with the machine plugged in and sleep disabled (a mid-write sleep on an external drive truncates parquet files).
- If DuckDB runs out of memory, lower `memory_limit` — it will spill to `E:/pm/tmp` and finish slower but correctly.
- If any step produces a surprising number, stop and report it to the user with the query used, rather than adjusting filters until it looks right.

## Definition of done for this phase

- [ ] `data.tar.zst` downloaded, verified (~36 GiB), extracted; original archive retained
- [ ] Schema notes + duplicate/timezone/unit findings recorded in README
- [ ] `markets_dedup.parquet`, `trades_dedup.parquet` built
- [ ] `classify()` ported from Le, unit-tested, cross-checked against Becker's `get_group()`
- [ ] Reconciliation table PASSES vs Le Table 1, saved to `results/table_reconciliation.csv`, reviewed by user
- [ ] `forecasts_unfiltered.parquet` + `forecasts.parquet` built with correct columns
- [ ] All 5 sanity tests green; spot-check dump reviewed by user
- [ ] Git history shows one commit per step; `phase1-done` tag pushed