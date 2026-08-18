# Phase 4–5: Prospective Walk-Forward, Significance, Robustness, Deliverables

## Context for Claude Code

Continuation of the Kalshi conformal recalibration paper. Phase 3 is complete and tagged `phase3-done`: `src/metrics.py` and `src/recalibrators.py` exist and are tested, the single-split benchmark (`results/table_M2_main.csv`) passed its exit tests, and Le's slopes are reproduced. Read `PROJECT_SUMMARY.md`, `PHASE2_3_METHODS.md`, and `README.md` first.

Same environment and rules as before: Windows, `py` / `.venv`, PowerShell, DuckDB with forward-slash paths, one commit per step pushed to origin, every result is a CSV in `results/` before it becomes a figure. Read only `data/derived/forecasts.parquet` and `forecasts_unfiltered.parquet`. Never touch the raw archive or dedup tables.

**Goal of this phase:** produce every table and figure the Results section needs (H1, H2, H3, CIs, robustness), the released `recalibration_table.csv`, and a one-command reproduction. At the end the paper is fully writable.

**Files to create:**
```
src/walk_forward.py       # Step 15
src/adaptive.py           # Step 15 (ACI implementation; fills the Phase 3 stub)
src/hypothesis_tables.py  # Step 16
src/bootstrap.py          # Step 17
src/robustness.py         # Step 18
src/build_recal_table.py  # Step 19
run_all.py                # Step 20
results_needed.md         # Step 20
tests/test_walk_forward.py
tests/test_adaptive.py
tests/test_bootstrap.py
```

**The one file that matters most:** `results/walk_forward_long.csv`. Every H1/H2/H3 table and figure aggregates from this single tidy file. Get its schema right first.

---

## Step 15 — Walk-forward protocol (`src/walk_forward.py`, `src/adaptive.py`)

### 15a. Window construction

Windows are defined on market `close_time` (the resolution timestamp). Rolling quarterly:

- Test quarter `Q` = calendar quarter (Jan–Mar, Apr–Jun, Jul–Sep, Oct–Dec).
- Calibration window for `Q` = the trailing 12 months of markets whose `close_time` falls strictly before the first day of `Q`.
- Roll across the archive. Determine the first usable test quarter automatically: the earliest quarter where every domain has ≥ 200 test rows at τ=24h AND the trailing-12-month calibration set has ≥ 1,000 rows pooled. Print the chosen start quarter and the per-quarter, per-domain counts to `results/table_wf_counts.csv` and to the console. Expect this to land around 2022; if it's later than 2023, report before proceeding.
- Last test quarter = the last full quarter before the data ends (data ends 2025-11-25, so Q3 2025 is the last complete quarter; Q4 2025 is partial, exclude it and say so).

**Leakage assertion, mandatory:** for every window, `assert cal_close_time.max() < test_close_time.min()`. Fail loudly.

### 15b. Methods to run per window

All from `recalibrators.py`: `Raw`, `Platt`, `Isotonic`, `HistogramBinning(10)`, `VennAbers`, `SplitConformal(Raw, mode='marginal')`, `SplitConformal(Raw, mode='mondrian')`, plus the two H3 comparators below.

Fit scope: run both `pooled` (one fit on all domains) and `per_domain` (one fit per domain) as in Phase 3. For conformal, `marginal` is pooled by definition and `mondrian` is per-domain by definition; do not double-run those.

For `VennAbers`: if the full test quarter is too slow, use a stratified-by-domain subsample of at most 5,000 test rows per quarter and record `n_test` honestly. Do not silently approximate.

### 15c. H3 comparators (implement in `src/adaptive.py`)

H3 asks whether coverage decays over time and whether adaptive updating repairs it. Three coverage curves per domain:

1. **Static-once**: fit `SplitConformal(mondrian)` ONCE on the calibration window of the first test quarter; apply that fixed threshold to every subsequent quarter without refitting. This is the "naive user" baseline that should decay under drift.
2. **Rolling-refit**: the standard walk-forward above (refit each quarter on trailing 12 months). This is what most of the paper uses.
3. **ACI** (Adaptive Conformal Inference, Gibbs & Candès 2021): within each test quarter, process test points in `close_time` order. Maintain `alpha_t`, starting at target `alpha=0.1`. After observing outcome `y_t`: `alpha_{t+1} = alpha_t + gamma * (alpha - err_t)` where `err_t = 1` if the set at time t did NOT contain `y_t`, else 0. Threshold at each step uses `alpha_t` on the calibration scores (clip `alpha_t` to [0.001, 0.999]; if `alpha_t` ≤ 0 the set is {0,1}, if ≥ 1 the set is empty). Use `gamma = 0.005` primary; also run `gamma ∈ {0.01, 0.02}` and report all three so the choice isn't cherry-picked. Run ACI both per-domain (Mondrian-style: separate alpha_t per domain) and pooled. Cite Gibbs & Candès 2024 in the docstring for the note that step-size can itself be selected adaptively; you are NOT implementing that, just noting it.

`tests/test_adaptive.py`: on i.i.d. synthetic data ACI long-run coverage converges to 1−alpha within ±1.5pp; on synthetic data with a sharp distribution shift halfway, static coverage drops after the shift while ACI recovers within a few hundred steps.

### 15d. Horizons

Run τ ∈ {24h, 1w} as the minimum. Then, if per-quarter counts are healthy, also run 6h and 1mo. Skip 1h unless everything else is done (the τ=1h selection effect documented in the README makes it the least clean horizon). Report which horizons ran.

### 15e. Output schema — `results/walk_forward_long.csv`

One row per (test_quarter, tau, method, fit_scope, domain, alpha, metric):
```
test_quarter   e.g. "2023Q2"
cal_start, cal_end, test_start, test_end   ISO dates
tau            "24h" | "1w" | ...
method         "raw"|"platt"|"isotonic"|"binning10"|"venn_abers"|"conformal_marginal"|"conformal_mondrian"|"static_once"|"aci_pooled_g0.005"|"aci_mondrian_g0.005"|...
fit_scope      "pooled"|"per_domain"|"na"
domain         "Politics"|...|"ALL"
alpha          0.05|0.1|0.2|NA
metric         "brier"|"log_loss"|"ece"|"coverage"|"avg_set_size"|"n_test"|"n_cal"
value          float
```
Also write per-quarter reliability-bin tables to `results/wf_reliability_bins.csv` (needed for the H1 diagrams).

`tests/test_walk_forward.py`: window boundaries never overlap; every window passes the leakage assertion; the long CSV has no duplicate keys; the sum of `n_test` across domains equals the pooled `n_test` per quarter.

Commit: `"Step 15: walk-forward + ACI — [horizons run, start quarter]"`

## Step 16 — Hypothesis tables and figures (`src/hypothesis_tables.py`)

Everything here reads `walk_forward_long.csv` only.

**H1 — proper scores.** `results/table_H1.csv`: method × domain × tau, mean Brier / log loss / ECE aggregated over all test quarters, weighted by `n_test`. Include `ALL`. Also `results/fig_H1_brier_delta.png/.pdf`: per domain, ΔBrier of each method vs Raw (negative = better), τ=24h and 1w side by side.

**H2 — coverage allocation.** `results/table_H2.csv`: per domain, per tau, mean coverage and mean set size for `conformal_marginal` vs `conformal_mondrian` at α=0.1 (and 0.05, 0.2 in extra columns). The expected pattern: marginal over-covers easy domains and under-covers hard ones; Mondrian is near 90% everywhere. `results/fig_H2_coverage_by_domain.png/.pdf`: grouped bar, domain on x, coverage on y, marginal vs Mondrian, nominal line at 0.9.

**H3 — coverage over time.** `results/table_H3.csv`: per test_quarter, per domain, coverage at α=0.1 for `static_once`, `conformal_mondrian` (rolling), and `aci_mondrian_g0.005`. `results/fig_H3_coverage_time.png/.pdf`: one panel per domain (2×3 grid), quarter on x, coverage on y, three lines, nominal 0.9 dashed, with the 2024 election quarters (2024Q3, 2024Q4) lightly shaded. Also a summary column: mean absolute deviation from 0.9 per method per domain, so H3 has one number per cell.

**Reliability diagrams.** `results/fig_reliability_wf.png/.pdf`: Politics and Sports at τ=24h and 1w (2×2), Raw vs Platt vs Venn–Abers, pooled across test quarters.

Every figure: axis labels with units, legend, a caption string written into `results/captions.md` that stands alone.

Commit: `"Step 16: H1/H2/H3 tables and figures"`

## Step 17 — Market-clustered bootstrap (`src/bootstrap.py`)

Rows within a market are correlated (same market at multiple τ, and within a τ the outcome is shared), so resample **markets**, not rows. Le clusters the same way; mirror it.

For each headline delta, 1,000 bootstrap replicates, 95% percentile CI:
- ΔBrier and Δlog loss vs Raw, for each of {platt, isotonic, binning10, venn_abers}, per domain and ALL, at τ ∈ {24h, 1w}.
- Δcoverage (mondrian − marginal) per domain at α=0.1, τ ∈ {24h, 1w}.
- ΔMAD-from-nominal (aci − static_once) and (rolling − static_once) per domain for H3.

Implementation: collect the per-row predictions from the walk-forward (save them during Step 15 as `data/derived/wf_predictions.parquet` with columns `ticker, test_quarter, tau, method, fit_scope, domain, p_hat, set0, set1, y`; gitignored). Bootstrap by sampling tickers with replacement, recomputing each metric on the resampled rows. Seed = 20260818. Vectorize with numpy; 1,000 reps over ~250k rows is a few minutes.

Output `results/table_bootstrap_ci.csv`: `comparison, domain, tau, alpha, point_estimate, ci_lo, ci_hi, n_markets`. Add a `significant` boolean (CI excludes 0).

`tests/test_bootstrap.py`: on synthetic data with a known true delta, the CI contains it ~95% of the time over 200 outer reps (loose tolerance: 90–99%).

Commit: `"Step 17: market-clustered bootstrap CIs"`

## Step 18 — Robustness (`src/robustness.py`)

Rerun the H1 and H2 headline numbers (τ=24h and 1w only, methods {raw, platt, binning10, venn_abers, conformal_marginal, conformal_mondrian}) under:
1. Price filter [0.02, 0.98] (from `forecasts_unfiltered.parquet`, apply `n_trades_market ≥ 10`).
2. Price filter [0.10, 0.90].
3. Volume filter `n_trades_market ≥ 100` (with primary price filter).
4. HistogramBinning `n_bins ∈ {5, 20}` (primary filters).

Reuse the walk-forward machinery; do not fork the code. Output `results/table_robustness.csv`: `variant, method, domain, tau, metric, value, delta_vs_primary`. Add one paragraph to `results/notes.md` stating whether the qualitative conclusions (sign of the H1 deltas, direction of the H2 gap) survive every variant.

Optional if time: per-year stability table `results/table_by_year.csv`.

Commit: `"Step 18: robustness sweeps"`

## Step 19 — Released artifact (`src/build_recal_table.py`)

Build `results/recalibration_table.csv`, the prospectively-validated analogue of Le's 216-cell matrix. For each (domain, tau) using the FULL archive as calibration data (this is the deployable table, so it uses everything; document that it is fit on all data while its validation is the walk-forward above):
- HistogramBinning(10): 10 rows per cell with `bin_lo, bin_hi, n, recal_prob, ci_lo, ci_hi`.
- Platt parameters `a, b` for the same cell (so a reader can compare to Le).
- Mondrian conformal threshold `q_hat` at α ∈ {0.05, 0.1, 0.2}.

Columns: `domain, tau, method, bin_idx, bin_lo, bin_hi, n, value, ci_lo, ci_hi, alpha, param_name`. Plus a `README_recalibration_table.md` explaining how to apply it: "given a Kalshi price p in domain d at horizon τ, find its bin, read `value`; for a 90% prediction set, include label k iff 1 − p̂_k ≤ q_hat(d, τ, 0.1)."

Commit: `"Step 19: released recalibration table"`

## Step 20 — Reproducibility and results audit

1. `run_all.py`: from `forecasts.parquet` (and unfiltered), regenerates every CSV and figure in `results/` in one command. Order: reproduce_le → run_single_split → walk_forward → hypothesis_tables → bootstrap → robustness → build_recal_table. Add a `--fast` flag that skips Venn–Abers subsampling and bootstrap for a smoke test.
2. Test it from a clean clone: `git clone` the repo to `E:\pm\tmp\clone-test`, copy in the two parquets, run `run_all.py --fast`, confirm it completes and the outputs match the committed CSVs (compare with `pandas.testing.assert_frame_equal` where deterministic).
3. `results_needed.md`: list every table and figure the paper requires and the file that provides it. Any file in `results/` not on the list gets flagged (scope creep) and either mapped or deleted. Any list entry without a file is a gap; report it.
4. Update `README.md` with a "Phase 4–5 closed" section: horizons run, start quarter, headline numbers for H1/H2/H3 with CIs, robustness verdict, and reproduction instructions.
5. Push. Tag `phase5-done`.

Commit: `"Step 20: run_all reproduction + results audit"`

---

## Review gates (stop and wait for the user)

1. **After Step 15:** report the chosen start quarter, per-quarter counts, which horizons ran, and one sanity number: rolling Mondrian coverage at α=0.1 pooled over all quarters should be in [88%, 92%]. Outside that is a bug.
2. **After Step 16:** show the H1, H2, H3 tables and figures with a one-line read of each. Do not interpret beyond that; the user writes the interpretation.
3. **After Step 20:** the results_needed.md audit and the clean-clone reproduction result.

## Rules that bite in this phase

- Leakage is still the top risk, now multiplied across ~15 windows. The assertion in 15a is not optional.
- H3 static-once must use ONLY the first calibration window. If its coverage doesn't drift at all, check that it isn't secretly refitting.
- ACI: `err_t` is computed against the ACTUAL outcome, which is only known after resolution; processing in `close_time` order within a quarter respects that. Never let a future outcome update `alpha_t`.
- Bootstrap resamples tickers. Never rows.
- Report surprises; don't tune them away. A genuine null on H1 in some domain is a finding.

## Definition of done

- [ ] `walk_forward_long.csv` with τ ≥ {24h, 1w}, leakage assertion passing every window
- [ ] ACI implemented and tested; static-once, rolling, ACI curves exist
- [ ] `table_H1.csv`, `table_H2.csv`, `table_H3.csv` and their figures, with standalone captions
- [ ] `table_bootstrap_ci.csv` with 95% CIs on every headline delta
- [ ] `table_robustness.csv` with a stated verdict
- [ ] `recalibration_table.csv` + its README
- [ ] `run_all.py` reproduces everything from a clean clone
- [ ] `results_needed.md` audit clean; README updated; tag `phase5-done` pushed
