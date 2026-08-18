# Phase 2–3: Reproduction & Recalibration Methods on One Split

## Context for Claude Code

Continuation of the Kalshi conformal recalibration paper. Phase 1 is complete: `data/derived/forecasts.parquet` (and `forecasts_unfiltered.parquet`) is frozen and tagged `phase1-done`. Everything in this phase reads ONLY from those two files. Do not touch `markets_dedup`, `trades_dedup`, or the raw Becker archive.

Read `PHASE1_DATA_ENGINEERING.md` and `README.md` first for context. Same environment rules apply: Windows, `py`, PowerShell, `.venv` in repo root, commit after every numbered step, every result lands as a CSV in `results/` before it becomes a figure or table.

**Goal of this phase:** by the end, `results/table_M2_main.csv` exists, the exit tests pass, and the user has a defensible single-split paper. Phase 4 (walk-forward) builds on the code written here.

**Files to create:**
```
src/metrics.py          # Step 11
src/recalibrators.py    # Step 13
src/reproduce_le.py     # Step 12
src/run_single_split.py # Step 14
tests/test_metrics.py
tests/test_recalibrators.py
```

**Notation used throughout:** `p` = market price in (0,1) treated as a probability forecast, `y` ∈ {0,1} outcome, `tau` = horizon, `domain` ∈ {Sports, Crypto, Politics, Finance, Weather, Entertainment}.

---

## Step 11 — Metrics module (`src/metrics.py`) with unit tests

Implement, all vectorized over numpy arrays:

- `brier(p, y) -> float`: mean of `(p - y)^2`.
- `log_loss(p, y) -> float`: clip `p` to `[1e-6, 1-1e-6]` first.
- `ece(p, y, n_bins=10) -> float`: expected calibration error with **equal-mass** bins (use `np.quantile` on `p` for bin edges, not equal-width). ECE = Σ_b (n_b/N) · |mean(y in b) − mean(p in b)|. Cite Guo et al. 2017 in the docstring.
- `coverage(sets, y) -> float`: `sets` is a boolean array of shape (n, 2) where `sets[i, k]` means label `k` is in the prediction set for observation i. Coverage = fraction of i where `sets[i, y[i]]` is True.
- `avg_set_size(sets) -> float`: mean of `sets.sum(axis=1)`.
- `reliability_diagram(p, y, n_bins=10, ax=None, label=None)`: plots mean predicted vs observed per equal-mass bin, with a y=x diagonal. Returns the axis. Also return the underlying bin table as a DataFrame so it can be saved to CSV.

`tests/test_metrics.py`, each on toy data with a known answer:
1. Perfect forecasts (`p = y`) → brier 0, log_loss ≈ 0, ece 0.
2. Constant `p = 0.5` on balanced `y` → brier 0.25, ece 0.
3. Constant `p = 0.9` on `y` all zeros → brier 0.81, ece 0.9.
4. Prediction sets that are always `{0,1}` → coverage 1.0, avg_set_size 2.0. Sets that are always `{y}` → coverage 1.0, size 1.0. Sets always `{1 - y}` → coverage 0.
5. `ece` bins are equal-mass: with `p` uniform on [0,1] and n=10000, every bin has ~1000 points.

Commit: `"Step 11: metrics module + tests"`

## Step 12 — Reproduce Le's logistic recalibration (`src/reproduce_le.py`)

Le (2026) fits `logit P(y=1) = a + b · logit(p)` per (domain × time-bin) cell and reports slope `b`. Slope > 1 means prices are underconfident (should be pushed toward extremes); < 1 means overconfident.

**Important:** Le's Table 3 uses their nine time bins on trade-level data, not our five horizons. Read `src/pipeline.py` and any calibration script in `E:\pm\prediction-market-calibration` to get their exact bin edges and their exact filter choices, and reproduce those for this step. Our five-horizon `forecasts.parquet` is used for everything else; this step may need its own extraction from `trades_dedup.parquet` following Le's definition. If so, write it as `src/le_time_bins.py`, keep it isolated, and do NOT let it modify `forecasts.parquet`.

Fit: `sklearn.linear_model.LogisticRegression(C=10, penalty='l2')` on the single feature `logit(p)`, where `logit(p) = log(p / (1-p))`. Extract `coef_[0][0]` as slope `b`, `intercept_[0]` as `a`. Record `n` per cell.

Also compute slopes on OUR five horizons from `forecasts.parquet` (domain × tau grid), because that's the grid the rest of the paper uses and it should tell the same qualitative story.

**Compare against Le:** load Le's published 216-cell calibration matrix CSV from the cloned repo. Match cells by (domain, time-bin). Produce:
- `results/table_reproduction.csv`: columns `domain, time_bin, n_ours, slope_ours, slope_le, delta`.
- `results/fig_reproduction.pdf` and `.png`: scatter of `slope_ours` vs `slope_le`, y=x line, points labeled or colored by domain.
- `results/table_slopes_ours_5h.csv`: our domain × tau slope grid.

**PASS conditions** (direction and magnitude, not exact decimals):
- Politics ≈ 1.8 at the 1-week to 1-month bins.
- Weather < 1 at short horizons.
- Sports ≈ 0.9–1.1 at short/medium horizons.
- Same qualitative ranking of domains as Le.
- Scatter hugs the y=x line; report the correlation.

If a specific domain is far off while others match, suspect the time-bin definition or a filter mismatch before anything else. Report the pass/fail per condition to the user.

Commit: `"Step 12: reproduction of Le slopes — [PASS/FAIL summary]"`

## Step 13 — Seven recalibrators (`src/recalibrators.py`) with tests

Common interface for every method:
```python
class Recalibrator:
    def fit(self, p_cal: np.ndarray, y_cal: np.ndarray) -> "Recalibrator": ...
    def predict_proba(self, p_test: np.ndarray) -> np.ndarray:  # P(y=1), shape (n,)
    def predict_set(self, p_test: np.ndarray, alpha: float) -> np.ndarray:  # bool (n, 2); optional
```

1. **`Raw`**: `predict_proba` returns `p_test` unchanged. Baseline.
2. **`Platt`**: logistic regression on `logit(p)`, same spec as Step 12 (C=10). Fit on calibration fold only.
3. **`Isotonic`**: `sklearn.isotonic.IsotonicRegression(out_of_bounds='clip')`, fit `p_cal -> y_cal`.
4. **`HistogramBinning(n_bins=10)`**: equal-mass bin edges from `np.quantile(p_cal, ...)`. Recalibrated probability for a test point = empirical outcome frequency of its bin. Store per-bin counts and successes so you can also return Clopper–Pearson 95% intervals per bin (`scipy.stats.beta.ppf`). Expose `bin_table()` returning edges, n, successes, freq, ci_lo, ci_hi. This is the guarantee-bearing estimator (Gupta et al. 2020; Gupta & Ramdas 2021).
5. **`VennAbers`** (inductive): for each test point with score `p_t`, fit isotonic regression on `p_cal ∪ {p_t}` twice, once with the test label 0 and once with 1. Read off the fitted value at `p_t` in each: `p0` and `p1`. Point prediction `p = p1 / (1 - p0 + p1)`; also return interval width `p1 - p0` via a `predict_interval()` method. Correctness before speed. Naive implementation is O(n_test × isotonic_fit); if that's too slow, first try running on a stratified subsample of the test fold and report that; do not silently approximate. Vectorization tips: sort `p_cal` once; use `sklearn.isotonic.IsotonicRegression` fresh per point, it's fine.
6. **`SplitConformal(base: Recalibrator)`**: wraps any of the above (default: `Raw`). Nonconformity score `s = 1 - p̂_y`, where `p̂_y` is the base method's probability for the TRUE label. On calibration fold compute scores, then threshold `q̂` = the `ceil((n+1)(1-alpha))/n` empirical quantile. Test-time set = `{k : 1 - p̂_k ≤ q̂}`. Two modes:
   - **marginal**: one `q̂` pooled across all domains.
   - **mondrian**: one `q̂` per domain; test point uses its own domain's threshold.
7. **`Adaptive` (ACI)**: DEFER to Phase 4. Stub the class with `raise NotImplementedError("Phase 4")` so the interface exists.

`tests/test_recalibrators.py`:
- Every method: `predict_proba` outputs are in [0,1] and shape (n,).
- `Platt`/`Isotonic`/`HistogramBinning`: on synthetic data where `y ~ Bernoulli(f(p))` for a known miscalibrating `f` (e.g. `f(p) = p^2`), the recalibrated Brier is lower than raw Brier.
- `HistogramBinning`: bin frequencies on the calibration fold equal `mean(y_cal)` within each bin exactly.
- `VennAbers`: `p0 ≤ p1` for every test point; point prediction lies in `[p0, p1]`.
- `SplitConformal` marginal: on i.i.d. synthetic data with n_cal = 5000, n_test = 5000, alpha = 0.1, coverage in [0.885, 0.915]. This test is the canary; if it fails, the implementation is wrong.
- `SplitConformal` mondrian: on synthetic data with two groups of very different difficulty, per-group coverage is each ≥ ~0.88 while marginal-mode per-group coverage differs substantially between groups.

Commit: `"Step 13: seven recalibrators + tests"`

## Step 14 — Single-split benchmark (`src/run_single_split.py`)

**Split rule, non-negotiable:** chronological by market `close_time`. Calibration fold = markets closing before `2024-01-01`; test fold = markets closing in 2024–2025. NEVER random. Any test-fold outcome touching a fit is leakage and invalidates the paper.

Run at `tau = 24h`, Kalshi, all six domains, from `forecasts.parquet`.

For each method in {Raw, Platt, Isotonic, HistogramBinning(10), VennAbers, Conformal-marginal(Raw), Conformal-mondrian(Raw)}, and separately for **pooled** (all domains) and **per-domain** fits:
- Compute Brier, log loss, ECE on the test fold.
- For conformal variants: coverage and avg set size at alpha ∈ {0.1 (primary), 0.05, 0.2}.
- Report per-domain test coverage for BOTH marginal and mondrian (this is the H2 preview).

Outputs:
- `results/table_M2_main.csv`: long tidy format, columns `method, fit_scope (pooled|per_domain), domain (or 'ALL'), tau, alpha (or NA), metric, value, n_test`.
- `results/fig_reliability_M2.png` and `.pdf`: 2×2 panel, Politics and Sports rows, Raw vs Platt vs VennAbers overlaid in each panel, from `metrics.reliability_diagram`.
- `results/table_reliability_bins_M2.csv`: the underlying bin tables.
- `results/table_binning_ci_M2.csv`: HistogramBinning `bin_table()` per domain (this seeds the paper's released recalibration table).

**Exit tests (report explicitly, in order):**
1. Conformal **marginal** coverage on the pooled test fold at alpha=0.1 ∈ [88.5%, 91.5%]. Outside → BUG. Stop, fix, do not interpret anything else.
2. VennAbers or HistogramBinning Brier ≤ Raw Brier in Politics. If NOT: investigate for bugs first (leakage, wrong fold, wrong label). If clean, report it as a genuine null result; that's a finding, not a failure.
3. Per-domain marginal coverage varies across domains (some domains under-covered, some over) while mondrian per-domain coverage is roughly uniform near 90%. If mondrian isn't fixing it, check that the per-domain quantile is actually being applied.

Present the main table and the three exit-test verdicts to the user and wait for review.

Commit: `"Step 14: single-split benchmark — exit tests [PASS/FAIL]"`; tag `phase3-done`.

---

## Rules that bite in this phase

- **Leakage** is the top risk. Every `fit()` call must see only calibration-fold data. Add an assertion in `run_single_split.py` that `max(close_time) of cal fold < min(close_time) of test fold`.
- **Coverage violations are bugs until proven otherwise.** Marginal split conformal is near-guaranteed by construction; if it's off, it's the code.
- Don't run analyses not mapped to a table above. If something looks interesting, note it in `results/notes.md` for later; do not chase it now.
- If any step surprises you numerically, stop and report the number and the query, rather than adjusting until it looks right.

## Definition of done

- [ ] `metrics.py` + tests green
- [ ] Le slopes reproduced; `table_reproduction.csv` and `fig_reproduction.*` exist; per-condition PASS/FAIL reported
- [ ] Seven recalibrators implemented (ACI stubbed); tests green including the conformal coverage canary
- [ ] `table_M2_main.csv` and reliability figure exist; three exit tests reported and reviewed by user
- [ ] One commit per step; `phase3-done` tag
