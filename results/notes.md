# Side observations parked for later phases (not chased now)

## 2026-08-18 — Step 12: market-level vs trade-level Politics slopes diverge

Le's trade-level, contract-weighted grid gives Politics 1w-1mo slope = 1.833
(the paper's headline underconfidence). Our market-level five-horizon grid
(one forecast per market = last trade before close - tau, unweighted) gives
Politics tau=1w slope = 0.871 (n=1,152) and tau=1mo = 1.120 (n=703).

Not a bug candidate: the 216-cell trade-level reproduction is exact to
machine precision, so both numbers are faithfully computed from the same
data. The estimands differ on three axes: (a) one snapshot per market vs
all trades pooled within a time window; (b) unweighted vs contract-weighted
(whale trades dominate Politics contract weight — Le's own appendix D
documents the size effect); (c) point-in-time (exactly 168h before close)
vs window [168h, 720h).

Implication to keep in mind for the paper: "Politics is underconfident" is
partly a trade/contract-weighting phenomenon and may attenuate at the
market level. Phase 4+ should decide which estimand the recalibration
claims are about, and possibly show both.

## 2026-08-18 — Step 14: three findings from the single split (tau=24h)

1. **Original split boundary infeasible.** cal < 2024-01-01 leaves 3/0/14
   calibration markets in Sports/Crypto/Politics (products launched
   2024-25). Per user decision: primary boundary 2025-07-01 (all domains
   feasible), original boundary kept as pooled-only robustness
   (table_M2_robustness_spec_split.csv).
2. **Conformal overcoverage is temporal shift, not a bug.** Marginal pooled
   coverage at alpha=.1 is 0.9223 vs the [0.885, 0.915] exit band. In-sample
   calibration-fold coverage at qhat is 0.9053 (= 0.90 + discrete-price tie
   mass; the implementation's finite-sample behavior is exactly nominal);
   the remaining +0.0170 is H1->H2 2025 distribution shift (test scores
   stochastically smaller). Direction is conservative (>= 1-alpha holds).
   This is the empirical motivation for Phase 4's adaptive conformal (ACI):
   a static split cannot track Kalshi's drift.
3. **Politics recalibration null at market level.** Politics cal/test base
   rates are 0.459/0.460 with mean price ~= base rate; Raw is already
   calibrated (slope 0.981 at 24h in table_slopes_ours_5h.csv). VA/HB tie
   Raw on Brier within 1% of its SE while slightly improving ECE; Platt
   improves all three metrics (0.17876 vs 0.18017 Brier). Consistent with
   note above: trade-level underconfidence (slope 1.833) does not manifest
   in market-level 24h snapshots. A genuine null, not a failure.

## 2026-08-18 — Phase 4 prep: Step 14 re-verified on a second platform

Phases 0-3 were built on Windows; Phase 4-5 runs on macOS off the same
exFAT drive (`.venv` is a Windows venv and cannot be used; the pinned
`requirements.txt` was installed into a CPython 3.13.15 venv instead, and
`src/paths.py` replaced the 22 hardcoded `E:/pm/...` paths with
repo-root-relative ones). Both frozen parquets hash-match the README
(`644C7CA4...`, `D504C1CB...`). Full suite 53/53 green after the rewiring.

Re-running `run_single_split.py` reproduces Step 14 exactly:
`table_M2_main.csv`, `table_binning_ci_M2.csv` and
`table_reliability_bins_M2.csv` agree to <= 2e-16 on every numeric cell,
labels identical. Exit-test verdicts identical (0.9223 / Politics null /
spread 0.077 -> 0.052).

**One cell does not reproduce, and it is a warning about ECE.**
`table_M2_robustness_spec_split.csv`, VennAbers/Finance/ECE: 0.018486
(Windows) vs 0.018201 (macOS), |diff| 2.85e-04. Cause: Venn-Abers output is
massively tied — 2,894 Finance test rows take only **33 distinct** p-hat
values, and **44% of rows sit exactly on an interior equal-mass bin edge**.
A 1-ulp relative jitter of p-hat moves this ECE by up to **5.7e-03**, twenty
times the observed cross-platform difference. So the difference is numerical
dust, but the implication is not: **ECE computed with equal-mass bins on
Venn-Abers output is not stable below ~5e-03.** Phase 4 must not read ECE
deltas for VennAbers (or HistogramBinning, same tie structure) finer than
that, and Step 20's clean-clone comparison needs a tolerance on those cells
rather than exact equality.

## 2026-08-18 — Phase 4 prep: exit test 1 overcoverage is drift, not domain mix

The Step 14 story (DATA_GUIDE 6.3) is that pooled marginal coverage 0.9223
decomposes into 0.9053 in-sample + 0.0170 of "H1->H2 2025 temporal shift".
The obvious alternative explanation was never tested: the domain mix moves
violently across the 2025-07-01 boundary — Sports is 8.0% of the
calibration fold and 73.7% of the test fold — so the gap could be
composition, not drift. It is not.

Verified independently of `run_single_split.py`, at qhat = 0.700
(rank 25,779 / 28,642):

  P_cal(s <  qhat) = 0.8988      P_cal(s == qhat) = 0.0065  (cents-grid ties)
  P_cal(s <= qhat) = 0.9053      P_test(s <= qhat) = 0.9223

Oaxaca-style split of the +0.0170:

  composition (domain mix)   +0.0013
  within-domain drift        +0.0163      <- 96% of the gap
  interaction                -0.0006

Reweighting the test fold to the calibration fold's domain mix leaves
coverage at 0.9216 (vs 0.9223 observed). The mix barely matters because
calibration-fold coverage is already similar across domains (0.879-0.916).

Direction is consistent: five of six domains drift the same way
(Sports +0.016, Politics +0.030, Finance +0.017, Weather +0.019,
Entertainment +0.016); only Crypto moves against it (-0.022, and Crypto is
the one domain that was *under*-covered at 0.857). A systematic, near-common
shift in the score distribution is exactly what ACI is designed to track,
so the Phase 4 H3 motivation stands on a verified premise rather than an
assumed one.

Leakage re-asserted independently: cal max close_time 2025-06-30 21:07:13Z
< test min 2025-07-01 03:52:32Z; 28,642 + 31,465 = 60,107 rows, none lost.

## 2026-08-18 — Step 16: a degenerate Mondrian threshold was inflating coverage

**Found while eyeballing the first draft of fig_H3_coverage_time**, where the
static-once curves for Sports and Politics sat at exactly 1.000 across every
quarter. Coverage of exactly 1.000 with average set size exactly 2.000 is the
signature of "every prediction set is {0, 1}", not of a well-behaved policy.

**Mechanism.** A Mondrian split-conformal threshold is the
`ceil((n+1)(1-alpha))/n` empirical quantile of that group's calibration
scores. When a group's calibration set is smaller than that rank demands --
`ceil((n+1)(1-alpha)) > n`, i.e. n < 9 at alpha=0.1 and n < 19 at alpha=0.05 --
no such order statistic exists and q_hat is `+inf`, so every label is admitted.
The Phase 3 `SplitConformal` fell back to the pooled threshold for a group
*absent* from calibration (n = 0) but not for a group merely *too small*
(n = 1, 2, 3), which is strictly worse. `StaticOnce` and `ACI` inherited the
same hole.

**Blast radius, measured before fixing anything.**

* Walk-forward static-once: the frozen 2022Q2 window had Sports n_cal=3 and
  Politics n_cal=1, so **25.3% of tau=24h test rows (9,704 / 38,346) and 35.8%
  at tau=6h (17,329 / 48,360) were covered by construction.** Pooled
  static-once coverage in 2025Q3 read 0.9612; the true value is 0.9132.
* Rolling Mondrian: 14 windows affected across 24h and 6h, all with tiny
  n_test (1-30 rows), so the pooled rolling numbers moved by <= 0.003.
* **The gate-1 report was wrong because of this.** It described a
  two-directional pattern -- static-once drifting *up* at tau=24h and *down* at
  tau=1w. The tau=1w decay is real (0 infinite thresholds there; static-once
  falls to 0.8334 by 2025Q3). The tau=24h "upward drift" was the artifact and
  largely disappears once fixed: 2025Q1/Q2/Q3 go 0.9086/0.9344/0.9612 ->
  0.8973/0.8985/0.9132.
* Phase 3 was affected too: `table_M2_robustness_spec_split.csv` (the
  cal < 2024-01-01 boundary, where Sports n_cal=3 and Politics n_cal=14) had
  Conformal-mondrian Sports coverage 1.000 / set size 2.000 at all three
  alphas, dragging pooled coverage to 0.9547 at alpha=0.1. Corrected values:
  Sports 0.9280, pooled 0.9186. The regenerated CSV is committed. **No Phase 3
  exit-test verdict changes** -- all three run on the primary 2025-07-01 split,
  whose smallest per-domain calibration set is 739, and that table reproduces
  to 1e-16.

**Fix.** Fall back to the pooled threshold whenever a group's own threshold is
not finite, in `SplitConformal.predict_set`, `StaticOnce.qhat` and
`ACI._scores_for`; record the fallback in `fellback_`. This is also what a
deployer holding one global threshold would actually do. Three regression
tests now construct a 3-market group and assert the sets are not all {0, 1}.

**Provenance is now in the output.** `table_H3.csv` carries
`static_once_n_cal_frozen` and `static_once_borrowed_pooled` per (tau, domain),
so a reader can see that the tau=24h static-once curves for Politics, Sports,
Crypto and Entertainment rest on 1, 3, 0 and 25 calibration markets
respectively and borrow the pooled threshold where they must.

**Lesson worth keeping:** coverage == 1.000 and average set size == 2.000 are
not "good results", they are a null-threshold alarm. Any conformal table should
be scanned for them before it is read.

## 2026-08-18 — Step 18: do the qualitative conclusions survive the sweeps?

Required verdict paragraph. Four variants at tau in {24h, 1w}, each compared
to the primary over the INTERSECTION of their quarterly schedules (a filter
changes how many markets clear the 1,000-row calibration floor, so schedules
differ; `n_quarters` in `table_robustness.csv` reports what is left).
`trades_ge_100` loses 6 of 14 quarters at tau=24h and `price_10_90` loses 2 of
4 at tau=1w, which is the main limit on what these sweeps can say.

**H1 — the sign of the deltas survives, with one exception.** Under
`price_02_98`, `price_10_90`, `bins5` and `bins20`, no recalibrator beats the
raw Kalshi price on Brier at either horizon; binning is worse the coarser it
gets (binning5 - raw = +0.0066 at 24h, binning20 - raw = +0.0011). The one
flip is **Platt under `trades_ge_100` at tau=24h, which beats raw by
0.000287** (0.192692 vs 0.192979, n=15,423 over 8 quarters). Two things make
this a weak exception rather than a counter-example. It is the same size as
the primary's own uncertainty -- the Step 17 bootstrap puts primary Platt at
+0.000119 with a 95% CI of [-0.000004, +0.000244], i.e. a width of 0.00025 --
and Platt is exactly the method whose interval straddles zero everywhere
else. The consistent reading across Steps 16-18 is that **Platt ties the raw
price and the flexible methods lose to it**; nothing here shows Platt
winning. No bootstrap was run on the variants, so this flip has no interval
of its own.

**H2 — the direction does NOT survive one variant.** Mondrian sits closer to
nominal than a marginal threshold under `price_10_90` (MAD 0.0198 -> 0.0140 at
24h, 0.0250 -> 0.0211 at 1w) and `trades_ge_100` (0.0177 -> 0.0152; 0.0273 ->
0.0209), reproducing the primary result. It **reverses under `price_02_98`**
at both horizons: MAD 0.0123 -> 0.0144 at 24h and 0.0136 -> 0.0165 at 1w --
i.e. with the widest price band, per-domain thresholds land further from
nominal than one pooled threshold. Note that the marginal MAD under
`price_02_98` (0.0123) is already the smallest of any variant: admitting
prices in [0.02, 0.05) and (0.95, 0.98] adds a mass of near-certain markets
that every domain gets right, which flattens the between-domain differences
Mondrian exists to exploit while still costing it the variance of estimating
six thresholds instead of one. **H2 should be stated as conditional on the
price band, not as unconditional.**

Two bugs were fixed in this step before these numbers were trusted, both
recorded because they are the kind that fail silently:

1. `walk_forward` emits `alpha` as the string `"NA"` for probability metrics
   in memory, which only becomes NaN after a CSV round-trip. The sweep
   filtered on `alpha.isna()`, so **every probability-metric row was silently
   dropped** and the first `table_robustness.csv` had 168 rows of conformal
   output instead of 1,290. Fixed with an explicit `pd.to_numeric(...,
   errors="coerce")`.
2. Because of (1) the H1 slice was empty, and an empty loop reported
   "H1 sign HOLDS in every variant" -- a **vacuous pass**. The verdict now
   raises if it has nothing to compare. A robustness check that cannot fail
   is worse than no robustness check.
