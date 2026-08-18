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
