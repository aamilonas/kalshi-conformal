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
