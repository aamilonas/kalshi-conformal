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
