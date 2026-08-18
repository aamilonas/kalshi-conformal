# How to use `recalibration_table.csv`

The deployable output of this project: given a Kalshi yes-price, it returns a
recalibrated probability and a distribution-free prediction set.

## What is in the file

One long table. Every row is keyed by `domain`, `tau` and `method`; which of
the remaining columns are populated depends on the method.

| `method` | `param_name` | populated columns | what `value` is |
|---|---|---|---|
| `binning10` | `recal_prob` | `bin_idx`, `bin_lo`, `bin_hi`, `n`, `ci_lo`, `ci_hi` | observed yes-rate of markets whose price fell in this bin |
| `platt` | `a_slope_on_logit_p`, `b_intercept` | `n` | a logistic coefficient |
| `conformal_mondrian` | `q_hat` | `n`, `alpha` | the nonconformity threshold at that `alpha` |

`domain` is one of Sports, Crypto, Politics, Finance, Weather, Entertainment
(Le's classifier; see the Phase 1 README for its quirks). `tau` is the
forecast horizon before market close: `6h`, `24h`, `1w`, `1mo`.

## Recipe 1 — a recalibrated probability

**Histogram binning** (the guarantee-bearing estimator). Take your price `p`,
find the row for your `(domain, tau)` where `bin_lo <= p <= bin_hi`, and read
`value`. `ci_lo`/`ci_hi` are Clopper-Pearson 95% intervals for that bin, and
`n` is how many markets it rests on — a bin with a wide interval is telling
you it does not know much.

Bin edges are equal-mass, so they are not round numbers and they differ by
cell. Edges are inclusive at both ends as published; a price on a shared
boundary belongs to the upper bin.

**Platt** if you want something smooth and monotone:

    logit(p) = log(p / (1 - p))
    p_recal  = 1 / (1 + exp(-(a * logit(p) + b)))

with `a` and `b` from the two `platt` rows of the same cell. `a` near 1 and
`b` near 0 means the market price is already close to calibrated in that
cell; that is the common case here — see the caveat below.

## Recipe 2 — a 90% prediction set

Split conformal, one threshold per domain (Mondrian). Let `p_hat_1 = p` and
`p_hat_0 = 1 - p` be the market's implied probabilities of yes and no. Look up
`q_hat` for your `(domain, tau)` at `alpha = 0.1`. Then

    include label k in the set  iff  1 - p_hat_k <= q_hat(domain, tau, 0.1)

The set can hold one label or both; both means the market is not confident
enough at this level to single one out. Use `alpha = 0.05` for a 95% set or
`0.2` for an 80% one. These thresholds are built on the raw price as the
underlying forecast, which is how they were validated.

### Worked example

Politics, `tau = 24h`, a market trading at **p = 0.72**.

* Binning: 0.72 falls in bin 7 (`bin_lo` 0.68, `bin_hi` 0.76), so the
  recalibrated probability is **0.567**, 95% CI [0.496, 0.637], from n = 201
  markets. The market price is well above the historical hit rate in this bin.
* Platt: a = 0.9814, b = -0.1739. `logit(0.72) = 0.9445`, so
  `0.9814 * 0.9445 - 0.1739 = 0.7530` and `p_recal = 0.680`.
* Prediction set at 90%: `q_hat = 0.69`. `1 - p_hat_1 = 0.28 <= 0.69`, so yes
  is in; `1 - p_hat_0 = 0.72 > 0.69`, so no is out. The set is **{yes}**.

The two recalibrators disagree here (0.567 vs 0.680) because binning is local
and noisy while Platt is a single monotone curve over the whole cell. That
spread is real uncertainty, not a bug — bin 7's confidence interval alone runs
from 0.50 to 0.64.

## Caveats you should read before applying this

1. **Fitted on the full archive.** Every number here is in-sample. The
   out-of-sample evidence lives in the walk-forward (`walk_forward_long.csv`,
   `table_H1.csv`, `table_bootstrap_ci.csv`), where the same estimators are
   refit each quarter and scored on later markets.
2. **Recalibration did not beat the raw price out of sample.** In the
   walk-forward, only 25 of 216 method x domain x horizon x scope cells
   improved on the raw Kalshi price on Brier score, histogram binning
   improved in none of 54, and the market-clustered bootstrap puts Platt's
   change indistinguishable from zero. Use this table to *quantify* how far a
   price sits from its historical hit rate; do not assume applying it makes
   your forecasts better. The prediction sets are on firmer ground — those
   come with a coverage guarantee that the walk-forward confirms.
3. **Coverage of the sets is conditional on the price band.** Thresholds were
   estimated on markets with `0.05 <= price <= 0.95` and at least 10 trades.
   Applying them to a market at 0.99 is extrapolation. The Step 18 sweeps also
   show the advantage of per-domain thresholds over one pooled threshold
   reverses under a wider [0.02, 0.98] band.
4. **Two cells are missing:** Crypto at `1w` (n = 148) and `1mo` (n = 78) fell
   below the 200-market floor for shipping a fitted cell. `tau = 1h` is absent
   from the whole table — it was never carried through the walk-forward, and
   the Phase 1 README documents a selection effect that makes it the least
   clean horizon.
5. **Sports, Crypto and Politics history is short.** Those products largely
   launched in 2024-25, so their cells rest on recent markets only. See the
   Limitations note in `results/notes.md`.

## Provenance

Built by `src/build_recal_table.py` from `data/derived/forecasts.parquet`
(241,342 rows, frozen at tag `phase1-done`). Regenerate with
`py run_all.py` or `py src/build_recal_table.py`.
