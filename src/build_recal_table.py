"""Step 19 — the released recalibration table.

`results/recalibration_table.csv` is the deployable analogue of Le's
216-cell matrix: for each (domain, tau) it ships the three things a user
needs to turn a Kalshi price into a calibrated probability or a prediction
set.

  binning10   ten equal-mass bins with the empirical outcome frequency and
              a Clopper-Pearson 95% interval per bin
  platt       the two logistic coefficients a, b on logit(p)
  conformal_mondrian
              the split-conformal threshold q_hat at alpha in
              {0.05, 0.1, 0.2}

**Fitted on everything, validated elsewhere.** Every cell here is fit on
the FULL archive, because a table meant for deployment should use all the
data it can. That means these numbers are in-sample and carry no
out-of-sample guarantee of their own. The evidence that the *procedure*
works prospectively is the walk-forward in Steps 15-17, where the same
estimators are refit each quarter and scored on markets that close later.
Read the two together: Steps 15-18 say what the method does, this table is
the method's output on all available history.

tau = 1h is deliberately absent. The Phase 1 README documents a selection
effect at that horizon (only 13,126 of 76,181 >=10-trade Crypto markets have
any trade one hour before close), and it was never carried through the
walk-forward, so nothing here would be validated. Shipping an unvalidated
cell in a table people are meant to apply would be worse than omitting it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from adaptive import conformal_qhat
from paths import DERIVED, RESULTS
from recalibrators import HistogramBinning, Platt

DOMAINS = ["Sports", "Crypto", "Politics", "Finance", "Weather",
           "Entertainment"]
TAUS = ["6h", "24h", "1w", "1mo"]          # the walk-forward horizons
ALPHAS = [0.05, 0.1, 0.2]
N_BINS = 10
MIN_CELL = 200        # below this a per-cell fit is not worth shipping

COLUMNS = ["domain", "tau", "method", "bin_idx", "bin_lo", "bin_hi", "n",
           "value", "ci_lo", "ci_hi", "alpha", "param_name"]


def _blank(**kw):
    row = {c: "" for c in COLUMNS}
    row.update(kw)
    return row


def build():
    fc = pd.read_parquet(f"{DERIVED}/forecasts.parquet")
    fc = fc[fc.domain.isin(DOMAINS) & fc.tau.isin(TAUS)]
    rows, skipped = [], []
    for tau in TAUS:
        for dom in DOMAINS:
            s = fc[(fc.tau == tau) & (fc.domain == dom)]
            p = s["price"].to_numpy()
            y = s["outcome"].to_numpy().astype(int)
            n = len(p)
            if n < MIN_CELL:
                skipped.append((dom, tau, n))
                continue

            hb = HistogramBinning(N_BINS).fit(p, y)
            bt = hb.bin_table()
            for _, b in bt.iterrows():
                rows.append(_blank(
                    domain=dom, tau=tau, method="binning10",
                    bin_idx=int(b["bin"]), bin_lo=b["lo"], bin_hi=b["hi"],
                    n=int(b["n"]), value=b["freq"], ci_lo=b["ci_lo"],
                    ci_hi=b["ci_hi"], param_name="recal_prob"))

            pl = Platt().fit(p, y)
            a = float(pl.clf_.coef_.ravel()[0])
            b0 = float(pl.clf_.intercept_.ravel()[0])
            for nm, v in [("a_slope_on_logit_p", a), ("b_intercept", b0)]:
                rows.append(_blank(domain=dom, tau=tau, method="platt",
                                   n=n, value=v, param_name=nm))

            scores = np.sort(np.where(y == 1, 1.0 - p, p))
            for al in ALPHAS:
                q = conformal_qhat(scores, al)
                if not np.isfinite(q):
                    skipped.append((dom, f"{tau} q_hat@{al}", n))
                    continue
                rows.append(_blank(
                    domain=dom, tau=tau, method="conformal_mondrian", n=n,
                    value=q, alpha=al, param_name="q_hat"))
    return pd.DataFrame(rows)[COLUMNS], skipped


def main():
    tbl, skipped = build()
    tbl.to_csv(f"{RESULTS}/recalibration_table.csv", index=False)
    print(f"Saved recalibration_table.csv ({len(tbl):,} rows)")
    cells = tbl.groupby(["tau", "method"]).size().unstack(fill_value=0)
    print("\nrows by tau x method:")
    print(cells.to_string())
    print(f"\ncells shipped: "
          f"{tbl.groupby(['domain','tau']).ngroups} of "
          f"{len(DOMAINS) * len(TAUS)} (domain x tau)")
    for dom, tau, n in skipped:
        print(f"  skipped {dom:<14} {tau:<14} n={n}")
    return tbl


if __name__ == "__main__":
    main()
