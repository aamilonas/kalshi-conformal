"""Step 17 — market-clustered bootstrap confidence intervals.

Rows within a market are correlated: the same market appears at several
horizons, and within a horizon every method scores the same realised
outcome. Resampling rows would treat those as independent draws and shrink
every interval. **We resample markets (tickers) with replacement, never
rows** — the same clustering Le uses.

Reads `data/derived/wf_predictions.parquet` (written by Step 15), so the
walk-forward is never rerun here.

Three families of comparison, all at tau in {24h, 1w}:

* **H1** dBrier and dLogLoss of each recalibrator vs the raw price, per
  domain and ALL, for both fit scopes. These are the headline: the H1
  result is carried by the sentence "not significantly different from
  zero", so `significant` and the interval must be readable straight off
  the table. They lead the file.
* **H2** dCoverage (Mondrian - marginal) per domain at alpha = 0.1.
* **H3** dMAD-from-nominal, (ACI - static-once) and (rolling - static-once)
  per domain.

Why the H1 and H2 statistics resample so cheaply: each is a mean of a
PER-MARKET difference. Brier is a mean of (p - y)^2, so

    dBrier = mean_i[(p_m,i - y_i)^2 - (p_raw,i - y_i)^2]

is the mean of one number per market. Resampling markets is then just
resampling that vector. H3's MAD is not a plain mean — it averages
|per-quarter coverage - 0.9| over quarters — so each replicate re-derives
the per-quarter means with a weighted bincount.

Seed 20260818; 1,000 replicates; 95% percentile intervals.

One caveat on the H3 intervals. dBrier, dLogLoss and dCoverage are means,
whose bootstrap distribution is centred on the sample value, so their
percentile intervals bracket the point estimate. MAD-from-nominal is not a
mean: it averages an absolute value of noisy per-quarter coverage, and
|.| is convex, so resampling inflates it (Jensen). The bias mostly cancels
in a DIFFERENCE of two MADs computed on the same resampled markets, but
not exactly -- in this run it leaves the point estimate a hair outside its
interval in 1 of 258 rows (excursion 1.6e-04 against a CI width of
7.3e-03, about 2%). The `boot_median` column reports the median of the
replicate distribution so the size of that shift is visible per row rather
than implicit. Read H3 intervals as "is the gap distinguishable from
zero", which is what they are for, not as a precise location for the gap.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from paths import DERIVED, RESULTS

SEED = 20260818
N_BOOT = 1000
CI = (2.5, 97.5)
NOMINAL = 0.9
ALPHA_PRIMARY = 0.1
TAUS = ["24h", "1w"]
DOMAINS = ["Sports", "Crypto", "Politics", "Finance", "Weather",
           "Entertainment"]
RECALS = ["platt", "isotonic", "binning10", "venn_abers"]
EPS = 1e-6


def _brier(p, y):
    return (p - y) ** 2


def _logloss(p, y):
    p = np.clip(p, EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log1p(-p))


def boot_mean(d, rng, n_boot=N_BOOT):
    """Percentile CI for the mean of a per-market vector."""
    n = len(d)
    if n < 2:
        return (float("nan"),) * 4
    point = float(d.mean())
    reps = np.empty(n_boot)
    for b in range(n_boot):
        reps[b] = d[rng.integers(0, n, n)].mean()
    lo, hi = np.percentile(reps, CI)
    return point, float(lo), float(hi), float(np.median(reps))


def _mad(cov, qidx, n_q, w=None):
    """Mean over quarters of |quarter coverage - nominal|."""
    if w is None:
        num = np.bincount(qidx, weights=cov, minlength=n_q)
        den = np.bincount(qidx, minlength=n_q).astype(float)
    else:
        num = np.bincount(qidx, weights=cov * w, minlength=n_q)
        den = np.bincount(qidx, weights=w, minlength=n_q)
    ok = den > 0
    return float(np.abs(num[ok] / den[ok] - NOMINAL).mean())


def boot_mad_diff(cov_a, cov_b, qidx, n_q, rng, n_boot=N_BOOT):
    """CI for MAD(a) - MAD(b), resampling markets."""
    n = len(cov_a)
    if n < 2:
        return (float("nan"),) * 4
    point = _mad(cov_a, qidx, n_q) - _mad(cov_b, qidx, n_q)
    reps = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        w = np.bincount(idx, minlength=n).astype(float)
        reps[b] = _mad(cov_a, qidx, n_q, w) - _mad(cov_b, qidx, n_q, w)
    lo, hi = np.percentile(reps, CI)
    return point, float(lo), float(hi), float(np.median(reps))


def _row(family, comparison, metric, tau, domain, scope, alpha, pt, lo, hi,
         med, n_markets, n_quarters):
    return dict(family=family, comparison=comparison, metric=metric, tau=tau,
                domain=domain, fit_scope=scope, alpha=alpha,
                point_estimate=pt, ci_lo=lo, ci_hi=hi, boot_median=med,
                significant=bool(np.isfinite(lo) and np.isfinite(hi)
                                 and (lo > 0 or hi < 0)),
                n_markets=int(n_markets), n_quarters=int(n_quarters))


def load_preds():
    return pd.read_parquet(f"{DERIVED}/wf_predictions.parquet")


def _wide(P, tau, scope, methods, value="p_hat"):
    """One row per market, one column per method. Markets missing any
    method are dropped so every comparison is paired on the same set."""
    s = P[(P.tau == tau) & (P.fit_scope == scope) & P.method.isin(methods)]
    if s.empty:
        return None
    w = s.pivot_table(index=["ticker", "domain", "test_quarter", "y"],
                      columns="method", values=value, aggfunc="first")
    w = w.dropna(subset=[m for m in methods if m in w.columns])
    if not set(methods) <= set(w.columns) or w.empty:
        return None
    return w.reset_index()


def h1_rows(P, rng):
    out = []
    for tau in TAUS:
        for scope in ["pooled", "per_domain"]:
            w = _wide(P, tau, scope, ["raw"] + RECALS)
            if w is None:
                continue
            y = w["y"].to_numpy(dtype=float)
            base = {"brier": _brier(w["raw"].to_numpy(), y),
                    "log_loss": _logloss(w["raw"].to_numpy(), y)}
            for meth in RECALS:
                p = w[meth].to_numpy()
                per_market = {"brier": _brier(p, y) - base["brier"],
                              "log_loss": _logloss(p, y) - base["log_loss"]}
                for dom in DOMAINS + ["ALL"]:
                    sel = (slice(None) if dom == "ALL"
                           else (w["domain"] == dom).to_numpy())
                    nq = w.loc[sel, "test_quarter"].nunique() \
                        if dom != "ALL" else w["test_quarter"].nunique()
                    for metric, d in per_market.items():
                        dd = d if dom == "ALL" else d[sel]
                        if len(dd) < 2:
                            continue
                        pt, lo, hi, med = boot_mean(dd, rng)
                        out.append(_row(
                            "H1", f"{meth} - raw", f"d{metric}", tau, dom,
                            scope, "NA", pt, lo, hi, med, len(dd), nq))
    return out


def h2_rows(P, rng):
    out = []
    cov = P[(P.alpha == ALPHA_PRIMARY)
            & P.method.isin(["conformal_marginal", "conformal_mondrian"])].copy()
    cov["covered"] = np.where(cov.y.to_numpy() == 1, cov.set1, cov.set0)
    for tau in TAUS:
        s = cov[cov.tau == tau]
        w = s.pivot_table(index=["ticker", "domain", "test_quarter"],
                          columns="method", values="covered",
                          aggfunc="first").dropna().reset_index()
        if w.empty:
            continue
        d_all = (w["conformal_mondrian"].to_numpy()
                 - w["conformal_marginal"].to_numpy())
        for dom in DOMAINS + ["ALL"]:
            sel = (np.ones(len(w), bool) if dom == "ALL"
                   else (w["domain"] == dom).to_numpy())
            if sel.sum() < 2:
                continue
            pt, lo, hi, med = boot_mean(d_all[sel], rng)
            out.append(_row("H2", "mondrian - marginal", "dcoverage", tau,
                            dom, "na", ALPHA_PRIMARY, pt, lo, hi, med,
                            int(sel.sum()),
                            w.loc[sel, "test_quarter"].nunique()))
    return out


def h3_rows(P, rng):
    out = []
    meths = ["static_once", "conformal_mondrian", "aci_mondrian_g0.005"]
    cov = P[(P.alpha == ALPHA_PRIMARY) & P.method.isin(meths)].copy()
    cov["covered"] = np.where(cov.y.to_numpy() == 1, cov.set1, cov.set0)
    for tau in TAUS:
        s = cov[cov.tau == tau]
        w = s.pivot_table(index=["ticker", "domain", "test_quarter"],
                          columns="method", values="covered",
                          aggfunc="first").dropna().reset_index()
        if w.empty:
            continue
        qs = sorted(w.test_quarter.unique())
        qmap = {q: i for i, q in enumerate(qs)}
        qidx_all = w.test_quarter.map(qmap).to_numpy()
        for dom in DOMAINS + ["ALL"]:
            sel = (np.ones(len(w), bool) if dom == "ALL"
                   else (w["domain"] == dom).to_numpy())
            if sel.sum() < 2:
                continue
            qidx = qidx_all[sel]
            for a, label in [("aci_mondrian_g0.005", "aci - static_once"),
                             ("conformal_mondrian", "rolling - static_once")]:
                pt, lo, hi, med = boot_mad_diff(
                    w[a].to_numpy()[sel], w["static_once"].to_numpy()[sel],
                    qidx, len(qs), rng)
                out.append(_row("H3", label, "dMAD_from_nominal", tau, dom,
                                "na", ALPHA_PRIMARY, pt, lo, hi, med,
                                int(sel.sum()), len(np.unique(qidx))))
    return out


def main():
    rng = np.random.default_rng(SEED)
    P = load_preds()
    print(f"loaded wf_predictions ({len(P):,} rows, "
          f"{P.ticker.nunique():,} distinct markets)")
    rows = h1_rows(P, rng); print(f"  H1: {len(rows)} comparisons")
    r2 = h2_rows(P, rng);   print(f"  H2: {len(r2)} comparisons")
    r3 = h3_rows(P, rng);   print(f"  H3: {len(r3)} comparisons")
    tbl = pd.DataFrame(rows + r2 + r3)
    # H1 leads the file: its intervals carry the paper's headline null.
    tbl["_f"] = tbl.family.map({"H1": 0, "H2": 1, "H3": 2})
    tbl = tbl.sort_values(["_f", "tau", "fit_scope", "metric", "domain",
                           "comparison"]).drop(columns="_f")
    tbl.to_csv(f"{RESULTS}/table_bootstrap_ci.csv", index=False)
    print(f"Saved table_bootstrap_ci.csv ({len(tbl):,} rows)")
    h1 = tbl[tbl.family == "H1"]
    print(f"\nH1: {int(h1.significant.sum())} of {len(h1)} intervals exclude "
          f"zero ({h1.significant.mean():.1%})")
    for fam in ["H2", "H3"]:
        f = tbl[tbl.family == fam]
        print(f"{fam}: {int(f.significant.sum())} of {len(f)} exclude zero")
    return tbl


if __name__ == "__main__":
    main()
