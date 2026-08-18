"""Step 18 — robustness sweeps over the filters and the bin count.

Reruns the H1 and H2 headline numbers at tau in {24h, 1w} under four
variants, reusing `walk_forward.run_window` rather than forking it:

  price_02_98    price in [0.02, 0.98], n_trades >= 10   (unfiltered source)
  price_10_90    price in [0.10, 0.90], n_trades >= 10   (unfiltered source)
  trades_ge_100  primary price band, n_trades >= 100     (unfiltered source)
  bins5 / bins20 primary filters, HistogramBinning(5) and (20)

The bin-count variants change nothing set-valued, so they skip the
conformal comparators and report only the binning rows.

**Comparability.** Each variant re-derives its own quarterly schedule,
because a filter changes how many markets clear the 1,000-row calibration
floor. Aggregating a variant over its own quarters and the primary over
its would compare different periods. Every delta_vs_primary here is
therefore computed over the INTERSECTION of that variant's quarters with
the primary's, and `n_quarters` reports how many that leaves.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from paths import DERIVED, RESULTS
import walk_forward as wf

TAUS = ["24h", "1w"]
DOMAINS = wf.DOMAINS
ALPHA = 0.1
PROB = ["raw", "platt", "binning10", "venn_abers"]
CONF = ["conformal_marginal", "conformal_mondrian"]
PROB_METRICS = ["brier", "log_loss", "ece"]
CONF_METRICS = ["coverage", "avg_set_size"]
UNFILTERED = f"{DERIVED}/forecasts_unfiltered.parquet"

VARIANTS = [
    dict(name="price_02_98", label="price in [0.02, 0.98], trades >= 10",
         load=dict(path=UNFILTERED, price_lo=0.02, price_hi=0.98,
                   min_trades=10), run=dict()),
    dict(name="price_10_90", label="price in [0.10, 0.90], trades >= 10",
         load=dict(path=UNFILTERED, price_lo=0.10, price_hi=0.90,
                   min_trades=10), run=dict()),
    dict(name="trades_ge_100", label="price in [0.05, 0.95], trades >= 100",
         load=dict(path=UNFILTERED, price_lo=0.05, price_hi=0.95,
                   min_trades=100), run=dict()),
    dict(name="bins5", label="HistogramBinning(5), primary filters",
         load=dict(), run=dict(n_bins=5, methods=["binning10"],
                               want_conformal=False,
                               method_names={"binning10": "binning5"})),
    dict(name="bins20", label="HistogramBinning(20), primary filters",
         load=dict(), run=dict(n_bins=20, methods=["binning10"],
                               want_conformal=False,
                               method_names={"binning10": "binning20"})),
]


def _wavg(g):
    w = g["n_test"].to_numpy(dtype=float)
    return pd.Series({"value": float(np.average(g["value"], weights=w)),
                      "n_test": int(w.sum()), "n_quarters": int(len(g))})


def run_variant(v):
    """Walk-forward under one variant. Returns its long rows."""
    sink, bins_out = wf.RowSink(), []
    for tau in TAUS:
        wf.run_tau(tau, sink, bins_out, run_kw=dict(want_rowlevel=False,
                                                    **v["run"]),
                   **v["load"])
    df = pd.DataFrame(sink.rows)
    df["variant"] = v["name"]
    return df


def aggregate(df, quarters_by_tau=None):
    """n-weighted aggregate, optionally restricted to given quarters."""
    keep = df[df.method.isin(PROB + CONF + ["binning5", "binning20"])
              & df.metric.isin(PROB_METRICS + CONF_METRICS)].copy()
    # walk_forward emits alpha as the string "NA" for probability metrics
    # in memory, which becomes NaN only after a CSV round-trip. Coerce, or
    # every probability row silently vanishes from the sweep.
    keep["alpha"] = pd.to_numeric(keep["alpha"], errors="coerce")
    keep = keep[keep.alpha.isna() | (keep.alpha == ALPHA)]
    if quarters_by_tau is not None:
        keep = keep[[q in quarters_by_tau.get(t, set())
                     for t, q in zip(keep.tau, keep.test_quarter)]]
    if keep.empty:
        return keep
    return (keep.groupby(["variant", "tau", "method", "fit_scope", "domain",
                          "metric"], as_index=False)
                .apply(_wavg, include_groups=False).reset_index(drop=True))


def main():
    primary = pd.read_csv(f"{RESULTS}/walk_forward_long.csv")
    primary = primary[primary.tau.isin(TAUS)].copy()
    primary["variant"] = "primary"
    prim_q = {t: set(g.test_quarter.unique())
              for t, g in primary.groupby("tau")}
    print("primary quarters: " + ", ".join(
        f"{t}={len(q)}" for t, q in prim_q.items()))

    parts, notes = [], []
    for v in VARIANTS:
        raw = run_variant(v)
        vq = {t: set(g.test_quarter.unique()) for t, g in raw.groupby("tau")}
        common = {t: prim_q.get(t, set()) & vq.get(t, set()) for t in TAUS}
        for t in TAUS:
            dropped = sorted(prim_q.get(t, set()) - common[t])
            if dropped:
                notes.append(f"{v['name']} tau={t}: {len(dropped)} primary "
                             f"quarter(s) unavailable ({', '.join(dropped)})")
        agg = aggregate(raw, common)
        agg["variant_label"] = v["label"]
        parts.append(agg)
        print(f"  {v['name']:<14} " + ", ".join(
            f"{t}: {len(common[t])} common quarters" for t in TAUS))
        # The primary side of every delta uses the SAME quarters.
        p = aggregate(primary, common)
        p["variant"] = f"primary@{v['name']}"
        parts.append(p)

    allv = pd.concat(parts, ignore_index=True)
    base = allv[allv.variant.str.startswith("primary@")].copy()
    base["variant"] = base.variant.str.replace("primary@", "", regex=False)
    base = base.rename(columns={"value": "primary_value",
                                "n_test": "primary_n_test"})
    # bins5/bins20 compare against binning10 under the primary filters.
    base_bins = base[base.method == "binning10"].copy()
    key = ["variant", "tau", "fit_scope", "domain", "metric"]
    out = allv[~allv.variant.str.startswith("primary@")].merge(
        base[key + ["method", "primary_value", "primary_n_test"]],
        on=key + ["method"], how="left")
    m = out.method.isin(["binning5", "binning20"])
    out = out.merge(
        base_bins[key + ["primary_value", "primary_n_test"]]
        .rename(columns={"primary_value": "pv_bin",
                         "primary_n_test": "pn_bin"}),
        on=key, how="left")
    out.loc[m, "primary_value"] = out.loc[m, "pv_bin"]
    out.loc[m, "primary_n_test"] = out.loc[m, "pn_bin"]
    out = out.drop(columns=["pv_bin", "pn_bin"])
    out["delta_vs_primary"] = out["value"] - out["primary_value"]

    dorder = {d: i for i, d in enumerate(DOMAINS + ["ALL"])}
    out["_o"] = out.domain.map(dorder)
    out = out.sort_values(["variant", "tau", "metric", "fit_scope", "_o",
                           "method"]).drop(columns="_o")
    cols = ["variant", "variant_label", "method", "domain", "tau",
            "fit_scope", "metric", "value", "primary_value",
            "delta_vs_primary", "n_test", "primary_n_test", "n_quarters"]
    out[cols].to_csv(f"{RESULTS}/table_robustness.csv", index=False)
    print(f"\nSaved table_robustness.csv ({len(out):,} rows)")
    for n in notes:
        print("  note: " + n)
    return out, primary


def verdict(out, primary_raw=None):
    """Do the two qualitative conclusions survive every variant?

    H1's conclusion is a SIGN: recalibration does not beat the raw price,
    i.e. delta-Brier vs raw >= 0. H2's is a DIRECTION: Mondrian sits closer
    to nominal than marginal, i.e. its mean |coverage - 0.9| is smaller.
    """
    lines, ok_h1, ok_h2 = [], True, True

    # --- H1: sign of (method - raw) Brier, pooled fit, domain ALL --------
    b = out[(out.metric == "brier") & (out.fit_scope == "pooled")
            & (out.domain == "ALL")]
    if b.empty:
        raise AssertionError(
            "no H1 rows to check — a vacuous 'HOLDS' verdict is worse than "
            "no verdict; the sweep produced nothing to test")
    checked = 0
    for variant, g in b.groupby("variant"):
        raw = g[g.method == "raw"].set_index("tau")["value"]
        for tau in TAUS:
            for _, r in g[(g.tau == tau) & (g.method != "raw")].iterrows():
                if tau not in raw.index:
                    continue
                d = r["value"] - raw[tau]
                checked += 1
                if d < 0:
                    ok_h1 = False
                    lines.append(f"    H1 SIGN FLIP: {variant} tau={tau} "
                                 f"{r['method']} beats raw by {-d:.6f}")
    if checked == 0:
        raise AssertionError("H1 sign check compared nothing")
    # The bin-count variants carry no 'raw' row of their own (they rerun
    # binning alone), so check their sign against the PRIMARY raw Brier.
    if primary_raw is not None:
        pb = out[(out.metric == "brier") & (out.fit_scope == "pooled")
                 & (out.domain == "ALL")
                 & out.method.isin(["binning5", "binning20"])]
        for _, r in pb.iterrows():
            base = primary_raw.get(r["tau"])
            if base is None:
                continue
            d = r["value"] - base
            if d < 0:
                ok_h1 = False
                lines.append(f"    H1 SIGN FLIP: {r['variant']} "
                             f"tau={r['tau']} {r['method']} beats raw by "
                             f"{-d:.6f}")
            else:
                lines.append(f"    H1 {r['variant']:<14} tau={r['tau']:<4} "
                             f"{r['method']} - raw Brier = {d:+.6f}")

    # --- H2: |cov - 0.9| marginal vs mondrian, per variant/tau -----------
    c = out[(out.metric == "coverage") & (out.domain != "ALL")]
    for (variant, tau), g in c.groupby(["variant", "tau"]):
        w = g.pivot_table(index="domain", columns="method", values="value")
        if not {"conformal_marginal", "conformal_mondrian"} <= set(w.columns):
            continue
        mad_m = (w["conformal_marginal"] - 0.9).abs().mean()
        mad_o = (w["conformal_mondrian"] - 0.9).abs().mean()
        flag = "" if mad_o <= mad_m else "   <== DIRECTION FLIP"
        if mad_o > mad_m:
            ok_h2 = False
        lines.append(f"    H2 {variant:<14} tau={tau:<4} "
                     f"MAD marginal {mad_m:.4f} -> mondrian {mad_o:.4f}{flag}")
    return ok_h1, ok_h2, lines


def by_year(primary):
    """Optional per-year stability table (Step 18, 'if time')."""
    d = primary[primary.metric.isin(PROB_METRICS + CONF_METRICS)
                & primary.method.isin(PROB + CONF)].copy()
    d = d[d.alpha.isna() | (d.alpha == ALPHA)]
    d["year"] = d.test_quarter.str[:4]
    d["variant"] = "primary"
    g = (d.groupby(["year", "tau", "method", "fit_scope", "domain",
                    "metric"], as_index=False)
          .apply(_wavg, include_groups=False).reset_index(drop=True))
    g.to_csv(f"{RESULTS}/table_by_year.csv", index=False)
    print(f"Saved table_by_year.csv ({len(g):,} rows)")
    return g


if __name__ == "__main__":
    out, primary = main()
    praw = aggregate(primary.assign(variant="primary"))
    praw = praw[(praw.method == "raw") & (praw.metric == "brier")
                & (praw.fit_scope == "pooled") & (praw.domain == "ALL")]
    ok1, ok2, lines = verdict(out, dict(zip(praw.tau, praw.value)))
    print("\n" + "=" * 68)
    print("ROBUSTNESS VERDICT")
    print("=" * 68)
    for ln in lines:
        print(ln)
    print(f"\n  H1 sign (no method beats raw on Brier): "
          f"{'HOLDS in every variant' if ok1 else 'FLIPS somewhere'}")
    print(f"  H2 direction (mondrian closer to nominal): "
          f"{'HOLDS in every variant' if ok2 else 'FLIPS somewhere'}")
    by_year(primary)
