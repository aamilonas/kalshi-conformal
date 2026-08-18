"""Step 16 — H1/H2/H3 tables and figures.

Aggregates `results/walk_forward_long.csv` (and, for the pooled reliability
diagrams only, `data/derived/wf_predictions.parquet` — the long file holds
metrics, not per-row predictions, and bins cannot be merged across quarters
after the fact because each quarter's equal-mass edges differ).

Aggregation is n_test-weighted across test quarters throughout. Per the
gate-1 ruling every per-domain row carries `n_test` and `n_quarters`, and
tau=1mo is excluded from H3 (two test quarters is not a time series).

Colours are the validated categorical slots 1-4 (adjacent-pair CVD dE 9.1
light, normal-vision 22.9; slots 1-3 clear the all-pairs gates). Slots 3
and 4 sit below 3:1 against the surface, so the relief rule applies: every
figure ships beside its CSV table and the line panels carry distinct
markers, i.e. identity is never colour alone.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from metrics import reliability_diagram
from paths import DERIVED, RESULTS

DOMAINS = ["Sports", "Crypto", "Politics", "Finance", "Weather",
           "Entertainment"]
NOMINAL = 0.9
ALPHA_PRIMARY = 0.1
ALPHAS = [0.1, 0.05, 0.2]
MIN_DOMAIN_TEST = 200
H3_TAUS = ["24h", "6h", "1w"]          # 1mo excluded: 2 quarters
PROB_METHODS = ["raw", "platt", "isotonic", "binning10", "venn_abers"]
H3_METHODS = {"static_once": "Static-once (frozen)",
              "conformal_mondrian": "Rolling refit",
              "aci_mondrian_g0.005": "ACI (gamma=0.005)"}

S1, S2, S3, S4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
H1_COLORS = {"platt": S1, "isotonic": S2, "binning10": S3, "venn_abers": S4}
H2_COLORS = {"conformal_marginal": S1, "conformal_mondrian": S2}
H3_STYLE = {"static_once": (S1, "o", "-"),
            "conformal_mondrian": (S2, "s", "--"),
            "aci_mondrian_g0.005": (S3, "^", "-.")}
PRETTY = {"raw": "Raw price", "platt": "Platt", "isotonic": "Isotonic",
          "binning10": "Histogram binning (10)", "venn_abers": "Venn-Abers",
          "conformal_marginal": "Conformal, marginal",
          "conformal_mondrian": "Conformal, Mondrian"}

CAPTIONS = []


def _style():
    plt.rcParams.update({
        "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9.5,
        "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.edgecolor": GRID, "axes.labelcolor": INK,
        "text.color": INK, "xtick.color": INK2, "ytick.color": INK2,
        "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "lines.linewidth": 2.0, "savefig.bbox": "tight"})


def _wavg(g):
    """n_test-weighted mean, plus the counts the ruling requires."""
    w = g["n_test"].to_numpy(dtype=float)
    v = g["value"].to_numpy(dtype=float)
    return pd.Series({"value": float(np.average(v, weights=w)) if w.sum()
                      else np.nan,
                      "n_test": int(w.sum()),
                      "n_quarters": int(len(g))})


def _qorder(qs):
    return sorted(set(qs), key=lambda q: pd.Period(q))


def load_long():
    L = pd.read_csv(f"{RESULTS}/walk_forward_long.csv")
    return L[L.method != "na"].copy()


# ----------------------------------------------------------------- H1
def build_H1(L):
    m = L[L.metric.isin(["brier", "log_loss", "ece"])
          & L.method.isin(PROB_METHODS)]
    agg = (m.groupby(["tau", "domain", "fit_scope", "method", "metric"],
                     as_index=False)
             .apply(_wavg, include_groups=False)
             .reset_index(drop=True))
    base = (agg[agg.method == "raw"]
            .rename(columns={"value": "raw_value"})
            [["tau", "domain", "fit_scope", "metric", "raw_value"]])
    agg = agg.merge(base, on=["tau", "domain", "fit_scope", "metric"],
                    how="left")
    agg["delta_vs_raw"] = agg["value"] - agg["raw_value"]
    order = {m: i for i, m in enumerate(PROB_METHODS)}
    agg["_o"] = agg.method.map(order)
    agg = agg.sort_values(["tau", "fit_scope", "domain", "metric", "_o"]) \
             .drop(columns="_o")
    return agg[["tau", "domain", "fit_scope", "method", "metric", "value",
                "raw_value", "delta_vs_raw", "n_test", "n_quarters"]]


def fig_H1(H1):
    _style()
    scope, taus = "pooled", ["24h", "1w"]
    d = H1[(H1.metric == "brier") & (H1.fit_scope == scope)
           & (H1.method != "raw")]
    doms = DOMAINS + ["ALL"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    meths = [m for m in PROB_METHODS if m != "raw"]
    width = 0.8 / len(meths)
    for ax, tau in zip(axes, taus):
        sub = d[d.tau == tau]
        x = np.arange(len(doms))
        for k, meth in enumerate(meths):
            vals, ns = [], []
            for dom in doms:
                r = sub[(sub.domain == dom) & (sub.method == meth)]
                vals.append(float(r.delta_vs_raw.iloc[0]) if len(r) else np.nan)
                ns.append(int(r.n_test.iloc[0]) if len(r) else 0)
            ax.bar(x + (k - (len(meths) - 1) / 2) * width, vals, width * 0.92,
                   label=PRETTY[meth], color=H1_COLORS[meth],
                   edgecolor="white", linewidth=0.8)
        ax.axhline(0, color=INK, linewidth=1.0)
        ax.set_xticks(x)
        n_by_dom = [int(sub[sub.domain == dom].n_test.max() or 0)
                    for dom in doms]
        q_by_dom = [int(sub[sub.domain == dom].n_quarters.max() or 0)
                    for dom in doms]
        ax.set_xticklabels([f"{dom}\nn={n:,}\nq={q}" for dom, n, q
                            in zip(doms, n_by_dom, q_by_dom)], fontsize=7)
        ax.set_title(f"tau = {tau}", loc="left")
        ax.grid(axis="y", alpha=0.7)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Brier score change vs raw price\n"
                       "(negative = recalibration helps)")
    axes[1].legend(frameon=False, loc="best")
    fig.suptitle("H1 — out-of-sample Brier change vs the raw Kalshi price "
                 "(pooled fit, n-weighted over test quarters)",
                 x=0.02, ha="left", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(f"{RESULTS}/fig_H1_brier_delta.png", dpi=220)
    fig.savefig(f"{RESULTS}/fig_H1_brier_delta.pdf")
    plt.close(fig)
    CAPTIONS.append(
        "**fig_H1_brier_delta** — Change in Brier score relative to the raw "
        "Kalshi price for four recalibrators (Platt, isotonic, 10-bin "
        "histogram binning, Venn-Abers), by domain, at tau = 24h (left) and "
        "tau = 1w (right). Negative bars mean recalibration improves on the "
        "market price. Each bar is the n_test-weighted mean over all test "
        "quarters of the walk-forward, using the pooled (all-domain) fit; "
        "n and q under each domain give the total test markets and the "
        "number of contributing quarters. Sports, Crypto and Politics "
        "contribute 2025 quarters only. Underlying values: table_H1.csv.")


# ----------------------------------------------------------------- H2
def build_H2(L):
    m = L[L.method.isin(["conformal_marginal", "conformal_mondrian"])
          & L.metric.isin(["coverage", "avg_set_size"])]
    agg = (m.groupby(["tau", "domain", "method", "alpha", "metric"],
                     as_index=False)
             .apply(_wavg, include_groups=False)
             .reset_index(drop=True))
    short = {"conformal_marginal": "marginal", "conformal_mondrian": "mondrian"}
    agg["col"] = (agg.metric.map({"coverage": "cov",
                                  "avg_set_size": "size"})
                  + "_" + agg.method.map(short)
                  + "_a" + agg.alpha.astype(str))
    wide = agg.pivot_table(index=["tau", "domain"], columns="col",
                           values="value").reset_index()
    counts = (agg[(agg.metric == "coverage")
                  & (agg.alpha == ALPHA_PRIMARY)
                  & (agg.method == "conformal_mondrian")]
              [["tau", "domain", "n_test", "n_quarters"]])
    wide = wide.merge(counts, on=["tau", "domain"], how="left")
    a = ALPHA_PRIMARY
    wide["gap_cov_a0.1"] = wide[f"cov_mondrian_a{a}"] - wide[f"cov_marginal_a{a}"]
    wide["abs_dev_marginal_a0.1"] = (wide[f"cov_marginal_a{a}"] - NOMINAL).abs()
    wide["abs_dev_mondrian_a0.1"] = (wide[f"cov_mondrian_a{a}"] - NOMINAL).abs()
    front = ["tau", "domain", "n_test", "n_quarters",
             f"cov_marginal_a{a}", f"cov_mondrian_a{a}", "gap_cov_a0.1",
             "abs_dev_marginal_a0.1", "abs_dev_mondrian_a0.1",
             f"size_marginal_a{a}", f"size_mondrian_a{a}"]
    rest = [c for c in wide.columns if c not in front]
    dorder = {d: i for i, d in enumerate(DOMAINS + ["ALL"])}
    wide["_o"] = wide.domain.map(dorder)
    return wide.sort_values(["tau", "_o"]).drop(columns="_o")[front + rest]


def fig_H2(H2):
    _style()
    taus = ["24h", "1w"]
    doms = DOMAINS + ["ALL"]
    a = ALPHA_PRIMARY
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, tau in zip(axes, taus):
        sub = H2[H2.tau == tau].set_index("domain")
        x = np.arange(len(doms))
        for k, (col, meth) in enumerate([(f"cov_marginal_a{a}",
                                          "conformal_marginal"),
                                         (f"cov_mondrian_a{a}",
                                          "conformal_mondrian")]):
            vals = [float(sub.loc[d, col]) if d in sub.index else np.nan
                    for d in doms]
            ax.bar(x + (k - 0.5) * 0.4, vals, 0.37, label=PRETTY[meth],
                   color=H2_COLORS[meth], edgecolor="white", linewidth=0.8)
        ax.axhline(NOMINAL, color=INK, linestyle="--", linewidth=1.2,
                   zorder=5)
        ax.annotate("nominal 0.90", (len(doms) - 0.4, NOMINAL),
                    xytext=(0, 4), textcoords="offset points",
                    ha="right", fontsize=7.5, color=INK)
        ax.set_xticks(x)
        lbl = []
        for d in doms:
            if d in sub.index:
                lbl.append(f"{d}\nn={int(sub.loc[d,'n_test']):,}"
                           f"\nq={int(sub.loc[d,'n_quarters'])}")
            else:
                lbl.append(f"{d}\n—")
        ax.set_xticklabels(lbl, fontsize=7)
        ax.set_title(f"tau = {tau}", loc="left")
        ax.grid(axis="y", alpha=0.7)
        ax.set_axisbelow(True)
        ax.set_ylim(0.80, 1.0)
    axes[0].set_ylabel("Empirical coverage at alpha = 0.1\n(fraction of "
                       "markets whose set contains the outcome)")
    axes[1].legend(frameon=False, loc="lower right")
    fig.suptitle("H2 — coverage at alpha = 0.1 by domain: one pooled "
                 "threshold vs one threshold per domain",
                 x=0.02, ha="left", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(f"{RESULTS}/fig_H2_coverage_by_domain.png", dpi=220)
    fig.savefig(f"{RESULTS}/fig_H2_coverage_by_domain.pdf")
    plt.close(fig)
    CAPTIONS.append(
        "**fig_H2_coverage_by_domain** — Empirical coverage of 90% split-"
        "conformal prediction sets by domain, comparing a single marginal "
        "threshold pooled over all domains against Mondrian thresholds fit "
        "per domain, at tau = 24h (left) and tau = 1w (right). The dashed "
        "line is the nominal 0.90 level. Values are n_test-weighted means "
        "over all walk-forward test quarters; n and q give test markets and "
        "contributing quarters per domain. A marginal threshold meets its "
        "target on average while distributing coverage unevenly across "
        "domains; Mondrian re-allocates it. Underlying values, including "
        "alpha = 0.05 and 0.2 and average set sizes: table_H2.csv.")


# ----------------------------------------------------------------- H3
def build_H3(L):
    m = L[(L.metric == "coverage") & (L.alpha == ALPHA_PRIMARY)
          & L.method.isin(H3_METHODS) & L.tau.isin(H3_TAUS)]
    short = {"static_once": "static_once", "conformal_mondrian": "rolling",
             "aci_mondrian_g0.005": "aci_g0.005"}
    w = m.pivot_table(index=["tau", "domain", "test_quarter"],
                      columns="method", values="value").reset_index()
    n = (m[m.method == "conformal_mondrian"]
         [["tau", "domain", "test_quarter", "n_test"]])
    w = w.merge(n, on=["tau", "domain", "test_quarter"], how="left")
    w = w.rename(columns={k: f"cov_{v}" for k, v in short.items()})
    for v in short.values():
        w[f"absdev_{v}"] = (w[f"cov_{v}"] - NOMINAL).abs()
    # Summary: one number per (tau, domain, method) cell, repeated on each
    # quarter row so the table reads without a second file.
    summ = (w.groupby(["tau", "domain"])
             .agg(**{f"mad_{v}": (f"absdev_{v}", "mean") for v in short.values()},
                  n_quarters=("test_quarter", "nunique"),
                  n_test_total=("n_test", "sum"))
             .reset_index())
    w = w.merge(summ, on=["tau", "domain"], how="left")
    # How much calibration data the FROZEN static-once threshold rests on,
    # and whether that domain had to borrow the pooled threshold.
    prov = (L[(L.method == "static_once")
              & L.metric.isin(["n_cal_frozen", "fellback_to_pooled"])]
            .pivot_table(index=["tau", "domain"], columns="metric",
                         values="value", aggfunc="first").reset_index())
    if len(prov):
        prov["n_cal_frozen"] = prov["n_cal_frozen"].astype(int)
        prov["static_once_borrowed_pooled"] = \
            prov.pop("fellback_to_pooled").astype(bool)
        prov = prov.rename(columns={"n_cal_frozen":
                                    "static_once_n_cal_frozen"})
        w = w.merge(prov, on=["tau", "domain"], how="left")
    w["_q"] = w.test_quarter.map(lambda q: pd.Period(q))
    dorder = {d: i for i, d in enumerate(DOMAINS + ["ALL"])}
    w["_o"] = w.domain.map(dorder)
    w = w.sort_values(["tau", "_o", "_q"]).drop(columns=["_q", "_o"])
    front = ["tau", "domain", "test_quarter", "n_test",
             "cov_static_once", "cov_rolling", "cov_aci_g0.005",
             "absdev_static_once", "absdev_rolling", "absdev_aci_g0.005",
             "mad_static_once", "mad_rolling", "mad_aci_g0.005",
             "n_quarters", "n_test_total",
             "static_once_n_cal_frozen", "static_once_borrowed_pooled"]
    front = [c for c in front if c in w.columns]
    return w[front]


def fig_H3(H3):
    """2x4: six domain panels at tau=24h, plus pooled 24h and pooled 1w.

    All eight share one y-axis, computed over BOTH horizons, so the
    upward drift of a frozen threshold at 24h and its downward drift at
    1w are legible against the same scale.
    """
    _style()
    plotted = []
    panels = [(d, "24h") for d in DOMAINS] + [("ALL", "24h"), ("ALL", "1w")]
    for dom, tau in panels:
        s = H3[(H3.tau == tau) & (H3.domain == dom)]
        if dom != "ALL":
            s = s[s.n_test >= MIN_DOMAIN_TEST]
        plotted.append(((dom, tau), s))
    vals = np.concatenate([s[[f"cov_{k}" for k in
                              ("static_once", "rolling", "aci_g0.005")]]
                           .to_numpy(dtype=float).ravel()
                           for _, s in plotted if len(s)])
    vals = vals[~np.isnan(vals)]
    lo, hi = np.nanmin(vals), np.nanmax(vals)
    pad = max(0.02, 0.08 * (hi - lo))
    ylim = (min(lo - pad, NOMINAL - 0.03), max(hi + pad, NOMINAL + 0.03))

    fig, axes = plt.subplots(2, 4, figsize=(15, 7), sharey=True)
    for ax, ((dom, tau), s) in zip(axes.ravel(), plotted):
        qs = _qorder(H3[H3.tau == tau].test_quarter)
        xi = {q: i for i, q in enumerate(qs)}
        for q in qs:                       # shade the 2024 election quarters
            if q in ("2024Q3", "2024Q4"):
                ax.axvspan(xi[q] - 0.5, xi[q] + 0.5, color=GRID, alpha=0.55,
                           zorder=0, linewidth=0)
        ax.axhline(NOMINAL, color=INK, linestyle="--", linewidth=1.1,
                   zorder=2)
        for meth, label in H3_METHODS.items():
            col = {"static_once": "cov_static_once",
                   "conformal_mondrian": "cov_rolling",
                   "aci_mondrian_g0.005": "cov_aci_g0.005"}[meth]
            c, mk, ls = H3_STYLE[meth]
            if len(s):
                ax.plot([xi[q] for q in s.test_quarter], s[col].to_numpy(),
                        color=c, marker=mk, linestyle=ls, markersize=5,
                        markeredgecolor="white", markeredgewidth=0.8,
                        label=label, zorder=3)
        ax.set_xticks(range(len(qs)))
        ax.set_xticklabels([q.replace("20", "") for q in qs], rotation=90,
                           fontsize=6.5)
        n = int(s.n_test.sum()) if len(s) else 0
        ttl = f"{dom} — tau={tau}" if dom == "ALL" else dom
        ax.set_title(f"{ttl}\nn={n:,}, q={len(s)}", loc="left", fontsize=8.5)
        ax.grid(axis="y", alpha=0.6)
        ax.set_axisbelow(True)
        ax.set_ylim(*ylim)
    for ax in axes[:, 0]:
        ax.set_ylabel("Coverage at alpha = 0.1")
    for ax in axes[1, :]:
        ax.set_xlabel("Test quarter")
    h, l = axes[0, 0].get_legend_handles_labels()
    if not h:
        h, l = axes[1, 3].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, ncol=3, loc="lower center",
               bbox_to_anchor=(0.5, -0.015))
    fig.suptitle("H3 — coverage at alpha = 0.1 by test quarter, under three "
                 "threshold policies (shaded: 2024 election quarters)",
                 x=0.01, ha="left", fontsize=10.5)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(f"{RESULTS}/fig_H3_coverage_time.png", dpi=220)
    fig.savefig(f"{RESULTS}/fig_H3_coverage_time.pdf")
    plt.close(fig)
    CAPTIONS.append(
        "**fig_H3_coverage_time** — Empirical coverage of 90% prediction "
        "sets by test quarter for three threshold policies: static-once "
        "(fit on the first calibration window and never refit), rolling "
        "refit (refit each quarter on the trailing 12 months), and adaptive "
        "conformal inference with gamma = 0.005. Six panels are per domain "
        "at tau = 24h; the last two are pooled over all domains at tau = 24h "
        "and tau = 1w. All panels share one y-axis so the direction of drift "
        "is comparable across horizons. Dashed line is nominal 0.90; shaded "
        "bands are the 2024 election quarters. Domain panels show only "
        f"quarters with at least {MIN_DOMAIN_TEST} test markets, so Sports, "
        "Crypto and Politics appear from 2025 only; n and q report the "
        "plotted markets and quarters. Where a domain had too little data "
        "in the frozen calibration window to resolve its own threshold it "
        "borrows the pooled one; table_H3.csv records this per domain in "
        "static_once_n_cal_frozen and static_once_borrowed_pooled. "
        "tau = 1mo is excluded (two quarters). "
        "Per-quarter values and mean absolute deviation from nominal: "
        "table_H3.csv.")


# ------------------------------------------------- reliability diagrams
def fig_reliability():
    _style()
    P = pd.read_parquet(f"{DERIVED}/wf_predictions.parquet",
                        columns=["tau", "method", "fit_scope", "domain",
                                 "p_hat", "y"])
    P = P[(P.fit_scope == "pooled")
          & P.method.isin(["raw", "platt", "venn_abers"])]
    colors = {"raw": S1, "platt": S2, "venn_abers": S3}
    rows, cols = ["Politics", "Sports"], ["24h", "1w"]
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 8.4))
    bins_out = []
    for i, dom in enumerate(rows):
        for j, tau in enumerate(cols):
            ax = axes[i][j]
            sub = P[(P.domain == dom) & (P.tau == tau)]
            n = 0
            for meth in ["raw", "platt", "venn_abers"]:
                s = sub[sub.method == meth]
                if len(s) < 20:
                    continue
                n = len(s)
                _, bt = reliability_diagram(s.p_hat.to_numpy(),
                                            s.y.to_numpy(), ax=ax,
                                            label=PRETTY[meth])
                ax.lines[-1].set_color(colors[meth])
                ax.lines[-1].set_marker({"raw": "o", "platt": "s",
                                         "venn_abers": "^"}[meth])
                ax.lines[-1].set_markersize(5)
                bt.insert(0, "method", meth)
                bt.insert(0, "tau", tau)
                bt.insert(0, "domain", dom)
                bins_out.append(bt)
            ax.set_title(f"{dom} — tau = {tau}   (n = {n:,})", loc="left")
            ax.set_xlabel("Mean forecast probability in bin")
            ax.set_ylabel("Observed outcome frequency in bin")
            ax.grid(alpha=0.6)
            ax.set_axisbelow(True)
            # reliability_diagram() builds a legend on every call, so the
            # entry for a series is captured before we restyle its marker.
            # Drop those and rebuild one legend from the final artists.
            leg = ax.get_legend()
            if leg is not None:
                leg.remove()
            if i == 0 and j == 0:
                ax.legend(frameon=False, loc="upper left")
    fig.suptitle("Reliability, pooled over all walk-forward test quarters "
                 "(equal-mass bins)", x=0.02, ha="left", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(f"{RESULTS}/fig_reliability_wf.png", dpi=220)
    fig.savefig(f"{RESULTS}/fig_reliability_wf.pdf")
    plt.close(fig)
    pd.concat(bins_out, ignore_index=True).to_csv(
        f"{RESULTS}/table_reliability_bins_wf.csv", index=False)
    CAPTIONS.append(
        "**fig_reliability_wf** — Reliability of the raw Kalshi price "
        "against Platt scaling and Venn-Abers, for Politics (top) and "
        "Sports (bottom) at tau = 24h (left) and tau = 1w (right). Points "
        "are equal-mass decile bins of the out-of-sample walk-forward "
        "predictions pooled over every test quarter, using the pooled fit; "
        "the diagonal is perfect calibration, points below it mean the "
        "forecast was too high. n is the number of test markets in the "
        "panel. Bin values: table_reliability_bins_wf.csv.")


def main():
    L = load_long()
    H1 = build_H1(L)
    H1.to_csv(f"{RESULTS}/table_H1.csv", index=False)
    print(f"Saved table_H1.csv ({len(H1):,} rows)")
    H2 = build_H2(L)
    H2.to_csv(f"{RESULTS}/table_H2.csv", index=False)
    print(f"Saved table_H2.csv ({len(H2):,} rows)")
    H3 = build_H3(L)
    H3.to_csv(f"{RESULTS}/table_H3.csv", index=False)
    print(f"Saved table_H3.csv ({len(H3):,} rows)")

    fig_H1(H1); fig_H2(H2); fig_H3(H3); fig_reliability()
    print("Saved fig_H1_brier_delta, fig_H2_coverage_by_domain, "
          "fig_H3_coverage_time, fig_reliability_wf (.png/.pdf)")

    with open(f"{RESULTS}/captions.md", "w", encoding="utf-8") as f:
        f.write("# Figure captions (Step 16)\n\nEach caption stands alone: "
                "it names the estimand, the fold, the weighting and the "
                "counts, so a reader never has to reconstruct them from the "
                "text.\n\n")
        f.write("\n\n".join(CAPTIONS) + "\n")
    print("Saved captions.md")
    return H1, H2, H3


if __name__ == "__main__":
    main()
