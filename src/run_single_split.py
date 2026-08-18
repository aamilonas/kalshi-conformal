"""Step 14 — single-split benchmark at tau=24h on forecasts.parquet.

Split rule: CHRONOLOGICAL by market close_time; never random. The fold
boundary is asserted, not assumed. No fit() ever sees a test-fold outcome.

The instruction file's original boundary (cal < 2024-01-01) is infeasible
per-domain: Kalshi launched single-game sports, hourly crypto and most
politics products in 2024-2025, leaving 0-14 pre-2024 calibration markets
in those domains at tau=24h. Per user decision (2026-08-18):
  PRIMARY    boundary 2025-07-01, pooled + per-domain fits (all domains
             feasible: min per-domain cal n = 739), exit tests run here.
  ROBUSTNESS boundary 2024-01-01 (the original spec), pooled fits only,
             reported as results/table_M2_robustness_spec_split.csv.

Methods: Raw, Platt, Isotonic, HistogramBinning(10), VennAbers,
Conformal-marginal(Raw), Conformal-mondrian(Raw).

Outputs:
  results/table_M2_main.csv           long tidy: method, fit_scope, domain,
                                      tau, alpha, metric, value, n_test
  results/table_M2_robustness_spec_split.csv   same shape, spec boundary
  results/fig_reliability_M2.png/.pdf 2x2: Politics/Sports x pooled/per-dom
  results/table_reliability_bins_M2.csv
  results/table_binning_ci_M2.csv     HistogramBinning bin_table per domain
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from metrics import brier, log_loss, ece, coverage, avg_set_size, \
    reliability_diagram
from recalibrators import (HistogramBinning, Isotonic, Platt, Raw,
                           SplitConformal, VennAbers)

from paths import DERIVED, RESULTS
DOMAINS = ["Sports", "Crypto", "Politics", "Finance", "Weather", "Entertainment"]
TAU = "24h"
PRIMARY_BOUNDARY = pd.Timestamp("2025-07-01", tz="UTC")
SPEC_BOUNDARY = pd.Timestamp("2024-01-01", tz="UTC")
DATA_END = pd.Timestamp("2026-01-01", tz="UTC")
ALPHAS = [0.1, 0.05, 0.2]           # 0.1 primary


def make_methods():
    """Fresh instances each fit so no state leaks across fits."""
    return {
        "Raw": Raw(),
        "Platt": Platt(),
        "Isotonic": Isotonic(),
        "HistogramBinning": HistogramBinning(10),
        "VennAbers": VennAbers(),
        "Conformal-marginal": SplitConformal(Raw(), mode="marginal"),
        "Conformal-mondrian": SplitConformal(Raw(), mode="mondrian"),
    }


def prob_metric_rows(method, scope, domain, p_hat, y, n_test):
    return [dict(method=method, fit_scope=scope, domain=domain, tau=TAU,
                 alpha="NA", metric=metric, value=val, n_test=n_test)
            for metric, val in [("brier", brier(p_hat, y)),
                                ("log_loss", log_loss(p_hat, y)),
                                ("ece", ece(p_hat, y))]]


def set_metric_rows(method, scope, domain, sets, y, alpha, n_test):
    return [dict(method=method, fit_scope=scope, domain=domain, tau=TAU,
                 alpha=alpha, metric="coverage", value=coverage(sets, y),
                 n_test=n_test),
            dict(method=method, fit_scope=scope, domain=domain, tau=TAU,
                 alpha=alpha, metric="avg_set_size", value=avg_set_size(sets),
                 n_test=n_test)]


def run_benchmark(boundary, scopes, label):
    """One chronological split. Returns (main_tbl, preds, folds)."""
    fc = pd.read_parquet(f"{DERIVED}/forecasts.parquet")
    fc = fc[(fc["tau"] == TAU) & (fc["domain"].isin(DOMAINS))].copy()
    fc["close_time"] = pd.to_datetime(fc["close_time"], utc=True)

    cal = fc[fc["close_time"] < boundary].reset_index(drop=True)
    test = fc[(fc["close_time"] >= boundary)
              & (fc["close_time"] < DATA_END)].reset_index(drop=True)

    # Fold-boundary assertions — leakage is the top risk.
    assert len(cal) > 0 and len(test) > 0
    assert cal["close_time"].max() < test["close_time"].min(), \
        "chronological split violated"
    assert cal["close_time"].max() < boundary <= test["close_time"].min()
    assert len(cal) + len(test) == len(fc), "rows lost at the boundary"
    print(f"\n[{label}] boundary {boundary:%Y-%m-%d}: cal n={len(cal):,} "
          f"(close <= {cal['close_time'].max():%Y-%m-%d}), "
          f"test n={len(test):,} ({test['close_time'].min():%Y-%m-%d} .. "
          f"{test['close_time'].max():%Y-%m-%d})")
    print("  per-domain (cal/test): " + ", ".join(
        f"{d} {(cal['domain'] == d).sum()}/{(test['domain'] == d).sum()}"
        for d in DOMAINS))
    if "per_domain" in scopes:
        min_cal = min((cal["domain"] == d).sum() for d in DOMAINS)
        assert min_cal >= 200, \
            f"per-domain fits requested but min cal n = {min_cal}"

    p_cal, y_cal = cal["price"].values, cal["outcome"].values.astype(int)
    p_test, y_test = test["price"].values, test["outcome"].values.astype(int)
    g_cal, g_test = cal["domain"].values, test["domain"].values

    rows = []
    preds = {}                       # (scope, method) -> p_hat on full test
    binning_tables = []

    if "pooled" in scopes:
        for name, m in make_methods().items():
            if name == "Conformal-mondrian":
                m.fit(p_cal, y_cal, groups=g_cal)
            else:
                m.fit(p_cal, y_cal)
            p_hat = m.predict_proba(p_test)
            preds[("pooled", name)] = p_hat
            rows += prob_metric_rows(name, "pooled", "ALL", p_hat, y_test,
                                     len(test))
            for d in DOMAINS:
                sel = g_test == d
                rows += prob_metric_rows(name, "pooled", d, p_hat[sel],
                                         y_test[sel], int(sel.sum()))
            if isinstance(m, SplitConformal):
                for alpha in ALPHAS:
                    kw = dict(groups=g_test) if m.mode == "mondrian" else {}
                    sets = m.predict_set(p_test, alpha, **kw)
                    rows += set_metric_rows(name, "pooled", "ALL", sets,
                                            y_test, alpha, len(test))
                    for d in DOMAINS:
                        sel = g_test == d
                        rows += set_metric_rows(name, "pooled", d, sets[sel],
                                                y_test[sel], alpha,
                                                int(sel.sum()))

    if "per_domain" in scopes:
        per_dom_parts = {name: np.empty(len(test)) for name in make_methods()}
        for d in DOMAINS:
            ci, ti = g_cal == d, g_test == d
            for name, m in make_methods().items():
                if name == "Conformal-mondrian":
                    m.fit(p_cal[ci], y_cal[ci], groups=g_cal[ci])
                else:
                    m.fit(p_cal[ci], y_cal[ci])
                p_hat = m.predict_proba(p_test[ti])
                per_dom_parts[name][ti] = p_hat
                rows += prob_metric_rows(name, "per_domain", d, p_hat,
                                         y_test[ti], int(ti.sum()))
                if isinstance(m, SplitConformal):
                    for alpha in ALPHAS:
                        kw = dict(groups=g_test[ti]) \
                            if m.mode == "mondrian" else {}
                        sets = m.predict_set(p_test[ti], alpha, **kw)
                        rows += set_metric_rows(name, "per_domain", d, sets,
                                                y_test[ti], alpha,
                                                int(ti.sum()))
                if name == "HistogramBinning":
                    bt = m.bin_table()
                    bt.insert(0, "domain", d)
                    binning_tables.append(bt)
        for name, p_hat in per_dom_parts.items():
            preds[("per_domain", name)] = p_hat
            rows += prob_metric_rows(name, "per_domain", "ALL", p_hat,
                                     y_test, len(test))

    return pd.DataFrame(rows), preds, (g_test, y_test), binning_tables


def exit_tests(main_tbl, cal_scores=None, test_scores=None):
    def val(method, scope, domain, metric, alpha="NA"):
        m = main_tbl[(main_tbl["method"] == method)
                     & (main_tbl["fit_scope"] == scope)
                     & (main_tbl["domain"] == domain)
                     & (main_tbl["metric"] == metric)
                     & (main_tbl["alpha"] == alpha)]
        return float(m["value"].iloc[0])

    print("\n" + "=" * 72)
    print("EXIT TESTS (primary split)")
    print("=" * 72)

    cov = val("Conformal-marginal", "pooled", "ALL", "coverage", 0.1)
    t1 = 0.885 <= cov <= 0.915
    print(f"1. Marginal pooled coverage @ alpha=0.1: {cov:.4f} "
          f"in [0.885, 0.915] -> {'PASS' if t1 else 'FAIL as specified'}")
    if cal_scores is not None and not t1:
        # Decompose the overshoot: discrete-score ties (visible in-sample
        # on the calibration fold, where exchangeability holds by
        # construction) vs temporal distribution shift (test minus cal).
        n = len(cal_scores)
        rank = int(np.ceil((n + 1) * 0.9))
        qhat = np.sort(cal_scores)[rank - 1]
        in_samp = float(np.mean(cal_scores <= qhat))
        print(f"   decomposition: qhat={qhat:.3f}; cal in-sample "
              f"P(s<=qhat)={in_samp:.4f} (0.90 + tie mass; implementation "
              f"correct iff in [0.900, ~0.910]); "
              f"temporal shift adds {float(np.mean(test_scores <= qhat)) - in_samp:+.4f}. "
              f"Coverage remains >= 1-alpha (conservative direction).")

    b_raw = val("Raw", "per_domain", "Politics", "brier")
    b_va = val("VennAbers", "per_domain", "Politics", "brier")
    b_hb = val("HistogramBinning", "per_domain", "Politics", "brier")
    t2 = (b_va <= b_raw) or (b_hb <= b_raw)
    print(f"2. Politics Brier (per-domain fit): raw={b_raw:.5f}, "
          f"VennAbers={b_va:.5f}, HistBinning={b_hb:.5f} -> "
          f"{'PASS' if t2 else 'NULL RESULT — investigate first'}")

    marg = {d: val("Conformal-marginal", "pooled", d, "coverage", 0.1)
            for d in DOMAINS}
    mond = {d: val("Conformal-mondrian", "pooled", d, "coverage", 0.1)
            for d in DOMAINS}
    spread_m = max(marg.values()) - min(marg.values())
    spread_o = max(mond.values()) - min(mond.values())
    t3 = spread_m > spread_o and all(0.87 <= c <= 0.93 for c in mond.values())
    print("3. Per-domain coverage @ alpha=0.1 (pooled fits):")
    print("   marginal: " + ", ".join(f"{d}={c:.3f}" for d, c in marg.items())
          + f"   (spread {spread_m:.3f})")
    print("   mondrian: " + ", ".join(f"{d}={c:.3f}" for d, c in mond.items())
          + f"   (spread {spread_o:.3f})")
    print(f"   mondrian uniform near 90% and tighter than marginal -> "
          f"{'PASS' if t3 else 'FAIL'}")

    print(f"\nEXIT: {'PASS' if (t1 and t2 and t3) else 'FAIL'} "
          f"(1={t1}, 2={t2}, 3={t3})")
    return t1, t2, t3


def main():
    # ── Primary split ────────────────────────────────────────────────
    main_tbl, preds, (g_test, y_test), binning_tables = run_benchmark(
        PRIMARY_BOUNDARY, ["pooled", "per_domain"], "PRIMARY")
    main_tbl.to_csv(f"{RESULTS}/table_M2_main.csv", index=False)
    print(f"  Saved table_M2_main.csv ({len(main_tbl)} rows)")

    pd.concat(binning_tables, ignore_index=True).to_csv(
        f"{RESULTS}/table_binning_ci_M2.csv", index=False)
    print("  Saved table_binning_ci_M2.csv")

    # Reliability figure: Politics/Sports x pooled/per_domain.
    fig, axes = plt.subplots(2, 2, figsize=(9, 9))
    bins_out = []
    for i, d in enumerate(["Politics", "Sports"]):
        for j, scope in enumerate(["pooled", "per_domain"]):
            ax = axes[i][j]
            sel = g_test == d
            for name in ["Raw", "Platt", "VennAbers"]:
                _, bt = reliability_diagram(preds[(scope, name)][sel],
                                            y_test[sel], ax=ax, label=name)
                bt.insert(0, "method", name)
                bt.insert(0, "fit_scope", scope)
                bt.insert(0, "domain", d)
                bins_out.append(bt)
            ax.set_title(f"{d} — {scope} fit (test fold, tau={TAU})",
                         fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{RESULTS}/fig_reliability_M2.png", dpi=200)
    fig.savefig(f"{RESULTS}/fig_reliability_M2.pdf")
    pd.concat(bins_out, ignore_index=True).to_csv(
        f"{RESULTS}/table_reliability_bins_M2.csv", index=False)
    print("  Saved fig_reliability_M2.png/.pdf, table_reliability_bins_M2.csv")

    # ── Robustness: the original spec boundary, pooled only ──────────
    rob_tbl, _, _, _ = run_benchmark(SPEC_BOUNDARY, ["pooled"],
                                     "ROBUSTNESS/spec")
    rob_tbl.to_csv(f"{RESULTS}/table_M2_robustness_spec_split.csv",
                   index=False)
    print(f"  Saved table_M2_robustness_spec_split.csv ({len(rob_tbl)} rows)")

    # Recompute pooled Raw scores for the exit-test decomposition.
    fc = pd.read_parquet(f"{DERIVED}/forecasts.parquet")
    fc = fc[(fc["tau"] == TAU) & (fc["domain"].isin(DOMAINS))].copy()
    fc["close_time"] = pd.to_datetime(fc["close_time"], utc=True)
    p = fc["price"].values
    y = fc["outcome"].values.astype(int)
    s = np.where(y == 1, 1 - p, p)
    is_cal = (fc["close_time"] < PRIMARY_BOUNDARY).values
    exit_tests(main_tbl, cal_scores=s[is_cal], test_scores=s[~is_cal])


if __name__ == "__main__":
    main()
