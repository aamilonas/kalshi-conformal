"""Step 12 — reproduce Le's logistic recalibration slopes.

Le (2026) fits logit P(y=1) = a + b*logit(p) per cell, contract-weighted,
with LogisticRegression(C=10, penalty='l2', solver='lbfgs') on prices in
cents clipped to [0.01, 0.99] after /100 (src/calibration.py::fit_logistic).

Le's published supplementary artifact (calibration_matrix_216.csv) is per
(domain, time_bin, size_bin) — their domain-x-time slopes CSV was not
published. The reproduction is therefore checked at the published
granularity: our refit per 216-cell vs their slope_b, cell for cell.
The named PASS conditions (Politics ~1.8 at 1w-1mo, etc.) are stated at
(domain x time_bin), so we also compute pooled contract-weighted slopes per
(domain, time_bin) — Le's own fit_slopes_by_domain_time estimator — on the
extraction whose trade counts match their matrix exactly.

Inputs:
  data/derived/le_time_bins.parquet      (built by src/le_time_bins.py)
  data/derived/forecasts.parquet         (our frozen five-horizon grid)
  <pm>/prediction-market-calibration/supplementary/calibration_matrix_216.csv

Outputs:
  results/table_reproduction.csv             216 cells: ours vs Le, + size_bin
  results/fig_reproduction.png/.pdf          scatter, y=x, colored by domain
  results/table_slopes_domain_time_9bin.csv  pooled domain x 9-bin slopes
  results/table_slopes_ours_5h.csv           our domain x tau slope grid
"""
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegression

from le_time_bins import BIN_LABELS, DOMAINS, SIZE_LABELS

from paths import DERIVED, RESULTS, LE_MATRIX
C_REG = 10.0
CELL_MIN = 200
COLORS = {"Politics": "#D62728", "Sports": "#1F77B4", "Weather": "#2CA02C",
          "Crypto": "#FF7F0E", "Finance": "#9467BD", "Entertainment": "#7F7F7F"}
TAUS = ["1h", "6h", "24h", "1w", "1mo"]


def fit_logistic_cents(prices_cents, outcomes, weights):
    """Le's fit_logistic, verbatim semantics. Returns (b, a) or None."""
    X = logit(np.clip(np.asarray(prices_cents, float) / 100.0, 0.01, 0.99))
    y = np.asarray(outcomes).astype(int)
    w = np.asarray(weights, float)
    if len(np.unique(y)) < 2:
        return None
    clf = LogisticRegression(C=C_REG, penalty="l2", solver="lbfgs",
                             max_iter=2000, fit_intercept=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit(X.reshape(-1, 1), y, sample_weight=w)
    return float(clf.coef_[0][0]), float(clf.intercept_[0])


def main():
    cells = pd.read_parquet(f"{DERIVED}/le_time_bins.parquet")
    le = pd.read_csv(LE_MATRIX)

    # ── 1. Cell-level reproduction vs the published 216-cell matrix ──
    rows = []
    for (domain, tbin, sbin), cell in cells.groupby(["domain", "tbin", "sbin"]):
        n_t = int(cell["n_trades"].sum())
        if n_t < CELL_MIN:
            continue
        res = fit_logistic_cents(cell["yes_price"].values,
                                 cell["is_yes"].values,
                                 cell["total_contracts"].values.astype(float))
        if res is None:
            continue
        rows.append(dict(domain=domain, time_bin=BIN_LABELS[int(tbin)],
                         size_bin=SIZE_LABELS[int(sbin)], n_ours=n_t,
                         slope_ours=res[0]))
    ours = pd.DataFrame(rows)

    rep = le.merge(ours, on=["domain", "time_bin", "size_bin"], how="outer",
                   indicator=True)
    n_le_only = int((rep["_merge"] == "left_only").sum())
    n_ours_only = int((rep["_merge"] == "right_only").sum())
    rep = rep[rep["_merge"] == "both"].copy()
    rep["slope_le"] = rep["slope_b"]
    rep["delta"] = rep["slope_ours"] - rep["slope_le"]
    rep["n_match"] = rep["n_ours"] == rep["n_trades"]

    out_cols = ["domain", "time_bin", "size_bin", "n_ours",
                "slope_ours", "slope_le", "delta"]
    rep = rep.sort_values(["domain", "time_bin_order", "size_bin_order"])
    rep[out_cols].to_csv(f"{RESULTS}/table_reproduction.csv", index=False)

    r = float(np.corrcoef(rep["slope_ours"], rep["slope_le"])[0, 1])
    print("Cell-level reproduction vs published calibration_matrix_216.csv:")
    print(f"  cells matched: {len(rep)}/216  "
          f"(le-only: {n_le_only}, ours-only: {n_ours_only})")
    print(f"  per-cell trade counts identical: {int(rep['n_match'].sum())}/{len(rep)}")
    print(f"  Pearson r = {r:.6f}   mean|delta| = {rep['delta'].abs().mean():.2e}"
          f"   max|delta| = {rep['delta'].abs().max():.2e}")
    print(f"  Saved table_reproduction.csv ({len(rep)} rows)")

    # ── 2. Scatter figure ────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(5, 5))
    lims = [min(rep["slope_le"].min(), rep["slope_ours"].min()) - 0.05,
            max(rep["slope_le"].max(), rep["slope_ours"].max()) + 0.05]
    ax.plot(lims, lims, ls="--", lw=1, color="0.6", zorder=1)
    for d in DOMAINS:
        sub = rep[rep["domain"] == d]
        ax.scatter(sub["slope_le"], sub["slope_ours"], s=22, color=COLORS[d],
                   label=d, alpha=0.85, zorder=2)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Le (2026) published slope $b$ (216-cell matrix)")
    ax.set_ylabel("Our refit slope $b$ (same cell definition)")
    ax.set_title(f"Cell-level reproduction of Le's slopes (r = {r:.6f})")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{RESULTS}/fig_reproduction.png", dpi=200)
    fig.savefig(f"{RESULTS}/fig_reproduction.pdf")
    print("  Saved fig_reproduction.png/.pdf")

    # ── 3. Pooled (domain x 9 time bins), Le's Table-3 estimator ─────
    rows = []
    for (domain, tbin), cell in cells.groupby(["domain", "tbin"]):
        n_t = int(cell["n_trades"].sum())
        if n_t < CELL_MIN:
            continue
        res = fit_logistic_cents(cell["yes_price"].values,
                                 cell["is_yes"].values,
                                 cell["total_contracts"].values.astype(float))
        if res is None:
            continue
        rows.append(dict(domain=domain, time_bin=BIN_LABELS[int(tbin)],
                         time_bin_order=int(tbin) + 1, n_trades=n_t,
                         slope_b=res[0], intercept_a=res[1]))
    dt = pd.DataFrame(rows).sort_values(["domain", "time_bin_order"])
    dt.to_csv(f"{RESULTS}/table_slopes_domain_time_9bin.csv", index=False)
    print("\nPooled contract-weighted (domain x time) slopes "
          "[= Le's fit_slopes_by_domain_time on the verified extraction]:")
    print(dt.pivot(index="domain", columns="time_bin", values="slope_b")
          [BIN_LABELS].round(3).to_string())

    # ── 4. Our five-horizon grid from forecasts.parquet (unweighted) ─
    fc = pd.read_parquet(f"{DERIVED}/forecasts.parquet")
    rows = []
    for (domain, tau), g in fc.groupby(["domain", "tau"]):
        res = fit_logistic_cents(g["price"].values * 100.0,
                                 g["outcome"].values, np.ones(len(g)))
        if res is None:
            continue
        rows.append(dict(domain=domain, tau=tau, n=len(g),
                         slope_b=res[0], intercept_a=res[1]))
    g5 = pd.DataFrame(rows)
    g5.to_csv(f"{RESULTS}/table_slopes_ours_5h.csv", index=False)
    print("\nOur (domain x tau) slopes from forecasts.parquet "
          "(unweighted, the paper's grid) — saved table_slopes_ours_5h.csv:")
    print(g5.pivot(index="domain", columns="tau", values="slope_b")
          [TAUS].round(3).to_string())

    # ── 5. PASS conditions ───────────────────────────────────────────
    def slope(domain, tb):
        m = dt[(dt["domain"] == domain) & (dt["time_bin"] == tb)]
        return float(m["slope_b"].iloc[0]) if len(m) else np.nan

    print("\n" + "=" * 72)
    print("PASS CONDITIONS  (1-4 on pooled domain x time slopes)")
    print("=" * 72)

    pol = {tb: slope("Politics", tb) for tb in ["1w-1mo", "1mo+"]}
    c1 = 1.5 <= pol["1w-1mo"] <= 2.1
    print(f"1. Politics ~1.8 at 1w-1mo bins: 1w-1mo={pol['1w-1mo']:.3f}, "
          f"1mo+={pol['1mo+']:.3f}  -> {'PASS' if c1 else 'FAIL'}")

    wx = {tb: slope("Weather", tb) for tb in ["0-1h", "1-3h", "3-6h"]}
    c2 = all(v < 1.0 for v in wx.values() if not np.isnan(v))
    print("2. Weather < 1 at short horizons: "
          + ", ".join(f"{k}={v:.3f}" for k, v in wx.items())
          + f"  -> {'PASS' if c2 else 'FAIL'}")

    sp = {tb: slope("Sports", tb)
          for tb in ["0-1h", "1-3h", "3-6h", "6-12h", "12-24h", "24-48h"]}
    c3 = all(0.85 <= v <= 1.15 for v in sp.values() if not np.isnan(v))
    print("3. Sports ~0.9-1.1 short/medium: "
          + ", ".join(f"{k}={v:.3f}" for k, v in sp.items())
          + f"  -> {'PASS' if c3 else 'FAIL'}")

    dom_ours = rep.groupby("domain").apply(
        lambda g: np.average(g["slope_ours"], weights=g["n_ours"]),
        include_groups=False)
    dom_le = rep.groupby("domain").apply(
        lambda g: np.average(g["slope_le"], weights=g["n_ours"]),
        include_groups=False)
    rho = float(dom_ours.rank().corr(dom_le.rank()))
    c4 = rho >= 0.9
    print(f"4. Same qualitative domain ranking (n-weighted mean cell slope): "
          f"spearman={rho:.3f} "
          f"(ours: {', '.join(dom_ours.sort_values(ascending=False).index)})"
          f"  -> {'PASS' if c4 else 'FAIL'}")

    c5 = r >= 0.95
    print(f"5. Scatter hugs y=x (216 cells): Pearson r = {r:.6f}  -> "
          f"{'PASS' if c5 else 'FAIL'}")

    print("\nOVERALL: " + ("PASS" if all([c1, c2, c3, c4, c5]) else "FAIL"))


if __name__ == "__main__":
    main()
