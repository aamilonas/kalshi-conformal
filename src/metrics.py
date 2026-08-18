"""Step 11 — metrics module.

Vectorized over numpy arrays. Proper scores (Brier, log loss), equal-mass
ECE, prediction-set metrics (coverage, average set size), and reliability
diagrams with their underlying bin tables.

Conventions: p = P(y=1) forecasts in [0,1], y = binary outcomes in {0,1};
`sets` = boolean array of shape (n, 2) where sets[i, k] means label k is in
the prediction set for observation i.
"""
import numpy as np
import pandas as pd


def brier(p, y):
    """Mean squared error of probability forecasts: mean((p - y)^2)."""
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    return float(np.mean((p - y) ** 2))


def log_loss(p, y):
    """Negative mean log likelihood; p clipped to [1e-6, 1 - 1e-6]."""
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, dtype=float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log1p(-p)))


def _equal_mass_bin_index(p, n_bins):
    """Assign each p to one of n_bins equal-mass bins via quantile edges.

    Returns (idx, edges) where idx in [0, n_bins). Points equal to an
    interior edge go to the upper bin; duplicated edges (heavily tied p)
    simply leave some bins empty, which callers must tolerate.
    """
    edges = np.quantile(p, np.linspace(0.0, 1.0, n_bins + 1))
    idx = np.searchsorted(edges[1:-1], p, side="right")
    return idx, edges


def ece(p, y, n_bins=10):
    """Expected calibration error with EQUAL-MASS bins (Guo et al. 2017,
    "On Calibration of Modern Neural Networks"), except bins are equal-mass
    (np.quantile edges) rather than equal-width so no bin is starved:

        ECE = sum_b (n_b / N) * |mean(y in b) - mean(p in b)|
    """
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    idx, _ = _equal_mass_bin_index(p, n_bins)
    total = 0.0
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        total += mask.mean() * abs(y[mask].mean() - p[mask].mean())
    return float(total)


def coverage(sets, y):
    """Fraction of observations whose true label is in the prediction set."""
    sets = np.asarray(sets, dtype=bool)
    y = np.asarray(y, dtype=int)
    return float(sets[np.arange(len(y)), y].mean())


def avg_set_size(sets):
    """Mean number of labels per prediction set."""
    sets = np.asarray(sets, dtype=bool)
    return float(sets.sum(axis=1).mean())


def reliability_bin_table(p, y, n_bins=10):
    """Equal-mass bin table underlying a reliability diagram."""
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    idx, edges = _equal_mass_bin_index(p, n_bins)
    rows = []
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        rows.append(dict(
            bin=b, lo=edges[b], hi=edges[b + 1], n=int(mask.sum()),
            mean_p=float(p[mask].mean()), mean_y=float(y[mask].mean()),
        ))
    return pd.DataFrame(rows)


def reliability_diagram(p, y, n_bins=10, ax=None, label=None):
    """Plot mean predicted vs observed per equal-mass bin, with y=x diagonal.

    Returns (ax, bin_table). The diagonal is drawn only when this call
    creates the axis, so overlaying several methods on one ax stays clean.
    """
    import matplotlib.pyplot as plt

    table = reliability_bin_table(p, y, n_bins)
    if ax is None:
        _, ax = plt.subplots(figsize=(4.2, 4.2))
        ax.plot([0, 1], [0, 1], ls="--", lw=1, color="0.55", zorder=1)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Observed frequency")
    ax.plot(table["mean_p"], table["mean_y"], marker="o", ms=4, lw=1.4,
            label=label, zorder=2)
    if label is not None:
        ax.legend(frameon=False, fontsize=8)
    return ax, table
