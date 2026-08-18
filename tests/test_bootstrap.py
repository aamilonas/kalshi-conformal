"""Step 17 tests — does the bootstrap actually cover?

The headline check is calibration of the interval itself: over many
independent datasets with a known true delta, a nominal 95% percentile
interval should contain the truth about 95% of the time.
"""
import numpy as np
import pandas as pd
import pytest

from bootstrap import boot_mad_diff, boot_mean

SEED = 20260818


def test_ci_covers_a_known_delta():
    """200 outer datasets, true mean delta known; 90-99% coverage allowed."""
    rng = np.random.default_rng(SEED)
    true_delta, n, hits = -0.004, 400, 0
    reps = 200
    for _ in range(reps):
        d = rng.normal(true_delta, 0.05, n)
        _, lo, hi, _ = boot_mean(d, rng, n_boot=200)
        hits += lo <= true_delta <= hi
    cov = hits / reps
    assert 0.90 <= cov <= 0.99, f"interval coverage {cov:.3f}"


def test_point_estimate_is_the_sample_mean():
    rng = np.random.default_rng(SEED)
    d = rng.normal(0.01, 0.1, 500)
    pt, lo, hi, med = boot_mean(d, rng, n_boot=200)
    assert pt == pytest.approx(d.mean())
    assert lo < pt < hi
    assert lo < med < hi


def test_interval_brackets_and_widens_with_noise():
    rng = np.random.default_rng(SEED)
    quiet = boot_mean(rng.normal(0, 0.01, 400), rng, n_boot=300)
    loud = boot_mean(rng.normal(0, 0.20, 400), rng, n_boot=300)
    assert (loud[2] - loud[1]) > (quiet[2] - quiet[1])


def test_clustering_matters_row_resampling_would_understate():
    """The reason Step 17 resamples markets and not rows.

    Build data where each market contributes many perfectly correlated
    rows. Resampling markets must give a WIDER interval than resampling
    rows, because the rows carry no independent information.
    """
    rng = np.random.default_rng(SEED + 1)
    n_markets, per_market = 120, 12
    market_effect = rng.normal(0.0, 0.05, n_markets)
    rows = np.repeat(market_effect, per_market)      # identical within market

    _, lo_m, hi_m, _ = boot_mean(market_effect, rng, n_boot=400)  # correct
    _, lo_r, hi_r, _ = boot_mean(rows, rng, n_boot=400)           # wrong
    assert (hi_m - lo_m) > 1.5 * (hi_r - lo_r), \
        f"market CI {hi_m-lo_m:.5f} not wider than row CI {hi_r-lo_r:.5f}"


def test_mad_diff_detects_a_real_gap_and_not_a_fake_one():
    """MAD-from-nominal: one policy on target, one biased away from it."""
    rng = np.random.default_rng(SEED + 2)
    n, n_q = 3000, 6
    qidx = rng.integers(0, n_q, n)
    on_target = (rng.uniform(size=n) < 0.90).astype(float)
    biased = (rng.uniform(size=n) < 0.80).astype(float)

    pt, lo, hi, _ = boot_mad_diff(on_target, biased, qidx, n_q, rng,
                                  n_boot=300)
    assert pt < 0 and hi < 0, f"failed to detect the gap: {pt:.4f} [{lo:.4f},{hi:.4f}]"

    same = (rng.uniform(size=n) < 0.90).astype(float)
    pt2, lo2, hi2, _ = boot_mad_diff(on_target, same, qidx, n_q, rng,
                                     n_boot=300)
    assert lo2 <= 0 <= hi2, f"spurious gap: {pt2:.4f} [{lo2:.4f},{hi2:.4f}]"


def test_degenerate_inputs_do_not_raise():
    rng = np.random.default_rng(SEED)
    for d in (np.array([]), np.array([0.5])):
        out = boot_mean(d, rng, n_boot=50)
        assert len(out) == 4 and all(np.isnan(v) for v in out)


def test_output_table_shape_and_significance_flag():
    from paths import RESULTS
    try:
        T = pd.read_csv(f"{RESULTS}/table_bootstrap_ci.csv")
    except FileNotFoundError:
        pytest.skip("table_bootstrap_ci.csv not built yet")
    need = {"family", "comparison", "metric", "tau", "domain", "fit_scope",
            "alpha", "point_estimate", "ci_lo", "ci_hi", "boot_median",
            "significant", "n_markets", "n_quarters"}
    assert need <= set(T.columns)
    fin = T.dropna(subset=["ci_lo", "ci_hi"])

    # H1 and H2 are means: the bootstrap distribution is centred on the
    # sample value, so the percentile interval must bracket it.
    mean_fam = fin[fin.family.isin(["H1", "H2"])]
    assert (mean_fam.ci_lo <= mean_fam.point_estimate).all()
    assert (mean_fam.point_estimate <= mean_fam.ci_hi).all()

    # H3's MAD-from-nominal is convex in noisy per-quarter coverage, so
    # resampling inflates it (Jensen) and the point estimate may sit just
    # outside. Bound that excursion at 10% of the interval width -- if it
    # grows past that the percentile interval is the wrong tool here.
    mad = fin[fin.family == "H3"]
    width = (mad.ci_hi - mad.ci_lo).to_numpy()
    excursion = np.maximum(mad.ci_lo - mad.point_estimate,
                           mad.point_estimate - mad.ci_hi).to_numpy()
    assert (excursion <= 0.10 * width).all(), \
        f"MAD bootstrap bias too large: {excursion.max():.2e}"
    expect = (fin.ci_lo > 0) | (fin.ci_hi < 0)
    assert (fin.significant.astype(bool) == expect).all()
    assert T.family.iloc[0] == "H1", "H1 must lead the file"
