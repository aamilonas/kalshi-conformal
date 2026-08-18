"""Step 11 tests — every metric checked on toy data with a known answer."""
import numpy as np

from metrics import (avg_set_size, brier, coverage, ece,
                     reliability_bin_table)


def test_1_perfect_forecasts():
    y = np.array([0, 1, 1, 0, 1, 0, 0, 1] * 25)
    p = y.astype(float)
    assert brier(p, y) == 0.0
    from metrics import log_loss
    assert log_loss(p, y) < 1e-5          # only the 1e-6 clip away from 0
    assert ece(p, y) == 0.0


def test_2_constant_half_on_balanced():
    y = np.array([0, 1] * 500)
    p = np.full(1000, 0.5)
    assert abs(brier(p, y) - 0.25) < 1e-12
    assert abs(ece(p, y)) < 1e-12


def test_3_constant_09_on_zeros():
    y = np.zeros(1000, dtype=int)
    p = np.full(1000, 0.9)
    assert abs(brier(p, y) - 0.81) < 1e-12
    assert abs(ece(p, y) - 0.9) < 1e-12


def test_4_prediction_set_metrics():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    n = len(y)

    full = np.ones((n, 2), dtype=bool)            # always {0, 1}
    assert coverage(full, y) == 1.0
    assert avg_set_size(full) == 2.0

    oracle = np.zeros((n, 2), dtype=bool)         # always {y}
    oracle[np.arange(n), y] = True
    assert coverage(oracle, y) == 1.0
    assert avg_set_size(oracle) == 1.0

    wrong = np.zeros((n, 2), dtype=bool)          # always {1 - y}
    wrong[np.arange(n), 1 - y] = True
    assert coverage(wrong, y) == 0.0


def test_5_ece_bins_are_equal_mass():
    rng = np.random.default_rng(42)
    p = rng.uniform(0, 1, 10_000)
    y = (rng.uniform(0, 1, 10_000) < p).astype(int)
    table = reliability_bin_table(p, y, n_bins=10)
    assert len(table) == 10
    assert table["n"].min() >= 990 and table["n"].max() <= 1010


def test_reliability_table_matches_ece():
    rng = np.random.default_rng(7)
    p = rng.uniform(0.05, 0.95, 5000)
    y = (rng.uniform(0, 1, 5000) < p ** 2).astype(int)   # miscalibrated
    t = reliability_bin_table(p, y, n_bins=10)
    ece_from_table = float(
        (t["n"] / t["n"].sum() * (t["mean_y"] - t["mean_p"]).abs()).sum())
    assert abs(ece_from_table - ece(p, y, n_bins=10)) < 1e-12
    assert ece(p, y) > 0.05                               # detects miscalibration
