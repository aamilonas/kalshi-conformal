"""Step 13 tests — recalibrator correctness, incl. the conformal canary."""
import numpy as np
import pytest

from metrics import brier, coverage
from recalibrators import (Adaptive, HistogramBinning, Isotonic, Platt, Raw,
                           SplitConformal, VennAbers)


def _miscalibrated(n, rng):
    """y ~ Bernoulli(p^2): prices systematically overconfident on the high
    side. Returns (p, y)."""
    p = rng.uniform(0.05, 0.95, n)
    y = (rng.uniform(0, 1, n) < p ** 2).astype(int)
    return p, y


@pytest.fixture(scope="module")
def synth():
    rng = np.random.default_rng(11)
    p_cal, y_cal = _miscalibrated(4000, rng)
    p_test, y_test = _miscalibrated(4000, rng)
    return p_cal, y_cal, p_test, y_test


def test_proba_range_and_shape(synth):
    p_cal, y_cal, p_test, _ = synth
    methods = [Raw(), Platt(), Isotonic(), HistogramBinning(10), VennAbers(),
               SplitConformal(Raw())]
    for m in methods:
        out = m.fit(p_cal, y_cal).predict_proba(p_test)
        assert out.shape == (len(p_test),), type(m).__name__
        assert np.all(out >= 0) and np.all(out <= 1), type(m).__name__


def test_recalibrated_beats_raw_brier(synth):
    p_cal, y_cal, p_test, y_test = synth
    raw = brier(p_test, y_test)
    for m in [Platt(), Isotonic(), HistogramBinning(10)]:
        b = brier(m.fit(p_cal, y_cal).predict_proba(p_test), y_test)
        assert b < raw, f"{type(m).__name__}: {b:.4f} !< raw {raw:.4f}"


def test_histogram_binning_exact_on_cal_fold(synth):
    p_cal, y_cal, _, _ = synth
    hb = HistogramBinning(10).fit(p_cal, y_cal)
    idx = np.searchsorted(hb.edges_[1:-1], p_cal, side="right")
    pred = hb.predict_proba(p_cal)
    for b in range(10):
        mask = idx == b
        assert mask.any()
        assert np.allclose(pred[mask], y_cal[mask].mean())
    t = hb.bin_table()
    assert list(t.columns) == ["bin", "lo", "hi", "n", "successes", "freq",
                               "ci_lo", "ci_hi"]
    assert (t["ci_lo"] <= t["freq"]).all() and (t["freq"] <= t["ci_hi"]).all()
    assert t["n"].sum() == len(p_cal)


def test_venn_abers_interval_brackets_point(synth):
    p_cal, y_cal, p_test, _ = synth
    va = VennAbers().fit(p_cal[:800], y_cal[:800])
    sub = p_test[:200]
    p0, p1 = va.predict_interval(sub)
    point = va.predict_proba(sub)
    assert np.all(p0 <= p1 + 1e-12)
    assert np.all(point >= p0 - 1e-12) and np.all(point <= p1 + 1e-12)


def test_split_conformal_marginal_canary():
    """THE canary: on i.i.d. data, marginal coverage at alpha=.1 must land
    in [0.885, 0.915]. Outside -> the implementation is wrong."""
    rng = np.random.default_rng(3)
    n = 5000
    p_cal = rng.uniform(0.02, 0.98, n)
    y_cal = (rng.uniform(0, 1, n) < p_cal).astype(int)
    p_test = rng.uniform(0.02, 0.98, n)
    y_test = (rng.uniform(0, 1, n) < p_test).astype(int)

    sc = SplitConformal(Raw()).fit(p_cal, y_cal)
    sets = sc.predict_set(p_test, alpha=0.1)
    cov = coverage(sets, y_test)
    assert 0.885 <= cov <= 0.915, f"coverage {cov:.4f} outside canary band"


def test_split_conformal_mondrian_vs_marginal_groups():
    """Two groups of very different difficulty: marginal per-group coverage
    splits apart; mondrian restores ~90% in each group."""
    rng = np.random.default_rng(5)
    n = 3000

    def make(group_hard):
        if group_hard:                      # prices carry no information
            p = rng.uniform(0.05, 0.95, n)
            y = (rng.uniform(0, 1, n) < 0.5).astype(int)
        else:                               # well-calibrated, extreme prices
            p = rng.choice([0.05, 0.95], n)
            y = (rng.uniform(0, 1, n) < p).astype(int)
        return p, y

    pA, yA = make(True)
    pB, yB = make(False)
    pA2, yA2 = make(True)
    pB2, yB2 = make(False)
    p_cal, y_cal = np.concatenate([pA, pB]), np.concatenate([yA, yB])
    g_cal = np.array(["A"] * n + ["B"] * n)
    p_test, y_test = np.concatenate([pA2, pB2]), np.concatenate([yA2, yB2])
    g_test = g_cal

    marg = SplitConformal(Raw()).fit(p_cal, y_cal)
    sets = marg.predict_set(p_test, alpha=0.1)
    covA = coverage(sets[:n], y_test[:n])
    covB = coverage(sets[n:], y_test[n:])
    assert abs(covA - covB) > 0.05          # marginal splits across groups
    assert covA < 0.885                     # hard group under-covered

    mond = SplitConformal(Raw(), mode="mondrian").fit(p_cal, y_cal,
                                                      groups=g_cal)
    msets = mond.predict_set(p_test, alpha=0.1, groups=g_test)
    mcovA = coverage(msets[:n], y_test[:n])
    mcovB = coverage(msets[n:], y_test[n:])
    assert mcovA >= 0.88 and mcovB >= 0.88


def test_adaptive_is_wired_to_aci():
    # Phase 4 implemented this; it is now adaptive.ACI. The interface is
    # deliberately different (ACI consumes outcomes in temporal order), so
    # the check is that it is wired up, not that it quacks like the others.
    from adaptive import ACI
    assert Adaptive is ACI
    rng = np.random.default_rng(0)
    p = rng.uniform(0.05, 0.95, 500)
    y = (rng.uniform(size=500) < p).astype(int)
    out = Adaptive(alpha=0.1, gamma=0.01).fit(p, y).run(
        p, y, order=np.arange(500))
    assert out["sets"].shape == (500, 2)
