"""Step 15c tests — ACI and the static-once comparator.

Two claims, both from Gibbs & Candes (2021):

1. Under exchangeability ACI's long-run coverage sits at the target level.
2. Under a sharp distribution shift a frozen threshold loses coverage while
   ACI walks its level back to target.

Synthetic design mirrors the real score: forecasts ``p`` are drawn on the
Kalshi-like support [0.05, 0.95] and labels are ``Bernoulli(p)``, so the
forecaster is calibrated by construction. The "shift" flips a fraction of
labels, which is exactly the failure the paper cares about — prices that
stop tracking outcomes.
"""
import numpy as np
import pytest

from adaptive import (ACI, ALPHA_CLIP, StaticOnce, _scores_from_labels
                      as _scores, conformal_qhat)

SEED = 20260818


def _synth(n, rng, flip=0.0):
    p = rng.uniform(0.05, 0.95, n)
    y = (rng.uniform(size=n) < p).astype(int)
    if flip > 0:
        m = rng.uniform(size=n) < flip
        y[m] = 1 - y[m]
    return p, y


def _cov(sets, y):
    return float(sets[np.arange(len(y)), y].mean())


def test_qhat_matches_split_conformal():
    """conformal_qhat must agree with the Phase 3 implementation."""
    from recalibrators import SplitConformal
    rng = np.random.default_rng(SEED)
    s = np.sort(rng.uniform(size=500))
    for a in (0.05, 0.1, 0.2, 0.5):
        assert conformal_qhat(s, a) == SplitConformal._qhat(s, a)


def test_qhat_degenerate_levels():
    """The empty-set branch is unreachable under the spec's alpha clip.

    rank = ceil((n+1)(1-alpha)); at the clip ceiling alpha=0.999 this is
    still >= 1 for every n, so ACI can narrow a set to a single label but
    can never emit an empty one. -inf needs alpha == 1 exactly.
    """
    s = np.sort(np.random.default_rng(SEED).uniform(size=50))
    assert conformal_qhat(s, 0.001) == np.inf      # rank > n -> cover both
    assert conformal_qhat(s, 0.999) == s[0]        # rank 1 -> tightest
    assert conformal_qhat(s, 1.0) == -np.inf       # rank < 1 -> empty
    assert conformal_qhat(np.array([]), 0.1) == np.inf
    for n in (1, 10, 1000, 100_000):
        assert int(np.ceil((n + 1) * (1 - ALPHA_CLIP[1]))) >= 1


def test_aci_iid_coverage_converges_to_target():
    rng = np.random.default_rng(SEED)
    p_cal, y_cal = _synth(5000, rng)
    p_te, y_te = _synth(20000, rng)
    out = (ACI(alpha=0.1, gamma=0.005, mode="pooled")
           .fit(p_cal, y_cal)
           .run(p_te, y_te, order=np.arange(len(p_te))))
    cov = float(out["covered"].mean())
    assert abs(cov - 0.9) <= 0.015, f"ACI i.i.d. coverage {cov:.4f}"


def test_aci_recovers_after_sharp_shift():
    rng = np.random.default_rng(SEED + 1)
    n_cal, n_half = 5000, 4000
    p_cal, y_cal = _synth(n_cal, rng)
    p_a, y_a = _synth(n_half, rng)
    p_b, y_b = _synth(n_half, rng, flip=0.35)
    p_te = np.concatenate([p_a, p_b])
    y_te = np.concatenate([y_a, y_b])
    g0 = np.zeros(n_cal, dtype=int)
    g = np.zeros(len(p_te), dtype=int)

    static = StaticOnce(alpha=0.1).fit(p_cal, y_cal, groups=g0)
    s_sets = static.predict_set(p_te, groups=g)
    cov_pre = _cov(s_sets[:n_half], y_te[:n_half])
    cov_post = _cov(s_sets[n_half:], y_te[n_half:])
    assert cov_post < cov_pre - 0.05, \
        f"static did not decay: pre={cov_pre:.3f} post={cov_post:.3f}"

    out = (ACI(alpha=0.1, gamma=0.02, mode="pooled")
           .fit(p_cal, y_cal)
           .run(p_te, y_te, order=np.arange(len(p_te))))
    tail = out["covered"][n_half + 500:]          # after a few hundred steps
    cov_aci = float(tail.mean())
    assert cov_aci > cov_post + 0.03, \
        f"ACI ({cov_aci:.3f}) failed to beat static ({cov_post:.3f})"
    assert abs(cov_aci - 0.9) <= 0.05, f"ACI post-shift {cov_aci:.4f}"


def test_static_once_refuses_to_refit():
    rng = np.random.default_rng(SEED)
    p, y = _synth(200, rng)
    g = np.zeros(200, dtype=int)
    s = StaticOnce().fit(p, y, groups=g)
    with pytest.raises(RuntimeError, match="refit"):
        s.fit(p, y, groups=g)


def test_aci_mondrian_streams_are_independent():
    """A drifting domain must not move the other domain's alpha_t."""
    rng = np.random.default_rng(SEED + 2)
    n = 4000
    p_cal, y_cal = _synth(n, rng)
    g_cal = np.array(["A", "B"]).repeat(n // 2)

    p_a, y_a = _synth(3000, rng)                 # stable
    p_b, y_b = _synth(3000, rng, flip=0.4)       # drifting
    p_te = np.concatenate([p_a, p_b])
    y_te = np.concatenate([y_a, y_b])
    g_te = np.array(["A"] * 3000 + ["B"] * 3000)

    out = (ACI(alpha=0.1, gamma=0.02, mode="mondrian")
           .fit(p_cal, y_cal, groups=g_cal)
           .run(p_te, y_te, groups=g_te, order=np.arange(len(p_te))))
    a_final = out["alpha_final"]
    assert a_final["B"] < a_final["A"], \
        f"drifting stream should end at a lower alpha: {a_final}"
    # The stable stream stays near target; the drifting one is pushed down.
    assert abs(a_final["A"] - 0.1) < 0.05


def test_aci_requires_temporal_order():
    rng = np.random.default_rng(SEED)
    p, y = _synth(100, rng)
    aci = ACI().fit(p, y)
    with pytest.raises(ValueError, match="close_time order"):
        aci.run(p, y)


def test_aci_alpha_updates_only_from_past():
    """alpha_t at step i must not depend on outcomes after i in `order`.

    Perturbing the last observed outcome may only change the levels used
    at later steps, never earlier ones.
    """
    rng = np.random.default_rng(SEED + 3)
    p_cal, y_cal = _synth(2000, rng)
    p_te, y_te = _synth(400, rng)
    order = np.arange(len(p_te))
    base = ACI(gamma=0.02).fit(p_cal, y_cal).run(p_te, y_te, order=order)
    y2 = y_te.copy()
    y2[300] = 1 - y2[300]
    alt = ACI(gamma=0.02).fit(p_cal, y_cal).run(p_te, y2, order=order)
    assert np.array_equal(base["alpha_t"][:301], alt["alpha_t"][:301])
    assert not np.array_equal(base["alpha_t"][301:], alt["alpha_t"][301:])


def test_tiny_group_falls_back_instead_of_covering_everything():
    """Regression: a Mondrian group too small to resolve alpha must NOT
    emit the trivial set {0,1} for every point.

    Found in Step 16: the first walk-forward calibration window had 3
    Sports and 1 Politics markets at tau=24h, so their frozen thresholds
    were +inf and 25.3% of the tau=24h test rows were 'covered' by
    construction. That turned 'these products did not exist yet' into an
    apparent upward coverage drift.
    """
    rng = np.random.default_rng(SEED)
    n_big = 2000
    p_big, y_big = _synth(n_big, rng)
    p_tiny, y_tiny = _synth(3, rng)              # 3 markets: rank 4 > 3
    p_cal = np.concatenate([p_big, p_tiny])
    y_cal = np.concatenate([y_big, y_tiny])
    g_cal = np.array(["Big"] * n_big + ["Tiny"] * 3)

    assert conformal_qhat(np.sort(_scores(p_tiny, y_tiny)), 0.1) == np.inf

    st = StaticOnce(alpha=0.1).fit(p_cal, y_cal, groups=g_cal)
    q_tiny = st.qhat("Tiny")
    assert np.isfinite(q_tiny), "tiny group still returns an infinite threshold"
    assert q_tiny == conformal_qhat(st.pooled_scores_, 0.1)
    assert st.fellback_.get("Tiny") == 3

    p_te, y_te = _synth(400, rng)
    g_te = np.array(["Tiny"] * 400)
    sets = st.predict_set(p_te, groups=g_te)
    assert not sets.all(), "every set is {0,1} — the artifact is still there"
    cov = _cov(sets, y_te)
    assert 0.85 <= cov <= 0.95, f"fallback coverage {cov:.3f} off nominal"


def test_aci_tiny_group_falls_back():
    rng = np.random.default_rng(SEED + 5)
    p_big, y_big = _synth(2000, rng)
    p_tiny, y_tiny = _synth(3, rng)
    p_cal = np.concatenate([p_big, p_tiny])
    y_cal = np.concatenate([y_big, y_tiny])
    g_cal = np.array(["Big"] * 2000 + ["Tiny"] * 3)
    p_te, y_te = _synth(600, rng)
    g_te = np.array(["Tiny"] * 600)
    out = (ACI(alpha=0.1, gamma=0.005, mode="mondrian")
           .fit(p_cal, y_cal, groups=g_cal)
           .run(p_te, y_te, groups=g_te, order=np.arange(600)))
    assert not out["sets"].all(), "ACI still covering everything by construction"
    assert abs(float(out["covered"].mean()) - 0.9) <= 0.05


def test_split_conformal_mondrian_tiny_group_falls_back():
    from recalibrators import SplitConformal
    rng = np.random.default_rng(SEED + 6)
    p_big, y_big = _synth(2000, rng)
    p_tiny, y_tiny = _synth(3, rng)
    p_cal = np.concatenate([p_big, p_tiny])
    y_cal = np.concatenate([y_big, y_tiny])
    g_cal = np.array(["Big"] * 2000 + ["Tiny"] * 3)
    p_te, y_te = _synth(400, rng)
    g_te = np.array(["Tiny"] * 400)
    m = SplitConformal(mode="mondrian").fit(p_cal, y_cal, groups=g_cal)
    sets = m.predict_set(p_te, 0.1, groups=g_te)
    assert not sets.all(), "SplitConformal mondrian still emits {0,1}"
    assert m.fellback_.get("Tiny") == 3
    assert 0.85 <= _cov(sets, y_te) <= 0.95
