"""Step 15c — Adaptive Conformal Inference and the static-once comparator.

Three coverage curves feed H3:

1. ``StaticOnce``   — a Mondrian split-conformal threshold fit ONCE on the
   calibration window of the first test quarter and never refit. The naive
   user; it should decay as Kalshi drifts.
2. rolling-refit    — plain ``SplitConformal(mode='mondrian')`` refit every
   quarter on the trailing 12 months. Lives in ``recalibrators.py``; the
   walk-forward driver handles it.
3. ``ACI``          — Adaptive Conformal Inference (Gibbs & Candes 2021,
   "Adaptive conformal inference under distribution shift", NeurIPS 34).

ACI update, applied after each test point's outcome is observed:

    alpha_{t+1} = alpha_t + gamma * (alpha - err_t)

where ``err_t = 1`` iff the prediction set at step t missed the realised
label. A miss pushes ``alpha_t`` down, which widens the next set; a run of
hits lets it drift back up. The calibration scores stay fixed within a
window — only the level moves.

Gibbs & Candes (2024, "Conformal inference for online prediction with
arbitrary distribution shifts", JMLR 25) show the step size itself can be
selected adaptively by aggregating experts over a grid of gammas. That is
NOT implemented here; we run a fixed grid (0.005 primary, plus 0.01 and
0.02) and report all three so the choice cannot be cherry-picked.

**Temporal honesty.** ``err_t`` needs the realised outcome, which a
deployed user only learns at resolution. Test points are therefore
processed in ``close_time`` order and ``alpha_t`` is updated strictly from
already-resolved outcomes. A future outcome never touches an earlier
threshold.
"""
from __future__ import annotations

import numpy as np

TARGET_ALPHA = 0.1
GAMMA_PRIMARY = 0.005
GAMMAS = (0.005, 0.01, 0.02)
ALPHA_CLIP = (0.001, 0.999)


def _scores_from_labels(p_hat, y):
    """Nonconformity s = 1 - p_hat_y, matching ``SplitConformal``."""
    p_hat = np.asarray(p_hat, dtype=float).ravel()
    y = np.asarray(y).astype(int).ravel()
    return np.where(y == 1, 1.0 - p_hat, p_hat)


def conformal_qhat(sorted_scores, alpha):
    """Split-conformal threshold: the ceil((n+1)(1-alpha))/n quantile.

    Returns ``+inf`` when the rank exceeds n (alpha too small for the
    calibration set to resolve -> cover both labels) and ``-inf`` when the
    rank falls below 1 (alpha ~ 1 -> empty set). Identical convention to
    ``recalibrators.SplitConformal._qhat``, extended at the low end.
    """
    n = len(sorted_scores)
    if n == 0:
        return np.inf
    rank = int(np.ceil((n + 1) * (1.0 - alpha)))
    if rank > n:
        return np.inf
    if rank < 1:
        return -np.inf
    return float(sorted_scores[rank - 1])


def _label_scores(p_hat):
    """(n, 2) array of s_k = 1 - p_hat_k for k in {0, 1}."""
    p_hat = np.asarray(p_hat, dtype=float).ravel()
    return np.column_stack([p_hat, 1.0 - p_hat])


class StaticOnce:
    """Mondrian split conformal frozen at its first calibration window.

    ``fit`` is callable only once; later calls raise. That is deliberate —
    H3's whole point is that this comparator does not refit, and a silent
    refit would erase the effect we are measuring.
    """

    def __init__(self, alpha=TARGET_ALPHA):
        self.alpha = alpha
        self._fitted = False

    def fit(self, p_cal, y_cal, groups):
        if self._fitted:
            raise RuntimeError(
                "StaticOnce refit attempted; it must keep the first "
                "calibration window (H3 depends on this)")
        scores = _scores_from_labels(p_cal, y_cal)
        g = np.asarray(groups)
        self.pooled_scores_ = np.sort(scores)
        self.group_scores_ = {k: np.sort(scores[g == k]) for k in np.unique(g)}
        self.cal_end_ = None          # set by the driver for provenance
        self._fitted = True
        return self

    def qhat(self, group, alpha=None):
        a = self.alpha if alpha is None else alpha
        sc = self.group_scores_.get(group)
        if sc is None or len(sc) == 0:
            sc = self.pooled_scores_
        return conformal_qhat(sc, a)

    def predict_set(self, p_test, groups, alpha=None):
        s = _label_scores(p_test)
        g = np.asarray(groups)
        q = np.array([self.qhat(k, alpha) for k in g])[:, None]
        return s <= q


class ACI:
    """Adaptive Conformal Inference over a fixed calibration score set.

    ``mode='pooled'``   — one ``alpha_t`` and one score set for everything.
    ``mode='mondrian'`` — an independent ``alpha_t`` and score set per
    domain, so a drifting domain does not drag the others.
    """

    def __init__(self, alpha=TARGET_ALPHA, gamma=GAMMA_PRIMARY,
                 mode="pooled"):
        if mode not in ("pooled", "mondrian"):
            raise ValueError(f"mode must be pooled|mondrian, got {mode!r}")
        self.alpha = alpha
        self.gamma = gamma
        self.mode = mode

    def fit(self, p_cal, y_cal, groups=None):
        scores = _scores_from_labels(p_cal, y_cal)
        self.pooled_scores_ = np.sort(scores)
        if self.mode == "mondrian":
            if groups is None:
                raise ValueError("mode='mondrian' requires groups=")
            g = np.asarray(groups)
            self.group_scores_ = {k: np.sort(scores[g == k])
                                  for k in np.unique(g)}
        else:
            self.group_scores_ = {}
        return self

    def _scores_for(self, key):
        sc = self.group_scores_.get(key)
        if sc is None or len(sc) == 0:
            return self.pooled_scores_
        return sc

    def run(self, p_test, y_test, groups=None, order=None):
        """Stream the test fold in temporal order, updating alpha_t.

        ``order`` is the index array that sorts the test fold by
        ``close_time``; pass it explicitly so the caller owns the temporal
        contract. Returns a dict with ``sets`` (n, 2), ``alpha_t`` (the
        level USED at each step, in original row order), ``covered``, and
        ``alpha_final`` per stream.
        """
        p_test = np.asarray(p_test, dtype=float).ravel()
        y_test = np.asarray(y_test).astype(int).ravel()
        n = len(p_test)
        if order is None:
            raise ValueError(
                "order= is required: ACI must consume the test fold in "
                "close_time order, never in storage order")
        order = np.asarray(order)
        if self.mode == "mondrian":
            if groups is None:
                raise ValueError("mode='mondrian' requires groups=")
            g = np.asarray(groups)
        else:
            g = np.zeros(n, dtype=int)          # single stream

        s_lab = _label_scores(p_test)
        sets = np.zeros((n, 2), dtype=bool)
        alpha_used = np.empty(n, dtype=float)
        alpha_t = {k: self.alpha for k in np.unique(g)}
        lo, hi = ALPHA_CLIP

        for i in order:
            key = g[i]
            a = alpha_t[key]
            alpha_used[i] = a
            if a <= 0.0:
                st = np.array([True, True])
            elif a >= 1.0:
                st = np.array([False, False])
            else:
                q = conformal_qhat(self._scores_for(key), a)
                st = s_lab[i] <= q
            sets[i] = st
            err = 0.0 if st[y_test[i]] else 1.0
            alpha_t[key] = float(np.clip(a + self.gamma * (self.alpha - err),
                                         lo, hi))

        covered = sets[np.arange(n), y_test]
        return dict(sets=sets, alpha_t=alpha_used, covered=covered,
                    alpha_final=dict(alpha_t))
