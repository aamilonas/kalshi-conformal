"""Step 13 — seven recalibrators behind one interface.

Every method consumes calibration-fold forecasts p_cal in (0,1) with binary
outcomes y_cal, and recalibrates test-fold forecasts p_test. No method may
ever see a test-fold outcome (leakage kills the paper).

    class Recalibrator:
        fit(p_cal, y_cal[, groups]) -> self
        predict_proba(p_test) -> (n,) array of P(y=1)
        predict_set(p_test, alpha[, groups]) -> bool (n, 2)   # conformal only

Methods: Raw, Platt, Isotonic, HistogramBinning (the guarantee-bearing
estimator; Gupta et al. 2020, Gupta & Ramdas 2021), inductive VennAbers,
SplitConformal (marginal | mondrian), Adaptive (ACI; stubbed to Phase 4).
"""
import warnings

import numpy as np
import pandas as pd
from scipy.special import logit
from scipy.stats import beta as beta_dist
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

CLIP_LO, CLIP_HI = 0.01, 0.99   # Le's clip, kept for comparability (Step 12)


def _as1d(p):
    return np.asarray(p, dtype=float).ravel()


class Recalibrator:
    def fit(self, p_cal, y_cal):
        return self

    def predict_proba(self, p_test):
        raise NotImplementedError

    def predict_set(self, p_test, alpha):
        raise NotImplementedError


class Raw(Recalibrator):
    """Identity baseline: the market price is the forecast."""

    def predict_proba(self, p_test):
        return _as1d(p_test).copy()


class Platt(Recalibrator):
    """Logistic recalibration on logit(p), same spec as Step 12 (C=10)."""

    def fit(self, p_cal, y_cal):
        X = logit(np.clip(_as1d(p_cal), CLIP_LO, CLIP_HI)).reshape(-1, 1)
        self.clf_ = LogisticRegression(C=10.0, penalty="l2", solver="lbfgs",
                                       max_iter=2000)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.clf_.fit(X, np.asarray(y_cal).astype(int))
        return self

    def predict_proba(self, p_test):
        X = logit(np.clip(_as1d(p_test), CLIP_LO, CLIP_HI)).reshape(-1, 1)
        return self.clf_.predict_proba(X)[:, 1]


class Isotonic(Recalibrator):
    """Monotone recalibration: isotonic regression of y on p."""

    def fit(self, p_cal, y_cal):
        self.iso_ = IsotonicRegression(out_of_bounds="clip")
        self.iso_.fit(_as1d(p_cal), _as1d(y_cal))
        return self

    def predict_proba(self, p_test):
        return np.clip(self.iso_.predict(_as1d(p_test)), 0.0, 1.0)


class HistogramBinning(Recalibrator):
    """Equal-mass histogram binning: p_test -> empirical outcome frequency
    of its calibration bin. Distribution-free guarantees per Gupta et al.
    2020 ("Distribution-free binary classification") and Gupta & Ramdas
    2021 ("Distribution-free calibration guarantees for histogram binning
    without sample splitting"). Stores per-bin counts/successes and exposes
    Clopper-Pearson 95% intervals via bin_table().
    """

    def __init__(self, n_bins=10):
        self.n_bins = n_bins

    def fit(self, p_cal, y_cal):
        p = _as1d(p_cal)
        y = _as1d(y_cal)
        self.edges_ = np.quantile(p, np.linspace(0.0, 1.0, self.n_bins + 1))
        idx = np.searchsorted(self.edges_[1:-1], p, side="right")
        self.n_ = np.bincount(idx, minlength=self.n_bins).astype(int)
        self.successes_ = np.bincount(idx, weights=y,
                                      minlength=self.n_bins).astype(int)
        self.global_mean_ = float(y.mean())
        with np.errstate(invalid="ignore", divide="ignore"):
            self.freq_ = np.where(self.n_ > 0,
                                  self.successes_ / np.maximum(self.n_, 1),
                                  self.global_mean_)
        return self

    def predict_proba(self, p_test):
        idx = np.searchsorted(self.edges_[1:-1], _as1d(p_test), side="right")
        return self.freq_[idx]

    def bin_table(self):
        n, s = self.n_, self.successes_
        ci_lo = np.where(s == 0, 0.0,
                         beta_dist.ppf(0.025, np.maximum(s, 1), n - s + 1))
        ci_hi = np.where(s == n, 1.0,
                         beta_dist.ppf(0.975, s + 1, np.maximum(n - s, 1)))
        return pd.DataFrame(dict(
            bin=np.arange(self.n_bins), lo=self.edges_[:-1],
            hi=self.edges_[1:], n=n, successes=s, freq=self.freq_,
            ci_lo=ci_lo, ci_hi=ci_hi))


class VennAbers(Recalibrator):
    """Inductive Venn-Abers predictor (Vovk & Petej 2014).

    For each test score p_t: fit isotonic on p_cal augmented with (p_t, 0)
    and read the fitted value at p_t -> p0; same with (p_t, 1) -> p1.
    Point prediction p1 / (1 - p0 + p1); interval width p1 - p0.

    Exact, not approximated: fits are computed once per UNIQUE test value
    and broadcast back. Kalshi prices are cents/100, so a test fold has at
    most ~91 distinct values regardless of its length.
    """

    def fit(self, p_cal, y_cal):
        self.p_cal_ = _as1d(p_cal)
        self.y_cal_ = _as1d(y_cal)
        return self

    def _p0_p1(self, p_test):
        uniq, inv = np.unique(_as1d(p_test), return_inverse=True)
        p0u = np.empty(len(uniq))
        p1u = np.empty(len(uniq))
        for i, pt in enumerate(uniq):
            x = np.append(self.p_cal_, pt)
            for lab, store in ((0.0, p0u), (1.0, p1u)):
                iso = IsotonicRegression(out_of_bounds="clip")
                iso.fit(x, np.append(self.y_cal_, lab))
                store[i] = iso.predict([pt])[0]
        return np.clip(p0u, 0, 1)[inv], np.clip(p1u, 0, 1)[inv]

    def predict_interval(self, p_test):
        return self._p0_p1(p_test)

    def predict_proba(self, p_test):
        p0, p1 = self._p0_p1(p_test)
        return p1 / (1.0 - p0 + p1)


class SplitConformal(Recalibrator):
    """Split conformal prediction sets over labels {0, 1}.

    Nonconformity score s = 1 - p_hat_y (base method's probability of the
    TRUE label). Threshold q_hat = ceil((n+1)(1-alpha))/n empirical
    quantile of calibration scores; test set = {k : 1 - p_hat_k <= q_hat}.

    mode='marginal': one pooled q_hat. mode='mondrian': one q_hat per group
    (domain); a test group unseen in calibration falls back to the pooled
    threshold. Note: the base method is fit on the same calibration fold
    its scores are computed on; exact validity holds for bases that need no
    fitting (Raw, the Step 14 default).
    """

    def __init__(self, base=None, mode="marginal"):
        self.base = base if base is not None else Raw()
        self.mode = mode

    def fit(self, p_cal, y_cal, groups=None):
        p = _as1d(p_cal)
        y = np.asarray(y_cal).astype(int)
        self.base.fit(p, y)
        phat = self.base.predict_proba(p)
        scores = np.where(y == 1, 1.0 - phat, phat)
        self.scores_ = np.sort(scores)
        if self.mode == "mondrian":
            if groups is None:
                raise ValueError("mode='mondrian' requires groups= in fit()")
            g = np.asarray(groups)
            self.group_scores_ = {k: np.sort(scores[g == k])
                                  for k in np.unique(g)}
        return self

    @staticmethod
    def _qhat(sorted_scores, alpha):
        n = len(sorted_scores)
        rank = int(np.ceil((n + 1) * (1.0 - alpha)))
        if rank > n:
            return np.inf          # cover everything
        return float(sorted_scores[rank - 1])

    def predict_proba(self, p_test):
        return self.base.predict_proba(_as1d(p_test))

    def predict_set(self, p_test, alpha, groups=None):
        phat = self.base.predict_proba(_as1d(p_test))
        s_label = np.column_stack([phat, 1.0 - phat])   # s_k = 1 - p_hat_k
        if self.mode == "mondrian":
            if groups is None:
                raise ValueError("mode='mondrian' requires groups= here")
            g = np.asarray(groups)
            q = np.empty(len(phat))
            fallback = self._qhat(self.scores_, alpha)
            for k in np.unique(g):
                sc = self.group_scores_.get(k)
                q[g == k] = self._qhat(sc, alpha) if sc is not None and len(sc) \
                    else fallback
            q = q[:, None]
        else:
            q = self._qhat(self.scores_, alpha)
        return s_label <= q


class Adaptive(Recalibrator):
    """Adaptive Conformal Inference (Gibbs & Candes 2021). Phase 4."""

    def fit(self, p_cal, y_cal):
        raise NotImplementedError("Phase 4")

    def predict_proba(self, p_test):
        raise NotImplementedError("Phase 4")

    def predict_set(self, p_test, alpha):
        raise NotImplementedError("Phase 4")
