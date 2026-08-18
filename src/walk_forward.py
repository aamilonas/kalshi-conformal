"""Step 15 — prospective walk-forward evaluation.

Rolling quarterly protocol on market ``close_time``:

    test quarter Q  = calendar quarter
    calibration     = the trailing 12 months of markets closing strictly
                      before the first day of Q

Every window is refit from scratch (except ``static_once``, which is the
point of H3). The leakage assertion runs on every window and is fatal.

Everything downstream — H1, H2, H3, the bootstrap, the robustness sweeps —
aggregates from the single tidy file this writes,
``results/walk_forward_long.csv``. Per-row predictions go to
``data/derived/wf_predictions.parquet`` (gitignored) so Step 17 never has
to rerun the walk-forward.

Deviation from the instruction file, deliberate and reported at gate 1
---------------------------------------------------------------------
PHASE4_5_WALKFORWARD.md 15a picks the start quarter as the earliest where
EVERY domain has >= 200 test rows at tau=24h and the trailing-12-month
calibration set has >= 1,000 rows, and expects that to land "around 2022".
It cannot. Kalshi launched single-game sports, hourly crypto and most
politics products in 2024-25 (Phase 3 notes, DATA_GUIDE 6.2), so the
per-domain clause is not satisfied until 2025Q1 and would leave THREE test
quarters — not enough for H3 to say anything about coverage over time.

The two clauses are therefore separated:

* the START QUARTER is set by the pooled clause alone (trailing-12-month
  calibration >= ``MIN_CAL_POOLED``), evaluated per tau;
* the per-domain clause becomes a per-row FLAG, not a gate. Every row
  carries its own ``n_test``/``n_cal``, and ``results/table_wf_counts.csv``
  marks which domain-quarters clear 200 test rows, so the strict schedule
  is recoverable as a filter over these results and the two can be compared
  side by side.

No coverage number is ever reported without the count it rests on.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

from adaptive import ACI, GAMMAS, StaticOnce, TARGET_ALPHA
from metrics import avg_set_size, brier, coverage, ece, log_loss, \
    reliability_bin_table
from paths import DERIVED, RESULTS
from recalibrators import HistogramBinning, Isotonic, Platt, Raw, \
    SplitConformal, VennAbers

DOMAINS = ["Sports", "Crypto", "Politics", "Finance", "Weather",
           "Entertainment"]
ALPHAS = [0.1, 0.05, 0.2]              # 0.1 primary
TAUS_PRIMARY = ["24h", "1w"]
TAUS_EXTRA = ["6h", "1mo"]

MIN_CAL_POOLED = 1000                  # 15a, pooled clause
MIN_DOMAIN_TEST = 200                  # 15a, per-domain clause -> flag only
MIN_DOMAIN_CAL = 100                   # below this a per-domain fit is junk
CAL_MONTHS = 12

PROB_METHODS = ["raw", "platt", "isotonic", "binning10", "venn_abers"]


def _make(name):
    return {"raw": Raw, "platt": Platt, "isotonic": Isotonic,
            "binning10": lambda: HistogramBinning(10),
            "venn_abers": VennAbers}[name]()


@dataclass
class Window:
    quarter: str
    cal_start: pd.Timestamp
    cal_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    cal: pd.DataFrame
    test: pd.DataFrame


def load(tau, path=None, price_lo=None, price_hi=None, min_trades=None):
    """Load one horizon. The filter arguments exist for Step 18 only."""
    src = path or f"{DERIVED}/forecasts.parquet"
    fc = pd.read_parquet(src)
    fc = fc[(fc["tau"] == tau) & (fc["domain"].isin(DOMAINS))].copy()
    if price_lo is not None:
        fc = fc[(fc["price"] >= price_lo) & (fc["price"] <= price_hi)]
    if min_trades is not None:
        fc = fc[fc["n_trades_market"] >= min_trades]
    fc["close_time"] = pd.to_datetime(fc["close_time"], utc=True)
    return fc.sort_values("close_time").reset_index(drop=True)


def build_windows(fc, min_cal=MIN_CAL_POOLED):
    """Quarterly windows, trailing-12-month calibration, no partial tail."""
    t = fc["close_time"]
    data_end = t.max()
    periods = pd.period_range(t.min().tz_convert(None),
                              t.max().tz_convert(None), freq="Q")
    out = []
    for per in periods:
        ts = per.to_timestamp(how="start").tz_localize("UTC")
        # Half-open [ts, te): te is the NEXT quarter's start exactly, so
        # consecutive test windows abut without sharing an instant.
        te = (per + 1).to_timestamp(how="start").tz_localize("UTC")
        if te > data_end:
            continue                                   # partial quarter
        cs = ts - pd.DateOffset(months=CAL_MONTHS)
        cal = fc[(t >= cs) & (t < ts)]
        test = fc[(t >= ts) & (t < te)]
        if len(cal) < min_cal or len(test) == 0:
            continue
        out.append(Window(str(per), cs, ts, ts, te,
                          cal.reset_index(drop=True),
                          test.reset_index(drop=True)))
    return out


def assert_no_leakage(w):
    """Non-negotiable. Fail loudly, per window."""
    cm, tm = w.cal["close_time"].max(), w.test["close_time"].min()
    if not cm < tm:
        raise AssertionError(
            f"LEAKAGE in {w.quarter}: cal max {cm} !< test min {tm}")
    if not (cm < w.test_start <= tm):
        raise AssertionError(
            f"LEAKAGE in {w.quarter}: boundary {w.test_start} not between "
            f"cal max {cm} and test min {tm}")
    if w.cal["close_time"].min() < w.cal_start:
        raise AssertionError(f"{w.quarter}: calibration window overruns start")


class RowSink:
    """Accumulates the tidy long rows and the per-row prediction rows."""

    def __init__(self):
        self.rows = []
        self.preds = []

    def metrics(self, w, tau, method, scope, domain, p_hat, y, alpha="NA"):
        n = len(y)
        if n == 0:
            return
        vals = [("brier", brier(p_hat, y)), ("log_loss", log_loss(p_hat, y))]
        if n >= 20:                       # ECE on <20 points is meaningless
            vals.append(("ece", ece(p_hat, y)))
        self._emit(w, tau, method, scope, domain, alpha, vals, n)

    def sets(self, w, tau, method, scope, domain, sets, y, alpha):
        n = len(y)
        if n == 0:
            return
        self._emit(w, tau, method, scope, domain, alpha,
                   [("coverage", coverage(sets, y)),
                    ("avg_set_size", avg_set_size(sets))], n)

    def counts(self, w, tau, domain, n_test, n_cal):
        self._emit(w, tau, "na", "na", domain, "NA",
                   [("n_test", float(n_test)), ("n_cal", float(n_cal))],
                   n_test)

    def _emit(self, w, tau, method, scope, domain, alpha, vals, n):
        for metric, value in vals:
            self.rows.append(dict(
                test_quarter=w.quarter,
                cal_start=w.cal_start.date().isoformat(),
                cal_end=w.cal_end.date().isoformat(),
                test_start=w.test_start.date().isoformat(),
                test_end=(w.test_end - pd.Timedelta(days=1)).date()
                .isoformat(),
                tau=tau, method=method, fit_scope=scope, domain=domain,
                alpha=alpha, metric=metric, value=float(value), n_test=int(n)))

    def rowlevel(self, w, tau, method, scope, tickers, domains, p_hat, y,
                 sets=None, alpha=np.nan):
        d = dict(ticker=tickers, test_quarter=w.quarter, tau=tau,
                 method=method, fit_scope=scope, domain=domains,
                 p_hat=p_hat, y=y, alpha=alpha)
        if sets is None:
            d["set0"] = np.nan
            d["set1"] = np.nan
        else:
            d["set0"] = sets[:, 0]
            d["set1"] = sets[:, 1]
        self.preds.append(pd.DataFrame(d))


def _per_domain_report(sink, w, tau, method, scope, p_hat, y, g, alpha="NA"):
    sink.metrics(w, tau, method, scope, "ALL", p_hat, y, alpha)
    for d in DOMAINS:
        s = g == d
        if s.any():
            sink.metrics(w, tau, method, scope, d, p_hat[s], y[s], alpha)


def _per_domain_sets(sink, w, tau, method, scope, sets, y, g, alpha):
    sink.sets(w, tau, method, scope, "ALL", sets, y, alpha)
    for d in DOMAINS:
        s = g == d
        if s.any():
            sink.sets(w, tau, method, scope, d, sets[s], y[s], alpha)


def run_window(w, tau, sink, static, bins_out, n_bins=10,
               methods=None, alphas=None, want_rowlevel=True):
    """One window: every method, both fit scopes, all three comparators."""
    assert_no_leakage(w)
    methods = methods or PROB_METHODS
    alphas = alphas or ALPHAS

    p_cal = w.cal["price"].to_numpy()
    y_cal = w.cal["outcome"].to_numpy().astype(int)
    g_cal = w.cal["domain"].to_numpy()
    p_test = w.test["price"].to_numpy()
    y_test = w.test["outcome"].to_numpy().astype(int)
    g_test = w.test["domain"].to_numpy()
    tick = w.test["ticker"].to_numpy()

    sink.counts(w, tau, "ALL", len(w.test), len(w.cal))
    for d in DOMAINS:
        sink.counts(w, tau, d, int((g_test == d).sum()),
                    int((g_cal == d).sum()))

    # ---- pooled probability fits -------------------------------------
    for name in methods:
        m = _make(name) if name != "binning10" else HistogramBinning(n_bins)
        m.fit(p_cal, y_cal)
        ph = m.predict_proba(p_test)
        _per_domain_report(sink, w, tau, name, "pooled", ph, y_test, g_test)
        if want_rowlevel:
            sink.rowlevel(w, tau, name, "pooled", tick, g_test, ph, y_test)
        if name in ("raw", "platt", "venn_abers"):
            for d in ("Politics", "Sports"):
                s = g_test == d
                if s.sum() >= 20:
                    bt = reliability_bin_table(ph[s], y_test[s])
                    bt.insert(0, "method", name)
                    bt.insert(0, "domain", d)
                    bt.insert(0, "tau", tau)
                    bt.insert(0, "test_quarter", w.quarter)
                    bins_out.append(bt)

    # ---- per-domain probability fits ----------------------------------
    for name in methods:
        ph = np.full(len(p_test), np.nan)
        for d in DOMAINS:
            ci, ti = g_cal == d, g_test == d
            if ci.sum() < MIN_DOMAIN_CAL or not ti.any():
                continue
            m = _make(name) if name != "binning10" else HistogramBinning(n_bins)
            m.fit(p_cal[ci], y_cal[ci])
            ph[ti] = m.predict_proba(p_test[ti])
            sink.metrics(w, tau, name, "per_domain", d, ph[ti], y_test[ti])
        ok = ~np.isnan(ph)
        if ok.any():
            sink.metrics(w, tau, name, "per_domain", "ALL", ph[ok],
                         y_test[ok])
            if want_rowlevel:
                sink.rowlevel(w, tau, name, "per_domain", tick[ok],
                              g_test[ok], ph[ok], y_test[ok])

    # ---- conformal: marginal (pooled) and mondrian (per-domain) -------
    marg = SplitConformal(Raw(), mode="marginal").fit(p_cal, y_cal)
    mond = SplitConformal(Raw(), mode="mondrian").fit(p_cal, y_cal,
                                                     groups=g_cal)
    for a in alphas:
        sm = marg.predict_set(p_test, a)
        so = mond.predict_set(p_test, a, groups=g_test)
        _per_domain_sets(sink, w, tau, "conformal_marginal", "pooled",
                         sm, y_test, g_test, a)
        _per_domain_sets(sink, w, tau, "conformal_mondrian", "per_domain",
                         so, y_test, g_test, a)
        if want_rowlevel and a == TARGET_ALPHA:
            ph = marg.predict_proba(p_test)
            sink.rowlevel(w, tau, "conformal_marginal", "pooled", tick,
                          g_test, ph, y_test, sm, a)
            sink.rowlevel(w, tau, "conformal_mondrian", "per_domain", tick,
                          g_test, ph, y_test, so, a)

    # ---- H3 comparator 1: static-once (never refits) ------------------
    if static is not None:
        for a in alphas:
            ss = static.predict_set(p_test, groups=g_test, alpha=a)
            _per_domain_sets(sink, w, tau, "static_once", "per_domain",
                             ss, y_test, g_test, a)
            if want_rowlevel and a == TARGET_ALPHA:
                sink.rowlevel(w, tau, "static_once", "per_domain", tick,
                              g_test, p_test, y_test, ss, a)
        # Provenance for H3: how much calibration data the FROZEN threshold
        # for each domain rests on, and whether it was estimable at all.
        # A domain that did not exist in the first window falls back to the
        # pooled threshold; without this column that is invisible.
        for d in DOMAINS:
            sink._emit(w, tau, "static_once", "per_domain", d, "NA",
                       [("n_cal_frozen", float(static.n_cal(d))),
                        ("fellback_to_pooled",
                         float(d in getattr(static, "fellback_", {})))],
                       int((g_test == d).sum()))

    # ---- H3 comparator 3: ACI, in close_time order --------------------
    order = np.argsort(w.test["close_time"].to_numpy(), kind="stable")
    for gamma in GAMMAS:
        for mode in ("pooled", "mondrian"):
            name = f"aci_{mode}_g{gamma}"
            aci = ACI(alpha=TARGET_ALPHA, gamma=gamma, mode=mode)
            aci.fit(p_cal, y_cal, groups=g_cal if mode == "mondrian" else None)
            out = aci.run(p_test, y_test,
                          groups=g_test if mode == "mondrian" else None,
                          order=order)
            scope = "pooled" if mode == "pooled" else "per_domain"
            _per_domain_sets(sink, w, tau, name, scope, out["sets"], y_test,
                             g_test, TARGET_ALPHA)
            if want_rowlevel and gamma == GAMMAS[0]:
                sink.rowlevel(w, tau, name, scope, tick, g_test, p_test,
                              y_test, out["sets"], TARGET_ALPHA)


def run_tau(tau, sink, bins_out, **load_kw):
    """All windows for one horizon. Returns the window list (may be empty)."""
    fc = load(tau, **load_kw)
    windows = build_windows(fc)
    if not windows:
        print(f"  tau={tau}: no window clears "
              f"min_cal={MIN_CAL_POOLED}; skipped")
        return []
    print(f"  tau={tau}: {len(windows)} windows, "
          f"{windows[0].quarter} .. {windows[-1].quarter}")
    # H3 static-once is frozen on the FIRST window's calibration set.
    w0 = windows[0]
    static = StaticOnce(alpha=TARGET_ALPHA).fit(
        w0.cal["price"].to_numpy(),
        w0.cal["outcome"].to_numpy().astype(int),
        groups=w0.cal["domain"].to_numpy())
    static.cal_end_ = w0.cal_end
    for w in windows:
        run_window(w, tau, sink, static, bins_out)
        print(f"    {w.quarter}  cal {len(w.cal):>6,} "
              f"({w.cal_start.date()}..{w.cal_end.date()})  "
              f"test {len(w.test):>6,}", flush=True)
    return windows


def counts_table(sink_rows):
    """results/table_wf_counts.csv — every count behind every number."""
    df = sink_rows[sink_rows.metric.isin(["n_test", "n_cal"])]
    wide = df.pivot_table(index=["tau", "test_quarter", "cal_start",
                                 "cal_end", "test_start", "test_end",
                                 "domain"],
                          columns="metric", values="value").reset_index()
    wide["n_test"] = wide["n_test"].astype(int)
    wide["n_cal"] = wide["n_cal"].astype(int)
    wide["meets_200_test"] = wide["n_test"] >= MIN_DOMAIN_TEST
    wide["per_domain_fit_ran"] = wide["n_cal"] >= MIN_DOMAIN_CAL
    return wide.sort_values(["tau", "test_quarter", "domain"])


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    taus = TAUS_PRIMARY + ([] if "--primary-only" in argv else TAUS_EXTRA)
    sink, bins_out = RowSink(), []
    print("Step 15 — walk-forward\n" + "=" * 60)
    ran = {}
    for tau in taus:
        ws = run_tau(tau, sink, bins_out)
        if ws:
            ran[tau] = (ws[0].quarter, ws[-1].quarter, len(ws))

    long = pd.DataFrame(sink.rows)
    key = ["test_quarter", "tau", "method", "fit_scope", "domain", "alpha",
           "metric"]
    dup = long.duplicated(key).sum()
    if dup:
        raise AssertionError(f"{dup} duplicate keys in walk_forward_long")
    long.to_csv(f"{RESULTS}/walk_forward_long.csv", index=False)
    print(f"\nSaved walk_forward_long.csv ({len(long):,} rows)")

    counts_table(long).to_csv(f"{RESULTS}/table_wf_counts.csv", index=False)
    print("Saved table_wf_counts.csv")

    if bins_out:
        rb = pd.concat(bins_out, ignore_index=True)
        assert not rb.isna().any().any(), "reliability bins have holes"
        rb.to_csv(f"{RESULTS}/wf_reliability_bins.csv", index=False)
        print(f"Saved wf_reliability_bins.csv ({len(rb):,} rows)")

    preds = pd.concat(sink.preds, ignore_index=True)
    preds.to_parquet(f"{DERIVED}/wf_predictions.parquet", index=False)
    print(f"Saved wf_predictions.parquet ({len(preds):,} rows)")

    print("\nhorizons run: " + ", ".join(
        f"{t} ({a}..{b}, {n} quarters)" for t, (a, b, n) in ran.items()))
    return long


if __name__ == "__main__":
    main()
