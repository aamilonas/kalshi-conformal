"""Step 15 tests — window construction, leakage, and long-file integrity.

These run against the real `forecasts.parquet`; window construction is
cheap (no fitting), so there is no reason to test it on a mock.
"""
import numpy as np
import pandas as pd
import pytest

import walk_forward as wf
from paths import RESULTS

TAU = "24h"


@pytest.fixture(scope="module")
def windows():
    return wf.build_windows(wf.load(TAU))


@pytest.fixture(scope="module")
def long():
    p = f"{RESULTS}/walk_forward_long.csv"
    try:
        return pd.read_csv(p)
    except FileNotFoundError:
        pytest.skip("walk_forward_long.csv not built yet")


def test_windows_exist(windows):
    assert len(windows) >= 8, f"only {len(windows)} windows"


def test_test_windows_never_overlap(windows):
    for a, b in zip(windows, windows[1:]):
        assert a.test_end <= b.test_start, f"{a.quarter} overlaps {b.quarter}"
    qs = [w.quarter for w in windows]
    assert len(set(qs)) == len(qs), "duplicate test quarters"


def test_every_window_passes_leakage_assertion(windows):
    for w in windows:
        wf.assert_no_leakage(w)
        assert w.cal["close_time"].max() < w.test["close_time"].min()


def test_leakage_assertion_actually_fires(windows):
    """A guard that never fails is not a guard. Poison one row."""
    w = windows[-1]
    bad = wf.Window(w.quarter, w.cal_start, w.cal_end, w.test_start,
                    w.test_end, w.cal.copy(), w.test.copy())
    bad.cal.loc[bad.cal.index[0], "close_time"] = w.test["close_time"].max()
    with pytest.raises(AssertionError, match="LEAKAGE"):
        wf.assert_no_leakage(bad)


def test_calibration_window_is_trailing_12_months(windows):
    for w in windows:
        assert w.cal_end - w.cal_start == pd.DateOffset(months=wf.CAL_MONTHS) \
            or (w.cal_end - pd.DateOffset(months=wf.CAL_MONTHS)) == w.cal_start
        assert w.cal["close_time"].min() >= w.cal_start
        assert w.cal["close_time"].max() < w.cal_end


def test_partial_final_quarter_excluded(windows):
    fc = wf.load(TAU)
    end = fc["close_time"].max()
    last = pd.Period(windows[-1].quarter)
    nxt = (last + 1).to_timestamp(how="end").tz_localize("UTC")
    assert nxt > end, "a quarter ending after the data would be partial"
    assert windows[-1].test_end <= end


def test_min_cal_respected(windows):
    for w in windows:
        assert len(w.cal) >= wf.MIN_CAL_POOLED


def test_long_file_has_no_duplicate_keys(long):
    key = ["test_quarter", "tau", "method", "fit_scope", "domain", "alpha",
           "metric"]
    assert not long.duplicated(key).any()


def test_domain_n_test_sums_to_pooled(long):
    n = long[long.metric == "n_test"]
    for (tau, q), g in n.groupby(["tau", "test_quarter"]):
        pooled = float(g.loc[g.domain == "ALL", "value"].iloc[0])
        parts = float(g.loc[g.domain != "ALL", "value"].sum())
        assert pooled == parts, f"{tau} {q}: {parts} != {pooled}"


def test_static_once_is_frozen_across_quarters(long):
    """Static-once must differ from rolling once the windows diverge.

    In the FIRST test quarter the two are fit on the same data and must
    agree exactly; later they must not, or static-once is secretly refitting.
    """
    c = long[(long.metric == "coverage") & (long.alpha == 0.1)
             & (long.domain == "ALL") & (long.tau == TAU)]
    if c.empty:
        pytest.skip("no coverage rows")
    qs = sorted(c.test_quarter.unique())
    def val(m, q):
        s = c[(c.method == m) & (c.test_quarter == q)]["value"]
        return None if s.empty else float(s.iloc[0])
    first = val("static_once", qs[0]), val("conformal_mondrian", qs[0])
    assert first[0] == pytest.approx(first[1]), \
        "first quarter: static and rolling share a calibration set"
    later = [q for q in qs[1:] if val("static_once", q) is not None]
    assert any(val("static_once", q) != val("conformal_mondrian", q)
               for q in later), "static_once appears to be refitting"


def test_counts_table_flags_thin_domain_quarters(long):
    t = wf.counts_table(long)
    assert {"meets_200_test", "per_domain_fit_ran"} <= set(t.columns)
    assert t["n_test"].min() >= 0
    # The strict 15a schedule must be recoverable as a filter.
    strict = (t[(t.tau == TAU) & (t.domain != "ALL")]
              .groupby("test_quarter")["meets_200_test"].all())
    assert strict.any(), "no quarter clears the strict per-domain rule"
