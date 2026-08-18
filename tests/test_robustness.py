"""Step 18 tests — guards for the two silent failures this step produced.

Neither bug raised anything; both just made rows disappear and left a
verdict that could only say "HOLDS".
"""
import numpy as np
import pandas as pd
import pytest

import robustness as rb


def _long(alpha_na):
    """Minimal walk_forward-shaped long rows. `alpha_na` picks the
    representation of 'no alpha': the in-memory string or the post-CSV NaN.
    """
    rows = []
    for q in ["2024Q1", "2024Q2"]:
        for meth, val in [("raw", 0.20), ("platt", 0.21)]:
            rows.append(dict(test_quarter=q, tau="24h", method=meth,
                             fit_scope="pooled", domain="ALL",
                             alpha=alpha_na, metric="brier", value=val,
                             n_test=100, variant="v"))
        rows.append(dict(test_quarter=q, tau="24h",
                         method="conformal_marginal", fit_scope="pooled",
                         domain="ALL", alpha=0.1, metric="coverage",
                         value=0.9, n_test=100, variant="v"))
    return pd.DataFrame(rows)


@pytest.mark.parametrize("alpha_na", ["NA", np.nan])
def test_probability_rows_survive_both_alpha_representations(alpha_na):
    """The bug: alpha=='NA' in memory is not NaN, so isna() dropped every
    probability metric and the sweep silently became conformal-only."""
    agg = rb.aggregate(_long(alpha_na))
    assert set(agg.method) == {"raw", "platt", "conformal_marginal"}, \
        f"lost probability rows for alpha={alpha_na!r}: {set(agg.method)}"
    assert float(agg[agg.method == "raw"].value.iloc[0]) == \
        pytest.approx(0.20)


def test_aggregate_is_n_weighted():
    df = _long(np.nan)
    df.loc[df.test_quarter == "2024Q1", "n_test"] = 300
    df.loc[(df.test_quarter == "2024Q1") & (df.method == "raw"),
           "value"] = 0.10
    agg = rb.aggregate(df)
    got = float(agg[(agg.method == "raw")].value.iloc[0])
    assert got == pytest.approx((0.10 * 300 + 0.20 * 100) / 400)


def test_verdict_refuses_to_pass_vacuously():
    """An empty H1 slice used to report 'HOLDS in every variant'."""
    empty = rb.aggregate(_long(np.nan))
    empty = empty[empty.metric == "coverage"]      # no brier rows at all
    with pytest.raises(AssertionError, match="vacuous"):
        rb.verdict(empty)


def test_verdict_catches_a_planted_sign_flip():
    df = _long(np.nan)
    df.loc[df.method == "platt", "value"] = 0.15   # beats raw 0.20
    ok_h1, _, lines = rb.verdict(rb.aggregate(df))
    assert not ok_h1
    assert any("SIGN FLIP" in ln for ln in lines)
