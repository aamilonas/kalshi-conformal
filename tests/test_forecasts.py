"""Step 10 — sanity suite for forecasts_unfiltered.parquet / forecasts.parquet.

Run with:  py -m pytest tests/test_forecasts.py -v
"""
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

DERIVED = "E:/pm/kalshi-conformal/data/derived"
RESULTS = Path("E:/pm/kalshi-conformal/results")

TAUS = {"1h": pd.Timedelta("1h"), "6h": pd.Timedelta("6h"),
        "24h": pd.Timedelta("24h"), "1w": pd.Timedelta("7D"),
        "1mo": pd.Timedelta("30D")}
TAU_ORDER = ["1h", "6h", "24h", "1w", "1mo"]
BIG_DOMAINS = ["Sports", "Crypto", "Finance", "Politics"]


@pytest.fixture(scope="module")
def unf():
    return pd.read_parquet(f"{DERIVED}/forecasts_unfiltered.parquet")


@pytest.fixture(scope="module")
def markets():
    con = duckdb.connect()
    df = con.sql(f"""
        SELECT ticker, domain, close_time,
               CASE WHEN result='yes' THEN 1 ELSE 0 END AS outcome
        FROM '{DERIVED}/markets_classified.parquet'
        WHERE result IN ('yes','no') AND close_time IS NOT NULL
    """).df()
    con.close()
    return df


# ── Test 1: join correctness ────────────────────────────────────────
def test_1_join_correctness(unf):
    """Every selected trade must be STRICTLY before close_time - tau.
    Catches a flipped ASOF inequality that eyeballing never would."""
    cutoff = unf["close_time"] - unf["tau"].map(TAUS)
    bad = unf[~(unf["trade_time"] < cutoff)]
    assert len(bad) == 0, f"{len(bad)} rows violate trade_time < close_time - tau:\n{bad.head()}"


# ── Test 2: base rates vs Step 8 ────────────────────────────────────
def test_2_base_rates(unf, markets):
    """Per-domain mean(outcome) at tau=1h (>=10-trade markets) must equal the
    base rate of the MATCHED population — resolved >=10-trade markets whose
    first trade is >=1h before close — recomputed independently from the base
    tables (no ASOF join). Step 8's population differs structurally: in
    Crypto/Finance most >=10-trade markets are hourlies with no trade 1h
    pre-close (76,181 vs 13,126 reachable in Crypto), so their Step 8 rates
    (40.7/37.7) CANNOT be matched at tau=1h (37.3/31.8) — a selection effect,
    not a join bug; deltas vs the matched population are ~0.00pp everywhere."""
    con = duckdb.connect()
    con.sql("SET TimeZone='UTC'")
    matched = con.sql(f"""
        WITH resolved AS (
          SELECT m.ticker, m.domain, m.close_time,
                 CASE WHEN m.result='yes' THEN 1.0 ELSE 0.0 END AS y
          FROM '{DERIVED}/markets_classified.parquet' m
          WHERE m.result IN ('yes','no') AND m.close_time IS NOT NULL
        ),
        t AS (
          SELECT ticker, COUNT(*) AS n, MIN(created_time) AS first_trade
          FROM '{DERIVED}/trades_dedup.parquet' GROUP BY ticker
        )
        SELECT r.domain, 100 * AVG(y) AS matched_base_rate
        FROM resolved r JOIN t ON t.ticker = r.ticker
        WHERE t.n >= 10 AND t.first_trade < r.close_time - INTERVAL '1 hour'
        GROUP BY r.domain
    """).df().set_index("domain")["matched_base_rate"]
    con.close()

    got = 100 * (unf[(unf["tau"] == "1h") & (unf["n_trades_market"] >= 10)]
                 .groupby("domain")["outcome"].mean())
    report = pd.DataFrame({"matched_base_rate": matched, "forecasts_1h_ge10": got}).dropna()
    report["delta_pp"] = report["forecasts_1h_ge10"] - report["matched_base_rate"]
    print("\n" + report.round(3).to_string())
    for d in BIG_DOMAINS:
        assert abs(report.loc[d, "delta_pp"]) < 1.0, \
            f"{d}: base-rate delta {report.loc[d, 'delta_pp']:.2f}pp exceeds 1.0pp"

    # Outcome integrity: within the forecast file, a ticker's outcome must be
    # constant and equal to the market table's outcome for every tau.
    per_ticker = unf.groupby("ticker")["outcome"].nunique()
    assert (per_ticker == 1).all(), "same ticker has different outcomes across rows"
    merged = unf.drop_duplicates("ticker")[["ticker", "outcome"]].merge(
        markets[["ticker", "outcome"]], on="ticker", suffixes=("_f", "_m"))
    assert (merged["outcome_f"] == merged["outcome_m"]).all(), \
        "forecast outcome disagrees with markets table"


# ── Test 3: cell sizes ──────────────────────────────────────────────
def test_3_cell_sizes(unf):
    grid = unf.pivot_table(index="domain", columns="tau", values="ticker",
                           aggfunc="count", fill_value=0)[TAU_ORDER]
    print("\n(domain x tau) row counts, unfiltered:")
    print(grid.to_string())
    for d in BIG_DOMAINS:
        assert d in grid.index, f"domain {d} missing entirely"
        for t in TAU_ORDER:
            assert grid.loc[d, t] >= 50, f"near-zero cell: {d} x {t} = {grid.loc[d, t]}"
    # Fewer markets have a trade 1mo before close than 1h before.
    for d in grid.index:
        counts = grid.loc[d, TAU_ORDER].tolist()
        assert all(a >= b for a, b in zip(counts, counts[1:])), \
            f"{d}: counts not non-increasing across taus: {counts}"


# ── Test 4: price distributions ─────────────────────────────────────
def test_4_price_distributions(unf):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    domains = sorted(unf["domain"].unique())
    fig, axes = plt.subplots(2, len(domains), figsize=(3.2 * len(domains), 6),
                             sharex=True)
    for j, d in enumerate(domains):
        for i, t in enumerate(["1h", "1mo"]):
            ax = axes[i, j]
            sub = unf[(unf["domain"] == d) & (unf["tau"] == t)]["price"]
            ax.hist(sub, bins=20, range=(0, 1), color="#1f77b4")
            ax.set_title(f"{d} τ={t} (n={len(sub):,})", fontsize=8)
    fig.tight_layout()
    RESULTS.mkdir(exist_ok=True)
    fig.savefig(RESULTS / "fig_price_hists.png", dpi=120)
    plt.close(fig)

    # Soft flag: at tau=1h mass should pile near 0/1.
    p1h = unf[unf["tau"] == "1h"]["price"]
    extreme = ((p1h < 0.1) | (p1h > 0.9)).mean()
    print(f"\ntau=1h mass in [0,0.1)+(0.9,1]: {100*extreme:.1f}%")
    if extreme < 0.25:
        print("WARNING: tau=1h price distribution looks flat — inspect fig_price_hists.png")


# ── Test 5: hand spot-check of five markets ─────────────────────────
def _pick_spotcheck_tickers(unf):
    # (domain, exact ticker or preferred prefixes). Prefix steering keeps the
    # dump human-recognizable: without it, "Sports" alphabetically picks
    # ARGINFLATIONM (Argentina inflation!, labeled Sports by Le's substring
    # quirk ARGI"NFL"ATIONM) and "Weather" picks the malformed '-23MAR-T2'.
    prefs = [("Politics", "PRES-2024-DJT", None),
             ("Sports", None, ("KXNFLGAME", "NFLGAME", "NBAGAME")),
             ("Weather", None, ("HIGHNY", "HIGHCHI", "HIGH")),
             ("Crypto", None, ("BTCD", "BTC")),
             ("Finance", None, ("INXD", "AAAGAS", "FED"))]
    have = unf[unf["tau"] == "1h"]
    picks = []
    for domain, exact, prefixes in prefs:
        cands = have[have["domain"] == domain]
        if exact is not None and (cands["ticker"] == exact).any():
            picks.append(exact)
            continue
        if prefixes:
            pref = cands[cands["ticker"].str.startswith(prefixes)]
            if len(pref):
                cands = pref
        # prefer a modest, readable history
        mid = cands[(cands["n_trades_market"] >= 20) & (cands["n_trades_market"] <= 200)]
        pool = mid if len(mid) else cands
        assert len(pool), f"no spot-check candidate in domain {domain}"
        picks.append(pool.sort_values("ticker")["ticker"].iloc[0])
    return picks


def test_5_spotcheck(unf, markets):
    con = duckdb.connect()
    tickers = _pick_spotcheck_tickers(unf)
    lines = []
    for tk in tickers:
        hist = con.sql(f"""
            SELECT trade_id, created_time, yes_price, "count"
            FROM '{DERIVED}/trades_dedup.parquet'
            WHERE ticker = '{tk}' ORDER BY created_time
        """).df()
        mrow = markets[markets["ticker"] == tk].iloc[0]
        lines.append("=" * 90)
        lines.append(f"TICKER {tk}  domain-close={mrow['close_time']}  "
                     f"outcome={mrow['outcome']}  n_trades={len(hist)}")
        if len(hist) <= 500:
            lines.append(hist.to_string(index=False))
        for tau_name, td in TAUS.items():
            cutoff = mrow["close_time"] - td
            before = hist[hist["created_time"] < cutoff]
            frow = unf[(unf["ticker"] == tk) & (unf["tau"] == tau_name)]
            lines.append("-" * 90)
            lines.append(f"tau={tau_name}  cutoff={cutoff}")
            if len(hist) > 500:
                lo = max(0, len(before) - 10)
                lines.append(hist.iloc[lo:len(before) + 10].to_string(index=False))
            if len(before) == 0:
                assert len(frow) == 0, f"{tk} tau={tau_name}: forecast exists but no trade before cutoff"
                lines.append("  no trade before cutoff -> correctly absent from forecasts")
                continue
            assert len(frow) == 1, f"{tk} tau={tau_name}: expected 1 forecast row, got {len(frow)}"
            frow = frow.iloc[0]
            last_t = before["created_time"].max()
            prices_at_last = set(before.loc[before["created_time"] == last_t, "yes_price"])
            assert frow["trade_time"] == last_t, \
                f"{tk} tau={tau_name}: forecast used {frow['trade_time']}, last pre-cutoff is {last_t}"
            assert round(frow["price"] * 100) in prices_at_last, \
                f"{tk} tau={tau_name}: price {frow['price']} not among trades at {last_t}"
            lines.append(f"  OK: forecast = {frow['price']:.2f} @ {frow['trade_time']} "
                         f"(genuinely last trade before cutoff)")
    con.close()
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "spotcheck.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nSpot-check dump written to {RESULTS / 'spotcheck.txt'} ({len(tickers)} tickers)")
