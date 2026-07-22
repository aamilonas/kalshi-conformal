"""Step 8 — RECONCILIATION GATE vs Le (2026) Table 1.

Reproduces Le's exact Table 1 queries (src/pipeline.py::load_kalshi_market_stats
in namanhzz/prediction-market-calibration) against OUR deduped tables, with Le's
DATE_CUTOFF (2025-12-31T23:59:59Z) so a newer archive still reconciles.

Le's Table 1 definitions:
  n_markets  = resolved markets (status='finalized', result in yes/no) having
               >= 10 trades with created_time <= cutoff
  n_trades   = count of those trades on those markets
  base_rate  = % of those markets with result = 'yes'

Writes results/table_reconciliation.csv. HARD STOP: user reviews before Step 9.
"""
import sys
from pathlib import Path

import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from classify_domains import DOMAINS, le_group  # noqa: E402

DERIVED = "E:/pm/kalshi-conformal/data/derived"
RESULTS = "E:/pm/kalshi-conformal/results"
DATE_CUTOFF = "2025-12-31T23:59:59Z"  # Le's config.DATE_CUTOFF

# Le (2026) Table 1 targets (from the phase instructions).
LE = {
    "Politics": {"markets": 6609, "trades": 4_900_000},
    "Sports": {"markets": 55637, "trades": 43_200_000},
    "TOTAL": {"markets": 210608, "trades": 64_700_000, "base_rate": 38.1},
}

con = duckdb.connect()
con.sql("SET memory_limit='8GB'")
con.sql("SET temp_directory='E:/pm/tmp'")

# ── Le's `resolved` query, verbatim logic, on deduped tables ──
resolved = con.sql(f"""
    SELECT m.ticker, m.event_ticker, m.result,
           regexp_extract(m.event_ticker, '^([A-Z0-9]+)', 1) AS cat_prefix,
           COUNT(*) AS n_trades
    FROM '{DERIVED}/markets_dedup.parquet' m
    INNER JOIN '{DERIVED}/trades_dedup.parquet' t ON t.ticker = m.ticker
    WHERE m.status = 'finalized' AND m.result IN ('yes', 'no')
          AND t.created_time <= TIMESTAMP '{DATE_CUTOFF}'
    GROUP BY m.ticker, m.event_ticker, m.result, cat_prefix
    HAVING COUNT(*) >= 10
""").df()
resolved["domain"] = resolved["event_ticker"].apply(le_group)

rows = []
for d in DOMAINS:
    sub = resolved[resolved["domain"] == d]
    rows.append(dict(
        domain=d,
        ours_markets=len(sub),
        ours_trades=int(sub["n_trades"].sum()),
        ours_base_rate=round(100.0 * (sub["result"] == "yes").mean(), 1) if len(sub) else None,
    ))
tot = resolved[resolved["domain"].isin(DOMAINS)]
rows.append(dict(
    domain="TOTAL",
    ours_markets=len(tot),
    ours_trades=int(tot["n_trades"].sum()),
    ours_base_rate=round(100.0 * (tot["result"] == "yes").mean(), 1),
))
tbl = pd.DataFrame(rows)

tbl["le_markets"] = tbl["domain"].map(lambda d: LE.get(d, {}).get("markets"))
tbl["le_trades"] = tbl["domain"].map(lambda d: LE.get(d, {}).get("trades"))
tbl["le_base_rate"] = tbl["domain"].map(lambda d: LE.get(d, {}).get("base_rate"))
tbl["delta_pct_markets"] = (100.0 * (tbl["ours_markets"] - tbl["le_markets"]) / tbl["le_markets"]).round(2)
tbl["delta_pct_trades"] = (100.0 * (tbl["ours_trades"] - tbl["le_trades"]) / tbl["le_trades"]).round(2)

out = tbl[["domain", "ours_markets", "le_markets", "delta_pct_markets",
           "ours_trades", "le_trades", "delta_pct_trades",
           "ours_base_rate", "le_base_rate"]]
print("=" * 100)
print(f"RECONCILIATION vs Le (2026) Table 1   [cutoff {DATE_CUTOFF}, our archive 2026-02-05]")
print("=" * 100)
print(out.to_string(index=False))

Path(RESULTS).mkdir(exist_ok=True)
out.to_csv(f"{RESULTS}/table_reconciliation.csv", index=False)
print(f"\nSaved {RESULTS}/table_reconciliation.csv")

# Context: full-archive per-domain stats with NO cutoff (expected to overshoot Le).
print("\nFor context — full archive (no cutoff), resolved markets on deduped tables:")
con.sql(f"""
    SELECT domain,
           COUNT(*) AS n_markets,
           SUM(CASE WHEN result IN ('yes','no') THEN 1 ELSE 0 END) AS n_resolved,
           ROUND(100 * AVG(CASE WHEN result='yes' THEN 1.0
                    WHEN result='no' THEN 0.0 END), 1) AS base_rate_pct
    FROM '{DERIVED}/markets_classified.parquet'
    GROUP BY domain ORDER BY n_markets DESC
""").show()
