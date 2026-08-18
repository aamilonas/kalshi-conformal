"""Step 9 — build the forecast dataset (the paper's core object).

For each horizon tau and each resolved market: forecast = price of the LAST
trade strictly before close_time - tau; label = resolution outcome.

Outputs:
  data/derived/forecasts_unfiltered.parquet  — everything
  data/derived/forecasts.parquet             — 0.05<=price<=0.95 AND n_trades_market>=10
"""
import duckdb
import pandas as pd

from paths import DERIVED, TMP

# Horizons in ABSOLUTE HOURS. INTERVAL 'n days' on TIMESTAMPTZ is calendar
# arithmetic in the session timezone: across a DST spring-forward, '7 days'
# subtracts only 167 real hours, which let 85 trades leak past the cutoff
# (caught by tests/test_forecasts.py::test_1_join_correctness). Hour units +
# UTC session make the arithmetic purely absolute.
HORIZONS = {"1h": "1 hour", "6h": "6 hours", "24h": "24 hours",
            "1w": "168 hours", "1mo": "720 hours"}

con = duckdb.connect()
con.sql("SET memory_limit='8GB'")
con.sql(f"SET temp_directory='{TMP}'")
con.sql("SET TimeZone='UTC'")

frames = []
for tau_name, tau_sql in HORIZONS.items():
    df = con.sql(f"""
      WITH resolved AS (
        SELECT ticker, domain, close_time,
               CASE WHEN result='yes' THEN 1 ELSE 0 END AS outcome,
               close_time - INTERVAL '{tau_sql}' AS cutoff
        FROM '{DERIVED}/markets_classified.parquet'
        WHERE result IN ('yes','no') AND close_time IS NOT NULL
      ),
      tcounts AS (
        SELECT ticker, COUNT(*) AS n_trades_market
        FROM '{DERIVED}/trades_dedup.parquet' GROUP BY ticker
      )
      SELECT r.ticker, r.domain, '{tau_name}' AS tau,
             t.yes_price / 100.0 AS price,
             r.outcome, r.close_time, t.created_time AS trade_time,
             tc.n_trades_market, t."count" AS trade_size
      FROM resolved r
      ASOF JOIN '{DERIVED}/trades_dedup.parquet' t
        ON r.ticker = t.ticker AND t.created_time < r.cutoff
      JOIN tcounts tc ON tc.ticker = r.ticker
    """).df()
    print(f"tau={tau_name:>3}: {len(df):,} market-forecasts")
    frames.append(df)

all_f = pd.concat(frames, ignore_index=True)
all_f.to_parquet(f"{DERIVED}/forecasts_unfiltered.parquet", index=False)
print(f"\nforecasts_unfiltered.parquet: {len(all_f):,} rows")

filt = all_f[(all_f["price"] >= 0.05) & (all_f["price"] <= 0.95)
             & (all_f["n_trades_market"] >= 10)]
filt.to_parquet(f"{DERIVED}/forecasts.parquet", index=False)
print(f"forecasts.parquet:            {len(filt):,} rows "
      f"({100*len(filt)/len(all_f):.1f}% kept by primary filters)")

print("\n(domain x tau) grid, filtered set:")
print(filt.pivot_table(index="domain", columns="tau", values="ticker",
                       aggfunc="count", fill_value=0)
      [list(HORIZONS)].to_string())
