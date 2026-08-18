"""Step 12 (extraction only) — Le's nine-time-bin trade-level dataset.

Reproduces load_kalshi_trades() from Le's prediction-market-calibration
src/pipeline.py, verbatim in semantics, against OUR deduped tables:

  - resolved markets: status='finalized' AND result IN ('yes','no')
  - trades with created_time <= Le's DATE_CUTOFF and created_time < close_time
  - markets kept only if they have >= 10 such trades (counted BEFORE the
    price filter, exactly like Le's market_counts CTE)
  - final trade rows filtered to 5 <= yes_price <= 95 (cents)
  - time bin on hours_to_close = epoch(close_time - created_time)/3600
    (absolute hours; epoch arithmetic is DST-proof)
  - size bin on t.count: Single=1, Small=2-10, Medium=11-100, Large=101+
  - pre-aggregated by (domain, tbin, sbin, yes_price, is_yes) with
    SUM(count) AS total_contracts, COUNT(*) AS n_trades

This is the ONE permitted read of trades_dedup.parquet in Phase 2-3, kept
isolated here. Output: data/derived/le_time_bins.parquet (gitignored).
Never touches forecasts.parquet. Domain labels come from
markets_classified.parquet, whose `domain` column is Le's get_group()
(equivalence proven by tests/test_classify.py).
"""
import duckdb

from paths import DERIVED, TMP
OUT = f"{DERIVED}/le_time_bins.parquet"

# Le's src/config.py TIME_BINS / SIZE_BINS, verbatim.
TIME_BINS = [
    (0, 1, "0-1h"), (1, 3, "1-3h"), (3, 6, "3-6h"), (6, 12, "6-12h"),
    (12, 24, "12-24h"), (24, 48, "24-48h"), (48, 168, "2d-1w"),
    (168, 720, "1w-1mo"), (720, 1e9, "1mo+"),
]
SIZE_BINS = [(1, 1, "Single"), (2, 10, "Small"), (11, 100, "Medium"),
             (101, int(1e9), "Large")]
BIN_LABELS = [label for _, _, label in TIME_BINS]
SIZE_LABELS = [label for _, _, label in SIZE_BINS]
DOMAINS = ["Sports", "Crypto", "Politics", "Finance", "Weather", "Entertainment"]
DATE_CUTOFF = "2025-12-31 23:59:59+00"  # Le's config.DATE_CUTOFF (UTC)


def time_bin_sql():
    parts = []
    for i, (lo, hi, _) in enumerate(TIME_BINS):
        if hi >= 1e9:
            parts.append(f"WHEN hours_to_close >= {lo} THEN {i}")
        else:
            parts.append(
                f"WHEN hours_to_close >= {lo} AND hours_to_close < {hi} THEN {i}")
    return "CASE " + " ".join(parts) + " ELSE -1 END"


def size_bin_sql():
    parts = []
    for i, (lo, hi, _) in enumerate(SIZE_BINS):
        if hi >= int(1e9):
            parts.append(f"WHEN trade_count >= {lo} THEN {i}")
        else:
            parts.append(f"WHEN trade_count >= {lo} AND trade_count <= {hi} THEN {i}")
    return "CASE " + " ".join(parts) + " ELSE -1 END"


if __name__ == "__main__":
    con = duckdb.connect()
    con.sql("SET memory_limit='8GB'")
    con.sql(f"SET temp_directory='{TMP}'")
    con.sql("SET TimeZone='UTC'")

    tb, sb = time_bin_sql(), size_bin_sql()
    domains_sql = ", ".join(f"'{d}'" for d in DOMAINS)

    con.sql(f"""
    COPY (
      WITH resolved AS (
        SELECT ticker, domain, close_time,
               CASE WHEN result='yes' THEN 1 ELSE 0 END AS is_yes
        FROM '{DERIVED}/markets_classified.parquet'
        WHERE status='finalized' AND result IN ('yes','no')
          AND domain IN ({domains_sql})
      ),
      trade_data AS (
        SELECT t.yes_price, t."count" AS trade_count, m.domain, m.is_yes,
               t.ticker,
               EXTRACT(EPOCH FROM (m.close_time - t.created_time))/3600.0
                 AS hours_to_close
        FROM '{DERIVED}/trades_dedup.parquet' t
        INNER JOIN resolved m ON t.ticker = m.ticker
        WHERE t.created_time <= TIMESTAMPTZ '{DATE_CUTOFF}'
          AND m.close_time > t.created_time
      ),
      market_counts AS (
        SELECT ticker FROM trade_data GROUP BY ticker HAVING COUNT(*) >= 10
      )
      SELECT td.domain, ({tb}) AS tbin,
             ({sb}) AS sbin, td.yes_price, td.is_yes,
             SUM(td.trade_count) AS total_contracts, COUNT(*) AS n_trades
      FROM trade_data td
      INNER JOIN market_counts mc ON td.ticker = mc.ticker
      WHERE td.yes_price BETWEEN 5 AND 95
        AND ({tb}) >= 0 AND ({sb}) >= 0
      GROUP BY td.domain, ({tb}), ({sb}), td.yes_price, td.is_yes
    ) TO '{OUT}' (FORMAT PARQUET)
    """)

    chk = con.sql(f"""
      SELECT domain, SUM(n_trades) AS n_trades, SUM(total_contracts) AS contracts
      FROM '{OUT}' GROUP BY domain ORDER BY n_trades DESC
    """).df()
    print(chk.to_string(index=False))
    print(f"\nTOTAL trades: {int(chk['n_trades'].sum()):,}")
    n_rows = con.sql(f"SELECT COUNT(*) FROM '{OUT}'").fetchone()[0]
    print(f"Aggregated rows written: {n_rows:,}")
