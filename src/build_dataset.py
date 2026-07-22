"""Step 6 — deduped base tables.

markets: keep the latest snapshot per ticker (by _fetched_at).
trades:  keep one row per trade_id (latest _fetched_at if refetched).
"""
import duckdb

DATA = "E:/pm/becker-data/data"
OUT = "E:/pm/kalshi-conformal/data/derived"

con = duckdb.connect()
con.sql("SET memory_limit='8GB'")
con.sql("SET temp_directory='E:/pm/tmp'")

n_m_raw = con.sql(f"SELECT COUNT(*) FROM '{DATA}/kalshi/markets/*.parquet'").fetchone()[0]
print(f"markets raw rows:   {n_m_raw:,}")

con.sql(f"""
COPY (
  SELECT * FROM '{DATA}/kalshi/markets/*.parquet'
  QUALIFY ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY _fetched_at DESC) = 1
) TO '{OUT}/markets_dedup.parquet' (FORMAT PARQUET)
""")
n_m = con.sql(f"SELECT COUNT(*) FROM '{OUT}/markets_dedup.parquet'").fetchone()[0]
print(f"markets deduped:    {n_m:,}  (removed {n_m_raw - n_m:,})")

n_t_raw = con.sql(f"SELECT COUNT(*) FROM '{DATA}/kalshi/trades/*.parquet'").fetchone()[0]
print(f"trades raw rows:    {n_t_raw:,}")

con.sql(f"""
COPY (
  SELECT * FROM '{DATA}/kalshi/trades/*.parquet'
  QUALIFY ROW_NUMBER() OVER (PARTITION BY trade_id ORDER BY _fetched_at DESC) = 1
) TO '{OUT}/trades_dedup.parquet' (FORMAT PARQUET)
""")
n_t = con.sql(f"SELECT COUNT(*) FROM '{OUT}/trades_dedup.parquet'").fetchone()[0]
print(f"trades deduped:     {n_t:,}  (removed {n_t_raw - n_t:,})")
