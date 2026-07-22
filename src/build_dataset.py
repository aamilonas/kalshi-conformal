"""Step 6 — deduped base tables.

markets: keep the latest snapshot per ticker (by _fetched_at).
trades:  keep one row per trade_id (latest _fetched_at if refetched).
"""
import os

import duckdb

DATA = "E:/pm/becker-data/data"
OUT = "E:/pm/kalshi-conformal/data/derived"

con = duckdb.connect()
con.sql("SET memory_limit='8GB'")
con.sql("SET temp_directory='E:/pm/tmp'")


def build(name, src_glob, key):
    """Dedup src into OUT/name keeping latest _fetched_at per key.
    Skips if a previous run already produced a complete file (a killed COPY
    leaves a 0-byte/short file that fails the count probe)."""
    dest = f"{OUT}/{name}"
    n_raw = con.sql(f"SELECT COUNT(*) FROM '{src_glob}'").fetchone()[0]
    print(f"{name}: raw rows {n_raw:,}", flush=True)

    if os.path.exists(dest):
        try:
            n_have = con.sql(f"SELECT COUNT(*) FROM '{dest}'").fetchone()[0]
            print(f"{name}: exists with {n_have:,} rows — skipping rebuild", flush=True)
            return
        except Exception:
            print(f"{name}: existing file unreadable (partial write) — rebuilding", flush=True)
            os.remove(dest)

    con.sql(f"""
    COPY (
      SELECT * FROM '{src_glob}'
      QUALIFY ROW_NUMBER() OVER (PARTITION BY {key} ORDER BY _fetched_at DESC) = 1
    ) TO '{dest}' (FORMAT PARQUET)
    """)
    n_out = con.sql(f"SELECT COUNT(*) FROM '{dest}'").fetchone()[0]
    print(f"{name}: deduped {n_out:,}  (removed {n_raw - n_out:,})", flush=True)


build("markets_dedup.parquet", f"{DATA}/kalshi/markets/*.parquet", "ticker")
build("trades_dedup.parquet", f"{DATA}/kalshi/trades/*.parquet", "trade_id")
print("STEP6_DONE", flush=True)
