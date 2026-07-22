"""Step 5 — schema inspection of the raw Becker archive.

Verifies the three known silent killers before anything is built:
1. duplicate market snapshots / duplicate trade_ids
2. timestamp timezone consistency (expect UTC)
3. price units (cents 1-99, not dollars)
"""
import duckdb

DATA = "E:/pm/becker-data/data"

con = duckdb.connect()
con.sql("SET memory_limit='8GB'")
con.sql("SET temp_directory='E:/pm/tmp'")

print("=" * 70)
print("KALSHI MARKETS schema")
print("=" * 70)
con.sql(f"DESCRIBE SELECT * FROM '{DATA}/kalshi/markets/*.parquet'").show(max_rows=50)
con.sql(f"SELECT * FROM '{DATA}/kalshi/markets/*.parquet' LIMIT 3").show(max_width=250)

print("=" * 70)
print("KALSHI TRADES schema")
print("=" * 70)
con.sql(f"DESCRIBE SELECT * FROM '{DATA}/kalshi/trades/*.parquet'").show(max_rows=50)
con.sql(f"SELECT * FROM '{DATA}/kalshi/trades/*.parquet' LIMIT 3").show(max_width=250)

print("=" * 70)
print("Row counts")
print("=" * 70)
con.sql(f"""
    SELECT (SELECT COUNT(*) FROM '{DATA}/kalshi/markets/*.parquet') AS n_market_rows,
           (SELECT COUNT(DISTINCT ticker) FROM '{DATA}/kalshi/markets/*.parquet') AS n_tickers,
           (SELECT COUNT(*) FROM '{DATA}/kalshi/trades/*.parquet') AS n_trade_rows,
           (SELECT COUNT(DISTINCT trade_id) FROM '{DATA}/kalshi/trades/*.parquet') AS n_trade_ids
""").show()

print("Silent killer 1a: duplicate market snapshots (max rows per ticker)")
con.sql(f"""
    SELECT MAX(cnt) AS max_snapshots, AVG(cnt) AS avg_snapshots FROM (
        SELECT ticker, COUNT(*) AS cnt
        FROM '{DATA}/kalshi/markets/*.parquet' GROUP BY ticker)
""").show()

print("Silent killer 1b: duplicate trades")
con.sql(f"""
    SELECT COUNT(*) - COUNT(DISTINCT trade_id) AS dup_trades
    FROM '{DATA}/kalshi/trades/*.parquet'
""").show()

print("Silent killer 3: price units — distribution of yes_price on trades")
con.sql(f"""
    SELECT MIN(yes_price) AS min_p, MAX(yes_price) AS max_p,
           AVG(yes_price) AS mean_p,
           MIN(no_price) AS min_no, MAX(no_price) AS max_no,
           SUM(CASE WHEN yes_price + no_price != 100 THEN 1 ELSE 0 END) AS bad_price_sum
    FROM '{DATA}/kalshi/trades/*.parquet'
""").show()

print("status / result value counts (markets)")
con.sql(f"""
    SELECT status, result, COUNT(*) AS n
    FROM '{DATA}/kalshi/markets/*.parquet'
    GROUP BY status, result ORDER BY n DESC
""").show(max_rows=30)

print("Silent killer 2: timezone spot check — 2024 presidential election (PRES-2024)")
print("AP called the race ~2025-11-06 05:34 ET = 10:34 UTC. If created_time is UTC,")
print("the final pre-close trades below should cluster in the early hours of Nov 6 UTC.")
con.sql(f"""
    SELECT ticker, event_ticker, status, result, close_time, open_time
    FROM '{DATA}/kalshi/markets/*.parquet'
    WHERE event_ticker LIKE 'PRES-2024%' ORDER BY ticker LIMIT 10
""").show(max_width=250)
con.sql(f"""
    SELECT ticker, created_time, yes_price, count, taker_side
    FROM '{DATA}/kalshi/trades/*.parquet'
    WHERE ticker = 'PRES-2024-DJT'
    ORDER BY created_time DESC LIMIT 10
""").show(max_width=250)

print("Timestamp dtype/timezone rendering check (raw min/max):")
con.sql(f"""
    SELECT MIN(created_time) AS first_trade, MAX(created_time) AS last_trade
    FROM '{DATA}/kalshi/trades/*.parquet'
""").show()
con.sql(f"""
    SELECT MIN(close_time) AS first_close, MAX(close_time) AS max_close
    FROM '{DATA}/kalshi/markets/*.parquet' WHERE close_time IS NOT NULL
""").show()
