"""Step 7 — Domain classification, ported exactly from Le (2026).

`le_classify.py` is a verbatim copy of `src/classify.py` from
namanhzz/prediction-market-calibration @ 143ca8739bbbd0663da2e9d50ee8012199110ec0.
We wrap it rather than reimplement it so our domain labels are identical to
the ones behind Le's Table 1.

Le's method: cat_prefix = leading [A-Z0-9]+ of event_ticker ('independent' if
empty/null), then get_group() does ordered case-insensitive SUBSTRING matching
over 571 patterns. Analysis domains are the 6 in DOMAINS; everything else
(Esports, Science/Tech, World Events, Media, Other) collapses to "Other".

Run as a script to classify markets_dedup.parquet -> markets_classified.parquet
and cross-check labels against Becker's get_group() on the same prefixes.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from le_classify import CATEGORY_SQL, get_group  # noqa: E402
from paths import (BECKER_CATEGORIES, DERIVED, RESULTS,  # noqa: E402
                   TMP)

DOMAINS = ["Sports", "Crypto", "Politics", "Finance", "Weather", "Entertainment"]

_PREFIX_RE = re.compile(r"^([A-Z0-9]+)")


def extract_cat_prefix(event_ticker: str | None) -> str:
    """Python mirror of Le's CATEGORY_SQL prefix extraction."""
    if not event_ticker:
        return "independent"
    m = _PREFIX_RE.match(event_ticker)
    return m.group(1) if m else "independent"


def classify(ticker: str, event_ticker: str) -> str:
    """Returns one of: Sports, Crypto, Politics, Finance, Weather, Entertainment, Other.

    `ticker` is accepted for interface compatibility but Le's scheme only uses
    event_ticker.
    """
    group = get_group(extract_cat_prefix(event_ticker))
    return group if group in DOMAINS else "Other"


def le_group(event_ticker: str | None) -> str:
    """Le's raw group label (11 groups) before collapsing to the 7-way domain."""
    return get_group(extract_cat_prefix(event_ticker))


def main():
    import duckdb
    import importlib.util
    import pandas as pd

    derived = DERIVED
    con = duckdb.connect()
    con.sql("SET memory_limit='8GB'")
    con.sql(f"SET temp_directory='{TMP}'")

    # Classify on distinct prefixes (cheap), then join back in SQL.
    prefixes = con.sql(f"""
        SELECT DISTINCT {CATEGORY_SQL} AS cat_prefix
        FROM '{derived}/markets_dedup.parquet'
    """).df()
    prefixes["le_group"] = prefixes["cat_prefix"].apply(get_group)
    prefixes["domain"] = prefixes["le_group"].where(
        prefixes["le_group"].isin(DOMAINS), "Other"
    )
    print(f"{len(prefixes)} distinct cat_prefixes")

    con.register("prefix_map", prefixes)
    con.sql(f"""
        COPY (
            SELECT m.*, p.cat_prefix, p.le_group, p.domain
            FROM '{derived}/markets_dedup.parquet' m
            JOIN prefix_map p ON ({CATEGORY_SQL.replace('event_ticker', 'm.event_ticker')}) = p.cat_prefix
        ) TO '{derived}/markets_classified.parquet' (FORMAT PARQUET)
    """)

    n_in = con.sql(f"SELECT COUNT(*) FROM '{derived}/markets_dedup.parquet'").fetchone()[0]
    n_out = con.sql(f"SELECT COUNT(*) FROM '{derived}/markets_classified.parquet'").fetchone()[0]
    print(f"markets_dedup: {n_in:,} rows -> markets_classified: {n_out:,} rows")
    assert n_in == n_out, "classification join changed row count — coverage-of-join bug"

    print("\nDomain composition (markets):")
    con.sql(f"""
        SELECT domain, le_group, COUNT(*) AS n
        FROM '{derived}/markets_classified.parquet'
        GROUP BY domain, le_group ORDER BY n DESC
    """).show(max_rows=40)

    # ── Cross-check vs Becker's get_group() on the same prefixes ──
    spec = importlib.util.spec_from_file_location(
        "becker_categories",
        BECKER_CATEGORIES,
    )
    becker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(becker)

    counts = con.sql(f"""
        SELECT {CATEGORY_SQL} AS cat_prefix, COUNT(*) AS n_markets
        FROM '{derived}/markets_dedup.parquet'
        GROUP BY 1
    """).df()
    counts["ours"] = counts["cat_prefix"].apply(get_group)
    counts["becker"] = counts["cat_prefix"].apply(becker.get_group)
    counts["agree"] = counts["ours"] == counts["becker"]

    total = counts["n_markets"].sum()
    agree = counts.loc[counts["agree"], "n_markets"].sum()
    print(f"\nBecker cross-check: {agree:,}/{total:,} markets agree ({100*agree/total:.2f}%)")

    dis = counts[~counts["agree"]].sort_values("n_markets", ascending=False)
    if len(dis):
        print("Top disagreements (markets weighted):")
        print(dis.head(25).to_string(index=False))
    per_domain = (
        counts.groupby("ours")
        .apply(lambda g: 100 * g.loc[~g["agree"], "n_markets"].sum() / g["n_markets"].sum(), include_groups=False)
        .round(2)
    )
    print("\nDisagreement % by our domain label:")
    print(per_domain.to_string())

    dis.to_csv(f"{RESULTS}/classify_crosscheck_disagreements.csv", index=False)
    print("\nSaved results/classify_crosscheck_disagreements.csv")


if __name__ == "__main__":
    main()
