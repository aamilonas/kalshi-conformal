"""Step 7 unit tests — domain classification port of Le (2026).

Two layers:
1. Hand-derived expectations for ~20 known tickers (regression freeze).
   Expectations follow Le's ORDERED SUBSTRING semantics, including its quirks
   (e.g. any prefix containing "SB" hits Sports/Super Bowl before later rules).
2. Exhaustive equivalence: our wrapper must agree with Le's original
   classify.py (in the cloned reference repo) on every prefix we feed it.
"""
import importlib.util

import pytest

from classify_domains import classify, extract_cat_prefix, le_group
from paths import LE_CLASSIFY

# (ticker, event_ticker, expected 7-way domain)
KNOWN = [
    ("PRES-2024-DJT",        "PRES-2024",          "Politics"),
    ("SENATEAZ-24-R",        "SENATEAZ-24",        "Politics"),
    ("TRUMPPARDON-25-X",     "TRUMPPARDON-25",     "Politics"),
    ("GOVSHUT-25",           "GOVSHUT-25",         "Politics"),
    ("MAYORNYC-25-ZM",       "MAYORNYC-25",        "Politics"),
    ("KXNFLGAME-25SEP04-DALPHI-PHI", "KXNFLGAME-25SEP04-DALPHI", "Sports"),
    ("NBAGAME-25-LAL",       "NBAGAME-25",         "Sports"),
    ("MLBGAME-25-NYY",       "MLBGAME-25",         "Sports"),
    ("KXWTAMATCH-25-SWI",    "KXWTAMATCH-25",      "Sports"),
    ("UFCFIGHT-25-JON",      "UFCFIGHT-25",        "Sports"),
    ("BTCD-25JUL2212-T118000", "BTCD-25JUL2212",   "Crypto"),
    ("ETHD-25JUL22-T3600",   "ETHD-25JUL22",       "Crypto"),
    ("KXXRPMAXY-25-T5",      "KXXRPMAXY-25",       "Crypto"),
    # NB: FEDDECISION is Politics in Le's scheme, not Finance — the short
    # pattern "EC" (Electoral College) is listed earlier and substring-matches
    # FEDD"EC"ISION. Le's Table 1 embeds this quirk; we must reproduce it.
    ("FEDDECISION-25SEP",    "FEDDECISION-25SEP",  "Politics"),
    ("FED-25DEC",            "FED-25DEC",          "Finance"),
    ("INXD-25JUL22-T6300",   "INXD-25JUL22",       "Finance"),
    ("TNOTED-25-T4",         "TNOTED-25",          "Finance"),
    ("CPIYOY-25-T3",         "CPIYOY-25",          "Finance"),
    ("HIGHNY-25JUL22-T99",   "HIGHNY-25JUL22",     "Weather"),
    ("RAINNYC-25JUL",        "RAINNYC-25JUL",      "Weather"),
    ("OSCARPIC-26-ANORA",    "OSCARPIC-26",        "Entertainment"),
    ("SPOTIFYD-25JUL22-DRAKE", "SPOTIFYD-25JUL22", "Entertainment"),
    # Groups outside Le's 6 analysis domains collapse to Other:
    ("LLM1-25-GPT5",         "LLM1-25",            "Other"),      # Science/Tech
    ("NOBELPEACE-26-DJT",    "NOBELPEACE-26",      "Other"),      # World Events
    ("MENTION-25JUL22-TARIFF", "MENTION-25JUL22",  "Other"),      # Media
]


@pytest.mark.parametrize("ticker,event_ticker,expected", KNOWN)
def test_known_tickers(ticker, event_ticker, expected):
    assert classify(ticker, event_ticker) == expected


def test_substring_order_quirk():
    # Le's list checks ("SB" -> Sports) long before ("SBADS" -> Entertainment),
    # and matching is substring-based, so Super-Bowl-ad prefixes land in Sports.
    # This freezes that behavior: we must match Le, not "fix" them.
    assert le_group("SBADS-25") == "Sports"


def test_prefix_extraction():
    assert extract_cat_prefix("KXNFLGAME-25SEP04-DALPHI") == "KXNFLGAME"
    assert extract_cat_prefix("PRES-2024") == "PRES"
    assert extract_cat_prefix("") == "independent"
    assert extract_cat_prefix(None) == "independent"
    assert extract_cat_prefix("-lowercase") == "independent"
    assert classify("X", None) == "Other"


def _load_le_original():
    spec = importlib.util.spec_from_file_location(
        "le_original", LE_CLASSIFY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_equivalence_with_le_original():
    """Our vendored copy must agree with Le's repo on every known prefix
    pattern and on every KNOWN case above."""
    le = _load_le_original()
    probes = [p for p, _, _, _ in le.SUBCATEGORY_PATTERNS]
    probes += [extract_cat_prefix(e) for _, e, _ in KNOWN]
    probes += ["independent", "ZZZUNKNOWN", "SBADS", "KXNFLGAME"]
    for p in probes:
        assert le_group(p) == le.get_group(p), f"divergence on prefix {p!r}"
