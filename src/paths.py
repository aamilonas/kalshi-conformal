"""Repo-root-relative path resolution.

Phases 0-3 hardcoded absolute Windows paths (``E:/pm/...``). Those work only
on the machine that built the data. Everything is resolved here instead,
relative to the repository root, so the same code runs on Windows and macOS
off the same external drive.

Layout assumed (unchanged from Phase 1; ``PM`` is the drive's ``pm`` dir):

    PM/
      kalshi-conformal/            <- ROOT (this repo)
        data/derived/              <- DERIVED
        results/                   <- RESULTS
      becker-data/data/            <- BECKER_DATA (raw archive; read-only)
      prediction-market-calibration/   <- LE_REPO (reference only)
      prediction-market-analysis/      <- BECKER_REPO (reference only)
      tmp/                         <- TMP (DuckDB spill)

Override with env vars ``KC_PM_ROOT`` (the ``pm`` directory) or ``KC_TMP``
if the sibling repos live elsewhere.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PM = Path(os.environ.get("KC_PM_ROOT", ROOT.parent))

# DuckDB and pandas both take these as strings; forward slashes on every OS.
DERIVED = (ROOT / "data" / "derived").as_posix()
RESULTS = (ROOT / "results").as_posix()
TMP = Path(os.environ.get("KC_TMP", PM / "tmp")).as_posix()

BECKER_DATA = (PM / "becker-data" / "data").as_posix()
BECKER_REPO = (PM / "prediction-market-analysis").as_posix()
LE_REPO = (PM / "prediction-market-calibration").as_posix()

LE_MATRIX = f"{LE_REPO}/supplementary/calibration_matrix_216.csv"
LE_CLASSIFY = f"{LE_REPO}/src/classify.py"
BECKER_CATEGORIES = (
    f"{BECKER_REPO}/src/analysis/kalshi/util/categories.py")

# DuckDB spills here during the heavy Phase 1 joins; harmless if unused.
Path(TMP).mkdir(parents=True, exist_ok=True)
