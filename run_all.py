"""Step 20 — one command regenerates every file in `results/`.

    py run_all.py              full pipeline
    py run_all.py --fast       smoke test: skips the bootstrap
    py run_all.py --list       show the stages and exit

Stage order matters: `walk_forward` writes `wf_predictions.parquet`, which
`bootstrap` consumes, and `hypothesis_tables`, `bootstrap` and `robustness`
all read `walk_forward_long.csv`.

Inputs are `data/derived/forecasts.parquet` and `forecasts_unfiltered.parquet`
— nothing else is required, with one documented exception. `reproduce_le`
also needs `le_time_bins.parquet`, a Phase 1 artifact built from the 3.2 GB
raw trade table, and the vendored Le reference repo. Neither travels with a
clone, so that stage SKIPS rather than fails when they are absent, and the
summary says so out loud. A skipped stage is never counted as a pass.

`--fast` skips the bootstrap only. Venn-Abers needs no subsampling here: the
Phase 3 implementation fits once per unique test price (at most ~91 cent
values), not once per row, so it is already cheap.

If the sibling reference repos live somewhere other than the parent of this
repo — which is the case when testing from a clone in a scratch directory —
point `KC_PM_ROOT` at the real `pm` directory.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
from paths import DERIVED, LE_MATRIX, RESULTS      # noqa: E402

# (name, script, needs, fast_skip)
STAGES = [
    ("reproduce_le", "reproduce_le.py",
     [f"{DERIVED}/le_time_bins.parquet", LE_MATRIX], False),
    ("run_single_split", "run_single_split.py",
     [f"{DERIVED}/forecasts.parquet"], False),
    ("walk_forward", "walk_forward.py",
     [f"{DERIVED}/forecasts.parquet"], False),
    ("hypothesis_tables", "hypothesis_tables.py",
     [f"{RESULTS}/walk_forward_long.csv"], False),
    ("bootstrap", "bootstrap.py",
     [f"{DERIVED}/wf_predictions.parquet"], True),
    ("robustness", "robustness.py",
     [f"{DERIVED}/forecasts_unfiltered.parquet"], False),
    ("build_recal_table", "build_recal_table.py",
     [f"{DERIVED}/forecasts.parquet"], False),
]


def missing(paths):
    return [p for p in paths if not Path(p).exists()]


def run(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fast", action="store_true",
                    help="smoke test: skip the bootstrap")
    ap.add_argument("--list", action="store_true", help="list stages, exit")
    ap.add_argument("--only", default=None,
                    help="comma-separated stage names to run")
    args = ap.parse_args(argv)

    if args.list:
        for name, script, needs, fs in STAGES:
            print(f"{name:<20} src/{script:<24}"
                  f"{'  [--fast skips]' if fs else ''}")
        return 0

    wanted = set(args.only.split(",")) if args.only else None
    Path(RESULTS).mkdir(parents=True, exist_ok=True)
    results, t_all = [], time.time()
    for name, script, needs, fast_skip in STAGES:
        if wanted is not None and name not in wanted:
            continue
        if args.fast and fast_skip:
            results.append((name, "SKIP", 0.0, "--fast"))
            print(f"[skip] {name} (--fast)")
            continue
        gone = missing(needs)
        if gone:
            results.append((name, "SKIP", 0.0,
                            f"missing {Path(gone[0]).name}"))
            print(f"[skip] {name}: missing {', '.join(Path(g).name for g in gone)}")
            continue
        print(f"\n{'=' * 70}\n[run ] {name}\n{'=' * 70}", flush=True)
        t0 = time.time()
        rc = subprocess.run([sys.executable, str(SRC / script)],
                            cwd=str(ROOT)).returncode
        dt = time.time() - t0
        results.append((name, "OK" if rc == 0 else "FAIL", dt, ""))
        if rc != 0:
            print(f"\n[FAIL] {name} exited {rc}; stopping.")
            break

    print(f"\n{'=' * 70}\nSUMMARY ({time.time() - t_all:.1f}s total)\n"
          f"{'=' * 70}")
    for name, status, dt, why in results:
        note = f"  ({why})" if why else ""
        print(f"  {status:<5} {name:<20} {dt:6.1f}s{note}")
    n_fail = sum(1 for _, s, _, _ in results if s == "FAIL")
    n_skip = sum(1 for _, s, _, _ in results if s == "SKIP")
    print(f"\n  {sum(1 for _, s, _, _ in results if s == 'OK')} ok, "
          f"{n_skip} skipped, {n_fail} failed")
    if n_skip:
        print("  NOTE: skipped stages did not run. This is not a pass.")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(run())
