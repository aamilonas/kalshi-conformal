# results_needed.md — the scope firewall

Transcribed 2026-08-18 from the "Results wiring (file → paper section)" list in
`../ai-writing-guideline.md`, with each entry marked against what `results/`
actually holds today. **Anything not mapped here is out of scope for the paper.**
This file records the design decision; it does not make one. To add a table or
figure to the paper, add it here first.

Status key: **EXISTS** = the file is in `results/` now. **PLANNED** = required by
the wiring list, not yet produced. **UNMAPPED** = in `results/` but the wiring
list gives it no paper slot; Angelo assigns a slot or declares it out of scope.

## Required by the wiring list

| File | Paper section | Status |
|------|---------------|--------|
| `table_reconciliation.csv` | Data §3.4 (the paper's Table 1) | **EXISTS** |
| `table_reproduction.csv` | Data §3.4 | **EXISTS** |
| `fig_reproduction.pdf` | Data §3.4 (fills `[REPRODUCTION FIGURE]`) | **EXISTS** |
| `table_M2_main.csv` | Early Results / sanity reference | **EXISTS** |
| walk-forward long tidy CSV (quarters × domains × methods) | Results §6.1–6.3, Appendix A | **EXISTS** as `walk_forward_long.csv` (Step 15, commit `7ff8cfa`) |
| `table_bootstrap_ci.csv` | Results §6.1–6.3 (significance for every headline delta) | **EXISTS** (Step 17) |
| `table_robustness.csv` + bin-sensitivity outputs | Robustness §7 | **PLANNED** (Phase 5) |
| `recalibration_table.csv` | Discussion, Conclusion, Appendix D | **PLANNED** (Phase 6) |
| reliability diagrams | Results, Appendix B | **EXISTS**: `fig_reliability_M2.pdf` / `.png` (single-split) and `fig_reliability_wf.pdf` / `.png` (walk-forward, Politics/Sports x 24h/1w, pooled over test quarters; Step 16, commit `b821c16`). |
| coverage-over-time figures | Results, Appendix B | **EXISTS** as `fig_H3_coverage_time.pdf` / `.png` (Step 16, commit `b821c16`) |
| `table_H1.csv` + `fig_H1_brier_delta.pdf` / `.png` | Results §6.1 (proper scores) | **EXISTS** (Step 16, commit `b821c16`) |
| `table_H2.csv` + `fig_H2_coverage_by_domain.pdf` / `.png` | Results §6.2 (coverage allocation) | **EXISTS** (Step 16, commit `b821c16`) |
| `table_H3.csv` | Results §6.3 (coverage over time) | **EXISTS** (Step 16, commit `b821c16`) |
| `fig_price_hists.png` | Data §3.2 or Appendix | **EXISTS** |
| `spotcheck.txt` | Data §3.2 or Appendix | **EXISTS** |

## In `results/` but not on the wiring list

| File | Status | Note |
|------|--------|------|
| `table_slopes_ours_5h.csv` | **UNMAPPED** | Our five-horizon market-level slopes. Natural companion to `table_reproduction.csv` in Data §3.4, and `results/notes.md` (Step 12) makes it load-bearing for the Politics trade-level-vs-market-level discrepancy the Discussion has to handle. Recommend mapping to Data §3.4. |
| `table_slopes_domain_time_9bin.csv` | **UNMAPPED** | Le's nine time bins reproduced. Same slot as above. |
| `table_reliability_bins_M2.csv` | **UNMAPPED** | The numeric backing for `fig_reliability_M2`. Appendix B if the diagram ships. |
| `table_binning_ci_M2.csv` | **UNMAPPED** | Clopper–Pearson intervals for histogram binning. Appendix B. |
| `table_M2_robustness_spec_split.csv` | **UNMAPPED** | The original `cal < 2024-01-01` boundary kept as pooled-only robustness after that boundary proved infeasible (`notes.md`, Step 14 finding 1). Belongs in Robustness §7, and the boundary change itself needs a sentence in Methods. |
| `table_wf_counts.csv` | **MAPPED** | Paper body — the sample table. Walk-forward sample sizes per quarter x domain x tau plus the `meets_200_test` flag. Assigned by Angelo 2026-08-18. |
| `wf_reliability_bins.csv` | **MAPPED** | Appendix. Per-quarter walk-forward reliability bins. Assigned by Angelo 2026-08-18. |
| `table_reliability_bins_wf.csv` | **MAPPED** | Appendix. Numeric backing for `fig_reliability_wf`, pooled over test quarters. Assigned by Angelo 2026-08-18. |
| `captions.md` | not a paper artifact | Standalone captions for the Step 16 figures. Feeds the figure environments; ships no numbers of its own. |
| `classify_crosscheck_disagreements.csv` | **UNMAPPED** | Classifier cross-check. Appendix C (domain-classifier rules) if anywhere. |
| `notes.md` | not a paper artifact | Parked side observations. Feeds prose, ships nothing. |

## Not in `results/` and not required

Nothing else. Any new file must be added to this table before it can appear in
the paper.

## Audit log

- 2026-08-18, Step 16 (`b821c16`): added the three H-tables and four figures
  above; the two rows the wiring list had as PLANNED for reliability and
  coverage-over-time are now EXISTS. Three files are left **UNMAPPED** for
  Angelo to slot or decline: `table_reliability_bins_wf.csv`,
  `wf_reliability_bins.csv`, `table_wf_counts.csv`.
- 2026-08-18, gate 2: Angelo assigned all three. `table_wf_counts.csv` goes to
  the **paper body** as the sample table; the two reliability-bin files go to
  the **appendix**. Six files from the pre-Phase-4 backlog remain UNMAPPED and
  still need a slot or a declination: `table_slopes_ours_5h.csv`,
  `table_slopes_domain_time_9bin.csv`, `table_reliability_bins_M2.csv`,
  `table_binning_ci_M2.csv`, `table_M2_robustness_spec_split.csv`,
  `classify_crosscheck_disagreements.csv`.
