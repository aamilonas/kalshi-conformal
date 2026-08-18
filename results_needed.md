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
| `table_robustness.csv` + bin-sensitivity outputs | Robustness §7 | **PLANNED** (Phase 5) |
| `recalibration_table.csv` | Discussion, Conclusion, Appendix D | **PLANNED** (Phase 6) |
| reliability diagrams | Results, Appendix B | **EXISTS** in part: `fig_reliability_M2.pdf` / `.png` (single-split). Per-quarter and per-domain walk-forward diagrams **PLANNED**. |
| coverage-over-time figures | Results, Appendix B | **PLANNED** |
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
| `table_wf_counts.csv` | **UNMAPPED** | Walk-forward cell counts. Appendix A. |
| `wf_reliability_bins.csv` | **UNMAPPED** | Walk-forward reliability bins. Appendix B. |
| `classify_crosscheck_disagreements.csv` | **UNMAPPED** | Classifier cross-check. Appendix C (domain-classifier rules) if anywhere. |
| `notes.md` | not a paper artifact | Parked side observations. Feeds prose, ships nothing. |

## Not in `results/` and not required

Nothing else. Any new file must be added to this table before it can appear in
the paper.
