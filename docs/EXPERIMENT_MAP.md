# Manuscript-to-artifact map

The active manuscript excerpts are in `paper/`. Historical section numbers in script names are retained to preserve traceability; use the mapping below.

| Paper item | Implementation / analysis | Recorded evidence |
|---|---|---|
| Exp-1: ten-graph comparison | `harness/seed_rep.py`, `harness/recost_reclaim_ab.py`, `harness/recost_all.py` | `results/seed_rep.json`, `results/reclaim_ab*.csv`, `results/resumm_recost*.csv`, `results/latency.csv`; exact displayed table: `paper/tabmain10ds.tex` |
| Exp-1: modularity along streams | graph stream drivers | `paper/figures/q_trend.png` |
| Exp-2: component ablation | `harness/exp2_full.py` | `results/exp2_full.json`, `results/exp2_contrib_ablation.json` |
| Selector exactness / scaling | `harness/cpp/dylo/selector_verify.cxx` | `results/openset_grid.json`, `results/openset_frontier*.json`; displayed values in `paper/exp2-selector.tex` |
| Deferral and selector timing | `harness/defer_seltime.py`, selection-timing patch | `results/defer_seltime.json`, `paper/exp2deferseltime.tex` |
| Selector fidelity | `harness/validate_peel_rate.py`, `harness/sel_ablation.py` | `results/validate_peel_rate.json`, `results/sel_ablation*` |
| Exp-3: batching | `harness/exp7_batchsize_multi.py` | `results/batchsize*.csv`, `paper/scripts/make_fig_exp3_frontier.py` |
| Exp-4: exchange rate | `harness/exp5_lambda.py` | `results/lambda.csv`, `paper/scripts/make_fig_exp4_row.py` |
| Structural-weight fidelity | `graphrag/proxy_fidelity.py`, `graphrag/audit_structural_weight.py` | `results/structural_weight/norecl327.csv`, `results/structural_weight/rebuild327_fresh.csv` |
| Exp-5: 32B maintenance | `experiments/case_study/replay.py` and its measured modules | `experiments/case_study/results/`, `bundles/maintenance-*.tar.gz` |
| Exp-5: fresh-index quality | `experiments/case_study/work/qwen32-std4-20261017/quality/` | `experiments/case_study/bundles/quality.tar.gz`; recompute with `audit.py` |

Graph experiments use structural weights and source deferral as described in the manuscript. The 32B case study uses the archived eager-source policy and membership-only report reuse. Both use the single archived standard implementation with their recorded configuration. Timing series, seed series and structural-cost series are distinguished by their source filenames; do not replace one series with another.

Figure scripts retain their documented provenance. In particular, the compact exchange-rate and batch-frontier layouts contain values recovered at displayed precision from existing vector figures. They reproduce presentation, not a new experimental measurement. The numerical observations and paper tables are frozen.
