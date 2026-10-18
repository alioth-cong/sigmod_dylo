# DyLo artifact

Artifact for **DyLo: Efficient Maintenance of Community Indexes for GraphRAG**, aligned with the manuscript snapshot of October 18, 2026.

This release contains one standard DyLo implementation and the baseline implementations used in the paper. The case study uses **Qwen2.5-32B-Instruct** for reports and answers and an independent **gpt-oss-120b** judge. The release contains eight measured methods, 480 final answers, and 3,360 order-swapped judgments.

## Verify the recorded results

No GPU or model requests are needed:

```sh
python3 scripts/validate_artifact.py
# Python 3.10+ and numpy:
python3 experiments/case_study/audit.py
# Requires pandas and scipy; checks the appendix structural-weight table:
python3 graphrag/audit_structural_weight.py
# C++17 compiler; compilation only:
make -j2
make -j2 ablations
```

The audit independently sums request ledgers, checks timing phase sums and the paper's displayed maintenance values, and recomputes all 28 quality scores and question-bootstrap confidence intervals from verdicts. Observations remain unchanged. See [the experiment map](docs/EXPERIMENT_MAP.md) and [case-study instructions](experiments/case_study/README.md).

## Paper case study

| Method | Graph-ready maintenance (s) | Summary interval (s) | Successful input + output tokens |
|---|---:|---:|---:|
| DyLo | 1926.507 | 1901.056 | 1,083,274 |
| Batch | 3464.828 | 3438.853 | 1,442,150 |
| Rebuild | 7523.545 | 7495.808 | 2,630,185 |
| HIT-Leiden | 3493.105 | 3467.632 | 1,448,906 |
| DynaMo | 3498.679 | 3473.001 | 1,464,195 |
| QCA | 3386.344 | 3360.405 | 1,411,751 |
| DF-Louvain | 3501.809 | 3475.966 | 1,477,994 |
| Delta-screening | 3590.582 | 3565.035 | 1,501,139 |

The full-precision source is `experiments/case_study/results/qwen32-all-baselines-20261017/all_results.json`. Each arm is one measured run. The primary token column follows the paper: successful update-summary input plus output; failed usage is recorded separately. Maintenance wall time includes retries and ends at a queryable index. Common extraction, initial-index construction, and answer evaluation are separate costs.

Structural cost in graph experiments, actual tokens in the case study, and elapsed maintenance time are distinct measurements. Model settings and graph defaults are recorded separately because the case study uses the archived eager-source configuration.

## Single standard implementation

`harness/cpp/dylo/driver_std4.cxx` is the sole standard DyLo source. Louvain, SLM, and Leiden are detector choices within it. `harness/cpp/dylo/ablations/` contains named patches for the paper's selector comparisons, instrumentation, and seed controls. These patches reconstruct their measured source exactly; they are not alternative standard releases. `make ablations` builds their explicit targets.

Baseline dependencies have pinned revisions in `tools/fetch_baselines.sh`. Java adapters differ between graph streams and the case-study warm-start protocol; these are experiment-specific input adapters for the same baseline algorithms. Third-party notices are retained.

## Contents

- `harness/`: graph algorithms, stream preparation, accounting and experiment scripts.
- `results/`: graph-benchmark and structural-weight evidence.
- `graphrag/`: structural-weight calibration tools.
- `experiments/case_study/`: measured 32B pipeline, frozen graph inputs, plans, final index tables, request ledgers, answers, judgments and independent audit.
- `paper/`: current experimental LaTeX excerpts and figures for identifying reported values.
- `CHECKSUMS.json`: SHA-256 of every published file except the checksum file itself.
- `provenance.json`: source and anonymized-publication hashes for copied materials and archive members.

Large graph datasets are downloaded from the public sources in `data/datasets.tsv`. Model weights, generated executables, and redundant vector caches are excluded. The supplied extracted case-study tables and initial 32B reports support replay without re-extracting documents. Dataset and model use remains subject to upstream terms. Repository code is covered by `LICENSE`; baseline licenses remain in force.
