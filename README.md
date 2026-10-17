# DyLo artifact

Code and evidence for *DyLo: Pricing LLM Re-Summarization in Incremental GraphRAG Index Maintenance*. This repository preserves the October 10 graph experiments and adds the October 16 maintenance case study, with a source/evidence snapshot assembled on October 17, 2026.

## Start here

The recorded results can be checked without a GPU or any new model requests:

```bash
python3 scripts/validate_artifact.py
# Requires numpy; verifies and unpacks the evidence archives, then recomputes results:
python3 experiments/case_study_20261016/audit.py
# C++17 compiler and Python 3.10+; CPU tests of the separate corrected candidate:
make corrected-test
```

See [case-study instructions](experiments/case_study_20261016/README.md) for dependencies, measured boundaries, result tables and optional generation replay. `CHECKSUMS.json` covers the published files; `provenance.json` records source and packaged hashes. No model weights are included.

## Which version produced which result?

| Material | Location | Status |
|---|---|---|
| Historical graph experiments and GraphRAG evaluation | `harness/`, `graphrag/`, `results/` | Preserved October 10 artifact; [original instructions](docs/LEGACY_ARTIFACT_20261010.md) describe this version only |
| Eight-method Qwen 7B maintenance and fresh-answer case study | `experiments/case_study_20261016/` | Completed observations; **original std4**, eager refresh, membership-based reuse |
| Corrected cost ledger, hierarchy and selector implementation | `experiments/corrected_std/` | Separate candidate with CPU tests; **not** the driver of the measured case study |
| Height-dependent structural/token cost diagnostics | `experiments/height_cost/` | Experimental variant; not promoted to the paper default |
| Qwen 32B protocol and CPU validation | `experiments/qwen32_pending_20261017/` | In progress at the snapshot; no completed performance/quality result asserted |
| Relevant manuscript source excerpts | `paper/experiments-20261017/` | Traceability snapshot, not a standalone compilable paper |

Structural cost compares algorithms under an abstract pricing model. Actual token usage is measured in one concrete case study. Graph-ready maintenance time checks whether savings translate into faster index maintenance. These quantities and versions should not be merged into a single metric. For the corrected candidate, deferred work is tracked with its pending-liability ledger; historical partition-recosting numbers are not a replacement for that accounting.

## Completed case study

All methods receive the same frozen extracted graph, extending 304 to 327 documents. The following values are single observed runs, not estimates over repeated timing trials.

| Method | Graph-ready → queryable (s) | Summary interval (s) | Successful input + output tokens |
|---|---:|---:|---:|
| DyLo | 547.908 | 521.659 | 1,157,226 |
| Batch | 957.892 | 931.337 | 1,588,879 |
| Rebuild | 2,144.711 | 2,116.064 | 2,965,947 |
| HIT-Leiden | 961.148 | 934.910 | 1,601,555 |
| DynaMo | 974.842 | 948.767 | 1,616,103 |
| QCA | 963.145 | 936.653 | 1,558,299 |
| DF-Louvain | 990.786 | 964.746 | 1,628,887 |
| DS-Louvain | 987.023 | 960.912 | 1,653,744 |

DyLo reduces measured maintenance time by 42.8% versus Batch and 74.5% versus Rebuild; successful tokens fall by 27.2% and 61.0%, respectively. The summary interval occupies 95.2–98.7% of the measured interval. Common document extraction, initial index construction, model startup and separately recorded repairs of missing initial reports are excluded. Retry time remains included; failed-attempt tokens are reported separately. This is not raw-document end-to-end ingestion time.

The package includes 480 newly generated answers, 3,360 order-swapped judgments and 28 question-bootstrap confidence intervals. Quality results are descriptive preferences, with different native report exposure across indexes; they do not establish a uniform quality win or a non-inferiority guarantee.

## Historical graph experiments

```bash
make
bash scripts/smoke_test.sh
```

The smoke test downloads its dataset. Baseline dependencies are fetched at pinned revisions through `tools/`; graph streams and third-party builds are not included. Consult the historical instructions for the original switches, dataset pipeline and script/result mapping. New case-study and corrected-candidate settings have their own instructions above.

## Included evidence and licensing

Compressed bundles contain the shared extracted input tables, all maintenance request/usage ledgers, final answers, oriented verdicts, protocols and representative plus failed maintenance API payloads. Repeated contexts, full raw API traffic, vector caches, model weights and generated executables are excluded. The case-study README states the exact scope and replay limitations.

Repository code uses the MIT license in `LICENSE`. Third-party baselines retain their upstream licenses; the redistributed DynaMo/QCA patch sources include the upstream MIT notice in `experiments/case_study_20261016/native/java/`. Dataset and model usage remain subject to their upstream terms.
