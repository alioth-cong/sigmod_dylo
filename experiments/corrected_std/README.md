# Corrected std candidate

This is a separate, unpromoted implementation snapshot. It did not produce the October 16 Qwen case-study timings and should not be substituted for original std4 when citing those results.

```bash
make -j2
make test
```

Requires a C++17 compiler and Python 3.10+. The suite exercises hierarchy/selector correctness, dynamic-programming caches, deferred source state, cost accounting, small streams, report validation and adapter acknowledgments. Adapter tests use mocks; passing them does not establish live-model answer quality.

`src/driver_std_corrected.cxx` is the candidate; `driver_std_reference.cxx` is its control, and `driver_std4.cxx` is retained for comparison. Generated aliases in `bin/` point to the corrected candidate only within this directory; historical `harness/` binaries are unchanged.

`observed/` preserves supplied development diagnostics. The October 17 freshness provenance review found an invalid chronological document mapping in the candidate evaluation input (sorted document IDs were interpreted as corpus positions). Those freshness results are invalid and require rebuilt inputs. They are not evidence of a completed corrected-std quality evaluation. See `observed/freshness_provenance_review_20261017.json`.

Use `score_cost.py` and the emitted executed/pending-liability ledger for deferred-cost analysis. Do not infer this candidate's amortized cost solely from consecutive community partitions. `graphrag_adapter.py` and `report_validation.py` contain the candidate integration and report validation path.
