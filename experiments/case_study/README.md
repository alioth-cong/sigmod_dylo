# Paper case study: Qwen2.5-32B

The measured update grows the shared extracted graph from 304 to 327 documents. All initial and final reports use Qwen2.5-32B-Instruct. DyLo uses the sole archived standard driver with eager-source refresh; incremental arms retain reports with unchanged community membership. Rebuild regenerates reports. The independent quality judge is local gpt-oss-120b.

## Inspect and audit

```sh
python3 scripts/validate_artifact.py                 # from repository root
python3 experiments/case_study/audit.py             # requires numpy
```

The audit reads compressed bundles directly. It checks successful native token totals and call counts against each trial, sums the recorded timing phases, checks the paper's displayed values, verifies all 480 final answers exist, and recalculates 3,360 oriented judgments into 28 scores and 95% question-bootstrap intervals (10,000 resamples, seed 17). It writes only `verification/audit.json`.

`results/qwen32-all-baselines-20261017/` is the canonical eight-method result set. `results/qwen32-std4-20261017/` contains shared setup and source protocol metadata from the same run, not another algorithm release. The eight-method result set is authoritative for the completed method count. Internal date-bearing module paths preserve the original import layout.

## Bundles

- `inputs-and-plans.tar.gz`: frozen extracted tables, 1,554 complete initial 32B reports, native DyLo stream, and CPU-validated refresh plans.
- `maintenance-<method>.tar.gz`: each accepted trial's result, phase timings, raw model request/response records, request ledgers, native outputs, and final parquet index tables. Vector databases are regenerated from these tables.
- `quality.tar.gz`: question set, rubric, prompts, 480 answers, query traces, raw evaluation requests, verdicts, calibration, and quality summaries.

The two largest archives are stored as numbered `.part001`, `.part002`, and subsequent volumes. The audit and unpack commands read these volumes automatically; their concatenation is byte-identical to the original gzip archive. Archives have neutral ownership and timestamps. Extracted observations are separate from optional fresh replay destinations:

```sh
python3 experiments/case_study/unpack.py
python3 experiments/case_study/unpack.py maintenance-dylo.tar.gz quality.tar.gz
```

The first command prepares `work/qwen32-std4-20261017/data/` and `plans/`; the second creates `observed/`. Extraction refuses to overwrite a different existing file.

## Optional replay

Replay creates new measurements; the supplied observations remain the source of the paper's values. No replay is necessary for the offline audit. Requirements: Python 3.10, GraphRAG 2.7.2, pandas, pyarrow, numpy, httpx, jsonschema, tiktoken, transformers, lancedb, networkx, and the recorded model-serving stack. Native adapters require GCC 11+ and Java 21. Consult `results/*/server_config.json` and protocols for the recorded service parameters.

1. Obtain Qwen2.5-32B-Instruct revision `5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd` and BAAI/bge-small-en-v1.5 locally. Set `QWEN_MODEL` to the local Qwen directory for quality tokenization. Start the Qwen OpenAI-compatible service on port 18720. Start the supplied embedding server with `EMB_PORT=18721 EMB_MODEL=/path/to/local/bge-small-en-v1.5 python3 experiments/case_study/work/qwen32-std4-20261017/embedding_server.py`. Use the model name `Qwen/Qwen2.5-32B-Instruct`, four concurrent summary calls, temperature 0, seed 17, no prefix/response cache, and the frozen generation policies. Model startup is outside the timed interval.
2. Prepare the pinned dependencies with `tools/fetch_baselines.sh`, `tools/build_dynamo.sh`, and `tools/build_puzzlef_driver.sh`. Compile `harness/java/BaselineDriver.java` against the built DynaMo jar into `harness/java/build/` and store its absolute jar classpath in `harness/java/build/.classpath`. Then run `python3 experiments/case_study/build_native.py`. `--standalone` builds only standard DyLo and HIT-Leiden.
3. Unpack inputs. Recreate the omitted initial vector cache using `python3 experiments/case_study/prepare_initial_vectors.py`; this calls only the embedding service on the supplied initial tables. The complete initial reports and setup-complete marker are supplied.
4. Set `E1_GPU_UUID` to the serving GPU UUID, `E1_ALLOWED_PIDS` to comma-separated GPU worker PIDs of your model/embedding services, and `QWEN32_CONCURRENCY=4`. The isolation guard rejects foreign compute processes.
5. Run one arm at a time: `python3 experiments/case_study/replay.py dylo`. Other arm names are `batch`, `rebuild`, `hit-leiden`, `dynamo`, `qca`, `df-louvain`, and `ds-louvain`. Partial timing trials are never silently resumed.

The model policies and native algorithm decisions are preserved. Packaging changes consist of path configuration, removing unused experiment entry points, sharing the standard source, and allowing the local GPU worker identity to be supplied explicitly.

## Accounting and quality

The paper's maintenance interval begins with the extracted graph ready and ends at an embedded, retrieval-checked, published index. It includes native bootstrap, community update, report preparation, generation retries and publication. Common extraction, the shared initial index and evaluation are excluded. Successful update-summary tokens are primary; failed attempts and unknown usage are separate fields. Every arm is one timing run.

Quality uses the first 60 archived questions with each index's complete root cut, 8,000-Qwen-token map-chunk and reduce evidence budgets, and an independent local GPT judge. Native cuts yield different total query work; evaluation usage is recorded separately. Win/tie/loss scores are 100/50/0, averaged over both orders per question. The 95% bootstrap intervals are per-comparison, without multiplicity correction.
