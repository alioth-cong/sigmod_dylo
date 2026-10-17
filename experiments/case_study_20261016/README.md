# October 16 Qwen 7B case study

This directory snapshots the completed eight-method experiment cited by the current manuscript excerpts in `../../paper/experiments-20261017/`. It contains observations, portable copies of the measurement code, frozen extracted inputs, and offline checks. It does not replace the historical graph experiments or use the corrected-std candidate.

## Offline result verification

```bash
python3 unpack.py
python3 audit.py
```

Run from this directory, or invoke each script by its path from the repository root. `audit.py` requires NumPy (the original environment used 1.26.4). It recomputes token counts from all attempt ledgers, checks timing phase sums and the shared graph fingerprint, validates all 480 answers, recomputes 3,360 oriented preference scores, and verifies all 28 bootstrap intervals (10,000 resamples, seed 17). It makes no model requests. Output goes to ignored `verification/`.

`unpack.py` verifies archive SHA-256 and rejects unsafe paths. Input archives populate `work/table5-std4-20261016/data/` and the shared native stream under `data/legacy/`; observations unpack into **`observed/`**, separately from fresh replay outputs.

## Observations and boundaries

- `results/all_results.csv` and `.json`: the eight-method time/token comparison.
- `results/all_phase_times.csv`: individual measured phases. The headline summary interval is `net_update_summary_wall_s`, including a few milliseconds of wrapper overhead beyond the delta-summary and merged-refresh sub-timers. The published summary percentage uses the sum of those two sub-timers.
- `results/evaluation_usage.csv` and `.json`: answer/judge inference usage, separate from maintenance usage.
- `results/quality/quality-full.csv` and `.json`: all 28 preferences and question-bootstrap intervals.
- `results/quality/questions.json`, protocols, rubric and calibration: the fixed 60-question evaluation.
- `observed/work/*/runs/<method>/`: results, coverage/isolation checks, plans, and **all** maintenance attempt ledgers, including initial repair and failed/retried requests.
- `observed/outputs/table5-all-baselines-20261016/quality/`: all 480 final answers, seven complete verdict files, usage ledgers and private arm/order mapping (released for reproducibility, not used to expose identities during judging).

Each arm starts from the same 304-document extracted tables and updates to the same 327-document graph. Primary timing covers table loading, native community update/bootstrap, refresh selection, context construction, delta summaries, merged refresh, embedding/write/publication and residual overhead. Document extraction, initial 304-document construction and model warmup are outside the interval. Missing old reports are repaired and timed separately; primary maintenance subtracts that repair interval. All repair costs are retained in the observations. Therefore this measurement supports graph-ready maintenance acceleration; it does not measure raw-document ingestion end to end.

Primary tokens sum native successful **input plus output** usage for delta and merged refresh. Failed tokens are a separate column (one failure each for HIT-Leiden and DF-Louvain); retry elapsed time remains included. Initial repair and answer/judge usage are separate. There is one timing run per method, on isolated H100 GPUs (first three arms on one GPU, the other five on another); there is no timing confidence interval or repeated-run dispersion claim.

The archived original std4 is eager (`DYLO_DEFER=0`), with lambda 0.1 and hierarchy gamma 5.0. Reuse follows the original membership-only rule: the first old community with the same entity-ID set and an existing report is inherited. It does **not** check unchanged evidence. The corrected-std and stricter evidence-refresh variants are separate work. Every final native community has a report, but community/entity coverage and native root-report exposure differ across methods.

Reports use Qwen2.5-7B-Instruct, concurrency 8, seed 17, a 2,048-token output cap, five concise findings, and a 4,000 cl100k context budget. Response and prefix caches are disabled for generation. Quality is re-evaluated on the new indexes using 60 fixed questions, native C0 root cuts and GraphRAG global map/reduce with an 8,000-token budget; input exposure is not matched across methods. An independent local GPT-OSS-120B judge evaluates both answer orders in four dimensions. Scores average both orders per question, then all questions. The intervals describe these 60 questions for one set of generated answers; they are not an equivalence/non-inferiority test.

## What the bundles contain

`bundles/manifest.json` lists sizes, SHA-256 hashes and file counts. The three input bundles contain the shared base/merged/delta tables and native stream. The maintenance bundle contains complete per-attempt ledgers and timing/result/plan records, plus shortest/median/longest successful payload samples and every failed payload per stage. The quality bundle contains all final answers and all oriented verdicts, full usage ledgers, rubric and mapping. Repeated prompt contexts, full raw generation/judge bodies, final Parquet indexes and vector caches are not included. They can be regenerated; complete raw archives were retained separately on the source machine. This is an audit package with an optional generation replay, not a frozen copy of every final index or API byte.

## CPU build and fresh planning

Use Python 3.10+, GCC 11+ (C++17) and, for the baseline adapters, JDK 21. Original Python package versions included GraphRAG 2.7.2, httpx 0.28.1, numpy 1.26.4, pandas 2.3.3, pyarrow 25.0.1, tiktoken 0.13.0, lancedb 0.37.1, jsonschema 4.26.0, pydantic 2.11.10, networkx 3.4.2 and scipy 1.15.3. Answer generation additionally needs Transformers and an already cached Qwen tokenizer. These describe the source environment, not a fully tested clean-install lockfile.

```bash
# From repository root: builds original std4 and standalone gated drivers.
python3 experiments/case_study_20261016/build_native.py --standalone
python3 experiments/case_study_20261016/reproduce.py --method dylo --plan-only
```

To build all baseline adapters, first fetch/build the pinned dependencies as described in the historical artifact (`tools/fetch_baselines.sh`, Java build helpers and PuzzleF driver helpers), then run `build_native.py` without `--standalone`. Set `JAVA` and `JAVAC` to JDK 21 executables if needed. The DynaMo and DS-Louvain plans use the historical harness binaries; the other baseline plans use the provided gated adapters. `--plan-only` invokes no model service, but requires GraphRAG and its context-builder dependencies. Available methods: `dylo`, `batch`, `rebuild`, `hit-leiden`, `dynamo`, `qca`, `df-louvain`, `ds-louvain`.

## Optional live replay

Start your own compatible Qwen and BGE-small-en-v1.5 services. Scripts do not start or stop model processes. Match the original generation parameters and disable prefix caching on the server. Embedding uses BGE-small-en-v1.5 (512-token truncation). Configure the complete endpoints as needed:

```bash
export QWEN_SUMMARY_URL=http://127.0.0.1:18630/v1/chat/completions
export DYLO_EMBEDDING_URL=http://127.0.0.1:18621/v1/embeddings
export SENSEMAKING_URL=http://127.0.0.1:18630/v1/chat/completions
export SENSEMAKING_JUDGE_URL=http://127.0.0.1:18652/v1/chat/completions
export QWEN_MODEL=Qwen/Qwen2.5-7B-Instruct
# Set these to the actual serving GPU and its allowed compute-process IDs:
export E1_GPU_UUID=GPU-your-serving-uuid
export E1_OWNED_GPU_PIDS=12345
python3 experiments/case_study_20261016/reproduce.py --method dylo --execute
```

The GPU monitor rejects foreign compute processes on the declared GPU. Use one fresh trial per arm; partial runs are deliberately not resumed. `--execute` rebuilds the initial vector cache outside the timer because the original cache is not distributed. Differences in that cache, libraries, server or GPU change publication time; reruns are new observations, not a promised bitwise reproduction of archived timing. Never copy `observed/` into fresh `work/*/runs/` to bypass measurement.

After generating all eight new indexes, the supplied `work/table5-all-baselines-20261016/quality/validate.py`, `generate.py` and `judge.py` provide the original quality pipeline. Install/cache the Qwen tokenizer, run `prepare_quality.py` to seed the fixed questions and rubric into the fresh output directory, then run the following commands from the repository root:

```bash
python3 experiments/case_study_20261016/prepare_quality.py
python3 experiments/case_study_20261016/work/table5-all-baselines-20261016/quality/validate.py
# Repeat for each of the eight methods:
python3 experiments/case_study_20261016/work/table5-all-baselines-20261016/quality/generate.py dylo full
# Repeat for each of the seven baselines:
JUDGE_BASELINE=batch python3 experiments/case_study_20261016/work/table5-all-baselines-20261016/quality/judge.py full
```

Do not copy archived answer/checkpoint/calibration files into this fresh output directory. Original protocols are archived, not auto-promoted as newly completed runs. `audit.py` always checks the archived observations, not newly generated results.

Portable source copies replace source-machine paths and require explicit GPU identity. Original and packaged source hashes are recorded in `../../provenance.json`; algorithm sources and observed measurements are retained without changing experimental conclusions.
