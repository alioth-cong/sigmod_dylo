# DyLo: Cost-Aware Incremental Community Detection for GraphRAG — Artifact

This repository contains the code, experiment scripts, and result files for the
submission *"DyLo: Cost-Aware Incremental Community Detection for GraphRAG"*.
It is anonymized for double-anonymous review.

Every number in the paper can be read from `results/` without re-running anything;
the sections below explain how to regenerate them.

```
harness/cpp/dylo/   DyLo (C++17, header-only, no dependencies) and experiment variants
harness/cpp/        thin drivers that wrap the third-party C++ baselines
harness/java/       driver that wraps the third-party Java baselines
harness/            Python experiment scripts (graph experiments, scoring, figures)
graphrag/           GraphRAG 2.7.2 integration, token accounting, end-to-end evaluation
graphrag/eval_data/ question sets, generated answers, and blind-judging records
tools/              dataset download, third-party fetch/build helpers
scripts/            smoke test
data/               datasets.tsv (sources + checksums); streams are generated here
results/            all result files used in the paper
```

## 1. Requirements

| | |
|---|---|
| Compiler | g++ ≥ 11 (C++17). DyLo itself has no external dependency. |
| Python | 3.10 with `numpy pandas pyarrow scipy matplotlib` (graph experiments) |
| Baselines | JDK 17 + Maven (Java baselines), OpenMP (C++ baselines), `python-igraph leidenalg` |
| End-to-end | `graphrag==2.7.2`, vLLM serving `Qwen/Qwen2.5-7B-Instruct` and `BAAI/bge-small-en-v1.5` |

The graph experiments need only the compiler and Python; a full-stream run of the
largest graph (LiveJournal) needs ~64 GB RAM. The end-to-end experiments need one GPU.

All scripts locate files relative to the repository root. Override with
`DYLO_ROOT=<repo>`; `DYLO_SCRATCH` (default `/tmp/dylo`) holds intermediate outputs and
`DYLO_MHRAG` points to the GraphRAG indexes of Section 7.

## 2. Quick start (≈2 minutes)

```bash
make                         # builds every DyLo driver into harness/cpp/dylo/
bash scripts/smoke_test.sh   # downloads Facebook, builds its stream, runs DyLo, checks the numbers
```

The smoke test runs DyLo-Leiden in the paper's default configuration on the Facebook
stream and checks mean modularity and re-summarization cost against `results/`.

## 3. Data

`tools/download_datasets.sh` downloads the ten public graphs listed in
`data/datasets.tsv` (SNAP and Network Repository), verifies checksums, and normalizes
each to an undirected simple edge list `data/edges/<name>.tsv`. Then

```bash
python3 harness/prep_data.py --name facebook     # -> data/runs/facebook/
```

shuffles the edges with seed 42, keeps 90% as the initial graph, and splits the
remaining 10% into insertion batches of 1,000 edges (`inc_1.txt … inc_K.txt`).
MultiHop-RAG (Section 7) is the public corpus of Tang and Yang (2024).

## 4. Running DyLo

```
harness/cpp/dylo/dylo_driver_std4 <instance> <initial.txt> <inc.txt> <numBatches> <outDir>
        instance ∈ {louvain, slm, leiden}
```

The paper's default configuration (λ = 0.1, source deferral with ε = 0.05, tree DP):

```bash
export DYLO_EMIT_HIER=1 DYLO_LAMBDA=1e-1 DYLO_AUDIT_K=8 DYLO_REBUILD_R=1e18 \
       DYLO_SIGMA_ONLY=1 DYLO_SIGMA=0.2 DYLO_FINAL_AUDIT=1 DYLO_TOPDOWN=0 \
       DYLO_HIER_GAMMA=0.1 DYLO_PRUNE=1 DYLO_PRUNE_MODE=leaf \
       DYLO_DEFER=1 DYLO_DEFER_EPS=0.05 DYLO_LAZY_CERT=1 DYLO_INC_TREE=1 \
       DYLO_LAZY_RECLAIM=1 DYLO_RECLAIM=1 DYLO_FAST_RECLAIM=1
```

Two kinds of switches behave differently:

| Kind | Variables | Semantics |
|---|---|---|
| presence | `RECLAIM FAST_RECLAIM SIGMA_ONLY FINAL_AUDIT DISABLE_BUCKET` | set = on, even `=0`; unset to turn off |
| value | `LAZY_CERT INC_TREE PRUNE DEFER TREE_DP SRC_CHARGE` | `=0` turns off |

Per batch the driver writes `metrics.tsv` (latency, Q, #communities), `resum.tsv`
(the refresh plan and its normalized cost), `selection.tsv` (every selection round:
predicted and realized ΔF), and, unless `DYLO_NO_COM=1`, the hierarchy `com_<k>_L<i>.txt`.

**Cost accounting.** With deferral on, use the `cost_norm` column of `resum.tsv`
(the refresh plan the driver actually executes). Re-deriving cost from consecutive
partitions (`harness/hier_cost.py`) charges a deferred source as soon as its member set
changes and therefore overstates cost; it is kept only to score methods that do not
emit a refresh plan (all baselines).

### Driver variants

All variants are bit-identical to `dylo_driver_std4` under default switches
(verified by comparing per-batch outputs; see `harness/exp2_driver_sweep.py`).

| Binary | Adds |
|---|---|
| `dylo_driver_std4sel` | `DYLO_SEL_MODE=dp\|greedy\|indep` selector ablation; `DYLO_SEL_SHADOW=1` per-round counterfactual log; `DYLO_MARGINAL_W=1` marginal in-batch weights |
| `dylo_driver_seltime` | per-round selection timing (tree DP vs. min-cut) |
| `dylo_driver_eps` | per-batch deferral and drift logs (`defer.tsv`, `drift.tsv`) |
| `dylo_driver_seedrep` | `DYLO_SEED` for seed dispersion |
| `dylo_driver_b3fix` | marginal bucket test switch `DYLO_MARGINAL_GAMMA` |
| `dylo_driver_r2greedy` | `DYLO_GREEDY=1` per-destination greedy |
| `selector_verify` | frozen-instance exactness: enumeration vs. min-cut vs. tree DP |

## 5. Baselines

```bash
bash tools/fetch_baselines.sh        # clones the public implementations at pinned commits
bash tools/build_dynamo.sh && bash tools/build_slm4j.sh && bash harness/java/build.sh
bash tools/build_puzzlef.sh && bash tools/build_puzzlef_driver.sh && bash tools/build_puzzlef_leiden_driver.sh
python3 harness/run.py --method <m> --dataset <name>
#   m ∈ {dynamo, batch, qca, gremod, df-louvain, ds-louvain, df-leiden, leiden-static, ...}
```

Each run writes the per-batch partitions to `data/runs/<name>/out_<m>/`;
`harness/recost_all.py` scores all of them with the structural cost of Definition 2.1.
HIT-Leiden partitions are emitted by `harness/emit_hit_leiden.py`.

## 6. Paper ↔ scripts ↔ results

Script names partly predate the final section numbering; the table is authoritative.

| Paper item | Script | Result file(s) |
|---|---|---|
| Table 3 (main: Q / latency / cost, 10 graphs) | `harness/recost_reclaim_ab.py`, `harness/recost_all.py`, `harness/assemble_cost_table.py` | `results/reclaim_ab.csv`, `results/resumm_recost.csv`, `results/reclaim_ab_livejournal.csv`, `results/resumm_recost_livejournal.csv`, `results/latency.csv` |
| Seed dispersion (appendix) | `harness/seed_rep.py` | `results/seed_rep.json` |
| Exp-2 component ablation | `harness/exp2_full.py` | `results/exp2_full.json`, `results/ablation.csv` |
| Exp-2b exactness and scaling | `harness/cpp/dylo/selector_verify.cxx` | `results/openset_grid.json`, `results/scalability.csv` |
| Exp-2b selection time (DP vs. min-cut) | `harness/defer_seltime.py` | `results/defer_seltime.json` |
| Selector fidelity (Validate / Peel) | `harness/validate_peel_rate.py` | `results/validate_peel_rate.json` |
| Selector ablation: tree DP vs. greedy vs. per-move pricing | `harness/sel_ablation.py`, `harness/sel_figs.py` | `results/sel_ablation.csv`, `results/sel_ablation_bigbatch.csv`, `results/sel_ablation/` |
| Deferral tolerance ε | `harness/eps_full.py` | `results/eps_full_resum.json`, `results/eps005_quick.csv` |
| Exp-3 batch size and ingestion rate | `harness/exp7_batchsize_multi.py` | `results/batchsize_multi.csv`, `results/batchsize.csv` |
| Exp-4 cost rate λ | `harness/exp5_lambda.py` | `results/lambda.csv` |
| Structural weights vs. tokens | `graphrag/proxy_dryrun.py`, `graphrag/proxy_v2_layers.py`, `graphrag/proxy_fidelity.py` | `results/proxy_v2/`, `results/b8_w_vs_tokens.csv`, `results/proxy_fidelity*.csv` |
| Weight-model sensitivity / per-call weights | `harness/rescore_sens.py`, `harness/layer4_run.py`, `harness/cpp/dylo/driver_std4w.cxx` | `results/rescore_sens_small.csv`, `results/layer4_main.csv` |
| Exp-5 end-to-end GraphRAG (tokens, quality) | `graphrag/` (see §7) | `results/tokensave.csv`, `results/quality.csv`, `graphrag/eval_data/` |
| LiveJournal | `harness/lj_run_all.py`, `harness/leiden_static_par.py` | `results/lj_summary.json` |

## 7. End-to-end GraphRAG experiments

`graphrag/dylo_patch.py` replaces GraphRAG's `update_communities` and
`update_community_reports` workflows: the merged graph is re-partitioned incrementally
(DyLo or a baseline, via `graphrag/dylo_cluster.py`), and only communities whose member set
changed get a new report; all other reports are inherited. Extraction, entity resolution,
embedding, and retrieval are unchanged. `graphrag/cost_meter.py` records the context
tokens of every report-generation call before the cache lookup.

`graphrag/proxy_dryrun.py` rebuilds each community's exact report prompt offline with
GraphRAG's own context builder and recomputes the cache key, mapping every community to
the LLM call that produced its report (100% of calls on full builds) without calling the LLM.

Answering and blind judging: `graphrag/sensemaking_eval.py` (answers),
`graphrag/make_arena_pack.py` (anonymized, order-swapped judge packs),
`graphrag/arena_full_ci.py` and `graphrag/arena_bt_length.py` (analysis). Generated answers and
the two judges' verdicts are in `graphrag/eval_data/` (judging protocol notes:
`graphrag/eval_data/judgments/gpt56sol/JUDGING_NOTES.md`).

## 8. License

MIT (see `LICENSE`). Third-party baselines are fetched from their original repositories
and keep their own licenses; they are not redistributed here.
