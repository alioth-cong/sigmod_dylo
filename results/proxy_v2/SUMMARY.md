# Structural weights vs. summarizer tokens (proxy v2)

Scripts: `graphrag/proxy_dryrun.py` (tier-A dry run with exact cache-key mapping), `graphrag/proxy_dryrun_all.sh`
(73 indexes), `graphrag/proxy_v2_fit.py` (per-index model comparison), `graphrag/proxy_v2_layers.py` (layers 1-3).
Per-arm summary: `layers_arms.csv`.

## GraphRAG 2.7.2 rules verified on the real pipeline (`max_input_length=4000`, `max_length=2000`, `max_tokens=8192`)
- The contexts of all levels are built before any report exists; a local context is sorted by edge degree and truncated
  to at most 4000 tokens. The sub-report substitution branch never runs (0 exceed flags in 91,559 calls; the longest context is exactly 4000).
  -> the input of an internal node is min(B, raw content of its subtree), independent of its number of children;
  23.1% of internal nodes and 0.9% of leaves are truncated.
- Each entity brings at most one relationship into the context (`groupby(...).agg("first")`), so the number of edges is
  about 0.94 x the number of entities and is collinear with it.
- The prompt template costs about 1,928 tokens per call, 61.7% of all input tokens; input is 64.5% of all tokens.
- Output: 9.3% of calls hit max_tokens=8192 (finish=length) and account for 43% of all output tokens; the median of
  normally finished reports is 1,116 tokens.

## Exact mapping
The cache key, sha256(json.dumps({parameters, response_format, messages})), can be recomputed from the rebuilt prompt.
All 17 full-rebuild indexes match 100%; incremental arms 87.5-100%; 91,048 of 91,559 calls overall (99.4%).
Local count of the rebuilt prompt vs. usage.prompt_tokens: `usage = 1.0713 * prompt_tok - 73.2`, R^2 = .9995.

## Layer 1: node level (calibrated on one full index, transferred to 89,530 calls of the other 72 indexes)
| Model | Leaf R^2 | Internal R^2 | All R^2 | Spearman | MAPE |
|---|---:|---:|---:|---:|---:|
| current w (linear map) | .050 | -1.539 | -.046 | .355 | 20.8% |
| B1 = min(B, a + b * subtree entities) | .892 | .914 | .950 | .959 | 5.2% |
| tier-A dry run | -- | -- | .9995 | -- | 0.4% |

B1 parameters are stable across indexes: min(6222, 2027 + 84.98 n) and min(6239, 2026 + 89.54 n) on two different
full rebuilds. On the two indexes of the proxy-validation table (B1 calibrated on an unrelated index):
retained index, 1,432/1,443 samples: leaf .890 / internal .935 / all .966, Spearman .956, MAPE 4.6% (current weights: all .097);
rebuilt index, 1,518/1,518 samples: leaf .906 / internal .927 / all .957, Spearman .959, MAPE 4.6% (current weights: all .155).

## Layer 2: batch level (63 method x batch points)
Input tokens: current weights R^2 .873 / Spearman .962; B1 R^2 .9993 / Spearman .9989 (median relative error 0.8%).
Input + output: B1 R^2 .998; input + 4 x output: B1 R^2 .994.

## Layer 3: method ranking (Kendall tau over 9-10 methods per batch; ground truth = input tokens)
| Batch | 327 | 465 | 511 | 609 | 609c | Mean |
|---|---:|---:|---:|---:|---:|---:|
| current w | .24 | .96 | .51 | .47 | .06 | .447 |
| + subtree entity count | .42 | .69 | .38 | .64 | .39 | .504 |
| + fixed per-call overhead | 1.00 | 1.00 | .96 | .91 | 1.00 | .973 |
| + cap (= B1) | .96 | 1.00 | .96 | 1.00 | .94 | .971 |

With output included: B1 + mean output gives tau .73-.91 for input+output and .38-.82 for input + 4 x output
(caused by the long output tail). DyLo is the cheapest method under all three ground truths in batches 465/511/609/609c
and third in batch 327 (DF-Leiden and GreMod issue only 48 and 56 calls there).

## Transfer to the SNAP graphs (std4 with deferral, Leiden, first-batch hierarchy)
Under B1 the max/min node-weight ratio shrinks from 557-22,143x to 2.8x, and the fixed overhead accounts for 35% (Facebook),
82% (Cit-HepTh), 82% (Epinions), and 77% (DBLP) of the total tree weight: cost is then dominated by how many summaries are rewritten.

## Layer 4 (std4 main-table configuration, 6 graphs x 3 instances; `driver_std4w.cxx`, `DYLO_WMODEL=b1`)
- Regression: with the switch off, bit-identical to std4 (3 graphs); `INC_TREE=0` is also bit-identical.
- Decisions: under B1 (automatic scale, lambda = 0.1) the selection phase commits **0 rounds** (0-31 with the current weights);
  the parking-priced variant behaves the same. Q differs by at most 2e-5, last-batch singletons and reclamation are almost unchanged,
  and cost differs by <1% (Cit-HepTh B1 cost +11%): in this configuration maintenance is dominated by parking and reclamation,
  and reclamation ignores lambda.
- Re-scoring (`results/rescore_sens_small.csv`; ratio = cheapest baseline / DyLo-Leiden, >1 means DyLo is cheaper):

| Weights | Facebook | Cit-HepTh | Cit-HepPh | Epinions | Twitter | DBLP |
|---|---:|---:|---:|---:|---:|---:|
| count (current, whole tree) | 1.40 | 3.85 | 2.62 | 6.91 | 2.40 | 4.25 |
| alpha, no cap (leaf level) | 1.30 | 3.57 | 2.56 | 5.26 | 2.36 | 4.20 |
| cap4k = B1 (leaf level) | 1.03 | 1.60 | 1.38 | **0.55** | **0.95** | 2.02 |
| cap4k = B1 (whole tree) | **0.66** | 1.80 | 1.57 | **0.35** | **0.72** | 2.32 |
| cap16k (leaf level) | 1.16 | 2.50 | 1.85 | 1.18 | 1.24 | 2.49 |

The fixed overhead alone does not change DyLo's lead; whether the lead holds depends on the cap (GraphRAG truncates
the context of large communities, which makes refreshing them cheap) and on the structural asymmetry between DyLo's whole
tree (including internal nodes) and the leaf-only flat baselines. On Epinions DyLo rewrites 329 summaries per batch
(dominated by reclamation), DF-Louvain 30.
