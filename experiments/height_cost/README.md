# Height-dependent cost diagnostics

Source and recorded structural-weight/token-correlation diagnostics for an experimental height-dependent variant. This variant has not replaced either historical graph drivers or original std4 in the completed Qwen maintenance case study.

```bash
cd experiments/height_cost/src
g++ -O2 -std=c++17 driver_height.cxx -o dylo_height
g++ -O2 -std=c++17 test_weights.cxx -o test_weights
./test_weights
```

`heightcap.tsv` and `heightstruct.tsv` retain the supplied layer parameters. `height_vs_std4w.patch` records the source delta. `results/` preserves the supplied diagnostic tables and validation metadata; these results should be interpreted with their own parameter settings, rather than merged with the paper's primary method comparison.
