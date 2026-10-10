#!/usr/bin/env python3
"""Build "exposure-matched" indexes: all eight arms expose exactly the same number of reports at level<=1.

Motivation (review: the end-to-end quality conclusion is doubly confounded):
  exposure -- DyLo exposes 558 reports at level<=1, the others 453-486, 14.8%-23.2% more;
  length   -- DyLo's median answer has 3164 characters, the others 2220-2696, 17.4%-42.5% longer,
          and the judges' length coefficients beta=+4.76/+5.84 far exceed any method effect
          (all other methods' strengths lie in [-0.65,+1.17]).
After length adjustment the GPT lower bound is only +0.054, and exposure was not adjusted at all.

This script removes **exposure** by construction: each arm's level<=1 reports are randomly down-sampled to
453 (the minimum over the eight arms, from rebuild) with a fixed seed. Reports at level>=2 are kept,
but global_search uses community_level=1, so only the level<=1 part is actually read.

Note that this **removes** reports rather than merging them: random down-sampling keeps each arm's own community structure
and only levels the "size of the retrievable candidate pool". The retrieval budget (reports actually read per question) was already
matched (DyLo 5.101 vs QCA 5.138) and needs no change.

Length matching is done by the answering script (same response_type + word limit + same max_tokens).
"""
import os as _os  # artifact: paths are relative to the repository root
DYLO_ROOT = _os.environ.get('DYLO_ROOT', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
DYLO_MHRAG = _os.environ.get('DYLO_MHRAG', _os.path.join(DYLO_ROOT, 'mhrag'))
DYLO_SCRATCH = _os.environ.get('DYLO_SCRATCH', '/tmp/dylo')
import random
import shutil
from pathlib import Path

import pandas as pd

B = Path(DYLO_MHRAG + "/clean_chain")
ARMS = {"m-dylo": "norecl327_g5", "m-rebuild": "rebuild327_fresh",
        "m-qca": "arm_qca", "m-df-louvain": "arm_df-louvain",
        "m-batch": "arm_batch", "m-dynamo": "arm_dynamo",
        "m-hit-leiden": "arm_hit-leiden", "m-ds-louvain": "arm_ds-louvain"}
SEED = 20260903

n1 = {}
for nm, src in ARMS.items():
    r = pd.read_parquet(B / src / "output/community_reports.parquet")
    n1[nm] = int((r.level <= 1).sum())
TARGET = min(n1.values())
print(f"level<=1 report counts {n1}")
print(f"down-sampling all to {TARGET} reports\n")

for nm, src in ARMS.items():
    d = B / nm
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    shutil.copytree(B / src / "output", d / "output")
    if (B / src / "prompts").exists():
        shutil.copytree(B / src / "prompts", d / "prompts")
    for f in (B / src).glob("*.yaml"):
        shutil.copy(f, d / f.name)
    (d / "cache").mkdir(exist_ok=True)
    r = pd.read_parquet(d / "output/community_reports.parquet")
    top = r.index[r.level <= 1].tolist()
    rng = random.Random(SEED)
    keep = set(rng.sample(top, TARGET))
    drop = [i for i in top if i not in keep]
    r2 = r.drop(index=drop).reset_index(drop=True)
    r2.to_parquet(d / "output/community_reports.parquet")
    print(f"  {nm:<16} {n1[nm]} -> {int((r2.level <= 1).sum())} reports "
          f"(dropped {len(drop)}), total reports {len(r)} -> {len(r2)}")
