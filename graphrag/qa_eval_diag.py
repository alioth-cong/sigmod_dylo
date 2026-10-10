#!/usr/bin/env python
"""Evaluate each arm's answer quality through the real GraphRAG retrieval pipeline.

Uses graphrag.api's local_search / global_search, i.e. the official retrieval implementation itself rather than
a rewrite -- the whole point of this experiment is "replace only the incremental community partition with DyLo, keep everything else official".

The API rather than the CLI: the CLI re-imports graphrag and reloads parquet for every question (most of the measured 40 s per question
is startup), whereas the API loads once and runs concurrently.

Scale trade-offs (measured on one GPU, via the API):
  local  3 s/question -> all 636 questions, the main evidence on HotpotQA
  global map-reduce over a whole level, >9 min per question; with dynamic_community_selection
         (relevance-scored descent level by level) about 9 s/question. But HotpotQA asks two-hop entity facts
         with short-span answers, which community summaries drop for corpus-level synthesis; measured, global
         answers "unable to answer" directly -- the question type is inherently mismatched with this path, so HotpotQA
         uses only local, and multi-hop synthesis is evaluated separately on MultiHop-RAG.

Usage: qa_eval.py <arm> <index dir> <local|global> [num questions]
"""
import os as _os  # artifact: paths are relative to the repository root
DYLO_ROOT = _os.environ.get('DYLO_ROOT', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
DYLO_MHRAG = _os.environ.get('DYLO_MHRAG', _os.path.join(DYLO_ROOT, 'mhrag'))
DYLO_SCRATCH = _os.environ.get('DYLO_SCRATCH', '/tmp/dylo')
import asyncio
import collections
import json
import os
import re
import string
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OUT = Path(DYLO_SCRATCH + "/qa")
OUT.mkdir(parents=True, exist_ok=True)
SEED = 20260823
CONC = int(os.environ.get("QA_CONC", "8"))


def _patch_litellm():
    import litellm.types.utils as U
    from litellm.types.llms.openai import ChatCompletionReasoningSummaryTextBlock as B

    U.ChatCompletionReasoningSummaryTextBlock = B
    for n in ("Message", "Delta", "Choices", "StreamingChoices",
              "ModelResponse", "ModelResponseStream"):
        c = getattr(U, n, None)
        if c is not None and hasattr(c, "model_rebuild"):
            try:
                c.model_rebuild()
            except Exception:
                pass


def norm(s: str) -> str:
    s = (s or "").lower()
    s = "".join(c for c in s if c not in set(string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", s).split())


def f1em(pred: str, golds: list[str]) -> tuple[float, int]:
    p = norm(pred)
    bf, be = 0.0, 0
    for g in golds:
        g = norm(g)
        be = max(be, int(p == g))
        pt, gt = p.split(), g.split()
        if not pt or not gt:
            continue
        c = collections.Counter(pt) & collections.Counter(gt)
        ns = sum(c.values())
        if ns:
            pr, rc = ns / len(pt), ns / len(gt)
            bf = max(bf, 2 * pr * rc / (pr + rc))
    return bf, be


def sample(n_local: int, n_global: int):
    """Stratified sampling; all four arms share the same questions so pairs are comparable."""
    import random
    qs = json.load(open("/tmp/qsel.json"))
    tr = [q for q in qs if q["arm"] == "treat"]
    ct = [q for q in qs if q["arm"] == "ctrl"]
    rng = random.Random(SEED)
    rng.shuffle(tr); rng.shuffle(ct)
    n_tr = round(n_local * len(tr) / (len(tr) + len(ct)))
    loc = tr[:n_tr] + ct[:n_local - n_tr]
    n_tr_g = round(n_global * len(tr) / (len(tr) + len(ct)))
    glo = tr[:n_tr_g] + ct[:n_global - n_tr_g]
    return loc, glo


async def main(arm: str, root: str, method: str, n: int) -> None:
    _patch_litellm()
    from graphrag.config.load_config import load_config
    import graphrag.api as api

    root_p = Path(root)
    cfg = load_config(root_p)
    out_dir = root_p / "output"

    def rd(name):
        p = out_dir / f"{name}.parquet"
        return pd.read_parquet(p) if p.exists() else None

    ents, comms = rd("entities"), rd("communities")
    reps, tus, rels = rd("community_reports"), rd("text_units"), rd("relationships")
    print(f"[{arm}] entities {len(ents)}, communities {len(comms)}, reports {len(reps)}", flush=True)

    loc_q, glo_q = sample(636, 40)   # local measured 3 s/question, run on the full set; global is expensive and supporting only
    qs = (loc_q if method == "local" else glo_q)[:n] if n else (
        loc_q if method == "local" else glo_q)
    # global uses a coarser level: it map-reduces every report of the level, and the finest level takes over ten minutes per question
    # diagnostic knob: the original implementation hard-coded level and response_type, and global at level=1
    # has only 11 communities on that level (1300 reports in total); the narrow retrieval surface was the prime suspect for 80% refusals.
    level = int(os.environ["QA_LEVEL"]) if os.environ.get("QA_LEVEL") \
            else (2 if method == "local" else 1)
    RESP = os.environ.get("QA_RESP", "Single sentence")

    sem = asyncio.Semaphore(CONC)
    res = []

    async def one(i, q):
        async with sem:
            t0 = time.time()
            try:
                if method == "local":
                    ans, _ = await api.local_search(
                        config=cfg, entities=ents, communities=comms,
                        community_reports=reps, text_units=tus,
                        relationships=rels, covariates=None,
                        community_level=level, response_type=RESP,
                        query=q["question"])
                else:
                    ans, _ = await api.global_search(
                        config=cfg, entities=ents, communities=comms,
                        community_reports=reps, community_level=level,
                        dynamic_community_selection=(os.environ.get("QA_DYN","0")=="1"),
                        response_type=RESP, query=q["question"])
            except Exception as e:  # a single failed question must not break the whole run
                print(f"  [{i}] failed {type(e).__name__}: {str(e)[:80]}", flush=True)
                ans = ""
            f, e = f1em(str(ans), q["answers"])
            res.append({"i": i, "grp": q["arm"], "q": q["question"],
                        "pred": str(ans)[:800], "f1": f, "em": e,
                        "unk": int(norm(str(ans)) in ("", "unknown")),
                        "sec": round(time.time() - t0, 1)})
            if len(res) % 10 == 0:
                print(f"  progress {len(res)}/{len(qs)}", flush=True)

    t0 = time.time()
    await asyncio.gather(*(one(i, q) for i, q in enumerate(qs)))
    el = time.time() - t0

    f = OUT / f"{arm}_{method}.json"
    json.dump(res, open(f, "w"), ensure_ascii=False)
    import statistics as st
    tr = [r["f1"] for r in res if r["grp"] == "treat"]
    ct = [r["f1"] for r in res if r["grp"] == "ctrl"]
    print(f"[{arm}/{method}] {len(res)} questions, {el/60:.1f} minutes")
    print(f"  F1={st.mean(r['f1'] for r in res):.4f} "
          f"EM={st.mean(r['em'] for r in res):.4f} "
          f"UNK={st.mean(r['unk'] for r in res):.1%} "
          f"treat={st.mean(tr) if tr else 0:.4f} ctrl={st.mean(ct) if ct else 0:.4f}")
    print(f"  -> {f}")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    a, r, m = sys.argv[1], sys.argv[2], sys.argv[3]
    k = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    asyncio.run(main(a, r, m, k))
