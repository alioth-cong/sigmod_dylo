#!/usr/bin/env python3
"""Build four indexes sharing one partition for B2 and compare them with four-criterion sensemaking blind judging.

Motivation: answering specific factual questions with token-F1 from level<=1 top-level summaries is a protocol mismatch -- measured:
global search at level=1 returned GraphRAG's fallback string for 30/30 questions
"I am sorry but I am unable to answer this question given the provided data."
(diagnosis in rerun/logs/qa_diag.log; real answers start only at level=3, F1 .0000->.0279).
GraphRAG's global search is designed for sensemaking, so four-criterion blind judging is used instead,
with mask controls in the same arena so that "can the instrument detect semantic loss" is testable.

All four arms share one partition (the same communities/entities/relationships/text_units);
only the content of community_reports differs -- exactly the single-variable isolation review B2 asked for
("in the arena, partition and refresh policy move together; the deferral variable
was never held fixed on its own"):

  eager      reference, every summary fresh
  deferred   37 communities replaced by their **actual stale summaries** from base304. Community ids
             are not stable across indexes (zero id overlap between the two), so alignment uses the set of member-entity titles
             as a fingerprint: 1147/1465 communities have identical member sets, and of these 37 have summary text
             that really differs -- the complete set of "stale summaries" that actually exist on this chain.
  mask37     randomly blank the same number (37) of summaries. **The same amount** of change as deferred, to distinguish
             "outdated content" from "missing content".
  mask40     randomly blank 40% (553 summaries). A strong positive control: if blind judging cannot even rank it last,
             the evaluation is insensitive to semantic loss and a "no difference" for deferred means nothing.
"""
import os as _os  # artifact: paths are relative to the repository root
DYLO_ROOT = _os.environ.get('DYLO_ROOT', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
DYLO_MHRAG = _os.environ.get('DYLO_MHRAG', _os.path.join(DYLO_ROOT, 'mhrag'))
DYLO_SCRATCH = _os.environ.get('DYLO_SCRATCH', '/tmp/dylo')
import json, random, shutil, sys
from pathlib import Path
import pandas as pd

BASE = Path(DYLO_MHRAG + "/clean_chain")
SRC, OLD = BASE / "norecl327_g5", BASE / "base304"
SEED = 20260902
BLANK = "(summary unavailable)"

def clone(name):
    d = BASE / name
    shutil.rmtree(d, ignore_errors=True); d.mkdir(parents=True)
    shutil.copytree(SRC / "output", d / "output")
    if (SRC / "prompts").exists(): shutil.copytree(SRC / "prompts", d / "prompts")
    for f in SRC.glob("*.yaml"): shutil.copy(f, d / f.name)
    (d / "cache").mkdir(exist_ok=True)
    return d

def rd(p, n): return pd.read_parquet(BASE / p / f"output/{n}.parquet")
A, Ac, Ae = rd("norecl327_g5","community_reports"), rd("norecl327_g5","communities"), rd("norecl327_g5","entities")
B, Bc, Be = rd("base304","community_reports"), rd("base304","communities"), rd("base304","entities")
print(f"reference has {len(A)} reports; old index {len(B)}")

# member-set fingerprint alignment (ids are unstable across indexes; entity titles are stable)
def fp(cdf, names):
    o = {}
    for _, r in cdf.iterrows():
        e = r.get("entity_ids")
        if e is None or len(e) == 0: continue
        o[frozenset(names.get(x, "") for x in e)] = r["community"]
    return o
fa = fp(Ac, dict(zip(Ae["id"], Ae["title"])))
fb = fp(Bc, dict(zip(Be["id"], Be["title"])))
Bm = dict(zip(B["community"], B["full_content"]))
Bs = dict(zip(B["community"], B["summary"])) if "summary" in B else {}
pos = {c: i for i, c in enumerate(A["community"])}
# note that communities and community_reports have different counts (1554 vs 1420):
# some communities have no report, so the old community must also have a report in Bm, otherwise KeyError.
stale = [(pos[fa[k]], fb[k]) for k in (set(fa) & set(fb))
         if fa[k] in pos and fb[k] in Bm
         and A["full_content"].iloc[pos[fa[k]]] != Bm[fb[k]]]
print(f"{len(set(fa)&set(fb))} communities aligned by member set, of which {len(stale)} have genuinely different summaries")

FC, SM = A.columns.get_loc("full_content"), (A.columns.get_loc("summary") if "summary" in A else None)
FD = A.columns.get_loc("findings") if "findings" in A else None

def blank(r, idx):
    for i in idx:
        r.iat[i, FC] = BLANK
        if SM is not None: r.iat[i, SM] = BLANK
        if FD is not None: r.iat[i, FD] = []
    return r

# eager
d = clone("ctrl_eager"); A.to_parquet(d / "output/community_reports.parquet")
print(f"eager  -> {d.name}  unchanged")
# a ladder of mask strengths: checks that four-criterion blind judging is sensitive to semantic loss and monotone in its strength.
# This is the positive control B2 asked for -- the original token-F1 is not monotone here (mask40 .1143 < mask80 .1401),
# so it cannot show that "the evaluation detects semantic loss".
for frac in (0.10, 0.20, 0.40):
    nm = f"ctrl_mask{int(frac*100)}"
    d = clone(nm); r = A.copy()
    rng = random.Random(SEED + int(frac * 100))
    k = int(len(r) * frac)
    blank(r, rng.sample(range(len(r)), k)).to_parquet(d / "output/community_reports.parquet")
    print(f"{nm:<14}-> blanked {k} / {len(r)} ({frac:.0%})")
