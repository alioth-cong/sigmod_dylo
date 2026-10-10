#!/usr/bin/env python
"""Pairwise judging v2: judge only questions answered by both sides, and store per-question, per-criterion verdicts.

The problem with v1: an empty answer on one side unconditionally awarded the win to the other (`if not a.strip(): return other`).
On MultiHop-RAG, base had 14 empty answers, so these 14 questions were handed to norecl on all four criteria,
inflating the win rate from ~64% to ~85% and making even the "control criterion" directness win across the board -- whereas
in the GraphRAG paper directness naturally opposes comprehensiveness/diversity, so winning all of them
is itself a sign of failed judging. An empty answer is a service failure, not answer quality, and must be excluded rather than scored.

Storing per-question verdicts has another use: testing whether the judge really discriminates criteria. If the four criteria
are judged almost identically per question, it is just picking "the better-looking answer overall", and the four scores are not independent.

Usage: judge_v2.py <arm A> <arm B> [output json]
"""
import os as _os  # artifact: paths are relative to the repository root
DYLO_ROOT = _os.environ.get('DYLO_ROOT', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
DYLO_MHRAG = _os.environ.get('DYLO_MHRAG', _os.path.join(DYLO_ROOT, 'mhrag'))
DYLO_SCRATCH = _os.environ.get('DYLO_SCRATCH', '/tmp/dylo')
import collections
import itertools
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

TMP = Path(DYLO_SCRATCH + "")
ANS = TMP / "sense_ans"
# The judge can be any OpenAI-compatible endpoint. A local 7B model is not adequate for this protocol: measured, directness
# went to the longer answer in all 11 decided cases, the four criteria agreed per question 75% of the time,
# collapsing four criteria into one. The GraphRAG paper used GPT-4.
#   JUDGE_API=https://api.openai.com/v1/chat/completions \
#   JUDGE_MODEL=gpt-4o JUDGE_KEY=sk-... python judge_v2.py mh-rebuild-b1 mh-norecl-b1
API = os.environ.get("JUDGE_API", "http://127.0.0.1:8000/v1/chat/completions")
MODEL = os.environ.get("JUDGE_MODEL", "Qwen/Qwen2.5-7B-Instruct")
KEY = os.environ.get("JUDGE_KEY", "")
CONC = int(os.environ.get("JUDGE_CONC", "8"))

DIMS = {
    "comprehensiveness": "How much detail does the answer provide to cover all "
                         "aspects and details of the question?",
    "diversity": "How varied and rich is the answer in providing different "
                 "perspectives and insights on the question?",
    "empowerment": "How well does the answer help the reader understand and make "
                   "informed judgements about the topic?",
    "directness": "How specifically and clearly does the answer address the question?",
}


def _ask(prompt: str) -> str:
    hdr = {"Authorization": f"Bearer {KEY}"} if KEY else {}
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 8, "temperature": 0}
    for _ in range(3):
        try:
            r = requests.post(API, json=body, headers=hdr, timeout=300)
            return r.json()["choices"][0]["message"]["content"].strip().upper()
        except Exception:  # noqa: BLE001
            continue
    return "TIE"


def main(na: str, nb: str, outp: str) -> None:
    A = json.load(open(ANS / f"{na}.json"))
    B = json.load(open(ANS / f"{nb}.json"))
    keep = [k for k in range(len(A))
            if A[k]["answer"].strip() and B[k]["answer"].strip()]
    drop_a = [k for k in range(len(A)) if not A[k]["answer"].strip()]
    drop_b = [k for k in range(len(B)) if not B[k]["answer"].strip()]
    print(f"{len(A)} questions; {na} empty {len(drop_a)}, {nb} empty {len(drop_b)}; "
          f"{len(keep)} questions actually judged")
    if drop_a:
        print(f"  {na} empty question ids {drop_a}")
    if drop_b:
        print(f"  {nb} empty question ids {drop_b}")

    verdicts: dict[str, dict[int, str]] = {}
    for dim, desc in DIMS.items():
        def pair(k, dim=dim, desc=desc):
            q, a, b = A[k]["question"], A[k]["answer"], B[k]["answer"]

            def ask(x, y):
                p = (f"Question: {q}\n\nAnswer A:\n{x[:3000]}\n\nAnswer B:\n{y[:3000]}\n\n"
                     f"Criterion - {dim}: {desc}\n"
                     f"Which answer is better on this criterion? "
                     f"Reply with exactly one token: A, B, or TIE.")
                o = _ask(p)
                return "A" if o.startswith("A") else ("B" if o.startswith("B") else "TIE")

            p1, p2 = ask(a, b), ask(b, a)   # both orders, to cancel position bias
            if p1 == "A" and p2 == "B":
                return na
            if p1 == "B" and p2 == "A":
                return nb
            return "TIE"

        with ThreadPoolExecutor(CONC) as ex:
            res = list(ex.map(pair, keep))
        verdicts[dim] = dict(zip(keep, res))
        c = collections.Counter(res)
        n = c[na] + c[nb]
        print(f"  {dim:<18} {na} {c[na]:>3} / {nb} {c[nb]:>3} / tie {c['TIE']:>3}"
              f"   {nb} win rate {(c[nb]/n if n else .5):.1%}")

    # criterion discrimination: are the four criteria judged identically per question
    print("\nPer-question agreement between criteria (closer to 100% = the judge does not distinguish criteria):")
    for d1, d2 in itertools.combinations(DIMS, 2):
        same = sum(1 for k in keep if verdicts[d1][k] == verdicts[d2][k])
        print(f"  {d1[:12]:<13} vs {d2[:12]:<13} {same}/{len(keep)} = {same/len(keep):.1%}")
    allsame = sum(1 for k in keep
                  if len({verdicts[d][k] for d in DIMS}) == 1)
    print(f"  all four criteria identical: {allsame}/{len(keep)} = {allsame/len(keep):.1%}")

    json.dump({"a": na, "b": nb, "keep": keep,
               "verdicts": {d: {str(k): v for k, v in verdicts[d].items()} for d in DIMS}},
              open(outp, "w"), ensure_ascii=False, indent=1)
    print(f"\nper-question verdicts saved: {outp}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2],
         sys.argv[3] if len(sys.argv) > 3 else str(TMP / "judge_v2.json"))
