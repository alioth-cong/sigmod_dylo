#!/usr/bin/env python
"""Generate sensemaking questions following the GraphRAG paper.

GraphRAG ships no QA benchmark with gold answers; its official evaluation gives a short description of the corpus,
has an LLM imagine N potential users with N tasks each and N questions per (user, task), and then uses
an LLM to judge answer quality pairwise. The questions deliberately do not target specific documents; they test corpus-level synthesis --
exactly what community summaries support, and exactly what DyLo's refresh policy can affect.

By contrast, HotpotQA-style two-hop entity questions have short-span answers, and community summaries drop such details
for a global view; measured: global search directly answers "unable to answer".

Usage: gen_sensemaking.py [num users] [tasks per user] [questions per task]
"""
import os as _os  # artifact: paths are relative to the repository root
DYLO_ROOT = _os.environ.get('DYLO_ROOT', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
DYLO_MHRAG = _os.environ.get('DYLO_MHRAG', _os.path.join(DYLO_ROOT, 'mhrag'))
DYLO_SCRATCH = _os.environ.get('DYLO_SCRATCH', '/tmp/dylo')
import json
import os
import re
import sys

import requests

import os
OUT = os.environ.get("GEN_OUT", DYLO_SCRATCH + "/sensemaking.json")
API = "http://127.0.0.1:8000/v1/chat/completions"
MODEL = "Qwen/Qwen2.5-7B-Instruct"

# describe the corpus truthfully: these documents are HotpotQA supporting paragraphs, Wikipedia-style entity introductions
CORPUS_DESC = os.environ.get("GEN_DESC", (
    "A collection of 7,859 short encyclopedic passages about people, creative works, "
    "organizations, places and events - novels and book series, musicians and albums, "
    "films and television, athletes and teams, companies, and geographic locations. "
    "Each passage introduces one subject and its relationships to others."
))


def chat(prompt: str, maxtok: int = 1500) -> str:
    r = requests.post(API, json={
        "model": MODEL, "messages": [{"role": "user", "content": prompt}],
        "max_tokens": maxtok, "temperature": 0.7}, timeout=600)
    return r.json()["choices"][0]["message"]["content"]


def main(n_user: int, n_task: int, n_q: int) -> None:
    p = (f"{CORPUS_DESC}\n\n"
         f"Given this dataset, identify {n_user} distinct potential users who would "
         f"analyse it, and for each user {n_task} distinct tasks they would perform. "
         f"Then for each (user, task) pair, generate {n_q} questions that require "
         f"understanding the ENTIRE dataset rather than looking up a single passage.\n"
         f"Return strict JSON only, no prose:\n"
         f'{{"users":[{{"user":"...","tasks":[{{"task":"...","questions":["...","..."]}}]}}]}}')
    raw = chat(p, maxtok=3000)
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        sys.exit(f"the model returned no JSON:\n{raw[:400]}")
    data = json.loads(m.group(0))

    qs = []
    for u in data.get("users", []):
        for t in u.get("tasks", []):
            for q in t.get("questions", []):
                qs.append({"user": u.get("user", ""), "task": t.get("task", ""),
                           "question": q})
    # de-duplicate: different personas often ask nearly identical questions
    seen, uniq = set(), []
    for q in qs:
        k = q["question"].strip().lower()
        if k in seen:
            continue
        seen.add(k)
        uniq.append(q)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(uniq, open(OUT, "w"), ensure_ascii=False, indent=1)
    print(f"generated {len(qs)} questions, {len(uniq)} after de-duplication -> {OUT}")
    for q in uniq[:4]:
        print(f"  [{q['user'][:26]}] {q['question'][:110]}")


if __name__ == "__main__":
    a = [int(x) for x in sys.argv[1:4]] or [5, 3, 4]
    main(*(a + [5, 3, 4][len(a):]))
