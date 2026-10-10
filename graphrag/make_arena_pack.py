#!/usr/bin/env python3
"""Arena-style blind judging: all methods' answers to a question are shown together and the judge gives a full ranking.

Why not pairwise matching: 5 arms make 10 pairs, and 8 md files per pair is 80 conversations; moreover each pair
is randomized and judged independently, so win rates need not be consistent (A beats B and B beats C does not imply A beats C).
A ranking puts all methods on one scale, and every pairwise relation follows from the same set of verdicts,
reducing the number of conversations to 16.

The two blinding safeguards remain:
  1) the arms are randomly permuted independently per question -> the letter-to-arm mapping differs per question and is stored separately in key.json;
  2) two orders are produced (order1/order2), the second with a different random permutation. Only questions judged in both orders count,
     which cancels position bias.

Usage: make_arena_pack.py <arm1> <arm2> ... [--per N] [--out dir]
"""
import os as _os  # artifact: paths are relative to the repository root
DYLO_ROOT = _os.environ.get('DYLO_ROOT', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
DYLO_MHRAG = _os.environ.get('DYLO_MHRAG', _os.path.join(DYLO_ROOT, 'mhrag'))
DYLO_SCRATCH = _os.environ.get('DYLO_SCRATCH', '/tmp/dylo')
import argparse
import json
import random
import sys
from pathlib import Path

TMP = Path(DYLO_SCRATCH + "")
ANS = TMP / "sense_ans"

DIMS = {
    "comprehensiveness": "How much detail does the answer provide to cover all "
                         "aspects and details of the question?",
    "diversity": "How varied and rich is the answer in providing different "
                 "perspectives and insights on the question?",
    "empowerment": "How well does the answer help the reader understand and make "
                   "informed judgements about the topic?",
    "directness": "How specifically and clearly does the answer address the question?",
}

HEAD = """# Ranked answer evaluation ({part})

You are grading answers produced by {n} different retrieval-augmented systems over
the same news corpus. You do not know which system produced which answer, and the
letter-to-system mapping is different for every question.

For EACH question, rank ALL {n} answers from best to worst on each of these criteria:

{crit}

Note on `directness`: it is a **control criterion** and is expected to trade off
against comprehensiveness and diversity. A longer, richer answer is usually LESS
direct. Rank this one independently of overall quality or length.

Output **only** a CSV with a header row and exactly {rows} rows per question
(one per criterion):

```
qid,criterion,ranking
{ex},comprehensiveness,{r1}
{ex},diversity,{r2}
{ex},empowerment,{r1}
{ex},directness,{r2}
```

`ranking` is the answer letters from best to worst, separated by `>`, with no
spaces, e.g. `{r1}`. Every letter must appear exactly once. Ties are not allowed
— break them by your best judgement. Do not add commentary.

---
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("arms", nargs="+")
    ap.add_argument("--per", type=int, default=8)
    ap.add_argument("--out", default=str(TMP / "arena_pack"))
    # take only a range of question ids, so two question sets can be judged separately: the main result and the robustness appendix are judged independently,
    # which allows comparing the two sets' BT results directly instead of inferring from proxies such as the number of unanswerable questions.
    ap.add_argument("--range", default="", help="e.g. 60:120 keeps only q60..q119")
    a = ap.parse_args()

    data = {}
    for arm in a.arms:
        p = ANS / f"{arm}.json"
        if not p.exists():
            sys.exit(f"missing answers: {p}")
        data[arm] = json.load(open(p))
    n = len(a.arms)
    letters = [chr(ord("A") + i) for i in range(n)]

    nq = min(len(v) for v in data.values())
    lo, hi = 0, nq
    if a.range:
        lo, hi = (int(x) for x in a.range.split(":"))
    keep = [k for k in range(lo, min(hi, nq))
            if all(data[arm][k] and data[arm][k]["answer"].strip() for arm in a.arms)]
    print(f"{n} arms, question range [{lo},{hi}), {len(keep)} questions answered by every arm")

    out = Path(a.out)
    if out.exists():
        for f in out.glob("*"):
            f.unlink()
    out.mkdir(parents=True, exist_ok=True)

    rng = random.Random(20260827)
    # order[o][k] = the letter order of this question in this order (letter i -> arms[order[i]])
    orders = []
    for o in range(2):
        m = {}
        for k in keep:
            idx = list(range(n))
            rng.shuffle(idx)
            m[k] = idx
        orders.append(m)

    crit = "\n".join(f"- **{d}**: {t}" for d, t in DIMS.items())
    files = {1: [], 2: []}
    for o in (1, 2):
        for ci in range(0, len(keep), a.per):
            chunk = keep[ci:ci + a.per]
            part = f"order{o}-part{ci // a.per + 1}"
            r1 = ">".join(letters)
            r2 = ">".join(reversed(letters))
            lines = [HEAD.format(part=part, n=n, crit=crit, rows=len(DIMS),
                                 ex=f"q{chunk[0]}", r1=r1, r2=r2)]
            for k in chunk:
                q = data[a.arms[0]][k]["question"]
                lines.append(f"## q{k}\n\n**Question:** {q}\n")
                for li, ai in enumerate(orders[o - 1][k]):
                    txt = data[a.arms[ai]][k]["answer"].strip()
                    lines.append(f"**Answer {letters[li]}:**\n\n{txt}\n")
                lines.append("---\n")
            p = out / f"{part}.md"
            p.write_text("\n".join(lines), encoding="utf-8")
            files[o].append(p.name)
            print(f"  {p.name}: {len(chunk)} questions, {p.stat().st_size/1024:.0f} KB")

    json.dump({"arms": a.arms, "letters": letters, "keep": keep,
               "orders": [{str(k): v for k, v in m.items()} for m in orders]},
              open(out / "key.json", "w"), indent=1)

    (out / "README.md").write_text(f"""# Arena blind judging: {n} methods ranked together

{len(keep)} questions, {n} answers each. The letter-to-method mapping **differs per question** and is stored in key.json,
to be restored after judging.

## How to use

1. Send `order1-part*.md` one by one to the judge model (a new conversation for each, to avoid carry-over),
   and save its CSV output under the same name with `.csv`.
2. Do the same for `order2-part*.md`. The second order uses a different random permutation; only questions judged in both orders count.
3. Put all CSV files back into this directory and run:

       python3 graphrag/tally_arena_pack.py {out}

   It prints each method's average rank per criterion and the **win rate between every pair** --
   all from the same verdicts and hence mutually consistent. Two diagnostics are also reported:
   - length preference: correlation between rank and answer length. If directness moves with the other three criteria, the judge is only rating length.
   - inter-criterion agreement: if the four rankings are nearly identical, the protocol has collapsed into a single criterion.

## Methods
{chr(10).join('- ' + x for x in a.arms)}
""", encoding="utf-8")
    print(f"\narena judge pack: {out}  ({len(files[1])+len(files[2])} md files)")


if __name__ == "__main__":
    main()
