#!/usr/bin/env python3
"""Package two arms' answers as blind-judging material ready to give to an external LLM judge.

Why judge packs: this machine has no external API key, and the local Qwen2.5-7B cannot do GraphRAG's
four-criterion pairwise judging -- measured: directness went to the longer answer in all 11 decided
cases and the four criteria agreed per question 75% of the time, i.e. the four criteria collapsed into "which is longer". The GraphRAG paper used
GPT-4, so the judge must be replaced.

The two blinding safeguards:
  1) per question, a random draw decides which arm is Answer A -- arm names never appear in the files shown to the judge,
     and the mapping is stored separately in key.json, untouched by humans;
  2) two versions are produced (order1 / order2 swap A and B) -- a win counts only if both agree,
     otherwise it is a tie, which cancels the LLM's position preference. This matches GraphRAG's original protocol.

Usage: make_judge_pack.py <arm A> <arm B> [questions per file]
"""
import os as _os  # artifact: paths are relative to the repository root
DYLO_ROOT = _os.environ.get('DYLO_ROOT', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
DYLO_MHRAG = _os.environ.get('DYLO_MHRAG', _os.path.join(DYLO_ROOT, 'mhrag'))
DYLO_SCRATCH = _os.environ.get('DYLO_SCRATCH', '/tmp/dylo')
import json
import os
import random
import sys
from pathlib import Path

TMP = Path(DYLO_SCRATCH + "")
ANS = TMP / "sense_ans"
OUT = TMP / "judge_pack"

DIMS = {
    "comprehensiveness": "How much detail does the answer provide to cover all "
                         "aspects and details of the question?",
    "diversity": "How varied and rich is the answer in providing different "
                 "perspectives and insights on the question?",
    "empowerment": "How well does the answer help the reader understand and make "
                   "informed judgements about the topic?",
    "directness": "How specifically and clearly does the answer address the question?",
}

HEAD = """# Pairwise answer evaluation ({part})

You are grading answers produced by two different retrieval-augmented systems over
the same news corpus. You do not know which system produced which answer.

For EACH question below, compare Answer A and Answer B on the four criteria:

{crit}

Note on `directness`: it is a **control criterion** and is expected to trade off
against comprehensiveness and diversity. A longer, richer answer is usually LESS
direct. Do not let overall answer quality or length decide this one.

Output **only** a CSV with a header row and exactly one row per question:

```
qid,comprehensiveness,diversity,empowerment,directness
{ex1},A,B,A,B
{ex2},TIE,A,A,TIE
```

Each cell must be exactly `A`, `B`, or `TIE`. Do not add commentary.

---
"""


def main(na: str, nb: str, per: int = 15) -> None:
    A = json.load(open(ANS / f"{na}.json"))
    B = json.load(open(ANS / f"{nb}.json"))
    keep = [k for k in range(len(A))
            if A[k] and B[k] and A[k]["answer"].strip() and B[k]["answer"].strip()]
    print(f"{na} vs {nb}: {len(A)} questions, {len(keep)} answered by both")

    rng = random.Random(20260826)
    # side[k] = True means Answer A of this question is na
    side = {k: rng.random() < 0.5 for k in keep}

    OUT.mkdir(parents=True, exist_ok=True)
    for f in OUT.glob("*"):
        f.unlink()

    crit = "\n".join(f"- **{d}**: {t}" for d, t in DIMS.items())

    def write_part(order: int) -> list[str]:
        files = []
        for ci in range(0, len(keep), per):
            chunk = keep[ci:ci + per]
            part = f"order{order}-part{ci // per + 1}"
            lines = [HEAD.format(part=part, crit=crit,
                                 ex1=f"q{chunk[0]}", ex2=f"q{chunk[1] if len(chunk) > 1 else chunk[0]}")]
            for k in chunk:
                first_is_na = side[k] if order == 1 else not side[k]
                ta = A[k]["answer"] if first_is_na else B[k]["answer"]
                tb = B[k]["answer"] if first_is_na else A[k]["answer"]
                lines.append(f"## q{k}\n\n**Question:** {A[k]['question']}\n\n"
                             f"**Answer A:**\n\n{ta.strip()}\n\n"
                             f"**Answer B:**\n\n{tb.strip()}\n\n---\n")
            p = OUT / f"{part}.md"
            p.write_text("\n".join(lines), encoding="utf-8")
            files.append(p.name)
            print(f"  {p.name}: {len(chunk)} questions, {p.stat().st_size/1024:.0f} KB")
        return files

    f1, f2 = write_part(1), write_part(2)

    json.dump({"a_arm": na, "b_arm": nb, "keep": keep,
               "order1_A_is_a_arm": {str(k): bool(side[k]) for k in keep}},
              open(OUT / "key.json", "w"), indent=1)

    (OUT / "README.md").write_text(f"""# Judge pack: {na} vs {nb}

{len(keep)} questions (those with non-empty answers on both sides). Arm names do not appear in order*.md;
the mapping is in `key.json`, to be restored after judging.

## How to use

1. Send the order1 files starting from `{f1[0]}` one by one to GPT-4 (a new conversation for each, to avoid carry-over),
   and save its CSV output under the same name, e.g. `order1-part1.csv`.
2. Repeat for the order2 files. order2 is the same questions with A and B swapped -- a win counts only if both
   verdicts agree, otherwise it is a tie, which cancels position preference.
3. Put all CSV files in this directory and run:

   python3 graphrag/tally_judge_pack.py

   It restores arm names, applies the two-order agreement rule, prints win rates per criterion, and also reports
   two diagnostics, "length preference" and "inter-criterion agreement" -- to confirm the new judge really distinguishes criteria
   rather than collapsing the four into one as the local 7B did.

## Files

- order1: {', '.join(f1)}
- order2: {', '.join(f2)}
- key.json: question id -> whether Answer A of order1 is {na}
""", encoding="utf-8")
    print(f"\njudge pack generated: {OUT}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 15)
