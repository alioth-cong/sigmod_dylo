#!/usr/bin/env python3
"""Collect the CSV files judged by the external judge, restore arm names, and compute win rates per criterion.

The two-order agreement rule is the same as GraphRAG's original protocol: a win counts only if order1 and order2 point to the same arm,
otherwise it is a tie. Two diagnostics verify that the new judge really distinguishes criteria:
  - length preference: share of verdicts favoring the longer answer. If directness is also close to 100%, it does not
    work as a control criterion (exactly what happened with the local 7B).
  - per-question inter-criterion agreement: if the four criteria nearly always agree, the judge only rates "which is better overall".
"""
import os as _os  # artifact: paths are relative to the repository root
DYLO_ROOT = _os.environ.get('DYLO_ROOT', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
DYLO_MHRAG = _os.environ.get('DYLO_MHRAG', _os.path.join(DYLO_ROOT, 'mhrag'))
DYLO_SCRATCH = _os.environ.get('DYLO_SCRATCH', '/tmp/dylo')
import csv
import itertools
import json
import sys
from collections import Counter
from pathlib import Path

TMP = Path(DYLO_SCRATCH + "")
PACK = Path(sys.argv[1]) if len(sys.argv) > 1 else TMP / "judge_pack"
DIMS = ["comprehensiveness", "diversity", "empowerment", "directness"]


def load(order: int) -> dict[int, dict[str, str]]:
    out: dict[int, dict[str, str]] = {}
    files = sorted(PACK.glob(f"order{order}-part*.csv"))
    if not files:
        sys.exit(f"no CSV found for order{order} ({PACK}/order{order}-part*.csv)")
    for f in files:
        for row in csv.DictReader(open(f, encoding="utf-8-sig")):
            q = (row.get("qid") or "").strip()
            if not q.startswith("q"):
                continue
            k = int(q[1:])
            out[k] = {d: (row.get(d) or "TIE").strip().upper()[:3] for d in DIMS}
    print(f"order{order}: {len(files)} CSV files, {len(out)} questions")
    return out


def main() -> None:
    key = json.load(open(PACK / "key.json"))
    na, nb = key["a_arm"], key["b_arm"]
    side = {int(k): v for k, v in key["order1_A_is_a_arm"].items()}
    o1, o2 = load(1), load(2)
    keep = sorted(set(o1) & set(o2) & set(side))
    print(f"questions judged in both orders: {len(keep)}\n")

    ans = {n: json.load(open(TMP / "sense_ans" / f"{n}.json")) for n in (na, nb)}

    verdicts: dict[str, dict[int, str]] = {d: {} for d in DIMS}
    for d in DIMS:
        for k in keep:
            # order1: A is na iff side[k]; order2 is the opposite
            def arm(letter: str, first_is_na: bool) -> str:
                if letter == "TIE":
                    return "TIE"
                return (na if first_is_na else nb) if letter == "A" \
                    else (nb if first_is_na else na)
            v1 = arm(o1[k][d], side[k])
            v2 = arm(o2[k][d], not side[k])
            verdicts[d][k] = v1 if (v1 == v2 and v1 != "TIE") else "TIE"

    print(f"{'criterion':<20}{na:>16}{nb:>16}{'tie':>6}{nb+' win rate':>12}")
    for d in DIMS:
        c = Counter(verdicts[d].values())
        n = c[na] + c[nb]
        print(f"{d:<20}{c[na]:>16}{c[nb]:>16}{c['TIE']:>6}"
              f"{(c[nb]/n if n else .5):>11.1%}")

    print("\nDiagnostic 1 -- length preference of verdicts (share favoring the longer answer):")
    for d in DIMS:
        lw = sw = 0
        for k in keep:
            v = verdicts[d][k]
            la, lb = len(ans[na][k]["answer"]), len(ans[nb][k]["answer"])
            if v == "TIE" or la == lb:
                continue
            longer = na if la > lb else nb
            lw += (v == longer); sw += (v != longer)
        n = lw + sw
        print(f"  {d:<20} longer wins {lw:>3} / shorter wins {sw:>3} = {(lw/n if n else .5):>6.1%}")
    print("  (if directness is also close to 100%, it does not work as a control criterion)")

    print("\nDiagnostic 2 -- per-question inter-criterion agreement:")
    for d1, d2 in itertools.combinations(DIMS, 2):
        s = sum(1 for k in keep if verdicts[d1][k] == verdicts[d2][k])
        print(f"  {d1[:13]:<14} vs {d2[:13]:<14} {s}/{len(keep)} = {s/len(keep):.1%}")
    allsame = sum(1 for k in keep if len({verdicts[d][k] for d in DIMS}) == 1)
    print(f"  all four criteria identical: {allsame}/{len(keep)} = {allsame/len(keep):.1%}")

    out = PACK / "result.json"
    json.dump({"a": na, "b": nb, "keep": keep,
               "verdicts": {d: {str(k): v for k, v in verdicts[d].items()} for d in DIMS}},
              open(out, "w"), ensure_ascii=False, indent=1)
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
