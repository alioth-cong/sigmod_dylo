#!/usr/bin/env python3
"""Collect arena rankings: restore method names, output average ranks and every pairwise win rate.

The two orders (order1/order2) use different random permutations; a pair's win counts only if both orders agree (the same method
precedes the other in both), otherwise it is a tie -- the same position-preference control as the
pairwise protocol.

The two diagnostics as before:
  - length preference: whether the top-ranked answer is always the longest. If directness moves with the other three criteria,
    the judge is only rating length.
  - inter-criterion agreement: if the four rankings are nearly identical, the protocol has collapsed into a single criterion.
"""
import os as _os  # artifact: paths are relative to the repository root
DYLO_ROOT = _os.environ.get('DYLO_ROOT', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
DYLO_MHRAG = _os.environ.get('DYLO_MHRAG', _os.path.join(DYLO_ROOT, 'mhrag'))
DYLO_SCRATCH = _os.environ.get('DYLO_SCRATCH', '/tmp/dylo')
import csv
import itertools
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

TMP = Path(DYLO_SCRATCH + "")
DIMS = ["comprehensiveness", "diversity", "empowerment", "directness"]
# directness is a control criterion: it naturally opposes the other three, since more comprehensive answers are less direct.
# It is therefore **excluded from the total** and reported separately, to falsify the alternative "the answers are just longer".
QUALITY = ["comprehensiveness", "diversity", "empowerment"]


def load_order(pack: Path, o: int, key) -> dict:
    """Return {(qid, dim): [methods from best to worst]}"""
    letters = key["letters"]
    arms = key["arms"]
    omap = {int(k): v for k, v in key["orders"][o - 1].items()}
    out = {}
    files = sorted(pack.glob(f"order{o}-part*.csv"))
    if not files:
        sys.exit(f"no CSV found for order{o} ({pack}/order{o}-part*.csv)")
    for f in files:
        for row in csv.DictReader(open(f, encoding="utf-8-sig")):
            q = (row.get("qid") or "").strip()
            dim = (row.get("criterion") or "").strip().lower()
            rk = (row.get("ranking") or "").strip().upper()
            if not q.startswith("q") or dim not in DIMS or not rk:
                continue
            k = int(q[1:])
            if k not in omap:
                continue
            seq = [x for x in rk.replace(" ", "").split(">") if x]
            if sorted(seq) != sorted(letters):
                continue          # letters missing or duplicated, row dropped
            out[(k, dim)] = [arms[omap[k][letters.index(x)]] for x in seq]
    print(f"order{o}: {len(files)} CSV files, {len(out)} rankings")
    return out


def main() -> None:
    pack = Path(sys.argv[1]) if len(sys.argv) > 1 else TMP / "arena_pack"
    key = json.load(open(pack / "key.json"))
    arms = key["arms"]
    o1, o2 = load_order(pack, 1, key), load_order(pack, 2, key)
    common = sorted(set(o1) & set(o2))
    qs = sorted({k for k, _ in common})
    print(f"{len(common)} (question, criterion) pairs judged in both orders, covering {len(qs)} questions\n")

    ans = {a: json.load(open(TMP / "sense_ans" / f"{a}.json")) for a in arms}

    # average rank (mean over both orders, 1 = best)
    print("Average rank (1 = best)")
    print(f"{'method':<22}" + "".join(f"{d[:13]:>15}" for d in DIMS) + f"{'overall':>10}")
    rank_by = defaultdict(lambda: defaultdict(list))
    for (k, dim) in common:
        for seq in (o1[(k, dim)], o2[(k, dim)]):
            for pos, a in enumerate(seq, 1):
                rank_by[a][dim].append(pos)
    for a in arms:
        row = [statistics.mean(rank_by[a][d]) if rank_by[a][d] else float("nan")
               for d in DIMS]
        print(f"{a:<22}" + "".join(f"{v:>15.2f}" for v in row)
              + f"{statistics.mean(row):>10.2f}")

    # pairwise win rates: count only if the relative order agrees in both orders
    print("\nPairwise win rates (row method vs column method; counted only when both orders agree)")
    for dim in DIMS:
        print(f"\n  [{dim}]")
        hdr = f"{'':<22}" + "".join(f"{b[:12]:>14}" for b in arms)
        print(hdr)
        for a in arms:
            cells = []
            for b in arms:
                if a == b:
                    cells.append(f"{'--':>14}")
                    continue
                w = l = 0
                for k in qs:
                    if (k, dim) not in o1 or (k, dim) not in o2:
                        continue
                    p1 = o1[(k, dim)].index(a) < o1[(k, dim)].index(b)
                    p2 = o2[(k, dim)].index(a) < o2[(k, dim)].index(b)
                    if p1 and p2:
                        w += 1
                    elif (not p1) and (not p2):
                        l += 1
                n = w + l
                cells.append(f"{(w/n if n else .5):>13.1%} ")
            print(f"{a:<22}" + "".join(cells))

    # diagnostic 1: is the top-ranked answer always the longest
    print("\nDiagnostic 1 -- is the top-ranked answer the longest (the higher the share, the more it looks like rating length only)")
    for dim in DIMS:
        hit = tot = 0
        for k in qs:
            if (k, dim) not in o1:
                continue
            lens = {a: len(ans[a][k]["answer"]) for a in arms}
            longest = max(lens, key=lens.get)
            for seq in (o1[(k, dim)], o2[(k, dim)]):
                tot += 1
                hit += (seq[0] == longest)
        print(f"  {dim:<20} {hit}/{tot} = {hit/tot:.1%}" if tot else f"  {dim}: no data")
    print("  (if directness is close to the other three criteria, it does not work as a control criterion)")

    # diagnostic 2: inter-criterion rank agreement (share with the same top answer)
    print("\nDiagnostic 2 -- inter-criterion agreement on the top answer")
    for d1, d2 in itertools.combinations(DIMS, 2):
        s = t = 0
        for k in qs:
            if (k, d1) in o1 and (k, d2) in o1:
                t += 1
                s += (o1[(k, d1)][0] == o1[(k, d2)][0])
        if t:
            print(f"  {d1[:13]:<14} vs {d2[:13]:<14} {s}/{t} = {s/t:.1%}")

    # ---- Bradley-Terry: more stable than average rank, and gives "how much stronger" with confidence intervals ----
    import sys as _s
    _s.path.insert(0, DYLO_ROOT + "/graphrag")
    from arena_bt import bt_with_ci
    print("\n" + "=" * 72)
    print("Bradley-Terry strength (higher is better; 0 = mean over methods; with 95% bootstrap confidence intervals)")
    both = {}
    for (k, dim) in common:
        both[(k, dim, 1)] = o1[(k, dim)]
        both[(k, dim, 2)] = o2[(k, dim)]
    qual = {(k, d, o): v for (k, d, o), v in both.items() if d in QUALITY}
    if qual:
        bt = bt_with_ci({(f"{k}_{o}", d): v for (k, d, o), v in qual.items()}, arms)
        print(f"\n  [quality pooled: {' + '.join(QUALITY)}]")
        for a in sorted(arms, key=lambda x: -bt[x][0]):
            s_, lo, hi = bt[a]
            print(f"    {a:<24}{s_:>7.3f}  [{lo:>6.3f}, {hi:>6.3f}]")
    for dim in DIMS:
        sub = {(f"{k}_{o}", dim): v for (k, d, o), v in both.items() if d == dim}
        if not sub:
            continue
        bt = bt_with_ci(sub, arms)
        tag = "control criterion, excluded from the total" if dim == "directness" else ""
        print(f"\n  [{dim}] {tag}")
        for a in sorted(arms, key=lambda x: -bt[x][0]):
            s_, lo, hi = bt[a]
            print(f"    {a:<24}{s_:>7.3f}  [{lo:>6.3f}, {hi:>6.3f}]")

    json.dump({"arms": arms,
               "mean_rank": {a: {d: statistics.mean(rank_by[a][d])
                                 for d in DIMS if rank_by[a][d]} for a in arms}},
              open(pack / "arena_result.json", "w"), ensure_ascii=False, indent=1)
    print(f"\nsaved: {pack/'arena_result.json'}")


if __name__ == "__main__":
    main()
