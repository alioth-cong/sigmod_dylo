#!/usr/bin/env python3
"""Proxy fidelity: correlation between the structural cost w(c) and measured summarizer input tokens.

The paper's Cost column is a **structural proxy** -- w(c) counts entities of a leaf community and
children of an internal node -- not real tokens. Review B8 noted that the proxy's token
fidelity was never validated, while the abstract placed proxy-based ratios next to measured-token ratios.

This script regresses on several real indexes. Ground truth comes from GraphRAG's own cache:
every LLM call under cache/community_reporting/ stores usage.prompt_tokens.
The cache key is a content hash and cannot be mapped to a community directly, so the generated report's "title" field is looked up
in community_reports.parquet, keeping only uniquely titled communities (to avoid one-to-many matches).

Correlation is reported at three levels:
  all / leaf subset / internal subset
because the two node kinds have different token conversion rates (leaves count entities, internal nodes count children), a pooled regression
is dragged down by the two slopes -- only separate fits show the proxy's behavior on the bulk (leaves, ~90%).

Also reported:
  Pearson R^2      numeric fidelity
  Spearman rho     rank fidelity (the selector only needs the right order)
  slope/intercept  token = a + b*w, b is the token cost per unit of w
"""
import os as _os  # artifact: paths are relative to the repository root
DYLO_ROOT = _os.environ.get('DYLO_ROOT', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
DYLO_MHRAG = _os.environ.get('DYLO_MHRAG', _os.path.join(DYLO_ROOT, 'mhrag'))
DYLO_SCRATCH = _os.environ.get('DYLO_SCRATCH', '/tmp/dylo')
import json
import math
import re
import sys
from pathlib import Path

import pandas as pd

B = Path(DYLO_MHRAG + "/clean_chain")


def collect(idx: Path):
    """Return [(community, level, w, tokens, is_leaf), ...]"""
    rep = pd.read_parquet(idx / "output/community_reports.parquet")
    com = pd.read_parquet(idx / "output/communities.parquet")

    kids, eids = {}, {}
    for _, r in com.iterrows():
        ch = r.get("children")
        kids[r["community"]] = list(ch) if ch is not None and len(ch) else []
        e = r.get("entity_ids")
        eids[r["community"]] = list(e) if e is not None else []
    # w(c): leaves count entities, internal nodes count children
    w = {c: (len(kids[c]) if kids[c] else len(eids.get(c, []))) for c in kids}

    tok = {}
    cdir = idx / "cache/community_reporting"
    if not cdir.is_dir():
        return []
    for f in cdir.glob("*"):
        try:
            d = json.load(f.open())["result"]["response"]
            m = re.search(r'"title"\s*:\s*"([^"]+)"',
                          d["choices"][0]["message"]["content"])
            if m:
                tok.setdefault(m.group(1).strip(), []).append(d["usage"]["prompt_tokens"])
        except Exception:  # noqa: BLE001
            continue

    out = []
    for _, r in rep.iterrows():
        c, t = r["community"], str(r["title"]).strip()
        if c not in w or t not in tok or len(tok[t]) != 1:
            continue          # duplicate title -> ambiguous mapping, dropped
        out.append((c, int(r["level"]), w[c], tok[t][0], 0 if kids.get(c) else 1))
    return out


def stats(rows, name):
    if len(rows) < 3:
        print(f"  {name:<24} n={len(rows)} too few")
        return None
    x = [float(r[2]) for r in rows]
    y = [float(r[3]) for r in rows]
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    sx = math.sqrt(sum((a - mx) ** 2 for a in x) / n)
    sy = math.sqrt(sum((a - my) ** 2 for a in y) / n)
    cov = sum((a - mx) * (b - my) for a, b in zip(x, y)) / n
    r = 0.0 if sx == 0 or sy == 0 else cov / (sx * sy)
    # Spearman: rank correlation
    def rank(v):
        o = sorted(range(len(v)), key=lambda i: v[i])
        rk = [0.0] * len(v)
        i = 0
        while i < len(o):
            j = i
            while j + 1 < len(o) and v[o[j + 1]] == v[o[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                rk[o[k]] = avg
            i = j + 1
        return rk
    rx, ry = rank(x), rank(y)
    mrx, mry = sum(rx) / n, sum(ry) / n
    srx = math.sqrt(sum((a - mrx) ** 2 for a in rx) / n)
    sry = math.sqrt(sum((a - mry) ** 2 for a in ry) / n)
    rho = 0.0 if srx == 0 or sry == 0 else \
        sum((a - mrx) * (b - mry) for a, b in zip(rx, ry)) / n / (srx * sry)
    b = cov / (sx * sx) if sx else 0.0
    a = my - b * mx
    print(f"  {name:<24} n={n:<5} R²={r*r:.4f}  ρ={rho:.4f}  "
          f"tok≈{a:.0f}+{b:.1f}·w")
    return dict(n=n, r2=r * r, rho=rho, a=a, b=b)


def main():
    idxs = sys.argv[1:] or ["norecl327_g5", "rebuild327", "rebuild350",
                            "norecl350", "rebuild419", "norecl419"]
    allrows = []
    res = {}
    for nm in idxs:
        p = B / nm
        if not (p / "output/community_reports.parquet").exists():
            print(f"\n{nm}: index missing"); continue
        rows = collect(p)
        if not rows:
            print(f"\n{nm}: no matchable samples"); continue
        print(f"\n=== {nm} ({len(rows)} communities matched) ===")
        res[nm] = {
            "all": stats(rows, "all"),
            "leaf": stats([r for r in rows if r[4] == 1], "leaf (entities)"),
            "int": stats([r for r in rows if r[4] == 0], "internal (children)"),
        }
        allrows += rows
    if allrows:
        print(f"\n=== all indexes pooled ({len(allrows)} communities) ===")
        res["POOLED"] = {
            "all": stats(allrows, "all"),
            "leaf": stats([r for r in allrows if r[4] == 1], "leaf"),
            "int": stats([r for r in allrows if r[4] == 0], "internal"),
        }
        lf = [r for r in allrows if r[4] == 1]
        it = [r for r in allrows if r[4] == 0]
        print(f"\n  leaf share {len(lf)/len(allrows):.1%}, "
              f"ratio of per-unit token rates between the two kinds "
              f"{res['POOLED']['int']['b']/res['POOLED']['leaf']['b']:.2f}x")
    out = Path(DYLO_ROOT + "/results/proxy_fidelity.json")
    json.dump(res, out.open("w"), indent=1, ensure_ascii=False)
    pd.DataFrame(allrows, columns=["community", "level", "w", "tokens", "leaf"]) \
      .to_csv(DYLO_ROOT + "/results/proxy_fidelity_rows.csv", index=False)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
