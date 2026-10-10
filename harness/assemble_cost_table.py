#!/usr/bin/env python3
"""Assemble the re-measured results into a cost table on a common scale.

All three inputs were measured on the **full native stream with native batch counts**, with consistent accounting:
  resumm_recost.csv  baseline methods (leaf cost for flat methods, hierarchical cost for methods with a real hierarchy)
  reclaim_ab.csv     DyLo std4, reclamation on/off A/B
  (reclamation on = the main-table configuration, verified to reproduce 13 of 14 cells)

Two outputs:
  1) main-table accounting (DyLo = reclamation on), filling in the four large graphs declared in Table 4 but missing from the main table;
  2) reclamation A/B, answering "what if singleton reclamation is off".

Usage: assemble_cost_table.py [--tex output.tex]
"""
import os as _os  # artifact: paths are relative to the repository root
DYLO_ROOT = _os.environ.get('DYLO_ROOT', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
DYLO_MHRAG = _os.environ.get('DYLO_MHRAG', _os.path.join(DYLO_ROOT, 'mhrag'))
DYLO_SCRATCH = _os.environ.get('DYLO_SCRATCH', '/tmp/dylo')
import argparse
import csv
import collections
from pathlib import Path

R = Path(DYLO_ROOT + "/results")

# paper row name -> method name in the re-computed results
ROWS = [
    ("From-scratch Leiden", "leiden-static"),
    ("DynaMo", "dynamo"),
    ("Batch", "batch"),
    ("QCA", "qca"),
    ("DF-Leiden", "df-leiden"),
    ("$\\Delta$-screen", "ds-louvain"),
    ("DF-Louvain", "df-louvain"),
    ("GreMod", "gremod"),
    ("HIT-Leiden", "hit-leiden"),
]
DS = ["facebook", "cit-hepth", "cit-hepph", "epinions", "twitter", "dblp",
      "gplus", "flickr", "youtube"]
NICE = {"facebook": "Facebook", "cit-hepth": "Cit-HepTh", "cit-hepph": "Cit-HepPh",
        "epinions": "Epinions", "twitter": "Twitter", "dblp": "DBLP",
        "gplus": "GPlus", "flickr": "Flickr", "youtube": "YouTube"}


def load():
    base = collections.defaultdict(dict)
    p = R / "resumm_recost.csv"
    if p.exists():
        for r in csv.DictReader(open(p)):
            if r["status"] == "ok":
                base[r["dataset"]][r["method"]] = float(r["cost_norm"])
    ab = collections.defaultdict(dict)
    p = R / "reclaim_ab.csv"
    if p.exists():
        for r in csv.DictReader(open(p)):
            if r["status"] == "ok":
                ab[r["dataset"]][(r["instance"], r["variant"])] = r
    return base, ab


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tex", default="")
    args = ap.parse_args()
    base, ab = load()

    for title, variant in [("main-table accounting (DyLo with singleton reclamation)", "recl"),
                           ("without singleton reclamation", "norecl")]:
        print(f"\n===== {title} =====")
        hdr = f"{'Method':<22}" + "".join(f"{NICE[d][:9]:>10}" for d in DS if d in base or d in ab)
        print(hdr)
        for name, m in ROWS:
            cells = []
            for d in DS:
                v = base.get(d, {}).get(m)
                cells.append(f"{v:>10.3f}" if v is not None else f"{'--':>10}")
            print(f"{name:<22}" + "".join(cells))
        for inst in ("louvain", "slm", "leiden"):
            cells = []
            for d in DS:
                r = ab.get(d, {}).get((inst, variant))
                cells.append(f"{float(r['cost_hier']):>10.3f}" if r else f"{'--':>10}")
            print(f"{'DyLo-'+inst:<22}" + "".join(cells))

    # reclamation A/B: cost ratio and quality cost
    print("\n===== reclamation on/off: cost ratio and Q cost =====")
    print(f"{'dataset':<12}{'instance':<9}{'recl':>9}{'no recl':>10}{'ratio':>9}"
          f"{'dQ':>10}{'communities recl->no recl':>22}")
    for d in DS:
        for inst in ("louvain", "slm", "leiden"):
            a = ab.get(d, {}).get((inst, "recl"))
            b = ab.get(d, {}).get((inst, "norecl"))
            if not (a and b):
                continue
            ca, cb = float(a["cost_hier"]), float(b["cost_hier"])
            dq = float(b["Q"]) - float(a["Q"])
            mult = ca / cb if cb > 0 else float("inf")
            print(f"{NICE[d]:<12}{inst:<9}{ca:>9.4f}{cb:>10.4f}{mult:>8.0f}x"
                  f"{dq:>+10.4f}{a['num_comms_end']+' → '+b['num_comms_end']:>22}")

    if args.tex:
        Path(args.tex).write_text(_tex(base, ab), encoding="utf-8")
        print(f"\ntex written: {args.tex}")


def _tex(base, ab) -> str:
    ds = [d for d in DS]
    L = ["% re-measured on a common scale: full native stream, native batch counts, same accounting for DyLo and the baselines",
         "\\begin{tabular}{l" + "r" * len(ds) + "}", "\\toprule",
         "Method & " + " & ".join(NICE[d] for d in ds) + " \\\\", "\\midrule"]
    for name, m in ROWS:
        cs = [f"${base.get(d, {}).get(m):.3f}$".replace("0.", ".")
              if base.get(d, {}).get(m) is not None else "--" for d in ds]
        L.append(f"{name} & " + " & ".join(cs) + " \\\\")
    L.append("\\midrule")
    for inst in ("louvain", "slm", "leiden"):
        cs = []
        for d in ds:
            r = ab.get(d, {}).get((inst, "recl"))
            cs.append(f"${float(r['cost_hier']):.3f}$".replace("0.", ".") if r else "--")
        L.append(f"\\DyLo{{}}-{inst} & " + " & ".join(cs) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
