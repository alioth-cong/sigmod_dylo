#!/usr/bin/env python3
"""
Regenerate the long-format result CSVs from per_batch/*.tsv (the source of truth),
and classify every (method,dataset) cell that did NOT complete as one of:
  ok | partial-reboot | TLE | OOM | no-run(by design) | not-run

Output files:
  results/latency.csv     method,dataset,value,unit,language,repo,commit,notes
  results/quality.csv     method,dataset,value,unit,language,repo,commit,notes
  results/resumm.csv      method,dataset,value,unit,language,repo,commit,notes
  results/scalability.csv method,|E|,latency_ms,modularity,language,repo,commit,notes
  results/failures.tsv    method,dataset,status,batches,detail
"""
import os as _os  # artifact: paths are relative to the repository root
DYLO_ROOT = _os.environ.get('DYLO_ROOT', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
DYLO_MHRAG = _os.environ.get('DYLO_MHRAG', _os.path.join(DYLO_ROOT, 'mhrag'))
DYLO_SCRATCH = _os.environ.get('DYLO_SCRATCH', '/tmp/dylo')
import csv
import json
import re
import sys
from pathlib import Path

ROOT      = Path(DYLO_ROOT + "")
RESULTS   = ROOT / "results"
PER_BATCH = RESULTS / "per_batch"
RUNS      = ROOT / "data" / "runs"
LOGS      = ROOT / "logs" / "runs"

sys.path.insert(0, str(ROOT/"harness"))
from run import METHODS  # noqa

TLE_BUDGET_S = 4 * 3600  # 4h per (method,dataset)

MAIN_DATASETS = ["facebook","cit-hepth","cit-hepph","epinions","twitter",
                 "gplus","dblp","youtube","flickr","livejournal"]

# By-design dispositions we assert from analysis (not derivable from a partial log).
BY_DESIGN = {
    ("dynamo","livejournal"):
        ("no-run", "DynaMo needs a serialized cumulative snapshot per batch; "
                   "4286 snapshots of a 38M-edge graph ~= 2.7TB disk -> infeasible"),
    ("louvain-static","livejournal"):
        ("TLE", "static rebuild per batch; 4286 batches x ~tens of s on 38M edges >> 4h"),
    ("leiden-static","livejournal"):
        ("TLE", "static rebuild per batch; 4286 batches on 38M edges >> 4h"),
    ("slm-static","livejournal"):
        ("TLE", "static rebuild per batch; 4286 batches on 38M edges >> 4h"),
    ("slm-static","gplus"):
        ("TLE", "genuine SLM from-scratch per batch; 1224 batches on 12M edges >> 4h"),
    ("batch","livejournal"):
        ("TLE", "batch-incremental too slow on 38M-edge graph; <1k/4286 batches in prior run"),
    ("qca","livejournal"):
        ("TLE", "ran the full 4h budget (14401s) and was killed at ~3400/4286; "
                "per-batch community-file I/O on the 38M-edge graph exceeds 4h"),
}

OOM_MARKERS = ("OutOfMemoryError", "java.lang.OutOfMemory", "std::bad_alloc",
               "bad_alloc", "Killed", "Cannot allocate memory", "GC overhead limit")


def read_pb(p: Path):
    """rows: [(batch, latency_ms, modularity, num_communities, cost_abs, cost_norm)]"""
    rows = []
    with open(p) as f:
        next(f)
        for line in f:
            c = line.rstrip("\n").split("\t")
            if len(c) < 6: continue
            try:
                rows.append((int(c[0]), float(c[1]), float(c[2]), int(c[3]),
                             float(c[4]) if c[4] != "" else float("nan"),
                             float(c[5]) if c[5] != "" else float("nan")))
            except ValueError:
                continue
    return rows


def n_batches_of(dataset: str):
    mp = RUNS / dataset / "meta.json"
    if mp.exists():
        try: return json.loads(mp.read_text())["n_batches"]
        except Exception: return None
    return None


def metrics_last_batch(method: str, dataset: str):
    """Last batch index present in the cell's out_dir/metrics.tsv, or -1."""
    m = RUNS / dataset / f"out_{method}" / "metrics.tsv"
    if not m.exists(): return -1
    last = -1
    try:
        for line in m.read_text().splitlines()[1:]:
            c = line.split("\t")
            if c and c[0].isdigit(): last = max(last, int(c[0]))
    except Exception:
        pass
    return last


def log_has_oom(method: str, dataset: str):
    lg = LOGS / f"{method}__{dataset}.log"
    if not lg.exists(): return None
    try:
        txt = lg.read_text(errors="replace")
    except Exception:
        return None
    for mk in OOM_MARKERS:
        if mk in txt:
            return mk
    return None


def classify(method: str, dataset: str, batches_done: int, n_batches, avg_lat_ms=None):
    """Return (status, detail) for a cell that is not fully complete."""
    if (method, dataset) in BY_DESIGN:
        st, d = BY_DESIGN[(method, dataset)]
        if batches_done > 0:
            d += f"; reached {batches_done}/{n_batches}"
        return st, d
    oom = log_has_oom(method, dataset)
    if oom:
        return "OOM", f"OOM marker '{oom}' in log; reached {max(batches_done,0)}/{n_batches}"
    # estimate full-run wall time from the partial per-batch latency
    if avg_lat_ms and n_batches:
        est_s = avg_lat_ms / 1000.0 * n_batches
        if est_s > TLE_BUDGET_S:
            return "TLE", (f"reached {batches_done}/{n_batches}; "
                           f"est full run {est_s/3600:.1f}h > 4h budget "
                           f"({avg_lat_ms/1000:.1f}s/batch)")
    if batches_done > 0:
        frac = batches_done / n_batches if n_batches else 0
        return "partial-reboot", (f"reached {batches_done}/{n_batches} "
                                  f"({frac*100:.0f}%); run interrupted by machine reboot")
    # nothing in per_batch; check raw metrics
    ml = metrics_last_batch(method, dataset)
    if ml > 0:
        return "partial-reboot", f"out_dir metrics to batch {ml}/{n_batches} but per_batch not built"
    if ml == 0 or (LOGS / f"{method}__{dataset}.log").exists():
        return "not-run", "attempted; no usable batches produced (likely interrupted before batch 1)"
    return "not-run", "no run attempted"


def write_csv(path: Path, header, rows):
    with open(path, "w") as g:
        w = csv.writer(g)
        w.writerow(header)
        for r in rows:
            w.writerow([r.get(h, "") for h in header])


FROMSCRATCH_REF = "louvain-static"        # P3 norm (b): from-scratch baseline (=1.0)
FROMSCRATCH_REF_LEIDEN = "leiden-static"  # P3 (writing-agent follow-up): Leiden-referenced norm


def load_ref_costabs(refmethod: str):
    """dataset -> {batch: cost_abs} for a from-scratch reference method."""
    ref = {}
    for pb in PER_BATCH.glob(f"{refmethod}__*.tsv"):
        dataset = pb.stem.split("__", 1)[1]
        ref[dataset] = {r[0]: r[4] for r in read_pb(pb) if r[0] >= 1 and r[4] == r[4]}
    return ref


def main():
    lat_rows, qual_rows, cost_rows, scal_rows = [], [], [], []
    failures = []  # (method, dataset, status, batches, detail)
    seen = set()
    fs_ref = load_ref_costabs(FROMSCRATCH_REF)               # P3 (louvain-static)
    fs_ref_leiden = load_ref_costabs(FROMSCRATCH_REF_LEIDEN) # P3 (leiden-static)

    for pb in sorted(PER_BATCH.glob("*.tsv")):
        name = pb.stem
        if "__" not in name: continue
        method, dataset = name.split("__", 1)
        if method not in METHODS: continue
        seen.add((method, dataset))
        meta = METHODS[method]
        rows = read_pb(pb)
        batch_rows = [r for r in rows if r[0] >= 1]
        nb = n_batches_of(dataset)
        if not batch_rows:
            failures.append((method, dataset, *classify(method, dataset, 0, nb)))
            continue

        avg_lat = sum(r[1] for r in batch_rows) / len(batch_rows)
        fmods = [r[2] for r in batch_rows if r[2] == r[2]]
        avg_mod = sum(fmods)/len(fmods) if fmods else float("nan")
        fcost = [r[5] for r in batch_rows if r[5] == r[5]]
        avg_cost = sum(fcost)/len(fcost) if fcost else float("nan")   # norm (a): / total weight
        ncostabs = sum(1 for r in batch_rows if r[4] == r[4])
        avg_cost_abs = (sum(r[4] for r in batch_rows if r[4]==r[4]) / ncostabs) if ncostabs else float("nan")
        # P3 norm (b): per-batch cost relative to the from-scratch reference (louvain-static),
        # so the from-scratch baseline is 1.0. Averaged over batches where the ref churned (>0).
        refd = fs_ref.get(dataset, {})
        ratios = [r[4]/refd[r[0]] for r in batch_rows
                  if r[4] == r[4] and r[0] in refd and refd[r[0]] > 0]
        avg_cost_fs = (sum(ratios)/len(ratios)) if ratios else float("nan")
        refl = fs_ref_leiden.get(dataset, {})
        ratiosl = [r[4]/refl[r[0]] for r in batch_rows
                   if r[4] == r[4] and r[0] in refl and refl[r[0]] > 0]
        avg_cost_fsl = (sum(ratiosl)/len(ratiosl)) if ratiosl else float("nan")

        complete = (nb is not None and len(batch_rows) >= nb)
        if complete:
            status, detail = "ok", ""
            note_status = "ok"
        else:
            status, detail = classify(method, dataset, len(batch_rows), nb, avg_lat)
            failures.append((method, dataset, status, f"{len(batch_rows)}/{nb}", detail))
            note_status = f"{status}: {len(batch_rows)}/{nb}"

        common = dict(method=method, dataset=dataset,
                      language=meta["language"], repo=meta["repo"], commit=meta["commit"],
                      notes=f"{meta['notes']}; {note_status}")
        lat_rows.append({**common, "value": f"{avg_lat:.3f}", "unit": "ms/batch"})
        qual_rows.append({**common, "value": f"{avg_mod:.6f}" if avg_mod==avg_mod else "NaN", "unit": "Q (modularity)"})
        cost_rows.append({
            "method": method, "dataset": dataset,
            "cost_norm_weight":            f"{avg_cost:.6f}"     if avg_cost==avg_cost     else "NaN",
            "cost_norm_fromscratch":       f"{avg_cost_fs:.6f}"  if avg_cost_fs==avg_cost_fs else "NaN",
            "cost_norm_fromscratch_leiden":f"{avg_cost_fsl:.6f}" if avg_cost_fsl==avg_cost_fsl else "NaN",
            "cost_abs":                    f"{avg_cost_abs:.3f}" if avg_cost_abs==avg_cost_abs else "NaN",
            "fromscratch_ref":             FROMSCRATCH_REF if (avg_cost_fs==avg_cost_fs) else "",
            "language": meta["language"], "repo": meta["repo"], "commit": meta["commit"],
            "notes": common["notes"]})

        m = re.match(r"^lj_(\d+)M$", dataset)
        if m:
            pN = int(m.group(1))
            mp = RUNS / dataset / "meta.json"
            edges = json.loads(mp.read_text()).get("edges", pN*1_000_000) if mp.exists() else pN*1_000_000
            scal_rows.append({"method": method, "|E|": edges, "latency_ms": f"{avg_lat:.3f}",
                              "modularity": f"{avg_mod:.6f}" if avg_mod==avg_mod else "NaN",
                              "language": meta["language"], "repo": meta["repo"], "commit": meta["commit"],
                              "notes": meta["notes"] + f"; {note_status}; prefix={pN}M"})

    # Enumerate expected main cells that produced no per_batch at all
    for d in MAIN_DATASETS:
        for mth in METHODS:
            if (mth, d) in seen: continue
            nb = n_batches_of(d)
            st, detail = classify(mth, d, 0, nb)
            failures.append((mth, d, st, f"0/{nb}" if nb else "0/?", detail))

    order = lambda r: (r.get("dataset",""), r.get("method",""))
    for rs in (lat_rows, qual_rows, cost_rows): rs.sort(key=order)
    scal_rows.sort(key=lambda r: (r.get("method",""), int(r.get("|E|",0))))

    # Manually-measured rows whose raw per_batch TSVs no longer exist (see
    # results/manual_rows/README.md). Append AFTER regeneration so they survive
    # re-finalization; a per_batch-derived row for the same (method,dataset) wins.
    def merge_manual(rows, fname):
        mp = RESULTS / "manual_rows" / fname
        if not mp.exists(): return rows
        have = {(r["method"], r["dataset"]) for r in rows}
        for r in csv.DictReader(open(mp)):
            if (r["method"], r["dataset"]) not in have:
                rows.append(r)
        return rows
    lat_rows  = merge_manual(lat_rows,  "latency_manual.csv")
    qual_rows = merge_manual(qual_rows, "quality_manual.csv")
    cost_rows = merge_manual(cost_rows, "resumm_manual.csv")
    for rs in (lat_rows, qual_rows, cost_rows): rs.sort(key=order)

    hmain = ["method","dataset","value","unit","language","repo","commit","notes"]
    write_csv(RESULTS/"latency.csv", hmain, lat_rows)
    write_csv(RESULTS/"quality.csv", hmain, qual_rows)
    # resumm.csv (P3): both normalizations + absolute cost
    #   cost_norm_weight      = dirty weight / total partition weight  (worst case=1.0)
    #   cost_norm_fromscratch = dirty weight / from-scratch (louvain-static) per-batch cost (from-scratch=1.0)
    write_csv(RESULTS/"resumm.csv",
              ["method","dataset","cost_norm_weight","cost_norm_fromscratch",
               "cost_norm_fromscratch_leiden","cost_abs",
               "fromscratch_ref","language","repo","commit","notes"], cost_rows)
    write_csv(RESULTS/"scalability.csv",
              ["method","|E|","latency_ms","modularity","language","repo","commit","notes"], scal_rows)

    # P1: hierarchical re-summarization cost, from per_batch_hier/*.tsv (dendrogram methods)
    hier_rows = []
    PB_HIER = RESULTS / "per_batch_hier"
    if PB_HIER.exists():
        for pb in sorted(PB_HIER.glob("*.tsv")):
            if "__" not in pb.stem: continue
            method, dataset = pb.stem.split("__", 1)
            if method not in METHODS: continue
            vals = []
            for line in pb.read_text().splitlines()[1:]:
                c = line.split("\t")
                if len(c) >= 3 and c[0].isdigit() and int(c[0]) >= 1:
                    try: vals.append((float(c[1]), float(c[2])))
                    except ValueError: pass
            if not vals: continue
            meta = METHODS[method]
            hier_rows.append({
                "method": method, "dataset": dataset,
                "cost_norm_hier": f"{sum(v[1] for v in vals)/len(vals):.6f}",
                "cost_abs_hier":  f"{sum(v[0] for v in vals)/len(vals):.3f}",
                "batches": len(vals),
                "language": meta["language"], "repo": meta["repo"], "commit": meta["commit"],
                "notes": meta["notes"]})
        hier_rows.sort(key=lambda r: (r["dataset"], r["method"]))
        write_csv(RESULTS/"resumm_hier.csv",
                  ["method","dataset","cost_norm_hier","cost_abs_hier","batches",
                   "language","repo","commit","notes"], hier_rows)

    # dedupe failures by (method,dataset), keeping the most specific status
    rank = {"ok":0,"partial-reboot":1,"TLE":2,"OOM":3,"no-run":4,"not-run":5}
    best = {}
    for rec in failures:
        m,d,st = rec[0],rec[1],rec[2]
        rest = rec[3:] if len(rec) > 3 else ("", "")
        if len(rest) == 1: rest = (rest[0], "")
        key=(m,d)
        if key not in best or rank.get(st,9) > rank.get(best[key][0],9):
            best[key] = (st, rest[0] if rest else "", rest[1] if len(rest)>1 else "")

    with open(RESULTS/"failures.tsv","w") as g:
        g.write("method\tdataset\tstatus\tbatches\tdetail\n")
        for (m,d),(st,batches,detail) in sorted(best.items(), key=lambda kv:(kv[1][0],kv[0][1],kv[0][0])):
            g.write(f"{m}\t{d}\t{st}\t{batches}\t{detail}\n")

    # counts
    from collections import Counter
    cnt = Counter(v[0] for v in best.values())
    print(f"latency.csv:{len(lat_rows)}  quality.csv:{len(qual_rows)}  resumm.csv:{len(cost_rows)}  scalability.csv:{len(scal_rows)}")
    print(f"complete cells: {sum(1 for r in lat_rows if r['notes'].endswith('; ok'))}")
    print(f"failures.tsv: {sum(cnt.values())} cells -> {dict(cnt)}")


if __name__ == "__main__":
    main()
