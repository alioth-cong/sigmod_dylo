#!/usr/bin/env python3
"""Summary, Pareto plots, and paper-ready tex table for the selector ablation.

Input: results/sel_ablation.csv (from sel_ablation.py). Output to results/sel_ablation/:
  summary.txt          Q / cost / committed moves / sum dF (% of DP) per graph and lambda
  matched_q.csv        cost needed for equal Q: each selector's (Q, cost) curve interpolated at the Q of every DP point
  dominance.csv        whether each non-DP point is strictly dominated by some DP point (Q not lower and cost not higher, at least one strictly)
  fig_sel_pareto.pdf   Q-cost Pareto (main: YouTube / GPlus / Facebook)
  fig_sel_pareto_all.pdf  all graphs
  fig_sel_objective.pdf   sum dF as % of DP vs lambda
  tab_sel.tex          single-column table of sum dF as % of DP
"""
import os as _os  # artifact: paths are relative to the repository root
DYLO_ROOT = _os.environ.get('DYLO_ROOT', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
DYLO_MHRAG = _os.environ.get('DYLO_MHRAG', _os.path.join(DYLO_ROOT, 'mhrag'))
DYLO_SCRATCH = _os.environ.get('DYLO_SCRATCH', '/tmp/dylo')
import csv
import collections
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator, NullFormatter  # noqa: E402

R = Path(DYLO_ROOT + "/results")
OUT = R / "sel_ablation"
NICE = {"facebook": "Facebook", "cit-hepth": "Cit-HepTh", "epinions": "Epinions",
        "dblp": "DBLP", "youtube": "YouTube", "gplus": "GPlus"}
MODES = ["dp", "greedy", "indep"]
LABEL = {"dp": "Tree DP (DyLo)", "greedy": "Per-destination greedy",
         "indep": "Per-move pricing"}
STYLE = {"dp": dict(color="#111111", ls="-", marker="o", ms=4.5, mfc="#111111"),
         "greedy": dict(color="#6f6f6f", ls="--", marker="s", ms=4.5, mfc="white"),
         "indep": dict(color="#9a9a9a", ls=":", marker="^", ms=5, mfc="#bdbdbd")}
plt.rcParams.update({"font.family": "serif", "pdf.fonttype": 42, "ps.fonttype": 42,
                     "font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 8,
                     "ytick.labelsize": 8, "axes.edgecolor": "#333333",
                     "axes.linewidth": 0.8})


def load():
    D = collections.defaultdict(dict)
    for r in csv.DictReader(open(R / "sel_ablation.csv")):
        if r["status"] != "ok":
            continue
        rec = {k: float(r[k]) for k in ("Q", "Q_end", "cost", "sel_cost", "moves", "sumDF", "sumGain")}
        D[r["dataset"]][(r["mode"], r["lam"])] = rec
    return D


def lams_of(d):
    return sorted({l for (_m, l) in d if l != "0"}, key=float)


def interp_cost(points, q):
    """points: [(Q, cost)]; return the cost needed to reach q (linear interpolation after sorting by Q), or None outside the range."""
    pts = sorted(points)
    if not pts or q < pts[0][0] - 1e-12 or q > pts[-1][0] + 1e-12:
        return None
    for (q0, c0), (q1, c1) in zip(pts, pts[1:]):
        if q0 - 1e-12 <= q <= q1 + 1e-12:
            return c0 if q1 == q0 else c0 + (c1 - c0) * (q - q0) / (q1 - q0)
    return pts[-1][1] if abs(q - pts[-1][0]) < 1e-12 else None


def main():
    D = load()
    OUT.mkdir(parents=True, exist_ok=True)
    order = [d for d in ["youtube", "gplus", "facebook", "epinions", "cit-hepth", "dblp"] if d in D]
    lines, mq_rows, dom_rows = [], [], []
    for ds in order:
        d = D[ds]
        z = d.get(("dp", "0"))
        lines.append(f"\n== {NICE[ds]} ==" + (f"  λ=0: Q={z['Q']:.5f} cost={z['cost']:.4f} moves={z['moves']:.0f}" if z else ""))
        for lam in lams_of(d):
            dp = d.get(("dp", lam))
            if not dp:
                continue
            cells = []
            for m in MODES:
                r = d.get((m, lam))
                if not r:
                    cells.append("—")
                    continue
                pct = 100 * r["sumDF"] / dp["sumDF"] if dp["sumDF"] else 100.0
                cells.append(f"{m}: Q={r['Q']:.5f} c={r['cost']:.4f} mv={r['moves']:.0f} ΣΔF={pct:.1f}%")
            lines.append(f"  λ={lam:<5} " + " | ".join(cells))
        # cost at equal Q + dominance
        curves = {m: [(r["Q"], r["cost"]) for (mm, l), r in d.items() if mm == m and l != "0"] for m in MODES}
        if z:
            for m in MODES:
                curves[m].append((z["Q"], z["cost"]))
        for lam in lams_of(d):
            dp = d.get(("dp", lam))
            if not dp:
                continue
            for m in ("greedy", "indep"):
                c = interp_cost(curves[m], dp["Q"])
                mq_rows.append(dict(dataset=ds, lam=lam, mode=m, Q_dp=dp["Q"], cost_dp=dp["cost"],
                                    cost_mode_at_Q=c if c is not None else "",
                                    ratio=(c / dp["cost"]) if c is not None else ""))
        for (m, lam), r in d.items():
            if m == "dp" or lam == "0":
                continue
            same = d.get(("dp", lam))
            if same and abs(same["Q"] - r["Q"]) < 1e-9 and abs(same["cost"] - r["cost"]) < 1e-9 \
                    and same["moves"] == r["moves"]:
                continue   # same decisions as DP at this lambda: not a selector-induced difference, excluded from the dominance test
            # substantive dominance: Q not lower (tolerance 1e-7), cost not higher (tolerance 0.1%), and a substantive advantage in at least one
            # (Q higher by more than 1e-6, or cost lower by more than 0.1%) -- excludes numerical jitter when all three coincide at large lambda
            dom = [l2 for (m2, l2), r2 in d.items() if m2 == "dp"
                   and r2["Q"] >= r["Q"] - 1e-7 and r2["cost"] <= r["cost"] * 1.001
                   and (r2["Q"] > r["Q"] + 1e-6 or r2["cost"] < r["cost"] * 0.999)]
            dom_rows.append(dict(dataset=ds, mode=m, lam=lam, Q=r["Q"], cost=r["cost"],
                                 dominated_by_dp_lams=";".join(sorted(dom, key=float))))
    (OUT / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for name, rows in (("matched_q.csv", mq_rows), ("dominance.csv", dom_rows)):
        if rows:
            with open(OUT / name, "w", newline="") as g:
                w = csv.DictWriter(g, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
    print("\n".join(lines))
    nd = sum(1 for r in dom_rows if r["dominated_by_dp_lams"])
    print(f"\nnon-DP points strictly dominated by DP: {nd}/{len(dom_rows)}")
    for r in dom_rows:
        if r["dominated_by_dp_lams"]:
            print(f"  {NICE[r['dataset']]:<10} {r['mode']:<7} lambda={r['lam']:<5} dominated by DP at lambda in {{{r['dominated_by_dp_lams']}}}")
    rat = [r for r in mq_rows if r["ratio"] != "" and abs(r["ratio"] - 1) > 1e-9]
    print(f"\ncost at equal Q ({len(rat)} cases different from 1):")
    for r in rat:
        print(f"  {NICE[r['dataset']]:<10} {r['mode']:<7} at DP lambda={r['lam']:<5} Q={r['Q_dp']:.5f}: cost {r['cost_mode_at_Q']:.4f} vs DP {r['cost_dp']:.4f} ({r['ratio']:.3f}x)")

    # ---------- Pareto plots ----------
    def pareto(dsets, fname):
        n = len(dsets)
        cols = min(n, 3)
        rows = math.ceil(n / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(2.3 * cols, 1.95 * rows + 0.35), squeeze=False)
        for ax, ds in zip(axes.flat, dsets):
            d = D[ds]
            base = d[("dp", "1e-1")]["Q"] if ("dp", "1e-1") in d else 0.0
            for zo, m in zip((3, 2, 1), MODES):   # DP drawn on top, not hidden by other markers
                pts = sorted([(r["cost"], (r["Q"] - base) * 1e4) for (mm, l), r in d.items()
                              if mm == m and l != "0"])
                if pts:
                    ax.plot(*zip(*pts), label=LABEL[m], lw=1.1, zorder=zo, **STYLE[m])
            z = d.get(("dp", "0"))
            if z:
                ax.plot([z["cost"]], [(z["Q"] - base) * 1e4], marker="*", ms=8, color="#333333",
                        ls="none", label=r"$\lambda=0$ (modularity only)")
            ax.set_title(NICE[ds], fontsize=9, pad=3)
            xs = [r["cost"] for r in d.values()]
            lo, hi = min(xs), max(xs)
            ticks = [t for t in (0.02, 0.05, 0.1, 0.2, 0.5, 1.0) if lo * 0.97 <= t <= hi * 1.03]
            if len(ticks) >= 2:
                ax.set_xscale("log")
                ax.xaxis.set_major_locator(FixedLocator(ticks))
                ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:g}"))
                ax.xaxis.set_minor_formatter(NullFormatter())
            else:   # less than one order of magnitude: a linear scale reads better
                ax.xaxis.set_major_locator(MaxNLocator(4))
            ax.grid(True, which="major", color="#e3e3e3", lw=0.5)
            ax.set_xlabel("Re-summarization cost")
        for ax in axes[:, 0]:
            ax.set_ylabel(r"$\bar Q-\bar Q_{\lambda=0.1}$ ($\times10^{-4}$)")
        for ax in list(axes.flat)[n:]:
            ax.axis("off")
        h, lab = axes.flat[0].get_legend_handles_labels()
        fig.legend(h, lab, loc="upper center", ncol=4, frameon=False, fontsize=7.5,
                   bbox_to_anchor=(0.5, 1.0), handlelength=2.2, columnspacing=1.0)
        fig.tight_layout(rect=(0, 0, 1, 0.92 if rows == 1 else 0.95))
        fig.savefig(OUT / fname)
        fig.savefig(OUT / fname.replace(".pdf", ".png"), dpi=200)
        plt.close(fig)

    main_ds = [d for d in ["youtube", "gplus", "facebook"] if d in D]
    if main_ds:
        pareto(main_ds, "fig_sel_pareto.pdf")
    pareto(order, "fig_sel_pareto_all.pdf")

    # ---------- sum dF as % of DP ----------
    fig, axes = plt.subplots(1, 2, figsize=(4.8, 2.0), sharey=True)
    gstyle = ["-", "--", ":", "-.", (0, (5, 1)), (0, (1, 1))]
    gmark = ["o", "s", "^", "D", "v", "P"]
    gshade = ["#111111", "#333333", "#555555", "#6f6f6f", "#9a9a9a", "#bdbdbd"]
    for ax, m in zip(axes, ("indep", "greedy")):
        for i, ds in enumerate(order):
            d = D[ds]
            pts = [(float(l), 100 * d[(m, l)]["sumDF"] / d[("dp", l)]["sumDF"])
                   for l in lams_of(d) if (m, l) in d and ("dp", l) in d and d[("dp", l)]["sumDF"] > 0]
            if pts:
                ax.plot(*zip(*sorted(pts)), ls=gstyle[i], marker=gmark[i], color=gshade[i],
                        ms=3.8, lw=1.0, mfc="white" if i % 2 else gshade[i], label=NICE[ds])
        ax.set_xscale("log")
        ax.set_title(LABEL[m], fontsize=9, pad=3)
        ax.set_xlabel(r"Cost rate $\lambda$")
        ax.grid(True, color="#e3e3e3", lw=0.5)
    axes[0].set_ylabel(r"Realized $\sum\Delta F$ (% of DP)")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=6, frameon=False, fontsize=7,
               bbox_to_anchor=(0.5, 1.02), handlelength=2.0, columnspacing=0.8)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(OUT / "fig_sel_objective.pdf")
    fig.savefig(OUT / "fig_sel_objective.png", dpi=200)
    plt.close(fig)

    # ---------- tex table: sum dF as % of DP ----------
    show = ["1e-4", "1e-3", "1e-2", "1e-1"]
    tex = ["% selector ablation (generated by sel_ablation.py / sel_report.py)",
           "\\begin{table}[t]", "\\centering",
           "\\caption{Realized objective $\\sum\\Delta F$ of per-destination greedy (G) and "
           "per-move pricing (P), as a percentage of the tree DP, by cost rate $\\lambda$.}",
           "\\label{tab:sel-objective}",
           "\\begin{tabular*}{\\linewidth}{@{\\extracolsep{\\fill}}l" + "rr" * len(show) + "@{}}",
           "\\toprule",
           "& " + " & ".join("\\multicolumn{2}{c}{$\\lambda=10^{-%d}$}" % int(l.split("e-")[1]) for l in show) + " \\\\",
           "\\textbf{Graph} & " + " & ".join(["\\textbf{G}", "\\textbf{P}"] * len(show)) + " \\\\",
           "\\midrule"]
    for ds in order:
        d = D[ds]
        cells = []
        for l in show:
            for m in ("greedy", "indep"):
                if (m, l) in d and ("dp", l) in d and d[("dp", l)]["sumDF"] > 0:
                    cells.append(f"{100 * d[(m, l)]['sumDF'] / d[('dp', l)]['sumDF']:.0f}")
                else:
                    cells.append("--")
        tex.append(f"{NICE[ds]} & " + " & ".join(cells) + " \\\\")
    tex += ["\\bottomrule", "\\end{tabular*}", "\\end{table}"]
    (OUT / "tab_sel.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
