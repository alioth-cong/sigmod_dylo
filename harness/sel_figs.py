#!/usr/bin/env python3
"""Paper figures and tables for the selector ablation (style consistent with the paper's other figures).

Input: results/sel_ablation.csv (lambda sweep, native 1000 edges/batch) and
results/sel_ablation_bigbatch.csv (merged batches). Output to --out (default results/sel_ablation/):
  sel_objective.pdf   sum dF as % of DP vs lambda: (a) per-move pricing (b) per-destination greedy
  sel_pareto.pdf      Q-cost Pareto: YouTube / GPlus / Facebook
  sel_pareto_app.pdf  Q gap to the DP frontier at equal cost: Epinions / DBLP / Cit-HepTh (appendix)
  tab_sel_objective.tex   sum dF as % of DP
  tab_sel_batch.tex       sum dF as % of DP with enlarged batches
"""
import os as _os  # artifact: paths are relative to the repository root
DYLO_ROOT = _os.environ.get('DYLO_ROOT', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
DYLO_MHRAG = _os.environ.get('DYLO_MHRAG', _os.path.join(DYLO_ROOT, 'mhrag'))
DYLO_SCRATCH = _os.environ.get('DYLO_SCRATCH', '/tmp/dylo')
import argparse
import collections
import csv
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mt  # noqa: E402

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "STIXGeneral"],
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "font.size": 6,
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
})
INK, MUTED, GRID = "#1a1a1a", "#52514e", "#e4e3df"
# selectors: the tree DP keeps the paper's DP blue; the two alternatives keep the min-cut / enumeration red and orange
SEL = {
    "dp": dict(label="Tree DP (DyLo)", color="#3f7fc4", marker="o", ms=2.4, lw=1.0, ls="-"),
    "greedy": dict(label="Per-destination greedy", color="#e8871a", marker="s", ms=2.2, lw=0.9,
                   ls=(0, (4, 1.5))),
    "indep": dict(label="Per-move pricing", color="#d0514f", marker="^", ms=2.6, lw=0.9,
                  ls=(0, (1.2, 1.2))),
}
# datasets: the first four keep Exp-4's colors and markers; YouTube / GPlus add two more
GRAPH = {
    "youtube": ("YouTube", "#1f4e79", "o"),
    "gplus": ("GPlus", "#00838f", "p"),
    "epinions": ("Epinions", "#a4343a", "D"),
    "dblp": ("DBLP", "#5b8f35", "^"),
    "cit-hepth": ("Cit-HepTh", "#e8871a", "s"),
    "facebook": ("Facebook", "#6b4c9a", "v"),
}
R = Path(DYLO_ROOT + "/results")


def style(ax, grid="both"):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, length=1.5, pad=1.5, labelsize=5.5)
    ax.grid(axis=grid, color=GRID, lw=0.4, zorder=0)
    ax.set_axisbelow(True)


def load(name):
    D = collections.defaultdict(dict)
    p = R / name
    if not p.exists():
        return D
    for r in csv.DictReader(open(p)):
        if r["status"] == "ok":
            D[r["dataset"]][(r["mode"], r["lam"])] = {k: float(r[k]) for k in
                                                      ("Q", "cost", "moves", "sumDF")}
    return D


def pct(d, m, lam):
    a, b = d.get((m, lam)), d.get(("dp", lam))
    if not a or not b:
        return None
    return 100.0 if b["sumDF"] <= 0 else 100.0 * a["sumDF"] / b["sumDF"]


LAMS = ["1e-4", "3e-4", "1e-3", "3e-3", "1e-2", "3e-2", "1e-1"]


def fig_objective(D, out):
    fig, axes = plt.subplots(1, 2, figsize=(3.35, 1.32), sharey=True)
    fig.subplots_adjust(left=0.105, right=0.99, bottom=0.21, top=0.76, wspace=0.12)
    for ax, m, title in zip(axes, ("indep", "greedy"),
                            ("(a) Per-move pricing", "(b) Per-destination greedy")):
        for g, (name, col, mk) in GRAPH.items():
            if g not in D:
                continue
            pts = [(float(l), pct(D[g], m, l)) for l in LAMS if pct(D[g], m, l) is not None]
            if pts:
                ax.plot(*zip(*pts), color=col, marker=mk, ms=2.3, lw=0.8, label=name)
        ax.set_xscale("log")
        ax.set_xlim(7e-5, 1.4e-1)
        ax.xaxis.set_major_locator(mt.FixedLocator([1e-4, 1e-3, 1e-2, 1e-1]))
        ax.xaxis.set_minor_locator(mt.NullLocator())
        ax.set_ylim(30, 104)
        ax.yaxis.set_major_locator(mt.FixedLocator([40, 60, 80, 100]))
        ax.set_xlabel(r"Cost rate $\lambda$", labelpad=1)
        ax.set_title(title, fontsize=6.5, pad=3)
        style(ax)
    axes[0].set_ylabel(r"$\sum\Delta F$ (% of tree DP)", labelpad=1.5)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=6, frameon=False, fontsize=5.5,
               bbox_to_anchor=(0.54, 1.0), handlelength=1.6, columnspacing=0.9,
               handletextpad=0.4)
    fig.savefig(out / "sel_objective.pdf")
    fig.savefig(out / "sel_objective.png", dpi=300)
    plt.close(fig)


def fig_pareto(D, graphs, out, fname):
    graphs = [g for g in graphs if g in D]
    if not graphs:
        return
    fig, axes = plt.subplots(1, len(graphs), figsize=(3.35, 1.32), squeeze=False)
    axes = axes[0]
    fig.subplots_adjust(left=0.1, right=0.975, bottom=0.21, top=0.77, wspace=0.45)
    for i, (ax, g) in enumerate(zip(axes, graphs)):
        d = D[g]
        base = d[("dp", "1e-1")]["Q"]
        for zo, m in zip((3, 2, 1), ("dp", "greedy", "indep")):
            # connect points in lambda order: when Q is not monotone in cost, sorting by cost would draw a zigzag
            pts = [(r["cost"], (r["Q"] - base) * 1e4) for _lam, r in
                   sorted(((float(l), r) for (mm, l), r in d.items() if mm == m and l != "0"),
                          key=lambda t: t[0])]
            s = SEL[m]
            ax.plot(*zip(*pts), color=s["color"], marker=s["marker"], ms=s["ms"], lw=s["lw"],
                    ls=s["ls"], label=s["label"], zorder=zo)
        ax.xaxis.set_major_locator(mt.MaxNLocator(3))
        ax.yaxis.set_major_locator(mt.MaxNLocator(4))
        ax.set_title(f"({chr(97 + i)}) {GRAPH[g][0]}", fontsize=6.5, pad=3)
        ax.set_xlabel("Cost", labelpad=1)
        style(ax)
    axes[0].set_ylabel(r"$\Delta\bar Q$ ($\times10^{-4}$)", labelpad=1.5)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=3, frameon=False, fontsize=5.5,
               bbox_to_anchor=(0.54, 1.0), handlelength=2.2, columnspacing=1.2)
    fig.savefig(out / f"{fname}.pdf")
    fig.savefig(out / f"{fname}.png", dpi=300)
    plt.close(fig)


def _dp_frontier(d):
    """Upper envelope of DP in the (cost, Q) plane; returns a function interpolating Q by cost."""
    pts = sorted((r["cost"], r["Q"]) for (m, l), r in d.items() if m == "dp" and l != "0")
    env = []
    for c, q in pts:                      # keep the running maximum of Q as cost increases
        q = max(q, env[-1][1]) if env else q
        env.append((c, q))

    def f(c):
        if c < env[0][0] * 0.98 or c > env[-1][0] * 1.02:   # allow 2% extrapolation at the ends
            return None
        c = min(max(c, env[0][0]), env[-1][0])
        for (c0, q0), (c1, q1) in zip(env, env[1:]):
            if c0 <= c <= c1:
                return q0 if c1 == c0 else q0 + (q1 - q0) * (c - c0) / (c1 - c0)
        return env[-1][1]
    return f, env


def fig_gap(D, graphs, out, fname):
    """Q difference to the DP frontier at equal cost (x1e-4): magnifies the differences to a readable scale."""
    graphs = [g for g in graphs if g in D]
    fig, axes = plt.subplots(1, len(graphs), figsize=(3.35, 1.18), squeeze=False, sharey=True)
    axes = axes[0]
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.25, top=0.74, wspace=0.12)
    lo_all, hi_all = 0.0, 0.0
    for i, (ax, g) in enumerate(zip(axes, graphs)):
        d = D[g]
        f, env = _dp_frontier(d)
        ax.axhline(0, color=SEL["dp"]["color"], lw=SEL["dp"]["lw"], zorder=3)
        ax.plot([c for c, _ in env], [0] * len(env), ls="-", lw=SEL["dp"]["lw"], marker="o",
                ms=SEL["dp"]["ms"], color=SEL["dp"]["color"], zorder=4, label=SEL["dp"]["label"])
        for m in ("greedy", "indep"):
            pts = []
            for lam, r in sorted(((float(l), r) for (mm, l), r in d.items() if mm == m and l != "0"),
                                 key=lambda t: t[0]):
                q = f(r["cost"])
                if q is not None:
                    pts.append((r["cost"], (r["Q"] - q) * 1e4))
            s = SEL[m]
            if pts:
                ax.plot(*zip(*pts), color=s["color"], marker=s["marker"], ms=s["ms"], lw=s["lw"],
                        ls=s["ls"], label=s["label"], zorder=2)
                lo_all = min(lo_all, min(y for _, y in pts))
                hi_all = max(hi_all, max(y for _, y in pts))
        ax.set_xscale("log")
        inr = lambda ts: [t for t in ts if env[0][0] * 0.93 <= t <= env[-1][0] * 1.08]  # noqa: E731
        ticks = inr((0.05, 0.1, 0.2, 0.4))
        if len(ticks) < 3:
            ticks = inr((0.05, 0.1, 0.15, 0.2, 0.3, 0.4))
        ax.xaxis.set_major_locator(mt.FixedLocator(ticks))
        ax.xaxis.set_major_formatter(mt.FuncFormatter(lambda v, _p: f"{v:g}"))
        ax.xaxis.set_minor_locator(mt.NullLocator())
        ax.set_xlim(env[0][0] * 0.93, env[-1][0] * 1.08)
        ax.set_title(f"({chr(97 + i)}) {GRAPH[g][0]}", fontsize=6.5, pad=3)
        ax.set_xlabel("Cost", labelpad=1)
        style(ax)
    axes[0].set_ylim(lo_all * 1.15, max(hi_all * 1.3, -lo_all * 0.12))
    axes[0].yaxis.set_major_locator(mt.MaxNLocator(4))
    axes[0].set_ylabel(r"$\bar Q-\bar Q_{\mathrm{DP}}$ ($\times10^{-4}$)", labelpad=1.5)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=3, frameon=False, fontsize=5.5,
               bbox_to_anchor=(0.54, 1.0), handlelength=2.2, columnspacing=1.2)
    fig.savefig(out / f"{fname}.pdf")
    fig.savefig(out / f"{fname}.png", dpi=300)
    plt.close(fig)


def tex_objective(D, out):
    show = ["1e-4", "1e-3", "1e-2", "1e-1"]
    order = [g for g in GRAPH if g in D]
    L = ["% generated by sel_figs.py: selector ablation, native 1000 edges/batch",
         "\\begin{table}[t]", "\\centering",
         "\\caption{Realized objective of per-destination greedy (G) and per-move pricing (P), "
         "in \\% of the tree DP's $\\sum\\Delta F$.}",
         "\\label{tab:sel-objective}",
         "\\begin{tabular*}{\\linewidth}{@{\\extracolsep{\\fill}}l" + "rr" * len(show) + "@{}}",
         "\\toprule",
         " & " + " & ".join("\\multicolumn{2}{c}{$\\lambda=10^{-%d}$}" % int(l.split("e-")[1])
                            for l in show) + " \\\\",
         "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(l){8-9}",
         "\\textbf{Graph} & " + " & ".join(["\\textbf{G}", "\\textbf{P}"] * len(show)) + " \\\\",
         "\\midrule"]
    for g in order:
        cells = []
        for l in show:
            for m in ("greedy", "indep"):
                v = pct(D[g], m, l)
                cells.append("--" if v is None else (f"{v:.0f}" if v < 99.95 else "100"))
        L.append(f"{GRAPH[g][0]} & " + " & ".join(cells) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular*}", "\\end{table}"]
    (out / "tab_sel_objective.tex").write_text("\n".join(L) + "\n", encoding="utf-8")


def tex_batch(D1, DB, out):
    rows = [("epinions", ["", "_x10", "_x41"]), ("youtube", ["", "_x10", "_x30"]),
            ("dblp", ["", "_x10", "_x30"])]
    L = ["% generated by sel_figs.py: enlarged batches (consecutive batches merged)",
         "\\begin{table}[t]", "\\centering",
         "\\caption{Realized objective in \\% of the tree DP when consecutive batches are merged "
         "(G: per-destination greedy; P: per-move pricing).}",
         "\\label{tab:sel-batch}",
         "\\begin{tabular*}{\\linewidth}{@{\\extracolsep{\\fill}}lrrrrr@{}}",
         "\\toprule",
         " & & \\multicolumn{2}{c}{$\\lambda=10^{-2}$} & \\multicolumn{2}{c}{$\\lambda=10^{-1}$} \\\\",
         "\\cmidrule(lr){3-4}\\cmidrule(l){5-6}",
         "\\textbf{Graph} & \\textbf{Edges/batch} & \\textbf{G} & \\textbf{P} & \\textbf{G} & \\textbf{P} \\\\",
         "\\midrule"]
    for g, sufs in rows:
        for s in sufs:
            d = (D1 if s == "" else DB).get(g + s)
            if not d:
                continue
            k = 1 if s == "" else int(s[2:])
            cells = []
            for l in ("1e-2", "1e-1"):
                for m in ("greedy", "indep"):
                    v = pct(d, m, l)
                    cells.append("--" if v is None else (f"{v:.0f}" if v < 99.95 else "100"))
            L.append(f"{GRAPH[g][0] if s == '' else ''} & {1000 * k:,} & " + " & ".join(cells) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular*}", "\\end{table}"]
    (out / "tab_sel_batch.tex").write_text("\n".join(L).replace(",", "{,}") + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(R / "sel_ablation"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    D = load("sel_ablation.csv")
    DB = load("sel_ablation_bigbatch.csv")
    fig_objective(D, out)
    fig_pareto(D, ["youtube", "gplus", "facebook"], out, "sel_pareto")
    fig_gap(D, ["epinions", "dblp", "cit-hepth"], out, "sel_pareto_app")
    tex_objective(D, out)
    tex_batch(D, DB, out)
    print("->", out, sorted(os.listdir(out)))


if __name__ == "__main__":
    main()
