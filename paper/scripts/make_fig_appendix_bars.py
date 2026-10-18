"""One single-column, two-panel figure built from appendix data.

Every value below is copied from appendix-e2e.tex (author-confirmed numbers):
  (a) Table "Exp-2b. Updated real-stream selector timings" (four graphs with a
      nontrivial selection workload; the others round to ~0 ms)
  (b) Table "Reference-answer token-F1 on MultiHop-RAG"
The structural-weight fidelity data are shown as a table in the main text.

Color semantics are shared across panels: blue = DyLo / our configuration,
orange = the alternative configuration or baselines.

Usage: python scripts/make_fig_appendix_bars.py <out_dir>
Writes dylo-appendix-panels.pdf (and a .png preview).
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

plt.rcParams.update({
    "font.family": "Times New Roman",
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "font.size": 6,
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
})

# Soft fills with a darker edge and a hatch, in the style of common systems
# papers: blue/dots = DyLo or tree DP; red/diagonals = min-cut or baselines.
BLUE = dict(color="#d6e6f7", edgecolor="#3f7fc4", hatch="....", lw=0.6)
RED = dict(color="#f8d4d2", edgecolor="#d0514f", hatch="////", lw=0.6)
plt.rcParams["hatch.linewidth"] = 0.4
INK = "#1a1a1a"
MUTED = "#52514e"
GRID = "#e4e3df"

out_dir = sys.argv[1] if len(sys.argv) > 1 else "."


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, length=1.5, pad=1.5, labelsize=5.5)
    ax.grid(axis="y", color=GRID, lw=0.4, zorder=0)
    ax.set_axisbelow(True)


fig, (b, c) = plt.subplots(1, 2, figsize=(3.35, 1.4),
                           gridspec_kw={"width_ratios": [1.0, 1.1],
                                        "wspace": 0.3})
bw = 0.38

# (a) selection time, deferral off vs on
graphs = ["Epinions", "DBLP", "Flickr", "YouTube"]
off = [2.07, 0.71, 19.0, 155.4]
on = [0.19, 0.15, 1.90, 3.99]
for i in range(4):
    b.bar(i - bw / 2, off[i], width=bw - 0.03, zorder=2, **RED)
    b.bar(i + bw / 2, on[i], width=bw - 0.03, zorder=2, **BLUE)
    b.text(i, off[i] * 1.5, f"{off[i] / on[i]:.0f}$\\times$", ha="center",
           va="bottom", fontsize=5.5, color=INK)
b.set_yscale("log")
b.set_ylim(0.08, 30000)
b.set_yticks([0.1, 1, 10, 100, 1000])
b.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
b.set_xticks(range(4))
b.set_xticklabels(graphs)
style(b)
b.bar(0, 0, label="min-cut", **RED)
b.bar(0, 0, label="tree DP", **BLUE)
b.legend(loc="upper center", ncol=2, frameon=False, fontsize=5.5,
         handlelength=0.8, handletextpad=0.3, columnspacing=0.8,
         borderaxespad=0.1)
b.set_title("(a) Selection time (ms)", fontsize=6.5, pad=3)

# (b) reference-answer token-F1
methods = ["DyLo", "QCA", "FS-Lv", "FS-Ld", "Batch", "DynaMo", "GreMod"]
f1 = [0.1866, 0.1723, 0.1662, 0.1570, 0.1546, 0.1397, 0.0886]
for i, v in enumerate(f1):
    c.bar(i, v, width=0.7, zorder=2, **(BLUE if i == 0 else RED))
c.set_xticks(range(len(methods)))
c.set_xticklabels(methods, rotation=45, ha="right", rotation_mode="anchor")
c.set_ylim(0, 0.21)
c.set_yticks([0, 0.1, 0.2])
c.set_yticklabels(["0", ".1", ".2"])
style(c)
c.set_title("(b) Token $F_1$", fontsize=6.5, pad=3)

fig.savefig(os.path.join(out_dir, "dylo-appendix-panels.pdf"),
            bbox_inches="tight", pad_inches=0.02)
fig.savefig(os.path.join(out_dir, "dylo-appendix-panels.png"), dpi=300,
            bbox_inches="tight", pad_inches=0.02)
