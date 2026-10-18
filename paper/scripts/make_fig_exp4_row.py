"""Five small panels for the full-width Figure 6 row.

(a, b) Exp-4 exchange-rate sweep. The values were read back from the vector
       paths of the author's figures/exp5_knee_{a,b}.pdf (exact point
       coordinates mapped through the axis grid lines); they reproduce the
       appendix numbers, e.g. Cost(1e-2)/Cost(0) = 0.110--0.137 and
       Epinions' 0.804 at 1e-3.
(c)    Exp-2b frozen-tree selector runtime (appendix Table tab:selector-exact).
       Instances with n <= 22 are binned in the table and plotted at the bin
       bound.
(d)    Exp-2b real-stream selector timings (appendix table).
(e)    Reference-answer token-F1 (appendix table).

Usage: python scripts/make_fig_exp4_row.py <out_dir>
Writes exp4_cost.pdf, exp4_qloss.pdf, exp2b_scaling.pdf, exp2b_seltime.pdf,
gt_f1.pdf.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mt

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "font.size": 6.5,
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "hatch.linewidth": 0.4,
})

W, H = 1.36, 1.3
MUTED = "#52514e"
GRID = "#e4e3df"
INK = "#1a1a1a"
BLUE = dict(color="#d6e6f7", edgecolor="#3f7fc4", hatch="....", lw=0.6)
RED = dict(color="#f8d4d2", edgecolor="#d0514f", hatch="////", lw=0.6)
DP_C, CUT_C, ENUM_C = "#3f7fc4", "#d0514f", "#e8871a"

LAMBDAS = [1e-3, 1e-2, 1e-1, 0.3, 1, 3, 10]
SERIES = [  # name, color, marker (as in the original figure)
    ("Cit-HepPh", "#1f4e79", "o"),
    ("Cit-HepTh", "#e8871a", "s"),
    ("DBLP", "#5b8f35", "^"),
    ("Epinions", "#a4343a", "D"),
    ("Facebook", "#6b4c9a", "v"),
    ("Flickr", "#00838f", "p"),
]
COST = {
    "Cit-HepPh": [0.1152, 0.1125, 0.1125, 0.1125, 0.1125, 0.1125, 0.1125],
    "Cit-HepTh": [0.2545, 0.1167, 0.1167, 0.1167, 0.1167, 0.1167, 0.1167],
    "DBLP": [0.1490, 0.1375, 0.1375, 0.1375, 0.1375, 0.1375, 0.1375],
    "Epinions": [0.8043, 0.1101, 0.1095, 0.1094, 0.1094, 0.1094, 0.1094],
    "Facebook": [0.8164, 0.4019, 0.3974, 0.3974, 0.3974, 0.3974, 0.3974],
    "Flickr": [0.1363, 0.1303, 0.1301, 0.1301, 0.1301, 0.1301, 0.1301],
}
QLOSS = {  # [Q(0) - Q(lambda)] x 1e3
    "Cit-HepPh": [0.09, 0.09, 0.09, 0.09, 0.10, 0.10, 0.10],
    "Cit-HepTh": [0.09, 0.14, 0.14, 0.14, 0.14, 0.14, 0.14],
    "DBLP": [0.18, 0.19, 0.19, 0.19, 0.19, 0.19, 0.19],
    "Epinions": [0.08, 0.67, 0.69, 0.70, 0.70, 0.70, 0.70],
    "Facebook": [0.00, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06],
    "Flickr": [0.07, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08],
}
# Frozen-tree selector runtime (ms per solve).
N_SMALL = [8, 12, 16, 20, 22]
ENUM_T = [0.0021, 0.0242, 0.1927, 3.9625, 13.681]
CUT_S = [0.0007, 0.0010, 0.0012, 0.0018, 0.0021]
DP_S = [0.0004, 0.0004, 0.0005, 0.0009, 0.0011]
N_LARGE = [50, 200, 1000, 5000, 20000]
CUT_L = [0.0076, 0.0316, 0.1816, 1.0172, 5.6968]
DP_L = [0.0027, 0.0108, 0.0500, 0.2873, 1.2407]

out_dir = sys.argv[1] if len(sys.argv) > 1 else "."


def style(ax, grid="y"):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, length=1.5, pad=1.5, labelsize=6)
    ax.grid(axis=grid, color=GRID, lw=0.4, zorder=0)
    ax.set_axisbelow(True)


def panel():
    # One fixed axes box for every panel, so the five sit on a common baseline.
    fig, ax = plt.subplots(figsize=(W, H))
    fig.subplots_adjust(left=0.27, right=0.985, bottom=0.27, top=0.97)
    return fig, ax


def save(fig, name):
    fig.savefig(os.path.join(out_dir, name + ".pdf"))
    fig.savefig(os.path.join(out_dir, name + ".png"), dpi=300)
    plt.close(fig)


def sweep(data, ylabel, ylim, yticks, name, legend):
    fig, ax = panel()
    ax.axvline(1e-2, color=MUTED, lw=0.6, ls=(0, (1.5, 1.5)), zorder=1)
    for s, col, mk in SERIES:
        ax.plot(LAMBDAS, data[s], color=col, marker=mk, ms=2.2, lw=0.8,
                label=s, zorder=3)
    ax.set_xscale("log")
    ax.set_xticks([1e-3, 1e-1, 10])
    ax.xaxis.set_minor_locator(mt.NullLocator())
    ax.set_ylim(*ylim)
    ax.set_yticks(yticks)
    ax.set_xlabel(r"$\lambda$", labelpad=1)
    ax.set_ylabel(ylabel, labelpad=2)
    style(ax)
    ax.text(1.2e-2, ylim[1], "knee", fontsize=5.5, color=MUTED, va="top")
    if legend:
        ax.legend(loc="upper right", bbox_to_anchor=(1.02, 0.86), ncol=2,
                  frameon=False, fontsize=5, handlelength=0.9,
                  handletextpad=0.25, columnspacing=0.4, borderaxespad=0.0,
                  labelspacing=0.15)
    save(fig, name)


sweep(COST, r"Cost$(\lambda)$/Cost$(0)$", (0, 1.0), [0, 0.5, 1.0],
      "exp4_cost", legend=True)
sweep(QLOSS, r"$[Q(0)-Q(\lambda)]\times10^3$", (0, 0.8), [0, 0.4, 0.8],
      "exp4_qloss", legend=False)

# (c) frozen-tree selector runtime against tree size
fig, ax = panel()
ax.plot(N_SMALL, ENUM_T, color=ENUM_C, marker="^", ms=2.4, lw=0.9)
ax.plot(N_SMALL + N_LARGE, CUT_S + CUT_L, color=CUT_C, marker="s", ms=2.0,
        lw=0.9)
ax.plot(N_SMALL + N_LARGE, DP_S + DP_L, color=DP_C, marker="o", ms=2.2,
        lw=1.1)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(6, 9e4)
ax.set_ylim(2e-4, 60)
ax.xaxis.set_major_locator(mt.FixedLocator([10, 1e2, 1e3, 1e4]))
ax.yaxis.set_major_locator(mt.FixedLocator([1e-3, 1e-2, 1e-1, 1, 10]))
ax.xaxis.set_minor_locator(mt.NullLocator())
ax.yaxis.set_minor_locator(mt.NullLocator())
ax.set_xlabel(r"Tree size $n$", labelpad=1)
ax.set_ylabel("Time per solve (ms)", labelpad=2)
style(ax, grid="both")
ax.annotate("", xy=(25, DP_S[-1]), xytext=(25, ENUM_T[-1]),
            arrowprops=dict(arrowstyle="<->", color=ENUM_C, lw=0.5,
                            shrinkA=1, shrinkB=1))
ax.text(29, 0.75, r"12,699$\times$", color=ENUM_C, fontsize=5.5, va="center")
ax.annotate("", xy=(2.7e4, DP_L[-1]), xytext=(2.7e4, CUT_L[-1]),
            arrowprops=dict(arrowstyle="<->", color=CUT_C, lw=0.5,
                            shrinkA=1, shrinkB=1))
ax.text(2.7e4, 8, r"4.6$\times$", color=CUT_C, fontsize=5.5, ha="center",
        va="bottom")
ax.text(8, 22, "enumeration", color=ENUM_C, fontsize=5.5, ha="left",
        va="bottom")
ax.text(3000, 0.9, "min-cut", color=CUT_C, fontsize=5.5, ha="right",
        va="bottom")
ax.text(1500, 0.012, "tree DP", color=DP_C, fontsize=5.5, ha="left",
        va="top")
save(fig, "exp2b_scaling")

# (d) real-stream selection time, deferral off vs on
graphs = ["Epinions", "DBLP", "Flickr", "YouTube"]
off = [2.07, 0.71, 19.0, 155.4]
on = [0.19, 0.15, 1.90, 3.99]
bw = 0.38
fig, ax = panel()
for i in range(4):
    ax.bar(i - bw / 2, off[i], width=bw - 0.03, zorder=2, **RED)
    ax.bar(i + bw / 2, on[i], width=bw - 0.03, zorder=2, **BLUE)
    ax.text(i, off[i] * 1.5, f"{off[i] / on[i]:.0f}$\\times$", ha="center",
            va="bottom", fontsize=5.5, color=INK)
ax.set_yscale("log")
ax.set_ylim(0.08, 4000)
ax.set_yticks([0.1, 1, 10, 100])
ax.yaxis.set_minor_locator(mt.NullLocator())
ax.set_xticks(range(4))
ax.set_xticklabels(graphs, rotation=30, ha="right", rotation_mode="anchor",
                   fontsize=6)
ax.set_ylabel("Selection time (ms)", labelpad=2)
style(ax)
ax.bar(0, 0, label="Deferral off", **RED)
ax.bar(0, 0, label="Deferral on", **BLUE)
ax.legend(loc="upper center", ncol=2, frameon=False, fontsize=5.5,
          handlelength=0.9, handletextpad=0.3, columnspacing=0.6,
          borderaxespad=0.0)
save(fig, "exp2b_seltime")

# (e) reference-answer token-F1
methods = ["DyLo", "QCA", "FS-Lv", "FS-Ld", "Batch", "DynaMo", "GreMod"]
f1 = [0.1866, 0.1723, 0.1662, 0.1570, 0.1546, 0.1397, 0.0886]
fig, ax = panel()
for i, v in enumerate(f1):
    ax.bar(i, v, width=0.7, zorder=2, **(BLUE if i == 0 else RED))
ax.set_xticks(range(len(methods)))
ax.set_xticklabels(methods, rotation=45, ha="right", rotation_mode="anchor",
                   fontsize=6)
ax.set_ylim(0, 0.21)
ax.set_yticks([0, 0.1, 0.2])
ax.set_ylabel("Token $F_1$", labelpad=2)
style(ax)
save(fig, "gt_f1")
