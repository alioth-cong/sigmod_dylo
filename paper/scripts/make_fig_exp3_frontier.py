"""Compact Exp-3 frontier panel for the five-panel experiment figure.

The plotted values are recovered to their original displayed precision from
figures/exp4_batch.pdf's vector paths (third panel), using its logarithmic
axis grid lines. This preserves all four curves and all sixteen observations
from the existing figure; no experiment is rerun or interpolated.

Usage: python scripts/make_fig_exp3_frontier.py OUTPUT_DIRECTORY
"""
from pathlib import Path
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mt

BATCH=[1,10,100,1000]
DELAY=[0.083,0.828,8.278,82.781]
SERIES=[
    ("MH eager", [113.5442,49.2932,26.4956,7.7388], "#1f4e79", "o", "-"),
    ("MH lazy", [115.5214,45.7760,25.2219,7.7208], "#1f4e79", "o", "--"),
    ("HepPh eager", [58.3974,57.8606,53.2983,26.3030], "#e8871a", "s", "-"),
    ("HepPh lazy", [33.4460,33.3073,31.4761,21.0177], "#e8871a", "s", "--"),
]
plt.rcParams.update({
    "font.family":"serif",
    "font.serif":["Times New Roman","Liberation Serif","DejaVu Serif"],
    "mathtext.fontset":"stix","pdf.fonttype":42,"font.size":6.5,
    "axes.linewidth":0.5,"xtick.major.width":0.5,"ytick.major.width":0.5,
})
out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=True)
fig,ax=plt.subplots(figsize=(1.36,1.3))
fig.subplots_adjust(left=0.27,right=0.985,bottom=0.27,top=0.97)
for label,y,color,marker,ls in SERIES:
    ax.plot(DELAY,y,label=label,color=color,marker=marker,ls=ls,lw=0.8,
            ms=2.2,mfc=color if ls=="-" else "white",mew=0.6,zorder=3)
ax.set_xscale("log");ax.set_yscale("log")
ax.set_xlim(.055,150);ax.set_ylim(6,180)
ax.set_xticks([.1,1,10,100]);ax.set_yticks([10,100])
ax.xaxis.set_minor_locator(mt.NullLocator())
ax.yaxis.set_minor_locator(mt.NullLocator())
ax.set_xlabel("Batch-closing delay (s)",labelpad=1)
ax.set_ylabel("Cost per edge",labelpad=2)
for side in ["top","right"]:ax.spines[side].set_visible(False)
for side in ["left","bottom"]:ax.spines[side].set_color("#52514e")
ax.tick_params(colors="#52514e",length=1.5,pad=1.5,labelsize=6)
ax.grid(color="#e4e3df",lw=.4,zorder=0);ax.set_axisbelow(True)
for x,y,b,offset in zip(DELAY,SERIES[0][1],BATCH,[(2,2),(1,3),(-1,4),(-13,3)]):
    ax.annotate(str(b),(x,y),xytext=offset,textcoords="offset points",
                fontsize=5,color="#1f4e79")
ax.legend(loc="lower left",frameon=False,fontsize=4.8,handlelength=1.4,
          handletextpad=.35,labelspacing=.2,borderpad=.1,borderaxespad=.15)
fig.savefig(out/"exp3_frontier.pdf")
fig.savefig(out/"exp3_frontier.png",dpi=400)
plt.close(fig)

