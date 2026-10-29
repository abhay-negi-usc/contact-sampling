#!/usr/bin/env python3
"""Compare the connectivity-graph variants of mode_graph_experiments.py: a dot plot of each variant's
agreement with the physical labels (NMI over all nodes, and over nodes near contact) and its local
disagreement, at a fixed k and at the variant's own k (eigengap, or basin count), after smoothing.

    python scripts/plot_mode_experiments.py EXPERIMENT_DIR [--k 16]
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#ecebe7"
FIXED, OWN = "#2a78d6", "#eb6834"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experiment_dir", type=Path)
    parser.add_argument("--k", type=int, default=16)
    args = parser.parse_args()
    rows = json.loads((args.experiment_dir / "experiments.json").read_text())["rows"]
    names = list(dict.fromkeys(r["variant"] for r in rows))
    pick = lambda name, own: next((r for r in rows if r["variant"] == name and r["smoothed"] == ("basins" not in name)
                                   and (r["k_is_eigengap"] if own else r["k"] == args.k)), None)
    fixed = {n: pick(n, False) for n in names}
    own = {n: pick(n, True) for n in names}
    rank = lambda r: r["nmi_full"] if math.isnan(r["nmi_near"]) else r["nmi_near"]
    names.sort(key=lambda n: rank(fixed[n] or own[n]))
    metrics = [("nmi_full", "NMI with physical label, all nodes"), ("nmi_near", "NMI, nodes within 1 mm of contact"),
               ("disagreement", "local disagreement (lower is better)")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 0.32 * len(names) + 1.6), sharey=True, facecolor=SURFACE)
    for ax, (key, title) in zip(axes, metrics):
        for y, n in enumerate(names):
            pts = [(r[key], c) for r, c in ((fixed[n], FIXED), (own[n], OWN)) if r is not None]
            if len(pts) == 2:
                ax.plot([pts[0][0], pts[1][0]], [y, y], color="#dcdbd5", linewidth=1, zorder=1)
            for x, c in pts:
                ax.scatter(x, y, s=42, color=c, edgecolors=SURFACE, linewidths=1.5, zorder=2)
        ax.set_title(title, color=INK, fontsize=9, loc="left")
        ax.set_facecolor(SURFACE)
        ax.grid(axis="x", color=GRID, linewidth=0.6)
        ax.tick_params(colors=INK_2, labelsize=8, length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color("#b9b8b1")
    axes[0].set_ylim(-0.7, len(names) - 0.3)
    axes[0].set_yticks(range(len(names)))
    axes[0].set_yticklabels([f"{n}  (own k {own[n]['k']})" if own[n] else n for n in names], color=INK, fontsize=7.5)
    fig.legend(handles=[Line2D([], [], marker="o", linestyle="", color=FIXED, label=f"k = {args.k}"),
                        Line2D([], [], marker="o", linestyle="", color=OWN, label="own k (eigengap, or basin count)")],
               loc="upper right", ncol=2, frameon=False, fontsize=8, labelcolor=INK_2)
    fig.suptitle("Connectivity-graph variants for hook pose modes (smoothed labels)", color=INK, fontsize=11, x=0.01,
                 ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(args.experiment_dir / "variant_comparison.png", facecolor=SURFACE, dpi=110)
    print(args.experiment_dir / "variant_comparison.png")


if __name__ == "__main__":
    main()
