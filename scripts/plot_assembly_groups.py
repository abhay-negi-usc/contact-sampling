#!/usr/bin/env python3
"""Rotating 3D GIF of the assembly groups (assembly_groups.py): contact poses coloured by the group of their
assembly trajectory, with a few of each group's trajectories drawn to the assembled pose, in the reduced
coordinates (lateral offset rho, radial tilt, depth).

    conda run -n mujoco_env python scripts/plot_assembly_groups.py --dir OUT_DIR
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D

from contact_mode_graph import COLORS, INK, MUTED, OTHER


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", type=Path, required=True, help="assembly_groups.py output directory")
    parser.add_argument("--names", nargs="*", help="a name per group, in the order of their size")
    parser.add_argument("--paths-per-group", type=int, default=6)
    parser.add_argument("--alpha", type=float, default=0.12)
    parser.add_argument("--frames", type=int, default=72)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--dpi", type=int, default=80)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    r, g = np.load(args.dir / "roadmap.npz"), np.load(args.dir / "groups.npz")
    q = r["nodes"][:int(r["n_contact"])]
    group, reps, paths, group_rep = g["group"], g["reps"], g["paths"], g["group_rep"]
    k = int(group_rep.max() + 1)
    order = np.argsort(-np.bincount(group, minlength=k + 1)[:k])  # colour by size, largest first
    colour = {int(gr): COLORS[i] if i < len(COLORS) else OTHER for i, gr in enumerate(order)}
    names = args.names or [f"group {i + 1}" for i in range(k)]
    rng = np.random.default_rng(args.seed)

    fig = plt.figure(figsize=(9, 7.5))
    fig.patch.set_facecolor("white")
    ax = fig.add_subplot(projection="3d")
    for gr in order:
        sel = group == gr
        ax.scatter(q[sel, 0], q[sel, 1], q[sel, 3], s=1.2, c=colour[int(gr)], alpha=args.alpha, linewidths=0, rasterized=True)
        mine = np.flatnonzero(group_rep == gr)
        for i in rng.choice(mine, min(args.paths_per_group, len(mine)), replace=False):
            p = paths[i]
            ax.plot(p[:, 0], p[:, 1], p[:, 3], color=colour[int(gr)], lw=1.4, alpha=0.9)
            ax.scatter(*p[-1, [0, 1, 3]], s=18, color=colour[int(gr)], edgecolors="white", linewidths=0.8)
    ax.scatter([0], [0], [50], s=120, marker="*", color=INK, zorder=5)
    ax.set_xlabel("lateral offset ρ [mm]", color=MUTED, fontsize=9)
    ax.set_ylabel("radial tilt [°]", color=MUTED, fontsize=9)
    ax.set_zlabel("depth [mm]", color=MUTED, fontsize=9)
    ax.invert_zaxis()
    ax.tick_params(colors=MUTED, labelsize=8)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_facecolor("white")
        pane.set_edgecolor("#d9d9d6")
    handles = [Line2D([], [], marker="o", ls="-", color=colour[int(gr)], label=f"{names[i]} ({np.mean(group == gr):.0%})")
               for i, gr in enumerate(order)]
    handles.append(Line2D([], [], marker="*", ls="", color=INK, markersize=10, label="assembled"))
    ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=9, labelcolor=INK)
    ax.set_title("Contact states grouped by the trajectory that assembles them\n(lines: assembly trajectories, dot: start)",
                 color=INK, fontsize=12)

    def frame(i):
        ax.view_init(elev=22, azim=-60 + 360 * i / args.frames)
        return []

    out = args.dir / "assembly_groups.gif"
    FuncAnimation(fig, frame, frames=args.frames).save(out, writer=PillowWriter(fps=args.fps), dpi=args.dpi)
    print(out)


if __name__ == "__main__":
    main()
