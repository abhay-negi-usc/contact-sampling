#!/usr/bin/env python3
"""Rotating 3D plot of contact samples coloured by contact mode, beside the graph of transitions
between the modes (the stratum graph of Kyaw & Kelly), or the same for the spectral clusters.

--by mode: each sample's analytical contact mode (peg_hole_analytic.py), grouped into the contact
    modes of the cylindrical peg and hole (MODES): four on top of the hole, the two one-point contacts
    in the bore, the two-point jam and the contacts with the floor. Rarer combinations fold into "other".
--by cluster: each sample's spectral cluster (run contact_graph_spectral.py first), named by the
    contact mode most of its samples hold.

Graph: a node per mode (or cluster), its area by its share of the samples; an edge wherever
trajectories step from one to the other (including a trajectory's first step from its parent sample),
its width by the log of the number of such steps. Nodes sit on a circle ordered by mean insertion depth.

The 3D points are thinned evenly in the reduced coordinates, as in plot_contact_samples_cylindrical.py.

    conda run -n mujoco_env python scripts/contact_mode_graph.py RUN_DIR --by mode
    conda run -n mujoco_env python scripts/contact_mode_graph.py RUN_DIR --by cluster
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

import peg_hole_analytic as A
from compare_samples_to_analytic import SAMPLE_TOL, SAMPLE_TOL_BORE
from contact_graph_spectral import load_run, transition_edges
from plot_contact_samples_cylindrical import subset

MODES = ["edge on inner rim", "edge on top face", "face on outer rim", "edge on outer rim", "side on rim",
         "edge on wall", "two-point jam", "on the floor"]
SINGLE = {"tip_edge/rim_in": 0, "tip_edge/top": 1, "tip_face/rim_out": 2, "tip_edge/rim_out": 3,
          "tip_side/rim_in": 4, "tip_edge/wall": 5}
# Categorical slots in fixed order; grey for whatever is folded away.
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
OTHER = "#a8a7a2"
INK, MUTED, GRID, EDGE = "#2b2b2b", "#6b6b6b", "#d9d9d6", "#9a9996"


def mode_category(mode):
    if "floor" in mode:
        return 7
    if "tip_edge/wall" in mode and "tip_side/rim_in" in mode:
        return 6
    return SINGLE.get(mode, len(MODES))


def sample_modes(run_dir, q, chunk=2_000_000):
    """Contact mode category (index into MODES, len(MODES) for other) of every sample, cached."""
    cache = Path(run_dir) / "contact_mode_category.npy"
    if cache.exists():
        return np.load(cache)
    out = np.empty(len(q), np.int8)
    for i in range(0, len(q), chunk):
        c, u = A.pose_from_reduced(q[i:i + chunk].astype(float))
        _, _, pairs = A.contact_modes(c, u, SAMPLE_TOL, SAMPLE_TOL_BORE)
        uniq, inv = np.unique(pairs, axis=0, return_inverse=True)
        cats = np.array([mode_category(A.mode_name(p)) for p in uniq], np.int8)
        out[i:i + chunk] = cats[inv.ravel()]
    np.save(cache, out)
    return out


def sample_clusters(run_dir, modes):
    """Spectral cluster of every sample (-1 outside the graph's largest component) and cluster names."""
    z = np.load(Path(run_dir) / "spectral" / "cells.npz")
    cl = z["cluster"][z["cell"]]
    k = cl.max() + 1
    names = []
    for c in range(k):
        counts = np.bincount(modes[cl == c], minlength=len(MODES) + 1)
        j = int(np.argmax(counts))
        label = MODES[j] if j < len(MODES) else "other"
        names.append(f"cluster {c}\n{counts[j] / counts.sum():.0%} {label}")
    return cl, names


def draw_graph(ax, names, colors, share, counts, order):
    """Nodes on a circle (in `order`, clockwise from the top), edges by log transition count."""
    n = len(names)
    angle = np.pi / 2 - 2 * np.pi * np.argsort(order) / n
    pos = np.c_[np.cos(angle), np.sin(angle)]
    top = counts.max()
    for i in range(n):
        for j in range(i + 1, n):
            if counts[i, j] > 0:
                w = 0.6 + 4.4 * np.log1p(counts[i, j]) / np.log1p(top)
                ax.plot(*pos[[i, j]].T, color=EDGE, linewidth=w, alpha=0.55, solid_capstyle="round", zorder=1)
    size = 100 + 900 * np.sqrt(share / share.max())
    ax.scatter(pos[:, 0], pos[:, 1], s=size, c=colors, edgecolors="white", linewidths=2, zorder=2)
    for i in range(n):
        x, y = 1.32 * pos[i]
        ha = "left" if x > 0.15 else "right" if x < -0.15 else "center"
        ax.text(x, y, f"{names[i]}\n{share[i]:.1%} of samples", ha=ha, va="center", fontsize=8, color=INK,
                linespacing=1.2)
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-1.7, 1.7)
    ax.set_aspect("equal")
    ax.axis("off")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--by", choices=["mode", "cluster"], default="mode")
    parser.add_argument("--out", type=Path, help="default: RUN_DIR/contact_<by>_graph.gif")
    parser.add_argument("--points", type=int, default=300_000)
    parser.add_argument("--alpha", type=float, default=0.06)
    parser.add_argument("--frames", type=int, default=72)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--dpi", type=int, default=80)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    out = args.out or args.run_dir / f"contact_{args.by}_graph.gif"

    q, traj, step, table = load_run(args.run_dir)
    modes = sample_modes(args.run_dir, q)
    if args.by == "mode":
        labels, names = modes.astype(np.int64), MODES + ["other"]
    else:
        labels, names = sample_clusters(args.run_dir, modes)
        labels = np.where(labels < 0, len(names), labels)
        names = names + ["outside the main component"]
    n = len(names)
    share = np.bincount(labels, minlength=n) / len(labels)
    present = np.flatnonzero(share > 0)

    a, b = transition_edges(labels, traj, step, table)
    counts = np.zeros((n, n))
    np.add.at(counts, (a, b), 1)
    counts = counts + counts.T
    mean_depth = np.bincount(labels, weights=q[:, 3], minlength=n) / np.maximum(np.bincount(labels, minlength=n), 1)
    if args.by == "mode":
        colors = COLORS + [OTHER]
        order = np.arange(n)  # MODES runs from the top of the hole down to the floor
    else:
        order = np.argsort(mean_depth[:-1])
        colors = [None] * n
        for slot, c in enumerate(order):
            colors[c] = COLORS[slot] if slot < len(COLORS) else OTHER
        colors[-1] = OTHER
        order = np.r_[order, n - 1]
    keep_nodes = [i for i in order if i in present]
    report = {names[i]: dict(share=float(share[i]), mean_depth_mm=float(mean_depth[i]),
                             transitions={names[j]: int(counts[i, j]) for j in keep_nodes if counts[i, j] > 0})
              for i in keep_nodes}
    (Path(str(out).replace(".gif", ".json"))).write_text(json.dumps(report, indent=1))
    print(f"{len(q)} samples; {n} {args.by}s; transitions between them:")
    for i in keep_nodes:
        print(f"  {share[i]:7.2%}  {names[i].replace(chr(10), ': '):45s} -> " + ", ".join(
            f"{names[j].splitlines()[0]} {int(counts[i, j])}" for j in keep_nodes if counts[i, j] > 0))

    idx = subset(q, args.points, [0.25] * 4, [0.05, 0.25, 0.25, 0.5], args.seed)
    pts, lab = q[idx].astype(float), labels[idx]

    fig = plt.figure(figsize=(14, 6.6), facecolor="white")
    ax3 = fig.add_axes([-0.02, 0.0, 0.56, 0.9], projection="3d")
    by_size = sorted(present, key=lambda i: -share[i])  # small groups drawn last, on top
    for i in by_size:
        sel = lab == i
        ax3.scatter(pts[sel, 0], pts[sel, 1], pts[sel, 3], s=1.2, c=colors[i], alpha=args.alpha, linewidths=0,
                    rasterized=True)
    ax3.set_zlim(pts[:, 3].max(), min(pts[:, 3].min(), -3.0))
    ax3.set_xlabel("tip radial offset ρ (mm)", color=MUTED, fontsize=9, labelpad=6)
    ax3.set_ylabel("radial tilt (°, + tip outward)", color=MUTED, fontsize=9, labelpad=6)
    ax3.set_zlabel("insertion depth (mm)", color=MUTED, fontsize=9, labelpad=6)
    ax3.tick_params(colors=MUTED, labelsize=8)
    for axis in (ax3.xaxis, ax3.yaxis, ax3.zaxis):
        axis.set_pane_color((1, 1, 1, 0))
        axis._axinfo["grid"].update(color=GRID, linewidth=0.5)
        axis.line.set_color(GRID)
    ax3.set_box_aspect(None, zoom=0.9)
    axg = fig.add_axes([0.54, 0.05, 0.45, 0.8])
    keep_order = [i for i in order if i in present]
    draw_graph(axg, [names[i] for i in keep_order], [colors[i] for i in keep_order], share[keep_order],
               counts[np.ix_(keep_order, keep_order)], np.arange(len(keep_order)))
    title = ("Contact modes of the peg and hole" if args.by == "mode"
             else "Spectral clusters of the contact-space connectivity graph")
    fig.suptitle(f"{title}: samples in symmetry-reduced coordinates (left) and transitions between them (right)",
                 color=INK, fontsize=12, y=0.97)

    def frame(i):
        ax3.view_init(elev=22, azim=-60 + 360 * i / args.frames)
        return []

    FuncAnimation(fig, frame, frames=args.frames).save(out, writer=PillowWriter(fps=args.fps), dpi=args.dpi)
    print(f"Wrote {args.frames} frames to {out}")


if __name__ == "__main__":
    main()
