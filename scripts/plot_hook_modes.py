#!/usr/bin/env python3
"""Plot the hook pose modes found by hook_mode_graph.py in x, y, z, with a rendered example of each mode.

    modes_xyz.gif   rotating 3-D view of the samples' hook positions coloured by mode (the largest seven
                    modes in fixed categorical order, the rest grey), with the reference trajectory and
                    the pre-assembly pose
    modes_xyz.png   one small panel per mode, its samples in colour over all samples in grey
    mode_poses.png  each mode's representative pose rendered (front and side views): the member sample
                    nearest the mode's mean pose

Positions are the hook frame's origin in the eyelet frame (mm); the rotation is not shown in x, y, z,
so modes that differ only in orientation overlap here - the rendered poses tell them apart.

    conda run -n mujoco_env python scripts/plot_hook_modes.py MODE_GRAPH_DIR
"""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D
from scipy.spatial.transform import Rotation

from build_separation_field import reference_path
from hook_eyelet_model import ASSET_DIR, build_model, set_hook_pose

HUES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
OTHER, CONTEXT, INK, INK_2, SURFACE, PATH = "#b9b8b1", "#dcdbd5", "#0b0b0b", "#52514e", "#fcfcfb", "#0b0b0b"
SCALE = np.array([1, 1, 1] + [30.0 * np.pi / 180] * 3)


def mode_colours(labels, k):
    order = np.argsort(-np.bincount(labels[labels >= 0], minlength=k))
    return order, {m: (HUES[r] if r < len(HUES) else OTHER) for r, m in enumerate(order)}


def style(ax):
    ax.set_facecolor(SURFACE)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor(SURFACE)
        axis.pane.set_edgecolor("#e0dfdb")
        axis._axinfo["grid"].update(color="#ecebe7", linewidth=0.5)
    ax.set_xlabel("x [mm]", color=INK_2, fontsize=8)
    ax.set_ylabel("y [mm]", color=INK_2, fontsize=8)
    ax.set_zlabel("z [mm]", color=INK_2, fontsize=8)
    ax.tick_params(colors=INK_2, labelsize=6)


def render_pose(renderer, m, d, pose, azimuth):
    set_hook_pose(m, d, pose[:3] * 1e-3, Rotation.from_rotvec(pose[3:], degrees=True).as_quat(scalar_first=True))
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [pose[0] * 5e-4, 0.0, 0.004]
    cam.distance, cam.azimuth, cam.elevation = 0.13, azimuth, -15
    renderer.update_scene(d, camera=cam)
    return renderer.render()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode_dir", type=Path)
    parser.add_argument("--reference", type=Path, default=ASSET_DIR / "hook_assembly_max_separation_trajectory.csv")
    parser.add_argument("--points", type=int, default=40000)
    parser.add_argument("--frames", type=int, default=72)
    args = parser.parse_args()
    s = np.load(args.mode_dir / "samples.npz")
    poses, labels = s["poses"], s["labels"]
    report = json.loads((args.mode_dir / "modes.json").read_text())
    k = report["k"]
    goal_mode = report["goal_mode"]
    order, colour = mode_colours(labels, k)
    sizes = np.bincount(labels[labels >= 0], minlength=k)
    keep = labels >= 0
    rng = np.random.default_rng(0)
    show = rng.choice(np.flatnonzero(keep), min(args.points, keep.sum()), replace=False)
    ref = reference_path(args.reference, spacing=0.5)

    # rotating view, coloured by mode
    fig = plt.figure(figsize=(7.2, 6.6), dpi=100, facecolor=SURFACE)
    ax = fig.add_subplot(projection="3d")
    ax.computed_zorder = False
    ax.scatter(*poses[show, :3].T, c=[colour[m] for m in labels[show]], s=1.2, alpha=0.25, linewidths=0,
               depthshade=False, zorder=1)
    ax.plot(*ref[:, :3].T, color=PATH, linewidth=1.5, zorder=3)
    ax.scatter(0, 0, -3, color=INK, s=30, depthshade=False, zorder=4)
    style(ax)
    handles = [Line2D([], [], marker="o", linestyle="", color=colour[m], markersize=6,
                      label=f"mode {m}{' (goal)' if m == goal_mode else ''}: {sizes[m]:,}") for m in order[:len(HUES)]]
    if k > len(HUES):
        handles.append(Line2D([], [], marker="o", linestyle="", color=OTHER, markersize=6,
                              label=f"{k - len(HUES)} smaller modes: {sizes[order[len(HUES):]].sum():,}"))
    handles += [Line2D([], [], color=PATH, linewidth=1.5, label="reference trajectory"),
                Line2D([], [], marker="o", linestyle="", color=INK, markersize=5, label="pre-assembly pose")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=7, frameon=False, labelcolor=INK_2)
    fig.suptitle(f"Hook pose modes in x, y, z ({k} modes, {keep.sum():,} samples)", color=INK, fontsize=11)
    anim = FuncAnimation(fig, lambda i: ax.view_init(elev=20, azim=-60 + 360 * i / args.frames), frames=args.frames)
    anim.save(args.mode_dir / "modes_xyz.gif", writer=PillowWriter(fps=12), savefig_kwargs={"facecolor": SURFACE})
    plt.close(fig)

    # small multiples: one panel per mode over grey context
    cols = min(k, 5)
    rows = int(np.ceil(k / cols))
    fig = plt.figure(figsize=(3.4 * cols, 3.2 * rows), dpi=90, facecolor=SURFACE)
    ctx = rng.choice(np.flatnonzero(keep), min(8000, keep.sum()), replace=False)
    for r, m in enumerate(order):
        ax = fig.add_subplot(rows, cols, r + 1, projection="3d")
        ax.computed_zorder = False
        ax.scatter(*poses[ctx, :3].T, c=CONTEXT, s=0.6, alpha=0.3, linewidths=0, depthshade=False, zorder=1)
        mine = np.flatnonzero(labels == m)
        mine = rng.choice(mine, min(6000, len(mine)), replace=False)
        ax.scatter(*poses[mine, :3].T, c=colour[m] if colour[m] != OTHER else "#52514e", s=1.2, alpha=0.5,
                   linewidths=0, depthshade=False, zorder=2)
        ax.plot(*ref[:, :3].T, color=PATH, linewidth=1, zorder=3)
        style(ax)
        ax.set_title(f"mode {m}{' (goal)' if m == goal_mode else ''}: {sizes[m]:,} samples", color=INK, fontsize=9)
        ax.view_init(elev=20, azim=-60)
    fig.tight_layout()
    fig.savefig(args.mode_dir / "modes_xyz.png", facecolor=SURFACE)
    plt.close(fig)

    # mode centres in x, pitch, z: the member nearest each mode's mean pose (a mean can fall outside a
    # non-convex mode), sized by the mode's samples, joined where modes neighbour each other
    centre = {}
    for m in range(k):
        mine = np.flatnonzero(labels == m)
        centre[m] = poses[mine[np.argmin(np.linalg.norm((poses[mine] - poses[mine].mean(0)) * SCALE, axis=1))]]
    xpz = lambda p: (p[..., 0], p[..., 4], p[..., 2])
    fig = plt.figure(figsize=(7.4, 6.8), dpi=100, facecolor=SURFACE)
    ax = fig.add_subplot(projection="3d")
    ax.computed_zorder = False
    ctx2 = rng.choice(np.flatnonzero(keep), min(15000, keep.sum()), replace=False)
    ax.scatter(*xpz(poses[ctx2]), c=[colour[m] for m in labels[ctx2]], s=0.8, alpha=0.08, linewidths=0,
               depthshade=False, zorder=1)
    for e in report.get("mode_edges", []):
        a, b = e["modes"]
        ax.plot(*zip(xpz(centre[a]), xpz(centre[b])), color="#b9b8b1",
                linewidth=0.4 + 0.6 * np.log10(e["lines_of_sight"]), zorder=2)
    ax.plot(*xpz(ref), color=PATH, linewidth=2.2, zorder=3)
    for m in range(k):
        ax.scatter(*xpz(centre[m]), s=40 + 500 * np.sqrt(sizes[m] / sizes.max()), color=colour[m],
                   edgecolors=SURFACE, linewidths=1.5, depthshade=False, zorder=4)
        ax.text(*xpz(centre[m]), f"  {m}{' goal' if m == goal_mode else ''}", color=INK, fontsize=8, zorder=5)
    ax.scatter(0, 0, -3, color=INK, s=25, marker="x", depthshade=False, zorder=5)
    ax.set_facecolor(SURFACE)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor(SURFACE)
        axis.pane.set_edgecolor("#e0dfdb")
        axis._axinfo["grid"].update(color="#ecebe7", linewidth=0.5)
    ax.set_xlabel("x [mm]", color=INK_2, fontsize=8)
    ax.set_ylabel("pitch (ry) [deg]", color=INK_2, fontsize=8)
    ax.set_zlabel("z [mm]", color=INK_2, fontsize=8)
    ax.tick_params(colors=INK_2, labelsize=6)
    fig.suptitle(f"Mode centres in x, pitch, z ({k} modes; size: samples; lines: neighbouring modes)",
                 color=INK, fontsize=10)
    fig.legend(handles=[Line2D([], [], color=PATH, linewidth=2.2, label="reference trajectory"),
                        Line2D([], [], marker="x", linestyle="", color=INK, label="pre-assembly pose")],
               loc="lower center", ncol=2, fontsize=7, frameon=False, labelcolor=INK_2)
    ax.view_init(elev=20, azim=-60)
    fig.savefig(args.mode_dir / "mode_centres_xpz.png", facecolor=SURFACE)
    anim = FuncAnimation(fig, lambda i: ax.view_init(elev=20, azim=-60 + 360 * i / args.frames), frames=args.frames)
    anim.save(args.mode_dir / "mode_centres_xpz.gif", writer=PillowWriter(fps=12), savefig_kwargs={"facecolor": SURFACE})
    plt.close(fig)

    # representative pose of each mode, rendered
    m_, d_ = build_model(offscreen_size=(320, 320)), None
    d_ = mujoco.MjData(m_)
    renderer = mujoco.Renderer(m_, 320, 320)
    fig, axs = plt.subplots(rows, cols * 2, figsize=(2.4 * cols * 2, 2.6 * rows), dpi=90, facecolor=SURFACE)
    axs = np.atleast_2d(axs)
    for ax in axs.ravel():
        ax.axis("off")
    try:
        for r, m in enumerate(order):
            mine = np.flatnonzero(labels == m)
            rep = mine[np.argmin(np.linalg.norm((poses[mine] - poses[mine].mean(0)) * SCALE, axis=1))]
            p = poses[rep]
            for v, az in enumerate((90, 0)):
                ax = axs[r // cols, 2 * (r % cols) + v]
                ax.imshow(render_pose(renderer, m_, d_, p, az))
                if v == 0:
                    ax.set_title(f"mode {m}{' (goal)' if m == goal_mode else ''}\n"
                                 f"({p[0]:.0f}, {p[1]:.0f}, {p[2]:.0f}) mm, ({p[3]:.0f}, {p[4]:.0f}, {p[5]:.0f})°",
                                 color=INK, fontsize=7, loc="left")
    finally:
        renderer.close()
    fig.tight_layout()
    fig.savefig(args.mode_dir / "mode_poses.png", facecolor=SURFACE)
    plt.close(fig)
    print("wrote modes_xyz.gif, modes_xyz.png, mode_poses.png in", args.mode_dir)


if __name__ == "__main__":
    main()
