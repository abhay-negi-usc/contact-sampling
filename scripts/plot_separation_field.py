#!/usr/bin/env python3
"""Rotating 3-D GIFs of the hook/eyelet separation field, for every triple of pose coordinates.

The field (build_separation_field.py) holds the hook-eyelet separation at hook poses x, y, z [mm]
and rotation vector rx, ry, rz [deg] (shown as roll, pitch and yaw; for this tube's ~90 deg of
pitch, Euler angles would be near gimbal lock). Only the 1 mm / 1 deg grid is plotted: it covers
the tube uniformly, so point density reflects pose-space volume, whereas the half-spacing
refinement covers only part of it. For display, each node is spread uniformly over its own grid
cell, so the grid does not show as a lattice.

Each of the 20 coordinate triples gets two GIFs, the other three coordinates projected out:
    density/      collision-free poses within --near-mm of contact, faint and uniform, so the
                  density shows where near-contact poses concentrate
    separation/   collision-free poses within --far-mm of contact, darker and more opaque the
                  closer to contact
The reference trajectory is drawn in orange, the assembled pose as a dark dot. Each folder also
gets overview.png, the first frame of all 20 triples.

    conda run -n mujoco_env python scripts/plot_separation_field.py FIELD_DIR --out-dir OUT_DIR
"""

import argparse
import itertools
import multiprocessing as mp
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgba
from matplotlib.lines import Line2D
from scipy.spatial.transform import Rotation

from build_separation_field import B, TUBE_OUT, UNIT_MM
from hook_eyelet_model import ASSET_DIR, load_trajectory

NAMES = ["x", "y", "z", "roll", "pitch", "yaw"]
LABELS = ["x [mm]", "y [mm]", "z [mm]", "roll (rx) [deg]", "pitch (ry) [deg]", "yaw (rz) [deg]"]
SURFACE, INK, INK_2 = "#fcfcfb", "#0b0b0b", "#52514e"
POINT_BLUE, PATH_ORANGE = "#2a78d6", "#eb6834"
# Sequential blue, step 250 (far) to 700 (near contact); 250 is the lightest that stays visible.
RAMP = LinearSegmentedColormap.from_list("blue_ramp", ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281",
                                                       "#0d366b"])


def sample_part(job):
    """Collision-free nodes of one coarse part within far_mm of contact, kept with probability keep.

    Also returns how many such nodes there are in total, and how many within near_mm.
    """
    path, near_mm, far_mm, keep, seed, spacing = job
    rng = np.random.default_rng(seed)
    local = np.indices((B,) * 6).reshape(6, -1).T
    feats, seps, total, total_near = [], [], 0, 0
    with np.load(path) as z:
        for b, v in zip(z["blocks"], z["values"]):
            v = v.reshape(-1)
            sel = (v != TUBE_OUT) & (v >= 0) & (v <= far_mm / UNIT_MM)
            total += int(sel.sum())
            total_near += int(np.sum(sel & (v <= near_mm / UNIT_MM)))
            sel &= rng.random(len(v)) < keep
            feats.append(((b * B + local[sel]) * spacing).astype(np.float32))
            seps.append(v[sel] * UNIT_MM)
    return np.vstack(feats), np.concatenate(seps), total, total_near


def style_axes(ax, dims):
    ax.set_facecolor(SURFACE)
    for d, setter in zip(dims, (ax.set_xlabel, ax.set_ylabel, ax.set_zlabel)):
        setter(LABELS[d], color=INK_2, fontsize=9, labelpad=6)
    ax.tick_params(colors=INK_2, labelsize=7)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor(SURFACE)
        axis.pane.set_edgecolor("#e0dfdb")
        axis._axinfo["grid"].update(color="#ecebe7", linewidth=0.5)


def draw(ax, dims, pts, rgba, size, ref, legend=True):
    a, b, c = dims
    ax.computed_zorder = False  # keep the trajectory and the assembled pose on top of the cloud
    ax.scatter(pts[:, a], pts[:, b], pts[:, c], c=rgba, s=size, linewidths=0, depthshade=False, zorder=1)
    ax.plot(ref[:, a], ref[:, b], ref[:, c], color=PATH_ORANGE, linewidth=2, zorder=3)
    ax.scatter([ref[-1, a]], [ref[-1, b]], [ref[-1, c]], color=INK, s=24, depthshade=False, zorder=4)
    style_axes(ax, dims)
    if legend:  # stand-in handles: the field's own markers are too faint to read in a legend
        handles = [Line2D([], [], marker="o", linestyle="", color=POINT_BLUE, markersize=4, label="field poses"),
                   Line2D([], [], color=PATH_ORANGE, linewidth=2, label="reference trajectory"),
                   Line2D([], [], marker="o", linestyle="", color=INK, markersize=5, label="assembled pose")]
        ax.legend(handles=handles, loc="upper left", fontsize=7, frameon=False, labelcolor=INK_2)


def render(job):
    dims, version, pts, rgba, size, ref, title, subtitle, colorbar, out, frames, fps = job
    fig = plt.figure(figsize=(6.4, 6.4), dpi=100, facecolor=SURFACE)
    ax = fig.add_subplot(projection="3d")
    draw(ax, dims, pts, rgba, size, ref)
    fig.suptitle(title, color=INK, fontsize=11, y=0.97)
    fig.text(0.5, 0.915, subtitle, color=INK_2, fontsize=8, ha="center")
    if colorbar is not None:
        cax = fig.add_axes([0.25, 0.06, 0.5, 0.015])
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=colorbar, cmap=RAMP.reversed()), cax=cax,
                          orientation="horizontal")
        cb.set_label("separation [mm] (darker and more opaque: closer to contact)", color=INK_2, fontsize=8)
        cb.ax.tick_params(colors=INK_2, labelsize=7)
    anim = FuncAnimation(fig, lambda i: ax.view_init(elev=20, azim=-60 + 360 * i / frames), frames=frames)
    anim.save(out, writer=PillowWriter(fps=fps), savefig_kwargs={"facecolor": SURFACE})
    plt.close(fig)
    return out


def overview(version_dir, combos, pts, rgba, size, ref, title):
    fig = plt.figure(figsize=(20, 16), dpi=80, facecolor=SURFACE)
    for k, dims in enumerate(combos):
        ax = fig.add_subplot(4, 5, k + 1, projection="3d")
        draw(ax, dims, pts, rgba, size, ref, legend=(k == 0))
        ax.view_init(elev=20, azim=-60)
    fig.suptitle(title, color=INK, fontsize=16)
    fig.savefig(version_dir / "overview.png", facecolor=SURFACE)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("field_dir", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=ASSET_DIR / "hook_assembly_max_separation_trajectory.csv")
    parser.add_argument("--near-mm", type=float, default=1.0)
    parser.add_argument("--far-mm", type=float, default=5.0)
    parser.add_argument("--points", type=int, default=200_000, help="points per plot")
    parser.add_argument("--frames", type=int, default=72)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--only", nargs=3, choices=NAMES, help="render just this triple (for a quick look)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    spacing = 1.0
    parts = sorted(p for p in (args.field_dir / "coarse").glob("part_*.npz") if not p.name.endswith(".tmp.npz"))
    keep = 0.03  # pool fraction, thinned to --points per plot below
    with mp.get_context("fork").Pool(args.workers) as pool:
        res = pool.map(sample_part, [(p, args.near_mm, args.far_mm, keep, args.seed + i, spacing)
                                     for i, p in enumerate(parts)])
    feat = np.vstack([r[0] for r in res])
    sep = np.concatenate([r[1] for r in res])
    # Grid nodes plot as a lattice; spread each over its own cell so density reads as density.
    feat += np.random.default_rng(args.seed).uniform(-spacing / 2, spacing / 2, feat.shape).astype(np.float32)
    n_far, n_near = sum(r[2] for r in res), sum(r[3] for r in res)
    print(f"{len(parts)} parts: {n_far} collision-free grid poses within {args.far_mm} mm of contact, "
          f"{n_near} within {args.near_mm} mm; {len(feat)} pooled", flush=True)

    _, pos, quat = load_trajectory(args.reference)
    ref = np.hstack([pos * 1e3, Rotation.from_quat(quat, scalar_first=True).as_rotvec(degrees=True)])
    rng = np.random.default_rng(args.seed)

    near = np.flatnonzero(sep <= args.near_mm)
    near = rng.choice(near, min(args.points, len(near)), replace=False)
    density_rgba = np.tile(to_rgba(POINT_BLUE, alpha=0.04), (len(near), 1))

    far = rng.choice(len(sep), min(args.points, len(sep)), replace=False)
    far = far[np.argsort(-sep[far])]  # nearest drawn last
    norm = Normalize(0.0, args.far_mm)
    sep_rgba = RAMP(1.0 - norm(sep[far]))
    sep_rgba[:, 3] = 0.02 + 0.10 * (1.0 - norm(sep[far])) ** 1.5

    combos = list(itertools.combinations(range(6), 3))
    if args.only:
        combos = [tuple(sorted(NAMES.index(n) for n in args.only))]
    versions = {
        "density": (feat[near], density_rgba, 1.5,
                    f"poses within {args.near_mm:g} mm of contact: {len(near):,} of {n_near:,} on the 1 mm/1° grid; "
                    "opacity shows density", None),
        "separation": (feat[far], sep_rgba, 1.5,
                       f"poses 0-{args.far_mm:g} mm from contact: {len(far):,} of {n_far:,} on the 1 mm/1° grid",
                       norm),
    }
    jobs = []
    for version, (pts, rgba, size, subtitle, colorbar) in versions.items():
        vdir = args.out_dir / version
        vdir.mkdir(parents=True, exist_ok=True)
        for dims in combos:
            title = f"Hook pose field: {', '.join(NAMES[d] for d in dims)}"
            out = vdir / f"{'_'.join(NAMES[d] for d in dims)}.gif"
            jobs.append((dims, version, pts, rgba, size, ref, title, subtitle, colorbar, out, args.frames, args.fps))
    with mp.get_context("fork").Pool(min(args.workers, len(jobs))) as pool:
        for out in pool.imap_unordered(render, jobs):
            print(f"wrote {out}", flush=True)
    if not args.only:
        for version, (pts, rgba, size, subtitle, _) in versions.items():
            overview(args.out_dir / version, combos, pts, rgba, size * 0.6, ref,
                     f"Hook pose field, all coordinate triples ({version}): {subtitle}")
            print(f"wrote {args.out_dir / version / 'overview.png'}", flush=True)


if __name__ == "__main__":
    main()
