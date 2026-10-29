#!/usr/bin/env python3
"""Rotating 3D GIF of peg/hole contact samples in symmetry-reduced cylindrical coordinates.

The hole and the peg are both turned, so a contact pose rotated about the hole axis is the same
contact. The tip's azimuth therefore carries no information and is dropped: each sample is placed
by the cylindrical coordinates of the peg tip (radial offset rho and insertion depth) and by the
peg's tilt resolved along the tip's radial direction (positive: the tip end leans outward). The
remaining tilt component, across the radial plane, is the colour; mirroring through the radial
plane flips its sign only, so its magnitude is shown.

Two panels rotate together: the full depth of the hole, and the top of the hole.

The sampler spreads its effort over contact modes, so thin modes hold many more samples per unit of
contact space than wide ones. The plotted points are therefore thinned evenly in the reduced
coordinates (at most m samples per cell, m set by --points; cells of --balance-top on top of the hole
and --balance-bore inside it, where the clearance is 0.5 mm), so that point density shows how the
contact space is covered rather than where the sampler spent its effort.

    conda run -n mujoco_env python scripts/plot_contact_samples_cylindrical.py RUN_DIR [--points 300000]
"""

import argparse
import glob
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LinearSegmentedColormap
from scipy.spatial.transform import Rotation

from peg_hole_model import BORE_TOP_Y, PEG_TIP_Y

# Blue sequential ramp, steps 250..700: the lighter steps vanish at low opacity on white.
BLUES = LinearSegmentedColormap.from_list(
    "blues", ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281",
              "#0d366b"])
INK, MUTED, GRID = "#2b2b2b", "#6b6b6b", "#d9d9d6"


def reduced_coords(pose):
    """rho [mm], radial tilt [deg], |tangential tilt| [deg], depth [mm] of the peg tip, as an (n, 4) array."""
    axis = Rotation.from_quat(pose[:, 3:], scalar_first=True).apply([0.0, 1.0, 0.0])  # towards the tip
    tip = pose[:, :3] * 1e3 + axis * PEG_TIP_Y
    rho = np.hypot(tip[:, 0], tip[:, 2])
    phi = np.arctan2(tip[:, 2], tip[:, 0])
    radial = axis[:, 0] * np.cos(phi) + axis[:, 2] * np.sin(phi)
    tangential = -axis[:, 0] * np.sin(phi) + axis[:, 2] * np.cos(phi)
    down = -axis[:, 1]
    return np.stack([rho, np.degrees(np.arctan2(radial, down)), np.degrees(np.abs(np.arctan2(tangential, down))),
                     BORE_TOP_Y - tip[:, 1]], axis=1)


def load_reduced(run_dir):
    """Reduced coords of every sample, from the cache compare_samples_to_analytic.py writes if present."""
    cache = Path(run_dir) / "reduced_coords.npy"
    if cache.exists():
        return np.load(cache)
    files = sorted(glob.glob(str(Path(run_dir) / "shard_*.npz")))
    return np.concatenate([reduced_coords(np.load(f)["pose"].astype(float)).astype(np.float32) for f in files])


def subset(q, n_points, cell_top, cell_bore, seed):
    """Indices of about n_points samples with at most m in each cell (cell_top at depth <= 0, cell_bore
    below); a plain random subset if cell_top is None."""
    rng = np.random.default_rng(seed)
    if cell_top is None:
        return np.flatnonzero(rng.random(len(q)) < n_points / len(q))
    inside = q[:, 3] > 0
    size = np.where(inside[:, None], np.asarray(cell_bore, np.float32), np.asarray(cell_top, np.float32))
    key = np.column_stack([inside, np.floor(q / size).astype(np.int64)])
    _, cid = np.unique(key, axis=0, return_inverse=True)
    cid = cid.ravel()
    perm = rng.permutation(len(q))
    order = np.argsort(cid[perm], kind="stable")
    sorted_cells = cid[perm][order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_cells)) + 1]
    counts = np.diff(np.r_[starts, len(order)])
    rank = np.arange(len(order)) - np.repeat(starts, counts)  # position within the cell, in random order
    lo, hi = 1, int(counts.max())
    while lo < hi:  # smallest cap m that keeps at least n_points
        m = (lo + hi) // 2
        if np.minimum(counts, m).sum() >= n_points:
            hi = m
        else:
            lo = m + 1
    return perm[order[rank < lo]]


def style(ax, title, depth_lim):
    ax.set_title(title, color=INK, fontsize=11, pad=0)
    ax.set_xlabel("tip radial offset ρ (mm)", color=MUTED, fontsize=9, labelpad=6)
    ax.set_ylabel("radial tilt (°, + tip outward)", color=MUTED, fontsize=9, labelpad=6)
    ax.set_zlabel("insertion depth (mm)", color=MUTED, fontsize=9, labelpad=6)
    ax.set_zlim(*depth_lim[::-1])  # deeper is lower
    ax.tick_params(colors=MUTED, labelsize=8)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((1, 1, 1, 0))
        axis._axinfo["grid"].update(color=GRID, linewidth=0.5)
        axis.line.set_color(GRID)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path, nargs="+", help="one or more peg_hole_contact_sampler run directories")
    parser.add_argument("--out", type=Path, help="GIF path (default: first RUN_DIR/contact_samples_cylindrical.gif)")
    parser.add_argument("--points", type=int, default=300_000, help="number of samples to plot")
    parser.add_argument("--balance-top", type=float, nargs=4, default=[0.25, 0.25, 0.25, 0.25],
                        help="thinning cell on top of the hole: rho [mm], tilt_r, |tilt_t| [deg], depth [mm]")
    parser.add_argument("--balance-bore", type=float, nargs=4, default=[0.05, 0.25, 0.25, 0.5],
                        help="thinning cell inside the bore")
    parser.add_argument("--random", action="store_true", help="plot a plain random subset instead")
    parser.add_argument("--alpha", type=float, default=0.025, help="point opacity")
    parser.add_argument("--top-depth", type=float, default=3.0, help="depth limit of the right-hand panel [mm]")
    parser.add_argument("--frames", type=int, default=72, help="frames per full turn")
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--elevation", type=float, default=22.0)
    parser.add_argument("--dpi", type=int, default=80)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    out = args.out or args.run_dir[0] / "contact_samples_cylindrical.gif"

    q_all = np.concatenate([load_reduced(r) for r in args.run_dir])
    q = q_all[subset(q_all, args.points, None if args.random else args.balance_top, args.balance_bore,
                     args.seed)].astype(float)
    rho, tilt_r, tilt_t, depth = q.T
    top = depth <= args.top_depth
    print(f"{len(q)} points, {top.mean():.1%} within {args.top_depth} mm of the top of the hole")

    fig = plt.figure(figsize=(13, 6.2), facecolor="white")
    panels = [(fig.add_axes([0.0, 0.0, 0.46, 0.9], projection="3d"), np.ones_like(top), "Full depth",
               (min(depth.min(), -3.0), depth.max())),
              (fig.add_axes([0.45, 0.0, 0.46, 0.9], projection="3d"), top,
               f"Top of the hole (depth ≤ {args.top_depth:g} mm)", (min(depth.min(), -3.0), args.top_depth))]
    tilt_max = max(np.abs(tilt_r).max(), tilt_t.max())
    for ax, sel, title, depth_lim in panels:
        sc = ax.scatter(rho[sel], tilt_r[sel], depth[sel], c=tilt_t[sel], cmap=BLUES, vmin=0, vmax=tilt_max,
                        s=1.0, alpha=args.alpha, linewidths=0, rasterized=True)
        style(ax, title, depth_lim)
        ax.set_box_aspect(None, zoom=0.9)
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=sc.norm, cmap=BLUES), cax=fig.add_axes([0.93, 0.3, 0.01, 0.4]))
    cbar.set_label("|tangential tilt| (°)", color=MUTED, fontsize=9)
    cbar.ax.tick_params(colors=MUTED, labelsize=8)
    cbar.outline.set_visible(False)
    how = "random subset" if args.random else "thinned evenly"
    fig.suptitle(f"Peg–hole contact samples in symmetry-reduced cylindrical coordinates "
                 f"({len(q):,} of {len(q_all):,} shown, {how})", color=INK, fontsize=12, x=0.46, y=0.98)

    def frame(i):
        for ax, *_ in panels:
            ax.view_init(elev=args.elevation, azim=-60 + 360 * i / args.frames)
        return []

    FuncAnimation(fig, frame, frames=args.frames).save(out, writer=PillowWriter(fps=args.fps), dpi=args.dpi)
    print(f"Wrote {args.frames} frames to {out}")


if __name__ == "__main__":
    main()
