#!/usr/bin/env python3
"""Rotating 3D GIF: the contact modes found from the CAD alone (cad_stratum_graph.py closure merge) beside
the analytical one-contact modes, in the symmetry-reduced coordinates (lateral offset rho, radial tilt,
depth), with one colour per physical mode in both panels.

Left: each sample coloured by its CAD cluster (named by its one contact, translated to the analytical
name only to pick the colour). Right: the analytical mode of each sample; samples with two or more
contacts are grey there (the codimension rule lets them join any of their one-contact modes).

    conda run -n mujoco_env python scripts/plot_cad_modes.py RUN_DIR [RUN_DIR ...] --labels closure_labels.npz
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D

from cad_contacts import analytic_name
from contact_mode_graph import COLORS, INK, MUTED, OTHER
from knn_pose_spectral import load_samples

CATEGORIES = ["edge on inner rim", "edge on top face", "face on outer rim", "edge on outer rim", "in the bore",
              "edge on outer wall", "side on outer rim", "on the step"]
OF_CONTACT = {"tip_edge/rim_in": 0, "tip_edge/top": 1, "tip_face/rim_out": 2, "tip_edge/rim_out": 3,
              "tip_side/rim_in": 4, "tip_edge/wall": 4, "tip_edge/floor": 4, "tip_edge/outer_wall": 5,
              "tip_side/rim_out": 6, "tip_edge/step": 7, "tip_edge/step_edge": 7, "tip_face/step_edge": 7}


def category(mode):
    """Colour index of a one-contact mode (analytical names), -1 for anything else."""
    contacts = set(mode.split(" + "))
    return OF_CONTACT.get(next(iter(contacts)), -1) if len(contacts) == 1 else -1


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path, nargs="+")
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--points", type=int, default=200_000)
    parser.add_argument("--alpha", type=float, default=0.08)
    parser.add_argument("--frames", type=int, default=72)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--dpi", type=int, default=80)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    z = np.load(args.labels)
    q_all, _, _ = load_samples(args.run_dir)
    pick = np.sort(np.random.default_rng(args.seed).choice(len(z["samples"]), min(args.points, len(z["samples"])), replace=False))
    q = q_all[z["samples"][pick]].astype(float)
    cad_cat = np.array([category(analytic_name(str(m))) for m in z["cluster_mode"]])[z["cluster"][pick]]
    an_cat = np.array([category(str(m)) for m in z["analytic_names"]])[z["analytic_mode"][pick]]

    fig = plt.figure(figsize=(13, 6.6))
    fig.patch.set_facecolor("white")
    axes = [fig.add_subplot(1, 2, i + 1, projection="3d") for i in range(2)]
    titles = ["Found from the CAD alone (no labels)", "Analytical one-contact modes (truth)"]
    for ax, cat, title in zip(axes, [cad_cat, an_cat], titles):
        for c in [-1] + list(range(len(CATEGORIES))):
            sel = cat == c
            ax.scatter(q[sel, 0], q[sel, 1], q[sel, 3], s=1.0, c=OTHER if c < 0 else COLORS[c], alpha=args.alpha,
                       linewidths=0, rasterized=True)
        ax.set_xlabel("lateral offset ρ [mm]", color=MUTED, fontsize=9)
        ax.set_ylabel("radial tilt [°]", color=MUTED, fontsize=9)
        ax.set_zlabel("depth [mm]", color=MUTED, fontsize=9)
        ax.invert_zaxis()
        ax.set_title(title, color=INK, fontsize=12)
        ax.tick_params(colors=MUTED, labelsize=8)
        for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
            pane.set_facecolor("white")
            pane.set_edgecolor("#d9d9d6")
    handles = [Line2D([], [], marker="o", ls="", color=COLORS[i], label=n) for i, n in enumerate(CATEGORIES)]
    handles.append(Line2D([], [], marker="o", ls="", color=OTHER, label="two or more contacts (right) / unassigned"))
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=9, labelcolor=INK)
    agree = (cad_cat == an_cat)[an_cat >= 0].mean()
    fig.suptitle(f"Contact modes of the peg on the hole: CAD feature contacts, merged into one-contact modes "
                 f"(agree with the truth on {agree:.0%} of one-contact samples)", color=INK, fontsize=12)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.12, wspace=0.02)

    def frame(i):
        for ax in axes:
            ax.view_init(elev=22, azim=-60 + 360 * i / args.frames)
        return []

    out = args.out or args.labels.with_name("cad_modes_vs_analytic.gif")
    FuncAnimation(fig, frame, frames=args.frames).save(out, writer=PillowWriter(fps=args.fps), dpi=args.dpi)
    print(f"{out} ({len(q):,} samples; CAD and truth agree on {agree:.1%} of the truth's one-contact samples)")


if __name__ == "__main__":
    main()
