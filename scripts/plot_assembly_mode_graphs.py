#!/usr/bin/env python3
"""Mode graphs of the assembly groups at each collision tolerance of the sweep (assembly_groups.py, then
assembly_tolerance_sweep.py, which saves tol_sweep_300.npz: a compatibility matrix S_<tol> over a subset of the paths).

A node per contact mode, its area by the mode's share of the contact poses, laid out in columns from beside
the tube through the mouth to the bore (each column by depth). Its pie: the share of the mode's contact poses
in each assembly group, PCCA+ (pcca.py) on that tolerance's compatibility with k from the largest eigengap;
a pose takes the group of the nearest path start. An edge's width: the mean compatibility (share of
collision-free blends) between the paths starting in the two modes. Groups are named and coloured by the main
run's groups (groups.npz) they hold, so a colour means the same group in every panel.

    conda run -n mujoco_env python scripts/plot_assembly_mode_graphs.py --dir OUT_DIR
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects
from matplotlib.patches import FancyArrowPatch, Patch, Wedge
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree

from contact_mode_graph import COLORS, EDGE, INK, MUTED, OTHER
from pcca import pcca_scan

# Display modes: (name, column) with column 0 beside the tube, 1 on the mouth, 2 in the bore.
NODES = [("on the step", 0), ("edge on outer wall", 0), ("outer wall + outer rim", 0), ("side on outer rim", 0),
         ("edge on outer rim", 1), ("face on outer rim", 1), ("edge on top face", 1), ("mouth, several points", 1),
         ("edge on inner rim", 1), ("side on inner rim", 2), ("edge on wall", 2), ("two-point jam", 2),
         ("on the floor", 2)]
SINGLE = {"tip_edge/outer_wall": "edge on outer wall", "tip_side/rim_out": "side on outer rim",
          "tip_edge/rim_out": "edge on outer rim", "tip_face/rim_out": "face on outer rim",
          "tip_edge/top": "edge on top face", "tip_edge/rim_in": "edge on inner rim",
          "tip_side/rim_in": "side on inner rim", "tip_edge/wall": "edge on wall"}
GROUP_COLOR = {"beside the tube": COLORS[0], "on the mouth": COLORS[1], "bore entrance": COLORS[2],
               "deep in the bore": COLORS[3], "in the bore": COLORS[5], "mouth + bore": COLORS[6],
               "outside + mouth": COLORS[4]}
L = 50.0  # mm per rad in the pose metric, as in assembly_groups.py


def display_mode(name):
    """Fold an analytical mode into a display mode (multi-contact modes into the node they belong with)."""
    parts = set(name.split(" + "))
    if any("floor" in p for p in parts):
        return "on the floor"
    if any("step" in p for p in parts):
        return "on the step"
    if "tip_edge/wall" in parts:
        return "two-point jam" if len(parts) > 1 else "edge on wall"
    if "tip_side/rim_in" in parts:
        return "side on inner rim"
    if "tip_edge/outer_wall" in parts and len(parts) > 1:
        return "outer wall + outer rim"
    if parts == {"tip_face/rim_out", "tip_edge/rim_out"}:
        return "face on outer rim"
    if len(parts) == 1:
        return SINGLE.get(next(iter(parts)), "other")
    return "mouth, several points"


def reference_labels(paths, lengths, group_rep):
    """The main run's groups, named by their path starts (both sides of the tube are "outside")."""
    start, names = paths[:, -1], {}
    for c in np.unique(group_rep):
        m = group_rep == c
        rho, dep, ln = np.median(start[m, 0]), np.median(start[m, 3]), np.median(lengths[m])
        names[c] = "outside" if rho > 15 else ("mouth" if dep < 1 else ("deep" if ln < 20 else "entrance"))
    return np.array([names[c] for c in group_rep])


def group_name(ref):
    """Name a group by the reference groups making up at least 20% of it."""
    held = {r for r in np.unique(ref) if np.mean(ref == r) >= 0.2}
    if held == {"outside"}:
        return "beside the tube"
    if held == {"mouth"}:
        return "on the mouth"
    if held == {"entrance"}:
        return "bore entrance"
    if held == {"deep"}:
        return "deep in the bore"
    if held == {"entrance", "deep"}:
        return "in the bore"
    if "outside" not in held:
        return "mouth + bore"
    return "outside + mouth" if held == {"outside", "mouth"} else " + ".join(sorted(held))


def draw(ax, pos, share, pies, compat, colors, n_paths, title, legend):
    top = max(share)
    radius = 0.12 + 0.30 * np.sqrt(share / top)
    for a in range(len(pos)):
        for b in range(a + 1, len(pos)):
            if compat[a, b] >= 0.02:
                hi, lo = (a, b) if pos[a, 1] > pos[b, 1] else (b, a)
                # within a column the nodes are collinear: bow those edges out to the left, longer ones further
                rad = 0.18 + 0.05 * abs(pos[a, 1] - pos[b, 1]) if pos[a, 0] == pos[b, 0] else 0.0
                ax.add_patch(FancyArrowPatch(pos[hi], pos[lo], arrowstyle="-", connectionstyle=f"arc3,rad={rad}",
                                             color=EDGE, lw=0.4 + 7 * compat[a, b], alpha=0.5, zorder=1,
                                             shrinkA=0, shrinkB=0, capstyle="round"))
    halo = [patheffects.withStroke(linewidth=3, foreground="white")]
    for a, (x, y) in enumerate(pos):
        t0 = 90.0
        for g in np.argsort(-pies[a]):
            if pies[a, g] <= 0:
                continue
            t1 = t0 - 360 * pies[a, g]
            ax.add_patch(Wedge((x, y), radius[a], t1, t0, facecolor=colors[g], edgecolor="white", lw=1.2, zorder=2))
            t0 = t1
        ax.text(x + radius[a] + 0.08, y, f"{NODES[a][0]}\n{share[a]:.1%} · {n_paths[a]} paths", ha="left",
                va="center", fontsize=7, color=INK, linespacing=1.15, zorder=3, path_effects=halo)
    for x, label in [(0, "beside the tube"), (1, "on the mouth"), (2, "in the bore")]:
        ax.text(x * 2.0, 0.75, label, ha="center", va="bottom", fontsize=8, color=MUTED, style="italic")
    ax.set_title(title, fontsize=9.5, color=INK, loc="left")
    ax.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=2, frameon=False, fontsize=7.5,
              labelcolor=INK, handlelength=1.0, columnspacing=1.2)
    ax.set_xlim(-1.4, 6.0)
    ax.set_ylim(-4.6, 1.1)
    ax.set_aspect("equal")
    ax.axis("off")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", type=Path, required=True, help="assembly_groups.py output directory")
    parser.add_argument("--sweep", type=Path, help="tolerance sweep (default DIR/tol_sweep_300.npz)")
    parser.add_argument("--max-k", type=int, default=6)
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()
    out = args.dir / "mode_graphs"
    out.mkdir(exist_ok=True)
    r, g = np.load(args.dir / "roadmap.npz"), np.load(args.dir / "groups.npz")
    sw = np.load(args.sweep or args.dir / "tol_sweep_300.npz")
    sub = sw["sub"]
    nodes, n_contact, names = r["nodes"], int(r["n_contact"]), list(r["mode_names"])
    node_names = [n for n, _ in NODES]
    disp = np.array([node_names.index(display_mode(n)) if display_mode(n) in node_names else -1 for n in names])
    pose_node = disp[r["contact_mode"]]
    reps = g["reps"][sub]
    rep_node = pose_node[reps]
    ref = reference_labels(g["paths"], g["lengths"], g["group_rep"])[sub]
    scale = np.array([1.0, np.radians(1) * L, np.radians(1) * L, 1.0])
    q = nodes[:n_contact]
    near = cKDTree(nodes[reps] * scale).query(q * scale)[1]

    keep = pose_node >= 0
    share = np.bincount(pose_node[keep], minlength=len(NODES)) / keep.sum()
    n_paths = np.bincount(rep_node[rep_node >= 0], minlength=len(NODES))
    pos = np.zeros((len(NODES), 2))
    for col in range(3):
        mine = [a for a, (_, c) in enumerate(NODES) if c == col]
        depth = [np.median(q[pose_node == a, 3]) for a in mine]
        for rank, a in enumerate(np.array(mine)[np.argsort(depth)]):
            pos[a] = (2.0 * col, -rank * 4.4 / max(len(mine) - 1, 1) + 0.2)

    tols = sorted(float(k[2:]) for k in sw.files if k.startswith("S_"))
    fig, axes = plt.subplots(2, 3, figsize=(17, 10.5))
    fig.patch.set_facecolor("white")
    for t, tol in enumerate(tols):
        S = sw[f"S_{tol}"]
        scan, vals = pcca_scan(csr_matrix(S), range(2, args.max_k + 1))
        k = int(np.argmax(vals[1:args.max_k] - vals[2:args.max_k + 1]) + 2)
        chi, crisp = scan[k]
        grp = chi.argmax(1)
        gnames = []
        for c in range(k):
            m = grp == c
            name = group_name(ref[m]) if m.any() else "empty"
            gnames.append(name + (" (sliver)" if m.mean() < 0.03 else ""))
        colors = [GROUP_COLOR.get(n, OTHER) for n in gnames]
        pose_grp = grp[near]
        pies = np.array([np.bincount(pose_grp[pose_node == a], minlength=k) / max((pose_node == a).sum(), 1)
                         for a in range(len(NODES))])
        compat = np.zeros((len(NODES), len(NODES)))
        for a in range(len(NODES)):
            for b in range(len(NODES)):
                ia, ib = np.flatnonzero(rep_node == a), np.flatnonzero(rep_node == b)
                block = S[np.ix_(ia, ib)]
                if a == b:
                    block = block[~np.eye(len(ia), dtype=bool)]
                compat[a, b] = block.mean() if block.size else 0.0
        order = np.argsort(-np.bincount(pose_grp, minlength=k))
        legend = [Patch(facecolor=colors[c], edgecolor="none", label=f"{gnames[c]} ({np.mean(pose_grp == c):.0%})")
                  for c in order if np.any(pose_grp == c)]
        tol_label = f"{tol * 1000:.0f} µm" if tol < 0.1 else f"{tol:g} mm"
        title = f"collision tolerance {tol_label}: {k} groups (largest eigengap), crispness {crisp:.2f}"
        draw(axes.flat[t], pos, share, pies, compat, colors, n_paths, title, legend)
        single, ax = plt.subplots(figsize=(8, 6.5))
        single.patch.set_facecolor("white")
        draw(ax, pos, share, pies, compat, colors, n_paths, title, legend)
        single.savefig(out / f"mode_graph_tol_{tol:g}mm.png", dpi=args.dpi, bbox_inches="tight")
        plt.close(single)
        print(f"tol {tol_label}: k = {k}, crispness {crisp:.2f}, groups " + ", ".join(
            f"{gnames[c]} {np.mean(pose_grp == c):.0%}" for c in order))
    key = axes.flat[len(tols)]
    key.axis("off")
    key.text(0.02, 0.95, "How to read these graphs", fontsize=11, color=INK, va="top", weight="bold")
    key.text(0.02, 0.86, "Node: a contact mode (multi-contact modes folded into the one\n"
                         "they belong with). Area: its share of the contact poses.\n\n"
                         "Pie: share of the mode's poses in each assembly group (PCCA+\n"
                         "on path compatibility, k from the largest eigengap). Colours\n"
                         "keep their meaning across panels.\n\n"
                         "Edge width: mean share of collision-free blends between the\n"
                         "assembly paths starting in the two modes (drawn if >= 0.02).\n\n"
                         "Columns: beside the tube | on the mouth | in the bore,\n"
                         "each from shallow (top) to deep (bottom).\n\n"
                         f"{len(sub)} of the {len(g['reps'])} assembly paths; a pose takes the\n"
                         "group of the nearest path start.",
             fontsize=9, color=INK, va="top", linespacing=1.35)
    fig.suptitle("Contact modes grouped by their assembly trajectory, as the collision tolerance loosens",
                 fontsize=13, color=INK, x=0.06, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out / "mode_graphs_by_tolerance.png", dpi=args.dpi, bbox_inches="tight")
    print(out / "mode_graphs_by_tolerance.png")


if __name__ == "__main__":
    main()
