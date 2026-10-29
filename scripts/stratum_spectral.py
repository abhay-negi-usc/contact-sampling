#!/usr/bin/env python3
"""Spectral clustering of the stratum graph: contact modes as nodes, joined by the transitions the
sampled trajectories make between them, grouped into families of modes.

Contact modes are discrete in which features touch, but the contact space is one connected manifold
on which pose and wrench vary continuously across most mode boundaries (an edge sliding round a rim
turns its contact normal smoothly from along the peg to across it). Clustering the samples therefore
cannot find the modes; clustering the modes by how freely motion flows between them can.

Nodes: every analytical contact mode (peg_hole_analytic.py) holding at least --min-samples samples.
Edges: the number of trajectory steps (and first steps from a parent sample) between two modes.
Spectral clustering: normalized Laplacian of that graph; the largest eigengap among the first
--max-k picks the number of families (unless --k); k-means on the row-normalized eigenvectors.
Families are compared with the known families of the peg and hole (peg_hole_analytic.family_of_mode:
in the bore, edge on inner rim, face on mouth, edge on mouth, beside the tube, on the step) and with
the regions they make up (inside the bore, on the mouth, outside the tube). Several runs (for example
the mouth and outer regions) are analysed together, each sample keeping its run.

Output in RUN_DIR/strata: summary.json, and contact_family_graph.gif, a rotating 3D plot of the
samples coloured by family beside the stratum graph with its nodes coloured the same way.

    conda run -n mujoco_env python scripts/stratum_spectral.py RUN_DIR
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

import peg_hole_analytic as A
from compare_samples_to_analytic import SAMPLE_TOL, SAMPLE_TOL_BORE
from contact_graph_spectral import load_run, transition_edges
from contact_mode_graph import COLORS, EDGE, GRID, INK, MUTED, OTHER
from plot_contact_samples_cylindrical import subset


def sample_fine_modes(run_dir, q, chunk=2_000_000):
    """Analytical contact mode of every sample: ids into a list of mode names (both cached)."""
    cache, names_file = Path(run_dir) / "contact_mode_fine.npy", Path(run_dir) / "contact_mode_fine.json"
    if cache.exists():
        return np.load(cache), json.loads(names_file.read_text())
    names, ids = {}, np.empty(len(q), np.int32)
    for i in range(0, len(q), chunk):
        c, u = A.pose_from_reduced(q[i:i + chunk].astype(float))
        _, _, pairs = A.contact_modes(c, u, SAMPLE_TOL, SAMPLE_TOL_BORE)
        uniq, inv = np.unique(pairs, axis=0, return_inverse=True)
        local = np.array([names.setdefault(A.mode_name(p), len(names)) for p in uniq], np.int32)
        ids[i:i + chunk] = local[inv.ravel()]
    name_list = sorted(names, key=names.get)
    np.save(cache, ids)
    names_file.write_text(json.dumps(name_list))
    return ids, name_list


def grouping(mode, level):
    """level "family": A.FAMILIES; level "region": inside the bore, on the mouth, outside the tube."""
    family = A.family_of_mode(mode)
    return family if level == "family" else A.REGION_OF_FAMILY[family]


def load_runs(runs):
    """Samples of one or more runs: reduced coords, trajectory ids made unique across runs, steps,
    parent table, analytical mode ids into one shared list of names, and the run index of each sample."""
    qs, trajs, steps, tables, modes, run_of = [], [], [], [], [], []
    names = {}
    for r, run in enumerate(runs):
        q, traj, step, table = load_run(run)
        ids, run_names = sample_fine_modes(run, q)
        remap = np.array([names.setdefault(n, len(names)) for n in run_names], np.int32)
        offset = np.int64(r) * 10**11  # trajectory ids are worker * 10**9 + count within a run
        qs.append(q)
        trajs.append(traj + offset)
        steps.append(step)
        tables.append({"traj": table["traj"] + offset, "parent_step": table["parent_step"],
                       "parent_traj": np.where(table["parent_traj"] >= 0, table["parent_traj"] + offset, -1)})
        modes.append(remap[ids])
        run_of.append(np.full(len(q), r, np.int8))
    table = {k: np.concatenate([t[k] for t in tables]) for k in tables[0]}
    return (np.concatenate(qs), np.concatenate(trajs), np.concatenate(steps), table, np.concatenate(modes),
            sorted(names, key=names.get), np.concatenate(run_of))


def short(mode):
    """Readable label of a mode: peg feature on hole feature, joined by '+'."""
    words = {"tip_face": "face", "tip_edge": "edge", "tip_side": "side", "rim_in": "inner rim", "rim_out": "outer rim",
             "top": "top face", "wall": "wall", "floor": "floor"}
    if mode == "none":
        return "no contact within tolerance"
    parts = []
    for pair in mode.split(" + "):
        peg, hole = pair.split("/")
        parts.append(f"{words.get(peg, peg)} on {words.get(hole, hole)}")
    return " + ".join(parts)


def compact(mode):
    """Short label of a mode for the graph: F/E/S (peg face, edge, side) on a hole feature."""
    if mode == "none":
        return "no contact"
    peg = {"tip_face": "F", "tip_edge": "E", "tip_side": "S"}
    hole = {"rim_in": "inner", "rim_out": "outer", "top": "top", "wall": "wall", "floor": "floor",
            "outer_wall": "outwall", "step": "step", "step_edge": "stepedge", "boss_wall": "boss"}
    return " + ".join(f"{peg.get(a, a)}·{hole.get(b, b)}" for a, b in (pair.split("/") for pair in mode.split(" + ")))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path, nargs="+", help="one or more sampler runs, analysed together")
    parser.add_argument("--out-dir", type=Path, help="default: RUN_DIR/strata for one run, else a folder "
                                                      "beside the first run's parent")
    parser.add_argument("--min-samples", type=int, default=2000, help="smallest mode kept as a node")
    parser.add_argument("--max-k", type=int, default=8)
    parser.add_argument("--k", type=int, help="number of mode families (default: largest eigengap)")
    parser.add_argument("--points", type=int, default=300_000)
    parser.add_argument("--alpha", type=float, default=0.06)
    parser.add_argument("--frames", type=int, default=72)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--dpi", type=int, default=80)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    runs = args.run_dir
    out = args.out_dir or (runs[0] / "strata" if len(runs) == 1 else
                           runs[0].parent.parent / ("strata_" + "+".join(f"{r.parent.name}/{r.name}" for r in runs)
                                                    ).replace("/", "_"))
    out.mkdir(parents=True, exist_ok=True)

    q, traj, step, table, mode, names, run_of = load_runs(runs)
    counts = np.bincount(mode, minlength=len(names))
    nodes = np.flatnonzero(counts >= args.min_samples)
    node_of = np.full(len(names), -1)
    node_of[nodes] = np.arange(len(nodes))
    print(f"{len(q)} samples in {len(names)} analytical modes; {len(nodes)} modes with >= {args.min_samples} samples "
          f"hold {counts[nodes].sum() / len(q):.2%} of them")

    a, b = transition_edges(mode, traj, step, table)
    keep = (node_of[a] >= 0) & (node_of[b] >= 0)
    n = len(nodes)
    W = np.zeros((n, n))
    np.add.at(W, (node_of[a[keep]], node_of[b[keep]]), 1)
    W = W + W.T
    deg = W.sum(1)
    Dm = 1 / np.sqrt(np.maximum(deg, 1e-12))
    L = np.eye(n) - Dm[:, None] * W * Dm[None, :]
    vals, vecs = np.linalg.eigh(L)
    k_gap = int(np.argmax(np.diff(vals)[1:args.max_k]) + 2)
    k = args.k or k_gap
    U = vecs[:, :k] / np.linalg.norm(vecs[:, :k], axis=1, keepdims=True)
    family = KMeans(k, n_init=50, random_state=args.seed).fit_predict(U)
    print("eigenvalues: " + " ".join(f"{v:.3f}" for v in vals[:args.max_k + 1]) + f"  -> k = {k} (eigengap {k_gap})")

    weights = counts[nodes]
    report = dict(eigenvalues=vals[:args.max_k + 1].tolist(), k_eigengap=k_gap, k=k, families=[], agreement={})
    for level in ("region", "family"):
        truth = [grouping(names[m], level) for m in nodes]
        _, t_ids = np.unique(truth, return_inverse=True)
        # agreement over samples: each sample carries its mode's family and its mode's group
        ari = adjusted_rand_score(np.repeat(t_ids, weights), np.repeat(family, weights))
        cont = np.zeros((k, t_ids.max() + 1))
        np.add.at(cont, (family, t_ids), weights)
        report["agreement"][level] = dict(ari=float(ari), purity=float(cont.max(1).sum() / cont.sum()))
        print(f"grouping {level}: ARI {ari:.3f}, purity {cont.max(1).sum() / cont.sum():.1%} (weighted by samples)")
    mean_depth = np.array([q[mode == m, 3].mean() for m in nodes])
    order = np.argsort([np.average(mean_depth[family == f], weights=weights[family == f]) for f in range(k)])
    for rank, f in enumerate(order):
        members = np.flatnonzero(family == f)
        members = members[np.argsort(-weights[members])]
        share = weights[members].sum() / weights.sum()
        groups = {g: float(weights[members][[grouping(names[nodes[m]], "family") == g for m in members]].sum()
                           / weights[members].sum()) for g in A.FAMILIES}
        report["families"].append(dict(family=rank, share=float(share), groups=groups,
                                       modes={short(names[nodes[m]]): int(weights[m]) for m in members}))
        print(f"\nfamily {rank}: {share:.1%} of samples; " + ", ".join(f"{g} {v:.0%}" for g, v in groups.items() if v > 0))
        for m in members[:12]:
            print(f"   {weights[m] / weights.sum():7.2%}  {short(names[nodes[m]])}")
        if len(members) > 12:
            print(f"   ... and {len(members) - 12} more")
    (out / "summary.json").write_text(json.dumps(report, indent=1))

    # rotating plot: samples coloured by family beside the stratum graph coloured the same way
    rank_of = np.empty(k, int)
    rank_of[order] = np.arange(k)
    colors = [COLORS[r] if r < len(COLORS) else OTHER for r in range(k)]
    sample_family = np.full(len(q), k)
    in_nodes = node_of[mode] >= 0
    sample_family[in_nodes] = rank_of[family[node_of[mode[in_nodes]]]]
    idx = subset(q, args.points, [0.25] * 4, [0.05, 0.25, 0.25, 0.5], args.seed)
    pts, lab = q[idx].astype(float), sample_family[idx]

    fig = plt.figure(figsize=(14.5, 7.2), facecolor="white")
    ax3 = fig.add_axes([-0.03, 0.02, 0.52, 0.86], projection="3d")
    for r in sorted(range(k + 1), key=lambda r: -(lab == r).sum()):
        sel = lab == r
        ax3.scatter(pts[sel, 0], pts[sel, 1], pts[sel, 3], s=1.2, c=colors[r] if r < k else OTHER, alpha=args.alpha,
                    linewidths=0, rasterized=True)
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

    axg = fig.add_axes([0.47, 0.03, 0.52, 0.85])
    # layout: families in a grid of up to three columns (shallow to deep), modes stacked within a family by size
    ncol = min(k, 3)
    col_w = 4.2 if k > 4 else 3.2
    rows = [order[i:i + ncol] for i in range(0, k, ncol)]
    row_h = [max((family == f).sum() for f in row) for row in rows]
    origin = {}
    y0 = 0.0
    for row, h in zip(rows, row_h):
        for c, f in enumerate(row):
            origin[f] = (c * col_w, y0)
        y0 -= h + 2.6
    pos = np.zeros((n, 2))
    label_it = np.zeros(n, bool)
    for f in order:
        members = np.flatnonzero(family == f)
        members = members[np.argsort(-weights[members])]
        label_it[members[:6]] = True  # the six largest modes of each family are named
        for j, m in enumerate(members):
            pos[m] = (origin[f][0], origin[f][1] - j * 1.0)
    top = W.max()
    for i in range(n):
        for j in range(i + 1, n):
            if W[i, j] > 0:
                lw = 0.3 + 4.0 * np.log1p(W[i, j]) / np.log1p(top)
                axg.plot(*pos[[i, j]].T, color=EDGE, linewidth=lw, alpha=0.35, zorder=1, solid_capstyle="round")
    size = 30 + 700 * np.sqrt(weights / weights.max())
    axg.scatter(pos[:, 0], pos[:, 1], s=size, c=[colors[rank_of[family[i]]] for i in range(n)], edgecolors="white",
                linewidths=1.5, zorder=2)
    for i in np.flatnonzero(label_it):
        axg.text(pos[i, 0] + 0.3, pos[i, 1], f"{compact(names[nodes[i]])}  {weights[i] / weights.sum():.1%}",
                 fontsize=6.5 if k > 4 else 7, color=INK, va="center", zorder=3)
    for r, f in enumerate(order):
        fam = report["families"][r]
        dominant = max(fam["groups"], key=fam["groups"].get)
        label = dominant.replace("peg ", "").replace("the ", "")
        unnamed = (family == f).sum() - min(6, (family == f).sum())
        axg.text(origin[f][0] - 0.2, origin[f][1] + 0.8, f"family {r}: {fam['share']:.0%} of samples\n"
                 f"{fam['groups'][dominant]:.0%} {label}", fontsize=7.5, color=INK, ha="left", va="bottom",
                 fontweight="bold")
        if unnamed:
            axg.text(origin[f][0] + 0.3, origin[f][1] - min(6, (family == f).sum()) * 1.0 + 0.1,
                     f"+ {unnamed} smaller modes", fontsize=6, color=MUTED, va="center")
    axg.text(-0.2, pos[:, 1].min() - 1.4, "F peg face, E peg edge, S peg side; on the hole: top face, inner or outer rim "
             "edge, bore wall, floor, outer wall (outwall), step and its outer edge (stepedge). Node area: share of "
             "samples; edge width: log number of transitions.", fontsize=6.5, color=MUTED, va="top", wrap=True)
    axg.set_xlim(-0.5, col_w * ncol + 0.6)
    axg.set_ylim(pos[:, 1].min() - 2.4, 2.2)
    axg.axis("off")
    fig.suptitle("Contact-mode families from spectral clustering of the stratum graph: samples (left) and modes with "
                 "their transitions (right)", color=INK, fontsize=12, y=0.98)

    def frame(i):
        ax3.view_init(elev=22, azim=-60 + 360 * i / args.frames)
        return []

    FuncAnimation(fig, frame, frames=args.frames).save(out / "contact_family_graph.gif", writer=PillowWriter(fps=args.fps),
                                                       dpi=args.dpi)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
