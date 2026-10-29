#!/usr/bin/env python3
"""Connectivity graph of a contact sampler run, its spectral clustering, and a check of the clusters
against the analytical contact modes of the cylindrical peg and hole.

Graph: the samples are pooled in the symmetry-reduced coordinates (rho, radial tilt, |tangential tilt|,
depth), where the workers' azimuth sectors coincide, into cells: --cell-top on top of the hole (depth
<= 0), --cell-bore inside it (finer in rho, as the bore's clearance is 0.5 mm). Two cells are joined
when a trajectory step, or a trajectory's first step from its parent sample, moves between them, so
every edge is a continuous motion through the contact space.

Spectral clustering: on the largest connected component, the smallest eigenvalues of the normalized
Laplacian L = I - D^-1/2 W D^-1/2; the largest gap among the first --max-k picks the number of clusters
k (unless --k), and k-means groups the row-normalized eigenvectors (Ng, Jordan and Weiss).

Cross-check: each cell's contact mode is the most common analytical mode (peg_hole_analytic.py) of a
random subset of its samples; clusters and modes are compared by a contingency table, purity,
normalized mutual information and the adjusted Rand index, for the eigengap k and for a range of k.

    conda run -n mujoco_env python scripts/contact_graph_spectral.py RUN_DIR
"""

import argparse
import glob
import json
import textwrap
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

import peg_hole_analytic as A
from compare_samples_to_analytic import SAMPLE_TOL, SAMPLE_TOL_BORE, sample_reduced

INK, MUTED, GRID = "#2b2b2b", "#6b6b6b", "#d9d9d6"
BLUE, BLUES = "#2a78d6", ["#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def load_run(run_dir):
    """Reduced coords, trajectory ids and steps of every sample, and the trajectory parent table."""
    traj, step = [], []
    for f in sorted(glob.glob(str(Path(run_dir) / "shard_*.npz"))):
        z = np.load(f)
        traj.append(z["traj"])
        step.append(z["step"])
    t = [np.load(f) for f in sorted(glob.glob(str(Path(run_dir) / "traj_*.npz")))]
    table = {k: np.concatenate([z[k] for z in t]) for k in ("traj", "parent_traj", "parent_step")}
    return sample_reduced(run_dir), np.concatenate(traj), np.concatenate(step), table


def cells_of(q, cell_top, cell_bore):
    """Cell id of every sample (0..n_cells-1) and the number of cells."""
    inside = q[:, 3] > 0
    size = np.where(inside[:, None], np.asarray(cell_bore, np.float32), np.asarray(cell_top, np.float32))
    idx = np.floor(q / size).astype(np.int64)
    key = inside.astype(np.int64)
    for i, (offset, bits) in enumerate([(0, 15), (2048, 12), (0, 12), (64, 10)]):
        key = (key << bits) | (idx[:, i] + offset)
    _, cell = np.unique(key, return_inverse=True)
    return cell, cell.max() + 1


def transition_edges(cell, traj, step, table):
    """Cell pairs joined by a trajectory step or by a trajectory's start from its parent sample."""
    order = np.lexsort((step, traj))
    tr, cs = traj[order], cell[order]
    same = tr[1:] == tr[:-1]
    a, b = cs[:-1][same], cs[1:][same]
    key = tr * (1 << 20) + step[order]

    def lookup(t, s):
        i = np.searchsorted(key, t * (1 << 20) + s)
        i = np.minimum(i, len(key) - 1)
        return np.where(key[i] == t * (1 << 20) + s, order[i], -1)

    has_parent = table["parent_traj"] >= 0
    child = lookup(table["traj"][has_parent], np.zeros(has_parent.sum(), np.int64))
    parent = lookup(table["parent_traj"][has_parent], table["parent_step"][has_parent].astype(np.int64))
    ok = (child >= 0) & (parent >= 0)
    a = np.concatenate([a, cell[child[ok]]])
    b = np.concatenate([b, cell[parent[ok]]])
    keep = a != b
    return a[keep], b[keep]


def cell_modes(q, cell, n_cells, per_cell, seed):
    """Most common analytical contact mode of up to per_cell random samples of each cell."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(cell))
    rank = np.empty(len(cell), np.int64)  # order of each sample within its cell, in the random order
    sorted_cells = cell[perm]
    o = np.argsort(sorted_cells, kind="stable")
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_cells[o])) + 1]
    run = np.arange(len(o)) - np.repeat(starts, np.diff(np.r_[starts, len(o)]))
    rank[perm[o]] = run
    pick = np.flatnonzero(rank < per_cell)
    c, u = A.pose_from_reduced(q[pick].astype(float))
    _, _, pairs = A.contact_modes(c, u, SAMPLE_TOL, SAMPLE_TOL_BORE)
    names = np.array([A.mode_name(p) for p in pairs])
    mode_list, mode_idx = np.unique(names, return_inverse=True)
    counts = sp.coo_matrix((np.ones(len(pick)), (cell[pick], mode_idx)), shape=(n_cells, len(mode_list))).tocsr()
    major = np.asarray(counts.argmax(axis=1)).ravel()
    share = np.asarray(counts.max(axis=1).todense()).ravel() / np.maximum(np.asarray(counts.sum(axis=1)).ravel(), 1)
    return mode_list, major, share


PHASES = ["on top / rims", "one point in the bore", "two-point jam", "on the floor"]


def phase_of(mode):
    """Classical peg-in-hole contact state of an analytical contact mode."""
    if "floor" in mode:
        return 3
    if "wall" in mode and "tip_side/rim_in" in mode:
        return 2
    if "wall" in mode or "tip_side" in mode or "shoulder" in mode:
        return 1
    return 0


def spectral(W, max_k, seed):
    """Smallest eigenpairs of the normalized Laplacian of W (shift-invert about zero)."""
    d = np.asarray(W.sum(axis=1)).ravel()
    Dm = sp.diags(1 / np.sqrt(d))
    L = sp.identity(W.shape[0], format="csc") - Dm @ W @ Dm
    vals, vecs = eigsh(L.tocsc(), k=max_k + 1, sigma=-1e-3, which="LM")
    order = np.argsort(vals)
    return vals[order], vecs[:, order]


def cluster(vecs, k, seed):
    U = vecs[:, :k]
    U = U / np.linalg.norm(U, axis=1, keepdims=True)
    return KMeans(k, n_init=10, random_state=seed).fit_predict(U)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--cell-top", type=float, nargs=4, default=[0.5, 0.5, 0.5, 0.5],
                        help="cell size on top of the hole: rho [mm], tilt_r, |tilt_t| [deg], depth [mm]")
    parser.add_argument("--cell-bore", type=float, nargs=4, default=[0.05, 0.5, 0.5, 1.0],
                        help="cell size inside the bore: rho [mm], tilt_r, |tilt_t| [deg], depth [mm]")
    parser.add_argument("--max-k", type=int, default=20, help="largest number of clusters considered")
    parser.add_argument("--k", type=int, help="number of clusters (default: largest eigengap)")
    parser.add_argument("--per-cell", type=int, default=20, help="samples per cell classified analytically")
    parser.add_argument("--major-modes", type=int, default=10, help="modes shown by name; the rest fold into 'other'")
    parser.add_argument("--out-dir", type=Path, help="default: RUN_DIR/spectral")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    out = args.out_dir or args.run_dir / "spectral"
    out.mkdir(parents=True, exist_ok=True)
    report = {}

    t0 = time.time()
    q, traj, step, table = load_run(args.run_dir)
    cell, n_cells = cells_of(q, args.cell_top, args.cell_bore)
    a, b = transition_edges(cell, traj, step, table)
    W = sp.coo_matrix((np.ones(2 * len(a)), (np.r_[a, b], np.r_[b, a])), shape=(n_cells, n_cells)).tocsr()
    W.data[:] = 1.0  # an edge per pair of cells with any transition between them
    n_comp, comp = connected_components(W, directed=False)
    sizes = np.bincount(comp)
    samples_per_cell = np.bincount(cell, minlength=n_cells)
    comp_samples = np.bincount(comp, weights=samples_per_cell)
    largest = int(np.argmax(sizes))
    print(f"{len(q)} samples, {n_cells} cells, {W.nnz // 2} edges ({time.time() - t0:.0f} s)")
    print(f"connected components: {n_comp}; largest has {sizes[largest]} cells ({sizes[largest] / n_cells:.2%}) and "
          f"{comp_samples[largest] / len(q):.4%} of the samples; the rest have at most {np.sort(sizes)[-2] if n_comp > 1 else 0} cells")
    report.update(samples=len(q), cells=int(n_cells), edges=int(W.nnz // 2), components=int(n_comp),
                  largest_component_cells=int(sizes[largest]),
                  largest_component_sample_fraction=float(comp_samples[largest] / len(q)))

    t0 = time.time()
    mode_list, cell_mode, cell_share = cell_modes(q, cell, n_cells, args.per_cell, args.seed)
    print(f"cell modes from up to {args.per_cell} samples per cell ({time.time() - t0:.0f} s); "
          f"median share of the cell's own mode {np.median(cell_share):.0%}")
    # modes of the cells outside the largest component, to see if any mode is cut off
    if n_comp > 1:
        off = comp != largest
        off_modes = np.bincount(cell_mode[off], weights=samples_per_cell[off], minlength=len(mode_list))
        report["outside_largest_component"] = {mode_list[i]: int(off_modes[i]) for i in np.argsort(-off_modes)[:10]
                                               if off_modes[i] > 0}
        print("  samples outside the largest component by mode:", report["outside_largest_component"])

    keep = comp == largest
    Wl = W[keep][:, keep]
    t0 = time.time()
    vals, vecs = spectral(Wl, args.max_k, args.seed)
    gaps = np.diff(vals)
    k_gap = int(np.argmax(gaps[1:args.max_k]) + 2)  # k eigenvalues below the largest gap, k >= 2
    k = args.k or k_gap
    print(f"eigenvalues ({time.time() - t0:.0f} s): " + " ".join(f"{v:.4f}" for v in vals[:args.max_k + 1]))
    print(f"largest eigengap after eigenvalue {k_gap} -> k = {k}")
    report.update(eigenvalues=vals.tolist(), k_eigengap=k_gap, k=k)

    weights = samples_per_cell[keep]
    modes_l = cell_mode[keep]
    totals = np.bincount(cell_mode, weights=samples_per_cell, minlength=len(mode_list))
    major = np.argsort(-totals)[:args.major_modes]
    named = np.full(len(mode_list), -1)
    named[major] = np.arange(len(major))
    labels_major = np.where(named[modes_l] >= 0, named[modes_l], len(major))
    major_names = [mode_list[i] for i in major] + ["other"]

    phase_l = np.array([phase_of(m) for m in mode_list])[modes_l]
    sweep = []
    for kk in range(2, args.max_k + 1):
        lab = cluster(vecs, kk, args.seed)
        row = dict(k=kk)
        for name, truth in (("mode", modes_l), ("phase", phase_l)):
            cont = np.zeros((kk, truth.max() + 1))
            np.add.at(cont, (lab, truth), weights)
            row.update({f"{name}_ari": float(adjusted_rand_score(truth, lab)),
                        f"{name}_nmi": float(normalized_mutual_info_score(truth, lab)),
                        f"{name}_purity": float(cont.max(axis=1).sum() / cont.sum())})
        sweep.append(row)
        if kk == k:
            labels = lab
    print("\n        vs analytical contact mode     vs contact state (phase)")
    print("   k    ARI    NMI  purity        ARI    NMI  purity   (per cell; purity weighted by samples)")
    for r in sweep:
        print(f"{r['k']:4d} {r['mode_ari']:6.3f} {r['mode_nmi']:6.3f} {r['mode_purity']:7.1%}    "
              f"{r['phase_ari']:6.3f} {r['phase_nmi']:6.3f} {r['phase_purity']:7.1%}"
              + ("   <- eigengap" if r['k'] == k_gap else ""))
    report["sweep"] = sweep

    # contingency of the chosen clustering: share of each cluster's samples in each major mode
    cont = np.zeros((k, len(major_names)))
    np.add.at(cont, (labels, labels_major), weights)
    row = cont / cont.sum(axis=1, keepdims=True)
    print(f"\ncluster  samples  dominant mode (share)   for k = {k}")
    clusters = []
    for i in np.argsort(-cont.sum(axis=1)):
        j = int(np.argmax(row[i]))
        clusters.append(dict(cluster=int(i), samples=int(cont[i].sum()), dominant=major_names[j], share=float(row[i, j]),
                             composition={major_names[m]: float(row[i, m]) for m in np.argsort(-row[i])[:4]
                                          if row[i, m] > 0.01}))
        print(f"{i:7d} {int(cont[i].sum()):9d}  {major_names[j]} ({row[i, j]:.0%})")
    report["clusters"] = clusters
    (out / "summary.json").write_text(json.dumps(report, indent=1))
    cell_cluster = np.full(n_cells, -1)
    cell_cluster[keep] = labels
    np.savez(out / "cells.npz", cell=cell, component=comp, cluster=cell_cluster, mode=cell_mode, mode_list=mode_list,
             mode_share=cell_share)
    # the same for the classical contact states
    cont_p = np.zeros((k, len(PHASES)))
    np.add.at(cont_p, (labels, phase_l), weights)
    row_p = cont_p / cont_p.sum(axis=1, keepdims=True)
    report["clusters_vs_phases"] = {int(i): {PHASES[j]: float(row_p[i, j]) for j in range(len(PHASES))} for i in range(k)}
    (out / "summary.json").write_text(json.dumps(report, indent=1))
    plot_eigenvalues(vals, k_gap, out / "eigenvalues.png")
    plot_contingency(row, cont.sum(axis=1), major_names, out / "clusters_vs_modes.png",
                     "Share of each spectral cluster's samples in each analytical contact mode")
    plot_contingency(row_p, cont_p.sum(axis=1), PHASES, out / "clusters_vs_states.png",
                     "Share of each spectral cluster's samples in each contact state")
    centres = cell_centres(q, cell, n_cells)
    plot_small_multiples(centres[keep], labels, k, samples_per_cell[keep], out / "clusters_3d.png")
    print(f"\nwrote {out}")


def plot_eigenvalues(vals, k_gap, path):
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ks = np.arange(1, len(vals) + 1)
    ax.plot(ks, vals, color=BLUE, linewidth=2, marker="o", markersize=8, markeredgecolor="white", markeredgewidth=2)
    ax.axvline(k_gap + 0.5, color=MUTED, linewidth=1, linestyle="--")
    ax.annotate(f"largest gap: k = {k_gap}", (k_gap + 0.6, vals[k_gap - 1]), color=INK, fontsize=9)
    ax.set_xlabel("index", color=MUTED)
    ax.set_ylabel("eigenvalue of the normalized Laplacian", color=MUTED)
    ax.set_title("Spectrum of the contact-space connectivity graph", color=INK, fontsize=11, loc="left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.5)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def cell_centres(q, cell, n_cells):
    """Mean reduced coords of each cell's samples."""
    out = np.zeros((n_cells, 4))
    for j in range(4):
        out[:, j] = np.bincount(cell, weights=q[:, j], minlength=n_cells)
    return out / np.bincount(cell, minlength=n_cells)[:, None]


def plot_small_multiples(centres, labels, k, weights, path, elev=22, azim=-60):
    """One 3D panel per cluster: its cells in blue over all cells in grey (reduced coordinates)."""
    order = np.argsort(-np.bincount(labels, weights=weights, minlength=k))
    ncol = min(4, k)
    nrow = int(np.ceil(k / ncol))
    fig = plt.figure(figsize=(4.2 * ncol, 3.9 * nrow), facecolor="white")
    lim = [(centres[:, i].min(), centres[:, i].max()) for i in range(4)]
    for n, c in enumerate(order):
        ax = fig.add_subplot(nrow, ncol, n + 1, projection="3d")
        sel = labels == c
        ax.scatter(centres[~sel, 0], centres[~sel, 1], centres[~sel, 3], s=0.5, c=GRID, alpha=0.15, linewidths=0,
                   rasterized=True)
        ax.scatter(centres[sel, 0], centres[sel, 1], centres[sel, 3], s=1.0, c=BLUE, alpha=0.5, linewidths=0,
                   rasterized=True)
        ax.set_xlim(*lim[0])
        ax.set_ylim(*lim[1])
        ax.set_zlim(lim[3][1], lim[3][0])
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(f"cluster {c}  ({weights[sel].sum() / weights.sum():.1%} of samples)", color=INK, fontsize=9)
        ax.set_xlabel("ρ (mm)", color=MUTED, fontsize=7, labelpad=-2)
        ax.set_ylabel("radial tilt (°)", color=MUTED, fontsize=7, labelpad=-2)
        ax.set_zlabel("depth (mm)", color=MUTED, fontsize=7, labelpad=-2)
        ax.tick_params(colors=MUTED, labelsize=6, pad=-2)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.set_pane_color((1, 1, 1, 0))
            axis._axinfo["grid"].update(color=GRID, linewidth=0.4)
    fig.suptitle("Spectral clusters of the contact-space connectivity graph (symmetry-reduced coordinates)",
                 color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_contingency(row, sizes, names, path, title):
    from matplotlib.colors import LinearSegmentedColormap
    order = np.argsort(-sizes)
    row = row[order]
    fig, ax = plt.subplots(figsize=(1.0 + 0.75 * len(names), 1.2 + 0.42 * len(row)))
    im = ax.imshow(row, cmap=LinearSegmentedColormap.from_list("b", BLUES), vmin=0, vmax=1, aspect="auto")
    for i in range(row.shape[0]):
        for j in range(row.shape[1]):
            if row[i, j] >= 0.005:
                ax.text(j, i, f"{row[i, j]:.0%}", ha="center", va="center", fontsize=7,
                        color="white" if row[i, j] > 0.55 else INK)
    ax.set_xticks(range(len(names)), [n.replace(" + ", "\n+ ") for n in names], rotation=60, ha="right", fontsize=7,
                  color=MUTED)
    ax.set_yticks(range(len(row)), [f"cluster {o}  ({int(sizes[o]):,})" for o in order], fontsize=7, color=MUTED)
    ax.set_title("\n".join(textwrap.wrap(title, 50)), color=INK, fontsize=10, loc="left")
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, shrink=0.6)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=MUTED, labelsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
