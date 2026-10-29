#!/usr/bin/env python3
"""Modes of the hook's pose relative to the eyelet, found from the geometry of its free space, and
the graph of transitions between them.

Samples: collision-free hook poses (convex-part separation >= 0, hook_eyelet_model.Separation) drawn
around the reference trajectory, reaching --reach mm / deg beyond the separation field's tube: pockets
of the tube reach the goal through poses outside it. Poses within --near-mm of contact are all kept,
those further out with probability --far-keep (the modes differ near contact; open space is one easy
region). Added: the pre-assembly pose (the goal), reference-trajectory waypoints, the pocket starts
and extra samples around each of them.

Line-of-sight graph: each sample is joined to those of its --k nearest neighbours (metric: mm, and
deg x 0.52 mm), and of --partners random samples within --radius, that it can see, i.e. the straight
motion between them stays collision-free (checked every --step mm / deg). Near neighbours nearly always
see each other; distant ones do within a roughly convex region of free space but not across the eyelet,
so spectral clustering of this graph (normalized Laplacian, largest eigengap, k-means on the
row-normalised eigenvectors, as in contact_graph_spectral.py) splits the free space into its convex-ish
regions: the modes. The mode graph joins modes wherever nearest-neighbour lines of sight cross.

Routes: for each pocket start, the modes it passes through on the way to the goal mode, and the
line-of-sight path that keeps the most clearance (widest path), shortcut, written as a trajectory with
the final lift into contact appended, checked densely and rendered.

    conda run -n mujoco_env python scripts/hook_mode_graph.py FIELD_DIR --out-dir OUT_DIR
"""

import argparse
import heapq
import json
import multiprocessing as mp
import subprocess
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import eigsh
from scipy.spatial import cKDTree

import validate_goal_field as vg
from build_separation_field import reference_path
from contact_graph_spectral import cluster
from goal_field import PRE_ASSEMBLY
from hook_eyelet_model import ASSET_DIR

SCRIPTS = Path(__file__).resolve().parent
SCALE = np.array([1, 1, 1] + [vg.LEVER_MM_PER_DEG] * 3)  # metric: mm, and deg x 0.52 mm
# Categorical hues in fixed order (largest modes first); the rest fold into OTHER.
HUES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
OTHER, INK, INK_2, SURFACE = "#c3c2b7", "#0b0b0b", "#52514e", "#fcfcfb"


def _init_worker():
    vg._init()


def sample_task(job):
    """Collision-free poses near the reference path (or a pocket), biased towards contact."""
    centres, half, n, near_mm, far_keep, seed = job
    rng = np.random.default_rng(seed)
    poses, seps = [], []
    while len(poses) < n:
        p = centres[rng.integers(len(centres))] + rng.uniform(-half, half, 6)
        s = vg._sep(p)[0]
        if s >= 0 and (s <= near_mm or rng.random() < far_keep):
            poses.append(p)
            seps.append(s)
    return np.array(poses), np.array(seps)


def edge_task(job):
    """Clearance along each straight segment (negative: blocked)."""
    a, b, step = job
    return np.array([vg.segment_min(p, q, step, floor=0.0) for p, q in zip(a, b)])


def check_pairs(poses, pairs, step, workers):
    """The pairs in line of sight, and the clearance along each."""
    chunks = np.array_split(np.arange(len(pairs)), workers * 8)
    with mp.get_context("fork").Pool(workers, initializer=_init_worker) as pool:
        res = pool.map(edge_task, [(poses[pairs[c, 0]], poses[pairs[c, 1]], step) for c in chunks])
    clear = np.concatenate(res)
    return pairs[clear >= 0], clear[clear >= 0]


def spectral_top(W, max_k, tol=1e-8, tau=0.0):
    """Smallest eigenpairs of the normalized Laplacian I - D^-1/2 W D^-1/2 (as in contact_graph_spectral.spectral),
    computed as the largest of D^-1/2 W D^-1/2, which Lanczos finds much faster than shift-invert. Large
    grid-like graphs have many close eigenvalues; a looser tol and more Lanczos vectors keep them quick.
    tau > 0 regularizes the degrees, D + tau I (Qin & Rohe 2013), so cutting off a few weakly attached
    nodes no longer looks like a good partition."""
    d = np.asarray(W.sum(axis=1)).ravel() + tau
    Dm = sp.diags(1 / np.sqrt(d))
    vals, vecs = eigsh((Dm @ W @ Dm).tocsr(), k=max_k + 1, which="LA", tol=tol,
                       ncv=min(W.shape[0] - 1, max(4 * (max_k + 1), 2 * (max_k + 1) + 1)))
    order = np.argsort(-vals)
    return 1 - vals[order], vecs[:, order]


def widest_path(n, edges, clear, src, dst):
    """Max-min clearance path from src to dst over the sample graph (ties: fewest edges)."""
    adj = [[] for _ in range(n)]
    for (i, j), c in zip(edges, clear):
        adj[i].append((j, c))
        adj[j].append((i, c))
    best = np.full(n, -np.inf)
    hops = np.full(n, np.inf)
    prev = np.full(n, -1)
    best[src], hops[src] = np.inf, 0
    heap = [(-np.inf, 0, src)]
    while heap:
        nb, h, u = heapq.heappop(heap)
        if -nb < best[u] or (-nb == best[u] and h > hops[u]):
            continue
        if u == dst:
            break
        for v, c in adj[u]:
            b = min(-nb, c)
            if b > best[v] or (b == best[v] and h + 1 < hops[v]):
                best[v], hops[v], prev[v] = b, h + 1, u
                heapq.heappush(heap, (-b, h + 1, v))
    if prev[dst] < 0 and dst != src:
        return None, None
    path = [dst]
    while path[-1] != src:
        path.append(prev[path[-1]])
    return path[::-1], best[dst]


def mode_route(mode_edges, k, src, dst):
    """Fewest modes from src to dst over the mode graph (BFS)."""
    nbrs = {m: set() for m in range(k)}
    for (a, b) in mode_edges:
        nbrs[a].add(b)
        nbrs[b].add(a)
    prev, queue = {src: None}, [src]
    for m in queue:
        if m == dst:
            break
        for nb in sorted(nbrs[m]):
            if nb not in prev:
                prev[nb] = m
                queue.append(nb)
    if dst not in prev:
        return None
    route = [dst]
    while prev[route[-1]] is not None:
        route.append(prev[route[-1]])
    return route[::-1]


def plot_modes(poses, labels, k, mode_edges, special, out):
    sizes = np.bincount(labels, minlength=k)
    order = np.argsort(-sizes)
    colour = {m: (HUES[r] if r < len(HUES) else OTHER) for r, m in enumerate(order)}
    fig = plt.figure(figsize=(14, 6.4), facecolor=SURFACE)
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    rng = np.random.default_rng(0)
    show = rng.choice(len(poses), min(len(poses), 40000), replace=False)
    ax.scatter(poses[show, 0], poses[show, 2], poses[show, 4], s=1.5, linewidths=0, alpha=0.35,
               c=[colour[m] for m in labels[show]], depthshade=False)
    for name, i in special.items():
        ax.scatter(*poses[i, [0, 2, 4]], s=40, color=INK, depthshade=False)
        ax.text(*poses[i, [0, 2, 4]], f"  {name}", color=INK, fontsize=8)
    ax.set_xlabel("x [mm]", color=INK_2), ax.set_ylabel("z [mm]", color=INK_2), ax.set_zlabel("pitch [deg]", color=INK_2)
    ax.set_title("samples coloured by mode", color=INK, fontsize=11)
    ax.view_init(elev=20, azim=-60)

    ax2 = fig.add_subplot(1, 2, 2, facecolor=SURFACE)
    centre = np.array([poses[labels == m].mean(0) for m in range(k)])
    for (a, b), cnt in mode_edges.items():
        ax2.plot(centre[[a, b], 0], centre[[a, b], 4], color="#b9b8b1", linewidth=0.5 + np.log10(cnt), zorder=1)
    for m in range(k):
        ax2.scatter(centre[m, 0], centre[m, 4], s=30 + 400 * np.sqrt(sizes[m] / sizes.max()), color=colour[m],
                    edgecolors=SURFACE, linewidths=2, zorder=2)
        ax2.text(centre[m, 0], centre[m, 4], str(m), color=INK, fontsize=7, ha="center", va="center", zorder=3)
    for name, i in special.items():
        m = labels[i]
        ax2.annotate(name, centre[m, [0, 4]], xytext=(8, 8), textcoords="offset points", color=INK, fontsize=8,
                     arrowprops=dict(arrowstyle="-", color=INK_2, linewidth=0.8))
    ax2.set_xlabel("mode centre x [mm]", color=INK_2), ax2.set_ylabel("mode centre pitch [deg]", color=INK_2)
    ax2.set_title(f"mode graph: {k} modes; edges where lines of sight cross (width: log count)",
                  color=INK, fontsize=11)
    for side in ("top", "right"):
        ax2.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE, dpi=110)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("field_dir", type=Path, help="separation field run (for the pocket list)")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=ASSET_DIR / "hook_assembly_max_separation_trajectory.csv")
    parser.add_argument("--pockets", type=int, nargs="*", default=[13, 14, 166, 16],
                        help="pocket ids (goal_field_pockets.json) whose starts get routes")
    parser.add_argument("--solve", type=int, nargs="*", default=[13, 14, 166],
                        help="pockets to route; the others are only located in the mode graph")
    parser.add_argument("--samples", type=int, default=40000)
    parser.add_argument("--pocket-samples", type=int, default=1500)
    parser.add_argument("--reach", type=float, default=10.0, help="beyond the field's 10 mm / 10 deg tube")
    parser.add_argument("--near-mm", type=float, default=5.0)
    parser.add_argument("--far-keep", type=float, default=0.15)
    parser.add_argument("--k", type=int, default=15, help="nearest neighbours tested for line of sight")
    parser.add_argument("--step", type=float, default=0.25, help="line-of-sight check spacing [mm and deg]")
    parser.add_argument("--samples-from", type=Path, help="graph.npz of an earlier run: reuse its samples and lines of sight")
    parser.add_argument("--partners", type=int, default=20, help="long-range line-of-sight partners per sample")
    parser.add_argument("--radius", type=float, default=25.0, help="their largest distance [mm-equivalent]")
    parser.add_argument("--max-k", type=int, default=30, help="largest number of modes considered")
    parser.add_argument("--modes", type=int, help="number of modes (default: largest eigengap)")
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    report = {"args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}}
    t_all = time.time()

    # pocket starts: the node of each pocket with the most clearance
    pz = np.load(args.field_dir / "goal_field_pockets.npz")
    starts = {}
    for p in args.pockets:
        m = pz["comp"] == p
        starts[f"pocket {p}"] = pz["coords"][m][np.argmax(pz["sep"][m])].astype(float)
    path = reference_path(args.reference, spacing=0.25)
    half = 10.0 + args.reach

    t0 = time.time()
    vg._init()
    w = args.workers
    special_poses = {"goal": PRE_ASSEMBLY.astype(float), **starts}
    if args.samples_from:  # reuse the samples and nearest-neighbour lines of sight of an earlier run
        g = np.load(args.samples_from)
        poses, seps, edges, clear = g["poses"], g["seps"], g["edges"], g["clear"]
        special = {name: int(np.argmin(np.abs(poses - p).sum(1))) for name, p in special_poses.items()}
        assert all(np.abs(poses[i] - special_poses[n]).max() < 1e-6 for n, i in special.items())
        print(f"{len(poses):,} samples and {len(edges):,} nearest-neighbour lines of sight from {args.samples_from}",
              flush=True)
    else:
        jobs = [(path, half, args.samples // w + 1, args.near_mm, args.far_keep, args.seed + i) for i in range(w)]
        jobs += [(s[None], 6.0, args.pocket_samples, args.near_mm, 1.0, args.seed + 1000 + i)
                 for i, s in enumerate(starts.values())]
        with mp.get_context("fork").Pool(w, initializer=_init_worker) as pool:
            res = pool.map(sample_task, jobs)
        ref = path[:: max(1, len(path) // 150)]
        poses = np.vstack([r[0] for r in res] + [np.array(list(special_poses.values())), ref])
        seps = np.concatenate([r[1] for r in res] + [vg._sep(np.array(list(special_poses.values()))), vg._sep(ref)])
        n0 = len(poses) - len(special_poses) - len(ref)
        special = {name: n0 + i for i, name in enumerate(special_poses)}
        print(f"{len(poses):,} samples ({time.time() - t0:.0f} s); within {args.near_mm} mm of contact: "
              f"{np.mean(seps <= args.near_mm):.0%}", flush=True)

        t0 = time.time()
        tree = cKDTree(poses * SCALE)
        _, nbr = tree.query(poses * SCALE, k=args.k + 1)
        pairs = np.unique(np.sort(np.column_stack([np.repeat(np.arange(len(poses)), args.k), nbr[:, 1:].ravel()]), 1),
                          axis=0)
        edges, clear = check_pairs(poses, pairs, args.step, w)
        print(f"{len(pairs):,} neighbour pairs, {len(edges) / len(pairs):.0%} in line of sight "
              f"({time.time() - t0:.0f} s)", flush=True)
    local = np.ones(len(edges), bool)  # nearest-neighbour edges: transitions between modes

    if args.partners:
        # Long-range lines of sight: within a convex region distant poses still see each other, across
        # the eyelet they do not; this is what lets the spectral cut find the regions.
        t0 = time.time()
        rng = np.random.default_rng(args.seed)
        P = poses * SCALE
        found = []
        for c0 in range(0, len(P), 2000):
            rows = np.arange(c0, min(c0 + 2000, len(P)))
            cand = rng.integers(0, len(P), (len(rows), 8 * args.partners))
            d = np.linalg.norm(P[cand] - P[rows, None], axis=2)
            ok = (d <= args.radius) & (cand != rows[:, None])
            for r, i in enumerate(rows):
                for j in cand[r, ok[r]][: args.partners]:
                    found.append((min(i, j), max(i, j)))
        pairs = np.unique(np.array(found), axis=0)
        e2, c2 = check_pairs(poses, pairs, args.step, w)
        print(f"{len(pairs):,} long-range pairs (within {args.radius} mm-equivalent), {len(e2) / len(pairs):.0%} "
              f"in line of sight ({time.time() - t0:.0f} s)", flush=True)
        edges, clear = np.vstack([edges, e2]), np.concatenate([clear, c2])
        local = np.r_[local, np.zeros(len(e2), bool)]
    np.savez(out / "graph.npz", poses=poses, seps=seps, edges=edges, clear=clear, local=local)

    W = sp.coo_matrix((np.ones(2 * len(edges)), (np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]])),
                      shape=(len(poses), len(poses))).tocsr()
    W.data[:] = 1.0
    n_comp, comp = connected_components(W, directed=False)
    largest = np.argmax(np.bincount(comp))
    keep = comp == largest
    report["components"] = {"count": int(n_comp), "largest": int(keep.sum()),
                            "special_outside_largest": [s for s, i in special.items() if not keep[i]]}
    t0 = time.time()
    vals, vecs = spectral_top(W[keep][:, keep], args.max_k)
    k_gap = int(np.argmax(np.diff(vals)[1:args.max_k]) + 2)
    k = args.modes or k_gap
    lab_keep = cluster(vecs, k, args.seed)
    labels = np.full(len(poses), -1)
    labels[keep] = lab_keep
    print(f"spectral ({time.time() - t0:.0f} s): eigenvalues " + " ".join(f"{v:.4f}" for v in vals[:args.max_k + 1])
          + f"; largest eigengap -> {k} modes", flush=True)
    report.update(eigenvalues=vals.tolist(), k_eigengap=k_gap, k=k)

    mode_edges = {}
    for (i, j) in edges[local]:
        a, b = labels[i], labels[j]
        if a >= 0 and b >= 0 and a != b:
            key = (min(a, b), max(a, b))
            mode_edges[key] = mode_edges.get(key, 0) + 1
    modes = []
    for m in range(k):
        sel = labels == m
        modes.append({"mode": m, "samples": int(sel.sum()), "centre": poses[sel].mean(0).round(1).tolist(),
                      "lo": np.percentile(poses[sel], 5, axis=0).round(1).tolist(),
                      "hi": np.percentile(poses[sel], 95, axis=0).round(1).tolist(),
                      "median_sep_mm": float(np.median(seps[sel])),
                      "neighbours": sorted(int(b if a == m else a) for (a, b) in mode_edges if m in (a, b))})
    report["modes"] = modes
    report["mode_edges"] = [{"modes": list(map(int, key)), "lines_of_sight": cnt} for key, cnt in mode_edges.items()]
    goal_mode = int(labels[special["goal"]])
    report["goal_mode"] = goal_mode
    plot_modes(poses[keep], labels[keep], k, mode_edges,
               {s: int(np.count_nonzero(keep[:i])) for s, i in special.items() if keep[i]}, out / "modes.png")

    routes = {}
    for name, i in special.items():
        if name == "goal":
            continue
        entry = {"mode": int(labels[i]), "separation_mm": float(seps[i])}
        if labels[i] >= 0:
            entry["mode_route"] = mode_route(mode_edges, k, int(labels[i]), goal_mode)
        pid = int(name.split()[-1])
        if pid in args.solve:
            sample_path, bottleneck = widest_path(len(poses), edges, clear, i, special["goal"])
            if sample_path is None:
                entry["status"] = "no line-of-sight path to the goal"
            else:
                entry["sample_modes"] = [int(labels[s]) for s in sample_path]
                pts = poses[sample_path]
                short = vg.shortcut(pts, max(0.0, min(bottleneck, seps[i]) - 0.1), args.step)
                full = np.vstack([short, np.zeros(6)[None]])  # final lift into resting contact
                entry.update(roadmap_bottleneck_mm=float(bottleneck), segments=len(short) - 1,
                             checked_min_mm=float(vg.path_min(short, 0.1)), length=vg.arc_length(full))
                traj = out / f"route_pocket_{pid}.csv"
                vg.write_trajectory(traj, full)
                subprocess.run([sys.executable, str(SCRIPTS / "assembly_trajectory_visual_verification.py"),
                                "--csv", str(traj), "--out", str(traj.with_suffix(".mp4"))], check=True,
                               capture_output=True)
                entry["status"] = "valid" if entry["checked_min_mm"] >= 0 else "invalid"
        routes[name] = entry
        print(f"{name}: {entry}", flush=True)
    report["routes"] = routes
    report["seconds"] = round(time.time() - t_all)
    (out / "modes.json").write_text(json.dumps(report, indent=2))
    np.savez(out / "samples.npz", poses=poses, seps=seps, labels=labels)


if __name__ == "__main__":
    main()
