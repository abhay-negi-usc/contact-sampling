#!/usr/bin/env python3
"""Group hook poses by the trajectory they need to assemble.

Two start poses belong to the same group when their assembly trajectories can be slid into each other
through free space: after aligning the two trajectories (dynamic time warping in the lever-10 pose metric),
every straight segment ("rung") between matched points must be collision-free. The straight rungs make this
a convexity test: a collision-free ladder of segments sweeps one trajectory into the other, so the two are
the same route, homotopic in free space, while starts on either side of a decision (go round the loop's left
or right, thread from the front or from behind, back out first) fail it.

    paths   starts spread evenly over the goal field's free nodes (farthest-point sampling among a random
            --pool of them); each start's trajectory to the assembled pose from the goal field
            (validate_goal_field.solve: connect to the grid, follow the field's moves, shortcut, lift)
    routes  each start's nearest --k starts are tested pairwise; groups are the connected components of
            the pairs that pass; the pool's nodes take the group of their nearest start

    python scripts/hook_assembly_routes.py paths FIELD_DIR EXPERIMENT_DIR --out-dir OUT
    python scripts/hook_assembly_routes.py routes FIELD_DIR EXPERIMENT_DIR --out-dir OUT
"""

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

import validate_goal_field as vg
from goal_field import GoalField, node_coords
from pcca import pcca_scan

LEVER = 10.0
SCALE = np.array([1, 1, 1] + [LEVER * np.pi / 180] * 3)


def farthest_points(x, n, seed=0):
    rng = np.random.default_rng(seed)
    chosen = [int(rng.integers(len(x)))]
    d = np.linalg.norm(x - x[chosen[0]], axis=1)
    for _ in range(n - 1):
        chosen.append(int(np.argmax(d)))
        d = np.minimum(d, np.linalg.norm(x - x[chosen[-1]], axis=1))
    return np.array(chosen)


def paths_stage(args):
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    f = vg._field = GoalField.load_compact(args.field_dir / "goal_field_compact.npz")
    rng = np.random.default_rng(args.seed)
    nodes = rng.choice(len(f.V), args.pool, replace=False)
    nodes = nodes[f.V[nodes] >= 0]
    poses = np.empty((len(nodes), 6))
    c = np.empty(6, np.int64)
    for k, i in enumerate(nodes):
        node_coords(i, c, f.block_coords, f.words, f.prefix, f.offset)
        poses[k] = c
    idx = farthest_points(poses * SCALE, args.starts, args.seed)
    np.save(out / "pool.npy", poses)
    print(f"{len(idx)} starts among {len(poses):,} connected field nodes ({time.time() - t0:.0f} s)", flush=True)
    opts = SimpleNamespace(check_step=args.check_step, shortcut_tol_mm=args.shortcut_tol_mm)
    t0 = time.time()
    with mp.get_context("fork").Pool(args.workers, initializer=vg._init) as pool:
        res = pool.map(vg.solve, [(k, poses[i], opts) for k, i in enumerate(idx)], chunksize=4)
    ok = [r[1] is not None and r[0]["status"] == "valid" for r in res]
    status = [r[0]["status"] for r in res]
    print(f"trajectories ({time.time() - t0:.0f} s): " + ", ".join(f"{s}: {status.count(s)}" for s in set(status)),
          flush=True)
    paths = [r[1] for r, good in zip(res, ok) if good]
    lengths = np.array([len(p) for p in paths])
    np.savez(out / "paths.npz", start_index=idx[np.array(ok)], points=np.vstack(paths),
             offsets=np.concatenate([[0], np.cumsum(lengths)]),
             clearance=np.array([r[0]["short_min_mm"] for r, good in zip(res, ok) if good]),
             length=np.array([r[0]["short_length"] for r, good in zip(res, ok) if good]))


def load_paths(out):
    z = np.load(out / "paths.npz")
    pts, off = z["points"], z["offsets"]
    return z["start_index"], [pts[off[i]:off[i + 1]] for i in range(len(off) - 1)], z


def resample(path, n):
    """n points evenly spaced in arc length (lever-10 metric) along a polyline."""
    y = path * SCALE
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(y, axis=0), axis=1))]
    if s[-1] == 0:
        return np.repeat(path[:1], n, axis=0)
    t = np.linspace(0, s[-1], n)
    return np.column_stack([np.interp(t, s, path[:, d]) for d in range(6)])


def dtw(a, b):
    """Monotone alignment of two resampled paths minimising the summed rung length; matched index pairs."""
    ya, yb = a * SCALE, b * SCALE
    cost = np.linalg.norm(ya[:, None] - yb[None], axis=2)
    n, m = cost.shape
    acc = np.full((n + 1, m + 1), np.inf)
    acc[0, 0] = 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            acc[i, j] = cost[i - 1, j - 1] + min(acc[i - 1, j], acc[i, j - 1], acc[i - 1, j - 1])
    i, j, pairs = n, m, []
    while i > 0 and j > 0:
        pairs.append((i - 1, j - 1))
        step = np.argmin([acc[i - 1, j - 1], acc[i - 1, j], acc[i, j - 1]])
        i, j = (i - 1, j - 1) if step == 0 else (i - 1, j) if step == 1 else (i, j - 1)
    return pairs[::-1]


def ladder_task(job):
    """For each pair of trajectories: the smallest separation over their aligned rungs [mm] (stops below 0), and
    the trajectory distance, the mean rung length [mm, lever-10 metric]."""
    pairs, paths, n, step = job
    out = np.empty((len(pairs), 2))
    for e, (a, b) in enumerate(pairs):
        A, B = resample(paths[a], n), resample(paths[b], n)
        match = dtw(A, B)
        worst = np.inf
        for i, j in match:
            if np.abs((A[i] - B[j]) * SCALE).max() < 1e-6:
                continue
            worst = min(worst, vg.segment_min(A[i], B[j], step, floor=0.0))
            if worst < 0:
                break
        out[e] = worst, np.mean([np.linalg.norm((A[i] - B[j]) * SCALE) for i, j in match])
    return out


def routes_stage(args):
    out = args.out_dir
    start_index, paths, z = load_paths(out)
    field_nodes = np.load(out / "pool.npy")
    starts = field_nodes[start_index]
    x = starts * SCALE
    _, nbr = cKDTree(x).query(x, k=args.k + 1)
    pairs = np.unique(np.sort(np.column_stack([np.repeat(np.arange(len(x)), args.k), nbr[:, 1:].ravel()]), 1), axis=0)
    t0 = time.time()
    cache = out / "routes.npz"
    if args.reuse and cache.exists() and np.array_equal(np.load(cache)["pairs"], pairs):
        worst, dist = np.load(cache)["worst"], np.load(cache)["dist"]
    else:
        chunks = np.array_split(np.arange(len(pairs)), args.workers * 8)
        with mp.get_context("fork").Pool(args.workers, initializer=vg._init) as pool:
            res = np.vstack(pool.map(ladder_task, [(pairs[c], paths, args.rungs, args.check_step) for c in chunks]))
        worst, dist = res[:, 0], res[:, 1]
    same = worst >= 0
    print(f"{len(pairs):,} neighbouring start pairs; {same.mean():.0%} have trajectories that slide into each other "
          f"({time.time() - t0:.0f} s)", flush=True)
    A = sp.coo_matrix((np.ones(same.sum()), (pairs[same, 0], pairs[same, 1])), shape=(len(x), len(x)))
    _, group = connected_components(A, directed=False)
    sizes = np.bincount(group)
    order = np.argsort(-sizes)
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    group = rank[group]
    sizes = np.bincount(group)
    print(f"{len(sizes)} groups; sizes " + ", ".join(map(str, sizes[:15])) + (" ..." if len(sizes) > 15 else ""),
          flush=True)
    # every pool node takes the group of its nearest start
    _, near = cKDTree(x).query(field_nodes * SCALE)
    labels = group[near]
    # PCCA+ on the chain over starts: likely moves between starts with similar trajectories, rare across a
    # failed ladder (different routes)
    w = np.exp(-dist ** 2 / np.median(dist) ** 2) * np.where(same, 1.0, args.fail_weight)
    W = sp.coo_matrix((np.r_[w, w], (np.r_[pairs[:, 0], pairs[:, 1]], np.r_[pairs[:, 1], pairs[:, 0]])),
                      shape=(len(x), len(x))).tocsr()
    W.eliminate_zeros()
    ncomp, comp = connected_components(W, directed=False)
    main = comp == np.argmax(np.bincount(comp))
    Wm = W[main][:, main]
    scan, vals = pcca_scan(Wm, range(2, args.max_k + 1))
    print(f"PCCA+ on {main.sum()} starts (largest of {ncomp} components); eigenvalues "
          + " ".join(f"{v:.4f}" for v in vals[:args.max_k + 1]), flush=True)
    memberships = {}
    for k, (chi, crisp) in scan.items():
        lab = np.full(len(x), -1)
        lab[main] = chi.argmax(1)
        sz = np.sort(np.bincount(lab[main]))[::-1]
        print(f"   k={k:2d}  crispness {crisp:.3f}  sizes {', '.join(map(str, sz))}  confidently assigned "
              f"(max membership > 0.8): {np.mean(chi.max(1) > 0.8):.0%}", flush=True)
        full = np.zeros((len(x), k))
        full[main] = chi
        memberships[f"chi_{k}"] = full
    np.savez(out / f"routes{args.tag}.npz", pairs=pairs, worst=worst, dist=dist, group=group, labels=labels, near=near,
             main=main, eigenvalues=vals, crispness=np.array([scan[k][1] for k in scan]), ks=np.array(list(scan)),
             **memberships)
    (out / f"routes{args.tag}.json").write_text(json.dumps({"groups": int(len(sizes)), "sizes": sizes.tolist(),
                                                 "pair_pass_share": float(same.mean()),
                                                 "pcca_crispness": {int(k): float(scan[k][1]) for k in scan}},
                                                indent=2))


def describe(path, loop):
    """Physical descriptors of one assembly trajectory (poses in mm / deg)."""
    y = path * SCALE
    to_goal = np.linalg.norm(y - y[-1], axis=1)
    return {"length_mm": float(np.sum(np.linalg.norm(np.diff(y, axis=0), axis=1))),
            "start_threaded": int(loop.threading(path[0]) > 0),
            "behind_share": float(np.mean(path[:, 0] > 2.0)),
            "backs_off_mm": float(max(0.0, to_goal.max() - to_goal[0])),
            "pitch_range_deg": float(np.ptp(path[:, 4])),
            "z_max_mm": float(path[:, 2].max()), "z_min_mm": float(path[:, 2].min())}


def plot_stage(args):
    import os
    os.environ.setdefault("MUJOCO_GL", "egl")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from PIL import Image, ImageDraw, ImageFont

    from hook_topology import HookLoop
    from render_mode_examples import Renderer

    out = args.out_dir
    start_index, paths, z = load_paths(out)
    r = np.load(out / "routes.npz")
    field_nodes = np.load(out / "pool.npy")
    starts = field_nodes[start_index]
    if args.labels == "groups":
        labels, name = r["group"], "ladder groups"
    else:
        labels, name = r[f"chi_{args.labels}"].argmax(1), f"PCCA+ k={args.labels}"
        labels[~r["main"]] = -1
    k = labels.max() + 1
    sizes = np.bincount(labels[labels >= 0], minlength=k)
    order = np.argsort(-sizes)
    res = [resample(p, args.rungs) for p in paths]
    loop = HookLoop(n_centre=8000)
    groups = []
    for g in order:
        mem = np.flatnonzero(labels == g)
        if len(mem) == 0:
            continue
        R = np.array([res[i] for i in mem]) * SCALE
        cost = np.array([np.mean(np.linalg.norm(R - R[j], axis=2)) for j in range(len(mem))]) if len(mem) < 1500 else \
            np.linalg.norm(R.mean(0) - R, axis=2).mean(1)
        med = int(mem[np.argmin(cost)])
        desc = [describe(paths[i], loop) for i in mem[: min(len(mem), 300)]]
        summary = {key: float(np.median([d[key] for d in desc])) for key in desc[0]}
        summary["start_threaded"] = float(np.mean([d["start_threaded"] for d in desc]))
        groups.append({"group": int(g), "starts": int(len(mem)), "medoid_start": int(med),
                       "start_median": np.round(np.median(starts[mem], 0), 1).tolist(), **summary})
        print(f"group {g:2d}: {len(mem):4d} starts  start median {np.round(np.median(starts[mem], 0), 0)}  length "
              f"{summary['length_mm']:5.1f} mm  threaded at start {summary['start_threaded']:4.0%}  behind "
              f"{summary['behind_share']:4.0%}  backs off {summary['backs_off_mm']:4.1f} mm  pitch range "
              f"{summary['pitch_range_deg']:4.0f} deg", flush=True)
    (out / f"route_groups_{args.labels}.json").write_text(json.dumps({"labels": name, "groups": groups}, indent=2))

    hues = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
    colour = {gr["group"]: (hues[i] if i < len(hues) else "#b9b8b1") for i, gr in enumerate(groups)}
    xpz = lambda q: (q[..., 0], q[..., 4], q[..., 2])
    rng = np.random.default_rng(0)
    fig = plt.figure(figsize=(8, 7), facecolor="#fcfcfb")
    ax = fig.add_subplot(projection="3d")
    ax.computed_zorder = False
    for gr in groups[::-1]:
        mem = np.flatnonzero(labels == gr["group"])
        for i in rng.choice(mem, min(25, len(mem)), replace=False):
            ax.plot(*xpz(paths[i]), color=colour[gr["group"]], linewidth=0.5, alpha=0.35, zorder=1)
    for gr in groups[:len(hues)]:
        ax.plot(*xpz(paths[gr["medoid_start"]]), color=colour[gr["group"]], linewidth=2.6, zorder=3,
                label=f"group {gr['group']}: {gr['starts']} starts")
        ax.scatter(*xpz(paths[gr["medoid_start"]][0]), color=colour[gr["group"]], s=30, zorder=4)
    ax.scatter(0, 0, 0, color="#0b0b0b", marker="x", s=40, zorder=5)
    ax.set_xlabel("x [mm]", fontsize=8), ax.set_ylabel("pitch (ry) [deg]", fontsize=8), ax.set_zlabel("z [mm]", fontsize=8)
    ax.tick_params(labelsize=6, colors="#52514e")
    fig.legend(loc="lower center", ncol=3, fontsize=7, frameon=False)
    fig.suptitle(f"Assembly trajectories by route group ({name}); thick: each group's most typical one", fontsize=10)
    anim = FuncAnimation(fig, lambda f: ax.view_init(elev=22, azim=-60 + 360 * f / 60), frames=60)
    anim.save(out / f"route_groups_{args.labels}.gif", writer=PillowWriter(fps=10))
    ax.view_init(elev=22, azim=-60)
    fig.savefig(out / f"route_groups_{args.labels}.png", facecolor="#fcfcfb", dpi=110)
    plt.close(fig)

    # film strip of each group's typical trajectory: frames along it, front and side views
    renderer = Renderer()
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 15)
    except OSError:
        font = ImageFont.load_default()
    strips = []
    for gr in groups[: args.strips]:
        path = paths[gr["medoid_start"]]
        frames = resample(path, args.frames)
        row = []
        for q in frames:
            front, _ = renderer.views(q, np.array([0.0, 0.0, -0.004]), 0.16)
            row.append(front.resize((180, 180)))
        strip = Image.new("RGB", (180 * len(row), 210), (252, 252, 251))
        ImageDraw.Draw(strip).text((6, 4), f"group {gr['group']} ({gr['starts']} starts): length {gr['length_mm']:.0f} mm, "
                                           f"threaded at start {gr['start_threaded']:.0%}, behind the loop "
                                           f"{gr['behind_share']:.0%} of the way, backs off {gr['backs_off_mm']:.1f} mm",
                                   fill=(11, 11, 11), font=font)
        for c, im in enumerate(row):
            strip.paste(im, (180 * c, 30))
        strips.append(strip)
    renderer.close()
    sheet = Image.new("RGB", (strips[0].width, sum(st.height for st in strips)), (252, 252, 251))
    y0 = 0
    for st in strips:
        sheet.paste(st, (0, y0))
        y0 += st.height
    sheet.save(out / f"route_strips_{args.labels}.png")
    # a mode directory of the starts, for render_mode_examples.py / plot_hook_modes.py
    md = out / f"modes_{args.labels}"
    md.mkdir(exist_ok=True)
    np.savez(md / "samples.npz", poses=starts, seps=z["clearance"], labels=labels)
    goal = int(labels[np.argmin(np.linalg.norm((starts - np.array([0, 0, -3, 0, 0, 0])) * SCALE, axis=1))])
    (md / "modes.json").write_text(json.dumps({"k": int(k), "goal_mode": goal, "mode_edges": [],
                                               "modes": [{"mode": int(g), "name": f"route group {g}"} for g in range(k)]}))


def cells_stage(args):
    """Route signatures from convex free cells (hook_contact_strata.py convex): the sequence of cells each
    trajectory passes through (deepest containing cell at each point, repeats collapsed). Starts whose
    signatures match after their first cell share a route; compared with the ladder and PCCA+ groups."""
    from sklearn.metrics import adjusted_rand_score

    out = args.out_dir
    start_index, paths, _ = load_paths(out)
    cz = np.load(args.experiment_dir / "strata" / "convex_cells.npz")
    n_cells = len([k for k in cz.files if k.startswith("A")])
    cells = [(cz[f"A{c}"], cz[f"b{c}"]) for c in range(n_cells)]

    def cell_of(y):
        best, depth = -1, 0.0
        for c, (A, b) in enumerate(cells):
            d = np.min(b - A @ y)
            if d >= depth:
                best, depth = c, d
        return best

    sigs = []
    for path in paths:
        seq = [cell_of(q * SCALE) for q in resample(path, args.rungs)]
        seq = [c for c in seq if c >= 0]
        sig = [c for i, c in enumerate(seq) if i == 0 or c != seq[i - 1]]
        sigs.append(tuple(sig[1:]))  # after the start's own cell
    keys, sig_label = np.unique(np.array([hash(t) for t in sigs]), return_inverse=True)
    r = np.load(out / "routes.npz")
    print(f"{len(keys)} distinct cell routes over {len(paths)} trajectories; covered points: "
          f"{np.mean([len(t) > 0 for t in sigs]):.0%} of trajectories pass through cells after their first", flush=True)
    sizes = np.sort(np.bincount(sig_label))[::-1]
    print("largest routes: " + ", ".join(map(str, sizes[:12])), flush=True)
    for name in ["group"] + [f"chi_{k}" for k in r["ks"]]:
        lab = r[name] if name == "group" else r[name].argmax(1)
        print(f"   ARI cell routes vs {name}: {adjusted_rand_score(sig_label, lab):.3f}", flush=True)
    np.save(out / "cell_routes.npy", sig_label)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["paths", "routes", "plot", "cells"])
    parser.add_argument("field_dir", type=Path)
    parser.add_argument("experiment_dir", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--starts", type=int, default=4000)
    parser.add_argument("--pool", type=int, default=200000, help="random field nodes the starts are spread over")
    parser.add_argument("--k", type=int, default=8, help="neighbouring starts tested per start")
    parser.add_argument("--rungs", type=int, default=48, help="points per trajectory when aligning")
    parser.add_argument("--check-step", type=float, default=0.25, help="[mm / deg]")
    parser.add_argument("--shortcut-tol-mm", type=float, default=0.25)
    parser.add_argument("--fail-weight", type=float, default=0.05,
                        help="PCCA+ chain: weight kept by neighbours whose trajectories do not slide into each other")
    parser.add_argument("--max-k", type=int, default=20)
    parser.add_argument("--reuse", action="store_true", help="routes: reuse the pair tests in routes.npz")
    parser.add_argument("--tag", default="", help="routes: suffix of the output files")
    parser.add_argument("--labels", default="groups", help="plot: 'groups' (ladder components) or a PCCA+ k")
    parser.add_argument("--strips", type=int, default=8, help="plot: groups shown as film strips")
    parser.add_argument("--frames", type=int, default=10, help="plot: frames per film strip")
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    {"paths": paths_stage, "routes": routes_stage, "plot": plot_stage, "cells": cells_stage}[args.stage](args)


if __name__ == "__main__":
    main()
