#!/usr/bin/env python3
"""Compare ways of building the connectivity graph for spectral discovery of the hook's pose modes.

Two node sets:
  samples  collision-free hook poses around the reference trajectory (as in hook_mode_graph.py), with
           line-of-sight edges to nearest neighbours under several pose metrics, and long-range ones
  grid     the separation field's free nodes on a --stride lattice of its 1 mm / 1 deg grid, joined to the
           next lattice node along each axis when every 1 mm node between them is free too: continuity at
           1 mm tolerance, with the smallest separation along the way as the edge's clearance

Every node gets physical labels to score clusters against (hook_topology.py): which hook parts thread the
loop's hole, which hook part and eyelet part are closest (from the closest-point pair), and the side of
the loop plane the hook's origin is on. A variant is a weighting of one node set's edges; each is
clustered as in contact_graph_spectral.py (normalized Laplacian, largest eigengap, k-means on the
row-normalised eigenvectors), at its eigengap k and at fixed k, optionally followed by smoothing (each
node takes the most common label of its neighbours, twice). Scores: purity, NMI and ARI against the
threading state and against the full physical label; local disagreement (share of a node's edges that
cross a cluster boundary); and the size of the eigengap.

    python scripts/mode_graph_experiments.py prepare-samples FIELD_DIR --out-dir OUT
    python scripts/mode_graph_experiments.py prepare-grid FIELD_DIR --out-dir OUT
    python scripts/mode_graph_experiments.py prepare-keypoints --out-dir OUT
    python scripts/mode_graph_experiments.py prepare-levers --out-dir OUT [--levers 2.5 5]
    python scripts/mode_graph_experiments.py prepare-wrench --out-dir OUT
    python scripts/mode_graph_experiments.py prepare-contact --out-dir OUT
    python scripts/mode_graph_experiments.py run --out-dir OUT [--only REGEX]
    python scripts/mode_graph_experiments.py run-basins --out-dir OUT

prepare-keypoints adds line-of-sight nearest-neighbour edges under a geometric pose metric to the samples:
the RMS displacement of points spread along the hook's centreline, which scales each rotation by how far
it moves the hook rather than by one lever. A run with --only reruns the matching variants and merges
their rows into experiments.json.

prepare-wrench adds contact wrenches to the samples. With the closest points p_hook, p_eyelet and the
normal n = (p_hook - p_eyelet) / |.| (the direction the eyelet pushes the hook), a unit push gives the
wrench [n, r x n] about the hook origin (r = p_hook - hook origin): the gradient of the separation with
respect to the hook's translation and rotation. The torque is divided by a lever L [mm], the dual of the
pose metric's scaling, before taking directions. Two versions:
  closest   the closest-point pair only (its force direction can flip where two features are equally close)
  multi     every hull pair within --wrench-band of the closest, weighted exp(-(d - d_min) / --wrench-decay),
            so two-point contacts and changes of the closest feature blend continuously
It also adds line-of-sight nearest-neighbour edges in a joint pose + wrench-direction space.

prepare-contact builds a third node set of poses in contact only: every sample (and --contact-copies
jittered copies of it) slides straight along its contact normal until the separation is within
--contact-tol of --contact-target. Separation cannot drop faster than the hook moves, so the slide stops at
contact; from inside, the normal flips and the same step pushes out. The contact poses are thinned to
one per --contact-cell cell of the lever-10 pose metric. Their nearest neighbours are joined by a sliding
check: along the straight segment, the separation's largest magnitude (how far the segment strays from
contact) and the largest jump of the segment projected back onto contact (where the touching feature
changes abruptly) are recorded, so variants can keep edges that slide within a tolerance.

run-basins clusters by clearance instead of by cuts: persistence-based clustering (ToMATo, Chazal et al.
2013) of the separation over a node set's continuity edges. Nodes are swept from the most to the least
clearance; a node joins the basin of its highest neighbour, and two basins meeting at a node (a door,
whose clearance is that node's) merge unless both peaks stand at least tau above the door and both
basins have --min-basin nodes by then. Each remaining basin is a mode; neighbouring modes are joined by
the clearance of their widest door.
"""

import argparse
import json
import multiprocessing as mp
import re
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import scipy.sparse as sp
from numba import njit
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

import validate_goal_field as vg
from build_separation_field import UNIT_MM, reference_path
from contact_graph_spectral import cluster
from goal_field import PRE_ASSEMBLY, STEPS, node_coords, node_id
from hook_eyelet_model import ASSET_DIR, Separation, build_model
from hook_mode_graph import sample_task, spectral_top
from hook_topology import CROOK, HookLoop, eyelet_part, hook_part

LEVERS = (10.0, 30.0, 60.0)  # mm of translation counted per radian of rotation, in the pose metric
_probe = None
_loop = None


class Probe:
    """Separation and closest-point pair at a pose."""

    def __init__(self):
        self.m = build_model()
        self.d = mujoco.MjData(self.m)
        self.sep = Separation(self.m)

    def __call__(self, pose):
        rot = Rotation.from_rotvec(pose[3:], degrees=True)
        self.d.qpos[:3], self.d.qpos[3:7] = pose[:3] * 1e-3, rot.as_quat(scalar_first=True)
        mujoco.mj_kinematics(self.m, self.d)
        dist, pts = self.sep(self.m, self.d)
        if pts is None:
            return dist * 1e3, np.full(3, np.nan), np.full(3, np.nan)
        return dist * 1e3, rot.inv().apply(pts[0] * 1e3 - pose[:3]), pts[1] * 1e3

    def normal(self, pose):
        """Separation [mm] and the unit normal along which separating moves the hook out of contact."""
        rot = Rotation.from_rotvec(pose[3:], degrees=True)
        self.d.qpos[:3], self.d.qpos[3:7] = pose[:3] * 1e-3, rot.as_quat(scalar_first=True)
        mujoco.mj_kinematics(self.m, self.d)
        dist, pts = self.sep(self.m, self.d, distmax=0.2)
        if pts is None:
            return dist * 1e3, np.full(3, np.nan)
        n = pts[0] - pts[1]
        return dist * 1e3, np.sign(dist) * n / np.linalg.norm(n)

    def wrench(self, pose, band, decay):
        """Closest-point and multi-contact wrenches [n, r x n] (torque in mm) for a unit push; see prepare-wrench."""
        rot = Rotation.from_rotvec(pose[3:], degrees=True)
        self.d.qpos[:3], self.d.qpos[3:7] = pose[:3] * 1e-3, rot.as_quat(scalar_first=True)
        mujoco.mj_kinematics(self.m, self.d)
        sep, m, d = self.sep, self.m, self.d
        d_min, pts = sep(m, d)
        if pts is None:
            return np.full(6, np.nan), np.full(6, np.nan)
        lower = np.linalg.norm(d.geom_xpos[sep.g1] - d.geom_xpos[sep.g2], axis=1) - sep.rsum
        bound = d_min + band * 1e-3
        unit = lambda ph, pe: np.r_[(ph - pe) / np.linalg.norm(ph - pe),
                                    np.cross(ph - pose[:3], (ph - pe) / np.linalg.norm(ph - pe))]
        closest = unit(pts[0] * 1e3, pts[1] * 1e3)
        total, wsum = np.zeros(6), 0.0
        for k in np.argsort(lower):
            if lower[k] >= bound:
                break
            dist = mujoco.mj_geomDistance(m, d, sep.g1[k], sep.g2[k], bound, sep._fromto)
            ph, pe = sep._fromto[:3] * 1e3, sep._fromto[3:] * 1e3
            if dist < bound and dist > 0 and np.linalg.norm(ph - pe) > 1e-9:
                w = np.exp(-(dist - d_min) * 1e3 / decay)
                total += w * unit(ph, pe)
                wsum += w
        return closest, total / wsum if wsum > 0 else closest


def _init_features():
    global _probe
    _probe = Probe()
    vg._init()


def feature_task(poses):
    out = np.empty((len(poses), 7))
    thread = np.empty(len(poses), np.int8)
    for i, p in enumerate(poses):
        s, hw, ew = _probe(p)
        out[i] = [s, *hw, *ew]
        thread[i] = _loop.threading(p)
    return out, thread


def wrench_task(job):
    poses, band, decay = job
    out = np.empty((len(poses), 12))
    for i, p in enumerate(poses):
        out[i, :6], out[i, 6:] = _probe.wrench(p, band, decay)
    return out


def contact_project(pose, target, tol, iters=8):
    """Slide the hook along its contact normal until the separation is within tol of target [mm]."""
    p = pose.copy()
    for _ in range(iters):
        s, n = _probe.normal(p)
        if abs(s - target) <= tol:
            return p, True
        if np.isnan(n[0]):
            break
        p[:3] -= (s - target) * n
    return p, False


def project_task(job):
    poses, target, tol = job
    out, ok = np.empty_like(poses), np.zeros(len(poses), bool)
    for i, p in enumerate(poses):
        out[i], ok[i] = contact_project(p, target, tol)
    return out, ok


def slide_task(job):
    """Per segment: largest |separation| along it, and largest jump of its projection onto contact [mm]."""
    a, b, step, target, reach = job
    out = np.empty((len(a), 2))
    for e, (p, q) in enumerate(zip(a, b)):
        worst, jump, prev = 0.0, 0.0, np.zeros(3)
        for c in vg.densify(p, q, step, include_end=True):
            s, n = _probe.normal(c)
            worst = max(worst, abs(s))
            if worst > reach:
                break
            shift = -(s - target) * n  # one projection step: exact while one feature touches
            jump = max(jump, np.linalg.norm(shift - prev)) if prev.any() else jump
            prev = shift
        out[e] = worst, jump
    return out


def features(poses, workers):
    """separation [mm], hook-frame and eyelet-frame closest points [mm], threading bitmask."""
    chunks = np.array_split(np.arange(len(poses)), workers * 8)
    with mp.get_context("fork").Pool(workers, initializer=_init_features) as pool:
        res = pool.map(feature_task, [poses[c] for c in chunks])
    return np.vstack([r[0] for r in res]), np.concatenate([r[1] for r in res])


def physical_labels(poses, feat, thread):
    """Threading state, and the full label: threading, closest hook part, closest eyelet part, side."""
    hp = np.where(np.isnan(feat[:, 1]), 0, hook_part(np.nan_to_num(feat[:, 1:4])))
    ep = np.where(np.isnan(feat[:, 4]), 9, eyelet_part(np.nan_to_num(feat[:, 4:7])))
    side = (poses[:, 0] > 0).astype(int)
    full = thread.astype(int) * 1000 + hp * 100 + ep * 10 + side
    return thread.astype(int), full


def check_task(job):
    a, b, step = job
    return np.array([vg.segment_min(p, q, step, floor=0.0) for p, q in zip(a, b)])


def prepare_samples(args):
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    global _loop
    _loop = HookLoop(n_centre=8000)
    path = reference_path(args.reference, spacing=0.25)
    vg._init()
    t0 = time.time()
    w = args.workers
    jobs = [(path, 10.0 + args.reach, args.samples // w + 1, args.near_mm, args.far_keep, args.seed + i) for i in range(w)]
    with mp.get_context("fork").Pool(w, initializer=_init_features) as pool:
        res = pool.map(sample_task, jobs)
    poses = np.vstack([r[0] for r in res] + [PRE_ASSEMBLY[None].astype(float), path[:: len(path) // 150]])
    feat, thread = features(poses, w)
    print(f"{len(poses):,} samples with features ({time.time() - t0:.0f} s)", flush=True)

    t0 = time.time()
    pairs, member = [], {}
    for lever in LEVERS:
        scale = np.array([1, 1, 1] + [lever * np.pi / 180] * 3)
        _, nbr = cKDTree(poses * scale).query(poses * scale, k=args.k + 1)
        pr = np.sort(np.column_stack([np.repeat(np.arange(len(poses)), args.k), nbr[:, 1:].ravel()]), 1)
        member[lever] = pr
        pairs.append(pr)
    rng = np.random.default_rng(args.seed)
    P = poses * np.array([1, 1, 1] + [30 * np.pi / 180] * 3)
    far = []
    for c0 in range(0, len(P), 2000):
        rows = np.arange(c0, min(c0 + 2000, len(P)))
        cand = rng.integers(0, len(P), (len(rows), 8 * args.partners))
        ok = (np.linalg.norm(P[cand] - P[rows, None], axis=2) <= args.radius) & (cand != rows[:, None])
        for r, i in enumerate(rows):
            far.extend((min(i, j), max(i, j)) for j in cand[r, ok[r]][: args.partners])
    member["far"] = np.array(far)
    pairs = np.unique(np.vstack(pairs + [member["far"]]), axis=0)
    chunks = np.array_split(np.arange(len(pairs)), w * 8)
    with mp.get_context("fork").Pool(w, initializer=_init_features) as pool:
        clear = np.concatenate(pool.map(check_task, [(poses[pairs[c, 0]], poses[pairs[c, 1]], args.step)
                                                     for c in chunks]))
    key = lambda pr: pr[:, 0].astype(np.int64) * len(poses) + pr[:, 1]
    k_all = key(pairs)
    masks = {str(name): np.isin(k_all, key(pr)) for name, pr in member.items()}
    print(f"{len(pairs):,} pairs, {np.mean(clear >= 0):.0%} in line of sight ({time.time() - t0:.0f} s)", flush=True)
    np.savez(out / "samples_graph.npz", poses=poses, feat=feat, thread=thread, pairs=pairs, clear=clear,
             **{f"in_{k}": v for k, v in masks.items()})


def hook_keypoints(n):
    """n points spread along the hook's centreline [mm, hook frame], farthest-point sampled from the crook's bottom."""
    c = HookLoop(n_centre=8000).centre
    idx = [int(np.argmin(c[:, 2]))]
    d = np.linalg.norm(c - c[idx[0]], axis=1)
    for _ in range(n - 1):
        idx.append(int(np.argmax(d)))
        d = np.minimum(d, np.linalg.norm(c - c[idx[-1]], axis=1))
    return c[idx]


def keypoint_embedding(poses, kp):
    """Hook keypoints in the eyelet frame, flattened and scaled so Euclidean distance is RMS displacement [mm]."""
    rot = Rotation.from_rotvec(poses[:, 3:], degrees=True)
    pts = np.stack([rot.apply(k) for k in kp], axis=1) + poses[:, None, :3]
    return pts.reshape(len(poses), -1) / np.sqrt(len(kp))


def add_knn_edges(g, emb, mask, args):
    """Add line-of-sight-checked nearest-neighbour pairs under embedding emb to graph g, marked g[mask]."""
    poses = g["poses"]
    t0 = time.time()
    _, nbr = cKDTree(emb).query(emb, k=args.k + 1)
    new = np.unique(np.sort(np.column_stack([np.repeat(np.arange(len(poses)), args.k), nbr[:, 1:].ravel()]), 1),
                    axis=0)
    key = lambda pr: pr[:, 0].astype(np.int64) * len(poses) + pr[:, 1]
    todo = new[~np.isin(key(new), key(g["pairs"]))]
    chunks = np.array_split(np.arange(len(todo)), args.workers * 8)
    vg._init()
    with mp.get_context("fork").Pool(args.workers, initializer=_init_features) as pool:
        clear = np.concatenate(pool.map(check_task, [(poses[todo[c, 0]], poses[todo[c, 1]], args.step)
                                                     for c in chunks]))
    for name in [k for k in g if k.startswith("in_")]:
        g[name] = np.r_[g[name], np.zeros(len(todo), bool)]
    g["pairs"], g["clear"] = np.vstack([g["pairs"], todo]), np.r_[g["clear"], clear]
    g[mask] = np.isin(key(g["pairs"]), key(new))
    print(f"{mask}: {len(new):,} pairs, {len(todo):,} new, {np.mean(clear >= 0):.0%} of those in line of sight "
          f"({time.time() - t0:.0f} s)", flush=True)


def prepare_keypoints(args):
    path = args.out_dir / "samples_graph.npz"
    g = dict(np.load(path))
    g["keypoints"] = hook_keypoints(args.keypoints)
    add_knn_edges(g, keypoint_embedding(g["poses"], g["keypoints"]), "in_keypoints", args)
    np.savez(path, **g)


def prepare_levers(args):
    path = args.out_dir / "samples_graph.npz"
    g = dict(np.load(path))
    for lever in args.levers:
        add_knn_edges(g, g["poses"] * np.array([1, 1, 1] + [lever * np.pi / 180] * 3), f"in_{float(lever)}", args)
    np.savez(path, **g)


def wrench_direction(g, kind, lever):
    """Unit 6-vectors [F, T / lever] of the samples' wrenches (kind: closest or multi); lever None: unit force only."""
    W = g[f"wrench_{kind}"].copy()
    if lever is None:
        W = W[:, :3]
    else:
        W[:, 3:] /= lever
    return np.nan_to_num(W / np.linalg.norm(W, axis=1, keepdims=True))


def prepare_wrench(args):
    path = args.out_dir / "samples_graph.npz"
    g = dict(np.load(path))
    poses = g["poses"]
    t0 = time.time()
    chunks = np.array_split(np.arange(len(poses)), args.workers * 8)
    with mp.get_context("fork").Pool(args.workers, initializer=_init_features) as pool:
        W = np.vstack(pool.map(wrench_task, [(poses[c], args.wrench_band, args.wrench_decay) for c in chunks]))
    g["wrench_closest"], g["wrench_multi"] = W[:, :6], W[:, 6:]
    cos = np.sum(wrench_direction(g, "closest", 10.0) * wrench_direction(g, "multi", 10.0), axis=1)
    print(f"wrenches for {len(poses):,} samples ({time.time() - t0:.0f} s); closest vs multi-contact direction: "
          f"median {np.degrees(np.median(np.arccos(np.clip(cos, -1, 1)))):.1f} deg, "
          f"{np.mean(cos < np.cos(np.radians(30))):.1%} differ by over 30 deg", flush=True)
    emb = np.c_[poses * np.array([1, 1, 1] + [30.0 * np.pi / 180] * 3),
                args.wrench_mm * wrench_direction(g, "multi", 30.0)]
    add_knn_edges(g, emb, "in_posewrench", args)
    np.savez(path, **g)


def prepare_contact(args):
    out = args.out_dir
    src = np.load(out / "samples_graph.npz")["poses"]
    rng = np.random.default_rng(args.seed)
    jitter = lambda x: x + rng.uniform(-1, 1, x.shape) * np.array([2.0] * 3 + [5.0] * 3)
    starts = np.vstack([src] + [jitter(src) for _ in range(args.contact_copies)])
    w = args.workers
    t0 = time.time()
    chunks = np.array_split(np.arange(len(starts)), w * 8)
    with mp.get_context("fork").Pool(w, initializer=_init_features) as pool:
        res = pool.map(project_task, [(starts[c], args.contact_target, args.contact_tol) for c in chunks])
    poses, ok = np.vstack([r[0] for r in res]), np.concatenate([r[1] for r in res])
    poses = poses[ok]
    cell = np.floor(poses * np.array([1, 1, 1] + [10.0 * np.pi / 180] * 3) / args.contact_cell).astype(np.int64)
    _, first = np.unique(cell, axis=0, return_index=True)
    poses = poses[np.sort(first)]
    print(f"{len(starts):,} starts, {ok.mean():.1%} reached contact, {len(poses):,} after thinning to one per "
          f"{args.contact_cell} mm cell ({time.time() - t0:.0f} s)", flush=True)
    global _loop
    _loop = HookLoop(n_centre=8000)
    feat, thread = features(poses, w)
    with mp.get_context("fork").Pool(w, initializer=_init_features) as pool:
        W = np.vstack(pool.map(wrench_task, [(poses[c], args.wrench_band, args.wrench_decay)
                                             for c in np.array_split(np.arange(len(poses)), w * 8)]))
    print(f"features and wrenches ({time.time() - t0:.0f} s); separation p50 {np.median(feat[:, 0]):.3f} mm, "
          f"max {feat[:, 0].max():.3f} mm; threaded {np.mean(thread > 0):.1%}", flush=True)

    t0 = time.time()
    pairs, member = [], {}
    for lever in args.contact_levers:
        x = poses * np.array([1, 1, 1] + [lever * np.pi / 180] * 3)
        _, nbr = cKDTree(x).query(x, k=args.k + 1)
        member[lever] = np.sort(np.column_stack([np.repeat(np.arange(len(poses)), args.k), nbr[:, 1:].ravel()]), 1)
        pairs.append(member[lever])
    pairs = np.unique(np.vstack(pairs), axis=0)
    chunks = np.array_split(np.arange(len(pairs)), w * 16)
    with mp.get_context("fork").Pool(w, initializer=_init_features) as pool:
        slide = np.vstack(pool.map(slide_task, [(poses[pairs[c, 0]], poses[pairs[c, 1]], args.step, args.contact_target,
                                                 args.contact_reach) for c in chunks]))
    key = lambda pr: pr[:, 0].astype(np.int64) * len(poses) + pr[:, 1]
    print(f"{len(pairs):,} neighbour pairs; segment strays from contact by p50 {np.median(slide[:, 0]):.2f} mm, "
          f"{np.mean(slide[:, 0] <= 0.25):.0%} within 0.25 mm, {np.mean(slide[:, 0] <= 1):.0%} within 1 mm; projection "
          f"jump p50 {np.median(slide[:, 1]):.2f} mm ({time.time() - t0:.0f} s)", flush=True)
    np.savez(out / "contact_graph.npz", poses=poses, feat=feat, thread=thread, pairs=pairs, stray=slide[:, 0],
             jump=slide[:, 1], clear=-slide[:, 0], wrench_closest=W[:, :6], wrench_multi=W[:, 6:],
             **{f"in_{float(k)}": np.isin(key(pairs), key(v)) for k, v in member.items()})


@njit(cache=True)
def lattice(stride, sep, block_coords, lut, lut_lo, lut_shape, words, prefix, offset):
    """Free nodes with all coordinates divisible by stride, and their axis edges to the next lattice node
    when every grid node between is free; the edge's clearance is the smallest separation along it."""
    n = len(sep)
    c, q = np.empty(6, np.int64), np.empty(6, np.int64)
    keep = np.zeros(n, np.bool_)
    for i in range(n):
        node_coords(i, c, block_coords, words, prefix, offset)
        ok = True
        for d in range(6):
            if c[d] % stride != 0:
                ok = False
                break
        keep[i] = ok
    ids = np.flatnonzero(keep)
    ea, eb, ec = [], [], []
    for i in ids:
        node_coords(i, c, block_coords, words, prefix, offset)
        for d in range(6):
            low, j = sep[i], -1
            for s in range(1, stride + 1):
                for e in range(6):
                    q[e] = c[e]
                q[d] = c[d] + s
                j = node_id(q, lut, lut_lo, lut_shape, words, prefix, offset)
                if j < 0:
                    break
                low = min(low, sep[j])
            if j >= 0:
                ea.append(i)
                eb.append(j)
                ec.append(low)
    return ids, np.array(ea), np.array(eb), np.array(ec)


def prepare_grid(args):
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    global _loop
    _loop = HookLoop(n_centre=8000)
    t0 = time.time()
    with np.load(args.field_dir / "goal_field_index.npz") as z:
        a = {k: z[k] for k in z.files}
    ids, ea, eb, ec = lattice(args.stride, a["sep"], a["block_coords"], a["lut"], a["lut_lo"], a["lut_shape"],
                              a["words"], a["prefix"], a["offset"])
    where = -np.ones(len(a["sep"]), np.int64)
    where[ids] = np.arange(len(ids))
    pairs = np.column_stack([where[ea], where[eb]])
    c = np.empty(6, np.int64)
    poses = np.empty((len(ids), 6))
    for k, i in enumerate(ids):
        node_coords(i, c, a["block_coords"], a["words"], a["prefix"], a["offset"])
        poses[k] = c
    print(f"stride {args.stride}: {len(ids):,} lattice nodes, {len(pairs):,} continuity edges "
          f"({time.time() - t0:.0f} s)", flush=True)
    t0 = time.time()
    feat, thread = features(poses, args.workers)
    print(f"features ({time.time() - t0:.0f} s)", flush=True)
    np.savez(out / f"grid{args.stride}_graph.npz", poses=poses, feat=feat, thread=thread, pairs=pairs,
             clear=ec * UNIT_MM)


@njit(cache=True)
def _find(parent, i):
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


@njit(cache=True)
def basins(f, indptr, indices, tau, min_size):
    n = len(f)
    parent = -np.ones(n, np.int64)
    peak = np.zeros(n)
    size = np.zeros(n, np.int64)
    for i in np.argsort(-f):
        best, best_f = -1, -np.inf
        for e in range(indptr[i], indptr[i + 1]):
            j = indices[e]
            if parent[j] >= 0 and f[j] > best_f:
                best, best_f = j, f[j]
        if best < 0:
            parent[i], peak[i], size[i] = i, f[i], 1
            continue
        r = _find(parent, best)
        parent[i] = r
        size[r] += 1
        for e in range(indptr[i], indptr[i + 1]):
            j = indices[e]
            if parent[j] < 0:
                continue
            a, b = _find(parent, j), _find(parent, i)
            if a == b:
                continue
            if size[a] < min_size or size[b] < min_size:
                lo, hi = (a, b) if size[a] < size[b] else (b, a)
            elif min(peak[a], peak[b]) - f[i] < tau:
                lo, hi = (a, b) if peak[a] < peak[b] else (b, a)
            else:
                continue
            parent[lo] = hi
            size[hi] += size[lo]
            peak[hi] = max(peak[hi], peak[lo])
    out = np.empty(n, np.int64)
    for i in range(n):
        out[i] = _find(parent, i)
    return out


def run_basins(args):
    out = args.out_dir
    sets = {"samples": (out / "samples_graph.npz", lambda g: g["pairs"][(g["clear"] >= 0) & g["in_10.0"]]),
            "grid": (out / f"grid{args.stride}_graph.npz", lambda g: g["pairs"])}
    path = out / "experiments.json"
    old = json.loads(path.read_text()) if path.exists() else {"rows": [], "spectra": {}}
    rows = [r for r in old["rows"] if "clearance basins" not in r["variant"]]
    doors = {}
    for node_set, (graph, edges) in sets.items():
        g = dict(np.load(graph))
        n, sep = len(g["poses"]), g["feat"][:, 0]
        pairs = edges(g)
        W = adjacency(n, pairs, np.ones(len(pairs)))
        _, comp = connected_components(W, directed=False)
        keep = comp == np.argmax(np.bincount(comp))
        Wk = W[keep][:, keep].tocsr()
        ref_thread, ref_full = physical_labels(g["poses"][keep], g["feat"][keep], g["thread"][keep])
        near = sep[keep] < args.near_contact
        idx = np.flatnonzero(keep)
        goal = np.argmin(np.linalg.norm((g["poses"][keep] - PRE_ASSEMBLY) * np.array([1, 1, 1] + [np.pi / 6] * 3), axis=1))
        for tau in args.tau:
            t0 = time.time()
            roots = basins(sep[keep], Wk.indptr, Wk.indices, tau, int(args.min_basin * len(idx)))
            _, lab = np.unique(roots, return_inverse=True)
            name = f"{node_set}: clearance basins, tau {tau:g} mm"
            Wc = Wk.tocoo()
            cross = lab[Wc.row] < lab[Wc.col]
            door = np.minimum(sep[keep][Wc.row[cross]], sep[keep][Wc.col[cross]])
            key = lab[Wc.row[cross]] * (lab.max() + 1) + lab[Wc.col[cross]]
            widest = {}
            for kk, dd in zip(key, door):
                widest[kk] = max(widest.get(kk, -np.inf), dd)
            doors[name] = [{"modes": [int(kk // (lab.max() + 1)), int(kk % (lab.max() + 1))], "door_mm": float(dd)}
                           for kk, dd in widest.items()]
            r = {"variant": name, "nodes": int(len(idx)), "edges": int(len(pairs)), "k": int(lab.max() + 1),
                 "k_is_eigengap": True, "eigengap": float("nan"), "smoothed": False, "goal_mode": int(lab[goal]),
                 **score(lab, ref_thread, ref_full, Wk, near)}
            rows.append(r)
            labels = np.full(n, -1)
            labels[keep] = lab
            np.save(out / f"labels_{node_set}_basins_tau{tau:g}.npy", labels)
            sizes = np.sort(np.bincount(lab))[::-1]
            print(f"{name}: {r['k']} basins (largest {', '.join(f'{x:,}' for x in sizes[:6])}) ({time.time() - t0:.0f} s)\n"
                  f"   purity full {r['purity_full']:.2f}  NMI full {r['nmi_full']:.2f}  ARI {r['ari_full']:.2f}  "
                  f"near-contact purity {r['purity_near']:.2f} NMI {r['nmi_near']:.2f}  disagreement {r['disagreement']:.3f}",
                  flush=True)
    path.write_text(json.dumps({"rows": rows, "spectra": old["spectra"]}, indent=2))
    (out / "basin_doors.json").write_text(json.dumps(doors, indent=2))


def core(W, c):
    """Nodes left after repeatedly removing those with fewer than c neighbours."""
    A = (W > 0).astype(np.int32).tocsr()
    alive = np.ones(W.shape[0], bool)
    deg = np.asarray(A.sum(axis=1)).ravel()
    while True:
        drop = alive & (deg < c)
        if not drop.any():
            return alive
        alive &= ~drop
        deg -= A @ drop.astype(np.int32)


def adjacency(n, pairs, w):
    return sp.coo_matrix((np.r_[w, w], (np.r_[pairs[:, 0], pairs[:, 1]], np.r_[pairs[:, 1], pairs[:, 0]])),
                         shape=(n, n)).tocsr()


def smooth(labels, W, rounds=2):
    """Each node takes the most common label among its neighbours (by edge weight), `rounds` times."""
    lab = labels.copy()
    k = lab.max() + 1
    for _ in range(rounds):
        votes = W @ sp.csr_matrix((np.ones(len(lab)), (np.arange(len(lab)), lab)), shape=(len(lab), k))
        lab = np.asarray(votes.argmax(axis=1)).ravel()
    return lab


def score(labels, ref_thread, ref_full, W, near):
    """near: nodes close to contact, where the closest parts are well defined; scored on their own too."""
    def purity(lab, ref):
        tab = sp.coo_matrix((np.ones(len(lab)), (lab, np.unique(ref, return_inverse=True)[1]))).toarray()
        return float(tab.max(axis=1).sum() / len(lab))
    Wc = W.tocoo()
    cross = labels[Wc.row] != labels[Wc.col]
    return {"purity_thread": purity(labels, ref_thread), "purity_full": purity(labels, ref_full),
            "nmi_thread": float(normalized_mutual_info_score(ref_thread, labels)),
            "nmi_full": float(normalized_mutual_info_score(ref_full, labels)),
            "ari_full": float(adjusted_rand_score(ref_full, labels)),
            "purity_near": purity(labels[near], ref_full[near]) if near.any() else float("nan"),
            "nmi_near": float(normalized_mutual_info_score(ref_full[near], labels[near])) if near.any() else float("nan"),
            "disagreement": float(Wc.data[cross].sum() / Wc.data.sum())}


def variants(out):
    """Name -> (node set, function of the loaded graph giving (pairs, weights))."""
    def los(mask_key=None, weight=None, min_clear=0.0):
        def f(g):
            keep = g["clear"] >= min_clear
            if mask_key:
                keep &= g[mask_key]
            w = np.ones(keep.sum()) if weight is None else weight(g, keep)
            return g["pairs"][keep], w
        return f
    clear_w = lambda g, k: np.clip(g["clear"][k] / 1.0, 0.05, 1.0)          # through tight spots: weak

    def contact_w(g, k, sigma=3.0):
        a, b = g["pairs"][k, 0], g["pairs"][k, 1]
        d2 = np.nansum((g["feat"][a, 1:7] - g["feat"][b, 1:7]) ** 2, axis=1)
        same_thread = g["thread"][a] == g["thread"][b]
        return np.exp(-d2 / (2 * sigma ** 2)) * np.where(same_thread, 1.0, 0.1)
    def distance_w(lever, local):
        """Gaussian of the pose distance (lever mm per radian, or None: RMS hook-keypoint displacement); sigma
        global (median edge) or per node (mean incident edge, self-tuning: w = exp(-d^2 / (sigma_a sigma_b)))."""
        def w(g, k):
            a, b = g["pairs"][k, 0], g["pairs"][k, 1]
            if lever is None:
                emb = keypoint_embedding(g["poses"], g["keypoints"])
                d = np.linalg.norm(emb[a] - emb[b], axis=1)
            else:
                scale = np.array([1, 1, 1] + [lever * np.pi / 180] * 3)
                d = np.linalg.norm((g["poses"][a] - g["poses"][b]) * scale, axis=1)
            if not local:
                return np.exp(-d ** 2 / (2 * np.median(d) ** 2))
            n = len(g["poses"])
            deg = np.bincount(a, minlength=n) + np.bincount(b, minlength=n)
            sigma = (np.bincount(a, d, n) + np.bincount(b, d, n)) / np.maximum(deg, 1)
            return np.exp(-d ** 2 / (sigma[a] * sigma[b]))
        return w

    def wrench_w(kind, lever, proximity=None):
        """Gaussian of the angle between the ends' wrench directions (lever None: force directions), sigma the
        median edge's. proximity [mm]: compare the wrench scaled by exp(-separation / proximity) instead, so
        it fades away from contact (sigma: the median over edges with both ends within 1 mm)."""
        def w(g, k):
            a, b = g["pairs"][k, 0], g["pairs"][k, 1]
            X = wrench_direction(g, kind, lever)
            if proximity is None:
                ang = np.arccos(np.clip(np.sum(X[a] * X[b], axis=1), -1, 1))
                return np.exp(-ang ** 2 / (2 * np.median(ang) ** 2))
            sep = g["feat"][:, 0]
            Y = X * np.exp(-np.clip(sep, 0, None) / proximity)[:, None]
            dist = np.linalg.norm(Y[a] - Y[b], axis=1)
            sigma = np.median(dist[(sep[a] < 1) & (sep[b] < 1)])
            return np.exp(-dist ** 2 / (2 * sigma ** 2))
        return w

    def both(*ws):
        return lambda g, k: np.prod([w(g, k) for w in ws], axis=0)

    def mutual(lever, k=12):
        """Nearest-neighbour edges (as in prepare-samples) kept only when each end is among the other's k nearest."""
        def f(g):
            n = len(g["poses"])
            x = g["poses"] * np.array([1, 1, 1] + [lever * np.pi / 180] * 3)
            _, nbr = cKDTree(x).query(x, k=k + 1)
            a, b = np.repeat(np.arange(n), k).astype(np.int64), nbr[:, 1:].ravel().astype(np.int64)
            both = np.isin(b * n + a, a * n + b)
            pr = np.sort(np.column_stack([a[both], b[both]]), 1)
            keep = (g["clear"] >= 0) & np.isin(g["pairs"][:, 0].astype(np.int64) * n + g["pairs"][:, 1],
                                               pr[:, 0] * n + pr[:, 1])
            return g["pairs"][keep], np.ones(keep.sum())
        return f

    def band(mask_key, mm, weight=None):
        """Line-of-sight edges between nodes within mm of contact only: the contact band on its own."""
        def f(g):
            sep = g["feat"][:, 0]
            keep = (g["clear"] >= 0) & g[mask_key] & (sep[g["pairs"][:, 0]] < mm) & (sep[g["pairs"][:, 1]] < mm)
            return g["pairs"][keep], np.ones(keep.sum()) if weight is None else weight(g, keep)
        return f

    def slide(mask_key, tol, jump=None, weight=None):
        """Contact poses' neighbour edges whose segment stays within tol [mm] of contact and, with jump, whose
        projection onto contact moves on by no more than jump [mm] per step."""
        def f(g):
            keep = g[mask_key] & (g["stray"] <= tol)
            if jump is not None:
                keep &= g["jump"] <= jump
            return g["pairs"][keep], np.ones(keep.sum()) if weight is None else weight(g, keep)
        return f

    def grid(weight=None, min_clear=0.0):
        def f(g):
            keep = g["clear"] >= min_clear
            return g["pairs"][keep], np.ones(keep.sum()) if weight is None else weight(g, keep)
        return f

    v = {
        "samples: knn30 + long-range (previous)": ("samples", lambda g: (g["pairs"][(g["clear"] >= 0) & (g["in_30.0"] | g["in_far"])],
                                                                          np.ones(((g["clear"] >= 0) & (g["in_30.0"] | g["in_far"])).sum()))),
        "samples: knn10": ("samples", los("in_10.0")),
        "samples: knn30": ("samples", los("in_30.0")),
        "samples: knn60": ("samples", los("in_60.0")),
        "samples: knn30 clearance-weighted": ("samples", los("in_30.0", clear_w)),
        "samples: knn30 contact-weighted": ("samples", los("in_30.0", contact_w)),
        "samples: knn10 contact-weighted": ("samples", los("in_10.0", contact_w)),
        "samples: knn30 distance-weighted": ("samples", los("in_30.0", distance_w(30.0, False))),
        "samples: knn30 distance-weighted, local scale": ("samples", los("in_30.0", distance_w(30.0, True))),
        "samples: knn10 distance-weighted, local scale": ("samples", los("in_10.0", distance_w(10.0, True))),
        "samples: knn60 distance-weighted, local scale": ("samples", los("in_60.0", distance_w(60.0, True))),
        "samples: knn10, clearance >= 0.25 mm": ("samples", los("in_10.0", min_clear=0.25)),
        "samples: knn10, clearance >= 0.5 mm": ("samples", los("in_10.0", min_clear=0.5)),
        "samples: knn10, clearance >= 1 mm": ("samples", los("in_10.0", min_clear=1.0)),
        "samples: knn10 mutual": ("samples", mutual(10.0)),
        "samples: knn30 mutual": ("samples", mutual(30.0)),
        "samples: keypoint knn": ("samples", los("in_keypoints")),
        "samples: keypoint knn distance-weighted, local scale": ("samples", los("in_keypoints", distance_w(None, True))),
        "samples: keypoint knn contact-weighted": ("samples", los("in_keypoints", contact_w)),
        "samples: keypoint knn, clearance >= 0.5 mm": ("samples", los("in_keypoints", min_clear=0.5)),
        "samples: knn10 contact-weighted, clearance >= 0.5 mm": ("samples", los("in_10.0", contact_w, 0.5)),
        "samples: knn2.5": ("samples", los("in_2.5")),
        "samples: knn5": ("samples", los("in_5.0")),
        "samples: knn5 contact-weighted": ("samples", los("in_5.0", contact_w)),
        "samples: knn10 force-direction-weighted": ("samples", los("in_10.0", wrench_w("closest", None))),
        "samples: knn10 multi-contact force-direction-weighted": ("samples", los("in_10.0", wrench_w("multi", None))),
        "samples: knn10 wrench-direction-weighted, L 10 mm": ("samples", los("in_10.0", wrench_w("closest", 10.0))),
        "samples: knn10 wrench-direction-weighted, L 30 mm": ("samples", los("in_10.0", wrench_w("closest", 30.0))),
        "samples: knn10 multi-contact wrench-direction-weighted, L 10 mm": ("samples", los("in_10.0", wrench_w("multi", 10.0))),
        "samples: knn10 multi-contact wrench-weighted, L 10 mm, fading over 1 mm": ("samples", los("in_10.0", wrench_w("multi", 10.0, 1.0))),
        "samples: knn10 multi-contact wrench-weighted, L 10 mm, fading over 3 mm": ("samples", los("in_10.0", wrench_w("multi", 10.0, 3.0))),
        "samples: knn10 contact- and multi-contact wrench-direction-weighted, L 10 mm": ("samples", los("in_10.0", both(contact_w, wrench_w("multi", 10.0)))),
        "samples: knn30 multi-contact wrench-direction-weighted, L 30 mm": ("samples", los("in_30.0", wrench_w("multi", 30.0))),
        "samples: knn30 contact- and multi-contact wrench-direction-weighted, L 30 mm": ("samples", los("in_30.0", both(contact_w, wrench_w("multi", 30.0)))),
        "samples: knn10 wrench-direction-weighted, L 10 mm, nodes within 1 mm of contact": ("samples", band("in_10.0", 1.0, wrench_w("closest", 10.0))),
        "samples: knn10 wrench-direction-weighted, L 10 mm, nodes within 2 mm of contact": ("samples", band("in_10.0", 2.0, wrench_w("closest", 10.0))),
        "samples: knn10 wrench-direction-weighted, L 10 mm, nodes within 3 mm of contact": ("samples", band("in_10.0", 3.0, wrench_w("closest", 10.0))),
        "samples: knn10 force-direction-weighted, nodes within 2 mm of contact": ("samples", band("in_10.0", 2.0, wrench_w("closest", None))),
        "samples: knn30 wrench-direction-weighted, L 30 mm, nodes within 2 mm of contact": ("samples", band("in_30.0", 2.0, wrench_w("closest", 30.0))),
        "samples: pose+wrench knn": ("samples", los("in_posewrench")),
        "samples: pose+wrench knn, multi-contact wrench-direction-weighted, L 30 mm": ("samples", los("in_posewrench", wrench_w("multi", 30.0))),
        "samples: knn10, nodes within 2 mm of contact": ("samples", band("in_10.0", 2.0)),
        "samples: keypoint knn, nodes within 2 mm of contact": ("samples", band("in_keypoints", 2.0)),
        "samples: knn10, regularized": ("samples", los("in_10.0"), {"reg": 1.0}),
        "samples: knn10 contact-weighted, regularized": ("samples", los("in_10.0", contact_w), {"reg": 1.0}),
        "samples: keypoint knn, regularized": ("samples", los("in_keypoints"), {"reg": 1.0}),
        "samples: keypoint knn contact-weighted, regularized": ("samples", los("in_keypoints", contact_w), {"reg": 1.0}),
        "contact: knn10 sliding within 0.1 mm": ("contact", slide("in_10.0", 0.1)),
        "contact: knn10 sliding within 0.25 mm": ("contact", slide("in_10.0", 0.25)),
        "contact: knn10 sliding within 0.5 mm": ("contact", slide("in_10.0", 0.5)),
        "contact: knn10 sliding within 1 mm": ("contact", slide("in_10.0", 1.0)),
        "contact: knn10 sliding within 0.5 mm, no jumps": ("contact", slide("in_10.0", 0.5, 0.25)),
        "contact: knn10 sliding within 0.1 mm, wrench-direction-weighted, L 10 mm": ("contact", slide("in_10.0", 0.1, None, wrench_w("closest", 10.0))),
        "contact: knn10 sliding within 0.25 mm, wrench-direction-weighted, L 10 mm": ("contact", slide("in_10.0", 0.25, None, wrench_w("closest", 10.0))),
        "contact: knn10 sliding within 0.5 mm, wrench-direction-weighted, L 10 mm": ("contact", slide("in_10.0", 0.5, None, wrench_w("closest", 10.0))),
        "contact: knn10 sliding within 0.5 mm, no jumps, wrench-direction-weighted, L 10 mm": ("contact", slide("in_10.0", 0.5, 0.25, wrench_w("closest", 10.0))),
        "contact: knn10 sliding within 0.5 mm, force-direction-weighted": ("contact", slide("in_10.0", 0.5, None, wrench_w("closest", None))),
        "contact: knn10 sliding within 0.5 mm, contact-weighted": ("contact", slide("in_10.0", 0.5, None, contact_w)),
        "contact: knn30 sliding within 0.5 mm": ("contact", slide("in_30.0", 0.5)),
        "contact: knn30 sliding within 0.5 mm, wrench-direction-weighted, L 30 mm": ("contact", slide("in_30.0", 0.5, None, wrench_w("closest", 30.0))),
        "grid: continuity distance-weighted, lever 10": ("grid", grid(distance_w(10.0, False))),
        "grid: continuity, regularized": ("grid", grid(), {"reg": 1.0}),
        "grid: continuity, 6-core, regularized": ("grid", grid(), {"reg": 1.0, "core": 6}),
        "grid: continuity distance-weighted, lever 10, 6-core, regularized": ("grid", grid(distance_w(10.0, False)), {"reg": 1.0, "core": 6}),
        "grid: continuity distance-weighted, lever 30, 6-core, regularized": ("grid", grid(distance_w(30.0, False)), {"reg": 1.0, "core": 6}),
        "grid: continuity clearance-weighted, 6-core, regularized": ("grid", grid(lambda g, k: np.clip(g["clear"][k] / 1.0, 0.05, 1.0)), {"reg": 1.0, "core": 6}),
        "grid: continuity contact-weighted, 6-core, regularized": ("grid", grid(contact_w), {"reg": 1.0, "core": 6}),
        "grid: continuity, clearance >= 0.5 mm, 6-core, regularized": ("grid", grid(min_clear=0.5), {"reg": 1.0, "core": 6}),
    }
    return v


def run(args):
    out = args.out_dir
    graphs = {"samples": out / "samples_graph.npz", "grid": out / f"grid{args.stride}_graph.npz",
              "contact": out / "contact_graph.npz"}
    loaded = {k: dict(np.load(p)) for k, p in graphs.items() if p.exists()}
    chosen = {name: v for name, v in variants(out).items() if re.search(args.only, name)}
    rows, spectra = [], {}
    if args.only and (out / "experiments.json").exists():
        old = json.loads((out / "experiments.json").read_text())
        rows = [r for r in old["rows"] if r["variant"] not in chosen]
        spectra = {k: v for k, v in old["spectra"].items() if k not in chosen}
    for name, (node_set, build, *opts) in chosen.items():
        opts = opts[0] if opts else {}
        needs = re.search(r"knn(\d+(\.\d+)?)", name)
        if node_set not in loaded or ("keypoint" in name and "keypoints" not in loaded[node_set]) or \
                (re.search("wrench|force", name) and "wrench_multi" not in loaded[node_set]) or \
                (needs and f"in_{float(needs.group(1))}" not in loaded[node_set]):
            continue
        g = loaded[node_set]
        n = len(g["poses"])
        pairs, w = build(g)
        W = adjacency(n, pairs, w)
        alive = core(W, opts["core"]) if "core" in opts else np.ones(n, bool)
        _, comp = connected_components(W[alive][:, alive], directed=False)
        keep = np.zeros(n, bool)
        keep[np.flatnonzero(alive)[comp == np.argmax(np.bincount(comp))]] = True
        Wk = W[keep][:, keep]
        t0 = time.time()
        tau = opts.get("reg", 0.0) * Wk.sum() / Wk.shape[0]  # a multiple of the mean degree
        vals, vecs = spectral_top(Wk, args.max_k, tol=args.tol, tau=tau)
        k_gap = int(np.argmax(np.diff(vals)[1:args.max_k]) + 2)
        spectra[name] = vals.tolist()
        ref_thread, ref_full = physical_labels(g["poses"][keep], g["feat"][keep], g["thread"][keep])
        near = g["feat"][keep, 0] < args.near_contact
        slug = re.sub(r"[^\w.-]+", "_", name.replace("+", "plus").replace(">=", "ge")).strip("_")
        for k in sorted({k_gap, *args.fixed_k}):
            lab = cluster(vecs, k, args.seed)
            for smoothed in (False, True):
                l2 = smooth(lab, Wk) if smoothed else lab
                r = {"variant": name, "nodes": int(keep.sum()), "edges": int(len(pairs)), "k": k,
                     "k_is_eigengap": k == k_gap, "eigengap": float(vals[k] - vals[k - 1]), "smoothed": smoothed,
                     **score(l2, ref_thread, ref_full, Wk, near)}
                rows.append(r)
                labels = np.full(n, -1)
                labels[keep] = l2
                np.save(out / f"labels_{slug}_k{k}{'_smoothed' if smoothed else ''}.npy", labels)
        print(f"{name}: {keep.sum():,} nodes, eigengap k = {k_gap} ({time.time() - t0:.0f} s)", flush=True)
        (out / "experiments.json").write_text(json.dumps({"rows": rows, "spectra": spectra}, indent=2))
        for r in rows[-2 * len({k_gap, *args.fixed_k}):]:
            print(f"   k={r['k']:2d}{'*' if r['k_is_eigengap'] else ' '} {'smoothed' if r['smoothed'] else '        '} "
                  f"purity thread {r['purity_thread']:.2f} full {r['purity_full']:.2f}  NMI thread {r['nmi_thread']:.2f} "
                  f"full {r['nmi_full']:.2f}  ARI {r['ari_full']:.2f}  near-contact purity {r['purity_near']:.2f} "
                  f"NMI {r['nmi_near']:.2f}  disagreement {r['disagreement']:.3f}", flush=True)
    (out / "experiments.json").write_text(json.dumps({"rows": rows, "spectra": spectra}, indent=2))
    fig, ax = plt.subplots(figsize=(9, 5), facecolor="#fcfcfb")
    for i, (name, vals) in enumerate(spectra.items()):
        ax.plot(range(len(vals)), vals, marker="o", markersize=3, linewidth=1, label=name,
                color=plt.cm.tab10(i % 10))
    ax.set_xlabel("index", color="#52514e"), ax.set_ylabel("eigenvalue of the normalized Laplacian", color="#52514e")
    ax.legend(fontsize=7, frameon=False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "spectra.png", facecolor="#fcfcfb", dpi=110)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["prepare-samples", "prepare-grid", "prepare-keypoints", "prepare-levers", "prepare-wrench",
                                          "prepare-contact", "run",
                                          "run-basins"])
    parser.add_argument("field_dir", type=Path, nargs="?")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=ASSET_DIR / "hook_assembly_max_separation_trajectory.csv")
    parser.add_argument("--samples", type=int, default=120000)
    parser.add_argument("--reach", type=float, default=10.0)
    parser.add_argument("--near-mm", type=float, default=5.0)
    parser.add_argument("--far-keep", type=float, default=0.1)
    parser.add_argument("--k", type=int, default=12, help="nearest neighbours per metric")
    parser.add_argument("--partners", type=int, default=10)
    parser.add_argument("--radius", type=float, default=25.0)
    parser.add_argument("--step", type=float, default=0.25)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--max-k", type=int, default=30)
    parser.add_argument("--tol", type=float, default=1e-4, help="eigensolver tolerance")
    parser.add_argument("--fixed-k", type=int, nargs="*", default=[9, 16])
    parser.add_argument("--only", default="", help="regex: run only the matching variants, merging the results")
    parser.add_argument("--keypoints", type=int, default=8)
    parser.add_argument("--levers", type=float, nargs="*", default=[2.5, 5.0], help="for prepare-levers [mm/rad]")
    parser.add_argument("--wrench-band", type=float, default=2.0, help="hull pairs within this of the closest [mm]")
    parser.add_argument("--wrench-decay", type=float, default=0.5, help="their weights' decay length [mm]")
    parser.add_argument("--contact-copies", type=int, default=2, help="jittered copies of each sample to project")
    parser.add_argument("--contact-target", type=float, default=0.02, help="separation of a contact pose [mm]")
    parser.add_argument("--contact-tol", type=float, default=0.02, help="[mm]")
    parser.add_argument("--contact-cell", type=float, default=0.75, help="thinning cell, lever-10 metric [mm]")
    parser.add_argument("--contact-levers", type=float, nargs="*", default=[10.0], help="kNN pose metrics [mm/rad]")
    parser.add_argument("--contact-reach", type=float, default=1.0, help="stop checking a segment beyond this [mm]")
    parser.add_argument("--wrench-mm", type=float, default=5.0, help="weight of the wrench direction in the joint kNN [mm]")
    parser.add_argument("--tau", type=float, nargs="*", default=[0.25, 0.5, 1.0, 2.0], help="basin persistence [mm]")
    parser.add_argument("--min-basin", type=float, default=0.005, help="smallest basin, as a share of the nodes")
    parser.add_argument("--near-contact", type=float, default=1.0, help="separation [mm] counted as near contact")
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    {"prepare-samples": prepare_samples, "prepare-grid": prepare_grid, "prepare-keypoints": prepare_keypoints,
     "prepare-levers": prepare_levers, "prepare-wrench": prepare_wrench, "prepare-contact": prepare_contact,
     "run": run, "run-basins": run_basins}[args.stage](args)


if __name__ == "__main__":
    main()
