#!/usr/bin/env python3
"""Precompute the hook/eyelet separation field over a pose tube around the reference trajectory.

Separation is the signed minimum distance between hook and eyelet (hook_eyelet_model.Separation):
positive with space between them, negative when they interfere, measured between the convex
collision parts, so it under-reports the true separation by at most ~0.5 mm. It is evaluated on a
6-D grid of hook poses in the eyelet frame: x, y, z [mm] and the rotation vector rx, ry, rz [deg],
the trajectory CSVs' convention. Nodes sit at integer multiples of the spacing, so the assembled
pose is a node.

Domain: every node within +/- --tube-mm and +/- --tube-deg, in each coordinate, of a pose on the
reference trajectory.

    pass 1  every tube node at --spacing (1 mm / 1 deg)
    pass 2  cells whose separation may lie in [--band-lo, --band-hi], judged from their 64 corner
            nodes widened by --margin, are refined to half spacing: the other 63 nodes
            cell + {0, 1/2}^6 of each

Output in --out-dir (meta.json, summary.json and npz parts; rerunning resumes):
    coarse/part_*.npz   blocks: (n, 6) int32 block coordinates b; values: (n, 8, 8, 8, 8, 8, 8) int16,
                        separation at node (8 b + local index) * spacing, in 0.01 mm, or TUBE_OUT
    fine/part_*.npz     cells: (n, 6) int32 node indices c of refined cells; values: (n, 64) int16,
                        separation at (c + k / 2) * spacing for k in {0, 1}^6 in C order, in 0.01 mm

    conda run -n mujoco_env python scripts/build_separation_field.py --out-dir /path/to/field
"""

import argparse
import itertools
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from hook_eyelet_model import ASSET_DIR, Separation, build_model, load_trajectory

UNIT_MM = 0.01
TUBE_OUT = np.iinfo(np.int16).min
B = 8  # nodes per block side
CORNERS = np.array(list(itertools.product((0, 1), repeat=6)))


def reference_path(csv_path, spacing):
    """Trajectory poses as (x, y, z mm, rx, ry, rz deg), densified to at most `spacing` per step."""
    _, pos, quat = load_trajectory(csv_path)
    feat = np.hstack([pos * 1e3, Rotation.from_quat(quat, scalar_first=True).as_rotvec(degrees=True)])
    dense = [feat[:1]]
    for a, b in zip(feat[:-1], feat[1:]):
        n = max(1, int(np.ceil(np.abs(b - a).max() / spacing)))
        dense.append(a + (b - a) * (np.arange(1, n + 1) / n)[:, None])
    return np.vstack(dense)


def in_tube(feat, path, half):
    """Mask of poses within `half` of some path pose in every coordinate."""
    mask = np.zeros(len(feat), bool)
    near = path[np.all(np.abs(path - feat.mean(0)) <= half + np.ptp(feat, 0) / 2, axis=1)]
    for p in near:
        mask |= np.all(np.abs(feat - p) <= half, axis=1)
    return mask


class Evaluator:
    def __init__(self):
        self.m = build_model()
        self.d = mujoco.MjData(self.m)
        self.sep = Separation(self.m)

    def __call__(self, feat):
        """Separation in 0.01 mm at poses (x, y, z mm, rx, ry, rz deg)."""
        if not len(feat):
            return np.zeros(0, np.int16)
        quat = Rotation.from_rotvec(feat[:, 3:], degrees=True).as_quat(scalar_first=True)
        out = np.empty(len(feat))
        for i, (p, q) in enumerate(zip(feat[:, :3] * 1e-3, quat)):
            self.d.qpos[:3], self.d.qpos[3:7] = p, q
            mujoco.mj_kinematics(self.m, self.d)
            out[i] = self.sep(self.m, self.d)[0]
        return np.clip(np.rint(out * 1e3 / UNIT_MM), -32767, 32767).astype(np.int16)


def save_part(directory, **arrays):
    """Write atomically, so an interrupted run never leaves a half-written part."""
    path = directory / f"part_{os.getpid()}_{time.time_ns()}.npz"
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def load_parts(directory, key):
    for f in sorted(directory.glob("part_*.npz")):
        if not f.name.endswith(".tmp.npz"):
            with np.load(f) as z:
                yield z[key], z["values"]


_eval = None
# Pass 2 workers inherit these from the parent when the pool forks: the coarse field stored
# compactly (block -> packed in-tube mask, in-tube values) and the keys of cells already refined.
_field = {}
_done_keys = np.zeros(0, np.int64)


def _init_worker():
    global _eval
    _eval = Evaluator()


def compact_block(values):
    mask = values.reshape(-1) != TUBE_OUT
    return np.packbits(mask), values.reshape(-1)[mask].copy()


def dense_block(block):
    entry = _field.get(tuple(block))
    if entry is None:
        return None
    bits, vals = entry
    out = np.full(B**6, TUBE_OUT, np.int16)
    out[np.unpackbits(bits, count=B**6).astype(bool)] = vals
    return out.reshape((B,) * 6)


def run_task(job):
    """Run a (task name, args, deadline) job unless past its deadline; report failures as results.

    An exception raised out of a pool task makes Pool.terminate() deadlock while tasks are still
    queued, so failures come back as an error string instead; the work is retried on the next run.
    """
    name, args, deadline = job
    if time.time() > deadline:
        return 0, 0, "skipped"
    try:
        return (*TASKS[name](*args), "")
    except Exception as e:
        return 0, 0, f"{name} {args[0] if name == 'fine' else ''}: {type(e).__name__}: {e}"


def coarse_task(blocks, path, half, spacing, out_dir):
    """Evaluate every tube node of a few blocks and write them as one part."""
    local = np.indices((B,) * 6).reshape(6, -1).T
    values = np.full((len(blocks),) + (B,) * 6, TUBE_OUT, np.int16)
    for v, b in zip(values, blocks):
        feat = (b * B + local) * spacing
        mask = in_tube(feat, path, half)
        v.reshape(-1)[mask] = _eval(feat[mask])
    save_part(out_dir / "coarse", blocks=np.asarray(blocks, np.int32), values=values)
    return len(blocks), int(np.sum(values != TUBE_OUT))


def fine_task(block, band, margin, spacing, cells_per_task, out_dir, deadline):
    """Refine one block's cells that may lie in the band: evaluate their 63 half-spacing nodes."""
    coarse = dense_block(block)
    idx = np.argwhere(refine_flags(block, coarse, band, margin))
    cells = (block * B + idx).astype(np.int16)
    base = coarse[tuple(idx.T)]
    if len(_done_keys):
        keep = ~np.isin(cell_keys(cells), _done_keys)
        cells, base = cells[keep], base[keep]
    written = 0
    for i in range(0, len(cells), cells_per_task):
        if time.time() > deadline:  # the rest of the block is picked up on the next run
            return 0, written
        c = cells[i: i + cells_per_task]
        values = np.empty((len(c), 64), np.int16)
        values[:, 0] = base[i: i + cells_per_task]
        feat = ((c[:, None, :] + CORNERS[None, 1:] / 2) * spacing).reshape(-1, 6)
        values[:, 1:] = _eval(feat).reshape(len(c), 63)
        save_part(out_dir / "fine", cells=c.astype(np.int32), values=values)
        written += len(c)
    return 1, written


TASKS = {"coarse": coarse_task, "fine": fine_task}


def cell_keys(cells):
    """One int64 per node index, 10 bits per coordinate."""
    c = np.asarray(cells, np.int64).reshape(-1, 6) + 512
    assert np.all((c >= 0) & (c < 1024)), "node index outside +/- 512"
    return np.bitwise_or.reduce(c << (10 * np.arange(6)), axis=1)


def tube_blocks(path, half, spacing):
    """Blocks intersecting the tube, nearest the reference path first (so a partial run is the useful part)."""
    lo = np.floor((path - half) / spacing / B).astype(int)
    hi = np.floor((path + half) / spacing / B).astype(int)
    blocks = set()
    for l, h in zip(lo, hi):
        blocks.update(itertools.product(*[range(a, b + 1) for a, b in zip(l, h)]))
    blocks = np.array(sorted(blocks))
    centres = (blocks + 0.5) * B * spacing
    dist = np.min([np.abs((centres - p) / half).max(1) for p in path[:: max(1, len(path) // 400)]], axis=0)
    return blocks[np.argsort(dist, kind="stable")]


def refine_flags(block, coarse, band, margin):
    """Cells (indexed by their base node) of `block` whose separation may lie in band."""
    ext = np.full((B + 1,) * 6, np.nan)
    for off in CORNERS:
        src = coarse if not off.any() else dense_block(block + off)
        if src is None:
            continue
        dst = tuple(slice(B, B + 1) if o else slice(0, B) for o in off)
        sub = tuple(slice(0, 1) if o else slice(0, B) for o in off)
        vals = src[sub].astype(float)
        ext[dst] = np.where(vals == TUBE_OUT, np.nan, vals * UNIT_MM)
    lo, hi = ext, ext
    for axis in range(6):
        a, b = [slice(None)] * 6, [slice(None)] * 6
        a[axis], b[axis] = slice(0, -1), slice(1, None)
        lo, hi = np.fmin(lo[tuple(a)], lo[tuple(b)]), np.fmax(hi[tuple(a)], hi[tuple(b)])
    base_in_tube = coarse != TUBE_OUT
    with np.errstate(invalid="ignore"):
        return base_in_tube & (lo - margin <= band[1]) & (hi + margin >= band[0])


def run(pool, jobs, label, total_units, count_label, log_every=60.0):
    """Run jobs returning (units done, count, error); log progress and errors.

    Returns (seconds, units done, total count, failed jobs, jobs skipped at the deadline).
    """
    t0, done, count, failed, skipped, last = time.time(), 0, 0, 0, 0, 0.0
    for i, (n, c, err) in enumerate(pool.imap_unordered(run_task, jobs), 1):
        done, count = done + n, count + c
        if err == "skipped":
            skipped += 1
        elif err:
            failed += 1
            print(f"[{time.strftime('%H:%M:%S')}] FAILED {err}", flush=True)
        if time.time() - last > log_every or i == len(jobs):
            last, el = time.time(), time.time() - t0
            eta = el / max(done, 1) * (total_units - done)
            print(f"[{time.strftime('%H:%M:%S')}] {label}: {done}/{total_units} ({done / max(total_units, 1):.1%}), "
                  f"{count} {count_label}, {el / 60:.1f} min elapsed, ETA {eta / 60:.1f} min", flush=True)
    return time.time() - t0, done, count, failed, skipped


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=ASSET_DIR / "hook_assembly_max_separation_trajectory.csv")
    parser.add_argument("--spacing", type=float, default=1.0, help="coarse node spacing [mm and deg]")
    parser.add_argument("--tube-mm", type=float, default=10.0)
    parser.add_argument("--tube-deg", type=float, default=10.0)
    parser.add_argument("--band", type=float, nargs=2, default=[0.0, 3.0],
                        help="separation range [mm] refined to half spacing")
    parser.add_argument("--margin", type=float, default=0.6,
                        help="widening [mm] of each cell's corner range when deciding refinement")
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--blocks-per-task", type=int, default=2)
    parser.add_argument("--cells-per-task", type=int, default=20000)
    parser.add_argument("--hours", type=float, default=float("inf"),
                        help="wall-clock budget; work left over is picked up by rerunning")
    parser.add_argument("--test-near-t", type=float, help="only the blocks nearest the reference at this time")
    parser.add_argument("--test-blocks", type=int, default=8)
    args = parser.parse_args()

    out = args.out_dir
    for sub in ("coarse", "fine"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    half = np.array([args.tube_mm] * 3 + [args.tube_deg] * 3)
    path = reference_path(args.reference, spacing=args.spacing / 4)
    blocks = tube_blocks(path, half, args.spacing)
    if args.test_near_t is not None:
        traj, _, _ = load_trajectory(args.reference)
        ref = reference_path(args.reference, spacing=1e9)[np.argmin(np.abs(traj["t"] - args.test_near_t))]
        centres = (blocks + 0.5) * B * args.spacing
        blocks = blocks[np.argsort(np.abs((centres - ref) / half).max(1))[: args.test_blocks]]
    meta = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    meta.update(unit_mm=UNIT_MM, tube_out=int(TUBE_OUT), block=B, n_blocks=len(blocks),
                coordinates="x, y, z [mm], rotation vector rx, ry, rz [deg]: hook pose in the eyelet frame "
                            "(trajectory CSV convention, see hook_eyelet_model.py)",
                separation="hook_eyelet_model.Separation (convex parts, libccd): conservative by <= ~0.5 mm")
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"{len(blocks)} blocks of {B}^6 nodes intersect the tube; {args.workers} workers", flush=True)

    global _field, _done_keys
    deadline = time.time() + args.hours * 3600
    done = {tuple(b) for bs, _ in load_parts(out / "coarse", "blocks") for b in bs}
    todo = [b for b in blocks if tuple(b) not in done]
    summary = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "hours_budget": args.hours, "blocks": len(blocks),
               "pass1_blocks_done_before": len(blocks) - len(todo)}
    with mp.get_context("fork").Pool(args.workers, initializer=_init_worker) as pool:
        jobs = [("coarse", (todo[i: i + args.blocks_per_task], path, half, args.spacing, out), deadline)
                for i in range(0, len(todo), args.blocks_per_task)]
        secs, n_done, _, failed, skipped = run(pool, jobs, "pass 1 blocks", len(todo), "tube nodes")
    summary.update(pass1_s=secs, pass1_blocks_done=n_done, pass1_failed_jobs=failed, pass1_skipped_jobs=skipped)
    pass1_complete = summary["pass1_blocks_done_before"] + n_done == len(blocks)

    if pass1_complete:
        wanted = {tuple(b) for b in blocks}
        _field = {tuple(b): compact_block(v) for bs, vs in load_parts(out / "coarse", "blocks")
                  for b, v in zip(bs, vs) if tuple(b) in wanted}
        _done_keys = np.concatenate([cell_keys(cs) for cs, _ in load_parts(out / "fine", "cells")]
                                    or [np.zeros(0, np.int64)])
        summary.update(coarse_nodes=int(sum(len(vals) for _, vals in _field.values())),
                       fine_cells_done_before=len(_done_keys))
        print(f"pass 1 complete: {summary['coarse_nodes']} tube nodes; {len(_done_keys)} cells already refined",
              flush=True)
        # Forked now, so the workers share the compact coarse field.
        with mp.get_context("fork").Pool(args.workers, initializer=_init_worker) as pool:
            jobs = [("fine", (b, args.band, args.margin, args.spacing, args.cells_per_task, out, deadline), deadline)
                    for b in blocks if tuple(b) in _field]
            secs, n_done, written, failed, skipped = run(pool, jobs, "pass 2 blocks", len(jobs), "cells refined")
        summary.update(pass2_s=secs, pass2_blocks_completed=n_done, pass2_jobs=len(jobs),
                       fine_cells_written=written, fine_nodes_written=63 * written,
                       pass2_failed_jobs=failed, pass2_skipped_jobs=skipped)
    else:
        print("pass 1 incomplete (budget or failures): rerun to finish it before refining", flush=True)
    summary["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
