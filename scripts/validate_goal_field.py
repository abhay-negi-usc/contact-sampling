#!/usr/bin/env python3
"""Validate the goal-directed field (goal_field.py) from random start poses.

Starts are drawn from the tube (a random reference-trajectory pose plus a uniform offset within
+/- 10 mm and +/- 10 deg) and kept if collision-free by at least --min-start-mm. For each start:

    connect   of the 64 grid nodes around the start, take the one maximising
              min(separation along the straight segment to it, its value V)
    follow    the field's moves from that node to the pre-assembly pose; no geometry queries, this
              is the part that runs online
    lift      the fixed 3 mm lift from the pre-assembly pose into resting contact
    shortcut  replace stretches of the grid path by straight segments that stay at least
              (predicted clearance - --shortcut-tol-mm) clear
    check     separation densely along both paths, every --check-step mm / deg, with the
              conservative convex-part measure (hook_eyelet_model.Separation)

The predicted clearance is min(connection, V). A path is valid if its checked separation stays
positive until the final lift, which ends in resting contact by design.

Writes results.csv, summary.json and validation.png to --out-dir, plus trajectory CSVs and videos
for the first --examples valid starts.

    conda run -n mujoco_env python scripts/validate_goal_field.py FIELD_DIR --out-dir OUT_DIR
"""

import argparse
import csv
import json
import multiprocessing as mp
import subprocess
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from build_separation_field import CORNERS, reference_path
from goal_field import PRE_ASSEMBLY, GoalField
from hook_eyelet_model import ASSET_DIR, Separation, build_model

SCRIPTS = Path(__file__).resolve().parent
ASSEMBLED = np.zeros(6)
LEVER_MM_PER_DEG = 30.0 * np.pi / 180  # arc length: 1 deg of rotation counts as 0.52 mm

_field = None  # loaded in the parent, shared with forked workers
_sep = None


class SeparationProbe:
    def __init__(self):
        self.m = build_model()
        self.d = mujoco.MjData(self.m)
        self.sep = Separation(self.m)

    def __call__(self, feat):
        """Separation [mm] at poses (x, y, z mm, rx, ry, rz deg)."""
        feat = np.atleast_2d(feat)
        quat = Rotation.from_rotvec(feat[:, 3:], degrees=True).as_quat(scalar_first=True)
        out = np.empty(len(feat))
        for i, (p, q) in enumerate(zip(feat[:, :3] * 1e-3, quat)):
            self.d.qpos[:3], self.d.qpos[3:7] = p, q
            mujoco.mj_kinematics(self.m, self.d)
            out[i] = self.sep(self.m, self.d)[0] * 1e3
        return out


def densify(a, b, step, include_end=False):
    n = max(1, int(np.ceil(np.abs(b - a).max() / step)))
    s = np.arange(n + 1 if include_end else n) / n
    return a + s[:, None] * (b - a)


def segment_min(a, b, step, floor=-np.inf):
    """Smallest separation along a -> b; stops early once below floor."""
    worst = np.inf
    for p in densify(a, b, step, include_end=True):
        worst = min(worst, _sep(p)[0])
        if worst < floor:
            break
    return worst


def path_min(points, step):
    return min(segment_min(a, b, step) for a, b in zip(points[:-1], points[1:]))


def shortcut(points, required, step):
    """Greedy: from each point jump to the farthest later point reachable by a clear straight segment."""
    out, i = [points[0]], 0
    while i < len(points) - 1:
        j = len(points) - 1
        while j > i + 1 and segment_min(points[i], points[j], step, floor=required) < required:
            j = i + (j - i) // 2
        out.append(points[j])
        i = j
    return np.array(out)


def arc_length(points):
    d = np.diff(points, axis=0)
    return float(np.sum(np.linalg.norm(np.hstack([d[:, :3], d[:, 3:] * LEVER_MM_PER_DEG]), axis=1)))


def solve(job):
    k, start, args = job
    rec = {"start": k, **{f"s{i}": float(v) for i, v in enumerate(start)}}
    t0 = time.perf_counter()
    best, best_score = None, -np.inf
    base = np.floor(start).astype(np.int64)
    for corner in base + CORNERS:
        v = _field.value_mm(corner)
        if v is None or v <= best_score:
            continue
        score = min(v, segment_min(start, corner.astype(float), args.check_step, floor=best_score))
        if score > best_score:
            best, best_score = corner, score
    rec["connect_ms"] = (time.perf_counter() - t0) * 1e3
    if best is None:
        rec.update(status="unconnected")
        return rec, None
    t1 = time.perf_counter()
    nodes = _field.follow(best)
    rec["follow_ms"] = (time.perf_counter() - t1) * 1e3
    rec.update(predicted_mm=best_score, field_value_mm=_field.value_mm(best), grid_steps=len(nodes) - 1)

    approach = np.vstack([start, nodes.astype(float)])  # ends at the pre-assembly pose
    lift = np.array([PRE_ASSEMBLY, ASSEMBLED], float)
    rec["raw_min_mm"] = path_min(approach, args.check_step)
    t2 = time.perf_counter()
    short = shortcut(approach, best_score - args.shortcut_tol_mm, args.check_step)
    rec["shortcut_ms"] = (time.perf_counter() - t2) * 1e3
    rec["short_min_mm"] = path_min(short, args.check_step)
    rec["lift_end_mm"] = float(_sep(ASSEMBLED)[0])
    rec.update(raw_length=arc_length(np.vstack([approach, lift[1:]])),
               short_length=arc_length(np.vstack([short, lift[1:]])), short_segments=len(short) - 1)
    rec["status"] = "valid" if min(rec["raw_min_mm"], rec["short_min_mm"]) > 0 else "invalid"
    return rec, np.vstack([short, lift[1:]])


def _init():
    global _sep
    _sep = SeparationProbe()


def write_trajectory(path, points, n=300):
    """Points (mm, deg) as a trajectory CSV, min-jerk timed along arc length."""
    d = np.diff(points, axis=0)
    seg = np.linalg.norm(np.hstack([d[:, :3], d[:, 3:] * LEVER_MM_PER_DEG]), axis=1)
    s_knots = np.concatenate([[0], np.cumsum(seg)]) / seg.sum()
    t = np.linspace(0, 1, n)
    s = 10 * t**3 - 15 * t**4 + 6 * t**5
    feat = np.array([np.interp(s, s_knots, points[:, j]) for j in range(6)]).T
    xyzw = Rotation.from_rotvec(feat[:, 3:], degrees=True).as_quat()
    np.savetxt(path, np.column_stack([t, feat[:, :3] * 1e-3, xyzw]), delimiter=",",
               header="t,x,y,z,qx,qy,qz,qw", comments="", fmt="%.10g")


def plot(results, out):
    ok = [r for r in results if r["status"] != "unconnected"]
    pred = np.array([r["predicted_mm"] for r in ok])
    raw = np.array([r["raw_min_mm"] for r in ok])
    short = np.array([r["short_min_mm"] for r in ok])
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.6), facecolor="#fcfcfb")
    for ax, y, title in [(axs[0], raw, "grid path (field moves)"), (axs[1], short, "after shortcutting")]:
        ax.set_facecolor("#fcfcfb")
        lim = [min(pred.min(), y.min(), 0) - 0.2, max(pred.max(), y.max()) + 0.2]
        ax.plot(lim, lim, color="#8a8984", linewidth=1, label="achieved = predicted")
        ax.axhline(0, color="#d03b3b", linewidth=1, label="contact")
        ax.scatter(pred, y, s=14, color="#2a78d6", alpha=0.7, linewidths=0, label="start pose")
        ax.set_xlabel("predicted clearance from the field [mm]", color="#52514e")
        ax.set_ylabel("checked minimum separation [mm]", color="#52514e")
        ax.set_title(title, color="#0b0b0b", fontsize=11)
        ax.tick_params(colors="#52514e")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.legend(frameon=False, fontsize=8, labelcolor="#52514e")
    fig.suptitle(f"Goal-directed field from {len(results)} random starts (conservative separation, dense check)",
                 color="#0b0b0b")
    fig.tight_layout()
    fig.savefig(out, facecolor="#fcfcfb", dpi=110)
    plt.close(fig)


def main():
    global _field
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("field_dir", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=ASSET_DIR / "hook_assembly_max_separation_trajectory.csv")
    parser.add_argument("--starts", type=int, default=200)
    parser.add_argument("--min-start-mm", type=float, default=0.5)
    parser.add_argument("--check-step", type=float, default=0.1, help="dense check spacing [mm and deg]")
    parser.add_argument("--shortcut-tol-mm", type=float, default=0.1)
    parser.add_argument("--examples", type=int, default=4, help="valid starts rendered to video")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    _field = GoalField.load(args.field_dir / "goal_field.npz")
    print(f"field loaded in {time.time() - t0:.0f} s", flush=True)
    _init()
    path = reference_path(args.reference, spacing=0.25)
    rng = np.random.default_rng(args.seed)
    starts = [path[0]]  # the original start first, for comparison with the optimised trajectories
    while len(starts) < args.starts:
        cand = path[rng.integers(len(path))] + rng.uniform(-10, 10, 6)
        if _sep(cand)[0] >= args.min_start_mm:
            starts.append(cand)

    jobs = [(k, s, args) for k, s in enumerate(starts)]
    with mp.get_context("fork").Pool(args.workers, initializer=_init) as pool:
        out = pool.map(solve, jobs)
    results = [r for r, _ in out]
    keys = sorted({k for r in results for k in r}, key=lambda k: (k not in ("start", "status"), k))
    with open(args.out_dir / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(results)

    ok = [r for r in results if r["status"] != "unconnected"]
    valid = [r for r in ok if r["status"] == "valid"]
    stat = lambda key, rs=ok: dict(zip(["min", "p10", "median", "p90", "max"],
                                        np.percentile([r[key] for r in rs], [0, 10, 50, 90, 100]).round(3).tolist()))
    summary = {
        "starts": len(results), "connected": len(ok), "valid": len(valid),
        "original_start": {k: results[0].get(k) for k in ("status", "predicted_mm", "raw_min_mm", "short_min_mm")},
        "predicted_mm": stat("predicted_mm"),
        "raw_min_minus_predicted_mm": dict(zip(["min", "median", "max"], np.percentile(
            [r["raw_min_mm"] - r["predicted_mm"] for r in ok], [0, 50, 100]).round(3).tolist())),
        "short_min_minus_predicted_mm": dict(zip(["min", "median", "max"], np.percentile(
            [r["short_min_mm"] - r["predicted_mm"] for r in ok], [0, 50, 100]).round(3).tolist())),
        "follow_ms": stat("follow_ms"), "connect_ms": stat("connect_ms"), "shortcut_ms": stat("shortcut_ms"),
        "grid_steps": stat("grid_steps"), "raw_length": stat("raw_length"), "short_length": stat("short_length"),
        "short_segments": stat("short_segments"),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    plot(results, args.out_dir / "validation.png")
    print(json.dumps(summary, indent=2), flush=True)

    for r, pts in [(r, p) for r, p in out if r["status"] == "valid"][: args.examples]:
        traj = args.out_dir / f"start_{r['start']:03d}.csv"
        write_trajectory(traj, pts)
        subprocess.run([sys.executable, str(SCRIPTS / "assembly_trajectory_visual_verification.py"), "--csv", str(traj),
                        "--out", str(traj.with_suffix(".mp4"))], check=True, capture_output=True)
        print(f"wrote {traj.with_suffix('.mp4')}", flush=True)


if __name__ == "__main__":
    main()
