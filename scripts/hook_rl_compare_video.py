#!/usr/bin/env python3
"""Side-by-side video of the hook RL benchmark's versions (RL1-RL8) on the same evaluation cases.

Reads what eval_hook_rl.py saved for each version in --bench/<dir>: the trajectories of the first episodes
(trajectories/ep<N>_*.csv), the believed eyelet pose at each step (trajectories/belief_ep<N>.csv), episodes.csv
(outcomes) and summary.json (the protocol: exploration, retracing). For each case in turn, a grid of tiles, one
version each, plays all versions in step: the hook (orange), the true eyelet (blue) and, translucent green, the
eyelet where the policy believes it is, which moves when an estimate corrects the belief; a yellow trace of the
hook's origin (the inside of its crook). Each tile's bar gives the time, the phase (exploring, retracing,
assembling), how far the belief is off, and when the version's episode has ended, its outcome: assembled (the
policy declared done while seated, and the pull test passed), a false declaration, or the time limit.

    conda run -n mujoco_env python scripts/hook_rl_compare_video.py \\
        --bench /media/rp/Elements1/abhay_ws/contact-sampling/hook_rl/bench_nominal/T10_stop --cases 0-9
"""

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import csv
import json
from pathlib import Path

import mediapy
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

import hook_rl_env as E
from assembly_trajectory_visual_verification import add_polyline, fit_camera, geom_world_vertices, put_text

ROWS = [("RL1", "RL1", "no-uncertainty policy, exact belief"),
        ("RL2", "RL2", "no-uncertainty policy"),
        ("RL3", "RL3", "+-10 policy, exact belief"),
        ("RL4", "RL4", "+-10 policy"),
        ("RL5", "RL5_retrace", "no-unc. policy + noise explore + ICP"),
        ("RL6", "RL6_retrace", "+-10 policy + noise explore + ICP"),
        ("RL7", "RL7", "+-10 policy trained with ICP in loop"),
        ("RL8", "RL8_retrace", "no-unc. policy + learned explorer + ICP")]
GHOST_RGBA = np.array([0.2, 0.85, 0.35, 0.3], np.float32)
TRACE_RGBA = (1.0, 0.85, 0.2, 0.9)
OK_RGB, BAD_RGB, WAIT_RGB = (90, 230, 110), (255, 80, 80), (255, 190, 60)
VIEW = (50, -25)  # azimuth, elevation deg
BAR_H, HEADER_H = 40, 66
DT = E.SUBSTEPS * E.TIMESTEP
HOLD_S = 1.5


def read_poses(path):
    """(positions m (T, 3), quaternions wxyz (T, 4)) of a trajectory CSV (t, x, y, z, qx, qy, qz, qw)."""
    a = np.loadtxt(path, delimiter=",", skiprows=1)
    return a[:, 1:4], a[:, [7, 4, 5, 6]]


def phases(online):
    """[(first step, phase name)] of a version's protocol (eval_hook_rl.py summary 'online')."""
    if online.get("explorer"):
        e = online["explore_policy_steps"]
        return [(0, "exploring (learned)")] + ([(e, "retracing"), (2 * e, "assembling")] if online.get("retrace")
                                               else [(e, "assembling")])
    e = online.get("explore_steps", 0)
    if e:
        return [(0, "exploring (noise)")] + ([(e, "retracing"), (2 * e, "assembling")] if online.get("retrace")
                                             else [(e, "assembling")])
    return [(0, "assembling")]


def phase_at(table, step):
    return [name for first, name in table if step >= first][-1]


class Version:
    def __init__(self, bench, name, folder, label, case):
        d = bench / folder
        summary = json.loads((d / "summary.json").read_text())
        with open(d / "episodes.csv") as f:
            self.rec = next(r for r in csv.DictReader(f) if int(r["episode"]) == case)
        traj = next(p for p in (d / "trajectories").glob(f"ep{case}_*.csv"))
        self.pos, self.quat = read_poses(traj)
        self.bel_pos, self.bel_quat = read_poses(d / "trajectories" / f"belief_ep{case}.csv")
        self.name, self.label = name, label
        self.phases = phases(summary["online"])
        self.no_error = summary["no_error"]
        self.steps = len(self.pos) - 1
        outcome = self.rec["outcome"]
        if self.rec["pull_ok"] == "True":
            self.result, self.color = "ASSEMBLED, declared done", OK_RGB
        elif outcome.startswith("declared done"):
            self.result, self.color = "FALSE DONE: " + outcome.split(": ", 1)[1], BAD_RGB
        elif outcome.startswith("never declared"):
            self.result, self.color = "TIME LIMIT: " + outcome.split(": ", 1)[1], WAIT_RGB
        else:
            self.result, self.color = outcome.upper(), BAD_RGB

    def belief_off(self, k):
        """(mm, deg) the belief is off at step k: the believed eyelet's pose in the true eyelet frame."""
        turn = Rotation.from_quat(self.bel_quat[k], scalar_first=True).magnitude()
        return np.linalg.norm(self.bel_pos[k]) * 1e3, np.degrees(turn)


class Renderer:
    def __init__(self, tile_w, tile_h):
        self.m = E.build_rl_model()
        self.d = mujoco.MjData(self.m)
        self.r = mujoco.Renderer(self.m, tile_h, tile_w)
        self.aspect = tile_w / tile_h
        self.hook_qadr = self.m.jnt_qposadr[self.m.joint("hook").id]
        self.eyelet_geom = self.m.geom("eyelet").id

    def pose(self, pos, quat):
        self.d.qpos[self.hook_qadr:self.hook_qadr + 3], self.d.qpos[self.hook_qadr + 3:self.hook_qadr + 7] = pos, quat
        mujoco.mj_kinematics(self.m, self.d)

    def camera(self, versions):
        """One camera for a case: holds every version's start, the assembled hook and the eyelet's loop."""
        pts = []
        for v in versions:
            self.pose(v.pos[0], v.quat[0])
            pts.append(geom_world_vertices(self.m, self.d, "hook"))
        self.pose(np.zeros(3), np.array([1.0, 0, 0, 0]))
        eyelet = geom_world_vertices(self.m, self.d, "eyelet")
        loop_width = np.ptp(eyelet[:, :2], axis=0).max()
        pts += [geom_world_vertices(self.m, self.d, "hook"), eyelet[eyelet[:, 2] > -loop_width]]
        return fit_camera(np.vstack(pts), *VIEW, self.m.vis.global_.fovy, self.aspect, margin=1.1)

    def tile(self, v, k, cam):
        """Version v at step k (held at its last step once ended)."""
        k = min(k, v.steps)
        self.pose(v.pos[k], v.quat[k])
        self.r.update_scene(self.d, camera=cam)
        scn = self.r.scene
        R_b = Rotation.from_quat(v.bel_quat[k], scalar_first=True).as_matrix()
        for g in [scn.geoms[i] for i in range(scn.ngeom)]:  # the believed eyelet: the eyelet moved by the belief
            if g.objtype == mujoco.mjtObj.mjOBJ_GEOM and g.objid == self.eyelet_geom and scn.ngeom < scn.maxgeom:
                ghost = scn.geoms[scn.ngeom]
                mujoco.mjv_initGeom(ghost, g.type, np.asarray(g.size, np.float64), R_b @ g.pos + v.bel_pos[k],
                                    (R_b @ g.mat.reshape(3, 3)).flatten().astype(np.float64), GHOST_RGBA)
                ghost.dataid = g.dataid
                scn.ngeom += 1
        if k > 0:
            add_polyline(scn, v.pos[:k + 1], TRACE_RGBA, 0.001)
        img = self.r.render().copy()
        put_text(img, f"{v.name}  {v.label}", (6, 16), scale=0.42)
        bar = np.zeros((BAR_H, img.shape[1], 3), np.uint8)
        mm, deg = v.belief_off(k)
        put_text(bar, f"t {k * DT:4.2f} s  {phase_at(v.phases, k):20s} belief off {mm:4.1f} mm {deg:4.1f} deg",
                 (6, 15), scale=0.4)
        if k == v.steps:
            put_text(bar, v.result, (6, 33), scale=0.45, color=v.color)
        return np.vstack([img, bar])


def parse_row(text):
    """NAME:FOLDER:LABEL -> (name, folder, label)."""
    name, folder, label = text.split(":", 2)
    return name, folder, label


def parse_cases(text):
    out = []
    for part in text.split(","):
        lo, _, hi = part.partition("-")
        out += list(range(int(lo), int(hi or lo) + 1))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bench", type=Path, required=True, help="directory of the versions' evaluations")
    parser.add_argument("--cases", type=parse_cases, default=parse_cases("0-9"), help="e.g. 0-9 or 0,3,5")
    parser.add_argument("--out", type=Path, help="default: --bench/videos/compare_cases<...>.mp4")
    parser.add_argument("--tile", type=lambda t: [int(v) for v in t.split("x")], default=[400, 300],
                        help="tile width x height (at most 640x480)")
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--rows", type=parse_row, action="append",
                        help="NAME:FOLDER:LABEL, a version to show (repeatable); default the eight of ROWS")
    args = parser.parse_args()
    rows = args.rows or ROWS
    columns = min(args.columns, len(rows))
    out = args.out or args.bench / "videos" / f"compare_cases{args.cases[0]}-{args.cases[-1]}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    ren = Renderer(*args.tile)
    rows_n = -(-len(rows) // columns)
    shape = (HEADER_H + rows_n * (args.tile[1] + BAR_H), columns * args.tile[0])
    total = 0
    with mediapy.VideoWriter(out, shape=shape, fps=round(1 / DT)) as writer:
        for n, case in enumerate(args.cases):
            versions = [Version(args.bench, name, folder, label, case) for name, folder, label in rows]
            cam = ren.camera(versions)
            ref = next((v for v in versions if not v.no_error), versions[0])  # tested with the case's belief error
            e = [float(ref.rec[f"err0_{a}"]) for a in ("x", "y", "z", "rx", "ry", "rz")]
            title = [f"case {case} ({n + 1}/{len(args.cases)})   belief error x {e[0]:+.1f} y {e[1]:+.1f} "
                     f"z {e[2]:+.1f} mm", f"belief rotation error ({e[3]:+.1f}, {e[4]:+.1f}, {e[5]:+.1f}) deg"]
            length = max(v.steps for v in versions) + round(HOLD_S / DT)
            for k in range(length + 1):
                tiles = [ren.tile(v, k, cam) for v in versions]
                tiles += [np.zeros_like(tiles[0])] * (rows_n * columns - len(tiles))
                grid = np.vstack([np.hstack(tiles[i:i + columns]) for i in range(0, len(tiles), columns)])
                header = np.zeros((HEADER_H, grid.shape[1], 3), np.uint8)
                put_text(header, title[0], (8, 18), scale=0.5)
                put_text(header, title[1], (8, 38), scale=0.5)
                put_text(header, "green: believed eyelet, blue: true eyelet, yellow: path of the hook's crook",
                         (8, 58), scale=0.42)
                writer.add_image(np.vstack([header, grid]))
            total += length + 1
            print(f"case {case}: {length + 1} frames; " + ", ".join(f"{v.name} {v.result}" for v in versions),
                  flush=True)
    print(f"wrote {out} ({total} frames, {total * DT:.0f} s)")


if __name__ == "__main__":
    main()
