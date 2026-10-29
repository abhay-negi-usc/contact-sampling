#!/usr/bin/env python3
"""Max-separation assembly trajectories from a grid of start poses around the assembled pose.

Each start is the assembled pose translated by a combination of --offsets-mm in x, y and z, with
the hook in its assembled orientation. Starts where the hook already touches or intersects the
eyelet (hull separation <= 0, which is conservative) are skipped, as is the assembled pose itself.

The initial guess lifts the hook straight up by --lift-mm, carries it to the pose the reference
trajectory (max separation from the original start) reaches at --join-t, before it threads the
eyelet, and then follows the reference in. optimize_max_separation_trajectory.py optimises each
guess and assembly_trajectory_visual_verification.py renders it. Lifting straight up assumes the
start is above or beside the eyelet, which holds for every collision-free start of the default grid.

Per case, <out-dir>/<case>/ gets init.csv, trajectory.csv, optimize.log and <case>.mp4, and
<out-dir>/summary.csv lists every start.

    conda run -n mujoco_env python scripts/solve_start_grid.py --out-dir /path/to/start_grid
"""

import argparse
import csv
import itertools
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from hook_eyelet_model import ASSET_DIR, Separation, build_model, load_trajectory
from refine_assembly_trajectory import save_trajectory, to_features

SCRIPTS = Path(__file__).resolve().parent
IDENTITY = Rotation.identity()
T_LIFTED, T_JOINED = 0.12, 0.4  # when the initial guess finishes lifting and reaches the reference


def case_name(offset_mm):
    return "start_" + "_".join(f"d{axis}{v:+03.0f}" for axis, v in zip("xyz", offset_mm))


def min_jerk(s):
    return 10 * s**3 - 15 * s**4 + 6 * s**5


def initial_guess(t, start, ref_t, ref_pos, ref_quat, lift, join_t):
    """Lift straight up, carry to the reference pose at join_t, then follow the reference in."""
    ref_rot = Slerp(ref_t, Rotation.from_quat(ref_quat, scalar_first=True))
    ref_at = lambda s: np.array([np.interp(s, ref_t, ref_pos[:, j]) for j in range(3)]).T
    lifted, joined = start + [0, 0, lift], ref_at(join_t)
    carry = Slerp([0, 1], Rotation.concatenate([IDENTITY, ref_rot(join_t)]))

    lifting, carrying, following = t <= T_LIFTED, (t > T_LIFTED) & (t <= T_JOINED), t > T_JOINED
    s_lift = min_jerk(t[lifting] / T_LIFTED)
    s_carry = min_jerk((t[carrying] - T_LIFTED) / (T_JOINED - T_LIFTED))
    s_ref = join_t + (t[following] - T_JOINED) / (1 - T_JOINED) * (1 - join_t)

    pos = np.vstack([start + s_lift[:, None] * (lifted - start),
                     lifted + s_carry[:, None] * (joined - lifted),
                     ref_at(s_ref)])
    rot = Rotation.concatenate([Rotation.identity(lifting.sum()), carry(s_carry), ref_rot(s_ref)])
    return pos, rot.as_quat(scalar_first=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--offsets-mm", type=float, nargs="+", default=[-10.0, 0.0, 10.0])
    parser.add_argument("--reference", type=Path, default=ASSET_DIR / "hook_assembly_max_separation_trajectory.csv")
    parser.add_argument("--lift-mm", type=float, default=20.0)
    parser.add_argument("--join-t", type=float, default=0.2,
                        help="time along the reference at which the initial guess joins it")
    parser.add_argument("--jobs", type=int, default=12, help="cases optimised in parallel")
    args = parser.parse_args()

    ref, ref_pos, ref_quat = load_trajectory(args.reference)
    t = ref["t"]
    m = build_model()
    d = mujoco.MjData(m)
    separation = Separation(m)

    def separation_mm(pos, quat):
        d.qpos[:7] = np.concatenate([pos, quat])
        mujoco.mj_kinematics(m, d)
        return separation(m, d)[0] * 1e3

    cases = []
    for offset in itertools.product(args.offsets_mm, repeat=3):
        start = np.array(offset) * 1e-3
        sep = separation_mm(start, [1, 0, 0, 0])
        status = "assembled pose" if not any(offset) else "solve" if sep > 0 else "clash"
        cases.append({"case": case_name(offset), "offset": offset, "start": start, "start_sep": sep, "status": status})

    def min_separation(path):
        _, p, q = load_trajectory(path)
        seps = np.array([separation_mm(a, b) for a, b in zip(p, q)])
        return seps[(t > 0.15) & (t < 0.85)].min(), seps[-1]

    to_solve = [c for c in cases if c["status"] == "solve"]
    print(f"{len(to_solve)} of {len(cases)} starts to solve; skipped: "
          + ", ".join(f"{c['case']} ({c['status']}, {c['start_sep']:+.2f} mm)" for c in cases if c["status"] != "solve"))
    for case in to_solve:
        case_dir = args.out_dir / case["case"]
        case_dir.mkdir(parents=True, exist_ok=True)
        pos, quat = initial_guess(t, case["start"], t, ref_pos, ref_quat, args.lift_mm * 1e-3, args.join_t)
        save_trajectory(case_dir / "init.csv", t, to_features(pos, quat), (case["start"], np.array([1.0, 0, 0, 0])),
                        (ref_pos[-1], ref_quat[-1]))
        print(f"  {case['case']}: initial guess min separation {min_separation(case_dir / 'init.csv')[0]:+.2f} mm")

    def solve(case):
        case_dir = args.out_dir / case["case"]
        init, traj = case_dir / "init.csv", case_dir / "trajectory.csv"
        with open(case_dir / "optimize.log", "w") as log:
            for cmd in ([SCRIPTS / "optimize_max_separation_trajectory.py", "--csv", init, "--init", init, "--out", traj],
                        [SCRIPTS / "assembly_trajectory_visual_verification.py", "--csv", traj,
                         "--out", case_dir / f"{case['case']}.mp4"]):
                result = subprocess.run([sys.executable, *map(str, cmd)], stdout=log, stderr=subprocess.STDOUT)
                if result.returncode:
                    case["status"] = f"failed ({Path(cmd[0]).name})"
                    return
        case["status"] = "solved"
        print(f"  {case['case']}: done", flush=True)

    with ThreadPoolExecutor(args.jobs) as pool:
        list(pool.map(solve, to_solve))
    for case in to_solve:
        if case["status"] == "solved":
            case["min_sep"], case["final_sep"] = min_separation(args.out_dir / case["case"] / "trajectory.csv")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.out_dir / "summary.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["case", "dx_mm", "dy_mm", "dz_mm", "status", "start_separation_mm",
                         "min_separation_mm (0.15<t<0.85)", "final_separation_mm"])
        for c in cases:
            writer.writerow([c["case"], *c["offset"], c["status"], f"{c['start_sep']:.3f}",
                             f"{c['min_sep']:.3f}" if "min_sep" in c else "", f"{c['final_sep']:.3f}" if "final_sep" in c else ""])
    print(f"Wrote {args.out_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
