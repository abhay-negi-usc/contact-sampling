#!/usr/bin/env python3
"""Refine the hook assembly trajectory into a smooth, collision-free one.

The hook pose (position and rotation vector in the eyelet frame) is a quintic B-spline in
time. Its first and last three control points are pinned to the original start pose and the
assembled pose, so the motion starts and ends at rest (zero velocity and acceleration). The
out-of-plane components (y, rx, rz) blend smoothly to zero, and the interior control points
of the planar ones (x, z, rotation about y) minimise

    jerk  +  w_track * deviation from the original  +  w_clear * clearance violations

where clearance is the signed distance between the hook's and eyelet's convex collision parts.
The parts enclose the true meshes, so hull clearance >= 0 means the real parts do not touch.
Over the final `--settle` fraction of the motion the required clearance ramps down to allow the
resting contact of the assembled pose.

The output CSV uses the input's conventions (hook pose in the eyelet frame, metres, quaternion
as qx, qy, qz, qw) on the same time grid.

    conda run -n mujoco_env python scripts/refine_assembly_trajectory.py
"""

import argparse
import time
from pathlib import Path

import mujoco
import numpy as np
from scipy.interpolate import BSpline
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from hook_eyelet_model import ASSET_DIR, COLLISION_GROUP, build_model, hook_eyelet_contacts, load_trajectory

DEGREE = 5
LEVER_MM = 30.0  # rotation vector [rad] -> mm, so rotations and translations weigh alike
# Hooking is a planar motion: optimise x, z and the rotation about y. The out-of-plane
# components (y, rx, rz) blend smoothly from their start values to zero.
PLANAR = [0, 2, 4]
OFF_PLANE = [1, 3, 5]


def to_features(pos, quat):
    """Poses -> (N, 6) [position mm, rotation vector scaled to mm]."""
    rotvec = Rotation.from_quat(quat, scalar_first=True).as_rotvec()
    return np.hstack([pos * 1e3, rotvec * LEVER_MM])


def from_features(feat):
    pos = feat[:, :3] * 1e-3
    quat = Rotation.from_rotvec(feat[:, 3:] / LEVER_MM).as_quat(scalar_first=True)
    return pos, quat


def basis(t, n_ctrl, deriv=0):
    """(len(t), n_ctrl) clamped uniform B-spline basis (or its derivative) on [0, 1]."""
    interior = np.linspace(0, 1, n_ctrl - DEGREE + 1)[1:-1]
    knots = np.r_[np.zeros(DEGREE + 1), interior, np.ones(DEGREE + 1)]
    spline = BSpline(knots, np.eye(n_ctrl), DEGREE)
    return (spline.derivative(deriv) if deriv else spline)(t)


class ClearanceCost:
    """Squared clearance violations between the hook and eyelet collision parts, per sample."""

    def __init__(self, model, required_mm):
        self.m, self.d = model, mujoco.MjData(model)
        self.required = np.asarray(required_mm)
        # Report contacts a little beyond the required clearance so violations are seen early.
        self.m.geom_margin[self.m.geom_group == COLLISION_GROUP] = (self.required.max() + 1.0) * 1e-3

    def min_dist_mm(self, pos, quat):
        out = np.full(len(pos), np.inf)
        for i, (p, q) in enumerate(zip(pos, quat)):
            out[i] = min((c[2] for c in self._contacts(p, q)), default=np.inf) * 1e3
        return out

    def per_sample(self, feat, rows):
        pos, quat = from_features(feat[rows])
        cost = np.zeros(len(rows))
        for j, (p, q, req) in enumerate(zip(pos, quat, self.required[rows])):
            dists = np.array([c[2] for c in self._contacts(p, q)]) * 1e3
            cost[j] = np.sum(np.maximum(0.0, req - dists) ** 2)
        return cost

    def _contacts(self, p, q):
        self.d.qpos[:7] = np.concatenate([p, q])
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_collision(self.m, self.d)
        return hook_eyelet_contacts(self.m, self.d)


def minimize_with_restarts(cost, x0, max_iter, restarts):
    """L-BFGS on a cost with an analytic+finite-difference gradient.

    Contact kinks make the line search give up early, so restart from where it stopped until
    the cost stops improving.
    """
    t0 = time.time()
    x, best = x0, np.inf
    for restart in range(restarts):
        res = minimize(cost, x, jac=True, method="L-BFGS-B", options={"maxiter": max_iter})
        print(f"  round {restart}: cost {res.fun:.4f} after {res.nit} iterations")
        x = res.x
        if res.fun > best - 1e-3 * abs(best):
            break
        best = res.fun
    print(f"optimised in {time.time() - t0:.0f} s")
    return x


def save_trajectory(path, t, feat, pose_start, pose_end):
    """Write poses in the input CSV's convention, with the end poses copied exactly."""
    pos, quat = from_features(feat)
    (pos[0], quat[0]), (pos[-1], quat[-1]) = pose_start, pose_end
    xyzw = np.roll(quat, -1, axis=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, np.column_stack([t, pos, xyzw]), delimiter=",", header="t,x,y,z,qx,qy,qz,qw",
               comments="", fmt="%.10g")
    print(f"Wrote {len(t)} waypoints to {path}")


def smoothness_report(t, feat):
    dt = np.diff(t).mean()
    vel = np.diff(feat, axis=0) / dt
    acc = np.diff(feat, n=2, axis=0) / dt**2
    jerk = np.diff(feat, n=3, axis=0) / dt**3
    reversals = int(np.sum(np.diff(np.sign(vel[:, 0][np.abs(vel[:, 0]) > 1e-6])) != 0))
    return (f"peak accel {np.linalg.norm(acc, axis=1).max():8.0f} mm/T^2, "
            f"RMS jerk {np.sqrt((jerk ** 2).sum(1).mean()):9.0f} mm/T^3, x-velocity reversals {reversals}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", type=Path, default=ASSET_DIR / "hook_assembly_ideal_trajectory.csv")
    parser.add_argument("--out", type=Path, default=ASSET_DIR / "hook_assembly_refined_trajectory.csv")
    parser.add_argument("--n-ctrl", type=int, default=20, help="B-spline control points")
    parser.add_argument("--clearance-mm", type=float, default=0.5,
                        help="hull clearance kept while the parts are not meant to touch")
    parser.add_argument("--settle", type=float, default=0.06,
                        help="final fraction of the motion over which the clearance ramps to resting contact")
    parser.add_argument("--contact-mm", type=float, default=-0.15,
                        help="hull clearance allowed at the assembled pose (the hulls bulge ~0.13 mm there)")
    parser.add_argument("--w-track", type=float, default=0.02, help="weight on deviation from the original [1/mm^2]")
    parser.add_argument("--w-clear", type=float, default=100.0, help="weight on clearance violations [1/mm^2]")
    parser.add_argument("--opt-stride", type=int, default=2, help="optimise on every n-th waypoint")
    parser.add_argument("--max-iter", type=int, default=300, help="L-BFGS iterations per round")
    parser.add_argument("--restarts", type=int, default=10, help="maximum L-BFGS rounds")
    args = parser.parse_args()

    traj, pos0, quat0 = load_trajectory(args.csv)
    t = traj["t"]
    feat0 = to_features(pos0, quat0)
    K = args.n_ctrl

    # Pin three control points at each end: start/end at rest on the original end poses.
    fixed = np.zeros((K, 6))
    fixed[:3], fixed[-3:] = feat0[0], feat0[-1]
    free_idx = np.arange(3, K - 3)

    t_opt = t[:: args.opt_stride] if t[:: args.opt_stride][-1] == t[-1] else np.r_[t[:: args.opt_stride], t[-1]]
    B = basis(t_opt, K)
    B_out = basis(t, K)
    t_dense = np.linspace(0, 1, 400)
    B3 = basis(t_dense, K, deriv=3)
    target = np.array([np.interp(t_opt, t, feat0[:, j]) for j in range(6)]).T

    # Initial guess: least-squares spline fit of the original trajectory for the planar
    # components, a straight blend of the control points for the out-of-plane ones.
    rhs = target - B @ fixed
    reg = 1e-6 * B3[:, free_idx]
    P_free, *_ = np.linalg.lstsq(np.vstack([B[:, free_idx], reg]),
                                 np.vstack([rhs, -1e-6 * (B3 @ fixed)]), rcond=None)
    blend = np.linspace(0, 1, len(free_idx) + 2)[1:-1, None]
    P_free[:, OFF_PLANE] = (1 - blend) * feat0[0, OFF_PLANE] + blend * feat0[-1, OFF_PLANE]
    fixed[free_idx[:, None], OFF_PLANE] = P_free[:, OFF_PLANE]

    required = args.clearance_mm + (args.contact_mm - args.clearance_mm) * np.clip(
        (t_opt - (1 - args.settle)) / args.settle, 0, 1)
    clearance = ClearanceCost(build_model(), required)
    rows_of = [np.nonzero(B[:, k])[0] for k in free_idx]
    jerk_ref = np.mean((B3 @ np.vstack([fixed[:3], P_free, fixed[-3:]])) ** 2)

    def unpack(x):
        P = fixed.copy()
        P[free_idx[:, None], PLANAR] = x.reshape(-1, len(PLANAR))
        return P

    def cost(x, h=0.05):
        P = unpack(x)
        F = B @ P
        jerk_res = B3 @ P
        track_res = (F - target)[:, PLANAR]
        col = clearance.per_sample(F, np.arange(len(F)))
        J = (np.mean(jerk_res ** 2) / jerk_ref + args.w_track * np.mean(track_res ** 2)
             + args.w_clear * col.sum())
        grad = (2 * B3.T @ jerk_res / jerk_res.size / jerk_ref)[free_idx][:, PLANAR]
        grad += (2 * args.w_track * B.T @ track_res / track_res.size)[free_idx]
        # Clearance gradient by forward differences, re-evaluating only the samples each
        # control point influences.
        for a, (k, rows) in enumerate(zip(free_idx, rows_of)):
            for b, j in enumerate(PLANAR):
                Fp = F.copy()
                Fp[rows, j] += h * B[rows, k]
                grad[a, b] += args.w_clear * (clearance.per_sample(Fp, rows) - col[rows]).sum() / h
        return J, grad.ravel()

    def report(label, F_out):
        pos, quat = from_features(F_out)
        d = clearance.min_dist_mm(pos, quat)
        free = t < 1 - args.settle
        print(f"{label:9s} hull clearance: min {d[free].min():+6.2f} mm before settling, "
              f"{d[~free].min():+6.2f} mm while settling | {smoothness_report(t, F_out)}")

    report("original", feat0)
    report("init fit", B_out @ unpack(P_free[:, PLANAR].ravel()))
    x = minimize_with_restarts(cost, P_free[:, PLANAR].ravel(), args.max_iter, args.restarts)

    F_out = B_out @ unpack(x)
    report("refined", F_out)
    print(f"max deviation from original: {np.linalg.norm(F_out[:, :3] - feat0[:, :3], axis=1).max():.2f} mm")
    save_trajectory(args.out, t, F_out, (pos0[0], quat0[0]), (pos0[-1], quat0[-1]))


if __name__ == "__main__":
    main()
