#!/usr/bin/env python3
"""Optimise the hook trajectory from its start pose to the assembled pose for maximum separation.

Separation is the signed minimum distance between the hook and the eyelet: positive with space
between them, negative when they interfere. It is measured between the convex collision parts,
which enclose the true meshes, so it under-reports the true separation by at most ~0.25 mm.

The assembled pose is itself in contact, so separation must fall to ~0 at the end, and a start
pose near the eyelet fixes a small separation at the beginning. The optimiser maximises the
worst-case separation s along the path, with the requirement tapering over the departure and the
final approach to whatever the end poses allow:

    maximise s   subject to   sep(t) >= floor(t) + (s - floor(t)) * r(t),
    r(t) = min(1, (t / depart) ** 3, ((1 - t) / settle) ** 3)

where floor is `contact` on arrival and min(contact, start separation) on departure (the cubic
taper matches how a rest-to-rest motion opens or closes a gap). The minimum is a smooth
log-sum-exp, and a small jerk penalty picks the smoothest of the best paths. The path is the
rest-to-rest quintic B-spline of refine_assembly_trajectory.py, but all six pose components are
free: centring the hook in the eyelet's hole needs y and the tilts as well as x, z and pitch.

    conda run -n mujoco_env python scripts/optimize_max_separation_trajectory.py
"""

import argparse
from pathlib import Path

import mujoco
import numpy as np

from hook_eyelet_model import ASSET_DIR, Separation, build_model, load_trajectory
from refine_assembly_trajectory import (basis, from_features, minimize_with_restarts, save_trajectory,
                                        smoothness_report, to_features)

R_MIN = 0.01  # below this taper the motion only has to stay collision-free


def taper_and_floor(t, depart, settle, contact, start_sep):
    r_start = np.minimum(1.0, t / depart) ** 3
    r_end = np.minimum(1.0, (1 - t) / settle) ** 3
    floor = np.where(r_start < r_end, min(contact, start_sep), contact)
    return np.minimum(r_start, r_end), floor


class SeparationProfile:
    def __init__(self):
        self.m = build_model()
        self.d = mujoco.MjData(self.m)
        self.sep = Separation(self.m)

    def __call__(self, feat):
        """Separation [mm] at each pose."""
        pos, quat = from_features(np.atleast_2d(feat))
        out = np.empty(len(pos))
        for i, (p, q) in enumerate(zip(pos, quat)):
            self.d.qpos[:7] = np.concatenate([p, q])
            mujoco.mj_kinematics(self.m, self.d)
            out[i] = self.sep(self.m, self.d)[0] * 1e3
        return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", type=Path, default=ASSET_DIR / "hook_assembly_ideal_trajectory.csv",
                        help="trajectory giving the start pose, the assembled pose and the time grid")
    parser.add_argument("--init", type=Path, default=ASSET_DIR / "hook_assembly_refined_trajectory.csv",
                        help="initial guess; should already thread the hook through the eyelet")
    parser.add_argument("--out", type=Path, default=ASSET_DIR / "hook_assembly_max_separation_trajectory.csv")
    parser.add_argument("--n-ctrl", type=int, default=20, help="B-spline control points")
    parser.add_argument("--depart", type=float, default=0.15,
                        help="initial fraction of the motion over which the separation requirement tapers")
    parser.add_argument("--settle", type=float, default=0.15,
                        help="final fraction of the motion over which the separation requirement tapers")
    parser.add_argument("--contact-mm", type=float, default=-0.15,
                        help="hull separation allowed at the assembled pose (the hulls bulge ~0.13 mm there)")
    parser.add_argument("--softmin-mm", type=float, nargs="+", default=[0.3, 0.1],
                        help="log-sum-exp temperatures of the soft minimum, optimised in turn; starting "
                             "smooth and tightening gives the same worst case with a smoother path")
    parser.add_argument("--w-jerk", type=float, default=0.05,
                        help="weight on jerk, relative to the initial guess's jerk [mm of separation]")
    parser.add_argument("--w-end", type=float, default=100.0,
                        help="weight on interference during the last moments of arrival [1/mm]")
    parser.add_argument("--opt-stride", type=int, default=2, help="optimise on every n-th waypoint")
    parser.add_argument("--max-iter", type=int, default=300, help="L-BFGS iterations per round")
    parser.add_argument("--restarts", type=int, default=10, help="maximum L-BFGS rounds")
    args = parser.parse_args()

    traj, pos0, quat0 = load_trajectory(args.csv)
    t = traj["t"]
    feat0 = to_features(pos0, quat0)
    init_traj, init_pos, init_quat = load_trajectory(args.init)
    feat_init = to_features(init_pos, init_quat)
    K = args.n_ctrl

    fixed = np.zeros((K, 6))
    fixed[:3], fixed[-3:] = feat0[0], feat0[-1]
    free_idx = np.arange(3, K - 3)

    t_opt = t[:: args.opt_stride] if t[:: args.opt_stride][-1] == t[-1] else np.r_[t[:: args.opt_stride], t[-1]]
    B = basis(t_opt, K)
    B_out = basis(t, K)
    B3 = basis(np.linspace(0, 1, 400), K, deriv=3)
    target = np.array([np.interp(t_opt, init_traj["t"], feat_init[:, j]) for j in range(6)]).T
    P_free, *_ = np.linalg.lstsq(B[:, free_idx], target - B @ fixed, rcond=None)

    beta, contact = args.softmin_mm[0], args.contact_mm
    profile = SeparationProfile()
    start_sep = profile(feat0[0])[0]
    taper, floor = taper_and_floor(t_opt, args.depart, args.settle, contact, start_sep)
    in_softmin = taper >= R_MIN
    rows_of = [np.nonzero(B[:, k])[0] for k in free_idx]
    jerk_ref = np.mean((B3 @ np.vstack([fixed[:3], P_free, fixed[-3:]])) ** 2)

    def unpack(x):
        P = fixed.copy()
        P[free_idx] = x.reshape(-1, 6)
        return P

    def per_sample(sep, rows):
        """Each sample's share of the soft minimum (unnormalised) and of the end-pose penalty."""
        r, s, f = taper[rows], in_softmin[rows], floor[rows]
        g = np.where(s, f + (sep - f) / np.where(s, r, 1.0), np.inf)
        return np.exp(-(g - g0) / beta), np.where(s, 0.0, np.maximum(0.0, f - sep) ** 2)

    def objective(soft_sum, end_sum):
        return -(g0 - beta * np.log(soft_sum)) + args.w_end * end_sum

    def cost(x, h=0.05):
        nonlocal g0
        P = unpack(x)
        F = B @ P
        sep = profile(F)
        g0 = np.min((floor + (sep - floor) / np.maximum(taper, R_MIN))[in_softmin])
        soft, end = per_sample(sep, np.arange(len(F)))
        J0 = objective(soft.sum(), end.sum())
        jerk_res = B3 @ P
        J = J0 + args.w_jerk * np.mean(jerk_res ** 2) / jerk_ref
        grad = (2 * args.w_jerk * B3.T @ jerk_res / jerk_res.size / jerk_ref)[free_idx]
        # Separation gradient by forward differences, re-evaluating only the samples each
        # control point influences.
        for a, (k, rows) in enumerate(zip(free_idx, rows_of)):
            for j in range(6):
                Fp = F[rows].copy()
                Fp[:, j] += h * B[rows, k]
                soft_p, end_p = per_sample(profile(Fp), rows)
                Jp = objective(soft.sum() - soft[rows].sum() + soft_p.sum(), end.sum() - end[rows].sum() + end_p.sum())
                grad[a, j] += (Jp - J0) / h
        return J, grad.ravel()

    g0 = 0.0

    def report(label, F_out):
        sep = profile(F_out)
        r, f = taper_and_floor(t, args.depart, args.settle, contact, start_sep)
        mid = (t > args.depart) & (t < 1 - args.settle)
        worst = np.min((f + (sep - f) / np.maximum(r, R_MIN))[r >= R_MIN])
        print(f"{label:10s} separation: min {sep[mid].min():+5.2f} mm between departure and final approach, "
              f"worst tapered {worst:+5.2f} mm, start {sep[0]:+5.2f} mm, final {sep[-1]:+5.2f} mm | "
              f"{smoothness_report(t, F_out)}")

    report(args.csv.stem[:10], feat0)
    report("init", feat_init)
    x = P_free.ravel()
    for beta in args.softmin_mm:
        print(f"soft-min temperature {beta} mm:")
        x = minimize_with_restarts(cost, x, args.max_iter, args.restarts)
    F_out = B_out @ unpack(x)
    report("optimised", F_out)
    save_trajectory(args.out, t, F_out, (pos0[0], quat0[0]), (pos0[-1], quat0[-1]))


if __name__ == "__main__":
    main()
