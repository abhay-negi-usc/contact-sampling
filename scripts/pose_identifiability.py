#!/usr/bin/env python3
"""Is the eyelet pose identifiable from a case's contact samples? Continuous and discrete symmetries of the solution
set, tested about the truth on pose_filter_bench's cases (same samples, same seeds):

local     Fisher information F = J^T J of the whitened residuals at the truth, J in body-frame twists of one
          tolerance unit (2 mm, 3 deg) per axis: an eigenvalue is the cost increase of a one-tolerance move along its
          eigenvector; below 3.84 (chi2_1, 95%) that direction is not determined to tolerance. The weakest
          eigenvector as a screw (axis, point, pitch), against the loop's axis (a near-circle, rms 0.38 mm off one).
profile   profile cost (Raue et al. 2009) along rotation about the loop's axis and along the weakest screw: at each
          point of the path the five other directions are re-optimised (Gauss-Newton). Flat over the whole range: a
          continuous symmetry of the data (structurally non-identifiable); bounded, but under 3.84 beyond the
          tolerance: practically non-identifiable. Its minimum's offset from the truth is how far the data's errors
          drag the optimum along the valley.
discrete  1024-start Gauss-Newton: distinct modes within 6 of the best cost that are outside the tolerance and off
          both profile paths, with their transforms relative to the truth (eyelet frame): discrete ambiguities.
Data: "exact" offsets every separation so the truth fits with zero residual (noise-free and model-consistent: only
the samples' geometry); "real" is the simulator's samples (at noise 0 its contact floor, 0.03-0.06 mm, remains).

    python scripts/pose_identifiability.py --streams S.npz --out I.npz --cases 32 --at 20 40 80 160 300
"""

import argparse
import time

import numpy as np
import torch
from scipy.spatial.transform import Rotation

import contact_gpu as G
import contact_information as CI
import pose_filters as PF
from info_metric_bench import sample_poses

TOL = np.array([2.0, 3.0])
SCALE = np.array([2.0, 2.0, 2.0, *np.deg2rad([3.0, 3.0, 3.0])])  # one tolerance unit as a twist (mm, rad)
LOOP_CENTRE = np.array([0.0, -0.28, -5.99])  # eyelet frame, mm: the loop's best-fit circle, axis along x
LOOP_AXIS = np.array([1.0, 0.0, 0.0])
CHI2_1 = 3.84


def hat(w):
    W = np.zeros((len(w), 3, 3))
    W[:, 0, 1], W[:, 0, 2], W[:, 1, 2] = -w[:, 2], w[:, 1], -w[:, 0]
    return W - W.transpose(0, 2, 1)


def compose(g, xi):
    """g (N, 6: mm, rotation vector deg, eyelet in the robot frame) moved by body-frame twists xi (N, 6: mm, rad):
    g exp(xi^)."""
    v, w = xi[:, :3], xi[:, 3:]
    th = np.linalg.norm(w, axis=1)
    t2 = np.maximum(th, 1e-9) ** 2
    a = np.where(th > 1e-6, (1 - np.cos(th)) / t2, 0.5)
    b = np.where(th > 1e-6, (th - np.sin(th)) / (t2 * np.maximum(th, 1e-9)), 1 / 6)
    W = hat(w)
    V = np.eye(3) + a[:, None, None] * W + b[:, None, None] * W @ W
    Rg = Rotation.from_rotvec(np.deg2rad(g[:, 3:]))
    return np.concatenate([g[:, :3] + Rg.apply(np.einsum("nab,nb->na", V, v)),
                           (Rg * Rotation.from_rotvec(w)).as_rotvec(degrees=True)], 1)


def residuals(prob, g):
    return prob.residuals(torch.as_tensor(g, dtype=torch.float32, device=prob.dev)).double().cpu().numpy()


def jac(prob, g, dirs, h=0.05):
    """Residuals (N, k) and their Jacobian (N, k, m) in the twists dirs (m, 6) by central differences."""
    N, m = len(g), len(dirs)
    probes = [g] + [compose(g, np.tile(s * h * d, (N, 1))) for d in dirs for s in (1, -1)]
    r = residuals(prob, np.concatenate(probes))
    rr = r[N:].reshape(m, 2, N, -1)
    return r[:N], ((rr[:, 0] - rr[:, 1]) / (2 * h)).transpose(1, 2, 0)


def gn(prob, g, dirs, iters=25):
    """Levenberg-Marquardt of every pose at once, moving only along the twists dirs. Returns (poses, costs)."""
    lam = np.full(len(g), 1e-2)
    r, J = jac(prob, g, dirs)
    c = (r ** 2).sum(-1)
    for _ in range(iters):
        A = J.transpose(0, 2, 1) @ J
        b = (J.transpose(0, 2, 1) @ r[..., None])[..., 0]
        d = np.diagonal(A, axis1=1, axis2=2)
        M = A + (lam[:, None] * d + 1e-9)[..., None] * np.eye(len(dirs))
        trial = compose(g, -np.linalg.solve(M, b[..., None])[..., 0] @ dirs)
        ct = (residuals(prob, trial) ** 2).sum(-1)
        better = ct < c
        g = np.where(better[:, None], trial, g)
        lam = np.where(better, np.maximum(lam / 10, 1e-6), lam * 10)
        if not better.any() and lam.min() > 1e4:
            break
        if better.any():
            r[better], J[better] = jac(prob, g[better], dirs)
            c = np.where(better, ct, c)
    return g, c


def screw(xi):
    """Axis direction, the axis' point nearest the origin, pitch (mm / rad) of a twist (mm, rad)."""
    v, w = xi[:3], xi[3:]
    n2 = w @ w
    if n2 < 1e-12:
        return None, None, np.inf
    return w / np.sqrt(n2), np.cross(w, v) / n2, (w @ v) / n2


def profile(prob, truth, d, s):
    """Profile cost along the twist d (per unit of s) from the truth: the other five directions (orthogonal to d in
    tolerance units) re-optimised at each s. Returns (optimised poses, costs)."""
    u = d / SCALE
    u = u / np.linalg.norm(u)
    B = np.linalg.svd(u[None])[2][1:] * SCALE  # (5, 6) twists
    g0 = compose(np.tile(truth, (len(s), 1)), s[:, None] * d[None])
    return gn(prob, g0, B)


def interval(s, delta, at):
    """The contiguous s-range about index `at` where delta <= CHI2_1; inf at an end that reaches the grid's edge."""
    lo = hi = at
    while lo > 0 and delta[lo - 1] <= CHI2_1:
        lo -= 1
    while hi < len(s) - 1 and delta[hi + 1] <= CHI2_1:
        hi += 1
    return (-np.inf if lo == 0 else s[lo]), (np.inf if hi == len(s) - 1 else s[hi])


def tol_dist(a, b):
    """Frame-free distance of poses a (n, 6) to b (6) in tolerance units: max(position mm / 2, angle deg / 3)."""
    e = np.atleast_2d(np.asarray([CI.relative(x, b) for x in np.atleast_2d(a)]))
    ang = np.linalg.norm(e[:, 3:], axis=1)
    return np.maximum(np.linalg.norm(e[:, :3], axis=1) / TOL[0], ang / TOL[1])


def analyse(prob, truth, seed, s_loop, s_weak):
    out = {}
    # local
    r0, J = jac(prob, truth[None], np.diag(SCALE))
    F = J[0].T @ J[0]
    lam, U = np.linalg.eigh(F)
    xi_w = U[:, 0] * SCALE
    axis, point, pitch = screw(xi_w)
    out.update(lam=lam, U=U, rot_share=float((U[3:, 0] ** 2).sum()), pitch=pitch,
               axis_angle=np.nan if axis is None else float(np.degrees(np.arccos(min(1, abs(axis @ LOOP_AXIS))))),
               axis_dist=np.nan if axis is None else float(np.linalg.norm(np.cross(LOOP_CENTRE - point, axis))),
               truth_cost=float((r0 ** 2).sum()))
    # profiles: rotation about the loop axis (s in deg), the weakest screw (s in tolerance units)
    w = np.deg2rad(1.0) * LOOP_AXIS
    d_loop = np.concatenate([-np.cross(w, LOOP_CENTRE), w])
    paths = []
    for name, d, s in (("loop", d_loop, s_loop), ("weak", xi_w, s_weak)):
        g, c = profile(prob, truth, d, s)
        b = int(np.argmin(c))
        delta = c - c[b]
        out[f"{name}_cost"], out[f"{name}_min_s"] = c, float(s[b])
        out[f"{name}_interval"] = interval(s, delta, b)
        inside = delta <= CHI2_1  # the 95% profile confidence set, possibly disjoint: its extent
        out[f"{name}_set"] = (-np.inf if inside[0] else s[inside].min(), np.inf if inside[-1] else s[inside].max())
        out[f"{name}_pose_err_at_min"] = CI.relative(truth, g[b])
        paths.append(g)
    paths = np.concatenate(paths)
    # discrete: distinct polished modes of a 1024-start search
    ms = PF.multistart(prob, n=1024, seed=seed)
    idx = PF.distinct_best(ms["x"], -ms["cost"], 32)
    x, c, _ = PF.gauss_newton(prob, ms["x"][idx], iters=30)
    x, c = x.cpu().numpy().astype(np.float64), c.cpu().numpy()
    best = min(c.min(), out["loop_cost"].min(), out["weak_cost"].min())
    keep = c <= best + 6
    modes, costs = x[keep], c[keep]
    dt = tol_dist(modes, truth)
    dpath = np.array([tol_dist(paths, m).min() for m in modes])
    cls = np.where(dt <= 1, "truth", np.where(dpath <= 1, "valley", "discrete"))
    out.update(modes=modes, mode_costs=costs, mode_class=cls, best_cost=float(best),
               mode_rel=np.array([CI.relative(m, truth) for m in modes]),  # the mode in the truth's (eyelet) frame
               search_best_err=CI.relative(truth, x[int(np.argmin(c))]))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--streams", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--cases", type=int, default=32)
    p.add_argument("--at", type=int, nargs="+", default=[20, 40, 80, 160, 300])
    p.add_argument("--noise", type=float, default=0.0)
    p.add_argument("--sigma", type=float, default=0.05)
    p.add_argument("--max-samples", type=int, default=150)
    p.add_argument("--bound", type=float, default=20.0)
    p.add_argument("--field", default=str(G.FIELD), help="eyelet field (contact_gpu.FIELD or FIELD_SMOOTH)")
    args = p.parse_args()
    streams = np.load(args.streams, allow_pickle=True)["streams"][:args.cases]
    model = G.GpuContactModel(field=args.field)
    s_loop = np.arange(-30, 30.01, 1.5)
    s_weak = np.arange(-8, 8.01, 0.4)
    recs = []
    for sid, s in enumerate(streams):
        t0 = time.perf_counter()
        for step in args.at:
            k = int(s["counts"][min(step, len(s["counts"]) - 1)])
            if k < 3:
                continue
            poses = sample_poses(s, step, args.noise, 0.0, args.max_samples, 1000 * sid + step)[0]
            truth = np.asarray(s["truth"], np.float64)
            seed = 1000 * sid + step + 7919 * int(round(args.noise * 10))
            rec = {"stream": sid, "step": step, "used": len(poses), "truth": truth}
            for data in ("exact", "real"):
                prob = PF.Problem(model, poses, args.sigma, args.bound)
                if data == "exact":
                    t = torch.tensor(truth[None], dtype=torch.float32, device=prob.dev)
                    prob.offset = -model.fast_separations(t, prob.pts)[0]
                rec[data] = analyse(prob, truth, seed, s_loop, s_weak)
            recs.append(rec)
        print(f"stream {sid} done ({time.perf_counter() - t0:.0f} s)", flush=True)
    np.savez(args.out, records=np.array(recs, dtype=object), s_loop=s_loop, s_weak=s_weak, args=str(vars(args)))


if __name__ == "__main__":
    main()
