#!/usr/bin/env python3
"""Active sensing for the weakly observed rotations (REPORT section 7): add a few probe contacts, chosen from the
current estimate to inform the directions the policy's samples leave weak, and measure what they buy.

candidates  hook poses touching the eyelet (its collision field) on the ring or the upper stem (z >= --zmin): an
            orientation the policy's streams produce, one hook point's ball tangent to a surface point, no other
            point penetrating. In the eyelet frame, so one set serves every case; with each its Jacobian row
            (body-frame twists of one tolerance unit, 2 mm / 3 deg: frame-invariant) and approach direction (the
            surface's inward normal).
design      at the case's SMC estimate (pose_filter_bench records), greedy D-optimal: the candidate of largest
            j^T F^-1 j, F the Fisher information of the samples so far (whitened, tolerance units), then F += j j^T;
            "random" draws feasible candidates uniformly.
execution   a guarded move: from the planned pose (the candidate at the estimate) backed off along the surface
            normal, straight along the approach until the hook first reaches the true eyelet's separation offset
            (the case's own: the median separation of its noise-free samples at the truth, the simulator's contact
            floor); the recorded pose takes the same hook-pose noise as the samples. A probe that meets nothing
            within the travel is dropped.
evaluation  batch SMC on samples + probes: polished estimate correct, P(correct), the truth mode (Gauss-Newton from
            the truth) correct, the Fisher eigenvalues at the truth.

    python scripts/probe_design.py --streams S.npz --bench policy_smc.npz --out P.npz --cases 32 --at 160 300
"""

import argparse
import time

import numpy as np
import torch
from scipy.spatial.transform import Rotation

import contact_gpu as G
import contact_information as CI
import pose_filters as PF
from info_metric_bench import noisy, sample_poses
from pose_identifiability import SCALE, jac
from uq_analysis import relative_many, within

TOL = (2.0, 3.0)
CAND = "/media/rp/Elements1/abhay_ws/contact-sampling/fields/probe_candidates.npz"


def to_eyelet(poses, g):
    """Hook poses (k, 7: mm, wxyz) from the robot frame into the frame of the eyelet at g (6)."""
    rg = Rotation.from_rotvec(np.deg2rad(g[3:]))
    r = rg.inv() * Rotation.from_quat(poses[:, 3:], scalar_first=True)
    return np.concatenate([rg.inv().apply(poses[:, :3] - g[:3]), r.as_quat(scalar_first=True)], 1)


def to_robot(poses, g):
    rg = Rotation.from_rotvec(np.deg2rad(g[3:]))
    r = rg * Rotation.from_quat(poses[:, 3:], scalar_first=True)
    return np.concatenate([rg.apply(poses[:, :3]) + g[:3], r.as_quat(scalar_first=True)], 1)


def build_candidates(model, streams, n=400000, zmin=-25.0, seed=0, path=CAND):
    rng = np.random.default_rng(seed)
    dev = model.device
    # surface points and outward normals: field nodes within a voxel of the surface, projected onto it
    lo, res = model.lo.cpu().numpy(), model.res
    shape = model.grid.shape
    near = (model.grid.abs() < res).nonzero().cpu().numpy()
    p = torch.as_tensor(lo + res * near, dtype=torch.float32, device=dev)
    p = p[p[:, 2] >= zmin]
    h = 0.05
    grad = torch.stack([(model.sdf(p + h * e) - model.sdf(p - h * e)) / (2 * h) for e in torch.eye(3, device=dev)], -1)
    nrm = grad / grad.norm(dim=-1, keepdim=True)
    q = p - model.sdf(p)[:, None] * nrm
    pick = rng.integers(0, len(q), n)
    q, nrm = q[pick].cpu().numpy().astype(np.float64), nrm[pick].cpu().numpy().astype(np.float64)
    # orientations the policy produces (hook in the eyelet frame), pooled over streams
    rots = np.concatenate([to_eyelet(s["poses"][rng.integers(0, len(s["poses"]), 200)], s["truth"])[:, 3:]
                           for s in streams])
    R = Rotation.from_quat(rots[rng.integers(0, len(rots), n)], scalar_first=True)
    line, radius = model.line.cpu().numpy().astype(np.float64), model.radius.cpu().numpy().astype(np.float64)
    j = rng.integers(0, len(line), n)
    t = q + np.maximum(radius[j], 0)[:, None] * nrm - R.apply(line[j])
    poses = np.concatenate([t, R.as_quat(scalar_first=True)], 1)
    ident = torch.zeros(1, 6, device=dev)
    sep = torch.cat([model.separations(ident, model.hook_points(poses[a:a + 4096]))[0]
                     for a in range(0, n, 4096)]).cpu().numpy()
    ok = np.abs(sep) <= 0.02  # touching, nothing else inside
    poses, q, nrm = poses[ok], q[ok], nrm[ok]
    prob = PF.Problem(model, poses, 1.0, 20.0)
    _, J = jac(prob, np.zeros((1, 6)), np.diag(SCALE))
    region = np.where(q[:, 2] < -15, "stem", np.where(np.linalg.norm(q[:, 1:] - [-0.28, -5.99], axis=1) < 7.5,
                                                      "ring inner", np.where(np.abs(q[:, 0]) > 2.3, "ring face",
                                                                             "ring outer")))
    np.savez(path, poses=poses, contact=q, normal=nrm, J=J[0], region=region)
    print(f"candidates: {ok.sum()} of {n} feasible; " + ", ".join(f"{r} {np.mean(region == r):.2f}"
                                                                 for r in np.unique(region)))


def design(F, Jc, n, rule, rng, allowed):
    """Indices of n candidates (rows Jc in tolerance units, whitened; those of mask allowed) by greedy D-optimality,
    or at random."""
    if rule == "random":
        return rng.choice(np.nonzero(allowed)[0], n, replace=False)
    F = F.copy() + 1e-6 * np.eye(6)
    out = []
    for _ in range(n):
        gain = np.einsum("ka,ab,kb->k", Jc, np.linalg.inv(F), Jc)
        gain[~allowed] = -np.inf
        gain[out] = -np.inf
        b = int(np.argmax(gain))
        out.append(b)
        F += np.outer(Jc[b], Jc[b])
    return np.array(out)


def guarded(model, planned, normal_robot, truth, target, back=3.0, travel=12.0, step=0.01):
    """Recorded hook poses (m, 7) of guarded moves along -normal from planned + back * normal, stopping where the
    separation from the eyelet at truth first falls to target; NaN rows where nothing is met."""
    u = np.arange(0.0, travel + step, step)
    out = np.full_like(planned, np.nan)
    g = torch.as_tensor(truth[None], dtype=torch.float32, device=model.device)
    for i, (p, n) in enumerate(zip(planned, normal_robot)):
        path = np.repeat(p[None], len(u), 0)
        path[:, :3] = p[:3] + (back - u)[:, None] * n
        s = model.separations(g, model.hook_points(path))[0].cpu().numpy()
        if s[0] <= target:  # already in contact at the start: back off further
            continue
        hit = np.nonzero(s <= target)[0]
        if not len(hit):
            continue
        k = hit[0]
        f = (s[k - 1] - target) / (s[k - 1] - s[k])
        out[i] = path[k - 1]
        out[i, :3] += f * (path[k, :3] - path[k - 1, :3])
    return out


def evaluate(prob, truth, seed):
    out = PF.smc(prob, n=4096, seed=seed)
    best, cost, _, _ = PF.polish(prob, out)
    est = best.cpu().numpy().astype(np.float64)
    err = CI.relative(truth, est)
    w = PF.normalise(out["logw"]).cpu().numpy()
    x = out["x"].cpu().numpy().astype(np.float64)
    p_correct = float(w @ within(relative_many(x, est), TOL))
    t = torch.tensor(truth[None], dtype=torch.float32, device=prob.dev)
    tm, _, _ = PF.gauss_newton(prob, t.clone(), iters=30)
    tm_err = CI.relative(truth, tm[0].cpu().numpy().astype(np.float64))
    _, J = jac(prob, truth[None], np.diag(SCALE))
    return {"err": err, "correct": bool(within(err, TOL)[0]), "p_correct": p_correct, "tm_err": tm_err,
            "tm_correct": bool(within(tm_err, TOL)[0]), "lam": np.linalg.eigvalsh(J[0].T @ J[0]), "est": est}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--streams", required=True)
    p.add_argument("--bench", required=True, help="pose_filter_bench SMC records of the same cases (the estimates)")
    p.add_argument("--out", required=True)
    p.add_argument("--cases", type=int, default=32)
    p.add_argument("--at", type=int, nargs="+", default=[160, 300])
    p.add_argument("--noise", type=float, nargs="+", default=[0.0, 0.3])
    p.add_argument("--sigma", type=float, nargs="+", default=[0.05, 0.33])
    p.add_argument("--probes", type=int, nargs="+", default=[2, 4, 8])
    p.add_argument("--ring", type=int, nargs="+", default=[2, 4], help="probe counts, D-optimal on the ring only")
    p.add_argument("--random", type=int, nargs="+", default=[4, 8], help="probe counts for the random control")
    p.add_argument("--zmin", type=float, default=-25.0)
    p.add_argument("--max-samples", type=int, default=150)
    p.add_argument("--resume", action="store_true", help="keep --out's records and skip their streams")
    args = p.parse_args()
    streams = np.load(args.streams, allow_pickle=True)["streams"]
    model = G.GpuContactModel()
    try:
        C = np.load(CAND)
    except FileNotFoundError:
        build_candidates(model, streams, zmin=args.zmin)
        C = np.load(CAND)
    bench = {(r["stream"], r["step"], float(r["noise"])): r for r in np.load(args.bench, allow_pickle=True)["records"]}
    recs = list(np.load(args.out, allow_pickle=True)["records"]) if args.resume else []
    done = {r["stream"] for r in recs}
    for sid in range(args.cases):
        if sid in done:
            continue
        s, t0 = streams[sid], time.perf_counter()
        truth = np.asarray(s["truth"], np.float64)
        for step in args.at:
            clean = sample_poses(s, step, 0.0, 0.0, args.max_samples, 1000 * sid + step)[0]
            t = torch.tensor(truth[None], dtype=torch.float32, device=model.device)
            floor = float(np.median(model.separations(t, model.hook_points(clean))[0].cpu().numpy()))
            for noise, sigma in zip(args.noise, args.sigma):
                b = bench.get((sid, step, noise))
                if b is None:
                    continue
                poses = sample_poses(s, step, noise, 0.0, args.max_samples, 1000 * sid + step)[0]
                seed = 1000 * sid + step + 7919 * int(round(noise * 10))
                rng = np.random.default_rng(seed + 1)
                est = np.asarray(b["pol_pose"], np.float64)
                base = PF.Problem(model, poses, sigma, 20.0)
                _, J = jac(base, est[None], np.diag(SCALE))
                F = J[0].T @ J[0]
                Jc = C["J"] / sigma
                variants = ([("none", 0)] + [("dopt", n) for n in args.probes] + [("dopt-ring", n) for n in args.ring]
                            + [("random", n) for n in args.random])
                ring = C["region"] != "stem"
                for rule, n in variants:
                    rec = {"stream": sid, "step": step, "noise": noise, "rule": rule, "n": n, "floor": floor}
                    if n:
                        idx = design(F, Jc, n, "random" if rule == "random" else "dopt", rng,
                                     ring if rule == "dopt-ring" else np.ones(len(Jc), bool))
                        planned = to_robot(C["poses"][idx], est)
                        nrm = Rotation.from_rotvec(np.deg2rad(est[3:])).apply(C["normal"][idx])
                        rec_poses = guarded(model, planned, nrm, truth, floor)
                        hit = ~np.isnan(rec_poses[:, 0])
                        rec.update(region=C["region"][idx], hit=hit, contact=C["contact"][idx])
                        probes = noisy(rec_poses[hit], noise, rng) if hit.any() else rec_poses[:0]
                        prob = PF.Problem(model, np.concatenate([poses, probes]), sigma, 20.0)
                    else:
                        prob = base
                    rec.update(evaluate(prob, truth, seed))
                    recs.append(rec)
        print(f"stream {sid} done ({time.perf_counter() - t0:.0f} s)", flush=True)
        np.savez(args.out, records=np.array(recs, dtype=object), args=str(vars(args)))


if __name__ == "__main__":
    main()
