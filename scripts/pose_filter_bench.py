#!/usr/bin/env python3
"""Benchmark of the particle estimators of the eyelet pose (pose_filters.py) on simulated contact streams.

Batch methods see, at each policy step of --at, the samples info_metric_bench.py drew there (sample_poses: at most
--max-samples evenly thinned, iid hook-pose noise of std --noise, the same seed), so their results pair case by
case with that benchmark's multi-start Levenberg-Marquardt. Sequential methods run once per stream through the
policy steps up to max(--at), each step's new samples (at most --per-step, evenly thinned; noise drawn per raw
sample from the stream's own seed) one observation, and report at each checkpoint.

Per case (stream, step, noise, method): the truth's cost and its mode (Gauss-Newton from the truth: the posterior
mode the data hold there), the estimator's best particle by likelihood (raw) and its Gauss-Newton polish (pol: the
point estimate), the particle-weighted mean (sequential), their errors (true pose relative to the estimate, x, y, z
mm, rotation vector deg) and costs (sum of squared whitened residuals on the batch samples), a subsample of the
particles with their log likelihoods and log weights (for uncertainty analyses), time and separations computed.

    python scripts/pose_filter_bench.py --streams S.npz --out R.npz --methods smc scaling multistart
"""

import argparse
import json
import time

import numpy as np
import torch

import contact_gpu as G
import contact_information as CI
import pose_filters as PF
from info_metric_bench import noisy, sample_poses

SNAPSHOTS = (1 / 16, 1 / 8, 1 / 4, 1 / 2)  # SMC power posteriors kept: sigma x 4, 2.8, 2, 1.4
BATCH = {
    "smc": lambda prob, seed, o: PF.smc(prob, n=o.get("n", 4096), seed=seed, ess_frac=o.get("ess", 0.5),
                                        moves=o.get("moves", 5), snapshots=o.get("snapshots", SNAPSHOTS)),
    "scaling": lambda prob, seed, o: PF.scaling_series(prob, n=o.get("n", 4096), per=o.get("per", 8), seed=seed),
    "multistart": lambda prob, seed, o: PF.multistart(prob, n=o.get("n", 1024), seed=seed),
    # with --debias: the search on the plain likelihood, then SMC bridged to the min-bias corrected posterior /
    # only the polish on it
    "smc-db": lambda prob, seed, o: (PF.smc_sequence([prob.plain(), prob], n=o.get("n", 4096), seed=seed,
                                                     moves=o.get("moves", 5))[-1] if prob.debias else
                                     PF.smc(prob, n=o.get("n", 4096), seed=seed, moves=o.get("moves", 5))),
    "multistart-db": lambda prob, seed, o: PF.multistart(prob.plain(), n=o.get("n", 1024), seed=seed),
}


def err(truth, g):
    return CI.relative(truth, np.asarray(g, np.float64))


def keep_particles(out, n_keep, gen_seed):
    x, ll, lw = out["x"], out["ll"], out["logw"]
    idx = torch.randperm(len(x), generator=torch.Generator().manual_seed(gen_seed))[:n_keep].to(x.device)
    return {"particles": x[idx].cpu().numpy().astype(np.float32), "particles_ll": ll[idx].cpu().numpy(),
            "particles_logw": lw[idx].cpu().numpy()}


def case_reference(prob, truth):
    t = torch.tensor(truth[None], dtype=torch.float32, device=prob.dev)
    truth_cost = float((prob.residuals(t).double() ** 2).sum())
    tm, tm_cost, _ = PF.gauss_newton(prob, t.clone(), iters=30)
    tm = tm[0].cpu().numpy().astype(np.float64)
    return {"truth_cost": truth_cost, "tm_pose": tm, "tm_cost": float(tm_cost[0]), "tm_err": err(truth, tm)}


def well_specified(prob, truth, seed):
    """Replace the data's residuals by the model's own: per-sample offsets so that the residuals at the truth are
    N(0, sigma^2) exactly (the samples' geometry kept): the calibration ceiling of an exact likelihood."""
    t = torch.tensor(truth[None], dtype=torch.float32, device=prob.dev)
    eps = torch.randn(prob.k, generator=torch.Generator(device=prob.dev).manual_seed(seed), device=prob.dev)
    prob.offset = eps * prob.sigma - prob.model.fast_separations(t, prob.pts)[0]


def summarise(prob, truth, out, rec, n_keep, seed):
    raw = out["x"][torch.argmax(out["ll"])].cpu().numpy().astype(np.float64)
    best, cost, polished, pcost = PF.polish(prob, out)
    best = best.cpu().numpy().astype(np.float64)
    w = PF.normalise(out["logw"])
    mean = (w[:, None].double() * out["x"].double()).sum(0).cpu().numpy()
    t = torch.tensor(np.stack([raw, mean]), dtype=torch.float32, device=prob.dev)
    c = (prob.residuals(t).double() ** 2).sum(-1).cpu().numpy()
    r, J = PF.jacobian(prob, torch.tensor(best[None], dtype=torch.float32, device=prob.dev))
    rec.update(pol_info=(J[0].T @ J[0]).double().cpu().numpy(), pol_resid=(r[0] * prob.sigma).cpu().numpy())
    rec.update(raw_pose=raw, raw_err=err(truth, raw), raw_cost=float(c[0]), pol_pose=best, pol_err=err(truth, best),
               pol_cost=cost, mean_pose=mean, mean_err=err(truth, mean), mean_cost=float(c[1]),
               modes_pose=polished.cpu().numpy(), modes_cost=pcost.cpu().numpy(), ess=PF.ess(out["logw"]),
               **keep_particles(out, n_keep, seed))
    return rec


def run_batch(model, streams, args, method, opts):
    recs = []
    for sid, s in streams:
        for step in args.at:
            for noise, sigma in zip(args.noise, args.sigma):
                k = int(s["counts"][min(step, len(s["counts"]) - 1)])
                if k < 3:
                    continue
                poses, idx, *_ = sample_poses(s, step, noise, args.bias, args.max_samples, 1000 * sid + step)
                prob = PF.Problem(model, poses, sigma, args.bound, offset=args.offset, robust=args.robust,
                                  debias=(noise, args.debias, 1000 * sid + step) if args.debias and noise > 0 else None)
                if args.synthetic:
                    well_specified(prob, s["truth"], 1000 * sid + step + 7919 * int(round(noise * 10)))
                rec = {"method": method, "stream": sid, "step": step, "noise": noise, "sigma": sigma, "samples": k,
                       "used": len(poses), "truth": s["truth"], **case_reference(prob, s["truth"])}
                prob.evals = 0
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                seed = 1000 * sid + step + 7919 * int(round(noise * 10))
                out = BATCH[method](prob, seed, opts)
                summarise(prob, s["truth"], out, rec, args.keep, seed)
                torch.cuda.synchronize()
                rec.update(time_s=time.perf_counter() - t0, evals=prob.evals, stages=out.get("stages", 0),
                           snapshots=out.get("snapshots", {}))
                recs.append(rec)
        print(f"{method} stream {sid} done", flush=True)
    return recs


def run_online(model, streams, args, method, opts):
    """smc_sequence over the checkpoints of --at: at each, the batch cases' samples (sample_poses)."""
    recs = []
    for sid, s in streams:
        for noise, sigma in zip(args.noise, args.sigma):
            steps = [st for st in args.at if int(s["counts"][min(st, len(s["counts"]) - 1)]) >= 3]
            if not steps:
                continue
            probs = [PF.Problem(model, sample_poses(s, st, noise, args.bias, args.max_samples, 1000 * sid + st)[0],
                                sigma, args.bound, offset=args.offset, robust=args.robust) for st in steps]
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            outs = PF.smc_sequence(probs, n=opts.get("n", 4096), seed=1000 * sid + 13, moves=opts.get("moves", 5))
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            for st, prob, out in zip(steps, probs, outs):
                rec = {"method": method, "stream": sid, "step": st, "noise": noise, "sigma": sigma,
                       "samples": int(s["counts"][min(st, len(s["counts"]) - 1)]), "used": prob.k,
                       "truth": s["truth"], **case_reference(prob, s["truth"])}
                summarise(prob, s["truth"], out, rec, args.keep, 1000 * sid + st)
                rec.update(time_s=dt / len(steps), evals=0, stages=out["stages"], snapshots={})
                recs.append(rec)
        print(f"{method} stream {sid} done", flush=True)
    return recs


def stream_observations(s, noise, per_step, last, seed):
    """[(policy step, noisy poses of that step's new samples, at most per_step)] up to policy step last."""
    rng = np.random.default_rng(seed)
    poses = noisy(s["poses"], noise, rng)
    counts = s["counts"]
    obs = []
    for step in range(1, min(last, len(counts) - 1) + 1):
        a, b = int(counts[step - 1]), int(counts[step])
        if b > a:
            idx = np.unique(np.linspace(a, b - 1, min(per_step, b - a)).astype(int))
            obs.append((step, poses[idx]))
    return obs


def run_sequential(model, streams, args, method, opts):
    kind, proposal = (method.split("-") + ["ball"])[:2]
    recs = []
    for sid, s in streams:
        for noise, sigma in zip(args.noise, args.sigma):
            obs = stream_observations(s, noise, opts.get("per_step", args.per_step), max(args.at), 1000 * sid + 7)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            outs = PF.sequential(model, obs, sigma, args.bound, kind=kind, n=opts.get("n", 2048),
                                 phi=opts.get("phi", 1.0), proposal=proposal, seed=1000 * sid + 11,
                                 rough=opts.get("rough", 0.1), ball=opts.get("ball"), checkpoints=args.at,
                                 bw_scale=opts.get("bw", 1.0))
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            for step in args.at:
                k = int(s["counts"][min(step, len(s["counts"]) - 1)])
                if k < 3:
                    continue
                poses, *_ = sample_poses(s, step, noise, args.bias, args.max_samples, 1000 * sid + step)
                prob = PF.Problem(model, poses, sigma, args.bound, offset=args.offset, robust=args.robust)
                out = dict(outs[step])
                out["ll"] = prob.loglik(out["x"])  # batch likelihood, to pick and polish the point estimate
                rec = {"method": method, "stream": sid, "step": step, "noise": noise, "sigma": sigma, "samples": k,
                       "used": len(poses), "truth": s["truth"], **case_reference(prob, s["truth"]),
                       "seq_samples": int(sum(len(p) for st, p in obs if st <= step))}
                summarise(prob, s["truth"], out, rec, args.keep, 1000 * sid + step)
                rec.update(time_s=dt, evals=0, stages=0)
                recs.append(rec)
        print(f"{method} stream {sid} done", flush=True)
    return recs


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--streams", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--methods", nargs="+", required=True,
                   help="smc, scaling, multistart, smc-db, multistart-db (batch); smc-online; cpf, mpf-uniform, "
                        "mpf-particle, mpf-ball (sequential)")
    p.add_argument("--opts", type=json.loads, default={}, help="per-method options, JSON {method: {...}}")
    p.add_argument("--at", type=int, nargs="+", default=[10, 20, 40, 80, 160, 300])
    p.add_argument("--noise", type=float, nargs="+", default=[0.0, 0.3])
    p.add_argument("--sigma", type=float, nargs="+")
    p.add_argument("--bias", type=float, default=0.0)
    p.add_argument("--bound", type=float, default=20.0)
    p.add_argument("--offset", type=float, default=0.0)
    p.add_argument("--robust", type=float, help="Student-t degrees of freedom (default Gaussian)")
    p.add_argument("--synthetic", action="store_true",
                   help="well-specified control: residuals at the truth replaced by N(0, sigma^2) (well_specified)")
    p.add_argument("--debias", type=int, default=0,
                   help="noise draws M of the min-bias correction (Problem.min_bias; batch methods, noise > 0)")
    p.add_argument("--max-samples", type=int, default=150)
    p.add_argument("--per-step", type=int, default=3)
    p.add_argument("--cases", type=int)
    p.add_argument("--first", type=int, default=0)
    p.add_argument("--keep", type=int, default=512, help="particles saved per case")
    args = p.parse_args()
    args.sigma = args.sigma or [max(0.05, 1.1 * n) for n in args.noise]  # as info_metric_bench
    data = np.load(args.streams, allow_pickle=True)["streams"]
    streams = list(enumerate(data))[args.first: None if args.cases is None else args.first + args.cases]
    model = G.GpuContactModel()
    recs = []
    t0 = time.time()
    for method in args.methods:
        opts = args.opts.get(method, {})
        fn = run_batch if method in BATCH else run_online if method == "smc-online" else run_sequential
        recs += fn(model, streams, args, method, opts)
        np.savez(args.out, records=np.array(recs, dtype=object), args=json.dumps(vars(args)))
    print("saved", args.out, f"{time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
