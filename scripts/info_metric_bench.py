#!/usr/bin/env python3
"""Tests of the contact information metric (contact_information.py) on simulated contact streams with ground truth.

For each stream (estimator_bench.py streams: hook poses at contact in the robot's frame, the true eyelet pose) and
each policy step of --at, the samples so far (at most --max-samples, evenly thinned) are optionally perturbed by
hook-pose noise of std --noise (mm and deg, independent per sample, a robot's pose measurement error) and a
constant error of the hook's mounting of std --bias drawn once per stream (the flange-to-hook transform off, as
hardware's peg origin was: correlated across samples, and not absorbable into the eyelet pose), then:
  oracle metrics   the information metrics at the true eyelet pose;
  truth mode       contact_information.solve started at the truth: the posterior mode the data hold there, with
                   its metrics and error (an oracle reference: which mode is the truth's);
  modes            contact_information.find_modes: the distinct posterior modes a multi-start search from the
                   initial belief and --starts - 1 prior draws finds (what an online estimator could do), each
                   with its pose, cost, Laplace covariance and error.
Errors: true pose relative to the estimate, in the estimate's frame (x, y, z mm, rotation vector deg). sigma is the
contact residual std the noise implies (--sigma; default from the noise) and the prior the benchmark's uniform
+-bound (--bound). Writes --out (npz: one record per stream, step and noise level).

    python scripts/info_metric_bench.py --streams S.npz --out R.npz --noise 0 0.3 --at 10 20 40 80 160 300
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import multiprocessing as mp
import time

import numpy as np
from scipy.spatial.transform import Rotation

import contact_information as CI

_MODEL = None


def init():
    global _MODEL
    _MODEL = CI.ContactModel()


def noisy(poses, std, rng):
    """Hook poses (k, 7) with independent position (mm) and rotation (deg) noise of std."""
    if std <= 0:
        return poses
    pos = poses[:, :3] + rng.normal(0, std, (len(poses), 3))
    turn = Rotation.from_rotvec(np.deg2rad(rng.normal(0, std, (len(poses), 3))))
    quat = (turn * Rotation.from_quat(poses[:, 3:], scalar_first=True)).as_quat(scalar_first=True)
    return np.concatenate([pos, quat], axis=1)


def sample_poses(stream, step, noise, bias, max_samples, seed):
    """(poses used, their indices, their policy steps, rng after the noise draws) of a stream at a policy step;
    callers given the same seed draw the same noise, so their cases pair with the bench's."""
    k = int(stream["counts"][min(step, len(stream["counts"]) - 1)])
    idx = np.unique(np.linspace(0, k - 1, min(k, max_samples)).astype(int)) if k else np.zeros(0, int)
    rng = np.random.default_rng(seed)
    poses = stream["poses"][idx]
    if bias > 0:  # one mounting error for the whole stream
        poses = CI.mount_offset(poses, rng.normal(0, bias, 6))
    poses = noisy(poses, noise, rng)
    return poses, idx, np.searchsorted(stream["counts"], idx, side="right"), rng


def run(job):
    sid, stream, step, noise, sigma, bound, max_samples, starts, seed, bias = job
    s_prior = CI.prior_std(bound)
    k = int(stream["counts"][min(step, len(stream["counts"]) - 1)])
    rec = {"stream": sid, "step": step, "noise": noise, "bias": bias, "samples": k, "truth": stream["truth"]}
    if k < 3:
        return rec
    poses, idx, steps, rng = sample_poses(stream, step, noise, bias, max_samples, seed)
    rec["used"], rec["sample_steps"] = len(idx), steps
    t0 = time.perf_counter()
    r_true, J_true = _MODEL.jacobian(stream["truth"], poses)
    m = CI.information_metrics(J_true, sigma, s_prior)
    rec.update(oracle_lambda=m["lambda"], oracle_vectors=m["vectors"], oracle_std=m["post_std"],
               oracle_gain=m["gain_nats"], resid_true=np.sqrt(np.mean(r_true ** 2)))
    g, cost, J = CI.solve(_MODEL, poses, stream["truth"], sigma, bound)
    mm = CI.information_metrics(J, sigma, s_prior)
    rec.update(local_err=CI.relative(stream["truth"], g), local_std=mm["post_std"], local_lambda=mm["lambda"],
               local_vectors=mm["vectors"], local_gain=mm["gain_nats"], local_cost=cost)
    modes = CI.find_modes(_MODEL, poses, sigma, bound, rng, starts=starts)
    weights, covs = CI.laplace_mixture(modes, sigma, s_prior)
    rec.update(mode_pose=np.array([m[0] for m in modes]), mode_cost=np.array([m[1] for m in modes]),
               mode_err=np.array([CI.relative(stream["truth"], m[0]) for m in modes]), mode_cov=np.array(covs),
               mode_weight=weights, mode_gain=np.array([CI.information_metrics(m[2], sigma, s_prior)["gain_nats"]
                                                        for m in modes]))
    rec["time_s"] = time.perf_counter() - t0
    return rec


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--streams", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--at", type=int, nargs="+", default=[10, 20, 40, 80, 160, 300])
    p.add_argument("--noise", type=float, nargs="+", default=[0.0, 0.3], help="hook-pose noise std, mm and deg")
    p.add_argument("--sigma", type=float, nargs="+", help="contact residual std per noise level (mm)")
    p.add_argument("--bias", type=float, default=0.0, help="per-stream hook mounting error std, mm and deg")
    p.add_argument("--bound", type=float, default=20.0, help="belief error bound per axis (mm, deg)")
    p.add_argument("--cases", type=int)
    p.add_argument("--max-samples", type=int, default=150)
    p.add_argument("--starts", type=int, default=24)
    p.add_argument("--workers", type=int, default=20)
    args = p.parse_args()
    streams = np.load(args.streams, allow_pickle=True)["streams"][: args.cases]
    sigmas = args.sigma or [max(0.05, 1.1 * n) for n in args.noise]  # sim contacts read ~0.03 mm at the truth
    jobs = [(sid, s, step, n, sg, args.bound, args.max_samples, args.starts, 1000 * sid + step, args.bias)
            for sid, s in enumerate(streams) for step in args.at for n, sg in zip(args.noise, sigmas)]
    jobs.sort(key=lambda j: -j[2])  # long ones first
    t0 = time.time()
    with mp.get_context("fork").Pool(args.workers, initializer=init) as pool:
        recs = []
        for i, r in enumerate(pool.imap_unordered(run, jobs)):
            recs.append(r)
            if (i + 1) % 50 == 0:
                print(f"{i + 1}/{len(jobs)} done, {time.time() - t0:.0f} s", flush=True)
    np.savez(args.out, records=np.array(recs, dtype=object), noise=args.noise, sigma=sigmas, at=args.at,
             bound=args.bound, bias=args.bias, max_samples=args.max_samples)
    print("saved", args.out, f"{time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
