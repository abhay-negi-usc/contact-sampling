#!/usr/bin/env python3
"""Gaussian (Laplace) posteriors from pose_filter_bench.py records, as particles for uq_analysis.py.

  mixture   every distinct polished mode (modes_pose) with its Laplace covariance (J^T J + prior)^-1 at the mode,
            weighted exp(-cost / 2) sqrt(det cov) (the Laplace evidence): the multi-start Laplace mixture of
            info_metric_bench, now over the modes a global search found
  local     one Gaussian, the Laplace covariance at the point estimate: the local (Fisher) uncertainty alone
Draws --keep particles from each (in the chart about each mode: x = mode + L z) and writes records with the same
fields as the input and method "<method>-mixture" / "<method>-local".

    python scripts/uq_extras.py --records policy_multistart.npz --streams S.npz --out policy_multistart_laplace.npz
"""

import argparse

import numpy as np
import torch

import contact_gpu as G
import pose_filters as PF
from info_metric_bench import sample_poses


def laplace_cov(prob, x, bound):
    _, J = PF.jacobian(prob, torch.as_tensor(x, dtype=torch.float32, device=prob.dev))
    info = (J.transpose(1, 2) @ J).double().cpu().numpy()
    prior = np.eye(6) * 3.0 / bound ** 2
    return np.linalg.inv(info + prior)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--records", required=True)
    p.add_argument("--streams", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--keep", type=int, default=512)
    p.add_argument("--bound", type=float, default=20.0)
    p.add_argument("--max-samples", type=int, default=150)
    args = p.parse_args()
    recs = np.load(args.records, allow_pickle=True)["records"]
    streams = np.load(args.streams, allow_pickle=True)["streams"]
    model = G.GpuContactModel()
    rng = np.random.default_rng(0)
    out = []
    for r in recs:
        s = streams[r["stream"]]
        poses, *_ = sample_poses(s, r["step"], r["noise"], 0.0, args.max_samples, 1000 * r["stream"] + r["step"])
        prob = PF.Problem(model, poses, r["sigma"], args.bound)
        modes, costs = np.asarray(r["modes_pose"], np.float64), np.asarray(r["modes_cost"], np.float64)
        covs = laplace_cov(prob, modes, args.bound)
        logw = -0.5 * costs + 0.5 * np.linalg.slogdet(covs)[1]
        w = np.exp(logw - logw.max())
        w /= w.sum()
        for kind in ("mixture", "local"):
            if kind == "mixture":
                comp = rng.choice(len(modes), args.keep, p=w)
                mu, cv = modes[comp], covs[comp]
            else:
                best = int(np.argmin(costs))
                mu = np.repeat(modes[best:best + 1], args.keep, 0)
                cv = np.repeat(covs[best:best + 1], args.keep, 0)
            z = rng.normal(size=(args.keep, 6))
            x = mu + np.einsum("nab,nb->na", np.linalg.cholesky(cv), z)
            q = dict(r)
            q["method"] = f"{r['method']}-{kind}"
            q["particles"] = x.astype(np.float32)
            q["particles_ll"] = np.zeros(args.keep)
            q["particles_logw"] = np.zeros(args.keep)
            q["snapshots"] = {}
            q["mixture_weights"] = w if kind == "mixture" else np.ones(1)
            out.append(q)
    np.savez(args.out, records=np.array(out, dtype=object))
    print("saved", args.out, len(out))


if __name__ == "__main__":
    main()
