#!/usr/bin/env python3
"""Score an exploration policy's contact streams with the batch SMC, as probe_design.py scores probe contacts.

Per case (insertion-policy stream sid at policy step): the insertion policy's samples (150, thinned, the same draws
as probe_design.py, so its "none" rows are this script's baseline) plus the explorer's (its whole episode, thinned
to --explore-samples), with the same hook-pose noise; batch SMC and probe_design.evaluate. "explorer-only" scores the
explorer's samples alone (once per stream and noise level). Explorer and insertion streams of the same cases
(estimator_bench.py streams with the same --seed: the same starts and belief errors) are paired by index, which
their truths check.

Also per stream: the share of the explorer's samples by eyelet region (probe_design.py's regions, from the hook
point closest to the true eyelet).

    python scripts/explorer_eval.py --streams POLICY.npz --explorer X.npz --label name --out R.npz
"""

import argparse
import time

import numpy as np
import torch

import contact_gpu as G
import pose_filters as PF
from info_metric_bench import noisy, sample_poses
from probe_design import evaluate, to_eyelet


def regions(model, poses, truth):
    """Eyelet region of each sample (k): its hook point closest to the eyelet at truth."""
    if not len(poses):
        return np.zeros(0, str)
    pts = model.hook_points(to_eyelet(poses, truth))
    d = model.sdf(pts) - model.radius
    q = pts[torch.arange(len(pts)), d.argmin(1)].cpu().numpy()
    return np.where(q[:, 2] < -15, "stem", np.where(np.linalg.norm(q[:, 1:] - [-0.28, -5.99], axis=1) < 7.5,
                                                    "ring inner", np.where(np.abs(q[:, 0]) > 2.3, "ring face",
                                                                           "ring outer")))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--streams", required=True, help="insertion-policy streams (estimator_bench.py)")
    p.add_argument("--explorer", required=True, help="the explorer's streams of the same cases")
    p.add_argument("--label", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--cases", type=int, default=32)
    p.add_argument("--at", type=int, nargs="+", default=[160, 300])
    p.add_argument("--noise", type=float, nargs="+", default=[0.0, 0.3])
    p.add_argument("--sigma", type=float, nargs="+", default=[0.05, 0.33])
    p.add_argument("--max-samples", type=int, default=150)
    p.add_argument("--explore-samples", type=int, default=400)
    args = p.parse_args()
    streams = np.load(args.streams, allow_pickle=True)["streams"]
    xstreams = np.load(args.explorer, allow_pickle=True)["streams"]
    model = G.GpuContactModel()
    recs = []
    for sid in range(args.cases):
        s, x, t0 = streams[sid], xstreams[sid], time.perf_counter()
        truth = np.asarray(s["truth"], np.float64)
        assert np.allclose(truth, x["truth"]), f"stream {sid}: the explorer's case differs"
        last = len(x["counts"]) - 1
        clean_x = sample_poses(x, last, 0.0, 0.0, args.explore_samples, 0)[0]
        reg = regions(model, clean_x, truth)
        for noise, sigma in zip(args.noise, args.sigma):
            xs = (sample_poses(x, last, noise, 0.0, args.explore_samples, 500000 + 1000 * sid)[0] if len(clean_x)
                  else clean_x)  # noisy() needs samples
            common = {"stream": sid, "noise": noise, "label": args.label, "n_explore": len(xs), "region": reg}
            if len(xs):
                rec = {**common, "step": 0, "rule": "explorer-only"}
                rec.update(evaluate(PF.Problem(model, xs, sigma, 20.0), truth, 1000 * sid + 7919 * int(round(noise * 10))))
                recs.append(rec)
            for step in args.at:
                poses = sample_poses(s, step, noise, 0.0, args.max_samples, 1000 * sid + step)[0]
                seed = 1000 * sid + step + 7919 * int(round(noise * 10))  # probe_design.py's
                rec = {**common, "step": step, "rule": "explorer"}
                rec.update(evaluate(PF.Problem(model, np.concatenate([poses, xs]), sigma, 20.0), truth, seed))
                recs.append(rec)
        print(f"stream {sid} done ({time.perf_counter() - t0:.0f} s, {len(clean_x)} explorer samples)", flush=True)
        np.savez(args.out, records=np.array(recs, dtype=object), args=str(vars(args)))


if __name__ == "__main__":
    main()
