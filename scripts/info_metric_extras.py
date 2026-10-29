#!/usr/bin/env python3
"""Quantities at the modes of info_metric_bench.py records for the further information and uncertainty measures
(contact_information.py: observed information, sandwich covariance, profile likelihood), without searching again.

For each record the same samples are drawn again (info_metric_bench.sample_poses, same seed), and at each mode found
and at the truth mode: the samples' residuals and Jacobian (and the residuals at the truth itself); at the best
mode, the cost Hessian. For
the first --profile streams at steps --profile-steps, the profile likelihood of each axis about the best mode over
--grid (mm or deg). Writes --out (npz: one record per input record, keyed by stream, step and noise).

    python scripts/info_metric_extras.py --records R.npz --streams S.npz --out R_extras.npz
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import multiprocessing as mp
import time

import numpy as np
from scipy.spatial.transform import Rotation

import contact_information as CI
from info_metric_bench import sample_poses

_MODEL = None


def init():
    global _MODEL
    _MODEL = CI.ContactModel()


def undo(truth, err):
    """The pose g with CI.relative(truth, g) = err."""
    r_e = Rotation.from_rotvec(np.deg2rad(err[3:]))
    r_t = Rotation.from_rotvec(np.deg2rad(truth[3:]))
    return np.concatenate([truth[:3] - (r_t * r_e.inv()).apply(err[:3]), (r_t * r_e.inv()).as_rotvec(degrees=True)])


def run(job):
    rec, stream, sigma, bound, max_samples, profile, grid = job
    poses, idx, steps, _ = sample_poses(stream, rec["step"], rec["noise"], rec.get("bias", 0.0), max_samples,
                                        1000 * rec["stream"] + rec["step"])
    t0 = time.perf_counter()
    local = undo(rec["truth"], rec["local_err"])
    out = {k: rec[k] for k in ("stream", "step", "noise")}
    out["bias"] = rec.get("bias", 0.0)
    out["sample_steps"], out["local_pose"] = steps, local
    rJ = [_MODEL.jacobian(g, poses) for g in list(rec["mode_pose"]) + [local]]
    out["mode_r"] = [x[0] for x in rJ[:-1]]
    out["mode_J"] = [x[1] for x in rJ[:-1]]
    out["local_r"], out["local_J"] = rJ[-1]
    out["truth_r"] = _MODEL.separations(rec["truth"], poses)  # for the conformal consistent set
    out["mount_J"] = CI.mount_jacobian(_MODEL, rec["mode_pose"][0], poses)  # for the consider covariance
    out["hess_best"] = CI.cost_hessian(_MODEL, rec["mode_pose"][0], poses, sigma)
    if profile:
        out["profile"] = np.array([CI.profile_interval(_MODEL, poses, rec["mode_pose"][0], sigma, bound, a, grid)[1]
                                   for a in range(6)])
        out["profile_grid"] = np.asarray(grid)
    out["time_s"] = time.perf_counter() - t0
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--records", required=True)
    p.add_argument("--streams", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--profile", type=int, default=24, help="streams to profile")
    p.add_argument("--profile-steps", type=int, nargs="+", default=[40, 160])
    p.add_argument("--profile-noise", type=float, default=0.3)
    p.add_argument("--grid", type=float, nargs="+", default=[-8, -4, -2, -1, -0.5, 0.5, 1, 2, 4, 8])
    p.add_argument("--workers", type=int, default=20)
    args = p.parse_args()
    data = np.load(args.records, allow_pickle=True)
    streams = np.load(args.streams, allow_pickle=True)["streams"]
    sigma = dict(zip(np.atleast_1d(data["noise"]), np.atleast_1d(data["sigma"])))
    bound = float(data["bound"])
    max_samples = int(data["max_samples"]) if "max_samples" in data.files else 150
    jobs = []
    for rec in data["records"]:
        if "mode_cost" not in rec or not len(rec["mode_cost"]):
            continue
        prof = (rec["stream"] < args.profile and rec["step"] in args.profile_steps
                and np.isclose(rec["noise"], args.profile_noise))
        jobs.append((rec, streams[rec["stream"]], sigma[rec["noise"]], bound, max_samples, prof, args.grid))
    jobs.sort(key=lambda j: -(j[5] * 1000 + j[0]["step"]))
    t0 = time.time()
    with mp.get_context("fork").Pool(args.workers, initializer=init) as pool:
        outs = []
        for i, o in enumerate(pool.imap_unordered(run, jobs)):
            outs.append(o)
            if (i + 1) % 100 == 0:
                print(f"{i + 1}/{len(jobs)} done, {time.time() - t0:.0f} s", flush=True)
    np.savez(args.out, records=np.array(outs, dtype=object))
    print("saved", args.out, f"{time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
