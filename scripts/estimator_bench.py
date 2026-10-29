#!/usr/bin/env python3
"""Offline benchmark of the contact-manifold estimators (contact_estimators.py) on simulated contact streams.

streams: a policy (train_hook_rl.py checkpoint: an assembly or an exploration policy, its own observation layout;
a declaration of completion is ignored) runs --cases held-out nominal starts (a seed of their own, not the
evaluation set's) with its belief of the eyelet off by a uniform error (--goal-error) and Gaussian action
noise of std --noise, for --max-steps policy steps (not stopping at success), recording every --record-every
physics steps of contact the hook's pose in the robot's frame and the hook-frame contact wrench
(hook_rl_env.py), with the policy step of each sample. Writes --streams (npz).

estimate: every estimator configuration (--configs, JSON list of {"name", "kind", ...ContactEstimator kwargs})
estimates the eyelet's pose from the samples up to each policy step of --at, starting from the initial belief,
as the online loop would; error = true pose relative to the estimate's, in the estimated eyelet frame (x, y, z
mm; rotation vector deg). Writes --out (CSV, one row per stream, step and configuration) and prints per
configuration and step the mean absolute error per axis, the share of estimates within 1 mm and 1 deg on every
axis, and the median compute time.

    python scripts/estimator_bench.py streams --policy P.pt --streams S.npz
    python scripts/estimator_bench.py estimate --streams S.npz --map MAP.csv --configs '[...]' --out R.csv
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import csv
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

AXES = ("x", "y", "z", "rx", "ry", "rz")


def record_streams(job):
    import torch
    import hook_rl_env as E
    from eval_hook_rl import load_policy
    policy, n, seed, noise, record_every, max_steps, goal_error = job
    torch.set_num_threads(1)
    ac, env_kwargs = load_policy(policy)
    obs_dim = ac.norm.mean.shape[0]
    env = E.HookAssemblyBatch(n, seed=seed, task="nominal", goal_error=goal_error, success="seat",
                              max_delta=env_kwargs.get("max_delta", E.MAX_DELTA), max_steps=max_steps,
                              auto_reset=False, end_on_success=False, record_every=record_every,
                              obs_layout=E.layout_of(obs_dim))
    rng = np.random.default_rng(seed + 7)
    obs = env.reset(level=env.task_level)
    truth = env.err.copy()
    counts = np.zeros((max_steps + 1, n), np.int64)
    first_success = np.full(n, -1)
    for step in range(1, max_steps + 1):
        with torch.no_grad():
            a = ac.mean(ac.norm(torch.as_tensor(obs[:, :obs_dim]))).numpy()[:, :6]
        obs, *_ = env.step(a + noise * rng.normal(size=a.shape))
        counts[step] = [len(r) for r in env.rec_pose]
        first_success[(first_success < 0) & (env.hold >= E.HOLD_STEPS)] = step
    out = []
    for i in range(n):
        poses, wrenches = env.recorded(i)
        out.append({"poses": poses, "wrenches": wrenches, "truth": truth[i], "counts": counts[:, i],
                    "first_success": first_success[i]})
    return out


def pose_error(truth, pos_mm, quat):
    """True eyelet pose relative to the estimated one, in the estimated frame: (x, y, z mm, rotvec deg)."""
    r_true = Rotation.from_rotvec(np.deg2rad(truth[3:]))
    r_est = Rotation.from_quat(quat, scalar_first=True)
    return np.concatenate([r_est.inv().apply(truth[:3] - pos_mm), np.degrees((r_est.inv() * r_true).as_rotvec())])


def run_estimates(job):
    import gc
    from contact_estimators import ContactEstimator
    configs, map_path, streams, at, min_samples = job
    rows = []
    for cfg in configs:  # one estimator at a time: each holds its own copy of the map
        kw = {k: v for k, v in cfg.items() if k not in ("name", "kind")}
        est = ContactEstimator(cfg["kind"], map_path, **kw)
        for sid, s in streams:
            for step in at:
                k = int(s["counts"][min(step, len(s["counts"]) - 1)])
                if k < min_samples:
                    continue
                t0 = time.perf_counter()
                pos, quat = est.estimate(s["poses"][:k], s["wrenches"][:k], np.zeros(3), np.array([1.0, 0, 0, 0]))
                dt = time.perf_counter() - t0
                e = pose_error(s["truth"], pos, quat)
                rows.append({"config": cfg["name"], "stream": sid, "step": step, "samples": k, "time_s": dt,
                             **{f"err_{a}": v for a, v in zip(AXES, e)},
                             **{f"init_{a}": v for a, v in zip(AXES, s["truth"])}})
        del est
        gc.collect()
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("streams")
    s.add_argument("--policy", type=Path, required=True)
    s.add_argument("--streams", type=Path, required=True)
    s.add_argument("--cases", type=int, default=64)
    s.add_argument("--workers", type=int, default=8)
    s.add_argument("--noise", type=float, default=0.0)
    s.add_argument("--record-every", type=int, default=1)
    s.add_argument("--max-steps", type=int, default=120)
    s.add_argument("--seed", type=int, default=90000)
    s.add_argument("--goal-error", type=float, nargs=6, default=[5.0, 5.0, 0.0, 5.0, 5.0, 5.0],
                   help="belief error bounds: x, y, z mm, rotation x, y, z deg")
    e = sub.add_parser("estimate")
    e.add_argument("--streams", type=Path, required=True)
    e.add_argument("--map", type=Path, required=True)
    e.add_argument("--configs", type=json.loads, required=True)
    e.add_argument("--at", type=int, nargs="+", default=[10, 20, 40, 80, 120])
    e.add_argument("--min-samples", type=int, default=20)
    e.add_argument("--max-streams", type=int)
    e.add_argument("--workers", type=int, default=8)
    e.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if args.cmd == "streams":
        per = int(np.ceil(args.cases / args.workers))
        jobs = [(str(args.policy), per, args.seed + 100 * w, args.noise, args.record_every, args.max_steps,
                 args.goal_error)
                for w in range(args.workers)]
        with mp.get_context("fork").Pool(args.workers) as pool:
            streams = [x for part in pool.map(record_streams, jobs) for x in part][: args.cases]
        np.savez_compressed(args.streams, streams=np.array(streams, dtype=object), noise=args.noise)
        n = np.array([len(x["poses"]) for x in streams])
        print(f"{len(streams)} streams; samples per stream p10/50/90 {np.percentile(n, [10, 50, 90]).round()}")
        return

    streams = list(enumerate(np.load(args.streams, allow_pickle=True)["streams"]))[: args.max_streams]
    chunks = [streams[w::args.workers] for w in range(args.workers)]
    jobs = [(args.configs, str(args.map), c, args.at, args.min_samples) for c in chunks if c]
    with mp.get_context("fork").Pool(len(jobs)) as pool:
        rows = [r for part in pool.map(run_estimates, jobs) for r in part]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{'config':24s} {'step':>4s} {'n':>4s} {'samples':>8s}  MAE x/y/z mm    rx/ry/rz deg  init xyz rx/ry/rz  "
          f"xy<1mm ryz<2deg  time s")
    for cfg in args.configs:
        for step in args.at:
            sel = [r for r in rows if r["config"] == cfg["name"] and r["step"] == step]
            if not sel:
                continue
            err = np.abs([[r[f"err_{a}"] for a in AXES] for r in sel])
            init = np.abs([[r[f"init_{a}"] for a in AXES] for r in sel])
            xy = np.mean(np.all(err[:, :2] <= 1.0, axis=1))
            ryz = np.mean(np.all(err[:, 4:] <= 2.0, axis=1))
            print(f"{cfg['name']:24s} {step:4d} {len(sel):4d} {np.median([r['samples'] for r in sel]):8.0f}  "
                  f"{err[:, 0].mean():4.2f} {err[:, 1].mean():4.2f} {err[:, 2].mean():4.2f}  "
                  f"{err[:, 3].mean():4.2f} {err[:, 4].mean():4.2f} {err[:, 5].mean():4.2f}  "
                  f"{init[:, :3].mean():4.2f} {init[:, 3:].mean():4.2f}  "
                  f"{xy:6.2f} {ryz:6.2f}  {np.median([r['time_s'] for r in sel]):6.2f}")


if __name__ == "__main__":
    main()
