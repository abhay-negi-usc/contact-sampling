#!/usr/bin/env python3
"""Build a contact map of the hook/eyelet pair in simulation, for the contact-manifold pose estimators.

A contact map is the set of hook poses relative to the eyelet at which the two touch, each with the wrench the
contact puts on the hook: the hardware maps of sunrise-wrapper-abhay were recorded by exploring contact with the
eyelet fixed at a known pose. Here policies (--policies, train_hook_rl.py checkpoints) assemble the hook from
nominal starts (hook_rl_env.py) with a wrong belief of the eyelet pose, wider than the benchmark's (--goal-error),
and with Gaussian noise of a per-episode scale in [0, --noise] added to their actions, so that they touch the
eyelet in many ways: misaligned approaches, jams, threading and seating. Episodes run to their time limit
(seated contact included). Every --record-every physics steps of contact the hook's pose in the true eyelet frame
and its contact wrench in the hook frame (a wrist force-torque sensor) are recorded.

Writes --out (CSV in the estimators' map format: x, y, z mm, qx, qy, qz, qw, force_X..torque_Z in N and N m about
the hook's origin) and a summary next to it. Use seeds that the evaluation does not.

    conda run -n mujoco_env python scripts/hook_contact_map.py --policies P1.pt P2.pt --out MAP.csv
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import torch

import hook_rl_env as E
from eval_hook_rl import load_policy

COLUMNS = ["x", "y", "z", "qx", "qy", "qz", "qw", "force_X", "force_Y", "force_Z", "torque_X", "torque_Y", "torque_Z"]


def collect(job):
    """Samples of one worker: (k, 13) rows of COLUMNS."""
    policy_path, n, rounds, seed, goal_error, noise, record_every, max_steps = job
    torch.set_num_threads(1)
    ac, env_kwargs = load_policy(policy_path)
    obs_dim = ac.norm.mean.shape[0]
    env = E.HookAssemblyBatch(n, seed=seed, task="nominal", goal_error=goal_error, success="seat",
                              max_delta=env_kwargs.get("max_delta", E.MAX_DELTA), max_steps=max_steps,
                              auto_reset=False, end_on_success=False, record_every=record_every)
    rng = np.random.default_rng(seed + 1)
    rows = []
    for _ in range(rounds):
        obs = env.reset(level=env.task_level)
        scale = rng.uniform(0, noise, (n, 1))
        for _ in range(max_steps):
            with torch.no_grad():
                a = ac.mean(ac.norm(torch.as_tensor(obs[:, :obs_dim]))).numpy()
            obs, *_ = env.step(a + scale * rng.normal(size=a.shape))
        for i in range(n):
            if env.rec_true[i]:
                t = np.array(env.rec_true[i])
                w = np.array(env.rec_wrench[i])
                rows.append(np.concatenate([t[:, :3], t[:, 4:7], t[:, 3:4], w], axis=1))  # quat wxyz -> xyzw
    return np.concatenate(rows) if rows else np.zeros((0, 13))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--policies", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=21)
    parser.add_argument("--envs-per-worker", type=int, default=16)
    parser.add_argument("--rounds", type=int, default=10, help="episodes per environment")
    parser.add_argument("--goal-error", type=float, nargs=6, default=[8.0, 8.0, 0.0, 8.0, 8.0, 8.0])
    parser.add_argument("--noise", type=float, default=0.6, help="largest per-episode action noise std")
    parser.add_argument("--record-every", type=int, default=2, help="physics steps between samples")
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=50000)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    jobs = [(str(args.policies[w % len(args.policies)]), args.envs_per_worker, args.rounds, args.seed + 1000 * w,
             args.goal_error, args.noise, args.record_every, args.max_steps) for w in range(args.workers)]
    t0 = time.time()
    with mp.get_context("fork").Pool(args.workers) as pool:
        parts = pool.map(collect, jobs)
    rows = np.concatenate(parts)
    header = ",".join(COLUMNS)
    np.savetxt(args.out, rows, delimiter=",", header=header, comments="", fmt="%.6g")
    f = np.linalg.norm(rows[:, 7:10], axis=1)
    summary = {"samples": int(len(rows)), "episodes": args.workers * args.envs_per_worker * args.rounds,
               "seconds": round(time.time() - t0, 1), "args": {k: str(v) for k, v in vars(args).items()},
               "force_N_p10_50_90": np.percentile(f, [10, 50, 90]).round(3).tolist(),
               "position_mm_min": rows[:, :3].min(0).round(1).tolist(),
               "position_mm_max": rows[:, :3].max(0).round(1).tolist()}
    args.out.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
