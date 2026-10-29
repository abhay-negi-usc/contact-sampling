#!/usr/bin/env python3
"""Evaluate a trained hook/eyelet assembly policy (train_hook_rl.py).

Runs --episodes episodes from the task's start distribution (hook_rl_env.py; --level picks a curriculum level
instead), or the fixed cases of --eval-set, with the policy's mean action, in --workers processes. The task, the
error of the believed eyelet pose and the success criterion default to those the policy was trained with
(--task, --goal-error, --success override them). An evaluation set (make with --make-eval-set N) holds N starts
in the believed eyelet frame, each with an error of that belief, valid both with the error and without it, so
that the same starts test a policy with and without uncertainty (--no-error zeroes the errors).

An episode succeeds when the hook holds the assembled region (as in training), or, for a policy that declares
completion itself (train_hook_rl.py --stop-action), when it declares done while seated (the time limit without a
declaration fails, and a declaration elsewhere is a false completion); every success is then checked
physically by the pull test: the hook is let go and pulled up for a second, and passes if the loop's top wire
catches it in the crook: it stays threaded and its origin, the inside of the crook, ends within PULL_POS_MM of
its assembled position. (The hook hangs a few degrees tilted, which moves the top of its 65 mm shaft by several
mm, so its whole pose is not compared.)
Two checks that the simulation was not cheated: the deepest hook/eyelet contact penetration of each episode, and
its largest one-step change in linking number, which moves continuously unless a wire passes through the other.
Failures are sorted by where they ended: out of bounds, threaded the wrong way (linking < -0.5), threaded but not
seated (linking > 0.5), near the loop but unthreaded (mean keypoint offset < 30 mm), or far.

Writes to --out-dir: episodes.csv (one row per episode), summary.json, and for the first --save-trajectories
episodes the hook's poses as trajectory CSVs (assets/hook convention; the believed eyelet's pose at each step,
in the same frame, in belief_ep<N>.csv), which
assembly_trajectory_visual_verification.py renders (--render renders the first --render successes and failures).

    conda run -n mujoco_env python scripts/eval_hook_rl.py RUN_DIR/model_last.pt --out-dir RUN_DIR/eval
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import csv
import json
import multiprocessing as mp
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

import hook_rl_env as E
from train_hook_rl import ActorCritic, parse_goal_error

PULL_POS_MM = 3.0
SCRIPTS = Path(__file__).resolve().parent
UNSET = object()  # --goal-error not given: the policy's own setting


def load_policy(path):
    """(policy, environment settings it was trained with). Older policies read fewer observation values (the
    first obs_dim; later values were appended)."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    obs_dim = ck["model"]["norm.mean"].shape[0]
    ac = ActorCritic(obs_dim, ck["model"]["log_std"].shape[0], actor_dim=ck["model"]["actor.0.weight"].shape[1],
                     critic_dim=ck.get("critic_dim"))
    ac.load_state_dict(ck["model"])
    ac.eval()
    return ac, ck.get("env", {})


def make_eval_set(path, n, seed, bounds=E.GOAL_ERROR_BOUNDS):
    """n nominal starts (believed eyelet frame) with belief errors, each valid with and without its error."""
    env = E.HookAssemblyBatch(1, task="nominal")
    rng = np.random.default_rng(seed)
    pos, quat, err = [], [], []
    while len(pos) < n:
        pert = rng.uniform(-E.NOMINAL_PERTURB, E.NOMINAL_PERTURB)
        p = env.nominal_pos + pert[:3]
        q = (Rotation.from_rotvec(np.deg2rad(pert[3:])) * env.nominal_rot).as_quat(scalar_first=True)
        e = rng.uniform(-bounds, bounds)
        ok = True
        for case_err in (e, np.zeros(6)):
            env.reset_to(0, p, q, case_err)
            tp, tq, *_ = env.poses()
            ok &= env.start_ok(0, tp[0], E.quat_to_mat(tq)[0])
            ok &= abs(env.geo.features(tp, E.quat_to_mat(tq))[1][0]) <= E.UNTHREADED_LINK
        if ok:
            pos.append(p), quat.append(q), err.append(e)
    np.savez(path, pos=np.array(pos), quat=np.array(quat), err=np.array(err), seed=seed, bounds=bounds)


_ESTIMATOR = None  # built once before the workers fork, which then share its map


def run_batch(job):
    """Episodes of one worker: list of per-episode dicts, plus pose sequences for those with keep=True."""
    ckpt, n, seed, first_id, keep, env_kwargs, level, cases, online = job
    torch.set_num_threads(1)
    ac, _ = load_policy(ckpt)
    obs_dim = ac.norm.mean.shape[0]
    est_kwargs = {}
    if online["estimator"]:
        est_kwargs = dict(estimator=_ESTIMATOR, record_every=online["record_every"], **online["schedule"])
    elif online["explore_steps"]:
        est_kwargs = dict(record_every=online["record_every"])
    explorer, explore_until = None, 0  # exploring until then: one estimate, then (retrace) the assembly policy
    if online["explorer"]:  # paired: the exploration policy first, then one estimate, then the assembly policy
        explorer, _ = load_policy(online["explorer"])
        explore_until = online["explore_policy_steps"]
        est_kwargs["obs_layout"] = "explore"
    elif online["retrace"]:  # the assembly policy's noisy steps are the exploration
        explore_until = online["explore_steps"]
    handover = explore_until * (2 if online["retrace"] else 1)  # retracing: back along the path explored
    env = E.HookAssemblyBatch(n, seed=seed, level=level, auto_reset=False, **env_kwargs, **est_kwargs)
    env.est_online = explorer is None
    noise_rng = np.random.default_rng(seed + 99)
    max_steps = env.max_steps
    obs = env.reset(level=level)
    if cases is not None:
        obs = np.stack([env.reset_to(i, *(c[i] for c in cases))[i] for i in range(n)])
    starts = env.poses()
    start_link = env.geo.features(starts[0], E.quat_to_mat(starts[1]))[1]
    active = np.ones(n, bool)
    records = [None] * n
    pen = np.zeros(n)  # deepest penetration so far, mm
    jump = np.zeros(n)  # largest one-step change of linking number
    link_prev = start_link.copy()
    traj = [[np.concatenate([starts[0][i] * 1e-3, starts[1][i]])] for i in range(n)]
    belief = [[believed_pose(env, i)] for i in range(n)]
    path = [(starts[0], E.quat_to_mat(starts[1]))]  # the hook's poses while exploring, from the start
    for step in range(max_steps):
        if explore_until and step == explore_until:
            for i in np.flatnonzero(active):
                env.estimate_now(i)
        if explore_until and step == handover:
            env.obs_layout, env.est_online = E.layout_of(obs_dim), True
            obs = env.observe()[0]
        if explore_until <= step < handover:
            a = retrace_action(env, path[2 * explore_until - 1 - step])
        else:
            pol = explorer if explorer is not None and step < explore_until else ac
            with torch.no_grad():
                a = pol.mean(pol.norm(torch.as_tensor(obs[:, :pol.norm.mean.shape[0]]))).numpy()
        if step < online["explore_steps"]:  # observation collection: noisy motion early in the episode
            a[:, :6] += online["explore_noise"] * noise_rng.normal(size=(n, 6))
            if online["no_declare_exploring"] and a.shape[1] == 7:
                a[:, 6] = -1.0
        if a.shape[1] == 6 and env.stop_action:  # exploring or retracing declares nothing
            a = np.concatenate([a, -np.ones((n, 1))], axis=1)
        obs, rew, done, timeout, info = env.step(a)
        pos, quat, *_ = env.poses()
        if step < explore_until:
            path.append((pos, E.quat_to_mat(quat)))
        link_now = env.geo.features(pos, E.quat_to_mat(quat))[1]
        for i in np.flatnonzero(active):
            d = env.d[i]
            depth = [-d.contact[k].dist for k in range(d.ncon)
                     if env.hook_body in (env.m.geom_bodyid[d.contact[k].geom1], env.m.geom_bodyid[d.contact[k].geom2])]
            pen[i] = max(pen[i], max(depth, default=0.0) * 1e3)
            jump[i] = max(jump[i], abs(link_now[i] - link_prev[i]))
        link_prev = link_now
        for i in np.flatnonzero(active):
            if keep[i]:
                traj[i].append(np.concatenate([pos[i] * 1e-3, quat[i]]))
                belief[i].append(believed_pose(env, i))
        if not info:
            continue
        for k, i in enumerate(info["index"]):
            if not active[i]:
                continue
            active[i] = False
            err = info["goal_err"][k]
            rec = {"episode": first_id + i, "success": bool(info["success"][k]), "out": bool(info["out"][k]),
                   "length": int(info["length"][k]), "return": float(info["return"][k]),
                   "final_kp_mm": float(info["final_kp_mm"][k]), "final_link": float(info["final_link"][k]),
                   "final_seat_mm": float(info["final_seat_mm"][k]),
                   "start_x": starts[0][i, 0], "start_y": starts[0][i, 1], "start_z": starts[0][i, 2],
                   "start_angle_deg": float(np.degrees(2 * np.arccos(np.clip(abs(starts[1][i, 0]), 0, 1)))),
                   "start_link": float(start_link[i]), "max_penetration_mm": float(pen[i]),
                   "max_link_jump": float(jump[i]), "pull_ok": None,
                   **{f"err_{d}": float(v) for d, v in zip(("x", "y", "z", "rx", "ry", "rz"), err)},
                   **{f"err0_{d}": float(v) for d, v in zip(("x", "y", "z", "rx", "ry", "rz"), info["goal_err0"][k])},
                   "est_count": int(info["est_count"][k]), "est_time_s": float(info["est_time_s"][k]),
                   "contact_samples": len(env.rec_pose[i]),
                   "stopped": bool(info["stopped"][k]) if env.stop_action else None, "seated": bool(info["seated"][k])}
            if rec["success"]:
                p, q = E.pull_test(env.m, env.d[i], env.hook_body)
                link = env.geo.features(p[None] * 1e3, E.quat_to_mat(q[None]))[1][0]
                rec["pull_pos_mm"] = float(np.linalg.norm(p) * 1e3)
                rec["pull_tilt_deg"] = float(np.degrees(2 * np.arccos(np.clip(abs(q[0]), 0, 1))))
                rec["pull_ok"] = bool(rec["pull_pos_mm"] < PULL_POS_MM and link > E.SUCCESS_LINK)
            rec["outcome"] = outcome(rec)
            records[i] = rec
        if not active.any():
            break
    return records, {i: (np.array(traj[i]), np.array(belief[i])) for i in range(n) if keep[i]}


def believed_pose(env, i):
    """The believed eyelet pose of episode i in the true eyelet frame: position m, quaternion wxyz."""
    return np.concatenate([env.bel_pos[i] * 1e-3, Rotation.from_matrix(env.bel_rot[i]).as_quat(scalar_first=True)])


def retrace_action(env, target):
    """Actions (n, 6) moving every hook toward its pose `target` (positions mm, rotation matrices; the robot's own
    earlier poses), as far as one step allows: in the believed eyelet frame, as the policies act."""
    pos, quat, *_ = env.poses()
    rot = E.quat_to_mat(quat)
    d_pos = np.einsum("nji,nj->ni", env.bel_rot, target[0] - pos)
    turn = Rotation.from_matrix(np.einsum("nij,nkj->nik", target[1], rot)).as_rotvec()
    d_rot = np.degrees(np.einsum("nji,nj->ni", env.bel_rot, turn))
    return np.clip(np.concatenate([d_pos, d_rot], axis=1) / env.max_delta, -1, 1)


def outcome(rec):
    """Where an episode ended; for a policy that declares completion, also whether it declared."""
    if rec["success"]:
        return "assembled" if rec["pull_ok"] else "assembled, fails pull test"
    if rec["out"]:
        return "out of bounds"
    if rec["stopped"] is not None and rec["seated"]:
        where = "seated"
    elif rec["final_link"] < -0.5:
        where = "threaded the wrong way"
    elif rec["final_link"] > 0.5:
        where = "threaded, not seated"
    else:
        where = "unthreaded, near the loop" if rec["final_kp_mm"] < 30 else "unthreaded, far"
    if rec["stopped"] is None:
        return where
    return ("declared done: " if rec["stopped"] else "never declared: ") + where


def save_trajectory(path, poses, dt):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
        for k, p in enumerate(poses):
            w.writerow([k * dt, *p[:3], *p[4:7], p[3]])


def belief_mae(records, prefix):
    """Mean absolute belief error per axis (x, y mm; rx, ry, rz deg; z is known)."""
    return {a: round(float(np.mean([abs(r[prefix + a]) for r in records])), 3) for a in ("x", "y", "rx", "ry", "rz")}


def wilson(k, n, z=1.96):
    """95% Wilson score interval of a success rate."""
    if n == 0:
        return [float("nan")] * 2
    p, d = k / n, 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 4), round(c + h, 4)]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--max-steps", type=int, help="default: the episode length the policy was trained with")
    parser.add_argument("--task", choices=["wide", "nominal"])
    parser.add_argument("--goal-error", type=parse_goal_error, default=UNSET)
    parser.add_argument("--success", choices=["pose", "seat"])
    parser.add_argument("--eval-set", type=Path, help="fixed cases (npz); overrides --episodes and --level")
    parser.add_argument("--make-eval-set", type=int, help="first write an evaluation set of this many cases")
    parser.add_argument("--eval-set-seed", type=int, default=777)
    parser.add_argument("--no-error", action="store_true", help="zero the evaluation set's belief errors")
    parser.add_argument("--save-trajectories", type=int, default=40)
    parser.add_argument("--render", type=int, default=0, help="render this many successes and failures")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--level", type=int, help="start level (default: the task)")
    parser.add_argument("--estimator", type=json.loads, help='online estimation, e.g. \'{"kind": "icp_sdf", '
                        '"wrench_weight": 10}\' (contact_estimators.ContactEstimator)')
    parser.add_argument("--map", type=Path, help="contact map of the estimator (hook_contact_map.py)")
    parser.add_argument("--record-every", type=int, default=1, help="physics steps between contact samples")
    parser.add_argument("--est-every", type=int, default=5)
    parser.add_argument("--est-min-samples", type=int, default=200)
    parser.add_argument("--est-new-samples", type=int, default=100)
    parser.add_argument("--explore-noise", type=float, default=0.0, help="action noise std while exploring")
    parser.add_argument("--explore-steps", type=int, default=0, help="policy steps of exploration noise")
    parser.add_argument("--explorer", type=Path, help="exploration policy (train_hook_rl.py --mode explore) run "
                        "first, for --explore-policy-steps, before one estimate and the assembly policy")
    parser.add_argument("--explore-policy-steps", type=int, default=40)
    parser.add_argument("--retrace", action="store_true", help="after exploring (--explorer, or the "
                        "--explore-steps of noise) and one estimate, retrace the explored path back to the start "
                        "before the assembly policy takes over")
    parser.add_argument("--no-declare-exploring", action="store_true",
                        help="no declaration of completion during the --explore-steps of exploration noise")
    parser.add_argument("--wrench-bias", type=float, nargs="+", metavar="N_NM",
                        help="wrist sensor offset bound per axis: force and torque, or six values (default: as "
                        "trained)")
    parser.add_argument("--wrench-scale", type=float,
                        help="wrist sensor gain error bound per axis (default: as trained)")
    parser.add_argument("--wrench-noise", type=float, nargs="+", metavar="N_NM",
                        help="wrist sensor noise std, force and torque or six values, every episode (default: as "
                        "trained)")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    _, env_kwargs = load_policy(args.checkpoint)
    for key in ("task", "success", "max_steps"):
        if getattr(args, key) is not None:
            env_kwargs[key] = getattr(args, key)
    if args.goal_error is not UNSET:
        env_kwargs["goal_error"] = args.goal_error
    if args.wrench_noise is not None:
        env_kwargs.update(wrench_noise=args.wrench_noise, wrench_noise_random=False)
    if args.wrench_bias is not None:
        env_kwargs["wrench_bias"] = args.wrench_bias
    if args.wrench_scale is not None:
        env_kwargs["wrench_scale"] = args.wrench_scale
    env_kwargs.setdefault("task", "wide")
    level = E.n_levels(env_kwargs["task"]) - 1 if args.level is None else args.level

    cases = None
    if args.make_eval_set:
        make_eval_set(args.eval_set, args.make_eval_set, args.eval_set_seed)
    if args.eval_set:
        z = np.load(args.eval_set)
        cases = [z["pos"], z["quat"], np.zeros_like(z["err"]) if args.no_error else z["err"]]
        args.episodes = len(z["pos"])
        env_kwargs["task"] = "nominal"
    online = {"estimator": args.estimator, "map": str(args.map) if args.map else None,
              "record_every": args.record_every, "explore_noise": args.explore_noise,
              "explore_steps": args.explore_steps,
              "explorer": str(args.explorer) if args.explorer else None,
              "explore_policy_steps": args.explore_policy_steps, "retrace": args.retrace,
              "no_declare_exploring": args.no_declare_exploring,
              "schedule": {"est_every": args.est_every, "est_min_samples": args.est_min_samples,
                           "est_new_samples": args.est_new_samples}}
    if args.estimator:
        from contact_estimators import ContactEstimator
        global _ESTIMATOR
        cfg = dict(args.estimator)
        _ESTIMATOR = ContactEstimator(cfg.pop("kind"), args.map, **cfg)
    per = int(np.ceil(args.episodes / args.workers))
    jobs = []
    for w in range(args.workers):
        sl = slice(w * per, min((w + 1) * per, args.episodes))
        n = sl.stop - sl.start
        if n <= 0:
            break
        keep = np.zeros(n, bool)
        keep[: max(0, min(n, args.save_trajectories - w * per))] = True
        jobs.append((str(args.checkpoint), n, args.seed + w, w * per, keep, env_kwargs, level,
                     None if cases is None else [c[sl] for c in cases], online))
    with mp.get_context("fork").Pool(len(jobs)) as pool:
        results = pool.map(run_batch, jobs)
    records = [r for rs, _ in results for r in rs][: args.episodes]
    trajs, beliefs = {}, {}
    for (rs, tr) in results:
        for i, (poses, bel) in tr.items():
            trajs[rs[i]["episode"]], beliefs[rs[i]["episode"]] = poses, bel

    keys = list(records[0].keys() | {"pull_pos_mm", "pull_tilt_deg"})
    with open(args.out_dir / "episodes.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted(keys))
        w.writeheader()
        w.writerows(records)
    outcomes = {}
    for r in records:
        outcomes[r["outcome"]] = outcomes.get(r["outcome"], 0) + 1
    succ = [r for r in records if r["success"]]
    n_ok = sum(bool(r["pull_ok"]) for r in records)
    summary = {"checkpoint": str(args.checkpoint), "env": {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                                                          for k, v in env_kwargs.items()},
               "eval_set": str(args.eval_set) if args.eval_set else None, "no_error": args.no_error,
               "online": online,
               "level": level, "episodes": len(records),
               "success": float(np.mean([r["success"] for r in records])),
               "assembled_and_pull_ok": n_ok / len(records), "assembled_and_pull_ok_ci95": wilson(n_ok, len(records)),
               "outcomes": dict(sorted(outcomes.items(), key=lambda kv: -kv[1])),
               "max_penetration_mm": np.percentile([r["max_penetration_mm"] for r in records], [50, 99, 100]).round(3).tolist(),
               "max_link_jump": np.percentile([r["max_link_jump"] for r in records], [50, 99, 100]).round(3).tolist(),
               "pull_tilt_deg": np.percentile([r["pull_tilt_deg"] for r in succ], [50, 99]).round(1).tolist()
               if succ else None,
               "time_to_success_s": (np.percentile([r["length"] for r in succ], [10, 50, 90]) * E.SUBSTEPS
                                     * E.TIMESTEP).round(2).tolist() if succ else None,
               "belief_err_mae_initial": belief_mae(records, "err0_"), "belief_err_mae_final": belief_mae(records, "err_"),
               "estimates_per_episode_mean": float(np.mean([r["est_count"] for r in records])),
               "declared_done": float(np.mean([bool(r["stopped"]) for r in records])),
               "false_done": float(np.mean([bool(r["stopped"]) and not r["success"] for r in records])),
               "estimate_time_s_per_call_median": float(np.median([r["est_time_s"] / r["est_count"] for r in records
                                                                    if r["est_count"]] or [float("nan")]))}
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    tdir = args.out_dir / "trajectories"
    tdir.mkdir(exist_ok=True)
    dt = E.SUBSTEPS * E.TIMESTEP
    by_id = {r["episode"]: r for r in records}
    for ep, poses in trajs.items():
        save_trajectory(tdir / f"ep{ep}_{by_id[ep]['outcome'].replace(' ', '_').replace(',', '').replace(':', '')}.csv", poses, dt)
        save_trajectory(tdir / f"belief_ep{ep}.csv", beliefs[ep], dt)
    if args.render:
        chosen = ([e for e in trajs if by_id[e]["pull_ok"]][: args.render]
                  + [e for e in trajs if not by_id[e]["pull_ok"]][: args.render])
        for ep in chosen:
            path = next(tdir.glob(f"ep{ep}_*.csv"))
            subprocess.run([sys.executable, str(SCRIPTS / "assembly_trajectory_visual_verification.py"), "--csv",
                            str(path), "--out", str(path.with_suffix(".mp4")), "--fps", str(round(1 / dt)),
                            "--tile-width", "480", "--tile-height", "360"], check=True)


if __name__ == "__main__":
    main()
