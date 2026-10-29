#!/usr/bin/env python3
"""Train a PPO policy for the hook/eyelet assembly task of hook_rl_env.py (arbitrary start pose, full knowledge of
the hook's pose relative to the eyelet).

Simulation runs on the CPU in --workers processes of --envs-per-worker episodes each; the policy trains on the GPU.
PPO (clipped objective, GAE, adaptive learning rate from the KL divergence, running observation normalisation);
episodes cut by the time limit are bootstrapped from the value of their last observation. The policy sees only the
hook's pose relative to the eyelet; the critic, by default, also the environment's privileged values
(--critic-obs). Episode length (--max-steps) and the action's per-dimension scale (--max-delta) are set here, as
are the task (--task: the wide start box, or the nominal pre-assembly start), the error of the believed eyelet
pose (--goal-error) and what counts as assembled (--success).

Curriculum: starts begin close to assembled (level 0, distance to go under 4 mm: see hook_rl_env.py) and move out
a level each time the success rate over the last --window episodes started at the current level reaches
--promote-at, up to the task itself (the last level). Throughout,
a share --task-fraction of episodes starts from the task, to track it, and a share --review from a level below
the current one, so the closer starts are not forgotten.

Per iteration a line of log.jsonl in --out-dir (curriculum level, success rate over the episodes that ended at
the current level and from the task, returns, lengths, losses, ...); checkpoints model_<iter>.pt and
model_last.pt, which also hold the curriculum level. The same rows go to Weights & Biases (--wandb-*; the run's
files in --out-dir/wandb, its id in --out-dir/wandb_run_id so that --resume continues it), plotted against
environment steps. Every --video-every iterations the policy runs the fixed validation case of hook_rl_video.py
with its mean action: a low-resolution video to wandb and --out-dir/videos, and its outcome as val_* values.

    conda run -n mujoco_env python scripts/train_hook_rl.py --out-dir /media/rp/Elements1/abhay_ws/contact-sampling/hook_rl/run_x
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")  # the workers' small numpy ops; set before numpy loads
os.environ.setdefault("MUJOCO_GL", "egl")  # validation videos render off screen; set before mujoco loads

import argparse
import json
import multiprocessing as mp
import time
import uuid
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import wandb

import hook_rl_env as E
import hook_rl_video

torch.distributions.Distribution.set_default_validate_args(False)


STOP_INIT_MEAN, STOP_INIT_STD = -1.0, 0.4  # a new declaration action: P(declare) ~0.6% per step at first


class VecEnv:
    def __init__(self, workers, envs_per_worker, seed, env_kwargs):
        ctx = mp.get_context("fork")  # before any CUDA work in this process
        self.conns, self.procs = [], []
        for w in range(workers):
            parent, child = ctx.Pipe()
            p = ctx.Process(target=E.worker, args=(child, envs_per_worker, seed + 1000 * w, env_kwargs), daemon=True)
            p.start()
            self.conns.append(parent)
            self.procs.append(p)
        self.per = envs_per_worker
        self.n = workers * envs_per_worker

    def reset(self):
        for c in self.conns:
            c.send(("reset", None))
        return np.concatenate([c.recv() for c in self.conns])

    def step(self, actions):
        for k, c in enumerate(self.conns):
            c.send(("step", actions[k * self.per:(k + 1) * self.per]))
        obs, rew, done, timeout, infos = [], [], [], [], []
        for k, c in enumerate(self.conns):
            o, r, d, t, info = c.recv()
            obs.append(o), rew.append(r), done.append(d), timeout.append(t)
            if info:
                info["index"] = info["index"] + k * self.per
                infos.append(info)
        info = {key: np.concatenate([i[key] for i in infos]) for key in infos[0]} if infos else {}
        return np.concatenate(obs), np.concatenate(rew), np.concatenate(done), np.concatenate(timeout), info

    def set(self, **attrs):
        for c in self.conns:
            c.send(("set", attrs))
        for c in self.conns:
            c.recv()

    def close(self):
        for c in self.conns:
            c.send(("close", None))
        for p in self.procs:
            p.join(timeout=5)


class Normalizer(nn.Module):
    """Running mean and variance of observations; normalised output clipped to +-clip."""

    def __init__(self, dim, clip=5.0):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.tensor(1e-4))
        self.clip = clip

    @torch.no_grad()
    def update(self, x):
        n = x.shape[0]
        mean, var = x.mean(0), x.var(0, unbiased=False)
        delta, total = mean - self.mean, self.count + n
        self.mean += delta * n / total
        self.var = (self.var * self.count + var * n + delta ** 2 * self.count * n / total) / total
        self.count = total

    def forward(self, x):
        return torch.clamp((x - self.mean) / torch.sqrt(self.var + 1e-8), -self.clip, self.clip)


def mlp(sizes, out):
    layers, last = [], sizes[0]
    for h in sizes[1:]:
        layers += [nn.Linear(last, h), nn.ELU()]
        last = h
    layers.append(nn.Linear(last, out))
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    """Gaussian policy on the first actor_dim observation values (the relative pose); critic on the first
    critic_dim (all of them, or the pose only)."""

    def __init__(self, obs_dim, act_dim, actor_dim=E.OBS_POSE_DIM, critic_dim=None, hidden=(256, 256, 128),
                 init_std=0.6):
        super().__init__()
        self.actor_dim, self.critic_dim = actor_dim, critic_dim or obs_dim
        self.norm = Normalizer(obs_dim)
        self.actor = mlp((self.actor_dim, *hidden), act_dim)
        self.critic = mlp((self.critic_dim, *hidden), 1)
        self.log_std = nn.Parameter(torch.full((act_dim,), float(np.log(init_std))))
        nn.init.orthogonal_(self.actor[-1].weight, 0.01)
        nn.init.zeros_(self.actor[-1].bias)

    def mean(self, obs_n):
        return self.actor(obs_n[:, :self.actor_dim])

    def dist(self, obs_n):
        return torch.distributions.Normal(self.mean(obs_n), self.log_std.exp().expand(obs_n.shape[0], -1))

    def value(self, obs_n):
        return self.critic(obs_n[:, :self.critic_dim]).squeeze(-1)

    @torch.no_grad()
    def act(self, obs_n):
        """(action, log probability, value) for normalised observations."""
        dist = self.dist(obs_n)
        a = dist.sample()
        return a, dist.log_prob(a).sum(-1), self.value(obs_n)


def init_policy(ac, state, layout):
    """Start ac (observation layout `layout`) from another policy's weights. With the same shapes all of them;
    otherwise (another layout: an exploration policy, or an assembly policy that declares completion, started
    from an assembly policy) matched by observation field (hook_rl_env.obs_fields): the first layers of actor and
    critic get the old weights of the inputs they share and zeros for the new ones, and the observation
    normalisation the old statistics of those inputs (the new ones are scaled to about 1 already); the critic
    is dropped if its inputs differ in kind (pose only against all). Further layers are copied, and outputs (a
    new seventh action, STOP_*) get the old values where they exist and zeros elsewhere."""
    own = ac.state_dict()
    if all(own[k].shape == v.shape for k, v in state.items()):
        ac.load_state_dict(state)
        return
    old_dim = state["norm.mean"].shape[0]
    src, dst, at = {}, {}, 0
    for name, size in E.obs_fields(E.layout_of(old_dim)):
        src[name], at = range(at, at + size), at + size
    at = 0
    for name, size in E.obs_fields(layout):
        dst[name], at = range(at, at + size), at + size
    pairs = [(i, j) for name in dst if name in src and src[name][-1] < old_dim for i, j in zip(src[name], dst[name])]
    old_cols, new_cols = (torch.tensor(c) for c in zip(*pairs))
    with torch.no_grad():
        for k, v in state.items():
            if k in ("norm.mean", "norm.var"):
                own[k][new_cols] = v[old_cols]
            elif k == "norm.count" or own[k].shape == v.shape:
                own[k].copy_(v)
            elif k in ("actor.0.weight", "critic.0.weight"):  # (out, in): by field
                if k == "critic.0.weight" and (v.shape[1] == state["actor.0.weight"].shape[1]) != (
                        own[k].shape[1] == own["actor.0.weight"].shape[1]):
                    continue
                own[k].zero_()
                keep = (old_cols < v.shape[1]) & (new_cols < own[k].shape[1])
                own[k][:, new_cols[keep]] = v[:, old_cols[keep]]
            else:  # outputs: the shared leading ones
                own[k].zero_()
                own[k][tuple(slice(0, n) for n in v.shape)] = v
    ac.load_state_dict(own)


def summarize(episodes, level, task_level):
    """Statistics of finished episodes (concatenated info dicts): overall, at the current level, from the task."""
    if not episodes:
        return {}
    ep = {k: np.concatenate([e[k] for e in episodes]) for k in episodes[0] if k not in ("index", "terminal_obs")}
    out = {"episodes": int(len(ep["success"])), "success": float(ep["success"].mean()),
           "out_of_bounds": float(ep["out"].mean()), "timeout": float(ep["timeout"].mean()),
           "ep_length": float(ep["length"].mean()), "ep_return": float(ep["return"].mean()),
           "final_dist_mm": float(np.median(ep["final_dist_mm"])), "final_link": float(np.median(ep["final_link"])),
           "belief_err_mm_start": float(np.linalg.norm(ep["goal_err0"][:, :2], axis=1).mean()),
           "belief_err_mm_end": float(np.linalg.norm(ep["goal_err"][:, :2], axis=1).mean()),
           "estimates_per_episode": float(ep["est_count"].mean()),
           "estimate_time_s": float(ep["est_time_s"].sum() / max(1, ep["est_count"].sum())),
           "explore_gain_mm": float(ep["explore_gain"].mean()), "contact_samples": float(ep["samples"].mean()),
           "declared_done": float(ep["stopped"].mean()), "false_done": float((ep["stopped"] & ~ep["seated"]).mean()),
           "seated_at_end": float(ep["seated"].mean())}
    for name, sel in (("level", ep["start_level"] == level), ("task", ep["start_level"] == task_level)):
        if sel.any():
            out[f"success_{name}"] = float(ep["success"][sel].mean())
            out[f"episodes_{name}"] = int(sel.sum())
            out[f"final_dist_mm_{name}"] = float(np.median(ep["final_dist_mm"][sel]))
            done = sel & ep["success"]
            out[f"ep_length_{name}"] = float(ep["length"][done].mean()) if done.any() else None
    out["success_by_level"] = {int(k): round(float(ep["success"][ep["start_level"] == k].mean()), 3)
                               for k in np.unique(ep["start_level"])}
    return out


def parse_goal_error(text):
    if text == "none":
        return None
    if text == "default":
        return E.GOAL_ERROR_BOUNDS.tolist()
    return [float(v) for v in text.split(",")]


DIMS = ("x", "y", "z", "rx", "ry", "rz", "stop")
WANDB_SECTIONS = {"ppo": ("lr", "policy_loss", "value_loss", "entropy", "kl", "clip_frac", "reward", "std"),
                  "perf": ("fps", "sim_s", "update_s", "hours"),
                  "curriculum": ("level", "level_window_success", "promoted_to", "success_by_level", "error_scale",
                   "error_window_success", "error_promoted_to", "sensor_scale", "sensor_window_success",
                   "sensor_promoted_to"),
                  "action": ("std_per_dim", "action_abs_mean", "action_saturated")}


def wandb_row(row):
    """A log row for wandb: keys grouped into sections (episode statistics by default, val_* as val), the
    per-dimension lists and the per-level dict split into one key each."""
    out = {}
    for k, v in row.items():
        if v is None:
            continue
        if k in ("iteration", "steps"):
            out[k] = v
            continue
        if k.startswith("val_"):
            section, k = "val", k[4:]
        else:
            section = next((s for s, keys in WANDB_SECTIONS.items() if k in keys), "episodes")
        if isinstance(v, list):
            out.update({f"{section}/{k}_{d}": x for d, x in zip(DIMS, v)})
        elif isinstance(v, dict):
            out.update({f"{section}/{k}_{key}": x for key, x in v.items()})
        else:
            out[f"{section}/{k}"] = v
    return out


def init_wandb(args, env_kwargs):
    if args.wandb_key_file:  # used for this run only; leaves any stored login (~/.netrc) alone
        key = args.wandb_key_file.read_text().strip()
        if not key:  # an empty key would silently fall back to the stored login
            raise SystemExit(f"no wandb API key in {args.wandb_key_file}")
        os.environ["WANDB_API_KEY"] = key
    id_file = args.out_dir / "wandb_run_id"
    run_id = id_file.read_text().strip() if id_file.exists() else uuid.uuid4().hex[:8]
    id_file.write_text(run_id)
    config = {**{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k != "wandb_key_file"},
              "env": env_kwargs}
    wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=args.wandb_name or args.out_dir.name,
               id=run_id, resume="allow", dir=args.out_dir, mode=args.wandb_mode, config=config)
    wandb.define_metric("steps")
    wandb.define_metric("*", step_metric="steps")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=22)
    parser.add_argument("--envs-per-worker", type=int, default=48)
    parser.add_argument("--rollout", type=int, default=32, help="policy steps per environment per iteration")
    parser.add_argument("--iterations", type=int, default=3000)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--minibatches", type=int, default=4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--desired-kl", type=float, default=0.01)
    parser.add_argument("--entropy", type=float, default=0.005)
    parser.add_argument("--value-coef", type=float, default=1.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--init-std", type=float, default=0.6)
    parser.add_argument("--promote-at", type=float, default=0.8, help="success rate that moves the curriculum on")
    parser.add_argument("--window", type=int, default=400, help="episodes at a level judged for promotion")
    parser.add_argument("--task-fraction", type=float, default=0.1)
    parser.add_argument("--review", type=float, default=0.2)
    parser.add_argument("--start-level", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=E.MAX_EPISODE_STEPS, help="policy steps per episode")
    parser.add_argument("--max-delta", type=lambda t: [float(v) for v in t.split(",")], default=list(E.MAX_DELTA),
                        help="action scale: x,y,z mm and rotation x,y,z deg per step")
    parser.add_argument("--critic-obs", choices=["full", "pose"], default="full")
    parser.add_argument("--task", choices=["wide", "nominal"], default="wide", help="start distribution (hook_rl_env.py)")
    parser.add_argument("--goal-error", type=parse_goal_error, default=None,
                        help="belief error bounds: 'none', 'default' (GOAL_ERROR_BOUNDS) or x,y,z mm,rx,ry,rz deg")
    parser.add_argument("--success", choices=["pose", "seat"], default="pose")
    parser.add_argument("--mode", choices=["assemble", "explore"], default="assemble",
                        help="explore: train an exploration policy for the estimator (hook_rl_env.py)")
    parser.add_argument("--stop-action", action="store_true", help="the policy declares completion itself and "
                        "is judged then (hook_rl_env.py, FALSE_STOP_PENALTY)")
    parser.add_argument("--error-levels", type=lambda t: [float(v) for v in t.split(",")],
                        help="curriculum over the belief error's size: scales of --goal-error, e.g. 0.33,0.5,0.67,1; "
                        "the next once the start curriculum is at the task and episodes at the current scale succeed "
                        "at --error-promote-at")
    parser.add_argument("--error-promote-at", type=float, default=0.7)
    parser.add_argument("--error-review", type=float, default=0.3,
                        help="share of episodes whose error scale is drawn below the current one")
    parser.add_argument("--sensor-levels", type=lambda t: [float(v) for v in t.split(",")],
                        help="curriculum over the wrist sensor errors' size (hook_rl_env.py sensor_scale): the next "
                        "once the start curriculum is at the task, the error curriculum at its last scale, and "
                        "episodes at the current scale succeed at --sensor-promote-at")
    parser.add_argument("--sensor-promote-at", type=float, default=0.8)
    parser.add_argument("--sensor-review", type=float, default=0.3,
                        help="share of episodes whose sensor error scale is drawn below the current one")
    parser.add_argument("--critic-warmup", type=int, default=0,
                        help="first iterations (of this run) that train the critic only, the actor held")
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", type=Path, help="checkpoint to continue from")
    parser.add_argument("--init-from", type=Path, help="start from this checkpoint's policy (weights only)")
    parser.add_argument("--estimator", type=json.loads, help='estimate the eyelet online in training, e.g. '
                        '\'{"kind": "icp", "wrench_weight": 10}\' (contact_estimators.ContactEstimator)')
    parser.add_argument("--map", type=Path, help="contact map of the estimator (hook_contact_map.py)")
    parser.add_argument("--record-every", type=int, default=1, help="physics steps between contact samples")
    parser.add_argument("--est-every", type=int, default=5)
    parser.add_argument("--est-min-samples", type=int, default=300)
    parser.add_argument("--est-new-samples", type=int, default=300)
    parser.add_argument("--wrench-noise", type=float, nargs="+", default=[0.0, 0.0], metavar="N_NM",
                        help="wrist sensor noise std: force and torque, or six values (hook_rl_env.py)")
    parser.add_argument("--wrench-noise-random", action="store_true",
                        help="each episode's noise std: --wrench-noise times a uniform draw in [0, 1]")
    parser.add_argument("--wrench-bias", type=float, nargs="+", default=[0.0, 0.0], metavar="N_NM",
                        help="wrist sensor offset bound per axis: force and torque, or six values (hook_rl_env.py)")
    parser.add_argument("--wrench-scale", type=float, default=0.0, help="wrist sensor gain error bound per axis")
    parser.add_argument("--wrench-obs", choices=["absolute", "unit"], default="absolute",
                        help="the policy's view of the wrench (hook_rl_env.py)")
    parser.add_argument("--video-every", type=int, default=50, help="iterations between validation videos (0: none)")
    parser.add_argument("--wandb-project", default="hook-eyelet-assembly")
    parser.add_argument("--wandb-entity", help="default: the logged-in account's default entity")
    parser.add_argument("--wandb-name", help="default: the name of --out-dir")
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online")
    parser.add_argument("--wandb-key-file", type=Path, help="file holding the wandb API key (default: stored login)")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "args.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2))

    level = args.start_level
    task_level = E.n_levels(args.task) - 1
    env_kwargs = {"max_steps": args.max_steps, "max_delta": args.max_delta, "task": args.task,
                  "goal_error": args.goal_error, "success": args.success, "mode": args.mode,
                  "stop_action": args.stop_action, "wrench_noise": args.wrench_noise,
                  "wrench_noise_random": args.wrench_noise_random, "wrench_bias": args.wrench_bias,
                  "wrench_scale": args.wrench_scale, "wrench_obs": args.wrench_obs}
    online = {}
    if args.estimator:  # built here once: the forked workers share its map
        from contact_estimators import ContactEstimator
        cfg = dict(args.estimator)
        online = dict(estimator=ContactEstimator(cfg.pop("kind"), args.map, **cfg), record_every=args.record_every,
                      est_every=args.est_every, est_min_samples=args.est_min_samples,
                      est_new_samples=args.est_new_samples)
    err_level = 0
    curriculum = {}
    if args.error_levels:
        curriculum = {"error_scale": args.error_levels[0], "error_review": args.error_review}
    sens_level = 0
    if args.sensor_levels:
        curriculum.update(sensor_scale=args.sensor_levels[0], sensor_review=args.sensor_review)
    env = VecEnv(args.workers, args.envs_per_worker, args.seed,
                 {"level": level, "task_fraction": args.task_fraction, "review": args.review, **env_kwargs, **online,
                  **curriculum})
    init_wandb(args, env_kwargs)  # after the workers fork, so they hold no wandb state
    if args.video_every:
        validation = {name: hook_rl_video.ValidationVideo({**env_kwargs, **online}, *case)
                      for name, case in hook_rl_video.cases(args.task).items()}
        (args.out_dir / "videos").mkdir(exist_ok=True)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    layout = "stop" if args.stop_action and args.mode == "assemble" else args.mode
    obs_dim, actor_dim = E.obs_dims(layout)
    act_dim = 7 if layout == "stop" else 6
    ac = ActorCritic(obs_dim, act_dim, actor_dim=actor_dim,
                     critic_dim=actor_dim if args.critic_obs == "pose" else obs_dim, init_std=args.init_std).to(dev)
    opt = torch.optim.Adam(ac.parameters(), lr=args.lr)
    lr, start_iter = args.lr, 0
    if args.resume:
        ck = torch.load(args.resume, map_location=dev, weights_only=False)
        ac.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        lr, start_iter = ck["lr"], ck["iteration"] + 1
        level = ck.get("level", level)
        env.set(level=level)
        if args.error_levels:
            err_level = ck.get("error_level", 0)
            env.set(error_scale=args.error_levels[err_level])
        if args.sensor_levels:
            sens_level = ck.get("sensor_level", 0)
            env.set(sensor_scale=args.sensor_levels[sens_level])
    else:
        new_stop = act_dim == 7
        if args.init_from:
            state = torch.load(args.init_from, map_location=dev, weights_only=False)["model"]
            init_policy(ac, state, layout)
            new_stop &= state["log_std"].shape[0] < 7
        if new_stop:  # the declaration starts unlikely (STOP_INIT_*), so early episodes are not cut short
            with torch.no_grad():
                ac.actor[-1].bias[6], ac.log_std[6] = STOP_INIT_MEAN, float(np.log(STOP_INIT_STD))

    N, T = env.n, args.rollout
    obs_buf = torch.zeros(T, N, obs_dim, device=dev)
    act_buf = torch.zeros(T, N, act_dim, device=dev)
    logp_buf, val_buf, rew_buf, done_buf = (torch.zeros(T, N, device=dev) for _ in range(4))
    obs = torch.as_tensor(env.reset(), device=dev)
    log = open(args.out_dir / "log.jsonl", "a")
    t_start = time.time()
    recent = deque(maxlen=args.window)  # outcomes of episodes started at the current level
    recent_err = deque(maxlen=args.window)  # ... at the task with the current error scale (not review)
    recent_sens = deque(maxlen=args.window)  # ... and the last error scale, with the current sensor scale

    for it in range(start_iter, args.iterations):
        t0 = time.time()
        episodes = []
        ac.eval()
        for t in range(T):
            ac.norm.update(obs)
            obs_n = ac.norm(obs)  # stored normalised, so the update sees what the policy acted on
            a, logp, v = ac.act(obs_n)
            o, r, d, timeout, info = env.step(a.cpu().numpy())
            r = torch.as_tensor(r, device=dev)
            if info:
                episodes.append(info)
                recent.extend(info["success"][info["start_level"] == level])
                if args.error_levels:
                    full = (info["start_level"] == task_level) & np.isclose(info["error_scale"],
                                                                           args.error_levels[err_level])
                    recent_err.extend(info["success"][full])
                if args.sensor_levels:
                    full = (info["start_level"] == task_level) & np.isclose(info["sensor_scale"],
                                                                            args.sensor_levels[sens_level])
                    if args.error_levels:
                        full &= np.isclose(info["error_scale"], args.error_levels[-1])
                    recent_sens.extend(info["success"][full])
                to = info["timeout"]
                if to.any():  # bootstrap episodes cut by the time limit from their last observation
                    idx = torch.as_tensor(info["index"][to], device=dev)
                    with torch.no_grad():
                        v_last = ac.value(ac.norm(torch.as_tensor(info["terminal_obs"][to], device=dev)))
                    r[idx] += args.gamma * v_last
            obs_buf[t], act_buf[t], logp_buf[t], val_buf[t] = obs_n, a, logp, v
            rew_buf[t], done_buf[t] = r, torch.as_tensor(d, device=dev, dtype=torch.float32)
            obs = torch.as_tensor(o, device=dev)
        t_sim = time.time() - t0

        with torch.no_grad():
            last_v = ac.value(ac.norm(obs))
            adv = torch.zeros_like(rew_buf)
            gae = torch.zeros(N, device=dev)
            for t in reversed(range(T)):
                nv = last_v if t == T - 1 else val_buf[t + 1]
                nonterm = 1.0 - done_buf[t]
                delta = rew_buf[t] + args.gamma * nv * nonterm - val_buf[t]
                gae = delta + args.gamma * args.lam * nonterm * gae
                adv[t] = gae
            ret = adv + val_buf

        ac.train()
        b_obs_n, b_act = obs_buf.reshape(-1, obs_dim), act_buf.reshape(-1, act_dim)
        warmup = it < start_iter + args.critic_warmup
        b_logp, b_val, b_adv, b_ret = (x.reshape(-1) for x in (logp_buf, val_buf, adv, ret))
        mb = b_obs_n.shape[0] // args.minibatches
        stats = {"policy_loss": [], "value_loss": [], "entropy": [], "kl": [], "clip_frac": []}
        for _ in range(args.epochs):
            perm = torch.randperm(b_obs_n.shape[0], device=dev)
            for k in range(args.minibatches):
                i = perm[k * mb:(k + 1) * mb]
                dist = ac.dist(b_obs_n[i])
                logp = dist.log_prob(b_act[i]).sum(-1)
                ratio = torch.exp(logp - b_logp[i])
                a_n = (b_adv[i] - b_adv[i].mean()) / (b_adv[i].std() + 1e-8)
                pl = torch.max(-a_n * ratio, -a_n * torch.clamp(ratio, 1 - args.clip, 1 + args.clip)).mean()
                v = ac.value(b_obs_n[i])
                v_clip = b_val[i] + torch.clamp(v - b_val[i], -args.clip, args.clip)
                vl = torch.max((v - b_ret[i]) ** 2, (v_clip - b_ret[i]) ** 2).mean()
                ent = dist.entropy().sum(-1).mean()
                loss = args.value_coef * vl if warmup else pl + args.value_coef * vl - args.entropy * ent
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(ac.parameters(), args.max_grad_norm)
                opt.step()
                with torch.no_grad():
                    kl = (b_logp[i] - logp).mean()  # approximate KL(old || new)
                    if warmup:
                        pass
                    elif kl > 2 * args.desired_kl:
                        lr = max(1e-6, lr / 1.5)
                    elif kl < 0.5 * args.desired_kl:
                        lr = min(1e-3, lr * 1.5)
                    for g in opt.param_groups:
                        g["lr"] = lr
                stats["policy_loss"].append(pl.item()), stats["value_loss"].append(vl.item())
                stats["entropy"].append(ent.item()), stats["kl"].append(kl.item())
                stats["clip_frac"].append(((ratio - 1).abs() > args.clip).float().mean().item())

        row = {"iteration": it, "steps": (it + 1) * N * T, "fps": round(N * T / (time.time() - t0)),
               "sim_s": round(t_sim, 2), "update_s": round(time.time() - t0 - t_sim, 2),
               "hours": round((time.time() - t_start) / 3600, 3), "lr": lr, "level": level,
               "level_window_success": float(np.mean(recent)) if recent else None,
               "reward": rew_buf.mean().item(), "std": ac.log_std.exp().mean().item(),
               "std_per_dim": ac.log_std.exp().tolist(),
               "action_abs_mean": act_buf.abs().mean((0, 1)).tolist(),
               "action_saturated": (act_buf.abs() >= 1).float().mean((0, 1)).tolist(),
               **{k: float(np.mean(v)) for k, v in stats.items()}, **summarize(episodes, level, task_level)}
        if level < task_level and len(recent) == args.window and np.mean(recent) >= args.promote_at:
            level += 1
            recent.clear()
            env.set(level=level)
            row["promoted_to"] = level
        if args.error_levels:
            row["error_scale"] = args.error_levels[err_level]
            row["error_window_success"] = float(np.mean(recent_err)) if recent_err else None
            if (level == task_level and err_level + 1 < len(args.error_levels) and len(recent_err) == args.window
                    and np.mean(recent_err) >= args.error_promote_at):
                err_level += 1
                recent_err.clear()
                env.set(error_scale=args.error_levels[err_level])
                row["error_promoted_to"] = args.error_levels[err_level]
        if args.sensor_levels:
            row["sensor_scale"] = args.sensor_levels[sens_level]
            row["sensor_window_success"] = float(np.mean(recent_sens)) if recent_sens else None
            if (level == task_level and (not args.error_levels or err_level + 1 == len(args.error_levels))
                    and sens_level + 1 < len(args.sensor_levels) and len(recent_sens) == args.window
                    and np.mean(recent_sens) >= args.sensor_promote_at):
                sens_level += 1
                recent_sens.clear()
                env.set(sensor_scale=args.sensor_levels[sens_level])
                row["sensor_promoted_to"] = args.sensor_levels[sens_level]
        media = {}
        if args.video_every and (it % args.video_every == 0 or it == args.iterations - 1):
            def policy(o):
                with torch.no_grad():
                    return ac.mean(ac.norm(torch.as_tensor(o, device=dev))).cpu().numpy()

            for name, case in validation.items():
                frames, result = case.run(policy, f"it {it} {name}")
                row.update({f"val_{name + '_' if name else ''}{k}": v for k, v in result.items()})
                path = args.out_dir / "videos" / f"validation{'_' + name if name else ''}_it{it:05d}.mp4"
                hook_rl_video.save(frames, path)
                outcome = ("assembled" if result["success"] else "out of bounds" if result["out"]
                           else "declared done, not seated" if result["stopped"] else "timed out")
                media[f"val/video{'_' + name if name else ''}"] = wandb.Video(str(path), format="mp4", caption=(
                    f"iteration {it}, {row['steps'] / 1e6:.1f}M steps, level {row['level']}: {outcome} "
                    f"after {result['steps']} steps"))
        log.write(json.dumps(row) + "\n")
        log.flush()
        wandb.log({**wandb_row(row), **media})
        nan = float("nan")
        window = row["level_window_success"]
        err_text = ""
        if args.error_levels:
            err_window = row["error_window_success"]
            err_text = (f"err x{row['error_scale']:.2f} ({nan if err_window is None else err_window:.2f}"
                        f"{', promoted' if 'error_promoted_to' in row else ''}) | ")
        if args.sensor_levels:
            sens_window = row["sensor_window_success"]
            err_text += (f"sensor x{row['sensor_scale']:.2f} ({nan if sens_window is None else sens_window:.2f}"
                         f"{', promoted' if 'sensor_promoted_to' in row else ''}) | ")
        print(f"it {it:5d} | {row['steps'] / 1e6:7.1f}M | fps {row['fps']:6d} | level {row['level']:2d} "
              f"(window {nan if window is None else window:.2f}"
              f"{', promoted' if 'promoted_to' in row else ''}) | task {row.get('success_task', nan):.2f} | "
              f"{err_text}len {row.get('ep_length', nan):5.1f} | dist {row.get('final_dist_mm', nan):5.1f} mm | "
              f"rew {row['reward']:+.3f} | std {row['std']:.3f} | kl {row['kl']:.4f} | lr {lr:.1e}", flush=True)
        if (it + 1) % args.save_every == 0 or it == args.iterations - 1:
            ck = {"model": ac.state_dict(), "optimizer": opt.state_dict(), "lr": lr, "iteration": it,
                  "level": level, "error_level": err_level, "sensor_level": sens_level, "env": env_kwargs,
                  "critic_dim": ac.critic_dim,
                  "actor_dim": ac.actor_dim,
                  "args": vars(args)}
            torch.save(ck, args.out_dir / f"model_{it + 1}.pt")
            torch.save(ck, args.out_dir / "model_last.pt")
    env.close()
    if args.video_every:
        for case in validation.values():
            case.close()
    wandb.finish()


if __name__ == "__main__":
    main()
