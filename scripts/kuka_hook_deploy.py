#!/usr/bin/env python3
"""Run a hook/eyelet assembly policy (train_hook_rl.py, declared completion) on a KUKA LBR iiwa 14 through Sunrise
(sunrise-wrapper-abhay's kuka_sunrise), or on kuka_hook_sim.py's simulated iiwa through the same loop.

Frames (4x4, metres): base (the robot's), flange (Sunrise's), hook (this repo's hook frame: origin the inside of the
crook, shaft along +z; the hardware's "peg", assembly_config.json pose_flange_to_peg) and eyelet (this repo's; the
hardware's "hole"). The assembled hook pose is the eyelet frame itself, so the true eyelet is
base_to_flange_success * flange_to_peg. bel is the believed eyelet: the true one moved by the belief error (the
true eyelet's pose in bel: x, y, z mm, rotation vector deg, as training's goal_error).

The policy's task frame is bel; it never sees the base frame:
  observation (STOP_ACTOR_DIM = 21): the hook's pose in bel, position / 50 mm and the first two columns of its
    rotation matrix; the contact wrench on the hook, in the hook frame with torque about the hook's origin
    (hook_rl_env.wrench_features); the last motion action.
  action (7, in [-1, 1]): a delta pose times max_delta (x, y, z mm; rotation vector x, y, z deg) along bel's axes,
    turning about the hook's origin (the crook), applied to the hook's measured pose; the seventh value > 0
    declares the hook assembled, which ends the episode with the robot holding still.
Translation and rotation axes are bel's, not the hook's: the attached part only supplies the point turned about
and the frame of the wrench.

Wrench: Sunrise's getExternalForceTorque(flange) (joint-torque estimate, flange frame, torque about the flange
origin), recorded every 10 ms while a motion runs. WrenchModel takes away a bias and an unmodelled load (mass and
centre, from free-space readings at several orientations: --calibrate) and a tare at each start, multiplies by
the sign convention (+1: the reading is the force the environment exerts on the robot, which is the policy's
force on the hook), and moves it to the hook's origin, hook frame. The policy is trained for gain +-15%, offset
+-0.3 N and noise up to 1 N, and the contact forces it makes are ~1-2 N: the load model is not optional on
hardware. The reading after a chunk is the median of its last TAIL_S of samples.

Execution: an action becomes one chunk, N_WAYPOINTS flange poses along the hook's straight-line, slerp-in-rotation
move about the crook, sent as one Sunrise spline (blocking) in Cartesian impedance; the next observation comes
after it ends, quasi-statically, as kuka_hook_sim found necessary. Guards stop the episode (the robot then holds
its last set point, compliant): a joint within --joint-margin of its limit, a target leaving the training bounds
(in bel), a chunk whose contact force or torque (50 ms running median) exceeds the limits (FORCE_LIMIT_N comment:
beyond them it is the gripper or arm, not modelled in training, touching the fixture).

Pull test (after a declared done): the hook is pulled PULL_MM along base +z (the eyelet stands upright in both
cells) and back; caught under the loop's top wire it moves less than PULL_CAUGHT of that, and the wire pushes
it down (a negative base-z force; a positive one with the hook caught means the wrench sign is flipped).

    export (mujoco_env): checkpoint -> numpy policy file (torch-free; the robot env has no torch) + parity check
        conda run -n mujoco_env python scripts/kuka_hook_deploy.py export CKPT --out POLICY.npz
    sim (mujoco_env): this loop on kuka_hook_sim's iiwa, the eval set's cases, Sunrise's sensing modelled (SimIO)
        conda run -n mujoco_env python scripts/kuka_hook_deploy.py sim POLICY.npz --eval-set SET.npz \\
            --cases 0-99 --out-dir OUT [--compare KUKA_SIM_RUN/episodes.csv]
    robot (sunrise_wrapper env, the ROS2ClientV2 app started on the pendant; it connects to this PC):
        /home/rp/miniconda3/envs/sunrise_wrapper/bin/python scripts/kuka_hook_deploy.py robot POLICY.npz \\
            --assembly hook_v6 [--check-only] [--calibrate] [--axes-test] [--step] [--episodes N] [--error ...]
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

SUNRISE_REPO = Path("/home/rp/abhay_ws/sunrise-wrapper-abhay")
BULK = Path("/media/rp/Elements1/abhay_ws/contact-sampling/hook_rl/deploy")
N_WAYPOINTS = 5
TAIL_S = 0.05  # s of samples at a chunk's end whose median is the observed wrench
GRAVITY = np.array([0.0, 0.0, -9.81])  # base frame (the iiwa stands on the table)
IIWA14_LIMITS_DEG = np.array([170.0, 120.0, 170.0, 120.0, 170.0, 120.0, 175.0])
JOINT_MARGIN_DEG = 2.0  # stop just before Sunrise's own limit stop; a wider margin cost far more episodes in sim
# On the hook, about its origin, after a GUARD_WINDOW running median. Simulated (swing cell, LOW): hook-eyelet
# contact alone peaks below 7.7 N and 0.24 N m; the gripper touching the fixture reaches 25 N, 5.8 N m.
FORCE_LIMIT_N, TORQUE_LIMIT_NM = 15.0, 1.0
GUARD_WINDOW = 5  # samples (50 ms)
TARE_MAX = (2.0, 0.3)  # N, N m: a start residual above this is contact or a stale load model, not drift
PULL_MM, PULL_CAUGHT = 8.0, 0.4
LIFT_MM = 100.0  # free-space station above the start (bel z) for --calibrate and --axes-test
CALIB_ROTVECS_DEG = [(0, 0, 0), (30, 0, 0), (-30, 0, 0), (0, 30, 0), (0, -30, 0), (0, 60, 0), (0, 0, 30),
                     (0, 0, -30), (30, 30, 0), (-30, 30, 0)]  # about the hook's origin, along bel's axes
SAMPLE_DT = 0.01  # Sunrise's recording period


# ---------------------------------------------------------------- frames
def make_tf(R, p):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, p
    return T


def inv_tf(T):
    return make_tf(T[:3, :3].T, -T[:3, :3].T @ T[:3, 3])


def kuka_tf(xyz_m, abc_deg):
    """A KUKA pose (x, y, z m; A, B, C deg = Rz(A) Ry(B) Rx(C)) as a 4x4."""
    return make_tf(Rotation.from_euler("ZYX", abc_deg, degrees=True).as_matrix(), np.asarray(xyz_m, dtype=float))


def error_tf(err):
    """T_bel_true of a belief error: the true eyelet in the believed frame (mm, rotation vector deg)."""
    return make_tf(Rotation.from_rotvec(np.deg2rad(err[3:])).as_matrix(), np.asarray(err[:3], dtype=float) * 1e-3)


def pose_tf(pos_mm, rotvec_deg):
    return make_tf(Rotation.from_rotvec(np.deg2rad(rotvec_deg)).as_matrix(), np.asarray(pos_mm, dtype=float) * 1e-3)


def interp_tf(T0, T1, u):
    """Straight line in position, slerp in rotation."""
    rv = Rotation.from_matrix(T1[:3, :3] @ T0[:3, :3].T).as_rotvec()
    return make_tf(Rotation.from_rotvec(u * rv).as_matrix() @ T0[:3, :3], (1 - u) * T0[:3, 3] + u * T1[:3, 3])


def twist(T0, T1):
    return np.concatenate([T1[:3, 3] - T0[:3, 3], Rotation.from_matrix(T1[:3, :3] @ T0[:3, :3].T).as_rotvec()])


def pose_mm_deg(T):
    """(position mm, rotation vector deg) of a 4x4."""
    return T[:3, 3] * 1e3, np.degrees(Rotation.from_matrix(T[:3, :3]).as_rotvec())


def action_path(T_bf, T_base_bel, T_fh, a6, max_delta, n=N_WAYPOINTS):
    """Flange poses (n, 4, 4) moving the hook by action a6 (6, in [-1, 1]) from its measured pose: the delta
    (times max_delta) along bel's axes, turning about the hook's origin; n evenly spaced waypoints, the last the
    target (hook_rl_env.HookAssemblyBatch.step's rule, in the robot's base frame)."""
    d = np.clip(a6, -1, 1) * max_delta
    T_bh = T_bf @ T_fh
    R_bel = T_base_bel[:3, :3]
    dp, rv = R_bel @ d[:3] * 1e-3, R_bel @ np.deg2rad(d[3:])
    T_hf = inv_tf(T_fh)
    return np.array([make_tf(Rotation.from_rotvec(s * rv).as_matrix() @ T_bh[:3, :3], T_bh[:3, 3] + s * dp) @ T_hf
                     for s in np.arange(1, n + 1) / n])


def hook_wrench_from_flange(w, T_fh):
    """Wrenches (..., 6) in the flange frame with torque about the flange origin -> hook frame, about its origin."""
    R, p = T_fh[:3, :3], T_fh[:3, 3]
    f = w[..., :3]
    tau = w[..., 3:] - np.cross(p, f)  # tau_B = tau_A + (r_A - r_B) x f, r_A = 0, r_B = p
    return np.concatenate([f @ R, tau @ R], axis=-1)


def linking(a, loop):
    """Gauss linking integral of open polylines a (..., n, 3) with the closed loop (m, 3): hook_linking.linking,
    copied because that module needs trimesh and numba, which the robot's env lacks."""
    b = np.vstack([loop, loop[:1]])
    p1, p2 = a[..., :-1, None, :], a[..., 1:, None, :]
    p3, p4 = b[:-1], b[1:]
    r13, r14, r23, r24 = p3 - p1, p4 - p1, p3 - p2, p4 - p2

    def unit(v):
        return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)

    n1, n2, n3, n4 = unit(np.cross(r13, r14)), unit(np.cross(r14, r24)), unit(np.cross(r24, r23)), unit(np.cross(r23, r13))
    omega = sum(np.arcsin(np.clip(np.sum(x * y, -1), -1, 1)) for x, y in ((n1, n2), (n2, n3), (n3, n4), (n4, n1)))
    sign = np.sign(np.sum(np.cross(p4 - p3, p2 - p1) * r13, -1))
    return (omega * sign).sum((-1, -2)) / (4 * np.pi)


# ---------------------------------------------------------------- policy
class Policy:
    """The exported actor (export): numpy only. Also carries the training settings the loop needs."""

    def __init__(self, path):
        z = np.load(path, allow_pickle=False)
        self.path = str(path)
        self.mean, self.var, self.clip = z["norm_mean"], z["norm_var"], float(z["norm_clip"])
        self.layers = [(z[f"W{k}"], z[f"b{k}"]) for k in range(int(z["n_layers"]))]
        self.max_delta = z["max_delta"]
        self.wrench_obs, self.wrench_scale, self.unit_floor = str(z["wrench_obs"]), z["wrench_obs_scale"], z["wrench_unit_floor"]
        self.bounds_mm = z["bounds_mm"]
        self.start_pos_mm, self.start_quat = z["start_pos_mm"], z["start_quat_wxyz"]
        self.ref_pos_mm, self.ref_quat = z["ref_pos_mm"], z["ref_quat_wxyz"]  # the reference assembly path, eyelet frame
        self.nominal_perturb, self.goal_error = z["nominal_perturb"], z["goal_error"]
        self.seat_mm, self.success_link = float(z["seat_mm"]), float(z["success_link"])
        self.link_hook, self.link_loop = z["link_hook"], z["link_loop"]
        self.checkpoint, self.trained = str(z["checkpoint"]), json.loads(str(z["env_json"]))

    def wrench_features(self, w):
        if self.wrench_obs == "absolute":
            return w / self.wrench_scale
        f, t = w[:3], w[3:]
        return np.concatenate([f / max(np.linalg.norm(f), self.unit_floor[0]),
                               t / max(np.linalg.norm(t), self.unit_floor[1])])

    def features(self, T_bel_hook, w_hook, prev_action):
        return np.concatenate([T_bel_hook[:3, 3] * 1e3 / 50.0, T_bel_hook[:3, :2].T.reshape(6),
                               self.wrench_features(w_hook), prev_action])

    def act(self, x):
        h = np.clip((x - self.mean) / np.sqrt(self.var + 1e-8), -self.clip, self.clip)
        for k, (W, b) in enumerate(self.layers):
            h = W @ h + b
            if k < len(self.layers) - 1:
                h = np.where(h > 0, h, np.expm1(np.minimum(h, 0.0)))  # ELU
        return h

    def judge(self, T_true_hook):
        """Against the true eyelet: (seated, origin offset mm, rotation offset deg, linking number)."""
        p, R = T_true_hook[:3, 3] * 1e3, T_true_hook[:3, :3]
        link = float(linking((self.link_hook @ R.T + p)[None], self.link_loop)[0])
        seat = float(np.linalg.norm(p))
        return (seat < self.seat_mm and link > self.success_link, seat,
                float(np.degrees(np.linalg.norm(Rotation.from_matrix(R).as_rotvec()))), link)


# ---------------------------------------------------------------- wrench
class WrenchModel:
    """Sunrise's flange-frame reading -> contact wrench on the robot (flange frame):
        sign * (reading - b - [mu g_F; pi x g_F] - tare),
    g_F gravity in the flange frame; b (6) a bias, mu an unmodelled load mass (kg; negative if Sunrise's load
    data overstate it), pi = mu c its first moment (c its centre, flange frame), all fitted by least squares from
    free-space readings (fit); tare a per-episode residual at the start."""

    def __init__(self, theta=None, sign=1.0):
        self.theta = np.zeros(10) if theta is None else np.asarray(theta, dtype=float)  # b_f, mu, b_tau, pi
        self.sign, self.tare, self.info = float(sign), np.zeros(6), {}

    @staticmethod
    def design(R_bf):
        g = np.einsum("nji,j->ni", R_bf, GRAVITY)  # R^T g
        A = np.zeros((len(g), 6, 10))
        A[:, :3, :3] = np.eye(3)
        A[:, :3, 3] = g
        A[:, 3:, 4:7] = np.eye(3)
        A[:, 3:, 7:] = -np.array([[[0, -z, y], [z, 0, -x], [-y, x, 0]] for x, y, z in g])  # pi x g = -[g]x pi
        return A

    def predict(self, R_bf):
        return self.design(R_bf) @ self.theta

    def fit(self, R_bf, raw):
        """Least squares over readings raw (n, 6) at flange orientations R_bf (n, 3, 3); residual rms (N, N m)."""
        A = self.design(R_bf)
        self.theta = np.linalg.lstsq(A.reshape(-1, 10), raw.reshape(-1), rcond=None)[0]
        res = raw - A @ self.theta
        mu = self.theta[3]
        self.info = {"bias": self.theta[[0, 1, 2, 4, 5, 6]].tolist(), "load_kg": float(mu),
                     "load_centre_mm": (self.theta[7:] / mu * 1e3).tolist() if abs(mu) > 1e-3 else None,
                     "residual_rms": [float(np.sqrt((res[:, :3] ** 2).mean())), float(np.sqrt((res[:, 3:] ** 2).mean()))],
                     "orientations": len(raw)}
        return self.info["residual_rms"]

    def contact(self, raw, R_bf):
        return self.sign * (raw - self.predict(R_bf) - self.tare)

    def save(self, path):
        Path(path).write_text(json.dumps({"theta": self.theta.tolist(), "sign": self.sign, **self.info}, indent=2))

    @classmethod
    def load(cls, path):
        d = json.loads(Path(path).read_text())
        wm = cls(d["theta"], d.get("sign", 1.0))
        wm.info = {k: v for k, v in d.items() if k not in ("theta", "sign")}
        return wm


def hook_wrenches(samples, wm, T_fh):
    """Contact wrenches on the hook (n, 6; hook frame, about its origin) of a chunk's samples."""
    if not len(samples["t"]):
        return np.zeros((0, 6))
    return hook_wrench_from_flange(wm.contact(samples["raw"], samples["T"][:, :3, :3]), T_fh)


def tail_median(samples, w):
    if not len(w):
        return None
    return np.median(w[samples["t"] >= samples["t"][-1] - TAIL_S + 1e-9], axis=0)


def filtered_peaks(w, window=GUARD_WINDOW):
    """Largest force and torque (N, N m) of wrenches (n, 6) after a running median over window samples: impacts
    while the arm moves spike for a sample or two (and the iiwa's estimate is noisy in motion); pushing lasts."""
    if not len(w):
        return 0.0, 0.0
    out = []
    for x in (np.linalg.norm(w[:, :3], axis=1), np.linalg.norm(w[:, 3:], axis=1)):
        out.append(float(np.median(np.lib.stride_tricks.sliding_window_view(x, window), axis=1).max()
                         if len(x) >= window else np.median(x)))
    return tuple(out)


# ---------------------------------------------------------------- robot interfaces
class RobotFault(RuntimeError):
    pass


class SunriseIO:
    """KukaRobot (sunrise-wrapper-abhay) with matrices in and out. Samples: t (n,) s, T (n, 4, 4) flange poses,
    raw (n, 6) flange-frame wrench readings, recorded by ROS2ClientV2 while a spline runs."""

    def __init__(self, robot, speed):
        from kuka_sunrise.kuka_motion import to_matrix
        from kuka_sunrise.common import SunriseMessages
        self.robot, self.speed, self.to_matrix, self.msg = robot, speed, to_matrix, SunriseMessages
        self.q_lo, self.q_hi = -np.deg2rad(IIWA14_LIMITS_DEG), np.deg2rad(IIWA14_LIMITS_DEG)
        self.speed_sent = None

    def flange(self):
        # dims=6: get_flange_pose(dims=7) hands Sunrise's radians to a converter that expects degrees
        return self.to_matrix(self.robot.get_flange_pose(dims=6, degrees=True), degrees=True)[0]

    def joints(self):
        return np.asarray(self.robot.get_joint_positions(), dtype=float).reshape(-1)[:7]

    def set_speed(self, speed):
        if speed != self.speed_sent:
            si = self.robot.sunrise_interface
            si.send_string(self.msg.change_joint_rel_vel)
            si.receive_string()
            si.send_string(str(speed))
            si.receive_string()
            self.speed_sent = speed

    def move_path(self, path, speed=None):
        self.set_speed(speed or self.speed)
        q = np.array([np.concatenate([T[:3, 3], Rotation.from_matrix(T[:3, :3]).as_quat()]) for T in path])  # xyzw
        self.robot.execute_cartesian_trajectory(q, moving_frame="flange", reference_frame="base")
        pose, raw, t = self.robot.sunrise_interface.receive_observations_sunrise_raw()  # mm, deg (A, B, C)
        T = np.array([kuka_tf(p[:3] * 1e-3, p[3:]) for p in pose]).reshape(-1, 4, 4)
        return {"t": np.asarray(t, dtype=float) * 1e-3, "T": T, "raw": np.asarray(raw, dtype=float).reshape(-1, 6)}

    def move_to(self, T_bf, speed):
        """LIN to a flange pose (approach and stations; not policy chunks)."""
        self.set_speed(speed)
        self.robot.move_cartesian(T_bf, interpolation="LIN")
        self.robot.reset_trajectory_continuity()

    def sample_static(self):
        """Readings at the current pose: Sunrise records only while moving, so dither 0.5 mm up and back."""
        T = self.flange()
        up = T.copy()
        up[2, 3] += 5e-4
        return self.move_path([up, T], speed=0.05)

    def close(self):
        self.robot.sunrise_interface.send_string(self.msg.close_connection)
        self.robot.sunrise_interface.close()


class SimIO:
    """kuka_hook_sim.KukaCell behind SunriseIO's interface. A chunk streams its flange waypoints, at the cell's
    impedance tool centre point, with a minimum-jerk profile over chunk_s, then holds settle_s (Sunrise's motion
    end, the client's 0.2 s and the round trips); samples every SAMPLE_DT while the chunk runs, not while it
    holds, as ROS2ClientV2 records. The reading: sign * (true contact wrench on the robot at the flange, as the
    iiwa estimates it from joint torques) + an unmodelled load (mass kg, centre mm in the flange frame) + bias +
    white noise. ref: the wrench kuka_hook_sim's own policy interface computes (hook frame), for the parity check."""

    def __init__(self, cell, chunk_s=0.6, settle_s=0.4, sign=1.0, load=(0.0, 0.0, 0.0, 0.0), bias=np.zeros(6),
                 noise=(0.0, 0.0), seed=0):
        import mujoco
        self.mujoco, self.cell = mujoco, cell
        self.chunk_ticks, self.settle_ticks = round(chunk_s / 1e-3), round(settle_s / 1e-3)
        self.sign, self.bias = sign, np.asarray(bias, dtype=float)
        self.load_kg, self.load_c = load[0], np.asarray(load[1:], dtype=float) * 1e-3
        self.noise = np.repeat(np.asarray(noise, dtype=float), 3)
        self.rng = np.random.default_rng(seed)
        self.T_flange_tcp = cell.T_flange_hook @ cell.T_hook_tcp
        self.q_lo, self.q_hi = cell.q_lo, cell.q_hi
        self.t, self.gripper_contact = 0.0, False  # gripper_contact: sim-only diagnosis, set by reading()

    def flange(self):
        return self.cell.flange_tf()

    def joints(self):
        return self.cell.d.qpos.copy()

    def reading(self):
        cell = self.cell
        T = cell.flange_tf()
        w, _ = cell.contact_wrench_flange()  # base frame, torque about the flange
        R = T[:3, :3]
        g = R.T @ GRAVITY * self.load_kg
        raw = (self.sign * np.concatenate([R.T @ w[:3], R.T @ w[3:]]) + np.concatenate([g, np.cross(self.load_c, g)])
               + self.bias + self.rng.normal(0.0, 1.0, 6) * self.noise)
        self.gripper_contact |= any(cell.m.geom(g).name.startswith("gripper") for k in range(cell.d.ncon)
                                    for g in (cell.d.contact[k].geom1, cell.d.contact[k].geom2))
        return T, raw, cell.hook_wrench()[0]

    def tick(self, T_d, v_d, out):
        self.cell.sim_step(T_d, v_d)
        self.t += 1e-3
        margin = np.minimum(self.cell.d.qpos - self.q_lo, self.q_hi - self.cell.d.qpos)
        if margin.min() < np.deg2rad(1.0):  # kuka_hook_sim's LIMIT_FAULT: Sunrise stops the robot
            raise RobotFault(f"joint limit A{int(np.argmin(margin)) + 1}")
        if out is not None and round(self.t / 1e-3) % round(SAMPLE_DT / 1e-3) == 0:
            T, raw, ref = self.reading()
            out["t"].append(self.t), out["T"].append(T), out["raw"].append(raw), out["ref"].append(ref)

    @staticmethod
    def pack(out):
        return {"t": np.array(out["t"]).reshape(-1), "T": np.array(out["T"]).reshape(-1, 4, 4),
                "raw": np.array(out["raw"]).reshape(-1, 6), "ref": np.array(out["ref"]).reshape(-1, 6)}

    def move_path(self, path, speed=None):
        pts = [self.cell.site_tf(self.cell.tcp)] + [T @ self.T_flange_tcp for T in path]
        out = {"t": [], "T": [], "raw": [], "ref": []}
        T_prev = pts[0]
        for k in range(1, self.chunk_ticks + 1):
            tau = k / self.chunk_ticks
            s = tau ** 3 * (10 - 15 * tau + 6 * tau ** 2) * (len(pts) - 1)
            i = min(int(s), len(pts) - 2)
            T_d = interp_tf(pts[i], pts[i + 1], s - i)
            self.tick(T_d, twist(T_prev, T_d) / 1e-3, out)
            T_prev = T_d
        for _ in range(self.settle_ticks):
            self.tick(T_prev, np.zeros(6), None)
        return self.pack(out)

    def move_to(self, T_bf, speed=None):
        """Placed by inverse kinematics (free-space stations only), then settled."""
        cell, m = self.cell, self.mujoco
        q = cell.ik(T_bf, cell.d.qpos.copy())
        cell.d.qpos[:], cell.d.qvel[:] = q, 0.0
        m.mj_forward(cell.m, cell.d)
        cell.q_post = q.copy()
        T = cell.site_tf(cell.tcp)
        for _ in range(200):
            self.tick(T, np.zeros(6), None)

    def sample_static(self):
        T = self.cell.site_tf(self.cell.tcp)
        out = {"t": [], "T": [], "raw": [], "ref": []}
        for _ in range(300):
            self.tick(T, np.zeros(6), out)
        return self.pack(out)


class FakeSunrise:
    """Stands in for the ROS2ClientV2 app (robot --fake-sunrise): connects to the wrapper's server and answers its
    messages as KukaRobot expects them (the protocol as read from sunrise_interface.py's side), on a kinematic
    robot: a motion puts the flange at its last pose and records 10 ms samples along the way (zero wrench plus
    noise, in free space everywhere). For rehearsing the robot command's flow and wire formats, not physics."""

    def __init__(self, port, T_flange, noise=(0.05, 0.002)):
        import socket
        import threading
        self.port, self.T, self.noise = port, T_flange.copy(), np.repeat(noise, 3)
        self.rng, self.obs, self.sock = np.random.default_rng(0), [], socket.socket()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def send(self, s):
        self.sock.sendall((s + "\r\n").encode())

    def recv(self):
        data = b""
        while not data.endswith(b"\n"):
            chunk = self.sock.recv(4096)
            if not chunk:
                raise ConnectionError
            data += chunk
        return data.decode().strip()

    def get_array(self):
        self.send("s")
        rows = int(self.recv())
        out = []
        for _ in range(rows):
            self.send("s")
            out.append([float(v) for v in self.recv().split(",")])
        return np.array(out)

    def send_array(self, a):
        a = np.atleast_2d(a)
        self.send(str(len(a)))
        if self.recv() != "s" or not a.size:
            return
        self.send(str(a.shape[1]))
        self.recv()
        self.sock.sendall(("\r\n".join(f"{v:.9g}" for v in a.reshape(-1)) + "\r\neof\r\n").encode())
        self.recv()

    @staticmethod
    def wire(T):  # x, y, z m; A, B, C rad
        return np.concatenate([T[:3, 3], Rotation.from_matrix(T[:3, :3]).as_euler("ZYX")])

    def move(self, poses):
        pts, self.obs = [self.T] + [kuka_tf(p[:3], np.degrees(p[3:])) for p in poses], []
        for i in range(len(pts) - 1):
            for u in np.linspace(0, 1, 20)[1:]:
                T = interp_tf(pts[i], pts[i + 1], u)
                self.obs.append(np.concatenate([[time.time() * 1e3 + len(self.obs) * 10], self.wire(T),
                                                self.rng.normal(0, 1, 6) * self.noise]))
        self.T = pts[-1]

    def run(self):
        for _ in range(100):
            try:
                self.sock.connect(("127.0.0.1", self.port))
                break
            except OSError:
                time.sleep(0.1)
        self.recv()  # "Okay"
        self.send("Establish_Communication")
        while True:
            try:
                cmd = self.recv()
            except (ConnectionError, OSError):
                return
            if cmd == "close_connection":
                return
            if cmd in ("change_ctrl_mode", "change_joint_rel_vel", "change_obs_time"):
                self.send("s")
                self.recv()
                self.send("Done")
            elif cmd == "set_impedance_values":
                self.get_array()
                self.send("Done")
            elif cmd == "get_fk":
                self.send_array(self.wire(self.T))
                self.send("Done")
            elif cmd == "send_joint_pos":
                self.send_array(np.deg2rad([0, 30, 0, -60, 0, 60, 0]))
                self.send("Done")
            elif cmd in ("move_lin", "move_ptp"):
                self.move(self.get_array())
                self.send("Done")
            elif cmd == "execute_cartesian_spline":
                poses = self.get_array()
                self.send("Done")
                self.recv()  # partial-insertion choice
                self.move(poses)
                self.send("Done")
            elif cmd == "recv_obs":
                self.send_array(np.array(self.obs).reshape(-1, 13))
                self.send("Done")


# ---------------------------------------------------------------- the loop
class Guard:
    def __init__(self, bounds_mm, q_lo, q_hi, joint_margin_deg=JOINT_MARGIN_DEG, force_n=FORCE_LIMIT_N,
                 torque_nm=TORQUE_LIMIT_NM):
        self.bounds, self.q_lo, self.q_hi = bounds_mm, q_lo, q_hi
        self.margin, self.force, self.torque = np.deg2rad(joint_margin_deg), force_n, torque_nm

    def joints(self, q):
        margin = np.minimum(q - self.q_lo, self.q_hi - q)
        return f"guard: joint A{int(np.argmin(margin)) + 1} near its limit" if margin.min() < self.margin else None

    def target(self, T_bel_hook):
        p = T_bel_hook[:3, 3] * 1e3
        return "guard: target out of bounds" if np.any((p < self.bounds[:, 0]) | (p > self.bounds[:, 1])) else None

    def wrench(self, w_hook):
        f, t = filtered_peaks(w_hook)
        if f > self.force:
            return f"guard: force {f:.1f} N"
        return f"guard: torque {t:.2f} N m" if t > self.torque else None


def tare(io, wm):
    """Re-zero the reading at the start pose: (residual force N, torque N m, its samples). Refused (None) when the
    residual is too large for drift (contact at the start, or a stale load model)."""
    smp = io.sample_static()
    if not len(smp["t"]):
        return None, smp
    res = np.median(smp["raw"] - wm.predict(smp["T"][:, :3, :3]), axis=0)
    size = (float(np.linalg.norm(res[:3])), float(np.linalg.norm(res[3:])))
    if size[0] > TARE_MAX[0] or size[1] > TARE_MAX[1]:
        return None, smp
    wm.tare = res
    return size, smp


def run_episode(pol, io, T_fh, T_base_bel, wm, guard, max_steps, first_samples, confirm=None, log=None):
    """The policy loop (module docstring). Returns (end reason, per-step records, all chunk samples)."""
    T_bel_base = inv_tf(T_base_bel)
    w = tail_median(first_samples, hook_wrenches(first_samples, wm, T_fh))
    w_hook = np.zeros(6) if w is None else w
    prev, steps, chunks, end = np.zeros(6), [], [], "max steps"
    for k in range(max_steps):
        T_bf, q = io.flange(), io.joints()
        T_bel_hook = T_bel_base @ T_bf @ T_fh
        x = pol.features(T_bel_hook, w_hook, prev)
        a = pol.act(x)
        rec = {"step": k, "T_bf": T_bf, "q": q, "x": x, "action": a, "w_hook": w_hook.copy(), "wall": time.time()}
        steps.append(rec)
        reason = guard.joints(q)
        if reason:
            end = reason
            break
        if a[6] > 0:
            end = "declared done"
            break
        a6 = np.clip(a[:6], -1, 1)
        path = action_path(T_bf, T_base_bel, T_fh, a6, pol.max_delta)
        reason = guard.target(T_bel_base @ path[-1] @ T_fh)
        if reason:
            end = reason
            break
        if confirm is not None and not confirm(k, a6 * pol.max_delta, T_bel_hook, w_hook):
            end = "operator abort"
            break
        try:
            smp = io.move_path(path)
        except RobotFault as e:
            end = f"robot fault: {e}"
            break
        wh = hook_wrenches(smp, wm, T_fh)
        chunks.append({**smp, "w_hook": wh, "step": np.full(len(smp["t"]), k)})
        rec.update(n_samples=len(wh), peak_force_raw=float(np.linalg.norm(wh[:, :3], axis=1).max()) if len(wh) else 0.0,
                   chunk_s=float(np.ptp(smp["t"])) if len(wh) else 0.0)
        rec["peak_force"], rec["peak_torque"] = filtered_peaks(wh)
        if log:
            p, r = pose_mm_deg(T_bel_hook)
            log(f"step {k:3d}  hook in bel {np.round(p, 1)} mm {np.round(r, 1)} deg  |F| {np.linalg.norm(w_hook[:3]):.2f} N"
                f"  action {np.round(a6 * pol.max_delta, 1)}  done {a[6]:+.2f}")
        reason = guard.wrench(wh)
        if reason:
            end = reason
            break
        w = tail_median(smp, wh)
        if w is not None:
            w_hook = w
        prev = a6
    return end, steps, chunks


def pull_test(io, wm, T_fh):
    """PULL_MM up along base z and back (module docstring): (caught, moved mm, base-z force on the hook N)."""
    T0 = io.flange()
    up = T0.copy()
    up[2, 3] += PULL_MM * 1e-3
    smp = io.move_path([up])
    T1 = io.flange()
    w = tail_median(smp, hook_wrenches(smp, wm, T_fh))
    fz = float(((T1 @ T_fh)[:3, :3] @ w[:3])[2]) if w is not None else float("nan")
    moved = float((T1[2, 3] - T0[2, 3]) * 1e3)
    io.move_path([T0])
    return moved < PULL_CAUGHT * PULL_MM, moved, fz


def stations(T_base_bel, T_bel_hook0, T_fh, rotvecs_deg, lift_mm=LIFT_MM):
    """Flange poses with the hook lift_mm above its start (bel z), turned about its origin by each rotation
    vector (bel's axes)."""
    T_bh = T_base_bel @ T_bel_hook0
    R_bel = T_base_bel[:3, :3]
    p = T_bh[:3, 3] + R_bel[:, 2] * lift_mm * 1e-3
    return [make_tf(Rotation.from_rotvec(R_bel @ np.deg2rad(rv)).as_matrix() @ T_bh[:3, :3], p) @ inv_tf(T_fh)
            for rv in rotvecs_deg]


def calibrate(io, wm, T_list, speed, log=print):
    """Fit the load model from free-space readings at the given flange poses (unreachable ones skipped)."""
    R, raw = [], []
    for k, T in enumerate(T_list):
        try:
            io.move_to(T, speed)
        except RuntimeError as e:
            log(f"  station {k}: skipped ({e})")
            continue
        smp = io.sample_static()
        if len(smp["t"]):
            R.append(io.flange()[:3, :3]), raw.append(np.median(smp["raw"], axis=0))
            log(f"  station {k}: reading {np.round(raw[-1], 3)}")
    if len(raw) < 4:
        raise RuntimeError(f"only {len(raw)} calibration stations reached")
    rms = wm.fit(np.array(R), np.array(raw))
    log(f"  load model: {json.dumps(wm.info)}  (residual rms {rms[0]:.3f} N, {rms[1]:.4f} N m)")
    return wm


def axes_test(io, pol, T_fh, T_base_bel, frac=0.5, log=print):
    """Each action dimension +frac then -frac in free space: the hook's measured motion in bel against the
    command. Rotations should leave the crook (the hook's origin) in place; watch it on the robot."""
    T_bel_base = inv_tf(T_base_bel)
    rows = []
    for i, name in enumerate("x y z rx ry rz".split()):
        for s in (frac, -frac):
            a = np.zeros(6)
            a[i] = s
            T0 = io.flange()
            io.move_path(action_path(T0, T_base_bel, T_fh, a, pol.max_delta))
            H0, H1 = T_bel_base @ T0 @ T_fh, T_bel_base @ io.flange() @ T_fh
            got = np.concatenate([(H1[:3, 3] - H0[:3, 3]) * 1e3,
                                  np.degrees(Rotation.from_matrix(H1[:3, :3] @ H0[:3, :3].T).as_rotvec())])
            rows.append((name, s * pol.max_delta[i], got))
            log(f"  {name} {s * pol.max_delta[i]:+5.1f}: moved {np.round(got[:3], 2)} mm, {np.round(got[3:], 2)} deg"
                f" (bel axes){'   crook drift %.2f mm' % np.linalg.norm(got[:3]) if i >= 3 else ''}")
    return rows


# ---------------------------------------------------------------- export
def cmd_export(args):
    import torch
    import hook_rl_env as E
    from eval_hook_rl import load_policy
    from hook_eyelet_model import ASSET_DIR, load_trajectory
    ac, env = load_policy(args.checkpoint)
    if ac.actor_dim != E.STOP_ACTOR_DIM or ac.log_std.shape[0] != 7:
        raise SystemExit(f"{args.checkpoint}: not a declared-completion policy (actor_dim {ac.actor_dim})")
    lin = [m for m in ac.actor if isinstance(m, torch.nn.Linear)]
    _, pos, quat = load_trajectory(ASSET_DIR / E.GUIDE_CSVS[0])
    geo = E.Geometry()
    out = {"norm_mean": ac.norm.mean[:ac.actor_dim].numpy().astype(np.float64),
           "norm_var": ac.norm.var[:ac.actor_dim].numpy().astype(np.float64), "norm_clip": ac.norm.clip,
           "n_layers": len(lin), "max_delta": np.asarray(env.get("max_delta", E.MAX_DELTA), dtype=float),
           "wrench_obs": env.get("wrench_obs", "absolute"), "wrench_obs_scale": E.WRENCH_OBS_SCALE,
           "wrench_unit_floor": np.array(E.WRENCH_UNIT_FLOOR), "bounds_mm": E.BOUNDS_MM,
           "start_pos_mm": pos[0] * 1e3, "start_quat_wxyz": quat[0], "nominal_perturb": E.NOMINAL_PERTURB,
           "ref_pos_mm": pos * 1e3, "ref_quat_wxyz": quat,
           "goal_error": np.asarray(env.get("goal_error", np.zeros(6)), dtype=float), "seat_mm": E.SEAT_MM,
           "success_link": E.SUCCESS_LINK, "link_hook": geo.link_hook, "link_loop": geo.link_loop,
           "checkpoint": str(Path(args.checkpoint).resolve()), "env_json": json.dumps(env, default=str)}
    for k, m in enumerate(lin):
        out[f"W{k}"], out[f"b{k}"] = m.weight.detach().double().numpy(), m.bias.detach().double().numpy()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **out)
    pol = Policy(args.out)  # parity with the torch actor on plausible observations
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.uniform(-2, 2, (2000, 3)), rng.uniform(-1, 1, (2000, 6)), rng.normal(0, 1, (2000, 6)),
                        rng.uniform(-1, 1, (2000, 6))], axis=1)
    full = np.zeros((len(x), ac.norm.mean.shape[0]), np.float32)
    full[:, :ac.actor_dim] = x
    with torch.no_grad():
        ref = ac.mean(ac.norm(torch.as_tensor(full))).numpy()
    err = np.abs(np.array([pol.act(v) for v in x]) - ref).max()
    print(f"wrote {args.out}; numpy vs torch actor, 2000 observations: max |diff| {err:.2e}")
    if err > 1e-4:
        raise SystemExit("parity check failed")


# ---------------------------------------------------------------- sim
def parse_cases(text):
    out = []
    for part in text.split(","):
        lo, _, hi = part.partition("-")
        out += list(range(int(lo), int(hi or lo) + 1))
    return out


def sim_worker(job):
    policy_path, cases, ids, kw = job
    import kuka_hook_sim as K
    from eval_hook_rl import outcome
    pol = Policy(policy_path)
    if kw["no_limit_avoidance"]:  # Sunrise's Cartesian motions keep the elbow angle; nothing steers off joint limits
        K.LIMIT_K = 0.0
    cell = K.KukaCell(impedance=kw["impedance"], tcp=kw["tcp"], cell=kw["cell_def"])
    if kw["start_seed"] == "continuation":  # starts' IK seeded where the arm gets following the reference path back
        q = cell.q_ref.copy()  # from the assembled pose, as Sunrise's motions keep the arm's configuration
        for p, qu in list(zip(pol.ref_pos_mm, pol.ref_quat))[::-1][::3]:  # from the assembled end
            q = cell._ik(cell.T_base_eyelet @ make_tf(Rotation.from_quat(qu, scalar_first=True).as_matrix(), p * 1e-3)
                         @ cell.T_hook_flange, q, 300)[0]
        cell.q_ref = q
    io = SimIO(cell, kw["chunk_s"], kw["settle_s"], sign=kw["sensor_sign"], load=kw["sensor_load"],
               bias=kw["sensor_bias"], noise=kw["sensor_noise"], seed=ids[0] if ids else 0)
    guard = Guard(pol.bounds_mm, io.q_lo, io.q_hi, kw["joint_margin"], kw["force_limit"], kw["torque_limit"])
    wm, calibrated, out = WrenchModel(sign=kw["assume_sign"]), not kw["calibrate"], []
    clean_sensor = (kw["sensor_sign"] == kw["assume_sign"] and not np.any(kw["sensor_load"])
                    and not np.any(kw["sensor_bias"]) and not np.any(kw["sensor_noise"]))
    for i, (pos, quat, err) in zip(ids, cases):
        rec = {"case": int(i)}
        try:
            cell.reset(pos, quat, err)
        except K.StartInCollision as e:
            out.append({**rec, "outcome": f"start in collision: {e}", "success": False})
            continue
        except RuntimeError as e:
            out.append({**rec, "outcome": f"no IK: {e}", "success": False})
            continue
        T_fh, T_true = cell.T_flange_hook, cell.T_base_eyelet
        T_base_bel = T_true @ inv_tf(error_tf(err))
        T_bel_hook0 = make_tf(Rotation.from_quat(quat, scalar_first=True).as_matrix(), np.asarray(pos) * 1e-3)
        if not calibrated:
            calibrate(io, wm, stations(T_base_bel, T_bel_hook0, T_fh, CALIB_ROTVECS_DEG), None, log=lambda s: None)
            rec["calib"] = json.dumps(wm.info)
            calibrated = True
            cell.reset(pos, quat, err)
        try:
            size, smp0 = tare(io, wm)
        except RobotFault as e:
            out.append({**rec, "outcome": f"robot fault at the start: {e}", "success": False})
            continue
        x0 = pol.features(inv_tf(T_base_bel) @ io.flange() @ T_fh, np.zeros(6), np.zeros(6))
        rec["parity_bel"] = float(np.abs(T_base_bel - cell.T_bel).max())
        rec["parity_obs_pose"] = float(np.abs(x0[:9] - cell.observe(pol.mean.shape[0])[0][:9]).max())
        if size is None:
            out.append({**rec, "outcome": "tare refused", "success": False})
            continue
        t0, io.gripper_contact = io.t, False
        end, steps, chunks = run_episode(pol, io, T_fh, T_base_bel, wm, guard, kw["max_steps"], smp0)
        seated, link, kp, seat, out_b = cell.judge()
        stopped = end == "declared done"
        rec.update(end=end, steps=sum("n_samples" in s for s in steps), stopped=stopped, seated=seated,
                   out=out_b, final_link=link, final_kp_mm=kp, final_seat_mm=seat, robot_time_s=io.t - t0,
                   success=bool(stopped and seated), pull_ok=None, tare_n=size[0], gripper_contact=io.gripper_contact,
                   **{f"peak_{k}": max([s.get(f"peak_{k}", 0.0) for s in steps] + [0.0])
                      for k in ("force", "torque", "force_raw")})
        own = pol.judge(inv_tf(T_true) @ io.flange() @ T_fh)
        rec.update(own_seated=own[0], own_seat_mm=own[1], own_rot_deg=own[2], own_link=own[3])
        if chunks and clean_sensor:
            w = np.concatenate([c["w_hook"] for c in chunks])
            ref = np.concatenate([c["ref"] for c in chunks])
            rec["parity_wrench"] = float(np.abs(w - ref).max())
        if stopped:  # the physical criterion for every declaration, seated within SEAT_MM or not
            rec["pull_any_ok"], rec["pull_pos_mm"] = cell.pull_test()
            rec["pull_ok"] = rec["success"] and rec["pull_any_ok"]
        if stopped and kw["pull_test"]:
            try:
                rec["own_pull_caught"], rec["own_pull_moved_mm"], rec["own_pull_fz_n"] = pull_test(io, wm, T_fh)
            except RobotFault:
                rec["own_pull_caught"] = None
        rec["outcome"] = end if end.startswith(("guard", "robot fault")) else outcome(rec)
        out.append(rec)
    return out


def cell_def(args):
    """kuka_hook_sim cell (KUKA poses mm, deg) of --cell with the mount, success pose and gripper overridden. A new
    mount without a success pose keeps the cell's eyelet where it was; without --gripper-box, a new mount gets
    only a clamp round the top 15 mm of the shaft (the adapter between it and the flange is not modelled)."""
    import kuka_hook_sim as K
    cell = dict(K.CELLS[args.cell])
    if args.flange_to_hook:
        if not args.success_flange and "base_to_eyelet" not in cell:
            E0 = kuka_tf(np.asarray(cell["base_to_flange_assembled"][:3]) * 1e-3, cell["base_to_flange_assembled"][3:]) \
                @ kuka_tf(np.asarray(cell["flange_to_hook"][:3]) * 1e-3, cell["flange_to_hook"][3:])
            cell["base_to_eyelet"] = [*(E0[:3, 3] * 1e3), *Rotation.from_matrix(E0[:3, :3]).as_euler("ZYX", degrees=True)]
        cell["flange_to_hook"] = list(args.flange_to_hook)
        T_fh = kuka_tf(np.asarray(args.flange_to_hook[:3]) * 1e-3, args.flange_to_hook[3:])
        cell["gripper"] = [(tuple((T_fh @ [-2e-3, 0.0, 0.062, 1.0])[:3] * 1e3), (7.0, 7.0, 8.0))]
    if args.success_flange:
        cell.pop("base_to_eyelet", None)
        cell["base_to_flange_assembled"] = list(args.success_flange)
    if args.gripper_box:
        cell["gripper"] = [(tuple(b[:3]), tuple(b[3:])) for b in args.gripper_box]
    return cell


def cmd_sim(args):
    import multiprocessing as mp
    from eval_hook_rl import wilson
    z = np.load(args.eval_set)
    ids = args.cases or list(range(len(z["pos"])))
    err = np.zeros_like(z["err"]) if args.no_error else z["err"]
    cases = [(z["pos"][i], z["quat"][i], err[i]) for i in ids]
    kw = {k: getattr(args, k) for k in ("impedance", "tcp", "cell", "chunk_s", "settle_s", "max_steps", "sensor_sign",
                                        "assume_sign", "sensor_load", "sensor_bias", "sensor_noise", "calibrate",
                                        "joint_margin", "force_limit", "torque_limit", "pull_test",
                                        "no_limit_avoidance", "start_seed")}
    kw["cell_def"] = cell_def(args)
    jobs = [(args.policy, cases[w::args.workers], ids[w::args.workers], kw) for w in range(args.workers)]
    with mp.get_context("fork").Pool(args.workers) as pool:
        recs = sorted([r for part in pool.map(sim_worker, jobs) for r in part], key=lambda r: r["case"])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.out_dir / "episodes.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r in recs for k in r}))
        w.writeheader()
        w.writerows(recs)
    ran = [r for r in recs if "end" in r]
    n_ok = sum(bool(r.get("pull_ok")) for r in recs)
    declared = [r for r in ran if r["stopped"]]
    summary = {"policy": args.policy, "eval_set": str(args.eval_set), "settings": {k: v for k, v in kw.items()},
               "no_error": args.no_error, "episodes": len(recs), "no_valid_start": len(recs) - len(ran),
               "assembled_and_pull_ok": n_ok / len(recs), "ci95": wilson(n_ok, len(recs)),
               "assembled_and_pull_ok_of_valid_starts": n_ok / max(len(ran), 1),
               "declared_and_pull_ok": sum(bool(r.get("pull_any_ok")) for r in recs) / len(recs),
               "declared_done": float(np.mean([r["stopped"] for r in ran])) if ran else None,
               "false_done": float(np.mean([r["stopped"] and not r["seated"] for r in ran])) if ran else None,
               "guard_or_fault_stop": float(np.mean([r["end"].startswith(("guard", "robot")) for r in ran])) if ran else None,
               "robot_time_to_success_s_p10_50_90": np.percentile([r["robot_time_s"] for r in ran if r.get("pull_ok")],
                                                                  [10, 50, 90]).round(1).tolist() if n_ok else None,
               "peak_force_n_p50_90_99": np.percentile([r["peak_force"] for r in ran], [50, 90, 99]).round(2).tolist() if ran else None,
               "peak_torque_nm_p50_90_99": np.percentile([r["peak_torque"] for r in ran], [50, 90, 99]).round(3).tolist() if ran else None,
               "gripper_contact": float(np.mean([r["gripper_contact"] for r in ran])) if ran else None,
               "own_judge_agrees": float(np.mean([r["own_seated"] == r["seated"] for r in ran])) if ran else None,
               "parity_max": {k: max([r[k] for r in recs if k in r] or [float("nan")])
                              for k in ("parity_bel", "parity_obs_pose", "parity_wrench")},
               "outcomes": {o: sum(r["outcome"] == o for r in recs) for o in sorted({r["outcome"] for r in recs})}}
    if declared and args.pull_test:
        truth = [bool(r.get("pull_ok")) for r in declared]
        mine = [bool(r.get("own_pull_caught")) for r in declared]
        summary["own_pull_test"] = {"declared": len(declared), "agrees_with_sim_pull_test": float(np.mean([a == b for a, b in zip(truth, mine)])),
                                    "caught_fz_n_p10_50_90": np.percentile([r["own_pull_fz_n"] for r, c in zip(declared, mine) if c],
                                                                           [10, 50, 90]).round(2).tolist() if any(mine) else None}
    if any("calib" in r for r in recs):
        summary["calibration"] = [json.loads(r["calib"]) for r in recs if "calib" in r]
    if args.compare:
        with open(args.compare) as f:
            ref = {int(r.get("case", r.get("episode"))): r["pull_ok"] == "True" for r in csv.DictReader(f)}
        pairs = [(bool(r.get("pull_ok")), ref[r["case"]]) for r in recs if r["case"] in ref]
        summary["compare"] = {"file": str(args.compare), "reference_success": float(np.mean([b for _, b in pairs])),
                              "only_this": sum(a and not b for a, b in pairs), "only_reference": sum(b and not a for a, b in pairs)}
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


# ---------------------------------------------------------------- robot
def cmd_robot(args):
    sys.path.insert(0, str(SUNRISE_REPO))
    from kuka_sunrise.kuka_motion import KukaRobot
    from kuka_sunrise.common import SunriseImpedanceValues
    pol = Policy(args.policy)
    asm = json.loads((SUNRISE_REPO / "config" / "assembly_config.json").read_text())[args.assembly]
    p = asm["pose_flange_to_peg"][0]  # m, deg: the mount the assembly's success pose was taught with
    T_fh_asm = kuka_tf(p[:3], p[3:])
    T_fh = (kuka_tf(np.asarray(args.flange_to_hook[:3]) * 1e-3, args.flange_to_hook[3:]) if args.flange_to_hook
            else T_fh_asm)
    if args.base_to_eyelet:
        T_true = kuka_tf(np.asarray(args.base_to_eyelet[:3]) * 1e-3, args.base_to_eyelet[3:])
    elif args.success_flange:  # taught with this run's mount: the assembled hook pose is the eyelet frame
        T_true = kuka_tf(np.asarray(args.success_flange[:3]) * 1e-3, args.success_flange[3:]) @ T_fh
    else:  # the eyelet where the assembly's own mount put it (it must not have moved since)
        s = asm["pose_base_to_flange_success"][0]
        T_true = kuka_tf(s[:3], s[3:]) @ T_fh_asm
    rng = np.random.default_rng(args.seed)
    out_dir = args.out_dir or BULK / f"robot_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    logf = open(out_dir / "session.log", "a")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    def ask(prompt):
        if args.yes:
            return True
        return input(f"{prompt} [y/N] ").strip().lower() == "y"

    wm = WrenchModel.load(args.wrench_model) if args.wrench_model else WrenchModel(sign=args.wrench_sign)
    if args.wrench_model and args.wrench_sign != wm.sign:
        wm.sign = args.wrench_sign
    log(f"policy {pol.path} (from {pol.checkpoint}); trained belief error +-{pol.goal_error.tolist()}")
    log(f"T_flange_hook\n{np.round(T_fh, 4)}\nT_base_eyelet (true)\n{np.round(T_true, 4)}")
    if not args.wrench_model and not args.calibrate:
        log("WARNING: no load model (--wrench-model / --calibrate): only the start tare corrects the wrench, so a "
            "Sunrise load-data error shows up as a force that turns with the hook")
    if args.fake_sunrise:
        args.system_ip = args.robot_ip = "127.0.0.1"
        T0 = T_true @ inv_tf(T_fh)
        T0[2, 3] += 0.15
        FakeSunrise(args.port, T0)
        log("FAKE SUNRISE: a kinematic stand-in answers on 127.0.0.1; nothing moves")
    log(f"waiting for the robot at {args.system_ip}:{args.port} (start ROS2ClientV2 on the pendant)")
    robot = KukaRobot()
    robot.configure({"system_IP": args.system_ip, "robot_IP": args.robot_ip, "port": args.port})
    io = SunriseIO(robot, args.speed)
    guard = Guard(pol.bounds_mm, io.q_lo, io.q_hi, args.joint_margin, args.force_limit, args.torque_limit)
    try:
        robot.set_control_mode_sunrise("CART_IMP", impedance_setting=getattr(SunriseImpedanceValues, args.impedance))
        T_bf, q = io.flange(), io.joints()
        p, r = pose_mm_deg(inv_tf(T_true) @ T_bf @ T_fh)
        log(f"hook in the true eyelet frame: {np.round(p, 1)} mm, rotation vector {np.round(r, 1)} deg")
        log(f"joints {np.round(np.degrees(q), 1)} deg; margins {np.round(np.degrees(np.minimum(q - io.q_lo, io.q_hi - q)), 1)}")
        if args.check_only:
            return
        for ep in range(args.episodes):
            err = (rng.uniform(-1, 1, 6) * args.random_error * pol.goal_error if args.random_error
                   else np.asarray(args.error, dtype=float))
            pert = (rng.uniform(-1, 1, 6) * pol.nominal_perturb if args.random_start
                    else np.asarray(args.start_perturb, dtype=float))
            T_base_bel = T_true @ inv_tf(error_tf(err))
            T_bel_hook0 = make_tf(Rotation.from_rotvec(np.deg2rad(pert[3:])).as_matrix()
                                  @ Rotation.from_quat(pol.start_quat, scalar_first=True).as_matrix(),
                                  (pol.start_pos_mm + pert[:3]) * 1e-3)
            T_start = T_base_bel @ T_bel_hook0 @ inv_tf(T_fh)
            T_lift = stations(T_base_bel, T_bel_hook0, T_fh, [(0, 0, 0)], args.lift_mm)[0]
            log(f"\n=== episode {ep}: belief error {np.round(err, 1)} (mm, deg), start perturbation {np.round(pert, 1)}")
            if not ask(f"move (LIN, speed {args.approach_speed}) to {args.lift_mm:.0f} mm above the start?"):
                break
            io.move_to(T_lift, args.approach_speed)
            if ep == 0 and args.calibrate:
                log("calibrating the load model (free space, the hook turned about its crook)")
                calibrate(io, wm, stations(T_base_bel, T_bel_hook0, T_fh, CALIB_ROTVECS_DEG, args.lift_mm),
                          args.approach_speed, log)
                wm.save(out_dir / "wrench_model.json")
                io.move_to(T_lift, args.approach_speed)
            if ep == 0 and args.axes_test:
                log("axes test: half-scale actions in free space; rotations must leave the crook in place")
                axes_test(io, pol, T_fh, T_base_bel, log=log)
                io.move_to(T_lift, args.approach_speed)
            if not ask("descend to the start pose?"):
                break
            io.move_to(T_start, args.approach_speed)
            size, smp0 = tare(io, wm)
            if size is None:
                log("tare refused: the reading at the start is too large (contact? stale load model?); stopping")
                break
            log(f"tare: residual {size[0]:.2f} N, {size[1]:.3f} N m")
            if not ask(f"run the policy (up to {args.max_steps} steps)?"):
                break

            def confirm(k, delta, T_bel_hook, w_hook):
                return input(f"  step {k}: delta {np.round(delta, 1)} (mm, deg, bel axes); Enter to execute, q to "
                             f"stop: ").strip().lower() != "q"

            t0 = time.time()
            end, steps, chunks = run_episode(pol, io, T_fh, T_base_bel, wm, guard, args.max_steps, smp0,
                                             confirm if args.step else None, log)
            T_end = io.flange()
            seated, seat, rot, link = pol.judge(inv_tf(T_true) @ T_end @ T_fh)
            summary = {"episode": ep, "end": end, "steps": len(steps), "wall_s": time.time() - t0,
                       "belief_error": err.tolist(), "start_perturbation": pert.tolist(), "seated": seated,
                       "seat_mm": seat, "rotation_deg": rot, "linking": link, "tare": size}
            if end == "declared done" and args.pull_test:
                summary["pull_caught"], summary["pull_moved_mm"], summary["pull_fz_n"] = pull_test(io, wm, T_fh)
                if summary["pull_caught"] and summary["pull_fz_n"] > 0.3:
                    log("WARNING: hook caught but the wire pushes it up: the wrench sign looks flipped (--wrench-sign)")
            log(json.dumps(summary))
            np.savez(out_dir / f"episode_{ep:03d}.npz",
                     **{k: np.array([s[k] for s in steps]) for k in ("T_bf", "q", "x", "action", "w_hook", "wall")},
                     **({f"chunk_{k}": np.concatenate([c[k] for c in chunks]) for k in ("t", "T", "raw", "w_hook", "step")}
                        if chunks else {}),
                     T_base_bel=T_base_bel, T_true=T_true, T_fh=T_fh, wrench_theta=wm.theta, tare=wm.tare)
            with open(out_dir / "episodes.jsonl", "a") as f:
                f.write(json.dumps(summary) + "\n")
            if ask("retrace the episode's path back to the start and lift?"):
                back = [s["T_bf"] for s in steps[::-1][::2]] + [T_start, T_lift]
                io.move_path(back, speed=args.approach_speed)
            elif not ask("reset by hand done (robot holds its pose); continue?"):
                break
    finally:
        io.close()
        logf.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    ex = sub.add_parser("export")
    ex.add_argument("checkpoint", type=Path)
    ex.add_argument("--out", type=Path, required=True)

    def common(p):
        p.add_argument("--max-steps", type=int, default=300)
        p.add_argument("--impedance", default="LOW", help="kuka_sunrise SunriseImpedanceValues preset")
        p.add_argument("--joint-margin", type=float, default=JOINT_MARGIN_DEG, help="deg")
        p.add_argument("--force-limit", type=float, default=FORCE_LIMIT_N, help="N, on the hook")
        p.add_argument("--torque-limit", type=float, default=TORQUE_LIMIT_NM, help="N m, about the hook's origin")
        p.add_argument("--no-pull-test", dest="pull_test", action="store_false")

    sm = sub.add_parser("sim")
    sm.add_argument("policy")
    sm.add_argument("--eval-set", type=Path, required=True)
    sm.add_argument("--cases", type=parse_cases)
    sm.add_argument("--no-error", action="store_true")
    sm.add_argument("--out-dir", type=Path, required=True)
    sm.add_argument("--workers", type=int, default=10)
    sm.add_argument("--cell", default="swing", help="kuka_hook_sim.CELLS: swing, hook_v6 (the hardware's mount)")
    sm.add_argument("--flange-to-hook", type=float, nargs=6, help="mount override: KUKA pose mm, deg (cell_def)")
    sm.add_argument("--success-flange", type=float, nargs=6, help="flange pose with the hook assembled (cell_def)")
    sm.add_argument("--gripper-box", type=float, nargs=6, action="append", metavar=("CX", "CY", "CZ", "HX", "HY", "HZ"),
                    help="gripper collision box in the flange frame, centre and half sizes mm (repeat; cell_def)")
    sm.add_argument("--tcp", choices=["hook", "flange"], default="hook",
                    help="impedance centre: hook (a Sunrise tool frame at the crook) or flange (lbr.move)")
    sm.add_argument("--chunk-s", type=float, default=0.6)
    sm.add_argument("--settle-s", type=float, default=0.4)
    sm.add_argument("--calibrate", action="store_true", help="fit the load model before the first episode")
    sm.add_argument("--sensor-sign", type=float, default=1.0, help="the simulated Sunrise's sign convention")
    sm.add_argument("--assume-sign", type=float, default=1.0, help="the loop's WrenchModel sign")
    sm.add_argument("--sensor-load", type=float, nargs=4, default=[0.0] * 4, metavar=("KG", "CX", "CY", "CZ"),
                    help="unmodelled load: mass kg, centre mm (flange frame)")
    sm.add_argument("--sensor-bias", type=float, nargs=6, default=[0.0] * 6)
    sm.add_argument("--sensor-noise", type=float, nargs=2, default=[0.0, 0.0], metavar=("N", "NM"))
    sm.add_argument("--compare", type=Path, help="kuka_hook_sim.py eval episodes.csv of the same cases")
    sm.add_argument("--start-seed", choices=["assembled", "continuation"], default="assembled",
                    help="IK seed of the starts: the assembled configuration (kuka_hook_sim's) or the one reached by "
                         "following the reference path back from it")
    sm.add_argument("--no-limit-avoidance", action="store_true",
                    help="drop kuka_hook_sim's null-space push away from joint limits (Sunrise has none)")
    common(sm)
    sm.set_defaults(max_steps=300)

    rb = sub.add_parser("robot")
    rb.add_argument("policy")
    rb.add_argument("--assembly", default="hook_v6", help="entry of sunrise-wrapper-abhay config/assembly_config.json")
    rb.add_argument("--flange-to-hook", type=float, nargs=6, help="override: KUKA pose mm, deg")
    rb.add_argument("--success-flange", type=float, nargs=6,
                    help="flange pose (KUKA mm, deg) with the hook assembled, taught with this mount")
    rb.add_argument("--base-to-eyelet", type=float, nargs=6, help="override, true eyelet: KUKA pose mm, deg")
    rb.add_argument("--system-ip", default="192.168.10.150")
    rb.add_argument("--robot-ip", default="192.168.10.122")
    rb.add_argument("--port", type=int, default=30015)
    rb.add_argument("--speed", type=float, default=0.1, help="Sunrise joint velocity (relative) of policy chunks")
    rb.add_argument("--approach-speed", type=float, default=0.1)
    rb.add_argument("--lift-mm", type=float, default=LIFT_MM)
    rb.add_argument("--episodes", type=int, default=1)
    rb.add_argument("--error", type=float, nargs=6, default=[0.0] * 6, help="belief error: mm, rotation vector deg")
    rb.add_argument("--random-error", type=float, default=0.0, help="uniform within this fraction of the trained bounds")
    rb.add_argument("--start-perturb", type=float, nargs=6, default=[0.0] * 6, help="start offset in bel: mm, deg")
    rb.add_argument("--random-start", action="store_true", help="uniform within training's NOMINAL_PERTURB")
    rb.add_argument("--seed", type=int, default=0)
    rb.add_argument("--wrench-model", type=Path, help="a WrenchModel json (from --calibrate)")
    rb.add_argument("--wrench-sign", type=float, default=1.0)
    rb.add_argument("--calibrate", action="store_true")
    rb.add_argument("--axes-test", action="store_true")
    rb.add_argument("--check-only", action="store_true", help="connect, report the state, do not move")
    rb.add_argument("--step", action="store_true", help="confirm every policy step")
    rb.add_argument("--yes", action="store_true", help="skip the confirmations (not --step's)")
    rb.add_argument("--fake-sunrise", action="store_true",
                    help="rehearse without the robot: a kinematic stand-in for ROS2ClientV2 on 127.0.0.1")
    rb.add_argument("--out-dir", type=Path)
    common(rb)
    rb.set_defaults(max_steps=150)
    args = parser.parse_args()
    {"export": cmd_export, "sim": cmd_sim, "robot": cmd_robot}[args.cmd](args)


if __name__ == "__main__":
    main()
