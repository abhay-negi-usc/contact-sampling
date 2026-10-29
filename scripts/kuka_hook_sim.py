#!/usr/bin/env python3
"""Hook/eyelet assembly policies (train_hook_rl.py, declared completion) run on a simulated KUKA LBR iiwa 14 R820.

Cell (CELLS): a gripper holds the hook at the flange, the hook's frame (this repo's: origin the inside of the crook,
shaft along +z) at flange_to_hook; the eyelet stands on a small mounting block at its foot (FLOOR_Z, as in
training) on a post. The hardware's peg and hole frames (sunrise-wrapper-abhay config/assembly_config.json) are
this repo's hook and eyelet frames: the reference trajectories are the same. "hook_v6" is the hardware cell;
"swing" (default) mounts the hook so the policy's threading swing turns A7 instead of bending the wrist, and places
the eyelet where the arm can follow most of the policy's paths (CELLS comment). KUKA poses x, y, z, A, B, C mean
Rz(A) Ry(B) Rx(C), as kuka_sunrise does.

Robot: MuJoCo Menagerie's iiwa 14 (joint torques, gravity on) under Cartesian impedance control at the tool centre
point (the hook's origin, --tcp; Sunrise moving a tool frame), as Sunrise's CART_IMP mode:
tau = J^T (K e + D (v_d - v)) + bias (gravity and Coriolis, the robot's model) + null space (the episode's start
posture, and a push away from joint limits within LIMIT_ZONE); stiffness per axis from kuka_sunrise's presets (LOW,
210 N/m and 13.5 N m/rad, is near the training gripper's ~250 N/m; HOOK_LOW, 1000 N/m and 25 N m/rad, is what the
hardware hook runs used); damping ratio 0.7, factorization design D = zeta (Lambda^1/2 K^1/2 + K^1/2 Lambda^1/2),
Lambda the tcp's operational-space inertia (a diagonal design over-damped the light wrist and went unstable at
1 kHz). A joint within LIMIT_FAULT of its limit stops the episode (a Sunrise limit fault).

Policy and robot, per policy step:
  observation: the hook's pose in the believed eyelet frame, T_bel^-1 T_base_flange T_flange_hook (forward
    kinematics; T_bel is the robot's belief of the eyelet in its base frame), position / 50 mm and two rotation
    columns; the external wrench at the flange as the iiwa estimates it from its joint torques,
    F = pinv(J_flange^T) tau_ext, moved to the hook's origin and expressed in the hook frame (viewed as the policy
    was trained to: hook_rl_env.wrench_features;
    --wrench ft: an ideal force sensor between flange and hook instead); the last motion action. The policy reads
    only these (its actor_dim inputs); the critic's privileged inputs are zeros.
  action chunk: a delta pose in the believed eyelet frame about the hook's origin (x, y, z mm, rotation vector deg,
    times max_delta) moves the hook's measured pose, as in training (the target restarts from the hook every
    step): T_hook_target (T_hook_target T_flange_hook^-1 for the flange). The robot executes it as impedance set
    points along the straight line (slerp in rotation) with a minimum-jerk profile over --chunk-s, then holds
    --settle-s. The training gripper (17 g on a soft weld) completes ~95% of a step in its 50 ms; the arm, with
    its inertia under compliant control, needs ~1 s to do that, and the policy, which sees no velocities, runs
    quasi-statically: commanding its full 7.5 mm / 12 deg steps in 50 ms swung the arm about the hook and sent the
    crook the wrong way. Declaring done (the seventh value > 0) holds the pose and ends the episode.
Outcome, judged against the true eyelet as in eval_hook_rl.py: declared done while seated (origin within SEAT_MM,
linking number above SUCCESS_LINK), then the pull test (the hook let go and pulled up, in the training model). A
hook leaving BOUNDS_MM is out of bounds. Starts come from the training model's evaluation sets, valid for the hook
alone: one the arm cannot reach (IK) or whose gripper or arm meets the fixture is reported, not run.

    conda run -n mujoco_env python scripts/kuka_hook_sim.py eval CHECKPOINT --eval-set SET.npz --cases 0-99 \\
        --out-dir OUT [--compare RL4/episodes.csv]
    conda run -n mujoco_env python scripts/kuka_hook_sim.py video --run OUT --cases 0-9
"""

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import csv
import json
import multiprocessing as mp
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import torch
from scipy.spatial.transform import Rotation, Slerp

import hook_rl_env as E
from eval_hook_rl import PULL_POS_MM, load_policy, outcome, wilson
from hook_eyelet_model import ASSET_DIR, COLLISION_GROUP, EYELET_YAW_DEG, HOOK_YAW_DEG, INCH, convex_parts, yaw_quat

IIWA_XML = Path("/home/rp/abhay_ws/peg_in_hole_dm_env/assets/kuka_iiwa_14/iiwa14.xml")  # MuJoCo Menagerie
# Cells (KUKA poses, mm and deg): the hook on the flange (flange_to_hook), the gripper (boxes in the flange frame:
# centre, half sizes, mm) and the true eyelet in the robot's base (base_to_eyelet, or where the hook was when
# assembled on the hardware). "hook_v6" is the hardware's: the shaft along the flange axis, so the threading swing
# about the hook's y axis (up to ~120 deg in the +-20 policy's paths) bends the wrist (A6, +-120 deg) and the arm
# reaches joint limits on most paths. "swing" turns the hook so that its y axis lies along the flange axis (A7,
# +-175 deg, makes the swing), the shaft held across the fingers 115 mm out; with the eyelet where a search of
# placements put it, the arm follows 36 of 40 of the policy's training-model paths by inverse kinematics (the
# rest reach a joint limit, mostly the elbow's, A4: the elbow angle is set by the shoulder-wrist distance, and
# tilting the hook +-50 deg swings the wrist centre 0.24 m about it).
CELLS = {"hook_v6": {"flange_to_hook": [-17.5, -17.5, 167.3, 180.0, 0.0, 180.0],
                     "base_to_flange_assembled": [551.67, -254.1, 260.91, -84.49, -0.93, 179.51],
                     "gripper": [((-16.0, -10.9, 108.3), (8.0, 4.0, 9.0)), ((-16.0, -24.1, 108.3), (8.0, 4.0, 9.0)),
                                 ((-16.0, -17.5, 91.3), (10.0, 11.0, 8.0)), ((0.0, 0.0, 41.3), (35.0, 35.0, 42.0))]},
         "swing": {"flange_to_hook": [0.0, 60.0, 115.0, -180.0, 0.0, -90.0],
                   "base_to_eyelet": [560.0, 250.0, 100.0, 195.0, 0.0, 0.0],
                   "gripper": [((6.6, 0.0, 108.0), (4.0, 9.0, 12.0)), ((-6.6, 0.0, 108.0), (4.0, 9.0, 12.0)),
                               ((0.0, 0.0, 88.0), (12.0, 12.0, 8.0)), ((0.0, 0.0, 40.0), (35.0, 35.0, 40.0))]}}
CELL = CELLS["swing"]
IMPEDANCE = {"HOOK_LOW": [1000.0, 1000.0, 1000.0, 25.0, 25.0, 25.0],  # kuka_sunrise/common.py: N/m, N m/rad
             "HOOK_MEDIUM": [1500.0, 1500.0, 1000.0, 25.0, 100.0, 50.0],
             "LOW": [210.0, 210.0, 210.0, 13.5, 13.5, 13.5]}
DAMPING_RATIO = 0.7
CONTROL_DT = 0.001
POLICY_DT = E.SUBSTEPS * E.TIMESTEP  # 50 ms: the training gripper's step
NULL_KP, NULL_KD = 10.0, 2.0  # null-space posture, N m/rad, N m s/rad
LIMIT_ZONE, LIMIT_K = np.deg2rad(20.0), 30.0  # null-space push away from a joint limit within the zone, N m/rad
LIMIT_FAULT = np.deg2rad(1.0)  # a joint this close to its limit stops the robot (a Sunrise limit fault)
LOG_TICKS = 50  # joint angles logged for videos every 50 ms of robot time
PLATE_HALF_MM = (40.0, 40.0, 5.0)  # the eyelet's mounting block, on a post down to the table (POST_RADIUS_MM)
POST_RADIUS_MM = 15.0


class StartInCollision(RuntimeError):
    pass


def kuka_tf(pose):
    """4x4 transform (m) of a KUKA pose: x, y, z mm, A, B, C deg."""
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("ZYX", pose[3:], degrees=True).as_matrix()
    T[:3, 3] = np.asarray(pose[:3], dtype=np.float64) * 1e-3
    return T


def make_tf(R, p):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, p
    return T


def inv_tf(T):
    return make_tf(T[:3, :3].T, -T[:3, :3].T @ T[:3, 3])


def error_tf(err):
    """T_bel_true (m) of a belief error: the true eyelet's pose in the believed frame (mm, rotation vector deg)."""
    return make_tf(Rotation.from_rotvec(np.deg2rad(err[3:])).as_matrix(), np.asarray(err[:3]) * 1e-3)


def wxyz(R):
    return Rotation.from_matrix(R).as_quat(scalar_first=True)


def vec(v):
    return " ".join(f"{x:.9g}" for x in v)


def build_kuka_model(cell=CELL):
    """(MjModel, T_base_eyelet (true eyelet frame in the robot's base, m), T_flange_hook)."""
    root = ET.parse(IIWA_XML).getroot()
    compiler = root.find("compiler")
    meshdir = IIWA_XML.parent / compiler.attrib.pop("meshdir")
    for mesh in root.iter("mesh"):
        mesh.set("file", str(meshdir / mesh.get("file")))
    root.find("option").set("timestep", str(CONTROL_DT))
    root.remove(root.find("keyframe"))
    vis = ET.SubElement(root, "visual")
    ET.SubElement(vis, "global", offwidth="1280", offheight="960")
    ET.SubElement(vis, "headlight", ambient="0.3 0.3 0.3", diffuse="0.4 0.4 0.4")
    asset, default, world = root.find("asset"), root.find("default"), root.find("worldbody")
    asset_dir, s = ASSET_DIR.resolve(), f"{INCH} {INCH} {INCH}"
    contact = {"friction": f"{E.FRICTION} 0.005 0.0001", "solref": f"{E.CONTACT_TC} 1"}
    ET.SubElement(ET.SubElement(default, "default", {"class": "part_visual"}), "geom", type="mesh", contype="0",
                  conaffinity="0", group="0", density="0")
    ET.SubElement(ET.SubElement(default, "default", {"class": "part_collision"}), "geom", type="mesh",
                  group=str(COLLISION_GROUP), density="0", **contact)
    parts = {}
    for name in ("hook", "eyelet"):
        ET.SubElement(asset, "mesh", name=name, file=str(asset_dir / f"{name}.stl"), scale=s)
        parts[name] = []
        for part in convex_parts(asset_dir, name):
            ET.SubElement(asset, "mesh", name=part.stem, file=str(part), scale=s)
            parts[name].append(part.stem)

    T_flange_hook = kuka_tf(cell["flange_to_hook"])
    T_base_eyelet = (kuka_tf(cell["base_to_eyelet"]) if "base_to_eyelet" in cell
                     else kuka_tf(cell["base_to_flange_assembled"]) @ T_flange_hook)
    link7 = next(b for b in root.iter("body") if b.get("name") == "link7")
    site = next(x for x in link7.iter("site") if x.get("name") == "attachment_site")
    site.set("name", "flange")
    T_link7_hook = make_tf(np.eye(3), np.array([float(v) for v in site.get("pos").split()])) @ T_flange_hook
    hook = ET.SubElement(link7, "body", name="hook", pos=vec(T_link7_hook[:3, 3]), quat=vec(wxyz(T_link7_hook[:3, :3])))
    mass, com, full = E.hook_inertial()
    ET.SubElement(hook, "inertial", pos=vec(com), mass=f"{mass:.6g}", fullinertia=vec(full))
    ET.SubElement(hook, "site", name="hook", size="0.002", rgba="1 0 0 1", group="4")
    frame = ET.SubElement(hook, "frame", quat=yaw_quat(HOOK_YAW_DEG))
    ET.SubElement(frame, "geom", {"class": "part_visual", "name": "hook", "mesh": "hook", "rgba": "0.95 0.55 0.15 1"})
    for p in parts["hook"]:
        ET.SubElement(frame, "geom", {"class": "part_collision", "mesh": p, "rgba": "0.9 0.5 0.1 0.4"})
    T_hook_flange = inv_tf(T_flange_hook)
    for k, (c, h) in enumerate(cell["gripper"]):  # flange frame -> hook frame
        T = T_hook_flange @ make_tf(np.eye(3), np.array(c) * 1e-3)
        ET.SubElement(hook, "geom", name=f"gripper{k}", type="box", pos=vec(T[:3, 3]), quat=vec(wxyz(T[:3, :3])),
                      size=vec(np.array(h) * 1e-3), rgba="0.25 0.25 0.28 0.3", density="0", **contact)

    fixture = ET.SubElement(world, "body", name="eyelet_fixture", pos=vec(T_base_eyelet[:3, 3]),
                            quat=vec(wxyz(T_base_eyelet[:3, :3])))
    ET.SubElement(fixture, "site", name="eyelet", size="0.002", rgba="0 0 1 1", group="4")
    ET.SubElement(fixture, "geom", name="plate", type="box", rgba="0.75 0.75 0.75 1",
                  pos=vec([0, 0, (E.FLOOR_Z - PLATE_HALF_MM[2]) * 1e-3]), size=vec(np.array(PLATE_HALF_MM) * 1e-3),
                  **contact)
    post_top = (E.FLOOR_Z - 2 * PLATE_HALF_MM[2]) * 1e-3
    post_len = post_top + T_base_eyelet[2, 3]  # down to the table (base z = 0; the fixture stands upright)
    if post_len > 0:
        ET.SubElement(fixture, "geom", name="post", type="cylinder", rgba="0.6 0.6 0.62 1",
                      pos=vec([0, 0, post_top - post_len / 2]), size=vec([POST_RADIUS_MM * 1e-3, post_len / 2]),
                      **contact)
    frame = ET.SubElement(fixture, "frame", quat=yaw_quat(EYELET_YAW_DEG))
    ET.SubElement(frame, "geom", {"class": "part_visual", "name": "eyelet", "mesh": "eyelet",
                                  "rgba": "0.25 0.45 0.85 1"})
    for p in parts["eyelet"]:
        ET.SubElement(frame, "geom", {"class": "part_collision", "mesh": p, "rgba": "0.2 0.4 0.8 0.4"})
    ET.SubElement(world, "geom", name="table", type="plane", pos="0 0 -0.001", size="1.5 1.5 0.01",
                  rgba="0.55 0.55 0.6 1", contype="0", conaffinity="0")
    xml = ET.tostring(root, encoding="unicode")
    return mujoco.MjModel.from_xml_string(xml), T_base_eyelet, T_flange_hook


class KukaCell:
    """The robot, the hooked gripper and the eyelet; Cartesian impedance at a tool centre point (tcp: the hook's
    frame, or the flange); the policy interface (module docstring)."""

    def __init__(self, impedance="HOOK_LOW", wrench="joint_torque", wrench_noise=(0.0, 0.0), seed=0,
                 max_delta=E.MAX_DELTA, setpoint="measured", max_dev=None, feedforward=True, tcp="hook",
                 chunk_s=0.6, settle_s=0.4, cell=CELL):
        self.setpoint, self.feedforward = setpoint, feedforward
        self.chunk_ticks, self.settle_ticks = round(chunk_s / CONTROL_DT), round(settle_s / CONTROL_DT)
        self.max_dev = None if max_dev is None else np.asarray(max_dev, dtype=np.float64)
        self.m, self.T_base_eyelet, self.T_flange_hook = build_kuka_model(cell)
        self.d = mujoco.MjData(self.m)
        self.T_hook_flange = inv_tf(self.T_flange_hook)
        self.flange = self.m.site("flange").id
        self.tcp = self.m.site(tcp).id
        self.T_hook_tcp = np.eye(4) if tcp == "hook" else self.T_hook_flange
        self.T_tcp_hook = inv_tf(self.T_hook_tcp)
        self.hook_body = self.m.body("hook").id
        self.robot_bodies = {self.m.body(f"link{k}").id for k in range(1, 8)} | {self.hook_body}
        self.fixture_body = self.m.body("eyelet_fixture").id
        self.ctrl_lo, self.ctrl_hi = self.m.actuator_ctrlrange.T
        self.q_lo, self.q_hi = self.m.jnt_range.T
        self.K = np.asarray(IMPEDANCE[impedance], dtype=np.float64)
        self.wrench_source, self.wrench_noise = wrench, np.repeat(np.asarray(wrench_noise, dtype=np.float64), 3)
        self.wrench_obs = "absolute"  # the policy's view of the wrench (hook_rl_env wrench_features), as trained
        self.rng = np.random.default_rng(seed)
        self.max_delta = np.asarray(max_delta, dtype=np.float64)
        self.geo = E.Geometry()
        self.jacp, self.jacr = np.zeros((3, 7)), np.zeros((3, 7))
        self.M = np.zeros((7, 7))
        self.q_ref = self.ik(self.T_base_eyelet @ self.T_hook_flange,
                             np.array([0.235, 0.716, -0.307, -1.831, 0.335, 0.646, -0.270]))  # Menagerie "home"
        self.q_post = self.q_ref.copy()
        self.log = None
        self.rl_m = E.build_rl_model()  # for the pull test

    # ------------------------------------------------------------ kinematics
    def site_tf(self, site, d=None):
        d = self.d if d is None else d
        return make_tf(d.site_xmat[site].reshape(3, 3), d.site_xpos[site].copy())

    def site_jac(self, site, d=None):
        """(6, 7) Jacobian of a site: linear velocity of its point, angular velocity (base frame)."""
        d = self.d if d is None else d
        mujoco.mj_jacSite(self.m, d, self.jacp, self.jacr, site)
        return np.vstack([self.jacp, self.jacr])

    def flange_tf(self, d=None):
        return self.site_tf(self.flange, d)

    def jacobian(self, d=None):
        return self.site_jac(self.flange, d)

    def ik(self, T_target, q_seed, iters=1000, tries=40):
        """Joint angles putting the flange at T_target within 0.02 mm and 0.01 deg (damped least squares, posture
        toward the seed in the null space, joint limits): from q_seed, else the wrist turned half a turn, else
        random perturbations of q_seed, the first that converges. Raises if none does."""
        rng = np.random.default_rng(0)
        seeds = [q_seed] + [q_seed + np.r_[0, 0, 0, 0, 0, 0, s] for s in (np.pi, -np.pi)]
        seeds += [q_seed + rng.normal(0, 0.5, 7) for _ in range(tries)]
        best_e = None
        for seed in seeds:
            q, e = self._ik(T_target, np.clip(seed, self.q_lo + 1e-3, self.q_hi - 1e-3), iters)
            if q is not None:
                return q
            if best_e is None or np.linalg.norm(e) < np.linalg.norm(best_e):
                best_e = e
        raise RuntimeError(f"IK did not converge: {np.linalg.norm(best_e[:3]) * 1e3:.2f} mm, "
                           f"{np.degrees(np.linalg.norm(best_e[3:])):.2f} deg")

    def _ik(self, T_target, q_seed, iters):
        d = mujoco.MjData(self.m)
        q = q_seed.copy()
        for _ in range(iters):
            d.qpos[:] = q
            mujoco.mj_kinematics(self.m, d)
            mujoco.mj_comPos(self.m, d)
            T = self.flange_tf(d)
            e = np.concatenate([T_target[:3, 3] - T[:3, 3],
                                Rotation.from_matrix(T_target[:3, :3] @ T[:3, :3].T).as_rotvec()])
            if np.linalg.norm(e[:3]) < 2e-5 and np.linalg.norm(e[3:]) < 1.7e-4:
                return q, e
            J = self.jacobian(d)
            JJ = J @ J.T + 1e-4 * np.eye(6)
            dq = J.T @ np.linalg.solve(JJ, e)
            dq += (np.eye(7) - np.linalg.pinv(J) @ J) @ (0.1 * (q_seed - q))
            q = np.clip(q + dq, self.q_lo + 1e-3, self.q_hi - 1e-3)
        return None, e

    # ------------------------------------------------------------ control
    def impedance_torque(self, T_d, v_d):
        """Cartesian impedance at the tcp toward set point T_d moving at v_d (6: m/s, rad/s), plus the robot's bias
        forces and posture: tau = J^T (K e + D (v_d - v)) + bias + null-space posture."""
        d = self.d
        T = self.site_tf(self.tcp)
        J = self.site_jac(self.tcp)
        mujoco.mj_fullM(self.m, self.M, d.qM)
        Minv = np.linalg.inv(self.M)
        Lam = np.linalg.inv(J @ Minv @ J.T)
        e = np.concatenate([T_d[:3, 3] - T[:3, 3], Rotation.from_matrix(T_d[:3, :3] @ T[:3, :3].T).as_rotvec()])
        w, V = np.linalg.eigh(Lam)  # factorization damping: D = zeta (Lam^1/2 K^1/2 + K^1/2 Lam^1/2)
        L = (V * np.sqrt(np.maximum(w, 0.0))) @ V.T
        Ks = np.diag(np.sqrt(self.K))
        D = DAMPING_RATIO * (L @ Ks + Ks @ L)
        F = self.K * e + D @ (v_d - J @ d.qvel)
        Jbar = Minv @ J.T @ Lam
        q = d.qpos
        away = (np.maximum(0.0, LIMIT_ZONE - (q - self.q_lo)) - np.maximum(0.0, LIMIT_ZONE - (self.q_hi - q)))
        null = (np.eye(7) - J.T @ Jbar.T) @ (NULL_KP * (self.q_post - q) - NULL_KD * d.qvel + LIMIT_K * away)
        return np.clip(J.T @ F + d.qfrc_bias + null, self.ctrl_lo, self.ctrl_hi)

    def clamp(self, T_d):
        """T_d held within max_dev (mm, deg) of the tcp's measured pose (Sunrise's max path deviation): (pose,
        whether it was clamped)."""
        T = self.site_tf(self.tcp)
        dp = T_d[:3, 3] - T[:3, 3]
        dr = Rotation.from_matrix(T_d[:3, :3] @ T[:3, :3].T).as_rotvec()
        lim_p, lim_r = self.max_dev[0] * 1e-3, np.deg2rad(self.max_dev[1])
        n_p, n_r = np.linalg.norm(dp), np.linalg.norm(dr)
        if n_p <= lim_p and n_r <= lim_r:
            return T_d, False
        dp = dp * min(1.0, lim_p / max(n_p, 1e-12))
        dr = dr * min(1.0, lim_r / max(n_r, 1e-12))
        return make_tf(Rotation.from_rotvec(dr).as_matrix() @ T[:3, :3], T[:3, 3] + dp), True

    def run_chunk(self, T_from, T_to):
        """Stream set points from T_from to T_to (tcp) over the chunk's duration along the straight line (slerp in
        rotation) with a minimum-jerk profile, starting and ending at rest, with the profile's velocity as the
        damping reference (feedforward); each held within the max path deviation of the measured pose if one is
        set. Returns the last set point."""
        slerp = Slerp([0.0, 1.0], Rotation.from_matrix([T_from[:3, :3], T_to[:3, :3]]))
        n = self.chunk_ticks
        span = np.concatenate([T_to[:3, 3] - T_from[:3, 3],
                               Rotation.from_matrix(T_to[:3, :3] @ T_from[:3, :3].T).as_rotvec()])
        for k in range(1, n + 1):
            tau = k / n
            s = tau ** 3 * (10 - 15 * tau + 6 * tau ** 2)
            v_d = span * 30 * tau ** 2 * (1 - tau) ** 2 / (n * CONTROL_DT) if self.feedforward else np.zeros(6)
            T_d = make_tf(slerp(s).as_matrix(), (1 - s) * T_from[:3, 3] + s * T_to[:3, 3])
            if self.max_dev is not None:
                T_d, clamped = self.clamp(T_d)
                if clamped:
                    v_d = np.zeros(6)
            self.sim_step(T_d, v_d)
        for _ in range(self.settle_ticks):  # hold the chunk's end so the robot settles before the next observation
            self.sim_step(T_d, np.zeros(6))
        return T_d

    def sim_step(self, T_d, v_d):
        """One control tick; every LOG_TICKS the joint angles go to self.log (a list, when set)."""
        self.d.ctrl[:] = self.impedance_torque(T_d, v_d)
        mujoco.mj_step(self.m, self.d)
        self.tick += 1
        if self.log is not None and self.tick % LOG_TICKS == 0:
            self.log.append((self.steps, self.d.qpos.copy()))

    # ------------------------------------------------------------ sensing
    def contact_wrench_flange(self):
        """External wrench at the flange (base frame: force N, torque N m about the flange), as the iiwa estimates
        it: the contacts' joint torques tau_ext mapped back through the flange Jacobian, pinv(J^T) tau_ext. Also
        whether any robot link other than link7 and the hook touches something."""
        d, f6, jac = self.d, np.zeros(6), np.zeros((3, 7))
        tau, link_contact = np.zeros(7), False
        for k in range(d.ncon):
            c = d.contact[k]
            b1, b2 = self.m.geom_bodyid[c.geom1], self.m.geom_bodyid[c.geom2]
            if (b1 in self.robot_bodies) == (b2 in self.robot_bodies):
                continue
            mujoco.mj_contactForce(self.m, d, k, f6)
            force = c.frame.reshape(3, 3).T @ f6[:3]  # on geom2's body
            body = b2 if b2 in self.robot_bodies else b1
            if body == b1:
                force = -force
            link_contact |= body not in (self.hook_body, self.m.body("link7").id)
            mujoco.mj_jac(self.m, d, jac, None, c.pos, body)
            tau += jac.T @ force
        return np.linalg.pinv(self.jacobian().T) @ tau, link_contact

    def hook_wrench(self):
        """The wrench the policy sees: at the hook's origin, in the hook frame (N, N m), and link contact."""
        T_bf = self.flange_tf()
        T_bh = T_bf @ self.T_flange_hook
        if self.wrench_source == "ft":  # an ideal force sensor between flange and hook: the hook's own contacts
            w = E.hook_wrench(self.m, self.d, self.hook_body)  # torque about the hook body's origin = hook frame
            f, m_h, link_contact = w[:3], w[3:], False
        else:
            w, link_contact = self.contact_wrench_flange()
            f, m_h = w[:3], w[3:] + np.cross(T_bf[:3, 3] - T_bh[:3, 3], w[:3])
        R = T_bh[:3, :3]
        out = np.concatenate([R.T @ f, R.T @ m_h])
        if self.wrench_noise.any():
            out = out + self.rng.normal(0.0, self.wrench_noise)
        return out, link_contact

    # ------------------------------------------------------------ episodes
    def reset(self, start_pos_mm, start_quat, err):
        """Start from a hook pose in the believed eyelet frame (mm, wxyz), the belief off by err (the true eyelet
        in the believed frame: mm, rotation vector deg). The believed eyelet becomes T_bel (robot base frame)."""
        self.err = np.asarray(err, dtype=np.float64)
        self.T_bel = self.T_base_eyelet @ inv_tf(error_tf(self.err))
        T_bel_hook = make_tf(Rotation.from_quat(start_quat, scalar_first=True).as_matrix(),
                             np.asarray(start_pos_mm) * 1e-3)
        q = self.ik(self.T_bel @ T_bel_hook @ self.T_hook_flange, self.q_ref)
        mujoco.mj_resetData(self.m, self.d)
        self.d.qpos[:] = q
        mujoco.mj_forward(self.m, self.d)
        for k in range(self.d.ncon):  # the gripper and arm, not modelled in training, can meet the fixture
            c = self.d.contact[k]
            b1, b2 = self.m.geom_bodyid[c.geom1], self.m.geom_bodyid[c.geom2]
            if (b1 in self.robot_bodies) != (b2 in self.robot_bodies):
                g = c.geom1 if b1 in self.robot_bodies else c.geom2
                raise StartInCollision(f"{self.m.geom(g).name or self.m.body(self.m.geom_bodyid[g]).name} "
                                       f"touches the fixture")
        self.q_post = q.copy()  # the null-space posture: the episode's start (another IK branch than q_ref may
        # need a self-motion through joint limits to reach q_ref)
        self.T_sp = self.site_tf(self.tcp)
        self.prev_action = np.zeros(6)
        self.steps = 0
        self.tick = 0
        self.fault = None

    def hook_in(self, T_ref):
        """The hook's pose in frame T_ref (base frame): (position mm, rotation matrix)."""
        T = inv_tf(T_ref) @ self.flange_tf() @ self.T_flange_hook
        return T[:3, 3] * 1e3, T[:3, :3]

    def observe(self, obs_dim):
        """The policy's observation (obs_dim values; only the actor's first STOP_ACTOR_DIM are filled)."""
        pos, R = self.hook_in(self.T_bel)
        w, link_contact = self.hook_wrench()
        obs = np.zeros(obs_dim, np.float32)
        obs[:E.STOP_ACTOR_DIM] = np.concatenate([pos / 50.0, R[:, :2].T.reshape(6),
                                                 E.wrench_features(w, self.wrench_obs), self.prev_action])
        return obs, w, link_contact

    def step(self, action):
        """Execute one policy action (7 values) as a chunk of impedance set points; True if it declared done.
        setpoint "measured": the delta moves the hook's measured pose (training's rule: the target restarts from
        the hook every step); "continuous": it moves the last commanded pose, the chunks joining into one path
        (as kuka_sunrise streams them), held within the max path deviation."""
        a = np.clip(np.asarray(action, dtype=np.float64), -1, 1)
        stop = a.shape[0] == 7 and a[6] > 0
        delta = np.zeros(6) if stop else a[:6] * self.max_delta
        T_from = self.site_tf(self.tcp) if self.setpoint == "measured" else self.T_sp
        T_bh = T_from @ self.T_tcp_hook
        R_bel = self.T_bel[:3, :3]
        T_target_hook = make_tf(Rotation.from_rotvec(R_bel @ np.deg2rad(delta[3:])).as_matrix() @ T_bh[:3, :3],
                                T_bh[:3, 3] + R_bel @ delta[:3] * 1e-3)
        self.T_sp = self.run_chunk(T_from, T_target_hook @ self.T_hook_tcp)
        self.prev_action = np.zeros(6) if stop else a[:6]
        self.steps += 1
        margin = np.minimum(self.d.qpos - self.q_lo, self.q_hi - self.d.qpos)
        if margin.min() < LIMIT_FAULT:
            self.fault = f"joint limit A{int(np.argmin(margin)) + 1}"
        return stop

    def judge(self):
        """Against the true eyelet: (seated, linking number, max pose-point offset mm, origin offset mm, out)."""
        pos, R = self.hook_in(self.T_base_eyelet)
        pts, link = self.geo.features(pos[None], R[None])
        kp = float(np.linalg.norm(pts[0] - self.geo.points, axis=1).max())
        seat = float(np.linalg.norm(pos))
        out = bool(np.any((pos < E.BOUNDS_MM[:, 0]) | (pos > E.BOUNDS_MM[:, 1])))
        return seat < E.SEAT_MM and link[0] > E.SUCCESS_LINK, float(link[0]), kp, seat, out

    def pull_test(self):
        """The training model's pull test from the hook's pose in the true eyelet frame: (ok, final offset mm)."""
        pos, R = self.hook_in(self.T_base_eyelet)
        d = mujoco.MjData(self.rl_m)
        q = np.concatenate([pos * 1e-3, wxyz(R)])
        d.qpos[:7], d.mocap_pos[0], d.mocap_quat[0] = q, q[:3], q[3:]
        p, qq = E.pull_test(self.rl_m, d, self.rl_m.body("hook").id)
        link = self.geo.features(p[None] * 1e3, E.quat_to_mat(qq[None]))[1][0]
        off = float(np.linalg.norm(p) * 1e3)
        return bool(off < PULL_POS_MM and link > E.SUCCESS_LINK), off


def run_episode(cell, ac, case, max_steps, keep=False):
    """One episode from case (start pos mm, quat, err): its record (and per-step trajectory if keep)."""
    obs_dim = ac.norm.mean.shape[0]
    cell.reset(*case)
    cell.log = [(0, cell.d.qpos.copy())] if keep else None
    traj = {"q": [cell.d.qpos.copy()], "wrench": [], "action": []}
    stop, peak_force, link_contact_any, t0 = False, 0.0, False, time.perf_counter()
    for _ in range(max_steps):
        obs, w, link_contact = cell.observe(obs_dim)
        peak_force = max(peak_force, float(np.linalg.norm(w[:3])))
        link_contact_any |= link_contact
        with torch.no_grad():
            a = ac.mean(ac.norm(torch.as_tensor(obs[None])))[0].numpy()
        stop = cell.step(a)
        if keep:
            traj["q"].append(cell.d.qpos.copy())
            traj["wrench"].append(w)
            traj["action"].append(a)
        seated, link, kp, seat, out = cell.judge()
        if stop or out or cell.fault:
            break
    rec = {"steps": cell.steps, "stopped": bool(stop), "seated": bool(seated), "out": bool(out),
           "success": bool(stop and seated and not cell.fault), "final_link": link, "final_kp_mm": kp,
           "final_seat_mm": seat, "fault": cell.fault, "peak_force_n": peak_force,
           "robot_link_contact": bool(link_contact_any), "pull_ok": None,
           "sim_s": time.perf_counter() - t0, **{f"err0_{a}": float(v) for a, v in zip("x y z rx ry rz".split(), case[2])}}
    rec["robot_time_s"] = cell.tick * CONTROL_DT
    if keep:
        traj["log_step"] = [k for k, _ in cell.log]
        traj["log_q"] = [q for _, q in cell.log]
        cell.log = None
    if rec["success"]:
        rec["pull_ok"], rec["pull_pos_mm"] = cell.pull_test()
    rec["outcome"] = f"robot fault: {cell.fault}" if cell.fault else outcome(rec)
    return rec, ({k: np.array(v) for k, v in traj.items()} if keep else None)


def worker(job):
    ckpt, cases, ids, keep, kw, max_steps = job
    torch.set_num_threads(1)
    ac, trained = load_policy(ckpt)
    cell = KukaCell(**kw)
    cell.wrench_obs = trained.get("wrench_obs", "absolute")
    out = []
    for i, case in zip(ids, cases):
        try:
            rec, traj = run_episode(cell, ac, case, max_steps, keep=i in keep)
        except StartInCollision as e:
            rec, traj = {"outcome": f"start in collision: {e}", "success": False, "pull_ok": False}, None
        except RuntimeError as e:  # unreachable start
            rec, traj = {"outcome": f"no IK: {e}", "success": False, "pull_ok": False}, None
        rec["case"] = int(i)
        out.append((rec, traj))
    return out


def parse_cases(text):
    out = []
    for part in text.split(","):
        lo, _, hi = part.partition("-")
        out += list(range(int(lo), int(hi or lo) + 1))
    return out


def cmd_eval(args):
    z = np.load(args.eval_set)
    ids = args.cases or list(range(len(z["pos"])))
    err = np.zeros_like(z["err"]) if args.no_error else z["err"]
    cases = [(z["pos"][i], z["quat"][i], err[i]) for i in ids]
    keep = set(ids[:args.save_trajectories])
    kw = dict(impedance=args.impedance, wrench=args.wrench, wrench_noise=args.wrench_noise, setpoint=args.setpoint,
              max_dev=None if args.max_dev[0] <= 0 else args.max_dev, feedforward=not args.no_feedforward,
              tcp=args.tcp, chunk_s=args.chunk_s, settle_s=args.settle_s, cell=CELLS[args.cell])
    chunks = [(args.checkpoint, cases[w::args.workers], ids[w::args.workers], keep, kw, args.max_steps)
              for w in range(args.workers)]
    with mp.get_context("fork").Pool(args.workers) as pool:
        results = [r for part in pool.map(worker, chunks) for r in part]
    results.sort(key=lambda r: r[0]["case"])
    records = [r for r, _ in results]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "trajectories").mkdir(exist_ok=True)
    for rec, traj in results:
        if traj is not None:
            np.savez(args.out_dir / "trajectories" / f"case{rec['case']}.npz", **traj, err=err[rec["case"]])
    fields = sorted({k for r in records for k in r})
    with open(args.out_dir / "episodes.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(records)
    n_ok = sum(bool(r.get("pull_ok")) for r in records)
    ran = [r for r in records if "steps" in r]
    ok_times = [r["robot_time_s"] for r in ran if r.get("pull_ok")]
    summary = {"checkpoint": str(args.checkpoint), "eval_set": str(args.eval_set), "no_error": args.no_error,
               "impedance": args.impedance, "wrench": args.wrench, "wrench_noise": args.wrench_noise,
               "setpoint": args.setpoint, "max_dev": args.max_dev, "feedforward": not args.no_feedforward,
               "tcp": args.tcp, "chunk_s": args.chunk_s, "settle_s": args.settle_s, "cell": args.cell,
               "robot_fault": float(np.mean([bool(r["fault"]) for r in ran])),
               "episodes": len(records), "no_valid_start": len(records) - len(ran),
               "assembled_and_pull_ok_of_valid_starts": n_ok / max(len(ran), 1),
               "peak_force_over_50n": float(np.mean([r["peak_force_n"] > 50 for r in ran])),
               "assembled_and_pull_ok": n_ok / len(records), "ci95": wilson(n_ok, len(records)),
               "declared_done": float(np.mean([r["stopped"] for r in ran])),
               "false_done": float(np.mean([r["stopped"] and not r["seated"] for r in ran])),
               "out_of_bounds": float(np.mean([r["out"] for r in ran])),
               "robot_link_contact": float(np.mean([r["robot_link_contact"] for r in ran])),
               "robot_time_to_success_s_p10_50_90": np.percentile(ok_times, [10, 50, 90]).round(1).tolist()
               if ok_times else None,
               "steps_to_success_p10_50_90": np.percentile([r["steps"] for r in ran if r.get("pull_ok")],
                                                           [10, 50, 90]).tolist() if ok_times else None,
               "peak_force_n_p50_90": np.percentile([r["peak_force_n"] for r in ran], [50, 90]).round(2).tolist(),
               "outcomes": {o: sum(r["outcome"] == o for r in records) for o in sorted({r["outcome"] for r in records})}}
    if args.compare:
        with open(args.compare) as f:
            ref = {int(r["episode"]): r["pull_ok"] == "True" for r in csv.DictReader(f)}
        a = [bool(r.get("pull_ok")) for r in records]
        b = [ref[r["case"]] for r in records]
        summary["compare"] = {"file": str(args.compare), "reference_success": float(np.mean(b)),
                              "only_kuka": int(sum(x and not y for x, y in zip(a, b))),
                              "only_reference": int(sum(y and not x for x, y in zip(a, b)))}
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


# ---------------------------------------------------------------- video
VIEWS = [("robot cell", 135, -20, 1.15), ("hook and eyelet", 50, -25, 1.3)]  # label, azimuth, elevation, margin
GHOST_RGBA = np.array([0.2, 0.85, 0.35, 0.3], np.float32)


def cmd_video(args):
    from assembly_trajectory_visual_verification import fit_camera, geom_world_vertices, put_text
    summary = json.loads((args.run / "summary.json").read_text())
    with open(args.run / "episodes.csv") as f:
        recs = {int(r["case"]): r for r in csv.DictReader(f)}
    cell = KukaCell(cell=CELLS[summary["cell"]])  # kinematics only
    m, d = cell.m, cell.d
    W, H = args.tile
    renderer = mujoco.Renderer(m, H, W)
    eyelet_geom = m.geom("eyelet").id
    out = args.out or args.run / "videos" / f"kuka_cases{args.cases[0]}-{args.cases[-1]}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    import mediapy
    dt = LOG_TICKS * CONTROL_DT * args.speed  # robot time per frame
    with mediapy.VideoWriter(out, shape=(H + 60, 2 * W), fps=round(1 / (LOG_TICKS * CONTROL_DT))) as writer:
        for case in args.cases:
            if not (args.run / "trajectories" / f"case{case}.npz").exists():
                print(f"case {case}: {recs[case]['outcome']} (no trajectory, skipped)", flush=True)
                continue
            z = np.load(args.run / "trajectories" / f"case{case}.npz")
            rec = recs[case]
            T_bel = cell.T_base_eyelet @ inv_tf(error_tf(z["err"]))
            qs, steps = z["log_q"][::args.speed], z["log_step"][::args.speed]
            # cameras: the whole robot over the episode, and the hook's path about the eyelet
            robot_pts, hook_pts = [], []
            for q in qs[:: max(1, len(qs) // 10)]:
                d.qpos[:] = q
                mujoco.mj_kinematics(m, d)
                robot_pts.append(d.xpos[1:9].copy())
                hook_pts.append(geom_world_vertices(m, d, "hook"))
            eyelet = geom_world_vertices(m, d, "eyelet")
            framings = [np.vstack(robot_pts + [eyelet, np.zeros((1, 3))]), np.vstack(hook_pts + [eyelet])]
            cams = [fit_camera(f, az, el, m.vis.global_.fovy, W / H, margin=mg)
                    for f, (_, az, el, mg) in zip(framings, VIEWS)]
            e = z["err"]
            result = ("ASSEMBLED, declared done" if rec["pull_ok"] == "True" else rec["outcome"].upper())
            T_move = T_bel @ inv_tf(cell.T_base_eyelet)  # the believed eyelet: the true one moved by this
            for k in range(len(qs) + round(1.5 / (LOG_TICKS * CONTROL_DT))):
                d.qpos[:] = qs[min(k, len(qs) - 1)]
                mujoco.mj_kinematics(m, d)
                tiles = []
                for (label, *_), cam in zip(VIEWS, cams):
                    renderer.update_scene(d, camera=cam)
                    scn = renderer.scene
                    for g in [scn.geoms[i] for i in range(scn.ngeom)]:
                        if g.objtype == mujoco.mjtObj.mjOBJ_GEOM and g.objid == eyelet_geom and scn.ngeom < scn.maxgeom:
                            ghost = scn.geoms[scn.ngeom]
                            mujoco.mjv_initGeom(ghost, g.type, np.asarray(g.size, np.float64),
                                                T_move[:3, :3] @ g.pos + T_move[:3, 3],
                                                (T_move[:3, :3] @ g.mat.reshape(3, 3)).flatten().astype(np.float64),
                                                GHOST_RGBA)
                            ghost.dataid = g.dataid
                            scn.ngeom += 1
                    img = renderer.render().copy()
                    put_text(img, label, (8, 20), scale=0.5)
                    tiles.append(img)
                bar = np.zeros((60, 2 * W, 3), np.uint8)
                j = min(k, len(qs) - 1)
                put_text(bar, f"case {case}  robot time {j * dt:5.1f} s ({args.speed}x)  policy step {steps[j]:3d}   "
                         f"belief error x {e[0]:+.1f} y {e[1]:+.1f} z {e[2]:+.1f} mm, rot ({e[3]:+.1f}, {e[4]:+.1f}, "
                         f"{e[5]:+.1f}) deg   green: believed eyelet", (8, 22), scale=0.45)
                if k >= len(qs) - 1:
                    put_text(bar, result, (8, 48), scale=0.55,
                             color=(90, 230, 110) if rec["pull_ok"] == "True" else (255, 80, 80))
                writer.add_image(np.vstack([np.hstack(tiles), bar]))
            print(f"case {case}: {rec['outcome']}", flush=True)
    print(f"wrote {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    ev = sub.add_parser("eval")
    ev.add_argument("checkpoint", type=Path)
    ev.add_argument("--eval-set", type=Path, required=True)
    ev.add_argument("--cases", type=parse_cases)
    ev.add_argument("--no-error", action="store_true")
    ev.add_argument("--out-dir", type=Path, required=True)
    ev.add_argument("--workers", type=int, default=8)
    ev.add_argument("--max-steps", type=int, default=300)
    ev.add_argument("--impedance", choices=sorted(IMPEDANCE), default="HOOK_LOW")
    ev.add_argument("--wrench", choices=["joint_torque", "ft"], default="joint_torque")
    ev.add_argument("--wrench-noise", type=float, nargs=2, default=[0.0, 0.0], metavar=("N", "NM"),
                    help="Gaussian noise std of the force (N) and torque (N m) the policy sees")
    ev.add_argument("--setpoint", choices=["continuous", "measured"], default="measured",
                    help="what an action's delta moves: the last commanded pose (chunks join into one path) or the "
                         "hook's measured pose (training's rule)")
    ev.add_argument("--cell", choices=sorted(CELLS), default="swing")
    ev.add_argument("--chunk-s", type=float, default=0.6, help="duration the robot takes for one action chunk")
    ev.add_argument("--settle-s", type=float, default=0.4, help="hold at the chunk's end before the next observation")
    ev.add_argument("--max-dev", type=float, nargs=2, default=[0.0, 0.0], metavar=("MM", "DEG"),
                    help="max path deviation of the set point from the flange's measured pose (0 0: none)")
    ev.add_argument("--tcp", choices=["hook", "flange"], default="hook",
                    help="where the impedance acts and the set point is held: the hook's frame or the flange")
    ev.add_argument("--no-feedforward", action="store_true", help="damp toward zero, not the set point's velocity")
    ev.add_argument("--save-trajectories", type=int, default=10)
    ev.add_argument("--compare", type=Path, help="episodes.csv of the same cases in the training model")
    vid = sub.add_parser("video")
    vid.add_argument("--run", type=Path, required=True)
    vid.add_argument("--cases", type=parse_cases, default=parse_cases("0-9"))
    vid.add_argument("--tile", type=lambda t: [int(v) for v in t.split("x")], default=[640, 480])
    vid.add_argument("--speed", type=int, default=2, help="robot time per video time")
    vid.add_argument("--out", type=Path)
    args = parser.parse_args()
    {"eval": cmd_eval, "video": cmd_video}[args.cmd](args)


if __name__ == "__main__":
    main()
