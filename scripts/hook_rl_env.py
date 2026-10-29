"""Hook/eyelet assembly as a reinforcement-learning task: from an arbitrary start pose, drive the hook onto the
eyelet's loop, knowing the hook's pose relative to the eyelet exactly.

Physics (MuJoCo, the collision parts of hook_eyelet_model.py): the eyelet is fixed at the origin, its stem
standing on a mounting plane at its foot (FLOOR_Z). The hook is a free steel body (17 g) held by a compliant
floating gripper: a soft weld (WELD_SOLREF, WELD_SOLIMP, WELD_TORQUESCALE) from the hook's origin, the inside of
its crook, to a mocap target pose. Gravity is off: the gripper carries the hook. With impedance 0.3 the weld is a
pose spring that settles a 2 mm / 5 deg move to ~95% within a policy step and presses a blocked hook with ~0.5 N
for 2 mm; MuJoCo's default impedance (0.9-0.95) made a blocked weld ~20x stiffer than in free space (6.8 N), and
torquescale 20 restores turns about the hook's x and y axes, which the weld's origin anchor, 24 mm from the
centre of mass, otherwise slowed to ~45% per step.

In the assembled pose the eyelet's top wire sits in the crook and the hook's shaft points up, so the hook
catches the loop from below, like a crane hook under an eye bolt; pull_test checks this by letting go and
pulling the hook up.

Action (6, in [-1, 1]): a delta pose, scaled per dimension by max_delta (x, y, z mm; rotation-vector x, y, z deg,
in the eyelet frame, about the hook's origin): the gripper's target is the hook's current pose moved by it. The
target starts from the current pose every step, so a blocked hook presses with one step's spring force at most.
The policy acts at 1 / (SUBSTEPS * timestep) = 20 Hz.

Observation: the policy sees the hook's pose relative to the eyelet, OBS_POSE_DIM values: position / 50 mm and
the first two columns of its rotation matrix (a continuous code for orientation). The critic, used only in
training, also gets OBS_DIM - OBS_POSE_DIM privileged values, all functions of the state: linear velocity /
100 mm/s; angular velocity / 2 rad/s; offsets of the hook's pose points from their assembled positions / 50 mm;
the linking number of hook and loop (hook_linking.py); the last action; distance to go / 100 mm and the progress
along the guide path it is measured to (GuidePath). Pose points: N_KEYPOINTS points along the hook's wire plus
two off the wire's plane, (0, +-20, 10) mm in the hook frame; the wire's straight shaft runs along the hook's z
axis, so points on the wire alone hardly register a turn about it.

Reward: potential shaping, r = Phi(s') - Phi(s) with Phi = -(distance to go) / KP_SCALE_MM; a step cost; and
SUCCESS_BONUS once the hook has stayed HOLD_STEPS steps in the assembled region (every pose point within
SUCCESS_KP_MM of its assembled position, linking above SUCCESS_LINK), which ends the episode. Nothing is paid per
step in the region: a reward of 0.5 per step there taught a policy to step in and out of it for the whole
episode, which out-earned the bonus for finishing. The distance to go (GuidePath) runs along the reference assembly trajectories
and charges for any mismatch in linking number, so it falls only by threading the loop: a plain pose distance
to the assembled pose led the first policy to park the hook beside the loop, turned 90 deg about its shaft.

Starts: the task draws the hook's origin uniformly in START_BOX_MM and its orientation uniformly, rejecting
poses in contact with the eyelet or the floor. Training uses a curriculum of N_LEVELS levels, ordered by how far
the start is from assembled: level k < N_LEVELS - 1 takes collision-free starts whose distance to go lies in
LEVEL_D_MM[k] .. LEVEL_D_MM[k + 1], and the last level is the task. Candidates are perturbed guide-path waypoints
(partly threaded starts) and poses scattered about the assembled one, kept if their distance to go falls in the
band. The trainer raises the level as the success rate at the current one passes a threshold (train_hook_rl.py).
Distance to go rather than pose offset sets the levels because pose offsets from the assembled pose jump from
threaded to unthreaded: of the starts within 6.6 mm and 18 deg of it 79% are threaded, within 9.5 mm and 25 deg 7%.

The nominal task (task="nominal") is a smaller one, the setting of the hardware experiments: the hook starts about
the pre-assembly pose the reference trajectories share, perturbed per axis by NOMINAL_PERTURB in the believed
eyelet frame, and unthreaded (all such starts are 57-94 mm to go); its curriculum keeps the distance-to-go bands
up to NOMINAL_TOP_MM. Goal-pose uncertainty (goal_error) makes the believed eyelet pose differ from the true one
by an error drawn per episode: the policy observes the hook's pose relative to the believed eyelet and moves in
its frame; the critic also gets the error. An estimator can correct the belief during the episode (set_belief).
"""

import time

import numpy as np
import mujoco
import trimesh
from scipy.spatial.transform import Rotation

from hook_eyelet_model import (ASSET_DIR, COLLISION_GROUP, INCH, convex_parts, load_trajectory, part_colors,
                               yaw_quat, EYELET_YAW_DEG, HOOK_YAW_DEG)
from hook_linking import centrelines, linking_many, resample

TIMESTEP = 0.002
SUBSTEPS = 25
WELD_SOLREF = "0.01 1"
WELD_SOLIMP = "0.3 0.3 0.001 0.5 2"
WELD_TORQUESCALE = 20.0
FRICTION = 0.5
CONTACT_TC = 0.004
STEEL_DENSITY = 7850.0
FLOOR_Z = -67.35  # mm: foot of the eyelet's stem

# Per policy step: x, y, z mm; rotation x, y, z deg. Tuned (hook_rl sweeps of 2026-10-26): the threading swing
# turns about y and starts can sit far above or below the loop, so z and rotation about y get the most range;
# steps of 2 mm / 5 deg or finer stalled for millions of steps at the threading level of the curriculum.
MAX_DELTA = np.array([4.0, 2.0, 6.0, 5.0, 10.0, 5.0])
N_KEYPOINTS = 6
OFF_PLANE_POINTS = np.array([[0.0, 20.0, 10.0], [0.0, -20.0, 10.0]])
N_POSE_POINTS = N_KEYPOINTS + len(OFF_PLANE_POINTS)
OBS_POSE_DIM = 3 + 6
OBS_DIM = OBS_POSE_DIM + 3 + 3 + 3 * N_POSE_POINTS + 1 + 6 + 2 + 6
EXPLORE_ACTOR_DIM = OBS_POSE_DIM + 6 + 6 + 2
EXPLORE_OBS_DIM = EXPLORE_ACTOR_DIM + 3 + 3 + 3 * N_POSE_POINTS + 1 + 2 + 6
STOP_ACTOR_DIM = OBS_POSE_DIM + 6 + 6
STOP_OBS_DIM = STOP_ACTOR_DIM + 3 + 3 + 3 * N_POSE_POINTS + 1 + 2 + 6


def obs_fields(layout):
    """[(name, size)] of an observation layout's values, in order (observe)."""
    pose = [("pos", 3), ("rot", 6)]
    privileged = [("vel", 3), ("omega", 3), ("off", 3 * N_POSE_POINTS), ("link", 1)]
    tail = [("dist", 1), ("progress", 1), ("err", 6)]
    return {"assemble": pose + privileged + [("prev_action", 6)] + tail,
            "explore": pose + [("wrench", 6), ("prev_action", 6), ("samples", 1), ("time", 1)] + privileged + tail,
            "stop": pose + [("wrench", 6), ("prev_action", 6)] + privileged + tail}[layout]


def layout_of(obs_dim):
    """The observation layout of a policy's obs_dim (older assembly policies read a prefix of "assemble")."""
    return {EXPLORE_OBS_DIM: "explore", STOP_OBS_DIM: "stop"}.get(obs_dim, "assemble")


def obs_dims(layout):
    """(observation size, actor's share of it) of an observation layout, "assemble", "explore" or "stop"."""
    return {"assemble": (OBS_DIM, OBS_POSE_DIM), "explore": (EXPLORE_OBS_DIM, EXPLORE_ACTOR_DIM),
            "stop": (STOP_OBS_DIM, STOP_ACTOR_DIM)}[layout]

KP_SCALE_MM = 10.0
LINK_COST_MM = 100.0  # distance to go per unit of linking-number mismatch
STEP_COST = 0.01
ACTION_COST = 0.01
SUCCESS_KP_MM = 2.0
SUCCESS_LINK = 0.9
SUCCESS_BONUS = 20.0
HOLD_STEPS = 5
OUT_OF_BOUNDS_PENALTY = 10.0
MAX_EPISODE_STEPS = 120  # 6 s; tuned policies assemble in ~2 s (60 steps also work, but train ~35% slower)

START_BOX_MM = np.array([[-80.0, 80.0], [-80.0, 80.0], [-30.0, 100.0]])
BOUNDS_MM = np.array([[-150.0, 150.0], [-150.0, 150.0], [FLOOR_Z - 5.0, 200.0]])
LEVEL_D_MM = np.array([0.0, 4, 8, 15, 25, 40, 55, 70, 85, 100, 120, 150])  # distance-to-go bands
N_LEVELS = len(LEVEL_D_MM)  # levels of the wide task; the last level is the task
GUIDE_CSVS = ["hook_assembly_ideal_trajectory.csv", "hook_assembly_max_separation_trajectory.csv",
              "hook_assembly_refined_trajectory.csv"]

# The nominal task: starts about the pre-assembly pose the reference trajectories share (unthreaded, 57-94 mm to
# go), perturbed per axis in the believed eyelet frame; its curriculum stops at the band reaching NOMINAL_TOP_MM.
NOMINAL_PERTURB = np.array([10.0, 10.0, 10.0, 15.0, 15.0, 15.0])  # mm, deg (rotation vector, about the hook origin)
NOMINAL_TOP_MM = 100.0
# Goal-pose uncertainty: the believed eyelet pose, relative to the true one, uniform per axis within these bounds
# (x, y, z mm; rotation vector x, y, z deg); z is known, as on the hardware hook setup.
GOAL_ERROR_BOUNDS = np.array([5.0, 5.0, 0.0, 5.0, 5.0, 5.0])
ERR_OBS_SCALE = 5.0
SEAT_MM = 3.0  # "seat" success: the hook's origin, the inside of its crook, this close to its assembled position
CONTACT_MIN_N = 0.01  # contact force that counts as touching, for recording
# Wrist sensor noise (wrench_noise=(N, N m), or per axis (fx, fy, fz, mx, my, mz)): white Gaussian on the hook-frame
# wrench, drawn anew for every reading the policy observes and every recorded sample; a sample then counts as touching
# when its measured force is above WRENCH_DETECT_SIGMA standard deviations of the noise (as a real sensor's noise floor
# allows), not the true force. wrench_noise_random: each episode's standard deviations are wrench_noise times a uniform
# draw in [0, 1] (training).
WRENCH_DETECT_SIGMA = 4.0
# Sensor calibration errors, drawn per episode (as left after taring at its start): wrench_bias (bounds as
# wrench_noise), an offset uniform per axis in [-bias, bias]; wrench_scale, a gain uniform per axis in [1 - scale,
# 1 + scale]. The reading is gain * true + bias + noise, for the policy and the recorded samples (whose contact gate
# then also clears the bias bound). wrench_obs, the policy's view of the reading: "absolute", / WRENCH_OBS_SCALE;
# "unit", force and torque each divided by max(its norm, WRENCH_UNIT_FLOOR): directions, blind to a common gain,
# shrinking to zero in free space. sensor_scale (a curriculum) multiplies the noise std and the offset and gain error
# bounds of each new episode, a share sensor_review of episodes drawing their scale uniformly below it.
WRENCH_UNIT_FLOOR = (0.2, 0.004)  # N, N m
UNTHREADED_LINK = 0.1


# Exploration (mode="explore"): the episode collects contact for the estimator; at its end the estimator runs once
# and the reward is the reduction of the belief error it achieves (belief_error_mm), times EXPLORE_GAIN. The policy
# then also sees the wrist wrench (hook frame, / WRENCH_OBS_SCALE), its last action, and how many samples
# (/ EXPLORE_SAMPLES_SCALE) and how much of the episode it has used.
EXPLORE_ROT_WEIGHT = 0.5  # mm-equivalent per deg of rotation error
EXPLORE_GAIN = 1.0
WRENCH_OBS_SCALE = np.array([2.0, 2.0, 2.0, 0.05, 0.05, 0.05])
EXPLORE_SAMPLES_SCALE = 500.0

# Declared completion (stop_action=True): the policy ends its episode itself, by a seventh action value above 0,
# as a robot with no feedback on success must; the episode is then judged at that moment, against the true eyelet
# (seated: the "seat" or "pose" region with linking above SUCCESS_LINK): SUCCESS_BONUS if so, FALSE_STOP_PENALTY
# if not. The time limit without a declaration is a failure. The step that declares holds the gripper still. The
# policy then also sees the wrist wrench (hook frame, / WRENCH_OBS_SCALE) and its last motion action ("stop"
# layout): pose alone cannot tell the hook seated against the loop's wire from the same believed pose in free
# space, when the belief is off.
FALSE_STOP_PENALTY = 20.0


def belief_error_mm(err):
    """Size of belief errors (..., 6: mm, rotation vector deg), mm-equivalent."""
    err = np.asarray(err)
    return np.linalg.norm(err[..., :3], axis=-1) + EXPLORE_ROT_WEIGHT * np.linalg.norm(err[..., 3:], axis=-1)


def n_levels(task):
    """Curriculum levels of a task, the task itself the last."""
    return N_LEVELS if task == "wide" else int(np.searchsorted(LEVEL_D_MM, NOMINAL_TOP_MM)) + 1


def hook_inertial():
    """(mass kg, centre of mass m, full inertia "Ixx Iyy Izz Ixy Ixz Iyz" kg m^2) of the steel hook, hook frame."""
    mesh = trimesh.load(ASSET_DIR / "hook.stl")
    mesh.apply_scale(INCH)
    mesh.apply_transform(trimesh.transformations.rotation_matrix(np.deg2rad(HOOK_YAW_DEG), [0, 0, 1]))
    mesh.density = STEEL_DENSITY
    inertia = mesh.moment_inertia  # about the centre of mass
    full = [inertia[0, 0], inertia[1, 1], inertia[2, 2], inertia[0, 1], inertia[0, 2], inertia[1, 2]]
    return mesh.mass, mesh.center_mass, full


def build_rl_model(floor=True):
    asset_dir = ASSET_DIR.resolve()
    s = f"{INCH} {INCH} {INCH}"
    meshes, geoms = [], {"hook": [], "eyelet": []}
    for name, hues in [("hook", (0.02, 0.15)), ("eyelet", (0.5, 0.7))]:
        parts = convex_parts(asset_dir, name)
        for part, rgba in zip(parts, part_colors(len(parts), hues, 0.8)):
            meshes.append(f'<mesh name="{part.stem}" file="convex/{name}/{part.name}" scale="{s}"/>')
            geoms[name].append(f'<geom class="collision" mesh="{part.stem}" '
                               f'rgba="{" ".join(f"{c:.3f}" for c in rgba)}"/>')
    mass, com, full = hook_inertial()
    floor_geom = (f'<geom name="floor" type="plane" pos="0 0 {FLOOR_Z * 1e-3}" size="0.3 0.3 0.01" '
                  f'rgba="0.8 0.8 0.8 1" friction="{FRICTION} 0.005 0.0001" solref="{CONTACT_TC} 1"/>'
                  if floor else "")
    xml = f"""
<mujoco model="hook_eyelet_rl">
  <compiler meshdir="{asset_dir}"/>
  <option timestep="{TIMESTEP}" gravity="0 0 0" integrator="implicitfast"/>
  <statistic meansize="0.01"/>
  <visual>
    <global offwidth="640" offheight="480"/>
    <headlight ambient="0.3 0.3 0.3" diffuse="0.4 0.4 0.4"/>
  </visual>
  <asset>
    <mesh name="hook" file="hook.stl" scale="{s}"/>
    <mesh name="eyelet" file="eyelet.stl" scale="{s}"/>
    {"".join(meshes)}
  </asset>
  <default>
    <default class="visual">
      <geom type="mesh" contype="0" conaffinity="0" group="0" density="0"/>
    </default>
    <default class="collision">
      <geom type="mesh" group="{COLLISION_GROUP}" density="0" friction="{FRICTION} 0.005 0.0001"
            solref="{CONTACT_TC} 1"/>
    </default>
  </default>
  <worldbody>
    <light directional="true" pos="0.2 -0.3 0.4" dir="-0.4 0.6 -0.8" diffuse="0.6 0.6 0.6"/>
    {floor_geom}
    <body name="eyelet" quat="{yaw_quat(EYELET_YAW_DEG)}">
      <inertial pos="0 0 0" mass="0.02" diaginertia="1e-6 1e-6 1e-6"/>
      <geom name="eyelet" class="visual" mesh="eyelet" rgba="0.25 0.45 0.85 1"/>
      {"".join(geoms["eyelet"])}
    </body>
    <body name="target" mocap="true"/>
    <body name="hook">
      <freejoint name="hook"/>
      <inertial pos="{com[0]} {com[1]} {com[2]}" mass="{mass}"
                fullinertia="{' '.join(f'{v:.6e}' for v in full)}"/>
      <frame quat="{yaw_quat(HOOK_YAW_DEG)}">
        <geom name="hook" class="visual" mesh="hook" rgba="0.95 0.55 0.15 1"/>
        {"".join(geoms["hook"])}
      </frame>
    </body>
  </worldbody>
  <equality>
    <weld name="grip" body1="target" body2="hook" relpose="0 0 0 1 0 0 0" solref="{WELD_SOLREF}"
          solimp="{WELD_SOLIMP}" torquescale="{WELD_TORQUESCALE}"/>
  </equality>
</mujoco>"""
    return mujoco.MjModel.from_xml_string(xml)


def quat_to_mat(q):
    """Rotation matrices (N, 3, 3) of unit quaternions (N, 4), wxyz."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                     2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                     2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], axis=1).reshape(-1, 3, 3)


class Geometry:
    """Hook points used for observations, reward and start checks (hook frame, mm)."""

    def __init__(self):
        hook, loop, self.hook_radius, _ = centrelines()
        self.points = np.vstack([resample(hook, N_KEYPOINTS), OFF_PLANE_POINTS])
        self.link_hook = resample(hook, 24)
        self.link_loop = np.ascontiguousarray(loop[::4])
        self.floor_check = resample(hook, 40)

    def features(self, pos_mm, rot):
        """(pose points (N, P, 3) mm, linking number (N,)) for hook positions (N, 3) mm, rotations (N, 3, 3)."""
        pts = np.einsum("nij,kj->nki", rot, self.points) + pos_mm[:, None]
        link = linking_many(np.einsum("nij,kj->nki", rot, self.link_hook) + pos_mm[:, None], self.link_loop)
        return pts, link


class GuidePath:
    """Distance to go to the assembled pose, measured along the reference assembly trajectories.

    For a pose q with pose points X(q) and linking number L(q):
        D(q) = min over waypoints k of  mean_p |X_p(q) - W_kp|  +  R_k  +  LINK_COST_MM |L(q) - L_k|
    with W_k, L_k the pose points and linking number of waypoint k and R_k the length of the rest of its
    trajectory (summed mean pose-point moves). The last waypoint is the assembled pose (D = 0 there). An
    unthreaded pose cannot count as near the threaded waypoints, so it is sent back along the path to where the
    reference trajectories enter the loop. D is a function of pose only, so shaping with it pays nothing for
    going round in circles.
    """

    def __init__(self, geo, every=2):
        pts, links, remaining, progress, poses, quats = [], [], [], [], [], []
        for csv in GUIDE_CSVS:
            _, pos, quat = load_trajectory(ASSET_DIR / csv)
            pos, quat = pos[::-1][::every][::-1], quat[::-1][::every][::-1]  # keep the last (assembled) waypoint
            poses.append(pos * 1e3), quats.append(quat)
            p, l = geo.features(pos * 1e3, quat_to_mat(quat))
            step = np.linalg.norm(np.diff(p, axis=0), axis=-1).mean(-1)
            rest = np.concatenate([np.cumsum(step[::-1])[::-1], [0.0]])
            pts.append(p), links.append(l), remaining.append(rest), progress.append(1 - rest / rest[0])
        self.points = np.concatenate(pts)
        self.links = np.concatenate(links)
        self.remaining = np.concatenate(remaining)
        self.progress = np.concatenate(progress)
        self.pos = np.concatenate(poses)  # waypoint poses: mm, Rotation
        self.rot = Rotation.from_quat(np.concatenate(quats), scalar_first=True)

    def distance(self, points, link):
        """(distance to go (N,) mm, progress (N,) in [0, 1] of the waypoint it is measured to)."""
        d = np.linalg.norm(points[:, None] - self.points[None], axis=-1).mean(-1)
        total = d + self.remaining + LINK_COST_MM * np.abs(link[:, None] - self.links)
        k = np.argmin(total, axis=1)
        return total[np.arange(len(k)), k], self.progress[k]


class HookAssemblyBatch:
    """n independent assembly episodes sharing one model, stepped together (one process).

    Each episode starts at curriculum level `level`, except a share `task_fraction` that starts at the last
    level (the task itself) and a share `review` at a level drawn uniformly below `level`.

    task "wide" or "nominal" (module docstring). goal_error: per-axis bounds of the uniform error of the belief
    (GOAL_ERROR_BOUNDS; None: exact), drawn per episode as the true eyelet's pose in the believed eyelet frame;
    the policy observes and acts in the believed eyelet frame, while reward, success and the critic's privileged
    values use the true pose (and the critic also gets the error). error_scale shrinks the bounds (a curriculum
    over the error's size, train_hook_rl.py --error-levels), and a share error_review of episodes draws its scale
    uniformly below that. success "pose" (every pose point within
    SUCCESS_KP_MM) or "seat" (the crook within SEAT_MM of its seat), either with the linking number above
    SUCCESS_LINK for HOLD_STEPS.

    The robot's frame of an episode is its initial believed eyelet frame: where an estimator observes the hook
    and expresses its estimate of the eyelet (update_belief), so it never sees the true eyelet frame, which is the
    simulation's world frame. record_every > 0 records, every that many physics steps, each episode's hook pose in
    the robot's frame and the wrench of its contacts in the hook frame (a wrist force-torque sensor) while the
    hook touches anything (contact force above CONTACT_MIN_N): rec_pose (mm, quaternion wxyz), rec_wrench (N, N m
    about the hook's origin), and rec_true (the pose in the true eyelet frame, for building contact maps only).
    end_on_success=False runs every episode to its time limit (to record seated contact too).

    estimator (contact_estimators.ContactEstimator; needs record_every) estimates the eyelet's pose during the
    episodes (estimate_due), the online contact-manifold estimation of RL5-RL7.

    mode "explore" trains an exploration policy instead (module constants EXPLORE_*): no assembly reward or
    success ending, and at the time limit, a true terminal here, the estimator runs once on the episode's samples
    and the reward is the reduction of belief_error_mm it achieves. obs_layout ("assemble" or "explore", default
    the mode's; obs_dims) can differ from the mode: an exploration policy hands over to an assembly policy.

    stop_action (mode "assemble"): actions have a seventh value, the declaration of completion (FALSE_STOP_PENALTY
    comment); success is then a declaration while seated, not holding the region, and the observation layout
    defaults to "stop". Six-value actions (an exploration policy's) declare nothing."""

    def __init__(self, n, seed=0, level=0, task_fraction=0.0, review=0.0, floor=True, max_steps=MAX_EPISODE_STEPS,
                 max_delta=MAX_DELTA, auto_reset=True, task="wide", goal_error=None, success="pose", record_every=0,
                 end_on_success=True, estimator=None, est_every=5, est_min_samples=200, est_new_samples=100,
                 mode="assemble", obs_layout=None, stop_action=False, error_scale=1.0, error_review=0.0,
                 wrench_noise=(0.0, 0.0), wrench_noise_random=False, wrench_bias=(0.0, 0.0), wrench_scale=0.0,
                 wrench_obs="absolute", sensor_scale=1.0, sensor_review=0.0):
        self.n, self.max_steps, self.auto_reset = n, max_steps, auto_reset
        self.max_delta = np.asarray(max_delta, dtype=np.float64)
        self.level, self.task_fraction, self.review = level, task_fraction, review
        self.task, self.success_mode = task, success
        self.n_levels = n_levels(task)
        self.task_level = self.n_levels - 1
        self.goal_error = np.zeros(6) if goal_error is None else np.asarray(goal_error, dtype=np.float64)
        self.error_scale, self.error_review = error_scale, error_review
        self.err_scale_drawn = np.zeros(n)
        self.bel_pos = np.zeros((n, 3))  # believed eyelet pose in the true eyelet frame: mm, rotation matrix
        self.bel_rot = np.tile(np.eye(3), (n, 1, 1))
        self.base_pos, self.base_rot = self.bel_pos.copy(), self.bel_rot.copy()  # the robot's frame, the same way
        self.err = np.zeros((n, 6))  # the true eyelet's pose in the believed frame (mm, rotation vector deg)
        self.record_every = record_every
        self.stop_action = stop_action and mode == "assemble"
        self.mode, self.obs_layout = mode, obs_layout or ("stop" if self.stop_action else mode)
        self.end_on_success = end_on_success and mode == "assemble"
        self.est_online = True  # re-estimate during the episode (estimate_due), in mode "assemble"
        self.estimator, self.est_every = estimator, est_every
        self.est_min_samples, self.est_new_samples = est_min_samples, est_new_samples
        self.est_last = np.zeros(n, np.int64)  # samples at the last estimate
        self.est_count = np.zeros(n, np.int64)
        self.est_time = np.zeros(n)
        self.err0 = np.zeros((n, 6))  # the belief's error at the start of the episode
        self.rec_pose, self.rec_wrench, self.rec_true = ([[] for _ in range(n)] for _ in range(3))
        _, pos, quat = load_trajectory(ASSET_DIR / GUIDE_CSVS[0])
        self.nominal_pos, self.nominal_rot = pos[0] * 1e3, Rotation.from_quat(quat[0], scalar_first=True)
        self.m = build_rl_model(floor=floor)
        self.d = [mujoco.MjData(self.m) for _ in range(n)]
        self.rng = np.random.default_rng(seed)
        self.wrench_noise = per_axis(wrench_noise)
        self.wrench_noise_random = wrench_noise_random
        self.noise_rng = np.random.default_rng(seed + 7919)  # apart from rng: noise leaves the starts drawn unchanged
        self.noise_std = np.tile(self.wrench_noise, (n, 1))  # this episode's, per episode
        self.wrench_bias, self.wrench_scale = per_axis(wrench_bias), wrench_scale
        self.wrench_obs = wrench_obs
        self.sensor_gain, self.sensor_bias = np.ones((n, 6)), np.zeros((n, 6))  # this episode's
        self.noisy_sensor = self.wrench_noise.any() or self.wrench_bias.any() or wrench_scale > 0
        self.sensor_scale, self.sensor_review = sensor_scale, sensor_review
        self.sensor_scale_drawn = np.ones(n)
        self.geo = Geometry()
        self.guide = GuidePath(self.geo)
        self.goal_points = self.geo.points
        self.hook_body = self.m.body("hook").id
        self.floor = floor
        self.prev_action = np.zeros((n, 6))
        self.steps = np.zeros(n, np.int64)
        self.hold = np.zeros(n, np.int64)
        self.ep_return = np.zeros(n)
        self.start_level = np.zeros(n, np.int64)
        self.phi = np.zeros(n)

    # ---------------------------------------------------------------- state access
    def poses(self):
        """(positions (n, 3) mm, quaternions (n, 4) wxyz, linear vel (n, 3) mm/s, angular vel world (n, 3) rad/s)."""
        q = np.array([d.qpos[:7] for d in self.d])
        v = np.array([d.qvel[:6] for d in self.d])
        rot = quat_to_mat(q[:, 3:])
        return q[:, :3] * 1e3, q[:, 3:], v[:, :3] * 1e3, np.einsum("nij,nj->ni", rot, v[:, 3:])

    def set_pose(self, i, pos_mm, quat):
        d = self.d[i]
        mujoco.mj_resetData(self.m, d)
        d.qpos[:3], d.qpos[3:7] = pos_mm * 1e-3, quat
        d.mocap_pos[0], d.mocap_quat[0] = pos_mm * 1e-3, quat
        mujoco.mj_forward(self.m, d)

    def record(self, i):
        """Append episode i's sensor sample if the hook is in contact (see the class docstring)."""
        d = self.d[i]
        wrench = hook_wrench(self.m, d, self.hook_body)
        if not self.noisy_sensor and np.linalg.norm(wrench[:3]) < CONTACT_MIN_N:
            return
        rot = quat_to_mat(d.qpos[3:7][None])[0]
        wrench = np.concatenate([rot.T @ wrench[:3], rot.T @ wrench[3:]])  # world -> hook frame
        if self.noisy_sensor:
            wrench = self.sensor_reading(wrench, i)
            gate = WRENCH_DETECT_SIGMA * self.noise_std[i, :3].max() + np.linalg.norm(self.wrench_bias[:3])
            if np.linalg.norm(wrench[:3]) < max(CONTACT_MIN_N, gate):
                return
        pos = d.qpos[:3] * 1e3
        base_t = self.base_rot[i].T
        self.rec_pose[i].append(np.concatenate([
            base_t @ (pos - self.base_pos[i]), Rotation.from_matrix(base_t @ rot).as_quat(scalar_first=True)]))
        self.rec_wrench[i].append(wrench)
        self.rec_true[i].append(d.qpos[:7].copy() * np.array([1e3, 1e3, 1e3, 1, 1, 1, 1]))

    def sensor_reading(self, wrench, idx):
        """The wrist sensor's reading of true hook-frame wrenches (6 or (k, 6)) of episodes idx: gain, bias, noise."""
        out = self.sensor_gain[idx] * wrench + self.sensor_bias[idx]
        if self.wrench_noise.any():
            out = out + self.noise_rng.normal(size=np.shape(wrench)) * self.noise_std[idx]
        return out

    def estimate_due(self):
        """Re-estimate the eyelet for every episode with est_new_samples new contact samples (est_min_samples in
        all) on an est_every-th step: from all its samples, starting from its belief, which the estimate replaces."""
        for i in range(self.n):
            k = len(self.rec_pose[i])
            if (k < self.est_min_samples or k - self.est_last[i] < self.est_new_samples
                    or self.steps[i] % self.est_every):
                continue
            self.estimate_now(i)

    def estimate_now(self, i):
        """Estimate the eyelet for episode i from all its samples (if est_min_samples), starting from its belief."""
        k = len(self.rec_pose[i])
        if k < self.est_min_samples:
            return
        t0 = time.perf_counter()
        pos, quat = self.estimator.estimate(*self.recorded(i), *self.believed_in_base(i))
        self.est_time[i] += time.perf_counter() - t0
        self.update_belief(i, pos, quat)
        self.est_last[i], self.est_count[i] = k, self.est_count[i] + 1

    def recorded(self, i):
        """Episode i's recorded samples: (poses (k, 7) in the robot's frame, wrenches (k, 6) in the hook frame)."""
        return (np.array(self.rec_pose[i]).reshape(-1, 7), np.array(self.rec_wrench[i]).reshape(-1, 6))

    def start_ok(self, i, pos_mm, rot):
        """No contact with the eyelet or the floor, and the wire clear of the floor."""
        if self.d[i].ncon > 0:
            return False
        z = (rot @ self.geo.floor_check.T).T[:, 2] + pos_mm[2]
        return not self.floor or z.min() > FLOOR_Z + self.geo.hook_radius + 0.5

    # ---------------------------------------------------------------- episodes
    def pick_level(self):
        r = self.rng.random()
        if r < self.task_fraction:
            return self.task_level
        if r < self.task_fraction + self.review and self.level > 0:
            return int(self.rng.integers(self.level))
        return self.level

    def set_belief(self, i, err):
        """Start episode i's belief: err is the true eyelet's pose in the believed eyelet frame (x, y, z mm,
        rotation vector deg). The believed frame becomes the robot's frame of the episode."""
        r = Rotation.from_rotvec(np.deg2rad(err[3:])).inv()
        self.set_believed_frame(i, -r.apply(err[:3]), r.as_matrix())
        self.base_pos[i], self.base_rot[i] = self.bel_pos[i], self.bel_rot[i]

    def set_believed_frame(self, i, pos_mm, rot):
        """The believed eyelet frame of episode i, in the true one (mm, rotation matrix)."""
        self.bel_pos[i], self.bel_rot[i] = pos_mm, rot
        r = Rotation.from_matrix(rot).inv()
        self.err[i, :3], self.err[i, 3:] = -r.apply(pos_mm), np.degrees(r.as_rotvec())

    def update_belief(self, i, pos_mm, quat):
        """An estimate of the eyelet's pose in the robot's frame (mm, quaternion wxyz) becomes episode i's belief."""
        base = Rotation.from_matrix(self.base_rot[i])
        self.set_believed_frame(i, self.base_pos[i] + base.apply(pos_mm),
                                (base * Rotation.from_quat(quat, scalar_first=True)).as_matrix())

    def believed_in_base(self, i):
        """Episode i's believed eyelet pose in the robot's frame (mm, quaternion wxyz): the estimator's prior."""
        base = Rotation.from_matrix(self.base_rot[i]).inv()
        return (base.apply(self.bel_pos[i] - self.base_pos[i]),
                (base * Rotation.from_matrix(self.bel_rot[i])).as_quat(scalar_first=True))

    def to_true(self, i, pos_mm, rot):
        """A pose in episode i's believed eyelet frame (mm, Rotation) in the true one."""
        bel = Rotation.from_matrix(self.bel_rot[i])
        return self.bel_pos[i] + bel.apply(pos_mm), bel * rot

    def draw_error(self, i):
        """A belief error for episode i: uniform within error_scale times the bounds (a share error_review
        within a scale drawn below it, so smaller errors stay trained)."""
        scale = self.error_scale
        if self.rng.random() < self.error_review:
            scale = self.rng.uniform(0.0, scale)
        self.err_scale_drawn[i] = scale
        self.set_belief(i, self.rng.uniform(-scale * self.goal_error, scale * self.goal_error))

    def sample_start(self, i, level=None):
        level = self.pick_level() if level is None else level
        self.draw_error(i)
        nominal = self.task == "nominal" and level == self.task_level
        lo, hi = (LEVEL_D_MM[level], LEVEL_D_MM[level + 1]) if level < self.task_level else (0.0, np.inf)
        for attempt in range(20000):
            if attempt % 2000 == 1999:  # this error leaves no valid start (the true eyelet where starts are): another
                self.draw_error(i)
            if nominal:  # about the reference start, in the believed eyelet frame
                pert = self.rng.uniform(-NOMINAL_PERTURB, NOMINAL_PERTURB)
                pos, rot = self.to_true(i, self.nominal_pos + pert[:3],
                                        Rotation.from_rotvec(np.deg2rad(pert[3:])) * self.nominal_rot)
            elif level == self.task_level:
                pos = self.rng.uniform(START_BOX_MM[:, 0], START_BOX_MM[:, 1])
                rot = Rotation.random(random_state=self.rng)
            elif self.rng.random() < 0.5:  # a perturbed guide-path waypoint whose own distance to go is near the band
                k = self.rng.choice(np.flatnonzero((self.guide.remaining >= 0.5 * lo) & (self.guide.remaining <= hi)))
                pos, rot = self.guide.pos[k], self.guide.rot[k]
                pos = pos + self.rng.normal(0, min(0.15 * hi, 15.0), 3)
                rot = Rotation.from_rotvec(self.rng.normal(0, np.deg2rad(min(0.5 * hi, 30.0)), 3)) * rot
            else:  # scattered about the assembled pose
                pos = self.rng.uniform(*(START_BOX_MM * min(1.0, hi / 120)).T)
                axis = self.rng.normal(size=3)
                rot = Rotation.from_rotvec(axis / np.linalg.norm(axis)
                                           * self.rng.uniform(0, np.deg2rad(min(180.0, 1.5 * hi))))
            self.set_pose(i, pos, rot.as_quat(scalar_first=True))
            if not self.start_ok(i, pos, rot.as_matrix()):
                continue
            if level < self.task_level or nominal:
                pts, link = self.geo.features(pos[None], rot.as_matrix()[None])
                if nominal and abs(link[0]) > UNTHREADED_LINK:
                    continue
                if not lo <= self.guide.distance(pts, link)[0][0] <= hi:
                    continue
            self.start_level[i] = level
            return
        raise RuntimeError(f"no start found at level {level}")

    def reset_idx(self, idx, level=None):
        for i in idx:
            self.sample_start(i, level)
        self.begin_episodes(idx)

    def reset_to(self, i, pos_mm, quat, err=None):
        """Start episode i from the given pose in its believed eyelet frame, the belief off by err (x, y, z mm,
        rotation vector deg; default exact): a fixed validation or evaluation case. Its observations."""
        self.set_belief(i, np.zeros(6) if err is None else np.asarray(err, dtype=np.float64))
        pos, rot = self.to_true(i, np.asarray(pos_mm, dtype=np.float64), Rotation.from_quat(quat, scalar_first=True))
        self.set_pose(i, pos, rot.as_quat(scalar_first=True))
        self.start_level[i] = self.task_level
        self.begin_episodes([i])
        return self.observe()[0]

    def begin_episodes(self, idx):
        for i in idx:
            self.rec_pose[i], self.rec_wrench[i], self.rec_true[i] = [], [], []
        self.prev_action[idx] = 0
        self.steps[idx] = 0
        self.hold[idx] = 0
        self.ep_return[idx] = 0
        self.est_last[idx], self.est_count[idx], self.est_time[idx] = 0, 0, 0.0
        self.err0[idx] = self.err[idx]
        if self.noisy_sensor:
            scale = np.full((len(idx), 1), float(self.sensor_scale))
            if self.sensor_review:
                review = self.noise_rng.random(len(idx)) < self.sensor_review
                scale[review, 0] = self.noise_rng.uniform(0.0, self.sensor_scale, review.sum())
            self.sensor_scale_drawn[idx] = scale[:, 0]
            u = self.noise_rng.uniform(size=(len(idx), 1)) if self.wrench_noise_random else 1.0
            self.noise_std[idx] = self.wrench_noise * u * scale
            if self.wrench_scale:
                self.sensor_gain[idx] = 1.0 + scale * self.noise_rng.uniform(-self.wrench_scale, self.wrench_scale,
                                                                             (len(idx), 6))
            if self.wrench_bias.any():
                self.sensor_bias[idx] = scale * self.wrench_bias * self.noise_rng.uniform(-1.0, 1.0, (len(idx), 6))
        pos, quat, *_ = self.poses()
        pts, link = self.geo.features(pos[idx], quat_to_mat(quat[idx]))
        self.phi[idx] = -self.guide.distance(pts, link)[0] / KP_SCALE_MM

    def reset(self, level=None):
        self.reset_idx(np.arange(self.n), level)
        return self.observe()[0]

    def observe(self):
        """(observations, true position mm, pose-point offsets mm, linking number, distance to go mm)."""
        pos, quat, vel, omega = self.poses()
        rot = quat_to_mat(quat)
        pts, link = self.geo.features(pos, rot)
        off = pts - self.goal_points
        dist, progress = self.guide.distance(pts, link)
        rel_pos = np.einsum("nji,nj->ni", self.bel_rot, pos - self.bel_pos)  # in the believed eyelet frame
        rel_rot = np.einsum("nji,njk->nik", self.bel_rot, rot)
        pose = [rel_pos / 50.0, rel_rot[:, :, :2].transpose(0, 2, 1).reshape(self.n, 6)]  # the policy's
        privileged = [vel / 100.0, omega / 2.0, off.reshape(self.n, -1) / 50.0, link[:, None]]
        if self.obs_layout == "assemble":
            return (np.concatenate(pose + privileged + [self.prev_action, dist[:, None] / 100.0, progress[:, None],
                                                        self.err / ERR_OBS_SCALE], axis=1).astype(np.float32),
                    pos, off, link, dist)
        wrench = np.array([hook_wrench(self.m, d, self.hook_body) for d in self.d])
        wrench = np.concatenate([np.einsum("nji,nj->ni", rot, wrench[:, :3]),
                                 np.einsum("nji,nj->ni", rot, wrench[:, 3:])], axis=1)  # hook frame
        if self.noisy_sensor:
            wrench = self.sensor_reading(wrench, slice(None))
        wrench = wrench_features(wrench, self.wrench_obs)
        if self.obs_layout == "stop":
            obs = np.concatenate(pose + [wrench, self.prev_action] + privileged
                                 + [dist[:, None] / 100.0, progress[:, None], self.err / ERR_OBS_SCALE], axis=1)
        else:
            samples = np.array([len(r) for r in self.rec_pose]) / EXPLORE_SAMPLES_SCALE
            obs = np.concatenate(pose + [wrench, self.prev_action,
                                         np.minimum(samples, 4.0)[:, None], (self.steps / self.max_steps)[:, None]]
                                 + privileged + [dist[:, None] / 100.0, progress[:, None],
                                                 self.err / ERR_OBS_SCALE], axis=1)
        return obs.astype(np.float32), pos, off, link, dist

    def step(self, action):
        """Advance every episode one policy step; finished episodes restart unless auto_reset is off. Returns
        (obs, reward, done, timeout, info), info describing the episodes that finished this step ("index", their
        last observation "terminal_obs", outcome and statistics)."""
        action = np.clip(np.asarray(action, dtype=np.float64), -1, 1)
        stop = np.zeros(self.n, bool)
        if action.shape[1] == 7:
            stop = (action[:, 6] > 0) & self.stop_action
            action = np.where(stop[:, None], 0.0, action[:, :6])  # declaring done holds the gripper still
        qpos = np.array([d.qpos[:7] for d in self.d])
        delta = action * self.max_delta  # in the believed eyelet frame: rotate both parts into the true one
        tpos = qpos[:, :3] + np.einsum("nij,nj->ni", self.bel_rot, delta[:, :3]) * 1e-3
        turn = np.einsum("nij,nj->ni", self.bel_rot, np.deg2rad(delta[:, 3:]))
        tquat = (Rotation.from_rotvec(turn) * Rotation.from_quat(qpos[:, 3:], scalar_first=True)
                 ).as_quat(scalar_first=True)
        for i, d in enumerate(self.d):
            d.mocap_pos[0], d.mocap_quat[0] = tpos[i], tquat[i]
            if not self.record_every:
                mujoco.mj_step(self.m, d, nstep=SUBSTEPS)
                continue
            for _ in range(SUBSTEPS // self.record_every):
                mujoco.mj_step(self.m, d, nstep=self.record_every)
                self.record(i)
        self.prev_action = action
        self.steps += 1
        if self.estimator is not None and self.mode == "assemble" and self.est_online:
            self.estimate_due()

        obs, pos, off, link, dist = self.observe()
        phi = -dist / KP_SCALE_MM
        err = np.linalg.norm(off, axis=-1)
        seat = np.linalg.norm(pos, axis=1)
        close = err.max(-1) < SUCCESS_KP_MM if self.success_mode == "pose" else seat < SEAT_MM
        in_goal = close & (link > SUCCESS_LINK)
        self.hold = np.where(in_goal, self.hold + 1, 0)
        success = self.hold >= HOLD_STEPS
        out = np.any((pos < BOUNDS_MM[:, 0]) | (pos > BOUNDS_MM[:, 1]), axis=1) | ~np.isfinite(pos).all(1)
        explore_gain = np.zeros(self.n)
        if self.mode == "assemble" and self.stop_action:  # judged when the policy declares done
            success = stop & in_goal
            reward = (phi - self.phi - STEP_COST - ACTION_COST * np.sum(action ** 2, 1) + SUCCESS_BONUS * success
                      - FALSE_STOP_PENALTY * (stop & ~in_goal) - OUT_OF_BOUNDS_PENALTY * out)
            terminal = stop | out
            timeout = ~terminal & (self.steps >= self.max_steps)
        elif self.mode == "assemble":
            reward = (phi - self.phi - STEP_COST - ACTION_COST * np.sum(action ** 2, 1)
                      + SUCCESS_BONUS * success - OUT_OF_BOUNDS_PENALTY * out)
            terminal = (success & self.end_on_success) | out
            timeout = ~terminal & (self.steps >= self.max_steps)
        else:  # exploration: at the time limit, estimate once; reward the reduction of the belief error
            terminal = out | (self.steps >= self.max_steps)
            timeout = np.zeros(self.n, bool)
            for i in np.flatnonzero(terminal & ~out):
                self.estimate_now(i)
            explore_gain[terminal] = belief_error_mm(self.err0[terminal]) - belief_error_mm(self.err[terminal])
            reward = (EXPLORE_GAIN * explore_gain - STEP_COST - ACTION_COST * np.sum(action ** 2, 1)
                      - OUT_OF_BOUNDS_PENALTY * out)
        self.phi = phi
        self.ep_return += reward
        done = terminal | timeout

        info = {}
        if done.any():
            idx = np.flatnonzero(done)
            info = {"index": idx, "terminal_obs": obs[idx].copy(), "success": success[idx], "out": out[idx],
                    "timeout": timeout[idx], "length": self.steps[idx].copy(), "return": self.ep_return[idx].copy(),
                    "start_level": self.start_level[idx].copy(), "final_kp_mm": err[idx].mean(-1),
                    "final_link": link[idx], "final_dist_mm": dist[idx], "final_seat_mm": seat[idx],
                    "goal_err": self.err[idx].copy(), "goal_err0": self.err0[idx].copy(),
                    "est_count": self.est_count[idx].copy(), "est_time_s": self.est_time[idx].copy(),
                    "explore_gain": explore_gain[idx], "samples": np.array([len(self.rec_pose[i]) for i in idx]),
                    "stopped": stop[idx], "seated": in_goal[idx], "error_scale": self.err_scale_drawn[idx].copy(),
                    "sensor_scale": self.sensor_scale_drawn[idx].copy()}
            if self.auto_reset:
                self.reset_idx(idx)
                obs[idx] = self.observe()[0][idx]
        return obs, reward.astype(np.float32), done, timeout, info


def per_axis(bound):
    """A wrench error bound (force, torque) or (fx, fy, fz, mx, my, mz) as six values."""
    bound = np.asarray(bound, dtype=np.float64)
    return np.repeat(bound, 3) if bound.size == 2 else bound


def wrench_features(wrench, mode="absolute"):
    """The policy's view of hook-frame wrench readings (..., 6) (wrench_obs, the sensor comment)."""
    if mode == "absolute":
        return wrench / WRENCH_OBS_SCALE
    f, t = wrench[..., :3], wrench[..., 3:]
    return np.concatenate([f / np.maximum(np.linalg.norm(f, axis=-1, keepdims=True), WRENCH_UNIT_FLOOR[0]),
                           t / np.maximum(np.linalg.norm(t, axis=-1, keepdims=True), WRENCH_UNIT_FLOOR[1])], axis=-1)


def hook_wrench(m, d, body):
    """Wrench on body from all its contacts, world frame, torque about the body's origin (N, N m)."""
    out, f, frame = np.zeros(6), np.zeros(6), np.zeros((3, 3))
    origin = d.xpos[body]
    for k in range(d.ncon):
        c = d.contact[k]
        b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
        if body not in (b1, b2):
            continue
        mujoco.mj_contactForce(m, d, k, f)
        frame[:] = c.frame.reshape(3, 3)
        force = frame.T @ f[:3]  # on geom2's body, from geom1's
        if b1 == body:
            force = -force
        out[:3] += force
        out[3:] += np.cross(c.pos - origin, force)
    return out


def contact_force(m, d, body):
    """Total normal force [N] of the contacts touching body."""
    f, total = np.zeros(6), 0.0
    for k in range(d.ncon):
        c = d.contact[k]
        if body in (m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]):
            mujoco.mj_contactForce(m, d, k, f)
            total += f[0]
    return total


def pull_test(m, d, body, seconds=1.0, accel=9.81):
    """Let go of the hook and pull it up (+z, `accel` times its weight) for `seconds`, as a load hanging from the
    eyelet would; returns its final (pos m, quat wxyz). An assembled hook is caught by the loop's top wire.

    Works on a copy of d with the gripper's weld off; m's gravity is restored."""
    d2 = mujoco.MjData(m)
    d2.qpos[:], d2.qvel[:] = d.qpos, 0
    d2.eq_active[m.equality("grip").id] = 0
    g = m.opt.gravity.copy()
    m.opt.gravity[:] = [0, 0, accel]
    try:
        mujoco.mj_forward(m, d2)
        mujoco.mj_step(m, d2, nstep=int(round(seconds / m.opt.timestep)))
    finally:
        m.opt.gravity[:] = g
    return d2.qpos[:3].copy(), d2.qpos[3:7].copy()


def worker(conn, n, seed, kwargs):
    """Serve a HookAssemblyBatch over a pipe: ("reset", None), ("step", actions), ("set", {attr: value}), ("close", None).
    An estimator given as a dict ({"kind", "map", ...ContactEstimator kwargs}) is built here, in the worker."""
    if isinstance(kwargs.get("estimator"), dict):
        from contact_estimators import ContactEstimator
        cfg = dict(kwargs["estimator"])
        kwargs = {**kwargs, "estimator": ContactEstimator(cfg.pop("kind"), cfg.pop("map"), **cfg)}
    env = HookAssemblyBatch(n, seed=seed, **kwargs)
    while True:
        cmd, arg = conn.recv()
        if cmd == "reset":
            conn.send(env.reset())
        elif cmd == "step":
            conn.send(env.step(arg))
        elif cmd == "set":
            for k, v in arg.items():
                setattr(env, k, v)
            conn.send(None)
        elif cmd == "close":
            conn.close()
            return
