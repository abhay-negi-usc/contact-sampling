"""MuJoCo model of the hook/eyelet pair shared by the trajectory scripts.

Trajectory convention: hook pose in the eyelet frame, identity = assembled. The STLs do not
sit in those frames as exported: the eyelet is yawed 90 deg about z (its loop stands across
the hook's plane of motion) and the hook 180 deg about its own z (so the open end leads).

Both parts are concave, and MuJoCo collides a mesh through its convex hull, so the original
meshes are visual only. Collision uses the convex parts written by convex_decompose_meshes.py;
they enclose the true surface, bulging past it by at most ~0.25 mm.

Convex collision uses MuJoCo's default libccd. The opt-in native CCD of MuJoCo 3.2.5 misses
some penetrations between these parts (reporting +2.4 mm for hulls overlapping by 2.3 mm) and
over-reports some clearances by ~0.1 mm.
"""

import colorsys
from pathlib import Path

import mujoco
import numpy as np

ASSET_DIR = Path(__file__).resolve().parents[1] / "assets" / "hook"
INCH = 0.0254  # the STLs are modelled in inches, the trajectory is in metres
EYELET_YAW_DEG = 90.0
HOOK_YAW_DEG = 180.0
COLLISION_GROUP = 3


def convex_parts(asset_dir, name):
    parts = sorted((asset_dir / "convex" / name).glob(f"{name}_*.stl"))
    if not parts:
        raise SystemExit(f"No convex parts in {asset_dir / 'convex' / name}; "
                         "run scripts/convex_decompose_meshes.py first.")
    return parts


def part_colors(n, hue_range, alpha):
    hues = np.linspace(*hue_range, n)
    return [(*colorsys.hsv_to_rgb(h, 0.8, 0.9), alpha) for h in hues]


def yaw_quat(deg):
    half = np.deg2rad(deg) / 2
    return f"{np.cos(half)} 0 0 {np.sin(half)}"


def build_model(asset_dir=ASSET_DIR, scale=INCH, eyelet_yaw_deg=EYELET_YAW_DEG, hook_yaw_deg=HOOK_YAW_DEG,
                offscreen_size=(640, 480)):
    asset_dir = Path(asset_dir).resolve()
    s = f"{scale} {scale} {scale}"
    meshes, geoms = [], {"hook": [], "eyelet": []}
    for name, hues in [("hook", (0.02, 0.15)), ("eyelet", (0.5, 0.7))]:
        parts = convex_parts(asset_dir, name)
        for part, rgba in zip(parts, part_colors(len(parts), hues, 0.8)):
            meshes.append(f'<mesh name="{part.stem}" file="convex/{name}/{part.name}" scale="{s}"/>')
            geoms[name].append(f'<geom class="collision" mesh="{part.stem}" '
                               f'rgba="{" ".join(f"{c:.3f}" for c in rgba)}"/>')
    xml = f"""
<mujoco model="hook_eyelet_assembly">
  <compiler meshdir="{asset_dir}"/>
  <option gravity="0 0 0"/>
  <statistic meansize="0.01"/>
  <visual>
    <global offwidth="{offscreen_size[0]}" offheight="{offscreen_size[1]}"/>
    <headlight ambient="0.3 0.3 0.3" diffuse="0.4 0.4 0.4"/>
    <scale framelength="1.2" framewidth="0.04"/>
  </visual>
  <asset>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.92 0.93 0.95" rgb2="0.6 0.65 0.72"
             width="256" height="256"/>
    <mesh name="hook" file="hook.stl" scale="{s}"/>
    <mesh name="eyelet" file="eyelet.stl" scale="{s}"/>
    {"".join(meshes)}
  </asset>
  <default>
    <default class="visual">
      <geom type="mesh" contype="0" conaffinity="0" group="0"/>
    </default>
    <default class="collision">
      <geom type="mesh" group="{COLLISION_GROUP}"/>
    </default>
  </default>
  <worldbody>
    <light directional="true" pos="0.2 -0.3 0.4" dir="-0.4 0.6 -0.8" diffuse="0.6 0.6 0.6"/>
    <light directional="true" pos="-0.2 0.3 0.4" dir="0.4 -0.6 -0.8" diffuse="0.3 0.3 0.3"/>
    <body name="eyelet" quat="{yaw_quat(eyelet_yaw_deg)}">
      <geom name="eyelet" class="visual" mesh="eyelet" rgba="0.25 0.45 0.85 1"/>
      {"".join(geoms["eyelet"])}
    </body>
    <!-- Posed through a free joint, not as a mocap body: MuJoCo treats mocap bodies as
         static and never collides two static bodies. -->
    <body name="hook">
      <freejoint name="hook"/>
      <frame quat="{yaw_quat(hook_yaw_deg)}">
        <geom name="hook" class="visual" mesh="hook" rgba="0.95 0.55 0.15 1"/>
        {"".join(geoms["hook"])}
      </frame>
    </body>
  </worldbody>
</mujoco>"""
    return mujoco.MjModel.from_xml_string(xml)


def load_trajectory(csv_path):
    traj = np.genfromtxt(csv_path, delimiter=",", names=True)
    pos = np.stack([traj["x"], traj["y"], traj["z"]], axis=1)
    quat = np.stack([traj["qw"], traj["qx"], traj["qy"], traj["qz"]], axis=1)  # MuJoCo is wxyz
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    return traj, pos, quat


def set_hook_pose(m, d, pos, quat):
    """Place the hook and update kinematics and contacts."""
    adr = m.jnt_qposadr[m.joint("hook").id]
    d.qpos[adr : adr + 7] = np.concatenate([pos, quat])
    mujoco.mj_forward(m, d)


class Separation:
    """Signed minimum distance between the hook and eyelet collision parts.

    Positive with space between the parts, negative (deepest penetration) when they
    interfere. Contacts cannot be used for this: MuJoCo's collision pipeline drops part pairs
    beyond ~2.5 mm even with a larger margin. Pairs are instead tested with mj_geomDistance
    in order of a bounding-sphere lower bound, stopping once no remaining pair can be closer.

    mj_geomDistance (libccd) inflates the parts by its distmax and measures their penetration,
    which underestimates distances well below distmax: with distmax 50 mm, parts 1 mm apart
    can read as 0.1 mm or even as penetrating. So the search runs over rising rungs of distmax
    (RUNGS, doubling from 0.5 mm) and stops at the first rung within which some pair lies.
    Every pair then lies beyond the rung below, half this one, where libccd reads within
    ~0.01 mm near contact and ~3% further out (checked against exact hull distances).
    Penetration depth errs the same way but more, at small distmax most, so a penetrating pair
    takes the largest of its readings over DEPTH_RUNGS. Even so, depths beyond ~1 mm can read up to
    ~2 mm too deep: trust the sign of a penetration, not its depth.
    """

    RUNGS = 0.0005 * 2.0 ** np.arange(12)
    DEPTH_RUNGS = (0.002, 0.008, 0.05)

    def __init__(self, m):
        hook = np.where((m.geom_group == COLLISION_GROUP) & (m.geom_bodyid == m.body("hook").id))[0]
        eyelet = np.where((m.geom_group == COLLISION_GROUP) & (m.geom_bodyid == m.body("eyelet").id))[0]
        self.g1, self.g2 = (a.ravel() for a in np.meshgrid(hook, eyelet, indexing="ij"))
        self.rsum = m.geom_rbound[self.g1] + m.geom_rbound[self.g2]
        self._fromto = np.zeros(6)

    def __call__(self, m, d, distmax=0.05):
        """(distance, closest points as (2, 3) array or None if beyond distmax) for the current kinematics."""
        lower = np.linalg.norm(d.geom_xpos[self.g1] - d.geom_xpos[self.g2], axis=1) - self.rsum
        order = np.argsort(lower)
        for rung in [r for r in self.RUNGS if r < distmax] + [distmax]:
            best, points = rung, None
            for k in order:
                if lower[k] >= best:
                    break
                dist = mujoco.mj_geomDistance(m, d, self.g1[k], self.g2[k], rung, self._fromto)
                pts = self._fromto.reshape(2, 3).copy()
                if dist < 0:  # every reading is a lower bound: keep the largest
                    for wider in self.DEPTH_RUNGS:
                        alt = mujoco.mj_geomDistance(m, d, self.g1[k], self.g2[k], wider, self._fromto)
                        if alt > dist:
                            dist, pts = alt, self._fromto.reshape(2, 3).copy()
                if dist < best:
                    best, points = dist, pts
            if points is not None:
                return best, points
        return distmax, None


def hook_eyelet_contacts(m, d):
    """(position, normal, signed distance) of every contact between the two bodies."""
    hook_id, eyelet_id = m.body("hook").id, m.body("eyelet").id
    contacts = []
    for c in d.contact[: d.ncon]:
        if {m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]} == {hook_id, eyelet_id}:
            contacts.append((c.pos.copy(), c.frame[:3].copy(), c.dist))
    return contacts
