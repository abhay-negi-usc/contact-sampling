"""MuJoCo model of the cylindrical peg/hole pair shared by the peg trajectory scripts.

Trajectory convention: peg pose in the hole frame, i.e. the pose of the peg STL's frame expressed
in the hole STL's frame, in metres. Both STLs are modelled in millimetres with the part's axis
along +y. The hole's 26 mm bore opens at y = 75 and is 50 mm deep, and the peg's 25 mm tip runs
50.5 mm from its shoulder at y = 35 to its end at y = 85.5. Assembled, the peg is flipped 180 deg
about x with its tip on the bore floor (ASSEMBLED_POS, ASSEMBLED_QUAT); the tip is longer than the
bore is deep, so the shoulder stops 0.5 mm above the hole's top face.

Both parts are concave, and MuJoCo collides a mesh through its convex hull, so the STLs are visual
only. Collision uses the turned profiles below: cylinders for the peg, and for the hole a box, a
cylinder and the bore walls split into convex sectors. The sectors' flat faces enclose the true
surface, bulging past it by at most r (1 - cos(pi / n_sectors)), 0.016 mm at the bore for 64.
Fillets and bolt holes are left out; the peg never reaches them.

The hole sits at the world origin with its axis along world +z, since MuJoCo cameras assume z up.
"""

from pathlib import Path

import mujoco
import numpy as np

from hook_eyelet_model import part_colors

ASSET_DIR = Path(__file__).resolve().parents[1] / "assets" / "cylinder"
MM = 1e-3  # the STLs are modelled in millimetres, the trajectory is in metres
COLLISION_GROUP = 3
N_SECTORS = 64

# Collision profiles in STL millimetres along the part's y axis.
PEG_CYLINDERS = [(0.0, 10.0, 32.0), (10.0, 35.0, 17.5), (35.0, 85.5, 12.5)]  # (y0, y1, radius)
PEG_CYLINDER_NAMES = ["flange", "shaft", "tip"]
HOLE_PLATE = (0.0, 15.0, 37.5)  # (y0, y1, half width) of the square base plate
HOLE_RINGS = [(15.0, 25.0, 0.0, 25.0), (25.0, 35.0, 13.0, 25.0), (35.0, 75.0, 13.0, 15.0)]  # (y0, y1, r_in, r_out)

PEG_TIP_Y = PEG_CYLINDERS[-1][1]
BORE_TOP_Y = HOLE_RINGS[-1][1]
BORE_FLOOR_Y = HOLE_RINGS[0][1]
ASSEMBLED_QUAT = np.array([0.0, 1.0, 0.0, 0.0])  # wxyz, 180 deg about x: the peg's +y points down the bore
ASSEMBLED_POS = np.array([0.0, BORE_FLOOR_Y + PEG_TIP_Y, 0.0]) * MM
HOLE_QUAT = np.array([np.sqrt(0.5), np.sqrt(0.5), 0.0, 0.0])  # hole frame in the world: +90 deg about x, y -> z


def sector_vertices(y0, y1, r_in, r_out, a0, a1, half_angle):
    """Corners of the convex prism enclosing the annular sector [a0, a1] x [r_in, r_out] x [y0, y1]."""
    radii = [r_in, r_out / np.cos(half_angle)]
    return np.array([[r * np.cos(a), y, r * np.sin(a)] for y in (y0, y1) for r in radii for a in (a0, a1)])


def fmt(values):
    return " ".join(f"{v:.6g}" for v in np.ravel(values))


def build_model(asset_dir=ASSET_DIR, offscreen_size=(640, 480), n_sectors=N_SECTORS, margin=1e-4):
    """`margin` [m] reports contacts slightly before touching, so a resting contact shows up."""
    asset_dir = Path(asset_dir).resolve()
    meshes = []
    peg_geoms = [f'<geom name="peg_{name}" class="collision" type="cylinder" size="{r * MM}" '
                 f'fromto="{fmt([0, y0 * MM, 0, 0, y1 * MM, 0])}" rgba="{fmt(rgba)}"/>'
                 for name, (y0, y1, r), rgba in zip(PEG_CYLINDER_NAMES, PEG_CYLINDERS,
                                                    part_colors(len(PEG_CYLINDERS), (0.02, 0.15), 0.8))]

    # Collision geoms are named hole_plate, hole_ring<i> or, for the sectors, hole_ring<i>_<k>.
    y0, y1, half = HOLE_PLATE
    hole_geoms = [f'<geom name="hole_plate" class="collision" type="box" '
                  f'size="{fmt(np.array([half, (y1 - y0) / 2, half]) * MM)}" '
                  f'pos="{fmt([0, (y0 + y1) / 2 * MM, 0])}" rgba="0.4 0.5 0.7 0.8"/>']
    half_angle = np.pi / n_sectors
    for i, (y0, y1, r_in, r_out) in enumerate(HOLE_RINGS):
        if r_in == 0:
            hole_geoms.append(f'<geom name="hole_ring{i}" class="collision" type="cylinder" size="{r_out * MM}" '
                              f'fromto="{fmt([0, y0 * MM, 0, 0, y1 * MM, 0])}" rgba="0.4 0.5 0.7 0.8"/>')
            continue
        colors = part_colors(n_sectors, (0.5, 0.7), 0.8)
        for k in range(n_sectors):
            name = f"hole_ring{i}_{k:03d}"
            verts = sector_vertices(y0, y1, r_in, r_out, 2 * k * half_angle, 2 * (k + 1) * half_angle, half_angle)
            meshes.append(f'<mesh name="{name}" vertex="{fmt(verts * MM)}"/>')
            hole_geoms.append(f'<geom name="{name}" class="collision" type="mesh" mesh="{name}" '
                              f'rgba="{fmt(colors[k])}"/>')

    xml = f"""
<mujoco model="peg_hole_assembly">
  <compiler meshdir="{asset_dir}"/>
  <option gravity="0 0 0">
    <flag nativeccd="enable" multiccd="enable"/>
  </option>
  <statistic meansize="0.02"/>
  <visual>
    <global offwidth="{offscreen_size[0]}" offheight="{offscreen_size[1]}"/>
    <headlight ambient="0.3 0.3 0.3" diffuse="0.4 0.4 0.4"/>
    <scale framelength="1.2" framewidth="0.04"/>
  </visual>
  <asset>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.92 0.93 0.95" rgb2="0.6 0.65 0.72"
             width="256" height="256"/>
    <mesh name="peg" file="cylinder_real_peg.stl" scale="{MM} {MM} {MM}"/>
    <mesh name="hole" file="cylinder_real_hole.stl" scale="{MM} {MM} {MM}"/>
    {"".join(meshes)}
  </asset>
  <default>
    <default class="visual">
      <geom type="mesh" contype="0" conaffinity="0" group="0"/>
    </default>
    <default class="collision">
      <geom group="{COLLISION_GROUP}" margin="{margin}"/>
    </default>
  </default>
  <worldbody>
    <light directional="true" pos="0.2 -0.3 0.4" dir="-0.4 0.6 -0.8" diffuse="0.6 0.6 0.6"/>
    <light directional="true" pos="-0.2 0.3 0.4" dir="0.4 -0.6 -0.8" diffuse="0.3 0.3 0.3"/>
    <body name="hole" quat="{fmt(HOLE_QUAT)}">
      <geom name="hole" class="visual" mesh="hole" rgba="0.25 0.45 0.85 1"/>
      {"".join(hole_geoms)}
    </body>
    <!-- Posed through a free joint, not as a mocap body: MuJoCo treats mocap bodies as
         static and never collides two static bodies. -->
    <body name="peg">
      <freejoint name="peg"/>
      <geom name="peg" class="visual" mesh="peg" rgba="0.95 0.55 0.15 1"/>
      {"".join(peg_geoms)}
    </body>
  </worldbody>
</mujoco>"""
    return mujoco.MjModel.from_xml_string(xml)


def hole_to_world(points):
    """Points in the hole frame -> world frame."""
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, HOLE_QUAT)
    return np.atleast_2d(points) @ R.reshape(3, 3).T


def peg_tip(pos, quat):
    """Centre of the peg's tip face in the hole frame, for peg poses in the hole frame."""
    R = np.zeros(9)
    tips = []
    for p, q in zip(np.atleast_2d(pos), np.atleast_2d(quat)):
        mujoco.mju_quat2Mat(R, q)
        tips.append(p + R.reshape(3, 3)[:, 1] * PEG_TIP_Y * MM)
    return np.array(tips)


def set_peg_pose(m, d, pos, quat):
    """Place the peg at a pose in the hole frame and update kinematics and contacts."""
    world_pos, world_quat = np.zeros(3), np.zeros(4)
    mujoco.mju_mulPose(world_pos, world_quat, np.zeros(3), HOLE_QUAT, pos, quat)
    adr = m.jnt_qposadr[m.joint("peg").id]
    d.qpos[adr : adr + 7] = np.concatenate([world_pos, world_quat])
    mujoco.mj_forward(m, d)


def peg_hole_contacts(m, d):
    """(position, normal, signed distance) of every contact between the two bodies, in the world frame."""
    peg_id, hole_id = m.body("peg").id, m.body("hole").id
    contacts = []
    for c in d.contact[: d.ncon]:
        if {m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]} == {peg_id, hole_id}:
            contacts.append((c.pos.copy(), c.frame[:3].copy(), c.dist))
    return contacts
