#!/usr/bin/env python3
"""Render a video of the hook following an assembly trajectory into the eyelet.

The eyelet is fixed at the world origin and the hook is posed kinematically at each
waypoint of the trajectory CSV (see hook_eyelet_model.py for the frame conventions and
the convex collision parts). Every frame shows the resulting hook/eyelet contacts.

Run with the existing `mujoco_env` conda env:
    conda run -n mujoco_env python scripts/convex_decompose_meshes.py   # once
    conda run -n mujoco_env python scripts/assembly_trajectory_visual_verification.py [--csv ...]
"""

import argparse
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import mediapy
import mujoco
import numpy as np

from hook_eyelet_model import (ASSET_DIR, COLLISION_GROUP, EYELET_YAW_DEG, HOOK_YAW_DEG, INCH, Separation,
                               build_model, hook_eyelet_contacts, load_trajectory, set_hook_pose)

REPO_ROOT = Path(__file__).resolve().parents[1]

# (label, azimuth, elevation, framing) for each tile of the 2x2 video grid.
VIEWS = [
    ("front (hook plane)", 90, 0, "wide"),
    ("side (loop face)", 0, 0, "wide"),
    ("isometric close-up", 50, -25, "loop"),
    ("front close-up", 90, 0, "loop"),
]
STATUS_BAR_H = 36
TOUCH_RGB = (255, 215, 0)
PENETRATION_RGB = (255, 40, 40)
SEPARATION_RGB = (0, 190, 255)
CLEAR_TEXT_RGB = (150, 220, 150)


def geom_world_vertices(m, d, geom_name):
    g = m.geom(geom_name)
    mesh_id = m.geom_dataid[g.id]
    start = m.mesh_vertadr[mesh_id]
    verts = m.mesh_vert[start : start + m.mesh_vertnum[mesh_id]]
    return d.geom_xpos[g.id] + verts @ d.geom_xmat[g.id].reshape(3, 3).T


def fit_camera(points, azimuth, elevation, fovy, aspect, margin=1.15):
    """Free camera at the given angles, placed so that all points fit in view."""
    az, el = np.deg2rad(azimuth), np.deg2rad(elevation)
    forward = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    up = np.array([-np.sin(el) * np.cos(az), -np.sin(el) * np.sin(az), np.cos(el)])
    right = np.cross(forward, up)

    lookat = 0.5 * (points.min(0) + points.max(0))
    rel = points - lookat
    half_w = np.abs(rel @ right).max() * margin
    half_h = np.abs(rel @ up).max() * margin
    half_depth = np.abs(rel @ forward).max()
    tan_half = np.tan(np.deg2rad(fovy) / 2)

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = lookat
    cam.azimuth, cam.elevation = azimuth, elevation
    cam.distance = max(half_h / tan_half, half_w / (tan_half * aspect)) + half_depth
    return cam


def add_polyline(scene, points, rgba, width):
    for a, b in zip(points[:-1], points[1:]):
        if scene.ngeom >= scene.maxgeom:
            return
        g = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.zeros(9),
                            np.asarray(rgba, dtype=np.float32))
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, width, a, b)
        scene.ngeom += 1


def project(scene, points, width, height):
    """Pixel coordinates of world points in the image rendered from `scene`."""
    cam = scene.camera[0]
    forward, up = np.array(cam.forward), np.array(cam.up)
    # camera[0] and [1] are the stereo eyes; a mono render uses their midpoint.
    cam_pos = 0.5 * (np.array(scene.camera[0].pos) + np.array(scene.camera[1].pos))
    rel = np.atleast_2d(points) - cam_pos
    x, y, z = rel @ np.cross(forward, up), rel @ up, rel @ forward
    f = height * cam.frustum_near / (cam.frustum_top - cam.frustum_bottom)
    return np.stack([width / 2 + f * x / z, height / 2 - f * y / z], axis=1).round().astype(int)


def draw_contacts(img, scene, contacts, tol, normal_length=0.004):
    """Contacts are drawn over the image: they lie inside the overlapping meshes."""
    h, w = img.shape[:2]
    for c_pos, c_normal, c_dist in contacts:
        color = PENETRATION_RGB if c_dist < -tol else TOUCH_RGB
        p, q = project(scene, [c_pos, c_pos + normal_length * c_normal], w, h)
        cv2.line(img, tuple(p), tuple(q), color, 1, cv2.LINE_AA)
        cv2.circle(img, tuple(p), 4, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(img, tuple(p), 3, color, -1, cv2.LINE_AA)


def draw_separation(img, scene, points):
    """Segment between the closest points of the two bodies."""
    h, w = img.shape[:2]
    p, q = project(scene, points, w, h)
    cv2.line(img, tuple(p), tuple(q), SEPARATION_RGB, 2, cv2.LINE_AA)
    for end in (p, q):
        cv2.circle(img, tuple(end), 3, SEPARATION_RGB, -1, cv2.LINE_AA)


def put_text(img, text, org, scale=0.6, color=(255, 255, 255)):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", type=Path, default=ASSET_DIR / "hook_assembly_ideal_trajectory.csv")
    parser.add_argument("--asset-dir", type=Path, default=ASSET_DIR,
                        help="directory with hook.stl, eyelet.stl and their convex/ parts")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "outputs" / "hook_assembly_trajectory.mp4")
    parser.add_argument("--mesh-scale", type=float, default=INCH, help="STL units -> metres")
    parser.add_argument("--eyelet-yaw-deg", type=float, default=EYELET_YAW_DEG,
                        help="rotation of the eyelet STL about z relative to the trajectory frame")
    parser.add_argument("--hook-yaw-deg", type=float, default=HOOK_YAW_DEG,
                        help="rotation of the hook STL about its own z relative to the trajectory's hook frame")
    parser.add_argument("--penetration-tol-mm", type=float, default=0.25,
                        help="contacts deeper than this are flagged as penetration rather than touching "
                             "(the convex parts bulge up to ~0.25 mm past the true surface)")
    parser.add_argument("--show-collision", action="store_true",
                        help="render the convex collision parts instead of the original meshes")
    parser.add_argument("--fps", type=int, default=30, help="one video frame per trajectory waypoint")
    parser.add_argument("--tile-width", type=int, default=640)
    parser.add_argument("--tile-height", type=int, default=480)
    parser.add_argument("--hold-start", type=float, default=0.5, help="seconds to hold the first frame")
    parser.add_argument("--hold-end", type=float, default=1.5, help="seconds to hold the last frame")
    args = parser.parse_args()

    m = build_model(args.asset_dir, args.mesh_scale, args.eyelet_yaw_deg, args.hook_yaw_deg,
                    (args.tile_width, args.tile_height))
    d = mujoco.MjData(m)
    traj, pos, quat = load_trajectory(args.csv)
    tol = args.penetration_tol_mm * 1e-3

    # Frame the wide views on everything the hook sweeps through plus the eyelet loop,
    # and the close-ups on the mating region around the assembled origin.
    swept = []
    for i in range(len(pos)):
        set_hook_pose(m, d, pos[i], quat[i])
        swept.append(geom_world_vertices(m, d, "hook"))
    eyelet = geom_world_vertices(m, d, "eyelet")
    loop_width = np.ptp(eyelet[:, :2], axis=0).max()
    loop = eyelet[eyelet[:, 2] > -loop_width]
    everything = np.vstack(swept + [eyelet])
    framing = {"wide": np.vstack(swept + [loop]),
               "loop": everything[np.linalg.norm(everything, axis=1) < loop_width]}

    fovy = m.vis.global_.fovy
    aspect = args.tile_width / args.tile_height
    cameras = [fit_camera(framing[f], az, el, fovy, aspect) for _, az, el, f in VIEWS]

    opt = mujoco.MjvOption()
    opt.frame = mujoco.mjtFrame.mjFRAME_BODY
    if args.show_collision:
        opt.geomgroup[0], opt.geomgroup[COLLISION_GROUP] = 0, 1
    path_width = 0.00025

    separation = Separation(m)
    renderer = mujoco.Renderer(m, args.tile_height, args.tile_width)
    frames, separations = [], []
    try:
        for i in range(len(pos)):
            set_hook_pose(m, d, pos[i], quat[i])
            contacts = hook_eyelet_contacts(m, d)
            sep, closest = separation(m, d)
            separations.append(sep)

            tiles = []
            for (label, *_), cam in zip(VIEWS, cameras):
                renderer.update_scene(d, camera=cam, scene_option=opt)
                add_polyline(renderer.scene, pos[i:], (0.2, 0.2, 0.2, 0.6), path_width)
                add_polyline(renderer.scene, pos[: i + 1], (0.1, 0.75, 0.2, 1.0), path_width)
                tile = renderer.render().copy()
                if sep > 0 and closest is not None:
                    draw_separation(tile, renderer.scene, closest)
                draw_contacts(tile, renderer.scene, contacts, tol)
                put_text(tile, label, (10, 25))
                tiles.append(tile)
            grid = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])])

            bar = np.zeros((STATUS_BAR_H, grid.shape[1], 3), dtype=np.uint8)
            angle = np.rad2deg(2 * np.arccos(np.clip(abs(quat[i, 0]), 0.0, 1.0)))
            p_mm = pos[i] * 1e3
            seg = f"seg {int(traj['seg_idx'][i]):2d}  " if "seg_idx" in traj.dtype.names else ""
            put_text(bar, f"t = {traj['t'][i]:.3f}  {seg}"
                          f"pos [mm] ({p_mm[0]:+6.2f}, {p_mm[1]:+6.2f}, {p_mm[2]:+6.2f})  "
                          f"rot {angle:5.1f} deg  {i + 1}/{len(pos)}",
                     (10, 24), scale=0.55)
            text = f"separation {sep * 1e3:+.2f} mm" if closest is not None else "separation > 50 mm"
            color = PENETRATION_RGB if sep < -tol else TOUCH_RGB if sep <= 0 else CLEAR_TEXT_RGB
            put_text(bar, text, (grid.shape[1] - 260, 24), scale=0.55, color=color)
            frames.append(np.vstack([grid, bar]))
    finally:
        renderer.close()
    separations = np.array(separations)

    frames = ([frames[0]] * round(args.hold_start * args.fps) + frames
              + [frames[-1]] * round(args.hold_end * args.fps))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    mediapy.write_video(args.out, frames, fps=args.fps)
    print(f"Wrote {len(frames)} frames ({len(frames) / args.fps:.1f} s) to {args.out}")

    worst = np.argmin(separations)
    print(f"Minimum separation {separations[worst] * 1e3:+.2f} mm at t = {traj['t'][worst]:.3f}; "
          f"penetration deeper than {args.penetration_tol_mm} mm in {(separations < -tol).sum()}/{len(pos)} waypoints")


if __name__ == "__main__":
    main()
