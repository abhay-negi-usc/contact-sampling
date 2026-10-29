#!/usr/bin/env python3
"""Render a video of the cylindrical peg following an assembly trajectory into the hole.

The hole is fixed at the world origin and the peg is posed kinematically at each waypoint of
the trajectory CSV (peg pose in the hole frame; see peg_hole_model.py for the conventions and the
collision geometry). The bore is hidden inside the hole, so the lower tiles render the hole
see-through and the collision geometry. Every frame shows the resulting peg/hole contacts.

Run with the existing `mujoco_env` conda env:
    conda run -n mujoco_env python scripts/peg_hole_assembly_trajectory.py   # writes the CSV
    conda run -n mujoco_env python scripts/peg_hole_trajectory_visual_verification.py [--csv ...]
"""

import argparse
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mediapy
import mujoco
import numpy as np

from assembly_trajectory_visual_verification import (PENETRATION_RGB, STATUS_BAR_H, TOUCH_RGB, add_polyline,
                                                      draw_contacts, fit_camera, geom_world_vertices, put_text)
from hook_eyelet_model import load_trajectory
from peg_hole_model import (ASSET_DIR, BORE_TOP_Y, COLLISION_GROUP, MM, build_model, hole_to_world,
                            peg_hole_contacts, peg_tip, set_peg_pose)
from peg_hole_assembly_trajectory import SEGMENTS

REPO_ROOT = Path(__file__).resolve().parents[1]

# (label, azimuth, elevation, framing, rendering) for each tile of the 2x2 video grid.
VIEWS = [
    ("side", 90, -10, "wide", "opaque"),
    ("isometric", 45, -30, "wide", "opaque"),
    ("x-ray side close-up", 90, -5, "bore", "xray"),
    ("collision geometry close-up", 45, -20, "bore", "collision"),
]
XRAY_ALPHA = 0.25


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", type=Path, default=ASSET_DIR / "peg_hole_assembly_trajectory.csv")
    parser.add_argument("--asset-dir", type=Path, default=ASSET_DIR,
                        help="directory with cylinder_real_peg.stl and cylinder_real_hole.stl")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "outputs" / "peg_hole_assembly_trajectory.mp4")
    parser.add_argument("--penetration-tol-mm", type=float, default=0.05,
                        help="contacts deeper than this are flagged as penetration rather than touching "
                             "(the bore's collision sectors bulge up to ~0.016 mm past the true surface)")
    parser.add_argument("--fps", type=int, default=30, help="one video frame per trajectory waypoint")
    parser.add_argument("--tile-width", type=int, default=640)
    parser.add_argument("--tile-height", type=int, default=480)
    parser.add_argument("--hold-start", type=float, default=0.5, help="seconds to hold the first frame")
    parser.add_argument("--hold-end", type=float, default=1.5, help="seconds to hold the last frame")
    args = parser.parse_args()

    m = build_model(args.asset_dir, (args.tile_width, args.tile_height))
    d = mujoco.MjData(m)
    traj, pos, quat = load_trajectory(args.csv)
    tips = peg_tip(pos, quat)
    tips_world = hole_to_world(tips)
    tol = args.penetration_tol_mm * 1e-3

    # Frame the wide views on everything the peg sweeps through plus the hole, and the
    # close-ups on the hole's tube and the part of the peg that enters it.
    swept = []
    for i in range(len(pos)):
        set_peg_pose(m, d, pos[i], quat[i])
        swept.append(geom_world_vertices(m, d, "peg"))
    hole = geom_world_vertices(m, d, "hole")
    everything = np.vstack(swept + [hole])
    tube_radius = 30 * MM
    framing = {"wide": everything,
               "bore": everything[(np.linalg.norm(everything[:, :2], axis=1) < tube_radius)
                                  & (everything[:, 2] > 15 * MM) & (everything[:, 2] < (BORE_TOP_Y + 30) * MM)]}

    fovy = m.vis.global_.fovy
    aspect = args.tile_width / args.tile_height
    cameras = [fit_camera(framing[f], az, el, fovy, aspect) for _, az, el, f, _ in VIEWS]

    options = {}
    for rendering in ("opaque", "xray", "collision"):
        opt = mujoco.MjvOption()
        if rendering == "collision":
            opt.geomgroup[0], opt.geomgroup[COLLISION_GROUP] = 0, 1
        options[rendering] = opt
    hole_geoms = m.geom_bodyid == m.body("hole").id
    opaque_rgba = m.geom_rgba.copy()
    xray_rgba = opaque_rgba.copy()
    xray_rgba[hole_geoms, 3] = XRAY_ALPHA
    path_width = 0.0004

    renderer = mujoco.Renderer(m, args.tile_height, args.tile_width)
    frames, penetrating_frames, deepest = [], [], (0.0, None)
    try:
        for i in range(len(pos)):
            set_peg_pose(m, d, pos[i], quat[i])
            contacts = peg_hole_contacts(m, d)
            min_dist = min((c[2] for c in contacts), default=np.inf)
            if min_dist < -tol:
                penetrating_frames.append(i)
            if min_dist < deepest[0]:
                deepest = (min_dist, i)

            tiles = []
            for (label, *_, rendering), cam in zip(VIEWS, cameras):
                m.geom_rgba[:] = opaque_rgba if rendering == "opaque" else xray_rgba
                renderer.update_scene(d, camera=cam, scene_option=options[rendering])
                add_polyline(renderer.scene, tips_world[i:], (0.2, 0.2, 0.2, 0.6), path_width)
                add_polyline(renderer.scene, tips_world[: i + 1], (0.1, 0.75, 0.2, 1.0), path_width)
                tile = renderer.render().copy()
                draw_contacts(tile, renderer.scene, contacts, tol, normal_length=0.008)
                put_text(tile, label, (10, 25))
                tiles.append(tile)
            grid = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])])

            bar = np.zeros((STATUS_BAR_H, grid.shape[1], 3), dtype=np.uint8)
            axis = tips[i] - pos[i]
            tilt = np.rad2deg(np.arccos(np.clip(-axis[1] / np.linalg.norm(axis), -1.0, 1.0)))
            depth_mm = BORE_TOP_Y - tips[i, 1] / MM
            p_mm = pos[i] / MM
            seg = f"{SEGMENTS[int(traj['seg_idx'][i])]:8s} " if "seg_idx" in traj.dtype.names else ""
            put_text(bar, f"t = {traj['t'][i]:.3f}  {seg}"
                          f"pos [mm] ({p_mm[0]:+.1f}, {p_mm[1]:+.1f}, {p_mm[2]:+.1f})  "
                          f"tilt {tilt:4.1f} deg  depth {depth_mm:+5.1f} mm  {i + 1}/{len(pos)}",
                     (10, 24), scale=0.55)
            if contacts:
                text = f"{len(contacts)} contacts, min dist {min_dist * 1e3:+.2f} mm"
                color = PENETRATION_RGB if min_dist < -tol else TOUCH_RGB
            else:
                text, color = "no contact", (150, 220, 150)
            put_text(bar, text, (grid.shape[1] - 330, 24), scale=0.55, color=color)
            frames.append(np.vstack([grid, bar]))
    finally:
        renderer.close()

    frames = ([frames[0]] * round(args.hold_start * args.fps) + frames
              + [frames[-1]] * round(args.hold_end * args.fps))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # mediapy's default for frames this large (qp 28) leaves ghosts of the peg on the plain background.
    mediapy.write_video(args.out, frames, fps=args.fps, qp=20)
    print(f"Wrote {len(frames)} frames ({len(frames) / args.fps:.1f} s) to {args.out}")

    if penetrating_frames:
        print(f"Penetration deeper than {args.penetration_tol_mm} mm in {len(penetrating_frames)}/{len(pos)} "
              f"waypoints; deepest {-deepest[0] * 1e3:.2f} mm at t = {traj['t'][deepest[1]]:.3f}")
    else:
        print(f"No penetration deeper than {args.penetration_tol_mm} mm along the trajectory")


if __name__ == "__main__":
    main()
