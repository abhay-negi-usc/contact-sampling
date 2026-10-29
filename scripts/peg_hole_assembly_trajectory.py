#!/usr/bin/env python3
"""Write a nominal assembly trajectory of the cylindrical peg into the hole.

Poses are the peg in the hole frame (see peg_hole_model.py), in metres, on a uniform normalised
time grid t in [0, 1]. The motion has three segments, each starting and ending at rest under a
minimum-jerk time scaling. The tip moves in a straight line and the orientation follows a slerp:

    0 approach  from a tilted start beside the hole to coaxial, tip --pre-insert-mm above the bore
    1 descend   along the axis to --entry-mm above the bore
    2 insert    along the axis until the tip rests on the bore floor

The start tilt leans the tip towards the hole. The columns match the hook trajectories:
t,x,y,z,qx,qy,qz,qw,seg_idx,alpha (alpha is the fraction of the segment's time).

    conda run -n mujoco_env python scripts/peg_hole_assembly_trajectory.py
"""

import argparse
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from peg_hole_model import (ASSEMBLED_POS, ASSEMBLED_QUAT, ASSET_DIR, BORE_FLOOR_Y, BORE_TOP_Y, MM, build_model,
                            peg_hole_contacts, peg_tip, set_peg_pose)

SEGMENTS = ["approach", "descend", "insert"]


def min_jerk(alpha):
    return alpha**3 * (10 - 15 * alpha + 6 * alpha**2)


def pose_from_tip(tip, rot):
    """Peg pose (position, wxyz quaternion) placing its tip at `tip` with orientation `rot`."""
    quat = rot.as_quat(scalar_first=True)
    return tip - peg_tip(np.zeros(3), quat)[0], quat


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=ASSET_DIR / "peg_hole_assembly_trajectory.csv")
    parser.add_argument("--start-offset-mm", type=float, nargs=2, default=[40.0, 25.0],
                        help="start tip offset from the hole axis along the hole's x and z")
    parser.add_argument("--start-height-mm", type=float, default=60.0, help="start tip height above the bore")
    parser.add_argument("--start-tilt-deg", type=float, default=25.0, help="start tilt of the peg from the hole axis")
    parser.add_argument("--pre-insert-mm", type=float, default=15.0, help="tip height above the bore after approach")
    parser.add_argument("--entry-mm", type=float, default=2.0, help="tip height above the bore where insertion starts")
    parser.add_argument("--durations", type=float, nargs=3, default=[0.45, 0.15, 0.40],
                        help="relative durations of the approach, descend and insert segments")
    parser.add_argument("--n-waypoints", type=int, default=241)
    args = parser.parse_args()

    assembled = Rotation.from_quat(ASSEMBLED_QUAT, scalar_first=True)
    offset = np.array([args.start_offset_mm[0], 0.0, args.start_offset_mm[1]])
    # Tilt about the horizontal axis perpendicular to the offset, so the tip leans towards the axis.
    tilt_axis = np.cross([0.0, -1.0, 0.0], -offset)
    start_rot = Rotation.from_rotvec(np.deg2rad(args.start_tilt_deg) * tilt_axis / np.linalg.norm(tilt_axis)) * assembled

    above = lambda h: np.array([0.0, BORE_TOP_Y + h, 0.0]) * MM
    via_tips = [offset * MM + above(args.start_height_mm), above(args.pre_insert_mm), above(args.entry_mm),
                np.array([0.0, BORE_FLOOR_Y, 0.0]) * MM]
    via_rots = Rotation.concatenate([start_rot, assembled, assembled, assembled])

    t = np.linspace(0.0, 1.0, args.n_waypoints)
    bounds = np.r_[0.0, np.cumsum(args.durations)] / np.sum(args.durations)
    seg = np.minimum(np.searchsorted(bounds, t, side="right") - 1, len(SEGMENTS) - 1)
    alpha = (t - bounds[seg]) / np.diff(bounds)[seg]
    pos, quat = np.zeros((len(t), 3)), np.zeros((len(t), 4))
    for i, (k, a) in enumerate(zip(seg, alpha)):
        s = min_jerk(a)
        rot = Slerp([0.0, 1.0], via_rots[k : k + 2])(s)
        pos[i], quat[i] = pose_from_tip((1 - s) * via_tips[k] + s * via_tips[k + 1], rot)
    pos[-1], quat[-1] = ASSEMBLED_POS, ASSEMBLED_QUAT  # exact, free of round-off

    m = build_model()
    d = mujoco.MjData(m)
    min_dist = np.full(len(t), np.inf)
    for i in range(len(t)):
        set_peg_pose(m, d, pos[i], quat[i])
        min_dist[i] = min((c[2] for c in peg_hole_contacts(m, d)), default=np.inf)
    first = np.argmax(np.isfinite(min_dist)) if np.isfinite(min_dist).any() else None
    print(f"First contact within {m.geom_margin.max() * 1e3:.2f} mm: "
          + (f"waypoint {first + 1}/{len(t)} (t = {t[first]:.3f}), min dist {min_dist[first] * 1e3:+.3f} mm"
             if first is not None else "none"))
    print(f"Assembled pose: min dist {min_dist[-1] * 1e3:+.3f} mm")

    xyzw = np.roll(quat, -1, axis=1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(args.out, np.column_stack([t, pos, xyzw, seg, alpha]), delimiter=",",
               header="t,x,y,z,qx,qy,qz,qw,seg_idx,alpha", comments="", fmt="%.10g")
    print(f"Wrote {len(t)} waypoints to {args.out}")


if __name__ == "__main__":
    main()
