#!/usr/bin/env python3
"""Render many example hook poses of each mode found by hook_mode_graph.py: a collage and a GIF per mode.

Examples are spread over the mode by farthest-point sampling in pose space (mm, and deg x 0.52 mm),
starting from the member nearest the mode's mean pose. Each example is shown from the front (looking
along the eyelet frame's y, the hook's plane of motion) and the side (looking along x, facing the loop),
with the eyelet fixed in view, and captioned with its pose and separation. The GIF visits its examples
in a nearest-neighbour tour, so consecutive frames differ little.

    mode_examples/mode_<m>_collage.png   --collage examples
    mode_examples/mode_<m>.gif           --gif examples

    conda run -n mujoco_env python scripts/render_mode_examples.py MODE_GRAPH_DIR
"""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation

from hook_eyelet_model import build_model, set_hook_pose

SCALE = np.array([1, 1, 1] + [30.0 * np.pi / 180] * 3)
TILE = 300
SURFACE, INK, INK_2 = (252, 252, 251), (11, 11, 11), (82, 81, 78)


def farthest_points(feat, n, start):
    chosen = [start]
    d = np.linalg.norm(feat - feat[start], axis=1)
    for _ in range(min(n, len(feat)) - 1):
        chosen.append(int(np.argmax(d)))
        d = np.minimum(d, np.linalg.norm(feat - feat[chosen[-1]], axis=1))
    return np.array(chosen)


def nn_tour(feat, idx):
    left, tour = list(idx[1:]), [idx[0]]
    while left:
        j = int(np.argmin(np.linalg.norm(feat[left] - feat[tour[-1]], axis=1)))
        tour.append(left.pop(j))
    return np.array(tour)


class Renderer:
    def __init__(self):
        self.m = build_model(offscreen_size=(TILE, TILE))
        self.d = mujoco.MjData(self.m)
        self.r = mujoco.Renderer(self.m, TILE, TILE)

    def views(self, pose, lookat, distance):
        set_hook_pose(self.m, self.d, pose[:3] * 1e-3,
                      Rotation.from_rotvec(pose[3:], degrees=True).as_quat(scalar_first=True))
        out = []
        for az in (90, 0):
            cam = mujoco.MjvCamera()
            cam.lookat[:], cam.distance, cam.azimuth, cam.elevation = lookat, distance, az, -10
            self.r.update_scene(self.d, camera=cam)
            out.append(Image.fromarray(self.r.render()))
        return out

    def close(self):
        self.r.close()


def tile(renderer, pose, sep, lookat, distance, font, small):
    front, side = renderer.views(pose, lookat, distance)
    img = Image.new("RGB", (2 * TILE, TILE + 34), SURFACE)
    img.paste(front, (0, 34))
    img.paste(side, (TILE, 34))
    draw = ImageDraw.Draw(img)
    draw.text((6, 3), f"pos ({pose[0]:.0f}, {pose[1]:.0f}, {pose[2]:.0f}) mm   "
                      f"rot ({pose[3]:.0f}, {pose[4]:.0f}, {pose[5]:.0f}) deg", fill=INK, font=font)
    draw.text((6, 19), f"separation {sep:.2f} mm        front | side", fill=INK_2, font=small)
    return img


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode_dir", type=Path)
    parser.add_argument("--collage", type=int, default=16)
    parser.add_argument("--gif", type=int, default=48)
    parser.add_argument("--frame-ms", type=int, default=350)
    args = parser.parse_args()
    s = np.load(args.mode_dir / "samples.npz")
    poses, seps, labels = s["poses"], s["seps"], s["labels"]
    report = json.loads((args.mode_dir / "modes.json").read_text())
    k, goal_mode = report["k"], report["goal_mode"]
    out = args.mode_dir / "mode_examples"
    out.mkdir(exist_ok=True)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 13)
        small = ImageFont.truetype("DejaVuSans.ttf", 11)
        big = ImageFont.truetype("DejaVuSans.ttf", 20)
    except OSError:
        font = small = big = ImageFont.load_default()

    renderer = Renderer()
    try:
        for m in range(k):
            members = np.flatnonzero(labels == m)
            feat = poses[members] * SCALE
            centre = int(np.argmin(np.linalg.norm(feat - feat.mean(0), axis=1)))
            spread = farthest_points(feat, max(args.collage, args.gif), centre)
            ex = poses[members[spread]]
            # frame the eyelet's loop and the examples' hook origins
            lookat = 0.5 * (np.array([0.0, 0.0, -5.0]) + np.median(ex[:, :3], axis=0)) * 1e-3
            distance = max(0.12, 2.2 * np.percentile(np.linalg.norm(ex[:, :3] * 1e-3 - lookat, axis=1), 90) + 0.07)
            name = f"mode {m}{' (goal)' if m == goal_mode else ''}"
            if m < len(report.get("modes", [])) and report["modes"][m].get("name"):
                name += f": {report['modes'][m]['name']}"

            tiles = [tile(renderer, poses[i], seps[i], lookat, distance, font, small)
                     for i in members[spread[: args.collage]]]
            cols = 4
            rows = int(np.ceil(len(tiles) / cols))
            w, h = tiles[0].size
            collage = Image.new("RGB", (cols * w, rows * h + 44), SURFACE)
            ImageDraw.Draw(collage).text((10, 10), f"{name}: {len(members):,} samples; {len(tiles)} examples spread "
                                                   "over the mode", fill=INK, font=big)
            for t, img in enumerate(tiles):
                collage.paste(img, ((t % cols) * w, 44 + (t // cols) * h))
            collage.save(out / f"mode_{m}_collage.png")

            frames = []
            for i in members[nn_tour(feat, spread[: args.gif])]:
                img = tile(renderer, poses[i], seps[i], lookat, distance, font, small)
                frame = Image.new("RGB", (img.width, img.height + 30), SURFACE)
                ImageDraw.Draw(frame).text((8, 6), f"{name}: {len(members):,} samples", fill=INK, font=font)
                frame.paste(img, (0, 30))
                frames.append(frame)
            frames[0].save(out / f"mode_{m}.gif", save_all=True, append_images=frames[1:], duration=args.frame_ms,
                           loop=0)
            print(f"{name}: collage and GIF written", flush=True)
    finally:
        renderer.close()


if __name__ == "__main__":
    main()
