"""Low-resolution videos of a policy on fixed validation cases, to watch training progress (train_hook_rl.py).

Each case is one start, in the believed eyelet frame, and the error of that belief (cases()). The wide task's
case is episode 27 of the final policy's evaluation (eval_hook_rl.py), with the hook's origin 73 mm beside and
26 mm below the seat, turned 160 deg and unthreaded, which that policy assembled in 34 steps (about its median).
The nominal task's cases start from the reference trajectories' pre-assembly pose itself, once with the belief
exact ("clean") and once off by VALIDATION_ERROR ("error"). The policy acts with its mean action, as in
evaluation. Each frame has two views at fixed cameras: a wide one that holds the start and the eyelet, and a
close-up facing the loop, where threading shows. A status bar gives the time, the linking number and the
distance to go, and the outcome over the last second of the video.
"""

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import mediapy
import mujoco
import numpy as np

import hook_rl_env as E
from assembly_trajectory_visual_verification import fit_camera, geom_world_vertices, put_text

VALIDATION_POS_MM = np.array([-72.872, -45.549, -25.918])
VALIDATION_QUAT = np.array([0.171177, -0.065426, -0.207079, 0.961008])  # wxyz
VALIDATION_ERROR = np.array([4.0, -4.0, 0.0, 3.0, -3.0, 3.0])  # mm, rotation vector deg
VIEWS = [("wide", 50, -25), ("loop close-up", 0, -10)]  # label, azimuth, elevation (deg)
BAR_H = 22
DT = E.SUBSTEPS * E.TIMESTEP
HOLD_END_S = 1.0


def cases(task):
    """{name: (start position mm, quaternion wxyz, belief error)} of a task's validation cases."""
    if task == "wide":
        return {"": (VALIDATION_POS_MM, VALIDATION_QUAT, np.zeros(6))}
    env = E.HookAssemblyBatch(1)
    start = (env.nominal_pos, env.nominal_rot.as_quat(scalar_first=True))
    return {"clean": (*start, np.zeros(6)), "error": (*start, VALIDATION_ERROR)}


class ValidationVideo:
    def __init__(self, env_kwargs, pos_mm, quat, err, tile_width=320, tile_height=240):
        self.case = (pos_mm, quat, err)
        self.env = E.HookAssemblyBatch(1, auto_reset=False, **env_kwargs)
        m, d = self.env.m, self.env.d[0]
        self.env.reset_to(0, *self.case)
        start = geom_world_vertices(m, d, "hook")
        self.env.set_pose(0, np.zeros(3), np.array([1.0, 0, 0, 0]))
        goal = geom_world_vertices(m, d, "hook")
        eyelet = geom_world_vertices(m, d, "eyelet")
        loop_width = np.ptp(eyelet[:, :2], axis=0).max()
        loop = eyelet[eyelet[:, 2] > -loop_width]
        near = np.vstack([goal, eyelet])
        framing = [np.vstack([start, goal, loop]), near[np.linalg.norm(near, axis=1) < loop_width]]
        self.cameras = [fit_camera(f, az, el, m.vis.global_.fovy, tile_width / tile_height, margin=1.3)
                        for f, (_, az, el) in zip(framing, VIEWS)]
        self.renderer = mujoco.Renderer(m, tile_height, tile_width)

    def run(self, policy, label=""):
        """Roll out policy (observations (1, OBS_DIM) -> actions (1, 6 or 7)) from the validation start: (frames,
        result: success, out, stopped (declared done), steps, final distance to go mm, final linking number,
        return)."""
        obs = self.env.reset_to(0, *self.case)
        frames = [self.frame(label, 0)]
        for step in range(1, self.env.max_steps + 1):
            obs, _, _, _, info = self.env.step(policy(obs))
            frames.append(self.frame(label, step))
            if info:
                break
        result = {"success": bool(info["success"][0]), "out": bool(info["out"][0]),
                  "stopped": bool(info["stopped"][0]), "steps": step,
                  "final_dist_mm": float(info["final_dist_mm"][0]), "final_link": float(info["final_link"][0]),
                  "return": float(info["return"][0])}
        outcome = ("ASSEMBLED" if result["success"] else "OUT OF BOUNDS" if result["out"]
                   else "DECLARED DONE, NOT SEATED" if result["stopped"] else "TIMED OUT")
        frames += [self.frame(label, step, outcome)] * round(HOLD_END_S / DT)
        return frames, result

    def frame(self, label, step, note=""):
        _, _, _, link, dist = self.env.observe()
        tiles = []
        for (name, *_), cam in zip(VIEWS, self.cameras):
            self.renderer.update_scene(self.env.d[0], camera=cam)
            tile = self.renderer.render().copy()
            put_text(tile, name, (6, 16), scale=0.45)
            tiles.append(tile)
        img = np.hstack(tiles)
        bar = np.zeros((BAR_H, img.shape[1], 3), np.uint8)
        put_text(bar, f"{label}  t {step * DT:4.2f} s  linking {link[0]:+.2f}  to go {dist[0]:5.1f} mm  {note}",
                 (6, 16), scale=0.45)
        return np.vstack([img, bar])

    def close(self):
        self.renderer.close()


def save(frames, path):
    mediapy.write_video(path, frames, fps=round(1 / DT))
