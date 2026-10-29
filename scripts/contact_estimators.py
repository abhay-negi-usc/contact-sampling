"""Contact-manifold estimators of the eyelet pose for the hook RL benchmark, from sunrise-wrapper-abhay.

Two families, used as they are in that repository's closed-loop experiments:
  "icp_sdf"  the proposed method: wrench-augmented multi-start ICP warm start, then the trust-region distance-field
             refinement ("MIP" SDF2; methodology_merged.tex), production settings (config/experiment.yaml);
  "icp"      its ICP warm start alone;
  "cpf"      the conventional particle filter on the same map and metric (particle_filter section; phi = 0);
  "mpf"      Koval et al.'s manifold particle filter (the same, with dual particles, phi = 0.1).

Everything an estimator sees is what the robot would: hook poses in the robot's frame (hook_rl_env.py: the
episode's initial believed eyelet frame, never the simulation's world frame, which is the true eyelet frame),
contact wrenches in the hook frame (the sensor's), and its current belief of the eyelet in the robot's frame as
the prior (live seeding: each call starts again from the belief and uses every observation so far). The eyelet's
z in the robot's frame is known (0): the estimators pin known axes to the values of the pose they are handed as
`true_base_to_hole`, so they get a stand-in that holds only that, the identity (the ICP and refinement read
only its known axes; the particle filter's known-axis lock, relative to it, then pins z = 0 and nothing else).
known_axes may also hold the rotations ("a", "b", "c": about z, y, x), which then stay at the episode's
initial belief (the identity in the robot's frame): translation-only estimation. prior_box (x, y, z mm, rotation
vector deg) rejects an estimate outside that box about the initial belief (the support of the belief error's
prior, inflated), keeping the current belief: with few or uninformative contacts the estimators can leap far off.

Poses at this interface: positions mm, quaternions wxyz; the estimators work in m and ZYX degrees (xyzabc).
"""

import contextlib
import io
import sys

import numpy as np
from scipy.spatial.transform import Rotation

SUNRISE = "/home/rp/abhay_ws/sunrise-wrapper-abhay"
CICP = "/home/rp/abhay_ws/cicp"
for _p in (CICP, SUNRISE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import cicp.cicp_util as SU  # noqa: E402
from loguru import logger  # noqa: E402

logger.remove()  # the estimators log every call
KINDS = ("icp_sdf", "icp", "cpf", "mpf")
MAP_FORMAT = {"translation_unit": "mm", "rotation_type": "quat", "quat_convention": "xyzw"}
KNOWN_AXES = ["z"]

# Production settings (config/experiment.yaml, geometries/hook_*.yaml), with the multi-start guess box and the
# particle prior sized for this benchmark's belief error (+-5 mm, +-5 deg; hook_v6 used 7.5 for 5).
GUESS_RANGE = [7.5, 7.5, 7.5, 7.5, 7.5, 7.5]  # mm, deg, in the estimated eyelet frame
ICP_INIT = dict(nn_backend="kdtree", icp_n_iter=10, obs_boundary_filter=False, use_wrench=True,
                wrench_method="map_augment", wrench_ratio_radius=1.0, torque_to_force_weight_ratio=10.0,
                residual_weight=False, residual_weight_scale=5.0, n_best_mean=10, use_lie_group=True,
                dim_weights=[1, 1, 1, 1, 1, 1], pin_known_per_iter=True, estimation_plot_range_mm=10.0,
                estimation_plot_range_deg=10.0)
ICP_CALL = dict(n_iter=10, num_guesses=25, aggregate="median", sample_size=10_000)  # mip.icp_call_kwargs
MIP = dict(formulation="sdf2", mm_per_deg=1.0, k_candidates=1, bounds_xi=[5.0] * 6, K_max=5,
           reject_on_regress=True, reject_margin=0.05, time_limit=150, mip_gap=0.05, threads=0,
           verbose_solver=False, warm_start_with_icp=True, icp_call_kwargs=ICP_CALL)
OUTER_SAMPLES = 100_000  # experiment.icp_sample_size
PF = dict(n_particles=500, eps=3.0, process_noise=[0.05] * 6, update_every=0.5, restart_each_call=True,
          max_updates=300, prior_mode="hole_additive", prior_support=False, prior_inflate=1.0,
          contact_force_threshold=None, mh_every=0, candidate_query=64, candidate_k=32, tr_correction=True,
          kde_bw_scale=1.0)


def to_xyzabc_m(pos_mm, quat_wxyz):
    """(N, 6) m, ZYX deg from positions mm and quaternions wxyz."""
    pos_mm, quat_wxyz = np.atleast_2d(pos_mm), np.atleast_2d(quat_wxyz)
    T = np.tile(np.eye(4), (len(pos_mm), 1, 1))
    T[:, :3, :3] = Rotation.from_quat(quat_wxyz, scalar_first=True).as_matrix()
    T[:, :3, 3] = pos_mm * 1e-3
    return SU.batch_matrices_to_poses_xyzabc(T)


def from_xyzabc_m(x):
    """(position mm, quaternion wxyz) of one xyzabc pose (m, ZYX deg)."""
    T = SU.batch_poses_xyzabc_to_matrices(np.asarray(x, dtype=np.float64).reshape(1, 6))[0]
    return T[:3, 3] * 1e3, Rotation.from_matrix(T[:3, :3]).as_quat(scalar_first=True)


class ContactEstimator:
    """One estimator over a contact map, reused across episodes (estimate() is stateless between calls)."""

    def __init__(self, kind, map_path, wrench_weight=10.0, map_downsample_n=None, seed=78, guess_range=GUESS_RANGE,
                 pf=None, known_axes=KNOWN_AXES, prior_box=None, icp_call=None):
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}")
        self.kind = kind
        self.icp_call = {**ICP_CALL, **(icp_call or {})}  # e.g. more multi-start guesses for a wider prior
        self.prior_box = None if prior_box is None else np.asarray(prior_box, dtype=np.float64)
        self.rejected = 0
        common = dict(map_path=str(map_path), map_format=MAP_FORMAT, map_downsample_n=map_downsample_n,
                      numpy_seed=seed, symmetry_axes=[], symmetry_success_value_m=[], known_axes=known_axes)
        with contextlib.redirect_stdout(io.StringIO()):
            if kind in ("icp_sdf", "icp"):
                from experiments.pose_estimation_mip import MIPPoseEstimator
                icp_init = {**ICP_INIT, **common, "wrench_weight": wrench_weight, "icp_guess_uniform_range": guess_range}
                self.est = MIPPoseEstimator(
                    **{k: common[k] for k in ("map_path", "map_format", "map_downsample_n", "numpy_seed",
                                              "symmetry_axes", "symmetry_success_value_m", "known_axes")},
                    pin_known_per_iter=True, estimation_plot_range_mm=10.0, estimation_plot_range_deg=10.0,
                    torque_to_force_weight_ratio=10.0, dim_weights=[1, 1, 1, 1, 1, 1], **MIP,
                    icp_init_kwargs=icp_init)
                self.est.set_plot_each_call(False)
                if kind == "icp":  # the warm start alone: call the polish ICP directly
                    self.est = self.est._icp_polish
            else:
                from experiments.pose_estimation.manifold_particle_filter import (ContactManifold,
                                                                                  ManifoldParticleFilter)
                manifold = ContactManifold(**common, use_wrench=True, wrench_weight=wrench_weight,
                                           torque_to_force_weight_ratio=10.0, mm_per_deg=1.0)
                hw = [v * (1e-3 if i < 3 else 1.0) for i, v in enumerate(guess_range)]
                cfg = {**PF, **(pf or {}), "phi": 0.0 if kind == "cpf" else 0.1}
                self.est = ManifoldParticleFilter(manifold, **cfg, prior_half_width=hw, seed=seed)

    def estimate(self, poses, wrenches, prior_pos_mm, prior_quat):
        """Eyelet pose in the robot's frame (mm, wxyz) from the hook poses (k, 7: mm, wxyz) and hook-frame
        wrenches (k, 6) recorded so far, starting from the current belief."""
        obs = to_xyzabc_m(poses[:, :3], poses[:, 3:])
        prior = to_xyzabc_m(prior_pos_mm, prior_quat)
        known = np.zeros((1, 6))  # stand-in: only z = 0 in the robot's frame is read (module docstring)
        self.est.reset()
        self.est.accumulate(obs, wrench=np.asarray(wrenches, dtype=np.float64))
        with contextlib.redirect_stdout(io.StringIO()):
            if self.kind == "icp_sdf":
                out = self.est.estimate_base_to_hole(current_base_to_hole_estimate=prior,
                                                     n_iter=self.icp_call["n_iter"],
                                                     num_guesses=self.icp_call["num_guesses"],
                                                     aggregate=self.icp_call["aggregate"],
                                                     sample_size=OUTER_SAMPLES, true_base_to_hole=known)
            elif self.kind == "icp":
                out = self.est.estimate_base_to_hole(current_base_to_hole_estimate=prior, true_base_to_hole=known,
                                                     **self.icp_call)
            else:
                out = self.est.estimate_base_to_hole(prior, true_base_to_hole=known)
        pos, quat = from_xyzabc_m(out)
        if self.prior_box is not None:  # outside the prior's support around the initial belief: keep the belief
            turn = np.degrees(Rotation.from_quat(quat, scalar_first=True).as_rotvec())
            if np.any(np.abs(pos) > self.prior_box[:3]) or np.any(np.abs(turn) > self.prior_box[3:]):
                self.rejected += 1
                return np.asarray(prior_pos_mm, dtype=np.float64), np.asarray(prior_quat, dtype=np.float64)
        return pos, quat
