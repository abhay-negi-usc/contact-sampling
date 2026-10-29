#!/usr/bin/env python3
"""Sample peg/hole contact poses and wrenches by walking over the strata of the contact space.

Follows Kyaw & Kelly, "Contact Modes Are Strata" (arXiv 2608.15541). Every pair of surfaces that can
touch has a signed distance phi_i(q) from MuJoCo, and the active set A(q) = {i : phi_i(q) = 0} names
the stratum (contact mode) holding q: a manifold of dimension 6 - |A| with tangent space ker J_A.
The peg moves within a stratum by small steps in that tangent space, each followed by a Newton
projection back onto phi_A = 0, so every sample is an exact contact pose, neither in free space nor
interpenetrating. A step makes a contact when it closes another gap (the projection then holds that
one too) and breaks one when the target pulls away from it, but it never breaks the last one.

Coverage: an RRT over contact poses in the error box BOX_LO..BOX_HI of coords() (tip lateral offset,
tilt, insertion depth). Targets are sampled uniformly over the lateral offset and tilt, with depth
near the top of the hole except for a --deep-fraction of them, and each extension walks from the
nearest node. Workers split the lateral offset into cells.

Wrench: at each sample the peg sits at its contact pose, slides along the walk at --speed and is
pressed into each active contact with an equal share of --press-n newtons. One MuJoCo constraint
solve (elliptic friction cones, mu = --friction) then gives the contact wrench: the normal forces
balance the press and sliding friction opposes the slip.

Constraints: MuJoCo reports one distance per geom pair and the bore wall is 64 sectors, so contacts
of one (peg cylinder, hole part) group whose normals agree within --cluster-deg merge into one
constraint held through its closest point. Sliding across a sector seam is then not a make/break
event and a face resting on a face is one constraint, while the two points of a jammed peg (opposed
normals) stay separate constraints.

Output: npz shards in --out-dir (plus meta.json and summary.json). shard_w<worker>_<k>.npz has one
row per sample:
    pose        peg in the hole frame: x, y, z [m], qw, qx, qy, qz
    wrench      contact wrench on the peg in the peg frame, about the peg origin: fx, fy, fz [N], tx, ty, tz [Nm]
    separation  signed distance between peg and hole [m]: < 0 interpenetrating, 0 touching, > 0 apart.
                Every sample is a contact pose, so it stays within --proj-tol-mm of 0.
    traj, step  trajectory id (worker * 10**9 + count) and step index along it. A trajectory is one
                continuous slide: step k + 1 is one walk step (at most --step mm/deg) from step k, and
                step 0 is one walk step from the trajectory's parent sample, given in traj_<worker>_<k>.npz
                (traj, parent_traj, parent_step; -1 for each worker's root). Each worker's
                trajectories therefore form one connected tree.
    time        seconds along the trajectory at --speed, from the parent sample
    twist       walk velocity at the peg tip, hole frame: vx, vy, vz [m/s], wx, wy, wz [rad/s]
    coords      see coords(): tip x, tip z [mm], tilt x, tilt z [deg], depth [mm]
    n_active    number of active contacts; contacts holds point [mm] and normal (hole -> peg) in the hole
                frame of up to MAX_CONTACTS of them, and features their (peg feature, hole feature) ids

    conda run -n mujoco_env python scripts/peg_hole_contact_sampler.py --workers 16 --minutes 30
"""

import argparse
import json
import multiprocessing as mp
import time
from queue import Empty
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import nnls
from scipy.spatial import cKDTree

from peg_hole_model import (BORE_FLOOR_Y, BORE_TOP_Y, COLLISION_GROUP, HOLE_QUAT, HOLE_RINGS, MM, PEG_CYLINDER_NAMES,
                            PEG_CYLINDERS, PEG_TIP_Y, build_model)

DEG = np.pi / 180
HOLE_PARTS = ["plate"] + [f"ring{i}" for i in range(len(HOLE_RINGS))]
R_WH = np.zeros(9)
mujoco.mju_quat2Mat(R_WH, HOLE_QUAT)
R_WH = R_WH.reshape(3, 3)
ASSEMBLED_AXIS = np.array([0.0, -1.0, 0.0])  # peg +y (towards the tip) when assembled, hole frame

# coords(): tip x, tip z [mm], tilt x, tilt z [deg], depth [mm]
COORD_NAMES = ["tip_x_mm", "tip_z_mm", "tilt_x_deg", "tilt_z_deg", "depth_mm"]
BOX_LO = np.array([-10.0, -10.0, -10.0, -10.0, -3.0])
BOX_HI = np.array([10.0, 10.0, 10.0, 10.0, BORE_TOP_Y - BORE_FLOOR_Y + 0.5])
TOP_DEPTH = (-3.0, 3.0)  # depth range of targets that slide over the top of the hole

PEG_FEATURES = ["tip_face", "tip_edge", "tip_side", "shoulder", "other"]
HOLE_FEATURES = ["top", "rim", "wall", "floor", "outer", "step", "other"]
STEP_Y, STEP_R = HOLE_RINGS[1][1], HOLE_RINGS[1][3]  # the step outside the tube and its outer radius

# Sampling regions: (lateral offset range [mm], error box of coords()). "mouth" is the original box;
# "outer" takes the tip out past the tube's outer wall and down to the step around it.
REGIONS = {
    "mouth": dict(rho=(0.0, np.inf), lo=BOX_LO, hi=BOX_HI, start_rho=5.0, deep_fraction=0.1),
    "outer": dict(rho=(8.0, 32.0), lo=np.array([-32.0, -32.0, -10.0, -10.0, -3.0]),
                  hi=np.array([32.0, 32.0, 10.0, 10.0, BORE_TOP_Y - STEP_Y + 1.0]), start_rho=20.0,
                  deep_fraction=0.5),
}
MAX_CONTACTS = 4
MATCH_DEG = 60.0


# --- poses -------------------------------------------------------------------------------------------
# A pose is (t, R): the peg tip centre in the hole frame [mm] and the peg's rotation in the hole frame.
# Steps are 6-vectors (mm, deg): a tip translation and a rotation vector about the tip.

def apply_step(t, R, dx):
    rot = np.zeros(4)
    mujoco.mju_axisAngle2Quat(rot, dx[3:] / max(np.linalg.norm(dx[3:]), 1e-300), np.linalg.norm(dx[3:]) * DEG)
    Rm = np.zeros(9)
    mujoco.mju_quat2Mat(Rm, rot)
    return t + dx[:3], Rm.reshape(3, 3) @ R


def pose_distance(t0, R0, t1, R1):
    """Distance between two poses: tip translation [mm] and rotation [deg] combined."""
    angle = np.arccos(np.clip((np.trace(R1 @ R0.T) - 1) / 2, -1.0, 1.0)) / DEG
    return np.hypot(np.linalg.norm(t1 - t0), angle)


def peg_origin(t, R):
    return t - R[:, 1] * PEG_TIP_Y


def mat2quat(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R.ravel())
    return q


def coords(t, R):
    axis = R[:, 1]
    tilt = np.cross(ASSEMBLED_AXIS, axis)
    s = np.linalg.norm(tilt)
    tilt = tilt / s * np.arctan2(s, axis @ ASSEMBLED_AXIS) / DEG if s > 1e-12 else np.zeros(3)
    return np.array([t[0], t[2], tilt[0], tilt[2], BORE_TOP_Y - t[1]])


def pose_from_coords(c):
    """Pose with the given coords and no spin about the peg axis."""
    R0 = np.diag([1.0, -1.0, -1.0])  # assembled orientation, 180 deg about x
    t = np.array([c[0], BORE_TOP_Y - c[4], c[1]])
    return apply_step(t, R0, np.r_[0, 0, 0, c[2], 0, c[3]])


def coords_step(t, R, target):
    """Direction (mm, deg) from (t, R) towards target coords, without spin."""
    dc = target - coords(t, R)
    return np.array([dc[0], -dc[4], dc[1], dc[2], 0.0, dc[3]])


def set_world_pose(m, d, t, R):
    adr = m.jnt_qposadr[m.joint("peg").id]
    d.qpos[adr : adr + 3] = R_WH @ peg_origin(t, R) * MM
    d.qpos[adr + 3 : adr + 7] = mat2quat(R_WH @ R)


# --- contact space -----------------------------------------------------------------------------------

def ldp(G, h, ridge=1e-9):
    """Least-distance program min ||x|| s.t. G x >= h, via NNLS (Lawson & Hanson); None if infeasible.

    A small ridge keeps the NNLS subproblems nonsingular when many contact rows are alike.
    """
    m, n = G.shape
    E = np.vstack([G.T, h[None, :], np.sqrt(ridge) * np.eye(m)])
    f = np.zeros(n + 1 + m)
    f[n] = 1.0
    u, _ = nnls(E, f)
    r = E[: n + 1] @ u - f[: n + 1]
    if abs(r[n]) < 1e-12:
        return None
    return -r[:n] / r[n]


class Contacts:
    """Contacts at one pose, raw (one per geom pair) and clustered into constraints.

    A cluster is represented by its closest point; support counts its points within the active
    tolerance of that one: 1 for a point contact, more for a line or face resting on a surface.
    """

    def __init__(self, raw_phi, raw_grad, phi, point, normal, grad, group, peg_point, support):
        self.raw_phi, self.raw_grad = raw_phi, raw_grad
        self.phi, self.point, self.normal, self.grad, self.group, self.peg_point, self.support = (
            phi, point, normal, grad, group, peg_point, support)

    def __len__(self):
        return len(self.phi)

    def anchors(self, idx):
        return [(self.group[i], self.normal[i].copy()) for i in idx]

    def rows(self, held):
        """Linear constraints G dx >= h: no contact point penetrates, and the held clusters stay closed."""
        G = np.vstack([self.raw_grad, -self.grad[held]])
        return G, np.r_[-self.raw_phi, self.phi[held]]


class ContactSpace:
    """Signed distances phi [mm] and gradients (mm/mm, mm/deg) of the peg/hole contacts at a pose."""

    def __init__(self, margin_mm, cluster_deg, proj_tol_mm, active_tol_mm):
        self.m = build_model(margin=margin_mm * MM)
        self.m.opt.enableflags &= ~int(mujoco.mjtEnableBit.mjENBL_MULTICCD)  # one point per geom pair is enough here
        self.m.opt.ccd_tolerance = 1e-10
        self.d = mujoco.MjData(self.m)
        self.cos_cluster = np.cos(cluster_deg * DEG)
        self.proj_tol, self.active_tol = proj_tol_mm, active_tol_mm
        self.peg_part = np.full(self.m.ngeom, -1)
        self.hole_part = np.full(self.m.ngeom, -1)
        for i in range(self.m.ngeom):
            name = self.m.geom(i).name
            if self.m.geom_group[i] != COLLISION_GROUP:
                continue
            if name.startswith("peg_"):
                self.peg_part[i] = PEG_CYLINDER_NAMES.index(name[4:])
            elif name.startswith("hole_"):
                self.hole_part[i] = HOLE_PARTS.index(name.split("_")[1])
        self.evals = 0

    def evaluate(self, t, R):
        self.evals += 1
        m, d = self.m, self.d
        set_world_pose(m, d, t, R)
        mujoco.mj_kinematics(m, d)
        mujoco.mj_collision(m, d)
        n = d.ncon
        if n == 0:
            e, g = np.zeros((0, 3)), np.zeros((0, 6))
            return Contacts(np.zeros(0), g, np.zeros(0), e, e, g, np.zeros(0, int), e, np.zeros(0, int))
        c = d.contact
        g1, g2 = c.geom1[:n], c.geom2[:n]
        peg_first = self.peg_part[g1] >= 0
        phi = c.dist[:n] / MM
        point = c.pos[:n] @ R_WH / MM
        normal = c.frame[:n, :3] @ R_WH
        normal[peg_first] *= -1  # MuJoCo's normal points from geom1 to geom2; make it point hole -> peg
        grad = np.hstack([normal, np.cross(point - t, normal) * DEG])
        group = (np.where(peg_first, self.peg_part[g1], self.peg_part[g2]) * len(HOLE_PARTS)
                 + np.where(peg_first, self.hole_part[g2], self.hole_part[g1]))
        # Merge contacts of one (peg part, hole part) group whose normals agree: they are one surface
        # touching another (a face or line contact spans several sectors), held through its closest point.
        keep, support = [], []
        for i in np.argsort(phi):
            for k, j in enumerate(keep):
                if group[j] == group[i] and normal[j] @ normal[i] > self.cos_cluster:
                    support[k] += phi[i] < phi[j] + self.active_tol
                    break
            else:
                keep.append(i)
                support.append(1)
        keep = np.array(keep)
        peg_point = (point[keep] + 0.5 * phi[keep, None] * normal[keep] - t) @ R  # on the peg, peg frame about the tip
        return Contacts(phi, grad, phi[keep], point[keep], normal[keep], grad[keep], group[keep], peg_point,
                        np.array(support))

    def match(self, C, anchors):
        """Indices of the clusters continuing the anchored constraints [(group, normal)], or None.

        A contact's normal turns quickly as it rounds an edge, so the match only rejects normals more
        than MATCH_DEG apart; opposed contacts, as in a jammed peg, still stay distinct. Two anchored
        contacts whose normals have come together continue as one merged constraint, which holds the
        closer of them and lets the other lift off. None if no anchor continues.
        """
        idx = []
        for group, normal in anchors:
            cand = [i for i in range(len(C)) if C.group[i] == group]
            if not cand:
                continue
            best = max(cand, key=lambda i: C.normal[i] @ normal)
            if C.normal[best] @ normal >= np.cos(MATCH_DEG * DEG) and best not in idx:
                idx.append(best)
        return idx or None

    def settle(self, t, R, anchors, max_iter=10):
        """Project onto the contact space: the anchored contacts closed, no point penetrating.

        Each iteration moves to the nearest pose satisfying the linearised constraints. If the
        anchored contacts are gone (the peg slid off an edge), the nearest contact is made instead.
        Returns (t, R, contacts, active clusters) or None.
        """
        last_err = np.inf
        for _ in range(max_iter):
            C = self.evaluate(t, R)
            if not len(C):
                return None
            held = self.match(C, anchors) or [int(np.argmin(C.phi))]
            anchors = C.anchors(held)
            err = max(np.abs(C.phi[held]).max(), -C.raw_phi.min())
            if err < self.proj_tol:
                return t, R, C, [i for i in range(len(C)) if C.phi[i] < self.active_tol]
            if err > last_err:  # diverging, e.g. an anchor has latched onto another surface
                return None
            last_err = err
            dx = ldp(*C.rows(held))
            if dx is None:
                return None
            t, R = apply_step(t, R, dx)
        return None

    def drop(self, t, R, step_mm, max_depth, near_mm=0.02):
        """Lower the peg along the bore axis until it touches, then project onto that contact.

        Each move is at most the gap to the hole, so it cannot pass through a surface, and a peg
        inside the bore falls to the floor.
        """
        while BORE_TOP_Y - t[1] < max_depth:
            C = self.evaluate(t, R)
            gap = C.phi.min() if len(C) else np.inf
            if gap < near_mm:
                return self.settle(t, R, C.anchors([int(np.argmin(C.phi))]))
            t = t + np.array([0.0, -min(step_mm, max(gap, near_mm / 2)), 0.0])
        return None

    def step(self, t, R, C, active, direction, eps):
        """One walk step of up to eps from a pose on the contact space: (t, R, contacts, active, step) or None.

        Contacts the direction pushes into stay closed and the rest may open, but at least one stays
        closed. The step follows the move closest to the direction that keeps those contacts closed
        and penetrates nothing (to first order), scaled to eps; a projection then corrects for
        curvature and closes any gap the step overran. Around sharp edges the step is halved until
        the projection succeeds and lands within twice the step of the start, so consecutive poses
        of a walk are never further apart than that.
        """
        push = C.grad[active] @ direction
        hold = [i for i, p in zip(active, push) if p <= 0]
        if not hold:  # the target pulls away from every contact: keep the one it pulls least
            hold = [active[int(np.argmin(push / np.linalg.norm(C.grad[active], axis=1)))]]
        want = direction * (eps / np.linalg.norm(direction))
        G, h = C.rows(hold)
        y = ldp(G, h - G @ want)
        if y is None:
            return None
        delta = want + y
        norm = np.linalg.norm(delta)
        if norm < 1e-3 * eps:  # blocked
            return None
        delta *= eps / norm
        for _ in range(4):
            res = self.settle(*apply_step(t, R, delta), C.anchors(hold))
            if res is not None and pose_distance(t, R, res[0], res[1]) <= 2 * np.linalg.norm(delta):
                return (*res, delta)
            delta = delta / 2
        return None


def features(C, active):
    """(peg feature, hole feature) ids of the active contacts."""
    out = []
    eps = 0.05  # mm
    r_tip = PEG_CYLINDERS[-1][2]
    for i in active:
        p = C.peg_point[i]  # peg frame about the tip: +y runs back up the peg
        r_p = np.hypot(p[0], p[2])
        if p[1] > -eps:
            pf = "tip_edge" if r_p > r_tip - eps else "tip_face"
        elif p[1] > -(PEG_TIP_Y - PEG_CYLINDERS[-1][0]) + eps:
            pf = "tip_side" if r_p > r_tip - eps else "other"
        else:
            pf = "shoulder" if abs(p[1] + PEG_TIP_Y - PEG_CYLINDERS[-1][0]) < eps else "other"
        q = C.point[i]
        r_h = np.hypot(q[0], q[2])
        r_bore = HOLE_RINGS[-1][2]
        at_top, at_wall = abs(q[1] - BORE_TOP_Y) < eps, abs(r_h - r_bore) < eps
        if at_top and at_wall:
            hf = "rim"
        elif at_top:
            hf = "top"
        elif at_wall and BORE_FLOOR_Y < q[1] < BORE_TOP_Y:
            hf = "wall"
        elif abs(q[1] - BORE_FLOOR_Y) < eps and r_h <= r_bore + eps:
            hf = "floor"
        elif abs(r_h - HOLE_RINGS[-1][3]) < eps and STEP_Y - eps < q[1] < BORE_TOP_Y:
            hf = "outer"  # the tube's outer wall
        elif abs(q[1] - STEP_Y) < eps and HOLE_RINGS[-1][3] - eps <= r_h <= STEP_R + eps:
            hf = "step"
        else:
            hf = "other"
        out.append((PEG_FEATURES.index(pf), HOLE_FEATURES.index(hf)))
    return out


# --- wrench ------------------------------------------------------------------------------------------

class WrenchSolver:
    """Contact wrench at a contact pose for a given press and walk velocity.

    One MuJoCo constraint solve with the peg at rest spreads the press over the contact points
    (multi-point contact patches for faces and edges). Coulomb sliding friction, mu times each
    point's normal force against its slip velocity under the walk, is then added point by point.
    """

    def __init__(self, friction, press_n, speed_mm_s, margin_mm):
        m = self.m = build_model(margin=margin_mm * MM)
        m.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
        col = m.geom_group == COLLISION_GROUP
        # Near-rigid contacts with no penetration spring, so the normal forces just balance the press.
        m.geom_solref[col] = [-1e-6, -1.0]
        m.geom_solimp[col, :3] = [0.995, 0.999, 0.001]
        self.peg = m.body("peg").id
        self.peg_geom = m.geom_bodyid == self.peg
        self.dof = m.jnt_dofadr[m.joint("peg").id]
        self.d = mujoco.MjData(m)
        self.mu, self.press, self.speed = friction, press_n, speed_mm_s

    def solve(self, t, R, C, active, step):
        m, d = self.m, self.d
        set_world_pose(m, d, t, R)
        d.qvel[:] = 0
        # Press each active contact along its normal with an equal share of the load.
        f = -self.press / len(active) * C.normal[active]  # hole frame, at the contact points
        F = f.sum(0)
        torque_tip = np.cross((C.point[active] - t) * MM, f).sum(0)
        mujoco.mj_kinematics(m, d)
        com = R_WH.T @ d.xipos[self.peg]
        d.xfrc_applied[self.peg, :3] = R_WH @ F
        d.xfrc_applied[self.peg, 3:] = R_WH @ (torque_tip + np.cross(t * MM - com, F))
        mujoco.mj_forward(m, d)

        n = d.ncon
        c = d.contact
        live = c.efc_address[:n] >= 0
        lam = np.where(live, d.efc_force[np.maximum(c.efc_address[:n], 0)], 0.0)
        normal = c.frame[:n, :3] @ R_WH
        normal[self.peg_geom[c.geom1[:n]]] *= -1  # hole -> peg
        point = c.pos[:n] @ R_WH  # m, hole frame
        u = step / max(np.linalg.norm(step), 1e-300)  # a zero step is a peg at rest: no sliding friction
        v_tip, w = u[:3] * self.speed * MM, u[3:] * self.speed * DEG  # hole frame, m/s and rad/s
        slip = v_tip + np.cross(w, point - t * MM)
        slip -= (slip * normal).sum(1, keepdims=True) * normal
        speed = np.linalg.norm(slip, axis=1, keepdims=True)
        sliding = speed > 1e-3 * self.speed * MM
        force = lam[:, None] * (normal - self.mu * np.where(sliding, slip / np.maximum(speed, 1e-300), 0.0))
        origin = peg_origin(t, R) * MM
        F = force.sum(0)
        torque = np.cross(point - origin, force).sum(0)
        return np.r_[R.T @ F, R.T @ torque], np.r_[v_tip, w], int(live.sum())


class AnalyticContactSpace(ContactSpace):
    """The same contact space on the exact geometry of peg_hole_analytic.py instead of MuJoCo's: a
    smooth bore and smooth rims (MuJoCo's are 64-gons, which put a crease in the contact manifold at
    every facet seam). Contacts are grouped per contact circle (tip edge, shoulder edge, inner rim,
    outer rim) and merged by normal as in ContactSpace."""

    def __init__(self, margin_mm, cluster_deg, proj_tol_mm, active_tol_mm):
        import peg_hole_analytic as A
        self.A = A
        self.margin = margin_mm
        self.cos_cluster = np.cos(cluster_deg * DEG)
        self.proj_tol, self.active_tol = proj_tol_mm, active_tol_mm
        self.evals = 0

    def evaluate(self, t, R):
        self.evals += 1
        u = -R[:, 1]  # up the peg
        rows = self.A.contact_details(t, u, self.margin)
        if len(rows) == 0:
            e, g = np.zeros((0, 3)), np.zeros((0, 6))
            return Contacts(np.zeros(0), g, np.zeros(0), e, e, g, np.zeros(0, int), e, np.zeros(0, int))
        group, phi, point, normal = rows[:, 0].astype(int), rows[:, 1], rows[:, 2:5], rows[:, 5:8]
        grad = np.hstack([normal, np.cross(point - t, normal) * DEG])
        keep, support = [], []
        for i in np.argsort(phi):
            for k, j in enumerate(keep):
                if group[j] == group[i] and normal[j] @ normal[i] > self.cos_cluster:
                    support[k] += phi[i] < phi[j] + self.active_tol
                    break
            else:
                keep.append(i)
                support.append(1)
        keep = np.array(keep)
        # contact point on the peg: a peg circle's point is on the peg; a rim's point is on the hole
        peg_pt = np.where((group[keep] < 2)[:, None], point[keep], point[keep] + phi[keep, None] * normal[keep])
        peg_point = (peg_pt - t) @ R
        return Contacts(phi, grad, phi[keep], point[keep], normal[keep], grad[keep], group[keep], peg_point,
                        np.array(support))


class AnalyticWrench:
    """Contact wrench from the active contacts themselves: an equal share of the press along each
    normal, and Coulomb sliding friction against each contact point's slip under the walk."""

    def __init__(self, friction, press_n, speed_mm_s):
        self.mu, self.press, self.speed = friction, press_n, speed_mm_s

    def solve(self, t, R, C, active, step):
        n = C.normal[active]
        point = C.point[active] * MM
        lam = self.press / len(active)
        u = step / max(np.linalg.norm(step), 1e-300)
        v_tip, w = u[:3] * self.speed * MM, u[3:] * self.speed * DEG
        slip = v_tip + np.cross(w, point - t * MM)
        slip -= (slip * n).sum(1, keepdims=True) * n
        speed = np.linalg.norm(slip, axis=1, keepdims=True)
        sliding = speed > 1e-3 * self.speed * MM
        force = lam * (n - self.mu * np.where(sliding, slip / np.maximum(speed, 1e-300), 0.0))
        origin = peg_origin(t, R) * MM
        F, torque = force.sum(0), np.cross(point - origin, force).sum(0)
        return np.r_[R.T @ F, R.T @ torque], np.r_[v_tip, w], len(active)


# --- sampling ----------------------------------------------------------------------------------------

# Contact modes (strata) that only occur with the tip in the bore are thin: the bore's clearance is
# 0.5 mm and its tilt range about a degree. Their nodes use finer bins and their walks shorter steps.
FINE_PEG_FEATURES = {"tip_side", "shoulder"}
FINE_HOLE_FEATURES = {"wall", "floor", "outer"}  # vertical walls: the peg slides along them in depth


def stratum_of(feats):
    """Contact mode of a pose: its sorted (peg feature, hole feature) pair ids."""
    return tuple(sorted(int(p) * len(HOLE_FEATURES) + int(h) for p, h in feats))


def is_fine(stratum):
    return any(PEG_FEATURES[k // len(HOLE_FEATURES)] in FINE_PEG_FEATURES
               or HOLE_FEATURES[k % len(HOLE_FEATURES)] in FINE_HOLE_FEATURES for k in stratum)


def bin_key(c, stratum, res, fine_res):
    """Node key of coords c in a contact mode: the mode and its bin. Coarse modes bin lateral offset
    and tilt at res (their depth follows from those); fine modes bin lateral offset, tilt and depth at
    fine_res = (lateral, tilt, depth)."""
    if is_fine(stratum):
        size = np.r_[fine_res[0], fine_res[0], fine_res[1], fine_res[1], fine_res[2]]
        return stratum, tuple(np.floor(c / size).astype(np.int64).tolist())
    return stratum, tuple(np.floor(c[:4] / res).astype(np.int64).tolist())


SAMPLE_DTYPES = dict(pose=np.float32, wrench=np.float32, separation=np.float32, traj=np.int64, step=np.int32,
                     time=np.float32, twist=np.float32, coords=np.float32, n_active=np.int32, contacts=np.float32,
                     features=np.int32, ncon=np.int32)
TRAJ_DTYPES = dict(traj=np.int64, parent_traj=np.int64, parent_step=np.int32)


class Shard:
    """Buffered writer of one worker's sample rows (shard_*.npz) and trajectory rows (traj_*.npz)."""

    def __init__(self, out_dir, worker, flush_every):
        self.out_dir, self.worker, self.flush_every, self.k = out_dir, worker, flush_every, 0
        self.rows = {name: [] for name in SAMPLE_DTYPES}
        self.trajs = {name: [] for name in TRAJ_DTYPES}

    def add_traj(self, **row):
        for k, v in row.items():
            self.trajs[k].append(v)

    def add(self, **row):
        for k, v in row.items():
            self.rows[k].append(v)
        if len(self.rows["pose"]) >= self.flush_every:
            self.flush()

    def flush(self):
        if not self.rows["pose"]:
            return
        name = f"w{self.worker:02d}_{self.k:04d}.npz"
        np.savez(self.out_dir / f"shard_{name}", **{k: np.asarray(v, SAMPLE_DTYPES[k]) for k, v in self.rows.items()})
        np.savez(self.out_dir / f"traj_{name}", **{k: np.asarray(v, TRAJ_DTYPES[k]) for k, v in self.trajs.items()})
        self.k += 1
        for v in [*self.rows.values(), *self.trajs.values()]:
            v.clear()


def worker(args, wid, sector, result_queue):
    rng = np.random.default_rng(args.seed + wid)
    space = (AnalyticContactSpace if args.geometry == "analytic" else ContactSpace)(
        args.margin_mm, args.cluster_deg, args.proj_tol_mm, args.active_tol_mm)
    # The solve's margin takes in every contact the walk calls active.
    wrench = (AnalyticWrench(args.friction, args.press_n, args.speed) if args.geometry == "analytic"
              else WrenchSolver(args.friction, args.press_n, args.speed, args.active_tol_mm * 1.25))
    shard = Shard(args.out_dir, wid, args.flush_every)

    region = REGIONS[args.region]
    box_lo, box_hi, (rho_lo, rho_hi) = region["lo"], region["hi"], region["rho"]

    def in_region(c):
        return np.all(c >= box_lo - 1e-9) and np.all(c <= box_hi + 1e-9) and rho_lo <= np.hypot(c[0], c[1]) <= rho_hi

    def sector_target():
        """A target in the region with its lateral offset in this worker's azimuth sector."""
        while True:
            target = rng.uniform(box_lo, box_hi)
            if np.isfinite(rho_hi):  # lateral offset uniform over the annulus
                r = np.sqrt(rng.uniform(rho_lo ** 2, rho_hi ** 2))
                a = rng.uniform(*sector)
                target[:2] = r * np.cos(a), r * np.sin(a)
            angle = np.arctan2(target[1], target[0]) % (2 * np.pi)
            if sector[0] <= angle < sector[1] and in_region(target):
                break
        if rng.random() > args.deep_fraction:
            target[4] = rng.uniform(*TOP_DEPTH)
        return target

    mid = (sector[0] + sector[1]) / 2
    r0 = region["start_rho"]
    start = pose_from_coords(np.r_[r0 * np.cos(mid), r0 * np.sin(mid), 0.0, 0.0, box_lo[4] - 10])
    res = space.drop(*start, step_mm=0.5 * args.margin_mm, max_depth=box_hi[4])
    if res is None:
        result_queue.put((wid, None))
        return
    t, R, C, active = res
    stats = dict(samples=0, trajectories=0, extensions=0, local_extensions=0, steps_failed=0, makes=0, breaks=0,
                 outside=0)

    def record(t, R, C, active, feats, delta, traj, step, time_s):
        w, twist, ncon = wrench.solve(t, R, C, active, delta)
        cont = np.zeros((MAX_CONTACTS, 6), np.float32)
        fe = np.full((MAX_CONTACTS, 2), -1, np.int32)
        for k, i in enumerate(active[:MAX_CONTACTS]):
            cont[k] = np.r_[C.point[i], C.normal[i]]
            fe[k] = feats[k]
        shard.add(pose=np.r_[peg_origin(t, R) * MM, mat2quat(R)], wrench=w, separation=C.raw_phi.min() * MM,
                  traj=traj, step=step, time=time_s, twist=twist, coords=coords(t, R), n_active=len(active),
                  contacts=cont, features=fe, ncon=ncon)
        stats["samples"] += 1

    # Nodes: one per visited (contact mode, bin), with its coords, its contact pose and the
    # (trajectory, step) of that sample. nodes_in lists each mode's nodes.
    nodes_c = np.zeros((1024, len(box_lo)))
    nodes_s, nodes_src, node_of_key, nodes_in = [], [], {}, {}

    def add_node(c1, t, R, feats, traj, step):
        nonlocal nodes_c
        stratum = stratum_of(feats)
        key = bin_key(c1, stratum, args.resolution, args.fine_resolution)
        if key in node_of_key:
            return
        if len(nodes_s) == len(nodes_c):
            nodes_c = np.concatenate([nodes_c, np.zeros_like(nodes_c)])
        node_of_key[key] = len(nodes_s)
        nodes_in.setdefault(stratum, []).append(len(nodes_s))
        nodes_c[len(nodes_s)] = c1
        nodes_s.append((t, R))
        nodes_src.append((traj, step))

    # The starting contact is the root trajectory, a single sample at rest.
    next_traj = wid * 10**9
    feats = features(C, active)
    record(t, R, C, active, feats, np.zeros(6), next_traj, 0, 0.0)
    shard.add_traj(traj=next_traj, parent_traj=-1, parent_step=-1)
    stats["trajectories"] += 1
    add_node(coords(t, R), t, R, feats, next_traj, 0)
    next_traj += 1
    tree, n_tree = None, 0
    t_start = time.time()
    t_end = t_start + args.minutes * 60
    history, t_log = [], t_start
    while time.time() < t_end and stats["samples"] < args.samples:
        if time.time() >= t_log:
            history.append((time.time() - t_start, stats["samples"], len(node_of_key)))
            t_log += args.log_every
        if rng.random() < args.local_fraction:
            # Balance effort over contact modes: a random mode, a random node in it, a target nearby.
            strata = list(nodes_in)
            stratum = strata[rng.integers(len(strata))]
            best = nodes_in[stratum][rng.integers(len(nodes_in[stratum]))]
            spread = (np.r_[args.fine_resolution[0], args.fine_resolution[0], args.fine_resolution[1],
                            args.fine_resolution[1], args.fine_resolution[2]] if is_fine(stratum)
                      else np.r_[args.resolution, args.resolution, args.resolution, args.resolution, 1.0])
            target = np.clip(nodes_c[best] + rng.normal(0.0, 4 * spread), box_lo, box_hi)
            best_d = np.linalg.norm(target - nodes_c[best])
            stats["local_extensions"] += 1
        else:
            # RRT: a target in the sector, extended from the nearest node. A k-d tree covers most
            # nodes and is rebuilt now and then; the newest nodes are scanned.
            target = sector_target()
            n_nodes = len(nodes_s)
            if n_nodes - n_tree > min(max(256, n_tree // 4), 4096):
                tree, n_tree = cKDTree(nodes_c[:n_nodes]), n_nodes
            best, best_d = None, np.inf
            if tree is not None:
                best_d, best = tree.query(target)
            if n_tree < n_nodes:
                dd = np.linalg.norm(nodes_c[n_tree:n_nodes] - target, axis=1)
                j = int(np.argmin(dd))
                if dd[j] < best_d:
                    best, best_d = n_tree + j, dd[j]
        # A node is a sample's contact pose, so the trajectory restarts exactly there.
        t, R = nodes_s[best]
        C = space.evaluate(t, R)
        active = [i for i in range(len(C)) if C.phi[i] < space.active_tol]
        if not active:
            stats["steps_failed"] += 1
            continue
        stratum = stratum_of(features(C, active))
        stats["extensions"] += 1
        traj, step, time_s = None, 0, 0.0  # a trajectory gets its id with its first sample
        last_d, path = best_d, 0.0
        while path < args.max_steps * args.step:  # the same path length whatever the step
            step_len = args.fine_step if is_fine(stratum) else args.step
            direction = coords_step(t, R, target)
            if np.linalg.norm(direction) < step_len:
                break
            res = space.step(t, R, C, active, direction, step_len)
            if res is None:
                stats["steps_failed"] += 1
                break
            t1, R1, C1, active1, delta = res
            c1 = coords(t1, R1)
            if not in_region(c1):
                stats["outside"] += 1
                break
            stats["makes"] += max(0, len(active1) - len(active))
            stats["breaks"] += max(0, len(active) - len(active1))
            t, R, C, active = t1, R1, C1, active1
            if traj is None:
                traj = next_traj
                next_traj += 1
                parent_traj, parent_step = nodes_src[best]
                shard.add_traj(traj=traj, parent_traj=parent_traj, parent_step=parent_step)
                stats["trajectories"] += 1
            else:
                step += 1
            time_s += np.linalg.norm(delta) / args.speed
            path += np.linalg.norm(delta)
            feats = features(C, active)
            stratum = stratum_of(feats)
            record(t, R, C, active, feats, delta, traj, step, time_s)
            add_node(c1, t, R, feats, traj, step)
            dist = np.linalg.norm(target - c1)
            if dist > last_d - 0.1 * step_len:
                break
            last_d = dist
    shard.flush()
    history.append((time.time() - t_start, stats["samples"], len(node_of_key)))
    stats["evals"] = space.evals
    stats["nodes"] = len(nodes_s)
    per_stratum = {stratum: len(v) for stratum, v in nodes_in.items()}
    result_queue.put((wid, (stats, list(node_of_key), history, per_stratum)))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path,
                        default=Path("/media/rp/Elements1/abhay_ws/contact-sampling/peg_hole_contact_samples"))
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--minutes", type=float, default=10.0, help="wall-clock budget per worker")
    parser.add_argument("--samples", type=int, default=10**9, help="sample budget per worker")
    parser.add_argument("--resolution", type=float, default=0.5,
                        help="node bin size of contact modes on top of the hole [mm, deg]")
    parser.add_argument("--fine-resolution", type=float, nargs=3, default=[0.1, 0.25, 1.0],
                        help="node bin size of contact modes inside the bore: lateral [mm], tilt [deg], depth [mm]")
    parser.add_argument("--step", type=float, default=0.25, help="walk step [mm, deg]; one sample per step")
    parser.add_argument("--fine-step", type=float, default=0.05, help="walk step in contact modes inside the bore")
    parser.add_argument("--local-fraction", type=float, default=0.5,
                        help="fraction of extensions that start from a random node of a random contact mode")
    parser.add_argument("--max-steps", type=int, default=40,
                        help="path length of an RRT extension, in steps of --step")
    parser.add_argument("--deep-fraction", type=float, default=None,
                        help="fraction of targets with depth anywhere in the box rather than near the top")
    parser.add_argument("--press-n", type=float, default=10.0, help="total press into the active contacts [N]")
    parser.add_argument("--friction", type=float, default=0.3)
    parser.add_argument("--speed", type=float, default=10.0, help="walk speed [mm/s and deg/s] for friction")
    parser.add_argument("--region", choices=list(REGIONS), default="mouth",
                        help="mouth: tip within 10 mm of the bore axis; outer: lateral offset 8-32 mm, out to "
                             "the tube's outer wall and down to the step around it")
    parser.add_argument("--geometry", choices=["mujoco", "analytic"], default="mujoco",
                        help="contact geometry: MuJoCo's (64-gon bore and rims) or the exact analytical one")
    parser.add_argument("--margin-mm", type=float, default=0.4, help="collision margin for seeing nearby gaps")
    parser.add_argument("--cluster-deg", type=float, default=20.0,
                        help="contacts on one part pair whose normals agree within this merge into one constraint")
    parser.add_argument("--proj-tol-mm", type=float, default=5e-4, help="projection tolerance on phi")
    parser.add_argument("--active-tol-mm", type=float, default=2e-3, help="gap below which a contact is active")
    parser.add_argument("--flush-every", type=int, default=200_000, help="samples per shard file")
    parser.add_argument("--log-every", type=float, default=15.0, help="seconds between coverage log entries")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.deep_fraction is None:
        args.deep_fraction = REGIONS[args.region]["deep_fraction"]

    args.out_dir = args.out_dir / time.strftime("run_%Y%m%d_%H%M%S")
    args.out_dir.mkdir(parents=True)
    # Split the lateral offset into azimuth sectors, one per worker: by symmetry they are alike, and
    # each reaches the bore at its apex.
    meta = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    region = REGIONS[args.region]
    meta.update(region=args.region, rho_range=list(region["rho"]), coord_names=COORD_NAMES,
                box_lo=region["lo"].tolist(), box_hi=region["hi"].tolist(),
                peg_features=PEG_FEATURES, hole_features=HOLE_FEATURES, paper="arXiv 2608.15541",
                conventions=__doc__)
    (args.out_dir / "meta.json").write_text(json.dumps(meta, indent=1))

    queue = mp.Queue()
    sectors = [(2 * np.pi * w / args.workers, 2 * np.pi * (w + 1) / args.workers) for w in range(args.workers)]
    procs = [mp.Process(target=worker, args=(args, w, sector, queue)) for w, sector in enumerate(sectors)]
    t0 = time.time()
    for p in procs:
        p.start()
    results = {}
    while len(results) < len(procs):
        try:
            wid, res = queue.get(timeout=10)
            results[wid] = res
        except Empty:
            crashed = [w for w, p in enumerate(procs) if p.exitcode not in (None, 0) and w not in results]
            if crashed:
                for p in procs:
                    p.terminate()
                raise SystemExit(f"workers {crashed} crashed")
    for p in procs:
        p.join()
    wall = time.time() - t0

    total = {}
    visited = set()
    history = {}  # log time -> [samples, nodes] summed over workers
    nodes_per_stratum = {}
    for wid in sorted(results):
        if results[wid] is None:
            print(f"worker {wid}: could not find a starting contact")
            continue
        stats, keys, hist, per_stratum = results[wid]
        visited.update(keys)
        for stratum, n in per_stratum.items():
            nodes_per_stratum[stratum] = nodes_per_stratum.get(stratum, 0) + n
        for k, v in stats.items():
            total[k] = total.get(k, 0) + v
        for i, (_, samples, bins) in enumerate(hist[:-1]):
            history.setdefault(i * args.log_every, [0, 0])
            history[i * args.log_every][0] += samples
            history[i * args.log_every][1] += bins
    coarse_bins = {b for stratum, b in visited if not is_fine(stratum)}
    fine_nodes = sum(1 for stratum, _ in visited if is_fine(stratum))
    n_bins = (int(round((BOX_HI[0] - BOX_LO[0]) / args.resolution)) + 1) ** 4
    print(f"{total.get('samples', 0)} samples in {wall:.0f} s ({total.get('samples', 0) / wall:.0f} samples/s) "
          f"-> {args.out_dir}")
    print("  " + ", ".join(f"{k} {v}" for k, v in total.items()))
    print(f"  coverage: {len(coarse_bins)} / {n_bins} lateral-tilt bins at {args.resolution} mm/deg on top of the hole "
          f"({len(coarse_bins) / n_bins:.2%}), {fine_nodes} fine bins of contact modes inside the bore")
    names = lambda stratum: " + ".join(f"{PEG_FEATURES[k // len(HOLE_FEATURES)]}/{HOLE_FEATURES[k % len(HOLE_FEATURES)]}"
                                       for k in stratum)
    print("  nodes per contact mode:")
    for stratum, n in sorted(nodes_per_stratum.items(), key=lambda kv: -kv[1])[:25]:
        print(f"    {n:9d}  {'fine  ' if is_fine(stratum) else 'coarse'}  {names(stratum)}")
    (args.out_dir / "summary.json").write_text(json.dumps(
        dict(wall_s=wall, lateral_tilt_bins=len(coarse_bins), lateral_tilt_bins_total=n_bins, fine_bins=fine_nodes,
             nodes_per_mode={names(k): v for k, v in nodes_per_stratum.items()},
             history_s_samples_nodes=sorted(history.items()), **total), indent=1))

if __name__ == "__main__":
    main()
