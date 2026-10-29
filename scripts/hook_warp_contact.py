#!/usr/bin/env python3
"""Batched hook-eyelet contact geometry on the GPU (NVIDIA Warp), from precomputed geometry, for contact sampling.

The parts are fixed, so the geometry is solved once: the eyelet's exact signed distance (BVH closest point,
winding-number sign; the angle-weighted pseudo-normal sign misfires on these STLs) on a fine grid (h) within
`margin` of the eyelet and a coarse one (0.5 mm) out to `far_margin`, with the eyelet mesh kept for exact closest
points and normals; the hook as dense surface samples grouped into 5 mm cells and 1 mm sub-cells; the hook's
centreline and the loop's hole for the threading test (hook_topology.HookLoop, as the CPU pipeline). The hook
samples and the threading data are cached in --cache; the grids are rebuilt on load (seconds).

Separation is the smallest signed distance of a hook surface point to the eyelet. Per pose, a cell is skipped when
its centre's distance minus its radius cannot beat the best so far (the distance is 1-Lipschitz), so only the cells
near contact are read point by point. Against brute-force mesh distances it is exact to ~0.01 mm within `cap`
(margin - 1 mm) of contact; beyond, it is estimated from the 1 mm sub-cells (within ~1 mm, enough to slide in from
far away); inside the eyelet it saturates at the wire radius (~2.5 mm), so for penetration it gives the sign, not
the depth. Its gradient with respect to the hook's translation is the contact normal n (eyelet pushing the hook),
with respect to its rotation r x n: the unit wrench [n, r x n].

Poses are (x, y, z mm, rotation vector deg) of the hook in the eyelet frame, as everywhere else.

    geo = WarpContact()
    sep = geo.separation(poses)                       # (n,) mm
    c = geo.contacts(poses)                           # up to MAXC distinct contacts: gaps, points, normals
    bits = geo.threading(poses)                       # hook_topology bitmask (TIP | CROOK | SHAFT)
    q, ok = geo.project(poses, target, tol)           # slide along the contact normal to the target separation
    stray, jump = geo.slide(a, b, step, target, reach)
    clear = geo.segment_min(a, b, step, floor)

    python scripts/hook_warp_contact.py validate EXPERIMENT_DIR   # against brute force and the CPU pipeline
"""

import argparse
import time
from pathlib import Path

import numpy as np
import trimesh
import warp as wp

from hook_eyelet_model import ASSET_DIR, EYELET_YAW_DEG, HOOK_YAW_DEG, INCH

CACHE = Path("/media/rp/Elements1/abhay_ws/contact-sampling/warp_cache")
MAXC = 4  # distinct contacts kept per pose, as hook_contact_strata
vec4i = wp.types.vector(length=4, dtype=wp.int32)


@wp.struct
class Geo:
    grid: wp.array3d(dtype=wp.float32)
    origin: wp.vec3
    inv_h: float
    grid2: wp.array3d(dtype=wp.float32)  # coarse, wider
    origin2: wp.vec3
    inv_h2: float
    far: float  # what the coarse grid returns outside itself: a lower bound of the true distance there
    mesh: wp.uint64
    c1: wp.array(dtype=wp.vec3)
    r1: wp.array(dtype=float)
    s1: wp.array(dtype=int)
    e1: wp.array(dtype=int)
    c2: wp.array(dtype=wp.vec3)
    r2: wp.array(dtype=float)
    s2: wp.array(dtype=int)
    e2: wp.array(dtype=int)
    pts: wp.array(dtype=wp.vec3)


@wp.kernel
def _grid_kernel(mesh: wp.uint64, origin: wp.vec3, h: float, out: wp.array3d(dtype=wp.float32)):
    i, j, k = wp.tid()
    p = origin + wp.vec3(float(i) * h, float(j) * h, float(k) * h)
    q = wp.mesh_query_point_sign_winding_number(mesh, p, 1.0e6)
    if q.result:
        out[i, j, k] = q.sign * wp.length(p - wp.mesh_eval_position(mesh, q.face, q.u, q.v))


@wp.func
def trilinear(grid: wp.array3d(dtype=wp.float32), x: wp.vec3, outside: float):
    """Trilinear read of grid at grid coordinates x; outside where x leaves the grid."""
    if x[0] < 0.0 or x[1] < 0.0 or x[2] < 0.0:
        return outside
    i = int(x[0])
    j = int(x[1])
    k = int(x[2])
    if i >= grid.shape[0] - 1 or j >= grid.shape[1] - 1 or k >= grid.shape[2] - 1:
        return outside
    fx = x[0] - float(i)
    fy = x[1] - float(j)
    fz = x[2] - float(k)
    c00 = grid[i, j, k] * (1.0 - fx) + grid[i + 1, j, k] * fx
    c10 = grid[i, j + 1, k] * (1.0 - fx) + grid[i + 1, j + 1, k] * fx
    c01 = grid[i, j, k + 1] * (1.0 - fx) + grid[i + 1, j, k + 1] * fx
    c11 = grid[i, j + 1, k + 1] * (1.0 - fx) + grid[i + 1, j + 1, k + 1] * fx
    return (c00 * (1.0 - fy) + c10 * fy) * (1.0 - fz) + (c01 * (1.0 - fy) + c11 * fy) * fz


@wp.func
def sdf(g: Geo, p: wp.vec3):
    """Eyelet signed distance at p: fine grid, else coarse grid, else a lower bound."""
    d = trilinear(g.grid, (p - g.origin) * g.inv_h, -1.0e9)
    if d == -1.0e9:
        d = trilinear(g.grid2, (p - g.origin2) * g.inv_h2, g.far)
    return d


@wp.func
def pose6(x: float, y: float, z: float, rx: float, ry: float, rz: float):
    rv = wp.vec3(rx, ry, rz) * (wp.pi / 180.0)
    ang = wp.length(rv)
    q = wp.quat_identity()
    if ang > 1.0e-9:
        q = wp.quat_from_axis_angle(rv / ang, ang)
    return wp.transform(wp.vec3(x, y, z), q)


@wp.func
def pose_at(poses: wp.array2d(dtype=float), i: int):
    return pose6(poses[i, 0], poses[i, 1], poses[i, 2], poses[i, 3], poses[i, 4], poses[i, 5])


@wp.func
def pose_between(a: wp.array2d(dtype=float), b: wp.array2d(dtype=float), i: int, s: float):
    return pose6(a[i, 0] + (b[i, 0] - a[i, 0]) * s, a[i, 1] + (b[i, 1] - a[i, 1]) * s, a[i, 2] + (b[i, 2] - a[i, 2]) * s,
                 a[i, 3] + (b[i, 3] - a[i, 3]) * s, a[i, 4] + (b[i, 4] - a[i, 4]) * s, a[i, 5] + (b[i, 5] - a[i, 5]) * s)


@wp.func
def closest(g: Geo, X: wp.transform, cap: float):
    """Smallest signed distance of the hook's surface points to the eyelet (cap if none is below cap), and its point."""
    best = cap
    barg = int(-1)
    slack = float(0.01)  # trilinear interpolation is not exactly 1-Lipschitz
    for a in range(g.c1.shape[0]):
        if sdf(g, wp.transform_point(X, g.c1[a])) - g.r1[a] - slack < best:
            for b in range(g.s1[a], g.e1[a]):
                if sdf(g, wp.transform_point(X, g.c2[b])) - g.r2[b] - slack < best:
                    for c in range(g.s2[b], g.e2[b]):
                        d = sdf(g, wp.transform_point(X, g.pts[c]))
                        if d < best:
                            best = d
                            barg = c
    return best, barg


@wp.func
def far_estimate(g: Geo, X: wp.transform):
    """Separation estimated from the 1 mm sub-cells (centre distance minus radius: a lower bound within ~1 mm), and
    the sub-cell; for poses beyond the fine grid's reach."""
    best = float(1.0e9)
    barg = int(-1)
    for b in range(g.c2.shape[0]):
        d = sdf(g, wp.transform_point(X, g.c2[b])) - g.r2[b]
        if d < best:
            best = d
            barg = b
    return best, barg


@wp.func
def normal_at(g: Geo, p: wp.vec3, d: float):
    """Unit normal along which the eyelet pushes a hook point p at signed distance d, and the eyelet's closest point."""
    q = wp.mesh_query_point_no_sign(g.mesh, p, 1.0e6)
    cp = wp.mesh_eval_position(g.mesh, q.face, q.u, q.v)
    v = p - cp
    ln = wp.length(v)
    n = wp.mesh_eval_face_normal(g.mesh, q.face)
    if ln > 1.0e-6:
        n = v / ln
        if d < 0.0:
            n = -n
    return n, cp


@wp.kernel
def _separation(poses: wp.array2d(dtype=float), g: Geo, cap: float, out: wp.array(dtype=float),
                arg: wp.array(dtype=int)):
    i = wp.tid()
    X = pose_at(poses, i)
    d, c = closest(g, X, cap)
    if c < 0:
        d, b = far_estimate(g, X)
    out[i] = d
    arg[i] = c


@wp.kernel
def _contacts(poses: wp.array2d(dtype=float), g: Geo, cap: float, band: float, merge: float,
              out_d: wp.array2d(dtype=float), out_i: wp.array2d(dtype=int)):
    """Up to 4 distinct local minima (hook points more than merge apart) within band of the closest."""
    i = wp.tid()
    X = pose_at(poses, i)
    best = cap
    dd = wp.vec4(cap, cap, cap, cap)
    ii = vec4i(-1, -1, -1, -1)
    slack = float(0.01)
    for a in range(g.c1.shape[0]):
        if sdf(g, wp.transform_point(X, g.c1[a])) - g.r1[a] - slack < best + band:
            for b in range(g.s1[a], g.e1[a]):
                if sdf(g, wp.transform_point(X, g.c2[b])) - g.r2[b] - slack < best + band:
                    for c in range(g.s2[b], g.e2[b]):
                        d = sdf(g, wp.transform_point(X, g.pts[c]))
                        if d < best + band:
                            p = g.pts[c]
                            j = int(-1)
                            for s in range(4):
                                if ii[s] >= 0:
                                    if wp.length(g.pts[ii[s]] - p) < merge:
                                        j = s
                            if j >= 0:
                                if d < dd[j]:
                                    dd[j] = d
                                    ii[j] = c
                            else:
                                w = int(0)
                                for s in range(1, 4):
                                    if dd[s] > dd[w]:
                                        w = s
                                if d < dd[w]:
                                    dd[w] = d
                                    ii[w] = c
                            if d < best:
                                best = d
    for s in range(4):
        out_d[i, s] = dd[s]
        out_i[i, s] = ii[s]


@wp.kernel
def _witness(poses: wp.array2d(dtype=float), g: Geo, idx: wp.array2d(dtype=int), dist: wp.array2d(dtype=float),
             ph: wp.array2d(dtype=wp.vec3), pe: wp.array2d(dtype=wp.vec3), nrm: wp.array2d(dtype=wp.vec3)):
    i, s = wp.tid()
    c = idx[i, s]
    if c >= 0:
        X = pose_at(poses, i)
        p = wp.transform_point(X, g.pts[c])
        n, cp = normal_at(g, p, dist[i, s])
        ph[i, s] = p
        pe[i, s] = cp
        nrm[i, s] = n


@wp.kernel
def _threading(poses: wp.array2d(dtype=float), centre: wp.array(dtype=wp.vec3), part: wp.array(dtype=int),
               hole: wp.array2d(dtype=wp.int32), y0: float, z0: float, res: float, band: float,
               out: wp.array(dtype=int)):
    i = wp.tid()
    X = pose_at(poses, i)
    bits = int(0)
    for c in range(centre.shape[0]):
        p = wp.transform_point(X, centre[c])
        if wp.abs(p[0]) < band:
            iy = int(wp.round((p[1] - y0) / res))
            iz = int(wp.round((p[2] - z0) / res))
            if iy >= 0 and iy < hole.shape[1] and iz >= 0 and iz < hole.shape[0]:
                if hole[iz, iy] != 0:
                    bits = bits | part[c]
    out[i] = bits


@wp.kernel
def _project(poses: wp.array2d(dtype=float), g: Geo, cap: float, target: float, tol: float, iters: int,
             out: wp.array2d(dtype=float), ok: wp.array(dtype=int)):
    """As mode_graph_experiments.contact_project: translate by -(s - target) n until |s - target| <= tol."""
    i = wp.tid()
    t = wp.vec3(poses[i, 0], poses[i, 1], poses[i, 2])
    q = wp.transform_get_rotation(pose_at(poses, i))
    done = int(0)
    for it in range(iters + 4):  # a few extra steps for poses that start beyond the fine grid
        X = wp.transform(t, q)
        d, c = closest(g, X, cap)
        if wp.abs(d - target) <= tol:
            done = 1
            break
        p = wp.vec3(0.0, 0.0, 0.0)
        if c >= 0:
            p = wp.transform_point(X, g.pts[c])
        else:  # beyond the fine grid: step in along the estimated normal
            d, b = far_estimate(g, X)
            p = wp.transform_point(X, g.c2[b])
        n, cp = normal_at(g, p, d)
        t = t - (d - target) * n
    out[i, 0] = t[0]
    out[i, 1] = t[1]
    out[i, 2] = t[2]
    out[i, 3] = poses[i, 3]
    out[i, 4] = poses[i, 4]
    out[i, 5] = poses[i, 5]
    ok[i] = done


@wp.kernel
def _slide(a: wp.array2d(dtype=float), b: wp.array2d(dtype=float), nstep: wp.array(dtype=int), g: Geo, cap: float,
           target: float, reach: float, stray: wp.array(dtype=float), jump: wp.array(dtype=float)):
    """As mode_graph_experiments.slide_task: largest |separation| along a -> b (stopping beyond reach) and largest
    jump of its projection step onto contact."""
    i = wp.tid()
    n = nstep[i]
    worst = float(0.0)
    jmp = float(0.0)
    prev = wp.vec3(0.0, 0.0, 0.0)
    first = int(1)
    for k in range(n + 1):
        X = pose_between(a, b, i, float(k) / float(n))
        d, c = closest(g, X, cap)
        worst = wp.max(worst, wp.abs(d))
        if worst > reach:
            break
        nn = wp.vec3(0.0, 0.0, 0.0)
        if c >= 0:
            nn, cp = normal_at(g, wp.transform_point(X, g.pts[c]), d)
        shift = -(d - target) * nn
        if first == 0:
            jmp = wp.max(jmp, wp.length(shift - prev))
        first = 0
        prev = shift
    stray[i] = worst
    jump[i] = jmp


@wp.kernel
def _segment_min(a: wp.array2d(dtype=float), b: wp.array2d(dtype=float), nstep: wp.array(dtype=int), g: Geo,
                 cap: float, floor: float, out: wp.array(dtype=float)):
    """As validate_goal_field.segment_min: smallest separation along a -> b, stopping once below floor."""
    i = wp.tid()
    n = nstep[i]
    worst = cap
    for k in range(n + 1):
        d, c = closest(g, pose_between(a, b, i, float(k) / float(n)), cap)
        worst = wp.min(worst, d)
        if worst < floor:
            break
    out[i] = worst


def _load_part(name, yaw):
    m = trimesh.load(ASSET_DIR / f"{name}.stl")
    m.apply_scale(INCH * 1e3)
    m.apply_transform(trimesh.transformations.rotation_matrix(np.deg2rad(yaw), [0, 0, 1]))
    return m


def _hierarchy(P):
    """Hook surface points sorted into 5 mm cells and 1 mm sub-cells: centres, bounding radii and index ranges."""
    k2 = np.floor(P).astype(np.int64)
    k1 = np.floor(k2 / 5.0).astype(np.int64)
    order = np.lexsort((k2[:, 2], k2[:, 1], k2[:, 0], k1[:, 2], k1[:, 1], k1[:, 0]))
    P, k1, k2 = P[order], k1[order], k2[order]
    s2 = np.flatnonzero(np.r_[True, (np.diff(k2, axis=0) != 0).any(1)])
    e2 = np.r_[s2[1:], len(P)]
    c2 = np.array([P[a:b].mean(0) for a, b in zip(s2, e2)])
    r2 = np.array([np.linalg.norm(P[a:b] - c, axis=1).max() for a, b, c in zip(s2, e2, c2)])
    g1 = k1[s2]
    s1 = np.flatnonzero(np.r_[True, (np.diff(g1, axis=0) != 0).any(1)])
    e1 = np.r_[s1[1:], len(s2)]
    c1 = np.array([P[s2[a]:e2[b - 1]].mean(0) for a, b in zip(s1, e1)])
    r1 = np.array([np.linalg.norm(P[s2[a]:e2[b - 1]] - c, axis=1).max() for a, b, c in zip(s1, e1, c1)])
    return dict(pts=P, c1=c1, r1=r1, s1=s1, e1=e1, c2=c2, r2=r2, s2=s2, e2=e2)


class WarpContact:
    def __init__(self, h=0.1, margin=12.0, far_margin=40.0, n_points=200000, device="cuda:0", cache=CACHE,
                 verbose=True):
        wp.init()
        self.dev, self.cap = device, margin - 1.0
        cache.mkdir(parents=True, exist_ok=True)
        path = cache / f"hook_points{n_points}.npz"
        if path.exists():
            z = dict(np.load(path))
        else:
            from hook_topology import HookLoop
            P, _ = trimesh.sample.sample_surface_even(_load_part("hook", HOOK_YAW_DEG), n_points, seed=0)
            z = _hierarchy(P)
            loop = HookLoop(n_centre=8000)  # the CPU pipeline's threading labels
            z.update(centre=loop.centre, centre_part=loop.centre_part, hole=loop.hole.astype(np.int32),
                     hole_y0=loop.ys[0], hole_z0=loop.zs[0], hole_res=loop.res)
            np.savez(path, **z)
        self.hook = z
        eye = _load_part("eyelet", EYELET_YAW_DEG)
        arr = lambda x, dt: wp.array(x, dtype=dt, device=device)
        self.mesh = wp.Mesh(points=arr(eye.vertices, wp.vec3), indices=arr(eye.faces.flatten().astype(np.int32), int),
                            support_winding_number=True)
        t0 = time.perf_counter()

        def build(step, pad):
            lo, hi = eye.bounds[0] - pad, eye.bounds[1] + pad
            shape = tuple(int(s) for s in np.ceil((hi - lo) / step).astype(int) + 1)
            grid = wp.zeros(shape, dtype=wp.float32, device=device)
            wp.launch(_grid_kernel, dim=shape, inputs=[self.mesh.id, wp.vec3(*lo), step, grid], device=device)
            return grid, lo, shape

        grid, lo, shape = build(h, margin)
        grid2, lo2, _ = build(0.5, far_margin)
        wp.synchronize()
        g = Geo()
        g.grid, g.origin, g.inv_h, g.mesh = grid, wp.vec3(*lo), 1.0 / h, self.mesh.id
        g.grid2, g.origin2, g.inv_h2, g.far = grid2, wp.vec3(*lo2), 2.0, far_margin
        for k in ("c1", "c2", "pts"):
            setattr(g, k, arr(z[k], wp.vec3))
        for k in ("r1", "r2"):
            setattr(g, k, arr(z[k].astype(np.float32), float))
        for k in ("s1", "e1", "s2", "e2"):
            setattr(g, k, arr(z[k].astype(np.int32), int))
        self.g = g
        self.centre = arr(z["centre"], wp.vec3)
        self.part = arr(z["centre_part"].astype(np.int32), int)
        self.hole = wp.array(z["hole"].astype(np.int32), dtype=wp.int32, device=device)
        if verbose:
            print(f"warp contact geometry: eyelet distance grid {shape} at {h} mm ({np.prod(shape) * 4 / 1e9:.2f} GB) "
                  f"and a 0.5 mm one to {far_margin:g} mm ({time.perf_counter() - t0:.1f} s); {len(z['pts']):,} hook "
                  f"points in {len(z['c1'])} cells; exact within {self.cap:g} mm", flush=True)

    def _p(self, poses):
        return wp.array(np.ascontiguousarray(poses, dtype=np.float32), dtype=float, device=self.dev)

    @staticmethod
    def _batches(n, size=1 << 20):
        return [(a, min(a + size, n)) for a in range(0, n, size)]

    def separation(self, poses, return_point=False):
        """Signed separation [mm] (estimated to ~1 mm beyond self.cap), and optionally the hook point index of the
        closest (-1 beyond self.cap)."""
        n = len(poses)
        sep, idx = np.empty(n, np.float32), np.empty(n, np.int32)
        for a, b in self._batches(n):
            out, arg = wp.empty(b - a, dtype=float, device=self.dev), wp.empty(b - a, dtype=int, device=self.dev)
            wp.launch(_separation, dim=b - a, inputs=[self._p(poses[a:b]), self.g, self.cap, out, arg], device=self.dev)
            sep[a:b], idx[a:b] = out.numpy(), arg.numpy()
        return (sep, idx) if return_point else sep

    def contacts(self, poses, band=2.0, merge=2.0):
        """Per pose: distinct contacts (up to MAXC, closest first) within band of the closest: dist (n, MAXC) [mm, inf
        where none], hook point (n, MAXC, 3) in the hook frame, hook and eyelet points in the eyelet frame and the
        normal along which the eyelet pushes the hook (n, MAXC, 3; NaN where none)."""
        n = len(poses)
        D = np.full((n, MAXC), np.inf, np.float32)
        out = {k: np.full((n, MAXC, 3), np.nan, np.float32) for k in ("hook_local", "hook", "eyelet", "normal")}
        pts = self.hook["pts"]
        for a, b in self._batches(n, 1 << 18):
            pw = self._p(poses[a:b])
            dd = wp.empty((b - a, MAXC), dtype=float, device=self.dev)
            ii = wp.empty((b - a, MAXC), dtype=int, device=self.dev)
            wp.launch(_contacts, dim=b - a, inputs=[pw, self.g, self.cap, band, merge, dd, ii], device=self.dev)
            d, i = dd.numpy(), ii.numpy()
            # tidy: drop slots beyond band of the closest, merge slots that ended up within merge, sort
            i = np.where(d <= d.min(1, keepdims=True) + band, i, -1)
            d = np.where(i >= 0, d, np.inf)
            for s in range(MAXC):
                for t in range(s + 1, MAXC):
                    near = (i[:, s] >= 0) & (i[:, t] >= 0) & (
                        np.linalg.norm(pts[np.maximum(i[:, s], 0)] - pts[np.maximum(i[:, t], 0)], axis=1) < merge)
                    lose = np.where(d[:, s] <= d[:, t], t, s)
                    for k in (s, t):
                        drop = near & (lose == k)
                        i[drop, k], d[drop, k] = -1, np.inf
            order = np.argsort(d, axis=1)
            d, i = np.take_along_axis(d, order, 1), np.take_along_axis(i, order, 1)
            vph, vpe, vn = (wp.zeros((b - a, MAXC), dtype=wp.vec3, device=self.dev) for _ in range(3))
            wp.launch(_witness, dim=(b - a, MAXC), inputs=[pw, self.g, wp.array(i, dtype=int, device=self.dev),
                                                           wp.array(np.nan_to_num(d, posinf=0.0), dtype=float,
                                                                    device=self.dev), vph, vpe, vn], device=self.dev)
            has = i >= 0
            D[a:b] = d
            out["hook_local"][a:b][has] = pts[i[has]]
            out["hook"][a:b][has], out["eyelet"][a:b][has], out["normal"][a:b][has] = \
                vph.numpy()[has], vpe.numpy()[has], vn.numpy()[has]
        return dict(dist=D, **out)

    def threading(self, poses, band=0.5):
        """hook_topology.HookLoop.threading for a batch of poses."""
        z, n = self.hook, len(poses)
        bits = np.empty(n, np.int32)
        for a, b in self._batches(n):
            out = wp.empty(b - a, dtype=int, device=self.dev)
            wp.launch(_threading, dim=b - a, inputs=[self._p(poses[a:b]), self.centre, self.part, self.hole,
                                                     float(z["hole_y0"]), float(z["hole_z0"]), float(z["hole_res"]),
                                                     band, out], device=self.dev)
            bits[a:b] = out.numpy()
        return bits

    def project(self, poses, target=0.02, tol=0.02, iters=8):
        """Slide each pose along its contact normal to separation target +- tol [mm]; (poses, reached)."""
        n = len(poses)
        res, ok = np.empty((n, 6), np.float64), np.empty(n, bool)
        for a, b in self._batches(n):
            out = wp.empty((b - a, 6), dtype=float, device=self.dev)
            k = wp.empty(b - a, dtype=int, device=self.dev)
            wp.launch(_project, dim=b - a, inputs=[self._p(poses[a:b]), self.g, self.cap, target, tol, iters, out, k],
                      device=self.dev)
            res[a:b], ok[a:b] = out.numpy(), k.numpy() > 0
        return res, ok

    @staticmethod
    def _nstep(a, b, step):
        return np.maximum(1, np.ceil(np.abs(b - a).max(1) / step)).astype(np.int32)

    def slide(self, a, b, step=0.25, target=0.02, reach=1.0):
        """Per segment a -> b: (stray, jump) [mm], as mode_graph_experiments.slide_task (stray capped just beyond
        reach: the slide is abandoned there)."""
        n = len(a)
        stray, jump = np.empty(n, np.float32), np.empty(n, np.float32)
        for s, e in self._batches(n, 1 << 18):
            o1, o2 = (wp.empty(e - s, dtype=float, device=self.dev) for _ in range(2))
            wp.launch(_slide, dim=e - s, inputs=[self._p(a[s:e]), self._p(b[s:e]),
                                                 wp.array(self._nstep(a[s:e], b[s:e], step), dtype=int, device=self.dev),
                                                 self.g, min(self.cap, reach + 0.01), target, reach, o1, o2],
                      device=self.dev)
            stray[s:e], jump[s:e] = o1.numpy(), o2.numpy()
        return stray, jump

    def segment_min(self, a, b, step=0.25, floor=-np.inf, cap=2.0):
        """Smallest separation along each segment a -> b [mm], capped at cap (cheap: farther cells are skipped),
        stopping once below floor."""
        n = len(a)
        out = np.empty(n, np.float32)
        for s, e in self._batches(n, 1 << 18):
            o = wp.empty(e - s, dtype=float, device=self.dev)
            wp.launch(_segment_min, dim=e - s, inputs=[self._p(a[s:e]), self._p(b[s:e]),
                                                       wp.array(self._nstep(a[s:e], b[s:e], step), dtype=int,
                                                                device=self.dev),
                                                       self.g, min(self.cap, cap), float(max(floor, -1e30)), o],
                      device=self.dev)
            out[s:e] = o.numpy()
        return out


@wp.kernel
def _brute(poses: wp.array2d(dtype=float), mesh: wp.uint64, pts: wp.array(dtype=wp.vec3), out: wp.array(dtype=float)):
    i, c = wp.tid()
    p = wp.transform_point(pose_at(poses, i), pts[c])
    q = wp.mesh_query_point_sign_winding_number(mesh, p, 100.0)
    if q.result:
        wp.atomic_min(out, i, q.sign * wp.length(p - wp.mesh_eval_position(mesh, q.face, q.u, q.v)))


def validate(args):
    """Separation against brute force over every hook point; threading, projection, normals and slides against the
    CPU pipeline (mode_graph_experiments, MuJoCo convex parts)."""
    import mode_graph_experiments as mge
    from hook_topology import HookLoop

    geo = WarpContact()
    exp = args.experiment_dir
    rng = np.random.default_rng(0)
    con = np.load(exp / "contact_graph.npz")
    sam = np.load(exp / "samples_graph.npz")["poses"]
    box = sam[rng.integers(len(sam), size=3000)] + rng.uniform(-6, 6, (3000, 6))
    poses = np.vstack([con["poses"][rng.choice(len(con["poses"]), 3000, replace=False)],
                       sam[rng.choice(len(sam), 3000, replace=False)], box])
    kind = np.repeat([0, 1, 2], 3000)
    geo.separation(poses[:10])
    t0 = time.perf_counter()
    sep = geo.separation(poses)
    print(f"separation of {len(poses)} poses: {time.perf_counter() - t0:.3f} s (incl. transfer)", flush=True)
    ref = wp.full(len(poses), 1.0e6, dtype=float, device=geo.dev)
    pts = geo.g.pts
    for a in range(0, len(poses), 50):
        b = min(a + 50, len(poses))
        wp.launch(_brute, dim=(b - a, pts.shape[0]), inputs=[geo._p(poses[a:b]), geo.mesh.id, pts, ref[a:b]],
                  device=geo.dev)
    ref = ref.numpy()
    for k, name in enumerate(["contact", "free samples", "random box"]):
        m = (kind == k) & (ref < geo.cap - 0.5) & (ref > -2.4)
        far = (kind == k) & (ref >= geo.cap)
        e = np.abs(sep[m] - ref[m])
        print(f"  {name:12s} |warp - brute force| within {geo.cap - 0.5:g} mm: median {np.median(e):.4f}, p99 "
              f"{np.percentile(e, 99):.4f}, max {e.max():.4f} mm; sign disagreements "
              f"{np.sum(np.sign(sep[kind == k]) != np.sign(ref[kind == k]))}"
              + (f"; beyond {geo.cap:g} mm ({far.sum()}): estimate - exact p5/p95 "
                 f"{np.percentile(sep[far] - ref[far], [5, 95]).round(2)} mm" if far.sum() > 5 else ""), flush=True)

    loop = HookLoop(n_centre=8000)
    sub = rng.choice(len(poses), 3000, replace=False)
    cpu = np.array([loop.threading(p) for p in poses[sub]])
    gpu = geo.threading(poses[sub])
    print(f"threading: {np.mean(cpu == gpu):.2%} identical to HookLoop.threading ({np.sum(cpu > 0)} threaded)", flush=True)

    mge._init_features()
    starts = poses[kind > 0]
    q, ok = geo.project(starts)
    after = geo.separation(q)
    print(f"projection: {ok.mean():.1%} reached 0.02 +- 0.02 mm (of starts beyond {geo.cap:g} mm: "
          f"{ok[sep[kind > 0] >= geo.cap].mean():.1%}); separation after p1/p50/p99 "
          f"{np.percentile(after[ok], [1, 50, 99]).round(4)} mm", flush=True)
    c = geo.contacts(con["poses"][:2000])
    ang = []
    for p, n_gpu in zip(con["poses"][:2000], c["normal"][:, 0]):
        s, n_cpu = mge._probe.normal(p)
        if not np.isnan(n_cpu[0]):
            ang.append(np.degrees(np.arccos(np.clip(n_cpu @ n_gpu, -1, 1))))
    print(f"contact normals vs CPU MuJoCo (convex parts): median {np.median(ang):.2f} deg, p95 "
          f"{np.percentile(ang, 95):.2f} deg; distinct contacts within 2 mm per pose "
          f"{np.bincount(np.isfinite(c['dist']).sum(1), minlength=MAXC + 1)}", flush=True)
    pr = con["pairs"][rng.choice(len(con["pairs"]), 200000, replace=False)]
    a, b = con["poses"][pr[:, 0]], con["poses"][pr[:, 1]]
    geo.slide(a[:10], b[:10])
    t0 = time.perf_counter()
    stray, jump = geo.slide(a, b)
    dt = time.perf_counter() - t0
    key = con["pairs"][:, 0].astype(np.int64) * len(con["poses"]) + con["pairs"][:, 1]
    cpu_stray = con["stray"][np.searchsorted(key, pr[:, 0].astype(np.int64) * len(con["poses"]) + pr[:, 1])]
    print(f"slides: {len(a):,} in {dt:.2f} s ({len(a) / dt:,.0f}/s); stray vs the CPU graph: median |diff| "
          f"{np.median(np.abs(stray - cpu_stray)):.3f} mm; within 0.25 mm: gpu {np.mean(stray <= 0.25):.1%}, cpu "
          f"{np.mean(cpu_stray <= 0.25):.1%}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["validate"])
    parser.add_argument("experiment_dir", type=Path)
    args = parser.parse_args()
    validate(args)


if __name__ == "__main__":
    main()
