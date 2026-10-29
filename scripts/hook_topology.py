"""How the hook sits relative to the eyelet's loop: which hook parts thread the hole, and which parts are closest.

The hook's centreline (its surface moved inward by the wire radius, densely sampled) is split into parts
in the hook frame: the crook (below CROOK_Z), the tip arm (above it, +x side) and the shaft (above it,
-x side). The loop's hole is the free region enclosed by the loop in its plane x = 0 of the eyelet frame
(the eyelet STL turned by EYELET_YAW_DEG), found by slicing the mesh with winding numbers.

    threading(pose)   bitmask of hook parts crossing the loop plane inside the hole (TIP | CROOK | SHAFT)
    hook_part(p)      part of the hook at hook-frame point p [mm]
    eyelet_part(p)    part of the eyelet at eyelet-frame point p [mm]: top wire, side (+y / -y), bottom, stem
    loop_side(p)      side of the loop's mid-plane (+1: +y, -1: -y) of eyelet-frame point p [mm]
"""

import functools

import numpy as np
import trimesh
from scipy import ndimage
from scipy.spatial.transform import Rotation

from hook_eyelet_model import ASSET_DIR, EYELET_YAW_DEG, HOOK_YAW_DEG, INCH

TIP, CROOK, SHAFT = 1, 2, 4
HOOK_PARTS = {TIP: "tip", CROOK: "crook", SHAFT: "shaft"}
EYELET_PARTS = ["top", "side+y", "side-y", "bottom", "stem"]
CROOK_Z = 4.0  # hook-frame height [mm] below which the centreline is the crook


def winding(points, triangles, chunk=200):
    """Generalised winding number of a closed triangle mesh at points (> 0.5: inside)."""
    out = np.zeros(len(points))
    for i in range(0, len(points), chunk):
        q = points[i:i + chunk, None, :]
        a, b, c = (triangles[None, :, k, :] - q for k in range(3))
        la, lb, lc = (np.linalg.norm(x, axis=-1) for x in (a, b, c))
        num = np.einsum("pfi,pfi->pf", a, np.cross(b, c))
        den = (la * lb * lc + np.einsum("pfi,pfi->pf", a, b) * lc + np.einsum("pfi,pfi->pf", b, c) * la
               + np.einsum("pfi,pfi->pf", c, a) * lb)
        out[i:i + chunk] = 2 * np.arctan2(num, den).sum(1) / (4 * np.pi)
    return out


def hook_part(p):
    """Part (TIP, CROOK or SHAFT) of hook-frame points p [mm], shape (..., 3)."""
    p = np.asarray(p)
    return np.where(p[..., 2] < CROOK_Z, CROOK, np.where(p[..., 0] > 0, TIP, SHAFT))


def eyelet_part(p):
    """Index into EYELET_PARTS of eyelet-frame points p [mm], shape (..., 3)."""
    p = np.asarray(p)
    z, y = p[..., 2], p[..., 1]
    return np.select([z >= -1.0, (z >= -14.0) & (y >= 0), z >= -14.0, z >= -20.0], [0, 1, 2, 3], 4)


@functools.cache
def loop_mid_y():
    """y of the loop's mid-plane [mm]: halfway across the loop's vertices (eyelet frame, above z = -12 mm)."""
    eye = trimesh.load(ASSET_DIR / "eyelet.stl")
    eye.apply_scale(INCH * 1e3)
    eye.apply_transform(trimesh.transformations.rotation_matrix(np.deg2rad(EYELET_YAW_DEG), [0, 0, 1]))
    v = eye.vertices[eye.vertices[:, 2] > -12]
    return float(v[:, 1].min() + v[:, 1].max()) / 2


def loop_side(p):
    """Side of the loop's mid-plane that eyelet-frame points p [mm], shape (..., 3), lie on: +1 (+y) or -1 (-y)."""
    return np.where(np.asarray(p)[..., 1] >= loop_mid_y(), 1, -1)


def threading_name(bits):
    """'unthreaded', 'tip threaded', 'tip+crook threaded', ... for a threading bitmask."""
    parts = [name for bit, name in HOOK_PARTS.items() if bits & bit]
    return "+".join(parts) + " threaded" if parts else "unthreaded"


class HookLoop:
    def __init__(self, n_centre=60000, res=0.25, seed=0):
        rz = lambda deg: trimesh.transformations.rotation_matrix(np.deg2rad(deg), [0, 0, 1])
        hook = trimesh.load(ASSET_DIR / "hook.stl")
        hook.apply_scale(INCH * 1e3)
        hook.apply_transform(rz(HOOK_YAW_DEG))
        radius = np.ptp(hook.vertices[:, 1]) / 2
        # Dense surface samples: an STL's straight runs have vertices only at their ends.
        pts, face = trimesh.sample.sample_surface_even(hook, n_centre, seed=seed)
        self.centre = pts - radius * hook.face_normals[face]
        self.centre_part = hook_part(self.centre)
        eye = trimesh.load(ASSET_DIR / "eyelet.stl")
        eye.apply_scale(INCH * 1e3)
        eye.apply_transform(rz(EYELET_YAW_DEG))
        self.ys, self.zs, self.res = np.arange(-15, 15, res), np.arange(-25, 10, res), res
        yy, zz = np.meshgrid(self.ys, self.zs)
        grid = np.column_stack([np.zeros(yy.size), yy.ravel(), zz.ravel()])
        material = (winding(grid, eye.triangles) > 0.5).reshape(yy.shape)
        labels, _ = ndimage.label(~material)
        border = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
        self.hole = (~material) & ~np.isin(labels, border)

    def threading(self, pose, band=0.5, parts=TIP | CROOK | SHAFT):
        """Bitmask of hook parts crossing the loop plane inside the hole; pose: mm and rotation vector [deg]."""
        sel = (self.centre_part & parts) > 0
        p = Rotation.from_rotvec(pose[3:], degrees=True).apply(self.centre[sel]) + pose[:3]
        near = np.abs(p[:, 0]) < band
        iy = np.rint((p[near, 1] - self.ys[0]) / self.res).astype(int)
        iz = np.rint((p[near, 2] - self.zs[0]) / self.res).astype(int)
        ok = (iy >= 0) & (iy < len(self.ys)) & (iz >= 0) & (iz < len(self.zs))
        inside = np.zeros(near.sum(), bool)
        inside[ok] = self.hole[iz[ok], iy[ok]]
        return int(np.bitwise_or.reduce(self.centre_part[sel][near][inside], initial=0))
