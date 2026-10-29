"""GPU separation model of the hook/eyelet pair, for particle estimators of the eyelet pose (pose_filters.py).

The eyelet is a signed distance field on a voxel grid (its frame, mm), the hook its wire centreline with a radius
per point (hook frame): the hook is the union of balls along the centreline, and its separation from the eyelet at
a relative pose is min_j sdf(R c_j + t) - r_j. This is Klingensmith, Koval et al.'s implicit contact manifold
(ICRA 2017: environment SDF, robot as spheres). Both are built from MuJoCo's collision geometry
(hook_eyelet_model: the convex parts, which bulge past the true surface by up to ~0.25 mm), so the field
matches what the simulation collides:
  sdf      distance to the boundary of the union of the eyelet's convex parts (dense boundary samples, kd-tree),
           negative inside any part; trilinear between nodes; beyond the grid, the border value plus the distance
           to the grid (an upper-bound-like extension; there the hook is mm away anyway);
  radius   each centreline point's distance to the boundary of the union of the hook's convex parts (the largest
           ball about it inside the hook); the flat end caps, which balls round off, as surface points of radius 0
           (the balls still miss the parts' bulges by ~0.02 mm, up to ~0.06: the model reads separations a few
           hundredths of a mm high).
Cached (FIELD) on first use. Poses as everywhere here: position mm, rotation vector deg; hook poses (k, 7) mm,
wxyz in the robot's frame, eyelet poses g (6) in the robot's frame (contact_information.py conventions).
"""

from pathlib import Path

import mujoco
import numpy as np
import torch
from scipy.spatial import ConvexHull, cKDTree

import hook_eyelet_model as H
from hook_linking import centrelines, resample

FIELD = Path("/media/rp/Elements1/abhay_ws/contact-sampling/fields/eyelet_sdf.npz")
# The eyelet's smooth visual mesh instead of its convex parts (same grid, same hook): the field without the
# parts' faceting and bulges, for testing what the collision geometry adds.
FIELD_SMOOTH = FIELD.with_name("eyelet_sdf_smooth.npz")


def collision_hulls(body):
    """Convex parts of a body as (vertices (n, 3) mm, hull equations (f, 4)) in its trajectory frame."""
    m = H.build_model()
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)  # hook at the identity: its parts sit in the hook frame
    out = []
    for g in np.where((m.geom_group == H.COLLISION_GROUP) & (m.geom_bodyid == m.body(body).id))[0]:
        mid = m.geom_dataid[g]
        v = m.mesh_vert[m.mesh_vertadr[mid]: m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
        w = (d.geom_xpos[g] + v @ d.geom_xmat[g].reshape(3, 3).T) * 1e3
        hull = ConvexHull(w)
        out.append((w[hull.vertices], hull.equations))
    return out


def boundary_samples(hulls, density, seed=0):
    """Samples of the boundary of the union of convex hulls, about `density` per mm^2."""
    rng = np.random.default_rng(seed)
    pts = []
    for i, (v, _) in enumerate(hulls):
        hull = ConvexHull(v)
        tri = v[hull.simplices]
        area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
        n = rng.multinomial(int(area.sum() * density), area / area.sum())
        f = np.repeat(np.arange(len(tri)), n)
        a, b = rng.random((2, len(f)))
        flip = a + b > 1
        a[flip], b[flip] = 1 - a[flip], 1 - b[flip]
        p = tri[f, 0] + a[:, None] * (tri[f, 1] - tri[f, 0]) + b[:, None] * (tri[f, 2] - tri[f, 0])
        others = [h for j, h in enumerate(hulls) if j != i]
        pts.append(p[~inside_any(p, others, margin=-0.01)])  # shared faces between parts are interior
    return np.vstack(pts)


def inside_any(p, hulls, margin=0.0):
    """Mask of points inside some hull by more than margin (each hull's bounding box first)."""
    out = np.zeros(len(p), bool)
    for v, eq in hulls:
        pad = max(0.0, -margin)
        box = np.all((p >= v.min(0) - pad) & (p <= v.max(0) + pad), axis=1) & ~out
        idx = np.where(box)[0]
        out[idx] = np.max(p[idx] @ eq[:, :3].T + eq[:, 3], axis=1) < -margin
    return out


def build_field(res=0.2, margin=6.0, density=100.0, line_spacing=0.4, cap_tol=0.05, cap_spacing=0.7):
    eyelet = collision_hulls("eyelet")
    allv = np.vstack([v for v, _ in eyelet])
    lo, hi = allv.min(0) - margin, allv.max(0) + margin
    shape = np.ceil((hi - lo) / res).astype(int) + 1
    tree = cKDTree(boundary_samples(eyelet, density))
    axes = [lo[a] + res * np.arange(shape[a]) for a in range(3)]
    sdf = np.empty(shape, np.float32)
    for i, x in enumerate(axes[0]):  # one x slab at a time
        p = np.stack(np.meshgrid([x], axes[1], axes[2], indexing="ij"), -1).reshape(-1, 3)
        dist = tree.query(p, workers=-1)[0]
        dist[inside_any(p, eyelet)] *= -1
        sdf[i] = dist.reshape(shape[1], shape[2])
    hook_line, *_ = centrelines()
    s = np.sum(np.linalg.norm(np.diff(hook_line, axis=0), axis=1))
    line = resample(hook_line, int(np.ceil(s / line_spacing)) + 1)
    hook = collision_hulls("hook")
    surface = boundary_samples(hook, density)
    radius = cKDTree(surface).query(line, workers=-1)[0]
    radius[~inside_any(line, hook)] *= -1  # (none expected)
    # The flat end caps: balls about the centreline round them off by up to ~1.2 mm. Their surface samples that the
    # balls miss by over cap_tol join the list as points of radius 0, ordered as a nearest-neighbour chain so
    # that neighbours in the list are neighbours on the cap (fast_separations' windows rely on it).
    gap = np.min(np.linalg.norm(surface[:, None] - line[None], axis=-1) - radius, axis=1)
    near = np.argmin(np.linalg.norm(surface[:, None] - line[None], axis=-1), axis=1)
    caps = []
    for end in (np.where(near <= 6)[0], np.where(near >= len(line) - 7)[0]):
        p = surface[end[gap[end] > cap_tol]]
        _, keep = np.unique(np.floor(p / cap_spacing).astype(int), axis=0, return_index=True)
        caps.append(chain(p[keep]))
    top_end = caps[0][np.argsort(np.linalg.norm(caps[0] - line[0], axis=1))[::-1]] if len(caps[0]) else caps[0]
    tip = caps[1][np.argsort(np.linalg.norm(caps[1] - line[-1], axis=1))] if len(caps[1]) else caps[1]
    points = np.vstack([chain(top_end, start=0), line, chain(tip, start=0)])
    radius = np.concatenate([np.zeros(len(top_end)), radius, np.zeros(len(tip))])
    FIELD.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(FIELD, sdf=sdf, lo=lo, res=res, line=points, radius=radius, n_top_cap=len(top_end),
                        n_tip_cap=len(tip))


def build_smooth_field(density=400.0, near=1.0, seed=0):
    """FIELD_SMOOTH: the eyelet's visual mesh (watertight) on FIELD's grid. Distance to dense surface samples, signed
    by the nearest sample's outward face normal; within `near` mm of the surface the point-to-plane distance of
    that sample's face (exact for a locally flat surface, so no sample-spacing error). The hook is FIELD's."""
    if not FIELD.exists():
        build_field()
    m = H.build_model()
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    g = next(g for g in np.where(m.geom_bodyid == m.body("eyelet").id)[0]
             if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and m.geom_group[g] != H.COLLISION_GROUP)
    mid = m.geom_dataid[g]
    v = m.mesh_vert[m.mesh_vertadr[mid]: m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
    tri = (d.geom_xpos[g] + v @ d.geom_xmat[g].reshape(3, 3).T)[m.mesh_face[m.mesh_faceadr[mid]:
                                                                             m.mesh_faceadr[mid] + m.mesh_facenum[mid]]] * 1e3
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = 0.5 * np.linalg.norm(cross, axis=1)
    normal = cross / np.linalg.norm(cross, axis=1, keepdims=True)
    rng = np.random.default_rng(seed)
    f = np.repeat(np.arange(len(tri)), rng.multinomial(int(area.sum() * density), area / area.sum()))
    a, b = rng.random((2, len(f)))
    flip = a + b > 1
    a[flip], b[flip] = 1 - a[flip], 1 - b[flip]
    pts = tri[f, 0] + a[:, None] * (tri[f, 1] - tri[f, 0]) + b[:, None] * (tri[f, 2] - tri[f, 0])
    tree = cKDTree(pts)
    with np.load(FIELD) as z:
        lo, res, shape, keep = z["lo"], float(z["res"]), z["sdf"].shape, {k: z[k] for k in
                                                                         ("line", "radius", "n_top_cap", "n_tip_cap")}
    axes = [lo[a] + res * np.arange(shape[a]) for a in range(3)]
    sdf = np.empty(shape, np.float32)
    for i, x in enumerate(axes[0]):
        p = np.stack(np.meshgrid([x], axes[1], axes[2], indexing="ij"), -1).reshape(-1, 3)
        dist, j = tree.query(p, workers=-1)
        plane = np.einsum("na,na->n", p - pts[j], normal[f[j]])
        sdf[i] = np.where(dist < near, plane, np.sign(plane) * dist).reshape(shape[1], shape[2])
    np.savez_compressed(FIELD_SMOOTH, sdf=sdf, lo=lo, res=res, **keep)


def chain(p, start=0):
    """Points reordered as a greedy nearest-neighbour path from p[start]."""
    if len(p) < 2:
        return p
    left = np.ones(len(p), bool)
    order = [start]
    left[start] = False
    for _ in range(len(p) - 1):
        d = np.linalg.norm(p - p[order[-1]], axis=1)
        d[~left] = np.inf
        order.append(int(np.argmin(d)))
        left[order[-1]] = False
    return p[order]


def rotvec_to_matrix(rv):
    """(..., 3, 3) rotation matrices of rotation vectors (..., 3) in radians (Rodrigues)."""
    th = torch.linalg.norm(rv, dim=-1, keepdim=True)
    small = th < 1e-8
    k = rv / torch.where(small, torch.ones_like(th), th)
    K = torch.zeros(rv.shape[:-1] + (3, 3), dtype=rv.dtype, device=rv.device)
    K[..., 0, 1], K[..., 0, 2], K[..., 1, 2] = -k[..., 2], k[..., 1], -k[..., 0]
    K = K - K.transpose(-1, -2)
    s, c = torch.sin(th)[..., None], torch.cos(th)[..., None]
    eye = torch.eye(3, dtype=rv.dtype, device=rv.device).expand_as(K)
    R = eye + s * K + (1 - c) * (K @ K)
    return torch.where(small[..., None], eye + K * th[..., None], R)


def quat_to_matrix(q):
    """(..., 3, 3) from quaternions wxyz (..., 4)."""
    q = q / torch.linalg.norm(q, dim=-1, keepdim=True)
    w, x, y, z = q.unbind(-1)
    return torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1).reshape(q.shape[:-1] + (3, 3))


def _trilinear(flat, dims, lo, res, local):
    """Signed distance at eyelet-frame points local (..., 3) from the flattened grid (C order, dims (3,))."""
    u = (local - lo) / res
    uc = torch.minimum(torch.maximum(u, torch.zeros_like(lo)), dims - 1.0001)
    outside = torch.linalg.norm(u - uc, dim=-1) * res
    i0 = uc.floor()
    f = uc - i0
    i0 = i0.long()
    ny, nz = dims[1].long(), dims[2].long()
    base = (i0[..., 0] * ny + i0[..., 1]) * nz + i0[..., 2]
    fx, fy, fz = f.unbind(-1)
    sx, sy = ny * nz, nz
    c00 = flat[base] + (flat[base + 1] - flat[base]) * fz
    c01 = flat[base + sy] + (flat[base + sy + 1] - flat[base + sy]) * fz
    c10 = flat[base + sx] + (flat[base + sx + 1] - flat[base + sx]) * fz
    c11 = flat[base + sx + sy] + (flat[base + sx + sy + 1] - flat[base + sx + sy]) * fz
    c0 = c00 + (c01 - c00) * fy
    c1 = c10 + (c11 - c10) * fy
    return c0 + (c1 - c0) * fx + outside


def _clustered(R, t, pts, radius, medoid, members, flat, dims, lo, res, top: int):
    """Separations (N, k): each cluster's medoid first, then every member of the `top` clusters whose medoids
    read lowest (members include the medoids). t (N, 1 or k, 3): the eyelet's position, per sample if k."""
    N, k = R.shape[0], pts.shape[0]
    local = torch.einsum("Nba,Nknb->Nkna", R, pts[:, medoid][None] - t[:, :, None, :])
    vc = _trilinear(flat, dims, lo, res, local) - radius[medoid]
    arg = torch.topk(vc, top, dim=-1, largest=False)[1]
    idx = members[arg].reshape(N, k, -1)  # N, k, top * C
    near = pts[torch.arange(k, device=R.device)[None, :, None], idx]
    local = torch.einsum("Nba,Nkwb->Nkwa", R, near - t[:, :, None, :])
    return (_trilinear(flat, dims, lo, res, local) - radius[idx]).amin(-1)


_CLUSTERED = torch.compile(_clustered, dynamic=True)


def clusters(points, line, stride=8, cap_size=16):
    """Candidate groups of hook points for fast_separations: (medoid indices (G,), member indices (G, M), padded
    with the medoid). Along the centreline (indices line[0]..line[1] - 1) every stride-th point with the points
    within stride of it (overlapping windows: neighbours along the wire); the cap points in compact clusters of at
    most cap_size, by recursive halving along the widest axis."""
    def split(idx):
        if len(idx) <= cap_size:
            return [idx]
        p = points[idx]
        order = idx[np.argsort(p[:, np.argmax(np.ptp(p, axis=0))])]
        return split(order[: len(order) // 2]) + split(order[len(order) // 2:])
    a, b = line
    groups = [(m, np.arange(max(a, m - stride), min(b, m + stride + 1))) for m in range(a, b, stride)]
    for cap in (np.arange(0, a), np.arange(b, len(points))):
        if len(cap):
            for c in split(cap):
                groups.append((c[np.argmin(np.linalg.norm(points[c] - points[c].mean(0), axis=1))], c))
    width = max(len(c) for _, c in groups)
    medoid = np.array([m for m, _ in groups])
    members = np.array([np.concatenate([c, np.full(width - len(c), m)]) for m, c in groups])
    return medoid, members


class GpuContactModel:
    """Separations of hook poses against eyelets at many poses at once (module docstring)."""

    def __init__(self, device="cuda", field=FIELD):
        field = Path(field)
        if not field.exists():
            build_smooth_field() if field == FIELD_SMOOTH else build_field()
        with np.load(field) as z:
            self.device = torch.device(device)
            f = torch.float32
            self.grid = torch.as_tensor(z["sdf"], dtype=f, device=self.device)
            self.lo = torch.as_tensor(z["lo"], dtype=f, device=self.device)
            self.res = float(z["res"])
            self.line = torch.as_tensor(z["line"], dtype=f, device=self.device)
            self.radius = torch.as_tensor(z["radius"], dtype=f, device=self.device)
            self.wire = (int(z["n_top_cap"]), len(z["line"]) - int(z["n_tip_cap"]))  # centreline index range
        self.shape = torch.tensor(self.grid.shape, dtype=f, device=self.device)
        self.hi = self.lo + (self.shape - 1) * self.res
        self._vol = self.grid.permute(2, 1, 0)[None, None]  # grid_sample indexes (x, y, z) as (W, H, D)
        self._flat = self.grid.contiguous().view(-1)
        self.set_clusters()

    def set_clusters(self, stride=8, cap_size=12):
        medoid, members = clusters(self.line.cpu().numpy(), self.wire, stride, cap_size)
        self.medoid = torch.as_tensor(medoid, device=self.device)
        self.members = torch.as_tensor(members, device=self.device)

    def sdf(self, p):
        """Signed distance (mm) of eyelet-frame points p (..., 3) mm."""
        shp = p.shape[:-1]
        p = p.reshape(-1, 3)
        q = torch.minimum(torch.maximum(p, self.lo), self.hi)
        u = (q - self.lo) / (self.hi - self.lo) * 2 - 1
        v = torch.nn.functional.grid_sample(self._vol, u.view(1, -1, 1, 1, 3), mode="bilinear",
                                            align_corners=True).view(-1)
        return (v + torch.linalg.norm(p - q, dim=-1)).view(shp)

    def hook_points(self, poses):
        """Centreline points of hook poses (k, 7: mm, wxyz) in the robot's frame: (k, n, 3)."""
        poses = torch.as_tensor(poses, dtype=torch.float32, device=self.device)
        R = quat_to_matrix(poses[:, 3:])
        return poses[:, None, :3] + torch.einsum("kab,nb->kna", R, self.line)

    def separations(self, g, pts, chunk=2 ** 24):
        """Separations (N, k) mm of hook centreline points pts (k, n, 3) (hook_points) from eyelets at poses
        g (N, 6: mm, rotation vector deg, torch). Differentiable in g."""
        R = rotvec_to_matrix(torch.deg2rad(g[:, 3:]))  # eyelet orientation in the robot's frame
        k, n, _ = pts.shape
        per = max(1, chunk // (k * n))
        out = []
        for a in range(0, len(g), per):
            local = torch.einsum("Nba,Nknb->Nkna", R[a:a + per], pts[None] - g[a:a + per, None, None, :3])
            out.append((self.sdf(local) - self.radius).amin(-1))
        return torch.cat(out)

    def fast_separations(self, g, pts, top=4, chunk=2 ** 22, shift=None):
        """separations() from the hook's point clusters (clusters()): exact unless the closest point lies in none
        of the `top` clusters whose medoids read lowest, and then only too high; compiled, not differentiable.
        shift (N, k, 3): each hook translated by it (mm, robot frame) for each eyelet pose."""
        R = rotvec_to_matrix(torch.deg2rad(g[:, 3:]))
        k = pts.shape[0]
        t = g[:, None, :3] if shift is None else g[:, None, :3] - shift
        per = max(1, chunk // (k * (len(self.medoid) + top * self.members.shape[1])))
        with torch.no_grad():
            return torch.cat([_CLUSTERED(R[a:a + per], t[a:a + per], pts, self.radius, self.medoid, self.members,
                                         self._flat, self.shape, self.lo, self.res, top)
                              for a in range(0, len(g), per)])
