"""Threading coordinate for the hook/eyelet pair: the Gauss linking integral of the hook's wire centreline
with the closed centreline of the eyelet's loop.

The linking integral of an open curve with a closed loop is continuous wherever the two curves do not cross,
and the wires cannot cross without interpenetrating, so along any collision-free motion it changes smoothly.
It is ~0 with the hook clear of the loop and +0.98 in the assembled pose (the hook threads the loop once, its
ends far from the hole); threading the other way round (the hook yawed 180 deg) gives -0.98. It does not say
which hook part threads: the shaft through the hole also reads near +-1.

Centrelines (eyelet frame, mm; see hook_eyelet_model.py for the frames): both wires lie in a plane (the hook in
y = 0 of its frame, the loop in x = 0), so each centreline is the medial line of the wire. It is found by
moving surface samples inward by the wire radius; the hook's samples are then ordered along their spanning
tree's longest path and the loop's by angle about the loop's centre. The open ends of the hook are extended
over the end caps by the wire radius. Cached in assets/hook/centrelines.npz.

    linking(hook_points, loop)        linking integral of hook polylines (..., n, 3) with the closed loop (m, 3)
    linking_many(hook_points, loop)   the same for (N, n, 3) polylines, compiled (~20x faster)
"""

import numpy as np
import scipy.sparse as sp
import trimesh
from numba import njit
from scipy.sparse.csgraph import minimum_spanning_tree, shortest_path
from scipy.spatial import cKDTree

from hook_eyelet_model import ASSET_DIR, EYELET_YAW_DEG, HOOK_YAW_DEG, INCH

CACHE = ASSET_DIR / "centrelines.npz"
LOOP_CENTRE_Z = -6.5  # roughly the middle of the loop's hole [mm], for ordering its centreline by angle


def _mesh(name, yaw_deg):
    mesh = trimesh.load(ASSET_DIR / f"{name}.stl")
    mesh.apply_scale(INCH * 1e3)
    mesh.apply_transform(trimesh.transformations.rotation_matrix(np.deg2rad(yaw_deg), [0, 0, 1]))
    return mesh


def _centre_cloud(mesh, plane_axis, n, seed=0):
    """Surface samples moved inward by the wire radius (half the mesh's thickness across its plane)."""
    radius = np.ptp(mesh.vertices[:, plane_axis]) / 2
    pts, face = trimesh.sample.sample_surface_even(mesh, n, seed=seed)
    return pts - radius * mesh.face_normals[face], radius


def hook_centreline(spacing=1.0):
    """Hook centreline from the shaft's top end to the tip, in the hook frame [mm]."""
    mesh = _mesh("hook", HOOK_YAW_DEG)
    c, radius = _centre_cloud(mesh, 1, 20000)
    # End caps move inward along the axis and smear across it: drop samples nearer the surface than the radius.
    dense, _ = trimesh.sample.sample_surface_even(mesh, 200000, seed=1)
    c = c[cKDTree(dense).query(c)[0] > 0.85 * radius]
    c = c[np.random.default_rng(0).choice(len(c), min(4000, len(c)), replace=False)]
    pairs = cKDTree(c).query_pairs(1.5, output_type="ndarray")
    w = np.linalg.norm(c[pairs[:, 0]] - c[pairs[:, 1]], axis=1) + 1e-9
    tree = minimum_spanning_tree(sp.coo_matrix((w, (pairs[:, 0], pairs[:, 1])), shape=(len(c), len(c))).tocsr())
    d0 = shortest_path(tree, directed=False, indices=0)
    a = int(np.argmax(np.where(np.isfinite(d0), d0, -1)))
    da, pred = shortest_path(tree, directed=False, indices=a, return_predecessors=True)
    b = int(np.argmax(np.where(np.isfinite(da), da, -1)))
    path = [b]
    while path[-1] != a:
        path.append(pred[path[-1]])
    path = np.array(path)
    s = da[path][cKDTree(c[path]).query(c)[1]]  # arc length of each sample's nearest point on the path
    nb = int(s.max() / spacing)
    bins = np.minimum((s / s.max() * nb).astype(int), nb - 1)
    line = np.array([c[bins == k].mean(0) for k in range(nb) if np.any(bins == k)])
    if line[0, 2] < line[-1, 2]:
        line = line[::-1]
    line[:, 1] = 0.0
    ends = []
    for end, inner in ((line[0], line[3]), (line[-1], line[-4])):
        t = end - inner
        ends.append(end + radius * t / np.linalg.norm(t))
    return np.vstack([ends[0], line, ends[1]]), radius


def loop_centreline(n=96):
    """Closed centreline of the eyelet's loop (without the stem), in the eyelet frame [mm]."""
    mesh = _mesh("eyelet", EYELET_YAW_DEG)
    # The loop's wire is slightly flattened in its plane (2.5 mm vs 2.76 mm across), so samples land a little
    # past the centre; the median over each angular bin is still on it.
    c, radius = _centre_cloud(mesh, 0, 40000)
    upper = c[c[:, 2] > -14]
    centre = np.array([(upper[:, 1].min() + upper[:, 1].max()) / 2, LOOP_CENTRE_Z])
    rel = c[:, 1:] - centre
    ang, rad = np.arctan2(rel[:, 1], rel[:, 0]), np.linalg.norm(rel, axis=1)
    ring = (rad > 5) & (rad < 12) & ~((np.abs(rel[:, 0]) < 2.0) & (c[:, 2] < -15.8))  # not the stem
    edges = np.linspace(-np.pi, np.pi, n + 1)
    loop = np.array([np.median(c[ring & (ang >= lo) & (ang < hi)], axis=0) for lo, hi in zip(edges[:-1], edges[1:])])
    loop[:, 0] = 0.0
    return loop, radius


def centrelines():
    """(hook centreline, loop centreline, hook wire radius, loop wire radius), mm; computed once and cached."""
    if CACHE.exists():
        with np.load(CACHE) as z:
            return z["hook"], z["loop"], float(z["hook_radius"]), float(z["loop_radius"])
    hook, hook_radius = hook_centreline()
    loop, loop_radius = loop_centreline()
    np.savez(CACHE, hook=hook, loop=loop, hook_radius=hook_radius, loop_radius=loop_radius)
    return hook, loop, hook_radius, loop_radius


def resample(line, n):
    """n points evenly spaced by arc length along an open polyline."""
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(line, axis=0), axis=1))])
    t = np.linspace(0, s[-1], n)
    return np.stack([np.interp(t, s, line[:, k]) for k in range(3)], axis=1)


def linking(a, loop):
    """Gauss linking integral of open polylines a (..., n, 3) with the closed polyline loop (m, 3).

    Sums the exact segment-pair terms (solid angle of the quadrilateral of their four ends, Klenin &
    Langowski, Biopolymers 2000), so it is exact for the polylines however coarse they are.
    """
    b = np.vstack([loop, loop[:1]])
    p1, p2 = a[..., :-1, None, :], a[..., 1:, None, :]
    p3, p4 = b[:-1], b[1:]
    r13, r14, r23, r24 = p3 - p1, p4 - p1, p3 - p2, p4 - p2

    def unit(v):
        return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)

    n1, n2, n3, n4 = unit(np.cross(r13, r14)), unit(np.cross(r14, r24)), unit(np.cross(r24, r23)), unit(np.cross(r23, r13))
    omega = sum(np.arcsin(np.clip(np.sum(x * y, -1), -1, 1)) for x, y in ((n1, n2), (n2, n3), (n3, n4), (n4, n1)))
    sign = np.sign(np.sum(np.cross(p4 - p3, p2 - p1) * r13, -1))
    return (omega * sign).sum((-1, -2)) / (4 * np.pi)


@njit(cache=True, inline="always")
def _cross_unit(a, b):
    c0, c1, c2 = a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]
    n = max(np.sqrt(c0 * c0 + c1 * c1 + c2 * c2), 1e-12)
    return (c0 / n, c1 / n, c2 / n)


@njit(cache=True, inline="always")
def _asin_dot(a, b):
    return np.arcsin(min(1.0, max(-1.0, a[0] * b[0] + a[1] * b[1] + a[2] * b[2])))


@njit(cache=True)
def linking_many(a, loop):
    """linking() for N open polylines a (N, n, 3) with the closed polyline loop (m, 3), compiled."""
    out = np.zeros(a.shape[0])
    m = loop.shape[0]
    for k in range(a.shape[0]):
        total = 0.0
        for i in range(a.shape[1] - 1):
            p1, p2 = a[k, i], a[k, i + 1]
            for j in range(m):
                p3, p4 = loop[j], loop[(j + 1) % m]
                r13 = (p3[0] - p1[0], p3[1] - p1[1], p3[2] - p1[2])
                r14 = (p4[0] - p1[0], p4[1] - p1[1], p4[2] - p1[2])
                r23 = (p3[0] - p2[0], p3[1] - p2[1], p3[2] - p2[2])
                r24 = (p4[0] - p2[0], p4[1] - p2[1], p4[2] - p2[2])
                n1, n2, n3, n4 = _cross_unit(r13, r14), _cross_unit(r14, r24), _cross_unit(r24, r23), _cross_unit(r23, r13)
                om = _asin_dot(n1, n2) + _asin_dot(n2, n3) + _asin_dot(n3, n4) + _asin_dot(n4, n1)
                e = (p4[0] - p3[0], p4[1] - p3[1], p4[2] - p3[2])
                f = (p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2])
                s = ((e[1] * f[2] - e[2] * f[1]) * r13[0] + (e[2] * f[0] - e[0] * f[2]) * r13[1]
                     + (e[0] * f[1] - e[1] * f[0]) * r13[2])
                total += om * np.sign(s)
        out[k] = total / (4 * np.pi)
    return out
