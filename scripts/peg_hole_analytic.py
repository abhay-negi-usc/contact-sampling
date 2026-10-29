"""Analytical contact geometry of the cylindrical peg and hole, independent of MuJoCo.

Both parts are solids of revolution built from the profiles in peg_hole_model.py (fillets and bolt
holes left out, as in the MuJoCo model): the hole about its axis (hole y), the peg about its own
axis. The signed distance from a point to either solid is therefore the exact 2D distance to its
profile, a union of rectangles in (radius, height).

Between these two solids the closest points (and so every contact) always involve one of four
circles: the peg's tip edge or shoulder edge against the hole, or the hole's inner or outer rim edge
against the peg. Flat-on-flat and side-on-wall contacts happen only at zero tilt, where the edges
touch as well. The separation is thus the smallest value of four one-dimensional circle searches,
and each local minimum at zero is one contact point, labelled by the peg and hole features it lies
on (PEG_FEATURES x HOLE_FEATURES). One-point, two-point, ... contact modes follow from the count.

Poses use the reduced coordinates of plot_contact_samples_cylindrical.py: tip radial offset rho [mm],
radial and tangential tilt [deg] (projected angles of the tip-ward axis, + tip outward), insertion
depth [mm]; the tip sits on the hole's +x side.
"""

import numba as nb
import numpy as np

from peg_hole_model import BORE_FLOOR_Y, BORE_TOP_Y, HOLE_PLATE, HOLE_RINGS, PEG_CYLINDERS, PEG_TIP_Y

# Profiles as rectangles (r0, r1, y0, y1): hole in (radius, hole y), peg in (radius, lambda), lambda
# measured from the tip face back up the peg.
HOLE_BOXES = np.array([[0.0, HOLE_PLATE[2], HOLE_PLATE[0], HOLE_PLATE[1]]]
                      + [[r_in, r_out, y0, y1] for y0, y1, r_in, r_out in HOLE_RINGS])
PEG_BOXES = np.array([[0.0, r, PEG_TIP_Y - y1, PEG_TIP_Y - y0] for y0, y1, r in PEG_CYLINDERS])
TIP_R, TIP_LEN = PEG_CYLINDERS[-1][2], PEG_TIP_Y - PEG_CYLINDERS[-1][0]
SHOULDER_R = PEG_CYLINDERS[-2][2]
BORE_R, TOP_R = HOLE_RINGS[-1][2], HOLE_RINGS[-1][3]
Y_TOP, Y_FLOOR = BORE_TOP_Y, BORE_FLOOR_Y
STEP_Y, STEP_R = HOLE_RINGS[1][1], HOLE_RINGS[1][3]  # the step outside the tube, and its outer edge
BOSS_Y0 = HOLE_RINGS[0][0]

# The four contact circles: (kind, axial position or height, radius). Kind 0 is a peg circle at
# lambda against the hole, kind 1 a hole circle at height y against the peg.
CIRCLES = np.array([[0, 0.0, TIP_R], [0, TIP_LEN, SHOULDER_R], [1, Y_TOP, BORE_R], [1, Y_TOP, TOP_R],
                    [1, STEP_Y, STEP_R]])
PEG_FEATURES = ["tip_face", "tip_edge", "tip_side", "shoulder_face", "shoulder_edge", "shaft_side", "other"]
HOLE_FEATURES = ["top", "rim_in", "rim_out", "wall", "floor", "outer_wall", "step", "step_edge", "boss_wall", "other"]
RIM_FEATURE = np.array([-1, -1, 1, 2, 7])  # hole feature of each hole circle (inner rim, outer rim, step edge)
N_HOLE_FEATURES = len(HOLE_FEATURES)
N_CIRCLE = 256  # samples per circle when finding contact points
N_CIRCLE_SEP = 96  # coarser, for the separation alone (its minimum is then refined)
MAX_ELEMENTS = 8


@nb.njit(cache=True)
def _sd_boxes(x, y, boxes):
    best = np.inf
    for k in range(boxes.shape[0]):
        hx = 0.5 * (boxes[k, 1] - boxes[k, 0])
        hy = 0.5 * (boxes[k, 3] - boxes[k, 2])
        dx = abs(x - boxes[k, 0] - hx) - hx
        dy = abs(y - boxes[k, 2] - hy) - hy
        ox, oy = max(dx, 0.0), max(dy, 0.0)
        v = np.sqrt(ox * ox + oy * oy) + min(max(dx, dy), 0.0)
        best = min(best, v)
    return best


@nb.njit(cache=True)
def _basis(u):
    t = np.array([1.0, 0.0, 0.0]) if abs(u[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = t - (t @ u) * u
    e1 /= np.sqrt(e1 @ e1)
    e2 = np.cross(u, e1)
    return e1, e2


@nb.njit(cache=True)
def _circle_point(k, s, c, u, e1, e2):
    """Point of circle k at parameter s, in the hole frame."""
    kind, h, rad = CIRCLES_NB[k, 0], CIRCLES_NB[k, 1], CIRCLES_NB[k, 2]
    cs, sn = np.cos(s), np.sin(s)
    if kind == 0:
        return c + h * u + rad * (cs * e1 + sn * e2)
    return np.array([rad * cs, h, rad * sn])


@nb.njit(cache=True)
def _circle_value(k, s, c, u, e1, e2):
    p = _circle_point(k, s, c, u, e1, e2)
    if CIRCLES_NB[k, 0] == 0:
        return _sd_boxes(np.sqrt(p[0] * p[0] + p[2] * p[2]), p[1], HOLE_BOXES_NB)
    v = p - c
    lam = v @ u
    w = v - lam * u
    return _sd_boxes(np.sqrt(w @ w), lam, PEG_BOXES_NB)


@nb.njit(cache=True)
def _golden(k, lo, hi, c, u, e1, e2, iters=40):
    g = 0.6180339887498949
    a, b = lo, hi
    x1, x2 = b - g * (b - a), a + g * (b - a)
    f1, f2 = _circle_value(k, x1, c, u, e1, e2), _circle_value(k, x2, c, u, e1, e2)
    for _ in range(iters):
        if f1 < f2:
            b, x2, f2 = x2, x1, f1
            x1 = b - g * (b - a)
            f1 = _circle_value(k, x1, c, u, e1, e2)
        else:
            a, x1, f1 = x1, x2, f2
            x2 = a + g * (b - a)
            f2 = _circle_value(k, x2, c, u, e1, e2)
    return (x1, f1) if f1 < f2 else (x2, f2)


@nb.njit(cache=True)
def _minima(k, c, u, e1, e2, thresh, out, n_out, n=N_CIRCLE):
    """Append the local minima of circle k below thresh to out[n_out:] as (k, s, value); return count.

    A plateau (a face lying flat on a face) counts once. Also returns the circle's global minimum.
    """
    vals = np.empty(n)
    for i in range(n):
        vals[i] = _circle_value(k, 2 * np.pi * i / n, c, u, e1, e2)
    gmin = np.inf
    best_i = 0
    for i in range(n):
        if vals[i] < gmin:
            gmin, best_i = vals[i], i
    _, gmin = _golden(k, 2 * np.pi * (best_i - 1) / n, 2 * np.pi * (best_i + 1) / n, c, u, e1, e2)
    # Each run of samples near the threshold is one contact region, so a face lying on a face is one
    # contact, not many; a sharp minimum (edge across edge) can sit up to one arc step below its samples.
    near = thresh + 2 * np.pi / n * CIRCLES_NB[k, 2]
    low = vals < near
    if not low.any():
        return n_out, gmin
    i0 = 0
    while i0 < n and low[i0]:
        i0 += 1
    if i0 == n:  # the whole circle is one run
        i0 = 0
    i = 0
    while i < n:
        if not low[(i0 + i) % n]:
            i += 1
            continue
        best = (i0 + i) % n
        while i < n and low[(i0 + i) % n]:
            if vals[(i0 + i) % n] < vals[best]:
                best = (i0 + i) % n
            i += 1
        s, v = _golden(k, 2 * np.pi * (best - 1) / n, 2 * np.pi * (best + 1) / n, c, u, e1, e2)
        if v < thresh and n_out < out.shape[0]:
            out[n_out, 0], out[n_out, 1], out[n_out, 2] = k, s, v
            n_out += 1
    return n_out, gmin


@nb.njit(cache=True)
def _separation(c, u):
    e1, e2 = _basis(u)
    out = np.empty((0, 3))
    sep = np.inf
    for k in range(CIRCLES_NB.shape[0]):
        _, g = _minima(k, c, u, e1, e2, -np.inf, out, 0, N_CIRCLE_SEP)
        sep = min(sep, g)
    return sep


@nb.njit(cache=True)
def _elements(c, u, thresh):
    e1, e2 = _basis(u)
    out = np.empty((MAX_ELEMENTS, 3))
    n = 0
    sep = np.inf
    for k in range(CIRCLES_NB.shape[0]):
        n, g = _minima(k, c, u, e1, e2, thresh, out, n)
        sep = min(sep, g)
    return out[:n], sep


@nb.njit(cache=True)
def _classify_hole(r, y, tol):
    if np.hypot(r - BORE_R, y - Y_TOP) < tol:
        return 1
    if np.hypot(r - TOP_R, y - Y_TOP) < tol:
        return 2
    if abs(y - Y_TOP) < tol and BORE_R <= r <= TOP_R:
        return 0
    if abs(r - BORE_R) < tol and Y_FLOOR - tol <= y <= Y_TOP:
        return 3
    if abs(y - Y_FLOOR) < tol and r <= BORE_R:
        return 4
    if np.hypot(r - STEP_R, y - STEP_Y) < tol:
        return 7
    if abs(r - TOP_R) < tol and STEP_Y - tol <= y <= Y_TOP:
        return 5
    if abs(y - STEP_Y) < tol and TOP_R <= r <= STEP_R:
        return 6
    if abs(r - STEP_R) < tol and BOSS_Y0 - tol <= y <= STEP_Y:
        return 8
    return 9


@nb.njit(cache=True)
def _classify_peg(w, lam, tol):
    if np.hypot(w - TIP_R, lam) < tol:
        return 1
    if abs(lam) < tol and w <= TIP_R:
        return 0
    if abs(w - TIP_R) < tol and 0 <= lam <= TIP_LEN:
        return 2
    if np.hypot(w - SHOULDER_R, lam - TIP_LEN) < tol:
        return 4
    if abs(lam - TIP_LEN) < tol and TIP_R <= w <= SHOULDER_R:
        return 3
    if abs(w - SHOULDER_R) < tol and lam >= TIP_LEN:
        return 5
    return 6


@nb.njit(cache=True)
def _contacts(c, u, tol, tol_bore):
    """Contact points: rows (peg feature, hole feature, x, y, z), and the separation.

    A contact counts within tol of touching, or within tol_bore on the inner rim and bore wall (the
    MuJoCo bore is a polygon). A point on a peg circle touching the hole's inner corner (wall and
    floor) yields both contacts; an edge-on-edge contact found from both circles is kept once.
    """
    el, sep = _elements(c, u, max(tol, tol_bore))
    e1, e2 = _basis(u)
    rows = np.empty((2 * MAX_ELEMENTS, 5))
    m = 0
    for i in range(el.shape[0]):
        k = int(el[i, 0])
        p = _circle_point(k, el[i, 1], c, u, e1, e2)
        if CIRCLES_NB[k, 0] == 0:
            pf = 1 if k == 0 else 4
            r, y = np.sqrt(p[0] * p[0] + p[2] * p[2]), p[1]
            near_wall = abs(r - BORE_R) < tol and Y_FLOOR - tol <= y <= Y_TOP - tol
            near_floor = abs(y - Y_FLOOR) < tol and r <= BORE_R + tol
            if near_wall and near_floor:
                for hf in (3, 4):
                    rows[m, 0], rows[m, 1], rows[m, 2:] = pf, hf, p
                    m += 1
                continue
            hf = _classify_hole(r, y, max(tol, tol_bore))
            if el[i, 2] > (tol_bore if hf == 1 or hf == 3 else tol):
                continue
        else:
            hf = RIM_FEATURE_NB[k]
            v = p - c
            lam = v @ u
            w = v - lam * u
            pf = _classify_peg(np.sqrt(w @ w), lam, max(tol, tol_bore))
            if el[i, 2] > (tol_bore if hf == 1 else tol):
                continue
            dup = False
            for j in range(m):  # edge on edge, already found from the peg circle
                dx = rows[j, 2:] - p
                if rows[j, 0] == pf and rows[j, 1] == hf and np.sqrt(dx @ dx) < 1.0:
                    dup = True
            if dup:
                continue
        rows[m, 0], rows[m, 1], rows[m, 2:] = pf, hf, p
        m += 1
    return rows[:m], sep


CIRCLES_NB, HOLE_BOXES_NB, PEG_BOXES_NB, RIM_FEATURE_NB = CIRCLES, HOLE_BOXES, PEG_BOXES, RIM_FEATURE


# --- poses ---------------------------------------------------------------------------------------

def pose_from_reduced(q):
    """(rho, tilt_r, tilt_t, depth) -> tip-face centre c [mm] and unit axis u up the peg, hole frame."""
    q = np.atleast_2d(q)
    c = np.stack([q[:, 0], Y_TOP - q[:, 3], np.zeros(len(q))], axis=1)
    u = np.stack([-np.tan(np.radians(q[:, 1])), np.ones(len(q)), -np.tan(np.radians(q[:, 2]))], axis=1)
    return c, u / np.linalg.norm(u, axis=1, keepdims=True)


def reduced_from_pose(pose):
    """Sampler poses (peg in the hole frame, m, wxyz) -> reduced coords (rho, tilt_r, |tilt_t|, depth)."""
    from scipy.spatial.transform import Rotation
    axis = Rotation.from_quat(pose[:, 3:], scalar_first=True).apply([0.0, 1.0, 0.0])  # towards the tip
    tip = pose[:, :3] * 1e3 + axis * PEG_TIP_Y
    rho = np.hypot(tip[:, 0], tip[:, 2])
    phi = np.arctan2(tip[:, 2], tip[:, 0])
    radial = axis[:, 0] * np.cos(phi) + axis[:, 2] * np.sin(phi)
    tangential = -axis[:, 0] * np.sin(phi) + axis[:, 2] * np.cos(phi)
    down = -axis[:, 1]
    return np.stack([rho, np.degrees(np.arctan2(radial, down)), np.degrees(np.abs(np.arctan2(tangential, down))),
                     Y_TOP - tip[:, 1]], axis=1)


def tip_and_axis(pose):
    """Sampler poses -> tip-face centre c [mm] and unit axis u up the peg (hole frame, any azimuth)."""
    from scipy.spatial.transform import Rotation
    a = Rotation.from_quat(pose[:, 3:], scalar_first=True).apply([0.0, 1.0, 0.0])
    return pose[:, :3] * 1e3 + a * PEG_TIP_Y, -a


# --- batch evaluation ----------------------------------------------------------------------------

@nb.njit(parallel=True, cache=True)
def separation(c, u):
    out = np.empty(len(c))
    for i in nb.prange(len(c)):
        out[i] = _separation(c[i], u[i])
    return out


@nb.njit(parallel=True, cache=True)
def contact_modes(c, u, tol, tol_bore):
    """Per pose: separation, number of contacts and up to 4 sorted (peg, hole) feature pair ids."""
    sep = np.empty(len(c))
    count = np.zeros(len(c), np.int64)
    pairs = np.full((len(c), 4), -1, np.int64)
    for i in nb.prange(len(c)):
        rows, s = _contacts(c[i], u[i], tol, tol_bore)
        sep[i] = s
        count[i] = rows.shape[0]
        ids = np.sort(rows[:, 0] * N_HOLE_FEATURES + rows[:, 1]).astype(np.int64)
        for j in range(min(4, len(ids))):
            pairs[i, j] = ids[j]
    return sep, count, pairs


def mode_name(pair_ids):
    names = [f"{PEG_FEATURES[p // N_HOLE_FEATURES]}/{HOLE_FEATURES[p % N_HOLE_FEATURES]}"
             for p in pair_ids if p >= 0]
    return " + ".join(names) if names else "none"


@nb.njit(parallel=True, cache=True)
def first_contact_depth(q, d_start, d_max, tol):
    """Lower the peg from depth d_start along the bore axis until it touches; depth (nan if none).

    Sphere tracing: a rigid move by less than the separation cannot cause penetration.
    """
    out = np.full(len(q), np.nan)
    for i in nb.prange(len(q)):
        t_r, t_t = np.radians(q[i, 1]), np.radians(q[i, 2])
        u = np.array([-np.tan(t_r), 1.0, -np.tan(t_t)])
        u /= np.sqrt(u @ u)
        d = d_start
        for _ in range(300):
            c = np.array([q[i, 0], Y_TOP - d, 0.0])
            s = _separation(c, u)
            if s < tol:
                out[i] = d
                break
            d += max(s, 0.25 * tol)
            if d > d_max:
                break
    return out


@nb.njit(parallel=True, cache=True)
def lateral_contacts(q, rho_max, n_grid, tol, out):
    """Contacts met moving the tip radially at fixed tilt and depth: rho of each sign change of the
    separation along [0, rho_max] (n_grid steps), refined by bisection. out[i] gets up to 4 values."""
    for i in nb.prange(len(q)):
        t_r, t_t = np.radians(q[i, 1]), np.radians(q[i, 2])
        u = np.array([-np.tan(t_r), 1.0, -np.tan(t_t)])
        u /= np.sqrt(u @ u)
        y = Y_TOP - q[i, 3]
        prev = _separation(np.array([0.0, y, 0.0]), u)
        n = 0
        for j in range(1, n_grid + 1):
            rho = rho_max * j / n_grid
            cur = _separation(np.array([rho, y, 0.0]), u)
            if (prev > 0) != (cur > 0) and n < out.shape[1]:
                lo, hi = rho - rho_max / n_grid, rho
                flo = prev
                for _ in range(50):
                    mid = 0.5 * (lo + hi)
                    fm = _separation(np.array([mid, y, 0.0]), u)
                    if (fm > 0) == (flo > 0):
                        lo, flo = mid, fm
                    else:
                        hi = mid
                    if hi - lo < tol:
                        break
                out[i, n] = 0.5 * (lo + hi)
                n += 1
            prev = cur


# --- multi-contact states --------------------------------------------------------------------------

@nb.njit(cache=True)
def _pose_q(q):
    t_r, t_t = np.radians(q[1]), np.radians(q[2])
    u = np.array([-np.tan(t_r), 1.0, -np.tan(t_t)])
    return np.array([q[0], Y_TOP - q[3], 0.0]), u / np.sqrt(u @ u)


@nb.njit(cache=True)
def _element_value(k, s0, q):
    """Local minimum of circle k near parameter s0 at reduced pose q: (s, value)."""
    c, u = _pose_q(q)
    e1, e2 = _basis(u)
    hw = 6 * np.pi / N_CIRCLE
    return _golden(k, s0 - hw, s0 + hw, c, u, e1, e2)


@nb.njit(cache=True)
def _close_elements(q, ks, ss, tol, iters=30, h=1e-6):
    """Newton (minimum-norm) on the reduced pose q until the given circle minima all touch."""
    m = len(ks)
    phi = np.empty(m)
    J = np.empty((m, 4))
    for _ in range(iters):
        for j in range(m):
            ss[j], phi[j] = _element_value(ks[j], ss[j], q)
        if np.abs(phi).max() < tol:
            return True
        for a in range(4):
            dq = q.copy()
            dq[a] += h
            for j in range(m):
                J[j, a] = (_element_value(ks[j], ss[j], dq)[1] - phi[j]) / h
        q -= J.T @ np.linalg.solve(J @ J.T + 1e-10 * np.eye(m), phi)
    return False


@nb.njit(parallel=True, cache=True)
def add_contact(q, near, tol, out_q, out_ok):
    """From contact states q, close the nearest other gap below `near` while keeping the existing
    contacts closed: the next stratum down (one more contact), if it is reachable this way."""
    for i in nb.prange(len(q)):
        c, u = _pose_q(q[i])
        el, _ = _elements(c, u, near)
        touching = el[:, 2] < tol
        n_t = touching.sum()
        if n_t == 0 or n_t == len(el):
            continue
        # the closest open gap joins the touching contacts
        j_best, v_best = -1, np.inf
        for j in range(len(el)):
            if not touching[j] and el[j, 2] < v_best:
                j_best, v_best = j, el[j, 2]
        # Touching contacts, each once: an edge on an edge shows on both of its circles.
        e1, e2 = _basis(u)
        order = np.empty(n_t + 1, np.int64)
        order[:n_t] = np.nonzero(touching)[0]
        order[n_t] = j_best
        ks = np.empty(n_t + 1, np.int64)
        ss = np.empty(n_t + 1)
        pts = np.empty((n_t + 1, 3))
        m = 0
        for j in order:
            p = _circle_point(int(el[j, 0]), el[j, 1], c, u, e1, e2)
            dup = False
            for l in range(m):
                if np.sqrt(((pts[l] - p) ** 2).sum()) < 0.5:
                    dup = True
            if dup:
                if j == j_best:  # the open gap is a contact we already hold
                    m = -1
                    break
                continue
            ks[m], ss[m], pts[m] = int(el[j, 0]), el[j, 1], p
            m += 1
        if m < 2:
            continue
        ks, ss = ks[:m], ss[:m]
        qq = q[i].copy()
        if _close_elements(qq, ks, ss, tol):
            c2, u2 = _pose_q(qq)
            if _separation(c2, u2) > -10 * tol:
                out_q[i] = qq
                out_ok[i] = True


@nb.njit(parallel=True, cache=True)
def snap(q, near, near_bore, tol):
    """Move reduced poses onto the analytical contact manifold: Newton (minimum norm) closing every
    active contact, each once: within `near` of touching, or `near_bore` on the inner rim and bore
    wall. Returns the snapped poses and whether they converged.

    Used to take sampler poses, which lie on MuJoCo's polygonal bore (up to 16 um off the true
    cylinder), onto the smooth geometry.
    """
    out = q.copy()
    ok = np.zeros(len(q), np.bool_)
    for i in nb.prange(len(q)):
        c, u = _pose_q(q[i])
        el, _ = _elements(c, u, max(near, near_bore))
        if len(el) == 0:
            continue
        e1, e2 = _basis(u)
        ks = np.empty(len(el), np.int64)
        ss = np.empty(len(el))
        pts = np.empty((len(el), 3))
        m = 0
        for j in range(len(el)):
            k = int(el[j, 0])
            p = _circle_point(k, el[j, 1], c, u, e1, e2)
            if CIRCLES_NB[k, 0] == 0:
                hf = _classify_hole(np.sqrt(p[0] * p[0] + p[2] * p[2]), p[1], near_bore)
            else:
                hf = RIM_FEATURE_NB[k]
            if el[j, 2] > (near_bore if hf == 1 or hf == 3 else near):
                continue
            dup = False
            for l in range(m):
                if np.sqrt(((pts[l] - p) ** 2).sum()) < 0.5:
                    dup = True
            if not dup:
                ks[m], ss[m], pts[m] = int(el[j, 0]), el[j, 1], p
                m += 1
        qq = q[i].copy()
        if _close_elements(qq, ks[:m], ss[:m], tol):
            out[i] = qq
            ok[i] = True
    return out, ok


# --- contact details for the sampler ---------------------------------------------------------------

@nb.njit(cache=True)
def _sd_hole_point(p):
    return _sd_boxes(np.sqrt(p[0] * p[0] + p[2] * p[2]), p[1], HOLE_BOXES_NB)


@nb.njit(cache=True)
def _sd_peg_point(p, c, u):
    v = p - c
    lam = v @ u
    w = v - lam * u
    return _sd_boxes(np.sqrt(w @ w), lam, PEG_BOXES_NB)


@nb.njit(cache=True)
def contact_details(c, u, thresh, h=1e-6):
    """Every contact within thresh of touching at the pose (tip-face centre c [mm], axis u up the peg):
    rows (circle, gap [mm], contact point x, y, z [mm], normal x, y, z), the normal pointing from the
    hole into the peg (the gradient of the other body's signed distance at the contact point)."""
    el, _ = _elements(c, u, thresh)
    e1, e2 = _basis(u)
    out = np.empty((len(el), 8))
    for i in range(len(el)):
        k = int(el[i, 0])
        p = _circle_point(k, el[i, 1], c, u, e1, e2)
        g = np.zeros(3)
        for a in range(3):
            dp = np.zeros(3)
            dp[a] = h
            if CIRCLES_NB[k, 0] == 0:  # a peg circle against the hole: away from the hole
                g[a] = (_sd_hole_point(p + dp) - _sd_hole_point(p - dp)) / (2 * h)
            else:  # a hole circle against the peg: towards the peg
                g[a] = -(_sd_peg_point(p + dp, c, u) - _sd_peg_point(p - dp, c, u)) / (2 * h)
        g /= max(np.sqrt(g @ g), 1e-300)
        out[i, 0], out[i, 1] = k, el[i, 2]
        out[i, 2:5] = p
        out[i, 5:8] = g
    return out


# --- contact-mode families ------------------------------------------------------------------------

FAMILIES = ["in the bore", "edge on inner rim", "face on mouth", "edge on mouth", "beside the tube", "on the step"]
REGION_OF_FAMILY = {"in the bore": "inside", "edge on inner rim": "mouth", "face on mouth": "mouth",
                    "edge on mouth": "mouth", "beside the tube": "outside", "on the step": "outside"}


def family_of_mode(mode):
    """Family of an analytical contact mode (a mode_name string): which part of the hole the peg is on."""
    parts = [p.split("/") for p in mode.split(" + ")] if mode != "none" else []
    holes = {h for _, h in parts}
    if holes & {"step", "step_edge", "boss_wall"}:
        return "on the step"
    if "outer_wall" in holes or ["tip_side", "rim_out"] in parts:
        return "beside the tube"
    in_bore = [p for p in parts if p[1] in ("wall", "floor") or p[0] in ("tip_side", "shoulder_face", "shoulder_edge")]
    rim_in = [p for p in parts if p == ["tip_edge", "rim_in"]]
    if rim_in and len(rim_in) + len(in_bore) == len(parts) and len(in_bore) <= 1:
        return "edge on inner rim"  # the tip edge crossing into the bore, alone or with one contact inside
    if in_bore:
        return "in the bore"
    return "face on mouth" if any(p[0] == "tip_face" for p in parts) else "edge on mouth"
