#!/usr/bin/env python3
"""Geometry-based pose modes of the hook on the eyelet: contact strata, flat patches and convex free cells.

Three decompositions, none of them a spectral cut of a neighbour graph:

primitives  Each wire's centreline (surface samples moved inward by the wire radius, then contracted onto
            the axis by repeated local averaging) is classified point by point from its local geometry:
            end (neighbourhood one-sided), junction (neighbourhood not a single curve), bend (curvature
            above --bend-curvature) or straight. Connected runs of one class are the primitives; a bend
            also has a side, inner (facing its centre of curvature) or outer.
strata      For each contact pose (contact_graph.npz from mode_graph_experiments.py prepare-contact), every
            hull pair within --contact-eps of the closest is a touching pair; touching pairs whose witness
            points lie within --merge-mm on both bodies are one contact. Each contact is named by the
            primitives (and sides) it touches, and the pose's stratum is the set of its contacts: one
            contact is a 5-D face of the contact set, two a 4-D crease where two faces meet, and so on.
            Modes are connected pieces of the one-contact faces (joined through edges that slide within
            0.25 mm of contact), and two faces neighbour each other where a two-contact crease touches both.
flat        Each stratum piece is split in two (along its principal axis in the lever-10 pose metric) until
            the contact normal (the wrench direction [n, r x n / 10 mm], the gradient of separation) varies
            over it by less than --flat-deg: a nearly flat, so nearly convex, patch of the contact set.
convex      An approximate convex decomposition of the free poses: polytopes grown as in IRIS (Deits &
            Tedrake 2015) with the contact poses, which sample the boundary of the free space densely, as
            the obstacles. From a seed, the nearest remaining boundary point in the current ellipsoid's
            metric gives a separating hyperplane, until none is left inside; the ellipsoid is then refit to
            the free samples inside, and the process repeated. Seeds are the free samples not yet covered,
            the most clear first. Each polytope is checked by sampling poses inside it and measuring their
            separation; free samples are labelled by the first polytope containing them.

    python scripts/hook_contact_strata.py primitives --out-dir OUT
    python scripts/hook_contact_strata.py strata EXPERIMENT_DIR --out-dir OUT
    python scripts/hook_contact_strata.py convex EXPERIMENT_DIR --out-dir OUT
"""

import argparse
import json
import textwrap
import multiprocessing as mp
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import scipy.sparse as sp
import trimesh
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from hook_eyelet_model import ASSET_DIR, EYELET_YAW_DEG, HOOK_YAW_DEG, INCH, Separation, build_model
from hook_topology import loop_mid_y, loop_side, threading_name

LEVER = 10.0  # mm per radian in the pose metric
SCALE = np.array([1, 1, 1] + [LEVER * np.pi / 180] * 3)
CLASSES = ["straight", "bend", "end", "junction"]
SURFACE, INK, INK_2 = "#fcfcfb", "#0b0b0b", "#52514e"
HUES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]


# --- CAD faces ----------------------------------------------------------------------------------------

def load_part(name):
    yaw = {"hook": HOOK_YAW_DEG, "eyelet": EYELET_YAW_DEG}[name]
    mesh = trimesh.load(ASSET_DIR / f"{name}.stl")
    mesh.apply_scale(INCH * 1e3)
    mesh.apply_transform(trimesh.transformations.rotation_matrix(np.deg2rad(yaw), [0, 0, 1]))
    return mesh


THIN = {"hook": 1, "eyelet": 0}  # the axis across each flat wire; its axis lies in that coordinate's zero plane


def cad_faces(name, args):
    """Segment the STL's triangles into CAD-like faces from the wire geometry.

    Each triangle's axis point is its centroid moved inward by the local wire radius (the radius from
    triangles facing across the wire's mid-plane, where centroid / normal along the thin axis gives it
    directly). Where the axis points nearby form a line, the triangle lies on the swept wire and is a
    cylinder face (axis curvature below --bend-curvature) or a torus face; otherwise it is an end cap
    (normal along the axis) or a blend (where wires merge, or a taper). Connected triangles of one class
    form a face; faces smaller than --min-face mm^2 join the neighbour they share most edges with. Torus
    triangles also get a side: inner (facing the bend's centre), outer, or flank.
    """
    mesh = load_part(name)
    thin = THIN[name]
    plane = [i for i in range(3) if i != thin]
    # uniform surface samples: long thin STL strips have all their centroids at one place
    S, tri = trimesh.sample.sample_surface_even(mesh, 40000, seed=1)
    N = mesh.face_normals[tri]
    across = np.abs(N[:, thin]) > 0.6
    r_across = S[across, thin] / N[across, thin]
    _, nb = cKDTree(S[across][:, plane]).query(S[:, plane], k=25)
    radius = np.median(r_across[nb], axis=1)
    axis = S - radius[:, None] * N
    lin, tangent = np.zeros(len(S)), np.zeros((len(S), 3))
    for i, nbr in enumerate(cKDTree(axis).query_ball_point(axis, 1.5)):
        X = axis[nbr] - axis[nbr].mean(0)
        w, v = np.linalg.eigh(X.T @ X)
        lin[i], tangent[i] = w[2] / max(w.sum(), 1e-12), v[:, 2]
    linear = lin > 0.9
    ltree = cKDTree(axis[linear])
    kappa, toward = np.zeros(len(S)), np.zeros((len(S), 3))
    for i, nbr in enumerate(ltree.query_ball_point(axis, args.curve_window)):
        if len(nbr) < 6:
            continue
        X = axis[linear][nbr] - axis[i]
        t = tangent[linear][nbr[np.argmin(np.linalg.norm(X, axis=1))]]  # the wire's own direction here
        tangent[i] = t
        sc = X @ t
        a = np.linalg.lstsq(np.c_[sc ** 2, sc, np.ones(len(sc))], X - np.outer(sc, t), rcond=None)[0][0]
        kappa[i] = 2 * np.linalg.norm(a)
        toward[i] = a / (np.linalg.norm(a) + 1e-12)
    cap = np.abs(np.einsum("ij,ij->i", N, tangent)) > 0.7
    scls = np.where(cap, 2, np.where(linear, np.where(kappa > args.bend_curvature, 1, 0), 3))
    # each triangle takes its samples' majority class (unsampled ones their nearest sample's)
    P = mesh.triangles_center
    votes = sp.coo_matrix((np.ones(len(S)), (tri, scls)), shape=(len(P), 4)).toarray()
    cls = np.where(votes.sum(1) > 0, votes.argmax(1), scls[cKDTree(S).query(P)[1]])
    # a straight run takes the bend direction of the nearest bend (within 15 mm), so its sides are named
    # consistently with the bends it joins: inner faces the centre of the J or loop
    bent = kappa > args.bend_curvature
    dist, j = cKDTree(axis[bent]).query(axis)
    toward = np.where(bent[:, None], toward, np.where((dist < 15)[:, None], toward[bent][j], 0.0))
    nearest = cKDTree(S).query(P)[1]
    radius, kappa, toward = radius[nearest], kappa[nearest], toward[nearest]
    N = mesh.face_normals
    # majority smoothing over the triangle adjacency, then connected faces of one class
    adj = sp.coo_matrix((np.ones(2 * len(mesh.face_adjacency)),
                         (mesh.face_adjacency.ravel(), mesh.face_adjacency[:, ::-1].ravel())),
                        shape=(len(P), len(P))).tocsr()
    for _ in range(3):
        votes = adj @ sp.csr_matrix((np.ones(len(P)), (np.arange(len(P)), cls)), shape=(len(P), 4))
        votes = votes.toarray() + np.eye(4)[cls] * 1.5
        cls = votes.argmax(1)
    fa = mesh.face_adjacency
    same = cls[fa[:, 0]] == cls[fa[:, 1]]
    A = sp.coo_matrix((np.ones(same.sum()), (fa[same, 0], fa[same, 1])), shape=(len(P), len(P)))
    _, face = connected_components(A, directed=False)
    area = np.bincount(face, mesh.area_faces)
    for f in np.argsort(area):
        if area[f] >= args.min_face or area[f] == 0:
            continue
        mine = np.isin(fa, np.flatnonzero(face == f))
        edge = mine[:, 0] ^ mine[:, 1]
        other = np.where(mine[edge, 0], face[fa[edge, 1]], face[fa[edge, 0]])
        if len(other):
            to = np.bincount(other).argmax()
            face[face == f] = to
            area[to], area[f] = area[to] + area[f], 0
    _, face = np.unique(face, return_inverse=True)
    # small faces that touch each other (a tapered tip and its cap; the neck where a loop closes onto its
    # stem) become one composite face
    area = np.bincount(face, mesh.area_faces)
    small = area < args.composite_face
    ff = face[fa]
    link = small[ff[:, 0]] & small[ff[:, 1]] & (ff[:, 0] != ff[:, 1])
    _, comp = connected_components(sp.coo_matrix((np.ones(link.sum()), (ff[link, 0], ff[link, 1])),
                                                 shape=(len(area), len(area))), directed=False)
    _, face = np.unique(comp[face], return_inverse=True)
    # sides: flank across the flat wire's mid-plane (- or + along the thin axis), else inner (facing the
    # local centre of curvature) or outer
    c = np.einsum("ij,ij->i", N, toward)
    side = np.where(np.abs(N[:, thin]) > 0.7, np.where(N[:, thin] < 0, 3, 4),
                    np.where(np.abs(toward).sum(1) == 0, 0, np.where(c > 0, 1, 2)))
    desc = []
    for f in range(face.max() + 1):
        sel = face == f
        k = np.bincount(cls[sel], weights=mesh.area_faces[sel], minlength=4).argmax()
        kinds = sorted({["cylinder", "torus", "end cap", "blend"][c] for c in np.unique(cls[sel])
                        if mesh.area_faces[sel & (cls == c)].sum() > 2.0})
        desc.append({"face": f, "type": ["cylinder", "torus", "end cap", "blend"][k] if len(kinds) == 1 else
                     "composite (" + ", ".join(kinds) + ")", "triangles": int(sel.sum()),
                     "area_mm2": round(float(mesh.area_faces[sel].sum()), 1),
                     "centre_mm": np.round(np.average(P[sel], axis=0, weights=mesh.area_faces[sel]), 1).tolist(),
                     "bend_radius_mm": round(float(1 / max(np.median(kappa[sel]), 1e-6)), 1) if k == 1 else None,
                     "wire_radius_mm": round(float(np.median(radius[sel])), 2)})
    sharp = mesh.face_adjacency_edges[np.degrees(mesh.face_adjacency_angles) > args.sharp_deg]
    return {"mesh": mesh, "face": face, "cls": cls, "side": side, "desc": desc, "rims": mesh.vertices[sharp]}


SIDES = ["", "inner", "outer", "flank-", "flank+"]


class Elements:
    """CAD face, side and rim proximity of points on one part (part frame, mm), from dense surface samples."""

    def __init__(self, z, step=0.1):
        self.tree = cKDTree(z["samples"])
        self.face, self.side = z["sample_face"], z["sample_side"]
        segs = [a + np.linspace(0, 1, max(2, int(np.linalg.norm(b - a) / step) + 1))[:, None] * (b - a)
                for a, b in z["rims"]]
        self.rims = cKDTree(np.vstack(segs)) if segs else None

    def __call__(self, p, rim_mm):
        i = self.tree.query(p)[1]
        rim = self.rims.query(p)[0] < rim_mm if self.rims is not None else np.zeros(len(p), bool)
        return self.face[i], self.side[i], rim


def primitives_stage(args):
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 7), facecolor=SURFACE)
    report = {}
    for ax, name in zip(axes, ("hook", "eyelet")):
        f = cad_faces(name, args)
        mesh = f["mesh"]
        pts, tri = trimesh.sample.sample_surface_even(mesh, 300000, seed=0)
        np.savez(out / f"faces_{name}.npz", face=f["face"], cls=f["cls"], side=f["side"], rims=f["rims"],
                 samples=pts, sample_face=f["face"][tri], sample_side=f["side"][tri])
        report[name] = f["desc"]
        i, j = [k for k in range(3) if k != THIN[name]]
        P = mesh.triangles_center
        for d in f["desc"]:
            sel = f["face"] == d["face"]
            ax.scatter(P[sel, i], P[sel, j], s=1.5, color=HUES[d["face"] % len(HUES)], label=f"{d['face']}: {d['type']}")
            ax.text(*np.array(d["centre_mm"])[[i, j]], f" {d['face']}", fontsize=9, color=INK)
        inner = f["side"] == 1
        ax.scatter(P[inner, i], P[inner, j], s=0.3, color=INK, alpha=0.3, label="inner side")
        for a, b in f["rims"]:
            ax.plot([a[i], b[i]], [a[j], b[j]], color=INK, linewidth=1.2)
        ax.set_aspect("equal")
        ax.set_title(f"{name}: CAD faces from the mesh (black: rims, dark dots: inner side)", color=INK,
                     fontsize=9, loc="left")
        ax.set_xlabel(f"{'xyz'[i]} [mm]", color=INK_2, fontsize=8), ax.set_ylabel(f"{'xyz'[j]} [mm]", color=INK_2, fontsize=8)
        ax.legend(fontsize=7, frameon=False, loc="best", markerscale=4)
        for sd in ("top", "right"):
            ax.spines[sd].set_visible(False)
        print(f"{name}: {len(f['desc'])} faces, {len(f['rims'])} rim edges", flush=True)
        for d in f["desc"]:
            print(f"   {d['face']}: {d['type']:8s} {d['area_mm2']:7.1f} mm2, centre {d['centre_mm']}, wire radius "
                  f"{d['wire_radius_mm']}" + (f", bend radius {d['bend_radius_mm']}" if d["bend_radius_mm"] else ""),
                  flush=True)
    fig.tight_layout()
    fig.savefig(out / "cad_faces.png", facecolor=SURFACE, dpi=110)
    (out / "cad_faces.json").write_text(json.dumps(report, indent=2))


def halves_stage(args):
    """Figure of the eyelet's CAD faces split into left (+y), centre and right (-y) halves (faces_eyelet.npz)."""
    out = args.out_dir
    z = np.load(out / "faces_eyelet.npz")
    aliases = json.loads((out / "aliases.json").read_text())
    S, face = z["samples"], z["sample_face"]
    half = eyelet_half(S, args.middle_mm)
    shade = {1: 1.0, 0: 0.45, -1: 0.7}
    fig, ax = plt.subplots(figsize=(7, 8), facecolor=SURFACE)
    for f in np.unique(face):
        for h in (1, 0, -1):
            sel = (face == f) & (half == h)
            if not sel.any():
                continue
            rgb = np.array(matplotlib.colors.to_rgb(HUES[f % len(HUES)]))
            ax.scatter(S[sel, 1], S[sel, 2], s=0.4, color=rgb * shade[h] + (1 - shade[h]) * 0.15)
            c = np.median(S[sel], 0)
            narrow = np.ptp(S[face == f, 1]) < 8  # a straight wire: stagger its three labels along it
            if sel.sum() > 500:
                ax.text(c[1], c[2] + (5 * h if narrow else 0), f"{aliases['eyelet'].get(str(f), f)} {HALF_NAMES[h]}",
                        fontsize=8, color=INK, ha={1: "right", 0: "center", -1: "left"}[h] if narrow else "center",
                        va="center", bbox=dict(boxstyle="round,pad=0.2", fc=SURFACE, ec="none", alpha=0.8))
    ax.axvline(loop_mid_y(), color=INK_2, linewidth=0.6, linestyle=":")
    ax.set_aspect("equal")
    ax.invert_xaxis()  # seen from the front (from -x, where the hook comes from): +y on the left
    ax.set_xlabel("y [mm], seen from the front: left = +y", color=INK_2, fontsize=8)
    ax.set_ylabel("z [mm]", color=INK_2, fontsize=8)
    ax.set_title(f"Eyelet CAD faces split into halves, seen from the front:\nleft (+y), centre (within "
                 f"{args.middle_mm:g} mm of the mid-plane), right (-y)", color=INK, fontsize=9, loc="left")
    for sd in ("top", "right"):
        ax.spines[sd].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "cad_eyelet_halves.png", facecolor=SURFACE, dpi=120)


# --- contact strata ------------------------------------------------------------------------------------

_ctx = {}
MAXC = 4  # contacts kept per pose


def _init_strata(faces_dir, eps, merge_mm, rim_mm):
    m = build_model()
    _ctx.update(m=m, d=mujoco.MjData(m), sep=Separation(m), eps=eps, merge=merge_mm, rim=rim_mm,
                hook=Elements(np.load(faces_dir / "faces_hook.npz")),
                eyelet=Elements(np.load(faces_dir / "faces_eyelet.npz")))


def strata_task(poses):
    """Per pose: up to MAXC contacts as (hook face, side, rim, eyelet face, side, rim, gap [0.01 mm]), closest first,
    and their hook-frame and eyelet-frame witness points."""
    m, d, S = _ctx["m"], _ctx["d"], _ctx["sep"]
    out = np.full((len(poses), MAXC, 7), -1, np.int32)
    wit = np.full((len(poses), MAXC, 6), np.nan)
    fromto = np.zeros(6)
    for n, pose in enumerate(poses):
        rot = Rotation.from_rotvec(pose[3:], degrees=True)
        d.qpos[:3], d.qpos[3:7] = pose[:3] * 1e-3, rot.as_quat(scalar_first=True)
        mujoco.mj_kinematics(m, d)
        lower = np.linalg.norm(d.geom_xpos[S.g1] - d.geom_xpos[S.g2], axis=1) - S.rsum
        reach = 5e-4  # contact poses sit at ~0.02 mm; libccd is exact to ~0.01 mm with this distmax
        found = []
        for k in np.flatnonzero(lower < reach):
            dist = mujoco.mj_geomDistance(m, d, S.g1[k], S.g2[k], reach, fromto)
            if dist < reach:
                found.append((dist * 1e3, fromto[:3] * 1e3, fromto[3:] * 1e3))
        if not found:
            continue
        found.sort(key=lambda f: f[0])
        touching = [f for f in found if f[0] <= found[0][0] + _ctx["eps"]]
        reps = []  # one representative (the closest pair) per distinct contact
        for f in touching:
            if all(np.linalg.norm(f[1] - r[1]) > _ctx["merge"] or np.linalg.norm(f[2] - r[2]) > _ctx["merge"]
                   for r in reps):
                reps.append(f)
        reps = reps[:MAXC]
        ph = rot.inv().apply(np.array([r[1] for r in reps]) - pose[:3])
        pe = np.array([r[2] for r in reps])
        hf, hs, hr = _ctx["hook"](ph, _ctx["rim"])
        ef, es, er = _ctx["eyelet"](pe, _ctx["rim"])
        for c in range(len(reps)):
            out[n, c] = hf[c], hs[c], hr[c], ef[c], es[c], er[c], round((reps[c][0] - reps[0][0]) * 100)
        wit[n, :len(reps)] = np.c_[ph, pe]
    return out, wit


NORMALS = {1: "-x", 2: "+x", 3: "-y", 4: "-z", 5: "+z", 6: "+y"}


def tag_of(args):
    return (f"formations_sided{args.middle_mm:g}{'_coarse' if args.coarse else ''}{'_absorb' if args.absorb else ''}"
            f"{'_normal' if args.normal_axis else ''}{'_rims' if args.rims else ''}_min{args.min_mode}{args.suffix}")


def normal_code(poses, witness):
    """Dominant eyelet-frame axis of the contact normal (eyelet pushing the hook): 1 -x (front face), 2 +x,
    3 -y, 4 -z, 5 +z (hook resting on top), 6 +y."""
    rot = Rotation.from_rotvec(poses[:, 3:], degrees=True)
    n = rot.apply(witness[:, :3]) + poses[:, :3] - witness[:, 3:]
    a = np.argmax(np.abs(n), axis=1)
    pos = n[np.arange(len(n)), a] > 0
    code = np.select([a == 0, a == 1, a == 2], [np.where(pos, 2, 1), np.where(pos, 6, 3), np.where(pos, 5, 4)])
    return np.where(np.isnan(n).any(1), -1, code)


def contact_name(c, faces):
    h = faces["hook"][c[0]]["type"].split(" (")[0]
    e = faces["eyelet"][c[3]]["type"].split(" (")[0]
    return (f"hook {c[0]}:{h}{'/' + SIDES[c[1]] if c[1] else ''}{' rim' if c[2] else ''}"
            f" ~ eyelet {c[3]}:{e}{'/' + SIDES[c[4]] if c[4] else ''}{' rim' if c[5] else ''}"
            + (f" [n {NORMALS.get(c[6], '?')}]" if len(c) > 6 else ""))


SIDE_NAMES = {1: "+y", -1: "-y", 0: "middle"}  # in formation keys' names
# the eyelet's elements are split into halves: left (+y: on the left facing the loop from the front, where the
# hook comes from), right (-y), and the centre strip within --middle-mm of the mid-plane
HALF_NAMES = {"+y": "left", "-y": "right", "middle": "centre", 1: "left", -1: "right", 0: "centre"}


def formation_name(key, faces, normal_axis):
    """Name of a formation key: its contact (and normal axis), then the side of the loop's mid-plane the
    contact lies on and the threading state, e.g. '... [n +z] {+y, tip threaded}'."""
    side, thread = key[-2], key[-1]
    return contact_name(key[:7] if normal_axis else key[:6], faces) + \
        f" {{{SIDE_NAMES[int(side)]}, {threading_name(int(thread))}}}"


MIDDLE_MM = 1.0  # contacts this close to the loop's mid-plane are in the middle


def eyelet_half(points, middle_mm=MIDDLE_MM):
    """Half of the eyelet that eyelet-frame points p [mm], shape (..., 3), lie on: +1 left (+y), -1 right
    (-y), 0 the centre strip within middle_mm of the loop's mid-plane. Every eyelet face is split this way."""
    y = np.asarray(points)[..., 1] - loop_mid_y()
    return np.where(np.abs(y) < middle_mm, 0, loop_side(points))


def contact_class(witness, thread, middle_mm=MIDDLE_MM):
    """(eyelet half, threading bitmask) per pose, from its contacts' witnesses (n, MAXC, 6; unused slots NaN):
    two poses in different classes never share a mode. The half is that of the closest contact's eyelet
    point (eyelet_half); a pose touching both the left and the right half at once (a hook symmetric about
    the mid-plane) counts as centre."""
    y = np.nan_to_num(witness[:, :, 4] - loop_mid_y())
    both = (y > middle_mm).any(1) & (y < -middle_mm).any(1)
    side = np.where(both, 0, eyelet_half(np.nan_to_num(witness[:, 0, 3:]), middle_mm))
    return np.c_[side, thread]


def strata_stage(args):
    out = args.out_dir
    g = np.load(args.experiment_dir / "contact_graph.npz")
    poses = g["poses"]
    w = args.workers
    t0 = time.time()
    chunks = np.array_split(np.arange(len(poses)), w * 16)
    with mp.get_context("fork").Pool(w, initializer=_init_strata,
                                      initargs=(out, args.contact_eps, args.merge_mm, args.rim_mm)) as pool:
        res = pool.map(strata_task, [poses[c] for c in chunks])
    contacts = np.concatenate([r[0] for r in res])
    witness = np.concatenate([r[1] for r in res])
    ncon = (contacts[:, :, 0] >= 0).sum(1)
    print(f"{len(poses):,} contact poses ({time.time() - t0:.0f} s); contacts per pose: "
          + ", ".join(f"{k}: {np.mean(ncon == k):.1%}" for k in range(MAXC + 1) if np.any(ncon == k)), flush=True)
    np.savez(out / "contacts.npz", contacts=contacts, witness=witness, ncon=ncon)


def coarse_contacts(c, aliases):
    """Contacts with each face replaced by the face naming its part (aliases.json hook_groups, eyelet_groups)
    and the wire sides dropped."""
    c = c.copy()
    for col, part in ((0, "hook"), (3, "eyelet")):
        groups = aliases[f"{part}_groups"]
        lut = np.arange(max(c[..., col].max(), max(map(int, groups))) + 2)
        for f, to in groups.items():
            lut[int(f)] = to
        c[..., col] = np.where(c[..., col] >= 0, lut[c[..., col]], c[..., col])
        c[..., col + 1] = np.where(c[..., col + 1] >= 0, 0, -1)
    return c


def audit(labels, cls):
    """Every mode must hold one side of the loop's mid-plane and one threading state: raise otherwise."""
    ok = labels >= 0
    _, c = np.unique(cls[ok], axis=0, return_inverse=True)
    per_mode = sp.coo_matrix((np.ones(ok.sum()), (labels[ok], c.ravel()))).tocsr()
    mixed = np.flatnonzero(np.diff(per_mode.indptr) > 1)
    if len(mixed):
        raise RuntimeError(f"modes {mixed.tolist()} mix sides or threading states")
    print(f"audit: all {labels.max() + 1} modes hold one side and one threading state; "
          f"{np.mean(~ok):.2%} of poses unassigned", flush=True)


def formations_stage(args):
    """Modes from contact formations: connected pieces of each formation, small pieces absorbed."""
    import mode_graph_experiments as mge

    exp, out = args.experiment_dir, args.out_dir
    g = np.load(exp / "contact_graph.npz")
    z = np.load(out / "contacts.npz")
    faces = json.loads((out / "cad_faces.json").read_text())
    poses, n = g["poses"], len(g["poses"])
    c = z["contacts"][:, :, :6].copy()
    if not args.rims:
        c[:, :, 2] = np.where(c[:, :, 2] >= 0, 0, -1)
        c[:, :, 5] = np.where(c[:, :, 5] >= 0, 0, -1)
    if args.coarse:  # parts instead of faces, no wire sides: where the contact is comes from the normal axis
        c = coarse_contacts(c, json.loads((out / "aliases.json").read_text()))
    first = c[:, 0]
    if args.normal_axis:
        first = np.c_[first, normal_code(poses, z["witness"][:, 0])]
    cls = contact_class(z["witness"], g["thread"], args.middle_mm)
    keys, form = np.unique(np.c_[first, cls], axis=0, return_inverse=True)
    form = form.ravel()
    names = [formation_name(k, faces, args.normal_axis) for k in keys]
    keep = g["in_10.0"] & (g["stray"] <= args.slide_mm)
    pairs = g["pairs"][keep]
    same = form[pairs[:, 0]] == form[pairs[:, 1]]
    A = sp.coo_matrix((np.ones(same.sum()), (pairs[same, 0], pairs[same, 1])), shape=(n, n))
    _, piece = connected_components(A, directed=False)
    # with --absorb, pieces below --min-mode poses join the neighbouring piece of the same class (side,
    # threading) they share most sliding edges with; without, every mode holds one formation
    alike = (cls[pairs[:, 0]] == cls[pairs[:, 1]]).all(1)
    W = mge.adjacency(n, pairs[alike], np.ones(alike.sum()))
    size = np.bincount(piece)
    for _ in range(20 if args.absorb else 0):
        small = size[piece] < args.min_mode
        if not small.any():
            break
        P = sp.csr_matrix((np.ones(n), (np.arange(n), piece)), shape=(n, size.size))
        links = (P.T @ W @ P).tolil()
        links.setdiag(0)
        links = links.tocsr()
        target = np.arange(size.size)
        for q in np.flatnonzero((size > 0) & (size < args.min_mode)):
            row = links.getrow(q)
            if row.nnz:
                cand = row.indices[np.argsort(-row.data)]
                big = cand[size[cand] >= args.min_mode]
                target[q] = big[0] if len(big) else cand[0]
        piece = target[piece]
        size = np.bincount(piece, minlength=size.size)
    # pieces still small are unassigned (-1), never put in a mode by pose distance alone (a near pose can lie
    # across the wire, threaded the other way); a piece holding --min-share of its class (a rarely sampled
    # side and threading state, such as the assembled hook in the middle) is a mode however small
    _, cid = np.unique(cls, axis=0, return_inverse=True)
    cid = cid.ravel()
    class_size = np.bincount(cid)
    piece_class = np.zeros(size.size, int)
    piece_class[piece] = cid
    large = (size >= args.min_mode) | (size >= args.min_share * class_size[piece_class])
    piece[~large[piece]] = -1
    labels = np.full(n, -1)
    _, labels[piece >= 0] = np.unique(piece[piece >= 0], return_inverse=True)
    k = labels.max() + 1
    audit(labels, cls)
    # describe the modes: their main formation, threading, side, size
    thread, x = g["thread"], poses[:, 0]
    modes = []
    for m in range(k):
        sel = labels == m
        fs, cnt = np.unique(form[sel], return_counts=True)
        top = np.argsort(-cnt)[:3]
        modes.append({"mode": m, "samples": int(sel.sum()), "threaded_share": float(np.mean(thread[sel] > 0)),
                      "front_share": float(np.mean(x[sel] < 0)), "centre": np.round(np.median(poses[sel], 0), 1).tolist(),
                      "formations": [{"formation": names[fs[t]], "share": round(float(cnt[t] / sel.sum()), 3)} for t in top]})
    order = np.argsort([-m["samples"] for m in modes])
    print(f"{len(keys)} formations -> {k} connected pieces of at least {args.min_mode} poses; "
          f"{np.mean(labels < 0):.2%} of poses unassigned", flush=True)
    for m in order:
        md = modes[m]
        print(f"  mode {m:3d} {md['samples']:7,d}  threaded {md['threaded_share']:4.0%}  front {md['front_share']:4.0%}  "
              f"{md['formations'][0]['share']:4.0%} {md['formations'][0]['formation']}", flush=True)
    # mode graph: sliding edges between modes, and two-contact creases
    cross = (labels[pairs[:, 0]] != labels[pairs[:, 1]]) & (labels[pairs[:, 0]] >= 0) & (labels[pairs[:, 1]] >= 0)
    ek, ec = np.unique(np.sort(np.c_[labels[pairs[cross, 0]], labels[pairs[cross, 1]]], 1), axis=0, return_counts=True)
    mode_edges = [{"modes": [int(a), int(b)], "lines_of_sight": int(cc)} for (a, b), cc in zip(ek, ec) if cc >= 20]
    # scores against the physical labels, as for the graph variants
    ok = labels >= 0
    ref_thread, ref_full = mge.physical_labels(poses[ok], g["feat"][ok], thread[ok])
    sc = mge.score(labels[ok], ref_thread, ref_full, W[ok][:, ok], g["feat"][ok, 0] < 1.0)
    tag = tag_of(args)
    np.save(exp / f"labels_contact_{tag}.npy", labels)
    path = exp / "experiments.json"
    old = json.loads(path.read_text()) if path.exists() else {"rows": [], "spectra": {}}
    name = f"contact: CAD contact formations, connected pieces{', normal axis' if args.normal_axis else ''}" \
           f"{', rims apart' if args.rims else ''}, min {args.min_mode}{args.suffix.replace('_', ' ')}"
    old["rows"] = [r for r in old["rows"] if r["variant"] != name] + [
        {"variant": name, "nodes": n, "edges": int(len(pairs)), "k": int(k), "k_is_eigengap": True,
         "eigengap": float("nan"), "smoothed": True, **sc}]
    path.write_text(json.dumps(old, indent=2))
    print(f"{k} modes; NMI full {sc['nmi_full']:.2f}, ARI {sc['ari_full']:.2f}, purity {sc['purity_full']:.2f}, "
          f"threading purity {sc['purity_thread']:.3f}, disagreement {sc['disagreement']:.3f}", flush=True)
    (out / f"{tag}.json").write_text(json.dumps({"k": int(k), "modes": modes, "mode_edges": mode_edges,
                                                 "formations": names}, indent=2))


class Classifier:
    """Mode of arbitrary poses: slide into contact along the contact normal, read the contact formation (with
    its side and threading state), and take the mode of the nearest labelled contact pose with that formation
    (of that side and threading state if the formation is unseen; -1 if the class is too). With --warp, the
    sliding and the contacts are computed on the GPU (hook_warp_contact) like a warp_contact_sampling contact set;
    otherwise on the CPU (MuJoCo convex parts) like a mode_graph_experiments one."""

    def __init__(self, exp, out, tag, args):
        import mode_graph_experiments as mge
        self.mge, self.geo = mge, None
        if args.warp:
            from hook_warp_contact import WarpContact
            self.geo = WarpContact(verbose=False)
            self.elements = (Elements(np.load(out / "faces_hook.npz")), Elements(np.load(out / "faces_eyelet.npz")))
            self.eps, self.rim = args.contact_eps, args.rim_mm
        else:
            mge._init_features()
            _init_strata(out, args.contact_eps, args.merge_mm, args.rim_mm)
        g = np.load(exp / "contact_graph.npz")
        z = np.load(out / "contacts.npz")
        self.labels = np.load(exp / f"labels_contact_{tag}.npy")
        self.normal = args.normal_axis
        self.coarse = json.loads((out / "aliases.json").read_text()) if args.coarse else None
        self.loop = None if args.warp else mge.HookLoop(n_centre=8000)  # as the contact graph's threading labels
        first = self._canon(z["contacts"][:, 0, :6])
        if self.normal:
            first = np.c_[first, normal_code(g["poses"], z["witness"][:, 0])]
        self.middle_mm = args.middle_mm
        cls = contact_class(z["witness"], g["thread"], self.middle_mm)
        self.keys, form = np.unique(np.c_[first, cls], axis=0, return_inverse=True)
        self.form = form.ravel()
        ok = self.labels >= 0
        self.x = g["poses"] * SCALE
        self.trees = {f: (np.flatnonzero(ok & (self.form == f)), cKDTree(self.x[ok & (self.form == f)]))
                      for f in np.unique(self.form[ok])}
        # an unseen formation falls back to the nearest labelled pose of the same class only
        cls_keys, cls_of = np.unique(cls, axis=0, return_inverse=True)
        cls_of = cls_of.ravel()
        self.by_class = {tuple(k): (np.flatnonzero(ok & (cls_of == i)), cKDTree(self.x[ok & (cls_of == i)]))
                         for i, k in enumerate(cls_keys) if (ok & (cls_of == i)).any()}

    def _canon(self, c):
        c = c.copy()
        c[..., 2] = np.where(c[..., 2] >= 0, 0, -1)
        c[..., 5] = np.where(c[..., 5] >= 0, 0, -1)
        return coarse_contacts(c, self.coarse) if self.coarse else c

    def _lookup(self, key, cls, q):
        hit = np.flatnonzero((self.keys == key).all(1))
        found = self.trees.get(hit[0]) if len(hit) else self.by_class.get(tuple(cls))
        return (int(self.labels[found[0][found[1].query(q * SCALE)[1]]]) if found else -1), (key if len(hit) else None)

    def _call_warp(self, poses, target, tol):
        from warp_contact_sampling import strata_arrays
        poses = np.atleast_2d(poses)
        sep0 = self.geo.separation(poses)
        q, _ = self.geo.project(poses, target, tol)
        con, wit, _ = strata_arrays(self.geo.contacts(q), *self.elements, self.eps, self.rim)
        key = self._canon(con[:, 0, :6])
        if self.normal:
            key = np.c_[key, normal_code(q, wit[:, 0])]
        cls = contact_class(wit, self.geo.threading(q), self.middle_mm)
        key = np.c_[key, cls]
        return [(*self._lookup(key[i], cls[i], q[i]), q[i], float(sep0[i])) for i in range(len(q))]

    def __call__(self, poses, target=0.02, tol=0.02):
        """(mode, formation key or None, contact pose, separation before sliding) per pose."""
        if self.geo is not None:
            return self._call_warp(poses, target, tol)
        res = []
        for p in poses:
            sep0 = self.mge._probe(p)[0]
            q, ok = self.mge.contact_project(p, target, tol)
            c, wit = strata_task(q[None])
            key = self._canon(c[0, 0, :6])
            if self.normal:
                key = np.r_[key, normal_code(q[None], wit[:, 0])]
            cls = contact_class(wit, np.array([self.loop.threading(q)]), self.middle_mm)[0]
            key = np.r_[key, cls]
            hit = np.flatnonzero((self.keys == key).all(1))
            found = self.trees.get(hit[0]) if len(hit) else self.by_class.get(tuple(cls))
            mode = int(self.labels[found[0][found[1].query(q * SCALE)[1]]]) if found else -1
            res.append((mode, key if len(hit) else None, q, sep0))
        return res


def trajectory_stage(args):
    """Modes met along the assembly trajectories, each pose slid into contact along its contact normal."""
    from build_separation_field import reference_path

    exp, out = args.experiment_dir, args.out_dir
    tag = args.labels or tag_of(args)
    clf = Classifier(exp, out, tag, args)
    info = json.loads((out / f"{tag}.json").read_text())
    names = {m["mode"]: m.get("name") or m["formations"][0]["formation"] for m in info["modes"]}
    names[-1] = "unassigned"
    report = {}
    goal = clf(np.zeros((1, 6)))[0]
    print(f"assembled pose: mode {goal[0]} ({names[goal[0]]})", flush=True)
    report["assembled_mode"] = goal[0]
    for csv in args.trajectories:
        path = reference_path(ASSET_DIR / csv, spacing=1.0)
        res = clf(path)
        seq = []
        for i, (m, key, q, sep0) in enumerate(res):
            if not seq or seq[-1]["mode"] != m:
                seq.append({"mode": m, "name": names[m], "from_step": i, "steps": 0, "min_sep_mm": sep0})
            seq[-1]["steps"] += 1
            seq[-1]["min_sep_mm"] = min(seq[-1]["min_sep_mm"], sep0)
        report[csv] = seq
        print(f"{csv}: {len(path)} poses at 1 mm / 1 deg spacing, {len(seq)} mode runs:", flush=True)
        for r in seq:
            print(f"   steps {r['from_step']:4d}+{r['steps']:3d}  mode {r['mode']:3d}  (clearance there >= "
                  f"{r['min_sep_mm']:5.2f} mm)  {r['name']}", flush=True)
    (out / f"{tag}_trajectories.json").write_text(json.dumps(report, indent=2))


def alias(name, aliases):
    """'hook 5:torus/inner ~ eyelet 1:torus/inner' -> 'crook inside ~ loop hole side', with aliases.json."""
    out = []
    name, extra = (name.split(" {")[0], name.split(" {")[1].rstrip("}")) if " {" in name else (name, None)
    half, thread = extra.split(", ") if extra else (None, "unthreaded")
    normal = name.split(" [n ")[1].rstrip("]") if " [n " in name else None
    for part, text in zip(("hook", "eyelet"), name.split(" [n ")[0].split(" ~ ")):
        face = text.split(" ")[1].split(":")[0]
        side = text.split("/")[1].split(" ")[0] if "/" in text else ""
        rim = " (rim)" if text.endswith(" rim") else ""
        element = aliases[part].get(face, face) + (f" {HALF_NAMES[half]}" if part == "eyelet" and half else "")
        out.append(f"{element} {aliases[part + '_sides'].get(side, side)}{rim}".strip())
    where = {"+z": "on top", "-z": "underneath", "-y": "pushed right", "+y": "pushed left", "-x": "in front",
             "+x": "behind"}
    text = " ~ ".join(out) + (f", {where.get(normal, normal)}" if normal else "")
    return text + ("" if thread == "unthreaded" else f"; {thread}")


def graph_stage(args):
    """Mode graph: sliding passages between modes, and hops to the assembled mode. Of the formation modes, or
    with --labels of any labelling described in strata/LABELS.json (hook_sided_modes.py)."""
    from scipy.sparse.csgraph import shortest_path

    exp, out = args.experiment_dir, args.out_dir
    tag = args.labels or tag_of(args)
    info = json.loads((out / f"{tag}.json").read_text())
    traj = json.loads((out / f"{tag}_trajectories.json").read_text())
    aliases = json.loads((out / "aliases.json").read_text())
    g = np.load(exp / "contact_graph.npz")
    labels = np.load(exp / f"labels_contact_{tag}.npy")
    goal = traj["assembled_mode"]
    k = info["k"]
    names = {m["mode"]: m.get("name") or alias(m["formations"][0]["formation"], aliases) for m in info["modes"]}
    ok = labels >= 0
    size = np.bincount(labels[ok], minlength=k)
    threaded = np.bincount(labels[ok], g["thread"][ok] > 0, minlength=k) / np.maximum(size, 1)
    front = np.bincount(labels[ok], g["poses"][ok, 0] < 0, minlength=k) / np.maximum(size, 1)
    # passages: sliding edges between two modes, over the smaller mode's poses (any classes: the hook may
    # slide across the mid-plane or thread on the way)
    pairs = g["pairs"][g["in_10.0"] & (g["stray"] <= args.slide_mm)]
    a, b = labels[pairs[:, 0]], labels[pairs[:, 1]]
    cross = (a != b) & (a >= 0) & (b >= 0)
    ek, ec = np.unique(np.sort(np.c_[a[cross], b[cross]], 1), axis=0, return_counts=True)
    edges = [(int(p), int(q), c / min(size[p], size[q])) for (p, q), c in zip(ek, ec)
             if c / min(size[p], size[q]) >= args.min_passage]
    A = sp.coo_matrix((np.ones(2 * len(edges)), (np.r_[[e[0] for e in edges], [e[1] for e in edges]],
                                                   np.r_[[e[1] for e in edges], [e[0] for e in edges]])), shape=(k, k))
    hops, pred = shortest_path(A, unweighted=True, indices=goal, return_predecessors=True)
    print(f"mode graph: {k} modes, {len(edges)} passages (at least {args.min_passage:.0%} of the smaller mode's sliding "
          f"edges); assembled mode {goal} ({names[goal]})", flush=True)
    report = []
    for m in sorted(range(k), key=lambda m: (hops[m], -size[m])):
        via, j = [], pred[m]
        while j >= 0 and j != goal:
            via.append(int(j))
            j = pred[j]
        report.append({"mode": m, "name": names[m], "samples": int(size[m]), "hops": None if np.isinf(hops[m]) else int(hops[m]),
                       "via": via, "threaded": float(threaded[m]), "front": float(front[m])})
        print(f"  {'-' if np.isinf(hops[m]) else int(hops[m])} hops  mode {m:3d} {size[m]:7,d}  threaded "
              f"{threaded[m]:4.0%}  front {front[m]:4.0%}  {names[m]:44s} "
              f"via {' > '.join(map(str, via)) or '-'}", flush=True)
    # layered figure: one column per hop count from the assembled mode
    col = np.where(np.isinf(hops), np.nanmax(np.where(np.isinf(hops), np.nan, hops)) + 1, hops).astype(int)
    pos = {}
    for c in np.unique(col):
        members = sorted(np.flatnonzero(col == c), key=lambda m: (-threaded[m], front[m], -size[m]))
        for r, m in enumerate(members):
            pos[m] = (c, -(r - (len(members) - 1) / 2))
    def label(m):  # contact on the first line(s), side and threading on the last
        head, _, tail = names[m].partition("; ")
        return "\n".join(textwrap.wrap(f"{m}: {head}", 36) + ([tail] if tail else []))

    ncol = col.max() + 1
    fig, ax = plt.subplots(figsize=(3.4 * ncol + 1.5, 0.7 * max(np.bincount(col)) + 2.5), facecolor=SURFACE)
    for a, b, w in edges:
        ax.plot(*zip(pos[a], pos[b]), color="#b9b8b1", linewidth=0.3 + 2.5 * np.sqrt(w), zorder=1, alpha=0.8)
    for m in range(k):
        thr = threaded[m]
        ax.scatter(*pos[m], s=20 + 600 * np.sqrt(size[m] / size.max()), zorder=2, edgecolors=SURFACE, linewidths=1.5,
                   color="#eb6834" if m == goal else ("#1baf7a" if thr > 0.5 else "#2a78d6"))
        ax.text(pos[m][0] + 0.05, pos[m][1], label(m), fontsize=7, color=INK, va="center", zorder=3, linespacing=1.1)
    ax.legend(handles=[plt.Line2D([], [], marker="o", linestyle="", color=c, label=l) for c, l in
                       (("#eb6834", "assembled mode"), ("#1baf7a", "threaded"), ("#2a78d6", "not threaded"))],
              loc="upper right", frameon=False, fontsize=8, labelcolor=INK_2)
    for c in np.unique(col):
        ax.text(c, max(p[1] for p in pos.values()) + 0.7,
                "assembled" if c == 0 else f"{c} passage{'s' if c > 1 else ''} away", ha="center", fontsize=9, color=INK_2)
    ax.set_xlim(-0.15, ncol + 0.05)
    fig.suptitle(f"Modes of the hook on the eyelet ({tag}, {k} modes): edges where the hook can slide from one to another, "
                 f"width = passage size", color=INK, fontsize=10, x=0.01, ha="left")
    ax.axis("off")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out / f"{tag}_graph.png", facecolor=SURFACE, dpi=110)
    (out / f"{tag}_graph.json").write_text(json.dumps({"goal": goal, "edges": [[int(a), int(b), float(w)] for a, b, w in edges],
                                                        "modes": report}, indent=2))


_clf = None


def _init_classifier(exp, out, tag, args):
    global _clf
    _clf = Classifier(exp, out, tag, args)


def classify_task(poses):
    res = _clf(poses)
    return np.array([r[0] for r in res]), np.array([r[3] for r in res])


def free_stage(args):
    """Free poses labelled by the contact mode they slide into along their contact normal: a generalized
    Voronoi diagram of the contact modes in pose space, whose cell walls lie where two are equally near."""
    import mode_graph_experiments as mge
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    exp, out = args.experiment_dir, args.out_dir
    tag = tag_of(args)
    g = np.load(exp / "samples_graph.npz")
    poses = g["poses"]
    t0 = time.time()
    chunks = np.array_split(np.arange(len(poses)), args.workers * 8)
    with mp.get_context("fork").Pool(args.workers, initializer=_init_classifier, initargs=(exp, out, tag, args)) as pool:
        res = pool.map(classify_task, [poses[c] for c in chunks])
    labels = np.concatenate([r[0] for r in res])
    print(f"{len(poses):,} free samples labelled by the contact mode they slide into ({time.time() - t0:.0f} s); "
          f"{len(np.unique(labels))} modes reached", flush=True)
    np.save(exp / f"labels_samples_gvd_{tag}.npy", labels)
    keep = (g["clear"] >= 0) & g["in_10.0"]
    W = mge.adjacency(len(poses), g["pairs"][keep], np.ones(keep.sum()))
    ref_thread, ref_full = mge.physical_labels(poses, g["feat"], g["thread"])
    sc = mge.score(labels, ref_thread, ref_full, W, g["feat"][:, 0] < 1.0)
    print(f"against the physical labels: NMI full {sc['nmi_full']:.2f}, ARI {sc['ari_full']:.2f}, purity {sc['purity_full']:.2f}, "
          f"threading purity {sc['purity_thread']:.3f}; local disagreement over line-of-sight edges {sc['disagreement']:.3f}",
          flush=True)
    for other in args.compare:
        lab = np.load(exp / other)
        ok = lab >= 0
        print(f"   vs {other}: ARI {adjusted_rand_score(lab[ok], labels[ok]):.3f}, NMI "
              f"{normalized_mutual_info_score(lab[ok], labels[ok]):.3f}", flush=True)
    path = exp / "experiments.json"
    old = json.loads(path.read_text()) if path.exists() else {"rows": [], "spectra": {}}
    name = f"samples: free poses by the contact-formation mode they slide into ({tag})"
    old["rows"] = [r for r in old["rows"] if r["variant"] != name] + [
        {"variant": name, "nodes": int(len(poses)), "edges": int(keep.sum()), "k": int(len(np.unique(labels))),
         "k_is_eigengap": True, "eigengap": float("nan"), "smoothed": True, **sc}]
    path.write_text(json.dumps(old, indent=2))


def flat_stage(args):
    """Split each formation mode into nearly flat patches of the contact set: bisect a patch (2-means on its
    wrench directions [n, r x n / --flat-lever], each half then split into its sliding-connected pieces)
    while the wrench direction spreads over it by more than --flat-deg (90th percentile angle from the
    patch's mean direction). The torque part turns with the hook, so upright and pitched hooks pressing
    the same face part ways."""
    import mode_graph_experiments as mge
    from sklearn.cluster import KMeans

    exp, out = args.experiment_dir, args.out_dir
    tag = tag_of(args)
    g = np.load(exp / "contact_graph.npz")
    base = np.load(exp / f"labels_contact_{tag}.npy")
    n = len(base)
    Wd = g["wrench_closest"].copy()
    Wd[:, 3:] /= args.flat_lever
    Wd /= np.linalg.norm(Wd, axis=1, keepdims=True)
    keep = g["in_10.0"] & (g["stray"] <= args.slide_mm)
    pairs = g["pairs"][keep]
    adj = mge.adjacency(n, pairs, np.ones(len(pairs))).tocsr()

    def spread(ids):
        m = Wd[ids].mean(0)
        m /= np.linalg.norm(m)
        return np.degrees(np.percentile(np.arccos(np.clip(Wd[ids] @ m, -1, 1)), 90))

    def pieces(ids):
        _, comp = connected_components(adj[ids][:, ids], directed=False)
        return [ids[comp == c] for c in np.unique(comp)]

    labels = np.full(n, -1)
    stack = [np.flatnonzero(base == m) for m in range(base.max() + 1)]
    done, small = [], []
    while stack:
        ids = stack.pop()
        if len(ids) < 2 * args.min_mode or spread(ids) <= args.flat_deg:
            done.append(ids)
            continue
        half = KMeans(2, n_init=4, random_state=0).fit_predict(Wd[ids])
        for h in range(2):
            for piece in pieces(ids[half == h]):
                (stack if len(piece) >= args.min_mode else small).append(piece)
    for c, ids in enumerate(done):
        labels[ids] = c
    # fragments below --min-mode join the patch they share most sliding edges with
    for ids in small:
        nb = adj[ids].indices
        nb = nb[labels[nb] >= 0]
        if len(nb):
            labels[ids] = np.bincount(labels[nb]).argmax()
    k = labels.max() + 1
    ok = labels >= 0
    ref_thread, ref_full = mge.physical_labels(g["poses"][ok], g["feat"][ok], g["thread"][ok])
    sc = mge.score(labels[ok], ref_thread, ref_full, adj[ok][:, ok], g["feat"][ok, 0] < 1.0)
    sp_ = [spread(np.flatnonzero(labels == c)) for c in range(k)]
    print(f"{base.max() + 1} formation modes -> {k} flat patches (wrench spread <= {args.flat_deg} deg, p90; "
          f"median patch spread {np.median(sp_):.0f} deg); NMI full {sc['nmi_full']:.2f}, ARI {sc['ari_full']:.2f}, "
          f"threading purity {sc['purity_thread']:.3f}, disagreement {sc['disagreement']:.3f}", flush=True)
    ftag = f"{tag}_flat{args.flat_deg:g}"
    np.save(exp / f"labels_contact_{ftag}.npy", labels)
    parent = {c: int(np.bincount(base[labels == c][base[labels == c] >= 0]).argmax()) for c in range(k)}
    (out / f"{ftag}.json").write_text(json.dumps({"k": int(k), "parent_mode": parent, "spread_deg": sp_}, indent=2))
    path = exp / "experiments.json"
    old = json.loads(path.read_text()) if path.exists() else {"rows": [], "spectra": {}}
    name = f"contact: CAD contact formations ({tag}), split into flat patches within {args.flat_deg:g} deg"
    old["rows"] = [r for r in old["rows"] if r["variant"] != name] + [
        {"variant": name, "nodes": int(ok.sum()), "edges": int(len(pairs)), "k": int(k), "k_is_eigengap": True,
         "eigengap": float("nan"), "smoothed": True, **sc}]
    path.write_text(json.dumps(old, indent=2))


# --- convex free cells ---------------------------------------------------------------------------------

def grow_cell(seed, boundary, tree, free, args):
    """IRIS-style convex cell around seed (lever-10 pose metric): hyperplanes at the nearest boundary points
    in the current ellipsoid's metric until none is left inside, then the ellipsoid (centred on the seed)
    takes the shape of the free samples inside. Returns (A, b) with the cell {y : A y <= b}."""
    d, C = seed.copy(), np.eye(6) * args.cell_r0
    for _ in range(args.cell_iters):
        near = boundary[tree.query_ball_point(d, args.cell_reach)]
        Ci = np.linalg.inv(C)
        A, b = [], []
        for axis in range(6):  # bounding box of the search reach
            e = np.eye(6)[axis]
            A += [e, -e]
            b += [d[axis] + args.cell_reach, -(d[axis] - args.cell_reach)]
        rem = near
        while len(rem):
            u = (rem - d) @ Ci.T
            j = np.argmin(np.einsum("ij,ij->i", u, u))
            a = Ci.T @ u[j]
            a /= np.linalg.norm(a)
            A.append(a)
            b.append(a @ rem[j])
            rem = rem[rem @ a < b[-1] - 1e-9]
        A, b = np.array(A), np.array(b)
        inside = free[np.all(free @ A.T <= b, axis=1)]
        if len(inside) < 12:
            break
        # the centre stays at the seed (so the cell keeps it); the shape follows the free samples inside
        C = np.linalg.cholesky((inside - d).T @ (inside - d) / len(inside) + 1e-6 * np.eye(6)) * args.cell_shrink
    return A, b


def cell_check_task(job):
    """Separation [mm] at random poses inside a cell (rejection sampling in the free samples' box)."""
    A, b, lo, hi, n, seed = job
    rng = np.random.default_rng(seed)
    pts = []
    for _ in range(200):
        y = rng.uniform(lo, hi, (n * 20, 6))
        pts.extend(y[np.all(y @ A.T <= b, axis=1)])
        if len(pts) >= n:
            break
    pts = np.array(pts[:n])
    if not len(pts):
        return np.zeros(0)
    m, d, S = _ctx["m"], _ctx["d"], _ctx["sep"]
    out = []
    for y in pts:
        pose = y / SCALE
        d.qpos[:3], d.qpos[3:7] = pose[:3] * 1e-3, Rotation.from_rotvec(pose[3:], degrees=True).as_quat(scalar_first=True)
        mujoco.mj_kinematics(m, d)
        out.append(S(m, d)[0] * 1e3)
    return np.array(out)


def convex_stage(args):
    import mode_graph_experiments as mge

    exp, out = args.experiment_dir, args.out_dir
    boundary = np.load(exp / "contact_graph.npz")["poses"] * SCALE
    g = np.load(exp / "samples_graph.npz")
    free, sep = g["poses"] * SCALE, g["feat"][:, 0]
    tree = cKDTree(boundary)
    covered = np.zeros(len(free), bool)
    depth = np.full(len(free), -np.inf)
    label = np.full(len(free), -1)
    cells, boxes = [], []
    t0 = time.time()
    while covered.mean() < args.cell_cover and len(cells) < args.max_cells:
        cand = np.flatnonzero(~covered)
        seed = free[cand[np.argmax(sep[cand])]]
        A, b = grow_cell(seed, boundary, tree, free, args)
        slack = np.min(b[None, :] - free @ A.T, axis=1)  # distance-like depth inside the cell
        inside = slack >= 0
        if inside.sum() < args.min_cell:
            covered[np.argmin(np.linalg.norm(free - seed, axis=1))] = True  # give up on this seed
            continue
        better = inside & (slack > depth)
        label[better], depth[better] = len(cells), slack[better]
        covered |= inside
        cells.append((A, b))
        boxes.append((free[inside].min(0), free[inside].max(0)))
        if len(cells) % 10 == 0:
            print(f"   {len(cells)} cells, {covered.mean():.1%} of free samples covered ({time.time() - t0:.0f} s)", flush=True)
    print(f"{len(cells)} convex cells cover {covered.mean():.1%} of the free samples ({time.time() - t0:.0f} s); cell sizes "
          f"(samples) p50 {np.median(np.bincount(label[label >= 0])):.0f}, max {np.bincount(label[label >= 0]).max()}", flush=True)
    # soundness: separation at random poses inside each cell
    lo, hi = free.min(0), free.max(0)
    jobs = [(A, b, np.maximum(lo, bl - 2), np.minimum(hi, bh + 2), args.cell_checks, c)
            for c, ((A, b), (bl, bh)) in enumerate(zip(cells, boxes))]
    with mp.get_context("fork").Pool(args.workers, initializer=_init_strata,
                                      initargs=(out, args.contact_eps, args.merge_mm, args.rim_mm)) as pool:
        checks = pool.map(cell_check_task, jobs)
    allc = np.concatenate([c for c in checks if len(c)])
    bad = [np.mean(c < 0) for c in checks if len(c)]
    print(f"soundness: {np.mean(allc < 0):.2%} of {len(allc):,} random poses inside the cells collide (worst cell "
          f"{max(bad):.0%}); separation inside p5 {np.percentile(allc, 5):.2f} mm", flush=True)
    keep = (g["clear"] >= 0) & g["in_10.0"]
    W = mge.adjacency(len(free), g["pairs"][keep], np.ones(keep.sum()))
    ok = label >= 0
    ref_thread, ref_full = mge.physical_labels(g["poses"][ok], g["feat"][ok], g["thread"][ok])
    sc = mge.score(label[ok], ref_thread, ref_full, W[ok][:, ok], g["feat"][ok, 0] < 1.0)
    print(f"{len(cells)} cells: NMI full {sc['nmi_full']:.2f}, ARI {sc['ari_full']:.2f}, purity {sc['purity_full']:.2f}, "
          f"threading purity {sc['purity_thread']:.3f}, disagreement {sc['disagreement']:.3f}", flush=True)
    np.save(exp / f"labels_samples_convex_cells.npy", label)
    np.savez(out / "convex_cells.npz", **{f"A{c}": A for c, (A, b) in enumerate(cells)},
             **{f"b{c}": b for c, (A, b) in enumerate(cells)}, collide_share=np.array([np.mean(c < 0) if len(c) else np.nan
                                                                                      for c in checks]))
    path = exp / "experiments.json"
    old = json.loads(path.read_text()) if path.exists() else {"rows": [], "spectra": {}}
    name = "samples: IRIS-style convex free cells (boundary = contact poses)"
    old["rows"] = [r for r in old["rows"] if r["variant"] != name] + [
        {"variant": name, "nodes": int(ok.sum()), "edges": int(keep.sum()), "k": len(cells), "k_is_eigengap": True,
         "eigengap": float("nan"), "smoothed": True, **sc}]
    path.write_text(json.dumps(old, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["primitives", "halves", "strata", "formations", "trajectory", "graph", "free", "convex",
                                          "flat"])
    parser.add_argument("experiment_dir", type=Path, nargs="?")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--contact-eps", type=float, default=0.1,
                        help="hull pairs within this of the closest touch too [mm]")
    parser.add_argument("--merge-mm", type=float, default=2.0, help="touching pairs this close on both parts are one contact")
    parser.add_argument("--rim-mm", type=float, default=0.3, help="a witness this close to a sharp edge is on the rim")
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--suffix", default="", help="appended to output names (for sensitivity runs)")
    parser.add_argument("--labels", help="trajectory: classify into labels_contact_LABELS.npy (described in "
                                         "strata/LABELS.json) instead of the formation modes")
    parser.add_argument("--normal-axis", action="store_true",
                        help="tell contacts apart by the dominant eyelet-frame axis of their normal")
    parser.add_argument("--rims", action="store_true", help="tell rim contacts from face contacts")
    parser.add_argument("--middle-mm", type=float, default=MIDDLE_MM,
                        help="contacts this close to the loop's mid-plane are on neither side but in the middle")
    parser.add_argument("--warp", action="store_true",
                        help="trajectory / free: slide into contact and read contacts on the GPU (hook_warp_contact), "
                             "for a contact set from warp_contact_sampling.py")
    parser.add_argument("--coarse", action="store_true",
                        help="formations of parts (aliases.json groups) instead of faces, without wire sides")
    parser.add_argument("--slide-mm", type=float, default=0.25, help="sliding edges joining a formation's poses")
    parser.add_argument("--min-mode", type=int, default=300, help="smaller pieces are unassigned (or absorbed)")
    parser.add_argument("--min-share", type=float, default=0.2,
                        help="a piece holding this share of its class (side, threading) is a mode however small")
    parser.add_argument("--absorb", action="store_true",
                        help="small pieces join a neighbouring piece of their class instead of staying unassigned")
    parser.add_argument("--flat-deg", type=float, default=30.0, help="largest wrench-direction spread of a flat patch")
    parser.add_argument("--flat-lever", type=float, default=10.0, help="torque scale of the wrench direction [mm]")
    parser.add_argument("--cell-r0", type=float, default=1.0, help="initial ellipsoid radius [mm]")
    parser.add_argument("--cell-reach", type=float, default=25.0, help="boundary points considered around a seed [mm]")
    parser.add_argument("--cell-iters", type=int, default=3)
    parser.add_argument("--cell-shrink", type=float, default=1.0, help="ellipsoid = this x the inside samples' covariance")
    parser.add_argument("--cell-cover", type=float, default=0.9, help="stop at this share of free samples covered")
    parser.add_argument("--max-cells", type=int, default=150)
    parser.add_argument("--min-cell", type=int, default=50, help="cells holding fewer free samples are dropped")
    parser.add_argument("--cell-checks", type=int, default=200, help="random poses checked per cell")
    parser.add_argument("--compare", nargs="*", default=[], help="label files to compare the free-pose modes with")
    parser.add_argument("--min-passage", type=float, default=0.02,
                        help="mode graph: keep passages carrying this share of the smaller mode's sliding edges")
    parser.add_argument("--trajectories", nargs="*", default=["hook_assembly_max_separation_trajectory.csv",
                                                              "hook_assembly_refined_trajectory.csv"])
    parser.add_argument("--curve-window", type=float, default=4.0, help="curvature fit radius [mm]")
    parser.add_argument("--bend-curvature", type=float, default=0.04, help="[1/mm]: a torus face's axis curves tighter")
    parser.add_argument("--min-face", type=float, default=8.0, help="smallest CAD face [mm^2]")
    parser.add_argument("--composite-face", type=float, default=60.0,
                        help="touching faces smaller than this [mm^2] become one composite face")
    parser.add_argument("--sharp-deg", type=float, default=45.0, help="dihedral angle of a rim edge [deg]")
    args = parser.parse_args()
    {"primitives": primitives_stage, "halves": halves_stage, "strata": strata_stage, "formations": formations_stage,
     "trajectory": trajectory_stage, "graph": graph_stage, "free": free_stage, "convex": convex_stage,
     "flat": flat_stage}[args.stage](args)


if __name__ == "__main__":
    main()
