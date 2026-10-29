#!/usr/bin/env python3
"""Index a CAD mesh (STL) into geometric features, as a B-rep would have them: faces (smooth patches of
triangles), edges (chains of sharp mesh edges between faces) and corners (where edge chains meet), each
classified by its fitted geometry (plane, cylinder, cone / circle, line).

Faces: triangles joined across mesh edges whose dihedral angle is below --sharp-deg (a tessellated
cylinder's facets meet at a few degrees, a real crease at tens), then split where a fillet joins two
primitives smoothly (planar, cylindrical and blend triangles, relative to the part's y axis). Edges: the sharp mesh edges, grouped
into chains that run between corners (vertices with other than two sharp edges) and split where the
pair of faces they separate changes. Each feature gets an id, a kind, its fitted parameters and the
mesh elements it holds, and the mesh's chord error (how far its facets fall inside a fitted cylinder),
which bounds how exactly the mesh can resolve contact. Features that are rotated copies of each other
about the part's axis are grouped into symmetry orbits (symmetry_orbits), the classes contact is read in.

    conda run -n mujoco_env python scripts/cad_features.py assets/cylinder/cylinder_real_peg.stl
"""

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def fit_face(mesh, tris):
    """Kind and parameters of a smooth patch: plane (one normal), cylinder or cone about the y axis."""
    n = mesh.face_normals[tris]
    a = mesh.area_faces[tris]
    v = mesh.vertices[np.unique(mesh.faces[tris])]
    mean_n = (n * a[:, None]).sum(0)
    if np.linalg.norm(mean_n) / a.sum() > 0.999:
        nn = mean_n / np.linalg.norm(mean_n)
        return dict(kind="plane", normal=nn.round(4).tolist(), offset=float((v @ nn).mean()))
    r = np.hypot(v[:, 0], v[:, 2])
    ny = n[:, 1]
    if np.abs(ny).max() < 0.02:
        return dict(kind="cylinder", radius=float(r.max()), y=[float(v[:, 1].min()), float(v[:, 1].max())],
                    outward=bool((n[:, 0] * mesh.triangles_center[tris][:, 0]
                                  + n[:, 2] * mesh.triangles_center[tris][:, 2]).mean() > 0))
    if np.ptp(ny) < 0.02:
        return dict(kind="cone", half_angle_deg=float(np.degrees(np.arcsin(abs(ny.mean())))),
                    r=[float(r.min()), float(r.max())], y=[float(v[:, 1].min()), float(v[:, 1].max())])
    return dict(kind="other")


def fit_edge(vertices):
    """Kind and parameters of an edge chain: circle about the y axis, or line."""
    r = np.hypot(vertices[:, 0], vertices[:, 2])
    if np.ptp(vertices[:, 1]) < 1e-6 and np.ptp(r) < 1e-3 * max(r.max(), 1):
        return dict(kind="circle", radius=float(r.mean()), y=float(vertices[0, 1]))
    d = vertices - vertices.mean(0)
    s = np.linalg.svd(d, compute_uv=False)
    if len(s) > 1 and s[1] < 1e-6 * max(s[0], 1e-9):
        return dict(kind="line", span=float(s[0]))
    return dict(kind="curve", n_vertices=int(len(vertices)))


def split_blended(mesh, face_of_tri, smooth_pairs):
    """Split smooth patches that hold several primitives joined by fillets: triangles are planar (normal
    along the part's y axis), cylindrical (normal across it) or blend, and a patch splits into connected
    runs of one class (and, for planes, one height)."""
    ny = mesh.face_normals[:, 1]
    cls = np.where(np.abs(ny) > 0.999, 0, np.where(np.abs(ny) < 0.02, 1, 2))
    height = np.round(mesh.triangles_center[:, 1], 3)
    a, b = smooth_pairs[:, 0], smooth_pairs[:, 1]
    same = (face_of_tri[a] == face_of_tri[b]) & (cls[a] == cls[b]) & ((cls[a] != 0) | (height[a] == height[b]))
    n_f = len(mesh.faces)
    g = coo_matrix((np.ones(same.sum()), (a[same], b[same])), shape=(n_f, n_f))
    return connected_components(g, directed=False)[1]


def index_mesh(path, sharp_deg=20.0):
    mesh = trimesh.load(path, process=True)
    mesh.merge_vertices()
    adj = mesh.face_adjacency
    angle = np.degrees(mesh.face_adjacency_angles)
    smooth = angle < sharp_deg
    n_f = len(mesh.faces)
    g = coo_matrix((np.ones(smooth.sum()), (adj[smooth, 0], adj[smooth, 1])), shape=(n_f, n_f))
    n_faces, face_of_tri = connected_components(g, directed=False)
    face_of_tri = split_blended(mesh, face_of_tri, adj[smooth])
    n_faces = face_of_tri.max() + 1
    faces = []
    for f in range(n_faces):
        tris = np.flatnonzero(face_of_tri == f)
        faces.append(dict(id=f, n_triangles=int(len(tris)), area=float(mesh.area_faces[tris].sum()), **fit_face(mesh, tris)))

    # sharp mesh edges, grouped into chains by the pair of faces they separate and by connectivity
    sharp = ~smooth
    sedges = mesh.face_adjacency_edges[sharp]
    side = np.sort(face_of_tri[adj[sharp]], 1)
    n_v = len(mesh.vertices)
    edges, vertex_edges = [], np.zeros(n_v, int)
    np.add.at(vertex_edges, sedges.ravel(), 1)
    for pair in np.unique(side, axis=0):
        sel = np.flatnonzero((side == pair).all(1))
        e = sedges[sel]
        gv = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n_v, n_v))
        _, comp = connected_components(gv, directed=False)
        for c in np.unique(comp[e[:, 0]]):
            ee = e[comp[e[:, 0]] == c]
            verts = np.unique(ee)
            edges.append(dict(id=len(edges), faces=pair.tolist(), n_mesh_edges=int(len(ee)),
                              length=float(np.linalg.norm(mesh.vertices[ee[:, 0]] - mesh.vertices[ee[:, 1]], axis=1).sum()),
                              mesh_edges=ee, **fit_edge(mesh.vertices[verts])))
    corners = np.flatnonzero((vertex_edges > 0) & (vertex_edges != 2))

    # chord error: how far a cylinder's facets fall inside its circle (from the facets' angular width)
    chord = 0.0
    for f in faces:
        if f["kind"] == "cylinder":
            tris = np.flatnonzero(face_of_tri == f["id"])
            v = mesh.vertices[np.unique(mesh.faces[tris])]
            angles = np.unique(np.round(np.arctan2(v[:, 2], v[:, 0]), 6))
            f["segments"] = int(round(2 * np.pi / np.median(np.diff(angles)))) if len(angles) > 2 else 0
            if f["segments"] >= 12:
                chord = max(chord, f["radius"] * (1 - np.cos(np.pi / f["segments"])))
    return mesh, face_of_tri, faces, edges, corners, chord


def symmetry_orbits(mesh, face_of_tri, faces, edges, tol=0.1):
    """Group features into orbits of the part's rotations about its y axis: features that are copies of
    each other turned about the axis (the flange's scallops, the base plate's sides) share an orbit, and
    a feature that goes all the way round (a circle, a turned face) is its own orbit. Each feature gets
    'orbit' (an id) and 'order' (how many copies), so contact classes can be read per orbit."""
    keys = []
    for f in faces:
        v = mesh.vertices[np.unique(mesh.faces[face_of_tri == f["id"]])]
        r = np.hypot(v[:, 0], v[:, 2])
        keys.append(("face", f["kind"], np.array([np.sqrt(f["area"]), r.min(), r.max(), v[:, 1].min(), v[:, 1].max()])))
    for e in edges:
        v = mesh.vertices[np.unique(e["mesh_edges"])]
        r = np.hypot(v[:, 0], v[:, 2])
        keys.append(("edge", e["kind"], np.array([e["length"], r.min(), r.max(), v[:, 1].min(), v[:, 1].max()])))
    orbit = np.full(len(keys), -1)
    for i, (t, kind, d) in enumerate(keys):  # copies: same type and kind, and sizes and extents within tol mm
        if orbit[i] < 0:
            orbit[i] = i
            for j in range(i + 1, len(keys)):
                if orbit[j] < 0 and keys[j][:2] == (t, kind) and np.abs(keys[j][2] - d).max() < tol:
                    orbit[j] = i
    _, orbit, order = np.unique(orbit, return_inverse=True, return_counts=True)
    for feat, o in zip(faces + edges, orbit):
        feat["orbit"], feat["order"] = int(o), int(order[o])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stl", type=Path, nargs="+")
    parser.add_argument("--sharp-deg", type=float, default=20.0)
    args = parser.parse_args()
    for path in args.stl:
        mesh, face_of_tri, faces, edges, corners, chord = index_mesh(path, args.sharp_deg)
        symmetry_orbits(mesh, face_of_tri, faces, edges)
        n_orbits = len({f["orbit"] for f in faces + edges})
        print(f"== {path.name}: {len(mesh.faces)} triangles, {len(faces)} faces, {len(edges)} edges, {len(corners)} corners, "
              f"{n_orbits} symmetry orbits, chord error {chord * 1e3:.1f} um")
        for f in sorted(faces, key=lambda f: -f["area"]):
            print("  face", json.dumps({k: v for k, v in f.items()}))
        for e in edges:
            print("  edge", json.dumps({k: v for k, v in e.items() if k != "mesh_edges"}))


if __name__ == "__main__":
    main()
