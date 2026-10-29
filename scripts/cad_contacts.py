#!/usr/bin/env python3
"""Contact modes from the CAD alone: for each sampled pose, which features of the peg (faces, edges of
cad_features.py) touch which features of the hole.

The mesh indexing gives each part's faces and edges and fits each one's exact geometry (plane, cylinder
about the part's axis, circle); contact is computed on these fitted primitives (as a STEP file's B-rep
would give them) rather than on the facets, whose chord error (~20 um) would blur every contact. A
contact between bodies in general position involves an edge of one or both: an edge on a face, a face
on an edge, or edge on edge (flush faces are measure zero). So every convex circular edge of either part
is sampled densely, and for each pose:
  1. the edge samples of each part are placed in the other part's frame and their exact distance to each
     of its features (bounded to the feature's extent) is found; the closest feature labels the sample;
  2. along each edge, the local minima of that gap below --tol (and standing out by --prominence) are
     contacts: (this edge, the other part's closest feature);
  3. two edges that cross each see the other's neighbouring face a few um from the crossing: a peg-edge
     and a hole-edge contact within --merge-link mm are one edge-on-edge contact; contacts of the same
     pair within --link mm are one.
The mode is the multiset of (peg feature, hole feature) pairs. Features keep the CAD's own descriptions;
the analytical names are attached only to cross-check against the known modes (--compare).
Not covered: the fillets (blend faces), which the CAD has at the base of the tube and the shaft and
the analytical model does not.

    conda run -n mujoco_env python scripts/cad_contacts.py RUN_DIR [RUN_DIR ...] --samples LABELS.npz --out OUT.npz
"""

import argparse
import glob
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks
from scipy.spatial.transform import Rotation

from cad_features import index_mesh

ASSETS = Path(__file__).resolve().parent.parent / "assets" / "cylinder"


class Part:
    """A part's features as exact primitives about its y axis: ('plane', y, r0, r1), ('cyl', r, y0, y1),
    ('circle', r, y); and dense samples of its convex circular edges."""

    def __init__(self, path, name, edge_step, sharp_deg):
        mesh, face_of_tri, faces, edges, _, _ = index_mesh(path, sharp_deg)
        convex = {tuple(sorted(e)): c for e, c in zip(mesh.face_adjacency_edges.tolist(), mesh.face_adjacency_convex)}
        self.names, self.prims = [], []
        for f in faces:
            v = mesh.vertices[np.unique(mesh.faces[face_of_tri == f["id"]])]
            r = np.hypot(v[:, 0], v[:, 2])
            if f["kind"] == "plane" and abs(f["normal"][1]) > 0.99:
                # inner radius: how close the face's triangles come to the axis (a disk need not have a vertex on it)
                T = mesh.triangles[face_of_tri == f["id"]]
                axis_pt = np.array([0.0, f["offset"], 0.0])
                near = closest_on_triangles(axis_pt[None], T[:, 0], T[:, 1], T[:, 2])
                self.prims.append(("plane", f["offset"], float(np.hypot(near[:, 0], near[:, 2]).min()), r.max()))
                self.names.append(f"{name}.plane(y{f['offset']:g})")
            elif f["kind"] == "cylinder" and f.get("segments", 0) >= 12:
                self.prims.append(("cyl", f["radius"], v[:, 1].min(), v[:, 1].max()))
                self.names.append(f"{name}.cyl(r{f['radius']:.1f})")
        pts, lab, run = [], [], []
        for e in edges:
            if e["kind"] != "circle":
                continue
            if np.mean([convex.get(tuple(sorted(m)), True) for m in e["mesh_edges"].tolist()]) < 0.5:
                continue  # a concave edge: touching near it is touching its two faces
            fid = len(self.names)
            self.prims.append(("circle", e["radius"], e["y"]))
            self.names.append(f"{name}.circle(r{e['radius']:.1f},y{e['y']:g})")
            n = int(np.ceil(2 * np.pi * e["radius"] / edge_step))
            a = np.linspace(0, 2 * np.pi, n, endpoint=False)
            pts.append(np.c_[e["radius"] * np.cos(a), np.full(n, e["y"]), e["radius"] * np.sin(a)])
            lab.append(np.full(n, fid))
            run.append(n)
        self.edge_pts, self.edge_lab = np.concatenate(pts), np.concatenate(lab)
        self.runs = np.r_[0, np.cumsum(run)]  # each circle's samples, in order round it

    def distances(self, x):
        """Exact distance of points x (in this part's frame) to each feature, within its extent."""
        rr, y = np.hypot(x[:, 0], x[:, 2]), x[:, 1]
        out = np.empty((len(x), len(self.prims)))
        for k, p in enumerate(self.prims):
            if p[0] == "plane":
                _, h, r0, r1 = p
                dr = np.maximum(np.maximum(r0 - rr, rr - r1), 0)
                out[:, k] = np.hypot(y - h, dr)
            elif p[0] == "cyl":
                _, r, y0, y1 = p
                dy = np.maximum(np.maximum(y0 - y, y - y1), 0)
                out[:, k] = np.hypot(rr - r, dy)
            else:
                _, r, h = p
                out[:, k] = np.hypot(rr - r, y - h)
        return out


def closest_on_segment(p, s0, s1):
    v = s1 - s0
    t = np.clip(((p - s0) * v).sum(-1) / np.maximum((v * v).sum(-1), 1e-30), 0, 1)
    return s0 + t[..., None] * v


def closest_on_triangles(p, a, b, c):
    """Closest point to p on each triangle abc (broadcast): the plane projection if it falls inside,
    else the closest point on the nearest side."""
    n = np.cross(b - a, c - a)
    nn = np.maximum((n * n).sum(-1, keepdims=True), 1e-30)
    q = p - ((p - a) * n).sum(-1, keepdims=True) / nn * n
    inside = ((np.cross(b - a, q - a) * n).sum(-1) >= 0) & ((np.cross(c - b, q - b) * n).sum(-1) >= 0) \
        & ((np.cross(a - c, q - c) * n).sum(-1) >= 0)
    sides = np.stack([closest_on_segment(p, a, b), closest_on_segment(p, b, c), closest_on_segment(p, c, a)])
    ds = np.linalg.norm(sides - p, axis=-1)
    on_side = np.take_along_axis(sides, ds.argmin(0)[None, ..., None], 0)[0]
    return np.where(inside[..., None], q, on_side)


PARTS = {}


def init(cfg):
    PARTS["peg"] = Part(ASSETS / "cylinder_real_peg.stl", "peg", cfg["edge_step"], cfg["sharp_deg"])
    PARTS["hole"] = Part(ASSETS / "cylinder_real_hole.stl", "hole", cfg["edge_step"], cfg["sharp_deg"])
    PARTS["cfg"] = cfg


def edge_contacts(part, x_other_frame, other, tol, prominence, coarse=25, coarse_bound=0.3):
    """Contacts along each of part's circles: local minima of the gap to the other part's features. A
    circle whose every --coarse-th sample is over coarse_bound mm away is skipped (between two such
    samples, ~0.5 mm apart, the gap cannot fall from that to zero)."""
    edge = np.array([p[0] == "circle" for p in other.prims])
    found = []
    for c in range(len(part.runs) - 1):
        s, e = part.runs[c], part.runs[c + 1]
        if other.distances(x_other_frame[s:e:coarse]).min() > coarse_bound:
            continue
        D = other.distances(x_other_frame[s:e])
        g = D.min(1)
        if g.min() >= tol:
            continue
        # a face's closest point on its boundary is the edge there: an edge within 1 um of the closest wins
        # (the fits of a face and of its edge differ by ~0.3 um)
        De = np.where(edge[None], D, np.inf)
        feat = np.where(De.min(1) <= g + 1e-3, De.argmin(1), D.argmin(1))
        shift = int(np.argmax(g))  # start the circle at its largest gap so no minimum straddles the seam
        gg = np.roll(g, -shift)
        peaks, _ = find_peaks(np.r_[-np.inf, -gg, -np.inf], height=-tol, prominence=prominence)
        for p in peaks - 1:
            j = (p + shift) % (e - s)
            found.append((s + j, part.edge_lab[s + j], feat[j], g[j]))
    return found


def classify(poses):
    peg, hole, cfg = PARTS["peg"], PARTS["hole"], PARTS["cfg"]
    out, gaps = [], []
    for p in poses:
        R = Rotation.from_quat(p[3:], scalar_first=True).as_matrix()
        t = p[:3] * 1e3
        P = peg.edge_pts @ R.T + t  # peg edges in the hole frame
        Q = (hole.edge_pts - t) @ R  # hole edges in the peg frame
        from_peg = [(P[i], fp, fh, g) for i, fp, fh, g in edge_contacts(peg, P, hole, cfg["tol"], cfg["prominence"])]
        from_hole = [(hole.edge_pts[i], fp, fh, g) for i, fh, fp, g in edge_contacts(hole, Q, peg, cfg["tol"], cfg["prominence"])]
        # two edges crossing: each sees the other's neighbouring face a few um from the crossing; one contact
        cands, used = [], set()
        for a_ in from_peg:
            near = [j for j, b_ in enumerate(from_hole) if j not in used and np.linalg.norm(a_[0] - b_[0]) < cfg["merge_link"]]
            if near:
                j = min(near, key=lambda j: from_hole[j][3])
                used.add(j)
                cands.append((a_[0], a_[1], from_hole[j][2], min(a_[3], from_hole[j][3])))
            else:
                cands.append(a_)
        cands += [b_ for j, b_ in enumerate(from_hole) if j not in used]
        cands.sort(key=lambda c: c[3])
        kept = []
        for loc, fp, fh, g in cands:
            if not any(k[1] == fp and k[2] == fh and np.linalg.norm(k[0] - loc) < cfg["link"] for k in kept):
                kept.append((loc, fp, fh, g))
        out.append([(int(fp), int(fh), float(g)) for _, fp, fh, g in kept])
        gaps.append(cands[0][3] if cands else np.inf)
    return out, gaps


def load_poses(runs, idx):
    """Sampler poses of the samples idx (sorted ids into the runs' samples, in order)."""
    out = np.empty((len(idx), 7))
    start = 0
    for run in runs:
        for f in sorted(glob.glob(str(Path(run) / "shard_*.npz"))):
            pose = np.load(f)["pose"]
            sel = np.flatnonzero((idx >= start) & (idx < start + len(pose)))
            out[sel] = pose[idx[sel] - start]
            start += len(pose)
    return out


# --- cross-check only: the analytical model's names of the CAD features ---------------------------------
ANALYTIC_NAMES = {"peg.plane(y85.5)": "tip_face", "peg.cyl(r12.5)": "tip_side", "peg.circle(r12.5,y85.5)": "tip_edge",
                  "peg.plane(y35)": "shoulder_face", "peg.circle(r17.5,y35)": "shoulder_edge", "peg.cyl(r17.5)": "shaft_side",
                  "hole.plane(y75)": "top", "hole.circle(r13.0,y75)": "rim_in", "hole.circle(r15.0,y75)": "rim_out",
                  "hole.cyl(r13.0)": "wall", "hole.plane(y25)": "floor", "hole.cyl(r15.0)": "outer_wall",
                  "hole.plane(y35)": "step", "hole.circle(r25.0,y35)": "step_edge", "hole.cyl(r25.0)": "boss_wall"}


def analytic_name(mode):
    if mode == "none":
        return mode
    pairs = [p.split("/") for p in mode.split(" + ")]
    return " + ".join(sorted(f"{ANALYTIC_NAMES.get(a, a)}/{ANALYTIC_NAMES.get(b, b)}" for a, b in pairs))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path, nargs="+")
    parser.add_argument("--samples", type=Path, required=True, help="npz with 'samples' (ids into the runs)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, help="first N samples only")
    parser.add_argument("--tol", type=float, default=0.002)
    parser.add_argument("--prominence", type=float, default=0.0005)
    parser.add_argument("--link", type=float, default=0.5, help="contacts of one pair within this [mm] are one")
    parser.add_argument("--merge-link", type=float, default=0.1,
                        help="a peg-edge and a hole-edge contact within this [mm] are one edge-on-edge contact")
    parser.add_argument("--edge-step", type=float, default=0.02)
    parser.add_argument("--sharp-deg", type=float, default=20.0)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--compare", type=Path, help="npz with 'samples', 'mode', 'mode_names' (analytical modes)")
    args = parser.parse_args()
    ids = np.sort(np.load(args.samples)["samples"])
    ids = ids[:args.limit] if args.limit else ids
    t0 = time.time()
    poses = load_poses(args.run_dir, ids)
    cfg = {k: getattr(args, k) for k in ("tol", "prominence", "link", "merge_link", "edge_step", "sharp_deg")}
    init(cfg)
    print(f"{len(ids):,} poses; peg features {PARTS['peg'].names}; hole features {PARTS['hole'].names} "
          f"({time.time() - t0:.0f} s)", flush=True)
    chunks = np.array_split(np.arange(len(poses)), max(1, len(poses) // 500))
    with mp.Pool(args.workers, initializer=init, initargs=(cfg,)) as pool:
        parts = pool.map(classify, [poses[c] for c in chunks])
    contacts = [c for p in parts for c in p[0]]
    gaps = np.array([g for p in parts for g in p[1]])
    pn, hn = PARTS["peg"].names, PARTS["hole"].names
    modes = [" + ".join(sorted(f"{pn[fp]}/{hn[fh]}" for fp, fh, _ in c)) or "none" for c in contacts]
    names, mode = np.unique(modes, return_inverse=True)
    print(f"{len(names)} CAD modes ({time.time() - t0:.0f} s); smallest gap p50 {np.median(gaps[np.isfinite(gaps)]) * 1e3:.2f} um, "
          f"p99 {np.percentile(gaps[np.isfinite(gaps)], 99) * 1e3:.2f} um; no contact within tol: {np.mean([m == 'none' for m in modes]):.2%}")
    for n, c in sorted(zip(names, np.bincount(mode)), key=lambda x: -x[1])[:20]:
        print(f"  {c / len(mode):6.2%}  {n}")
    np.savez_compressed(args.out, samples=ids, mode=mode, mode_names=names, gap=gaps,
                        n_contacts=np.array([len(c) for c in contacts]))
    args.out.with_suffix(".json").write_text(json.dumps(dict(args={k: str(v) for k, v in vars(args).items()},
                                                             peg_features=pn, hole_features=hn), indent=1))
    if args.compare:
        z = np.load(args.compare)
        ref = dict(zip(z["samples"].tolist(), [str(z["mode_names"][m]) for m in z["mode"]]))
        cad = [analytic_name(str(names[m])) for m in mode]
        an = [ref[i] for i in ids.tolist()]
        same = np.array([a == b for a, b in zip(cad, an)])
        same_set = np.array([sorted(set(a.split(" + "))) == sorted(set(b.split(" + "))) for a, b in zip(cad, an)])
        print(f"agreement with the analytical modes (1 um): exact {same.mean():.2%}, as contact sets {same_set.mean():.2%}")
        from collections import Counter
        for (a, b), c in Counter((a, b) for a, b, s in zip(an, cad, same_set) if not s).most_common(12):
            print(f"  {c / len(an):6.2%}  analytic: {a}   CAD: {b}")


if __name__ == "__main__":
    main()
