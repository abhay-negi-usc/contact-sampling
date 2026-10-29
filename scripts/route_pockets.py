#!/usr/bin/env python3
"""Route every pocket of the goal field to the goal through the line-of-sight roadmap.

Pockets are the collision-free nodes of the goal field (goal_field.py) not connected to the goal inside
the field's tube: their way out runs outside it, or through a diagonal move the field's single-axis
moves miss. Each pocket is first classified: if the hook's shaft passes through the eyelet's hole at any
of its nodes, the hook would have to slide its shaft out of the loop, a case that is not encountered and
is skipped. The shaft is the hook's centreline (its surface moved inward by the wire radius) above
--shaft-z mm; the hole is the free region enclosed by the loop in its plane x = 0 of the eyelet frame.

Every other pocket gets --local samples around the node with the most clearance (its start), joined by
line of sight to their nearest neighbours in the roadmap of hook_mode_graph.py (graph.npz) and in the
pocket's own samples. One widest-path tree grown from the goal over the whole roadmap then gives each
pocket the route that keeps the most clearance; the route is shortcut, the final lift into contact
appended, and checked densely (every 0.1 mm / deg).

Output in --out-dir: pocket_routes.json (per pocket: shaft through the hole?, status, clearances) and
pocket_routes.npz (each routed pocket's route as poses, start first).

    conda run -n mujoco_env python scripts/route_pockets.py FIELD_DIR --graph MODE_GRAPH_DIR/graph.npz
"""

import argparse
import heapq
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import trimesh
from scipy import ndimage
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

import validate_goal_field as vg
from hook_eyelet_model import ASSET_DIR, EYELET_YAW_DEG, HOOK_YAW_DEG, INCH
from hook_mode_graph import SCALE, _init_worker, edge_task, sample_task


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


class ShaftThroughHole:
    """Does the hook's shaft cross the eyelet's loop plane inside its hole?"""

    def __init__(self, shaft_z_mm, res=0.25):
        rz = lambda deg: trimesh.transformations.rotation_matrix(np.deg2rad(deg), [0, 0, 1])
        hook = trimesh.load(ASSET_DIR / "hook.stl")
        hook.apply_scale(INCH * 1e3)
        hook.apply_transform(rz(HOOK_YAW_DEG))
        radius = np.ptp(hook.vertices[:, 1]) / 2
        # Dense surface samples: an STL's straight runs have vertices only at their ends.
        pts, face = trimesh.sample.sample_surface_even(hook, 60000, seed=0)
        centre = pts - radius * hook.face_normals[face]
        # The shaft rises on the -x side of the hook frame; the tip arm (+x) threads the hole normally.
        self.shaft = centre[(centre[:, 2] > shaft_z_mm) & (centre[:, 0] < 5.0)]
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

    def __call__(self, pose, band=0.5):
        """pose: x, y, z [mm], rotation vector [deg]."""
        p = Rotation.from_rotvec(pose[3:], degrees=True).apply(self.shaft) + pose[:3]
        near = p[np.abs(p[:, 0]) < band]
        iy = np.rint((near[:, 1] - self.ys[0]) / self.res).astype(int)
        iz = np.rint((near[:, 2] - self.zs[0]) / self.res).astype(int)
        ok = (iy >= 0) & (iy < len(self.ys)) & (iz >= 0) & (iz < len(self.zs))
        return bool(np.any(self.hole[iz[ok], iy[ok]]))


def widest_tree(n, edges, clear, root):
    """Max-min clearance from every node to root over the graph (ties: fewest edges); parent pointers."""
    adj = [[] for _ in range(n)]
    for (i, j), c in zip(edges, clear):
        adj[i].append((j, c))
        adj[j].append((i, c))
    best = np.full(n, -np.inf)
    hops = np.full(n, np.iinfo(np.int64).max)
    parent = np.full(n, -1)
    best[root], hops[root] = np.inf, 0
    heap = [(-np.inf, 0, root)]
    while heap:
        nb, h, u = heapq.heappop(heap)
        if -nb < best[u] or (-nb == best[u] and h > hops[u]):
            continue
        for v, c in adj[u]:
            b = min(-nb, c)
            if b > best[v] or (b == best[v] and h + 1 < hops[v]):
                best[v], hops[v], parent[v] = b, h + 1, u
                heapq.heappush(heap, (-b, h + 1, v))
    return best, parent


def finish_route(job):
    """Shortcut a roadmap route, append the final lift into contact and check it densely."""
    p, pts, required, step, bottleneck = job
    short = vg.shortcut(pts, required, step)
    full = np.vstack([short, np.zeros(6)[None]])
    checked = vg.path_min(short, 0.1)
    return p, full, {"status": "valid" if checked >= 0 else "invalid", "roadmap_bottleneck_mm": bottleneck,
                     "checked_min_mm": float(checked), "segments": len(short) - 1, "length": vg.arc_length(full)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("field_dir", type=Path)
    parser.add_argument("--graph", type=Path, required=True, help="graph.npz from hook_mode_graph.py")
    parser.add_argument("--out-dir", type=Path, help="default: next to the graph")
    parser.add_argument("--local", type=int, default=200, help="samples around each pocket start")
    parser.add_argument("--local-half", type=float, default=6.0, help="their box half-size [mm and deg]")
    parser.add_argument("--k", type=int, default=15)
    parser.add_argument("--step", type=float, default=0.25)
    parser.add_argument("--shaft-z", type=float, default=15.0, help="hook centreline above this is the shaft [mm]")
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    out = args.out_dir or args.graph.parent
    t_all = time.time()

    pz = np.load(args.field_dir / "goal_field_pockets.npz")
    coords, comp, psep = pz["coords"].astype(float), pz["comp"], pz["sep"]
    pockets = np.unique(comp)
    shaft = ShaftThroughHole(args.shaft_z)
    info = {}
    rng = np.random.default_rng(args.seed)
    for p in pockets:
        m = np.flatnonzero(comp == p)
        start = coords[m[np.argmax(psep[m])]]
        probe = np.r_[m[np.argmax(psep[m])], rng.choice(m, min(len(m), 50), replace=False)]
        info[int(p)] = {"nodes": len(m), "start": start.tolist(), "start_sep_mm": float(psep[m].max()),
                        "shaft_through_hole": any(shaft(coords[i]) for i in probe)}
    feasible = [p for p in info if not info[p]["shaft_through_hole"]]
    print(f"{len(pockets)} pockets: {len(pockets) - len(feasible)} with the shaft through the hole (skipped), "
          f"{len(feasible)} to route", flush=True)

    g = np.load(args.graph)
    poses, seps, edges, clear = g["poses"], g["seps"], g["edges"], g["clear"]
    goal = int(np.argmin(np.abs(poses - np.array([0, 0, -3, 0, 0, 0])).sum(1)))
    vg._init()
    jobs = [(np.array(info[p]["start"])[None], args.local_half, args.local, 5.0, 1.0, args.seed + p) for p in feasible]
    t0 = time.time()
    with mp.get_context("fork").Pool(args.workers, initializer=_init_worker) as pool:
        res = pool.map(sample_task, jobs)
    starts = np.array([info[p]["start"] for p in feasible])
    new = np.vstack([starts] + [r[0] for r in res])
    new_sep = np.concatenate([vg._sep(starts)] + [r[1] for r in res])
    base = len(poses)
    poses, seps = np.vstack([poses, new]), np.concatenate([seps, new_sep])
    start_idx = {p: base + i for i, p in enumerate(feasible)}
    tree = cKDTree(poses * SCALE)
    _, nbr = tree.query(new * SCALE, k=args.k + 1)
    src = np.repeat(np.arange(base, len(poses)), args.k)
    pairs = np.unique(np.sort(np.column_stack([src, nbr[:, 1:].ravel()]), 1), axis=0)
    chunks = np.array_split(np.arange(len(pairs)), args.workers * 8)
    with mp.get_context("fork").Pool(args.workers, initializer=_init_worker) as pool:
        res = pool.map(edge_task, [(poses[pairs[c, 0]], poses[pairs[c, 1]], args.step) for c in chunks])
    c_new = np.concatenate(res)
    edges = np.vstack([edges, pairs[c_new >= 0]])
    clear = np.concatenate([clear, c_new[c_new >= 0]])
    print(f"{len(new):,} pocket samples, {len(pairs):,} new pairs, {np.mean(c_new >= 0):.0%} in line of sight "
          f"({time.time() - t0:.0f} s)", flush=True)

    t0 = time.time()
    best, parent = widest_tree(len(poses), edges, clear, goal)
    print(f"widest-path tree over {len(poses):,} samples ({time.time() - t0:.0f} s)", flush=True)
    jobs = []
    for p in feasible:
        i = start_idx[p]
        if not np.isfinite(best[i]) or parent[i] < 0:
            info[p]["status"] = "no route"
            continue
        path = [i]
        while path[-1] != goal:
            path.append(parent[path[-1]])
        jobs.append((p, poses[path], max(0.0, min(best[i], seps[i]) - 0.1), args.step, float(best[i])))
    t0 = time.time()
    with mp.get_context("fork").Pool(args.workers, initializer=_init_worker) as pool:
        done = pool.map(finish_route, jobs)
    routes = {}
    for p, full, entry in done:
        info[p].update(entry)
        routes[f"pocket_{p}"] = full
    print(f"{len(done)} routes shortcut and checked ({time.time() - t0:.0f} s)", flush=True)
    counts = {}
    for p in feasible:
        counts[info[p]["status"]] = counts.get(info[p]["status"], 0) + 1
    summary = {"pockets": len(pockets), "shaft_through_hole": len(pockets) - len(feasible), "routed": counts,
               "nodes_in_valid_pockets": sum(info[p]["nodes"] for p in feasible if info[p].get("status") == "valid"),
               "seconds": round(time.time() - t_all)}
    (out / "pocket_routes.json").write_text(json.dumps({"summary": summary, "pockets": info}, indent=2))
    np.savez(out / "pocket_routes.npz", **routes)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
