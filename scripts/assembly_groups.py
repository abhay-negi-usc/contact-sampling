#!/usr/bin/env python3
"""Group contact states by the trajectory that assembles them: two states are in one group when their
assembly paths can be blended into each other in a straight line without passing through the hole.

Assembly path of a pose: the shortest collision-free path to the assembled pose (tip face on the floor,
centred and upright) through a roadmap of the sampled contact poses and of free poses, in the
symmetry-reduced coordinates x = (rho, depth, L tilt_r, L tilt_t) [mm] (L = --length mm per radian,
roughly how far a tilt moves the peg's lower half); roadmap edges are straight moves between nearest
neighbours, kept if the peg never penetrates the hole along them (analytical separation >= -tol, every
--check-step mm). The path is then shortcut (each corner skipped while the straight move stays free),
which leaves the fewest straight moves needed, and resampled every --spacing mm of arc length back from the
goal (held at the start beyond its length), so paths that share their way in coincide point by point.

Convexity: paths P and Q are compatible if the blends (1 - a) P + a Q (a = 1/(n+1) .. n/(n+1), point by
point) are themselves collision-free: one path deforms into the other through straight-line homotopy, so
the two states are assembled "the same way"; the share of free blends is their compatibility. Groups:
PCCA+ (pcca.py) on the Markov chain that moves between representatives in proportion to compatibility,
its metastable sets being sets of mutually compatible paths, the number from the largest eigengap; and,
for reference, complete-linkage clustering of strict incompatibility (pairwise compatible at every a).
Representatives are random contact poses; every other contact pose takes the group of its nearest one.

The analytical contact modes (peg_hole_analytic.py) are used only afterwards, to read the groups.

    conda run -n mujoco_env python scripts/assembly_groups.py RUN_DIR [RUN_DIR ...] --pool POOL.npz --out-dir OUT
"""

import argparse
import json
import time
from pathlib import Path

import numba as nb
import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

import peg_hole_analytic as A
from codim_aware_scores import merge_bore, metrics, resolve
from knn_pose_spectral import load_samples
from pcca import pcca_scan
from stratum_spectral import short

BOX = np.array([[0.0, 32.0], [-10.0, 10.0], [0.0, 10.0], [-8.0, 50.0]])  # rho, tilt_r, tilt_t, depth
GOAL = np.array([0.0, 0.0, 0.0, 49.99])


@nb.njit(cache=True)
def _sep_q(q0, q1, q2, q3):
    tr, tt = np.radians(q1), np.radians(q2)
    u = np.array([-np.tan(tr), 1.0, -np.tan(tt)])
    u /= np.sqrt(u @ u)
    return A._separation(np.array([q0, A.Y_TOP - q3, 0.0]), u)


@nb.njit(parallel=True, cache=True)
def segments_free(qa, qb, scale, step, tol):
    """Whether the straight move from qa[i] to qb[i] (reduced coords) stays free, checked every step mm
    of the pose metric (scale turns reduced coords into it)."""
    out = np.ones(len(qa), np.bool_)
    for i in nb.prange(len(qa)):
        d = np.sqrt((((qb[i] - qa[i]) * scale) ** 2).sum())
        n = max(2, int(np.ceil(d / step)) + 1)
        for m in range(n):
            t = m / (n - 1)
            q = qa[i] + t * (qb[i] - qa[i])
            if _sep_q(q[0], q[1], abs(q[2]), q[3]) < -tol:
                out[i] = False
                break
    return out


@nb.njit(parallel=True, cache=True)
def blends_free(paths, pi, pj, alphas, tol):
    """How many of the blends (1 - a) P + a Q (a in alphas) of P = paths[pi[k]], Q = paths[pj[k]] are free."""
    out = np.zeros(len(pi), np.int64)
    M = paths.shape[1]
    for k in nb.prange(len(pi)):
        P, Q = paths[pi[k]], paths[pj[k]]
        for a in alphas:
            free = True
            for m in range(M):
                q = (1 - a) * P[m] + a * Q[m]
                if _sep_q(q[0], q[1], abs(q[2]), q[3]) < -tol:
                    free = False
                    break
            out[k] += free
    return out


def free_samples(n, box, rng, min_sep, batch=200_000):
    """n poses drawn uniformly in box (reduced coords) whose separation exceeds min_sep."""
    got = []
    while sum(len(g) for g in got) < n:
        q = rng.uniform(box[:, 0], box[:, 1], (batch, 4))
        c, u = A.pose_from_reduced(q)
        got.append(q[A.separation(c, u) > min_sep])
    return np.concatenate(got)[:n]


def shortcut(path, scale, step, tol, chunk=24):
    """Fewest straight moves: from each kept corner jump to the furthest later point still in sight."""
    keep, i = [0], 0
    while i < len(path) - 1:
        nxt = i + 1
        for hi in range(len(path) - 1, i, -chunk):  # furthest first, a chunk at a time
            cand = np.arange(hi, max(hi - chunk, i), -1)
            ok = segments_free(np.repeat(path[i:i + 1], len(cand), 0), path[cand], scale, step, tol)
            if ok.any():
                nxt = int(cand[np.argmax(ok)])
                break
        i = nxt
        keep.append(i)
    return path[keep]


def from_goal(path, scale, spacing, n):
    """The path as n points every spacing mm of arc length back from the goal (index 0 is the goal),
    held at its start beyond its length: paths that share their way in coincide at equal indices."""
    rev = path[::-1]
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(rev * scale, axis=0), axis=1))]
    s = np.minimum(np.arange(n) * spacing, d[-1])
    return np.stack([np.interp(s, d, rev[:, k]) for k in range(4)], 1), d[-1]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path, nargs="+")
    parser.add_argument("--pool", type=Path, required=True, help="pool with 1 um modes (knn_pose_spectral.py)")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--contact-nodes", type=int, default=40_000)
    parser.add_argument("--free-nodes", type=int, default=60_000)
    parser.add_argument("--bore-nodes", type=int, default=15_000)
    parser.add_argument("--k", type=int, default=12)
    parser.add_argument("--length", type=float, default=50.0, help="mm per radian of tilt in the pose metric")
    parser.add_argument("--check-step", type=float, default=0.05)
    parser.add_argument("--tol", type=float, default=0.005, help="penetration allowed (contact, sliding) [mm]")
    parser.add_argument("--reps", type=int, default=1000)
    parser.add_argument("--spacing", type=float, default=0.5, help="path points every this many mm from the goal")
    parser.add_argument("--alphas", type=int, default=5, help="blends checked per pair, a = 1/(n+1) .. n/(n+1)")
    parser.add_argument("--max-k", type=int, default=12, help="largest number of PCCA+ groups scanned")
    parser.add_argument("--k-groups", type=int, help="PCCA+ groups (default: largest eigengap)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    scale = np.array([1.0, np.radians(1) * args.length, np.radians(1) * args.length, 1.0])
    t0 = time.time()

    roadmap_file = args.out_dir / "roadmap.npz"
    if roadmap_file.exists():
        z = np.load(roadmap_file)
        nodes, n_contact, contact_mode, names = z["nodes"], int(z["n_contact"]), z["contact_mode"], list(z["mode_names"])
        ei, ej, w = z["ei"], z["ej"], z["w"]
    else:
        q_all, _, _ = load_samples(args.run_dir)
        pz = np.load(args.pool)
        pick = np.sort(rng.choice(len(pz["pool"]), args.contact_nodes, replace=False))
        contact = q_all[pz["pool"][pick]].astype(float)
        contact_mode, names = pz["mode"][pick], list(pz["mode_names"])
        free = free_samples(args.free_nodes, BOX, rng, 0.02)
        bore = free_samples(args.bore_nodes, np.array([[0.0, 0.5], [-1.2, 1.2], [0.0, 1.2], [0.0, 49.99]]), rng, 0.0)
        nodes = np.r_[contact, free, bore, GOAL[None]]
        n_contact = len(contact)
        print(f"{len(nodes):,} nodes: {len(contact):,} contact, {len(free):,} free, {len(bore):,} in the bore, goal "
              f"({time.time() - t0:.0f} s)", flush=True)
        x = nodes * scale
        _, nb_ = cKDTree(x).query(x, k=args.k + 1)
        ei, ej = np.repeat(np.arange(len(x)), args.k), nb_[:, 1:].ravel()
        e = np.unique(np.sort(np.c_[ei, ej], 1), axis=0)
        ok = segments_free(nodes[e[:, 0]], nodes[e[:, 1]], scale, args.check_step, args.tol)
        ei, ej = e[ok, 0], e[ok, 1]
        w = np.linalg.norm(x[ei] - x[ej], axis=1)
        print(f"{len(e):,} neighbour moves, {ok.mean():.0%} free ({time.time() - t0:.0f} s)", flush=True)
        np.savez(roadmap_file, nodes=nodes, n_contact=n_contact, contact_mode=contact_mode, mode_names=np.array(names),
                 ei=ei, ej=ej, w=w)
    n = len(nodes)
    G = coo_matrix((np.r_[w, w], (np.r_[ei, ej], np.r_[ej, ei])), shape=(n, n)).tocsr()
    dist, pred = dijkstra(G, indices=n - 1, return_predecessors=True)
    reach = np.isfinite(dist[:n_contact])
    print(f"{reach.mean():.1%} of contact nodes reach the goal ({time.time() - t0:.0f} s)", flush=True)

    reps = np.sort(rng.choice(np.flatnonzero(reach), min(args.reps, reach.sum()), replace=False))
    short_paths, links, lengths, lift = [], [], [], []
    for r in reps:
        chain = [r]
        while chain[-1] != n - 1:
            chain.append(pred[chain[-1]])
        sc = shortcut(nodes[chain], scale, args.check_step, args.tol)
        short_paths.append(sc)
        links.append(len(sc) - 1)
        lengths.append(np.linalg.norm(np.diff(sc * scale, axis=0), axis=1).sum())
        lift.append(sc[:, 3].min())
    n_pts = int(np.ceil(max(lengths) / args.spacing)) + 1
    paths = np.array([from_goal(sc, scale, args.spacing, n_pts)[0] for sc in short_paths])
    print(f"{len(reps)} assembly paths: straight moves p50 {np.median(links):.0f} (max {max(links)}), length p50 "
          f"{np.median(lengths):.1f} mm ({time.time() - t0:.0f} s)", flush=True)

    pi, pj = np.triu_indices(len(reps), 1)
    alphas = np.arange(1, args.alphas + 1) / (args.alphas + 1)
    n_free = blends_free(paths, pi, pj, alphas, args.tol)
    compat = n_free / len(alphas)
    ok = n_free == len(alphas)
    print(f"{len(pi):,} pairs of paths: {ok.mean():.1%} blend without collision at every a, {np.mean(n_free > 0):.1%} "
          f"at some ({time.time() - t0:.0f} s)", flush=True)
    strict = fcluster(linkage((~ok).astype(float), method="complete"), 0.5, criterion="distance") - 1
    # PCCA+ on the chain that moves between representatives in proportion to how well their paths blend
    S = np.zeros((len(reps), len(reps)))
    S[pi, pj] = S[pj, pi] = compat
    S[np.arange(len(reps)), np.arange(len(reps))] = 1.0
    scan, vals = pcca_scan(csr_matrix(S), range(2, args.max_k + 1))
    k_gap = int(np.argmax(vals[1:args.max_k] - vals[2:args.max_k + 1]) + 2)
    print("PCCA+ eigenvalues " + " ".join(f"{v:.3f}" for v in vals[:args.max_k + 1]) + f"; largest gap after k = {k_gap}")
    for kk, (chi, crisp) in scan.items():
        print(f"   k={kk:2d}  crispness {crisp:.3f}  confidently assigned (max membership > 0.8): {np.mean(chi.max(1) > 0.8):.0%}")
    k_use = args.k_groups or k_gap
    chi = scan[k_use][0]
    group_rep = chi.argmax(1)
    print(f"strict convex groups (complete linkage): {strict.max() + 1}; PCCA+ groups: {k_use}", flush=True)
    # every contact node: the group of its nearest representative
    near = cKDTree(nodes[reps] * scale).query(nodes[:n_contact] * scale)[1]
    group = group_rep[near]
    group[~reach] = group_rep.max() + 1

    k = group.max() + 1
    cont = np.zeros((k, len(names)))
    np.add.at(cont, (group, contact_mode), 1)
    base, bnames = resolve(cont, names)
    bb, bbn = merge_bore(base, bnames)
    fam = np.array([A.FAMILIES.index(A.family_of_mode(nm)) for nm in names])
    cf = np.zeros((k, len(A.FAMILIES)))
    np.add.at(cf, (group, fam[contact_mode]), 1)
    report = dict(args={a: str(v) for a, v in vars(args).items()}, groups=[],
                  scores={"base, bore merged": metrics(bb), "family": metrics(cf), "exact mode": metrics(cont)})
    sizes = np.bincount(group, minlength=k) / len(group)
    lab = lambda p: p if " " in p else short(p)
    for g in np.argsort(-sizes):
        if sizes[g] < 0.002:
            continue
        rs = np.flatnonzero(group_rep == g)
        sel = group == g
        share = bb[g] / max(bb[g].sum(), 1)
        fshare = cf[g] / max(cf[g].sum(), 1)
        info = dict(group=int(g), share=float(sizes[g]), representatives=int(len(rs)),
                    straight_moves=float(np.median(np.array(links)[rs])) if len(rs) else None,
                    highest_tip_depth=float(np.median(np.array(lift)[rs])) if len(rs) else None,
                    rho=np.percentile(nodes[:n_contact][sel, 0], [5, 50, 95]).round(1).tolist(),
                    depth=np.percentile(nodes[:n_contact][sel, 3], [5, 50, 95]).round(1).tolist(),
                    families={A.FAMILIES[i]: round(float(fshare[i]), 3) for i in np.argsort(-fshare) if fshare[i] >= 0.02},
                    modes={lab(bbn[i]): round(float(share[i]), 3) for i in np.argsort(-share) if share[i] >= 0.02})
        report["groups"].append(info)
        print(f"group {g:3d} {sizes[g]:6.1%} ({len(rs)} reps; {info['straight_moves']} moves, tip rises to depth "
              f"{info['highest_tip_depth']}) rho {info['rho']} depth {info['depth']}\n      families: "
              + ", ".join(f"{a} {b:.0%}" for a, b in info["families"].items()) + "\n      modes: "
              + ", ".join(f"{a} {b:.0%}" for a, b in info["modes"].items()), flush=True)
    for lvl, v in report["scores"].items():
        print(f"{lvl:18s} ARI {v['ari']:.3f}  homogeneity {v['homogeneity']:.3f}  completeness {v['completeness']:.3f}")
    (args.out_dir / "summary.json").write_text(json.dumps(report, indent=1))
    np.savez_compressed(args.out_dir / "groups.npz", reps=reps, group_rep=group_rep, group=group, paths=paths,
                        links=np.array(links), lengths=np.array(lengths), compat=compat, pi=pi, pj=pj, chi=chi,
                        strict=strict, eigenvalues=vals)


if __name__ == "__main__":
    main()
