#!/usr/bin/env python3
"""Geometry-based discovery of the contact modes: decompose the sampled contact space into pieces that
each lie on one smooth analytic hypersurface, then score the pieces against the known modes.

Why: a contact mode with one contact lies on one constraint hypersurface phi(q) = 0 (one peg feature
against one hole feature), an analytic function of the pose. Neighbouring modes meet tangentially (the
contact normal turns continuously across the boundary, so their union is C1 and distances, normals and
the press wrench barely change there), but each continues along its own equation: the curvature jumps.
One implicit polynomial of low degree fits a whole mode to a micrometre or so and leaves its neighbour
within a fraction of a millimetre of the boundary, so "lies on one polynomial hypersurface" is a
criterion for one mode that needs no labels. A mode with several contacts lies on the intersection of
several hypersurfaces: its samples fit each of its neighbours' polynomials.

Method, an approximate convex decomposition with "flat" generalised to "on one analytic hypersurface",
in the pose metric (rho, depth, L tilt_r, L tilt_t) [mm], L the tip radius:
  1. Over-segment the samples into small patches (k-means).
  2. Split: a patch that does not lie on one implicit polynomial (95th-percentile geometric distance
     above --tol) is bisected, recursively, down to --min-patch points; what still does not fit is left
     over (corners, seams, mixed bits).
  3. Merge: pieces that are neighbours (they share nearest-neighbour pairs: locality only, the decision
     is the fit) are merged greedily, cheapest first, while one polynomial fits their union within
     --tol, each part on its own as well as the whole (so a small piece cannot hide in a large one).
     The degree rises with the piece's size up to --degree (Taubin's gradient-weighted fit, distance
     |f| / |grad f|).
  4. Left-over points join whichever neighbouring piece's hypersurface is closest.
  5. Continuation merge (merge_continuations): touching pieces whose shared boundary one local
     polynomial fits on both sides are one stratum split only because no single low-degree polynomial
     covers it; they merge.
Scoring: against the analytical modes with contacts within 1 um (pool of knn_pose_spectral.py
--contact-tol 0.001), with the codimension rule of codim_aware_scores.py.

    conda run -n mujoco_env python scripts/strata_decomposition.py RUN_DIR [RUN_DIR ...] --pool POOL.npz --out-dir OUT
"""

import argparse
import heapq
import itertools
import json
import time
from pathlib import Path

import numpy as np
from scipy.linalg import eigh
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans, MiniBatchKMeans

import peg_hole_analytic as A
from codim_aware_scores import merge_bore, metrics, resolve
from knn_pose_spectral import REGIONS, load_samples
from stratum_spectral import short

_EXPONENTS = {}


def exponents(d):
    """Monomial exponents in 4 variables up to total degree d, the constant first."""
    if d not in _EXPONENTS:
        _EXPONENTS[d] = np.array(sorted((e for e in itertools.product(range(d + 1), repeat=4) if sum(e) <= d), key=sum))
    return _EXPONENTS[d]


def design(xs, E):
    """Monomials of the points xs and their gradients (one matrix per variable)."""
    P = xs[:, None, :] ** E[None]
    grads = []
    for j in range(4):
        Pj = P.copy()
        Pj[:, :, j] = np.where(E[:, j] > 0, E[:, j] * xs[:, None, j] ** np.maximum(E[:, j] - 1, 0), 0.0)
        grads.append(Pj.prod(2))
    return P.prod(2), grads


def fit(pts, d):
    """Taubin fit of an implicit polynomial of degree d: minimise sum f^2 over sum |grad f|^2."""
    mu, s = pts.mean(0), max(pts.std(0).mean(), 1e-9)
    E = exponents(d)
    M_, G = design((pts - mu) / s, E)
    M = M_.T @ M_ / len(pts)
    N = sum(g.T @ g for g in G) / len(pts)
    Mr = M[1:, 1:] - np.outer(M[0, 1:], M[0, 1:]) / M[0, 0]  # the constant term has no gradient: eliminate it
    N11 = N[1:, 1:] + 1e-12 * np.trace(N[1:, 1:]) * np.eye(len(E) - 1)
    c1 = eigh(Mr, N11, subset_by_index=[0, 0])[1][:, 0]
    return mu, s, E, np.r_[-M[0, 1:] @ c1 / M[0, 0], c1]


def distance(pts, model):
    """Geometric distance |f| / |grad f| of points to a fitted hypersurface [mm]."""
    mu, s, E, c = model
    M_, G = design((pts - mu) / s, E)
    g = np.sqrt(sum((gj @ c) ** 2 for gj in G))
    return np.abs(M_ @ c) / np.maximum(g, 1e-15) * s


class Decomposer:
    def __init__(self, x, tol, max_degree, seed, n_fit=2500, n_eval=4000):
        self.x, self.tol, self.max_degree = x, tol, max_degree
        self.rng = np.random.default_rng(seed)
        self.n_fit, self.n_eval = n_fit, n_eval
        self.fits = 0

    def degree(self, n):
        return min(self.max_degree, 2 if n < 100 else 3 if n < 300 else 4)

    def sub(self, idx, n):
        return idx if len(idx) <= n else self.rng.choice(idx, n, replace=False)

    def model(self, idx):
        self.fits += 1
        return fit(self.x[self.sub(idx, self.n_fit)], self.degree(len(idx)))

    def residual(self, idx, model):
        return float(np.percentile(distance(self.x[self.sub(idx, self.n_eval)], model), 95))

    def split(self, idx, min_patch):
        """Pieces of idx that each fit one polynomial, and the points that fit none."""
        if len(idx) >= 2 * min_patch or len(idx) >= 15:
            if self.residual(idx, self.model(idx)) <= self.tol:
                return [idx], np.array([], int)
        if len(idx) < 2 * min_patch:
            return [], idx
        halves = KMeans(2, n_init=3, random_state=0).fit_predict(self.x[idx])
        pieces, left = [], [np.array([], int)]
        for h in (0, 1):
            p, l = self.split(idx[halves == h], min_patch)
            pieces += p
            left.append(l)
        return pieces, np.concatenate(left)

    def merge_cost(self, a, b):
        """Largest 95th-percentile distance of either part to one polynomial fitted to their union; a small
        piece that is clearly off a large piece's own surface is rejected without the refit."""
        big, small = (a, b) if len(a) >= len(b) else (b, a)
        if len(big) > 4 * len(small) and len(big) >= 300:
            quick = self.residual(small, self.model(big))
            if quick > 3 * self.tol:
                return quick
        m = self.model(np.concatenate([a, b]))
        return max(self.residual(a, m), self.residual(b, m))


def decompose(x, args):
    t0 = time.time()
    D = Decomposer(x, args.tol, args.degree, args.seed)
    n_patch = max(2, len(x) // args.patch)
    patch = MiniBatchKMeans(n_patch, batch_size=8192, n_init=3, random_state=args.seed).fit_predict(x)
    pieces, left = [], []
    for p in range(n_patch):
        idx = np.flatnonzero(patch == p)
        pc, lf = D.split(idx, args.min_patch)
        pieces += pc
        left.append(lf)
    left = np.concatenate(left)
    print(f"{n_patch} patches -> {len(pieces)} pieces on one hypersurface, {len(left)} points left over "
          f"({time.time() - t0:.0f} s, {D.fits} fits)", flush=True)

    label = np.full(len(x), -1)
    for r, idx in enumerate(pieces):
        label[idx] = r
    nb = cKDTree(x).query(x, k=args.k + 1)[1][:, 1:]
    a, b = label[np.repeat(np.arange(len(x)), args.k)], label[nb.ravel()]
    ok = (a >= 0) & (b >= 0) & (a != b)
    pairs, counts = np.unique(np.sort(np.c_[a[ok], b[ok]], 1), axis=0, return_counts=True)
    members = {r: idx for r, idx in enumerate(pieces)}
    neighbours = {r: set() for r in members}
    for (p, q_), c in zip(pairs, counts):
        if c >= args.min_shared:
            neighbours[p].add(q_)
            neighbours[q_].add(p)
    version = {r: 0 for r in members}
    heap = []
    for p in members:
        for q_ in neighbours[p]:
            if p < q_:
                heapq.heappush(heap, (D.merge_cost(members[p], members[q_]), p, q_, 0, 0))
    print(f"{len(heap)} neighbouring pairs costed ({time.time() - t0:.0f} s)", flush=True)
    next_id = len(members)
    merges = 0
    while heap:
        cost, p, q_, vp, vq = heapq.heappop(heap)
        if cost > args.tol:
            break
        if p not in members or q_ not in members or version[p] != vp or version[q_] != vq:
            continue
        r = next_id
        next_id += 1
        members[r] = np.concatenate([members.pop(p), members.pop(q_)])
        neighbours[r] = (neighbours.pop(p) | neighbours.pop(q_)) - {p, q_}
        version[r] = 0
        for n in neighbours[r]:
            neighbours[n] -= {p, q_}
            neighbours[n].add(r)
            heapq.heappush(heap, (D.merge_cost(members[r], members[n]), min(r, n), max(r, n),
                                  version[min(r, n)], version[max(r, n)]))
        merges += 1
        if merges % 200 == 0:
            print(f"  {merges} merges, {len(members)} pieces, largest {max(len(m) for m in members.values())} "
                  f"({time.time() - t0:.0f} s, {D.fits} fits)", flush=True)
    print(f"{merges} merges -> {len(members)} pieces ({time.time() - t0:.0f} s, {D.fits} fits)", flush=True)

    ids = sorted(members, key=lambda r: -len(members[r]))
    label = np.full(len(x), -1)
    for new, r in enumerate(ids):
        label[members[r]] = new
    # left-over points: the closest hypersurface among their neighbours' pieces
    models = {}
    for _ in range(3):
        todo = np.flatnonzero(label < 0)
        if not len(todo):
            break
        for i in todo:
            cands = set(label[nb[i]]) - {-1}
            if not cands:
                continue
            best, best_d = -1, np.inf
            for c in cands:
                if c not in models:
                    models[c] = D.model(np.flatnonzero(label == c))
                dd = distance(x[i:i + 1], models[c])[0]
                if dd < best_d:
                    best, best_d = c, dd
            label[i] = best
    label[label < 0] = label.max() + 1
    return label


def merge_continuations(x, label, args):
    """Second stage: merge neighbouring pieces that are one analytic surface across their shared boundary.
    A stratum too large for one polynomial of --degree ends up as several pieces; across their boundary
    the surface continues analytically, while across a mode boundary the curvature jumps. So for each
    pair of touching pieces, windows of --window-k samples are taken about points of their boundary, one
    polynomial is fitted to each window, and the pair's cost is the --window-pct percentile over windows (the worst, by default) of the larger
    95th-percentile distance of either side. Pairs below --tol are merged (union-find)."""
    t0 = time.time()
    D = Decomposer(x, args.tol, 3, args.seed)
    tree = cKDTree(x)
    nb = tree.query(x, k=args.k + 1)[1][:, 1:]
    a, b = label[np.repeat(np.arange(len(x)), args.k)], label[nb.ravel()]
    ok = a != b
    pairs, counts = np.unique(np.sort(np.c_[a[ok], b[ok]], 1), axis=0, return_counts=True)
    pairs = pairs[counts >= args.min_shared]
    rng = np.random.default_rng(args.seed)
    src = np.repeat(np.arange(len(x)), args.k)[ok]
    costs = []
    for p, q_ in pairs:
        border = np.unique(src[((a[ok] == p) & (b[ok] == q_)) | ((a[ok] == q_) & (b[ok] == p))])
        centres = rng.choice(border, min(args.windows, len(border)), replace=False)
        wc = []
        for c in centres:
            w = tree.query(x[c], k=args.window_k)[1]
            sa, sb = w[label[w] == p], w[label[w] == q_]
            if len(sa) < 20 or len(sb) < 20:
                continue
            m = fit(x[np.r_[sa, sb]], 3)
            wc.append(max(np.percentile(distance(x[sa], m), 95), np.percentile(distance(x[sb], m), 95)))
        costs.append(np.percentile(wc, args.window_pct) if wc else np.inf)
    costs = np.array(costs)
    parent = np.arange(label.max() + 1)

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for (p, q_), c in sorted(zip(pairs.tolist(), costs), key=lambda t: t[1]):
        if c <= args.tol:
            parent[root(p)] = root(q_)
    roots = np.array([root(i) for i in range(len(parent))])
    _, new = np.unique(roots[label], return_inverse=True)
    sizes = np.bincount(new)
    _, new = np.unique(-sizes[new] * (len(sizes) + 1) + new, return_inverse=True)  # largest first
    print(f"continuation merge: {len(pairs)} touching pairs, {int((costs <= args.tol).sum())} continue one surface; "
          f"{label.max() + 1} -> {new.max() + 1} pieces ({time.time() - t0:.0f} s)", flush=True)
    return new, pairs, costs


def report(label, mode, names, q, mixed=0.1, top=12):
    """Scores (exact modes, base modes, base modes with the bore's one-contact modes merged; family, region)
    and the base modes of the largest pieces."""
    k = label.max() + 1
    cont = np.zeros((k, len(names)))
    np.add.at(cont, (label, mode), 1)
    base, bnames = resolve(cont, names)
    bb, bbn = merge_bore(base, bnames)
    fam = np.array([A.FAMILIES.index(A.family_of_mode(n)) for n in names])
    reg = np.array([REGIONS.index(A.REGION_OF_FAMILY[A.FAMILIES[f]]) for f in fam])
    cf, cr = np.zeros((k, len(A.FAMILIES))), np.zeros((k, 3))
    np.add.at(cf, (label, fam[mode]), 1)
    np.add.at(cr, (label, reg[mode]), 1)
    scores = {"exact mode": metrics(cont), "base": metrics(base), "base, bore merged": metrics(bb),
              "family": metrics(cf), "region": metrics(cr)}
    sizes = np.bincount(label) / len(label)
    pieces = []
    share = bb / np.maximum(bb.sum(1, keepdims=True), 1)
    lab = lambda p: p if " " in p else short(p)
    for c in np.argsort(-sizes)[:top]:
        big = [(lab(bbn[i]), round(float(share[c, i]), 3)) for i in np.argsort(-share[c]) if share[c, i] >= 0.03]
        sel = label == c
        pieces.append(dict(piece=int(c), share=float(sizes[c]), base_modes=big,
                           depth=np.percentile(q[sel, 3], [5, 50, 95]).round(1).tolist(),
                           rho=np.percentile(q[sel, 0], [5, 50, 95]).round(1).tolist()))
    impure = float(sum(bb[c].sum() for c in range(k) if (share[c] >= mixed).sum() > 1) / bb.sum())
    return dict(pieces_total=int(k), pieces_over_1pct=int((sizes > 0.01).sum()), scores=scores,
                share_in_mixed_pieces=impure, largest=pieces)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path, nargs="+")
    parser.add_argument("--pool", type=Path, required=True, help="pool cache with 1 um modes (knn_pose_spectral.py)")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--points", type=int, default=100_000)
    parser.add_argument("--tol", type=float, default=0.005, help="largest distance from one hypersurface [mm]")
    parser.add_argument("--degree", type=int, default=4, help="highest polynomial degree")
    parser.add_argument("--patch", type=int, default=150, help="mean points per initial patch")
    parser.add_argument("--min-patch", type=int, default=20)
    parser.add_argument("--k", type=int, default=10, help="nearest neighbours that define which pieces touch")
    parser.add_argument("--min-shared", type=int, default=3)
    parser.add_argument("--length", type=float, default=A.TIP_R)
    parser.add_argument("--windows", type=int, default=8, help="boundary windows per touching pair (stage 2)")
    parser.add_argument("--window-k", type=int, default=1000, help="samples per boundary window (stage 2)")
    parser.add_argument("--window-pct", type=float, default=100, help="percentile over a pair's windows (100: worst)")
    parser.add_argument("--labels", type=Path, help="start stage 2 from a stage-1 labels file instead of decomposing")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"n{args.points}_tol{args.tol * 1e3:g}um_d{args.degree}_p{args.patch}"

    q_all, _, _ = load_samples(args.run_dir)
    z = np.load(args.pool)
    pool, mode_pool, names = z["pool"], z["mode"], list(z["mode_names"])
    pick = np.sort(np.random.default_rng(args.seed).choice(len(pool), args.points, replace=False))
    q, mode = q_all[pool[pick]].astype(float), mode_pool[pick]
    x = np.c_[q[:, 0], q[:, 3], np.radians(q[:, 1]) * args.length, np.radians(q[:, 2]) * args.length]

    if args.labels:
        z1 = np.load(args.labels)
        assert np.array_equal(z1["samples"], pool[pick]), "stage-1 labels are for other samples"
        stages = {"stage1": z1["label"]}
    else:
        stages = {"stage1": decompose(x, args)}
    stages["stage2"], pairs, costs = merge_continuations(x, stages["stage1"], args)
    out = {}
    for stage, label in stages.items():
        rep = report(label, mode, names, q)
        rep["args"] = {k: str(v) for k, v in vars(args).items()}
        out[stage] = rep
        s = rep["scores"]
        print(f"[{tag} {stage}] {rep['pieces_total']} pieces ({rep['pieces_over_1pct']} over 1%), "
              f"{rep['share_in_mixed_pieces']:.1%} of samples in pieces that mix distinct modes")
        for lvl, v in s.items():
            print(f"    {lvl:18s} ARI {v['ari']:.3f}  homogeneity {v['homogeneity']:.3f}  completeness {v['completeness']:.3f}"
                  f"  purity {v['purity']:.3f}")
        for p in rep["largest"]:
            print(f"    piece {p['piece']:3d} {p['share']:6.1%} depth {p['depth']} rho {p['rho']}  "
                  + ", ".join(f"{n} {v:.0%}" for n, v in p["base_modes"]))
    (args.out_dir / f"summary_{tag}.json").write_text(json.dumps(out, indent=1))
    np.savez_compressed(args.out_dir / f"labels_{tag}.npz", samples=pool[pick], label=stages["stage1"],
                        label2=stages["stage2"], mode=mode, mode_names=np.array(names), pairs=pairs, pair_costs=costs)


if __name__ == "__main__":
    main()
