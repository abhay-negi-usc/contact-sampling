#!/usr/bin/env python3
"""Unsupervised spectral clustering of the sampled contact space on a nearest-neighbour pose graph,
scored afterwards against the known contact modes of the peg and hole.

Nothing about the contact modes goes into the clustering: the graph is built from the peg poses alone,
and the analytical modes (peg_hole_analytic.py) are only used to score the clusters it gives.

Nodes: --points samples of one or more sampler runs, in the symmetry-reduced coordinates (rho, radial
tilt, |tangential tilt|, depth), drawn at random (--sampling random, as dense as the sampler was) or one
per cell of a uniform grid in the pose metric (--sampling even, as dense as the contact space is large).
Pose distance: d^2 = d_rho^2 + d_depth^2 + (L d_tilt_r)^2 + (L d_tilt_t)^2 with the tilts in radians and
L (--length) the lever that turns a tilt into a displacement, by default the tip radius: how far a tilt
moves the tip edge.
Edges: each node's --k nearest neighbours in that metric and nothing else (no long-range partners, no
trajectory links). Weights from the pose distance: exp(-d^2 / 2 s^2) with s the median neighbour
distance (--weights gauss), or exp(-d^2 / s_i s_j) with s_i the distance to node i's 7th neighbour
(--weights local, the self-tuning scale, which follows the changes in sampling density).
Wrench (--features): the sampler's measured contact wrench, in the reduced frame (tip on +x, mirrored
with the tangential tilt) with the torque about the tip centre over the tip radius, joins the pose as a
second block of coordinates: the whole wrench, its direction, or the force direction. The block is
scaled so that, between pose neighbours, its median difference is --wrench-weight times their median
pose distance: where the wrench jumps (a contact made or broken, a normal turning) poses that are close
become far apart.
Stability (--stable-only): only poses that hold under a press, kept from a random --pool of samples. A
pose holds if its contacts, each pushing along its normal within its Coulomb cone (--friction), can
balance a press straight down into the hole, the robot holding the orientation (so supplying any
moment): the contact wrench, not the contact mode, decides. A tip edge on a rim is held only while its
normal is within the friction angle of vertical; a tip sliding down a wall is not held.
Clustering: normalized Laplacian on the largest connected component, k from its largest eigengap among
the first --max-k, k-means on the row-normalised eigenvectors (Ng, Jordan and Weiss); also at each
--fixed-k for reference (choosing those k uses knowledge of the answer).
Scoring: --eval random samples take the cluster of their nearest node, so scores are per sample as in
stratum_spectral.py; ARI, homogeneity, completeness, V-measure (NMI) and purity against the regions
(inside the bore, on the mouth, outside the tube), the families (peg_hole_analytic.FAMILIES) and the
fine modes, and each family's best cluster.

    conda run -n mujoco_env python scripts/knn_pose_spectral.py RUN_DIR [RUN_DIR ...] --out-dir OUT
"""

import argparse
import glob
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
from scipy.optimize import lsq_linear
from scipy.spatial import cKDTree
from sklearn.metrics import adjusted_rand_score, homogeneity_completeness_v_measure
from sklearn.metrics.cluster import contingency_matrix

import peg_hole_analytic as A
from compare_samples_to_analytic import SAMPLE_TOL, SAMPLE_TOL_BORE
from mode_discovery_bench import knn_graph, reduced_frame, spectral_labels
from plot_contact_samples_cylindrical import load_reduced
from stratum_spectral import sample_fine_modes, short

REGIONS = ["inside", "mouth", "outside"]


def load_samples(runs):
    """Reduced coords of every sample of the runs, and its analytical mode id into one list of names."""
    qs, modes, names = [], [], {}
    for run in runs:
        q = load_reduced(run)
        ids, run_names = sample_fine_modes(run, q)
        remap = np.array([names.setdefault(n, len(names)) for n in run_names], np.int32)
        qs.append(q)
        modes.append(remap[ids])
    return np.concatenate(qs), np.concatenate(modes), sorted(names, key=names.get)


def measured_wrench(runs, idx):
    """Sampler wrench of the samples idx (sorted ids into the runs' samples, in order) in the reduced
    frame: force [N] and torque about the tip centre / tip radius [N], mirrored where the tangential tilt
    is negative (as |tilt_t| is), and the reduced coords it implies (a check on the sample order)."""
    out, q_out = np.empty((len(idx), 6)), np.empty((len(idx), 4))
    start = 0
    for run in runs:
        for f in sorted(glob.glob(str(Path(run) / "shard_*.npz"))):
            z = np.load(f)
            pose = z["pose"]
            sel = np.flatnonzero((idx >= start) & (idx < start + len(pose)))
            if len(sel):
                p, w = pose[idx[sel] - start].astype(float), z["wrench"][idx[sel] - start].astype(float)
                q, rot, to_reduced, tip = reduced_frame(p)
                F = rot.apply(w[:, :3])
                T = rot.apply(w[:, 3:]) + np.cross(p[:, :3] - tip * 1e-3, F)  # about the tip centre [N m]
                Fr, Tr = to_reduced.apply(F), to_reduced.apply(T) / (A.TIP_R * 1e-3)
                flip = q[:, 2] < 0  # mirror z: a force's z and a moment's x, y change sign
                Fr[flip, 2] *= -1
                Tr[flip, :2] *= -1
                out[sel] = np.c_[Fr, Tr]
                q_out[sel] = np.c_[q[:, :2], np.abs(q[:, 2]), q[:, 3]]
            start += len(pose)
    return out, q_out


def press_test(q, friction, tol, tol_bore, press=10.0, n_edges=8):
    """Whether each pose holds under a press straight down, the orientation held (its contacts' friction
    cones can make an upward force), and its friction-free press wrench: the press shared equally along
    the contact normals, in the reduced frame (which pose_from_reduced works in), torque about the tip
    centre over the tip radius. Contacts: within tol of touching (tol_bore with the tip in the bore)."""
    c, u = A.pose_from_reduced(np.asarray(q, float))
    a = np.linspace(0, 2 * np.pi, n_edges, endpoint=False)
    out, wrench = np.zeros(len(q), bool), np.zeros((len(q), 6))
    for i in range(len(q)):
        rows = A.contact_details(c[i], u[i], tol_bore if q[i, 3] > 0 else tol)
        if len(rows) == 0:
            continue
        n = rows[:, 5:8]
        f = press / len(rows) * n
        wrench[i] = np.r_[f.sum(0), np.cross(rows[:, 2:5] - c[i], f).sum(0) / A.TIP_R]
        t1 = np.cross(n, np.where(np.abs(n[:, :1]) < 0.9, [[1.0, 0, 0]], [[0, 0, 1.0]]))
        t1 /= np.linalg.norm(t1, axis=1, keepdims=True)
        t2 = np.cross(n, t1)
        G = (n[:, None] + friction * (np.cos(a)[None, :, None] * t1[:, None] + np.sin(a)[None, :, None] * t2[:, None]))
        G = G.reshape(-1, 3).T
        up = np.array([0.0, 1.0, 0.0])
        out[i] = np.linalg.norm(G @ lsq_linear(G, up, bounds=(0, np.inf), method="bvls").x - up) < 1e-3
    return out, wrench


def press_test_parallel(q, friction, tol, tol_bore, chunk=5000):
    with mp.Pool(min(20, mp.cpu_count())) as pool:
        parts = pool.starmap(press_test, [(q[s:s + chunk], friction, tol, tol_bore) for s in range(0, len(q), chunk)])
    return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])


def contact_modes_at(q, tol, chunk=500_000):
    """Analytical contact mode of each pose with contacts counted within tol: ids into a list of names."""
    names, ids = {}, np.empty(len(q), np.int32)
    for i in range(0, len(q), chunk):
        c, u = A.pose_from_reduced(np.asarray(q[i:i + chunk], float))
        _, _, pairs = A.contact_modes(c, u, tol, tol)
        uniq, inv = np.unique(pairs, axis=0, return_inverse=True)
        ids[i:i + chunk] = np.array([names.setdefault(A.mode_name(p), len(names)) for p in uniq], np.int32)[inv.ravel()]
    return ids, sorted(names, key=names.get)


FEATURES = ["pose", "wrench", "wrench-dir", "force-dir", "press-wrench", "press-wrench-dir", "press-force-dir",
            "press-frame", "press-force-dir+frame"]


def frame_residuals(f, u):
    """How far the force is from a direction fixed to the peg (along or across its axis u: a peg face or
    side touching) and from one fixed to the hole (vertical or horizontal: a hole face touching) [rad]."""
    f = f / np.maximum(np.linalg.norm(f, axis=1, keepdims=True), 1e-9)
    a = np.arcsin(np.clip(np.abs((f * u).sum(1)), 0, 1))  # elevation from the peg's cross-section
    b = np.arcsin(np.clip(np.abs(f[:, 1]), 0, 1))  # elevation from the hole's horizontal plane
    return np.c_[np.minimum(a, np.pi / 2 - a), np.minimum(b, np.pi / 2 - b)]


def wrench_block(measured, pressed, kind, q):
    """The wrench coordinates of a --features choice: measured (sliding, with friction) or pressed."""
    w = pressed if kind.startswith("press-") else measured
    kind = kind.removeprefix("press-")
    if kind == "wrench":
        return w
    frame = frame_residuals(w[:, :3], A.pose_from_reduced(np.asarray(q, float))[1]) if "frame" in kind else None
    if kind == "frame":
        return frame
    v = w if kind == "wrench-dir" else w[:, :3]
    v = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)
    return np.c_[v, frame] if frame is not None else v


def pose_metric(q, length):
    """Coordinates in which the Euclidean distance is the pose distance [mm]."""
    x = np.asarray(q, np.float64).copy()
    x[:, 1:3] *= np.radians(1.0) * length
    return x


def even_subset(x, n, rng, iters=12):
    """About n indices, one at random from each occupied cell of a uniform grid sized to give n."""
    lo = x.min(0)
    perm = rng.permutation(len(x))

    def cells(h):
        idx = np.floor((x[perm] - lo) / h).astype(np.int64)
        dims = idx.max(0) + 1
        key = np.ravel_multi_index(idx.T, dims)
        return np.unique(key, return_index=True)[1]

    a, b = 1e-3, float(np.ptp(x, 0).max())
    for _ in range(iters):  # bisect the cell size on a log scale
        h = np.sqrt(a * b)
        first = cells(h)
        a, b = (h, b) if len(first) > n else (a, h)
    first = cells(b)
    pick = perm[first]
    return np.sort(rng.choice(pick, n, replace=False) if len(pick) > n else pick), b


def edge_weights(x, i, j, dd, kind, k_scale=7):
    if kind == "gauss":
        return np.exp(-dd ** 2 / (2 * np.median(dd) ** 2))
    s = cKDTree(x).query(x, k=k_scale + 1)[0][:, -1]
    return np.exp(-dd ** 2 / (s[i] * s[j]))


def scores(truth, labels):
    h, c, v = homogeneity_completeness_v_measure(truth, labels)
    cont = contingency_matrix(truth, labels)
    return dict(ari=float(adjusted_rand_score(truth, labels)), homogeneity=float(h), completeness=float(c),
                v_measure=float(v), purity=float(cont.max(0).sum() / cont.sum()))


def composition(labels, mode, fam, q, names, top=8):
    """What each cluster holds: its families, its largest modes, and the pose range it covers."""
    out = []
    for c in np.unique(labels):
        sel = labels == c
        f = np.bincount(fam[sel], minlength=len(A.FAMILIES)) / sel.sum()
        m = np.bincount(mode[sel], minlength=len(names)) / sel.sum()
        lo, med, hi = np.percentile(q[sel], [5, 50, 95], axis=0)
        out.append(dict(cluster=int(c), share=float(sel.mean()),
                        families={A.FAMILIES[i]: round(float(f[i]), 3) for i in np.argsort(-f) if f[i] > 0.01},
                        modes={short(names[i]): round(float(m[i]), 3) for i in np.argsort(-m)[:top] if m[i] > 0.01},
                        pose={k: [round(float(a), 1), round(float(b), 1), round(float(d), 1)] for k, a, b, d in
                              zip(["rho", "tilt_r", "tilt_t", "depth"], lo, med, hi)}))
    return out


def per_family(fam, labels):
    """Each family's share, its spread over the clusters, and its best cluster's recall and precision."""
    out = {}
    cont = contingency_matrix(fam, labels)  # families x clusters, rows in sorted family id order
    fam_ids, clusters = np.unique(fam), np.unique(labels)
    for r, f in enumerate(fam_ids):
        row = cont[r]
        b = int(row.argmax())
        rec, prec = row[b] / row.sum(), row[b] / cont[:, b].sum()
        out[A.FAMILIES[f]] = dict(share=float(row.sum() / cont.sum()), best=int(clusters[b]), recall=float(rec),
                                  precision=float(prec), f1=float(2 * rec * prec / (rec + prec)),
                                  spread={int(clusters[c]): round(float(row[c] / row.sum()), 3)
                                          for c in np.flatnonzero(row / row.sum() > 0.01)})
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path, nargs="+", help="one or more sampler runs, analysed together")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--points", type=int, default=200_000, help="graph nodes")
    parser.add_argument("--sampling", choices=["random", "even"], default="random")
    parser.add_argument("--length", type=float, default=A.TIP_R, help="lever turning a tilt into mm [mm/rad]")
    parser.add_argument("--k", type=int, default=15, help="nearest neighbours per node")
    parser.add_argument("--weights", choices=["gauss", "local"], default="gauss")
    parser.add_argument("--features", choices=FEATURES, default="pose")
    parser.add_argument("--wrench-weight", type=float, default=1.0)
    parser.add_argument("--stable-only", action="store_true", help="only poses that hold under a press")
    parser.add_argument("--friction", type=float, default=0.3)
    parser.add_argument("--contact-tol", type=float, help="count contacts within this gap [mm], for the press wrench "
                                                          "and the modes scored against (default: 5 um, 20 um in the bore)")
    parser.add_argument("--pool", type=int, default=2_000_000,
                        help="random samples given a wrench and a stability test (with --features or --stable-only)")
    parser.add_argument("--max-k", type=int, default=10)
    parser.add_argument("--fixed-k", type=int, nargs="*", default=[3, 6])
    parser.add_argument("--eval", type=int, default=1_000_000, help="random samples scored")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tag = (f"{args.sampling}_{args.weights}_L{args.length:g}_k{args.k}_n{args.points}"
           + (f"_{args.features}{args.wrench_weight:g}" if args.features != "pose" else "")
           + ("_stable" if args.stable_only else "")
           + (f"_tol{args.contact_tol * 1e3:g}um" if args.contact_tol else ""))
    rng = np.random.default_rng(args.seed)

    t0 = time.time()
    q, mode, names = load_samples(args.run_dir)
    x = pose_metric(q, args.length)
    print(f"{len(q):,} samples, {len(names)} modes ({time.time() - t0:.0f} s)", flush=True)
    extra = None
    if args.features != "pose" or args.stable_only or args.contact_tol:
        pool = np.sort(np.random.default_rng(args.seed).choice(len(q), min(args.pool, len(q)), replace=False))
        tol, tol_bore = (args.contact_tol, args.contact_tol) if args.contact_tol else (SAMPLE_TOL, SAMPLE_TOL_BORE)
        cache = args.out_dir / (f"pool_seed{args.seed}_n{len(pool)}_mu{args.friction:g}"
                                + (f"_tol{args.contact_tol * 1e3:g}um" if args.contact_tol else "") + ".npz")
        if cache.exists():
            z = np.load(cache)
            wrench, pressed, stable = z["wrench"], z["pressed"], z["stable"]
            pool_mode, pool_names = (z["mode"], list(z["mode_names"])) if args.contact_tol else (None, None)
        else:
            wrench, q_check = measured_wrench(args.run_dir, pool)
            print(f"wrench of {len(pool):,} pool samples; their reduced coords differ from the cache by at most "
                  f"{np.abs(q_check - q[pool]).max():.2e} ({time.time() - t0:.0f} s)", flush=True)
            stable, pressed = press_test_parallel(q[pool], args.friction, tol, tol_bore)
            pool_mode, pool_names = contact_modes_at(q[pool], tol) if args.contact_tol else (None, None)
            extra_fields = dict(mode=pool_mode, mode_names=np.array(pool_names)) if args.contact_tol else {}
            np.savez(cache, pool=pool, wrench=wrench, pressed=pressed, stable=stable, **extra_fields)
        print(f"{stable.mean():.1%} of the pool holds under a press ({time.time() - t0:.0f} s)", flush=True)
        if args.contact_tol:  # score against the modes at this tolerance (only pool samples are ever scored)
            mode = np.full(len(q), -1, np.int32)
            mode[pool], names = pool_mode, pool_names
            print(f"{len(names)} modes with contacts within {args.contact_tol * 1e3:g} um", flush=True)
        keep = stable if args.stable_only else np.ones(len(pool), bool)
        cand = pool[keep]
        if args.features != "pose":
            block = wrench_block(wrench, pressed, args.features, q[pool])
            extra = lambda s: block[np.searchsorted(pool, s)]  # samples drawn from the pool only
    else:
        cand = np.arange(len(q))
    if args.sampling == "random":
        nodes, cell = np.sort(rng.choice(cand, args.points, replace=False)), None
    else:
        sub, cell = even_subset(x[cand], args.points, rng)
        nodes = cand[sub]
    xn = x[nodes]
    if extra is not None:  # scale the wrench block against the pose distance between pose neighbours
        i, j, dd = knn_graph(xn, args.k)
        dw = np.linalg.norm(extra(nodes)[i] - extra(nodes)[j], axis=1)
        scale = args.wrench_weight * np.median(dd) / max(np.median(dw), 1e-12)
        x_of = lambda s: np.c_[x[s], scale * extra(s)]
        print(f"wrench block scale {scale:.3g} mm per unit (median neighbour difference {np.median(dw):.3g})")
    else:
        x_of = lambda s: x[s]
    xn = x_of(nodes)
    i, j, dd = knn_graph(xn, args.k)
    w = edge_weights(xn, i, j, dd, args.weights)
    print(f"{len(nodes):,} nodes" + (f" (grid cell {cell:.3f} mm)" if cell else "") + f", {len(i):,} edges, "
          f"median neighbour distance {np.median(dd):.3f} mm ({time.time() - t0:.0f} s)", flush=True)

    family_of_name = np.array([A.FAMILIES.index(A.family_of_mode(n)) for n in names])
    region_of_name = np.array([REGIONS.index(A.REGION_OF_FAMILY[A.FAMILIES[f]]) for f in family_of_name])
    ev = np.sort(rng.choice(cand, min(args.eval, len(cand)), replace=False))
    nearest = cKDTree(xn).query(x_of(ev))[1]
    truth = {"region": region_of_name[mode[ev]], "family": family_of_name[mode[ev]], "mode": mode[ev]}

    report = dict(args={k: str(v) for k, v in vars(args).items()}, nodes=len(nodes), edges=len(i),
                  median_neighbour_mm=float(np.median(dd)), grid_cell_mm=cell, runs=[])
    saved = {}
    for k in [None] + args.fixed_k:
        t1 = time.time()
        labels, vals = spectral_labels(len(nodes), i, j, w, args.max_k, k, eig_tol=1e-6, ncv=6 * (args.max_k + 1))
        k_used = int(labels.max() + 1)
        lab = labels[nearest]
        lab[lab < 0] = k_used  # nodes off the largest component form one more group
        run = dict(k=k_used, chosen="eigengap" if k is None else "fixed",
                   eigenvalues=[round(float(v), 5) for v in vals], off_component=float((labels < 0).mean()),
                   cluster_share={c: round(float(s), 3) for c, s in enumerate(np.bincount(lab) / len(lab)) if s},
                   scores={lvl: scores(t, lab) for lvl, t in truth.items()},
                   families=per_family(truth["family"], lab),
                   clusters=composition(lab, truth["mode"], truth["family"], q[ev], names))
        report["runs"].append(run)
        saved[f"k{k_used}_{run['chosen']}"] = lab.astype(np.int8)
        s = run["scores"]
        print(f"[{tag}] k = {k_used} ({run['chosen']}) eigenvalues " + " ".join(f"{v:.4f}" for v in vals)
              + f" ({time.time() - t1:.0f} s)", flush=True)
        for lvl in truth:
            print(f"    {lvl:7s} ARI {s[lvl]['ari']:.3f}  homogeneity {s[lvl]['homogeneity']:.3f}  completeness "
                  f"{s[lvl]['completeness']:.3f}  V {s[lvl]['v_measure']:.3f}  purity {s[lvl]['purity']:.3f}")
        for f, r in run["families"].items():
            print(f"    {f:18s} {r['share']:.3f}  best {r['best']} recall {r['recall']:.2f} precision "
                  f"{r['precision']:.2f} F1 {r['f1']:.2f}  spread {r['spread']}")
    (args.out_dir / f"summary_{tag}.json").write_text(json.dumps(report, indent=1))
    np.savez_compressed(args.out_dir / f"labels_{tag}.npz", samples=ev, modes=mode[ev], mode_names=np.array(names),
                        **saved)  # sample ids into the runs, in order, and the modes they were scored against


if __name__ == "__main__":
    main()
