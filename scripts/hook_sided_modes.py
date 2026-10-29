#!/usr/bin/env python3
"""Coarser contact modes of the hook on the eyelet, each on one side of the loop's mid-plane and in one
threading state.

The contact poses (mode_graph_experiments.py prepare-contact) fall into classes by the side of the loop's
mid-plane their contact point lies on (+y, -y, or the middle within --middle-mm of it) and their threading
state (hook_contact_strata.contact_class); a mode never mixes classes, and no pose joins a mode by pose
distance alone. Within each class, the sliding graph
(neighbour edges whose segment stays within --slide-mm of contact; with --wrench weighted by how much the
wrench direction turns along them) is split by one of the earlier techniques:

    spectral  normalized-Laplacian spectral clustering of each connected piece, k at the largest eigengap
              (k = 1 when the first gap is the largest; at most one mode per --min-mode poses)
    pcca      PCCA+ memberships of the random walk on each piece, k the largest whose crispness reaches
              --crisp (else 1)
    lump      PCCA+ on the formation modes (hook_contact_strata.py formations, --formations TAG): their
              sliding-edge counts form a small chain per class whose metastable sets are unions of
              formation modes; k the largest (up to all of them) whose crispness reaches --crisp, else the
              formation modes stay apart

Pieces below --min-mode poses stay unassigned. Every result is audited (one side and one threading state
per mode) and scored like the graph variants, also against the physical label with the side added; the
labels go to labels_contact_<name>.npy, the modes' descriptions to strata/<name>.json, a row to
experiments.json.

    python scripts/hook_sided_modes.py EXPERIMENT_DIR --method spectral [--wrench]
    python scripts/hook_sided_modes.py EXPERIMENT_DIR --method lump --formations formations_sided_normal_min100
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

import mode_graph_experiments as mge
from contact_graph_spectral import cluster
from hook_contact_strata import (HALF_NAMES, MIDDLE_MM, alias, audit, coarse_contacts, contact_class, formation_name,
                                 normal_code)
from hook_mode_graph import spectral_top
from hook_topology import threading_name
from pcca import dominant, pcca


def class_graph(g, cls_id, args):
    """Sliding edges within one class, and their weights."""
    keep = g["in_10.0"] & (g["stray"] <= args.slide_mm)
    pairs = g["pairs"][keep]
    pairs = pairs[cls_id[pairs[:, 0]] == cls_id[pairs[:, 1]]]
    w = np.ones(len(pairs))
    if args.wrench:
        X = mge.wrench_direction(g, "closest", args.lever)
        ang = np.arccos(np.clip(np.sum(X[pairs[:, 0]] * X[pairs[:, 1]], axis=1), -1, 1))
        w = np.exp(-ang ** 2 / (2 * np.median(ang) ** 2))
    return pairs, w


def spectral_split(W, args):
    """Labels of one connected piece: k-means on its normalized-Laplacian eigenvectors, k at the largest gap."""
    max_k = max(1, min(args.max_k, W.shape[0] // args.min_mode))
    if max_k == 1:
        return np.zeros(W.shape[0], int), 1
    vals, vecs = spectral_top(W, max_k, tol=1e-6)
    k = int(np.argmax(np.diff(vals)[:max_k]) + 1)
    if k == 1:
        return np.zeros(W.shape[0], int), k
    return mge.smooth(cluster(vecs, k, 0), W), k


def connect(labels, W):
    """Each mode keeps its largest piece connected by sliding edges (W holds only edges within a class); a
    detached fragment joins the mode it shares most sliding edges with, or is unassigned if it has none."""
    n = len(labels)
    for _ in range(5):
        Wc = W.tocoo()
        same = (labels[Wc.row] == labels[Wc.col]) & (labels[Wc.row] >= 0)
        _, comp = connected_components(sp.coo_matrix((np.ones(same.sum()), (Wc.row[same], Wc.col[same])),
                                                     shape=(n, n)), directed=False)
        ok = labels >= 0
        main = {}
        for c, m in zip(*np.unique(np.c_[comp[ok], labels[ok]], axis=0).T):
            main.setdefault(m, []).append(c)
        size = np.bincount(comp, minlength=comp.max() + 1)
        frags = [c for cs in main.values() if len(cs) > 1 for c in sorted(cs, key=lambda c: -size[c])[1:]]
        if not frags:
            return labels
        labels = labels.copy()
        for c in frags:
            ids = np.flatnonzero(comp == c)
            nb = W[ids].tocoo()
            other = labels[nb.col]
            other = other[(other >= 0) & (other != labels[ids[0]])]
            labels[ids] = np.bincount(other).argmax() if len(other) else -1
    return labels


def pcca_pick(X, args, rows=4000, seed=0, max_k=None, fallback=None):
    """PCCA+ memberships for k = 2 .. max_k (default --max-k) from the dominant eigenvectors X (first column
    constant): the largest k whose crispness reaches --crisp, else `fallback` (default: one set). The transform is
    optimised on a subset of rows (the extremes of each eigenvector and a random sample), which bounds the
    memberships; all rows then take it."""
    n = len(X)
    if n > rows:
        rng = np.random.default_rng(seed)
        ext = np.r_[X.argmax(0), X.argmin(0)]
        sub = np.unique(np.r_[ext, rng.choice(n, rows, replace=False)])
    else:
        sub = np.arange(n)
    best = (np.zeros(n, int), 1) if fallback is None else (fallback, int(fallback.max()) + 1)
    scan = []
    for k in range(2, min(max_k or args.max_k, X.shape[1] - 1, len(sub) - 1) + 1):
        _, crisp, A = pcca(None, k, X[sub])
        chi = np.clip(X[:, :k] @ A, 0, None)
        chi /= chi.sum(1, keepdims=True)
        scan.append(f"{k}:{crisp:.2f}")
        if crisp >= args.crisp:
            best = (chi.argmax(1), k)
    return best[0], best[1], " ".join(scan)


def pcca_split(W, args):
    X, _ = dominant(W, args.max_k + 1)
    lab, k, scan = pcca_pick(X, args)
    return mge.smooth(lab, W) if k > 1 else lab, k, scan


def dense_dominant(C, k):
    """dominant() for a small dense symmetric count matrix."""
    d = C.sum(1)
    vals, V = np.linalg.eigh(C / np.sqrt(np.outer(d, d)))
    order = np.argsort(-vals)[:k]
    X = V[:, order] / np.sqrt(d)[:, None] * np.sqrt(d.sum())
    X[:, 0] = 1.0
    return X, vals[order]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experiment_dir", type=Path)
    parser.add_argument("--method", choices=["spectral", "pcca", "lump"], required=True)
    parser.add_argument("--formations", default="formations_sided1_coarse_normal_min100",
                        help="formation-mode tag: the microstates of lump, and the names of every method's modes")
    parser.add_argument("--wrench", action="store_true", help="weight sliding edges by wrench-direction change")
    parser.add_argument("--lever", type=float, default=10.0, help="torque scale of the wrench direction [mm]")
    parser.add_argument("--slide-mm", type=float, default=0.25)
    parser.add_argument("--middle-mm", type=float, default=MIDDLE_MM,
                        help="contacts this close to the loop's mid-plane are on neither side but in the middle")
    parser.add_argument("--max-k", type=int, default=12, help="largest number of modes per piece")
    parser.add_argument("--crisp", type=float, default=0.8, help="crispness a PCCA+ split must reach")
    parser.add_argument("--min-mode", type=int, default=100, help="smaller pieces stay unassigned")
    parser.add_argument("--near-contact", type=float, default=1.0)
    args = parser.parse_args()

    exp = args.experiment_dir
    g = dict(np.load(exp / "contact_graph.npz"))
    z = np.load(exp / "strata" / "contacts.npz")
    n = len(g["poses"])
    cls = contact_class(z["witness"], g["thread"], args.middle_mm)
    cls_keys, cls_id = np.unique(cls, axis=0, return_inverse=True)
    cls_id = cls_id.ravel()
    pairs, w = class_graph(g, cls_id, args)
    W = mge.adjacency(n, pairs, w)
    form = np.load(exp / f"labels_contact_{args.formations}.npy")
    aliases = json.loads((exp / "strata" / "aliases.json").read_text())
    # each pose's own coarse formation (part ~ part, normal axis), to name the modes by
    c = z["contacts"][:, 0, :6].copy()
    c[:, 2] = np.where(c[:, 2] >= 0, 0, -1)
    c[:, 5] = np.where(c[:, 5] >= 0, 0, -1)
    pose_keys, pose_form = np.unique(np.c_[coarse_contacts(c, aliases), normal_code(g["poses"], z["witness"][:, 0]), cls],
                                     axis=0, return_inverse=True)
    pose_form = pose_form.ravel()
    faces = json.loads((exp / "strata" / "cad_faces.json").read_text())
    pose_names = [alias(formation_name(k, faces, True), aliases).split("; ")[0] for k in pose_keys]

    labels = np.full(n, -1)
    t0 = time.time()
    if args.method == "lump":
        M = form.max() + 1
        ok = form >= 0
        P = sp.csr_matrix((np.ones(ok.sum()), (np.flatnonzero(ok), form[ok])), shape=(n, M))
        C = (P.T @ W @ P).toarray()
        micro_cls = np.array([cls_id[form == m][0] for m in range(M)])
    for c, key in enumerate(cls_keys):
        tag = f"loop {HALF_NAMES[int(key[0])]}, {threading_name(int(key[1]))}"
        if args.method == "lump":
            ms = np.flatnonzero(micro_cls == c)
            if not len(ms):
                continue
            Cc = C[np.ix_(ms, ms)]
            ncomp, comp = connected_components(sp.csr_matrix(Cc - np.diag(np.diag(Cc))), directed=False)
            for q in range(ncomp):
                mq = ms[comp == q]
                if len(mq) == 1:
                    lab, k, scan = np.zeros(1, int), 1, "-"
                else:
                    # every grouping is tried; with none crisp, the formation modes stay apart (never all merged)
                    X, _ = dense_dominant(C[np.ix_(mq, mq)], len(mq))
                    lab, k, scan = pcca_pick(X, args, max_k=len(mq) - 1, fallback=np.arange(len(mq)))
                base = labels.max() + 1
                for m, l in zip(mq, lab):
                    labels[form == m] = base + l
                print(f"class {tag:22s} piece of {len(mq):3d} formation modes -> {k} modes   crispness {scan}",
                      flush=True)
            continue
        ids = np.flatnonzero(cls_id == c)
        Wc = W[ids][:, ids]
        ncomp, comp = connected_components(Wc, directed=False)
        for q in np.flatnonzero(np.bincount(comp) >= args.min_mode):
            sel = ids[comp == q]
            Wq = W[sel][:, sel]
            if args.method == "spectral":
                lab, k = spectral_split(Wq, args)
                scan = ""
            else:
                lab, k, scan = pcca_split(Wq, args)
            labels[sel] = labels.max() + 1 + lab
            print(f"class {tag:22s} piece of {len(sel):7,d} poses -> {k} modes {scan}", flush=True)
    labels = connect(labels, W)
    # modes too small after splitting stay unassigned too
    size = np.bincount(labels[labels >= 0])
    labels[(labels >= 0) & (size[np.maximum(labels, 0)] < args.min_mode)] = -1
    _, labels[labels >= 0] = np.unique(labels[labels >= 0], return_inverse=True)
    k = labels.max() + 1
    audit(labels, cls)

    ok = labels >= 0
    ref_thread, ref_full = mge.physical_labels(g["poses"][ok], g["feat"][ok], g["thread"][ok])
    side = cls[ok, 0] + 1
    Wall = mge.adjacency(n, g["pairs"][g["in_10.0"] & (g["stray"] <= args.slide_mm)], np.ones(
        int((g["in_10.0"] & (g["stray"] <= args.slide_mm)).sum())))
    sc = mge.score(labels[ok], ref_thread, ref_full, Wall[ok][:, ok], g["feat"][ok, 0] < args.near_contact)
    sided = ref_full * 3 + side
    sc["nmi_sided"] = float(mge.normalized_mutual_info_score(sided, labels[ok]))
    sc["purity_sided"] = float(sp.coo_matrix((np.ones(ok.sum()), (labels[ok], np.unique(sided, return_inverse=True)[1])))
                               .toarray().max(1).sum() / ok.sum())
    name = f"sided{args.middle_mm:g}_{args.method}{'_wrench' if args.wrench else ''}" + \
           (f"_{args.formations.split('_', 2)[2]}" if args.method == "lump" else "") + \
           (f"_crisp{args.crisp}" if args.method != "spectral" else "")
    np.save(exp / f"labels_contact_{name}.npy", labels)
    modes = []
    for m in range(k):
        sel = labels == m
        f, cnt = np.unique(pose_form[sel], return_counts=True)
        top = np.argsort(-cnt)[:4]
        key = cls[sel][0]
        main = ", ".join(f"{pose_names[f[t]].split(' ~ ')[0]} {pose_names[f[t]].split(', ')[-1]} "
                         f"{cnt[t] / sel.sum():.0%}" for t in top[:3] if cnt[t] / sel.sum() >= 0.05)
        modes.append({"mode": m, "samples": int(sel.sum()), "side": int(key[0]), "threading": int(key[1]),
                      "formations": [{"name": pose_names[f[t]], "share": round(float(cnt[t] / sel.sum()), 3)}
                                     for t in top],
                      "name": f"{main}; loop {HALF_NAMES[int(key[0])]}"
                              + ("" if key[1] == 0 else f", {threading_name(int(key[1]))}")})
    (exp / "strata" / f"{name}.json").write_text(json.dumps({"k": int(k), "modes": modes}, indent=2))
    path = exp / "experiments.json"
    old = json.loads(path.read_text()) if path.exists() else {"rows": [], "spectra": {}}
    variant = f"contact: {name.replace('_', ' ')}"
    old["rows"] = [r for r in old["rows"] if r["variant"] != variant] + [
        {"variant": variant, "nodes": n, "edges": int(len(pairs)), "k": int(k), "k_is_eigengap": True,
         "eigengap": float("nan"), "smoothed": True, **sc}]
    path.write_text(json.dumps(old, indent=2))
    print(f"{name}: {k} modes ({time.time() - t0:.0f} s); NMI full {sc['nmi_full']:.2f} sided {sc['nmi_sided']:.2f}, "
          f"purity full {sc['purity_full']:.2f} sided {sc['purity_sided']:.2f}, threading purity "
          f"{sc['purity_thread']:.3f}, disagreement {sc['disagreement']:.3f}, unassigned {np.mean(~ok):.2%}", flush=True)
    for md in sorted(modes, key=lambda m: -m["samples"]):
        print(f"   {md['mode']:3d} {md['samples']:7,d}  {md['name']}", flush=True)


if __name__ == "__main__":
    main()
