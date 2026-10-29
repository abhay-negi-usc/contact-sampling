#!/usr/bin/env python3
"""Stratum graph of CAD contact modes: the unsupervised counterpart of stratum_spectral.py.

Nodes: contact modes found from the CAD alone (cad_contacts.py: which indexed peg feature touches which
indexed hole feature) on whole sampled trajectories. Edges: the trajectory steps between modes.
Spectral clustering of the normalized Laplacian groups the modes into families (k from the largest
eigengap, and at fixed k). The analytical model enters only afterwards, to score: the CAD modes against
the analytical modes (at --truth-tol), and the families against the known families and regions, with the
codimension rule of codim_aware_scores.py (a mode may be grouped with any mode whose contacts it holds).
Closure merge (no spectral step): each multi-contact CAD mode joins whichever of its own one-contact
modes it trades most transitions with, leaving one cluster per one-contact mode.

    conda run -n mujoco_env python scripts/cad_stratum_graph.py RUN_DIR [RUN_DIR ...] --subset SUBSET.npz --cad CAD.npz --out-dir OUT
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

import peg_hole_analytic as A
from cad_contacts import analytic_name
from codim_aware_scores import merge_bore, metrics, resolve
from contact_graph_spectral import transition_edges
from knn_pose_spectral import REGIONS, contact_modes_at, load_samples


def families(W, k_max, k, seed):
    deg = W.sum(1)
    Dm = 1 / np.sqrt(np.maximum(deg, 1e-12))
    L = np.eye(len(W)) - Dm[:, None] * W * Dm[None, :]
    vals, vecs = np.linalg.eigh(L)
    k_gap = int(np.argmax(np.diff(vals)[1:k_max]) + 2)
    k = k or k_gap
    U = vecs[:, :k] / np.maximum(np.linalg.norm(vecs[:, :k], axis=1, keepdims=True), 1e-12)
    return KMeans(k, n_init=50, random_state=seed).fit_predict(U), vals, k_gap


def score(lab, mode, names):
    """Scores of sample labels lab against analytical modes (ids into names), as in strata_decomposition."""
    k = lab.max() + 1
    cont = np.zeros((k, len(names)))
    np.add.at(cont, (lab, mode), 1)
    base, bnames = resolve(cont, names)
    bb, bbn = merge_bore(base, bnames)
    fam = np.array([A.FAMILIES.index(A.family_of_mode(n)) for n in names])
    reg = np.array([REGIONS.index(A.REGION_OF_FAMILY[A.FAMILIES[f]]) for f in fam])
    cf, cr = np.zeros((k, len(A.FAMILIES))), np.zeros((k, 3))
    np.add.at(cf, (lab, fam[mode]), 1)
    np.add.at(cr, (lab, reg[mode]), 1)
    return {"exact mode": metrics(cont), "base, bore merged": metrics(bb), "family": metrics(cf), "region": metrics(cr)}, bb, bbn


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path, nargs="+")
    parser.add_argument("--subset", type=Path, required=True, help="samples of whole trajectories, with traj and step")
    parser.add_argument("--cad", type=Path, required=True, help="cad_contacts.py output for those samples")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--min-samples", type=int, default=300)
    parser.add_argument("--max-k", type=int, default=10)
    parser.add_argument("--fixed-k", type=int, nargs="*", default=[4, 6])
    parser.add_argument("--truth-tol", type=float, default=0.001,
                        help="contact tolerance of the analytical modes scored against [mm] (larger for poses "
                             "sampled on other geometry, e.g. MuJoCo's 64-gon bore)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    sub, cad = np.load(args.subset), np.load(args.cad)
    assert np.array_equal(np.sort(sub["samples"]), cad["samples"])
    order = np.argsort(sub["samples"])
    traj, step = sub["traj"][order], sub["step"][order]
    cmode, cnames = cad["mode"], [str(n) for n in cad["mode_names"]]
    q_all, _, _ = load_samples(args.run_dir)
    amode, anames = contact_modes_at(q_all[cad["samples"]], args.truth_tol)

    # the CAD modes against the analytical ones (names attached for the cross-check only)
    cad_as_analytic = np.array([analytic_name(cnames[m]) for m in cmode])
    an = np.array(anames)[amode]
    same = cad_as_analytic == an
    same_set = np.array([set(a.split(" + ")) == set(b.split(" + ")) for a, b in zip(cad_as_analytic, an)])
    print(f"{len(cmode):,} samples on {len(np.unique(traj)):,} trajectories; {len(cnames)} CAD modes, {len(anames)} analytical; "
          f"agreement exact {same.mean():.2%}, as contact sets {same_set.mean():.2%}")

    counts = np.bincount(cmode, minlength=len(cnames))
    nodes = np.flatnonzero(counts >= args.min_samples)
    node_of = np.full(len(cnames), -1)
    node_of[nodes] = np.arange(len(nodes))
    a, b = transition_edges(cmode, traj, step, {"traj": np.zeros(0, np.int64), "parent_traj": np.zeros(0, np.int64),
                                                 "parent_step": np.zeros(0, np.int64)})
    keep = (node_of[a] >= 0) & (node_of[b] >= 0)
    W = np.zeros((len(nodes), len(nodes)))
    np.add.at(W, (node_of[a[keep]], node_of[b[keep]]), 1)
    W = W + W.T
    print(f"{len(nodes)} CAD modes with >= {args.min_samples} samples hold {counts[nodes].sum() / len(cmode):.1%}; "
          f"{int(keep.sum()):,} transitions between them")
    report = dict(agreement=dict(exact=float(same.mean()), sets=float(same_set.mean())), runs=[])
    on_node = node_of[cmode] >= 0
    for k in [None] + args.fixed_k:
        fam, vals, k_gap = families(W, args.max_k, k, args.seed)
        lab = fam[node_of[cmode[on_node]]]
        sc, bb, bbn = score(lab, amode[on_node], anames)
        kk = int(fam.max() + 1)
        print(f"== k = {kk} ({'eigengap' if k is None else 'fixed'}); eigenvalues " + " ".join(f"{v:.3f}" for v in vals[:args.max_k + 1]))
        for lvl, v in sc.items():
            print(f"    {lvl:18s} ARI {v['ari']:.3f}  homogeneity {v['homogeneity']:.3f}  completeness {v['completeness']:.3f}  purity {v['purity']:.3f}")
        fams = []
        for f in range(kk):
            members = nodes[fam == f]
            share = counts[members].sum() / counts[nodes].sum()
            top = sorted(members, key=lambda m: -counts[m])[:6]
            shares = bb[f] / max(bb[f].sum(), 1)
            print(f"    family {f} {share:6.1%}: " + "; ".join(f"{analytic_name(cnames[m])} ({counts[m] / len(cmode):.1%})" for m in top))
            print(f"         base modes: " + ", ".join(f"{bbn[i]} {shares[i]:.0%}" for i in np.argsort(-shares) if shares[i] >= 0.05))
            fams.append(dict(family=f, share=float(share), modes={cnames[m]: int(counts[m]) for m in members}))
        report["runs"].append(dict(k=kk, chosen="eigengap" if k is None else "fixed", eigenvalues=vals[:args.max_k + 1].tolist(),
                                   scores=sc, families=fams))
    # closure merge: each multi-contact mode joins the one-contact mode (one of its own contacts) it trades
    # most transitions with; the result is one cluster per one-contact mode, no spectral step
    n = len(cnames)
    T = np.zeros((n, n))
    np.add.at(T, (a, b), 1)
    T = T + T.T
    contact_sets = [frozenset(m.split(" + ")) if m != "none" else frozenset() for m in cnames]
    # one-contact modes: one distinct contact (a contact made at two points is still that contact); the
    # largest mode with a given contact stands for it
    single = {}
    for i in np.argsort(-counts):
        if len(contact_sets[i]) == 1:
            single.setdefault(next(iter(contact_sets[i])), i)
    parent = np.arange(n)
    for i, cs in enumerate(contact_sets):
        if len(cs) == 1:
            parent[i] = single[next(iter(cs))]
            continue
        cands = [single[c] for c in cs if c in single] or [j for j in single.values()]
        parent[i] = max(cands, key=lambda j: (T[i, j], counts[j]))
    _, lab = np.unique(parent[cmode], return_inverse=True)
    sc, bb, bbn = score(lab, amode, anames)
    print(f"== closure merge: {len(cnames)} CAD modes -> {lab.max() + 1} one-contact modes")
    for lvl, v in sc.items():
        print(f"    {lvl:18s} ARI {v['ari']:.3f}  homogeneity {v['homogeneity']:.3f}  completeness {v['completeness']:.3f}  purity {v['purity']:.3f}")
    sizes = np.bincount(lab) / len(lab)
    clusters = []
    for c in np.argsort(-sizes):
        if sizes[c] < 0.003:
            continue
        shares = bb[c] / max(bb[c].sum(), 1)
        rep_mode = cnames[np.unique(parent[cmode])[c]]
        print(f"    {sizes[c]:6.1%}  {analytic_name(rep_mode):28s} <- base modes: " + ", ".join(f"{bbn[i]} {shares[i]:.0%}" for i in np.argsort(-shares) if shares[i] >= 0.03))
        clusters.append(dict(mode=rep_mode, share=float(sizes[c]), base_modes={bbn[i]: float(shares[i]) for i in np.argsort(-shares) if shares[i] >= 0.03}))
    report["closure_merge"] = dict(scores=sc, clusters=clusters)
    reps = np.unique(parent[cmode])
    np.savez_compressed(args.out_dir / "closure_labels.npz", samples=cad["samples"], cluster=lab,
                        cluster_mode=np.array([cnames[r] for r in reps]), analytic_mode=amode, analytic_names=np.array(anames))
    (args.out_dir / "summary.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
