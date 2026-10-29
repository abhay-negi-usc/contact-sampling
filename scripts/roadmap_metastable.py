#!/usr/bin/env python3
"""Metastable regions of a peg diffusing through the free space and along the contacts: PCCA+ (pcca.py) on
the random walk over the collision-free roadmap of assembly_groups.py.

The walk moves along roadmap edges (straight collision-free moves between neighbouring poses) with weight
exp(-d^2 / s^2), s the median edge length. Regions it stays in for a long time are joined to the rest only
through narrow passages (the bore's 0.5 mm clearance, the way round the rim), so they are the stages an
assembly has to pass through; contact states in the same region need no narrow passage between them.
The analytical modes are used only to read the regions.

    conda run -n mujoco_env python scripts/roadmap_metastable.py --roadmap ROADMAP.npz --out-dir OUT
"""

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

import peg_hole_analytic as A
from codim_aware_scores import merge_bore, metrics, resolve
from pcca import pcca_scan
from stratum_spectral import short


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--roadmap", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-k", type=int, default=10)
    parser.add_argument("--k", type=int, help="regions (default: largest eigengap)")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    z = np.load(args.roadmap)
    nodes, n_contact, mode, names = z["nodes"], int(z["n_contact"]), z["contact_mode"], list(z["mode_names"])
    ei, ej, d = z["ei"], z["ej"], z["w"]
    w = np.exp(-d ** 2 / np.median(d) ** 2)
    n = len(nodes)
    W = sp.coo_matrix((np.r_[w, w], (np.r_[ei, ej], np.r_[ej, ei])), shape=(n, n)).tocsr()
    ncomp, comp = connected_components(W, directed=False)
    main_ = comp == np.argmax(np.bincount(comp))
    Wm = W[main_][:, main_]
    scan, vals = pcca_scan(Wm, range(2, args.max_k + 1))
    k_gap = int(np.argmax(vals[1:args.max_k] - vals[2:args.max_k + 1]) + 2)
    print(f"{main_.sum():,} of {n:,} nodes connected; eigenvalues " + " ".join(f"{v:.5f}" for v in vals[:args.max_k + 1])
          + f"; largest gap after k = {k_gap}")
    for k, (chi, crisp) in scan.items():
        print(f"   k={k:2d}  crispness {crisp:.3f}  confidently assigned (> 0.8): {np.mean(chi.max(1) > 0.8):.0%}")
    report = dict(eigenvalues=vals.tolist(), k_gap=k_gap, runs={})
    idx = np.flatnonzero(main_)
    on = main_[:n_contact]
    for k in sorted({args.k or k_gap, 3, 4, 5}):
        chi = scan[k][0]
        region = np.full(n, -1)
        region[idx] = chi.argmax(1)
        rc, mc = region[:n_contact][on], mode[on]
        cont = np.zeros((k, len(names)))
        np.add.at(cont, (rc, mc), 1)
        base, bnames = resolve(cont, names)
        bb, bbn = merge_bore(base, bnames)
        fam = np.array([A.FAMILIES.index(A.family_of_mode(m)) for m in names])
        cf = np.zeros((k, len(A.FAMILIES)))
        np.add.at(cf, (rc, fam[mc]), 1)
        print(f"== k = {k}: codim ARI {metrics(bb)['ari']:.3f} H {metrics(bb)['homogeneity']:.3f} C {metrics(bb)['completeness']:.3f}"
              f" | family ARI {metrics(cf)['ari']:.3f}")
        regions = []
        for g in range(k):
            sel_all = region == g
            share_c = (rc == g).mean()
            fs = cf[g] / max(cf[g].sum(), 1)
            ms = bb[g] / max(bb[g].sum(), 1)
            lab = lambda p: p if " " in p else short(p)
            info = dict(region=g, nodes=int(sel_all.sum()), contact_share=float(share_c),
                        free_share=float(np.mean(sel_all[n_contact:])),
                        rho=np.percentile(nodes[sel_all, 0], [5, 50, 95]).round(1).tolist(),
                        depth=np.percentile(nodes[sel_all, 3], [5, 50, 95]).round(1).tolist(),
                        families={A.FAMILIES[i]: round(float(fs[i]), 3) for i in np.argsort(-fs) if fs[i] >= 0.02},
                        modes={lab(bbn[i]): round(float(ms[i]), 3) for i in np.argsort(-ms) if ms[i] >= 0.03})
            regions.append(info)
            print(f"   region {g}: {sel_all.sum():,} nodes ({share_c:.0%} of contact poses), rho {info['rho']} depth {info['depth']}"
                  f"\n      families: " + ", ".join(f"{a} {b:.0%}" for a, b in info["families"].items())
                  + "\n      modes: " + ", ".join(f"{a} {b:.0%}" for a, b in info["modes"].items()))
        report["runs"][k] = dict(regions=regions, scores=dict(codim=metrics(bb), family=metrics(cf)))
        np.save(args.out_dir / f"regions_k{k}.npy", region)
    (args.out_dir / "summary.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
