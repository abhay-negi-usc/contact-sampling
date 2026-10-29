#!/usr/bin/env python3
"""How the assembly groups of assembly_groups.py change as the collision tolerance loosens: a split that rests on
blends dipping only a little into the hole vanishes once that much penetration is allowed, while one that rests on
blends through the tube wall survives.

For a random subset of the run's paths (--paths), the share of collision-free blends of every pair is recomputed at
each tolerance of --tols (the run's own tolerance reuses its saved compatibility), and PCCA+ (pcca.py) is scanned
on each. Writes OUT_DIR/tol_sweep_<paths>.npz (sub: the paths' indices in groups.npz; S_<tol>: their compatibility
matrix), which plot_assembly_mode_graphs.py draws, and a summary (eigenvalues, crispness) beside it.

    conda run -n mujoco_env python scripts/assembly_tolerance_sweep.py --dir OUT_DIR
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

from assembly_groups import blends_free
from pcca import pcca_scan


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", type=Path, required=True, help="assembly_groups.py output directory")
    parser.add_argument("--paths", type=int, default=300, help="paths in the subset (pairs grow as its square)")
    parser.add_argument("--tols", type=float, nargs="+", default=[0.005, 0.05, 0.1, 0.3, 1.0],
                        help="penetration allowed along a blend [mm]")
    parser.add_argument("--max-k", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    g = np.load(args.dir / "groups.npz")
    run = json.loads((args.dir / "summary.json").read_text())["args"]
    alphas = np.arange(1, int(run["alphas"]) + 1) / (int(run["alphas"]) + 1)
    sub = np.sort(np.random.default_rng(args.seed).choice(len(g["paths"]), args.paths, replace=False))
    paths = g["paths"][sub]
    pi, pj = np.triu_indices(len(sub), 1)
    saved, summary = dict(sub=sub), {}
    for tol in args.tols:
        t0 = time.time()
        if np.isclose(tol, float(run["tol"])):
            S = np.eye(len(g["paths"]))
            S[g["pi"], g["pj"]] = S[g["pj"], g["pi"]] = g["compat"]
            S = S[np.ix_(sub, sub)]
        else:
            S = np.eye(len(sub))
            S[pi, pj] = S[pj, pi] = blends_free(paths, pi, pj, alphas, tol) / len(alphas)
        scan, vals = pcca_scan(csr_matrix(S), range(2, args.max_k + 1))
        k = int(np.argmax(vals[1:args.max_k] - vals[2:args.max_k + 1]) + 2)
        print(f"tol {tol} mm ({time.time() - t0:.0f} s): eigenvalues {np.round(vals[:args.max_k + 1], 3)}, largest gap "
              f"after k = {k}; crispness " + " ".join(f"k{j} {c:.2f}" for j, (_, c) in scan.items()), flush=True)
        saved[f"S_{tol}"] = S
        summary[tol] = dict(eigenvalues=vals[:args.max_k + 1].tolist(), k_gap=k,
                            crispness={j: c for j, (_, c) in scan.items()})
    out = args.dir / f"tol_sweep_{args.paths}"
    np.savez_compressed(out.with_suffix(".npz"), **saved)
    out.with_suffix(".json").write_text(json.dumps(summary, indent=1))
    print(out.with_suffix(".npz"))


if __name__ == "__main__":
    main()
