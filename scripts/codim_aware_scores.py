#!/usr/bin/env python3
"""Agreement of contact-mode clusters with the known modes of the peg and hole when a mode of higher
codimension may be grouped with any mode of lower codimension it contains.

A mode is a set of contacts (peg feature / hole feature pairs, peg_hole_analytic.py); its stratum lies
in the closure of the stratum of every subset of those contacts, so a cluster holding a two-contact mode
together with one of its one-contact modes has not mixed anything. Each mode with several contacts is
therefore resolved, within its cluster, to whichever of its single contacts is most common there (counted
over every mode of the cluster that has it), and the clusters are scored against these single-contact
"base" modes: ARI, homogeneity, completeness, purity, and each cluster's base modes. Two base modes that
each hold at least --mixed of one cluster are a real mix of distinct modes. Level "base, bore merged"
also groups the one-contact modes inside the bore (edge on wall, side on inner rim, edge on floor).

    conda run -n mujoco_env python scripts/codim_aware_scores.py knn LABELS.npz RUN_DIR [RUN_DIR ...]
    conda run -n mujoco_env python scripts/codim_aware_scores.py stratum SUMMARY.json RUN_DIR [RUN_DIR ...]
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.special import comb

from knn_pose_spectral import load_samples
from stratum_spectral import short

BORE_ONE_POINT = {"tip_edge/wall", "tip_side/rim_in", "tip_edge/floor"}


def contacts(mode):
    """The distinct contacts of a mode (a contact made at two points counts once)."""
    return sorted(set(mode.split(" + ")))


def resolve(cont, names):
    """clusters x modes counts -> clusters x base modes counts, and the base mode names."""
    sets = [contacts(n) for n in names]
    base = sorted({p for s in sets for p in s})
    col = {p: i for i, p in enumerate(base)}
    out = np.zeros((len(cont), len(base)))
    for c, row in enumerate(cont):
        score = np.zeros(len(base))
        for m in np.flatnonzero(row):
            score[[col[p] for p in sets[m]]] += row[m]
        for m in np.flatnonzero(row):
            out[c, col[max(sets[m], key=lambda p: score[col[p]])]] += row[m]
    return out, base


def merge_bore(cont, base):
    keep = [i for i, p in enumerate(base) if p not in BORE_ONE_POINT]
    bore = [i for i, p in enumerate(base) if p in BORE_ONE_POINT]
    return np.c_[cont[:, keep], cont[:, bore].sum(1)], [base[i] for i in keep] + ["one point in the bore"]


def metrics(cont):
    """ARI, homogeneity, completeness and purity of a clusters x classes table of counts."""
    cont = cont[cont.sum(1) > 0][:, cont.sum(0) > 0]
    a, b, t = comb(cont.sum(1), 2).sum(), comb(cont.sum(0), 2).sum(), comb(cont.sum(), 2)
    e = a * b / t
    ari = (comb(cont, 2).sum() - e) / ((a + b) / 2 - e)
    p = cont / cont.sum()
    pk, pc = p.sum(1), p.sum(0)
    nz = p > 0
    h_c, h_k = -(pc[pc > 0] * np.log(pc[pc > 0])).sum(), -(pk[pk > 0] * np.log(pk[pk > 0])).sum()
    h_c_k = -(p[nz] * np.log((p / pk[:, None])[nz])).sum()
    h_k_c = -(p[nz] * np.log((p / pc[None, :])[nz])).sum()
    return dict(ari=float(ari), homogeneity=float(1 - h_c_k / h_c), completeness=float(1 - h_k_c / h_k),
                purity=float(cont.max(1).sum() / cont.sum()))


def label(p):
    return p if " " in p else short(p)


def report(cont, names, mixed):
    """Print and return the scores at each level and the mixes of distinct base modes per cluster."""
    base_cont, base = resolve(cont, names)
    levels = {"exact mode": (cont, names), "base": (base_cont, base),
              "base, bore merged": merge_bore(base_cont, base)}
    out = dict(scores={}, clusters=[])
    for lvl, (c, _) in levels.items():
        s = metrics(c)
        out["scores"][lvl] = s
        print(f"{lvl:18s} ARI {s['ari']:.3f}  homogeneity {s['homogeneity']:.3f}  completeness {s['completeness']:.3f}"
              f"  purity {s['purity']:.3f}")
    c, cls = levels["base, bore merged"]
    for k in range(len(c)):
        share = c[k] / c[k].sum()
        big = [(label(cls[i]), round(float(share[i]), 3)) for i in np.argsort(-share) if share[i] >= mixed]
        out["clusters"].append(dict(cluster=k, share=float(c[k].sum() / c.sum()), base_modes=big,
                                    mixed=len(big) > 1))
        print(f"  cluster {k}  {c[k].sum() / c.sum():.3f}  {'MIXED' if len(big) > 1 else 'clean'}  "
              + ", ".join(f"{n} {s:.0%}" for n, s in big))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("kind", choices=["knn", "stratum"])
    parser.add_argument("result", type=Path, help="labels_*.npz of knn_pose_spectral.py or summary.json of "
                                                  "stratum_spectral.py")
    parser.add_argument("run_dir", type=Path, nargs="+", help="the runs the result was made from, in order")
    parser.add_argument("--mixed", type=float, default=0.1, help="share of a cluster that makes a base mode count")
    args = parser.parse_args()
    _, mode, names = load_samples(args.run_dir)
    out = {}
    if args.kind == "knn":
        z = np.load(args.result)
        m = mode[z["samples"]]
        if "modes" in z.files:  # scored against modes at its own contact tolerance
            m, names = z["modes"], list(z["mode_names"])
        for key in z.files:
            if key in ("samples", "modes", "mode_names"):
                continue
            lab = z[key].astype(int)
            cont = np.zeros((lab.max() + 1, len(names)))
            np.add.at(cont, (lab, m), 1)
            print(f"== {key}")
            out[key] = report(cont, names, args.mixed)
    else:
        raw_of = {short(n): n for n in names}
        s = json.loads(args.result.read_text())
        cont = np.zeros((len(s["families"]), len(names)))
        for f in s["families"]:
            for name, count in f["modes"].items():
                cont[f["family"], names.index(raw_of[name])] = count
        print(f"== stratum families, k = {len(s['families'])}")
        out["stratum"] = report(cont, names, args.mixed)
    args.result.with_name(args.result.stem + "_codim.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
