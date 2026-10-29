#!/usr/bin/env python3
"""Turn one clustering from mode_graph_experiments.py into a mode directory that plot_hook_modes.py and
render_mode_examples.py read: samples.npz (poses, seps, labels) and modes.json (k, goal_mode, mode_edges).

The goal mode is the one holding the node nearest the pre-assembly pose. Two modes neighbour each other
when at least --min-edges of the node set's continuity edges (line-of-sight nearest-neighbour edges for
samples, lattice edges for the grid, edges sliding within 0.5 mm of contact for contact poses) join them.
Each mode's summary lists its most common physical labels (hook_topology.py): threading, closest hook
part, closest eyelet part, and the side of the loop plane.

    python scripts/export_experiment_modes.py EXPERIMENT_DIR LABELS_NAME --out MODE_DIR
"""

import argparse
import json
from pathlib import Path

import numpy as np

from goal_field import PRE_ASSEMBLY
from hook_topology import EYELET_PARTS, HOOK_PARTS
from mode_graph_experiments import physical_labels

SCALE = np.array([1, 1, 1] + [30.0 * np.pi / 180] * 3)


def describe(full):
    thread, hp, ep, side = full // 1000, full // 100 % 10, full // 10 % 10, full % 10
    t = "+".join(HOOK_PARTS[b] for b in HOOK_PARTS if thread & b) or "none"
    return (f"threads {t}; closest {HOOK_PARTS.get(hp, '-')} to {EYELET_PARTS[ep] if ep < len(EYELET_PARTS) else '-'}; "
            f"x {'>' if side else '<='} 0")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experiment_dir", type=Path)
    parser.add_argument("labels", help="labels file name, e.g. labels_samples_knn10_k10_smoothed.npy")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--min-edges", type=int, default=20)
    args = parser.parse_args()
    node_set = next((n for n in ("grid", "contact") if args.labels.startswith(f"labels_{n}")), "samples")
    graph = next(args.experiment_dir.glob("grid*_graph.npz")) if node_set == "grid" else \
        args.experiment_dir / f"{node_set}_graph.npz"
    g = np.load(graph)
    poses, feat, thread = g["poses"], g["feat"], g["thread"]
    labels = np.load(args.experiment_dir / args.labels)
    keep = labels >= 0
    k = int(labels.max() + 1)
    if node_set == "grid":
        pairs = g["pairs"]
    elif node_set == "contact":
        pairs = g["pairs"][g["in_10.0"] & (g["stray"] <= 0.5)]
    else:
        pairs = g["pairs"][(g["clear"] >= 0) & (g["in_10.0"] | g["in_30.0"] | g["in_60.0"])]
    a, b = labels[pairs[:, 0]], labels[pairs[:, 1]]
    cross = (a >= 0) & (b >= 0) & (a != b)
    keys, counts = np.unique(np.sort(np.column_stack([a[cross], b[cross]]), 1), axis=0, return_counts=True)
    mode_edges = [{"modes": [int(p), int(q)], "lines_of_sight": int(c)}
                  for (p, q), c in zip(keys, counts) if c >= args.min_edges]

    idx = np.flatnonzero(keep)
    goal_mode = int(labels[idx[np.argmin(np.linalg.norm((poses[idx] - PRE_ASSEMBLY) * SCALE, axis=1))]])
    _, full = physical_labels(poses, feat, thread)
    modes = []
    for m in range(k):
        mine = np.flatnonzero(labels == m)
        lab, cnt = np.unique(full[mine], return_counts=True)
        top = np.argsort(-cnt)[:4]
        modes.append({"mode": m, "samples": int(len(mine)),
                      "centre": np.round(np.median(poses[mine], 0), 1).tolist(),
                      "median_sep_mm": float(np.nanmedian(feat[mine, 0])),
                      "threaded_share": float(np.mean(thread[mine] > 0)),
                      "top_labels": [{"label": describe(int(lab[t])), "share": round(float(cnt[t] / len(mine)), 3)}
                                     for t in top],
                      "neighbours": sorted({q for e in mode_edges for q in e["modes"] if m in e["modes"]} - {m})})
    args.out.mkdir(parents=True, exist_ok=True)
    report = {"source": str(args.experiment_dir / args.labels), "node_set": node_set, "k": k,
              "goal_mode": goal_mode, "modes": modes, "mode_edges": mode_edges}
    (args.out / "modes.json").write_text(json.dumps(report, indent=2))
    np.savez(args.out / "samples.npz", poses=poses, seps=feat[:, 0], labels=labels)
    print(f"{args.labels}: {k} modes, goal mode {goal_mode}, {len(mode_edges)} mode edges -> {args.out}")
    for md in modes:
        print(f"  mode {md['mode']:2d}{'*' if md['mode'] == goal_mode else ' '} {md['samples']:7,}  "
              f"{md['top_labels'][0]['share']:.0%} {md['top_labels'][0]['label']}")


if __name__ == "__main__":
    main()
