#!/usr/bin/env python3
"""The stratification found from the CAD alone, as a lattice: each CAD contact mode (cad_contacts.py) is a
node on the level of its number of distinct contacts; a line joins a mode to each mode one contact
smaller that it contains (its stratum lies in that one's closure), drawn heavier the more trajectory
steps cross between them; colour is the one-contact mode the closure merge put it in.

    conda run -n mujoco_env python scripts/plot_cad_lattice.py --subset SUBSET.npz --cad CAD.npz --out OUT.png
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cad_contacts import analytic_name
from contact_graph_spectral import transition_edges
from contact_mode_graph import COLORS, EDGE, INK, MUTED, OTHER
from plot_cad_modes import category

SHORT = {"tip_edge": "E", "tip_face": "F", "tip_side": "S", "rim_in": "inner rim", "rim_out": "outer rim", "top": "top",
         "wall": "wall", "floor": "floor", "outer_wall": "outer wall", "step": "step", "step_edge": "step edge"}


def short(mode):
    return " + ".join(f"{SHORT.get(a, a)}·{SHORT.get(b, b)}" for a, b in (p.split("/") for p in sorted(set(mode.split(" + ")))))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--subset", type=Path, required=True)
    parser.add_argument("--cad", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--min-samples", type=int, default=300)
    args = parser.parse_args()
    sub, cad = np.load(args.subset), np.load(args.cad)
    order = np.argsort(sub["samples"])
    traj, step = sub["traj"][order], sub["step"][order]
    cmode, cnames = cad["mode"], [str(n) for n in cad["mode_names"]]
    counts = np.bincount(cmode, minlength=len(cnames))
    a, b = transition_edges(cmode, traj, step, {"traj": np.zeros(0, np.int64), "parent_traj": np.zeros(0, np.int64),
                                                 "parent_step": np.zeros(0, np.int64)})
    T = np.zeros((len(cnames), len(cnames)))
    np.add.at(T, (a, b), 1)
    T += T.T
    sets = [frozenset(m.split(" + ")) if m != "none" else frozenset() for m in cnames]
    nodes = [i for i in np.argsort(-counts) if counts[i] >= args.min_samples and len(sets[i]) >= 1]
    # merge modes with the same distinct contacts (a contact at two points is one contact)
    key = {}
    for i in nodes:
        key.setdefault(sets[i], []).append(i)
    groups = list(key.items())
    size = {s: counts[ids].sum() for s, ids in groups}
    singles = [s for s, _ in groups if len(s) == 1]
    parent = {}
    for s, ids in groups:
        if len(s) == 1:
            parent[s] = s
        else:
            cands = [t for t in singles if t <= s] or singles
            parent[s] = max(cands, key=lambda t: sum(T[i, j] for i in ids for j in key[t]))
    level = {s: len(s) for s, _ in groups}
    single_order = sorted(singles, key=lambda t: -size[t])
    pos = {}
    for lv in sorted(set(level.values())):
        row = [s for s, _ in groups if level[s] == lv]
        row.sort(key=lambda s: (single_order.index(parent[s]), -size[s]))
        xs = np.linspace(0, 1, len(row) + 2)[1:-1]
        for s, xv in zip(row, xs):
            pos[s] = (xv, lv)

    fig, ax = plt.subplots(figsize=(15, 7.5))
    fig.patch.set_facecolor("white")
    trans = []
    for s, ids in groups:
        for t, jds in groups:
            if len(t) == len(s) - 1 and t <= s:
                trans.append((s, t, sum(T[i, j] for i in ids for j in jds)))
    wmax = max(w for *_, w in trans) if trans else 1
    for s, t, w in trans:
        (x0, y0), (x1, y1) = pos[s], pos[t]
        ax.plot([x0, x1], [y0, y1], color=EDGE, lw=0.4 + 3.5 * np.sqrt(w / wmax), alpha=0.6, zorder=1)
    total = counts.sum()
    for s, ids in groups:
        name = analytic_name(cnames[ids[0]])
        c = category(analytic_name(cnames[key[parent[s]][0]]))
        x, y = pos[s]
        ax.scatter([x], [y], s=60 + 2500 * np.sqrt(size[s] / total), color=OTHER if c < 0 else COLORS[c],
                   edgecolors="white", linewidths=2, zorder=2)
        ax.text(x, y - 0.13 if y > 1 else y - 0.16, short(name), ha="center", va="top", fontsize=7 if y > 1 else 8,
                color=INK, rotation=0 if y == 1 else 30, rotation_mode="anchor")
    ax.set_yticks(sorted(set(level.values())))
    ax.set_yticklabels([f"{lv} contact{'s' if lv > 1 else ''}" for lv in sorted(set(level.values()))], color=MUTED)
    ax.set_xticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_ylim(max(level.values()) + 0.5, 0.4)
    ax.set_title("Stratification found from the CAD: contact modes by number of contacts, lines to the modes each one "
                 "contains (heavier: more trajectory crossings), colour: the one-contact mode it merges into",
                 color=INK, fontsize=11, wrap=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=110)
    print(f"{args.out}: {len(groups)} modes ({len(singles)} one-contact)")


if __name__ == "__main__":
    main()
