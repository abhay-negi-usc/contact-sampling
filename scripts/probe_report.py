#!/usr/bin/env python3
"""Report of probe_design.py records: accuracy, truth mode, P(correct) and the weak Fisher eigenvalues against the
number of probes per design rule, the regions the design picks, and a figure.

    python scripts/probe_report.py --records P.npz --out P.md --fig P.png
"""

import argparse
from collections import Counter

import numpy as np

from uq_analysis import auc, ece

RULES = (("dopt", "D-optimal, ring or stem"), ("dopt-ring", "D-optimal, ring only"), ("random", "random feasible"))


def figure(recs, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pose_filter_plots as P
    P.style()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.7), gridspec_kw={"width_ratios": [1, 1, 1]})
    colors = {"dopt": P.S1, "dopt-ring": P.S2, "random": P.S3}
    for ax, noise in zip(axes[:2], (0.0, 0.3)):
        for rule, label in RULES:
            pts = [(0, np.mean([r["correct"] for r in recs if r["noise"] == noise and r["rule"] == "none"]))]
            for n in sorted({r["n"] for r in recs if r["rule"] == rule}):
                pts.append((n, np.mean([r["correct"] for r in recs if r["noise"] == noise and r["rule"] == rule
                                        and r["n"] == n])))
            x, y = zip(*pts)
            ax.plot(x, y, "-o", color=colors[rule], label=label, zorder=3)
            ax.text(x[-1] + 0.2, y[-1], f"{y[-1]:.2f}", va="center", fontsize=8.5, color=P.INK2)
        tm = np.mean([r["tm_correct"] for r in recs if r["noise"] == noise and r["rule"] == "none"])
        ax.axhline(tm, color=P.AXIS, lw=1)
        ax.text(8.8, tm - 0.07, f"no probes: best fit at truth {tm:.2f}", ha="right", va="top", fontsize=8, color=P.INK2)
        ax.set_xlim(-0.3, 9.3)
        ax.set_ylim(0, 1.02)
        ax.set_xticks([0, 2, 4, 8])
        ax.set_xlabel("Probe contacts added")
        ax.set_title(f"SMC estimate correct, {P.NOISE_LABEL[noise]}", fontsize=10.5)
    axes[0].set_ylabel("Share of cases correct (2 mm and 3°)")
    axes[0].legend(loc="lower left", bbox_to_anchor=(0, 1.08), ncol=3, handlelength=1.6, columnspacing=1.4)
    ax = axes[2]
    for rule, label in RULES:
        pts = [(0, np.median([r["lam"][0] for r in recs if r["noise"] == 0.0 and r["rule"] == "none"]))]
        for n in sorted({r["n"] for r in recs if r["rule"] == rule}):
            pts.append((n, np.median([r["lam"][0] for r in recs if r["noise"] == 0.0 and r["rule"] == rule
                                      and r["n"] == n])))
        x, y = zip(*pts)
        ax.plot(x, y, "-o", color=colors[rule], zorder=3)
    ax.axhline(3.84, color=P.AXIS, lw=1)
    ax.text(8.8, 3.84 * 1.2, "not determined to tolerance below", ha="right", fontsize=8, color=P.INK2)
    ax.set_yscale("log")
    ax.set_xlim(-0.3, 9.3)
    ax.set_xticks([0, 2, 4, 8])
    ax.set_xlabel("Probe contacts added")
    ax.set_ylabel("Weakest Fisher eigenvalue at the truth (median)")
    ax.set_title("Weakest direction's information, noise 0", fontsize=10.5)
    P.headline(fig, "Two probe contacts placed by the D-optimal rule make the weak rotations observable",
               "Policy streams 0–31 at steps 160 and 300 (64 cases per noise level): the policy's 150 samples plus "
               "probes planned from the SMC estimate and executed as guarded moves on the true eyelet.")
    fig.subplots_adjust(left=0.055, right=0.985, top=0.74, bottom=0.13, wspace=0.28)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--records", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--fig")
    args = p.parse_args()
    recs = list(np.load(args.records, allow_pickle=True)["records"])
    L = ["# Active sensing: probe contacts for the weak rotations\n",
         "Per case: the policy's samples (150, thinned) plus n probes, batch SMC. Correct: 2 mm and 3 deg. lam: Fisher "
         "eigenvalues at the truth (tolerance units; < 3.84 = not determined to tolerance).\n"]
    L.append("| noise | step | rule | probes | cases | correct | truth mode correct | P(correct) mean | AUC | ECE | "
             "lam1 median | lam2 median | lam1 < 3.84 | probes hit |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    combos = sorted({(r["noise"], r["rule"], r["n"]) for r in recs},
                    key=lambda k: (k[0], ["none", "dopt", "dopt-ring", "random"].index(k[1]), k[2]))
    for step in sorted({r["step"] for r in recs}) + ["all"]:
        for noise, rule, n in combos:
            R = [r for r in recs if r["noise"] == noise and r["rule"] == rule and r["n"] == n
                 and (step == "all" or r["step"] == step)]
            y = np.array([r["correct"] for r in R])
            pc = np.array([r["p_correct"] for r in R])
            lam = np.array([r["lam"] for r in R])
            hit = np.mean(np.concatenate([r["hit"] for r in R])) if n else np.nan
            L.append(f"| {noise} | {step} | {rule} | {n} | {len(R)} | {y.mean():.2f} | "
                     f"{np.mean([r['tm_correct'] for r in R]):.2f} | {pc.mean():.2f} | "
                     f"{auc(pc, y) if 0 < y.mean() < 1 else float('nan'):.2f} | {ece(pc, y):.3f} | "
                     f"{np.median(lam[:, 0]):.3g} | {np.median(lam[:, 1]):.3g} | {np.mean(lam[:, 0] < 3.84):.2f} | "
                     f"{hit:.2f} |")
    L.append("\n## Regions the D-optimal rule picks (share of probes)\n")
    for rule in ("dopt", "dopt-ring"):
        for n in sorted({r["n"] for r in recs if r["rule"] == rule}):
            c = Counter(np.concatenate([r["region"] for r in recs if r["rule"] == rule and r["n"] == n]))
            tot = sum(c.values())
            L.append(f"- {rule}, {n} probes: " + ", ".join(f"{k} {v / tot:.2f}" for k, v in c.most_common()))
    rot = {}
    for rule in ("none", "dopt"):
        R = [r for r in recs if r["rule"] == rule and r["n"] in (0, 2)]
        rot[rule] = np.median(np.abs([r["err"][3:] for r in R]), 0)
    L.append("\nMedian |rotation error| (rx ry rz deg), all cases: no probes " + " ".join(f"{v:.2f}" for v in rot["none"])
             + "; 2 D-optimal probes " + " ".join(f"{v:.2f}" for v in rot["dopt"]) + "\n")
    open(args.out, "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    if args.fig:
        figure(recs, args.fig)


if __name__ == "__main__":
    main()
