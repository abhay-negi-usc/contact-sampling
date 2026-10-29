#!/usr/bin/env python3
"""Report of pose_identifiability.py records: the local weak directions (as screws, against the loop's axis and the
samples' contact centroid), the profile confidence sets, whether the truth mode's error on the real data lies in
the weak directions, and the rival modes split into the weak family and discrete ambiguities.

    python scripts/pose_identifiability_report.py --records I.npz --bench policy_smc.npz --streams S.npz --out I.md
"""

import argparse

import numpy as np
import torch
from scipy.spatial.transform import Rotation

import contact_gpu as G
import contact_information as CI
from info_metric_bench import sample_poses
from pose_identifiability import CHI2_1, LOOP_AXIS, LOOP_CENTRE, SCALE, screw


def contact_points(model, poses, truth):
    """Eyelet-frame contact point of each sample: the hook point of least separation, moved onto the eyelet's
    surface along the field's gradient (k, 3)."""
    pts = model.hook_points(poses)
    R = Rotation.from_rotvec(np.deg2rad(truth[3:])).as_matrix()
    Rt = torch.as_tensor(R, dtype=torch.float32, device=model.device)
    local = (pts - torch.as_tensor(truth[:3], dtype=torch.float32, device=model.device)) @ Rt
    sep = model.sdf(local.reshape(-1, 3)).reshape(local.shape[:2]) - model.radius
    p = local[torch.arange(len(local)), sep.argmin(1)]
    h = 0.05
    grad = torch.stack([(model.sdf(p + h * e) - model.sdf(p - h * e)) / (2 * h)
                        for e in torch.eye(3, device=model.device)], -1)
    n = grad / grad.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    return (p - model.sdf(p)[:, None] * n).cpu().numpy().astype(np.float64)


def line_dist(point, axis, q):
    return float(np.linalg.norm(np.cross(q - point, axis)))


def prominent_dip(cost, s, band=(14, 22), below=6.0, barrier=2.0):
    """Whether the profile has a local minimum within band (deg, either side) that fits within `below` of the
    truth's cost and is separated from the truth by a rise of at least `barrier`: a discrete near-solution, not a
    ripple on a flat valley."""
    i0 = int(np.argmin(np.abs(s)))
    d = cost - cost[i0]
    for i in range(1, len(s) - 1):
        if band[0] <= abs(s[i]) <= band[1] and d[i] <= min(d[i - 1], d[i + 1]) and d[i] < below:
            if d[min(i, i0): max(i, i0) + 1].max() - d[i] >= barrier:
                return True
    return False


def twist_of(rel):
    """Body twist in tolerance units of a pose given in the reference's frame (first order in translation)."""
    return np.concatenate([rel[:3], np.deg2rad(rel[3:])]) / SCALE


def figure(rows, recs, s_loop, path):
    """Fisher spectrum at the truth, profile along the loop rotation, where the truth mode's error lies, rivals."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pose_filter_plots as P
    P.style()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.9), gridspec_kw={"width_ratios": [1.1, 1.2, 0.9]})
    steps, colors = (20, 80, 300), (P.S1, P.S2, P.S3)
    ax = axes[0]
    for st, col, dx in zip(steps, colors, (-0.18, 0, 0.18)):
        lam = np.array([r["lam"] for r in rows if r["step"] == st])
        x = np.arange(1, 7) + dx
        q = np.percentile(lam, [25, 50, 75], axis=0)
        ax.vlines(x, q[0], q[2], color=col, lw=1.5, alpha=0.5, zorder=2)
        ax.plot(x, q[1], "o", color=col, label=f"step {st}", zorder=3)
    ax.axhline(CHI2_1, color=P.AXIS, lw=1)
    ax.text(6.45, CHI2_1 * 1.25, "not determined to tolerance below", ha="right", fontsize=8, color=P.INK2)
    ax.set_yscale("log")
    ax.set_xticks(range(1, 7))
    ax.set_xticklabels(["1", "2\nrotation", "3", "4", "5\ntranslation", "6"], fontsize=8.5)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("Cost increase for a one-tolerance move")
    ax.set_title("Fisher eigenvalues at the truth (median, IQR)", fontsize=10.5)
    ax.legend(loc="upper left", handletextpad=0.2)
    ax = axes[1]
    ax.axvspan(-3, 3, color=P.GRID, alpha=0.6, lw=0, zorder=0)
    ax.text(0, 47, "tolerance", ha="center", fontsize=8, color=P.INK2)
    for st, col in zip(steps, colors):
        D = np.array([r["exact"]["loop_cost"] - r["exact"]["loop_cost"].min() for r in recs if r["step"] == st])
        ax.plot(s_loop, np.median(D, 0), color=col, label=f"step {st}", zorder=3)
    ax.axhline(CHI2_1, color=P.AXIS, lw=1)
    ax.set_ylim(0, 50)
    ax.set_xlim(s_loop[0], s_loop[-1])
    ax.set_xlabel("Rotation about the loop's axis from the truth, deg")
    ax.set_ylabel("Profile cost above the minimum")
    ax.set_title("Noise-free data: profile along the loop rotation (median)", fontsize=10.5)
    ax.legend(loc="upper right", handletextpad=0.4)
    ax = axes[2]
    wrong = [r for r in rows if "tm_ok" in r and not r["tm_ok"]]
    sh = np.median([r["tm_share"] for r in wrong], 0)
    y = np.arange(3)[::-1]
    ax.barh(y + 0.17, sh, height=0.3, color=P.S1, label=f"truth mode's error ({len(wrong)} wrong cases)", zorder=3)
    ax.barh(y - 0.17, [1 / 6, 2 / 6, 3 / 6], height=0.3, color=P.AXIS, label="if it were isotropic", zorder=3)
    for yi, v in zip(y, sh):
        ax.text(v + 0.02, yi + 0.17, f"{v:.2f}", va="center", fontsize=8.5, color=P.INK)
    ax.set_yticks(y)
    ax.set_yticklabels(["weakest", "two weakest", "three weakest"])
    ax.set_xlim(0, 1.12)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Share of the error's displacement")
    ax.set_title("Noise 0: where the misses lie", fontsize=10.5)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.06), fontsize=8.5)
    P.headline(fig, "The contacts leave three rotations weakly constrained; rotation about the loop's axis is a "
                    "near-symmetry",
               "Policy streams 0–31, noise-free model-consistent data (left, middle) and the simulator's noise-0 samples "
               "(right). Translation is determined 100–10,000× more tightly than rotation.")
    fig.subplots_adjust(left=0.06, right=0.985, top=0.76, bottom=0.17, wspace=0.42)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def figure_fields(recs, other, s_loop, path, labels=("collision hulls", "smooth mesh")):
    """Median noise-free profile along the loop rotation, two eyelet fields side by side, at two steps."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pose_filter_plots as P
    P.style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, st in zip(axes, (80, 300)):
        ax.axvspan(-3, 3, color=P.GRID, alpha=0.6, lw=0, zorder=0)
        for R, col, label in ((recs, P.S1, labels[0]), (other, P.S2, labels[1])):
            D = np.array([r["exact"]["loop_cost"] - r["exact"]["loop_cost"].min() for r in R if r["step"] == st])
            ax.plot(s_loop, np.median(D, 0), color=col, label=label, zorder=3)
        ax.axhline(CHI2_1, color=P.AXIS, lw=1)
        ax.set_ylim(0, 40)
        ax.set_xlim(s_loop[0], s_loop[-1])
        ax.set_xlabel("Rotation about the loop's axis from the truth, deg")
        ax.set_title(f"Step {st}", fontsize=10.5)
    axes[0].set_ylabel("Profile cost above the minimum (median)")
    axes[0].legend(loc="lower left", bbox_to_anchor=(0, 1.1), ncol=2, handlelength=1.6)
    P.headline(fig, "The smooth eyelet is more symmetric than its collision hulls, not less",
               "Noise-free, model-consistent data on the same samples (policy streams 0–31). Gray band: the 3° "
               "tolerance; gray line: the 95% threshold (3.84).")
    fig.subplots_adjust(left=0.07, right=0.98, top=0.7, bottom=0.14, wspace=0.06)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--records", required=True)
    p.add_argument("--bench", required=True, help="pose_filter_bench records of the same cases (truth modes)")
    p.add_argument("--streams", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--fig", help="figure path (png)")
    p.add_argument("--compare", nargs=2, metavar=("RECORDS", "FIG"),
                   help="records of the same cases on another field, and the path of the profile comparison")
    p.add_argument("--field", default=str(G.FIELD), help="eyelet field (contact_gpu.FIELD or FIELD_SMOOTH)")
    args = p.parse_args()
    z = np.load(args.records, allow_pickle=True)
    recs, s_loop = list(z["records"]), z["s_loop"]
    targs = eval(str(z["args"]))
    bench = {(r["stream"], r["step"]): r for r in np.load(args.bench, allow_pickle=True)["records"]
             if r["noise"] == targs["noise"]}
    streams = np.load(args.streams, allow_pickle=True)["streams"]
    model = G.GpuContactModel(field=args.field)
    rows = []
    for r in recs:
        s = streams[r["stream"]]
        poses = sample_poses(s, r["step"], targs["noise"], 0.0, targs["max_samples"], 1000 * r["stream"] + r["step"])[0]
        cp = contact_points(model, poses, r["truth"])
        cen = cp.mean(0)
        ex, re = r["exact"], r["real"]
        row = {"step": r["step"], "used": r["used"], "lam": ex["lam"], "spread": float(np.sqrt(((cp - cen) ** 2).sum(1).mean())),
               "centroid": cen}
        for i in range(3):
            xi = ex["U"][:, i] * SCALE
            axis, point, pitch = screw(xi)
            row[f"w{i}"] = {"rot": float((ex["U"][3:, i] ** 2).sum()),
                            "loop_angle": float(np.degrees(np.arccos(min(1, abs(axis @ LOOP_AXIS))))),
                            "loop_dist": line_dist(point, axis, LOOP_CENTRE), "cen_dist": line_dist(point, axis, cen),
                            "pitch": float(pitch)}
        b = bench.get((r["stream"], r["step"]))
        if b is not None:
            rel = CI.relative(b["tm_pose"], r["truth"])  # the truth mode in the truth's frame
            c = ex["U"].T @ twist_of(rel)
            row["tm_ok"] = bool(np.linalg.norm(b["tm_err"][:3]) <= 2 and np.linalg.norm(b["tm_err"][3:]) <= 3)
            row["tm_share"] = np.cumsum(c ** 2)[:3] / (c ** 2).sum()
            row["tm_size"] = float(np.linalg.norm(c))
        for name, d in (("exact", ex), ("real", re)):
            lo, hi = d["loop_set"]
            row[f"{name}_loop_set"] = (lo, hi)
            row[f"{name}_loop_min"] = d["loop_min_s"]
            wlo, whi = d["weak_set"]
            row[f"{name}_weak_set"] = (wlo, whi)
            # rival modes outside the tolerance: in the span of the two weakest directions, or discrete
            kinds = []
            for m, cls in zip(d["mode_rel"], d["mode_class"]):
                if cls == "truth":
                    continue
                c = ex["U"].T @ twist_of(m)
                off = float(np.sqrt((c[2:] ** 2).sum()))  # tolerance units outside the two weakest directions
                kinds.append(("valley" if cls == "valley" else ("weak2" if off <= 0.5 else "discrete"), m, off))
            row[f"{name}_rivals"] = kinds
            row[f"{name}_search_ok"] = bool(np.linalg.norm(d["search_best_err"][:3]) <= 2
                                            and np.linalg.norm(d["search_best_err"][3:]) <= 3)
        rows.append(row)

    steps = sorted({r["step"] for r in rows})
    L = ["# Identifiability of the eyelet pose from the contact samples\n",
         f"{len(rows)} cases (policy streams 0-{targs['cases'] - 1}, steps {steps}), noise {targs['noise']}, sigma "
         f"{targs['sigma']} mm. Tolerance units: 2 mm, 3 deg. Fisher eigenvalue = cost increase of a one-tolerance "
         f"move along its direction at the truth; < {CHI2_1} means the direction is not determined to tolerance.\n"]
    L.append("## Local: Fisher eigenvalues at the truth (median per step)\n")
    L.append("| step | cases | samples | l1 | l2 | l3 | l4 | l5 | l6 | l1 < 3.84 | l2 < 3.84 | l3 < 3.84 | contact spread mm |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for st in steps + ["all"]:
        R = [r for r in rows if st == "all" or r["step"] == st]
        lam = np.array([r["lam"] for r in R])
        L.append(f"| {st} | {len(R)} | {np.median([r['used'] for r in R]):.0f} | "
                 + " | ".join(f"{v:.3g}" for v in np.median(lam, 0)) + " | "
                 + " | ".join(f"{np.mean(lam[:, i] < CHI2_1):.2f}" for i in range(3))
                 + f" | {np.median([r['spread'] for r in R]):.1f} |")
    L.append("\n## The three weakest directions as screws (median; share of cases)\n")
    L.append("| direction | rotation share | axis angle to loop axis, deg | axis to loop centre, mm | axis to contact "
             "centroid, mm | abs pitch, mm/rad | rotation > 0.9 | within 15 deg of loop axis |")
    L.append("|---|---|---|---|---|---|---|---|")
    for i in range(3):
        W = [r[f"w{i}"] for r in rows]
        g = lambda k: np.array([w[k] for w in W])
        L.append(f"| {i + 1} | {np.median(g('rot')):.2f} | {np.median(g('loop_angle')):.1f} | "
                 f"{np.median(g('loop_dist')):.2f} | {np.median(g('cen_dist')):.2f} | {np.median(np.abs(g('pitch'))):.2f} | "
                 f"{np.mean(g('rot') > 0.9):.2f} | {np.mean(g('loop_angle') < 15):.2f} |")
    T = [r for r in rows if "tm_ok" in r]
    wrong = [r for r in T if not r["tm_ok"]]
    if wrong:
        sh = np.array([r["tm_share"] for r in wrong])
        L.append(f"\n## Real data: where the truth mode's error lies\n\nTruth mode wrong in {len(wrong)} of {len(T)} "
                 f"cases. Share of its displacement (tolerance units) along the weakest direction / two weakest / three "
                 f"weakest at the truth: median {np.median(sh[:, 0]):.2f} / {np.median(sh[:, 1]):.2f} / "
                 f"{np.median(sh[:, 2]):.2f} (isotropic: 0.17 / 0.33 / 0.50). Its size: median "
                 f"{np.median([r['tm_size'] for r in wrong]):.2f} tolerance units.\n")
    L.append("## Profile along rotation about the loop's axis (95% confidence set, deg)\n")
    L.append("| data | step | set inside +-3 | set beyond +-3 | unbounded on +-30 | median extent | minimum beyond 3 deg "
             "of the truth |")
    L.append("|---|---|---|---|---|---|---|")
    for name in ("exact", "real"):
        for st in steps + ["all"]:
            R = [r for r in rows if st == "all" or r["step"] == st]
            lo = np.array([r[f"{name}_loop_set"][0] for r in R])
            hi = np.array([r[f"{name}_loop_set"][1] for r in R])
            ext = np.maximum(-lo, hi)
            inside = (lo >= -3) & (hi <= 3)
            unb = ~np.isfinite(ext)
            L.append(f"| {name} | {st} | {inside.mean():.2f} | {(~inside).mean():.2f} | {unb.mean():.2f} | "
                     f"{np.median(ext):.1f} | {np.mean(np.abs([r[f'{name}_loop_min'] for r in R]) > 3):.2f} |")
    L.append("\nProminent dips along the loop rotation (a local minimum at 14-22 deg with cost < 6 above the truth's, "
             "behind a barrier of at least 2): " + ", ".join(
                 f"step {st} {np.mean([prominent_dip(r['exact']['loop_cost'], s_loop) for r in recs if r['step'] == st]):.2f}"
                 for st in steps) + "\n")
    L.append("\n## Profile along the weakest screw (95% confidence set, tolerance units)\n")
    L.append("| data | step | set inside +-1 | unbounded on +-8 | median extent |")
    L.append("|---|---|---|---|---|")
    for name in ("exact", "real"):
        for st in steps + ["all"]:
            R = [r for r in rows if st == "all" or r["step"] == st]
            lo = np.array([r[f"{name}_weak_set"][0] for r in R])
            hi = np.array([r[f"{name}_weak_set"][1] for r in R])
            ext = np.maximum(-lo, hi)
            L.append(f"| {name} | {st} | {np.mean((lo >= -1) & (hi <= 1)):.2f} | {np.mean(~np.isfinite(ext)):.2f} | "
                     f"{np.median(ext):.2f} |")
    L.append("\n## Rival modes (1024-start Gauss-Newton; within 6 of the best cost, outside the tolerance)\n")
    L.append("valley: within one tolerance of a profile path; weak2: within 0.5 tolerance units of the span of the two "
             "weakest directions; discrete: neither.\n")
    L.append("| data | step | search best correct | any rival | valley | weak2 | discrete |")
    L.append("|---|---|---|---|---|---|---|")
    for name in ("exact", "real"):
        for st in steps + ["all"]:
            R = [r for r in rows if st == "all" or r["step"] == st]
            has = lambda kind: np.mean([any(k[0] == kind for k in r[f"{name}_rivals"]) for r in R])
            L.append(f"| {name} | {st} | {np.mean([r[f'{name}_search_ok'] for r in R]):.2f} | "
                     f"{np.mean([len(r[f'{name}_rivals']) > 0 for r in R]):.2f} | {has('valley'):.2f} | "
                     f"{has('weak2'):.2f} | {has('discrete'):.2f} |")
    disc = [(r["step"], k[1], k[2]) for r in rows for k in r["exact_rivals"] if k[0] == "discrete"]
    if disc:
        L.append("\nDiscrete rivals on the exact data, relative to the truth (eyelet frame: x y z mm, rotation angle deg, "
                 "rotation axis):\n")
        L.append("| step | x | y | z | angle | axis | off the weak span |")
        L.append("|---|---|---|---|---|---|---|")
        for st, m, off in sorted(disc, key=lambda t: t[0])[:40]:
            ang = np.linalg.norm(m[3:])
            ax = m[3:] / max(ang, 1e-9)
            L.append(f"| {st} | {m[0]:.2f} | {m[1]:.2f} | {m[2]:.2f} | {ang:.1f} | "
                     f"({ax[0]:.2f}, {ax[1]:.2f}, {ax[2]:.2f}) | {off:.2f} |")
    open(args.out, "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    np.save(args.out.replace(".md", "_rows.npy"), np.array(rows, dtype=object), allow_pickle=True)
    if args.fig:
        figure(rows, recs, s_loop, args.fig)
    if args.compare:
        figure_fields(recs, list(np.load(args.compare[0], allow_pickle=True)["records"]), s_loop, args.compare[1])


if __name__ == "__main__":
    main()
