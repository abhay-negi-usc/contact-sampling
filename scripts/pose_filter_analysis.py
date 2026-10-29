#!/usr/bin/env python3
"""Phase A analysis of pose_filter_bench.py records: accuracy, search vs identifiability failures, error axes.

correct      error norms within --tol (mm, deg), of the polished estimate (pol), the best raw particle (raw) and the
             weighted mean (mean);
search fail  the estimate's cost exceeds the truth mode's (Gauss-Newton from the truth) by more than --margin: a
             better solution exists that the method did not find;
ident. fail  wrong although no worse in cost than the truth mode: the data do not single out the truth;
truth mode   whether the truth's own mode is correct (if not, even a perfect search is wrong: noise / model error).
With --lm (info_metric_bench records), the old 24-start Levenberg-Marquardt's best mode on the same cases.

    python scripts/pose_filter_analysis.py --records R1.npz R2.npz --lm policy.npz --out A.md
"""

import argparse
from collections import defaultdict

import numpy as np


def load(paths):
    """Records of every file; a method run in several files (a sweep) is labelled by the file's stem."""
    recs = []
    for p in paths:
        stem = p.rsplit("/", 1)[-1].rsplit(".", 1)[0]
        for r in np.load(p, allow_pickle=True)["records"]:
            r["method"] = stem if stem.startswith("seq_") else r["method"]
            recs.append(r)
    return recs


LOOP_CENTRE = np.array([0.0, -0.28, -5.99])  # eyelet frame, mm: the loop's best-fit circle (axis along x)


def modulo_loop(truth, est, grid=np.arange(-45, 45.01, 0.5)):
    """Error of est against the truth turned about the loop's own axis by the angle that fits best (the loop is a
    near circle: rms 0.38 mm off one): (x, y, z mm, rotation vector deg), the error modulo that near-symmetry."""
    from scipy.spatial.transform import Rotation
    import contact_information as CI
    r_t = Rotation.from_rotvec(np.deg2rad(truth[3:]))
    best, best_e = np.inf, None
    for th in grid:
        rx = Rotation.from_rotvec([np.deg2rad(th), 0, 0])
        t2 = np.concatenate([truth[:3] + r_t.apply(LOOP_CENTRE - rx.apply(LOOP_CENTRE)),
                             (r_t * rx).as_rotvec(degrees=True)])
        e = CI.relative(t2, est)
        score = max(np.linalg.norm(e[:3]) / 2.0, np.linalg.norm(e[3:]) / 3.0)
        if score < best:
            best, best_e = score, e
    return best_e


def ok(e, tol):
    e = np.atleast_2d(e)
    return (np.linalg.norm(e[:, :3], axis=1) <= tol[0]) & (np.linalg.norm(e[:, 3:], axis=1) <= tol[1])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--records", nargs="+", required=True)
    p.add_argument("--lm", help="info_metric_bench records of the same cases")
    p.add_argument("--tol", type=float, nargs=2, default=[2.0, 3.0])
    p.add_argument("--margin", type=float, default=6.0)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    recs = load(args.records)
    lm = {}
    if args.lm:
        for r in np.load(args.lm, allow_pickle=True)["records"]:
            if "mode_err" in r and len(r["mode_err"]):
                lm[(r["stream"], r["step"], float(r["noise"]))] = r["mode_err"][0]
    groups = defaultdict(list)
    for r in recs:
        groups[(r["method"], float(r["noise"]))].append(r)
    tol = args.tol
    lines = [f"# Particle estimators: accuracy and failure modes\n",
             f"Correct = error norms <= {tol[0]} mm and {tol[1]} deg. Search failure = cost > truth-mode cost + "
             f"{args.margin}. Identifiability failure = wrong yet no worse than the truth mode.\n"]
    lines.append("| method | noise | cases | correct pol | raw | mean | 5/5 pol | truth mode correct | search fail | "
                 "ident. fail | old LM correct (same cases) | median time s |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for (m, noise), rs in sorted(groups.items()):
        pol = ok(np.array([r["pol_err"] for r in rs]), tol)
        raw = ok(np.array([r["raw_err"] for r in rs]), tol)
        mean = ok(np.array([r["mean_err"] for r in rs]), tol)
        p55 = ok(np.array([r["pol_err"] for r in rs]), (5, 5))
        tm = ok(np.array([r["tm_err"] for r in rs]), tol)
        delta = np.array([r["pol_cost"] - r["tm_cost"] for r in rs])
        sfail = delta > args.margin
        ifail = ~pol & ~sfail
        keys = [(r["stream"], r["step"], float(r["noise"])) for r in rs]
        old = [ok(lm[k], tol)[0] for k in keys if k in lm]
        lines.append(f"| {m} | {noise} | {len(rs)} | {pol.mean():.2f} | {raw.mean():.2f} | {mean.mean():.2f} | "
                     f"{p55.mean():.2f} | {tm.mean():.2f} | {sfail.mean():.2f} | {ifail.mean():.2f} | "
                     f"{np.mean(old) if old else float('nan'):.2f} ({len(old)}) | "
                     f"{np.median([r['time_s'] for r in rs]):.1f} |")
    lines.append("\n## By policy step (polished estimate correct; search failure)\n")
    steps = sorted({r["step"] for r in recs})
    lines.append("| method | noise | " + " | ".join(f"step {s}" for s in steps) + " |")
    lines.append("|---|---|" + "---|" * len(steps))
    for (m, noise), rs in sorted(groups.items()):
        cells = []
        for s in steps:
            sel = [r for r in rs if r["step"] == s]
            if not sel:
                cells.append("-")
                continue
            c = ok(np.array([r["pol_err"] for r in sel]), tol).mean()
            f = np.mean([r["pol_cost"] - r["tm_cost"] > args.margin for r in sel])
            cells.append(f"{c:.2f} / {f:.2f}")
        lines.append(f"| {m} | {noise} | " + " | ".join(cells) + " |")
    lines.append("\n## Error per axis (median |error| of the polished estimate; x y z mm, rx ry rz deg)\n")
    lines.append("| method | noise | x | y | z | rx | ry | rz | truth mode: x | y | z | rx | ry | rz |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for (m, noise), rs in sorted(groups.items()):
        e = np.median(np.abs([r["pol_err"] for r in rs]), axis=0)
        t = np.median(np.abs([r["tm_err"] for r in rs]), axis=0)
        lines.append(f"| {m} | {noise} | " + " | ".join(f"{v:.2f}" for v in e) + " | "
                     + " | ".join(f"{v:.2f}" for v in t) + " |")
    lines.append("\n## Correct modulo the loop's near-symmetry (truth turned about the loop's axis to fit best)\n")
    lines.append("| method | noise | correct pol, modulo loop | truth mode, modulo loop |")
    lines.append("|---|---|---|---|")
    for (m, noise), rs in sorted(groups.items()):
        e = np.array([modulo_loop(r["truth"], r["pol_pose"]) for r in rs])
        t = np.array([modulo_loop(r["truth"], r["tm_pose"]) for r in rs])
        lines.append(f"| {m} | {noise} | {ok(e, tol).mean():.2f} | {ok(t, tol).mean():.2f} |")
    open(args.out, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
