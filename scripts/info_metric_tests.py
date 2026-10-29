#!/usr/bin/env python3
"""Hypothesis tests of the contact information metric (contact_information.py) on info_metric_bench.py records.

An estimate is correct when its error is within --tol (mm, deg; norms of the translation and rotation parts). The
estimate is the lowest-cost mode of the multi-start search; the "truth mode" is the search started at the truth
(an oracle reference). Criteria were fixed before the results were seen:

  H1 local observability  a few contacts constrain all six pose directions locally: the prior-whitened information
                          at the truth exceeds the prior in every direction (lambda_min > 1) in >= 90% of cases
                          with >= 20 samples.
  H2 local calibration    the truth mode's Laplace covariance predicts its error: per-axis |z| < 1.96 in 90-98% of
                          cases, median NEES within a factor 2 of chi2_6's (5.35).
  H3 failure taxonomy     wrong estimates are mostly global, not a lack of local information: of the failures,
                          < 25% have the truth mode itself off by more than the tolerance (local deficit); the rest
                          are ambiguity (a wrong mode fits as well as the truth mode: cost <= truth-mode cost + 6)
                          or search failure (the truth mode fits better but was not found).
  H4 fit test             the best mode's cost against chi2_N (N samples) flags search failures: AUC >= 0.9.
  H5 sufficiency rule     "sufficient" = fit passes (cost <= chi2_N 99.9%), the best mode's mixture weight >= 0.95
                          and its Laplace std <= --std-max (mm, deg): precision (share correct when sufficient)
                          >= 0.9, and its score (weight if the fit passes, else 0) ranks correctness better (AUC)
                          than the sample count, the information gain or the best mode's std alone.
  H6 calibrated weights   the best mode's mixture weight is a calibrated probability of being correct among cases
                          that pass the fit test: expected calibration error <= 0.1.
  H7 residual vs error    (the user's ICP finding, 2026-10-28: the residual tracks the true error only within the
                          convergence range) over every mode found: Spearman rho(cost / N, error) > 0 (p < 0.01) for
                          modes within 3x the tolerance of the truth, and |rho| < 0.2 or rho < 0 beyond 10x; within a
                          case, the lower-cost of two modes is the more accurate one only when both are near.
Writes --out (markdown) and prints it.

    python scripts/info_metric_tests.py --records R1.npz [R2.npz ...] --out report.md
"""

import argparse
from pathlib import Path

import numpy as np
from scipy.stats import chi2, rankdata, spearmanr

import contact_information as CI

AX = CI.AXES


def load(paths):
    recs = []
    for p in paths:
        data = np.load(p, allow_pickle=True)
        for r in data["records"]:
            if "mode_cost" in r and len(r["mode_cost"]):
                r = dict(r)
                r["set"] = Path(p).stem
                r["bound"] = float(data["bound"])
                recs.append(r)
    return recs


def correct(err, tol):
    return np.linalg.norm(err[:3]) <= tol[0] and np.linalg.norm(err[3:]) <= tol[1]


def auc(score, label):
    """Probability that a positive scores above a negative (ties half)."""
    score, label = np.asarray(score, float), np.asarray(label, bool)
    if label.all() or not label.any():
        return np.nan
    r = rankdata(score)
    n1 = label.sum()
    return (r[label].sum() - n1 * (n1 + 1) / 2) / (n1 * (~label).sum())


def local_cov(r):
    s = CI.prior_std(r["bound"])
    V, lam = r["local_vectors"], r["local_lambda"]
    return np.diag(s) @ V @ np.diag(1 / (1 + lam)) @ V.T @ np.diag(s)


def derive(r, tol, std_max):
    n = r["used"]
    best_std = np.sqrt(np.diag(r["mode_cov"][0]))
    fit_ok = r["mode_cost"][0] <= chi2.ppf(0.999, n)
    out = dict(set=r["set"], noise=r["noise"], step=r["step"], n=n, samples=r["samples"],
               correct=correct(r["mode_err"][0], tol), truth_mode_ok=correct(r["local_err"], tol),
               ambiguous=r["mode_cost"][0] <= r["local_cost"] + 6.0, fit_ok=fit_ok, cost_per_n=r["mode_cost"][0] / n,
               w=r["mode_weight"][0], gap=(r["mode_cost"][1] - r["mode_cost"][0]) if len(r["mode_cost"]) > 1 else 1e6,
               gain=r["mode_gain"][0], oracle_lam_min=r["oracle_lambda"].min(), best_std=best_std,
               std_ok=best_std[:3].max() <= std_max[0] and best_std[3:].max() <= std_max[1])
    out["sufficient"] = out["fit_ok"] and out["w"] >= 0.95 and out["std_ok"]
    out["score"] = out["w"] if fit_ok else 0.0
    e, C = r["local_err"], local_cov(r)
    out["z"] = e / np.sqrt(np.diag(C))
    out["nees"] = e @ np.linalg.solve(C, e)
    return out


def residual_vs_error(recs, tol):
    """H7 lines: Spearman rho between a mode's cost per sample and its error (in tolerance units, the larger of
    translation / tol[0] and rotation / tol[1]) by error band, over every mode found plus each truth mode; and the
    share of mode pairs within a case where the lower-cost mode is the more accurate."""
    def units(e):
        return max(np.linalg.norm(e[:3]) / tol[0], np.linalg.norm(e[3:]) / tol[1])

    pts = [(c / r["used"], units(e)) for r in recs for c, e in zip(r["mode_cost"], r["mode_err"])]
    pts += [(r["local_cost"] / r["used"], units(r["local_err"])) for r in recs]
    x, d = np.array(pts).T
    L = ["H7, residual (cost / N) vs error (d = error in tolerance units), every mode and truth mode:", "",
         "| error band | modes | cost/N p50 | Spearman rho | p |", "|---|---|---|---|---|"]
    for lo, hi in [(0, 1), (1, 3), (0, 3), (3, 10), (10, np.inf), (0, np.inf)]:
        m = (d >= lo) & (d < hi)
        rho, pv = spearmanr(x[m], d[m]) if m.sum() > 5 else (np.nan, np.nan)
        L.append(f"| {lo}-{hi} | {m.sum()} | {np.median(x[m]) if m.any() else np.nan:.2f} | {fmt(rho)} | {fmt(pv, 3)} |")
    near, far, mixed = [], [], []
    for r in recs:
        modes = [(c, units(e)) for c, e in zip(r["mode_cost"], r["mode_err"])] + [(r["local_cost"], units(r["local_err"]))]
        for i in range(len(modes)):
            for j in range(i + 1, len(modes)):
                (ci, di), (cj, dj) = modes[i], modes[j]
                if abs(di - dj) < 0.5 or abs(ci - cj) < 1e-6:
                    continue
                ok = (ci < cj) == (di < dj)
                (near if max(di, dj) <= 3 else far if min(di, dj) > 3 else mixed).append(ok)
    L += ["", "| mode pairs in a case | pairs | lower cost is more accurate |", "|---|---|---|"]
    for name, v in [("both within 3x tolerance", near), ("one near, one beyond", mixed), ("both beyond 3x", far)]:
        L.append(f"| {name} | {len(v)} | {fmt(np.mean(v) if v else np.nan)} |")
    return L + [""]


def fmt(x, nd=2):
    return "-" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--records", nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--tol", type=float, nargs=2, default=[2.0, 3.0], help="mm, deg")
    p.add_argument("--std-max", type=float, nargs=2, default=[1.0, 1.5], help="mm, deg")
    args = p.parse_args()
    recs = load(args.records)
    rows = [derive(r, args.tol, args.std_max) for r in recs]
    L = [f"# Contact information metric: hypothesis tests", "",
         f"Correct = best-mode error within {args.tol[0]} mm and {args.tol[1]} deg (norms); "
         f"sufficient = fit passes, weight >= 0.95, Laplace std <= {args.std_max[0]} mm / {args.std_max[1]} deg.", ""]
    groups = sorted({(r["set"], r["noise"]) for r in rows})
    for set_, noise in groups:
        G = [r for r in rows if r["set"] == set_ and r["noise"] == noise]
        steps = sorted({r["step"] for r in G})
        L += [f"## {set_}, hook-pose noise {noise} mm / deg ({len(G)} cases)", "",
              "| step | n | samples p50 | correct | H1 lam_min>1 | truth mode ok | failures: ambiguous / search / local "
              "| H2 |z|<1.96 x..rz | H2 NEES p50 | fit ok | sufficient: share / precision | weight ECE |",
              "|" + "---|" * 12]
        for step in steps + ["all"]:
            S = [r for r in G if step == "all" or r["step"] == step]
            S20 = [r for r in S if r["samples"] >= 20]
            fail = [r for r in S if not r["correct"]]
            local = sum(not r["truth_mode_ok"] for r in fail)
            amb = sum(r["truth_mode_ok"] and r["ambiguous"] for r in fail)
            srch = sum(r["truth_mode_ok"] and not r["ambiguous"] for r in fail)
            cov = np.mean([np.abs(r["z"]) < 1.96 for r in S], axis=0)
            suff = [r for r in S if r["sufficient"]]
            prec = np.mean([r["correct"] for r in suff]) if suff else np.nan
            fit = [r for r in S if r["fit_ok"]]
            ece = np.nan
            if fit:
                w, c = np.array([r["w"] for r in fit]), np.array([r["correct"] for r in fit], float)
                bins = np.digitize(w, [0.5, 0.8, 0.95, 0.99])
                ece = sum(abs(w[bins == b].mean() - c[bins == b].mean()) * (bins == b).sum()
                          for b in np.unique(bins)) / len(w)
            L.append(f"| {step} | {len(S)} | {np.median([r['samples'] for r in S]):.0f} | "
                     f"{np.mean([r['correct'] for r in S]):.2f} | "
                     f"{fmt(np.mean([r['oracle_lam_min'] > 1 for r in S20]) if S20 else np.nan)} | "
                     f"{np.mean([r['truth_mode_ok'] for r in S]):.2f} | {amb} / {srch} / {local} | "
                     f"{' '.join(f'{v:.2f}' for v in cov)} | {np.median([r['nees'] for r in S]):.1f} | "
                     f"{np.mean([r['fit_ok'] for r in S]):.2f} | {len(suff) / len(S):.2f} / {fmt(prec)} | {fmt(ece)} |")
        lab = np.array([r["correct"] for r in G])
        srch_lab = np.array([r["truth_mode_ok"] and not r["ambiguous"] and not r["correct"] for r in G])
        L += ["", "Ranking correctness (AUC; H5: the score should beat the others; H4: cost/N for search failures):", "",
              "| score | AUC correct |", "|---|---|"]
        for name, sc in [("H5 score (weight if fit passes)", [r["score"] for r in G]),
                         ("sample count", [r["samples"] for r in G]),
                         ("information gain at the best mode (nats)", [r["gain"] for r in G]),
                         ("- max Laplace std, translation (mm)", [-r["best_std"][:3].max() for r in G]),
                         ("- max Laplace std, rotation (deg)", [-r["best_std"][3:].max() for r in G]),
                         ("cost gap to the second mode", [r["gap"] for r in G]),
                         ("best mode weight alone", [r["w"] for r in G]),
                         ("oracle lambda_min (not deployable)", [r["oracle_lam_min"] for r in G])]:
            L.append(f"| {name} | {fmt(auc(sc, lab))} |")
        L += [f"| H4: cost/N, search failures vs the rest | {fmt(auc([r['cost_per_n'] for r in G], srch_lab))} |", ""]
        L += residual_vs_error([r for r in recs if r["set"] == set_ and r["noise"] == noise], args.tol)
        fit = [r for r in G if r["fit_ok"]]
        if fit:
            L += ["Weight reliability (cases passing the fit test):", "", "| weight bin | cases | mean weight | correct |",
                  "|---|---|---|---|"]
            w = np.array([r["w"] for r in fit])
            c = np.array([r["correct"] for r in fit], float)
            for lo, hi in [(0, 0.5), (0.5, 0.8), (0.8, 0.95), (0.95, 0.99), (0.99, 1.01)]:
                m = (w >= lo) & (w < hi)
                if m.any():
                    L.append(f"| {lo}-{min(hi, 1)} | {m.sum()} | {w[m].mean():.2f} | {c[m].mean():.2f} |")
            L.append("")
    text = "\n".join(L) + "\n"
    args.out.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
