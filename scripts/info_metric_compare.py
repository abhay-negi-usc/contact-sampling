#!/usr/bin/env python3
"""Comparison of information and uncertainty measures (contact_information.py) on info_metric_bench.py records and
their info_metric_extras.py quantities, against the truth.

Information (does the data suffice?): each measure scores every case; AUC of the score for "the best mode is
correct" (error within --tol). Measures:
  count                 samples so far (baseline);
  FIM D / E / A, DFS    Gauss-Newton Fisher information at the best mode, prior-whitened: Rodgers' information
                        content (bits), lambda_min, -trace of the whitened posterior, degrees of freedom for signal;
  observed D            information content of the observed (full-Hessian) information;
  sandwich A            -trace of the whitened robust posterior;
  entropy gain          prior entropy minus the Laplace mixture's (upper bound, Rényi-2);
  mode entropy          -H(weights) over the modes found (ambiguity);
  weight, fit p, H5     the best mode's mixture weight, the chi2 p-value of its cost, the H5 score;
  region mass           the Laplace mixture's mass within --tol of the best mode (a decision-region test, Javdani
                        et al. 2014): the probability it is correct, also checked for calibration;
  consistent modes      a conformal set-membership test (after Yang & Pavone 2023's PURSE): the bound on a case's
                        largest contact residual at the truth calibrated on even streams (95%), then the number of
                        distinct modes found within it; sufficient = exactly one, the best.
Uncertainty (is the predicted spread right?): for each method the truth's highest-posterior-density level u over
cases (uniform if calibrated): coverage of the 68% and 95% regions, KS distance from uniform, median NEES (single
Gaussians). Methods: Gauss-Newton Laplace, observed information, sandwich, cluster sandwich (clusters: policy steps),
Laplace mixtures over the modes (Gauss-Newton or sandwich covariances); with a mounting error, the consider
covariance for its std (Tapley et al.; Brossard et al.'s bias term). Split conformal: each method's
nonconformity score (-log density at the truth) calibrated on even streams, coverage on odd streams and on the
other sets; the inflation the 95% region needs. Profile likelihood: per-axis 95% interval coverage on the profiled
subset, against the Laplace intervals on the same cases. Writes --out (markdown) and prints it.

    python scripts/info_metric_compare.py --pairs R.npz:R_extras.npz [...] --out compare.md
"""

import argparse
from pathlib import Path

import numpy as np
from scipy.stats import chi2, kstest

import contact_information as CI
from info_metric_tests import auc, correct, fmt


def merge(pairs):
    cases = []
    for pair in pairs:
        rec_path, ext_path = pair.split(":")
        data = np.load(rec_path, allow_pickle=True)
        sig = dict(zip(np.atleast_1d(data["noise"]), np.atleast_1d(data["sigma"])))
        ext = {(e["stream"], e["step"], e["noise"]): e for e in np.load(ext_path, allow_pickle=True)["records"]}
        for r in data["records"]:
            key = (r["stream"], r["step"], r["noise"])
            if key in ext:
                c = {**dict(r), **ext[key], "set": Path(rec_path).stem, "sigma": sig[r["noise"]],
                     "bound": float(data["bound"])}
                cases.append(c)
    return cases


def whiten_trace(C, s):
    return float(np.trace(C / np.outer(s, s)))


def measures(c, tol, rng):
    """Scores and per-method (weights, means, covs) of one case."""
    s = CI.prior_std(c["bound"])
    n, sigma = c["used"], c["sigma"]
    J, r = c["mode_J"][0], c["mode_r"][0]
    gn = CI.information_metrics(J, sigma, s)
    obs_cov, nonpos = CI.observed_cov(c["hess_best"], s)
    sand = CI.sandwich_cov(J, r, sigma, s)
    clus = CI.sandwich_cov(J, r, sigma, s, c["sample_steps"])
    means = np.array([CI.relative(g, c["mode_pose"][0]) for g in c["mode_pose"]])
    w = c["mode_weight"]
    covs_gn = list(c["mode_cov"])
    covs_sw = [CI.sandwich_cov(Jk, rk, sigma, s) for Jk, rk in zip(c["mode_J"], c["mode_r"])]
    logw = -0.5 * c["mode_cost"] + 0.5 * np.array([np.linalg.slogdet(C)[1] for C in covs_sw])
    w_sw = np.exp(logw - logw.max())
    w_sw = w_sw / w_sw.sum()
    H0 = CI.uniform_entropy(c["bound"])
    ent = CI.mixture_entropies(w, means, covs_gn)
    obs_lam = np.linalg.eigvalsh(np.diag(s) @ np.linalg.inv(obs_cov) @ np.diag(s) - np.eye(6))
    score = {
        "count": c["samples"],
        "FIM D (info bits)": CI.rodgers(gn["lambda"])["info_bits"],
        "FIM E (lambda_min)": gn["lambda"].min(),
        "FIM A (-trace)": -whiten_trace(gn["post_cov"], s),
        "Rodgers DFS": CI.rodgers(gn["lambda"])["dfs"],
        "observed D (info bits)": CI.rodgers(obs_lam)["info_bits"],
        "sandwich A (-trace)": -whiten_trace(sand, s),
        "entropy gain, mixture upper bound": H0 - ent["upper"],
        "entropy gain, Renyi-2": H0 - ent["renyi2"],
        "-mode entropy H(w)": -ent["mode"],
        "best mode weight": w[0],
        "fit p-value": chi2.sf(c["mode_cost"][0], n),
        "H5 score": w[0] if c["mode_cost"][0] <= chi2.ppf(0.999, n) else 0.0,
    }
    score["H5 score x fit p"] = score["H5 score"] * min(1.0, 10 * score["fit p-value"])
    score["decision-region mass"] = CI.region_mass(w, means, covs_gn, tol, rng)
    methods = {"Gauss-Newton Laplace": ([1.0], means[:1], [gn["post_cov"]]),
               "observed information": ([1.0], means[:1], [obs_cov]),
               "sandwich": ([1.0], means[:1], [sand]),
               "cluster sandwich": ([1.0], means[:1], [clus]),
               "Laplace mixture (GN)": (w, means, covs_gn),
               "Laplace mixture (sandwich)": (w_sw, means, covs_sw)}
    if c.get("bias", 0.0) > 0:
        methods["consider covariance"] = ([1.0], means[:1], [CI.consider_cov(J, c["mount_J"], sigma, s, c["bias"])])
    e = c["mode_err"][0]
    u, nll, nees = {}, {}, {}
    for name, (ww, mm, cc) in methods.items():
        nees[name] = (e - mm[0]) @ np.linalg.solve(cc[0], e - mm[0]) if len(ww) == 1 else np.nan
        u[name] = CI.hpd_level(e, ww, mm, cc, rng, 3000) if len(ww) > 1 else chi2.cdf(nees[name], 6)
        nll[name] = -CI.mixture_logpdf(e, ww, mm, cc)
    return dict(score=score, u=u, nll=nll, nees_m=nees, nonpos=nonpos, correct=correct(e, tol),
                max_truth_r=np.abs(c["truth_r"]).max(), max_mode_r=np.array([np.abs(r).max() for r in c["mode_r"]]),
                nees=e @ np.linalg.solve(gn["post_cov"], e), err=e, std_gn=np.sqrt(np.diag(gn["post_cov"])))


def profile_lines(cases, tol):
    prof = [c for c in cases if "profile" in c]
    if not prof:
        return []
    hits_p, hits_l, widths = [], [], []
    for c in prof:
        grid = c["profile_grid"]
        e = c["mode_err"][0]
        s = CI.prior_std(c["bound"])
        sd = np.sqrt(np.diag(c["mode_cov"][0]))
        for a in range(6):
            pr = np.r_[c["profile"][a], 0.0]
            g = np.r_[grid, 0.0]
            o = np.argsort(g)
            g, pr = g[o], pr[o] - min(0.0, pr.min())  # from the profile's own minimum
            inside = pr <= 3.84
            lo = g[inside].min() if inside.any() else 0.0
            hi = g[inside].max() if inside.any() else 0.0
            lo = -np.inf if inside[0] else lo  # still inside at the grid's end: open interval
            hi = np.inf if inside[-1] else hi
            hits_p.append(lo <= e[a] <= hi)
            hits_l.append(abs(e[a]) <= 1.96 * sd[a])
            widths.append((hi - lo) if np.isfinite(hi - lo) else np.nan)
    hits_p, hits_l = np.array(hits_p).reshape(-1, 6), np.array(hits_l).reshape(-1, 6)
    open_ = np.isnan(np.array(widths)).reshape(-1, 6)
    return ["", f"Profile likelihood ({len(prof)} profiled cases; 95% per-axis intervals, x y z rx ry rz):", "",
            "| interval | coverage per axis | open (flat beyond the +-8 grid) |", "|---|---|---|",
            f"| profile likelihood | {' '.join(f'{v:.2f}' for v in hits_p.mean(0))} | "
            f"{' '.join(f'{v:.2f}' for v in open_.mean(0))} |",
            f"| Laplace (Gauss-Newton) | {' '.join(f'{v:.2f}' for v in hits_l.mean(0))} | - |", ""]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pairs", nargs="+", required=True, help="records.npz:extras.npz")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--tol", type=float, nargs=2, default=[2.0, 3.0])
    args = p.parse_args()
    rng = np.random.default_rng(0)
    cases = merge(args.pairs)
    for c in cases:
        c.update(measures(c, args.tol, rng))
    L = ["# Information and uncertainty measures compared", "",
         f"Correct = best-mode error within {args.tol[0]} mm and {args.tol[1]} deg (norms).", ""]
    groups = sorted({(c["set"], c["noise"], c.get("bias", 0.0)) for c in cases})
    for g in groups:
        G = [c for c in cases if (c["set"], c["noise"], c.get("bias", 0.0)) == g]
        lab = np.array([c["correct"] for c in G])
        even_g = np.array([c["stream"] % 2 == 0 for c in G])
        mt = np.array([c["max_truth_r"] for c in G])
        eps = np.quantile(mt[even_g], min(1.0, 0.95 * (1 + 1 / even_g.sum())))
        for c in G:
            c["n_consistent"] = int((c["max_mode_r"] <= eps).sum())
            c["score"]["-consistent modes (conformal set)"] = -c["n_consistent"] if c["max_mode_r"][0] <= eps else -99
        L += [f"## {g[0]}, noise {g[1]}, bias {g[2]} ({len(G)} cases, {lab.mean():.2f} correct)", "",
              "| information measure | AUC correct | AUC, cases passing the fit test |", "|---|---|---|"]
        fit = np.array([c["score"]["fit p-value"] > 0.001 for c in G])
        for name in G[0]["score"]:
            sc = np.array([c["score"][name] for c in G])
            L.append(f"| {name} | {fmt(auc(sc, lab))} | {fmt(auc(sc[fit], lab[fit]))} |")
        L += ["", "| uncertainty method | cases | 68% coverage | 95% coverage | KS from uniform | median NEES | "
              "std inflation for 95% (even streams) | conformal coverage, odd streams |", "|---|---|---|---|---|---|---|---|"]
        for subset, S in [("all", G), ("correct", [c for c in G if c["correct"]])]:
            if not S:
                continue
            for name in S[0]["u"]:
                u = np.array([c["u"][name] for c in S])
                nll = np.array([c["nll"][name] for c in S])
                even = np.array([c["stream"] % 2 == 0 for c in S])
                q = np.quantile(nll[even], min(1.0, 0.95 * (1 + 1 / max(even.sum(), 1)))) if even.any() else np.nan
                cov_odd = np.mean(nll[~even] <= q) if (~even).any() else np.nan
                ne = np.array([c["nees_m"][name] for c in S])
                infl = np.sqrt(np.quantile(ne[even], 0.95) / chi2.ppf(0.95, 6)) if even.any() and np.isfinite(ne).all() else np.nan
                L.append(f"| {name} ({subset}) | {len(S)} | {np.mean(u <= 0.683):.2f} | {np.mean(u <= 0.95):.2f} | "
                         f"{kstest(u, 'uniform').statistic:.2f} | {fmt(np.median(ne), 1)} | {fmt(infl, 1)} | {fmt(cov_odd)} |")
        one = np.array([c["n_consistent"] == 1 and c["max_mode_r"][0] <= eps for c in G])
        L += ["", f"Conformal consistent set: residual bound {eps:.2f} mm (95%, even streams); the truth satisfies it in "
              f"{np.mean(mt[~even_g] <= eps):.2f} of odd-stream cases; exactly one consistent mode (the best) in "
              f"{one.mean():.2f} of cases, correct in {fmt(lab[one].mean() if one.any() else np.nan)} of those "
              f"(odd streams: {fmt(lab[one & ~even_g].mean() if (one & ~even_g).any() else np.nan)}); no consistent mode in "
              f"{np.mean([c['n_consistent'] == 0 for c in G]):.2f}."]
        fit_ok = np.array([c["score"]["fit p-value"] > 0.001 for c in G])
        rm = np.array([c["score"]["decision-region mass"] for c in G])
        L += ["", "Decision-region mass as a probability of being correct (cases passing the fit test):", "",
              "| mass bin | cases | mean mass | correct |", "|---|---|---|---|"]
        for lo, hi in [(0, 0.5), (0.5, 0.8), (0.8, 0.95), (0.95, 0.99), (0.99, 1.01)]:
            m = fit_ok & (rm >= lo) & (rm < hi)
            if m.any():
                L.append(f"| {lo}-{min(hi, 1)} | {m.sum()} | {rm[m].mean():.2f} | {lab[m].mean():.2f} |")
        nonpos = np.mean([c["nonpos"] > 0 for c in G])
        L += ["", f"Observed information has a non-positive curvature direction at the best mode in {nonpos:.2f} of "
              f"cases; median NEES (Gauss-Newton) {np.median([c['nees'] for c in G]):.1f} (chi2_6: 5.35).", ""]
        L += profile_lines(G, args.tol)
    sets = sorted({c["set"] for c in cases})
    if len(sets) > 1:
        L += ["## Transfer of conformal calibration (calibrated on all cases of one set, applied to another)", "",
              "| method | from -> to | 95% coverage |", "|---|---|---|"]
        for name in cases[0]["nll"]:
            for a in sets:
                for b in sets:
                    if a == b:
                        continue
                    A = np.array([c["nll"][name] for c in cases if c["set"] == a])
                    Bv = np.array([c["nll"][name] for c in cases if c["set"] == b])
                    q = np.quantile(A, 0.95)
                    L.append(f"| {name} | {a} -> {b} | {np.mean(Bv <= q):.2f} |")
        L.append("")
    text = "\n".join(L) + "\n"
    args.out.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
