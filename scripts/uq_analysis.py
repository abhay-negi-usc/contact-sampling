#!/usr/bin/env python3
"""Phase B: information metrics and uncertainty quantification from particle posteriors (pose_filter_bench.py).

The posterior of the estimate's error e = relative(truth, estimate) is represented by the particles x_j as
e_j = relative(x_j, estimate). Per case:
  P(correct)      share of particles within --tol of the estimate: the posterior probability the estimate is correct
  calibration     marginal PIT per axis (share of e_j below the true error; uniform when calibrated), HPD level of the
                  true error under a Gaussian KDE of the e_j (share of particles of higher density; uniform when
                  calibrated), NEES of the true error against the particles' mean and covariance (chi2_6 when
                  calibrated and Gaussian)
  entropy         Gaussian entropy of the particle covariance, Kozachenko-Leonenko k-NN entropy, both as information
                  gain over the +-bound uniform prior (nats)
  modes           distinct polished modes within 6 of the best cost (multimodality), and the particle mass outside the
                  estimate's cluster
  local           Fisher information at the estimate (J^T J, whitened), its D / E / A criteria, Rodgers' DFS, the
                  Laplace covariance (with the prior's) and its ratio to the particle covariance
  fit             chi2 p-value of the estimate's cost on used - 6 degrees of freedom, rms residual
Tests B1-B5 (HYPOTHESES.md). For SMC records with snapshots, B1 also at each power posterior (sigma / sqrt(beta)).

    python scripts/uq_analysis.py --records policy_smc.npz --out B.md
"""

import argparse

import numpy as np
from scipy import special, stats
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

AXES = ("x", "y", "z", "rx", "ry", "rz")


def relative_many(x, ref):
    """Poses x (n, 6) in ref's frame (n, 6): contact_information.relative, vectorised."""
    r_ref = Rotation.from_rotvec(np.deg2rad(ref[3:]))
    pos = r_ref.inv().apply(x[:, :3] - ref[:3])
    rot = (r_ref.inv() * Rotation.from_rotvec(np.deg2rad(x[:, 3:]))).as_rotvec(degrees=True)
    return np.concatenate([pos, rot], axis=1)


def within(e, tol):
    e = np.atleast_2d(e)
    return (np.linalg.norm(e[:, :3], axis=1) <= tol[0]) & (np.linalg.norm(e[:, 3:], axis=1) <= tol[1])


def knn_entropy(x, k=4):
    """Kozachenko-Leonenko differential entropy (nats) of samples x (n, d), duplicates removed."""
    x = np.unique(np.round(x, 6), axis=0)
    n, d = x.shape
    if n <= k + 1:
        return np.nan
    eps = cKDTree(x).query(x, k + 1)[0][:, k]
    eps = np.maximum(eps, 1e-9)
    log_vd = d / 2 * np.log(np.pi) - special.gammaln(d / 2 + 1)
    return special.digamma(n) - special.digamma(k) + log_vd + d * np.mean(np.log(eps))


def hpd_level(e, e_true):
    """Share of particles whose Gaussian-KDE density exceeds the density at the true error (Scott bandwidth)."""
    n, d = e.shape
    cov = np.cov(e.T) * n ** (-2.0 / (d + 4)) + 1e-9 * np.eye(d)
    Li = np.linalg.inv(np.linalg.cholesky(cov))
    z = e @ Li.T
    zt = e_true @ Li.T

    d2 = ((z[:, None, :] - z[None]) ** 2).sum(-1)
    np.fill_diagonal(d2, np.inf)  # leave one out: a particle's own kernel would inflate its density
    dens = special.logsumexp(-0.5 * d2, axis=1) - np.log(n - 1)
    dens_t = special.logsumexp(-0.5 * ((zt[None] - z) ** 2).sum(-1)) - np.log(n)
    return float(np.mean(dens > dens_t))


def auc(score, label):
    score, label = np.asarray(score, float), np.asarray(label, bool)
    m = np.isfinite(score)
    score, label = score[m], label[m]
    if label.all() or not label.any():
        return np.nan
    return stats.mannwhitneyu(score[label], score[~label]).statistic / (label.sum() * (~label).sum())


def ece(p, y, bins=10):
    p, y = np.asarray(p), np.asarray(y, float)
    edges = np.linspace(0, 1, bins + 1)
    out = 0.0
    for a, b in zip(edges[:-1], edges[1:]):
        m = (p >= a) & ((p < b) | (b == 1))
        if m.any():
            out += m.mean() * abs(p[m].mean() - y[m].mean())
    return out


def beta_rule(resid, sigma):
    """Power-posterior exponent from the fit: (sigma / sigma_hat)^2 times k_eff / k, sigma_hat the residuals' rms at
    the estimate and k_eff / k = (1 - rho) / (1 + rho) for their lag-1 autocorrelation rho in sample order (an AR(1)
    effective sample size: correlated errors along the contact path carry less information than k independent
    ones). Capped at 1."""
    r = np.asarray(resid, float)
    s_hat2 = max(np.mean(r ** 2), 1e-12)
    rho = np.corrcoef(r[:-1], r[1:])[0, 1] if len(r) > 3 and r.std() > 0 else 0.0
    rho = np.clip(np.nan_to_num(rho), 0.0, 0.95)
    return float(min(1.0, sigma ** 2 / s_hat2 * (1 - rho) / (1 + rho)))


def max_mass(particles, tol, extra=None):
    """The Bayes estimate for a +-tol criterion: among the particles (and extra candidates), the pose with the most
    particle mass within tol of it (position distance and rotation angle are frame-free). Returns (pose, mass)."""
    x = particles.astype(np.float64)
    cand = x if extra is None else np.vstack([x, np.asarray(extra, np.float64)])
    q = Rotation.from_rotvec(np.deg2rad(x[:, 3:])).as_quat()
    qc = Rotation.from_rotvec(np.deg2rad(cand[:, 3:])).as_quat()
    dpos = np.linalg.norm(cand[:, None, :3] - x[None, :, :3], axis=-1)
    ang = 2 * np.degrees(np.arccos(np.clip(np.abs(qc @ q.T), 0, 1)))
    mass = ((dpos <= tol[0]) & (ang <= tol[1])).mean(1)
    b = int(np.argmax(mass))
    return cand[b], float(mass[b])


def case_metrics(r, particles, tol, bound, prior_std):
    est = r["pol_pose"]
    e_true = r["pol_err"]
    e = relative_many(particles.astype(np.float64), est)
    out = {"p_correct": within(e, tol).mean(), "p_correct55": within(e, (5, 5)).mean()}
    mu, cov = e.mean(0), np.cov(e.T) + 1e-9 * np.eye(6)
    out["post_std"] = np.sqrt(np.diag(cov))
    out["pit"] = (e < e_true).mean(0)
    out["hpd"] = hpd_level(e, e_true)
    out["nees"] = float((e_true - mu) @ np.linalg.solve(cov, e_true - mu))
    h_prior = 6 * np.log(2 * bound)
    sign, logdet = np.linalg.slogdet(cov)
    out["gain_gauss"] = h_prior - 0.5 * (6 * np.log(2 * np.pi * np.e) + logdet)
    out["gain_knn"] = h_prior - knn_entropy(e)
    out["trace_post"] = float(np.trace(cov))
    out["rx_std"] = out["post_std"][3]
    return out


def local_metrics(r, bound):
    info = r["pol_info"]
    s_prior = np.full(6, bound / np.sqrt(3))
    lam = np.clip(np.linalg.eigvalsh(np.diag(s_prior) @ info @ np.diag(s_prior)), 0, None)
    lap = np.linalg.inv(info + np.diag(1 / s_prior ** 2))
    return {"fim_d": 0.5 * np.log1p(lam).sum(), "fim_e": lam.min(), "fim_a": -np.trace(lap),
            "dfs": (lam / (1 + lam)).sum(), "lap_std": np.sqrt(np.diag(lap)), "lap_cov": lap}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--records", nargs="+", required=True)
    p.add_argument("--tol", type=float, nargs=2, default=[2.0, 3.0])
    p.add_argument("--bound", type=float, default=20.0)
    p.add_argument("--out", required=True)
    p.add_argument("--max-stream", type=int, help="only streams below this index")
    p.add_argument("--steps", type=int, nargs="+", help="only these policy steps")
    p.add_argument("--label", help="method label for every record (comparing runs of one method)")
    args = p.parse_args()
    recs = []
    for path in args.records:
        recs += list(np.load(path, allow_pickle=True)["records"])
    recs = [r for r in recs if (args.max_stream is None or r["stream"] < args.max_stream)
            and (args.steps is None or r["step"] in args.steps)]
    if args.label:
        for r in recs:
            r["method"] = args.label
    tol = args.tol
    rows = []
    for r in recs:
        m = {"method": r["method"], "noise": float(r["noise"]), "step": r["step"], "stream": r["stream"],
             "used": r["used"], "samples": r["samples"], "err": r["pol_err"], "correct": within(r["pol_err"], tol)[0],
             "correct55": within(r["pol_err"], (5, 5))[0]}
        m.update(case_metrics(r, r["particles"], tol, args.bound, None))
        mm_pose, m["p_max"] = max_mass(r["particles"], tol, r["modes_pose"])
        m["mm_err"] = relative_many(np.asarray([r["truth"]], np.float64), mm_pose)[0]
        m["mm_correct"] = within(m["mm_err"], tol)[0]
        if "pol_info" in r:
            m.update(local_metrics(r, args.bound))
        dof = max(r["used"] - 6, 1)
        m["fit_p"] = stats.chi2.sf(r["pol_cost"], dof)
        m["rms"] = float(np.sqrt(np.mean(r["pol_resid"] ** 2))) if "pol_resid" in r else np.nan
        near = r["modes_cost"] <= r["pol_cost"] + 6
        far = [np.linalg.norm(relative_many(r["modes_pose"][i:i + 1].astype(np.float64), r["pol_pose"])[0][3:]) > 3
               for i in range(len(r["modes_pose"]))]
        m["rival_modes"] = int(np.sum(near & np.array(far)))
        m["snap"] = {}
        for beta, (xs, _) in r.get("snapshots", {}).items():
            m["snap"][beta] = case_metrics(r, xs, tol, args.bound, None)
        if m["snap"] and "pol_resid" in r:
            m["beta_rule"] = beta_rule(r["pol_resid"], r["sigma"])
            avail = np.array(sorted(list(m["snap"]) + [1.0]))
            pick = avail[np.argmin(np.abs(np.log(avail) - np.log(m["beta_rule"])))]
            m["snap"]["rule"] = m["snap"][pick] if pick < 1 else {k: m[k] for k in m["snap"][avail[0]]}
            m["beta_pick"] = pick
        m["modes_err"] = [relative_many(np.asarray([g], np.float64), r["truth"]) for g in r["modes_pose"]]
        m["modes_cost"] = r["modes_cost"]
        rows.append(m)

    L = ["# Information metrics and uncertainty from particle posteriors\n",
         f"Correct = error norms <= {tol[0]} mm and {tol[1]} deg. Records: {', '.join(args.records)}\n"]
    for key in sorted({(m["method"], m["noise"]) for m in rows}):
        R = [m for m in rows if (m["method"], m["noise"]) == key]
        y = np.array([m["correct"] for m in R])
        ym = np.array([m["mm_correct"] for m in R])
        L.append(f"\n## {key[0]}, noise {key[1]} ({len(R)} cases, {y.mean():.2f} correct; max-mass estimate "
                 f"{ym.mean():.2f} correct)\n")
        # B2: sufficiency measures
        L.append("| measure | AUC correct (2/3) | AUC correct (5/5) |")
        L.append("|---|---|---|")
        y55 = np.array([m["correct55"] for m in R])
        measures = {
            "sample count": [m["used"] if m["used"] < 150 else m["samples"] for m in R],
            "P(correct) from the posterior": [m["p_correct"] for m in R],
            "P(correct 5/5) from the posterior": [m["p_correct55"] for m in R],
            "-trace posterior cov": [-m["trace_post"] for m in R],
            "-rx posterior std": [-m["rx_std"] for m in R],
            "info gain, Gaussian (nats)": [m["gain_gauss"] for m in R],
            "info gain, k-NN entropy (nats)": [m["gain_knn"] for m in R],
            "FIM D at estimate": [m.get("fim_d", np.nan) for m in R],
            "FIM E at estimate": [m.get("fim_e", np.nan) for m in R],
            "FIM A at estimate": [m.get("fim_a", np.nan) for m in R],
            "Rodgers DFS": [m.get("dfs", np.nan) for m in R],
            "fit p-value": [m["fit_p"] for m in R],
            "-rival modes": [-m["rival_modes"] for m in R],
        }
        for name, s in measures.items():
            L.append(f"| {name} | {auc(s, y):.2f} | {auc(s, y55):.2f} |")
        pm = np.array([m["p_max"] for m in R])
        L.append(f"\nMax-mass estimate: P_max predicts its correctness with AUC {auc(pm, ym):.2f}, ECE "
                 f"{ece(pm, ym):.3f}; mean P_max {pm.mean():.2f} vs observed {ym.mean():.2f}.\n")
        L.append("| P_max bin | cases | mean P_max | observed (max-mass estimate correct) |")
        L.append("|---|---|---|---|")
        for a, b in [(0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.01)]:
            mk = (pm >= a) & (pm < b)
            if mk.any():
                L.append(f"| {a:.1f}-{min(b, 1):.1f} | {mk.sum()} | {pm[mk].mean():.2f} | {ym[mk].mean():.2f} |")
        pc = np.array([m["p_correct"] for m in R])
        pc55 = np.array([m["p_correct55"] for m in R])
        L.append(f"\nP(correct) reliability: ECE {ece(pc, y):.3f} (2/3), {ece(pc55, y55):.3f} (5/5); mean predicted "
                 f"{pc.mean():.2f} vs observed {y.mean():.2f}.\n")
        L.append("| P(correct) bin | cases | mean predicted | observed |")
        L.append("|---|---|---|---|")
        for a, b in [(0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.01)]:
            mk = (pc >= a) & (pc < b)
            if mk.any():
                L.append(f"| {a:.1f}-{min(b, 1):.1f} | {mk.sum()} | {pc[mk].mean():.2f} | {y[mk].mean():.2f} |")
        # B1: calibration
        L.append("\n| posterior | HPD KS | 68% cov | 95% cov | PIT KS per axis (x y z rx ry rz) | median NEES |")
        L.append("|---|---|---|---|---|---|")

        def calib(ms, label):
            h = np.array([q["hpd"] for q in ms])
            pit = np.array([q["pit"] for q in ms])
            ks = stats.kstest(h, "uniform").statistic
            kss = " ".join(f"{stats.kstest(pit[:, a], 'uniform').statistic:.2f}" for a in range(6))
            L.append(f"| {label} | {ks:.2f} | {np.mean(h <= 0.68):.2f} | {np.mean(h <= 0.95):.2f} | {kss} | "
                     f"{np.median([q['nees'] for q in ms]):.1f} |")

        calib(R, "beta 1 (sigma)")
        for beta in sorted({b for m in R for b in m["snap"] if b != "rule"}, reverse=True):
            calib([m["snap"][beta] for m in R if beta in m["snap"]], f"beta {beta:.4g} (sigma x {beta ** -0.5:.2f})")
        if any("rule" in m["snap"] for m in R):
            calib([m["snap"]["rule"] for m in R if "rule" in m["snap"]], "beta from the fit (rule)")
            picks = np.array([m["beta_pick"] for m in R if "beta_pick" in m])
            L.append(f"\nRule's beta picks: " + ", ".join(f"{b:.4g}: {np.mean(picks == b):.2f}"
                                                         for b in sorted(set(picks))) + "\n")
            pr = np.array([m["snap"]["rule"]["p_correct"] for m in R if "rule" in m["snap"]])
            yr = np.array([m["correct"] for m in R if "rule" in m["snap"]])
            L.append(f"P(correct) under the rule's posterior: AUC {auc(pr, yr):.2f}, ECE {ece(pr, yr):.3f}, mean "
                     f"predicted {pr.mean():.2f} vs observed {yr.mean():.2f}\n")
            for beta in sorted({b for m in R for b in m["snap"] if b != "rule"}, reverse=True):
                pb = np.array([m["snap"][beta]["p_correct"] for m in R if beta in m["snap"]])
                yb = np.array([m["correct"] for m in R if beta in m["snap"]])
                L.append(f"P(correct) at beta {beta:.4g}: AUC {auc(pb, yb):.2f}, ECE {ece(pb, yb):.3f}, mean "
                         f"predicted {pb.mean():.2f}\n")
        # B3: entropy vs error
        mag = np.array([max(np.linalg.norm(m["err"][:3]) / tol[0], np.linalg.norm(m["err"][3:]) / tol[1]) for m in R])
        L.append("\n| measure | Spearman with -log error magnitude |")
        L.append("|---|---|")
        for name in ("info gain, Gaussian (nats)", "info gain, k-NN entropy (nats)", "FIM D at estimate",
                     "FIM A at estimate", "Rodgers DFS", "P(correct) from the posterior", "sample count"):
            s = np.array(measures[name], float)
            ok = np.isfinite(s)
            L.append(f"| {name} | {stats.spearmanr(s[ok], -np.log(mag[ok])).statistic:.2f} |")
        # B4: local vs particle
        if "lap_std" in R[0]:
            ratio = np.array([m["post_std"] / m["lap_std"] for m in R])
            uni = np.array([m["rival_modes"] == 0 for m in R])
            L.append("\nParticle std / Laplace std at the estimate (median per axis x y z rx ry rz): all "
                     + " ".join(f"{v:.2f}" for v in np.median(ratio, 0)) + f"; single mode ({uni.sum()}) "
                     + " ".join(f"{v:.2f}" for v in np.median(ratio[uni], 0)) + f"; rival modes ({(~uni).sum()}) "
                     + (" ".join(f"{v:.2f}" for v in np.median(ratio[~uni], 0)) if (~uni).any() else "-") + "\n")
            lap_nees = []
            for m in R:
                e = m["err"]
                lap_nees.append(e @ np.linalg.solve(m["lap_cov"], e))
            lap_nees = np.array(lap_nees)
            L.append(f"Laplace NEES median {np.median(lap_nees):.1f} (chi2_6 median 5.35); 95% coverage "
                     f"{np.mean(lap_nees <= stats.chi2.ppf(0.95, 6)):.2f}; single-mode cases "
                     f"{np.mean(lap_nees[uni] <= stats.chi2.ppf(0.95, 6)):.2f}\n")
        # B5: residual vs error across the modes of a case
        near_rho, far_rho = [], []
        for m in R:
            errs = np.array([max(np.linalg.norm(e[0][:3]) / tol[0], np.linalg.norm(e[0][3:]) / tol[1])
                             for e in m["modes_err"]])
            c = np.asarray(m["modes_cost"])
            if len(c) < 4:
                continue
            for sel, acc in ((errs <= 3, near_rho), (errs > 3, far_rho)):
                if sel.sum() >= 4 and np.ptp(c[sel]) > 0:
                    acc.append(stats.spearmanr(c[sel], errs[sel]).statistic)
        L.append(f"Within-case Spearman(cost, error) over polished modes: within 3x tolerance median "
                 f"{np.nanmedian(near_rho) if near_rho else np.nan:.2f} ({len(near_rho)} cases), beyond "
                 f"{np.nanmedian(far_rho) if far_rho else np.nan:.2f} ({len(far_rho)} cases)\n")
    open(args.out, "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    np.save(args.out.replace(".md", "_rows.npy"), np.array(rows, dtype=object), allow_pickle=True)


if __name__ == "__main__":
    main()
