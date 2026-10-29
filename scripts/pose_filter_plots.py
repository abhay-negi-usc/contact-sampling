#!/usr/bin/env python3
"""Figures for the pose-filter study (REPORT.md in --dir): the estimators compared, where the remaining error comes
from, and how well the SMC posterior's uncertainty (P(correct), credible regions) works.

  fig1_estimators.png      correct / search failure / time of every estimator on the same cases
  fig2_error_budget.png    batch SMC's cases split into correct, ambiguous data, data optimum wrong, search failure
  fig3_accuracy_by_step.png  SMC accuracy against the ceilings, by policy step
  fig4_metric.png          sufficiency measures: AUC, and precision against share declared
  fig5_calibration.png     P(correct) reliability on both stream sets, raw and Platt-recalibrated
  fig6_coverage.png        credible-region coverage: SMC and Laplace, real and exact likelihood
The plotted numbers are also written to plot_data.json.

    python scripts/pose_filter_plots.py --dir /media/.../pose_filters
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.optimize import minimize

TOL = (2.0, 3.0)
MARGIN = 6.0  # search failure: cost above the truth mode's by more than this

# Reference palette (dataviz skill), light mode: categorical slots 1-4 in order, chrome and ink.
S1, S2, S3, S4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
NOISE_COLOR = {0.0: S1, 0.3: S2}
NOISE_LABEL = {0.0: "noise 0", 0.3: "noise 0.3 mm / deg"}

METHODS = [  # key, file, label; best first, grouped by family
    ("smc", "policy_smc.npz", "Tempered SMC (batch)"),
    ("multistart", "policy_multistart.npz", "Gauss-Newton, 1024 starts"),
    ("online", "policy_online.npz", "Online SMC (IBIS)"),
    ("scaling", "policy_scaling.npz", "Scaling series"),
    ("mpfball_mix", "seq_mpfball_mix.npz", "MPF-Ball, mixture φ 0.5"),
    ("mpfball", "seq_mpfball.npz", "MPF-Ball"),
    ("mpfball_wide", "seq_mpfball_wide.npz", "MPF-Ball, wide"),
    ("mpfparticle", "seq_mpfparticle.npz", "MPF-Particle"),
    ("mpfuniform_mix", "seq_mpfuniform_mix.npz", "MPF-Uniform, mixture"),
    ("cpf_r05", "seq_cpf_r05.npz", "CPF, roughening 0.5"),
    ("cpf", "seq_cpf.npz", "CPF, roughening 0.1"),
    ("lm", None, "Old 24-start LM"),
]
GROUP_BREAKS = (3, 4, 11)  # hairlines after batch+online, scaling series, the Koval filters


def ok(e, tol=TOL):
    e = np.atleast_2d(e)
    return (np.linalg.norm(e[:, :3], axis=1) <= tol[0]) & (np.linalg.norm(e[:, 3:], axis=1) <= tol[1])


def auc(score, label):
    score, label = np.asarray(score, float), np.asarray(label, bool)
    m = np.isfinite(score)
    score, label = score[m], label[m]
    return stats.mannwhitneyu(score[label], score[~label]).statistic / (label.sum() * (~label).sum())


def wilson(k, n, z=1.96):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


BINS = (0, .1, .3, .5, .7, .9, 1.0001)


def ece(p, y):
    b = np.digitize(p, BINS) - 1
    return float(sum(abs(p[b == i].mean() - y[b == i].mean()) * (b == i).mean()
                     for i in range(len(BINS) - 1) if (b == i).any()))


def reliability(p, y):
    b = np.digitize(p, BINS) - 1
    out = []
    for i in range(len(BINS) - 1):
        m = b == i
        if m.sum():
            lo, hi = wilson(y[m].sum(), m.sum())
            out.append([float(p[m].mean()), float(y[m].mean()), int(m.sum()), float(lo), float(hi)])
    return out


def logit(p):
    p = np.clip(p, 1e-3, 1 - 1e-3)
    return np.log(p / (1 - p))


def platt(p, y):
    """logit p' = a logit p + b, fitted by maximum likelihood (as scratchpad recal.py / REPORT section 5)."""
    nll = lambda w: np.sum(np.logaddexp(0, -(w[0] * logit(p) + w[1])) * y
                           + np.logaddexp(0, w[0] * logit(p) + w[1]) * (1 - y))
    a, b = minimize(nll, [1.0, 0.0]).x
    return float(a), float(b)


def selective(score, y):
    """Precision against share declared as the threshold on score sweeps its unique values (ties kept together)."""
    score, y = np.asarray(score, float), np.asarray(y, float)
    m = np.isfinite(score)
    score, y = score[m], y[m]
    share, prec = [], []
    for t in np.unique(score)[::-1]:
        d = score >= t
        share.append(d.mean())
        prec.append(y[d].mean())
    return share, prec


def collect(d):
    recs = {}
    for key, f, _ in METHODS:
        if f:
            recs[key] = {(r["stream"], r["step"], float(r["noise"])): r
                         for r in np.load(f"{d}/{f}", allow_pickle=True)["records"]}
    lm = {(r["stream"], r["step"], float(r["noise"])): r["mode_err"][0]
          for r in np.load(f"{d}/../info_metric/policy.npz", allow_pickle=True)["records"]
          if "mode_err" in r and len(r["mode_err"])}
    common = set.intersection(*[set(v) for v in recs.values()]) & set(lm)
    out = {"common_cases": len(common) // 2, "methods": {}}
    for key, _, label in METHODS:
        row = {"label": label}
        for noise in (0.0, 0.3):
            ks = sorted(k for k in common if k[2] == noise)
            if key == "lm":
                row[str(noise)] = {"correct": float(ok(np.array([lm[k] for k in ks])).mean())}
                continue
            rs = [recs[key][k] for k in ks]
            row[str(noise)] = {
                "correct": float(ok(np.array([r["pol_err"] for r in rs])).mean()),
                "search_fail": float(np.mean([r["pol_cost"] - r["tm_cost"] > MARGIN for r in rs])),
                "truth_mode": float(ok(np.array([r["tm_err"] for r in rs])).mean()),
                "time_s": float(np.median([r["time_s"] for r in recs[key].values() if r["noise"] == noise]))}
        out["methods"][key] = row

    def budget(rs):
        pol = ok(np.array([r["pol_err"] for r in rs]))
        tm = ok(np.array([r["tm_err"] for r in rs]))
        sf = np.array([r["pol_cost"] - r["tm_cost"] > MARGIN for r in rs]) & ~pol
        return {"n": len(rs), "correct": float(pol.mean()), "ambiguous": float((~pol & ~sf & tm).mean()),
                "optimum_wrong": float((~pol & ~sf & ~tm).mean()), "search_fail": float(sf.mean())}

    transfer = list(np.load(f"{d}/noise_smc.npz", allow_pickle=True)["records"])
    out["budget"] = {f"{s}|{noise}": budget([r for r in rs if r["noise"] == noise])
                     for s, rs in (("policy", list(recs["smc"].values())), ("transfer", transfer))
                     for noise in (0.0, 0.3)}

    synth = list(np.load(f"{d}/synthetic_smc.npz", allow_pickle=True)["records"])
    by_step = {}
    for noise in (0.0, 0.3):
        rows = {"steps": [], "smc": [], "online": [], "truth_mode": [], "n": [], "exact_steps": [], "exact_tm": [],
                "exact_n": []}
        for s in sorted({k[1] for k in recs["smc"]}):
            rs = [r for k, r in recs["smc"].items() if k[1] == s and k[2] == noise]
            rows["steps"].append(s)
            rows["n"].append(len(rs))
            rows["smc"].append(float(ok(np.array([r["pol_err"] for r in rs])).mean()))
            rows["truth_mode"].append(float(ok(np.array([r["tm_err"] for r in rs])).mean()))
            sy = [r for r in synth if r["step"] == s and r["noise"] == noise]
            if sy:
                rows["exact_steps"].append(s)
                rows["exact_n"].append(len(sy))
                rows["exact_tm"].append(float(ok(np.array([r["tm_err"] for r in sy])).mean()))
        by_step[str(noise)] = rows
    out["by_step"] = by_step

    rows = np.load(f"{d}/uq_smc_rows.npy", allow_pickle=True)
    measures = {  # name: score (larger = more likely correct), as in uq_analysis.py
        "P(correct)": lambda m: m["p_correct"],
        "Information gain, Gaussian": lambda m: m["gain_gauss"],
        "Information gain, k-NN": lambda m: m["gain_knn"],
        "−trace posterior covariance": lambda m: -m["trace_post"],
        "Fisher D at the estimate": lambda m: m["fim_d"],
        "Rodgers DFS": lambda m: m["dfs"],
        "−rival modes": lambda m: -m["rival_modes"],
        "χ² fit p-value": lambda m: m["fit_p"],
        "Sample count": lambda m: m["used"] if m["used"] < 150 else m["samples"],
    }
    metric = {"auc": {}, "selective": {}}
    for name, f in measures.items():
        metric["auc"][name] = {}
        for noise in (0.0, 0.3):
            R = [m for m in rows if m["noise"] == noise]
            metric["auc"][name][str(noise)] = float(auc([f(m) for m in R], [m["correct"] for m in R]))
    for noise in (0.0, 0.3):
        R = [m for m in rows if m["noise"] == noise]
        y = np.array([m["correct"] for m in R], float)
        sel = {"base_rate": float(y.mean())}
        for name in ("P(correct)", "Fisher D at the estimate", "Sample count"):
            sh, pr = selective([measures[name](m) for m in R], y)
            sel[name] = {"share": sh, "precision": pr}
        p = np.array([m["p_correct"] for m in R])
        sel["rule_0.9"] = {"share": float((p >= 0.9).mean()), "precision": float(y[p >= 0.9].mean())}
        metric["selective"][str(noise)] = sel
    out["metric"] = metric

    transfer_rows = np.load(f"{d}/uq_transfer_rows.npy", allow_pickle=True)
    calib = {}
    for noise in (0.0, 0.3):
        A = [m for m in rows if m["noise"] == noise]
        B = [m for m in transfer_rows if m["noise"] == noise]
        pa, ya = np.array([m["p_correct"] for m in A]), np.array([m["correct"] for m in A], float)
        pb, yb = np.array([m["p_correct"] for m in B]), np.array([m["correct"] for m in B], float)
        a, b = platt(pa, ya)
        pc = 1 / (1 + np.exp(-(a * logit(pb) + b)))
        calib[str(noise)] = {"platt": [a, b], "series": {
            "Policy set": {"ece": ece(pa, ya), "bins": reliability(pa, ya), "n": len(A)},
            "Transfer set": {"ece": ece(pb, yb), "bins": reliability(pb, yb), "n": len(B)},
            "Transfer set, Platt map from the policy set": {"ece": ece(pc, yb), "bins": reliability(pc, yb),
                                                            "n": len(B)}}}
    out["calibration"] = calib

    levels = np.linspace(0, 1, 101)
    real = np.load(f"{d}/uq_calib_base_rows.npy", allow_pickle=True)
    exact = np.load(f"{d}/uq_synthetic_rows.npy", allow_pickle=True)
    cover = {"levels": levels.tolist()}
    for noise in (0.0, 0.3):
        series = {}
        for name, R, kind in (("SMC posterior, simulator contacts", real, "smc"),
                              ("SMC posterior, exact likelihood", exact, "smc"),
                              ("Laplace (local Fisher), exact likelihood", exact, "laplace")):
            R = [m for m in R if m["noise"] == noise]
            if kind == "smc":
                lev = np.array([m["hpd"] for m in R])
                nees = np.median([m["nees"] for m in R])
            else:
                q = np.array([m["err"] @ np.linalg.solve(m["lap_cov"], m["err"]) for m in R])
                lev, nees = stats.chi2.cdf(q, 6), np.median(q)
            series[name] = {"coverage": [float(np.mean(lev <= a)) for a in levels], "n": len(R),
                            "nees_median": float(nees), "cov95": float(np.mean(lev <= 0.95))}
        cover[str(noise)] = series
    out["coverage"] = cover
    return out


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "sans-serif", "font.size": 10, "text.color": INK,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "axes.titlesize": 11, "axes.titleweight": "semibold", "axes.titlelocation": "left",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-", "axes.axisbelow": True,
        "xtick.color": AXIS, "ytick.color": AXIS, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "legend.frameon": False, "legend.fontsize": 9, "lines.linewidth": 2, "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round", "lines.markersize": 7, "lines.markeredgecolor": SURFACE,
        "lines.markeredgewidth": 1.5})


def headline(fig, title, sub):
    fig.text(0.01, 0.985, title, fontsize=13, fontweight="semibold", color=INK, va="top")
    fig.text(0.01, 0.935, sub, fontsize=9.5, color=INK2, va="top")


def fig_estimators(D, path):
    M = D["methods"]
    keys = [k for k, _, _ in METHODS]
    y = np.arange(len(keys))[::-1].astype(float)
    fig, axes = plt.subplots(1, 3, figsize=(13, 5.6), sharey=True, gridspec_kw={"width_ratios": [1.15, 1.15, 0.9]})
    off = {0.0: 0.14, 0.3: -0.14}
    for ax, field, title in ((axes[0], "correct", "Correct (2 mm and 3°)"),
                             (axes[1], "search_fail", "Search failure"),
                             (axes[2], "time_s", "Median time per case, s")):
        for noise in (0.0, 0.3):
            v = np.array([M[k][str(noise)].get(field, np.nan) for k in keys])
            ax.plot(v, y + off[noise], "o", color=NOISE_COLOR[noise], label=NOISE_LABEL[noise], zorder=3)
        ax.set_title(title, fontsize=10.5)
        for b in GROUP_BREAKS:
            ax.axhline(len(keys) - b - 0.5, color=GRID, lw=0.8, zorder=1)
        ax.grid(axis="y", visible=False)
    for noise in (0.0, 0.3):
        tm = M["smc"][str(noise)]["truth_mode"]
        axes[0].axvline(tm, color=NOISE_COLOR[noise], lw=1, alpha=0.6, zorder=2)
        axes[0].text(tm, len(keys) - 0.35, f"best fit at\ntruth {tm:.2f}", ha="center", va="bottom", fontsize=8,
                     color=INK2)
    for noise in (0.0, 0.3):  # the winner's values, labelled
        v = M["smc"][str(noise)]["correct"]
        axes[0].text(v + 0.025, y[0] + off[noise], f"{v:.2f}", va="center", fontsize=8.5, color=INK)
    axes[0].set_xlim(0, 1)
    axes[1].set_xlim(-0.02, 1)
    axes[2].set_xscale("log")
    axes[2].set_xlim(0.8, 150)
    axes[2].text(M["online"]["0.0"]["time_s"] * 1.15, y[2] - 0.45, "per checkpoint", fontsize=8, color=INK2)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([M[k]["label"] for k in keys])
    axes[0].set_ylim(-0.7, len(keys) - 0.1 + 0.9)
    for t in axes[0].get_yticklabels()[:3]:
        t.set_color(INK)
        t.set_fontweight("semibold")
    axes[0].legend(loc="lower right", handletextpad=0.2)
    headline(fig, "Global search (batch SMC, 1024-start Gauss-Newton, online SMC) finds the best fit; Koval's "
                  "sequential filters lose it",
             f"{D['common_cases']} cases per noise level (24 policy streams, steps 20–300), the same for every method. "
             "Search failure: the found mode's cost exceeds the truth's own mode by more than 6.\n"
             "Times are wall-clock on a shared GPU and indicative only.")
    fig.subplots_adjust(left=0.2, right=0.985, top=0.8, bottom=0.08, wspace=0.08)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_budget(D, path):
    B = D["budget"]
    order = ["policy|0.0", "policy|0.3", "transfer|0.0", "transfer|0.3"]
    labels = [f"{'Policy' if k.startswith('policy') else 'Transfer'} set, "
              f"{NOISE_LABEL[float(k.split('|')[1])]} ({B[k]['n']})" for k in order]
    parts = [("correct", "Correct", S1, "white"),
             ("ambiguous", "Wrong; another mode fits as well (data ambiguous)", S2, "white"),
             ("optimum_wrong", "Wrong; even the best fit at the truth is off (noise, model error)", S3, "white"),
             ("search_fail", "Search failure", S4, INK)]
    fig, ax = plt.subplots(figsize=(11, 3.9))
    y = np.arange(len(order))[::-1]
    left = np.zeros(len(order))
    for key, label, color, ink in parts:
        v = np.array([B[k][key] for k in order])
        ax.barh(y, v, left=left, height=0.5, color=color, edgecolor=SURFACE, linewidth=2, label=label, zorder=3)
        for yi, l, w in zip(y, left, v):
            if w >= 0.06:
                ax.text(l + w / 2, yi, f"{w:.0%}", ha="center", va="center", fontsize=9, color=ink, zorder=4)
        left += v
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper left", bbox_to_anchor=(0, 1.2), ncol=2, handlelength=1.2, columnspacing=1.5)
    headline(fig, "Batch SMC's errors are the data's, not the search's",
             "Every SMC case, split by why it is wrong. Search failures are under 1%; at noise 0.3 most errors are "
             "cases where the contacts' best fit is itself off.")
    fig.subplots_adjust(left=0.27, right=0.98, top=0.7, bottom=0.1)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_by_step(D, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, noise in zip(axes, (0.0, 0.3)):
        S = D["by_step"][str(noise)]
        x = np.arange(len(S["steps"]))
        xe = [S["steps"].index(s) for s in S["exact_steps"]]
        for vals, xs, color, label in ((S["exact_tm"], xe, S3, "Best fit at truth, exact likelihood (32 streams)"),
                                       (S["truth_mode"], x, S2, "Best fit at truth (ceiling for a perfect search)"),
                                       (S["smc"], x, S1, "Batch SMC estimate")):
            ax.plot(xs, vals, "-o", color=color, label=label, zorder=3)
            ax.text(xs[-1] + 0.15, vals[-1], f"{vals[-1]:.2f}", va="center", fontsize=8.5, color=INK2)
        ax.set_xticks(x)
        ax.set_xticklabels(S["steps"])
        ax.set_xlim(-0.3, len(x) - 0.4)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("Policy step (contact samples collected so far)")
        ax.set_title(NOISE_LABEL[noise].capitalize(), fontsize=10.5)
    axes[0].set_ylabel("Share of cases correct")
    axes[0].legend(loc="lower left", bbox_to_anchor=(0, 1.1), ncol=3, handlelength=1.6, columnspacing=1.4)
    headline(fig, "Below the ceiling the data are ambiguous; the ceiling itself is set by noise and model error",
             "Policy set, about 115 cases per step and noise level. Blue to orange: rival modes that fit as well. "
             "Orange to green: likelihood misspecification.\nThe fall after step 80 is the even thinning to 150 "
             "samples: 600 samples recover 12–20 points at step 300.")
    fig.subplots_adjust(left=0.07, right=0.97, top=0.68, bottom=0.13, wspace=0.08)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_metric(D, path):
    A = D["metric"]["auc"]
    names = list(A)
    fig = plt.figure(figsize=(13, 5.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1, 1], wspace=0.28)
    ax = fig.add_subplot(gs[0])
    y = np.arange(len(names))[::-1].astype(float)
    off = {0.0: 0.13, 0.3: -0.13}
    for noise in (0.0, 0.3):
        v = [A[n][str(noise)] for n in names]
        ax.plot(v, y + off[noise], "o", color=NOISE_COLOR[noise], label=NOISE_LABEL[noise], zorder=3)
        ax.text(v[0] + 0.012, y[0] + off[noise], f"{v[0]:.2f}", va="center", fontsize=8.5, color=INK)
    ax.axvline(0.5, color=AXIS, lw=1)
    ax.text(0.505, -0.75, "chance", fontsize=8, color=INK2)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.get_yticklabels()[0].set_fontweight("semibold")
    ax.get_yticklabels()[0].set_color(INK)
    ax.set_xlim(0.3, 1.0)
    ax.set_ylim(-1, len(names) - 0.4)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("AUC for predicting that the estimate is correct")
    ax.set_title("Which measure says the data suffice?", fontsize=10.5)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.04), ncol=2, handletextpad=0.2)
    colors = {"P(correct)": S1, "Fisher D at the estimate": S2, "Sample count": S3}
    for i, noise in enumerate((0.0, 0.3)):
        ax = fig.add_subplot(gs[i + 1])
        S = D["metric"]["selective"][str(noise)]
        for name, color in colors.items():
            ax.plot(S[name]["share"], S[name]["precision"], color=color, label=name, zorder=3)
        r = S["rule_0.9"]
        ax.plot(r["share"], r["precision"], "o", color=S1, zorder=4, markersize=8)
        ax.annotate(f"P(correct) ≥ 0.9:\n{r['precision']:.2f} correct,\n{r['share']:.0%} declared",
                    (r["share"], r["precision"]), xytext=(0.1, 0.74) if noise == 0 else (0.14, 0.86),
                    fontsize=8.5, color=INK, va="top", arrowprops={"arrowstyle": "-", "color": INK2, "lw": 0.8})
        ax.axhline(S["base_rate"], color=AXIS, lw=1)
        ax.text(0.99, S["base_rate"] - 0.015, f"all cases {S['base_rate']:.2f}", ha="right", va="top", fontsize=8,
                color=INK2)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.03)
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.set_xlabel("Share of cases declared sufficient")
        ax.set_title(f"Stop-rule precision, noise {noise:g}", fontsize=10.5)
        if i == 0:
            ax.set_ylabel("Share of declared cases that are correct")
            ax.legend(loc="lower left", bbox_to_anchor=(0, 1.08), ncol=3, handlelength=1.4, columnspacing=1.2)
    headline(fig, "P(correct), the posterior mass within tolerance of the estimate, is the sufficiency metric",
             "Batch SMC on the policy set, 693 cases per noise level. Each curve sweeps a threshold on its measure; "
             "the sample count barely beats picking cases at random.")
    fig.subplots_adjust(left=0.18, right=0.985, top=0.76, bottom=0.11)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_calibration(D, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.4), sharey=True)
    colors = [S1, S2, S3]
    for ax, noise in zip(axes, (0.0, 0.3)):
        C = D["calibration"][str(noise)]
        ax.plot([0, 1], [0, 1], color=AXIS, lw=1, zorder=1)
        for (name, s), color in zip(C["series"].items(), colors):
            b = np.array(s["bins"])
            ax.vlines(b[:, 0], b[:, 3], b[:, 4], color=color, lw=1, alpha=0.45, zorder=2)
            ax.plot(b[:, 0], b[:, 1], "-o", color=color, zorder=3,
                    label=name.replace(", Platt", ",\nPlatt") + f": ECE {s['ece']:.3f}")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_xlabel("Predicted P(correct), bin mean")
        ax.set_title(NOISE_LABEL[noise].capitalize(), fontsize=10.5)
        ax.legend(loc="upper left", fontsize=8.5, handlelength=1.5, labelspacing=0.7)
        a, b = C["platt"]
        ax.text(0.98, 0.03, f"Platt map: logit p′ = {a:.2f} logit p {'−' if b < 0 else '+'} {abs(b):.2f}",
                ha="right", fontsize=8.5, color=INK2)
    axes[0].set_ylabel("Observed share correct (95% interval)")
    headline(fig, "P(correct) is calibrated at noise 0.3; at noise 0 one Platt map, fitted once, fixes it",
             "Bins 0–.1–.3–.5–.7–.9–1; the gray diagonal is perfect calibration. The map fitted on the policy set is "
             "applied unchanged to the transfer set\n(streams_T20z_noise, 470 cases per noise level).")
    fig.subplots_adjust(left=0.07, right=0.98, top=0.83, bottom=0.1, wspace=0.08)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_coverage(D, path):
    C = D["coverage"]
    lv = np.array(C["levels"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.4), sharey=True)
    colors = [S1, S2, S3]
    for ax, noise in zip(axes, (0.0, 0.3)):
        ax.plot([0, 1], [0, 1], color=AXIS, lw=1, zorder=1)
        ax.text(0.86, 0.875, "calibrated", rotation=45, fontsize=8, color=INK2, ha="center", va="bottom",
                rotation_mode="anchor", transform_rotates_text=True)
        for (name, s), color in zip(C[str(noise)].items(), colors):
            ax.plot(lv, s["coverage"], color=color, zorder=3,
                    label=name.replace(", ", ",\n") + f"\n95% region → {s['cov95']:.2f}, median NEES "
                                                         f"{s['nees_median']:.1f}")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_xlabel("Nominal credible level")
        ax.set_title(NOISE_LABEL[noise].capitalize(), fontsize=10.5)
        ax.legend(loc="upper left", fontsize=8.5, handlelength=1.5, labelspacing=0.8)
    axes[0].set_ylabel("Share of cases whose truth is inside the region")
    headline(fig, "The SMC sampler is calibrated; the real likelihood is not, and Laplace fails even when it is",
             "Calibration subset: 32 policy streams × 5 steps, 148 cases per noise level. Exact likelihood: residuals "
             "at the truth replaced by N(0, σ²).\nA calibrated posterior gives median NEES 5.35 (χ²₆).")
    fig.subplots_adjust(left=0.07, right=0.98, top=0.83, bottom=0.1, wspace=0.08)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", required=True, help="pose_filters results directory")
    p.add_argument("--out", help="figure directory (default: DIR/figures)")
    p.add_argument("--reuse", action="store_true", help="plot from an existing plot_data.json")
    args = p.parse_args()
    out = args.out or f"{args.dir}/figures"
    os.makedirs(out, exist_ok=True)
    cache = f"{out}/plot_data.json"
    if args.reuse and os.path.exists(cache):
        D = json.load(open(cache))
    else:
        D = collect(args.dir)
        json.dump(D, open(cache, "w"), indent=1)
    style()
    fig_estimators(D, f"{out}/fig1_estimators.png")
    fig_budget(D, f"{out}/fig2_error_budget.png")
    fig_by_step(D, f"{out}/fig3_accuracy_by_step.png")
    fig_metric(D, f"{out}/fig4_metric.png")
    fig_calibration(D, f"{out}/fig5_calibration.png")
    fig_coverage(D, f"{out}/fig6_coverage.png")
    print(f"figures in {out}")


if __name__ == "__main__":
    main()
