#!/usr/bin/env python3
"""Table of the hook RL uncertainty benchmark (RL1-RL8), from eval_hook_rl.py outputs in --bench/<name>/
(or the directory --use NAME=DIR names).

Every experiment runs the same evaluation set (the same starts, and the same belief errors when tested with
uncertainty), so experiments are also compared per case: for a pair A, B, the cases only A assembles and only B
assembles, with McNemar's exact two-sided p-value. Writes --bench/benchmark.md and prints it.

    python scripts/compile_rl_bench.py --bench /media/rp/Elements1/abhay_ws/contact-sampling/hook_rl/bench_nominal \
        --use RL5=RL5_dev --use RL6=RL6_dev
"""

import argparse
import csv
import json
from math import comb
from pathlib import Path

ROWS = [("RL1", "none", "none", "-"), ("RL2", "none", "goal pose", "-"), ("RL3", "goal pose", "none", "-"),
        ("RL4", "goal pose", "goal pose", "-"), ("RL5", "none", "goal pose", "contact manifold"),
        ("RL6", "goal pose", "goal pose", "contact manifold"),
        ("RL7", "goal pose, estimator in loop", "goal pose", "contact manifold"),
        ("RL8", "none; explorer: goal pose", "goal pose", "contact manifold, learned exploration")]
PAIRS = [("RL2", "RL1"), ("RL4", "RL2"), ("RL5", "RL2"), ("RL6", "RL4"), ("RL6", "RL5"), ("RL7", "RL6"),
         ("RL7", "RL4"), ("RL8", "RL5"), ("RL8", "RL2"), ("RL8", "RL4")]


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def load(bench, name):
    d = bench / name  # name: a directory under bench
    if not (d / "summary.json").exists():
        return None, None
    with open(d / "episodes.csv") as f:
        ok = {int(r["episode"]): r["pull_ok"] == "True" for r in csv.DictReader(f)}
    return json.loads((d / "summary.json").read_text()), ok


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bench", type=Path, required=True)
    parser.add_argument("--use", action="append", default=[], metavar="NAME=DIR",
                        help="read row NAME from --bench/DIR instead of --bench/NAME")
    args = parser.parse_args()
    dirs = dict(u.split("=", 1) for u in args.use)
    lines = ["| | Trained with uncertainty | Tested with uncertainty | Estimator | Assembled, pull test (95% CI) | "
             "Failures | Time to assemble p10/50/90 (s) | Belief error MAE x, y mm / rx, ry, rz deg: start -> end | "
             "Estimates per episode (s each) |", "|" + "---|" * 9]
    outcomes = {}
    for name, train, test, est in ROWS:
        s, ok = load(args.bench, dirs.get(name, name))
        if s is None:
            lines.append(f"| {name} | {train} | {test} | {est} | (not run) | | | | |")
            continue
        outcomes[name] = ok
        fails = ", ".join(f"{v} {k}" for k, v in s["outcomes"].items() if k != "assembled") or "none"
        lo, hi = s["assembled_and_pull_ok_ci95"]
        b0, b1 = s.get("belief_err_mae_initial"), s.get("belief_err_mae_final")
        belief = (f"{b0['x']:.2f}, {b0['y']:.2f} / {b0['rx']:.2f}, {b0['ry']:.2f}, {b0['rz']:.2f} -> "
                  f"{b1['x']:.2f}, {b1['y']:.2f} / {b1['rx']:.2f}, {b1['ry']:.2f}, {b1['rz']:.2f}") if b0 else ""
        n_est = s.get("estimates_per_episode_mean", 0.0)
        t_est = s.get("estimate_time_s_per_call_median")
        lines.append(f"| {name} | {train} | {test} | {est} | {100 * s['assembled_and_pull_ok']:.1f}% "
                     f"({100 * lo:.1f}-{100 * hi:.1f}) | {fails} | "
                     f"{'/'.join(f'{t:.2f}' for t in s['time_to_success_s'] or [])} | {belief} | "
                     f"{n_est:.1f}" + (f" ({t_est:.2f})" if n_est else "") + " |")
    lines += ["", "Paired on the same cases:", "", "| A vs B | Only A assembles | Only B assembles | McNemar p |",
              "|---|---|---|---|"]
    for a, b in PAIRS:
        if a in outcomes and b in outcomes:
            only_a = sum(outcomes[a][k] and not outcomes[b][k] for k in outcomes[a])
            only_b = sum(outcomes[b][k] and not outcomes[a][k] for k in outcomes[a])
            lines.append(f"| {a} vs {b} | {only_a} | {only_b} | {mcnemar(only_a, only_b):.2g} |")
    if dirs:
        lines += ["", "Rows read from: " + ", ".join(f"{k} = {v}" for k, v in dirs.items())]
    text = "\n".join(lines) + "\n"
    (args.bench / "benchmark.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
