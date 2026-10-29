#!/usr/bin/env python3
"""Hook RL benchmark (RL1-RL8) across wrist wrench noise levels: success per version and level, each level paired
against the noise-free run on the same cases (cases only noise-free assembles, only noisy assembles, McNemar p).
Noise levels are the --sweep/n<force std> directories (eval_hook_rl.py --wrench-noise outputs, one per version);
the noise-free runs are --reference/<version>. Rows --extra NAME (e.g. a policy retrained with noise) are read from
the same levels, their noise-free runs from --extra-reference/NAME (default --sweep/noise_free). Other sensor
conditions: --condition DIR=LABEL columns, in order, instead of the noise levels; --rows limits the RL1-RL8 rows.
Writes --sweep/benchmark.md and prints it.

    python scripts/compile_wnoise_bench.py --sweep $B/T20z_wnoise --reference $B/T20z_stop
"""

import argparse
from pathlib import Path

from compile_rl_bench import ROWS, load, mcnemar


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sweep", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True, help="the noise-free runs")
    parser.add_argument("--extra", action="append", default=[], metavar="NAME")
    parser.add_argument("--extra-reference", type=Path, help="the --extra rows' noise-free runs")
    parser.add_argument("--condition", action="append", default=[], metavar="DIR=LABEL")
    parser.add_argument("--rows", type=lambda t: t.split(","), help="of RL1-RL8 (default: all)")
    args = parser.parse_args()
    if args.condition:
        levels = [(args.sweep / c.split("=", 1)[0], c.split("=", 1)[1]) for c in args.condition]
    else:
        levels = [(d, f"σ {d.name[1:]} N") for d in sorted((d for d in args.sweep.glob("n[0-9]*") if d.is_dir()),
                                                           key=lambda d: float(d.name[1:]))]
    head = "| | Noise-free | " + " | ".join(label for _, label in levels) + " |"
    lines = ["Assembled (pull test), false declarations of done in brackets; per condition: cases only noise-free / "
             "only this condition assembles, McNemar p.", "", head, "|" + "---|" * (len(levels) + 2)]
    extra_ref = args.extra_reference or args.sweep / "noise_free"
    rows = ([(r[0], args.reference) for r in ROWS if args.rows is None or r[0] in args.rows]
            + [(e, extra_ref) for e in args.extra])
    for name, reference in rows:
        s0, ok0 = load(reference, name)
        cells = [f"{100 * s0['assembled_and_pull_ok']:.1f}% ({100 * s0['false_done']:.1f})" if s0 else "-"]
        for d, _ in levels:
            s, ok = load(d, name)
            if s is None:
                cells.append("(not run)")
                continue
            cell = f"{100 * s['assembled_and_pull_ok']:.1f}% ({100 * s['false_done']:.1f})"
            if ok0 is not None:
                only0 = sum(ok0[k] and not ok[k] for k in ok0)
                only1 = sum(ok[k] and not ok0[k] for k in ok0)
                cell += f"; {only0}/{only1}, p {mcnemar(only0, only1):.2g}"
            cells.append(cell)
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    text = "\n".join(lines) + "\n"
    (args.sweep / "benchmark.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
