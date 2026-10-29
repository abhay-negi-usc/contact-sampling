#!/usr/bin/env python3
"""Check a contact sampler run against the analytical contact modes of the peg and hole.

Ground truth (peg_hole_analytic.py, no MuJoCo): contact states in the reduced coordinates (rho, radial
tilt, |tangential tilt|, depth), generated mode by mode:
  * lowering the peg along the bore axis from above at random offset and tilt (first contact),
  * moving the tip radially at random tilt and depth inside the bore (wall and rim contacts),
  * from those, closing the nearest open gap while keeping the others closed (two-point states),
    and once more (three-point states).
Ground truth is kept to rho <= 10 mm and tilt <= 10 deg, which lies inside the sampler's box at any
azimuth. Each ground-truth state is covered when a sample lies within --radius (mm and deg) of it in
the reduced coordinates; coverage is reported per contact mode.

    conda run -n mujoco_env python scripts/compare_samples_to_analytic.py RUN_DIR
"""

import argparse
import glob
import json
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

import peg_hole_analytic as A

GT_TOL = 1e-4  # mm: a ground-truth contact touches to within this
# mm: MuJoCo's bore is a 64-gon, inside the true circle by up to 16 um, and the outer rim's corners
# stand out of it by about 2 um.
SAMPLE_TOL, SAMPLE_TOL_BORE = 5e-3, 2e-2


def sample_reduced(run_dir):
    """Reduced coords of every sample of a run (cached next to the shards)."""
    cache = Path(run_dir) / "reduced_coords.npy"
    if cache.exists():
        return np.load(cache)
    parts = [A.reduced_from_pose(np.load(f)["pose"].astype(float)).astype(np.float32)
             for f in sorted(glob.glob(str(Path(run_dir) / "shard_*.npz")))]
    out = np.concatenate(parts)
    np.save(cache, out)
    return out


def in_box(q):
    tilt = np.degrees(np.arctan(np.hypot(np.tan(np.radians(q[:, 1])), np.tan(np.radians(q[:, 2])))))
    return (q[:, 0] <= 10) & (tilt <= 10) & (q[:, 3] >= -3) & (q[:, 3] <= 50.5)


def ground_truth(n_lower, n_lateral, seed):
    """Contact states of every mode. Half of each batch is drawn near the bore (small offset and
    tilt), where the modes are narrow, so that they are not left to a handful of states."""
    rng = np.random.default_rng(seed)

    def tilts(n, radius):
        r, a = radius * np.sqrt(rng.random(n)), np.pi * rng.random(n)  # uniform over the half disk, tilt_t >= 0
        return r * np.cos(a), r * np.sin(a)

    half = lambda n: np.r_[np.full(n - n // 2, 1.0), np.full(n // 2, 0.0)]
    # first contact lowering from above
    wide = half(n_lower)
    tr, tt = tilts(n_lower, np.where(wide == 1, 10.0, 3.0))
    q = np.stack([np.where(wide == 1, 10.0, 1.5) * rng.random(n_lower), tr, tt, np.zeros(n_lower)], axis=1)
    q[:, 3] = A.first_contact_depth(q, -6.0, 50.6, 1e-6)
    lowered = q[~np.isnan(q[:, 3])]
    # radial contacts inside the bore
    tr, tt = tilts(n_lateral, np.where(half(n_lateral) == 1, 10.0, 2.0))
    q = np.stack([np.zeros(n_lateral), tr, tt, 0.05 + 49.95 * rng.random(n_lateral)], axis=1)
    rho = np.full((n_lateral, 4), np.nan)
    A.lateral_contacts(q, 3.0, 200, 1e-7, rho)
    lateral = np.concatenate([np.column_stack([rho[:, j], q[:, 1:]]) for j in range(4)])
    lateral = lateral[~np.isnan(lateral[:, 0])]
    states = [lowered, lateral]
    # one more contact, twice
    for _ in range(2):
        q = np.concatenate(states)
        out = q.copy()
        ok = np.zeros(len(q), bool)
        A.add_contact(q, 1.0, 1e-7, out, ok)
        states.append(out[ok])
    gt = np.concatenate(states)
    gt[:, 2] = np.abs(gt[:, 2])
    return gt[in_box(gt)]


def classify(q, tol, tol_bore):
    c, u = A.pose_from_reduced(q)
    sep, count, pairs = A.contact_modes(c, u, tol, tol_bore)
    return sep, count, np.array([A.mode_name(p) for p in pairs])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--n-lower", type=int, default=400_000)
    parser.add_argument("--n-lateral", type=int, default=200_000)
    parser.add_argument("--radius", type=float, default=0.5, help="coverage radius in reduced coords [mm, deg]")
    parser.add_argument("--density-radius", type=float, default=0.25,
                        help="radius for counting samples around each ground-truth state [mm, deg]")
    parser.add_argument("--gt", type=Path, default=Path("/media/rp/Elements1/abhay_ws/contact-sampling/"
                                                         "peg_hole_analytic_ground_truth.npz"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    t0 = time.time()
    if args.gt.exists():
        z = np.load(args.gt)
        gt, gt_mode, gt_sep = z["q"], z["mode"], z["sep"]
    else:
        gt = ground_truth(args.n_lower, args.n_lateral, args.seed)
        gt_sep, _, gt_mode = classify(gt, GT_TOL, GT_TOL)
        np.savez(args.gt, q=gt, mode=gt_mode, sep=gt_sep)
    print(f"ground truth: {len(gt)} contact states ({time.time() - t0:.0f} s), separation within "
          f"[{gt_sep.min() * 1e3:+.3f}, {gt_sep.max() * 1e3:+.3f}] um")

    t0 = time.time()
    samples = sample_reduced(args.run_dir)
    tree = cKDTree(samples)
    dist, _ = tree.query(gt, k=1, workers=-1)
    density = tree.query_ball_point(gt, args.density_radius, return_length=True, workers=-1)
    print(f"{len(samples)} samples, nearest-sample search {time.time() - t0:.0f} s")

    radii = sorted(set([0.1, 0.25, args.radius]))
    rows = []
    for mode in np.unique(gt_mode):
        sel = gt_mode == mode
        d = dist[sel]
        rows.append(dict(mode=str(mode), contacts=0 if mode == "none" else mode.count("+") + 1, n=int(sel.sum()),
                         covered={str(r): float(np.mean(d <= r)) for r in radii}, median_dist=float(np.median(d)),
                         p95_dist=float(np.percentile(d, 95)), median_density=float(np.median(density[sel]))))
    rows.sort(key=lambda r: (r["contacts"], -r["n"]))
    head = " ".join(f"{'<=' + str(r):>7}" for r in radii)
    print(f"\n{'contacts':>8} {'GT':>7} {head} {'median':>7} {'p95':>6} {'dens':>6}  mode")
    print(f"{'':8} {'states':>7} {'(fraction of GT states with a sample within r mm/deg)':^{8 * len(radii)}}"
          f"{'dist':>6} {'dist':>6} {'n<' + str(args.density_radius):>6}")
    for r in rows:
        cov = " ".join(f"{r['covered'][str(x)]:7.1%}" for x in radii)
        print(f"{r['contacts']:8d} {r['n']:7d} {cov} {r['median_dist']:7.3f} {r['p95_dist']:6.3f} "
              f"{r['median_density']:6.0f}  {r['mode']}")
    total = {str(r): float(np.mean(dist <= r)) for r in radii}
    print("\noverall coverage: " + ", ".join(f"{v:.1%} within {k}" for k, v in total.items())
          + f" of {len(gt)} ground-truth states")
    (Path(args.run_dir) / "analytic_coverage.json").write_text(json.dumps(
        dict(radii=radii, density_radius=args.density_radius, overall=total, modes=rows), indent=1))
    np.savez(Path(args.run_dir) / "analytic_gt_coverage.npz", dist=dist.astype(np.float32),
             density=np.asarray(density, np.int32))


if __name__ == "__main__":
    main()
