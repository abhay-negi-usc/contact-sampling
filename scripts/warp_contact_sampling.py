#!/usr/bin/env python3
"""Contact sampling of the hook on the eyelet on the GPU (hook_warp_contact), writing the same files as the CPU
pipeline (mode_graph_experiments prepare-samples / prepare-contact, hook_contact_strata strata), so the mode stages
(hook_contact_strata formations ..., hook_sided_modes) run on the result unchanged.

    samples   free poses (seeds), from three sources:
                wide      uniform in +-(--wide-half) mm / deg around the max-separation reference path, kept within
                          --near-mm of contact or with probability --far-keep (as the CPU pipeline)
                narrow    uniform in +-(--narrow-mm, --narrow-deg) around both reference paths, kept as wide
                threaded  uniform in +-(--threaded-mm, --threaded-deg) around the threaded stretches of both paths,
                          kept only if threaded (the sparsely sampled states near the goal)
              -> samples_graph.npz: poses, feat, thread, source, kNN pairs (lever 10) with their clearance
    contact   seeds and --copies jittered copies slid along the contact normal to --contact-target, thinned to one
              per --contact-cell cell (lever-10 metric); features, threading, closest-point and multi-contact
              wrenches; kNN pairs with how far each slide strays from contact and how far its projection jumps
              -> contact_graph.npz
    strata    the CAD elements (face, side, rim of each part) of every pose's distinct contacts within
              --contact-eps of the closest -> strata/contacts.npz (CAD faces and aliases copied from --cad-dir)

    python scripts/warp_contact_sampling.py samples OUT
    python scripts/warp_contact_sampling.py contact OUT
    python scripts/warp_contact_sampling.py strata OUT
"""

import argparse
import shutil
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from build_separation_field import reference_path
from hook_eyelet_model import ASSET_DIR
from hook_warp_contact import MAXC, WarpContact

PATHS = ("hook_assembly_max_separation_trajectory.csv", "hook_assembly_ideal_trajectory.csv")
SOURCES = ("wide", "narrow", "threaded")
LEVER = 10.0
SCALE = np.array([1, 1, 1] + [LEVER * np.pi / 180] * 3)


def draw(geo, centres, half, n, keep, rng, batch=1 << 20):
    """n poses uniform in centre +- half (centres drawn uniformly), filtered by keep(poses, separation); and the
    acceptance rate."""
    got, kept, tried = 0, [], 0
    while got < n:
        c = centres[rng.integers(len(centres), size=batch)] + rng.uniform(-1, 1, (batch, 6)) * half
        s = geo.separation(c)
        k = keep(c, s)
        kept.append(np.c_[c[k], s[k]])
        got += k.sum()
        tried += batch
    out = np.vstack(kept)[:n]
    return out[:, :6], out[:, 6], got / tried


def knn_pairs(poses, k):
    x = poses * SCALE
    _, nbr = cKDTree(x).query(x, k=k + 1)
    pr = np.sort(np.column_stack([np.repeat(np.arange(len(poses)), k), nbr[:, 1:].ravel()]), 1)
    pr = pr[pr[:, 0] != pr[:, 1]]
    return np.unique(pr, axis=0)


def features(geo, poses, band=2.0):
    """mode_graph_experiments.features (separation, hook-frame and eyelet-frame closest points), the contacts."""
    c = geo.contacts(poses, band=band)
    feat = np.c_[c["dist"][:, 0], c["hook_local"][:, 0], c["eyelet"][:, 0]].astype(np.float64)
    feat[~np.isfinite(feat[:, 0]), 0] = geo.separation(poses[~np.isfinite(feat[:, 0])])
    return feat, c


def wrenches(poses, c, decay):
    """Closest-point and multi-contact unit wrenches [n, r x n] (torque about the hook origin, mm), as
    mode_graph_experiments.Probe.wrench: the multi-contact one weights each distinct contact within the band by
    exp(-(d - d_min) / decay)."""
    r = c["hook"] - poses[:, None, :3]
    unit = np.concatenate([c["normal"], np.cross(r, c["normal"])], axis=2)
    closest = unit[:, 0]
    d = c["dist"]
    w = np.where(np.isfinite(d) & (d > 0), np.exp(-(d - d[:, :1]) / decay), 0.0)
    multi = np.einsum("nk,nkj->nj", w, np.nan_to_num(unit)) / np.maximum(w.sum(1, keepdims=True), 1e-12)
    multi = np.where(w.sum(1, keepdims=True) > 0, multi, closest)
    return closest, multi


def samples_stage(args):
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    geo = WarpContact()
    rng = np.random.default_rng(args.seed)
    paths = [reference_path(ASSET_DIR / p, spacing=0.25) for p in PATHS]
    t0 = time.time()
    near_or_far = lambda c, s: (s >= 0) & ((s <= args.near_mm) | (rng.random(len(s)) < args.far_keep))
    parts, info = [], {}
    p, s, n = draw(geo, paths[0], args.wide_half, args.wide, near_or_far, rng)
    parts.append((p, s, 0))
    info["wide"] = n
    half = np.array([args.narrow_mm] * 3 + [args.narrow_deg] * 3)
    p, s, n = draw(geo, np.vstack(paths), half, args.narrow, near_or_far, rng)
    parts.append((p, s, 1))
    info["narrow"] = n
    both = np.vstack(paths)
    threaded = both[geo.threading(both) > 0]
    half = np.array([args.threaded_mm] * 3 + [args.threaded_deg] * 3)

    def free_threaded(c, s):
        k = s >= 0
        k[k] = geo.threading(c[k]) > 0
        return k

    p, s, n = draw(geo, threaded, half, args.threaded, free_threaded, rng)
    parts.append((p, s, 2))
    info["threaded"] = n
    poses = np.vstack([q for q, _, _ in parts])
    source = np.concatenate([np.full(len(q), k, np.int8) for q, _, k in parts])
    for (q, _, k), name in zip(parts, SOURCES):
        print(f"  {name:9s} {len(q):9,d} seeds ({info[name]:.2%} of candidates kept)", flush=True)
    feat, _ = features(geo, poses, band=1e-3)
    thread = geo.threading(poses)
    print(f"{len(poses):,} seeds with features ({time.time() - t0:.0f} s); threaded {np.mean(thread > 0):.1%} "
          f"(by source: {', '.join(f'{n} {np.mean(thread[source == k] > 0):.1%}' for k, n in enumerate(SOURCES))})",
          flush=True)
    t0 = time.time()
    pairs = knn_pairs(poses, args.k)
    clear = geo.segment_min(poses[pairs[:, 0]], poses[pairs[:, 1]], args.step, floor=0.0)  # capped at 2 mm
    print(f"{len(pairs):,} neighbour pairs, {np.mean(clear >= 0):.0%} in line of sight ({time.time() - t0:.0f} s)",
          flush=True)
    np.savez(out / "samples_graph.npz", poses=poses, feat=feat, thread=thread.astype(np.int8), source=source,
             pairs=pairs, clear=clear, **{"in_10.0": np.ones(len(pairs), bool)})


def contact_stage(args):
    out = args.out_dir
    geo = WarpContact()
    z = np.load(out / "samples_graph.npz")
    src, src_label = z["poses"], z["source"]
    rng = np.random.default_rng(args.seed)
    jitter = lambda x: x + rng.uniform(-1, 1, x.shape) * np.array([2.0] * 3 + [5.0] * 3)
    starts = np.vstack([src] + [jitter(src) for _ in range(args.copies)])
    label = np.tile(src_label, args.copies + 1)
    t0 = time.time()
    poses, ok = geo.project(starts, args.contact_target, args.contact_tol)
    poses, label = poses[ok], label[ok]
    cell = np.floor(poses * SCALE / args.contact_cell).astype(np.int64)
    _, first = np.unique(cell, axis=0, return_index=True)
    first = np.sort(first)
    poses, label = poses[first], label[first]
    print(f"{len(starts):,} starts, {ok.mean():.1%} reached contact, {len(poses):,} after thinning to one per "
          f"{args.contact_cell} mm cell ({time.time() - t0:.0f} s); by seed source: "
          + ", ".join(f"{n} {np.sum(label == k):,}" for k, n in enumerate(SOURCES)), flush=True)
    t0 = time.time()
    feat, c = features(geo, poses)
    thread = geo.threading(poses)
    closest, multi = wrenches(poses, c, args.wrench_decay)
    print(f"features and wrenches ({time.time() - t0:.0f} s); separation p50 {np.median(feat[:, 0]):.3f} mm, max "
          f"{feat[:, 0].max():.3f} mm; threaded {np.mean(thread > 0):.1%} ({np.sum(thread > 0):,} poses)", flush=True)
    t0 = time.time()
    pairs = knn_pairs(poses, args.k)
    stray, jump = geo.slide(poses[pairs[:, 0]], poses[pairs[:, 1]], args.step, args.contact_target, args.contact_reach)
    print(f"{len(pairs):,} neighbour pairs; segment strays from contact by p50 {np.median(stray):.2f} mm, "
          f"{np.mean(stray <= 0.25):.0%} within 0.25 mm, {np.mean(stray <= 1):.0%} within 1 mm; projection jump p50 "
          f"{np.median(jump):.2f} mm ({time.time() - t0:.0f} s)", flush=True)
    np.savez(out / "contact_graph.npz", poses=poses, feat=feat, thread=thread.astype(np.int8), source=label,
             pairs=pairs, stray=stray, jump=jump, clear=-stray, wrench_closest=closest, wrench_multi=multi,
             **{"in_10.0": np.ones(len(pairs), bool)})
    np.savez(out / "contact_points.npz", **c)


def strata_arrays(c, hook_el, eye_el, eps, rim_mm):
    """hook_contact_strata.strata_task's output from contacts (hook_warp_contact.WarpContact.contacts): per pose up to
    MAXC distinct contacts within eps of the closest as (hook face, side, rim, eyelet face, side, rim, gap [0.01 mm]),
    and their hook-frame and eyelet-frame witness points."""
    d = c["dist"]
    n = len(d)
    touch = np.isfinite(d) & (d <= d[:, :1] + eps)
    out = np.full((n, MAXC, 7), -1, np.int32)
    wit = np.full((n, MAXC, 6), np.nan)
    hf, hs, hr = hook_el(c["hook_local"][touch], rim_mm)
    ef, es, er = eye_el(c["eyelet"][touch], rim_mm)
    gap = np.round((d - d[:, :1]) * 100)[touch]
    out[touch] = np.c_[hf, hs, hr, ef, es, er, gap].astype(np.int32)
    wit[touch] = np.c_[c["hook_local"][touch], c["eyelet"][touch]]
    return out, wit, touch.sum(1)


def strata_stage(args):
    from hook_contact_strata import Elements

    out, strata = args.out_dir, args.out_dir / "strata"
    strata.mkdir(exist_ok=True)
    for f in ("faces_hook.npz", "faces_eyelet.npz", "cad_faces.json", "aliases.json"):
        shutil.copy(args.cad_dir / f, strata / f)
    z = dict(np.load(out / "contact_points.npz"))
    t0 = time.time()
    contacts, witness, ncon = strata_arrays(z, Elements(np.load(strata / "faces_hook.npz")),
                                            Elements(np.load(strata / "faces_eyelet.npz")), args.contact_eps, args.rim_mm)
    print(f"{len(ncon):,} contact poses ({time.time() - t0:.0f} s); contacts per pose: "
          + ", ".join(f"{k}: {np.mean(ncon == k):.1%}" for k in range(MAXC + 1) if np.any(ncon == k)), flush=True)
    np.savez(strata / "contacts.npz", contacts=contacts, witness=witness, ncon=ncon)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["samples", "contact", "strata"])
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--wide", type=int, default=400000, help="seeds from the wide tube")
    parser.add_argument("--wide-half", type=float, default=20.0, help="[mm and deg], as the CPU pipeline")
    parser.add_argument("--near-mm", type=float, default=5.0)
    parser.add_argument("--far-keep", type=float, default=0.1)
    parser.add_argument("--narrow", type=int, default=300000, help="seeds from the narrow tubes")
    parser.add_argument("--narrow-mm", type=float, default=3.0)
    parser.add_argument("--narrow-deg", type=float, default=6.0)
    parser.add_argument("--threaded", type=int, default=300000, help="threaded seeds")
    parser.add_argument("--threaded-mm", type=float, default=6.0)
    parser.add_argument("--threaded-deg", type=float, default=20.0)
    parser.add_argument("--k", type=int, default=12, help="nearest neighbours (lever-10 metric)")
    parser.add_argument("--step", type=float, default=0.25, help="segment check step [mm / deg]")
    parser.add_argument("--copies", type=int, default=1, help="jittered copies of each seed to slide into contact")
    parser.add_argument("--contact-target", type=float, default=0.02)
    parser.add_argument("--contact-tol", type=float, default=0.02)
    parser.add_argument("--contact-cell", type=float, default=0.75)
    parser.add_argument("--contact-reach", type=float, default=1.0)
    parser.add_argument("--wrench-decay", type=float, default=0.5)
    parser.add_argument("--contact-eps", type=float, default=0.1, help="contacts this close to the closest touch too")
    parser.add_argument("--rim-mm", type=float, default=0.3)
    parser.add_argument("--cad-dir", type=Path,
                        default=Path("/media/rp/Elements1/abhay_ws/contact-sampling/separation_field/run_20261024/"
                                     "mode_experiments/strata"))
    args = parser.parse_args()
    {"samples": samples_stage, "contact": contact_stage, "strata": strata_stage}[args.stage](args)


if __name__ == "__main__":
    main()
