#!/usr/bin/env python3
"""Split the concave hook/eyelet meshes into convex parts for MuJoCo collision.

MuJoCo collides every mesh geom through its convex hull, so the hook and the eyelet
used directly would collide as solid blobs with no crook and no hole. CoACD splits each
mesh into convex pieces whose union follows the original surface; the simulation uses
one collision geom per piece and keeps the original mesh for rendering only.

Parts are written in the STL's own units to <asset-dir>/convex/<name>/<name>_NNN.stl.

    conda run -n mujoco_env python scripts/convex_decompose_meshes.py
"""

import argparse
from pathlib import Path

import coacd
import trimesh

ASSET_DIR = Path(__file__).resolve().parents[1] / "assets" / "hook"


def decompose(mesh_path, out_dir, threshold, seed):
    mesh = trimesh.load(mesh_path)
    parts = coacd.run_coacd(coacd.Mesh(mesh.vertices, mesh.faces), threshold=threshold,
                            max_convex_hull=-1, seed=seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob(f"{mesh_path.stem}_*.stl"):
        stale.unlink()
    hulls = [trimesh.Trimesh(v, f) for v, f in parts]
    for i, hull in enumerate(hulls):
        hull.export(out_dir / f"{mesh_path.stem}_{i:03d}.stl")
    vol_ratio = sum(h.volume for h in hulls) / mesh.volume
    print(f"{mesh_path.name}: {len(hulls)} convex parts -> {out_dir} (hull/mesh volume {vol_ratio:.3f})")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--asset-dir", type=Path, default=ASSET_DIR)
    parser.add_argument("--names", nargs="+", default=["hook", "eyelet"])
    # 0.008 gives ~20-30 parts per mesh with hulls bulging <0.25 mm past the true surface;
    # 0.004 gives ~130 parts for little further gain, 0.03 bulges by up to ~1 mm.
    parser.add_argument("--threshold", type=float, default=0.008,
                        help="CoACD concavity threshold; lower = more, tighter parts")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    coacd.set_log_level("error")
    for name in args.names:
        decompose(args.asset_dir / f"{name}.stl", args.asset_dir / "convex" / name, args.threshold, args.seed)


if __name__ == "__main__":
    main()
