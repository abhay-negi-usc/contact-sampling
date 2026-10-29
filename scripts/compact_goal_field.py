#!/usr/bin/env python3
"""Fold the routed pockets into the goal field and write it in compact form; check it against the full one.

Pockets (route_pockets.py) with a valid route get values and moves too: inside a pocket, V is the most
clearance any path to the pocket's start keeps (capped by the route's checked clearance), steps go to
the start (stored as DIST_POCKET + steps), and the start's route leads out to the assembled pose. Pockets
with the shaft through the hole, or without a route, stay unreachable (V = -1).

Compact form (GoalField.save_compact): the free-node bitmask, the block list, V (int16), dist (uint16) and
the pocket routes, zlib-compressed; the rest of the index is rebuilt on load and moves are derived.

Checks: the compact field reloads to the same V and dist and the same index; paths followed with derived
moves from --check random nodes take as many steps as the stored moves and never lower V; and paths from
random nodes of --check-pockets routed pockets reach the assembled pose clear of the eyelet (dense check).

    conda run -n mujoco_env python scripts/compact_goal_field.py FIELD_DIR --routes MODE_GRAPH_DIR
"""

import argparse
import heapq
import json
import time
from collections import deque
from pathlib import Path

import numpy as np

import validate_goal_field as vg
from build_separation_field import UNIT_MM
from goal_field import DIST_POCKET, STEPS, GoalField, node_coords


def pocket_values(field, ids, coords, start, cap):
    """V (capped) and steps to start inside one pocket, as in solve_values / solve_moves."""
    local = {tuple(c): k for k, c in enumerate(coords)}
    sep = field.sep[ids].astype(np.int64)
    s = local[tuple(start)]
    best = np.full(len(ids), -1)
    best[s] = sep[s]
    heap = [(-best[s], s)]
    while heap:  # widest path from the start
        b, u = heapq.heappop(heap)
        if -b < best[u]:
            continue
        for st in STEPS:
            v = local.get(tuple(coords[u] + st))
            if v is not None and min(-b, sep[v]) > best[v]:
                best[v] = min(-b, sep[v])
                heapq.heappush(heap, (-best[v], v))
    V = np.minimum(best, cap)
    steps = np.full(len(ids), -1)
    steps[s] = 0
    queue = deque([s])
    while queue:  # BFS back from the start over moves that do not lower V
        p = queue.popleft()
        for st in STEPS:
            q = local.get(tuple(coords[p] + st))
            if q is not None and steps[q] < 0 and V[q] <= V[p]:
                steps[q] = steps[p] + 1
                queue.append(q)
    assert np.all(steps >= 0)
    return V, steps


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("field_dir", type=Path)
    parser.add_argument("--routes", type=Path, required=True, help="directory of pocket_routes.json/.npz")
    parser.add_argument("--out", type=Path, help="default: FIELD_DIR/goal_field_compact.npz")
    parser.add_argument("--check", type=int, default=2000)
    parser.add_argument("--check-pockets", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    out = args.out or args.field_dir / "goal_field_compact.npz"
    rng = np.random.default_rng(args.seed)
    report = {}

    t0 = time.time()
    field = GoalField.load(args.field_dir / "goal_field.npz")
    full_move = field.move
    print(f"full field loaded ({time.time() - t0:.0f} s)", flush=True)
    pz = np.load(args.field_dir / "goal_field_pockets.npz")
    info = json.loads((args.routes / "pocket_routes.json").read_text())["pockets"]
    routes = np.load(args.routes / "pocket_routes.npz")
    pocket_nodes = {}
    for p, rec in info.items():
        if rec.get("status") != "valid":
            continue
        coords = pz["coords"][pz["comp"] == int(p)]
        ids = np.array([field.node(c) for c in coords])
        start = np.array(rec["start"], np.int64)
        cap = int(np.floor(rec["checked_min_mm"] / UNIT_MM))
        V, steps = pocket_values(field, ids, coords, start, cap)
        field.V[ids] = V
        field.dist[ids] = DIST_POCKET + steps
        field.portals[int(field.node(start))] = routes[f"pocket_{p}"]
        pocket_nodes[int(p)] = ids
    assert field.dist.max() < 65535
    report["pockets_folded_in"] = len(pocket_nodes)
    report["pocket_nodes_folded_in"] = int(sum(len(v) for v in pocket_nodes.values()))
    report["unreachable_free_nodes"] = int(np.sum(field.V < 0))

    t0 = time.time()
    field.save_compact(out)
    report["compact_mb"] = round(out.stat().st_size / 1e6, 1)
    report["full_mb"] = round((args.field_dir / "goal_field.npz").stat().st_size / 1e6, 1)
    report["save_s"] = round(time.time() - t0)
    t0 = time.time()
    compact = GoalField.load_compact(out)
    report["load_compact_s"] = round(time.time() - t0, 1)
    print(f"compact field: {report['compact_mb']} MB (full {report['full_mb']} MB); "
          f"loads in {report['load_compact_s']} s", flush=True)

    for k in ("V", "dist", "words", "prefix", "offset", "lut"):
        a, b = getattr(field, k), getattr(compact, k)
        assert np.array_equal(a.astype(np.int64), b.astype(np.int64)), f"{k} differs after reload"
    report["reload_identical"] = True

    connected = np.flatnonzero((field.V >= 0) & (field.dist < DIST_POCKET))
    c = np.empty(6, np.int64)
    steps_equal, v_kept, times = 0, 0, []
    field.move = full_move
    for i in rng.choice(connected, args.check, replace=False):
        node_coords(i, c, field.block_coords, field.words, field.prefix, field.offset)
        t1 = time.perf_counter()
        path = compact.follow(c.copy())
        times.append(time.perf_counter() - t1)
        ref = field.follow(c.copy())
        steps_equal += len(path) == len(ref)
        v_path = min(compact.V[compact.node(p.astype(np.int64))] for p in path)
        v_kept += v_path >= compact.V[i]
    report.update(derived_paths_same_steps=f"{steps_equal}/{args.check}", derived_paths_keep_V=f"{v_kept}/{args.check}",
                  follow_ms_median=round(1e3 * float(np.median(times)), 2), follow_ms_max=round(1e3 * max(times), 2))
    print(f"derived moves: same steps {steps_equal}/{args.check}, V kept {v_kept}/{args.check}; "
          f"follow {report['follow_ms_median']} ms median", flush=True)

    vg._init()
    vg._field = compact
    checked = []
    for p in rng.choice(sorted(pocket_nodes), min(args.check_pockets, len(pocket_nodes)), replace=False):
        ids = pocket_nodes[p]
        i = rng.choice(ids)
        node_coords(i, c, compact.block_coords, compact.words, compact.prefix, compact.offset)
        path = compact.follow(c.copy())
        approach = path[:-1]  # the last leg is the final lift into resting contact
        checked.append({"pocket": int(p), "nodes": len(ids), "V_mm": float(compact.V[i] * UNIT_MM),
                        "checked_min_mm": float(vg.path_min(approach, 0.1)), "poses": len(path)})
    report["pocket_paths"] = checked
    report["pocket_paths_valid"] = f"{sum(r['checked_min_mm'] >= 0 for r in checked)}/{len(checked)}"
    print(f"pocket paths clear of the eyelet: {report['pocket_paths_valid']}", flush=True)
    out.with_suffix(".json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "pocket_paths"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
