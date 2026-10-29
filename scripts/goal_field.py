#!/usr/bin/env python3
"""Goal-directed field: from every collision-free pose, the best worst-case separation that can be kept
all the way to assembly, and the next step that keeps it.

Built offline from the 1 mm / 1 deg grid of the separation field (build_separation_field.py). Nodes are
the grid's collision-free poses (separation >= 0); a move is one grid step in one coordinate (12
neighbours). The goal is PRE_ASSEMBLY, the hook 3 mm straight below the assembled pose, where the
separation peaks at 2.5 mm; the final 3 mm lift into resting contact is a fixed last move.

    V(q)    = max over paths q -> goal of the smallest separation along the path
    move(q) = a step to a neighbour n with V(n) >= V(q), fewest steps to the goal among such moves

So following move() from q never takes the worst-case separation below V(q), and reaches the goal.
This is the bottleneck (max-min) Bellman equation V(q) = max_n min(sep(q), V(n)), solved in one pass
rather than by sweeps: nodes are added in decreasing separation with union-find, and a node's value is
the separation level at which it first joins the goal's component. move() comes from a breadth-first
search back from the goal over steps that do not lower V. Values are node-based, so a path between
nodes is only as clear as a dense check says; extracted paths are always re-checked.

    conda run -n mujoco_env python scripts/goal_field.py FIELD_DIR    # writes FIELD_DIR/goal_field.npz
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from numba import njit

from build_separation_field import B, TUBE_OUT, UNIT_MM, load_parts

PRE_ASSEMBLY = np.array([0, 0, -3, 0, 0, 0], np.int64)  # grid node: hook 3 mm below the assembled pose
STEPS = np.array([[s if i == d else 0 for i in range(6)] for d in range(6) for s in (-1, 1)], np.int64)
NO_MOVE = 255
# Nodes of a routed pocket store DIST_POCKET + their steps to the pocket's start, where a stored route
# (a "portal", route_pockets.py) leads out of the pocket to the assembled pose.
DIST_POCKET = 60000


@njit(cache=True, inline="always")
def popcount(x):
    x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return np.int64((x * np.uint64(0x0101010101010101)) >> np.uint64(56))


@njit(cache=True)
def node_id(c, lut, lut_lo, lut_shape, words, prefix, offset):
    """Index of the free node at grid coordinates c, or -1 if it is not a free node of the field."""
    bl, lin = 0, 0
    for i in range(6):
        b = c[i] // B
        k = b - lut_lo[i]
        if k < 0 or k >= lut_shape[i]:
            return -1
        bl = bl * lut_shape[i] + k
        lin = lin * B + (c[i] - b * B)
    blk = lut[bl]
    if blk < 0:
        return -1
    w, bit = lin >> 6, lin & 63
    word = words[blk, w]
    if (word >> np.uint64(bit)) & np.uint64(1) == 0:
        return -1
    return offset[blk] + prefix[blk, w] + popcount(word & ((np.uint64(1) << np.uint64(bit)) - np.uint64(1)))


@njit(cache=True)
def node_coords(i, out, block_coords, words, prefix, offset):
    """Grid coordinates of free node i, written to out."""
    lo, hi = 0, len(offset) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if offset[mid] <= i:
            lo = mid
        else:
            hi = mid
    blk, r = lo, i - offset[lo]
    a, z = 0, words.shape[1]
    while z - a > 1:
        mid = (a + z) // 2
        if prefix[blk, mid] <= r:
            a = mid
        else:
            z = mid
    word, k, bit = words[blk, a], r - prefix[blk, a], 0
    while True:
        if (word >> np.uint64(bit)) & np.uint64(1):
            if k == 0:
                break
            k -= 1
        bit += 1
    lin = a * 64 + bit
    for d in range(5, -1, -1):
        out[d] = block_coords[blk, d] * B + lin % B
        lin //= B


@njit(cache=True)
def find(parent, x):
    root = np.int64(x)
    while parent[root] != root:
        root = np.int64(parent[root])
    x = np.int64(x)
    while parent[x] != root:
        parent[x], x = root, np.int64(parent[x])
    return root


@njit(cache=True)
def mark(V, nxt, head, level):
    """Give every member of the component listed from head the value level."""
    x = head
    while x != -1:
        V[x] = level
        x = nxt[x]


@njit(cache=True)
def solve_values(sep, goal, block_coords, lut, lut_lo, lut_shape, words, prefix, offset):
    """V for every free node (-1 if never connected to the goal), by union-find in decreasing separation."""
    n = len(sep)
    smax = sep.max()
    start = np.zeros(smax + 2, np.int64)
    for i in range(n):
        start[sep[i]] += 1
    acc = 0
    for v in range(smax, -1, -1):  # descending order of separation
        cnt = start[v]
        start[v] = acc
        acc += cnt
    order = np.empty(n, np.int32)
    for i in range(n):
        order[start[sep[i]]] = i
        start[sep[i]] += 1

    parent = np.arange(n, dtype=np.int32)
    rank = np.zeros(n, np.uint8)
    nxt = np.full(n, -1, np.int32)  # member lists, headed by each component's root
    tail = np.arange(n, dtype=np.int32)
    has_goal = np.zeros(n, np.uint8)
    added = np.zeros(n, np.uint8)
    V = np.full(n, -1, np.int16)
    c, nb = np.empty(6, np.int64), np.empty(6, np.int64)
    for idx in range(n):
        u = order[idx]
        level = sep[u]
        added[u] = 1
        if u == goal:
            has_goal[u], V[u] = 1, level
        node_coords(u, c, block_coords, words, prefix, offset)
        for s in range(12):
            for d in range(6):
                nb[d] = c[d] + STEPS[s, d]
            v = node_id(nb, lut, lut_lo, lut_shape, words, prefix, offset)
            if v < 0 or added[v] == 0:
                continue
            ru, rv = find(parent, u), find(parent, v)
            if ru == rv:
                continue
            if has_goal[ru] and not has_goal[rv]:  # rv's component joins the goal at this level
                mark(V, nxt, rv, level)
            elif has_goal[rv] and not has_goal[ru]:
                mark(V, nxt, ru, level)
            if rank[ru] < rank[rv]:
                ru, rv = rv, ru
            parent[rv] = ru
            if rank[ru] == rank[rv]:
                rank[ru] += 1
            nxt[tail[ru]] = rv
            tail[ru] = tail[rv]
            has_goal[ru] |= has_goal[rv]
    return V


@njit(cache=True)
def solve_moves(V, goal, block_coords, lut, lut_lo, lut_shape, words, prefix, offset):
    """Steps to the goal and the move to take, by BFS back from the goal over moves that keep V."""
    n = len(V)
    dist = np.full(n, -1, np.int32)
    move = np.full(n, NO_MOVE, np.uint8)
    queue = np.empty(n, np.int32)
    head, tail = 0, 1
    queue[0], dist[goal] = goal, 0
    c, nb = np.empty(6, np.int64), np.empty(6, np.int64)
    while head < tail:
        p = queue[head]
        head += 1
        node_coords(p, c, block_coords, words, prefix, offset)
        for s in range(12):
            for d in range(6):
                nb[d] = c[d] + STEPS[s, d]
            q = node_id(nb, lut, lut_lo, lut_shape, words, prefix, offset)
            if q < 0 or dist[q] >= 0 or V[q] < 0 or V[q] > V[p]:
                continue
            dist[q] = dist[p] + 1
            move[q] = s ^ 1  # the opposite step leads from q back to p
            queue[tail] = q
            tail += 1
    return dist, move


class GoalField:
    """The free-node index plus V, dist and (optionally) move; coordinates are grid nodes (1 mm / 1 deg).

    The full form (save/load) stores every array. The compact form (save_compact/load_compact) stores only
    the free-node bitmask, the block list, V, dist and the pocket routes: the rest of the index is rebuilt
    on load, and moves are derived on the fly, as a step to a neighbour one step closer with V no lower.
    """

    KEYS = ("block_coords", "lut", "lut_lo", "lut_shape", "words", "prefix", "offset", "sep", "V", "dist", "move")

    def __init__(self, **arrays):
        for k in self.KEYS:
            setattr(self, k, arrays.get(k))
        self.goal = int(arrays["goal"])
        # pocket routes: start node -> poses (mm / deg) from the pocket's start to the assembled pose
        self.portals = {}
        if "portal_nodes" in arrays:
            off = arrays["portal_offsets"]
            for k, node in enumerate(arrays["portal_nodes"]):
                self.portals[int(node)] = arrays["portal_poses"][off[k]:off[k + 1]].astype(float)

    @classmethod
    def load(cls, path):
        with np.load(path) as z:
            return cls(**{k: z[k] for k in z.files})

    def save(self, path):
        np.savez(path, goal=self.goal, **{k: getattr(self, k) for k in self.KEYS})

    def save_compact(self, path):
        nodes = np.array(sorted(self.portals), np.int64)
        poses = [self.portals[n] for n in nodes]
        np.savez_compressed(
            path, goal=self.goal, block_coords=self.block_coords.astype(np.int16), words=self.words, V=self.V,
            dist=self.dist.astype(np.uint16), portal_nodes=nodes,
            portal_offsets=np.concatenate([[0], np.cumsum([len(p) for p in poses])]).astype(np.int64),
            portal_poses=(np.vstack(poses) if poses else np.zeros((0, 6))).astype(np.float32))

    @classmethod
    def load_compact(cls, path):
        with np.load(path) as z:
            a = {k: z[k] for k in z.files}
        a["block_coords"] = a["block_coords"].astype(np.int64)
        a["dist"] = a["dist"].astype(np.int32)
        a["dist"][a["dist"] == np.iinfo(np.uint16).max] = -1  # unreachable: -1 was stored as uint16
        counts = np.bitwise_count(a["words"]).astype(np.int32)
        a["prefix"] = np.cumsum(counts, axis=1, dtype=np.int32) - counts
        a["offset"] = np.concatenate([[0], np.cumsum(counts.sum(1, dtype=np.int64))])
        lo = a["block_coords"].min(0)
        shape = a["block_coords"].max(0) - lo + 1
        lut = np.full(int(np.prod(shape)), -1, np.int32)
        lut[np.ravel_multi_index(tuple((a["block_coords"] - lo).T), tuple(shape))] = np.arange(len(a["block_coords"]))
        a.update(lut=lut, lut_lo=lo, lut_shape=shape)
        return cls(**a)

    def index(self):
        return self.lut, self.lut_lo, self.lut_shape, self.words, self.prefix, self.offset

    def node(self, coords):
        return node_id(np.asarray(coords, np.int64), *self.index())

    def value_mm(self, coords):
        i = self.node(coords)
        return None if i < 0 or self.V[i] < 0 else self.V[i] * UNIT_MM

    def next_step(self, i, c):
        """Index into STEPS of the move from node i (at c): stored, or derived from dist and V."""
        if self.move is not None:
            return self.move[i]
        for s in range(len(STEPS)):
            n = self.node(c + STEPS[s])
            if n >= 0 and self.dist[n] == self.dist[i] - 1 and self.V[n] >= self.V[i]:
                return s
        raise RuntimeError(f"no move from node {i}")

    def follow(self, coords):
        """Poses (mm / deg) from coords to the goal, or None if coords is not connected.

        Ends at the pre-assembly pose, or for a pocket at the assembled pose via the pocket's stored route.
        """
        i = self.node(coords)
        if i < 0 or self.V[i] < 0:
            return None
        c = np.asarray(coords, np.int64).copy()
        path = [c.astype(float)]
        while i != self.goal:
            if self.dist[i] == DIST_POCKET:  # a pocket's start: take its stored route out
                return np.vstack([np.array(path), self.portals[i][1:]])
            c += STEPS[self.next_step(i, c)]
            i = self.node(c)
            path.append(c.astype(float))
        return np.array(path)


def build_index(field_dir):
    """Packed bitmask index of the free nodes (separation >= 0) of the 1 mm grid, block by block."""
    blocks, words, seps = [], [], []
    for bs, vs in load_parts(field_dir / "coarse", "blocks"):
        for b, v in zip(bs, vs):
            v = v.reshape(-1)
            free = (v != TUBE_OUT) & (v >= 0)
            if free.any():
                blocks.append(b)
                words.append(np.packbits(free, bitorder="little").view(np.uint64))
                seps.append(v[free])
    block_coords = np.array(blocks, np.int64)
    words = np.stack(words)
    counts = np.bitwise_count(words).astype(np.int32)
    prefix = np.cumsum(counts, axis=1, dtype=np.int32) - counts
    offset = np.concatenate([[0], np.cumsum(counts.sum(1, dtype=np.int64))])
    lut_lo = block_coords.min(0)
    lut_shape = block_coords.max(0) - lut_lo + 1
    lut = np.full(int(np.prod(lut_shape)), -1, np.int32)
    lut[np.ravel_multi_index(tuple((block_coords - lut_lo).T), tuple(lut_shape))] = np.arange(len(block_coords))
    return dict(block_coords=block_coords, lut=lut, lut_lo=lut_lo, lut_shape=lut_shape, words=words,
                prefix=prefix, offset=offset, sep=np.concatenate(seps))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("field_dir", type=Path)
    parser.add_argument("--out", type=Path, help="default: FIELD_DIR/goal_field.npz")
    args = parser.parse_args()
    out = args.out or args.field_dir / "goal_field.npz"

    t0 = time.time()
    cache = out.with_name(out.stem + "_index.npz")  # reading 8,000 parts takes minutes; reuse across runs
    if cache.exists():
        with np.load(cache) as z:
            arrays = {k: z[k] for k in z.files}
    else:
        arrays = build_index(args.field_dir)
        np.savez(cache, **arrays)
    index = tuple(arrays[k] for k in ("lut", "lut_lo", "lut_shape", "words", "prefix", "offset"))
    n = len(arrays["sep"])
    c = np.empty(6, np.int64)
    for i in np.random.default_rng(0).integers(0, n, 1000):
        node_coords(i, c, arrays["block_coords"], arrays["words"], arrays["prefix"], arrays["offset"])
        assert node_id(c, *index) == i, "index round trip failed"
    goal = node_id(PRE_ASSEMBLY, *index)
    assert goal >= 0, "pre-assembly pose is not a free node of the field"
    print(f"{n:,} free nodes in {len(arrays['block_coords']):,} blocks; index built in {time.time() - t0:.0f} s; "
          f"pre-assembly separation {arrays['sep'][goal] * UNIT_MM:.2f} mm", flush=True)

    t1 = time.time()
    V = solve_values(arrays["sep"], goal, arrays["block_coords"], *index)
    print(f"values in {time.time() - t1:.0f} s: {np.sum(V >= 0):,} nodes connected to the goal", flush=True)
    t2 = time.time()
    dist, move = solve_moves(V, goal, arrays["block_coords"], *index)
    print(f"moves in {time.time() - t2:.0f} s: {np.sum(dist >= 0):,} nodes reached, at most {dist.max()} steps",
          flush=True)
    assert np.array_equal(dist >= 0, V >= 0), "BFS must reach every connected node"

    field = GoalField(goal=goal, V=V, dist=dist, move=move, **arrays)
    field.save(out)
    connected = V[V >= 0] * UNIT_MM
    summary = {"free_nodes": n, "connected_nodes": int(np.sum(V >= 0)), "pre_assembly": PRE_ASSEMBLY.tolist(),
               "value_mm_percentiles": dict(zip(["p1", "p10", "p50", "p90", "p99"],
                                                np.percentile(connected, [1, 10, 50, 90, 99]).round(2).tolist())),
               "max_steps_to_goal": int(dist.max()), "build_s": round(time.time() - t0)}
    out.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
