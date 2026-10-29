#!/usr/bin/env python3
"""Benchmark for self-discovering contact modes, checked against the known modes of the peg and hole.

Dataset: an evenly thinned subset of a sampler run in the symmetry-reduced coordinates (rho, radial
tilt, signed tangential tilt, depth) with what a robot could observe: the pose, the measured wrench
(with friction), the walk velocity and the trajectory links. The analytical contact modes
(peg_hole_analytic.py) and the simulator's contact normals are kept for scoring and diagnostics only.

Features (cached with the dataset):
  pose       the reduced coordinates
  wrench     measured wrench in the reduced frame (tip on +x), torque about the tip / 12.5 mm, in N
  normal     unit normal of the sampled contact manifold from local PCA of the poses (sign-free: the
             outer product's upper triangle), and the surface variation lambda_min / sum(lambda)
  curvature  principal curvatures of a local quadratic fit of the manifold about that normal
  contact    friction-free contact wrench from the simulator's contact points and normals (privileged)

Truth levels: "family" (in the bore / face on mouth / edge on mouth / edge on inner rim, the four
families the stratum graph gives) and "mode" (every analytical mode).

    conda run -n mujoco_env python scripts/mode_discovery_bench.py RUN_DIR separability
    conda run -n mujoco_env python scripts/mode_discovery_bench.py RUN_DIR cluster --method spectral-normal
"""

import argparse
import glob
import json
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from peg_hole_model import BORE_TOP_Y, PEG_TIP_Y
from plot_contact_samples_cylindrical import subset

FAMILIES = ["in the bore", "face on mouth", "edge on mouth", "edge on inner rim"]


def family_of(mode):
    """The four contact-mode families of the stratum graph (stratum_spectral.py, k = 4)."""
    parts = mode.split(" + ")
    in_bore = [p for p in parts if any(s in p for s in ("wall", "floor", "tip_side", "shoulder"))]
    rim_in = [p for p in parts if p == "tip_edge/rim_in"]
    if rim_in and len(rim_in) + len(in_bore) == len(parts) and len(in_bore) <= 1:
        return 3  # the tip edge crossing the inner rim, alone or with one contact inside the bore
    if in_bore:
        return 0
    return 1 if "tip_face" in mode else 2


def reduced_frame(pose):
    """Signed reduced coords and the rotation taking hole-frame vectors into the reduced frame."""
    rot = Rotation.from_quat(pose[:, 3:], scalar_first=True)
    axis = rot.apply([0.0, 1.0, 0.0])  # towards the tip
    tip = pose[:, :3] * 1e3 + axis * PEG_TIP_Y
    phi = np.arctan2(tip[:, 2], tip[:, 0])
    to_reduced = Rotation.from_rotvec(np.c_[np.zeros_like(phi), phi, np.zeros_like(phi)])  # tip onto +x
    a = to_reduced.apply(axis)
    q = np.c_[np.hypot(tip[:, 0], tip[:, 2]), np.degrees(np.arctan2(a[:, 0], -a[:, 1])),
              np.degrees(np.arctan2(a[:, 2], -a[:, 1])), BORE_TOP_Y - tip[:, 1]]
    return q, rot, to_reduced, tip


def build(run_dir, work, n_points, k_geom, seed, sampling, snap=False):
    """Dataset and features, cached in RUN_DIR/discovery/dataset.npz."""
    work.mkdir(parents=True, exist_ok=True)
    cache = work / "dataset.npz"
    if cache.exists():
        return dict(np.load(cache, allow_pickle=True))
    t0 = time.time()
    q_all = np.load(Path(run_dir) / "reduced_coords.npy")
    rng = np.random.default_rng(seed)
    if sampling == "balanced":  # evenly over the reduced coordinates
        idx = subset(q_all, n_points, [0.25] * 4, [0.05, 0.25, 0.25, 0.5], seed)
    else:  # as sampled: dense where the sampler found thin contact modes
        idx = np.arange(len(q_all))
    idx = np.sort(rng.choice(idx, min(n_points, len(idx)), replace=False))
    files = sorted(glob.glob(str(Path(run_dir) / "shard_*.npz")))
    starts = np.r_[0, np.cumsum([np.load(f)["pose"].shape[0] for f in files])]
    cols = {k: [] for k in ("pose", "wrench", "twist", "contacts", "n_active", "traj", "step")}
    for i, f in enumerate(files):
        sel = idx[(idx >= starts[i]) & (idx < starts[i + 1])] - starts[i]
        if len(sel):
            z = np.load(f)
            for k in cols:
                cols[k].append(z[k][sel])
    d = {k: np.concatenate(v) for k, v in cols.items()}
    mode_ids = np.load(Path(run_dir) / "contact_mode_fine.npy")[idx]
    names = json.loads((Path(run_dir) / "contact_mode_fine.json").read_text())
    pose = d["pose"].astype(float)
    q, rot, to_reduced, tip = reduced_frame(pose)
    if snap:  # onto the smooth analytical contact manifold (the sampler's bore is a 64-gon)
        import peg_hole_analytic as A
        q_snap, ok = A.snap(q, 5e-3, 2e-2, 1e-8)
        moved = np.linalg.norm(q_snap - q, axis=1)
        ok &= moved < 0.05  # a snap is a small correction; a large move found some other contact state
        q = np.where(ok[:, None], q_snap, q)
        print(f"snapped {ok.mean():.2%} of the poses onto the smooth geometry (the rest kept); moved by p50 "
              f"{np.median(moved[ok]):.2e}, p99 {np.percentile(moved[ok], 99):.2e} (mm, deg)")

    # measured wrench in the reduced frame, torque about the tip
    F = rot.apply(d["wrench"][:, :3].astype(float))
    T = rot.apply(d["wrench"][:, 3:].astype(float)) + np.cross(pose[:, :3] - tip * 1e-3, F)
    wrench = np.c_[to_reduced.apply(F), to_reduced.apply(T) / 0.0125]
    twist = np.c_[to_reduced.apply(d["twist"][:, :3].astype(float)), to_reduced.apply(d["twist"][:, 3:].astype(float))]
    # friction-free contact wrench (privileged): 10 N shared over the simulator's active contacts
    c = d["contacts"].astype(float)
    n_use = np.minimum(d["n_active"], 4)
    share = np.where(np.arange(4)[None, :] < n_use[:, None], 10.0 / np.maximum(n_use, 1)[:, None], 0.0)
    Fc = (share[:, :, None] * c[:, :, 3:6]).sum(1)
    Tc = (share[:, :, None] * np.cross((c[:, :, :3] - tip[:, None, :]) * 1e-3, c[:, :, 3:6])).sum(1)
    contact = np.c_[to_reduced.apply(Fc), to_reduced.apply(Tc) / 0.0125]

    # local geometry of the sampled manifold (3D in the 4D reduced coords)
    _, nb = cKDTree(q).query(q, k=k_geom)
    X = q[nb] - q[:, None, :]
    cov = np.einsum("nki,nkj->nij", X, X) / k_geom
    lam, vec = np.linalg.eigh(cov)
    normal = vec[:, :, 0]
    variation = lam[:, 0] / lam.sum(1)
    E = vec[:, :, 1:]  # tangent basis
    t = np.einsum("nki,nij->nkj", X, E)
    h = np.einsum("nki,ni->nk", X, normal)
    iu = np.triu_indices(3)
    A = np.concatenate([0.5 * t[:, :, iu[0]] * t[:, :, iu[1]] * np.where(iu[0] == iu[1], 1.0, 2.0)[None, None, :],
                        t, np.ones(t.shape[:2] + (1,))], axis=2)
    coef = np.linalg.solve(np.einsum("nki,nkj->nij", A, A) + 1e-9 * np.eye(A.shape[2]),
                           np.einsum("nki,nk->ni", A, h)[..., None])[..., 0]
    H = np.zeros((len(q), 3, 3))
    H[:, iu[0], iu[1]] = coef[:, :6]
    H[:, iu[1], iu[0]] = coef[:, :6]
    curv = np.sort(np.linalg.eigvalsh(H), axis=1)
    resid = np.sqrt(np.mean((np.einsum("nki,ni->nk", A, coef) - h) ** 2, axis=1))
    iu4 = np.triu_indices(4)
    normal_feat = (normal[:, :, None] * normal[:, None, :])[:, iu4[0], iu4[1]]
    # sign-free curvatures: the normal's sign is arbitrary, so use |k| sorted
    curv_feat = np.sort(np.abs(curv), axis=1)

    data = dict(idx=idx, q=q, pose=pose, wrench=wrench, twist=twist, contact=contact, normal=normal, normal_feat=normal_feat,
                variation=variation, curv=curv_feat, fit_resid=resid, traj=d["traj"], step=d["step"],
                mode=mode_ids, mode_names=np.array(names), family=np.array([family_of(names[m]) for m in mode_ids]))
    np.savez(cache, **data)
    print(f"built {len(q)} samples with features ({time.time() - t0:.0f} s) -> {cache}")
    return data


FEATURE_SETS = {
    "pose": lambda d: d["q"],
    "pose+wrench": lambda d: np.c_[d["q"], d["wrench"]],
    "pose+normal": lambda d: np.c_[d["q"], d["normal_feat"], d["variation"]],
    "pose+normal+curvature": lambda d: np.c_[d["q"], d["normal_feat"], d["variation"], d["curv"], d["fit_resid"]],
    "pose+wrench+normal+curvature": lambda d: np.c_[d["q"], d["wrench"], d["normal_feat"], d["variation"], d["curv"],
                                                    d["fit_resid"]],
    "pose+contact (privileged)": lambda d: np.c_[d["q"], d["contact"]],
}


def separability(d, seed):
    """Upper bound: how well a supervised classifier on each feature set recovers the truth, with
    spatial-block cross-validation (blocks of 2 mm/deg in the reduced coords) so that near-duplicate
    neighbours do not leak between folds."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import balanced_accuracy_score, confusion_matrix
    from sklearn.model_selection import GroupKFold

    blocks = np.unique(np.floor(d["q"] / 2.0).astype(np.int64), axis=0, return_inverse=True)[1].ravel()
    modes, counts = np.unique(d["mode"], return_counts=True)
    common = np.isin(d["mode"], modes[counts >= 200])
    results = {}
    for name, f in FEATURE_SETS.items():
        X = f(d)
        row = {}
        for level, y, sel in (("family", d["family"], np.ones(len(X), bool)), ("mode", d["mode"], common)):
            pred = np.empty(sel.sum(), int)
            for tr, te in GroupKFold(5).split(X[sel], y[sel], blocks[sel]):
                clf = HistGradientBoostingClassifier(max_iter=200, random_state=seed).fit(X[sel][tr], y[sel][tr])
                pred[te] = clf.predict(X[sel][te])
            row[level] = float(balanced_accuracy_score(y[sel], pred))
            if level == "family":
                cm = confusion_matrix(y[sel], pred, labels=range(len(FAMILIES)), normalize="true")
                row["family_recall"] = {FAMILIES[i]: float(cm[i, i]) for i in range(len(FAMILIES))}
        results[name] = row
        print(f"{name:30s} family {row['family']:.3f}  mode {row['mode']:.3f}  | recall: "
              + ", ".join(f"{k} {v:.2f}" for k, v in row["family_recall"].items()), flush=True)
    return results


# --- frame cue -----------------------------------------------------------------------------------------

def frame_cue(run_dir, d, source="poses", k_pool=64, k_use=24, per_traj=2, lever=0.05):
    """Friction-free contact normal at each sample, and its alignment with the peg axis and the hole
    axis (cached in RUN_DIR/discovery/frame_cue_<source>.npz).

    The generalized contact normal (force, and moment about the tip) is the one direction in which the
    peg cannot move while keeping contact. It is found as the direction least supported by tangent
    vectors at the sample:
      steps  the walk steps of nearby samples (each is tangent to the contact manifold)
      poses  pose differences to nearby samples, which span the manifold however the walks moved
      both   the two together
    plus the two motions that are always free by symmetry: the peg's spin about its own axis and a
    rotation of everything about the hole axis. Neighbours come from different trajectories (at most
    per_traj each). Returns, per sample,
      a = |force . peg axis|   1 on a peg face, 0 on the peg side
      b = |force . hole axis|  1 on the hole's top face or floor, 0 on the bore wall
      two = s5 / s4, the second least-supported direction against the tangent ones (two contacts)
    """
    cache = Path(run_dir) / f"frame_cue_{source}.npz"
    if cache.exists():
        return dict(np.load(cache))
    if source == "full":
        out = frame_cue_full(d["pose"], lever=lever)
        np.savez(cache, **out)
        return out
    if source.startswith("quadric"):  # "quadric" or "quadric-shift<n>"
        shift = int(source.split("shift")[1]) if "shift" in source else 0
        out = frame_cue_quadric(d["q"], lever=lever, shift=shift)
        np.savez(cache, **out)
        return out
    q, tw, traj = d["q"], d["twist"], d["traj"]
    n = len(q)
    tr, tt = np.radians(q[:, 1]), np.radians(q[:, 2])
    u = np.c_[-np.tan(tr), np.ones(n), -np.tan(tt)]
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    c = np.c_[q[:, 0], BORE_TOP_Y - q[:, 3], np.zeros(n)] * 1e-3  # tip, m, reduced frame
    _, nb = cKDTree(q).query(q, k=k_pool)
    # at most per_traj neighbours from any one trajectory, nearest first
    tid = traj[nb]
    order = np.argsort(tid, axis=1, kind="stable")
    st = np.take_along_axis(tid, order, axis=1)
    pos = np.arange(k_pool)[None, :].repeat(n, 0)
    run_start = np.maximum.accumulate(np.where(np.c_[np.ones((n, 1), bool), st[:, 1:] != st[:, :-1]], pos, 0),
                                      axis=1)
    rank = np.empty_like(pos)
    np.put_along_axis(rank, order, pos - run_start, axis=1)
    pick = np.argsort(rank >= per_traj, axis=1, kind="stable")[:, :k_use]
    nb = np.take_along_axis(nb, pick, axis=1)
    tangents = []
    if source in ("steps", "both"):
        tangents.append(np.c_[tw[:, :3], tw[:, 3:] * lever][nb])
    if source in ("poses", "both"):
        dc = c[nb] - c[:, None, :]
        w = np.cross(u[:, None, :], u[nb])  # small rotation taking this axis to the neighbour's
        tangents.append(np.concatenate([dc, w * lever], axis=2))
    y = np.array([0.0, 1.0, 0.0])
    spin = np.c_[np.zeros((n, 3)), u * lever]
    about_hole = np.c_[np.cross(y, c), np.tile(y * lever, (n, 1))]
    M = np.concatenate(tangents + [spin[:, None, :], about_hole[:, None, :]], axis=1)
    M /= np.maximum(np.linalg.norm(M, axis=2, keepdims=True), 1e-12)
    _, s, vt = np.linalg.svd(M, full_matrices=False)
    f = vt[:, -1, :3]
    f /= np.maximum(np.linalg.norm(f, axis=1, keepdims=True), 1e-12)
    out = dict(a=np.abs((f * u).sum(1)), b=np.abs(f[:, 1]), two=s[:, -2] / s[:, -3])
    np.savez(cache, **out)
    return out


def axis_of(q):
    tr, tt = np.radians(q[..., 1]), np.radians(q[..., 2])
    u = np.stack([-np.tan(tr), np.ones_like(tr), -np.tan(tt)], axis=-1)
    return u / np.linalg.norm(u, axis=-1, keepdims=True)


def frame_cue_quadric(q, k=32, lever=0.05, h=1e-4, shift=0):
    """Contact normal from a second-order fit of the sampled manifold in the reduced coordinates,
    lifted exactly to force and moment.

    Local PCA gives an approximate normal; a quadratic fit of the heights along it (as in the curvature
    feature) adds the linear term that removes the secant bias, giving the tangent space at the sample
    itself. Its three tangent vectors are mapped to twists (tip velocity, angular velocity) through the
    reduced chart (rho and depth move the tip, the two tilts rotate the axis about the tip), the two
    symmetry motions are added, and the contact normal is their orthogonal complement.
    """
    n = len(q)
    _, nb = cKDTree(q).query(q, k=k)
    X = q[nb] - q[:, None, :]
    lam, vec = np.linalg.eigh(np.einsum("nki,nkj->nij", X, X) / k)
    normal, E = vec[:, :, 0], vec[:, :, 1:]
    t = np.einsum("nki,nij->nkj", X, E)
    hgt = np.einsum("nki,ni->nk", X, normal)
    iu = np.triu_indices(3)
    A = np.concatenate([0.5 * t[:, :, iu[0]] * t[:, :, iu[1]] * np.where(iu[0] == iu[1], 1.0, 2.0)[None, None, :],
                        t, np.ones(t.shape[:2] + (1,))], axis=2)
    coef = np.linalg.solve(np.einsum("nki,nkj->nij", A, A) + 1e-9 * np.eye(A.shape[2]),
                           np.einsum("nki,nk->ni", A, hgt)[..., None])[..., 0]
    g = coef[:, 6:9]  # slope of the heights at the sample: the PCA normal's tilt from the true one
    nq = normal - np.einsum("nij,nj->ni", E, g)
    nq /= np.linalg.norm(nq, axis=1, keepdims=True)
    if shift:
        # Neighbourhood shifting: near a crease (two contact sheets meeting, as at the jam) a centred
        # neighbourhood straddles both and its fit is a blend. Take the normal of the neighbour among
        # the nearest `shift` whose own fit has the smallest residual: a neighbourhood on one sheet.
        resid = np.sqrt(np.mean((np.einsum("nki,ni->nk", A, coef) - hgt) ** 2, axis=1))
        cand = nb[:, :shift]
        best = cand[np.arange(n), np.argmin(resid[cand], axis=1)]
        nq = nq[best] * np.sign((nq[best] * nq).sum(1, keepdims=True) + 1e-300)
    # tangent basis at the sample: orthogonal complement of the corrected normal
    P = np.eye(4)[None] - nq[:, :, None] * nq[:, None, :]
    tb = np.linalg.eigh(P)[1][:, :, 1:]  # eigenvalue 1 eigenvectors
    # chart Jacobian: reduced-coordinate rates -> twist (v [m], omega * lever [m])
    u = axis_of(q)
    J = np.zeros((n, 6, 4))
    J[:, 0, 0] = 1e-3
    J[:, 1, 3] = -1e-3
    for col in (1, 2):
        dq = np.zeros(4)
        dq[col] = h
        du = (axis_of(q + dq) - axis_of(q - dq)) / (2 * h)  # per degree
        J[:, 3:, col] = np.cross(u, du) * lever
    T = np.einsum("nij,njk->nik", J, tb).transpose(0, 2, 1)  # three tangent twists
    c = np.c_[q[:, 0], BORE_TOP_Y - q[:, 3], np.zeros(n)] * 1e-3
    y = np.array([0.0, 1.0, 0.0])
    spin = np.c_[np.zeros((n, 3)), u * lever]
    about_hole = np.c_[np.cross(y, c), np.tile(y * lever, (n, 1))]
    M = np.concatenate([T, spin[:, None, :], about_hole[:, None, :]], axis=1)
    M /= np.maximum(np.linalg.norm(M, axis=2, keepdims=True), 1e-12)
    vt = np.linalg.svd(M, full_matrices=True)[2]  # five tangents in six dimensions: the null vector is the sixth
    f = vt[:, -1, :3]
    f /= np.maximum(np.linalg.norm(f, axis=1, keepdims=True), 1e-12)
    return dict(a=np.abs((f * u).sum(1)), b=np.abs(f[:, 1]), two=lam[:, 1] / lam[:, 2])


def full_coords(pose):
    """Unreduced coords: tip x, tip z [mm], tilt rotation vector x, z [deg], depth [mm] (the sampler's)."""
    rot = Rotation.from_quat(pose[:, 3:], scalar_first=True)
    axis = rot.apply([0.0, 1.0, 0.0])  # towards the tip
    tip = pose[:, :3] * 1e3 + axis * PEG_TIP_Y
    down = np.array([0.0, -1.0, 0.0])
    cr = np.cross(down, axis)
    sn = np.linalg.norm(cr, axis=1, keepdims=True)
    tilt = cr / np.maximum(sn, 1e-12) * np.degrees(np.arctan2(sn, (axis * down).sum(1, keepdims=True)))
    return np.c_[tip[:, 0], tip[:, 2], tilt[:, 0], tilt[:, 2], BORE_TOP_Y - tip[:, 1]]


def frame_cue_full(pose, k=48, lever=0.05, h=1e-4):
    """As frame_cue_quadric, but in the unreduced coordinates, which are smooth everywhere (the
    reduced ones are polar about the bore axis, singular at rho = 0, where the bore states lie). The
    manifold is 4D in 5D; its four tangents plus the peg's spin leave the contact normal."""
    c5 = full_coords(pose)
    n = len(c5)

    def up_axis(c):  # up the peg, from the tilt rotation vector
        rv = np.radians(np.stack([c[..., 2], np.zeros_like(c[..., 2]), c[..., 3]], axis=-1))
        return -Rotation.from_rotvec(rv.reshape(-1, 3)).apply([0.0, -1.0, 0.0]).reshape(rv.shape)

    _, nb = cKDTree(c5).query(c5, k=k)
    X = c5[nb] - c5[:, None, :]
    vec = np.linalg.eigh(np.einsum("nki,nkj->nij", X, X) / k)[1]
    normal, E = vec[:, :, 0], vec[:, :, 1:]
    t = np.einsum("nki,nij->nkj", X, E)
    hgt = np.einsum("nki,ni->nk", X, normal)
    iu = np.triu_indices(4)
    A = np.concatenate([0.5 * t[:, :, iu[0]] * t[:, :, iu[1]] * np.where(iu[0] == iu[1], 1.0, 2.0)[None, None, :],
                        t, np.ones(t.shape[:2] + (1,))], axis=2)
    coef = np.linalg.solve(np.einsum("nki,nkj->nij", A, A) + 1e-9 * np.eye(A.shape[2]),
                           np.einsum("nki,nk->ni", A, hgt)[..., None])[..., 0]
    nq = normal - np.einsum("nij,nj->ni", E, coef[:, 10:14])
    nq /= np.linalg.norm(nq, axis=1, keepdims=True)
    P = np.eye(5)[None] - nq[:, :, None] * nq[:, None, :]
    tb = np.linalg.eigh(P)[1][:, :, 1:]  # four tangents
    u = up_axis(c5)
    J = np.zeros((n, 6, 5))
    J[:, 0, 0] = 1e-3
    J[:, 2, 1] = 1e-3
    J[:, 1, 4] = -1e-3
    for col in (2, 3):
        dc = np.zeros(5)
        dc[col] = h
        du = (up_axis(c5 + dc) - up_axis(c5 - dc)) / (2 * h)
        J[:, 3:, col] = np.cross(u, du) * lever
    T = np.einsum("nij,njk->nik", J, tb).transpose(0, 2, 1)
    spin = np.c_[np.zeros((n, 3)), u * lever]
    M = np.concatenate([T, spin[:, None, :]], axis=1)
    M /= np.maximum(np.linalg.norm(M, axis=2, keepdims=True), 1e-12)
    vt = np.linalg.svd(M, full_matrices=True)[2]
    f = vt[:, -1, :3]
    f /= np.maximum(np.linalg.norm(f, axis=1, keepdims=True), 1e-12)
    resid = np.sqrt(np.mean((np.einsum("nki,ni->nk", A, coef) - hgt) ** 2, axis=1))
    return dict(a=np.abs((f * u).sum(1)), b=np.abs(f[:, 1]), two=resid)


def frame_features(fc):
    eps = 1e-6
    return np.c_[np.log10(1 - fc["a"] + eps), np.log10(fc["a"] + eps), np.log10(1 - fc["b"] + eps),
                 np.log10(fc["b"] + eps), fc["two"]]


def frame_features_best(run_dir, d):
    """Each alignment indicator at its strongest across the step and pose estimators: noise and the
    secant bias of pose differences only ever move an alignment away from exactly 0 or 1."""
    X = [frame_features(frame_cue(run_dir, d, src)) for src in ("steps", "poses")]
    return np.c_[np.minimum(X[0][:, :4], X[1][:, :4]), X[0][:, 4]]


def type_regions(types, nb, min_size):
    """Connected regions of equal contact type over the kNN graph; regions under min_size join their
    neighbours' majority."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    n = len(types)
    i = np.repeat(np.arange(n), nb.shape[1])
    j = nb.ravel()
    same = types[i] == types[j]
    _, comp = connected_components(coo_matrix((np.ones(same.sum()), (i[same], j[same])), shape=(n, n)), directed=False)
    sizes = np.bincount(comp)
    labels = np.where(sizes[comp] >= min_size, comp, -1)
    _, labels[labels >= 0] = np.unique(labels[labels >= 0], return_inverse=True)
    for _ in range(100):
        todo = np.flatnonzero(labels < 0)
        if not len(todo):
            break
        votes = labels[nb[todo]]
        counts = np.zeros((len(todo), labels.max() + 1), int)
        rows = np.repeat(np.arange(len(todo)), nb.shape[1])
        ok = votes.ravel() >= 0
        np.add.at(counts, (rows[ok], votes.ravel()[ok]), 1)
        has = counts.sum(1) > 0
        labels[todo[has]] = counts[has].argmax(1)
    labels[labels < 0] = labels.max() + 1
    return labels


def bimodal_flag(x, seed):
    """Split one indicator at the gap of its two-component 1D Gaussian mixture (no labels)."""
    from sklearn.mixture import GaussianMixture
    g = GaussianMixture(2, random_state=seed).fit(x[:, None])
    low = int(np.argmin(g.means_.ravel()))
    return g.predict(x[:, None]) == low


# --- effective resistance / spectral sparsification ------------------------------------------------------

def effective_resistance(n, i, j, w, k_jl, seed, tol=1e-4, maxiter=1500):
    """Approximate effective resistances of the edges (i, j, w) of a connected graph
    (Spielman & Srivastava): R_ij = ||Z (e_i - e_j)||^2 with Z = Q W^1/2 B L^+, Q a k_jl x m random
    +-1/sqrt(k_jl) matrix, so k_jl Laplacian solves (conjugate gradients, Jacobi preconditioned)."""
    import scipy.sparse as sp
    from scipy.sparse.linalg import LinearOperator, cg

    m = len(w)
    B = sp.coo_matrix((np.r_[np.ones(m), -np.ones(m)], (np.r_[np.arange(m), np.arange(m)], np.r_[i, j])),
                      shape=(m, n)).tocsr()
    L = (B.T @ sp.diags(w) @ B).tocsr()
    L = L + 1e-10 * sp.identity(n)  # fix the constant null vector
    dinv = 1 / L.diagonal()
    pre = LinearOperator((n, n), matvec=lambda x: dinv * x)
    rng = np.random.default_rng(seed)
    Z = np.empty((k_jl, n))
    for r in range(k_jl):
        y = B.T @ (np.sqrt(w) * rng.choice([-1.0, 1.0], m)) / np.sqrt(k_jl)
        y -= y.mean()
        Z[r], _ = cg(L, y, rtol=tol, maxiter=maxiter, M=pre)
    return ((Z[:, i] - Z[:, j]) ** 2).sum(0)


def resistance_study(d, args):
    """Do high-leverage edges (weight x effective resistance, the edges spectral sparsification keeps)
    mark contact-mode boundaries? Two graphs: pose kNN, and pose kNN weighted by the frame cue."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from sklearn.metrics import adjusted_rand_score, roc_auc_score

    q = d["q"]
    i, j, dd = knn_graph(q, args.k_graph)
    pose_w = np.exp(-dd ** 2 / (2 * np.median(dd) ** 2))
    X = frame_features(frame_cue(args.work, d, "quadric"))[:, :4]
    dx = np.linalg.norm(X[i] - X[j], axis=1)
    graphs = {"pose": pose_w, "pose x frame cue": pose_w * np.exp(-dx ** 2 / (2 * np.median(dx) ** 2))}
    cross_mode = d["mode"][i] != d["mode"][j]
    cross_family = d["family"][i] != d["family"][j]
    names = d["mode_names"]
    out = {}
    for gname, w in graphs.items():
        t0 = time.time()
        keep = w > 1e-8
        n_comp, comp = connected_components(coo_matrix((w[keep], (i[keep], j[keep])), shape=(len(q),) * 2),
                                            directed=False)
        big = comp == np.argmax(np.bincount(comp))
        e = keep & big[i] & big[j]
        relabel = np.full(len(q), -1)
        relabel[big] = np.arange(big.sum())
        R = effective_resistance(big.sum(), relabel[i[e]], relabel[j[e]], w[e], args.k_jl, args.seed)
        lev = np.zeros(len(w))
        lev[e] = w[e] * R
        row = dict(components=int(n_comp), auc_cross_mode=float(roc_auc_score(cross_mode[e], lev[e])),
                   auc_cross_family=float(roc_auc_score(cross_family[e], lev[e])))
        # leverage by mode (edges inside one mode): lower-dimensional strata should stand out
        inside = e & ~cross_mode
        by_mode = {}
        for m in np.argsort(-np.bincount(d["mode"]))[:8]:
            sel = inside & (d["mode"][i] == m)
            by_mode[str(names[m])] = float(np.median(lev[sel])) if sel.any() else None
        row["median_leverage_by_mode"] = by_mode
        # cut the highest-leverage edges: do the pieces left follow the families?
        cuts = {}
        for pct in (90, 95, 98):
            thr = np.percentile(lev[e], pct)
            ok = e & (lev <= thr)
            _, cc = connected_components(coo_matrix((np.ones(ok.sum()), (i[ok], j[ok])), shape=(len(q),) * 2),
                                         directed=False)
            sizes = np.bincount(cc)
            labels = np.where(sizes[cc] >= args.min_size, cc, -1)
            _, labels[labels >= 0] = np.unique(labels[labels >= 0], return_inverse=True)
            labels[labels < 0] = labels.max() + 1
            cuts[f"cut top {100 - pct}%"] = dict(pieces=int(labels.max()), family_ari=float(
                adjusted_rand_score(d["family"], labels)))
        row["cuts"] = cuts
        out[gname] = row
        print(f"\n[{gname}] {n_comp} components; leverage separates cross-mode edges with AUC "
              f"{row['auc_cross_mode']:.3f}, cross-family AUC {row['auc_cross_family']:.3f} ({time.time() - t0:.0f} s)")
        print("  median leverage of edges inside each mode: " + "; ".join(
            f"{k[:28]} {v:.3f}" for k, v in by_mode.items() if v is not None))
        print("  cutting the highest-leverage edges: " + "; ".join(
            f"{k}: {v['pieces']} pieces, family ARI {v['family_ari']:.3f}" for k, v in cuts.items()))
    return out


# --- unsupervised methods ------------------------------------------------------------------------------

def knn_graph(X, k):
    """Symmetric kNN pairs (i < j) and their distances."""
    dist, nb = cKDTree(X).query(X, k=k + 1)
    i = np.repeat(np.arange(len(X)), k)
    j = nb[:, 1:].ravel()
    dd = dist[:, 1:].ravel()
    lo, hi = np.minimum(i, j), np.maximum(i, j)
    pairs, first = np.unique(np.c_[lo, hi], axis=0, return_index=True)
    return pairs[:, 0], pairs[:, 1], dd[first]


def spectral_labels(n, i, j, w, k_max, k=None, eig_tol=0.0, ncv=None):
    """Normalized spectral clustering (Ng-Jordan-Weiss) on the largest connected component; other
    points get -1. k from the largest eigengap among the first k_max unless given. eig_tol and ncv go to
    ARPACK: a looser tolerance and a wider Krylov space converge where many eigenvalues crowd near zero."""
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components
    from scipy.sparse.linalg import eigsh
    from sklearn.cluster import KMeans

    live = w > 1e-8  # links weighted down to nothing are cuts, not edges
    i, j, w = i[live], j[live], w[live]
    W = sp.coo_matrix((np.r_[w, w], (np.r_[i, j], np.r_[j, i])), shape=(n, n)).tocsr()
    _, comp = connected_components(W, directed=False)
    big = comp == np.argmax(np.bincount(comp))
    Wl = W[big][:, big]
    deg = np.asarray(Wl.sum(1)).ravel()
    Dm = sp.diags(1 / np.sqrt(deg))
    L = (sp.identity(Wl.shape[0]) - Dm @ Wl @ Dm).tocsc()
    vals, vecs = eigsh(L, k=k_max + 1, sigma=-1e-3, which="LM", tol=eig_tol, ncv=ncv)
    o = np.argsort(vals)
    vals, vecs = vals[o], vecs[:, o]
    k = k or int(np.argmax(np.diff(vals)[1:k_max]) + 2)
    U = vecs[:, :k] / np.linalg.norm(vecs[:, :k], axis=1, keepdims=True)
    labels = np.full(n, -1)
    labels[big] = KMeans(k, n_init=10, random_state=0).fit_predict(U)
    return labels, vals


def smooth_labels(labels, nb, iters):
    """Majority vote over each point's neighbours, repeated: spatially coherent regions."""
    for _ in range(iters):
        votes = labels[nb]
        n_lab = labels.max() + 1
        counts = np.zeros((len(labels), n_lab), int)
        np.add.at(counts, (np.repeat(np.arange(len(labels)), nb.shape[1]), votes.ravel()), 1)
        labels = counts.argmax(1)
    return labels


def region_growing(d, nb, angle_deg, variation_pct, min_size):
    """Grow regions over the kNN graph across neighbours whose normals agree within angle_deg, from
    points whose surface variation is below its variation_pct percentile; the rest (edges, creases)
    and regions under min_size join their neighbours' majority."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    n = len(d["q"])
    flat = d["variation"] < np.percentile(d["variation"], variation_pct)
    i = np.repeat(np.arange(n), nb.shape[1])
    j = nb.ravel()
    cos = np.abs((d["normal"][i] * d["normal"][j]).sum(1))
    ok = flat[i] & flat[j] & (cos > np.cos(np.radians(angle_deg)))
    _, comp = connected_components(coo_matrix((np.ones(ok.sum()), (i[ok], j[ok])), shape=(n, n)), directed=False)
    sizes = np.bincount(comp)
    big = sizes[comp] >= min_size
    labels = np.where(big & flat, comp, -1)
    _, labels[labels >= 0] = np.unique(labels[labels >= 0], return_inverse=True)
    for _ in range(50):  # unassigned points take their neighbours' majority
        todo = labels < 0
        if not todo.any():
            break
        votes = labels[nb[todo]]
        for r, row in zip(np.flatnonzero(todo), votes):
            row = row[row >= 0]
            if len(row):
                labels[r] = np.bincount(row).argmax()
    labels[labels < 0] = labels.max() + 1
    return labels


def evaluate(d, labels):
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    out = {}
    for level, y in (("family", d["family"]), ("mode", d["mode"])):
        cont = np.zeros((labels.max() + 1, y.max() + 1))
        np.add.at(cont, (labels, y), 1)
        out[level] = dict(ari=float(adjusted_rand_score(y, labels)), nmi=float(normalized_mutual_info_score(y, labels)),
                          purity=float(cont.max(1).sum() / len(y)))
    # each family's best cluster: share of the family in it, and that cluster's share of the family
    cont = np.zeros((labels.max() + 1, len(FAMILIES)))
    np.add.at(cont, (labels, d["family"]), 1)
    out["family_match"] = {}
    for f, name in enumerate(FAMILIES):
        c = int(np.argmax(cont[:, f] / np.maximum(cont.sum(1), 1) * (cont[:, f] > 0.05 * cont[:, f].sum())))
        out["family_match"][name] = dict(recall=float(cont[c, f] / cont[:, f].sum()),
                                         precision=float(cont[c, f] / cont[c].sum()))
    out["clusters"] = int(labels.max() + 1)
    return out


def patch_merge(d, nb, args):
    """Over-segment the poses into small patches, describe each by the median frame-cue signature of
    its samples, and merge touching patches with the most similar signatures (Ward linkage restricted
    to patch adjacency) down to k regions. Medians over a patch suppress the noise of single samples."""
    from scipy.sparse import coo_matrix
    from sklearn.cluster import AgglomerativeClustering, MiniBatchKMeans
    from sklearn.preprocessing import StandardScaler

    q = d["q"]
    X = frame_features(frame_cue(args.work, d, args.frame_source))[:, :4]
    patch = MiniBatchKMeans(args.patches, random_state=args.seed, batch_size=20000, n_init=3).fit_predict(q)
    order = np.argsort(patch)
    bounds = np.r_[0, np.cumsum(np.bincount(patch, minlength=args.patches))]
    P = np.array([np.median(X[order[bounds[p]:bounds[p + 1]]], axis=0) for p in range(args.patches)])
    i = np.repeat(np.arange(len(q)), nb.shape[1])
    j = nb.ravel()
    touch = patch[i] != patch[j]
    adj = coo_matrix((np.ones(touch.sum()), (patch[i][touch], patch[j][touch])), shape=(args.patches,) * 2)
    adj = ((adj + adj.T) > 0).astype(float)
    labels = AgglomerativeClustering(args.k or 6, connectivity=adj, linkage="ward").fit_predict(
        StandardScaler().fit_transform(P))
    return labels[patch]


def cluster(d, method, args):
    """Labels for every point of the dataset by one of the unsupervised methods."""
    from sklearn.neighbors import KNeighborsClassifier

    q = d["q"]
    nb = cKDTree(q).query(q, k=args.k_graph + 1)[1][:, 1:]
    if method.startswith("spectral"):
        rng = np.random.default_rng(args.seed)
        sub = np.sort(rng.choice(len(q), min(args.spectral_points, len(q)), replace=False))
        i, j, dd = knn_graph(q[sub], args.k_graph)
        w = np.exp(-dd ** 2 / (2 * np.median(dd) ** 2))
        if "normal" in method:
            dn = 1 - np.abs((d["normal"][sub][i] * d["normal"][sub][j]).sum(1))
            w = w * np.exp(-dn / np.median(dn) / args.normal_scale)
        if "curv" in method:
            c = np.log(d["curv"][sub] + 1e-6)
            dc = np.linalg.norm(c[i] - c[j], axis=1)
            w = w * np.exp(-dc ** 2 / (2 * (args.curv_scale * np.median(dc)) ** 2))
        if "wrench" in method:
            dw = np.linalg.norm(d["wrench"][sub][i] - d["wrench"][sub][j], axis=1)
            w = w * np.exp(-dw ** 2 / (2 * (args.wrench_scale * np.median(dw)) ** 2))
        lab_sub, vals = spectral_labels(len(sub), i, j, w, args.max_k, args.k)
        print("  eigenvalues: " + " ".join(f"{v:.4f}" for v in vals[:args.max_k + 1]))
        keep = lab_sub >= 0
        feats = q[sub][keep]
        return KNeighborsClassifier(1).fit(feats, lab_sub[keep]).predict(q)
    if method == "region-growing":
        return region_growing(d, nb, args.angle, args.variation_pct, args.min_size)
    if method == "patch-merge":
        return patch_merge(d, nb, args)
    if method.startswith("frame"):
        X = (frame_features_best(args.work, d) if args.frame_source == "best"
             else frame_features(frame_cue(args.work, d, args.frame_source)))
        if method == "frame-regions":
            # an alignment is exact below the noise floor (--exact); with --exact 0 each indicator is
            # split at the gap of its own distribution instead
            flags = (X[:, :4] < np.log10(args.exact) if args.exact > 0
                     else np.stack([bimodal_flag(X[:, c], args.seed) for c in range(4)], axis=1))
            types = (flags * (1 << np.arange(4))).sum(1)
            return type_regions(types, nb, args.min_size)
        if method == "frame-flags":
            # each indicator's low mode: peg face, peg side, hole face, bore wall; then two contacts
            flags = np.stack([bimodal_flag(X[:, c], args.seed) for c in range(4)], axis=1)
            two = ~bimodal_flag(-X[:, 4], args.seed)  # high s5/s4: a second constrained direction
            code = (flags * (1 << np.arange(4))).sum(1) + 16 * two
            _, lab = np.unique(code, return_inverse=True)
        else:
            from sklearn.mixture import GaussianMixture
            from sklearn.preprocessing import StandardScaler
            Z = StandardScaler().fit_transform(X)
            if args.k:
                gm = GaussianMixture(args.k, random_state=args.seed).fit(Z)
            else:  # BIC picks the number of components
                fits = [GaussianMixture(c, random_state=args.seed).fit(Z) for c in range(2, args.max_k + 1)]
                gm = min(fits, key=lambda g: g.bic(Z))
            lab = gm.predict(Z)
        return smooth_labels(lab.ravel(), nb, args.smooth_iters)
    if method in ("gmm-smooth", "hdbscan"):
        from sklearn.preprocessing import StandardScaler
        F = StandardScaler().fit_transform(np.c_[d["normal_feat"], np.log(d["curv"] + 1e-6), np.log(d["variation"])])
        if method == "gmm-smooth":
            from sklearn.mixture import GaussianMixture
            lab = GaussianMixture(args.k or 6, covariance_type="full", random_state=args.seed).fit_predict(F)
            return smooth_labels(lab, nb, args.smooth_iters)
        from sklearn.cluster import HDBSCAN
        rng = np.random.default_rng(args.seed)
        sub = np.sort(rng.choice(len(q), min(args.spectral_points, len(q)), replace=False))
        Z = np.c_[q / q.std(0), F][sub]
        lab = HDBSCAN(min_cluster_size=args.min_size).fit_predict(Z)
        keep = lab >= 0
        return KNeighborsClassifier(1).fit(Z[keep], lab[keep]).predict(np.c_[q / q.std(0), F])
    raise ValueError(method)


METHODS = ["spectral-pose", "spectral-normal", "spectral-normal-curv", "spectral-wrench", "region-growing",
           "gmm-smooth", "hdbscan", "frame-flags", "frame-gmm", "frame-regions", "patch-merge"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("task", choices=["build", "separability", "cluster", "resistance"])
    parser.add_argument("--method", nargs="+", default=METHODS, choices=METHODS)
    parser.add_argument("--k", type=int, help="number of clusters (default: method's own choice)")
    parser.add_argument("--max-k", type=int, default=10)
    parser.add_argument("--k-graph", type=int, default=15, help="neighbours in the clustering graph")
    parser.add_argument("--spectral-points", type=int, default=40_000)
    parser.add_argument("--normal-scale", type=float, default=1.0)
    parser.add_argument("--curv-scale", type=float, default=1.0)
    parser.add_argument("--wrench-scale", type=float, default=1.0)
    parser.add_argument("--angle", type=float, default=10.0, help="region growing: normal agreement [deg]")
    parser.add_argument("--variation-pct", type=float, default=80.0, help="region growing: seeds below this percentile")
    parser.add_argument("--min-size", type=int, default=500)
    parser.add_argument("--k-jl", type=int, default=32, help="random projections for effective resistances")
    parser.add_argument("--patches", type=int, default=3000, help="patch-merge: number of initial patches")
    parser.add_argument("--smooth-iters", type=int, default=5)
    parser.add_argument("--exact", type=float, default=1e-4,
                        help="frame-regions: an alignment counts as exact below this (0: split by mixture)")
    parser.add_argument("--frame-source", default="best",
                        help="tangent vectors for the frame cue's contact normal")
    parser.add_argument("--points", type=int, default=200_000)
    parser.add_argument("--snap", action="store_true",
                        help="move the poses onto the smooth analytical contact manifold first")
    parser.add_argument("--sampling", choices=["balanced", "random"], default="balanced",
                        help="benchmark subset: even over the reduced coordinates, or at the sampler's density")
    parser.add_argument("--k-geom", type=int, default=48, help="neighbours for the local geometry")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.work = args.run_dir / "discovery" / f"{args.sampling}{args.points}{'_snap' if args.snap else ''}"
    d = build(args.run_dir, args.work, args.points, args.k_geom, args.seed, args.sampling, args.snap)
    fam = np.bincount(d["family"], minlength=len(FAMILIES)) / len(d["family"])
    print(f"{len(d['q'])} samples; families: " + ", ".join(f"{n} {s:.1%}" for n, s in zip(FAMILIES, fam)))
    if args.task == "separability":
        res = separability(d, args.seed)
        (args.work / "separability.json").write_text(json.dumps(res, indent=1))
    if args.task == "resistance":
        res = resistance_study(d, args)
        (args.work / "resistance.json").write_text(json.dumps(res, indent=1))
    if args.task == "cluster":
        path = args.work / "cluster_results.json"
        allres = json.loads(path.read_text()) if path.exists() else {}
        for method in args.method:
            t0 = time.time()
            labels = cluster(d, method, args)
            r = evaluate(d, labels)
            tag = method + (f" k={args.k}" if args.k else "")
            allres[tag] = r
            np.save(args.work / f"labels_{tag.replace(' ', '_')}.npy", labels)
            fm = ", ".join(f"{n} {v['recall']:.2f}/{v['precision']:.2f}" for n, v in r["family_match"].items())
            print(f"{tag:26s} {r['clusters']:3d} clusters | family ARI {r['family']['ari']:.3f} NMI "
                  f"{r['family']['nmi']:.3f} purity {r['family']['purity']:.1%} | mode ARI {r['mode']['ari']:.3f} | "
                  f"best cluster recall/precision: {fm} ({time.time() - t0:.0f} s)", flush=True)
        path.write_text(json.dumps(allres, indent=1))


if __name__ == "__main__":
    main()
