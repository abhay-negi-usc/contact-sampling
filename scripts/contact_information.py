"""How much contact samples say about the eyelet pose, and the uncertainty they leave (hook/eyelet).

Measurement model: a contact sample is a hook pose T in the robot's frame at which the hook touches the eyelet, so
the signed hook-eyelet separation s(G^-1 T) is ~0 at the true eyelet pose G (a little negative: the soft contact
penetrates; `offset_mm`). A correction xi of an eyelet pose G (x, y, z mm along G's axes, then a rotation vector
deg: G' = G * (Exp(rot), trans), as estimator_bench.pose_error measures errors) changes each sample's separation by
J_i xi to first order: J_i (1 x 6) is the contact normal and its moment about the eyelet origin (central differences
of hook_eyelet_model.Separation). With independent errors of std sigma the samples' Fisher information about xi is
I = sum J_i^T J_i / sigma^2. Together with the belief error's prior (uniform within +-bound per axis, so std
bound / sqrt(3)) the Laplace posterior covariance is (I + P0)^-1.

Metrics, in prior-whitened coordinates (W = S I S, S the prior stds per axis):
  eigenvalues of W     data-to-prior information along each direction; > 1: the data dominate the prior there;
  their eigenvectors   which pose directions the contacts constrain (and which they leave to the prior);
  information gain     0.5 sum log(1 + lambda), nats: the Gaussian mutual information between data and pose;
  posterior std        per axis, sqrt(diag((I + P0)^-1)).
Poses as 6-vectors: position mm, rotation vector deg, in the robot's frame.
"""

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

import hook_eyelet_model as H

AXES = ("x", "y", "z", "rx", "ry", "rz")
STEP = np.array([0.25, 0.25, 0.25, 0.25, 0.25, 0.25])  # central-difference steps, mm and deg


def compose(g, xi):
    """Eyelet pose g (6) corrected by xi (6, or (k, 6)) in g's own frame: (k, 6)."""
    xi = np.atleast_2d(xi)
    r = Rotation.from_rotvec(np.deg2rad(g[3:]))
    pos = g[:3] + r.apply(xi[:, :3])
    rot = (r * Rotation.from_rotvec(np.deg2rad(xi[:, 3:]))).as_rotvec(degrees=True)
    return np.concatenate([pos, rot], axis=1)


def relative(g, g_ref):
    """Pose g in g_ref's frame (6): the error convention of estimator_bench.pose_error with g the truth."""
    r_ref = Rotation.from_rotvec(np.deg2rad(g_ref[3:]))
    return np.concatenate([r_ref.inv().apply(g[:3] - g_ref[:3]),
                           (r_ref.inv() * Rotation.from_rotvec(np.deg2rad(g[3:]))).as_rotvec(degrees=True)])


class ContactModel:
    """Signed separations of recorded hook poses against an eyelet at a given pose, and their Jacobians."""

    def __init__(self, distmax=0.01):
        self.m = H.build_model()
        self.d = mujoco.MjData(self.m)
        self.sep = H.Separation(self.m)
        self.distmax = distmax

    def separations(self, g, poses):
        """Separation (mm) of each hook pose (k, 7: mm, wxyz in the robot's frame) from an eyelet at g (6)."""
        r = Rotation.from_rotvec(np.deg2rad(g[3:]))
        pos = r.inv().apply(poses[:, :3] - g[:3]) * 1e-3
        quat = (r.inv() * Rotation.from_quat(poses[:, 3:], scalar_first=True)).as_quat(scalar_first=True)
        out = np.empty(len(poses))
        for k in range(len(poses)):
            H.set_hook_pose(self.m, self.d, pos[k], quat[k])
            out[k] = self.sep(self.m, self.d, distmax=self.distmax)[0] * 1e3
        return out

    def jacobian(self, g, poses):
        """(separations (k,) mm at g, Jacobian (k, 6) mm per mm / deg of a correction of g)."""
        s0 = self.separations(g, poses)
        J = np.empty((len(poses), 6))
        for a in range(6):
            step = np.zeros(6)
            step[a] = STEP[a]
            J[:, a] = (self.separations(compose(g, step)[0], poses)
                       - self.separations(compose(g, -step)[0], poses)) / (2 * STEP[a])
        return s0, J


def prior_std(bound):
    """Stds of a uniform belief error within +-bound per axis (6, or one value for all)."""
    return np.broadcast_to(np.asarray(bound, dtype=np.float64), (6,)) / np.sqrt(3.0)


def information_metrics(J, sigma, s_prior, free=None):
    """Metrics of the samples' information (module docstring) for Jacobian rows J (k, 6), per-sample error std
    sigma (mm; scalar or (k,)) and prior stds s_prior (6). free: estimated axes (default all); the rest are known
    (no information needed, zero posterior std). Returns a dict of numpy values."""
    free = np.ones(6, bool) if free is None else np.asarray(free, bool)
    w = np.broadcast_to(np.asarray(sigma, dtype=np.float64), (len(J),))
    info = (J / w[:, None]).T @ (J / w[:, None])
    Jf = info[np.ix_(free, free)]
    S = np.diag(s_prior[free])
    lam, vec = np.linalg.eigh(S @ Jf @ S)
    lam = np.clip(lam, 0.0, None)
    post = np.zeros((6, 6))
    post[np.ix_(free, free)] = S @ np.linalg.inv(S @ Jf @ S + np.eye(free.sum())) @ S
    return {"info": info, "lambda": lam, "vectors": vec, "gain_nats": 0.5 * np.log1p(lam).sum(),
            "n_constrained": int((lam > 1.0).sum()), "post_cov": post, "post_std": np.sqrt(np.diag(post))}


def solve(model, poses, start, sigma, bound, offset_mm=0.0, free=None, iters=15, lm=1e-2, tol=1e-4):
    """A posterior mode of the eyelet pose (6) by Levenberg-Marquardt from start (6): minimises the contact cost
    sum ((s_i + offset) / sigma)^2 over the free axes, the prior flat within +-bound of the initial belief (a
    barrier outside it). Returns (pose, cost, Jacobian at the pose)."""
    free = np.ones(6, bool) if free is None else np.asarray(free, bool)
    bound = np.broadcast_to(np.asarray(bound, dtype=np.float64), (6,))
    g = np.asarray(start, dtype=np.float64).copy()

    def cost_of(g, s):
        return np.sum(((s + offset_mm) / sigma) ** 2) + np.sum((np.maximum(np.abs(g) - bound, 0) / 0.5) ** 2)

    s, J = model.jacobian(g, poses)
    cost = cost_of(g, s)
    for _ in range(iters):
        Jw = J[:, free] / sigma
        A, b = Jw.T @ Jw, Jw.T @ ((s + offset_mm) / sigma)
        ridge = 1e-9 * max(np.trace(A), 1.0) * np.eye(len(A))
        while True:
            step = np.zeros(6)
            step[free] = -np.linalg.solve(A + lm * np.diag(np.diag(A)) + ridge, b)
            g_new = compose(g, step)[0]
            cost_new = cost_of(g_new, model.separations(g_new, poses))
            if cost_new < cost or lm > 1e6:
                break
            lm *= 10.0
        if cost_new >= cost:
            break
        improved = cost - cost_new
        g, cost, lm = g_new, cost_new, max(lm / 10.0, 1e-6)
        s, J = model.jacobian(g, poses)
        if improved < tol * cost:
            break
    return g, cost, J


def distinct(g, kept, sep_mm, sep_deg):
    """Whether pose g lies farther than sep_mm or sep_deg from every pose in kept."""
    for h in kept:
        e = relative(g, h)
        if np.linalg.norm(e[:3]) < sep_mm and np.linalg.norm(e[3:]) < sep_deg:
            return False
    return True


def find_modes(model, poses, sigma, bound, rng, starts=24, screen=40, keep=4, sep_mm=2.0, sep_deg=3.0,
               offset_mm=0.0, free=None):
    """Distinct posterior modes, lowest cost first: Levenberg-Marquardt from the initial belief and starts - 1
    uniform draws within +-bound on screen evenly spaced samples, then the keep best distinct ones refined on all
    samples. Returns a list of (pose, cost, Jacobian)."""
    sub = poses[np.unique(np.linspace(0, len(poses) - 1, min(screen, len(poses))).astype(int))]
    inits = [np.zeros(6)] + list(rng.uniform(-bound, bound, (starts - 1, 6)))
    screened = sorted((solve(model, sub, g0, sigma, bound, offset_mm, free, iters=10)[:2] for g0 in inits),
                      key=lambda x: x[1])
    picked = []
    for g, _ in screened:
        if distinct(g, [p for p, *_ in picked], sep_mm, sep_deg):
            picked.append((g, None))
        if len(picked) == keep:
            break
    modes = []
    for g, _ in picked:
        g, cost, J = solve(model, poses, g, sigma, bound, offset_mm, free)
        if distinct(g, [m[0] for m in modes], sep_mm, sep_deg):
            modes.append((g, cost, J))
    return sorted(modes, key=lambda m: m[1])


def laplace_mixture(modes, sigma, s_prior, free=None, temperature=1.0):
    """Laplace approximation of the posterior over the modes: per mode (weight, covariance (6, 6) of a correction
    of its pose). Weight ~ exp(-cost / (2 T)) sqrt(det cov) (flat prior; the Gaussian prior P0 only regularises
    directions without information); temperature T > 1 discounts the evidence when sigma is too small."""
    free = np.ones(6, bool) if free is None else np.asarray(free, bool)
    out, logw = [], []
    for g, cost, J in modes:
        m = information_metrics(J, sigma * np.sqrt(temperature), s_prior, free)
        cov = m["post_cov"]
        sign, logdet = np.linalg.slogdet(cov[np.ix_(free, free)])
        logw.append(-cost / (2 * temperature) + 0.5 * logdet)
        out.append(cov)
    logw = np.array(logw)
    w = np.exp(logw - logw.max())
    return w / w.sum(), out


# ---------------------------------------------------------------- further information and uncertainty measures
# Inverse-problem, registration and Bayesian measures beyond the Gauss-Newton Fisher information above, each a
# function of quantities at a mode: residuals r (k,) and Jacobian J (k, 6) of the samples, the cost Hessian, the
# Laplace mixture over modes.


def rodgers(lam):
    """Rodgers' (Inverse Methods for Atmospheric Sounding, 2000) measures of prior-whitened eigenvalues lam:
    Shannon information content 0.5 sum log2(1 + lam) bits, and degrees of freedom for signal sum lam / (1 + lam),
    the number of independent pose directions the data (rather than the prior) determine, 0 to 6."""
    lam = np.clip(np.asarray(lam, dtype=np.float64), 0.0, None)
    return {"info_bits": 0.5 * np.log2(1.0 + lam).sum(), "dfs": (lam / (1.0 + lam)).sum()}


def cost_hessian(model, g, poses, sigma, offset_mm=0.0, h=STEP):
    """Finite-difference Hessian (6, 6) of the contact cost sum ((s_i + offset) / sigma)^2 over a correction of g;
    half of it is the observed Fisher information (Efron & Hinkley 1978), second-order terms included."""
    def cost(xi):
        return np.sum(((model.separations(compose(g, xi)[0], poses) + offset_mm) / sigma) ** 2)

    c0 = cost(np.zeros(6))
    Hm = np.zeros((6, 6))
    for a in range(6):
        ea = np.zeros(6)
        ea[a] = h[a]
        Hm[a, a] = (cost(ea) - 2 * c0 + cost(-ea)) / h[a] ** 2
        for b in range(a + 1, 6):
            eb = np.zeros(6)
            eb[b] = h[b]
            Hm[a, b] = Hm[b, a] = (cost(ea + eb) - cost(ea - eb) - cost(-ea + eb) + cost(-ea - eb)) / (4 * h[a] * h[b])
    return Hm


def observed_cov(Hm, s_prior):
    """Posterior covariance from the observed information 0.5 Hm plus the prior's; directions of non-positive
    curvature (a saddle or a flat valley) are given the prior's information only. Returns (cov, n_nonpositive)."""
    A = 0.25 * (Hm + Hm.T)  # half the symmetrised Hessian
    w, V = np.linalg.eigh(A)
    bad = int((w <= 0).sum())
    A = V @ np.diag(np.clip(w, 0.0, None)) @ V.T + np.diag(1.0 / s_prior ** 2)
    return np.linalg.inv(A), bad


def sandwich_cov(J, r, sigma, s_prior, clusters=None):
    """Robust (Huber-White) posterior covariance A^-1 (B + P0) A^-1: A = J^T J / sigma^2 + P0, B the empirical
    variance of the score sum_i J_i r_i / sigma^2, its terms summed within clusters first (correlated samples, e.g.
    one policy step's); with a correct noise model B = J^T J / sigma^2 and it equals the Gauss-Newton posterior.
    The data-driven counterpart of Censi's (2007) ICP covariance."""
    P0 = np.diag(1.0 / s_prior ** 2)
    A = J.T @ J / sigma ** 2 + P0
    g = J * (r / sigma ** 2)[:, None]
    if clusters is not None:
        g = np.array([g[clusters == c].sum(0) for c in np.unique(clusters)])
    n, k = len(g), 6
    B = g.T @ g * (n / max(n - k, 1))
    Ai = np.linalg.inv(A)
    return Ai @ (B + P0) @ Ai


def gaussian_entropy(cov):
    """Differential entropy (nats) of a Gaussian with covariance cov."""
    sign, logdet = np.linalg.slogdet(2 * np.pi * np.e * cov)
    return 0.5 * logdet


def uniform_entropy(bound):
    """Differential entropy (nats) of the uniform belief error within +-bound per axis (mm and deg)."""
    return float(np.sum(np.log(2.0 * np.broadcast_to(np.asarray(bound, dtype=np.float64), (6,)))))


def _log_gauss(x, mean, cov):
    """Log density of N(mean, cov) at x (6) or at each row of x (n, 6)."""
    d = np.atleast_2d(x) - mean
    sign, logdet = np.linalg.slogdet(2 * np.pi * cov)
    out = -0.5 * (logdet + np.einsum("ni,ni->n", d, np.linalg.solve(cov, d.T).T))
    return out if np.ndim(x) == 2 else out[0]


def mixture_logpdf(x, weights, means, covs):
    """Log density of a Gaussian mixture at x (6) or at each row of x (n, 6)."""
    parts = [np.log(w) + _log_gauss(x, mu, c) for w, mu, c in zip(weights, means, covs) if w > 0]
    return np.logaddexp.reduce(np.array(parts), axis=0)


def mixture_entropies(weights, means, covs):
    """Entropies (nats) of a Gaussian mixture (means as corrections of a common pose, (m, 6)):
    mode            Shannon entropy of the weights (exp of it: the effective number of competing modes);
    upper           sum w_k H_k + H(w) (Huber et al. 2008);
    pairwise_lower  Kolchinsky & Tracey (2017) lower bound with Bhattacharyya distances;
    renyi2          collision entropy -log sum_ij w_i w_j N(mu_i; mu_j, S_i + S_j), exact for mixtures."""
    w = np.asarray(weights, dtype=np.float64)
    nz = w > 0
    H_k = np.array([gaussian_entropy(c) for c in covs])
    mode = float(-(w[nz] * np.log(w[nz])).sum())
    m = len(w)
    bh = np.zeros((m, m))
    lg = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            S = 0.5 * (covs[i] + covs[j])
            d = means[i] - means[j]
            bh[i, j] = (d @ np.linalg.solve(S, d) / 8 + 0.5 * (np.linalg.slogdet(S)[1]
                        - 0.5 * (np.linalg.slogdet(covs[i])[1] + np.linalg.slogdet(covs[j])[1])))
            lg[i, j] = _log_gauss(means[i], means[j], covs[i] + covs[j])
    lower = float((w * H_k).sum() - (w[nz] * np.log((np.exp(-bh) @ w)[nz])).sum())
    renyi2 = float(-np.log(np.sum(np.outer(w, w) * np.exp(lg))))
    return {"mode": mode, "upper": float((w * H_k).sum() + mode), "pairwise_lower": lower, "renyi2": renyi2}


def hpd_level(x, weights, means, covs, rng, n=4000):
    """Highest-posterior-density level of point x under a Gaussian mixture: the posterior mass of points denser
    than x (Monte Carlo). Uniform on [0, 1] over cases for a calibrated posterior; > 0.95 = outside the 95% region."""
    w = np.asarray(weights, dtype=np.float64)
    k = rng.choice(len(w), size=n, p=w / w.sum())
    pts = np.concatenate([rng.multivariate_normal(means[j], covs[j], size=(k == j).sum()) for j in range(len(w))])
    return float(np.mean(mixture_logpdf(pts, w, means, covs) > mixture_logpdf(x, w, means, covs)))


def profile_interval(model, poses, g, sigma, bound, axis, grid, offset_mm=0.0):
    """Profile likelihood (Raue et al. 2009) along one axis of a correction of mode g: for each value v in grid the
    cost minimised over the other five axes with this one held at v. Returns (grid, profile cost - cost at g); the
    95% interval is where it stays below 3.84 (chi2_1), a flat profile = practically non-identifiable axis."""
    free = np.ones(6, bool)
    free[axis] = False
    c0 = solve(model, poses, g, sigma, bound, offset_mm, iters=0)[1]
    out = []
    for v in grid:
        xi = np.zeros(6)
        xi[axis] = v
        out.append(solve(model, poses, compose(g, xi)[0], sigma, bound, offset_mm, free=free, iters=8)[1] - c0)
    return np.asarray(grid), np.array(out)


def mount_offset(poses, c):
    """Hook poses (k, 7: mm, wxyz) with the hook's mounting off by c (x, y, z mm, rotation vector deg, in the hook
    frame): each pose T becomes T * C, the same error for every sample."""
    rot = Rotation.from_quat(poses[:, 3:], scalar_first=True)
    quat = (rot * Rotation.from_rotvec(np.deg2rad(c[3:]))).as_quat(scalar_first=True)
    return np.concatenate([poses[:, :3] + rot.apply(c[:3]), quat], axis=1)


def mount_jacobian(model, g, poses):
    """Jacobian (k, 6) of the samples' separations at eyelet pose g with respect to a mounting error c of the hook
    (mount_offset), by central differences."""
    J = np.empty((len(poses), 6))
    for a in range(6):
        c = np.zeros(6)
        c[a] = STEP[a]
        J[:, a] = (model.separations(g, mount_offset(poses, c)) - model.separations(g, mount_offset(poses, -c))) / (2 * STEP[a])
    return J


def consider_cov(J, Jc, sigma, s_prior, mount_std):
    """Consider covariance (Tapley, Schutz & Born 2004; Brossard et al. 2020's bias term) of a correction of the
    mode: the Gauss-Newton posterior plus S P_c S^T, S = A^-1 J^T Jc / sigma^2 the sensitivity of the estimate to an
    unestimated mounting error c of std mount_std (mm, deg) per axis, A = J^T J / sigma^2 + P0."""
    A = J.T @ J / sigma ** 2 + np.diag(1.0 / s_prior ** 2)
    Ai = np.linalg.inv(A)
    S = Ai @ J.T @ Jc / sigma ** 2
    return Ai + S @ np.diag(np.broadcast_to(np.asarray(mount_std, dtype=np.float64), (6,)) ** 2) @ S.T


def region_mass(weights, means, covs, tol, rng, n=4000):
    """Posterior mass within tol (mm, deg; norms of the translation and rotation parts) of the first mode's pose:
    the probability that it is correct to the task's tolerance (a decision-region test, cf. Javdani et al. 2014)."""
    w = np.asarray(weights, dtype=np.float64)
    k = rng.choice(len(w), size=n, p=w / w.sum())
    pts = np.concatenate([rng.multivariate_normal(means[j], covs[j], size=(k == j).sum()) for j in range(len(w))])
    d = pts - means[0]
    return float(np.mean((np.linalg.norm(d[:, :3], axis=1) <= tol[0]) & (np.linalg.norm(d[:, 3:], axis=1) <= tol[1])))
