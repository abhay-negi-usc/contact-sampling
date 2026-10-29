"""Particle estimators of the eyelet pose from contact samples (hook/eyelet), on the GPU separation model.

The posterior of the eyelet pose g (6: x, y, z mm, rotation vector deg, in the robot's frame) given contact samples
(hook poses at contact) is p(g | D) ~ prior(g) prod_i N(s_i(g) + offset; 0, sigma^2), s_i the signed hook-eyelet
separation of sample i (contact_gpu.GpuContactModel), the prior uniform within +-bound per axis (the belief error's
support). With sigma ~0.05-0.3 mm against a +-20 mm / deg prior the posterior fills ~1e-10 of the prior's volume
and lies along the samples' contact manifold: a conventional particle filter starves (Koval et al. 2015), and a
local search from a few starts lands in a wrong basin (info_metric_bench: 24-start Levenberg-Marquardt correct in
1-4% of cases). The estimators:

batch (all samples at once)
  smc            adaptive tempered SMC (Del Moral, Doucet & Jasra 2006; Jasra et al. 2011): from the prior through
                 p_beta ~ prior L^beta, beta 0 -> 1 chosen to keep the effective sample size at ess_frac N;
                 resample, then random-walk Metropolis moves with the particles' covariance at per-particle scales
                 (multi-scale, for multimodal clouds)
  scaling        Petrovskaya & Khatib's scaling series (IJRR 2011): delta-neighbourhoods shrinking by zoom per
                 stage (volume halves), M uniform samples in each kept particle's neighbourhood, likelihood
                 annealed as L^(1 / tau), tau = (delta / delta_final)^2, pruned by weighted resampling
  multistart     Gauss-Newton (Levenberg-Marquardt) from many prior draws at once (the batch analogue of
                 Koval's projection onto the contact manifold from uniform initialisations)
  smc_sequence   the online version: SMC through the posteriors of successive checkpoints (IBIS / resample-move,
                 Chopin 2002), particles carried over, each bridged to the next by tempering
sequential (policy step by policy step, the samples of each step one observation; Koval et al. / Klingensmith,
Koval et al. ICRA 2017)
  cpf            conventional particle filter: static state with roughening noise, likelihood weights
  mpf            manifold particle filter: dual particles drawn on the step's contact manifold by projection
                 (Gauss-Newton on the step's squared separations) from uniform prior draws ("uniform"), the
                 previous particles ("particle") or uniform draws in balls about them ("ball"), weighted by a kernel
                 density estimate of the previous belief; phi of the particles dual, the rest conventional
                 (mixture proposal, Thrun et al. 2001 / Koval et al. 2015)
Every estimator returns particles (N, 6), their log weights and log likelihoods. polish() refines the best distinct
particles by Gauss-Newton (point estimate). Particles live in the (position, rotation vector) chart: within the
prior's +-20 deg its distortion is small.
"""

import math

import numpy as np
import torch

import contact_gpu as G

AXES = ("x", "y", "z", "rx", "ry", "rz")


class Problem:
    """The samples, noise model and prior of one estimation."""

    def __init__(self, model, poses, sigma, bound, offset=0.0, robust=None, offset_tau=None, debias=None):
        self.model = model
        self.dev = model.device
        self.pts = model.hook_points(poses)
        self.k = len(poses)
        self.sigma, self.offset, self.robust = float(sigma), float(offset), robust
        self.offset_tau = offset_tau  # std (mm) of a common separation offset marginalised out (None: none)
        self.bound = torch.as_tensor(np.broadcast_to(np.asarray(bound, np.float64), (6,)).copy(),
                                     dtype=torch.float32, device=self.dev)
        self.evals = 0  # particle-sample separations computed
        self.debias = None
        if debias is not None:  # (hook-pose noise std mm / deg, draws M, seed)
            self.debias = int(debias[1])
            self.pts_noisy = noise_copies(self.pts, torch.as_tensor(poses[:, :3], dtype=torch.float32,
                                                                    device=self.dev), *debias)

    def subset(self, idx):
        """The same problem on samples idx only."""
        p = Problem.__new__(Problem)
        p.__dict__.update(self.__dict__)
        p.pts, p.k = self.pts[idx], len(idx)
        if self.debias:
            p.pts_noisy = self.pts_noisy.view(self.k, self.debias, *self.pts.shape[1:])[idx].flatten(0, 1)
        return p

    def plain(self):
        """The same problem without the min-bias correction."""
        p = Problem.__new__(Problem)
        p.__dict__.update(self.__dict__)
        p.debias = None
        return p

    def residuals(self, g):
        """Whitened residuals (N, k). With offset_tau, a common offset delta ~ N(0, tau^2) of every separation
        (the hook seated against two or more points reads the least of their noisy distances: biased low) is
        integrated out: sum r^2 - (sum r)^2 / (k + (sigma / tau)^2) = |r - a sum(r)|^2."""
        self.evals += len(g) * self.k
        sep = self.model.fast_separations(g, self.pts)
        if self.debias:
            sep = sep - self.min_bias(g, sep)
        r = (sep + self.offset) / self.sigma
        if self.offset_tau:
            lam = (self.sigma / self.offset_tau) ** 2
            a = (1 - math.sqrt(lam / (self.k + lam))) / self.k
            r = r - a * r.sum(-1, keepdim=True)
        return r

    def min_bias(self, g, sep, step=0.05):
        """Expected separation (N, k) the hook-pose noise adds to a sample read at contact: the hook seated against
        several points reads the least of their noisy distances, biased low (-0.1 mm at 0.3 mm / deg in the
        seated phase), which a pose shift then absorbs. Each hook is first moved onto the contact manifold of
        g along its separation's gradient (translation, forward differences), then the mean separation of its M
        fixed noisy copies (common random numbers) is the bias; ~zero against a single face."""
        eye = torch.eye(3, device=self.dev) * step
        grad = torch.stack([self.model.fast_separations(g, self.pts, shift=eye[a].expand(1, 1, 3)) - sep
                            for a in range(3)], -1) / step
        shift = -(sep / (grad ** 2).sum(-1).clamp(min=0.25))[..., None] * grad
        m = self.debias
        self.evals += len(g) * self.k * (3 + m)
        noisy = self.model.fast_separations(g, self.pts_noisy, shift=shift.repeat_interleave(m, 1))
        return noisy.view(len(g), self.k, m).mean(-1)

    def loglik_of(self, r):
        if self.robust is None:
            return -0.5 * (r.double() ** 2).sum(-1)
        nu = self.robust  # Student-t with nu degrees of freedom
        return -0.5 * (nu + 1) * torch.log1p(r.double() ** 2 / nu).sum(-1)

    def loglik(self, g):
        return self.loglik_of(self.residuals(g))

    def inside(self, g):
        return torch.all(g.abs() <= self.bound, dim=-1)

    def prior(self, n, gen):
        return (torch.rand(n, 6, generator=gen, device=self.dev) * 2 - 1) * self.bound


def noise_copies(pts, origin, std, m, seed):
    """M copies (k * M, n, 3) of hook points pts (k, n, 3) under fixed draws of the bench's hook-pose noise
    (info_metric_bench.noisy: position N(0, std) mm, a turn N(0, std) deg per axis about the hook's origin)."""
    gen = torch.Generator(device=pts.device).manual_seed(int(seed))
    k = len(pts)
    turn = torch.randn(k, m, 3, generator=gen, device=pts.device) * math.radians(std)
    move = torch.randn(k, m, 3, generator=gen, device=pts.device) * std
    R = G.rotvec_to_matrix(turn.reshape(-1, 3)).reshape(k, m, 3, 3)
    rel = pts - origin[:, None, :]
    out = torch.einsum("kmab,knb->kmna", R, rel) + (origin[:, None, :] + move)[:, :, None, :]
    return out.reshape(k * m, -1, 3)


def generator(seed, dev="cuda"):
    gen = torch.Generator(device=dev)
    gen.manual_seed(int(seed))
    return gen


def normalise(logw):
    w = torch.exp(logw - logw.max())
    return w / w.sum()


def ess(logw):
    w = normalise(logw)
    return float(1.0 / (w ** 2).sum())


def resample(logw, gen, n=None):
    """Systematic resampling: indices (n,)."""
    w = normalise(logw)
    n = len(w) if n is None else n
    u = (torch.rand(1, generator=gen, device=w.device, dtype=w.dtype) + torch.arange(n, device=w.device)) / n
    c = torch.cumsum(w, 0)
    c[-1] = 1.0
    return torch.searchsorted(c, u).clamp(max=len(w) - 1)


def weighted_cov(x, w):
    m = (w[:, None] * x).sum(0)
    d = x - m
    return (w[:, None] * d).T @ d


def rw_moves(prob, x, ll, beta, gen, steps, scales=(1.0, 0.3, 0.1, 0.03, 0.01)):
    """Random-walk Metropolis on prior * L^beta, proposal covariance the particles' times 2.38^2 / 6 times a
    per-particle scale drawn from `scales`. Returns (x, ll, acceptance rate)."""
    cov = weighted_cov(x.double(), torch.full((len(x),), 1.0 / len(x), dtype=torch.float64, device=x.device))
    L = torch.linalg.cholesky(cov + 1e-6 * torch.eye(6, dtype=torch.float64, device=x.device)).float()
    sc = torch.as_tensor(scales, dtype=torch.float32, device=x.device)
    acc = 0.0
    for _ in range(steps):
        s = sc[torch.randint(len(sc), (len(x),), generator=gen, device=x.device)] * (2.38 / math.sqrt(6))
        prop = x + s[:, None] * (torch.randn(x.shape, generator=gen, device=x.device) @ L.T)
        ok = prob.inside(prop)
        llp = torch.full_like(ll, -math.inf)
        if ok.any():
            llp[ok] = prob.loglik(prop[ok])
        a = torch.log(torch.rand(len(x), generator=gen, device=x.device, dtype=torch.float64)) < beta * (llp - ll)
        x = torch.where(a[:, None], prop, x)
        ll = torch.where(a, llp, ll)
        acc += float(a.float().mean()) / steps
    return x, ll, acc


def smc(prob, n=4096, seed=0, ess_frac=0.5, moves=5, final_moves=10, max_stages=400, snapshots=(), keep=512):
    """Adaptive tempered SMC (module docstring). The schedule also stops at each beta of `snapshots` and records
    `keep` of its particles there: the power posteriors prior * L^beta, i.e. the posterior with sigma / sqrt(beta).
    Returns dict(x, logw, ll, stages, betas, snapshots {beta: (x, ll)})."""
    gen = generator(seed)
    x = prob.prior(n, gen)
    ll = prob.loglik(x)
    beta, betas, snaps = 0.0, [], {}
    targets = sorted(b for b in snapshots if 0 < b < 1)
    for _ in range(max_stages):
        if beta >= 1.0:
            break
        lo, hi = 0.0, 1.0 - beta
        if ess(hi * ll) < ess_frac * n:
            for _ in range(50):  # bisection on the increment
                mid = 0.5 * (lo + hi)
                lo, hi = (mid, hi) if ess(mid * ll) >= ess_frac * n else (lo, mid)
            delta = max(lo, 1e-12)
        else:
            delta = hi
        if targets and beta + delta >= targets[0]:
            delta = targets[0] - beta
        beta = 1.0 if beta + delta >= 1.0 - 1e-12 else beta + delta
        idx = resample(delta * ll, gen)
        x, ll = x[idx], ll[idx]
        x, ll, _ = rw_moves(prob, x, ll, beta, gen, final_moves if beta >= 1.0 else moves)
        betas.append(beta)
        if targets and abs(beta - targets[0]) < 1e-12:
            snaps[targets.pop(0)] = (x[:keep].cpu().numpy().astype(np.float32), ll[:keep].cpu().numpy())
    return {"x": x, "logw": torch.zeros(n, dtype=torch.float64, device=x.device), "ll": ll, "stages": len(betas),
            "betas": betas, "snapshots": snaps}


def smc_sequence(probs, n=4096, seed=0, ess_frac=0.5, moves=5, final_moves=10, max_stages=400):
    """SMC through a sequence of posteriors (one per checkpoint: probs, each with all samples so far): the prior to
    the first by likelihood tempering, then each to the next by the geometric bridge
    p^(1 - g) p_next^g, g 0 -> 1 adaptively (ESS), with resampling and random-walk moves (Chopin's IBIS /
    resample-move for a static parameter, Del Moral, Doucet & Jasra 2006). Particles carry over between
    checkpoints. Returns a list of dict(x, logw, ll, stages) per checkpoint."""
    gen = generator(seed)
    x = probs[0].prior(n, gen)
    ll_prev = torch.zeros(n, dtype=torch.float64, device=x.device)  # the prior: no likelihood yet
    outs = []
    for j, prob in enumerate(probs):
        ll_new = prob.loglik(x)
        prev_prob = probs[j - 1] if j else None
        g, stages = 0.0, 0
        while g < 1.0 and stages < max_stages:
            inc = ll_new - ll_prev
            lo, hi = 0.0, 1.0 - g
            if ess(hi * inc) < ess_frac * n:
                for _ in range(50):
                    mid = 0.5 * (lo + hi)
                    lo, hi = (mid, hi) if ess(mid * inc) >= ess_frac * n else (lo, mid)
                d = max(lo, 1e-12)
            else:
                d = hi
            g = 1.0 if g + d >= 1.0 - 1e-12 else g + d
            idx = resample(d * inc, gen)
            x, ll_new, ll_prev = x[idx], ll_new[idx], ll_prev[idx]
            x, ll_new, ll_prev = bridge_moves(prob, prev_prob, x, ll_new, ll_prev, g, gen,
                                              final_moves if g >= 1.0 else moves)
            stages += 1
        outs.append({"x": x.clone(), "logw": torch.zeros(n, dtype=torch.float64, device=x.device),
                     "ll": ll_new.clone(), "stages": stages})
        ll_prev = ll_new
    return outs


def bridge_moves(prob, prev_prob, x, ll_new, ll_prev, g, gen, steps, scales=(1.0, 0.3, 0.1, 0.03, 0.01)):
    """Random-walk Metropolis on prior * L_prev^(1 - g) * L_new^g (L_prev = 1 for the first checkpoint)."""
    cov = weighted_cov(x.double(), torch.full((len(x),), 1.0 / len(x), dtype=torch.float64, device=x.device))
    L = torch.linalg.cholesky(cov + 1e-6 * torch.eye(6, dtype=torch.float64, device=x.device)).float()
    sc = torch.as_tensor(scales, dtype=torch.float32, device=x.device)
    for _ in range(steps):
        s = sc[torch.randint(len(sc), (len(x),), generator=gen, device=x.device)] * (2.38 / math.sqrt(6))
        prop = x + s[:, None] * (torch.randn(x.shape, generator=gen, device=x.device) @ L.T)
        ok = prob.inside(prop)
        pn = torch.full_like(ll_new, -math.inf)
        pp = torch.zeros_like(ll_prev)
        if ok.any():
            pn[ok] = prob.loglik(prop[ok])
            if prev_prob is not None:
                pp[ok] = prev_prob.loglik(prop[ok])
        if prev_prob is None:
            log_a = g * (pn - ll_new)
        else:
            log_a = (1 - g) * (pp - ll_prev) + g * (pn - ll_new)
        a = torch.log(torch.rand(len(x), generator=gen, device=x.device, dtype=torch.float64)) < log_a
        x = torch.where(a[:, None], prop, x)
        ll_new = torch.where(a, pn, ll_new)
        ll_prev = torch.where(a, pp, ll_prev)
    return x, ll_new, ll_prev


def scaling_series(prob, n=4096, per=8, seed=0, delta_final=None, max_stages=200):
    """Petrovskaya & Khatib's scaling series (module docstring), with delta in the chart's units (mm = deg).
    Returns dict(x, logw, ll, stages)."""
    gen = generator(seed)
    dim = 6
    zoom = 2.0 ** (-1.0 / dim)
    delta_final = prob.sigma if delta_final is None else delta_final
    delta = float(prob.bound.max())
    x = prob.prior(n, gen)
    ll = prob.loglik(x)
    stages = 0
    while stages < max_stages:
        stages += 1
        delta *= zoom
        tau = max(1.0, (delta / delta_final) ** 2)
        keep = resample(ll / tau, gen, n // per)  # prune: weighted resampling to n / per centres
        centres = x[keep]
        x = (centres[:, None, :] + (torch.rand(len(centres), per, dim, generator=gen, device=x.device) * 2 - 1)
             * delta).reshape(-1, dim)
        x = torch.maximum(torch.minimum(x, prob.bound), -prob.bound)
        ll = prob.loglik(x)
        if tau <= 1.0:
            break
    return {"x": x, "logw": ll.clone(), "ll": ll, "stages": stages}


def jacobian(prob, x, h=0.1):
    """(whitened residuals (N, k), Jacobian (N, k, 6)) by central differences in the chart."""
    N = len(x)
    steps = torch.eye(6, device=x.device) * h
    probe = torch.cat([x, (x[:, None, :] + steps).reshape(-1, 6), (x[:, None, :] - steps).reshape(-1, 6)])
    r = prob.residuals(probe)
    r0 = r[:N]
    rp = r[N:7 * N].reshape(N, 6, -1)
    rm = r[7 * N:].reshape(N, 6, -1)
    return r0, ((rp - rm) / (2 * h)).transpose(1, 2)


def gauss_newton(prob, x, iters=20, lam=1e-2, tol=1e-5):
    """Levenberg-Marquardt of every particle at once on the whitened squared residuals, clamped to the prior box.
    Returns (x, cost = sum r^2 (N,), J (N, k, 6))."""
    lam = torch.full((len(x),), lam, device=x.device)
    r, J = jacobian(prob, x)
    cost = (r.double() ** 2).sum(-1)
    active = torch.ones(len(x), dtype=torch.bool, device=x.device)
    for _ in range(iters):
        A = J.transpose(1, 2) @ J
        b = (J.transpose(1, 2) @ r[..., None])[..., 0]
        diag = torch.diagonal(A, dim1=1, dim2=2)
        M = A + (lam[:, None] * diag + 1e-6 * diag.mean(-1, keepdim=True) + 1e-9)[..., None] * torch.eye(6, device=x.device)
        step = -torch.linalg.solve(M, b)
        trial = torch.maximum(torch.minimum(x + step, prob.bound), -prob.bound)
        ct = (prob.residuals(trial).double() ** 2).sum(-1)
        better = (ct < cost) & active
        improvement = torch.where(better, cost - ct, torch.zeros_like(cost))
        x = torch.where(better[:, None], trial, x)
        lam = torch.where(better, (lam / 10).clamp(min=1e-6), lam * 10)
        active = active & ~(better & (improvement < tol * cost)) & (lam < 1e6)
        if not active.any():
            break
        if better.any():
            rb, Jb = jacobian(prob, x[better])
            r, J = r.clone(), J.clone()
            r[better], J[better] = rb, Jb
            cost = torch.where(better, ct, cost)
    return x, cost, J


def multistart(prob, n=1024, seed=0, iters=25):
    """Gauss-Newton from n prior draws (and the initial belief). Returns dict(x, logw, ll, cost)."""
    gen = generator(seed)
    x0 = prob.prior(n, gen)
    x0[0] = 0.0
    x, cost, _ = gauss_newton(prob, x0, iters=iters)
    ll = -0.5 * cost
    return {"x": x, "logw": ll.clone(), "ll": ll, "cost": cost}


def distinct_best(x, score, k, sep_mm=1.0, sep_deg=1.5):
    """Indices of up to k highest-scoring particles at least sep_mm or sep_deg apart (chart distance)."""
    order = torch.argsort(score, descending=True)
    keep = []
    xs = x[order]
    for i in range(len(order)):
        if len(keep) == k:
            break
        if keep:
            d = (xs[i] - x[torch.stack(keep)]).abs()
            if torch.any((d[:, :3].norm(dim=-1) < sep_mm) & (d[:, 3:].norm(dim=-1) < sep_deg)):
                continue
        keep.append(order[i])
    return torch.stack(keep)


def polish(prob, out, k=16, iters=20):
    """Gauss-Newton from the k best distinct particles (by log likelihood). Returns (best pose (6,), its cost,
    all polished poses (k', 6), their costs)."""
    idx = distinct_best(out["x"], out["ll"], k)
    x, cost, _ = gauss_newton(prob, out["x"][idx], iters=iters)
    b = int(torch.argmin(cost))
    return x[b], float(cost[b]), x, cost


# ------------------------------------------------------------------------------------------------ sequential


def kde_logpdf(x, centres, logw, rough=0.0, bw_scale=1.0):
    """Log of a Gaussian kernel density estimate of the propagated previous belief: weighted centres, kernel the
    transition's (roughening noise rough, so exact for it) plus Scott's smoothing of their covariance."""
    w = normalise(logw)
    neff = 1.0 / float((w ** 2).sum())
    cov = weighted_cov(centres.double(), w) * (bw_scale * neff ** (-1.0 / 10.0)) ** 2  # Scott: n^(-1/(d+4))
    eye = torch.eye(6, dtype=torch.float64, device=x.device)
    L = torch.linalg.cholesky(cov + (rough ** 2 + 1e-6) * eye)
    Li = torch.linalg.inv(L)
    zc = (centres.double() @ Li.T).float()  # whitened: kernel distances are Euclidean
    zx = (x.double() @ Li.T).float()
    lw = torch.log(w.clamp(min=1e-300)).float()
    out = torch.cat([torch.logsumexp(lw - 0.5 * torch.cdist(zx[a:a + 1024], zc) ** 2, dim=1)
                     for a in range(0, len(x), 1024)]).double()
    return out - torch.log(torch.diagonal(L)).sum() - 3 * math.log(2 * math.pi)


def project(prob, x0, iters=8, eps=None):
    """Projection of x0 onto the problem's contact manifold (Gauss-Newton on the squared separations; Koval et
    al.'s Eq. 3). Returns (x, accepted mask: mean squared whitened residual below eps)."""
    x, cost, _ = gauss_newton(prob, x0, iters=iters)
    eps = 4.0 if eps is None else eps
    return x, cost / prob.k < eps


def sequential(model, steps, sigma, bound, kind="cpf", n=2048, phi=1.0, proposal="ball", seed=0, rough=0.1,
               ball=None, checkpoints=(), max_tries=3, bw_scale=1.0):
    """Sequential filter (module docstring) over observations steps = [(policy step, poses (m, 7))] in order.
    At each checkpoint policy step, records the particles. Returns {step: dict(x, logw)}."""
    gen = generator(seed)
    dummy = Problem(model, np.zeros((1, 7)) + [0, 0, 0, 1, 0, 0, 0], sigma, bound)
    x = dummy.prior(n, gen)
    logw = torch.zeros(n, dtype=torch.float64, device=x.device)
    ball = 2.0 * rough if ball is None else ball
    out, cps = {}, sorted(checkpoints)
    for step, poses in steps:
        while cps and cps[0] < step:  # record before this step's update
            out[cps.pop(0)] = {"x": x.clone(), "logw": logw.clone()}
        if not len(poses):
            continue
        prob = Problem(model, poses, sigma, bound)
        # propagate (static state, roughening) and resample when degenerate
        if ess(logw) < 0.5 * n:
            idx = resample(logw, gen)
            x, logw = x[idx], torch.zeros(n, dtype=torch.float64, device=x.device)
        prev_x, prev_logw = x, logw
        xp = x + rough * torch.randn(x.shape, generator=gen, device=x.device)
        xp = torch.maximum(torch.minimum(xp, prob.bound), -prob.bound)
        n_dual = int(round(phi * n)) if kind == "mpf" else 0
        parts = []
        if n_dual < n:  # conventional: likelihood weights
            sel = torch.arange(n_dual, n, device=x.device)
            lw = logw[sel] + prob.loglik(xp[sel])
            parts.append((xp[sel], lw, (n - n_dual) / n))
        if n_dual:  # dual: on the manifold, KDE weights against the propagated previous belief
            got = []
            for _ in range(max_tries):
                need = n_dual - sum(len(g) for g in got)
                if need <= 0:
                    break
                if proposal == "uniform":
                    x0 = prob.prior(need, gen)
                else:
                    i0 = resample(prev_logw, gen, need)
                    x0 = prev_x[i0]
                    if proposal == "ball":
                        x0 = x0 + (torch.rand(x0.shape, generator=gen, device=x.device) * 2 - 1) * ball
                xm, ok = project(prob, torch.maximum(torch.minimum(x0, prob.bound), -prob.bound))
                got.append(xm[ok])
            xd = torch.cat(got)[:n_dual] if got else torch.zeros(0, 6, device=x.device)
            if len(xd):
                lw = kde_logpdf(xd, prev_x, prev_logw, rough, bw_scale)
                parts.append((xd, lw, len(xd) / n))
        if not parts:  # no projection reached the manifold: the conventional update alone
            parts.append((xp, logw + prob.loglik(xp), 1.0))
        # mixture: each part normalised, weighted by its share
        xs, lws = [], []
        tot = sum(p[2] for p in parts)
        for xi, lw, share in parts:
            lws.append(torch.log(normalise(lw).clamp(min=1e-300)) + math.log(share / tot))
            xs.append(xi)
        x = torch.cat(xs)
        logw = torch.cat(lws)
        if len(x) < n:  # dual projections failed: refill from the conventional part
            idx = resample(logw, gen, n)
            x, logw = x[idx], torch.zeros(n, dtype=torch.float64, device=x.device)
    for c in cps:
        out[c] = {"x": x.clone(), "logw": logw.clone()}
    return out
