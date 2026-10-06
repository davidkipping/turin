"""Collapsed limb darkening: integrate the quadratic law out of the LinEph fit.

``--ld=collapsed`` samples ``(dP, dtau0, k, beta, T14)`` from the **marginal**
posterior, with the limb darkening integrated out of the log-density, and
then draws ``(q1, q2)`` for each kept sample from its **conditional**
``p(q | theta, D)``. Marginal target plus conditional draws of the
integrated block is collapsed Gibbs sampling. It reproduces the default
posterior, ``q1`` and ``q2`` included, with two fewer sampled dimensions --
and those two are the most wall-prone of turin's bounded parameters.

Two things it is **not**, both measured by SquishierPlanet (who proposed and
validated the scheme; ``docs/upstream/turin_collapsed_ld_*`` in that repo)
and rejected, so do not "simplify" toward either:

- *profiling* the limb darkening -- sampling ``L(x*) + prior`` with omega
  set to its maximiser at each theta. Biased k, b and T14 by up to 0.38 sigma
  and inflated k's width by 46% on KOI-518.02;
- *non-collapsed* Gibbs -- alternating theta | omega and omega | theta
  without marginalising. Correct, but 5x slower, since q1 and T14 are
  correlated (and it would need anvil >= 0.4.0, the target then changing
  between segments).

How. Every physical quadratic law's normalised light curve is exactly a
convex combination ``sum_j omega_j F_j`` of the three vertex laws of the
Kipping (2013) triangle (:func:`turin.model.vertex_flux_devs`), with omega
on the 2-simplex -- which is precisely the q-box. Write ``x = (omega1,
omega2)``. Uniform ``(q1, q2)`` induces ``log p(x) = -3 log sum_j
omega_j / F*_j``. The deviation is linear in ``x``; with turin's
multiplicative baseline, ``(x, c)`` is bilinear. For each ``x`` the baseline
coefficients ``c`` are profiled exactly as turin always does -- that is the
only profiling here -- giving ``L(x)``, whose gradient and Gauss-Newton
Hessian in ``x`` come from the envelope theorem (the omega columns with the
baseline projected out, a Schur complement). That theorem needs ``c`` to be
the *flux-space* chi-squared's exact minimiser, so this needs
``--PL=exact``: under ``ratio`` the formulas would be wrong, not merely
untested.

The marginal is then ``log int_triangle exp(L(x)) p(x) dx`` under the
quadratic model of ``L`` about an expansion point ``x*`` (one Newton step
from the centroid plus an exact QP projection onto the triangle): ``L(x*) +
1/2 g H^-1 g + log Z``, with ``Z`` the Gaussian mass in the triangle
weighted by the prior, by 20 x 20 Gauss-Legendre in whitened coordinates.
Nothing is maximised in the sampled value: ``L(x*)`` and ``1/2 g H^-1 g``
together are just the quadratic model's peak, which ``log Z`` integrates
around.
"""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np

from . import model as _model
from . import profile as _profile
from .likelihood import BYTES_PER_POINT, ProfiledTransitLogProb

#: Per-(chain, point) bytes for block sizing in this mode. Measured 271 B
#: against the default path's 121 B (a fresh-process MLX probe of one
#: compiled value+grad at 512 chains on KOI-518.02), i.e. 2.24x; 2.5x keeps
#: a margin. Three vertex light curves and their projected columns are live
#: where the default path holds one.
COLLAPSED_BYTES_PER_POINT = int(2.5 * BYTES_PER_POINT)
#: Newton+projection passes before the final evaluation at the expansion
#: point. One ("two solve passes") against three differed by at most 6e-9
#: in log-density over 2,000 posterior draws in float64; zero passes reached
#: 5e-4, at the float32 floor, so it is not offered.
N_NEWTON = 1
#: Gauss-Legendre nodes, outer (per piece) and inner (per node), for the
#: triangle integral. 12 x 12 reached 0.02 in the tails, 8 x 8 reached 0.25.
N_GL_OUTER = 20
N_GL_INNER = 20
#: Half-width, in whitened units, beyond which the Gaussian is clipped.
Z_WINDOW = 8.0
#: Relative ridge on the 2x2 omega curvature, keeping a weakly constrained
#: omega's Hessian SPD. Applied once, after summing blocks, so blocking does
#: not change the answer.
RIDGE_H_REL = 1e-7
#: Independence-MH steps per conditional draw. The proposal is the quadratic
#: model's exact truncated Gaussian, so acceptance is high (0.93 measured);
#: eight steps leave the draw exact to the conditional at no meaningful cost.
N_MH = 8
#: Points in the tabulated z1 marginal of the proposal. The proposal's own
#: density enters the MH ratio, so any resolution is exact.
N_TAB = 400

_FSTAR = np.array(_model.VERTEX_FSTAR, dtype=np.float64)


# ----------------------------------------------------------------- host maps

def omega_to_q(omega):
    """Light-curve weights ``(..., 3)`` -> Kipping ``(q1, q2)``, float64."""
    lam = np.asarray(omega, dtype=np.float64) / _FSTAR
    lam = lam / lam.sum(axis=-1, keepdims=True)
    s = lam[..., 1] + lam[..., 2]
    return s ** 2, lam[..., 1] / np.maximum(s, 1e-300)


def q_to_omega(q1, q2):
    """Kipping ``(q1, q2)`` -> light-curve weights ``(..., 3)``, float64."""
    sq = np.sqrt(np.asarray(q1, dtype=np.float64))
    q2 = np.asarray(q2, dtype=np.float64)
    lam = np.stack([1.0 - sq, sq * q2, sq * (1.0 - q2)], axis=-1)
    om = lam * _FSTAR
    return om / om.sum(axis=-1, keepdims=True)


def x_to_q(x):
    """``x = (omega1, omega2)`` ``(..., 2)`` -> ``(q1, q2)``."""
    x = np.asarray(x, dtype=np.float64)
    om = np.stack([1.0 - x[..., 0] - x[..., 1], x[..., 0], x[..., 1]], -1)
    return omega_to_q(om)


def log_prior_x_np(x):
    """Log-density of ``x`` induced by uniform ``(q1, q2)`` (unnormalised)."""
    x = np.asarray(x, dtype=np.float64)
    s = ((1.0 - x[..., 0] - x[..., 1]) / _FSTAR[0]
         + x[..., 0] / _FSTAR[1] + x[..., 1] / _FSTAR[2])
    return -3.0 * np.log(s)


def _log_prior_x(x1, x2):
    f0, f1, f2 = _model.VERTEX_FSTAR
    s = (1.0 - x1 - x2) / f0 + x1 / f1 + x2 / f2
    return -3.0 * mx.log(s)


# ---------------------------------------------------- the omega-profiled block

class _VertexBlock:
    """One epoch block's vertex light curves (independent of omega)."""

    def __init__(self, design, D):
        self.design = design
        self.D0 = D[0]
        self.d1 = D[1] - D[0]
        self.d2 = D[2] - D[0]

    def arrays(self):
        return [self.D0, self.d1, self.d2]


def _block_terms(blk, x1, x2, derivs=True):
    """``L(x)`` for one block, with the baseline profiled exactly; and, if
    ``derivs``, its gradient ``g (C, 2)`` and Gauss-Newton Hessian ``H (C, 3)``
    packed ``[a, b, c]`` -- the omega columns with the baseline columns
    projected out through the same Cholesky (the Schur complement)."""
    d = blk.design
    n = d.n_cols
    f_dev = blk.D0 + x1[:, None, None] * blk.d1 + x2[:, None, None] * blk.d2
    f = 1.0 + f_dev
    M = mx.einsum("epij,cep->ceij", d.G, f * f) + d.diag_fix[None]
    ridge = _profile.RIDGE_REL.get(M.dtype, _profile.RIDGE_REL_DEFAULT)
    M = M * (1.0 + ridge * mx.eye(n, dtype=M.dtype)[None, None])
    chol = _profile._cholesky_factor(M, n)

    def sub(rhs):
        return _profile._cholesky_substitute(chol, rhs, n)

    c = sub(mx.einsum("epi,cep->cei", d.Lw, f * (d.y_dev - f_dev)))
    Lc = _profile.poly(d, c)
    resid = (d.y_dev - f_dev) - f * Lc
    loglik = mx.sum(_profile.chi2_terms(d, resid), axis=-1)
    if not derivs:
        return loglik
    g = 1.0 + Lc
    C1 = blk.d1 * g
    C2 = blk.d2 * g
    C1t = C1 - f * _profile.poly(d, sub(mx.einsum("epi,cep->cei", d.Lw, f * C1)))
    C2t = C2 - f * _profile.poly(d, sub(mx.einsum("epi,cep->cei", d.Lw, f * C2)))
    s = d.inv_sigma * d.mask
    r = resid * s
    a1, a2 = C1 * s, C2 * s
    b1, b2 = C1t * s, C2t * s

    def red(z):
        return mx.sum(z, axis=(1, 2))

    gvec = mx.stack([red(r * a1), red(r * a2)], axis=-1)
    Hent = mx.stack([red(b1 * b1), red(b1 * b2), red(b2 * b2)], axis=-1)
    return loglik, gvec, Hent


def _sum_terms(blocks, x1, x2):
    """``(L, g, H)`` summed over blocks, then the relative ridge (once, after
    the sum, which is what keeps the result block-invariant)."""
    L = g = H = None
    for blk in blocks:
        l, gv, h = _block_terms(blk, x1, x2)
        L = l if L is None else L + l
        g = gv if g is None else g + gv
        H = h if H is None else H + h
    eps = RIDGE_H_REL * (H[:, 0] + H[:, 2])
    H = H + mx.stack([eps, mx.zeros_like(eps), eps], axis=-1)
    return L, g, H


def _sum_loglik(blocks, x1, x2):
    """``L(x)`` alone, summed over blocks: the same arithmetic as
    :func:`_sum_terms`'s first output, without the derivative solves."""
    L = None
    for blk in blocks:
        l = _block_terms(blk, x1, x2, derivs=False)
        L = l if L is None else L + l
    return L


def _solve2(H, g):
    """``H^-1 g`` for packed symmetric 2x2 ``H (C, 3)`` and ``g (C, 2)``."""
    a, b, c = H[:, 0], H[:, 1], H[:, 2]
    det = a * c - b * b
    return mx.stack([(c * g[:, 0] - b * g[:, 1]) / det,
                     (a * g[:, 1] - b * g[:, 0]) / det], axis=-1)


def _project_triangle(xh1, xh2, H):
    """Minimiser of ``(x - xh)^T H (x - xh)`` over the triangle: exact QP."""
    a, b, c = H[:, 0], H[:, 1], H[:, 2]
    inside = (xh1 >= 0) & (xh2 >= 0) & (xh1 + xh2 <= 1)
    best_x1, best_x2 = xh1, xh2
    best_q = mx.full(xh1.shape, float("inf"), dtype=xh1.dtype)
    for (p1, p2, e1, e2) in ((0.0, 0.0, 1.0, 0.0),      # x2 = 0
                             (0.0, 0.0, 0.0, 1.0),      # x1 = 0
                             (1.0, 0.0, -1.0, 1.0)):    # x1 + x2 = 1
        r1, r2 = xh1 - p1, xh2 - p2
        num = e1 * (a * r1 + b * r2) + e2 * (b * r1 + c * r2)
        den = a * e1 * e1 + 2.0 * b * e1 * e2 + c * e2 * e2
        t = mx.clip(num / den, 0.0, 1.0)
        y1, y2 = p1 + t * e1, p2 + t * e2
        d1, d2 = y1 - xh1, y2 - xh2
        q = a * d1 * d1 + 2.0 * b * d1 * d2 + c * d2 * d2
        better = q < best_q
        best_x1 = mx.where(better, y1, best_x1)
        best_x2 = mx.where(better, y2, best_x2)
        best_q = mx.where(better, q, best_q)
    return mx.where(inside, xh1, best_x1), mx.where(inside, xh2, best_x2)


def constrained_optimum(blocks, n_chains, dtype):
    """The expansion point ``x*`` and the quadratic model there.

    Start at the centroid, take :data:`N_NEWTON` Newton steps each projected
    onto the triangle, then evaluate ``(L, g, H)`` at the result. Returns
    ``(x1, x2, L, g, H)``.

    ``x*`` is **detached** before the final evaluation. The collapsed value
    depends on the expansion point only through the non-quadratic part of
    ``L`` -- for an exactly quadratic ``L``, ``L(x*) + 1/2 g H^-1 g`` and
    ``x* + H^-1 g`` are the same for every ``x*`` -- so the gradient need not
    flow back through the first pass. Measured: values unchanged, gradients
    within 4.6e-6 of max|grad| against float64 including edge-active ``x*``,
    per-evaluation MLX peak 0.489 -> 0.390 GB, about 11% faster.
    """
    x1 = mx.full((n_chains,), 1.0 / 3.0, dtype=dtype)
    x2 = mx.full((n_chains,), 1.0 / 3.0, dtype=dtype)
    for _ in range(N_NEWTON):
        L, g, H = _sum_terms(blocks, x1, x2)
        step = _solve2(H, g)
        x1, x2 = _project_triangle(x1 + step[:, 0], x2 + step[:, 1], H)
    x1, x2 = mx.stop_gradient(x1), mx.stop_gradient(x2)
    L, g, H = _sum_terms(blocks, x1, x2)
    return x1, x2, L, g, H


# ------------------------------------------- Gaussian mass in the triangle

_GL_CACHE = {}


def _gl(dtype):
    if dtype not in _GL_CACHE:
        gx, gw = np.polynomial.legendre.leggauss(N_GL_OUTER)
        gy, gv = np.polynomial.legendre.leggauss(N_GL_INNER)
        _GL_CACHE[dtype] = tuple(mx.array(a, dtype=dtype)
                                 for a in (gx, gw, gy, gv))
    return _GL_CACHE[dtype]


def log_gauss_triangle(xh1, xh2, H):
    """``log int_T exp(-1/2 (x - xh)^T H (x - xh)) p(x) dx``, shape ``(C,)``.

    Whitened by ``H = L L^T``, ``z = L^T (x - xh)``: the triangle maps to a
    triangle, integrated by outer Gauss-Legendre over ``z1`` (split at the
    vertices' ``z1`` so the inner limits are linear on each piece) and inner
    Gauss-Legendre over ``z2`` between the edges, both clipped to ``|z| <=``
    :data:`Z_WINDOW`. Log-domain throughout; the prior is evaluated at the
    point mapped back, clipped into the box.
    """
    dt = xh1.dtype
    a, b, c = H[:, 0], H[:, 1], H[:, 2]
    l11 = mx.sqrt(a)
    l21 = b / l11
    l22 = mx.sqrt(mx.maximum(c - l21 * l21, 1e-30 * c))
    verts = ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))
    Z = [(l11 * (v1 - xh1) + l21 * (v2 - xh2), l22 * (v2 - xh2))
         for v1, v2 in verts]
    z1s = mx.sort(mx.stack([z[0] for z in Z], -1), axis=-1)       # (C, 3)
    W = Z_WINDOW
    gx, gw, gy, gv = _gl(dt)
    big = mx.array(1e30, dtype=dt)
    logs = []
    for lo_i, hi_i in ((0, 1), (1, 2)):
        lo = mx.clip(z1s[:, lo_i], -W, W)
        hi = mx.clip(z1s[:, hi_i], -W, W)
        half = 0.5 * (hi - lo)
        s = (0.5 * (hi + lo))[:, None] + half[:, None] * gx[None, :]   # (C, n)
        ylo, yhi = big, -big
        for ia, ib in ((0, 1), (1, 2), (2, 0)):
            za1, za2 = Z[ia]
            zb1, zb2 = Z[ib]
            den = zb1 - za1
            ok_den = mx.abs(den) > 1e-12
            safe = mx.where(ok_den, den, mx.ones_like(den))
            t = (s - za1[:, None]) / safe[:, None]
            yv = za2[:, None] + t * (zb2 - za2)[:, None]
            valid = ok_den[:, None] & (t >= 0) & (t <= 1)
            ylo = mx.minimum(ylo, mx.where(valid, yv, big))
            yhi = mx.maximum(yhi, mx.where(valid, yv, -big))
        ylo = mx.clip(ylo, -W, W)
        yhi = mx.maximum(mx.clip(yhi, -W, W), ylo)
        yh = 0.5 * (yhi - ylo)
        yy = (0.5 * (yhi + ylo))[..., None] + yh[..., None] * gy      # (C, n, m)
        lt = (-0.5 * (s[..., None] ** 2 + yy ** 2)
              + mx.log(gw)[None, :, None] + mx.log(gv)[None, None, :]
              + mx.log(mx.maximum(yh, 1e-30))[..., None]
              + mx.log(mx.maximum(half, 1e-30))[:, None, None])
        dx2 = yy / l22[:, None, None]
        dx1 = (s[..., None] - l21[:, None, None] * dx2) / l11[:, None, None]
        px1 = mx.clip(xh1[:, None, None] + dx1, 0.0, 1.0)
        px2 = mx.clip(xh2[:, None, None] + dx2, 0.0, 1.0)
        lt = lt + _log_prior_x(px1, px2)
        empty = (hi - lo) <= 0
        lt = mx.where(empty[:, None, None], mx.full(lt.shape, -1e30, dtype=dt),
                      lt)
        logs.append(lt.reshape(lt.shape[0], -1))
    return (mx.logsumexp(mx.concatenate(logs, axis=-1), axis=-1)
            - mx.log(l11 * l22))


# -------------------------------------------------------------- log-density

class MarginalLDLogProb(ProfiledTransitLogProb):
    """turin's LinEph log-density over ``(dP, dtau0, k, beta, T14)`` with the
    quadratic limb darkening integrated out (see the module docstring).

    Built through :func:`turin.likelihood.build_target`, which picks this
    class from ``layout.ld``. Reuses the parent's grid, design and epoch
    blocks; overrides the per-evaluation arithmetic only.
    """

    def __init__(self, layout, centering, epoch_data, orders, *,
                 profile_mode="exact", geometry="circular", **kw):
        if layout.mode != "lineph":
            raise ValueError(
                "collapsed limb darkening is LinEph-only: a shared omega "
                "couples every epoch, which breaks the per-epoch "
                "factorisation the TTV fit's grid-Gibbs relies on")
        if layout.ld != "collapsed":
            raise ValueError("MarginalLDLogProb needs a layout built with "
                             "lineph_layout(..., ld='collapsed')")
        if profile_mode != "exact":
            raise ValueError(
                f"collapsed limb darkening needs profile_mode='exact', not "
                f"{profile_mode!r}: the omega gradient and Hessian come from "
                f"the envelope theorem, which holds only when the baseline is "
                f"the exact flux-space minimiser")
        if geometry != "circular":
            raise ValueError("collapsed limb darkening needs "
                             "geometry='circular' (ld_basis exists only on "
                             "MetalPlanet's tau kernel)")
        if not _model.HAS_TAU_KERNEL:
            raise ValueError("collapsed limb darkening needs MetalPlanet's "
                             "tau kernel")
        kw.setdefault("bytes_per_point", COLLAPSED_BYTES_PER_POINT)
        super().__init__(layout, centering, epoch_data, orders,
                         profile_mode=profile_mode, geometry=geometry, **kw)

    def unpack(self, v):
        col = lambda i: v[:, i:i + 1]
        dP, dtau0, k, beta, T14 = (col(i) for i in range(5))
        b = _model.impact_parameter(beta, k, self.layout.b_prior)
        return dict(dP=dP, dtau0=dtau0, dtau=None, k=k, beta=beta, b=b,
                    T14=T14, period=self.layout.P_ref + dP)

    def _vertex_blocks(self, p):
        return [_VertexBlock(blk.design, _model.vertex_flux_devs(
                    blk.grid, mid=self.mid_times(blk.grid, p, blk.lo, blk.hi),
                    k=p["k"], b=p["b"], T14=p["T14"], period=p["period"]))
                for blk in self.blocks]

    def __call__(self, v):
        p = self.unpack(v)
        x1, x2, L, g, H = constrained_optimum(self._vertex_blocks(p),
                                              v.shape[0], v.dtype)
        step = _solve2(H, g)
        quad = 0.5 * (g[:, 0] * step[:, 0] + g[:, 1] * step[:, 1])
        logZ = log_gauss_triangle(x1 + step[:, 0], x2 + step[:, 1], H)
        return L + quad + logZ + self.log_prior(p)

    def epoch_log_lik(self, v):
        raise NotImplementedError(
            "collapsed limb darkening couples every epoch through the shared "
            "omega integral, so the per-epoch factorisation grid-Gibbs relies "
            "on does not exist")

    # -- helpers for the conditional draws, the ML row and the tests -------

    def ld_terms(self, v):
        """``(x1, x2, L, g, H)`` at the expansion point ``x*``."""
        p = self.unpack(v)
        return constrained_optimum(self._vertex_blocks(p), v.shape[0], v.dtype)

    def loglik_at(self, v, x):
        """``L(x)`` -- baseline profiled, no priors -- at ``x (C, 2)``."""
        blocks = self._vertex_blocks(self.unpack(v))
        x = mx.array(np.asarray(x), dtype=self.dtype) if not isinstance(
            x, mx.array) else x.astype(self.dtype)
        return _sum_loglik(blocks, x[:, 0], x[:, 1])

    def loglik_grid(self, v_row, X):
        """``L`` at many ``x``, ``X (G, 2)``, for one parameter vector
        ``v_row (dim,)``. The vertex light curves are computed once and
        broadcast, so a dense lattice costs one kernel launch."""
        v = mx.array(np.asarray(v_row, dtype=np.float64)[None, :],
                     dtype=self.dtype)
        blocks = self._vertex_blocks(self.unpack(v))
        X = mx.array(np.asarray(X, dtype=np.float64), dtype=self.dtype)
        return _sum_loglik(blocks, X[:, 0], X[:, 1])

    def conditional_mode_ld(self, v):
        """``(q1, q2)`` at the conditional mode ``x*(v)``, host float64
        ``(C,)`` each: the single point estimate the ML-row light curve
        (lcdata, fold plot, ``ml_params``) is drawn with. Every posterior
        ``q1``/``q2`` value comes from :class:`OmegaSampler` instead."""
        x1, x2, _, _, _ = self.ld_terms(v)
        x = np.stack([np.array(x1, dtype=np.float64),
                      np.array(x2, dtype=np.float64)], -1)
        return x_to_q(x)

    def full_model(self, v, *, ld=None):
        if ld is None:
            if v.shape[0] != 1:
                raise ValueError("full_model without ld= takes one row")
            q1, q2 = self.conditional_mode_ld(v)
            ld = (float(q1[0]), float(q2[0]))
        return super().full_model(v, ld=ld)


# ------------------------------------------------ conditional draws (Gibbs)

class OmegaSampler:
    """Exact draws of ``omega | theta`` -- the Gibbs half of collapsed Gibbs.

    For each theta row, independence Metropolis-Hastings against the exact
    conditional ``L(x) + log p(x)``, with the quadratic model's truncated
    Gaussian as the proposal, sampled exactly in whitened coordinates
    (``z1`` from its tabulated marginal, ``z2`` from a truncated normal
    between the triangle's edges). The proposal density in the MH ratio is
    the one actually sampled, so the draw is exact at any table resolution.

    Every chain starts from a **proposal draw**, never from ``x*``: when the
    limb darkening is weakly constrained ``x*`` often lies on a triangle
    edge, and rounding in the whitening can put it a hair outside the
    proposal's support, where its density is ``-inf`` and an MH chain
    started there can never leave. That froze 5% of draws at exactly
    ``q2 = 0`` or ``q1 = 1`` in the first version. There is deliberately no
    way to pass a starting state.

    Host loops are vectorised over rows; :func:`_sample_rows_loop` and
    :func:`_logq_rows_loop` in the tests are the per-row reference they are
    pinned against.
    """

    def __init__(self, lp_f32, *, n_mh=N_MH, n_tab=N_TAB, batch=2048, seed=0):
        if not isinstance(lp_f32, MarginalLDLogProb):
            raise TypeError("OmegaSampler needs a MarginalLDLogProb")
        self.lp = lp_f32
        self.n_mh = int(n_mh)
        self.n_tab = int(n_tab)
        self.batch = int(batch)
        self.rng = np.random.default_rng(seed)

    # -- proposal geometry ------------------------------------------------

    @staticmethod
    def _whiten(H):
        l11 = np.sqrt(H[:, 0])
        l21 = H[:, 1] / l11
        l22 = np.sqrt(np.maximum(H[:, 2] - l21 ** 2, 1e-30 * H[:, 2]))
        return l11, l21, l22

    @staticmethod
    def _ybounds(Zv, s):
        """The triangle's ``z2`` interval at ``z1 = s``; ``Zv (C, 3, 2)``,
        ``s (C, n)``."""
        ys_lo = np.full(s.shape, np.inf)
        ys_hi = np.full(s.shape, -np.inf)
        for ia, ib in ((0, 1), (1, 2), (2, 0)):
            a1, a2 = Zv[:, ia, 0:1], Zv[:, ia, 1:2]
            b1, b2 = Zv[:, ib, 0:1], Zv[:, ib, 1:2]
            den = b1 - a1
            with np.errstate(divide="ignore", invalid="ignore"):
                t = (s - a1) / den
                ok = (np.abs(den) > 1e-12) & (t >= 0) & (t <= 1)
                y = a2 + t * (b2 - a2)
            ys_lo = np.where(ok, np.minimum(ys_lo, y), ys_lo)
            ys_hi = np.where(ok, np.maximum(ys_hi, y), ys_hi)
        ys_lo = np.clip(ys_lo, -Z_WINDOW, Z_WINDOW)
        ys_hi = np.clip(ys_hi, -Z_WINDOW, Z_WINDOW)
        return ys_lo, np.maximum(ys_hi, ys_lo)

    def _tables(self, xh, H):
        from scipy.special import ndtr

        l11, l21, l22 = self._whiten(H)
        V = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        dv = V[None, :, :] - xh[:, None, :]                       # (C, 3, 2)
        z1 = l11[:, None] * dv[..., 0] + l21[:, None] * dv[..., 1]
        z2 = l22[:, None] * dv[..., 1]
        Zv = np.stack([z1, z2], -1)
        lo = np.clip(z1.min(1), -Z_WINDOW, Z_WINDOW)
        hi = np.clip(z1.max(1), -Z_WINDOW, Z_WINDOW)
        grid = lo[:, None] + (hi - lo)[:, None] * np.linspace(
            0.0, 1.0, self.n_tab)[None, :]
        ylo, yhi = self._ybounds(Zv, grid)
        m = np.exp(-0.5 * grid ** 2) * np.maximum(ndtr(yhi) - ndtr(ylo), 0.0)
        cdf = np.concatenate([
            np.zeros((m.shape[0], 1)),
            np.cumsum(0.5 * (m[:, 1:] + m[:, :-1]) * np.diff(grid, axis=1),
                      axis=1)], 1)
        return dict(l=(l11, l21, l22), Zv=Zv, grid=grid, m=m, cdf=cdf)

    def _sample(self, T, xh):
        """One proposal draw per row, vectorised. Same RNG stream order as the
        per-row reference: ``z1`` uniforms first, then ``z2``'s."""
        from scipy.special import ndtr, ndtri

        l11, l21, l22 = T["l"]
        grid, m, cdf = T["grid"], T["m"], T["cdf"]
        C = xh.shape[0]
        rows = np.arange(C)
        u = self.rng.uniform(size=C) * cdf[:, -1]
        j = np.clip((cdf < u[:, None]).sum(axis=1) - 1, 0, self.n_tab - 2)
        g0, g1 = grid[rows, j], grid[rows, j + 1]
        m0, m1 = m[rows, j], m[rows, j + 1]
        h = g1 - g0
        r = u - cdf[rows, j]
        with np.errstate(divide="ignore", invalid="ignore"):
            slope = np.where(h > 0, (m1 - m0) / h, 0.0)
            flat = np.abs(slope) < 1e-14 * np.maximum(m0, 1e-300)
            t_flat = r / np.maximum(m0, 1e-300)
            t_quad = (-m0 + np.sqrt(np.maximum(m0 * m0 + 2.0 * slope * r,
                                               0.0))) / slope
        t = np.where(flat, t_flat, t_quad)
        z1 = g0 + np.minimum(np.maximum(t, 0.0), h)
        ylo, yhi = self._ybounds(T["Zv"], z1[:, None])
        ylo, yhi = ylo[:, 0], yhi[:, 0]
        a, b = ndtr(ylo), ndtr(yhi)
        z2 = ndtri(np.clip(a + self.rng.uniform(size=C) * (b - a),
                           1e-300, 1.0 - 1e-16))
        dx2 = z2 / l22
        dx1 = (z1 - l21 * dx2) / l11
        x = np.clip(xh + np.stack([dx1, dx2], -1), 0.0, 1.0)
        # onto the simplex too: inside the triangle up to rounding already
        return x / np.maximum(1.0, x.sum(-1, keepdims=True))

    def _logq(self, T, xh, x):
        """Log-density of the proposal actually sampled, at ``x (C, 2)``.

        Points within ``1e-9 (1 + Z_WINDOW)`` of the support count as inside
        and are evaluated at the clamped point: an edge point is in the
        support, and rounding must not say otherwise.
        """
        from scipy.special import ndtr

        l11, l21, l22 = T["l"]
        grid, m, cdf = T["grid"], T["m"], T["cdf"]
        dx = x - xh
        z1 = l11 * dx[:, 0] + l21 * dx[:, 1]
        z2 = l22 * dx[:, 1]
        tol = 1e-9 * (1.0 + Z_WINDOW)
        g_lo, g_hi, total = grid[:, 0], grid[:, -1], cdf[:, -1]
        ok = (z1 >= g_lo - tol) & (z1 <= g_hi + tol) & (total > 0)
        zi = np.minimum(np.maximum(z1, g_lo), g_hi)
        rows = np.arange(x.shape[0])
        k = np.clip((grid <= zi[:, None]).sum(axis=1) - 1, 0, self.n_tab - 2)
        ga, gb = grid[rows, k], grid[rows, k + 1]
        with np.errstate(divide="ignore", invalid="ignore"):
            w = np.where(gb > ga, (zi - ga) / (gb - ga), 0.0)
            mz = (m[rows, k] * (1.0 - w) + m[rows, k + 1] * w) / total
        ylo, yhi = self._ybounds(T["Zv"], zi[:, None])
        ylo, yhi = ylo[:, 0], yhi[:, 0]
        ok &= (z2 >= ylo - tol) & (z2 <= yhi + tol)
        mass = ndtr(yhi) - ndtr(ylo)
        ok &= mass > 0
        with np.errstate(divide="ignore", invalid="ignore"):
            cz2 = np.exp(-0.5 * z2 ** 2) / math.sqrt(2.0 * math.pi) / mass
            val = np.log(mz * cz2) + np.log(l11 * l22)
        return np.where(ok, val, -np.inf)

    # -- the draw --------------------------------------------------------

    def _draw_batch(self, v_model):
        lp = self.lp
        v = mx.array(np.asarray(v_model, dtype=np.float32))
        blocks = lp._vertex_blocks(lp.unpack(v))
        mx.eval([a for blk in blocks for a in blk.arrays()])   # reuse below
        x1, x2, _, g, H = constrained_optimum(blocks, v.shape[0], lp.dtype)
        x = np.stack([np.array(x1, np.float64), np.array(x2, np.float64)], -1)
        g = np.array(g, np.float64)
        H = np.array(H, np.float64)
        det = H[:, 0] * H[:, 2] - H[:, 1] ** 2
        xh = x + np.stack([(H[:, 2] * g[:, 0] - H[:, 1] * g[:, 1]) / det,
                           (H[:, 0] * g[:, 1] - H[:, 1] * g[:, 0]) / det], -1)
        eps = 1e-6                       # x* is float32; 1 - t + t != 1
        on_edge = ((x[:, 0] <= eps) | (x[:, 1] <= eps)
                   | (x[:, 0] + x[:, 1] >= 1.0 - eps))
        T = self._tables(xh, H)

        def target(xx):
            xm = mx.array(xx, dtype=lp.dtype)
            L = np.array(_sum_loglik(blocks, xm[:, 0], xm[:, 1]), np.float64)
            return L + log_prior_x_np(xx)

        cur = self._sample(T, xh)
        lt_cur, lq_cur = target(cur), self._logq(T, xh, cur)
        acc = np.zeros(cur.shape[0])
        for _ in range(self.n_mh):
            prop = self._sample(T, xh)
            lt_p, lq_p = target(prop), self._logq(T, xh, prop)
            with np.errstate(invalid="ignore"):
                ok = np.log(self.rng.uniform(size=cur.shape[0])) < (
                    (lt_p - lq_p) - (lt_cur - lq_cur))
            cur = np.where(ok[:, None], prop, cur)
            lt_cur = np.where(ok, lt_p, lt_cur)
            lq_cur = np.where(ok, lq_p, lq_cur)
            acc += ok
        return cur, acc / self.n_mh, on_edge

    def step(self, v_model):
        """Draw ``x`` for every row of ``v_model (C, 5)`` (model space).

        Returns ``(x (C, 2), accept (C,), x_star_on_edge (C,) bool)``.
        """
        v_model = np.asarray(v_model, dtype=np.float64)
        out = [self._draw_batch(v_model[s:s + self.batch])
               for s in range(0, v_model.shape[0], self.batch)]
        return tuple(np.concatenate(parts) for parts in zip(*out))


def draw_limb_darkening(lp_f32, v_model, *, n_mh=N_MH, batch=2048, seed=0):
    """Conditional ``(q1, q2)`` draws for every row of ``v_model``.

    Returns ``(q1 (C,), q2 (C,), accept_mean, edge_share)`` -- the latter
    two for the run log (mean MH acceptance; share of rows whose expansion
    point ``x*`` sits on a triangle edge or vertex).
    """
    sampler = OmegaSampler(lp_f32, n_mh=n_mh, batch=batch, seed=seed)
    x, acc, on_edge = sampler.step(v_model)
    q1, q2 = x_to_q(x)
    return q1, q2, float(np.mean(acc)), float(np.mean(on_edge))
