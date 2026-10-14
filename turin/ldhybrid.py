"""Collapsed (marginalised) hybrid4 limb darkening: ``--law=hybrid4 --ld=collapsed``.

The hybrid4 analogue of :mod:`turin.ldmarg` (the quadratic law's), and
collapsed Gibbs sampling in the same sense: the sampler targets the
**marginal** ``p(theta | D)`` with the four limb-darkening weights integrated
out, and the weights are then **drawn from their conditional** for every kept
``theta`` to fill the ``q1..q4`` product columns. Not profiling.

Every law on hybrid4's weight simplex is a convex mix of five generator light
curves, ``f - 1 = sum_g omega_g (F_g - 1)`` with ``omega`` on the 4-simplex
(:func:`turin.model.generator_flux_devs`), and uniform weights induce the
prior ``log p(omega) = -5 log sum_g omega_g / F*_g``. With
``x = (omega_1..omega_4)`` the baseline-profiled ``L(x)`` is handled exactly as
in ``ldmarg`` -- the Legendre baseline profiled for each ``x``, the gradient
and Gauss-Newton curvature from the envelope theorem through the Schur
complement (hence ``--PL=exact``) -- and the sampled value is the integral of
its quadratic model about a detached expansion point ``x*``:

    L(x*) + 1/2 |u|^2 + log Z,   u = R^-T g,   H = R^T R,

``Z`` the prior-weighted Gaussian mass over the simplex. What changes from 2-D:

- ``Z`` by **GHK** on a fixed 1024-point scrambled-Sobol rule with a layout
  frozen once per fit (:mod:`turin.simplexmass`), where the triangle used
  Gauss-Legendre. Measured against a 2^18 reference at posterior draws: max
  0.015 nats on KOI-518.02, 95th percentile 0.03 (max 0.085) on KOI-448.02's
  two-solution posterior; ~1/N, and not improved by re-choosing the layout.
- All algebra on the **QR factor** ``R`` of the projected columns (two-pass
  Gram-Schmidt, TSQR over epoch blocks), never on ``H``: on KOI-448.02 ``H``
  reaches cond 7e9, where a float32 solve is wrong by 12x; ``R``'s is <= 8.5e4.
- ``x*``: the quadratic model at the centroid, then its exact constrained
  optimum over the simplex (face enumeration, each face solved by QR on
  ``R``). With that ``x*`` the collapsed value matched the true
  ``log int exp(L) p`` to < 1e-4 nats at every posterior draw on both
  targets.
"""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np

from . import model as _model
from . import profile as _profile
from . import simplexmass as _sm
from .likelihood import BYTES_PER_POINT, ProfiledTransitLogProb

LAW = "hybrid4"
#: number of generators (G) and of free weights (D = G - 1)
G = len(_model.ld_names(LAW)) + 1
D = G - 1
#: Per-(chain, point) bytes for block sizing: five generator curves and four
#: projected columns live per point, against the default path's one curve
#: and the quadratic collapsed mode's three; scaled from the quadratic
#: mode's measured 2.5x (``ldmarg.COLLAPSED_BYTES_PER_POINT``).
COLLAPSED_HYBRID_BYTES_PER_POINT = int(4.0 * BYTES_PER_POINT)
#: Newton+projection passes before the final evaluation. 1, 2 and 3 agreed
#: to < 1e-4 nats at every posterior draw on KOI-518.02 and KOI-448.02.
N_NEWTON = 1
#: GHK rule size and its default seed. A fit stores the points themselves.
N_GHK = 1024
GHK_SEED = 2024
#: Relative ridge on the curvature, as ``ldmarg.RIDGE_H_REL``; applied once,
#: after combining blocks.
RIDGE_H_REL = 1e-7
#: Independence-MH proposals per conditional draw (fresh GHK draws).
N_PROP = 32

FSTAR = np.array(_model.generator_norms(LAW), dtype=np.float64)
INV_FSTAR = 1.0 / FSTAR
DEFAULT_LAYOUT = (0, tuple(range(1, G)))


# ----------------------------------------------------------------- host maps

def x_to_omega(x):
    x = np.asarray(x, dtype=np.float64)
    return np.concatenate([1.0 - x.sum(-1, keepdims=True), x], -1)


def omega_to_w(om):
    """Light-curve weights ``omega (..., G)`` -> shape weights ``w (..., D)``:
    ``lambda_g ∝ omega_g / F*_g``, ``w = lambda[1:]``."""
    lam = np.maximum(np.asarray(om, dtype=np.float64), 0.0) / FSTAR
    lam = lam / lam.sum(-1, keepdims=True)
    return lam[..., 1:]


def w_to_omega(w):
    w = np.asarray(w, dtype=np.float64)
    lam = np.concatenate([1.0 - w.sum(-1, keepdims=True), w], -1)
    om = lam * FSTAR
    return om / om.sum(-1, keepdims=True)


def x_to_q(x):
    """``x (..., D)`` -> the sampled-mode coordinates ``q1..q4`` (..., D)."""
    from elome.ld import q_from_simplex_np

    return np.clip(q_from_simplex_np(omega_to_w(x_to_omega(x))), 0.0, 1.0)


def q_to_x(q):
    return w_to_omega(_model.ld_weights_np(q, LAW))[..., 1:]


def log_prior_x_np(x):
    """``-G log sum_g omega_g/F*_g`` (unnormalised); -inf off the simplex."""
    om = x_to_omega(x)
    ok = np.all(om >= -1e-12, -1)
    s = np.maximum((np.maximum(om, 0.0) / FSTAR).sum(-1), 1e-300)
    return np.where(ok, -G * np.log(s), -np.inf)


# ---------------------------------------------------- the omega-profiled block

class _GenBlock:
    """One epoch block's generator light curves (independent of omega)."""

    def __init__(self, design, F):
        self.design = design
        self.D0 = F[0]
        self.d = [F[j] - F[0] for j in range(1, G)]

    def arrays(self):
        return [self.D0] + self.d


def _block_terms(blk, x, derivs=True):
    """``L(x)`` for one block, the baseline profiled exactly; and, if
    ``derivs``, the gradient ``g (C, D)`` and the per-block QR factor
    ``R (C, D, D)`` of the projected, whitened omega columns (the Schur
    complement), whose ``R^T R`` is the Gauss-Newton curvature."""
    d = blk.design
    n = d.n_cols
    f_dev = blk.D0
    for j in range(D):
        f_dev = f_dev + x[:, j][:, None, None] * blk.d[j]
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
    gfac = 1.0 + Lc
    s = d.inv_sigma * d.mask
    r = resid * s
    C = x.shape[0]
    gs, cols = [], []
    for j in range(D):
        Cj = blk.d[j] * gfac
        Ct = Cj - f * _profile.poly(d, sub(mx.einsum("epi,cep->cei", d.Lw,
                                                     f * Cj)))
        gs.append(mx.sum(r * Cj * s, axis=(1, 2)))
        cols.append((Ct * s).reshape(C, -1))
    return loglik, mx.stack(gs, axis=-1), _sm.mgs_r(cols)


def _sum_terms(blocks, x):
    """``(L, g, R)`` over blocks: ``L`` and ``g`` summed, ``R`` by TSQR with
    the relative ridge added once (block-invariant)."""
    L = g = None
    Rs = []
    for blk in blocks:
        l, gv, R = _block_terms(blk, x)
        L = l if L is None else L + l
        g = gv if g is None else g + gv
        Rs.append(R)
    return L, g, _sm.tsqr(Rs, RIDGE_H_REL)


def _sum_loglik(blocks, x):
    L = None
    for blk in blocks:
        l = _block_terms(blk, x, derivs=False)
        L = l if L is None else L + l
    return L


def expansion_point(blocks, n_chains, dtype):
    """``x*`` and the quadratic model there: ``(x, L, g, R)``.

    :data:`N_NEWTON` passes: the quadratic model at the current point (the
    centroid first), its **exact** constrained optimum over the simplex
    (:func:`simplexmass.qp_simplex_r`, in the curvature's metric), then
    ``x*`` detached and ``(L, g, R)`` evaluated there. The exact QP matters:
    a Euclidean projection of the Newton step lands far from the optimum
    along stiff directions when the curvature is anisotropic, and the
    quadratic model about such a point is wrong where the mass is (measured
    on a well-measured synthetic transit: errors of nats to thousands of
    nats). A non-finite model (a degenerate block) keeps the centroid.
    """
    x = mx.full((n_chains, D), 1.0 / G, dtype=dtype)
    for _ in range(N_NEWTON):
        L, g, R = _sum_terms(blocks, x)
        b = mx.einsum("cij,cj->ci", R, x) + _sm.solve_upper_t(R, g)
        xn = _sm.qp_simplex_r(R, b)
        x = mx.where(mx.all(mx.isfinite(xn), axis=-1, keepdims=True), xn, x)
    x = mx.stop_gradient(x)
    L, g, R = _sum_terms(blocks, x)
    return x, L, g, R


# -------------------------------------------------------------- log-density

class MarginalHybridLogProb(ProfiledTransitLogProb):
    """turin's LinEph log-density over ``(dP, dtau0, k, beta, T14)`` with the
    hybrid4 limb darkening integrated out (see the module docstring).

    ``ghk_layout`` ``(dep, order)`` and ``ghk_points`` ``(N, 4)`` define the
    GHK rule and are part of the target: a fit freezes them at its MAP and
    stores both, so a resume or replot rebuilds the identical density.
    """

    def __init__(self, layout, centering, epoch_data, orders, *,
                 profile_mode="exact", geometry="circular", ghk_layout=None,
                 ghk_points=None, **kw):
        if layout.mode != "lineph":
            raise ValueError(
                "collapsed limb darkening is LinEph-only: shared weights "
                "couple every epoch, which breaks the per-epoch factorisation "
                "the TTV fit's grid-Gibbs relies on")
        if layout.ld != "collapsed" or layout.law != LAW:
            raise ValueError("MarginalHybridLogProb needs a layout built with "
                             "lineph_layout(..., ld='collapsed', "
                             "law='hybrid4')")
        if profile_mode != "exact":
            raise ValueError(
                f"collapsed limb darkening needs profile_mode='exact', not "
                f"{profile_mode!r}: the omega gradient and curvature come from "
                f"the envelope theorem, which holds only when the baseline is "
                f"the exact flux-space minimiser")
        if geometry != "circular":
            raise ValueError("collapsed limb darkening needs "
                             "geometry='circular' (ld_basis exists only on "
                             "elome's tau kernel)")
        if not _model.HAS_TAU_KERNEL:
            raise ValueError("collapsed limb darkening needs elome's "
                             "tau kernel")
        kw.setdefault("bytes_per_point", COLLAPSED_HYBRID_BYTES_PER_POINT)
        super().__init__(layout, centering, epoch_data, orders,
                         profile_mode=profile_mode, geometry=geometry, **kw)
        self.set_ghk(ghk_layout, ghk_points)

    def set_ghk(self, ghk_layout=None, ghk_points=None):
        """Install the GHK layout and rule (both part of the density)."""
        dep, order = ghk_layout if ghk_layout is not None else DEFAULT_LAYOUT
        order = tuple(int(o) for o in order)
        T, t = _sm.layout_map(int(dep), order, G)
        pts = (np.asarray(ghk_points, dtype=np.float64)
               if ghk_points is not None else _sm.sobol_points(N_GHK, D,
                                                               GHK_SEED))
        if pts.ndim != 2 or pts.shape[1] != D:
            raise ValueError(f"ghk_points must be (N, {D}), got {pts.shape}")
        self.ghk_layout = (int(dep), order)
        self.ghk_points = pts
        self._T = mx.array(T, dtype=self.dtype)
        self._t = mx.array(t, dtype=self.dtype)
        self._U = mx.array(pts, dtype=self.dtype)

    def unpack(self, v):
        lay = self.layout
        col = lambda n: v[:, lay.index(n):lay.index(n) + 1]
        dP, dtau0, k, beta, T14 = (col(n) for n in
                                   ("dP", "dtau0", "k", "beta", "T14"))
        b = _model.impact_parameter(beta, k, lay.b_prior)
        return dict(dP=dP, dtau0=dtau0, dtau=None, k=k, beta=beta, b=b,
                    T14=T14, period=lay.P_ref + dP)

    def _gen_blocks(self, p):
        return [_GenBlock(blk.design, _model.generator_flux_devs(
                    blk.grid, mid=self.mid_times(blk.grid, p, blk.lo, blk.hi),
                    k=p["k"], b=p["b"], T14=p["T14"], period=p["period"],
                    law=LAW))
                for blk in self.blocks]

    def _gauss(self, x, g, R):
        """``(u, my, Ly)``: the whitened gradient, and the mean and lower
        covariance factor of the free weights under the quadratic model.
        ``Cy = T H^-1 T^T = M^T M`` with ``M = R^-T T^T``, so ``Ly`` is the
        transpose of ``M``'s QR factor -- no ``H`` is ever formed."""
        u = _sm.solve_upper_t(R, g)
        xh = x + _sm.solve_upper(R, u)
        my = xh @ self._T.T + self._t[None, :]
        C = x.shape[0]
        Tm = self._T
        cols = [_sm.solve_upper_t(R, mx.broadcast_to(Tm[c][None, :], (C, D)))
                for c in range(D)]                             # columns of M
        R2 = _sm.mgs_r(cols)
        Ly = mx.swapaxes(R2, -1, -2)
        return u, my, Ly

    def __call__(self, v):
        p = self.unpack(v)
        x, L, g, R = expansion_point(self._gen_blocks(p), v.shape[0], v.dtype)
        u, my, Ly = self._gauss(x, g, R)
        dep, order = self.ghk_layout
        logZ = _sm.ghk_log_mass(my, Ly, self._U, dep, order, INV_FSTAR)
        return L + 0.5 * mx.sum(u * u, axis=-1) + logZ + self.log_prior(p)

    def epoch_log_lik(self, v):
        raise NotImplementedError(
            "collapsed limb darkening couples every epoch through the shared "
            "omega integral, so the per-epoch factorisation grid-Gibbs relies "
            "on does not exist")

    # -- helpers for the conditional draws, the ML row and the tests -------

    def ld_terms(self, v):
        """``(x*, L, g, R)`` at the expansion point."""
        return expansion_point(self._gen_blocks(self.unpack(v)), v.shape[0],
                               self.dtype)

    def loglik_at(self, v, x):
        """``L(x)`` -- baseline profiled, no priors -- at ``x (C, 4)``."""
        blocks = self._gen_blocks(self.unpack(v))
        x = (x.astype(self.dtype) if isinstance(x, mx.array)
             else mx.array(np.asarray(x), dtype=self.dtype))
        return _sum_loglik(blocks, x)

    def loglik_grid(self, v_row, X):
        """``L`` at many ``x`` (``X (N, 4)``) for one parameter vector: the
        generator curves are computed once and broadcast."""
        v = mx.array(np.asarray(v_row, dtype=np.float64)[None, :],
                     dtype=self.dtype)
        blocks = self._gen_blocks(self.unpack(v))
        X = mx.array(np.asarray(X, dtype=np.float64), dtype=self.dtype)
        return _sum_loglik(blocks, X)

    def conditional_mode_ld(self, v):
        """``q1..q4`` at the conditional mode -- the exact QP over the simplex
        of the quadratic model, on the host -- as a tuple of ``(C,)`` arrays:
        the single point estimate the ML-row light curve is drawn with.
        Every posterior ``q`` value comes from :class:`HybridOmegaSampler`."""
        x, L, g, R = self.ld_terms(v)
        x = np.array(x, dtype=np.float64)
        g = np.array(g, dtype=np.float64)
        R = np.array(R, dtype=np.float64)
        out = []
        for i in range(x.shape[0]):
            H = R[i].T @ R[i]
            xh = x[i] + np.linalg.lstsq(H, g[i], rcond=None)[0]
            out.append(_sm.qp_simplex_np(xh, H))
        q = x_to_q(np.array(out))
        return tuple(q[:, j] for j in range(D))

    def full_model(self, v, *, ld=None):
        if ld is None:
            if v.shape[0] != 1:
                raise ValueError("full_model without ld= takes one row")
            ld = tuple(float(a[0]) for a in self.conditional_mode_ld(v))
        return super().full_model(v, ld=ld)


def freeze_layout(lp, v_row):
    """The GHK layout to freeze for a fit, chosen at ``v_row`` (the MAP, in
    model space) in float64 on the host (:func:`simplexmass.choose_layout`)."""
    with mx.stream(mx.cpu):
        x, _, g, R = lp.ld_terms(mx.array(np.asarray(v_row, dtype=np.float64)
                                          [None, :], dtype=lp.dtype))
        x = np.array(x, dtype=np.float64)[0]
        R = np.array(R, dtype=np.float64)[0]
        g = np.array(g, dtype=np.float64)[0]
    H = R.T @ R
    xs = _sm.qp_simplex_np(x + np.linalg.lstsq(H, g, rcond=None)[0], H)
    return _sm.choose_layout(xs, H)


# ------------------------------------------------ conditional draws (Gibbs)

class HybridOmegaSampler:
    """Exact draws of ``omega | theta`` -- the Gibbs half of collapsed Gibbs.

    For each theta row, an independence Metropolis-Hastings chain through
    :data:`N_PROP` fresh GHK draws of the quadratic model's prior-weighted
    truncated Gaussian, against the exact conditional ``L(x) + log p(x)``.
    A GHK draw's density is the Gaussian over the product of its truncation
    masses, so the MH weight is ``(L - L_model)(x) + lw(x)`` with ``lw`` the
    GHK log-weight: exact at any rule. The chain **starts from a proposal
    draw**, never from ``x*``, which may sit on a face (the edge-atom lesson
    of ``ldmarg.OmegaSampler``); there is no way to pass a start.
    """

    def __init__(self, lp_f32, *, n_prop=N_PROP, batch=1024, seed=0):
        if not isinstance(lp_f32, MarginalHybridLogProb):
            raise TypeError("HybridOmegaSampler needs a MarginalHybridLogProb")
        self.lp = lp_f32
        self.n_prop = int(n_prop)
        self.batch = int(batch)
        self.rng = np.random.default_rng(seed)

    def _draw_batch(self, v_model):
        lp = self.lp
        v = mx.array(np.asarray(v_model, dtype=np.float32), dtype=lp.dtype)
        blocks = lp._gen_blocks(lp.unpack(v))
        mx.eval([a for blk in blocks for a in blk.arrays()])
        x, L, g, R = expansion_point(blocks, v.shape[0], lp.dtype)
        u, my, Ly = lp._gauss(x, g, R)
        C = v.shape[0]
        dep, order = lp.ghk_layout
        U = np.clip(self.rng.uniform(size=(C, self.n_prop, D)), 1e-7,
                    1.0 - 1e-7)
        lw, Y = _sm.ghk_terms(my, Ly, mx.array(U, dtype=lp.dtype), dep, order,
                              INV_FSTAR)
        lw = np.array(lw, dtype=np.float64)
        X = _sm.omega_from_free(np.array(Y, dtype=np.float64), dep,
                                order)[..., 1:]                 # (C, P, D)
        X = np.clip(X, 0.0, 1.0)
        X = X / np.maximum(1.0, X.sum(-1, keepdims=True))
        xs = np.array(x, dtype=np.float64)
        gn = np.array(g, dtype=np.float64)
        Rn = np.array(R, dtype=np.float64)
        dx = X - xs[:, None, :]
        Lq = (gn[:, None, :] * dx).sum(-1) - 0.5 * np.sum(
            np.einsum("cij,cpj->cpi", Rn, dx) ** 2, -1)       # minus L*
        Lt = np.stack([np.array(_sum_loglik(blocks, mx.array(
            X[:, i, :], dtype=lp.dtype)), dtype=np.float64)
            for i in range(self.n_prop)], -1) - np.array(L, np.float64)[:, None]
        logw = (Lt - Lq) + lw
        logw = np.where(np.isfinite(logw), logw, -np.inf)
        cur = np.zeros(C, dtype=int)
        wcur = logw[:, 0]
        acc = np.zeros(C)
        logu = np.log(self.rng.uniform(size=(C, self.n_prop)))
        for i in range(1, self.n_prop):
            with np.errstate(invalid="ignore"):
                ok = logu[:, i] < logw[:, i] - wcur
            cur = np.where(ok, i, cur)
            wcur = np.where(ok, logw[:, i], wcur)
            acc += ok
        on_face = np.any(x_to_omega(xs) < 1e-6, -1)
        return X[np.arange(C), cur], acc / (self.n_prop - 1), on_face

    def step(self, v_model):
        """Draw ``x`` for every row of ``v_model (C, 5)`` (model space).
        Returns ``(x (C, 4), accept (C,), x_star_on_face (C,) bool)``."""
        v_model = np.asarray(v_model, dtype=np.float64)
        out = [self._draw_batch(v_model[s:s + self.batch])
               for s in range(0, v_model.shape[0], self.batch)]
        return tuple(np.concatenate(parts) for parts in zip(*out))


def draw_limb_darkening(lp_f32, v_model, *, n_prop=N_PROP, batch=1024,
                        seed=0):
    """Conditional ``q1..q4`` draws for every row of ``v_model``.

    Returns ``(q (C, 4), accept_mean, face_share)``: the sampled-mode
    coordinates, the mean MH acceptance, and the share of rows whose
    expansion point sits on a face of the simplex.
    """
    sampler = HybridOmegaSampler(lp_f32, n_prop=n_prop, batch=batch,
                                 seed=seed)
    x, acc, on_face = sampler.step(v_model)
    return x_to_q(x), float(np.mean(acc)), float(np.mean(on_face))
