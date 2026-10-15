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

from . import likelihood as _lik
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
#: Absolute ridge on the curvature, in nats over the weight coordinates
#: (0.1.64). Where the data do not see the limb darkening at all -- no
#: transit in the data, k -> 0 -- the curvature vanishes with the relative
#: ridge, the model's Gaussian centres ~1e10 from the simplex, and GHK
#: cannot cancel the resulting magnitudes: on KOI-2650.01 the collapsed value
#: came out +3,456 nats too high at k = 0 (float64; float32 similar), a
#: false peak that trapped 70 of 512 chains. The simplex is at most sqrt(2)
#: across, so this ridge moves the integrand by at most 1e-3 nats anywhere;
#: measured at those thetas, the error went to +0.004 (1e-6 and 1e-3 agree;
#: 0.1 costs 0.03).
RIDGE_H_ABS = 1e-3
#: Steps per conditional draw (:class:`HybridOmegaSampler`): 128 was needed
#: by the hardest synthetic case, the mass in a corner of the simplex.
#: Fixed, never adaptive: stopping on the chain's own history (e.g. after k
#: acceptances) is itself biased.
N_STEPS = 128
#: Exact-HMC travel time before the scalings in ``_draw_batch``. pi/8, not
#: the textbook pi/2: the moves bounce off the simplex's faces, and the work
#: is the bounces. Against independent references on six synthetic thetas
#: and a vertex case, pi/8 was as accurate as pi/4 and pi/2 at 128 steps
#: (every case within ~1 standard error) and 1.5x cheaper than pi/2.
HMC_T = 0.125 * math.pi

FSTAR = np.array(_model.generator_norms(LAW), dtype=np.float64)
INV_FSTAR = 1.0 / FSTAR
#: ``log`` of the induced prior's normalising constant: uniform ``q`` is
#: uniform ``w`` (density 4! on its simplex), and the projective map to
#: ``omega`` has Jacobian ``prod(1/F*) (sum omega/F*)^-G``, so
#: ``p(x) = 4! prod(1/F*) (sum omega/F*)^-G`` exactly. Added to the
#: collapsed value so it is on sampled mode's scale (it was 0.7685 high).
LOG_PRIOR_NORM = math.log(math.factorial(D)) - float(np.sum(np.log(FSTAR)))
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
    """``-G log sum_g omega_g/F*_g`` (unnormalised: add
    :data:`LOG_PRIOR_NORM` for the density); -inf off the simplex."""
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
    the relative and absolute ridges added once (block-invariant)."""
    L = g = None
    Rs = []
    for blk in blocks:
        l, gv, R = _block_terms(blk, x)
        L = l if L is None else L + l
        g = gv if g is None else g + gv
        Rs.append(R)
    return L, g, _sm.tsqr(Rs, RIDGE_H_REL, RIDGE_H_ABS)


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
        p = dict(dP=dP, dtau0=dtau0, dtau=None, k=k, beta=beta, b=b,
                 T14=T14, period=lay.P_ref + dP)
        if lay.oblate:
            p.update(_lik.oblate_fields(col("fc"), col("fs"), k))
        return p

    def _gen_blocks(self, p):
        return [_GenBlock(blk.design, _model.generator_flux_devs(
                    blk.grid, mid=self.mid_times(blk.grid, p, blk.lo, blk.hi),
                    k=p.get("k_flux", p["k"]), b=p["b"], T14=p["T14"],
                    period=p["period"], law=LAW, f=p.get("f_flux"),
                    theta=p.get("theta")))
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
        # GHK integrates the unnormalised prior; LOG_PRIOR_NORM normalises it
        return (L + 0.5 * mx.sum(u * u, axis=-1) + logZ + LOG_PRIOR_NORM
                + self.log_prior(p))

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

    For each theta row, a Metropolis chain on the simplex against the exact
    conditional ``L(x) + log p(x)``. Its moves leave the quadratic model's
    Gaussian truncated to the simplex invariant, so a Metropolis test on
    ``(L - L_model) + log p`` alone makes the exact conditional stationary;
    the model is close to the truth (acceptance ~0.9 measured). Each step is,
    with equal probability, **exact HMC** in the whitened coordinates
    (:func:`_exact_hmc`, Pakman & Paninski 2014: closed-form trajectories,
    analytic reflections, no step size), which mixes the interior, or an
    exact 1-D draw along one of the simplex's edge directions
    (:func:`_edge_move`), which mixes a corner -- a needle in whitened
    coordinates, where HMC alone stalled. Plain hit-and-run, tried first,
    took hundreds of steps in a corner.

    This replaced (0.1.64) an independence sampler through fresh GHK draws.
    That one is exact too, but a GHK draw's weight degenerates when the
    conditional is narrow and away from the model's Gaussian -- acceptance
    fell to 0.4-1.9% on synthetic thetas and 32 proposals left the product
    q's 0.4 sd off on KOI-4016.01 -- and no affordable chain length fixed
    it. The chain starts **strictly inside** the simplex next to ``x*``
    (:meth:`_start`), never on ``x*`` itself, which may sit on a face: the
    edge-atom lesson of ``ldmarg.OmegaSampler``, whose chains could not leave
    such a start. These chains move (acceptance ~0.9), and a proposal that
    rounds onto a face is refused (:func:`_inside`).
    """

    def __init__(self, lp_f32, *, n_steps=N_STEPS, batch=1024, seed=0):
        if not isinstance(lp_f32, MarginalHybridLogProb):
            raise TypeError("HybridOmegaSampler needs a MarginalHybridLogProb")
        self.lp = lp_f32
        self.n_steps = int(n_steps)
        self.batch = int(batch)
        self.rng = np.random.default_rng(seed)

    def _start(self, xs):
        """Just inside the simplex at ``x*``: (1 - 1e-6) x* + 1e-6 centroid,
        after clipping float32 rounding. The model's mass sits against ``x*``
        (its constrained optimum), so the chain starts where it is; GHK
        starts, tried first, left 128-step chains biased where the mass is a
        thin layer far from the model's mean."""
        x0 = np.clip(xs, 0.0, None)
        x0 = x0 / np.maximum(1.0, x0.sum(-1, keepdims=True))
        x0 = (1.0 - 1e-6) * x0 + 1e-6 / G
        return np.where(_inside(x0)[:, None], x0, 1.0 / G)

    def _draw_batch(self, v_model):
        lp = self.lp
        rng = self.rng
        v = mx.array(np.asarray(v_model, dtype=np.float32), dtype=lp.dtype)
        blocks = lp._gen_blocks(lp.unpack(v))
        mx.eval([a for blk in blocks for a in blk.arrays()])
        x, L, g, R = expansion_point(blocks, v.shape[0], lp.dtype)
        C = v.shape[0]
        xs = np.array(x, dtype=np.float64)
        gn = np.array(g, dtype=np.float64)
        Rn = np.array(R, dtype=np.float64)
        Hn = np.einsum("cki,ckj->cij", Rn, Rn)
        L0 = np.array(L, dtype=np.float64)

        def true_l(X):
            return np.array(_sum_loglik(blocks, mx.array(X, dtype=lp.dtype)),
                            dtype=np.float64) - L0

        def corr(X, Lt):
            dx = X - xs
            Lq = (gn * dx).sum(-1) - 0.5 * np.einsum("ci,cij,cj->c", dx, Hn, dx)
            return (Lt - Lq) + log_prior_x_np(X)

        # whitened coordinates of the model N(xh, H^-1): z = R (x - xh);
        # the simplex is F z + f >= 0 with F = A R^-1, f = A xh + a0, where
        # omega = A x + a0 (omega_0 = 1 - sum x, omega_j = x_j)
        Rinv = np.linalg.inv(Rn)
        xh = xs + np.einsum("cij,cj->ci", Rinv, np.einsum(
            "cji,cj->ci", Rinv, gn))                          # x* + H^-1 g
        Am = np.vstack([-np.ones(D), np.eye(D)])              # (G, D)
        a0 = np.r_[1.0, np.zeros(D)]
        F = np.einsum("gi,cij->cgj", Am, Rinv)                 # (C, G, D)
        f = xh @ Am.T + a0                                     # (C, G)

        # Travel time. Where the model's mean is far outside the simplex the
        # truncated mass is a thin layer d = |R (x* - xh)| sds out (x* is the
        # model's exact constrained optimum, so d is the distance from the
        # mean to the simplex), the Gaussian looks like a slope of 1/d there,
        # and a pi/2 trajectory would cross it thousands of times (measured:
        # every move refused at d ~ 30). T ~ 1/d keeps the step on that scale;
        # it depends on theta alone, so the move stays exact, and the jitter
        # breaks periodic orbits.
        dist = np.linalg.norm(np.einsum("cij,cj->ci", Rn, xs - xh), axis=-1)
        # ... and where the curvature is weak (k -> 0: the ridge alone) the
        # whole simplex is small in whitened units and a unit-speed trajectory
        # would cross it again and again, so T is also capped at the
        # simplex's median edge length in those units (theta alone; inactive
        # on the acceptance targets, whose median edge is 8-17).
        width = np.median(np.linalg.norm(np.einsum("cij,ej->cei", Rn, _EDGES),
                                         axis=-1), axis=-1)
        T0 = HMC_T * np.minimum(1.0 / np.maximum(1.0, dist),
                                np.minimum(1.0, width))

        cur = self._start(xs)
        ccur = corr(cur, true_l(cur))
        acc = np.zeros(C)
        for _ in range(self.n_steps):
            # a mixture of two moves, each leaving the truncated model
            # invariant, picked independently of the state: exact HMC in the
            # whitened coordinates (the interior), or an exact 1-D draw along
            # a direction trading weight between two omega components (the
            # simplex's own edges; a corner of the simplex is a needle in the
            # whitened coordinates, where HMC bounces without end -- measured,
            # 79% of its moves hit the bounce cap at a vertex)
            use_hmc = rng.random(C) < 0.5
            z0 = np.einsum("cij,cj->ci", Rn, cur - xh)
            T = T0 * rng.uniform(0.5, 1.5, C)
            s0 = rng.standard_normal((C, D))
            h = np.flatnonzero(use_hmc)
            z1 = z0.copy()
            ok_h = np.zeros(C, dtype=bool)
            z1[h], ok_h[h] = _exact_hmc(z0[h], s0[h], F[h], f[h], T=T[h])
            p_h = xh + np.einsum("cij,cj->ci", Rinv, z1)
            p_e, ok_e = _edge_move(cur, xs, gn, Hn, rng)
            prop = np.where(use_hmc[:, None], p_h, p_e)
            ok = np.where(use_hmc, ok_h, ok_e) & _inside(prop)
            safe = np.where(ok[:, None], prop, cur)
            cprop = np.where(ok, corr(safe, true_l(safe)), -np.inf)
            with np.errstate(invalid="ignore"):
                take = ok & (np.log(rng.random(C)) < cprop - ccur)
            cur = np.where(take[:, None], prop, cur)
            ccur = np.where(take, cprop, ccur)
            acc += take
        on_face = np.any(x_to_omega(xs) < 1e-6, -1)
        return cur, acc / max(self.n_steps, 1), on_face

    def step(self, v_model):
        """Draw ``x`` for every row of ``v_model (C, dim)`` (model space).
        Returns ``(x (C, 4), accept (C,), x_star_on_face (C,) bool)``."""
        v_model = np.asarray(v_model, dtype=np.float64)
        out = [self._draw_batch(v_model[s:s + self.batch])
               for s in range(0, v_model.shape[0], self.batch)]
        return tuple(np.concatenate(parts) for parts in zip(*out))


def _inside(X):
    """Strictly inside the simplex, and mapping to q strictly inside the
    box: a point rounded onto a face would be an atom on the q box's edge
    (3 in 614,400 GHK-based draws once did), and faces have probability
    zero, so excluding them is exact."""
    q = x_to_q(X)
    return (np.all(x_to_omega(X) > 0.0, -1)
            & np.all((q > 0.0) & (q < 1.0), -1))


def _exact_hmc(z0, s, F, f, T=0.5 * math.pi, max_bounces=200):
    """One iteration of exact HMC for ``N(0, I)`` truncated to
    ``F z + f >= 0`` (Pakman & Paninski 2014): the trajectory
    ``z(t) = z0 cos t + s sin t`` is followed for time ``T`` (per row),
    reflecting analytically off each wall it meets. The map is
    volume-preserving and
    reversible, so the truncated Gaussian is invariant with no Metropolis
    test. Returns ``(z, ok)``; a row needing more than ``max_bounces``
    reflections stays at ``z0`` (``ok`` False) -- the bounce count is the
    same forwards and backwards, so refusing those keeps it reversible.
    Rows are independent; arrays ``(C, D)``, ``F (C, G, D)``, ``f (C, G)``.
    """
    z = z0.copy()
    v = s.copy()
    left = np.broadcast_to(np.asarray(T, dtype=np.float64),
                           (z.shape[0],)).copy()
    nb = np.zeros(z.shape[0], dtype=int)
    idx = np.arange(z.shape[0])            # rows still travelling
    for _ in range(max_bounces + 1):
        Fi, fi, zi, vi = F[idx], f[idx], z[idx], v[idx]
        A = np.einsum("cgd,cd->cg", Fi, zi)
        B = np.einsum("cgd,cd->cg", Fi, vi)
        u = np.hypot(A, B)
        phi = np.arctan2(B, A)
        with np.errstate(invalid="ignore", divide="ignore"):
            c = -fi / u
            th = np.where(np.abs(c) <= 1.0, np.mod(phi + np.arccos(
                np.clip(c, -1.0, 1.0)), 2 * math.pi), np.inf)
        th = np.where(th > 1e-10, th, np.inf)       # not the wall we sit on
        j = np.argmin(th, -1)
        r = np.arange(len(idx))
        t_hit = th[r, j]
        hit = t_hit < left[idx]
        t = np.where(hit, t_hit, left[idx])
        ct, st = np.cos(t)[:, None], np.sin(t)[:, None]
        zn = zi * ct + vi * st
        vn = -zi * st + vi * ct
        Fj = Fi[r, j]
        refl = vn - 2.0 * (np.sum(Fj * vn, -1) / np.sum(Fj * Fj, -1)
                           )[:, None] * Fj
        z[idx] = zn
        v[idx] = np.where(hit[:, None], refl, vn)
        left[idx] -= t
        nb[idx] += hit
        idx = idx[hit]
        if idx.size == 0:
            break
    live = np.zeros(z.shape[0], dtype=bool)
    live[idx] = True
    ok = ~live & (nb <= max_bounces)
    return np.where(ok[:, None], z, z0), ok


#: the directions trading weight between two omega components, in x
_EDGES = np.array([np.r_[np.eye(G)[i] - np.eye(G)[j]][1:]
                   for i in range(G) for j in range(i + 1, G)])


def _edge_move(cur, xs, g, H, rng):
    """An exact draw of the quadratic model restricted to the line through
    ``cur`` along a random :data:`_EDGES` direction (a 1-D normal truncated
    to the simplex). Reversible for the truncated model: the direction is
    chosen independently of the state, and the step is a Gibbs draw."""
    C = cur.shape[0]
    d = _EDGES[rng.integers(len(_EDGES), size=C)]
    a = np.einsum("ci,cij,cj->c", d, H, d)
    b = ((g - np.einsum("cij,cj->ci", H, cur - xs)) * d).sum(-1)
    om = x_to_omega(cur)
    dom = np.concatenate([-d.sum(-1, keepdims=True), d], -1)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = -om / dom
    lo = np.max(np.where(dom > 0, r, -np.inf), -1)
    hi = np.min(np.where(dom < 0, r, np.inf), -1)
    sa = np.sqrt(a)
    m = b / a
    t = m + _truncnorm((lo - m) * sa, (hi - m) * sa, rng.random(C)) / sa
    return cur + t[:, None] * d, np.isfinite(t)


def _truncnorm(a, b, u):
    """Standard normal truncated to ``[a, b]``, by inversion of ``u``;
    float64, tail-stable (scipy's log-space ``truncnorm``)."""
    from scipy.stats import truncnorm

    z = truncnorm.ppf(u, a, b)
    return np.clip(np.where(np.isfinite(z), z, 0.5 * (a + b)), a, b)


def draw_limb_darkening(lp_f32, v_model, *, n_steps=N_STEPS, batch=1024,
                        seed=0):
    """Conditional ``q1..q4`` draws for every row of ``v_model``.

    Returns ``(q (C, 4), accept_mean, face_share)``: the sampled-mode
    coordinates, the mean MH acceptance, and the share of rows whose
    expansion point sits on a face of the simplex.
    """
    sampler = HybridOmegaSampler(lp_f32, n_steps=n_steps, batch=batch,
                                 seed=seed)
    x, acc, on_face = sampler.step(v_model)
    return x_to_q(x), float(np.mean(acc)), float(np.mean(on_face))
