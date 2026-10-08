"""Profile-likelihood Legendre detrending, batched over chains and epochs.

Each epoch's local baseline is ``g(t) = 1 + L c``, with ``L`` the Legendre
design matrix on times mapped to [-1, 1] across the epoch's data, and the
total model is ``f_transit * g``. The coefficients ``c`` are never sampled:
they are solved analytically at every log-density evaluation. This is a
profile likelihood -- a plug-in maximum-likelihood estimate of the nuisance
parameters -- not a marginalization, so no ``-0.5 log det`` Occam factor is
added. That matches hurin.

Three solves are available, selected by ``--PL``:

``exact`` (default)
    The true flux-space profile. Minimizing ``sum w (y - f (1 + L c))^2``
    over ``c`` gives normal equations in the design matrix ``A = diag(f) L``:

        (A^T W A) c = A^T W (y - f)

    Correct, but the matrix carries ``f``, so both the ``O(n_pts m^2)``
    contraction and the ``O(m^3)`` factorization are redone every evaluation.

``ratio``
    hurin's form (``transit_fit.py:323``): a weighted least-squares fit in
    *ratio* space, ``g_obs = y / max(f, 0.5)`` against ``L`` with weights
    ``1/sigma^2``, which drops the ``f`` factors from the design and the
    weights. It agrees with ``exact`` to O(depth). Its structural advantage
    is that the normal matrix is then independent of the transit parameters,
    so it is factorized once at setup and each evaluation is two triangular
    substitutions.

``hybrid``
    Both: ``ratio``'s static factor used as a *preconditioner*, then
    :data:`N_REFINE` steps of iterative refinement toward ``exact``. Since
    ``||I - M_ratio^-1 M_exact|| = O(depth)``, the error contracts by a factor
    of the transit depth per step, and the fixed point is the exact normal
    equations. Each step costs two ``O(n_pts m)`` matvecs, and the exact
    matrix is never formed.

    It is worth using only for a **large** basis. Measured end to end
    (512 chains, 30 epochs, 2,573 points, value+grad, compiled):

        basis cols      exact     hybrid      ratio   hybrid/exact
                 3     13.2ms     12.8ms     11.1ms          0.97x
                 6     18.2ms     21.9ms     15.7ms          1.20x
                10     21.2ms     27.1ms     13.9ms          1.27x
                16     54.5ms     30.3ms     20.3ms          0.56x
                24    216.2ms     55.7ms     14.4ms          0.26x
                40   1785.4ms    101.4ms     36.6ms          0.06x

    The crossover is around 12-14 columns. Below it the profile solve is not
    the bottleneck at all (the transit model is), and hybrid's extra
    triangular substitutions lose to one fused contraction. Above it
    ``exact`` collapses -- its ``O(m^3)`` factorization is *unrolled
    elementwise ops*, so m=40 issues on the order of 10,000 kernels -- while
    hybrid stays near-linear because the exact matrix is never formed.

    Note the crossover is a property of this implementation, not of the
    mathematics: a proper batched Cholesky would push it out. Since turin's
    cross-validation caps the Legendre order at 5 (six columns), turin today
    lives to the left of it -- but none of these numbers need to be trusted
    by hand. ``--PL`` defaults to ``auto``, which measures all three modes on
    the actual target and machine; see :mod:`turin.plselect`.

Implementation notes that matter:

- **The linear algebra is an unrolled Cholesky in elementwise ops.** MLX's
  ``linalg.cholesky``/``solve``/``inv`` are CPU-only *and* have no VJP, so
  they cannot appear in a compiled GPU log-density at all. The systems are at
  most 6x6 and of fixed size, so unrolling is both possible and fast.
- **Nothing static is rebuilt per call.** The design matrix, the weights and
  the contraction tensors depend only on the time grid, so they are built
  once on the host. In ``ratio`` mode even the normal matrix is static, and
  only the right-hand side moves.
- **Weights are normalized per epoch.** ``c`` is invariant to scaling ``W``
  by a constant, so the solve uses ``mask * (sigma_med/sigma)^2``, whose
  normal matrix has entries of order the point count rather than order
  ``1/sigma^2 ~ 1e8``. The likelihood itself still uses the true ``sigma``.
- **Inactive columns are pinned, not regularized.** Columns above an epoch's
  cross-validated order are zeroed, which leaves their rows and columns of
  the normal matrix exactly zero. hurin relies on a ``1e-10 I`` Tikhonov term
  to invert that block; in float32 ``1e-10`` is far below the round-off of a
  diagonal of order 100 and vanishes, leaving a singular system. turin
  instead puts an exact 1 on those diagonal entries, so the solve returns
  ``c_j = 0`` for them by construction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mlx.core as mx
import numpy as np

from .prep import basis_x, legendre_matrix

PROFILE_MODES = ("exact", "hybrid", "ratio")

#: Refinement steps for ``hybrid``. Each step contracts the error by a factor
#: of order the transit depth. Measured relative error against ``exact``:
#:
#:     depth     k=0      k=1      k=2      k=3
#:     1e-4    6.7e-5   5.8e-9   1e-12        -
#:     1e-3    8.1e-4   7.0e-7   6.0e-10  7e-13
#:     1e-2    8.5e-3   7.3e-5   6.3e-7   5.4e-9
#:     5e-2    4.2e-2   1.8e-3   7.4e-5   3.1e-6
#:
#: Three puts every depth up to a per-cent below float32's own 1.2e-7
#: precision, i.e. indistinguishable from ``exact`` on the production path,
#: and still beats ``ratio`` by four orders on a 5% eclipse. Each step costs
#: two O(n_pts * m) matvecs, so this is cheap insurance.
N_REFINE = 3

#: Relative ridge on active diagonal entries, by working precision. Scale-free
#: so that it survives float32 -- hurin's absolute ``1e-10`` against a
#: diagonal of order 100 is below float32 round-off and does nothing at all.
#: It is only ever needed for an epoch whose real points cluster at one time
#: (nearly collinear Legendre columns); inactive columns are handled exactly
#: by ``diag_fix``, and the order capping prevents underdetermined systems.
#: float32 gets 1e-6, about 8 ulp, which is below its own noise; float64 gets
#: 1e-12, so the reference path stays faithful to the unregularized solution.
RIDGE_REL = {mx.float32: 1e-6, mx.float64: 1e-12}
RIDGE_REL_DEFAULT = 1e-6


@dataclass
class ProfileDesign:
    """Static, GPU-resident tensors for the per-epoch profile solve.

    Shapes: ``E`` epochs, ``P`` padded points per epoch, ``n`` columns.
    """

    L: mx.array          # (E, P, n) Legendre design, inactive columns zeroed
    Lw: mx.array         # (E, P, n) L scaled by the normalized solve weights
    G: mx.array          # (E, P, n, n) Lw_i * L_j, for the exact normal matrix
    M_ratio: mx.array    # (E, n, n) static normal matrix for ratio mode
    #: (E, n, n) its Cholesky factor, precomputed in float64 on the host.
    #: This is the structural payoff of the ratio formulation: the normal
    #: matrix does not depend on the transit parameters, so it is factorized
    #: once at setup and the hot loop is two triangular solves.
    chol_ratio: mx.array
    diag_fix: mx.array   # (E, n, n) puts 1 on inactive diagonal entries
    y_dev: mx.array      # (E, P) observed flux MINUS 1, padded with 0.0
    inv_sigma: mx.array  # (E, P) 1/sigma with padded points zeroed
    mask: mx.array       # (E, P) 1.0 for real points
    orders: np.ndarray   # (E,) active order per epoch, after capping
    n_real_per_epoch: np.ndarray  # (E,) real point count, host-side
    n_real: int          # total number of real data points
    #: float64 host constant: -N/2 (the recentring offset) - sum log sigma
    #: - (N/2) log(2 pi). Kept off the graph so the float32 sum stays O(sqrt N).
    log_const: float
    n_cols: int
    dtype: mx.Dtype

    @property
    def n_epochs(self) -> int:
        return self.L.shape[0]

    def select(self, lo: int, hi: int) -> "ProfileDesign":
        """A view over epochs [lo, hi), the unit of likelihood chunking.

        ``log_const`` is deliberately zeroed: it is a per-fit constant the
        caller adds once, not once per block. Everything here is host-side
        slicing or an MLX view, with no ``eval`` -- blocks are built at setup
        so that nothing in the traced log-density forces a synchronization.
        """
        return ProfileDesign(
            L=self.L[lo:hi], Lw=self.Lw[lo:hi], G=self.G[lo:hi],
            M_ratio=self.M_ratio[lo:hi], chol_ratio=self.chol_ratio[lo:hi],
            diag_fix=self.diag_fix[lo:hi],
            y_dev=self.y_dev[lo:hi], inv_sigma=self.inv_sigma[lo:hi],
            mask=self.mask[lo:hi], orders=self.orders[lo:hi],
            n_real_per_epoch=self.n_real_per_epoch[lo:hi],
            n_real=int(self.n_real_per_epoch[lo:hi].sum()),
            log_const=0.0, n_cols=self.n_cols, dtype=self.dtype,
        )


def build_design(epoch_data, orders, dtype=mx.float32):
    """Build the static profile-solve tensors from segmented epoch data.

    ``epoch_data`` is :func:`turin.prep.segment_epochs` output; ``orders`` is
    the per-epoch Legendre order from
    :func:`turin.prep.optimize_legendre_orders`. Each order is capped at
    ``n_real - 1`` so no epoch is ever handed an underdetermined system.
    """
    times = np.asarray(epoch_data["times_padded"], dtype=np.float64)
    flux = np.asarray(epoch_data["flux_padded"], dtype=np.float64)
    ferr = np.asarray(epoch_data["ferr_padded"], dtype=np.float64)
    mask = np.asarray(epoch_data["mask"], dtype=np.float64)
    n_epochs, max_pts = times.shape
    orders = np.asarray(orders, dtype=int).copy()
    if orders.shape != (n_epochs,):
        raise ValueError(
            f"orders has shape {orders.shape}, expected ({n_epochs},)")
    n_real_per_epoch = mask.sum(axis=1).astype(int)
    orders = np.clip(orders, 0, np.maximum(n_real_per_epoch - 1, 0))
    n_cols = int(orders.max()) + 1

    # design matrix on times mapped to [-1, 1] across each epoch's own data
    # (prep.basis_x: same polynomial space as the window map, conditioned);
    # padded slots sit at x = 0 and are killed by the mask
    x = basis_x(epoch_data)
    L = np.zeros((n_epochs, max_pts, n_cols))
    for i in range(n_epochs):
        Li = legendre_matrix(x[i], n_cols - 1)
        # zero the columns above this epoch's active order
        Li[:, orders[i] + 1:] = 0.0
        L[i] = Li

    # solve weights, normalized per epoch: c is invariant to the scale, and
    # this keeps the normal matrix at order n_pts instead of order 1/sigma^2
    inv_var = mask / np.maximum(ferr, 1e-300) ** 2
    scale = np.where(n_real_per_epoch > 0,
                     np.median(np.where(mask > 0, inv_var, np.nan), axis=1),
                     1.0)
    scale = np.where(np.isfinite(scale) & (scale > 0), scale, 1.0)
    w_solve = inv_var / scale[:, None]

    Lw = w_solve[:, :, None] * L
    G = Lw[:, :, :, None] * L[:, :, None, :]
    M_ratio = np.einsum("epi,epj->eij", Lw, L)

    # exact 1.0 on the diagonal of every inactive column, so c_j = 0 there
    diag_fix = np.zeros((n_epochs, n_cols, n_cols))
    for i in range(n_epochs):
        for j in range(orders[i] + 1, n_cols):
            diag_fix[i, j, j] = 1.0

    # Factorize the static ratio-mode normal matrix once, in float64. The
    # same pinning and ridge the hot path would apply are baked in here.
    ridge = RIDGE_REL.get(dtype, RIDGE_REL_DEFAULT)
    M_r = (M_ratio + diag_fix) * (1.0 + ridge * np.eye(n_cols))
    chol_ratio = np.linalg.cholesky(M_r)

    inv_sigma = np.where(mask > 0, 1.0 / np.maximum(ferr, 1e-300), 0.0)
    n_real = int(mask.sum())

    # float64 host constants: the recentring offset plus the Gaussian
    # normalization hurin also carries, so exported log-likelihoods are absolute
    log_sigma_sum = float(np.sum(np.where(mask > 0, np.log(np.maximum(ferr, 1e-300)), 0.0)))
    log_const = (-0.5 * n_real - log_sigma_sum
                 - 0.5 * n_real * math.log(2.0 * math.pi))

    up = lambda a: mx.array(np.ascontiguousarray(a), dtype=dtype)
    return ProfileDesign(
        L=up(L), Lw=up(Lw), G=up(G), M_ratio=up(M_ratio),
        chol_ratio=up(chol_ratio), diag_fix=up(diag_fix),
        y_dev=up(flux - 1.0), inv_sigma=up(inv_sigma), mask=up(mask),
        orders=orders,
        n_real_per_epoch=n_real_per_epoch, n_real=n_real,
        log_const=log_const, n_cols=n_cols, dtype=dtype,
    )


def _cholesky_factor(M, n):
    """Cholesky factor of a batched SPD matrix, unrolled. ``M``: (..., n, n)."""
    a = [[M[..., i, j] for j in range(n)] for i in range(n)]
    chol = [[None] * n for _ in range(n)]
    for j in range(n):
        d = a[j][j]
        for k in range(j):
            d = d - chol[j][k] * chol[j][k]
        # the diagonal is >= RIDGE_REL * trace by construction; the floor is
        # belt-and-braces so a pathological epoch cannot produce a NaN that
        # poisons every chain's likelihood
        ljj = mx.sqrt(mx.maximum(d, 1e-30))
        chol[j][j] = ljj
        for i in range(j + 1, n):
            s = a[i][j]
            for k in range(j):
                s = s - chol[i][k] * chol[j][k]
            chol[i][j] = s / ljj
    return chol


def _cholesky_substitute(chol, rhs, n):
    """Forward then back substitution against a Cholesky factor.

    ``chol`` is the nested-list factor from :func:`_cholesky_factor`, or any
    lower-triangular factor sliced the same way. ``rhs``: (..., n).
    """
    b = [rhs[..., i] for i in range(n)]

    # forward substitution, L z = rhs
    z = [None] * n
    for i in range(n):
        s = b[i]
        for k in range(i):
            s = s - chol[i][k] * z[k]
        z[i] = s / chol[i][i]

    # back substitution, L^T c = z
    c = [None] * n
    for i in reversed(range(n)):
        s = z[i]
        for k in range(i + 1, n):
            s = s - chol[k][i] * c[k]
        c[i] = s / chol[i][i]

    return mx.stack(c, axis=-1)


def _ratio_factor(design, n):
    """The precomputed ratio-mode Cholesky factor, sliced for substitution."""
    return [[design.chol_ratio[None, :, i, j] for j in range(n)]
            for i in range(n)]


def _cholesky_solve(M, rhs, n):
    """Solve ``M c = rhs`` for symmetric positive-definite ``M``, unrolled.

    ``M``: (..., n, n); ``rhs``: (..., n). Returns (..., n). Pure elementwise
    MLX arithmetic on scalar slices: no ``linalg``, GPU-resident,
    differentiable, and fixed-shape so it compiles.
    """
    return _cholesky_substitute(_cholesky_factor(M, n), rhs, n)


def solve_coefficients(design, f_dev, mode="exact"):
    """Profiled Legendre coefficients, ``(n_chains, n_epochs, n_cols)``.

    ``f_dev``: the transit flux **deviation** ``f - 1``, shape
    ``(n_chains, n_epochs, max_pts)``.

    Everything is formed from deviations. The residual that drives the solve
    is ``y - f = y_dev - f_dev``, a difference of two quantities of order
    1e-3 rather than of two quantities near 1: forming it as ``y - f`` in
    float32 cancels four digits and was measured to put 1.6% error into
    ``dlogL/dk``.
    """
    d = design
    n = d.n_cols
    resid = d.y_dev - f_dev          # = y - f, without the cancellation

    if mode == "exact":
        # M_ij = sum_p w_p f_p^2 L_pi L_pj ; rhs_i = sum_p w_p f_p L_pi (y_p - f_p)
        f = 1.0 + f_dev
        M = mx.einsum("epij,cep->ceij", d.G, f * f)
        rhs = mx.einsum("epi,cep->cei", d.Lw, f * resid)
        # pin inactive columns, then a scale-free ridge on what remains
        M = M + d.diag_fix[None]
        ridge = RIDGE_REL.get(M.dtype, RIDGE_REL_DEFAULT)
        M = M * (1.0 + ridge * mx.eye(n, dtype=M.dtype)[None, None])
        return _cholesky_solve(M, rhs, n)

    if mode == "hybrid":
        # Exact accuracy at close to ratio's cost.
        #
        # The exact normal matrix is M_e = L^T W F^2 L with F = diag(f), and
        # the ratio matrix M_r = L^T W L is M_e with F -> I, so
        # ||I - M_r^-1 M_e|| = O(depth). That makes M_r an excellent
        # *preconditioner*, and M_r is exactly what was factorized once at
        # setup. So: solve with M_r, then refine.
        #
        #     c_0 = M_r^-1 b
        #     c_k = c_{k-1} + M_r^-1 (b - M_e c_{k-1})
        #
        # The fixed point satisfies b = M_e c, i.e. the exact normal
        # equations, and the error contracts by O(depth) per step. The trick
        # is that M_e c never needs M_e to be *formed*: it is
        # ``L^T W (f^2 (L c))``, two matvecs at O(n_pts * m) -- never the
        # O(n_pts * m^2) contraction nor the O(m^3) factorization.
        f = 1.0 + f_dev
        f2 = f * f
        b = mx.einsum("epi,cep->cei", d.Lw, f * resid)
        chol = _ratio_factor(d, n)
        c = _cholesky_substitute(chol, b, n)
        for _ in range(N_REFINE):
            Mc = mx.einsum("epi,cep->cei", d.Lw,
                           f2 * mx.einsum("epi,cei->cep", d.L, c))
            c = c + _cholesky_substitute(chol, b - Mc, n)
        return c

    if mode == "ratio":
        # hurin: fit y/f - 1 against L, with the transit factored out of both
        # the design and the weights. g_obs - 1 = (y - f)/f, formed from the
        # deviation.
        #
        # The structural payoff: the normal matrix does not depend on the
        # transit parameters, so its Cholesky factor was computed once, in
        # float64, at setup -- pinning and ridge already applied. The hot loop
        # is two triangular substitutions, with neither the O(n_pts * n^2)
        # contraction nor the O(n^3) factorization that `exact` must redo on
        # every evaluation. The saving grows with the size of the basis.
        f_safe = mx.maximum(1.0 + f_dev, 0.5)
        rhs = mx.einsum("epi,cep->cei", d.Lw, resid / f_safe)
        return _cholesky_substitute(_ratio_factor(d, n), rhs, n)

    raise ValueError(
        f"unknown profile mode {mode!r}; expected one of {PROFILE_MODES}")


def poly(design, coeffs):
    """``L c``, the baseline deviation, ``(n_chains, n_epochs, max_pts)``."""
    return mx.einsum("epi,cei->cep", design.L, coeffs)


def baseline(design, coeffs):
    """``1 + L c`` evaluated at every point, ``(n_chains, n_epochs, max_pts)``."""
    return 1.0 + poly(design, coeffs)


def residual_dev(design, f_dev, coeffs):
    """``y - model`` in deviation space, ``(n_chains, n_epochs, max_pts)``.

    ``y - f (1 + L c) = (y_dev - f_dev) - (1 + f_dev) L c``: every term is
    small, so no float32 cancellation against 1 ever happens.
    """
    Lc = poly(design, coeffs)
    return (design.y_dev - f_dev) - (1.0 + f_dev) * Lc


def detrended_model(design, f_dev, mode="exact"):
    """The full model ``f * (1 + L c)`` (absolute flux) and the coefficients."""
    c = solve_coefficients(design, f_dev, mode)
    return (1.0 + f_dev) * baseline(design, c), c


def chi2_terms(design, residual):
    """Recentred per-epoch chi-squared contribution, ``(n_chains, n_epochs)``.

    ``residual`` is ``y - model`` (see :func:`residual_dev`). Returns
    ``sum_p mask * 0.5 * (1 - r) * (1 + r)`` with ``r = residual/sigma``,
    which equals ``N_real/2 - 0.5 * sum r^2``. The exact ``-N/2`` lives in
    ``design.log_const`` as a float64 host constant.

    Why not simply ``-0.5 * sum r^2``: in float32 that sum is of order
    ``N/2 ~ 1e4-1e5``, whose ulp is comparable to the ~1-unit Metropolis
    scale, while the recentred form is of order ``sqrt(N/2)``. ``1 - r`` is
    exact for ``0.5 <= r <= 2`` by Sterbenz's lemma. This also sharpens HMC's
    energy difference, which ``anvil.validate_precision`` cannot observe.
    """
    r = residual * design.inv_sigma
    return mx.sum(design.mask * 0.5 * (1.0 - r) * (1.0 + r), axis=-1)


def log_likelihood(design, f_dev, mode="exact"):
    """Profiled Gaussian log-likelihood term, plus model and coefficients.

    Returns ``(term, model, coeffs)`` where ``term`` is ``(n_chains,)``. The
    float64 ``design.log_const`` is added on the host by the caller (see
    :mod:`turin.likelihood`); this returns the float32 graph term only.
    """
    c = solve_coefficients(design, f_dev, mode)
    resid = residual_dev(design, f_dev, c)
    term = mx.sum(chi2_terms(design, resid), axis=-1)
    return term, (1.0 + f_dev) * baseline(design, c), c


def solve_coefficients_np(epoch_data, orders, f_transit, mode="exact"):
    """Float64 NumPy reference solve, for tests and post-processing.

    Takes the absolute flux ``f_transit`` (not the deviation) and is written
    straight from the normal equations with ``np.linalg.solve``, so it shares
    no code and no conventions with the MLX path.
    """
    times = np.asarray(epoch_data["times_padded"], dtype=np.float64)
    flux = np.asarray(epoch_data["flux_padded"], dtype=np.float64)
    ferr = np.asarray(epoch_data["ferr_padded"], dtype=np.float64)
    mask = np.asarray(epoch_data["mask"], dtype=np.float64)
    x_all = basis_x(epoch_data)

    f_transit = np.atleast_3d(np.asarray(f_transit, dtype=np.float64))
    n_chains, n_epochs, _ = f_transit.shape
    orders = np.clip(np.asarray(orders, dtype=int), 0,
                     np.maximum(mask.sum(axis=1).astype(int) - 1, 0))
    n_cols = int(orders.max()) + 1

    out = np.zeros((n_chains, n_epochs, n_cols))
    for ch in range(n_chains):
        for e in range(n_epochs):
            k = orders[e] + 1
            sel = mask[e] > 0
            if not sel.any():
                continue
            Lf = legendre_matrix(x_all[e][sel], k - 1)
            w = 1.0 / ferr[e][sel] ** 2
            f = f_transit[ch, e][sel]
            if mode in ("exact", "hybrid"):
                # hybrid's fixed point IS the exact normal equations
                A = f[:, None] * Lf
                target = flux[e][sel] - f
            elif mode == "ratio":
                A = Lf
                target = flux[e][sel] / np.maximum(f, 0.5) - 1.0
            else:
                raise ValueError(f"unknown profile mode {mode!r}")
            AtW = A.T * w[None, :]
            out[ch, e, :k] = np.linalg.solve(AtW @ A, AtW @ target)
    return out
