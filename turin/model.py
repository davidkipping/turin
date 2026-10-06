"""The MLX forward model: parameters -> per-epoch mid-times -> z -> flux.

turin does not use MetalPlanet's sampler-facing API
(``metalplanet.anvil.make_quad_transit_flux``): that bakes in a linear
ephemeris, offers no exposure integration, carries only a scalar baseline,
and parametrizes by a/R*. turin needs per-epoch mid-transit times, Kepler
long-cadence integration, per-epoch Legendre baselines, and hurin's ``T14``
parametrization. So turin builds the sky-projected separation ``z`` itself
and calls ``metalplanet.metal.flux_dev_metal``, a fused fp32 Metal kernel
with an analytic VJP in ``z`` that falls back to MetalPlanet's fp64 analytic
path on the CPU stream. That fallback is what gives anvil its
``log_prob_hi`` for free.

Two orbit geometries are available, because hurin's was an approximation:

``circular`` (default)
    The true circular projection, ``z^2 = a^2 sin^2(phi) + b^2 cos^2(phi)``
    with ``phi = 2 pi tau / P``, via ``metalplanet.orbit.separation_circular``.
    a/R* comes from T14 by the Seager & Mallen-Ornelas Eq. 8 inversion, the
    same relation turin reports ``log10_rho`` through, so the fitted model
    and the derived stellar density are mutually consistent.

``chord``
    hurin's geometry, for parity checks. jaxoplanet's ``TransitOrbit`` is a
    straight-chord constant-speed model: ``z^2 = (v tau)^2 + b^2`` with
    ``v = 2 sqrt((1+k)^2 - b^2) / T14`` and ``b`` constant across the
    transit. It agrees with ``circular`` only to O((T14/P)^2) — measured
    4.6e-6 in flux at T14/P = 0.019.

Limb darkening is Kipping (2013) throughout, via
``metalplanet.ld.q_to_u``: ``u1 = 2 sqrt(q1) q2``, ``u2 = sqrt(q1) (1 - 2 q2)``,
whose unit square in (q1, q2) maps onto the whole physically-allowed (u1, u2)
triangle. hurin used a variant missing both factors of two until turin's port
surfaced it; hurin 0.1.68 adopted the correct map, so both packages now agree
and turin carries no alternative.

Note the quadratic law is exactly the N=2 case of the polynomial law
MetalPlanet implements (``I(mu)/I0 = 1 - sum u_n (1-mu)^n``), so Kipping's
reparameterization applies directly; verified in tests/test_model.py. The
Green's basis MetalPlanet uses internally is an affine change of basis for
the integration, not a different law.

Validated against three independent references (see tests/test_model.py):
fp64 circular agrees with MetalPlanet's batman-style frontend to 3.3e-16,
fp32 to 1.6e-7 (MetalPlanet's documented fp32 floor), and ``chord`` reproduces
hurin/jaxoplanet to 2.2e-7, which is hurin's own float32 floor. Against
hurin >= 0.1.68 the orbit model is the only remaining difference.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mlx.core as mx
import numpy as np
import metalplanet
from metalplanet.ld import q_to_u
from metalplanet.metal import flux_dev_metal
from metalplanet.orbit import separation_circular
from metalplanet.trig import sincos

#: MetalPlanet's tau-input kernel integrates the exposure *inside* the kernel,
#: by the contact rule, so the sub-exposure axis never reaches MLX. turin
#: asked for it in ``docs/upstream/metalplanet_prompt.md``; it landed on
#: 2026-09-29. Tests set this to False to exercise the fallback.
#:
#: **turin adopted it for accuracy, not speed**, and the distinction matters
#: because the brief expected the opposite. At turin's sizes (~2,500 points,
#: 8 sub-exposures) it measured *2x slower* than expanding the axis --
#: MetalPlanet's 2.3x win was at 15 sub-exposures, where materializing the
#: expansion dominates. What it buys instead:
#:
#: * flux error 6.4e-8 against 1.5e-4, i.e. the old route was wrong by 2.3%
#:   of a transit depth at the sub-exposure count Kipping (2010) Eq. 40 picks;
#: * and, far more seriously, ``dF/d(period)`` correct to 1e-2 against
#:   **98x wrong, with the wrong sign**. Supersampling a kinked integrand
#:   gives an error that oscillates as nodes cross the contacts, so
#:   differentiating it amplifies that error enormously even though the value
#:   is merely mediocre.
#:
#: The posteriors turin produced before this were still valid -- HMC's
#: Metropolis step uses the log-density, not the gradient, so a poor force
#: field costs trajectory efficiency rather than correctness -- but there is
#: no reason to keep paying for it.
HAS_TAU_KERNEL = hasattr(metalplanet, "flux_dev_from_tau")
#: Gauss-Legendre nodes per contact sub-interval.
#:
#: Judged on the log-likelihood, which is what the sampler sees. The kernel's
#: own partial dF/d(period) converges slowly (rel. error 4.3e-2 at five nodes,
#: 1.1e-2 at nine), and nine was first chosen on that basis, while SIMD
#: divergence made the kernel's cost look flat in N_GL. Diluted through the
#: whole likelihood the error is small, and with divergence gone
#: (phase_order) the cost is not flat. KOI-448.02, float64 against n_gl=32,
#: at 24 posterior draws; gradient errors in units of the posterior sd, i.e.
#: the force error HMC feels (the typical force is ~1):
#:
#:     n_gl  sd logL error  worst grad error x sd       v+g, 512 chains
#:        5      5.0e-4       5.3e-3 (beta)               9.1 ms
#:        9      3.0e-5       1.0e-3 (beta)              14.1 ms
#:       12                                              18.0 ms
#:   float32 noise alone, at either n_gl:  1.2e-2 (beta)
#:
#: So at five the quadrature error is below float32's own on the gradient,
#: and 40x below it on the value (the --PL probe accepts sd 0.02).
N_GL = 5

#: Disc-integrated flux of the three vertex laws of the Kipping (2013)
#: triangle, ``(u1, u2) = (0, 0), (2, -1), (0, 1)``: the normalisers that turn
#: :func:`vertex_flux_devs`'s unnormalised basis into light curves, and the
#: ``F*`` of the collapsed-LD prior. Python floats, so nothing float64 leaks
#: into a float32 graph.
VERTEX_FSTAR = (math.pi, math.pi / 2.0, 5.0 * math.pi / 6.0)

#: (b, k) prior parameterizations, after hurin's ``_sample_b_k``. Each maps a
#: sampled fraction in [0, 1] to the impact parameter.
B_PRIORS = ("transiting", "nongrazing", "box")
GEOMETRIES = ("circular", "chord")


def limb_dark_coeffs(q1, q2):
    """Kipping (2013) (q1, q2) -> quadratic (u1, u2), MLX, differentiable."""
    return q_to_u(q1, q2)


def limb_dark_coeffs_np(q1, q2):
    """Float64 host-side replica of :func:`limb_dark_coeffs`."""
    sq1 = np.sqrt(np.maximum(np.asarray(q1, dtype=np.float64), 1e-12))
    q2 = np.asarray(q2, dtype=np.float64)
    return 2.0 * sq1 * q2, sq1 * (1.0 - 2.0 * q2)


def impact_parameter(beta, k, b_prior="transiting"):
    """Impact parameter from the sampled fraction ``beta`` in [0, 1].

    ``transiting`` spans the whole transiting region b < 1+k, ``nongrazing``
    the non-grazing triangle b < 1-k, and ``box`` hurin's historical
    independent uniform b ~ U(0, 2). The prior's area/taper factors live in
    ``likelihood.py``; this is only the coordinate map.
    """
    if b_prior == "transiting":
        return beta * (1.0 + k)
    if b_prior == "nongrazing":
        return beta * (1.0 - k)
    if b_prior == "box":
        return beta * 2.0
    raise ValueError(f"unknown b_prior {b_prior!r}; expected one of {B_PRIORS}")


def a_over_rstar(T14, period, k, b):
    """a/R* from the transit duration: S&MO Eq. 8 inverted.

    ``a/R* = sqrt(b^2 + ((1+k)^2 - b^2) / sin^2(pi T14 / P))``

    The same relation turin reports ``log10_rho`` through, so the fitted
    geometry and the derived density are consistent.

    Uses MetalPlanet's Cody-Waite ``sincos`` rather than ``mx.sin``: MLX's
    fp64 trigonometry is only float32-accurate (measured 2.6e-8 relative),
    which would put a staircase of that size into the fp64 reference path
    that ``validate_precision`` and ``certify`` are supposed to adjudicate.
    ``sin`` is floored rather than clipped so the backward pass stays finite
    as T14 -> 0.
    """
    sin_alpha, _ = sincos(math.pi * T14 / period)
    sin_alpha = mx.maximum(sin_alpha, 1e-6)
    num = mx.maximum((1.0 + k) ** 2 - b**2, 0.0)
    return mx.sqrt(b**2 + num / sin_alpha**2)


def chord_speed(T14, k, b):
    """Sky-plane speed of jaxoplanet's straight-chord orbit, in R*/day."""
    return 2.0 * mx.sqrt(mx.maximum((1.0 + k) ** 2 - b**2, 0.0)) / T14


def separation_chord(tau, T14, k, b):
    """hurin/jaxoplanet geometry: a straight chord crossed at constant speed.

    Beyond the contact points the true model has no transit, and jaxoplanet
    enforces that with a sign flag; here ``z`` already exceeds 1+k there, so
    the flux deviation is identically zero and no mask is needed.
    """
    x = chord_speed(T14, k, b) * tau
    return mx.sqrt(x * x + b * b)


@dataclass
class EpochGrid:
    """Static, GPU-resident per-epoch data for one fit.

    Built once by :func:`build_grid` from the float64 host arrays. Times are
    per-epoch residuals (O(hours)), epoch numbers are exact small integers,
    and ``d_arr`` is the float64 remainder of the recentring (order 0), so
    nothing here loses precision in float32.
    """

    times: mx.array        # (n_epochs, max_pts) time minus epoch centre
    n_arr: mx.array        # (n_epochs,) integer epoch number vs the ephemeris
    d_arr: mx.array        # (n_epochs,) tau0_ref + n P_ref - centre, ~0
    sub_offsets: mx.array   # (n_sub,) sub-exposure offsets, fallback route only
    P_ref: float
    tau0_ref: float
    dtype: mx.Dtype
    #: exposure duration in days, for the in-kernel contact rule
    exp_time: float = 0.0
    n_gl: int = N_GL
    #: (n_epochs*max_pts,) int32 permutation putting the flattened points in
    #: order of time from predicted mid-transit, and its inverse; see
    #: :func:`phase_order`. None evaluates in storage order.
    order: mx.array | None = None
    unorder: mx.array | None = None

    @property
    def n_epochs(self) -> int:
        return self.times.shape[0]

    @property
    def max_pts(self) -> int:
        return self.times.shape[1]

    @property
    def n_sub(self) -> int:
        return self.sub_offsets.shape[0]

    def select(self, lo: int, hi: int) -> "EpochGrid":
        """A view over epochs [lo, hi): the unit of likelihood chunking.

        The point order is recomputed for the slice, since the permutation
        indexes the flattened block. Only called while building, so the host
        round-trip this costs never reaches a traced call.
        """
        order = unorder = None
        if self.order is not None:
            order, unorder = phase_order(np.array(self.times[lo:hi]),
                                         np.array(self.d_arr[lo:hi]))
        return EpochGrid(
            times=self.times[lo:hi],
            n_arr=self.n_arr[lo:hi],
            d_arr=self.d_arr[lo:hi],
            sub_offsets=self.sub_offsets,
            P_ref=self.P_ref,
            tau0_ref=self.tau0_ref,
            dtype=self.dtype,
            exp_time=self.exp_time,
            n_gl=self.n_gl,
            order=order,
            unorder=unorder,
        )


def phase_order(times, d_arr):
    """Permutation sorting flattened points by time from predicted mid-transit.

    MetalPlanet's kernel runs one point per GPU thread, and threads execute
    in SIMD groups of 32 that take every branch any member takes. A point in
    transit costs ~8x one out of transit (the contact rule, the occultation
    integrals), so storage order -- every epoch's window in turn, ~10% of it
    in transit -- puts a transit point in most groups and makes nearly all of
    them pay the in-transit price. Sorted, the in-transit points share a few
    groups and the rest exit early. Compiled value+grad of the whole
    log-density on KOI-448.02 at 512 chains: 45.8 ms in storage order,
    17.1 ms sorted, 14.1 ms with :func:`build_grid`'s padding move as well.
    Log-likelihoods were bit-identical; gradients agree to 2e-7 (summation
    order).

    The key is the reference ephemeris, fixed at build time, so the order is
    static. Parameters move the true mid-times by far less than a window, and
    order affects only speed: every point is still evaluated at its own time.
    """
    key = (np.asarray(times, dtype=np.float64)
           - np.asarray(d_arr, dtype=np.float64)[:, None]).ravel()
    order = np.argsort(key, kind="stable")
    unorder = np.empty_like(order)
    unorder[order] = np.arange(order.size)
    return (mx.array(order.astype(np.int32)),
            mx.array(unorder.astype(np.int32)))


def build_grid(centering, sub_offsets, dtype=mx.float32, *,
               exp_time=0.0, n_gl=N_GL, mask=None, sort_points=True):
    """Upload the host-side centred arrays as an :class:`EpochGrid`.

    ``centering`` is :func:`turin.prep.centering_constants` output;
    ``sub_offsets`` comes from :func:`turin.prep.supersample_offsets`.
    Every array is converted with an explicit ``dtype``: ``mx.array`` of a
    float64 NumPy array silently yields float32 otherwise.

    ``mask`` (the epoch data's, 1 for real points) moves padded slots from
    the epoch centre, where :func:`turin.prep.segment_epochs` parks them, to
    a quarter period from mid-transit. The likelihood weights them by zero
    either way, but at the centre every padded slot costs the kernel an
    in-transit evaluation -- 8.5% of KOI-448.02's slots, as many again as its
    real in-transit points. At quadrature they are far out of transit for any
    geometry, so their deviation is exactly zero.

    ``sort_points`` attaches :func:`phase_order`.
    """
    times = np.array(centering["times_centered"], dtype=np.float64)
    d_arr = np.asarray(centering["d_arr"], dtype=np.float64)
    if mask is not None:
        pad = np.asarray(mask) <= 0
        times[pad] = np.broadcast_to(
            d_arr[:, None] + 0.25 * float(centering["P_ref"]), times.shape)[pad]
    order = unorder = None
    if sort_points:
        order, unorder = phase_order(times, d_arr)
    return EpochGrid(
        times=mx.array(np.ascontiguousarray(times), dtype=dtype),
        n_arr=mx.array(np.ascontiguousarray(centering["n_arr"]), dtype=dtype),
        d_arr=mx.array(np.ascontiguousarray(centering["d_arr"]), dtype=dtype),
        sub_offsets=mx.array(np.ascontiguousarray(sub_offsets), dtype=dtype),
        P_ref=float(centering["P_ref"]),
        tau0_ref=float(centering["tau0_ref"]),
        dtype=dtype,
        exp_time=float(exp_time),
        n_gl=int(n_gl),
        order=order,
        unorder=unorder,
    )


def mid_times_lineph(grid, dP, dtau0):
    """Per-epoch mid-transit offsets under a linear ephemeris.

    ``tau'_i = dtau0 + n_i dP + d_i`` (hurin's centred-coordinate form), with
    ``dP`` and ``dtau0`` offsets from the recentred input ephemeris. Every
    term is O(1), which is the whole point of centring.

    ``dP``, ``dtau0``: (n_chains, 1). Returns (n_chains, n_epochs).
    """
    return dtau0 + grid.n_arr[None, :] * dP + grid.d_arr[None, :]


def mid_times_ttv(grid, dtau):
    """Per-epoch mid-transit offsets with one free time per epoch.

    ``tau'_i = dtau_i + d_i``, with ``dtau_i`` an offset from epoch i's
    predicted time. ``dtau``: (n_chains, n_epochs).
    """
    return dtau + grid.d_arr[None, :]


def time_from_mid(grid, mid):
    """Time of every *sample* relative to its own mid-transit.

    Returns (n_chains, n_epochs, max_pts). The sub-exposure axis is not here:
    the tau kernel integrates the exposure internally, and the fallback route
    adds the axis itself in :func:`_expanded_flux_dev`.
    """
    return grid.times[None, :, :] - mid[:, :, None]


def _expanded_flux_dev(grid, tau, *, k, b, T14, q1, q2, period, geometry):
    """Fallback: materialize the sub-exposure axis and average it in MLX.

    What turin did before MetalPlanet's tau kernel existed, and still the only
    route for ``geometry="chord"``, which the kernel does not implement. Costs
    ``n_sub`` times the memory -- measured 2.78 GB against 51 MB at 512 chains
    x 5,000 points x 15 sub-exposures -- and, more importantly, integrates by
    supersampling, whose error at a realistic ``n_sub`` is ~1.5e-4, about 2% of
    a transit depth and 1000x the float32 floor.
    """
    n_chains = tau.shape[0]
    nodes = tau[..., None] + grid.sub_offsets[None, None, None, :]
    flat = nodes.reshape(n_chains, -1)

    if geometry == "circular":
        z = separation_circular(flat, period, b, a_over_rstar(T14, period, k, b))
    elif geometry == "chord":
        z = separation_chord(flat, T14, k, b)
    else:
        raise ValueError(
            f"unknown geometry {geometry!r}; expected one of {GEOMETRIES}")

    u1, u2 = limb_dark_coeffs(q1, q2)
    dev = flux_dev_metal(z, k, u1, u2)
    dev = dev.reshape(n_chains, grid.n_epochs, grid.max_pts,
                      grid.sub_offsets.shape[0])
    return mx.mean(dev, axis=-1)


def transit_flux_dev(grid, *, mid, k, b, T14, q1, q2, period,
                     geometry="circular"):
    """Exposure-averaged transit flux **deviation**, ``f - 1``.

    Shape ``(n_chains, n_epochs, max_pts)``. Scalar-per-chain parameters
    (``k``, ``b``, ``T14``, ``q1``, ``q2``, ``period``) have shape
    ``(n_chains, 1)``; ``mid`` is ``(n_chains, n_epochs)``.

    The deviation, not the flux, is the primary quantity: it is what
    MetalPlanet computes natively, and everything downstream works in
    deviation space so that no float32 subtraction ever cancels two numbers
    near 1 to recover something near 1e-4. That is anvil's rule 3, and
    ignoring it cost a measured 1.6% on ``dlogL/dk``.

    The sub-exposure axis is averaged away here, so callers never see it. It
    is also the memory driver -- the fused kernel's backward pass writes
    several arrays of the *expanded* size -- which is why the likelihood
    chunks over epoch blocks.
    """
    if geometry not in GEOMETRIES:
        raise ValueError(
            f"unknown geometry {geometry!r}; expected one of {GEOMETRIES}")
    n_chains = mid.shape[0]
    tau = time_from_mid(grid, mid)

    if not (HAS_TAU_KERNEL and geometry == "circular"):
        return _expanded_flux_dev(grid, tau, k=k, b=b, T14=T14, q1=q1, q2=q2,
                                  period=period, geometry=geometry)

    # The tau kernel: exposure integrated in registers by the contact rule,
    # so no sub-exposure axis exists to hold. Parameters stay (n_chains, 1);
    # MetalPlanet canonicalizes that per chain itself.
    # Points go through in phase order (see phase_order) and come back out;
    # the two gathers cost well under a millisecond.
    u1, u2 = limb_dark_coeffs(q1, q2)
    flat = tau.reshape(n_chains, -1)
    if grid.order is not None:
        flat = mx.take(flat, grid.order, axis=1)
    dev = metalplanet.flux_dev_from_tau(
        flat, period,
        a_over_rstar(T14, period, k, b), b, k, u1, u2,
        exp_time=grid.exp_time,
        integration="contact" if grid.exp_time > 0 else "none",
        n_gl=grid.n_gl)
    if grid.order is not None:
        dev = mx.take(dev, grid.unorder, axis=1)
    return dev.reshape(n_chains, grid.n_epochs, grid.max_pts)


def vertex_flux_devs(grid, *, mid, k, b, T14, period):
    """Light-curve deviations of the three vertex limb-darkening laws.

    Returns ``[F0 - 1, F1 - 1, F2 - 1]``, each ``(n_chains, n_epochs,
    max_pts)``, for the Kipping-triangle vertices ``(u1, u2) = (0, 0),
    (2, -1), (0, 1)``. Every physical quadratic law's deviation is the convex
    combination ``sum_j omega_j (F_j - 1)`` with ``omega`` on the 2-simplex,
    which is what lets :mod:`turin.ldmarg` integrate the limb darkening out.

    One kernel launch: MetalPlanet's ``ld_basis=True`` (>= 0.7.0) returns the
    unnormalised deficits ``B`` for intensities ``mu^0, mu^1, mu^2``, and
    ``F0 - 1 = B0/pi``, ``F1 - 1 = B2/(pi/2)``, ``F2 - 1 = (2 B1 - B2)/(5 pi/6)``
    (verified to 3.5e-18 in float64 against three scalar calls). Same route as
    :func:`transit_flux_dev`'s tau-kernel branch, including the phase-order
    gathers, which act on axis 1 and so carry the trailing basis axis through
    unchanged. There is deliberately no fallback: the caller is gated on
    ``capabilities.MIN_METALPLANET``, and ``geometry="chord"`` has no tau
    kernel to ask.
    """
    if not HAS_TAU_KERNEL:
        raise RuntimeError("vertex_flux_devs needs MetalPlanet's tau kernel "
                           "(flux_dev_from_tau)")
    n_chains = mid.shape[0]
    flat = time_from_mid(grid, mid).reshape(n_chains, -1)
    if grid.order is not None:
        flat = mx.take(flat, grid.order, axis=1)
    basis = metalplanet.flux_dev_from_tau(
        flat, period, a_over_rstar(T14, period, k, b), b, k,
        exp_time=grid.exp_time,
        integration="contact" if grid.exp_time > 0 else "none",
        n_gl=grid.n_gl, ld_basis=True)                       # (C, m, 3)
    if grid.order is not None:
        basis = mx.take(basis, grid.unorder, axis=1)
    shape = (n_chains, grid.n_epochs, grid.max_pts)
    b0 = basis[..., 0].reshape(shape)
    b1 = basis[..., 1].reshape(shape)
    b2 = basis[..., 2].reshape(shape)
    f0, f1, f2 = VERTEX_FSTAR
    return [b0 / f0, b2 / f1, (2.0 * b1 - b2) / f2]


def transit_flux(grid, **kw):
    """Exposure-averaged transit flux, ``1 + f_dev``.

    For references, plots and exports. The sampling path uses
    :func:`transit_flux_dev` directly and stays in deviation space.
    """
    return 1.0 + transit_flux_dev(grid, **kw)


def log10_rho_from_T14(T14, period, k, b):
    """log10 of the mean stellar density in kg/m^3, from the fitted duration.

    S&MO Eq. 8 inverted for a/R*, then Kipping (2010) Eq. 3. NumPy, float64:
    a reporting quantity, computed once per draw after sampling.
    """
    G_SI = 6.67430e-11
    P_sec = np.asarray(period, dtype=np.float64) * 86400.0
    sin_alpha = np.clip(np.sin(np.pi * np.asarray(T14, dtype=np.float64)
                               / np.asarray(period, dtype=np.float64)),
                        1e-10, 1.0)
    aRs = np.sqrt(b**2 + ((1.0 + k) ** 2 - b**2) / sin_alpha**2)
    return np.log10((3.0 * np.pi / (G_SI * P_sec**2)) * aRs**3)
