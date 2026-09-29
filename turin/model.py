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

Likewise two limb-darkening maps, because hurin's is not the one it claims:

``kipping`` (default)
    Kipping (2013), as ``metalplanet.ld.q_to_u`` implements it:
    ``u1 = 2 sqrt(q1) q2``, ``u2 = sqrt(q1) (1 - 2 q2)``. The unit square in
    (q1, q2) maps onto the whole physically-allowed (u1, u2) triangle.

``hurin``
    ``u1 = sqrt(q1) q2``, ``u2 = sqrt(q1) (1 - q2)`` — hurin's
    ``transit_fit.transit_model``, whose docstring cites Kipping (2013) but
    drops both factors of two. Its q2 is twice Kipping's, and its image is
    only the ``u1, u2 >= 0`` sub-region: it cannot represent the negative
    ``u2`` that quadratic-law fits to real stars often prefer. Provided for
    parity runs, not recommended for science.

Validated against three independent references (see tests/test_model.py):
fp64 circular agrees with MetalPlanet's batman-style frontend to 3.3e-16,
fp32 to 1.6e-7 (MetalPlanet's documented fp32 floor), and ``chord`` plus the
``hurin`` LD map reproduces hurin/jaxoplanet to 2.2e-7, which is hurin's own
float32 floor.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mlx.core as mx
import numpy as np
from metalplanet.ld import q_to_u
from metalplanet.metal import flux_dev_metal
from metalplanet.orbit import separation_circular
from metalplanet.trig import sincos

#: (b, k) prior parameterizations, after hurin's ``_sample_b_k``. Each maps a
#: sampled fraction in [0, 1] to the impact parameter.
B_PRIORS = ("transiting", "nongrazing", "box")
GEOMETRIES = ("circular", "chord")
LD_MAPS = ("kipping", "hurin")


def limb_dark_coeffs(q1, q2, ld_map="kipping"):
    """(q1, q2) -> (u1, u2) under the chosen map; see the module docstring."""
    if ld_map == "kipping":
        return q_to_u(q1, q2)
    if ld_map == "hurin":
        sq1 = mx.sqrt(mx.maximum(q1, 1e-12))
        return sq1 * q2, sq1 * (1.0 - q2)
    raise ValueError(f"unknown ld_map {ld_map!r}; expected one of {LD_MAPS}")


def limb_dark_coeffs_np(q1, q2, ld_map="kipping"):
    """Float64 host-side replica of :func:`limb_dark_coeffs`."""
    sq1 = np.sqrt(np.maximum(np.asarray(q1, dtype=np.float64), 1e-12))
    q2 = np.asarray(q2, dtype=np.float64)
    if ld_map == "kipping":
        return 2.0 * sq1 * q2, sq1 * (1.0 - 2.0 * q2)
    if ld_map == "hurin":
        return sq1 * q2, sq1 * (1.0 - q2)
    raise ValueError(f"unknown ld_map {ld_map!r}; expected one of {LD_MAPS}")


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
    sub_offsets: mx.array   # (n_sub,) sub-exposure offsets about each sample
    P_ref: float
    tau0_ref: float
    dtype: mx.Dtype

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
        """A view over epochs [lo, hi): the unit of likelihood chunking."""
        return EpochGrid(
            times=self.times[lo:hi],
            n_arr=self.n_arr[lo:hi],
            d_arr=self.d_arr[lo:hi],
            sub_offsets=self.sub_offsets,
            P_ref=self.P_ref,
            tau0_ref=self.tau0_ref,
            dtype=self.dtype,
        )


def build_grid(centering, sub_offsets, dtype=mx.float32):
    """Upload the host-side centred arrays as an :class:`EpochGrid`.

    ``centering`` is :func:`turin.prep.centering_constants` output;
    ``sub_offsets`` comes from :func:`turin.prep.supersample_offsets`.
    Every array is converted with an explicit ``dtype``: ``mx.array`` of a
    float64 NumPy array silently yields float32 otherwise.
    """
    return EpochGrid(
        times=mx.array(np.ascontiguousarray(centering["times_centered"]),
                       dtype=dtype),
        n_arr=mx.array(np.ascontiguousarray(centering["n_arr"]), dtype=dtype),
        d_arr=mx.array(np.ascontiguousarray(centering["d_arr"]), dtype=dtype),
        sub_offsets=mx.array(np.ascontiguousarray(sub_offsets), dtype=dtype),
        P_ref=float(centering["P_ref"]),
        tau0_ref=float(centering["tau0_ref"]),
        dtype=dtype,
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
    """Time of every sub-exposure node relative to its own mid-transit.

    Returns (n_chains, n_epochs, max_pts, n_sub).
    """
    tau = grid.times[None, :, :] - mid[:, :, None]
    return tau[..., None] + grid.sub_offsets[None, None, None, :]


def transit_flux_dev(grid, *, mid, k, b, T14, q1, q2, period,
                     geometry="circular", ld_map="kipping"):
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
    n_chains = mid.shape[0]
    tau = time_from_mid(grid, mid)
    flat = tau.reshape(n_chains, -1)

    if geometry == "circular":
        aRs = a_over_rstar(T14, period, k, b)
        z = separation_circular(flat, period, b, aRs)
    elif geometry == "chord":
        z = separation_chord(flat, T14, k, b)
    else:
        raise ValueError(
            f"unknown geometry {geometry!r}; expected one of {GEOMETRIES}")

    u1, u2 = limb_dark_coeffs(q1, q2, ld_map)
    # Parameters stay (n_chains, 1). The fused fp32 kernel canonicalizes that
    # to (n_chains,) itself, while the fp64 analytic fallback broadcasts it
    # against z's (n_chains, m) -- which a flat (n_chains,) would not do.
    dev = flux_dev_metal(z, k, u1, u2)
    dev = dev.reshape(n_chains, grid.n_epochs, grid.max_pts, grid.n_sub)
    return mx.mean(dev, axis=-1)


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
