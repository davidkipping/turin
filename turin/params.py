"""Parameter layouts, bounds and prior terms for the two fit modes.

Model space is what the likelihood consumes: bounded, well-conditioned, and
holding *offsets* for the two time-like parameters (see
:mod:`turin.prep`). anvil's ``ParamSpec``/``Transform`` maps it to and from
the unconstrained space the samplers work in, and carries the float64
``report_offset`` that turns offsets back into absolute periods and epochs.

Sampled parameters, following hurin:

LinEph (7)
    ``dP``, ``dtau0`` (offsets from the recentred ephemeris), ``k``,
    ``beta`` (the impact-parameter fraction), ``T14``, ``q1``, ``q2``.

TTV (5 + one per epoch)
    ``k``, ``beta``, ``T14``, ``q1``, ``q2``, then ``dtau_<n>`` per epoch --
    an offset from that epoch's predicted time. The period is held at the
    recentred ephemeris value, as in hurin.

``b`` is sampled as a fraction of its k-dependent upper bound rather than
directly, which is how a fixed box in model space can represent hurin's
conditional-uniform (b, k) construction. The area and taper terms that make
the joint prior what hurin intends live in :func:`bk_log_prior`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import mlx.core as mx
import numpy as np

from .model import B_PRIORS

#: Half-width of the dP and dtau0 priors, in days. hurin's values; kept
#: because a uniform prior shifted with its bounds has identical
#: unconstrained coordinates, so widths are the one thing that changes them.
DP_HALF = 0.1
DTAU0_HALF = 0.1

#: Kipping (2025) soft TTV prior: a quadratic ramp in ln|TTV/P| above a
#: threshold, free below it. Keep in sync with :mod:`turin.seeding`, which
#: replicates this on the host when scoring template-sweep seeds.
TTV_LOG_THRESHOLD = -7.34
TTV_LOG_SCALE = 1.71

LINEPH_BASE = ("dP", "dtau0", "k", "beta", "T14", "q1", "q2")
TTV_BASE = ("k", "beta", "T14", "q1", "q2")


@dataclass
class ParamLayout:
    """Names, bounds and reporting offsets for one fit mode."""

    mode: str                    # "lineph" or "ttv"
    names: tuple                 # model-space parameter names, in order
    lo: np.ndarray               # (dim,) lower bounds
    hi: np.ndarray               # (dim,) upper bounds
    report_offset: np.ndarray    # (dim,) added in float64 when reporting
    b_prior: str
    P_ref: float
    tau0_ref: float
    #: (n_epochs,) predicted-time offsets d_arr, and integer epoch numbers
    n_arr: np.ndarray = field(default_factory=lambda: np.zeros(0))
    centers_abs: np.ndarray = field(default_factory=lambda: np.zeros(0))

    @property
    def dim(self) -> int:
        return len(self.names)

    def index(self, name) -> int:
        return self.names.index(name)

    @property
    def n_epochs(self) -> int:
        return len(self.n_arr)

    def specs(self):
        """anvil ``ParamSpec`` list for this layout (imported lazily)."""
        import anvil

        return [
            anvil.ParamSpec(name, lo=float(self.lo[i]), hi=float(self.hi[i]),
                            report_offset=float(self.report_offset[i]))
            for i, name in enumerate(self.names)
        ]

    def transform(self):
        """anvil ``Transform`` for this layout."""
        import anvil

        return anvil.Transform(self.specs())


def _shape_bounds(eph, k_min=0.0, k_max=1.0, T14_max=None):
    """Bounds for the five shape parameters, in hurin's order after (dP, dtau0)."""
    if T14_max is None:
        # hurin: three times the archive duration, which is generous but
        # finite; callers may override from a physical density limit
        T14_max = 3.0 * eph["duration"] / 24.0
    return (
        ("k", k_min, k_max),
        ("beta", 0.0, 1.0),
        ("T14", 0.0, float(T14_max)),
        ("q1", 0.0, 1.0),
        ("q2", 0.0, 1.0),
    )


def lineph_layout(eph, b_prior="transiting", T14_max=None):
    """Parameter layout for the linear-ephemeris fit."""
    if b_prior not in B_PRIORS:
        raise ValueError(f"unknown b_prior {b_prior!r}; expected {B_PRIORS}")
    rows = [("dP", -DP_HALF, DP_HALF), ("dtau0", -DTAU0_HALF, DTAU0_HALF)]
    rows += list(_shape_bounds(eph, T14_max=T14_max))
    names = tuple(r[0] for r in rows)
    lo = np.array([r[1] for r in rows], dtype=np.float64)
    hi = np.array([r[2] for r in rows], dtype=np.float64)
    off = np.zeros(len(rows))
    off[0] = float(eph["period"])     # dP reports as an absolute period
    off[1] = float(eph["epoch"])      # dtau0 reports as an absolute epoch
    return ParamLayout(mode="lineph", names=names, lo=lo, hi=hi,
                       report_offset=off, b_prior=b_prior,
                       P_ref=float(eph["period"]),
                       tau0_ref=float(eph["epoch"]))


def ttv_layout(eph, centering, tau_half, b_prior="transiting", T14_max=None):
    """Parameter layout for the per-epoch-timing fit.

    ``tau_half`` is the half-width of each epoch's timing prior, in days:
    either ``--TTVmax`` or the data-driven extent, one value per epoch.
    """
    if b_prior not in B_PRIORS:
        raise ValueError(f"unknown b_prior {b_prior!r}; expected {B_PRIORS}")
    rows = list(_shape_bounds(eph, T14_max=T14_max))
    n_arr = np.asarray(centering["n_arr"], dtype=np.float64)
    centers = np.asarray(centering["centers_abs"], dtype=np.float64)
    tau_half = np.broadcast_to(np.asarray(tau_half, dtype=np.float64),
                               (len(n_arr),)).copy()
    if np.any(tau_half <= 0):
        raise ValueError("every epoch needs a positive timing prior half-width")

    names = tuple(r[0] for r in rows)
    lo = [r[1] for r in rows]
    hi = [r[2] for r in rows]
    off = [0.0] * len(rows)
    for i, n in enumerate(n_arr):
        # dtau_i is an offset from epoch i's PREDICTED time, so it reports as
        # an absolute mid-transit time
        names = names + (f"dtau_{int(round(n))}",)
        lo.append(-float(tau_half[i]))
        hi.append(float(tau_half[i]))
        off.append(float(eph["epoch"] + n * eph["period"]))

    return ParamLayout(
        mode="ttv", names=names, lo=np.array(lo), hi=np.array(hi),
        report_offset=np.array(off), b_prior=b_prior,
        P_ref=float(eph["period"]), tau0_ref=float(eph["epoch"]),
        n_arr=n_arr, centers_abs=centers)


def bk_log_prior(beta, k, b_prior="transiting"):
    """Prior terms that a box in (beta, k) does not already provide.

    Sampling ``beta ~ U(0, 1)`` and setting ``b = beta * bound(k)`` gives
    ``p(b | k) = 1 / bound(k)``, i.e. conditionally uniform, matching hurin's
    ``dist.Uniform(0, bound(k))``. What remains is:

    ``transiting``
        ``+log1p(k)`` makes the *joint* (b, k) density uniform over the
        transiting region rather than merely conditionally uniform, and the
        linear grazing taper ``log1p(-u)``, ``u = (b - (1-k)) / 2k``, halves
        the grazing band. Together these give exactly 1:1 marginal prior odds
        grazing to non-grazing and a uniform marginal ``p(k)``
        (the band has width ``2k`` and the taper integrates to 1/2 over it,
        contributing ``k``, against ``1-k`` non-grazing).
    ``nongrazing``
        ``+log1p(-k)``, the area of the ``b < 1-k`` triangle.
    ``box``
        nothing: independent uniforms, so the box is the whole prior.

    Returns a term of shape ``(n_chains,)``.
    """
    if b_prior == "box":
        return mx.zeros(beta.shape[:-1] if beta.ndim > 1 else beta.shape,
                        dtype=beta.dtype)

    k_flat = k.reshape(-1)
    beta_flat = beta.reshape(-1)
    if b_prior == "nongrazing":
        return mx.log1p(-k_flat)
    if b_prior == "transiting":
        area = mx.log1p(k_flat)
        # b = beta*(1+k), so u = (beta*(1+k) - (1-k)) / (2k)
        b = beta_flat * (1.0 + k_flat)
        u = (b - (1.0 - k_flat)) / (2.0 * mx.maximum(k_flat, 1e-12))
        # the clip floors the density at 1e-6 exactly at b = 1+k, keeping the
        # log and its gradient finite; mx.where evaluates both branches, so
        # the clip must be inside it
        taper = mx.where(u > 0.0,
                         mx.log1p(-mx.clip(u, 0.0, 1.0 - 1e-6)),
                         mx.zeros_like(u))
        return area + taper
    raise ValueError(f"unknown b_prior {b_prior!r}; expected {B_PRIORS}")


def ttv_log_prior(dtau, period):
    """Kipping (2025) soft prior on the timing offsets, shape ``(n_chains,)``.

    A quadratic ramp in ``ln|TTV/P|`` above ``-7.34`` with scale ``1.71``,
    and free below it -- i.e. no penalty for ``|TTV/P|`` under about 6.5e-4.
    ``dtau`` is already the deviation from the predicted time, so it *is* the
    TTV. ``period`` may be a scalar or ``(n_chains, 1)``.
    """
    log_abs_rel = mx.log(mx.abs(dtau / period) + 1e-30)
    excess = mx.maximum(log_abs_rel - TTV_LOG_THRESHOLD, 0.0)
    return -0.5 * mx.sum((excess / TTV_LOG_SCALE) ** 2, axis=-1)


def ttv_log_prior_np(dtau, period):
    """Float64 host replica of :func:`ttv_log_prior`, for seed scoring."""
    log_abs_rel = np.log(np.abs(np.asarray(dtau, dtype=np.float64) / period)
                         + 1e-30)
    excess = np.maximum(log_abs_rel - TTV_LOG_THRESHOLD, 0.0)
    return -0.5 * np.sum((excess / TTV_LOG_SCALE) ** 2, axis=-1)


def density_T14_max(period, k, b, rho_min_cgs=0.02):
    """Largest physically plausible T14, from a minimum stellar density.

    hurin falls back to this when the archive duration is missing. Inverting
    Kepler's third law for a circular orbit at density ``rho_min``:
    ``a/R* = ((G rho P^2) / (3 pi))^(1/3)``, then S&MO for the duration.
    """
    G_cgs = 6.67430e-8
    P_sec = float(period) * 86400.0
    aRs = ((G_cgs * rho_min_cgs * P_sec**2) / (3.0 * math.pi)) ** (1.0 / 3.0)
    num = max((1.0 + k) ** 2 - b**2, 1e-12)
    den = max(aRs**2 - b**2, num)
    return float(period) / math.pi * math.asin(min(1.0, math.sqrt(num / den)))
