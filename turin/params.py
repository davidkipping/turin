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

Under ``--law=hybrid4`` the limb darkening is ``q1 .. q4`` instead of
``q1, q2`` (:func:`turin.model.ld_names`), so LinEph samples 9 and TTV
``7 + one per epoch``; :attr:`ParamLayout.n_shape` is where the epoch times
start.

Under ``--oblate`` (hybrid4; sampled LinEph only, TTV for ``--mle``) two
columns follow ``T14``:
``fc, fs = sqrt(f) (cos 2theta, sin 2theta)``, the projected flattening and
long-axis sky angle of an oblate planet (:func:`oblate_from_fcfs`).

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

from . import model as _model
from .model import B_PRIORS, LAWS, ld_names

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

#: How the quadratic limb darkening is handled. ``sampled`` (default) puts
#: ``q1, q2`` in the sampled vector. ``collapsed`` integrates it out of the
#: log-density and draws ``q1, q2`` afterwards from their conditional
#: (collapsed Gibbs; see :mod:`turin.ldmarg`), so they are not sampled.
LD_MODES = ("sampled", "collapsed")
#: The LinEph sampled vector under ``--ld=collapsed``.
LINEPH_BASE_COLLAPSED = ("dP", "dtau0", "k", "beta", "T14")

#: Largest projected flattening an oblate fit allows (SquishierPlanet's).
#: ``(fc, fs)`` live in the box ``[-sqrt(F_MAX), sqrt(F_MAX)]^2`` and the
#: prior cuts it to the disc ``f <= F_MAX``. Uniform on the disc is exactly
#: uniform in ``f`` and in ``theta`` (the ``sqrt(e) cos(omega)`` trick); the
#: linear form ``f (cos 2theta, sin 2theta)`` would be ``p(f) ~ f``, which
#: quadrupled SquishierPlanet's false-oblate weight. The 2theta is because an
#: ellipse is unchanged by a half-turn.
F_MAX = 0.6
OBLATE_NAMES = ("fc", "fs")
#: elome's oblate domain is ``k <= (1 - f)^1.5`` (at most two limb
#: crossings); outside it the kernel returns NaN. The likelihood refuses
#: points this close to the edge, before they reach the kernel.
OBLATE_DOMAIN_MARGIN = 1e-6


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
    #: limb-darkening mode (:data:`LD_MODES`). A ``collapsed`` layout has no
    #: ``q1, q2``; :func:`turin.likelihood.build_target` reads this to choose
    #: the likelihood class, so a 5-column layout can never reach the default
    #: class (whose ``unpack`` would slice an empty column, not raise).
    ld: str = "sampled"
    #: limb-darkening law (:data:`turin.model.LAWS`); it sets which
    #: coordinates the layout carries (:func:`turin.model.ld_names`)
    law: str = "quadratic"
    #: an oblate planet: ``fc, fs`` follow ``T14`` (hybrid4; a TTV layout
    #: only for ``--mle``, :func:`turin.likelihood.build_target`)
    oblate: bool = False
    #: Gaussian prior on log10(rho*/kg m^-3) as ``(mean, sigma)``; ``()``
    #: for none (:func:`rhostar_log_prior`)
    rhostar: tuple = ()

    @property
    def ld_names(self) -> tuple:
        """The law's sampled limb-darkening coordinates in this layout
        (none under ``--ld=collapsed``)."""
        return tuple(n for n in ld_names(self.law) if n in self.names)

    @property
    def n_shape(self) -> int:
        """Columns before the epoch times: ``k, beta, T14``, the oblate pair
        if any, and the limb darkening (5 quadratic, 7 hybrid4); a TTV
        vector's ``dtau`` start."""
        return 3 + 2 * self.oblate + len(self.ld_names)

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


def _shape_bounds(eph, k_min=0.0, k_max=1.0, T14_max=None, law="quadratic",
                  oblate=False):
    """Bounds for the shape parameters, in hurin's order after (dP, dtau0):
    ``k, beta, T14``, ``fc, fs`` if oblate, and the law's limb-darkening
    coordinates, each uniform on [0, 1] (Kipping's ``q1, q2``; hybrid4's
    curvature-basis ``q1..q4``, :func:`turin.model.ld_weights`)."""
    if T14_max is None:
        # hurin: three times the archive duration, which is generous but
        # finite; callers may override from a physical density limit
        T14_max = 3.0 * eph["duration"] / 24.0
    return (
        ("k", k_min, k_max),
        ("beta", 0.0, 1.0),
        ("T14", 0.0, float(T14_max)),
    ) + tuple((n, -math.sqrt(F_MAX), math.sqrt(F_MAX))
              for n in (OBLATE_NAMES if oblate else ())) \
        + tuple((n, 0.0, 1.0) for n in ld_names(law))


def lineph_layout(eph, b_prior="transiting", T14_max=None, ld="sampled",
                  law="quadratic", oblate=False, rhostar=()):
    """Parameter layout for the linear-ephemeris fit.

    ``ld="collapsed"`` drops the limb-darkening coordinates from the sampled
    vector: the limb darkening is integrated out of the log-density instead
    (:class:`turin.ldmarg.MarginalLDLogProb` for the quadratic law,
    :class:`turin.ldhybrid.MarginalHybridLogProb` for hybrid4).
    """
    if b_prior not in B_PRIORS:
        raise ValueError(f"unknown b_prior {b_prior!r}; expected {B_PRIORS}")
    if ld not in LD_MODES:
        raise ValueError(f"unknown ld mode {ld!r}; expected {LD_MODES}")
    if law not in LAWS:
        raise ValueError(f"unknown limb-darkening law {law!r}; expected {LAWS}")
    if ld == "collapsed" and law not in ("quadratic", "hybrid4"):
        raise ValueError(f"--ld=collapsed is not implemented for {law}")
    if oblate and law != "hybrid4":
        raise ValueError(f"an oblate planet takes --law=hybrid4, not {law}")
    rows = [("dP", -DP_HALF, DP_HALF), ("dtau0", -DTAU0_HALF, DTAU0_HALF)]
    rows += list(_shape_bounds(eph, T14_max=T14_max, law=law, oblate=oblate))
    if ld == "collapsed":
        rows = [r for r in rows if r[0] not in ld_names(law)]
    names = tuple(r[0] for r in rows)
    lo = np.array([r[1] for r in rows], dtype=np.float64)
    hi = np.array([r[2] for r in rows], dtype=np.float64)
    off = np.zeros(len(rows))
    off[0] = float(eph["period"])     # dP reports as an absolute period
    off[1] = float(eph["epoch"])      # dtau0 reports as an absolute epoch
    return ParamLayout(mode="lineph", names=names, lo=lo, hi=hi,
                       report_offset=off, b_prior=b_prior,
                       P_ref=float(eph["period"]),
                       tau0_ref=float(eph["epoch"]), ld=ld, law=law,
                       oblate=bool(oblate), rhostar=_rhostar(rhostar))


def ttv_layout(eph, centering, tau_half, b_prior="transiting", T14_max=None,
               law="quadratic", oblate=False, rhostar=(), ld="sampled"):
    """Parameter layout for the per-epoch-timing fit.

    ``tau_half`` is the half-width of each epoch's timing prior, in days:
    either ``--TTVmax`` or the data-driven extent, one value per epoch.

    ``ld="collapsed"`` drops the limb-darkening coordinates, as in
    :func:`lineph_layout`, for the **profiled** likelihood of ``--mle``
    only (:class:`turin.ldmarg.ProfileLDLogProb`,
    :class:`turin.ldhybrid.ProfileHybridLogProb`). Sampling refuses it: an
    integrated-out limb darkening couples every epoch, which breaks the
    per-epoch factorisation grid-Gibbs relies on.

    ``oblate=True`` (hybrid4) is for ``--mle`` too: ``build_target`` refuses
    to sample it (see there).
    """
    if b_prior not in B_PRIORS:
        raise ValueError(f"unknown b_prior {b_prior!r}; expected {B_PRIORS}")
    if law not in LAWS:
        raise ValueError(f"unknown limb-darkening law {law!r}; expected {LAWS}")
    if ld not in LD_MODES:
        raise ValueError(f"unknown ld mode {ld!r}; expected {LD_MODES}")
    if ld == "collapsed" and law not in ("quadratic", "hybrid4"):
        raise ValueError(f"--ld=collapsed is not implemented for {law}")
    if oblate and law != "hybrid4":
        raise ValueError(f"an oblate planet takes --law=hybrid4, not {law}")
    rows = list(_shape_bounds(eph, T14_max=T14_max, law=law, oblate=oblate))
    if ld == "collapsed":
        rows = [r for r in rows if r[0] not in ld_names(law)]
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
        n_arr=n_arr, centers_abs=centers, ld=ld, law=law,
        oblate=bool(oblate), rhostar=_rhostar(rhostar))


def _rhostar(prior):
    """``(mean, sigma)`` as floats, or ``()`` for no prior."""
    if not prior:
        return ()
    mean, sigma = (float(x) for x in prior)
    if not sigma > 0:
        raise ValueError(f"the rho* prior needs sigma > 0, got {sigma}")
    return (mean, sigma)


def rhostar_log_prior(T14, period, k, b, prior):
    """``-0.5 ((log10 rho* - mean)/sigma)^2``, shape ``(n_chains,)``: a
    Gaussian prior on the stellar density implied by the fitted transit
    (``model.log10_rho_mx``, the ``logrho`` product's quantity). Its
    normalisation is a constant and is left out."""
    mean, sigma = prior
    z = (_model.log10_rho_mx(T14, period, k, b) - mean) / sigma
    return (-0.5 * z * z)[:, 0]


def oblate_from_fcfs(fc, fs):
    """``(f, theta)`` from the sampled ``(fc, fs)``: ``f = fc^2 + fs^2``,
    ``theta = atan2(fs, fc) / 2`` in radians, on [-pi/2, pi/2] (MLX)."""
    return fc * fc + fs * fs, 0.5 * mx.arctan2(fs, fc)


def oblate_from_fcfs_np(fc, fs):
    """Host replica of :func:`oblate_from_fcfs`, with ``theta`` in degrees
    (the products' unit; elome's ``TransitModel`` also takes degrees)."""
    fc = np.asarray(fc, dtype=np.float64)
    fs = np.asarray(fs, dtype=np.float64)
    return fc * fc + fs * fs, np.degrees(0.5 * np.arctan2(fs, fc))


def oblate_ok(k, f):
    """Where an oblate point is allowed: inside the disc ``f <= F_MAX`` and
    elome's domain ``k <= (1 - f)^1.5`` (with a hair of margin), as a mask
    from the parameters alone, so the likelihood can keep the kernel away
    from NaN rather than clean it up afterwards (MLX)."""
    edge = (1.0 - mx.minimum(f, 1.0)) ** 1.5 * (1.0 - OBLATE_DOMAIN_MARGIN)
    return (f <= F_MAX) & (k < edge)


def bk_log_prior(beta, k, b_prior="transiting"):
    """Prior terms that a box in (beta, k) does not already provide.

    Sampling ``beta ~ U(0, 1)`` and setting ``b = beta * bound(k)`` gives
    ``p(b | k) = 1 / bound(k)``, i.e. conditionally uniform, matching hurin's
    ``dist.Uniform(0, bound(k))``. What remains is:

    ``transiting``
        ``+log1p(k)`` makes the *joint* (b, k) density uniform over the
        transiting region rather than merely conditionally uniform, and the
        linear grazing taper ``log1p(-u)``, ``u = (b - (1-k)) / 2k``, halves
        the grazing band. Together these give a uniform marginal ``p(k)``:
        at each ``k`` the band has width ``2k`` and the taper integrates to
        1/2 over it, so it carries mass exactly ``k`` against ``1-k``
        non-grazing, summing to 1 for every ``k``. Untapered, ``p(k)`` would
        go as ``1+k``.

        The grazing odds are therefore ``k/(1-k)`` **at fixed k**, and 1:1
        only *marginally*, because ``k`` and ``1-k`` each integrate to 1/2
        over the default ``k ~ U(0, 1)``. Both are pinned by tests; keep them
        apart when quoting, since at ``k = 0.02`` grazing is ~2% of the prior
        mass, not half of it.
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
