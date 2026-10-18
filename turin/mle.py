"""Maximum-likelihood fits (``--mle``): LinEph, round or oblate.

The **likelihood alone** is maximised: turin's priors (the b-k grazing
taper, ``--rhostar``) are off, and only hard domains stay -- the parameter
boxes, the ``b_prior`` geometry ``b = beta (1 +/- k)``, and the oblate
domain ``k < (1-f)^1.5``, ``f <= F_MAX`` (``likelihood_only=True``).

The limb darkening is **profiled**: each law's light curve is a convex mix
of generator curves, so at every theta the best limb darkening is a small
QP over the simplex (:class:`turin.ldmarg.ProfileLDLogProb`,
:class:`turin.ldhybrid.ProfileHybridLogProb`). That is exact for a
maximum, ``max_{theta,x} L = max_theta max_x L(theta, x)``, and reaches a
limb darkening on the simplex's faces directly, which the joint search in
``q`` approaches only as ``u`` runs off to infinity.

1. ``anvil.optimize`` (``objective="mle"``) over theta from
   :data:`N_STARTS` starts around the input ephemeris.
2. The limb darkening at the optimum, ``q`` from the profile's ``x*``.
3. Uncertainties from the **joint** likelihood over (theta, q) at that
   point: ``anvil.optimize`` started there converges in place and returns
   its curvature. The reported sds are Laplace sds, from
   :data:`N_LAPLACE_DRAWS` draws of that Gaussian in u mapped through the
   transform, so derived rows (``b``, ``log10_rho``, ``w``, ``f``, ...) get
   them the same way. A parameter whose optimum is on a bound has none, and
   the others are then the sds on that face.
4. Under ``--oblate``, the round model is fitted too and the two compared.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import mlx.core as mx
import numpy as np

from . import likelihood as _likelihood
from . import outputs as _outputs
from . import params as _params
from . import seeding as _seeding

#: Starts for the search (a 0.25 sd ball in u around the input ephemeris,
#: as the sampler's MAP search).
N_STARTS = 128
#: Laplace draws behind every reported sd.
N_LAPLACE_DRAWS = 4000
#: Product mode stem: ``{target}_lineph_mle_{name}.{ext}``, apart from any
#: MCMC lineage's ``{target}_lineph_*``.
MODE = "lineph_mle"


@dataclass
class MleFit:
    """One maximum-likelihood fit. ``rows``: (name, value, sd, at_bound) in
    physical units, the joint layout's parameters then the derived ones."""

    label: str
    layout: object               # theta-only (profiled) layout
    joint_layout: object         # theta + q layout
    lp: object                   # float32 profiled likelihood (for curves)
    v_model: np.ndarray          # joint model-space best fit
    rows: list
    lnL: float
    n_points: int
    n_params: int
    n_starts_converged: int
    n_starts: int
    top_spread: float
    notes: list = field(default_factory=list)

    @property
    def bic(self):
        return self.n_params * math.log(self.n_points) - 2.0 * self.lnL

    @property
    def aic(self):
        return 2.0 * self.n_params - 2.0 * self.lnL

    def value(self, name):
        return next(r[1] for r in self.rows if r[0] == name)


#: Seeds an oblate search gets from the round optimum: f = F_FLOOR at this
#: many angles. Without them the search can miss the nested round solution
#: where the likelihood is nearly flat in f: KOI-1430.03 (Kepler long
#: cadence) ended 0.076 nats *below* the round fit, at f = 0.598.
N_ROUND_SEEDS = 8


def fit(prepared, centering, epoch_data, orders, *, law, oblate, b_prior,
        build_kw, seed=0, log=print, round_fit=None):
    """Fit one model; returns an :class:`MleFit`. An oblate fit given the
    round one (``round_fit``) also starts from it, at f = ``F_FLOOR``, so
    it cannot end below it."""
    import anvil

    label = ("oblate" if oblate else "round") + f" {law}"
    eph = prepared.eph
    kw = dict(build_kw, profile_mode="exact", likelihood_only=True)
    lay = _params.lineph_layout(eph, b_prior=b_prior, law=law, oblate=oblate,
                                ld="collapsed")
    tr, T, lo, hi = _likelihood.build_target(lay, centering, epoch_data,
                                             orders, **kw)
    rng = np.random.default_rng(seed)
    v0 = _seeding.initial_model_vector(lay, eph)
    starts = (T.from_model_np(v0)[None, :]
              + 0.25 * rng.standard_normal((N_STARTS, lay.dim)))
    if oblate and round_fit is not None:
        r = math.sqrt(_seeding.F_FLOOR)
        ang = 2.0 * math.pi * np.arange(N_ROUND_SEEDS) / N_ROUND_SEEDS
        V = np.tile(np.r_[round_fit.v_model[:5], 0.0, 0.0], (N_ROUND_SEEDS, 1))
        V[:, lay.index("fc")], V[:, lay.index("fs")] = r * np.cos(ang), \
            r * np.sin(ang)
        starts = np.concatenate([starts, T.from_model_np(V)])
    res = anvil.optimize(tr, starts, objective="mle", seed=seed,
                         max_rows=_seeding.MAX_ROWS)
    n_conv = int(np.sum(res.status == "converged"))
    log(f"  [{label}] ln L {res.log_value:.3f} from {len(starts)} starts "
        f"({n_conv} converged) in {res.wall_time:.1f}s")

    # the limb darkening at the optimum: the profile's own maximiser
    with mx.stream(mx.cpu):
        q = hi.conditional_mode_ld(mx.array(res.model[None, :],
                                            dtype=mx.float64))
    q = np.array([float(np.asarray(c)[0]) for c in q])

    # the joint likelihood there: in place, for the curvature
    jlay = _params.lineph_layout(eph, b_prior=b_prior, law=law, oblate=oblate)
    jtr, jT, _, _ = _likelihood.build_target(jlay, centering, epoch_data,
                                             orders, **kw)
    v_joint = np.r_[res.model, np.clip(q, 1e-7, 1.0 - 1e-7)]
    joint = anvil.optimize(jtr, jT.from_model_np(v_joint), n_starts=8,
                           start_spread=1e-4, objective="mle", seed=seed,
                           max_rows=_seeding.MAX_ROWS)
    notes = []
    if joint.log_value > res.log_value + 0.01:
        # the profile is exact, so this should not happen; say so if it does
        notes.append(f"the joint search found ln L {joint.log_value:.3f}, "
                     f"{joint.log_value - res.log_value:.3f} above the profile")

    at_bound = np.asarray(joint.at_bound, dtype=bool)
    draws, held, why = _laplace_draws(joint, jT, jlay, at_bound, rng)
    if why:
        notes.append(why)
    names = list(jlay.names)
    v_best = np.r_[res.model, q]
    best_phys = jT.to_physical(v_best[None, :])
    phys = jT.to_physical(draws) if draws is not None else None

    rows = []
    for i, n in enumerate(names):
        sd = (float(np.std(phys[:, i])) if phys is not None and not held[i]
              else float("nan"))
        rows.append((n, float(best_phys[0, i]), sd, bool(at_bound[i])))
    held_names = {n for n, h in zip(names, held) if h}
    for n, val, d in _derived(jlay, best_phys, phys, names):
        # a derived row inherits "no sd" from any held parameter it uses
        sd = (float(np.std(d)) if d is not None
              and not (_DERIVED_FROM(n, jlay) & held_names) else float("nan"))
        rows.append((n, val, sd, False))

    if oblate:
        f_best = next(r[1] for r in rows if r[0] == "f")
        if f_best > 0.99 * _params.F_MAX:
            notes.append(f"f = {f_best:.3f} is at the edge of its range "
                         f"(F_MAX = {_params.F_MAX}): the likelihood barely "
                         f"tells f apart here, so its value means little; "
                         f"see the round-vs-oblate comparison")
    return MleFit(
        label=label, layout=lay, joint_layout=jlay, lp=lo, v_model=v_best,
        rows=rows, lnL=float(res.log_value), n_points=int(lo.n_real),
        n_params=int(jlay.dim + np.sum(np.asarray(orders) + 1)),
        n_starts_converged=n_conv, n_starts=len(starts),
        top_spread=_seeding.top_spread(res), notes=notes)


#: Smallest curvature (nats per unit u squared) a direction must have for
#: a Laplace sd: below it the sd in u exceeds 2, most of a bounded
#: parameter's box. KOI-518.02 (b ~ 0, where L is flat in b) measured
#: -0.02 and +0.05 along beta (with q2), float32 noise; its constrained
#: directions are 1e3-1e5.
MIN_CURVATURE = 0.25


def _laplace_draws(joint, transform, layout, at_bound, rng):
    """``(draws, held, note)``: ``N_LAPLACE_DRAWS`` model-space draws from
    the Laplace Gaussian in u at the joint optimum, and the parameters held
    at the optimum instead of drawn.

    Held: any on a bound, and -- one at a time, until the curvature is
    negative definite with every direction above :data:`MIN_CURVATURE` --
    the parameter loading most on the flattest direction. That is the
    likelihood being flat (b near 0) or not concave (an oblate fit of a
    round planet: f = fc^2 + fs^2 is stationary at f = 0) along it, where
    the curvature says nothing about the width. The other sds are then
    conditional on the held values."""
    held = at_bound.copy()
    H = joint.hessian_u
    if H is None or not np.all(np.isfinite(H[np.ix_(~held, ~held)])):
        return None, held, "no curvature at the optimum: sds not available"
    names = list(layout.names)
    flat = []
    while (~held).any():
        free = np.flatnonzero(~held)
        A = -H[np.ix_(free, free)]
        lam, P = np.linalg.eigh(0.5 * (A + A.T))
        if lam[0] > MIN_CURVATURE:
            break
        j = free[int(np.argmax(np.abs(P[:, 0])))]
        held[j] = True
        flat.append(names[j])
    else:
        return None, held, "no direction is constrained: sds not available"
    note = None
    if flat:
        note = (f"no Laplace sd for {', '.join(flat)}: the likelihood is "
                f"flat or not concave there at the optimum (e.g. b near 0, "
                f"or f near 0 under --oblate); the other sds hold "
                f"{'it' if len(flat) == 1 else 'them'} at the optimum")
        if layout.oblate and {"fc", "fs"} & set(flat):
            note += ". The round-vs-oblate comparison measures the evidence for f"
    u = np.repeat(joint.u[None, :], N_LAPLACE_DRAWS, axis=0)
    z = rng.standard_normal((N_LAPLACE_DRAWS, free.size))
    u[:, free] += (z / np.sqrt(lam)) @ P.T
    return transform.model_np(u), held, note


def _DERIVED_FROM(name, layout):
    """The fitted parameters a derived row depends on."""
    if name == "b":
        return {"beta", "k"}
    if name == "log10_rho":
        return {"dP", "T14", "k", "beta"}
    if name.startswith("w"):
        return set(layout.ld_names)
    if name in ("f", "theta"):
        return {"fc", "fs"}
    return {"k", "fc", "fs"}                                   # A, B


def _derived(layout, best_phys, phys, names):
    """(name, value, draws or None) for the derived rows."""
    from .pipeline import _ld_weight_draws, _oblate_draws

    out = []
    b0 = _outputs.derived_b(best_phys, names, layout.b_prior)
    rho0 = _outputs.log10_rho_draws(best_phys, names, layout.b_prior)
    b = _outputs.derived_b(phys, names, layout.b_prior) if phys is not None \
        else None
    rho = (_outputs.log10_rho_draws(phys, names, layout.b_prior)
           if phys is not None else None)
    out += [("b", float(b0[0]), b), ("log10_rho", float(rho0[0]), rho)]
    for fn in (_ld_weight_draws, _oblate_draws):
        n0, d0 = fn(layout, best_phys, names)
        if not n0:
            continue
        _, d = fn(layout, phys, names) if phys is not None else (n0, None)
        out += [(n, float(d0[0, j]), None if d is None else d[:, j])
                for j, n in enumerate(n0)]
    return out


def compare(oblate_fit, round_fit):
    """The round-vs-oblate comparison: a dict of the reported numbers.

    The models are nested (f = 0), so ``dlnL >= 0`` up to the optimiser's
    tolerance. No p-value: at f = 0 theta is undefined and f sits on its
    boundary, so Wilks' chi-squared reference does not hold."""
    from .pipeline import (ALIGNED_B_MAX, ALIGNED_F_MIN, ALIGNED_THETA_MAX)

    dlnL = oblate_fit.lnL - round_fit.lnL
    n = oblate_fit.n_points
    aligned = (oblate_fit.value("b") < ALIGNED_B_MAX
               and abs(oblate_fit.value("theta")) < ALIGNED_THETA_MAX
               and oblate_fit.value("f") > ALIGNED_F_MIN)
    return dict(lnL_oblate=oblate_fit.lnL, lnL_round=round_fit.lnL,
                dlnL=dlnL, dBIC=2.0 * math.log(n) - 2.0 * dlnL,
                dAIC=4.0 - 2.0 * dlnL, aligned_central=bool(aligned),
                n_points=n)
