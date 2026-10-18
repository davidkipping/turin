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
3. Under ``--oblate``, the round model is fitted first, the oblate search
   starts from its optimum at f ~ 0, and the two are compared.

The result is the best fit and nothing else: no uncertainties (the user's
decision -- an MLE is the best fit, error bars come from sampling).
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
#: Product mode stem: ``{target}_lineph_mle_{name}.{ext}``, apart from any
#: MCMC lineage's ``{target}_lineph_*``.
MODE = "lineph_mle"


@dataclass
class MleFit:
    """One maximum-likelihood fit. ``rows``: (name, value) in physical
    units, the parameters (limb darkening included) then the derived ones."""

    label: str
    layout: object               # theta-only (profiled) layout
    joint_layout: object         # theta + q layout (the rows' names)
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
        return dict(self.rows)[name]


#: An oblate search starts from the round optimum (the user's scheme): its
#: geometry plus this sd in u, and f = F_FLOOR (just off f = 0, where
#: (fc, fs) feel no gradient) at random angles. Measured against a ball
#: around the input ephemeris (alone, and with 8 round seeds), 3 seeds
#: each: on six PHOENIX truths (f = 0, 0.1; theta 0-60; b 0.3, 0.7) all
#: reach the same ln L (within 0.01), this one 20-40% faster; on the user's
#: five Kepler targets (f unconstrained, optima on a flat ridge at the
#: F_MAX edge) it is best on KOI-1430.03 (7957.49 against the ball's
#: 7956.95-7957.43, which once ended *below* the round fit), level on
#: 4087.01 and 1361.01, and 0.04-0.27 short of the ball's best seed on
#: 2650.01 and 4016.01. It cannot end below the round fit.
ROUND_SEED_SD = 0.02


def fit(prepared, centering, epoch_data, orders, *, law, oblate, b_prior,
        build_kw, seed=0, log=print, round_fit=None):
    """Fit one model; returns an :class:`MleFit`. An oblate fit needs the
    round one (``round_fit``): every start is its optimum, at f =
    ``F_FLOOR`` (see :data:`ROUND_SEED_SD`)."""
    import anvil

    label = ("oblate" if oblate else "round") + f" {law}"
    eph = prepared.eph
    kw = dict(build_kw, profile_mode="exact", likelihood_only=True)
    lay = _params.lineph_layout(eph, b_prior=b_prior, law=law, oblate=oblate,
                                ld="collapsed")
    tr, T, lo, hi = _likelihood.build_target(lay, centering, epoch_data,
                                             orders, **kw)
    rng = np.random.default_rng(seed)
    if oblate:
        if round_fit is None:
            raise ValueError("an oblate maximum-likelihood fit starts from "
                             "the round one: pass round_fit")
        ang = rng.uniform(0.0, 2.0 * math.pi, N_STARTS)
        V = np.tile(np.r_[round_fit.v_model[:5], 0.0, 0.0], (N_STARTS, 1))
        r = math.sqrt(_seeding.F_FLOOR)
        V[:, lay.index("fc")] = r * np.cos(ang)
        V[:, lay.index("fs")] = r * np.sin(ang)
        starts = T.from_model_np(V)
        starts[:, :5] += ROUND_SEED_SD * rng.standard_normal((N_STARTS, 5))
    else:
        v0 = _seeding.initial_model_vector(lay, eph)
        starts = (T.from_model_np(v0)[None, :]
                  + 0.25 * rng.standard_normal((N_STARTS, lay.dim)))
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

    jlay = _params.lineph_layout(eph, b_prior=b_prior, law=law, oblate=oblate)
    names = list(jlay.names)
    v_best = np.r_[res.model, q]
    best_phys = jlay.transform().to_physical(v_best[None, :])
    rows = [(n, float(best_phys[0, i])) for i, n in enumerate(names)]
    rows += [(n, val) for n, val in _derived(jlay, best_phys, names)]
    notes = []
    if oblate:
        f_best = dict(rows)["f"]
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


def _derived(layout, best_phys, names):
    """(name, value) for the derived rows, at the best fit."""
    from .pipeline import _ld_weight_draws, _oblate_draws

    out = [("b", float(_outputs.derived_b(best_phys, names,
                                          layout.b_prior)[0])),
           ("log10_rho", float(_outputs.log10_rho_draws(
               best_phys, names, layout.b_prior)[0]))]
    for fn in (_ld_weight_draws, _oblate_draws):
        n0, d0 = fn(layout, best_phys, names)
        out += [(n, float(d0[0, j])) for j, n in enumerate(n0)]
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
