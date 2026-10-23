"""Maximum-likelihood fits (``--mle``): LinEph, round or oblate, and TTV.

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
   :data:`N_STARTS` starts around the input ephemeris, half of them with b
   spread over its whole range (:func:`round_starts`).
2. The limb darkening at the optimum, ``q`` from the profile's ``x*``.
3. Under ``--oblate``, the round model is fitted first, the oblate search
   starts from its optimum at f ~ 0, and the two are compared.
4. TTV (:func:`fit_ttv`) starts from the LinEph fit and alternates an exact
   per-epoch timing step (:func:`timing_step`: every epoch's whole box on a
   grid, the limb darkening held) with a joint polish, until a timing step
   moves no epoch.

The result is the best fit and nothing else: no uncertainties (the user's
decision -- an MLE is the best fit, error bars come from sampling).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import mlx.core as mx
import numpy as np

from . import likelihood as _likelihood
from . import outputs as _outputs
from . import params as _params
from . import seeding as _seeding

#: anvil.optimize's Laplace step after the search. Off: --mle reports the
#: best fit only, and the step can raise. A TTV epoch with no transit data
#: has an exactly zero Hessian row; rounding gives eigh a smallest eigenvalue
#: of +1.8e-12, so anvil's negative-definite check passes and its inverse
#: raises LinAlgError (KOI-4087.01, anvil 0.10.1).
HESSIAN = False

#: Starts for the round search: a 0.25 sd ball in u around the input
#: ephemeris, as the sampler's MAP search, with :data:`N_SPREAD_BETA` of
#: them moved to beta spread evenly over (0, 1).
N_STARTS = 128
#: The ball alone missed the best fit on two of the user's five targets
#: (0.1.74, against every draw of a LinEph MCMC run, on the exact likelihood
#: alone): KOI-4016.01 and KOI-1430.03 hold a second peak at b ~ 0.85-0.88
#: with weak limb darkening, above the central chord with limb darkening on
#: a corner where all 128 starts landed, by 0.33 and 0.10 nats. Low SNR lets
#: b and the limb darkening trade off, so the starts must cover b.
N_SPREAD_BETA = 64
#: Product mode stem: ``{target}_lineph_mle_{name}.{ext}``, apart from any
#: MCMC lineage's ``{target}_lineph_*``.
MODE = "lineph_mle"
#: The TTV fit's stem: {target}_ttv_mle_{name}.{ext}.
MODE_TTV = "ttv_mle"


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
    #: TTV: the sampled-LD likelihood (float32), for per-epoch terms, which
    #: the profiled one cannot give (a shared LD couples the epochs)
    lp_joint: object = None

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
        starts = round_starts(lay, T, eph, rng)
    res = anvil.optimize(tr, starts, objective="mle", seed=seed,
                         max_rows=_seeding.MAX_ROWS, hessian=HESSIAN)
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
    notes = _f_edge_notes(rows) if oblate else []
    return MleFit(
        label=label, layout=lay, joint_layout=jlay, lp=lo, v_model=v_best,
        rows=rows, lnL=float(res.log_value), n_points=int(lo.n_real),
        n_params=int(jlay.dim + np.sum(np.asarray(orders) + 1)),
        n_starts_converged=n_conv, n_starts=len(starts),
        top_spread=_seeding.top_spread(res), notes=notes)


def _f_edge_notes(rows):
    """An oblate fit's note when f sits at the edge of its range."""
    f_best = dict(rows)["f"]
    if f_best > 0.99 * _params.F_MAX:
        return [f"f = {f_best:.3f} is at the edge of its range (F_MAX = "
                f"{_params.F_MAX}): the likelihood barely tells f apart "
                f"here, so its value means little; see the round-vs-oblate "
                f"comparison"]
    return []


def round_starts(layout, transform, eph, rng):
    """The round search's starts, in u: :data:`N_STARTS` in a 0.25 sd ball
    around the input ephemeris, the first :data:`N_SPREAD_BETA` with beta
    stratified over (0, 1) (one uniform draw per stratum, kept 0.005 off the
    box ends, where u is infinite)."""
    v0 = _seeding.initial_model_vector(layout, eph)
    u = (transform.from_model_np(v0)[None, :]
         + 0.25 * rng.standard_normal((N_STARTS, layout.dim)))
    V = transform.model_np(u)
    n = N_SPREAD_BETA
    V[:n, layout.index("beta")] = np.clip(
        (np.arange(n) + rng.uniform(size=n)) / n, 0.005, 0.995)
    return transform.from_model_np(V)


#: Grid cells per epoch for the TTV timing step: as grid-Gibbs, ~1/115 of
#: the default box (+-4.45 T14), refined by :data:`N_ZOOM` points over
#: +-1.5 cells around the best one.
N_TIMING_GRID = 512
N_ZOOM = 64
#: An epoch whose term moves less than this over its whole box has no
#: transit data its time could explain: the time is arbitrary.
FLAT_NATS = 0.01
#: An epoch counts as moved (and so blocks the certificate) only if its
#: term rose by more than this; a smaller rise is still taken. The polish
#: stops at anvil's tolerance, so each time sits a hair off its own best,
#: and without a floor the certificate never fired on KOI-315.01 (oblate
#: hybrid4: five passes, each moving 1-6 epochs for +0.000 nats).
MOVE_NATS = 1e-3


@dataclass
class TimingStep:
    """One exact timing step: ``v`` with every epoch time at its best."""

    v: np.ndarray            # model-space vector after the step
    moved: np.ndarray        # (E,) bool: its term rose by > MOVE_NATS
    gain: np.ndarray         # (E,) float64 ln L gained per epoch (>= 0)
    flat: np.ndarray         # (E,) bool: term flat over the box


def timing_step(lp32, lp64, layout, v, *, n_grid=N_TIMING_GRID,
                n_zoom=N_ZOOM):
    """Put every epoch's time at its best given everything else.

    ``lp32``, ``lp64``: the sampled-LD TTV likelihood
    (:class:`turin.likelihood.ProfiledTransitLogProb`) in float32 and
    float64; ``layout`` its layout; ``v`` ``(dim,)`` in model space, the
    limb darkening held at its current value. Given shape and limb
    darkening each epoch's term depends on its own time alone, so one
    batched :meth:`epoch_log_lik` call scores every epoch's grid over its
    whole box (as :class:`turin.gibbs.GridGibbs` does, no prior here). The
    best cell is refined on a finer grid, and an epoch moves only where
    float64 says its term rose, so the step never lowers ln L. It counts as
    ``moved`` only above :data:`MOVE_NATS`. Returns a :class:`TimingStep`."""
    s = layout.n_shape
    lo, hi = layout.lo[s:], layout.hi[s:]
    E = layout.n_epochs
    width = (hi - lo) / n_grid
    mids = lo[:, None] + width[:, None] * (np.arange(n_grid)[None, :] + 0.5)

    def terms32(T):                          # T (G, E) -> (G, E)
        V = np.repeat(np.asarray(v, dtype=np.float64)[None, :], T.shape[0], 0)
        V[:, s:] = T
        return np.array(lp32.epoch_log_lik(mx.array(V, dtype=mx.float32)),
                        dtype=np.float64)

    L = terms32(mids.T)                                        # (G, E)
    flat = (L.max(axis=0) - L.min(axis=0)) < FLAT_NATS
    best = mids[np.arange(E), np.argmax(L, axis=0)]
    off = np.linspace(-1.5, 1.5, n_zoom)
    Z = np.clip(best[None, :] + off[:, None] * width[None, :],
                lo + 1e-9 * (hi - lo), hi - 1e-9 * (hi - lo))  # (n_zoom, E)
    cand = Z[np.argmax(terms32(Z), axis=0), np.arange(E)]

    v_new = np.array(v, dtype=np.float64)
    v_new[s:] = cand
    with mx.stream(mx.cpu):
        t64 = np.array(lp64.epoch_log_lik(mx.array(
            np.stack([np.asarray(v, dtype=np.float64), v_new]),
            dtype=mx.float64)), dtype=np.float64)
    rose = (t64[1] > t64[0]) & ~flat
    gain = np.where(rose, t64[1] - t64[0], 0.0)
    out = np.array(v, dtype=np.float64)
    out[s:] = np.where(rose, cand, out[s:])
    return TimingStep(v=out, moved=gain > MOVE_NATS, gain=gain, flat=flat)


#: TTV search: starts per joint polish (fewer than LinEph's 128: each start's
#: Newton step sends 2 x dim gradient rows, and dim grows with the epochs),
#: half with beta spread over (0, 1) as :func:`round_starts`.
N_TTV_STARTS = 32
#: Spread of the polish's starts in u: the shape as the LinEph ball, the
#: epoch times barely (a 0.25 ball in u moves each time by ~0.56 T14,
#: out of the dip the timing step put it in).
TTV_SHAPE_SD = 0.25
TTV_TIME_SD = 0.002
#: Timing step + polish passes before giving up on the certificate.
MAX_TTV_PASSES = 5
#: An oblate TTV fit's first polish: copies of the round TTV fit at f =
#: F_FLOOR and random angles (with the other nested start), in place of the
#: last plain starts.
N_TTV_ANGLES = 8


def lineph_to_ttv(layout, v_lineph):
    """The TTV vector (shape, then times) of a LinEph one (``dP, dtau0`` then
    the same shape): ``dtau_i = dtau0 + n_i dP`` (nested models)."""
    v = np.asarray(v_lineph, dtype=np.float64)
    return np.r_[v[2:2 + layout.n_shape], v[1] + layout.n_arr * v[0]]


def fit_ttv(prepared, centering, epoch_data, orders, *, law, b_prior,
            build_kw, lineph_fit, tau_half, seed=0, log=print, oblate=False,
            round_fit=None):
    """The TTV maximum-likelihood fit, by block coordinate ascent from the
    LinEph one (``lineph_fit``, an :class:`MleFit`): an exact timing step
    (:func:`timing_step`, limb darkening held at the profile's maximiser),
    then a joint polish (``anvil.optimize`` on the profiled-LD target,
    every parameter), until a timing step moves no epoch -- the certificate
    that no epoch's time can gain alone. Starts nested (``dtau_i = dtau0 +
    n_i dP``), so it cannot end below LinEph. Returns an :class:`MleFit`.

    ``oblate=True`` (hybrid4): ``lineph_fit`` is the oblate LinEph fit and
    ``round_fit`` the round TTV one, the two models this one nests (the
    first up to the period, the second at f = 0). It starts from the better
    of the two -- the round TTV fit with f = ``F_FLOOR`` -- and the first
    polish also starts from the other and from :data:`N_TTV_ANGLES` copies
    of the round fit at random angles (the round-first scheme of
    :data:`ROUND_SEED_SD`)."""
    import anvil

    if oblate and round_fit is None:
        raise ValueError("an oblate TTV fit starts from the round one: pass "
                         "round_fit")
    label = ("oblate " if oblate else "") + f"ttv {law}"
    eph = prepared.eph
    kw = dict(build_kw, profile_mode="exact", likelihood_only=True)
    play = _params.ttv_layout(eph, centering, tau_half, b_prior=b_prior,
                              law=law, oblate=oblate, ld="collapsed")
    tr, T, lo, hi = _likelihood.build_target(play, centering, epoch_data,
                                             orders, **kw)
    slay = _params.ttv_layout(eph, centering, tau_half, b_prior=b_prior,
                              law=law, oblate=oblate)
    _, _, s_lo, s_hi = _likelihood.build_target(slay, centering, epoch_data,
                                                orders, **kw)
    s, n_ld, E = play.n_shape, len(slay.ld_names), play.n_epochs
    rng = np.random.default_rng(seed)
    notes = []

    def value(theta):
        with mx.stream(mx.cpu):
            return float(np.array(hi(mx.array(theta[None], dtype=mx.float64)))
                         [0]) + lo.log_const

    def ld_at(theta):
        with mx.stream(mx.cpu):
            q = hi.conditional_mode_ld(mx.array(theta[None], dtype=mx.float64))
        return np.array([float(np.asarray(c)[0]) for c in q])

    theta = lineph_to_ttv(play, lineph_fit.v_model)
    box = play.hi[s:] * (1 - 1e-6)
    if np.any(np.abs(theta[s:]) > box):
        notes.append(f"{int(np.sum(np.abs(theta[s:]) > box))} epoch(s)' "
                     f"LinEph times lie outside their timing box; started "
                     f"at its edge")
        theta[s:] = np.clip(theta[s:], -box, box)
    lnL = value(theta)
    extra = None                 # oblate: more starts for the first polish
    if oblate:
        rv = round_fit.v_model
        r_theta = np.r_[rv[:3], rv[round_fit.joint_layout.n_shape:]]

        def at_floor(ang):       # the round TTV fit at f = F_FLOOR
            r = math.sqrt(_seeding.F_FLOOR)
            return np.r_[r_theta[:3], r * np.cos(ang), r * np.sin(ang),
                         r_theta[3:]]

        from_round = at_floor(rng.uniform(0.0, 2.0 * math.pi))
        l_round = value(from_round)
        log(f"  [{label}] starts: the oblate LinEph fit, ln L {lnL:.3f} "
            f"(LinEph {lineph_fit.lnL:.3f}); the round TTV fit at f = "
            f"{_seeding.F_FLOOR:g}, ln L {l_round:.3f} (round "
            f"{round_fit.lnL:.3f})")
        U_extra = T.from_model_np(np.vstack(
            [at_floor(a) for a in rng.uniform(0.0, 2.0 * math.pi,
                                              N_TTV_ANGLES)]))
        U_extra[:, :3] += ROUND_SEED_SD * rng.standard_normal(
            (N_TTV_ANGLES, 3))
        U_extra[:, s:] += TTV_TIME_SD * rng.standard_normal(
            (N_TTV_ANGLES, E))
        if l_round > lnL:
            theta, other, lnL = from_round, theta, l_round
        else:
            other = from_round
        extra = np.vstack([T.from_model_np(other[None])[0], U_extra])
    else:
        log(f"  [{label}] start: the LinEph best fit, ln L {lnL:.3f} "
            f"(LinEph {lineph_fit.lnL:.3f})")

    certified, res = False, None
    for npass in range(MAX_TTV_PASSES + 1):
        q = ld_at(theta)
        t0 = time.perf_counter()
        step = timing_step(s_lo, s_hi, slay,
                           np.r_[theta[:s], q, theta[s:]])
        t_step = time.perf_counter() - t0
        if res is not None and not step.moved.any():
            certified = True
            log(f"  [{label}] certified: a timing step at the polished fit "
                f"moves no epoch (none gains > {MOVE_NATS:g} alone)")
            break
        theta = np.r_[step.v[:s], step.v[s + n_ld:]]    # never lowers ln L
        if npass == MAX_TTV_PASSES:
            break
        log(f"  [{label}] pass {npass + 1}: timing step moved "
            f"{int(step.moved.sum())} of {E} epochs (+{step.gain.sum():.3f} "
            f"in ln L) in {t_step:.1f}s")
        u0 = T.from_model_np(theta)
        U = np.tile(u0, (N_TTV_STARTS, 1))
        U[1:, :s] += TTV_SHAPE_SD * rng.standard_normal((N_TTV_STARTS - 1, s))
        U[1:, s:] += TTV_TIME_SD * rng.standard_normal((N_TTV_STARTS - 1, E))
        n = N_TTV_STARTS // 2
        V = T.model_np(U)
        V[1:n + 1, play.index("beta")] = np.clip(
            (np.arange(n) + rng.uniform(size=n)) / n, 0.005, 0.995)
        U = T.from_model_np(V)
        if extra is not None and npass == 0:
            U[-len(extra):] = extra      # after the beta-stratified rows
        res = anvil.optimize(tr, U, objective="mle", seed=seed,
                             max_rows=_seeding.rows_for(
                                 epoch_data["mask"].size),
                             hessian=HESSIAN)
        log(f"  [{label}] pass {npass + 1}: polish ln L "
            f"{res.log_value:.3f} in {res.wall_time:.1f}s")
        theta = np.asarray(res.model, dtype=np.float64)
    if not certified:
        notes.append(f"the timing step still moved epochs after "
                     f"{MAX_TTV_PASSES} passes: not certified that no "
                     f"epoch's time can gain alone")
    if step.flat.any():
        notes.append(f"{int(step.flat.sum())} epoch(s) have no transit data "
                     f"their time could explain (term flat over the box): "
                     f"their times are arbitrary")
    lnL = value(theta)
    if lnL < lineph_fit.lnL - 1e-3:
        notes.append(f"ln L {lnL:.3f} is below LinEph's {lineph_fit.lnL:.3f}"
                     f": the search failed")
    if oblate and lnL < round_fit.lnL - 1e-3:
        notes.append(f"ln L {lnL:.3f} is below the round TTV fit's "
                     f"{round_fit.lnL:.3f}: the search failed")
    edge = np.abs(theta[s:]) > 0.99 * play.hi[s:]
    if edge.any():
        notes.append(f"{int(edge.sum())} epoch time(s) within 1% of the "
                     f"timing box edge (no prior holds them)")

    q = ld_at(theta)
    v_best = np.r_[theta[:s], q, theta[s:]]
    names = list(slay.names)
    best_phys = slay.transform().to_physical(v_best[None, :])
    rows = [(n, float(best_phys[0, i])) for i, n in enumerate(names)]
    rows += [(n, val) for n, val in _derived(slay, best_phys, names)]
    if oblate:
        notes += _f_edge_notes(rows)
    return MleFit(
        label=label, layout=play, joint_layout=slay, lp=lo,
        v_model=v_best, rows=rows, lnL=lnL, n_points=int(lo.n_real),
        n_params=int(slay.dim + np.sum(np.asarray(orders) + 1)),
        n_starts_converged=int(np.sum(res.status == "converged")),
        n_starts=N_TTV_STARTS, top_spread=_seeding.top_spread(res),
        notes=notes, lp_joint=s_lo)


def _derived(layout, best_phys, names):
    """(name, value) for the derived rows, at the best fit."""
    from .pipeline import _ld_weight_draws, _oblate_draws

    out = [("b", float(_outputs.derived_b(best_phys, names,
                                          layout.b_prior)[0])),
           ("log10_rho", float(_outputs.log10_rho_draws(
               best_phys, names, layout.b_prior, P_ref=layout.P_ref)[0]))]
    for fn in (_ld_weight_draws, _oblate_draws):
        n0, d0 = fn(layout, best_phys, names)
        out += [(n, float(d0[0, j])) for j, n in enumerate(n0)]
    return out


def compare_ttv(ttv_fit, lineph_fit):
    """LinEph vs TTV: a dict of the reported numbers. n_extra is the
    TTV fit's extra parameters (the epoch times for LinEph's period and
    epoch). No p-value: a timing likelihood with several peaks per epoch
    is not the regular case Wilks' theorem needs."""
    dlnL = ttv_fit.lnL - lineph_fit.lnL
    n = ttv_fit.n_points
    k = ttv_fit.n_params - lineph_fit.n_params
    return dict(lnL_ttv=ttv_fit.lnL, lnL_lineph=lineph_fit.lnL, dlnL=dlnL,
                n_extra=int(k), dBIC=k * math.log(n) - 2.0 * dlnL,
                dAIC=2.0 * k - 2.0 * dlnL, n_points=n)


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
