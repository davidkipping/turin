"""Initialization: per-epoch timing seeds and the MAP chains start from.

anvil's guidance is explicit that chains must start in a tight ball (~1e-3
in unconstrained units) around a preliminary fit: a wide start on a
likelihood with flat plateaus -- a proposed transit that misses the data
entirely -- strands chains at zero gradient and collapses the shared step
size for everyone. This module finds that fit.

Two pieces:

:func:`template_sweep_taus`
    hurin's fix for its headline failure mode. Sweeping the transit template
    across each epoch's allowed timing range and seeding at the per-epoch
    posterior peak is what stops chains settling into a secondary timing mode
    and reporting a confident wrong answer (hurin's KOI-5162.01 sat 35.6 logL
    below the global optimum with R-hat <= 1.01). Ported with all four of the
    guards hurin accumulated: a 0.55*T14 edge margin, posterior rather than
    likelihood scoring, interior-local-maxima-only candidates, and a
    near-tie break toward the predicted time. It also reports each epoch's
    rival gap so the caller can warn about shape-sensitive seeds.

:func:`find_map`
    The fit's best point, by ``anvil.optimize`` (anvil >= 0.9): many starts
    at once, L-BFGS then damped Newton, with the Laplace covariance that
    :meth:`anvil.OptimizeResult.laplace_starts` spreads curved posteriors'
    chains over. turin keeps only the choices: which density, and the
    start-up for oblate fits (:func:`floor_oblateness`, :func:`chain_starts`).
"""

from __future__ import annotations

import dataclasses

import mlx.core as mx
import numpy as np

from . import model as _model
from . import params as _params
from . import profile as _profile

#: Grid points per epoch in the template sweep. hurin's value.
N_SWEEP_GRID = 512
#: Candidates within this many log-units of the best are treated as tied, and
#: broken toward the predicted transit time.
TIE_LOGL = 10.0
#: A seed whose best rival peak (more than one T14 away) is within this many
#: log-units is shape-sensitive: the fit will sit confidently wherever it
#: starts, so the caller should warn.
RIVAL_GAP_WARN = 10.0
#: Keep the whole template on real data. A template hanging half off the
#: window can out-score the true transit on a deep edge feature by tens of
#: log-units (hurin measured +39 on KOI-448.02 epoch +11).
EDGE_MARGIN_T14 = 0.55


def _epoch_time_extent(epoch_data):
    """Per-epoch first and last real point, in epoch-centred coordinates."""
    mask = np.asarray(epoch_data["mask"]).astype(bool)
    times = np.asarray(epoch_data["times_padded"], dtype=np.float64)
    centers = np.asarray(epoch_data["epoch_centers"], dtype=np.float64)
    centred = times - centers[:, None]
    lo = np.where(mask, centred, np.inf).min(axis=1)
    hi = np.where(mask, centred, -np.inf).max(axis=1)
    return lo, hi


def template_sweep_taus(epoch_data, centering, design, *, tau_half,
                        k, b, T14, period, q1=None, q2=None, q=None,
                        law="quadratic", num_resample=1,
                        exposure_time=0.0, n_grid=N_SWEEP_GRID,
                        profile_mode="exact", geometry="circular",
                        dtype=mx.float32):
    """Per-epoch timing seeds from a dense template sweep.

    Evaluates the profiled log-likelihood with the transit template held at
    the given shape and slid across each epoch's allowed range, all epochs and
    all grid points in one batched GPU call. Returns
    ``(seeds, rival_gap, scores, grids)``:

    ``seeds``
        ``(n_epochs,)`` timing offsets from each epoch's predicted time, in
        days -- directly the ``dtau_i`` model-space values.
    ``rival_gap``
        ``(n_epochs,)`` log-units by which the seed beats the best peak more
        than one T14 away; ``inf`` when there is no such rival.
    ``scores``, ``grids``
        ``(n_epochs, n_grid)`` posterior scores and the offsets they were
        evaluated at, for diagnostics and plots.

    The sweep exploits the batch dimension differently from the sampler: the
    "chain" axis carries grid positions, and every chain shares one shape.
    """
    n_epochs = int(np.asarray(epoch_data["mask"]).shape[0])
    tau_half = np.broadcast_to(np.asarray(tau_half, dtype=np.float64),
                               (n_epochs,)).astype(np.float64)
    d_arr = np.asarray(centering["d_arr"], dtype=np.float64)

    # allowed range per epoch, clipped inward so the whole template stays on
    # real data; epochs with no usable range collapse to the predicted time
    t_lo, t_hi = _epoch_time_extent(epoch_data)
    margin = EDGE_MARGIN_T14 * float(T14)
    lo = np.maximum(-tau_half, t_lo + margin)
    hi = np.minimum(tau_half, t_hi - margin)
    degenerate = hi <= lo
    lo = np.where(degenerate, 0.0, lo)
    hi = np.where(degenerate, 0.0, hi)

    frac = np.linspace(0.0, 1.0, n_grid)
    grids = lo[:, None] + (hi - lo)[:, None] * frac[None, :]

    # one batched evaluation: the leading axis is grid position, and every
    # epoch is offset by its own row of the grid
    grid = _model.build_grid(centering,
                             _supersample(exposure_time, num_resample),
                             dtype=dtype, exp_time=exposure_time,
                             mask=epoch_data["mask"],
                             exp_epoch=epoch_data.get("exposure"))
    scores = np.empty((n_epochs, n_grid), dtype=np.float64)
    col = lambda v: mx.full((n_grid, 1), float(v), dtype=dtype)
    dtau = mx.array(np.ascontiguousarray(grids.T), dtype=dtype)  # (n_grid, E)
    mid = dtau + mx.array(np.ascontiguousarray(d_arr), dtype=dtype)[None, :]
    if q is None:
        q = (q1, q2)
    f_dev = _model.transit_flux_dev(
        grid, mid=mid, k=col(k), b=col(b), T14=col(T14),
        q=tuple(col(x) for x in q), law=law, period=col(period),
        geometry=geometry)
    c = _profile.solve_coefficients(design, f_dev, profile_mode)
    resid = _profile.residual_dev(design, f_dev, c)
    # (n_grid, n_epochs) -> per-epoch rows
    per_epoch = np.array(_profile.chi2_terms(design, resid),
                         dtype=np.float64).T
    scores[:] = per_epoch

    # Score the POSTERIOR, not the likelihood: adding the same soft TTV prior
    # the sampler uses stops an extreme-|TTV| junk dip being seeded in
    # preference to what the fit would actually favour. The penalty is gentle
    # (~1 log-unit at |TTV/P| ~ 0.008), so a decisively displaced transit is
    # unaffected.
    # grids[..., None] makes each grid point its own one-element TTV vector,
    # which ttv_log_prior_np reduces over -> (n_epochs, n_grid)
    scores = scores + _params.ttv_log_prior_np(grids[..., None], period)

    seeds = np.zeros(n_epochs)
    rival_gap = np.full(n_epochs, np.inf)
    for i in range(n_epochs):
        row = scores[i]
        if degenerate[i] or not np.all(np.isfinite(row)):
            continue
        # Interior local maxima only: an argmax sitting on the grid boundary
        # is the template chasing a feature outside the allowed range, not a
        # transit detection.
        interior = np.zeros(n_grid, dtype=bool)
        interior[1:-1] = ((row[1:-1] >= row[:-2]) & (row[1:-1] >= row[2:]))
        if not interior.any():
            continue
        peak = row[interior].max()
        tied = np.where(interior & (row >= peak - TIE_LOGL))[0]
        j = int(tied[np.argmin(np.abs(grids[i, tied]))])
        seeds[i] = grids[i, j]

        far = np.abs(grids[i] - seeds[i]) > float(T14)
        if far.any():
            rival_gap[i] = row[j] - row[far].max()

    # never seed outside the prior
    seeds = np.clip(seeds, -0.95 * tau_half, 0.95 * tau_half)
    return seeds, rival_gap, scores, grids


def _supersample(exposure_time, num_resample):
    from .prep import supersample_offsets

    return supersample_offsets(exposure_time, num_resample)


#: Largest batch ``anvil.optimize`` sends the target in one call. Its Newton
#: step evaluates ``2 dim`` gradient rows per start, so its natural batch is
#: far larger than any turin sends while sampling; this keeps a collapsed
#: hybrid4 target (~3 MB per row) to the memory of a 512-chain evaluation.
MAX_ROWS = 512

#: Rows x points one ``anvil.optimize`` call may carry, for ``--mle``. macOS
#: kills a GPU command buffer that runs too long ("Impacting Interactivity"),
#: and the optimizer's finite-difference Hessian sends 512 rows at a time: on
#: a 50,000-point oblate TTV target (1,900 parameters) that is a ~7 s launch,
#: and KOI-7259.01 and KOI-1631.01 both lost hours of work to the watchdog
#: there. 8e6 gives ~128 rows (~1.6 s) at 50k points. Per row the cost is
#: flat from 128 rows up (12 ms vs 13.6 ms at 512), so smaller batches cost
#: nothing from 128 rows up (64 rows measured 35% slower per row, and the
#: 5-parameter LinEph fit of KOI-1631.01 took 195 s at 64 rows against 123 s
#: at 512, so only the TTV polish uses this); a target under 15,625 points
#: keeps 512.
ROW_POINTS = 8_000_000
MIN_ROWS = 128


def rows_for(n_points, max_rows=MAX_ROWS):
    """The ``max_rows`` for a target of ``n_points`` (padded) points: a power
    of two, at most ``max_rows``, at least :data:`MIN_ROWS`, with rows x points near
    :data:`ROW_POINTS`."""
    n = max(1, int(ROW_POINTS // max(1, int(n_points))))
    return int(max(MIN_ROWS, min(max_rows, 1 << (n.bit_length() - 1))))


def find_map(target, u_starts, *, seed=0, max_rows=MAX_ROWS):
    """The fit's best point from ``u_starts`` ((n, dim), unconstrained), by
    ``anvil.optimize``; returns its ``OptimizeResult``.

    ``objective="density"``: the sampler's own log-density in u, Jacobian
    included -- the point the chains are started around, which the Jacobian
    keeps off the box edges where ChEES can freeze a chain. turin's priors
    (b-k, TTV, oblateness, rho*) sit in its ``model_log_prob``, not on the
    ParamSpecs, so anvil's ``"mle"`` and ``"map"`` would both include them.
    """
    import anvil

    return anvil.optimize(target, np.asarray(u_starts, dtype=np.float64),
                          objective="density", seed=seed, max_rows=max_rows)


def top_spread(res):
    """Spread of the top decile of the starts' final values, in log-units.
    Large means the best starts ended in different basins: an early
    multimodality warning. anvil's ``optima`` is not used for this: it calls
    starts 0.05 apart in u distinct, so a wide posterior's one optimum shows
    up as several (KOI-518.02 LinEph: 7 within 0.007 nats, along beta and q
    whose sd in u is 0.8-1.2)."""
    v = np.sort(res.values[np.isfinite(res.values)])
    if not v.size:
        return float("inf")
    top = v[-max(1, v.size // 10):]
    return float(top.max() - top.min())


#: ``top_spread`` above which the log warns of more than one basin.
TOP_SPREAD_WARN = 10.0


#: Smallest flattening the chains of an oblate fit are centred on.
#: f = fc^2 + fs^2 has a stationary point at the origin, so a centre at
#: f ~ 0 would start every chain where (fc, fs) feel almost no gradient,
#: and at exactly 0 theta's atan2 gives a NaN one. (Before elome 0.18.6 its
#: float32 kernel also gave d/df = 0 below f = 1e-5; it now takes d/df from
#: the ellipse there.) f = 1e-4 (fc = 0.01) is far inside any posterior
#: width in f (SquishierPlanet: sd(f) ~ 0.05 at 10 ppm).
F_FLOOR = 1e-4


def floor_oblateness(u, transform, layout, f_floor=F_FLOOR):
    """``u`` (unconstrained, ``(dim,)``) with its oblateness raised to at
    least ``f_floor``, keeping ``theta`` (``theta = 0`` from the origin)."""
    v = np.array(transform.model_np(np.asarray(u, dtype=np.float64)[None, :])[0])
    i, j = layout.index("fc"), layout.index("fs")
    f = v[i] ** 2 + v[j] ** 2
    if f >= f_floor:
        return np.asarray(u, dtype=np.float64)
    if f > 0:
        v[[i, j]] *= np.sqrt(f_floor / f)
    else:
        v[i], v[j] = np.sqrt(f_floor), 0.0
    return np.array(transform.from_model_np(v), dtype=np.float64)


def chain_starts(res, n_chains, *, seed=0, sigma_cap=0.5):
    """Starts for an oblate fit: ``res.laplace_starts`` (SquishierPlanet's
    ChEES froze every chain started in a small ball around an oblate MAP).

    anvil gives no covariance unless the Hessian is negative definite, and a
    round planet's is not: its optimum sits at f ~ 0, where f = fc^2 + fs^2
    is stationary. turin's recipe
    (SquishierPlanet's ``laplace_start``) floors the curvature instead, so
    such a direction is treated as flat and capped at ``sigma_cap``; that
    floor is applied here before anvil draws. A 1e-3 ball is the fallback
    when the Hessian is not finite. Returns ``(u0, n_flat)``, ``n_flat``
    the directions floored, or None for the ball.
    """
    n_flat = 0
    if res.cov_u is None and res.hessian_u is not None:
        free = ~res.at_bound
        A = -res.hessian_u[np.ix_(free, free)]
        if np.all(np.isfinite(A)):
            lam, P = np.linalg.eigh(0.5 * (A + A.T))
            n_flat = int(np.sum(lam < 1.0 / sigma_cap**2))
            lam = np.maximum(lam, 1.0 / sigma_cap**2)
            cov = np.full_like(res.hessian_u, np.nan)
            cov[np.ix_(free, free)] = (P / lam) @ P.T
            res = dataclasses.replace(res, cov_u=cov)
    if res.cov_u is None:
        return res.ball(n_chains, seed=seed), None
    return res.laplace_starts(n_chains, sigma_cap=sigma_cap, seed=seed), n_flat


def default_shape_start(eph, layout):
    """hurin's physically sensible starting shape, in model space.

    Never the prior median: for ``b ~ U(0, 2)`` that is ``b = 1``, exactly the
    grazing boundary, which is what ``init_to_uniform`` got wrong in hurin.
    """
    depth_rel = max(float(eph.get("depth", 0.0)) / 1e6, 1e-8)
    k0 = float(np.clip(np.sqrt(depth_rel), 0.005, 0.5))
    T14_0 = float(eph["duration"]) / 24.0
    b0 = 0.3
    if layout.b_prior == "nongrazing":
        beta0 = min(0.95, b0 / max(1.0 - k0, 1e-6))
    elif layout.b_prior == "box":
        beta0 = b0 / 2.0
    else:
        beta0 = b0 / (1.0 + k0)
    T14_hi = float(layout.hi[layout.index("T14")])
    # every limb-darkening coordinate starts mid-box: Kipping's (0.5, 0.5),
    # and for hybrid4 a point well inside the weight simplex
    start = {"k": k0, "beta": float(np.clip(beta0, 0.01, 0.95)),
             "T14": float(np.clip(T14_0, 1e-4, 0.95 * T14_hi)),
             **{n: 0.5 for n in _model.ld_names(layout.law)}}
    if getattr(layout, "oblate", False):
        # round: the MAP's starts spread (fc, fs) by ~0.1 either way, and
        # floor_oblateness keeps the chains' centre off f = 0
        start.update(fc=0.0, fs=0.0)
    return start


def initial_model_vector(layout, eph, *, shape=None, dtau=None, dP=0.0,
                         dtau0=0.0):
    """Assemble a model-space starting vector for either fit mode."""
    s = dict(default_shape_start(eph, layout))
    if shape:
        s.update({k: float(v) for k, v in shape.items() if k in s})
    if layout.mode == "lineph":
        # by name, so a layout without q1, q2 (--ld=collapsed) needs no case
        vals = dict(s, dP=dP, dtau0=dtau0)
        return np.array([vals[n] for n in layout.names], dtype=np.float64)
    base = np.array([s[n] for n in layout.names[:layout.n_shape]],
                    dtype=np.float64)
    n_ep = layout.n_epochs
    taus = np.zeros(n_ep) if dtau is None else np.asarray(dtau, dtype=np.float64)
    if taus.shape != (n_ep,):
        raise ValueError(f"dtau has shape {taus.shape}, expected ({n_ep},)")
    # keep strictly inside the box: the transform maps the open interval, and
    # a value exactly on the bound is +-inf in unconstrained space
    lo, hi = layout.lo[layout.n_shape:], layout.hi[layout.n_shape:]
    taus = np.clip(taus, 0.98 * lo, 0.98 * hi)
    return np.concatenate([base, taus])
