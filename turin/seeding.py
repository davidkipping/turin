"""Initialization: per-epoch timing seeds and a batched MAP optimizer.

anvil supplies no optimizer and no mode finder, and its guidance is explicit
that chains must start in a tight ball (~1e-3 in unconstrained units) around
a preliminary fit: a wide start on a likelihood with flat plateaus -- a
proposed transit that misses the data entirely -- strands chains at zero
gradient and collapses the shared step size for everyone. So turin brings its
own initialization, and this module is it.

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
    A batched multi-start gradient ascent on the log-density itself, run on
    the GPU across all starts at once, since "many chains" is exactly the
    shape this hardware likes. Its spread across starts doubles as an early
    multimodality warning.
"""

from __future__ import annotations

from dataclasses import dataclass

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
                        k, b, T14, q1, q2, period, num_resample=1,
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
    grid = _model.build_grid(centering, _supersample(exposure_time,
                                                     num_resample),
                             dtype=dtype)
    scores = np.empty((n_epochs, n_grid), dtype=np.float64)
    col = lambda v: mx.full((n_grid, 1), float(v), dtype=dtype)
    dtau = mx.array(np.ascontiguousarray(grids.T), dtype=dtype)  # (n_grid, E)
    mid = dtau + mx.array(np.ascontiguousarray(d_arr), dtype=dtype)[None, :]
    f_dev = _model.transit_flux_dev(
        grid, mid=mid, k=col(k), b=col(b), T14=col(T14), q1=col(q1),
        q2=col(q2), period=col(period), geometry=geometry)
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


@dataclass
class MapResult:
    """Outcome of the multi-start MAP search, in unconstrained space."""

    u_best: np.ndarray        # (dim,) the best start's final position
    log_prob_best: float
    u_all: np.ndarray         # (n_starts, dim) every start's final position
    log_prob_all: np.ndarray  # (n_starts,)
    n_iter: int
    #: spread of the top decile's log-density; large means the starts did not
    #: agree, i.e. the posterior probably has more than one basin
    top_spread: float

    @property
    def multimodal_warning(self) -> bool:
        return self.top_spread > 10.0

    def ball(self, n_chains, spread=1e-3, seed=0):
        """An initialization ball around the MAP, anvil's recommended start."""
        rng = np.random.default_rng(seed)
        u = (self.u_best[None, :]
             + spread * rng.standard_normal((n_chains, self.u_best.size)))
        return mx.array(u.astype(np.float32))


def find_map(target, u_starts, *, n_iter=400, lr=0.05, clip=10.0):
    """Batched gradient ascent on ``target.log_prob``, all starts at once.

    Plain Adam in unconstrained space. The starts are the batch dimension, so
    this costs about as much as ``n_iter`` sampler iterations no matter how
    many starts there are -- which is why using a few hundred is free and
    worth it as a multimodality probe.

    ``clip`` bounds the per-component step in unconstrained units, which keeps
    a start that lands on a plateau (or on anvil's saturating bounded-parameter
    edge) from being flung somewhere unrecoverable.
    """
    u = mx.array(u_starts, dtype=mx.float32)
    m = mx.zeros_like(u)
    v = mx.zeros_like(u)
    b1, b2, eps = 0.9, 0.999, 1e-8

    def neg(uu):
        return -mx.sum(target.log_prob(uu))

    grad_fn = mx.compile(mx.grad(neg))
    for t in range(1, n_iter + 1):
        g = grad_fn(u)
        g = mx.where(mx.isfinite(g), g, mx.zeros_like(g))
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        mhat = m / (1 - b1**t)
        vhat = v / (1 - b2**t)
        step = lr * mhat / (mx.sqrt(vhat) + eps)
        u = u - mx.clip(step, -clip, clip)
        mx.eval(u, m, v)

    lp = np.array(target.log_prob(u), dtype=np.float64)
    lp = np.where(np.isfinite(lp), lp, -np.inf)
    u_np = np.array(u, dtype=np.float64)
    best = int(np.argmax(lp))

    finite = lp[np.isfinite(lp)]
    if finite.size:
        top = np.sort(finite)[-max(1, finite.size // 10):]
        spread = float(top.max() - top.min())
    else:
        spread = float("inf")

    return MapResult(u_best=u_np[best], log_prob_best=float(lp[best]),
                     u_all=u_np, log_prob_all=lp, n_iter=n_iter,
                     top_spread=spread)


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
    return {"k": k0, "beta": float(np.clip(beta0, 0.01, 0.95)),
            "T14": float(np.clip(T14_0, 1e-4, 0.95 * T14_hi)),
            "q1": 0.5, "q2": 0.5}


def initial_model_vector(layout, eph, *, shape=None, dtau=None, dP=0.0,
                         dtau0=0.0):
    """Assemble a model-space starting vector for either fit mode."""
    s = dict(default_shape_start(eph, layout))
    if shape:
        s.update({k: float(v) for k, v in shape.items() if k in s})
    if layout.mode == "lineph":
        return np.array([dP, dtau0, s["k"], s["beta"], s["T14"],
                         s["q1"], s["q2"]], dtype=np.float64)
    base = np.array([s["k"], s["beta"], s["T14"], s["q1"], s["q2"]],
                    dtype=np.float64)
    n_ep = layout.n_epochs
    taus = np.zeros(n_ep) if dtau is None else np.asarray(dtau, dtype=np.float64)
    if taus.shape != (n_ep,):
        raise ValueError(f"dtau has shape {taus.shape}, expected ({n_ep},)")
    # keep strictly inside the box: the transform maps the open interval, and
    # a value exactly on the bound is +-inf in unconstrained space
    lo, hi = layout.lo[5:], layout.hi[5:]
    taus = np.clip(taus, 0.98 * lo, 0.98 * hi)
    return np.concatenate([base, taus])
