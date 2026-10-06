"""The log-density anvil samples: transit model, profiled baselines, priors.

turin writes its own likelihood rather than using
``anvil.precision.ChunkedGaussianLogLike`` for a structural reason: that class
chunks the *data* axis, which would cut across epoch boundaries, and the
profile solve is per-epoch. turin chunks by **epoch block** instead, which
bounds peak memory the same way while keeping each epoch's solve whole.

What is in the graph and what is not:

- The float32 term returned by :meth:`ProfiledTransitLogProb.__call__` is the
  *recentred* log-density -- it omits an additive constant. That is anvil's
  rule 4, and the reason is that the constant is of order ``N/2 + sum log
  sigma`` (thousands), whose float32 ulp rivals the ~1-unit Metropolis scale,
  while the recentred value is of order ``sqrt(N)``. MCMC is invariant to an
  additive constant, so the sampler never needs it.
- ``.log_const`` holds that constant in float64 for whoever reports an
  absolute log-likelihood (the chain exports, the maximum-likelihood row).

The ``noise`` seam exists for the GP direction: :class:`WhiteNoise` is the
only v1 implementation, and it is what makes the likelihood a plain diagonal
Gaussian over the profiled residuals. Swapping in ``anvil-gp``'s
``GPLogLike`` later means supplying the transit-times-baseline product as its
``mean_fn`` rather than changing anything here.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import numpy as np

from . import model as _model
from . import params as _params
from . import profile as _profile

#: Rough bytes per (chain, point, sub-exposure) for the forward pass and the
#: fused kernel's backward grids together, used only to size epoch blocks.
#:
#: It was set from MetalPlanet's documented ~12 B/pt forward and ~32 B/pt
#: transient and described here as "deliberately pessimistic". It is not:
#: a fresh-process MLX probe of one compiled value+grad at 512 chains
#: measured **121 B per (chain, point)** on the default path, so 48 is about
#: 2.5x optimistic. Nothing has broken because the 2 GiB budget below is
#: generous enough that real targets take one block anyway (at 223 points per
#: epoch it only splits past ~390 epochs), but do not read this as a safety
#: margin -- there is none. Raise it, rather than trusting it, before relying
#: on blocking to fit a large target. ``--ld=collapsed`` passes its own,
#: 2.5x larger figure (``ldmarg.COLLAPSED_BYTES_PER_POINT``): it measured 271
#: against this path's 121 B.
BYTES_PER_POINT = 48
#: Budget used to pick the block size. **Not a ceiling on peak memory.**
#: Inside one compiled value+grad the whole graph is live at once, so every
#: block's backward intermediates coexist and splitting buys much less than
#: the arithmetic suggests: measured, forcing 3 blocks moved the peak from
#: 0.174 to 0.153 GB on the default path and 0.390 to 0.320 GB on a 3-wide
#: variant. Blocking bounds the size of any one kernel launch; it does not
#: bound the peak.
DEFAULT_BLOCK_BUDGET_BYTES = 1 << 31   # 2 GiB


class WhiteNoise:
    """Independent Gaussian errors at the reported uncertainties.

    Stateless: the weights and the normalization live in the
    :class:`turin.profile.ProfileDesign`, which already holds ``inv_sigma``
    and ``log_const``. This class exists to name the assumption and to mark
    the seam where a correlated-noise model would go.
    """

    name = "white"
    n_hyper = 0

    def log_prob_terms(self, design, residual):
        return _profile.chi2_terms(design, residual)


def epoch_block_size(n_chains, n_epochs, max_pts, n_sub,
                     budget_bytes=DEFAULT_BLOCK_BUDGET_BYTES, itemsize=4,
                     bytes_per_point=BYTES_PER_POINT):
    """How many epochs to evaluate at once, from a memory budget.

    Returns at least 1, so a single pathological epoch still runs (and fails
    loudly on allocation rather than silently producing nothing).

    See :data:`BYTES_PER_POINT` and :data:`DEFAULT_BLOCK_BUDGET_BYTES`: the
    constant is measured to be ~2.5x optimistic, and the budget sizes a
    launch rather than capping peak memory.
    """
    per_epoch = max(1, n_chains * max_pts * max(1, n_sub)
                    * bytes_per_point * itemsize // 4)
    return int(max(1, min(n_epochs, budget_bytes // per_epoch)))


@dataclass
class _Block:
    lo: int
    hi: int
    grid: _model.EpochGrid
    design: _profile.ProfileDesign


class ProfiledTransitLogProb:
    """Model-space log-density, batched over chains.

    Call signature is anvil's: ``(n_chains, dim) -> (n_chains,)``, built from
    MLX ops only so it survives ``mx.compile`` inside a kernel step.

    Instances are cheap to hold but not cheap to build: the epoch blocks and
    every static tensor are prepared once here, deliberately, so that nothing
    inside the traced call slices, evaluates or synchronizes.
    """

    def __init__(self, layout, centering, epoch_data, orders, *,
                 num_resample=1, exposure_time=0.0, profile_mode="exact",
                 geometry="circular", noise=None,
                 n_chains_hint=512, dtype=mx.float32,
                 budget_bytes=DEFAULT_BLOCK_BUDGET_BYTES,
                 bytes_per_point=BYTES_PER_POINT):
        if profile_mode not in _profile.PROFILE_MODES:
            raise ValueError(f"unknown profile mode {profile_mode!r}")
        self.layout = layout
        self.mode = layout.mode
        self.profile_mode = profile_mode
        self.geometry = geometry
        self.noise = noise or WhiteNoise()
        self.dtype = dtype
        self.dim = layout.dim
        self.ld = layout.ld

        sub_offsets = _prep_offsets(exposure_time, num_resample)
        full_grid = _model.build_grid(centering, sub_offsets, dtype=dtype,
                                      exp_time=exposure_time,
                                      mask=epoch_data["mask"])
        full_design = _profile.build_design(epoch_data, orders, dtype=dtype)
        if full_grid.n_epochs != full_design.n_epochs:
            raise ValueError(
                f"grid has {full_grid.n_epochs} epochs but the profile design "
                f"has {full_design.n_epochs}")
        if self.mode == "ttv" and layout.n_epochs != full_grid.n_epochs:
            raise ValueError(
                f"ttv layout covers {layout.n_epochs} epochs but the grid has "
                f"{full_grid.n_epochs}")

        self.n_epochs = full_grid.n_epochs
        self.n_real = full_design.n_real
        self.log_const = full_design.log_const
        self.n_sub = full_grid.n_sub
        self.max_pts = full_grid.max_pts

        step = epoch_block_size(n_chains_hint, self.n_epochs, self.max_pts,
                                self.n_sub, budget_bytes,
                                itemsize=8 if dtype == mx.float64 else 4,
                                bytes_per_point=bytes_per_point)
        self.blocks = [
            _Block(lo, min(lo + step, self.n_epochs),
                   full_grid.select(lo, min(lo + step, self.n_epochs)),
                   full_design.select(lo, min(lo + step, self.n_epochs)))
            for lo in range(0, self.n_epochs, step)
        ]
        self.block_size = step
        # kept for post-processing, which wants the whole grid at once
        self.grid = full_grid
        self.design = full_design

    # -- parameter unpacking ---------------------------------------------

    def unpack(self, v):
        """Model-space vector -> the named quantities the model needs.

        ``v``: ``(n_chains, dim)``. Columns are kept as ``(n_chains, 1)`` so
        they broadcast against the per-epoch and per-point axes, and so
        MetalPlanet's fp64 fallback broadcasts them against ``z``.
        """
        col = lambda i: v[:, i:i + 1]
        if self.mode == "lineph":
            dP, dtau0 = col(0), col(1)
            k, beta, T14, q1, q2 = (col(i) for i in range(2, 7))
            dtau = None
        else:
            k, beta, T14, q1, q2 = (col(i) for i in range(5))
            dP = mx.zeros_like(k)
            dtau0 = None
            dtau = v[:, 5:]
        b = _model.impact_parameter(beta, k, self.layout.b_prior)
        return dict(dP=dP, dtau0=dtau0, dtau=dtau, k=k, beta=beta, b=b,
                    T14=T14, q1=q1, q2=q2, period=self.layout.P_ref + dP)

    def mid_times(self, grid, p, lo, hi):
        """Per-epoch mid-transit offsets for epochs [lo, hi)."""
        if self.mode == "lineph":
            return _model.mid_times_lineph(grid, p["dP"], p["dtau0"])
        return _model.mid_times_ttv(grid, p["dtau"][:, lo:hi])

    # -- the log-density --------------------------------------------------

    def log_prior(self, p):
        """Prior terms beyond the bounding box, shape ``(n_chains,)``."""
        term = _params.bk_log_prior(p["beta"], p["k"], self.layout.b_prior)
        if self.mode == "ttv":
            term = term + _params.ttv_log_prior(p["dtau"], self.layout.P_ref)
        return term

    def __call__(self, v):
        p = self.unpack(v)
        total = None
        for blk in self.blocks:
            f_dev = _model.transit_flux_dev(
                blk.grid, mid=self.mid_times(blk.grid, p, blk.lo, blk.hi),
                k=p["k"], b=p["b"], T14=p["T14"], q1=p["q1"], q2=p["q2"],
                period=p["period"], geometry=self.geometry)
            c = _profile.solve_coefficients(blk.design, f_dev,
                                            self.profile_mode)
            resid = _profile.residual_dev(blk.design, f_dev, c)
            # sum within the block first, then across blocks: a two-level
            # reduction tree, which is what keeps the float32 error at
            # O(eps sqrt(n_blocks)) rather than O(eps n)
            term = mx.sum(self.noise.log_prob_terms(blk.design, resid), axis=-1)
            total = term if total is None else total + term
        return total + self.log_prior(p)

    def epoch_log_lik(self, v):
        """Per-epoch profiled log-likelihood terms, ``(n_chains, n_epochs)``.

        The summands of :meth:`__call__` before the sum over epochs, without
        priors or the float64 constant. Given the shape parameters each term
        depends on its own epoch's time only, which is what lets the
        grid-Gibbs move score every epoch from one evaluation.
        """
        p = self.unpack(v)
        terms = []
        for blk in self.blocks:
            f_dev = _model.transit_flux_dev(
                blk.grid, mid=self.mid_times(blk.grid, p, blk.lo, blk.hi),
                k=p["k"], b=p["b"], T14=p["T14"], q1=p["q1"], q2=p["q2"],
                period=p["period"], geometry=self.geometry)
            c = _profile.solve_coefficients(blk.design, f_dev,
                                            self.profile_mode)
            resid = _profile.residual_dev(blk.design, f_dev, c)
            terms.append(self.noise.log_prob_terms(blk.design, resid))
        return mx.concatenate(terms, axis=-1)

    # -- reporting helpers ------------------------------------------------

    def log_prob_absolute(self, v):
        """The log-density with the float64 constant restored (host float64)."""
        return np.asarray(self(v), dtype=np.float64) + self.log_const

    def full_model(self, v, *, ld=None):
        """Detrended model and coefficients over every epoch, for plots.

        Returns ``(model, coeffs)`` with shapes ``(n_chains, n_epochs,
        max_pts)`` and ``(n_chains, n_epochs, n_cols)``. Not used on the
        sampling path -- it deliberately skips the block chunking.

        ``ld=(q1, q2)``, host floats, overrides the limb darkening in ``v``;
        it is how a layout without ``q1, q2`` (``--ld=collapsed``) still gets
        a light curve. ``None`` reads them from ``v`` as before.
        """
        p = self.unpack(v)
        if ld is None:
            q1, q2 = p["q1"], p["q2"]
        else:
            q1 = mx.full(p["k"].shape, float(ld[0]), dtype=self.dtype)
            q2 = mx.full(p["k"].shape, float(ld[1]), dtype=self.dtype)
        f_dev = _model.transit_flux_dev(
            self.grid, mid=self.mid_times(self.grid, p, 0, self.n_epochs),
            k=p["k"], b=p["b"], T14=p["T14"], q1=q1, q2=q2,
            period=p["period"], geometry=self.geometry)
        return _profile.detrended_model(self.design, f_dev, self.profile_mode)


def _prep_offsets(exposure_time, num_resample):
    from .prep import supersample_offsets

    return supersample_offsets(exposure_time, num_resample)


def build_target(layout, centering, epoch_data, orders, *, fp64=True, **kw):
    """Assemble the anvil target for one fit.

    Returns ``(target, transform, log_prob, log_prob_hi)``, where ``target``
    is an ``anvil.TransformedLogDensity`` over the unconstrained space and
    ``log_prob`` is the float32 model-space callable.

    ``fp64=True`` also builds the float64 CPU replica anvil needs for
    ``validate_precision``, ``certify`` and ``reanchor_every``. It costs a
    second copy of the static tensors (a few MB) and nothing per call.
    """
    import anvil

    lo = ProfiledTransitLogProb(layout, centering, epoch_data, orders,
                                dtype=mx.float32, **kw)
    hi = None
    if fp64:
        with mx.stream(mx.cpu):
            hi = ProfiledTransitLogProb(layout, centering, epoch_data, orders,
                                        dtype=mx.float64, **kw)

    transform = layout.transform()
    target = anvil.TransformedLogDensity(
        lo, transform,
        model_log_prob_hi=(_Fp64Wrapper(hi) if hi is not None else None))
    return target, transform, lo, hi


class _Fp64Wrapper:
    """Runs the float64 log-density on the CPU stream, as MLX requires.

    anvil's ``TransformedLogDensity.log_prob_hi`` already enters a CPU stream
    and hands over float64 model-space values, but the context is kept here so
    that turin can also call this directly (for the maximum-likelihood row, and
    for tests) without repeating it.

    Note anvil's float64 path is **value-only**: ``log_prob_hi`` converts ``u``
    to NumPy before transforming, which severs the autodiff graph, so its
    gradient is identically zero. That is by design -- ``validate_precision``,
    ``certify`` and ``reanchor_every`` all want values -- and HMC always takes
    its gradients from the float32 path. To finite-difference a float64
    gradient, differentiate a :class:`ProfiledTransitLogProb` built with
    ``dtype=mx.float64`` directly, in model space.
    """

    def __init__(self, inner):
        self.inner = inner

    def __call__(self, v):
        with mx.stream(mx.cpu):
            return self.inner(v if v.dtype == mx.float64
                              else v.astype(mx.float64))
