"""Grid-Gibbs: an exact per-epoch timing move, interleaved with ChEES-HMC.

Why it exists. A weakly constrained transit has a timing posterior with
several separated bumps, and ChEES does not move chains between them: the
pooled draws then weight each bump by however many chains happened to settle
there, and the per-epoch R-hat never passes. On KOI-4848.01 (Kepler, 4
transits) this left one epoch's timing marginal 0.16 in total variation from
the exact answer after 2,100 draws/chain, with R-hat 1.63; interleaving this
move brought it to 0.02 and R-hat 1.05 for +21% wall clock
(``benchmarks/gibbs_prototype.py``).

Why it is exact. Given the shape parameters, each epoch's likelihood term and
timing prior depend on that epoch's time alone, so the conditional of the
whole timing vector factorizes into independent per-epoch conditionals.
Each chain draws every ``dtau_i`` from a piecewise-constant density over a
grid spanning that epoch's whole prior box -- so it can land in any bump,
including ones no chain has visited -- and a Metropolis-Hastings correction
against the exact log-density makes the move leave the posterior invariant
whatever the grid resolution.

Why it is cheap. Setting every epoch to grid point g at once, one evaluation
of :meth:`ProfiledTransitLogProb.epoch_log_lik` scores all epochs at that
point, so a sweep costs ``n_grid`` evaluations per chain however many epochs
there are: O(N), not O(N^2), which is what keeps it usable for short-period
targets with hundreds of transits.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import numpy as np

from . import params as _params

#: Grid cells per epoch. The MH correction makes any resolution exact; this
#: one gave 93-97% acceptance on Kepler long-cadence targets.
N_GRID = 512
#: Draws per chain between Gibbs sweeps. 25, not 100: on the Kepler targets
#: where a transit has two timing modes (KOI-7776.01, KOI-5162.01), sweeping
#: every 100 left both unconverged at the 16,384-draw cap (162 / 88 min of
#: TTV sampling); every 25 converged both at 9,300 draws (105 / 58 min). On a
#: target with no mode switching (KOI-5228.01) it cost +10% time and still
#: improved the timing posteriors (worst distance from the exact marginal
#: 0.032 -> 0.011), because exact redraws also mix within a mode.
SEGMENT = 25


@dataclass
class SweepStats:
    """Per-epoch outcome of one sweep, averaged over chains."""

    accept: np.ndarray        # (n_epochs,) MH acceptance
    mode_change: np.ndarray   # (n_epochs,) fraction moved by more than T14


class GridGibbs:
    """The grid-Gibbs move over every ``dtau_i``, for a TTV layout.

    ``lp`` is a :class:`turin.likelihood.ProfiledTransitLogProb` in float32;
    it should be built with ``n_chains_hint`` equal to ``batch`` so its epoch
    blocking matches the evaluation size used here.
    """

    def __init__(self, lp, transform, layout, *, T14, n_grid=N_GRID,
                 segment=SEGMENT, batch=4096, seed=0):
        if layout.mode != "ttv":
            raise ValueError("grid-Gibbs applies to the TTV fit only")
        self.lp, self.transform, self.layout = lp, transform, layout
        #: draws per chain between sweeps; the round loop reads it
        self.segment = int(segment)
        self.n_ep = layout.n_epochs
        self.T14 = float(T14)
        self.batch = int(batch)
        self.rng = np.random.default_rng(seed)
        lo, hi = layout.lo[5:], layout.hi[5:]
        self.lo, self.hi = lo, hi
        self.edges = lo[:, None] + (hi - lo)[:, None] * np.linspace(
            0.0, 1.0, n_grid + 1)[None, :]                    # (E, G+1)
        self.width = (hi - lo) / n_grid                       # (E,)
        self.mids = 0.5 * (self.edges[:, 1:] + self.edges[:, :-1])
        self.n_grid = n_grid
        # the soft TTV prior is per epoch, so it rides along with the grid
        self.prior_grid = _params.ttv_log_prior_np(
            self.mids[..., None], layout.P_ref)               # (E, G)

    def _epoch_terms(self, V):
        out = []
        for s in range(0, V.shape[0], self.batch):
            out.append(np.array(self.lp.epoch_log_lik(
                mx.array(V[s:s + self.batch], dtype=mx.float32)),
                dtype=np.float64))
        return np.concatenate(out)                            # (n, E)

    def _conditional(self, v_dtau):
        """Per-epoch log conditional at given times, ``(C, E)``."""
        return (self._epoch_terms(v_dtau)
                + _params.ttv_log_prior_np(
                    v_dtau[:, 5:, None], self.layout.P_ref))

    def sweep(self, u):
        """One sweep over every epoch time of every chain.

        ``u``: ``(n_chains, dim)`` unconstrained positions. Returns the new
        positions and a :class:`SweepStats`.
        """
        v = self.transform.model_np(np.asarray(u, dtype=np.float64))
        C, E, G = v.shape[0], self.n_ep, self.n_grid

        # score every grid point of every epoch for every chain: row (c, g)
        # sets epoch i to its own grid point g, for all i at once
        V = np.repeat(v, G, axis=0)
        V[:, 5:] = np.tile(self.mids.T, (C, 1))
        L = (self._epoch_terms(V).reshape(C, G, E).transpose(0, 2, 1)
             + self.prior_grid[None, :, :])                   # (C, E, G)
        logq = L - L.max(axis=2, keepdims=True)
        logq -= np.log(np.exp(logq).sum(axis=2, keepdims=True))

        # independence proposal: a cell by its mass, uniform within it
        cdf = np.cumsum(np.exp(logq), axis=2)
        j = np.minimum((cdf < self.rng.uniform(size=(C, E, 1))).sum(axis=2),
                       G - 1)                                 # (C, E)
        new = (self.edges[np.arange(E)[None, :], j]
               + self.width[None, :] * self.rng.uniform(size=(C, E)))
        span = self.hi - self.lo
        new = np.clip(new, self.lo + 1e-9 * span, self.hi - 1e-9 * span)
        old = v[:, 5:].copy()          # v is overwritten below
        j_old = np.clip(((old - self.lo) // self.width).astype(int), 0, G - 1)

        # MH correction against the exact per-epoch conditional; epochs are
        # independent given shape, so each is accepted on its own
        v_new = v.copy()
        v_new[:, 5:] = new
        rows, cols = np.arange(C)[:, None], np.arange(E)[None, :]
        log_a = (self._conditional(v_new) - self._conditional(v)
                 + logq[rows, cols, j_old] - logq[rows, cols, j])
        acc = np.log(self.rng.uniform(size=(C, E))) < log_a
        v[:, 5:] = np.where(acc, new, old)

        stats = SweepStats(
            accept=acc.mean(axis=0),
            mode_change=(acc & (np.abs(new - old) > self.T14)).mean(axis=0))
        return self.transform.from_model_np(v), stats
