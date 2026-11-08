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
        s = layout.n_shape              # epoch times start here
        self._s = s
        lo, hi = layout.lo[s:], layout.hi[s:]
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
                    v_dtau[:, self._s:, None], self.layout.P_ref))

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
        V[:, self._s:] = np.tile(self.mids.T, (C, 1))
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
        old = v[:, self._s:].copy()    # v is overwritten below
        j_old = np.clip(((old - self.lo) // self.width).astype(int), 0, G - 1)

        # MH correction against the exact per-epoch conditional; epochs are
        # independent given shape, so each is accepted on its own
        v_new = v.copy()
        v_new[:, self._s:] = new
        rows, cols = np.arange(C)[:, None], np.arange(E)[None, :]
        log_a = (self._conditional(v_new) - self._conditional(v)
                 + logq[rows, cols, j_old] - logq[rows, cols, j])
        acc = np.log(self.rng.uniform(size=(C, E))) < log_a
        v[:, self._s:] = np.where(acc, new, old)

        stats = SweepStats(
            accept=acc.mean(axis=0),
            mode_change=(acc & (np.abs(new - old) > self.T14)).mean(axis=0))
        return self.transform.from_model_np(v), stats


@dataclass
class EphemerisSweepStats:
    """Outcome of one :class:`EphemerisGibbs` sweep, averaged over chains."""

    accept: np.ndarray        # (2,) MH acceptance for dP, dtau0
    jump: np.ndarray          # (2,) fraction moved by more than one unit
    de_accept: float = np.nan  # the DE steps' acceptance
    de_move: np.ndarray = None  # (2,) their rms move in dP, dtau0, in sd


class EphemerisGibbs:
    """An exact Gibbs move on a Gaia fit's ``dP`` and then ``dtau0``.

    Why it exists. Sparse photometry puts one or two points in each transit,
    so a period or epoch change that moves a point across a transit's edge
    changes the likelihood by tens of nats in minutes, and inside those edges
    it is flat. ChEES tunes its step to the edges and crawls across the flat
    lobes between them: on Kepler-12 b, HAT-P-1 b and Kepler-43 b (Gaia DR3)
    every chain crossed the whole period range, but R-hat on dP was still
    1.02-1.21 at the 16,384-draw cap.

    How. Each coordinate in turn is drawn from its conditional given every
    other parameter, by the same independence proposal as
    :class:`GridGibbs` (a cell of a grid over the coordinate's whole box by
    its mass, uniform within it) and a Metropolis-Hastings correction
    against the exact log-density, so any grid is exact. The box is the
    seeded peak's (:func:`turin.params.gaia_peak_box`), so the move never
    leaves it. Nothing factorises here: each grid point is one full
    evaluation, ``2 n_grid`` per chain per sweep, which a Gaia light curve
    (~150 points) makes cheap.

    ``lp`` is the float32 model-space log-density the sampler targets, prior
    included; ``units`` the two resolution units, for the log; ``de``, if
    given, an :class:`EnsembleDE` run after each sweep.
    """

    def __init__(self, lp, transform, layout, *, units, n_grid=N_GRID,
                 segment=SEGMENT, batch=16384, seed=0, de=None):
        if not layout.bands:
            raise ValueError("the ephemeris Gibbs move is for Gaia fits")
        self.lp, self.transform, self.layout = lp, transform, layout
        self.de = de
        self.label = "the period and epoch" + (
            "" if de is None else
            f", then {de.n_steps} DE steps on {', '.join(de.col_names)},")
        self.segment = int(segment)
        self.batch = int(batch)
        self.rng = np.random.default_rng(seed)
        self.cols = (layout.index("dP"), layout.index("dtau0"))
        self.units = np.asarray(units, dtype=np.float64)
        self.n_grid = int(n_grid)
        lo = layout.lo[list(self.cols)]
        hi = layout.hi[list(self.cols)]
        self.lo, self.hi = lo, hi
        self.edges = lo[:, None] + (hi - lo)[:, None] * np.linspace(
            0.0, 1.0, self.n_grid + 1)[None, :]                # (2, G+1)
        self.width = (hi - lo) / self.n_grid
        self.mids = 0.5 * (self.edges[:, 1:] + self.edges[:, :-1])

    def _log_prob(self, V):
        out = []
        for s in range(0, V.shape[0], self.batch):
            out.append(np.array(self.lp(
                mx.array(V[s:s + self.batch], dtype=mx.float32)),
                dtype=np.float64))
        return np.concatenate(out)

    def _step(self, v, i):
        """Redraw coordinate ``i`` (0: dP, 1: dtau0) of every chain."""
        C, G, col = v.shape[0], self.n_grid, self.cols[i]
        V = np.repeat(v, G, axis=0)
        V[:, col] = np.tile(self.mids[i], C)
        L = self._log_prob(V).reshape(C, G)
        # a non-finite cell (outside a prior's support) gets no mass
        L = np.where(np.isfinite(L), L, -np.inf)
        logq = L - L.max(axis=1, keepdims=True)
        logq -= np.log(np.exp(logq).sum(axis=1, keepdims=True))

        cdf = np.cumsum(np.exp(logq), axis=1)
        j = np.minimum((cdf < self.rng.uniform(size=(C, 1))).sum(axis=1),
                       G - 1)
        new = self.edges[i, j] + self.width[i] * self.rng.uniform(size=C)
        span = self.hi[i] - self.lo[i]
        new = np.clip(new, self.lo[i] + 1e-9 * span, self.hi[i] - 1e-9 * span)
        old = v[:, col].copy()
        j_old = np.clip(((old - self.lo[i]) // self.width[i]).astype(int),
                        0, G - 1)

        v_new = v.copy()
        v_new[:, col] = new
        rows = np.arange(C)
        with np.errstate(invalid="ignore"):
            log_a = (self._log_prob(v_new) - self._log_prob(v)
                     + logq[rows, j_old] - logq[rows, j])
        acc = np.log(self.rng.uniform(size=C)) < log_a
        v[:, col] = np.where(acc, new, old)
        jump = acc & (np.abs(new - old) > self.units[i])
        return acc.mean(), jump.mean()

    def sweep(self, u):
        """One sweep (dP, then dtau0) over every chain. ``u``: ``(n_chains,
        dim)`` unconstrained positions. Returns the new positions and an
        :class:`EphemerisSweepStats`."""
        v = self.transform.model_np(np.asarray(u, dtype=np.float64))
        acc, jump = zip(*(self._step(v, i) for i in (0, 1)))
        u = self.transform.from_model_np(v)
        st = EphemerisSweepStats(accept=np.array(acc), jump=np.array(jump))
        if self.de is not None:
            u, st.de_accept, move = self.de.run(u)
            st.de_move = move[:2]
        return u, st

    def describe(self, stats, t_gibbs, elapsed):
        """One log line per round."""
        acc = np.mean([s.accept for s in stats], axis=0)
        jump = np.mean([s.jump for s in stats], axis=0)
        line = (f"  ephemeris Gibbs: {len(stats)} sweeps, {t_gibbs:.1f}s "
                f"({t_gibbs / max(elapsed, 1e-9):.0%} of the round), MH "
                f"acceptance dP {acc[0]:.2f}, dtau0 {acc[1]:.2f}; chains "
                f"moving more than one resolution unit per sweep: dP "
                f"{jump[0]:.0%}, dtau0 {jump[1]:.0%}")
        if self.de is None:
            return line
        move = np.mean([s.de_move for s in stats], axis=0)
        return (line + f"\n  DE steps: acceptance "
                f"{np.mean([s.de_accept for s in stats]):.3f}; rms move per "
                f"sweep in the ensemble's sd: dP {move[0]:.2f}, dtau0 "
                f"{move[1]:.2f} (an independent redraw is 1.41)")


#: Differential-evolution steps after each ephemeris Gibbs sweep. At
#: Kepler-12 b's (Gaia DR3) final positions, 512 chains, 100 steps took
#: 0.35 s and moved dP and dtau0 by ~1.0 sd rms, 200 by 1.3 (1.41 is an
#: independent redraw); a Gibbs sweep took 0.27 s.
DE_STEPS = 100
#: The coordinates DE proposes along: the geometry, through which the
#: period and epoch are coupled to the shape. Over all 14 coordinates the
#: same steps moved dP and dtau0 half as far (acceptance 1.9% vs 1.6%).
DE_COLS = ("dP", "dtau0", "k", "beta", "T14")
#: Share of proposals at gamma = 1, which jumps between the ensemble's
#: clusters (ter Braak 2006).
DE_UNIT_GAMMA = 0.1
#: Gaussian jitter added to every proposal, in u, so the move is
#: irreducible; far below any posterior width.
DE_JITTER = 1e-6


class EnsembleDE:
    """Differential-evolution Metropolis steps (ter Braak 2006) on a Gaia
    fit's geometric coordinates, run after each :class:`EphemerisGibbs`
    sweep.

    Why it exists. With one point in transit per band (Kepler-12 b), that
    point's depth fixes where it lies inside the transit for a given k, b
    and T14, so the epoch moves with the shape. The Gibbs move redraws dP
    and dtau0 at a fixed shape, where each chain's conditional is 3-6x
    narrower than the marginal, and ChEES crawls along the coupled
    direction. A proposal along the difference of two other chains follows
    the ensemble's own shape, coupling included.

    Why it is exact. The chains are split at random into two halves; each
    chain of one half proposes ``x + gamma (x_j - x_k) + eps`` from two
    distinct chains of the other half, which is held fixed meanwhile, so the
    proposal is symmetric and a Metropolis test against the sampler's own
    log-density in u (Jacobian included) leaves the joint posterior of all
    chains invariant; then the halves swap. The chains become dependent,
    as in any ensemble sampler, and the move cannot reach a region no chain
    spans: the Gibbs move and ChEES explore, DE moves along what they found.

    ``target`` is the anvil target the sampler runs (``log_prob(u)``).
    """

    def __init__(self, target, layout, *, n_steps=DE_STEPS, cols=DE_COLS,
                 seed=0):
        self.target = target
        self.n_steps = int(n_steps)
        self.col_names = tuple(cols)
        self.cols = np.array([layout.index(c) for c in cols])
        self.gamma = 2.38 / np.sqrt(2 * len(self.cols))
        self.rng = np.random.default_rng(seed)

    def _log_prob(self, U):
        return np.array(self.target.log_prob(mx.array(U, dtype=mx.float32)),
                        dtype=np.float64)

    def run(self, u):
        """``n_steps`` steps of every chain from ``u`` (``(n_chains,
        dim)``). Returns the new positions, the acceptance, and each
        coordinate's rms move in units of the ensemble's sd at the start."""
        u = np.array(u, dtype=np.float64)
        C, d, cols, rng = u.shape[0], len(self.cols), self.cols, self.rng
        if C < 6:
            raise ValueError("the DE move needs at least 6 chains")
        u0 = u[:, cols].copy()
        lp = self._log_prob(u)
        acc = []
        for _ in range(self.n_steps):
            perm = rng.permutation(C)
            halves = (perm[:C // 2], perm[C // 2:])
            for A, B in (halves, halves[::-1]):
                n, m = len(A), len(B)
                j = rng.integers(0, m, n)
                k = (j + rng.integers(1, m, n)) % m      # k != j
                gam = np.where(rng.uniform(size=(n, 1)) < DE_UNIT_GAMMA,
                               1.0, self.gamma)
                prop = u[A]
                prop[:, cols] += (gam * (u[B[j]][:, cols] - u[B[k]][:, cols])
                                  + DE_JITTER * rng.standard_normal((n, d)))
                lp_new = self._log_prob(prop)
                with np.errstate(invalid="ignore"):
                    ok = np.log(rng.uniform(size=n)) < lp_new - lp[A]
                u[A[ok]] = prop[ok]
                lp[A[ok]] = lp_new[ok]
                acc.append(ok.mean())
        sd = np.maximum(u0.std(axis=0), 1e-300)
        move = np.sqrt(np.mean((u[:, cols] - u0) ** 2, axis=0)) / sd
        return u, float(np.mean(acc)), move
