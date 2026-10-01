"""Grid-Gibbs: the per-epoch timing move must be exact.

The move exists so chains can cross between timing modes that ChEES never
crosses. That is only worth having if it leaves the posterior invariant, so
the central test is distributional: with the shape held fixed, chains that
all start at one point must, after a few sweeps, be distributed as the exact
conditional of each epoch time -- including an epoch with two bumps.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from turin import capabilities, gibbs, likelihood, params, prep
from tests.test_seeding import (EPH, K_T, B_T, T14_T, P_REF, build_dataset)


def _setup(dtau, yerr, decoy=None, tau_half=0.12):
    ed, cen, _, n_ep = build_dataset(dtau, yerr=yerr, decoy=decoy)
    layout = params.ttv_layout(EPH, cen, tau_half)
    orders = np.full(n_ep, 2)
    kw = dict(num_resample=7, exposure_time=29.4 / 1440, fp64=False)
    _, transform, lp, _ = likelihood.build_target(
        layout, cen, ed, orders, n_chains_hint=4096, **kw)
    shape = np.array([K_T, B_T / (1 + K_T), T14_T, 0.30, 0.225])
    return layout, transform, lp, n_ep, shape


def test_epoch_terms_sum_to_the_log_density_minus_priors():
    layout, transform, lp, n_ep, shape = _setup(0.0, 1.5e-4)
    rng = np.random.default_rng(0)
    v = np.tile(np.concatenate([shape, np.zeros(n_ep)]), (8, 1))
    v[:, 5:] += rng.uniform(-0.05, 0.05, (8, n_ep))
    vm = mx.array(v, dtype=mx.float32)
    terms = np.array(lp.epoch_log_lik(vm), dtype=np.float64)
    total = np.array(lp(vm), dtype=np.float64)
    prior = np.array(lp.log_prior(lp.unpack(vm)), dtype=np.float64)
    assert terms.shape == (8, n_ep)
    # float32 sums in a different order: compare at float32 resolution
    np.testing.assert_allclose(terms.sum(axis=1), total - prior, rtol=1e-6)


def _exact_conditional(lp, layout, shape, i, n=4000):
    lo, hi = layout.lo[5 + i], layout.hi[5 + i]
    grid = np.linspace(lo, hi, n + 2)[1:-1]
    V = np.tile(np.concatenate([shape, np.zeros(layout.n_epochs)]), (n, 1))
    V[:, 5 + i] = grid
    l = np.array(lp.epoch_log_lik(mx.array(V, dtype=mx.float32)),
                 dtype=np.float64)[:, i]
    l = l + params.ttv_log_prior_np(grid[:, None], P_REF)
    p = np.exp(l - l.max())
    return grid, p / p.sum()


@pytest.mark.parametrize("decoy", [None, 0.07])
def test_sweeps_reproduce_the_exact_conditional(decoy):
    """With shape fixed, the move alone must sample p(dtau_i | shape, D).

    A low-SNR dataset (each transit ~2 sigma) gives broad conditionals; the
    decoy case adds a second dip 0.07 d away in every epoch, so each epoch's
    conditional has two separated bumps -- the case ChEES cannot cross.
    """
    layout, transform, lp, n_ep, shape = _setup(0.0, 2.5e-3, decoy=decoy)
    C = 4096
    v0 = np.tile(np.concatenate([shape, np.zeros(n_ep)]), (C, 1))
    u = transform.from_model_np(v0)
    move = gibbs.GridGibbs(lp, transform, layout, T14=T14_T, n_grid=256,
                           batch=4096, seed=1)
    for _ in range(4):
        u, st = move.sweep(u)
    v = transform.model_np(u)
    np.testing.assert_allclose(v[:, :5], v0[:, :5], rtol=1e-5)
    assert st.accept.min() > 0.5

    for i in range(n_ep):
        grid, p = _exact_conditional(lp, layout, shape, i)
        edges = np.linspace(layout.lo[5 + i], layout.hi[5 + i], 41)
        ref = np.histogram(grid, bins=edges, weights=p)[0]
        got = np.histogram(v[:, 5 + i], bins=edges)[0] / C
        tvd = 0.5 * np.abs(ref - got).sum()
        # 4096 draws over 40 bins: sampling noise alone gives ~0.03-0.05
        assert tvd < 0.08, (i, tvd)


def test_two_bump_conditional_is_crossed_in_one_sweep():
    """Chains all start in one bump; after one sweep a share proportional to
    the other bump's mass must be there."""
    layout, transform, lp, n_ep, shape = _setup(0.0, 2.5e-3, decoy=0.07)
    grid, p = _exact_conditional(lp, layout, shape, 0)
    far_mass = p[grid > 0.035].sum()
    # genuinely two bumps; chains start in the one at 0, whichever is larger
    assert 0.05 < far_mass < 0.95, far_mass

    C = 4096
    v0 = np.tile(np.concatenate([shape, np.zeros(n_ep)]), (C, 1))
    move = gibbs.GridGibbs(lp, transform, layout, T14=T14_T, n_grid=256,
                           batch=4096, seed=2)
    u, st = move.sweep(transform.from_model_np(v0))
    v = transform.model_np(u)
    got = np.mean(v[:, 5] > 0.035)
    assert abs(got - far_mass) < 0.05, (got, far_mass)
    # the logged statistic must count those crossings; T14 here is 0.175 d,
    # wider than the 0.07 d bump separation, so ask with a narrower one
    move = gibbs.GridGibbs(lp, transform, layout, T14=0.03, n_grid=256,
                           batch=4096, seed=2)
    _, st = move.sweep(transform.from_model_np(v0))
    assert abs(st.mode_change[0] - far_mass) < 0.06, (st.mode_change[0],
                                                       far_mass)


def test_set_positions_fallback_refuses_an_unknown_state():
    class FakeState:
        kernel = "EnsembleKernel"
        state = {"u": None, "log_prob": None}

    with pytest.raises(RuntimeError, match="gibbsgrid=off"):
        capabilities.set_positions(FakeState(), np.zeros((2, 3)), None)


def test_grid_gibbs_refuses_a_lineph_layout():
    layout = params.lineph_layout(EPH)
    with pytest.raises(ValueError, match="TTV"):
        gibbs.GridGibbs(None, None, layout, T14=T14_T)
