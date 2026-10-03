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


def test_grid_gibbs_refuses_a_lineph_layout():
    layout = params.lineph_layout(EPH)
    with pytest.raises(ValueError, match="TTV"):
        gibbs.GridGibbs(None, None, layout, T14=T14_T)


def test_round_loop_moves_chains_through_with_positions():
    """The round loop with grid-Gibbs, end to end: chains are moved between
    segments with anvil's ResumeState.with_positions, the draws pool into
    one Continuation, and the move is logged."""
    from turin import sampling, seeding

    layout, transform, lp, n_ep, shape = _setup(0.0, 2.5e-3, decoy=0.07)
    ed, cen, _, _ = build_dataset(0.0, yerr=2.5e-3, decoy=0.07)
    target, transform, _, _ = likelihood.build_target(
        layout, cen, ed, np.full(n_ep, 2), num_resample=7,
        exposure_time=29.4 / 1440, n_chains_hint=64)
    cfg = sampling.SamplerConfig(n_chains=64, n_warmup=60, n_samples=100,
                                 max_samples=300, seed=3).for_mode(layout.dim)
    v0 = np.concatenate([shape, np.zeros(n_ep)])
    u_map = transform.from_model_np(v0[None, :])[0]
    res = seeding.MapResult(u_best=u_map, log_prob_best=0.0,
                            u_all=u_map[None, :], log_prob_all=np.zeros(1),
                            n_iter=0, top_spread=0.0)
    move = gibbs.GridGibbs(lp, transform, layout, T14=T14_T, n_grid=128,
                           batch=4096, seed=5)
    lines = []
    results, verdict, total = sampling.run_rounds(
        target, list(layout.names), res.ball(cfg.n_chains, seed=4), cfg,
        log=lines.append, move=move, segment=50)
    assert any("grid-Gibbs:" in l for l in lines)
    assert isinstance(results, sampling.Continuation)
    assert results.get_chain().shape == (total, 64, layout.dim)
    assert np.all(np.isfinite(results.get_log_prob()))


def test_epoch_snr_matches_the_injected_transit():
    """Per-epoch SNR = sqrt(2 dlnL) against a k->0 null that still fits its
    own baseline. On synthetic data with a curved baseline it must match the
    noise-free transit's sqrt(sum((F-1)/sigma)^2), epoch by epoch. (The old
    null, a flat 1.0 with no baseline, credited baseline structure to the
    transit.)"""
    from turin import model as M, pipeline

    yerr = 2.5e-3
    layout, transform, lp, n_ep, shape = _setup(0.0, yerr)
    ed, cen, _, _ = build_dataset(0.0, yerr=yerr)
    v_true = np.concatenate([shape, np.zeros(n_ep)])
    snr = pipeline._epoch_snr(lp, layout, v_true)

    grid = M.build_grid(cen, prep.supersample_offsets(29.4 / 1440, 7),
                        dtype=mx.float64, exp_time=29.4 / 1440)
    col = lambda v: mx.array([[float(v)]], dtype=mx.float64)
    with mx.stream(mx.cpu):
        mid = M.mid_times_ttv(grid, mx.zeros((1, n_ep), dtype=mx.float64))
        f = np.array(M.transit_flux(grid, mid=mid, k=col(K_T), b=col(B_T),
                                    T14=col(T14_T), q1=col(0.30),
                                    q2=col(0.225), period=col(P_REF)),
                     dtype=np.float64)[0]
    mask = np.asarray(ed["mask"]) > 0
    expected = np.sqrt(np.sum(np.where(mask, (f - 1.0) / yerr, 0.0) ** 2,
                              axis=1))
    assert expected.min() > 8                      # a real detection per epoch
    # noise moves 2 dlnL by ~2 expected-SNR, i.e. SNR by ~1
    np.testing.assert_allclose(snr, expected, atol=3.0)
