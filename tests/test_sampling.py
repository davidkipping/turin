"""The sampling layer, including a full injection-recovery.

The recovery tests are the ones that matter: they exercise the whole stack
(prep -> model -> profile -> likelihood -> seeding -> anvil) against data whose
answer is known. Marked ``slow`` so the fast suite stays fast.
"""

from __future__ import annotations

import anvil
import mlx.core as mx
import numpy as np
import pytest

from turin import (capabilities, likelihood, model as M, params, prep,
                   sampling, seeding)

P_REF, EPOCH, DUR_H = 9.3456, 120.5, 4.2
K_T, B_T, Q1_T, Q2_T = 0.095, 0.30, 0.35, 0.25
T14_T = DUR_H / 24.0
YERR = 2.5e-4
EPH = {"period": P_REF, "epoch": EPOCH, "duration": DUR_H, "depth": 9025.0}


def make_dataset(dtau_true=0.0, n_periods=10.0, seed=7, yerr=YERR):
    rng = np.random.default_rng(seed)
    t = np.arange(100.0, 100.0 + n_periods * P_REF, 29.4 / 1440.0)
    tw, fw, ew = prep.extract_near_transit_data(
        t, np.ones_like(t), np.full(t.size, yerr), P_REF, EPOCH, DUR_H)
    ed = prep.segment_epochs(tw, fw, ew, P_REF, EPOCH, DUR_H)
    cen = prep.centering_constants(ed, EPH)
    n_ep = ed["n_epochs"]
    dtau = np.broadcast_to(np.asarray(dtau_true, dtype=np.float64),
                           (n_ep,)).copy()

    grid = M.build_grid(cen, prep.supersample_offsets(29.4 / 1440, 7),
                        dtype=mx.float64)
    col = lambda v: mx.array([[float(v)]], dtype=mx.float64)
    with mx.stream(mx.cpu):
        f = np.array(M.transit_flux(
            grid, mid=M.mid_times_ttv(grid, mx.array(dtau[None, :],
                                                     dtype=mx.float64)),
            k=col(K_T), b=col(B_T), T14=col(T14_T), q1=col(Q1_T),
            q2=col(Q2_T), period=col(P_REF)), dtype=np.float64)[0]

    x = (ed["times_padded"] - ed["epoch_centers"][:, None]) / ed["half_window"]
    base = 1.0 + 2.5e-4 * x - 1.8e-4 * (1.5 * x**2 - 0.5)
    ed = dict(ed)
    ed["flux_padded"] = np.where(
        ed["mask"] > 0, f * base + rng.normal(0.0, yerr, f.shape), 1.0)
    return ed, cen, dtau, n_ep, np.full(n_ep, 2)


def lineph_target(ed, cen, orders, **kw):
    layout = params.lineph_layout(EPH)
    target, transform, lp, hi = likelihood.build_target(
        layout, cen, ed, orders, num_resample=7, exposure_time=29.4 / 1440,
        n_chains_hint=256, **kw)
    return layout, target, transform, lp


V_TRUE = np.array([0.0, 0.0, K_T, B_T / (1 + K_T), T14_T, Q1_T, Q2_T])


# -- configuration -----------------------------------------------------

def test_config_resolves_chains_against_dimension():
    cfg = sampling.SamplerConfig(n_chains=64, dense=True)
    # dense needs 4*dim chains or anvil silently downgrades it
    got = cfg.for_mode(40)
    assert got.n_chains >= 160 and got.dense is True

    auto = sampling.SamplerConfig(n_chains=512).for_mode(7)
    assert auto.dense is True and auto.n_chains == 512

    ens = sampling.SamplerConfig(sampler="ensemble", n_walkers=128).for_mode(7)
    assert ens.n_chains % 2 == 0 and ens.n_chains >= 2 * 7
    assert ens.dense is False and ens.n_samples == 20000


def test_make_kernel_rejects_unknown_sampler():
    ed, cen, _, _, orders = make_dataset()
    _, target, _, _ = lineph_target(ed, cen, orders, fp64=False)
    with pytest.raises(ValueError, match="unknown sampler"):
        sampling.make_kernel(target, sampling.SamplerConfig(sampler="nope"))


def test_ess_floor_is_lower_for_timing_parameters():
    assert sampling._ess_floor("k") == sampling.ESS_MIN
    assert sampling._ess_floor("dtau_3") == sampling.ESS_MIN_TAU
    assert sampling.ESS_MIN_TAU < sampling.ESS_MIN


def test_capabilities_detect_and_summarize():
    caps = capabilities.detect()
    assert isinstance(caps.anvil_resume, bool)
    assert isinstance(caps.metalplanet_flux_dev_from_tau, bool)
    text = caps.summary()
    assert "anvil" in text and "MetalPlanet" in text
    # the field is deliberately unknown until a run has happened
    assert caps.anvil_per_chain_divergences is None


def test_check_precision_passes_on_a_well_conditioned_fit():
    ed, cen, _, _, orders = make_dataset()
    layout, target, transform, _ = lineph_target(ed, cen, orders)
    u0 = mx.array(np.repeat(transform.from_model_np(V_TRUE)[None, :], 32,
                            axis=0).astype(np.float32))
    report = sampling.check_precision(target, u0, log=None, strict=True)
    assert report.max_abs_err < 0.1


# -- trapped-chain detection -------------------------------------------

class _FakeResults:
    """Minimal stand-in for anvil's Results, to test the health logic."""

    def __init__(self, logp, accept, extras=None):
        self._lp = np.asarray(logp, dtype=np.float64)
        self.accept_fraction = np.asarray(accept, dtype=np.float64)
        self.extras = extras or {}

    def get_log_prob(self, discard=0, thin=1, flat=False):
        return self._lp


def test_chain_health_flags_a_secondary_mode_chain():
    """The stretch-move failure: low log-prob, perfectly healthy acceptance."""
    rng = np.random.default_rng(0)
    lp = rng.normal(-700.0, 1.0, (200, 32))
    lp[:, 5] -= 1500.0                     # one chain in a deep secondary mode
    acc = np.full(32, 0.6)
    health = sampling.chain_health(_FakeResults(lp, acc))
    assert health.any_trapped and 5 in health.trapped
    assert "secondary mode" in health.reason[5]
    assert "WARNING" in health.describe()


def test_chain_health_distinguishes_a_boundary_trap():
    """Low log-prob plus constant divergence is the ChEES boundary trap."""
    rng = np.random.default_rng(1)
    lp = rng.normal(-700.0, 1.0, (100, 16))
    lp[:, 3] -= 9000.0
    acc = np.full(16, 0.6)
    acc[3] = 0.01
    div = np.zeros(16)
    div[3] = 100                            # diverged on every iteration
    health = sampling.chain_health(
        _FakeResults(lp, acc, extras={"divergent_per_chain": div}))
    assert 3 in health.trapped
    assert "boundary trap" in health.reason[3]


def test_chain_health_passes_a_clean_ensemble():
    rng = np.random.default_rng(2)
    lp = rng.normal(-700.0, 1.0, (200, 64))
    health = sampling.chain_health(_FakeResults(lp, np.full(64, 0.65)))
    assert not health.any_trapped
    assert "healthy" in health.describe()


def test_width_breakdown_exposes_inflation_from_trapped_chains():
    """The whole point: a couple of trapped chains must not silently widen a
    reported posterior."""
    ed, cen, _, _, orders = make_dataset()
    layout, target, transform, _ = lineph_target(ed, cen, orders, fp64=False)
    rng = np.random.default_rng(3)
    n_draws, n_chains = 60, 24
    u0 = transform.from_model_np(V_TRUE)
    chain = u0[None, None, :] + 0.01 * rng.standard_normal(
        (n_draws, n_chains, layout.dim))
    chain[:, :2, :] += 3.0                  # two chains far away

    class R:
        def get_chain(self, discard=0, thin=1, flat=False):
            c = chain.astype(np.float32)
            return c.reshape(-1, layout.dim) if flat else c

    health = sampling.ChainHealth(
        n_chains=n_chains, trapped=np.array([0, 1]),
        logp_median=np.zeros(n_chains), accept=np.zeros(n_chains),
        divergences=None, bulk_logp=0.0)
    rows = sampling.width_breakdown(transform, R(), list(layout.names), health)
    assert rows is not None
    # at least one parameter's all-chain width is clearly inflated
    assert max(r[3] for r in rows) > 2.0
    names = [r[0] for r in rows]
    assert names == list(layout.names)

    # and with nothing flagged there is nothing to compare
    clean = sampling.ChainHealth(
        n_chains=n_chains, trapped=np.array([], dtype=int),
        logp_median=np.zeros(n_chains), accept=np.zeros(n_chains),
        divergences=None, bulk_logp=0.0)
    assert sampling.width_breakdown(transform, R(), list(layout.names),
                                    clean) is None


# -- end to end --------------------------------------------------------

@pytest.mark.slow
def test_lineph_injection_recovery():
    """The whole stack must recover injected parameters within ~1 sigma."""
    ed, cen, _, n_ep, orders = make_dataset()
    layout, target, transform, lp = lineph_target(ed, cen, orders)

    v0 = seeding.initial_model_vector(layout, EPH)
    rng = np.random.default_rng(0)
    u_starts = (transform.from_model_np(v0)[None, :]
                + 0.25 * rng.standard_normal((128, layout.dim)))
    map_res = seeding.find_map(target, u_starts.astype(np.float32), n_iter=400)
    assert np.isfinite(map_res.log_prob_best)

    cfg = sampling.SamplerConfig(n_chains=256, n_warmup=400, n_samples=300,
                                 max_samples=1200, seed=3).for_mode(layout.dim)
    u0 = map_res.ball(cfg.n_chains, seed=1)
    sampling.check_precision(target, u0, log=None)
    results, verdict, total = sampling.run_rounds(
        target, list(layout.names), u0, cfg, log=lambda m: None)

    assert verdict.n_divergent < 0.02 * total * cfg.n_chains
    assert not verdict.health.any_trapped, verdict.health.describe()

    post = sampling.physical_draws(transform, results)
    truth = V_TRUE + layout.report_offset
    for i, name in enumerate(layout.names):
        med, sd = np.median(post[:, i]), np.std(post[:, i])
        assert sd > 0, name
        z = abs(med - truth[i]) / sd
        assert z < 3.0, f"{name}: {med:.6g} vs truth {truth[i]:.6g} ({z:.1f}s)"

    # the geometric parameters must actually be constrained, not prior-wide
    assert np.std(post[:, layout.index("k")]) < 0.2 * K_T
    assert np.std(post[:, layout.index("T14")]) < 0.1 * T14_T
    # and the period must be pinned far better than its prior half-width
    assert np.std(post[:, layout.index("dP")]) < 1e-3


@pytest.mark.slow
def test_ttv_injection_recovery_of_displaced_transits():
    """TTV mode plus template-sweep seeding must find displaced transits.

    hurin's headline failure mode, end to end: every transit is displaced by
    more than a cadence, which a zero-TTV initialization would not find.
    """
    shift = 0.025
    ed, cen, dtau, n_ep, orders = make_dataset(dtau_true=shift)
    layout = params.ttv_layout(EPH, cen, tau_half=0.08)
    target, transform, lp, _ = likelihood.build_target(
        layout, cen, ed, orders, num_resample=7, exposure_time=29.4 / 1440,
        n_chains_hint=256)

    # seed the timings from the template sweep at the default shape
    shape = seeding.default_shape_start(EPH, layout)
    b0 = M.impact_parameter(shape["beta"], shape["k"], layout.b_prior)
    seeds, gap, _, _ = seeding.template_sweep_taus(
        ed, cen, lp.design, tau_half=0.08, k=shape["k"], b=b0,
        T14=shape["T14"], q1=shape["q1"], q2=shape["q2"], period=P_REF,
        num_resample=7, exposure_time=29.4 / 1440, dtype=mx.float32)
    np.testing.assert_allclose(seeds, dtau, atol=0.01)

    v0 = seeding.initial_model_vector(layout, EPH, dtau=seeds)
    rng = np.random.default_rng(4)
    u_starts = (transform.from_model_np(v0)[None, :]
                + 0.1 * rng.standard_normal((128, layout.dim)))
    map_res = seeding.find_map(target, u_starts.astype(np.float32), n_iter=400)

    cfg = sampling.SamplerConfig(n_chains=256, n_warmup=400, n_samples=300,
                                 max_samples=900, seed=5).for_mode(layout.dim)
    u0 = map_res.ball(cfg.n_chains, seed=2)
    results, verdict, total = sampling.run_rounds(
        target, list(layout.names), u0, cfg, log=lambda m: None)

    post = sampling.physical_draws(transform, results)
    # every transit time must come back at its injected value
    for i in range(n_ep):
        j = 5 + i
        med, sd = np.median(post[:, j]), np.std(post[:, j])
        want = layout.report_offset[j] + dtau[i]
        assert sd < 0.01, f"epoch {i} timing width {sd}"
        assert abs(med - want) < max(4 * sd, 2e-3), (
            f"epoch {i}: {med:.5f} vs {want:.5f} (sd {sd:.5f})")
    # and the shape parameters are still recovered
    assert abs(np.median(post[:, layout.index("k")]) - K_T) < 0.2 * K_T
