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
                        dtype=mx.float64, exp_time=29.4 / 1440)
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
    assert isinstance(caps.metalplanet_flux_dev_from_tau, bool)
    text = caps.summary()
    assert "anvil" in text and "MetalPlanet" in text
    assert "yes  anvil >= 0.3.0" in text


@pytest.mark.parametrize("version,ok", [
    ("0.3.0", True), ("0.3.0.dev2", True), ("0.3.1", True), ("1.0", True),
    ("0.2.0", False), ("0.1.0.dev0", False), ("unknown", False)])
def test_anvil_version_gate(version, ok):
    assert capabilities.anvil_ok(version) is ok


def test_require_anvil_names_the_upgrade_command(monkeypatch):
    import anvil

    assert capabilities.require_anvil() == anvil.__version__
    monkeypatch.setattr(anvil, "__version__", "0.2.0")
    with pytest.raises(SystemExit, match="pip install -U"):
        capabilities.require_anvil()
    assert "NO   anvil >= 0.3.0" in capabilities.detect().summary()


def test_assess_matches_a_per_parameter_diagnose_reference():
    """The single memory-budgeted diagnose call replaced a per-parameter
    loop; it must give the same R-hat and ESS, including a real R-hat
    signal and autocorrelated chains."""
    import anvil

    rng = np.random.default_rng(2)
    S, C, D = 3000, 64, 6
    x = np.zeros((S, C, D), np.float32)
    e = rng.standard_normal((S, C, D)).astype(np.float32)
    for t in range(1, S):
        x[t] = 0.9 * x[t - 1] + e[t]
    x[:, :8, 2] += 3.0

    class R:
        extras = {"n_divergent": 0}
        accept_fraction = np.full(C, 0.8)
        warmup_trace = None

        def get_chain(self, discard=0, thin=1, flat=False):
            c = x[discard::thin]
            return c.reshape(-1, D) if flat else c

        def get_log_prob(self, discard=0, thin=1, flat=False):
            lp = -0.5 * np.sum(x[discard::thin] ** 2, axis=-1)
            return lp.reshape(-1) if flat else lp

    v = sampling.assess(R(), list("abcdef"))
    ref = [anvil.diagnose(x[..., i:i + 1]) for i in range(D)]
    np.testing.assert_array_equal(v.rhat, [float(r.rhat[0]) for r in ref])
    # ESS goes through anvil's float32 GPU autocovariance, accumulated in
    # different groupings by the joint and per-parameter calls: measured
    # 1.2e-6 relative. ESS is reported as an integer and gated at 100/400.
    np.testing.assert_allclose(v.ess, [float(r.ess_bulk[0]) for r in ref],
                               rtol=1e-5)
    assert v.worst_rhat[0] == "c" and v.worst_rhat[1] > 1.05


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


# -- resumed continuation ----------------------------------------------

class _Gaussian:
    """A trivial target with anvil's LogDensity shape, for driver tests."""

    dim = 4
    supports_grad = True

    def log_prob(self, u):
        return -0.5 * mx.sum(u * u, axis=-1)

    def log_prob_and_grad(self, u):
        out, vjps = mx.vjp(self.log_prob, [u],
                           [mx.ones(u.shape[:1], dtype=u.dtype)])
        return out[0], vjps[0]


def _drive(max_samples=120, n_samples=40, seed=3):
    t = _Gaussian()
    u0 = mx.array(np.random.default_rng(0).standard_normal(
        (64, t.dim)).astype(np.float32))
    cfg = sampling.SamplerConfig(n_chains=64, n_warmup=120,
                                 n_samples=n_samples, max_samples=max_samples,
                                 seed=seed).for_mode(t.dim)
    logs = []
    res, verdict, total = sampling.run_rounds(
        t, list("abcd"), u0, cfg, log=logs.append)
    return t, res, verdict, total, "\n".join(logs)


def test_extension_is_a_continuation_and_pools_its_draws():
    """With resume, later rounds continue the same chains, so they pool."""
    t, res, verdict, total, text = _drive()
    assert "continuation, adaptation frozen" in text
    assert "NEW chain" not in text
    assert isinstance(res, sampling.Continuation)
    # every requested draw is reported, not just the last round's
    assert verdict.n_draws == total
    chain = res.get_chain()
    assert chain.shape == (total, 64, t.dim)
    assert res.get_log_prob().shape == (total, 64)
    assert res.get_chain(flat=True).shape == (total * 64, t.dim)


def test_a_pooled_continuation_still_samples_the_right_distribution():
    """Pooling is only valid if the segments really are one chain."""
    t, res, verdict, total, text = _drive()
    flat = res.get_chain(flat=True).astype(np.float64)
    np.testing.assert_allclose(flat.mean(axis=0), 0.0, atol=0.05)
    np.testing.assert_allclose(flat.std(axis=0), 1.0, rtol=0.05)
    assert not verdict.health.any_trapped


def test_continuation_delegates_state_saving_to_the_latest_segment(tmp_path):
    """Resume state must describe where the chains ARE, not where they were."""
    import anvil

    t, res, _, _, _ = _drive()
    path = str(tmp_path / "state.npz")
    res.save_state(path)
    state = anvil.load_state(path)
    again = anvil.run(anvil.ChEESHMC(t, max_leapfrog=32), t, resume=state,
                      n_samples=10, progress=False)
    assert again.get_chain().shape == (10, 64, t.dim)


def test_per_chain_divergences_are_now_available():
    """anvil reports the vector, so the trapped-chain check can tell the two
    failure modes apart rather than guessing."""
    _, res, _, _, _ = _drive()
    per_chain = res.extras["divergent_per_chain"]
    assert len(per_chain) == 64
    assert int(np.sum(per_chain)) == int(res.extras["n_divergent"])


def test_continuation_presents_the_results_interface_turin_consumes():
    """A hand-built Continuation, so the contract is pinned without sampling."""
    class Seg:
        def __init__(self, chain, lp, n=2):
            self._c, self._lp = chain, lp
            self.final_state = {"u": chain[-1]}
            self.warmup_trace = {"iter": [1]}
            self.extras = {"n_divergent": n}
            self.accept_fraction = np.full(chain.shape[1], 0.7)
            self.n_chains, self.dim = chain.shape[1], chain.shape[2]
            self.n_warmup, self.thin = 10, 1

        def get_chain(self, discard=0, thin=1, flat=False):
            return self._c

        def get_log_prob(self, discard=0, thin=1, flat=False):
            return self._lp

    rng = np.random.default_rng(0)
    segs = [Seg(rng.standard_normal((5, 3, 2)), rng.standard_normal((5, 3)))
            for _ in range(3)]
    cont = sampling.Continuation(segs)
    assert cont.get_chain().shape == (15, 3, 2)
    assert cont.get_log_prob().shape == (15, 3)
    assert cont.get_chain(flat=True).shape == (45, 2)
    # accounting comes from the latest segment, as anvil specifies
    assert cont.extras is segs[-1].extras
    assert cont.final_state is segs[-1].final_state
    # warmup happened in the first segment
    assert cont.warmup_trace is segs[0].warmup_trace
    # and the health check reads it without special-casing
    health = sampling.chain_health(cont)
    assert health.n_chains == 3
