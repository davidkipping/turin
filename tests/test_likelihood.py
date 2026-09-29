"""The assembled log-density: priors, epoch blocking, and the anvil contract.

The prior tests check the *claims* hurin makes about its (b, k)
parameterization -- uniform marginal in k, 1:1 grazing odds -- rather than
just re-deriving its algebra, since those claims are the reason the
parameterization exists.
"""

from __future__ import annotations

import anvil
import mlx.core as mx
import numpy as np
import pytest

from turin import likelihood, model as M, params, prep

P_REF, EPOCH, DUR_H = 9.3456, 120.5, 4.2
K_T, B_T, T14_T, Q1_T, Q2_T = 0.085, 0.35, DUR_H / 24.0, 0.30, 0.225
YERR = 2e-4
EPH = {"period": P_REF, "epoch": EPOCH, "duration": DUR_H, "depth": 7000.0}


@pytest.fixture(scope="module")
def dataset():
    """A synthetic multi-epoch light curve with per-epoch quadratic trends."""
    rng = np.random.default_rng(11)
    t = np.arange(100.0, 175.0, 29.4 / 1440.0)
    t = t[(t < 128.0) | (t > 130.2)]
    tw, fw, ew = prep.extract_near_transit_data(
        t, np.ones_like(t), np.full(t.size, YERR), P_REF, EPOCH, DUR_H)
    ed = prep.segment_epochs(tw, fw, ew, P_REF, EPOCH, DUR_H)
    cen = prep.centering_constants(ed, EPH)
    n_ep = ed["n_epochs"]

    grid = M.build_grid(cen, prep.supersample_offsets(29.4 / 1440, 7),
                        dtype=mx.float64, exp_time=29.4 / 1440)
    one = lambda v: mx.array([[float(v)]], dtype=mx.float64)
    with mx.stream(mx.cpu):
        f_true = np.array(M.transit_flux(
            grid, mid=mx.zeros((1, n_ep), dtype=mx.float64), k=one(K_T),
            b=one(B_T), T14=one(T14_T), q1=one(Q1_T), q2=one(Q2_T),
            period=one(P_REF)), dtype=np.float64)[0]

    x = (ed["times_padded"] - ed["epoch_centers"][:, None]) / ed["half_window"]
    trend = 1.0 + 3e-4 * x + 2e-4 * (1.5 * x**2 - 0.5) - 1e-4
    obs = f_true * trend + rng.normal(0.0, YERR, f_true.shape)
    ed = dict(ed)
    ed["flux_padded"] = np.where(ed["mask"] > 0, obs, 1.0)
    return ed, cen, np.full(n_ep, 2)


@pytest.fixture(scope="module")
def lineph(dataset):
    ed, cen, orders = dataset
    layout = params.lineph_layout(EPH, b_prior="transiting")
    target, transform, lp, hi = likelihood.build_target(
        layout, cen, ed, orders, num_resample=7, exposure_time=29.4 / 1440,
        n_chains_hint=64)
    return layout, target, transform, lp, hi


V_TRUE = np.array([0.0, 0.0, K_T, B_T / (1 + K_T), T14_T, Q1_T, Q2_T])


def u_ball(transform, dim, n=8, spread=1e-3, seed=0):
    u0 = transform.from_model_np(V_TRUE)
    r = np.random.default_rng(seed)
    return mx.array((u0[None, :] + spread * r.standard_normal((n, dim))
                     ).astype(np.float32))


# -- priors ------------------------------------------------------------

def test_bk_prior_gives_uniform_k_and_equal_grazing_odds():
    """hurin's two claims for the ``transiting`` prior, tested as claims.

    Sampling (beta, k) uniformly on the unit square and weighting by
    exp(bk_log_prior) must reproduce: a uniform marginal in k, and marginal
    prior odds of 1:1 for grazing (b > 1-k) against non-grazing.
    """
    rng = np.random.default_rng(3)
    n = 400_000
    beta = rng.uniform(0.0, 1.0, n)
    k = rng.uniform(0.0, 1.0, n)
    with mx.stream(mx.cpu):
        w = np.exp(np.array(params.bk_log_prior(
            mx.array(beta.reshape(-1, 1), dtype=mx.float64),
            mx.array(k.reshape(-1, 1), dtype=mx.float64), "transiting"),
            dtype=np.float64))

    b = beta * (1.0 + k)
    grazing = b > (1.0 - k)
    odds = w[grazing].sum() / w[~grazing].sum()
    assert abs(odds - 1.0) < 0.02, f"grazing:non-grazing odds {odds}"

    # marginal p(k) uniform: equal weight in each quartile of k
    q = [w[(k >= a) & (k < a + 0.25)].sum() for a in (0.0, 0.25, 0.5, 0.75)]
    q = np.array(q) / np.sum(q)
    assert np.all(np.abs(q - 0.25) < 0.01), f"p(k) quartiles {q}"


def test_bk_prior_nongrazing_and_box():
    with mx.stream(mx.cpu):
        col = lambda v: mx.array([[v]], dtype=mx.float64)
        ng = float(params.bk_log_prior(col(0.5), col(0.3), "nongrazing")[0])
        np.testing.assert_allclose(ng, np.log1p(-0.3), rtol=1e-12)
        bx = params.bk_log_prior(col(0.5), col(0.3), "box")
        assert float(bx[0]) == 0.0
        with pytest.raises(ValueError, match="unknown b_prior"):
            params.bk_log_prior(col(0.5), col(0.3), "nope")


def test_bk_prior_is_finite_at_the_grazing_edge():
    """b -> 1+k must not send the log-prior to -inf and kill the gradient."""
    with mx.stream(mx.cpu):
        beta = mx.array(np.array([[1.0], [1.0 - 1e-9], [0.999999]]),
                        dtype=mx.float64)
        k = mx.full((3, 1), 0.3, dtype=mx.float64)
        v = np.array(params.bk_log_prior(beta, k, "transiting"),
                     dtype=np.float64)
        assert np.all(np.isfinite(v)), v
        g = mx.grad(lambda bb: mx.sum(
            params.bk_log_prior(bb, k, "transiting")))(beta)
        assert np.all(np.isfinite(np.array(g)))


def test_ttv_prior_matches_hurin_formula():
    """Kipping (2025) ramp: free below the threshold, quadratic above."""
    dtau = np.array([[0.0, 1e-5, 1e-3, 0.05]])
    with mx.stream(mx.cpu):
        got = float(params.ttv_log_prior(
            mx.array(dtau, dtype=mx.float64), P_REF)[0])
    r = np.log(np.abs(dtau / P_REF) + 1e-30)
    excess = np.maximum(r - params.TTV_LOG_THRESHOLD, 0.0)
    expected = -0.5 * np.sum((excess / params.TTV_LOG_SCALE) ** 2)
    np.testing.assert_allclose(got, expected, rtol=1e-12)
    np.testing.assert_allclose(
        params.ttv_log_prior_np(dtau, P_REF)[0], expected, rtol=1e-12)

    # a TTV well below |TTV/P| ~ 6.5e-4 costs nothing at all
    small = np.array([[1e-4 * P_REF * 1e-3]])
    with mx.stream(mx.cpu):
        assert float(params.ttv_log_prior(
            mx.array(small, dtype=mx.float64), P_REF)[0]) == 0.0
    # and a large one is penalized
    assert params.ttv_log_prior_np(np.array([[0.3]]), P_REF)[0] < -1.0


# -- the log-density ---------------------------------------------------

def test_log_prob_shape_and_absolute_value(lineph, dataset):
    layout, target, transform, lp, hi = lineph
    ed, _, _ = dataset
    u = u_ball(transform, layout.dim, n=8)
    val = np.array(target.log_prob(u), dtype=np.float64)
    assert val.shape == (8,)
    assert np.all(np.isfinite(val))

    # chi2/N near 1 at the truth: the data were generated from this model
    v = transform.model_np(np.array(u, dtype=np.float64))
    lp_only = np.array(lp(mx.array(v.astype(np.float32))), dtype=np.float64)
    chi2_over_n = (lp.n_real - 2.0 * lp_only[0]) / lp.n_real
    assert 0.9 < chi2_over_n < 1.1, chi2_over_n

    # the recentred term is small; the constant that makes it absolute is not
    assert abs(lp_only[0]) < 50.0
    assert abs(lp.log_const) > 0.5 * lp.n_real


def test_batching_is_independent(dataset, lineph):
    """Row i of a batched call is row i's own log-density, and nothing else.

    Exactness is asserted in float64. In float32 the reduction order depends
    on the batch shape, so rows agree only to the float32 noise floor -- which
    ``validate_precision`` independently bounds at ~1e-3 for this target.
    """
    ed, cen, orders = dataset
    layout = params.lineph_layout(EPH)
    with mx.stream(mx.cpu):
        hi64 = likelihood.ProfiledTransitLogProb(
            layout, cen, ed, orders, num_resample=7,
            exposure_time=29.4 / 1440, n_chains_hint=8, dtype=mx.float64)
        rng = np.random.default_rng(2)
        v = V_TRUE[None, :] + 1e-3 * rng.standard_normal((6, layout.dim))
        many = np.array(hi64(mx.array(v, dtype=mx.float64)), dtype=np.float64)
        for i in range(v.shape[0]):
            one = np.array(hi64(mx.array(v[i:i + 1], dtype=mx.float64)),
                           dtype=np.float64)
            np.testing.assert_allclose(many[i], one[0], rtol=1e-12)

    _, target, transform, lp, _ = lineph
    u = u_ball(transform, layout.dim, n=6, spread=5e-3, seed=2)
    many32 = np.array(target.log_prob(u), dtype=np.float64)
    for i in range(u.shape[0]):
        one32 = np.array(target.log_prob(u[i:i + 1]), dtype=np.float64)
        assert abs(many32[i] - one32[0]) < 0.02


@pytest.mark.parametrize("budget", [1 << 31, 1 << 16, 1])
def test_epoch_blocking_does_not_change_the_answer(dataset, budget):
    """Chunking is a memory strategy, not an approximation."""
    ed, cen, orders = dataset
    layout = params.lineph_layout(EPH)
    kw = dict(num_resample=7, exposure_time=29.4 / 1440, n_chains_hint=64,
              dtype=mx.float64)
    with mx.stream(mx.cpu):
        lp = likelihood.ProfiledTransitLogProb(
            layout, cen, ed, orders, budget_bytes=budget, **kw)
        single = likelihood.ProfiledTransitLogProb(
            layout, cen, ed, orders, budget_bytes=1 << 31, **kw)
        v = mx.array(np.repeat(V_TRUE[None, :], 3, axis=0), dtype=mx.float64)
        got = np.array(lp(v), dtype=np.float64)
        ref = np.array(single(v), dtype=np.float64)
    if budget == 1 << 31:
        assert len(lp.blocks) == 1
    else:
        assert len(lp.blocks) > 1
    np.testing.assert_allclose(got, ref, rtol=1e-11)


@pytest.mark.parametrize("budget", [1 << 31, 1 << 16])
def test_point_order_and_padding_placement_do_not_change_the_answer(
        dataset, budget, monkeypatch):
    """Phase-sorting the points and parking padded slots at quadrature are
    kernel-speed measures (model.phase_order), not approximations: value and
    gradient must match storage order with padding at the epoch centre."""
    ed, cen, orders = dataset
    assert (ed["mask"] == 0).any(), "fixture must exercise padding"
    layout = params.lineph_layout(EPH)
    kw = dict(num_resample=7, exposure_time=29.4 / 1440, n_chains_hint=64,
              dtype=mx.float64, budget_bytes=budget)
    v = mx.array(V_TRUE[None, :] + 1e-3 * np.random.default_rng(3)
                 .standard_normal((4, V_TRUE.size)), dtype=mx.float64)
    build = M.build_grid

    def run():
        with mx.stream(mx.cpu):
            lp = likelihood.ProfiledTransitLogProb(layout, cen, ed, orders,
                                                   **kw)
            val, g = mx.value_and_grad(lambda x: mx.sum(lp(x)))(v)
            return lp, np.array(lp(v)), np.array(g)

    lp, val, grad = run()
    assert all(b.grid.order is not None for b in lp.blocks)
    pad = ed["mask"] == 0
    tau = np.array(lp.grid.times) - np.array(lp.grid.d_arr)[:, None]
    np.testing.assert_allclose(tau[pad], 0.25 * P_REF, rtol=1e-12)

    monkeypatch.setattr(M, "build_grid", lambda *a, mask=None, **k: build(
        *a, **{**k, "sort_points": False}))
    lp0, val0, grad0 = run()
    assert all(b.grid.order is None for b in lp0.blocks)
    np.testing.assert_allclose(val, val0, rtol=1e-13)
    np.testing.assert_allclose(grad, grad0, rtol=1e-10, atol=1e-10)


def test_phase_order_sorts_by_time_from_mid_and_inverts():
    times = np.array([[0.3, -0.1, 0.0], [0.2, -0.4, 0.1]])
    d = np.array([0.05, -0.1])
    order, unorder = M.phase_order(times, d)
    key = (times - d[:, None]).ravel()
    o, u = np.array(order), np.array(unorder)
    assert np.all(np.diff(key[o]) >= 0)
    np.testing.assert_array_equal(o[u], np.arange(key.size))


def test_validate_precision_is_ok(lineph):
    """anvil's own float32 adjudication must pass before any production fit."""
    layout, target, transform, lp, hi = lineph
    rep = anvil.validate_precision(target, u_ball(transform, layout.dim, n=32))
    assert "WARNING" not in str(rep), str(rep)
    assert rep.max_abs_err < 1.0, str(rep)


def _grads(dataset, exposure_time, num_resample):
    """float32 and float64 analytic gradients, plus a float64 FD closure."""
    ed, cen, orders = dataset
    layout = params.lineph_layout(EPH)
    kw = dict(num_resample=num_resample, exposure_time=exposure_time,
              n_chains_hint=8)
    lo = likelihood.ProfiledTransitLogProb(layout, cen, ed, orders,
                                           dtype=mx.float32, **kw)
    with mx.stream(mx.cpu):
        hi = likelihood.ProfiledTransitLogProb(layout, cen, ed, orders,
                                               dtype=mx.float64, **kw)
        g64 = np.array(mx.grad(lambda vv: mx.sum(hi(vv)))(
            mx.array(V_TRUE[None, :], dtype=mx.float64)), dtype=np.float64)[0]
    g32 = np.array(mx.grad(lambda vv: mx.sum(lo(vv)))(
        mx.array(V_TRUE[None, :].astype(np.float32))), dtype=np.float64)[0]

    def fd(i):
        h = 1e-6 * max(1e-3, abs(V_TRUE[i]))
        vp, vm = V_TRUE.copy(), V_TRUE.copy()
        vp[i] += h
        vm[i] -= h
        with mx.stream(mx.cpu):
            f = lambda v: float(np.array(
                hi(mx.array(v[None, :], dtype=mx.float64)))[0])
            return (f(vp) - f(vm)) / (2 * h)

    return layout, g32, g64, fd


def test_fp64_gradient_matches_finite_differences_without_exposure(dataset):
    """The strict check, where finite differences are a clean reference.

    With the exposure integrated, they are not -- see the test below.
    """
    layout, _, g64, fd = _grads(dataset, exposure_time=0.0, num_resample=1)
    for i, name in enumerate(layout.names):
        d = fd(i)
        assert abs(g64[i] - d) / max(abs(d), 1.0) < 1e-5, (
            f"{name}: fp64 analytic {g64[i]} vs fd {d}")


def test_fp32_gradient_tracks_fp64_under_exposure_integration(dataset):
    """At production settings, compare the two precisions of the same model.

    Finite differences are deliberately *not* the reference here. The contact
    rule freezes its quadrature split points -- exact for the integral, since
    moving an interior split of a continuous integrand cancels -- so a central
    difference, which recomputes the splits at each step, measures the
    quadrature error's parameter dependence rather than the gradient. Worse,
    MetalPlanet's own docs note that a difference straddling a contact is
    wrong at any step size, because the light curve's tau-derivative genuinely
    jumps there. So FD gets a loose sanity bound and the precise claim is
    float32 against float64.
    """
    layout, g32, g64, fd = _grads(dataset, exposure_time=29.4 / 1440,
                                  num_resample=7)
    for i, name in enumerate(layout.names):
        scale = max(abs(g64[i]), 1.0)
        # float32 agrees to ~1e-4, except k, whose gradient is a cancellation
        # between three channels (the radius ratio, the impact parameter and
        # a/R*). That does not bias the posterior: leapfrog with any smooth
        # force field stays volume-preserving and reversible, and the
        # Metropolis step uses the log-density, which validate_precision
        # bounds at ~1e-3. It costs a little trajectory efficiency.
        tol = 2e-2 if name == "k" else 2e-3
        assert abs(g32[i] - g64[i]) / scale < tol, (
            f"{name}: fp32 {g32[i]} vs fp64 {g64[i]}")
        # FD still has to agree to the quadrature's own parameter sensitivity
        assert abs(g64[i] - fd(i)) / scale < 1e-3, name


def test_log_prob_compiles(lineph):
    """mx.compile must trace it: no host syncs, no data-dependent control flow."""
    layout, target, transform, lp, hi = lineph
    u = u_ball(transform, layout.dim, n=16)
    plain = np.array(target.log_prob(u), dtype=np.float64)
    compiled = mx.compile(lambda uu: target.log_prob(uu))
    got = np.array(compiled(u), dtype=np.float64)
    # compilation fuses and so reassociates float32 reductions; the agreement
    # to expect is the float32 noise floor, not bit-equality
    assert np.abs(got - plain).max() < 0.02
    # and value+grad, which is what the kernel actually calls
    step = mx.compile(lambda uu: target.log_prob_and_grad(uu))
    val, grad = step(u)
    assert np.all(np.isfinite(np.array(grad)))


# -- TTV mode ----------------------------------------------------------

def test_ttv_layout_and_log_density(dataset):
    ed, cen, orders = dataset
    n_ep = ed["n_epochs"]
    layout = params.ttv_layout(EPH, cen, tau_half=0.02)
    assert layout.dim == 5 + n_ep
    assert layout.names[:5] == params.TTV_BASE
    assert all(nm.startswith("dtau_") for nm in layout.names[5:])
    # each timing parameter reports as an absolute mid-transit time
    for i, n in enumerate(cen["n_arr"]):
        off = layout.report_offset[5 + i]
        np.testing.assert_allclose(off, EPOCH + n * P_REF, rtol=1e-12)

    target, transform, lp, hi = likelihood.build_target(
        layout, cen, ed, orders, num_resample=7,
        exposure_time=29.4 / 1440, n_chains_hint=64)
    v = np.concatenate([[K_T, B_T / (1 + K_T), T14_T, Q1_T, Q2_T],
                        np.zeros(n_ep)])
    u = transform.from_model_np(v)
    val = np.array(target.log_prob(mx.array(u[None, :].astype(np.float32))),
                   dtype=np.float64)
    assert np.isfinite(val[0])

    # shifting one epoch's time away from the truth must lower the density
    v2 = v.copy()
    v2[5] = 0.01
    u2 = transform.from_model_np(v2)
    val2 = np.array(target.log_prob(mx.array(u2[None, :].astype(np.float32))),
                    dtype=np.float64)
    assert val2[0] < val[0] - 1.0


def test_ttv_layout_rejects_bad_prior_width(dataset):
    _, cen, _ = dataset
    with pytest.raises(ValueError, match="positive timing prior"):
        params.ttv_layout(EPH, cen, tau_half=0.0)
    with pytest.raises(ValueError, match="unknown b_prior"):
        params.ttv_layout(EPH, cen, tau_half=0.01, b_prior="nope")


def test_mismatched_epoch_counts_are_rejected(dataset):
    ed, cen, orders = dataset
    layout = params.lineph_layout(EPH)
    with pytest.raises(ValueError, match="orders has shape"):
        likelihood.ProfiledTransitLogProb(layout, cen, ed, orders[:-1])


def test_epoch_block_size_respects_the_budget():
    # a generous budget takes everything in one block
    assert likelihood.epoch_block_size(512, 50, 100, 7, 1 << 34) == 50
    # a tiny budget still returns at least one epoch
    assert likelihood.epoch_block_size(512, 50, 100, 7, 1) == 1
    # and it is monotone in the budget
    sizes = [likelihood.epoch_block_size(512, 50, 100, 7, 1 << s)
             for s in range(20, 34)]
    assert sizes == sorted(sizes)
