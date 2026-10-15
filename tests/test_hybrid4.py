"""The hybrid4 limb-darkening law (``--law=hybrid4``).

hybrid4 is elome's (and SquishierPlanet's, LDhybrid v1.0's) rational law,
``I/I(1) = 1 - sum_j w_j S_j(mu)`` with four weights on its curvature
basis (LDhybrid's ``docs/upstream/hybrid4_curvature_simplex.md``; elome >=
0.17, turin 0.1.67). turin samples it as LDhybrid does: ``q1..q4`` uniform
on the unit cube, mapped onto the simplex ``w_j >= 0, sum w_j <= 1``,
uniform there, depth ``s = sum w = q1^(1/4)`` first.

The references here are independent of turin's route: elome's own frontend
(``TransitModel``, contact-rule exposures) for the light curve, the brief's
map and check values written out in the test, the four profile conditions
evaluated from the brief's basis for what the simplex means, the flat
Dirichlet's moments for uniformity, and float64 for the float32 path.
"""

from __future__ import annotations

import math
import os
import sys

import mlx.core as mx
import numpy as np
import pytest
from elome.api import TransitModel, TransitParams

from turin import gibbs, likelihood, params, prep, sampling, seeding
from turin import model as M

P_REF, EPOCH, DUR_H = 9.3456, 120.5, 4.2
K_T, B_T = 0.095, 0.30
T14_T = DUR_H / 24.0
Q_T = (0.55, 0.40, 0.65, 0.30)
YERR = 2.5e-4
EPH = {"period": P_REF, "epoch": EPOCH, "duration": DUR_H, "depth": 9025.0}
EXP = 29.4 / 1440
TIMES = np.linspace(-0.25, 0.25, 601)


def _curvature_w(q):
    """LDhybrid's hybrid4 q -> w, written out from the brief, independently
    of turin and elome: depth s = q1^(1/4) first, then stick-breaking."""
    q = np.asarray(q, dtype=np.float64)
    s = q[..., 0] ** 0.25
    a, b = q[..., 1] ** (1 / 3), q[..., 2] ** 0.5
    return np.stack([s * (1 - a), s * a * (1 - b),
                     s * a * b * (1 - q[..., 3]), s * a * b * q[..., 3]], -1)


#: The brief's basis, rows S1..S4 over T = (1 - mu^4, (1 - mu^2)^2, Pi_e0,
#: Pi_e1), and the poles it was built on.
EPS4 = (0.028601207249486824, 0.38582417584735224)
BASIS4 = np.array([
    [0.5, 0.5, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0006260842489149928, 0.001404013755411423, 1.020963273947444,
     -0.02299337195177007],
    [-0.08116019831892918, -0.2568535772254558, -0.0005906631447752816,
     1.33860443868916]])


def _profile_conditions(w, x):
    """I, g = dI/dx, g', g'' at x = mu^2 for curvature-basis weights ``w``
    (..., 4), from the brief's basis and the pole shapes
    Pi_e(x) = a ((x + e)^-2 - (1 + e)^-2), a = 1 / (e^-2 - (1 + e)^-2)."""
    a = [1.0 / (e ** -2 - (1 + e) ** -2) for e in EPS4]
    x = np.asarray(x)[:, None]
    T = [1 - x ** 2, (1 - x) ** 2] + [
        a[i] * ((x + e) ** -2 - (1 + e) ** -2) for i, e in enumerate(EPS4)]
    dT = [-2 * x, -2 * (1 - x)] + [
        -2 * a[i] * (x + e) ** -3 for i, e in enumerate(EPS4)]
    d2T = [-2 + 0 * x, 2 + 0 * x] + [
        6 * a[i] * (x + e) ** -4 for i, e in enumerate(EPS4)]
    d3T = [0 * x, 0 * x] + [
        -24 * a[i] * (x + e) ** -5 for i, e in enumerate(EPS4)]
    c = np.atleast_2d(w) @ BASIS4                       # weights on T
    out = []
    for terms, sign, base in ((T, -1, 1.0), (dT, -1, 0.0), (d2T, -1, 0.0),
                              (d3T, -1, 0.0)):
        M_ = np.stack([t[:, 0] for t in terms], -1)     # (n_x, 4)
        out.append(base + sign * c @ M_.T)              # (n_w, n_x)
    return out                                          # I, g, g', g''


# -- the parametrisation ----------------------------------------------------

def test_names_and_layouts():
    assert M.ld_names("quadratic") == ("q1", "q2")
    assert M.ld_names("hybrid4") == ("q1", "q2", "q3", "q4")
    with pytest.raises(ValueError, match="hybrid5"):
        M.ld_names("hybrid5")

    lin = params.lineph_layout(EPH, law="hybrid4")
    assert lin.names == ("dP", "dtau0", "k", "beta", "T14",
                         "q1", "q2", "q3", "q4")
    assert lin.law == "hybrid4" and lin.n_shape == 7
    assert np.all(lin.lo[5:] == 0.0) and np.all(lin.hi[5:] == 1.0)
    quad = params.lineph_layout(EPH)
    assert quad.names == params.LINEPH_BASE and quad.n_shape == 5

    cen = {"n_arr": np.arange(3.0), "centers_abs": np.zeros(3)}
    ttv = params.ttv_layout(EPH, cen, 0.05, law="hybrid4")
    assert ttv.names[:7] == ("k", "beta", "T14", "q1", "q2", "q3", "q4")
    assert ttv.names[7:] == ("dtau_0", "dtau_1", "dtau_2")
    assert ttv.n_shape == 7

    # collapsed hybrid4 samples the geometry alone (tests/test_ldhybrid.py)
    col = params.lineph_layout(EPH, ld="collapsed", law="hybrid4")
    assert col.names == ("dP", "dtau0", "k", "beta", "T14")
    assert col.law == "hybrid4" and col.ld_names == ()


def test_q_maps_onto_the_curvature_simplex_as_ldhybrid_does():
    q = np.random.default_rng(3).uniform(size=(5000, 4))
    want = _curvature_w(q)
    np.testing.assert_allclose(M.ld_weights_np(q), want, rtol=0, atol=1e-15)
    cols = [mx.array(q[:, j:j + 1], dtype=mx.float64) for j in range(4)]
    with mx.stream(mx.cpu):
        got = np.array(M.ld_weights(cols), dtype=np.float64)
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-15)
    assert np.all(want >= 0) and np.all(want.sum(axis=1) <= 1 + 1e-15)

    sp_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                          "..", "SquishierPlanet"))
    if os.path.isdir(sp_dir):
        sys.path.insert(0, sp_dir)
        try:
            import squishierplanet as sp
        except Exception:                      # pragma: no cover
            sp = None
        finally:
            sys.path.remove(sp_dir)
        if sp is not None and hasattr(sp, "hybrid4_from_q"):
            np.testing.assert_allclose(want, sp.hybrid4_from_q(q), atol=1e-15)

    # the brief's check values, elome's own maps, and the way back
    w5 = M.ld_weights_np(np.full(4, 0.5))
    np.testing.assert_allclose(
        w5, [0.1734764882, 0.1954827707, 0.2359685782, 0.2359685782],
        atol=1e-10)
    np.testing.assert_allclose(
        w5 @ BASIS4, [0.0677347237, 0.2219429445, 0.2407758742, 0.3104428728],
        atol=1e-10)
    from elome.hybrid import HYBRID4_BASIS, HYBRID4_EPS
    from elome.ld import hybrid4_from_q_np, hybrid4_to_q_np
    np.testing.assert_allclose(HYBRID4_EPS, EPS4, rtol=0, atol=1e-15)
    np.testing.assert_allclose(HYBRID4_BASIS, BASIS4, rtol=0, atol=1e-14)
    np.testing.assert_allclose(hybrid4_from_q_np(q), want, rtol=0, atol=1e-15)
    np.testing.assert_allclose(hybrid4_to_q_np(want), q, rtol=0, atol=1e-9)
    # not the stick-breaking map turin used before 0.1.67
    assert np.abs(want[:, 0] - (1 - q[:, 0] ** 0.25)).max() > 0.1
    # depth first: 1 - I(0) = sum w = q1^(1/4)
    np.testing.assert_allclose(want.sum(-1), q[:, 0] ** 0.25, atol=1e-15)


def test_the_curvature_simplex_is_exactly_the_physical_profiles():
    """Inside: every q gives I >= 0, g >= 0, g' <= 0, g'' >= 0 on [0, 1]
    (x = mu^2). The vertices are the brief's. Just outside each of the five
    faces, some condition fails, so the region is not merely sufficient."""
    x = np.linspace(0.0, 1.0, 2001)
    q = np.random.default_rng(5).uniform(size=(4000, 4))
    I, g, g1, g2 = _profile_conditions(M.ld_weights_np(q), x)
    tol = 1e-9
    assert I.min() >= -tol and g.min() >= -tol
    assert g1.max() <= tol and g2.min() >= -tol

    I, g, g1, g2 = _profile_conditions(np.eye(4), x)
    np.testing.assert_allclose(I[:, -1], 1.0, atol=1e-12)         # I(1) = 1
    np.testing.assert_allclose(I[2:, 0], 0.0, atol=1e-12)         # S3, S4
    np.testing.assert_allclose(g[2:, -1], 0.0, atol=1e-9)
    np.testing.assert_allclose(g1[2:, -1], 0.0, atol=1e-9)
    np.testing.assert_allclose(g2[2, -1], 0.0, atol=1e-9)         # S3
    np.testing.assert_allclose(g2[3, 0], 0.0, atol=1e-6)          # S4

    centre = np.full(4, 0.2)
    faces = [np.eye(4)[j] for j in range(4)] + [-np.ones(4) / 4]
    for n in faces:                     # w_j = 0 faces, then sum w = 1
        on = centre.copy()
        if n.sum() > 0:
            on[n > 0] = 0.0
            on = on * (1 / on.sum()) * 0.8 if on.sum() else on
            out = on - 1e-3 * n
        else:
            on = centre / centre.sum()
            out = on * (1 + 1e-3)
        I, g, g1, g2 = _profile_conditions(out, x)
        worst = min(I.min(), g.min(), -g1.max(), g2.min())
        assert worst < -1e-6, (n, worst)


def test_uniform_q_gives_uniform_weights_on_the_simplex():
    """A flat Dirichlet with 5 components: E[w_j] = 1/5, Var = 4/150,
    Cov(w_i, w_j) = -1/150. 200,000 draws, so 5 sigma is ~0.0016."""
    w = M.ld_weights_np(np.random.default_rng(11).uniform(size=(200_000, 4)))
    np.testing.assert_allclose(w.mean(axis=0), 0.2, atol=2e-3)
    C = np.cov(w.T)
    np.testing.assert_allclose(np.diag(C), 4 / 150, atol=8e-4)
    off = C[~np.eye(4, dtype=bool)]
    np.testing.assert_allclose(off, -1 / 150, atol=8e-4)


# -- the light curve --------------------------------------------------------

def _grid(dtype, times=TIMES):
    cen = dict(times_centered=np.asarray(times)[None, :], n_arr=np.zeros(1),
               d_arr=np.zeros(1), P_ref=P_REF, tau0_ref=0.0)
    return M.build_grid(cen, prep.supersample_offsets(EXP, 1), dtype=dtype,
                        exp_time=EXP)


def _turin_flux(dtype, q=Q_T, k=K_T, b=B_T):
    col = lambda v: mx.array([[float(v)]], dtype=dtype)
    f = M.transit_flux(_grid(dtype), mid=col(0.0), k=col(k), b=col(b),
                       T14=col(T14_T), q=tuple(col(x) for x in q),
                       law="hybrid4", period=col(P_REF))
    return np.array(f, dtype=np.float64)[0, 0]


def _elome_flux(q=Q_T, k=K_T, b=B_T):
    with mx.stream(mx.cpu):
        aRs = float(M.a_over_rstar(*[mx.array([[v]], dtype=mx.float64)
                                     for v in (T14_T, P_REF, k, b)])[0, 0])
    pars = TransitParams()
    pars.t0, pars.per, pars.rp, pars.a = 0.0, P_REF, k, aRs
    pars.inc = math.degrees(math.acos(b / aRs))
    pars.ecc, pars.w = 0.0, 90.0
    pars.limb_dark = "hybrid4"
    pars.u = [float(x) for x in _curvature_w(np.asarray(q))]
    return TransitModel(pars, TIMES, dtype=mx.float64, exp_time=EXP,
                        integration="contact", n_gl=M.N_GL).light_curve(pars)


@pytest.mark.parametrize("b", [0.30, 0.95])      # 0.95: grazing (1-k = 0.905)
def test_hybrid4_light_curve_matches_elomes_frontend(b):
    ref = _elome_flux(b=b)
    with mx.stream(mx.cpu):
        got64 = _turin_flux(mx.float64, b=b)
    got32 = _turin_flux(mx.float32, b=b)
    assert 1 - ref.min() > 1e-3, "sanity: a visible transit"
    assert np.abs(got64 - ref).max() < 1e-13
    assert np.abs(got32 - ref).max() < 2e-6


def test_the_weights_reach_the_light_curve():
    """Different q, different limb darkening: the law is not ignored."""
    with mx.stream(mx.cpu):
        a = _turin_flux(mx.float64, q=(0.2, 0.5, 0.5, 0.5))
        b = _turin_flux(mx.float64, q=(0.9, 0.5, 0.5, 0.5))
    assert np.abs(a - b).max() > 1e-4


def test_hybrid4_refusals():
    col = lambda v: mx.array([[float(v)]], dtype=mx.float64)
    kw = dict(mid=col(0.0), k=col(K_T), b=col(B_T), T14=col(T14_T),
              period=col(P_REF))
    with mx.stream(mx.cpu):
        with pytest.raises(ValueError, match="circular"):
            M.transit_flux_dev(_grid(mx.float64), q=(col(0.5),) * 4,
                               law="hybrid4", geometry="chord", **kw)
        with pytest.raises(ValueError, match="takes 4"):
            M.transit_flux_dev(_grid(mx.float64), q=(col(0.5),) * 2,
                               law="hybrid4", **kw)


# -- the likelihood ---------------------------------------------------------

def make_dataset(dtau_true=0.0, seed=7, yerr=YERR):
    rng = np.random.default_rng(seed)
    t = np.arange(100.0, 100.0 + 10.0 * P_REF, EXP)
    tw, fw, ew = prep.extract_near_transit_data(
        t, np.ones_like(t), np.full(t.size, yerr), P_REF, EPOCH, DUR_H)
    ed = prep.segment_epochs(tw, fw, ew, P_REF, EPOCH, DUR_H)
    cen = prep.centering_constants(ed, EPH)
    n_ep = ed["n_epochs"]
    dtau = np.broadcast_to(np.asarray(dtau_true, dtype=np.float64),
                           (n_ep,)).copy()
    grid = M.build_grid(cen, prep.supersample_offsets(EXP, 7),
                        dtype=mx.float64, exp_time=EXP)
    col = lambda v: mx.array([[float(v)]], dtype=mx.float64)
    with mx.stream(mx.cpu):
        f = np.array(M.transit_flux(
            grid, mid=M.mid_times_ttv(grid, mx.array(dtau[None, :],
                                                     dtype=mx.float64)),
            k=col(K_T), b=col(B_T), T14=col(T14_T),
            q=tuple(col(x) for x in Q_T), law="hybrid4",
            period=col(P_REF)), dtype=np.float64)[0]
    x = prep.basis_x(ed)
    base = 1.0 + 2.5e-4 * x - 1.8e-4 * (1.5 * x**2 - 0.5)
    ed = dict(ed)
    ed["flux_padded"] = np.where(
        ed["mask"] > 0, f * base + rng.normal(0.0, yerr, f.shape), 1.0)
    return ed, cen, dtau, n_ep, np.full(n_ep, 2)


V_TRUE = np.array([0.0, 0.0, K_T, B_T / (1 + K_T), T14_T, *Q_T])


@pytest.fixture(scope="module")
def dataset():
    return make_dataset()


def test_float32_log_density_and_gradient_track_float64(dataset):
    ed, cen, _, _, orders = dataset
    layout = params.lineph_layout(EPH, law="hybrid4")
    _, tr, lo, hi = likelihood.build_target(
        layout, cen, ed, orders, num_resample=7, exposure_time=EXP,
        n_chains_hint=64)
    V = np.tile(V_TRUE, (64, 1))
    V[:, 2:] *= 1 + 0.03 * np.random.default_rng(1).standard_normal((64, 7))
    V[:, 5:] = np.clip(V[:, 5:], 0.02, 0.98)
    f32 = np.array(lo(mx.array(V, dtype=mx.float32)), dtype=np.float64)
    with mx.stream(mx.cpu):
        f64 = np.array(hi(mx.array(V, dtype=mx.float64)), dtype=np.float64)
    err = f32 - f64
    # the quadratic law's baseline on a fit this size is ~3e-4 median
    assert np.std(err) < 2e-2 and np.all(np.isfinite(f32))

    g32 = np.array(mx.grad(lambda v: lo(v).sum())(
        mx.array(V[:4], dtype=mx.float32)), dtype=np.float64)
    with mx.stream(mx.cpu):
        g64 = np.array(mx.grad(lambda v: hi(v).sum())(
            mx.array(V[:4], dtype=mx.float64)), dtype=np.float64)
    scale = np.maximum(np.abs(g64), 1.0)
    rel = np.abs(g32 - g64) / scale
    # k carries the known three-channel cancellation (likelihood tests)
    k = layout.index("k")
    assert np.delete(rel, k, axis=1).max() < 5e-3, rel
    assert rel[:, k].max() < 3e-2


def test_truth_beats_a_shifted_law(dataset):
    """The data carry the injected hybrid4 law: the log-density at the
    truth exceeds the same geometry with the limb darkening moved."""
    ed, cen, _, _, orders = dataset
    layout = params.lineph_layout(EPH, law="hybrid4")
    _, _, _, hi = likelihood.build_target(
        layout, cen, ed, orders, num_resample=7, exposure_time=EXP,
        n_chains_hint=4)
    off = V_TRUE.copy()
    # a much weaker limb, I(0) 0.53 against 0.14 (198 nats). Not an arbitrary
    # q: (0.15, 0.85, 0.2, 0.9), this test's choice before the curvature
    # basis (0.1.67), lands on a law whose light curve this data cannot tell
    # from the truth's (1.7 nats the other way), though its weights differ
    off[5:] = (0.05, 0.5, 0.5, 0.5)
    with mx.stream(mx.cpu):
        L = np.array(hi(mx.array(np.stack([V_TRUE, off]), dtype=mx.float64)))
    assert L[0] - L[1] > 20.0, L


def test_ttv_epoch_terms_factorise_and_grid_gibbs_moves_only_times(dataset):
    """With four LD columns the times start at column 7, not 5: the
    per-epoch terms must still sum to the likelihood, and a grid-Gibbs sweep
    must leave every shape column, the weights included, untouched."""
    ed, cen, _, n_ep, orders = dataset
    layout = params.ttv_layout(EPH, cen, 0.03, law="hybrid4")
    _, tr, lo, _ = likelihood.build_target(
        layout, cen, ed, orders, num_resample=7, exposure_time=EXP,
        n_chains_hint=16, fp64=False)
    rng = np.random.default_rng(2)
    v = np.tile(np.concatenate([V_TRUE[2:], np.zeros(n_ep)]), (16, 1))
    v[:, 7:] = 0.01 * rng.uniform(-1, 1, (16, n_ep))
    vm = mx.array(v, dtype=mx.float32)
    terms = np.array(lo.epoch_log_lik(vm), dtype=np.float64)
    p = lo.unpack(vm)
    total = np.array(lo(vm), dtype=np.float64)
    prior = np.array(lo.log_prior(p), dtype=np.float64)
    np.testing.assert_allclose(terms.sum(axis=1) + prior, total, rtol=0,
                               atol=0.05)
    assert terms.shape == (16, n_ep)
    np.testing.assert_array_equal(np.array(p["q"][3])[:, 0],
                                  v[:, 6].astype(np.float32))

    move = gibbs.GridGibbs(lo, tr, layout, T14=T14_T, batch=16, seed=0)
    u = tr.from_model_np(v)
    u_new, stats = move.sweep(u)
    v_new = tr.model_np(np.asarray(u_new))
    np.testing.assert_allclose(v_new[:, :7], v[:, :7], rtol=0, atol=1e-6)
    assert np.any(np.abs(v_new[:, 7:] - v[:, 7:]) > 1e-4)


def test_seeding_starts_inside_the_simplex():
    layout = params.lineph_layout(EPH, law="hybrid4")
    v0 = seeding.initial_model_vector(layout, EPH)
    assert v0.shape == (9,) and np.all(v0[5:] == 0.5)
    w = M.ld_weights_np(v0[5:])
    assert np.all(w > 0.05) and w.sum() < 0.95


@pytest.mark.slow
def test_lineph_injection_recovery_with_hybrid4():
    """The whole stack recovers a hybrid4 injection, geometry within 3
    sigma; the limb darkening is weakly constrained by design, so its
    coordinates are checked for consistency, not tightness."""
    ed, cen, _, _, orders = make_dataset()
    layout = params.lineph_layout(EPH, law="hybrid4")
    target, transform, lp, _ = likelihood.build_target(
        layout, cen, ed, orders, num_resample=7, exposure_time=EXP,
        n_chains_hint=256)
    v0 = seeding.initial_model_vector(layout, EPH)
    rng = np.random.default_rng(0)
    u_starts = (transform.from_model_np(v0)[None, :]
                + 0.25 * rng.standard_normal((128, layout.dim)))
    map_res = seeding.find_map(target, u_starts.astype(np.float32), n_iter=400)
    cfg = sampling.SamplerConfig(n_chains=256, n_warmup=400, n_samples=300,
                                 max_samples=1200, seed=3).for_mode(layout.dim)
    u0 = map_res.ball(cfg.n_chains, seed=1)
    sampling.check_precision(target, u0, log=None)
    results, verdict, total = sampling.run_rounds(
        target, list(layout.names), u0, cfg, log=lambda m: None)
    assert not verdict.health.any_trapped, verdict.health.describe()

    post = sampling.physical_draws(transform, results)
    truth = V_TRUE + layout.report_offset
    for i, name in enumerate(layout.names):
        med, sd = np.median(post[:, i]), np.std(post[:, i])
        assert sd > 0, name
        assert abs(med - truth[i]) / sd < 3.0, (name, med, truth[i], sd)
    assert np.std(post[:, layout.index("k")]) < 0.2 * K_T
    assert np.std(post[:, layout.index("T14")]) < 0.1 * T14_T
