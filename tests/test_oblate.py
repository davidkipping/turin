"""Oblate planets (``--oblate``): a projected ellipse with flattening ``f``
and long-axis sky angle ``theta``, hybrid4 limb darkening, LinEph.

turin samples ``(fc, fs) = sqrt(f) (cos 2theta, sin 2theta)`` on the disc
``f <= F_MAX`` (SquishierPlanet's coordinates) and hands ``f, theta`` to
elome's oblate kernel. The references here are independent of turin's
route: elome's own frontend (``TransitModel``, contact-rule exposures,
theta in degrees) and SquishierPlanet's float64 light curve for the curve,
uniform-disc moments for the prior, the sampled oblate likelihood for the
collapsed conditional, importance sampling of the true likelihood for the
collapsed value, and float64 for float32.
"""

from __future__ import annotations

import math
import os
import sys

import mlx.core as mx
import numpy as np
import pytest
from elome.api import TransitModel, TransitParams
from scipy.stats import qmc

from turin import likelihood, params, seeding
from turin import ldhybrid as LH
from turin import model as M

sys.path.insert(0, os.path.dirname(__file__))
import test_hybrid4 as H4                                   # noqa: E402

F_T, TH_T = 0.1, 30.0                     # the brief's injection: f, theta (deg)


def _fcfs(f, theta_deg):
    t = 2.0 * math.radians(theta_deg)
    return math.sqrt(f) * math.cos(t), math.sqrt(f) * math.sin(t)


def _vec(k=H4.K_T, b=H4.B_T, f=F_T, th=TH_T, T14=H4.T14_T, q=H4.Q_T):
    return np.array([0.0, 0.0, k, b / (1 + k), T14, *_fcfs(f, th), *q])


# -- the parametrisation ------------------------------------------------------

def test_layouts_and_refusals():
    lay = params.lineph_layout(H4.EPH, law="hybrid4", oblate=True)
    assert lay.names == ("dP", "dtau0", "k", "beta", "T14", "fc", "fs",
                         "q1", "q2", "q3", "q4")
    assert lay.n_shape == 9 and lay.oblate
    np.testing.assert_allclose(lay.hi[5:7], math.sqrt(params.F_MAX))
    np.testing.assert_allclose(lay.lo[5:7], -math.sqrt(params.F_MAX))
    col = params.lineph_layout(H4.EPH, law="hybrid4", oblate=True,
                               ld="collapsed")
    assert col.names == ("dP", "dtau0", "k", "beta", "T14", "fc", "fs")
    assert col.n_shape == 5
    # a round layout is exactly what it was
    assert params.lineph_layout(H4.EPH, law="hybrid4").names == \
        ("dP", "dtau0", "k", "beta", "T14", "q1", "q2", "q3", "q4")
    with pytest.raises(ValueError, match="hybrid4"):
        params.lineph_layout(H4.EPH, oblate=True)
    ed, cen, _, _, _ = H4.make_dataset()
    with pytest.raises(ValueError, match="LinEph-only"):
        params.ttv_layout(H4.EPH, cen, 0.03, law="hybrid4", oblate=True)


def test_fcfs_map_and_its_uniform_prior():
    """``(fc, fs)`` uniform on the disc is uniform in ``f`` on [0, F_MAX]
    and in ``theta`` on [-90, 90] degrees, independently."""
    rng = np.random.default_rng(3)
    r = math.sqrt(params.F_MAX)
    X = rng.uniform(-r, r, size=(400_000, 2))
    X = X[(X ** 2).sum(1) <= params.F_MAX]
    f, th = params.oblate_from_fcfs_np(X[:, 0], X[:, 1])
    assert abs(f.mean() - 0.3) < 2e-3 and abs(f.var() - 0.03) < 1e-3
    assert abs(th.mean()) < 0.5 and abs(th.var() - 2700.0) < 20.0
    assert abs(np.corrcoef(f, th)[0, 1]) < 0.01
    fc, fs = _fcfs(F_T, TH_T)
    f1, th1 = params.oblate_from_fcfs_np(fc, fs)
    assert abs(f1 - F_T) < 1e-15 and abs(th1 - TH_T) < 1e-12
    with mx.stream(mx.cpu):
        f2, th2 = params.oblate_from_fcfs(mx.array(fc, dtype=mx.float64),
                                          mx.array(fs, dtype=mx.float64))
    assert abs(float(f2) - F_T) < 1e-15
    assert abs(math.degrees(float(th2)) - TH_T) < 1e-12


# -- the light curve ------------------------------------------------------------

def _turin_flux(dtype, f=F_T, th=TH_T, b=H4.B_T, exp=True, times=H4.TIMES):
    cen = dict(times_centered=np.asarray(times)[None, :], n_arr=np.zeros(1),
               d_arr=np.zeros(1), P_ref=H4.P_REF, tau0_ref=0.0)
    grid = M.build_grid(cen, H4.prep.supersample_offsets(H4.EXP, 1),
                        dtype=dtype, exp_time=H4.EXP if exp else 0.0)
    col = lambda v: mx.array([[float(v)]], dtype=dtype)
    kw = {} if f is None else dict(f=col(f), theta=col(math.radians(th)))
    out = M.transit_flux(grid, mid=col(0.0), k=col(H4.K_T), b=col(b),
                         T14=col(H4.T14_T), q=tuple(col(x) for x in H4.Q_T),
                         law="hybrid4", period=col(H4.P_REF), **kw)
    return np.array(out, dtype=np.float64)[0, 0]


def _a_rs(b):
    with mx.stream(mx.cpu):
        return float(M.a_over_rstar(*[mx.array([[v]], dtype=mx.float64)
                                      for v in (H4.T14_T, H4.P_REF, H4.K_T,
                                                b)])[0, 0])


@pytest.mark.parametrize("b, th", [(0.7, 30.0), (0.7, -60.0), (0.3, 30.0),
                                   (0.3, 0.0)])
def test_oblate_light_curve_matches_elomes_frontend(b, th):
    aRs = _a_rs(b)
    pars = TransitParams()
    pars.t0, pars.per, pars.rp, pars.a = 0.0, H4.P_REF, H4.K_T, aRs
    pars.inc = math.degrees(math.acos(b / aRs))
    pars.ecc, pars.w = 0.0, 90.0
    pars.limb_dark = "hybrid4"
    pars.u = [float(x) for x in H4._stick(np.asarray(H4.Q_T))]
    pars.f, pars.theta = F_T, th                  # the frontend takes degrees
    ref = TransitModel(pars, H4.TIMES, dtype=mx.float64, exp_time=H4.EXP,
                       integration="contact", n_gl=M.N_GL).light_curve(pars)
    with mx.stream(mx.cpu):
        got64 = _turin_flux(mx.float64, th=th, b=b)
    got32 = _turin_flux(mx.float32, th=th, b=b)
    assert 1 - ref.min() > 1e-3, "sanity: a visible transit"
    assert np.abs(got64 - ref).max() < 1e-13
    assert np.abs(got32 - ref).max() < 2e-6


def _squishierplanet():
    sp_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                          "..", "SquishierPlanet"))
    if not os.path.isdir(sp_dir):
        return None
    sys.path.insert(0, sp_dir)
    try:
        from squishierplanet import laws, model
        return laws, model
    except Exception:                               # pragma: no cover
        return None
    finally:
        sys.path.remove(sp_dir)


@pytest.mark.parametrize("b, th", [(0.7, 30.0), (0.3, -60.0)])
def test_oblate_light_curve_matches_squishierplanet(b, th):
    """SquishierPlanet's float64 oblate light curve (no exposure
    integration), which elome itself was validated against to 6e-16. A sign
    slip in theta or b fails here at b = 0.7."""
    sp = _squishierplanet()
    if sp is None:
        pytest.skip("SquishierPlanet is not cloned beside turin")
    laws, model = sp
    aRs = _a_rs(b)
    w = H4._stick(np.asarray(H4.Q_T))
    ref = model.light_curve(np.asarray(H4.TIMES), laws.hybrid("hybrid4", w),
                            r_eff=H4.K_T, f=F_T, theta=math.radians(th),
                            period=H4.P_REF, a=aRs, i=math.acos(b / aRs),
                            t0=0.0)
    with mx.stream(mx.cpu):
        got = _turin_flux(mx.float64, th=th, b=b, exp=False)
    assert 1 - ref.min() > 1e-3
    assert np.abs(got - ref).max() < 1e-12


def test_theta_has_period_a_half_turn_and_f_matters():
    with mx.stream(mx.cpu):
        a = _turin_flux(mx.float64, th=30.0)
        b = _turin_flux(mx.float64, th=210.0)
        c = _turin_flux(mx.float64, f=0.3)
        d = _turin_flux(mx.float64, th=-30.0)
    assert np.abs(a - b).max() < 1e-14
    assert np.abs(a - c).max() > 1e-5          # flattening changes the curve
    assert np.abs(a - d).max() > 1e-6          # and theta's sign does, b > 0


def test_a_round_fit_never_hands_elome_an_oblate_keyword(monkeypatch):
    """elome's f = 0 is not bitwise its spherical forms (9.7e-8 of the depth
    measured), so a round fit must not pass f at all: that is what keeps
    every pre-oblate path bitwise."""
    seen = []
    real = M.elome.flux_dev_from_tau

    def spy(*a, **kw):
        seen.append(("f" in kw, "theta" in kw))
        return real(*a, **kw)

    monkeypatch.setattr(M.elome, "flux_dev_from_tau", spy)
    ed, cen, _, _, orders = H4.make_dataset()
    for ld in ("sampled", "collapsed"):
        lay = params.lineph_layout(H4.EPH, law="hybrid4", ld=ld)
        _, _, lo, _ = likelihood.build_target(
            lay, cen, ed, orders, num_resample=7, exposure_time=H4.EXP,
            n_chains_hint=2, fp64=False)
        v = H4.V_TRUE if ld == "sampled" else H4.V_TRUE[:5]
        mx.eval(lo(mx.array(np.tile(v, (2, 1)), dtype=mx.float32)))
    assert seen and not any(a or b for a, b in seen)


# -- the likelihood ---------------------------------------------------------------

@pytest.fixture(scope="module")
def data():
    return H4.make_dataset()


def _targets(data, ld="sampled", **kw):
    ed, cen, _, _, orders = data
    lay = params.lineph_layout(H4.EPH, law="hybrid4", oblate=True, ld=ld)
    target, _, lo, hi = likelihood.build_target(
        lay, cen, ed, orders, num_resample=7, exposure_time=H4.EXP,
        n_chains_hint=16, **kw)
    return lay, target, lo, hi


def test_f_to_zero_is_the_round_planet_to_float32_tolerance(data):
    """Not bitwise, by design (elome's f = 0 switch), but the density of an
    oblate fit at f = 1e-10 is the round hybrid4 density."""
    ed, cen, _, _, orders = data
    _, _, lo, hi = _targets(data)
    round_ = params.lineph_layout(H4.EPH, law="hybrid4")
    _, _, rlo, rhi = likelihood.build_target(
        round_, cen, ed, orders, num_resample=7, exposure_time=H4.EXP,
        n_chains_hint=4)
    v = _vec(f=1e-10)
    vr = np.delete(v, [5, 6])
    with mx.stream(mx.cpu):
        a = float(hi(mx.array(v[None], dtype=mx.float64))[0])
        b = float(rhi(mx.array(vr[None], dtype=mx.float64))[0])
    assert abs(a - b) < 1e-8, a - b
    a32 = float(lo(mx.array(v[None], dtype=mx.float32))[0])
    b32 = float(rlo(mx.array(vr[None], dtype=mx.float32))[0])
    assert abs(a32 - b32) < 1e-2


@pytest.mark.parametrize("ld", ["sampled", "collapsed"])
def test_outside_the_disc_or_the_domain_is_minus_inf_with_finite_gradients(
        data, ld):
    """elome returns NaN for k > (1 - f)^1.5; f > F_MAX is outside the
    prior. Both give -inf, never NaN, and a finite gradient (the kernel is
    kept inside its domain rather than masked afterwards)."""
    lay, _, lo, hi = _targets(data, ld=ld)
    V = np.stack([_vec(), _vec(k=0.9, f=0.3),
                  np.r_[_vec()[:5], 0.7, 0.7, H4.Q_T]])
    V = V[:, :lay.dim]
    v32 = mx.array(V, dtype=mx.float32)
    val = np.array(lo(v32))
    assert np.isfinite(val[0]) and np.all(val[1:] == -np.inf)
    with mx.stream(mx.cpu):
        v64 = np.array(hi(mx.array(V, dtype=mx.float64)))
    assert np.isfinite(v64[0]) and np.all(v64[1:] == -np.inf)
    g = np.array(mx.grad(lambda v: mx.sum(mx.where(
        mx.isfinite(lo(v)), lo(v), 0.0)))(v32))
    assert np.all(np.isfinite(g))
    assert np.all(g[0, 5:7] != 0)              # the shape feels the gradient


def test_float32_tracks_float64(data):
    lay, _, lo, hi = _targets(data)
    rng = np.random.default_rng(1)
    V = np.tile(_vec(), (16, 1))
    V[:, 2:5] *= 1 + 0.01 * rng.standard_normal((16, 3))
    V[:, 5:7] += 0.03 * rng.standard_normal((16, 2))
    f32 = np.array(lo(mx.array(V, dtype=mx.float32)), dtype=np.float64)
    with mx.stream(mx.cpu):
        f64 = np.array(hi(mx.array(V, dtype=mx.float64)))
    assert np.all(np.isfinite(f32)) and np.std(f32 - f64) < 2e-2
    g32 = np.array(mx.grad(lambda v: lo(v).sum())(
        mx.array(V[:4], dtype=mx.float32)), dtype=np.float64)
    ed, cen, _, _, orders = data
    with mx.stream(mx.cpu):
        h = likelihood.ProfiledTransitLogProb(
            lay, cen, ed, orders, num_resample=7, exposure_time=H4.EXP,
            n_chains_hint=4, dtype=mx.float64)
        g64 = np.array(mx.grad(lambda v: h(v).sum())(
            mx.array(V[:4], dtype=mx.float64)))
    rel = np.abs(g32 - g64) / np.maximum(np.abs(g64), 1.0)
    k = lay.index("k")
    assert np.delete(rel, k, axis=1).max() < 1e-2, rel
    assert rel[:, k].max() < 3e-2


def test_collapsed_conditional_is_the_sampled_oblate_likelihood(data):
    ed, cen, _, _, orders = data
    sam, _, _, _ = _targets(data)
    col = params.lineph_layout(H4.EPH, law="hybrid4", oblate=True,
                               ld="collapsed")
    with mx.stream(mx.cpu):
        lp = likelihood.ProfiledTransitLogProb(
            sam, cen, ed, orders, num_resample=7, exposure_time=H4.EXP,
            n_chains_hint=8, dtype=mx.float64)
        hc = LH.MarginalHybridLogProb(
            col, cen, ed, orders, num_resample=7, exposure_time=H4.EXP,
            n_chains_hint=8, dtype=mx.float64)
        q = np.random.default_rng(5).uniform(0.05, 0.95, size=(6, 4))
        V = np.column_stack([np.tile(_vec()[:7], (6, 1)), q])
        vm = mx.array(V, dtype=mx.float64)
        want = np.array(lp(vm)) - np.array(lp.log_prior(lp.unpack(vm)))
        got = np.array(hc.loglik_at(mx.array(V[:, :7], dtype=mx.float64),
                                    LH.q_to_x(q)))
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-8)


@pytest.fixture(scope="module")
def weak():
    """A noisier copy: the limb darkening barely constrained, so uniform-q
    importance sampling of the true likelihood is an efficient reference."""
    ed, cen, _, _, orders = H4.make_dataset(yerr=2e-3)
    col = params.lineph_layout(H4.EPH, law="hybrid4", oblate=True,
                               ld="collapsed")
    with mx.stream(mx.cpu):
        hi = LH.MarginalHybridLogProb(col, cen, ed, orders, num_resample=7,
                                      exposure_time=H4.EXP, n_chains_hint=8,
                                      dtype=mx.float64)
    lo = LH.MarginalHybridLogProb(col, cen, ed, orders, num_resample=7,
                                  exposure_time=H4.EXP, n_chains_hint=1024,
                                  dtype=mx.float32)
    return hi, lo


def test_collapsed_oblate_value_and_draws_are_the_exact_marginal_and_conditional(
        weak):
    hi, lo = weak
    v = _vec()[:7]
    Q = np.clip(qmc.Sobol(4, scramble=True, seed=3).random(2 ** 15), 1e-12,
                1 - 1e-12)
    with mx.stream(mx.cpu):
        L = np.array(hi.loglik_grid(v, LH.q_to_x(Q)))
        vm = mx.array(v[None], dtype=mx.float64)
        got = float(hi(vm)[0]) - float(hi.log_prior(hi.unpack(vm))[0])
    truth = float(np.log(np.mean(np.exp(L - L.max()))) + L.max())
    assert abs(got - truth) < 0.03                 # GHK's rule error
    w = np.exp(L - L.max())
    w /= w.sum()
    assert 1.0 / np.sum(w ** 2) > 1000
    mu = w @ Q
    sd = np.sqrt(w @ (Q - mu) ** 2)
    q, acc, _ = LH.draw_limb_darkening(lo, np.tile(v, (2048, 1)), seed=4)
    assert np.all((q > 0) & (q < 1)) and acc > 0.5
    assert np.all(np.abs(q.mean(0) - mu) / sd < 0.1)
    np.testing.assert_allclose(q.std(0) / sd, 1.0, atol=0.1)


# -- the start --------------------------------------------------------------------

def test_floor_oblateness_keeps_chains_off_the_dead_zone(data):
    lay, _, _, _ = _targets(data)
    T = lay.transform()
    u = T.from_model_np(_vec(f=1e-9))
    v = T.model_np(seeding.floor_oblateness(u, T, lay)[None, :])[0]
    f, th = params.oblate_from_fcfs_np(v[5], v[6])
    assert abs(f - seeding.F_FLOOR) < 1e-9 and abs(th - TH_T) < 1e-6
    u0 = T.from_model_np(np.r_[_vec()[:5], 0.0, 0.0, H4.Q_T])
    v0 = T.model_np(seeding.floor_oblateness(u0, T, lay)[None, :])[0]
    assert abs(v0[5] ** 2 + v0[6] ** 2 - seeding.F_FLOOR) < 1e-9
    u1 = T.from_model_np(_vec())                      # already oblate enough
    assert np.array_equal(seeding.floor_oblateness(u1, T, lay), u1)


def test_laplace_starts_are_spread_and_finite(data):
    lay, tr, _, _ = _targets(data, ld="collapsed")
    T = lay.transform()
    u_map = T.from_model_np(_vec()[:7])
    u0, info = seeding.laplace_starts(tr, u_map, 64, seed=1)
    u0 = np.array(u0, dtype=np.float64)
    assert not info["fallback"]
    lp = np.array(tr.log_prob(mx.array(u0, dtype=mx.float32)))
    assert np.all(np.isfinite(lp))
    spread = u0.std(0)
    assert np.all(spread > 1e-4) and np.all(spread <= 0.5 * 0.5 * 1.3)


# -- end to end ---------------------------------------------------------------------

def _injection(seed=21):
    """Three epochs at 1-minute cadence, no exposure smearing, 30 ppm: an
    oblate signal measurable in a test-sized dataset (sd(f) ~ 0.004
    measured). At 10 ppm the posterior is so sharp (sd(k) = 3e-4) that the
    MAP search stops ~9,000 log-units short of the peak and warmup is spent
    descending; at 30 ppm it stops ~130 short, which warmup absorbs."""
    from turin import prep
    k, b, f, th = 0.15, 0.7, F_T, TH_T
    P, epoch, dur_h = H4.P_REF, H4.EPOCH, H4.DUR_H
    eph = dict(H4.EPH, depth=k * k * 1e6)
    T14 = dur_h / 24.0
    cad = 1.0 / 1440.0
    t = np.concatenate([np.arange(epoch + n * P - 1.2 * T14,
                                  epoch + n * P + 1.2 * T14, cad)
                        for n in range(3)])
    yerr = 3e-5
    tw, fw, ew = prep.extract_near_transit_data(
        t, np.ones_like(t), np.full(t.size, yerr), P, epoch, dur_h)
    ed = prep.segment_epochs(tw, fw, ew, P, epoch, dur_h)
    cen = prep.centering_constants(ed, eph)
    grid = M.build_grid(cen, prep.supersample_offsets(cad, 1),
                        dtype=mx.float64, exp_time=0.0)
    col = lambda v: mx.array([[float(v)]], dtype=mx.float64)
    with mx.stream(mx.cpu):
        flux = np.array(M.transit_flux(
            grid, mid=mx.zeros((1, ed["n_epochs"]), dtype=mx.float64),
            k=col(k), b=col(b), T14=col(T14), period=col(P),
            q=tuple(col(x) for x in H4.Q_T), law="hybrid4",
            f=col(f), theta=col(math.radians(th))), dtype=np.float64)[0]
    x = prep.basis_x(ed)
    base = 1.0 + 2e-4 * x
    ed = dict(ed)
    rng = np.random.default_rng(seed)
    ed["flux_padded"] = np.where(ed["mask"] > 0,
                                 flux * base + rng.normal(0, yerr, flux.shape),
                                 1.0)
    truth = np.array([0.0, 0.0, k, b / (1 + k), T14, *_fcfs(f, th)])
    return ed, cen, eph, np.full(ed["n_epochs"], 1), truth


@pytest.mark.slow
def test_lineph_injection_recovery_oblate_collapsed():
    """f = 0.1, theta = 30 deg, b = 0.7 (the brief's injection), collapsed
    hybrid4 (the --oblate default): f and theta within 3 sd of the truth
    and measured, every geometry parameter within 3 sd, no trapped or
    frozen chains."""
    from turin import sampling
    ed, cen, eph, orders, truth = _injection()
    lay = params.lineph_layout(eph, law="hybrid4", oblate=True,
                               ld="collapsed")
    tr, T, lo, _ = likelihood.build_target(
        lay, cen, ed, orders, num_resample=1, exposure_time=0.0,
        n_chains_hint=128)
    rng = np.random.default_rng(0)
    v0 = seeding.initial_model_vector(lay, eph)
    starts = T.from_model_np(v0)[None, :] + 0.25 * rng.standard_normal(
        (64, lay.dim))
    map_res = seeding.find_map(tr, starts.astype(np.float32))
    u_c = seeding.floor_oblateness(map_res.u_best, T, lay)
    frozen = LH.freeze_layout(lo, T.model_np(u_c[None, :])[0])
    tr, T, lo, _ = likelihood.build_target(
        lay, cen, ed, orders, num_resample=1, exposure_time=0.0,
        n_chains_hint=128, ghk_layout=frozen, ghk_points=lo.ghk_points)
    u0, _ = seeding.laplace_starts(tr, u_c, 128, seed=1)
    cfg = sampling.SamplerConfig(n_chains=128, n_warmup=300, n_samples=300,
                                 max_samples=1200).for_mode(lay.dim)
    res, verdict, _ = sampling.run_rounds(tr, list(lay.names), u0, cfg,
                                          log=lambda m: None)
    assert not verdict.frozen and not verdict.health.any_trapped
    assert verdict.converged, verdict.worst_rhat
    d = T.model_np(np.asarray(res.get_chain(flat=True), dtype=np.float64))
    for i in range(2, 5):
        assert abs(np.median(d[:, i]) - truth[i]) < 3 * d[:, i].std(), \
            lay.names[i]
    f, th = params.oblate_from_fcfs_np(d[:, 5], d[:, 6])
    assert abs(np.median(f) - F_T) < 3 * f.std() and f.std() < 0.02
    assert abs(np.median(th) - TH_T) < 3 * th.std() and th.std() < 10.0


# -- the stellar-density prior and the degeneracy warning (0.1.66) ---------------

@pytest.mark.parametrize("oblate, ld", [(True, "collapsed"), (True, "sampled"),
                                        (False, "sampled"), (False, "collapsed")])
def test_rhostar_prior_is_a_gaussian_in_log10_rho(data, oblate, ld):
    """The density with ``--rhostar=m,s`` minus the density without is
    ``-0.5 ((log10 rho* - m)/s)^2``, with log10 rho* the ``logrho``
    product's quantity (the host float64 reporting function, not the MLX
    one under test), for every layout it can sit on."""
    ed, cen, _, _, orders = data
    prior = (3.2, 0.05)
    lays = [params.lineph_layout(H4.EPH, law="hybrid4", oblate=oblate,
                                 ld=ld, rhostar=r) for r in ((), prior)]
    vals = []
    rng = np.random.default_rng(4)
    V = np.tile(_vec(), (5, 1))
    V[:, 2:5] *= 1 + 0.02 * rng.standard_normal((5, 3))
    if not oblate:
        V = np.delete(V, [5, 6], axis=1)
    V = V[:, :lays[0].dim]
    for lay in lays:
        _, _, _, hi = likelihood.build_target(
            lay, cen, ed, orders, num_resample=7, exposure_time=H4.EXP,
            n_chains_hint=8)
        with mx.stream(mx.cpu):
            vals.append(np.array(hi(mx.array(V, dtype=mx.float64))))
    k, beta, T14 = V[:, 2], V[:, 3], V[:, 4]
    b = M.impact_parameter(beta, k, "transiting")
    rho = M.log10_rho_from_T14(T14, H4.P_REF + V[:, 0], k, b)
    want = -0.5 * ((rho - prior[0]) / prior[1]) ** 2
    np.testing.assert_allclose(vals[1] - vals[0], want, rtol=1e-9, atol=1e-9)
    assert lays[1].rhostar == prior and lays[0].rhostar == ()


def test_rhostar_parses_and_is_refused_when_malformed():
    from turin import cli
    a = cli.parse_args(["--KOI-1.01", "--rhostar=3.149,0.04"])
    assert a.rhostar == (3.149, 0.04) and not a.oblate
    assert cli.parse_args(["--KOI-1.01"]).rhostar is None
    for bad in ("3.15", "3.15,0", "3.15,-1", "a,b", "3.15,0.1,2"):
        with pytest.raises(SystemExit, match="MEAN,SIGMA"):
            cli.parse_args(["--KOI-1.01", f"--rhostar={bad}"])


def test_rhostar_is_a_guard_with_none_as_a_value():
    from turin import outputs
    from test_cli_outputs import _state
    assert outputs.ResumeState.rhostar == () and \
        "rhostar" in outputs.ResumeState.GUARDS
    st = _state(rhostar=(3.2, 0.05))
    st.check(rhostar=(3.2, 0.05))                       # must not raise
    with pytest.raises(SystemExit, match="--rhostar=3.2,0.05"):
        st.check(rhostar=())
    with pytest.raises(SystemExit, match="no --rhostar"):
        _state().check(rhostar=(3.2, 0.05))
    with pytest.raises(SystemExit, match="no --oblate"):
        _state().check(oblate=True)


def test_aligned_central_share_counts_the_false_mode_family():
    from turin import pipeline
    f = np.array([0.19, 0.19, 0.19, 0.01, 0.19])
    th = np.array([2.0, -14.0, 40.0, 0.0, 3.0])
    b = np.array([0.02, 0.05, 0.02, 0.0, 0.4])
    # in: rows 0 and 1; out: theta too large, f too small, b too large
    assert pipeline.aligned_central_share(f, th, b) == pytest.approx(0.4)
