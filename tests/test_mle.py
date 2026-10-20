"""The likelihood alone and the profiled limb darkening behind ``--mle``.

References that share none of the profile's machinery: a dense lattice (or
random points) over the limb-darkening simplex, polished by float64 SLSQP on
the true ``L(theta, x)``; central finite differences of the float64 value
(at ``exp_time = 0``, since FD is invalid under the contact rule); and the
joint maximum over (theta, q) of the sampled-LD likelihood, found by
``anvil.optimize`` without any profiling.
"""

from __future__ import annotations

import os
import sys

import mlx.core as mx
import numpy as np
import pytest
from scipy.optimize import minimize

from turin import ldhybrid, ldmarg, likelihood, params, seeding

sys.path.insert(0, os.path.dirname(__file__))
import test_hybrid4 as H4                                   # noqa: E402
import test_ldmarg as LM                                    # noqa: E402
import test_oblate as OB                                    # noqa: E402


def _profile_targets(ed, cen, orders, eph, *, law="quadratic", oblate=False,
                     exposure_time=LM.EXP, **kw):
    lay = params.lineph_layout(eph, law=law, oblate=oblate, ld="collapsed")
    tr, T, lo, hi = likelihood.build_target(
        lay, cen, ed, orders, num_resample=1, exposure_time=exposure_time,
        n_chains_hint=16, likelihood_only=True, **kw)
    return lay, tr, T, lo, hi


def _hi(hi, V):
    with mx.stream(mx.cpu):
        return np.array(hi(mx.array(np.atleast_2d(V), dtype=mx.float64)),
                        dtype=np.float64)


def _brute_max(hi, theta, d, rng):
    """max_x L(theta, x) over the d-simplex, without the profile: the best of
    many points (a lattice for d = 2, Dirichlet draws otherwise, plus every
    vertex), polished by float64 SLSQP from the best five. The generator
    curves are computed once (an oblate float64 launch is slow) and ``L(x)``
    is the modules' own baseline-profiled sum at fixed x."""
    with mx.stream(mx.cpu):
        v = mx.array(np.asarray(theta, dtype=np.float64)[None], dtype=mx.float64)
        p = hi.unpack(v)
        if d == 2:
            blocks = hi._vertex_blocks(p)
            Lx = lambda X: ldmarg._sum_loglik(blocks, X[:, 0], X[:, 1])
        else:
            blocks = hi._gen_blocks(p)
            Lx = lambda X: ldhybrid._sum_loglik(blocks, X)

    def L(X):
        with mx.stream(mx.cpu):
            return np.array(Lx(mx.array(np.atleast_2d(X), dtype=mx.float64)),
                            dtype=np.float64)

    if d == 2:
        X, _, _ = LM._lattice(160)
    else:
        X = rng.dirichlet(np.ones(d + 1), 20000)[:, :d]
        X = np.concatenate([X, np.eye(d), np.zeros((1, d))])
    vals = L(X)
    best = float(vals.max())
    cons = [{"type": "ineq", "fun": lambda x: 1.0 - np.sum(x)}]
    for i in np.argsort(-vals)[:5]:
        r = minimize(lambda x: -float(L(x)[0]), X[i], method="SLSQP",
                     bounds=[(0.0, 1.0)] * d, constraints=cons,
                     options={"ftol": 1e-14, "maxiter": 400})
        if np.all(r.x >= -1e-12) and np.sum(r.x) <= 1 + 1e-12:
            best = max(best, -r.fun)
    return best


# -- the profile is the maximum over the simplex ------------------------------

@pytest.mark.parametrize("case", list(LM.CASES))
def test_quadratic_profile_is_the_maximum_over_the_triangle(case):
    """Every LM case, x* inside, on an edge ("weak"), at a corner, grazing."""
    ed, cen, orders = LM._dataset(**LM.CASES[case])
    _, _, _, _, hi = _profile_targets(ed, cen, orders, LM.EPH)
    theta = LM.CASE_THETA[case]
    rng = np.random.default_rng(0)
    for th in (theta, theta * (1 + 0.01 * rng.standard_normal(theta.size))):
        prof = float(_hi(hi, th)[0])
        brute = _brute_max(hi, th, 2, rng)
        assert prof >= brute - 1e-7, (case, prof - brute)
        assert prof <= brute + 1e-7, (case, prof - brute)


@pytest.mark.parametrize("oblate", [False, True])
def test_hybrid4_profile_is_the_maximum_over_the_simplex(oblate):
    """hybrid4 round and oblate, at the truth and at a point well off it
    (where x* moves onto a face)."""
    ed, cen, _, _, orders = H4.make_dataset()
    _, _, _, _, hi = _profile_targets(ed, cen, orders, H4.EPH, law="hybrid4",
                                      oblate=oblate, exposure_time=H4.EXP)
    v = OB._vec()[:7] if oblate else H4.V_TRUE[:5]
    rng = np.random.default_rng(1)
    for th in (v, v * (1 + np.r_[0, 0, 0.03, -0.1, 0.02, 0.2, 0.2][:v.size])):
        prof = float(_hi(hi, th)[0])
        brute = _brute_max(hi, th, 4, rng)
        assert abs(prof - brute) < 1e-6, (oblate, prof - brute)
        with mx.stream(mx.cpu):
            x = np.array(hi.ld_terms(mx.array(th[None], dtype=mx.float64))[0])
        assert np.all(x >= 0) and x.sum() <= 1 + 1e-12


# -- the gradient is the profile's (envelope theorem) --------------------------

@pytest.mark.parametrize("law, case", [("quadratic", "moderate"),
                                       ("quadratic", "weak"),
                                       ("hybrid4", None)])
def test_envelope_gradient_matches_finite_differences(law, case):
    if law == "quadratic":
        ed, cen, orders = LM._dataset(**LM.CASES[case])
        eph, th = LM.EPH, LM.CASE_THETA[case]
    else:
        ed, cen, _, _, orders = H4.make_dataset()
        eph, th = H4.EPH, H4.V_TRUE[:5] * np.r_[1, 1, 1.02, 0.9, 1.01]
    _, _, _, _, hi = _profile_targets(ed, cen, orders, eph, law=law,
                                      exposure_time=0.0)
    with mx.stream(mx.cpu):
        g = np.array(mx.grad(lambda v: mx.sum(hi(v)))(
            mx.array(th[None], dtype=mx.float64)))[0]
    h = np.array([1e-7, 1e-7, 1e-7, 1e-6, 1e-7])
    fd = np.array([(_hi(hi, th + h[i] * e)[0] - _hi(hi, th - h[i] * e)[0])
                   / (2 * h[i]) for i, e in enumerate(np.eye(th.size))])
    np.testing.assert_allclose(g, fd, rtol=2e-5, atol=1e-3 * np.abs(fd).max())


# -- the likelihood alone ---------------------------------------------------------

def test_likelihood_only_drops_the_priors_and_keeps_the_oblate_domain():
    """The sampled-LD likelihood is the sampling density minus the b-k and
    rho* priors exactly; the absolute normalisation reaches anvil; and an
    oblate point outside elome's domain is still -inf, gradient finite."""
    ed, cen, _, _, orders = H4.make_dataset()
    kw = dict(num_resample=1, exposure_time=H4.EXP, n_chains_hint=8)
    lay = params.lineph_layout(H4.EPH, law="hybrid4", rhostar=(3.2, 0.05))
    _, _, post, _ = likelihood.build_target(lay, cen, ed, orders, **kw)
    tr, _, lik, _ = likelihood.build_target(lay, cen, ed, orders,
                                            likelihood_only=True, **kw)
    V = np.tile(H4.V_TRUE, (4, 1)) * (1 + 0.01 * np.random.default_rng(2)
                                      .standard_normal((4, H4.V_TRUE.size)))
    v = mx.array(V, dtype=mx.float32)
    p = post.unpack(v)
    want = (np.array(post(v)) - np.array(params.bk_log_prior(
        p["beta"], p["k"], lay.b_prior)) - np.array(params.rhostar_log_prior(
            p["T14"], p["period"], p["k"], p["b"], lay.rhostar)))
    np.testing.assert_allclose(np.array(lik(v)), want, rtol=0, atol=2e-3)
    assert tr.model_log_norm == lik.log_const != 0.0

    olay = params.lineph_layout(H4.EPH, law="hybrid4", oblate=True,
                                ld="collapsed")
    for dtype in (mx.float32, mx.float64):
        with mx.stream(mx.cpu if dtype == mx.float64 else mx.gpu):
            lp = ldhybrid.ProfileHybridLogProb(
                olay, cen, ed, orders, num_resample=1, exposure_time=H4.EXP,
                n_chains_hint=4, dtype=dtype, prior=False)
            bad = OB._vec(k=0.9, f=0.5)[:7]               # k > (1 - f)^1.5
            v = mx.array(np.stack([OB._vec()[:7], bad]), dtype=dtype)
            val = np.array(lp(v))
            g = np.array(mx.grad(lambda x: mx.sum(mx.where(
                mx.isfinite(lp(x)), lp(x), 0.0)))(v))
        assert np.isfinite(val[0]) and val[1] == -np.inf
        assert np.all(np.isfinite(g))


def test_sampling_targets_are_unchanged_by_the_switch():
    """``prior=True`` (every sampling target) is the default: the collapsed
    and sampled sampling densities are what they were, and the profile
    classes are reached only through ``likelihood_only``."""
    ed, cen, orders = LM._dataset()
    kw = dict(num_resample=1, exposure_time=LM.EXP, n_chains_hint=8)
    lay = params.lineph_layout(LM.EPH, ld="collapsed")
    _, _, lo, _ = likelihood.build_target(lay, cen, ed, orders, **kw)
    assert type(lo) is ldmarg.MarginalLDLogProb and lo.prior
    _, _, lo, _ = likelihood.build_target(lay, cen, ed, orders,
                                          likelihood_only=True, **kw)
    assert type(lo) is ldmarg.ProfileLDLogProb and not lo.prior


# -- the maximum equals the joint maximum ------------------------------------------

@pytest.mark.parametrize("law", ["quadratic", "hybrid4"])
def test_profiled_mle_is_the_joint_mle(law):
    """The joint sampled-LD likelihood (theta and q, no profiling anywhere)
    cannot beat the profiled maximum from wide starts, and at the profile's
    (theta, q) it has the profile's value and is already at its optimum.

    The profile can beat the joint search: on hybrid4 the joint ended 0.010
    nats short, with weights on the simplex's faces (q on the box edge, u
    unbounded), which the profile's exact QP reaches directly."""
    import anvil

    if law == "quadratic":
        ed, cen, orders = LM._dataset()
        eph, v_true = LM.EPH, np.r_[LM.THETA, 0.30, 0.225]
    else:
        ed, cen, _, _, orders = H4.make_dataset()
        eph, v_true = H4.EPH, H4.V_TRUE
    kw = dict(num_resample=1, exposure_time=LM.EXP if law == "quadratic"
              else H4.EXP, n_chains_hint=128, likelihood_only=True)
    play = params.lineph_layout(eph, law=law, ld="collapsed")
    ptr, pT, _, phi = likelihood.build_target(play, cen, ed, orders, **kw)
    jlay = params.lineph_layout(eph, law=law)
    jtr, jT, _, _ = likelihood.build_target(jlay, cen, ed, orders, **kw)
    rng = np.random.default_rng(3)
    rp = anvil.optimize(ptr, pT.from_model_np(v_true[:5])[None]
                        + 0.25 * rng.standard_normal((64, 5)), objective="mle")
    rj = anvil.optimize(jtr, jT.from_model_np(v_true)[None]
                        + 0.25 * rng.standard_normal((64, jlay.dim)),
                        objective="mle")
    assert rj.log_value <= rp.log_value + 2e-3

    with mx.stream(mx.cpu):
        q = phi.conditional_mode_ld(mx.array(rp.model[None], dtype=mx.float64))
    q = np.clip(np.array([float(c[0]) for c in q]), 1e-7, 1 - 1e-7)
    v_joint = np.r_[rp.model, q]
    u_joint = jT.from_model_np(v_joint)
    with mx.stream(mx.cpu):
        at = float(np.array(jtr.log_likelihood_hi(
            mx.array(u_joint[None], dtype=mx.float64)))[0]) + jtr.model_log_norm
    assert at == pytest.approx(rp.log_value, abs=1e-4)
    polish = anvil.optimize(jtr, u_joint, n_starts=8, start_spread=1e-4,
                            objective="mle")
    assert polish.log_value == pytest.approx(rp.log_value, abs=2e-3)
    sd = rp.sd if rp.sd is not None else np.full(5, np.inf)
    np.testing.assert_array_less(np.abs(rp.model - polish.model[:5]),
                                 0.05 * sd + 1e-9)


@pytest.mark.parametrize("b_prior", ["transiting", "nongrazing"])
def test_round_starts_cover_the_impact_parameter(b_prior):
    """Low SNR lets b trade off against the limb darkening, so the round
    search must start across b: KOI-4016.01 and KOI-1430.03 have their best
    fit at b ~ 0.85-0.9, which a ball around the input ephemeris missed
    (0.1.75). Every stratum of beta holds a start; the rest stay a ball."""
    from turin import mle

    lay = params.lineph_layout(LM.EPH, b_prior=b_prior, ld="collapsed")
    T = lay.transform()
    U = mle.round_starts(lay, T, LM.EPH, np.random.default_rng(0))
    assert U.shape == (mle.N_STARTS, lay.dim) and np.all(np.isfinite(U))
    beta = T.model_np(U)[:, lay.index("beta")]
    n = mle.N_SPREAD_BETA
    assert n >= mle.N_STARTS // 4
    assert np.array_equal(np.floor(beta[:n] * n), np.arange(n))
    ball = T.model_np(U[n:])
    v0 = seeding.initial_model_vector(lay, LM.EPH)
    assert np.ptp(ball[:, lay.index("beta")]) < 0.6
    assert np.allclose(np.median(ball, axis=0), v0, atol=0.1)


# -- TTV: the profile over a TTV layout (0.1.81) -------------------------------

TAU_HALF = 0.03


def _ttv_targets(ed, cen, orders, eph, *, law="quadratic", ld="collapsed",
                 exposure_time=LM.EXP, **kw):
    lay = params.ttv_layout(eph, cen, TAU_HALF, law=law, ld=ld)
    kw.setdefault("n_chains_hint", 16)
    tr, T, lo, hi = likelihood.build_target(
        lay, cen, ed, orders, num_resample=1, exposure_time=exposure_time,
        likelihood_only=True, **kw)
    return lay, tr, T, lo, hi


def _law_case(law):
    """(ed, cen, orders, eph, exposure, LinEph theta (dP, dtau0, k, beta,
    T14) off the truth, the sampled-LD truth's q)."""
    if law == "quadratic":
        ed, cen, orders = LM._dataset()
        return (ed, cen, orders, LM.EPH, LM.EXP,
                LM.THETA + np.r_[2e-5, -6e-4, 0, 0, 0], np.r_[0.30, 0.225])
    ed, cen, _, _, orders = H4.make_dataset()
    return (ed, cen, orders, H4.EPH, H4.EXP,
            H4.V_TRUE[:5] + np.r_[2e-5, -6e-4, 0, 0, 0], H4.V_TRUE[5:])


def _ttv_theta(lay, lineph_theta):
    """The TTV vector of a LinEph one: dtau_i = dtau0 + n_i dP."""
    dP, dtau0 = lineph_theta[:2]
    return np.r_[lineph_theta[2:5], dtau0 + lay.n_arr * dP]


@pytest.mark.parametrize("law", ["quadratic", "hybrid4"])
def test_ttv_profile_equals_the_lineph_profile_on_a_linear_ephemeris(law):
    """Nested models: at dtau_i = dtau0 + n_i dP the TTV profile is the
    LinEph profile, value and gradient, so a TTV fit started from the LinEph
    optimum cannot end below it -- exactly at dP = 0. Otherwise only nearly:
    TTV mode holds the period at the reference (as hurin), and the period
    also sets the transit's shape through a/R* (measured: dP = 2e-5 d moves
    ln L by 8e-9 and 2e-7 nats on these two cases)."""
    ed, cen, orders, eph, exp, th, _ = _law_case(law)
    th = th * np.r_[0, 1, 1, 1, 1]
    _, _, _, _, lhi = _profile_targets(ed, cen, orders, eph, law=law,
                                       exposure_time=exp)
    tlay, _, _, _, thi = _ttv_targets(ed, cen, orders, eph, law=law,
                                      exposure_time=exp)
    assert type(thi).__name__ == ("ProfileLDLogProb" if law == "quadratic"
                                  else "ProfileHybridLogProb")
    assert tlay.n_shape == 3 and tlay.dim == 3 + tlay.n_epochs
    tth = _ttv_theta(tlay, th)
    assert float(_hi(thi, tth)[0]) == pytest.approx(float(_hi(lhi, th)[0]),
                                                    abs=1e-9)
    with mx.stream(mx.cpu):
        gl = np.array(mx.grad(lambda v: mx.sum(lhi(v)))(
            mx.array(th[None], dtype=mx.float64)))[0]
        gt = np.array(mx.grad(lambda v: mx.sum(thi(v)))(
            mx.array(tth[None], dtype=mx.float64)))[0]
    np.testing.assert_allclose(gt[:3], gl[2:5], rtol=1e-8, atol=1e-8)
    # the chain rule back to LinEph's dtau0 (dP also acts through the
    # period's effect on the shape, which TTV mode does not have)
    assert gt[3:].sum() == pytest.approx(gl[1], rel=1e-8, abs=1e-6)


@pytest.mark.parametrize("law", ["quadratic", "hybrid4"])
def test_ttv_profile_is_the_maximum_over_the_simplex(law):
    """Off a linear ephemeris (every epoch shifted differently), the TTV
    profile is the brute-force maximum over the limb-darkening simplex."""
    ed, cen, orders, eph, exp, th, _ = _law_case(law)
    tlay, _, _, _, thi = _ttv_targets(ed, cen, orders, eph, law=law,
                                      exposure_time=exp)
    rng = np.random.default_rng(5)
    tth = _ttv_theta(tlay, th)
    tth[3:] += 2e-3 * rng.standard_normal(tlay.n_epochs)
    d = 2 if law == "quadratic" else 4
    for v in (tth, tth * np.r_[1.02, 0.9, 1.01, np.ones(tlay.n_epochs)]):
        prof = float(_hi(thi, v)[0])
        brute = _brute_max(thi, v, d, rng)
        assert abs(prof - brute) < (1e-7 if d == 2 else 1e-6), prof - brute


def test_ttv_envelope_gradient_matches_finite_differences():
    ed, cen, orders = LM._dataset(**LM.CASES["moderate"])
    tlay, _, _, _, thi = _ttv_targets(ed, cen, orders, LM.EPH,
                                      exposure_time=0.0)
    th = _ttv_theta(tlay, LM.CASE_THETA["moderate"])
    th[3:] += 1e-3 * np.random.default_rng(6).standard_normal(tlay.n_epochs)
    with mx.stream(mx.cpu):
        g = np.array(mx.grad(lambda v: mx.sum(thi(v)))(
            mx.array(th[None], dtype=mx.float64)))[0]
    cols = [0, 1, 2, 3, 3 + tlay.n_epochs // 2]
    h = np.r_[1e-7, 1e-6, 1e-7, 1e-7, 1e-7]
    eye = np.eye(th.size)
    fd = np.array([(_hi(thi, th + h[j] * eye[i])[0]
                    - _hi(thi, th - h[j] * eye[i])[0]) / (2 * h[j])
                   for j, i in enumerate(cols)])
    np.testing.assert_allclose(g[cols], fd, rtol=2e-5,
                               atol=1e-3 * np.abs(fd).max())


def test_ttv_likelihood_only_drops_the_timing_prior():
    """The sampled-LD TTV likelihood is the sampling density minus the b-k
    and TTV priors exactly."""
    ed, cen, orders = LM._dataset()
    kw = dict(num_resample=1, exposure_time=LM.EXP, n_chains_hint=8)
    lay = params.ttv_layout(LM.EPH, cen, TAU_HALF)
    _, _, post, _ = likelihood.build_target(lay, cen, ed, orders, **kw)
    tr, _, lik, _ = likelihood.build_target(lay, cen, ed, orders,
                                            likelihood_only=True, **kw)
    rng = np.random.default_rng(7)
    V = np.tile(np.r_[_ttv_theta(lay, LM.THETA)[:3], 0.30, 0.225,
                      np.zeros(lay.n_epochs)], (4, 1))
    V[:, 5:] = 0.02 * rng.uniform(-1, 1, (4, lay.n_epochs))  # into the ramp
    v = mx.array(V, dtype=mx.float32)
    p = post.unpack(v)
    want = (np.array(post(v))
            - np.array(params.bk_log_prior(p["beta"], p["k"], lay.b_prior))
            - np.array(params.ttv_log_prior(p["dtau"], lay.P_ref)))
    np.testing.assert_allclose(np.array(lik(v)), want, rtol=0, atol=2e-3)
    assert tr.model_log_norm == lik.log_const != 0.0


@pytest.mark.parametrize("law", ["quadratic", "hybrid4"])
def test_ttv_profiled_mle_is_the_joint_mle(law):
    """As the LinEph test: the joint (theta, q) TTV search from the same
    starts cannot beat the profiled TTV maximum, and at the profile's
    (theta, q) the joint likelihood has the profile's value."""
    import anvil

    ed, cen, orders, eph, exp, th, q_true = _law_case(law)
    play, ptr, pT, _, phi = _ttv_targets(ed, cen, orders, eph, law=law,
                                         exposure_time=exp, n_chains_hint=128)
    jlay, jtr, jT, _, _ = _ttv_targets(ed, cen, orders, eph, law=law,
                                       ld="sampled", exposure_time=exp,
                                       n_chains_hint=128)
    v0 = _ttv_theta(play, th)
    rng = np.random.default_rng(8)

    def starts(T, v, d):
        U = T.from_model_np(v)[None] + 0.25 * rng.standard_normal((64, d))
        U[:, 3:3 + play.n_epochs] = T.from_model_np(v)[3:3 + play.n_epochs]
        return U

    rp = anvil.optimize(ptr, starts(pT, v0, play.dim), objective="mle")
    vj = np.r_[v0[:3], q_true, v0[3:]]
    Uj = jT.from_model_np(vj)[None] + 0.25 * rng.standard_normal(
        (64, jlay.dim))
    times = slice(jlay.n_shape, None)
    Uj[:, times] = jT.from_model_np(vj)[times]
    rj = anvil.optimize(jtr, Uj, objective="mle")
    assert rj.log_value <= rp.log_value + 2e-3

    with mx.stream(mx.cpu):
        q = phi.conditional_mode_ld(mx.array(rp.model[None], dtype=mx.float64))
    q = np.clip(np.array([float(c[0]) for c in q]), 1e-7, 1 - 1e-7)
    u_joint = jT.from_model_np(np.r_[rp.model[:3], q, rp.model[3:]])
    with mx.stream(mx.cpu):
        at = float(np.array(jtr.log_likelihood_hi(
            mx.array(u_joint[None], dtype=mx.float64)))[0]) + jtr.model_log_norm
    assert at == pytest.approx(rp.log_value, abs=1e-4)


def test_ttv_profile_refusals():
    """Sampling keeps refusing a collapsed TTV layout (grid-Gibbs needs the
    per-epoch factorisation); the profile takes it but has no per-epoch
    terms; ttv_layout checks its ld mode."""
    ed, cen, orders = LM._dataset()
    kw = dict(num_resample=1, exposure_time=LM.EXP, n_chains_hint=4)
    col = params.ttv_layout(LM.EPH, cen, TAU_HALF, ld="collapsed")
    with pytest.raises(ValueError, match="LinEph-only"):
        likelihood.build_target(col, cen, ed, orders, **kw)
    hcol = params.ttv_layout(H4.EPH, cen, TAU_HALF, law="hybrid4",
                             ld="collapsed")
    with pytest.raises(ValueError, match="LinEph-only"):
        likelihood.build_target(hcol, cen, ed, orders, **kw)
    _, _, lo, _ = likelihood.build_target(col, cen, ed, orders,
                                          likelihood_only=True, **kw)
    assert type(lo) is ldmarg.ProfileLDLogProb
    with pytest.raises(NotImplementedError, match="factorisation"):
        lo.epoch_log_lik(mx.zeros((1, col.dim), dtype=mx.float32))
    with pytest.raises(ValueError, match="ld mode"):
        params.ttv_layout(LM.EPH, cen, TAU_HALF, ld="nope")
