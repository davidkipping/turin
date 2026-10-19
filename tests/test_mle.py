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
