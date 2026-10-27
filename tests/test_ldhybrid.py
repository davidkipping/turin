"""Collapsed hybrid4 limb darkening (``--law=hybrid4 --ld=collapsed``).

Each piece against a reference that does not share its approximation:
scipy for the normal tails, NumPy's QR and a host QP for the linear algebra,
direct Monte Carlo from the Gaussian for GHK, turin's sampled hybrid4
likelihood for the conditional, importance sampling of the *true* likelihood
for the collapsed value, uniform-simplex quadrature weighted by the true
likelihood for the conditional draws, float64 for float32.
"""

from __future__ import annotations

import math
import sys
import os

import mlx.core as mx
import numpy as np
import pytest
from scipy.special import log_ndtr, ndtr
from scipy.stats import qmc

from turin import likelihood, params
from turin import ldhybrid as LH
from turin import model as M
from turin import normtail as NT
from turin import simplexmass as SM

sys.path.insert(0, os.path.dirname(__file__))
import test_hybrid4 as H4                                   # noqa: E402


# -- normal tails ------------------------------------------------------------

def _cdf_fraction(z, a, b):
    if a >= 0:
        la, lb, lz = log_ndtr(-a), log_ndtr(-b), log_ndtr(-z)
        return -np.expm1(lz - la) / -np.expm1(lb - la)
    if b <= 0:
        la, lb, lz = log_ndtr(a), log_ndtr(b), log_ndtr(z)
        return -np.expm1(la - lz) * np.exp(lz - lb) / -np.expm1(la - lb)
    return (ndtr(z) - ndtr(a)) / (ndtr(b) - ndtr(a))


def test_log_ndtr_matches_scipy_in_both_tails():
    x = np.linspace(-40.0, 8.0, 3001)
    with mx.stream(mx.cpu):
        got = np.array(NT.log_ndtr(mx.array(x, dtype=mx.float64)))
    ref = log_ndtr(x)
    assert np.max(np.abs(got - ref) / np.maximum(np.abs(ref), 1e-12)) < 1e-5
    assert np.max(np.abs(got - ref)) < 1e-6


def test_ndtri_exp_inverts_log_ndtr_where_trunc_normal_uses_it():
    """The lower tail and the centre. Past z ~ 4, log Phi(z) ~ -3e-5 holds
    too little of z to invert (measured 1e-4 at 4-5, 7e-3 at 5-5.3), which
    is why trunc_normal reflects every interval to a non-positive centre."""
    z = np.linspace(-37.0, 4.0, 2001)
    with mx.stream(mx.cpu):
        got = np.array(NT.ndtri_exp(mx.array(log_ndtr(z), dtype=mx.float64)))
    assert np.max(np.abs(got - z) / np.maximum(np.abs(z), 1.0)) < 1e-6


def test_truncated_normal_draws_are_exact_and_masses_correct():
    rng = np.random.default_rng(0)
    a = np.r_[rng.uniform(-30, 30, 4000), rng.uniform(-30, 0, 1000)]
    b = np.r_[a[:4000] + rng.exponential(1.0, 4000),
              a[4000:] + rng.uniform(0, 60, 1000)]       # wide straddles too
    v = rng.uniform(1e-6, 1 - 1e-6, a.size)
    with mx.stream(mx.cpu):
        z, lm = NT.trunc_normal(*(mx.array(t, dtype=mx.float64)
                                  for t in (a, b, v)))
    z, lm = np.array(z), np.array(lm)
    assert np.all((z >= a) & (z <= b))
    frac = np.array([_cdf_fraction(z[i], a[i], b[i]) for i in range(a.size)])
    want = np.where(a + b > 0, 1.0 - v, v)    # the reflected branch reverses v
    assert np.max(np.abs(frac - want)) < 2e-6
    ref = np.array([_cdf_fraction(b[i], a[i], b[i]) for i in range(3)])
    assert np.allclose(ref, 1.0)
    mass = np.where(a >= 0, log_ndtr(-a) + np.log(-np.expm1(log_ndtr(-b) - log_ndtr(-a))),
                    np.where(b <= 0, log_ndtr(b) + np.log(-np.expm1(log_ndtr(a) - log_ndtr(b))),
                             np.log(np.maximum(ndtr(b) - ndtr(a), 1e-300))))
    assert np.max(np.abs(lm - mass)) < 1e-5


def test_truncated_normal_gradient_is_finite_deep_in_a_tail():
    a = mx.array([25.0, -40.0, 30.0], dtype=mx.float32)
    b = a + mx.array([1e-3, 1e-4, 2.0], dtype=mx.float32)
    g = mx.grad(lambda s: mx.sum(NT.trunc_normal(a * s, b * s,
                mx.full(a.shape, 0.3, dtype=mx.float32))[1]))(
        mx.array(1.0, dtype=mx.float32))
    assert np.isfinite(float(g))


# -- linear algebra and the QP ------------------------------------------------

def test_mgs_and_tsqr_match_numpy_qr():
    rng = np.random.default_rng(1)
    A = rng.normal(size=(3, 400, 4)) * np.array([1, 1e-3, 1e2, 1])
    with mx.stream(mx.cpu):
        R = np.array(SM.mgs_r([mx.array(A[:, :, j], dtype=mx.float64)
                               for j in range(4)]))
        R2 = np.array(SM.tsqr([SM.mgs_r([mx.array(A[:, s, j], dtype=mx.float64)
                                         for j in range(4)])
                               for s in (slice(0, 150), slice(150, 400))]))
    ref = []
    for a in A:
        r = np.linalg.qr(a)[1]
        ref.append(r * np.sign(np.diag(r))[:, None])
    ref = np.array(ref)
    assert np.abs(R - ref).max() / np.abs(ref).max() < 1e-13
    assert np.abs(R2 - ref).max() / np.abs(ref).max() < 1e-13


def test_simplex_qp_on_r_is_the_exact_constrained_optimum():
    rng = np.random.default_rng(2)
    worst = 0.0
    for scale in (1.0, 1e3, 1e8):
        for _ in range(10):
            A = rng.normal(size=(4, 4)) * np.array([1, 0.3, 1e-2, 3])
            H = scale * (A @ A.T + 1e-4 * np.eye(4))
            xh = rng.dirichlet(np.ones(5))[1:] + rng.normal(0, 0.3, 4)
            ref = SM.qp_simplex_np(xh, H)
            R = np.linalg.cholesky(H).T
            with mx.stream(mx.cpu):
                x = np.array(SM.qp_simplex_r(
                    mx.array(R[None], dtype=mx.float64),
                    mx.array((R @ xh)[None], dtype=mx.float64)))[0]
            f = lambda z: (z - xh) @ H @ (z - xh)
            assert np.all(x >= 0) and x.sum() <= 1 + 1e-12
            worst = max(worst, (f(x) - f(ref)) / max(f(ref), 1e-12))
    assert worst < 1e-8
    # and no lattice point beats it (a check independent of both solvers)
    U = qmc.Sobol(5, scramble=True, seed=3).random(2 ** 14)
    E = -np.log(U)
    X = (E / E.sum(1, keepdims=True))[:, 1:]
    q = np.einsum("ni,ij,nj->n", X - xh, H, X - xh)
    assert f(x) <= q.min() + 1e-9 * max(1.0, q.min())


# -- GHK ----------------------------------------------------------------------

def test_the_metal_qp_matches_the_graph_and_is_as_accurate():
    """float32 on the GPU runs the QP as one Metal kernel per chain: it
    agrees with the MLX graph to float32 rounding and is no less accurate
    against float64, on random problems whose column scales span e^+-3
    (both dimensions turin uses: hybrid4's 4, the quadratic's 2)."""
    for D in (4, 2):
        rng = np.random.default_rng(D)
        C = 4000
        A = rng.standard_normal((C, D, D)) * np.exp(rng.uniform(-3, 3, (C, 1, D)))
        R = np.linalg.qr(A)[1]
        R = R * np.sign(np.diagonal(R, axis1=1, axis2=2))[:, :, None]
        b = 3 * rng.standard_normal((C, D))
        R32, b32 = mx.array(R, dtype=mx.float32), mx.array(b, dtype=mx.float32)
        xk = np.array(SM.qp_simplex_r(R32, b32))           # the kernel
        xg = np.array(SM._qp_simplex_r_graph(R32, b32))
        with mx.stream(mx.cpu):
            x64 = np.array(SM.qp_simplex_r(mx.array(R, dtype=mx.float64),
                                           mx.array(b, dtype=mx.float64)))
        ek, eg = np.abs(xk - x64).max(1), np.abs(xg - x64).max(1)
        assert np.all(xk >= 0) and np.all(xk.sum(1) <= 1 + 1e-6)
        assert np.median(np.abs(xk - xg).max(1)) < 1e-6
        assert np.median(ek) <= 2 * np.median(eg) + 1e-9
        assert np.quantile(ek, 0.99) <= 2 * np.quantile(eg, 0.99) + 1e-9


def _gauss_case(rng, scale):
    A = rng.normal(size=(4, 4))
    H = scale * (A @ A.T + 0.05 * np.eye(4))
    xh = rng.dirichlet(np.ones(5))[1:] + rng.normal(0, 0.1, 4)
    return xh, H


def _ghk_tail_case(C=48, D=4, seed=0):
    """(my, Ly) spanning the hard cases: centres well outside the simplex
    (intervals hundreds of sd into a tail, reflected and not) and narrow
    intervals deep in a tail, where the gradient is a small difference of
    large terms."""
    rng = np.random.default_rng(seed)
    my = rng.uniform(-0.3, 0.6, (C, D))
    my[: C // 3] *= rng.uniform(10, 60, (C // 3, 1))         # far outside
    Ly = np.zeros((C, D, D))
    for c in range(C):
        A = rng.standard_normal((D, D)) * np.exp(rng.uniform(-5, 1, D))
        Ly[c] = np.linalg.cholesky(A @ A.T + 1e-7 * np.eye(D))
    # collapsed intervals: the first weight pressed against 1 leaves the
    # next an interval narrower than the eps floor (float64's too)
    for c in range(4):
        my[c] = [1.5, 0.1, 0.1, 0.1]
        Ly[c] = np.diag([1e-3, 1e4, 1.0, 1.0])
    return my, Ly


def test_ghk_hand_vjp_is_autodiffs_gradient_and_the_same_value():
    """``ghk_lw_vjp`` (float32 on the GPU's ``ghk_log_mass``): the value is
    ``ghk_terms``'s bitwise in both precisions. In float64 the hand gradient
    equals autodiff's to 1e-6 under a cotangent weighting every point alike
    (the formulas are exact everywhere, degenerate points included). In
    float32 it is judged on what the density uses, the logsumexp over
    points, and must be no less accurate than float32 autodiff per chain --
    on centres far outside the simplex and narrow intervals deep in a tail
    (where taking ratios as exp(log phi - log P) once cost 0.4%). Points
    with zero-width intervals hundreds of sd out (weight ~e^-1e5 in the
    logsumexp) round differently from autodiff there, harmlessly."""
    my, Ly = _ghk_tail_case()
    U = SM.sobol_points(512, 4, 7)
    dep, order = 1, (3, 0, 4, 2)
    W = np.random.default_rng(1).standard_normal((my.shape[0], 512))

    def obj_all(lw):                 # every point weighted alike
        return (mx.sum(lw * mx.array(W, dtype=lw.dtype) * 1e-3)
                + mx.sum(mx.logsumexp(lw, axis=-1)))

    def obj_used(lw):                # what ghk_log_mass differentiates
        return mx.sum(mx.logsumexp(lw, axis=-1))

    hand = lambda m, L, Ux: SM.ghk_lw_vjp(m, L, Ux, dep, order, LH.INV_FSTAR)
    auto = lambda m, L, Ux: SM.ghk_terms(m, L, Ux, dep, order, LH.INV_FSTAR)[0]

    def run(fn, dt, stream, obj):
        with mx.stream(stream):
            m, L, Ux = (mx.array(a, dtype=dt) for a in (my, Ly, U))
            val = np.array(fn(m, L, Ux), dtype=np.float64)
            g = mx.grad(lambda a, b: obj(fn(a, b, Ux)), argnums=(0, 1))(m, L)
            return val, [np.array(x, dtype=np.float64) for x in g]

    def err(g, ref):     # per chain, relative to that chain's largest entry
        e0 = np.abs(g[0] - ref[0]).max(1) / np.abs(ref[0]).max(1)
        e1 = (np.abs(g[1] - ref[1]).max((1, 2))
              / np.abs(ref[1]).max((1, 2)))
        return np.maximum(e0, e1)

    v64h, g64h = run(hand, mx.float64, mx.cpu, obj_all)
    v64a, g64a = run(auto, mx.float64, mx.cpu, obj_all)
    assert np.array_equal(v64h, v64a)
    assert err(g64h, g64a).max() < 1e-6

    v32h, g32h = run(hand, mx.float32, mx.gpu, obj_used)
    v32a, g32a = run(auto, mx.float32, mx.gpu, obj_used)
    _, ref = run(auto, mx.float64, mx.cpu, obj_used)
    assert np.array_equal(v32h, v32a)
    assert np.all(np.isfinite(g32h[0])) and np.all(np.isfinite(g32h[1]))
    eh, ea = err(g32h, ref), err(g32a, ref)
    assert np.all(eh <= 3 * ea + 1e-6), (eh, ea)


@pytest.mark.parametrize("case", ["tail", "moderate"])
def test_ghk_metal_kernels_match_the_graph_in_value_and_gradient(case):
    """turin.ghkmetal (float32 on the GPU): ``lw`` and the gradient of the
    logsumexp over points -- what ``ghk_log_mass`` uses -- against float64,
    no less accurate than the MLX graph and its autodiff. Values: logZ per
    chain relative to max(|logZ|, 1) (deep-tail chains reach logZ ~ -1e8,
    where a float32 ulp is 8 nats). Gradients: every chain within float32's
    reach (2e-4 relative) or 3x autodiff's own error, and no worse than it
    at the worst chain."""
    if case == "tail":
        my, Ly = _ghk_tail_case()
        U, dep, order = SM.sobol_points(512, 4, 7), 1, (3, 0, 4, 2)
    else:
        rng = np.random.default_rng(3)
        C, D = 128, 4
        my = rng.uniform(-0.3, 0.6, (C, D))
        Ly = np.zeros((C, D, D))
        for c in range(C):
            A = rng.standard_normal((D, D)) * np.exp(rng.uniform(-4, -1, D))
            Ly[c] = np.linalg.cholesky(A @ A.T + 1e-6 * np.eye(D))
        U, dep, order = SM.sobol_points(1024, 4, 2024), 0, (1, 2, 3, 4)
    from turin import ghkmetal as GM

    m, L, Ux = (mx.array(a, dtype=mx.float32) for a in (my, Ly, U))
    metal = lambda a, b: GM.ghk_lw(a, b, Ux, dep, order, LH.INV_FSTAR)
    graph = lambda a, b: SM.ghk_terms(a, b, Ux, dep, order, LH.INV_FSTAR)[0]
    with mx.stream(mx.cpu):
        a6 = [mx.array(a) for a in (my, Ly, U)]
        lw64 = np.array(SM.ghk_terms(*a6, dep, order, LH.INV_FSTAR)[0])
        g64 = [np.array(x) for x in mx.grad(
            lambda a, b: mx.sum(mx.logsumexp(SM.ghk_terms(
                a, b, a6[2], dep, order, LH.INV_FSTAR)[0], axis=-1)),
            argnums=(0, 1))(a6[0], a6[1])]

    def lse(x):
        return x.max(-1) + np.log(np.exp(x - x.max(-1, keepdims=True)).sum(-1))

    sc = np.maximum(np.abs(lse(lw64)), 1.0)
    rel = lambda f: np.abs(lse(np.array(f(m, L), dtype=np.float64))
                           - lse(lw64)) / sc
    rk, rg = rel(metal), rel(graph)
    assert np.all(np.isfinite(rk))
    assert rk.max() <= 2 * rg.max() + 1e-6
    assert np.median(rk) <= 2 * np.median(rg) + 1e-7

    def grad_err(f):
        g = [np.array(x, dtype=np.float64) for x in mx.grad(
            lambda a, b: mx.sum(mx.logsumexp(f(a, b), axis=-1)),
            argnums=(0, 1))(m, L)]
        e0 = np.abs(g[0] - g64[0]).max(1) / np.abs(g64[0]).max(1)
        e1 = np.abs(g[1] - g64[1]).max((1, 2)) / np.abs(g64[1]).max((1, 2))
        return np.maximum(e0, e1)

    ek, ea = grad_err(metal), grad_err(graph)
    assert np.all(np.isfinite(ek))
    assert np.all((ek <= 2e-4) | (ek <= 3 * ea)), (ek.max(), ea.max())
    assert ek.max() <= 2 * ea.max() + 1e-6


@pytest.mark.parametrize("C,N", [(1, 4), (1, 1024), (3, 8)])
def test_ghk_metal_kernels_take_small_inputs(C, N):
    """MLX passes a small kernel input in Metal's constant address space,
    a large one in device memory; the kernels compile for both (a single
    chain, as the MAP search's rescoring sends, failed to build before)."""
    from turin import ghkmetal as GM

    rng = np.random.default_rng(C * N)
    my = mx.array(rng.uniform(0, 0.3, (C, 4)), dtype=mx.float32)
    Ly = mx.array(np.tile(np.diag([0.1, 0.05, 0.03, 0.02]), (C, 1, 1)),
                  dtype=mx.float32)
    U = mx.array(SM.sobol_points(N, 4, 1), dtype=mx.float32)
    order = (1, 2, 3, 4)
    a = np.array(GM.ghk_lw(my, Ly, U, 0, order, LH.INV_FSTAR))
    b = np.array(SM.ghk_terms(my, Ly, U, 0, order, LH.INV_FSTAR)[0])
    np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-4)
    g = mx.grad(lambda m: mx.sum(GM.ghk_lw(m, Ly, U, 0, order,
                                           LH.INV_FSTAR)))(my)
    assert np.isfinite(np.array(g)).all()


def test_ghk_matches_direct_monte_carlo():
    """``log int_simplex exp(-1/2 (x-xh)^T H (x-xh)) p(x) dx`` by GHK vs by
    direct sampling of the Gaussian (cases where enough of it is inside)."""
    rng = np.random.default_rng(1)
    r2 = np.random.default_rng(9)
    U = SM.sobol_points(1024, 4, 2024)
    checked = 0
    for scale in (30.0, 300.0, 3000.0):
        for _ in range(3):
            xh, H = _gauss_case(rng, scale)
            xs = SM.qp_simplex_np(xh, H)
            dep, order = SM.choose_layout(xs, H)
            T, t = SM.layout_map(dep, order, 5)
            Ly = np.linalg.cholesky(T @ np.linalg.inv(H) @ T.T)
            with mx.stream(mx.cpu):
                got = float(SM.ghk_log_mass(
                    mx.array((T @ xh + t)[None], dtype=mx.float64),
                    mx.array(Ly[None], dtype=mx.float64),
                    mx.array(U, dtype=mx.float64), dep, order,
                    LH.INV_FSTAR)[0])
            X = xh + r2.standard_normal((400_000, 4)) @ np.linalg.cholesky(
                np.linalg.inv(H)).T
            lp = LH.log_prior_x_np(X)
            inside = np.isfinite(lp)
            if inside.mean() < 0.02:
                continue
            vals = np.where(inside, np.exp(np.where(inside, lp, 0.0)), 0.0)
            direct = (np.log(vals.mean()) + 2 * math.log(2 * math.pi)
                      - 0.5 * np.linalg.slogdet(H)[1])
            se = vals.std() / vals.mean() / math.sqrt(len(vals))
            assert abs(got - direct) < 4 * se + 0.02, (got, direct, se)
            checked += 1
    assert checked >= 5


# -- the generators, the prior, the conditional --------------------------------

def test_every_hybrid4_law_is_a_convex_mix_of_the_generators():
    cen = dict(times_centered=np.linspace(-0.3, 0.3, 400)[None, :],
               n_arr=np.zeros(1), d_arr=np.zeros(1), P_ref=9.0, tau0_ref=0.0)
    col = lambda v: mx.array([[float(v)]], dtype=mx.float64)
    with mx.stream(mx.cpu):
        grid = M.build_grid(cen, np.zeros(1), dtype=mx.float64,
                            exp_time=29.4 / 1440, mask=np.ones((1, 400)))
        kw = dict(mid=mx.zeros((1, 1), dtype=mx.float64), k=col(0.1),
                  b=col(0.95), T14=col(0.15), period=col(9.0))
        F = [np.array(f)[0, 0] for f in M.generator_flux_devs(grid, **kw)]
        for q in np.random.default_rng(0).uniform(size=(6, 4)):
            om = LH.w_to_omega(M.ld_weights_np(q))
            ref = np.array(M.transit_flux_dev(
                grid, q=tuple(col(x) for x in q), law="hybrid4", **kw))[0, 0]
            assert np.abs(sum(o * f for o, f in zip(om, F)) - ref).max() < 1e-13
    assert np.allclose(LH.x_to_q(LH.q_to_x(np.array([[0.3, 0.6, 0.2, 0.9]]))),
                       [[0.3, 0.6, 0.2, 0.9]], atol=1e-12)


def test_uniform_q_induces_the_minus_five_prior_on_omega():
    """E[omega] under uniform q, against the same expectation under the
    claimed density -5 log sum omega/F* (by uniform-simplex quadrature)."""
    q = np.random.default_rng(4).uniform(size=(400_000, 4))
    om = LH.w_to_omega(M.ld_weights_np(q))
    U = qmc.Sobol(5, scramble=True, seed=6).random(2 ** 18)
    E = -np.log(U)
    OM = E / E.sum(1, keepdims=True)
    w = np.exp(LH.log_prior_x_np(OM[:, 1:]))
    np.testing.assert_allclose(om.mean(0), w @ OM / w.sum(), atol=2e-3)


@pytest.fixture(scope="module")
def data():
    return H4.make_dataset()


def _targets(data, **kw):
    ed, cen, _, _, orders = data
    lay = params.lineph_layout(H4.EPH, ld="collapsed", law="hybrid4")
    with mx.stream(mx.cpu):
        hi = LH.MarginalHybridLogProb(lay, cen, ed, orders, num_resample=7,
                                      exposure_time=H4.EXP, n_chains_hint=8,
                                      dtype=mx.float64, **kw)
    return lay, hi


def test_conditional_loglik_is_the_sampled_hybrid4_likelihood(data):
    ed, cen, _, _, orders = data
    _, hi = _targets(data)
    sam = params.lineph_layout(H4.EPH, law="hybrid4")
    with mx.stream(mx.cpu):
        lp = likelihood.ProfiledTransitLogProb(
            sam, cen, ed, orders, num_resample=7, exposure_time=H4.EXP,
            n_chains_hint=8, dtype=mx.float64)
        q = np.random.default_rng(5).uniform(0.05, 0.95, size=(6, 4))
        v7 = np.column_stack([np.tile(H4.V_TRUE[:5], (6, 1)), q])
        vm = mx.array(v7, dtype=mx.float64)
        want = np.array(lp(vm)) - np.array(lp.log_prior(lp.unpack(vm)))
        got = np.array(hi.loglik_at(mx.array(v7[:, :5], dtype=mx.float64),
                                    LH.q_to_x(q)))
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-8)


def test_collapsed_value_is_the_true_integral(data):
    """The collapsed value against ``log int exp(L_true(x)) p(x) dx`` by
    importance sampling the true likelihood from the model's own GHK draws:
    this isolates the quadratic model's error (GHK's own is tested below)."""
    _, hi = _targets(data)
    rng = np.random.default_rng(3)
    for i in range(3):
        v = H4.V_TRUE[:5].copy()
        if i:
            v[2:] *= 1 + 0.004 * rng.standard_normal(3)
        with mx.stream(mx.cpu):
            vm = mx.array(v[None], dtype=mx.float64)
            x, L, g, R = hi.ld_terms(vm)
            u, my, Ly = hi._gauss(x, g, R)
            dep, order = hi.ghk_layout
            lw, Y = SM.ghk_terms(my, Ly, mx.array(SM.sobol_points(4096, 4, 77)
                                                  [None], dtype=mx.float64),
                                 dep, order, LH.INV_FSTAR)
            X = np.clip(SM.omega_from_free(np.array(Y)[0], dep, order)[:, 1:],
                        0, 1)
            Lt = np.array(hi.loglik_grid(v, X))
        lw = np.array(lw)[0]
        xs, Ls, gs, Rs = (np.array(a)[0] for a in (x, L, g, R))
        Lq = Ls + (X - xs) @ gs - 0.5 * np.sum(((X - xs) @ Rs.T) ** 2, 1)
        a = lw + (Lt - Lq)
        ratio = (np.log(np.exp(a - a.max()).sum()) + a.max()
                 - np.log(np.exp(lw - lw.max()).sum()) - lw.max())
        assert abs(ratio) < 2e-3, ratio


def test_expansion_point_is_the_exact_optimum_of_the_centroid_model(data):
    """x* is the curvature-metric QP optimum of the model at the centroid
    (as Gate A validated), not a Euclidean projection of the Newton step."""
    _, hi = _targets(data)
    v = H4.V_TRUE[:5].copy()
    with mx.stream(mx.cpu):
        vm = mx.array(v[None], dtype=mx.float64)
        blocks = hi._gen_blocks(hi.unpack(vm))
        x0 = mx.full((1, 4), 0.2, dtype=mx.float64)
        _, g0, R0 = LH._sum_terms(blocks, x0)
        xs = np.array(hi.ld_terms(vm)[0])[0]
    g0, R0 = np.array(g0)[0], np.array(R0)[0]
    H = R0.T @ R0
    want = SM.qp_simplex_np(np.full(4, 0.2) + np.linalg.solve(H, g0), H)
    np.testing.assert_allclose(xs, want, atol=1e-8)


def test_ghk_rule_error_at_the_posterior(data):
    """N = 1024 against a 2^18 randomised reference, on the model at three
    posterior-like rows (measured on real targets: max 0.015 and 95th
    percentile 0.03 nats)."""
    _, hi = _targets(data)
    rng = np.random.default_rng(6)
    for i in range(3):
        v = H4.V_TRUE[:5].copy()
        v[2:] *= 1 + 0.004 * rng.standard_normal(3)
        with mx.stream(mx.cpu):
            x, L, g, R = hi.ld_terms(mx.array(v[None], dtype=mx.float64))
            u, my, Ly = hi._gauss(x, g, R)
            dep, order = hi.ghk_layout
            got = float(SM.ghk_log_mass(my, Ly, hi._U, dep, order,
                                        LH.INV_FSTAR)[0])
            refs = [float(SM.ghk_log_mass(my, Ly, mx.array(
                SM.sobol_points(2 ** 14, 4, 100 + r), dtype=mx.float64),
                dep, order, LH.INV_FSTAR)[0]) for r in range(16)]
        ref = max(refs) + math.log(np.mean(np.exp(np.array(refs) - max(refs))))
        assert abs(got - ref) < 0.03, (got, ref)


def test_float32_tracks_float64_and_blocking_is_exact(data):
    ed, cen, _, _, orders = data
    lay, hi = _targets(data)
    lo = LH.MarginalHybridLogProb(lay, cen, ed, orders, num_resample=7,
                                  exposure_time=H4.EXP, n_chains_hint=64,
                                  dtype=mx.float32)
    rng = np.random.default_rng(1)
    V = np.tile(H4.V_TRUE[:5], (64, 1))
    V[:, 2:] *= 1 + 0.01 * rng.standard_normal((64, 3))
    f32 = np.array(lo(mx.array(V, dtype=mx.float32)), dtype=np.float64)
    with mx.stream(mx.cpu):
        f64 = np.array(hi(mx.array(V, dtype=mx.float64)))
        hb = LH.MarginalHybridLogProb(lay, cen, ed, orders, num_resample=7,
                                      exposure_time=H4.EXP, n_chains_hint=64,
                                      dtype=mx.float64, budget_bytes=1)
        fb = np.array(hb(mx.array(V, dtype=mx.float64)))
    assert len(hb.blocks) > 1
    assert np.abs(fb - f64).max() < 1e-9
    assert np.std(f32 - f64) < 5e-3
    g32 = np.array(mx.grad(lambda v: lo(v).sum())(
        mx.array(V[:6], dtype=mx.float32)), dtype=np.float64)
    with mx.stream(mx.cpu):
        g64 = np.array(mx.grad(lambda v: hi(v).sum())(
            mx.array(V[:6], dtype=mx.float64)))
    assert (np.abs(g32 - g64) / np.maximum(np.abs(g64), 1.0)).max() < 1e-2


def test_batched_rows_do_not_mix(data):
    """Every per-chain quantity -- x*, L, g, R, the Gaussian, log Z, the
    conditional draws' inputs -- computed for a batch equals the same row
    computed alone (an axis slip in an einsum or a reduction would couple
    chains and still pass every single-theta test)."""
    _, hi = _targets(data)
    rng = np.random.default_rng(3)
    V = np.tile(H4.V_TRUE[:5], (5, 1))
    V[:, 2:] *= 1 + 0.03 * rng.standard_normal((5, 3))

    def parts(rows):
        v = mx.array(rows, dtype=mx.float64)
        x, L, g, R = LH.expansion_point(hi._gen_blocks(hi.unpack(v)),
                                        len(rows), mx.float64)
        u, my, Ly = hi._gauss(x, g, R)
        return [np.array(a) for a in (x, L, g, R, my, Ly, hi(v))]

    with mx.stream(mx.cpu):
        whole = parts(V)
        for i in range(len(V)):
            for w, one in zip(whole, parts(V[i:i + 1])):
                np.testing.assert_allclose(w[i], one[0], rtol=1e-10,
                                           atol=1e-10)


def test_detached_gradient_matches_finite_differences_without_exposure():
    """x* is detached; the value depends on it only through L's departure
    from its quadratic model, so the autodiff gradient must match the total
    derivative. Exposure off: FD is invalid under the contact rule."""
    ed, cen, _, _, orders = H4.make_dataset()
    lay = params.lineph_layout(H4.EPH, ld="collapsed", law="hybrid4")
    with mx.stream(mx.cpu):
        hi = LH.MarginalHybridLogProb(lay, cen, ed, orders, num_resample=1,
                                      exposure_time=0.0, n_chains_hint=4,
                                      dtype=mx.float64)
        v = H4.V_TRUE[:5].copy()
        g = np.array(mx.grad(lambda z: hi(z).sum())(
            mx.array(v[None], dtype=mx.float64)))[0]
        for i, h in ((1, 1e-7), (2, 1e-7), (3, 1e-6), (4, 1e-7)):
            vp, vm = v.copy(), v.copy()
            vp[i] += h
            vm[i] -= h
            fd = (float(np.array(hi(mx.array(vp[None], dtype=mx.float64)))[0])
                  - float(np.array(hi(mx.array(vm[None], dtype=mx.float64)))[0])
                  ) / (2 * h)
            assert abs(g[i] - fd) / max(abs(g[i]), 1.0) < 2e-3, (i, g[i], fd)


def test_conditional_draws_reproduce_the_exact_conditional(data):
    ed, cen, _, _, orders = data
    lay, hi = _targets(data)
    lo = LH.MarginalHybridLogProb(lay, cen, ed, orders, num_resample=7,
                                  exposure_time=H4.EXP, n_chains_hint=1024,
                                  dtype=mx.float32)
    v = H4.V_TRUE[:5].copy()
    n = 4096
    q, acc, _ = LH.draw_limb_darkening(lo, np.tile(v, (n, 1)), seed=4)
    om = LH.w_to_omega(M.ld_weights_np(q))
    assert np.all((q > 0) & (q < 1)) and np.all(om > 0)     # no wall atoms
    U = qmc.Sobol(5, scramble=True, seed=9).random(2 ** 16)
    E = -np.log(U)
    OM = E / E.sum(1, keepdims=True)
    with mx.stream(mx.cpu):
        L = np.concatenate([np.array(hi.loglik_grid(v, OM[i:i + 8192, 1:]))
                            for i in range(0, len(OM), 8192)])
    a = L + LH.log_prior_x_np(OM[:, 1:])
    w = np.exp(a - a.max())
    w /= w.sum()
    mu = w @ OM
    sd = np.sqrt(w @ (OM - mu) ** 2)
    se = om.std(0) / math.sqrt(n / 10)
    assert np.all(np.abs(om.mean(0) - mu) < 4 * se), (om.mean(0), mu)
    np.testing.assert_allclose(om.std(0) / sd, 1.0, atol=0.05)
    assert acc > 0.1


def test_target_compiles_and_the_ghk_rule_is_part_of_it(data):
    ed, cen, _, _, orders = data
    lay, hi = _targets(data)
    lo = LH.MarginalHybridLogProb(lay, cen, ed, orders, num_resample=7,
                                  exposure_time=H4.EXP, n_chains_hint=16,
                                  dtype=mx.float32)
    V = mx.array(np.tile(H4.V_TRUE[:5], (16, 1)), dtype=mx.float32)
    plain = np.array(lo(V))
    val, grad = mx.compile(lambda v: mx.value_and_grad(
        lambda w: lo(w).sum())(v))(V)
    assert np.isfinite(np.array(grad)).all()
    assert abs(float(val) - plain.sum()) < 0.05 * 16
    # a stored layout and rule rebuild the identical density
    _, other = _targets(data, ghk_layout=(2, (1, 4, 3, 0)),
                        ghk_points=SM.sobol_points(256, 4, 11))
    _, same = _targets(data, ghk_layout=other.ghk_layout,
                       ghk_points=other.ghk_points)
    with mx.stream(mx.cpu):
        Vd = mx.array(np.tile(H4.V_TRUE[:5], (2, 1)), dtype=mx.float64)
        assert np.array_equal(np.array(other(Vd)), np.array(same(Vd)))
        assert not np.array_equal(np.array(other(Vd)), np.array(hi(Vd)))


def test_constructor_refusals(data):
    ed, cen, _, _, orders = data
    lay = params.lineph_layout(H4.EPH, ld="collapsed", law="hybrid4")
    with pytest.raises(ValueError, match="exact"):
        LH.MarginalHybridLogProb(lay, cen, ed, orders, profile_mode="ratio")
    with pytest.raises(ValueError, match="circular"):
        LH.MarginalHybridLogProb(lay, cen, ed, orders, geometry="chord")
    with pytest.raises(ValueError, match="hybrid4"):
        LH.MarginalHybridLogProb(params.lineph_layout(H4.EPH, ld="collapsed"),
                                 cen, ed, orders)
    with pytest.raises(ValueError, match="LinEph-only"):
        LH.MarginalHybridLogProb(params.ttv_layout(H4.EPH, cen, 0.03,
                                                   law="hybrid4"),
                                 cen, ed, orders)
    target, _, lo, hi = likelihood.build_target(
        lay, cen, ed, orders, num_resample=7, exposure_time=H4.EXP,
        n_chains_hint=4)
    assert isinstance(lo, LH.MarginalHybridLogProb)
    assert isinstance(hi, LH.MarginalHybridLogProb)


# -- 0.1.64: what the real-target acceptance suite found ---------------------
# Weakly constrained limb darkening (KOI-4016.01: SNR 22, 8 transits) and a
# chain that wandered to k = 0 (KOI-2650.01) broke three things the
# well-constrained synthetic target above cannot see. A noisier copy of it,
# and thetas away from its truth, reproduce each.

@pytest.fixture(scope="module")
def weak():
    ed, cen, _, _, orders = H4.make_dataset(yerr=2e-3)
    lay = params.lineph_layout(H4.EPH, ld="collapsed", law="hybrid4")
    with mx.stream(mx.cpu):
        hi = LH.MarginalHybridLogProb(lay, cen, ed, orders, num_resample=7,
                                      exposure_time=H4.EXP, n_chains_hint=8,
                                      dtype=mx.float64)
    lo = LH.MarginalHybridLogProb(lay, cen, ed, orders, num_resample=7,
                                  exposure_time=H4.EXP, n_chains_hint=2048,
                                  dtype=mx.float32)
    return hi, lo


def _true_log_marginal(hi, v, n=2 ** 16, seed=3):
    """``log int exp(L(q)) dq`` over the q cube (uniform q is the prior)."""
    Q = np.clip(qmc.Sobol(4, scramble=True, seed=seed).random(n), 1e-12,
                1 - 1e-12)
    with mx.stream(mx.cpu):
        L = np.array(hi.loglik_grid(v, LH.q_to_x(Q)))
    return float(np.log(np.mean(np.exp(L - L.max()))) + L.max())


def _collapsed_minus_prior(lp, V, dtype, stream):
    with mx.stream(stream):
        vm = mx.array(np.atleast_2d(V), dtype=dtype)
        return (np.array(lp(vm), dtype=np.float64)
                - np.array(lp.log_prior(lp.unpack(vm)), dtype=np.float64))


def test_the_prior_normaliser_is_exact():
    """``p(x) = 4! prod(1/F*) (sum omega/F*)^-5`` integrates to one."""
    U = qmc.Sobol(5, scramble=True, seed=1).random(2 ** 20)
    E = -np.log(U)
    X = (E / E.sum(1, keepdims=True))[:, 1:]
    lp = LH.log_prior_x_np(X)
    integral = np.log(np.mean(np.exp(lp - lp.max()))) + lp.max() - math.log(24)
    assert abs(integral + LH.LOG_PRIOR_NORM) < 2e-3, integral


def test_collapsed_value_is_the_true_marginal_on_sampled_modes_scale(weak):
    """Absolute, not up to a constant: at the truth, the collapsed value minus
    the theta prior is ``log int exp(L) p dx``, so the loglike column means
    the same in both modes (it was 0.7685 high before the normaliser). Away
    from the posterior, where the conditional is narrow and off the model's
    axis, the 1,024-point GHK rule errs by up to ~2 nats (0.004-0.08 at
    4,096): measured, documented, not tested here."""
    hi, _ = weak
    v = np.array([0.0, 0.0, 0.095, 0.30, H4.T14_T])
    got = _collapsed_minus_prior(hi, v, mx.float64, mx.cpu)[0]
    assert abs(got - _true_log_marginal(hi, v)) < 0.02


def test_no_transit_gives_the_flat_answer_in_both_precisions(weak):
    """k -> 0: the likelihood is flat in the limb darkening, so the marginal
    is L itself. Without the absolute ridge the curvature vanishes and GHK
    returned thousands of nats too much (KOI-2650.01's false peak)."""
    hi, lo = weak
    V = np.array([[0.0, 0.0, k, 0.3, H4.T14_T] for k in (0.0, 1e-5, 1e-3)])
    with mx.stream(mx.cpu):
        flat = np.array([float(hi.loglik_grid(v, np.full((1, 4), 0.2))[0])
                         for v in V])
    f64 = _collapsed_minus_prior(hi, V, mx.float64, mx.cpu)
    f32 = _collapsed_minus_prior(lo, V, mx.float32, mx.gpu)
    np.testing.assert_allclose(f64, flat, atol=0.03)     # GHK's rule error
    np.testing.assert_allclose(f32, f64, atol=0.05)


def test_float32_tracks_float64_across_a_ball_reaching_no_transit(weak):
    hi, lo = weak
    rng = np.random.default_rng(2)
    V = np.column_stack([np.zeros((48, 2)), rng.uniform(0.0, 0.12, 48),
                         rng.uniform(0.0, 0.95, 48),
                         rng.uniform(0.6, 1.2, 48) * H4.T14_T])
    f64 = _collapsed_minus_prior(hi, V, mx.float64, mx.cpu)
    f32 = _collapsed_minus_prior(lo, V, mx.float32, mx.gpu)
    assert np.all(np.isfinite(f32))
    assert np.max(np.abs(f32 - f64)) < 0.05, np.max(np.abs(f32 - f64))


def _box_reference(hi, v):
    """Exact conditional of q: SNIS of the true L times the prior, uniform
    over x's bounding box, then twice over a box (mean +/- 8 sd) around
    that; no turin sampling. In x, where the conditional is near-Gaussian,
    not in q: under the curvature basis (0.1.67) q1 = depth^4 gives a thin,
    long-tailed q1 that a q box cut short (q1's sd 0.0076 against 0.017 at
    k 0.095, b 0.7), with an ESS of 226 where x's is 1,700."""
    S = qmc.Sobol(4, scramble=True, seed=9).random(2 ** 18)

    def snis(lo_b, hi_b):
        X = lo_b + S * (hi_b - lo_b)
        lp = LH.log_prior_x_np(X)
        X, lp = X[np.isfinite(lp)], lp[np.isfinite(lp)]
        with mx.stream(mx.cpu):
            L = np.array(hi.loglik_grid(v, X))
        w = np.exp(L + lp - np.max(L + lp))
        w /= w.sum()
        xm = w @ X
        Q = LH.x_to_q(X)
        mu = w @ Q
        return (xm, np.sqrt(w @ (X - xm) ** 2), mu,
                np.sqrt(w @ (Q - mu) ** 2), 1.0 / np.sum(w ** 2))

    xm, xs, *_ = snis(np.zeros(4), np.ones(4))
    for _ in range(2):
        s = np.maximum(xs, 1e-3)
        xm, xs, mu, sd, ess = snis(np.clip(xm - 8 * s, 0, 1),
                                   np.clip(xm + 8 * s, 0, 1))
    return mu, sd, ess


def _vertex_reference(hi, v):
    """The conditional piled into the vertex omega_0 = 1: SNIS from
    independent exponentials at the gradient's rates (its own shape)."""
    with mx.stream(mx.cpu):
        x, L, g, R = hi.ld_terms(mx.array(v[None], dtype=mx.float64))
    assert np.allclose(np.array(x)[0], 0.0)
    rate = np.maximum(-np.array(g, dtype=np.float64)[0], 1.0)
    X = np.random.default_rng(5).exponential(1.0 / rate, size=(2 ** 16, 4))
    with mx.stream(mx.cpu):
        Lt = np.array(hi.loglik_grid(v, X))
    lw = np.where(X.sum(1) < 1, Lt + LH.log_prior_x_np(X)
                  - np.sum(np.log(rate) - rate * X, 1), -np.inf)
    w = np.exp(lw - lw.max())
    w /= w.sum()
    Q = LH.x_to_q(X)
    mu = w @ Q
    return mu, np.sqrt(w @ (Q - mu) ** 2), 1.0 / np.sum(w ** 2)


@pytest.mark.parametrize("v, ref", [
    ([0.0, 0.0, 0.095, 0.30, H4.T14_T], _box_reference),   # the truth
    ([0.0, 0.0, 0.06, 0.70, 0.105], _box_reference),       # narrow, off-axis
    ([0.0, 0.0, 0.095, 0.70, 0.105], _box_reference),
    ([0.0, 0.0, 0.095, 0.90, 0.175], _vertex_reference),   # a corner
])
def test_conditional_draws_where_the_model_fits_loosely(weak, v, ref):
    """The independence sampler through GHK draws (<= 0.1.63) failed here by
    up to 23 sd at 32 proposals and 5 sd at 2,048; GHK's sequential weights
    degenerate when the conditional is narrow and away from the model's
    Gaussian. Every case within 0.1 conditional sd and widths within 10%."""
    hi, lo = weak
    v = np.array(v)
    mu, sd, ess = ref(hi, v)
    assert ess > 300
    n = 2048
    q, acc, _ = LH.draw_limb_darkening(lo, np.tile(v, (n, 1)), seed=4)
    assert np.all((q > 0) & (q < 1))                         # no wall atoms
    err = np.abs(q.mean(0) - mu) / sd
    assert np.all(err < 0.1), err
    np.testing.assert_allclose(q.std(0) / sd, 1.0, atol=0.1)
    assert acc > 0.5


def test_the_collapsed_machinery_imports_without_the_data_layer():
    """ldhybrid, simplexmass, likelihood and mle load without lightkurve or
    astropy (``prep`` imports the data layer on first use): other packages
    import these pieces on their own data (WASP103b_CHEOPS)."""
    import subprocess

    code = ("import sys, turin.ldhybrid, turin.simplexmass, turin.likelihood, "
            "turin.mle; bad = [m for m in ('lightkurve', 'astropy') "
            "if m in sys.modules]; assert not bad, bad")
    subprocess.run([sys.executable, "-c", code], check=True)
