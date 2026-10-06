"""Collapsed limb darkening (``turin.ldmarg``) against brute force.

The scheme replaces sampling ``(q1, q2)`` with an integral, so every piece is
checked against something that does not share its approximations:

- the conditional log-density at a given ``(q1, q2)`` against turin's own
  sampled-mode density at the same vector;
- the prior on ``x`` against a histogram of uniform ``(q1, q2)`` pushed
  through the map;
- the expansion point against a dense triangle lattice;
- the collapsed (marginal) value against brute-force integration of
  ``exp(L(x)) p(x)`` over that lattice, with the true ``L``, not the
  quadratic model;
- the conditional draws against the exact lattice conditional, and against
  the reference's per-row sampler;
- the detached expansion point against finite differences, where finite
  differences are valid (no exposure contact rule).

Three synthetic targets: a moderately constrained limb darkening, a weak one
whose expansion point sits on a triangle edge, and one generated at the
``q = (1, 0)`` corner so the conditional piles against the box boundary --
the case where the first version of the conditional sampler left 5% of
draws exactly on the edge. Tolerances are from measurement, with margin;
each test says what it measured.
"""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np
import pytest
from scipy.special import logsumexp

from turin import ldmarg as LD
from turin import likelihood, model as M, params, prep

P_REF, EPOCH, DUR_H = 9.3456, 120.5, 4.2
T14_T = DUR_H / 24.0
EPH = {"period": P_REF, "epoch": EPOCH, "duration": DUR_H, "depth": 6000.0}
THETA = np.array([0.0, 0.0, 0.08, 0.35 / 1.08, T14_T])     # (dP, dtau0, k, beta, T14)
EXP = 29.4 / 1440

CASES = {
    "moderate": dict(),
    "weak": dict(yerr=1e-3),                  # x* on the x2 = 0 edge
    "corner": dict(q1=1.0, q2=0.0),           # conditional against q1=1, q2=0
    "grazing": dict(b=0.95),                  # b > 1 - k = 0.92: no inner contacts
}
#: the parameter vector each case is evaluated at (its own truth)
CASE_THETA = {name: THETA.copy() for name in CASES}
CASE_THETA["grazing"][3] = 0.95 / 1.08      # beta = b / (1 + k)


def _dataset(k=0.08, b=0.35, q1=0.30, q2=0.225, yerr=4e-4, seed=11, n_per=6):
    rng = np.random.default_rng(seed)
    t = np.arange(100.0, 100.0 + n_per * P_REF, EXP)
    tw, fw, ew = prep.extract_near_transit_data(
        t, np.ones_like(t), np.full(t.size, yerr), P_REF, EPOCH, DUR_H)
    ed = prep.segment_epochs(tw, fw, ew, P_REF, EPOCH, DUR_H)
    cen = prep.centering_constants(ed, EPH)
    grid = M.build_grid(cen, prep.supersample_offsets(EXP, 1),
                        dtype=mx.float64, exp_time=EXP)
    c = lambda v: mx.array([[float(v)]], dtype=mx.float64)
    with mx.stream(mx.cpu):
        f = np.array(M.transit_flux(
            grid, mid=mx.zeros((1, ed["n_epochs"]), dtype=mx.float64),
            k=c(k), b=c(b), T14=c(T14_T), q1=c(q1), q2=c(q2),
            period=c(P_REF)), dtype=np.float64)[0]
    x = (ed["times_padded"] - ed["epoch_centers"][:, None]) / ed["half_window"]
    obs = (f * (1 + 2e-4 * x - 1e-4 * (1.5 * x ** 2 - 0.5))
           + rng.normal(0, yerr, f.shape))
    ed = dict(ed)
    ed["flux_padded"] = np.where(ed["mask"] > 0, obs, 1.0)
    return ed, cen, np.full(ed["n_epochs"], 2)


def _targets(case):
    ed, cen, orders = _dataset(**CASES[case])
    kw = dict(num_resample=1, exposure_time=EXP, n_chains_hint=64)
    sampled = likelihood.build_target(params.lineph_layout(EPH), cen, ed,
                                      orders, **kw)
    collapsed = likelihood.build_target(
        params.lineph_layout(EPH, ld="collapsed"), cen, ed, orders, **kw)
    return dict(data=(ed, cen, orders), sampled=sampled, collapsed=collapsed)


@pytest.fixture(scope="module")
def targets():
    return {case: _targets(case) for case in CASES}


def _lattice(n):
    """Triangle lattice ``i + j <= n`` with trapezoid weights."""
    i, j = np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing="ij")
    keep = i + j <= n
    X = np.stack([i[keep], j[keep]], -1) / n
    w = np.ones(len(X))
    edge = (X[:, 0] == 0) | (X[:, 1] == 0) | np.isclose(X.sum(1), 1)
    w[edge] = 0.5
    corner = ((X == 0).all(1) | np.isclose(X, [1, 0]).all(1)
              | np.isclose(X, [0, 1]).all(1))
    w[corner] = 1.0 / 6.0
    return X, w, 1.0 / n


def _L_on(hi, theta, X, chunk=4096):
    with mx.stream(mx.cpu):
        return np.concatenate([
            np.array(hi.loglik_grid(theta, X[s:s + chunk]), dtype=np.float64)
            for s in range(0, len(X), chunk)])


def _ld_terms64(hi, theta):
    with mx.stream(mx.cpu):
        v = mx.array(theta[None], dtype=mx.float64)
        return [np.array(a, dtype=np.float64) for a in hi.ld_terms(v)]


# ---------------------------------------------------------------- maps, prior

def test_q_omega_maps_round_trip():
    rng = np.random.default_rng(0)
    q = rng.uniform(0.01, 0.99, size=(1000, 2))
    om = LD.q_to_omega(q[:, 0], q[:, 1])
    assert np.allclose(om.sum(-1), 1.0) and np.all(om > 0)
    back = np.stack(LD.omega_to_q(om), -1)
    np.testing.assert_allclose(back, q, rtol=0, atol=1e-12)
    np.testing.assert_allclose(np.stack(LD.x_to_q(om[:, 1:]), -1), q,
                               rtol=0, atol=1e-12)


def test_prior_on_x_is_the_pushforward_of_uniform_q():
    """The whole equivalence with sampled mode rests on this density.

    Measured: histogram/pdf ratio mean 1.0002, sd 0.013 over 253 interior
    bins with 2e6 draws (Poisson noise alone is ~0.01 there)."""
    rng = np.random.default_rng(0)
    q = rng.uniform(size=(2_000_000, 2))
    x = LD.q_to_omega(q[:, 0], q[:, 1])[:, 1:]
    nb = 24
    hist, xe, ye = np.histogram2d(x[:, 0], x[:, 1], bins=nb,
                                  range=[[0, 1], [0, 1]], density=True)
    xc, yc = 0.5 * (xe[1:] + xe[:-1]), 0.5 * (ye[1:] + ye[:-1])
    XX, YY = np.meshgrid(xc, yc, indexing="ij")
    inside = XX + YY < 1 - 1.5 / nb          # bins wholly inside the triangle
    X, w, h = _lattice(600)
    Z = np.sum(w * np.exp(LD.log_prior_x_np(X))) * h * h
    pdf = np.exp(LD.log_prior_x_np(np.stack([XX, YY], -1))) / Z
    ratio = hist[inside] / pdf[inside]
    assert abs(ratio.mean() - 1) < 0.005, ratio.mean()
    assert ratio.std() < 0.03, ratio.std()


# ------------------------------------------------- the conditional density

def test_conditional_log_density_matches_the_sampled_one(targets):
    """At a given ``(q1, q2)``, ``L(x(q)) + prior`` is turin's sampled-mode
    density at the 7-vector. Measured: <= 5e-14 in float64 (the brief's 2e-5
    was a looser measurement of the same identity)."""
    t = targets["moderate"]
    hi_s, hi_c = t["sampled"][3], t["collapsed"][3]
    with mx.stream(mx.cpu):
        v5 = mx.array(THETA[None], dtype=mx.float64)
        prior = float(np.array(hi_c.log_prior(hi_c.unpack(v5)))[0])
        for q1, q2 in ((0.30, 0.225), (0.6, 0.4), (0.05, 0.9), (0.97, 0.03)):
            x = LD.q_to_omega(q1, q2)[None, 1:]
            cond = float(np.array(hi_c.loglik_at(v5, x))[0]) + prior
            v7 = mx.array(np.r_[THETA, q1, q2][None], dtype=mx.float64)
            samp = float(np.array(hi_s(v7))[0])
            assert abs(cond - samp) < 1e-10, (q1, q2, cond, samp)


@pytest.mark.parametrize("case", list(CASES))
def test_expansion_point_beats_a_triangle_lattice(targets, case):
    """One Newton step plus the exact QP projection must land at least as high
    as every point of a 45,451-point lattice. Measured margins +1e-5 to
    +3e-3, all positive, including the edge-active case."""
    hi = targets[case]["collapsed"][3]
    theta = CASE_THETA[case]
    x1, x2, Ls, _, _ = _ld_terms64(hi, theta)
    X, _, _ = _lattice(300)
    assert Ls[0] >= _L_on(hi, theta, X).max() - 1e-9, case
    assert 0 <= x1[0] <= 1 and 0 <= x2[0] <= 1 and x1[0] + x2[0] <= 1 + 1e-12


@pytest.mark.parametrize("case", list(CASES))
def test_collapsed_value_matches_brute_force_integration(targets, case):
    """The marginal against ``log sum exp(L(x)) p(x)`` on a lattice, using the
    true ``L`` rather than its quadratic model. Measured at n = 300: 3e-5 to
    4e-5, converging to the non-quadratic residual (the brief: 4e-3).

    The lattice must resolve the narrow direction of the omega posterior, or
    the brute force is the inaccurate side; that is asserted, not assumed.
    """
    hi = targets[case]["collapsed"][3]
    theta = CASE_THETA[case]
    _, _, _, _, H = _ld_terms64(hi, theta)
    a, b, c = H[0]
    sigma_min = 1.0 / math.sqrt(max(np.linalg.eigvalsh([[a, b], [b, c]])))
    X, w, h = _lattice(300)
    assert h < sigma_min / 5, (case, h, sigma_min)
    brute = (logsumexp(_L_on(hi, theta, X) + LD.log_prior_x_np(X), b=w)
             + math.log(h * h))
    with mx.stream(mx.cpu):
        v5 = mx.array(theta[None], dtype=mx.float64)
        coll = float(np.array(hi(v5))[0]
                     - np.array(hi.log_prior(hi.unpack(v5)))[0])
    assert abs(coll - brute) < 2e-4, (case, coll - brute)


# ----------------------------------------------------- derivatives, blocking

def test_detached_expansion_point_gradient_matches_finite_differences():
    """``x*`` is detached before the final pass; the gradient must still be
    the derivative of the collapsed value. At ``exp_time = 0`` -- under the
    contact rule a finite difference re-places the split points and measures
    something else (see CLAUDE.md). Measured: <= 6e-6 of max|grad|."""
    ed, cen, orders = _dataset()
    lay = params.lineph_layout(EPH, ld="collapsed")
    theta = THETA + np.array([1e-4, -5e-4, 0, 0, 0])
    steps = (1e-6, 1e-6, 1e-7, 1e-6, 1e-7)
    with mx.stream(mx.cpu):
        hi = LD.MarginalLDLogProb(lay, cen, ed, orders, exposure_time=0.0,
                                  num_resample=1, dtype=mx.float64)
        f = lambda v: float(np.array(hi(mx.array(v[None], dtype=mx.float64)))[0])
        g = np.array(mx.grad(lambda v: mx.sum(hi(v)))(
            mx.array(theta[None], dtype=mx.float64)))[0]
        fd = np.array([(f(theta + h * e) - f(theta - h * e)) / (2 * h)
                       for e, h in zip(np.eye(5), steps)])
    assert np.all(np.isfinite(g))
    assert np.max(np.abs(g - fd)) / np.max(np.abs(fd)) < 2e-5, (g, fd)


def test_blocking_does_not_change_the_collapsed_density():
    """L, g and H are sums over epochs and the ridge is applied after the
    sum. Measured, 1 block against 6: value identical, gradient 7e-16."""
    ed, cen, orders = _dataset()
    lay = params.lineph_layout(EPH, ld="collapsed")
    kw = dict(num_resample=1, exposure_time=EXP, n_chains_hint=64,
              dtype=mx.float64)
    rng = np.random.default_rng(0)
    V = THETA[None] + 1e-3 * rng.standard_normal((4, 5)) * np.r_[1e-3, 1e-3, 1, 1, 1]
    with mx.stream(mx.cpu):
        one = LD.MarginalLDLogProb(lay, cen, ed, orders, **kw)
        many = LD.MarginalLDLogProb(lay, cen, ed, orders, budget_bytes=1, **kw)
        assert len(one.blocks) == 1 and len(many.blocks) == ed["n_epochs"]
        Vm = mx.array(V, dtype=mx.float64)
        v1, g1 = mx.value_and_grad(lambda v: mx.sum(one(v)))(Vm)
        vm, gm = mx.value_and_grad(lambda v: mx.sum(many(v)))(Vm)
        assert abs(float(v1) - float(vm)) <= 1e-10 * abs(float(v1)) + 1e-10
        assert float(mx.max(mx.abs(g1 - gm)) / mx.max(mx.abs(g1))) < 1e-10


def test_collapsed_mode_sizes_blocks_for_its_larger_footprint():
    """Same budget, same dtype: the collapsed class takes ~2.5x fewer epochs
    per block (measured footprint 271 against 121 B per chain-point)."""
    assert LD.COLLAPSED_BYTES_PER_POINT == int(2.5 * likelihood.BYTES_PER_POINT)
    ed, cen, orders = _dataset(n_per=14)
    n_ep, max_pts = ed["n_epochs"], ed["mask"].shape[1]
    budget = 10 * 64 * max_pts * likelihood.BYTES_PER_POINT   # 10 default epochs
    kw = dict(n_chains_hint=64, budget_bytes=budget, exposure_time=EXP)
    default = likelihood.ProfiledTransitLogProb(
        params.lineph_layout(EPH), cen, ed, orders, **kw)
    collapsed = LD.MarginalLDLogProb(
        params.lineph_layout(EPH, ld="collapsed"), cen, ed, orders, **kw)
    assert n_ep >= 10 and default.block_size == 10
    assert collapsed.block_size == 4                # 10 / 2.5


def test_fp32_collapsed_tracks_fp64_well_inside_the_probe_bar(targets):
    """sd of float32 - float64 over a u-ball, against the --PL probe's 0.02
    floor. Measured: 2e-4 to 3e-4."""
    target, transform, lo, hi = targets["moderate"]["collapsed"]
    u0 = transform.from_model_np(THETA)
    U = u0[None] + 1e-2 * np.random.default_rng(1).standard_normal((64, 5))
    V = transform.model_np(U)
    d32 = np.array(lo(mx.array(V, dtype=mx.float32)), dtype=np.float64)
    with mx.stream(mx.cpu):
        d64 = np.array(hi(mx.array(V, dtype=mx.float64)), dtype=np.float64)
    assert (d32 - d64).std() < 0.02 and np.all(np.isfinite(d32))


# ------------------------------------------------------- conditional draws

def _exact_conditional(hi, theta, n=300):
    X, w, _ = _lattice(n)
    Xc = np.clip(X, 1e-12, 1 - 1e-12)
    lw = _L_on(hi, theta, X) + LD.log_prior_x_np(Xc) + np.log(w)
    p = np.exp(lw - lw.max())
    p /= p.sum()
    return LD.x_to_q(Xc), p


@pytest.mark.parametrize("case", list(CASES))
def test_conditional_draws_reproduce_the_exact_conditional(targets, case):
    """4,096 draws at one theta against the lattice conditional with the true
    ``L``. Measured: means within 0.8 standard errors, sds within 2%, TV on
    40 bins 0.02-0.04, acceptance 0.94-0.97."""
    lo, hi = targets[case]["collapsed"][2], targets[case]["collapsed"][3]
    theta = CASE_THETA[case]
    C = 4096
    q1, q2, acc, _ = LD.draw_limb_darkening(lo, np.tile(theta, (C, 1)), seed=3)
    (q1l, q2l), p = _exact_conditional(hi, theta)
    edges = np.linspace(0, 1, 41)
    for got, ref in ((q1, q1l), (q2, q2l)):
        mu = np.sum(p * ref)
        sd = math.sqrt(np.sum(p * (ref - mu) ** 2))
        assert abs(got.mean() - mu) < 4 * sd / math.sqrt(C), (case, got.mean(), mu)
        assert abs(got.std() / sd - 1) < 0.10, (case, got.std(), sd)
        tv = 0.5 * np.abs(np.histogram(got, edges)[0] / C
                          - np.histogram(ref, edges, weights=p)[0]).sum()
        assert tv < 0.08, (case, tv)
    assert acc > 0.5, acc


def test_no_conditional_draw_lies_on_the_box_edge(targets):
    """The edge-atom regression. With the conditional piled against the
    ``q1 = 1, q2 = 0`` corner, draws come within ~1e-4 of the boundary but
    must never sit on it: that is a measure-zero event for a correct
    sampler, and the first version put 5% of draws there."""
    lo, hi = targets["corner"]["collapsed"][2], targets["corner"]["collapsed"][3]
    C = 8192
    rng = np.random.default_rng(0)
    thetas = THETA[None] + rng.standard_normal((C, 5)) * np.r_[0, 0, 1e-3, 1e-2, 1e-4]
    sampler = LD.OmegaSampler(lo, seed=4)
    x, acc, _ = sampler.step(thetas)
    q1, q2 = LD.x_to_q(x)
    assert np.all((x[:, 0] > 0) & (x[:, 1] > 0) & (x.sum(1) < 1))
    assert np.all((q1 > 0) & (q1 < 1) & (q2 > 0) & (q2 < 1))
    assert q1.max() > 0.99 and q2.min() < 0.01     # the corner was reached
    x1s, x2s, _, _, _ = (np.array(a, np.float64) for a in
                         lo.ld_terms(mx.array(thetas[:64], dtype=mx.float32)))
    assert not np.any((x[:64, 0] == x1s) & (x[:64, 1] == x2s))   # never x*


# --------------------------------- the vectorised sampler vs the per-row one

def _sample_rows_loop(sampler, T, xh):
    """The reference implementation's per-row proposal draw, verbatim in
    behaviour (SquishierPlanet's ``OmegaGibbs._sample``), for pinning."""
    from scipy.special import ndtr, ndtri

    l11, l21, l22 = T["l"]
    C = xh.shape[0]
    grid, m, cdf = T["grid"], T["m"], T["cdf"]
    z1 = np.empty(C)
    for i in range(C):
        u = sampler.rng.uniform() * cdf[i, -1]
        j = np.clip(np.searchsorted(cdf[i], u) - 1, 0, sampler.n_tab - 2)
        g0, g1 = grid[i, j], grid[i, j + 1]
        m0, m1 = m[i, j], m[i, j + 1]
        h = g1 - g0
        r = u - cdf[i, j]
        slope = (m1 - m0) / h if h > 0 else 0.0
        if abs(slope) < 1e-14 * max(m0, 1e-300):
            t = r / max(m0, 1e-300)
        else:
            t = (-m0 + math.sqrt(max(m0 * m0 + 2 * slope * r, 0.0))) / slope
        z1[i] = g0 + min(max(t, 0.0), h)
    ylo, yhi = sampler._ybounds(T["Zv"], z1[:, None])
    a, b = ndtr(ylo[:, 0]), ndtr(yhi[:, 0])
    z2 = ndtri(np.clip(a + sampler.rng.uniform(size=C) * (b - a),
                       1e-300, 1 - 1e-16))
    dx2 = z2 / l22
    dx1 = (z1 - l21 * dx2) / l11
    x = np.clip(xh + np.stack([dx1, dx2], -1), 0.0, 1.0)
    return x / np.maximum(1.0, x.sum(-1, keepdims=True))


def _logq_rows_loop(sampler, T, xh, x):
    from scipy.special import ndtr

    l11, l21, l22 = T["l"]
    dx = x - xh
    z1 = l11 * dx[:, 0] + l21 * dx[:, 1]
    z2 = l22 * dx[:, 1]
    grid, m, cdf = T["grid"], T["m"], T["cdf"]
    out = np.full(x.shape[0], -np.inf)
    tol = 1e-9 * (1.0 + LD.Z_WINDOW)
    for i in range(x.shape[0]):
        g = grid[i]
        if not (g[0] - tol <= z1[i] <= g[-1] + tol) or cdf[i, -1] <= 0:
            continue
        zi = min(max(z1[i], g[0]), g[-1])
        mz = np.interp(zi, g, m[i]) / cdf[i, -1]
        ylo, yhi = sampler._ybounds(T["Zv"][i:i + 1], np.array([[zi]]))
        ylo, yhi = ylo[0, 0], yhi[0, 0]
        if not (ylo - tol <= z2[i] <= yhi + tol):
            continue
        mass = ndtr(yhi) - ndtr(ylo)
        if mass <= 0:
            continue
        cz2 = math.exp(-0.5 * z2[i] ** 2) / math.sqrt(2 * math.pi) / mass
        out[i] = math.log(mz * cz2) + math.log(l11[i] * l22[i])
    return out


@pytest.mark.parametrize("case", ["moderate", "corner"])
def test_vectorised_sampler_matches_the_per_row_reference(targets, case):
    lo = targets[case]["collapsed"][2]
    rng = np.random.default_rng(1)
    thetas = THETA[None] + rng.standard_normal((256, 5)) * np.r_[0, 0, 1e-3, 1e-2, 1e-4]
    x1, x2, _, g, H = (np.array(a, np.float64) for a in
                       lo.ld_terms(mx.array(thetas, dtype=mx.float32)))
    det = H[:, 0] * H[:, 2] - H[:, 1] ** 2
    xh = np.stack([x1, x2], -1) + np.stack(
        [(H[:, 2] * g[:, 0] - H[:, 1] * g[:, 1]) / det,
         (H[:, 0] * g[:, 1] - H[:, 1] * g[:, 0]) / det], -1)
    vec = LD.OmegaSampler(lo, seed=9)
    ref = LD.OmegaSampler(lo, seed=9)
    T = vec._tables(xh, H)
    xv = vec._sample(T, xh)
    xr = _sample_rows_loop(ref, T, xh)
    np.testing.assert_allclose(xv, xr, rtol=0, atol=1e-12)
    np.testing.assert_allclose(vec._logq(T, xh, xv), _logq_rows_loop(ref, T, xh, xv),
                               rtol=1e-10, atol=1e-10)
    # and at points the tables were not built from, including outside the
    # support (both must give -inf there)
    probe = np.vstack([xv[:64] * 0.97 + 0.01, [[1.2, 0.1], [-0.1, 0.5]]])
    T66 = {k: (tuple(a[:66] for a in v) if k == "l" else v[:66])
           for k, v in T.items()}
    lq_v = vec._logq(T66, xh[:66], probe)
    lq_r = _logq_rows_loop(ref, T66, xh[:66], probe)
    assert np.all(np.isneginf(lq_v[-2:])) and np.all(np.isneginf(lq_r[-2:]))
    np.testing.assert_allclose(lq_v, lq_r, rtol=1e-10, atol=1e-10)


# ------------------------------------------------------------ the class API

def test_refusals():
    ed, cen, orders = _dataset()
    col = params.lineph_layout(EPH, ld="collapsed")
    with pytest.raises(ValueError, match="envelope theorem"):
        LD.MarginalLDLogProb(col, cen, ed, orders, profile_mode="ratio")
    with pytest.raises(ValueError, match="circular"):
        LD.MarginalLDLogProb(col, cen, ed, orders, geometry="chord")
    with pytest.raises(ValueError, match="ld='collapsed'"):
        LD.MarginalLDLogProb(params.lineph_layout(EPH), cen, ed, orders)
    ttv = params.ttv_layout(EPH, cen, tau_half=0.05)
    with pytest.raises(ValueError, match="LinEph-only"):
        LD.MarginalLDLogProb(ttv, cen, ed, orders)
    with pytest.raises(ValueError, match="ld_mode"):
        likelihood.build_target(params.lineph_layout(EPH), cen, ed, orders,
                                ld_mode="collapsed", fp64=False)
    with pytest.raises(TypeError, match="MarginalLDLogProb"):
        LD.OmegaSampler(likelihood.ProfiledTransitLogProb(
            params.lineph_layout(EPH), cen, ed, orders))
    lp = LD.MarginalLDLogProb(col, cen, ed, orders)
    with pytest.raises(NotImplementedError, match="factorisation"):
        lp.epoch_log_lik(mx.array(THETA[None], dtype=mx.float32))


def test_build_target_dispatches_on_the_layout(targets):
    t = targets["moderate"]
    assert type(t["collapsed"][2]).__name__ == "MarginalLDLogProb"
    assert type(t["collapsed"][3]).__name__ == "MarginalLDLogProb"
    assert type(t["sampled"][2]) is likelihood.ProfiledTransitLogProb
    assert t["collapsed"][1].dim == 5 and t["sampled"][1].dim == 7


def test_ml_row_model_uses_the_conditional_mode(targets):
    """``full_model`` without ``ld=`` draws the single displayed light curve
    at the conditional mode ``x*``; with ``ld=`` it is exactly the sampled
    class's model at that ``(q1, q2)``."""
    t = targets["moderate"]
    lo_c, lo_s = t["collapsed"][2], t["sampled"][2]
    v5 = mx.array(THETA[None], dtype=mx.float32)
    (q1,), (q2,) = lo_c.conditional_mode_ld(v5)
    assert 0 < q1 < 1 and 0 < q2 < 1
    m_c, _ = lo_c.full_model(v5)
    m_s, _ = lo_s.full_model(mx.array(np.r_[THETA, q1, q2][None],
                                      dtype=mx.float32))
    np.testing.assert_allclose(np.array(m_c), np.array(m_s), rtol=0, atol=1e-6)
    with pytest.raises(ValueError, match="one row"):
        lo_c.full_model(mx.array(np.tile(THETA, (2, 1)), dtype=mx.float32))


def test_the_target_compiles(targets):
    """anvil compiles the log-density inside its kernels."""
    target, transform, lo, hi = targets["moderate"]["collapsed"]
    u = mx.array(np.tile(transform.from_model_np(THETA), (8, 1)), dtype=mx.float32)
    f = mx.compile(target.log_prob_and_grad)
    val, grad = f(u)
    plain = target.log_prob(u)
    assert np.allclose(np.array(val), np.array(plain), atol=1e-3)
    assert np.all(np.isfinite(np.array(grad)))
