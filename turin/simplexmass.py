"""Gaussian mass over a probability simplex, for collapsed hybrid limb darkening.

The pieces :mod:`turin.ldhybrid` needs once the limb-darkening weights live on
a (G-1)-simplex rather than turin's quadratic triangle (``ldmarg``):

- small batched linear algebra on the **triangular factor** ``R`` of the
  Gauss-Newton curvature, never on ``H = R^T R`` itself: a two-pass modified
  Gram-Schmidt QR (:func:`mgs_r`), TSQR across epoch blocks, unrolled
  triangular solves. Measured on KOI-448.02's posterior, ``H`` reaches a
  condition number of 7e9, where any float32 solve with it is wrong by an
  order of magnitude; ``R``'s is the square root (<= 8.5e4), which float32
  handles. MLX's ``linalg`` is CPU-only, so everything is elementwise;
- :func:`qp_simplex_r`, the exact constrained optimum of the quadratic model
  over the simplex, by face enumeration on ``R`` (the expansion point);
  :func:`project_simplex`, a Euclidean projection kept as a safe fallback;
- the **GHK** estimator (:func:`ghk_terms`, :func:`ghk_log_mass`): the
  free weights drawn sequentially from tail-stable truncated normals on
  ``[0, 1 - sum(drawn)]``, the exact sequential form of the dependent
  weight's ``>= 0``, on a fixed scrambled-Sobol rule, so the estimate is
  deterministic and differentiable. The prior enters as an exact per-point
  weight. turin's weights are already normalised, so SquishierPlanet's
  extra draw of the normalisation direction is not needed;
- host-side helpers: the frozen layout (:func:`choose_layout`, which weight
  is dependent and the order of the rest), the Sobol rule, and the exact QP
  over the simplex by face enumeration (:func:`qp_simplex_np`), used for
  the single ML display row.
"""

from __future__ import annotations

import itertools
import math

import mlx.core as mx
import numpy as np

from .normtail import trunc_normal

_TWO_LOG_2PI = 2.0 * math.log(2.0 * math.pi)


# ------------------------------------------------- batched linear algebra

def mgs_r(cols):
    """``R`` of a thin QR of the matrix whose columns are ``cols``.

    ``cols``: a list of D arrays ``(C, n)`` (the reduction runs over the last
    axis). Two-pass modified Gram-Schmidt (re-orthogonalisation, for
    float32). A column that is numerically zero after projection (an epoch
    block with no transit in it) gives a zero row and column instead of a
    NaN. Returns ``(C, D, D)``, upper triangular, non-negative diagonal.
    """
    D = len(cols)
    C = cols[0].shape[0]
    dt = cols[0].dtype
    tiny = 1e-300 if dt == mx.float64 else 1e-30
    zero = mx.zeros((C,), dtype=dt)
    qs = []
    rows = [[zero] * D for _ in range(D)]
    for j in range(D):
        v = cols[j]
        r = [zero] * j
        for _pass in range(2):
            for i in range(j):
                c = mx.sum(qs[i] * v, axis=-1)
                v = v - c[:, None] * qs[i]
                r[i] = r[i] + c
        nv = mx.sqrt(mx.sum(v * v, axis=-1))
        live = nv > tiny
        qs.append(mx.where(live[:, None], v / mx.maximum(nv, tiny)[:, None],
                           mx.zeros_like(v)))
        for i in range(j):
            rows[i][j] = r[i]
        rows[j][j] = nv
    return mx.stack([mx.stack(rows[i], axis=-1) for i in range(D)], axis=-2)


def tsqr(Rs, ridge_rel=0.0, ridge_abs=0.0):
    """Combine per-block factors ``Rs`` (list of ``(C, D, D)``) into one,
    adding ``eps = ridge_rel * trace(R^T R) + ridge_abs`` to the curvature
    (as rows ``sqrt(eps) I``): the factor of ``sum_b R_b^T R_b + eps I``."""
    D = Rs[0].shape[-1]
    trace = sum(mx.sum(R * R, axis=(-2, -1)) for R in Rs)
    s = mx.sqrt(ridge_rel * trace + ridge_abs)             # (C,)
    eye = mx.eye(D, dtype=Rs[0].dtype)
    cols = []
    for j in range(D):
        parts = [R[:, :, j] for R in Rs] + [s[:, None] * eye[None, j, :]]
        cols.append(mx.concatenate(parts, axis=-1))
    return mgs_r(cols)


def solve_upper(R, b):
    """``x`` with ``R x = b``; ``R (C, D, D)`` upper, ``b (C, D)``."""
    D = R.shape[-1]
    x = [None] * D
    for i in range(D - 1, -1, -1):
        s = b[:, i]
        for k in range(i + 1, D):
            s = s - R[:, i, k] * x[k]
        x[i] = s / R[:, i, i]
    return mx.stack(x, axis=-1)


def solve_upper_t(R, b):
    """``y`` with ``R^T y = b`` (forward substitution)."""
    D = R.shape[-1]
    y = [None] * D
    for i in range(D):
        s = b[:, i]
        for k in range(i):
            s = s - R[:, k, i] * y[k]
        y[i] = s / R[:, i, i]
    return mx.stack(y, axis=-1)


def project_simplex(om):
    """Euclidean projection of ``om (C, G)`` onto the probability simplex."""
    G = om.shape[-1]
    u = -mx.sort(-om, axis=-1)
    css = mx.cumsum(u, axis=-1) - 1.0
    k = mx.arange(1, G + 1).astype(om.dtype)
    cond = (u - css / k) > 0
    rho = mx.sum(cond.astype(mx.int32), axis=-1, keepdims=True) - 1
    rho = mx.maximum(rho, 0)
    theta = mx.take_along_axis(css, rho, axis=-1) / (rho.astype(om.dtype) + 1.0)
    return mx.maximum(om - theta, 0.0)


def _lstsq(cols, b):
    """Least squares ``min |A z - b|`` for ``A``'s columns ``cols`` (each
    ``(C, m)``), by QR; ``(z (C, k), residual^2 (C,))``. A column with no
    independent part gives a non-finite ``z``, which callers treat as an
    infeasible face."""
    k = len(cols)
    if k == 0:
        return None, mx.sum(b * b, axis=-1)
    Rf = mgs_r(cols)
    # Q^T b without forming Q: solve R^T (Q^T b) = A^T b
    atb = mx.stack([mx.sum(c * b, axis=-1) for c in cols], axis=-1)
    qtb = solve_upper_t(Rf, atb)
    z = solve_upper(Rf, qtb)
    res = mx.sum(b * b, axis=-1) - mx.sum(qtb * qtb, axis=-1)
    return z, res


_FACE_SETS = {}


def _face_sets(G):
    if G not in _FACE_SETS:
        _FACE_SETS[G] = [S for r in range(1, G + 1)
                         for S in itertools.combinations(range(G), r)]
    return _FACE_SETS[G]


def qp_simplex_r(R, b):
    """``argmin |R x - b|^2`` over ``{x >= 0, sum x <= 1}``, exact and
    batched: every support of ``omega = (1 - sum x, x)`` is solved as a small
    least-squares problem on ``R``'s columns (by QR, so the conditioning is
    ``R``'s, the square root of the curvature's), and the best feasible one
    kept. ``R (C, D, D)``, ``b (C, D)`` -> ``x (C, D)``. Not differentiable
    (the caller detaches it).

    float32 on the GPU runs one Metal kernel, one thread per chain
    (:func:`_qp_simplex_r_metal`); everything else -- float64, the CPU
    stream -- the MLX graph (:func:`_qp_simplex_r_graph`), the reference.
    The graph is thousands of tiny operations (faces x columns x passes) and
    cost 1.5-2.3 sampled-hybrid4 evaluations per collapsed evaluation
    (0.1.94); the same steps in one thread agree to float32 rounding."""
    if (R.dtype == mx.float32 and _gpu_stream_active()
            and mx.metal.is_available()):
        return _qp_simplex_r_metal(mx.stop_gradient(R), mx.stop_gradient(b))
    return _qp_simplex_r_graph(R, b)


def _gpu_stream_active():
    try:
        return mx.default_device() == mx.Device(mx.DeviceType.gpu)
    except Exception:
        return False


_QP_KERNELS = {}

_QP_SOURCE = """
    uint c = thread_position_in_grid.x;
    if (c >= (uint)R_shape[0]) return;
    const float tol = 1e-7f;
    const float tiny = 1e-30f;
    float Rm[D][D];
    float bv[D];
    for (int i = 0; i < D; ++i) {
        bv[i] = b[c * D + i];
        for (int j = 0; j < D; ++j) Rm[i][j] = R[(c * D + i) * D + j];
    }
    float best_x[D];
    for (int j = 0; j < D; ++j) best_x[j] = 1.0f / (float)(D + 1);
    float best_q = INFINITY;
    for (int f = 0; f < NF; ++f) {
        const int m = FACES[f];
        const bool has0 = (m & 1) != 0;
        int idx[D];
        int k = 0;
        for (int g = 1; g <= D; ++g) if (m & (1 << g)) idx[k++] = g - 1;
        if (!has0 && k == 0) continue;
        int ncol = has0 ? k : k - 1;
        int last = has0 ? -1 : idx[k - 1];
        float A[D][D];
        float rhs[D];
        for (int r = 0; r < D; ++r) {
            float sub = has0 ? 0.0f : Rm[r][last];
            rhs[r] = bv[r] - sub;
            for (int i = 0; i < ncol; ++i) A[r][i] = Rm[r][idx[i]] - sub;
        }
        float z[D];
        if (ncol > 0) {
            /* two-pass modified Gram-Schmidt, as mgs_r */
            float Q[D][D];
            float Rf[D][D];
            for (int i = 0; i < D; ++i) for (int j = 0; j < D; ++j) Rf[i][j] = 0.0f;
            for (int j = 0; j < ncol; ++j) {
                float v[D];
                for (int r = 0; r < D; ++r) v[r] = A[r][j];
                float rj[D];
                for (int i = 0; i < D; ++i) rj[i] = 0.0f;
                for (int pass = 0; pass < 2; ++pass) {
                    for (int i = 0; i < j; ++i) {
                        float cd = 0.0f;
                        for (int r = 0; r < D; ++r) cd += Q[r][i] * v[r];
                        for (int r = 0; r < D; ++r) v[r] = v[r] - cd * Q[r][i];
                        rj[i] = rj[i] + cd;
                    }
                }
                float nn = 0.0f;
                for (int r = 0; r < D; ++r) nn += v[r] * v[r];
                float nv = metal::sqrt(nn);
                bool live = nv > tiny;
                for (int r = 0; r < D; ++r) Q[r][j] = live ? v[r] / metal::max(nv, tiny) : 0.0f;
                for (int i = 0; i < j; ++i) Rf[i][j] = rj[i];
                Rf[j][j] = nv;
            }
            /* R^T y = A^T rhs, then R z = y (as _lstsq) */
            float y[D];
            for (int i = 0; i < ncol; ++i) {
                float atb = 0.0f;
                for (int r = 0; r < D; ++r) atb += A[r][i] * rhs[r];
                float sacc = atb;
                for (int kk = 0; kk < i; ++kk) sacc = sacc - Rf[kk][i] * y[kk];
                y[i] = sacc / Rf[i][i];
            }
            for (int i = ncol - 1; i >= 0; --i) {
                float sacc = y[i];
                for (int kk = i + 1; kk < ncol; ++kk) sacc = sacc - Rf[i][kk] * z[kk];
                z[i] = sacc / Rf[i][i];
            }
        }
        float x[D];
        for (int j = 0; j < D; ++j) x[j] = 0.0f;
        if (has0) {
            for (int i = 0; i < ncol; ++i) x[idx[i]] = z[i];
        } else {
            float tot = 0.0f;
            for (int i = 0; i < ncol; ++i) { x[idx[i]] = z[i]; tot = tot + z[i]; }
            x[last] = 1.0f - tot;
        }
        bool ok = true;
        float sx = 0.0f;
        for (int j = 0; j < D; ++j) {
            ok = ok && metal::isfinite(x[j]) && (x[j] >= -tol);
            sx += x[j];
        }
        ok = ok && (sx <= 1.0f + tol);
        if (!ok) continue;
        float q = 0.0f;
        for (int r = 0; r < D; ++r) {
            float rr = 0.0f;
            for (int j = 0; j < D; ++j) rr += Rm[r][j] * x[j];
            rr = rr - bv[r];
            q += rr * rr;
        }
        if (q < best_q) {
            best_q = q;
            for (int j = 0; j < D; ++j) best_x[j] = x[j];
        }
    }
    float tot = 0.0f;
    for (int j = 0; j < D; ++j) { best_x[j] = metal::clamp(best_x[j], 0.0f, 1.0f); tot += best_x[j]; }
    float den = metal::max(1.0f, tot);
    for (int j = 0; j < D; ++j) out[c * D + j] = best_x[j] / den;
"""


def _qp_simplex_r_metal(R, b):
    """:func:`qp_simplex_r` in one Metal thread per chain (float32): the
    graph's faces in its order, its two-pass MGS and solves, the same
    feasibility tolerance and strict-improvement tie rule."""
    C, D, _ = R.shape
    if D not in _QP_KERNELS:
        masks = [sum(1 << g for g in S) for S in _face_sets(D + 1)]
        header = (f"constant constexpr int D = {D};\n"
                  f"constant constexpr int NF = {len(masks)};\n"
                  f"constant int FACES[{len(masks)}] = "
                  f"{{{', '.join(map(str, masks))}}};\n")
        _QP_KERNELS[D] = mx.fast.metal_kernel(
            name=f"turin_qp_simplex_r_{D}", input_names=["R", "b"],
            output_names=["out"], source=_QP_SOURCE, header=header)
    (out,) = _QP_KERNELS[D](
        inputs=[R, b], grid=(max(C, 1), 1, 1),
        threadgroup=(min(256, max(C, 1)), 1, 1),
        output_shapes=[(C, D)], output_dtypes=[mx.float32])
    return out


def _qp_simplex_r_graph(R, b):
    """:func:`qp_simplex_r` as an MLX graph (any dtype or stream)."""
    C, D, _ = R.shape
    G = D + 1
    dt = R.dtype
    tol = 1e-7 if dt == mx.float32 else 1e-12
    best_x = mx.full((C, D), 1.0 / G, dtype=dt)
    best_q = mx.full((C,), float("inf"), dtype=dt)
    col = [R[:, :, j] for j in range(D)]
    for S in _face_sets(G):
        free = [j - 1 for j in S if j >= 1]
        if 0 in S:
            z, _ = _lstsq([col[j] for j in free], b)
            parts = {}
            if z is not None:
                for i, j in enumerate(free):
                    parts[j] = z[:, i]
        else:
            if not free:
                continue
            last = free[-1]
            rest = free[:-1]
            bb = b - col[last]
            z, _ = _lstsq([col[j] - col[last] for j in rest], bb)
            parts = {}
            tot = mx.zeros((C,), dtype=dt)
            if z is not None:
                for i, j in enumerate(rest):
                    parts[j] = z[:, i]
                    tot = tot + z[:, i]
            parts[last] = 1.0 - tot
        x = mx.stack([parts.get(j, mx.zeros((C,), dtype=dt))
                      for j in range(D)], axis=-1)
        ok = mx.all(mx.isfinite(x), axis=-1)
        ok = ok & mx.all(x >= -tol, axis=-1) & (mx.sum(x, axis=-1) <= 1 + tol)
        x = mx.where(ok[:, None], x, best_x)
        r = mx.einsum("cij,cj->ci", R, x) - b
        q = mx.where(ok, mx.sum(r * r, axis=-1), mx.full((C,), float("inf"),
                                                         dtype=dt))
        better = q < best_q
        best_x = mx.where(better[:, None], x, best_x)
        best_q = mx.where(better, q, best_q)
    best_x = mx.clip(best_x, 0.0, 1.0)
    return best_x / mx.maximum(1.0, mx.sum(best_x, axis=-1, keepdims=True))


# ------------------------------------------------------------------ GHK

def layout_map(dep, order, G):
    """``(T, t)`` with the free weights ``y = omega[order] = T x + t``, for
    ``x = omega[1:]`` and ``omega_0 = 1 - sum(x)``. ``|det T| = 1``."""
    E = np.vstack([-np.ones(G - 1), np.eye(G - 1)])
    e0 = np.r_[1.0, np.zeros(G - 1)]
    order = list(order)
    if dep in order or sorted(order + [dep]) != list(range(G)):
        raise ValueError(f"bad GHK layout dep={dep}, order={order}")
    return E[order], e0[order]


def ghk_terms(my, Ly, U, dep, order, inv_fstar, *, eps_gap=None):
    """Sequential GHK draws and their log-weights.

    ``my (C, D)`` and lower-triangular ``Ly (C, D, D)``: the mean and
    covariance factor of the free weights ``y`` (:func:`layout_map`).
    ``U``: ``(N, D)`` (one rule shared by every chain) or ``(C, N, D)``
    uniforms in (0, 1). ``inv_fstar``: the G generators' ``1/F*``.

    Returns ``(lw (C, N), Y (C, N, D))``: ``lw`` is the sum of the 1-D
    truncation log-masses plus the prior ``-G log sum_g omega_g/F*_g`` at
    the drawn point. Accumulations go through stacked arrays and reductions,
    not chained elementwise adds, which MLX fuses into one Metal kernel and
    can refuse to compile ("too many inputs fused").
    """
    C, D = my.shape
    dt = my.dtype
    eps_gap = eps_gap if eps_gap is not None else (
        1e-9 if dt == mx.float64 else 1e-6)
    shared = U.ndim == 2
    N = U.shape[-2]
    zs, ys, lms = [], [], []
    for k in range(D):
        if k:
            Z = mx.stack(zs, axis=-1)                              # (C, N, k)
            cond = my[:, k:k + 1] + (Z @ Ly[:, k, :k, None])[..., 0]
            ub = 1.0 - mx.sum(mx.stack(ys, axis=-1), axis=-1)
        else:
            cond = mx.broadcast_to(my[:, :1], (C, N))
            ub = mx.ones((C, N), dtype=dt)
        lkk = Ly[:, k, k][:, None]
        lo = (0.0 - cond) / lkk
        hi = mx.maximum((ub - cond) / lkk, lo + eps_gap)
        uk = (mx.broadcast_to(U[None, :, k], (C, N)) if shared
              else U[:, :, k])
        zk, lm = trunc_normal(lo, hi, uk.astype(dt))
        zs.append(zk)
        ys.append(cond + lkk * zk)
        lms.append(lm)
    Y = mx.stack(ys, axis=-1)
    G = D + 1
    inv_dep = float(inv_fstar[dep])
    coef = mx.array([float(inv_fstar[g]) - inv_dep for g in order], dtype=dt)
    s = mx.sum(Y * coef, axis=-1) + inv_dep
    tiny = 1e-300 if dt == mx.float64 else 1e-30
    lprior = -G * mx.log(mx.maximum(s, tiny))
    lw = mx.sum(mx.stack(lms, axis=-1), axis=-1) + lprior
    return lw, Y


def ghk_log_mass(my, Ly, U, dep, order, inv_fstar):
    """``log int_simplex exp(-1/2 (y - my)^T Cy^-1 (y - my)) p(y) dy``,
    ``Cy = Ly Ly^T``: the Gaussian's normaliser times GHK's estimate of the
    prior-weighted probability, ``(C,)``."""
    lw, _ = ghk_terms(my, Ly, U, dep, order, inv_fstar)
    N = U.shape[-2]
    logdet = mx.sum(mx.log(mx.abs(mx.stack(
        [Ly[:, k, k] for k in range(Ly.shape[-1])], axis=-1))), axis=-1)
    return mx.logsumexp(lw, axis=-1) - math.log(N) + _TWO_LOG_2PI + logdet


def omega_from_free(Y, dep, order):
    """Free weights ``Y (..., D)`` -> ``omega (..., G)`` (numpy)."""
    Y = np.asarray(Y, dtype=np.float64)
    om = np.zeros(Y.shape[:-1] + (Y.shape[-1] + 1,))
    om[..., list(order)] = Y
    om[..., dep] = 1.0 - Y.sum(-1)
    return om


# ------------------------------------------------------------ host helpers

def sobol_points(n, dim, seed):
    """A scrambled Sobol rule ``(n, dim)`` in (0, 1), clipped away from the
    ends. Stored with a fit (``ResumeState``) rather than regenerated, so a
    scipy upgrade cannot change a resumed target."""
    from scipy.stats import qmc

    pts = qmc.Sobol(dim, scramble=True, seed=seed).random(n)
    return np.clip(pts, 1e-7, 1.0 - 1e-7)


def choose_layout(xs, H):
    """``(dep, order)`` for the GHK, frozen once per fit: the dependent
    weight is the one with the most slack at ``x*`` in units of its
    standard deviation, so the coupling constraint rarely binds; the rest
    are drawn walls-first. Any layout is unbiased; this keeps the variance
    low. ``xs (D,)``, ``H (D, D)``, float64."""
    xs = np.asarray(xs, dtype=np.float64)
    D = xs.size
    om = np.r_[1.0 - xs.sum(), xs]
    E = np.vstack([-np.ones(D), np.eye(D)])
    cov = E @ np.linalg.pinv(H) @ E.T
    sd = np.sqrt(np.maximum(np.diag(cov), 1e-30))
    slack = om / sd
    dep = int(np.argmax(slack))
    order = [int(g) for g in np.argsort(slack, kind="stable") if g != dep]
    return dep, tuple(order)


_FACES = {}


def qp_simplex_np(xh, H):
    """``argmin (x - xh)^T H (x - xh)`` over ``{x >= 0, sum x <= 1}``, exact,
    by enumerating the supports of ``omega = (1 - sum x, x)``. Host float64,
    one row: the ML display point."""
    xh = np.asarray(xh, dtype=np.float64)
    D = xh.size
    G = D + 1
    if G not in _FACES:
        _FACES[G] = [S for r in range(1, G + 1)
                     for S in itertools.combinations(range(G), r)]
    best, bq = None, np.inf
    for S in _FACES[G]:
        free = [j - 1 for j in S if j >= 1]
        x = np.zeros(D)
        try:
            if 0 in S:
                if free:
                    x[free] = np.linalg.solve(H[np.ix_(free, free)],
                                              H[free] @ xh)
                if np.any(x < -1e-12) or x.sum() > 1 + 1e-12:
                    continue
            else:
                if not free:
                    continue
                Hf = H[np.ix_(free, free)]
                one = np.ones(len(free))
                a = np.linalg.solve(Hf, one)
                c = np.linalg.solve(Hf, H[free] @ xh)
                x[free] = c + (1.0 - one @ c) / (one @ a) * a
                if np.any(x < -1e-12):
                    continue
        except np.linalg.LinAlgError:
            continue
        q = (x - xh) @ H @ (x - xh)
        if q < bq:
            best, bq = x, q
    if best is None:                                  # pragma: no cover
        best = np.full(D, 1.0 / G)
    x = np.clip(best, 0.0, 1.0)
    return x / max(1.0, x.sum())
