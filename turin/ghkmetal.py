"""GHK on a fixed rule as two Metal kernels: the per-point log-weights
``lw`` (forward) and their vector-Jacobian product (backward), for collapsed
hybrid4's float32 GPU path.

:func:`turin.simplexmass.ghk_terms` (an MLX graph) is the reference and stays
the path for float64 and the CPU stream; :func:`ghk_lw` here computes the
same function point by point, one thread per (chain, point):

- the forward is ``_ghk_forward``'s arithmetic, step for step: normtail's
  ``log_ndtr`` (NR erfcc in log form), ``ndtri_exp`` (Acklam plus one Newton
  step) and ``trunc_normal`` (reflection to a non-positive centre, one
  lower-tail form), the ``eps`` floor on ``hi`` and the clip of the draw;
- the backward recomputes each point's forward in registers and runs the
  reverse pass of :func:`turin.simplexmass.ghk_lw_vjp` (``_tn_partials``:
  autodiff's chain in closed form, from the derivative of the log Phi
  approximation itself, with MLX's tie rules), then sums over the points of
  each simdgroup (``simd_sum``) and leaves the rest of the sum to MLX.

Metal's standard library has no ``log1p`` or ``expm1``; the classic accurate
constructions are used (``log(u) * x / (u - 1)`` with ``u = 1 + x``, and its
``expm1`` counterpart). The value then agrees with the graph to float32
rounding rather than bitwise, which is checked against float64 by the tests.
"""

from __future__ import annotations

import math

import mlx.core as mx

from .normtail import _AK_A, _AK_B, _AK_C, _AK_D, _NR

_HEADER = r"""
constant constexpr float LOG2PI_F = 1.8378770664093453f;
constant constexpr float SQRT2_F = 1.4142135623730951f;
constant constexpr float LOG2_F = 0.6931471805599453f;

inline float log1p_acc(float x) {
    float u = 1.0f + x;
    if (u == 1.0f) return x;
    return metal::precise::log(u) * (x / (u - 1.0f));
}
inline float expm1_acc(float x) {
    if (metal::fabs(x) < 1e-5f) return x + 0.5f * x * x;
    float u = metal::precise::exp(x);
    if (u == 1.0f) return x;
    float um1 = u - 1.0f;
    if (um1 == -1.0f) return -1.0f;
    return um1 * x / metal::precise::log(u);
}
inline float logaddexp_f(float a, float b) {
    float m = metal::max(a, b);
    if (metal::isinf(m) && m < 0.0f) return m;
    return m + log1p_acc(metal::precise::exp(-metal::fabs(a - b)));
}
inline float sigmoid_f(float x) {
    return 1.0f / (1.0f + metal::precise::exp(-x));
}
inline float log_erfc_f(float z) {
    float t = 1.0f / (1.0f + 0.5f * z);
    float acc = NR9;
    acc = acc * t + NR8; acc = acc * t + NR7; acc = acc * t + NR6;
    acc = acc * t + NR5; acc = acc * t + NR4; acc = acc * t + NR3;
    acc = acc * t + NR2; acc = acc * t + NR1; acc = acc * t + NR0;
    return metal::precise::log(t) - z * z + acc;
}
inline float log_ndtr_f(float x) {
    float z = metal::fabs(x) / SQRT2_F;
    float le = log_erfc_f(z) - LOG2_F;
    if (x < 0.0f) return le;
    return log1p_acc(-metal::precise::exp(le));
}
/* the derivative of log_ndtr_f's approximation (simplexmass._dlog_ndtr) */
inline float dlog_ndtr_f(float x) {
    float z = metal::fabs(x) / SQRT2_F;
    float t = 1.0f / (1.0f + 0.5f * z);
    float acc = NR9, dacc = 0.0f;
    dacc = dacc * t + acc; acc = acc * t + NR8;
    dacc = dacc * t + acc; acc = acc * t + NR7;
    dacc = dacc * t + acc; acc = acc * t + NR6;
    dacc = dacc * t + acc; acc = acc * t + NR5;
    dacc = dacc * t + acc; acc = acc * t + NR4;
    dacc = dacc * t + acc; acc = acc * t + NR3;
    dacc = dacc * t + acc; acc = acc * t + NR2;
    dacc = dacc * t + acc; acc = acc * t + NR1;
    dacc = dacc * t + acc; acc = acc * t + NR0;
    float g = (0.5f * t + 2.0f * z + 0.5f * t * t * dacc) / SQRT2_F;
    if (x < 0.0f) return g;
    float le = metal::precise::log(t) - z * z + acc - LOG2_F;
    return g * metal::precise::exp(le) / (-expm1_acc(le));
}
inline float ndtri_exp_f(float lp) {
    const float p_lo = 0.02425f;
    bool lower = lp < metal::precise::log(p_lo);
    float lq = log1p_acc(-metal::precise::exp(metal::min(lp, -1e-7f)));
    bool upper = lq < metal::precise::log(p_lo);
    float z;
    if (lower) {
        float q = metal::precise::sqrt(-2.0f * lp);
        float num = ((((AKC0 * q + AKC1) * q + AKC2) * q + AKC3) * q + AKC4) * q + AKC5;
        float den = (((AKD0 * q + AKD1) * q + AKD2) * q + AKD3) * q + 1.0f;
        z = num / den;
    } else if (upper) {
        float q = metal::precise::sqrt(-2.0f * lq);
        float num = ((((AKC0 * q + AKC1) * q + AKC2) * q + AKC3) * q + AKC4) * q + AKC5;
        float den = (((AKD0 * q + AKD1) * q + AKD2) * q + AKD3) * q + 1.0f;
        z = -num / den;
    } else {
        float q = metal::precise::exp(lp) - 0.5f;
        float r = q * q;
        float num = (((((AKA0 * r + AKA1) * r + AKA2) * r + AKA3) * r + AKA4) * r + AKA5) * q;
        float den = ((((AKB0 * r + AKB1) * r + AKB2) * r + AKB3) * r + AKB4) * r + 1.0f;
        z = num / den;
    }
    float lz = log_ndtr_f(z);
    float lphi = -0.5f * z * z - 0.5f * LOG2PI_F;
    return z - (lz - lp) / metal::precise::exp(lphi - lz);
}

/* one GHK point's forward, keeping what the backward needs */
struct Fwd {
    float z[D], lo[D], hir[D], hi[D], cond[D], y[D];
    float lw, s;
};

/* a template over the pointers: MLX passes a small input in the constant
   address space and a large one in device */
template <typename PM, typename PL, typename PU>
inline Fwd ghk_point(PM my, PL Ly, PU U, uint c, uint n) {
    Fwd f;
    float lmsum = 0.0f;
    for (int k = 0; k < D; ++k) {
        float cond = my[c * D + k];
        float ub = 1.0f;
        if (k > 0) {
            float dot = 0.0f;
            for (int j = 0; j < k; ++j) dot += f.z[j] * Ly[(c * D + k) * D + j];
            cond = cond + dot;
            float ys = 0.0f;
            for (int j = 0; j < k; ++j) ys += f.y[j];
            ub = 1.0f - ys;
        }
        float l = Ly[(c * D + k) * D + k];
        float lo = (0.0f - cond) / l;
        float hir = (ub - cond) / l;
        float hi = metal::max(hir, lo + EPS_GAP);
        float v = U[n * D + k];
        bool flip = (lo + hi) > 0.0f;
        float a2 = flip ? -hi : lo;
        float b2 = flip ? -lo : hi;
        float la = log_ndtr_f(a2), lb = log_ndtr_f(b2);
        float lm = lb + metal::precise::log(-expm1_acc(metal::min(la - lb, -1e-30f)));
        float lp = metal::min(logaddexp_f(la, metal::precise::log(v) + lm), 0.0f);
        float z2 = ndtri_exp_f(lp);
        float z = flip ? -z2 : z2;
        z = metal::clamp(z, lo, hi);
        f.z[k] = z; f.lo[k] = lo; f.hir[k] = hir; f.hi[k] = hi;
        f.cond[k] = cond; f.y[k] = cond + l * z;
        lmsum += lm;
    }
    float s = INV_DEP;
    for (int k = 0; k < D; ++k) s += f.y[k] * COEF[k];
    f.s = s;
    f.lw = lmsum - (float)(D + 1) * metal::precise::log(metal::max(s, 1e-30f));
    return f;
}
"""

_FWD_SOURCE = r"""
    uint n = thread_position_in_grid.x;
    uint c = thread_position_in_grid.y;
    uint N = (uint)U_shape[0];
    if (n >= N) return;
    Fwd f = ghk_point(my, Ly, U, c, n);
    lw[c * N + n] = f.lw;
"""

_BWD_SOURCE = r"""
    uint n = thread_position_in_grid.x;
    uint c = thread_position_in_grid.y;
    uint N = (uint)U_shape[0];
    float dmy[D];
    float dL[D][D];
    for (int i = 0; i < D; ++i) {
        dmy[i] = 0.0f;
        for (int j = 0; j < D; ++j) dL[i][j] = 0.0f;
    }
    if (n < N) {
        Fwd f = ghk_point(my, Ly, U, c, n);
        float gb = gbar[c * N + n];
        float dprior = (f.s > 1e-30f) ? -(float)(D + 1) * gb / f.s : 0.0f;
        float ybar[D], zbar[D];
        for (int k = 0; k < D; ++k) { ybar[k] = dprior * COEF[k]; zbar[k] = 0.0f; }
        for (int k = D - 1; k >= 0; --k) {
            float l = Ly[(c * D + k) * D + k];
            float z = f.z[k], lo = f.lo[k], hir = f.hir[k], hi = f.hi[k];
            float v = U[n * D + k];
            bool flip = (lo + hi) > 0.0f;
            float cond_bar = ybar[k];
            float l_bar = ybar[k] * z;
            float zb = zbar[k] + ybar[k] * l;
            float a2 = flip ? -hi : lo;
            float b2 = flip ? -lo : hi;
            float z2 = flip ? -z : z;
            /* _tn_partials */
            float la = log_ndtr_f(a2), lb = log_ndtr_f(b2);
            float d = la - lb;
            bool clamped = d >= -1e-30f;
            float dmin = metal::min(d, -1e-30f);
            float q = -expm1_acc(dmin);
            float lm = lb + metal::precise::log(q);
            float ed_q = metal::precise::exp(dmin) / q;
            float dlm_dla = clamped ? 0.0f : -ed_q;
            float dlm_dlb = clamped ? 1.0f : 1.0f + ed_q;
            float dla = dlog_ndtr_f(a2), dlb = dlog_ndtr_f(b2);
            float dlm_da = dlm_dla * dla;
            float dlm_db = dlm_dlb * dlb;
            float t2 = metal::precise::log(v) + lm;
            bool capped = logaddexp_f(la, t2) >= 0.0f;
            float sa = sigmoid_f(la - t2);
            float dz_dlp = capped ? 0.0f : 1.0f / dlog_ndtr_f(z2);
            float dz_da = dz_dlp * (sa * dla + (1.0f - sa) * dlm_da);
            float dz_db = dz_dlp * ((1.0f - sa) * dlm_db);
            float dlm_dlo = flip ? -dlm_db : dlm_da;
            float dlm_dhi = flip ? -dlm_da : dlm_db;
            float dz_dlo = flip ? dz_db : dz_da;
            float dz_dhi = flip ? dz_da : dz_db;
            bool at_lo = z <= lo, at_hi = z >= hi;
            if (at_lo || at_hi) {
                dz_dlo = at_lo ? 1.0f : 0.0f;
                dz_dhi = at_hi ? 1.0f : 0.0f;
            }
            float lo_bar = gb * dlm_dlo + zb * dz_dlo;
            float hi_bar = gb * dlm_dhi + zb * dz_dhi;
            bool raw = hir > lo + EPS_GAP;
            float hir_bar = raw ? hi_bar : 0.0f;
            lo_bar = lo_bar + (raw ? 0.0f : hi_bar);
            cond_bar = cond_bar - (lo_bar + hir_bar) / l;
            l_bar = l_bar - (lo_bar * lo + hir_bar * hir) / l;
            float ub_bar = hir_bar / l;
            dmy[k] += cond_bar;
            for (int j = 0; j < k; ++j) {
                dL[k][j] += cond_bar * f.z[j];
                zbar[j] += cond_bar * Ly[(c * D + k) * D + j];
                ybar[j] -= ub_bar;
            }
            dL[k][k] += l_bar;
        }
    }
    /* one partial per simdgroup; MLX sums the rest */
    uint ngrp = (N + 31u) / 32u;
    uint g = n / 32u;
    for (int i = 0; i < D; ++i) {
        float sm = metal::simd_sum(dmy[i]);
        if (metal::simd_is_first()) part[(c * ngrp + g) * NP + i] = sm;
        for (int j = 0; j <= i; ++j) {
            float sl = metal::simd_sum(dL[i][j]);
            if (metal::simd_is_first())
                part[(c * ngrp + g) * NP + D + i * (i + 1) / 2 + j] = sl;
        }
    }
"""

_KERNELS = {}


def _constants(D, dep, order, inv_fstar):
    inv_dep = float(inv_fstar[dep])
    coef = [float(inv_fstar[g]) - inv_dep for g in order]
    lines = [f"constant constexpr int D = {D};",
             f"constant constexpr int NP = {D + D * (D + 1) // 2};",
             "constant constexpr float EPS_GAP = 1e-6f;",
             f"constant constexpr float INV_DEP = {inv_dep!r}f;",
             f"constant float COEF[{D}] = {{{', '.join(f'{c!r}f' for c in coef)}}};"]
    for i, c in enumerate(_NR):
        lines.append(f"constant constexpr float NR{i} = {c!r}f;")
    for name, cs in (("AKA", _AK_A), ("AKB", _AK_B), ("AKC", _AK_C), ("AKD", _AK_D)):
        for i, c in enumerate(cs):
            lines.append(f"constant constexpr float {name}{i} = {c!r}f;")
    return "\n".join(lines) + "\n"


def _kernels(D, dep, order, inv_fstar):
    key = (D, int(dep), tuple(int(o) for o in order),
           tuple(float(f) for f in inv_fstar))
    if key not in _KERNELS:
        header = _constants(D, dep, order, inv_fstar) + _HEADER
        tag = abs(hash(key)) % 10 ** 8
        fwd = mx.fast.metal_kernel(
            name=f"turin_ghk_fwd_{tag}", input_names=["my", "Ly", "U"],
            output_names=["lw"], source=_FWD_SOURCE, header=header)
        bwd = mx.fast.metal_kernel(
            name=f"turin_ghk_bwd_{tag}", input_names=["my", "Ly", "U", "gbar"],
            output_names=["part"], source=_BWD_SOURCE, header=header)
        _KERNELS[key] = (fwd, bwd, _make_fn(fwd, bwd, D))
    return _KERNELS[key]


def _make_fn(fwd, bwd, D):
    NP = D + D * (D + 1) // 2
    tri = [(i, j) for i in range(D) for j in range(i + 1)]

    def launch_fwd(my, Ly, U):
        C, N = my.shape[0], U.shape[0]
        (lw,) = fwd(inputs=[my, Ly, U], grid=(N, C, 1),
                    threadgroup=(min(256, N), 1, 1),
                    output_shapes=[(C, N)], output_dtypes=[mx.float32])
        return lw

    @mx.custom_function
    def lw_fn(my, Ly, U):
        return launch_fwd(my, Ly, U)

    @lw_fn.vjp
    def lw_vjp(primals, cotangent, output):
        my, Ly, U = primals
        C, N = my.shape[0], U.shape[0]
        ngrp = (N + 31) // 32
        gbar = cotangent if not isinstance(cotangent, (list, tuple)) \
            else cotangent[0]
        (part,) = bwd(inputs=[my, Ly, U, gbar.astype(mx.float32)],
                      grid=(ngrp * 32, C, 1), threadgroup=(min(256, ngrp * 32), 1, 1),
                      output_shapes=[(C, ngrp, NP)],
                      output_dtypes=[mx.float32], init_value=0.0)
        tot = mx.sum(part, axis=1)                                # (C, NP)
        dmy = tot[:, :D]
        rows = []
        for i in range(D):
            rows.append(mx.stack(
                [tot[:, D + tri.index((i, j))] if j <= i
                 else mx.zeros((C,), dtype=mx.float32) for j in range(D)],
                axis=-1))
        dLy = mx.stack(rows, axis=-2)
        return dmy, dLy, mx.zeros_like(U)

    return lw_fn


def ghk_lw(my, Ly, U, dep, order, inv_fstar):
    """``lw (C, N)`` of :func:`turin.simplexmass.ghk_terms` for float32
    ``my (C, D)``, ``Ly (C, D, D)`` and a shared rule ``U (N, D)``, by the
    Metal kernels, differentiable in ``(my, Ly)``."""
    D = my.shape[-1]
    fn = _kernels(D, dep, order, inv_fstar)[2]
    return fn(my.astype(mx.float32), Ly.astype(mx.float32),
              U.astype(mx.float32))
