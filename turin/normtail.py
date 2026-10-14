"""Tail-stable normal-distribution functions in MLX, for the GHK estimator.

MLX has ``erf`` but no ``erfc``, and ``1 - erf`` is useless in the tails,
where GHK spends much of its time (a truncation interval deep in one tail of
the conditional normal). MLX's own float64 ``erf`` is also only ~2e-4
accurate, and its float64 ``exp`` ~1e-7, so these formulas serve the float64
replica as well; their ~1e-7 relative error is far below anything the
sampler or ``certify`` can see.

Ported from SquishierPlanet's float32 path
(``studies/collapsed_hybrid/elome_target.py``, validated there against a
float64 reference):

- :func:`log_ndtr` -- ``log Phi(x)`` from Numerical Recipes' ``erfcc`` in
  log form (relative error of erfc < 1.2e-7 everywhere), with no ``exp``;
- :func:`ndtri_exp` -- ``Phi^-1(exp(lp))``, Acklam's rational approximation
  driven by ``log p`` in both tails, plus one Newton step;
- :func:`trunc_normal` -- inversion draws from a standard normal truncated to
  ``[a, b]`` and the log of the interval's mass, with ``expm1`` rather than
  ``log1p(-exp(.))``, which is ``log1p(-1) = -inf`` (a NaN gradient) when
  the interval sits deep in a tail and its two ends round together. One
  lower-tail formula after reflecting the interval to a non-positive centre,
  rather than three branches all evaluated under ``where``: a third of the
  work, and the GHK estimator's cost is mostly this.
"""

from __future__ import annotations

import math

import mlx.core as mx

LOG2PI = math.log(2.0 * math.pi)
_SQRT2 = math.sqrt(2.0)
_LOG2 = math.log(2.0)

#: Numerical Recipes' erfcc coefficients
_NR = (-1.26551223, 1.00002368, 0.37409196, 0.09678418, -0.18628806,
       0.27886807, -1.13520398, 1.48851587, -0.82215223, 0.17087277)
#: Acklam's inverse-normal coefficients
_AK_A = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
_AK_B = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01)
_AK_C = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
_AK_D = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00)


def _horner(cs, x):
    acc = cs[0]
    for c in cs[1:]:
        acc = acc * x + c
    return acc


def _log_erfc(z):
    """``log erfc(z)`` for ``z >= 0``: ``log t - z^2 + P(t)``, no exp."""
    t = 1.0 / (1.0 + 0.5 * z)
    acc = _NR[9]
    for c in _NR[8::-1]:
        acc = acc * t + c
    return mx.log(t) - z * z + acc


def log_ndtr(x):
    """``log Phi(x)``, accurate in both tails, in ``x``'s dtype."""
    z = mx.abs(x) / _SQRT2
    le = _log_erfc(z) - _LOG2                           # log Phi(-|x|)
    neg = x < 0
    return mx.where(neg, le, mx.log1p(-mx.exp(mx.where(neg, -1.0, le))))


def ndtri_exp(lp):
    """``Phi^-1(exp(lp))`` for ``lp <= 0``."""
    p_lo = 0.02425
    lower = lp < math.log(p_lo)
    lq = mx.log1p(-mx.exp(mx.minimum(lp, -1e-7)))       # log(1 - p)
    upper = lq < math.log(p_lo)
    ql = mx.sqrt(-2.0 * mx.where(lower, lp, -1.0))
    z_lo = _horner(_AK_C, ql) / (_horner(_AK_D, ql) * ql + 1.0)
    qu = mx.sqrt(-2.0 * mx.where(upper, lq, -1.0))
    z_up = -_horner(_AK_C, qu) / (_horner(_AK_D, qu) * qu + 1.0)
    p = mx.exp(mx.where(lower | upper, -0.7, lp))
    q = p - 0.5
    r = q * q
    z_mid = _horner(_AK_A, r) * q / (_horner(_AK_B, r) * r + 1.0)
    z = mx.where(lower, z_lo, mx.where(upper, z_up, z_mid))
    lz = log_ndtr(z)
    lphi = -0.5 * z * z - 0.5 * LOG2PI
    return z - (lz - lp) / mx.exp(lphi - lz)


def trunc_normal(a, b, v):
    """Draw from N(0, 1) truncated to ``[a, b]`` by inverting the uniform
    ``v``; returns ``(z, log P(a <= Z <= b))``.

    The interval is reflected so its centre is non-positive, ``[a', b'] =
    [-b, -a]`` when ``a + b > 0``, and then one lower-tail form serves every
    case: ``log P = log Phi(b') + log(-expm1(log Phi(a') - log Phi(b')))``
    and ``z' = Phi^-1(Phi(a') + v P)``, all in log space. After the
    reflection ``Phi(a')`` is at most 1/2, so neither end rounds to 1 unless
    the interval is wide and the draw lies beyond ~5 sigma at its far end
    (probability ~1e-7). The reflected branch maps ``v`` to ``z`` in reverse,
    which is still an exact draw for uniform ``v``.
    """
    flip = (a + b) > 0
    a2 = mx.where(flip, -b, a)
    b2 = mx.where(flip, -a, b)
    la, lb = log_ndtr(a2), log_ndtr(b2)
    lm = lb + mx.log(-mx.expm1(mx.minimum(la - lb, -1e-30)))
    z2 = ndtri_exp(mx.minimum(mx.logaddexp(la, mx.log(v) + lm), 0.0))
    z = mx.where(flip, -z2, z2)
    return mx.clip(z, a, b), lm
