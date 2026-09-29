"""Choosing the profile-likelihood solve empirically, per target.

``--PL`` picks how the nuisance baseline coefficients are solved (see
:mod:`turin.profile`). Which mode is right is **not** a property of the code,
it is a property of the target, and the deciding quantity is not obvious:

    sd of  logL_mode(float32) - logL_exact(float64)

over a ball of parameters. The *standard deviation*, not the mean: a constant
offset in the log-density leaves the posterior untouched, which is exactly
what anvil's own ``certify`` says when it prints the float32 error mean as
"irrelevant: it cancels". Measured on two synthetic targets:

    u-ball spread   1000 ppm: exact/hybrid/ratio    1% depth: exact/hybrid/ratio
            1e-3     0.022 / 0.022 / 0.022           0.19 / 0.19 / 0.19
            1e-2     0.024 / 0.024 / 0.024           0.19 / 0.19 / 0.77
            1e-1     0.026 / 0.026 / 0.026           0.19 / 0.19 / 7.7
            3e-1     0.025 / 0.025 / 0.027           0.21 / 0.21 / 60

Three things follow, and each is a design constraint here.

**The answer is target-dependent.** ``ratio`` costs nothing at Kepler depths
and is unusable at a per cent. No fixed default is right for both.

**The ball must be the posterior's width -- measured, not assumed.** This is
the subtle part, and getting it wrong inverts the answer. Too narrow (anvil's
1e-3 initialization ball) and every mode passes, because a nearly-constant
offset cancels. Too wide (a fixed 0.3) and ``ratio`` is rejected everywhere,
because it is being judged far out in the prior where the sampler never goes.
Measured on the table above, a *fixed* 0.3 ball rejected ``ratio`` at 1000 ppm
in five seeds out of six, purely from where the draws landed.

So the width is calibrated per target: probe once, measure the median
log-density drop, and rescale to the width where that drop is ``dim/2`` --
what a 1-sigma ball costs for a Gaussian posterior in ``dim`` dimensions.
Because the drop is quadratic in the width near a mode, one probe fixes the
scale. The criterion is then checked at that width and at twice it, as margin
against the posterior not being Gaussian.

**The threshold must be relative.** At 1% depth ``exact`` itself carries
sd ~ 0.19 from float32 alone, so any absolute bar either rejects everything
deep or waves through everything shallow. A mode qualifies when it adds little
to the float32 noise ``exact`` already has.

The probe also finds the ``exact``/``hybrid`` crossover (around 14 basis
columns, on this machine) empirically for the actual problem, instead of
trusting a number measured once somewhere else.

Cost: about 2 seconds, against sampling rounds of minutes.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import mlx.core as mx
import numpy as np

from . import likelihood as _likelihood
from . import profile as _profile

#: Ball width used only to *calibrate*, before the real widths are derived
#: from it. Any value works; the calibration rescales from whatever it
#: measures.
CALIBRATION_SPREAD = 0.05
#: Multiples of the calibrated posterior width to test the criterion at. The
#: second is margin against the posterior not being locally Gaussian.
WIDTH_MULTIPLES = (1.0, 2.0)
#: Target median log-density drop that defines "one posterior width": for a
#: Gaussian posterior in ``dim`` dimensions a 1-sigma ball drops by dim/2.
TARGET_DROP_PER_DIM = 0.5
#: States per width.
N_PROBE = 48
#: A mode qualifies if its sd is within this factor of ``exact``'s own ...
SD_FACTOR = 1.5
#: ... or below this floor, whichever is larger. The floor keeps a target
#: whose float32 noise is tiny from rejecting a mode over a negligible
#: absolute difference.
SD_FLOOR = 0.02
#: A qualifying mode must beat ``exact`` by at least this much *and* by more
#: than the measurement can confuse (see :func:`_time_modes`) to displace it.
SPEED_MARGIN = 0.10
#: Timing repetitions. Measured spread across identical repeats is 29-96% even
#: at 240 evaluations per mode, and does not converge with more, so these are
#: sized for a usable range rather than a precise mean.
TIMING_REPS = 6
TIMING_INNER = 8
#: Wall-clock budget for the timing phase. A slow mode (exact at 30 basis
#: columns costs ~270ms per evaluation) would otherwise dominate the probe;
#: repetitions stop once this is spent, keeping at least two.
TIMING_BUDGET_S = 4.0


@dataclass
class PLRow:
    """One mode's probe result."""

    mode: str
    sd_by_spread: dict
    sd_worst: float
    mean_offset: float
    ms: float               # best observed, the robust estimator
    ms_worst: float = 0.0   # worst observed; the pair brackets the noise
    qualifies: bool = False
    note: str = ""


@dataclass
class PLChoice:
    """What the probe decided, and the evidence for it."""

    mode: str
    reason: str
    rows: list = field(default_factory=list)
    sd_exact: float = 0.0
    threshold: float = 0.0
    width: float = 0.0            # calibrated posterior width, u-space
    widths: tuple = ()            # the widths the criterion was tested at
    calib_drop: float = 0.0       # median logL drop at the calibration probe
    elapsed: float = 0.0
    probed: bool = True

    def row(self, mode):
        for r in self.rows:
            if r.mode == mode:
                return r
        return None

    def describe(self):
        if not self.probed:
            return f"  --PL={self.mode} ({self.reason})"
        head = (f"  PL probe, {self.elapsed:.1f}s: posterior width calibrated "
                f"to {self.width:.3g} in u-space, tested at "
                + " and ".join(f"{w:.3g}" for w in self.widths) + "\n"
                f"  threshold: sd <= {self.threshold:.4g} "
                f"(exact's own float32 sd is {self.sd_exact:.4g})\n"
                f"    {'mode':>7s} {'sd(worst)':>11s} "
                f"{'ms/eval (best-worst)':>20s} verdict")
        lines = [head]
        for r in self.rows:
            verdict = "ok" if r.qualifies else "REJECTED"
            mark = " <-- chosen" if r.mode == self.mode else ""
            lines.append(f"    {r.mode:>7s} {r.sd_worst:11.4g} "
                         f"{r.ms:6.2f}-{r.ms_worst:<6.2f}ms {verdict}{mark}"
                         + (f"  ({r.note})" if r.note else ""))
        lines.append(f"  chose {self.mode}: {self.reason}")
        return "\n".join(lines)


def _time_modes(lps, v, reps=TIMING_REPS, inner=TIMING_INNER,
                budget_s=TIMING_BUDGET_S):
    """Best-of, round-robin timing of several compiled value+grad graphs.

    Returns ``{mode: (best, worst)}``. Three things matter here, all learned
    by measuring.

    **Round-robin**: timing the modes one after another gave 6.2 / 11.2 /
    14.9 ms in measurement order -- pure drift. Interleaving the repetitions
    gives 13.8 / 17.8 / 13.0, matching careful standalone benchmarks.

    **Minimum, not mean**: these are short GPU kernels, contaminated upward by
    scheduling and thermal state, never downward.

    **The spread is reported, because it is large.** Across five identical
    repeats the observed spread was 29-96% per mode even at 240 evaluations,
    and it does not shrink with more. At Legendre basis sizes the modes differ
    by less than that, so a caller must not rank them on the best value alone;
    :func:`select_pl_mode` requires a candidate's *worst* time to beat
    ``exact``'s *best*.
    """
    fns = {}
    for mode, lp in lps.items():
        fns[mode] = mx.compile(lambda vv, f=lp: mx.vjp(
            f, [vv], [mx.ones((vv.shape[0],), dtype=vv.dtype)])[1][0])
        mx.eval(fns[mode](v))                       # compile, off the clock
    seen = {mode: [] for mode in lps}
    started = time.perf_counter()
    for rep in range(reps):
        for mode, fn in fns.items():           # round robin, not mode by mode
            t0 = time.perf_counter()
            for _ in range(inner):
                mx.eval(fn(v))
            seen[mode].append((time.perf_counter() - t0) / inner * 1e3)
        # always complete a whole round, so every mode has the same count
        if rep + 1 >= 2 and time.perf_counter() - started > budget_s:
            break
    return {mode: (min(ts), max(ts)) for mode, ts in seen.items()}


def _calibrate_width(ref, u_map, transform, dim, *, n_probe=N_PROBE, seed=0,
                     probe_spread=CALIBRATION_SPREAD):
    """The u-space ball width whose median log-density drop is ``dim/2``.

    Near a mode the drop is quadratic in the width, so a single probe fixes
    the scale: ``s* = s0 * sqrt(target / drop(s0))``. Returns
    ``(width, measured_drop_at_probe)``, the width clipped to a sane range so
    a pathological measurement cannot produce a degenerate ball.
    """
    rng = np.random.default_rng(seed)
    with mx.stream(mx.cpu):
        lp_map = float(np.array(ref(mx.array(
            transform.model_np(u_map[None, :]), dtype=mx.float64)))[0])
        u = u_map[None, :] + probe_spread * rng.standard_normal(
            (n_probe, u_map.size))
        lp = np.array(ref(mx.array(transform.model_np(u), dtype=mx.float64)),
                      dtype=np.float64)
    drop = float(np.median(lp_map - lp[np.isfinite(lp)])) if np.isfinite(
        lp).any() else float("nan")
    target = TARGET_DROP_PER_DIM * dim
    if not np.isfinite(drop) or drop <= 0:
        # u_map is not a mode (or the density is flat here): fall back to the
        # probe width rather than inventing a scale
        return probe_spread, drop
    width = probe_spread * np.sqrt(target / drop)
    return float(np.clip(width, 1e-4, 1.0)), drop


def select_pl_mode(layout, centering, epoch_data, orders, u_map, transform, *,
                   build_kwargs=None, n_chains=512,
                   width_multiples=WIDTH_MULTIPLES, n_probe=N_PROBE,
                   sd_factor=SD_FACTOR, sd_floor=SD_FLOOR,
                   speed_margin=SPEED_MARGIN, modes=_profile.PROFILE_MODES,
                   seed=0, log=None):
    """Measure every ``--PL`` mode on this target and choose one.

    ``u_map`` is the maximum-a-posteriori position in unconstrained space.
    The probe balls are centred there and scaled to the posterior's own width,
    so the modes are compared where the sampler will actually spend its time.

    Returns a :class:`PLChoice`. ``exact`` always qualifies: it is the
    reference, compared against the same computation in float64.
    """
    t_start = time.perf_counter()
    build_kwargs = dict(build_kwargs or {})
    for key in ("profile_mode", "dtype"):
        build_kwargs.pop(key, None)
    u_map = np.asarray(u_map, dtype=np.float64).reshape(-1)

    def build(mode, dtype):
        return _likelihood.ProfiledTransitLogProb(
            layout, centering, epoch_data, orders, profile_mode=mode,
            dtype=dtype, **build_kwargs)

    with mx.stream(mx.cpu):
        ref = build("exact", mx.float64)

    width, drop = _calibrate_width(ref, u_map, transform, layout.dim,
                                   n_probe=n_probe, seed=seed)
    widths = tuple(width * m for m in width_multiples)

    rng = np.random.default_rng(seed + 1)
    balls, ref_lp = {}, {}
    for w in widths:
        balls[w] = transform.model_np(
            u_map[None, :] + w * rng.standard_normal((n_probe, u_map.size)))
        with mx.stream(mx.cpu):
            ref_lp[w] = np.array(ref(mx.array(balls[w], dtype=mx.float64)),
                                 dtype=np.float64)

    lps = {mode: build(mode, mx.float32) for mode in modes}
    timing = _time_modes(lps, mx.array(np.repeat(
        transform.model_np(u_map[None, :]), n_chains, axis=0
    ).astype(np.float32)))

    rows = []
    for mode, lp in lps.items():
        sd_by_width, offsets = {}, []
        for w in widths:
            got = np.array(lp(mx.array(balls[w].astype(np.float32))),
                           dtype=np.float64)
            d = got - ref_lp[w]
            d = d[np.isfinite(d)]
            sd_by_width[w] = float(d.std()) if d.size else float("inf")
            offsets.append(float(d.mean()) if d.size else float("nan"))
        rows.append(PLRow(mode=mode, sd_by_spread=sd_by_width,
                          sd_worst=max(sd_by_width.values()),
                          mean_offset=float(np.nanmean(offsets)),
                          ms=timing[mode][0], ms_worst=timing[mode][1]))

    by_mode = {r.mode: r for r in rows}
    exact = by_mode.get("exact")
    sd_exact = exact.sd_worst if exact else 0.0
    threshold = max(sd_floor, sd_factor * sd_exact)

    for r in rows:
        if r.mode == "exact":
            r.qualifies, r.note = True, "reference"
        else:
            r.qualifies = bool(np.isfinite(r.sd_worst)
                               and r.sd_worst <= threshold)
            if not r.qualifies:
                r.note = f"{r.sd_worst / max(sd_exact, 1e-30):.0f}x exact's sd"

    best, reason = exact, "it was the fastest qualifying mode"
    if exact is None:                      # pragma: no cover - modes= override
        best = min((r for r in rows if r.qualifies), key=lambda r: r.ms,
                   default=rows[0])
        reason = "fastest qualifying mode"
    else:
        # Decisive means: the candidate's WORST time beats exact's BEST, by
        # the margin. The per-mode timing spread is comparable to the
        # differences between modes at Legendre sizes, so anything weaker
        # would let noise pick the likelihood.
        cands = [r for r in rows if r.qualifies and r.mode != "exact"
                 and r.ms_worst < exact.ms * (1.0 - speed_margin)]
        if cands:
            best = min(cands, key=lambda r: r.ms)
            gain = (exact.ms - best.ms) / exact.ms if exact.ms > 0 else 0.0
            reason = (f"qualifies (sd {best.sd_worst:.3g} <= {threshold:.3g}) "
                      f"and is decisively faster: its slowest run "
                      f"({best.ms_worst:.2f}ms) beat exact's fastest "
                      f"({exact.ms:.2f}ms), {gain * 100:.0f}% on best-vs-best")
        elif any(not r.qualifies for r in rows):
            reason = "the alternatives did not meet the precision bar"
        elif any(r.ms < exact.ms for r in rows if r.mode != "exact"):
            reason = ("no alternative was decisively faster -- the timing "
                      "spread is as large as the difference between modes")

    choice = PLChoice(mode=best.mode, reason=reason, rows=rows,
                      sd_exact=sd_exact, threshold=threshold, width=width,
                      widths=widths, calib_drop=drop,
                      elapsed=time.perf_counter() - t_start)
    if log:
        log(choice.describe())
    return choice


def fixed_choice(mode, reason="set on the command line"):
    """A :class:`PLChoice` for an explicitly requested mode, with no probing."""
    return PLChoice(mode=mode, reason=reason, probed=False)
