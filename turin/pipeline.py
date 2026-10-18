"""Fit orchestration: one target, end to end, in hurin's order.

Download and condition the light curve, cross-validate the Legendre orders,
fit the linear ephemeris, then fit per-epoch transit times seeded from the
LinEph maximum-likelihood shape. Products are re-exported after every sampling
round, so a long run is interruptible and its partial results are usable --
hurin's behaviour, and the reason its resume state exists.
"""

from __future__ import annotations

import dataclasses
import math
import os
import time

import mlx.core as mx
import numpy as np

from . import MODEL_REV as _MODEL_REV
from . import __version__
from . import capabilities as _caps
from . import likelihood as _likelihood
from . import outputs as _outputs
from . import params as _params
from . import plots as _plots
from . import plselect as _plselect
from . import prep as _prep
from . import profile as _profile
from . import sampling as _sampling
from . import seeding as _seeding

#: Transit times shown at each end of a long TTV fit's corner plot; the ones
#: between are replaced by a "..." row and column.
CORNER_EPOCHS = 7

#: Minimum seconds between writes of the chains tarball and the PDFs during
#: a run; they are always written after round 0 and at the end.
HEAVY_EVERY_S = 1200

#: Export row budget for collapsed hybrid4, whose every exported row needs a
#: conditional q draw (~1.3 ms each, measured at 128 steps): 160,000 rows
#: keeps an export near 3.5 min where the general budget (1,000,000) would
#: take 20+. Thinning is systematic and R-hat/ESS still see every draw, so
#: only the products' row count changes.
_EXPORT_ROWS_COLLAPSED_HYBRID = 160_000

PARAM_LABELS = {
    "dP": "$P$ (d)", "dtau0": r"$\tau_0$", "k": "$k = R_p/R_\\star$",
    "b": "$b$", "T14": "$T_{14}$ (d)", "q1": "$q_1$",
    "q2": "$q_2$", "q3": "$q_3$", "q4": "$q_4$",
    "w1": "$w_1$", "w2": "$w_2$", "w3": "$w_3$", "w4": "$w_4$",
    "fc": r"$\sqrt{f}\cos 2\theta$", "fs": r"$\sqrt{f}\sin 2\theta$",
    "f": "$f$", "theta": r"$\theta$ (deg)", "A": "$A$", "B": "$B$",
}




def _ghk_kw(layout, state):
    """``build_target`` keywords for a collapsed hybrid4 target resumed from
    ``state``: its frozen GHK layout and points. Empty otherwise."""
    if (layout.ld != "collapsed" or layout.law != "hybrid4" or state is None
            or getattr(state, "ghk_layout", None) is None):
        return {}
    return dict(ghk_layout=state.ghk_layout, ghk_points=state.ghk_points)

def _rhostar(args):
    """The ``--rhostar`` prior as the layout and the resume guard carry it:
    ``(mean, sigma)``, or ``()`` for none (so "none" is a value a resumed
    lineage is checked against, not a skipped guard)."""
    prior = getattr(args, "rhostar", None)
    return tuple(float(x) for x in prior) if prior else ()


def _model_ld_names(law):
    from .model import ld_names

    return ld_names(law)

def _label(name):
    if name in PARAM_LABELS:
        return PARAM_LABELS[name]
    if name.startswith("dtau_"):
        return rf"$\tau_{{{name[5:]}}}$"
    return name


def _gibbsgrid(mode, args):
    """The grid-Gibbs setting in effect for this fit.

    Only TTV fits have epoch times to move, and only ChEES has the state
    turin knows how to move chains in, so everything else is "off" whatever
    the flag says.
    """
    return ("on" if mode == "ttv" and args.sampler == "chees"
            and args.gibbsgrid == "on" else "off")


def run(args, log=print):
    """Run every requested fit for one target. Returns a process exit code."""
    _caps.require_anvil()        # before anything touches the disk
    from .cli import (collapsed_ld_problem, law_problem, mle_problem,
                      oblate_problem, resolve_ld, resolve_mle)

    resolve_mle(args)
    resolve_ld(args)
    _caps.require_elome_for_law(args.law)
    problem = (mle_problem(args) or oblate_problem(args) or law_problem(args)
               or collapsed_ld_problem(args))
    if problem:
        raise SystemExit(f"turin: {problem}")
    _outputs.set_run_status("")
    _outputs.set_run_tag(args.tag)
    target = args.target
    outdir = args.outdir or target
    os.makedirs(outdir, exist_ok=True)

    log(f"turin {__version__} — {target}")
    log(_caps.detect().summary())

    if args.fresh:
        removed = _outputs.clear_products(outdir, target)
        if removed:
            log(f"  --fresh: removed {len(removed)} existing product(s)")

    t_start = time.perf_counter()
    prepared = _prep.prepare_data(
        target, ttv_max_days=args.ttv_max_days or 0.0,
        sc_override=args.sc_override, log=log)

    if getattr(args, "replot", False):
        return replot(args, prepared, outdir, log=log)

    log(f"[{target}] Cross-validating Legendre orders...")
    cv = _prep.cv_orders(prepared.epoch_data, prepared.eph["duration"],
                         tau_shift_max=args.ttv_max_days or 0.0,
                         log=lambda m: log(f"  {m}"))

    if getattr(args, "mle", False):
        code = _run_mle(args, prepared, cv, outdir, log=log)
        log(f"[{target}] done in {time.perf_counter() - t_start:.1f}s")
        return code

    lineph_ml = None
    outcome = {}             # mode -> final Verdict, or a note if skipped
    for mode in args.modes:
        try:
            lineph_ml = _fit_mode(
                mode, args, prepared, cv, outdir, log=log,
                outcome=outcome, lineph_ml=lineph_ml) or lineph_ml
        except SystemExit:
            raise
        except Exception as exc:
            log(f"[{target}] {mode} FAILED: {type(exc).__name__}: {exc}")
            raise

    log(f"[{target}] done in {time.perf_counter() - t_start:.1f}s")
    return _report_outcome(target, outcome, log)


#: --replot's "same model" test: the rebuilt float64 log-density at the
#: fit's ML row must match the stored value to this many nats (scaled up
#: by 1 + |logL|/5e4 for the float32 rounding of very large likelihoods).
#: Reproducible fits agree to ~1e-3; the smallest real change measured
#: (22 points of 213, KOI-5749.01) missed by 128.
_ML_REPLAY_TOL = 0.05


def replot(args, prepared, outdir, *, log=print):
    """Redraw a finished lineage's figures and ttv_times, without sampling.

    turin skips a converged fit outright, and only writes figures during
    sampling, so figures made by an older turin stay old. This rebuilds them
    from the saved products: the layout and likelihood exactly as fitted (the
    recorded Legendre orders, profile mode and limb-darkening mode), the
    draws from the chains tarball, and the ML row from its loglike column.
    It rewrites the corner, fold and O-C figures and ttv_times.csv (its O-C
    and SNR columns depend on code); the summary, chains, logrho, lcdata and
    resume state are left as the fit wrote them, since R-hat and ESS cannot
    be recomputed from the thinned chains.

    It refuses unless the rebuilt model reproduces the fit's stored best
    log-density at its ML row (`_ML_REPLAY_TOL`), checked in float64. That
    is the test of "same model" here, not MODEL_REV: a revision that changed
    other targets' likelihoods (0.1.50 changed two of 37) leaves most fits
    exactly reproducible, and a fit whose data or model did change misses
    by far more (KOI-5749.01 against the old segmentation: 128 nats, where
    reproducible fits agree to 1e-3).
    """
    import io
    import tarfile

    import pandas as pd

    target = prepared.target
    ed = prepared.epoch_data
    centering = _prep.centering_constants(ed, prepared.eph)
    done_any = False
    for mode in args.modes:
        state = _outputs.load_resume(outdir, target, mode)
        if state is None:
            log(f"[{target}] {mode}: no saved fit to replot")
            continue
        state.check(b_prior=args.b_prior, geometry=args.geometry,
                    ttv_max=args.ttv_max_days, gibbsgrid=_gibbsgrid(mode, args),
                    ld=getattr(args, "ld", "sampled"),
                    law=getattr(args, "law", "quadratic"),
                    oblate=getattr(args, "oblate", False),
                    rhostar=_rhostar(args))
        chains_path = _outputs.product_path(outdir, target, mode, "chains",
                                            "csv.tar.gz")
        summary_path = _outputs.product_path(outdir, target, mode, "summary",
                                             "csv")
        if not (os.path.exists(chains_path) and os.path.exists(summary_path)):
            log(f"[{target}] {mode}: chains or summary missing; skipped")
            continue

        orders = np.asarray(state.legendre_orders)
        if orders.size != ed["n_epochs"]:
            # the epoch selection changed since this fit (0.1.55 fits epochs
            # whose transit fell in a gap); the replay below cannot even be
            # built, so say why rather than fail on a shape
            log(f"[{target}] {mode}: fitted {orders.size} epochs, but this "
                f"turin selects {ed['n_epochs']} (fitted under MODEL_REV "
                f"{getattr(state, 'model_rev', 1)}); refit it rather than "
                "replot")
            continue
        layout, _ = _mode_layout(mode, args, prepared, centering, ed)
        target_fn, transform, lp, _ = _likelihood.build_target(
            layout, centering, ed, orders, profile_mode=state.profile_mode,
            ld_mode=layout.ld, num_resample=prepared.num_resample,
            exposure_time=prepared.exposure_time, geometry=args.geometry,
            n_chains_hint=64, fp64=True, **_ghk_kw(layout, state))
        collapsed = layout.ld == "collapsed"
        names = list(layout.names) + (
            list(_model_ld_names(layout.law)) if collapsed else [])

        tf = tarfile.open(chains_path)
        d = pd.read_csv(io.StringIO(
            tf.extractfile(tf.getmembers()[0]).read().decode()), comment="#")
        missing = [n for n in names if n not in d.columns]
        if missing:
            log(f"[{target}] {mode}: the fit has no {', '.join(missing[:3])}"
                f"{' ...' if len(missing) > 3 else ''} (its epochs differ "
                "from this turin's selection); refit it rather than replot")
            continue
        phys = d[names].to_numpy(dtype=np.float64)
        b_draws = d["b"].to_numpy(dtype=np.float64)
        best = int(np.argmax(d["loglike"].to_numpy()))
        v_ml = (phys[best, :layout.dim]
                - np.asarray(layout.report_offset, dtype=np.float64))
        stored = float(d["loglike"].iloc[best])
        replayed = float(np.asarray(target_fn.log_prob_hi(
            transform.from_model_np(v_ml[None, :])))[0]) + lp.log_const
        miss = replayed - stored
        rev = getattr(state, "model_rev", 1)
        if not abs(miss) <= _ML_REPLAY_TOL * (1.0 + abs(stored) / 5e4):
            log(f"[{target}] {mode}: this turin's model does not reproduce "
                f"the fit (best log-density {stored:.4f} stored, "
                f"{replayed:.4f} now, fitted under MODEL_REV {rev}); "
                "refit it rather than replot")
            continue

        summary = {}
        rhats = []
        for row in open(summary_path).read().splitlines()[2:]:
            f = row.split(",")
            summary[f[0]] = {"median": float(f[1]), "std": float(f[2])}
            if f[5]:
                rhats.append((float(f[5]), f[0]))
        if state.done:
            _outputs.set_run_status("")
        else:
            r, p = max(rhats)
            _outputs.set_run_status(
                f"UNCONVERGED: worst R-hat {r:.4f} ({p}) at "
                f"{state.n_samples_done} draws/chain")

        log(f"[{target}] {mode}: replotting from {len(d)} saved draws "
            f"(MODEL_REV {rev}; best log-density reproduced to {miss:+.1e})")
        model, baseline, q_ml = _ml_model(lp, v_ml, collapsed)
        _write_figures(mode, target, outdir, names, phys, b_draws, lp, v_ml,
                       layout, centering, ed, baseline, log=log, ld=q_ml)
        if mode == "ttv":
            _write_ttv_products(target, outdir, phys, names, layout,
                                centering, ed, model, summary, lp, v_ml,
                                plot=True, log=log)
        done_any = True
    _outputs.set_run_status("")
    return 0 if done_any else 1


#: Exit code for a run that finished but left a fit unconverged at the draw
#: cap; its products are complete and stamped UNCONVERGED on line 2.
EXIT_UNCONVERGED = 3


def _report_outcome(target, outcome, log):
    """One verdict line per fit; the process exit code."""
    unconverged = []
    for mode, v in outcome.items():
        if isinstance(v, str):
            log(f"[{target}] {mode}: {v}")
        elif v.converged:
            log(f"[{target}] {mode}: converged at {v.n_draws} draws/chain")
        else:
            unconverged.append(mode)
            log(f"[{target}] {mode}: NOT CONVERGED at {v.n_draws} draws/chain"
                f" (worst R-hat {v.worst_rhat[1]:.4f}, {v.worst_rhat[0]}); "
                "products are written and stamped UNCONVERGED")
    return EXIT_UNCONVERGED if unconverged else 0


def _mode_layout(mode, args, prepared, centering, epoch_data):
    """The parameter layout a fit mode samples, and (TTV) its timing half-width."""
    if mode == "ttv":
        if args.ttv_max_days:
            tau_half = np.full(epoch_data["n_epochs"], args.ttv_max_days)
        else:
            lo, hi = _seeding._epoch_time_extent(epoch_data)
            tau_half = np.maximum(
                np.minimum(-lo, hi) - 0.55 * (prepared.eph["duration"] / 24.0),
                2.0 * prepared.cadence_days)
        return _params.ttv_layout(prepared.eph, centering, tau_half,
                                  b_prior=args.b_prior,
                                  law=getattr(args, "law", "quadratic"),
                                  rhostar=_rhostar(args)), tau_half
    return _params.lineph_layout(prepared.eph, b_prior=args.b_prior,
                                 ld=getattr(args, "ld", "sampled"),
                                 law=getattr(args, "law", "quadratic"),
                                 oblate=getattr(args, "oblate", False),
                                 rhostar=_rhostar(args)), None


def transit_snrs(prepared, centering, shape, *, b_prior="transiting",
                 law="quadratic"):
    """Expected detection SNR of each epoch's transit, ``sqrt(sum (d/sigma)^2)``.

    ``d`` is the transit's own depth at each real point (the model with its
    baseline divided out) for ``shape`` -- a dict of ``k, beta, T14`` and
    ``law``'s limb-darkening coordinates -- on the input ephemeris, and ``sigma`` the point's error. This is
    the matched-filter SNR the epoch would have on noiseless data: it counts
    the data actually present, so a transit cut by a gap or a window edge
    scores lower, and one that fell in a gap scores 0. Returns
    ``(n_epochs,)``.
    """
    ed = prepared.epoch_data
    layout = _params.lineph_layout(prepared.eph, b_prior=b_prior, law=law)
    # the baseline is divided out, so its order is irrelevant: constant
    orders = np.zeros(ed["n_epochs"], dtype=int)
    _, _, lp, _ = _likelihood.build_target(
        layout, centering, ed, orders, profile_mode="exact",
        num_resample=prepared.num_resample,
        exposure_time=prepared.exposure_time, n_chains_hint=4, fp64=False)
    v = np.array([0.0, 0.0] + [float(shape[n]) for n in layout.names[2:]])
    model, base, _ = _ml_model(lp, v, False)
    mask = np.asarray(ed["mask"]) > 0
    dev = np.where(mask, model / np.where(base != 0, base, 1.0) - 1.0, 0.0)
    err = np.asarray(ed["ferr_padded"], dtype=np.float64)
    return np.sqrt(np.sum(np.where(mask, (dev / err) ** 2, 0.0), axis=1))


def _ttv_snr_gate(args, prepared, centering, outdir, lineph_ml, *, log,
                  outcome):
    """Skip the TTV fit when a single transit is too weak to time.

    The statistic is the median of :func:`transit_snrs` over epochs with
    data in transit, at the LinEph best fit (this run's, or the saved
    lineage's). Below ``--TTVsnr`` (default 3) the per-epoch times come back
    close to their priors, the sampler spends hours on near-prior multimodal
    timing (KOI-4926.01: median 2.6, 10.4 h, unconverged at the cap), and
    the freed times let the shape fit noise (its T14 came back 12.3 h
    against LinEph's 6.8 h). Writes ``<target>_ttv_skipped.csv`` and records
    the reason as the mode's outcome. Returns True when skipping.
    """
    target = prepared.target
    note = _outputs.ttv_skipped_path(outdir, target)
    threshold = float(getattr(args, "ttv_snr_min", TTV_SNR_MIN) or 0.0)
    if threshold <= 0:
        return False
    shape = lineph_ml
    if not shape:
        saved = _outputs.load_resume(outdir, target, "lineph")
        shape = saved.ml_params if saved is not None else None
    law = getattr(args, "law", "quadratic")
    need = ("k", "beta", "T14") + _model_ld_names(law)
    if not shape or not all(n in shape for n in need):
        log("  per-transit SNR gate not evaluated: no LinEph best fit for "
            "this lineage (run --modes=lineph first, or --TTVsnr=0 to "
            "silence this)")
        return False
    snr = transit_snrs(prepared, centering, shape, b_prior=args.b_prior,
                       law=law)
    covered = snr[snr > _SNR_COVERED]
    if covered.size == 0:
        typical = 0.0
    else:
        typical = float(np.median(covered))
    n_arr = np.asarray(centering["n_arr"], dtype=int)
    summary = (f"median expected SNR of a single transit {typical:.2f} over "
               f"{covered.size} of {snr.size} epochs with data in transit "
               f"(range {covered.min() if covered.size else 0:.2f}-"
               f"{covered.max() if covered.size else 0:.2f}; all transits "
               f"together {float(np.sqrt(np.sum(snr ** 2))):.1f})")
    if typical >= threshold:
        log(f"  per-transit SNR: {summary}; >= {threshold:g}, fitting")
        if os.path.exists(note):
            os.remove(note)               # a stale note from a skipped run
        return False
    reason = (f"skipped: {summary}, below --TTVsnr={threshold:g}; the "
              "transit times would be close to their priors. LinEph is the "
              "result for this target; --TTVsnr=0 fits TTVs anyway")
    log(f"  per-transit SNR: {summary}")
    log(f"  TTV fit SKIPPED: below --TTVsnr={threshold:g}. The LinEph fit "
        "is this target's result; --TTVsnr=0 fits anyway")
    _outputs.export_ttv_skipped(outdir, target, reason, n_arr, snr, log=log)
    outcome["ttv"] = reason
    return True


#: --TTVsnr's default. A single transit at SNR 3 times to roughly a third of
#: its duration; below it the timing posteriors are close to their priors.
TTV_SNR_MIN = 3.0

#: An epoch counts as having data in transit above this expected SNR (a
#: transit that fell entirely in a gap scores exactly 0).
_SNR_COVERED = 1e-3


def _run_mle(args, prepared, cv, outdir, *, log):
    """``--mle``: a maximum-likelihood LinEph fit (:mod:`turin.mle`), and
    under ``--oblate`` the round model too, with their comparison. No
    sampling, no resume state: products are ``{target}_lineph_mle_*``."""
    from . import mle as _mle

    target = prepared.target
    epoch_data = prepared.epoch_data
    centering = _prep.centering_constants(epoch_data, prepared.eph)
    orders = np.asarray(cv["orders"])
    _outputs.set_run_status("maximum likelihood")
    log(f"[{target}] lineph maximum likelihood: {epoch_data['n_epochs']} "
        f"epochs, {int(epoch_data['mask'].sum())} points; the likelihood "
        f"alone, limb darkening ({args.law}) profiled")
    build_kw = dict(num_resample=prepared.num_resample,
                    exposure_time=prepared.exposure_time,
                    geometry=args.geometry, n_chains_hint=_mle.N_STARTS)
    fit_kw = dict(law=args.law, b_prior=args.b_prior, build_kw=build_kw,
                  seed=args.seed, log=log)
    if args.oblate:
        # round first: every oblate start is its optimum at f ~ 0 (nested)
        rnd = _mle.fit(prepared, centering, epoch_data, orders, oblate=False,
                       **fit_kw)
        fits = [_mle.fit(prepared, centering, epoch_data, orders, oblate=True,
                         round_fit=rnd, **fit_kw), rnd]
    else:
        fits = [_mle.fit(prepared, centering, epoch_data, orders,
                         oblate=False, **fit_kw)]
    for f in fits:
        _log_mle(f, log)
        if f.top_spread > _seeding.TOP_SPREAD_WARN:
            log(f"    WARNING: the best starts disagree by "
                f"{f.top_spread:.1f} log-units — more than one basin")
    main = fits[0]
    _outputs.export_mle(outdir, target, "fit", main, log=log)

    lay = main.layout
    v_theta = main.v_model[:lay.dim]
    model, baseline, q_ml = _ml_model(main.lp, v_theta, collapsed=True)
    _outputs.export_lcdata(outdir, target, _mle.MODE, epoch_data, model,
                           log=log)
    _fold_figure("lineph", target, outdir, main.lp, v_theta, lay, centering,
                 epoch_data, baseline, log=log, ld=q_ml, stem=_mle.MODE)

    if args.oblate:
        _outputs.export_mle(outdir, target, "round", fits[1], log=log)
        cmp = _mle.compare(main, fits[1])
        _outputs.export_mle_compare(outdir, target, cmp, log=log)
        log(f"  oblate vs round: dlnL = {cmp['dlnL']:.3f}, "
            f"dBIC = {cmp['dBIC']:.2f}, dAIC = {cmp['dAIC']:.2f} "
            f"(negative favours oblate; no p-value: at f = 0 theta is "
            f"undefined and f is on its boundary)")
        if cmp["aligned_central"]:
            log("  WARNING: the oblate optimum is an aligned oblate planet "
                "on a central chord (b < 0.1, |theta| < 15 deg, f > 0.05), "
                "the family that mimics a round planet at larger b; "
                "compare a --rhostar posterior before reading f as a "
                "detection")
    return 0


def _log_mle(fit, log):
    log(f"  {fit.label}: ln L = {fit.lnL:.3f}, BIC = {fit.bic:.2f} "
        f"({fit.n_params} parameters incl. baseline, {fit.n_points} points)")
    for name, val in fit.rows:
        log(f"    {name:>10s} = {val:.10g}")
    for note in fit.notes:
        log(f"    note: {note}")


def _fit_mode(mode, args, prepared, cv, outdir, *, log, outcome,
              lineph_ml=None):
    """Run one fit mode. Returns the ML shape dict for LinEph, else None."""
    target = prepared.target
    epoch_data = prepared.epoch_data
    centering = _prep.centering_constants(epoch_data, prepared.eph)
    orders = np.asarray(cv["orders"])

    n_ep = epoch_data["n_epochs"]
    log(f"[{target}] {mode}: {n_ep} epochs, "
        f"{int(epoch_data['mask'].sum())} points, "
        f"{prepared.num_resample} sub-exposures")

    # resume state and its guards
    prior_state = None if args.fresh else _outputs.load_resume(
        outdir, target, mode)
    # before the resume checks: a fit that will not run needs none of them
    # (an unfinished TTV lineage from an older MODEL_REV would refuse first).
    # A converged one still reports as converged below.
    finished = (prior_state is not None and prior_state.done
                and not args.extend2)
    if mode == "ttv" and not finished and _ttv_snr_gate(
            args, prepared, centering, outdir, lineph_ml, log=log,
            outcome=outcome):
        return None
    if prior_state is not None:
        prior_state.check_model_rev(log=log)
        prior_state.check(
            b_prior=args.b_prior, geometry=args.geometry,
            sampler=args.sampler, ttv_max=args.ttv_max_days,
            # "auto" is a request to measure, not a model choice, so it can
            # never conflict with what a previous run resolved to
            profile_mode=(None if args.profile_mode == "auto"
                          else args.profile_mode),
            gibbsgrid=_gibbsgrid(mode, args),
            ld=getattr(args, "ld", "sampled"),
            law=getattr(args, "law", "quadratic"),
            oblate=getattr(args, "oblate", False),
            rhostar=_rhostar(args))
        want_extend = args.extend1 if mode == "lineph" else args.extend2
        if prior_state.done and not want_extend:
            log(f"  {mode} already converged "
                f"({prior_state.n_samples_done} draws/chain); skipping. "
                "Use --extend1/--extend2 to add more, or --fresh to redo.")
            outcome[mode] = "converged earlier"
            return prior_state.ml_params or None
        log(f"  resuming from {prior_state.n_samples_done} draws/chain")

    layout, tau_half = _mode_layout(mode, args, prepared, centering,
                                    epoch_data)

    build_kw = dict(num_resample=prepared.num_resample,
                    exposure_time=prepared.exposure_time,
                    geometry=args.geometry, n_chains_hint=args.chains)

    # A resumed run keeps the mode it was sampled under: a continuation must
    # not silently change its own likelihood. An explicit --PL wins otherwise.
    if layout.ld == "collapsed":
        # not a choice: the omega gradient and Hessian need the exact profile
        if prior_state is not None and prior_state.profile_mode != "exact":
            raise SystemExit(
                f"turin: this collapsed lineage records "
                f"profile_mode={prior_state.profile_mode!r}, which collapsed "
                f"limb darkening cannot have been sampled under; use --fresh")
        pl_choice = _plselect.fixed_choice(
            "exact", "required by --ld=collapsed: the omega gradient and "
            "Hessian come from the envelope theorem, which needs the exact "
            "flux-space profile")
    elif layout.oblate and args.profile_mode == "auto":
        pl_choice = _plselect.fixed_choice(
            "exact", "--oblate: probing the solves would cost dozens of "
            "evaluations of the oblate kernel (~40-85x a round one)")
    elif prior_state is not None and args.profile_mode == "auto":
        pl_choice = _plselect.fixed_choice(
            prior_state.profile_mode, "carried over from the resumed run")
    elif args.profile_mode != "auto":
        pl_choice = _plselect.fixed_choice(args.profile_mode)
    else:
        pl_choice = None                     # probed below, once the MAP exists

    provisional = pl_choice.mode if pl_choice else "exact"
    # collapsed hybrid4: a resumed lineage rebuilds its stored GHK rule; a
    # fresh one starts on the default layout and freezes its own at the MAP
    build_kw.update(_ghk_kw(layout, prior_state))
    target_obj, transform, lp, hi = _likelihood.build_target(
        layout, centering, epoch_data, orders,
        profile_mode=provisional, ld_mode=layout.ld, **build_kw)
    log(f"  log-density: dim {layout.dim}, "
        f"{len(lp.blocks)} epoch block(s) of <= {lp.block_size}")
    if layout.rhostar:
        m, sd = layout.rhostar
        log(f"  prior: log10(rho*/kg m^-3) ~ N({m:g}, {sd:g}), through the "
            f"fitted a/R* (rho* known to ~{100 * math.log(10) * sd:.0f}%)")
    if layout.ld == "collapsed":
        log(f"  limb darkening ({layout.law}): collapsed -- integrated out of "
            f"the log-density; {', '.join(_model_ld_names(layout.law))} drawn "
            "afterwards from their exact conditional")
    if prepared.exposure_time > 0:
        from .model import HAS_TAU_KERNEL, N_GL

        if HAS_TAU_KERNEL and args.geometry == "circular":
            log(f"  exposure integration: contact rule in-kernel, "
                f"n_gl={N_GL} ({5 * N_GL} evaluations per point)")
        else:
            log(f"  exposure integration: supersampling, "
                f"{prepared.num_resample} nodes per point"
                + ("" if args.geometry == "circular"
                   else " (chord geometry has no kernel path)"))
    if pl_choice is not None:
        log(pl_choice.describe())

    # ---- initialization
    shape0 = dict(lineph_ml) if (mode == "ttv" and lineph_ml) else None
    seeds = None
    if mode == "ttv":
        s = shape0 or _seeding.default_shape_start(prepared.eph, layout)
        from .model import impact_parameter

        b0 = float(impact_parameter(s["beta"], s["k"], layout.b_prior))
        log("  seeding transit times by template sweep...")
        seeds, rival_gap, _, _ = _seeding.template_sweep_taus(
            epoch_data, centering, lp.design, tau_half=tau_half,
            k=s["k"], b=b0, T14=s["T14"],
            q=tuple(s[n] for n in layout.ld_names), law=layout.law,
            period=layout.P_ref, num_resample=prepared.num_resample,
            exposure_time=prepared.exposure_time,
            profile_mode=provisional, geometry=args.geometry)
        weak = np.where(rival_gap < _seeding.RIVAL_GAP_WARN)[0]
        if weak.size:
            log(f"  note: {weak.size} epoch(s) have a competing timing mode "
                "more than one T14 from the seed"
                + ("; grid-Gibbs samples across them"
                   if _gibbsgrid(mode, args) == "on" else
                   "; with --gibbsgrid=off their times are shape-sensitive"))
            for i in weak[:6]:
                head = (f"    epoch {int(centering['n_arr'][i]):+d}: "
                        f"seed {seeds[i] * 1440:+.1f} min, ")
                if rival_gap[i] >= 0:
                    log(head + f"best rival {rival_gap[i]:.1f} log-units "
                        "below it")
                else:
                    # the seeder takes the interior peak nearest the
                    # prediction among those within TIE_LOGL of the best,
                    # so a higher score elsewhere is either such a peak or
                    # a window edge / slope it does not treat as a transit
                    log(head + f"a point more than one T14 away scores "
                        f"{-rival_gap[i]:.1f} log-units higher (a near-tie "
                        "peak further from the prediction, or a window "
                        "edge)")

    v0 = _seeding.initial_model_vector(layout, prepared.eph, shape=shape0,
                                       dtau=seeds)
    cfg = _sampling.SamplerConfig(
        sampler=args.sampler, n_chains=args.chains, n_warmup=args.warmup,
        n_samples=args.samples, max_samples=args.max_samples,
        max_leapfrog=args.max_leapfrog, seed=args.seed).for_mode(layout.dim)

    resume_for_anvil = None
    if prior_state is not None and \
            prior_state.anvil_state_path and \
            os.path.exists(prior_state.anvil_state_path):
        import anvil

        resume_for_anvil = anvil.load_state(prior_state.anvil_state_path)
        u0 = None
        log("  continuing anvil's adapted state")
    elif prior_state is not None and prior_state.last_u is not None:
        u0 = mx.array(np.asarray(prior_state.last_u, dtype=np.float32))
        log(f"  restarting from the previous {u0.shape[0]} final positions")
    else:
        rng = np.random.default_rng(args.seed)
        log("  finding the MAP from 128 starts (anvil.optimize)...")
        u_starts = (transform.from_model_np(v0)[None, :]
                    + 0.25 * rng.standard_normal((128, layout.dim)))
        map_res = _seeding.find_map(target_obj, u_starts, seed=args.seed)
        n_conv = int(np.sum(map_res.status == "converged"))
        log(f"    MAP log-density {map_res.value:.3f} in "
            f"{map_res.wall_time:.1f}s ({n_conv}/{map_res.n_starts} starts "
            "converged)")
        spread = _seeding.top_spread(map_res)
        if spread > _seeding.TOP_SPREAD_WARN:
            log(f"    WARNING: the best starts disagree by {spread:.1f} "
                "log-units — the posterior probably has more than one basin")
        if pl_choice is None:
            # Probe at the MAP: the ball is scaled to the posterior's own
            # width, so the modes are compared where sampling happens.
            pl_choice = _plselect.select_pl_mode(
                layout, centering, epoch_data, orders, map_res.u,
                transform, build_kwargs=build_kw, n_chains=cfg.n_chains,
                seed=args.seed, log=log)
        u0 = map_res.ball(cfg.n_chains, seed=args.seed + 1)
        if layout.oblate:
            # chains centred off f = 0 (elome's fp32 d/df is 0 below 1e-5)
            # and spread over the Laplace approximation once the final
            # target exists, below
            u_center = _seeding.floor_oblateness(map_res.u, transform,
                                                 layout)
            if not np.array_equal(u_center, map_res.u):
                log(f"    MAP at f < {_seeding.F_FLOOR:g}: chains centred at "
                    f"f = {_seeding.F_FLOOR:g} instead")
            map_res = dataclasses.replace(map_res, u=u_center)
            u0 = None
        if layout.ld == "collapsed" and layout.law == "hybrid4":
            # GHK's layout is frozen once per fit, at the MAP, and rebuilt
            # into the target; the init ball carries over (same parameters)
            from . import ldhybrid as _ldh

            v_map = transform.model_np(np.asarray(map_res.u)[None, :])[0]
            frozen = _ldh.freeze_layout(hi if hi is not None else lp, v_map)
            build_kw.update(ghk_layout=frozen, ghk_points=lp.ghk_points)
            target_obj, transform, lp, hi = _likelihood.build_target(
                layout, centering, epoch_data, orders,
                profile_mode=provisional, ld_mode=layout.ld, **build_kw)
            log(f"    GHK layout frozen at the MAP: dependent weight "
                f"{frozen[0]}, order {list(frozen[1])} "
                f"({lp.ghk_points.shape[0]} points)")

    if pl_choice is None:                    # resumed without a MAP step
        pl_choice = _plselect.fixed_choice(
            provisional, "carried over from the resumed run")
    if pl_choice.mode != provisional:
        # the unconstrained space is identical across modes, so the MAP and
        # its init ball carry over unchanged
        target_obj, transform, lp, hi = _likelihood.build_target(
            layout, centering, epoch_data, orders,
            profile_mode=pl_choice.mode, ld_mode=layout.ld, **build_kw)

    if layout.oblate and u0 is None and resume_for_anvil is None:
        u0, n_flat = _seeding.chain_starts(map_res, cfg.n_chains,
                                           seed=args.seed + 1)
        log("    chains spread over the Laplace approximation at the MAP"
            + (" (Hessian not usable: a 1e-3 ball instead)" if n_flat is None
               else f" ({n_flat} flat direction(s) capped)"))
    if u0 is not None:
        _sampling.check_precision(target_obj, u0, log=log,
                                  strict=not args.no_strict_precision)

    # ---- sample, exporting after every round
    state_holder = {}

    # The chains tarball and the PDFs are the expensive products; they are
    # written after round 0 (so a long run shows something early), then at
    # most every HEAVY_EVERY_S, and always on the final round. The small
    # products and the resume state are written every round.
    last_heavy = {"t": None}

    def on_round(results, verdict, rnd):
        # the round loop stops on convergence or at the draw cap, so either
        # makes this the final round: always written in full, with the
        # outcome stamped on every product
        final = verdict.converged or verdict.n_draws >= cfg.max_samples
        if verdict.converged:
            _outputs.set_run_status("")
        elif final:
            _outputs.set_run_status(
                f"UNCONVERGED: worst R-hat {verdict.worst_rhat[1]:.4f} "
                f"({verdict.worst_rhat[0]}) at the {cfg.max_samples} "
                "draws/chain cap")
        else:
            _outputs.set_run_status(
                f"in progress: round {rnd}, not yet converged")
        heavy = (final or last_heavy["t"] is None
                 or time.perf_counter() - last_heavy["t"] >= HEAVY_EVERY_S)
        _export_all(mode, args, prepared, epoch_data, centering, orders,
                    layout, transform, lp, results, verdict, outdir,
                    log=log, state_holder=state_holder, cfg=cfg,
                    pl_mode=pl_choice.mode, heavy=heavy)
        if heavy:
            last_heavy["t"] = time.perf_counter()
        outcome[mode] = verdict

    move = None
    if _gibbsgrid(mode, args) == "on":
        from . import gibbs as _gibbs

        # its own float32 log-density, blocked for the large batches a
        # sweep evaluates; same layout and profile mode as the sampler's
        _, _, lp_gibbs, _ = _likelihood.build_target(
            layout, centering, epoch_data, orders,
            profile_mode=pl_choice.mode, fp64=False,
            **dict(build_kw, n_chains_hint=4096))
        move = _gibbs.GridGibbs(lp_gibbs, transform, layout,
                                T14=prepared.eph["duration"] / 24.0,
                                batch=4096, seed=args.seed)

    results, verdict, total = _sampling.run_rounds(
        target_obj, list(layout.names), u0, cfg, log=log, on_round=on_round,
        resume_state=resume_for_anvil, move=move)

    cert = _sampling.certify(target_obj, results, list(layout.names), log=log)
    if cert is not None:
        state_holder["certify"] = str(cert)

    ml = state_holder.get("ml_params")
    return ml if mode == "lineph" else None


def _write_ttv_products(target, outdir, phys, names, layout, centering,
                        epoch_data, model, summary, lp, v_ml, *, plot, log):
    """ttv_times.csv, and (``plot``) the O-C diagram."""
    rows = _ttv_rows(phys, names, layout, centering, epoch_data, model,
                     summary, lp=lp, v_ml=v_ml)
    _outputs.export_ttv_times(outdir, target, "ttv", rows, log=log)
    if plot:
        try:
            _plots.oc_plot(
                _outputs.product_path(outdir, target, "ttv", "oc", "pdf"),
                [r["epoch"] for r in rows], [r["tmid"] for r in rows],
                [r["tmid_err"] for r in rows],
                title=f"{target} transit timing", log=log)
        except Exception as exc:
            log(f"    O-C plot skipped: {exc}")


def _corner_columns(names):
    """Which parameters a corner plot shows, and where its "..." gap goes.

    Every shape parameter and every transit time; past 2 * CORNER_EPOCHS
    transit times, the first and last CORNER_EPOCHS with a "..." row and
    column between them (5 + 7 + 1 + 7 = 20 panels for a TTV fit). With
    exactly 2 * CORNER_EPOCHS times nothing is left out, so there is no gap.
    Returns ``(column indices, gap_after)``, ``gap_after`` being the position
    in that list after which the gap goes, or None.
    """
    shape = [i for i, n in enumerate(names) if not n.startswith("dtau_")]
    times = [i for i, n in enumerate(names) if n.startswith("dtau_")]
    if len(times) <= 2 * CORNER_EPOCHS:
        return shape + times, None
    times = times[:CORNER_EPOCHS] + times[-CORNER_EPOCHS:]
    return shape + times, len(shape) + CORNER_EPOCHS - 1


def _write_figures(mode, target, outdir, names, phys, b_draws, lp, v_ml,
                   layout, centering, epoch_data, baseline, *, log, ld=None):
    """Corner and fold PDFs, from the (subsampled) physical draws."""
    try:
        # show the impact parameter b itself, not the sampled coordinate
        # beta = b / b_max(k), which is a prior device, not a physical
        # quantity
        cols, gap_after = _corner_columns(names)
        c_names = ["b" if names[i] == "beta" else names[i] for i in cols]
        c_draws = phys[:, cols].copy()
        if "beta" in names:
            c_draws[:, c_names.index("b")] = b_draws
        _plots.corner_plot(
            _outputs.product_path(outdir, target, mode, "corner", "pdf"),
            c_draws, [_label(n) for n in c_names],
            title=f"{target} {mode}", gap_after=gap_after, log=log)
    except Exception as exc:
        log(f"    corner plot skipped: {exc}")

    _fold_figure(mode, target, outdir, lp, v_ml, layout, centering,
                 epoch_data, baseline, log=log, ld=ld)


def _fold_figure(mode, target, outdir, lp, v_ml, layout, centering,
                 epoch_data, baseline, *, log, ld=None, stem=None):
    """The fold PDF at one parameter vector. ``stem`` names the product
    (``mode`` by default; ``--mle`` writes ``lineph_mle``)."""
    stem = stem or mode
    mid_abs = _mid_times_absolute(mode, v_ml, layout, centering)
    try:
        # span the folded data itself, whatever the window width
        real = np.asarray(epoch_data["mask"]) > 0
        offsets = (np.asarray(epoch_data["times_padded"], dtype=np.float64)
                   - mid_abs[:, None])[real]
        tt, tf = _plots.model_grid(lp, v_ml, T14=v_ml[layout.index("T14")],
                                   half_span=1.001 * np.max(np.abs(offsets)),
                                   ld=ld)
        _plots.fold_plot(
            _outputs.product_path(outdir, target, stem, "fold", "pdf"),
            epoch_data=epoch_data, mid_times=mid_abs, baseline=baseline,
            model_grid_t=tt, model_grid_f=tf,
            period=layout.P_ref + (v_ml[0] if mode == "lineph" else 0.0),
            title=f"{target} {stem}", colour=_plots.MODE_COLOURS[mode],
            n_bins_from=epoch_data["n_epochs"], log=log)
    except Exception as exc:
        log(f"    fold plot skipped: {exc}")


def _ml_model(lp, v_ml, collapsed):
    """Detrended model and baseline at the ML parameters (model space).

    Returns ``(model, baseline, q_ml)``; ``q_ml`` is the collapsed mode's
    display limb darkening (the conditional mode at the ML theta), else None.
    """
    q_ml = None
    with mx.stream(mx.cpu):
        v_row = mx.array(np.asarray(v_ml)[None, :].astype(np.float32))
        if collapsed:
            # one curve needs one limb darkening: the conditional mode at the
            # ML theta. A display point estimate only; the posterior q's in
            # every other product are the conditional draws.
            q_ml = tuple(float(np.asarray(a)[0])
                         for a in lp.conditional_mode_ld(v_row))
        model, coeffs = lp.full_model(v_row, ld=q_ml)
        model = np.array(model, dtype=np.float64)[0]
        baseline = np.array(_profile.baseline(lp.design, coeffs),
                            dtype=np.float64)[0]
    return model, baseline, q_ml



def _ld_weight_draws(layout, phys, names):
    """``(w names, (n, n_w) draws)`` of a non-quadratic law's physical
    weights, from its sampled ``q``; ``([], None)`` for the quadratic law,
    whose ``q1, q2`` are the reported coordinates as in hurin."""
    if layout.law == "quadratic":
        return [], None
    from .model import ld_names, ld_weights_np

    qn = ld_names(layout.law)
    q = np.column_stack([phys[:, names.index(n)] for n in qn])
    return [f"w{j + 1}" for j in range(len(qn))], ld_weights_np(q, layout.law)

def _oblate_draws(layout, phys, names):
    """``(names, (n, 4) draws)`` of an oblate planet's reported shape: ``f``,
    ``theta`` (degrees, on [-90, 90]) and the semi-axes ``A = k/sqrt(1-f)``,
    ``B = A (1 - f)``, from the sampled ``fc, fs`` and ``k``; ``([], None)``
    for a round one."""
    if not layout.oblate:
        return [], None
    f, theta = _params.oblate_from_fcfs_np(phys[:, names.index("fc")],
                                           phys[:, names.index("fs")])
    A = phys[:, names.index("k")] / np.sqrt(1.0 - f)
    return ["f", "theta", "A", "B"], np.column_stack([f, theta, A, A * (1 - f)])


#: SquishierPlanet's false oblate mode: a family of aligned oblate planets
#: (theta ~ 0, f ~ 0.19) on a central chord (b ~ 0), 2 log-likelihood units
#: worse than a round truth at b = 0.3 but broad enough to hold ~20% of the
#: mass under a uniform-in-f prior. Its posterior share is always logged
#: under --oblate; above ALIGNED_WARN_SHARE it is a warning.
ALIGNED_B_MAX = 0.1
ALIGNED_THETA_MAX = 15.0
ALIGNED_F_MIN = 0.05
ALIGNED_WARN_SHARE = 0.05


def aligned_central_share(f, theta_deg, b):
    """Share of draws in the aligned-oblate-on-a-central-chord family."""
    f, th, b = (np.asarray(x, dtype=np.float64) for x in (f, theta_deg, b))
    return float(np.mean((b < ALIGNED_B_MAX)
                         & (np.abs(th) < ALIGNED_THETA_MAX)
                         & (f > ALIGNED_F_MIN)))


def _ml_row(results, lp, transform):
    """The maximum-likelihood draw over every draw, in model space.

    Indexes the one winning draw rather than flattening the whole chain to
    float64 (540 MB at the draw cap for dim 8).
    """
    lpv = np.asarray(results.get_log_prob(), dtype=np.float64)  # (S, C)
    s_i, c_i = np.unravel_index(int(np.argmax(lpv)), lpv.shape)
    row = np.asarray(results.get_chain()[s_i, c_i], dtype=np.float64)
    return transform.model_np(row[None, :])[0], float(lpv[s_i, c_i])


def _export_all(mode, args, prepared, epoch_data, centering, orders, layout,
                transform, lp, results, verdict, outdir, *, log,
                state_holder, cfg, pl_mode, heavy=True):
    """Write this mode's products. Called after each sampling round.

    Everything is computed from a bounded, systematic subsample of the pooled
    draws (``sampling.export_thin``); R-hat and ESS in the summary come from
    ``verdict``, which saw every draw. ``heavy=False`` skips the chains
    tarball and the PDFs -- the expensive products -- and writes the small
    ones and the resume state, which are what make a run interruptible.
    """
    target = prepared.target
    names_s = list(layout.names)            # what was sampled
    names = names_s
    collapsed = layout.ld == "collapsed"
    thin = _sampling.export_thin(
        results, max_rows=_EXPORT_ROWS_COLLAPSED_HYBRID
        if collapsed and layout.law == "hybrid4" else _sampling.EXPORT_MAX_ROWS)
    ld_cols = list(_model_ld_names(layout.law))
    if collapsed:
        # collapsed Gibbs: the limb darkening was integrated out of the
        # sampled target, so each kept draw gets it from its exact
        # conditional here. The sampler needs model-space draws (physical
        # ones carry dP/dtau0's report offsets).
        flat = results.get_chain(thin=thin, flat=True).astype(np.float64)
        v_model = transform.model_np(flat)
        if layout.law == "hybrid4":
            from . import ldhybrid as _ldh

            q, acc, edge = _ldh.draw_limb_darkening(lp, v_model,
                                                    seed=args.seed)
            where = "a face of the weight simplex"
        else:
            from . import ldmarg as _ldmarg

            q1, q2, acc, edge = _ldmarg.draw_limb_darkening(lp, v_model,
                                                            seed=args.seed)
            q = np.column_stack([q1, q2])
            where = "a triangle edge"
        phys = np.column_stack([transform.to_physical(v_model), q])
        names = names_s + ld_cols
        log(f"  limb darkening: {len(q)} conditional draws of "
            f"{', '.join(ld_cols)} (MH acceptance {acc:.2f}; expansion point "
            f"on {where} for {100 * edge:.0f}% of draws)")
    else:
        phys = _sampling.physical_draws(transform, results, thin=thin)

    # anvil reports rank-normalized bulk ESS only (no tail ESS), so hurin's
    # Tail_ESS column is written empty rather than filled with a placeholder.
    # R-hat/ESS exist only for sampled parameters; collapsed q1, q2 are exact
    # conditional draws, so their rows leave the diagnostics blank, as the
    # derived b and log10_rho rows do.
    summary = _outputs.summarize(phys[:, :len(names_s)], names_s,
                                 rhat=verdict.rhat, ess_bulk=verdict.ess,
                                 ess_tail=None)
    if collapsed:
        summary.update(_outputs.summarize(phys[:, len(names_s):], ld_cols))

    # derived quantities, computed after sampling in float64
    b_draws = _outputs.derived_b(phys, names, layout.b_prior)
    rho = _outputs.log10_rho_draws(
        phys, names, layout.b_prior,
        P_ref=None if mode == "lineph" else layout.P_ref)
    derived = _outputs.summarize(
        np.column_stack([b_draws, rho]), ["b", "log10_rho"])
    summary.update(derived)
    # hybrid4: the physical shape weights w, derived from the sampled q
    # (uniform on the cube -> uniform on the simplex), reported beside them
    w_names, w_draws = _ld_weight_draws(layout, phys, names)
    if w_names:
        summary.update(_outputs.summarize(w_draws, w_names))
    # oblate: f, theta (degrees) and the semi-axes, derived from fc, fs, k
    ob_names, ob_draws = _oblate_draws(layout, phys, names)
    if ob_names:
        summary.update(_outputs.summarize(ob_draws, ob_names))
        f16, f50, f84, f95 = np.percentile(ob_draws[:, 0], [16, 50, 84, 95])
        log(f"  oblateness: f median {f50:.4f} (16-84%: {f16:.4f}-{f84:.4f}), "
            f"95% upper limit {f95:.4f}; prior uniform in f on [0, "
            f"{_params.F_MAX:g}] and in theta; T14 is the area-equivalent "
            "circle's duration")
        share = aligned_central_share(ob_draws[:, 0], ob_draws[:, 1], b_draws)
        log(f"  aligned oblate on a central chord (b < {ALIGNED_B_MAX:g}, "
            f"|theta| < {ALIGNED_THETA_MAX:g} deg, f > {ALIGNED_F_MIN:g}): "
            f"{100 * share:.1f}% of draws")
        if share > ALIGNED_WARN_SHARE:
            log(f"  WARNING: {100 * share:.0f}% of the posterior is an "
                "aligned oblate planet on a central chord, which mimics a "
                "round planet at larger b (SquishierPlanet: ~20% of the mass "
                "for a round planet at b = 0.3 with no density prior). Read "
                "it as the b-f-theta degeneracy, not a detection, unless a "
                "stellar-density prior keeps it: "
                + ("this fit has one, at "
                   f"{100 * math.log(10) * layout.rhostar[1]:.0f}%"
                   if layout.rhostar else
                   "try --rhostar=MEAN,SIGMA (log10 kg m^-3)"))

    _outputs.export_summary(outdir, target, mode, summary, log=log)
    _outputs.export_logrho(outdir, target, mode, rho, log=log)

    if heavy:
        cols = (list(names) + ["b", "log10_rho"] + w_names + ob_names
                + ["loglike"])
        arrays = [phys[:, i] for i in range(phys.shape[1])] + [
            b_draws, rho] + [w_draws[:, j] for j in range(len(w_names))] + [
            ob_draws[:, j] for j in range(len(ob_names))] + [
            np.asarray(results.get_log_prob(thin=thin, flat=True),
                       dtype=np.float64) + lp.log_const]
        _outputs.export_chains(outdir, target, mode, cols, arrays, thin=thin,
                               n_total=results.get_log_prob().size, log=log)

    # maximum-likelihood model, for the light-curve export and the plots
    v_ml, logl_ml = _ml_row(results, lp, transform)
    model, baseline, q_ml = _ml_model(lp, v_ml, collapsed)
    v_ml_named = v_ml if q_ml is None else np.concatenate([v_ml, q_ml])
    _outputs.export_lcdata(outdir, target, mode, epoch_data, model, log=log)

    keep = ("k", "beta", "T14", "fc", "fs") + _model_ld_names(layout.law)
    state_holder["ml_params"] = {
        n: float(v_ml_named[i]) for i, n in enumerate(names) if n in keep}

    # ---- figures
    if heavy:
        _write_figures(mode, target, outdir, names, phys, b_draws, lp, v_ml,
                       layout, centering, epoch_data, baseline, log=log,
                       ld=q_ml)

    if mode == "ttv":
        _write_ttv_products(target, outdir, phys, names, layout, centering,
                            epoch_data, model, summary, lp, v_ml,
                            plot=heavy, log=log)

    # ---- resume state
    # a write failure (disk full, permissions) must not kill the fit: the
    # resume state then falls back to restarting from last_u
    anvil_state_path = _outputs.product_path(
        outdir, target, mode, "anvilstate", "npz")
    try:
        results.save_state(anvil_state_path)
    except Exception as exc:
        log(f"    anvil state not saved: {exc}")
        anvil_state_path = None

    state = _outputs.ResumeState(
        target=target, mode=mode, turin_version=__version__,
        launch_command=_outputs.launch_command(), tag=args.tag,
        b_prior=layout.b_prior, profile_mode=pl_mode,
        geometry=args.geometry, sampler=args.sampler,
        model_rev=_MODEL_REV, gibbsgrid=_gibbsgrid(mode, args),
        ld=layout.ld, law=layout.law, oblate=layout.oblate,
        rhostar=layout.rhostar,
        ghk_layout=getattr(lp, "ghk_layout", None),
        ghk_points=getattr(lp, "ghk_points", None),
        n_chains=cfg.n_chains, ttv_max=args.ttv_max_days,
        n_durations=float(prepared.n_durations),
        legendre_orders=np.asarray(orders), exposure_time=prepared.exposure_time,
        num_resample=prepared.num_resample,
        n_samples_done=int(verdict.n_draws), done=bool(verdict.converged),
        ml_params=state_holder.get("ml_params", {}),
        anvil_state_path=anvil_state_path,
        last_u=np.asarray(results.final_state["u"], dtype=np.float32),
    )
    _outputs.save_resume(outdir, target, mode, state, log=log)

    # width breakdown, if any chain looked trapped
    rows = _sampling.width_breakdown(transform, results, names_s,
                                     verdict.health)
    if rows:
        log("  widths with / without the flagged chains:")
        for name, sd_all, sd_bulk, ratio in rows:
            if ratio > 1.2:
                log(f"    {name:10s} {sd_all:.4g} vs {sd_bulk:.4g} "
                    f"({ratio:.2f}x inflated)")


def _mid_times_absolute(mode, v_ml, layout, centering):
    """Absolute mid-transit time per epoch at the ML parameters, float64."""
    n_arr = np.asarray(centering["n_arr"], dtype=np.float64)
    centers = np.asarray(centering["centers_abs"], dtype=np.float64)
    d_arr = np.asarray(centering["d_arr"], dtype=np.float64)
    if mode == "lineph":
        return centers + (v_ml[1] + n_arr * v_ml[0] + d_arr)
    return centers + (v_ml[layout.n_shape:] + d_arr)


def _epoch_snr(lp, layout, v_ml):
    """Per-epoch detection SNR at the ML parameters: sqrt(2 dlnL).

    dlnL is each epoch's profiled log-likelihood with the transit minus the
    same with k -> 0, so the null still has its own Legendre baseline fitted.
    (The old null was a flat 1.0 with no baseline, which credited the
    baseline's fit to the transit: 92 for one epoch of KOI-5616.01, whose
    catalogue SNR over all transits is 7.8.)
    """
    v = np.vstack([v_ml, v_ml]).astype(np.float64)
    v[1, layout.index("k")] = 1e-9
    with mx.stream(mx.cpu):
        L = np.array(lp.epoch_log_lik(mx.array(v.astype(np.float32))),
                     dtype=np.float64)
    return np.sqrt(np.maximum(2.0 * (L[0] - L[1]), 0.0))


def _ttv_rows(phys, names, layout, centering, epoch_data, model, summary, *,
              lp, v_ml):
    """Per-epoch timing rows for the TTV export, with fit diagnostics."""
    n_arr = np.asarray(centering["n_arr"], dtype=np.float64)
    keys = [f"dtau_{int(n)}" for n in n_arr]
    P_fit, tau0_fit, _ = _outputs.fit_linear_ephemeris(
        n_arr, [summary[k]["median"] for k in keys],
        [summary[k]["std"] for k in keys])

    mask = np.asarray(epoch_data["mask"]) > 0
    y = np.asarray(epoch_data["flux_padded"], dtype=np.float64)
    e = np.asarray(epoch_data["ferr_padded"], dtype=np.float64)
    m = np.asarray(model, dtype=np.float64)

    snrs = _epoch_snr(lp, layout, v_ml)
    rows = []
    for i, n in enumerate(n_arr):
        key = f"dtau_{int(n)}"
        tmid = summary[key]["median"]
        err = summary[key]["std"]
        sel = mask[i]
        chi2 = float(np.sum(((y[i][sel] - m[i][sel]) / e[i][sel]) ** 2))
        snr = float(snrs[i])
        oc = (tmid - (tau0_fit + P_fit * n)) * 1440.0
        rows.append(dict(epoch=int(n), tmid=tmid, tmid_err=err, ttv_min=oc,
                         ttv_err_min=err * 1440.0, snr=snr,
                         npts=int(sel.sum()), chi2=chi2))
    return rows
