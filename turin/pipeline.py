"""Fit orchestration: one target, end to end, in hurin's order.

Download and condition the light curve, cross-validate the Legendre orders,
fit the linear ephemeris, then fit per-epoch transit times seeded from the
LinEph maximum-likelihood shape. Products are re-exported after every sampling
round, so a long run is interruptible and its partial results are usable --
hurin's behaviour, and the reason its resume state exists.
"""

from __future__ import annotations

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

PARAM_LABELS = {
    "dP": "$P$ (d)", "dtau0": r"$\tau_0$", "k": "$k = R_p/R_\\star$",
    "beta": r"$b / b_{\max}$", "T14": "$T_{14}$ (d)", "q1": "$q_1$",
    "q2": "$q_2$",
}


def _label(name):
    if name in PARAM_LABELS:
        return PARAM_LABELS[name]
    if name.startswith("dtau_"):
        return rf"$\tau_{{{name[5:]}}}$"
    return name


def run(args, log=print):
    """Run every requested fit for one target. Returns a process exit code."""
    _outputs.set_run_tag(args.tag)
    target = args.target
    outdir = args.outdir or target
    os.makedirs(outdir, exist_ok=True)

    caps = _caps.detect()
    log(f"turin {__version__} — {target}")
    log(caps.summary())
    if not caps.anvil_resume:
        log("  note: without anvil resume, each extension repeats warmup and "
            "is a new chain rather than a continuation")

    if args.fresh:
        removed = _outputs.clear_products(outdir, target)
        if removed:
            log(f"  --fresh: removed {len(removed)} existing product(s)")

    t_start = time.perf_counter()
    prepared = _prep.prepare_data(
        target, ttv_max_days=args.ttv_max_days or 0.0,
        sc_override=args.sc_override, log=log)

    # window widening for declared TTVs: identical windows to a default run
    # for TTVmax <= 3.5 T14, wider only when the timing prior needs it
    T14_days = prepared.eph["duration"] / 24.0
    n_durations = _prep.N_DURATIONS
    if args.ttv_max_days:
        if args.ttv_max_days >= 0.5 * prepared.eph["period"]:
            raise SystemExit(
                f"turin: --TTVmax={args.ttv_max_min} min is at least half the "
                f"period ({prepared.eph['period']:.4f} d); the epochs would "
                "overlap")
        n_durations = max(n_durations, args.ttv_max_days / T14_days + 1.5)
        if n_durations * T14_days > 0.45 * prepared.eph["period"]:
            log(f"  warning: windows of {n_durations:.1f} durations span more "
                "than 45% of the period; epochs may be poorly separated")
        log(f"  --TTVmax widened the windows to {n_durations:.2f} durations")

    log(f"[{target}] Cross-validating Legendre orders...")
    cv = _prep.optimize_legendre_orders(
        prepared.time, prepared.flux, prepared.flux_err,
        prepared.eph["period"], prepared.eph["epoch"],
        prepared.eph["duration"], n_durations=n_durations,
        tau_shift_max=args.ttv_max_days or 0.0,
        log=lambda m: log(f"  {m}"))

    lineph_ml = None
    for mode in args.modes:
        try:
            lineph_ml = _fit_mode(
                mode, args, prepared, cv, outdir, caps, log=log,
                n_durations=n_durations, lineph_ml=lineph_ml) or lineph_ml
        except SystemExit:
            raise
        except Exception as exc:
            log(f"[{target}] {mode} FAILED: {type(exc).__name__}: {exc}")
            raise

    log(f"[{target}] done in {time.perf_counter() - t_start:.1f}s")
    return 0


def _fit_mode(mode, args, prepared, cv, outdir, caps, *, log, n_durations,
              lineph_ml=None):
    """Run one fit mode. Returns the ML shape dict for LinEph, else None."""
    target = prepared.target
    tau_shift = args.ttv_max_days or 0.0

    tw, fw, ew = _prep.extract_near_transit_data(
        prepared.time, prepared.flux, prepared.flux_err,
        prepared.eph["period"], prepared.eph["epoch"],
        prepared.eph["duration"], n_durations=n_durations)
    epoch_data = _prep.segment_epochs(
        tw, fw, ew, prepared.eph["period"], prepared.eph["epoch"],
        prepared.eph["duration"], n_durations=n_durations,
        tau_shift_max=tau_shift)
    centering = _prep.centering_constants(epoch_data, prepared.eph)
    orders = np.asarray(cv["orders"])[:epoch_data["n_epochs"]]
    if orders.size != epoch_data["n_epochs"]:
        # the CV ran on its own segmentation; fall back to hurin's default
        orders = np.full(epoch_data["n_epochs"], 2)

    n_ep = epoch_data["n_epochs"]
    log(f"[{target}] {mode}: {n_ep} epochs, "
        f"{int(epoch_data['mask'].sum())} points, "
        f"{prepared.num_resample} sub-exposures")

    # resume state and its guards
    prior_state = None if args.fresh else _outputs.load_resume(
        outdir, target, mode)
    if prior_state is not None:
        prior_state.check_model_rev(log=log)
        prior_state.check(
            b_prior=args.b_prior, geometry=args.geometry,
            sampler=args.sampler, ttv_max=args.ttv_max_days,
            # "auto" is a request to measure, not a model choice, so it can
            # never conflict with what a previous run resolved to
            profile_mode=(None if args.profile_mode == "auto"
                          else args.profile_mode))
        want_extend = args.extend1 if mode == "lineph" else args.extend2
        if prior_state.done and not want_extend:
            log(f"  {mode} already converged "
                f"({prior_state.n_samples_done} draws/chain); skipping. "
                "Use --extend1/--extend2 to add more, or --fresh to redo.")
            return prior_state.ml_params or None
        log(f"  resuming from {prior_state.n_samples_done} draws/chain")

    # timing priors
    if mode == "ttv":
        if args.ttv_max_days:
            tau_half = np.full(n_ep, args.ttv_max_days)
        else:
            lo, hi = _seeding._epoch_time_extent(epoch_data)
            tau_half = np.maximum(
                np.minimum(-lo, hi) - 0.55 * (prepared.eph["duration"] / 24.0),
                2.0 * prepared.cadence_days)
        layout = _params.ttv_layout(prepared.eph, centering, tau_half,
                                   b_prior=args.b_prior)
    else:
        layout = _params.lineph_layout(prepared.eph, b_prior=args.b_prior)

    build_kw = dict(num_resample=prepared.num_resample,
                    exposure_time=prepared.exposure_time,
                    geometry=args.geometry, n_chains_hint=args.chains)

    # A resumed run keeps the mode it was sampled under: a continuation must
    # not silently change its own likelihood. An explicit --PL wins otherwise.
    if prior_state is not None and args.profile_mode == "auto":
        pl_choice = _plselect.fixed_choice(
            prior_state.profile_mode, "carried over from the resumed run")
    elif args.profile_mode != "auto":
        pl_choice = _plselect.fixed_choice(args.profile_mode)
    else:
        pl_choice = None                     # probed below, once the MAP exists

    provisional = pl_choice.mode if pl_choice else "exact"
    target_obj, transform, lp, hi = _likelihood.build_target(
        layout, centering, epoch_data, orders,
        profile_mode=provisional, **build_kw)
    log(f"  log-density: dim {layout.dim}, "
        f"{len(lp.blocks)} epoch block(s) of <= {lp.block_size}")
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
            k=s["k"], b=b0, T14=s["T14"], q1=s["q1"], q2=s["q2"],
            period=layout.P_ref, num_resample=prepared.num_resample,
            exposure_time=prepared.exposure_time,
            profile_mode=provisional, geometry=args.geometry)
        weak = np.where(rival_gap < _seeding.RIVAL_GAP_WARN)[0]
        if weak.size:
            log(f"  WARNING: {weak.size} epoch(s) have a rival timing mode "
                f"within {_seeding.RIVAL_GAP_WARN:.0f} log-units of the seed; "
                "those transit times are shape-sensitive")
            for i in weak[:6]:
                log(f"    epoch {int(centering['n_arr'][i]):+d}: "
                    f"seed {seeds[i] * 1440:+.1f} min, "
                    f"rival gap {rival_gap[i]:.1f}")

    v0 = _seeding.initial_model_vector(layout, prepared.eph, shape=shape0,
                                       dtau=seeds)
    cfg = _sampling.SamplerConfig(
        sampler=args.sampler, n_chains=args.chains, n_warmup=args.warmup,
        n_samples=args.samples, max_samples=args.max_samples,
        max_leapfrog=args.max_leapfrog, seed=args.seed).for_mode(layout.dim)

    resume_for_anvil = None
    if prior_state is not None and caps.anvil_resume and \
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
        log(f"  finding the MAP from {min(256, 4 * cfg.n_chains // 4)} starts...")
        rng = np.random.default_rng(args.seed)
        n_starts = 128
        u_starts = (transform.from_model_np(v0)[None, :]
                    + 0.25 * rng.standard_normal((n_starts, layout.dim)))
        t0 = time.perf_counter()
        map_res = _seeding.find_map(target_obj, u_starts.astype(np.float32))
        log(f"    MAP log-density {map_res.log_prob_best:.3f} "
            f"in {time.perf_counter() - t0:.1f}s")
        if map_res.multimodal_warning:
            log(f"    WARNING: the best starts disagree by "
                f"{map_res.top_spread:.1f} log-units — the posterior probably "
                "has more than one basin")
        if pl_choice is None:
            # Probe at the MAP: the ball is scaled to the posterior's own
            # width, so the modes are compared where sampling happens.
            pl_choice = _plselect.select_pl_mode(
                layout, centering, epoch_data, orders, map_res.u_best,
                transform, build_kwargs=build_kw, n_chains=cfg.n_chains,
                seed=args.seed, log=log)
        u0 = map_res.ball(cfg.n_chains, seed=args.seed + 1)

    if pl_choice is None:                    # resumed without a MAP step
        pl_choice = _plselect.fixed_choice(
            provisional, "carried over from the resumed run")
    if pl_choice.mode != provisional:
        # the unconstrained space is identical across modes, so the MAP and
        # its init ball carry over unchanged
        target_obj, transform, lp, hi = _likelihood.build_target(
            layout, centering, epoch_data, orders,
            profile_mode=pl_choice.mode, **build_kw)

    if u0 is not None:
        _sampling.check_precision(target_obj, u0, log=log,
                                  strict=not args.no_strict_precision)

    # ---- sample, exporting after every round
    state_holder = {}

    def on_round(results, verdict, rnd):
        _export_all(mode, args, prepared, epoch_data, centering, orders,
                    layout, transform, lp, results, verdict, outdir, caps,
                    log=log, state_holder=state_holder, cfg=cfg,
                    pl_mode=pl_choice.mode)

    results, verdict, total = _sampling.run_rounds(
        target_obj, list(layout.names), u0, cfg, log=log, on_round=on_round,
        resume_state=resume_for_anvil)

    cert = _sampling.certify(target_obj, results, list(layout.names), log=log)
    if cert is not None:
        state_holder["certify"] = str(cert)

    ml = state_holder.get("ml_params")
    return ml if mode == "lineph" else None


def _ml_row(results, lp, transform):
    """The maximum-likelihood draw, in model space, and its log-density."""
    lpv = np.asarray(results.get_log_prob(flat=True), dtype=np.float64)
    flat = results.get_chain(flat=True).astype(np.float64)
    j = int(np.argmax(lpv))
    return transform.model_np(flat[j:j + 1])[0], float(lpv[j])


def _export_all(mode, args, prepared, epoch_data, centering, orders, layout,
                transform, lp, results, verdict, outdir, caps, *, log,
                state_holder, cfg, pl_mode):
    """Write every product for this mode. Called after each sampling round."""
    target = prepared.target
    names = list(layout.names)
    phys = _sampling.physical_draws(transform, results)

    # anvil reports rank-normalized bulk ESS only (no tail ESS), so hurin's
    # Tail_ESS column is written empty rather than filled with a placeholder.
    summary = _outputs.summarize(phys, names, rhat=verdict.rhat,
                                 ess_bulk=verdict.ess, ess_tail=None)

    # derived quantities, computed after sampling in float64
    b_draws = _outputs.derived_b(phys, names, layout.b_prior)
    rho = _outputs.log10_rho_draws(
        phys, names, layout.b_prior,
        P_ref=None if mode == "lineph" else layout.P_ref)
    derived = _outputs.summarize(
        np.column_stack([b_draws, rho]), ["b", "log10_rho"])
    summary.update(derived)

    _outputs.export_summary(outdir, target, mode, summary, log=log)
    _outputs.export_logrho(outdir, target, mode, rho, log=log)

    cols = list(names) + ["b", "log10_rho", "loglike"]
    arrays = [phys[:, i] for i in range(phys.shape[1])] + [
        b_draws, rho,
        np.asarray(results.get_log_prob(flat=True), dtype=np.float64)
        + lp.log_const]
    _outputs.export_chains(outdir, target, mode, cols, arrays, log=log)

    # maximum-likelihood model, for the light-curve export and the plots
    v_ml, logl_ml = _ml_row(results, lp, transform)
    with mx.stream(mx.cpu):
        model, coeffs = lp.full_model(
            mx.array(v_ml[None, :].astype(np.float32)))
        model = np.array(model, dtype=np.float64)[0]
        baseline = np.array(_profile.baseline(lp.design, coeffs),
                            dtype=np.float64)[0]
    _outputs.export_lcdata(outdir, target, mode, epoch_data, model, log=log)

    state_holder["ml_params"] = {
        n: float(v_ml[i]) for i, n in enumerate(names)
        if n in ("k", "beta", "T14", "q1", "q2")}

    # ---- figures
    try:
        _plots.corner_plot(
            _outputs.product_path(outdir, target, mode, "corner", "pdf"),
            phys[:, :min(len(names), 7)],
            [_label(n) for n in names[:min(len(names), 7)]],
            title=f"{target} {mode}", log=log)
    except Exception as exc:
        log(f"    corner plot skipped: {exc}")

    mid_abs = _mid_times_absolute(mode, v_ml, layout, centering)
    try:
        tt, tf = _plots.model_grid(lp, v_ml, T14=v_ml[layout.index("T14")])
        _plots.fold_plot(
            _outputs.product_path(outdir, target, mode, "fold", "pdf"),
            epoch_data=epoch_data, mid_times=mid_abs, baseline=baseline,
            model_grid_t=tt, model_grid_f=tf,
            period=layout.P_ref + (v_ml[0] if mode == "lineph" else 0.0),
            title=f"{target} {mode}", colour=_plots.MODE_COLOURS[mode],
            n_bins_from=epoch_data["n_epochs"], log=log)
    except Exception as exc:
        log(f"    fold plot skipped: {exc}")

    if mode == "ttv":
        rows = _ttv_rows(phys, names, layout, centering, epoch_data, model,
                         summary)
        _outputs.export_ttv_times(outdir, target, mode, rows, log=log)
        try:
            _plots.oc_plot(
                _outputs.product_path(outdir, target, mode, "oc", "pdf"),
                [r["epoch"] for r in rows], [r["tmid"] for r in rows],
                [r["tmid_err"] for r in rows],
                title=f"{target} transit timing", log=log)
        except Exception as exc:
            log(f"    O-C plot skipped: {exc}")

    # ---- resume state
    anvil_state_path = None
    if caps.anvil_state_io:
        try:
            anvil_state_path = _outputs.product_path(
                outdir, target, mode, "anvilstate", "npz")
            results.save_state(anvil_state_path)
        except Exception as exc:
            log(f"    anvil state not saved: {exc}")
            anvil_state_path = None

    state = _outputs.ResumeState(
        target=target, mode=mode, turin_version=__version__,
        launch_command=_outputs.launch_command(), tag=args.tag,
        b_prior=layout.b_prior, profile_mode=pl_mode,
        geometry=args.geometry, sampler=args.sampler,
        model_rev=_MODEL_REV,
        n_chains=cfg.n_chains, ttv_max=args.ttv_max_days,
        n_durations=float(epoch_data["half_window"]
                          / (prepared.eph["duration"] / 24.0)),
        legendre_orders=np.asarray(orders), exposure_time=prepared.exposure_time,
        num_resample=prepared.num_resample,
        n_samples_done=int(verdict.n_draws), done=bool(verdict.converged),
        ml_params=state_holder.get("ml_params", {}),
        anvil_state_path=anvil_state_path,
        last_u=np.asarray(results.final_state["u"], dtype=np.float32),
    )
    _outputs.save_resume(outdir, target, mode, state, log=log)

    # width breakdown, if any chain looked trapped
    rows = _sampling.width_breakdown(transform, results, names, verdict.health)
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
    return centers + (v_ml[5:] + d_arr)


def _ttv_rows(phys, names, layout, centering, epoch_data, model, summary):
    """Per-epoch timing rows for the TTV export, with fit diagnostics."""
    n_arr = np.asarray(centering["n_arr"], dtype=np.float64)
    P_fit, tau0_fit, _ = _outputs.fit_linear_ephemeris(
        n_arr, [summary[f"dtau_{int(n)}"]["median"] for n in n_arr])

    mask = np.asarray(epoch_data["mask"]) > 0
    y = np.asarray(epoch_data["flux_padded"], dtype=np.float64)
    e = np.asarray(epoch_data["ferr_padded"], dtype=np.float64)
    m = np.asarray(model, dtype=np.float64)

    rows = []
    for i, n in enumerate(n_arr):
        key = f"dtau_{int(n)}"
        tmid = summary[key]["median"]
        err = summary[key]["std"]
        sel = mask[i]
        chi2 = float(np.sum(((y[i][sel] - m[i][sel]) / e[i][sel]) ** 2))
        # detection SNR against a no-transit null on the same points
        chi2_null = float(np.sum(((y[i][sel] - 1.0) / e[i][sel]) ** 2))
        snr = float(np.sqrt(max(chi2_null - chi2, 0.0)))
        oc = (tmid - (tau0_fit + P_fit * n)) * 1440.0
        rows.append(dict(epoch=int(n), tmid=tmid, tmid_err=err, ttv_min=oc,
                         ttv_err_min=err * 1440.0, snr=snr,
                         npts=int(sel.sum()), chi2=chi2))
    return rows
