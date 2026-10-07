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

#: Transit times shown at each end of a long TTV fit's corner plot; the ones
#: between are replaced by a "..." row and column.
CORNER_EPOCHS = 7

#: Minimum seconds between writes of the chains tarball and the PDFs during
#: a run; they are always written after round 0 and at the end.
HEAVY_EVERY_S = 1200

PARAM_LABELS = {
    "dP": "$P$ (d)", "dtau0": r"$\tau_0$", "k": "$k = R_p/R_\\star$",
    "b": "$b$", "T14": "$T_{14}$ (d)", "q1": "$q_1$",
    "q2": "$q_2$",
}


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
    if getattr(args, "ld", "sampled") == "collapsed":
        from .cli import collapsed_ld_problem

        problem = collapsed_ld_problem(args)
        if problem:
            raise SystemExit(f"turin: {problem}")
        _caps.require_metalplanet()
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

    log(f"[{target}] Cross-validating Legendre orders...")
    cv = _prep.cv_orders(prepared.epoch_data, prepared.eph["duration"],
                         tau_shift_max=args.ttv_max_days or 0.0,
                         log=lambda m: log(f"  {m}"))

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
            ld=getattr(args, "ld", "sampled"))
        want_extend = args.extend1 if mode == "lineph" else args.extend2
        if prior_state.done and not want_extend:
            log(f"  {mode} already converged "
                f"({prior_state.n_samples_done} draws/chain); skipping. "
                "Use --extend1/--extend2 to add more, or --fresh to redo.")
            outcome[mode] = "converged earlier"
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
        layout = _params.lineph_layout(prepared.eph, b_prior=args.b_prior,
                                       ld=getattr(args, "ld", "sampled"))

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
    elif prior_state is not None and args.profile_mode == "auto":
        pl_choice = _plselect.fixed_choice(
            prior_state.profile_mode, "carried over from the resumed run")
    elif args.profile_mode != "auto":
        pl_choice = _plselect.fixed_choice(args.profile_mode)
    else:
        pl_choice = None                     # probed below, once the MAP exists

    provisional = pl_choice.mode if pl_choice else "exact"
    target_obj, transform, lp, hi = _likelihood.build_target(
        layout, centering, epoch_data, orders,
        profile_mode=provisional, ld_mode=layout.ld, **build_kw)
    log(f"  log-density: dim {layout.dim}, "
        f"{len(lp.blocks)} epoch block(s) of <= {lp.block_size}")
    if layout.ld == "collapsed":
        log("  limb darkening: collapsed -- integrated out of the "
            "log-density; q1, q2 drawn afterwards from their exact "
            "conditional")
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
            k=s["k"], b=b0, T14=s["T14"], q1=s["q1"], q2=s["q2"],
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
            profile_mode=pl_choice.mode, ld_mode=layout.ld, **build_kw)

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
            _outputs.product_path(outdir, target, mode, "fold", "pdf"),
            epoch_data=epoch_data, mid_times=mid_abs, baseline=baseline,
            model_grid_t=tt, model_grid_f=tf,
            period=layout.P_ref + (v_ml[0] if mode == "lineph" else 0.0),
            title=f"{target} {mode}", colour=_plots.MODE_COLOURS[mode],
            n_bins_from=epoch_data["n_epochs"], log=log)
    except Exception as exc:
        log(f"    fold plot skipped: {exc}")


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
    thin = _sampling.export_thin(results)
    collapsed = layout.ld == "collapsed"
    if collapsed:
        # collapsed Gibbs: q1, q2 were integrated out of the sampled target,
        # so each kept draw gets them from its exact conditional here. The
        # sampler needs model-space draws (physical ones carry dP/dtau0's
        # report offsets).
        from . import ldmarg as _ldmarg

        flat = results.get_chain(thin=thin, flat=True).astype(np.float64)
        v_model = transform.model_np(flat)
        q1, q2, acc, edge = _ldmarg.draw_limb_darkening(lp, v_model,
                                                        seed=args.seed)
        phys = np.column_stack([transform.to_physical(v_model), q1, q2])
        names = names_s + ["q1", "q2"]
        log(f"  limb darkening: {len(q1)} conditional draws of q1, q2 "
            f"(MH acceptance {acc:.2f}; expansion point on a triangle edge "
            f"for {100 * edge:.0f}% of draws)")
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
        summary.update(_outputs.summarize(phys[:, len(names_s):], ["q1", "q2"]))

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

    if heavy:
        cols = list(names) + ["b", "log10_rho", "loglike"]
        arrays = [phys[:, i] for i in range(phys.shape[1])] + [
            b_draws, rho,
            np.asarray(results.get_log_prob(thin=thin, flat=True),
                       dtype=np.float64) + lp.log_const]
        _outputs.export_chains(outdir, target, mode, cols, arrays, thin=thin,
                               n_total=results.get_log_prob().size, log=log)

    # maximum-likelihood model, for the light-curve export and the plots
    v_ml, logl_ml = _ml_row(results, lp, transform)
    q_ml = None
    v_ml_named = v_ml
    with mx.stream(mx.cpu):
        v_row = mx.array(v_ml[None, :].astype(np.float32))
        if collapsed:
            # one curve needs one (q1, q2): the conditional mode at the ML
            # theta. A display point estimate only; the posterior q1, q2 in
            # every other product are the conditional draws above.
            (cq1,), (cq2,) = lp.conditional_mode_ld(v_row)
            q_ml = (float(cq1), float(cq2))
            v_ml_named = np.concatenate([v_ml, q_ml])
        model, coeffs = lp.full_model(v_row, ld=q_ml)
        model = np.array(model, dtype=np.float64)[0]
        baseline = np.array(_profile.baseline(lp.design, coeffs),
                            dtype=np.float64)[0]
    _outputs.export_lcdata(outdir, target, mode, epoch_data, model, log=log)

    state_holder["ml_params"] = {
        n: float(v_ml_named[i]) for i, n in enumerate(names)
        if n in ("k", "beta", "T14", "q1", "q2")}

    # ---- figures
    if heavy:
        _write_figures(mode, target, outdir, names, phys, b_draws, lp, v_ml,
                       layout, centering, epoch_data, baseline, log=log,
                       ld=q_ml)

    if mode == "ttv":
        rows = _ttv_rows(phys, names, layout, centering, epoch_data, model,
                         summary, lp=lp, v_ml=v_ml)
        _outputs.export_ttv_times(outdir, target, mode, rows, log=log)
        if heavy:
            try:
                _plots.oc_plot(
                    _outputs.product_path(outdir, target, mode, "oc", "pdf"),
                    [r["epoch"] for r in rows], [r["tmid"] for r in rows],
                    [r["tmid_err"] for r in rows],
                    title=f"{target} transit timing", log=log)
            except Exception as exc:
                log(f"    O-C plot skipped: {exc}")

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
        ld=layout.ld,
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
    return centers + (v_ml[5:] + d_arr)


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
