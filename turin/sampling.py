"""The sampling driver: anvil kernels, convergence, diagnostics, resume.

Replaces hurin's NUTS-plus-chain-doubling loop. The shape of the computation
is different in a way that matters: hurin ran 2 chains and lengthened them,
while ChEES-HMC decorrelates in one or two iterations and so wants *many*
chains and few draws. turin therefore defaults to hundreds of chains, and
"extending" means drawing more iterations from the same ensemble.

Three things here exist because of measured failures in the packages turin
builds on, not from caution:

- **The trapped-chain check.** ``anvil-gp/docs/anvil-stretch-width.md``
  records 2 of 1024 chains sitting 1,500 log-units below the bulk in a
  secondary mode, which inflated a reported width 4x while max R-hat was
  1.037 -- the only tell. anvil does not report this, so turin computes it
  from per-chain log-probability and acceptance and refuses to quote widths
  without saying so.
- **Chains are never silently dropped.** A flagged chain is reported, and the
  bulk-only widths are reported beside the all-chain widths, so the difference
  is visible rather than decided here.
- **Precision is adjudicated before and after.** ``validate_precision`` at the
  initialization ball, ``certify`` on the draws.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import mlx.core as mx
import numpy as np

from . import capabilities as _caps

#: Convergence gates. hurin used R-hat < 1.1 and bulk ESS > 400 on b, k, T14
#: only; turin gates every sampled parameter, and asks more of R-hat because
#: many-chain ChEES makes it cheap. Per-epoch timing parameters get a lower
#: ESS bar: there are dozens of them and each is informed by one transit.
RHAT_MAX = 1.01
ESS_MIN = 400
ESS_MIN_TAU = 100

#: Working-set budget for ``anvil.diagnose`` (bytes); see :func:`assess`.
DIAGNOSE_MEMORY_BUDGET = 256 * 2**20

#: Export works on a systematic subsample of the pooled draws at most this
#: many rows (and this many values, so a high-dimensional fit gets fewer
#: rows). Summaries, the chains product, plots and the float32 certificate
#: are limited by effective sample size long before this; R-hat and ESS are
#: always computed on every draw.
EXPORT_MAX_ROWS = 1_000_000
EXPORT_MAX_VALUES = 16_000_000

#: A chain whose median log-probability is this far below the ensemble median
#: is reported as trapped. The stretch-move finding was 1,500 log-units, so
#: this is deliberately sensitive.
TRAPPED_LOGP_DROP = 10.0
#: ... or whose acceptance is this fraction of the ensemble median.
TRAPPED_ACCEPT_RATIO = 0.2


@dataclass
class SamplerConfig:
    """Everything that decides how the sampling runs."""

    sampler: str = "chees"          # "chees" or "ensemble"
    n_chains: int = 512
    #: 800, not 400: on KOI-7567.01's TTV fit (seed 0) warmup 400 hit the
    #: draw cap unconverged in 107 min of sampling, 800 converged in 67 min;
    #: on KOI-5162.01 LinEph, whose period posterior is bimodal, it cost
    #: +12 min (72 vs 60) with no change in rounds. 1600 was slower on both.
    n_warmup: int = 800
    n_samples: int = 300
    max_samples: int = 16384        # per chain, summed over rounds
    max_leapfrog: int = 128
    dense: bool | None = None       # None: decide from n_chains vs dim
    seed: int = 0
    thin: int = 1
    #: ensemble defaults, used when sampler == "ensemble"
    n_walkers: int = 128
    ensemble_warmup: int = 500
    ensemble_samples: int = 20000

    def for_mode(self, dim):
        """A copy with chain count and preconditioner resolved against ``dim``.

        ``dense=True`` needs at least ``4*dim`` chains or anvil silently
        downgrades it to diagonal, so the chain count is raised to meet the
        request rather than letting the request be quietly dropped.
        """
        cfg = SamplerConfig(**vars(self))
        if cfg.sampler == "ensemble":
            # the stretch move needs an even count and at least 2*dim walkers
            cfg.n_chains = max(cfg.n_walkers, 4 * dim)
            cfg.n_chains += cfg.n_chains % 2
            cfg.n_warmup = cfg.ensemble_warmup
            cfg.n_samples = cfg.ensemble_samples
            cfg.dense = False
            return cfg
        cfg.n_chains = max(cfg.n_chains, 4 * dim)
        if cfg.dense is None:
            cfg.dense = cfg.n_chains >= 4 * dim
        return cfg


@dataclass
class ChainHealth:
    """Per-chain trapping diagnosis, from log-probability and acceptance."""

    n_chains: int
    trapped: np.ndarray             # (n_trapped,) chain indices
    logp_median: np.ndarray         # (n_chains,)
    accept: np.ndarray              # (n_chains,)
    divergences: np.ndarray | None  # (n_chains,) if anvil reports them
    bulk_logp: float
    reason: dict = field(default_factory=dict)

    @property
    def any_trapped(self) -> bool:
        return self.trapped.size > 0

    def describe(self):
        if not self.any_trapped:
            return (f"  all {self.n_chains} chains healthy "
                    f"(median log-prob spread "
                    f"{np.ptp(self.logp_median):.2f})")
        lines = [f"  WARNING: {self.trapped.size} of {self.n_chains} chains "
                 f"look trapped; their draws are included but widths are "
                 f"reported both ways"]
        for i in self.trapped[:8]:
            lines.append(
                f"    chain {int(i):4d}: log-prob median "
                f"{self.logp_median[i]:.1f} "
                f"({self.logp_median[i] - self.bulk_logp:+.1f} vs bulk), "
                f"accept {self.accept[i]:.3f}"
                + (f", divergences {int(self.divergences[i])}"
                   if self.divergences is not None else "")
                + f"  [{self.reason.get(int(i), '')}]")
        if self.trapped.size > 8:
            lines.append(f"    ... and {self.trapped.size - 8} more")
        return "\n".join(lines)


def chain_health(results):
    """Diagnose trapped chains without dropping any.

    Two independent signals, because they mean different things. A chain low
    in log-probability *and* diverging is stuck against a bounded parameter's
    edge (anvil's ChEES boundary trap). One low in log-probability with healthy
    acceptance is in a secondary mode, which is the stretch-move failure.
    """
    lp = np.asarray(results.get_log_prob(), dtype=np.float64)   # (draws, chains)
    med = np.median(lp, axis=0)
    accept = np.asarray(results.accept_fraction, dtype=np.float64)
    div = (results.extras or {}).get("divergent_per_chain")
    div = None if div is None else np.asarray(div, dtype=np.float64)

    bulk = float(np.median(med))
    mad = float(np.median(np.abs(med - bulk))) * 1.4826
    drop = max(TRAPPED_LOGP_DROP, 5.0 * mad)
    acc_bulk = float(np.median(accept))

    low_lp = med < bulk - drop
    low_acc = accept < TRAPPED_ACCEPT_RATIO * acc_bulk
    flagged = np.where(low_lp | low_acc)[0]

    reason = {}
    for i in flagged:
        if low_lp[i] and (div is not None and div[i] > 0.5 * len(lp)):
            reason[int(i)] = "low log-prob and diverging: boundary trap"
        elif low_lp[i] and low_acc[i]:
            reason[int(i)] = "low log-prob, low acceptance: stuck"
        elif low_lp[i]:
            reason[int(i)] = "low log-prob, healthy acceptance: secondary mode"
        else:
            reason[int(i)] = "low acceptance only"

    return ChainHealth(n_chains=lp.shape[1], trapped=flagged, logp_median=med,
                       accept=accept, divergences=div, bulk_logp=bulk,
                       reason=reason)


@dataclass
class Verdict:
    """Convergence assessment for one round."""

    converged: bool
    rhat: np.ndarray
    ess: np.ndarray
    names: list
    n_draws: int
    n_divergent: int
    health: ChainHealth
    worst_rhat: tuple
    worst_ess: tuple
    warmup_verdict: str = ""
    skew: float = 0.0
    excess_kurtosis: float = 0.0

    def describe(self):
        lines = [
            f"  draws/chain {self.n_draws}, divergences {self.n_divergent}",
            f"  worst R-hat {self.worst_rhat[1]:.4f} ({self.worst_rhat[0]}), "
            f"min bulk ESS {self.worst_ess[1]:.0f} ({self.worst_ess[0]})",
        ]
        if self.warmup_verdict:
            lines.append(f"  warmup: {self.warmup_verdict}")
        lines.append(f"  posterior shape after whitening: skew "
                     f"{self.skew:.2f}, excess kurtosis "
                     f"{self.excess_kurtosis:.2f}"
                     + ("  (curved: a dense mass matrix will not help much)"
                        if abs(self.skew) > 1.0 else ""))
        lines.append(self.health.describe())
        lines.append("  CONVERGED" if self.converged
                     else "  not yet converged")
        return "\n".join(lines)


def _ess_floor(name):
    return ESS_MIN_TAU if name.startswith("dtau_") else ESS_MIN


def assess(results, names, *, settle_tol=0.05):
    """Convergence and health verdict for a completed round."""
    import anvil

    chain = results.get_chain()
    # anvil >= 0.3 accumulates the autocovariance in chunks and ranks one
    # parameter at a time, so a small budget bounds the MLX peak whatever
    # the dimension: measured at the draw cap, 256 MiB gives 0.55 GB and is
    # as fast as larger budgets (0.7 s at dim 8, 13.7 s at dim 105).
    diag = anvil.diagnose(chain, names=list(names),
                          memory_budget=DIAGNOSE_MEMORY_BUDGET)
    rhat = np.asarray(diag.rhat, dtype=np.float64)
    ess = np.asarray(diag.ess_bulk, dtype=np.float64)

    ok = np.all(rhat < RHAT_MAX) and all(
        ess[i] > _ess_floor(n) for i, n in enumerate(names))

    warmup_verdict = ""
    if getattr(results, "warmup_trace", None):
        try:
            warmup_verdict = str(
                anvil.warmup_report(results, settle_tol=settle_tol))
        except Exception as exc:                      # diagnostic only
            warmup_verdict = f"(warmup_report failed: {exc})"

    skew = exk = 0.0
    try:
        t = export_thin(results)
        flat = chain[::t].reshape(-1, chain.shape[-1]).astype(np.float64)
        s, k = anvil.whitened_shape(flat)
        skew, exk = float(np.max(np.abs(s))), float(np.max(np.abs(k)))
    except Exception:
        pass

    i_r, i_e = int(np.argmax(rhat)), int(np.argmin(ess))
    return Verdict(
        converged=bool(ok), rhat=rhat, ess=ess, names=list(names),
        n_draws=chain.shape[0], n_divergent=int(
            (results.extras or {}).get("n_divergent", 0)),
        health=chain_health(results),
        worst_rhat=(names[i_r], float(rhat[i_r])),
        worst_ess=(names[i_e], float(ess[i_e])),
        warmup_verdict=warmup_verdict, skew=skew, excess_kurtosis=exk,
    )


def make_kernel(target, cfg, log=None):
    """Build the anvil kernel described by ``cfg``."""
    import anvil

    if cfg.sampler == "chees":
        if log and cfg.dense and cfg.n_chains < 4 * target.dim:
            log("  note: too few chains for a dense mass matrix; anvil will "
                "fall back to diagonal")
        return anvil.ChEESHMC(target, max_leapfrog=cfg.max_leapfrog,
                              dense=bool(cfg.dense))
    if cfg.sampler == "ensemble":
        return anvil.EnsembleKernel(
            target,
            moves=[(anvil.StretchMove(), 0.5), (anvil.DEMove(), 0.5)],
            seed=cfg.seed)
    raise ValueError(f"unknown sampler {cfg.sampler!r}; "
                     "expected 'chees' or 'ensemble'")


def check_precision(target, u0, log=print, strict=True):
    """Run anvil's float32 adjudication before committing to a fit."""
    import anvil

    probe = u0[:32] if u0.shape[0] > 32 else u0
    report = anvil.validate_precision(target, probe)
    text = str(report)
    if log:
        for line in text.splitlines():
            log(f"  {line}")
    if strict and "WARNING" in text:
        raise RuntimeError(
            "float32 log-density error is comparable to the Metropolis scale; "
            "refusing to sample. Check the data conditioning (see "
            "docs/hurin-differences.md) or run with --no-strict-precision.")
    return report


class Continuation:
    """The draws of several resumed segments, presented as one ``Results``.

    Valid because the segments are one Markov chain: every segment after the
    first is an anvil ``resume`` with adaptation frozen, so nothing re-adapts
    between them (a grid-Gibbs move between segments is an exact move of the
    same chain, so it keeps this true).

    Presents the slice of ``anvil.Results`` that turin consumes. Per-segment
    accounting (acceptance, divergences) is taken from the latest segment,
    matching anvil's own contract that those describe the resumed segment.
    """

    def __init__(self, segments):
        self.segments = list(segments)
        last = self.segments[-1]
        self.final_state = last.final_state
        self.warmup_trace = self.segments[0].warmup_trace
        self.extras = last.extras
        self.accept_fraction = last.accept_fraction
        self.n_chains = last.n_chains
        self.dim = last.dim
        self.n_warmup = self.segments[0].n_warmup
        self.thin = last.thin
        self._last = last
        self._chain = None
        self._lp = None

    def get_chain(self, discard=0, thin=1, flat=False):
        # concatenated once and cached: the segments never change, and every
        # consumer (assess, export, certify) asks for the pooled array
        if self._chain is None:
            self._chain = np.concatenate(
                [s.get_chain() for s in self.segments], axis=0)
        chain = self._chain[discard::thin]
        return chain.reshape(-1, chain.shape[-1]) if flat else chain

    def get_log_prob(self, discard=0, thin=1, flat=False):
        if self._lp is None:
            self._lp = np.concatenate(
                [s.get_log_prob() for s in self.segments], axis=0)
        lp = self._lp[discard::thin]
        return lp.reshape(-1) if flat else lp

    def save_state(self, path):
        return self._last.save_state(path)


def run_rounds(target, names, u0, cfg, *, log=print, on_round=None,
               resume_state=None, move=None, segment=100):
    """Sample, assess, and extend until converged or out of budget.

    ``on_round(results, verdict, round_index)`` is called after every round, so
    the caller can re-export products each time -- hurin's behaviour, and what
    makes a long run interruptible.

    Extension uses anvil's ``resume``, which continues the same chains with
    their adapted step size, trajectory length and preconditioner frozen, and
    with the key stream continuing rather than forking (hence ``seed=None`` on
    a continuation -- passing a seed would deliberately fork the segment).
    Because those segments are one Markov chain, their draws are **pooled**:
    a round that fails the convergence gates still contributes its samples,
    so extending is cheap in both warmup and information.

    ``move``, if given, is a :class:`turin.gibbs.GridGibbs` applied every
    ``segment`` draws: each round is drawn as several resumed segments with a
    sweep between them (and one before a cross-process resume). ChEES
    alternated with an exact move is still one Markov chain, so the segments
    pool exactly as before. Chains are moved with anvil's
    ``ResumeState.with_positions``, which recomputes every cached per-chain
    quantity at the new positions.
    """
    import anvil

    kernel = make_kernel(target, cfg, log=log)
    total = 0
    segments = []
    results = None
    verdict = None

    def gibbs(rs):
        nonlocal t_gibbs
        tg = time.perf_counter()
        u_new, st = move.sweep(np.array(rs.state["u"], dtype=np.float64))
        rs = rs.with_positions(mx.array(u_new.astype(np.float32)), target)
        t_gibbs += time.perf_counter() - tg
        sweep_stats.append(st)
        return rs

    for rnd in range(64):           # a bound, not an expectation
        n_samples = cfg.n_samples if rnd == 0 else min(
            cfg.n_samples * 2 ** rnd, max(1, cfg.max_samples - total))
        chunks = ([n_samples] if move is None else
                  [segment] * (n_samples // segment)
                  + ([n_samples % segment] if n_samples % segment else []))
        t_gibbs = 0.0
        sweep_stats = []
        round_segs = []

        t0 = time.perf_counter()
        for ci, chunk in enumerate(chunks):
            kw = dict(n_samples=chunk, thin=cfg.thin, progress=False)
            first = rnd == 0 and ci == 0
            if first and resume_state is not None:
                log(f"  round {rnd}: resuming {n_samples} draws/chain "
                    "with frozen adaptation")
                rs = resume_state if move is None else gibbs(resume_state)
                # u0 is unused on a resume, but anvil checks its shape -- a
                # free assertion that the stored state matches the chains
                results = anvil.run(kernel, target, u0, resume=rs,
                                    n_warmup=0, **kw)
            elif first:
                log(f"  round {rnd}: {cfg.n_warmup} warmup + {n_samples} "
                    f"draws/chain on {cfg.n_chains} chains"
                    + ("" if move is None else
                       f", grid-Gibbs on the epoch times every {segment}"))
                results = anvil.run(kernel, target, u0,
                                    n_warmup=cfg.n_warmup, seed=cfg.seed, **kw)
            else:
                if ci == 0:
                    log(f"  round {rnd}: +{n_samples} draws/chain "
                        f"(continuation, adaptation frozen, "
                        f"{total} already banked)")
                rs = results if move is None else gibbs(results.resume_state())
                # seed=None continues the key stream instead of forking it
                results = anvil.run(kernel, target, resume=rs, n_warmup=0,
                                    **kw)
            round_segs.append(results)
        el = time.perf_counter() - t0
        total += n_samples
        segments.extend(round_segs)

        reported = (Continuation(segments) if len(segments) > 1
                    else results)
        if len(round_segs) > 1:
            _round_accounting(reported, round_segs)
        verdict = assess(reported, names)
        log(f"  round {rnd} took {el:.1f}s "
            f"({n_samples * cfg.n_chains / max(el, 1e-9):.0f} draws/s)"
            + (f"; pooled {verdict.n_draws} draws/chain over "
               f"{len(segments)} segments" if len(segments) > 1 else ""))
        if sweep_stats:
            log(_describe_sweeps(sweep_stats, names, t_gibbs, el))
        log(verdict.describe())
        if on_round is not None:
            on_round(reported, verdict, rnd)
        _caps.clear_mlx_cache()

        if verdict.converged:
            break
        if total >= cfg.max_samples:
            log(f"  stopping at the {cfg.max_samples} draws/chain cap "
                "without meeting the convergence gates")
            break

    return reported, verdict, total


def _round_accounting(reported, round_segs):
    """Divergences and acceptance over a whole round, not its last segment.

    anvil reports both per segment; a round drawn as several segments would
    otherwise describe only its final ``segment`` draws.
    """
    extras = dict(round_segs[-1].extras or {})
    extras["n_divergent"] = sum(int((s.extras or {}).get("n_divergent", 0))
                                for s in round_segs)
    if all("divergent_per_chain" in (s.extras or {}) for s in round_segs):
        extras["divergent_per_chain"] = np.sum(
            [np.asarray(s.extras["divergent_per_chain"]) for s in round_segs],
            axis=0)
    reported.extras = extras
    reported.accept_fraction = np.mean(
        [np.asarray(s.accept_fraction, dtype=np.float64) for s in round_segs],
        axis=0)


def _describe_sweeps(stats, names, t_gibbs, elapsed):
    """One log line per round for the grid-Gibbs move."""
    acc = np.mean([s.accept for s in stats], axis=0)
    hop = np.mean([s.mode_change for s in stats], axis=0)
    epochs = [n for n in names if n.startswith("dtau_")]
    hops = [f"{epochs[i][5:]} ({hop[i]:.0%})" for i in np.argsort(-hop)
            if hop[i] >= 0.01][:6]
    return (f"  grid-Gibbs: {len(stats)} sweeps, {t_gibbs:.1f}s "
            f"({t_gibbs / max(elapsed, 1e-9):.0%} of the round), MH "
            f"acceptance {acc.min():.2f}-{acc.max():.2f}; chains changing "
            f"timing mode per sweep: "
            + (", ".join(f"epoch {h}" for h in hops) if hops else "none"))


def certify(target, results, names, *, n_probe=256, target_ess=1e4, log=print):
    """Measure and correct the float32 bias in posterior means.

    The float32 log-density is deterministic, so the sampler is exactly
    stationary for a slightly tilted target; a few hundred float64 probes
    measure the resulting bias in any posterior mean and remove it.
    """
    import anvil

    try:
        # the certificate is a few hundred float64 probes; a bounded,
        # systematic subsample of the draws serves it as well as all of them
        draws = results.get_chain(thin=export_thin(results), flat=True)
        cert = anvil.certify(target, draws, n_probe=n_probe,
                             target_ess=target_ess, names=list(names))
    except Exception as exc:
        if log:
            log(f"  certify unavailable: {exc}")
        return None
    if log:
        for line in str(cert).splitlines():
            log(f"  {line}")
    return cert


def export_thin(results):
    """Thinning factor that brings the pooled draws within the export budget.

    Systematic (every t-th draw of every chain), so each chain stays equally
    represented. 1 when everything fits.
    """
    # a view of the stored (cached, for a Continuation) array: no copy
    n_draws, n_chains, dim = results.get_chain().shape
    max_rows = min(EXPORT_MAX_ROWS, EXPORT_MAX_VALUES // max(1, dim + 3))
    return max(1, -(-n_draws * n_chains // max_rows))


def physical_draws(transform, results, *, discard=0, thin=1):
    """Flat posterior draws in float64 physical units."""
    flat = results.get_chain(discard=discard, thin=thin,
                             flat=True).astype(np.float64)
    return transform.to_physical(transform.model_np(flat))


def width_breakdown(transform, results, names, health):
    """Parameter widths with and without the flagged chains.

    The stretch-move finding is that a couple of trapped chains can inflate a
    reported width fourfold while every summary statistic looks fine. Quoting
    both numbers makes that visible instead of hidden.
    """
    chain = results.get_chain(thin=export_thin(results))  # (draws, chains, dim)
    flat_all = chain.reshape(-1, chain.shape[-1]).astype(np.float64)
    phys_all = transform.to_physical(transform.model_np(flat_all))

    keep = np.ones(chain.shape[1], dtype=bool)
    keep[health.trapped] = False
    if keep.all() or not keep.any():
        return None

    bulk = chain[:, keep, :].reshape(-1, chain.shape[-1]).astype(np.float64)
    phys_bulk = transform.to_physical(transform.model_np(bulk))
    rows = []
    for i, name in enumerate(names):
        sd_all = float(np.std(phys_all[:, i]))
        sd_bulk = float(np.std(phys_bulk[:, i]))
        ratio = sd_all / sd_bulk if sd_bulk > 0 else np.inf
        rows.append((name, sd_all, sd_bulk, ratio))
    return rows
