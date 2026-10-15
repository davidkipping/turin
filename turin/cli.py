"""The kwargs command line: ``turin --KOI-448.02``.

hurin's grammar, kept deliberately: the target *is* a flag, everything is
``--name`` or ``--name=value``, parsing is a regex scan of ``argv`` rather than
argparse, and an unrecognized flag is rejected with a "did you mean" hint
instead of being ignored. The point is that a typo in a long batch command
fails immediately rather than silently changing the fit.

Flags new in turin are the ones that expose choices hurin did not have: the
sampler, the profile form, and the two model conventions where hurin's was
either an approximation or a mistake (see ``docs/hurin-differences.md``).
"""

from __future__ import annotations

import difflib
import os
import re
import sys
from dataclasses import dataclass, field

from . import __version__
from .model import B_PRIORS, GEOMETRIES, LAWS
from .params import LD_MODES
from .profile import PROFILE_MODES

#: ``--PL`` accepts any solve mode, or "auto" to measure and choose.
PL_CHOICES = ("auto",) + tuple(PROFILE_MODES)

TARGET_RE = re.compile(r"^--(KOI-\d+\.\d+|TOI-\d+\.\d+)$", re.IGNORECASE)
MODES = ("lineph", "ttv")
SAMPLERS = ("chees", "ensemble")
GIBBSGRID = ("on", "off")
LD_CHOICES = tuple(LD_MODES)
LAW_CHOICES = tuple(LAWS)

_BOOL_FLAGS = {
    "--version": "show_version",
    "--help": "show_help",
    "-h": "show_help",
    "--clear-cache": "clear_cache",
    "--fresh": "fresh",
    "--extend1": "extend1",
    "--extend2": "extend2",
    "--nongrazing": "nongrazing",
    "--sc": "sc_override",
    "--no-strict-precision": "no_strict_precision",
    "--capabilities": "show_capabilities",
    "--replot": "replot",
    "--oblate": "oblate",
}
_VALUE_FLAGS = {
    "--chains": "chains",
    "--bprior": "b_prior",
    "--TTVmax": "ttv_max_min",
    "--TTVsnr": "ttv_snr_min",
    "--tag": "tag",
    "--sampler": "sampler",
    "--PL": "profile_mode",
    "--geometry": "geometry",
    "--warmup": "warmup",
    "--samples": "samples",
    "--max-samples": "max_samples",
    "--modes": "modes_raw",
    "--cache-dir": "cache_dir",
    "--outdir": "outdir",
    "--leapfrog": "max_leapfrog",
    "--seed": "seed",
    "--gibbsgrid": "gibbsgrid",
    "--ld": "ld",
    "--law": "law",
}

USAGE = f"""turin {__version__} — GPU transit fitting for Kepler and TESS

usage: turin --KOI-448.02 [options]
       turin --TOI-406.01 [options]

  --fresh                 ignore existing resume state for this lineage
  --extend1 / --extend2   extend the LinEph / TTV fit by one round
  --modes=lineph,ttv      which fits to run (default both)
  --tag=NAME              namespace this run into its own resume lineage
  --replot                redraw a finished fit's figures and ttv_times.csv
                          from its saved products, without sampling (for
                          figures made by an older turin)

  --chains=N              sampling chains (default 512, raised to 4*dim)
  --sampler=chees|ensemble  gradient-based (default) or gradient-free
  --warmup=N              warmup iterations (default 800)
  --samples=N --max-samples=N
  --leapfrog=N            ChEES trajectory cap (default 128)
  --gibbsgrid=on|off      TTV fits: redraw every transit time from its exact
                          grid conditional between ChEES segments, so chains
                          move between timing modes (default on). off gives
                          independent ChEES chains only
  --seed=N                random seed, a non-negative integer (default 0)

  --bprior=transiting|nongrazing|box   (b, k) prior (default transiting)
  --nongrazing            alias for --bprior=nongrazing
  --TTVmax=MINUTES        declared TTV amplitude; sets the timing priors
  --TTVsnr=X              skip the TTV fit when the median expected SNR of
                          a single transit is below X (default 3): its
                          times would be little more than their priors.
                          Computed from the LinEph best fit over epochs
                          with data in transit; 0 always fits
  --PL=auto|exact|hybrid|ratio
                          how the profile likelihood solves for the baseline
                          coefficients. The default, auto, measures all three
                          on this target -- precision against a float64
                          reference over the posterior's own width, then
                          speed -- and picks the fastest that is no less
                          precise than exact. Takes a few seconds and is
                          recorded in the products. Naming a mode skips it:
                          exact is the true flux-space profile, ratio is
                          hurin's O(transit depth) form, hybrid refines
                          ratio's static factorization back to exact
  --ld=sampled|collapsed  limb darkening. sampled (the default, except
                          collapsed under --oblate) samples the
                          law's q's (q1, q2; q1..q4 for hybrid4) with
                          everything else. collapsed integrates the limb
                          darkening out of the log-density and draws the
                          q's afterwards from their exact conditional
                          (collapsed Gibbs): same posterior, two (four)
                          fewer sampled dimensions. Quadratic: ~2.2x the
                          per-evaluation memory; hybrid4: 3.7-8.7x the
                          wall time on Kepler targets. LinEph only
                          (--modes=lineph); needs --PL=exact or auto
                          (auto resolves to exact) and circular geometry
  --law=quadratic|hybrid4 limb-darkening law. quadratic (default) samples
                          Kipping's q1, q2. hybrid4 is elome's four-weight
                          rational law (LDhybrid v1.0; SquishierPlanet
                          measures a worst-case light-curve error of 1.0 ppm
                          on PHOENIX J-band stars, against 65 for quadratic),
                          sampled through q1..q4 in the unit cube, mapped
                          uniformly onto its weight simplex (SquishierPlanet's
                          stick-breaking); the weights w1..w4 are reported as
                          derived. Needs circular geometry. With
                          --ld=collapsed (LinEph only) the four weights are
                          integrated out, as for the quadratic law
  --oblate                an oblate planet: projected flattening f and
                          long-axis sky angle theta, sampled as
                          fc, fs = sqrt(f) (cos 2theta, sin 2theta), uniform
                          in f on [0, 0.6] and in theta; f, theta (degrees)
                          and the semi-axes A, B are reported, k is the
                          area-equivalent radius and T14 that circle's
                          duration. Needs --law=hybrid4, --modes=lineph and
                          circular geometry; implies --ld=collapsed unless
                          --ld=sampled is given. The oblate kernel costs
                          ~40-85x a round one
  --geometry=circular|chord   true circular orbit (default) or hurin's chord

  --sc                    prefer short cadence
  --cache-dir=PATH --outdir=PATH
  --clear-cache           delete cached light curves and exit
  --capabilities          report what the installed anvil/elome can do
  --version --help

Exit codes: 0 every fit converged; 3 finished, but a fit stopped at the
draw cap unconverged (its products say so on line 2); anything else is an
error.

Against hurin >= 0.1.68 the orbit model is the only model difference:
--geometry=chord --PL=ratio reproduces hurin. See docs/hurin-differences.md.
"""


@dataclass
class Args:
    target: str | None = None
    show_version: bool = False
    show_help: bool = False
    show_capabilities: bool = False
    replot: bool = False
    clear_cache: bool = False
    fresh: bool = False
    extend1: bool = False
    extend2: bool = False
    nongrazing: bool = False
    sc_override: bool = False
    no_strict_precision: bool = False
    chains: int = 512
    b_prior: str = "transiting"
    ttv_max_min: float | None = None
    ttv_snr_min: float = 3.0
    tag: str | None = None
    sampler: str = "chees"
    profile_mode: str = "auto"
    geometry: str = "circular"
    warmup: int = 800
    samples: int = 300
    max_samples: int = 16384
    max_leapfrog: int = 128
    seed: int = 0
    gibbsgrid: str = "on"
    #: None until resolved (:func:`resolve_ld`): "collapsed" under --oblate,
    #: "sampled" otherwise
    ld: str | None = None
    law: str = "quadratic"
    oblate: bool = False
    modes: tuple = MODES
    cache_dir: str | None = None
    outdir: str | None = None

    @property
    def ttv_max_days(self):
        return (None if self.ttv_max_min is None
                else float(self.ttv_max_min) / 1440.0)


def _fail(msg, hint_from=None, vocabulary=()):
    if hint_from:
        close = difflib.get_close_matches(hint_from, list(vocabulary), n=1,
                                         cutoff=0.5)
        if close:
            msg += f"  Did you mean {close[0]}?"
    raise SystemExit(f"turin: {msg}\n\nRun 'turin --help' for usage.")


def _choice(name, value, allowed):
    if value not in allowed:
        _fail(f"--{name}={value!r} is not one of {', '.join(allowed)}",
              value, allowed)
    return value


def parse_args(argv=None):
    """Parse ``argv`` into :class:`Args`, rejecting anything unrecognized."""
    argv = list(sys.argv[1:] if argv is None else argv)
    args = Args()
    vocabulary = (list(_BOOL_FLAGS) + [f"{k}=" for k in _VALUE_FLAGS]
                  + ["--KOI-<n>.<nn>", "--TOI-<n>.<nn>"])

    for token in argv:
        m = TARGET_RE.match(token)
        if m:
            if args.target:
                _fail(f"more than one target given ({args.target} and "
                      f"{m.group(1).upper()})")
            args.target = m.group(1).upper()
            continue
        if token in _BOOL_FLAGS:
            setattr(args, _BOOL_FLAGS[token], True)
            continue
        if "=" in token:
            key, _, value = token.partition("=")
            if key in _VALUE_FLAGS:
                _assign(args, _VALUE_FLAGS[key], key, value)
                continue
        _fail(f"unrecognized argument {token!r}", token, vocabulary)

    if args.nongrazing:
        args.b_prior = "nongrazing"
    _choice("bprior", args.b_prior, B_PRIORS)
    _choice("sampler", args.sampler, SAMPLERS)
    _choice("PL", args.profile_mode, PL_CHOICES)
    _choice("geometry", args.geometry, GEOMETRIES)
    _choice("gibbsgrid", args.gibbsgrid, GIBBSGRID)
    resolve_ld(args)
    _choice("ld", args.ld, LD_CHOICES)
    _choice("law", args.law, LAW_CHOICES)
    for mode in args.modes:
        _choice("modes", mode, MODES)
    problem = (oblate_problem(args) or law_problem(args)
               or collapsed_ld_problem(args))
    if problem:
        _fail(problem)
    if args.tag is not None and not re.fullmatch(r"[A-Za-z0-9_-]+", args.tag):
        _fail(f"--tag={args.tag!r} must match [A-Za-z0-9_-]+")
    return args


def resolve_ld(args):
    """Settle ``args.ld`` if it was not given: ``collapsed`` under
    ``--oblate`` (SquishierPlanet: the GHK overhead is ~5% of an oblate
    evaluation, and on their hardest oblate geometry only collapsed chains
    converged), ``sampled`` otherwise. Idempotent; the pipeline calls it too,
    for code that builds ``Args`` without :func:`parse_args`."""
    if getattr(args, "ld", None) is None:
        args.ld = "collapsed" if getattr(args, "oblate", False) else "sampled"
    return args.ld


def oblate_problem(args):
    """Why ``--oblate`` cannot run in this combination, or None."""
    if not getattr(args, "oblate", False):
        return None
    law = getattr(args, "law", "quadratic")
    if law != "hybrid4":
        return (f"--oblate needs --law=hybrid4, not --law={law}: the "
                "quadratic law has no elementary light curve for an ellipse "
                "(its linear term needs elliptic integrals), and elome's "
                "oblate kernel takes the hybrid laws only")
    if "ttv" in args.modes:
        return ("--oblate is LinEph-only: pass --modes=lineph. A TTV fit "
                "would pay the oblate kernel (~40-85x) on every grid-Gibbs "
                "sweep, and oblateness is a shape property that LinEph "
                "measures from every epoch together")
    if args.geometry != "circular":
        return ("--oblate needs --geometry=circular: elome's oblate kernel "
                "is on its tau route")
    return None


def law_problem(args):
    """Why this combination cannot run with a non-quadratic ``--law``, or
    None. Checked here and again by the pipeline, as
    :func:`collapsed_ld_problem` is."""
    law = getattr(args, "law", "quadratic")
    if law == "quadratic":
        return None
    if args.geometry != "circular":
        return (f"--law={law} needs --geometry=circular: chord is for hurin "
                "parity, and hurin has only the quadratic law; the hybrid "
                "laws run on elome's tau kernel")
    return None


def collapsed_ld_problem(args):
    """Why this combination cannot run with ``--ld=collapsed``, or None.

    Checked here and again by the pipeline (a resumed lineage, or code that
    builds ``Args`` directly, never passes through :func:`parse_args`).
    """
    if getattr(args, "ld", "sampled") != "collapsed":
        return None
    if args.profile_mode in ("ratio", "hybrid"):
        return (f"--ld=collapsed needs --PL=exact (or auto), not "
                f"--PL={args.profile_mode}: the omega gradient and Hessian "
                f"come from the envelope theorem, which holds only when the "
                f"baseline is the exact flux-space minimiser, so {args.profile_mode} "
                f"would be wrong rather than merely untested")
    if "ttv" in args.modes:
        return ("--ld=collapsed is LinEph-only: pass --modes=lineph. A shared "
                "limb darkening couples every epoch, which breaks the "
                "per-epoch factorisation the TTV fit's grid-Gibbs relies on")
    if args.geometry != "circular":
        return ("--ld=collapsed needs --geometry=circular: it uses "
                "elome's ld_basis, which exists only on the tau kernel")
    return None


def _assign(args, field_name, flag, raw):
    ints = {"chains", "warmup", "samples", "max_samples", "max_leapfrog",
            "seed"}
    # every integer flag must be positive except --seed, whose default is 0
    least = 0 if field_name == "seed" else 1
    try:
        if field_name in ints:
            value = int(raw)
            if value < least:
                raise ValueError
        elif field_name == "ttv_max_min":
            value = float(raw)
            if value <= 0:
                raise ValueError
        elif field_name == "ttv_snr_min":
            value = float(raw)
            if not value >= 0:
                raise ValueError
        elif field_name == "modes_raw":
            value = tuple(m.strip() for m in raw.split(",") if m.strip())
            if not value:
                raise ValueError
            args.modes = value
            return
        else:
            value = raw
            if not value:
                raise ValueError
    except ValueError:
        nonneg = least == 0 or field_name == "ttv_snr_min"
        _fail(f"{flag} needs a {'non-negative' if nonneg else 'positive'} "
              f"{'integer' if field_name in ints else 'value'}, got {raw!r}")
    setattr(args, field_name, value)


def main(argv=None):
    args = parse_args(argv)

    if args.show_help:
        print(USAGE)
        return 0
    if args.show_version:
        print(f"turin {__version__}")
        return 0
    if args.show_capabilities:
        from . import capabilities

        print(f"turin {__version__} — external package capabilities")
        print(capabilities.detect().summary())
        return 0

    from .data import lightcurve

    if args.cache_dir:
        lightcurve.set_cache_dir(args.cache_dir)
    if args.clear_cache:
        n = _clear_cache(lightcurve.CACHE_DIR)
        print(f"turin: removed {n} cached light curve(s) from "
              f"{lightcurve.CACHE_DIR}")
        return 0

    if not args.target:
        _fail("no target given. Use --KOI-448.02 or --TOI-406.01")

    from . import pipeline

    return pipeline.run(args)


def _clear_cache(cache_dir):
    if not os.path.isdir(cache_dir):
        return 0
    n = 0
    for fn in os.listdir(cache_dir):
        if fn.endswith(".pkl"):
            os.remove(os.path.join(cache_dir, fn))
            n += 1
    return n


if __name__ == "__main__":       # pragma: no cover
    raise SystemExit(main())
