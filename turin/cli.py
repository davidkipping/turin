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
from .model import B_PRIORS, GEOMETRIES
from .profile import PROFILE_MODES

#: ``--PL`` accepts any solve mode, or "auto" to measure and choose.
PL_CHOICES = ("auto",) + tuple(PROFILE_MODES)

TARGET_RE = re.compile(r"^--(KOI-\d+\.\d+|TOI-\d+\.\d+)$", re.IGNORECASE)
MODES = ("lineph", "ttv")
SAMPLERS = ("chees", "ensemble")

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
}
_VALUE_FLAGS = {
    "--chains": "chains",
    "--bprior": "b_prior",
    "--TTVmax": "ttv_max_min",
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
}

USAGE = f"""turin {__version__} — GPU transit fitting for Kepler and TESS

usage: turin --KOI-448.02 [options]
       turin --TOI-406.01 [options]

  --fresh                 ignore existing resume state for this lineage
  --extend1 / --extend2   extend the LinEph / TTV fit by one round
  --modes=lineph,ttv      which fits to run (default both)
  --tag=NAME              namespace this run into its own resume lineage

  --chains=N              sampling chains (default 512, raised to 4*dim)
  --sampler=chees|ensemble  gradient-based (default) or gradient-free
  --warmup=N --samples=N --max-samples=N
  --leapfrog=N            ChEES trajectory cap (default 128)
  --seed=N

  --bprior=transiting|nongrazing|box   (b, k) prior (default transiting)
  --nongrazing            alias for --bprior=nongrazing
  --TTVmax=MINUTES        declared TTV amplitude; sets the timing priors
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
  --geometry=circular|chord   true circular orbit (default) or hurin's chord

  --sc                    prefer short cadence
  --cache-dir=PATH --outdir=PATH
  --clear-cache           delete cached light curves and exit
  --capabilities          report what the installed anvil/MetalPlanet can do
  --version --help

Against hurin >= 0.1.68 the orbit model is the only model difference:
--geometry=chord --PL=ratio reproduces hurin. See docs/hurin-differences.md.
"""


@dataclass
class Args:
    target: str | None = None
    show_version: bool = False
    show_help: bool = False
    show_capabilities: bool = False
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
    tag: str | None = None
    sampler: str = "chees"
    profile_mode: str = "auto"
    geometry: str = "circular"
    warmup: int = 400
    samples: int = 300
    max_samples: int = 16384
    max_leapfrog: int = 128
    seed: int = 0
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
    for mode in args.modes:
        _choice("modes", mode, MODES)
    if args.tag is not None and not re.fullmatch(r"[A-Za-z0-9_-]+", args.tag):
        _fail(f"--tag={args.tag!r} must match [A-Za-z0-9_-]+")
    return args


def _assign(args, field_name, flag, raw):
    ints = {"chains", "warmup", "samples", "max_samples", "max_leapfrog",
            "seed"}
    try:
        if field_name in ints:
            value = int(raw)
            if value <= 0:
                raise ValueError
        elif field_name == "ttv_max_min":
            value = float(raw)
            if value <= 0:
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
        _fail(f"{flag} needs a positive "
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
