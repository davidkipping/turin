"""The external packages turin depends on: a minimum anvil, and what it reports.

elome (formerly MetalPlanet), anvil and anvil-gp are developed separately.
turin used to feature-detect each anvil capability it wanted and carry a
fallback for installs that lacked it. Every one of those has since landed
upstream, so turin **requires a minimum anvil** (:data:`MIN_ANVIL`) and uses
its API directly: resumable runs, state save/load, the survivable prior
boundary, per-chain divergence counts, ``ResumeState.with_positions``, the
memory-bounded ``diagnose`` and the end-of-run lost-chain judgement.
:func:`require_anvil` is the one check, and it names the upgrade command
rather than letting an old install fail somewhere deep in a fit.

When turin starts relying on a new upstream feature, raise
:data:`MIN_ANVIL` in the same change, rather than adding detection plus a
fallback. elome is gated per feature, not as a package: ``--law=hybrid4``
needs :data:`MIN_ELOME_HYBRID4` (:func:`require_elome_for_law`), and every
other elome release under that name (0.14.0 on) has everything turin uses,
``ld_basis`` included; ``pyproject.toml`` pins the version turin was
verified against. (Under its old name, MetalPlanet >= 0.7.0 was checked for
``--ld=collapsed`` alone; that gate became vacuous with the rename.)
"""

from __future__ import annotations

import re
from dataclasses import dataclass

#: The oldest anvil turin runs against.
#:
#: 0.7.1 from turin 0.1.60, because turin reads
#: ``results.extras["lost_chains"]`` -- anvil's end-of-run judgement of
#: chains that are lost rather than exploring, made on an eager
#: re-evaluation of the target as well as the kernel's record
#: (0.6.0-0.6.1), which can disagree at a float32 feasibility edge -- and
#: folds it into its own trapped-chain report (``sampling.chain_health``).
#: Two further fixes come with it: since 0.7.0 a chain at a non-finite or
#: failure-sentinel log-density no longer drags ChEES's shared step size to
#: zero for every chain (healthy runs bit-identical), and since 0.5.0 an MLX
#: compile refusal makes ``run`` continue uncompiled rather than abort.
#: 0.3.0 was the first whose ``diagnose`` ranks correctly between 2,095,104
#: and 2^21 rows; 0.4.0 fixed a silent bug for a target mutated between
#: ``run`` calls, which turin's static target never hit (grid-Gibbs moves
#: *positions* through ``ResumeState.with_positions``).
#: 0.9.0 since turin 0.1.71: ``anvil.optimize`` finds the fit's best point
#: and the Laplace starts (``seeding.find_map``), replacing turin's own Adam.
MIN_ANVIL = (0, 9, 0)

UPGRADE_HINT = ('pip install -U "anvil-mcmc @ '
                'git+https://github.com/davidkipping/anvil.git"')
#: The oldest elome whose hybrid4 weights are on LDhybrid's curvature basis
#: (0.17.0, with ``hybrid4_from_q``). Earlier releases read the same four
#: numbers as weights on other shapes -- a different simplex of profiles --
#: so turin 0.1.67's ``q1..q4`` would mean something else there; the gate
#: refuses rather than fail on the import inside a fit.
MIN_ELOME_HYBRID4 = (0, 17, 0)

#: The install command for elome. Separate constant on purpose: pointing a
#: missing-elome message at the anvil command sends the reader to the wrong
#: package.
ELOME_HINT = "pip install -U elome"


def version_tuple(version):
    """Leading numeric components of a version: '0.3.0.dev1' -> (0, 3, 0)."""
    m = re.match(r"\s*(\d+(?:\.\d+)*)", str(version))
    return tuple(int(p) for p in m.group(1).split(".")) if m else ()


def anvil_ok(version):
    return version_tuple(version) >= MIN_ANVIL


def require_anvil():
    """Exit with the upgrade command if the installed anvil is too old."""
    import anvil

    have = getattr(anvil, "__version__", "unknown")
    if not anvil_ok(have):
        need = ".".join(map(str, MIN_ANVIL))
        raise SystemExit(
            f"turin needs anvil >= {need}, but {have} is installed. "
            f"Upgrade with:\n  {UPGRADE_HINT}")
    return have


def elome_hybrid4_ok(version):
    return version_tuple(version) >= MIN_ELOME_HYBRID4


def require_elome_for_law(law):
    """Exit with the upgrade command if ``law`` needs a newer elome."""
    if law != "hybrid4":
        return
    import elome

    have = getattr(elome, "__version__", "unknown")
    if not elome_hybrid4_ok(have):
        need = ".".join(map(str, MIN_ELOME_HYBRID4))
        raise SystemExit(
            f"turin: --law=hybrid4 needs elome >= {need} (hybrid4's weights "
            f"on LDhybrid's curvature basis), but {have} is installed. "
            f"Upgrade with:\n  {ELOME_HINT}")


@dataclass(frozen=True)
class Capabilities:
    """The installed versions, and the one optional feature turin reports."""

    #: elome exposes a tau-input fused kernel with in-kernel exposure
    #: integration. **Used as the primary route** since 0.1.6, for the default
    #: ``geometry="circular"``; ``model.HAS_TAU_KERNEL`` is the same test.
    #: Without it turin falls back to building ``z`` itself and supersampling
    #: in MLX (``model._expanded_flux_dev``), which is the only route for
    #: ``geometry="chord"`` and is *less accurate*: supersampling costs ~2% of
    #: a transit depth and gets ``dF/d(period)`` ~100x wrong with the wrong
    #: sign. So this line reports whether exposures are integrated correctly,
    #: not an unused extra. Every elome release has it.
    elome_flux_dev_from_tau: bool
    anvil_version: str = ""
    elome_version: str = ""

    def summary(self):
        """The run-log header: versions, and whether anvil is new enough."""
        need = ".".join(map(str, MIN_ANVIL))
        ok = anvil_ok(self.anvil_version)
        return "\n".join([
            f"  anvil {self.anvil_version}, elome {self.elome_version}",
            f"  {'yes' if ok else 'NO ':3s}  anvil >= {need} (resume, state "
            "save/load, per-chain divergences, with_positions, bounded "
            "diagnose, lost-chain judgement)"
            + ("" if ok else f"  ->  {UPGRADE_HINT}"),
            f"  {'yes' if self.elome_flux_dev_from_tau else 'no ':3s}  "
            "elome tau-input kernel"
            + (" (in-kernel contact-rule exposure integration and ld_basis; "
               "used for geometry=circular)" if self.elome_flux_dev_from_tau
               else " MISSING: exposures would be supersampled, ~2% of a "
               "depth and dF/dP ~100x wrong  ->  " + ELOME_HINT),
            f"  {'yes' if elome_hybrid4_ok(self.elome_version) else 'NO ':3s}  "
            f"elome >= {'.'.join(map(str, MIN_ELOME_HYBRID4))} (--law=hybrid4 "
            "on the curvature basis)"
            + ("" if elome_hybrid4_ok(self.elome_version)
               else f"  ->  {ELOME_HINT}"),
        ])


def detect():
    """Report the installed packages. Cheap; never raises on an old anvil,
    so ``turin --capabilities`` can say what is wrong."""
    import anvil
    import elome

    return Capabilities(
        elome_flux_dev_from_tau=hasattr(elome, "flux_dev_from_tau"),
        anvil_version=getattr(anvil, "__version__", "unknown"),
        elome_version=getattr(elome, "__version__", "unknown"),
    )


def clear_mlx_cache():
    """Return MLX's cached buffers to the system between rounds.

    MLX keeps freed GPU buffers for reuse, so a long fit's allocation only
    ever grows; measured at 11-20 GB on Kepler targets that ran to the draw
    cap. ``mx.clear_cache`` on current MLX, ``mx.metal.clear_cache`` before.
    """
    import mlx.core as mx

    fn = getattr(mx, "clear_cache", None) or getattr(
        getattr(mx, "metal", None), "clear_cache", None)
    if fn is not None:
        fn()
