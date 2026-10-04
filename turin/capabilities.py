"""The external packages turin depends on: a minimum anvil, and what it reports.

MetalPlanet, anvil and anvil-gp are developed separately. turin used to
feature-detect each anvil capability it wanted and carry a fallback for
installs that lacked it. Every one of those has since landed upstream, and
anvil 0.3.0 also fixed a silent rank corruption in ``diagnose`` that older
versions carry, so turin now **requires anvil >= 0.3.0** and uses its API
directly: resumable runs, state save/load, the survivable prior boundary,
per-chain divergence counts, ``ResumeState.with_positions`` and the
memory-bounded ``diagnose``. :func:`require_anvil` is the one check, and it
names the upgrade command rather than letting an old install fail somewhere
deep in a fit.

When turin starts relying on a new upstream feature, raise
:data:`MIN_ANVIL` (or add a MetalPlanet minimum the same way) in the same
change, rather than adding detection plus a fallback.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

#: The oldest anvil turin runs against. 0.3.0 is the first with everything
#: turin uses, and the first whose ``diagnose`` ranks correctly between
#: 2,095,104 and 2^21 rows.
#:
#: **Not raised to 0.4.0, deliberately.** anvil 0.4.0 fixes a silent
#: correctness bug -- ``mx.compile`` froze whatever a kernel's traced graph
#: read from the *target*, so a target mutated between ``run`` calls kept
#: being sampled as it was at the first trace, giving a healthy-looking run
#: of the old posterior. turin cannot hit it: its target holds only static
#: tensors, and grid-Gibbs moves *positions* through
#: ``ResumeState.with_positions`` without touching the target, which anvil
#: documents as bit-identical. Per the rule below, the minimum rises with
#: the first call that needs it.
#:
#: **The first change that mutates the target between segments must raise
#: this to (0, 4, 0)** -- a Gibbs block held on the target, a tempering
#: beta, a swapped dataset, a retrained surrogate. The failure is silent and
#: survives every convergence check, so it will not be caught downstream.
MIN_ANVIL = (0, 3, 0)

UPGRADE_HINT = ('pip install -U "anvil-mcmc @ '
                'git+https://github.com/davidkipping/anvil.git"')
#: The same, for MetalPlanet. Separate constant on purpose: pointing a
#: missing-MetalPlanet message at the anvil command sends the reader to the
#: wrong package.
MP_UPGRADE_HINT = ('pip install -U "metalplanet @ '
                   'git+https://github.com/davidkipping/MetalPlanet.git"')


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


@dataclass(frozen=True)
class Capabilities:
    """The installed versions, and the one optional feature turin reports."""

    #: MetalPlanet exposes a tau-input fused kernel with in-kernel exposure
    #: integration. **Used as the primary route** since 0.1.6, for the default
    #: ``geometry="circular"``; ``model.HAS_TAU_KERNEL`` is the same test.
    #: Without it turin falls back to building ``z`` itself and supersampling
    #: in MLX (``model._expanded_flux_dev``), which is the only route for
    #: ``geometry="chord"`` and is *less accurate*: supersampling costs ~2% of
    #: a transit depth and gets ``dF/d(period)`` ~100x wrong with the wrong
    #: sign. So this line reports whether exposures are integrated correctly,
    #: not an unused extra.
    metalplanet_flux_dev_from_tau: bool
    anvil_version: str = ""
    metalplanet_version: str = ""

    def summary(self):
        """The run-log header: versions, and whether anvil is new enough."""
        need = ".".join(map(str, MIN_ANVIL))
        ok = anvil_ok(self.anvil_version)
        return "\n".join([
            f"  anvil {self.anvil_version}, "
            f"MetalPlanet {self.metalplanet_version}",
            f"  {'yes' if ok else 'NO ':3s}  anvil >= {need} (resume, state "
            "save/load, per-chain divergences, with_positions, bounded "
            "diagnose)" + ("" if ok else f"  ->  {UPGRADE_HINT}"),
            f"  {'yes' if self.metalplanet_flux_dev_from_tau else 'no ':3s}  "
            "MetalPlanet tau-input kernel"
            + (" (in-kernel contact-rule exposure integration; used for "
               "geometry=circular)" if self.metalplanet_flux_dev_from_tau else
               " MISSING: exposures would be supersampled, ~2% of a depth and "
               "dF/dP ~100x wrong  ->  " + MP_UPGRADE_HINT),
        ])


def detect():
    """Report the installed packages. Cheap; never raises on an old anvil,
    so ``turin --capabilities`` can say what is wrong."""
    import anvil
    import metalplanet

    return Capabilities(
        metalplanet_flux_dev_from_tau=hasattr(metalplanet,
                                              "flux_dev_from_tau"),
        anvil_version=getattr(anvil, "__version__", "unknown"),
        metalplanet_version=getattr(metalplanet, "__version__", "unknown"),
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
