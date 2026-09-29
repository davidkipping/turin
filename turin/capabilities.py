"""Feature detection for the external packages, with fallbacks.

MetalPlanet, anvil and anvil-gp are developed separately and turin must work
against whatever is currently published. Anything turin would like but cannot
rely on is detected here rather than assumed, so that a missing feature
degrades in one known place instead of raising somewhere deep in a fit.

The briefs in ``docs/upstream/`` ask for the features that are currently
missing. When one lands, the detection here picks it up with no other change.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass


def _has_param(fn, name):
    try:
        return name in inspect.signature(fn).parameters
    except (TypeError, ValueError):       # builtins, C extensions
        return False


@dataclass(frozen=True)
class Capabilities:
    """What the installed versions of the external packages can do."""

    #: ``anvil.run(..., resume=...)`` continues a run with its adapted
    #: step size, trajectory length and preconditioner frozen. Without it an
    #: extension has to re-run warmup, which is both slower and not a
    #: continuation of the same Markov chain.
    anvil_resume: bool
    #: ``Results.save_state`` / ``anvil.load_state`` for cross-process resume.
    anvil_state_io: bool
    #: per-chain divergence counts in ``Results.extras``. ``None`` because it
    #: is a property of a completed run, not of the API surface: use
    #: :func:`per_chain_divergences` on a ``Results`` to find out.
    anvil_per_chain_divergences: bool | None
    #: a progress callback on ``anvil.run``.
    anvil_callback: bool
    #: anvil's bounded-parameter log-Jacobian stays finite at the prior edge,
    #: so a chain that reaches one can come back. A behaviour fix with no API
    #: surface, so it is probed by evaluating the transform directly.
    anvil_boundary_survivable: bool
    #: MetalPlanet exposes a tau-input fused kernel with in-kernel exposure
    #: integration. **Detected but not yet used**: turin builds ``z`` itself
    #: and calls ``flux_dev_metal``, which is correct and is what every
    #: accuracy figure in the tests was measured against. Switching over is a
    #: performance change that needs its own parity tests, so it is deliberate
    #: rather than automatic.
    metalplanet_flux_dev_from_tau: bool
    anvil_version: str = ""
    metalplanet_version: str = ""

    def summary(self):
        """One line per capability, for the run log."""
        rows = [
            ("anvil resume (frozen adaptation)", self.anvil_resume,
             "extensions re-run warmup"),
            ("anvil state save/load", self.anvil_state_io,
             "resume state stored by turin, chains restart"),
            ("anvil survivable prior boundary", self.anvil_boundary_survivable,
             "a chain reaching a bound is lost; the MAP-centred init ball "
             "keeps chains away from one"),
            ("anvil progress callback", self.anvil_callback,
             "anvil prints its own progress"),
        ]
        out = [f"  anvil {self.anvil_version}, "
               f"MetalPlanet {self.metalplanet_version}"]
        for name, have, fallback in rows:
            out.append(f"  {'yes' if have else 'no ':3s}  {name}"
                       + ("" if have else f"  ->  {fallback}"))
        out.append("  ?    anvil per-chain divergences  ->  checked per run")
        out.append(
            f"  {'yes' if self.metalplanet_flux_dev_from_tau else 'no ':3s}  "
            "MetalPlanet tau-input kernel (available but not yet used: turin "
            "builds z itself)")
        return "\n".join(out)


def detect():
    """Probe the installed packages. Cheap; safe to call more than once."""
    import anvil
    import metalplanet

    results_cls = getattr(anvil, "Results", None)
    return Capabilities(
        anvil_resume=_has_param(anvil.run, "resume"),
        anvil_state_io=(hasattr(results_cls, "save_state")
                        and hasattr(anvil, "load_state")),
        anvil_per_chain_divergences=None,
        anvil_callback=_has_param(anvil.run, "callback"),
        anvil_boundary_survivable=_boundary_survivable(),
        metalplanet_flux_dev_from_tau=hasattr(metalplanet,
                                              "flux_dev_from_tau"),
        anvil_version=getattr(anvil, "__version__", "unknown"),
        metalplanet_version=getattr(metalplanet, "__version__", "unknown"),
    )


def _boundary_survivable():
    """Does a bounded parameter's log-Jacobian stay finite deep in the tail?

    The probe anvil's own reply suggests: before the fix ``mx.sigmoid(25)``
    saturates and ``log_det_jac`` returns ``-inf``, which made the boundary
    absorbing under ChEES.
    """
    try:
        import mlx.core as mx
        from anvil import ParamSpec, Transform

        tr = Transform([ParamSpec("p", lo=0.0, hi=1.0)])
        return bool(mx.isfinite(tr.log_det_jac(mx.array([[25.0]]))).item())
    except Exception:
        return False


def per_chain_divergences(results):
    """Per-chain divergence counts, or ``None`` if anvil does not report them.

    anvil accumulates these internally as a ``(n_chains,)`` array and then
    sums them into ``extras["n_divergent"]``; the brief in
    ``docs/upstream/anvil_prompt.md`` asks for the vector to be kept.
    """
    extras = getattr(results, "extras", {}) or {}
    for key in ("divergent_per_chain", "n_divergent_per_chain"):
        if key in extras:
            return extras[key]
    return None
