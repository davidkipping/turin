"""turin: GPU transit fitting for Kepler and TESS.

Profile-likelihood Legendre detrending (from hurin) with the MetalPlanet
forward model and the anvil sampler, on Apple Silicon via MLX.

The version here is the runtime source of truth (the install is editable);
keep it in sync with pyproject.toml and add a VERSIONS.md row per commit.
"""

__version__ = "0.1.38"

#: Revision of the *likelihood itself*, as distinct from the package version.
#:
#: Bump this in any commit that changes the value of the log-density for a
#: fixed parameter vector -- a model correction, a prior change, a different
#: profile formulation -- and say so in the VERSIONS.md row. Chains sampled
#: under an older revision cannot be continued, and
#: :meth:`turin.outputs.ResumeState.check_model_rev` refuses to try.
#:
#: This is deliberately separate from the configuration guards in
#: ``ResumeState.GUARDS``, which catch a user *asking* for a different model.
#: MODEL_REV catches the model changing underneath an unchanged command line.
#: Following hurin, which added the same mechanism in 0.1.68 after a
#: limb-darkening fix silently altered its likelihood.
MODEL_REV = 5
