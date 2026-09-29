"""turin: GPU transit fitting for Kepler and TESS.

Profile-likelihood Legendre detrending (from hurin) with the MetalPlanet
forward model and the anvil sampler, on Apple Silicon via MLX.

The version here is the runtime source of truth (the install is editable);
keep it in sync with pyproject.toml and add a VERSIONS.md row per commit.
"""

__version__ = "0.1.3"
