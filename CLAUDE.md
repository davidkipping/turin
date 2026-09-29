# CLAUDE.md

Guidance for Claude Code (claude.ai/code) working in this repository.

## Project overview

**turin** fits transiting-exoplanet light curves from Kepler and TESS on
the Apple GPU. It is the successor to **hurin** (`../hurin`), keeping that
package's science pipeline — MAST download, NASA Exoplanet Archive
ephemerides, per-epoch windowing, cross-validated Legendre order
selection, and **profile-likelihood polynomial detrending** — while
replacing the two compute-bound halves:

| | hurin (CPU) | turin (GPU) |
|---|---|---|
| forward model | jaxoplanet | **MetalPlanet** (`../MetalPlanet`, MLX/Metal) |
| sampler | NumPyro NUTS, 2 chains | **anvil** (`../anvil`): ChEES-HMC, hundreds of chains |

Correlated-noise (GP) likelihoods via `anvil-gp` (`../anvil-gp`) are a
planned future direction, not in v1; `likelihood.py` keeps a noise-model
seam for them.

**Scope of v1:** LinEph and TTV fit modes (pyhurin's batch pipeline),
circular orbits, white noise. Deferred: TTV-PL (hurin's Newton-profiled
per-epoch times), GPs, eccentricity, the Flask web frontend.

## The three external packages

MetalPlanet, anvil and anvil-gp are **strictly external**. Never edit
them from this repo. Where turin needs something they lack, the workflow
is to write a brief in `docs/upstream/` for that package's own Claude
session; the user runs it there and reports back. Existing briefs:

- `docs/upstream/anvil_prompt.md` — resumable runs (needed for
  `--extend`/auto-resume), the ChEES bounded-parameter boundary trap,
  per-chain divergence counts.
- `docs/upstream/metalplanet_prompt.md` — optional: a `tau`-input fused
  kernel with in-kernel exposure integration.

`turin/capabilities.py` feature-detects every upstream capability and
falls back when it is absent. **turin must always run against the
packages as currently published on GitHub.** When adding a dependency on
an upstream feature, add the detection and the fallback in the same
change.

## Architecture

```
turin/
  cli.py          kwargs CLI (argv regex grammar + did-you-mean)
  data/           lightcurve.py, ephemeris.py, preprocessing.py (ported from hurin)
  prep.py         windows, epoch segmentation, centering, Legendre CV, exposure nodes
  model.py        MLX transit model: params -> per-epoch tau -> z -> flux
  profile.py      batched per-epoch Legendre profile solve (exact | ratio)
  likelihood.py   epoch-block-chunked Gaussian logL, priors, anvil Transform/target
  params.py       LinEph/TTV parameter specs, b-k prior modes, derived quantities
  seeding.py      template-sweep tau seeds, batched multi-start MAP
  sampling.py     anvil driver, convergence loop, diagnostics, trapped-chain checks
  outputs.py      CSV/PDF products, resume state
  capabilities.py upstream feature detection + fallbacks
```

## Load-bearing invariants

### Epoch-centred time coordinates (float32 safety)

The GPU is float32, whose resolution at absolute BKJD/BTJD magnitudes
(~1500 d) is 5-10 s — a hard floor on timing precision, and MetalPlanet
measures 1.7e-4 in flux after 1,000 orbits without this. So, exactly as in
hurin and as `metalplanet.orbit.epoch_center_times` prescribes:

- All large-minus-large subtraction happens **once, in float64 NumPy**, on
  the host. Times reaching MLX are per-epoch residuals; epoch numbers are
  exact small integers.
- **Sampled parameters are offsets, never absolutes**: `dP` and `dtau0`
  from the recentred input ephemeris, `dtau_i` from each epoch's predicted
  time. Absolutes are reconstructed in float64 immediately after sampling,
  so every downstream consumer sees physical units.
- `anvil.ParamSpec(report_offset=...)` carries the absolute value back out
  in float64.

### The float32 discipline (anvil's five rules)

Read `../anvil/docs/precision.md` before touching `likelihood.py`. The
ones that bite:

- **`mx.array(np_float64_array)` silently yields float32.** Always pass
  `dtype=`. A `np.float64` scalar entering a compiled graph force-evaluates
  it mid-trace.
- **Recentre the likelihood**: accumulate `0.5*(1-r)*(1+r)` and keep the
  exact `-N/2` as a float64 host constant, rather than `-0.5*r**2`. This
  sharpens HMC's energy difference, which `validate_precision` cannot see.
- **Stay in deviation space end to end.** The model returns `f - 1`
  (`model.transit_flux_dev`), the design stores `y - 1` (`y_dev`), and the
  residual is formed as `(y_dev - f_dev) - (1 + f_dev)*L c`. Never subtract
  two quantities near 1 to recover one near 1e-4: doing so cancels four
  digits, and it measurably cost 1.6% on `dlogL/dk` and 8x on the
  log-density error before it was fixed.
- Run `anvil.validate_precision(target, ball)` before every production
  fit, and `certify` after. The measured baseline for a 772-point,
  9-epoch fit is a median float32 error of 3e-4 against float64.
- **float32 gradients are less accurate than the float32 log-density**, and
  that is fine. `dlogL/dk` carries ~8e-3 relative error because `k` acts
  through three partly cancelling channels. It does not bias the posterior:
  leapfrog with any smooth force field is still volume-preserving and
  reversible, and the Metropolis step uses the log-density. Validate
  gradients against float64 autodiff (and closed forms), not against the
  float32 path.

### The profile likelihood

Each epoch's local baseline is `g(t) = 1 + L c` with `L` the Legendre
design matrix on times mapped to [-1, 1] within the epoch window; the
total model is `f_transit * g`. The coefficients `c` are **not sampled**:
they are solved analytically inside every log-density evaluation (a
profile likelihood — a plug-in MLE of the nuisance parameters, with no
`-0.5 log det` Occam factor).

Two modes, `--profile=`:

- **`exact`** (default): the true flux-space profile, design matrix
  `F L` (with `F = diag(f_transit)`) against target `y - f_transit`.
- **`ratio`**: hurin's form, a WLS fit in ratio space (`y / f_transit`,
  weights `1/sigma^2`), which drops the `f_transit` factors from the
  weights. It agrees with `exact` to O(depth) and exists to reproduce
  hurin's likelihood surface for validation.

Implementation notes: design matrices are built once in NumPy and uploaded
(never rebuilt inside the traced likelihood); columns above each epoch's
CV-selected order are masked, with masked diagonal entries set to 1 so the
batched shape stays fixed; `+1e-10 I` Tikhonov regularization as in hurin.
**MLX `linalg` runs on the CPU only**, so the per-epoch solve is an
unrolled Cholesky in elementwise MLX ops (at most 6x6, fixed size,
differentiable, GPU-resident).

### Why turin does not use MetalPlanet's sampler-facing API

`metalplanet.anvil.make_quad_transit_flux` and `make_transit_target` bake
in a linear ephemeris, offer no exposure integration, carry only a scalar
`df0` baseline, and parametrize by a/R*. turin needs per-epoch mid-times,
long-cadence integration, per-epoch polynomials, and hurin's
`T14` parametrization. So turin builds `tau` and `z` itself and calls
**`metalplanet.metal.flux_dev_metal(z, r, u1, u2)`** — a fused fp32 kernel
with a VJP in `z`, which falls back to the analytic fp64 path on the CPU
stream (that fallback is turin's `log_prob_hi`). Similarly turin writes its
own likelihood rather than using `ChunkedGaussianLogLike`, which chunks the
data axis across epoch boundaries and so cannot hold a per-epoch solve.

### Known upstream hazards

- **ChEES can permanently freeze a chain at a bounded parameter's edge**
  (`../anvil-gp/docs/anvil-chees-boundary.md`). All of turin's parameters
  are bounded and transit posteriors sit at edges (grazing geometries,
  low-signal epochs). Mitigated by the MAP-centred 1e-3 init ball; a
  frozen chain is reported, not silently pooled.
- **The stretch move traps chains in secondary modes and the reported
  width inherits it** — 4x too wide, R-hat 1.037 the only tell
  (`../anvil-gp/docs/anvil-stretch-width.md`). turin therefore always runs
  a trapped-chain check: per-chain median log-prob and acceptance against
  the ensemble, with widths broken down by chain before they are quoted.
- **`dense=True` silently downgrades** to diagonal below `4*dim` chains.

## Conventions inherited from hurin

- **One patch bump per commit.** 0.1.N is commit N; update
  `pyproject.toml`, `turin/__init__.py` and a `VERSIONS.md` row together.
- **Provenance stamping** on every product: `# turin <version> | <launch
  command>` on **line 2** of CSVs (after the header, because
  `np.genfromtxt(names=True)` treats a leading comment as the header), a
  `turin_version.txt` member appended after the CSV in tarballs (readers
  use `getmembers()[0]`), PDF Creator/Subject metadata, and version +
  command keys in resume state.
- **Product filenames and columns match hurin exactly**, so existing
  analysis scripts keep working: 7 LinEph products, 9 TTV.
- `--tag=<name>` namespaces a run into an independent auto-resume lineage,
  inserted into every product filename before the extension.
- Resume state records `bprior`, `ttv_max`, `tag`, `profile`, `sampler`
  and chain count; a mismatched resume exits naming the stored value.

## Environment

Dev environment is `turin/.venv` (Python 3.11), with MetalPlanet and anvil
installed editable from the sibling clones:

```bash
.venv/bin/python -m pip install -e ../MetalPlanet -e ../anvil
.venv/bin/python -m pip install -e . --no-deps
.venv/bin/python -m pytest
```

No conda environment on this machine has MLX; the `hurin` conda env
(`~/miniconda3/envs/hurin`) has jaxoplanet and NumPyro and is used only to
generate hurin parity fixtures.
