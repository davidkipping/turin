# Task: a per-point-time entry point with in-kernel exposure integration for `MetalPlanet`

**This is a performance and accuracy request, not a blocker.** turin works
today on MetalPlanet as shipped, by the route described under "What turin
does today" below. Please weigh it on its own merits; if you take it, it
also closes a real gap in MetalPlanet's own sampler story (exposure
integration exists only on the batman-style frontend, not on the path a
sampler uses).

**Who is asking.** `turin`
(`/Users/dkipping/Storage1/Work/Documents/Transit_Work/CODES/turin`) is the
GPU successor to `hurin`: Kepler/TESS transit fitting with MetalPlanet as
the forward model and anvil as the sampler. Two things about turin drive
this request:

1. **It fits long-cadence data.** Kepler's 29.4-minute integration is
   comparable to a short-period planet's entire ingress, so finite-exposure
   integration is mandatory, not a refinement. hurin computed a
   supersampling factor from Kipping (2010) Eq. 40 and evaluated the model
   at every sub-exposure node.
2. **It fits per-transit mid-times (TTV mode).** Each epoch has its own
   mid-transit time, 30-80 of them sampled jointly with the shape
   parameters. So the linear ephemeris baked into
   `epoch_center_times` + `tau_from_epochs` + the fused model kernel does
   not describe turin's model, and `make_quad_transit_flux`'s `(v, x)`
   contract cannot express it.

## Context to read first

Paths relative to
`/Users/dkipping/Storage1/Work/Documents/Transit_Work/CODES/`:

1. `MetalPlanet/metalplanet/orbit.py` — `separation_circular` and
   `tau_from_epochs`; the `tau -> z` half of what is being asked for.
2. `MetalPlanet/metalplanet/metal.py` — `flux_dev_metal` (the `z`-input
   fused kernel turin uses today) and `make_model_core_metal` (the
   whole-model kernel, whose VJP returns zeros for the time input because
   times are data there — the assumption turin breaks).
3. `MetalPlanet/metalplanet/exposure.py` — `contact_offsets`,
   `contact_geometry`, `exposure_nodes`: the branchless five-interval
   Gauss-Legendre rule, already written, currently reachable only from
   `api.TransitModel(integration="contact")`.
4. `MetalPlanet/metalplanet/api.py` lines 196-219 — the supersampling
   expansion and the frontend's fp32 time re-centring.
5. `MetalPlanet/README.md` lines 192-197 — the measured accuracy table
   that motivates preferring the contact rule over supersampling.

## What turin does today

turin computes the separation itself and calls the `z`-input kernel:

```python
# tau: (n_chains, n_epochs * max_pts * n_sub), built from turin's own
# per-epoch mid-times (TTV) or linear ephemeris (LinEph), in
# epoch-centred float64-preprocessed coordinates
z   = separation_circular(tau, period, b, a)        # MLX graph
dev = flux_dev_metal(z, k, u1, u2)                  # fused kernel, VJP in z
f   = 1.0 + mean_over_subexposures(dev)
```

This is correct and differentiable — `flux_dev_metal`'s VJP in `z` chains
through `separation_circular` cleanly, and MetalPlanet's fp64 CPU fallback
gives turin its `log_prob_hi` path for free. Two costs:

- **The sub-exposure axis materializes.** Every intermediate, and every
  grid the analytic VJP writes, is `n_sub` times larger than the data.
  At 512 chains × 50 epochs × 100 points × `n_sub` = 15, that is 38 M
  points forward and, at the ~32 B/point the VJP transiently needs,
  ~1.2 GB of gradient buffer for what is a 5,000-point light curve.
  `n_sub` is set by accuracy, so it cannot simply be reduced.
- **`separation_circular` is an unfused graph.** Roughly a dozen
  elementwise kernels over that inflated array, plus their backward pass,
  where the fused path does the same work inside one launch.

## What is being asked for

A `tau`-input companion to `flux_dev_metal`, with the exposure integration
inside the kernel so the sub-exposure axis never reaches MLX:

```python
flux_dev_from_tau(
    tau,                 # (n, m) fp32: time since each point's OWN mid-transit
    period, a, b, r, u1, u2,   # (n,) or scalar, broadcast as flux_dev_metal does
    *, exp_time=0.0,           # exposure duration, same units as tau
    integration="contact",     # "contact" | "supersample" | "none"
    n_gl=5, n_sub=1,
) -> mx.array                  # (n, m) flux deviation, exposure-averaged
```

- **VJP in `tau` and in every parameter.** The `tau` gradient is the part
  that matters and the part the existing whole-model kernel does not
  provide: turin's `tau` depends on sampled parameters (a period offset, a
  reference-epoch offset, or one offset per epoch), so it is emphatically
  not data. With `d(dev)/d(tau)` available, turin builds `tau` however its
  parametrization requires and lets MLX chain the rest — which keeps
  MetalPlanet out of the business of knowing about TTVs, linear
  ephemerides, or turin's parameter vector.
- **`integration="contact"` is the one turin most wants**, because
  MetalPlanet's own numbers say it is strictly better than supersampling:
  25 evaluations per exposure for 8.9e-8 maximum error against 101
  evaluations for 1.7e-5, on a 29-minute exposure. turin would drop
  hurin's Kipping Eq. 40 supersampling factor entirely in favour of it.
  The nodes are already differentiable in the parameters
  (`tests/test_exposure.py:143-154`), and `exposure.py` already computes
  them — the request is to run that rule inside the kernel, accumulating
  the weighted sum per output point, rather than expanding nodes into an
  MLX array.
- `integration="none"` should be exactly today's `flux_dev_metal`
  behaviour with `tau -> z` folded in, so turin has one entry point for
  both cadences.
- Same graceful-degradation contract as `flux_dev_metal`: fp64, CPU
  stream, or a machine where the kernel probe fails falls back to the MLX
  graph path (`separation_circular` + `flux_dev_analytic` +
  `exposure_nodes`). turin depends on that fp64 fallback for anvil's
  `validate_precision` and `certify`, so please keep it.

A circular orbit is all turin v1 needs. If the eccentric anchored form
comes along for free, good, but do not let it enlarge the job.

## Acceptance

- Parity with `api.TransitModel` at matched settings: fp64 path agreeing
  with `integration="contact"` and with `supersample_factor` to ~1e-12,
  fp32 kernel to ≤5e-7 (the existing kernel-vs-graph tolerance).
- Gradients in `tau` and all parameters against fp64 central finite
  differences, on a boundary-heavy point set (`z ≈ r`, `z ≈ 1-r`,
  `z ≈ 1+r`, `tau = 0`, `b = 0`, grazing `b ≈ 1-r`), and finite
  everywhere — the both-branch sanitization invariant.
- A memory and throughput measurement against the current
  `separation_circular` + `flux_dev_metal` + host-side averaging route, at
  something like 512 chains × 5,000 points × 15 sub-exposures, reported in
  `docs/` or `benchmarks/`. If it does not actually win, that is a useful
  answer too and turin will stay on the present route.

## What turin does if this does not land

Nothing changes: turin keeps the `separation_circular` + `flux_dev_metal`
route above, chunked over epoch blocks to bound the gradient buffer, and
computes its own supersampling nodes from hurin's Kipping Eq. 40 factor.
turin feature-detects (`hasattr(metalplanet, "flux_dev_from_tau")`) and
switches over if it appears, so this can land whenever it is worth landing.
