# Reply: `flux_dev_from_tau` has landed

Answering `metalplanet_prompt.md`. On `main` at
`https://github.com/davidkipping/MetalPlanet` (commit `2aaa4ce`,
2026-09-29). 394 tests green, 59 of them new
(`tests/test_tau_kernel.py`).

It was worth taking, and the measurement is better than the brief
guessed — the memory argument turned out to understate it, because the
gradient buffer is not the only thing that scales with `n_sub`.

## Feature detection

```python
hasattr(metalplanet, "flux_dev_from_tau")   # True
```

Signature is exactly the brief's:

```python
metalplanet.flux_dev_from_tau(
    tau,                        # (n, m) or (m,), time since each point's OWN mid-transit
    period, a, b, r, u1, u2,    # scalars or (n,), per chain
    *, exp_time=0.0,
    integration="contact",      # "contact" | "supersample" | "none"
    n_gl=5, n_sub=1,
) -> mx.array                   # (n, m) flux deviation, exposure-averaged
```

`b` is the impact parameter (`a cos i`), matching `separation_circular`'s
third argument, not the inclination. Circular only, as the brief allowed.

## The measurement

`benchmarks/bench_tau_kernel.py`, M2 Max, 512 chains x 5,000 points x 15
sub-exposures, against `separation_circular` + `flux_dev_metal` +
averaging outside the kernel:

| route | evals/pt | forward | value+grad | peak MB (v+g) | max abs error |
|---|---:|---:|---:|---:|---:|
| expanded axis (today) | 15 | 41.0 ms | 99.7 ms | 2776 | 1.6e-4 |
| in-kernel supersample | 15 | 14.5 ms (**2.8x**) | 28.0 ms (**3.6x**) | 51 (**55x**) | 1.6e-4 |
| in-kernel contact, n_gl=5 | 25 | 24.4 ms (**1.7x**) | 43.9 ms (**2.3x**) | 51 (**55x**) | 1.3e-7 |

Middle row is the *same* arithmetic as the first, so its 2.8x / 3.6x is
purely the cost of materialising the sub-exposure axis in MLX. Your
estimate of ~1.2 GB for the gradient buffer was close on that term alone;
measured peak is 2.78 GB, because the forward intermediates of
`separation_circular` are live at the same time.

The third row is the one you said you wanted, and it is both faster than
what you run today **and** ~1,200x more accurate. Errors are against an
fp64 contact-rule reference at `n_gl=12` — deliberately neither of the
rules being compared, since scoring supersampling against a supersampled
reference flatters it as `n` approaches the reference's own.

At matched accuracy the gap is not a factor of three. Supersampling
converges as O(1/N) on a kinked curve, so 1e-6 needs `n_sub ~ 2,271`,
which at 512 chains would ask for ~368 GB and page. Measured at 7 chains
where both stay resident: **30x** forward, **57x** value+grad, on
160-517x less memory. Drop the Kipping Eq. 40 factor.

## Acceptance, item by item

**Parity with `api.TransitModel` at matched settings.** fp64 graph path
agrees to **3.8e-16 to 5.0e-16** across `integration="contact"`
(n_gl 5 and 7), `supersample_factor` (11 and 15) and instantaneous —
better than the 1e-12 asked for. fp32 kernel **<= 2.3e-7**, inside the
5e-7 kernel-vs-graph tolerance.

**Gradients against fp64 central finite differences**, on the
boundary-heavy set you named (`z ~ r`, `z ~ 1-r`, `z ~ 1+r`, `tau = 0`,
`b = 0`, grazing `b ~ 1-r`), x {none, contact, supersample}: agreement
converges as O(h^2) down to **2e-9**, which is the difference's own
truncation error and not a floor of the gradient.

One caveat worth stating plainly, because it will show up if you run your
own FD check: **a central difference straddling a contact is wrong no
matter how small h is.** The light curve's tau-derivative genuinely jumps
there, so FD reports the average of two different one-sided slopes. The
tests exclude those points explicitly and say why. Instantaneous flux
kinks *at* a contact; the contact-split average kinks half an exposure
either side (a contact entering or leaving the window); a supersampled
average kinks once per node.

**Finite everywhere**: 20,000 points each sitting on a boundary, offsets
down to 1e-7, all three rules — no NaN or inf in any gradient.

## Three things to know before you switch

**1. `integration="none"` is not bit-identical to your current route** —
it agrees to **2.5e-7**, the standing kernel-vs-graph tolerance. The
kernel computes `z` from `tau` in registers with `metal::precise::`
intrinsics; the graph computes it with MLX's elementwise ops in a
different order. Pinned as a test, `test_none_matches_the_route_it_replaces`.

**2. The quadrature's split points are frozen, deliberately, in both
paths.** Moving an interior split point of a continuous integrand adds
+f(c)dc to one interval and -f(c)dc to the next, which cancel — so the
exact integral does not depend on where it is split, and not chasing the
contacts' parameter dependence is exact rather than an approximation.
(The window *ends* are genuine Leibniz boundary terms; the kernel carries
those.) This matters to you because it is what makes fp64 certification
meaningful: the graph path computes the same *function*, not merely the
same value, so the gradient `validate_precision` checks is the gradient
that runs in fp32.

Building it surfaced a real leak worth knowing about if you ever call
`exposure_nodes` directly: it converts contact *phases* to times with
`period / 2 pi`, which gave the graph path a period dependence in the
split points that the kernel did not have. Worth **9e-4** relative on
`d/d(period)` — 2,000x the fp32 noise floor, and it would have looked
like a kernel bug.

**3. fp64 puts itself on the CPU stream.** MLX has no float64 on Metal at
all — even slicing an fp64 array on the GPU stream raises — so this is a
device question, not a dispatch one, and `flux_dev_from_tau` enters
`mx.stream(mx.cpu)` itself rather than raising and expecting the caller
to have known. Your *surrounding* graph is still yours: an fp64
`log_prob_hi` wants its own stream context, exactly as `TransitModel`
does for its fp64 graph.

## Smaller notes

- `tau` is expected within half a period of its own mid-transit — that is
  what "its own" means. Outside that the kernel still returns the right
  answer (`z` is periodic, and the far side is cut by `cos phi <= 0` as
  `separation_circular` does) but the contact rule places its split
  points around the epoch-0 transit, where the flux is zero anyway.
- Out of transit the result is **exactly** 0.0, not a rounding of it, in
  all three modes — so `df0 + flux_dev_from_tau(...)` keeps working.
- Parameters are canonicalised per chain from scalars, 0-d, `(n,)`,
  `(n,1)` or `(1,1)`, and cast to `tau`'s precision; a mixed-width
  parameter will not reach the kernel as a mismatched input.
- Gradients survive `mx.compile(mx.grad(...))` bit-identically to eager —
  pinned as a test, since that is the shape an engine step actually has.
- Bad `integration`, `n_gl`/`n_sub` < 1, negative `exp_time` and a
  3-D `tau` all raise `ValueError`. `exp_time=0.0` silently means
  instantaneous whatever `integration` says.

## What did not come along

The eccentric anchored form. The brief said not to let it enlarge the
job, and it would have: the anchored solve needs per-point Kepler
iteration in the same registers, and the contact geometry becomes
`contact_geometry(a, ecc, esw, ci)` rather than `(a, b)`. `flux_dev_from_tau`
is circular. If turin v2 wants it, it is a contained addition — the
photometric core is already factored out as a device function
(`mp_phot` / `mp_phot_d`) precisely so a second orbit front-end can reuse it.

## Where to look

- `metalplanet/metal.py` — `flux_dev_from_tau`, `_TAU_FWD_SRC`,
  `_TAU_VJP_SRC`, `_tau_graph`
- `tests/test_tau_kernel.py` — 59 tests
- `benchmarks/bench_tau_kernel.py` — the table above
- `README.md`, "Sampling per-transit times"
- `benchmarks/RESULTS.md`, "In-kernel exposure integration"
