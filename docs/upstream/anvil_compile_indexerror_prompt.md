# Task: treat MLX's compile-time `IndexError` as a compile refusal

**Who is asking.** `turin` (github.com/davidkipping/turin) fits Kepler/TESS
transit light curves with anvil. Measured on an Apple M-series Mac, anvil
0.11.0, mlx 0.32.3. turin 0.1.90 works around it (see "What turin does
now"), so nothing is blocking. This is a robustness request, one small
change, and an optional finding about MLX itself.

**Deliverable discipline.** As before: no new dependencies, a test beside
the existing ones, `docs/` updated, version bumped, pushed to
`https://github.com/davidkipping/anvil`. Say what landed and how turin
should detect it.

## The failure

`anvil.optimize` on an oblate-planet target raised, at
`anvil/compile.py:85` (the `self._compiled_fn(*args, **kwargs)` call in
`CompiledOrEager.__call__`):

```
IndexError: unordered_map::at: key not found
```

It is raised by MLX's compiler (a C++ `std::out_of_range` surfaced as
`IndexError`), on the **value+gradient** graph of the float32 optimizer, and
it is deterministic. `is_compile_refusal` accepts only a `RuntimeError`
starting with `[compile]`, and `__call__` guards only the `_evaluate` call
with `except RuntimeError`, so this exception propagates out of `optimize`
and ends the fit. The 0.5.0 behaviour ("an MLX compile refusal makes `run`
continue uncompiled") does not apply to it. 8 of the 11 large targets turin
tried in one batch died this way.

## What was ruled out (turin side, measured)

- **Not stale compile-cache id reuse.** A held reference to the compiled
  functions does not help, and a fresh `mx.compile` of the same function
  fails the same way in the same process. The same function runs eagerly.
- **Not the batch size** (fails at 1 row), **not one epoch**, **not point
  counts** (total, padded or in-transit).
- **It depends on the graph's structure: the number of epoch blocks.** turin
  evaluates the likelihood as a sum over blocks of epochs. For the oblate
  model, the compiled value+gradient passes at 3 blocks and fails at 4 or
  more:
  - KOI-254.01 passes at 3 and fails when forced to 6.
  - KOI-1225.01 passes at 504 epochs (3 blocks) and fails at 508 (4 blocks).
  - Round (spherical) layouts compile at any block count (26 tried).

## Asks

1. **Primary (anvil):** make `CompiledOrEager` treat this failure like a
   compile refusal: fall back to eager, warn once, set `compiled = False`,
   call `on_refusal`. The error is raised at the compiled call, not only at
   `_evaluate`, so the guard has to cover both. I don't know what MLX's
   other compile-time failures look like, so how narrowly to match is your
   call; the string `unordered_map::at` is what is known. It is also your
   call whether an `IndexError` should be treated this way at all, or
   re-raised with a clearer message. Either beats a dead `optimize`.
2. **Optional:** whether this is worth a minimal MLX issue. turin has not
   reduced it to MLX alone; the smallest reproducer is turin's.

## Reproducer

turin's pushed test pins it. On turin 0.1.90 (`turin/likelihood.py`,
`MAX_OBLATE_BLOCKS`; `tests/test_oblate.py::test_an_oblate_target_is_capped_at_three_blocks_because_mlx_cannot_compile_more`):

1. Set `likelihood.MAX_OBLATE_BLOCKS = 99` to remove the cap.
2. Run that test. It builds an oblate target with more than 3 blocks and
   compiles `mx.vjp` of `tr.log_prob` over a 2-row float32 batch, and fails
   with the `IndexError` above. With the cap in place it passes.

## What turin does now

Oblate layouts are capped at 3 epoch blocks, so the failing graph is never
built. The cap costs no memory (blocking bounds a kernel launch, not the
peak: 1.16-1.74 GB at 3 blocks against 1.43-2.61 GB at 1, measured), and
round layouts are untouched. The cap stays whatever anvil does: an eager
fallback on a 60,000-point oblate fit would be far slower than the compiled
one. turin would not need a newer anvil for it, so `MIN_ANVIL` would not
change.
