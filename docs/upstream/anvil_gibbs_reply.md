# anvil's reply: `ResumeState.with_positions` (anvil `849159f`)

Landed as proposed, plus one optional argument:

```python
ResumeState.with_positions(u, target, kernel=None) -> ResumeState
```

- Recomputation goes through a new `Kernel.refresh(state, u, target)`, a
  **classmethod** so a state loaded from disk can reach it by kernel name.
  The base implementation covers `{"u", "log_prob"}` (the ensemble kernel)
  and raises `NotImplementedError` for any cached key it does not rebuild;
  ChEES overrides it to rebuild `grad` too. A kernel whose `refresh` is an
  instance method needs `kernel=` passed an instance.
- Validates shape and finiteness of `u`, and refuses positions where the
  target is not finite ("would strand them").
- `iteration`, `seed`, `params` and `kernel_ckpt` are carried over, so the
  next segment draws the same keys it would have without the move.
- Returns a new `ResumeState`; the original is untouched.

turin needs no change to use it: `capabilities.set_positions` already called
`resume_state.with_positions(u, target)` when present.

## Verified on turin's side (0.1.20)

- `turin --capabilities` reports it; the venv was moved to `849159f`.
- `tests/test_gibbs.py::test_round_loop_with_anvils_with_positions_matches_the_fallback`
  runs the real round loop with grid-Gibbs through anvil's method and through
  turin's fallback, and requires **bit-identical** chains. They are.
- Full suite: 185 passed, 14 skipped.

## A note on the brief itself

The anvil session reported that it could not find `turin/turin/gibbs.py` or
`capabilities.set_positions`, which the brief told it to read. The turin
commit that added them (`57651fa`) had not been pushed when the brief was
handed over. It worked from the brief's own description instead. **Push the
turin commit a brief refers to before handing the brief over.**

## Later: fallback removed (turin 0.1.24)

turin now requires anvil >= 0.3.0 (`capabilities.require_anvil`), so
turin's fallback and the bit-identity test against it were removed. Validated by a golden run before and after: chains,
log-probabilities, divergences, R-hat and trapped-chain flags bit-identical;
ESS within 1.2e-6 relative (float32 regrouping in anvil's autocovariance).
