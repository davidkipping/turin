# Version history

One patch bump per commit: 0.1.N corresponds to commit number N
(zero-based from the initial commit). Every commit bumps the version in
`pyproject.toml` and `turin/__init__.py` (kept in sync; `__init__.py` is
the runtime source of truth since the install is editable) and adds a row
here.

| Version | Commit | Date | Change |
|---------|--------|------|--------|
| 0.1.12 | — | 2026-09-29 | Docs: README gains a Performance section (measured 0.1.7 → now on two real targets); hurin-differences records why KOI-5162.01's wall time is not comparable across the supersampling → tau-kernel switch. (MODEL_REV unchanged at 3) |
| 0.1.11 | — | 2026-09-29 | Docs: restore CLAUDE.md's point-order section (accidentally removed in 0.1.9), add the end-to-end KOI-448.02 timing (549 s → 118 s, medians within 0.02σ), retire the stale "tau kernel is 2x slower" note. (MODEL_REV unchanged at 3) |
| 0.1.10 | — | 2026-09-29 | Version bump only: the intended docs edit failed in a script and landed in 0.1.11 |
| 0.1.9 | — | 2026-09-29 | **MODEL_REV 2→3.** N_GL 9→5: judged on the log-likelihood rather than the kernel's own dF/dP, five is 40x below the float32 noise floor (sd 5e-4) and now 1.55x faster (14.1 → 9.1 ms on KOI-448.02) |
| 0.1.8 | — | 2026-09-29 | Feed MetalPlanet's kernel points in phase order and park padded slots at quadrature, removing SIMD divergence: log-density value+grad 3.3x faster on KOI-448.02 (45.8 → 14.1 ms). Values bit-identical. (MODEL_REV unchanged at 2) |
| 0.1.7 | — | 2026-09-29 | Drop the chain-by-chain float64 workaround: MetalPlanet b3e9872 fixed the graph-path batching. Output bit-identical. (MODEL_REV unchanged at 2) |
| 0.1.6 | — | 2026-09-29 | **MODEL_REV 1→2.** Integrate MetalPlanet's tau kernel: exposures integrated in-kernel by the contact rule, fixing a 1.5e-4 flux error (~2% of a depth) and a dF/d(period) that was ~100x wrong with the wrong sign. Use anvil's resume: extensions are true continuations and their draws pool |
| 0.1.5 | — | 2026-09-29 | --PL defaults to auto: measure all three solves per target and choose. Add MODEL_REV likelihood-revision flagging. Private GitHub repo. (MODEL_REV unchanged at 1) |
| 0.1.4 | — | 2026-09-29 | Measure the exact/hybrid crossover end to end (~14 basis columns) and document which --PL mode suits which regime |
| 0.1.3 | — | 2026-09-29 | Rename --profile to --PL and add --PL=hybrid (preconditioned refinement); drop the pre-0.1.68 hurin limb-darkening map now that hurin is fixed |
| 0.1.2 | — | 2026-09-29 | Pre-factorize the static ratio-mode normal matrix (1.7-5.5x faster, growing with basis size); track hurin 0.1.68's limb-darkening fix |
| 0.1.1 | — | 2026-09-29 | Leave Tail_ESS blank (anvil reports bulk ESS only); real-data validation against hurin on KOI-448.02 and KOI-5162.01 |
| 0.1.0 | — | 2026-09-28 | Initial build: NumPy pipeline ported from hurin, MLX forward model, profiled Legendre likelihood, template-sweep seeding, anvil driver, hurin-format products, kwargs CLI |
