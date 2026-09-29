# Version history

One patch bump per commit: 0.1.N corresponds to commit number N
(zero-based from the initial commit). Every commit bumps the version in
`pyproject.toml` and `turin/__init__.py` (kept in sync; `__init__.py` is
the runtime source of truth since the install is editable) and adds a row
here.

| Version | Commit | Date | Change |
|---------|--------|------|--------|
| 0.1.0 | — | 2026-09-28 | Initial build: NumPy pipeline ported from hurin, MLX forward model, profiled Legendre likelihood, template-sweep seeding, anvil driver, hurin-format products, kwargs CLI |
