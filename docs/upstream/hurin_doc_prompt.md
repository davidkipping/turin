# Task: reword hurin's "1:1 grazing odds" in the two user-facing strings

**Who is asking.** `turin`
(`/Users/dkipping/Storage1/Work/Documents/Transit_Work/CODES/turin`) is the
GPU successor to hurin, and it ports hurin's `(b, k)` prior unchanged —
`turin/model.py:impact_parameter` is the coordinate map and
`turin/params.py:bk_log_prior` the weights, both written from
`hurin/transit_fit.py:_sample_b_k`. While documenting the three `--bprior`
modes in turin's README I verified the prior numerically and then went
looking for how hurin describes it.

**This is a documentation-only request. hurin's prior is correct.** No
numbers change, nothing needs a `_MODEL_REV` bump, and no existing resume
pickle is invalidated. I am not asking for a code change beyond two strings.

**The finding, stated fairly.** `hurin/transit_fit.py:669` is already the
clearest of the three descriptions and needs nothing — it says
`Factorized: p(k) is uniform on (0,1) and P(grazing | k) = k`, which is
exactly the right statement, and its "untapered gives 2:1 pro-grazing"
aside is correct too (I measured 1.9985). The problem is confined to the
two strings a *user* reads, where the `1:1` figure appears without the
conditioning that makes it true.

## What I verified

Midpoint quadrature in `beta` at fixed `k`, against
`turin.params.bk_log_prior` (a faithful port of hurin's construction; I did
not run hurin itself, so please sanity-check one value against
`_sample_b_k` before acting):

| k | grazing prior mass | total mass | odds grazing:non-grazing |
|---|---|---|---|
| 0.02 | 0.020000 | 1.00000000 | 0.0204 |
| 0.05 | 0.050000 | 1.00000000 | 0.0526 |
| 0.10 | 0.100000 | 1.00000002 | 0.1111 |
| 0.30 | 0.300000 | 1.00000004 | 0.4286 |
| 0.50 | 0.500000 | 1.00000000 | 1.0000 |

So `P(grazing | k) = k` exactly, the total is 1 at every `k` (equivalently,
`p(k)` is exactly uniform — that is what the taper buys; untapered `p(k)`
would go as `1 + k`), and the odds at fixed `k` are `k/(1-k)`. Integrated
over `k ~ U(0, 1)`, `k` and `1-k` each give 1/2, so the **marginal** odds
are 1:1 — measured 1.0003 over 4e6 samples. Every claim in hurin's
`transit_fit.py` docstring checks out.

**Why it still misleads.** `1:1` is a property of the prior construction
integrated over the whole `k` range. It is not a property of anyone's fit.
A user fitting a 500 ppm transit has `k ~ 0.02`, where the grazing prior
mass is 2%, not 50% — the `1:1` figure overstates it by 25x for that
target. I briefly talked myself into believing hurin had a documentation
*error* here on the strength of one sloppy measurement (I had drawn
`k ~ U(0, 0.3)` and read off 0.176, which is just `E[k]/E[1-k]`); it does
not. But if the phrasing can mislead someone who is actively reading the
source, it will mislead someone reading a run log.

## The two changes

### 1. `hurin/batch.py:160`, `_parse_bprior`'s docstring — the one that matters

This is the worst of the three, because it attaches the ratio to *the
planet*:

    Default "transiting" = the b < 1+k transiting region (Espinoza 2018)
    with a grazing taper giving exactly 1:1 marginal prior odds that the
    planet is grazing; ...

"1:1 marginal prior odds **that the planet is grazing**" reads as "this
target has even odds of grazing". The word `marginal` is carrying the
entire qualification and will not survive a skim. Suggested:

    Default "transiting" = the b < 1+k transiting region (Espinoza 2018)
    with a grazing taper: p(k) stays uniform and P(grazing | k) = k, so a
    small planet is a priori unlikely to graze. The *marginal* odds over
    k ~ U(0,1) are 1:1; untapered they would be 2:1 pro-grazing.
    "nongrazing" = uniform over the b < 1-k triangle; "box" = legacy
    independent uniforms (only needed to extend runs made before the
    transiting default).

### 2. `hurin/batch.py:1216`, the line printed on every run

    "transiting": " — b < 1 + Rp/R* (Espinoza 2018) with grazing"
                  " taper (1:1 grazing odds)",

Nothing here is false, but this is the string users actually read, and
`1:1` is the least useful number to hand them — nobody can apply it to
their own target. Suggested:

    "transiting": " — b < 1 + Rp/R* (Espinoza 2018) with grazing"
                  " taper; P(grazing | k) = k, p(k) uniform",

That version is actionable: a user knows roughly what `k` their target has
and can read off their own grazing prior weight.

### Not requested: `hurin/transit_fit.py:669`

Correct and complete as written. If you want it to read better for a
skimmer, moving the existing `Factorized: p(k) is uniform on (0,1) and
P(grazing | k) = k` sentence *above* the `EXACTLY 1:1` one would put the
per-`k` statement first. Purely cosmetic, and your call.

## The general rule I would suggest adopting

State `P(grazing | k) = k` in user-facing strings; keep `1:1` for the
theory docstring, where its conditioning is visible in the surrounding
text. The first describes the reader's fit, the second describes the prior
construction, and users are reading about their fit.

## One thing worth recording so it is not re-introduced

The `1:1` claim is true **only because `k_max = 1`**. It is the fact that
`k` and `1-k` both integrate to 1/2 over `k ~ U(0, 1)` that makes the
marginal odds equal. If hurin ever narrows the `k` prior to a realistic
range — say `U(0, 0.3)`, which would be a defensible tightening for most
Kepler targets — the marginal odds become `E[k]/E[1-k] = 0.18` and the
`1:1` statement stops being true at all, rather than merely being easy to
misread. A one-line comment beside the `k` prior bound noting that
dependency would stop someone reinstating the phrase later.

## Deliverable discipline

hurin's conventions, as I understand them from the repo: one patch version
bump per commit (`pyproject.toml`, `hurin/__init__.py`, and a `VERSIONS.md`
row), and push to `https://github.com/davidkipping/hurin`. Since this is
documentation only, please say so in the VERSIONS row and leave
`_MODEL_REV` alone — changing it would refuse every existing resume pickle
for no reason.

When you are done, please reply in
`turin/docs/upstream/hurin_reply.md` with what landed and the final
wording, so turin's README (which now documents both readings of the odds,
under "The (b, k) prior") can be kept consistent with hurin's.

If you disagree that this is worth changing, that is a legitimate answer —
say so and why, and turin will simply note the divergence in
`docs/hurin-differences.md`.
