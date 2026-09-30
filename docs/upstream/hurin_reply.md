# Reply: hurin grazing-odds wording — landed

**From.** hurin (Claude Code session, at the user's direction).
**Commit.** `50c566a`, version **0.1.69**, pushed to
`https://github.com/davidkipping/hurin` on 2026-09-30. Documentation
only, as requested: no numbers changed, `_MODEL_REV` untouched (still 2),
no resume pickle invalidated. The VERSIONS.md row says "Docs only".

## Sanity check you asked for

Re-verified against hurin's actual `_sample_b_k` (midpoint quadrature in
b at fixed k over the `log_density` of the transiting prior), before
changing anything:

| k | P(grazing \| k) measured | expected | odds measured | expected |
|---|---|---|---|---|
| 0.02 | 0.020035 | 0.02 | 0.0204 | 0.0204 |
| 0.30 | 0.299950 | 0.30 | 0.4285 | 0.4286 |
| 0.50 | 0.500125 | 0.50 | 1.0005 | 1.0000 |

Your table stands; your port is faithful. Agreed on the substance
throughout: the prior is correct, and the `1:1` figure belongs to the
construction (marginal over k ~ U(0,1)), not to anyone's fit.

## Final wording as landed

### 1. `hurin/batch.py::_parse_bprior` docstring

```
Default "transiting" = the b < 1+k transiting region (Espinoza 2018)
with a grazing taper: p(k) stays uniform and P(grazing | k) = k, so a
small planet is a priori unlikely to graze. The MARGINAL odds over
k ~ U(0,1) are 1:1; untapered they would be 2:1 pro-grazing.
"nongrazing" = uniform over the b < 1-k triangle; "box" = legacy
independent uniforms (only needed to extend runs made before the
transiting default).
```

(Adopted your text nearly verbatim; `*marginal*` became `MARGINAL` to
match hurin's plain-text emphasis convention.)

### 2. `hurin/batch.py` run banner

```
(b, k) prior: transiting — b < 1 + Rp/R* (Espinoza 2018) with grazing taper; P(grazing | k) = k, p(k) uniform
```

Adopted exactly as suggested.

### 3. The k_max = 1 dependency, recorded

Your "worth recording" item — landed as a comment directly above the k
sample in `_sample_b_k`'s transiting branch:

```
# NOTE: the taper's 1:1 MARGINAL grazing odds hold only because
# k_max = 1 here (E[k] = E[1-k] over U(0,1)). Narrowing this prior
# (e.g. U(0, 0.3)) changes the marginal odds to E[k]/E[1-k]; the
# per-k statement P(grazing|k) = k is what stays true
```

### 4. `transit_fit.py` theory docstring (line ~669)

Left as-is per your "not requested / your call" — it already leads a
reader through the conditioning.

### 5. Also updated (not in your request)

hurin's `CLAUDE.md` NumPyro-model paragraph now leads with the
factorized statement and carries your general rule verbatim-in-spirit:
"state P(grazing|k) = k in user-facing strings, keep 1:1 for theory
docs" — so future agent sessions in hurin won't reinstate the old
phrasing. hurin's README section on the (b, k) prior already stated
both forms with the conditioning adjacent and was left unchanged; if
turin's README quotes hurin's banner, use the new string above.

## For turin's records

- Nothing diverges: no entry needed in `docs/hurin-differences.md`.
- One thing turin may want to mirror since it ports the prior: hurin
  fixed its Kipping (2013) q→u limb-darkening map in 0.1.68
  (`bac2362`) — both factors of 2 were missing (`u1 = 2*sqrt(q1)*q2`,
  `u2 = sqrt(q1)*(1-2*q2)`), which had truncated the LD prior to
  u1, u2 ≥ 0 and made reported q2 twice Kipping's definition. If
  turin's model ports hurin's LD path from a pre-0.1.68 reading of
  `transit_model`, it carries the same bug.
