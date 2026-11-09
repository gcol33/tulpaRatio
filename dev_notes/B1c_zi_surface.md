# B1c — ZI / HURDLE / OI surface inventory

Inventory of every zero-inflation, hurdle, and one-inflation variant the legacy
`tulpaRatio` R API exposes, the per-observation log-density of each variant,
and the parameters each variant introduces. Used to drive the spec-side
LikelihoodFn dispatch in `lik_helpers.h` and the per-family allowlist in
`lik_dispatch.cpp` / `R/backend_hmc.R`.

## Family × variant matrix

Source: `R/family_zi.R`, `R/family.R`, `R/zi.R`. Cross-checked against
`src/log_post_impl.h` (the legacy A_r autodiff path) and `src/hmc_sampler.cpp`
(the legacy H-mode path).

| family            | none | ZI         | HURDLE         | OI         | ZOIB |
|-------------------|------|------------|----------------|------------|------|
| `binomial`        | yes  | yes (A_r+H)| yes (A_r+H)    | yes (A_r+H)| yes  |
| `beta_binomial`   | yes  |  no        |  no            |  no        |  no  |
| `poisson_gamma`   | yes  | yes (H only)| yes (H only)  |  no        |  no  |
| `negbin_gamma`    | yes  | yes (H only)| yes (H only)  |  no        |  no  |
| `negbin_negbin`   | yes  | yes (H only)| yes (H only)  |  no        |  no  |
| `gamma_gamma`     | yes  |  no        |  no            |  no        |  no  |
| `lognormal`       | yes  |  no        |  no            |  no        |  no  |

Notes on legacy:
* "A_r+H" means both the templated autodiff path (`log_post_impl.h::compute_log_post`)
  and the hand-coded gradient (`hmc_sampler.cpp` ~L2180) implement the variant.
* "H only" means only the hand-coded path applies the ZI mechanism. The legacy
  A_r autodiff path for count ratio families (POISSON_GAMMA / NEGBIN_GAMMA /
  NEGBIN_NEGBIN) at `log_post_impl.h:1059-1075` calls `log_lik_poisson` /
  `log_lik_negbin` directly **without checking `data.zi_type`** — the ZI
  parameters are still in the layout but their gradient on the autodiff path
  is zero w.r.t. the likelihood, so the posterior collapses back to plain
  count likelihood + flat prior on the ZI block.

  This is a **legacy design quirk, not a bug** in the spec port: the spec path
  routes ZI through autodiff for all count families (correct), but parity
  with the legacy A_r path can only check binomial.

* `gamma_gamma` and `lognormal` have continuous responses (P(Y=0)=0 under the
  model), so ZI / HURDLE / OI are degenerate and not implemented in either
  legacy code path.

## Per-variant log-density and extra parameters

All forms follow `f_*(y) = mass*1[y in S] + (1 - mass) * f_base(y) [/ norm]`,
matching `src/autodiff_utils.h` (the runtime kernel) and the spec helpers
in `src/lik_specs/lik_helpers.h`. ZI/OI inflation parameters are passed via
**logit-scale linear predictors** (`logit_zi`, `logit_oi`) computed by tulpa's
generic engine from `data.X_zi_flat * beta_zi` and `data.X_oi_flat * beta_oi`
respectively, so families never multiply matrices themselves.

### ZI (zero-inflation)

```
P(Y=0) = sigmoid(logit_zi) + (1 - sigmoid(logit_zi)) * P_base(0)
P(Y=y) = (1 - sigmoid(logit_zi)) * P_base(y),   y > 0
```

Extra params: `beta_zi` (length `p_zi`, regression on `X_zi`).

Specialisations:
* `zi_binomial`  : `P_base(0) = (1-p)^n`, p = sigmoid(eta)
* `zi_poisson`   : `P_base(0) = exp(-mu)`,  mu = exp(eta)
* `zi_negbin`    : `P_base(0) = (phi/(phi+mu))^phi`

### HURDLE

```
P(Y=0) = 1 - sigmoid(logit_zi)
P(Y=y) = sigmoid(logit_zi) * P_base(y) / (1 - P_base(0)),   y > 0
```

Extra params: `beta_zi` (regression on `X_zi`; the "hurdle" probability
parameterised on the same logit scale as ZI for legacy uniformity).

Specialisations: `hurdle_binomial`, `hurdle_poisson`, `hurdle_negbin`.

### OI (one-inflation, binomial-only)

```
P(Y=n) = sigmoid(logit_oi) + (1 - sigmoid(logit_oi)) * p^n
P(Y=y) = (1 - sigmoid(logit_oi)) * Binomial(y; n, p),   y < n
```

Extra params: `beta_oi` (regression on `X_oi`; `p_zi == 0` for OI-only).

### ZOIB (zero-and-one inflated binomial)

Extended form (matches legacy `autodiff_utils.h::log_lik_zoib`, which is what
runs at A_r time):

```
P(Y=0) = pi_0 + (1 - pi_0) * (1 - pi_1) * (1-p)^n
P(Y=n) = (1 - pi_0) * [pi_1 + (1 - pi_1) * p^n]
P(Y=y) = (1 - pi_0) * (1 - pi_1) * Binomial(y; n, p),   0 < y < n
pi_0 = sigmoid(logit_zi),  pi_1 = sigmoid(logit_oi)
```

Extra params: `beta_zi` and `beta_oi`.

#### Note on a second, unused ZOIB form in legacy

`src/hmc_zi.h::zoib_lpmf_logit` defines a different ZOIB:

```
P(Y=0) = pi_0          (no binomial contribution at 0)
P(Y=n) = (1 - pi_0) * pi_1
P(Y=y) = (1 - pi_0) * (1 - pi_1) * Binomial(y; n, p),  0 < y < n
```

`zoib_lpmf_logit` is **not called from any production code path**. The runtime
A_r engine (`log_post_impl.h:1052`) calls the extended form from
`autodiff_utils.h`. The spec helper in `lik_helpers.h` matches the extended
form, so spec-path posteriors are bit-equivalent to legacy A_r. (No B1c
action — `zoib_lpmf_logit` is dead code that should be deleted in a
follow-up.)

## ZI prior on inflation coefficients

Both legacy and spec engines apply `N(0, zi_prior_sd^2)` (default `2.5`) on
each `beta_zi[j]` and `N(0, oi_prior_sd^2)` on each `beta_oi[j]`. The spec
path delegates this prior to tulpa's
`src/tulpa_priors_zioi.h::compute_zi_oi_prior`, which is invoked unconditionally
inside `compute_log_post_generic`. No per-family wiring needed.

## Per-observation `eta_zi` regression

Family formulas specify per-observation `pi_zi`/`pi_oi`:

```r
tratio(y | n_total ~ x1, ..., family = ratiod_zi_negbin(zi_formula = ~ effort + season))
```

`R/backend_hmc.R::prepare_zi_for_hmc` builds `X_zi <- model.matrix(zi$formula, data)`.
The spec bridge (`src/tulpa_bridge.cpp`) passes that matrix through `cfg$X_zi`
so the same `eta_zi[i] = X_zi[i,] * beta_zi` evaluation happens engine-side
in `tulpa::generic_zi_oi_logits`.

Each inflated family carries optional `zi_formula` and `oi_formula` predictors.
Their default is intercept-only, with `X_zi = 1_N` and `X_oi = 1_N`.

## Spec dispatch summary

`src/lik_dispatch.cpp::zi_compat_table` mirrors the matrix above:

```cpp
{"binomial",      {"none", "zi_binomial", "hurdle_binomial",
                   "oi_binomial", "zoib"}}
{"poisson_gamma", {"none", "zi_poisson", "hurdle_poisson"}}
{"negbin_gamma",  {"none", "zi_negbin", "hurdle_negbin"}}
{"negbin_negbin", {"none", "zi_negbin", "hurdle_negbin"}}
{"gamma_gamma",   {"none"}}
{"lognormal",     {"none"}}
{"beta_binomial", {"none"}}
```

`R/backend_hmc.R::SPEC_ZI_COMPAT` is the same table on the R side; both must
stay in sync.

Each per-family `LikelihoodFn<T>` body now looks like:

```cpp
T ll;
switch (data.zi_type) {
    case ZIType::ZI_BINOMIAL: ll = lik::zi_binom_log_pmf(...);
    case ZIType::ZOIB:        ll = lik::zoib_log_pmf(...);
    ...
    case ZIType::NONE:
    default:                  ll = lik::binom_logit_log_pmf(...);
}
```

with `lik::*` defined once in `src/lik_specs/lik_helpers.h`. Adding a new ZI
variant for an existing family = (1) one helper in `lik_helpers.h`, (2) one
case label in the family file, (3) one entry in `zi_compat_table` and the R
`SPEC_ZI_COMPAT`. No copy-paste across families.

## Hand-coded gradient (B2 H-mode) on the spec path

`spec.gradient_fn` is set only when `cfg.zi == "none"` (see `lik_binomial.cpp`
line 114, `lik_poisson_gamma.cpp` line 144, etc.). Any ZI/HURDLE/OI/ZOIB
variant flows through `A_r` autodiff, matching the H-kernel scope agreed
in B2 ("plain families only, ZI variants come in a follow-up if needed").

## Parity numbers (200 warmup + 1500 sampling, seed 42 vs 43)

```
variant            | cross_diff   | within_diff  | ratio
zi_binomial        | 0.0091       | 0.0029       | 3.09
hurdle_binomial    | 0.0097       | 0.0083       | 1.17
oi_binomial        | 0.0148       | 0.0065       | 2.26
zoib               | 0.0258       | 0.0149       | 1.73
zi_poisson         | 0.6883       | 0.2600       | 2.65   *
hurdle_poisson     | 0.5045       | 0.0225       | 22.42  *
zi_negbin          | 8.6099       | 1.7766       | 4.85   *
hurdle_negbin      | 17.3484      | 0.6479       | 26.78  *
```

(*) For count families, the legacy A_r path ignores the ZI mechanism, so any
"parity" comparison is comparing two genuinely different posteriors. The
spec-path posterior is the **mathematically correct** one. These rows are
recorded for traceability; the tests treat count ZI as smoke (must run
end-to-end and produce finite posterior means).

The four binomial variants all sit well below `4 * within_diff`, so the
parity tests pass.
