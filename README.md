# tulpaRatio

*a ratio with its uncertainty kept*

[![R-CMD-check](https://github.com/gcol33/tulpaRatio/actions/workflows/R-CMD-check.yml/badge.svg)](https://github.com/gcol33/tulpaRatio/actions/workflows/R-CMD-check.yml)
[![Codecov test coverage](https://codecov.io/gh/gcol33/tulpaRatio/graph/badge.svg)](https://app.codecov.io/gh/gcol33/tulpaRatio)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Bayesian hierarchical models for ratios, rates, and proportions, with inference on the numerator and denominator rather than their quotient.**

Give it the two counts, not the ratio. `tulpaRatio` fits the numerator and
denominator processes jointly, with shared random effects between them, and
derives the ratio posterior afterwards with uncertainty carried through from
both components. Sampling runs on a native HMC/NUTS backend built on the
[tulpa](https://github.com/gcol33/tulpa) engine, with no Stan or JAGS to
install. The usual approach divides first and models the quotient (or pins
the denominator into an `offset`); that discards the shared structure and
mis-states the uncertainty.

```r
library(tulpaRatio)

# catch-per-unit-effort: model catch and effort, derive CPUE
fit <- tratio(
  catch | effort ~ depth + season + (1 | site),
  data   = trawl_data,
  family = ratiod_poisson_gamma()
)

ratio(fit)                    # CPUE posteriors, uncertainty propagated
ratio_contrast(fit, ~ season) # compare seasons on the derived ratio
```

## Derived, not divided

Dividing two measured quantities inherits the error of both and creates
heteroscedasticity by construction: a site with more effort gives a more
precise CPUE, yet a model fit to the quotient treats every ratio as equally
informative. The offset fix assumes effort acts on catch with a coefficient of
exactly 1 and that effort carries no structure of its own. `tulpaRatio` keeps
both processes as what they are, models them together, and computes the ratio
from the joint posterior.

```r
# the offset approach assumes proportionality and a structureless denominator
glm(catch ~ depth + season + offset(log(effort)), family = poisson, data = df)

# tulpaRatio models both processes, derives the ratio post hoc
tratio(catch | effort ~ depth + season + (1 | site),
       data = df, family = ratiod_poisson_gamma())
```

See [Why Ratios Are Not Data](https://gillescolling.com/tulpaRatio/articles/philosophy.html)
for the full argument.

## Model families

| Family | Numerator | Denominator | Use case |
|--------|-----------|-------------|----------|
| `ratiod_negbin_negbin()` | NegBin | NegBin | Count/count ratios (relative abundance) |
| `ratiod_poisson_gamma()` | Poisson | Gamma | Count/continuous effort (CPUE) |
| `ratiod_binomial()` | Binomial | Known trials | Success/trial proportions |
| `ratiod_negbin_gamma()` | NegBin | Gamma | Overdispersed count/effort |
| `ratiod_beta_binomial()` | Beta-Binomial | Known trials | Overdispersed proportions |
| `ratiod_gamma_gamma()` | Gamma | Gamma | Continuous/continuous ratios |
| `ratiod_lognormal()` | Lognormal | Lognormal | Continuous ratios on the log scale |

Zero-inflation, one-inflation, hurdle, and zero-and-one-inflated variants are
available for the relevant families.

## Hierarchical, spatial, and temporal structure

- **Shared random effects** between numerator and denominator by default;
  `(1 + x | site)` correlated and `(1 + x || site)` uncorrelated slopes;
  nested `(1 | site/plot)` and crossed `(1 | site) + (1 | year)`.
- **Latent factors** (`latent_factor()`) for unmeasured confounders shared
  across the two processes.
- **Spatial**: `spatial_car()`, `spatial_bym2()`, and Gaussian-process priors
  for areal or point-referenced data.
- **Temporal**: `temporal_ar1()`, `temporal_rw1()`, `temporal_rw2()`.
- **Spatiotemporal**: `spatiotemporal()` for Knorr-Held Type I-IV interactions
  and `spatiotemporal_gp()` for non-separable space-time GPs.

## Inference backends and tiers

`mode = "auto"` picks an exact or structured backend for the model and never
silently drops to an approximation that lacks a convergence guarantee.

- **Tier 1 (exact):** HMC/NUTS (default), elliptical slice sampling,
  Pólya-Gamma Gibbs for binomial models.
- **Tier 2 (structured):** nested Laplace approximation for large datasets, integrating the hyperparameters on an outer grid.
- **Tier 3 (optimized):** variational inference and stochastic-gradient MCMC,
  available by explicit opt-in only.

LOO and WAIC are provided for model comparison. See
[inference_mode_info()] for the tier system.

## Process-specific predictors

The numerator and denominator can take different predictors:

```r
fit <- tratio(
  species_count | total_count ~ (1 | site),
  formula_num   = ~ depth + temperature,   # numerator process
  formula_denom = ~ sampling_effort,        # denominator process
  data   = df,
  family = ratiod_negbin_negbin()
)
```

## Installation

```r
install.packages("pak")                   # development version
pak::pak("gcol33/tulpaRatio")
```

## Documentation

- [Quick Start](https://gillescolling.com/tulpaRatio/articles/getting-started.html)
- [Why Ratios Are Not Data](https://gillescolling.com/tulpaRatio/articles/philosophy.html)
- [Complete Workflows](https://gillescolling.com/tulpaRatio/articles/workflows.html)
- [Spatial and Temporal Models](https://gillescolling.com/tulpaRatio/articles/spatial-temporal.html)
- [Random Effects](https://gillescolling.com/tulpaRatio/articles/random-effects.html)
- [Zero-Inflation](https://gillescolling.com/tulpaRatio/articles/zero-inflation.html)

## Support

> "Software is like sex: it's better when it's free." — Linus Torvalds

I'm a PhD student who builds R packages in my free time because I believe good tools
should be free and open. I started these projects for my own work and figured others
might find them useful too.

If this package saved you some time, buying me a coffee is a nice way to say thanks.
It helps with my coffee addiction.

[![Buy Me A Coffee](https://img.shields.io/badge/-Buy%20me%20a%20coffee-FFDD00?logo=buymeacoffee&logoColor=black)](https://buymeacoffee.com/gcol33)

## License

MIT (see the LICENSE.md file)

## Citation

```bibtex
@software{tulpaRatio,
  author = {Colling, Gilles},
  title  = {tulpaRatio: Bayesian Hierarchical Models for Ratios, Rates, and Proportions},
  year   = {2025},
  url    = {https://github.com/gcol33/tulpaRatio}
}
```
