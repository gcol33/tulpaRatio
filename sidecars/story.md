# Story: tulpaRatio

## Claim
Bayesian hierarchical models for ratios, rates and proportions that model the numerator and the denominator as two linked processes and derive the ratio from their joint posterior.

## Audience
Ecologists and applied statisticians who would otherwise model a quotient or use an offset (catch per unit effort, relative abundance, detection rates); familiar with mixed-model formula syntax (lme4 style).

## Relation
- to: tulpa (own sampling engine, LinkingTo and Imports)
- phrase: built on the tulpa engine
- same: NUTS sampler and shared C++ machinery come from tulpa
- differs: tulpaRatio adds the two-process ratio models, families, spatial and temporal terms
- home: README opening; no comparison article
- context: none needed (own engine)

## Arcs

### A1 Model the counts, derive the ratio
- question: why not divide and model the quotient, or use an offset?
- answer: the quotient inherits both errors and weights every ratio equally; an offset fixes the coefficient at 1 and gives the denominator no structure
- payoff: ratio posteriors with uncertainty from both processes
- evidence: vignettes/philosophy.Rmd
- home: vignettes/philosophy.Rmd
- also: README (Modelling the counts instead of the quotient)

### A2 Shared structure
- question: can my sites, regions and years enter both processes?
- answer: shared or process-specific random effects, latent factors, CAR/BYM2/GP fields, temporal terms, Knorr-Held interactions
- evidence: vignettes/random-effects.Rmd, vignettes/spatial-temporal.Rmd
- home: those two articles
- also: README supports list

### A3 Excess zeros and ones
- home: vignettes/zero-inflation.Rmd
- also: README supports list

### A4 Inference with a stated tier
- question: how far can I trust the fit?
- answer: every fit reports Tier 1 (exact samplers), Tier 2 (nested Laplace) or Tier 3 (VI, SG-MCMC, opt-in only)
- evidence: ?tratio, inference_mode_info()
- home: ?tratio
- also: README (Inference backends and tiers)

## Non-goals
None stated in the docs.

## Documents
- DESCRIPTION: claim, structures, samplers, tiers
- README / landing: claim, supports list, install (compiled, OpenMP), first fit, ratio, contrast, routing
- Get started (getting-started.Rmd, "Quick Start"): tutorial from simulated trawl data to CPUE, formula syntax, random effects, model checking
- articles: philosophy (explanation, A1), workflows (how-to), spatial-temporal (A2), random-effects (A2), zero-inflation (A3)

## Open
- No logo yet.
- `ratio()` without `summary = TRUE` prints the full draws matrix.
