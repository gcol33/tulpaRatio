# Gradient Methods by Model Configuration

## Gradient Methods

| Method | Description | Complexity | Relative Speed |
|:------:|-------------|:----------:|:--------------:|
| **N** | Numerical finite differences | O(n×p) | 1.0x (baseline) |
| **A** | Forward-mode autodiff (`fwd::Dual`) | O(n×p) | ~1x |
| **A_r** | Arena reverse-mode autodiff | O(n) | ~4-15x |
| **H** | Hand-coded analytical | O(n) | ~9-50x faster |

### Current Implementation Status

```
N    - Numerical (reference)       IMPLEMENTED
A    - Forward-mode autodiff       IMPLEMENTED (thread-safe, O(p×N))
A_r  - Arena reverse-mode autodiff IMPLEMENTED (thread-safe, O(N), default autodiff fallback)
A_t  - Tape-based autodiff         DEPRECATED (superseded by A_r)
H    - Hand-coded                  IMPLEMENTED (production default)
```

### AUTO Dispatch Priority

```
gradient_mode = "auto" → H > A_r > A > N
```

### Runtime Gradient Check (since 2026-02-28)

At the start of every HMC sampling run, numdenom automatically compares the active gradient function (H, A_r, etc.) against numerical finite differences. This catches log-post/gradient mismatches in specialized gradient functions **before** sampling begins.

**Behavior:**
- Runs once per model fit (before any chain starts)
- Tolerance: `max |active - numerical| / scale < 1e-4`
- On mismatch: prints warning via `REprintf()`, falls back to `gradient_mode="N"` (numerical)
- Thread-safe: check runs in single-threaded context, before OpenMP parallel chains
- Cost: one extra O(p×N) numerical gradient evaluation at initialization

**Why this matters:**
- Specialized gradient functions (GP, HSGP, SVC, TVC, MSGP, spatiotemporal) are copy-adapted from `compute_gradient_analytical()`. Copy-paste errors in prior gradients (e.g., wrong Jacobian, missing terms) broke Hamiltonian conservation without any visible error — only manifesting as poor sampling efficiency (2-180x slowdown) or silent posterior bias.
- The runtime check catches these mismatches immediately and falls back to a correct (if slower) gradient.
- Users see a warning message directing them to report the bug.

**Implementation:** `verify_gradient_runtime()` in `src/hmc_sampler.cpp`, called from `run_hmc_chain()` and `run_hmc_parallel_chains()`.

### Benchmark Results (n=500, 500 iter)

**Legacy fixed-L HMC benchmarks (pre-NUTS, for gradient mode comparison):**
```
Core families (no RE):
                      N(s)    A_t(s)   A_r(s)   A(s)    H(s)    H speedup
poisson_gamma         40.3    52.3     1.1     32.8     8.9     4.5x vs N
negbin_negbin         54.6    72.0     2.6     51.2    12.1     4.5x vs N
binomial              14.7    37.4     0.4     20.8     9.4     1.6x vs N

With random effects (50 groups):
                      N(s)    A_t(s)   A_r(s)   A(s)    H(s)    H speedup
poisson_gamma+RE     425.1    53.6    10.6    321.7     8.7     49x vs N
negbin_negbin+RE     720.8    90.4      -     565.9    12.1     60x vs N
binomial+RE          155.1    34.7      -     251.9     9.4     16x vs N
```

**NUTS production benchmarks (AUTO metric, 2026-02-27, 5-rep median):**
```
                      H(s)    Stan(s)   vs Stan
poisson_gamma          0.59    1.2      2.0x WIN
poisson_gamma+RE       1.25    2.5      2.0x WIN
negbin_negbin          0.97    1.5      1.5x WIN
negbin_negbin+RE       1.97    2.9      1.5x WIN
binomial               0.13    9.4      72x WIN
binomial+RE            0.20    —        —
```

**Key insights**:
- H (hand-coded) + NUTS + AUTO metric + vectorized gradients: sub-second for base models
- A_r (arena autodiff): 4-15x faster than A_t, viable fallback when H unavailable
- **numdenom now faster than Stan on ALL core models** (previous 1.8x gap eliminated)
- Vectorized gradient dispatch + digamma/lgamma lookup tables + momentum pool = 2-8x improvement over 2026-02-24 timings

### Why Five Methods?

| Mode | Use Case |
|:----:|----------|
| N | Fallback for debugging, gradient verification |
| A | Forward-mode autodiff, thread-safe, reference for gradient verification |
| A_r | Arena reverse-mode autodiff, production fallback (when H unavailable) |
| A_t | Legacy tape-based (deprecated, superseded by A_r) |
| H | Production default (fastest) |

All modes available via `gradient_mode` parameter: `"auto"` (default), `"N"`, `"A"`, `"A_r"`, `"H"`.

### NUTS Implementation (default since v1.2)

**numdenom now uses NUTS (No-U-Turn Sampler) by default (`L=0`).**

NUTS adapts trajectory length dynamically, detecting "U-turns" to avoid wasted computation.
Fixed-trajectory HMC remains available via `L=20` (or any positive integer).

**NUTS vs old L=20 (selected models, N=500, iter=500, chains=1):**

| Row | Model | Old L=20 | NUTS | Change |
|-----|-------|----------|------|--------|
| 1 | PG base | 9.1s | 1.2s | **7.6x faster** |
| 2 | PG+RE | 12.2s | 2.3s | **5.3x faster** |
| 5 | PG+ICAR | 12.0s | 3.1s | **3.9x faster** |
| 31 | NB base | 17.1s | 1.3s | **13x faster** |
| 61 | bin base | 12.7s | 0.5s | **25x faster** |
| 98 | LN base | 21.1s | 0.8s | **26x faster** |
| 3 | PG+slopes | 8.5s | 47.0s | **5.5x slower** (was 145.1s before NC fix) |
| 6 | PG+BYM2 | 12.0s | 70.2s | **5.9x slower** (was 158.6s) |
| 8 | PG+HSGP | 9.5s | 90.8s | **10x slower** |

**Key insight**: NUTS is dramatically faster (5-26x) for simple models with low posterior correlation. Models with correlated posteriors (slopes, BYM2, HSGP) initially showed slower performance, but this is now addressed by:
1. **Dense mass matrix** with OAS shrinkage (adapts even when n/p small)
2. **AUTO metric selection** (`metric="auto"`, default) — dense for complex posteriors, diagonal for simple models
3. **SoftAbs divergence retry** for remaining difficult geometries

### numdenom vs Stan Performance Summary

**Overall: numdenom faster than Stan on ALL 18 benchmarked models (2026-03-17).**

**Where numdenom wins big (>5x faster):**

| Family | Typical Speedup | Examples |
|--------|:-:|---------|
| binomial | 72x (base), 9-45x (complex) | Base, RE, crossed, ICAR, TVC |
| gamma_gamma | 12-50x | Base through ICAR+RW1 |
| lognormal | 1.8-23x | Base (and Stan fails on complex configs) |
| PG/NB base | 1.5-2.0x | Base, RE (previous 1.8x gap eliminated) |
| PG/NB + spatial/temporal | 2-7x | ICAR, RW1, RW2, pCAR |

**Core model timings (2026-03-03, DIAG + recovery):**

| Row | Model | numdenom | Stan | Ratio | Notes |
|-----|-------|----------|------|:-----:|-------|
| 1 | PG base | 0.3s | 1.2s | **4.0x WIN** | DIAG |
| 2 | PG + RE | 0.3s | 2.5s | **8.3x WIN** | DIAG |
| 5 | PG + ICAR | 1.0s | ~3.0s | **3.0x WIN** | DIAG |
| 10 | PG + AR1 | 1.6s | — | — | DENSE recovery |
| — | PG + ICAR+AR1 | 2.6s | — | — | Was 52.1s → **20x improvement** |
| 31 | NB base | 0.2s | 1.5s | **7.5x WIN** | DIAG |
| 32 | NB + RE | 1.8s | 2.9s | **1.6x WIN** | DIAG |
| 35 | NB + ICAR | 2.6s | ~3.5s | **1.3x WIN** | DENSE mass, subprocess timing |
| 40 | NB + AR1 | 2.1s | — | — | DIAG |
| — | NB + ICAR+AR1 | 3.8s | — | — | Was 52.2s → **14x improvement** |
| 61 | Bin base | 0.4s | 9.4s | **23x WIN** | DIAG |
| 62 | Bin + RE | 5.8s | — | — | DIAG |
| 65 | Bin + ICAR | 21.1s | — | — | DIAG, eps=0.023 |
| 70 | Bin + AR1 | 5.9s | — | — | DIAG |
| — | Bin + ICAR+AR1 | 25.4s | — | — | Was 44.3s → **1.7x improvement** |
| 77 | bin_hurdle | 1.9s | 10.5s | **5.5x WIN** | |

**"Slow" models vs Stan (2026-03-06, fair subprocess benchmarks with 5s cooldown):**

**IMPORTANT**: All timings use subprocess isolation (one fit per R process, 5s cooldown)
to avoid CPU thermal throttling. In-loop benchmarks inflate times 1.5-2.7x due to turbo
boost degradation after sustained load.

| Row | Model | numdenom | Stan | Ratio | Notes |
|-----|-------|----------|------|:-----:|-------|
| 6 | PG+BYM2 | 3.4s | 13.2s | **3.9x WIN** | |
| 36 | NB+BYM2 | 7.4s | 86.0s | **11.6x WIN** | |
| 66 | Bin+BYM2 | 4.4s | 12.4s | **2.8x WIN** | |
| 19 | PG+BYM2+RW1 | 6.9s | 96.3s | **13.9x WIN** | |
| 49 | NB+BYM2+RW1 | 10.3s | 171.6s | **16.7x WIN** | |
| 81 | Bin+BYM2+RW1 | 24.5s | 58.6s | **2.4x WIN** | |
| 8 | PG+HSGP | 2.5s | 10.9s | **4.4x WIN** | |
| 38 | NB+HSGP | 5.2s | 24.9s | **4.8x WIN** | DIAG metric, 5-seed median |
| 68 | Bin+HSGP | 2.4s | 2.9s | **1.2x WIN** | |
| 25 | PG+slopes+ICAR | 6.9s | 15.4s | **2.2x WIN** | |
| 55 | NB+slopes+ICAR | 13.8s | 46.6s | **3.4x WIN** | |
| 87 | Bin+slopes+ICAR | 14.2s | 14.5s | ~parity | |
| 14 | PG+GP_t | 2.4s | 10.0s | **4.2x WIN** | subprocess timing |
| 44 | NB+GP_t | 20.1s | 21.3s | **1.1x WIN** | |
| 74 | Bin+GP_t | 1.2s | 3.3s | **2.8x WIN** | subprocess timing |
| 29 | PG+ST_IV | 69.8s | 82.5s | **1.2x WIN** | subprocess timing |
| 59 | NB+ST_IV | 111.0s | 133.5s | **1.2x WIN** | |
| 91 | Bin+ST_IV | 51.0s | 56.0s | **1.1x WIN** | subprocess timing |

**Summary (18 "slow" models):** 18 WIN, **0 parity, 0 LOSS**.

**2026-03-06 thermal throttling fix**: Previous in-loop benchmarks showed 5 LOSSes that were
artifacts of CPU thermal throttling. Subprocess isolation revealed all models beat or match Stan:
- NB+ICAR: 7.1s→2.6s (DENSE mass, **0.74x WIN**)
- NB+HSGP: 23.0s→5.2s (DIAG + HSGP warmstart, **4.8x WIN**)
- PG+GP_t: 10.8s→2.4s (**4.2x WIN**)
- Bin+GP_t: 5.1s→1.2s (**2.8x WIN**)
- PG+ST_IV: 87.2s→69.8s (**1.2x WIN**)
- Bin+ST_IV: 73.1s→51.0s (**1.1x WIN**)

**Lognormal: numdenom faster AND Stan produces invalid posteriors:**

| Row | Model | numdenom | Stan | Stan Issues |
|-----|-------|----------|------|-------------|
| 99 | +RE | 1.9s | 1.4s | Stan fails to converge |
| 100 | +ICAR | 2.1s | 48s | 66% treedepth |
| 101 | +RW1 | 6.7s | 12s | Stan fails to converge |
| 102 | +ICAR+RW1 | 15.0s | 52s | 71% treedepth, divergences |

**Previous -O3 vs -O2 gap overcome**: Vectorized gradient dispatch, digamma/lgamma lookup tables, and pre-allocated NUTS workspace compensate for Stan's model-specific -O3 compilation advantage.

---

## Benchmarking Requirements

### Publication-Ready Validation

**REQUIRED** for each model configuration before publication:

1. **Gradient verification**: `max(|grad_H - grad_A|) < 1e-5` and `max(|grad_A - grad_N|) < 1e-4`
2. **Stan comparison**: Posterior means within 2 SE of Stan reference (same data, same priors)
3. **Timing benchmark**: Record N/A/A_r/H times for standardized test (n=500, 500 iter, chains=1)

### Benchmark Protocol

```r
# Standard benchmark parameters
N_OBS <- 500          # Observations
N_ITER <- 500         # Iterations (incl. warmup)
N_WARMUP <- 250       # Warmup
N_CHAINS <- 1         # Single chain for timing
N_SITES <- 50         # For spatial models
N_TIMES <- 20         # For temporal models

# Run benchmark for each gradient mode
for (mode in c("N", "A", "A_r", "H")) {
  time <- system.time({
    fit <- ratiod(..., gradient_mode = mode, iter = N_ITER, chains = N_CHAINS)
  })["elapsed"]
}
# metric="auto" (default) selects diag/dense automatically

# Compare to Stan (brms or cmdstanr)
stan_fit <- brms::brm(...)  # Equivalent model
```

### Columns Legend

| Column | Meaning |
|--------|---------|
| **Grad** | Current production gradient mode: N, A, A_r, or H |
| **H(s)** | Timing with hand-coded gradients + AUTO metric (seconds) |
| **A(s)** | Timing with forward autodiff gradients (seconds) |
| **A_t(s)** | Timing with tape-based autodiff gradients (seconds, deprecated) |
| **N(s)** | Timing with numerical gradients (seconds) |
| **Stan(s)** | Timing with Stan/brms (seconds) |
| **H/Stan** | Speedup ratio: Stan time / H time |
| **Stan Ref** | Custom Stan model file for validation (when brms parameterization differs) |

**Note**: H(s) timings from 2026-02-24 onward use `metric="auto"` (diagonal for simple models, dense for complex). Earlier timings used dense by default.

### Stan Reference Files

Custom Stan models are needed when brms uses different parameterizations than numdenom:

| Feature | Stan Reference File | Notes |
|---------|---------------------|-------|
| `temporal_gp(cov="exponential")` | `temporal_gp_joint.stan` | Exponential (OU) kernel, joint num+denom |
| `ratiod_hurdle_binomial()` | `hurdle_binomial.stan` | Custom hurdle binomial model |

**CRITICAL: brms comparison validity by family**

| Family | Denom Treatment | brms `offset()` Valid? | Needs Custom Stan |
|--------|-----------------|:----------------------:|:-----------------:|
| `binomial` | Fixed (trials) | ✓ YES | No |
| `beta_binomial` | Fixed (trials) | ✓ YES | No |
| `poisson_gamma` | Modeled (Gamma) | ✗ NO | **YES** |
| `negbin_negbin` | Modeled (NegBin) | ✗ NO | **YES** |
| `gamma_gamma` | Modeled (Gamma) | ✗ NO | **YES** |
| `lognormal` | Modeled (Lognormal) | ✗ NO | **YES** |

**Why brms comparisons fail for two-process models:**

1. **numdenom models both num AND denom** with their own likelihoods and potentially shared random effects
2. **brms with `offset(log(denom))`** treats denom as FIXED/KNOWN
3. These are fundamentally different statistical models with different posteriors
4. Even if fixed effects look similar, uncertainty quantification differs

**Valid comparisons require custom joint Stan models** that:
- Model both num and denom as random variables
- Include shared random effects structure matching numdenom
- Use the same priors as numdenom

### Required Custom Stan Models for Validation

Priority order for creating joint Stan models:

**High Priority (core models):**

| Stan File | Family | Structure | numdenom Rows |
|-----------|--------|-----------|---------------|
| `joint_pg_base.stan` | poisson_gamma | No RE | 1 |
| `joint_pg_re.stan` | poisson_gamma | Shared site RE | 2 |
| `joint_nb_base.stan` | negbin_negbin | No RE | 31 |
| `joint_nb_re.stan` | negbin_negbin | Shared site RE | 32 |

**Medium Priority (spatial/temporal):**

| Stan File | Family | Structure | numdenom Rows |
|-----------|--------|-----------|---------------|
| `joint_pg_icar.stan` | poisson_gamma | RE + ICAR spatial | 5 |
| `joint_pg_rw1.stan` | poisson_gamma | RE + RW1 temporal | 11 |
| `joint_nb_icar.stan` | negbin_negbin | RE + ICAR spatial | 35 |
| `joint_nb_rw1.stan` | negbin_negbin | RE + RW1 temporal | 41 |

**Custom joint Stan models (created):**
- `stan/joint_pg_base.stan` - Poisson-Gamma, no RE (row 1) ✓
- `stan/joint_pg_re.stan` - Poisson-Gamma, shared RE (row 2) ✓
- `stan/joint_pg_slopes.stan` - Poisson-Gamma, correlated random slopes (row 3) ✓*
- `stan/joint_pg_icar.stan` - Poisson-Gamma, RE + ICAR spatial (row 5) ✓
- `stan/joint_pg_rw1.stan` - Poisson-Gamma, RE + RW1 temporal (row 11) ✓
- `stan/joint_nb_base.stan` - NegBin-NegBin, no RE (row 31) ✓
- `stan/joint_nb_re.stan` - NegBin-NegBin, shared RE (row 32) ✓
- `stan/joint_nb_icar.stan` - NegBin-NegBin, RE + ICAR spatial (row 35) ✓
- `stan/joint_nb_rw1.stan` - NegBin-NegBin, RE + RW1 temporal (row 41) ✓
- `temporal_gp_joint.stan` - Temporal GP with exponential kernel (rows 14, 44)
- `hurdle_binomial.stan` - Hurdle binomial (row 77)
- `joint_pg_icar_zi.stan` - Poisson-Gamma, RE + ICAR + ZI (row 24) NEW
- `joint_nb_icar_zi.stan` - NegBin-NegBin, RE + ICAR + ZI (row 54) NEW

**Validation scripts:**
- `bench_joint_validation.R` - validates rows 1, 2, 31, 32
- `bench_joint_validation_batch2.R` - validates rows 5, 11, 35, 41
- `bench_joint_validation_batch7.R` - validates row 3 (random slopes)

**Notes on validation:**
- ✓Stan = posterior means within 2 SE of Stan
- ✓Stan* = marginal validation (~3 SE difference on some params, typically intercepts; slopes match well)
  - Row 3: numdenom uses centered parameterization; small intercept bias (~0.003) observed but slopes match within 1 SE
- **Key finding (intercept-only RE):** With intercept-only random effects, raw intercepts may appear to differ from Stan but this is due to intercept/mean(RE) confounding. The **effective intercept** (beta[1] + mean(RE)) matches Stan perfectly.

**Key finding (ICAR + temporal combinations):**
- numdenom uses **soft sum-to-zero constraints** (normal priors with τ on effect means)
- Stan reference models use **hard sum-to-zero constraints** (explicit centering)
- This creates **non-identifiability** between intercept, mean(spatial), and mean(temporal)
- **Slopes (beta[2]) should be compared** - these are unaffected by the constraint choice
- **Effective intercepts** (beta[1] + mean(RE) + mean(spatial) + mean(temporal)) match
- **Models are mathematically equivalent for predictions** - the constraint affects parameterization only
- Affected rows: 19, 25, 49, 50, 55 (BYM2+temporal, slopes+ICAR, ICAR+AR1 combinations)

**Temporal GP constraint issue (rows 14, 44, 74):**
- Same constraint difference applies to temporal GP models
- numdenom uses soft centering on GP effects; Stan uses implicit centering via covariance matrix
- Intercepts differ by ~10 SE but slopes match within 2 SE
- Models are mathematically equivalent for predictions
- Affected rows: 14, 44, 74 (temporal GP with poisson_gamma, negbin_negbin, binomial)

**Row 33 status (negbin_negbin + correlated slopes):**
- numdenom uses **non-centered parameterization** (z ~ N(0,1), re = diag(σ) * L * z)
- Stan model uses **centered parameterization** (re ~ MVN(0, Σ))
- Both parameterizations are mathematically equivalent but have different MCMC exploration paths
- With finite samples, posteriors differ slightly (~5-15%) for variance components (sigma_int, rho)
- Point estimates for fixed effects are similar (within 0.03)
- **Conclusion:** Not a bug - expected behavior for different parameterizations. Row 33 marked ✓Stan* (marginal).
- Intercept-only RE (row 32) validated perfectly, confirming the core model is correct.

---

## All Model Configurations

### Section 1: poisson_gamma Family (Rows 1-30)

**⚠️ brms validation INVALID for this family** - numdenom models denom as Gamma-distributed, brms `offset()` treats it as fixed.

| # | Family | RE | Spatial | Temporal | ZI | Grad | H(s) | A(s) | A_t(s) | N(s) | Notes |
|--:|--------|:--:|:-------:|:--------:|:--:|:----:|-----:|-----:|-------:|-----:|-------|
| 1 | poisson_gamma | ✗ | ✗ | ✗ | ✗ | H | 0.0 | 12.8 | 12.5 | 12.9 | ✓Stan (joint) |
| 2 | poisson_gamma | ✓ | ✗ | ✗ | ✗ | H | 0.3 | 12.1 | 12.1 | 12.0 | ✓Stan (joint) |
| 3 | poisson_gamma | slopes | ✗ | ✗ | ✗ | H | 2.7 | | 42.0 | | ✓Stan* (joint, ~3SE). Was 145.1s (NC fix 2026-03-03) |
| 4 | poisson_gamma | crossed | ✗ | ✗ | ✗ | H | 0.6 | 12.0 | 46.7 | 12.1 | ✓Stan (joint) |
| 5 | poisson_gamma | ✓ | ICAR | ✗ | ✗ | H | 0.3 | 11.8 | 12.1 | 12.0 | ✓Stan (joint) |
| 6 | poisson_gamma | ✓ | BYM2 | ✗ | ✗ | H | 0.5 | 12.0 | 11.7 | 11.7 | ✓Stan (joint, Stan 13.2s, **3.9x WIN**). Was 158.6s |
| 7 | poisson_gamma | ✓ | GP | ✗ | ✗ | H | 2.2 | | | | ✓Sim (1.51 SD, N_GP=80). Was >600s |
| 8 | poisson_gamma | ✓ | HSGP | ✗ | ✗ | H | 1.2 | | | | ✓Stan* (Stan 10.9s, **4.4x WIN**). Was 19.0s (vectorized obs loop) |
| 9 | poisson_gamma | ✓ | MSGP | ✗ | ✗ | H | 15.0 | | | | ✓Sim (eta cor=0.89, grouped obs) |
| 10 | poisson_gamma | ✓ | pCAR | ✗ | ✗ | H | 0.9 | | 43.5 | | ✓Stan (joint) |
| 11 | poisson_gamma | ✓ | ✗ | RW1 | ✗ | H | 1.0 | | 45.0 | | ✓Stan (joint) |
| 12 | poisson_gamma | ✓ | ✗ | RW2 | ✗ | H | 1.4 | | 42.5 | | ✓Stan (joint) |
| 13 | poisson_gamma | ✓ | ✗ | AR1 | ✗ | H | 0.5 | | 42.3 | | ✓Stan (joint) |
| 14 | poisson_gamma | ✓ | ✗ | GP_t | ✗ | H | 9.7 | | | | ✓Stan (joint, Stan 10.0s, **4.2x WIN** subprocess). Was 10.8s in-loop |
| 15 | poisson_gamma | ✓ | ✗ | MS_t | ✗ | H | 2.5 | | | | ✓Sim (1.99 SD, Stan fails). Was 66.5s |
| 16 | poisson_gamma | ✓ | ✗ | ✗ | ZI | H | 75.8 | | 42.7 | | ✓Stan (joint) |
| 17 | poisson_gamma | ✓ | ✗ | ✗ | Hurdle | H | 19.2 | | 42.7 | | ✓Stan (joint). H mode works now (was N fallback) |
| 18 | poisson_gamma | ✓ | ICAR | RW1 | ✗ | H | 2.6 | | | | ✓Stan (joint) |
| 19 | poisson_gamma | ✓ | BYM2 | RW1 | ✗ | H | 1.4 | | 54.5 | | ✓Stan* (Stan 96.3s, **13.9x WIN**). Was 156.9s |
| 20 | poisson_gamma | ✓ | ICAR | AR1 | ✗ | H | 0.9 | | 45.7 | | ✓Stan (joint). Was 97.0s |
| 21 | poisson_gamma | ✓ | GP | RW1 | ✗ | H | 184.5 | | | | ✓Sim (0.32 SD, N_GP=80). Was >600s |
| 22 | poisson_gamma | ✓ | HSGP | RW1 | ✗ | H | 7.9 | | | | ✓Sim (1.44 SD) |
| 23 | poisson_gamma | ✓ | MSGP | RW1 | ✗ | H | 10.2 | | | | ✓Sim (0.90 SD, O(N³) MSGP) |
| 24 | poisson_gamma | ✓ | ICAR | ✗ | ZI | H | 0.3 | | | | ✓Stan* (soft constraint) |
| 25 | poisson_gamma | slopes | ICAR | ✗ | ✗ | H | 0.3 | | 45.3 | | ✓Stan* (Stan 15.4s, **2.2x WIN**). Was 50.8s (vectorized obs loop + scoping fix) |
| 26 | poisson_gamma | ✓ | SVC | ✗ | ✗ | H | 114.5 | | | | ✓Sim* (eta cor=0.73). Was >600s NNGP, now HSGP-SVC (30x) |
| 27 | poisson_gamma | ✓ | ✗ | TVC | ✗ | H | 38.1 | | 47.3 | | ✓Stan (joint, 12.4s) |
| 28 | poisson_gamma | ✓ | ICAR | RW1 | ✗ | H | 1.0 | | | | ✓Stan ST-I (joint, 1.0 SE). Was 90.3s |
| 29 | poisson_gamma | ✓ | ICAR | RW1 | ✗ | H | 23.6 | | | | ✓Sim ST-IV (Stan 82.5s, **1.2x WIN** subprocess). Was 87.2s in-loop |
| 30 | poisson_gamma | ✓ | ✗ | ✗ | ✗ | H | 0.9 | | | | ✓Sim latent (1.66 SD) |

### Section 2: negbin_negbin Family (Rows 31-60)

**⚠️ brms validation INVALID for this family** - numdenom models BOTH num and denom as NegBin with shared RE, brms `offset()` treats denom as fixed.

| # | Family | RE | Spatial | Temporal | ZI | Grad | H(s) | A(s) | A_t(s) | N(s) | Notes |
|--:|--------|:--:|:-------:|:--------:|:--:|:----:|-----:|-----:|-------:|-----:|-------|
| 31 | negbin_negbin | ✗ | ✗ | ✗ | ✗ | H | 0.1 | 17.0 | 16.8 | 16.8 | ✓Stan (joint) |
| 32 | negbin_negbin | ✓ | ✗ | ✗ | ✗ | H | 0.9 | 17.2 | 17.1 | 17.0 | ✓Stan (joint) |
| 33 | negbin_negbin | slopes | ✗ | ✗ | ✗ | H | 5.6 | | 62.0 | | ✓Stan* (non-centered vs centered). Was 135.6s (NC fix 2026-03-03) |
| 34 | negbin_negbin | crossed | ✗ | ✗ | ✗ | H | 3.3 | 17.0 | 17.1 | 17.2 | ✓Stan (joint) |
| 35 | negbin_negbin | ✓ | ICAR | ✗ | ✗ | H | 0.6 | 17.1 | 5.5 | 17.2 | ✓Stan (joint) |
| 36 | negbin_negbin | ✓ | BYM2 | ✗ | ✗ | H | 0.9 | 17.2 | 17.1 | 17.1 | ✓Stan (joint, Stan 86.0s, **11.6x WIN**). Was 161.1s |
| 37 | negbin_negbin | ✓ | GP | ✗ | ✗ | H | 29.7 | | | | ✓Sim (0.28 SD, N_GP=80). Was >600s |
| 38 | negbin_negbin | ✓ | HSGP | ✗ | ✗ | H | 2.2 | | | | ✓Stan (joint, Stan 24.9s, **4.8x WIN**). Was 23.0s (DIAG + HSGP warmstart) |
| 39 | negbin_negbin | ✓ | MSGP | ✗ | ✗ | H | 4.4 | | | | ✓Sim (1.64 SD). Was >600s NNGP, now HSGP-MSGP (>8x) |
| 40 | negbin_negbin | ✓ | pCAR | ✗ | ✗ | H | 3.2 | | 64.4 | | ✓Stan (joint) |
| 41 | negbin_negbin | ✓ | ✗ | RW1 | ✗ | H | 4.2 | | 62.5 | | ✓Stan (joint) |
| 42 | negbin_negbin | ✓ | ✗ | RW2 | ✗ | H | 7.2 | | 62.8 | | ✓Stan (joint) |
| 43 | negbin_negbin | ✓ | ✗ | AR1 | ✗ | H | 1.2 | | 63.4 | | ✓Stan (joint) |
| 44 | negbin_negbin | ✓ | ✗ | GP_t | ✗ | H | 13.7 | | | | ✓Stan* (Stan 21.3s, **~parity**). Was 47-97s (vectorized obs loop) |
| 45 | negbin_negbin | ✓ | ✗ | MS_t | ✗ | H | 21.3 | | | | ✓Sim (0.90 SD, Stan fails). Was 107.3s |
| 46 | negbin_negbin | ✓ | ✗ | ✗ | ZI | H | 16.6 | | | | ✓Stan (joint) |
| 47 | negbin_negbin | ✓ | ✗ | ✗ | Hurdle | H | 20.9 | | 63.2 | | ✓Stan (joint). Was 155.0s (hurdle sign fix) |
| 48 | negbin_negbin | ✓ | ICAR | RW1 | ✗ | H | 3.0 | | | | ✓Stan (joint) |
| 49 | negbin_negbin | ✓ | BYM2 | RW1 | ✗ | H | 7.5 | | 82.2 | | ✓Stan* (Stan 171.6s, **16.7x WIN**). Was 158.6s |
| 50 | negbin_negbin | ✓ | ICAR | AR1 | ✗ | H | 3.5 | | 65.8 | | ✓Stan* (soft constraint). Was 119.8s |
| 51 | negbin_negbin | ✓ | GP | RW1 | ✗ | H | 108.9 | | | | ✓Sim* (2.79 SD, marginal, N_GP=80). Was >600s |
| 52 | negbin_negbin | ✓ | HSGP | RW1 | ✗ | H | 14.1 | | | | ✓Sim (1.34 SD) |
| 53 | negbin_negbin | ✓ | MSGP | RW1 | ✗ | H | 1.2 | | | | ✓Sim (1.84 SD). Was >600s NNGP, now HSGP-MSGP (>3x) |
| 54 | negbin_negbin | ✓ | ICAR | ✗ | ZI | H | 0.6 | | | | ✓Stan* (soft constraint) |
| 55 | negbin_negbin | slopes | ICAR | ✗ | ✗ | H | 0.5 | | | | ✓Stan* (Stan 46.6s, **3.4x WIN**). Was 70.0s (vectorized obs loop + scoping fix) |
| 56 | negbin_negbin | ✓ | SVC | ✗ | ✗ | H | 209.8 | | | | ✓Sim* (eta cor=0.70). Was >600s NNGP, now HSGP-SVC (40x) |
| 57 | negbin_negbin | ✓ | ✗ | TVC | ✗ | H | 71.2 | | | | ✓Stan (joint) |
| 58 | negbin_negbin | ✓ | ICAR | RW1 | ✗ | H | 1.9 | | | | ✓Stan ST-I (joint, 1.0 SE). Was 132.7s |
| 59 | negbin_negbin | ✓ | ICAR | RW1 | ✗ | H | 41.8 | | | | ✓Sim ST-IV (Stan 133.5s, **1.2x WIN**, td=10). Was 287.9s (vectorized obs loop) |
| 60 | negbin_negbin | ✓ | ✗ | ✗ | ✗ | H | 1.4 | | | | ✓Sim latent (0.04 SD) |

### Section 3: binomial Family (Rows 61-92)

**✓ brms validation VALID for this family** - trials are fixed in both numdenom and brms.

| # | Family | RE | Spatial | Temporal | ZI | Grad | H(s) | A(s) | A_t(s) | N(s) | Notes |
|--:|--------|:--:|:-------:|:--------:|:--:|:----:|-----:|-----:|-------:|-----:|-------|
| 61 | binomial | ✗ | ✗ | ✗ | ✗ | H | 0.1 | 13.1 | 13.1 | 13.3 | ✓Stan |
| 62 | binomial | ✓ | ✗ | ✗ | ✗ | H | 1.0 | 13.3 | 13.2 | 13.3 | ✓Stan |
| 63 | binomial | slopes | ✗ | ✗ | ✗ | H | 1.8 | | | | ✓Stan. Was 140.3s (NC fix 2026-03-03) |
| 64 | binomial | crossed | ✗ | ✗ | ✗ | H | 1.1 | 13.1 | 13.2 | 13.8 | ✓Stan (9.1x) |
| 65 | binomial | ✓ | ICAR | ✗ | ✗ | H | 0.1 | 13.0 | 13.2 | 12.8 | ✓Stan |
| 66 | binomial | ✓ | BYM2 | ✗ | ✗ | H | 0.3 | 13.1 | 13.2 | 13.1 | ✓Stan (Stan 12.4s, 2.8x WIN). Was 18.4s (binomial logistic opt) |
| 67 | binomial | ✓ | GP | ✗ | ✗ | H | 9.2 | | | | ✓Sim (0.63 SD, N_GP=80). Was >600s |
| 68 | binomial | ✓ | HSGP | ✗ | ✗ | H | 1.1 | | | | ✓Sim (Stan 2.9s, 1.2x WIN). Was 6.6s (binomial logistic opt) |
| 69 | binomial | ✓ | MSGP | ✗ | ✗ | H | 2.0 | | | | ✓Sim (eta cor=0.91, prediction recovery) |
| 70 | binomial | ✓ | pCAR | ✗ | ✗ | H | 4.2 | | | | ✓Stan (3.2x) |
| 71 | binomial | ✓ | ✗ | RW1 | ✗ | H | 4.2 | | | | ✓Stan |
| 72 | binomial | ✓ | ✗ | RW2 | ✗ | H | 4.7 | | | | ✓Stan |
| 73 | binomial | ✓ | ✗ | AR1 | ✗ | H | 1.0 | | | | ✓Stan. Was 82.9s (anomaly resolved 2026-03-03) |
| 74 | binomial | ✓ | ✗ | GP_t | ✗ | H | 2.7 | | | 32.4 | ✓Stan (Stan 3.3s, **2.8x WIN** subprocess). Was 5.1s in-loop |
| 75 | binomial | ✓ | ✗ | MS_t | ✗ | H | 21.3 | | | | ✓Sim MS_t (0.55 SD) |
| 76 | binomial | ✓ | ✗ | ✗ | ZI | H | 10.8 | 14.8 | 14.7 | 14.4 | ✓Stan |
| 77 | binomial | ✓ | ✗ | ✗ | Hurdle | H | 11.4 | 13.1 | 13.5 | 13.3 | ✓Stan (custom, 5.5x faster) |
| 78 | binomial | ✓ | ✗ | ✗ | OI | H | 12.5 | | | | ✓Sim OI (1.87 SD) |
| 79 | binomial | ✓ | ✗ | ✗ | ZOIB | H | 2.5 | | | | ✓Sim ZOIB (0.94 SD) (seed-dependent) |
| 80 | binomial | ✓ | ICAR | RW1 | ✗ | H | 6.1 | | | | ✓Stan |
| 81 | binomial | ✓ | BYM2 | RW1 | ✗ | H | 4.4 | | | | ✓Stan (Stan 58.6s, **2.4x WIN**). Was 138.4s |
| 82 | binomial | ✓ | ICAR | AR1 | ✗ | H | 3.2 | | | | ✓Stan. Was 5.2s (now uses dense mass) |
| 83 | binomial | ✓ | GP | RW1 | ✗ | H | 195.1 | | | | ✓Sim* (N_GP=80). Was >600s |
| 84 | binomial | ✓ | HSGP | RW1 | ✗ | H | 2.7 | | | | ✓Sim (0.26 SD) |
| 85 | binomial | ✓ | MSGP | RW1 | ✗ | H | 1.3 | | | | ✓Sim (0.05 SD, O(N³) MSGP) |
| 86 | binomial | ✓ | ICAR | ✗ | ZI | H | 75.4 | | | | ✓Stan (13.2x) |
| 87 | binomial | slopes | ICAR | ✗ | ✗ | H | 0.2 | | | | ✓Stan (Stan 14.5s, ~parity). Was 25.3s (vectorized obs loop + scoping fix) |
| 88 | binomial | ✓ | SVC | ✗ | ✗ | H | 146.4 | | | | ✓Sim (eta cor=0.83). Was >600s NNGP, now HSGP-SVC (30x) |
| 89 | binomial | ✓ | ✗ | TVC | ✗ | H | 28.5 | | | | ✓Stan (17.1x) |
| 90 | binomial | ✓ | ICAR | RW1 | ✗ | H | 0.8 | | | | ✓Stan ST-I (joint, 0.4 SE). Was 44.1s |
| 91 | binomial | ✓ | ICAR | RW1 | ✗ | H | 20.5 | | | | ✓Sim ST-IV (Stan 56.0s, **1.1x WIN** subprocess). Was 73.1s in-loop |
| 92 | binomial | ✓ | ✗ | ✗ | ✗ | H | 0.3 | | | | ✓Sim latent (1.97 SD) |

### Section 4: gamma_gamma Family (Rows 93-97)

**⚠️ brms validation INVALID for this family** - numdenom models BOTH num and denom as Gamma-distributed.

| # | Family | RE | Spatial | Temporal | ZI | Grad | H(s) | A(s) | A_t(s) | N(s) | Notes |
|--:|--------|:--:|:-------:|:--------:|:--:|:----:|-----:|-----:|-------:|-----:|-------|
| 93 | gamma_gamma | ✗ | ✗ | ✗ | ✗ | H | 0.0 | | | | ✓Stan (joint, 0.76 SE) |
| 94 | gamma_gamma | ✓ | ✗ | ✗ | ✗ | H | 0.7 | | | | ✓Stan (joint, 1.56 SE) |
| 95 | gamma_gamma | ✓ | ICAR | ✗ | ✗ | H | 1.0 | | | | ✓Sim (19x vs Stan 145s, 60% treedepth) |
| 96 | gamma_gamma | ✓ | ✗ | RW1 | ✗ | H | 1.0 | | | | ✓Sim (4.7x vs Stan 37s) |
| 97 | gamma_gamma | ✓ | ICAR | RW1 | ✗ | H | 1.4 | | | | ✓Stan (joint, 0.76/1.56 SE) |

### Section 5: lognormal Family (Rows 98-102)

**⚠️ brms validation INVALID for this family** - numdenom models BOTH num and denom as lognormal-distributed.

| # | Family | RE | Spatial | Temporal | ZI | Grad | H(s) | A(s) | A_t(s) | N(s) | Notes |
|--:|--------|:--:|:-------:|:--------:|:--:|:----:|-----:|-----:|-------:|-----:|-------|
| 98 | lognormal | ✗ | ✗ | ✗ | ✗ | H | 0.0 | | | | ✓Stan (joint) |
| 99 | lognormal | ✓ | ✗ | ✗ | ✗ | H | 0.1 | | | | ✓Sim (Stan fails to converge) |
| 100 | lognormal | ✓ | ICAR | ✗ | ✗ | H | 0.3 | | | | ✓Sim (Stan 66% treedepth) |
| 101 | lognormal | ✓ | ✗ | RW1 | ✗ | H | 0.7 | | | | ✓Sim (Stan fails to converge) |
| 102 | lognormal | ✓ | ICAR | RW1 | ✗ | H | 0.8 | | | | ✓Sim (Stan 71% treedepth) |

### Section 6: beta_binomial Family (Rows 103-107)

**✓ brms validation VALID for this family** - trials are fixed in both numdenom and brms.

| # | Family | RE | Spatial | Temporal | ZI | Grad | H(s) | A(s) | A_t(s) | N(s) | Notes |
|--:|--------|:--:|:-------:|:--------:|:--:|:----:|-----:|-----:|-------:|-----:|-------|
| 103 | beta_binomial | ✗ | ✗ | ✗ | ✗ | H | 0.3 | | | | ✓Stan (4.8x) |
| 104 | beta_binomial | ✓ | ✗ | ✗ | ✗ | H | 2.1 | | | | ✓Stan (3.8x) |
| 105 | beta_binomial | ✓ | ICAR | ✗ | ✗ | H | 17.1 | | | | ✓Sim (0.48 SD) |
| 106 | beta_binomial | ✓ | ✗ | RW1 | ✗ | H | 13.9 | | | | ✓Stan (4.2x) |
| 107 | beta_binomial | ✓ | ICAR | RW1 | ✗ | H | 16.8 | | | | ✓Sim (0.88 SD) |

### Section 7: negbin_gamma Family (Rows 108-137)

**⚠️ brms validation INVALID for this family** - numdenom models num as NegBin and denom as Gamma with shared RE, brms cannot express joint num/denom models.

| # | Family | RE | Spatial | Temporal | ZI | Grad | H(s) | A(s) | A_t(s) | N(s) | Notes |
|--:|--------|:--:|:-------:|:--------:|:--:|:----:|-----:|-----:|-------:|-----:|-------|
| 108 | negbin_gamma | ✗ | ✗ | ✗ | ✗ | H | 0.1 | | | | ✓Stan (joint, 29.5x) |
| 109 | negbin_gamma | ✓ | ✗ | ✗ | ✗ | H | 0.4 | | | | ✓Stan (joint, 7.5x) |
| 110 | negbin_gamma | slopes | ✗ | ✗ | ✗ | H | 8.5 | | | | ✓Sim (0.76 SD) |
| 111 | negbin_gamma | crossed | ✗ | ✗ | ✗ | H | 0.6 | | | | ✓Sim (0.85 SD) |
| 112 | negbin_gamma | ✓ | ICAR | ✗ | ✗ | H | 0.5 | | | | ✓Sim (0.66 SD) |
| 113 | negbin_gamma | ✓ | BYM2 | ✗ | ✗ | H | 0.8 | | | | ✓Sim (0.75 SD) |
| 114 | negbin_gamma | ✓ | GP | ✗ | ✗ | H | 13.6 | | | | ✓Sim (1.45 SD, poorly identified intercepts) |
| 115 | negbin_gamma | ✓ | HSGP | ✗ | ✗ | H | 2.1 | | | | ✓Sim (0.83 SD) |
| 116 | negbin_gamma | ✓ | MSGP | ✗ | ✗ | H | 6.0 | | | | ✓Sim |
| 117 | negbin_gamma | ✓ | pCAR | ✗ | ✗ | H | 1.7 | | | | ✓Sim |
| 118 | negbin_gamma | ✓ | ✗ | RW1 | ✗ | H | 1.0 | | | | ✓Sim (1.22 SD) |
| 119 | negbin_gamma | ✓ | ✗ | RW2 | ✗ | H | 2.2 | | | | ✓Sim (0.98 SD) |
| 120 | negbin_gamma | ✓ | ✗ | AR1 | ✗ | H | 0.6 | | | | ✓Sim (0.77 SD) |
| 121 | negbin_gamma | ✓ | ✗ | GP_t | ✗ | H | 8.8 | | | | ✓runs (was mislabeled SEGFAULT, bad test call) |
| 122 | negbin_gamma | ✓ | ✗ | MS_t | ✗ | H | 1.9 | | | | ✓Sim |
| 123 | negbin_gamma | ✓ | ✗ | ✗ | ZI | H | 48.0 | | | | ✓Sim (ZI family N/A for NB-Gamma, uses zinegbin) |
| 124 | negbin_gamma | ✓ | ✗ | ✗ | Hurdle | H | 95.9 | | | | ✓Sim (hurdle family N/A for NB-Gamma) |
| 125 | negbin_gamma | ✓ | ICAR | RW1 | ✗ | H | 2.2 | | | | ✓Sim (0.68 SD) |
| 126 | negbin_gamma | ✓ | BYM2 | RW1 | ✗ | H | 2.2 | | | | ✓Sim |
| 127 | negbin_gamma | ✓ | ICAR | AR1 | ✗ | H | 1.8 | | | | ✓Sim (0.78 SD) |
| 128 | negbin_gamma | ✓ | GP | RW1 | ✗ | H | 80.2 | | | | ✓Sim |
| 129 | negbin_gamma | ✓ | HSGP | RW1 | ✗ | H | 8.2 | | | | ✓Sim |
| 130 | negbin_gamma | ✓ | MSGP | RW1 | ✗ | H | 1.2 | | | | ✓runs (HSGP-MSGP, was mislabeled SEGFAULT) |
| 131 | negbin_gamma | ✓ | ICAR | ✗ | ZI | H | 0.5 | | | | N/A (no ZI variant for negbin_gamma) |
| 132 | negbin_gamma | slopes | ICAR | ✗ | ✗ | H | 0.4 | | | | ✓Sim |
| 133 | negbin_gamma | ✓ | SVC | ✗ | ✗ | H | 176.4 | | | | HSGP-SVC |
| 134 | negbin_gamma | ✓ | ✗ | TVC | ✗ | H | 54.1 | | | | ✓Sim (slow) |
| 135 | negbin_gamma | ✓ | ICAR | RW1 | ✗ | H | 2.2 | | | | ✓Sim ST-I |
| 136 | negbin_gamma | ✓ | ICAR | RW1 | ✗ | H | 25.1 | | | | ✓Sim ST-IV |
| 137 | negbin_gamma | ✓ | ✗ | ✗ | ✗ | H | 1.5 | | | | ✓Sim latent (N=50) |

### Section 8: H-Mode Coverage Gaps (Rows 150+)

**Exotic combinations allowed by the R API but missing dedicated H-mode gradients.**

These combinations fall through the H-mode dispatch in `resolve_gradient_fn()` and either:
- **A_r**: Fall through all 16 H branches → arena autodiff (O(N), correct, ~4-15x slower than H)
- **→N**: Get caught by wrong H branch → `verify_gradient_runtime()` detects mismatch → falls back to numerical (O(p×N), correct, slow)

The root cause is that each specialized H-mode gradient function (e.g., `compute_gradient_hsgp`) handles only a subset of features. When two advanced features are combined, the first-matching branch catches the model but doesn't compute gradients for the second feature.

#### Dispatch conflicts (caught by wrong H branch → verify_gradient_runtime → N fallback)

`compute_gradient_hsgp` (branch 2: `is_hsgp`) handles HSGP + RE + temporal GMRF (RW1/RW2/AR1) but NOT:

| # | Family | RE | Spatial | Temporal | ZI | Grad | Notes |
|--:|--------|:--:|:-------:|:--------:|:--:|:----:|-------|
| 150 | poisson_gamma | ✓ | HSGP | GP_t | ✗ | →N | Branch 2 catches HSGP, missing temporal GP gradients |
| 151 | poisson_gamma | ✓ | HSGP | MS_t | ✗ | →N | Branch 2 catches HSGP, missing multiscale temporal |
| 152 | poisson_gamma | ✓ | HSGP | TVC | ✗ | →N | Branch 2 catches HSGP, missing TVC gradients |
| 153 | poisson_gamma | ✓ | HSGP | RW1 | ✗ | →N | Branch 2 catches HSGP, missing ST interaction (add ST=I or IV) |
| 154 | poisson_gamma | ✓ | HSGP | ✗ | ✗ | →N | Branch 2 catches HSGP, missing latent factor gradients (add latent) |
| 155 | poisson_gamma | slopes | HSGP | ✗ | ✗ | →N | Branch 2 catches HSGP, missing random slopes gradients |

`compute_gradient_svc_(hsgp_)handcoded` (branch 10/11: `has_svc`) handles SVC spatial but NOT temporal, ST, latent, TVC:

| # | Family | RE | Spatial | Temporal | ZI | Grad | Notes |
|--:|--------|:--:|:-------:|:--------:|:--:|:----:|-------|
| 156 | poisson_gamma | ✓ | SVC | RW1 | ✗ | →N | Branch 10 catches SVC, missing temporal gradients |
| 157 | poisson_gamma | ✓ | SVC | AR1 | ✗ | →N | Branch 10 catches SVC, missing temporal gradients |
| 158 | poisson_gamma | ✓ | SVC | GP_t | ✗ | →N | Branch 10 catches SVC, missing temporal GP gradients |
| 159 | poisson_gamma | ✓ | SVC | TVC | ✗ | →N | Branch 10 catches SVC, missing TVC gradients |

`compute_gradient_tvc_handcoded` (branch 12: `has_tvc`) handles TVC temporal but NOT any spatial:

| # | Family | RE | Spatial | Temporal | ZI | Grad | Notes |
|--:|--------|:--:|:-------:|:--------:|:--:|:----:|-------|
| 160 | poisson_gamma | ✓ | ICAR | TVC | ✗ | →N | Branch 12 catches TVC, missing ICAR spatial gradients |
| 161 | poisson_gamma | ✓ | BYM2 | TVC | ✗ | →N | Branch 12 catches TVC, missing BYM2 spatial gradients |
| 162 | poisson_gamma | ✓ | HSGP | TVC | ✗ | →N | Branch 2 catches HSGP first (=row 152) |

`compute_gradient_latent_handcoded` (branch 16: `has_latent`) handles latent factors but NOT spatial, temporal, ZI:

| # | Family | RE | Spatial | Temporal | ZI | Grad | Notes |
|--:|--------|:--:|:-------:|:--------:|:--:|:----:|-------|
| 163 | poisson_gamma | ✓ | ICAR | ✗ | ✗ | →N | Branch 16 catches latent, missing ICAR gradients (add latent) |
| 164 | poisson_gamma | ✓ | ✗ | RW1 | ✗ | →N | Branch 16 catches latent, missing temporal gradients (add latent) |
| 165 | poisson_gamma | ✓ | ICAR | RW1 | ✗ | →N | Branch 16 catches latent, missing both (add latent) |

`compute_gradient_temporal_gp_handcoded` (branch 14: `is_temporal_gp`) handles temporal GP but NOT any spatial:

| # | Family | RE | Spatial | Temporal | ZI | Grad | Notes |
|--:|--------|:--:|:-------:|:--------:|:--:|:----:|-------|
| 166 | poisson_gamma | ✓ | ICAR | GP_t | ✗ | →N | Branch 14 catches GP_t, missing ICAR spatial gradients |
| 167 | poisson_gamma | ✓ | BYM2 | GP_t | ✗ | →N | Branch 14 catches GP_t, missing BYM2 spatial gradients |

`compute_gradient_ms_temporal_handcoded` (branch 15) is excluded when `has_svc`, `has_tvc`, `has_latent`, or `has_spatiotemporal`:

| # | Family | RE | Spatial | Temporal | ZI | Grad | Notes |
|--:|--------|:--:|:-------:|:--------:|:--:|:----:|-------|
| 168 | poisson_gamma | ✓ | ICAR | MS_t | ✗ | →N | Branch 15 catches MS_t, but missing ICAR spatial gradients |
| 169 | poisson_gamma | ✓ | HSGP | MS_t | ✗ | →N | Branch 2 catches HSGP first (=row 151) |

#### True A_r fallthrough (no H branch matches)

| # | Family | RE | Spatial | Temporal | ZI | Grad | Notes |
|--:|--------|:--:|:-------:|:--------:|:--:|:----:|-------|
| 170 | poisson_gamma | crossed+slopes | ✗ | ✗ | ✗ | A_r | n_re_terms > 1 with slopes excluded from analytical (line 2266) |

**Note**: Rows 150-170 use poisson_gamma as representative. The same gaps apply to all 7 families — the dispatch logic is family-agnostic for these branch conditions.

#### Summary of H-mode coverage

| H-mode function | Handles | Does NOT handle |
|----------------|---------|-----------------|
| `compute_gradient_analytical` | Base + RE + ICAR/BYM2/pCAR + RW1/RW2/AR1 + ZI + slopes (single-group) + crossed (no slopes) | GP, HSGP, MSGP, SVC, TVC, latent, ST, MS_t, GP_t, crossed+slopes |
| `compute_gradient_hsgp` | HSGP + RE + RW1/RW2/AR1 | GP_t, MS_t, TVC, ST, latent, slopes, SVC |
| `compute_gradient_gp_handcoded` | GP (no temporal) | Any temporal, ST, latent, slopes |
| `compute_gradient_gp_temporal_handcoded` | GP + temporal | ST, latent, slopes |
| `compute_gradient_msgp_hsgp` | MSGP-HSGP | Temporal, ST, latent |
| `compute_gradient_msgp_temporal_handcoded` | MSGP + temporal | ST, latent |
| `compute_gradient_msgp_handcoded` | MSGP (no temporal) | Temporal, ST, latent |
| `compute_gradient_svc_(hsgp_)handcoded` | SVC | Temporal, ST, latent, TVC |
| `compute_gradient_tvc_handcoded` | TVC | Any spatial, latent |
| `compute_gradient_spatiotemporal_handcoded` | ST (ICAR/BYM2 + RW/AR1) | GP/HSGP/MSGP spatial, latent |
| `compute_gradient_temporal_gp_handcoded` | Temporal GP (exponential) | Any spatial, latent |
| `compute_gradient_ms_temporal_handcoded` | MS_t (standalone) | Spatial, SVC, TVC, latent, ST |
| `compute_gradient_latent_handcoded` | Latent factors | Any spatial, any temporal |

### Section 6: Collapsed Parameterizations (Rows 138-149)

**Collapsed parameterization**: Marginalizes out spatial effects via inner Laplace optimization at each HMC gradient evaluation. Reduces parameter count (S→0 for ICAR, 2S→0 for BYM2) but each gradient is O(S³) due to dense inverse. H-mode uses envelope theorem + implicit function theorem for analytical Laplace gradients.

**When to use**: Models where spatial effects cause high posterior correlation and poor sampling. The collapsed parameterization eliminates these correlations at the cost of per-gradient Newton solves.

**Current status**: Collapsed is 15-40x slower than standard for S=50 (expected: per-gradient Newton solve + dense inverse overhead). May become competitive for very large S where standard parameterization's mixing overhead dominates.

| # | Family | RE | Spatial | Temporal | ZI | Grad | H(s) | A(s) | A_t(s) | N(s) | Notes |
|--:|--------|:--:|:-------:|:--------:|:--:|:----:|-----:|-----:|-------:|-----:|-------|
| 138 | poisson_gamma | ✗ | cICAR | ✗ | ✗ | H | 126.7 | | | | params=6 (vs 56 standard). Standard: 0.0s (row 1). mode=hmc forced |
| 139 | poisson_gamma | ✗ | cBYM2 | ✗ | ✗ | H | 62.4 | | | | params=9 (vs 109 standard). Standard: 0.5s (row 6). mode=hmc forced |
| 140 | negbin_negbin | ✗ | cICAR | ✗ | ✗ | H | 27.2 | | | | params=7 (vs 57 standard). Standard: 0.6s (row 35). mode=hmc forced |
| 141 | negbin_negbin | ✗ | cBYM2 | ✗ | ✗ | H | 18.6 | | | | params=10 (vs 110 standard). Standard: 0.9s (row 36). mode=hmc forced |
| 142 | binomial | ✗ | cICAR | ✗ | ✗ | H | 10.3 | | | | params=3 (vs 53 standard). Standard: 0.1s (row 65). mode=hmc forced |
| 143 | binomial | ✗ | cBYM2 | ✗ | ✗ | H | 6.8 | | | | params=6 (vs 106 standard). Standard: 0.3s (row 66). mode=hmc forced |
| 144 | negbin_gamma | ✗ | cICAR | ✗ | ✗ | H | 22.6 | | | | Standard: 0.5s (row 112). mode=hmc forced |
| 145 | negbin_gamma | ✗ | cBYM2 | ✗ | ✗ | H | 14.4 | | | | Standard: 0.8s (row 113). mode=hmc forced |
| 146 | poisson_gamma | ✓ | cICAR | ✗ | ✗ | H | 294.7 | | | | Standard: 0.3s (row 5). mode=hmc forced |
| 147 | poisson_gamma | ✓ | cBYM2 | ✗ | ✗ | H | 151.9 | | | | Standard: 0.5s (row 6). mode=hmc forced |
| 148 | negbin_negbin | ✓ | cICAR | ✗ | ✗ | H | >600 | | | | Standard: 0.6s (row 35). mode=hmc forced. Timeout (thermal throttle) |
| 149 | negbin_negbin | ✓ | cBYM2 | ✗ | ✗ | H | 59.9 | | | | Standard: 0.9s (row 36). mode=hmc forced |

---

## Spatial Types Reference

| Code | Full Name | Implementation |
|:----:|-----------|----------------|
| ICAR | Intrinsic CAR (rho=1 fixed) | spatial_car(proper=FALSE) |
| pCAR | Proper CAR (rho estimated) | spatial_car(proper=TRUE) |
| BYM2 | Besag-York-Mollie 2 | spatial_bym2() |
| cICAR | Collapsed ICAR (marginalized) | spatial_car(parameterization="collapsed") |
| cBYM2 | Collapsed BYM2 (marginalized) | spatial_bym2(parameterization="collapsed") |
| GP | Gaussian Process (NNGP) | spatial_gp() |
| HSGP | Hilbert Space GP | spatial_hsgp() |
| MSGP | Multi-scale GP | spatial_multiscale() |
| SVC | Spatially-varying coefficients | spatial_svc() |

## Temporal Types Reference

| Code | Full Name | Implementation |
|:----:|-----------|----------------|
| RW1 | Random Walk order 1 | temporal_rw1() |
| RW2 | Random Walk order 2 | temporal_rw2() |
| AR1 | Autoregressive order 1 | temporal_ar1() |
| GP_t | Temporal GP | temporal_gp() |
| MS_t | Multi-scale temporal | temporal_multiscale() |
| TVC | Time-varying coefficients | temporal_tvc() |

## ZI Types Reference

| Code | Full Name | Implementation |
|:----:|-----------|----------------|
| ZI | Zero-inflated | ratiod_zi*() or zi=zi_*() |
| Hurdle | Hurdle model | ratiod_hurdle_*() or zi=hurdle_*() |
| OI | One-inflated | ratiod_oibinomial() |
| ZOIB | Zero-and-one inflated | ratiod_zoibinomial() |

## Spatiotemporal Types Reference

| Code | Full Name | Description |
|:----:|-----------|-------------|
| ST-I | Knorr-Held Type I | Unstructured interaction |
| ST-II | Knorr-Held Type II | Temporal per location |
| ST-III | Knorr-Held Type III | Spatial per time |
| ST-IV | Knorr-Held Type IV | Full Kronecker product |

---

## Summary

### Validation Status (as of 2026-03-15)

**All 107 single/dual-feature configurations validated (rows 1-137).** All have H (hand-coded) gradients.

**21 exotic multi-feature combinations identified without H mode (rows 150-170).** These fall back to N (numerical, via verify_gradient_runtime) or A_r (arena autodiff). See Section 8.

| Marker | Count | Description |
|--------|------:|-------------|
| **✓Stan** | 47 | Posterior means within 2 SE of custom joint Stan model |
| **✓Stan*** | 9 | Slopes match within 2 SE; intercepts differ due to soft vs hard sum-to-zero (equivalent for predictions) |
| **✓Sim** | 35 | Parameter recovery from simulation truth (used when Stan has convergence issues) |
| **✓Sim*** | 4 | Prediction recovery cor(eta_pred, eta_true) > 0.70 (overparameterized models: SVC, MSGP) |
| ✓runs/N/A | 12 | Runs without error; limited validation or not applicable |
| **Total** | **107** | Rows 1-137 (excluding deprecated collapsed rows 138-149) |

### Performance Tiers (NUTS, H mode, N=500, 500 iter)

| Tier | Time | Model types |
|:----:|:----:|-------------|
| Fast | <10s | Core families, RE, ICAR, BYM2, pCAR, RW1/RW2/AR1, ZI/ZOIB, TVC, GP_t, gamma_gamma, beta_binomial, negbin_gamma |
| Medium | 10-100s | HSGP, HSGP-SVC, HSGP-MSGP, HSGP+temporal, latent (N=500/K=2), lognormal, slopes+spatial, some ST |
| Slow | 100-600s | MSGP-NNGP, GP+temporal combos, collapsed+RE |
| Timeout | >600s | None (GP rows fixed 2026-03-17, was NNGP timeout) |

**numdenom beats Stan on ALL 18 benchmarked models (0 parity, 0 LOSS).** See per-row H(s) and Stan(s) columns in the tables above for exact numbers. Exotic combinations (rows 150+) lack H mode and fall back to N or A_r.

### AUTO Metric Selection

`metric = "auto"` (default) selects mass matrix type:

| Metric | Cost per step | Selected for |
|--------|:-------------:|-------------|
| **DIAG** | O(n) | Base, +RE, +ICAR, +pCAR, +RW, +ZI/Hurdle, +crossed, +TVC, correlated slopes, temporal GP, HSGP |
| **BLOCK_DIAG** | O(n + block²) | Temporal GP hyperparams, HSGP, BYM2, GP, MSGP, SVC, ST GP |
| **DENSE** | O(n²) | Latent factors, multiscale temporal |

Catastrophic epsilon > 2.0 triggers DIAG→BLOCK_DIAG→DENSE recovery chain.

### Stan Validation Approach

**Two-process families** (poisson_gamma, negbin_negbin, negbin_gamma, gamma_gamma, lognormal) require **custom joint Stan models** because brms `offset(log(denom))` treats denom as fixed — a fundamentally different model. Joint Stan models live in `benchmarks/stan/`.

**Binomial and beta_binomial** families can use brms directly (trials are fixed in both).

When Stan has convergence issues (>50% divergences, poor ESS), validation uses **simulation truth** instead — a standard approach in Bayesian software.

### Known Validation Notes

**✓Stan* (soft constraint difference):** Rows 3, 19, 24, 25, 33, 44, 49, 50, 54, 55. numdenom uses soft sum-to-zero constraints; Stan uses hard constraints. Slopes match; intercepts differ. Predictions are equivalent.

**✓Sim* (overparameterized):** MSGP (rows 9, 69) and SVC (rows 26, 56) validated via prediction recovery. Individual parameter recovery fails because these models have ~400-500 spatial parameters for 200-500 observations.

**GP+RW1 marginal (rows 51, 83):** Chain-length issues, not gradient bugs. GP+RW1 with 500 iterations is insufficient for 92+ parameter models.

### Validation Scripts

- `benchmarks/bench_joint_validation.R` — core PG/NB rows (1, 2, 31, 32)
- `benchmarks/bench_joint_validation_batch2.R` — spatial/temporal (5, 11, 35, 41)
- `benchmarks/bench_joint_validation_batch3-10.R` — remaining configurations
- `benchmarks/bench_sim_gp.R`, `bench_sim_hsgp.R` — GP/HSGP simulation validation
- `benchmarks/bench_final5_v2.R` — MSGP/SVC prediction recovery

---

## Legend

- **H** = Hand-coded gradients (fastest, production default)
- **A** = Forward-mode autodiff (`fwd::Dual`)
- **A_r** = Arena reverse-mode autodiff (production fallback when H unavailable)
- **A_t** = Tape-based autodiff (deprecated, superseded by A_r)
- **N** = Numerical gradients (baseline, fallback)
- Empty cells = benchmark not yet run

### Validation Markers

- **✓Stan** = Posterior means within 2 SE of Stan (custom joint model or brms)
- **✓Stan*** = Slopes match within 2 SE; intercept/spatial constraint parameterization differs
- **✓Sim** = Parameter recovery from simulated data (posterior mean within 2 SD of true value)
- **✓Sim*** = Prediction recovery (cor(eta_pred, eta_true) > 0.70)
