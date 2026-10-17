# Pólya-Gamma Backend Extension Plan

## Executive Summary

Extend the Pólya-Gamma (PG) Gibbs sampler from binomial-only to all applicable families. PG augmentation enables efficient conjugate Gibbs sampling for models with logit links, avoiding the computational overhead of HMC while maintaining Tier 1 (exact) inference guarantees.

**Reference**: Polson, Scott & Windle (2013). "Bayesian Inference for Logistic Models Using Pólya-Gamma Latent Variables." JASA 108(504):1339-1349.

**Scope**: PG augmentation is designed for **logit/binomial likelihoods**. However, negative binomial can also use PG via the Zhou et al. (2012) compound representation with Chinese Restaurant Table (CRT) sampling for the dispersion parameter. This is more complex but provides a complete PG-based inference suite — no HMC required for the core numdenom families. Poisson-Gamma remains HMC-only (no natural PG scheme).

---

## Current State

### What Exists (v1.2)

| Component | Status | File |
|-----------|--------|------|
| PG RNG (Devroye method) | Complete | `src/pg_rng.cpp/.h` |
| Binomial + RE | Complete | `src/pg_binomial.cpp/.h` |
| Binomial + ICAR | Complete | `src/pg_spatial.cpp/.h` |
| Binomial + BYM2 | Complete | `src/pg_spatial.cpp/.h` |
| Binomial + GP (NNGP) | Complete | `src/pg_spatial.cpp/.h` |
| Binomial + RSR | Complete | `src/pg_spatial.cpp/.h` |
| Binomial + Temporal | Complete | `R/backend_pg.R` |
| R interface | Complete | `R/backend_pg.R` |

### Current Limitations

1. **Only `ratiod_binomial()`** - Other families use HMC
2. **No random slopes** - Warning issued, slopes ignored
3. **No ZI binomial variants** - `ratiod_zibinomial()` etc. use HMC
4. **No crossed RE with slopes** - Only crossed intercepts supported

---

## PG Applicability by Family

### Direct PG Applicability (logit link)

| Family | PG Applicable | Reason |
|--------|:-------------:|--------|
| `ratiod_binomial()` | **Yes** | Core PG use case |
| `ratiod_zibinomial()` | **Yes** | Two logit links: P(zero) + P(success\|non-zero) |
| `ratiod_oibinomial()` | **Yes** | Two logit links: P(one) + P(success\|non-one) |
| `ratiod_zoibinomial()` | **Yes** | Three logit links |
| `ratiod_hurdle_binomial()` | **Yes** | Two logit links |

### Indirect PG Applicability (via augmentation)

| Family | PG Applicable | Method | Complexity |
|--------|:-------------:|--------|------------|
| `ratiod_negbin_negbin()` | **Yes** | PG + CRT (Zhou et al. 2012) | High |
| `ratiod_poisson_gamma()` | No | No natural PG scheme | — |
| `ratiod_beta_binomial()` | **Partial** | PG for binomial component | Medium |
| `ratiod_gamma_gamma()` | No | No logit component | — |
| `ratiod_lognormal()` | No | Gaussian, not logit | — |

### Negative Binomial Note

Negative binomial can use a compound representation:
```
Y ~ NB(r, p)  ≡  Y | λ ~ Poisson(λ), λ ~ Gamma(r, p/(1-p))
```

The `p` parameter can be modeled with logit link and PG augmentation (Zhou et al., 2012), but:
- Requires Chinese Restaurant Table (CRT) distribution for `r` updates
- More complex than pure Gibbs
- Marginal benefit over HMC is smaller for NB than binomial

**Recommendation**: Implement PG for binomial variants first (Phase 1), defer NB-PG to Phase 2 if benchmarks show benefit.

---

## Implementation Phases

### Phase 1: ZI/Hurdle Binomial Variants (Priority: High)

These are direct extensions — same PG machinery, just two linked binomial regressions.

#### 1.1 Zero-Inflated Binomial (`ratiod_zibinomial()`)

**Model**:
```
P(Y = 0) = π + (1 - π) * (1 - p)^n
P(Y = k) = (1 - π) * Binomial(k | n, p)   for k > 0

logit(π) = X_zi * β_zi + Z_zi * b_zi      # Zero-inflation probability
logit(p) = X * β + Z * b                   # Success probability (if not structural zero)
```

**PG augmentation**:
- Introduce latent `z_i ∈ {0, 1}` indicating structural zero (z=1) vs count process (z=0)
- For z-component: `ω_zi ~ PG(1, η_zi)` where `η_zi = logit(π_i)`
- For count component: `ω_i ~ PG(n_i, η_i)` where `η_i = logit(p_i)`
- Both become conditionally Gaussian given PG draws

**Files to create**:
- `src/pg_zi_binomial.h` - C++ sampler
- `src/pg_zi_binomial.cpp` - Implementation

**Gibbs steps**:
1. Sample `z_i | y_i, π_i, p_i` (Bernoulli, closed form)
2. Sample `ω_zi | z_i, η_zi` (PG, for ZI component)
3. Sample `β_zi, b_zi | z, ω_zi, ...` (Gaussian)
4. Sample `ω_i | y_i, z_i, η_i` (PG, for count component, only where z=0)
5. Sample `β, b | y, z, ω, ...` (Gaussian)
6. Sample variance components (half-Cauchy MH or slice)

#### 1.2 One-Inflated Binomial (`ratiod_oibinomial()`)

Analogous to ZI but `z_i = 1` means `y_i = n_i` (all successes).

#### 1.3 Zero-One Inflated Binomial (`ratiod_zoibinomial()`)

Three-component model:
```
P(Y = 0) = π_0
P(Y = n) = π_1
P(Y = k) = (1 - π_0 - π_1) * Binomial(k | n, p)   for 0 < k < n
```

Requires multinomial latent `z_i ∈ {0, 1, 2}` with stick-breaking or softmax.

#### 1.4 Hurdle Binomial (`ratiod_hurdle_binomial()`)

Two-part model:
```
P(Y = 0) = 1 - θ
P(Y = k | Y > 0) = TruncatedBinomial(k | n, p, k > 0)   for k > 0
```

Similar to ZI but truncated binomial in second stage.

### Phase 2: Random Slopes (Priority: Medium)

Currently warned and ignored. Full implementation requires:

**Model extension**:
```
η_i = X_i * β + Z_i * b_g[i]
b_g ~ MVN(0, Σ_b)    # Correlated intercept + slopes
```

**Challenge**: PG gives conditionally Gaussian for `b_g`, but `Σ_b` update requires:
- Inverse-Wishart (conjugate but restrictive)
- Or separation strategy (LKJ + half-Cauchy for variances)

**Files to modify**:
- `src/pg_binomial.h/.cpp` - Add slope handling
- `R/backend_pg.R` - Enable slope terms

**Gibbs steps** (new):
1. Sample `b_g | ω, β, Σ_b` (multivariate Gaussian)
2. Sample `Σ_b | b` (inverse-Wishart or parameter-expanded)

### Phase 3: Negative Binomial with PG+CRT (Priority: High)

**Why implement**:
- Completes the PG backend for all count-based families (binomial + negbin)
- Demonstrates full data augmentation mastery — not just "easy" PG
- Enables pure Gibbs sampling for `ratiod_negbin_negbin()` (no MH, no gradients)
- Differentiator vs other packages (most stop at binomial PG)
- CRT is elegant and underappreciated in ecology

**Model** (NB2 parameterization):
```
Y_i ~ NB(r, p_i)
E[Y_i] = μ_i = r * p_i / (1 - p_i)
Var[Y_i] = μ_i + μ_i² / r

logit(p_i) = η_i = X_i * β + Z_i * b
```

**Key insight**: The NB can be written as a Poisson-Gamma mixture, but for PG we use:
```
Y_i | ω_i ~ ... (conditionally Gaussian in η after PG augmentation)
```

**Augmentation scheme**:
1. `ω_i ~ PG(y_i + r, η_i)` — Pólya-Gamma for the logit component
2. `L_i ~ CRT(y_i, r)` — Chinese Restaurant Table for dispersion update

**Complete Gibbs sampler**:

```
Initialize: β, b, r, σ²_b

For t = 1, ..., T:
  # 1. Sample PG auxiliary variables
  For i = 1, ..., n:
    η_i = X_i * β + Z_i * b
    ω_i ~ PG(y_i + r, η_i)

  # 2. Update regression coefficients (conjugate Gaussian)
  κ_i = (y_i - r) / 2
  # Pseudo-observations: κ_i / ω_i ~ N(η_i, 1/ω_i)
  Ω = diag(ω)
  V_β = (X'ΩX + Σ_0^{-1})^{-1}
  μ_β = V_β * X'Ω(κ/ω - Zb)
  β ~ N(μ_β, V_β)

  # 3. Update random effects (conjugate Gaussian)
  V_b = (Z'ΩZ + I/σ²_b)^{-1}
  μ_b = V_b * Z'Ω(κ/ω - Xβ)
  b ~ N(μ_b, V_b)

  # 4. Update RE variance (half-Cauchy via slice or MH)
  σ²_b ~ p(σ²_b | b)

  # 5. Sample CRT auxiliary variables for dispersion
  For i = 1, ..., n:
    L_i ~ CRT(y_i, r)   # Number of "tables" in CRP

  # 6. Update dispersion r (Gamma conjugate)
  # Prior: r ~ Gamma(a_r, b_r)
  L_total = Σ L_i
  p_mean = mean(p_i)  # or use sufficient statistic
  r ~ Gamma(a_r + L_total, b_r - Σ log(1 - p_i))
```

**CRT Distribution**:

The Chinese Restaurant Table distribution `L ~ CRT(y, r)` counts the number of tables when `y` customers are seated in a Chinese Restaurant Process with concentration `r`:

```
P(L = l | y, r) ∝ |s(y, l)| * r^l   for l = 1, ..., y
```

where `|s(y, l)|` are unsigned Stirling numbers of the first kind.

**Sampling CRT(y, r)**:
```cpp
int sample_crt(int y, double r) {
  if (y == 0) return 0;
  int L = 0;
  for (int i = 1; i <= y; i++) {
    // Probability of new table for customer i
    double p_new_table = r / (r + i - 1);
    if (R::runif(0, 1) < p_new_table) {
      L++;
    }
  }
  return L;
}
```

This is O(y) per observation — efficient for moderate counts.

**Files to create**:
- `src/pg_negbin.h` — NB-PG sampler header
- `src/pg_negbin.cpp` — NB-PG implementation
- `src/crt_rng.h` — CRT sampler header
- `src/crt_rng.cpp` — CRT implementation

**Extensions for numdenom**:
- `ratiod_negbin_negbin()`: Two NB processes, each with PG+CRT
- Shared random effects enter both linear predictors
- Dispersion `r_num` and `r_denom` updated independently
- Spatial effects: same ICAR/BYM2/GP machinery as binomial PG

### Phase 4: Spatial Extensions for ZI (Priority: Medium)

Once Phase 1 complete, add spatial support:

| Spatial Type | ZI Extension | Complexity |
|--------------|--------------|------------|
| ICAR | Shared spatial for both processes | Low |
| BYM2 | Shared or separate BYM2 | Medium |
| GP/NNGP | Shared GP or bivariate GP | High |

**Key decision**: Should ZI probability and count probability share the same spatial field?
- **Shared** (default): One spatial field enters both linear predictors
- **Separate**: Two independent spatial fields (doubles parameters)

Recommend shared by default (follows numdenom philosophy), separate as option.

---

## File Structure

### New Files (Phase 1)

```
src/
├── pg_zi_binomial.h        # ZI binomial sampler header
├── pg_zi_binomial.cpp      # ZI binomial implementation
├── pg_oi_binomial.h        # OI binomial sampler
├── pg_oi_binomial.cpp
├── pg_zoi_binomial.h       # ZOI binomial sampler
├── pg_zoi_binomial.cpp
├── pg_hurdle_binomial.h    # Hurdle binomial sampler
└── pg_hurdle_binomial.cpp

R/
└── backend_pg.R            # Modify: add dispatch for ZI families
```

### New Files (Phase 2)

```
src/
├── pg_binomial_slopes.h    # Binomial with random slopes
└── pg_binomial_slopes.cpp
```

### New Files (Phase 3)

```
src/
├── pg_negbin.h             # NB with PG+CRT augmentation
├── pg_negbin.cpp
├── crt_rng.h               # Chinese Restaurant Table sampler
└── crt_rng.cpp
```

---

## API Design

### User-Facing (No Change)

```r
# Existing API works automatically
fit <- ratiod(
  successes | trials ~ x + (1 | site),
  data = df,
  family = ratiod_zibinomial(),  # NEW: Now uses PG backend
  mode = "pg"                     # Or auto-selected
)
```

### Backend Selection Logic

```r
can_use_pg_backend <- function(family) {
  # Current
  if (family$numerator$distribution == "binomial") return(TRUE)

  # Phase 1 additions
  if (family$name %in% c("zibinomial", "oibinomial",
                         "zoibinomial", "hurdle_binomial")) return(TRUE)

  # Phase 3 addition (if implemented)
  # if (family$numerator$distribution == "neg_binomial_2") return(TRUE)

  FALSE
}
```

---

## Benchmarking Requirements

### Phase 1 Benchmarks

For each ZI variant, compare:

| Config | PG Time | HMC Time | Speedup | ESS/s (PG) | ESS/s (HMC) |
|--------|---------|----------|---------|------------|-------------|
| N=500, no RE | | | | | |
| N=500, 1 RE | | | | | |
| N=500, ICAR | | | | | |
| N=500, BYM2 | | | | | |
| N=500, GP | | | | | |

**Target**: PG should be ≥2x faster than HMC for binomial-family models.

### Validation Protocol

1. **Posterior means**: Within 0.1 SD of HMC reference
2. **Posterior SDs**: Within 20% of HMC reference
3. **Coverage**: 95% intervals achieve 93-97% coverage on simulated data
4. **ESS**: Bulk and tail ESS within 50% of HMC

---

## Testing Strategy

### Unit Tests

```r
# tests/testthat/test-pg-zi-binomial.R

test_that("PG ZI binomial recovers parameters", {
  set.seed(123)
  sim <- simulate_zibinomial(n = 200, n_sites = 10,
                              pi_true = 0.3, p_true = 0.6)

  fit_pg <- ratiod(y | n ~ x + (1 | site), data = sim$data,
                   family = ratiod_zibinomial(), mode = "pg",
                   iter = 2000, warmup = 1000, chains = 2)

  fit_hmc <- ratiod(y | n ~ x + (1 | site), data = sim$data,
                    family = ratiod_zibinomial(), mode = "hmc",
                    iter = 2000, warmup = 1000, chains = 2)

  # Compare posteriors
  expect_equal(coef(fit_pg)["pi"], coef(fit_hmc)["pi"], tolerance = 0.1)
  expect_equal(coef(fit_pg)["p"], coef(fit_hmc)["p"], tolerance = 0.1)
})
```

### Integration Tests

- ZI + ICAR spatial
- ZI + BYM2 spatial
- ZI + random slopes (Phase 2)
- ZI + temporal (RW1, AR1)

---

## Timeline & Dependencies

```
Phase 1: ZI Binomial Variants [HIGH]
├── 1.1 ratiod_zibinomial() PG
├── 1.2 ratiod_oibinomial() PG
├── 1.3 ratiod_zoibinomial() PG
└── 1.4 ratiod_hurdle_binomial() PG

Phase 2: Random Slopes [MEDIUM]
├── 2.1 Binomial + slopes
├── 2.2 ZI variants + slopes
└── 2.3 Update covariance estimation

Phase 3: Negative Binomial + CRT [HIGH]
├── 3.1 CRT sampler implementation
├── 3.2 NB-PG sampler (single NB process)
├── 3.3 ratiod_negbin_negbin() with dual PG+CRT
├── 3.4 ZI-NegBin variants (ratiod_zinegbin, ratiod_hurdle_negbin)
└── 3.5 Benchmarking vs HMC

Phase 4: Spatial Extensions [MEDIUM]
├── 4.1 ZI + ICAR/BYM2
├── 4.2 ZI + GP/NNGP
├── 4.3 NegBin + spatial
└── 4.4 Shared vs separate spatial fields
```

**Dependencies**:
- Phase 2 depends on Phase 1 (build on ZI infrastructure)
- Phase 3 independent (can start immediately)
- Phase 4 depends on Phase 1 + Phase 3

---

## Why This Matters (The Flex)

Most R packages that implement Pólya-Gamma stop at basic binomial regression:
- **BayesLogit**: Binomial only
- **pgdraw**: Just the RNG
- **spOccupancy**: Binomial (occupancy/detection)

numdenom with full PG+CRT will offer:

| Feature | Others | numdenom |
|---------|:------:|:--------:|
| Binomial PG | ✓ | ✓ |
| ZI/Hurdle Binomial PG | — | ✓ |
| Negative Binomial PG+CRT | — | ✓ |
| ZI-NegBin PG+CRT | — | ✓ |
| Spatial (ICAR/BYM2/GP) | partial | ✓ |
| Two-process joint models | — | ✓ |
| Shared latent structure | — | ✓ |

**Result**: A complete data augmentation Gibbs sampler for hierarchical ratio models — pure conjugate sampling, no gradients, no tuning, guaranteed mixing for well-specified models.

This positions numdenom as the reference implementation for PG-based ecological modeling, extending spOccupancy's philosophy from occupancy to ratio inference.

---

## References

1. Polson, N.G., Scott, J.G., and Windle, J. (2013). "Bayesian Inference for Logistic Models Using Pólya-Gamma Latent Variables." JASA 108(504):1339-1349.

2. Zhou, M., Li, L., Dunson, D., and Carin, L. (2012). "Lognormal and Gamma Mixed Negative Binomial Regression." ICML.

3. Doser, J.W., Finley, A.O., Kéry, M., and Zipkin, E.F. (2022). "spOccupancy: An R package for single-species, multi-species, and integrated spatial occupancy models." MEE 13(8):1670-1678.

4. Datta, A., Banerjee, S., Finley, A.O., and Gelfand, A.E. (2016). "Hierarchical Nearest-Neighbor Gaussian Process Models for Large Geostatistical Datasets." JASA 111(514):800-812.

---

## Appendix: PG Sampling Algorithm

### PG(b, c) Sampler (Devroye Method for b=1)

For `ω ~ PG(1, c)`:

```cpp
double rpg1(double c) {
  // Sample J*(1,c) using Devroye's method
  double z = std::abs(c) / 2.0;
  double K = M_PI * M_PI / 8.0 + z * z / 2.0;

  while (true) {
    // Proposal from mixture of truncated exponential and inverse Gaussian
    double X = sample_jacobi_tilted(z);

    // Accept/reject
    double S = mass_ratio(X, z);
    if (R::runif(0, 1) < S) {
      return X / 4.0;  // Transform J* to PG
    }
  }
}
```

For `ω ~ PG(b, c)` with integer `b`:
```cpp
double rpg_int(int b, double c) {
  double sum = 0.0;
  for (int i = 0; i < b; i++) {
    sum += rpg1(c);
  }
  return sum;
}
```

### Conditional Updates After PG Augmentation

Given `ω_i ~ PG(n_i, η_i)` and `κ_i = y_i - n_i/2`:

**Observation "pseudo-data"**:
```
κ_i / ω_i ~ N(η_i, 1/ω_i)
```

**Beta update** (normal-normal conjugacy):
```
β | ω, κ, b ~ N(μ_post, Σ_post)

Σ_post = (X'ΩX + Σ_0^{-1})^{-1}
μ_post = Σ_post (X'Ω(κ/ω - Zb) + Σ_0^{-1} μ_0)
```

where `Ω = diag(ω_1, ..., ω_n)`.

This is standard weighted least squares — fast and numerically stable.

---

## Appendix: CRT Sampling Algorithm

### Chinese Restaurant Table Distribution

The CRT distribution arises from the Chinese Restaurant Process (CRP). When `y` customers are seated sequentially with concentration parameter `r`, each new customer either:
- Joins an existing table with probability `∝ (number at table)`
- Starts a new table with probability `∝ r`

The total number of tables `L ~ CRT(y, r)` has PMF:
```
P(L = l) = Γ(r) / Γ(y + r) * |s(y, l)| * r^l
```

where `|s(y, l)|` are unsigned Stirling numbers of the first kind.

### Direct Sampling Algorithm

```cpp
// Sample L ~ CRT(y, r)
// O(y) algorithm via sequential Bernoulli trials
int sample_crt(int y, double r) {
  if (y == 0) return 0;
  if (r <= 0) return 0;

  int L = 0;
  for (int i = 1; i <= y; i++) {
    // Customer i starts new table with this probability
    double p_new = r / (r + static_cast<double>(i) - 1.0);
    if (R::runif(0.0, 1.0) < p_new) {
      L++;
    }
  }
  return L;
}

// Vectorized version for efficiency
Rcpp::IntegerVector sample_crt_vec(Rcpp::IntegerVector y, double r) {
  int n = y.size();
  Rcpp::IntegerVector L(n);

  for (int j = 0; j < n; j++) {
    L[j] = sample_crt(y[j], r);
  }
  return L;
}
```

### Properties

- `E[L | y, r] = r * (ψ(y + r) - ψ(r))` where `ψ` is the digamma function
- `E[L | y, r] ≈ r * log(1 + y/r)` for large `y`
- When `y = 0`, `L = 0` deterministically
- When `r → ∞`, `L → y` (every customer gets own table)
- When `r → 0`, `L → 1` (all customers at one table)

### Why CRT for Negative Binomial?

The NB(r, p) distribution can be represented as:
```
Y | λ ~ Poisson(λ)
λ ~ Gamma(r, p/(1-p))
```

The key identity (Zhou et al., 2012):
```
(1 - p)^r = E[p^L]   where L ~ CRT(Y, r)
```

This allows conjugate Gamma updates for `r`:
```
r | Y, L ~ Gamma(a + Σ L_i, b - Σ log(1 - p_i))
```

Combined with PG for the `p_i` updates, this gives a complete conjugate Gibbs sampler.
