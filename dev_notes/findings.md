# numdenom Known Issues and Findings

## RESOLVED: Spatial GP and HSGP Segfaults

**Status**: FIXED (2026-02-01)

### Original Symptoms
- `spatial_gp()` crashed with segmentation fault (exit code 139) during HMC sampling
- `spatial_hsgp()` crashed with segmentation fault (exit code 139) during HMC sampling
- Both crashed early in sampling (before any iterations complete)

### Root Cause
The `nn_neighbor_dist` 3D array (N × nn × nn) was flattened incorrectly when passing from R to C++.

**The bug**: R uses column-major storage while C++ expects row-major access. The original code used `aperm(c(3, 2, 1))` which did NOT produce the correct memory layout.

- C++ accesses as: `i * nn * nn + j1 * nn + j2` (i slowest, j2 fastest)
- The `aperm(c(3, 2, 1))` permutation produced a completely different ordering
- For element `[i=2, j1=1, j2=3]`: R index was 622, but C++ expected index 213

### The Fix
In `R/backend_hmc.R` (lines 2839-2855), replaced the buggy `aperm()` call with an explicit triple loop:

```r
# Old (buggy):
nn_neighbor_dist_flat <- as.vector(aperm(nn_info$nn_neighbor_dist, c(3, 2, 1)))

# New (correct):
nn <- gp$nn
N_gp <- nrow(nn_info$nn_idx)
nn_neighbor_dist_flat <- numeric(N_gp * nn * nn)
for (i in seq_len(N_gp)) {
  for (j1 in seq_len(nn)) {
    for (j2 in seq_len(nn)) {
      cpp_idx <- (i - 1L) * nn * nn + (j1 - 1L) * nn + (j2 - 1L) + 1L
      nn_neighbor_dist_flat[cpp_idx] <- nn_info$nn_neighbor_dist[i, j1, j2]
    }
  }
}
```

### Verification
Both `spatial_gp()` and `spatial_hsgp()` now work correctly:
- Small test (N=30, 50 iterations): SUCCESS
- Larger test (N=80, 500 iterations): Completed in ~150s with accurate parameter recovery

### Affected Model Configurations (Now Working)
- Row 7: negbin_negbin + spatial_gp ✓
- Row 8: negbin_negbin + spatial_hsgp ✓
- Row 27: negbin_negbin + spatial_gp + slopes ✓
- Row 37: binomial + spatial_gp ✓
- Row 38: binomial + spatial_hsgp ✓
- (and all equivalent rows for poisson_gamma)

### Note on Temporal GP
`temporal_gp()` was always working correctly. The issue was specific to **spatial** GP models due to the NNGP neighbor distance array flattening.

---

# Temporal Multiscale Gradient Implementation - Findings

## Problem Statement

`temporal_multiscale()` models run at ~700s while other temporal methods (RW1, AR1, GP) run at ~6-10s. Investigation revealed that temporal_multiscale lacks efficient gradient implementations (A_t and H modes), falling back to forward autodiff (A mode) which is O(p×N).

## Files Created

### 1. `src/hmc_temporal_multiscale_autodiff.h`
Templated functions for A_t (tape autodiff) support:
- `rw1_log_lik<T>()` - RW1 prior log-likelihood (cyclic and non-cyclic)
- `rw2_log_lik<T>()` - RW2 prior log-likelihood
- `ar1_log_lik<T>()` - AR1 prior log-likelihood with stationary variance
- `iid_log_lik<T>()` - IID prior log-likelihood
- `multiscale_temporal_log_lik<T>()` - Combined trend + seasonal + short-term
- `log_prior_sigma2_temporal_pc<T>()` - PC prior for variance parameters
- `log_prior_rho<T>()` - Beta prior for AR1 correlation
- `compute_temporal_eta<T>()` - Compute total temporal effect per observation

### 2. `src/hmc_multiscale_temporal_grad.h`
Hand-coded analytical gradients for H mode (O(n) complexity):
- `rw1_grad_phi()` - Gradient w.r.t. RW1 effects
- `rw1_grad_log_sigma2()` - Gradient w.r.t. log(sigma2) for RW1
- `rw1_cyclic_grad_phi()` - Cyclic RW1 gradients (for seasonal)
- `rw2_grad_phi()` - Gradient w.r.t. RW2 effects
- `rw2_grad_log_sigma2()` - Gradient w.r.t. log(sigma2) for RW2
- `ar1_grad_phi()` - Gradient w.r.t. AR1 effects
- `ar1_grad_log_sigma2()` - Gradient w.r.t. log(sigma2) for AR1
- `ar1_grad_logit_rho()` - Gradient w.r.t. logit(rho)
- `iid_grad_phi()` - Gradient w.r.t. IID effects
- `pc_prior_grad_log_sigma2()` - PC prior gradient
- `multiscale_temporal_prior_gradients()` - Combined gradient computation

### 3. `src/hmc_sampler.cpp` modifications
- Added `#include "hmc_multiscale_temporal_grad.h"`
- Added `compute_gradient_multiscale_temporal_handcoded()` function (~300 lines)
- Fixed `digamma()` calls to use `R::digamma()` for scalar values

## Issues Encountered

### Issue 1: digamma() Function Signature
**Error**: `no matching function for call to 'digamma(double&)'`
**Cause**: Rcpp's `digamma()` expects a vector, not scalar
**Fix**: Changed to `R::digamma(scalar_value)` at lines 4637, 4650, 4653

### Issue 2: H Gradient Produces NaN Values
**Symptom**: All posterior samples are NaN when H dispatch is active
**Location**: `compute_gradient_multiscale_temporal_handcoded()` in hmc_sampler.cpp
**Status**: FIXED (2026-02-27)

**Root cause**: Four bugs in `src/hmc_multiscale_temporal_grad.h` and `src/hmc_sampler.cpp`:

1. **`ar1_grad_logit_rho()` line 200**: Wrong formula for stationary variance contribution.
   - Before: `rho / one_m_rho2 * phi[0]^2` (missing normalization term, wrong denominator)
   - After: `-rho / one_m_rho2 + rho * inv_sigma2 * phi[0]^2` (correct: normalization + quadratic)
   - This was the primary NaN source: when rho → ±1, `1/(1-rho^2)` explodes, and the missing
     normalization term meant no cancellation occurred.

2. **All `*_grad_log_sigma2()` functions**: Extra `* sigma2` factor (Jacobian applied twice).
   - The comment correctly derived `d/d(log_sigma2) = -0.5*(n-1) + 0.5*quad/sigma2`
   - But then the return statement multiplied by sigma2 AGAIN: `return (correct) * sigma2`
   - Fixed by removing the spurious `* sigma2` from all five functions:
     `rw1_grad_log_sigma2`, `rw1_cyclic_grad_log_sigma2`, `rw2_grad_log_sigma2`,
     `ar1_grad_log_sigma2`, `iid_grad_log_sigma2`

3. **Missing Jacobian `+1`** for log_sigma2 parameters in hmc_sampler.cpp.
   - `compute_log_post` includes `+= log_sigma2` as Jacobian correction (d/d(log_s2) = 1)
   - The H gradient omitted this +1 for all three sigma2 parameters (trend, seasonal, short)
   - Confirmed by comparing with the `log_tau_spatial` gradient which correctly has `+ 1.0`

4. **Prior gradient for logit_rho** off by factor of 2.
   - `compute_log_post` adds Beta(2,2) prior `log(u)+log(1-u)` PLUS Jacobian `log(u)+log(1-u)`
   - Total: `2*[log(u)+log(1-u)]`, gradient = `2*(1-2u)`
   - H gradient had `1-2u` (factor of 2 missing)

**Verification**: After fix, H and A modes produce identical draws (max diff = 0.000000)
for all three component combinations tested (RW1+seasonal+AR1, RW1-only, RW2+IID).

### Issue 3: A_t/A_r/A Mode Segfaults for Multiscale Temporal
**Symptom**: Segmentation fault when running with A_t mode
**Location**: `log_post_impl.h` — missing multiscale temporal support
**Status**: FIXED (2026-02-27)

**Root cause**: `log_post_impl.h` (the templated log-posterior used by all autodiff modes) had NO
multiscale temporal code at all. It handled regular temporal (`has_temporal`) but had zero handling
of `has_multiscale_temporal`. This meant:
1. No extraction of trend/seasonal/short_term parameters from theta vector
2. No priors added for multiscale temporal effects or hyperparameters
3. No temporal effects added to eta_num/eta_denom in the likelihood loop
4. The autodiff modes computed a log-posterior that ignored all multiscale temporal structure

Note: The prior fields DO exist in `ModelData` (lines 191-199 of `hmc_sampler.h`) — the earlier
diagnosis that "fields don't exist" was incorrect.

**Fix**: Added complete multiscale temporal support to `log_post_impl.h`:
1. Added `#include "hmc_temporal_multiscale_autodiff.h"`
2. Added parameter extraction + PC priors + Beta(2,2) rho prior + GMRF log-likelihood block
3. Added `compute_temporal_eta()` precomputation and eta contribution in observation loop
4. Follows exact same structure as reference `compute_log_post()` in `hmc_sampler.cpp`

**Verification**: All 4 gradient modes produce identical results (max diff = 0.000000) across 3
component combinations (RW1+seasonal+AR1, RW1-only, RW2+IID). Test: `benchmarks/test_ms_temporal_autodiff.R`

## Current State

- All 4 gradient modes (H, A_t, A_r, A) work correctly for multiscale temporal
- Gradient verification: All modes produce identical draws (max diff = 0.000000)
- At N=500, iter=500: ~120s for all gradient modes (gradient cost is negligible; bottleneck is
  NUTS with epsilon=0.0012 requiring ~265k leapfrog steps due to difficult posterior geometry)
- H mode provides no significant speedup over A at this model size because per-gradient cost
  is microseconds; improvement would show at N>5000 where O(p×N) becomes material

## Testing Approach

To debug H gradients:
```r
# Test with single iteration to capture gradient values
fit <- ratiod(
  y_num | y_denom ~ x + temporal_multiscale(time),
  data = df,
  family = ratiod_negbin_negbin(),
  iter = 10, warmup = 5, chains = 1,
  gradient_mode = "H",
  verbose = TRUE
)
```

Compare against:
```r
fit_A <- ratiod(..., gradient_mode = "A")
```

## References

- Existing temporal implementations: `src/hmc_temporal.h`, `src/hmc_temporal_grad.h`
- Multiscale data structures: `src/hmc_temporal_multiscale.h`
- Parameter layout: `ParamLayout` struct in `src/hmc_sampler.h`

---

# Lognormal + Random Effects Validation Mismatch (Row 99)

## Status: ROOT CAUSE IDENTIFIED - HMC Mixing Issue (Not a Model Bug)

### Problem Statement

Lognormal family models with random effects (Row 99 in gradient_methods.md) show apparent posterior mismatch against Stan reference implementations.

### Root Cause Analysis (2026-02-06)

**The model is correct. The issue is HMC sampling efficiency.**

#### Evidence

Comparing ESS (Effective Sample Size) from 8000 draws (4 chains × 2000 post-warmup):

| Parameter | numdenom ESS | Stan ESS | Ratio |
|-----------|--------------|----------|-------|
| sigma_num | 31 | 10315 | 333x worse |
| sigma_denom | 41 | 9075 | 221x worse |
| sigma_re | 12 | 1935 | 161x worse |

**numdenom HMC is essentially stuck** with ESS of 12-41 from 8000 draws. R-hat values are 1.02-1.04 (should be <1.01).

In contrast, **negbin_negbin + RE** works perfectly:
- phi_num: ESS=1998, R-hat=1.001
- phi_denom: ESS=1988, R-hat=1.000
- sigma_re: ESS=1253, R-hat=1.001

#### Why This Happens

1. **Lognormal uses A-mode (forward autodiff)** while negbin uses H-mode (hand-coded gradients)
   - Both compute correct gradients mathematically
   - But A-mode is ~40x slower (173s vs 4s for Stan equivalent)

2. **Slow gradients → poor HMC adaptation**
   - Fewer leapfrog evaluations during warmup
   - Poor mass matrix estimation
   - Inadequate step size tuning

3. **Lognormal + RE has challenging geometry**
   - Funnel-shaped posterior around sigma_re
   - Requires precise mass matrix scaling to navigate efficiently

#### Validation: Row 98 Passes

Base lognormal **without RE** passes validation with longer chains:
```
sigma_num: nd=0.47601 stan=0.47572 (0.76 SE, 0.06% bias) => PASS
sigma_denom: nd=0.52392 stan=0.52431 (0.93 SE, -0.07% bias) => PASS
```

This confirms the lognormal likelihood and priors are correctly implemented.

### Fix Applied (2026-02-07)

**H-mode gradients for lognormal are NOW ENABLED and CORRECT.**

The original "mismatch" was a testing artifact, not a gradient bug:
- When comparing A-mode vs H-mode with the same seed (`set.seed(123)` before each fit), draws are **exactly identical**
- The apparent mismatch occurred because original tests didn't reset the seed between fits
- Different random streams → different MCMC chains → apparent "mismatch"

**Verification:**
```r
# Same seed before each fit → identical results
set.seed(123); fit_A <- ratiod(..., gradient_mode = "A")
set.seed(123); fit_H <- ratiod(..., gradient_mode = "H")
all.equal(as.matrix(fit_A$draws), as.matrix(fit_H$draws))  # TRUE
```

**Change made:**
- `hmc_sampler.cpp` line 1707: Added `LOGNORMAL` to `can_use_analytical_gradient()`
- Now lognormal uses H-mode by default (O(N) complexity, ~same speed as other families)

**Row 99 (lognormal + RE) mixing: FIXED (2026-02-27)**

Root cause was diagonal mass matrix. The sigma_re ↔ z funnel geometry requires dense mass to
capture the correlation. Fixed by adding `layout.has_re` to the dense metric trigger in
`hmc_sampler.cpp` line 8566.

Before (diagonal): ESS 12-41 from 8000 draws, Rhat 1.02-1.04
After (dense):     ESS 1944-2811 from 4000 draws, Rhat 1.005 — **57-176x improvement**

Negbin+RE unaffected (ESS 2273+, Rhat 1.001). Dense mass is cheap for typical RE models (p < 200).

### Benchmark Scripts

- `benchmarks/bench_lognormal_validation.R`: Row 98/99 validation
- `benchmarks/stan/joint_ln_re.stan`: Stan reference model

### Previous Fix (Still Valid)

Non-centered RE handling was added (2026-02-02):
1. **hmc_sampler.cpp**: Reads `re_parameterization` from R
2. **log_post_impl.h**: RE extraction handles non-centered: `re = sigma * z`
3. **log_post_impl.h**: RE prior uses τ=1 for non-centered

This fix was necessary but insufficient - the mixing issue remains.

---

# Repo Audit: Parameterization & Prior Consistency (2026-02-02)

## Summary

Audit to find similar issues to the RE parameterization mismatch. The key question: do `hmc_sampler.cpp` (H gradients) and `log_post_impl.h` (A/A_t autodiff) compute the same priors?

## Findings

### 1. Single-Term Random Effects (Simple RE)
**Status**: FIXED

- `log_post_impl.h` now handles both centered and non-centered parameterization
- `hmc_sampler.cpp` reads `re_parameterization` from R and uses consistent prior

### 2. Multi-Term RE with Slopes
**Status**: POTENTIAL ISSUE

- **hmc_sampler.cpp**: Lines 1943-2040 have hand-coded gradients for correlated slopes
- **log_post_impl.h**: Lines 63-86 handle only single `sigma_re` and `re[]` vector
- **Gap**: Multi-term RE with correlation (LKJ prior, Cholesky factor) NOT in log_post_impl.h
- **Impact**: Models with random slopes fall back to A mode or use H with different prior formulation

### 3. GP Spatial (spatial_gp)
**Status**: POTENTIAL ISSUE

- **hmc_sampler.cpp**: Lines 2150-2350 handle GP priors (PC prior on sigma2, uniform on phi)
- **log_post_impl.h**: GP support exists but may have different prior formulation
- **Note**: GP was fixed for segfaults; prior consistency not verified

### 4. HSGP Spatial (spatial_hsgp)
**Status**: POTENTIAL ISSUE

- **hmc_sampler.cpp**: Lines 2400-2550 handle HSGP priors
- **log_post_impl.h**: HSGP case exists but sparse
- **Note**: HSGP shares GP fix but prior consistency not verified

### 5. Multiscale Temporal (temporal_multiscale)
**Status**: KNOWN ISSUE (documented above)

- **hmc_sampler.cpp**: Has multiscale gradients but produces NaN
- **log_post_impl.h**: Missing prior fields in ModelData struct
- **Impact**: Falls back to A mode, ~700s runtime

### 6. Spatiotemporal (spatiotemporal)
**Status**: POTENTIAL ISSUE

- **hmc_sampler.cpp**: Lines 3200-3400 handle spatiotemporal priors
- **log_post_impl.h**: Spatiotemporal support exists but sparse
- **Note**: Similar structure to multiscale temporal

### 7. TVC (Time-Varying Coefficients)
**Status**: PARTIAL

- **hmc_sampler.cpp**: Uses external `compute_gradient_tvc_handcoded()`
- **log_post_impl.h**: TVC partially handled via separate functions
- **Note**: May be consistent but needs verification

## Recommended Actions

1. ~~**Priority 1**: Fix Row 99 (lognormal + RE) - remaining sigma mismatch~~ DONE (dense mass matrix)
2. **Priority 2**: Verify GP/HSGP prior consistency (spatial models are working but not validated against Stan)
3. ~~**Priority 3**: Fix multiscale temporal A_t/H modes~~ DONE
4. **Priority 4**: Audit multi-term RE with slopes

## Testing Strategy

For each structure, compare:
```r
# Same model, different gradient modes
fit_A <- ratiod(..., gradient_mode = "A")
fit_H <- ratiod(..., gradient_mode = "H")

# Compare posterior means - should be identical
all.equal(colMeans(fit_A$draws), colMeans(fit_H$draws), tolerance = 0.01)
```
