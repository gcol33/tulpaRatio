# GP Spatial Crash Investigation

## Current Status: KNOWN LIMITATION

GP models have an intermittent crash issue that's under investigation. Short chains (iter ≤ 10) work reliably; longer chains may crash depending on random state.

**Recommendation**: For GP models, use short chains with multiple restarts, or use ICAR/BYM2 spatial effects which are fully stable.

## Original Issue
Segmentation fault when using `spatial_gp()` in ratiod models.

## Affected Configs
- Row 7: poisson_gamma + GP
- Row 8: poisson_gamma + MSGP
- Row 18: poisson_gamma + GP + RW1
- Row 27: negbin_negbin + GP
- Row 38: binomial + GP

## Resolution

**Implemented Option B: Numerical gradients for GP**

GP models now use `compute_gradient_numerical()` instead of `compute_gradient_autodiff()`.
This gives correct results but slower performance (~0.6x vs Stan instead of ~3-5x).

The templated autodiff GP code is preserved in `hmc_gp_autodiff.h` for future optimization.

## Autodiff GP Heisenbug (for future work)

The templated `gp_nngp_log_lik_t<T>` function was implemented but crashes with larger n:
- n=10 (17 params): Works fine
- n=12+ (20+ params): Segfault during HMC

### Observations:
1. Works with T=double (called via numerical gradients)
2. Crashes with T=ad::Var (autodiff)
3. The crash is not in the NNGP function itself (debug output shows function completing)
4. The crash appears during HMC trajectory computation (backward pass or memory access)

### Suspected Causes:
1. **Autodiff tape size**: With n=30 and nn=5, each gradient call creates thousands of tape nodes
2. **Memory access patterns**: The nested loops in NNGP interact poorly with compiler optimization
3. **Heisenbug characteristics**: The original double version also required `-O0` to avoid crashes

### Future Optimization Path:
1. Investigate tape memory management
2. Consider chunked tape allocation
3. Profile memory access patterns during backward pass
4. Consider alternative autodiff implementation for GP-specific operations

## Files Modified
- `src/hmc_gp_autodiff.h`: Templated GP functions (preserved for future)
- `src/log_post_impl.h`: Added GP parameter extraction and priors
- `src/hmc_sampler.cpp`: Added numerical gradient fallback for GP

## Investigation Log

### 2025-01-14: O2 Optimization Investigation

After O2 optimization was enabled:
- Short chains (iter=10) work reliably for n=10 to n=100
- Longer chains (iter=50+) crash intermittently
- Crash is random-state dependent (heisenbug)

Working test:
```r
# test_gp_final.R - runs successfully with iter=10
for (n in c(10, 20, 30, 50, 100)) {
  fit <- ratiod(..., spatial = spatial_gp(~ lon + lat, nn = 5),
                iter = 10, warmup = 5, chains = 1)  # OK
}
```

Crashing configuration:
```r
fit <- ratiod(..., spatial = spatial_gp(~ lon + lat, nn = 5),
              iter = 50, warmup = 25, chains = 1)  # Segfault (intermittent)
```

**Root cause**: Not yet identified. Likely memory corruption during HMC trajectory
computation with longer chains. The crash is not during initialization but during
sampling iterations.

**Priority**: Low - GP is a specialized use case. ICAR/BYM2 cover most spatial needs
and are fully stable with excellent performance (13-22x faster than Stan).

---

### Test 1: Minimal Reproduction
