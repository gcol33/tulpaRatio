# Continuation Prompt: GP Heisenbug Debug

Copy this to start a new session:

---

## Context

numdenom package - GP spatial heisenbug debugging

Branch: `feature/gp-heisenbug`

## Problem

GP spatial models (`spatial_gp()`) crash intermittently:
- **Working**: iter ≤ 10 for any n (10-100)
- **Crashing**: iter ≥ 50 crashes intermittently (random-state dependent)

Affected rows in gradient_methods.md: 7, 8, 18, 27, 38 (all GP configs)

## Symptoms

1. Segfault during HMC sampling (not initialization)
2. Crash is random-state dependent (heisenbug)
3. Short chains work, longer chains crash
4. Currently using numerical gradients (0.6x Stan speed)

## Key Files

- `src/hmc_gp.cpp` - NNGP log-likelihood (double version)
- `src/hmc_gp_autodiff.h` - Templated GP for autodiff (crashes)
- `src/hmc_sampler.cpp:1635` - Gradient dispatch (GP → numerical fallback)
- `test_gp_final.R` - Working test case (iter=10)
- `PLAN_GP_HEISENBUG.md` - Debug plan with hypotheses

## Hypotheses

1. **Memory corruption in NNGP**: Complex index arithmetic in neighbor loops
2. **Numerical instability**: Cholesky on near-singular matrices
3. **Stack overflow**: Deep allocation in HMC trajectory
4. **Race condition**: If OpenMP enabled

## Next Steps

1. Build with AddressSanitizer: `PKG_CXXFLAGS="-fsanitize=address" R CMD INSTALL .`
2. Run `test_gp_final.R` with iter=50 to trigger crash
3. Identify crash location from ASan output
4. Fix based on findings

## Commands

```bash
git checkout feature/gp-heisenbug
git pull origin feature/gp-heisenbug
```

## Goal

Make GP models stable for iter=400+ so they can be validated like other configs (target: >1x Stan speed with autodiff gradients).

---
