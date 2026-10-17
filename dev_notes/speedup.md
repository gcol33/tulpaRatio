# Autodiff Speedup Plan

## Design Principle: N/A/H Modes

All three gradient methods remain available and switchable:

| Mode | Description | Use Case |
|------|-------------|----------|
| **N** | Numerical (finite diff) | Fallback, debugging, new model validation |
| **A** | Autodiff | Reference implementation, moderate speed |
| **H** | Hand-coded | Production default when available, max speed |

**H does NOT replace A** - it becomes the better default. A remains for:
- Correctness validation (`gradient_check(mode="A") vs mode="H"`)
- Debugging when H has issues
- Rapid prototyping of new model features

```r
# User can switch modes:
ratiod(..., gradient_mode = "auto")  # Default: use H if available, else A, else N
ratiod(..., gradient_mode = "A")     # Force autodiff
ratiod(..., gradient_mode = "N")     # Force numerical (slow but safe)
```

---

## Current State

GP-based models (rows 7, 8, 18, 27, 38) achieve only ~3x speedup vs Stan, while other models achieve 5-27x. Root cause: autodiff overhead in matrix operations.

| Model Type | Current Speedup | Target |
|------------|-----------------|--------|
| GP/MSGP | ~3x | ~10x |
| HSGP | ~25x | (already optimized) |
| Other autodiff | 5-27x | (acceptable) |

---

## Phase 1: Low-Hanging Fruit ❌ INEFFECTIVE FOR AUTODIFF

**Investigation (2026-01-19):** These optimizations don't help GP autodiff because the bottleneck is tape node creation, not memory allocation. For autodiff Vars, every `Var(0.0)` creates a tape node regardless of vector reuse.

**Baseline:** ~75 sec for N=100, nn=10, iter=300 binomial GP

### 1.1 Preallocate work vectors outside loop ❌ NO EFFECT

**Tested:** Vector preallocation inside loop vs outside loop
**Result:** No measurable improvement for autodiff path
**Reason:** `std::vector<Var>.resize()` still creates tape nodes via Var constructor

### 1.2 Replace `std::pow(x, 2)` with `x * x` ❌ ALREADY DONE IN 1.3

**Status:** Distance computation already uses cached values from Phase 1.3

### 1.3 Cache neighbor distances ✅ DONE (+1.5%)

**Files:** `R/spatial.R`, `R/backend_hmc.R`, `src/hmc_gp.h`, `src/hmc_gp_autodiff.h`, `src/hmc_sampler.cpp`

**Problem:** Distance matrix among neighbors is recomputed every gradient evaluation, but coordinates are static.

**Fix:** Precompute `nn_neighbor_dist[i, j1, j2]` for each observation during data setup in R, pass to C++.
- R computes N × k × k array in `compute_nngp_neighbors()`
- Flattened to 1D vector with `aperm(arr, c(3, 2, 1))` for C++ row-major access
- C++ accesses as `gp_data.nn_neighbor_dist[i * nn * nn + j1 * nn + j2]`

**Effort:** 2 hours | **Impact:** +1.5% actual (see speedup_benchmarks.md)

---

## Phase 2: Autodiff Infrastructure ❌ UNLIKELY TO HELP

**Investigation (2026-01-19):** Tested arena allocation and static dispatch. Both made performance WORSE or had no effect. The bottleneck is the sheer number of tape nodes created (one per Var operation), not the allocation strategy.

### 2.1 Arena allocator for tape nodes ❌ TESTED - NO IMPROVEMENT

**Tested:** Pre-reserving nodes vector with 65K capacity
**Result:** No improvement; resize() still creates Var objects
**Reason:** For Var type, vector operations invoke constructors that add tape nodes

### 2.2 Replace `std::function` with static dispatch ❌ TESTED - SLOWER

**File:** `src/autodiff.h`

**Problem:** Each node stores `std::function<void()> backward` which involves:
- Heap allocation for lambda capture
- Virtual dispatch overhead
- Poor inlining

**Fix:** Use tagged union or template-based approach:
```cpp
enum class OpType { ADD, MUL, DIV, EXP, LOG, SQRT, ... };

struct Node {
    double value;
    double adjoint;
    OpType op;
    size_t arg1, arg2;  // indices to parent nodes
};

// Backward pass uses switch instead of virtual call
void backward_node(Node& n, std::vector<Node>& nodes) {
    switch (n.op) {
        case OpType::ADD:
            nodes[n.arg1].adjoint += n.adjoint;
            nodes[n.arg2].adjoint += n.adjoint;
            break;
        case OpType::MUL:
            nodes[n.arg1].adjoint += n.adjoint * nodes[n.arg2].value;
            nodes[n.arg2].adjoint += n.adjoint * nodes[n.arg1].value;
            break;
        // ...
    }
}
```

**Effort:** 8 hours | **Impact:** ~30-40% speedup

---

### 2.3 Expression templates (optional, complex)

**Problem:** `a + b * c` creates intermediate `Var` objects.

**Fix:** Use CRTP-based expression templates (like Eigen does):
```cpp
template<typename L, typename R>
struct AddExpr {
    const L& l;
    const R& r;
    double eval() const { return l.eval() + r.eval(); }
};

template<typename L, typename R>
AddExpr<L, R> operator+(const L& l, const R& r) {
    return AddExpr<L, R>{l, r};
}
```

**Effort:** 16+ hours | **Impact:** ~20% speedup | **Risk:** High complexity

---

## Phase 3: GP-Specific Optimizations (Expected: ~8x → ~12x)

### 3.1 Specialized Cholesky gradient

**Problem:** Templated `cholesky_decompose_t` is scalar O(n³) loops. Eigen's LLT uses BLAS but we can't differentiate through it.

**Fix:** Implement analytical gradient for Cholesky:
```cpp
// Given L = chol(A), compute dA from dL
// dA = L^{-T} * tril(L^T * dL) * L^{-1}
// This is O(n³) but with BLAS ops, not scalar loops

void cholesky_backward(
    const Eigen::MatrixXd& L,      // Forward Cholesky factor
    const Eigen::MatrixXd& dL,     // Adjoint of L
    Eigen::MatrixXd& dA            // Output: adjoint of A
) {
    // Efficient reverse-mode differentiation
    Eigen::MatrixXd S = L.triangularView<Eigen::Lower>().solve(dL);
    S = S.triangularView<Eigen::Lower>();
    dA = L.triangularView<Eigen::Lower>().transpose().solve(S);
    dA = 0.5 * (dA + dA.transpose());
}
```

**Effort:** 6 hours | **Impact:** ~40-50% speedup for GP

---

### 3.2 Fused NNGP kernel

**Problem:** Current code makes many small Cholesky calls (one per observation).

**Fix:** Batch operations where possible, use blocked algorithms:
```cpp
// Process observations in blocks of 32
constexpr int BLOCK_SIZE = 32;
for (int block = 0; block < N; block += BLOCK_SIZE) {
    // Process block with better cache utilization
}
```

**Effort:** 8 hours | **Impact:** ~15-20% speedup

---

### 3.3 Hand-coded GP gradients (nuclear option)

**Problem:** All autodiff overhead.

**Fix:** Like HSGP, derive and implement analytical gradients for full NNGP likelihood.

The NNGP log-likelihood gradient w.r.t. parameters can be derived analytically:
- ∂LL/∂w[i] involves only neighbors of i
- ∂LL/∂σ² and ∂LL/∂φ involve sums over all observations

**Effort:** 16+ hours | **Impact:** ~3x speedup (match HSGP) | **Risk:** Complex derivation

---

## Phase 4: Parallelization (Expected: additional ~2x on multicore)

### 4.1 OpenMP for observation loop

**File:** `src/hmc_gp_autodiff.h`

**Problem:** N observations processed sequentially.

**Fix:**
```cpp
#pragma omp parallel for reduction(+:log_lik) schedule(dynamic)
for (int i = 1; i < N; i++) {
    // ... (need thread-local work vectors)
}
```

**Caveat:** Requires thread-local tape or tape-free gradient accumulation.

**Effort:** 4 hours | **Impact:** ~1.5-2x on 4+ cores

---

## Implementation Order

| Step | Phase | Expected Gain | Cumulative | Effort |
|------|-------|---------------|------------|--------|
| 1 | 1.1 Preallocate vectors | +25% | ~3.8x | 1h |
| 2 | 1.2 Replace pow | +5% | ~4.0x | 15m |
| 3 | 1.3 Cache distances | +10% | ~4.4x | 2h |
| 4 | 2.1 Arena allocator | +25% | ~5.5x | 4h |
| 5 | 2.2 Static dispatch | +30% | ~7.2x | 8h |
| 6 | 3.1 Cholesky gradient | +40% | ~10x | 6h |
| 7 | 4.1 OpenMP | +50% | ~15x | 4h |

**Total effort:** ~25 hours for ~5x improvement (3x → 15x)

---

## Validation

After each phase:
1. Run `devtools::test()` - ensure correctness
2. Run `benchmark_verify.R` - measure speedup
3. Compare gradients: numerical vs analytical (max abs diff < 1e-5)

---

## Phase 5: Hand-Coded Gradients (Final Goal: ~25x like HSGP)

**Prerequisite:** Complete Phases 1-4 first. Optimized autodiff benefits ALL models, not just GP.

**Key principle:** H is added alongside A, not replacing it.

```cpp
// In compute_gradient():
void compute_gradient(..., GradientMode mode) {
    if (mode == GradientMode::AUTO) {
        // Priority: H > A > N
        if (has_handcoded_gradient(model_config)) {
            mode = GradientMode::H;
        } else if (can_use_autodiff(model_config)) {
            mode = GradientMode::A;
        } else {
            mode = GradientMode::N;
        }
    }

    switch (mode) {
        case GradientMode::H: compute_gradient_handcoded(...); break;
        case GradientMode::A: compute_gradient_autodiff(...); break;
        case GradientMode::N: compute_gradient_numerical(...); break;
    }
}
```

### 5.1 Hand-code GP gradients

Once autodiff is mature, derive analytical gradients for NNGP:

```
∂LL/∂w[i] = -resid[i]/var[i] + Σ_j (contribution from obs j where i is neighbor)
∂LL/∂σ² = Σ_i [ -1/(2σ²) + resid[i]²/(2σ⁴) + covariance terms ]
∂LL/∂φ = Σ_i [ ∂cov/∂φ terms through conditional mean/var ]
```

**Effort:** 16 hours | **Impact:** Match HSGP (~25x)

### 5.2 Hand-code MSGP gradients

Similar to GP but with two scales (local + regional).

**Effort:** 12 hours | **Impact:** ~25x

### 5.3 Hand-code GP+temporal gradients

Combine GP spatial gradient with temporal gradient (already have `hmc_temporal_autodiff.h`).

**Effort:** 8 hours | **Impact:** ~25x

---

## Strategy Summary

```
Phase 1-2: Optimize autodiff infrastructure (benefits ALL autodiff models)
     ↓
Phase 3-4: GP-specific optimizations + parallelization
     ↓
Phase 5: Hand-coded gradients for GP/MSGP (final ~25x target)
```

**Why this order?**
1. Autodiff improvements benefit 54 model configurations, not just GP
2. Hand-coded gradients are error-prone; optimized autodiff serves as reference
3. Each phase provides measurable speedup and validates the approach

---

## References

- Stan Math autodiff: https://github.com/stan-dev/math
- Eigen LLT differentiation: https://arxiv.org/abs/1602.07527
- NNGP: Datta et al. (2016) "Hierarchical Nearest-Neighbor Gaussian Process Models"
