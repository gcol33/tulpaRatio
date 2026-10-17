# GP Optimization Benchmarks

Tracking actual performance gains from speedup.md phases.

## Test Configuration
- **N:** 100 observations
- **nn:** 10 nearest neighbors (GP); 5 local + 10 regional (MSGP)
- **iter:** 300 (warmup: 100)
- **chains:** 1
- **Model:** binomial

## Current Results (2026-01-19)

| Model | Time (sec) | Samples/sec | Gradient Mode | Notes |
|-------|-----------|-------------|---------------|-------|
| Simple (no spatial) | 4.87 | 41.10 | H (hand-coded) | Fast |
| GP (autodiff) | 75.28 | 2.66 | A (autodiff) | Baseline |
| GP (hand-coded) | **13.87** | **14.42** | H (hand-coded) | **5.4x speedup!** |
| MSGP | 22.32 | 8.96 | H (hand-coded) | 3.4x faster than GP autodiff |

## Investigation Results (2026-01-19)

### Phase 1: Vector/Memory Optimizations
| Optimization | Expected | Actual | Notes |
|-------------|----------|--------|-------|
| 1.1 Preallocate vectors | +25% | 0% | Var constructor creates tape nodes regardless |
| 1.2 Replace pow | +5% | N/A | Already done via cached distances |
| 1.3 Cache distances | +10% | +1.5% | Minimal impact |

### Phase 2: Autodiff Infrastructure
| Optimization | Expected | Actual | Notes |
|-------------|----------|--------|-------|
| 2.1 Arena allocator | +25% | 0% | Var allocation not bottleneck |
| 2.2 Static dispatch | +35% | -40% | Made things SLOWER |

## Root Cause Analysis

The GP autodiff is slow because:
1. Every `Var(0.0)` creates a tape node, even constants
2. NNGP makes N × (Cholesky + solves) = thousands of tape nodes per gradient
3. Each Cholesky creates ~n² nodes for zeros that get overwritten
4. Backward pass iterates through all nodes with std::function calls

## Conclusion

**Autodiff infrastructure optimizations don't help GP models.**

The only effective solution is **hand-coded GP gradients** (Phase 5.1):
- MSGP already has hand-coded gradients: 22 sec vs 75 sec for autodiff
- ✅ **GP hand-coded gradients implemented**: 75 sec → 14 sec (**5.4x speedup**)
- Exceeds the expected ~3x improvement

## Implementation Details

Phase 5.1 implemented via `compute_gradient_gp_handcoded()`:
1. Uses `gp_nngp_gradients()` for analytical NNGP prior gradients
2. Hand-coded likelihood gradients for Binomial, NegBin, Poisson-Gamma
3. Avoids all autodiff tape overhead
