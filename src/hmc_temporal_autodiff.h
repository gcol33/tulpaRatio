// hmc_temporal_autodiff.h
// Templated temporal functions for autodiff support
// Works with both double (for evaluation) and ad::Var (for gradients)

#ifndef RATIOD_HMC_TEMPORAL_AUTODIFF_H
#define RATIOD_HMC_TEMPORAL_AUTODIFF_H

#include <vector>
#include <cmath>
#include "hmc_temporal.h"
#include "autodiff_utils.h"

namespace ratiod_temporal {

using namespace ratiod::math;

// =============================================================================
// Templated RW1 quadratic form
// =============================================================================

// The pointer form is the implementation: callers that hold a strided slice of a
// flat coefficient vector (a spatiotemporal location's series, one TVC term's
// trajectory) can pass an offset instead of copying into a vector first.
template<typename T>
T rw1_quadratic_form_t(
    const T* phi,
    int T_len,
    bool cyclic
) {
    T quad = T(0.0);

    // Sum of squared differences
    for (int t = 0; t < T_len - 1; t++) {
        T diff = phi[t + 1] - phi[t];
        quad = quad + diff * diff;
    }

    // Cyclic: add edge from T to 1
    if (cyclic && T_len > 0) {
        T diff = phi[0] - phi[T_len - 1];
        quad = quad + diff * diff;
    }

    return quad;
}

// =============================================================================
// Templated RW2 quadratic form
// =============================================================================

template<typename T>
T rw1_quadratic_form_t(const std::vector<T>& phi, int T_len, bool cyclic) {
    return rw1_quadratic_form_t(phi.data(), T_len, cyclic);
}

template<typename T>
T rw2_quadratic_form_t(
    const T* phi,
    int T_len,
    bool cyclic
) {
    if (T_len < 3) return T(0.0);

    T quad = T(0.0);

    // Sum of squared second differences
    for (int t = 0; t < T_len - 2; t++) {
        T diff2 = phi[t] - T(2.0) * phi[t + 1] + phi[t + 2];
        quad = quad + diff2 * diff2;
    }

    // Cyclic: wrap around
    if (cyclic && T_len >= 3) {
        // t = T-2
        T diff2_1 = phi[T_len - 2] - T(2.0) * phi[T_len - 1] + phi[0];
        quad = quad + diff2_1 * diff2_1;
        // t = T-1
        T diff2_2 = phi[T_len - 1] - T(2.0) * phi[0] + phi[1];
        quad = quad + diff2_2 * diff2_2;
    }

    return quad;
}

// =============================================================================
// Templated AR1 log-density
// =============================================================================

template<typename T>
T rw2_quadratic_form_t(const std::vector<T>& phi, int T_len, bool cyclic) {
    return rw2_quadratic_form_t(phi.data(), T_len, cyclic);
}

}  // namespace ratiod_temporal

#endif  // RATIOD_HMC_TEMPORAL_AUTODIFF_H
