// autodiff_utils.h
// Scalar-templated inverse links and dot products for ratio likelihoods and priors.

#ifndef RATIOD_AUTODIFF_UTILS_H
#define RATIOD_AUTODIFF_UTILS_H

#include <cmath>
#include <limits>
#include <vector>
#include <Rcpp.h>
#include "ratio_math.h"
#include "link_function.h"
#include <tulpa/count_math.h>
#include "ratio_margins.h"

namespace ratiod {
namespace math {
using namespace tulpa::math;

// ============================================================================
// Dot product that works with all types
// ============================================================================

template<typename T>
inline T dot_product(const std::vector<T>& x, const std::vector<T>& y) {
    T sum = T(0.0);
    for (size_t i = 0; i < x.size(); i++) {
        sum = sum + x[i] * y[i];
    }
    return sum;
}

} }
#endif
