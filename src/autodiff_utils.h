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
// Inverse links of the linear predictors
// ============================================================================

// The logit of the probability the link maps eta to. Under the logit link
// that is eta itself, and the expression stays the one the canonical model
// has always differentiated.
template<typename T>
inline T prob_logit(const T& eta, ratiod_link::LinkType link) {
    if (link == ratiod_link::LinkType::LOGIT) return eta;
    const ratiod_link::ValueDeriv vd = ratiod_link::prob_logit_vd(get_value(eta), link);
    return lift_unary(eta, vd.value, vd.deriv);
}

// The mean the link maps eta to.
template<typename T>
inline T link_mean(const T& eta, ratiod_link::LinkType link) {
    if (link == ratiod_link::LinkType::LOG) return safe_exp(eta);
    const ratiod_link::ValueDeriv vd = ratiod_link::link_mean_vd(get_value(eta), link);
    return lift_unary(eta, vd.value, vd.deriv);
}

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
