#ifndef TULPARATIO_RATIO_MATH_H
#define TULPARATIO_RATIO_MATH_H
#include <tulpa/ad_scalar_math.h>
#include <tulpa/scalar_priors.h>
#include <limits>
#include "link_function.h"
namespace ratiod { namespace math {
using namespace tulpa::math;
// ============================================================================
// Inverse links of the linear predictors
// ============================================================================

// The probability's logit equals eta under the logit link.
// Noncanonical links supply its value and first derivative.
// lift_unary propagates that derivative through the scalar type.
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


template<typename T>
inline T log1p_exp(const T& x) {
    if (get_value(x) > 0.0) return x + log1p_fn(safe_exp(-x));
    return log1p_fn(safe_exp(x));
}

template<typename T>
inline T log_inv_logit(const T& x) {
    return -log1p_exp(-x);
}

template<typename T>
inline T log1m_inv_logit(const T& x) {
    return -log1p_exp(x);
}

template<typename T>
inline T log_add_exp(const T& a, const T& b) {
    const bool a_hi = get_value(a) >= get_value(b);
    const T& hi = a_hi ? a : b;
    const T& lo = a_hi ? b : a;
    if (!std::isfinite(get_value(hi))) return T(get_value(hi));
    return hi + log1p_fn(safe_exp(lo - hi));
}

inline double pc_rate(double U, double alpha) {
    return -std::log(alpha) / U;
}

template<class T>
inline T pc_sigma_log_density(const T& sigma, double lambda) {
    return T(std::log(lambda)) - T(lambda) * sigma;
}

template<class T>
inline T pc_log_density_rate(const T& x, double slope, double lambda) {
    return pc_sigma_log_density(safe_exp(T(slope) * x), lambda)
         + T(std::log(std::abs(slope))) + T(slope) * x;
}

template<class T>
inline T pc_log_density(const T& x, double slope, double U, double alpha) {
    return pc_log_density_rate(x, slope, pc_rate(U, alpha));
}

inline double pc_scale_gradient_rate(double sigma, double slope, double lambda) {
    return slope * (1.0 - lambda * sigma);
}

inline double pc_scale_gradient(double sigma, double slope, double U, double alpha) {
    return pc_scale_gradient_rate(sigma, slope, pc_rate(U, alpha));
}

template<typename T>
inline T pc_log_prior_log_precision(const T& log_tau, double U, double alpha) {
    return pc_log_density(log_tau, -0.5, U, alpha);
}

inline double pc_log_prior_log_precision_grad(double log_tau, double U, double alpha) {
    return pc_scale_gradient(std::exp(-0.5 * log_tau), -0.5, U, alpha);
}

template<typename T>
inline T log1m_exp(const T& x) {
    if (get_value(x) > -1e-3) {
        return safe_log(-x)
            + log1p_fn(x * T(0.5) + x * x * T(1.0 / 6.0) + x * x * x * T(1.0 / 24.0));
    }
    return log1p_fn(-safe_exp(x));
}

using tulpa::math::log_prior_normal;

template<typename T>
inline T log_prior_gamma(const T& log_phi, double shape, double rate) {
    T phi = safe_exp(log_phi);
    return tulpa::math::log_prior_gamma_value(log_phi,phi,shape,rate);
}

inline double log_prior_gamma_score(double phi, double shape, double rate) {
    return tulpa::math::log_prior_gamma_gradient(phi,shape,rate);
}

inline double log_prior_gamma_grad(double log_phi, double shape, double rate) {
    return log_prior_gamma_score(std::exp(log_phi),shape,rate);
}

inline double log_prior_half_cauchy_grad(double log_sigma, double scale) {
    return tulpa::math::log_prior_half_cauchy_gradient(std::exp(log_sigma), scale);
}

inline double pc_prior_grad_log_sigma2(double sigma2, double U, double alpha) {
    return pc_scale_gradient(std::sqrt(sigma2), 0.5, U, alpha);
}
} }
#endif
