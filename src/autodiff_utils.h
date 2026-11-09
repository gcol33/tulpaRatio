// autodiff_utils.h
// Templated math functions that work with double, ad::Var (tape), fwd::Dual (forward),
// and arena::Var (arena reverse-mode)
// Enables single implementation of compute_log_post for both evaluation and gradient

#ifndef RATIOD_AUTODIFF_UTILS_H
#define RATIOD_AUTODIFF_UTILS_H

#include <cmath>
#include <limits>
#include <vector>
#include <Rcpp.h>
#include "ratio_math.h"
#include "link_function.h"
#include <tulpa/count_math.h>

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

// Dot product with double design matrix and T coefficients
template<typename T>
inline T dot_product_mixed(const double* x, const std::vector<T>& y, int n) {
    T sum = T(0.0);
    for (int i = 0; i < n; i++) {
        sum = sum + x[i] * y[i];
    }
    return sum;
}

// Simpler: dot product where x is double array, y is T vector starting at offset
template<typename T>
inline T dot_product_mixed(const double* x, const T* y, int n) {
    T sum = T(0.0);
    for (int i = 0; i < n; i++) {
        sum = sum + x[i] * y[i];
    }
    return sum;
}

// ============================================================================
// Likelihood functions (templated)
// ============================================================================

// log(1 + exp(x)) without overflow at either end


// log(inv_logit(x)) and log(1 - inv_logit(x)), finite for every finite x:
// forming 1 - inv_logit(x) first rounds to zero once x > ~36.7




// log(exp(a) + exp(b))


// The negative binomial's log zero mass phi log(phi / (phi + mu)), written as
// -phi log1p(mu / phi) so it stays exact as phi outgrows mu, where the ratio
// rounds to one
struct NegbinMath {
    template <typename T> static double value(const T& x) { return get_value(x); }
    template <typename T> static T log(const T& x) { return safe_log(x); }
    template <typename T> static T log1p(const T& x) { return log1p_fn(x); }
    template <typename T> static T log1pmx(const T& x) { using tulpa::math::log1pmx; return log1pmx(x); }
    template <typename T> static T lgamma(const T& x) { return lgamma_fn(x); }
};

template<typename T>
inline T negbin_log_p0(const T& mu, const T& phi) {
    return tulpa::math::negbin_log_p0<T, NegbinMath>(mu, phi);
}

template<typename T>
inline T log_lik_poisson(int y, const T& mu) {
    // y * log(mu) - mu - lgamma(y+1)
    // Note: lgamma(y+1) is constant w.r.t. parameters, but we include it for correctness
    return y * safe_log(mu) - mu - lgamma_fn(T(y + 1.0));
}

template<typename T>
inline T log_lik_gamma(double y, const T& shape, const T& mu) {
    return tulpa::math::gamma_log_density(y, mu, shape, std::log(y));
}

template<typename T>
inline T log_lik_negbin(int y, const T& mu, const T& phi) {
    return tulpa::math::negbin_log_pmf<T, NegbinMath>(y, mu, phi);
}

// ============================================================================
// Binomial-type log-likelihoods (templated)
// ============================================================================

// Each takes the success probability through its logit q, so log p and
// log(1 - p) stay finite at every q.

template<typename T>
inline T log_lik_binomial(int y, int n, const T& q) {
    // y * log(p) + (n-y) * log(1-p) + lchoose(n, y)
    // lchoose is constant w.r.t. p
    T log_lik = y * log_inv_logit(q) + (n - y) * log1m_inv_logit(q);
    log_lik = log_lik + portable_lchoose(n, y);  // constant
    return log_lik;
}

// Zero-inflated binomial: P(Y=0) = zi + (1-zi) * Binom(0|n,p)
template<typename T>
inline T log_lik_zi_binomial(int y, int n, const T& q, const T& logit_zi) {
    if (y == 0) {
        return log_add_exp(log_inv_logit(logit_zi),
                           log1m_inv_logit(logit_zi) + n * log1m_inv_logit(q));
    }
    return log1m_inv_logit(logit_zi) + log_lik_binomial(y, n, q);
}

// One-inflated binomial: P(Y=n) = oi + (1-oi) * Binom(n|n,p)
template<typename T>
inline T log_lik_oi_binomial(int y, int n, const T& q, const T& logit_oi) {
    if (y == n) {
        return log_add_exp(log_inv_logit(logit_oi),
                           log1m_inv_logit(logit_oi) + n * log_inv_logit(q));
    }
    return log1m_inv_logit(logit_oi) + log_lik_binomial(y, n, q);
}

// Zero-and-one inflated binomial (ZOIB)
//   P(Y=0) = zi + (1-zi) (1-oi) Binom(0|n,p)
//   P(Y=n) = (1-zi) (oi + (1-oi) Binom(n|n,p))
//   P(Y=y) = (1-zi) (1-oi) Binom(y|n,p) otherwise
template<typename T>
inline T log_lik_zoib(int y, int n, const T& q, const T& logit_zi, const T& logit_oi) {
    const T log_1m_zi = log1m_inv_logit(logit_zi);
    const T log_1m_oi = log1m_inv_logit(logit_oi);
    if (y == 0) {
        return log_add_exp(log_inv_logit(logit_zi),
                           log_1m_zi + log_1m_oi + n * log1m_inv_logit(q));
    }
    if (y == n) {
        return log_1m_zi + log_add_exp(log_inv_logit(logit_oi),
                                       log_1m_oi + n * log_inv_logit(q));
    }
    return log_1m_zi + log_1m_oi + log_lik_binomial(y, n, q);
}

// PC prior on the standard deviation sigma = tau^(-1/2) of a field with
// precision tau: sigma ~ Exponential(lambda), lambda = -log(alpha) / U, so
// that P(sigma > U) = alpha. As a density of the sampled coordinate log tau,
// lambda exp(-lambda sigma) |d sigma / d log tau| = lambda exp(-lambda sigma)
// sigma / 2, which is proper at both ends.


// sigma = exp(slope * x). The Jacobian is |slope| * sigma.












// Its derivative with respect to log tau: lambda sigma / 2 - 1 / 2.


// log(1 - exp(x)) for x < 0, the log mass a zero-truncated distribution keeps.
// Near x = 0, 1 - exp(x) rounds to zero long before it is zero; there
// -expm1(x) = -x (1 + x/2 + x^2/6 + x^3/24 + ...), so the logarithm is
// log(-x) + log1p(x/2 + x^2/6 + x^3/24), exact to ~1e-14 for |x| < 1e-3.


// Hurdle binomial: separate zero vs non-zero
template<typename T>
inline T log_lik_hurdle_binomial(int y, int n, const T& q, const T& logit_theta) {
    if (y == 0) return log1m_inv_logit(logit_theta);
    // theta P(Y=y|Y>0), P(Y=y|Y>0) = Binom(y|n,p) / (1 - (1-p)^n)
    return log_inv_logit(logit_theta) + log_lik_binomial(y, n, q)
        - log1m_exp(n * log1m_inv_logit(q));
}

// ============================================================================
// New family likelihoods: Gamma-Gamma, Lognormal, Beta-Binomial
// ============================================================================

// Gamma-Gamma: Both numerator and denominator are Gamma distributed
// Using mean parameterization: mu = shape/rate, so rate = shape/mu
template<typename T>
inline T log_lik_gamma_gamma(double y_num, double y_denom,
                             const T& mu_num, const T& mu_denom,
                             const T& shape_num, const T& shape_denom) {
    return log_lik_gamma(y_num, shape_num, mu_num)
         + log_lik_gamma(y_denom, shape_denom, mu_denom);
}

// Lognormal: log(y) ~ Normal(mu, sigma^2)
// Log-likelihood: -log(y) - log(sigma) - 0.5*log(2*pi) - 0.5*((log(y)-mu)/sigma)^2
template<typename T>
inline T log_lik_lognormal(double y, const T& mu, const T& sigma) {
    double log_y = std::log(y);
    T z = (log_y - mu) / sigma;
    // Omitting constant -0.5*log(2*pi)
    return -std::log(y) - safe_log(sigma) - T(0.5) * z * z;
}

// Lognormal-Lognormal: Both numerator and denominator are Lognormal
template<typename T>
inline T log_lik_lognormal_lognormal(double y_num, double y_denom,
                                     const T& mu_num, const T& mu_denom,
                                     const T& sigma_num, const T& sigma_denom) {
    return log_lik_lognormal(y_num, mu_num, sigma_num) +
           log_lik_lognormal(y_denom, mu_denom, sigma_denom);
}

// Beta-Binomial: y ~ BetaBinomial(n, alpha, beta)
// Using mean-precision parameterization: mu = alpha/(alpha+beta), phi = alpha+beta
// So alpha = mu*phi, beta = (1-mu)*phi, mu taken through its logit q
template<typename T>
inline T log_lik_beta_binomial(int y, int n, const T& q, const T& phi) {
    return tulpa::math::beta_binomial_log_pmf<T, NegbinMath>(y, n,
        safe_exp(log_inv_logit(q)), safe_exp(log1m_inv_logit(q)), phi, portable_lchoose(n,y));
}

// ============================================================================
// Prior distributions (templated)
// ============================================================================

// Normal prior: -0.5 * tau * x^2 (ignoring constants)


// Gamma prior on phi (log scale): (shape-1)*log(phi) - rate*phi + log(phi)


}  // namespace math
}  // namespace ratiod

#endif  // RATIOD_AUTODIFF_UTILS_H
