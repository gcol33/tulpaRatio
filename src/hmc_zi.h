#ifndef RATIOD_HMC_ZI_H
#define RATIOD_HMC_ZI_H

// Zero-inflation and hurdle model support for ratiod HMC
// Provides log-likelihood functions for ZI-Poisson, ZI-NegBin,
// and hurdle variants
//
// The count-response densities below are templated on the scalar type, so
// compute_log_post (T = double) and compute_log_post_impl<T> (double for
// evaluation, an autodiff scalar for the arena/forward/tape gradient modes)
// evaluate the same expression rather than two transcriptions of it.

#include <cmath>
#include <algorithm>
#include "autodiff_utils.h"

namespace ratiod_zi {

using ratiod::math::safe_exp;
using ratiod::math::safe_log;
using ratiod::math::safe_max;
using ratiod::math::log1p_fn;
using ratiod::math::lgamma_fn;
using ratiod::math::inv_logit;
using ratiod::math::get_value;
using ratiod::math::log1m_exp;

// Enum for ZI model types
enum class ZIType {
  NONE,           // No zero-inflation
  ZI_POISSON,     // Zero-inflated Poisson
  ZI_NEGBIN,      // Zero-inflated negative binomial
  HURDLE_POISSON, // Hurdle Poisson (truncated at zero)
  HURDLE_NEGBIN,  // Hurdle negative binomial
  ZI_BINOMIAL,    // Zero-inflated binomial
  HURDLE_BINOMIAL,// Hurdle binomial
  OI_BINOMIAL,    // One-inflated binomial (excess y=n)
  ZOIB            // Zero-and-one inflated binomial
};

// Parse ZI type from string
inline ZIType parse_zi_type(const std::string& zi_type_str) {
  if (zi_type_str == "none" || zi_type_str == "") {
    return ZIType::NONE;
  } else if (zi_type_str == "zi_poisson") {
    return ZIType::ZI_POISSON;
  } else if (zi_type_str == "zi_negbin") {
    return ZIType::ZI_NEGBIN;
  } else if (zi_type_str == "hurdle_poisson") {
    return ZIType::HURDLE_POISSON;
  } else if (zi_type_str == "hurdle_negbin") {
    return ZIType::HURDLE_NEGBIN;
  } else if (zi_type_str == "zi_binomial") {
    return ZIType::ZI_BINOMIAL;
  } else if (zi_type_str == "hurdle_binomial") {
    return ZIType::HURDLE_BINOMIAL;
  } else if (zi_type_str == "oi_binomial") {
    return ZIType::OI_BINOMIAL;
  } else if (zi_type_str == "zoib" || zi_type_str == "zoibinomial") {
    return ZIType::ZOIB;
  } else {
    return ZIType::NONE;
  }
}

// Log of 1 + exp(x), numerically stable
template<typename T>
inline T log1pexp(const T& x) {
  const double xv = get_value(x);
  if (xv > 35.0) return x;
  if (xv < -10.0) return safe_exp(x);
  return log1p_fn(safe_exp(x));
}

// Logistic sigmoid function
template<typename T>
inline T logistic(const T& x) {
  return inv_logit(x);
}

// Log of logistic sigmoid (log(1/(1+exp(-x))))
template<typename T>
inline T log_logistic(const T& x) {
  return -log1pexp(-x);
}

// Log of 1 - logistic(x) = log(exp(-x)/(1+exp(-x))) = -x - log(1+exp(-x))
template<typename T>
inline T log1m_logistic(const T& x) {
  return -log1pexp(x);
}

// Log factorial. Constant in the parameters, so it stays a double on every path.
inline double lfactorial(int n) {
  return lgamma_fn(n + 1.0);
}

// ============================================================================
// Standard log-likelihoods (for reference)
// ============================================================================

// Poisson log-PMF: y ~ Poisson(mu)
template<typename T>
inline T poisson_lpmf(int y, const T& mu) {
  if (get_value(mu) <= 0) return T(-1e10);
  return y * safe_log(mu) - mu - lfactorial(y);
}

// Negative binomial log-PMF: y ~ NegBin(mu, phi)
// Using NB2 parameterization: Var(Y) = mu + mu^2/phi
template<typename T>
inline T negbin_lpmf(int y, const T& mu, const T& phi) {
  if (get_value(mu) <= 0 || get_value(phi) <= 0) return T(-1e10);
  T r = phi;                // size parameter
  T p = phi / (phi + mu);   // success probability
  return lgamma_fn(T(y) + r) - lgamma_fn(r) - lfactorial(y) +
         r * safe_log(p) + y * safe_log(T(1.0) - p);
}

// ============================================================================
// Zero-inflated likelihoods
// ============================================================================

// Zero-inflated Poisson log-PMF (logit scale for zi)
// logit_zi: logit of zero-inflation probability
template<typename T>
inline T zi_poisson_lpmf_logit(int y, const T& mu, const T& logit_zi) {
  if (y == 0) {
    // log(P(Y=0)) = log(sigmoid(logit_zi) + sigmoid(-logit_zi) * exp(-mu))
    T log_zi = log_logistic(logit_zi);
    T log_1m_zi = log1m_logistic(logit_zi);
    T log_p0_count = -mu;

    T a = log_zi;
    T b = log_1m_zi + log_p0_count;
    T max_ab = safe_max(a, b);
    return max_ab + safe_log(safe_exp(a - max_ab) + safe_exp(b - max_ab));
  } else {
    return log1m_logistic(logit_zi) + poisson_lpmf(y, mu);
  }
}

// Zero-inflated negative binomial log-PMF (logit scale for zi)
template<typename T>
inline T zi_negbin_lpmf_logit(int y, const T& mu, const T& phi, const T& logit_zi) {
  if (y == 0) {
    // log(P(Y=0)) = log(sigmoid(logit_zi) + sigmoid(-logit_zi) * P_NB(0))
    T log_zi = log_logistic(logit_zi);
    T log_1m_zi = log1m_logistic(logit_zi);

    // P_NB(0) = (phi / (phi + mu))^phi
    T log_p0_count = phi * safe_log(phi / (phi + mu));

    T a = log_zi;
    T b = log_1m_zi + log_p0_count;
    T max_ab = safe_max(a, b);
    return max_ab + safe_log(safe_exp(a - max_ab) + safe_exp(b - max_ab));
  } else {
    return log1m_logistic(logit_zi) + negbin_lpmf(y, mu, phi);
  }
}

// ============================================================================
// Hurdle likelihoods
// ============================================================================

// Truncated Poisson log-PMF (conditional on y > 0)
// P(Y=y|Y>0) = Poisson(y|mu) / (1 - exp(-mu))
template<typename T>
inline T truncated_poisson_lpmf(int y, const T& mu) {
  if (y <= 0) return T(-1e10);
  return poisson_lpmf(y, mu) - log1m_exp(-mu);
}

// Truncated negative binomial log-PMF (conditional on y > 0)
template<typename T>
inline T truncated_negbin_lpmf(int y, const T& mu, const T& phi) {
  if (y <= 0) return T(-1e10);
  return negbin_lpmf(y, mu, phi) - log1m_exp(phi * safe_log(phi / (phi + mu)));
}

// Hurdle Poisson log-PMF (logit scale for hurdle)
// logit_theta: logit of P(Y > 0)
template<typename T>
inline T hurdle_poisson_lpmf_logit(int y, const T& mu, const T& logit_theta) {
  if (y == 0) {
    // P(Y=0) = 1 - theta = sigmoid(-logit_theta)
    return log1m_logistic(logit_theta);
  } else {
    // P(Y=y|Y>0) * theta
    return log_logistic(logit_theta) + truncated_poisson_lpmf(y, mu);
  }
}

// Hurdle negative binomial log-PMF (logit scale for hurdle)
template<typename T>
inline T hurdle_negbin_lpmf_logit(int y, const T& mu, const T& phi, const T& logit_theta) {
  if (y == 0) {
    return log1m_logistic(logit_theta);
  } else {
    return log_logistic(logit_theta) + truncated_negbin_lpmf(y, mu, phi);
  }
}

// ============================================================================
// Unified interface
// ============================================================================

// Compute ZI/hurdle log-likelihood for a single observation
// logit_zi_or_theta: logit-scale parameter for ZI prob or hurdle prob
// phi: overdispersion (only used for negbin variants)
template<typename T>
inline T zi_log_likelihood(
    int y, const T& mu, const T& phi, const T& logit_zi_or_theta,
    ZIType zi_type) {

  switch (zi_type) {
    case ZIType::ZI_POISSON:
      return zi_poisson_lpmf_logit(y, mu, logit_zi_or_theta);

    case ZIType::ZI_NEGBIN:
      return zi_negbin_lpmf_logit(y, mu, phi, logit_zi_or_theta);

    case ZIType::HURDLE_POISSON:
      return hurdle_poisson_lpmf_logit(y, mu, logit_zi_or_theta);

    case ZIType::HURDLE_NEGBIN:
      return hurdle_negbin_lpmf_logit(y, mu, phi, logit_zi_or_theta);

    case ZIType::NONE:
    default:
      // No ZI, return standard likelihood
      if (get_value(phi) > 0) {
        return negbin_lpmf(y, mu, phi);
      } else {
        return poisson_lpmf(y, mu);
      }
  }
}

// ============================================================================
// Gradients for ZI parameters
// ============================================================================

// Gradient of ZI-Poisson log-likelihood w.r.t. logit_zi
inline double zi_poisson_grad_logit_zi(int y, double mu, double logit_zi) {
  double zi = logistic(logit_zi);
  double one_m_zi = 1.0 - zi;

  if (y == 0) {
    // d/d(logit_zi) log(zi + (1-zi)*exp(-mu))
    // = [zi*(1-zi) - zi*(1-zi)*exp(-mu)] / [zi + (1-zi)*exp(-mu)]
    // = zi*(1-zi)*(1 - exp(-mu)) / [zi + (1-zi)*exp(-mu)]
    double exp_neg_mu = std::exp(-mu);
    double p0 = zi + one_m_zi * exp_neg_mu;
    return zi * one_m_zi * (1.0 - exp_neg_mu) / p0;
  } else {
    // d/d(logit_zi) log(1-zi) = -zi
    return -zi;
  }
}

// Gradient of ZI-NegBin log-likelihood w.r.t. logit_zi
inline double zi_negbin_grad_logit_zi(int y, double mu, double phi, double logit_zi) {
  double zi = logistic(logit_zi);
  double one_m_zi = 1.0 - zi;

  if (y == 0) {
    double p0_nb = std::pow(phi / (phi + mu), phi);
    double p0 = zi + one_m_zi * p0_nb;
    return zi * one_m_zi * (1.0 - p0_nb) / p0;
  } else {
    return -zi;
  }
}

// Gradient of hurdle model log-likelihood w.r.t. logit_theta
inline double hurdle_grad_logit_theta(int y, double logit_theta) {
  double theta = logistic(logit_theta);

  if (y == 0) {
    // d/d(logit_theta) log(1-theta) = -theta
    return -theta;
  } else {
    // d/d(logit_theta) log(theta) = 1-theta
    return 1.0 - theta;
  }
}

// P0 / (1 - P0) from log P0: the odds of the zero class a zero-truncated
// distribution removes, without forming 1 - P0, which rounds to zero for a
// small mean.
inline double zero_odds(double log_p0) {
  return 1.0 / std::expm1(-log_p0);
}

// Derivatives of log theta + log TruncNB(y | mu, phi), y > 0, with respect to
// eta = log mu (`resid`) and phi (`grad_phi`). The truncation subtracts
// log(1 - P0), log P0 = phi log(phi / (phi + mu)).
inline void hurdle_negbin_positive_grads(int y, double mu, double phi,
                                         double& resid, double& grad_phi) {
  const double denom = mu + phi;
  const double log_ratio = std::log(phi / denom);
  const double odds = zero_odds(phi * log_ratio);
  resid = y - mu * (y + phi) / denom - odds * phi * mu / denom;
  grad_phi = ratiod::math::portable_digamma(y + phi) - ratiod::math::portable_digamma(phi)
             + log_ratio + (mu - y) / denom
             + odds * (log_ratio + mu / denom);
}

}  // namespace ratiod_zi

#endif  // QUOTR_HMC_ZI_H
