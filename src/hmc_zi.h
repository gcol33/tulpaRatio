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
#include <limits>
#include <string>
#include "autodiff_utils.h"
#include "enum_names.h"

namespace ratiod_zi {

using ratiod::math::get_value;
using ratiod::math::log1m_exp;
using ratiod::math::log_add_exp;
using ratiod::math::log_inv_logit;
using ratiod::math::log1m_inv_logit;
using ratiod::math::log_lik_poisson;
using ratiod::math::log_lik_negbin;
using ratiod::math::negbin_log_p0;

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

// The inflation and hurdle names R sends
inline constexpr ratiod_enum::EnumName<ZIType> ZI_TYPE_NAMES[] = {
  {"none", ZIType::NONE},
  {"zi_poisson", ZIType::ZI_POISSON},
  {"zi_negbin", ZIType::ZI_NEGBIN},
  {"hurdle_poisson", ZIType::HURDLE_POISSON},
  {"hurdle_negbin", ZIType::HURDLE_NEGBIN},
  {"zi_binomial", ZIType::ZI_BINOMIAL},
  {"hurdle_binomial", ZIType::HURDLE_BINOMIAL},
  {"oi_binomial", ZIType::OI_BINOMIAL},
  {"zoib", ZIType::ZOIB}};

inline ZIType parse_zi_type(const std::string& s) {
  return ratiod_enum::parse_enum(ZI_TYPE_NAMES, s, "zero/one inflation type");
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
    return log_add_exp(log_inv_logit(logit_zi), log1m_inv_logit(logit_zi) - mu);
  }
  return log1m_inv_logit(logit_zi) + log_lik_poisson(y, mu);
}

// Zero-inflated negative binomial log-PMF (logit scale for zi)
template<typename T>
inline T zi_negbin_lpmf_logit(int y, const T& mu, const T& phi, const T& logit_zi) {
  if (y == 0) {
    // log(P(Y=0)) = log(sigmoid(logit_zi) + sigmoid(-logit_zi) * P_NB(0))
    return log_add_exp(log_inv_logit(logit_zi),
                       log1m_inv_logit(logit_zi) + negbin_log_p0(mu, phi));
  }
  return log1m_inv_logit(logit_zi) + log_lik_negbin(y, mu, phi);
}

// ============================================================================
// Hurdle likelihoods
// ============================================================================

// Truncated Poisson log-PMF (conditional on y > 0)
// P(Y=y|Y>0) = Poisson(y|mu) / (1 - exp(-mu))
template<typename T>
inline T truncated_poisson_lpmf(int y, const T& mu) {
  if (y <= 0) return T(-std::numeric_limits<double>::infinity());
  return log_lik_poisson(y, mu) - log1m_exp(-mu);
}

// Truncated negative binomial log-PMF (conditional on y > 0)
template<typename T>
inline T truncated_negbin_lpmf(int y, const T& mu, const T& phi) {
  if (y <= 0) return T(-std::numeric_limits<double>::infinity());
  return log_lik_negbin(y, mu, phi) - log1m_exp(negbin_log_p0(mu, phi));
}

// Hurdle Poisson log-PMF (logit scale for hurdle)
// logit_theta: logit of P(Y > 0)
template<typename T>
inline T hurdle_poisson_lpmf_logit(int y, const T& mu, const T& logit_theta) {
  // P(Y=0) = 1 - theta; P(Y=y) = theta P(Y=y|Y>0)
  if (y == 0) return log1m_inv_logit(logit_theta);
  return log_inv_logit(logit_theta) + truncated_poisson_lpmf(y, mu);
}

// Hurdle negative binomial log-PMF (logit scale for hurdle)
template<typename T>
inline T hurdle_negbin_lpmf_logit(int y, const T& mu, const T& phi, const T& logit_theta) {
  if (y == 0) return log1m_inv_logit(logit_theta);
  return log_inv_logit(logit_theta) + truncated_negbin_lpmf(y, mu, phi);
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
        return log_lik_negbin(y, mu, phi);
      } else {
        return log_lik_poisson(y, mu);
      }
  }
}

}  // namespace ratiod_zi

#endif  // QUOTR_HMC_ZI_H
