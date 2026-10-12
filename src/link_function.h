// link_function.h
// Link functions of the linear predictors, evaluated in double precision with
// their first derivatives. The templated log posterior lifts each value and
// derivative pair onto whichever scalar type it runs on (autodiff_utils.h).

#ifndef RATIOD_LINK_FUNCTION_H
#define RATIOD_LINK_FUNCTION_H

#include <cmath>
#include <string>

namespace ratiod_link {

enum class LinkType { LOG, LOGIT, PROBIT, CLOGLOG, IDENTITY, INVERSE };

// A value of a function of eta and its derivative with respect to eta.
struct ValueDeriv {
  double value;
  double deriv;
};

// log Phi(x), the log standard normal CDF. erfc keeps full relative accuracy
// down to x = -37; below that the leading terms of the asymptotic series
// log Phi(x) = -x^2/2 - log(-x) - log(2 pi)/2 + log(1 - 1/x^2 + 3/x^4) are
// exact to double precision.
inline double log_normal_cdf(double x) {
  if (x > -37.0) return std::log(0.5 * std::erfc(-x / std::sqrt(2.0)));
  const double x2 = x * x;
  return -0.5 * x2 - std::log(-x) - 0.5 * std::log(2.0 * M_PI)
         + std::log1p(-1.0 / x2 + 3.0 / (x2 * x2));
}

inline double log_normal_pdf(double x) {
  return -0.5 * x * x - 0.5 * std::log(2.0 * M_PI);
}

// The logit of the probability a link maps eta to, with its derivative.
// Every probability in the likelihoods (a success probability, an inflation
// or hurdle probability) enters through its logit, so the existing logit-scale
// kernels serve every link.
//
//   probit:  logit = log Phi(eta) - log Phi(-eta),
//            d/deta = phi(eta) / Phi(eta) + phi(eta) / Phi(-eta)
//   cloglog: p = 1 - exp(-e^eta), so log(1 - p) = -e^eta and
//            logit = log p + e^eta, d/deta = e^eta / p
inline ValueDeriv prob_logit_vd(double eta, LinkType link) {
  switch (link) {
    case LinkType::PROBIT: {
      const double lp = log_normal_cdf(eta);
      const double lq = log_normal_cdf(-eta);
      const double lpdf = log_normal_pdf(eta);
      return {lp - lq, std::exp(lpdf - lp) + std::exp(lpdf - lq)};
    }
    case LinkType::CLOGLOG: {
      const double e = std::exp(eta);
      const double p = -std::expm1(-e);
      return {std::log(p) + e, e / p};
    }
    default:
      return {eta, 1.0};
  }
}

// The mean a link maps eta to, with its derivative. A mean that leaves the
// positive half-line (identity or inverse link) has no density; the caller
// checks the value and returns -Inf.
inline ValueDeriv link_mean_vd(double eta, LinkType link) {
  switch (link) {
    case LinkType::IDENTITY:
      return {eta, 1.0};
    case LinkType::INVERSE:
      return {1.0 / eta, -1.0 / (eta * eta)};
    default: {
      const double m = std::exp(eta);
      return {m, m};
    }
  }
}

inline bool is_probability_link(LinkType link) {
  return link == LinkType::LOGIT || link == LinkType::PROBIT ||
         link == LinkType::CLOGLOG;
}

// Parse a link name; false when the name is not a link.
inline bool parse_link(const std::string& name, LinkType& out) {
  if (name == "log")      { out = LinkType::LOG;      return true; }
  if (name == "logit")    { out = LinkType::LOGIT;    return true; }
  if (name == "probit")   { out = LinkType::PROBIT;   return true; }
  if (name == "cloglog")  { out = LinkType::CLOGLOG;  return true; }
  if (name == "identity") { out = LinkType::IDENTITY; return true; }
  if (name == "inverse")  { out = LinkType::INVERSE;  return true; }
  return false;
}

}  // namespace ratiod_link

#endif  // RATIOD_LINK_FUNCTION_H
