// Spatial covariance kernels and derivatives for ratio-model Gaussian fields.
// Neighbour conditioning uses the engine's shared numeric primitives.

#ifndef RATIOD_HMC_COV_H
#define RATIOD_HMC_COV_H

#include <vector>
#include <cmath>
#include <string>

#include "autodiff_utils.h"
#include "enum_names.h"
#include <tulpa/gp_conditioning.h>

namespace ratiod_cov {

using namespace ratiod::math;

// Covariance function types
enum class CovType { EXPONENTIAL, MATERN, GAUSSIAN, SPHERICAL };

// The cov = names, in the order of the integer codes R/spatial.R's
// cov_type_code() sends
inline constexpr ratiod_enum::EnumName<CovType> COV_TYPE_NAMES[] = {
  {"exponential", CovType::EXPONENTIAL}, {"matern", CovType::MATERN},
  {"gaussian", CovType::GAUSSIAN},       {"spherical", CovType::SPHERICAL}};

inline CovType parse_cov_type(const std::string& s) {
  return ratiod_enum::parse_enum(COV_TYPE_NAMES, s, "covariance");
}

// sqrt(3), to the precision of the type rather than of a typed-out literal.
inline double sqrt3() { return std::sqrt(3.0); }

// -----------------------------------------------------------------------------
// The kernels. phi is a lengthscale in all four: the distance at which the
// correlation has decayed by a fixed factor, so the same prior on phi means the
// same thing whichever kernel is chosen.
// -----------------------------------------------------------------------------

// Exponential: sigma2 * exp(-d / phi)
template <typename T>
inline T cov_exponential(double d, const T& sigma2, const T& phi) {
  return sigma2 * safe_exp(-T(d) / phi);
}

// Matern 3/2: sigma2 * (1 + sqrt(3) d / phi) * exp(-sqrt(3) d / phi)
template <typename T>
inline T cov_matern32(double d, const T& sigma2, const T& phi) {
  T r = T(sqrt3() * d) / phi;
  return sigma2 * (T(1.0) + r) * safe_exp(-r);
}

// Matern 5/2: sigma2 * (1 + u + u^2 / 3) * exp(-u), u = sqrt(5) d / phi
template <typename T>
inline T cov_matern52(double d, const T& sigma2, const T& phi) {
  T u = T(std::sqrt(5.0) * d) / phi;
  return sigma2 * (T(1.0) + u + u * u / T(3.0)) * safe_exp(-u);
}

// Gaussian (squared exponential): sigma2 * exp(-0.5 * (d / phi)^2).
//
// The 0.5 is what makes phi the lengthscale here as well: it is the standard
// squared-exponential form and the nu -> infinity limit of Matern at a fixed
// phi. Without it phi differs from the lengthscale of the other three kernels
// by sqrt(2) while sharing their prior.
template <typename T>
inline T cov_gaussian(double d, const T& sigma2, const T& phi) {
  T r = T(d) / phi;
  return sigma2 * safe_exp(T(-0.5) * r * r);
}

// Spherical: sigma2 * (1 - 1.5 r + 0.5 r^3) for r = d / phi < 1, else 0
template <typename T>
inline T cov_spherical(double d, const T& sigma2, const T& phi) {
  if (d >= get_value(phi)) return T(0.0);
  T r = T(d) / phi;
  return sigma2 * (T(1.0) - T(1.5) * r + T(0.5) * r * r * r);
}

template <typename T>
inline T compute_cov(double d, const T& sigma2, const T& phi, CovType cov_type) {
  switch (cov_type) {
    case CovType::EXPONENTIAL:
      return cov_exponential(d, sigma2, phi);
    case CovType::MATERN:
      return cov_matern32(d, sigma2, phi);
    case CovType::GAUSSIAN:
      return cov_gaussian(d, sigma2, phi);
    case CovType::SPHERICAL:
      return cov_spherical(d, sigma2, phi);
  }
  return cov_exponential(d, sigma2, phi);
}

// -----------------------------------------------------------------------------
// dk(d) / dphi, for the analytic gradient paths. Written against the kernels
// directly above so the two cannot drift.
// -----------------------------------------------------------------------------

inline double dcov_dphi(double d, double sigma2, double phi, double cov_val,
                        CovType cov_type) {
  if (d < 1e-10) return 0.0;
  switch (cov_type) {
    case CovType::EXPONENTIAL:
      // k = s exp(-d/phi); dk/dphi = k d / phi^2
      return cov_val * d / (phi * phi);
    case CovType::MATERN: {
      // k = s (1+u) exp(-u), u = sqrt(3) d / phi; dk/dphi = k u^2 / (phi (1+u))
      double u = sqrt3() * d / phi;
      return (1.0 + u > 1e-10) ? cov_val * u * u / (phi * (1.0 + u)) : 0.0;
    }
    case CovType::GAUSSIAN:
      // k = s exp(-0.5 d^2/phi^2); dk/dphi = k d^2 / phi^3
      return cov_val * d * d / (phi * phi * phi);
    case CovType::SPHERICAL: {
      // k = s (1 - 1.5 r + 0.5 r^3), r = d/phi; dk/dphi = 1.5 s (r - r^3) / phi.
      // Not expressible through cov_val, which is why sigma2 is an argument.
      if (d >= phi) return 0.0;
      double r = d / phi;
      return 1.5 * sigma2 * (r - r * r * r) / phi;
    }
  }
  return cov_val * d / (phi * phi);
}

// -----------------------------------------------------------------------------
// The neighbour block
// -----------------------------------------------------------------------------

// One ridge for every NNGP neighbour block, on every path. A ridge is a
// covariance in its own right -- the kernel plus independent noise of that
// variance -- so the density it defines has a gradient. A floor applied to a
// pivot partway through a factorization is not: it changes the matrix by an
// amount that depends on where the factorization had got to, and the analytic
// derivative of the unfloored kernel no longer describes it.
template<class T>
inline T st_cov_value(double d, const T& sigma2, const T& phi, CovType type) {
  return compute_cov(d, sigma2, phi, type);
}
inline double st_cov_dphi(double d, double sigma2, double phi, double value, CovType type) {
  return dcov_dphi(d, sigma2, phi, value, type);
}

using tulpa_gp_conditioning::NNGP_RIDGE;
using tulpa_gp_conditioning::NNGP_MIN_COND_VAR;
using tulpa_gp_conditioning::nngp_chol;
using tulpa_gp_conditioning::nngp_forward_solve;
using tulpa_gp_conditioning::nngp_back_solve;
using tulpa_gp_conditioning::nngp_floor_cond_var;
}  // namespace ratiod_cov

#endif  // RATIOD_HMC_COV_H
