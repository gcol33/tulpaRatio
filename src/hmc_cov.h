// hmc_cov.h
// The covariance family the NNGP paths share, and the neighbour-block Cholesky
// that consumes it.
//
// The GP path, the SVC path and their two templated twins each used to carry a
// copy of the four kernels and of the small dense Cholesky over a neighbour
// set. The copies drifted: the Gaussian kernel was written with and without the
// 0.5 in its exponent, Matern 3/2 carried sqrt(3) exactly in two of them and
// rounded to 1.732050808 in the third, one had no spherical case and fell
// through to exponential, and the four regularized the neighbour block four
// different ways (1e-8 always, 1e-8 on failure, a 1e-6 floor on the pivot,
// 1e-4 always). A value and the gradient that differentiates it came from
// different copies, so the sampler ran on a gradient that was not the gradient
// of the density it reported.
//
// Everything below is templated on the scalar type, so one definition serves
// the double evaluation and the ad::Var / fwd::Dual / arena::Var gradients.

#ifndef RATIOD_HMC_COV_H
#define RATIOD_HMC_COV_H

#include <vector>
#include <cmath>
#include <string>

#include "autodiff_utils.h"
#include "enum_names.h"

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
constexpr double NNGP_RIDGE = 1e-8;

// Cholesky A = L L' of the n x n neighbour block, lower triangular, with
// `ridge` (NNGP_RIDGE unless the caller's matrix already carries its own)
// added to each diagonal entry. Returns false if a pivot is non-positive even
// so, which the callers turn into a rejected parameter state rather than a
// factor built from a substituted pivot.
//
// The pointer forms write into a caller-owned buffer, since the gradient paths
// carry one workspace across the whole location loop; the vector forms size
// their output and forward to them. Only the lower triangle of L is read, and
// the strict upper triangle is zeroed.
template <typename T>
inline bool nngp_chol(const T* A, int n, T* L, double ridge = NNGP_RIDGE) {
  for (int j = 0; j < n; j++) {
    for (int i = 0; i < j; i++) L[i * n + j] = T(0.0);

    T sum = A[j * n + j] + T(ridge);
    for (int k = 0; k < j; k++) {
      sum = sum - L[j * n + k] * L[j * n + k];
    }
    if (get_value(sum) <= 0.0) return false;
    L[j * n + j] = safe_sqrt(sum);

    for (int i = j + 1; i < n; i++) {
      T sum_ij = A[i * n + j];
      for (int k = 0; k < j; k++) {
        sum_ij = sum_ij - L[i * n + k] * L[j * n + k];
      }
      L[i * n + j] = sum_ij / L[j * n + j];
    }
  }
  return true;
}

// Solve L y = b.
template <typename T>
inline void nngp_forward_solve(const T* L, int n, const T* b, T* y) {
  for (int i = 0; i < n; i++) {
    T sum = b[i];
    for (int j = 0; j < i; j++) {
      sum = sum - L[i * n + j] * y[j];
    }
    y[i] = sum / L[i * n + i];
  }
}

// Solve L' x = y.
template <typename T>
inline void nngp_back_solve(const T* L, int n, const T* y, T* x) {
  for (int i = n - 1; i >= 0; i--) {
    T sum = y[i];
    for (int j = i + 1; j < n; j++) {
      sum = sum - L[j * n + i] * x[j];
    }
    x[i] = sum / L[i * n + i];
  }
}

template <typename T>
inline bool nngp_chol(const std::vector<T>& A, int n, std::vector<T>& L,
                      double ridge = NNGP_RIDGE) {
  L.assign(static_cast<size_t>(n) * n, T(0.0));
  return nngp_chol(A.data(), n, L.data(), ridge);
}

template <typename T>
inline void nngp_forward_solve(const std::vector<T>& L, int n,
                               const std::vector<T>& b, std::vector<T>& y) {
  y.resize(n);
  nngp_forward_solve(L.data(), n, b.data(), y.data());
}

template <typename T>
inline void nngp_back_solve(const std::vector<T>& L, int n,
                            const std::vector<T>& y, std::vector<T>& x) {
  x.resize(n);
  nngp_back_solve(L.data(), n, y.data(), x.data());
}

// One floor on the NNGP conditional variance, on every path. The copies carried
// 1e-10, 1e-6, and a 1e-4 blend that kept 1% of the gradient; at 1e-10 the floor
// binds only where the neighbour set has already reproduced the location
// exactly, so the density is the unfloored one wherever one exists.
constexpr double NNGP_MIN_COND_VAR = 1e-10;

template <typename T>
inline T nngp_floor_cond_var(const T& v) {
  if (get_value(v) < NNGP_MIN_COND_VAR) return T(NNGP_MIN_COND_VAR);
  return v;
}

}  // namespace ratiod_cov

#endif  // RATIOD_HMC_COV_H
