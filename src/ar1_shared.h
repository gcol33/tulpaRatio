// ar1_shared.h
// The quantities every AR1 correlation in the package reads: the floored
// 1 - rho^2, the map from the sampled logit coordinate, the prior that
// coordinate carries, and the AR1 precision itself.
//
// All of it exists once because all of it is read from more than one place that
// must agree. A density and its analytic gradient that floor 1 - rho^2 at
// different points, or with an added epsilon on one side and a clamp on the
// other, are orders of magnitude apart wherever the floor binds; a prior
// written out at the density and again at the gradient is a prior that can
// drift; and a field reached through a Kronecker product needs the same
// precision the scalar density evaluates termwise, as an operator.

#ifndef RATIOD_AR1_SHARED_H
#define RATIOD_AR1_SHARED_H

#include <cmath>

#include "autodiff_utils.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace ratiod_ar1 {

// The one floor on 1 - rho^2, applied as a clamp.
constexpr double ONE_MINUS_RHO2_FLOOR = 1e-10;

template <typename T>
inline T one_minus_rho2(const T& rho) {
  return ratiod::math::safe_max(T(1.0) - rho * rho,
                                T(ONE_MINUS_RHO2_FLOOR));
}

// d/d(rho) of log(1 - rho^2), flat wherever the clamp above binds so that a
// gradient cannot report curvature the density does not have.
inline double dlog_one_minus_rho2_drho(double rho) {
  const double raw = 1.0 - rho * rho;
  return (raw > ONE_MINUS_RHO2_FLOOR) ? (-2.0 * rho / raw) : 0.0;
}

// The sum of every entry of an n-step AR1 correlation matrix,
// sum_{t,t'} rho^|t - t'| = n + 2 sum_{k=1}^{n-1} (n - k) rho^k, by Horner.
// Positive for every rho in (-1, 1), as 1' C 1 of a correlation matrix C.
template <typename T>
inline T correlation_sum(const T& rho, int n) {
  T acc = T(0.0);
  for (int k = n - 1; k >= 1; k--) acc = acc * rho + T(static_cast<double>(n - k));
  return T(static_cast<double>(n)) + T(2.0) * rho * acc;
}

// d/d(rho) of correlation_sum, 2 sum_{k=1}^{n-1} k (n - k) rho^(k-1).
inline double dcorrelation_sum_drho(double rho, int n) {
  double acc = 0.0;
  for (int k = n - 1; k >= 1; k--) acc = acc * rho + static_cast<double>(k * (n - k));
  return 2.0 * acc;
}

// ---------------------------------------------------------------------------
// The sampled coordinate
// ---------------------------------------------------------------------------

// A correlation on (-1, 1) read from its sampled logit coordinate:
// rho = 2u - 1 at u = logit^-1(logit_rho).
template <typename T>
inline T rho_from_logit(const T& logit_rho) {
  return T(2.0) / (T(1.0) + ratiod::math::safe_exp(-logit_rho)) - T(1.0);
}

// d(rho) / d(logit(u)) at rho = 2u - 1, which is 2 * u * (1 - u).
inline double drho_dlogit(double rho) {
  return 0.5 * (1.0 - rho * rho);
}

// Beta(a, b) prior on u, expressed in the sampled coordinate logit(u) with the
// transform's log-Jacobian log(u) + log(1 - u) folded in:
//
//   log p(u) + log|du / dlogit(u)| = a log u + b log(1 - u) + const
//
// The package maps a correlation onto u two ways -- u = (rho + 1) / 2 for a rho
// on (-1, 1), u = rho for a rho on (0, 1) -- and each site passes its own u.
template <typename T>
inline T log_prior_logit_rho(const T& u, double a, double b) {
  return T(a) * ratiod::math::safe_log(u) +
         T(b) * ratiod::math::safe_log(T(1.0) - u);
}

// d/dlogit(u) of the above.
inline double log_prior_logit_rho_grad(double u, double a, double b) {
  return a * (1.0 - u) - b * u;
}

// The same prior and gradient for a correlation already read on (-1, 1).
template <typename T>
inline T log_prior_rho(const T& rho, double a, double b) {
  return log_prior_logit_rho((rho + T(1.0)) / T(2.0), a, b);
}

inline double log_prior_rho_grad(double rho, double a, double b) {
  return log_prior_logit_rho_grad(0.5 * (rho + 1.0), a, b);
}

// AR1 correlation on (-1, 1): Beta(2, 2) on u = (rho + 1) / 2, the prior
// priors_default() documents for a temporal correlation.
constexpr double RHO_PRIOR_A = 2.0;
constexpr double RHO_PRIOR_B = 2.0;

// ---------------------------------------------------------------------------
// The AR1 precision
// ---------------------------------------------------------------------------
//
// R(rho) is the AR1 precision at unit conditional precision on n >= 2 points:
// tridiagonal with R[0,0] = R[n-1,n-1] = 1, R[t,t] = 1 + rho^2 in the interior
// and R[t,t+1] = R[t+1,t] = -rho, so that
//
//   x' R x = (1 - rho^2) x[0]^2 + sum_{t>=1} (x[t] - rho x[t-1])^2,
//   log|R| = log(1 - rho^2).
//
// A field of precision tau then has log-density
// 0.5 (n log tau + log|R|) - 0.5 tau x' R x, up to the 2pi constant. Fewer than
// two points carry no temporal structure and contribute nothing, matching the
// convention the RW quadratic forms already follow.

template <typename T>
inline void ar1_precision_apply(const T* x, int n, const T& rho, T* out) {
  if (n < 2) {
    if (n == 1) out[0] = T(0.0);
    return;
  }
  const T rho2 = rho * rho;
  out[0] = x[0] - rho * x[1];
  for (int t = 1; t < n - 1; t++) {
    out[t] = (T(1.0) + rho2) * x[t] - rho * (x[t - 1] + x[t + 1]);
  }
  out[n - 1] = x[n - 1] - rho * x[n - 2];
}

template <typename T>
inline T ar1_quadratic_form(const T* x, int n, const T& rho) {
  if (n < 2) return T(0.0);
  T interior = T(0.0);
  for (int t = 1; t < n - 1; t++) interior = interior + x[t] * x[t];
  T cross = T(0.0);
  for (int t = 1; t < n; t++) cross = cross + x[t] * x[t - 1];
  return x[0] * x[0] + x[n - 1] * x[n - 1] +
         (T(1.0) + rho * rho) * interior - T(2.0) * rho * cross;
}

// d/d(rho) of the quadratic form G contracted against R(rho), where G is
// whatever Gram matrix the form contracts: G = x x' for a plain vector, and
// [delta[*,t1]' Q_s delta[*,t2]] for a field carrying R as one Kronecker
// margin. Only two sums of G survive the derivative --
//
//   interior = sum_{t=1}^{n-2} G[t][t],   cross = sum_{t=1}^{n-1} G[t][t-1]
//
// -- so a caller never forms G.
inline double ar1_quadratic_form_drho(double interior, double cross,
                                      double rho) {
  return 2.0 * rho * interior - 2.0 * cross;
}

// The AR1 log-prior at precision tau, without the 2pi constant, which is the
// form the GMRF branches beside it are written in.
template <typename T>
inline T ar1_log_prior(const T* x, int n, const T& rho, const T& tau) {
  if (n < 2) return T(0.0);
  return T(0.5 * n) * ratiod::math::safe_log(tau) +
         T(0.5) * ratiod::math::safe_log(one_minus_rho2(rho)) -
         T(0.5) * tau * ar1_quadratic_form(x, n, rho);
}

// The normalized AR1 log-density.
template <typename T>
inline T ar1_log_density(const T* x, int n, const T& rho, const T& tau) {
  if (n < 2) return T(0.0);
  return ar1_log_prior(x, n, rho, tau) - T(0.5 * n * std::log(2.0 * M_PI));
}

}  // namespace ratiod_ar1

#endif  // RATIOD_AR1_SHARED_H
