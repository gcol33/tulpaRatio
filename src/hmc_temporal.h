// hmc_temporal.h
// Temporal random effects support for HMC backend
// Supports RW1, RW2, and AR1 temporal structures

#ifndef RATIOD_HMC_TEMPORAL_H
#define RATIOD_HMC_TEMPORAL_H

#define _USE_MATH_DEFINES  // For M_PI on Windows
#include <vector>
#include <cmath>
#include <string>
#include <utility>

#include "enum_names.h"
#include <tulpa/sum_to_zero.h>  // rw1_rank / rw2_rank / s2z_component_sum
#include "ar1_shared.h"              // the floored 1 - rho^2

// Fallback definition of M_PI if not provided by <cmath>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace ratiod_temporal {

// =====================================================================
// Temporal structure types
// =====================================================================

enum class TemporalType { NONE, RW1, RW2, AR1, IID, GP };

// The R-side name of each temporal structure. "multiscale" is not here: it is
// a composite the data builder reads before any single structure.
inline constexpr ratiod_enum::EnumName<TemporalType> TEMPORAL_TYPE_NAMES[] = {
  {"none", TemporalType::NONE}, {"rw1", TemporalType::RW1},
  {"rw2", TemporalType::RW2},   {"ar1", TemporalType::AR1},
  {"iid", TemporalType::IID},   {"gp", TemporalType::GP}};

inline TemporalType parse_temporal_type(const std::string& s) {
  return ratiod_enum::parse_enum(TEMPORAL_TYPE_NAMES, s, "temporal structure");
}

// Temporal data container
struct TemporalData {
  TemporalType type;
  std::vector<int> time_index;   // Maps obs to time point (1-based, 0 = no temporal)
  std::vector<int> group_index;  // Maps obs to temporal group (1-based, for panel data)
  int n_times;                   // Number of time points
  int n_groups;                  // Number of groups (1 if no grouping)
  int n_temporal_params;         // Total temporal parameters (n_times * n_groups)
  bool cyclic;                   // Whether RW is cyclic
  bool shared;                   // Whether effect is shared between num/denom

  // Prior parameters
  double tau_temporal_shape;     // Gamma shape for precision
  double tau_temporal_rate;      // Gamma rate for precision
};

// =====================================================================
// RW1 precision matrix quadratic form
// =====================================================================

// Compute phi' Q_RW1 phi for RW1 prior
// Q_RW1 is the first-order random walk precision matrix
inline double rw1_quadratic_form(
    const double* phi,
    int T,
    bool cyclic
) {
  double quad = 0.0;

  // Sum of squared differences
  for (int t = 0; t < T - 1; t++) {
    double diff = phi[t + 1] - phi[t];
    quad += diff * diff;
  }

  // Cyclic: add edge from T to 1
  if (cyclic) {
    double diff = phi[0] - phi[T - 1];
    quad += diff * diff;
  }

  return quad;
}

// =====================================================================
// RW2 precision matrix quadratic form
// =====================================================================

// Compute phi' Q_RW2 phi for RW2 prior
// Q_RW2 penalizes second differences (curvature)
inline double rw2_quadratic_form(
    const double* phi,
    int T,
    bool cyclic
) {
  if (T < 3) return 0.0;

  double quad = 0.0;

  // Sum of squared second differences
  for (int t = 0; t < T - 2; t++) {
    double diff2 = phi[t] - 2.0 * phi[t + 1] + phi[t + 2];
    quad += diff2 * diff2;
  }

  // Cyclic: wrap around
  if (cyclic && T >= 3) {
    // t = T-2
    double diff2_1 = phi[T - 2] - 2.0 * phi[T - 1] + phi[0];
    quad += diff2_1 * diff2_1;
    // t = T-1
    double diff2_2 = phi[T - 1] - 2.0 * phi[0] + phi[1];
    quad += diff2_2 * diff2_2;
  }

  return quad;
}

// =====================================================================
// AR1 log-density
// =====================================================================

// Compute log-density for AR1 process
// phi[t] | phi[t-1] ~ N(rho * phi[t-1], sigma^2)
// Marginal variance: sigma^2 / (1 - rho^2)
//
// The exponent and the normalizer are ratiod_ar1's precision R(rho), the same
// matrix the spatiotemporal interaction reads as a Kronecker margin.
inline double ar1_log_density(
    const double* phi,
    int T,
    double rho,
    double tau  // precision = 1/sigma^2
) {
  return ratiod_ar1::ar1_log_density(phi, T, rho, tau);
}

// =====================================================================
// Non-centered AR1 parameterization helpers
// =====================================================================

// Forward recursion: z (N(0,1) innovations) → phi (actual effects)
// phi[0] = z[0] / sqrt(tau * (1 - rho^2))
// phi[t] = rho * phi[t-1] + z[t] / sqrt(tau)   for t >= 1
inline void ar1_nc_forward(
    const double* z, double* phi, int T,
    double rho, double tau
) {
  if (T <= 0) return;
  double omr2 = ratiod_ar1::one_minus_rho2(rho);
  double inv_sqrt_tau = 1.0 / std::sqrt(tau);
  double inv_sqrt_tau_omr2 = inv_sqrt_tau / std::sqrt(omr2);

  phi[0] = z[0] * inv_sqrt_tau_omr2;
  for (int t = 1; t < T; t++) {
    phi[t] = rho * phi[t - 1] + z[t] * inv_sqrt_tau;
  }
}

// Compute gradients for non-centered AR1 parameterization.
// Given grad_phi_lik = d(likelihood)/d(phi), computes:
//   grad_z[t] = d(log_post)/d(z[t]) including N(0,1) prior
//   returns (grad_log_tau_from_lik, grad_logit_rho_from_lik)
//     — likelihood contributions to hyperparameter gradients via chain rule
//
// The caller is responsible for adding the tau and rho PRIOR gradients separately.
inline std::pair<double, double> ar1_nc_gradient(
    const double* z,           // NC parameters from param vector
    const double* phi,         // Reconstructed effects from ar1_nc_forward
    const double* grad_phi_lik,// d(likelihood)/d(phi)
    double* grad_z,            // OUTPUT: final d(log_post)/d(z)
    int T,
    double rho,
    double tau
) {
  if (T <= 0) return {0.0, 0.0};
  if (T == 1) {
    // Single time point: phi[0] = z[0] / sqrt(tau*(1-rho^2))
    double omr2 = ratiod_ar1::one_minus_rho2(rho);
    double inv_sqrt_tau_omr2 = 1.0 / std::sqrt(tau * omr2);
    grad_z[0] = grad_phi_lik[0] * inv_sqrt_tau_omr2 - z[0];
    // tau gradient: d(phi[0])/d(log_tau) = -0.5 * phi[0]
    double glt = grad_phi_lik[0] * (-0.5 * phi[0]);
    // rho gradient: d(phi[0])/d(rho) = phi[0] * rho / (1-rho^2)
    double gr = grad_phi_lik[0] * (phi[0] * rho / omr2);
    double glr = gr * ratiod_ar1::drho_dlogit(rho);
    return {glt, glr};
  }

  double omr2 = ratiod_ar1::one_minus_rho2(rho);
  double inv_sqrt_tau = 1.0 / std::sqrt(tau);
  double inv_sqrt_tau_omr2 = inv_sqrt_tau / std::sqrt(omr2);

  // ---- Backward adjoint: adj[t] accumulates all downstream effects ----
  // adj[t] = grad_phi_lik[t] + rho * adj[t+1]
  std::vector<double> adj(T);
  adj[T - 1] = grad_phi_lik[T - 1];
  for (int t = T - 2; t >= 0; t--) {
    adj[t] = grad_phi_lik[t] + rho * adj[t + 1];
  }

  // ---- z gradients: adj * d(phi)/d(z) + N(0,1) prior ----
  grad_z[0] = adj[0] * inv_sqrt_tau_omr2 - z[0];
  for (int t = 1; t < T; t++) {
    grad_z[t] = adj[t] * inv_sqrt_tau - z[t];
  }

  // ---- log_tau gradient via chain rule through phi transformation ----
  // d(phi[0])/d(log_tau) = -0.5 * phi[0]
  // d(phi[t])/d(log_tau) = rho * d(phi[t-1])/d(log_tau) - 0.5 * (phi[t] - rho*phi[t-1])
  //
  // These derivatives are total: each already carries the recursion forward.
  // They pair with the DIRECT partial grad_phi_lik[t], not with adj[t], which
  // carries the same recursion backward -- using both counts it twice.
  double dphi_dlt = -0.5 * phi[0];
  double grad_log_tau = grad_phi_lik[0] * dphi_dlt;
  for (int t = 1; t < T; t++) {
    dphi_dlt = rho * dphi_dlt - 0.5 * (phi[t] - rho * phi[t - 1]);
    grad_log_tau += grad_phi_lik[t] * dphi_dlt;
  }

  // ---- rho gradient via chain rule through phi transformation ----
  // d(phi[0])/d(rho) = phi[0] * rho / (1 - rho^2)
  // d(phi[t])/d(rho) = phi[t-1] + rho * d(phi[t-1])/d(rho)
  double dphi_drho = phi[0] * rho / omr2;
  double grad_rho = grad_phi_lik[0] * dphi_drho;
  for (int t = 1; t < T; t++) {
    dphi_drho = phi[t - 1] + rho * dphi_drho;
    grad_rho += grad_phi_lik[t] * dphi_drho;
  }
  // Chain rule from rho on (-1, 1) to its sampled logit coordinate.
  double grad_logit_rho = grad_rho * ratiod_ar1::drho_dlogit(rho);

  return {grad_log_tau, grad_logit_rho};
}

// =====================================================================
// Temporal log-prior contribution
// =====================================================================

// Compute log-prior for temporal effects (single group)
inline double temporal_log_prior(
    const double* phi,
    int T,
    TemporalType type,
    double tau,           // precision for RW1/RW2, or conditional precision for AR1
    double rho,           // AR1 autocorrelation (ignored for RW)
    bool cyclic
) {
  double log_prior = 0.0;

  if (type == TemporalType::RW1) {
    // RW1: p(phi|tau) propto tau^{(T-1)/2} exp(-0.5 * tau * phi' Q phi)
    double quad = rw1_quadratic_form(phi, T, cyclic);
    int rank = tulpa::rw1_rank(T, cyclic);
    log_prior += 0.5 * rank * std::log(tau);
    log_prior -= 0.5 * tau * quad;

  } else if (type == TemporalType::RW2) {
    // RW2: p(phi|tau) propto tau^{(T-2)/2} exp(-0.5 * tau * phi' Q phi)
    double quad = rw2_quadratic_form(phi, T, cyclic);
    int rank = tulpa::rw2_rank(T, cyclic);
    log_prior += 0.5 * rank * std::log(tau);
    log_prior -= 0.5 * tau * quad;

  } else if (type == TemporalType::AR1) {
    // AR1: proper prior
    log_prior += ar1_log_density(phi, T, rho, tau);

  } else if (type == TemporalType::IID) {
    // IID N(0, 1/tau): sum of independent normal log-densities
    double sigma2 = 1.0 / tau;
    double log_norm = -0.5 * std::log(2.0 * M_PI * sigma2);
    for (int t = 0; t < T; t++) {
      log_prior += log_norm - 0.5 * phi[t] * phi[t] / sigma2;
    }
  }

  return log_prior;
}

} // namespace ratiod_temporal

#endif // QUOTR_HMC_TEMPORAL_H
