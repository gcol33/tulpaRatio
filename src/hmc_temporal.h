// hmc_temporal.h
// Temporal random effects support for HMC backend
// Supports RW1, RW2, and AR1 temporal structures

#ifndef TULPARATIO_HMC_TEMPORAL_H
#define TULPARATIO_HMC_TEMPORAL_H

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

// The joint density of grouped temporal trajectories. Intrinsic walks can
// carry one precision-scaled constant per group for sum-to-zero augmentation.
template<class T>
inline T temporal_log_prior(const T* phi, int n_times, TemporalType type,
                             const T& log_tau, const T& tau, const T& rho,
                             bool cyclic, int n_groups = 1, bool augment = false,
                             T log_prior = T(0.0)) {
  if (type == TemporalType::RW1 || type == TemporalType::RW2) {
    T quad = T(0.0);
    for (int g = 0; g < n_groups; ++g) {
      const T* w = phi + g*n_times;
      if (type == TemporalType::RW1) {
        for (int t = 1; t < n_times; ++t) {
          const T diff = w[t] - w[t-1];
          quad = quad + diff*diff;
        }
        if (cyclic && n_times > 0) {
          const T diff = w[0] - w[n_times-1];
          quad = quad + diff*diff;
        }
      } else {
        for (int t = 2; t < n_times; ++t) {
          const T diff = w[t] - T(2.0)*w[t-1] + w[t-2];
          quad = quad + diff*diff;
        }
        if (cyclic && n_times >= 3) {
          const T a = w[n_times-2] - T(2.0)*w[n_times-1] + w[0];
          const T b = w[n_times-1] - T(2.0)*w[0] + w[1];
          quad = quad + a*a + b*b;
        }
      }
    }
    T aug_quad = T(0.0);
    if (augment) for (int g = 0; g < n_groups; ++g) {
      const T sum = tulpa::s2z_component_sum(phi,g*n_times,n_times);
      aug_quad = aug_quad + tulpa::s2z_aug_coef(T(1.0),n_times)*sum*sum;
    }
    quad = quad + aug_quad;
    const int rank = (type == TemporalType::RW1 ? tulpa::rw1_rank(n_times,cyclic)
                                                : tulpa::rw2_rank(n_times,cyclic))*n_groups
                     + (augment ? n_groups : 0);
    log_prior = log_prior + T(0.5*rank)*log_tau;
    log_prior = log_prior - T(0.5)*tau*quad;
  } else if (type == TemporalType::AR1) {
    for (int g = 0; g < n_groups; ++g)
      log_prior = log_prior + ratiod_ar1::ar1_log_density(phi+g*n_times,n_times,rho,tau);
  } else if (type == TemporalType::IID) {
    T quad = T(0.0);
    for (int i = 0; i < n_times*n_groups; ++i) quad = quad + phi[i]*phi[i];
    const int n = n_times*n_groups;
    log_prior = log_prior + T(0.5*n)*log_tau - T(0.5*n*std::log(2.0*M_PI));
    log_prior = log_prior - T(0.5)*tau*quad;
  }
  return log_prior;
}

} // namespace ratiod_temporal

#endif // TULPARATIO_HMC_TEMPORAL_H
