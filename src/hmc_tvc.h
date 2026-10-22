// hmc_tvc.h
// Temporally-Varying Coefficients (TVC) for HMC backend
// Supports RW1, RW2, AR1, and GP temporal structures for coefficients

#ifndef RATIOD_HMC_TVC_H
#define RATIOD_HMC_TVC_H

#include <vector>
#include <cmath>
#include "hmc_temporal_autodiff.h"  // Templated RW1/RW2/AR1 implementations
#include <tulpa/soft_sum_to_zero.h>  // s2z_precision

namespace ratiod_tvc {

using ratiod_temporal::TemporalType;
using ratiod_temporal::rw1_quadratic_form;
using ratiod_temporal::rw2_quadratic_form;
using ratiod_temporal::ar1_log_density;
using ratiod_temporal::rw1_quadratic_form_t;
using ratiod_temporal::rw2_quadratic_form_t;
using ratiod_temporal::ar1_log_density_t;

// TVC data structure
struct TVCData {
  int n_obs = 0;                      // Number of observations
  int n_times = 0;                    // Number of unique time points
  int n_tvc = 0;                      // Number of TVC terms
  int n_groups = 1;                   // Number of groups (for panel data)

  std::vector<int> time_index;        // Maps obs to time point (1-based)
  std::vector<int> group_index;       // Maps obs to group (1-based)
  std::vector<int> tvc_indices;       // Which design matrix columns have TVCs

  std::vector<double> X_tvc;          // Design matrix subset for TVC (n_obs x n_tvc)

  TemporalType structure;             // RW1, RW2, AR1, or GP
  bool shared = false;                // Whether TVC is shared between num/denom
  bool cyclic = false;                // Whether temporal structure is cyclic
};

// Scratch buffers for compute_gradient_tvc_handcoded. Held per-thread (via
// RATIOD_TLS_WORKSPACE at the call site), never as members of TVCData: TVCData
// hangs off the single ModelData that every chain thread shares, so buffers
// living there are the same memory for every chain and a concurrent write from
// one chain overwrites what another chain's gradient evaluation is reading.
struct TVCGradWorkspace {
  std::vector<double> tau;           // size n_tvc
  std::vector<double> rho;           // size n_tvc
  std::vector<double> w_flat;        // size n_groups * n_tvc * n_times
  std::vector<double> eta;           // size n_obs
  std::vector<double> grad_w;        // size n_groups * n_tvc * n_times
  std::vector<double> grad_log_tau;  // size n_tvc
  std::vector<double> grad_logit_rho;// size n_tvc
  std::vector<double> grad_w_jg;     // size n_times (reused per group-term)
  std::vector<double> d_buf;         // size n_times (RW2 second differences)

  void resize(int n_tvc, int n_times, int n_groups, int n_obs) {
    int n_w = n_groups * n_tvc * n_times;
    tau.resize(n_tvc);
    rho.resize(n_tvc);
    w_flat.resize(n_w);
    eta.resize(n_obs);
    grad_w.resize(n_w);
    grad_log_tau.resize(n_tvc);
    grad_logit_rho.resize(n_tvc);
    grad_w_jg.resize(n_times);
    d_buf.resize(n_times);
  }
};

// -----------------------------------------------------------------------------
// TVC log-prior
// -----------------------------------------------------------------------------

// Compute log-prior for a single TVC term's temporal trajectory
// w: temporal trajectory (length n_times)
// tau: precision parameter
// rho: AR1 correlation (only used if structure == AR1)
template <typename T>
inline T tvc_term_log_prior(
    const T* w,
    int n_times,
    TemporalType structure,
    const T& tau,
    const T& rho,
    bool cyclic = false
) {
  using ratiod::math::safe_log;
  T log_prior = T(0.0);

  if (structure == TemporalType::RW1) {
    T quad = rw1_quadratic_form_t(w, n_times, cyclic);
    log_prior = log_prior + T(0.5 * tulpa::rw1_rank(n_times, cyclic)) * safe_log(tau)
                          - T(0.5) * tau * quad;

  } else if (structure == TemporalType::RW2) {
    T quad = rw2_quadratic_form_t(w, n_times, cyclic);
    log_prior = log_prior + T(0.5 * tulpa::rw2_rank(n_times, cyclic)) * safe_log(tau)
                          - T(0.5) * tau * quad;

  } else if (structure == TemporalType::AR1) {
    log_prior = log_prior + ar1_log_density_t(w, n_times, rho, tau);

  } else if (structure == TemporalType::IID) {
    // IID: independent N(0, 1/tau) for each time point
    T quad = T(0.0);
    for (int t = 0; t < n_times; t++) quad = quad + w[t] * w[t];
    log_prior = log_prior + T(0.5 * n_times) * safe_log(tau)
                          - T(0.5) * tau * quad;
  }

  return log_prior;
}

// Compute log-prior for all TVC terms
// w_flat: all TVC values (n_times * n_tvc * n_groups, flattened)
// tau: vector of precisions (length n_tvc)
// rho: vector of AR1 correlations (length n_tvc, only for AR1)
template <typename T>
inline T tvc_log_prior(
    const std::vector<T>& w_flat,
    const TVCData& tvc_data,
    const std::vector<T>& tau,
    const std::vector<T>& rho
) {
  const int n_times = tvc_data.n_times;
  const int n_tvc = tvc_data.n_tvc;
  const int n_groups = tvc_data.n_groups;

  T log_prior = T(0.0);

  // Layout: w_flat[g * n_tvc * n_times + j * n_times + t]
  for (int g = 0; g < n_groups; g++) {
    for (int j = 0; j < n_tvc; j++) {
      const T* w_jg = &w_flat[(g * n_tvc + j) * n_times];
      T rho_j = (tvc_data.structure == TemporalType::AR1) ? rho[j] : T(0.0);
      log_prior = log_prior + tvc_term_log_prior(w_jg, n_times, tvc_data.structure,
                                                 tau[j], rho_j, tvc_data.cyclic);
    }
  }

  return log_prior;
}

// -----------------------------------------------------------------------------
// TVC contribution to linear predictor
// -----------------------------------------------------------------------------

// Compute TVC contribution to linear predictor for all observations
// eta_tvc[i] = sum_j X_tvc[i,j] * w[time_index[i], j, group_index[i]]
template <typename T>
inline void compute_tvc_eta(
    const std::vector<T>& w_flat,  // n_groups * n_tvc * n_times
    const TVCData& tvc_data,
    std::vector<T>& eta_tvc         // Output: length n_obs
) {
  const int N = tvc_data.n_obs;
  const int n_times = tvc_data.n_times;
  const int n_tvc = tvc_data.n_tvc;

  eta_tvc.assign(N, T(0.0));

  for (int i = 0; i < N; i++) {
    const int t = tvc_data.time_index[i] - 1;  // 0-based
    const int g = tvc_data.group_index[i] - 1;  // 0-based

    for (int j = 0; j < n_tvc; j++) {
      // w_flat layout: [g * n_tvc * n_times + j * n_times + t]
      eta_tvc[i] = eta_tvc[i] + T(tvc_data.X_tvc[i * n_tvc + j])
                              * w_flat[(g * n_tvc + j) * n_times + t];
    }
  }
}

// -----------------------------------------------------------------------------
// Sum-to-zero constraint for identifiability
// -----------------------------------------------------------------------------

// Apply soft sum-to-zero constraint to TVC (for each term and group). Each
// pinned sum runs over the n_times coefficients of one (group, term), so the
// precision is s2z_precision(n_times); it is derived here rather than taken
// from the caller so no call site can pass a kappa where a precision is meant.
template <typename T>
inline T tvc_sum_to_zero_penalty(
    const std::vector<T>& w_flat,
    const TVCData& tvc_data
) {
  int n_times = tvc_data.n_times;
  int n_tvc = tvc_data.n_tvc;
  int n_groups = tvc_data.n_groups;

  const double lambda = tulpa::s2z_precision(n_times);
  T penalty = T(0.0);

  for (int g = 0; g < n_groups; g++) {
    for (int j = 0; j < n_tvc; j++) {
      T sum = T(0.0);
      for (int t = 0; t < n_times; t++) {
        sum = sum + w_flat[(g * n_tvc + j) * n_times + t];
      }
      penalty = penalty - T(0.5 * lambda) * sum * sum;
    }
  }

  return penalty;
}

// -----------------------------------------------------------------------------
// Prior on TVC hyperparameters
// -----------------------------------------------------------------------------

// Log prior for TVC precision (PC prior style)
// Favors smaller variance = simpler (more constant) coefficients
// d/d(log_tau) of the Gamma(shape, rate) prior on tau taken with the Jacobian of
// the log transform, i.e. of (shape-1)*log_tau - rate*tau + log_tau. Derived
// once so the analytic gradient paths cannot drift from the density again.
inline double log_prior_tau_gamma_grad(double tau, double shape, double rate) {
  return shape - rate * tau;
}

// -----------------------------------------------------------------------------
// Gradient helpers (for HMC)
// -----------------------------------------------------------------------------

// Gradient of RW1 log-prior w.r.t. w
inline void rw1_gradient(
    const double* w,
    int n_times,
    double tau,
    double* grad_w
) {
  // d/dw_t [0.5 * tau * sum((w_{t+1} - w_t)^2)]
  // = tau * (2*w_t - w_{t-1} - w_{t+1}) for interior
  // = tau * (w_t - w_{t+1}) for t=0
  // = tau * (w_t - w_{t-1}) for t=T-1

  for (int t = 0; t < n_times; t++) {
    if (t == 0) {
      grad_w[t] = -tau * (w[1] - w[0]);
    } else if (t == n_times - 1) {
      grad_w[t] = -tau * (w[t] - w[t-1]);
    } else {
      grad_w[t] = -tau * (2.0 * w[t] - w[t-1] - w[t+1]);
    }
  }
}

// Gradient of RW2 log-prior w.r.t. w
inline void rw2_gradient(
    const double* w,
    int n_times,
    double tau,
    double* grad_w
) {
  // Second difference: d_t = w_t - 2*w_{t+1} + w_{t+2}
  // Quadratic form: sum(d_t^2)
  // Gradient is more complex, compute numerically if needed

  // For simplicity, use finite differences
  std::vector<double> w_copy(w, w + n_times);
  double eps = 1e-6;

  // Base quadratic form
  double base_quad = rw2_quadratic_form(w, n_times, false);

  for (int t = 0; t < n_times; t++) {
    w_copy[t] = w[t] + eps;
    double quad_plus = rw2_quadratic_form(w_copy.data(), n_times, false);
    grad_w[t] = -tau * (quad_plus - base_quad) / eps;
    w_copy[t] = w[t];
  }
}

// The structures a time-varying coefficient can follow
inline constexpr ratiod_enum::EnumName<TemporalType> TVC_STRUCTURE_NAMES[] = {
  {"rw1", TemporalType::RW1}, {"rw2", TemporalType::RW2},
  {"ar1", TemporalType::AR1}, {"iid", TemporalType::IID}};

inline TemporalType parse_tvc_structure(const std::string& s) {
  return ratiod_enum::parse_enum(TVC_STRUCTURE_NAMES, s, "time-varying coefficient structure");
}

} // namespace ratiod_tvc

#endif // RATIOD_HMC_TVC_H
