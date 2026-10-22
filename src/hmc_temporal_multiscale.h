// hmc_temporal_multiscale.h
// Multi-scale temporal decomposition: trend + seasonal + short-term
// Builds on existing RW1/RW2/AR1 infrastructure
//
// Templated over the scalar type: double for the plain sampler, the autodiff
// types for gradient modes. One body per kernel, so the log-posterior the
// sampler evaluates and the one the autodiff paths differentiate cannot carry
// different terms.

#ifndef RATIOD_HMC_TEMPORAL_MULTISCALE_H
#define RATIOD_HMC_TEMPORAL_MULTISCALE_H

#include <vector>
#include <cmath>
#include <string>
#include "hmc_temporal.h"  // For TemporalType enum
#include "autodiff_utils.h"

#include <tulpa/sum_to_zero.h>  // rw1_rank / rw2_rank / s2z_aug_* / component sums
#include "hmc_temporal_nc.h"     // the intrinsic arms' non-centred coordinate

namespace ratiod_temporal {

using ratiod::math::safe_log;
using ratiod::math::safe_sqrt;

// TemporalType enum is defined in hmc_temporal.h

// Multi-scale temporal data structure
struct MultiscaleTemporalData {
  int n_times = 0;                   // Number of unique time points
  int n_groups = 0;                  // Number of groups (for panel data)
  int n_obs = 0;                     // Total observations

  std::vector<int> time_index;       // Time index for each observation (1-based)
  std::vector<int> group_index;      // Group index for each observation (1-based)

  // Component specifications
  TemporalType trend_type = TemporalType::NONE;    // rw1, rw2, or none
  int seasonal_period = 0;                          // 0 if no seasonal, else period (e.g., 12)
  TemporalType short_term_type = TemporalType::NONE; // ar1, iid, or none

  bool shared = true;                // Shared between num/denom

  // Whether the intrinsic arms are sampled in their non-centred coordinate:
  // the trend and seasonal blocks then hold z ~ N(0, I) and the effects are
  // sigma * A^{-1} z (hmc_temporal_nc.h). The short-term arm is proper and is
  // the same block either way.
  bool noncentered = false;
};

// The order of an arm's differencing stencil, which is what the transform and
// its adjoint are written against.
inline int ms_arm_order(TemporalType type) {
  return (type == TemporalType::RW2) ? 2 : 1;
}

// The effects an arm contributes to eta, read from the block the sampler moves
// in. One reader, called wherever a density, a gradient or the sample store
// needs the arm, so the coordinate lives in one place rather than a branch per
// site.
template <typename T>
inline void ms_arm_effects(const std::vector<T>& block, int order, bool cyclic,
                           const T& sigma2, bool noncentered,
                           std::vector<T>& out) {
  const int n = static_cast<int>(block.size());
  out.resize(n);
  if (n == 0) return;
  if (!noncentered) {
    for (int t = 0; t < n; t++) out[t] = block[t];
    return;
  }
  ratiod_temporal_nc::rw_nc_forward(block.data(), n, order, cyclic,
                                    safe_sqrt(sigma2), out.data());
}

// -----------------------------------------------------------------------------
// RW1 log-likelihood (intrinsic first-order random walk)
// -----------------------------------------------------------------------------

// Log-likelihood for RW1: sum of (phi[t] - phi[t-1])^2 / (2*sigma2).
// `augment` identifies the constant null direction by Q -> Q + 11'/n, the
// construction tulpa/sum_to_zero.h describes; see multiscale_temporal_log_lik.
template <typename T>
inline T rw1_log_lik(
    const std::vector<T>& phi,  // Length T
    const T& sigma2,
    bool cyclic = false,
    bool augment = false
) {
  int n = static_cast<int>(phi.size());
  if (n < 2) return T(0.0);

  T quad = T(0.0);

  // First differences
  for (int t = 1; t < n; t++) {
    T diff = phi[t] - phi[t - 1];
    quad = quad + diff * diff;
  }

  // Cyclic: add connection from last to first
  if (cyclic) {
    T diff = phi[0] - phi[n - 1];
    quad = quad + diff * diff;
  }

  int rank = tulpa::rw1_rank(n, cyclic);

  // RW1's null space is the constant alone, so the augmentation fills it and
  // the field becomes full rank. The quadratic and the rank move together here
  // rather than the caller adding one and this adding the other.
  if (augment) {
    const T s = tulpa::s2z_component_sum(phi.data(), 0, n);
    quad = quad + tulpa::s2z_aug_coef(T(1.0), n) * s * s;
    rank = tulpa::s2z_aug_rank(rank, 1);
  }

  return T(-0.5) * quad / sigma2
       - T(0.5 * rank) * safe_log(T(2.0 * M_PI) * sigma2);
}

// -----------------------------------------------------------------------------
// RW2 log-likelihood (intrinsic second-order random walk)
// -----------------------------------------------------------------------------

// Log-likelihood for RW2: sum of (phi[t] - 2*phi[t-1] + phi[t-2])^2 / (2*sigma2)
template <typename T>
inline T rw2_log_lik(
    const std::vector<T>& phi,  // Length T
    const T& sigma2,
    bool cyclic = false,
    bool augment = false
) {
  int n = static_cast<int>(phi.size());
  if (n < 3) return T(0.0);

  T quad = T(0.0);

  // Second differences
  for (int t = 2; t < n; t++) {
    T diff2 = phi[t] - T(2.0) * phi[t - 1] + phi[t - 2];
    quad = quad + diff2 * diff2;
  }

  // Cyclic: add wrap-around connections
  if (cyclic) {
    T diff2_1 = phi[0] - T(2.0) * phi[n - 1] + phi[n - 2];
    T diff2_2 = phi[1] - T(2.0) * phi[0] + phi[n - 1];
    quad = quad + diff2_1 * diff2_1 + diff2_2 * diff2_2;
  }

  int rank = tulpa::rw2_rank(n, cyclic);

  // Only the constant direction is filled. A non-cyclic RW2 also annihilates a
  // linear ramp, which the intercept does not absorb and a slope term does, so
  // that direction is left to the temporal covariate and the field stays
  // deficient by one: s2z_aug_rank takes rank(Q) and the number of directions
  // actually filled rather than assuming the field length.
  if (augment) {
    const T s = tulpa::s2z_component_sum(phi.data(), 0, n);
    quad = quad + tulpa::s2z_aug_coef(T(1.0), n) * s * s;
    rank = tulpa::s2z_aug_rank(rank, 1);
  }

  return T(-0.5) * quad / sigma2
       - T(0.5 * rank) * safe_log(T(2.0 * M_PI) * sigma2);
}

// -----------------------------------------------------------------------------
// AR1 log-likelihood (stationary first-order autoregressive)
// -----------------------------------------------------------------------------

// Log-likelihood for AR1: phi[t] = rho * phi[t-1] + epsilon[t]
template <typename T>
inline T ar1_log_lik(
    const std::vector<T>& phi,  // Length T
    const T& sigma2,            // Innovation variance
    const T& rho                // Autocorrelation (-1 < rho < 1)
) {
  int n = static_cast<int>(phi.size());
  if (n < 2) return T(0.0);

  T log_lik = T(0.0);

  // Marginal distribution of first observation
  T marginal_var = sigma2 / ratiod_ar1::one_minus_rho2(rho);
  log_lik = log_lik - T(0.5) * safe_log(T(2.0 * M_PI) * marginal_var);
  log_lik = log_lik - T(0.5) * phi[0] * phi[0] / marginal_var;

  // Conditional distributions
  for (int t = 1; t < n; t++) {
    T resid = phi[t] - rho * phi[t - 1];
    log_lik = log_lik - T(0.5) * safe_log(T(2.0 * M_PI) * sigma2);
    log_lik = log_lik - T(0.5) * resid * resid / sigma2;
  }

  return log_lik;
}

// -----------------------------------------------------------------------------
// IID log-likelihood (independent identically distributed)
// -----------------------------------------------------------------------------

template <typename T>
inline T iid_log_lik(
    const std::vector<T>& phi,  // Length T
    const T& sigma2
) {
  int n = static_cast<int>(phi.size());
  T log_lik = T(0.0);

  for (int t = 0; t < n; t++) {
    log_lik = log_lik - T(0.5) * safe_log(T(2.0 * M_PI) * sigma2);
    log_lik = log_lik - T(0.5) * phi[t] * phi[t] / sigma2;
  }

  return log_lik;
}

// -----------------------------------------------------------------------------
// Multi-scale temporal log-likelihood
// -----------------------------------------------------------------------------

// Combined log-likelihood for trend + seasonal + short-term
template <typename T>
inline T multiscale_temporal_log_lik(
    const std::vector<T>& trend,       // Length n_times (or empty)
    const std::vector<T>& seasonal,    // Length seasonal_period (or empty)
    const std::vector<T>& short_term,  // Length n_times (or empty)
    const T& sigma2_trend,
    const T& sigma2_seasonal,
    const T& sigma2_short,
    const T& rho_short,                // Only used if short_term is AR1
    const MultiscaleTemporalData& temp_data
) {
  T log_lik = T(0.0);

  // Trend and seasonal are both intrinsic and both enter the SAME linear
  // predictor, so each carries a constant null direction that is unidentified
  // against the intercept and against the other component. Both are augmented,
  // and compute_ms_effect_by_time centres both on the way into eta -- the two
  // halves of one construction. The short-term arm is proper (AR1/IID),
  // identifies its own level, and is left alone.

  // In the non-centred coordinate the block IS z: the arm's prior and the
  // transform's Jacobian cancel term for term, so what is left is N(0, I) and
  // sigma2 reaches the field through the transform alone.
  const bool nc = temp_data.noncentered;

  // Trend component
  if (temp_data.trend_type == TemporalType::RW1 && !trend.empty()) {
    log_lik = log_lik + (nc
        ? ratiod_temporal_nc::rw_nc_log_prior(
              trend.data(), static_cast<int>(trend.size()), 1, false)
        : rw1_log_lik(trend, sigma2_trend, false, true));
  } else if (temp_data.trend_type == TemporalType::RW2 && !trend.empty()) {
    log_lik = log_lik + (nc
        ? ratiod_temporal_nc::rw_nc_log_prior(
              trend.data(), static_cast<int>(trend.size()), 2, false)
        : rw2_log_lik(trend, sigma2_trend, false, true));
  }

  // Seasonal component (always cyclic RW1)
  if (temp_data.seasonal_period > 0 && !seasonal.empty()) {
    log_lik = log_lik + (nc
        ? ratiod_temporal_nc::rw_nc_log_prior(
              seasonal.data(), static_cast<int>(seasonal.size()), 1, true)
        : rw1_log_lik(seasonal, sigma2_seasonal, true, true));
  }

  // Short-term component
  if (temp_data.short_term_type == TemporalType::AR1 && !short_term.empty()) {
    log_lik = log_lik + ar1_log_lik(short_term, sigma2_short, rho_short);
  } else if (temp_data.short_term_type == TemporalType::IID && !short_term.empty()) {
    log_lik = log_lik + iid_log_lik(short_term, sigma2_short);
  }

  return log_lik;
}

// The two intrinsic arms' effects, read from the blocks a gradient function
// holds. The pointers are the blocks themselves in the centred coordinate and
// the transformed buffers otherwise, so a caller indexes one thing either way.
struct MSArmEffects {
  std::vector<double> trend_buf, seasonal_buf;
  const double* trend = nullptr;
  const double* seasonal = nullptr;
};

inline MSArmEffects ms_read_arms(const double* trend, int n_trend,
                                 const double* seasonal, int n_seasonal,
                                 double sigma2_trend, double sigma2_seasonal,
                                 const MultiscaleTemporalData& d) {
  MSArmEffects out;
  out.trend = trend;
  out.seasonal = seasonal;
  if (!d.noncentered) return out;
  if (trend != nullptr && n_trend > 0) {
    const std::vector<double> blk(trend, trend + n_trend);
    ms_arm_effects(blk, ms_arm_order(d.trend_type), false, sigma2_trend, true,
                   out.trend_buf);
    out.trend = out.trend_buf.data();
  }
  if (seasonal != nullptr && n_seasonal > 0 && d.seasonal_period > 0) {
    const std::vector<double> blk(seasonal, seasonal + n_seasonal);
    ms_arm_effects(blk, 1, true, sigma2_seasonal, true, out.seasonal_buf);
    out.seasonal = out.seasonal_buf.data();
  }
  return out;
}

// -----------------------------------------------------------------------------
// Compute total temporal effect at each observation
// -----------------------------------------------------------------------------

// The multiscale effect at each time point, which is what every eta assembly
// adds. The single source of truth for what the linear predictor sees: the
// templated density indexes it through compute_temporal_eta below, and each
// analytic gradient path builds it once per time point and indexes it per
// observation.
//
// The intrinsic arms are centred on their way in. Their augmented prior gives
// each constant direction the arm's own precision (order 1) rather than the
// stiff pin that preceded it, so leaving the constant in eta would free the
// level instead of fixing it -- and trend and seasonal land on the same linear
// predictor, so their constants are unidentified against each other as well as
// against the intercept. The short-term arm is proper and keeps its own level.
template <typename T>
inline void compute_ms_effect_by_time(
    const T* trend, int n_trend,
    const T* seasonal, int n_seasonal,
    const T* short_term, int n_short,
    int seasonal_period,
    int n_times,
    T* effect_by_time  // Output: length n_times
) {
  const T trend_mean = (trend != nullptr && n_trend > 0)
      ? tulpa::s2z_component_mean(trend, 0, n_trend)
      : T(0.0);
  const T seasonal_mean = (seasonal != nullptr && n_seasonal > 0)
      ? tulpa::s2z_component_mean(seasonal, 0, n_seasonal)
      : T(0.0);

  for (int t = 0; t < n_times; t++) {
    T effect = T(0.0);

    // Trend contribution
    if (trend != nullptr && t < n_trend) {
      effect = effect + (trend[t] - trend_mean);
    }

    // Seasonal contribution (wrap around using modulo)
    if (seasonal != nullptr && seasonal_period > 0) {
      int s_idx = t % seasonal_period;
      if (s_idx < n_seasonal) {
        effect = effect + (seasonal[s_idx] - seasonal_mean);
      }
    }

    // Short-term contribution
    if (short_term != nullptr && t < n_short) {
      effect = effect + short_term[t];
    }

    effect_by_time[t] = effect;
  }
}

// eta_temporal[i] = the multiscale effect at observation i's time point.
template <typename T>
inline void compute_temporal_eta(
    const std::vector<T>& trend,
    const std::vector<T>& seasonal,
    const std::vector<T>& short_term,
    const MultiscaleTemporalData& temp_data,
    std::vector<T>& eta_temporal  // Output: length n_obs
) {
  const int N = temp_data.n_obs;
  const int n_times = temp_data.n_times > 0 ? temp_data.n_times : 0;

  std::vector<T> effect_by_time(static_cast<std::size_t>(n_times), T(0.0));
  compute_ms_effect_by_time(
      trend.empty() ? nullptr : trend.data(), static_cast<int>(trend.size()),
      seasonal.empty() ? nullptr : seasonal.data(),
      static_cast<int>(seasonal.size()),
      short_term.empty() ? nullptr : short_term.data(),
      static_cast<int>(short_term.size()),
      temp_data.seasonal_period, n_times, effect_by_time.data());

  eta_temporal.assign(static_cast<std::size_t>(N), T(0.0));
  for (int i = 0; i < N; i++) {
    int t_idx = temp_data.time_index[i] - 1;  // Convert to 0-based
    if (t_idx >= 0 && t_idx < n_times) eta_temporal[i] = effect_by_time[t_idx];
  }
}

// -----------------------------------------------------------------------------
// Priors for temporal hyperparameters
// -----------------------------------------------------------------------------

// PC prior for temporal variance (favor simpler models with smaller variance)
template <typename T>
inline T log_prior_sigma2_temporal_pc(const T& sigma2, double U, double alpha) {
  T rate = T(-std::log(alpha) / U);
  T sigma = safe_sqrt(sigma2);
  return safe_log(rate) - rate * sigma - safe_log(T(2.0) * sigma);
}

} // namespace ratiod_temporal

#endif // RATIOD_HMC_TEMPORAL_MULTISCALE_H
