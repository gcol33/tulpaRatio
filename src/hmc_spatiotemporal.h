// hmc_spatiotemporal.h
// Spatiotemporal interaction effects for HMC backend
// Supports Type I-IV interactions (Knorr-Held) and separable/non-separable GP

#ifndef RATIOD_HMC_SPATIOTEMPORAL_H
#define RATIOD_HMC_SPATIOTEMPORAL_H

#include <vector>
#include <cmath>
#include <limits>
#include "hmc_cov.h"
#include <tulpa/st_gp_kernel.h>
#include <tulpa/st_margins.h>
#include <tulpa/st_data.h>
#include "hmc_temporal.h"
#include "hmc_svc.h"
#include "autodiff_utils.h"
#include "ar1_shared.h"
#include <tulpa/soft_sum_to_zero.h>  // s2z_precision
#include "hmc_temporal_autodiff.h"
#include "hmc_svc_autodiff.h"

namespace ratiod_spatiotemporal {

using ratiod_temporal::TemporalType;
using ratiod_svc::CovType;
using ratiod_svc_ad::compute_cov;
using namespace ratiod::math;

// =====================================================================
// Spatiotemporal interaction types
// =====================================================================

enum class STType {
  NONE,
  TYPE_I,      // IID interaction
  TYPE_II,     // Structured time, unstructured space
  TYPE_III,    // Structured space, unstructured time
  TYPE_IV,     // Fully structured (Kronecker)
  SEPARABLE,   // Separable GP
  NONSEP_GP    // Non-separable GP
};

// The rank the interaction's time margin contributes to the GMRF normalizer,
// which is what multiplies log(tau). RW1 and RW2 are improper and contribute
// their own rank; AR1 is proper on (-1, 1) and contributes the full T, with its
// rho-dependent log|R| carried alongside rather than folded in here. Every site
// that normalizes an interaction reads this one function -- the density, the
// non-centered rank correction and both hand-coded gradients -- because a rank
// the density and its gradient disagree on is a tau posterior that moves under
// a Hamiltonian nothing scores. A margin the interaction cannot express
// contributes no quadratic form, so it contributes no normalizer either --
// the two halves of the same prior.
inline int st_time_rank(TemporalType type, int T, bool cyclic) {
  switch (type) {
    case TemporalType::RW1: return tulpa::rw1_rank(T, cyclic);
    case TemporalType::RW2: return tulpa::rw2_rank(T, cyclic);
    case TemporalType::AR1: return (T >= 2) ? T : 0;
    default:                return 0;
  }
}

// Non-separability types for the GP interaction. Two of the four names this
// carried are gone, for different reasons.
//
// Cressie-Huang (1999) had no branch of its own in the covariance and was
// evaluated as PRODUCT. A class that silently means another one is worse than
// an absent option.
//
// The ADDITIVE kernel C_s(h) + C_t(u) cannot be an interaction covariance on a
// complete grid. Its rank there is S + T - 1 out of S * T -- for any four cells
// (s1,t1), (s1,t2), (s2,t1), (s2,t2) the vector (1, -1, -1, 1) is in its null
// space, and that vector IS an interaction. So it assigns the block it is a
// prior for zero variance in every direction the block exists to carry, and
// what a neighbour Cholesky then factorizes is the ridge. Measured on a 6 x 5
// grid: rank 10 of 30, smallest eigenvalue -1.6e-15, against a product kernel
// at condition number 8.4 and a Gneiting kernel at 10.8.
enum class NonsepType {
  PRODUCT,   // C_s(h) * C_t(u), the separable arm
  GNEITING   // Gneiting (2002) class
};

// =====================================================================
// Data structures
// =====================================================================

// Main spatiotemporal data container
struct SpatiotemporalData {
  tulpa::GraphPartition spatial_partition;
  STType type;
  bool shared;

  int n_spatial;           // Number of spatial units (S)
  int n_times;             // Number of time points (T)
  int n_params;            // Total interaction parameters (S * T for most types)

  // Observation indexing (length N)
  std::vector<int> s_idx;  // Spatial index for each obs (1-based)
  std::vector<int> t_idx;  // Temporal index for each obs (1-based)
  std::vector<int> st_flat; // Flattened index: (s-1)*T + t

  // Spatial structure (for Type II, III, IV)
  bool spatial_is_gp;
  // CAR structure
  std::vector<int> adj_row_ptr;   // CSR row pointers
  std::vector<int> adj_col_idx;   // CSR column indices
  std::vector<int> n_neighbors;   // Neighbors per spatial unit
  bool spatial_proper;             // Proper CAR vs ICAR
  double bym2_scale;               // BYM2 scaling factor

  // Temporal structure (for Type II, III, IV)
  TemporalType temporal_type;
  bool temporal_cyclic;

  // GP parameters (SEPARABLE / NONSEP_GP). Everything here is indexed by the
  // interaction's own field index k, which runs over n_params -- the S x T grid
  // for spatiotemporal(), one entry per observation for spatiotemporal_gp() --
  // and NOT by observation. st_gp_nngp_scan reads coords and time_values at the
  // entries of nn_order, which are field indices.
  //
  // Every member carries a default, because the enum arms and nn are read
  // before anything can check whether the structure was filled: ModelData is
  // default-initialized, so a member without one is indeterminate rather than
  // unset. st_gp_structure_filled() is the check itself.
  NonsepType nonsep_type = NonsepType::PRODUCT;
  CovType cov_space = CovType::EXPONENTIAL;
  CovType cov_time = CovType::EXPONENTIAL;
  int nn = 0;                      // NNGP neighbours per location
  std::vector<double> coords;      // n_params x 2, row-major
  std::vector<double> time_values; // n_params
  std::vector<int> nn_idx;         // n_params x nn, 1-based within the ordering, 0 absent
  std::vector<double> nn_dist_space; // n_params x nn
  std::vector<double> nn_dist_time;  // n_params x nn
  std::vector<int> nn_order;       // ordering position -> field index, 0-based
  std::vector<int> nn_order_inv;   // field index -> ordering position, 0-based

  // Prior parameters
  double sigma2_prior_U;           // PC prior: P(sigma > U) = alpha
  double sigma2_prior_alpha;
  // Beta(a, b) on u = (rho + 1) / 2 for an AR1 time margin, the same
  // parameterization ratiod_ar1 reads everywhere else.
  double rho_prior_a = ratiod_ar1::RHO_PRIOR_A;
  double rho_prior_b = ratiod_ar1::RHO_PRIOR_B;
  double phi_space_prior_lower;    // Uniform bounds for spatial range
  double phi_space_prior_upper;
  double phi_time_prior_lower;     // Uniform bounds for temporal range
  double phi_time_prior_upper;

  // HSGP-ST: spectral basis spatiotemporal interaction
  bool is_hsgp = false;            // If true, spatial precision is spectral (diagonal)
  int hsgp_m_total = 0;            // m^2 basis functions
};

// =====================================================================
// Type I: IID interaction
// =====================================================================

// Log-prior for Type I (IID) interaction
// delta[s,t] ~ N(0, sigma2) independently
template<typename Scalar>
inline Scalar type_i_log_prior(
    const Scalar* delta,  // Flattened S*T vector
    int S,
    int T,
    const Scalar& tau  // precision = 1/sigma2
) {
  int n = S * T;
  Scalar quad = Scalar(0.0);

  for (int i = 0; i < n; i++) {
    quad = quad + delta[i] * delta[i];
  }

  // log p(delta|tau) = (n/2) log(tau) - (tau/2) sum(delta^2) + const
  return Scalar(0.5 * n) * safe_log(tau) - Scalar(0.5) * tau * quad;
}

// =====================================================================
// Type II: Structured time at each location
// =====================================================================

// Log-prior for Type II interaction
// For each spatial unit s, delta[s,*] follows temporal structure
template<typename Scalar>
inline Scalar type_ii_log_prior(
    const Scalar* delta,  // Flattened S*T vector (column-major: delta[s + S*t])
    int S,
    int T,
    const Scalar& tau,
    const Scalar& rho,    // AR1 autocorrelation; unread by the RW margins
    TemporalType temp_type,
    bool cyclic
) {
  Scalar log_prior = Scalar(0.0);

  // For each spatial unit
  for (int s = 0; s < S; s++) {
    // Extract temporal series for this location
    std::vector<Scalar> delta_s(T);
    for (int t = 0; t < T; t++) {
      delta_s[t] = delta[s * T + t];  // Column-major
    }

    // Apply temporal prior
    if (temp_type == TemporalType::RW1) {
      Scalar quad = ratiod_temporal::rw1_quadratic_form_t(delta_s, T, cyclic);
      int rank = tulpa::rw1_rank(T, cyclic);
      log_prior = log_prior + Scalar(0.5 * rank) * safe_log(tau)
                            - Scalar(0.5) * tau * quad;
    } else if (temp_type == TemporalType::RW2) {
      Scalar quad = ratiod_temporal::rw2_quadratic_form_t(delta_s, T, cyclic);
      int rank = tulpa::rw2_rank(T, cyclic);
      log_prior = log_prior + Scalar(0.5 * rank) * safe_log(tau)
                            - Scalar(0.5) * tau * quad;
    } else if (temp_type == TemporalType::AR1) {
      // Proper on (-1, 1), so the whole T-dimensional normalizer applies and
      // carries log|R(rho)| with it.
      log_prior = log_prior +
                  ratiod_ar1::ar1_log_prior(delta_s.data(), T, rho, tau);
    }
  }

  return log_prior;
}

// =====================================================================
// Type III: Structured space at each time point
// =====================================================================

// Compute ICAR quadratic form for a single vector
template<typename Scalar>
inline Scalar icar_quad_form(
    const Scalar* phi,
    int n,
    const std::vector<int>& adj_row_ptr,
    const std::vector<int>& adj_col_idx
) {
  Scalar quad = Scalar(0.0);

  for (int i = 0; i < n; i++) {
    for (int idx = adj_row_ptr[i]; idx < adj_row_ptr[i + 1]; idx++) {
      int j = adj_col_idx[idx] - 1;  // Convert to 0-based
      if (j > i) {  // Only count each edge once
        Scalar diff = phi[i] - phi[j];
        quad = quad + diff * diff;
      }
    }
  }

  return quad;
}

// Log-prior for Type III interaction
// For each time point t, delta[*,t] follows spatial structure (ICAR)
template<typename Scalar>
inline Scalar type_iii_log_prior(
    const Scalar* delta,  // Flattened S*T vector
    int S,
    int T,
    const Scalar& tau,
    const std::vector<int>& adj_row_ptr,
    const std::vector<int>& adj_col_idx,
    int rank_deficiency
) {
  Scalar log_prior = Scalar(0.0);
  int rank = S - rank_deficiency;

  // For each time point
  for (int t = 0; t < T; t++) {
    // Extract spatial field for this time point
    std::vector<Scalar> delta_t(S);
    for (int s = 0; s < S; s++) {
      delta_t[s] = delta[s * T + t];  // Column-major
    }

    // Apply ICAR prior
    Scalar quad = icar_quad_form(delta_t.data(), S, adj_row_ptr, adj_col_idx);
    log_prior = log_prior + Scalar(0.5 * rank) * safe_log(tau)
                          - Scalar(0.5) * tau * quad;
  }

  return log_prior;
}

// =====================================================================
// Type IV: Fully structured (Kronecker)
// =====================================================================

// Log-prior for Type IV interaction
// Precision Q_delta = Q_s (x) Q_t (Kronecker product)
// Quadratic form: delta' Q_delta delta = sum over (s1,t1,s2,t2) of delta[s1,t1] * Q_s[s1,s2] * Q_t[t1,t2] * delta[s2,t2]
// This can be computed as: sum_t1,t2 Q_t[t1,t2] * (delta[*,t1]' Q_s delta[*,t2])
template<typename Scalar>
inline Scalar type_iv_log_prior(
    const Scalar* delta,  // Flattened S*T vector (column-major)
    int S,
    int T,
    const Scalar& tau_space,
    const Scalar& tau_time,
    const Scalar& rho,    // AR1 autocorrelation of the time margin
    const std::vector<int>& adj_row_ptr,
    const std::vector<int>& adj_col_idx,
    TemporalType temp_type,
    bool cyclic,
    int rank_space
) {
  const Scalar quad = tulpa::st_kronecker_quadratic(delta, S, T,
      adj_row_ptr, adj_col_idx, [&](const std::vector<Scalar>& difference) {
          if (temp_type == TemporalType::RW1)
              return ratiod_temporal::rw1_quadratic_form_t(difference, T, cyclic);
          if (temp_type == TemporalType::RW2)
              return ratiod_temporal::rw2_quadratic_form_t(difference, T, cyclic);
          return ratiod_ar1::ar1_quadratic_form(difference.data(), T, rho);
      });

  // Compute log-determinant contribution
  // For Kronecker: log|Q_s (x) Q_t| = rank(Q_t)*log|Q_s|_+ + rank(Q_s)*log|Q_t|_+
  // The spatial factor's own log-determinant is free of both hyperparameters and
  // is dropped with the other constants; the AR1 time margin's is not, since
  // log|R(rho)| = log(1 - rho^2) moves with the sampled correlation.

  int rank_time = st_time_rank(temp_type, T, cyclic);

  int total_rank = rank_space * rank_time;

  Scalar log_prior = Scalar(0.5 * total_rank) * safe_log(tau_space * tau_time);
  if (temp_type == TemporalType::AR1 && T >= 2) {
    log_prior = log_prior + Scalar(0.5 * rank_space) *
                            safe_log(ratiod_ar1::one_minus_rho2(rho));
  }
  log_prior = log_prior - Scalar(0.5) * tau_space * tau_time * quad;

  return log_prior;
}

// =====================================================================
// The spatiotemporal GP covariance
// =====================================================================

// Every arm is written as a CORRELATION r(h, u) with r(0, 0) == 1, and the one
// place a covariance is formed multiplies it by sigma2. Three things follow.
// The marginal variance is sigma2 whichever non-separability is chosen, which
// is what the PC prior on sigma is a prior on. The neighbour block's diagonal
// is the kernel's own value at zero separation instead of a constant written
// beside it, so the matrix is the one the kernel defines. And sigma2 factors
// out of the block entirely, which is what lets its gradient be written in
// closed form with no extra solve (st_prior_grad.h).
//
// The kernels themselves are ratiod_cov's, at unit variance -- the same four
// the spatial GP, the SVC and their templated twins read, so a range prior
// means the same thing on this field as on those.

// Gneiting (2002) eq. (14) at a = c = tau = 1, alpha = 1, gamma = 1/2, beta = 1:
//
//   r(h, u) = base^-1 exp(-hs / sqrt(base)),   base = ut^2 + 1
//
// with hs = h / phi_space and ut = |u| / phi_time. The four shape constants are
// fixed rather than estimated, which is what the range derivatives below are
// written for.
using tulpa_st_gp::gneiting_corr;
using tulpa_st_gp::st_corr;
using tulpa_st_gp::st_cov;
using tulpa_st_gp::st_corr_dphi;
using tulpa_st_gp::st_pair_separation;
using tulpa_st_gp::st_gp_structure_filled;
using tulpa_st_gp::st_gp_nngp_scan;
using tulpa_st_gp::st_gp_nc_forward;
using tulpa_st_gp::st_gp_nngp_log_lik;
using tulpa_st_gp::emit_margin_factor;
using tulpa_st_gp::StGpAutoregression;
using tulpa_st_gp::st_gp_autoregression_back;
using tulpa_st_gp::st_gp_margin_factor;
using tulpa_st_gp::st_gp_margin_half_logdet;
using tulpa_st_gp::st_gp_margin_log_normalizer;
using tulpa_st_gp::StGpMarginNormalizer;
using tulpa_st_gp::st_gp_margin_normalizer_grad;

// =====================================================================
// Knorr-Held precision stencils
// =====================================================================

// The rank of the interaction's GMRF precision Q, the coefficient of
// 0.5 * log(tau) in its normalizer: S*T for the iid Type I, S copies of the
// time margin for Type II, T copies of the ICAR margin for Type III and the
// product of the two margins' ranks for Type IV.
inline int st_gmrf_rank(const SpatiotemporalData& st) {
  const int S = st.n_spatial;
  const int T = st.n_times;
  const int rank_time = st_time_rank(st.temporal_type, T, st.temporal_cyclic);
  switch (st.type) {
    case STType::TYPE_I:   return S * T;
    case STType::TYPE_II:  return S * rank_time;
    case STType::TYPE_III: return (S - tulpa::st_partition_count(st.spatial_partition)) * T;
    case STType::TYPE_IV:  return (S - tulpa::st_partition_count(st.spatial_partition)) * rank_time;
    default:               return 0;
  }
}

// Spatial sums constrain each graph component at each time; temporal sums
// constrain each site's constant. Their intersection has one constant per
// component. These coordinates carry a tau-free prior when noncentered.
inline int st_constrained_null_dim(const SpatiotemporalData& st) {
  const int S = st.n_spatial;
  const int T = st.n_times;
  const bool time_null = st.temporal_type == TemporalType::RW1 ||
                         st.temporal_type == TemporalType::RW2;
  const int components = tulpa::st_partition_count(st.spatial_partition);
  switch (st.type) {
    case STType::TYPE_I:   return 0;
    case STType::TYPE_II:  return time_null ? S : 0;
    case STType::TYPE_III: return components * T;
    case STType::TYPE_IV:  return components * T + (time_null ? S - components : 0);
    default:               return 0;
  }
}

// The power of tau in the non-centered interaction's density, times two:
// the GMRF normalizer tau^(rank/2), the Jacobian tau^(-ST/2) of
// delta = z / sqrt(tau), and back the directions the sum-to-zero penalty on z
// pins (st_constrained_null_dim). Zero whenever the constraints pin the whole
// null space.
inline int st_noncentered_tau_power(const SpatiotemporalData& st) {
  return st_gmrf_rank(st) + st_constrained_null_dim(st) - st.n_params;
}

// Twice the centered log-tau correction: the constraints remove S + C(T-1)
// coordinates for C spatial components. Only the constrained null directions
// were absent from the unconstrained precision's rank.
inline int st_centered_tau_correction(const SpatiotemporalData& st) {
  const int components = tulpa::st_partition_count(st.spatial_partition);
  const int n_constraints = st.n_spatial + components * (st.n_times - 1);
  return st_constrained_null_dim(st) - n_constraints;
}

// Whether the interaction's time margin is an AR1 the margin penalty pins
// proper directions of.
inline bool st_ar1_margin_pinned(const SpatiotemporalData& st) {
  return (st.type == STType::TYPE_II || st.type == STType::TYPE_IV) &&
         st.temporal_type == TemporalType::AR1 && st.n_times >= 2;
}

// The AR1 constraint normalizer counts S-C spatial contrasts for C components.
// The covariance sum gives 1' R^-1 1 through its stationary variance factor;
// Type II also constrains one proper temporal copy in the spatial mean.
template <typename Scalar>
inline Scalar st_ar1_margin_log_normalizer(const Scalar& rho,
                                           const SpatiotemporalData& st) {
  if (!st_ar1_margin_pinned(st)) return Scalar(0.0);
  const int S = st.n_spatial;
  const int contrasts = S - tulpa::st_partition_count(st.spatial_partition);
  const double n_log_det = (st.type == STType::TYPE_II) ? S : contrasts;
  return Scalar(0.5 * contrasts) *
             safe_log(ratiod_ar1::correlation_sum(rho, st.n_times)) -
         Scalar(0.5 * n_log_det) * safe_log(ratiod_ar1::one_minus_rho2(rho));
}

inline double st_ar1_margin_log_normalizer_drho(double rho,
                                                const SpatiotemporalData& st) {
  if (!st_ar1_margin_pinned(st)) return 0.0;
  const int S = st.n_spatial;
  const int contrasts = S - tulpa::st_partition_count(st.spatial_partition);
  const double n_log_det = (st.type == STType::TYPE_II) ? S : contrasts;
  return 0.5 * contrasts * ratiod_ar1::dcorrelation_sum_drho(rho, st.n_times) /
             ratiod_ar1::correlation_sum(rho, st.n_times) -
         0.5 * n_log_det * ratiod_ar1::dlog_one_minus_rho2_drho(rho);
}

// The normalizer the interaction's density carries for its margin penalty, on
// delta (the centred and the collapsed coordinates, and the non-centred GP,
// whose penalty stays on delta). A GP type's covariance is proper in every
// direction the penalty pins, so its normalizer is the full
// st_gp_margin_log_normalizer, which moves with the ranges; a Knorr-Held type
// keeps the count of pinned directions st_centered_tau_correction gives, plus
// the rho term of an AR1 time margin (st_ar1_margin_log_normalizer).
// Returns false where the GP term cannot be formed.
template <typename Scalar>
inline bool st_margin_normalizer(const Scalar& tau, const Scalar& rho,
                                 const Scalar& phi_space,
                                 const Scalar& phi_time,
                                 const SpatiotemporalData& st, Scalar* out) {
  if (st.type == STType::SEPARABLE || st.type == STType::NONSEP_GP) {
    return st_gp_margin_log_normalizer<Scalar>(Scalar(1.0) / tau, phi_space,
                                               phi_time, st, out);
  }
  *out = Scalar(0.5 * st_centered_tau_correction(st)) * safe_log(tau) +
         st_ar1_margin_log_normalizer(rho, st);
  return true;
}

// Whether the interaction is sampled non-centered: z with delta = z / sqrt(tau),
// for the Knorr-Held GMRF types. The spectral (HSGP) margin and the GP types
// are always centered.
inline bool st_noncentered(int parameterization, bool is_hsgp,
                           const SpatiotemporalData& st) {
  if (parameterization != 1 || is_hsgp) return false;
  return st.type == STType::TYPE_I || st.type == STType::TYPE_II ||
         st.type == STType::TYPE_III || st.type == STType::TYPE_IV ||
         st.type == STType::SEPARABLE || st.type == STType::NONSEP_GP;
}

// The two non-centred coordinates: a Knorr-Held GMRF samples z with
// delta = z / sqrt(tau), a GP interaction z with delta the NNGP
// autoregression run forward from z (st_gp_nc_forward).
inline bool st_noncentered_gp(int parameterization, bool is_hsgp,
                              const SpatiotemporalData& st) {
  return st_noncentered(parameterization, is_hsgp, st) &&
         (st.type == STType::SEPARABLE || st.type == STType::NONSEP_GP);
}

inline bool st_noncentered_gmrf(int parameterization, bool is_hsgp,
                                const SpatiotemporalData& st) {
  return st_noncentered(parameterization, is_hsgp, st) &&
         !st_noncentered_gp(parameterization, is_hsgp, st);
}

// out[s,t] = (Q_t x[s,:])_t for every unit s: the time margin's precision
// (RW1, cyclic or not; RW2; AR1 at rho) applied within each unit. The field
// is stored unit-major, x[s * T + t].
inline void st_apply_time_margin(const SpatiotemporalData& st, const double* x,
                                 double rho, double* out) {
  const int S = st.n_spatial;
  const int T = st.n_times;
  for (int s = 0; s < S; s++) {
    const double* xs = &x[s * T];
    double* os = &out[s * T];
    if (st.temporal_type == TemporalType::RW1) {
      for (int t = 0; t < T; t++) {
        double v = 0.0;
        if (t > 0) v += xs[t] - xs[t - 1];
        else if (st.temporal_cyclic && T > 1) v += xs[0] - xs[T - 1];
        if (t < T - 1) v += xs[t] - xs[t + 1];
        else if (st.temporal_cyclic && T > 1) v += xs[T - 1] - xs[0];
        os[t] = v;
      }
    } else if (st.temporal_type == TemporalType::RW2) {
      for (int t = 0; t < T; t++) os[t] = 0.0;
      for (int k = 0; T >= 3 && k < (st.temporal_cyclic ? T : T - 2); k++) {
        const int next = (k + 1) % T, after = (k + 2) % T;
        const double d2 = xs[k] - 2.0 * xs[next] + xs[after];
        os[k] += d2;
        os[next] -= 2.0 * d2;
        os[after] += d2;
      }
    } else if (st.temporal_type == TemporalType::AR1 && T >= 2) {
      ratiod_ar1::ar1_precision_apply(xs, T, rho, os);
    } else {
      for (int t = 0; t < T; t++) os[t] = 0.0;
    }
  }
}

// out[s,t] = (Q_s x[:,t])_s for every time t: the ICAR precision of the
// adjacency applied within each time point.
inline void st_apply_space_margin(const SpatiotemporalData& st, const double* x,
                                  double* out) {
  const int S = st.n_spatial;
  const int T = st.n_times;
  for (int s = 0; s < S; s++) {
    const int n_neigh = st.adj_row_ptr[s + 1] - st.adj_row_ptr[s];
    for (int t = 0; t < T; t++) {
      double v = n_neigh * x[s * T + t];
      for (int idx = st.adj_row_ptr[s]; idx < st.adj_row_ptr[s + 1]; idx++) {
        v -= x[(st.adj_col_idx[idx] - 1) * T + t];
      }
      out[s * T + t] = v;
    }
  }
}

// =====================================================================
// Master function: spatiotemporal log-prior
// =====================================================================

template<typename Scalar>
inline Scalar spatiotemporal_log_prior(
    const Scalar* delta,
    const Scalar& tau,
    const Scalar& tau2,        // Second precision (for Type IV: temporal)
    const Scalar& rho,         // AR1 autocorrelation if needed
    const Scalar& phi_space,   // GP range parameters
    const Scalar& phi_time,
    const SpatiotemporalData& st_data
) {
  if (st_data.type == STType::NONE) {
    return Scalar(0.0);
  }

  int S = st_data.n_spatial;
  int T = st_data.n_times;

  switch (st_data.type) {
    case STType::TYPE_I:
      return type_i_log_prior(delta, S, T, tau);

    case STType::TYPE_II:
      return type_ii_log_prior(delta, S, T, tau, rho,
                               st_data.temporal_type, st_data.temporal_cyclic);

    case STType::TYPE_III:
      return type_iii_log_prior(delta, S, T, tau,
          st_data.adj_row_ptr, st_data.adj_col_idx,
          tulpa::st_partition_count(st_data.spatial_partition));

    case STType::TYPE_IV:
      return type_iv_log_prior(delta, S, T, tau, tau2, rho,
                               st_data.adj_row_ptr, st_data.adj_col_idx,
          st_data.temporal_type, st_data.temporal_cyclic,
          S - tulpa::st_partition_count(st_data.spatial_partition));

    case STType::SEPARABLE:
    case STType::NONSEP_GP: {
      // tau is the field's precision, so 1/tau is the marginal variance the
      // correlation is scaled by.
      const Scalar sigma2 = Scalar(1.0) / tau;
      return st_gp_nngp_log_lik(delta, sigma2, phi_space, phi_time, st_data);
    }

    default:
      return Scalar(0.0);
  }
}

// =====================================================================
// Gradient helpers (numerical)
// =====================================================================

inline void spatiotemporal_gradient_delta(
    const double* delta,
    double tau,
    double tau2,
    double rho,
    double phi_space,
    double phi_time,
    const SpatiotemporalData& st_data,
    double* grad,
    double epsilon = 1e-6
) {
  int n_params = st_data.n_params;

  double base_ll = spatiotemporal_log_prior(delta, tau, tau2, rho,
                                            phi_space, phi_time, st_data);

  std::vector<double> delta_plus(delta, delta + n_params);

  for (int i = 0; i < n_params; i++) {
    delta_plus[i] = delta[i] + epsilon;
    double ll_plus = spatiotemporal_log_prior(delta_plus.data(), tau, tau2, rho,
                                              phi_space, phi_time, st_data);
    grad[i] = (ll_plus - base_ll) / epsilon;
    delta_plus[i] = delta[i];
  }
}

// =====================================================================
// Sum-to-zero constraint for interactions
// =====================================================================

// Apply soft sum-to-zero constraint marginally. Each margin gets the precision
// for its own length -- a space margin sums S terms, a time margin sums T --
// so the two cannot share one constant (see tulpa/soft_sum_to_zero.h).
template<typename Scalar>
inline Scalar st_sum_to_zero_penalty(
    const Scalar* delta,
    int S,
    int T,
    bool marginal_space = true,
    bool marginal_time = true,
    const tulpa::GraphPartition& partition = tulpa::GraphPartition{}
) {
  return tulpa::st_margin_penalty(delta, S, T, marginal_space, marginal_time, partition);
}

} // namespace ratiod_spatiotemporal

#endif // RATIOD_HMC_SPATIOTEMPORAL_H
