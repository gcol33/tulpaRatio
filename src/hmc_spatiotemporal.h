// hmc_spatiotemporal.h
// Spatiotemporal interaction effects for HMC backend
// Supports Type I-IV interactions (Knorr-Held) and separable/non-separable GP

#ifndef RATIOD_HMC_SPATIOTEMPORAL_H
#define RATIOD_HMC_SPATIOTEMPORAL_H

#include <vector>
#include <cmath>
#include <limits>
#include "hmc_cov.h"
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

// Whether the interaction's time margin carries a correlation to estimate.
// Type II applies the temporal precision within each spatial unit and Type IV
// carries it as a Kronecker margin, so both read rho. Type I is iid over the
// whole grid and Type III's time margin is unstructured, so neither has one;
// the GP types carry a range instead. HSGP-ST reads the margin per basis
// function, which the caller adds since it is a property of the data, not the
// interaction type.
inline bool st_time_margin_is_structured(STType type) {
  return type == STType::TYPE_II || type == STType::TYPE_IV;
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
    int rank_deficiency = 1  // 1 for ICAR
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
    bool cyclic
) {
  // For Type IV, we compute the Kronecker quadratic form
  // delta' (Q_s (x) Q_t) delta

  // Step 1: Compute Q_s * delta for each time point
  // Q_s[i,j] = n_neighbors[i] if i==j, -1 if neighbors, 0 otherwise
  std::vector<std::vector<Scalar>> Q_s_delta(T, std::vector<Scalar>(S, Scalar(0.0)));

  for (int t = 0; t < T; t++) {
    for (int i = 0; i < S; i++) {
      Scalar val = Scalar(0.0);
      int n_neigh = adj_row_ptr[i + 1] - adj_row_ptr[i];
      val = val + Scalar(n_neigh) * delta[i * T + t];  // Diagonal: n_neighbors

      for (int idx = adj_row_ptr[i]; idx < adj_row_ptr[i + 1]; idx++) {
        int j = adj_col_idx[idx] - 1;  // 0-based
        val = val - delta[j * T + t];  // Off-diagonal: -1
      }
      Q_s_delta[t][i] = val;
    }
  }

  // Step 2: Compute delta[*,t1]' Q_s delta[*,t2] for all t1, t2
  std::vector<std::vector<Scalar>> inner_prods(T, std::vector<Scalar>(T, Scalar(0.0)));

  for (int t1 = 0; t1 < T; t1++) {
    for (int t2 = 0; t2 < T; t2++) {
      Scalar ip = Scalar(0.0);
      for (int s = 0; s < S; s++) {
        ip = ip + delta[s * T + t1] * Q_s_delta[t2][s];
      }
      inner_prods[t1][t2] = ip;
    }
  }

  // Step 3: Apply temporal quadratic form
  Scalar quad = Scalar(0.0);

  if (temp_type == TemporalType::RW1) {
    // RW1: Q_t[t,t] = 2 (interior), 1 (boundary); Q_t[t,t+1] = -1
    for (int t = 0; t < T - 1; t++) {
      // (delta[*,t] - delta[*,t+1])' Q_s (delta[*,t] - delta[*,t+1])
      // = inner_prods[t,t] - 2*inner_prods[t,t+1] + inner_prods[t+1,t+1]
      quad = quad + inner_prods[t][t] - Scalar(2.0) * inner_prods[t][t + 1]
                  + inner_prods[t + 1][t + 1];
    }
    if (cyclic) {
      quad = quad + inner_prods[T-1][T-1] - Scalar(2.0) * inner_prods[T-1][0]
                  + inner_prods[0][0];
    }
  } else if (temp_type == TemporalType::RW2) {
    // RW2: second differences
    for (int t = 0; t < T - 2; t++) {
      // (delta[*,t] - 2*delta[*,t+1] + delta[*,t+2])' Q_s (...)
      std::vector<Scalar> diff2(S);
      for (int s = 0; s < S; s++) {
        diff2[s] = delta[s * T + t] - Scalar(2.0) * delta[s * T + t + 1]
                 + delta[s * T + t + 2];
      }
      // Apply Q_s to diff2
      for (int s = 0; s < S; s++) {
        Scalar Qs_diff2 = Scalar(0.0);
        int n_neigh = adj_row_ptr[s + 1] - adj_row_ptr[s];
        Qs_diff2 = Qs_diff2 + Scalar(n_neigh) * diff2[s];
        for (int idx = adj_row_ptr[s]; idx < adj_row_ptr[s + 1]; idx++) {
          int j = adj_col_idx[idx] - 1;
          Qs_diff2 = Qs_diff2 - diff2[j];
        }
        quad = quad + diff2[s] * Qs_diff2;
      }
    }
    // Cyclic extension if needed
  } else if (temp_type == TemporalType::AR1 && T >= 2) {
    // R(rho) contracted against the spatial inner products. R is tridiagonal,
    // so only the diagonal, its interior, and the first off-diagonal of
    // inner_prods survive -- the same two sums ratiod_ar1's scalar quadratic
    // form takes over x[t]^2 and x[t] x[t-1].
    Scalar interior = Scalar(0.0);
    for (int t = 1; t < T - 1; t++) interior = interior + inner_prods[t][t];
    Scalar cross = Scalar(0.0);
    for (int t = 1; t < T; t++) cross = cross + inner_prods[t][t - 1];
    quad = inner_prods[0][0] + inner_prods[T - 1][T - 1]
         + (Scalar(1.0) + rho * rho) * interior
         - Scalar(2.0) * rho * cross;
  }

  // Compute log-determinant contribution
  // For Kronecker: log|Q_s (x) Q_t| = rank(Q_t)*log|Q_s|_+ + rank(Q_s)*log|Q_t|_+
  // The spatial factor's own log-determinant is free of both hyperparameters and
  // is dropped with the other constants; the AR1 time margin's is not, since
  // log|R(rho)| = log(1 - rho^2) moves with the sampled correlation.

  int rank_space = S - 1;  // ICAR rank deficiency
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
template <typename Scalar>
inline Scalar gneiting_corr(double h, double u, const Scalar& phi_space,
                            const Scalar& phi_time) {
  const Scalar ut = Scalar(std::abs(u)) / phi_time;
  const Scalar base = ut * ut + Scalar(1.0);
  const Scalar hs = Scalar(h) / phi_space;
  return safe_exp(-hs / safe_sqrt(base)) / base;
}

template <typename Scalar>
inline Scalar st_corr(double h, double u, const Scalar& phi_space,
                      const Scalar& phi_time, const SpatiotemporalData& st) {
  if (st.nonsep_type == NonsepType::GNEITING) {
    return gneiting_corr(h, u, phi_space, phi_time);
  }
  const Scalar ks = ratiod_cov::compute_cov(h, Scalar(1.0), phi_space, st.cov_space);
  const Scalar kt = ratiod_cov::compute_cov(u, Scalar(1.0), phi_time, st.cov_time);
  return ks * kt;  // PRODUCT: the separable arm
}

template <typename Scalar>
inline Scalar st_cov(double h, double u, const Scalar& sigma2,
                     const Scalar& phi_space, const Scalar& phi_time,
                     const SpatiotemporalData& st) {
  return sigma2 * st_corr(h, u, phi_space, phi_time, st);
}

// dr/d(phi_space) and dr/d(phi_time), for the analytic gradient path. Written
// directly under the correlation they differentiate and in terms of the same
// intermediate quantities; cpp_gradient_check("stgp*") is what holds the two
// together.
inline void st_corr_dphi(double h, double u, double phi_space, double phi_time,
                         const SpatiotemporalData& st,
                         double* dr_dphi_space, double* dr_dphi_time) {
  if (st.nonsep_type == NonsepType::GNEITING) {
    const double ut = std::abs(u) / phi_time;
    const double base = ut * ut + 1.0;
    const double sqrt_base = std::sqrt(base);
    const double hs = h / phi_space;
    const double r = std::exp(-hs / sqrt_base) / base;
    // hs enters only through -hs/sqrt(base), and d(hs)/d(phi_space) is
    // -hs/phi_space.
    *dr_dphi_space = r * hs / (phi_space * sqrt_base);
    // base enters twice, and d(base)/d(phi_time) is -2 ut^2 / phi_time.
    const double dr_dbase = r * (-1.0 / base + 0.5 * hs / (base * sqrt_base));
    *dr_dphi_time = dr_dbase * (-2.0 * ut * ut / phi_time);
    return;
  }
  const double ks = ratiod_cov::compute_cov(h, 1.0, phi_space, st.cov_space);
  const double kt = ratiod_cov::compute_cov(u, 1.0, phi_time, st.cov_time);
  const double dks = ratiod_cov::dcov_dphi(h, 1.0, phi_space, ks, st.cov_space);
  const double dkt = ratiod_cov::dcov_dphi(u, 1.0, phi_time, kt, st.cov_time);
  *dr_dphi_space = dks * kt;
  *dr_dphi_time = ks * dkt;
}

// The separation between two grid cells, by their field index. Both the
// neighbour block and its derivative read it, so the geometry is written once.
inline void st_pair_separation(const SpatiotemporalData& st, int a, int b,
                               double* h, double* u) {
  if (a == b) {
    *h = 0.0;
    *u = 0.0;
    return;
  }
  const double dx = st.coords[a * 2] - st.coords[b * 2];
  const double dy = st.coords[a * 2 + 1] - st.coords[b * 2 + 1];
  *h = std::sqrt(dx * dx + dy * dy);
  *u = std::abs(st.time_values[a] - st.time_values[b]);
}

// Whether the interaction carries the NNGP structure its GP types need. The
// members are filled at the .Call boundary and the R front door refuses a
// configuration that cannot fill them; this is what keeps a half-filled
// structure from being read as geometry.
inline bool st_gp_structure_filled(const SpatiotemporalData& st) {
  const int N = st.n_params;
  if (N <= 0 || st.nn <= 0) return false;
  if (static_cast<int>(st.nn_order.size()) != N) return false;
  if (static_cast<int>(st.coords.size()) != 2 * N) return false;
  if (static_cast<int>(st.time_values.size()) != N) return false;
  const size_t need = static_cast<size_t>(N) * st.nn;
  return st.nn_idx.size() == need && st.nn_dist_space.size() == need &&
         st.nn_dist_time.size() == need;
}

// =====================================================================
// The NNGP conditional decomposition
// =====================================================================

// One pass over the field's Vecchia decomposition, shared by the density and by
// the analytic gradient so the two cannot disagree about what the prior is.
// `visit` is called once per location, in the NNGP ordering, with that
// location's neighbour block already factorized:
//
//   visit(i, obs_idx, nb, n_nb, L, c_vec, alpha, cond_mean, cond_var, floored)
//
// where i is the position in the ordering, obs_idx and nb[j] are field indices,
// alpha = C^-1 c, and n_nb == 0 marks the marginal arm -- the first location,
// and any location whose neighbour list is empty -- for which L, c_vec and
// alpha carry nothing and cond_var is sigma2.
//
// Returns false if the structure is not filled, or if a neighbour block could
// not be factorized. Both are a rejected parameter state at the caller rather
// than a substituted pivot or a read past the end of an empty vector.
template <typename Scalar, typename Visit>
inline bool st_gp_nngp_scan(const Scalar* w, const Scalar& sigma2,
                            const Scalar& phi_space, const Scalar& phi_time,
                            const SpatiotemporalData& st, Visit&& visit) {
  if (!st_gp_structure_filled(st)) return false;

  const int N = st.n_params;
  const int nn = st.nn;
  const Scalar zero(0.0);

  visit(0, st.nn_order[0], static_cast<const int*>(nullptr), 0,
        static_cast<const Scalar*>(nullptr), static_cast<const Scalar*>(nullptr),
        static_cast<const Scalar*>(nullptr), zero, sigma2, false);

  std::vector<int> nb;
  std::vector<Scalar> c_vec, C_mat, L, y, alpha;

  for (int i = 1; i < N; i++) {
    const int obs_idx = st.nn_order[i];

    int n_nb = 0;
    for (int j = 0; j < nn; j++) {
      if (st.nn_idx[static_cast<size_t>(i) * nn + j] > 0) n_nb++;
    }
    if (n_nb == 0) {
      visit(i, obs_idx, static_cast<const int*>(nullptr), 0,
            static_cast<const Scalar*>(nullptr), static_cast<const Scalar*>(nullptr),
            static_cast<const Scalar*>(nullptr), zero, sigma2, false);
      continue;
    }

    nb.resize(n_nb);
    c_vec.resize(n_nb);
    for (int j = 0; j < n_nb; j++) {
      const size_t f = static_cast<size_t>(i) * nn + j;
      nb[j] = st.nn_order[st.nn_idx[f] - 1];
      c_vec[j] = st_cov(st.nn_dist_space[f], st.nn_dist_time[f], sigma2,
                        phi_space, phi_time, st);
    }

    C_mat.assign(static_cast<size_t>(n_nb) * n_nb, zero);
    for (int j1 = 0; j1 < n_nb; j1++) {
      for (int j2 = j1; j2 < n_nb; j2++) {
        double h12, u12;
        st_pair_separation(st, nb[j1], nb[j2], &h12, &u12);
        const Scalar cov_val = st_cov(h12, u12, sigma2, phi_space, phi_time, st);
        C_mat[static_cast<size_t>(j1) * n_nb + j2] = cov_val;
        C_mat[static_cast<size_t>(j2) * n_nb + j1] = cov_val;
      }
    }

    if (!ratiod_cov::nngp_chol(C_mat, n_nb, L)) return false;
    ratiod_cov::nngp_forward_solve(L, n_nb, c_vec, y);
    ratiod_cov::nngp_back_solve(L, n_nb, y, alpha);

    Scalar cond_mean = zero;
    Scalar c_Cinv_c = zero;
    for (int j = 0; j < n_nb; j++) {
      cond_mean = cond_mean + alpha[j] * w[nb[j]];
      c_Cinv_c = c_Cinv_c + c_vec[j] * alpha[j];
    }
    const Scalar cond_var_raw = sigma2 - c_Cinv_c;
    const bool floored =
        (get_value(cond_var_raw) < ratiod_cov::NNGP_MIN_COND_VAR);
    const Scalar cond_var = ratiod_cov::nngp_floor_cond_var(cond_var_raw);

    visit(i, obs_idx, static_cast<const int*>(nb.data()), n_nb,
          static_cast<const Scalar*>(L.data()),
          static_cast<const Scalar*>(c_vec.data()),
          static_cast<const Scalar*>(alpha.data()), cond_mean, cond_var, floored);
  }

  return true;
}

// Log-density of the interaction field under the NNGP approximation.
template<typename Scalar>
inline Scalar st_gp_nngp_log_lik(
    const Scalar* w,           // ST effect, length st_data.n_params
    const Scalar& sigma2,
    const Scalar& phi_space,
    const Scalar& phi_time,
    const SpatiotemporalData& st_data
) {
  Scalar log_lik = Scalar(0.0);
  const bool ok = st_gp_nngp_scan<Scalar>(
      w, sigma2, phi_space, phi_time, st_data,
      [&](int /*i*/, int obs_idx, const int* /*nb*/, int /*n_nb*/,
          const Scalar* /*L*/, const Scalar* /*c_vec*/, const Scalar* /*alpha*/,
          const Scalar& cond_mean, const Scalar& cond_var, bool /*floored*/) {
        const Scalar resid = w[obs_idx] - cond_mean;
        log_lik = log_lik - Scalar(0.5) * safe_log(Scalar(2.0 * M_PI) * cond_var)
                          - Scalar(0.5) * resid * resid / cond_var;
      });
  if (!ok) return Scalar(-std::numeric_limits<double>::infinity());
  return log_lik;
}

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
    case STType::TYPE_III: return (S - 1) * T;
    case STType::TYPE_IV:  return (S - 1) * rank_time;
    default:               return 0;
  }
}

// How many null directions of Q the sum-to-zero penalties pin: the row sums
// pin each time's spatial constant, the column sums each unit's temporal
// constant. Under the non-centered parameterization those directions carry a
// tau-free proper prior on z, so they drop out of the tau normalizer; the
// null directions they leave free (an RW2 margin's linear trends) keep
// theirs. Type I is full rank; Type II's time margin has a constant null
// direction per unit when it is RW1 or RW2; Type III's ICAR margin has one
// per time; Type IV's Kronecker null space contains both families, sharing
// the grand constant.
inline int st_constrained_null_dim(const SpatiotemporalData& st) {
  const int S = st.n_spatial;
  const int T = st.n_times;
  const bool time_null = st.temporal_type == TemporalType::RW1 ||
                         st.temporal_type == TemporalType::RW2;
  switch (st.type) {
    case STType::TYPE_I:   return 0;
    case STType::TYPE_II:  return time_null ? S : 0;
    case STType::TYPE_III: return T;
    case STType::TYPE_IV:  return T + (time_null ? S - 1 : 0);
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

// The centered density's correction to its tau normalizer, times two. The
// sum-to-zero penalties constrain the row and column sums of delta, S + T - 1
// directions; in their hard limit the field lives on the subspace they leave,
// whose normalizer counts only the directions Q makes proper there. The
// per-type priors count rank(Q), which is right only where every constrained
// direction is a null direction of Q (Type IV with an RW margin); the
// constrained directions Q makes proper (all of them for Type I and the GP
// types) are taken back out here.
inline int st_centered_tau_correction(const SpatiotemporalData& st) {
  const int n_constraints = st.n_spatial + st.n_times - 1;
  return st_constrained_null_dim(st) - n_constraints;
}

// Whether the interaction is sampled non-centered: z with delta = z / sqrt(tau),
// for the Knorr-Held GMRF types. The spectral (HSGP) margin and the GP types
// are always centered.
inline bool st_noncentered(int parameterization, bool is_hsgp,
                           const SpatiotemporalData& st) {
  if (parameterization != 1 || is_hsgp) return false;
  return st.type == STType::TYPE_I || st.type == STType::TYPE_II ||
         st.type == STType::TYPE_III || st.type == STType::TYPE_IV;
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
      for (int k = 0; k + 2 < T; k++) {
        const double d2 = xs[k] - 2.0 * xs[k + 1] + xs[k + 2];
        os[k] += d2;
        os[k + 1] -= 2.0 * d2;
        os[k + 2] += d2;
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
                                st_data.adj_row_ptr, st_data.adj_col_idx);

    case STType::TYPE_IV:
      return type_iv_log_prior(delta, S, T, tau, tau2, rho,
                               st_data.adj_row_ptr, st_data.adj_col_idx,
                               st_data.temporal_type, st_data.temporal_cyclic);

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
    bool marginal_time = true
) {
  Scalar penalty = Scalar(0.0);

  if (marginal_space) {
    // For each time point, spatial effects sum to zero
    const double lambda_s = tulpa::s2z_precision(S);
    for (int t = 0; t < T; t++) {
      Scalar sum = Scalar(0.0);
      for (int s = 0; s < S; s++) {
        sum = sum + delta[s * T + t];
      }
      penalty = penalty - Scalar(0.5 * lambda_s) * sum * sum;
    }
  }

  if (marginal_time) {
    // For each spatial unit, temporal effects sum to zero
    const double lambda_t = tulpa::s2z_precision(T);
    for (int s = 0; s < S; s++) {
      Scalar sum = Scalar(0.0);
      for (int t = 0; t < T; t++) {
        sum = sum + delta[s * T + t];
      }
      penalty = penalty - Scalar(0.5 * lambda_t) * sum * sum;
    }
  }

  return penalty;
}

} // namespace ratiod_spatiotemporal

#endif // RATIOD_HMC_SPATIOTEMPORAL_H
