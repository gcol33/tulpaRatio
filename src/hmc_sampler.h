// hmc_sampler.h
// Full HMC/NUTS backend with spatial, temporal, and zero-inflation support
// Supports ICAR/BYM2 spatial effects, RW/AR1 temporal, and ZI/hurdle models

#ifndef QUOTR_HMC_SAMPLER_H
#define QUOTR_HMC_SAMPLER_H

#include <Rcpp.h>
#include <vector>
#include <cmath>
#include <cstring>
#include <random>
#include "linalg_fast.h"
#include "hmc_temporal.h"
#include "hmc_temporal_gp.h"
#include "hmc_zi.h"
#include "hmc_svc.h"
#include "hmc_gp.h"
#include "hmc_temporal_multiscale.h"
#include "hmc_latent.h"
#include "hmc_spatiotemporal.h"
#include "hmc_hsgp.h"
#include "hmc_tvc.h"
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace ratiod_hmc {

using ratiod_temporal::TemporalType;
using ratiod_temporal::TemporalData;
using ratiod_temporal_gp::TemporalGPData;
using ratiod_temporal_gp::TemporalCovType;
using ratiod_zi::ZIType;
using ratiod_gp::GPData;
using ratiod_gp::MultiscaleGPData;
using ratiod_gp::CovType;
using ratiod_temporal::MultiscaleTemporalData;
using ratiod_spatiotemporal::STType;
using ratiod_spatiotemporal::SpatiotemporalData;
using ratiod_spatiotemporal::NonsepType;

// =====================================================================
// Model configuration
// =====================================================================

enum class ModelType { BINOMIAL, NEGBIN_NEGBIN, POISSON_GAMMA, NEGBIN_GAMMA, GAMMA_GAMMA, LOGNORMAL, BETA_BINOMIAL };
enum class SpatialType { NONE, ICAR, BYM2, GP, MULTISCALE_GP, HSGP, CAR_PROPER };

// Gradient computation methods
// AUTO: Use fastest available (H > A_r > A > A_t > N)
// NUMERICAL (N): Finite differences - slow but always works
// AUTODIFF_TAPE (A_t): Tape-based reverse-mode - slow due to heap allocation
// AUTODIFF_ARENA (A_r): Arena-based reverse-mode - fast O(N), ~10-30x over A_t
// AUTODIFF_FORWARD (A): Forward-mode dual numbers - O(p*N), thread-safe
// HANDCODED (H): Analytical gradients - fastest when available
enum class GradientMode { AUTO, NUMERICAL, AUTODIFF_TAPE, AUTODIFF_ARENA, AUTODIFF_FORWARD, HANDCODED };

// Mass matrix type for NUTS
enum class MassMatrixType { DIAG, DENSE, BLOCK_DIAG, AUTO };

// Parse gradient mode from string
inline GradientMode parse_gradient_mode(const std::string& mode_str) {
    if (mode_str == "auto" || mode_str == "AUTO") return GradientMode::AUTO;
    if (mode_str == "N" || mode_str == "numerical") return GradientMode::NUMERICAL;
    if (mode_str == "A_t" || mode_str == "autodiff_tape") return GradientMode::AUTODIFF_TAPE;
    if (mode_str == "A_r" || mode_str == "arena" || mode_str == "autodiff_arena") return GradientMode::AUTODIFF_ARENA;
    if (mode_str == "A" || mode_str == "autodiff" || mode_str == "forward") return GradientMode::AUTODIFF_FORWARD;
    if (mode_str == "H" || mode_str == "handcoded" || mode_str == "analytical") return GradientMode::HANDCODED;
    return GradientMode::AUTO;  // Default
}

// Parse metric type from string
inline MassMatrixType parse_metric_type(const std::string& metric_str) {
    if (metric_str == "dense" || metric_str == "DENSE") return MassMatrixType::DENSE;
    if (metric_str == "block_diag" || metric_str == "BLOCK_DIAG") return MassMatrixType::BLOCK_DIAG;
    if (metric_str == "auto" || metric_str == "AUTO") return MassMatrixType::AUTO;
    return MassMatrixType::DIAG;  // Default
}

// Human-readable metric name for verbose logging
inline const char* metric_name(MassMatrixType t) {
    switch (t) {
        case MassMatrixType::DIAG: return "DIAG";
        case MassMatrixType::DENSE: return "DENSE";
        case MassMatrixType::BLOCK_DIAG: return "BLOCK_DIAG";
        case MassMatrixType::AUTO: return "AUTO";
    }
    return "UNKNOWN";
}

// Set global gradient mode (defined in hmc_sampler.cpp)
void set_gradient_mode(GradientMode mode);

// Function pointer type for gradient computation (eliminates per-call dispatch overhead)
struct ModelData;
struct ParamLayout;
using GradientFn = void(*)(
    const std::vector<double>&, const ModelData&, const ParamLayout&,
    std::vector<double>&, double*);

// Resolve the gradient function pointer once based on mode + model config
GradientFn resolve_gradient_fn(GradientMode mode, const ModelData& data, const ParamLayout& layout);
const char* gradient_fn_name(GradientFn fn);

// Errors when the active gradient mode differentiates the templated log
// posterior and the model carries a structure that density cannot express.
// Call on the R thread, before sampling starts.
void require_autodiff_supported(const ModelData& data, const ParamLayout& layout);

// Model data container with spatial support
struct ModelData {
  // Unique ID for cache invalidation (incremented per construction)
  // Prevents stale VecGradWorkspace cache when consecutive model fits
  // allocate ModelData at the same stack address.
  uint64_t unique_id;
  static inline uint64_t next_id() {
    static uint64_t counter = 0;
    return ++counter;
  }
  ModelData() : unique_id(next_id()) {}

  // Response data
  std::vector<int> y_num;
  std::vector<int> y_denom;
  std::vector<double> y_num_cont;    // For continuous numerator (gamma_gamma, lognormal)
  std::vector<double> y_denom_cont;  // For continuous denominator (poisson_gamma, gamma_gamma, lognormal)

  // Design matrices (stored as flat vectors for cache efficiency)
  std::vector<double> X_num_flat;
  std::vector<double> X_denom_flat;
  int p_num = 0, p_denom = 0;

  // Random effects (supports multiple crossed RE terms with slopes)
  std::vector<int> re_group;  // 1-based group index (0 = no RE) - legacy single term
  int n_re_groups = 0;        // Number of groups for first RE term

  // Multi-term RE structure
  int n_re_terms = 0;                          // Number of RE terms (0 if none)
  std::vector<std::vector<int>> re_group_multi; // [term][obs] -> group index (1-based) LEGACY
  std::vector<int> re_group_multi_flat;        // [obs * n_re_terms + term] -> group index (1-based), obs-major
  std::vector<int> re_n_groups_multi;          // Groups per term
  std::vector<int> re_offsets;                 // Offset in flattened RE parameter vector per term
  int total_re_groups = 0;                     // Sum of all groups across terms

  // Random slopes support
  bool has_re_slopes = false;                  // Whether any RE term has slopes
  bool has_re_correlated_slopes = false;       // Whether any RE term has correlated slopes
  std::vector<int> re_n_coefs;                 // Coefficients per group per term (1 = intercept only)
  std::vector<std::vector<double>> re_slope_matrices; // [term] -> flattened [N x n_slopes] slope design matrix
  std::vector<int> re_n_slopes;                // Number of slope variables per term
  std::vector<bool> re_correlated;             // Whether each term has correlated slopes
  std::vector<int> re_n_chol;                  // Cholesky parameters per term (k*(k-1)/2 for correlated, 0 otherwise)
  int total_re_params = 0;                     // Total RE parameters (groups * coefs)
  int total_sigma_params = 0;                  // Total variance parameters (sum of n_coefs)
  int total_chol_params = 0;                   // Total Cholesky correlation parameters

  // RE parameterization: 0 = centered (default), 1 = non-centered
  // Non-centered stores z ~ N(0,1) instead of re ~ N(0, sigma^2)
  // and computes re = sigma * z (or re = diag(sigma) * L * z for correlated)
  int re_parameterization = 0;                 // 0 = centered, 1 = non-centered

  // Spatial structure
  SpatialType spatial_type = SpatialType::NONE;
  std::vector<int> spatial_group;    // Maps obs to spatial unit (1-based)
  int n_spatial_units = 0;
  std::vector<int> adj_row_ptr;      // CSR format: row pointers
  std::vector<int> adj_col_idx;      // CSR format: column indices
  std::vector<int> n_neighbors;      // Number of neighbors per unit
  double bym2_scale_factor = 1.0;    // For BYM2 scaling
  // Proper CAR (rho estimated from data, Q = D - rho*W): rho support and prior
  double car_rho_lower = 0.0;
  double car_rho_upper = 1.0;
  double car_rho_prior_a = 1.0;      // Beta(a, b) on rho, scaled to (lower, upper)
  double car_rho_prior_b = 1.0;
  // Whether a proper CAR field's mean is removed from eta. Q(rho) is full rank,
  // so the prior identifies the mean, but the LIKELIHOOD cannot see it apart
  // from the intercept: eta reads beta_0 + phi_s, and moving the intercept by c
  // against every phi by -c leaves eta where it was. Removing the direction from
  // eta leaves the intercept carrying the level and the field carrying
  // deviations, with the prior still reading the raw field, which is the
  // treatment an SVC term already takes.
  bool car_center = true;
  // Precision mass matrix data (precomputed from Q)
  std::vector<double> spatial_Q_inv;    // (Q + lambda*I)^{-1}, column-major [S×S]
  std::vector<double> spatial_L_Q;      // Cholesky L of (Q + lambda*I), column-major [S×S]

  // Temporal structure
  TemporalType temporal_type = TemporalType::NONE;
  std::vector<int> temporal_time_idx;   // Maps obs to time point (1-based, 0 = no temporal)
  std::vector<int> temporal_group_idx;  // Maps obs to temporal group (1-based)
  int n_times = 0;                      // Number of time points
  int n_temporal_groups = 1;            // Number of temporal groups (1 if no grouping)
  int n_temporal_params = 0;            // Total temporal parameters
  bool temporal_cyclic = false;         // Whether RW is cyclic
  bool temporal_shared = true;          // Whether effect is shared between num/denom
  // AR1 coordinate the sampler moves in: 0 = the effects themselves,
  // 1 = the N(0, I) innovations they are reconstructed from. Only AR1 has a
  // second coordinate; RW1 and RW2 are intrinsic and IID is already unit-scale.
  int temporal_parameterization = 0;
  double tau_temporal_shape = 1.0;      // Gamma shape for temporal precision
  double tau_temporal_rate = 0.01;      // Gamma rate for temporal precision
  // Beta(a, b) on u = (rho + 1) / 2 for an AR1 temporal correlation, which is
  // what priors_default() documents and temporal_ar1(rho_prior =) overrides.
  double temporal_rho_prior_a = ratiod_ar1::RHO_PRIOR_A;
  double temporal_rho_prior_b = ratiod_ar1::RHO_PRIOR_B;

  // Zero-inflation structure
  ZIType zi_type = ZIType::NONE;
  std::vector<double> X_zi_flat;        // Design matrix for ZI probability (flat)
  int p_zi = 0;                         // Number of ZI predictors
  double zi_prior_sd = 1.0;             // Prior SD for ZI coefficients

  // One-inflation structure (for OI-binomial and ZOIB models)
  std::vector<double> X_oi_flat;        // Design matrix for OI probability (flat)
  int p_oi = 0;                         // Number of OI predictors
  double oi_prior_sd = 1.0;             // Prior SD for OI coefficients

  // SVC (Spatially-Varying Coefficients) structure
  ratiod_svc::SVCData svc_data;
  bool has_svc = false;
  bool svc_is_hsgp = false;             // HSGP approximation for SVC
  ratiod_hsgp::HSGPData svc_hsgp_data;  // Shared HSGP basis for all SVC terms
  int svc_hsgp_m_per_dim = 6;           // Basis functions per dimension for SVC-HSGP
  double svc_hsgp_boundary_factor = 1.5;
  double svc_sigma2_prior_scale = 1.0;  // Half-Cauchy scale for sigma2
  // Uniform prior bounds for phi. apply_svc_params() overwrites both from the
  // spec's `range` whenever has_svc, so these carry the same numbers as
  // svc_default_range() in R rather than a second pair a model could run under.
  double svc_phi_prior_lower = 0.3;
  double svc_phi_prior_upper = 30.0;

  // GP spatial structure (single-scale)
  GPData gp_data;
  bool has_gp = false;
  double gp_sigma2_prior_U = 1.0;       // PC prior: P(sigma > U) = alpha
  double gp_sigma2_prior_alpha = 0.01;
  double gp_phi_prior_lower = 0.01;     // Uniform prior bounds for range
  double gp_phi_prior_upper = 10.0;
  int gp_parameterization = 1;          // 0=centered, 1=non-centered (default NC)

  // Collapsed parameterization flags (marginalize inner spatial effects via Laplace)
  // All collapsed logic lives in hmc_icar_collapsed.h and hmc_gp_collapsed.h
  bool icar_collapsed = false;       // Collapsed ICAR (marginalize phi)
  bool bym2_collapsed = false;       // Collapsed BYM2 (marginalize phi+theta)
  bool gp_collapsed = false;         // Collapsed GP (marginalize w)

  // Multi-scale GP spatial structure
  MultiscaleGPData multiscale_gp_data;
  bool has_multiscale_gp = false;
  bool msgp_is_hsgp = false;                    // HSGP approximation for MSGP
  int msgp_parameterization = 0;                // 0=centered, 1=non-centered
  ratiod_hsgp::HSGPData msgp_hsgp_data;         // Shared HSGP basis for both scales
  double ms_sigma2_local_prior_U = 1.0;
  double ms_sigma2_local_prior_alpha = 0.01;
  double ms_sigma2_regional_prior_U = 1.0;
  double ms_sigma2_regional_prior_alpha = 0.01;
  // Lengthscale prior means for HSGP-MSGP (log-scale, for LogNormal prior)
  double ms_log_ls_local_mean = -1.0;            // log(~0.37) — short range
  double ms_log_ls_local_sd = 0.5;
  double ms_log_ls_regional_mean = 1.0;          // log(~2.7) — long range
  double ms_log_ls_regional_sd = 0.5;

  // Multi-scale temporal structure
  MultiscaleTemporalData multiscale_temporal_data;
  bool has_multiscale_temporal = false;
  double ms_sigma2_trend_prior_U = 1.0;
  double ms_sigma2_trend_prior_alpha = 0.01;
  double ms_sigma2_seasonal_prior_U = 1.0;
  double ms_sigma2_seasonal_prior_alpha = 0.01;
  double ms_sigma2_short_prior_U = 1.0;
  double ms_sigma2_short_prior_alpha = 0.01;

  // Temporal GP structure (for irregularly-spaced time series)
  TemporalGPData temporal_gp_data;
  bool has_temporal_gp = false;
  double temporal_gp_sigma2_prior_U = 1.0;    // PC prior: P(sigma > U) = alpha
  double temporal_gp_sigma2_prior_alpha = 0.01;
  double temporal_gp_phi_prior_lower = 0.01;  // Uniform prior bounds for range
  double temporal_gp_phi_prior_upper = 10.0;
  int temporal_gp_parameterization = 1;       // 0=centered, 1=non-centered (default NC)

  // HSGP (Hilbert Space GP) structure
  ratiod_hsgp::HSGPData hsgp_data;
  bool has_hsgp = false;
  int hsgp_m_per_dim = 15;             // Basis functions per dimension
  double hsgp_boundary_factor = 1.5;   // Boundary factor (c)

  // RSR (Restricted Spatial Regression) structure
  bool has_rsr = false;
  std::vector<double> rsr_projection;   // P_perp matrix (n x n, flattened)
  int rsr_n = 0;                        // Dimension of projection matrix

  // Latent factors for unmeasured confounders
  bool has_latent = false;
  int latent_n_factors = 0;             // Number of latent factors (K)
  bool latent_shared = true;            // Whether factors enter both num and denom
  bool latent_scale = true;             // Whether to standardize factors
  int latent_constraint = 0;            // 0 = sum_to_zero, 1 = first_zero
  double latent_sigma_prior_rate = 1.0; // Exponential rate for PC prior on sigma

  // Spatiotemporal interaction
  bool has_spatiotemporal = false;
  bool st_is_hsgp = false;                // HSGP-ST: spectral basis interaction
  SpatiotemporalData spatiotemporal_data;
  ratiod_hsgp::HSGPData st_hsgp_data;    // HSGP basis for ST interaction (separate from main HSGP)
  int st_parameterization = 0;            // 0=centered, 1=non-centered (NC requires spectral decomposition)
  double st_sigma2_prior_U = 1.0;       // PC prior for interaction variance
  double st_sigma2_prior_alpha = 0.01;
  double st_phi_space_prior_lower = 0.01;  // Uniform bounds for spatial range
  double st_phi_space_prior_upper = 10.0;
  double st_phi_time_prior_lower = 0.01;   // Uniform bounds for temporal range
  double st_phi_time_prior_upper = 10.0;

  // Kronecker precision data for ST_IV (precomputed in R)
  std::vector<double> st_Qs_inv;   // (Q_space + lambda*I)^{-1}, column-major S×S
  std::vector<double> st_Ls;       // Cholesky L of (Q_space + lambda*I), column-major S×S
  std::vector<double> st_Qt_inv;   // (Q_time + lambda*I)^{-1}, column-major T×T
  std::vector<double> st_Lt;       // Cholesky L of (Q_time + lambda*I), column-major T×T

  // TVC (Temporally-Varying Coefficients) structure
  ratiod_tvc::TVCData tvc_data;
  bool has_tvc = false;
  double tvc_tau_shape = 1.0;           // Gamma shape for TVC precision
  double tvc_tau_rate = 0.01;           // Gamma rate for TVC precision

  // Dimensions
  int N = 0;

  // Prior parameters
  double sigma_beta = 1.0;
  double sigma_re_scale = 1.0;
  double phi_prior_shape = 1.0;
  double phi_prior_rate = 1.0;
  double tau_spatial_shape = 1.0;
  double tau_spatial_rate = 1.0;

  // Model type
  ModelType model_type = ModelType::BINOMIAL;

  // Parallelization
  int n_threads = 1;
  // Across-chain core budget. Bounds the multi-chain OpenMP team size so the
  // team stays the same size across successive fits in one session; a team
  // that grows then shrinks corrupts the GOMP thread pool and faults at
  // process teardown on Windows.
  int n_cores = 1;
};

// =====================================================================
// Parameter layout
// =====================================================================

// Parameter vector layout (order matters for cache efficiency):
// [beta_num, beta_denom, log_sigma_re?, re?, log_phi_num?, log_phi_denom?,
//  log_tau_spatial?, spatial?, log_sigma_bym2?, logit_rho_bym2?, theta_bym2?]

struct ParamLayout {
  int beta_num_start, beta_num_end;
  int beta_denom_start, beta_denom_end;
  int log_sigma_re_idx;       // Legacy: index for single RE term
  int re_start, re_end;       // Legacy: bounds for single RE term

  // Multi-term RE layout (supports intercept-only terms)
  std::vector<int> log_sigma_re_multi;  // Index of log_sigma_re for each term
  std::vector<int> re_start_multi;      // Start index for each RE term
  std::vector<int> re_end_multi;        // End index for each RE term

  // Random slopes layout (when has_re_slopes is true)
  bool has_re_slopes = false;
  bool has_re_correlated_slopes = false;
  // For each term: number of coefficients (1 = intercept only, >1 = with slopes)
  std::vector<int> re_n_coefs_multi;
  // For each term: whether correlated (for | syntax) vs uncorrelated (for || syntax)
  std::vector<bool> re_correlated_multi;
  // For slopes: log_sigma_re_slopes[term] = [idx_intercept, idx_slope1, idx_slope2, ...]
  std::vector<std::vector<int>> log_sigma_re_slopes;
  // For correlated slopes: Cholesky lower-triangular parameters (off-diagonal only)
  // chol_re_multi[term] = [L_21, L_31, L_32, ...] (column-major lower triangle)
  std::vector<int> chol_re_start_multi;  // Start index for Cholesky params for each term
  std::vector<int> chol_re_end_multi;    // End index for Cholesky params for each term
  // re_coefs[term][group] = [re_intercept, re_slope1, re_slope2, ...]
  // We track start/end per term, and n_coefs per term to decode within
  int log_phi_num_idx;
  int log_phi_denom_idx;
  int log_tau_spatial_idx;
  int spatial_start, spatial_end;
  // BYM2 extras (Riebler reparameterization: sigma_total, rho)
  // sigma_s = sigma_total * sqrt(rho), sigma_u = sigma_total * sqrt(1-rho)
  int log_sigma_bym2_idx;     // log(sigma_total): total spatial SD
  int logit_rho_bym2_idx;     // logit(rho): mixing parameter (structured fraction)
  int theta_bym2_start, theta_bym2_end;
  // Proper CAR extra: rho (correlation strength), shares log_tau_spatial_idx
  // and spatial_start/end with ICAR above.
  bool is_car_proper = false;
  int logit_rho_car_idx;      // logit-scaled rho in (car_rho_lower, car_rho_upper)

  // Temporal parameters
  int log_tau_temporal_idx;             // Log precision for temporal
  int logit_rho_ar1_idx;                // AR1 autocorrelation (logit scale)
  int temporal_start, temporal_end;     // Temporal effect parameters

  // Temporal GP parameters (for irregularly-spaced time series)
  int log_sigma2_temporal_gp_idx;       // Log marginal variance
  int logit_phi_temporal_gp_idx;        // Logit-bounded length-scale

  // Zero-inflation parameters
  int beta_zi_start, beta_zi_end;       // ZI regression coefficients

  // One-inflation parameters (for OI-binomial and ZOIB)
  int beta_oi_start, beta_oi_end;       // OI regression coefficients

  // SVC parameters
  int log_sigma2_svc_start, log_sigma2_svc_end;  // Log spatial variance per SVC term
  int log_phi_svc_start, log_phi_svc_end;        // Log range parameter per SVC term
  int svc_w_start, svc_w_end;                    // SVC values (n_obs x n_svc)

  // GP spatial parameters
  int log_sigma2_gp_idx;                // Log spatial variance
  int log_phi_gp_idx;                   // Log range parameter
  int gp_w_start, gp_w_end;             // GP spatial effects (n_obs)

  // Multi-scale GP parameters
  int log_sigma2_gp_local_idx;          // Log variance for local scale
  int log_phi_gp_local_idx;             // Log range for local scale
  int log_sigma2_gp_regional_idx;       // Log variance for regional scale
  int log_phi_gp_regional_idx;          // Log range for regional scale
  int gp_local_start, gp_local_end;     // Local GP effects
  int gp_regional_start, gp_regional_end; // Regional GP effects

  // Multi-scale temporal parameters
  int log_sigma2_trend_idx;             // Log variance for trend
  int log_sigma2_seasonal_idx;          // Log variance for seasonal
  int log_sigma2_short_idx;             // Log variance for short-term
  int logit_rho_short_idx;              // AR1 autocorrelation for short-term
  int trend_start, trend_end;           // Trend effects (n_times)
  int seasonal_start, seasonal_end;     // Seasonal effects (period)
  int short_term_start, short_term_end; // Short-term effects (n_times)

  // HSGP parameters
  int log_sigma2_hsgp_idx;              // Log spatial variance
  int log_lengthscale_hsgp_idx;         // Log lengthscale
  int hsgp_beta_start, hsgp_beta_end;   // HSGP basis coefficients (m^2)

  // Latent factor parameters
  int log_sigma_latent_start, log_sigma_latent_end;  // Log sigma per factor
  int latent_factor_start, latent_factor_end;        // Factor scores (N x K)

  // Spatiotemporal interaction parameters
  int log_tau_st_idx;                                // Log precision for ST interaction
  int log_tau_st2_idx;                               // Second precision (Type IV)
  int logit_rho_st_idx;                              // AR1 autocorrelation if ST temporal is AR1
  int log_phi_st_space_idx;                          // Log spatial range (GP-based)
  int log_phi_st_time_idx;                           // Log temporal range (GP-based)
  int st_delta_start, st_delta_end;                  // ST interaction effects (S * T or m^2 * T)
  // HSGP-ST: separate hyperparameters for spectral basis interaction
  int log_sigma2_st_hsgp_idx;                        // Log variance for ST HSGP kernel
  int log_lengthscale_st_hsgp_idx;                   // Log lengthscale for ST HSGP kernel
  bool is_st_hsgp;                                   // True if ST uses HSGP basis

  // TVC (Temporally-Varying Coefficients) parameters
  int log_tau_tvc_start, log_tau_tvc_end;            // Log precision per TVC term
  int logit_rho_tvc_start, logit_rho_tvc_end;        // AR1 correlations (if structure == AR1)
  int tvc_w_start, tvc_w_end;                        // TVC values (n_groups * n_tvc * n_times)

  int total_params = 0;

  bool has_re = false;
  bool has_phi_num = false;
  bool has_phi_denom = false;
  bool has_spatial = false;
  bool is_bym2 = false;
  bool is_gp = false;
  // Collapsed parameterization flags (mirror of ModelData collapsed flags)
  bool is_icar_collapsed = false;
  bool is_bym2_collapsed = false;
  bool is_gp_collapsed = false;
  bool is_multiscale_gp = false;
  bool is_hsgp = false;
  bool has_temporal = false;
  bool is_ar1 = false;
  bool is_temporal_gp = false;
  bool has_multiscale_temporal = false;
  bool has_zi = false;
  bool has_oi = false;  // For OI-binomial and ZOIB models
  bool has_svc = false;
  bool has_latent = false;
  bool has_spatiotemporal = false;
  bool is_st_gp = false;
  bool has_tvc = false;
};

ParamLayout compute_param_layout(const ModelData& data);
int get_n_params(const ModelData& data);

// =====================================================================
// Log-posterior computation (with OpenMP parallelization)
// =====================================================================

// Main log-posterior function: compute_log_post_impl<double>, so that a term
// cannot be present in the density the H path evaluates and absent from the one
// the autodiff modes differentiate.
//
// When skip_obs_loop=true, returns only prior+structural terms (O(p+S+T)),
// skipping the O(N) observation loop. Used by fused gradient+log_post
// computation, which passes the temporal GP's prior in the same way when it has
// already accumulated it.
double compute_log_post(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    bool skip_obs_loop = false,
    const double* precomputed_tgp_log_prior = nullptr
);

// Gradient computation (with optional fused log-posterior)
// When log_post_out is non-null, the log-posterior is computed alongside
// the gradient in a single pass, avoiding redundant O(N) computation.
void compute_gradient(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    std::vector<double>& grad,
    double* log_post_out = nullptr
);

// Compute ICAR quadratic form: phi' Q phi
double icar_quadratic_form(
    const std::vector<double>& phi,
    const ModelData& data
);

// =====================================================================
// HMC/NUTS sampler structures
// =====================================================================

struct LeapfrogResult {
  std::vector<double> q;
  std::vector<double> p;
  double log_prob;
  bool divergent;
};

// NUTS-specific structures
struct LeapfrogResultWithGrad {
  std::vector<double> q, p, grad;
  double log_prob;
  bool divergent;
};

struct NUTSNode {
  std::vector<double> q, p, grad;
  double log_prob;
};

struct NUTSTreeResult {
  NUTSNode left, right;
  std::vector<double> q_proposal;
  std::vector<double> grad_proposal;
  double log_prob_proposal;
  int n_valid;
  bool divergent, stop;
  double sum_accept_prob;
  int n_leapfrog;
  double sum_log_weight;
};

// =====================================================================
// Optimized NUTS infrastructure (zero-allocation tree building)
// =====================================================================

// Result from in-place leapfrog (no vectors allocated)
struct LeapfrogInPlaceResult {
  double log_prob;
  bool divergent;
};

// Lightweight tree statistics (returned by build_tree_fast)
// Endpoints tracked via workspace slot indices, not owned vectors
struct TreeStats {
  int left_slot;            // Workspace slot index for left endpoint
  int right_slot;           // Workspace slot index for right endpoint
  int proposal_slot;        // Workspace slot index for proposal
  double sum_log_weight;
  double sum_accept_prob;
  double log_prob_proposal;
  int n_valid;
  int n_leapfrog;
  bool divergent;
  bool stop;

  // Generalized U-turn criterion (Betancourt 2017, Stan implementation)
  std::vector<double> rho;          // Cumulative momentum sum across trajectory
  std::vector<double> p_sharp_beg;  // M^{-1} * p at trajectory beginning
  std::vector<double> p_sharp_end;  // M^{-1} * p at trajectory end
  std::vector<double> p_beg;        // Raw momentum at trajectory beginning
  std::vector<double> p_end;        // Raw momentum at trajectory end

  // Pre-allocate all U-turn vectors to avoid heap allocation at every leaf node
  void init_vectors(int n) {
    rho.resize(n);
    p_sharp_beg.resize(n);
    p_sharp_end.resize(n);
    p_beg.resize(n);
    p_end.resize(n);
  }
};

// Generalized U-turn criterion: check if momenta are aligned with integrated direction
// Returns true if trajectory should CONTINUE (no U-turn detected)
inline bool compute_criterion(const double* p_sharp_minus, const double* p_sharp_plus,
                              const double* rho, int n) {
  double dot_fwd = 0.0, dot_bwd = 0.0;
  int i = 0;
  for (; i + 3 < n; i += 4) {
    dot_fwd += p_sharp_plus[i]   * rho[i]   + p_sharp_plus[i+1] * rho[i+1]
             + p_sharp_plus[i+2] * rho[i+2] + p_sharp_plus[i+3] * rho[i+3];
    dot_bwd += p_sharp_minus[i]   * rho[i]   + p_sharp_minus[i+1] * rho[i+1]
             + p_sharp_minus[i+2] * rho[i+2] + p_sharp_minus[i+3] * rho[i+3];
  }
  for (; i < n; i++) {
    dot_fwd += p_sharp_plus[i] * rho[i];
    dot_bwd += p_sharp_minus[i] * rho[i];
  }
  return (dot_fwd > 0.0) && (dot_bwd > 0.0);
}

// Fused U-turn criterion: constructs rho = a + b and computes dot products in one O(n) pass.
// Also writes constructed rho to rho_out for downstream use.
inline bool compute_criterion_fused(const double* p_sharp_minus, const double* p_sharp_plus,
                                    const double* a, const double* b,
                                    double* rho_out, int n) {
  double dot_fwd = 0.0, dot_bwd = 0.0;
  int i = 0;
  for (; i + 3 < n; i += 4) {
    double r0 = a[i]   + b[i];
    double r1 = a[i+1] + b[i+1];
    double r2 = a[i+2] + b[i+2];
    double r3 = a[i+3] + b[i+3];
    rho_out[i]   = r0; rho_out[i+1] = r1;
    rho_out[i+2] = r2; rho_out[i+3] = r3;
    dot_fwd += p_sharp_plus[i]   * r0 + p_sharp_plus[i+1]  * r1
             + p_sharp_plus[i+2] * r2 + p_sharp_plus[i+3]  * r3;
    dot_bwd += p_sharp_minus[i]   * r0 + p_sharp_minus[i+1] * r1
             + p_sharp_minus[i+2] * r2 + p_sharp_minus[i+3] * r3;
  }
  for (; i < n; i++) {
    double r = a[i] + b[i];
    rho_out[i] = r;
    dot_fwd += p_sharp_plus[i] * r;
    dot_bwd += p_sharp_minus[i] * r;
  }
  return (dot_fwd > 0.0) && (dot_bwd > 0.0);
}

// Pre-allocated buffer pool for NUTS tree building
// Eliminates all heap allocations in the build_tree hot path
struct NUTSWorkspace {
  int n;                    // n_params
  int max_depth;
  int stride;               // 3 * n (q + p + grad per slot)
  int total_slots;

  // Resolved gradient function pointer (set once, avoids per-call dispatch)
  GradientFn gradient_fn = nullptr;

  // Contiguous pool: [slot][q|p|grad], each n doubles wide
  std::vector<double> pool;
  std::vector<double> log_probs;  // One per slot

  // Slot allocator (stack-based, reset per top-level build_tree call)
  int next_slot;

  // Working buffers for compute_gradient (needs std::vector interface)
  std::vector<double> params_buf;
  std::vector<double> grad_buf;

  // Scratch buffer for dense mass matrix matvec (p doubles)
  std::vector<double> dense_scratch;

  // Pre-allocated merge buffers for build_tree_fast (depth-indexed, no per-merge alloc)
  // 4 buffers per depth level: rho_init, p_init_end, p_sharp_init_end, rho_check
  std::vector<double> merge_pool;
  static constexpr int MERGE_RHO_INIT = 0;
  static constexpr int MERGE_P_INIT_END = 1;
  static constexpr int MERGE_PSHARP_INIT_END = 2;
  static constexpr int MERGE_RHO_CHECK = 3;
  static constexpr int MERGE_BUFS_PER_DEPTH = 4;

  double* merge_buf(int depth, int buf_idx) {
    return &merge_pool[static_cast<size_t>(depth * MERGE_BUFS_PER_DEPTH + buf_idx) * n];
  }

  // Pre-allocated iteration-level vectors (reused across NUTS iterations)
  std::vector<double> iter_rho, iter_rho_bck, iter_rho_fwd;
  std::vector<double> iter_p_sharp_init;
  std::vector<double> iter_p_fwd_beg, iter_p_fwd_end;
  std::vector<double> iter_p_bck_beg, iter_p_bck_end;
  std::vector<double> iter_p_sharp_fwd_beg, iter_p_sharp_fwd_end;
  std::vector<double> iter_p_sharp_bck_beg, iter_p_sharp_bck_end;
  std::vector<double> iter_rho_seam;

  // Slot layout:
  //   Slots 0, 1: persistent trajectory endpoints (node_left, node_right)
  //   Slots 2+: allocated by build_tree_fast via alloc_slot()
  static constexpr int NODE_LEFT_SLOT = 0;
  static constexpr int NODE_RIGHT_SLOT = 1;
  static constexpr int TREE_START_SLOT = 2;

  void init(int np, int max_d) {
    n = np;
    max_depth = max_d;
    stride = 3 * np;
    // Max slots: 2 persistent + 2^max_depth for tree (generous upper bound)
    total_slots = 2 + (1 << max_d);
    pool.resize(static_cast<size_t>(total_slots) * stride);
    log_probs.resize(total_slots);
    params_buf.resize(np);
    grad_buf.resize(np);
    dense_scratch.resize(np);
    next_slot = TREE_START_SLOT;
    // Pre-allocate merge buffers: 4 per depth level
    merge_pool.resize(static_cast<size_t>(MERGE_BUFS_PER_DEPTH) * max_d * np, 0.0);
    // Pre-allocate iteration-level vectors
    iter_rho.resize(np);
    iter_rho_bck.resize(np);
    iter_rho_fwd.resize(np);
    iter_p_sharp_init.resize(np);
    iter_p_fwd_beg.resize(np);
    iter_p_fwd_end.resize(np);
    iter_p_bck_beg.resize(np);
    iter_p_bck_end.resize(np);
    iter_p_sharp_fwd_beg.resize(np);
    iter_p_sharp_fwd_end.resize(np);
    iter_p_sharp_bck_beg.resize(np);
    iter_p_sharp_bck_end.resize(np);
    iter_rho_seam.resize(np);
  }

  // Allocate a fresh slot (stack-based, no deallocation)
  // Returns -1 if workspace is exhausted
  int alloc_slot() {
    if (next_slot >= total_slots) return -1;
    return next_slot++;
  }

  // Reset allocator for new top-level build_tree call
  // Preserves persistent slots 0 and 1
  void reset_tree() {
    next_slot = TREE_START_SLOT;
  }

  // Access helpers (raw pointers into contiguous pool)
  double* q_at(int slot) { return &pool[slot * stride]; }
  double* p_at(int slot) { return &pool[slot * stride + n]; }
  double* grad_at(int slot) { return &pool[slot * stride + 2 * n]; }
  double& logp_at(int slot) { return log_probs[slot]; }

  // Copy full node (q + p + grad + log_prob) between slots
  void copy_node(int dst, int src) {
    std::memcpy(&pool[dst * stride], &pool[src * stride],
                stride * sizeof(double));
    log_probs[dst] = log_probs[src];
  }

  // Load node data from std::vector sources into a slot
  void load_node(int slot, const double* q, const double* p,
                 const double* grad, double log_prob) {
    std::memcpy(q_at(slot), q, n * sizeof(double));
    std::memcpy(p_at(slot), p, n * sizeof(double));
    std::memcpy(grad_at(slot), grad, n * sizeof(double));
    log_probs[slot] = log_prob;
  }
};

// =====================================================================
// Block-diagonal mass block (max 4×4, stack-allocated)
// =====================================================================

struct MassBlock {
  int start = 0;          // First param index in full parameter vector
  int size = 0;           // Block size (2-4)
  bool adapted = false;

  // Block mass storage (column-major, max 4×4)
  double inv_mass[16] = {};    // C_block (covariance block)
  double L_inv_mass[16] = {};  // Cholesky L where LL^T = C_block

  // Block-local Welford covariance accumulator
  int welford_n = 0;
  double welford_mean[4] = {};
  double welford_M2[16] = {};  // Running sum for covariance (column-major)

  void init(int s, int sz) {
    start = s;
    size = sz;
    adapted = false;
    std::memset(inv_mass, 0, sizeof(inv_mass));
    std::memset(L_inv_mass, 0, sizeof(L_inv_mass));
    // Initialize as identity
    for (int i = 0; i < sz; i++) {
      inv_mass[i * 4 + i] = 1.0;  // Using stride=4 (max block size)
      L_inv_mass[i * 4 + i] = 1.0;
    }
    reset_welford();
  }

  void reset_welford() {
    welford_n = 0;
    std::memset(welford_mean, 0, sizeof(welford_mean));
    std::memset(welford_M2, 0, sizeof(welford_M2));
  }

  // Extract block params and update Welford running stats
  void welford_update(const double* full_params) {
    welford_n++;
    double delta[4];
    for (int i = 0; i < size; i++) {
      delta[i] = full_params[start + i] - welford_mean[i];
      welford_mean[i] += delta[i] / welford_n;
    }
    for (int i = 0; i < size; i++) {
      double dx_new = full_params[start + i] - welford_mean[i];
      for (int j = 0; j <= i; j++) {
        double val = dx_new * delta[j];
        welford_M2[j * 4 + i] += val;  // stride=4
        if (i != j) {
          welford_M2[i * 4 + j] += val;
        }
      }
    }
  }

  // Compute covariance from Welford stats, Cholesky decompose, set adapted
  bool update_from_welford() {
    if (welford_n < 10) return false;

    // Compute sample covariance with small regularization
    double cov[16] = {};
    double scale = 1.0 / (welford_n - 1);
    for (int i = 0; i < size; i++) {
      for (int j = 0; j < size; j++) {
        cov[i * 4 + j] = welford_M2[j * 4 + i] * scale;  // Note: M2 is col-major with stride 4
      }
      cov[i * 4 + i] += 1e-8;  // Regularization
    }

    // Try Cholesky decomposition
    double L[16] = {};
    if (!cholesky_small(cov, L, size)) return false;

    // Success: copy to block storage
    std::memcpy(inv_mass, cov, sizeof(inv_mass));
    std::memcpy(L_inv_mass, L, sizeof(L_inv_mass));
    adapted = true;
    return true;
  }

  // Tiny Cholesky for k<=4 (direct formula, no Eigen)
  // A and L use stride=4 (max block size)
  static bool cholesky_small(const double* A, double* L, int k) {
    std::memset(L, 0, 16 * sizeof(double));
    for (int i = 0; i < k; i++) {
      double sum = 0.0;
      for (int p = 0; p < i; p++) {
        sum += L[i * 4 + p] * L[i * 4 + p];
      }
      double diag = A[i * 4 + i] - sum;
      if (diag <= 0.0) return false;
      L[i * 4 + i] = std::sqrt(diag);
      for (int j = i + 1; j < k; j++) {
        double s = 0.0;
        for (int p = 0; p < i; p++) {
          s += L[j * 4 + p] * L[i * 4 + p];
        }
        L[j * 4 + i] = (A[j * 4 + i] - s) / L[i * 4 + i];
      }
    }
    return true;
  }

  // result[0..size-1] = C_block * p[start..start+size-1]
  void matvec(const double* p_full, double* result) const {
    const double* pb = p_full + start;
    for (int i = 0; i < size; i++) {
      double sum = 0.0;
      for (int j = 0; j < size; j++) {
        sum += inv_mass[i * 4 + j] * pb[j];
      }
      result[i] = sum;
    }
  }

  // p_block^T * C_block * p_block
  double quadform(const double* p_full) const {
    const double* pb = p_full + start;
    double result = 0.0;
    for (int i = 0; i < size; i++) {
      for (int j = 0; j < size; j++) {
        result += pb[i] * inv_mass[i * 4 + j] * pb[j];
      }
    }
    return result;
  }

  // Sample momentum for block: p_block = L^{-T} z (back-substitution on tiny L)
  void sample_momentum(double* p_full, std::mt19937& rng) const {
    if (!adapted) return;  // Non-adapted blocks use diagonal path
    std::normal_distribution<double> normal(0.0, 1.0);
    double z[4];
    for (int i = 0; i < size; i++) z[i] = normal(rng);

    // Back-substitution: solve L^T * p = z (upper triangular)
    double* pb = p_full + start;
    for (int i = size - 1; i >= 0; i--) {
      double sum = z[i];
      for (int j = i + 1; j < size; j++) {
        sum -= L_inv_mass[j * 4 + i] * pb[j];  // L^T[i][j] = L[j][i]
      }
      pb[i] = sum / L_inv_mass[i * 4 + i];
    }
  }
};

// =====================================================================
// Precision-informed mass block (heap-allocated, arbitrary size)
// Used for ICAR/BYM2 spatial params where Q (precision) is known analytically.
// Unlike MassBlock (≤4, stack), this handles S×S blocks (S=50 typical).
// NOT adapted from samples — uses fixed analytical precision.
// =====================================================================

struct PrecisionBlock {
  int start = 0;        // First param index in full parameter vector
  int size = 0;         // Block dimension S
  bool active = false;

  // M^{-1} = Q_reg_inv: (Q + lambda*I)^{-1}, column-major S×S
  std::vector<double> Q_inv;
  // L_Q: Cholesky factor where L*L^T = Q + lambda*I, column-major S×S
  // Used for momentum sampling: p_block = L^{-T} * z
  std::vector<double> L_chol;

  void init(int s, int sz, const double* q_inv_data, const double* l_chol_data) {
    start = s;
    size = sz;
    active = true;
    int nn = static_cast<int>(static_cast<size_t>(sz) * sz);
    Q_inv.assign(q_inv_data, q_inv_data + nn);
    L_chol.assign(l_chol_data, l_chol_data + nn);
  }

  // result[0..size-1] = Q_inv * p[start..start+size-1]
  void matvec(const double* p_full, double* result) const {
    const double* pb = p_full + start;
    Eigen::Map<const Eigen::MatrixXd> M(Q_inv.data(), size, size);
    Eigen::Map<const Eigen::VectorXd> pv(pb, size);
    Eigen::Map<Eigen::VectorXd> rv(result, size);
    rv.noalias() = M.selfadjointView<Eigen::Lower>() * pv;
  }

  // p_block^T * Q_inv * p_block
  double quadform(const double* p_full) const {
    const double* pb = p_full + start;
    Eigen::Map<const Eigen::MatrixXd> M(Q_inv.data(), size, size);
    Eigen::Map<const Eigen::VectorXd> pv(pb, size);
    return pv.dot(M.selfadjointView<Eigen::Lower>() * pv);
  }

  // Sample momentum for block: solve L^T * p = z (back-substitution)
  void sample_momentum(double* p_full, std::mt19937& rng) const {
    std::normal_distribution<double> normal(0.0, 1.0);
    std::vector<double> z(size);
    for (int i = 0; i < size; i++) z[i] = normal(rng);
    double* pb = p_full + start;
    Eigen::Map<const Eigen::MatrixXd> L(L_chol.data(), size, size);
    Eigen::Map<const Eigen::VectorXd> zv(z.data(), size);
    Eigen::Map<Eigen::VectorXd> pv(pb, size);
    pv.noalias() = L.transpose().triangularView<Eigen::Upper>().solve(zv);
  }
};

// =====================================================================
// Kronecker precision block for spatiotemporal (ST) interaction params.
// M = Q_space ⊗ Q_time, M^{-1} = Q_space^{-1} ⊗ Q_time^{-1}
// Never forms the full (S*T)×(S*T) matrix — O(S*T*(S+T)) operations.
// =====================================================================

struct KroneckerBlock {
  int start = 0;        // First ST param index in full parameter vector
  int S = 0;            // Spatial dimension
  int T = 0;            // Temporal dimension
  bool active = false;

  // Spatial: Q_space_inv (S×S), L_space (Cholesky of Q_space, S×S)
  std::vector<double> Qs_inv;  // column-major
  std::vector<double> Ls;      // column-major

  // Temporal: Q_time_inv (T×T), L_time (Cholesky of Q_time, T×T)
  std::vector<double> Qt_inv;  // column-major
  std::vector<double> Lt;      // column-major

  // Scratch buffers (pre-allocated)
  mutable std::vector<double> scratch_ST;  // S*T work buffer

  void init(int st, int ns, int nt,
            const double* qs_inv, const double* ls,
            const double* qt_inv, const double* lt) {
    start = st;
    S = ns;
    T = nt;
    active = true;
    int ss = S * S, tt = T * T;
    Qs_inv.assign(qs_inv, qs_inv + ss);
    Ls.assign(ls, ls + ss);
    Qt_inv.assign(qt_inv, qt_inv + tt);
    Lt.assign(lt, lt + tt);
    scratch_ST.resize(static_cast<size_t>(S) * T);
  }

  // Compute (A ⊗ B) * vec(X) = vec(B * X * A^T)
  // where X is S×T (column-major), A is T×T, B is S×S
  // result = vec(B * X * A^T)
  // This is the standard Kronecker-vector product identity.
  void kron_matvec(const double* As, int na, const double* Bt, int nb,
                   const double* x, double* result) const {
    // Step 1: tmp = X * A^T  (S×T * T×T = S×T)
    // X is S×T column-major, A^T is T×T
    Eigen::Map<const Eigen::MatrixXd> X(x, S, T);
    Eigen::Map<const Eigen::MatrixXd> A(Bt, T, T);  // temporal
    Eigen::Map<const Eigen::MatrixXd> B(As, S, S);   // spatial
    Eigen::Map<Eigen::MatrixXd> R(result, S, T);

    // (B ⊗ A) * vec(X) = vec(B * X * A^T)
    // But our params are stored with spatial varying fastest: param[s + t*S]
    // So X_{s,t} = x[s + t*S] which is column-major S×T.
    // M^{-1} = Qs_inv ⊗ Qt_inv
    // M^{-1} * x = vec(Qs_inv * X * Qt_inv^T)
    R.noalias() = B * X * A.transpose();
  }

  // result[0..S*T-1] = (Qs_inv ⊗ Qt_inv) * p[start..start+S*T-1]
  void matvec(const double* p_full, double* result) const {
    kron_matvec(Qs_inv.data(), S, Qt_inv.data(), T,
                p_full + start, result);
  }

  // p_block^T * (Qs_inv ⊗ Qt_inv) * p_block
  double quadform(const double* p_full) const {
    matvec(p_full, scratch_ST.data());
    const double* pb = p_full + start;
    double qf = 0.0;
    int ST = S * T;
    for (int i = 0; i < ST; i++) qf += pb[i] * scratch_ST[i];
    return qf;
  }

  // Sample momentum: p ~ N(0, M) where M = Qs ⊗ Qt
  // p = (Ls ⊗ Lt)^{-T} * z = vec(Ls^{-T} * Z * Lt^{-1})
  // where Z is S×T standard normal
  void sample_momentum(double* p_full, std::mt19937& rng) const {
    std::normal_distribution<double> normal(0.0, 1.0);
    int ST = S * T;
    // Generate Z ~ N(0,I) as S×T matrix
    std::vector<double> Z(ST);
    for (int i = 0; i < ST; i++) Z[i] = normal(rng);

    double* pb = p_full + start;
    Eigen::Map<Eigen::MatrixXd> Zm(Z.data(), S, T);
    Eigen::Map<const Eigen::MatrixXd> Lsm(Ls.data(), S, S);
    Eigen::Map<const Eigen::MatrixXd> Ltm(Lt.data(), T, T);
    Eigen::Map<Eigen::MatrixXd> Pm(pb, S, T);

    // (Ls ⊗ Lt)^{-T} * z = vec(Ls^{-T} * Z * Lt^{-1})
    // Step 1: solve Ls^T * tmp = Z  →  tmp = Ls^{-T} * Z
    Eigen::MatrixXd tmp = Lsm.transpose().triangularView<Eigen::Upper>().solve(Zm);
    // Step 2: solve tmp2 * Lt^T = tmp  →  tmp2 = tmp * Lt^{-T}
    //   which is (Lt^{-T} * tmp^T)^T = (Lt * tmp^T)^{-T}
    //   Actually: tmp2 * Lt^T = tmp → tmp2 = tmp * Lt^{-T}
    //   Transpose: Lt^{-1} * tmp^T → solve Lt * Y = tmp^T → Y = Lt^{-1} * tmp^T
    //   Then result = Y^T
    Eigen::MatrixXd Y = Ltm.triangularView<Eigen::Lower>().solve(tmp.transpose());
    Pm = Y.transpose();
  }
};

// =====================================================================
// Sparse GMRF block for ST_IV spatiotemporal interaction.
// Uses Eigen sparse Cholesky for:
//   1. Block Gibbs sampling: delta ~ N(Q^{-1}b, Q^{-1})
//   2. Mass matrix operations (momentum, kinetic energy, inv_mass*p)
// Q = tau * (Q_s ⊗ Q_t) + lambda*I (posterior precision for delta block)
// =====================================================================

struct SparseGMRFBlock {
  int start = 0;        // First ST param index in full parameter vector
  int S = 0;            // Spatial dimension
  int T = 0;            // Temporal dimension
  bool active = false;
  bool factorized = false;

  // Sparse precision and its Cholesky factorization
  Eigen::SparseMatrix<double> Q_sparse;  // ST×ST sparse precision matrix
  Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> llt;  // Cholesky LL^T = Q

  // Scratch buffers (pre-allocated)
  mutable Eigen::VectorXd scratch_vec;

  void init(int st_start, int ns, int nt) {
    start = st_start;
    S = ns;
    T = nt;
    active = true;
    factorized = false;
    int ST = S * T;
    scratch_vec.resize(ST);
  }

  // Build Q_sparse = tau * (Q_s ⊗ Q_t) + diag_correction
  // adj_row_ptr/adj_col_idx: CSR adjacency for spatial graph (1-based col_idx!)
  // temp_type: RW1, RW2, AR1
  // tau: precision parameter
  // h_lik: diagonal of likelihood Hessian (length ST), or nullptr for prior-only
  // diag_ridge: Tikhonov ridge lambda*I added to the diagonal so the Kronecker
  //   of two intrinsic precisions factorizes. This is NOT the sum-to-zero
  //   penalty, which is rank-1 per margin (lambda*11' over each margin, see
  //   st_sum_to_zero_penalty) and does not fill the Kronecker null space. The
  //   two are different objects and take different constants; sizing this one
  //   from s2z_precision would put a ~1e6 ridge on the factorization.
  void build_and_factorize(
      const std::vector<int>& adj_row_ptr,
      const std::vector<int>& adj_col_idx,
      ratiod_temporal::TemporalType temp_type,
      bool cyclic,
      double tau,
      const double* h_lik,  // diagonal Hessian correction (length S*T), can be nullptr
      double diag_ridge = 0.001
  ) {
    int ST = S * T;

    // Build spatial Laplacian Q_s (S×S)
    // Q_s[i,i] = degree(i), Q_s[i,j] = -1 if adjacent
    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(ST * 7);  // Rough estimate: ~7 nonzeros per row

    // Build temporal precision Q_t (T×T) as dense small matrix
    Eigen::MatrixXd Qt = Eigen::MatrixXd::Zero(T, T);
    if (temp_type == ratiod_temporal::TemporalType::RW1) {
      for (int t = 0; t < T - 1; t++) {
        Qt(t, t) += 1.0;
        Qt(t + 1, t + 1) += 1.0;
        Qt(t, t + 1) = -1.0;
        Qt(t + 1, t) = -1.0;
      }
    } else if (temp_type == ratiod_temporal::TemporalType::RW2) {
      for (int t = 0; t < T - 2; t++) {
        Qt(t, t) += 1.0;
        Qt(t + 1, t + 1) += 4.0;
        Qt(t + 2, t + 2) += 1.0;
        Qt(t, t + 1) += -2.0;
        Qt(t + 1, t) += -2.0;
        Qt(t, t + 2) += 1.0;
        Qt(t + 2, t) += 1.0;
        Qt(t + 1, t + 2) += -2.0;
        Qt(t + 2, t + 1) += -2.0;
      }
      // Fix double-counted diagonal
      for (int t = 0; t < T; t++) Qt(t, t) = 0.0;
      Eigen::MatrixXd D = Eigen::MatrixXd::Zero(T - 2, T);
      for (int t = 0; t < T - 2; t++) {
        D(t, t) = 1.0; D(t, t + 1) = -2.0; D(t, t + 2) = 1.0;
      }
      Qt = D.transpose() * D;
    } else {
      // AR1 with rho=0.5 as default approximation
      for (int t = 0; t < T; t++) Qt(t, t) = 1.0;
      for (int t = 0; t < T - 1; t++) {
        Qt(t, t + 1) = -0.5;
        Qt(t + 1, t) = -0.5;
      }
    }

    // Build Kronecker product Q_kron = Q_s ⊗ Q_t as sparse triplets
    // (Q_s ⊗ Q_t)[(s1*T+t1), (s2*T+t2)] = Q_s[s1,s2] * Q_t[t1,t2]
    // For each spatial pair (s1,s2) with Q_s[s1,s2] != 0:
    //   For each temporal pair (t1,t2) with Q_t[t1,t2] != 0:
    //     Add Q_s[s1,s2] * Q_t[t1,t2] at row s1*T+t1, col s2*T+t2
    for (int s1 = 0; s1 < S; s1++) {
      int n_neigh = adj_row_ptr[s1 + 1] - adj_row_ptr[s1];
      double qs_diag = static_cast<double>(n_neigh);

      // Diagonal spatial block: Q_s[s1,s1] = degree
      for (int t1 = 0; t1 < T; t1++) {
        for (int t2 = 0; t2 < T; t2++) {
          if (Qt(t1, t2) != 0.0) {
            triplets.emplace_back(s1 * T + t1, s1 * T + t2, tau * qs_diag * Qt(t1, t2));
          }
        }
      }

      // Off-diagonal spatial: Q_s[s1,s2] = -1 for neighbors
      for (int idx = adj_row_ptr[s1]; idx < adj_row_ptr[s1 + 1]; idx++) {
        int s2 = adj_col_idx[idx] - 1;  // Convert 1-based to 0-based
        for (int t1 = 0; t1 < T; t1++) {
          for (int t2 = 0; t2 < T; t2++) {
            if (Qt(t1, t2) != 0.0) {
              triplets.emplace_back(s1 * T + t1, s2 * T + t2, tau * (-1.0) * Qt(t1, t2));
            }
          }
        }
      }
    }

    // Add diagonal corrections: likelihood Hessian + ridge + regularization
    double reg = 1e-6;  // Numerical regularization for rank deficiency
    for (int k = 0; k < ST; k++) {
      double diag_add = diag_ridge + reg;
      if (h_lik) diag_add += h_lik[k];     // likelihood curvature
      triplets.emplace_back(k, k, diag_add);
    }

    Q_sparse.resize(ST, ST);
    Q_sparse.setFromTriplets(triplets.begin(), triplets.end());
    Q_sparse.makeCompressed();

    // Cholesky factorization
    llt.compute(Q_sparse);
    factorized = (llt.info() == Eigen::Success);
  }

  // Sample delta from GMRF conditional: delta ~ N(Q^{-1}b, Q^{-1})
  // b = grad_delta_lik (likelihood gradient wrt delta)
  // Returns new delta values in delta_out (length S*T)
  void sample_conditional(const double* b, double* delta_out, std::mt19937& rng) const {
    if (!factorized) return;
    int ST = S * T;
    std::normal_distribution<double> normal(0.0, 1.0);

    // Step 1: Solve Q * mean = b  →  mean = Q^{-1} * b
    Eigen::Map<const Eigen::VectorXd> bv(b, ST);
    Eigen::VectorXd mean = llt.solve(bv);

    // Step 2: Sample z ~ N(0, I)
    Eigen::VectorXd z(ST);
    for (int i = 0; i < ST; i++) z[i] = normal(rng);

    // Step 3: delta = mean + perturbation from N(0, Q^{-1})
    // SimplicialLLT factorizes as: P * Q * P^T = L * L^T
    // To sample from N(0, Q^{-1}): pert = P^T * L^{-T} * z
    //   Var(pert) = P^T L^{-T} L^{-1} P = P^T (LL^T)^{-1} P = (P^T LL^T P)^{-1} = Q^{-1} ✓
    auto perm = llt.permutationP();
    // Get L as a concrete sparse matrix (avoids const-view issues)
    Eigen::SparseMatrix<double> L_mat = llt.matrixL();
    // Solve L^T * v = z (upper triangular solve on L^T)
    Eigen::SparseMatrix<double> Lt_mat = L_mat.transpose();
    Eigen::VectorXd v = Lt_mat.triangularView<Eigen::Upper>().solve(z);
    // Un-permute: pert = P^T * v
    Eigen::VectorXd pert = perm.transpose() * v;

    Eigen::Map<Eigen::VectorXd> out(delta_out, ST);
    out = mean + pert;
  }

  // Mass matrix operations (for HMC momentum/kinetic energy)
  // M = Q (posterior precision), M^{-1} = Q^{-1}

  // result = Q^{-1} * p  (inv_mass * momentum)
  void inv_mass_matvec(const double* p_full, double* result) const {
    if (!factorized) return;
    int ST = S * T;
    Eigen::Map<const Eigen::VectorXd> pv(p_full + start, ST);
    Eigen::VectorXd sol = llt.solve(pv);
    std::memcpy(result, sol.data(), ST * sizeof(double));
  }

  // p^T * Q^{-1} * p  (kinetic energy contribution)
  double quadform(const double* p_full) const {
    if (!factorized) return 0.0;
    int ST = S * T;
    Eigen::Map<const Eigen::VectorXd> pv(p_full + start, ST);
    Eigen::VectorXd sol = llt.solve(pv);
    return pv.dot(sol);
  }

  // Sample momentum p ~ N(0, Q):  p = L^T * z where LL^T = Q
  void sample_momentum(double* p_full, std::mt19937& rng) const {
    if (!factorized) return;
    int ST = S * T;
    std::normal_distribution<double> normal(0.0, 1.0);
    Eigen::VectorXd z(ST);
    for (int i = 0; i < ST; i++) z[i] = normal(rng);

    // P * Q * P^T = L * L^T  →  p ~ N(0, Q) needs p = P^T * L^T * P * z
    // But simpler: p_perm = L^T * z ~ N(0, L^T L) = N(0, PQP^T)
    // Then p = P^T * p_perm ~ N(0, P^T PQP^T P) = N(0, Q) ✓
    auto perm = llt.permutationP();
    Eigen::SparseMatrix<double> L_mat = llt.matrixL();
    Eigen::VectorXd p_perm = L_mat.transpose() * z;  // L^T * z
    Eigen::VectorXd p_vec = perm.transpose() * p_perm;  // P^T * (L^T * z)
    double* pb = p_full + start;
    std::memcpy(pb, p_vec.data(), ST * sizeof(double));
  }
};

// =====================================================================
// Dense mass matrix for NUTS (encapsulates diag + dense state)
// =====================================================================

struct DenseMassMatrix {
  int n = 0;                              // Dimension
  MassMatrixType type = MassMatrixType::DIAG;
  bool adapted = false;

  // Diagonal (always available, used as fallback)
  std::vector<double> inv_mass_diag;      // M^{-1} diagonal = variance
  std::vector<double> sqrt_mass_diag;     // sqrt(M) diagonal = 1/sqrt(variance) for p sampling

  // Dense (only when type == DENSE)
  std::vector<double> inv_mass_dense;     // Full p×p M^{-1} = regularized sample covariance (column-major)
  std::vector<double> L_inv_mass;         // Cholesky factor L where LL^T = M^{-1} (column-major)

  // Scratch buffer for dense matvec results (avoids per-call allocation)
  std::vector<double> scratch;

  // Block-diagonal (only when type == BLOCK_DIAG)
  std::vector<MassBlock> blocks;
  std::vector<bool> in_block;  // in_block[i] = true if param i belongs to a block

  // Precision-informed blocks (optional, independent of type)
  // These override the mass for specific param ranges using known precision structure.
  PrecisionBlock precision_block;    // ICAR/BYM2 spatial block (DISABLED)
  KroneckerBlock kronecker_block;    // ST_IV Kronecker block (DISABLED)
  SparseGMRFBlock sparse_gmrf;       // ST_IV sparse GMRF mass + Gibbs sampling

  void init(int dim, MassMatrixType t) {
    n = dim;
    type = t;
    adapted = false;
    inv_mass_diag.assign(dim, 1.0);
    sqrt_mass_diag.assign(dim, 1.0);
    scratch.resize(dim);
    if (t == MassMatrixType::DENSE) {
      inv_mass_dense.assign(static_cast<size_t>(dim) * dim, 0.0);
      L_inv_mass.assign(static_cast<size_t>(dim) * dim, 0.0);
      // Initialize as identity
      for (int i = 0; i < dim; i++) {
        inv_mass_dense[static_cast<size_t>(i) * dim + i] = 1.0;
        L_inv_mass[static_cast<size_t>(i) * dim + i] = 1.0;
      }
    }
    if (t == MassMatrixType::BLOCK_DIAG) {
      in_block.assign(dim, false);
    }
  }

  // Initialize block-diagonal structure from block specifications
  // block_specs: vector of (start_index, block_size) pairs
  void init_block_diag(int dim, const std::vector<std::pair<int,int>>& block_specs) {
    init(dim, MassMatrixType::BLOCK_DIAG);
    blocks.clear();
    blocks.reserve(block_specs.size());
    for (const auto& spec : block_specs) {
      MassBlock blk;
      blk.init(spec.first, spec.second);
      blocks.push_back(blk);
      for (int i = spec.first; i < spec.first + spec.second; i++) {
        if (i < dim) in_block[i] = true;
      }
    }
  }

  // Update dense mass matrix from sample covariance
  // Returns true on success, false on Cholesky failure (degrades to diagonal)
  // Uses Eigen LLT for Cholesky decomposition
  bool update_from_covariance(const double* cov, int n_samples);

  // Sample momentum: p ~ N(0, M) where M = C^{-1}
  // DIAG: p[i] = z * sqrt_mass_diag[i]
  // BLOCK_DIAG: diagonal for non-block params, L^{-T} z for block params
  // DENSE: solve L^T * p = z  (back-substitution)
  // Uses Eigen triangular solve for dense case (n>=16) for SIMD acceleration.
  void sample_momentum(double* p, std::mt19937& rng) const {
    std::normal_distribution<double> normal(0.0, 1.0);
    if (type == MassMatrixType::BLOCK_DIAG && adapted) {
      // First: diagonal for all params
      for (int i = 0; i < n; i++) {
        p[i] = normal(rng) * sqrt_mass_diag[i];
      }
      // Then: overwrite block params with correlated samples
      for (const auto& blk : blocks) {
        if (blk.adapted) {
          blk.sample_momentum(p, rng);
        }
      }
    } else if (type == MassMatrixType::DIAG || !adapted) {
      for (int i = 0; i < n; i++) {
        p[i] = normal(rng) * sqrt_mass_diag[i];
      }
    } else {
      // Dense: p = L^{-T} z where LL^T = C (inv_mass)
      // We need p ~ N(0, C^{-1}), so sample z ~ N(0, I), then p = L^{-T} z
      std::vector<double> z(n);
      for (int i = 0; i < n; i++) {
        z[i] = normal(rng);
      }
      if (n >= 16) {
        Eigen::Map<const Eigen::MatrixXd> Lm(L_inv_mass.data(), n, n);
        Eigen::Map<const Eigen::VectorXd> zv(z.data(), n);
        Eigen::Map<Eigen::VectorXd> pv(p, n);
        // Solve L^T * p = z: transpose L then use upper-triangular solve
        pv.noalias() = Lm.transpose().triangularView<Eigen::Upper>().solve(zv);
      } else {
        ratiod_linalg::tri_solve_upper_transpose(L_inv_mass.data(), z.data(), p, n);
      }
    }
    // Precision/Kronecker blocks override their param ranges
    if (precision_block.active) precision_block.sample_momentum(p, rng);
    if (kronecker_block.active) kronecker_block.sample_momentum(p, rng);
    if (sparse_gmrf.active && sparse_gmrf.factorized) sparse_gmrf.sample_momentum(p, rng);
  }

  // Check if param i belongs to a precision, kronecker, or sparse GMRF block
  inline bool in_precision_block(int i) const {
    if (precision_block.active &&
        i >= precision_block.start &&
        i < precision_block.start + precision_block.size) return true;
    if (kronecker_block.active &&
        i >= kronecker_block.start &&
        i < kronecker_block.start + kronecker_block.S * kronecker_block.T) return true;
    if (sparse_gmrf.active && sparse_gmrf.factorized &&
        i >= sparse_gmrf.start &&
        i < sparse_gmrf.start + sparse_gmrf.S * sparse_gmrf.T) return true;
    return false;
  }

  // Kinetic energy: 0.5 * p^T * C * p  where C = M^{-1}
  // Uses Eigen BLAS for dense case (n>=16) for SIMD acceleration.
  double kinetic_energy(const double* p) const {
    // Precision/Kronecker/Sparse blocks: compute their contribution separately
    double ke_prec = 0.0;
    if (precision_block.active) ke_prec += precision_block.quadform(p);
    if (kronecker_block.active) ke_prec += kronecker_block.quadform(p);
    if (sparse_gmrf.active && sparse_gmrf.factorized) ke_prec += sparse_gmrf.quadform(p);

    if (type == MassMatrixType::BLOCK_DIAG && adapted) {
      double ke = 0.0;
      for (int i = 0; i < n; i++) {
        if (!in_block[i] && !in_precision_block(i)) {
          ke += inv_mass_diag[i] * p[i] * p[i];
        }
      }
      for (const auto& blk : blocks) {
        if (blk.adapted) {
          ke += blk.quadform(p);
        } else {
          for (int i = blk.start; i < blk.start + blk.size; i++) {
            if (!in_precision_block(i))
              ke += inv_mass_diag[i] * p[i] * p[i];
          }
        }
      }
      return 0.5 * (ke + ke_prec);
    } else if (type == MassMatrixType::DIAG || !adapted) {
      if (!precision_block.active && !kronecker_block.active) {
        return 0.5 * ratiod_linalg::weighted_norm_squared(p, inv_mass_diag.data(), n);
      }
      // Skip precision block params in diagonal sum
      double ke = 0.0;
      for (int i = 0; i < n; i++) {
        if (!in_precision_block(i))
          ke += inv_mass_diag[i] * p[i] * p[i];
      }
      return 0.5 * (ke + ke_prec);
    } else if (n >= 16) {
      // Dense: full matrix handles all params including precision block range
      // But if precision blocks are active, we need to exclude their range
      // from the dense contribution and add the precision block contribution instead.
      // For simplicity: if precision blocks active, fall back to per-element
      if (precision_block.active || kronecker_block.active) {
        double ke = 0.0;
        for (int i = 0; i < n; i++) {
          if (in_precision_block(i)) continue;
          for (int j = 0; j < n; j++) {
            if (in_precision_block(j)) continue;
            ke += p[i] * inv_mass_dense[static_cast<size_t>(j) * n + i] * p[j];
          }
        }
        return 0.5 * (ke + ke_prec);
      }
      Eigen::Map<const Eigen::MatrixXd> Am(inv_mass_dense.data(), n, n);
      Eigen::Map<const Eigen::VectorXd> pv(p, n);
      return 0.5 * pv.dot(Am.selfadjointView<Eigen::Lower>() * pv);
    } else {
      return 0.5 * ratiod_linalg::quadratic_form(p, inv_mass_dense.data(), n);
    }
  }

  // Compute C * p (for leapfrog position update: q += eps * C * p)
  // Result written to `result` buffer.
  // Uses Eigen BLAS for dense case (n>=16) for SIMD acceleration.
  void inv_mass_times_p(const double* p, double* result) const {
    if (type == MassMatrixType::BLOCK_DIAG && adapted) {
      for (int i = 0; i < n; i++) {
        result[i] = inv_mass_diag[i] * p[i];
      }
      for (const auto& blk : blocks) {
        if (blk.adapted) {
          double tmp[4];
          blk.matvec(p, tmp);
          for (int i = 0; i < blk.size; i++) {
            result[blk.start + i] = tmp[i];
          }
        }
      }
    } else if (type == MassMatrixType::DIAG || !adapted) {
      for (int i = 0; i < n; i++) {
        result[i] = inv_mass_diag[i] * p[i];
      }
    } else if (n >= 16) {
      Eigen::Map<const Eigen::MatrixXd> Am(inv_mass_dense.data(), n, n);
      Eigen::Map<const Eigen::VectorXd> pv(p, n);
      Eigen::Map<Eigen::VectorXd> rv(result, n);
      rv.noalias() = Am.selfadjointView<Eigen::Lower>() * pv;
    } else {
      ratiod_linalg::symmatvec(inv_mass_dense.data(), p, result, n);
    }
    // Precision/Kronecker blocks override their param ranges
    if (precision_block.active) {
      std::vector<double> tmp(precision_block.size);
      precision_block.matvec(p, tmp.data());
      for (int i = 0; i < precision_block.size; i++) {
        result[precision_block.start + i] = tmp[i];
      }
    }
    if (kronecker_block.active) {
      int ST = kronecker_block.S * kronecker_block.T;
      std::vector<double> tmp(ST);
      kronecker_block.matvec(p, tmp.data());
      for (int i = 0; i < ST; i++) {
        result[kronecker_block.start + i] = tmp[i];
      }
    }
    if (sparse_gmrf.active && sparse_gmrf.factorized) {
      int ST = sparse_gmrf.S * sparse_gmrf.T;
      std::vector<double> tmp(ST);
      sparse_gmrf.inv_mass_matvec(p, tmp.data());
      for (int i = 0; i < ST; i++) {
        result[sparse_gmrf.start + i] = tmp[i];
      }
    }
  }

  // Compute diag(C) * p — uses diagonal only, even when dense is available.
  // Kept for backwards compatibility / debugging. The NUTS U-turn criterion
  // now uses inv_mass_times_p() for correct geometry with dense mass.
  void inv_mass_diag_times_p(const double* p, double* result) const {
    for (int i = 0; i < n; i++) {
      result[i] = inv_mass_diag[i] * p[i];
    }
  }

  // Set metric directly from precomputed G^{-1} and its Cholesky L.
  // Used by SoftAbs per-trajectory metric retry. No shrinkage applied.
  void set_from_metric(const std::vector<double>& g_inv,
                       const std::vector<double>& l_g_inv) {
    inv_mass_dense = g_inv;
    L_inv_mass = l_g_inv;
    for (int i = 0; i < n; i++) {
      inv_mass_diag[i] = g_inv[static_cast<size_t>(i) * n + i];
      sqrt_mass_diag[i] = 1.0 / std::sqrt(std::max(inv_mass_diag[i], 1e-10));
    }
    adapted = true;
  }

  // Set diagonal mass from WelfordStats output (same interface as before)
  // When type==DENSE, also populate the dense matrices as diagonal so that
  // the dense code paths (sample_momentum, kinetic_energy, inv_mass_times_p)
  // produce correct results even before full covariance is available.
  // When type==BLOCK_DIAG, diagonal is set normally; blocks are adapted separately
  // via their own Welford accumulators.
  void set_diagonal(const std::vector<double>& inv_m, const std::vector<double>& sqrt_m) {
    inv_mass_diag = inv_m;
    sqrt_mass_diag = sqrt_m;
    adapted = true;

    if (type == MassMatrixType::DENSE && !inv_mass_dense.empty()) {
      // Populate dense matrices as diagonal so dense code paths work correctly
      std::fill(inv_mass_dense.begin(), inv_mass_dense.end(), 0.0);
      std::fill(L_inv_mass.begin(), L_inv_mass.end(), 0.0);
      for (int i = 0; i < n; i++) {
        inv_mass_dense[static_cast<size_t>(i) * n + i] = inv_m[i];
        // L where LL^T = inv_mass (diagonal): L[i,i] = sqrt(inv_mass[i])
        L_inv_mass[static_cast<size_t>(i) * n + i] = std::sqrt(inv_m[i]);
      }
    }
  }
};

// =====================================================================
// Online full covariance estimator (Welford's algorithm for outer products)
// =====================================================================

class WelfordCovStats {
public:
  int dim;
  int n;
  std::vector<double> mean;
  std::vector<double> M2;  // dim×dim: running sum of (x - mean_old)(x - mean_new)^T

  WelfordCovStats() : dim(0), n(0) {}

  explicit WelfordCovStats(int d) : dim(d), n(0),
    mean(d, 0.0), M2(static_cast<size_t>(d) * d, 0.0) {}

  void update(const std::vector<double>& x) {
    n++;
    std::vector<double> delta(dim);
    for (int i = 0; i < dim; i++) {
      delta[i] = x[i] - mean[i];
      mean[i] += delta[i] / n;
    }
    // Outer product update: M2 += (x - mean_new) * delta^T
    for (int i = 0; i < dim; i++) {
      double dx_new = x[i] - mean[i];
      for (int j = 0; j <= i; j++) {
        double val = dx_new * delta[j];
        M2[static_cast<size_t>(j) * dim + i] += val;
        if (i != j) {
          M2[static_cast<size_t>(i) * dim + j] += val;  // Symmetric
        }
      }
    }
  }

  // Get covariance with Oracle Approximating Shrinkage (OAS)
  // (Chen, Wiesel, Eldar, Hero 2010)
  //
  // Σ_shrunk = (1 - ρ) * S + ρ * (tr(S)/p) * I
  //
  // OAS automatically adapts shrinkage intensity:
  //   - When n < p (rank-deficient): ρ → 1, shrinks heavily toward scaled identity
  //   - When n >> p: ρ → 0, recovers the unregularized sample covariance
  // Guarantees positive definiteness when ρ > 0.
  //
  // shrinkage_intensity is set as a side-effect for logging.
  mutable double shrinkage_intensity = 0.0;

  std::vector<double> covariance() const {
    std::vector<double> cov(static_cast<size_t>(dim) * dim, 0.0);
    if (n < 2) {
      // Return identity
      for (int i = 0; i < dim; i++) {
        cov[static_cast<size_t>(i) * dim + i] = 1.0;
      }
      shrinkage_intensity = 1.0;
      return cov;
    }

    // Step 1: Compute sample covariance S = M2 / (n - 1)
    double inv_nm1 = 1.0 / (n - 1);
    for (int i = 0; i < dim; i++) {
      for (int j = 0; j < dim; j++) {
        cov[static_cast<size_t>(j) * dim + i] =
            M2[static_cast<size_t>(j) * dim + i] * inv_nm1;
      }
      // Ensure diagonal is positive
      double& diag = cov[static_cast<size_t>(i) * dim + i];
      diag = std::max(1e-6, diag);
    }

    // Step 2: Compute OAS shrinkage intensity
    // tr(S) and tr(S^2)
    double trS = 0.0;
    double trS2 = 0.0;
    for (int i = 0; i < dim; i++) {
      trS += cov[static_cast<size_t>(i) * dim + i];
    }
    // tr(S^2) = sum of all S_ij^2 (Frobenius norm squared)
    for (int i = 0; i < dim; i++) {
      for (int j = 0; j < dim; j++) {
        double s_ij = cov[static_cast<size_t>(j) * dim + i];
        trS2 += s_ij * s_ij;
      }
    }

    // OAS formula (Eq. 23 from Chen et al. 2010):
    // ρ = clamp( ((1 - 2/p) * tr(S²) + tr(S)²) /
    //            ((n + 1 - 2/p) * (tr(S²) - tr(S)²/p)), 0, 1 )
    double p = static_cast<double>(dim);
    double nf = static_cast<double>(n);
    double trS_sq = trS * trS;
    double numer = (1.0 - 2.0 / p) * trS2 + trS_sq;
    double denom = (nf + 1.0 - 2.0 / p) * (trS2 - trS_sq / p);

    double rho;
    if (std::abs(denom) < 1e-12) {
      // Denominator ~0 means S ≈ c*I already, minimal shrinkage needed
      rho = 0.0;
    } else {
      rho = std::max(0.0, std::min(1.0, numer / denom));
    }

    // When n < p, sample covariance is rank-deficient (rank n-1 < p).
    // OAS can underestimate the needed shrinkage for singular matrices.
    // Floor ρ at (1 - n/p) to fill the rank gap: this ensures the
    // p-(n-1) zero eigenvalues get lifted to ρ * tr(S)/p > 0.
    if (n < dim) {
      double rho_floor = 1.0 - nf / p;
      rho = std::max(rho, rho_floor);
    }

    // Even when n > p, moderate n/p ratios (2-5) can produce poorly
    // conditioned matrices. Apply a floor that decays as n/p grows.
    // At n/p=2: floor=0.08, n/p=5: floor=0.05, n/p=10+: floor=0.0
    if (nf < 10.0 * p) {
      double rho_cond_floor = 0.1 * (1.0 - nf / (10.0 * p));
      rho = std::max(rho, rho_cond_floor);
    }
    shrinkage_intensity = rho;

    // Step 3: Apply shrinkage: Σ = (1 - ρ) * S + ρ * (tr(S)/p) * I
    double target_diag = trS / p;  // Scaled identity target
    double one_minus_rho = 1.0 - rho;
    for (int i = 0; i < dim; i++) {
      for (int j = 0; j < dim; j++) {
        size_t idx = static_cast<size_t>(j) * dim + i;
        if (i == j) {
          cov[idx] = one_minus_rho * cov[idx] + rho * target_diag;
        } else {
          cov[idx] = one_minus_rho * cov[idx];
        }
      }
    }

    return cov;
  }

  void reset() {
    n = 0;
    std::fill(mean.begin(), mean.end(), 0.0);
    std::fill(M2.begin(), M2.end(), 0.0);
  }
};

struct DualAveraging {
  double mu, log_epsilon_bar, H_bar;
  double gamma, t0, kappa;
  double target_accept;  // Target acceptance rate (dimension-adaptive)
  int m;

  // Default constructor with dimension-adaptive target
  // target_boost: additional boost to target acceptance for challenging models
  //               (e.g., +0.10 for MSGP+temporal combinations)
  DualAveraging(double epsilon_init = 1.0, int n_params = 1, double target_boost = 0.0);
  double update(double alpha);
  double final_epsilon() const;

  // Compute dimension-adaptive target acceptance rate
  // Higher targets (closer to 1) = smaller step sizes = fewer divergences
  // but slower exploration. Stan default is 0.80.
  // target_boost: additional boost for challenging model combinations
  static double compute_target(int n_params, double target_boost = 0.0) {
    // Use higher targets (0.75-0.85) to avoid divergences in
    // challenging models (ICAR, hierarchical negbin, etc.)
    double base_target;
    if (n_params <= 5) base_target = 0.85;
    else if (n_params <= 20) base_target = 0.82;
    else if (n_params <= 50) base_target = 0.80;
    else if (n_params <= 100) base_target = 0.78;
    else base_target = 0.75;

    // Apply boost, cap at 0.99
    return std::min(0.99, base_target + target_boost);
  }
};

struct ChainState {
  std::vector<double> q;
  double log_prob;
  double epsilon;
  DualAveraging da;
  std::mt19937 rng;
  int n_divergent;
};

// Pure C++ result struct (safe for OpenMP parallel regions)
struct HMCResultCpp {
  std::vector<double> samples_flat;  // n_sample × n_params, row-major contiguous
  int n_params_stored = 0;
  std::vector<double> log_prob;
  std::vector<double> accept_prob;
  std::vector<int> n_leapfrog;
  std::vector<int> divergent;
  std::vector<int> treedepth;    // Actual tree depth per iteration (NUTS only)
  double epsilon;
  int n_warmup;
  int n_sample;
  int chain_id;
  int n_max_treedepth = 0;       // Count of iterations hitting max treedepth
  std::string sampler;           // Sampler name (e.g., "NUTS", "HMC", "NUTS->HMC(L=10)")

  // Collapsed mode draws (populated only when collapsed parameterization active)
  int n_gp_collapsed = 0;                         // N_gp if collapsed GP, 0 otherwise
  int n_icar_collapsed = 0;                        // S if collapsed ICAR/BYM2, 0 otherwise
  std::vector<double> gp_w_star_flat;              // w* draws (n_sample x N_gp, row-major)
  std::vector<double> icar_phi_star_flat;          // phi* draws (n_sample x S, row-major)
  std::vector<double> bym2_theta_star_flat;        // theta* draws (n_sample x S, row-major)

  // Row access for flat storage
  double* sample_row(int i) { return &samples_flat[i * n_params_stored]; }
  const double* sample_row(int i) const { return &samples_flat[i * n_params_stored]; }
};

// R-compatible result struct (create outside parallel regions)
struct HMCResult {
  Rcpp::NumericMatrix samples;
  Rcpp::NumericVector log_prob;
  Rcpp::NumericVector accept_prob;
  Rcpp::IntegerVector n_leapfrog;
  Rcpp::IntegerVector treedepth;
  Rcpp::IntegerVector divergent;
  double epsilon;
  int n_warmup;
  int n_sample;
  int chain_id;
  std::string sampler;

  // Collapsed mode draws (populated only when collapsed parameterization active)
  int n_gp_collapsed = 0;
  int n_icar_collapsed = 0;
  Rcpp::NumericMatrix gp_w_star;
  Rcpp::NumericMatrix icar_phi_star;
  Rcpp::NumericMatrix bym2_theta_star;
};

// Convert C++ result to R result (call outside parallel region)
inline HMCResult cpp_to_r_result(const HMCResultCpp& cpp_result, int n_params) {
  HMCResult r_result;
  r_result.samples = Rcpp::NumericMatrix(cpp_result.n_sample, n_params);
  r_result.log_prob = Rcpp::NumericVector(cpp_result.n_sample);
  r_result.accept_prob = Rcpp::NumericVector(cpp_result.n_sample);
  r_result.n_leapfrog = Rcpp::IntegerVector(cpp_result.n_sample);
  r_result.treedepth = Rcpp::IntegerVector(cpp_result.n_sample);
  r_result.divergent = Rcpp::IntegerVector(cpp_result.n_sample);
  r_result.epsilon = cpp_result.epsilon;
  r_result.n_warmup = cpp_result.n_warmup;
  r_result.n_sample = cpp_result.n_sample;
  r_result.chain_id = cpp_result.chain_id;
  r_result.sampler = cpp_result.sampler;

  for (int i = 0; i < cpp_result.n_sample; i++) {
    const double* row = cpp_result.sample_row(i);
    for (int j = 0; j < n_params; j++) {
      r_result.samples(i, j) = row[j];
    }
    r_result.log_prob[i] = cpp_result.log_prob[i];
    r_result.accept_prob[i] = cpp_result.accept_prob[i];
    r_result.n_leapfrog[i] = cpp_result.n_leapfrog[i];
    r_result.treedepth[i] = cpp_result.treedepth[i];
    r_result.divergent[i] = cpp_result.divergent[i];
  }

  // Collapsed GP: copy w* draws
  if (cpp_result.n_gp_collapsed > 0) {
    int n_gp = cpp_result.n_gp_collapsed;
    r_result.n_gp_collapsed = n_gp;
    r_result.gp_w_star = Rcpp::NumericMatrix(cpp_result.n_sample, n_gp);
    for (int i = 0; i < cpp_result.n_sample; i++) {
      for (int j = 0; j < n_gp; j++) {
        r_result.gp_w_star(i, j) = cpp_result.gp_w_star_flat[i * n_gp + j];
      }
    }
  }

  // Collapsed ICAR/BYM2: copy phi* and theta* draws
  if (cpp_result.n_icar_collapsed > 0) {
    int S = cpp_result.n_icar_collapsed;
    r_result.n_icar_collapsed = S;
    r_result.icar_phi_star = Rcpp::NumericMatrix(cpp_result.n_sample, S);
    for (int i = 0; i < cpp_result.n_sample; i++) {
      for (int j = 0; j < S; j++) {
        r_result.icar_phi_star(i, j) = cpp_result.icar_phi_star_flat[i * S + j];
      }
    }
    // BYM2: also copy theta*
    if (!cpp_result.bym2_theta_star_flat.empty()) {
      r_result.bym2_theta_star = Rcpp::NumericMatrix(cpp_result.n_sample, S);
      for (int i = 0; i < cpp_result.n_sample; i++) {
        for (int j = 0; j < S; j++) {
          r_result.bym2_theta_star(i, j) = cpp_result.bym2_theta_star_flat[i * S + j];
        }
      }
    }
  }

  return r_result;
}

// NUTS helper function declarations
double nuts_log_sum_exp(double a, double b);
double nuts_compute_hamiltonian(double log_prob, const std::vector<double>& p,
                                const std::vector<double>& inv_mass, int n);
bool nuts_check_uturn(const std::vector<double>& q_minus, const std::vector<double>& q_plus,
                      const std::vector<double>& p_minus, const std::vector<double>& p_plus,
                      const std::vector<double>& inv_mass, int n);
LeapfrogResultWithGrad leapfrog_step_with_grad(
    const std::vector<double>& q, const std::vector<double>& p,
    const std::vector<double>& grad,
    double epsilon, const std::vector<double>& inv_mass,
    bool use_mass, const ModelData& data, const ParamLayout& layout);
NUTSTreeResult build_tree(const NUTSNode& node, int direction, int depth,
                          double epsilon, const std::vector<double>& inv_mass,
                          bool use_mass, double H0, double delta_max,
                          const ModelData& data, const ParamLayout& layout,
                          std::mt19937& rng);

// Optimized NUTS: zero-allocation in-place leapfrog + buffer pool tree building
// Pointer-based Hamiltonian (no vector overhead)
double nuts_compute_hamiltonian_fast(double log_prob, const double* p,
                                     const DenseMassMatrix& mass, int n);
// Pointer-based U-turn check
bool nuts_check_uturn_fast(const double* q_minus, const double* q_plus,
                           const double* p_minus, const double* p_plus,
                           const DenseMassMatrix& mass, double* scratch, int n);
// In-place leapfrog step operating on workspace slot
LeapfrogInPlaceResult leapfrog_step_inplace(
    NUTSWorkspace& ws, int slot, double epsilon,
    const DenseMassMatrix& mass,
    const ModelData& data, const ParamLayout& layout);
// Zero-allocation recursive tree builder
TreeStats build_tree_fast(
    NUTSWorkspace& ws, int input_slot, int direction, int depth,
    double epsilon, const DenseMassMatrix& mass,
    double H0, double delta_max,
    const ModelData& data, const ParamLayout& layout,
    std::mt19937& rng);

// =====================================================================
// Sampler functions
// =====================================================================

// Single leapfrog step
LeapfrogResult leapfrog_step(
    const std::vector<double>& q,
    const std::vector<double>& p,
    double epsilon,
    const ModelData& data,
    const ParamLayout& layout
);

// Find reasonable initial step size
double find_reasonable_epsilon(
    const std::vector<double>& q,
    const ModelData& data,
    const ParamLayout& layout,
    std::mt19937& rng
);

// Mass-aware version: uses diagonal mass matrix for leapfrog and kinetic energy
double find_reasonable_epsilon(
    const std::vector<double>& q,
    const ModelData& data,
    const ParamLayout& layout,
    std::mt19937& rng,
    const std::vector<double>& inv_mass
);

// Dense-mass-aware version: uses full DenseMassMatrix for momentum, leapfrog, and kinetic energy
double find_reasonable_epsilon_dense(
    const std::vector<double>& q,
    const ModelData& data,
    const ParamLayout& layout,
    std::mt19937& rng,
    const DenseMassMatrix& mass
);

// Run single HMC chain (C++ version - safe for parallel)
// riemannian: 1 = retry divergent trajectories under a SoftAbs metric,
//              0 = off (default)
HMCResultCpp run_hmc_chain_cpp(
    const std::vector<double>& q_init,
    const ModelData& data,
    const ParamLayout& layout,
    int n_iter,
    int n_warmup,
    int L,
    int chain_id,
    unsigned int seed,
    bool verbose,
    int max_treedepth = 10,
    MassMatrixType metric_type = MassMatrixType::DIAG,
    double adapt_delta = -1.0,
    int riemannian = 0
);

// Run single HMC chain (R wrapper)
HMCResult run_hmc_chain(
    const std::vector<double>& q_init,
    const ModelData& data,
    const ParamLayout& layout,
    int n_iter,
    int n_warmup,
    int L,
    int chain_id,
    unsigned int seed,
    bool verbose,
    int max_treedepth = 10,
    MassMatrixType metric_type = MassMatrixType::DIAG,
    double adapt_delta = -1.0,
    int riemannian = 0
);

// Run multiple chains in parallel (across-chain parallelization)
std::vector<HMCResult> run_hmc_parallel_chains(
    const std::vector<double>& q_init,
    const ModelData& data,
    int n_iter,
    int n_warmup,
    int L,
    int n_chains,
    unsigned int seed,
    bool verbose,
    int max_treedepth = 10,
    MassMatrixType metric_type = MassMatrixType::DIAG,
    double adapt_delta = -1.0,
    int riemannian = 0
);

// =====================================================================
// SoftAbs per-trajectory metric (Riemannian-like divergence retry)
// =====================================================================

// Compute full Hessian via finite differences of the H-mode gradient.
// H[i,j] = (grad_j(q + h*e_i) - grad_j(q)) / h
// Cost: (p+1) gradient evaluations.
void compute_hessian_finite_diff(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    std::vector<double>& hessian,
    double h = 1e-5
);

// Compute SoftAbs metric from negative Hessian.
// G = Q diag(f(λ_i)) Q^T where f(λ) = λ * coth(α * λ)
// Returns G^{-1} and its Cholesky L. Returns false on failure.
bool compute_softabs_metric(
    const std::vector<double>& neg_hessian,
    int p,
    double alpha,
    std::vector<double>& G_inv,
    std::vector<double>& L_G_inv
);

// =====================================================================
// The standalone temporal field's effects
// =====================================================================

// Whether the spatial block's mean is removed on its way into eta.
//
// An intrinsic ICAR/BYM2 precision is rank-deficient on the constant
// direction: the prior augments it and the likelihood never sees it
// (spatial_field_constraint.h). Proper CAR is full rank at rho < 1, so its
// mean is identified by its own prior, and centring would impose a constraint
// the model does not carry. A collapsed field has no sampled phi to centre.
//
// Every density and every gradient that reads the spatial block asks here, so
// a gradient and the log posterior it is evaluated against cannot disagree
// about which field the likelihood saw.
inline bool spatial_block_is_centred(const ModelData& data,
                                     const ParamLayout& layout) {
  if (!layout.has_spatial || layout.spatial_start < 0) return false;
  if (data.icar_collapsed || data.bym2_collapsed) return false;
  if (layout.is_car_proper) return data.car_center;
  return true;
}

// Whether the spatial PRIOR reads a field the likelihood no longer sees whole.
// An intrinsic Q annihilates the constant, so its quadratic form is the same at
// the raw field and the centred one and either pointer does. Q(rho) does not,
// so a centred proper CAR field has to hand its prior the raw values or the
// constant direction would carry no prior at all.
inline bool spatial_prior_reads_raw(const ModelData& data,
                                    const ParamLayout& layout) {
  return layout.is_car_proper && spatial_block_is_centred(data, layout);
}

// Whether the AR1 temporal field is sampled in its non-centred coordinate:
// the parameter block holds z ~ N(0, I) rather than the effects themselves.
inline bool temporal_ar1_nc(const ModelData& data, const ParamLayout& layout) {
  return layout.has_temporal &&
         data.temporal_type == TemporalType::AR1 &&
         data.temporal_parameterization == 1;
}

// Whether an intrinsic standalone walk is sampled in its non-centred
// coordinate: the block holds z ~ N(0, I) and the effects are
// sigma * A^{-1} z (hmc_temporal_nc.h).
//
// The prior augments ONE global constant whatever the number of groups, so
// with several groups the transform carries the G - 1 improper group contrasts
// beside the scaled directions rather than augmenting each group's own
// constant.
inline bool temporal_rw_nc(const ModelData& data, const ParamLayout& layout) {
  return layout.has_temporal && !layout.is_temporal_gp &&
         (data.temporal_type == TemporalType::RW1 ||
          data.temporal_type == TemporalType::RW2) &&
         data.temporal_parameterization == 1 &&
         ratiod_temporal_nc::nc_applies(
             data.n_times,
             data.temporal_type == TemporalType::RW2 ? 2 : 1,
             data.temporal_cyclic);
}

// Whether the standalone temporal field is an intrinsic walk. RW1 and RW2 have
// a constant null direction that is unidentified against the intercept; the
// prior augments it (Q -> Q + 11'/n) and the effects are centred on their way
// into eta, the two halves of tulpa/sum_to_zero.h's construction. AR1, IID and
// the temporal GP are proper and identify their own level.
//
// Exactly ONE direction is treated: the single GLOBAL constant. With several
// groups the walks are disconnected, so the prior's null space is G constants,
// but a per-group level shifts eta only for that group's observations and the
// likelihood identifies it. Augmenting per group would put a proper
// N(0, 1/tau) prior on those G - 1 contrasts and shrink them, which is a
// different model.
inline bool temporal_is_intrinsic(const ModelData& data,
                                  const ParamLayout& layout) {
  return layout.has_temporal && !layout.is_temporal_gp &&
         (data.temporal_type == TemporalType::RW1 ||
          data.temporal_type == TemporalType::RW2);
}

// Whether the temporal effects differ from the block the sampler moves in, so
// a stored draw has to carry the transform rather than the coordinate: a
// non-centred AR1 samples innovations, and an intrinsic walk's level is
// removed on the way into eta. The reported field is then the one the
// likelihood saw, which is what the R side adds back into eta.
inline bool temporal_effects_transformed(const ModelData& data,
                                         const ParamLayout& layout) {
  return temporal_ar1_nc(data, layout) || temporal_is_intrinsic(data, layout);
}

// The temporal effects at `params`: the values that enter the linear
// predictor. Read in place when the sampled block already holds them,
// reconstructed into `buf` when a non-centred AR1 holds innovations instead,
// and centred into `buf` when an intrinsic walk's constant has to be removed.
// Every density and every gradient reads the effects, so the transform lives
// here and nowhere else.
inline const double* temporal_effects(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    std::vector<double>& buf
) {
  const double* z = &params[layout.temporal_start];

  if (temporal_rw_nc(data, layout)) {
    // The block holds z; the effects are sigma * A^{-1} z, then centred on
    // their way into eta as the centred coordinate's are.
    const int n = layout.temporal_end - layout.temporal_start;
    const double sigma =
        1.0 / std::sqrt(std::exp(params[layout.log_tau_temporal_idx]));
    buf.assign(n, 0.0);
    ratiod_temporal_nc::rw_nc_grouped_forward(
        z, data.n_times, data.n_temporal_groups,
        data.temporal_type == TemporalType::RW2 ? 2 : 1,
        data.temporal_cyclic, sigma, buf.data());
    (void)tulpa::s2z_centre_component(buf.data(), 0, n);
    return buf.data();
  }

  if (temporal_is_intrinsic(data, layout)) {
    const int n = layout.temporal_end - layout.temporal_start;
    buf.assign(z, z + n);
    (void)tulpa::s2z_centre_component(buf.data(), 0, n);
    return buf.data();
  }

  if (!temporal_ar1_nc(data, layout)) return z;

  const int T = data.n_times;
  const int G = data.n_temporal_groups;
  buf.assign(static_cast<size_t>(T) * G, 0.0);

  const double tau = std::exp(params[layout.log_tau_temporal_idx]);
  const double rho = ratiod_ar1::rho_from_logit(params[layout.logit_rho_ar1_idx]);
  for (int g = 0; g < G; g++) {
    ratiod_temporal::ar1_nc_forward(z + g * T, buf.data() + g * T, T, rho, tau);
  }
  return buf.data();
}

// The temporal block as a density or a gradient needs it: `phi`, the effects
// that enter the linear predictor, and `z`, the coordinate the sampler moves
// in. The two are the same pointer unless the field is non-centred or centred,
// and the buffer backing either lives as long as the view does. `raw_sum`
// carries the freed constant direction of an intrinsic walk, which the
// augmented quadratic form needs and centring has removed from `phi`.
struct TemporalView {
  std::vector<double> buf;
  const double* phi = nullptr;
  const double* z = nullptr;
  double raw_sum = 0.0;
  bool centred = false;
  bool nc = false;

  void read(const std::vector<double>& params, const ModelData& data,
            const ParamLayout& layout) {
    z = &params[layout.temporal_start];
    centred = temporal_is_intrinsic(data, layout);
    nc = temporal_rw_nc(data, layout);
    // In the non-centred coordinate the block is z, not the field, and the
    // augmented quadratic that raw_sum feeds has already cancelled against the
    // transform's Jacobian, so there is nothing for it to carry.
    if (centred && !nc) {
      raw_sum = tulpa::s2z_component_sum(
          z, 0, layout.temporal_end - layout.temporal_start);
    }
    phi = temporal_effects(params, data, layout, buf);
  }
};

// =====================================================================
// The multi-scale GP field's effects
// =====================================================================

// Whether the multi-scale GP field is sampled in its non-centred coordinate:
// each scale's parameter block holds z ~ N(0, I) rather than the effects. The
// HSGP arm's basis coefficients are already such a coordinate, so the flag
// governs the two-neighbour-set arm alone.
inline bool msgp_nc(const ModelData& data, const ParamLayout& layout) {
  return layout.is_multiscale_gp && data.has_multiscale_gp &&
         !data.msgp_is_hsgp && data.msgp_parameterization == 1;
}

// The multi-scale field as a gradient path needs it: the effects each scale
// contributes to eta, and the adjoint that carries the likelihood back onto
// whichever coordinate is being sampled.
//
// The gradient slots (3, 4) are not the density's (1, 2): a fused gradient
// evaluates the density once its own backward pass has read the cached
// factors, and the two would otherwise share a workspace.
struct MultiscaleGPView {
  ratiod_gp::MultiscaleGPFieldT<double, 3> field;
  const double* z_local = nullptr;
  const double* z_regional = nullptr;
  double sigma2_local = 0.0, phi_local = 0.0;
  double sigma2_regional = 0.0, phi_regional = 0.0;

  const double* local() const { return field.local; }
  const double* regional() const { return field.regional; }

  void read(const std::vector<double>& params, const ModelData& data,
            const ParamLayout& layout) {
    z_local = &params[layout.gp_local_start];
    z_regional = &params[layout.gp_regional_start];
    sigma2_local = std::exp(params[layout.log_sigma2_gp_local_idx]);
    phi_local = std::exp(params[layout.log_phi_gp_local_idx]);
    sigma2_regional = std::exp(params[layout.log_sigma2_gp_regional_idx]);
    phi_regional = std::exp(params[layout.log_phi_gp_regional_idx]);
    field.read(z_local, z_regional, sigma2_local, phi_local,
               sigma2_regional, phi_regional, data.multiscale_gp_data,
               msgp_nc(data, layout));
  }

  // The field's own prior and the likelihood's derivative with respect to the
  // effects, onto the sampled coordinate.
  //
  // Both scales enter eta through the same sum w_local[loc] + w_regional[loc],
  // so the likelihood's derivative with respect to either is the one vector
  // `dL_dw`, indexed by location.
  //
  // Centred, that vector lands on each block and the NNGP density supplies the
  // prior. Non-centred, each scale runs the same adjoint the single-scale
  // field runs: it returns the FULL z gradient -- it seeds each entry with
  // -z before adding the likelihood's -- and the two contributions the
  // transform carries to that scale's variance and range.
  void accumulate(const double* dL_dw, const ModelData& data,
                  const ParamLayout& layout, double* grad) const {
    const int n = data.multiscale_gp_data.n_obs;
    if (field.noncentered) {
      std::vector<double> grad_z(n, 0.0);
      double grad_log_sigma2_lik = 0.0, grad_log_phi_lik = 0.0, grad_log_phi_jac = 0.0;

      ratiod_gp::nngp_nc_backward(z_local, sigma2_local, phi_local,
                                  field.gp_local, field.ws_local(),
                                  dL_dw, grad_z.data(),
                                  grad_log_sigma2_lik, grad_log_phi_lik,
                                  grad_log_phi_jac);
      for (int i = 0; i < n; i++) grad[layout.gp_local_start + i] += grad_z[i];
      grad[layout.log_sigma2_gp_local_idx] += grad_log_sigma2_lik;
      grad[layout.log_phi_gp_local_idx] += grad_log_phi_lik;

      std::fill(grad_z.begin(), grad_z.end(), 0.0);
      ratiod_gp::nngp_nc_backward(z_regional, sigma2_regional, phi_regional,
                                  field.gp_regional, field.ws_regional(),
                                  dL_dw, grad_z.data(),
                                  grad_log_sigma2_lik, grad_log_phi_lik,
                                  grad_log_phi_jac);
      for (int i = 0; i < n; i++) grad[layout.gp_regional_start + i] += grad_z[i];
      grad[layout.log_sigma2_gp_regional_idx] += grad_log_sigma2_lik;
      grad[layout.log_phi_gp_regional_idx] += grad_log_phi_lik;
      return;
    }

    std::vector<double> w_local(field.local, field.local + n);
    std::vector<double> w_regional(field.regional, field.regional + n);
    ratiod_gp::NNGPGradients prior_local, prior_regional;
    ratiod_gp::gp_nngp_gradients(w_local, sigma2_local, phi_local,
                                 field.gp_local, prior_local);
    ratiod_gp::gp_nngp_gradients(w_regional, sigma2_regional, phi_regional,
                                 field.gp_regional, prior_regional);
    for (int i = 0; i < n; i++) {
      grad[layout.gp_local_start + i] += prior_local.grad_w[i] + dL_dw[i];
      grad[layout.gp_regional_start + i] += prior_regional.grad_w[i] + dL_dw[i];
    }
    grad[layout.log_sigma2_gp_local_idx] += prior_local.grad_log_sigma2;
    grad[layout.log_phi_gp_local_idx] += prior_local.grad_log_phi;
    grad[layout.log_sigma2_gp_regional_idx] += prior_regional.grad_log_sigma2;
    grad[layout.log_phi_gp_regional_idx] += prior_regional.grad_log_phi;
  }
};

} // namespace ratiod_hmc

#endif // QUOTR_HMC_SAMPLER_H
