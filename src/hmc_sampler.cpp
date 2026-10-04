// hmc_sampler.cpp
// Full HMC/NUTS backend with spatial, temporal, and ZI support
// Provides Stan-free Bayesian inference for all ratiod models

#include "hmc_sampler.h"
#include <tulpa/walnuts.h>
#include "omp_chain_team.h"
#include "tls_workspace.h"
#include "linalg_fast.h"
#include <RcppEigen.h>
#include "hmc_progress.h"
#include "autodiff.h"
#include "autodiff_utils.h"
#include "ar1_shared.h"
#include "hmc_gp_autodiff.h"
#include "hmc_gp_collapsed.h"
#include <tulpa/soft_sum_to_zero.h>
#include "spatial_field_constraint.h"
#include "hmc_icar_collapsed.h"
#include "hmc_car_proper.h"
#include "hmc_temporal_autodiff.h"
#include "hmc_tvc_grad.h"
#include "hmc_multiscale_temporal_grad.h"
#include "st_prior_grad.h"
#include "cov_type_code.h"
#include "hmc_model_data_blocks.h"
#include "hmc_re_blocks_grad.h"
#include <Rcpp.h>

// Include log_post_impl.h AFTER hmc_sampler.h so types are defined
#include "log_post_impl.h"
#include <cmath>
#include <algorithm>
#include <limits>
#include <atomic>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace Rcpp;

namespace ratiod_hmc {

// =====================================================================
// Dense mass matrix: Cholesky decomposition via Eigen
// =====================================================================

bool DenseMassMatrix::update_from_covariance(const double* cov, int n_samples) {
  // Map the covariance data into an Eigen matrix (column-major)
  Eigen::Map<const Eigen::MatrixXd> C(cov, n, n);

  // Eigendecomposition for condition number control.
  // Without conditioning, ill-conditioned mass matrices force epsilon to be
  // tiny (driven by the stiffest direction), making sampling extremely slow.
  // E.g., HSGP+RW1 gets epsilon=3.2e-5 unconditioned vs ~0.01 conditioned.
  //
  // Clip eigenvalue ratio to MAX_COND so the step size ratio between the
  // loosest and stiffest directions is at most sqrt(MAX_COND) ≈ 100:1.
  constexpr double MAX_COND = 1e4;

  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(C);
  if (eig.info() != Eigen::Success) {
    // Eigendecomposition failed — degrade to diagonal
    type = MassMatrixType::DIAG;
    adapted = true;
    for (int i = 0; i < n; i++) {
      double var_i = cov[static_cast<size_t>(i) * n + i];
      inv_mass_diag[i] = std::max(1e-3, std::min(var_i, 1e3));
      sqrt_mass_diag[i] = 1.0 / std::sqrt(inv_mass_diag[i]);
    }
    return false;
  }

  Eigen::VectorXd evals = eig.eigenvalues();
  double lambda_max = evals.maxCoeff();
  double lambda_floor = std::max(lambda_max / MAX_COND, 1e-8);
  bool clipped = false;
  for (int i = 0; i < n; i++) {
    if (evals[i] < lambda_floor) {
      evals[i] = lambda_floor;
      clipped = true;
    }
  }

  // Reconstruct conditioned covariance: V * diag(λ_clipped) * V^T
  Eigen::MatrixXd C_cond;
  if (clipped) {
    const Eigen::MatrixXd& V = eig.eigenvectors();
    C_cond = V * evals.asDiagonal() * V.transpose();
  } else {
    C_cond = C;
  }

  // Cholesky of (conditioned) covariance — guaranteed to succeed after clipping
  Eigen::LLT<Eigen::MatrixXd> llt(C_cond);
  if (llt.info() != Eigen::Success) {
    // Should not happen after eigenvalue clipping, but handle gracefully
    type = MassMatrixType::DIAG;
    adapted = true;
    for (int i = 0; i < n; i++) {
      double var_i = cov[static_cast<size_t>(i) * n + i];
      inv_mass_diag[i] = std::max(1e-3, std::min(var_i, 1e3));
      sqrt_mass_diag[i] = 1.0 / std::sqrt(inv_mass_diag[i]);
    }
    return false;
  }

  // Store conditioned covariance as inv_mass_dense
  std::memcpy(inv_mass_dense.data(), C_cond.data(),
              static_cast<size_t>(n) * n * sizeof(double));

  // Store Cholesky factor L
  Eigen::MatrixXd L_mat = llt.matrixL();
  std::memcpy(L_inv_mass.data(), L_mat.data(),
              static_cast<size_t>(n) * n * sizeof(double));

  // Also update diagonal for fallback and find_reasonable_epsilon compatibility
  for (int i = 0; i < n; i++) {
    double var_i = C_cond(i, i);
    inv_mass_diag[i] = std::max(1e-3, std::min(var_i, 1e3));
    sqrt_mass_diag[i] = 1.0 / std::sqrt(inv_mass_diag[i]);
  }

  adapted = true;
  return true;
}

// =====================================================================
// Parameter layout computation
// =====================================================================

ParamLayout compute_param_layout(const ModelData& data) {
  ParamLayout layout;
  int idx = 0;

  // Fixed effects numerator
  layout.beta_num_start = idx;
  idx += data.p_num;
  layout.beta_num_end = idx;

  // Fixed effects denominator
  layout.beta_denom_start = idx;
  idx += data.p_denom;
  layout.beta_denom_end = idx;

  // Random effects (supports multiple crossed RE terms with slopes)
  layout.has_re = (data.n_re_groups > 0 || data.total_re_groups > 0);
  layout.has_re_slopes = data.has_re_slopes;
  layout.has_re_correlated_slopes = data.has_re_correlated_slopes;

  if (data.has_re_slopes && data.n_re_terms > 0) {
    // Random slopes case: need sigma per coefficient type + Cholesky params + RE effects
    int n_terms = data.n_re_terms;

    layout.log_sigma_re_multi.resize(n_terms);
    layout.log_sigma_re_slopes.resize(n_terms);
    layout.re_start_multi.resize(n_terms);
    layout.re_end_multi.resize(n_terms);
    layout.re_n_coefs_multi.resize(n_terms);
    layout.re_correlated_multi.resize(n_terms);
    layout.chol_re_start_multi.resize(n_terms);
    layout.chol_re_end_multi.resize(n_terms);

    // First pass: allocate sigma parameters for each term
    for (int t = 0; t < n_terms; t++) {
      int n_coefs = data.re_n_coefs[t];
      layout.re_n_coefs_multi[t] = n_coefs;
      layout.re_correlated_multi[t] = data.re_correlated[t];

      // Allocate log_sigma for each coefficient type (intercept, slopes)
      layout.log_sigma_re_slopes[t].resize(n_coefs);
      for (int c = 0; c < n_coefs; c++) {
        layout.log_sigma_re_slopes[t][c] = idx++;
      }
      // Legacy: point to first sigma for backwards compat
      layout.log_sigma_re_multi[t] = layout.log_sigma_re_slopes[t][0];
    }

    // Second pass: allocate Cholesky parameters for correlated terms
    for (int t = 0; t < n_terms; t++) {
      int n_chol = data.re_n_chol[t];  // k*(k-1)/2 for correlated, 0 otherwise
      if (n_chol > 0) {
        layout.chol_re_start_multi[t] = idx;
        idx += n_chol;
        layout.chol_re_end_multi[t] = idx;
      } else {
        layout.chol_re_start_multi[t] = -1;
        layout.chol_re_end_multi[t] = -1;
      }
    }

    // Third pass: allocate RE effects for each term
    for (int t = 0; t < n_terms; t++) {
      int n_groups = data.re_n_groups_multi[t];
      int n_coefs = data.re_n_coefs[t];

      layout.re_start_multi[t] = idx;
      idx += n_groups * n_coefs;  // Each group has n_coefs parameters
      layout.re_end_multi[t] = idx;
    }

    // Legacy fields: point to first term
    layout.log_sigma_re_idx = layout.log_sigma_re_multi[0];
    layout.re_start = layout.re_start_multi[0];
    layout.re_end = layout.re_end_multi[0];

  } else if (data.n_re_terms > 1) {
    // Multiple RE terms (intercept only): allocate sigma and RE for each term
    layout.log_sigma_re_multi.resize(data.n_re_terms);
    layout.log_sigma_re_slopes.resize(data.n_re_terms);
    layout.re_start_multi.resize(data.n_re_terms);
    layout.re_end_multi.resize(data.n_re_terms);
    layout.re_n_coefs_multi.resize(data.n_re_terms, 1);  // All intercept-only
    layout.re_correlated_multi.resize(data.n_re_terms, false);
    layout.chol_re_start_multi.resize(data.n_re_terms, -1);
    layout.chol_re_end_multi.resize(data.n_re_terms, -1);

    for (int t = 0; t < data.n_re_terms; t++) {
      layout.log_sigma_re_multi[t] = idx++;
      // The per-coefficient sigma index table the slope layout fills, at the
      // one coefficient an intercept-only term has, so a loop over
      // (term, coefficient) reads either layout.
      layout.log_sigma_re_slopes[t].assign(1, layout.log_sigma_re_multi[t]);
    }
    for (int t = 0; t < data.n_re_terms; t++) {
      layout.re_start_multi[t] = idx;
      idx += data.re_n_groups_multi[t];
      layout.re_end_multi[t] = idx;
    }

    // Set legacy fields to first term for backwards compatibility
    layout.log_sigma_re_idx = layout.log_sigma_re_multi[0];
    layout.re_start = layout.re_start_multi[0];
    layout.re_end = layout.re_end_multi[0];
  } else if (layout.has_re) {
    // Single RE term (intercept only)
    layout.log_sigma_re_idx = idx++;
    layout.re_start = idx;
    idx += data.n_re_groups;
    layout.re_end = idx;

    // Also set multi arrays for consistency
    layout.log_sigma_re_multi.resize(1);
    layout.log_sigma_re_multi[0] = layout.log_sigma_re_idx;
    layout.re_start_multi.resize(1);
    layout.re_start_multi[0] = layout.re_start;
    layout.re_end_multi.resize(1);
    layout.re_end_multi[0] = layout.re_end;
    layout.re_n_coefs_multi.resize(1, 1);
    layout.re_correlated_multi.resize(1, false);
    layout.chol_re_start_multi.resize(1, -1);
    layout.chol_re_end_multi.resize(1, -1);
  } else {
    layout.log_sigma_re_idx = -1;
    layout.re_start = layout.re_end = -1;
  }

  // Overdispersion / shape / sigma parameters
  // NEGBIN_NEGBIN: phi_num (overdispersion for num), phi_denom (overdispersion for denom)
  // POISSON_GAMMA: phi_denom (shape for gamma denom)
  // GAMMA_GAMMA: phi_num (shape for num), phi_denom (shape for denom)
  // LOGNORMAL: phi_num (sigma for num), phi_denom (sigma for denom)
  // BETA_BINOMIAL: phi_num (precision parameter)
  layout.has_phi_num = (data.model_type == ModelType::NEGBIN_NEGBIN ||
                        data.model_type == ModelType::NEGBIN_GAMMA ||
                        data.model_type == ModelType::GAMMA_GAMMA ||
                        data.model_type == ModelType::LOGNORMAL ||
                        data.model_type == ModelType::BETA_BINOMIAL);
  layout.has_phi_denom = (data.model_type == ModelType::NEGBIN_NEGBIN ||
                          data.model_type == ModelType::POISSON_GAMMA ||
                          data.model_type == ModelType::NEGBIN_GAMMA ||
                          data.model_type == ModelType::GAMMA_GAMMA ||
                          data.model_type == ModelType::LOGNORMAL);

  if (layout.has_phi_num) {
    layout.log_phi_num_idx = idx++;
  } else {
    layout.log_phi_num_idx = -1;
  }
  if (layout.has_phi_denom) {
    layout.log_phi_denom_idx = idx++;
  } else {
    layout.log_phi_denom_idx = -1;
  }

  // Spatial effects (ICAR/BYM2/proper CAR only - GP handled separately below)
  layout.has_spatial = (data.spatial_type == SpatialType::ICAR ||
                        data.spatial_type == SpatialType::BYM2 ||
                        data.spatial_type == SpatialType::CAR_PROPER);
  layout.is_bym2 = (data.spatial_type == SpatialType::BYM2);
  layout.is_car_proper = (data.spatial_type == SpatialType::CAR_PROPER);
  layout.is_icar_collapsed = (data.spatial_type == SpatialType::ICAR && data.icar_collapsed);
  layout.is_bym2_collapsed = (data.spatial_type == SpatialType::BYM2 && data.bym2_collapsed);

  if (layout.has_spatial) {
    if (layout.is_bym2) {
      // BYM2 Riebler: log_sigma_total, logit_rho, [phi_scaled, theta if not collapsed]
      layout.log_sigma_bym2_idx = idx++;
      layout.logit_rho_bym2_idx = idx++;
      if (data.bym2_collapsed) {
        // Collapsed: phi and theta marginalized out, not in param vector
        layout.spatial_start = layout.spatial_end = -1;
        layout.theta_bym2_start = layout.theta_bym2_end = -1;
      } else {
        layout.spatial_start = idx;
        idx += data.n_spatial_units;  // phi_scaled (structured)
        layout.spatial_end = idx;
        layout.theta_bym2_start = idx;
        idx += data.n_spatial_units;  // theta (unstructured)
        layout.theta_bym2_end = idx;
      }
      layout.log_tau_spatial_idx = -1;
    } else {
      // ICAR / proper CAR: log_tau, [logit_rho if proper CAR], [phi if not collapsed]
      layout.log_tau_spatial_idx = idx++;
      layout.logit_rho_car_idx = layout.is_car_proper ? idx++ : -1;
      if (data.icar_collapsed) {
        // Collapsed: phi marginalized out, not in param vector (ICAR only;
        // proper CAR never collapses -- blocked at the R level)
        layout.spatial_start = layout.spatial_end = -1;
      } else {
        layout.spatial_start = idx;
        idx += data.n_spatial_units;
        layout.spatial_end = idx;
      }
      layout.log_sigma_bym2_idx = -1;
      layout.logit_rho_bym2_idx = -1;
      layout.theta_bym2_start = layout.theta_bym2_end = -1;
    }
  } else {
    layout.log_tau_spatial_idx = -1;
    layout.logit_rho_car_idx = -1;
    layout.spatial_start = layout.spatial_end = -1;
    layout.log_sigma_bym2_idx = -1;
    layout.logit_rho_bym2_idx = -1;
    layout.theta_bym2_start = layout.theta_bym2_end = -1;
  }

  // Temporal effects
  layout.has_temporal = (data.temporal_type != TemporalType::NONE);
  layout.is_ar1 = (data.temporal_type == TemporalType::AR1);
  layout.is_temporal_gp = (data.temporal_type == TemporalType::GP);

  if (layout.has_temporal) {
    if (layout.is_temporal_gp) {
      // Temporal GP: log_sigma2 + log_phi + effects
      layout.log_sigma2_temporal_gp_idx = idx++;
      layout.logit_phi_temporal_gp_idx = idx++;
      layout.log_tau_temporal_idx = -1;  // Not used for GP
      layout.logit_rho_ar1_idx = -1;
    } else {
      // RW1/RW2/AR1: log_tau + effects (+ rho for AR1)
      layout.log_tau_temporal_idx = idx++;
      layout.log_sigma2_temporal_gp_idx = -1;
      layout.logit_phi_temporal_gp_idx = -1;

      // AR1 also has rho parameter
      if (layout.is_ar1) {
        layout.logit_rho_ar1_idx = idx++;
      } else {
        layout.logit_rho_ar1_idx = -1;
      }
    }

    // Temporal effects: n_times * n_groups parameters
    layout.temporal_start = idx;
    idx += data.n_temporal_params;
    layout.temporal_end = idx;
  } else {
    layout.log_tau_temporal_idx = -1;
    layout.logit_rho_ar1_idx = -1;
    layout.log_sigma2_temporal_gp_idx = -1;
    layout.logit_phi_temporal_gp_idx = -1;
    layout.temporal_start = layout.temporal_end = -1;
  }

  // Zero-inflation parameters
  layout.has_zi = (data.zi_type != ZIType::NONE);

  if (layout.has_zi) {
    layout.beta_zi_start = idx;
    idx += data.p_zi;
    layout.beta_zi_end = idx;
  } else {
    layout.beta_zi_start = layout.beta_zi_end = -1;
  }

  // One-inflation parameters (for OI-binomial and ZOIB)
  layout.has_oi = (data.zi_type == ZIType::OI_BINOMIAL || data.zi_type == ZIType::ZOIB);

  if (layout.has_oi && data.p_oi > 0) {
    layout.beta_oi_start = idx;
    idx += data.p_oi;
    layout.beta_oi_end = idx;
  } else {
    layout.beta_oi_start = layout.beta_oi_end = -1;
  }

  // GP spatial parameters
  layout.is_gp = (data.spatial_type == SpatialType::GP);
  layout.is_multiscale_gp = (data.spatial_type == SpatialType::MULTISCALE_GP);

  layout.is_gp_collapsed = layout.is_gp && data.has_gp && data.gp_collapsed;

  if (layout.is_gp && data.has_gp) {
    layout.log_sigma2_gp_idx = idx++;
    layout.log_phi_gp_idx = idx++;
    if (!data.gp_collapsed) {
      // Standard: allocate slots for GP effects
      layout.gp_w_start = idx;
      idx += data.gp_data.n_obs;
      layout.gp_w_end = idx;
    } else {
      // Collapsed: GP effects marginalized out, not in param vector
      layout.gp_w_start = layout.gp_w_end = -1;
    }
  } else {
    layout.log_sigma2_gp_idx = -1;
    layout.log_phi_gp_idx = -1;
    layout.gp_w_start = layout.gp_w_end = -1;
  }

  // Multi-scale GP parameters
  if (layout.is_multiscale_gp && data.has_multiscale_gp) {
    // Number of spatial effects per scale: m^2 for HSGP, n_obs for NNGP
    int n_per_scale = data.msgp_is_hsgp ? data.msgp_hsgp_data.m_total
                                        : data.multiscale_gp_data.n_obs;
    // Local scale
    layout.log_sigma2_gp_local_idx = idx++;
    layout.log_phi_gp_local_idx = idx++;  // log_lengthscale for HSGP
    layout.gp_local_start = idx;
    idx += n_per_scale;
    layout.gp_local_end = idx;

    // Regional scale
    layout.log_sigma2_gp_regional_idx = idx++;
    layout.log_phi_gp_regional_idx = idx++;
    layout.gp_regional_start = idx;
    idx += n_per_scale;
    layout.gp_regional_end = idx;
  } else {
    layout.log_sigma2_gp_local_idx = -1;
    layout.log_phi_gp_local_idx = -1;
    layout.gp_local_start = layout.gp_local_end = -1;
    layout.log_sigma2_gp_regional_idx = -1;
    layout.log_phi_gp_regional_idx = -1;
    layout.gp_regional_start = layout.gp_regional_end = -1;
  }

  // Multi-scale temporal parameters
  layout.has_multiscale_temporal = data.has_multiscale_temporal;

  if (layout.has_multiscale_temporal) {
    // Trend component
    if (data.multiscale_temporal_data.trend_type != ratiod_temporal::TemporalType::NONE) {
      layout.log_sigma2_trend_idx = idx++;
      layout.trend_start = idx;
      idx += data.multiscale_temporal_data.n_times;
      layout.trend_end = idx;
    } else {
      layout.log_sigma2_trend_idx = -1;
      layout.trend_start = layout.trend_end = -1;
    }

    // Seasonal component
    if (data.multiscale_temporal_data.seasonal_period > 0) {
      layout.log_sigma2_seasonal_idx = idx++;
      layout.seasonal_start = idx;
      idx += data.multiscale_temporal_data.seasonal_period;
      layout.seasonal_end = idx;
    } else {
      layout.log_sigma2_seasonal_idx = -1;
      layout.seasonal_start = layout.seasonal_end = -1;
    }

    // Short-term component
    if (data.multiscale_temporal_data.short_term_type != ratiod_temporal::TemporalType::NONE) {
      layout.log_sigma2_short_idx = idx++;
      if (data.multiscale_temporal_data.short_term_type == ratiod_temporal::TemporalType::AR1) {
        layout.logit_rho_short_idx = idx++;
      } else {
        layout.logit_rho_short_idx = -1;
      }
      layout.short_term_start = idx;
      idx += data.multiscale_temporal_data.n_times;
      layout.short_term_end = idx;
    } else {
      layout.log_sigma2_short_idx = -1;
      layout.logit_rho_short_idx = -1;
      layout.short_term_start = layout.short_term_end = -1;
    }
  } else {
    layout.log_sigma2_trend_idx = -1;
    layout.trend_start = layout.trend_end = -1;
    layout.log_sigma2_seasonal_idx = -1;
    layout.seasonal_start = layout.seasonal_end = -1;
    layout.log_sigma2_short_idx = -1;
    layout.logit_rho_short_idx = -1;
    layout.short_term_start = layout.short_term_end = -1;
  }

  // SVC (Spatially-Varying Coefficients) parameters
  layout.has_svc = data.has_svc;
  if (layout.has_svc && data.svc_data.n_svc > 0) {
    // Log sigma2 per SVC term (spatial variance)
    layout.log_sigma2_svc_start = idx;
    idx += data.svc_data.n_svc;
    layout.log_sigma2_svc_end = idx;

    // Log phi/lengthscale per SVC term (spatial range)
    layout.log_phi_svc_start = idx;
    idx += data.svc_data.n_svc;
    layout.log_phi_svc_end = idx;

    // SVC spatial parameters:
    //   NNGP: w_flat[j * n_obs + i] for j in 0..n_svc-1, i in 0..n_obs-1
    //   HSGP: beta[j * m_total + k] for j in 0..n_svc-1, k in 0..m^2-1
    layout.svc_w_start = idx;
    if (data.svc_is_hsgp) {
      idx += data.svc_data.n_svc * data.svc_hsgp_data.m_total;
    } else {
      idx += data.svc_data.n_svc * data.svc_data.n_obs;
    }
    layout.svc_w_end = idx;
  } else {
    layout.log_sigma2_svc_start = layout.log_sigma2_svc_end = -1;
    layout.log_phi_svc_start = layout.log_phi_svc_end = -1;
    layout.svc_w_start = layout.svc_w_end = -1;
  }

  // Latent factors for unmeasured confounders
  layout.has_latent = data.has_latent;
  if (layout.has_latent && data.latent_n_factors > 0) {
    // Log sigma for each factor
    layout.log_sigma_latent_start = idx;
    idx += data.latent_n_factors;
    layout.log_sigma_latent_end = idx;

    // Factor scores (N x K)
    layout.latent_factor_start = idx;
    idx += data.N * data.latent_n_factors;
    layout.latent_factor_end = idx;
  } else {
    layout.log_sigma_latent_start = layout.log_sigma_latent_end = -1;
    layout.latent_factor_start = layout.latent_factor_end = -1;
  }

  // Spatiotemporal interaction
  layout.has_spatiotemporal = data.has_spatiotemporal;
  layout.is_st_gp = (data.has_spatiotemporal &&
                     (data.spatiotemporal_data.type == STType::SEPARABLE ||
                      data.spatiotemporal_data.type == STType::NONSEP_GP));

  if (layout.has_spatiotemporal && data.spatiotemporal_data.type != STType::NONE) {
    // log_tau for interaction precision
    layout.log_tau_st_idx = idx++;

    // Second precision removed for Type IV (single tau suffices)
    layout.log_tau_st2_idx = -1;

    // AR1 rho, where the interaction's time margin is one it reads. A Type I
    // interaction is iid over the whole grid and a Type III's time margin is
    // unstructured, so neither has a correlation to estimate; allocating one
    // there leaves a coordinate the sampler moves against nothing but its own
    // prior.
    const bool st_reads_time_margin =
        ratiod_spatiotemporal::st_time_margin_is_structured(
            data.spatiotemporal_data.type) || data.st_is_hsgp;
    if (data.spatiotemporal_data.temporal_type == TemporalType::AR1 &&
        st_reads_time_margin) {
      layout.logit_rho_st_idx = idx++;
    } else {
      layout.logit_rho_st_idx = -1;
    }

    // GP range parameters (for separable/non-separable GP)
    if (layout.is_st_gp) {
      layout.log_phi_st_space_idx = idx++;
      layout.log_phi_st_time_idx = idx++;
    } else {
      layout.log_phi_st_space_idx = -1;
      layout.log_phi_st_time_idx = -1;
    }

    // HSGP-ST: separate sigma2 and lengthscale for spectral basis interaction
    layout.is_st_hsgp = data.st_is_hsgp;
    if (data.st_is_hsgp) {
      layout.log_sigma2_st_hsgp_idx = idx++;
      layout.log_lengthscale_st_hsgp_idx = idx++;
    } else {
      layout.log_sigma2_st_hsgp_idx = -1;
      layout.log_lengthscale_st_hsgp_idx = -1;
    }

    // Spatiotemporal interaction effects
    layout.st_delta_start = idx;
    idx += data.spatiotemporal_data.n_params;
    layout.st_delta_end = idx;
  } else {
    layout.log_tau_st_idx = -1;
    layout.log_tau_st2_idx = -1;
    layout.logit_rho_st_idx = -1;
    layout.log_phi_st_space_idx = -1;
    layout.log_phi_st_time_idx = -1;
    layout.log_sigma2_st_hsgp_idx = -1;
    layout.log_lengthscale_st_hsgp_idx = -1;
    layout.is_st_hsgp = false;
    layout.st_delta_start = layout.st_delta_end = -1;
  }

  // HSGP (Hilbert Space GP) parameters
  layout.is_hsgp = (data.spatial_type == SpatialType::HSGP);
  if (layout.is_hsgp && data.has_hsgp) {
    layout.log_sigma2_hsgp_idx = idx++;
    layout.log_lengthscale_hsgp_idx = idx++;
    layout.hsgp_beta_start = idx;
    idx += data.hsgp_data.m_total;  // m^2 basis coefficients
    layout.hsgp_beta_end = idx;
  } else {
    layout.log_sigma2_hsgp_idx = -1;
    layout.log_lengthscale_hsgp_idx = -1;
    layout.hsgp_beta_start = layout.hsgp_beta_end = -1;
  }

  // TVC (Temporally-Varying Coefficients) parameters
  layout.has_tvc = data.has_tvc;
  if (layout.has_tvc && data.tvc_data.n_tvc > 0) {
    // Log precision per TVC term
    layout.log_tau_tvc_start = idx;
    idx += data.tvc_data.n_tvc;
    layout.log_tau_tvc_end = idx;

    // AR1 rho parameters (only if structure is AR1)
    if (data.tvc_data.structure == ratiod_temporal::TemporalType::AR1) {
      layout.logit_rho_tvc_start = idx;
      idx += data.tvc_data.n_tvc;
      layout.logit_rho_tvc_end = idx;
    } else {
      layout.logit_rho_tvc_start = layout.logit_rho_tvc_end = -1;
    }

    // TVC values: w[g, j, t] for g in groups, j in tvc terms, t in times
    // Layout: w_flat[g * n_tvc * n_times + j * n_times + t]
    layout.tvc_w_start = idx;
    idx += data.tvc_data.n_groups * data.tvc_data.n_tvc * data.tvc_data.n_times;
    layout.tvc_w_end = idx;
  } else {
    layout.log_tau_tvc_start = layout.log_tau_tvc_end = -1;
    layout.logit_rho_tvc_start = layout.logit_rho_tvc_end = -1;
    layout.tvc_w_start = layout.tvc_w_end = -1;
  }

  layout.total_params = idx;
  return layout;
}

int get_n_params(const ModelData& data) {
  ParamLayout layout = compute_param_layout(data);
  return layout.total_params;
}

// =====================================================================
// Likelihood functions
// =====================================================================

inline double log_lik_binomial(int y, int n, double eta) {
  // Numerically stable binomial log-likelihood
  if (eta > 0) {
    return y * eta - n * eta - n * std::log(1.0 + std::exp(-eta));
  } else {
    return y * eta - n * std::log(1.0 + std::exp(eta));
  }
}

inline double log_lik_negbin(int y, double mu, double phi) {
  if (mu <= 0 || phi <= 0) return -1e10;
  return std::lgamma(y + phi) - std::lgamma(phi) - std::lgamma(y + 1.0)
       + phi * std::log(phi / (mu + phi))
       + y * std::log(mu / (mu + phi));
}

inline double log_lik_poisson(int y, double mu) {
  if (mu <= 0) return -1e10;
  return y * std::log(mu) - mu - std::lgamma(y + 1.0);
}

inline double log_lik_gamma(double y, double shape, double mu) {
  if (y <= 0 || shape <= 0 || mu <= 0) return -1e10;
  double rate = shape / mu;
  return shape * std::log(rate) + (shape - 1.0) * std::log(y)
       - rate * y - std::lgamma(shape);
}

// Include vectorized gradient header AFTER log_lik_* functions and hmc_sampler.h
// so all types and helpers are defined.
#include "hmc_gradient_vectorized.h"

// Thread-local vectorized gradient workspace (avoids per-call allocation)
RATIOD_TLS_WORKSPACE_FN(vectorized::VecGradWorkspace, vec_grad_ws)

// =====================================================================
// Observation log-likelihood helper (fused with gradient computation)
// Matches compute_log_post observation loop exactly. Used by specialized
// H gradient functions to avoid a separate O(N) pass.
// =====================================================================

inline double compute_obs_ll(
    const ModelData& data, int i,
    double eta_num, double eta_denom,
    double phi_num, double phi_denom
) {
  if (data.model_type == ModelType::BINOMIAL) {
    // The eta-form likelihood drops the binomial coefficient, which is
    // constant in eta but not in the data. compute_log_post's own
    // observation loop adds it back, so a fused value without it reports a
    // different density than the one the gradient beside it describes.
    return log_lik_binomial(data.y_num[i], data.y_denom[i], eta_num)
         + ratiod::math::portable_lchoose(data.y_denom[i], data.y_num[i]);
  } else if (data.model_type == ModelType::NEGBIN_NEGBIN) {
    double mu_num = std::exp(eta_num);
    double mu_denom = std::exp(eta_denom);
    return log_lik_negbin(data.y_num[i], mu_num, phi_num)
         + log_lik_negbin(data.y_denom[i], mu_denom, phi_denom);
  } else if (data.model_type == ModelType::POISSON_GAMMA) {
    double mu_num = std::exp(eta_num);
    double mu_denom = std::exp(eta_denom);
    return log_lik_poisson(data.y_num[i], mu_num)
         + log_lik_gamma(data.y_denom_cont[i], phi_denom, mu_denom);
  } else if (data.model_type == ModelType::NEGBIN_GAMMA) {
    double mu_num = std::exp(eta_num);
    double mu_denom = std::exp(eta_denom);
    return log_lik_negbin(data.y_num[i], mu_num, phi_num)
         + log_lik_gamma(data.y_denom_cont[i], phi_denom, mu_denom);
  } else if (data.model_type == ModelType::GAMMA_GAMMA) {
    double mu_num = std::exp(eta_num);
    double mu_denom = std::exp(eta_denom);
    return log_lik_gamma(data.y_num_cont[i], phi_num, mu_num)
         + log_lik_gamma(data.y_denom_cont[i], phi_denom, mu_denom);
  } else if (data.model_type == ModelType::LOGNORMAL) {
    double log_y_num = std::log(data.y_num_cont[i]);
    double log_y_denom = std::log(data.y_denom_cont[i]);
    double z_num = (log_y_num - eta_num) / phi_num;
    double z_denom = (log_y_denom - eta_denom) / phi_denom;
    return -log_y_num - std::log(phi_num) - 0.5 * z_num * z_num
           -log_y_denom - std::log(phi_denom) - 0.5 * z_denom * z_denom;
  } else if (data.model_type == ModelType::BETA_BINOMIAL) {
    double p = 1.0 / (1.0 + std::exp(-eta_num));
    int y = data.y_num[i];
    int n = data.y_denom[i];
    double alpha = p * phi_num;
    double beta_param = (1.0 - p) * phi_num;
    return std::lgamma(y + alpha) + std::lgamma(n - y + beta_param) - std::lgamma(n + phi_num)
         - std::lgamma(alpha) - std::lgamma(beta_param) + std::lgamma(phi_num)
         + ratiod::math::portable_lchoose(n, y);
  }
  return 0.0;
}

// =====================================================================
// ICAR quadratic form: phi' Q phi
// =====================================================================

// Pointer-based version (zero allocation, used in hot gradient path)
static inline double icar_quadratic_form_ptr(
    const double* phi, int J,
    const ModelData& data
) {
  return ratiod::icar_quadratic_form_t(phi, J, data);
}

double icar_quadratic_form(
    const std::vector<double>& phi,
    const ModelData& data
) {
  return icar_quadratic_form_ptr(phi.data(), data.n_spatial_units, data);
}
// =====================================================================
// Log-posterior computation
// =====================================================================

// One density, in one place, so a term cannot reach the value the H path
// reports without also reaching the gradient the autodiff modes take of it.
// What used to keep the two apart -- the collapsed marginals and proper CAR,
// which no autodiff scalar can express, the parallel observation loop, the
// per-thread workspaces, and what a fused caller has already paid for -- lives
// behind `if constexpr` and LogPostOptions inside the template.
double compute_log_post(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    bool skip_obs_loop,
    const double* precomputed_tgp_log_prior
) {
  ratiod::LogPostOptions opts;
  opts.skip_obs_loop = skip_obs_loop;
  opts.precomputed_tgp_log_prior = precomputed_tgp_log_prior;
  return ratiod::compute_log_post_impl<double>(params, data, layout, opts);
}

// =====================================================================
// Analytical gradient for simple Poisson-Gamma models
// O(n) instead of O(n*p) - huge speedup for typical models
// =====================================================================

bool can_use_analytical_gradient(const ModelData& data, const ParamLayout& layout) {
  // Hand-coded gradients for basic models without complex structure
  bool is_basic_family = (data.model_type == ModelType::POISSON_GAMMA ||
                          data.model_type == ModelType::NEGBIN_NEGBIN ||
                          data.model_type == ModelType::NEGBIN_GAMMA ||
                          data.model_type == ModelType::BINOMIAL ||
                          data.model_type == ModelType::GAMMA_GAMMA ||
                          data.model_type == ModelType::BETA_BINOMIAL ||
                          data.model_type == ModelType::LOGNORMAL);

  // Check if spatial type is one we have hand-coded gradients for
  bool spatial_is_icar_bym2 = (data.spatial_type == SpatialType::ICAR ||
                               data.spatial_type == SpatialType::BYM2);

  // Temporal is OK alone or combined with ICAR/BYM2 spatial (no spatiotemporal interaction)
  // Note: Temporal GP is excluded - use autodiff for that
  bool temporal_ok = !layout.has_temporal ||
                     (layout.has_temporal && !layout.is_temporal_gp &&
                      !layout.has_spatiotemporal &&
                      (!layout.has_spatial || spatial_is_icar_bym2));

  // Spatial is OK for ICAR/BYM2 (alone or combined with temporal)
  bool spatial_ok = !layout.has_spatial ||
                    (layout.has_spatial && spatial_is_icar_bym2 && !layout.has_spatiotemporal);

  // ZI is OK for basic models, including with ICAR/BYM2 spatial and temporal
  // Components are additive in eta; ZI modifies residuals independently
  bool zi_ok = !layout.has_zi ||
               (layout.has_zi &&
                (!layout.has_spatial || spatial_is_icar_bym2));

  // Random slopes: both correlated (|) and uncorrelated (||) are supported
  // Can combine with ICAR/BYM2 spatial, temporal, and ZI
  // Components are additive in eta; gradient scatter is independent
  bool slopes_ok = !layout.has_re_slopes ||
                   (layout.has_re_slopes &&
                    (!layout.has_spatial || spatial_is_icar_bym2));

  return (is_basic_family &&
          !layout.is_gp && !layout.is_multiscale_gp && !layout.is_hsgp &&
          !layout.is_icar_collapsed && !layout.is_bym2_collapsed &&
          temporal_ok && spatial_ok && zi_ok && slopes_ok &&
          !layout.has_latent && !layout.has_spatiotemporal &&
          !layout.has_multiscale_temporal && !layout.has_tvc &&
          !layout.has_svc &&  // SVC has its own gradient function
          (data.n_re_terms <= 1 ||
           (data.n_re_terms > 1 && !layout.has_re_slopes)));  // Crossed RE (intercept-only) is OK
}

// Forward declarations of shared gradient helpers (defined after main gradient function)
struct CommonGradParams;
static inline CommonGradParams extract_common_params(
    const std::vector<double>& params, const ParamLayout& layout);
static inline void beta_gradient_prior(
    const ModelData& data, const ParamLayout& layout,
    const double* beta_num, const double* beta_denom, double* grad);
static inline void phi_gradient_prior(
    const ModelData& data, const ParamLayout& layout,
    double phi_num, double phi_denom, double* grad);
static inline void compute_obs_residuals(
    const ModelData& data, int i,
    double eta_num, double eta_denom,
    double phi_num, double phi_denom,
    double& dLL_deta_num, double& dLL_deta_denom);
static inline void scatter_beta_gradients(
    const ModelData& data, const ParamLayout& layout,
    int i, double dLL_deta_num, double dLL_deta_denom, double* grad);
static inline void scatter_re_gradient(
    const ModelData& data, const ParamLayout& layout,
    int i, double dLL_deta_num, double dLL_deta_denom, double* grad);
static inline void accumulate_phi_likelihood_grad(
    const ModelData& data, const ParamLayout& layout,
    int i, double eta_num, double eta_denom,
    double phi_num, double phi_denom, double* grad);
static inline void tau_temporal_prior_grad(
    const ModelData& data, const ParamLayout& layout,
    double tau_temporal, double* grad);
static inline void temporal_gmrf_prior_grad(
    const ModelData& data, const ParamLayout& layout,
    double tau_temporal, double rho_ar1,
    const TemporalView& tview, int T_len,
    const double* grad_temporal_lik, double* grad);
static inline double read_temporal_rho_ar1(
    const std::vector<double>& params, const ModelData& data,
    const ParamLayout& layout);
static inline void temporal_rho_prior_grad(
    const ModelData& data, const ParamLayout& layout,
    double rho_ar1, double* grad);
static inline double gp_pc_prior_grad_log_sigma2(
    double sigma2, double U, double alpha);
static inline void spatial_gmrf_prior_grad(
    const ModelData& data, const ParamLayout& layout,
    const double* phi_spatial, const double* phi_spatial_prior,
    double phi_raw_sum, bool centered,
    double tau_spatial,
    double sigma_s_bym2, double sigma_u_bym2, double rho_bym2,
    double rho_car,
    const double* theta_bym2,
    const double* grad_spatial_lik,
    double* grad);
static inline void spatial_lik_grad_to_param_scale(
    const ModelData& data, const ParamLayout& layout,
    double sigma_s_bym2, double sigma_u_bym2,
    const double* theta_bym2, const double* grad_theta_lik,
    double* grad_spatial_lik, double* grad);

void compute_gradient_analytical(
    const std::vector<double>& params_in,
    const ModelData& data,
    const ParamLayout& layout,
    std::vector<double>& grad,
    double* log_post_out = nullptr
) {
  // Centre the spatial block once, matching compute_log_post. The vectorized
  // fast paths index params directly, so centring here is what makes them see
  // the constrained field. The returned gradient is with respect to the raw
  // parameters, which is why the spatial likelihood gradient is projected below.
  std::vector<double> params_centered;
  double phi_raw_sum = 0.0;
  const bool center_spatial = spatial_block_is_centred(data, layout);
  if (center_spatial) {
    params_centered = params_in;
    phi_raw_sum = ratiod_constraints::center_spatial_block(
        params_centered, layout.spatial_start, data.n_spatial_units);
  }
  const std::vector<double>& params = center_spatial ? params_centered : params_in;

  int n_params = params.size();
  grad.assign(n_params, 0.0);

  // Fused log-posterior computation: accumulate observation log-likelihood
  // alongside gradients to avoid a separate O(N) pass.
  const bool compute_lp = (log_post_out != nullptr);
  double obs_log_lik = 0.0;

  // Extract parameters
  const double* beta_num = &params[layout.beta_num_start];
  const double* beta_denom = &params[layout.beta_denom_start];

  double log_sigma_re = 0.0, sigma_re = 1.0, tau_re = 1.0;
  const double* re = nullptr;
  if (layout.has_re) {
    log_sigma_re = params[layout.log_sigma_re_idx];
    sigma_re = std::exp(log_sigma_re);
    tau_re = 1.0 / (sigma_re * sigma_re + 1e-10);
    re = &params[layout.re_start];
  }

  double phi_num = 1.0, log_phi_num = 0.0;
  double phi_denom = 1.0, log_phi_denom = 0.0;
  if (layout.has_phi_num) {
    log_phi_num = params[layout.log_phi_num_idx];
    phi_num = std::exp(log_phi_num);
  }
  if (layout.has_phi_denom) {
    log_phi_denom = params[layout.log_phi_denom_idx];
    phi_denom = std::exp(log_phi_denom);
  }

  // ============ Prior gradients (cheap) ============

  // Beta priors: N(0, sigma_beta^2)
  beta_gradient_prior(data, layout, beta_num, beta_denom, grad.data());

  // sigma_re: Half-Cauchy prior with scale = data.sigma_re_scale (via log transform)
  // log_post = -log(1 + (sigma/scale)^2) + log(sigma) (Jacobian)
  // d/d(log_sigma) = -2*(sigma/scale)^2/(1+(sigma/scale)^2) + 1
  if (layout.has_re) {
    double ratio = sigma_re / data.sigma_re_scale;
    double ratio_sq = ratio * ratio;
    grad[layout.log_sigma_re_idx] = -2.0 * ratio_sq / (1.0 + ratio_sq) + 1.0;
  }

  // phi priors: Gamma(shape, rate) via log transform
  phi_gradient_prior(data, layout, phi_num, phi_denom, grad.data());

  // RE priors, and the effective effects a non-centred correlated term's
  // loops read. See hmc_re_blocks_grad.h.
  ReBlockGrad re_ws;
  if (!re_blocks_prior_grad(params, data, layout, grad, re_ws)) return;
  std::vector<std::vector<double>>& grad_re_slopes_lik = re_ws.lik;
  const int n_re_terms_slopes = re_ws.n_terms;
  const bool slopes_nc = re_ws.nc;
  const std::vector<double>& re_nc_flat = re_ws.re_nc_flat;

  // ============ Temporal prior gradients ============
  double log_tau_temporal = 0.0, tau_temporal = 1.0;
  double logit_rho_ar1 = 0.0, rho_ar1 = 0.0;
  int T_len = 0;
  const double* phi_temporal = nullptr;
  std::vector<double> grad_temporal_lik;  // Likelihood contribution
  TemporalView tview;

  if (layout.has_temporal) {
    log_tau_temporal = params[layout.log_tau_temporal_idx];
    tau_temporal = std::exp(log_tau_temporal);
    T_len = layout.temporal_end - layout.temporal_start;
    tview.read(params, data, layout);
    phi_temporal = tview.phi;
    grad_temporal_lik.assign(T_len, 0.0);

    // tau prior: Gamma(shape, rate) via log transform
    tau_temporal_prior_grad(data, layout, tau_temporal, grad.data());

    // AR1: extract rho and add prior
    if (data.temporal_type == TemporalType::AR1 && layout.logit_rho_ar1_idx >= 0) {
      logit_rho_ar1 = params[layout.logit_rho_ar1_idx];
      rho_ar1 = ratiod_ar1::rho_from_logit(logit_rho_ar1);
      temporal_rho_prior_grad(data, layout, rho_ar1, grad.data());
    }
  }

  // ============ Spatial prior gradients (ICAR and BYM2) ============
  double log_tau_spatial = 0.0, tau_spatial = 1.0;
  double sigma_s_bym2 = 1.0, sigma_u_bym2 = 1.0;
  double rho_bym2 = 0.5;  // Riebler mixing parameter
  int n_spatial = 0;
  const double* phi_spatial = nullptr;
  const double* theta_bym2 = nullptr;
  std::vector<double> grad_spatial_lik;  // Likelihood contribution

  if (layout.has_spatial) {
    n_spatial = data.n_spatial_units;
    phi_spatial = &params[layout.spatial_start];
    grad_spatial_lik.assign(n_spatial, 0.0);

    if (data.spatial_type == SpatialType::BYM2) {
      // BYM2 Riebler: derive sigma_s, sigma_u from sigma_total, rho
      double sigma_total = std::exp(params[layout.log_sigma_bym2_idx]);
      double logit_rho = params[layout.logit_rho_bym2_idx];
      rho_bym2 = 1.0 / (1.0 + std::exp(-logit_rho));
      sigma_s_bym2 = sigma_total * std::sqrt(rho_bym2);
      sigma_u_bym2 = sigma_total * std::sqrt(1.0 - rho_bym2);
      theta_bym2 = &params[layout.theta_bym2_start];

      // Half-Cauchy prior on sigma_total
      double ratio = sigma_total / data.sigma_re_scale;
      double ratio_sq = ratio * ratio;
      grad[layout.log_sigma_bym2_idx] = -2.0 * ratio_sq / (1.0 + ratio_sq) + 1.0;

      // Uniform(0,1) = Beta(1,1) on rho with logit Jacobian:
      // d/d(logit_rho) [log(rho) + log(1-rho)] = (1-rho) - rho = 1 - 2*rho
      grad[layout.logit_rho_bym2_idx] = 1.0 - 2.0 * rho_bym2;

      // Initialize theta gradients (N(0,1) prior: d/d(theta) = -theta)
      for (int s = 0; s < n_spatial; s++) {
        grad[layout.theta_bym2_start + s] = -theta_bym2[s];
      }
    } else {
      // ICAR: extract tau
      log_tau_spatial = params[layout.log_tau_spatial_idx];
      tau_spatial = std::exp(log_tau_spatial);

      // Gamma prior on tau via log transform
      grad[layout.log_tau_spatial_idx] = (data.tau_spatial_shape - 1.0)
                                         - data.tau_spatial_rate * tau_spatial + 1.0;
    }
  }

  // ============ Zero-inflation prior gradients ============
  const double* beta_zi = nullptr;
  std::vector<double> grad_beta_zi;
  double tau_zi = 1.0;

  if (layout.has_zi && data.p_zi > 0) {
    beta_zi = &params[layout.beta_zi_start];
    tau_zi = 1.0 / (data.zi_prior_sd * data.zi_prior_sd + 1e-10);
    grad_beta_zi.assign(data.p_zi, 0.0);

    // N(0, zi_prior_sd^2) prior on ZI coefficients
    for (int j = 0; j < data.p_zi; j++) {
      grad[layout.beta_zi_start + j] = -tau_zi * beta_zi[j];
    }
  }

  // ============ One-inflation (OI) prior gradients ============
  const double* beta_oi = nullptr;
  std::vector<double> grad_beta_oi;
  double tau_oi = 1.0;

  if (layout.has_oi && data.p_oi > 0) {
    beta_oi = &params[layout.beta_oi_start];
    tau_oi = 1.0 / (data.oi_prior_sd * data.oi_prior_sd + 1e-10);
    grad_beta_oi.assign(data.p_oi, 0.0);

    // N(0, oi_prior_sd^2) prior on OI coefficients
    for (int j = 0; j < data.p_oi; j++) {
      grad[layout.beta_oi_start + j] = -tau_oi * beta_oi[j];
    }
  }

  // ============ Likelihood gradients (O(n)) ============

  // Try fused single-pass gradient first (best for small p <= 4).
  // Then fall back to 3-pass vectorized (better for larger p with Eigen).
  // Finally fall back to scalar loop for complex models (ZI, slopes, etc.).
  bool used_vectorized = vectorized::dispatch_fused_gradient(
      params, data, layout, grad, obs_log_lik,
      compute_lp, grad_temporal_lik, grad_spatial_lik, vec_grad_ws());

  // Fall back to 3-pass vectorized for p > 4 (Eigen matvec is beneficial)
  if (!used_vectorized) {
    used_vectorized = vectorized::dispatch_vectorized_gradient(
        params, data, layout, grad, obs_log_lik,
        compute_lp, grad_temporal_lik, grad_spatial_lik, vec_grad_ws());
  }

  // Hybrid path for slopes models (no ZI/OI): vectorize X*beta + residuals,
  // scalar loop for slopes RE expansion + gradient scatter
  if (!used_vectorized && (layout.has_re_slopes || layout.has_re_correlated_slopes) &&
      !layout.has_zi && !layout.has_oi &&
      !data.has_svc && !data.has_tvc && !data.has_latent &&
      !data.has_spatiotemporal && !data.has_temporal_gp && !data.has_multiscale_temporal &&
      data.spatial_type != SpatialType::GP &&
      data.spatial_type != SpatialType::MULTISCALE_GP &&
      data.spatial_type != SpatialType::HSGP) {

    const int N = data.N;
    const bool is_binomial = (data.model_type == ModelType::BINOMIAL ||
                              data.model_type == ModelType::BETA_BINOMIAL);

    // --- Pass 1: Vectorized eta base (Eigen matvec) ---
    using RowMajorMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using VectorXd = Eigen::VectorXd;
    vec_grad_ws().init(N);

    Eigen::Map<const RowMajorMatrix> X_num(data.X_num_flat.data(), N, data.p_num);
    Eigen::Map<const VectorXd> b_num(beta_num, data.p_num);
    Eigen::Map<VectorXd> eta_n(vec_grad_ws().eta_num.data(), N);
    eta_n.noalias() = X_num * b_num;

    if (!is_binomial && data.p_denom > 0) {
      Eigen::Map<const RowMajorMatrix> X_denom(data.X_denom_flat.data(), N, data.p_denom);
      Eigen::Map<const VectorXd> b_denom(beta_denom, data.p_denom);
      Eigen::Map<VectorXd> eta_d(vec_grad_ws().eta_denom.data(), N);
      eta_d.noalias() = X_denom * b_denom;
    } else if (!is_binomial) {
      std::memset(vec_grad_ws().eta_denom.data(), 0, N * sizeof(double));
    }

    // Pre-compute sigma values for NC slopes (avoid N * n_slopes exp() calls)
    std::vector<std::vector<double>> precomp_sigma(n_re_terms_slopes);
    if (slopes_nc) {
      for (int t = 0; t < n_re_terms_slopes; t++) {
        int n_coefs = layout.re_n_coefs_multi[t];
        precomp_sigma[t].resize(n_coefs);
        for (int c = 0; c < n_coefs; c++) {
          precomp_sigma[t][c] = std::exp(params[layout.log_sigma_re_slopes[t][c]]);
        }
      }
    }

    // Scalar loop: add slopes RE + spatial + temporal to eta
    // Track per-obs indices for scatter pass
    std::vector<int> obs_s_idx(N, -1);       // spatial group index
    std::vector<int> obs_t_idx(N, -1);       // temporal flat index
    for (int i = 0; i < N; i++) {
      // Slopes RE contribution
      if (layout.has_re_slopes && n_re_terms_slopes > 0) {
        int re_group_idx_i = data.re_group_multi_flat[i * data.n_re_terms + 0];
        if (re_group_idx_i > 0) {
          int g = re_group_idx_i - 1;
          int n_coefs = layout.re_n_coefs_multi[0];
          int re_base = layout.re_start_multi[0] + g * n_coefs;

          bool is_corr_t = !re_nc_flat.empty() &&
                           layout.re_correlated_multi.size() > 0 &&
                           layout.re_correlated_multi[0] && n_coefs > 1;
          bool is_uncorr_nc = !is_corr_t && slopes_nc;

          // Intercept
          double re_contrib;
          if (is_corr_t) {
            re_contrib = re_nc_flat[re_base];
          } else if (is_uncorr_nc) {
            re_contrib = precomp_sigma[0][0] * params[re_base];
          } else {
            re_contrib = params[re_base];
          }

          // Slope contributions
          int n_slopes = n_coefs - 1;
          if (n_slopes > 0 && !data.re_slope_matrices[0].empty()) {
            for (int s = 0; s < n_slopes; s++) {
              double x_slope = data.re_slope_matrices[0][i * n_slopes + s];
              double re_slope;
              if (is_corr_t) {
                re_slope = re_nc_flat[re_base + 1 + s];
              } else if (is_uncorr_nc) {
                re_slope = precomp_sigma[0][1 + s] * params[re_base + 1 + s];
              } else {
                re_slope = params[re_base + 1 + s];
              }
              re_contrib += re_slope * x_slope;
            }
          }

          vec_grad_ws().eta_num[i] += re_contrib;
          if (!is_binomial) vec_grad_ws().eta_denom[i] += re_contrib;
        }
      }

      // Spatial effect (ICAR or BYM2 only — GP/HSGP/MSGP excluded above)
      if (layout.has_spatial && !data.spatial_group.empty() && data.spatial_group[i] > 0) {
        int s = data.spatial_group[i] - 1;
        obs_s_idx[i] = s;
        double spatial_eff;
        if (data.spatial_type == SpatialType::BYM2) {
          spatial_eff = sigma_s_bym2 * data.bym2_scale_factor * phi_spatial[s] + sigma_u_bym2 * theta_bym2[s];
        } else {
          spatial_eff = phi_spatial[s];
        }
        vec_grad_ws().eta_num[i] += spatial_eff;
        if (!is_binomial) vec_grad_ws().eta_denom[i] += spatial_eff;
      }

      // Temporal effect
      if (layout.has_temporal && !data.temporal_time_idx.empty() && data.temporal_time_idx[i] > 0) {
        int t = data.temporal_time_idx[i] - 1;
        int g = data.temporal_group_idx[i] - 1;
        int t_flat = g * data.n_times + t;
        obs_t_idx[i] = t_flat;
        vec_grad_ws().eta_num[i] += phi_temporal[t_flat];
        if (!is_binomial) vec_grad_ws().eta_denom[i] += phi_temporal[t_flat];
      }
    }

    // --- Pass 2+3: Vectorized residuals + beta grads (template-dispatched) ---
    {
      double grad_phi_num_lik_v = 0.0, grad_phi_denom_lik_v = 0.0;
      vectorized::dispatch_residuals_and_beta_grads(
          data, layout,
          vec_grad_ws().eta_num.data(), vec_grad_ws().eta_denom.data(),
          vec_grad_ws().resid_num.data(), vec_grad_ws().resid_denom.data(),
          grad.data(), grad_phi_num_lik_v, grad_phi_denom_lik_v,
          obs_log_lik, compute_lp, phi_num, phi_denom, vec_grad_ws());
    }

    // Scatter residuals to slopes RE, spatial, temporal gradient buffers
    for (int i = 0; i < N; i++) {
      double dLL_num = vec_grad_ws().resid_num[i];
      double dLL_denom = vec_grad_ws().resid_denom[i];
      double dLL_shared = dLL_num + dLL_denom;

      // Slopes RE gradient scatter
      if (layout.has_re_slopes && n_re_terms_slopes > 0) {
        int re_group_idx_i = data.re_group_multi_flat[i * data.n_re_terms + 0];
        if (re_group_idx_i > 0) {
          int g = re_group_idx_i - 1;
          int n_coefs = layout.re_n_coefs_multi[0];

          // Intercept gradient
          grad_re_slopes_lik[0][g * n_coefs] += dLL_shared;

          // Slope gradients (chain rule: d(LL)/d(re_slope) = d(LL)/d(eta) * x_slope)
          int n_slopes = n_coefs - 1;
          if (n_slopes > 0 && !data.re_slope_matrices[0].empty()) {
            for (int s = 0; s < n_slopes; s++) {
              double x_slope = data.re_slope_matrices[0][i * n_slopes + s];
              grad_re_slopes_lik[0][g * n_coefs + 1 + s] += dLL_shared * x_slope;
            }
          }
        }
      }

      // Spatial gradient scatter
      if (obs_s_idx[i] >= 0) {
        int s = obs_s_idx[i];
        if (data.spatial_type == SpatialType::BYM2) {
          grad_spatial_lik[s] += dLL_shared * sigma_s_bym2 * data.bym2_scale_factor;
          grad[layout.theta_bym2_start + s] += dLL_shared * sigma_u_bym2;
        } else {
          grad_spatial_lik[s] += dLL_shared;
        }
      }

      // Temporal gradient scatter
      if (obs_t_idx[i] >= 0) {
        grad_temporal_lik[obs_t_idx[i]] += data.temporal_shared ? dLL_shared : dLL_num;
      }
    }

    used_vectorized = true;
  }

  if (!used_vectorized) {

  // Accumulators for beta gradients (will be added via X' * residual)
  std::vector<double> grad_beta_num(data.p_num, 0.0);
  std::vector<double> grad_beta_denom(data.p_denom, 0.0);
  double grad_phi_num_lik = 0.0;
  double grad_phi_denom_lik = 0.0;

  // Every likelihood sum the observation loop builds has a region in each
  // thread's slot of `red_partials`; the slots are added afterwards in
  // thread-index order. Nothing is combined by the runtime: libomp's atomic
  // and tree-reduction paths for doubles drop updates on aarch64-w64-mingw
  // (omp_sum.h), and a critical section or atomic sums in thread-arrival
  // order, which varies from run to run. The buffer is sized by
  // omp_get_max_threads() and the region below carries no num_threads clause,
  // so both read the same value and every thread of the team has a slot. The
  // backends that take a per-fit thread count scope it (omp_thread_scope.h),
  // which holds that value still for the length of a fit.
  #ifdef _OPENMP
  const int grad_team = std::max(1, omp_get_max_threads());
  #else
  const int grad_team = 1;
  #endif
  const bool acc_zi = layout.has_zi && data.p_zi > 0;
  const bool acc_oi = layout.has_oi && data.p_oi > 0;
  const bool acc_slopes = layout.has_re_slopes && n_re_terms_slopes > 0 &&
                          !grad_re_slopes_lik.empty();
  const bool acc_bym2 = layout.has_spatial &&
                        data.spatial_type == SpatialType::BYM2;
  const int n_re_acc = layout.has_re ? data.n_re_groups : 0;
  const int n_re_crossed_acc =
      (layout.has_re && data.n_re_terms > 1) ? data.total_re_groups : 0;
  const int n_slopes_acc =
      acc_slopes ? (int)grad_re_slopes_lik[0].size() : 0;
  struct {
    int beta_num, beta_denom, phi_num, phi_denom, re, re_crossed, ll,
        beta_zi, beta_oi, re_slopes, temporal, spatial, theta_bym2, stride;
  } acc_at;
  {
    int n = 0;
    auto region = [&n](int len) { const int at = n; n += len; return at; };
    acc_at.beta_num = region(data.p_num);
    acc_at.beta_denom = region(data.p_denom);
    acc_at.phi_num = region(1);
    acc_at.phi_denom = region(1);
    acc_at.re = region(n_re_acc);
    acc_at.re_crossed = region(n_re_crossed_acc);
    acc_at.ll = region(1);
    acc_at.beta_zi = region(acc_zi ? data.p_zi : 0);
    acc_at.beta_oi = region(acc_oi ? data.p_oi : 0);
    acc_at.re_slopes = region(n_slopes_acc);
    acc_at.temporal = region((int)grad_temporal_lik.size());
    acc_at.spatial = region((int)grad_spatial_lik.size());
    acc_at.theta_bym2 = region(acc_bym2 ? n_spatial : 0);
    // Whole cache lines per slot, so neighbouring threads share none.
    acc_at.stride = (n + 7) / 8 * 8;
  }
  std::vector<double> red_partials((size_t)grad_team * acc_at.stride, 0.0);

  #ifdef _OPENMP
  #pragma omp parallel
  #endif
  {
    #ifdef _OPENMP
    double* acc = &red_partials[(size_t)omp_get_thread_num() * acc_at.stride];
    #else
    double* acc = red_partials.data();
    #endif
    // Pre-allocated per-obs group index buffer for crossed RE (reused across iterations)
    std::vector<int> re_idx_multi_buf(
        (layout.has_re && data.n_re_terms > 1) ? data.n_re_terms : 0, -1);

    #ifdef _OPENMP
    #pragma omp for schedule(static)
    #endif
    for (int i = 0; i < data.N; i++) {
      // Compute linear predictors
      double eta_num = ratiod_linalg::dot_product(
          &data.X_num_flat[i * data.p_num], beta_num, data.p_num);
      double eta_denom = (data.p_denom > 0) ?
          ratiod_linalg::dot_product(
              &data.X_denom_flat[i * data.p_denom], beta_denom, data.p_denom) :
          0.0;

      // Add RE if present
      int re_idx = -1;
      int re_term_idx = -1;
      int re_group_idx = -1;
      int re_n_coefs_i = 1;
      std::vector<double> re_slope_x_i;  // Slope design values for this obs
      int n_crossed_terms = 0;
      if (layout.has_re) {
        if (layout.has_re_slopes && n_re_terms_slopes > 0) {
          // Random slopes case: handle first term only (single RE term for H gradients)
          re_term_idx = 0;
          re_group_idx = data.re_group_multi_flat[i * data.n_re_terms + 0];
          if (re_group_idx > 0) {
            int g = re_group_idx - 1;
            re_n_coefs_i = layout.re_n_coefs_multi[0];
            int re_base = layout.re_start_multi[0] + g * re_n_coefs_i;

            // For correlated slopes: use pre-computed non-centered re (re_nc_flat)
            // For uncorrelated NC slopes: compute re = sigma * z on the fly
            // For uncorrelated centered slopes: params store re directly
            bool is_corr_t = !re_nc_flat.empty() &&
                             layout.re_correlated_multi.size() > 0 &&
                             layout.re_correlated_multi[0] && re_n_coefs_i > 1;
            bool is_uncorr_nc = !is_corr_t && slopes_nc;

            // Intercept contribution
            double re_contrib;
            if (is_corr_t) {
              re_contrib = re_nc_flat[re_base];
            } else if (is_uncorr_nc) {
              double sigma_int = std::exp(params[layout.log_sigma_re_slopes[0][0]]);
              re_contrib = sigma_int * params[re_base];
            } else {
              re_contrib = params[re_base];
            }

            // Slope contributions
            int n_slopes = re_n_coefs_i - 1;
            re_slope_x_i.resize(n_slopes);
            if (n_slopes > 0 && !data.re_slope_matrices[0].empty()) {
              for (int s = 0; s < n_slopes; s++) {
                double x_slope = data.re_slope_matrices[0][i * n_slopes + s];
                re_slope_x_i[s] = x_slope;
                double re_slope;
                if (is_corr_t) {
                  re_slope = re_nc_flat[re_base + 1 + s];
                } else if (is_uncorr_nc) {
                  double sigma_s = std::exp(params[layout.log_sigma_re_slopes[0][1 + s]]);
                  re_slope = sigma_s * params[re_base + 1 + s];
                } else {
                  re_slope = params[re_base + 1 + s];
                }
                re_contrib += re_slope * x_slope;
              }
            }

            eta_num += re_contrib;
            if (data.model_type != ModelType::BINOMIAL) {
              eta_denom += re_contrib;
            }
          }
        } else if (data.n_re_terms > 1) {
          // Crossed RE (multiple intercept-only terms)
          // Non-centered: re_val = sigma * z; centered: re_val = params directly
          n_crossed_terms = data.n_re_terms;
          for (int t = 0; t < n_crossed_terms; t++) {
            int group_idx = data.re_group_multi_flat[i * n_crossed_terms + t];
            if (group_idx > 0) {
              int g = group_idx - 1;
              re_idx_multi_buf[t] = g;
              double z_or_re = params[layout.re_start_multi[t] + g];
              double re_val = z_or_re;
              if (data.re_parameterization == 1) {
                double sigma_t = std::exp(params[layout.log_sigma_re_multi[t]]);
                re_val = sigma_t * z_or_re;
              }
              eta_num += re_val;
              if (data.model_type != ModelType::BINOMIAL) {
                eta_denom += re_val;
              }
            } else {
              re_idx_multi_buf[t] = -1;
            }
          }
        } else if (data.re_group[i] > 0) {
          // Simple intercept-only RE (single term)
          // Non-centered: re = sigma * z; centered: re = params directly
          re_idx = data.re_group[i] - 1;
          double re_val = re[re_idx];
          if (data.re_parameterization == 1) {
            re_val = sigma_re * re_val;  // re_val was z, now sigma*z
          }
          eta_num += re_val;
          if (data.model_type != ModelType::BINOMIAL) {
            eta_denom += re_val;
          }
        }
      }
      // Add temporal effect if present
      int t_idx = -1;
      if (layout.has_temporal && !data.temporal_time_idx.empty() && data.temporal_time_idx[i] > 0) {
        int t = data.temporal_time_idx[i] - 1;
        int g = data.temporal_group_idx[i] - 1;
        t_idx = g * data.n_times + t;  // Panel temporal: flat index
        double temporal_effect = phi_temporal[t_idx];
        eta_num += temporal_effect;
        if (data.model_type != ModelType::BINOMIAL) {
          eta_denom += temporal_effect;
        }
      }

      // Add spatial effect if present
      int s_idx = -1;
      double d_spatial_d_phi = 0.0;  // Derivative of spatial_effect wrt phi_spatial
      double d_spatial_d_theta = 0.0;  // Derivative of spatial_effect wrt theta_bym2
      if (layout.has_spatial && !data.spatial_group.empty() && data.spatial_group[i] > 0) {
        s_idx = data.spatial_group[i] - 1;
        double spatial_effect;
        if (data.spatial_type == SpatialType::BYM2) {
          // BYM2: spatial_effect = sigma_s * scale * phi + sigma_u * theta
          double scaled_phi = phi_spatial[s_idx] * data.bym2_scale_factor;
          spatial_effect = sigma_s_bym2 * scaled_phi + sigma_u_bym2 * theta_bym2[s_idx];
          d_spatial_d_phi = sigma_s_bym2 * data.bym2_scale_factor;
          d_spatial_d_theta = sigma_u_bym2;
        } else {
          // ICAR: spatial_effect = phi_spatial
          spatial_effect = phi_spatial[s_idx];
          d_spatial_d_phi = 1.0;
        }
        eta_num += spatial_effect;
        if (data.model_type != ModelType::BINOMIAL) {
          eta_denom += spatial_effect;
        }
      }

      double resid_num = 0.0;
      double resid_denom = 0.0;
      double grad_phi_num_i = 0.0;
      double grad_phi_denom_i = 0.0;
      double grad_logit_zi_i = 0.0;
      double grad_logit_oi_i = 0.0;

      // Compute ZI linear predictor if applicable
      double logit_zi = 0.0;
      double zi_prob = 0.0;
      if (layout.has_zi && data.p_zi > 0) {
        logit_zi = ratiod_linalg::dot_product(
            &data.X_zi_flat[i * data.p_zi], beta_zi, data.p_zi);
        zi_prob = 1.0 / (1.0 + std::exp(-logit_zi));
      }

      // Compute OI linear predictor if applicable
      double logit_oi = 0.0;
      double oi_prob = 0.0;
      if (layout.has_oi && data.p_oi > 0) {
        logit_oi = ratiod_linalg::dot_product(
            &data.X_oi_flat[i * data.p_oi], beta_oi, data.p_oi);
        oi_prob = 1.0 / (1.0 + std::exp(-logit_oi));
      }

      if (data.model_type == ModelType::BINOMIAL) {
        // ---- BINOMIAL ----
        // p = inv_logit(eta_num), LL = y*log(p) + (n-y)*log(1-p)
        // d(LL)/d(eta) = y - n*p
        double p = 1.0 / (1.0 + std::exp(-eta_num));
        int n_trials = data.y_denom[i];
        int y_num_i = data.y_num[i];

        if (layout.has_zi && data.zi_type == ratiod_zi::ZIType::ZI_BINOMIAL) {
          // ZI-Binomial
          if (y_num_i == 0) {
            // P(Y=0) = zi + (1-zi)*(1-p)^n
            double p0_binom = std::pow(1.0 - p, n_trials);  // (1-p)^n
            double p0 = zi_prob + (1.0 - zi_prob) * p0_binom;
            // d(LL)/d(eta) = (1-zi) * d((1-p)^n)/d(eta) / p0
            // d((1-p)^n)/d(eta) = n*(1-p)^(n-1) * (-p*(1-p)) = -n*p*(1-p)^n
            resid_num = -(1.0 - zi_prob) * n_trials * p * p0_binom / p0;
            // Gradient w.r.t. logit_zi
            grad_logit_zi_i = zi_prob * (1.0 - zi_prob) * (1.0 - p0_binom) / p0;
          } else {
            // P(Y=y) = (1-zi) * Binomial(y|n,p)
            resid_num = y_num_i - n_trials * p;
            grad_logit_zi_i = -zi_prob;  // d/d(logit_zi) log(1-zi) = -zi
          }
        } else if (layout.has_zi && data.zi_type == ratiod_zi::ZIType::HURDLE_BINOMIAL) {
          // Hurdle-Binomial
          if (y_num_i == 0) {
            // P(Y=0) = 1 - theta
            resid_num = 0.0;  // No p contribution when y=0
            grad_logit_zi_i = -zi_prob;  // d/d(logit_theta) log(1-theta) = -theta
          } else {
            // P(Y=y|Y>0) * theta = theta * TruncBinomial(y|n,p)
            double p0_binom = std::pow(1.0 - p, n_trials);
            double normalizer = 1.0 - p0_binom;
            if (normalizer < 1e-12) normalizer = 1e-12;
            // Gradient from truncated binomial
            // d(log(1-(1-p)^n))/d(eta) = n*p*(1-p)^n / (1-(1-p)^n)
            double grad_normalizer = n_trials * p * p0_binom / normalizer;
            resid_num = (y_num_i - n_trials * p) - grad_normalizer;
            grad_logit_zi_i = 1.0 - zi_prob;  // d/d(logit_theta) log(theta) = 1-theta
          }
        } else if (layout.has_oi && data.zi_type == ratiod_zi::ZIType::OI_BINOMIAL) {
          // OI-Binomial (One-inflation only)
          // P(Y=n) = oi + (1-oi) * p^n
          // P(Y=y, y<n) = (1-oi) * Binomial(y|n,p)
          if (y_num_i == n_trials) {
            // y = n (structural one or binomial one)
            double pn = std::pow(p, n_trials);  // p^n
            double P_yn = oi_prob + (1.0 - oi_prob) * pn;
            if (P_yn < 1e-12) P_yn = 1e-12;
            // d(log P)/d(eta) = (1-oi) * n * p^(n-1) * p*(1-p) / P
            //                 = (1-oi) * n * p^n * (1-p) / P
            resid_num = (1.0 - oi_prob) * n_trials * pn * (1.0 - p) / P_yn;
            // d(log P)/d(logit_oi) = oi*(1-oi)*(1 - p^n) / P
            grad_logit_oi_i = oi_prob * (1.0 - oi_prob) * (1.0 - pn) / P_yn;
          } else {
            // y < n: P(Y=y) = (1-oi) * Binomial(y|n,p)
            resid_num = y_num_i - n_trials * p;  // Standard binomial residual
            grad_logit_oi_i = -oi_prob;  // d/d(logit_oi) log(1-oi) = -oi
          }
        } else if (layout.has_oi && data.zi_type == ratiod_zi::ZIType::ZOIB) {
          // ZOIB (Zero-One Inflated Binomial) - MIXTURE MODEL
          // P(Y=0) = zi + (1-zi)*(1-oi)*(1-p)^n
          // P(Y=n) = (1-zi)*(oi + (1-oi)*p^n)
          // P(Y=y, 0<y<n) = (1-zi)*(1-oi)*Binomial(y|n,p)
          // Note: zi_prob = zi, oi_prob = oi
          if (y_num_i == 0) {
            // y = 0: P = zi + (1-zi)*(1-oi)*(1-p)^n = A + B
            double binom_zero = std::pow(1.0 - p, n_trials);  // (1-p)^n
            double A = zi_prob;  // structural zero component
            double B = (1.0 - zi_prob) * (1.0 - oi_prob) * binom_zero;  // binomial zero
            double P = A + B;
            if (P < 1e-12) P = 1e-12;

            // d(log P)/d(eta) = (1-zi)*(1-oi) * d((1-p)^n)/d(eta) / P
            // d((1-p)^n)/d(eta) = n*(1-p)^(n-1) * (-p*(1-p)) = -n*(1-p)^n * p
            double d_binom_d_eta = -n_trials * binom_zero * p;
            resid_num = (1.0 - zi_prob) * (1.0 - oi_prob) * d_binom_d_eta / P;

            // d(log P)/d(logit_zi) = [dA - B] * zi*(1-zi) / P
            // dA/d(logit_zi) = zi*(1-zi), dB/d(logit_zi) = -(1-oi)*binom_zero * zi*(1-zi)
            grad_logit_zi_i = zi_prob * (1.0 - zi_prob) * (1.0 - (1.0 - oi_prob) * binom_zero) / P;

            // d(log P)/d(logit_oi) = dB/d(logit_oi) / P
            // dB/d(logit_oi) = -(1-zi)*binom_zero * oi*(1-oi)
            grad_logit_oi_i = -(1.0 - zi_prob) * binom_zero * oi_prob * (1.0 - oi_prob) / P;

          } else if (y_num_i == n_trials) {
            // y = n: P = (1-zi)*(oi + (1-oi)*p^n) = (1-zi)*C
            double pn = std::pow(p, n_trials);  // p^n
            double C = oi_prob + (1.0 - oi_prob) * pn;  // oi + (1-oi)*p^n
            double P = (1.0 - zi_prob) * C;
            if (P < 1e-12) P = 1e-12;

            // d(log P)/d(eta) = (1-zi)*(1-oi) * d(p^n)/d(eta) / P
            // d(p^n)/d(eta) = n*p^(n-1) * p*(1-p) = n*p^n*(1-p)
            double d_pn_d_eta = n_trials * pn * (1.0 - p);
            resid_num = (1.0 - zi_prob) * (1.0 - oi_prob) * d_pn_d_eta / P;

            // d(log P)/d(logit_zi) = d(log(1-zi))/d(logit_zi) = -zi
            grad_logit_zi_i = -zi_prob;

            // d(log P)/d(logit_oi) = dC/d(logit_oi) / C
            // dC/d(logit_oi) = oi*(1-oi) - p^n * oi*(1-oi) = oi*(1-oi)*(1 - p^n)
            grad_logit_oi_i = oi_prob * (1.0 - oi_prob) * (1.0 - pn) / C;

          } else {
            // 0 < y < n: P = (1-zi)*(1-oi)*Binomial
            // log P = log(1-zi) + log(1-oi) + log_binom
            resid_num = y_num_i - n_trials * p;  // Standard binomial residual
            grad_logit_zi_i = -zi_prob;  // d/d(logit_zi) log(1-zi) = -zi
            grad_logit_oi_i = -oi_prob;  // d/d(logit_oi) log(1-oi) = -oi
          }
        } else {
          // Standard binomial (no ZI)
          resid_num = y_num_i - n_trials * p;
        }
        // No denominator contribution for binomial

      } else if (data.model_type == ModelType::NEGBIN_NEGBIN) {
        // ---- NEGBIN_NEGBIN ----
        double mu_num = std::exp(eta_num);
        double mu_denom = std::exp(eta_denom);
        int y_num_i = data.y_num[i];
        int y_denom_i = data.y_denom[i];

        // Denominator NegBin gradient (always standard, not ZI)
        double denom_d = mu_denom + phi_denom;
        resid_denom = y_denom_i - mu_denom * (y_denom_i + phi_denom) / denom_d;
        grad_phi_denom_i = ratiod::math::portable_digamma(y_denom_i + phi_denom) - ratiod::math::portable_digamma(phi_denom)
                           + std::log(phi_denom / denom_d)
                           + (mu_denom - y_denom_i) / denom_d;

        // Numerator with ZI handling
        if (layout.has_zi && data.zi_type == ratiod_zi::ZIType::ZI_NEGBIN) {
          // ZI-NegBin numerator
          double p0_nb = std::pow(phi_num / (phi_num + mu_num), phi_num);

          if (y_num_i == 0) {
            // P(Y=0) = zi + (1-zi)*p0_nb
            double p0 = zi_prob + (1.0 - zi_prob) * p0_nb;
            // d(LL)/d(mu) = (1-zi) * d(p0_nb)/d(mu) / p0
            // d(p0_nb)/d(mu) = phi * (phi/(phi+mu))^phi * (-1/(phi+mu)) = -phi * p0_nb / (phi+mu)
            double d_p0_nb_d_mu = -phi_num * p0_nb / (phi_num + mu_num);
            resid_num = (1.0 - zi_prob) * d_p0_nb_d_mu * mu_num / p0;
            // Gradient w.r.t. logit_zi
            grad_logit_zi_i = zi_prob * (1.0 - zi_prob) * (1.0 - p0_nb) / p0;
            // phi gradient for ZI-NegBin at y=0 (complex, using approximation)
            grad_phi_num_i = (1.0 - zi_prob) * p0_nb * (std::log(phi_num / (phi_num + mu_num)) + mu_num / (phi_num + mu_num)) / p0;
          } else {
            // P(Y=y) = (1-zi) * NB(y|mu,phi)
            double denom_num = mu_num + phi_num;
            resid_num = y_num_i - mu_num * (y_num_i + phi_num) / denom_num;
            grad_logit_zi_i = -zi_prob;  // d/d(logit_zi) log(1-zi) = -zi
            grad_phi_num_i = ratiod::math::portable_digamma(y_num_i + phi_num) - ratiod::math::portable_digamma(phi_num)
                             + std::log(phi_num / denom_num)
                             + (mu_num - y_num_i) / denom_num;
          }
        } else if (layout.has_zi && data.zi_type == ratiod_zi::ZIType::HURDLE_NEGBIN) {
          // Hurdle-NegBin numerator
          if (y_num_i == 0) {
            // P(Y=0) = 1 - theta, where theta = sigmoid(logit_zi) here represents P(Y>0)
            // Note: for hurdle, logit_zi parameterizes theta = P(Y>0), so zi_prob IS theta
            resid_num = 0.0;  // No mu contribution when y=0
            grad_logit_zi_i = -zi_prob;  // d/d(logit_theta) log(1-theta) = -theta
            grad_phi_num_i = 0.0;
          } else {
            // P(Y=y|Y>0) * theta = theta * TruncNB(y|mu,phi)
            double p0_nb = std::pow(phi_num / (phi_num + mu_num), phi_num);
            double log_normalizer = std::log(1.0 - p0_nb);
            double denom_num = mu_num + phi_num;
            // Gradient from truncated NB: NB residual MINUS normalizer correction
            // LL = log NB(y) - log(1-p0), so d(LL)/d(eta) = NB_resid - d(log(1-p0))/d(mu)*mu
            // d(log(1-p0))/d(mu) = phi*p0 / ((phi+mu)*(1-p0))
            resid_num = y_num_i - mu_num * (y_num_i + phi_num) / denom_num
                        - phi_num * p0_nb * mu_num / ((phi_num + mu_num) * (1.0 - p0_nb));
            grad_logit_zi_i = 1.0 - zi_prob;  // d/d(logit_theta) log(theta) = 1-theta
            grad_phi_num_i = ratiod::math::portable_digamma(y_num_i + phi_num) - ratiod::math::portable_digamma(phi_num)
                             + std::log(phi_num / denom_num)
                             + (mu_num - y_num_i) / denom_num;
            // Truncation correction for phi gradient
            grad_phi_num_i += p0_nb * (std::log(phi_num / (phi_num + mu_num)) + mu_num / (phi_num + mu_num)) / (1.0 - p0_nb);
          }
        } else {
          // Standard NegBin (no ZI)
          double denom_num = mu_num + phi_num;
          resid_num = y_num_i - mu_num * (y_num_i + phi_num) / denom_num;
          grad_phi_num_i = ratiod::math::portable_digamma(y_num_i + phi_num) - ratiod::math::portable_digamma(phi_num)
                           + std::log(phi_num / denom_num)
                           + (mu_num - y_num_i) / denom_num;
        }

      } else if (data.model_type == ModelType::POISSON_GAMMA) {
        // ---- POISSON_GAMMA ----
        double mu_num = std::exp(eta_num);
        double mu_denom = std::exp(eta_denom);
        int y_num_i = data.y_num[i];

        // Denominator: Gamma (always standard) — skip if y <= 0
        double y_denom_i = data.y_denom_cont[i];
        double grad_phi_gamma = 0.0;
        if (y_denom_i > 0.0) {
          resid_denom = phi_denom * (y_denom_i / mu_denom - 1.0);
          double rate = phi_denom / mu_denom;
          grad_phi_gamma = std::log(rate) + 1.0 + std::log(y_denom_i)
                                  - ratiod::math::portable_digamma(phi_denom) - rate * y_denom_i / phi_denom;
        }

        // Numerator with ZI handling
        if (layout.has_zi && data.zi_type == ratiod_zi::ZIType::ZI_POISSON) {
          // ZI-Poisson numerator
          double exp_neg_mu = std::exp(-mu_num);

          if (y_num_i == 0) {
            // P(Y=0) = zi + (1-zi)*exp(-mu)
            double p0 = zi_prob + (1.0 - zi_prob) * exp_neg_mu;
            // d(LL)/d(eta) = d(LL)/d(mu) * mu = -(1-zi)*exp(-mu)*mu / p0
            resid_num = -(1.0 - zi_prob) * exp_neg_mu * mu_num / p0;
            grad_logit_zi_i = zi_prob * (1.0 - zi_prob) * (1.0 - exp_neg_mu) / p0;
            grad_phi_denom_i = grad_phi_gamma;  // Only gamma part
          } else {
            // P(Y=y) = (1-zi) * Poisson(y|mu)
            resid_num = y_num_i - mu_num;
            grad_logit_zi_i = -zi_prob;
            grad_phi_denom_i = grad_phi_gamma;
          }
        } else if (layout.has_zi && data.zi_type == ratiod_zi::ZIType::HURDLE_POISSON) {
          // Hurdle-Poisson numerator
          if (y_num_i == 0) {
            resid_num = 0.0;
            grad_logit_zi_i = -zi_prob;  // zi_prob is theta here
            grad_phi_denom_i = grad_phi_gamma;
          } else {
            // Truncated Poisson: d(LL)/d(eta) = y - mu - mu*exp(-mu)/(1-exp(-mu))
            // LL = log Poi(y) - log(1-exp(-mu)), correction is SUBTRACTED
            double exp_neg_mu = std::exp(-mu_num);
            resid_num = y_num_i - mu_num - mu_num * exp_neg_mu / (1.0 - exp_neg_mu);
            grad_logit_zi_i = 1.0 - zi_prob;
            grad_phi_denom_i = grad_phi_gamma;
          }
        } else {
          // Standard Poisson (no ZI)
          resid_num = y_num_i - mu_num;
          grad_phi_denom_i = grad_phi_gamma;
        }

      } else if (data.model_type == ModelType::NEGBIN_GAMMA) {
        // ---- NEGBIN_GAMMA ----
        // NegBin numerator + Gamma denominator
        double mu_num = std::exp(eta_num);
        double mu_denom = std::exp(eta_denom);
        int y_num_i = data.y_num[i];

        // Denominator: Gamma (always standard) — skip if y <= 0
        double y_denom_i = data.y_denom_cont[i];
        double grad_phi_gamma = 0.0;
        if (y_denom_i > 0.0) {
          resid_denom = phi_denom * (y_denom_i / mu_denom - 1.0);
          double rate = phi_denom / mu_denom;
          grad_phi_gamma = std::log(rate) + 1.0 + std::log(y_denom_i)
                                  - ratiod::math::portable_digamma(phi_denom) - rate * y_denom_i / phi_denom;
        }

        // Numerator with ZI handling (same as NEGBIN_NEGBIN)
        if (layout.has_zi && data.zi_type == ratiod_zi::ZIType::ZI_NEGBIN) {
          // ZI-NegBin numerator (same logic as NEGBIN_NEGBIN ZI case)
          double p0_nb = std::pow(phi_num / (phi_num + mu_num), phi_num);

          if (y_num_i == 0) {
            double p0 = zi_prob + (1.0 - zi_prob) * p0_nb;
            // d(p0_nb)/d(mu) = -phi * p0_nb / (phi+mu)
            double d_p0_nb_d_mu = -phi_num * p0_nb / (phi_num + mu_num);
            resid_num = (1.0 - zi_prob) * d_p0_nb_d_mu * mu_num / p0;
            grad_logit_zi_i = zi_prob * (1.0 - zi_prob) * (1.0 - p0_nb) / p0;
            grad_phi_num_i = (1.0 - zi_prob) * p0_nb * (std::log(phi_num / (phi_num + mu_num)) + mu_num / (phi_num + mu_num)) / p0;
          } else {
            double denom_num = mu_num + phi_num;
            resid_num = y_num_i - mu_num * (y_num_i + phi_num) / denom_num;
            grad_logit_zi_i = -zi_prob;
            grad_phi_num_i = ratiod::math::portable_digamma(y_num_i + phi_num) - ratiod::math::portable_digamma(phi_num)
                             + std::log(phi_num / denom_num)
                             + (mu_num - y_num_i) / denom_num;
          }
          grad_phi_denom_i = grad_phi_gamma;
        } else if (layout.has_zi && data.zi_type == ratiod_zi::ZIType::HURDLE_NEGBIN) {
          // Hurdle-NegBin numerator (same as NEGBIN_NEGBIN hurdle)
          if (y_num_i == 0) {
            resid_num = 0.0;
            grad_logit_zi_i = -zi_prob;  // d/d(logit_theta) log(1-theta)
            grad_phi_num_i = 0.0;
          } else {
            double p0_nb = std::pow(phi_num / (phi_num + mu_num), phi_num);
            double denom_num = mu_num + phi_num;
            resid_num = y_num_i - mu_num * (y_num_i + phi_num) / denom_num
                        - phi_num * p0_nb * mu_num / ((phi_num + mu_num) * (1.0 - p0_nb));
            grad_logit_zi_i = 1.0 - zi_prob;  // d/d(logit_theta) log(theta)
            grad_phi_num_i = ratiod::math::portable_digamma(y_num_i + phi_num) - ratiod::math::portable_digamma(phi_num)
                             + std::log(phi_num / denom_num)
                             + (mu_num - y_num_i) / denom_num;
            // Truncation correction for phi gradient
            grad_phi_num_i += p0_nb * (std::log(phi_num / (phi_num + mu_num)) + mu_num / (phi_num + mu_num)) / (1.0 - p0_nb);
          }
          grad_phi_denom_i = grad_phi_gamma;
        } else {
          // Standard NegBin numerator (no ZI)
          double denom_num = mu_num + phi_num;
          resid_num = y_num_i - mu_num * (y_num_i + phi_num) / denom_num;
          grad_phi_num_i = ratiod::math::portable_digamma(y_num_i + phi_num) - ratiod::math::portable_digamma(phi_num)
                           + std::log(phi_num / denom_num)
                           + (mu_num - y_num_i) / denom_num;
          grad_phi_denom_i = grad_phi_gamma;
        }

      } else if (data.model_type == ModelType::GAMMA_GAMMA) {
        // ---- GAMMA_GAMMA ----
        // Gamma requires y > 0; skip contributions for y <= 0 (matches log_lik_gamma)
        double mu_num = std::exp(eta_num);
        double mu_denom = std::exp(eta_denom);
        double y_num_i = data.y_num_cont[i];
        double y_denom_i = data.y_denom_cont[i];

        if (y_num_i > 0.0) {
          resid_num = phi_num * (y_num_i / mu_num - 1.0);
          double rate_num = phi_num / mu_num;
          grad_phi_num_i = std::log(rate_num) + 1.0 + std::log(y_num_i)
                           - ratiod::math::portable_digamma(phi_num) - y_num_i / mu_num;
        }
        if (y_denom_i > 0.0) {
          resid_denom = phi_denom * (y_denom_i / mu_denom - 1.0);
          double rate_denom = phi_denom / mu_denom;
          grad_phi_denom_i = std::log(rate_denom) + 1.0 + std::log(y_denom_i)
                             - ratiod::math::portable_digamma(phi_denom) - y_denom_i / mu_denom;
        }

      } else if (data.model_type == ModelType::LOGNORMAL) {
        // ---- LOGNORMAL ----
        // Both numerator and denominator are Lognormal distributed
        // log(y) ~ Normal(mu, sigma^2), so y ~ Lognormal(mu, sigma^2)
        // LL = -log(y) - log(sigma) - 0.5*((log(y) - mu)/sigma)^2
        // d(LL)/d(eta) = d(LL)/d(mu) = (log(y) - mu) / sigma^2
        // (Note: eta IS mu for lognormal, so no chain rule needed)
        double mu_num = eta_num;  // mu is directly the linear predictor
        double mu_denom = eta_denom;
        double y_num_i = data.y_num_cont[i];
        double y_denom_i = data.y_denom_cont[i];
        double log_y_num = std::log(y_num_i);
        double log_y_denom = std::log(y_denom_i);

        // phi_num, phi_denom are sigma (std dev on log scale)
        double sigma_num = phi_num;
        double sigma_denom = phi_denom;
        double sigma_num_sq = sigma_num * sigma_num;
        double sigma_denom_sq = sigma_denom * sigma_denom;

        // Residuals (gradient w.r.t. mu)
        resid_num = (log_y_num - mu_num) / sigma_num_sq;
        resid_denom = (log_y_denom - mu_denom) / sigma_denom_sq;

        // Sigma gradients: d(LL)/d(sigma), NOT d(LL)/d(log_sigma)
        // Because the accumulation code multiplies by phi to get d(LL)/d(log_phi)
        // d(LL)/d(sigma) = -1/sigma + z^2/sigma = (-1 + z^2) / sigma
        double z_num = (log_y_num - mu_num) / sigma_num;
        double z_denom = (log_y_denom - mu_denom) / sigma_denom;
        grad_phi_num_i = (-1.0 + z_num * z_num) / sigma_num;
        grad_phi_denom_i = (-1.0 + z_denom * z_denom) / sigma_denom;

      } else if (data.model_type == ModelType::BETA_BINOMIAL) {
        // ---- BETA_BINOMIAL ----
        // Overdispersed binomial: y ~ BetaBinom(n, alpha, beta)
        // where p = alpha/(alpha+beta), phi = alpha + beta (concentration)
        // We parameterize: logit(p) = eta, phi = overdispersion
        // alpha = p * phi, beta_param = (1-p) * phi
        // LL = lgamma(y+alpha) + lgamma(n-y+beta) - lgamma(n+phi)
        //      - lgamma(alpha) - lgamma(beta) + lgamma(phi) + lchoose(n,y)
        double p = 1.0 / (1.0 + std::exp(-eta_num));
        int y_i = data.y_num[i];
        int n_i = data.y_denom[i];
        double alpha = p * phi_num;
        double beta_param = (1.0 - p) * phi_num;

        // d(LL)/d(eta) = d(LL)/d(p) * d(p)/d(eta) where d(p)/d(eta) = p*(1-p)
        // d(LL)/d(p) = phi * (digamma(y+alpha) - digamma(n-y+beta) - digamma(alpha) + digamma(beta))
        double psi_y_alpha = ratiod::math::portable_digamma(y_i + alpha);
        double psi_nmy_beta = ratiod::math::portable_digamma(n_i - y_i + beta_param);
        double psi_alpha = ratiod::math::portable_digamma(alpha);
        double psi_beta = ratiod::math::portable_digamma(beta_param);
        double dLL_dp = phi_num * (psi_y_alpha - psi_nmy_beta - psi_alpha + psi_beta);
        resid_num = dLL_dp * p * (1.0 - p);

        // d(LL)/d(phi) where phi = alpha + beta
        // d(LL)/d(phi) = p*digamma(y+alpha) + (1-p)*digamma(n-y+beta) - digamma(n+phi)
        //               - p*digamma(alpha) - (1-p)*digamma(beta) + digamma(phi)
        double psi_n_phi = ratiod::math::portable_digamma(n_i + phi_num);
        double psi_phi = ratiod::math::portable_digamma(phi_num);
        grad_phi_num_i = p * psi_y_alpha + (1.0 - p) * psi_nmy_beta - psi_n_phi
                         - p * psi_alpha - (1.0 - p) * psi_beta + psi_phi;

        // No denominator contribution for beta-binomial (like binomial)
      }

      // Accumulate ZI coefficient gradients
      if (acc_zi) {
        for (int j = 0; j < data.p_zi; j++) {
          acc[acc_at.beta_zi + j] +=
              data.X_zi_flat[i * data.p_zi + j] * grad_logit_zi_i;
        }
      }

      // Accumulate OI coefficient gradients
      if (acc_oi) {
        for (int j = 0; j < data.p_oi; j++) {
          acc[acc_at.beta_oi + j] +=
              data.X_oi_flat[i * data.p_oi + j] * grad_logit_oi_i;
        }
      }

      // Accumulate beta gradients: grad += X[i,:] * resid
      for (int j = 0; j < data.p_num; j++) {
        acc[acc_at.beta_num + j] += data.X_num_flat[i * data.p_num + j] * resid_num;
      }
      // For BINOMIAL, beta_denom doesn't affect likelihood
      if (data.model_type != ModelType::BINOMIAL) {
        for (int j = 0; j < data.p_denom; j++) {
          acc[acc_at.beta_denom + j] +=
              data.X_denom_flat[i * data.p_denom + j] * resid_denom;
        }
      }

      // Accumulate RE gradient
      if (layout.has_re_slopes && re_group_idx > 0) {
        // Random slopes case: gradient for intercept and each slope
        double re_grad_base = resid_num;
        if (data.model_type != ModelType::BINOMIAL) {
          re_grad_base += resid_denom;
        }
        int g = re_group_idx - 1;
        int n_coefs = re_n_coefs_i;

        // Intercept gradient: same as simple RE
        acc[acc_at.re_slopes + g * n_coefs] += re_grad_base;

        // Slope gradients: multiply by slope design value
        int n_slopes = n_coefs - 1;
        for (int s = 0; s < n_slopes; s++) {
          acc[acc_at.re_slopes + g * n_coefs + 1 + s] +=
              re_grad_base * re_slope_x_i[s];
        }
      } else if (n_crossed_terms > 0) {
        // Crossed RE (multiple intercept-only terms)
        double re_grad_i = resid_num;
        if (data.model_type != ModelType::BINOMIAL) {
          re_grad_i += resid_denom;
        }
        for (int t = 0; t < n_crossed_terms; t++) {
          if (re_idx_multi_buf[t] >= 0) {
            acc[acc_at.re_crossed + data.re_offsets[t] + re_idx_multi_buf[t]] +=
                re_grad_i;
          }
        }
      } else if (re_idx >= 0) {
        // Simple intercept-only RE (single term)
        double re_grad_i = resid_num;
        if (data.model_type != ModelType::BINOMIAL) {
          re_grad_i += resid_denom;  // Shared RE affects both processes
        }
        acc[acc_at.re + re_idx] += re_grad_i;
      }

      // Accumulate temporal gradient (from likelihood)
      if (t_idx >= 0) {
        double temp_grad_i = resid_num;
        if (data.model_type != ModelType::BINOMIAL) {
          temp_grad_i += resid_denom;
        }
        acc[acc_at.temporal + t_idx] += temp_grad_i;
      }

      // Accumulate spatial gradient (from likelihood)
      if (s_idx >= 0) {
        double lik_grad = resid_num;
        if (data.model_type != ModelType::BINOMIAL) {
          lik_grad += resid_denom;
        }
        // For ICAR: grad_spatial[s] += lik_grad (d_spatial_d_phi = 1)
        // For BYM2: grad_phi[s] += lik_grad * d_spatial_d_phi, grad_theta[s] += lik_grad * d_spatial_d_theta
        acc[acc_at.spatial + s_idx] += lik_grad * d_spatial_d_phi;

        if (acc_bym2) {
          acc[acc_at.theta_bym2 + s_idx] += lik_grad * d_spatial_d_theta;
        }
      }

      // Accumulate phi gradients
      acc[acc_at.phi_num] += grad_phi_num_i;
      acc[acc_at.phi_denom] += grad_phi_denom_i;

      // Fused log-likelihood: compute per-observation log_lik using already-computed
      // intermediates. This avoids a separate O(N) pass through compute_log_post.
      if (compute_lp) {
        double ll_i = 0.0;
        if (data.model_type == ModelType::BINOMIAL) {
          double p_i = 1.0 / (1.0 + std::exp(-eta_num));
          int n_trials = data.y_denom[i];
          int y_i = data.y_num[i];
          if (data.zi_type == ratiod_zi::ZIType::ZI_BINOMIAL) {
            ll_i = ratiod_zi::zi_binomial_lpmf_logit(y_i, n_trials, p_i, logit_zi);
          } else if (data.zi_type == ratiod_zi::ZIType::HURDLE_BINOMIAL) {
            ll_i = ratiod_zi::hurdle_binomial_lpmf_logit(y_i, n_trials, p_i, logit_zi);
          } else if (data.zi_type == ratiod_zi::ZIType::OI_BINOMIAL) {
            ll_i = ratiod_zi::oi_binomial_lpmf_logit(y_i, n_trials, p_i, logit_oi);
          } else if (data.zi_type == ratiod_zi::ZIType::ZOIB) {
            ll_i = ratiod_zi::zoib_lpmf_logit(y_i, n_trials, p_i, logit_zi, logit_oi);
          } else {
            ll_i = log_lik_binomial(y_i, n_trials, eta_num);
          }
        } else if (data.model_type == ModelType::NEGBIN_NEGBIN) {
          double mu_num_i = std::exp(eta_num);
          double mu_denom_i = std::exp(eta_denom);
          if (layout.has_zi) {
            ll_i = ratiod_zi::zi_log_likelihood(data.y_num[i], mu_num_i, phi_num,
                                               logit_zi, data.zi_type);
          } else {
            ll_i = log_lik_negbin(data.y_num[i], mu_num_i, phi_num);
          }
          ll_i += log_lik_negbin(data.y_denom[i], mu_denom_i, phi_denom);
        } else if (data.model_type == ModelType::POISSON_GAMMA) {
          double mu_num_i = std::exp(eta_num);
          double mu_denom_i = std::exp(eta_denom);
          if (layout.has_zi) {
            ll_i = ratiod_zi::zi_log_likelihood(data.y_num[i], mu_num_i, phi_num,
                                               logit_zi, data.zi_type);
          } else {
            ll_i = log_lik_poisson(data.y_num[i], mu_num_i);
          }
          ll_i += log_lik_gamma(data.y_denom_cont[i], phi_denom, mu_denom_i);
        } else if (data.model_type == ModelType::NEGBIN_GAMMA) {
          double mu_num_i = std::exp(eta_num);
          double mu_denom_i = std::exp(eta_denom);
          if (layout.has_zi) {
            ll_i = ratiod_zi::zi_log_likelihood(data.y_num[i], mu_num_i, phi_num,
                                               logit_zi, data.zi_type);
          } else {
            ll_i = log_lik_negbin(data.y_num[i], mu_num_i, phi_num);
          }
          ll_i += log_lik_gamma(data.y_denom_cont[i], phi_denom, mu_denom_i);
        } else if (data.model_type == ModelType::GAMMA_GAMMA) {
          double mu_num_i = std::exp(eta_num);
          double mu_denom_i = std::exp(eta_denom);
          ll_i = log_lik_gamma(data.y_num_cont[i], phi_num, mu_num_i);
          ll_i += log_lik_gamma(data.y_denom_cont[i], phi_denom, mu_denom_i);
        } else if (data.model_type == ModelType::LOGNORMAL) {
          double log_y_num_i = std::log(data.y_num_cont[i]);
          double log_y_denom_i = std::log(data.y_denom_cont[i]);
          double z_num_i = (log_y_num_i - eta_num) / phi_num;
          double z_denom_i = (log_y_denom_i - eta_denom) / phi_denom;
          ll_i = -log_y_num_i - std::log(phi_num) - 0.5 * z_num_i * z_num_i;
          ll_i += -log_y_denom_i - std::log(phi_denom) - 0.5 * z_denom_i * z_denom_i;
        } else if (data.model_type == ModelType::BETA_BINOMIAL) {
          double p_i = 1.0 / (1.0 + std::exp(-eta_num));
          int y_i = data.y_num[i];
          int n_i = data.y_denom[i];
          double alpha_i = p_i * phi_num;
          double beta_i = (1.0 - p_i) * phi_num;
          ll_i = std::lgamma(y_i + alpha_i) + std::lgamma(n_i - y_i + beta_i) - std::lgamma(n_i + phi_num);
          ll_i += -std::lgamma(alpha_i) - std::lgamma(beta_i) + std::lgamma(phi_num);
          ll_i += ratiod::math::portable_lchoose(n_i, y_i);
        }
        acc[acc_at.ll] += ll_i;
      }
    }
  }  // end parallel

  // Deterministic reduction: thread-index order, fixed team size.
  for (int th = 0; th < grad_team; th++) {
    const double* slot = &red_partials[(size_t)th * acc_at.stride];
    for (int j = 0; j < data.p_num; j++) grad_beta_num[j] += slot[acc_at.beta_num + j];
    for (int j = 0; j < data.p_denom; j++) grad_beta_denom[j] += slot[acc_at.beta_denom + j];
    grad_phi_num_lik += slot[acc_at.phi_num];
    grad_phi_denom_lik += slot[acc_at.phi_denom];
    for (int g = 0; g < n_re_acc; g++) {
      grad[layout.re_start + g] += slot[acc_at.re + g];
    }
    // Crossed intercept-only RE (n_re_terms > 1, not slopes) scatter to
    // non-contiguous grad positions.
    if (n_re_crossed_acc > 0) {
      for (int t = 0; t < (int)data.re_n_groups_multi.size(); t++) {
        int re_start_t = layout.re_start_multi[t];
        int offset_t = data.re_offsets[t];
        for (int g = 0; g < data.re_n_groups_multi[t]; g++) {
          grad[re_start_t + g] += slot[acc_at.re_crossed + offset_t + g];
        }
      }
    }
    obs_log_lik += slot[acc_at.ll];
    if (acc_zi) {
      for (int j = 0; j < data.p_zi; j++) grad_beta_zi[j] += slot[acc_at.beta_zi + j];
    }
    if (acc_oi) {
      for (int j = 0; j < data.p_oi; j++) grad_beta_oi[j] += slot[acc_at.beta_oi + j];
    }
    for (int k = 0; k < n_slopes_acc; k++) {
      grad_re_slopes_lik[0][k] += slot[acc_at.re_slopes + k];
    }
    for (int t = 0; t < (int)grad_temporal_lik.size(); t++) {
      grad_temporal_lik[t] += slot[acc_at.temporal + t];
    }
    for (int s = 0; s < (int)grad_spatial_lik.size(); s++) {
      grad_spatial_lik[s] += slot[acc_at.spatial + s];
    }
    if (acc_bym2) {
      for (int s = 0; s < n_spatial; s++) {
        grad[layout.theta_bym2_start + s] += slot[acc_at.theta_bym2 + s];
      }
    }
  }

  // Add likelihood gradients to total
  for (int j = 0; j < data.p_num; j++) {
    grad[layout.beta_num_start + j] += grad_beta_num[j];
  }
  for (int j = 0; j < data.p_denom; j++) {
    grad[layout.beta_denom_start + j] += grad_beta_denom[j];
  }

  // Phi gradients (with Jacobian for log transform)
  if (layout.has_phi_num) {
    grad[layout.log_phi_num_idx] += phi_num * grad_phi_num_lik;
  }
  if (layout.has_phi_denom) {
    grad[layout.log_phi_denom_idx] += phi_denom * grad_phi_denom_lik;
  }

  // ZI coefficient gradients (likelihood contribution)
  if (layout.has_zi && data.p_zi > 0) {
    for (int j = 0; j < data.p_zi; j++) {
      grad[layout.beta_zi_start + j] += grad_beta_zi[j];
    }
  }

  // OI coefficient gradients (likelihood contribution)
  if (layout.has_oi && data.p_oi > 0) {
    for (int j = 0; j < data.p_oi; j++) {
      grad[layout.beta_oi_start + j] += grad_beta_oi[j];
    }
  }

  } // end if (!used_vectorized)

  // The chain rule back onto the sampled parameters, for every RE term and
  // both parameterizations. Outside the used_vectorized check: the hybrid
  // slopes path sets used_vectorized = true and still fills the block's
  // likelihood accumulator through its own scatter pass.
  re_blocks_writeback(params, data, layout, grad, re_ws);

  // ============ Temporal GMRF prior gradients ============
  if (layout.has_temporal && T_len > 0) {
    temporal_gmrf_prior_grad(data, layout, tau_temporal, rho_ar1,
                             tview, T_len, grad_temporal_lik.data(), grad.data());
  }

  // ============ Spatial GMRF prior gradients (ICAR and BYM2) ============
  if (layout.has_spatial && n_spatial > 0) {
    spatial_gmrf_prior_grad(data, layout, phi_spatial, phi_spatial, phi_raw_sum,
                            center_spatial,
                            tau_spatial, sigma_s_bym2, sigma_u_bym2, rho_bym2, 0.0,
                            theta_bym2, grad_spatial_lik.data(), grad.data());
  }

  // Fused log-posterior output: combine prior/structural terms with observation log-lik.
  // Prior/structural terms are computed via compute_log_post with skip_obs_loop=true (O(p+S+T)).
  // Observation log-lik was accumulated inline during the gradient computation (O(N)).
  // Total: one O(N) pass instead of two.
  // compute_log_post centres its own copy and recovers phi_raw_sum from the raw
  // field, so it takes params_in; the centred copy would zero that sum and drop
  // the freed constant direction from the prior.
  if (log_post_out) {
    *log_post_out = compute_log_post(params_in, data, layout, /*skip_obs_loop=*/true) + obs_log_lik;
  }

}

// Gradient mode: controls which gradient function is used.
// Moved here (before verify_gradient_runtime) so mode-aware functions
// can reference it. Previously defined later in the file.
static GradientMode g_gradient_mode = GradientMode::AUTO;

// =====================================================================
// Numerical gradient (fallback for complex models)
// =====================================================================

void compute_gradient_numerical(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    std::vector<double>& grad,
    double* log_post_out = nullptr
) {
  int n = params.size();
  grad.resize(n);

  // Compute log_post at central point if requested (cheap: one extra eval)
  if (log_post_out) {
    *log_post_out = compute_log_post(params, data, layout);
  }

  double h = 1e-5;

  for (int i = 0; i < n; i++) {
    std::vector<double> params_plus = params;
    std::vector<double> params_minus = params;

    params_plus[i] = params[i] + h;
    params_minus[i] = params[i] - h;

    double f_plus = compute_log_post(params_plus, data, layout);
    double f_minus = compute_log_post(params_minus, data, layout);

    grad[i] = (f_plus - f_minus) / (2.0 * h);
  }
}

// =====================================================================
// Numerical gradient using compute_log_post_impl<double>
// Used for verifying A_r/A modes which use compute_log_post_impl<T>.
// This ensures the numerical reference matches the same function that
// the autodiff mode differentiates (important when parameterization
// differs from H-mode, e.g. non-centered RE).
// =====================================================================

void compute_gradient_numerical_impl(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    std::vector<double>& grad,
    double* log_post_out = nullptr
) {
  int n = params.size();
  grad.resize(n);

  if (log_post_out) {
    *log_post_out = ratiod::compute_log_post_impl(params, data, layout);
  }

  double h = 1e-5;

  for (int i = 0; i < n; i++) {
    std::vector<double> params_plus = params;
    std::vector<double> params_minus = params;

    params_plus[i] = params[i] + h;
    params_minus[i] = params[i] - h;

    double f_plus = ratiod::compute_log_post_impl(params_plus, data, layout);
    double f_minus = ratiod::compute_log_post_impl(params_minus, data, layout);

    grad[i] = (f_plus - f_minus) / (2.0 * h);
  }
}

// =====================================================================
// Unified gradient interface
// =====================================================================

// Debug: compare analytical vs numerical gradients
bool verify_gradient(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    double tol = 1e-4
) {
  std::vector<double> grad_analytical, grad_numerical;
  compute_gradient_analytical(params, data, layout, grad_analytical);
  compute_gradient_numerical(params, data, layout, grad_numerical);

  double max_diff = 0.0;
  int worst_idx = -1;
  for (size_t i = 0; i < grad_analytical.size(); i++) {
    double diff = std::abs(grad_analytical[i] - grad_numerical[i]);
    double scale = std::max(1.0, std::max(std::abs(grad_analytical[i]), std::abs(grad_numerical[i])));
    double rel_diff = diff / scale;
    if (rel_diff > max_diff) {
      max_diff = rel_diff;
      worst_idx = i;
    }
  }

  if (max_diff > tol) {
    Rcpp::Rcerr << "Gradient mismatch! Max rel diff: " << max_diff
                << " at param " << worst_idx
                << " (analytical: " << grad_analytical[worst_idx]
                << ", numerical: " << grad_numerical[worst_idx] << ")\n";
    return false;
  }
  return true;
}

// Move every structured field off zero, deterministically.
//
// initialize_hmc_params_full() starts each field block at exactly 0.0, and a
// quadratic-form prior -tau/2 phi' Q phi contributes nothing to any gradient
// there: a wrong Q, a sign error on it, a missing quadratic term and an absent
// sum-to-zero penalty are all invisible, because the analytic and the numerical
// gradient agree on the same zero. The offline harness evaluates away from the
// origin for exactly this reason (src/test_gradient_check.cpp). The offset is
// small, bounded and a function of the index alone, so the check point is
// reproducible and stays inside any prior's support.
std::vector<double> offset_structured_blocks(const std::vector<double>& q,
                                             const ParamLayout& layout) {
  std::vector<double> out = q;
  auto jitter = [&](int start, int end) {
    if (start < 0 || end <= start) return;
    for (int i = start; i < end && i < static_cast<int>(out.size()); i++) {
      out[i] += 0.1 * std::sin(1.7 * static_cast<double>(i) + 0.3);
    }
  };

  jitter(layout.spatial_start, layout.spatial_end);
  jitter(layout.theta_bym2_start, layout.theta_bym2_end);
  jitter(layout.temporal_start, layout.temporal_end);
  jitter(layout.hsgp_beta_start, layout.hsgp_beta_end);
  jitter(layout.tvc_w_start, layout.tvc_w_end);
  jitter(layout.svc_w_start, layout.svc_w_end);
  jitter(layout.gp_w_start, layout.gp_w_end);
  jitter(layout.gp_local_start, layout.gp_local_end);
  jitter(layout.gp_regional_start, layout.gp_regional_end);
  jitter(layout.trend_start, layout.trend_end);
  jitter(layout.seasonal_start, layout.seasonal_end);
  jitter(layout.short_term_start, layout.short_term_end);
  jitter(layout.st_delta_start, layout.st_delta_end);
  jitter(layout.latent_factor_start, layout.latent_factor_end);
  for (size_t t = 0; t < layout.re_start_multi.size(); t++) {
    jitter(layout.re_start_multi[t], layout.re_end_multi[t]);
  }
  return out;
}

// Runtime gradient check: compare compute_gradient() dispatcher output against
// numerical gradients of the density it is supposed to be the gradient of,
// once, before sampling starts. Covers every specialized gradient function the
// dispatcher can select (GP, HSGP, SVC, TVC, MSGP, spatiotemporal, ...), at a
// point where the structured fields are nonzero.
bool verify_gradient_runtime(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    double tol = 1e-4
) {
  std::vector<double> grad_active, grad_numerical;
  compute_gradient(params, data, layout, grad_active, nullptr);

  // For A_r/A modes: use compute_log_post_impl<double> as numerical reference,
  // since those modes differentiate compute_log_post_impl<T> (which may use
  // different parameterization than H-mode's compute_log_post, e.g. non-centered RE).
  // For H mode: use compute_log_post (H-mode function) as reference.
  if (g_gradient_mode == GradientMode::AUTODIFF_ARENA ||
      g_gradient_mode == GradientMode::AUTODIFF_FORWARD ||
      g_gradient_mode == GradientMode::AUTODIFF_TAPE) {
    compute_gradient_numerical_impl(params, data, layout, grad_numerical);
  } else {
    compute_gradient_numerical(params, data, layout, grad_numerical);
  }

  double max_diff = 0.0;
  int worst_idx = -1;
  for (size_t i = 0; i < grad_active.size(); i++) {
    double diff = std::abs(grad_active[i] - grad_numerical[i]);
    double scale = std::max(1.0, std::max(std::abs(grad_active[i]),
                                           std::abs(grad_numerical[i])));
    double rel_diff = diff / scale;
    if (rel_diff > max_diff) {
      max_diff = rel_diff;
      worst_idx = i;
    }
  }

  if (max_diff > tol) {
    // Use REprintf for immediate output, then Rcpp::warning for R-level notice
    REprintf("[tulpaRatio] WARNING: gradient mismatch detected at param %d!\n"
             "  max |active - numerical| / scale = %.6e (tol = %.1e)\n"
             "  active[%d] = %.8e, numerical[%d] = %.8e\n"
             "  This indicates a bug in the specialized gradient function.\n"
             "  Falling back to numerical gradients for safety.\n",
             worst_idx, max_diff, tol,
             worst_idx, grad_active[worst_idx],
             worst_idx, grad_numerical[worst_idx]);
    return false;
  }
  return true;
}

// Run the runtime check and fall back to numerical gradients if it fails.
// Both the single-chain and the multi-chain entry points call this, so a
// mismatch is reported the same way at either -- an R-level warning, which is
// what an expect_warning(..., NA) guard in the tests can see. Both call sites
// are on the R thread, before any parallel region opens.
void verify_gradient_or_fallback(const std::vector<double>& q_init,
                                 const ModelData& data,
                                 const ParamLayout& layout) {
  if (g_gradient_mode == GradientMode::NUMERICAL) return;

  const std::vector<double> probe = offset_structured_blocks(q_init, layout);
  if (verify_gradient_runtime(probe, data, layout, 1e-4)) return;

  g_gradient_mode = GradientMode::NUMERICAL;
  Rcpp::warning(
    "Gradient mismatch detected: active gradient function disagrees with "
    "numerical gradients (max rel diff > 1e-4). Falling back to numerical "
    "gradients (mode='N'). This is slower but correct. Please report this "
    "as a bug at https://github.com/gcol33/tulpaRatio/issues"
  );
}

// =====================================================================
// Autodiff gradient (O(n) - works for ALL models)
// =====================================================================

void compute_gradient_autodiff(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    std::vector<double>& grad,
    double* log_post_out = nullptr
) {
    using namespace ratiod::ad;

    // Thread-safe: each call gets its own tape via RAII
    TapeScope tape_scope;
    Tape* tape = tape_scope.tape;

    // Create autodiff variables from parameters
    std::vector<Var> params_ad = make_vars(tape, params);

    // Compute log posterior using templated implementation
    Var log_post = ratiod::compute_log_post_impl(params_ad, data, layout);

    // Extract log_post value before backward pass (free: already computed)
    if (log_post_out) *log_post_out = log_post.val();

    // Backward pass to compute gradients
    log_post.backward();

    // Extract gradients
    grad = get_adjoints(params_ad);

    // TapeScope destructor handles cleanup
}

// =====================================================================
// Arena-based reverse-mode autodiff gradient (O(N) - fast, all models)
// Uses contiguous SoA memory layout with pre-computed partials.
// ~10-30x faster than tape autodiff, within 50% of hand-coded speed.
// =====================================================================

void compute_gradient_arena(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    std::vector<double>& grad,
    double* log_post_out = nullptr
) {
    using namespace ratiod::arena;

    int n_nodes_used = 0;
    {
        // Thread-safe: each call gets its own arena via RAII
        ArenaScope scope;
        Arena* arena = scope.arena();

        // Create autodiff variables from parameters
        std::vector<Var> params_ar = make_vars(arena, params);

        // Compute log posterior using templated implementation
        Var log_post = ratiod::compute_log_post_impl(params_ar, data, layout);

        // Extract log_post value before backward pass
        if (log_post_out) *log_post_out = log_post.val();

        // Backward pass to compute gradients
        log_post.backward();

        // Extract gradients
        grad = get_adjoints(params_ar);

        n_nodes_used = arena->size();
        // ArenaScope destructor handles cleanup
    }

    (void)n_nodes_used;  // suppress unused warning
}

// =====================================================================
// Forward-mode autodiff gradient (O(n×p) - but ~10x faster than tape)
// Uses dual numbers for efficient gradient computation without heap allocation
// =====================================================================

void compute_gradient_forward(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    std::vector<double>& grad,
    double* log_post_out = nullptr
) {
    int n_params = static_cast<int>(params.size());
    grad.assign(n_params, 0.0);

    // Forward-mode: compute one gradient component per forward pass
    // Seed each parameter in turn and evaluate
    std::vector<fwd::Dual> params_dual(n_params);

    for (int i = 0; i < n_params; i++) {
        // Seed parameter i: value=params[i], gradient=1.0
        // All others: value=params[j], gradient=0.0
        for (int j = 0; j < n_params; j++) {
            params_dual[j].val = params[j];
            params_dual[j].grad = (j == i) ? 1.0 : 0.0;
        }

        // Compute log posterior with dual numbers
        fwd::Dual log_post = ratiod::compute_log_post_impl(params_dual, data, layout);

        // Extract gradient component
        grad[i] = log_post.grad;

        // Extract log_post value on first pass (free: already computed)
        if (i == 0 && log_post_out) *log_post_out = log_post.val;
    }
}

// =====================================================================
// RE gradient helpers for specialized gradient functions
// Handles both centered and non-centered parameterizations correctly
// =====================================================================

// Initialize RE gradient with prior contribution
static inline void re_gradient_prior(
    const ModelData& data,
    const ParamLayout& layout,
    const double* re,   // re[g] = params[re_start + g]
    double* grad,
    double sigma_re
) {
    if (!layout.has_re || data.n_re_groups <= 0) return;

    // Half-Cauchy prior on sigma_re (log-scale)
    double ratio = sigma_re / data.sigma_re_scale;
    double ratio_sq = ratio * ratio;
    grad[layout.log_sigma_re_idx] = -2.0 * ratio_sq / (1.0 + ratio_sq) + 1.0;

    if (data.re_parameterization == 1) {
        // Non-centered: z ~ N(0,1), prior grad = -z
        for (int g = 0; g < data.n_re_groups; g++) {
            grad[layout.re_start + g] = -re[g];
        }
    } else {
        // Centered: re ~ N(0, sigma^2)
        double tau_re = 1.0 / (sigma_re * sigma_re + 1e-10);
        for (int g = 0; g < data.n_re_groups; g++) {
            grad[layout.re_start + g] = -tau_re * re[g];
            grad[layout.log_sigma_re_idx] += tau_re * re[g] * re[g] - 1.0;
        }
    }
}

// Get RE value for observation (handles NC -> sigma*z transformation)
static inline double re_value_for_eta(
    const double* re,
    int g,
    double sigma_re,
    int re_parameterization
) {
    double val = re[g];
    if (re_parameterization == 1) val *= sigma_re;
    return val;
}

// Apply NC chain rule transformation after observation loop
// Must be called AFTER observation loop has accumulated likelihood gradients in grad[re+g]
static inline void re_gradient_nc_transform(
    const ModelData& data,
    const ParamLayout& layout,
    const double* params,
    double* grad,
    double sigma_re
) {
    if (!layout.has_re || data.n_re_groups <= 0 || data.re_parameterization != 1) return;

    double sigma_lik_grad = 0.0;
    for (int g = 0; g < data.n_re_groups; g++) {
        double z_g = params[layout.re_start + g];
        // Extract centered lik grad: total - prior = grad[re+g] - (-z_g) = grad[re+g] + z_g
        double centered_lik = grad[layout.re_start + g] + z_g;
        // z gradient = prior + chain rule through sigma*z
        grad[layout.re_start + g] = -z_g + sigma_re * centered_lik;
        // sigma gradient from likelihood: z_g * d_ll/d_re_g
        sigma_lik_grad += z_g * centered_lik;
    }
    // d_ll/d_log_sigma = sigma * sum(z_g * d_ll/d_re_g)
    grad[layout.log_sigma_re_idx] += sigma_re * sigma_lik_grad;
}

// =====================================================================
// Shared gradient building blocks for specialized H-mode functions
// These helpers extract the duplicated code from 11 specialized gradient
// functions into single-source-of-truth implementations.
// All are static inline — zero overhead, compiler inlines them.
// =====================================================================

// Common parameters extracted from the parameter vector
struct CommonGradParams {
    const double* beta_num;
    const double* beta_denom;
    double sigma_re;
    const double* re;
    double phi_num;
    double phi_denom;
};

// Extract common parameters from the HMC parameter vector
static inline CommonGradParams extract_common_params(
    const std::vector<double>& params,
    const ParamLayout& layout
) {
    CommonGradParams cp;
    cp.beta_num = &params[layout.beta_num_start];
    cp.beta_denom = &params[layout.beta_denom_start];
    cp.sigma_re = layout.has_re ? std::exp(params[layout.log_sigma_re_idx]) : 1.0;
    cp.re = layout.has_re ? &params[layout.re_start] : nullptr;
    cp.phi_num = layout.has_phi_num ? std::exp(params[layout.log_phi_num_idx]) : 1.0;
    cp.phi_denom = layout.has_phi_denom ? std::exp(params[layout.log_phi_denom_idx]) : 1.0;
    return cp;
}

// Beta N(0, sigma_beta^2) prior gradient
// d/d(beta) = -tau_beta * beta where tau_beta = 1/sigma_beta^2
static inline void beta_gradient_prior(
    const ModelData& data, const ParamLayout& layout,
    const double* beta_num, const double* beta_denom,
    double* grad
) {
    double tau_beta = 1.0 / (data.sigma_beta * data.sigma_beta);
    for (int j = 0; j < data.p_num; j++) {
        grad[layout.beta_num_start + j] = -tau_beta * beta_num[j];
    }
    for (int j = 0; j < data.p_denom; j++) {
        grad[layout.beta_denom_start + j] = -tau_beta * beta_denom[j];
    }
}

// Phi Gamma(shape, rate) prior gradient on log-scale
// d/d(log_phi) = shape - rate*phi
// (equivalently: (shape-1) - rate*phi + 1 with Jacobian expanded)
static inline void phi_gradient_prior(
    const ModelData& data, const ParamLayout& layout,
    double phi_num, double phi_denom,
    double* grad
) {
    if (layout.has_phi_num) {
        grad[layout.log_phi_num_idx] = data.phi_prior_shape
                                       - data.phi_prior_rate * phi_num;
    }
    if (layout.has_phi_denom) {
        grad[layout.log_phi_denom_idx] = data.phi_prior_shape
                                         - data.phi_prior_rate * phi_denom;
    }
}

// Per-observation residual computation (dLL/deta for each family)
// Handles all model types: BINOMIAL, NEGBIN_NEGBIN, POISSON_GAMMA,
// NEGBIN_GAMMA, and catch-all (GAMMA_GAMMA, LOGNORMAL, BETA_BINOMIAL)
static inline void compute_obs_residuals(
    const ModelData& data, int i,
    double eta_num, double eta_denom,
    double phi_num, double phi_denom,
    double& dLL_deta_num, double& dLL_deta_denom
) {
    dLL_deta_num = 0.0;
    dLL_deta_denom = 0.0;

    if (data.model_type == ModelType::BINOMIAL) {
        double p = 1.0 / (1.0 + std::exp(-eta_num));
        dLL_deta_num = data.y_num[i] - data.y_denom[i] * p;
    } else if (data.model_type == ModelType::NEGBIN_NEGBIN) {
        double mu_num = std::exp(eta_num);
        double mu_denom = std::exp(eta_denom);
        dLL_deta_num = data.y_num[i] - mu_num * (data.y_num[i] + phi_num) / (mu_num + phi_num);
        dLL_deta_denom = data.y_denom[i] - mu_denom * (data.y_denom[i] + phi_denom) / (mu_denom + phi_denom);
    } else if (data.model_type == ModelType::POISSON_GAMMA) {
        double mu_num = std::exp(eta_num);
        double mu_denom = std::exp(eta_denom);
        dLL_deta_num = data.y_num[i] - mu_num;
        // Gamma requires y > 0; skip if y_denom_cont <= 0 (matches log_lik_gamma)
        dLL_deta_denom = (data.y_denom_cont[i] > 0.0)
            ? phi_denom * (data.y_denom_cont[i] / mu_denom - 1.0) : 0.0;
    } else if (data.model_type == ModelType::NEGBIN_GAMMA) {
        double mu_num = std::exp(eta_num);
        double mu_denom = std::exp(eta_denom);
        double denom_nb = mu_num + phi_num;
        dLL_deta_num = data.y_num[i] - mu_num * (data.y_num[i] + phi_num) / denom_nb;
        dLL_deta_denom = (data.y_denom_cont[i] > 0.0)
            ? phi_denom * (data.y_denom_cont[i] / mu_denom - 1.0) : 0.0;
    } else {
        // GAMMA_GAMMA, LOGNORMAL, BETA_BINOMIAL catch-all
        double mu_num = std::exp(eta_num);
        double mu_denom = std::exp(eta_denom);
        dLL_deta_num = (data.y_num_cont[i] > 0.0)
            ? phi_num * (data.y_num_cont[i] / mu_num - 1.0) : 0.0;
        dLL_deta_denom = (data.y_denom_cont[i] > 0.0)
            ? phi_denom * (data.y_denom_cont[i] / mu_denom - 1.0) : 0.0;
    }
}

// Scatter residuals to beta gradient slots
static inline void scatter_beta_gradients(
    const ModelData& data, const ParamLayout& layout,
    int i, double dLL_deta_num, double dLL_deta_denom,
    double* grad
) {
    for (int j = 0; j < data.p_num; j++) {
        grad[layout.beta_num_start + j] += dLL_deta_num * data.X_num_flat[i * data.p_num + j];
    }
    for (int j = 0; j < data.p_denom; j++) {
        grad[layout.beta_denom_start + j] += dLL_deta_denom * data.X_denom_flat[i * data.p_denom + j];
    }
}

// Scatter residuals to RE gradient slot
static inline void scatter_re_gradient(
    const ModelData& data, const ParamLayout& layout,
    int i, double dLL_deta_num, double dLL_deta_denom,
    double* grad
) {
    if (layout.has_re && data.re_group[i] > 0) {
        int g = data.re_group[i] - 1;
        grad[layout.re_start + g] += dLL_deta_num + dLL_deta_denom;
    }
}

// Per-observation phi likelihood gradient accumulation
// Handles NB phi_num, NB phi_denom, and Gamma phi_denom
static inline void accumulate_phi_likelihood_grad(
    const ModelData& data, const ParamLayout& layout,
    int i, double eta_num, double eta_denom,
    double phi_num, double phi_denom,
    double* grad
) {
    // phi_num gradient (NB families only)
    if (layout.has_phi_num) {
        if (data.model_type == ModelType::NEGBIN_NEGBIN ||
            data.model_type == ModelType::NEGBIN_GAMMA) {
            double mu_num = std::exp(eta_num);
            double y = data.y_num[i];
            double dLL_dphi = ratiod::math::portable_digamma(y + phi_num) - ratiod::math::portable_digamma(phi_num)
                             + std::log(phi_num / (mu_num + phi_num)) + 1.0
                             - (y + phi_num) / (mu_num + phi_num);
            grad[layout.log_phi_num_idx] += dLL_dphi * phi_num;
        }
    }

    // phi_denom gradient (NB or Gamma families)
    if (layout.has_phi_denom) {
        if (data.model_type == ModelType::NEGBIN_NEGBIN) {
            double mu_denom = std::exp(eta_denom);
            double y = data.y_denom[i];
            double dLL_dphi = ratiod::math::portable_digamma(y + phi_denom) - ratiod::math::portable_digamma(phi_denom)
                             + std::log(phi_denom / (mu_denom + phi_denom)) + 1.0
                             - (y + phi_denom) / (mu_denom + phi_denom);
            grad[layout.log_phi_denom_idx] += dLL_dphi * phi_denom;
        } else if (data.model_type == ModelType::POISSON_GAMMA ||
                   data.model_type == ModelType::NEGBIN_GAMMA) {
            double y = data.y_denom_cont[i];
            if (y > 0.0) {  // Gamma requires y > 0
                double mu_denom = std::exp(eta_denom);
                double rate = phi_denom / mu_denom;
                double dLL_dphi = std::log(rate) + 1.0 + std::log(y)
                                 - ratiod::math::portable_digamma(phi_denom) - y / mu_denom;
                grad[layout.log_phi_denom_idx] += dLL_dphi * phi_denom;
            }
        }
    }
}

// Temporal tau prior gradient on log-scale
// d/d(log_tau) = shape - rate*tau
// (equivalently: (shape-1) - rate*tau + 1 with Jacobian expanded)
static inline void tau_temporal_prior_grad(
    const ModelData& data, const ParamLayout& layout,
    double tau_temporal, double* grad
) {
    grad[layout.log_tau_temporal_idx] = data.tau_temporal_shape
                                        - data.tau_temporal_rate * tau_temporal;
}

// The temporal AR1 correlation at this parameter vector, on (-1, 1). A model
// without one reads 0, which leaves every term depending on it inert.
static inline double read_temporal_rho_ar1(
    const std::vector<double>& params, const ModelData& data,
    const ParamLayout& layout
) {
    if (data.temporal_type == TemporalType::AR1 && layout.logit_rho_ar1_idx >= 0)
        return ratiod_ar1::rho_from_logit(params[layout.logit_rho_ar1_idx]);
    return 0.0;
}

// Its Beta prior's gradient in the sampled coordinate. Assigns rather than
// accumulates, like tau_temporal_prior_grad beside it, so it runs before
// temporal_gmrf_prior_grad's +=.
static inline void temporal_rho_prior_grad(
    const ModelData& data, const ParamLayout& layout,
    double rho_ar1, double* grad
) {
    if (data.temporal_type == TemporalType::AR1 && layout.logit_rho_ar1_idx >= 0)
        grad[layout.logit_rho_ar1_idx] = ratiod_ar1::log_prior_rho_grad(
            rho_ar1, data.temporal_rho_prior_a, data.temporal_rho_prior_b);
}

// =====================================================================
// Temporal prior gradient helper (RW1 / RW2 / AR1 / IID)
// Shared by all gradient functions that include temporal effects.
// Writes the sampled block's gradients, the tau gradient, and the rho
// gradient (for AR1).
//
// `tview` carries the effects (`phi`, from temporal_effects()), the block the
// sampler moves in (`z`), and an intrinsic walk's pre-centring sum.
// `grad_temporal_lik[0..T_len-1]` holds the likelihood contributions with
// respect to the effects, and the gradient written out is always with respect
// to the sampled block.
//
// It identifies an intrinsic walk the way compute_log_post does -- centring
// plus the augmented constant direction, see tulpa/sum_to_zero.h -- so that the
// gradient and the density it is returned alongside describe the same target.
// =====================================================================
static inline void temporal_gmrf_prior_grad(
    const ModelData& data, const ParamLayout& layout,
    double tau_temporal, double rho_ar1,
    const TemporalView& tview, int T_len,
    const double* grad_temporal_lik,
    double* grad
) {
    const double* phi_temporal = tview.phi;
    const double* z_temporal = tview.z;
    int T = data.n_times;
    int n_groups = data.n_temporal_groups;

    if (temporal_ar1_nc(data, layout)) {
        // Non-centred AR1: the chain rule through phi = L(tau, rho) z carries
        // the likelihood to z and to both hyperparameters, and seeds z's own
        // N(0, I) prior, so nothing here initializes the block first.
        double grad_log_tau = 0.0, grad_logit_rho = 0.0;
        for (int gg = 0; gg < n_groups; gg++) {
            const int base = layout.temporal_start + gg * T;
            std::pair<double, double> hyper = ratiod_temporal::ar1_nc_gradient(
                z_temporal + gg * T, phi_temporal + gg * T,
                grad_temporal_lik + gg * T, &grad[base],
                T, rho_ar1, tau_temporal);
            grad_log_tau += hyper.first;
            grad_logit_rho += hyper.second;
        }
        grad[layout.log_tau_temporal_idx] += grad_log_tau;
        if (layout.logit_rho_ar1_idx >= 0) {
            grad[layout.logit_rho_ar1_idx] += grad_logit_rho;
        }
        return;
    }

    if (temporal_rw_nc(data, layout)) {
        // Non-centred walk: eta reads the centred effects, so the likelihood
        // gradient is projected as in the centred coordinate and then carried
        // through phi = sigma * A^{-1} z. tau reaches eta only through that
        // transform, and the arm's own prior in this coordinate is N(0, I) on
        // the scaled directions and flat on the rest.
        const int order = (data.temporal_type == TemporalType::RW2) ? 2 : 1;
        const double sigma = 1.0 / std::sqrt(tau_temporal);
        std::vector<double> g_proj(grad_temporal_lik,
                                   grad_temporal_lik + T_len);
        (void)tulpa::s2z_centre_component(g_proj.data(), 0, T_len);

        std::vector<double> gz(T_len, 0.0);
        double g_log_sigma2 = 0.0;
        ratiod_temporal_nc::rw_nc_grouped_backward(
            g_proj.data(), z_temporal, data.n_times, data.n_temporal_groups,
            order, data.temporal_cyclic, sigma, gz.data(), &g_log_sigma2);
        ratiod_temporal_nc::rw_nc_grouped_log_prior_grad(
            z_temporal, data.n_times, data.n_temporal_groups, order,
            data.temporal_cyclic, gz.data());

        for (int t = 0; t < T_len; t++) {
            grad[layout.temporal_start + t] = gz[t];
        }
        // sigma2 = 1 / tau, so d/d(log tau) = -d/d(log sigma2).
        grad[layout.log_tau_temporal_idx] += -g_log_sigma2;
        return;
    }

    // Initialize temporal gradients with likelihood contribution. An intrinsic
    // walk enters eta centred, so eta depends on the sampled block only through
    // phi - mean(phi) and the gradient with respect to that block is the
    // projection (I - 11'/T_len) of the likelihood gradient: subtract its mean
    // over the whole field, matching the single global constant the prior
    // augments below.
    for (int t = 0; t < T_len; t++) {
        grad[layout.temporal_start + t] = grad_temporal_lik[t];
    }
    if (tview.centred) {
        (void)tulpa::s2z_centre_component(grad, layout.temporal_start, T_len);
    }

    // Gradient and quadratic contribution of the freed constant direction. One
    // global direction, so both are formed from the whole field's raw sum.
    const double aug_grad = tview.centred
        ? ratiod_constraints::free_direction_grad(tview.raw_sum, T_len, tau_temporal)
        : 0.0;
    const double aug_quad = tview.centred
        ? ratiod_constraints::free_direction_quad(tview.raw_sum, T_len)
        : 0.0;

    if (data.temporal_type == TemporalType::RW1) {
        double total_qf = 0.0;
        int total_rank = 0;
        for (int gg = 0; gg < n_groups; gg++) {
            const double* phi_g = phi_temporal + gg * T;
            int base = layout.temporal_start + gg * T;
            double qf = 0.0;
            for (int t = 0; t < T; t++) {
                double g = 0.0;
                if (t > 0) {
                    g += tau_temporal * (phi_g[t - 1] - phi_g[t]);
                    qf += (phi_g[t] - phi_g[t - 1]) * (phi_g[t] - phi_g[t - 1]);
                }
                if (t < T - 1) g += tau_temporal * (phi_g[t + 1] - phi_g[t]);
                grad[base + t] += g;
            }
            if (data.temporal_cyclic) {
                double dc = phi_g[0] - phi_g[T - 1];
                qf += dc * dc;
                grad[base + 0] -= tau_temporal * dc;
                grad[base + T - 1] += tau_temporal * dc;
            }
            total_qf += qf;
            total_rank += tulpa::rw1_rank(T, data.temporal_cyclic);
        }
        total_rank = tulpa::s2z_aug_rank(total_rank, tview.centred ? 1 : 0);
        for (int t = 0; t < T_len; t++) grad[layout.temporal_start + t] += aug_grad;
        grad[layout.log_tau_temporal_idx] +=
            0.5 * total_rank - 0.5 * tau_temporal * (total_qf + aug_quad);

    } else if (data.temporal_type == TemporalType::RW2) {
        double total_qf = 0.0;
        int total_rank = 0;
        for (int gg = 0; gg < n_groups; gg++) {
            const double* phi_g = phi_temporal + gg * T;
            int base = layout.temporal_start + gg * T;
            double qf = 0.0;
            for (int t = 0; t < T; t++) {
                double g = 0.0;
                if (t >= 2) g -= tau_temporal * (phi_g[t - 2] - 2.0 * phi_g[t - 1] + phi_g[t]);
                if (t >= 1 && t < T - 1) g += 2.0 * tau_temporal * (phi_g[t - 1] - 2.0 * phi_g[t] + phi_g[t + 1]);
                if (t < T - 2) g -= tau_temporal * (phi_g[t] - 2.0 * phi_g[t + 1] + phi_g[t + 2]);
                grad[base + t] += g;
            }
            for (int t = 2; t < T; t++) {
                double d2 = phi_g[t - 2] - 2.0 * phi_g[t - 1] + phi_g[t];
                qf += d2 * d2;
            }
            if (data.temporal_cyclic && T >= 3) {
                double d2_a = phi_g[T - 2] - 2.0 * phi_g[T - 1] + phi_g[0];
                double d2_b = phi_g[T - 1] - 2.0 * phi_g[0] + phi_g[1];
                qf += d2_a * d2_a + d2_b * d2_b;
                grad[base + T - 2] -= tau_temporal * d2_a;
                grad[base + T - 1] += 2.0 * tau_temporal * d2_a;
                grad[base + 0] -= tau_temporal * d2_a;
                grad[base + T - 1] -= tau_temporal * d2_b;
                grad[base + 0] += 2.0 * tau_temporal * d2_b;
                grad[base + 1] -= tau_temporal * d2_b;
            }
            total_qf += qf;
            total_rank += tulpa::rw2_rank(T, data.temporal_cyclic);
        }
        // A non-cyclic RW2 keeps its per-group LINEAR null direction, which the
        // augmentation does not touch; s2z_aug_rank adds only the one constant.
        total_rank = tulpa::s2z_aug_rank(total_rank, tview.centred ? 1 : 0);
        for (int t = 0; t < T_len; t++) grad[layout.temporal_start + t] += aug_grad;
        grad[layout.log_tau_temporal_idx] +=
            0.5 * total_rank - 0.5 * tau_temporal * (total_qf + aug_quad);

    } else if (data.temporal_type == TemporalType::AR1) {
        double omr2 = ratiod_ar1::one_minus_rho2(rho_ar1);
        double total_qf = 0.0, total_gr = 0.0;
        for (int gg = 0; gg < n_groups; gg++) {
            const double* phi_g = phi_temporal + gg * T;
            int base = layout.temporal_start + gg * T;
            grad[base] += -tau_temporal * omr2 * phi_g[0];
            if (T > 1) grad[base] += tau_temporal * rho_ar1 * (phi_g[1] - rho_ar1 * phi_g[0]);
            double qf = omr2 * phi_g[0] * phi_g[0];
            for (int t = 1; t < T; t++) {
                double r = phi_g[t] - rho_ar1 * phi_g[t - 1];
                qf += r * r;
                double g = -tau_temporal * r;
                if (t < T - 1) g += tau_temporal * rho_ar1 * (phi_g[t + 1] - rho_ar1 * phi_g[t]);
                grad[base + t] += g;
            }
            total_qf += qf;
            total_gr += tau_temporal * rho_ar1 * phi_g[0] * phi_g[0];
            for (int t = 1; t < T; t++) {
                total_gr += tau_temporal * (phi_g[t] - rho_ar1 * phi_g[t - 1]) * phi_g[t - 1];
            }
        }
        grad[layout.log_tau_temporal_idx] += 0.5 * T_len - 0.5 * tau_temporal * total_qf;
        if (layout.logit_rho_ar1_idx >= 0) {
            double gr = -n_groups * rho_ar1 / omr2 + total_gr;
            grad[layout.logit_rho_ar1_idx] += gr * ratiod_ar1::drho_dlogit(rho_ar1);
        }

    } else if (data.temporal_type == TemporalType::IID) {
        double total_qf = 0.0;
        for (int gg = 0; gg < n_groups; gg++) {
            const double* phi_g = phi_temporal + gg * T;
            int base = layout.temporal_start + gg * T;
            for (int t = 0; t < T; t++) {
                grad[base + t] += -tau_temporal * phi_g[t];
                total_qf += phi_g[t] * phi_g[t];
            }
        }
        grad[layout.log_tau_temporal_idx] += 0.5 * T_len - 0.5 * tau_temporal * total_qf;
    }
}

// =====================================================================
// PC prior gradient on log(sigma2) for GP variances
// Returns d log_prior / d log_sigma2 INCLUDING Jacobian for exp transform.
// Formula: -0.5 * rate * sigma + 0.5  where rate = -log(alpha) / U
// =====================================================================
static inline double gp_pc_prior_grad_log_sigma2(
    double sigma2, double U, double alpha
) {
    double sigma = std::sqrt(sigma2 + 1e-10);
    double rate = -std::log(alpha + 1e-10) / (U + 1e-10);
    return -0.5 * rate * sigma + 0.5;
}


// =====================================================================
// Spatial ICAR/BYM2 GMRF prior gradient.
//
// The single source of truth for the intrinsic spatial field's gradient, used
// by every gradient path that carries one. It identifies the field the way
// compute_log_post does -- hard centring plus the freed constant direction,
// see spatial_field_constraint.h -- so that the gradient and the density it is
// returned alongside describe the same target.
//
// phi_spatial must be the centred field and phi_raw_sum its pre-centring sum,
// both as produced by CenteredSpatialParams. grad_spatial_lik holds the
// accumulated likelihood gradient with respect to the phi *parameters*, so for
// BYM2 the sigma_s * scale chain factor is already applied by the caller's
// scatter. theta_bym2 gradients stay with the caller, which sets the prior term
// and accumulates the likelihood term during its own scatter.
// =====================================================================
// The multi-feature gradient paths scatter the spatial field's likelihood
// gradient in eta units (a bare dLL per observation). spatial_gmrf_prior_grad
// works in phi-parameter units, which for BYM2 differ by the sigma_s * scale
// chain factor; the vectorized main path applies that during its own scatter.
// Settles the BYM2 theta block at the same time, which stays with the caller
// because its likelihood term is scattered per observation.
static inline void spatial_lik_grad_to_param_scale(
    const ModelData& data, const ParamLayout& layout,
    double sigma_s_bym2, double sigma_u_bym2,
    const double* theta_bym2, const double* grad_theta_lik,
    double* grad_spatial_lik, double* grad
) {
    if (data.spatial_type != SpatialType::BYM2) return;
    const int S = data.n_spatial_units;
    const double chain = sigma_s_bym2 * data.bym2_scale_factor;
    for (int s = 0; s < S; s++) {
        grad_spatial_lik[s] *= chain;
        grad[layout.theta_bym2_start + s] =
            grad_theta_lik[s] * sigma_u_bym2 - theta_bym2[s];
    }
}

static inline void spatial_gmrf_prior_grad(
    const ModelData& data, const ParamLayout& layout,
    const double* phi_spatial,
    const double* phi_spatial_prior,
    double phi_raw_sum,
    bool centered,
    double tau_spatial,
    double sigma_s_bym2, double sigma_u_bym2,
    double rho_bym2,
    double rho_car,
    const double* theta_bym2,
    const double* grad_spatial_lik,
    double* grad
) {
    const int S = data.n_spatial_units;
    const bool is_bym2 = (data.spatial_type == SpatialType::BYM2);
    const bool is_car_proper = (data.spatial_type == SpatialType::CAR_PROPER);

    // The likelihood sees the centred field, so its gradient with respect to
    // the raw parameters is the projection (I - 11'/n). Applied here only:
    // d(eta)/d(sigma_bym2) is a function of the centred field with no such
    // projection, so grad_spatial_lik itself must stay unprojected.
    double glik_mean = 0.0;
    if (centered) {
        for (int s = 0; s < S; s++) glik_mean += grad_spatial_lik[s];
        glik_mean /= static_cast<double>(S);
    }
    for (int s = 0; s < S; s++) {
        grad[layout.spatial_start + s] = grad_spatial_lik[s] - glik_mean;
    }

    // ICAR prior: -0.5 * tau * phi' * Q * phi where Q_ij = n_neighbors[i] if i=j, -1 if i~j
    // d/d(phi[i]) = -tau * (n_neighbors[i]*phi[i] - sum_{j~i} phi[j])
    // Proper CAR: Q(rho)_ij = n_neighbors[i] if i=j, -rho if i~j; no freed direction.
    double icar_quad = 0.0;
    // Gradient of the freed constant direction; see spatial_field_constraint.h.
    const double free_scale = is_bym2 ? 1.0 : tau_spatial;
    const double free_grad = is_car_proper ? 0.0 :
        ratiod_constraints::free_direction_grad(phi_raw_sum, S, free_scale);
    for (int i = 0; i < S; i++) {
        double Qphi_i = data.n_neighbors[i] * phi_spatial_prior[i];
        const double w = is_car_proper ? rho_car : 1.0;
        for (int k = data.adj_row_ptr[i]; k < data.adj_row_ptr[i + 1]; k++) {
            const int j = data.adj_col_idx[k];
            Qphi_i -= w * phi_spatial_prior[j];
            if (j > i && !is_car_proper) {
                const double diff = phi_spatial_prior[i] - phi_spatial_prior[j];
                icar_quad += diff * diff;
            }
        }

        if (is_bym2) {
            // For BYM2, ICAR prior has no tau scaling (it's absorbed into sigma/rho)
            grad[layout.spatial_start + i] += -Qphi_i + free_grad;
        } else {
            grad[layout.spatial_start + i] += -tau_spatial * Qphi_i + free_grad;
        }
    }

    if (is_bym2) {
        // BYM2 Riebler: transform (grad_sigma_s, grad_sigma_u) to (grad_log_sigma, grad_logit_rho)
        // grad_sigma_s_lik = d(LL)/d(sigma_s) * sigma_s  (chain rule for log)
        // grad_sigma_u_lik = d(LL)/d(sigma_u) * sigma_u
        double grad_sigma_s_lik = 0.0;
        double grad_sigma_u_lik = 0.0;
        for (int s = 0; s < S; s++) {
            const double scaled_phi = phi_spatial[s] * data.bym2_scale_factor;
            const double d_LL_d_spatial =
                grad_spatial_lik[s] / (sigma_s_bym2 * data.bym2_scale_factor);
            grad_sigma_s_lik += d_LL_d_spatial * sigma_s_bym2 * scaled_phi;
            grad_sigma_u_lik += d_LL_d_spatial * sigma_u_bym2 * theta_bym2[s];
        }
        grad[layout.log_sigma_bym2_idx] += grad_sigma_s_lik + grad_sigma_u_lik;
        grad[layout.logit_rho_bym2_idx] += 0.5 * ((1.0 - rho_bym2) * grad_sigma_s_lik
                                                    - rho_bym2 * grad_sigma_u_lik);
    } else if (is_car_proper) {
        // tau gradient: same shape as plain ICAR, but the quadratic form is
        // rho-weighted (car_quadratic_form), not the rho=1 special case above.
        const double quad = ratiod_car_proper::car_quadratic_form(
            phi_spatial_prior, S, data.adj_row_ptr, data.adj_col_idx,
            data.n_neighbors, rho_car);
        grad[layout.log_tau_spatial_idx] += 0.5 * S - 0.5 * tau_spatial * quad;

        // rho gradient: 0.5*log|Q(rho)| has no closed form cheaper than a
        // dense Cholesky (trace(Q^-1 W) needs the inverse), so it is central-
        // differenced here, on this one scalar, and chain-ruled to logit
        // scale. The quadratic-form part is exact (it is linear in rho).
        auto gmrf_term = [&](double rho) {
            auto Q = ratiod_car_proper::compute_car_precision(
                S, data.adj_row_ptr, data.adj_col_idx, data.n_neighbors, rho);
            const double logdet = ratiod_car_proper::car_log_det(S, Q);
            const double q = ratiod_car_proper::car_quadratic_form(
                phi_spatial_prior, S, data.adj_row_ptr, data.adj_col_idx,
                data.n_neighbors, rho);
            return 0.5 * logdet - 0.5 * tau_spatial * q;
        };
        const double eps = 1e-5;
        const double rho_p = std::min(rho_car + eps, data.car_rho_upper - 1e-9);
        const double rho_m = std::max(rho_car - eps, data.car_rho_lower + 1e-9);
        const double d_gmrf_d_rho = (gmrf_term(rho_p) - gmrf_term(rho_m)) / (rho_p - rho_m);
        const double u = (rho_car - data.car_rho_lower) / (data.car_rho_upper - data.car_rho_lower);
        const double d_rho_d_logit = (data.car_rho_upper - data.car_rho_lower) * u * (1.0 - u);
        grad[layout.logit_rho_car_idx] += d_gmrf_d_rho * d_rho_d_logit;
    } else {
        // Plain ICAR: tau gradient. (Q + 11'/J) is full rank, so the exponent on
        // tau is n/2 and the quadratic form carries the freed direction.
        // log_post = 0.5*n*log(tau) - 0.5*tau*(quad + sum^2/n) + const
        const double quad_full =
            icar_quad + ratiod_constraints::free_direction_quad(phi_raw_sum, S);
        grad[layout.log_tau_spatial_idx] += 0.5 * S - 0.5 * tau_spatial * quad_full;
    }
}

// =====================================================================
// GP gradient (hand-coded, ~3x faster than autodiff)
// Uses analytical gradients from gp_nngp_gradients for NNGP prior
// =====================================================================

void compute_gradient_gp_handcoded(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    std::vector<double>& grad,
    double* log_post_out = nullptr
) {
    double obs_log_lik = 0.0;
    int n_params = params.size();
    grad.assign(n_params, 0.0);

    // Extract common parameters
    auto cp = extract_common_params(params, layout);
    const double* beta_num = cp.beta_num;
    const double* beta_denom = cp.beta_denom;
    double sigma_re = cp.sigma_re;
    const double* re = cp.re;
    double phi_num = cp.phi_num;
    double phi_denom = cp.phi_denom;

    // GP parameters
    int N_gp = data.gp_data.n_obs;
    double sigma2_gp = std::exp(params[layout.log_sigma2_gp_idx]);
    double phi_gp = std::exp(params[layout.log_phi_gp_idx]);

    if (phi_gp < data.gp_phi_prior_lower || phi_gp > data.gp_phi_prior_upper) {
        return;
    }

    // Non-centered parameterization: params store z ~ N(0,1), reconstruct w
    const bool use_nc = (data.gp_parameterization == 1);
    RATIOD_TLS_WORKSPACE(ratiod_gp::NNGPNCWorkspace, nc_ws);
    const bool fuse_lp = (log_post_out != nullptr) && !layout.has_zi;
    if (log_post_out && layout.has_zi) *log_post_out = compute_log_post(params, data, layout);

    // Get spatial effects: either w directly (centered) or reconstruct from z (NC)
    std::vector<double> gp_w(N_gp);
    if (use_nc) {
        // Forward pass: z -> w
        const double* z_params = &params[layout.gp_w_start];
        ratiod_gp::nngp_nc_forward(z_params, sigma2_gp, phi_gp, data.gp_data, nc_ws);
        // Use reconstructed w
        std::memcpy(gp_w.data(), nc_ws.w.data(), N_gp * sizeof(double));
    } else {
        for (int i = 0; i < N_gp; i++) {
            gp_w[i] = params[layout.gp_w_start + i];
        }
    }

    // Prior gradients
    beta_gradient_prior(data, layout, beta_num, beta_denom, grad.data());
    re_gradient_prior(data, layout, re, grad.data(), sigma_re);
    phi_gradient_prior(data, layout, phi_num, phi_denom, grad.data());

    // PC prior on GP variance
    grad[layout.log_sigma2_gp_idx] = gp_pc_prior_grad_log_sigma2(
        sigma2_gp, data.gp_sigma2_prior_U, data.gp_sigma2_prior_alpha);

    // Uniform prior on phi - just Jacobian for log transform
    grad[layout.log_phi_gp_idx] = 1.0;

    // NC: no Jacobian gradient needed (cancels with NNGP normalizing constant)

    if (!use_nc) {
        // =====================================================================
        // Centered: NNGP prior gradients on w
        // =====================================================================
        ratiod_gp::NNGPGradients nngp_grads;
        ratiod_gp::gp_nngp_gradients(gp_w, sigma2_gp, phi_gp, data.gp_data, nngp_grads);

        for (int i = 0; i < N_gp; i++) {
            grad[layout.gp_w_start + i] += nngp_grads.grad_w[i];
        }
        grad[layout.log_sigma2_gp_idx] += nngp_grads.grad_log_sigma2;
        grad[layout.log_phi_gp_idx] += nngp_grads.grad_log_phi;
    }

    // =========================================================================
    // Data likelihood loop (vectorized eta + per-obs scatter)
    // =========================================================================
    const int N = data.N;
    const bool is_binomial = (data.model_type == ModelType::BINOMIAL ||
                              data.model_type == ModelType::BETA_BINOMIAL);

    // --- Pass 1: Vectorized eta computation (Eigen matvec) ---
    using RowMajorMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using VectorXd = Eigen::VectorXd;

    vec_grad_ws().init(N);
    Eigen::Map<const RowMajorMatrix> X_num(data.X_num_flat.data(), N, data.p_num);
    Eigen::Map<const VectorXd> b_num(beta_num, data.p_num);
    Eigen::Map<VectorXd> eta_n(vec_grad_ws().eta_num.data(), N);
    eta_n.noalias() = X_num * b_num;

    if (!is_binomial) {
        Eigen::Map<const RowMajorMatrix> X_denom(data.X_denom_flat.data(), N, data.p_denom);
        Eigen::Map<const VectorXd> b_denom(beta_denom, data.p_denom);
        Eigen::Map<VectorXd> eta_d(vec_grad_ws().eta_denom.data(), N);
        eta_d.noalias() = X_denom * b_denom;
    } else {
        std::memset(vec_grad_ws().eta_denom.data(), 0, N * sizeof(double));
    }

    // Add RE + GP effects per-obs (using reconstructed w for NC)
    for (int i = 0; i < N; i++) {
        if (layout.has_re && data.re_group[i] > 0) {
            int g = data.re_group[i] - 1;
            double re_eff = re_value_for_eta(re, g, sigma_re, data.re_parameterization);
            vec_grad_ws().eta_num[i] += re_eff;
            if (!is_binomial) vec_grad_ws().eta_denom[i] += re_eff;
        }
        int loc_i = data.gp_data.obs_to_loc[i];
        double gp_effect = gp_w[loc_i];
        vec_grad_ws().eta_num[i] += gp_effect;
        if (!is_binomial && data.gp_data.shared) vec_grad_ws().eta_denom[i] += gp_effect;
    }

    // --- Pass 2+3: Vectorized residuals + beta grads (template-dispatched) ---
    {
        double grad_phi_num_lik = 0.0, grad_phi_denom_lik = 0.0;
        vectorized::dispatch_residuals_and_beta_grads(
            data, layout,
            vec_grad_ws().eta_num.data(), vec_grad_ws().eta_denom.data(),
            vec_grad_ws().resid_num.data(), vec_grad_ws().resid_denom.data(),
            grad.data(), grad_phi_num_lik, grad_phi_denom_lik,
            obs_log_lik, fuse_lp, phi_num, phi_denom, vec_grad_ws());
    }

    // Scatter residuals to RE gradients
    for (int i = 0; i < N; i++) {
        if (layout.has_re && data.re_group[i] > 0) {
            grad[layout.re_start + data.re_group[i] - 1] += vec_grad_ws().resid_num[i] + vec_grad_ws().resid_denom[i];
        }
    }

    if (use_nc) {
        // =====================================================================
        // NC: Accumulate dL/dw from residuals, then backward pass for z gradients
        // =====================================================================
        std::vector<double> dL_dw(N_gp, 0.0);
        for (int i = 0; i < N; i++) {
            int loc_i = data.gp_data.obs_to_loc[i];
            double dLL = data.gp_data.shared
                ? (vec_grad_ws().resid_num[i] + vec_grad_ws().resid_denom[i])
                : vec_grad_ws().resid_num[i];
            dL_dw[loc_i] += dLL;
        }

        // Backward pass: dL/dw -> grad_z, grad_log_sigma2, grad_log_phi
        std::vector<double> grad_z(N_gp, 0.0);
        double grad_log_sigma2_lik = 0.0, grad_log_phi_lik = 0.0, grad_log_phi_jac = 0.0;
        const double* z_params = &params[layout.gp_w_start];
        ratiod_gp::nngp_nc_backward(
            z_params, sigma2_gp, phi_gp, data.gp_data, nc_ws,
            dL_dw.data(), grad_z.data(),
            grad_log_sigma2_lik, grad_log_phi_lik, grad_log_phi_jac);

        // nngp_nc_backward returns the FULL z gradient -- its own contract is
        // "prior + likelihood", and it seeds each entry with -z[loc] before
        // adding sqrt(d_i) * adj_i. Subtracting z_params here as well applied
        // the N(0,1) prior twice.
        for (int i = 0; i < N_gp; i++) {
            grad[layout.gp_w_start + i] += grad_z[i];
        }

        // Add likelihood contributions to hyperparameters
        // (no Jacobian — cancels with NNGP normalizing constant)
        grad[layout.log_sigma2_gp_idx] += grad_log_sigma2_lik;
        grad[layout.log_phi_gp_idx] += grad_log_phi_lik;
    } else {
        // Centered: scatter residuals directly to GP w gradients
        for (int i = 0; i < N; i++) {
            int loc_i = data.gp_data.obs_to_loc[i];
            double dLL_dspatial = data.gp_data.shared
                ? (vec_grad_ws().resid_num[i] + vec_grad_ws().resid_denom[i])
                : vec_grad_ws().resid_num[i];
            grad[layout.gp_w_start + loc_i] += dLL_dspatial;
        }
    }

    re_gradient_nc_transform(data, layout, params.data(), grad.data(), sigma_re);

    // Log-posterior computation
    if (fuse_lp) {
        if (use_nc) {
            // NC: use full compute_log_post to ensure perfect consistency
            *log_post_out = compute_log_post(params, data, layout);
        } else {
            *log_post_out = compute_log_post(params, data, layout, /*skip_obs_loop=*/true) + obs_log_lik;
        }
    }

}

// =====================================================================
// Collapsed GP gradient (hand-coded)
// GP effects marginalized via inner Laplace — only hyperparams in HMC
// =====================================================================

// collapsed_gp_ws() declared earlier (shared with compute_log_post)

void compute_gradient_gp_collapsed(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    std::vector<double>& grad,
    double* log_post_out = nullptr
) {
    int n_params = params.size();
    grad.assign(n_params, 0.0);

    // Extract common parameters
    auto cp = extract_common_params(params, layout);
    const double* beta_num = cp.beta_num;
    const double* beta_denom = cp.beta_denom;
    double sigma_re = cp.sigma_re;
    const double* re = cp.re;
    double phi_num = cp.phi_num;
    double phi_denom = cp.phi_denom;

    // GP hyperparameters
    double sigma2_gp = std::exp(params[layout.log_sigma2_gp_idx]);
    double phi_gp = std::exp(params[layout.log_phi_gp_idx]);

    if (phi_gp < data.gp_phi_prior_lower || phi_gp > data.gp_phi_prior_upper) {
        if (log_post_out) *log_post_out = -INFINITY;
        return;
    }

    int N_gp = data.gp_data.n_obs;
    int N = data.N;
    bool is_binomial = (data.model_type == ModelType::BINOMIAL ||
                        data.model_type == ModelType::BETA_BINOMIAL);

    // ---- Inner Laplace: find w* ----
    double collapsed_lp = collapsed_gp_find_mode(
        beta_num, beta_denom, sigma2_gp, phi_gp,
        phi_num, phi_denom, data, collapsed_gp_ws());

    // ---- Prior gradients (outer params only) ----
    beta_gradient_prior(data, layout, beta_num, beta_denom, grad.data());
    re_gradient_prior(data, layout, re, grad.data(), sigma_re);
    phi_gradient_prior(data, layout, phi_num, phi_denom, grad.data());

    // GP hyperparameter priors
    grad[layout.log_sigma2_gp_idx] = gp_pc_prior_grad_log_sigma2(
        sigma2_gp, data.gp_sigma2_prior_U, data.gp_sigma2_prior_alpha);
    grad[layout.log_phi_gp_idx] = 1.0;  // Uniform prior Jacobian

    // ---- Data likelihood gradient at w* ----
    // Compute residuals at the mode w*
    std::vector<double> resid_num(N), resid_denom(N);
    collapsed_gp_compute_residuals(
        collapsed_gp_ws().w_star.data(), beta_num, beta_denom,
        phi_num, phi_denom, data,
        resid_num.data(), resid_denom.data());

    // Scatter to beta gradients + phi likelihood gradient
    for (int i = 0; i < N; i++) {
        for (int p = 0; p < data.p_num; p++) {
            grad[layout.beta_num_start + p] += resid_num[i] * data.X_num_flat[i * data.p_num + p];
        }
        if (!is_binomial) {
            for (int p = 0; p < data.p_denom; p++) {
                grad[layout.beta_denom_start + p] += resid_denom[i] * data.X_denom_flat[i * data.p_denom + p];
            }
        }
        // Scatter to RE gradients
        if (layout.has_re && data.re_group.size() > (size_t)i && data.re_group[i] > 0) {
            grad[layout.re_start + data.re_group[i] - 1] += resid_num[i] + resid_denom[i];
        }

        // Phi (dispersion) likelihood gradient
        if (layout.has_phi_num || layout.has_phi_denom) {
            int loc_i = data.gp_data.obs_to_loc[i];
            double eta_num_i = 0.0, eta_denom_i = 0.0;
            for (int p = 0; p < data.p_num; p++)
                eta_num_i += data.X_num_flat[i * data.p_num + p] * beta_num[p];
            if (!is_binomial) {
                for (int p = 0; p < data.p_denom; p++)
                    eta_denom_i += data.X_denom_flat[i * data.p_denom + p] * beta_denom[p];
            }
            eta_num_i += collapsed_gp_ws().w_star[loc_i];
            if (!is_binomial && data.gp_data.shared) eta_denom_i += collapsed_gp_ws().w_star[loc_i];
            if (layout.has_re && data.re_group.size() > (size_t)i && data.re_group[i] > 0) {
                double re_val = (data.re_parameterization == 1) ?
                    sigma_re * re[data.re_group[i] - 1] : re[data.re_group[i] - 1];
                eta_num_i += re_val;
                if (!is_binomial) eta_denom_i += re_val;
            }
            accumulate_phi_likelihood_grad(data, layout, i, eta_num_i, eta_denom_i,
                                            phi_num, phi_denom, grad.data());
        }
    }

    // ---- GP hyperparameter gradients from NNGP prior at w* ----
    // d/d(log sigma2) and d/d(log phi) of log p_NNGP(w*|sigma2,phi)
    // Use the existing NNGP gradient function
    ratiod_gp::NNGPGradients nngp_grads;
    std::vector<double> w_star_vec(collapsed_gp_ws().w_star.begin(),
                                    collapsed_gp_ws().w_star.end());
    ratiod_gp::gp_nngp_gradients(w_star_vec, sigma2_gp, phi_gp,
                                  data.gp_data, nngp_grads);
    grad[layout.log_sigma2_gp_idx] += nngp_grads.grad_log_sigma2;
    grad[layout.log_phi_gp_idx] += nngp_grads.grad_log_phi;

    // ---- Laplace correction gradient via numerical differentiation ----
    // The Laplace log-det depends on w* which depends on ALL params implicitly.
    // We compute d/dθ [-0.5 log det(W+Q)] numerically for each outer param,
    // using warm-started Newton solves (1-2 iters each from current w*).
    // For non-GP params (beta, phi, RE), we reuse the NNGP structure from
    // collapsed_gp_ws() (Q doesn't change), skipping NNGP rebuild.
    {
        const double eps = 1e-5;
        std::vector<double> params_pert = params;
        const int gp_idx1 = layout.log_sigma2_gp_idx;
        const int gp_idx2 = layout.log_phi_gp_idx;

        for (int j = 0; j < n_params; j++) {
            double orig = params_pert[j];
            bool is_gp_hyperparam = (j == gp_idx1 || j == gp_idx2);

            // Forward perturbation
            params_pert[j] = orig + eps;
            double sigma2_p = std::exp(params_pert[gp_idx1]);
            double phi_p = std::exp(params_pert[gp_idx2]);
            const double* beta_num_p = &params_pert[layout.beta_num_start];
            const double* beta_denom_p = is_binomial ? beta_num_p : &params_pert[layout.beta_denom_start];
            double phi_num_p = layout.has_phi_num ? std::exp(params_pert[layout.log_phi_num_idx]) : phi_num;
            double phi_denom_p = layout.has_phi_denom ? std::exp(params_pert[layout.log_phi_denom_idx]) : phi_denom;
            double ld_plus = laplace_log_det_full(
                beta_num_p, beta_denom_p, sigma2_p, phi_p,
                phi_num_p, phi_denom_p, data, collapsed_gp_ws().w_star,
                is_gp_hyperparam ? nullptr : &collapsed_gp_ws(),
                is_gp_hyperparam);

            // Backward perturbation
            params_pert[j] = orig - eps;
            sigma2_p = std::exp(params_pert[gp_idx1]);
            phi_p = std::exp(params_pert[gp_idx2]);
            beta_num_p = &params_pert[layout.beta_num_start];
            beta_denom_p = is_binomial ? beta_num_p : &params_pert[layout.beta_denom_start];
            phi_num_p = layout.has_phi_num ? std::exp(params_pert[layout.log_phi_num_idx]) : phi_num;
            phi_denom_p = layout.has_phi_denom ? std::exp(params_pert[layout.log_phi_denom_idx]) : phi_denom;
            double ld_minus = laplace_log_det_full(
                beta_num_p, beta_denom_p, sigma2_p, phi_p,
                phi_num_p, phi_denom_p, data, collapsed_gp_ws().w_star,
                is_gp_hyperparam ? nullptr : &collapsed_gp_ws(),
                is_gp_hyperparam);

            grad[j] += (ld_plus - ld_minus) / (2.0 * eps);
            params_pert[j] = orig;
        }
    }

    // ---- NC transform for RE ----
    re_gradient_nc_transform(data, layout, params.data(), grad.data(), sigma_re);

    // ---- Log-posterior ----
    if (log_post_out) {
        // Compute full log-posterior including priors on outer params
        // collapsed_lp already has data_ll + nngp_prior
        double lp = collapsed_lp;

        // Laplace correction: -0.5 * log det(W + Q) via sparse Cholesky
        lp += collapsed_gp_ws().laplace_log_det;

        // Beta priors
        for (int p = 0; p < data.p_num; p++)
            lp += -0.5 * beta_num[p] * beta_num[p] / (data.sigma_beta * data.sigma_beta);
        for (int p = 0; p < data.p_denom; p++)
            lp += -0.5 * beta_denom[p] * beta_denom[p] / (data.sigma_beta * data.sigma_beta);

        // RE priors
        if (layout.has_re) {
            int n_re = layout.re_end - layout.re_start;
            double sigma_re2 = sigma_re * sigma_re;
            for (int g = 0; g < n_re; g++) {
                if (data.re_parameterization == 1) {
                    // NC: z ~ N(0,1)
                    lp += -0.5 * re[g] * re[g];
                } else {
                    lp += -0.5 * re[g] * re[g] / sigma_re2;
                }
            }
            // sigma_re half-Cauchy prior (log-scale)
            double ratio = sigma_re / data.sigma_re_scale;
            lp += -std::log(1.0 + ratio * ratio)
                  + params[layout.log_sigma_re_idx];
        }

        // GP hyperparameter priors
        double sigma_gp = std::sqrt(sigma2_gp);
        double rate = -std::log(data.gp_sigma2_prior_alpha) / data.gp_sigma2_prior_U;
        lp += std::log(rate) - rate * sigma_gp - std::log(2.0 * sigma_gp)
              + params[layout.log_sigma2_gp_idx];
        // phi uniform + Jacobian
        lp += params[layout.log_phi_gp_idx]
              - std::log(data.gp_phi_prior_upper - data.gp_phi_prior_lower);

        // Phi (dispersion) priors
        if (layout.has_phi_num) {
            double log_phi = params[layout.log_phi_num_idx];
            lp += log_phi;  // Jacobian for log transform (exponential prior)
        }
        if (layout.has_phi_denom) {
            double log_phi = params[layout.log_phi_denom_idx];
            lp += log_phi;
        }

        *log_post_out = lp;
    }
}

// =====================================================================
// Collapsed ICAR/BYM2 gradient
// ICAR: H-mode (analytical envelope + analytical Laplace via implicit fn thm)
// BYM2: numerical fallback (H-mode planned)
// =====================================================================

void compute_gradient_icar_collapsed(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    std::vector<double>& grad,
    double* log_post_out = nullptr
) {
    int n_params = params.size();
    grad.assign(n_params, 0.0);

    // Extract common parameters
    auto cp = extract_common_params(params, layout);
    const double* beta_num = cp.beta_num;
    const double* beta_denom = cp.beta_denom;
    double sigma_re = cp.sigma_re;
    const double* re = cp.re;
    double phi_num = cp.phi_num;
    double phi_denom = cp.phi_denom;

    bool is_binomial = (data.model_type == ModelType::BINOMIAL ||
                        data.model_type == ModelType::BETA_BINOMIAL);
    int N = data.N;
    int S = data.n_spatial_units;

    // Pre-compute actual RE values (NC → actual)
    std::vector<double> re_vals;
    if (layout.has_re) {
        int n_re = layout.re_end - layout.re_start;
        re_vals.resize(n_re);
        for (int g = 0; g < n_re; g++) {
            re_vals[g] = (data.re_parameterization == 1) ? sigma_re * re[g] : re[g];
        }
    }

    // Companion plain temporal GMRF term, held fixed for the inner Laplace
    // the same way beta and re_vals are (see collapsed_temporal_obs_offset).
    CollapsedTemporalOffset collapsed_temporal_offset =
        collapsed_temporal_obs_offset(data, layout, params);
    const double* temporal_offset_num = collapsed_temporal_offset.num_ptr();
    const double* temporal_offset_denom = collapsed_temporal_offset.denom_ptr();

    // Spatial hyperparameters
    bool is_bym2 = layout.is_bym2_collapsed;
    double tau = 0.0, sigma_total = 0.0, rho = 0.0;
    double a = 0.0, c_bym2 = 0.0;

    if (is_bym2) {
        sigma_total = std::exp(params[layout.log_sigma_bym2_idx]);
        double logit_rho = params[layout.logit_rho_bym2_idx];
        rho = 1.0 / (1.0 + std::exp(-logit_rho));
        a = sigma_total * std::sqrt(rho) * data.bym2_scale_factor;
        c_bym2 = sigma_total * std::sqrt(1.0 - rho);
    } else {
        tau = std::exp(params[layout.log_tau_spatial_idx]);
    }

    // ---- Inner Laplace: find phi* (and theta* for BYM2) ----
    double collapsed_lp;
    if (is_bym2) {
        collapsed_lp = collapsed_bym2_find_mode(
            beta_num, beta_denom, sigma_total, rho, data.bym2_scale_factor,
            phi_num, phi_denom,
            re_vals.empty() ? nullptr : re_vals.data(),
            data, collapsed_icar_ws(), 20, 1e-6,
            temporal_offset_num, temporal_offset_denom);
    } else {
        collapsed_lp = collapsed_icar_find_mode(
            beta_num, beta_denom, tau, phi_num, phi_denom,
            re_vals.empty() ? nullptr : re_vals.data(),
            data, collapsed_icar_ws(), 20, 1e-6,
            temporal_offset_num, temporal_offset_denom);
    }

    // ---- Outer priors (simple analytical, don't depend on phi*) ----
    beta_gradient_prior(data, layout, beta_num, beta_denom, grad.data());
    re_gradient_prior(data, layout, re, grad.data(), sigma_re);
    phi_gradient_prior(data, layout, phi_num, phi_denom, grad.data());

    // Filled by whichever branch below runs; read afterwards to scatter a
    // companion temporal term's likelihood gradient (both were computed at
    // the correct, temporal-inclusive eta).
    std::vector<double> resid_num(N), resid_denom(N);

    // ---- ICAR H-mode: analytical envelope + analytical Laplace ----
    // BYM2: numerical fallback (TODO: implement BYM2 H-mode)
    if (!is_bym2) {
        // === Part A: Envelope theorem gradient ===
        // At mode, ∂f/∂φ = 0, so d/dθ[f(φ*,θ)] = ∂f/∂θ|_{φ*}

        // A1: Data LL gradient via residual scattering
        collapsed_icar_compute_residuals(
            collapsed_icar_ws(), beta_num, beta_denom,
            phi_num, phi_denom,
            re_vals.empty() ? nullptr : re_vals.data(),
            0.0, 0.0,  // not BYM2
            data, resid_num.data(), resid_denom.data(),
            temporal_offset_num, temporal_offset_denom);

        // Scatter residuals to beta_num: X_num' * resid_num
        for (int k = 0; k < data.p_num; k++) {
            double sum = 0.0;
            for (int i = 0; i < N; i++)
                sum += resid_num[i] * data.X_num_flat[i * data.p_num + k];
            grad[layout.beta_num_start + k] += sum;
        }
        // Scatter to beta_denom: X_denom' * resid_denom
        if (!is_binomial) {
            for (int k = 0; k < data.p_denom; k++) {
                double sum = 0.0;
                for (int i = 0; i < N; i++)
                    sum += resid_denom[i] * data.X_denom_flat[i * data.p_denom + k];
                grad[layout.beta_denom_start + k] += sum;
            }
        }
        // Scatter to RE (w.r.t. centered values)
        if (layout.has_re) {
            int n_re = layout.re_end - layout.re_start;
            for (int i = 0; i < N; i++) {
                if (data.re_group.size() > (size_t)i && data.re_group[i] > 0) {
                    int g = data.re_group[i] - 1;
                    if (g < n_re) {
                        grad[layout.re_start + g] += resid_num[i] + resid_denom[i];
                    }
                }
            }
        }

        // A2: Dispersion parameter gradients from data LL at mode
        // (envelope theorem: ∂LL/∂log_phi evaluated at φ*)
        if (layout.has_phi_num || layout.has_phi_denom) {
            for (int i = 0; i < N; i++) {
                int s = data.spatial_group[i] - 1;
                double eta_num_i = 0.0, eta_denom_i = 0.0;
                for (int p = 0; p < data.p_num; p++)
                    eta_num_i += data.X_num_flat[i * data.p_num + p] * beta_num[p];
                if (!is_binomial) {
                    for (int p = 0; p < data.p_denom; p++)
                        eta_denom_i += data.X_denom_flat[i * data.p_denom + p] * beta_denom[p];
                }
                eta_num_i += collapsed_icar_ws().phi_star[s];
                if (!is_binomial) eta_denom_i += collapsed_icar_ws().phi_star[s];
                if (re_vals.data() && data.re_group.size() > (size_t)i && data.re_group[i] > 0)  {
                    eta_num_i += re_vals[data.re_group[i] - 1];
                    if (!is_binomial) eta_denom_i += re_vals[data.re_group[i] - 1];
                }
                if (temporal_offset_num != nullptr) eta_num_i += temporal_offset_num[i];
                if (!is_binomial && temporal_offset_denom != nullptr) eta_denom_i += temporal_offset_denom[i];

                double mu_num = std::exp(std::min(eta_num_i, 20.0));

                // Per-family dispersion gradients
                switch (data.model_type) {
                    case ModelType::POISSON_GAMMA: {
                        if (layout.has_phi_denom) {
                            double mu_d = std::exp(std::min(eta_denom_i, 20.0));
                            double y_d = data.y_denom_cont[i];
                            double alpha = phi_denom;
                            // d/d(log_alpha)[LL_denom] = alpha * dLL/dalpha
                            // dLL/dalpha = log(alpha) + 1 - digamma(alpha) + log(y/mu) - y/mu
                            grad[layout.log_phi_denom_idx] += alpha * (
                                std::log(alpha) + 1.0 - R::digamma(alpha)
                                + std::log(std::max(y_d, 1e-10)) - std::log(mu_d)
                                - y_d / mu_d);
                        }
                        break;
                    }
                    case ModelType::NEGBIN_NEGBIN: {
                        if (layout.has_phi_num) {
                            double r = phi_num;
                            double y = data.y_num[i];
                            grad[layout.log_phi_num_idx] += r * (
                                R::digamma(y + r) - R::digamma(r)
                                + std::log(r) + 1.0
                                - std::log(mu_num + r) - (y + r) / (mu_num + r));
                        }
                        if (layout.has_phi_denom && !is_binomial) {
                            double mu_d = std::exp(std::min(eta_denom_i, 20.0));
                            double y_d = (double)data.y_denom[i];
                            double r_d = phi_denom;
                            grad[layout.log_phi_denom_idx] += r_d * (
                                R::digamma(y_d + r_d) - R::digamma(r_d)
                                + std::log(r_d) + 1.0
                                - std::log(mu_d + r_d) - (y_d + r_d) / (mu_d + r_d));
                        }
                        break;
                    }
                    case ModelType::NEGBIN_GAMMA: {
                        if (layout.has_phi_num) {
                            double r = phi_num;
                            double y = data.y_num[i];
                            grad[layout.log_phi_num_idx] += r * (
                                R::digamma(y + r) - R::digamma(r)
                                + std::log(r) + 1.0
                                - std::log(mu_num + r) - (y + r) / (mu_num + r));
                        }
                        if (layout.has_phi_denom) {
                            double mu_d = std::exp(std::min(eta_denom_i, 20.0));
                            double y_d = data.y_denom_cont[i];
                            double alpha = phi_denom;
                            grad[layout.log_phi_denom_idx] += alpha * (
                                std::log(alpha) + 1.0 - R::digamma(alpha)
                                + std::log(std::max(y_d, 1e-10)) - std::log(mu_d)
                                - y_d / mu_d);
                        }
                        break;
                    }
                    default:
                        break;
                }
            }
        }

        // A3: ICAR prior gradient w.r.t. log_tau (envelope: holding φ* fixed)
        // d/d(log_tau)[-0.5*tau*phi*'Q*phi* + 0.5*(S-1)*log(tau)]
        //   = -0.5*tau*phi*'Q*phi* + 0.5*(S-1)
        {
            std::vector<double> Qphi(S);
            icar_precision_matvec(collapsed_icar_ws().phi_star.data(), Qphi.data(), S,
                                  data.adj_row_ptr, data.adj_col_idx, data.n_neighbors);
            double phiQphi = 0.0;
            for (int s = 0; s < S; s++) phiQphi += collapsed_icar_ws().phi_star[s] * Qphi[s];
            grad[layout.log_tau_spatial_idx] += -0.5 * tau * phiQphi + 0.5 * (S - 1);
        }

        // === Part B: H-mode Laplace gradient ===
        auto laplace_result = compute_laplace_gradient_icar_H(
            collapsed_icar_ws(), beta_num, beta_denom,
            tau, phi_num, phi_denom,
            re_vals.empty() ? nullptr : re_vals.data(),
            data, layout, n_params, tulpa::s2z_precision(data.n_spatial_units),
            temporal_offset_num, temporal_offset_denom);

        if (laplace_result.success) {
            for (int j = 0; j < n_params; j++) {
                grad[j] += laplace_result.laplace_grad[j];
            }
        } else {
            // Fallback: numerical Laplace gradient for all params
            // (recomputes mode_lp + Laplace via central differences)
            auto collapsed_log_post = [&](const std::vector<double>& p) -> double {
                auto cp_l = extract_common_params(p, layout);
                double tau_l = std::exp(p[layout.log_tau_spatial_idx]);
                std::vector<double> re_vals_l;
                if (layout.has_re) {
                    int n_re = layout.re_end - layout.re_start;
                    re_vals_l.resize(n_re);
                    for (int g = 0; g < n_re; g++) {
                        re_vals_l[g] = (data.re_parameterization == 1)
                            ? cp_l.sigma_re * cp_l.re[g] : cp_l.re[g];
                    }
                }
                CollapsedTemporalOffset temporal_offset_l =
                    collapsed_temporal_obs_offset(data, layout, p);
                CollapsedICARWorkspace temp_ws;
                temp_ws.init(S, false);
                temp_ws.phi_star = collapsed_icar_ws().phi_star;
                temp_ws.mode_found = true;
                double mode_lp = collapsed_icar_find_mode(
                    cp_l.beta_num, cp_l.beta_denom, tau_l,
                    cp_l.phi_num, cp_l.phi_denom,
                    re_vals_l.empty() ? nullptr : re_vals_l.data(),
                    data, temp_ws, 20, 1e-6,
                    temporal_offset_l.num_ptr(), temporal_offset_l.denom_ptr());
                double lp = mode_lp + temp_ws.laplace_log_det;
                lp += (data.tau_spatial_shape - 1.0) * std::log(tau_l) - data.tau_spatial_rate * tau_l
                      + p[layout.log_tau_spatial_idx];
                return lp;
            };
            const double eps = 1e-5;
            std::vector<double> params_pert = params;
            for (int j = 0; j < n_params; j++) {
                double orig = params_pert[j];
                params_pert[j] = orig + eps;
                double lp_plus = collapsed_log_post(params_pert);
                params_pert[j] = orig - eps;
                double lp_minus = collapsed_log_post(params_pert);
                grad[j] += (lp_plus - lp_minus) / (2.0 * eps);
                params_pert[j] = orig;
            }
        }

        // === Part C: Spatial hyperparameter prior gradient (analytical) ===
        // Gamma(shape, rate) prior on tau, on log scale:
        // d/d(log_tau)[(shape-1)*log(tau) - rate*tau + log_tau]
        //   = (shape-1) - rate*tau + 1
        grad[layout.log_tau_spatial_idx] += (data.tau_spatial_shape - 1.0)
                                           - data.tau_spatial_rate * tau + 1.0;

    } else {
        // ---- BYM2: H-mode analytical gradient ----
        // Same structure as ICAR: envelope + analytical Laplace + outer priors

        // === Part A: Envelope theorem gradient ===
        // A1: Data LL gradient via residual scattering
        collapsed_icar_compute_residuals(
            collapsed_icar_ws(), beta_num, beta_denom,
            phi_num, phi_denom,
            re_vals.empty() ? nullptr : re_vals.data(),
            a, c_bym2,  // BYM2 scaling
            data, resid_num.data(), resid_denom.data(),
            temporal_offset_num, temporal_offset_denom);

        // Scatter residuals to beta_num
        for (int k = 0; k < data.p_num; k++) {
            double sum = 0.0;
            for (int i = 0; i < N; i++)
                sum += resid_num[i] * data.X_num_flat[i * data.p_num + k];
            grad[layout.beta_num_start + k] += sum;
        }
        // Scatter to beta_denom
        if (!is_binomial) {
            for (int k = 0; k < data.p_denom; k++) {
                double sum = 0.0;
                for (int i = 0; i < N; i++)
                    sum += resid_denom[i] * data.X_denom_flat[i * data.p_denom + k];
                grad[layout.beta_denom_start + k] += sum;
            }
        }
        // Scatter to RE
        if (layout.has_re) {
            int n_re = layout.re_end - layout.re_start;
            for (int i = 0; i < N; i++) {
                if (data.re_group.size() > (size_t)i && data.re_group[i] > 0) {
                    int g = data.re_group[i] - 1;
                    if (g < n_re) {
                        grad[layout.re_start + g] += resid_num[i] + resid_denom[i];
                    }
                }
            }
        }

        // A2: Dispersion parameter gradients from data LL at mode (envelope)
        if (layout.has_phi_num || layout.has_phi_denom) {
            for (int i = 0; i < N; i++) {
                int s = data.spatial_group[i] - 1;
                double b_s = a * collapsed_icar_ws().phi_star[s]
                           + c_bym2 * collapsed_icar_ws().theta_star[s];
                double eta_num_i = 0.0, eta_denom_i = 0.0;
                for (int p = 0; p < data.p_num; p++)
                    eta_num_i += data.X_num_flat[i * data.p_num + p] * beta_num[p];
                if (!is_binomial) {
                    for (int p = 0; p < data.p_denom; p++)
                        eta_denom_i += data.X_denom_flat[i * data.p_denom + p] * beta_denom[p];
                }
                eta_num_i += b_s;
                if (!is_binomial) eta_denom_i += b_s;
                if (re_vals.data() && data.re_group.size() > (size_t)i && data.re_group[i] > 0) {
                    eta_num_i += re_vals[data.re_group[i] - 1];
                    if (!is_binomial) eta_denom_i += re_vals[data.re_group[i] - 1];
                }
                if (temporal_offset_num != nullptr) eta_num_i += temporal_offset_num[i];
                if (!is_binomial && temporal_offset_denom != nullptr) eta_denom_i += temporal_offset_denom[i];
                double mu_num = std::exp(std::min(eta_num_i, 20.0));

                switch (data.model_type) {
                    case ModelType::POISSON_GAMMA: {
                        if (layout.has_phi_denom) {
                            double mu_d = std::exp(std::min(eta_denom_i, 20.0));
                            double y_d = data.y_denom_cont[i];
                            double alpha = phi_denom;
                            grad[layout.log_phi_denom_idx] += alpha * (
                                std::log(alpha) + 1.0 - R::digamma(alpha)
                                + std::log(std::max(y_d, 1e-10)) - std::log(mu_d)
                                - y_d / mu_d);
                        }
                        break;
                    }
                    case ModelType::NEGBIN_NEGBIN: {
                        if (layout.has_phi_num) {
                            double r = phi_num;
                            double y = data.y_num[i];
                            grad[layout.log_phi_num_idx] += r * (
                                R::digamma(y + r) - R::digamma(r)
                                + std::log(r) + 1.0
                                - std::log(mu_num + r) - (y + r) / (mu_num + r));
                        }
                        if (layout.has_phi_denom && !is_binomial) {
                            double mu_d = std::exp(std::min(eta_denom_i, 20.0));
                            double y_d = (double)data.y_denom[i];
                            double r_d = phi_denom;
                            grad[layout.log_phi_denom_idx] += r_d * (
                                R::digamma(y_d + r_d) - R::digamma(r_d)
                                + std::log(r_d) + 1.0
                                - std::log(mu_d + r_d) - (y_d + r_d) / (mu_d + r_d));
                        }
                        break;
                    }
                    case ModelType::NEGBIN_GAMMA: {
                        if (layout.has_phi_num) {
                            double r = phi_num;
                            double y = data.y_num[i];
                            grad[layout.log_phi_num_idx] += r * (
                                R::digamma(y + r) - R::digamma(r)
                                + std::log(r) + 1.0
                                - std::log(mu_num + r) - (y + r) / (mu_num + r));
                        }
                        if (layout.has_phi_denom) {
                            double mu_d = std::exp(std::min(eta_denom_i, 20.0));
                            double y_d = data.y_denom_cont[i];
                            double alpha = phi_denom;
                            grad[layout.log_phi_denom_idx] += alpha * (
                                std::log(alpha) + 1.0 - R::digamma(alpha)
                                + std::log(std::max(y_d, 1e-10)) - std::log(mu_d)
                                - y_d / mu_d);
                        }
                        break;
                    }
                    default:
                        break;
                }
            }
        }

        // A3: BYM2 prior envelope gradients for log_sigma and logit_rho
        // (envelope: holding phi*, theta* fixed, differentiate data LL through b_s)
        {
            double grad_sigma_env = 0.0;
            double grad_rho_env = 0.0;
            double da_drho = a * (1.0 - rho) / 2.0;
            double dc_drho = -c_bym2 * rho / 2.0;
            for (int s = 0; s < S; s++) {
                // Per-site residual sum (from residuals already computed)
                double r_sum = 0.0;
                for (int i = 0; i < N; i++) {
                    if (data.spatial_group[i] - 1 == s)
                        r_sum += resid_num[i] + resid_denom[i];
                }
                double b_s = a * collapsed_icar_ws().phi_star[s]
                           + c_bym2 * collapsed_icar_ws().theta_star[s];
                grad_sigma_env += r_sum * b_s;  // d/d(log_sigma) = b_s
                double d_rho_s = da_drho * collapsed_icar_ws().phi_star[s]
                               + dc_drho * collapsed_icar_ws().theta_star[s];
                grad_rho_env += r_sum * d_rho_s;
            }
            grad[layout.log_sigma_bym2_idx] += grad_sigma_env;
            grad[layout.logit_rho_bym2_idx] += grad_rho_env;
        }

        // === Part B: H-mode Laplace gradient ===
        auto laplace_result = compute_laplace_gradient_bym2_H(
            collapsed_icar_ws(), beta_num, beta_denom,
            a, c_bym2, rho,
            phi_num, phi_denom,
            re_vals.empty() ? nullptr : re_vals.data(),
            data, layout, n_params, tulpa::s2z_precision(data.n_spatial_units),
            temporal_offset_num, temporal_offset_denom);

        if (laplace_result.success) {
            for (int j = 0; j < n_params; j++) {
                grad[j] += laplace_result.laplace_grad[j];
            }
        } else {
            // Numerical fallback for Laplace gradient only
            auto laplace_only = [&](const std::vector<double>& p) -> double {
                auto cp_l = extract_common_params(p, layout);
                double sigma_l = std::exp(p[layout.log_sigma_bym2_idx]);
                double logit_rho_l = p[layout.logit_rho_bym2_idx];
                double rho_l = 1.0 / (1.0 + std::exp(-logit_rho_l));
                double a_l = sigma_l * std::sqrt(rho_l) * data.bym2_scale_factor;
                double c_l = sigma_l * std::sqrt(1.0 - rho_l);
                std::vector<double> re_vals_l;
                if (layout.has_re) {
                    int n_re = layout.re_end - layout.re_start;
                    re_vals_l.resize(n_re);
                    for (int g = 0; g < n_re; g++) {
                        re_vals_l[g] = (data.re_parameterization == 1)
                            ? cp_l.sigma_re * cp_l.re[g] : cp_l.re[g];
                    }
                }
                CollapsedTemporalOffset temporal_offset_l =
                    collapsed_temporal_obs_offset(data, layout, p);
                CollapsedICARWorkspace temp_ws;
                temp_ws.init(S, true);
                temp_ws.phi_star = collapsed_icar_ws().phi_star;
                temp_ws.theta_star = collapsed_icar_ws().theta_star;
                temp_ws.mode_found = true;
                collapsed_bym2_find_mode(
                    cp_l.beta_num, cp_l.beta_denom, sigma_l, rho_l, data.bym2_scale_factor,
                    cp_l.phi_num, cp_l.phi_denom,
                    re_vals_l.empty() ? nullptr : re_vals_l.data(),
                    data, temp_ws, 20, 1e-6,
                    temporal_offset_l.num_ptr(), temporal_offset_l.denom_ptr());
                return temp_ws.laplace_log_det;
            };
            const double eps = 1e-5;
            std::vector<double> params_pert = params;
            for (int j = 0; j < n_params; j++) {
                double orig = params_pert[j];
                params_pert[j] = orig + eps;
                double ld_plus = laplace_only(params_pert);
                params_pert[j] = orig - eps;
                double ld_minus = laplace_only(params_pert);
                grad[j] += (ld_plus - ld_minus) / (2.0 * eps);
                params_pert[j] = orig;
            }
        }

        // === Part C: BYM2 hyperparameter prior gradients ===
        // Sigma prior: half-Cauchy via d/d(log_sigma)[-log(1 + (sigma/s)^2) + log_sigma]
        {
            double ratio_sc = sigma_total / data.sigma_re_scale;
            double r2 = ratio_sc * ratio_sc;
            grad[layout.log_sigma_bym2_idx] += -2.0 * r2 / (1.0 + r2) + 1.0;
        }
        // Rho prior: Uniform(0,1) Jacobian: log(rho) + log(1-rho)
        // d/d(logit_rho) = d(log(rho))/d(logit_rho) + d(log(1-rho))/d(logit_rho)
        //                = (1-rho) + (-rho) = 1 - 2*rho
        grad[layout.logit_rho_bym2_idx] += 1.0 - 2.0 * rho;
    }

    // ---- Companion temporal GMRF: gradient + prior ----
    // resid_num/resid_denom above are computed at the correct (temporal-
    // inclusive) eta, so scattering them into the temporal likelihood
    // gradient is the same envelope-theorem step as the beta/RE scatter.
    if (collapsed_temporal_offset.num.size() == (size_t)N) {
        int T_temporal = layout.temporal_end - layout.temporal_start;
        double tau_temporal = std::exp(params[layout.log_tau_temporal_idx]);
        double rho_ar1 = read_temporal_rho_ar1(params, data, layout);
        // tau_temporal_prior_grad assigns (=) rather than accumulates, so it
        // has to run before temporal_gmrf_prior_grad's += -- same order as
        // every other call site (e.g. compute_gradient_composite).
        tau_temporal_prior_grad(data, layout, tau_temporal, grad.data());
        temporal_rho_prior_grad(data, layout, rho_ar1, grad.data());

        // Seed with the Laplace log-det's own temporal gradient (already added
        // into grad[] by the Part B loop above -- log det(H) depends on eta,
        // hence on the temporal offset, through the curvature W_data, not just
        // through phi*'s value). temporal_gmrf_prior_grad assigns (=) rather
        // than accumulates when it writes the likelihood term, so that
        // contribution has to be folded in here first or it gets overwritten.
        std::vector<double> grad_temporal_lik(T_temporal, 0.0);
        for (int t = 0; t < T_temporal; t++) {
            grad_temporal_lik[t] = grad[layout.temporal_start + t];
        }
        for (int i = 0; i < N; i++) {
            if (data.temporal_time_idx.empty() || data.temporal_time_idx[i] <= 0) continue;
            int t = data.temporal_time_idx[i] - 1;
            int g = data.temporal_group_idx[i] - 1;
            int t_base = g * data.n_times + t;
            if (t_base < 0 || t_base >= T_temporal) continue;
            grad_temporal_lik[t_base] += data.temporal_shared ? (resid_num[i] + resid_denom[i]) : resid_num[i];
        }
        TemporalView tview;
        tview.read(params, data, layout);
        temporal_gmrf_prior_grad(data, layout, tau_temporal, rho_ar1,
                                 tview, T_temporal,
                                 grad_temporal_lik.data(), grad.data());
    }

    // ---- NC transform for RE ----
    re_gradient_nc_transform(data, layout, params.data(), grad.data(), sigma_re);

    // ---- Log-posterior ----
    // A companion temporal term adds prior mass (temporal_gmrf_prior_grad above
    // has no matching log-density helper), so fall back to the authoritative
    // compute_log_post rather than hand-derive that term a second time here.
    if (log_post_out && collapsed_temporal_offset.num.size() == (size_t)N) {
        *log_post_out = compute_log_post(params, data, layout);
    } else if (log_post_out) {
        double lp = collapsed_lp + collapsed_icar_ws().laplace_log_det;

        // Beta priors
        for (int p = 0; p < data.p_num; p++)
            lp += -0.5 * beta_num[p] * beta_num[p] / (data.sigma_beta * data.sigma_beta);
        for (int p = 0; p < data.p_denom; p++)
            lp += -0.5 * beta_denom[p] * beta_denom[p] / (data.sigma_beta * data.sigma_beta);

        // RE priors
        if (layout.has_re) {
            int n_re = layout.re_end - layout.re_start;
            double sigma_re2 = sigma_re * sigma_re;
            for (int g = 0; g < n_re; g++) {
                if (data.re_parameterization == 1) {
                    lp += -0.5 * re[g] * re[g];
                } else {
                    lp += -0.5 * re[g] * re[g] / sigma_re2;
                }
            }
            double ratio_hc = sigma_re / data.sigma_re_scale;
            lp += -std::log(1.0 + ratio_hc * ratio_hc) + params[layout.log_sigma_re_idx];
        }

        // Spatial hyperparameter priors
        if (is_bym2) {
            double ratio_sc = sigma_total / data.sigma_re_scale;
            lp += -std::log(1.0 + ratio_sc * ratio_sc) + params[layout.log_sigma_bym2_idx];
            lp += std::log(rho) + std::log(1.0 - rho);
        } else {
            lp += (data.tau_spatial_shape - 1.0) * std::log(tau) - data.tau_spatial_rate * tau
                  + params[layout.log_tau_spatial_idx];
        }

        // Phi (dispersion) priors
        if (layout.has_phi_num) lp += params[layout.log_phi_num_idx];
        if (layout.has_phi_denom) lp += params[layout.log_phi_denom_idx];

        *log_post_out = lp;
    }
}

// =====================================================================
// GP + Temporal gradient (hand-coded)
// Combines GP spatial with temporal RW1/RW2/AR1
// =====================================================================

void compute_gradient_gp_temporal_handcoded(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    std::vector<double>& grad,
    double* log_post_out = nullptr
) {
    const bool fuse_lp = (log_post_out != nullptr) && !layout.has_zi;
    if (log_post_out && layout.has_zi) *log_post_out = compute_log_post(params, data, layout);
    double obs_log_lik = 0.0;
    int n_params = params.size();
    grad.assign(n_params, 0.0);

    auto cp = extract_common_params(params, layout);
    const double* beta_num = cp.beta_num;
    const double* beta_denom = cp.beta_denom;
    double sigma_re = cp.sigma_re;
    const double* re = cp.re;
    double phi_num = cp.phi_num;
    double phi_denom = cp.phi_denom;

    int N_gp = data.gp_data.n_obs;
    double sigma2_gp = std::exp(params[layout.log_sigma2_gp_idx]);
    double phi_gp = std::exp(params[layout.log_phi_gp_idx]);
    std::vector<double> gp_w(N_gp);
    for (int i = 0; i < N_gp; i++) gp_w[i] = params[layout.gp_w_start + i];

    double tau_temporal = std::exp(params[layout.log_tau_temporal_idx]);
    int T_len = layout.temporal_end - layout.temporal_start;
    TemporalView tview;
    tview.read(params, data, layout);
    const double* phi_temporal = tview.phi;
    double rho_ar1 = read_temporal_rho_ar1(params, data, layout);

    // Prior gradients
    beta_gradient_prior(data, layout, beta_num, beta_denom, grad.data());
    re_gradient_prior(data, layout, re, grad.data(), sigma_re);
    phi_gradient_prior(data, layout, phi_num, phi_denom, grad.data());

    grad[layout.log_sigma2_gp_idx] = gp_pc_prior_grad_log_sigma2(
        sigma2_gp, data.gp_sigma2_prior_U, data.gp_sigma2_prior_alpha);
    grad[layout.log_phi_gp_idx] = 1.0;
    tau_temporal_prior_grad(data, layout, tau_temporal, grad.data());
    temporal_rho_prior_grad(data, layout, rho_ar1, grad.data());

    // NNGP prior gradients
    ratiod_gp::NNGPGradients nngp_grads;
    ratiod_gp::gp_nngp_gradients(gp_w, sigma2_gp, phi_gp, data.gp_data, nngp_grads);
    for (int i = 0; i < N_gp; i++) grad[layout.gp_w_start + i] += nngp_grads.grad_w[i];
    grad[layout.log_sigma2_gp_idx] += nngp_grads.grad_log_sigma2;
    grad[layout.log_phi_gp_idx] += nngp_grads.grad_log_phi;

    // Likelihood (vectorized eta + per-obs scatter)
    const int N = data.N;
    const bool is_binomial = (data.model_type == ModelType::BINOMIAL ||
                              data.model_type == ModelType::BETA_BINOMIAL);
    std::vector<double> grad_temporal_lik(T_len, 0.0);
    std::vector<int> obs_t_idx(N, -1);

    using RowMajorMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using VectorXd = Eigen::VectorXd;
    vec_grad_ws().init(N);
    Eigen::Map<const RowMajorMatrix> X_num(data.X_num_flat.data(), N, data.p_num);
    Eigen::Map<const VectorXd> b_num(beta_num, data.p_num);
    Eigen::Map<VectorXd> eta_n(vec_grad_ws().eta_num.data(), N);
    eta_n.noalias() = X_num * b_num;
    if (!is_binomial) {
        Eigen::Map<const RowMajorMatrix> X_denom(data.X_denom_flat.data(), N, data.p_denom);
        Eigen::Map<const VectorXd> b_denom(beta_denom, data.p_denom);
        Eigen::Map<VectorXd> eta_d(vec_grad_ws().eta_denom.data(), N);
        eta_d.noalias() = X_denom * b_denom;
    } else {
        std::memset(vec_grad_ws().eta_denom.data(), 0, N * sizeof(double));
    }

    for (int i = 0; i < N; i++) {
        if (layout.has_re && data.re_group[i] > 0) {
            double re_eff = re_value_for_eta(re, data.re_group[i] - 1, sigma_re, data.re_parameterization);
            vec_grad_ws().eta_num[i] += re_eff;
            if (!is_binomial) vec_grad_ws().eta_denom[i] += re_eff;
        }
        int loc_i = data.gp_data.obs_to_loc[i];
        double gp_effect = gp_w[loc_i];
        vec_grad_ws().eta_num[i] += gp_effect;
        if (!is_binomial && data.gp_data.shared) vec_grad_ws().eta_denom[i] += gp_effect;

        if (!data.temporal_time_idx.empty() && i < (int)data.temporal_time_idx.size() && data.temporal_time_idx[i] > 0) {
            int t = data.temporal_time_idx[i] - 1;
            int g = data.temporal_group_idx[i] - 1;
            int t_idx = g * data.n_times + t;
            if (t_idx >= 0 && t_idx < T_len) {
                obs_t_idx[i] = t_idx;
                vec_grad_ws().eta_num[i] += phi_temporal[t_idx];
                if (!is_binomial && data.temporal_shared) vec_grad_ws().eta_denom[i] += phi_temporal[t_idx];
            }
        }
    }

    {
        double grad_phi_num_lik = 0.0, grad_phi_denom_lik = 0.0;
        vectorized::dispatch_residuals_and_beta_grads(
            data, layout,
            vec_grad_ws().eta_num.data(), vec_grad_ws().eta_denom.data(),
            vec_grad_ws().resid_num.data(), vec_grad_ws().resid_denom.data(),
            grad.data(), grad_phi_num_lik, grad_phi_denom_lik,
            obs_log_lik, fuse_lp, phi_num, phi_denom, vec_grad_ws());
    }

    for (int i = 0; i < N; i++) {
        double dLL_num = vec_grad_ws().resid_num[i];
        double dLL_denom = vec_grad_ws().resid_denom[i];
        if (layout.has_re && data.re_group[i] > 0) {
            grad[layout.re_start + data.re_group[i] - 1] += dLL_num + dLL_denom;
        }
        int loc_i = data.gp_data.obs_to_loc[i];
        grad[layout.gp_w_start + loc_i] += data.gp_data.shared ? (dLL_num + dLL_denom) : dLL_num;
        if (obs_t_idx[i] >= 0) grad_temporal_lik[obs_t_idx[i]] += data.temporal_shared ? (dLL_num + dLL_denom) : dLL_num;
    }

    // Temporal GMRF gradients
    temporal_gmrf_prior_grad(data, layout, tau_temporal, rho_ar1,
                             tview, T_len, grad_temporal_lik.data(), grad.data());

    // Non-centered RE chain rule transformation
    re_gradient_nc_transform(data, layout, params.data(), grad.data(), sigma_re);

    if (fuse_lp) *log_post_out = compute_log_post(params, data, layout, /*skip_obs_loop=*/true) + obs_log_lik;
}

// =====================================================================
// Temporal GP (standalone) hand-coded gradients
// Temporal GP with exponential covariance uses state-space AR(1) form
// =====================================================================

void compute_gradient_temporal_gp_handcoded(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    std::vector<double>& grad,
    double* log_post_out = nullptr
) {
    // Fused log-posterior: accumulate obs log-lik during gradient loop,
    // then add prior/structural terms via skip_obs_loop=true (avoids 2nd O(N) pass).
    // Also fuse temporal GP prior to avoid redundant NC forward pass in compute_log_post.
    const bool fuse_lp = (log_post_out != nullptr) && !layout.has_zi;
    if (log_post_out && layout.has_zi) *log_post_out = compute_log_post(params, data, layout);
    double obs_log_lik = 0.0;
    double tgp_lp_accum = 0.0;  // Accumulates temporal GP prior terms
    int n_params = params.size();
    grad.assign(n_params, 0.0);

    // Extract common parameters
    auto cp = extract_common_params(params, layout);
    const double* beta_num = cp.beta_num;
    const double* beta_denom = cp.beta_denom;
    double sigma_re = cp.sigma_re;
    const double* re = cp.re;
    double phi_num = cp.phi_num;
    double phi_denom = cp.phi_denom;

    // Temporal GP hyperparameters
    double sigma2_tgp = std::exp(params[layout.log_sigma2_temporal_gp_idx]);
    double logit_phi_val = params[layout.logit_phi_temporal_gp_idx];

    // Logit-bounded phi: phi = lower + range * sigmoid(logit_phi)
    double phi_lower = data.temporal_gp_phi_prior_lower;
    double phi_upper = data.temporal_gp_phi_prior_upper;
    double phi_range = phi_upper - phi_lower;
    double sigmoid_val = 1.0 / (1.0 + std::exp(-logit_phi_val));
    double phi_tgp = phi_lower + phi_range * sigmoid_val;

    // Conversion factor: grad_logit = grad_log * chi
    // where chi = (phi - lower)(upper - phi) / (phi * range)
    double chi_tgp = (phi_tgp - phi_lower) * (phi_upper - phi_tgp) / (phi_tgp * phi_range);

    // Temporal effects: n_temporal_groups * n_times parameters
    int T_times = data.n_times;
    int n_groups = data.n_temporal_groups;
    const double* phi_temporal = &params[layout.temporal_start];
    int T_len = layout.temporal_end - layout.temporal_start;

    // Non-centered parameterization: params store z ~ N(0,1), reconstruct f
    const bool use_nc = (data.temporal_gp_parameterization == 1);
    RATIOD_TLS_WORKSPACE(ratiod_temporal_gp::TemporalGPNCWorkspace, nc_ws);
    const double* f_temporal = phi_temporal;  // Default: centered, f stored directly

    if (use_nc) {
        nc_ws.init(T_times, n_groups);
        ratiod_temporal_gp::temporal_gp_nc_forward(
            phi_temporal, T_times, n_groups,
            sigma2_tgp, phi_tgp,
            data.temporal_gp_data.time_values, nc_ws);
        f_temporal = nc_ws.f.data();  // Use reconstructed f for eta
        std::memset(nc_ws.dL_df.data(), 0, T_len * sizeof(double));
    }

    // ---- Prior gradients ----
    beta_gradient_prior(data, layout, beta_num, beta_denom, grad.data());
    re_gradient_prior(data, layout, re, grad.data(), sigma_re);
    phi_gradient_prior(data, layout, phi_num, phi_denom, grad.data());

    // Temporal GP hyperparameter priors
    // sigma2: PC prior => d/d(log_sigma2) [ log(rate) - rate*sqrt(sigma2) - log(2*sqrt(sigma2)) + log_sigma2 ]
    //       = -0.5 * rate * sqrt(sigma2) + 0.5
    double sigma_tgp = std::sqrt(sigma2_tgp);
    double rate_tgp = -std::log(data.temporal_gp_sigma2_prior_alpha) / data.temporal_gp_sigma2_prior_U;
    grad[layout.log_sigma2_temporal_gp_idx] = -0.5 * rate_tgp * sigma_tgp + 0.5;

    // phi: Logit-bounded Jacobian gradient
    // Jacobian = log(phi - lower) + log(upper - phi) - log(range)
    // d/d(logit_phi) = (upper + lower - 2*phi) / range
    grad[layout.logit_phi_temporal_gp_idx] = (phi_upper + phi_lower - 2.0 * phi_tgp) / phi_range;

    if (use_nc) {
        // ---- NC prior: z ~ N(0, I) ----
        // Jacobian of transform f = g(z, sigma2, phi):
        //   log|det(df/dz)| = T*log(sigma) + 0.5*sum_{t>=1} log(1-rho_t^2)
        // d/d(log_sigma2) of Jacobian = T/2 per group
        grad[layout.log_sigma2_temporal_gp_idx] += 0.5 * T_times * n_groups;

        // d/d(log_phi) of Jacobian = -sum rho^2*(dt/phi) / (1-rho^2)
        // Convert to logit scale: multiply by chi
        double jac_phi_log = 0.0;
        for (int t = 1; t < T_times; t++) {
            double rho_t = nc_ws.rho[t - 1];
            double rho2 = rho_t * rho_t;
            double dt = data.temporal_gp_data.time_values[t] - data.temporal_gp_data.time_values[t - 1];
            double dt_over_phi = dt / phi_tgp;
            double one_minus_rho2 = ratiod_ar1::one_minus_rho2(rho_t);
            jac_phi_log -= rho2 * dt_over_phi / one_minus_rho2;
        }
        grad[layout.logit_phi_temporal_gp_idx] += jac_phi_log * n_groups * chi_tgp;

        // z prior: d/dz = -z (each z ~ N(0,1))
        for (int t = 0; t < T_len; t++) {
            grad[layout.temporal_start + t] = -phi_temporal[t];
        }

        // Fuse temporal GP prior log-prob (avoids redundant NC forward pass in compute_log_post)
        if (fuse_lp) {
            double log_sigma2 = params[layout.log_sigma2_temporal_gp_idx];
            // sigma2 PC prior + Jacobian
            tgp_lp_accum += std::log(rate_tgp) - rate_tgp * sigma_tgp
                          - std::log(2.0 * sigma_tgp) + log_sigma2;
            // phi logit Jacobian
            tgp_lp_accum += std::log(phi_tgp - phi_lower)
                          + std::log(phi_upper - phi_tgp)
                          - std::log(phi_range);
            // NC Jacobian: T*log(sigma) + 0.5*sum log(1-rho^2)  per group
            double nc_jac = T_times * std::log(sigma_tgp);
            for (int t = 1; t < T_times; t++) {
                double one_m_rho2 = ratiod_ar1::one_minus_rho2(nc_ws.rho[t-1]);
                nc_jac += 0.5 * std::log(one_m_rho2);
            }
            tgp_lp_accum += nc_jac * n_groups;
            // z ~ N(0,I) prior
            for (int t = 0; t < T_len; t++) {
                tgp_lp_accum += -0.5 * phi_temporal[t] * phi_temporal[t];
            }
        }
    } else {
        // ---- Centered: temporal GP prior gradients (state-space exponential form) ----
        for (int g = 0; g < n_groups; g++) {
            int offset = g * T_times;

            double f0 = phi_temporal[offset];
            grad[layout.temporal_start + offset] += -f0 / sigma2_tgp;

            double grad_log_sigma2_prior = -0.5 + 0.5 * f0 * f0 / sigma2_tgp;
            double grad_log_phi_prior = 0.0;

            for (int t = 1; t < T_times; t++) {
                double dt = data.temporal_gp_data.time_values[t] - data.temporal_gp_data.time_values[t - 1];
                double rho = std::exp(-dt / phi_tgp);
                double rho2 = rho * rho;
                double cv = sigma2_tgp * ratiod_ar1::one_minus_rho2(rho);

                double f_prev = phi_temporal[offset + t - 1];
                double f_curr = phi_temporal[offset + t];
                double r = f_curr - rho * f_prev;

                grad[layout.temporal_start + offset + t] += -r / cv;
                grad[layout.temporal_start + offset + t - 1] += rho * r / cv;

                grad_log_sigma2_prior += -0.5 + 0.5 * r * r / cv;

                double dt_over_phi = dt / phi_tgp;
                grad_log_phi_prior += dt_over_phi * (
                    sigma2_tgp * rho2 / cv
                    + rho * r * f_prev / cv
                    + sigma2_tgp * rho2 * r * r / (cv * cv)
                );
            }

            grad[layout.log_sigma2_temporal_gp_idx] += grad_log_sigma2_prior;
            grad[layout.logit_phi_temporal_gp_idx] += grad_log_phi_prior * chi_tgp;
        }
    }

    // ---- Likelihood gradients (vectorized) ----
    // Thread-local buffer avoids heap allocation per gradient call
    RATIOD_TLS_WORKSPACE(std::vector<double>, grad_temporal_lik);
    grad_temporal_lik.assign(T_len, 0.0);

    const int N = data.N;
    const bool is_binomial = (data.model_type == ModelType::BINOMIAL ||
                              data.model_type == ModelType::BETA_BINOMIAL);

    // --- Pass 1: Vectorized eta computation (Eigen matvec + scalar expansion) ---
    using RowMajorMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using VectorXd = Eigen::VectorXd;

    vec_grad_ws().init(N);

    Eigen::Map<const RowMajorMatrix> X_num(data.X_num_flat.data(), N, data.p_num);
    Eigen::Map<const VectorXd> b_num(beta_num, data.p_num);
    Eigen::Map<VectorXd> eta_n(vec_grad_ws().eta_num.data(), N);
    eta_n.noalias() = X_num * b_num;

    if (!is_binomial) {
        Eigen::Map<const RowMajorMatrix> X_denom(data.X_denom_flat.data(), N, data.p_denom);
        Eigen::Map<const VectorXd> b_denom(beta_denom, data.p_denom);
        Eigen::Map<VectorXd> eta_d(vec_grad_ws().eta_denom.data(), N);
        eta_d.noalias() = X_denom * b_denom;
    }

    // Add RE to eta
    if (layout.has_re) {
        for (int i = 0; i < N; i++) {
            if (data.re_group[i] > 0) {
                int g = data.re_group[i] - 1;
                double re_eff = re_value_for_eta(re, g, sigma_re, data.re_parameterization);
                vec_grad_ws().eta_num[i] += re_eff;
                if (!is_binomial) vec_grad_ws().eta_denom[i] += re_eff;
            }
        }
    }

    // Add temporal GP effect to eta
    for (int i = 0; i < N; i++) {
        if (!data.temporal_time_idx.empty() && data.temporal_time_idx[i] > 0) {
            int t = data.temporal_time_idx[i] - 1;
            int g = (i < (int)data.temporal_group_idx.size() && data.temporal_group_idx[i] > 0)
                    ? data.temporal_group_idx[i] - 1 : 0;
            int flat_idx = g * T_times + t;
            if (flat_idx >= 0 && flat_idx < T_len) {
                vec_grad_ws().eta_num[i] += f_temporal[flat_idx];
                if (!is_binomial && data.temporal_shared) vec_grad_ws().eta_denom[i] += f_temporal[flat_idx];
            }
        }
    }

    // --- Pass 2+3: Vectorized residuals + beta grads (template-dispatched) ---
    {
        double grad_phi_num_lik = 0.0, grad_phi_denom_lik = 0.0;
        vectorized::dispatch_residuals_and_beta_grads(
            data, layout,
            vec_grad_ws().eta_num.data(), vec_grad_ws().eta_denom.data(),
            vec_grad_ws().resid_num.data(), vec_grad_ws().resid_denom.data(),
            grad.data(), grad_phi_num_lik, grad_phi_denom_lik,
            obs_log_lik, fuse_lp, phi_num, phi_denom, vec_grad_ws());
    }

    // Scatter residuals to RE gradient
    if (layout.has_re) {
        for (int i = 0; i < N; i++) {
            if (data.re_group[i] > 0)
                grad[layout.re_start + data.re_group[i] - 1] += vec_grad_ws().resid_num[i] + vec_grad_ws().resid_denom[i];
        }
    }

    // Scatter residuals to temporal lik gradient
    for (int i = 0; i < N; i++) {
        if (!data.temporal_time_idx.empty() && data.temporal_time_idx[i] > 0) {
            int t = data.temporal_time_idx[i] - 1;
            int g = (i < (int)data.temporal_group_idx.size() && data.temporal_group_idx[i] > 0)
                    ? data.temporal_group_idx[i] - 1 : 0;
            int flat_idx = g * T_times + t;
            if (flat_idx >= 0 && flat_idx < T_len)
                grad_temporal_lik[flat_idx] += data.temporal_shared
                    ? (vec_grad_ws().resid_num[i] + vec_grad_ws().resid_denom[i])
                    : vec_grad_ws().resid_num[i];
        }
    }

    if (use_nc) {
        // NC: backward pass converts dL/df -> dL/dz and accumulates sigma2/phi grads
        std::memcpy(nc_ws.dL_df.data(), grad_temporal_lik.data(), T_len * sizeof(double));

        double grad_log_sigma2_lik = 0.0, grad_log_phi_lik_tgp = 0.0;
        // Write z gradients directly to grad[] (backward uses = not +=, safe)
        ratiod_temporal_gp::temporal_gp_nc_backward(
            phi_temporal, T_times, n_groups,
            sigma2_tgp, phi_tgp,
            data.temporal_gp_data.time_values,
            nc_ws, &grad[layout.temporal_start],
            grad_log_sigma2_lik, grad_log_phi_lik_tgp);
        grad[layout.log_sigma2_temporal_gp_idx] += grad_log_sigma2_lik;
        grad[layout.logit_phi_temporal_gp_idx] += grad_log_phi_lik_tgp * chi_tgp;
    } else {
        // Centered: add likelihood contribution to temporal effects directly
        for (int t = 0; t < T_len; t++) {
            grad[layout.temporal_start + t] += grad_temporal_lik[t];
        }
    }

    // Phi gradients already applied by dispatch_residuals_and_beta_grads() (log-transform Jacobian included)

    // Non-centered RE chain rule transformation
    re_gradient_nc_transform(data, layout, params.data(), grad.data(), sigma_re);

    if (fuse_lp) {
        if (use_nc && tgp_lp_accum != 0.0) {
            // Fused: temporal GP prior already accumulated, skip in compute_log_post
            *log_post_out = compute_log_post(params, data, layout, /*skip_obs_loop=*/true,
                                             &tgp_lp_accum) + obs_log_lik;
        } else {
            // Centered or no accumulation: let compute_log_post do the temporal GP prior
            *log_post_out = compute_log_post(params, data, layout, /*skip_obs_loop=*/true) + obs_log_lik;
        }
    }
}

// =====================================================================
// Multi-scale GP + Temporal hand-coded gradients
// Combines MSGP spatial gradients with temporal GMRF gradients
// =====================================================================

void compute_gradient_msgp_temporal_handcoded(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    std::vector<double>& grad,
    double* log_post_out = nullptr
) {
    // Fused log-posterior: accumulate obs log-lik during gradient loop,
    // then add prior/structural terms via skip_obs_loop=true (avoids 2nd O(N) pass)
    const bool fuse_lp = (log_post_out != nullptr) && !layout.has_zi;
    if (log_post_out && layout.has_zi) *log_post_out = compute_log_post(params, data, layout);
    double obs_log_lik = 0.0;
    int n_params = params.size();
    grad.assign(n_params, 0.0);

    // Extract common parameters
    auto cp = extract_common_params(params, layout);
    const double* beta_num = cp.beta_num;
    const double* beta_denom = cp.beta_denom;
    double sigma_re = cp.sigma_re;
    const double* re = cp.re;
    double phi_num = cp.phi_num;
    double phi_denom = cp.phi_denom;

    // Multi-scale GP parameters
    int N_gp = data.multiscale_gp_data.n_obs;
    double sigma2_local = std::exp(params[layout.log_sigma2_gp_local_idx]);
    double phi_local = std::exp(params[layout.log_phi_gp_local_idx]);
    double sigma2_regional = std::exp(params[layout.log_sigma2_gp_regional_idx]);
    double phi_regional = std::exp(params[layout.log_phi_gp_regional_idx]);

    // Temporal parameters
    double tau_temporal = std::exp(params[layout.log_tau_temporal_idx]);
    int T_len = layout.temporal_end - layout.temporal_start;
    TemporalView tview;
    tview.read(params, data, layout);
    const double* phi_temporal = tview.phi;
    double rho_ar1 = read_temporal_rho_ar1(params, data, layout);

    // Bounds check for phi
    if (phi_local < data.multiscale_gp_data.range_local_lower ||
        phi_local > data.multiscale_gp_data.range_local_upper ||
        phi_regional < data.multiscale_gp_data.range_regional_lower ||
        phi_regional > data.multiscale_gp_data.range_regional_upper) {
        return;
    }

    // The field, in whichever coordinate it is sampled in
    MultiscaleGPView msgp;
    msgp.read(params, data, layout);
    const double* w_local = msgp.local();
    const double* w_regional = msgp.regional();
    std::vector<double> dL_dw_ms(N_gp, 0.0);

    // =========================================================================
    // Prior gradients
    // =========================================================================
    beta_gradient_prior(data, layout, beta_num, beta_denom, grad.data());
    re_gradient_prior(data, layout, re, grad.data(), sigma_re);
    phi_gradient_prior(data, layout, phi_num, phi_denom, grad.data());

    // PC priors on MSGP variances
    grad[layout.log_sigma2_gp_local_idx] = gp_pc_prior_grad_log_sigma2(
        sigma2_local, data.ms_sigma2_local_prior_U, data.ms_sigma2_local_prior_alpha);
    grad[layout.log_sigma2_gp_regional_idx] = gp_pc_prior_grad_log_sigma2(
        sigma2_regional, data.ms_sigma2_regional_prior_U, data.ms_sigma2_regional_prior_alpha);
    grad[layout.log_phi_gp_local_idx] = 1.0;
    grad[layout.log_phi_gp_regional_idx] = 1.0;

    // Temporal prior
    tau_temporal_prior_grad(data, layout, tau_temporal, grad.data());
    temporal_rho_prior_grad(data, layout, rho_ar1, grad.data());

    // =========================================================================
    // Likelihood loop (vectorized eta + per-obs scatter)
    // =========================================================================
    const int N = data.N;
    const bool is_binomial = (data.model_type == ModelType::BINOMIAL ||
                              data.model_type == ModelType::BETA_BINOMIAL);
    std::vector<double> grad_temporal_lik(T_len, 0.0);
    std::vector<int> obs_t_idx(N, -1);

    using RowMajorMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using VectorXd = Eigen::VectorXd;
    vec_grad_ws().init(N);
    Eigen::Map<const RowMajorMatrix> X_num(data.X_num_flat.data(), N, data.p_num);
    Eigen::Map<const VectorXd> b_num(beta_num, data.p_num);
    Eigen::Map<VectorXd> eta_n(vec_grad_ws().eta_num.data(), N);
    eta_n.noalias() = X_num * b_num;
    if (!is_binomial) {
        Eigen::Map<const RowMajorMatrix> X_denom(data.X_denom_flat.data(), N, data.p_denom);
        Eigen::Map<const VectorXd> b_denom(beta_denom, data.p_denom);
        Eigen::Map<VectorXd> eta_d(vec_grad_ws().eta_denom.data(), N);
        eta_d.noalias() = X_denom * b_denom;
    } else {
        std::memset(vec_grad_ws().eta_denom.data(), 0, N * sizeof(double));
    }

    for (int i = 0; i < N; i++) {
        if (layout.has_re && data.re_group[i] > 0) {
            double re_eff = re_value_for_eta(re, data.re_group[i] - 1, sigma_re, data.re_parameterization);
            vec_grad_ws().eta_num[i] += re_eff;
            if (!is_binomial) vec_grad_ws().eta_denom[i] += re_eff;
        }
        int loc_i = data.multiscale_gp_data.obs_to_loc[i];
        double ms_spatial = w_local[loc_i] + w_regional[loc_i];
        vec_grad_ws().eta_num[i] += ms_spatial;
        if (!is_binomial && data.multiscale_gp_data.shared) vec_grad_ws().eta_denom[i] += ms_spatial;

        if (!data.temporal_time_idx.empty() && i < (int)data.temporal_time_idx.size() && data.temporal_time_idx[i] > 0) {
            int t = data.temporal_time_idx[i] - 1;
            int g = data.temporal_group_idx[i] - 1;
            int t_idx = g * data.n_times + t;
            if (t_idx >= 0 && t_idx < T_len) {
                obs_t_idx[i] = t_idx;
                vec_grad_ws().eta_num[i] += phi_temporal[t_idx];
                if (!is_binomial && data.temporal_shared) vec_grad_ws().eta_denom[i] += phi_temporal[t_idx];
            }
        }
    }

    {
        double grad_phi_num_lik = 0.0, grad_phi_denom_lik = 0.0;
        vectorized::dispatch_residuals_and_beta_grads(
            data, layout,
            vec_grad_ws().eta_num.data(), vec_grad_ws().eta_denom.data(),
            vec_grad_ws().resid_num.data(), vec_grad_ws().resid_denom.data(),
            grad.data(), grad_phi_num_lik, grad_phi_denom_lik,
            obs_log_lik, fuse_lp, phi_num, phi_denom, vec_grad_ws());
    }

    for (int i = 0; i < N; i++) {
        double dLL_num = vec_grad_ws().resid_num[i];
        double dLL_denom = vec_grad_ws().resid_denom[i];
        if (layout.has_re && data.re_group[i] > 0) {
            grad[layout.re_start + data.re_group[i] - 1] += dLL_num + dLL_denom;
        }
        int loc_i = data.multiscale_gp_data.obs_to_loc[i];
        double dLL_dspatial = data.multiscale_gp_data.shared ? (dLL_num + dLL_denom) : dLL_num;
        dL_dw_ms[loc_i] += dLL_dspatial;
        if (obs_t_idx[i] >= 0) grad_temporal_lik[obs_t_idx[i]] += data.temporal_shared ? (dLL_num + dLL_denom) : dLL_num;
    }

    // The field's prior and the likelihood scatter, onto the sampled coordinate
    msgp.accumulate(dL_dw_ms.data(), data, layout, grad.data());

    // Temporal GMRF gradients
    temporal_gmrf_prior_grad(data, layout, tau_temporal, rho_ar1,
                             tview, T_len, grad_temporal_lik.data(), grad.data());

    // Non-centered RE chain rule transformation
    re_gradient_nc_transform(data, layout, params.data(), grad.data(), sigma_re);

    if (fuse_lp) *log_post_out = compute_log_post(params, data, layout, /*skip_obs_loop=*/true) + obs_log_lik;
}

// =====================================================================
// SVC gradient (hand-coded, ~3x faster than autodiff)
// Uses analytical gradients from svc_nngp_gradients for NNGP prior
// =====================================================================

void compute_gradient_svc_handcoded(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    std::vector<double>& grad,
    double* log_post_out = nullptr
) {
    // Fused log-posterior: accumulate obs log-lik during gradient loop,
    // then add prior/structural terms via skip_obs_loop=true (avoids 2nd O(N) pass)
    const bool fuse_lp = (log_post_out != nullptr) && !layout.has_zi;
    if (log_post_out && layout.has_zi) *log_post_out = compute_log_post(params, data, layout);
    double obs_log_lik = 0.0;
    int n_params = params.size();
    grad.assign(n_params, 0.0);

    // Extract common parameters
    auto cp = extract_common_params(params, layout);
    const double* beta_num = cp.beta_num;
    const double* beta_denom = cp.beta_denom;
    double sigma_re = cp.sigma_re;
    const double* re = cp.re;
    double phi_num = cp.phi_num;
    double phi_denom = cp.phi_denom;

    // SVC parameters (per-thread scratch: data.svc_data is shared by every
    // chain thread, so its own buffers cannot be written from here)
    int n_svc = data.svc_data.n_svc;
    int N_obs = data.svc_data.n_obs;

    RATIOD_TLS_WORKSPACE(ratiod_svc::SVCGradWorkspace, svc_scratch);
    svc_scratch.resize(n_svc, N_obs);

    double* svc_sigma2 = svc_scratch.sigma2.data();
    double* svc_phi = svc_scratch.phi.data();
    for (int j = 0; j < n_svc; j++) {
        svc_sigma2[j] = std::exp(params[layout.log_sigma2_svc_start + j]);
        svc_phi[j] = std::exp(params[layout.log_phi_svc_start + j]);

        // Outside the uniform prior's support the density is -Inf, so the
        // gradient is zero -- but the fused log posterior has to be written
        // too, or the caller reads whatever was in it and cannot tell the
        // proposal left the support.
        if (svc_phi[j] < data.svc_phi_prior_lower || svc_phi[j] > data.svc_phi_prior_upper) {
            if (log_post_out) *log_post_out = compute_log_post(params, data, layout);
            return;
        }
    }

    // The SVC field at the sampled coordinate, into per-thread scratch
    double* svc_w_flat = svc_scratch.w_flat.data();
    const double* svc_block = &params[layout.svc_w_start];
    ratiod_svc::svc_field<ratiod_svc::SVC_GRAD_SLOT>(svc_block, svc_sigma2, svc_phi, data.svc_data,
                          data.svc_gp_view, data.svc_noncentered, svc_w_flat);

    // =========================================================================
    // Prior gradients
    // =========================================================================

    beta_gradient_prior(data, layout, beta_num, beta_denom, grad.data());
    re_gradient_prior(data, layout, re, grad.data(), sigma_re);
    phi_gradient_prior(data, layout, phi_num, phi_denom, grad.data());

    // SVC hyperparameter priors
    for (int j = 0; j < n_svc; j++) {
        // Half-Cauchy on sigma, d/d(log_sigma2) = -ratio^2/(1+ratio^2) + 1
        double sigma = std::sqrt(svc_sigma2[j]);
        double scale = data.svc_sigma2_prior_scale;
        double ratio = sigma / scale;
        double ratio_sq = ratio * ratio;
        grad[layout.log_sigma2_svc_start + j] = -ratio_sq / (1.0 + ratio_sq) + 1.0;

        // Uniform prior on phi - just Jacobian
        grad[layout.log_phi_svc_start + j] = 1.0;
    }

    // =========================================================================
    // NNGP field prior (centred: on w; non-centred: N(0, I) on z, seeded by
    // svc_field_accumulate below)
    // =========================================================================
    if (!data.svc_noncentered) {
        ratiod_svc::svc_nngp_prior_grads(
            svc_w_flat, svc_sigma2, svc_phi, data.svc_data,
            layout.svc_w_start, layout.log_sigma2_svc_start, layout.log_phi_svc_start,
            svc_scratch, grad.data());
    }

    // The field prior's gradient above is placed on the uncentred w and is
    // NOT projected. The likelihood's is: eta reads the CENTRED field, so its
    // gradient is accumulated apart in svc_scratch.lik_grad and projected onto
    // the sum-to-zero subspace by svc_center_terms below.
    double* svc_lik_grad = svc_scratch.lik_grad.data();
    std::fill(svc_scratch.lik_grad.begin(), svc_scratch.lik_grad.end(), 0.0);

    // The field the likelihood reads is w minus its per-term mean; the field
    // the NNGP prior above reads is w itself. Holding the means rather than a
    // centred copy lets one buffer serve both.
    double* svc_term_mean = svc_scratch.term_mean.data();
    ratiod_svc::svc_term_means(svc_w_flat, data.svc_data, svc_term_mean);

    // =========================================================================
    // Data likelihood loop (vectorized eta + per-obs scatter)
    // =========================================================================
    const int N = data.N;
    const bool is_binomial = (data.model_type == ModelType::BINOMIAL ||
                              data.model_type == ModelType::BETA_BINOMIAL);

    using RowMajorMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using VectorXd = Eigen::VectorXd;
    vec_grad_ws().init(N);
    Eigen::Map<const RowMajorMatrix> X_num(data.X_num_flat.data(), N, data.p_num);
    Eigen::Map<const VectorXd> b_num(beta_num, data.p_num);
    Eigen::Map<VectorXd> eta_n(vec_grad_ws().eta_num.data(), N);
    eta_n.noalias() = X_num * b_num;
    if (!is_binomial) {
        Eigen::Map<const RowMajorMatrix> X_denom(data.X_denom_flat.data(), N, data.p_denom);
        Eigen::Map<const VectorXd> b_denom(beta_denom, data.p_denom);
        Eigen::Map<VectorXd> eta_d(vec_grad_ws().eta_denom.data(), N);
        eta_d.noalias() = X_denom * b_denom;
    } else {
        std::memset(vec_grad_ws().eta_denom.data(), 0, N * sizeof(double));
    }

    for (int i = 0; i < N; i++) {
        if (layout.has_re && data.re_group[i] > 0) {
            double re_eff = re_value_for_eta(re, data.re_group[i] - 1, sigma_re, data.re_parameterization);
            vec_grad_ws().eta_num[i] += re_eff;
            if (!is_binomial) vec_grad_ws().eta_denom[i] += re_eff;
        }
        double svc_effect = 0.0;
        for (int j = 0; j < n_svc; j++) {
            svc_effect += data.svc_data.X_svc[i * n_svc + j] *
                          (svc_w_flat[j * N_obs + i] - svc_term_mean[j]);
        }
        vec_grad_ws().eta_num[i] += svc_effect;
        if (!is_binomial && data.svc_data.shared) vec_grad_ws().eta_denom[i] += svc_effect;
    }

    {
        double grad_phi_num_lik = 0.0, grad_phi_denom_lik = 0.0;
        vectorized::dispatch_residuals_and_beta_grads(
            data, layout,
            vec_grad_ws().eta_num.data(), vec_grad_ws().eta_denom.data(),
            vec_grad_ws().resid_num.data(), vec_grad_ws().resid_denom.data(),
            grad.data(), grad_phi_num_lik, grad_phi_denom_lik,
            obs_log_lik, fuse_lp, phi_num, phi_denom, vec_grad_ws());
    }

    for (int i = 0; i < N; i++) {
        double dLL_num = vec_grad_ws().resid_num[i];
        double dLL_denom = vec_grad_ws().resid_denom[i];
        if (layout.has_re && data.re_group[i] > 0) {
            grad[layout.re_start + data.re_group[i] - 1] += dLL_num + dLL_denom;
        }
        double dLL_dsvc = data.svc_data.shared ? (dLL_num + dLL_denom) : dLL_num;
        for (int j = 0; j < n_svc; j++) {
            svc_lik_grad[j * N_obs + i] += dLL_dsvc * data.svc_data.X_svc[i * n_svc + j];
        }
    }

    ratiod_svc::svc_center_terms(svc_lik_grad, data.svc_data);
    ratiod_svc::svc_field_accumulate<ratiod_svc::SVC_GRAD_SLOT>(
        svc_block, svc_sigma2, svc_phi, data.svc_data, data.svc_gp_view,
        data.svc_noncentered, svc_lik_grad, layout.svc_w_start,
        layout.log_sigma2_svc_start, layout.log_phi_svc_start, grad.data());

    // Non-centered RE chain rule transformation
    re_gradient_nc_transform(data, layout, params.data(), grad.data(), sigma_re);

    if (fuse_lp) *log_post_out = compute_log_post(params, data, layout, /*skip_obs_loop=*/true) + obs_log_lik;
}

// =====================================================================
// HSGP-SVC gradient (hand-coded)
// Uses HSGP basis function approximation for spatially-varying coefficients.
// Each SVC term k has its own GP with sigma2_k, lengthscale_k, beta_k[m^2].
// All terms share the same basis matrix Phi (same coordinates/eigenvalues).
// =====================================================================

void compute_gradient_svc_hsgp_handcoded(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    std::vector<double>& grad,
    double* log_post_out = nullptr
) {
    // Thread-local HSGP workspace (one per SVC term, reused)
    RATIOD_TLS_WORKSPACE(ratiod_hsgp::HSGPWorkspace, hsgp_ws);
    hsgp_ws.init(data.N, data.svc_hsgp_data.m_total);

    const bool fuse_lp = (log_post_out != nullptr) && !layout.has_zi;
    if (log_post_out && layout.has_zi) *log_post_out = compute_log_post(params, data, layout);
    double obs_log_lik = 0.0;
    int n_params = params.size();
    grad.assign(n_params, 0.0);

    auto cp = extract_common_params(params, layout);
    const double* beta_num = cp.beta_num;
    const double* beta_denom = cp.beta_denom;
    double sigma_re = cp.sigma_re;
    const double* re = cp.re;
    double phi_num = cp.phi_num;
    double phi_denom = cp.phi_denom;

    const int N = data.N;
    const int n_svc = data.svc_data.n_svc;
    const int m_total = data.svc_hsgp_data.m_total;
    const bool is_binomial = (data.model_type == ModelType::BINOMIAL ||
                              data.model_type == ModelType::BETA_BINOMIAL);

    // Per-term HSGP parameters: sigma2_k, lengthscale_k, beta_k[m_total]
    // Stored in params as: [log_sigma2_svc[0..n_svc], log_phi_svc[0..n_svc], beta[0..n_svc*m_total]]
    // (log_phi_svc holds log_lengthscale for HSGP-SVC)

    // Evaluate f_k(s_i) for each SVC term and accumulate SVC contribution to eta
    // svc_f[j][i] = Phi * (sqrt(S_j) * beta_j)  -- spatial function for SVC term j at obs i
    // We compute this iteratively, storing each f_k in hsgp_ws.hsgp_f
    std::vector<double> svc_eta(N, 0.0);
    // Store per-term f values for gradient backprop (n_svc * N)
    std::vector<double> svc_f_all(n_svc * N);

    for (int j = 0; j < n_svc; j++) {
        double sigma2_j = std::exp(params[layout.log_sigma2_svc_start + j]);
        double lengthscale_j = std::exp(params[layout.log_phi_svc_start + j]);
        const double* beta_j = &params[layout.svc_w_start + j * m_total];

        // Evaluate f_j = Phi * (sqrt(S_j) ⊙ beta_j)
        ratiod_hsgp::hsgp_evaluate_ws(beta_j, sigma2_j, lengthscale_j,
                                       data.svc_hsgp_data, hsgp_ws);

        // Store f_j and accumulate into svc_eta
        std::memcpy(&svc_f_all[j * N], hsgp_ws.hsgp_f.data(), N * sizeof(double));
        for (int i = 0; i < N; i++) {
            svc_eta[i] += data.svc_data.X_svc[i * n_svc + j] * hsgp_ws.hsgp_f[i];
        }
    }

    // --- Prior gradients ---
    beta_gradient_prior(data, layout, beta_num, beta_denom, grad.data());
    re_gradient_prior(data, layout, re, grad.data(), sigma_re);
    phi_gradient_prior(data, layout, phi_num, phi_denom, grad.data());

    // Per-term HSGP hyperparameter priors
    for (int j = 0; j < n_svc; j++) {
        double sigma2_j = std::exp(params[layout.log_sigma2_svc_start + j]);
        double log_ls_j = params[layout.log_phi_svc_start + j];

        // PC prior on sigma: log_post += -rate*sigma + 0.5*log_sigma2
        // d/d(log_sigma2) = -0.5*rate*sigma + 0.5
        double sigma_j = std::sqrt(sigma2_j);
        double rate_sigma = 4.6;  // -log(0.01) / 1.0
        grad[layout.log_sigma2_svc_start + j] = -0.5 * rate_sigma * sigma_j + 0.5;

        // LogNormal(0,1) on lengthscale: d/d(log_ell) = -log_ell
        grad[layout.log_phi_svc_start + j] = -log_ls_j;

        // N(0,I) prior on beta: d/d(beta_j_k) = -beta_j_k
        for (int k = 0; k < m_total; k++) {
            grad[layout.svc_w_start + j * m_total + k] = -params[layout.svc_w_start + j * m_total + k];
        }
    }

    // --- Vectorized observation loop ---
    vec_grad_ws().init(N);
    using RowMajorMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using VectorXd = Eigen::VectorXd;

    Eigen::Map<const RowMajorMatrix> X_num(data.X_num_flat.data(), N, data.p_num);
    Eigen::Map<const VectorXd> b_num(beta_num, data.p_num);
    Eigen::Map<VectorXd> eta_n(vec_grad_ws().eta_num.data(), N);
    eta_n.noalias() = X_num * b_num;

    if (!is_binomial) {
        Eigen::Map<const RowMajorMatrix> X_denom(data.X_denom_flat.data(), N, data.p_denom);
        Eigen::Map<const VectorXd> b_denom(beta_denom, data.p_denom);
        Eigen::Map<VectorXd> eta_d(vec_grad_ws().eta_denom.data(), N);
        eta_d.noalias() = X_denom * b_denom;
    } else {
        std::memset(vec_grad_ws().eta_denom.data(), 0, N * sizeof(double));
    }

    // Add RE
    if (layout.has_re) {
        for (int i = 0; i < N; i++) {
            if (data.re_group[i] > 0) {
                double re_eff = re_value_for_eta(re, data.re_group[i] - 1, sigma_re, data.re_parameterization);
                vec_grad_ws().eta_num[i] += re_eff;
                if (!is_binomial) vec_grad_ws().eta_denom[i] += re_eff;
            }
        }
    }

    // Add SVC effect
    for (int i = 0; i < N; i++) {
        vec_grad_ws().eta_num[i] += svc_eta[i];
        if (!is_binomial && data.svc_data.shared) vec_grad_ws().eta_denom[i] += svc_eta[i];
    }

    // Compute residuals and beta grads
    {
        double grad_phi_num_lik = 0.0, grad_phi_denom_lik = 0.0;
        vectorized::dispatch_residuals_and_beta_grads(
            data, layout,
            vec_grad_ws().eta_num.data(), vec_grad_ws().eta_denom.data(),
            vec_grad_ws().resid_num.data(), vec_grad_ws().resid_denom.data(),
            grad.data(), grad_phi_num_lik, grad_phi_denom_lik,
            obs_log_lik, fuse_lp, phi_num, phi_denom, vec_grad_ws());
    }

    // RE gradients
    if (layout.has_re) {
        for (int i = 0; i < N; i++) {
            if (data.re_group[i] > 0) {
                grad[layout.re_start + data.re_group[i] - 1] += vec_grad_ws().resid_num[i] + vec_grad_ws().resid_denom[i];
            }
        }
    }

    // --- HSGP gradient backprop per SVC term ---
    // grad_f_k[i] = dLL/d(svc_eta_i) * X_svc[i,k]
    // where dLL/d(svc_eta_i) = resid_num[i] + (shared ? resid_denom[i] : 0)
    for (int j = 0; j < n_svc; j++) {
        double sigma2_j = std::exp(params[layout.log_sigma2_svc_start + j]);
        double lengthscale_j = std::exp(params[layout.log_phi_svc_start + j]);
        const double* beta_j = &params[layout.svc_w_start + j * m_total];

        // Build grad_f for this SVC term
        double* grad_f_ptr = hsgp_ws.grad_f.data();
        for (int i = 0; i < N; i++) {
            double dLL = data.svc_data.shared
                ? (vec_grad_ws().resid_num[i] + vec_grad_ws().resid_denom[i])
                : vec_grad_ws().resid_num[i];
            grad_f_ptr[i] = dLL * data.svc_data.X_svc[i * n_svc + j];
        }

        // Re-evaluate sqrt_S for this term (needed by gradient computation)
        // hsgp_evaluate_ws caches sqrt_S in workspace
        ratiod_hsgp::hsgp_evaluate_ws(beta_j, sigma2_j, lengthscale_j,
                                       data.svc_hsgp_data, hsgp_ws);

        // Compute HSGP gradients for beta_j, sigma2_j, lengthscale_j
        double grad_log_sigma2_j, grad_log_lengthscale_j;
        ratiod_hsgp::hsgp_compute_gradients_ws(beta_j, sigma2_j, lengthscale_j,
                                                data.svc_hsgp_data, hsgp_ws,
                                                grad_log_sigma2_j, grad_log_lengthscale_j);

        // Accumulate into gradient vector
        for (int k = 0; k < m_total; k++) {
            grad[layout.svc_w_start + j * m_total + k] += hsgp_ws.grad_beta_out[k];
        }
        grad[layout.log_sigma2_svc_start + j] += grad_log_sigma2_j;
        grad[layout.log_phi_svc_start + j] += grad_log_lengthscale_j;
    }

    re_gradient_nc_transform(data, layout, params.data(), grad.data(), sigma_re);

    if (fuse_lp) *log_post_out = compute_log_post(params, data, layout, /*skip_obs_loop=*/true) + obs_log_lik;
}

// =====================================================================
// TVC gradient (hand-coded, ~3x faster than autodiff)
// Uses analytical gradients from hmc_tvc_grad.h for RW1/RW2/AR1 priors
// =====================================================================

void compute_gradient_tvc_handcoded(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    std::vector<double>& grad,
    double* log_post_out = nullptr
) {
    // Fused log-posterior: accumulate obs log-lik during gradient loop,
    // then add prior/structural terms via skip_obs_loop=true (avoids 2nd O(N) pass)
    const bool fuse_lp = (log_post_out != nullptr) && !layout.has_zi;
    if (log_post_out && layout.has_zi) *log_post_out = compute_log_post(params, data, layout);
    double obs_log_lik = 0.0;
    int n_params = params.size();
    grad.assign(n_params, 0.0);

    // Extract common parameters
    auto cp = extract_common_params(params, layout);
    const double* beta_num = cp.beta_num;
    const double* beta_denom = cp.beta_denom;
    double sigma_re = cp.sigma_re;
    const double* re = cp.re;
    double phi_num = cp.phi_num;
    double phi_denom = cp.phi_denom;

    // TVC parameters (per-thread scratch: data.tvc_data is shared by every
    // chain thread, so its own buffers cannot be written from here)
    int n_tvc = data.tvc_data.n_tvc;
    int n_times = data.tvc_data.n_times;
    int n_groups = data.tvc_data.n_groups;
    int n_w = n_groups * n_tvc * n_times;

    RATIOD_TLS_WORKSPACE(ratiod_tvc::TVCGradWorkspace, tvc_scratch);
    tvc_scratch.resize(n_tvc, n_times, n_groups, data.N);

    double* tvc_tau = tvc_scratch.tau.data();
    double* tvc_rho = tvc_scratch.rho.data();
    for (int j = 0; j < n_tvc; j++) {
        tvc_tau[j] = std::exp(params[layout.log_tau_tvc_start + j]);
        if (data.tvc_data.structure == ratiod_temporal::TemporalType::AR1) {
            double logit_rho = params[layout.logit_rho_tvc_start + j];
            double u = 1.0 / (1.0 + std::exp(-logit_rho));
            tvc_rho[j] = 2.0 * u - 1.0;
        } else {
            tvc_rho[j] = 0.0;
        }
    }

    // Extract TVC w values into per-thread scratch
    double* tvc_w_flat = tvc_scratch.w_flat.data();
    for (int k = 0; k < n_w; k++) {
        tvc_w_flat[k] = params[layout.tvc_w_start + k];
    }

    // =========================================================================
    // Prior gradients
    // =========================================================================

    beta_gradient_prior(data, layout, beta_num, beta_denom, grad.data());
    re_gradient_prior(data, layout, re, grad.data(), sigma_re);
    phi_gradient_prior(data, layout, phi_num, phi_denom, grad.data());

    // TVC hyperparameter priors: Gamma on tau at the configured hyperparameters.
    for (int j = 0; j < n_tvc; j++) {
        grad[layout.log_tau_tvc_start + j] = ratiod_tvc::log_prior_tau_gamma_grad(
            tvc_tau[j], data.tvc_tau_shape, data.tvc_tau_rate);
    }

    // AR1: the rho prior in the coordinate the sampler moves
    if (data.tvc_data.structure == ratiod_temporal::TemporalType::AR1) {
        for (int j = 0; j < n_tvc; j++) {
            double u = (tvc_rho[j] + 1.0) / 2.0;  // u in (0,1)
            grad[layout.logit_rho_tvc_start + j] = ratiod_ar1::log_prior_logit_rho_grad(
                u, ratiod_ar1::RHO_PRIOR_A, ratiod_ar1::RHO_PRIOR_B);
        }
    }

    // =========================================================================
    // Compute TVC prior gradients using zero-allocation workspace version
    // =========================================================================
    ratiod_tvc::TVCGradientWS tvc_ws;
    tvc_ws.grad_w = tvc_scratch.grad_w.data();
    tvc_ws.grad_log_tau = tvc_scratch.grad_log_tau.data();
    tvc_ws.grad_logit_rho = tvc_scratch.grad_logit_rho.data();
    tvc_ws.grad_w_jg = tvc_scratch.grad_w_jg.data();
    tvc_ws.d_buf = tvc_scratch.d_buf.data();
    tvc_ws.n_w = n_w;
    tvc_ws.n_tvc = n_tvc;
    ratiod_tvc::tvc_prior_gradients_ws(tvc_w_flat, data.tvc_data, tvc_tau, tvc_rho, tvc_ws);

    // Add TVC prior gradient contributions
    for (int k = 0; k < n_w; k++) {
        grad[layout.tvc_w_start + k] += tvc_ws.grad_w[k];
    }
    for (int j = 0; j < n_tvc; j++) {
        grad[layout.log_tau_tvc_start + j] += tvc_ws.grad_log_tau[j];
    }
    if (data.tvc_data.structure == ratiod_temporal::TemporalType::AR1) {
        for (int j = 0; j < n_tvc; j++) {
            grad[layout.logit_rho_tvc_start + j] += tvc_ws.grad_logit_rho[j];
        }
    }

    // =========================================================================
    // Precompute TVC contribution to linear predictor (pre-allocated buffer)
    // =========================================================================
    double* tvc_eta = tvc_scratch.eta.data();
    std::fill(tvc_eta, tvc_eta + data.N, 0.0);
    for (int i = 0; i < data.N; i++) {
        int t = data.tvc_data.time_index[i] - 1;  // 0-based
        int g = data.tvc_data.group_index[i] - 1;  // 0-based

        for (int j = 0; j < n_tvc; j++) {
            double x_ij = data.tvc_data.X_tvc[i * n_tvc + j];
            double w_jgt = tvc_w_flat[(g * n_tvc + j) * n_times + t];
            tvc_eta[i] += x_ij * w_jgt;
        }
    }

    // =========================================================================
    // Data likelihood loop (vectorized eta + per-obs scatter)
    // =========================================================================
    const int N = data.N;
    const bool is_binomial = (data.model_type == ModelType::BINOMIAL ||
                              data.model_type == ModelType::BETA_BINOMIAL);

    // --- Pass 1: Vectorized eta computation (Eigen matvec) ---
    using RowMajorMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using VectorXd = Eigen::VectorXd;

    vec_grad_ws().init(N);
    Eigen::Map<const RowMajorMatrix> X_num(data.X_num_flat.data(), N, data.p_num);
    Eigen::Map<const VectorXd> b_num(beta_num, data.p_num);
    Eigen::Map<VectorXd> eta_n(vec_grad_ws().eta_num.data(), N);
    eta_n.noalias() = X_num * b_num;

    if (!is_binomial) {
        Eigen::Map<const RowMajorMatrix> X_denom(data.X_denom_flat.data(), N, data.p_denom);
        Eigen::Map<const VectorXd> b_denom(beta_denom, data.p_denom);
        Eigen::Map<VectorXd> eta_d(vec_grad_ws().eta_denom.data(), N);
        eta_d.noalias() = X_denom * b_denom;
    } else {
        std::memset(vec_grad_ws().eta_denom.data(), 0, N * sizeof(double));
    }

    // Add RE + TVC effects per-obs
    for (int i = 0; i < N; i++) {
        if (layout.has_re && data.re_group[i] > 0) {
            int g = data.re_group[i] - 1;
            double re_eff = re_value_for_eta(re, g, sigma_re, data.re_parameterization);
            vec_grad_ws().eta_num[i] += re_eff;
            if (!is_binomial) vec_grad_ws().eta_denom[i] += re_eff;
        }
        double tvc_effect = tvc_eta[i];
        vec_grad_ws().eta_num[i] += tvc_effect;
        if (!is_binomial && data.tvc_data.shared) vec_grad_ws().eta_denom[i] += tvc_effect;
    }

    // --- Pass 2+3: Vectorized residuals + beta grads (template-dispatched) ---
    {
        double grad_phi_num_lik = 0.0, grad_phi_denom_lik = 0.0;
        vectorized::dispatch_residuals_and_beta_grads(
            data, layout,
            vec_grad_ws().eta_num.data(), vec_grad_ws().eta_denom.data(),
            vec_grad_ws().resid_num.data(), vec_grad_ws().resid_denom.data(),
            grad.data(), grad_phi_num_lik, grad_phi_denom_lik,
            obs_log_lik, fuse_lp, phi_num, phi_denom, vec_grad_ws());
    }

    // Scatter residuals to RE and TVC gradients
    for (int i = 0; i < N; i++) {
        if (layout.has_re && data.re_group[i] > 0) {
            grad[layout.re_start + data.re_group[i] - 1] += vec_grad_ws().resid_num[i] + vec_grad_ws().resid_denom[i];
        }
        double dLL_dtvc = data.tvc_data.shared
            ? (vec_grad_ws().resid_num[i] + vec_grad_ws().resid_denom[i])
            : vec_grad_ws().resid_num[i];
        int t = data.tvc_data.time_index[i] - 1;
        int g = data.tvc_data.group_index[i] - 1;
        for (int j = 0; j < n_tvc; j++) {
            double x_ij = data.tvc_data.X_tvc[i * n_tvc + j];
            int w_idx = (g * n_tvc + j) * n_times + t;
            grad[layout.tvc_w_start + w_idx] += dLL_dtvc * x_ij;
        }
    }

    re_gradient_nc_transform(data, layout, params.data(), grad.data(), sigma_re);

    if (fuse_lp) *log_post_out = compute_log_post(params, data, layout, /*skip_obs_loop=*/true) + obs_log_lik;
}

// =====================================================================
// Latent factor gradient (hand-coded, O(N*K))
// Uses analytical gradients for latent factor models
// =====================================================================

void compute_gradient_latent_handcoded(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    std::vector<double>& grad,
    double* log_post_out = nullptr
) {
    // Fused log-posterior: accumulate obs log-lik during gradient loop,
    // then add prior/structural terms via skip_obs_loop=true (avoids 2nd O(N) pass)
    const bool fuse_lp = (log_post_out != nullptr) && !layout.has_zi;
    if (log_post_out && layout.has_zi) *log_post_out = compute_log_post(params, data, layout);
    double obs_log_lik = 0.0;
    int n_params = params.size();
    grad.assign(n_params, 0.0);

    // Extract common parameters
    auto cp = extract_common_params(params, layout);
    const double* beta_num = cp.beta_num;
    const double* beta_denom = cp.beta_denom;
    double sigma_re = cp.sigma_re;
    const double* re = cp.re;
    double phi_num = cp.phi_num;
    double phi_denom = cp.phi_denom;

    // Latent factor parameters
    int K = data.latent_n_factors;
    int N = data.N;

    // Extract log_sigma for latent factors
    std::vector<double> log_sigma_latent(K);
    std::vector<double> sigma_latent(K);
    for (int k = 0; k < K; k++) {
        log_sigma_latent[k] = params[layout.log_sigma_latent_start + k];
        sigma_latent[k] = std::exp(log_sigma_latent[k]);
    }

    // Extract factors (unconstrained)
    int n_factor_params = N * K;
    std::vector<double> factors_raw(n_factor_params);
    for (int j = 0; j < n_factor_params; j++) {
        factors_raw[j] = params[layout.latent_factor_start + j];
    }

    // Apply constraint to get constrained factors
    std::vector<double> factors_constrained = factors_raw;
    if (data.latent_constraint == 0) {  // SUM_TO_ZERO
        for (int k = 0; k < K; k++) {
            double sum = 0.0;
            for (int i = 0; i < N; i++) {
                sum += factors_constrained[i * K + k];
            }
            double mean = sum / N;
            for (int i = 0; i < N; i++) {
                factors_constrained[i * K + k] -= mean;
            }
        }
    } else {  // FIRST_ZERO
        for (int k = 0; k < K; k++) {
            double first_val = factors_constrained[k];  // factors[0, k]
            for (int i = 0; i < N; i++) {
                factors_constrained[i * K + k] -= first_val;
            }
        }
    }

    // Precompute latent contribution to eta
    std::vector<double> latent_eta(N, 0.0);
    for (int i = 0; i < N; i++) {
        for (int k = 0; k < K; k++) {
            latent_eta[i] += factors_constrained[i * K + k] * sigma_latent[k];
        }
    }

    // =========================================================================
    // Prior gradients
    // =========================================================================

    beta_gradient_prior(data, layout, beta_num, beta_denom, grad.data());
    re_gradient_prior(data, layout, re, grad.data(), sigma_re);
    phi_gradient_prior(data, layout, phi_num, phi_denom, grad.data());

    // Latent sigma prior: Exponential(rate) on sigma, with Jacobian for log transform
    // log p(log_sigma) = log(rate) + log_sigma - rate * sigma
    // d/d(log_sigma) = 1 - rate * sigma
    double latent_rate = data.latent_sigma_prior_rate;
    for (int k = 0; k < K; k++) {
        grad[layout.log_sigma_latent_start + k] = 1.0 - latent_rate * sigma_latent[k];
    }

    // Latent factor prior: N(0, 1) on constrained factors
    // The autodiff applies the prior to constrained factors, then chain-rules to raw factors.
    // Direct computation: d(-0.5*f_constrained^2)/d(f_raw) = -f_constrained * d(f_constrained)/d(f_raw)
    // For SUM_TO_ZERO: d(fc[i])/d(fr[j]) = delta_ij - 1/N, so gradient = -fc[i] + mean(fc) = -fc[i]
    // For FIRST_ZERO: d(fc[i])/d(fr[j]) = delta_ij - delta_j0, similar result
    // In both cases, the prior gradient w.r.t. raw factors equals -constrained_factor
    // But we need to apply the chain rule properly below, so here we just store the
    // gradient w.r.t. constrained factors.

    // Note: The prior is conceptually on the constrained factors (which sum to zero),
    // and the gradient flows back through the constraint transformation.
    // We handle this by computing prior gradient on constrained factors,
    // then adding it to grad_factors_constrained, which gets chain-ruled below.

    // =========================================================================
    // Likelihood loop - compute dLL/deta and chain-rule to all parameters
    // =========================================================================

    // Gradients to accumulate for latent factors (on constrained factors first)
    std::vector<double> grad_factors_constrained(n_factor_params, 0.0);

    for (int i = 0; i < N; i++) {
        // Linear predictors
        double eta_num = 0.0, eta_denom = 0.0;
        for (int j = 0; j < data.p_num; j++) {
            eta_num += data.X_num_flat[i * data.p_num + j] * beta_num[j];
        }
        for (int j = 0; j < data.p_denom; j++) {
            eta_denom += data.X_denom_flat[i * data.p_denom + j] * beta_denom[j];
        }

        // Random effects (handles NC parameterization)
        if (layout.has_re && data.re_group[i] > 0) {
            int g = data.re_group[i] - 1;
            double re_eff = re_value_for_eta(re, g, sigma_re, data.re_parameterization);
            eta_num += re_eff;
            eta_denom += re_eff;
        }

        // Latent effect
        double latent_effect = latent_eta[i];
        if (data.latent_shared) {
            eta_num += latent_effect;
            eta_denom += latent_effect;
        } else {
            eta_num += latent_effect;
        }

        if (fuse_lp) obs_log_lik += compute_obs_ll(data, i, eta_num, eta_denom, phi_num, phi_denom);

        double dLL_deta_num = 0.0, dLL_deta_denom = 0.0;
        compute_obs_residuals(data, i, eta_num, eta_denom, phi_num, phi_denom, dLL_deta_num, dLL_deta_denom);

        // Total gradient through latent effect
        double dLL_dlatent = data.latent_shared ?
                             (dLL_deta_num + dLL_deta_denom) : dLL_deta_num;

        scatter_beta_gradients(data, layout, i, dLL_deta_num, dLL_deta_denom, grad.data());
        scatter_re_gradient(data, layout, i, dLL_deta_num, dLL_deta_denom, grad.data());

        // Gradients for latent factors (on constrained space)
        // eta_latent[i] = sum_k factor[i,k] * sigma[k]
        // d(LL)/d(factor[i,k]) = dLL_dlatent * sigma[k]
        // d(LL)/d(log_sigma[k]) += dLL_dlatent * factor[i,k] * sigma[k]
        for (int k = 0; k < K; k++) {
            grad_factors_constrained[i * K + k] = dLL_dlatent * sigma_latent[k];
            grad[layout.log_sigma_latent_start + k] += dLL_dlatent * factors_constrained[i * K + k] * sigma_latent[k];
        }

        accumulate_phi_likelihood_grad(data, layout, i, eta_num, eta_denom, phi_num, phi_denom, grad.data());
    }

    // =========================================================================
    // Add prior gradient to grad_factors_constrained
    // =========================================================================
    // Prior: N(0, 1) on constrained factors, log p = -0.5 * f_constrained^2
    // Gradient: d(-0.5 * f^2)/d(f) = -f
    // For FIRST_ZERO: constrained factor 0 is always 0, so skip it
    int prior_start = (data.latent_constraint == 0) ? 0 : 1;
    for (int k = 0; k < K; k++) {
        for (int i = prior_start; i < N; i++) {
            grad_factors_constrained[i * K + k] += -factors_constrained[i * K + k];
        }
    }

    // =========================================================================
    // Apply constraint chain-rule to get gradients on raw (unconstrained) factors
    // =========================================================================

    // For sum-to-zero: d(LL)/d(factor_raw[j,k]) = d(LL)/d(factor_constrained[j,k])
    //                                           - (1/N) * sum_i d(LL)/d(factor_constrained[i,k])
    // For first-zero: d(LL)/d(factor_raw[0,k]) = -sum_{i>0} d(LL)/d(factor_constrained[i,k])
    //                 d(LL)/d(factor_raw[j,k]) = d(LL)/d(factor_constrained[j,k]) for j > 0

    if (data.latent_constraint == 0) {  // SUM_TO_ZERO
        for (int k = 0; k < K; k++) {
            // Compute mean gradient for this factor
            double sum_grad = 0.0;
            for (int i = 0; i < N; i++) {
                sum_grad += grad_factors_constrained[i * K + k];
            }
            double mean_grad = sum_grad / N;

            // Adjust each gradient
            for (int i = 0; i < N; i++) {
                grad[layout.latent_factor_start + i * K + k] +=
                    grad_factors_constrained[i * K + k] - mean_grad;
            }
        }
    } else {  // FIRST_ZERO
        for (int k = 0; k < K; k++) {
            // Gradient for factor_raw[0,k] = -sum of gradients for i > 0
            double sum_grad = 0.0;
            for (int i = 1; i < N; i++) {
                sum_grad += grad_factors_constrained[i * K + k];
                // Gradient for factor_raw[i,k] = gradient of constrained[i,k] for i > 0
                grad[layout.latent_factor_start + i * K + k] += grad_factors_constrained[i * K + k];
            }
            grad[layout.latent_factor_start + k] += -sum_grad;  // factor_raw[0,k]
        }
    }

    // Non-centered RE chain rule transformation
    re_gradient_nc_transform(data, layout, params.data(), grad.data(), sigma_re);

    if (fuse_lp) *log_post_out = compute_log_post(params, data, layout, /*skip_obs_loop=*/true) + obs_log_lik;
}

// =====================================================================
// GP gradient via autodiff (O(N*nn^3) - much faster than numerical O(N^2))
// Uses templated NNGP likelihood from hmc_gp_autodiff.h
// =====================================================================

// =====================================================================
// Common autodiff prior setup (beta, RE, phi priors)
// Shared by gp_autodiff, msgp_autodiff, gp_temporal_autodiff.
// Returns log_post, sigma_re, phi_num, phi_denom via output parameters.
// =====================================================================
struct AutodiffCommonResult {
    ratiod::ad::Var log_post;
    ratiod::ad::Var sigma_re;
    ratiod::ad::Var phi_num;
    ratiod::ad::Var phi_denom;
};

static inline AutodiffCommonResult add_common_priors_ad(
    ratiod::ad::Tape* tape,
    const std::vector<ratiod::ad::Var>& params_ad,
    const ModelData& data,
    const ParamLayout& layout
) {
    using namespace ratiod::ad;
    using namespace ratiod::math;

    Var log_post(tape, 0.0);

    // Fixed effects priors: N(0, sigma_beta^2)
    double tau_beta = 1.0 / (data.sigma_beta * data.sigma_beta);
    for (int j = 0; j < data.p_num; j++) {
        Var beta = params_ad[layout.beta_num_start + j];
        log_post = log_post - (0.5 * tau_beta) * beta * beta;
    }
    for (int j = 0; j < data.p_denom; j++) {
        Var beta = params_ad[layout.beta_denom_start + j];
        log_post = log_post - (0.5 * tau_beta) * beta * beta;
    }

    // Random effects priors (if present)
    Var sigma_re(tape, 1.0);
    if (layout.has_re && data.n_re_groups > 0) {
        Var log_sigma_re = params_ad[layout.log_sigma_re_idx];
        sigma_re = safe_exp(log_sigma_re);

        Var ratio = sigma_re / data.sigma_re_scale;
        log_post = log_post - safe_log(1.0 + ratio * ratio);
        log_post = log_post + log_sigma_re;  // Jacobian

        if (data.re_parameterization == 1) {
            for (int g = 0; g < data.n_re_groups; g++) {
                Var re_g = params_ad[layout.re_start + g];
                log_post = log_post - 0.5 * re_g * re_g;
            }
        } else {
            Var tau_re = 1.0 / (sigma_re * sigma_re + 1e-10);
            for (int g = 0; g < data.n_re_groups; g++) {
                Var re_g = params_ad[layout.re_start + g];
                log_post = log_post - 0.5 * tau_re * re_g * re_g;
                log_post = log_post + 0.5 * safe_log(tau_re);
            }
        }
    }

    // Overdispersion priors (Gamma)
    Var phi_num(tape, 1.0);
    Var phi_denom(tape, 1.0);
    if (layout.has_phi_num) {
        Var log_phi = params_ad[layout.log_phi_num_idx];
        phi_num = safe_exp(log_phi);
        log_post = log_post + (data.phi_prior_shape - 1.0) * log_phi
                            - data.phi_prior_rate * phi_num + log_phi;
    }
    if (layout.has_phi_denom) {
        Var log_phi = params_ad[layout.log_phi_denom_idx];
        phi_denom = safe_exp(log_phi);
        log_post = log_post + (data.phi_prior_shape - 1.0) * log_phi
                            - data.phi_prior_rate * phi_denom + log_phi;
    }

    return {log_post, sigma_re, phi_num, phi_denom};
}

// =====================================================================
// Multi-scale GP gradient (hand-coded, ~2-3x faster than autodiff)
// Uses analytical gradients for w and numerical for sigma2/phi
// =====================================================================

void compute_gradient_msgp_handcoded(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    std::vector<double>& grad,
    double* log_post_out = nullptr
) {
    // Fused log-posterior: accumulate obs log-lik during gradient loop,
    // then add prior/structural terms via skip_obs_loop=true (avoids 2nd O(N) pass)
    const bool fuse_lp = (log_post_out != nullptr) && !layout.has_zi;
    if (log_post_out && layout.has_zi) *log_post_out = compute_log_post(params, data, layout);
    double obs_log_lik = 0.0;
    int n_params = params.size();
    grad.assign(n_params, 0.0);

    // Extract common parameters
    auto cp = extract_common_params(params, layout);
    const double* beta_num = cp.beta_num;
    const double* beta_denom = cp.beta_denom;
    double sigma_re = cp.sigma_re;
    const double* re = cp.re;
    double phi_num = cp.phi_num;
    double phi_denom = cp.phi_denom;

    // Multi-scale GP parameters
    int N_gp = data.multiscale_gp_data.n_obs;

    double log_sigma2_local = params[layout.log_sigma2_gp_local_idx];
    double log_phi_local = params[layout.log_phi_gp_local_idx];
    double sigma2_local = std::exp(log_sigma2_local);
    double phi_local = std::exp(log_phi_local);

    double log_sigma2_regional = params[layout.log_sigma2_gp_regional_idx];
    double log_phi_regional = params[layout.log_phi_gp_regional_idx];
    double sigma2_regional = std::exp(log_sigma2_regional);
    double phi_regional = std::exp(log_phi_regional);

    // Bounds check for phi
    if (phi_local < data.multiscale_gp_data.range_local_lower ||
        phi_local > data.multiscale_gp_data.range_local_upper ||
        phi_regional < data.multiscale_gp_data.range_regional_lower ||
        phi_regional > data.multiscale_gp_data.range_regional_upper) {
        return; // Out of bounds - return zero gradient
    }

    // The field, in whichever coordinate it is sampled in
    MultiscaleGPView msgp;
    msgp.read(params, data, layout);
    const double* w_local = msgp.local();
    const double* w_regional = msgp.regional();
    std::vector<double> dL_dw_ms(N_gp, 0.0);

    // =========================================================================
    // Prior gradients
    // =========================================================================

    beta_gradient_prior(data, layout, beta_num, beta_denom, grad.data());
    re_gradient_prior(data, layout, re, grad.data(), sigma_re);
    phi_gradient_prior(data, layout, phi_num, phi_denom, grad.data());

    // PC priors on GP variances
    grad[layout.log_sigma2_gp_local_idx] = gp_pc_prior_grad_log_sigma2(
        sigma2_local, data.ms_sigma2_local_prior_U, data.ms_sigma2_local_prior_alpha);
    grad[layout.log_sigma2_gp_regional_idx] = gp_pc_prior_grad_log_sigma2(
        sigma2_regional, data.ms_sigma2_regional_prior_U, data.ms_sigma2_regional_prior_alpha);

    // Jacobians for log-transforms
    grad[layout.log_phi_gp_local_idx] = 1.0;    // Uniform prior, just Jacobian
    grad[layout.log_phi_gp_regional_idx] = 1.0;

    // =========================================================================
    // Data likelihood loop
    // =========================================================================
    for (int i = 0; i < data.N; i++) {
        // Linear predictors
        double eta_num = 0.0, eta_denom = 0.0;
        for (int j = 0; j < data.p_num; j++) {
            eta_num += data.X_num_flat[i * data.p_num + j] * beta_num[j];
        }
        for (int j = 0; j < data.p_denom; j++) {
            eta_denom += data.X_denom_flat[i * data.p_denom + j] * beta_denom[j];
        }

        // Random effects (handles NC parameterization)
        if (layout.has_re && data.re_group[i] > 0) {
            int g = data.re_group[i] - 1;
            double re_eff = re_value_for_eta(re, g, sigma_re, data.re_parameterization);
            eta_num += re_eff;
            eta_denom += re_eff;
        }

        // Multi-scale GP spatial effect (map observation to unique location)
        int loc_i = data.multiscale_gp_data.obs_to_loc[i];
        double ms_spatial = w_local[loc_i] + w_regional[loc_i];
        if (data.multiscale_gp_data.shared) {
            eta_num += ms_spatial;
            eta_denom += ms_spatial;
        } else {
            eta_num += ms_spatial;
        }

        if (fuse_lp) obs_log_lik += compute_obs_ll(data, i, eta_num, eta_denom, phi_num, phi_denom);

        double dLL_deta_num = 0.0, dLL_deta_denom = 0.0;
        compute_obs_residuals(data, i, eta_num, eta_denom, phi_num, phi_denom, dLL_deta_num, dLL_deta_denom);

        scatter_beta_gradients(data, layout, i, dLL_deta_num, dLL_deta_denom, grad.data());
        scatter_re_gradient(data, layout, i, dLL_deta_num, dLL_deta_denom, grad.data());

        // Gradients for GP spatial effects (from likelihood, mapped to unique location)
        double dLL_dspatial = data.multiscale_gp_data.shared ?
                              (dLL_deta_num + dLL_deta_denom) : dLL_deta_num;
        dL_dw_ms[loc_i] += dLL_dspatial;

        accumulate_phi_likelihood_grad(data, layout, i, eta_num, eta_denom, phi_num, phi_denom, grad.data());
    }

    // The field's prior and the likelihood scatter, onto the sampled coordinate
    msgp.accumulate(dL_dw_ms.data(), data, layout, grad.data());

    // Non-centered RE chain rule transformation
    re_gradient_nc_transform(data, layout, params.data(), grad.data(), sigma_re);

    if (fuse_lp) *log_post_out = compute_log_post(params, data, layout, /*skip_obs_loop=*/true) + obs_log_lik;
}

// =====================================================================
// HSGP gradient (O(N*M^2) - analytical, ~50x faster than numerical)
// =====================================================================

void compute_gradient_hsgp(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    std::vector<double>& grad,
    double* log_post_out = nullptr
) {
    // Thread-local workspace eliminates 9+ heap allocations per call
    RATIOD_TLS_WORKSPACE(ratiod_hsgp::HSGPWorkspace, hsgp_ws);
    hsgp_ws.init(data.N, data.hsgp_data.m_total);

    // Fused log-posterior: accumulate obs log-lik during gradient loop,
    // then add prior/structural terms via skip_obs_loop=true (avoids 2nd O(N) pass)
    const bool fuse_lp = (log_post_out != nullptr) && !layout.has_zi;
    if (log_post_out && layout.has_zi) *log_post_out = compute_log_post(params, data, layout);
    double obs_log_lik = 0.0;
  int n_params = params.size();
  grad.assign(n_params, 0.0);

  // Extract common parameters
  auto cp = extract_common_params(params, layout);
  const double* beta_num = cp.beta_num;
  const double* beta_denom = cp.beta_denom;
  double sigma_re = cp.sigma_re;
  const double* re = cp.re;
  double phi_num = cp.phi_num;
  double phi_denom = cp.phi_denom;

  // HSGP parameters
  double log_sigma2 = params[layout.log_sigma2_hsgp_idx];
  double log_lengthscale = params[layout.log_lengthscale_hsgp_idx];
  double sigma2_hsgp = std::exp(log_sigma2);
  double lengthscale_hsgp = std::exp(log_lengthscale);

  int m_total = data.hsgp_data.m_total;
  const double* hsgp_beta_ptr = &params[layout.hsgp_beta_start];

  // Evaluate HSGP spatial effect (uses workspace, zero allocation)
  ratiod_hsgp::hsgp_evaluate_ws(hsgp_beta_ptr, sigma2_hsgp, lengthscale_hsgp,
                                 data.hsgp_data, hsgp_ws);

  // Temporal parameters (for HSGP + temporal combinations)
  double tau_temporal = 0.0;
  int T_len = 0;
  const double* phi_temporal = nullptr;
  TemporalView tview;
  double rho_ar1 = 0.0;
  if (layout.has_temporal) {
    tau_temporal = std::exp(params[layout.log_tau_temporal_idx]);
    T_len = layout.temporal_end - layout.temporal_start;
    tview.read(params, data, layout);
    phi_temporal = tview.phi;
    rho_ar1 = read_temporal_rho_ar1(params, data, layout);
  }

  // Zero the grad_f buffer (workspace, no allocation)
  std::memset(hsgp_ws.grad_f.data(), 0, data.N * sizeof(double));
  std::vector<double> grad_temporal_lik(T_len, 0.0);

  // --- Prior gradients ---

  beta_gradient_prior(data, layout, beta_num, beta_denom, grad.data());
  re_gradient_prior(data, layout, re, grad.data(), sigma_re);
  phi_gradient_prior(data, layout, phi_num, phi_denom, grad.data());

  // HSGP prior gradients (will be added to by hsgp_compute_gradients_ws)
  double sigma = std::sqrt(sigma2_hsgp);
  double rate_sigma = 4.6;
  grad[layout.log_sigma2_hsgp_idx] = -0.5 * rate_sigma * sigma + 0.5 - 0.5;

  // LogNormal(0,1) on lengthscale
  grad[layout.log_lengthscale_hsgp_idx] = -log_lengthscale;

  // N(0, I) prior on beta: d/d(beta_j) = -beta_j
  for (int j = 0; j < m_total; j++) {
    grad[layout.hsgp_beta_start + j] = -hsgp_beta_ptr[j];
  }

  // Temporal prior on tau (Gamma) and rho (Beta)
  if (layout.has_temporal) {
    tau_temporal_prior_grad(data, layout, tau_temporal, grad.data());
    temporal_rho_prior_grad(data, layout, rho_ar1, grad.data());
  }

  // --- Vectorized observation loop (3-pass: Eigen matvec, scalar residuals, Eigen scatter) ---
  const double* hsgp_f = hsgp_ws.hsgp_f.data();
  double* grad_f_ptr = hsgp_ws.grad_f.data();

  const int N = data.N;
  const bool is_binomial = (data.model_type == ModelType::BINOMIAL ||
                            data.model_type == ModelType::BETA_BINOMIAL);

  // Use thread-local workspace for eta/resid buffers (zero allocation)
  vec_grad_ws().init(N);

  // --- Pass 1: Vectorized linear predictor computation ---
  using RowMajorMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  using VectorXd = Eigen::VectorXd;

  Eigen::Map<const RowMajorMatrix> X_num(data.X_num_flat.data(), N, data.p_num);
  Eigen::Map<const VectorXd> b_num(beta_num, data.p_num);
  Eigen::Map<VectorXd> eta_n(vec_grad_ws().eta_num.data(), N);
  eta_n.noalias() = X_num * b_num;

  if (!is_binomial) {
    Eigen::Map<const RowMajorMatrix> X_denom(data.X_denom_flat.data(), N, data.p_denom);
    Eigen::Map<const VectorXd> b_denom(beta_denom, data.p_denom);
    Eigen::Map<VectorXd> eta_d(vec_grad_ws().eta_denom.data(), N);
    eta_d.noalias() = X_denom * b_denom;
  }

  // Add RE (expand to dense and vectorized add)
  if (layout.has_re) {
    for (int i = 0; i < N; i++) {
      if (data.re_group[i] > 0) {
        int g = data.re_group[i] - 1;
        double re_eff = re_value_for_eta(re, g, sigma_re, data.re_parameterization);
        vec_grad_ws().eta_num[i] += re_eff;
        if (!is_binomial) vec_grad_ws().eta_denom[i] += re_eff;
      }
    }
  }

  // Add temporal (expand to observation level)
  if (layout.has_temporal && !data.temporal_time_idx.empty()) {
    for (int i = 0; i < N; i++) {
      if (i < (int)data.temporal_time_idx.size() && data.temporal_time_idx[i] > 0) {
        int t = data.temporal_time_idx[i] - 1;
        int g = data.temporal_group_idx[i] - 1;
        int t_idx = g * data.n_times + t;
        if (t_idx >= 0 && t_idx < T_len) {
          vec_grad_ws().eta_num[i] += phi_temporal[t_idx];
          if (!is_binomial && data.temporal_shared) vec_grad_ws().eta_denom[i] += phi_temporal[t_idx];
        }
      }
    }
  }

  // Add HSGP spatial effect (vectorized)
  Eigen::Map<const VectorXd> hsgp_fv(hsgp_f, N);
  eta_n += hsgp_fv;
  if (data.hsgp_data.shared && !is_binomial) {
    Eigen::Map<VectorXd>(vec_grad_ws().eta_denom.data(), N) += hsgp_fv;
  }

  // --- Pass 2+3: Vectorized residuals + beta grads (template-dispatched) ---
  {
    double grad_phi_num_lik = 0.0, grad_phi_denom_lik = 0.0;
    vectorized::dispatch_residuals_and_beta_grads(
        data, layout,
        vec_grad_ws().eta_num.data(), vec_grad_ws().eta_denom.data(),
        vec_grad_ws().resid_num.data(), vec_grad_ws().resid_denom.data(),
        grad.data(), grad_phi_num_lik, grad_phi_denom_lik,
        obs_log_lik, fuse_lp, phi_num, phi_denom, vec_grad_ws());
  }

  // Accumulate grad_f for HSGP from residuals
  for (int i = 0; i < N; i++) {
    grad_f_ptr[i] = data.hsgp_data.shared
        ? (vec_grad_ws().resid_num[i] + vec_grad_ws().resid_denom[i])
        : vec_grad_ws().resid_num[i];
  }

  // RE gradients (scatter from residuals to group-level)
  if (layout.has_re) {
    for (int i = 0; i < N; i++) {
      scatter_re_gradient(data, layout, i, vec_grad_ws().resid_num[i],
                          vec_grad_ws().resid_denom[i], grad.data());
    }
  }

  // Temporal likelihood gradients (scatter to temporal buffer)
  if (layout.has_temporal && !data.temporal_time_idx.empty()) {
    for (int i = 0; i < N; i++) {
      if (i < (int)data.temporal_time_idx.size() && data.temporal_time_idx[i] > 0) {
        int t = data.temporal_time_idx[i] - 1;
        int g = data.temporal_group_idx[i] - 1;
        int t_idx = g * data.n_times + t;
        if (t_idx >= 0 && t_idx < T_len) {
          double lik_grad = data.temporal_shared ?
            (vec_grad_ws().resid_num[i] + vec_grad_ws().resid_denom[i]) :
            vec_grad_ws().resid_num[i];
          grad_temporal_lik[t_idx] += lik_grad;
        }
      }
    }
  }

  // Compute HSGP parameter gradients using workspace (zero allocation)
  double hsgp_grad_log_sigma2, hsgp_grad_log_lengthscale;
  ratiod_hsgp::hsgp_compute_gradients_ws(hsgp_beta_ptr, sigma2_hsgp, lengthscale_hsgp,
                                          data.hsgp_data, hsgp_ws,
                                          hsgp_grad_log_sigma2, hsgp_grad_log_lengthscale);

  // Add likelihood contribution to HSGP gradients
  for (int j = 0; j < m_total; j++) {
    grad[layout.hsgp_beta_start + j] += hsgp_ws.grad_beta_out[j];
  }
  grad[layout.log_sigma2_hsgp_idx] += hsgp_grad_log_sigma2;
  grad[layout.log_lengthscale_hsgp_idx] += hsgp_grad_log_lengthscale;

  // Temporal GMRF gradients
  if (layout.has_temporal && T_len > 0) {
    temporal_gmrf_prior_grad(data, layout, tau_temporal, rho_ar1,
                             tview, T_len, grad_temporal_lik.data(), grad.data());
  }

    // Non-centered RE chain rule transformation
    re_gradient_nc_transform(data, layout, params.data(), grad.data(), sigma_re);

    if (fuse_lp) *log_post_out = compute_log_post(params, data, layout, /*skip_obs_loop=*/true) + obs_log_lik;
}

// =====================================================================
// HSGP-MSGP gradient (hand-coded)
// Two independent HSGP evaluations with shared basis matrix
// =====================================================================

void compute_gradient_msgp_hsgp(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    std::vector<double>& grad,
    double* log_post_out = nullptr
) {
    // Two thread-local HSGP workspaces (one per scale)
    RATIOD_TLS_WORKSPACE(ratiod_hsgp::HSGPWorkspace, ws_local);
    RATIOD_TLS_WORKSPACE(ratiod_hsgp::HSGPWorkspace, ws_regional);
    ws_local.init(data.N, data.msgp_hsgp_data.m_total);
    ws_regional.init(data.N, data.msgp_hsgp_data.m_total);

    const bool fuse_lp = (log_post_out != nullptr) && !layout.has_zi;
    if (log_post_out && layout.has_zi) *log_post_out = compute_log_post(params, data, layout);
    double obs_log_lik = 0.0;
    int n_params = params.size();
    grad.assign(n_params, 0.0);

    // Extract common parameters
    auto cp = extract_common_params(params, layout);
    const double* beta_num = cp.beta_num;
    const double* beta_denom = cp.beta_denom;
    double sigma_re = cp.sigma_re;
    const double* re = cp.re;
    double phi_num = cp.phi_num;
    double phi_denom = cp.phi_denom;

    // HSGP-MSGP parameters
    double log_sigma2_local = params[layout.log_sigma2_gp_local_idx];
    double log_ls_local = params[layout.log_phi_gp_local_idx];
    double sigma2_local = std::exp(log_sigma2_local);
    double ls_local = std::exp(log_ls_local);

    double log_sigma2_regional = params[layout.log_sigma2_gp_regional_idx];
    double log_ls_regional = params[layout.log_phi_gp_regional_idx];
    double sigma2_regional = std::exp(log_sigma2_regional);
    double ls_regional = std::exp(log_ls_regional);

    int m_total = data.msgp_hsgp_data.m_total;
    const double* beta_local = &params[layout.gp_local_start];
    const double* beta_regional = &params[layout.gp_regional_start];

    // Evaluate HSGP spatial effects for both scales
    ratiod_hsgp::hsgp_evaluate_ws(beta_local, sigma2_local, ls_local,
                                   data.msgp_hsgp_data, ws_local);
    ratiod_hsgp::hsgp_evaluate_ws(beta_regional, sigma2_regional, ls_regional,
                                   data.msgp_hsgp_data, ws_regional);

    // Temporal parameters (for HSGP-MSGP + temporal combinations)
    double tau_temporal = 0.0;
    int T_len = 0;
    const double* phi_temporal = nullptr;
    TemporalView tview;
    double rho_ar1 = 0.0;
    if (layout.has_temporal) {
        tau_temporal = std::exp(params[layout.log_tau_temporal_idx]);
        T_len = layout.temporal_end - layout.temporal_start;
        tview.read(params, data, layout);
        phi_temporal = tview.phi;
        rho_ar1 = read_temporal_rho_ar1(params, data, layout);
    }

    // Zero grad_f buffers
    std::memset(ws_local.grad_f.data(), 0, data.N * sizeof(double));
    // ws_regional.grad_f will get the same values (shared residuals)
    std::vector<double> grad_temporal_lik(T_len, 0.0);

    // --- Prior gradients ---
    beta_gradient_prior(data, layout, beta_num, beta_denom, grad.data());
    re_gradient_prior(data, layout, re, grad.data(), sigma_re);
    phi_gradient_prior(data, layout, phi_num, phi_denom, grad.data());

    // PC priors on sigma for both scales
    double sigma_local = std::sqrt(sigma2_local);
    double rate_local = -std::log(data.ms_sigma2_local_prior_alpha) / data.ms_sigma2_local_prior_U;
    grad[layout.log_sigma2_gp_local_idx] = -0.5 * rate_local * sigma_local + 0.5 - 0.5;

    double sigma_regional = std::sqrt(sigma2_regional);
    double rate_regional = -std::log(data.ms_sigma2_regional_prior_alpha) / data.ms_sigma2_regional_prior_U;
    grad[layout.log_sigma2_gp_regional_idx] = -0.5 * rate_regional * sigma_regional + 0.5 - 0.5;

    // LogNormal priors on lengthscales
    double z_local = (log_ls_local - data.ms_log_ls_local_mean) / data.ms_log_ls_local_sd;
    grad[layout.log_phi_gp_local_idx] = -z_local / data.ms_log_ls_local_sd;

    double z_regional = (log_ls_regional - data.ms_log_ls_regional_mean) / data.ms_log_ls_regional_sd;
    grad[layout.log_phi_gp_regional_idx] = -z_regional / data.ms_log_ls_regional_sd;

    // N(0, I) prior on beta: d/d(beta_j) = -beta_j
    for (int j = 0; j < m_total; j++) {
        grad[layout.gp_local_start + j] = -beta_local[j];
        grad[layout.gp_regional_start + j] = -beta_regional[j];
    }

    // Temporal prior on tau/rho
    if (layout.has_temporal) {
        tau_temporal_prior_grad(data, layout, tau_temporal, grad.data());
        temporal_rho_prior_grad(data, layout, rho_ar1, grad.data());
    }

    // --- Vectorized observation loop ---
    const int N = data.N;
    const bool is_binomial = (data.model_type == ModelType::BINOMIAL ||
                              data.model_type == ModelType::BETA_BINOMIAL);
    const bool shared = data.multiscale_gp_data.shared;

    vec_grad_ws().init(N);

    // Pass 1: vectorized linear predictor
    using RowMajorMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using VectorXd = Eigen::VectorXd;

    Eigen::Map<const RowMajorMatrix> X_num(data.X_num_flat.data(), N, data.p_num);
    Eigen::Map<const VectorXd> b_num(beta_num, data.p_num);
    Eigen::Map<VectorXd> eta_n(vec_grad_ws().eta_num.data(), N);
    eta_n.noalias() = X_num * b_num;

    if (!is_binomial) {
        Eigen::Map<const RowMajorMatrix> X_denom(data.X_denom_flat.data(), N, data.p_denom);
        Eigen::Map<const VectorXd> b_denom(beta_denom, data.p_denom);
        Eigen::Map<VectorXd> eta_d(vec_grad_ws().eta_denom.data(), N);
        eta_d.noalias() = X_denom * b_denom;
    }

    // Add RE
    if (layout.has_re) {
        for (int i = 0; i < N; i++) {
            if (data.re_group[i] > 0) {
                int g = data.re_group[i] - 1;
                double re_eff = re_value_for_eta(re, g, sigma_re, data.re_parameterization);
                vec_grad_ws().eta_num[i] += re_eff;
                if (!is_binomial) vec_grad_ws().eta_denom[i] += re_eff;
            }
        }
    }

    // Add temporal
    if (layout.has_temporal && !data.temporal_time_idx.empty()) {
        for (int i = 0; i < N; i++) {
            if (i < (int)data.temporal_time_idx.size() && data.temporal_time_idx[i] > 0) {
                int t = data.temporal_time_idx[i] - 1;
                int g = data.temporal_group_idx[i] - 1;
                int t_idx = g * data.n_times + t;
                if (t_idx >= 0 && t_idx < T_len) {
                    vec_grad_ws().eta_num[i] += phi_temporal[t_idx];
                    if (!is_binomial && data.temporal_shared) vec_grad_ws().eta_denom[i] += phi_temporal[t_idx];
                }
            }
        }
    }

    // Add combined HSGP-MSGP spatial effect: f_local + f_regional (vectorized)
    Eigen::Map<const VectorXd> f_local_v(ws_local.hsgp_f.data(), N);
    Eigen::Map<const VectorXd> f_regional_v(ws_regional.hsgp_f.data(), N);
    eta_n += f_local_v + f_regional_v;
    if (shared && !is_binomial) {
        Eigen::Map<VectorXd>(vec_grad_ws().eta_denom.data(), N) += f_local_v + f_regional_v;
    }

    // Pass 2+3: vectorized residuals + beta grads
    {
        double grad_phi_num_lik = 0.0, grad_phi_denom_lik = 0.0;
        vectorized::dispatch_residuals_and_beta_grads(
            data, layout,
            vec_grad_ws().eta_num.data(), vec_grad_ws().eta_denom.data(),
            vec_grad_ws().resid_num.data(), vec_grad_ws().resid_denom.data(),
            grad.data(), grad_phi_num_lik, grad_phi_denom_lik,
            obs_log_lik, fuse_lp, phi_num, phi_denom, vec_grad_ws());
    }

    // Accumulate grad_f for both HSGP scales from residuals
    // Both scales share the same grad_f (additive model)
    for (int i = 0; i < N; i++) {
        double gf = shared ? (vec_grad_ws().resid_num[i] + vec_grad_ws().resid_denom[i])
                           : vec_grad_ws().resid_num[i];
        ws_local.grad_f[i] = gf;
    }
    // Copy to regional workspace (same values)
    std::memcpy(ws_regional.grad_f.data(), ws_local.grad_f.data(), N * sizeof(double));

    // RE gradients
    if (layout.has_re) {
        for (int i = 0; i < N; i++) {
            scatter_re_gradient(data, layout, i, vec_grad_ws().resid_num[i],
                                vec_grad_ws().resid_denom[i], grad.data());
        }
    }

    // Temporal likelihood gradients
    if (layout.has_temporal && !data.temporal_time_idx.empty()) {
        for (int i = 0; i < N; i++) {
            if (i < (int)data.temporal_time_idx.size() && data.temporal_time_idx[i] > 0) {
                int t = data.temporal_time_idx[i] - 1;
                int g = data.temporal_group_idx[i] - 1;
                int t_idx = g * data.n_times + t;
                if (t_idx >= 0 && t_idx < T_len) {
                    double lik_grad = data.temporal_shared ?
                        (vec_grad_ws().resid_num[i] + vec_grad_ws().resid_denom[i]) :
                        vec_grad_ws().resid_num[i];
                    grad_temporal_lik[t_idx] += lik_grad;
                }
            }
        }
    }

    // Compute HSGP parameter gradients for both scales
    double grad_log_sigma2_local, grad_log_ls_local;
    ratiod_hsgp::hsgp_compute_gradients_ws(beta_local, sigma2_local, ls_local,
                                            data.msgp_hsgp_data, ws_local,
                                            grad_log_sigma2_local, grad_log_ls_local);

    double grad_log_sigma2_regional, grad_log_ls_regional;
    ratiod_hsgp::hsgp_compute_gradients_ws(beta_regional, sigma2_regional, ls_regional,
                                            data.msgp_hsgp_data, ws_regional,
                                            grad_log_sigma2_regional, grad_log_ls_regional);

    // Add likelihood contribution to HSGP gradients
    for (int j = 0; j < m_total; j++) {
        grad[layout.gp_local_start + j] += ws_local.grad_beta_out[j];
        grad[layout.gp_regional_start + j] += ws_regional.grad_beta_out[j];
    }
    grad[layout.log_sigma2_gp_local_idx] += grad_log_sigma2_local;
    grad[layout.log_phi_gp_local_idx] += grad_log_ls_local;
    grad[layout.log_sigma2_gp_regional_idx] += grad_log_sigma2_regional;
    grad[layout.log_phi_gp_regional_idx] += grad_log_ls_regional;

    // Temporal GMRF gradients
    if (layout.has_temporal && T_len > 0) {
        temporal_gmrf_prior_grad(data, layout, tau_temporal, rho_ar1,
                                 tview, T_len, grad_temporal_lik.data(), grad.data());
    }

    // Non-centered RE chain rule transformation
    re_gradient_nc_transform(data, layout, params.data(), grad.data(), sigma_re);

    if (fuse_lp) *log_post_out = compute_log_post(params, data, layout, /*skip_obs_loop=*/true) + obs_log_lik;
}

// =====================================================================
// Multiscale temporal gradient (hand-coded, rows 15, 45, 75)
// Uses analytical gradients from hmc_multiscale_temporal_grad.h
// Supports optional ICAR/BYM2 spatial
// =====================================================================

void compute_gradient_ms_temporal_handcoded(
    const std::vector<double>& params_in,
    const ModelData& data,
    const ParamLayout& layout,
    std::vector<double>& grad,
    double* log_post_out = nullptr
) {
    // Centre the spatial block once, matching compute_log_post; see
    // spatial_field_constraint.h. The returned gradient is with respect to the
    // raw parameters, which is what spatial_gmrf_prior_grad projects for.
    const bool center_spatial = spatial_block_is_centred(data, layout);
    ratiod_constraints::CenteredSpatialParams centering(
        params_in, center_spatial, layout.spatial_start, data.n_spatial_units);
    const std::vector<double>& params = centering.params();

    const bool fuse_lp = (log_post_out != nullptr);
    double obs_log_lik = 0.0;
    int n_params = params.size();
    grad.assign(n_params, 0.0);

    // Extract common parameters
    auto cp = extract_common_params(params, layout);
    const double* beta_num = cp.beta_num;
    const double* beta_denom = cp.beta_denom;
    double sigma_re = cp.sigma_re;
    const double* re = cp.re;
    double phi_num = cp.phi_num;
    double phi_denom = cp.phi_denom;

    // Spatial parameters (ICAR/BYM2 if present)
    double tau_spatial = 0.0;
    const double* spatial_phi = nullptr;
    double sigma_s_bym2 = 0.0, sigma_u_bym2 = 0.0;
    double rho_bym2 = 0.5;
    const double* theta_bym2 = nullptr;
    if (layout.has_spatial) {
        if (!layout.is_bym2) tau_spatial = std::exp(params[layout.log_tau_spatial_idx]);
        spatial_phi = &params[layout.spatial_start];
        if (layout.is_bym2) {
            double sigma_total = std::exp(params[layout.log_sigma_bym2_idx]);
            double logit_rho = params[layout.logit_rho_bym2_idx];
            rho_bym2 = 1.0 / (1.0 + std::exp(-logit_rho));
            sigma_s_bym2 = sigma_total * std::sqrt(rho_bym2);
            sigma_u_bym2 = sigma_total * std::sqrt(1.0 - rho_bym2);
            theta_bym2 = &params[layout.theta_bym2_start];
        }
    }

    // Multiscale temporal parameters
    const auto& mst = data.multiscale_temporal_data;
    int n_trend = layout.trend_end - layout.trend_start;
    int n_seasonal = layout.seasonal_end - layout.seasonal_start;
    int n_short = layout.short_term_end - layout.short_term_start;

    const double* trend = (n_trend > 0) ? &params[layout.trend_start] : nullptr;
    const double* seasonal = (n_seasonal > 0) ? &params[layout.seasonal_start] : nullptr;
    const double* short_term = (n_short > 0) ? &params[layout.short_term_start] : nullptr;

    double sigma2_trend = (n_trend > 0) ? std::exp(params[layout.log_sigma2_trend_idx]) : 1.0;
    double sigma2_seasonal = (n_seasonal > 0) ? std::exp(params[layout.log_sigma2_seasonal_idx]) : 1.0;
    double sigma2_short = (n_short > 0) ? std::exp(params[layout.log_sigma2_short_idx]) : 1.0;
    double rho_short = 0.5;
    if (mst.short_term_type == TemporalType::AR1 && layout.logit_rho_short_idx >= 0) {
        double logit_rho = params[layout.logit_rho_short_idx];
        double u = 1.0 / (1.0 + std::exp(-logit_rho));
        rho_short = 2.0 * u - 1.0;
    }

    // =========================================================================
    // Prior gradients
    // =========================================================================
    beta_gradient_prior(data, layout, beta_num, beta_denom, grad.data());
    re_gradient_prior(data, layout, re, grad.data(), sigma_re);
    phi_gradient_prior(data, layout, phi_num, phi_denom, grad.data());

    // Spatial prior gradients (ICAR/BYM2)
    if (layout.has_spatial && !layout.is_bym2) {
        // ICAR: Gamma prior on tau_spatial
        grad[layout.log_tau_spatial_idx] = (data.tau_spatial_shape - 1.0) - data.tau_spatial_rate * tau_spatial + 1.0;
    }
    if (layout.is_bym2) {
        // BYM2 Riebler prior gradients
        double sigma_total = sigma_s_bym2 / std::sqrt(rho_bym2);
        double ratio = sigma_total / data.sigma_re_scale;
        grad[layout.log_sigma_bym2_idx] = -2.0 * ratio * ratio / (1.0 + ratio * ratio) + 1.0;
        grad[layout.logit_rho_bym2_idx] = 1.0 - 2.0 * rho_bym2;
    }

    // PC priors on multiscale temporal variances + Jacobian (+1 for log_sigma2 = log(exp(·)))
    if (n_trend > 0) {
        grad[layout.log_sigma2_trend_idx] = ratiod_temporal_grad::pc_prior_grad_log_sigma2(
            sigma2_trend, data.ms_sigma2_trend_prior_U, data.ms_sigma2_trend_prior_alpha) + 1.0;
    }
    if (n_seasonal > 0) {
        grad[layout.log_sigma2_seasonal_idx] = ratiod_temporal_grad::pc_prior_grad_log_sigma2(
            sigma2_seasonal, data.ms_sigma2_seasonal_prior_U, data.ms_sigma2_seasonal_prior_alpha) + 1.0;
    }
    if (n_short > 0) {
        grad[layout.log_sigma2_short_idx] = ratiod_temporal_grad::pc_prior_grad_log_sigma2(
            sigma2_short, data.ms_sigma2_short_prior_U, data.ms_sigma2_short_prior_alpha) + 1.0;
    }
    if (mst.short_term_type == TemporalType::AR1 && layout.logit_rho_short_idx >= 0) {
        // The rho prior in the coordinate the sampler moves
        double u = (rho_short + 1.0) / 2.0;
        grad[layout.logit_rho_short_idx] = ratiod_ar1::log_prior_logit_rho_grad(
            u, ratiod_ar1::RHO_PRIOR_A, ratiod_ar1::RHO_PRIOR_B);
    }

    // =========================================================================
    // Likelihood loop
    // =========================================================================
    std::vector<double> grad_trend_lik(n_trend, 0.0);
    std::vector<double> grad_seasonal_lik(n_seasonal, 0.0);
    std::vector<double> grad_short_lik(n_short, 0.0);
    std::vector<double> grad_spatial_lik;
    if (layout.has_spatial) grad_spatial_lik.assign(data.n_spatial_units, 0.0);
    std::vector<double> grad_theta_lik;
    if (layout.is_bym2) grad_theta_lik.assign(data.n_spatial_units, 0.0);

    // Vectorized eta computation
    const int N = data.N;
    const bool is_binomial = (data.model_type == ModelType::BINOMIAL ||
                              data.model_type == ModelType::BETA_BINOMIAL);
    using RowMajorMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using VectorXd = Eigen::VectorXd;
    vec_grad_ws().init(N);
    Eigen::Map<const RowMajorMatrix> X_num(data.X_num_flat.data(), N, data.p_num);
    Eigen::Map<const VectorXd> b_num(beta_num, data.p_num);
    Eigen::Map<VectorXd> eta_n(vec_grad_ws().eta_num.data(), N);
    eta_n.noalias() = X_num * b_num;
    if (!is_binomial) {
        Eigen::Map<const RowMajorMatrix> X_denom(data.X_denom_flat.data(), N, data.p_denom);
        Eigen::Map<const VectorXd> b_denom(beta_denom, data.p_denom);
        Eigen::Map<VectorXd> eta_d(vec_grad_ws().eta_denom.data(), N);
        eta_d.noalias() = X_denom * b_denom;
    } else {
        std::memset(vec_grad_ws().eta_denom.data(), 0, N * sizeof(double));
    }

    // Pre-assemble temporal effect per time point (avoids per-obs modulo)
    const int n_times = mst.n_times;
    // eta reads the arms' effects, which are the blocks themselves unless the
    // intrinsic arms are sampled in their non-centred coordinate.
    const auto ms_arms = ratiod_temporal::ms_read_arms(
        trend, n_trend, seasonal, n_seasonal, sigma2_trend, sigma2_seasonal, mst);
    std::vector<double> ms_effect_by_time(n_times, 0.0);
    ratiod_temporal::compute_ms_effect_by_time(
        ms_arms.trend, n_trend, ms_arms.seasonal, n_seasonal, short_term, n_short,
        mst.seasonal_period, n_times, ms_effect_by_time.data());

    // Per-obs: add RE + spatial + multiscale temporal effects
    std::vector<int> obs_s_unit(N, -1);
    std::vector<int> obs_t_idx(N, -1);
    for (int i = 0; i < N; i++) {
        if (layout.has_re && data.re_group[i] > 0) {
            double re_eff = re_value_for_eta(re, data.re_group[i] - 1, sigma_re, data.re_parameterization);
            vec_grad_ws().eta_num[i] += re_eff;
            if (!is_binomial) vec_grad_ws().eta_denom[i] += re_eff;
        }
        if (layout.has_spatial && data.spatial_group[i] > 0) {
            int s_unit = data.spatial_group[i] - 1;
            obs_s_unit[i] = s_unit;
            double spatial_eff;
            if (layout.is_bym2) {
                spatial_eff = sigma_s_bym2 * data.bym2_scale_factor * spatial_phi[s_unit] + sigma_u_bym2 * theta_bym2[s_unit];
            } else {
                spatial_eff = spatial_phi[s_unit];
            }
            vec_grad_ws().eta_num[i] += spatial_eff;
            if (!is_binomial) vec_grad_ws().eta_denom[i] += spatial_eff;
        }
        if (!mst.time_index.empty() && i < (int)mst.time_index.size() && mst.time_index[i] > 0) {
            int t_idx = mst.time_index[i] - 1;
            obs_t_idx[i] = t_idx;
            vec_grad_ws().eta_num[i] += ms_effect_by_time[t_idx];
            if (!is_binomial && mst.shared) vec_grad_ws().eta_denom[i] += ms_effect_by_time[t_idx];
        }
    }

    {
        double grad_phi_num_lik = 0.0, grad_phi_denom_lik = 0.0;
        vectorized::dispatch_residuals_and_beta_grads(
            data, layout,
            vec_grad_ws().eta_num.data(), vec_grad_ws().eta_denom.data(),
            vec_grad_ws().resid_num.data(), vec_grad_ws().resid_denom.data(),
            grad.data(), grad_phi_num_lik, grad_phi_denom_lik,
            obs_log_lik, fuse_lp, phi_num, phi_denom, vec_grad_ws());
    }

    // Scatter residuals
    for (int i = 0; i < N; i++) {
        double dLL_num = vec_grad_ws().resid_num[i];
        double dLL_denom = vec_grad_ws().resid_denom[i];
        if (layout.has_re && data.re_group[i] > 0) {
            grad[layout.re_start + data.re_group[i] - 1] += dLL_num + dLL_denom;
        }
        double dLL_shared = dLL_num + dLL_denom;
        int s_unit = obs_s_unit[i];
        if (layout.has_spatial && s_unit >= 0) {
            grad_spatial_lik[s_unit] += dLL_shared;
            if (layout.is_bym2) grad_theta_lik[s_unit] += dLL_shared;
        }
        int t_idx = obs_t_idx[i];
        if (t_idx >= 0) {
            double dLL_temporal = mst.shared ? dLL_shared : dLL_num;
            if (trend != nullptr && t_idx < n_trend) grad_trend_lik[t_idx] += dLL_temporal;
            if (seasonal != nullptr && mst.seasonal_period > 0) {
                int s_idx = t_idx % mst.seasonal_period;
                if (s_idx < n_seasonal) grad_seasonal_lik[s_idx] += dLL_temporal;
            }
            if (short_term != nullptr && t_idx < n_short) grad_short_lik[t_idx] += dLL_temporal;
        }
    }

    // Spatial GMRF prior gradients (ICAR/BYM2)
    if (layout.has_spatial) {
        spatial_lik_grad_to_param_scale(data, layout, sigma_s_bym2, sigma_u_bym2,
                                        theta_bym2, grad_theta_lik.data(),
                                        grad_spatial_lik.data(), grad.data());
        spatial_gmrf_prior_grad(data, layout, spatial_phi, spatial_phi,
                                centering.raw_sum(),
                                centering.active(), tau_spatial,
                                sigma_s_bym2, sigma_u_bym2, rho_bym2, 0.0, theta_bym2,
                                grad_spatial_lik.data(), grad.data());
    }

    // =========================================================================
    // Multiscale temporal GMRF prior gradients
    // =========================================================================
    ratiod_temporal_grad::MultiscaleTemporalGradients ms_grads;
    ratiod_temporal_grad::multiscale_temporal_prior_gradients(
        trend, n_trend,
        seasonal, n_seasonal,
        short_term, n_short,
        sigma2_trend, sigma2_seasonal, sigma2_short, rho_short,
        mst, ms_grads);
    double nc_g_log_s2_trend = 0.0, nc_g_log_s2_seasonal = 0.0;
    ratiod_temporal_grad::project_ms_lik_gradients(
        grad_trend_lik.data(), n_trend, grad_seasonal_lik.data(), n_seasonal, mst,
        {trend, ms_arms.trend, sigma2_trend, &nc_g_log_s2_trend},
        {seasonal, ms_arms.seasonal, sigma2_seasonal, &nc_g_log_s2_seasonal});

    for (int t = 0; t < n_trend; t++) grad[layout.trend_start + t] = grad_trend_lik[t] + ms_grads.grad_trend[t];
    for (int t = 0; t < n_seasonal; t++) grad[layout.seasonal_start + t] = grad_seasonal_lik[t] + ms_grads.grad_seasonal[t];
    for (int t = 0; t < n_short; t++) grad[layout.short_term_start + t] = grad_short_lik[t] + ms_grads.grad_short_term[t];
    if (n_trend > 0)
        grad[layout.log_sigma2_trend_idx] += ms_grads.grad_log_sigma2_trend + nc_g_log_s2_trend;
    if (n_seasonal > 0)
        grad[layout.log_sigma2_seasonal_idx] += ms_grads.grad_log_sigma2_seasonal + nc_g_log_s2_seasonal;
    if (n_short > 0) grad[layout.log_sigma2_short_idx] += ms_grads.grad_log_sigma2_short;
    if (mst.short_term_type == TemporalType::AR1 && layout.logit_rho_short_idx >= 0) {
        grad[layout.logit_rho_short_idx] += ms_grads.grad_logit_rho_short;
    }

    // Non-centered RE chain rule transformation
    re_gradient_nc_transform(data, layout, params.data(), grad.data(), sigma_re);

    // params_in, not params: compute_log_post centres its own copy and recovers
    // phi_raw_sum from the raw field, which the centred copy has zeroed.
    if (fuse_lp) *log_post_out = compute_log_post(params_in, data, layout, /*skip_obs_loop=*/true) + obs_log_lik;
}

// =====================================================================
// Spatiotemporal interaction gradient (hand-coded, rows 28-29, 58-59, 90-91)
// Supports Knorr-Held Type I-IV with ICAR spatial + RW1/RW2 temporal
// =====================================================================

void compute_gradient_spatiotemporal_handcoded(
    const std::vector<double>& params_in,
    const ModelData& data,
    const ParamLayout& layout,
    std::vector<double>& grad,
    double* log_post_out = nullptr
) {
    // Centre the spatial block once, matching compute_log_post; see
    // spatial_field_constraint.h. The returned gradient is with respect to the
    // raw parameters, which is what spatial_gmrf_prior_grad projects for.
    const bool center_spatial = spatial_block_is_centred(data, layout);
    ratiod_constraints::CenteredSpatialParams centering(
        params_in, center_spatial, layout.spatial_start, data.n_spatial_units);
    const std::vector<double>& params = centering.params();

    const bool fuse_lp = (log_post_out != nullptr);
    double obs_log_lik = 0.0;
    int n_params = params.size();
    grad.assign(n_params, 0.0);

    // Extract common parameters
    auto cp = extract_common_params(params, layout);
    const double* beta_num = cp.beta_num;
    const double* beta_denom = cp.beta_denom;
    double sigma_re = cp.sigma_re;
    const double* re = cp.re;
    double phi_num = cp.phi_num;
    double phi_denom = cp.phi_denom;

    // Spatial parameters (ICAR/BYM2)
    double tau_spatial = 0.0;
    const double* spatial_phi = nullptr;
    double sigma_s_bym2 = 0.0, sigma_u_bym2 = 0.0;
    double rho_bym2 = 0.5;
    const double* theta_bym2 = nullptr;
    if (layout.has_spatial) {
        if (!layout.is_bym2) tau_spatial = std::exp(params[layout.log_tau_spatial_idx]);
        spatial_phi = &params[layout.spatial_start];
        if (layout.is_bym2) {
            double sigma_total = std::exp(params[layout.log_sigma_bym2_idx]);
            double logit_rho = params[layout.logit_rho_bym2_idx];
            rho_bym2 = 1.0 / (1.0 + std::exp(-logit_rho));
            sigma_s_bym2 = sigma_total * std::sqrt(rho_bym2);
            sigma_u_bym2 = sigma_total * std::sqrt(1.0 - rho_bym2);
            theta_bym2 = &params[layout.theta_bym2_start];
        }
    }

    // Temporal parameters
    double tau_temporal = 0.0;
    int T_temporal = 0;
    const double* phi_temporal = nullptr;
    TemporalView tview;
    double rho_ar1 = 0.0;
    if (layout.has_temporal) {
        tau_temporal = std::exp(params[layout.log_tau_temporal_idx]);
        T_temporal = layout.temporal_end - layout.temporal_start;
        tview.read(params, data, layout);
        phi_temporal = tview.phi;
        rho_ar1 = read_temporal_rho_ar1(params, data, layout);
    }

    // Spatiotemporal interaction parameters
    const auto& st = data.spatiotemporal_data;
    int ST = st.n_params;
    double tau_st = std::exp(params[layout.log_tau_st_idx]);
    // The interaction's own time-margin correlation. The layout carries one
    // only where the margin is AR1; elsewhere 0 leaves R(rho) at the identity
    // and every term reading it inert.
    double rho_st = (layout.logit_rho_st_idx >= 0)
        ? ratiod_ar1::rho_from_logit(params[layout.logit_rho_st_idx])
        : 0.0;
    double sigma2_st_hsgp = 1.0, lengthscale_st_hsgp = 1.0;
    if (data.st_is_hsgp) {
        sigma2_st_hsgp = std::exp(params[layout.log_sigma2_st_hsgp_idx]);
        lengthscale_st_hsgp = std::exp(params[layout.log_lengthscale_st_hsgp_idx]);
    }

    // The GP types' two ranges. The prior is uniform inside its bounds, so it
    // leaves only the log Jacobian on each; the field's own contribution to
    // both arrives with the interaction's prior below.
    double phi_st_space = 1.0, phi_st_time = 1.0;
    if (layout.is_st_gp) {
        phi_st_space = std::exp(params[layout.log_phi_st_space_idx]);
        phi_st_time = std::exp(params[layout.log_phi_st_time_idx]);
    }

    // NC reparameterization: params store z, reconstruct delta
    const bool st_use_nc = (data.st_parameterization == 1 &&
                            st.type == STType::TYPE_IV);
    const double* z_or_delta = &params[layout.st_delta_start];
    RATIOD_TLS_WORKSPACE(std::vector<double>, st_delta_buf);
    const double* delta;
    double inv_scale = 1.0;
    if (st_use_nc) {
        inv_scale = 1.0 / std::sqrt(tau_st);
        st_delta_buf.resize(ST);
        for (int k = 0; k < ST; k++) st_delta_buf[k] = z_or_delta[k] * inv_scale;
        delta = st_delta_buf.data();
    } else {
        delta = z_or_delta;
    }

    // =========================================================================
    // Prior gradients for base parameters
    // =========================================================================
    beta_gradient_prior(data, layout, beta_num, beta_denom, grad.data());
    re_gradient_prior(data, layout, re, grad.data(), sigma_re);
    phi_gradient_prior(data, layout, phi_num, phi_denom, grad.data());

    // Spatial prior (ICAR: Gamma on tau)
    if (layout.has_spatial && !layout.is_bym2) {
        grad[layout.log_tau_spatial_idx] = (data.tau_spatial_shape - 1.0) - data.tau_spatial_rate * tau_spatial + 1.0;
    }
    if (layout.is_bym2) {
        // BYM2 Riebler prior gradients
        double sigma_total = sigma_s_bym2 / std::sqrt(rho_bym2);
        double ratio = sigma_total / data.sigma_re_scale;
        grad[layout.log_sigma_bym2_idx] = -2.0 * ratio * ratio / (1.0 + ratio * ratio) + 1.0;
        grad[layout.logit_rho_bym2_idx] = 1.0 - 2.0 * rho_bym2;
    }

    // Temporal prior (Gamma on tau, Beta on rho)
    if (layout.has_temporal) {
        tau_temporal_prior_grad(data, layout, tau_temporal, grad.data());
        temporal_rho_prior_grad(data, layout, rho_ar1, grad.data());
    }

    // ST interaction prior on tau_st (PC prior: exponential on sigma_st)
    {
        double sigma_st = 1.0 / std::sqrt(tau_st);
        double lambda = -std::log(data.st_sigma2_prior_alpha) / data.st_sigma2_prior_U;
        // d log_prior / d log_tau = d/d(tau) * tau
        // log_prior = log(lambda) - lambda*sigma - log(2*sigma) + log_tau (Jacobian)
        // sigma = 1/sqrt(tau), d sigma/d tau = -0.5 * tau^{-3/2}
        // d log_prior / d tau = (lambda/(2*sigma*tau) + 1/(2*sigma^2*tau))* (-sigma/tau ... )
        // Simpler: work through chain rule
        // d/d(log_tau) = [lambda * 0.5 * sigma - 0.5] * (-1) + 1
        // = 0.5 * (1 - lambda * sigma) + 1 - 0.5
        // = 1.0 - 0.5 * lambda * sigma
        // Actually, from log_post code:
        // log p = log(lambda) - lambda*sigma - log(2*sigma) + log_tau
        // d/d(log_tau) = lambda * (-d sigma/d log_tau) - 1/sigma * (-d sigma/d log_tau) + 1
        // d sigma/d log_tau = d sigma/d tau * tau = -0.5 * tau^{-3/2} * tau = -0.5/sqrt(tau) = -0.5*sigma
        // So: d/d(log_tau) = lambda * 0.5 * sigma + (1/sigma) * 0.5 * sigma + 1
        //                  = 0.5 * lambda * sigma + 0.5 + 1
        grad[layout.log_tau_st_idx] = 0.5 * lambda * sigma_st + 0.5 + 1.0;
    }
    // tau_st2 removed — single tau for all ST types

    // ST interaction ranges: uniform inside the bounds, so each carries only
    // its log Jacobian. Outside them the density is -inf and the state is
    // rejected on the value, not steered on the gradient.
    if (layout.is_st_gp) {
        grad[layout.log_phi_st_space_idx] = 1.0;
        grad[layout.log_phi_st_time_idx] = 1.0;
    }

    // ST interaction prior on rho_st (Beta on u = (rho + 1) / 2)
    if (layout.logit_rho_st_idx >= 0) {
        grad[layout.logit_rho_st_idx] = ratiod_ar1::log_prior_rho_grad(
            rho_st, st.rho_prior_a, st.rho_prior_b);
    }

    // HSGP-ST hyperparameter priors: PC on sigma, LogNormal(0, 1) on the
    // lengthscale. The field's own contribution to both arrives below.
    if (data.st_is_hsgp) {
        grad[layout.log_sigma2_st_hsgp_idx] =
            -0.5 * 4.6 * std::sqrt(sigma2_st_hsgp) + 0.5;
        grad[layout.log_lengthscale_st_hsgp_idx] =
            -params[layout.log_lengthscale_st_hsgp_idx];
    }

    // =========================================================================
    // =========================================================================
    // Vectorized likelihood loop
    // =========================================================================
    std::vector<double> grad_spatial_lik;
    if (layout.has_spatial) grad_spatial_lik.assign(data.n_spatial_units, 0.0);
    std::vector<double> grad_theta_lik;
    if (layout.is_bym2) grad_theta_lik.assign(data.n_spatial_units, 0.0);
    std::vector<double> grad_temporal_lik(T_temporal, 0.0);
    std::vector<double> grad_delta_lik(ST, 0.0);

    const int N = data.N;
    const bool is_binomial = (data.model_type == ModelType::BINOMIAL ||
                              data.model_type == ModelType::BETA_BINOMIAL);

    // --- Pass 1: Vectorized eta computation (Eigen matvec + scalar expansion) ---
    using RowMajorMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using VectorXd = Eigen::VectorXd;

    vec_grad_ws().init(N);

    Eigen::Map<const RowMajorMatrix> X_num(data.X_num_flat.data(), N, data.p_num);
    Eigen::Map<const VectorXd> b_num(beta_num, data.p_num);
    Eigen::Map<VectorXd> eta_n(vec_grad_ws().eta_num.data(), N);
    eta_n.noalias() = X_num * b_num;

    if (!is_binomial) {
        Eigen::Map<const RowMajorMatrix> X_denom(data.X_denom_flat.data(), N, data.p_denom);
        Eigen::Map<const VectorXd> b_denom(beta_denom, data.p_denom);
        Eigen::Map<VectorXd> eta_d(vec_grad_ws().eta_denom.data(), N);
        eta_d.noalias() = X_denom * b_denom;
    }

    // Add RE, spatial, temporal, and ST effects to eta (scalar expansion)
    for (int i = 0; i < N; i++) {
        // RE
        if (layout.has_re && data.re_group[i] > 0) {
            double re_eff = re_value_for_eta(re, data.re_group[i] - 1, sigma_re, data.re_parameterization);
            vec_grad_ws().eta_num[i] += re_eff;
            if (!is_binomial) vec_grad_ws().eta_denom[i] += re_eff;
        }
        // Spatial
        if (layout.has_spatial && data.spatial_group[i] > 0) {
            int s_unit = data.spatial_group[i] - 1;
            double spatial_eff;
            if (layout.is_bym2) {
                spatial_eff = sigma_s_bym2 * data.bym2_scale_factor * spatial_phi[s_unit] + sigma_u_bym2 * theta_bym2[s_unit];
            } else {
                spatial_eff = spatial_phi[s_unit];
            }
            vec_grad_ws().eta_num[i] += spatial_eff;
            if (!is_binomial) vec_grad_ws().eta_denom[i] += spatial_eff;
        }
        // Temporal
        if (layout.has_temporal && !data.temporal_time_idx.empty() &&
            i < (int)data.temporal_time_idx.size() && data.temporal_time_idx[i] > 0) {
            int t = data.temporal_time_idx[i] - 1;
            int g = data.temporal_group_idx[i] - 1;
            int t_base = g * data.n_times + t;
            if (t_base >= 0 && t_base < T_temporal) {
                vec_grad_ws().eta_num[i] += phi_temporal[t_base];
                if (!is_binomial && data.temporal_shared) vec_grad_ws().eta_denom[i] += phi_temporal[t_base];
            }
        }
        // Spatiotemporal interaction
        if (st.st_flat[i] > 0) {
            double st_effect = delta[st.st_flat[i] - 1];
            vec_grad_ws().eta_num[i] += st_effect;
            if (!is_binomial && st.shared) vec_grad_ws().eta_denom[i] += st_effect;
        }
    }

    // --- Pass 2+3: Vectorized residuals + beta grads (template-dispatched) ---
    {
        double grad_phi_num_lik = 0.0, grad_phi_denom_lik = 0.0;
        vectorized::dispatch_residuals_and_beta_grads(
            data, layout,
            vec_grad_ws().eta_num.data(), vec_grad_ws().eta_denom.data(),
            vec_grad_ws().resid_num.data(), vec_grad_ws().resid_denom.data(),
            grad.data(), grad_phi_num_lik, grad_phi_denom_lik,
            obs_log_lik, fuse_lp, phi_num, phi_denom, vec_grad_ws());
    }

    // Scatter residuals to RE, spatial, temporal, and ST gradient buffers
    for (int i = 0; i < N; i++) {
        double dLL_num = vec_grad_ws().resid_num[i];
        double dLL_denom = vec_grad_ws().resid_denom[i];
        double dLL_shared = dLL_num + dLL_denom;

        // RE
        if (layout.has_re && data.re_group[i] > 0)
            grad[layout.re_start + data.re_group[i] - 1] += dLL_shared;

        // Spatial
        if (layout.has_spatial && data.spatial_group[i] > 0) {
            int s_unit = data.spatial_group[i] - 1;
            if (layout.is_bym2) { grad_spatial_lik[s_unit] += dLL_shared; grad_theta_lik[s_unit] += dLL_shared; }
            else { grad_spatial_lik[s_unit] += dLL_shared; }
        }

        // Temporal
        if (layout.has_temporal && !data.temporal_time_idx.empty() &&
            i < (int)data.temporal_time_idx.size() && data.temporal_time_idx[i] > 0) {
            int t = data.temporal_time_idx[i] - 1;
            int g = data.temporal_group_idx[i] - 1;
            int t_base = g * data.n_times + t;
            if (t_base >= 0 && t_base < T_temporal)
                grad_temporal_lik[t_base] += data.temporal_shared ? dLL_shared : dLL_num;
        }

        // Spatiotemporal interaction
        if (st.st_flat[i] > 0) {
            int st_idx = st.st_flat[i] - 1;
            grad_delta_lik[st_idx] += st.shared ? dLL_shared : dLL_num;
        }
    }

    // Spatial GMRF prior gradients (ICAR/BYM2)
    if (layout.has_spatial) {
        spatial_lik_grad_to_param_scale(data, layout, sigma_s_bym2, sigma_u_bym2,
                                        theta_bym2, grad_theta_lik.data(),
                                        grad_spatial_lik.data(), grad.data());
        spatial_gmrf_prior_grad(data, layout, spatial_phi, spatial_phi,
                                centering.raw_sum(),
                                centering.active(), tau_spatial,
                                sigma_s_bym2, sigma_u_bym2, rho_bym2, 0.0, theta_bym2,
                                grad_spatial_lik.data(), grad.data());
    }

    // Temporal GMRF prior gradients
    if (layout.has_temporal && T_temporal > 0) {
        temporal_gmrf_prior_grad(data, layout, tau_temporal, rho_ar1,
                                 tview, T_temporal, grad_temporal_lik.data(), grad.data());
    }

    // =========================================================================
    // Spatiotemporal interaction prior gradients
    // =========================================================================
    // One implementation, shared with compute_gradient_composite
    // (st_prior_grad.h): the interaction's field prior, its sum-to-zero
    // margins, and what both contribute to tau, rho and the HSGP-ST spectral
    // hyperparameters. The log-prior comes back with the gradient so
    // compute_log_post below does not evaluate the Kronecker quadratic form a
    // second time.
    const ratiod_spatiotemporal::StPriorGrad st_pg =
        ratiod_spatiotemporal::st_interaction_prior_grad(
            st, data.st_is_hsgp, data.st_hsgp_data,
            z_or_delta, delta, grad_delta_lik.data(),
            tau_st, rho_st, sigma2_st_hsgp, lengthscale_st_hsgp,
            phi_st_space, phi_st_time,
            st_use_nc, &grad[layout.st_delta_start]);

    double st_lp_accum = st_pg.log_prior;
    grad[layout.log_tau_st_idx] += st_pg.log_tau;
    if (layout.logit_rho_st_idx >= 0) {
        grad[layout.logit_rho_st_idx] += st_pg.logit_rho;
    }
    if (data.st_is_hsgp) {
        grad[layout.log_sigma2_st_hsgp_idx] += st_pg.log_sigma2_hsgp;
        grad[layout.log_lengthscale_st_hsgp_idx] += st_pg.log_lengthscale_hsgp;
    }
    if (layout.is_st_gp) {
        grad[layout.log_phi_st_space_idx] += st_pg.log_phi_space;
        grad[layout.log_phi_st_time_idx] += st_pg.log_phi_time;
    }

    // Non-centered RE chain rule transformation
    re_gradient_nc_transform(data, layout, params.data(), grad.data(), sigma_re);

    // params_in, not params: compute_log_post centres its own copy and recovers
    // phi_raw_sum from the raw field, which the centred copy has zeroed.
    if (fuse_lp) *log_post_out = compute_log_post(params_in, data, layout, /*skip_obs_loop=*/true, &st_lp_accum) + obs_log_lik;
}

// =====================================================================
// Composite hand-coded gradient: handles ANY combination of features.
// This is the catch-all H-mode function for exotic multi-feature combos
// that no specialized gradient function covers (e.g., HSGP+TVC, SVC+RW1,
// latent+spatial, etc.). Slower than specialized functions but much faster
// than A_r/N fallback.
//
// Architecture: single observation loop with conditional feature blocks.
// Each feature contributes additively to eta; gradient scattering is
// independent per feature. Structural/prior gradients computed after the
// observation loop.
// =====================================================================

void compute_gradient_composite(
    const std::vector<double>& params_in,
    const ModelData& data,
    const ParamLayout& layout,
    std::vector<double>& grad,
    double* log_post_out = nullptr
) {
    // Centre the spatial block once, matching compute_log_post; see
    // spatial_field_constraint.h. The returned gradient is with respect to the
    // raw parameters, which is what spatial_gmrf_prior_grad projects for.
    // Proper CAR is full rank (rho < 1) and already identified, so it is not
    // centered like ICAR/BYM2's rank-deficient Q.
    const bool center_spatial = spatial_block_is_centred(data, layout);
    ratiod_constraints::CenteredSpatialParams centering(
        params_in, center_spatial, layout.spatial_start, data.n_spatial_units);
    const std::vector<double>& params = centering.params();

    const bool fuse_lp = (log_post_out != nullptr) && !layout.has_zi;
    // params_in, not params: compute_log_post centres its own copy and recovers
    // phi_raw_sum from the raw field, which the centred copy has zeroed.
    if (log_post_out && layout.has_zi) *log_post_out = compute_log_post(params_in, data, layout);
    double obs_log_lik = 0.0;
    int n_params = params.size();
    grad.assign(n_params, 0.0);

    const int N = data.N;
    const bool is_binomial = (data.model_type == ModelType::BINOMIAL ||
                              data.model_type == ModelType::BETA_BINOMIAL);

    // =========================================================================
    // Phase 1: Extract all parameters
    // =========================================================================
    auto cp = extract_common_params(params, layout);
    const double* beta_num = cp.beta_num;
    const double* beta_denom = cp.beta_denom;
    double sigma_re = cp.sigma_re;
    const double* re = cp.re;
    double phi_num = cp.phi_num;
    double phi_denom = cp.phi_denom;

    // --- Spatial (ICAR/BYM2/pCAR) ---
    double tau_spatial = 0.0;
    const double* spatial_phi = nullptr;
    const double* spatial_phi_prior = nullptr;
    double sigma_s_bym2 = 0.0, sigma_u_bym2 = 0.0, rho_bym2 = 0.5, rho_car = 0.0;
    const double* theta_bym2 = nullptr;
    // A collapsed field has no latent block for the layout to allocate, so
    // spatial_start is -1 and this arm would index in front of the parameter
    // vector. resolve_gradient_fn does not send a collapsed model here; the
    // condition says so rather than relying on it.
    const bool has_icar_bym2 = layout.has_spatial && !layout.is_hsgp && !layout.is_gp &&
                                !layout.is_multiscale_gp && !layout.has_svc &&
                                !data.icar_collapsed && !data.bym2_collapsed;
    if (has_icar_bym2) {
        if (!layout.is_bym2) tau_spatial = std::exp(params[layout.log_tau_spatial_idx]);
        spatial_phi = &params[layout.spatial_start];
        spatial_phi_prior = spatial_prior_reads_raw(data, layout)
                                ? &params_in[layout.spatial_start]
                                : spatial_phi;
        if (layout.is_car_proper) {
            const double u = 1.0 / (1.0 + std::exp(-params[layout.logit_rho_car_idx]));
            rho_car = data.car_rho_lower + (data.car_rho_upper - data.car_rho_lower) * u;
        }
        if (layout.is_bym2) {
            double sigma_total = std::exp(params[layout.log_sigma_bym2_idx]);
            double logit_rho = params[layout.logit_rho_bym2_idx];
            rho_bym2 = 1.0 / (1.0 + std::exp(-logit_rho));
            sigma_s_bym2 = sigma_total * std::sqrt(rho_bym2);
            sigma_u_bym2 = sigma_total * std::sqrt(1.0 - rho_bym2);
            theta_bym2 = &params[layout.theta_bym2_start];
        }
    }

    // --- GP spatial (NNGP) ---
    // The field, its two hyperparameters and the non-centred forward transform,
    // read off the same ratiod_gp entry points compute_gradient_gp_handcoded
    // uses. Without this arm the composite is not a catch-all for a GP main
    // effect: the block reaches eta in compute_log_post and comes back at zero
    // in the gradient, which is what a GP paired with an interaction, a latent
    // factor or a TVC falls through to.
    RATIOD_TLS_WORKSPACE(ratiod_gp::NNGPNCWorkspace, gp_nc_ws);
    const bool has_nngp_gp = layout.is_gp && data.has_gp && !data.gp_collapsed;
    const int N_gp_comp = has_nngp_gp ? data.gp_data.n_obs : 0;
    double gp_sigma2_comp = 0.0, gp_phi_comp = 0.0;
    bool gp_use_nc = false;
    std::vector<double> gp_w_comp;
    if (has_nngp_gp) {
        gp_sigma2_comp = std::exp(params[layout.log_sigma2_gp_idx]);
        gp_phi_comp = std::exp(params[layout.log_phi_gp_idx]);
        // Outside its uniform bounds the density is -Inf, so the point is
        // rejected whatever the gradient says; return the zero gradient the
        // dedicated GP path returns there rather than a number read off a
        // covariance built at a range the prior excludes.
        if (gp_phi_comp < data.gp_phi_prior_lower || gp_phi_comp > data.gp_phi_prior_upper) {
            if (log_post_out && !layout.has_zi)
                *log_post_out = compute_log_post(params_in, data, layout);
            return;
        }
        gp_use_nc = (data.gp_parameterization == 1);
        gp_w_comp.resize(N_gp_comp);
        if (gp_use_nc) {
            ratiod_gp::nngp_nc_forward(&params[layout.gp_w_start], gp_sigma2_comp,
                                       gp_phi_comp, data.gp_data, gp_nc_ws);
            std::memcpy(gp_w_comp.data(), gp_nc_ws.w.data(), N_gp_comp * sizeof(double));
        } else {
            for (int i = 0; i < N_gp_comp; i++)
                gp_w_comp[i] = params[layout.gp_w_start + i];
        }
    }

    // --- Multi-scale GP spatial (two NNGP scales) ---
    const bool has_msgp_nngp = layout.is_multiscale_gp && data.has_multiscale_gp &&
                               !data.msgp_is_hsgp;
    const int N_msgp = has_msgp_nngp ? data.multiscale_gp_data.n_obs : 0;
    double ms_sigma2_local = 0.0, ms_phi_local = 0.0;
    double ms_sigma2_regional = 0.0, ms_phi_regional = 0.0;
    MultiscaleGPView msgp;
    std::vector<double> grad_ms_w_lik;
    if (has_msgp_nngp) {
        ms_sigma2_local = std::exp(params[layout.log_sigma2_gp_local_idx]);
        ms_phi_local = std::exp(params[layout.log_phi_gp_local_idx]);
        ms_sigma2_regional = std::exp(params[layout.log_sigma2_gp_regional_idx]);
        ms_phi_regional = std::exp(params[layout.log_phi_gp_regional_idx]);
        const auto& msd = data.multiscale_gp_data;
        if (ms_phi_local < msd.range_local_lower || ms_phi_local > msd.range_local_upper ||
            ms_phi_regional < msd.range_regional_lower ||
            ms_phi_regional > msd.range_regional_upper) {
            if (log_post_out && !layout.has_zi)
                *log_post_out = compute_log_post(params_in, data, layout);
            return;
        }
        msgp.read(params, data, layout);
        grad_ms_w_lik.assign(N_msgp, 0.0);
    }

    // --- HSGP spatial ---
    RATIOD_TLS_WORKSPACE(ratiod_hsgp::HSGPWorkspace, hsgp_ws);
    double hsgp_sigma2 = 0.0, hsgp_lengthscale = 0.0;
    const double* hsgp_beta_ptr = nullptr;
    if (layout.is_hsgp && data.has_hsgp) {
        hsgp_sigma2 = std::exp(params[layout.log_sigma2_hsgp_idx]);
        hsgp_lengthscale = std::exp(params[layout.log_lengthscale_hsgp_idx]);
        hsgp_beta_ptr = &params[layout.hsgp_beta_start];
        hsgp_ws.init(data.hsgp_data.n_obs, data.hsgp_data.m_total);
        ratiod_hsgp::hsgp_evaluate_ws(hsgp_beta_ptr, hsgp_sigma2, hsgp_lengthscale,
                                       data.hsgp_data, hsgp_ws);
    }

    // --- Temporal (RW1/RW2/AR1 GMRF) ---
    double tau_temporal = 0.0;
    int T_temporal = 0;
    const double* phi_temporal = nullptr;
    TemporalView tview;
    double rho_ar1 = 0.0;
    const bool has_gmrf_temporal = layout.has_temporal && !layout.is_temporal_gp &&
                                   !layout.has_multiscale_temporal && !layout.has_tvc;
    if (has_gmrf_temporal) {
        tau_temporal = std::exp(params[layout.log_tau_temporal_idx]);
        T_temporal = layout.temporal_end - layout.temporal_start;
        tview.read(params, data, layout);
        phi_temporal = tview.phi;
        rho_ar1 = read_temporal_rho_ar1(params, data, layout);
    }

    // --- Temporal GP ---
    const bool has_temporal_gp = layout.is_temporal_gp && layout.has_temporal;
    int T_gp = 0;
    const double* z_temporal_gp = nullptr;
    RATIOD_TLS_WORKSPACE(ratiod_temporal_gp::TemporalGPNCWorkspace, nc_ws_composite);
    RATIOD_TLS_WORKSPACE(std::vector<double>, temporal_gp_f);
    double sigma2_tgp_comp = 0.0, phi_tgp_comp = 0.0;
    double phi_lower_tgp = 0.0, phi_upper_tgp = 0.0;
    bool use_nc_tgp = false;
    int n_groups_gp = 0;
    if (has_temporal_gp) {
        T_gp = data.n_times;
        n_groups_gp = data.n_temporal_groups;
        z_temporal_gp = &params[layout.temporal_start];
        int total_gp = n_groups_gp * T_gp;
        temporal_gp_f.resize(total_gp);

        sigma2_tgp_comp = std::exp(params[layout.log_sigma2_temporal_gp_idx]);
        double phi_gp_raw = params[layout.logit_phi_temporal_gp_idx];
        phi_lower_tgp = data.temporal_gp_phi_prior_lower;
        phi_upper_tgp = data.temporal_gp_phi_prior_upper;
        double phi_range = phi_upper_tgp - phi_lower_tgp;
        phi_tgp_comp = phi_lower_tgp + phi_range / (1.0 + std::exp(-phi_gp_raw));

        use_nc_tgp = (data.temporal_gp_parameterization == 1);
        if (use_nc_tgp) {
            nc_ws_composite.init(T_gp, n_groups_gp);
            ratiod_temporal_gp::temporal_gp_nc_forward(
                z_temporal_gp, T_gp, n_groups_gp, sigma2_tgp_comp, phi_tgp_comp,
                data.temporal_gp_data.time_values, nc_ws_composite);
            for (int k = 0; k < total_gp; k++) temporal_gp_f[k] = nc_ws_composite.f[k];
        } else {
            for (int k = 0; k < total_gp; k++) temporal_gp_f[k] = z_temporal_gp[k];
        }
    }

    // --- TVC ---
    RATIOD_TLS_WORKSPACE(std::vector<double>, tvc_eta_precomp);
    int n_tvc = 0, n_tvc_times = 0, n_tvc_groups = 1, n_w = 0;
    RATIOD_TLS_WORKSPACE(std::vector<double>, tvc_tau_buf);
    RATIOD_TLS_WORKSPACE(std::vector<double>, tvc_rho_buf);
    RATIOD_TLS_WORKSPACE(std::vector<double>, tvc_w_flat_buf);
    if (layout.has_tvc && data.has_tvc) {
        n_tvc = data.tvc_data.n_tvc;
        n_tvc_times = data.tvc_data.n_times;
        n_tvc_groups = data.tvc_data.n_groups;
        n_w = n_tvc_groups * n_tvc * n_tvc_times;

        tvc_tau_buf.resize(n_tvc);
        tvc_rho_buf.resize(n_tvc);
        tvc_w_flat_buf.resize(n_w);

        for (int j = 0; j < n_tvc; j++) {
            tvc_tau_buf[j] = std::exp(params[layout.log_tau_tvc_start + j]);
            if (data.tvc_data.structure == ratiod_temporal::TemporalType::AR1) {
                double logit_rho = params[layout.logit_rho_tvc_start + j];
                double u = 1.0 / (1.0 + std::exp(-logit_rho));
                tvc_rho_buf[j] = 2.0 * u - 1.0;
            } else {
                tvc_rho_buf[j] = 0.0;
            }
        }
        for (int k = 0; k < n_w; k++) tvc_w_flat_buf[k] = params[layout.tvc_w_start + k];

        // Precompute TVC eta contribution
        tvc_eta_precomp.assign(N, 0.0);
        for (int i = 0; i < N; i++) {
            int t = data.tvc_data.time_index[i] - 1;
            int g = data.tvc_data.group_index[i] - 1;
            for (int j = 0; j < n_tvc; j++) {
                int w_idx = (g * n_tvc + j) * n_tvc_times + t;
                tvc_eta_precomp[i] += data.tvc_data.X_tvc[i * n_tvc + j] * tvc_w_flat_buf[w_idx];
            }
        }
    }

    // --- SVC, on either approximation ---
    // The NNGP arm carries the field itself; the HSGP arm carries its basis
    // coefficients. Both reach eta through svc_eta_precomp, so the observation
    // loop reads one quantity either way.
    RATIOD_TLS_WORKSPACE(ratiod_hsgp::HSGPWorkspace, svc_hsgp_ws);
    int n_svc = 0, svc_m_total = 0;
    RATIOD_TLS_WORKSPACE(std::vector<double>, svc_eta_precomp);
    RATIOD_TLS_WORKSPACE(std::vector<double>, svc_f_all);
    const bool has_svc_hsgp = layout.has_svc && data.has_svc && data.svc_is_hsgp;
    const bool has_svc_nngp = layout.has_svc && data.has_svc && !data.svc_is_hsgp;
    const int N_svc = has_svc_nngp ? data.svc_data.n_obs : 0;
    std::vector<double> svc_w_nngp;
    std::vector<double> svc_term_mean;
    std::vector<double> svc_sigma2_c, svc_phi_c;
    if (has_svc_nngp) {
        n_svc = data.svc_data.n_svc;
        svc_w_nngp.resize(static_cast<size_t>(N_svc) * n_svc);
        svc_sigma2_c.resize(n_svc);
        svc_phi_c.resize(n_svc);
        for (int j = 0; j < n_svc; j++) {
            svc_sigma2_c[j] = std::exp(params[layout.log_sigma2_svc_start + j]);
            svc_phi_c[j] = std::exp(params[layout.log_phi_svc_start + j]);
            if (svc_phi_c[j] < data.svc_phi_prior_lower ||
                svc_phi_c[j] > data.svc_phi_prior_upper) {
                if (log_post_out && !layout.has_zi)
                    *log_post_out = compute_log_post(params_in, data, layout);
                return;
            }
        }
        ratiod_svc::svc_field<ratiod_svc::SVC_GRAD_SLOT>(&params[layout.svc_w_start], svc_sigma2_c.data(),
                              svc_phi_c.data(), data.svc_data, data.svc_gp_view,
                              data.svc_noncentered, svc_w_nngp.data());
        // eta reads the CENTRED field, the NNGP prior below reads w itself.
        svc_term_mean.assign(n_svc, 0.0);
        ratiod_svc::svc_term_means(svc_w_nngp.data(), data.svc_data,
                                   svc_term_mean.data());
        svc_eta_precomp.assign(N, 0.0);
        for (int i = 0; i < N; i++) {
            double eff = 0.0;
            for (int j = 0; j < n_svc; j++)
                eff += data.svc_data.X_svc[i * n_svc + j] *
                       (svc_w_nngp[j * N_svc + i] - svc_term_mean[j]);
            svc_eta_precomp[i] = eff;
        }
    }
    if (has_svc_hsgp) {
        n_svc = data.svc_data.n_svc;
        svc_m_total = data.svc_hsgp_data.m_total;
        svc_hsgp_ws.init(data.svc_hsgp_data.n_obs, svc_m_total);
        svc_eta_precomp.assign(N, 0.0);
        svc_f_all.resize(n_svc * N);

        for (int j = 0; j < n_svc; j++) {
            double sigma2_j = std::exp(params[layout.log_sigma2_svc_start + j]);
            double lengthscale_j = std::exp(params[layout.log_phi_svc_start + j]);
            const double* beta_j = &params[layout.svc_w_start + j * svc_m_total];

            ratiod_hsgp::hsgp_evaluate_ws(beta_j, sigma2_j, lengthscale_j,
                                           data.svc_hsgp_data, svc_hsgp_ws);

            for (int i = 0; i < N; i++) {
                double f_ji = svc_hsgp_ws.hsgp_f[i];
                svc_f_all[j * N + i] = f_ji;
                double x_ij = data.svc_data.X_svc[i * n_svc + j];
                svc_eta_precomp[i] += x_ij * f_ji;
            }
        }
    }

    // --- Latent factors ---
    int K_latent = 0;
    RATIOD_TLS_WORKSPACE(std::vector<double>, latent_eta_precomp);
    RATIOD_TLS_WORKSPACE(std::vector<double>, factors_constrained);
    RATIOD_TLS_WORKSPACE(std::vector<double>, sigma_latent_vec);
    if (layout.has_latent && data.latent_n_factors > 0) {
        K_latent = data.latent_n_factors;
        sigma_latent_vec.resize(K_latent);
        for (int k = 0; k < K_latent; k++)
            sigma_latent_vec[k] = std::exp(params[layout.log_sigma_latent_start + k]);

        int n_factor_params = N * K_latent;
        factors_constrained.resize(n_factor_params);
        for (int j = 0; j < n_factor_params; j++)
            factors_constrained[j] = params[layout.latent_factor_start + j];

        // Apply sum-to-zero constraint
        if (data.latent_constraint == 0) {
            for (int k = 0; k < K_latent; k++) {
                double sum = 0.0;
                for (int i = 0; i < N; i++) sum += factors_constrained[i * K_latent + k];
                double mean = sum / N;
                for (int i = 0; i < N; i++) factors_constrained[i * K_latent + k] -= mean;
            }
        }

        latent_eta_precomp.assign(N, 0.0);
        for (int i = 0; i < N; i++)
            for (int k = 0; k < K_latent; k++)
                latent_eta_precomp[i] += factors_constrained[i * K_latent + k] * sigma_latent_vec[k];
    }

    // --- Spatiotemporal ---
    const bool has_st = layout.has_spatiotemporal &&
                        data.spatiotemporal_data.type != STType::NONE &&
                        layout.st_delta_start >= 0 && layout.log_tau_st_idx >= 0;
    double tau_st = 0.0;
    double rho_st = 0.0;
    double sigma2_st_hsgp = 1.0, lengthscale_st_hsgp = 1.0;
    double phi_st_space = 1.0, phi_st_time = 1.0;
    const double* st_delta = nullptr;
    RATIOD_TLS_WORKSPACE(std::vector<double>, st_delta_buf);
    bool st_use_nc = false;
    double inv_scale_st = 1.0;
    const double* z_or_delta_st = nullptr;
    int ST_n = 0;
    if (has_st) {
        const auto& st = data.spatiotemporal_data;
        ST_n = st.n_params;
        tau_st = std::exp(params[layout.log_tau_st_idx]);
        // The layout carries a time-margin correlation only where the margin
        // is AR1; elsewhere 0 leaves R(rho) at the identity.
        if (layout.logit_rho_st_idx >= 0)
            rho_st = ratiod_ar1::rho_from_logit(params[layout.logit_rho_st_idx]);
        if (data.st_is_hsgp) {
            sigma2_st_hsgp = std::exp(params[layout.log_sigma2_st_hsgp_idx]);
            lengthscale_st_hsgp = std::exp(params[layout.log_lengthscale_st_hsgp_idx]);
        }
        // The GP types' two ranges. The prior is uniform inside its bounds, so it
        // leaves only the log Jacobian on each; the field's own contribution to
        // both arrives with the interaction's prior below.
        if (layout.is_st_gp) {
            phi_st_space = std::exp(params[layout.log_phi_st_space_idx]);
            phi_st_time = std::exp(params[layout.log_phi_st_time_idx]);
        }
        st_use_nc = (data.st_parameterization == 1 && st.type == STType::TYPE_IV);
        z_or_delta_st = &params[layout.st_delta_start];
        if (st_use_nc) {
            inv_scale_st = 1.0 / std::sqrt(tau_st);
            st_delta_buf.resize(ST_n);
            for (int k = 0; k < ST_n; k++) st_delta_buf[k] = z_or_delta_st[k] * inv_scale_st;
            st_delta = st_delta_buf.data();
        } else {
            st_delta = z_or_delta_st;
        }
    }

    // --- Random effects, one block per term ---
    // The same prior phase, eta contribution and chain rule
    // compute_gradient_analytical uses, from hmc_re_blocks_grad.h: crossed and
    // nested terms, random slopes, correlated or not, centred or not. The
    // legacy single-term block is the one-term case of it.
    ReBlockGrad re_ws;

    // --- Multiscale temporal ---
    const bool has_ms_temporal = layout.has_multiscale_temporal;
    int n_trend = 0, n_seasonal = 0, n_short = 0;
    const double* trend = nullptr;
    const double* seasonal = nullptr;
    const double* short_term = nullptr;
    double sigma2_trend = 1.0, sigma2_seasonal = 1.0, sigma2_short = 1.0;
    double rho_short = 0.5;
    RATIOD_TLS_WORKSPACE(std::vector<double>, ms_effect_by_time_c);
    ratiod_temporal::MSArmEffects ms_arms_c;
    RATIOD_TLS_WORKSPACE(std::vector<double>, grad_trend_lik_c);
    RATIOD_TLS_WORKSPACE(std::vector<double>, grad_seasonal_lik_c);
    RATIOD_TLS_WORKSPACE(std::vector<double>, grad_short_lik_c);
    RATIOD_TLS_WORKSPACE(std::vector<int>, obs_t_idx_ms_c);
    if (has_ms_temporal) {
        const auto& mst = data.multiscale_temporal_data;
        n_trend = layout.trend_end - layout.trend_start;
        n_seasonal = layout.seasonal_end - layout.seasonal_start;
        n_short = layout.short_term_end - layout.short_term_start;
        if (n_trend > 0) { trend = &params[layout.trend_start]; sigma2_trend = std::exp(params[layout.log_sigma2_trend_idx]); }
        if (n_seasonal > 0) { seasonal = &params[layout.seasonal_start]; sigma2_seasonal = std::exp(params[layout.log_sigma2_seasonal_idx]); }
        if (n_short > 0) { short_term = &params[layout.short_term_start]; sigma2_short = std::exp(params[layout.log_sigma2_short_idx]); }
        if (mst.short_term_type == TemporalType::AR1 && layout.logit_rho_short_idx >= 0) {
            double u = 1.0 / (1.0 + std::exp(-params[layout.logit_rho_short_idx]));
            rho_short = 2.0 * u - 1.0;
        }
        // Pre-assemble temporal effect per time point. eta reads the arms'
        // effects, which are the blocks unless the intrinsic arms are sampled
        // in their non-centred coordinate.
        ms_arms_c = ratiod_temporal::ms_read_arms(
            trend, n_trend, seasonal, n_seasonal, sigma2_trend, sigma2_seasonal, mst);
        ms_effect_by_time_c.assign(mst.n_times, 0.0);
        ratiod_temporal::compute_ms_effect_by_time(
            ms_arms_c.trend, n_trend, ms_arms_c.seasonal, n_seasonal,
            short_term, n_short,
            mst.seasonal_period, mst.n_times, ms_effect_by_time_c.data());
        grad_trend_lik_c.assign(n_trend, 0.0);
        grad_seasonal_lik_c.assign(n_seasonal, 0.0);
        grad_short_lik_c.assign(n_short, 0.0);
        obs_t_idx_ms_c.assign(N, -1);
    }

    // =========================================================================
    // Phase 2: Prior gradients (independent of data, computed first)
    // =========================================================================
    beta_gradient_prior(data, layout, beta_num, beta_denom, grad.data());
    if (!re_blocks_prior_grad(params, data, layout, grad, re_ws)) return;
    phi_gradient_prior(data, layout, phi_num, phi_denom, grad.data());

    // ICAR/BYM2/pCAR spatial priors
    if (has_icar_bym2 && !layout.is_bym2) {
        grad[layout.log_tau_spatial_idx] = (data.tau_spatial_shape - 1.0) - data.tau_spatial_rate * tau_spatial + 1.0;
    }
    if (has_icar_bym2 && layout.is_bym2) {
        double sigma_total = sigma_s_bym2 / std::sqrt(rho_bym2);
        double ratio = sigma_total / data.sigma_re_scale;
        grad[layout.log_sigma_bym2_idx] = -2.0 * ratio * ratio / (1.0 + ratio * ratio) + 1.0;
        grad[layout.logit_rho_bym2_idx] = 1.0 - 2.0 * rho_bym2;
    }
    if (has_icar_bym2 && layout.is_car_proper) {
        // Beta(a, b) on rho with the logit-transform Jacobian folded in; see
        // the matching derivation in compute_log_post.
        const double u = (rho_car - data.car_rho_lower) / (data.car_rho_upper - data.car_rho_lower);
        grad[layout.logit_rho_car_idx] = data.car_rho_prior_a * (1.0 - u) - data.car_rho_prior_b * u;
    }

    // GP (NNGP) priors: PC on sigma2, uniform on phi (log-transform Jacobian
    // only). Under the non-centred parameterization the field's own prior is
    // N(0, I) on z and is seeded by nngp_nc_backward, so it is not added twice.
    if (has_nngp_gp) {
        grad[layout.log_sigma2_gp_idx] += gp_pc_prior_grad_log_sigma2(
            gp_sigma2_comp, data.gp_sigma2_prior_U, data.gp_sigma2_prior_alpha);
        grad[layout.log_phi_gp_idx] += 1.0;
        if (!gp_use_nc) {
            ratiod_gp::NNGPGradients nngp_grads;
            ratiod_gp::gp_nngp_gradients(gp_w_comp, gp_sigma2_comp, gp_phi_comp,
                                         data.gp_data, nngp_grads);
            for (int i = 0; i < N_gp_comp; i++)
                grad[layout.gp_w_start + i] += nngp_grads.grad_w[i];
            grad[layout.log_sigma2_gp_idx] += nngp_grads.grad_log_sigma2;
            grad[layout.log_phi_gp_idx] += nngp_grads.grad_log_phi;
        }
    }

    // Multi-scale GP priors: PC on each scale's variance, uniform on each
    // range (log-transform Jacobian only). Each field's own prior reaches the
    // sampled coordinate through the view's adjoint, below.
    if (has_msgp_nngp) {
        grad[layout.log_sigma2_gp_local_idx] += gp_pc_prior_grad_log_sigma2(
            ms_sigma2_local, data.ms_sigma2_local_prior_U, data.ms_sigma2_local_prior_alpha);
        grad[layout.log_sigma2_gp_regional_idx] += gp_pc_prior_grad_log_sigma2(
            ms_sigma2_regional, data.ms_sigma2_regional_prior_U,
            data.ms_sigma2_regional_prior_alpha);
        grad[layout.log_phi_gp_local_idx] += 1.0;
        grad[layout.log_phi_gp_regional_idx] += 1.0;
    }

    // SVC (NNGP) priors: half-Cauchy on each sigma, uniform on each range, and
    // the NNGP field prior. Centred, that prior is placed on the UNCENTRED w and
    // so is not projected; non-centred, it is N(0, I) on z and is seeded by
    // svc_field_accumulate below. The likelihood's gradient is grad_svc_w_lik.
    if (has_svc_nngp) {
        for (int j = 0; j < n_svc; j++) {
            const double ratio = std::sqrt(svc_sigma2_c[j]) / data.svc_sigma2_prior_scale;
            const double ratio_sq = ratio * ratio;
            grad[layout.log_sigma2_svc_start + j] += -ratio_sq / (1.0 + ratio_sq) + 1.0;
            grad[layout.log_phi_svc_start + j] += 1.0;
        }
        if (!data.svc_noncentered) {
            RATIOD_TLS_WORKSPACE(ratiod_svc::SVCGradWorkspace, svc_scratch);
            svc_scratch.resize(n_svc, N_svc);
            ratiod_svc::svc_nngp_prior_grads(
                svc_w_nngp.data(), svc_sigma2_c.data(), svc_phi_c.data(),
                data.svc_data, layout.svc_w_start, layout.log_sigma2_svc_start,
                layout.log_phi_svc_start, svc_scratch, grad.data());
        }
    }

    // HSGP priors (must match hardcoded rate=4.6 in compute_log_post)
    if (layout.is_hsgp && data.has_hsgp) {
        double sigma = std::sqrt(hsgp_sigma2);
        double rate = 4.6;  // Matches compute_log_post line 1457
        grad[layout.log_sigma2_hsgp_idx] = -0.5 * rate * sigma + 0.5 - 0.5;
        double log_ls = params[layout.log_lengthscale_hsgp_idx];
        grad[layout.log_lengthscale_hsgp_idx] = -log_ls;  // LogNormal(0,1) prior
        // N(0,I) prior on beta
        int M = data.hsgp_data.m_total;
        for (int j = 0; j < M; j++)
            grad[layout.hsgp_beta_start + j] = -hsgp_beta_ptr[j];
    }

    // Temporal GMRF priors
    if (has_gmrf_temporal) {
        tau_temporal_prior_grad(data, layout, tau_temporal, grad.data());
        temporal_rho_prior_grad(data, layout, rho_ar1, grad.data());
    }

    // Temporal GP priors
    if (has_temporal_gp) {
        double sigma_gp = std::sqrt(sigma2_tgp_comp);
        double rate_gp = -std::log(data.temporal_gp_sigma2_prior_alpha) / data.temporal_gp_sigma2_prior_U;
        grad[layout.log_sigma2_temporal_gp_idx] = -0.5 * rate_gp * sigma_gp + 0.5;
        // Phi: logit-bounded Jacobian
        double phi_range_tgp = phi_upper_tgp - phi_lower_tgp;
        grad[layout.logit_phi_temporal_gp_idx] = (phi_upper_tgp + phi_lower_tgp - 2.0 * phi_tgp_comp) / phi_range_tgp;

        if (use_nc_tgp) {
            // NC Jacobian: d/d(log_sigma2) of log|det(df/dz)| = T/2 per group
            grad[layout.log_sigma2_temporal_gp_idx] += 0.5 * T_gp * n_groups_gp;

            // NC Jacobian: d/d(log_phi) = -sum rho^2*(dt/phi) / (1-rho^2) per group
            double chi_tgp_prior = (phi_tgp_comp - phi_lower_tgp) * (phi_upper_tgp - phi_tgp_comp) /
                                   (phi_tgp_comp * phi_range_tgp);
            double jac_phi_log = 0.0;
            for (int t = 1; t < T_gp; t++) {
                double rho_t = nc_ws_composite.rho[t - 1];
                double rho2 = rho_t * rho_t;
                double dt = data.temporal_gp_data.time_values[t] - data.temporal_gp_data.time_values[t - 1];
                double one_minus_rho2 = ratiod_ar1::one_minus_rho2(rho_t);
                jac_phi_log -= rho2 * (dt / phi_tgp_comp) / one_minus_rho2;
            }
            grad[layout.logit_phi_temporal_gp_idx] += jac_phi_log * n_groups_gp * chi_tgp_prior;
        }
    }

    // TVC priors: Gamma on tau at the configured hyperparameters.
    if (layout.has_tvc && data.has_tvc) {
        for (int j = 0; j < n_tvc; j++) {
            grad[layout.log_tau_tvc_start + j] = ratiod_tvc::log_prior_tau_gamma_grad(
                tvc_tau_buf[j], data.tvc_tau_shape, data.tvc_tau_rate);
            if (data.tvc_data.structure == ratiod_temporal::TemporalType::AR1) {
                double u = (tvc_rho_buf[j] + 1.0) / 2.0;
                grad[layout.logit_rho_tvc_start + j] = ratiod_ar1::log_prior_logit_rho_grad(
                    u, ratiod_ar1::RHO_PRIOR_A, ratiod_ar1::RHO_PRIOR_B);
            }
        }
    }

    // SVC-HSGP priors: PC prior on sigma, LogNormal(0,1) on lengthscale, N(0,I) on beta
    // Must match compute_log_post: -rate*sigma + 0.5*log_sigma2 (Jacobian)
    if (layout.has_svc && data.has_svc && data.svc_is_hsgp) {
        double rate_svc = 4.6;  // Matches compute_log_post line 1781
        for (int j = 0; j < n_svc; j++) {
            double sigma2_j = std::exp(params[layout.log_sigma2_svc_start + j]);
            double sigma_j = std::sqrt(sigma2_j);
            // d/d(log_sigma2) [-rate*sigma + 0.5*log_sigma2] = -0.5*rate*sigma + 0.5
            grad[layout.log_sigma2_svc_start + j] = -0.5 * rate_svc * sigma_j + 0.5;
            // LogNormal(0,1) on lengthscale: d/d(log_ls) [-0.5*log_ls^2] = -log_ls
            double log_ls_j = params[layout.log_phi_svc_start + j];
            grad[layout.log_phi_svc_start + j] = -log_ls_j;
            // N(0,I) prior on SVC beta
            for (int m = 0; m < svc_m_total; m++)
                grad[layout.svc_w_start + j * svc_m_total + m] = -params[layout.svc_w_start + j * svc_m_total + m];
        }
    }

    // Latent priors
    if (K_latent > 0) {
        double latent_rate = data.latent_sigma_prior_rate;
        for (int k = 0; k < K_latent; k++)
            grad[layout.log_sigma_latent_start + k] = 1.0 - latent_rate * sigma_latent_vec[k];
    }

    // Multiscale temporal priors (PC priors on variances + AR1 rho prior)
    if (has_ms_temporal) {
        const auto& mst = data.multiscale_temporal_data;
        if (n_trend > 0)
            grad[layout.log_sigma2_trend_idx] = ratiod_temporal_grad::pc_prior_grad_log_sigma2(
                sigma2_trend, data.ms_sigma2_trend_prior_U, data.ms_sigma2_trend_prior_alpha) + 1.0;
        if (n_seasonal > 0)
            grad[layout.log_sigma2_seasonal_idx] = ratiod_temporal_grad::pc_prior_grad_log_sigma2(
                sigma2_seasonal, data.ms_sigma2_seasonal_prior_U, data.ms_sigma2_seasonal_prior_alpha) + 1.0;
        if (n_short > 0)
            grad[layout.log_sigma2_short_idx] = ratiod_temporal_grad::pc_prior_grad_log_sigma2(
                sigma2_short, data.ms_sigma2_short_prior_U, data.ms_sigma2_short_prior_alpha) + 1.0;
        if (mst.short_term_type == TemporalType::AR1 && layout.logit_rho_short_idx >= 0) {
            double u = (rho_short + 1.0) / 2.0;
            grad[layout.logit_rho_short_idx] = ratiod_ar1::log_prior_logit_rho_grad(
                u, ratiod_ar1::RHO_PRIOR_A, ratiod_ar1::RHO_PRIOR_B);
        }
    }

    // ST interaction hyperparameter priors: PC on sigma_st, Beta on the time
    // margin's correlation, and for HSGP-ST a PC on sigma and a LogNormal(0, 1)
    // on the lengthscale. The field's own contribution to each arrives below.
    if (has_st) {
        double sigma_st = 1.0 / std::sqrt(tau_st);
        double lambda = -std::log(data.st_sigma2_prior_alpha) / data.st_sigma2_prior_U;
        grad[layout.log_tau_st_idx] = 0.5 * lambda * sigma_st + 0.5 + 1.0;
        if (layout.is_st_gp) {
            // Uniform inside the bounds: each range carries only its log
            // Jacobian, the same term the density adds.
            grad[layout.log_phi_st_space_idx] = 1.0;
            grad[layout.log_phi_st_time_idx] = 1.0;
        }
        if (layout.logit_rho_st_idx >= 0) {
            grad[layout.logit_rho_st_idx] = ratiod_ar1::log_prior_rho_grad(
                rho_st, data.spatiotemporal_data.rho_prior_a,
                data.spatiotemporal_data.rho_prior_b);
        }
        if (data.st_is_hsgp) {
            grad[layout.log_sigma2_st_hsgp_idx] =
                -0.5 * 4.6 * std::sqrt(sigma2_st_hsgp) + 0.5;
            grad[layout.log_lengthscale_st_hsgp_idx] =
                -params[layout.log_lengthscale_st_hsgp_idx];
        }
    }

    // =========================================================================
    // Phase 3: Observation loop — compute eta, residuals, scatter gradients
    // =========================================================================
    RATIOD_TLS_WORKSPACE(std::vector<double>, grad_spatial_lik);
    RATIOD_TLS_WORKSPACE(std::vector<double>, grad_theta_lik);
    RATIOD_TLS_WORKSPACE(std::vector<double>, grad_temporal_lik);
    RATIOD_TLS_WORKSPACE(std::vector<double>, grad_hsgp_f);
    RATIOD_TLS_WORKSPACE(std::vector<double>, grad_delta_lik);
    RATIOD_TLS_WORKSPACE(std::vector<double>, grad_tvc_w);
    RATIOD_TLS_WORKSPACE(std::vector<double>, grad_factors_c);
    RATIOD_TLS_WORKSPACE(std::vector<double>, grad_svc_f);
    RATIOD_TLS_WORKSPACE(std::vector<double>, grad_svc_w_lik);
    RATIOD_TLS_WORKSPACE(std::vector<double>, grad_gp_w_lik);

    if (has_icar_bym2) grad_spatial_lik.assign(data.n_spatial_units, 0.0);
    if (has_icar_bym2 && layout.is_bym2) grad_theta_lik.assign(data.n_spatial_units, 0.0);
    if (has_gmrf_temporal) grad_temporal_lik.assign(T_temporal, 0.0);
    if (has_temporal_gp) grad_temporal_lik.assign(data.temporal_gp_data.n_groups * T_gp, 0.0);
    if (layout.is_hsgp && data.has_hsgp) grad_hsgp_f.assign(N, 0.0);
    if (has_nngp_gp) grad_gp_w_lik.assign(N_gp_comp, 0.0);
    if (has_st) grad_delta_lik.assign(ST_n, 0.0);
    if (layout.has_tvc && data.has_tvc) grad_tvc_w.assign(n_w, 0.0);
    if (K_latent > 0) grad_factors_c.assign(N * K_latent, 0.0);
    if (layout.has_svc && data.has_svc && data.svc_is_hsgp) grad_svc_f.assign(n_svc * N, 0.0);
    if (has_svc_nngp) grad_svc_w_lik.assign(static_cast<size_t>(n_svc) * N_svc, 0.0);

    // Phi likelihood gradient accumulators
    double grad_phi_num_lik_comp = 0.0;
    double grad_phi_denom_lik_comp = 0.0;

    // Scalar observation loop (handles all features uniformly)
    for (int i = 0; i < N; i++) {
        // Compute eta_num and eta_denom
        double eta_num_i = 0.0, eta_denom_i = 0.0;
        for (int j = 0; j < data.p_num; j++) eta_num_i += data.X_num_flat[i * data.p_num + j] * beta_num[j];
        if (!is_binomial)
            for (int j = 0; j < data.p_denom; j++) eta_denom_i += data.X_denom_flat[i * data.p_denom + j] * beta_denom[j];

        // Random effects
        {
            const double re_eta_i = re_blocks_eta(params, data, layout, re_ws, i);
            eta_num_i += re_eta_i;
            if (!is_binomial) eta_denom_i += re_eta_i;
        }

        // ICAR/BYM2 spatial
        if (has_icar_bym2 && data.spatial_group[i] > 0) {
            int s_unit = data.spatial_group[i] - 1;
            double spatial_eff;
            if (layout.is_bym2)
                spatial_eff = sigma_s_bym2 * data.bym2_scale_factor * spatial_phi[s_unit] + sigma_u_bym2 * theta_bym2[s_unit];
            else
                spatial_eff = spatial_phi[s_unit];
            eta_num_i += spatial_eff;
            if (!is_binomial) eta_denom_i += spatial_eff;
        }

        // HSGP spatial
        if (layout.is_hsgp && data.has_hsgp) {
            double hsgp_eff = hsgp_ws.hsgp_f[i];
            eta_num_i += hsgp_eff;
            if (!is_binomial && data.hsgp_data.shared) eta_denom_i += hsgp_eff;
        }

        // Multi-scale GP spatial
        if (has_msgp_nngp) {
            const int loc_i = data.multiscale_gp_data.obs_to_loc[i];
            const double ms_eff = msgp.local()[loc_i] + msgp.regional()[loc_i];
            eta_num_i += ms_eff;
            if (!is_binomial && data.multiscale_gp_data.shared) eta_denom_i += ms_eff;
        }

        // GP spatial (NNGP)
        if (has_nngp_gp) {
            double gp_eff = gp_w_comp[data.gp_data.obs_to_loc[i]];
            eta_num_i += gp_eff;
            if (!is_binomial && data.gp_data.shared) eta_denom_i += gp_eff;
        }

        // Temporal GMRF
        if (has_gmrf_temporal && !data.temporal_time_idx.empty() &&
            i < (int)data.temporal_time_idx.size() && data.temporal_time_idx[i] > 0) {
            int t = data.temporal_time_idx[i] - 1;
            int g = data.temporal_group_idx[i] - 1;
            int t_base = g * data.n_times + t;
            if (t_base >= 0 && t_base < T_temporal) {
                eta_num_i += phi_temporal[t_base];
                if (!is_binomial && data.temporal_shared) eta_denom_i += phi_temporal[t_base];
            }
        }

        // Temporal GP
        if (has_temporal_gp && !data.temporal_time_idx.empty() &&
            i < (int)data.temporal_time_idx.size() && data.temporal_time_idx[i] > 0) {
            int t = data.temporal_time_idx[i] - 1;
            int g = data.temporal_group_idx[i] - 1;
            int t_base = g * T_gp + t;
            if (t_base >= 0 && t_base < (int)temporal_gp_f.size()) {
                eta_num_i += temporal_gp_f[t_base];
                if (!is_binomial && data.temporal_shared) eta_denom_i += temporal_gp_f[t_base];
            }
        }

        // TVC
        if (layout.has_tvc && data.has_tvc) {
            eta_num_i += tvc_eta_precomp[i];
            if (!is_binomial) eta_denom_i += tvc_eta_precomp[i];
        }

        // SVC, on either approximation
        if (has_svc_hsgp || has_svc_nngp) {
            eta_num_i += svc_eta_precomp[i];
            if (!is_binomial) eta_denom_i += svc_eta_precomp[i];
        }

        // Latent factors
        if (K_latent > 0) {
            if (data.latent_shared) {
                eta_num_i += latent_eta_precomp[i];
                eta_denom_i += latent_eta_precomp[i];
            } else {
                eta_num_i += latent_eta_precomp[i];
            }
        }

        // Multiscale temporal
        if (has_ms_temporal) {
            const auto& mst = data.multiscale_temporal_data;
            if (!mst.time_index.empty() && i < (int)mst.time_index.size() && mst.time_index[i] > 0) {
                int t_idx = mst.time_index[i] - 1;
                obs_t_idx_ms_c[i] = t_idx;
                eta_num_i += ms_effect_by_time_c[t_idx];
                if (!is_binomial && mst.shared) eta_denom_i += ms_effect_by_time_c[t_idx];
            }
        }

        // Spatiotemporal
        if (has_st) {
            double st_eff = 0.0;
            if (data.st_is_hsgp) {
                // HSGP-ST: Phi-weighted basis-temporal interaction
                int t = data.spatiotemporal_data.t_idx[i] - 1;
                int M_st = data.st_hsgp_data.m_total;
                int T_st_c = data.spatiotemporal_data.n_times;
                for (int j = 0; j < M_st; j++)
                    st_eff += data.st_hsgp_data.phi_flat[i * M_st + j] * st_delta[j * T_st_c + t];
            } else if (data.spatiotemporal_data.st_flat[i] > 0) {
                st_eff = st_delta[data.spatiotemporal_data.st_flat[i] - 1];
            }
            eta_num_i += st_eff;
            if (!is_binomial && data.spatiotemporal_data.shared) eta_denom_i += st_eff;
        }

        // Compute residuals
        double dLL_num = 0.0, dLL_denom = 0.0;
        double mu_num = std::exp(eta_num_i);
        double mu_denom = is_binomial ? 0.0 : std::exp(eta_denom_i);

        switch (data.model_type) {
            case ModelType::BINOMIAL: {
                double p = 1.0 / (1.0 + std::exp(-eta_num_i));
                dLL_num = data.y_num[i] - data.y_denom[i] * p;
                break;
            }
            case ModelType::BETA_BINOMIAL: {
                double p = 1.0 / (1.0 + std::exp(-eta_num_i));
                double phi_bb = phi_num;
                double a = p * phi_bb;
                double b = (1.0 - p) * phi_bb;
                double psi_ab = R::digamma(a + b);
                dLL_num = phi_bb * (R::digamma(data.y_num[i] + a) - R::digamma(data.y_denom[i] - data.y_num[i] + b)
                          + R::digamma(b) - R::digamma(a)) * p * (1.0 - p);
                break;
            }
            case ModelType::NEGBIN_NEGBIN: {
                double y_n = data.y_num[i], y_d = data.y_denom[i];
                double dn_n = mu_num + phi_num, dn_d = mu_denom + phi_denom;
                dLL_num = y_n - mu_num * (y_n + phi_num) / dn_n;
                dLL_denom = y_d - mu_denom * (y_d + phi_denom) / dn_d;
                // phi likelihood gradient (d/d(log_phi))
                grad_phi_num_lik_comp += R::digamma(y_n + phi_num) - R::digamma(phi_num)
                    + std::log(phi_num / dn_n) + (mu_num - y_n) / dn_n;
                grad_phi_denom_lik_comp += R::digamma(y_d + phi_denom) - R::digamma(phi_denom)
                    + std::log(phi_denom / dn_d) + (mu_denom - y_d) / dn_d;
                break;
            }
            case ModelType::POISSON_GAMMA: {
                double y_d_cont = data.y_denom_cont[i];
                dLL_num = data.y_num[i] - mu_num;
                if (y_d_cont > 0) {
                    dLL_denom = phi_denom * (y_d_cont / mu_denom - 1.0);
                    double log_rate = std::log(phi_denom / mu_denom);
                    grad_phi_denom_lik_comp += log_rate + 1.0 + std::log(std::max(y_d_cont, 1e-10))
                        - R::digamma(phi_denom) - y_d_cont / mu_denom;
                }
                break;
            }
            case ModelType::NEGBIN_GAMMA: {
                double y_n = data.y_num[i];
                double y_d_cont = data.y_denom_cont[i];
                double dn_n = mu_num + phi_num;
                dLL_num = y_n - mu_num * (y_n + phi_num) / dn_n;
                grad_phi_num_lik_comp += R::digamma(y_n + phi_num) - R::digamma(phi_num)
                    + std::log(phi_num / dn_n) + (mu_num - y_n) / dn_n;
                if (y_d_cont > 0) {
                    dLL_denom = phi_denom * (y_d_cont / mu_denom - 1.0);
                    double log_rate = std::log(phi_denom / mu_denom);
                    grad_phi_denom_lik_comp += log_rate + 1.0 + std::log(std::max(y_d_cont, 1e-10))
                        - R::digamma(phi_denom) - y_d_cont / mu_denom;
                }
                break;
            }
            case ModelType::GAMMA_GAMMA: {
                double y_n_cont = data.y_denom_cont.empty() ? (double)data.y_num[i] : data.y_denom_cont[i];
                // For GG, both y_num and y_denom are continuous — use y_denom_cont for denom, y_num cast for num
                double y_num_d = (double)data.y_num[i];  // y_num is stored as int but represents continuous
                dLL_num = phi_num * (y_num_d / mu_num - 1.0);
                dLL_denom = phi_denom * (y_n_cont / mu_denom - 1.0);
                grad_phi_num_lik_comp += phi_num * (std::log(phi_num / mu_num) + 1.0
                    + std::log(std::max(y_num_d, 1e-10)) - R::digamma(phi_num) - y_num_d / mu_num);
                grad_phi_denom_lik_comp += phi_denom * (std::log(phi_denom / mu_denom) + 1.0
                    + std::log(std::max(y_n_cont, 1e-10)) - R::digamma(phi_denom) - y_n_cont / mu_denom);
                break;
            }
            case ModelType::LOGNORMAL: {
                double log_y_num = std::log(std::max((double)data.y_num[i], 1e-10));
                double log_y_denom = std::log(std::max(data.y_denom_cont[i], 1e-10));
                double sigma2_num = 1.0 / phi_num;
                double sigma2_denom = 1.0 / phi_denom;
                dLL_num = (log_y_num - eta_num_i) / sigma2_num;
                dLL_denom = (log_y_denom - eta_denom_i) / sigma2_denom;
                // phi = 1/sigma2, log_phi gradient
                double resid_n = log_y_num - eta_num_i;
                double resid_d = log_y_denom - eta_denom_i;
                grad_phi_num_lik_comp += 0.5 * (resid_n * resid_n * phi_num - 1.0);
                grad_phi_denom_lik_comp += 0.5 * (resid_d * resid_d * phi_denom - 1.0);
                break;
            }
            default: break;
        }

        if (fuse_lp) {
            // Accumulate log-likelihood (simplified — full version in compute_log_post)
            // This is approximate; for exact fused LP, use compute_log_post with skip_obs_loop
        }

        double dLL_shared = dLL_num + dLL_denom;

        // Scatter to beta
        for (int j = 0; j < data.p_num; j++) grad[layout.beta_num_start + j] += dLL_num * data.X_num_flat[i * data.p_num + j];
        if (!is_binomial)
            for (int j = 0; j < data.p_denom; j++) grad[layout.beta_denom_start + j] += dLL_denom * data.X_denom_flat[i * data.p_denom + j];

        // Scatter to the RE blocks
        re_blocks_scatter(data, layout, re_ws, i, dLL_shared, grad);

        // Scatter to ICAR/BYM2 spatial
        if (has_icar_bym2 && data.spatial_group[i] > 0) {
            int s_unit = data.spatial_group[i] - 1;
            grad_spatial_lik[s_unit] += dLL_shared;
            if (layout.is_bym2) grad_theta_lik[s_unit] += dLL_shared;
        }

        // Scatter to HSGP
        if (layout.is_hsgp && data.has_hsgp) {
            grad_hsgp_f[i] = dLL_num + (data.hsgp_data.shared ? dLL_denom : 0.0);
        }

        // Scatter to the multi-scale GP field, by location rather than by row
        if (has_msgp_nngp) {
            const int loc_i = data.multiscale_gp_data.obs_to_loc[i];
            grad_ms_w_lik[loc_i] +=
                data.multiscale_gp_data.shared ? dLL_shared : dLL_num;
        }

        // Scatter to the GP field, by location rather than by row
        if (has_nngp_gp) {
            grad_gp_w_lik[data.gp_data.obs_to_loc[i]] +=
                data.gp_data.shared ? dLL_shared : dLL_num;
        }

        // Scatter to temporal GMRF
        if (has_gmrf_temporal && !data.temporal_time_idx.empty() &&
            i < (int)data.temporal_time_idx.size() && data.temporal_time_idx[i] > 0) {
            int t = data.temporal_time_idx[i] - 1;
            int g = data.temporal_group_idx[i] - 1;
            int t_base = g * data.n_times + t;
            if (t_base >= 0 && t_base < T_temporal)
                grad_temporal_lik[t_base] += data.temporal_shared ? dLL_shared : dLL_num;
        }

        // Scatter to temporal GP
        if (has_temporal_gp && !data.temporal_time_idx.empty() &&
            i < (int)data.temporal_time_idx.size() && data.temporal_time_idx[i] > 0) {
            int t = data.temporal_time_idx[i] - 1;
            int g = data.temporal_group_idx[i] - 1;
            int t_base = g * T_gp + t;
            if (t_base >= 0 && t_base < (int)temporal_gp_f.size())
                grad_temporal_lik[t_base] += data.temporal_shared ? dLL_shared : dLL_num;
        }

        // Scatter to TVC
        if (layout.has_tvc && data.has_tvc) {
            int t = data.tvc_data.time_index[i] - 1;
            int g = data.tvc_data.group_index[i] - 1;
            for (int j = 0; j < n_tvc; j++) {
                int w_idx = (g * n_tvc + j) * n_tvc_times + t;
                grad_tvc_w[w_idx] += dLL_shared * data.tvc_data.X_tvc[i * n_tvc + j];
            }
        }

        // Scatter to SVC
        if (has_svc_hsgp) {
            for (int j = 0; j < n_svc; j++) {
                double x_ij = data.svc_data.X_svc[i * n_svc + j];
                grad_svc_f[j * N + i] = dLL_shared * x_ij;
            }
        }
        if (has_svc_nngp) {
            const double dLL_svc = data.svc_data.shared ? dLL_shared : dLL_num;
            for (int j = 0; j < n_svc; j++)
                grad_svc_w_lik[j * N_svc + i] +=
                    dLL_svc * data.svc_data.X_svc[i * n_svc + j];
        }

        // Scatter to latent factors
        if (K_latent > 0) {
            double dLL_latent = data.latent_shared ? dLL_shared : dLL_num;
            for (int k = 0; k < K_latent; k++) {
                grad_factors_c[i * K_latent + k] += dLL_latent * sigma_latent_vec[k];
                grad[layout.log_sigma_latent_start + k] += dLL_latent * factors_constrained[i * K_latent + k] * sigma_latent_vec[k];
            }
        }

        // Scatter to multiscale temporal
        if (has_ms_temporal && obs_t_idx_ms_c[i] >= 0) {
            const auto& mst = data.multiscale_temporal_data;
            int t_idx = obs_t_idx_ms_c[i];
            double dLL_temporal = mst.shared ? dLL_shared : dLL_num;
            if (trend != nullptr && t_idx < n_trend) grad_trend_lik_c[t_idx] += dLL_temporal;
            if (seasonal != nullptr && mst.seasonal_period > 0) {
                int s_idx = t_idx % mst.seasonal_period;
                if (s_idx < n_seasonal) grad_seasonal_lik_c[s_idx] += dLL_temporal;
            }
            if (short_term != nullptr && t_idx < n_short) grad_short_lik_c[t_idx] += dLL_temporal;
        }

        // Scatter to ST interaction
        if (has_st) {
            double dLL_st = data.spatiotemporal_data.shared ? dLL_shared : dLL_num;
            if (data.st_is_hsgp) {
                int t = data.spatiotemporal_data.t_idx[i] - 1;
                int M_st = data.st_hsgp_data.m_total;
                int T_st_c = data.spatiotemporal_data.n_times;
                for (int j = 0; j < M_st; j++)
                    grad_delta_lik[j * T_st_c + t] += data.st_hsgp_data.phi_flat[i * M_st + j] * dLL_st;
            } else if (data.spatiotemporal_data.st_flat[i] > 0) {
                grad_delta_lik[data.spatiotemporal_data.st_flat[i] - 1] += dLL_st;
            }
        }
    }

    // Add phi likelihood gradient contributions (multiply by phi for log-scale Jacobian)
    if (layout.has_phi_num) {
        grad[layout.log_phi_num_idx] += grad_phi_num_lik_comp * phi_num;
    }
    if (layout.has_phi_denom) {
        grad[layout.log_phi_denom_idx] += grad_phi_denom_lik_comp * phi_denom;
    }

    // =========================================================================
    // Phase 4: Structural/GMRF prior gradients (post-observation loop)
    // =========================================================================

    // ICAR/BYM2 spatial GMRF
    if (has_icar_bym2) {
        spatial_lik_grad_to_param_scale(data, layout, sigma_s_bym2, sigma_u_bym2,
                                        theta_bym2,
                                        layout.is_bym2 ? grad_theta_lik.data() : nullptr,
                                        grad_spatial_lik.data(), grad.data());
        spatial_gmrf_prior_grad(data, layout, spatial_phi, spatial_phi_prior,
                                centering.raw_sum(),
                                centering.active(), tau_spatial,
                                sigma_s_bym2, sigma_u_bym2, rho_bym2, rho_car, theta_bym2,
                                grad_spatial_lik.data(), grad.data());
    }

    // Multi-scale GP field: its own prior and the likelihood scatter, onto
    // whichever coordinate the two scales are sampled in.
    if (has_msgp_nngp) {
        msgp.accumulate(grad_ms_w_lik.data(), data, layout, grad.data());
    }

    // GP (NNGP) field: the likelihood scatter reaches the sampled parameters
    // through the non-centred backward pass, or lands on them directly.
    if (has_nngp_gp) {
        if (gp_use_nc) {
            std::vector<double> grad_z(N_gp_comp, 0.0);
            double grad_log_sigma2_lik = 0.0, grad_log_phi_lik = 0.0, grad_log_phi_jac = 0.0;
            ratiod_gp::nngp_nc_backward(
                &params[layout.gp_w_start], gp_sigma2_comp, gp_phi_comp,
                data.gp_data, gp_nc_ws, grad_gp_w_lik.data(), grad_z.data(),
                grad_log_sigma2_lik, grad_log_phi_lik, grad_log_phi_jac);
            for (int i = 0; i < N_gp_comp; i++)
                grad[layout.gp_w_start + i] += grad_z[i];
            grad[layout.log_sigma2_gp_idx] += grad_log_sigma2_lik;
            grad[layout.log_phi_gp_idx] += grad_log_phi_lik;
        } else {
            for (int i = 0; i < N_gp_comp; i++)
                grad[layout.gp_w_start + i] += grad_gp_w_lik[i];
        }
    }

    if (has_svc_nngp) {
        ratiod_svc::svc_center_terms(grad_svc_w_lik.data(), data.svc_data);
        ratiod_svc::svc_field_accumulate<ratiod_svc::SVC_GRAD_SLOT>(
            &params[layout.svc_w_start], svc_sigma2_c.data(), svc_phi_c.data(),
            data.svc_data, data.svc_gp_view, data.svc_noncentered,
            grad_svc_w_lik.data(), layout.svc_w_start,
            layout.log_sigma2_svc_start, layout.log_phi_svc_start, grad.data());
    }

    // HSGP spectral density gradients
    if (layout.is_hsgp && data.has_hsgp) {
        int M = data.hsgp_data.m_total;
        std::copy(grad_hsgp_f.begin(), grad_hsgp_f.end(), hsgp_ws.grad_f.begin());
        double grad_log_sigma2 = 0.0, grad_log_lengthscale = 0.0;
        ratiod_hsgp::hsgp_compute_gradients_ws(
            hsgp_beta_ptr, hsgp_sigma2, hsgp_lengthscale,
            data.hsgp_data, hsgp_ws, grad_log_sigma2, grad_log_lengthscale);
        grad[layout.log_sigma2_hsgp_idx] += grad_log_sigma2;
        grad[layout.log_lengthscale_hsgp_idx] += grad_log_lengthscale;
        for (int j = 0; j < M; j++)
            grad[layout.hsgp_beta_start + j] += hsgp_ws.grad_beta_out[j];
    }

    // Temporal GMRF prior
    if (has_gmrf_temporal && T_temporal > 0) {
        temporal_gmrf_prior_grad(data, layout, tau_temporal, rho_ar1,
                                 tview, T_temporal, grad_temporal_lik.data(), grad.data());
    }

    // Temporal GP backward pass
    if (has_temporal_gp) {
        int T_len_gp = layout.temporal_end - layout.temporal_start;

        if (use_nc_tgp) {
            // Copy likelihood gradients to nc workspace
            for (int k = 0; k < n_groups_gp * T_gp; k++)
                nc_ws_composite.dL_df[k] = grad_temporal_lik[k];

            double grad_log_sigma2_gp = 0.0, grad_log_phi_gp = 0.0;
            ratiod_temporal_gp::temporal_gp_nc_backward(
                z_temporal_gp, T_gp, n_groups_gp, sigma2_tgp_comp, phi_tgp_comp,
                data.temporal_gp_data.time_values, nc_ws_composite,
                &grad[layout.temporal_start],
                grad_log_sigma2_gp, grad_log_phi_gp);
            grad[layout.log_sigma2_temporal_gp_idx] += grad_log_sigma2_gp;
            // Convert log_phi gradient to logit_phi gradient
            double chi_tgp = (phi_tgp_comp - phi_lower_tgp) * (phi_upper_tgp - phi_tgp_comp) /
                             (phi_tgp_comp * (phi_upper_tgp - phi_lower_tgp));
            grad[layout.logit_phi_temporal_gp_idx] += grad_log_phi_gp * chi_tgp;
        } else {
            // Centered: just add likelihood gradients
            for (int k = 0; k < T_len_gp; k++)
                grad[layout.temporal_start + k] = grad_temporal_lik[k];
        }
    }

    // Multiscale temporal GMRF prior gradients
    if (has_ms_temporal) {
        ratiod_temporal_grad::MultiscaleTemporalGradients ms_grads;
        ratiod_temporal_grad::multiscale_temporal_prior_gradients(
            trend, n_trend, seasonal, n_seasonal, short_term, n_short,
            sigma2_trend, sigma2_seasonal, sigma2_short, rho_short,
            data.multiscale_temporal_data, ms_grads);
        double nc_g_log_s2_trend_c = 0.0, nc_g_log_s2_seasonal_c = 0.0;
        ratiod_temporal_grad::project_ms_lik_gradients(
            grad_trend_lik_c.data(), n_trend, grad_seasonal_lik_c.data(), n_seasonal,
            data.multiscale_temporal_data,
            {trend, ms_arms_c.trend, sigma2_trend, &nc_g_log_s2_trend_c},
            {seasonal, ms_arms_c.seasonal, sigma2_seasonal, &nc_g_log_s2_seasonal_c});
        for (int t = 0; t < n_trend; t++) grad[layout.trend_start + t] = grad_trend_lik_c[t] + ms_grads.grad_trend[t];
        for (int t = 0; t < n_seasonal; t++) grad[layout.seasonal_start + t] = grad_seasonal_lik_c[t] + ms_grads.grad_seasonal[t];
        for (int t = 0; t < n_short; t++) grad[layout.short_term_start + t] = grad_short_lik_c[t] + ms_grads.grad_short_term[t];
        if (n_trend > 0)
            grad[layout.log_sigma2_trend_idx] += ms_grads.grad_log_sigma2_trend + nc_g_log_s2_trend_c;
        if (n_seasonal > 0)
            grad[layout.log_sigma2_seasonal_idx] += ms_grads.grad_log_sigma2_seasonal + nc_g_log_s2_seasonal_c;
        if (n_short > 0) grad[layout.log_sigma2_short_idx] += ms_grads.grad_log_sigma2_short;
        if (data.multiscale_temporal_data.short_term_type == TemporalType::AR1 && layout.logit_rho_short_idx >= 0)
            grad[layout.logit_rho_short_idx] += ms_grads.grad_logit_rho_short;
    }

    // TVC structural prior gradients
    if (layout.has_tvc && data.has_tvc) {
        // Initialize TVC gradient workspace
        RATIOD_TLS_WORKSPACE(ratiod_tvc::TVCGradientWS, tvc_grad_ws);
        RATIOD_TLS_WORKSPACE(std::vector<double>, tvc_grad_w_buf);
        RATIOD_TLS_WORKSPACE(std::vector<double>, tvc_grad_log_tau_buf);
        RATIOD_TLS_WORKSPACE(std::vector<double>, tvc_grad_logit_rho_buf);
        RATIOD_TLS_WORKSPACE(std::vector<double>, tvc_grad_w_jg_buf);
        RATIOD_TLS_WORKSPACE(std::vector<double>, tvc_d_buf);
        tvc_grad_w_buf.assign(n_w, 0.0);
        tvc_grad_log_tau_buf.assign(n_tvc, 0.0);
        tvc_grad_logit_rho_buf.assign(n_tvc, 0.0);
        tvc_grad_w_jg_buf.resize(n_tvc_times);
        tvc_d_buf.resize(n_tvc_times);

        tvc_grad_ws.grad_w = tvc_grad_w_buf.data();
        tvc_grad_ws.grad_log_tau = tvc_grad_log_tau_buf.data();
        tvc_grad_ws.grad_logit_rho = tvc_grad_logit_rho_buf.data();
        tvc_grad_ws.grad_w_jg = tvc_grad_w_jg_buf.data();
        tvc_grad_ws.d_buf = tvc_d_buf.data();
        tvc_grad_ws.n_w = n_w;
        tvc_grad_ws.n_tvc = n_tvc;

        ratiod_tvc::tvc_prior_gradients_ws(
            tvc_w_flat_buf.data(), data.tvc_data,
            tvc_tau_buf.data(), tvc_rho_buf.data(), tvc_grad_ws);

        // Add likelihood + prior to main gradient
        for (int k = 0; k < n_w; k++)
            grad[layout.tvc_w_start + k] += grad_tvc_w[k] + tvc_grad_w_buf[k];
        for (int j = 0; j < n_tvc; j++) {
            grad[layout.log_tau_tvc_start + j] += tvc_grad_log_tau_buf[j];
            if (data.tvc_data.structure == ratiod_temporal::TemporalType::AR1)
                grad[layout.logit_rho_tvc_start + j] += tvc_grad_logit_rho_buf[j];
        }
    }

    // SVC spectral density gradients
    if (layout.has_svc && data.has_svc && data.svc_is_hsgp) {
        for (int j = 0; j < n_svc; j++) {
            double sigma2_j = std::exp(params[layout.log_sigma2_svc_start + j]);
            double lengthscale_j = std::exp(params[layout.log_phi_svc_start + j]);
            const double* beta_j = &params[layout.svc_w_start + j * svc_m_total];

            // Re-evaluate to cache sqrt_S
            ratiod_hsgp::hsgp_evaluate_ws(beta_j, sigma2_j, lengthscale_j,
                                           data.svc_hsgp_data, svc_hsgp_ws);

            // Copy per-SVC-term grad_f
            for (int i = 0; i < N; i++) svc_hsgp_ws.grad_f[i] = grad_svc_f[j * N + i];

            double gls2 = 0.0, gll = 0.0;
            ratiod_hsgp::hsgp_compute_gradients_ws(
                beta_j, sigma2_j, lengthscale_j,
                data.svc_hsgp_data, svc_hsgp_ws, gls2, gll);

            grad[layout.log_sigma2_svc_start + j] += gls2;
            grad[layout.log_phi_svc_start + j] += gll;
            for (int m = 0; m < svc_m_total; m++)
                grad[layout.svc_w_start + j * svc_m_total + m] += svc_hsgp_ws.grad_beta_out[m];
        }
    }

    // Latent factor constraint chain rule
    if (K_latent > 0) {
        // N(0,1) prior on constrained factors
        for (int k = 0; k < K_latent; k++)
            for (int i = 0; i < N; i++)
                grad_factors_c[i * K_latent + k] -= factors_constrained[i * K_latent + k];

        // Chain rule: constrained -> raw
        if (data.latent_constraint == 0) {
            // Sum-to-zero: d/d(raw[i,k]) = d/d(constrained[i,k]) - mean(d/d(constrained[:,k]))
            for (int k = 0; k < K_latent; k++) {
                double sum_gc = 0.0;
                for (int i = 0; i < N; i++) sum_gc += grad_factors_c[i * K_latent + k];
                double mean_gc = sum_gc / N;
                for (int i = 0; i < N; i++)
                    grad[layout.latent_factor_start + i * K_latent + k] += grad_factors_c[i * K_latent + k] - mean_gc;
            }
        } else {
            for (int j = 0; j < N * K_latent; j++)
                grad[layout.latent_factor_start + j] += grad_factors_c[j];
        }
    }

    // ST interaction prior gradients: the same implementation
    // compute_gradient_spatiotemporal_handcoded drives (st_prior_grad.h).
    if (has_st) {
        const ratiod_spatiotemporal::StPriorGrad st_pg =
            ratiod_spatiotemporal::st_interaction_prior_grad(
                data.spatiotemporal_data, data.st_is_hsgp, data.st_hsgp_data,
                z_or_delta_st, st_delta, grad_delta_lik.data(),
                tau_st, rho_st, sigma2_st_hsgp, lengthscale_st_hsgp,
                phi_st_space, phi_st_time,
                st_use_nc, &grad[layout.st_delta_start]);

        grad[layout.log_tau_st_idx] += st_pg.log_tau;
        if (layout.logit_rho_st_idx >= 0) {
            grad[layout.logit_rho_st_idx] += st_pg.logit_rho;
        }
        if (data.st_is_hsgp) {
            grad[layout.log_sigma2_st_hsgp_idx] += st_pg.log_sigma2_hsgp;
            grad[layout.log_lengthscale_st_hsgp_idx] += st_pg.log_lengthscale_hsgp;
        }
        if (layout.is_st_gp) {
            grad[layout.log_phi_st_space_idx] += st_pg.log_phi_space;
            grad[layout.log_phi_st_time_idx] += st_pg.log_phi_time;
        }
    }

    // The chain rule back onto the sampled parameters.
    re_blocks_writeback(params, data, layout, grad, re_ws);

    // Fused log-posterior. params_in, not params: compute_log_post centres its
    // own copy and recovers phi_raw_sum from the raw field, which the centred
    // copy has zeroed.
    if (fuse_lp && !layout.has_zi) {
        *log_post_out = compute_log_post(params_in, data, layout);
    }
}

// g_gradient_mode defined earlier in file (before verify_gradient_runtime)

// Set global gradient mode (called at start of sampling)
void set_gradient_mode(GradientMode mode) {
    g_gradient_mode = mode;
}

void compute_gradient(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    std::vector<double>& grad,
    double* log_post_out
) {
    // Delegate to resolve_gradient_fn() — single source of truth for dispatch logic.
    // For hot paths (NUTS leapfrog), callers should resolve once via resolve_gradient_fn()
    // and reuse the pointer. This convenience wrapper is for cold-path callers.
    GradientFn fn = resolve_gradient_fn(g_gradient_mode, data, layout);
    fn(params, data, layout, grad, log_post_out);
}

// =====================================================================
// The latent blocks a model carries, as one mask. A specialized gradient
// declares the blocks it writes; a block the model carries outside that set is
// one the function leaves at zero while every other block is differentiated at
// a linear predictor missing it.
//
// The temporal margin contributes exactly one bit: a GP margin is
// GF_TEMPORAL_GP and not GF_TEMPORAL, the two being different blocks with
// different gradients, so a function writing one does not write the other.
// =====================================================================

enum GradFeature : unsigned {
  GF_AREAL          = 1u << 0,   // intrinsic ICAR / BYM2 field
  GF_GP             = 1u << 1,
  GF_MULTISCALE_GP  = 1u << 2,
  GF_HSGP           = 1u << 3,
  GF_SVC            = 1u << 4,
  GF_TEMPORAL       = 1u << 5,   // RW1 / RW2 / AR1 margin
  GF_TEMPORAL_GP    = 1u << 6,
  GF_MS_TEMPORAL    = 1u << 7,
  GF_TVC            = 1u << 8,
  GF_SPATIOTEMPORAL = 1u << 9,
  GF_LATENT         = 1u << 10,
  GF_RE_SLOPES      = 1u << 11,  // no specialized function writes the slope block
  // A collapsed field is a marginal, not a latent block: it is its own bit
  // rather than GF_AREAL / GF_GP so that a function written for the sampled
  // field cannot be selected for it.
  GF_AREAL_COLLAPSED = 1u << 12,
  GF_GP_COLLAPSED    = 1u << 13,
  // Crossed / nested RE terms. The specialized functions read the legacy
  // single-term block (data.re_group, layout.re_start), which holds one term
  // of several, so a model carrying more than one needs its own bit to fall
  // through to the composite on.
  GF_RE_MULTI        = 1u << 14,
  // Proper CAR carries a rho of its own and a log-determinant that moves with
  // it, and its full-rank precision leaves the field's mean in the likelihood
  // where an intrinsic field's is removed. A function written for the
  // intrinsic field writes neither, so proper CAR gets its own bit rather than
  // GF_AREAL and reaches only what names it.
  GF_AREAL_PROPER    = 1u << 15
};

inline unsigned model_grad_features(const ModelData& data, const ParamLayout& layout) {
  unsigned f = 0u;
  const bool areal_collapsed = layout.is_icar_collapsed || layout.is_bym2_collapsed;
  if (layout.has_spatial)
    f |= areal_collapsed  ? GF_AREAL_COLLAPSED
       : layout.is_car_proper ? GF_AREAL_PROPER
                              : GF_AREAL;
  if (layout.is_gp && data.has_gp)
    f |= (layout.is_gp_collapsed && data.gp_collapsed) ? GF_GP_COLLAPSED : GF_GP;
  if (layout.is_multiscale_gp && data.has_multiscale_gp) f |= GF_MULTISCALE_GP;
  if (layout.is_hsgp && data.has_hsgp)                   f |= GF_HSGP;
  if (layout.has_svc && data.has_svc)                    f |= GF_SVC;
  if (layout.is_temporal_gp)                             f |= GF_TEMPORAL_GP;
  else if (layout.has_temporal)                          f |= GF_TEMPORAL;
  if (layout.has_multiscale_temporal)                    f |= GF_MS_TEMPORAL;
  if (layout.has_tvc && data.has_tvc)                    f |= GF_TVC;
  if (layout.has_spatiotemporal)                         f |= GF_SPATIOTEMPORAL;
  if (layout.has_latent && data.latent_n_factors > 0)    f |= GF_LATENT;
  if (layout.has_re_slopes)                              f |= GF_RE_SLOPES;
  if (data.n_re_terms > 1)                               f |= GF_RE_MULTI;
  return f;
}

// =====================================================================
// Resolve gradient function pointer — SINGLE SOURCE OF TRUTH for dispatch.
// Called once at sampling start, returns a function pointer to eliminate
// per-call branching during leapfrog steps. compute_gradient() delegates here.
// =====================================================================

GradientFn resolve_gradient_fn(GradientMode mode, const ModelData& data, const ParamLayout& layout) {
    // Explicit mode overrides
    if (mode == GradientMode::NUMERICAL)
        return &compute_gradient_numerical;
    if (mode == GradientMode::AUTODIFF_TAPE)
        return &compute_gradient_autodiff;
    if (mode == GradientMode::AUTODIFF_ARENA)
        return &compute_gradient_arena;
    if (mode == GradientMode::AUTODIFF_FORWARD)
        return &compute_gradient_forward;

    // A restricted spatial field reaches eta through a projection across all
    // observations, which no hand-coded gradient carries; the arena gradient
    // differentiates the log posterior that does.
    if (data.has_rsr)
        return &compute_gradient_arena;

    // AUTO or HANDCODED: use fastest available (H > A_r > A > N)
    if (can_use_analytical_gradient(data, layout)) {
        return &compute_gradient_analytical;
    }
    // Specialized H-mode functions. Each writes the blocks its own body writes
    // and no others, so a model carrying anything outside that set falls
    // through to the composite. The mask passed to covers() is what the
    // function handles; covers() is the whole of the exclusion, in place of
    // one negation per feature per function. A bit added to GradFeature is a
    // block every entry that does not name it falls through on, rather than
    // one every guard has to remember to exclude.
    const unsigned mf = model_grad_features(data, layout);
    auto covers = [mf](unsigned writes) { return (mf & ~writes) == 0u; };

    if (layout.is_hsgp && data.has_hsgp &&
        covers(GF_HSGP | GF_TEMPORAL))
        return &compute_gradient_hsgp;
    // Collapsed GP: analytical gradient + numerical Laplace correction
    if (layout.is_gp_collapsed && data.has_gp && data.gp_collapsed &&
        covers(GF_GP_COLLAPSED))
        return &compute_gradient_gp_collapsed;
    // Collapsed ICAR/BYM2: analytical gradient + numerical Laplace correction
    if ((layout.is_icar_collapsed || layout.is_bym2_collapsed) &&
        covers(GF_AREAL_COLLAPSED | GF_TEMPORAL))
        return &compute_gradient_icar_collapsed;
    // A collapsed field alongside anything else. Its marginal is an inner
    // Laplace at a mode that moves with every other block in eta, which the
    // two functions above carry for a companion temporal margin and for
    // nothing else, and which the composite has no branch for at all. The
    // numerical gradient of the density is the correct one for the
    // combination, so it is what the dispatch returns, rather than a
    // specialized function that would leave the second block at zero.
    if (mf & (GF_AREAL_COLLAPSED | GF_GP_COLLAPSED))
        return &compute_gradient_numerical;
    if (layout.is_gp && data.has_gp && covers(GF_GP))
        return &compute_gradient_gp_handcoded;
    if (layout.is_multiscale_gp && data.has_multiscale_gp && data.msgp_is_hsgp &&
        covers(GF_MULTISCALE_GP | GF_TEMPORAL))
        return &compute_gradient_msgp_hsgp;
    if (layout.is_multiscale_gp && data.has_multiscale_gp && layout.has_temporal &&
        covers(GF_MULTISCALE_GP | GF_TEMPORAL))
        return &compute_gradient_msgp_temporal_handcoded;
    if (layout.is_multiscale_gp && data.has_multiscale_gp &&
        covers(GF_MULTISCALE_GP))
        return &compute_gradient_msgp_handcoded;
    if (layout.is_gp && covers(GF_GP | GF_TEMPORAL))
        return &compute_gradient_gp_temporal_handcoded;
    if (layout.has_svc && data.has_svc && data.svc_is_hsgp && covers(GF_SVC))
        return &compute_gradient_svc_hsgp_handcoded;
    if (layout.has_svc && data.has_svc && covers(GF_SVC))
        return &compute_gradient_svc_handcoded;
    if (layout.has_tvc && data.has_tvc && covers(GF_TVC))
        return &compute_gradient_tvc_handcoded;
    if (layout.has_spatiotemporal &&
        data.spatiotemporal_data.type != STType::NONE &&
        layout.st_delta_start >= 0 && layout.log_tau_st_idx >= 0 &&
        covers(GF_AREAL | GF_TEMPORAL | GF_SPATIOTEMPORAL))
        return &compute_gradient_spatiotemporal_handcoded;
    if (layout.is_temporal_gp && layout.has_temporal &&
        data.temporal_gp_data.cov_type == ratiod_temporal_gp::TemporalCovType::EXPONENTIAL &&
        covers(GF_TEMPORAL_GP))
        return &compute_gradient_temporal_gp_handcoded;
    if (layout.has_multiscale_temporal &&
        covers(GF_AREAL | GF_MS_TEMPORAL))
        return &compute_gradient_ms_temporal_handcoded;
    // compute_gradient_latent_handcoded writes no interaction block, so a model
    // carrying one falls through to the composite. Measured on the stgp_latent
    // fixture before the mask was derived: the latent factors came back at
    // -2.34 against a finite difference of 0.114.
    if (layout.has_latent && data.latent_n_factors > 0 && covers(GF_LATENT))
        return &compute_gradient_latent_handcoded;

    // Composite H-mode: catch-all for exotic multi-feature combinations
    // that no specialized function above handles (e.g., HSGP+TVC, SVC+RW1,
    // latent+spatial, crossed+slopes). Slower than specialized but faster than A_r.
    return &compute_gradient_composite;
}

// The name resolve_gradient_fn() settled on, so that which function a model
// selects is assertable from R rather than inferred from whether its gradient
// happens to come out right.
const char* gradient_fn_name(GradientFn fn) {
  if (fn == &compute_gradient_numerical) return "numerical";
  if (fn == &compute_gradient_autodiff) return "autodiff_tape";
  if (fn == &compute_gradient_arena) return "autodiff_arena";
  if (fn == &compute_gradient_forward) return "autodiff_forward";
  if (fn == &compute_gradient_analytical) return "analytical";
  if (fn == &compute_gradient_hsgp) return "hsgp";
  if (fn == &compute_gradient_gp_collapsed) return "gp_collapsed";
  if (fn == &compute_gradient_icar_collapsed) return "icar_collapsed";
  if (fn == &compute_gradient_gp_handcoded) return "gp";
  if (fn == &compute_gradient_msgp_hsgp) return "msgp_hsgp";
  if (fn == &compute_gradient_msgp_temporal_handcoded) return "msgp_temporal";
  if (fn == &compute_gradient_msgp_handcoded) return "msgp";
  if (fn == &compute_gradient_gp_temporal_handcoded) return "gp_temporal";
  if (fn == &compute_gradient_svc_hsgp_handcoded) return "svc_hsgp";
  if (fn == &compute_gradient_svc_handcoded) return "svc";
  if (fn == &compute_gradient_tvc_handcoded) return "tvc";
  if (fn == &compute_gradient_spatiotemporal_handcoded) return "spatiotemporal";
  if (fn == &compute_gradient_temporal_gp_handcoded) return "temporal_gp";
  if (fn == &compute_gradient_ms_temporal_handcoded) return "ms_temporal";
  if (fn == &compute_gradient_latent_handcoded) return "latent";
  if (fn == &compute_gradient_composite) return "composite";
  return "unknown";
}

// Refuse an autodiff gradient mode on a model the templated log posterior
// cannot express. Called on the R thread before sampling starts, because
// resolve_gradient_fn runs on OpenMP workers where a throw cannot cross the
// region boundary. The H and N modes carry the analytic density and are
// unaffected.
void require_autodiff_supported(const ModelData& data, const ParamLayout& layout) {
  if (g_gradient_mode != GradientMode::AUTODIFF_ARENA &&
      g_gradient_mode != GradientMode::AUTODIFF_FORWARD &&
      g_gradient_mode != GradientMode::AUTODIFF_TAPE) return;

  const char* gap = ratiod::log_post_impl_gap(data, layout);
  if (gap == nullptr) return;

  const char* mode_name =
      (g_gradient_mode == GradientMode::AUTODIFF_ARENA)   ? "A_r" :
      (g_gradient_mode == GradientMode::AUTODIFF_FORWARD) ? "A"   : "A_t";
  Rcpp::stop(std::string("gradient_mode '") + mode_name +
             "' differentiates the templated log posterior, which does not "
             "express the " + gap + " parameterization: that field is "
             "marginalized out by locating its mode and adding a Laplace "
             "correction, which is not a closed-form function of the "
             "parameters. Use gradient_mode = \"H\" or \"N\".");
}

// =====================================================================
// Dual averaging for step size adaptation
// =====================================================================

DualAveraging::DualAveraging(double epsilon_init, int n_params, double target_boost)
  : mu(std::log(10.0 * epsilon_init)), log_epsilon_bar(std::log(epsilon_init)), H_bar(0.0),
    gamma(0.05), t0(10.0), kappa(0.75),
    target_accept(compute_target(n_params, target_boost)), m(0) {}

double DualAveraging::update(double alpha) {
  m++;
  double w = 1.0 / (m + t0);
  H_bar = (1.0 - w) * H_bar + w * (target_accept - alpha);
  double log_epsilon = mu - std::sqrt((double)m) / gamma * H_bar;
  // Clamp log_epsilon to reasonable range
  // Lower bound: exp(-14) ≈ 8e-7, Upper bound: exp(2) ≈ 7.4
  log_epsilon = std::max(-14.0, std::min(log_epsilon, 2.0));
  double epsilon = std::exp(log_epsilon);
  double m_w = std::pow((double)m, -kappa);
  log_epsilon_bar = m_w * log_epsilon + (1.0 - m_w) * log_epsilon_bar;
  return epsilon;
}

double DualAveraging::final_epsilon() const {
  return std::exp(log_epsilon_bar);
}

// =====================================================================
// Welford's online algorithm for mean and variance
// Used for diagonal mass matrix estimation during warmup
// =====================================================================

class WelfordStats {
public:
  int n;
  std::vector<double> mean;
  std::vector<double> M2;  // Sum of squared differences from mean

  WelfordStats(int dim) : n(0), mean(dim, 0.0), M2(dim, 0.0) {}

  void update(const std::vector<double>& x) {
    n++;
    for (size_t i = 0; i < x.size(); i++) {
      double delta = x[i] - mean[i];
      mean[i] += delta / n;
      double delta2 = x[i] - mean[i];
      M2[i] += delta * delta2;
    }
  }

  std::vector<double> variance() const {
    std::vector<double> var(mean.size());
    if (n < 2) {
      // Return unit variance if not enough samples
      std::fill(var.begin(), var.end(), 1.0);
    } else {
      for (size_t i = 0; i < mean.size(); i++) {
        var[i] = M2[i] / (n - 1);
        // Ensure minimum variance to avoid numerical issues
        if (var[i] < 1e-6) var[i] = 1e-6;
      }
    }
    return var;
  }

  // Get inverse mass matrix (= variance, regularized for stability)
  // For HMC: M = diag(1/var), so M^{-1} = diag(var)
  // High variance parameters should move faster in position space
  // Uses Stan-style Bayesian shrinkage toward unit variance to prevent
  // extreme mass matrix entries from small sample sizes
  std::vector<double> inv_mass() const {
    auto var = variance();
    double shrink = (n < 2) ? 0.0 : (double)n / (n + 5.0);
    for (size_t i = 0; i < var.size(); i++) {
      var[i] = shrink * var[i] + 1e-3 * (5.0 / (n + 5.0));
      // Safety clamp for extreme values
      var[i] = std::max(1e-3, std::min(var[i], 1e3));
    }
    return var;
  }

  // Get sqrt of mass matrix for momentum sampling
  // Since M = diag(1/var), sqrt(M) = diag(1/sqrt(var))
  // p ~ N(0, M), so p_i = z_i / sqrt(var_i)
  // Uses the same regularized variance as inv_mass()
  std::vector<double> sqrt_mass() const {
    auto inv_m = inv_mass();
    std::vector<double> sqrt_m(inv_m.size());
    for (size_t i = 0; i < inv_m.size(); i++) {
      sqrt_m[i] = 1.0 / std::sqrt(inv_m[i]);
    }
    return sqrt_m;
  }

  void reset() {
    n = 0;
    std::fill(mean.begin(), mean.end(), 0.0);
    std::fill(M2.begin(), M2.end(), 0.0);
  }
};

// =====================================================================
// Leapfrog integrator
// =====================================================================

LeapfrogResult leapfrog_step(
    const std::vector<double>& q,
    const std::vector<double>& p,
    double epsilon,
    const ModelData& data,
    const ParamLayout& layout
) {
  int n = q.size();
  LeapfrogResult result;
  result.q = q;
  result.p = p;
  result.divergent = false;

  std::vector<double> grad(n);

  // Half step for momentum
  compute_gradient(result.q, data, layout, grad);
  for (int i = 0; i < n; i++) {
    result.p[i] += 0.5 * epsilon * grad[i];
  }

  // Full step for position
  for (int i = 0; i < n; i++) {
    result.q[i] += epsilon * result.p[i];
  }

  // Half step for momentum (fused gradient + log_prob)
  compute_gradient(result.q, data, layout, grad, &result.log_prob);
  for (int i = 0; i < n; i++) {
    result.p[i] += 0.5 * epsilon * grad[i];
  }

  if (!std::isfinite(result.log_prob)) {
    result.divergent = true;
  }

  // Also check for extreme parameter values
  for (int i = 0; i < n; i++) {
    if (std::abs(result.q[i]) > 1e10 || !std::isfinite(result.q[i])) {
      result.divergent = true;
      break;
    }
  }

  return result;
}

// =====================================================================
// Find reasonable initial step size
// =====================================================================

// Compute diagonal mass matrix from gradient magnitudes
// This provides automatic scaling for poorly-conditioned posteriors
std::vector<double> compute_diagonal_mass(
    const std::vector<double>& q,
    const ModelData& data,
    const ParamLayout& layout
) {
  int n = q.size();
  std::vector<double> grad(n);
  compute_gradient(q, data, layout, grad);

  std::vector<double> mass(n);
  for (int i = 0; i < n; i++) {
    // Use gradient magnitude as rough estimate of curvature
    // Mass ~ 1/variance, so larger gradient -> larger mass -> smaller step in that direction
    double abs_grad = std::abs(grad[i]);
    // Clamp to reasonable range [1, 1000]
    mass[i] = std::max(1.0, std::min(abs_grad, 1000.0));
  }

  return mass;
}

// Leapfrog step with diagonal mass matrix
LeapfrogResult leapfrog_step_mass(
    const std::vector<double>& q,
    const std::vector<double>& p,
    double epsilon,
    const std::vector<double>& inv_mass,  // inverse mass (1/M)
    const ModelData& data,
    const ParamLayout& layout
) {
  int n = q.size();
  LeapfrogResult result;
  result.q = q;
  result.p = p;
  result.divergent = false;

  std::vector<double> grad(n);

  // Half step for momentum
  compute_gradient(result.q, data, layout, grad);
  for (int i = 0; i < n; i++) {
    result.p[i] += 0.5 * epsilon * grad[i];
  }

  // Full step for position (scaled by inverse mass)
  for (int i = 0; i < n; i++) {
    result.q[i] += epsilon * inv_mass[i] * result.p[i];
  }

  // Half step for momentum (fused gradient + log_prob)
  compute_gradient(result.q, data, layout, grad, &result.log_prob);
  for (int i = 0; i < n; i++) {
    result.p[i] += 0.5 * epsilon * grad[i];
  }

  if (!std::isfinite(result.log_prob)) {
    result.divergent = true;
  }

  // Check for extreme parameter values
  for (int i = 0; i < n; i++) {
    if (std::abs(result.q[i]) > 1e10 || !std::isfinite(result.q[i])) {
      result.divergent = true;
      break;
    }
  }

  return result;
}

double find_reasonable_epsilon(
    const std::vector<double>& q,
    const ModelData& data,
    const ParamLayout& layout,
    std::mt19937& rng
) {
  // Stan-style algorithm: start at epsilon=1, double or halve until
  // acceptance probability crosses 0.5
  int n = q.size();

  std::normal_distribution<double> normal(0.0, 1.0);
  std::vector<double> p(n);
  for (int i = 0; i < n; i++) {
    p[i] = normal(rng);
  }

  // Fused: compute gradient + log_post in a single O(N) pass
  // (eliminates redundant compute_log_post call)
  double log_prob_init;
  std::vector<double> grad_init(n);
  compute_gradient(q, data, layout, grad_init, &log_prob_init);
  double kinetic_init = 0.5 * ratiod_linalg::norm_squared(p.data(), n);
  double H_init = -log_prob_init + kinetic_init;

  double epsilon = 1.0;

  LeapfrogResult lf = leapfrog_step(q, p, epsilon, data, layout);
  double kinetic_new = 0.5 * ratiod_linalg::norm_squared(lf.p.data(), n);
  double H_new = -lf.log_prob + kinetic_new;
  double delta_H = H_new - H_init;

  // Determine direction: if accept prob > 0.5, increase; else decrease
  // accept_prob = exp(-delta_H), so > 0.5 iff delta_H < log(2)
  int direction = (!std::isfinite(delta_H) || delta_H > std::log(2.0)) ? -1 : 1;

  for (int iter = 0; iter < 50; iter++) {
    if (direction == 1) {
      epsilon *= 2.0;
    } else {
      epsilon *= 0.5;
    }

    if (epsilon < 1e-10 || epsilon > 1e5) break;

    lf = leapfrog_step(q, p, epsilon, data, layout);
    if (!std::isfinite(lf.log_prob)) {
      if (direction == 1) break;  // Was increasing, hit instability
      continue;  // Was decreasing, keep going
    }

    kinetic_new = 0.5 * ratiod_linalg::norm_squared(lf.p.data(), n);
    H_new = -lf.log_prob + kinetic_new;
    delta_H = H_new - H_init;

    // Stop when we cross the 0.5 acceptance threshold
    if (direction == 1 && (!std::isfinite(delta_H) || delta_H > std::log(2.0))) break;
    if (direction == -1 && std::isfinite(delta_H) && delta_H < std::log(2.0)) break;
  }

  return std::max(1e-10, std::min(epsilon, 1e3));
}

// Mass-aware version: uses diagonal mass matrix for leapfrog and kinetic energy
double find_reasonable_epsilon(
    const std::vector<double>& q,
    const ModelData& data,
    const ParamLayout& layout,
    std::mt19937& rng,
    const std::vector<double>& inv_mass
) {
  int n = q.size();

  std::normal_distribution<double> normal(0.0, 1.0);
  std::vector<double> p(n);
  // Sample p ~ N(0, M) where M = diag(1/inv_mass)
  for (int i = 0; i < n; i++) {
    p[i] = normal(rng) / std::sqrt(inv_mass[i]);
  }

  // Fused: compute gradient + log_post in a single O(N) pass
  double log_prob_init;
  std::vector<double> grad_init(n);
  compute_gradient(q, data, layout, grad_init, &log_prob_init);
  double kinetic_init = 0.0;
  for (int i = 0; i < n; i++) {
    kinetic_init += p[i] * p[i] * inv_mass[i];
  }
  kinetic_init *= 0.5;
  double H_init = -log_prob_init + kinetic_init;

  double epsilon = 1.0;

  LeapfrogResult lf = leapfrog_step_mass(q, p, epsilon, inv_mass, data, layout);
  double kinetic_new = 0.0;
  for (int i = 0; i < n; i++) {
    kinetic_new += lf.p[i] * lf.p[i] * inv_mass[i];
  }
  kinetic_new *= 0.5;
  double H_new = -lf.log_prob + kinetic_new;
  double delta_H = H_new - H_init;

  int direction = (!std::isfinite(delta_H) || delta_H > std::log(2.0)) ? -1 : 1;

  for (int iter = 0; iter < 50; iter++) {
    if (direction == 1) {
      epsilon *= 2.0;
    } else {
      epsilon *= 0.5;
    }

    if (epsilon < 1e-10 || epsilon > 1e5) break;

    lf = leapfrog_step_mass(q, p, epsilon, inv_mass, data, layout);
    if (!std::isfinite(lf.log_prob)) {
      if (direction == 1) break;
      continue;
    }

    kinetic_new = 0.0;
    for (int i = 0; i < n; i++) {
      kinetic_new += lf.p[i] * lf.p[i] * inv_mass[i];
    }
    kinetic_new *= 0.5;
    H_new = -lf.log_prob + kinetic_new;
    delta_H = H_new - H_init;

    if (direction == 1 && (!std::isfinite(delta_H) || delta_H > std::log(2.0))) break;
    if (direction == -1 && std::isfinite(delta_H) && delta_H < std::log(2.0)) break;
  }

  return std::max(1e-10, std::min(epsilon, 1e3));
}

// Dense-mass-aware version: uses full DenseMassMatrix for momentum sampling,
// leapfrog integration, and kinetic energy computation.
// This ensures the step size is calibrated for the rotated phase space
// dynamics of the dense mass matrix, not just the diagonal approximation.
double find_reasonable_epsilon_dense(
    const std::vector<double>& q,
    const ModelData& data,
    const ParamLayout& layout,
    std::mt19937& rng,
    const DenseMassMatrix& mass
) {
  int n = q.size();

  // Sample momentum using the full dense mass matrix: p ~ N(0, M)
  std::vector<double> p(n);
  // const_cast is safe here - sample_momentum only reads rng state
  const_cast<DenseMassMatrix&>(mass).sample_momentum(p.data(), rng);

  // Compute gradient + log_post
  double log_prob_init;
  std::vector<double> grad_init(n);
  compute_gradient(q, data, layout, grad_init, &log_prob_init);

  // Kinetic energy using full dense mass matrix
  double kinetic_init = mass.kinetic_energy(p.data());
  double H_init = -log_prob_init + kinetic_init;

  double epsilon = 1.0;

  // Single leapfrog step using dense mass matrix
  auto do_leapfrog = [&](const std::vector<double>& q_in,
                         const std::vector<double>& p_in,
                         double eps) -> std::pair<std::vector<double>, std::vector<double>> {
    std::vector<double> q_out = q_in;
    std::vector<double> p_out = p_in;
    std::vector<double> grad(n);
    compute_gradient(q_out, data, layout, grad);

    // Half step for momentum
    for (int i = 0; i < n; i++) p_out[i] += 0.5 * eps * grad[i];

    // Full step for position: q += eps * M^{-1} * p
    std::vector<double> Mp(n);
    mass.inv_mass_times_p(p_out.data(), Mp.data());
    for (int i = 0; i < n; i++) q_out[i] += eps * Mp[i];

    // Compute gradient at new position
    compute_gradient(q_out, data, layout, grad);

    // Half step for momentum
    for (int i = 0; i < n; i++) p_out[i] += 0.5 * eps * grad[i];

    return {q_out, p_out};
  };

  auto [q_new, p_new] = do_leapfrog(q, p, epsilon);

  // Check if log_prob is finite at new position
  double log_prob_new;
  std::vector<double> grad_new(n);
  compute_gradient(q_new, data, layout, grad_new, &log_prob_new);

  double kinetic_new = mass.kinetic_energy(p_new.data());
  double H_new = -log_prob_new + kinetic_new;
  double delta_H = H_new - H_init;

  int direction = (!std::isfinite(delta_H) || delta_H > std::log(2.0)) ? -1 : 1;

  for (int iter = 0; iter < 50; iter++) {
    if (direction == 1) {
      epsilon *= 2.0;
    } else {
      epsilon *= 0.5;
    }

    if (epsilon < 1e-10 || epsilon > 1e5) break;

    auto [q_try, p_try] = do_leapfrog(q, p, epsilon);

    double lp_try;
    std::vector<double> grad_try(n);
    compute_gradient(q_try, data, layout, grad_try, &lp_try);

    if (!std::isfinite(lp_try)) {
      if (direction == 1) break;
      continue;
    }

    kinetic_new = mass.kinetic_energy(p_try.data());
    H_new = -lp_try + kinetic_new;
    delta_H = H_new - H_init;

    if (direction == 1 && (!std::isfinite(delta_H) || delta_H > std::log(2.0))) break;
    if (direction == -1 && std::isfinite(delta_H) && delta_H < std::log(2.0)) break;
  }

  return std::max(1e-10, std::min(epsilon, 1e3));
}

// =====================================================================
// NUTS (No-U-Turn Sampler) helper functions
// =====================================================================

double nuts_log_sum_exp(double a, double b) {
  double m = std::max(a, b);
  if (!std::isfinite(m)) return m;
  return m + std::log(std::exp(a - m) + std::exp(b - m));
}

double nuts_compute_hamiltonian(double log_prob, const std::vector<double>& p,
                                const std::vector<double>& inv_mass, int n) {
  double kinetic = 0.0;
  for (int i = 0; i < n; i++) {
    kinetic += p[i] * p[i] * inv_mass[i];
  }
  return -log_prob + 0.5 * kinetic;
}

bool nuts_check_uturn(const std::vector<double>& q_minus, const std::vector<double>& q_plus,
                      const std::vector<double>& p_minus, const std::vector<double>& p_plus,
                      const std::vector<double>& inv_mass, int n) {
  // Generalized U-turn criterion (Betancourt 2017, Section 3.2)
  // Check both directions: (q+ - q-) . (M^-1 p-) and (q+ - q-) . (M^-1 p+)
  double dot_fwd = 0.0, dot_bwd = 0.0;
  for (int i = 0; i < n; i++) {
    double dq = q_plus[i] - q_minus[i];
    dot_fwd += dq * (inv_mass[i] * p_plus[i]);
    dot_bwd += dq * (inv_mass[i] * p_minus[i]);
  }
  return (dot_fwd < 0.0) || (dot_bwd < 0.0);
}

LeapfrogResultWithGrad leapfrog_step_with_grad(
    const std::vector<double>& q, const std::vector<double>& p,
    const std::vector<double>& grad,
    double epsilon, const std::vector<double>& inv_mass,
    bool use_mass, const ModelData& data, const ParamLayout& layout) {

  int n = q.size();
  LeapfrogResultWithGrad result;
  result.q = q;
  result.p = p;
  result.grad.resize(n);
  result.divergent = false;

  // Half step for momentum using provided gradient
  for (int i = 0; i < n; i++) {
    result.p[i] += 0.5 * epsilon * grad[i];
  }

  // Full step for position
  if (use_mass) {
    for (int i = 0; i < n; i++) {
      result.q[i] += epsilon * inv_mass[i] * result.p[i];
    }
  } else {
    for (int i = 0; i < n; i++) {
      result.q[i] += epsilon * result.p[i];
    }
  }

  // Compute gradient and log_prob at new position (fused: single O(N) pass)
  compute_gradient(result.q, data, layout, result.grad, &result.log_prob);

  // Half step for momentum using new gradient
  for (int i = 0; i < n; i++) {
    result.p[i] += 0.5 * epsilon * result.grad[i];
  }

  // Check for divergence
  if (!std::isfinite(result.log_prob)) {
    result.divergent = true;
  }
  for (int i = 0; i < n; i++) {
    if (std::abs(result.q[i]) > 1e10 || !std::isfinite(result.q[i])) {
      result.divergent = true;
      break;
    }
  }

  return result;
}

NUTSTreeResult build_tree(const NUTSNode& node, int direction, int depth,
                          double epsilon, const std::vector<double>& inv_mass,
                          bool use_mass, double H0, double delta_max,
                          const ModelData& data, const ParamLayout& layout,
                          std::mt19937& rng) {
  int n = node.q.size();
  NUTSTreeResult result;

  if (depth == 0) {
    // Base case: single leapfrog step
    LeapfrogResultWithGrad lf = leapfrog_step_with_grad(
      node.q, node.p, node.grad,
      direction * epsilon, inv_mass, use_mass, data, layout
    );

    double H_new = nuts_compute_hamiltonian(lf.log_prob, lf.p, inv_mass, n);
    double delta_H = H_new - H0;

    result.left.q = lf.q;
    result.left.p = lf.p;
    result.left.grad = lf.grad;
    result.left.log_prob = lf.log_prob;
    result.right = result.left;

    result.q_proposal = std::move(lf.q);
    result.grad_proposal = std::move(lf.grad);
    result.log_prob_proposal = lf.log_prob;

    // Multinomial weight: log(weight) = -H_new (proportional to exp(-H))
    result.sum_log_weight = -H_new;

    // Divergence check
    result.divergent = lf.divergent || (delta_H > delta_max);
    result.stop = result.divergent;

    // Valid if not divergent
    result.n_valid = result.divergent ? 0 : 1;

    // Acceptance statistic: min(1, exp(-delta_H))
    double accept_stat = std::min(1.0, std::exp(-delta_H));
    if (!std::isfinite(accept_stat)) accept_stat = 0.0;
    result.sum_accept_prob = accept_stat;
    result.n_leapfrog = 1;

    return result;
  }

  // Recursive case: build inner subtree
  NUTSTreeResult inner = build_tree(node, direction, depth - 1,
                                     epsilon, inv_mass, use_mass, H0, delta_max,
                                     data, layout, rng);
  result = std::move(inner);

  if (result.stop) return result;

  // Build outer subtree from the appropriate endpoint
  // (read from result, since inner is moved-from)
  NUTSNode outer_start;
  if (direction == 1) {
    outer_start = result.right;
  } else {
    outer_start = result.left;
  }

  NUTSTreeResult outer = build_tree(outer_start, direction, depth - 1,
                                     epsilon, inv_mass, use_mass, H0, delta_max,
                                     data, layout, rng);

  // Combine results (use result.xxx since inner was moved into result)
  result.n_leapfrog += outer.n_leapfrog;
  result.sum_accept_prob += outer.sum_accept_prob;
  result.divergent = result.divergent || outer.divergent;

  // Multinomial sampling: accept outer proposal with probability
  // exp(outer.sum_log_weight) / exp(log_sum_exp(result.sum_log_weight, outer.sum_log_weight))
  double new_sum_log_weight = nuts_log_sum_exp(result.sum_log_weight, outer.sum_log_weight);
  double accept_prob_outer = std::exp(outer.sum_log_weight - new_sum_log_weight);
  if (!std::isfinite(accept_prob_outer)) accept_prob_outer = 0.0;

  std::uniform_real_distribution<double> unif(0.0, 1.0);
  if (unif(rng) < accept_prob_outer) {
    result.q_proposal = std::move(outer.q_proposal);
    result.grad_proposal = std::move(outer.grad_proposal);
    result.log_prob_proposal = outer.log_prob_proposal;
  }
  // else keep inner proposal (already in result from move)

  result.sum_log_weight = new_sum_log_weight;
  result.n_valid = result.n_valid + outer.n_valid;

  // Update tree endpoints
  // result already holds inner's data from the move above
  if (direction == 1) {
    // result.left is already inner.left (correct from move)
    result.right = std::move(outer.right);
  } else {
    result.left = std::move(outer.left);
    // result.right is already inner.right (correct from move)
  }

  // Check U-turn on the combined trajectory
  result.stop = outer.stop ||
    nuts_check_uturn(result.left.q, result.right.q,
                     result.left.p, result.right.p,
                     inv_mass, n);

  return result;
}

// =====================================================================
// Optimized NUTS: zero-allocation infrastructure
// =====================================================================

// Pointer-based Hamiltonian (avoids std::vector overhead)
double nuts_compute_hamiltonian_fast(double log_prob, const double* p,
                                     const DenseMassMatrix& mass, int n) {
  return -log_prob + mass.kinetic_energy(p);
}

// Pointer-based U-turn check
// scratch: temporary buffer of size n (for dense matvec result)
bool nuts_check_uturn_fast(const double* q_minus, const double* q_plus,
                           const double* p_minus, const double* p_plus,
                           const DenseMassMatrix& mass, double* scratch, int n) {
  if (mass.type == MassMatrixType::BLOCK_DIAG && mass.adapted) {
    // Block-diagonal: use inv_mass_times_p which handles blocks correctly
    mass.inv_mass_times_p(p_plus, scratch);
    double dot_fwd = 0.0;
    for (int i = 0; i < n; i++) {
      dot_fwd += (q_plus[i] - q_minus[i]) * scratch[i];
    }
    mass.inv_mass_times_p(p_minus, scratch);
    double dot_bwd = 0.0;
    for (int i = 0; i < n; i++) {
      dot_bwd += (q_plus[i] - q_minus[i]) * scratch[i];
    }
    return (dot_fwd < 0.0) || (dot_bwd < 0.0);
  } else if (mass.type == MassMatrixType::DIAG || !mass.adapted) {
    // Diagonal path (fast, unrolled)
    double dot_fwd = 0.0, dot_bwd = 0.0;
    const double* inv_mass = mass.inv_mass_diag.data();
    int i = 0;
    for (; i + 3 < n; i += 4) {
      double dq0 = q_plus[i]   - q_minus[i];
      double dq1 = q_plus[i+1] - q_minus[i+1];
      double dq2 = q_plus[i+2] - q_minus[i+2];
      double dq3 = q_plus[i+3] - q_minus[i+3];
      dot_fwd += dq0 * (inv_mass[i]   * p_plus[i])
               + dq1 * (inv_mass[i+1] * p_plus[i+1])
               + dq2 * (inv_mass[i+2] * p_plus[i+2])
               + dq3 * (inv_mass[i+3] * p_plus[i+3]);
      dot_bwd += dq0 * (inv_mass[i]   * p_minus[i])
               + dq1 * (inv_mass[i+1] * p_minus[i+1])
               + dq2 * (inv_mass[i+2] * p_minus[i+2])
               + dq3 * (inv_mass[i+3] * p_minus[i+3]);
    }
    for (; i < n; i++) {
      double dq = q_plus[i] - q_minus[i];
      dot_fwd += dq * (inv_mass[i] * p_plus[i]);
      dot_bwd += dq * (inv_mass[i] * p_minus[i]);
    }
    return (dot_fwd < 0.0) || (dot_bwd < 0.0);
  } else {
    // Dense path: compute (q+ - q-) . (C * p+) and (q+ - q-) . (C * p-)
    // Use scratch for C * p
    mass.inv_mass_times_p(p_plus, scratch);
    double dot_fwd = 0.0;
    for (int i = 0; i < n; i++) {
      dot_fwd += (q_plus[i] - q_minus[i]) * scratch[i];
    }
    mass.inv_mass_times_p(p_minus, scratch);
    double dot_bwd = 0.0;
    for (int i = 0; i < n; i++) {
      dot_bwd += (q_plus[i] - q_minus[i]) * scratch[i];
    }
    return (dot_fwd < 0.0) || (dot_bwd < 0.0);
  }
}

// In-place leapfrog step operating on a workspace slot
// Mutates q, p, grad in the slot directly — zero heap allocation
LeapfrogInPlaceResult leapfrog_step_inplace(
    NUTSWorkspace& ws, int slot, double epsilon,
    const DenseMassMatrix& mass,
    const ModelData& data, const ParamLayout& layout) {

  double* q = ws.q_at(slot);
  double* p = ws.p_at(slot);
  double* grad = ws.grad_at(slot);
  int n = ws.n;

  LeapfrogInPlaceResult result;
  result.divergent = false;

  // Half step for momentum using current gradient
  ratiod_linalg::axpy(0.5 * epsilon, grad, p, n);

  // Full step for position: q += eps * C * p
  if (!mass.adapted) {
    // Identity mass: q += eps * p
    ratiod_linalg::axpy(epsilon, p, q, n);
  } else if (mass.type == MassMatrixType::BLOCK_DIAG) {
    // Block-diagonal: diagonal for non-block params, dense for block params
    // First pass: diagonal for all params
    ratiod_linalg::axpy_weighted(epsilon, mass.inv_mass_diag.data(), p, q, n);
    // Second pass: overwrite block params with dense contribution
    for (const auto& blk : mass.blocks) {
      if (blk.adapted) {
        double tmp[4];
        blk.matvec(p, tmp);
        for (int i = 0; i < blk.size; i++) {
          // Undo diagonal contribution, apply block contribution
          q[blk.start + i] += epsilon * (tmp[i] - mass.inv_mass_diag[blk.start + i] * p[blk.start + i]);
        }
      }
    }
  } else if (mass.type == MassMatrixType::DIAG) {
    // Diagonal: q[i] += eps * inv_mass[i] * p[i]
    ratiod_linalg::axpy_weighted(epsilon, mass.inv_mass_diag.data(), p, q, n);
  } else {
    // Dense: q += eps * C * p  (Eigen BLAS for n>=16, scalar fallback below)
    if (n >= 16) {
      Eigen::Map<const Eigen::MatrixXd> Am(mass.inv_mass_dense.data(), n, n);
      Eigen::Map<const Eigen::VectorXd> pv(p, n);
      Eigen::Map<Eigen::VectorXd> qv(q, n);
      qv.noalias() += epsilon * (Am.selfadjointView<Eigen::Lower>() * pv);
    } else {
      ratiod_linalg::axpy_matvec(epsilon, mass.inv_mass_dense.data(), p, q, n);
    }
  }
  // Precision/Kronecker blocks: undo base contribution, apply block M^{-1}
  if (mass.precision_block.active) {
    const auto& pb = mass.precision_block;
    std::vector<double> tmp(pb.size);
    pb.matvec(p, tmp.data());
    for (int i = 0; i < pb.size; i++) {
      // Undo the diagonal (or dense) contribution that was already applied
      q[pb.start + i] -= epsilon * mass.inv_mass_diag[pb.start + i] * p[pb.start + i];
      // Apply precision block contribution
      q[pb.start + i] += epsilon * tmp[i];
    }
  }
  if (mass.kronecker_block.active) {
    const auto& kb = mass.kronecker_block;
    int ST = kb.S * kb.T;
    std::vector<double> tmp(ST);
    kb.matvec(p, tmp.data());
    for (int i = 0; i < ST; i++) {
      q[kb.start + i] -= epsilon * mass.inv_mass_diag[kb.start + i] * p[kb.start + i];
      q[kb.start + i] += epsilon * tmp[i];
    }
  }

  // Compute gradient + log_prob at new position (fused: single O(N) pass)
  // Uses pre-resolved function pointer to skip 15+ branch dispatch per leapfrog step
  std::memcpy(ws.params_buf.data(), q, n * sizeof(double));
  ws.gradient_fn(ws.params_buf, data, layout, ws.grad_buf, &ws.logp_at(slot));
  std::memcpy(grad, ws.grad_buf.data(), n * sizeof(double));
  result.log_prob = ws.logp_at(slot);

  // Half step for momentum using new gradient
  ratiod_linalg::axpy(0.5 * epsilon, grad, p, n);

  // Divergence check (skip param scan if log_prob already non-finite)
  if (!std::isfinite(result.log_prob)) {
    result.divergent = true;
  } else {
    for (int i = 0; i < n; i++) {
      if (std::abs(q[i]) > 1e10 || !std::isfinite(q[i])) {
        result.divergent = true;
        break;
      }
    }
  }

  return result;
}

// Zero-allocation recursive tree builder
// Uses workspace slot indices instead of vector copies
TreeStats build_tree_fast(
    NUTSWorkspace& ws, int input_slot, int direction, int depth,
    double epsilon, const DenseMassMatrix& mass,
    double H0, double delta_max,
    const ModelData& data, const ParamLayout& layout,
    std::mt19937& rng) {

  int n = ws.n;
  TreeStats stats;

  if (depth == 0) {
    stats.init_vectors(n);  // Pre-allocate U-turn vectors (avoids per-leaf heap allocation)
    // Base case: single leapfrog step in-place on input_slot
    LeapfrogInPlaceResult lf = leapfrog_step_inplace(
      ws, input_slot, direction * epsilon, mass, data, layout
    );

    double H_new = nuts_compute_hamiltonian_fast(
      lf.log_prob, ws.p_at(input_slot), mass, n
    );
    double delta_H = H_new - H0;

    // Both endpoints are the same slot (single node)
    stats.left_slot = input_slot;
    stats.right_slot = input_slot;
    stats.proposal_slot = input_slot;
    stats.log_prob_proposal = lf.log_prob;

    // Multinomial weight: log(weight) = H0 - H_new (relative, Stan-style)
    stats.sum_log_weight = H0 - H_new;

    // Divergence check
    stats.divergent = lf.divergent || (delta_H > delta_max);
    stats.stop = stats.divergent;
    stats.n_valid = stats.divergent ? 0 : 1;

    // Acceptance statistic
    double accept_stat = std::min(1.0, std::exp(-delta_H));
    if (!std::isfinite(accept_stat)) accept_stat = 0.0;
    stats.sum_accept_prob = accept_stat;
    stats.n_leapfrog = 1;

    // Generalized U-turn: track rho, p_sharp, p at this leaf
    const double* p_ptr = ws.p_at(input_slot);

    std::memcpy(stats.rho.data(), p_ptr, n * sizeof(double));
    std::memcpy(stats.p_beg.data(), p_ptr, n * sizeof(double));
    std::memcpy(stats.p_end.data(), p_ptr, n * sizeof(double));

    // p_sharp = M^{-1} * p  — use full mass matrix for U-turn criterion.
    // Dense mass captures correlation structure; using diagonal p_sharp would
    // make NUTS unable to detect turns in correlated directions, causing
    // trees to grow to max depth on correlated posteriors (slopes, BYM2, HSGP).
    mass.inv_mass_times_p(p_ptr, stats.p_sharp_beg.data());
    std::memcpy(stats.p_sharp_end.data(), stats.p_sharp_beg.data(), n * sizeof(double));

    return stats;
  }

  // Recursive case: build inner subtree
  TreeStats inner = build_tree_fast(
    ws, input_slot, direction, depth - 1,
    epsilon, mass, H0, delta_max, data, layout, rng
  );

  stats = std::move(inner);

  if (stats.stop) return stats;

  // Copy the appropriate endpoint to a fresh slot for outer start
  int start_slot = ws.alloc_slot();
  if (start_slot < 0) {
    stats.stop = true;
    return stats;
  }
  if (direction == 1) {
    ws.copy_node(start_slot, stats.right_slot);
  } else {
    ws.copy_node(start_slot, stats.left_slot);
  }

  // Build outer subtree from the copy
  TreeStats outer = build_tree_fast(
    ws, start_slot, direction, depth - 1,
    epsilon, mass, H0, delta_max, data, layout, rng
  );

  // Combine results
  stats.n_leapfrog += outer.n_leapfrog;
  stats.sum_accept_prob += outer.sum_accept_prob;
  stats.divergent = stats.divergent || outer.divergent;

  // Multinomial sampling
  double new_sum_log_weight = nuts_log_sum_exp(stats.sum_log_weight, outer.sum_log_weight);
  double accept_prob_outer = std::exp(outer.sum_log_weight - new_sum_log_weight);
  if (!std::isfinite(accept_prob_outer)) accept_prob_outer = 0.0;

  std::uniform_real_distribution<double> unif(0.0, 1.0);
  if (unif(rng) < accept_prob_outer) {
    stats.proposal_slot = outer.proposal_slot;
    stats.log_prob_proposal = outer.log_prob_proposal;
  }

  stats.sum_log_weight = new_sum_log_weight;
  stats.n_valid = stats.n_valid + outer.n_valid;

  // === SAVE BOUNDARY VALUES AS COPIES BEFORE MOVES ===
  // "init" = inner (built first), "final" = outer (extends from init)
  // These copies are needed because moves below invalidate the originals
  // Uses pre-allocated depth-indexed merge buffers (no per-merge heap allocation)
  double* p_init_end = ws.merge_buf(depth, NUTSWorkspace::MERGE_P_INIT_END);
  double* p_sharp_init_end = ws.merge_buf(depth, NUTSWorkspace::MERGE_PSHARP_INIT_END);
  double* rho_init = ws.merge_buf(depth, NUTSWorkspace::MERGE_RHO_INIT);
  double* rho_check = ws.merge_buf(depth, NUTSWorkspace::MERGE_RHO_CHECK);

  const double* src_p = (direction == 1) ? stats.p_end.data() : stats.p_beg.data();
  std::memcpy(p_init_end, src_p, n * sizeof(double));
  const double* src_ps = (direction == 1) ? stats.p_sharp_end.data() : stats.p_sharp_beg.data();
  std::memcpy(p_sharp_init_end, src_ps, n * sizeof(double));
  std::memcpy(rho_init, stats.rho.data(), n * sizeof(double));

  // Pointers to final's boundary (safe: these outer members are NOT moved)
  const double* p_final_beg_ptr = (direction == 1) ? outer.p_beg.data() : outer.p_end.data();
  const double* p_sharp_final_beg_ptr = (direction == 1) ? outer.p_sharp_beg.data() : outer.p_sharp_end.data();

  // === UPDATE TREE ENDPOINTS (moves invalidate init's boundary refs) ===
  if (direction == 1) {
    stats.right_slot = outer.right_slot;
    stats.p_sharp_end = std::move(outer.p_sharp_end);
    stats.p_end = std::move(outer.p_end);
  } else {
    stats.left_slot = outer.left_slot;
    stats.p_sharp_beg = std::move(outer.p_sharp_beg);
    stats.p_beg = std::move(outer.p_beg);
  }

  // Combine rho = rho_init + rho_final
  for (int i = 0; i < n; i++) {
    stats.rho[i] = rho_init[i] + outer.rho[i];
  }

  // === GENERALIZED U-TURN CRITERION (Stan-style, 3 juncture checks) ===
  // Check 1: Full merged trajectory — merged endpoints vs merged rho
  bool persist = compute_criterion(stats.p_sharp_beg.data(), stats.p_sharp_end.data(),
                                   stats.rho.data(), n);

  // After update, far endpoints depend on direction:
  // direction == 1: init's far = stats.beg (left, unchanged), final's far = stats.end (right, updated)
  // direction == -1: init's far = stats.end (right, unchanged), final's far = stats.beg (left, updated)
  const double* init_far_psharp = (direction == 1) ? stats.p_sharp_beg.data() : stats.p_sharp_end.data();
  const double* final_far_psharp = (direction == 1) ? stats.p_sharp_end.data() : stats.p_sharp_beg.data();

  // Check 2: Init subtree + seam from final (rho = rho_init + p_final_beg)
  // Fused: rho construction + dot product in single O(n) pass
  persist &= compute_criterion_fused(init_far_psharp, p_sharp_final_beg_ptr,
                                     rho_init, p_final_beg_ptr, rho_check, n);

  // Check 3: Seam from init + final subtree (rho = rho_final + p_init_end)
  persist &= compute_criterion_fused(p_sharp_init_end, final_far_psharp,
                                     outer.rho.data(), p_init_end, rho_check, n);

  stats.stop = outer.stop || !persist;

  return stats;
}

// =====================================================================
// The WALNUTS model over this sampler's metric
// =====================================================================

// What tulpa::walnuts_transition() reads of a chain: the dispatched gradient,
// and the kinetic energy, metric product and momentum draw of the adapted
// mass matrix, with the drift theta += step * M^-1 rho through the same
// product the NUTS U-turn criterion uses.
struct RatioWalnutsModel {
  const DenseMassMatrix& mass;
  GradientFn gradient_fn;
  const ModelData& data;
  const ParamLayout& layout;
  int n;

  void gradient(const std::vector<double>& theta, std::vector<double>& grad,
                double* log_density) {
    gradient_fn(theta, data, layout, grad, log_density);
  }
  double kinetic_energy(const double* rho) const {
    return mass.kinetic_energy(rho);
  }
  void inv_mass_times_p(const double* rho, double* out) const {
    mass.inv_mass_times_p(rho, out);
  }
  void drift(double step, double* theta, const double* rho,
             double* scratch) const {
    mass.inv_mass_times_p(rho, scratch);
    for (int i = 0; i < n; i++) theta[i] += step * scratch[i];
  }
  template <class Rng>
  void sample_momentum(double* rho, Rng& rng) const {
    mass.sample_momentum(rho, rng);
  }
};

// =====================================================================
// Run single HMC chain
// =====================================================================

// Pure C++ version - safe for OpenMP parallel regions
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
    int max_treedepth,
    MassMatrixType metric_type,
    double adapt_delta,
    const tulpa::WalnutsConfig* walnuts
) {
  int n_params = q_init.size();
  int n_sample = n_iter - n_warmup;
  bool use_nuts = (L == 0);

  HMCResultCpp result;
  result.n_params_stored = n_params;
  result.samples_flat.resize(static_cast<size_t>(n_sample) * n_params);
  result.log_prob.resize(n_sample);
  result.accept_prob.resize(n_sample);
  result.n_leapfrog.resize(n_sample, L);
  result.divergent.resize(n_sample, 0);
  result.treedepth.resize(n_sample, 0);
  result.n_warmup = n_warmup;
  result.n_sample = n_sample;
  result.chain_id = chain_id;
  result.n_max_treedepth = 0;

  // Collapsed GP: allocate w* storage
  if (data.gp_collapsed && data.has_gp) {
      result.n_gp_collapsed = data.gp_data.n_obs;
      result.gp_w_star_flat.resize(static_cast<size_t>(n_sample) * data.gp_data.n_obs, 0.0);
  }

  // Collapsed ICAR/BYM2: allocate phi*/theta* storage
  if (data.icar_collapsed || data.bym2_collapsed) {
      int S = data.n_spatial_units;
      result.n_icar_collapsed = S;
      result.icar_phi_star_flat.resize(static_cast<size_t>(n_sample) * S, 0.0);
      if (data.bym2_collapsed) {
          result.bym2_theta_star_flat.resize(static_cast<size_t>(n_sample) * S, 0.0);
      }
  }

  std::mt19937 rng(seed + chain_id * 12345);
  std::normal_distribution<double> normal(0.0, 1.0);
  std::uniform_real_distribution<double> unif(0.0, 1.0);

  // Reset VecGradWorkspace cache for new model fit
  vec_grad_ws().cached_data_id = 0;

  std::vector<double> q = q_init;

  // For NUTS: fuse initial log_post + gradient into single O(N) pass
  std::vector<double> current_grad(n_params);
  double log_prob_current;
  if (use_nuts) {
    compute_gradient(q, data, layout, current_grad, &log_prob_current);
  } else {
    // Use same log-post function as the active gradient mode
    if (g_gradient_mode == GradientMode::AUTODIFF_ARENA ||
        g_gradient_mode == GradientMode::AUTODIFF_FORWARD ||
        g_gradient_mode == GradientMode::AUTODIFF_TAPE) {
      log_prob_current = ratiod::compute_log_post_impl(q, data, layout);
    } else {
      log_prob_current = compute_log_post(q, data, layout);
    }
  }

  double epsilon = find_reasonable_epsilon(q, data, layout, rng);

  // Compute target_boost for challenging model combinations
  // MSGP and GP with temporal are particularly challenging
  double target_boost = 0.0;
  if (data.has_multiscale_gp) {
    target_boost += 0.10;  // MSGP models need higher target acceptance
    if (layout.has_temporal) {
      target_boost += 0.05;  // MSGP + temporal is even more challenging
    }
  } else if (data.spatial_type == SpatialType::GP) {
    target_boost += 0.05;  // GP models moderately challenging
    if (layout.has_temporal) {
      target_boost += 0.05;  // GP + temporal combination
    }
  }
  DualAveraging da(epsilon, n_params, target_boost);

  // For NUTS: model-adaptive target acceptance
  // Store in nuts_target_accept for reuse at mass window boundaries (avoids bug
  // where da.target_accept was reset to 0.80 at each window reset).
  double nuts_target_accept = 0.80;
  if (use_nuts) {
    if (adapt_delta > 0) {
      // User override
      nuts_target_accept = adapt_delta;
    } else {
      // Auto-select based on model complexity
      nuts_target_accept = 0.80;  // Stan default base

      // BYM2: high correlation between ICAR phi + unstructured theta
      if (data.spatial_type == SpatialType::BYM2) {
        nuts_target_accept = 0.90;
      }
      // ICAR: correlated spatial params need slightly higher target
      else if (data.spatial_type == SpatialType::ICAR) {
        nuts_target_accept = 0.85;
      }

      // Correlated random slopes add funnel geometry
      if (data.has_re_correlated_slopes) {
        nuts_target_accept = std::max(nuts_target_accept, 0.90);
      }

      // Temporal GP NC: z ~ N(0,1) decorrelates parameters, lower target OK
      // Benchmarked: 0.70 gives 20% fewer LF steps and 50% less seed variance
      if (layout.is_temporal_gp && nuts_target_accept > 0.70) {
        nuts_target_accept = 0.70;
      }

      nuts_target_accept = std::min(0.99, nuts_target_accept);
    }
    da.target_accept = nuts_target_accept;
  }

  // current_grad already computed above (fused with log_prob for NUTS)

  // Mass matrix adaptation
  // Resolve AUTO metric: DIAG default with DIAG→DENSE identity recovery.
  //
  // Key insight: adapted DENSE mass (where adapted=true) incurs O(n²) per leapfrog
  // step for matvec/kinetic/p_sharp operations. For n=54 (RE models), this is 22x
  // slower per step than identity mass (adapted=false). DIAG→identity recovery gives
  // fast per-step execution while still finding correct epsilon via dual averaging.
  //
  // Strategy: Start with DIAG for most models. If DIAG fails catastrophically
  // (epsilon > 2.0 at warmup end), recover to DENSE identity (adapted=false).
  // This gives:
  //   - O(n) per-step cost (identity path)
  //   - Correct epsilon from find_reasonable_epsilon
  //   - No divergences (epsilon small enough for the geometry)
  //
  // Only genuinely complex posteriors (correlated slopes, BYM2, GP, SVC) start
  // with DENSE, where the adapted covariance actually helps sampling efficiency
  // enough to justify the O(n²) per-step cost.
  //
  // HISTORY:
  // 2026-02-27: has_re/has_temporal in needs_dense → 1.25s PG+RE (adapted dense)
  // 2026-02-28: TVC gradient fix → removed TVC
  // 2026-03-03: Removed has_re/has_temporal → DIAG + recovery. PG+RE: 0.8s
  //   (identity dense, 22x faster per step). Adapted dense measured at 17.7s
  //   due to O(n²) per-step cost for n=54.
  constexpr int DENSE_MAX_PARAMS = 200;
  MassMatrixType effective_metric = metric_type;
  bool auto_selected_diag = false;
  // Block specs for BLOCK_DIAG: (start_index, block_size) pairs
  std::vector<std::pair<int,int>> block_specs;

  if (effective_metric == MassMatrixType::AUTO) {
    // Build block_specs from param layout: detect pairs of correlated hyperparameters.
    // Each block captures a small dense correlation (2-4 params) at O(block²) cost,
    // avoiding full O(n²) DENSE while handling the key correlations DIAG misses.
    // NOTE: temporal_gp excluded — DIAG is faster (2.11s vs 2.39s, 5 seeds Bin+GP_t).
    // NOTE: HSGP excluded from AUTO blocks — DIAG is faster for HSGP-only (29k LF
    // vs 39k LF), and HSGP+temporal uses full DENSE (tested BLOCK_DIAG 2026-03-10:
    // worse performance and more divergences than DENSE).
    if (layout.is_bym2 && layout.log_sigma_bym2_idx >= 0 &&
        layout.logit_rho_bym2_idx == layout.log_sigma_bym2_idx + 1) {
      block_specs.push_back({layout.log_sigma_bym2_idx, 2});
    }
    if (layout.is_gp && layout.log_sigma2_gp_idx >= 0 &&
        layout.log_phi_gp_idx == layout.log_sigma2_gp_idx + 1) {
      block_specs.push_back({layout.log_sigma2_gp_idx, 2});
    }
    if (layout.is_multiscale_gp) {
      if (layout.log_sigma2_gp_local_idx >= 0 &&
          layout.log_phi_gp_local_idx == layout.log_sigma2_gp_local_idx + 1) {
        block_specs.push_back({layout.log_sigma2_gp_local_idx, 2});
      }
      if (layout.log_sigma2_gp_regional_idx >= 0 &&
          layout.log_phi_gp_regional_idx == layout.log_sigma2_gp_regional_idx + 1) {
        block_specs.push_back({layout.log_sigma2_gp_regional_idx, 2});
      }
    }
    if (layout.has_svc && layout.log_sigma2_svc_start >= 0 &&
        layout.log_phi_svc_start >= 0) {
      int n_svc = layout.log_sigma2_svc_end - layout.log_sigma2_svc_start;
      for (int t = 0; t < n_svc; t++) {
        int sigma_idx = layout.log_sigma2_svc_start + t;
        int phi_idx = layout.log_phi_svc_start + t;
        if (phi_idx == sigma_idx + n_svc) {
          // SVC sigma2 and phi are in separate contiguous blocks; can't form a 2x2 block
          // unless they're consecutive. Skip non-consecutive pairs.
        }
      }
      // SVC layout: [sigma2_0, sigma2_1, ..., phi_0, phi_1, ...]
      // These aren't consecutive pairs, so we'd need per-SVC blocks of non-contiguous params.
      // For now, only handle n_svc=1 where sigma2 and phi are adjacent:
      if (n_svc == 1 && layout.log_phi_svc_start == layout.log_sigma2_svc_start + 1) {
        block_specs.push_back({layout.log_sigma2_svc_start, 2});
      }
    }
    if (layout.is_st_gp && layout.log_phi_st_space_idx >= 0 &&
        layout.log_phi_st_time_idx == layout.log_phi_st_space_idx + 1) {
      block_specs.push_back({layout.log_phi_st_space_idx, 2});
    }
    if (layout.has_multiscale_temporal) {
      // Multiscale temporal hyperparams form a natural block (3-4 params):
      // log_sigma2_trend, log_sigma2_seasonal, log_sigma2_short [, logit_rho_short]
      // These are correlated but the temporal effects themselves (phi) are not
      // strongly correlated with the hyperparams → BLOCK_DIAG, not full DENSE.
      int ms_block_start = -1;
      int ms_block_size = 0;
      if (layout.log_sigma2_trend_idx >= 0) {
        ms_block_start = layout.log_sigma2_trend_idx;
        ms_block_size = 1;
      }
      if (layout.log_sigma2_seasonal_idx >= 0) {
        if (ms_block_start < 0) ms_block_start = layout.log_sigma2_seasonal_idx;
        ms_block_size++;
      }
      if (layout.log_sigma2_short_idx >= 0) {
        if (ms_block_start < 0) ms_block_start = layout.log_sigma2_short_idx;
        ms_block_size++;
      }
      if (layout.logit_rho_short_idx >= 0) {
        ms_block_size++;
      }
      if (ms_block_start >= 0 && ms_block_size >= 2 && ms_block_size <= 4) {
        block_specs.push_back({ms_block_start, ms_block_size});
      }
    }
    // Correlated slopes: Cholesky params form a natural block
    if (layout.has_re_correlated_slopes) {
      for (size_t t = 0; t < layout.chol_re_start_multi.size(); t++) {
        if (layout.re_correlated_multi[t]) {
          int chol_start = layout.chol_re_start_multi[t];
          int chol_size = layout.chol_re_end_multi[t] - chol_start;
          // Also include the corresponding sigma params
          // For now, just handle the Cholesky block if size <= 4
          if (chol_size >= 2 && chol_size <= 4) {
            block_specs.push_back({chol_start, chol_size});
          }
        }
      }
    }

    // NB phi params: overdispersion params are often correlated
    // A 2×2 block captures their joint curvature cheaply (4 extra multiplies/step)
    if (layout.has_phi_num && layout.has_phi_denom &&
        layout.log_phi_denom_idx == layout.log_phi_num_idx + 1) {
      block_specs.push_back({layout.log_phi_num_idx, 2});
    }

    // First check if model needs full DENSE (before block decision)
    // NB+ICAR: NegBin's digamma curvature creates strong correlations between
    // spatial phi params and overdispersion that BLOCK_DIAG's small blocks can't
    // capture. DENSE mass doubles the step size, cutting treedepth from 8-9 to 5-6,
    // which more than pays for the O(n²) per-step cost at p~108.
    // GP_t: NC z-sigma2-phi funnel creates erratic treedepth (2-10) with DIAG.
    // At p≤50, DENSE overhead is negligible.
    bool is_nb_family = (data.model_type == ModelType::NEGBIN_NEGBIN ||
                         data.model_type == ModelType::NEGBIN_GAMMA);
    bool is_icar = (data.spatial_type == SpatialType::ICAR);
    bool is_binomial_family = (data.model_type == ModelType::BINOMIAL ||
                              data.model_type == ModelType::BETA_BINOMIAL);
    // HSGP+temporal: 36 HSGP basis coefs and 20 temporal effects have complex
    // cross-correlations that DIAG can't handle (106 div) and BLOCK_DIAG misses
    // (16 div, eps~0.006). DENSE with eigenvalue conditioning captures the geometry
    // correctly (0-1 div). Tested BLOCK_DIAG (2026-03-10): 303s/0div PG, 214s/16div NB,
    // 133s/3div Bin — worse than DENSE (211s/3div, 176s/1div, 142s/0div).
    bool hsgp_temporal = layout.is_hsgp && data.has_hsgp && layout.has_temporal;

    bool needs_full_dense = layout.has_latent ||  // N×K latent factors
                            hsgp_temporal ||  // HSGP+temporal cross-correlations
                            (is_nb_family && is_icar && n_params <= DENSE_MAX_PARAMS) ||  // NB+ICAR
                            (is_binomial_family && is_icar && n_params <= DENSE_MAX_PARAMS);  // Bin+ICAR

    // HSGP-only (no temporal): DIAG outperforms BLOCK_DIAG (29k LF/6 div vs 39k LF/15 div).
    // HSGP-only (no temporal): DIAG outperforms BLOCK_DIAG (29k LF/6 div vs
    // 39k LF/15 div). The real correlations are between lengthscale and m^2 basis
    // coefficients, which small blocks can't capture. Block adaptation adds noise.
    // HSGP+temporal uses full DENSE (handled above in needs_full_dense).
    bool prefer_diag = layout.is_hsgp && data.has_hsgp && !layout.has_temporal;

    if (needs_full_dense) {
      effective_metric = MassMatrixType::DENSE;
      block_specs.clear();
      auto_selected_diag = false;
    } else if (!block_specs.empty() && !prefer_diag) {
      // BLOCK_DIAG: captures key correlations without full O(n²)
      effective_metric = MassMatrixType::BLOCK_DIAG;
      auto_selected_diag = false;
    } else {
      // No blocks detected, no DENSE needed — fall back to DIAG
      effective_metric = MassMatrixType::DIAG;
      auto_selected_diag = true;
    }

    if (verbose) {
      REprintf("  [METRIC] auto -> %s (p=%d", metric_name(effective_metric), n_params);
      if (effective_metric == MassMatrixType::BLOCK_DIAG) {
        REprintf(", %d blocks:", (int)block_specs.size());
        for (const auto& bs : block_specs) {
          REprintf(" [%d,%d)", bs.first, bs.first + bs.second);
        }
      }
      if (layout.has_re) REprintf(", re");
      if (layout.has_re_correlated_slopes) REprintf(", correlated_slopes");
      if (layout.has_temporal) REprintf(", temporal");
      if (layout.is_bym2) REprintf(", bym2");
      if (layout.is_hsgp) REprintf(", hsgp");
      if (layout.is_gp) REprintf(", gp");
      if (layout.is_multiscale_gp) REprintf(", msgp");
      if (layout.is_temporal_gp) REprintf(", temporal_gp");
      if (layout.has_multiscale_temporal) REprintf(", ms_temporal");
      if (layout.has_svc) REprintf(", svc");
      if (layout.has_spatiotemporal) REprintf(", spatiotemporal");
      if (layout.has_latent) REprintf(", latent");
      if (layout.has_tvc) REprintf(", tvc");
      REprintf(")\n");
    }
  }
  // Also build block_specs when user explicitly requests BLOCK_DIAG
  if (effective_metric == MassMatrixType::BLOCK_DIAG && block_specs.empty()) {
    // User forced block_diag but AUTO didn't run — detect blocks from layout
    if (layout.is_temporal_gp && layout.log_sigma2_temporal_gp_idx >= 0 &&
        layout.logit_phi_temporal_gp_idx == layout.log_sigma2_temporal_gp_idx + 1) {
      block_specs.push_back({layout.log_sigma2_temporal_gp_idx, 2});
    }
    if (layout.is_hsgp && layout.log_sigma2_hsgp_idx >= 0 &&
        layout.log_lengthscale_hsgp_idx == layout.log_sigma2_hsgp_idx + 1) {
      block_specs.push_back({layout.log_sigma2_hsgp_idx, 2});
    }
    if (layout.is_bym2 && layout.log_sigma_bym2_idx >= 0 &&
        layout.logit_rho_bym2_idx == layout.log_sigma_bym2_idx + 1) {
      block_specs.push_back({layout.log_sigma_bym2_idx, 2});
    }
    if (layout.is_gp && layout.log_sigma2_gp_idx >= 0 &&
        layout.log_phi_gp_idx == layout.log_sigma2_gp_idx + 1) {
      block_specs.push_back({layout.log_sigma2_gp_idx, 2});
    }
    if (layout.is_multiscale_gp) {
      if (layout.log_sigma2_gp_local_idx >= 0 &&
          layout.log_phi_gp_local_idx == layout.log_sigma2_gp_local_idx + 1) {
        block_specs.push_back({layout.log_sigma2_gp_local_idx, 2});
      }
      if (layout.log_sigma2_gp_regional_idx >= 0 &&
          layout.log_phi_gp_regional_idx == layout.log_sigma2_gp_regional_idx + 1) {
        block_specs.push_back({layout.log_sigma2_gp_regional_idx, 2});
      }
    }
    if (layout.is_st_gp && layout.log_phi_st_space_idx >= 0 &&
        layout.log_phi_st_time_idx == layout.log_phi_st_space_idx + 1) {
      block_specs.push_back({layout.log_phi_st_space_idx, 2});
    }
    if (layout.has_phi_num && layout.has_phi_denom &&
        layout.log_phi_denom_idx == layout.log_phi_num_idx + 1) {
      block_specs.push_back({layout.log_phi_num_idx, 2});
    }
    // If still no blocks found, fall back to DIAG
    if (block_specs.empty()) {
      effective_metric = MassMatrixType::DIAG;
      if (verbose) {
        REprintf("  [BLOCK_DIAG] No correlated hyperparameter pairs found, falling back to DIAG\n");
      }
    }
  }

  // Auto-downgrade dense to diagonal when n_params too large
  // Dense needs O(p^2) storage and O(p^3) Cholesky; also needs n_warmup >= p samples
  if (effective_metric == MassMatrixType::DENSE && n_params > DENSE_MAX_PARAMS) {
    if (verbose) {
      REprintf("  [DENSE] n_params=%d > %d: auto-downgrading to diagonal\n",
               n_params, DENSE_MAX_PARAMS);
    }
    effective_metric = MassMatrixType::DIAG;
  }

  DenseMassMatrix mass;
  try {
    if (effective_metric == MassMatrixType::BLOCK_DIAG && !block_specs.empty()) {
      mass.init_block_diag(n_params, block_specs);
    } else {
      mass.init(n_params, effective_metric);
    }
  } catch (const std::bad_alloc&) {
    if (effective_metric == MassMatrixType::DENSE) {
      if (verbose) {
        REprintf("  [DENSE] Allocation failed for p=%d, falling back to diagonal\n", n_params);
      }
      effective_metric = MassMatrixType::DIAG;
      mass.init(n_params, effective_metric);
    } else {
      throw;
    }
  }

  // Warm-start mass matrix from model structure.
  // Sets informed diagonal entries for parameter groups with known posterior scale,
  // giving the step size tuner a reasonable starting point even before warmup
  // samples are collected. This is critical for HSGP (m^2 basis coefficients),
  // BYM2 (spatial + IID), and correlated slopes (z ~ N(0,1)) models where the
  // identity mass causes excessively small epsilon → deep NUTS trees.
  {
    std::vector<double> inv_m(n_params, 1.0);
    std::vector<double> sqrt_m(n_params, 1.0);
    bool any_informed = false;

    // HSGP basis coefficients: beta_j ~ N(0, 1) → posterior variance ≈ 1
    // Hyperparameters: log_sigma2 ~ prior with moderate variance,
    //                  log_lengthscale ~ LogNormal(0,1) → variance ≈ 1
    if (layout.is_hsgp) {
      for (int j = layout.hsgp_beta_start; j < layout.hsgp_beta_end; j++) {
        inv_m[j] = 1.0;  // N(0,1) prior → unit scale
      }
      inv_m[layout.log_sigma2_hsgp_idx] = 1.0;
      inv_m[layout.log_lengthscale_hsgp_idx] = 1.0;
      any_informed = true;
    }

    // ICAR: phi[s] precision ≈ degree (number of neighbors)
    // Higher degree → smaller variance → tighter mass
    if (layout.has_spatial && !layout.is_bym2 &&
        data.spatial_type == SpatialType::ICAR && !data.adj_row_ptr.empty()) {
      for (int s = 0; s < (layout.spatial_end - layout.spatial_start); s++) {
        int degree = data.adj_row_ptr[s + 1] - data.adj_row_ptr[s];
        // ICAR precision diagonal ≈ degree; variance ≈ 1/degree
        double var_est = 1.0 / std::max(1.0, (double)degree);
        inv_m[layout.spatial_start + s] = var_est;
      }
      any_informed = true;
    }

    // BYM2: spatial phi ~ ICAR (eigenvalue-scaled), theta ~ N(0, I)
    // Riebler parameterization: phi[s] ≈ scale_factor variance
    if (layout.is_bym2) {
      double sf = std::max(data.bym2_scale_factor, 0.1);
      for (int s = layout.spatial_start; s < layout.spatial_end; s++) {
        inv_m[s] = sf * sf;  // ICAR variance ~ scale_factor^2
      }
      for (int s = layout.theta_bym2_start; s < layout.theta_bym2_end; s++) {
        inv_m[s] = 1.0;  // IID: N(0,1)
      }
      inv_m[layout.log_sigma_bym2_idx] = 1.0;
      inv_m[layout.logit_rho_bym2_idx] = 4.0;  // logit scale: wider
      any_informed = true;
    }

    // Correlated slopes: z ~ N(0, 1) (non-centered), Cholesky raw ~ tanh
    if (layout.has_re_correlated_slopes) {
      // RE slopes z values
      for (int j = layout.re_start; j < layout.re_end; j++) {
        inv_m[j] = 1.0;  // z ~ N(0,1)
      }
      any_informed = true;
    }

    // Temporal effects: RW1/RW2 have known precision structure
    if (layout.has_temporal) {
      // Temporal effects: moderate scale (tau-dependent, start at 1.0)
      for (int j = layout.temporal_start; j < layout.temporal_end; j++) {
        inv_m[j] = 1.0;
      }
      // AR1 rho: logit scale variance ≈ 4
      if (layout.logit_rho_ar1_idx >= 0) {
        inv_m[layout.logit_rho_ar1_idx] = 4.0;
      }
      any_informed = true;
    }

    // Non-centered RE: z ~ N(0, 1) → unit scale
    if (layout.has_re && data.re_parameterization == 1) {  // 1 = non-centered
      for (int j = layout.re_start; j < layout.re_end; j++) {
        inv_m[j] = 1.0;
      }
      any_informed = true;
    }

    // GP/SVC/TVC hyperparameters: moderate scale
    if (layout.is_gp) {
      if (layout.log_sigma2_gp_idx >= 0) inv_m[layout.log_sigma2_gp_idx] = 1.0;
      if (layout.log_phi_gp_idx >= 0) inv_m[layout.log_phi_gp_idx] = 1.0;
      any_informed = true;
    }

    // Temporal GP: NC z ~ N(0,1), logit_phi has wider posterior scale
    if (layout.is_temporal_gp) {
      for (int j = layout.temporal_start; j < layout.temporal_end; j++) {
        inv_m[j] = 1.0;  // z ~ N(0,1) for NC
      }
      if (layout.log_sigma2_temporal_gp_idx >= 0)
        inv_m[layout.log_sigma2_temporal_gp_idx] = 1.0;
      if (layout.logit_phi_temporal_gp_idx >= 0)
        inv_m[layout.logit_phi_temporal_gp_idx] = 4.0;  // logit scale: wider
      any_informed = true;
    }

    if (any_informed) {
      // Compute sqrt_mass from inv_mass
      for (int i = 0; i < n_params; i++) {
        inv_m[i] = std::max(1e-3, std::min(inv_m[i], 1e3));
        sqrt_m[i] = 1.0 / std::sqrt(inv_m[i]);
      }
      mass.set_diagonal(inv_m, sqrt_m);
      if (verbose) {
        REprintf("  [WARMSTART] Initialized mass matrix from model structure\n");
      }
    }
  }

  // Initialize sparse GMRF block for ST_IV spatiotemporal interaction.
  // Uses sparse Cholesky of posterior precision Q = tau*(Q_s⊗Q_t) + diag(H_lik).
  // At warmup end, extracts diag(Q^{-1}) to set precision-informed diagonal mass.
  // NOTE: Factorization happens later (after warmup discovers tau and H_lik).
  // Centered-only: the raw parameter this reads at warmup end (q[st_delta_start+k],
  // see the [SPARSE_GMRF] block below) IS delta under the centered parameterization.
  // Under non-centered (data.st_parameterization == 1) that slot holds z = delta *
  // sqrt(tau) instead, which this Q = tau*(Qs⊗Qt) + diag(H_lik) construction does not
  // describe -- z's natural scale is tau-free and near-unit, which the ordinary
  // Welford-adapted mass below already handles.
  bool use_sparse_gmrf_mass = (data.st_parameterization == 0);
  if (use_sparse_gmrf_mass && data.has_spatiotemporal && data.spatiotemporal_data.type == STType::TYPE_IV) {
    int st_S = data.spatiotemporal_data.n_spatial;
    int st_T = data.spatiotemporal_data.n_times;
    mass.sparse_gmrf.init(layout.st_delta_start, st_S, st_T);
    if (verbose) {
      REprintf("  [SPARSE_GMRF] ST_IV block initialized: %dx%d=%d params at offset %d\n",
               st_S, st_T, st_S * st_T, layout.st_delta_start);
    }
  }

  // Recompute epsilon with warm-start mass (if informed)
  // This gives the dual averaging a better starting point when mass is pre-set
  if (mass.type != MassMatrixType::DIAG && !mass.inv_mass_diag.empty()) {
    epsilon = find_reasonable_epsilon(q, data, layout, rng, mass.inv_mass_diag);
  }

  WelfordStats mass_stats(n_params);              // Always track diagonal
  WelfordCovStats cov_stats(n_params);            // Only used when dense
  bool use_mass_matrix = false;

  // L-BFGS mass matrix adaptation (warmup-only)
  // Uses L-BFGS to learn curvature during warmup, then switches to standard HMC
  bool use_lbfgs = data.has_multiscale_gp &&
                   data.multiscale_gp_data.sampler == ratiod_gp::MSGPSampler::LBFGS;
  ratiod_gp::LBFGSState lbfgs_state;
  std::vector<double> q_prev, grad_prev;
  bool lbfgs_initialized = false;
  bool lbfgs_warmup_done = false;  // After warmup, use standard HMC
  if (use_lbfgs) {
    lbfgs_state = ratiod_gp::LBFGSState(10, n_params);
    q_prev.resize(n_params);
    grad_prev.resize(n_params);
  }

  // Stan-style expanding warmup windows for mass matrix adaptation
  // Phase 1: [0, init_buffer) - step size adaptation only
  // Phase 2: [init_buffer, n_warmup - term_buffer) - mass matrix adaptation
  //   Windows double in size: 25, 50, 100, 200, ...
  //   Last window extends to fill remaining space
  // Phase 3: [n_warmup - term_buffer, n_warmup) - final step size tuning
  // Models with structured warm-start (ICAR degree, BYM2 scale, HSGP) already
  // have reasonable mass, so we can start adaptation earlier (init_buffer=25).
  // This saves ~50 iterations of deep-tree warmup. Temporal_gp warm-start is
  // trivial (identity-like) so it still needs the full 75 iterations.
  bool has_structured_warmstart = layout.is_hsgp || layout.is_bym2 ||
    (layout.has_spatial && !layout.is_bym2 && data.spatial_type == SpatialType::ICAR && !data.adj_row_ptr.empty());
  int init_buffer = has_structured_warmstart ? 25 : 75;
  int term_buffer = 50;
  // For high-dimensional models (p>80), a 25-sample first mass window gives
  // very noisy variance estimates (25 samples / 108 params = 0.23 samples/param).
  // Skip the tiny first window by using a larger init_window (=50), so the first
  // mass update has ~50 samples. This trades one less mass update for better quality.
  int init_window = (n_params > 80) ? 50 : 25;

  // Dense mass models: balance final step size tuning vs warmup budget.
  // Models with p>100 need sufficient mass adaptation windows (warmup is fixed),
  // so keep term_buffer moderate. Previous 75 was too aggressive — used 30%
  // of warmup for final tuning, leaving fewer samples for mass adaptation.
  if (effective_metric == MassMatrixType::DENSE && n_params > 100) {
    term_buffer = 60;  // Reduced from 75 — saves 15 iterations for mass adaptation
  }

  // Note: For p~24, first mass window (25 samples < 29 needed) fails,
  // but this is fine — better to wait for more samples than set a poor
  // mass estimate early. The second window (100+ samples) gives good mass.

  // Adjust for short warmup
  if (n_warmup < init_buffer + term_buffer + init_window) {
    init_buffer = std::max(1, n_warmup / 5);
    term_buffer = std::max(1, n_warmup / 10);
    init_window = std::max(1, n_warmup - init_buffer - term_buffer);
  }

  // Compute mass adaptation window endpoints
  std::vector<int> mass_window_ends;
  {
    int adapt_end = n_warmup - term_buffer;
    if (adapt_end <= init_buffer) {
      // No room for mass adaptation windows
      mass_window_ends.push_back(std::max(1, adapt_end));
    } else {
      int next_end = init_buffer + init_window;
      int win_size = init_window;
      while (next_end < adapt_end) {
        int next_win = 2 * win_size;
        if (next_end + next_win > adapt_end) {
          // Extend current window to fill remaining space
          mass_window_ends.push_back(adapt_end);
          break;
        }
        mass_window_ends.push_back(next_end);
        win_size = next_win;
        next_end += win_size;
      }
      if (mass_window_ends.empty() || mass_window_ends.back() < adapt_end) {
        mass_window_ends.push_back(adapt_end);
      }
    }
  }
  int next_window_idx = 0;

  // Pre-allocate NUTS workspace (zero-allocation tree building)
  NUTSWorkspace nuts_ws;
  std::vector<double> _nuts_p;              // Momentum sampling buffer
  std::vector<double> _nuts_q_proposal;     // Persistent proposal (survives tree resets)
  std::vector<double> _nuts_grad_proposal;  // Persistent proposal gradient
  if (use_nuts) {
    nuts_ws.init(n_params, max_treedepth);
    nuts_ws.gradient_fn = resolve_gradient_fn(g_gradient_mode, data, layout);
    _nuts_p.resize(n_params);
    _nuts_q_proposal.resize(n_params);
    _nuts_grad_proposal.resize(n_params);
  }

  int sample_idx = 0;
  int n_accept = 0;
  int n_divergent = 0;
  // Adaptive NUTS→fixed-L switching: monitor early sampling for max treedepth
  int nuts_probe_window = std::min(20, n_sample);  // Check first 20 sampling iterations
  int nuts_probe_maxd = 0;  // Count of maxd hits in probe window
  bool nuts_probing = use_nuts && (L == 0);  // Only probe when using NUTS by default

  // WALNUTS in place of the NUTS trajectory (tulpa/walnuts.h): an exact kernel
  // whose macro step is subdivided where the local curvature needs it.
  // `epsilon` is the macro step; dual averaging tunes it on the mean
  // acceptance of each macro step's coarsest subdivision.
  const bool use_walnuts = (walnuts != nullptr) && use_nuts;
  tulpa::WalnutsWorkspace walnuts_ws;
  if (use_walnuts) {
    walnuts_ws.init(n_params, max_treedepth);
    result.sampler = "WALNUTS";
  }

  int warmup_total_leapfrog = 0;  // TEMP: diagnostic counter
  // Note: warmup divergences are normal for DIAG models and resolve via dual
  // averaging — only final epsilon matters (checked at warmup end).

  if (verbose && layout.is_temporal_gp) {
    REprintf("  [MASS-WINDOWS] n_warmup=%d, windows=[", n_warmup);
    for (size_t w = 0; w < mass_window_ends.size(); w++) {
      REprintf("%d%s", mass_window_ends[w], w + 1 < mass_window_ends.size() ? "," : "");
    }
    REprintf("]\n");
  }
  for (int iter = 0; iter < n_iter; iter++) {
    bool is_warmup = (iter < n_warmup);
    // Check if we've reached a mass adaptation window boundary
    if (is_warmup && next_window_idx < (int)mass_window_ends.size() &&
        iter == mass_window_ends[next_window_idx]) {
      bool dense_covariance_set = false;  // Track if DENSE covariance (not just diagonal) succeeded this window
      // Dense mass matrix: try full covariance first
      // OAS shrinkage guarantees PD even when n < p, so we can lower the
      // threshold from n_params+5.  For large p the original threshold is
      // unreachable during warmup (e.g. p=159, need 164 but only get 125).
      // New threshold: min(p+5, max(50, p/2))  — for p=159 this is 79.
      int dense_threshold = std::min(n_params + 5,
                                     std::max(50, n_params / 2));
      if (mass.type == MassMatrixType::DENSE && cov_stats.n >= dense_threshold) {
        auto cov = cov_stats.covariance();
        if (mass.update_from_covariance(cov.data(), cov_stats.n)) {
          use_mass_matrix = true;
          dense_covariance_set = true;
          if (verbose) {
            REprintf("  [DENSE] Window %d (iter %d): dense mass SET (n=%d, p=%d, OAS shrinkage=%.3f)\n",
                     next_window_idx, iter, cov_stats.n, n_params,
                     cov_stats.shrinkage_intensity);
          }
        } else {
          // Cholesky failed — mass auto-degraded to DIAG, use diagonal stats
          if (verbose) {
            REprintf("  [DENSE] Window %d (iter %d): Cholesky FAILED (cov_stats.n=%d, p=%d)\n",
                     next_window_idx, iter, cov_stats.n, n_params);
          }
          if (mass_stats.n >= 10) {
            mass.set_diagonal(mass_stats.inv_mass(), mass_stats.sqrt_mass());
            use_mass_matrix = true;
          }
        }
      } else if (mass.type == MassMatrixType::DENSE) {
        // Not enough samples for dense yet — use diagonal as interim
        if (verbose) {
          REprintf("  [DENSE] Window %d (iter %d): not enough samples (cov_stats.n=%d, need=%d)\n",
                   next_window_idx, iter, cov_stats.n, dense_threshold);
        }
        if (mass_stats.n >= 10) {
          mass.set_diagonal(mass_stats.inv_mass(), mass_stats.sqrt_mass());
          use_mass_matrix = true;
        }
      } else if (mass.type == MassMatrixType::BLOCK_DIAG) {
        // Block-diagonal: set diagonal for all params, then adapt block covariances
        if (mass_stats.n >= 10) {
          mass.set_diagonal(mass_stats.inv_mass(), mass_stats.sqrt_mass());
          use_mass_matrix = true;
        }
        int n_adapted = 0;
        for (auto& blk : mass.blocks) {
          if (blk.update_from_welford()) {
            n_adapted++;
          }
        }
        if (verbose && n_adapted > 0) {
          REprintf("  [BLOCK_DIAG] Window %d (iter %d): %d/%d blocks adapted (n=%d)\n",
                   next_window_idx, iter, n_adapted, (int)mass.blocks.size(), mass_stats.n);
        }
        // Reset block Welford accumulators for next window
        for (auto& blk : mass.blocks) {
          blk.reset_welford();
        }
      } else if (mass_stats.n >= 10) {
        // Diagonal path
        mass.set_diagonal(mass_stats.inv_mass(), mass_stats.sqrt_mass());
        use_mass_matrix = true;
      }

      // Temporal GP NC: z ~ N(0,1) by construction → optimal diag mass ≈ 1.0.
      // With limited warmup samples, noisy variance estimates for 20 z params
      // create unbalanced mass → small epsilon. Fix z entries to 1.0 so the
      // step size is driven by the hyperparameters (beta, sigma2, phi) only.
      if (verbose && layout.is_temporal_gp) {
        REprintf("  [Z-DEBUG] Window %d (iter %d): use_mass=%d, tgp=%d, nc=%d, ts=%d, te=%d, mass_n=%d\n",
                 next_window_idx, iter, (int)use_mass_matrix,
                 (int)layout.is_temporal_gp, data.temporal_gp_parameterization,
                 layout.temporal_start, layout.temporal_end, mass_stats.n);
      }
      if (use_mass_matrix && layout.is_temporal_gp &&
          data.temporal_gp_parameterization == 1 &&
          layout.temporal_start >= 0 && layout.temporal_end > layout.temporal_start) {
        if (verbose) {
          REprintf("  [Z-FREEZE] Window %d: z mass before=[", next_window_idx);
          for (int j = layout.temporal_start; j < std::min(layout.temporal_end, layout.temporal_start + 5); j++) {
            REprintf("%.3f%s", mass.inv_mass_diag[j], j < layout.temporal_start + 4 ? "," : "");
          }
          REprintf("...], hyper=[");
          // Print beta and hyperparams
          for (int j = 0; j < std::min(4, layout.temporal_start); j++) {
            REprintf("%.3f%s", mass.inv_mass_diag[j], j < 3 ? "," : "");
          }
          REprintf("], sigma2=%.3f, phi=%.3f\n",
                   layout.log_sigma2_temporal_gp_idx >= 0 ? mass.inv_mass_diag[layout.log_sigma2_temporal_gp_idx] : -1.0,
                   layout.logit_phi_temporal_gp_idx >= 0 ? mass.inv_mass_diag[layout.logit_phi_temporal_gp_idx] : -1.0);
        }
        for (int j = layout.temporal_start; j < layout.temporal_end; j++) {
          mass.inv_mass_diag[j] = 1.0;
          mass.sqrt_mass_diag[j] = 1.0;
        }
      }

      mass_stats.reset();
      // For dense: only reset cov_stats when full covariance was successfully
      // computed THIS window. Otherwise keep accumulating across windows until
      // we have enough samples. This prevents the chicken-and-egg problem
      // where short windows never collect enough.
      // NOTE: We use dense_covariance_set (not mass.adapted) because
      // set_diagonal() also sets adapted=true, which would incorrectly
      // trigger a reset when we're still building up covariance samples.
      if (mass.type != MassMatrixType::DENSE || dense_covariance_set) {
        cov_stats.reset();
      }
      // Re-initialize step size with current mass matrix (A3)
      // Use dense-aware version when dense mass is adapted, so the step size
      // is calibrated for the rotated phase space (not just the diagonal).
      if (use_mass_matrix && mass.type == MassMatrixType::DENSE && mass.adapted) {
        epsilon = find_reasonable_epsilon_dense(q, data, layout, rng, mass);
      } else if (use_mass_matrix) {
        epsilon = find_reasonable_epsilon(q, data, layout, rng, mass.inv_mass_diag);
      } else {
        epsilon = find_reasonable_epsilon(q, data, layout, rng);
      }
      da = DualAveraging(epsilon, n_params, target_boost);
      if (use_nuts) da.target_accept = nuts_target_accept;  // Preserve model-adaptive target

      next_window_idx++;
    }

    // L-BFGS: transition from L-BFGS to standard HMC at end of warmup
    // Extract diagonal mass matrix from learned curvature
    if (use_lbfgs && !lbfgs_warmup_done && iter == n_warmup - 1 && lbfgs_initialized) {
      // Use gamma from L-BFGS as uniform scaling for mass matrix
      // gamma = (s^T y) / (y^T y) approximates average inverse Hessian scaling
      double gamma = lbfgs_state.gamma;
      if (gamma > 0.01 && gamma < 100.0) {
        // Set inv_mass = gamma * I (larger gamma = larger variance = larger step in that direction)
        std::vector<double> inv_m(n_params, gamma);
        std::vector<double> sqrt_m(n_params, 1.0 / std::sqrt(gamma));
        mass.set_diagonal(inv_m, sqrt_m);
        use_mass_matrix = true;
      }
      lbfgs_warmup_done = true;
    }

    // =========================================================================
    // NUTS or fixed-trajectory HMC
    // =========================================================================
    double alpha = 0.0;
    bool divergent = false;
    int iter_n_leapfrog = L;
    int iter_treedepth = 0;

    if (use_nuts && !(use_lbfgs && !lbfgs_warmup_done)) {
      if (use_walnuts) {
        RatioWalnutsModel walnuts_model{mass, nuts_ws.gradient_fn, data,
                                        layout, n_params};
        tulpa::WalnutsTransitionResult w = tulpa::walnuts_transition(
          q, current_grad, log_prob_current, epsilon, max_treedepth,
          *walnuts, walnuts_model, walnuts_ws, rng);
        alpha = w.mean_accept;
        divergent = w.divergent;
        iter_n_leapfrog = w.n_grad;
        iter_treedepth = w.depth;
      } else {
      // -----------------------------------------------------------------
      // NUTS: No-U-Turn Sampler (optimized zero-allocation path)
      // -----------------------------------------------------------------

      auto& p = _nuts_p;

      // Step size jitter: ±20% random noise per trajectory
      // Prevents systematic step-size resonances that cause divergences.
      // Only during post-warmup sampling — warmup needs stable epsilon for adaptation.
      double eps_iter = epsilon;
      if (!is_warmup) {
        double jitter = 1.0 + 0.2 * (2.0 * unif(rng) - 1.0);  // U[0.8, 1.2]
        eps_iter = epsilon * jitter;
      }

      // Sample momentum p ~ N(0, M) where M = C^{-1}
      mass.sample_momentum(p.data(), rng);

      // Initial Hamiltonian (pointer-based, no vector overhead)
      double H0 = nuts_compute_hamiltonian_fast(
        log_prob_current, p.data(), mass, n_params
      );
      double delta_max = 1000.0;

      // Load current state into workspace persistent slots
      nuts_ws.load_node(NUTSWorkspace::NODE_LEFT_SLOT,
                        q.data(), p.data(), current_grad.data(), log_prob_current);
      nuts_ws.load_node(NUTSWorkspace::NODE_RIGHT_SLOT,
                        q.data(), p.data(), current_grad.data(), log_prob_current);

      // Initialize persistent proposal buffers (pre-allocated, no per-iter malloc)
      auto& q_proposal_data = _nuts_q_proposal;
      auto& grad_proposal_data = _nuts_grad_proposal;
      std::memcpy(q_proposal_data.data(), q.data(), n_params * sizeof(double));
      std::memcpy(grad_proposal_data.data(), current_grad.data(), n_params * sizeof(double));
      double log_prob_proposal = log_prob_current;
      double sum_log_weight = 0.0;  // Relative weights: log(exp(H0 - H0)) = 0

      int total_leapfrog = 0;
      double sum_accept_prob = 0.0;
      divergent = false;

      // Generalized U-turn tracking at top level (Stan-style)
      // rho = total momentum sum. rho_bck/rho_fwd = halves for 3-juncture checks.
      // At each iteration the entire old trajectory becomes one half,
      // the new subtree becomes the other half (Stan's approach).
      // Uses pre-allocated workspace vectors (no per-iteration heap allocation).
      auto& rho = nuts_ws.iter_rho;
      std::memcpy(rho.data(), p.data(), n_params * sizeof(double));
      auto& rho_bck = nuts_ws.iter_rho_bck;
      auto& rho_fwd = nuts_ws.iter_rho_fwd;
      std::fill(rho_bck.begin(), rho_bck.end(), 0.0);
      std::fill(rho_fwd.begin(), rho_fwd.end(), 0.0);

      // p_sharp = M^{-1} * p at initial point — full mass for correct U-turn geometry
      auto& p_sharp_init = nuts_ws.iter_p_sharp_init;
      mass.inv_mass_times_p(p.data(), p_sharp_init.data());

      // Boundary momenta: _end = far endpoint, _beg = origin-facing boundary
      // Stan naming: bck_end=bck_bck, bck_beg=bck_fwd, fwd_beg=fwd_bck, fwd_end=fwd_fwd
      auto& p_fwd_beg = nuts_ws.iter_p_fwd_beg;
      auto& p_fwd_end = nuts_ws.iter_p_fwd_end;
      auto& p_bck_beg = nuts_ws.iter_p_bck_beg;
      auto& p_bck_end = nuts_ws.iter_p_bck_end;
      std::memcpy(p_fwd_beg.data(), p.data(), n_params * sizeof(double));
      std::memcpy(p_fwd_end.data(), p.data(), n_params * sizeof(double));
      std::memcpy(p_bck_beg.data(), p.data(), n_params * sizeof(double));
      std::memcpy(p_bck_end.data(), p.data(), n_params * sizeof(double));
      auto& p_sharp_fwd_beg = nuts_ws.iter_p_sharp_fwd_beg;
      auto& p_sharp_fwd_end = nuts_ws.iter_p_sharp_fwd_end;
      auto& p_sharp_bck_beg = nuts_ws.iter_p_sharp_bck_beg;
      auto& p_sharp_bck_end = nuts_ws.iter_p_sharp_bck_end;
      std::memcpy(p_sharp_fwd_beg.data(), p_sharp_init.data(), n_params * sizeof(double));
      std::memcpy(p_sharp_fwd_end.data(), p_sharp_init.data(), n_params * sizeof(double));
      std::memcpy(p_sharp_bck_beg.data(), p_sharp_init.data(), n_params * sizeof(double));
      std::memcpy(p_sharp_bck_end.data(), p_sharp_init.data(), n_params * sizeof(double));

      // Build tree until U-turn or max depth
      for (int j = 0; j < max_treedepth; j++) {
        std::uniform_int_distribution<int> dir_dist(0, 1);
        int direction = 2 * dir_dist(rng) - 1;

        nuts_ws.reset_tree();

        int start_slot = nuts_ws.alloc_slot();
        if (start_slot < 0) break;
        if (direction == 1) {
          nuts_ws.copy_node(start_slot, NUTSWorkspace::NODE_RIGHT_SLOT);
        } else {
          nuts_ws.copy_node(start_slot, NUTSWorkspace::NODE_LEFT_SLOT);
        }

        // Stan: relabel halves before building subtree
        // Entire old trajectory becomes one half; new subtree is the other
        if (direction == 1) {
          // Extending forward: old trajectory → backward half
          std::memcpy(rho_bck.data(), rho.data(), n_params * sizeof(double));
          std::memcpy(p_bck_beg.data(), p_fwd_end.data(), n_params * sizeof(double));
          std::memcpy(p_sharp_bck_beg.data(), p_sharp_fwd_end.data(), n_params * sizeof(double));
        } else {
          // Extending backward: old trajectory → forward half
          std::memcpy(rho_fwd.data(), rho.data(), n_params * sizeof(double));
          std::memcpy(p_fwd_beg.data(), p_bck_end.data(), n_params * sizeof(double));
          std::memcpy(p_sharp_fwd_beg.data(), p_sharp_bck_end.data(), n_params * sizeof(double));
        }

        TreeStats subtree = build_tree_fast(
          nuts_ws, start_slot, direction, j,
          eps_iter, mass, H0, delta_max,
          data, layout, rng
        );

        total_leapfrog += subtree.n_leapfrog;
        sum_accept_prob += subtree.sum_accept_prob;

        if (subtree.divergent) {
          divergent = true;
        }

        if (!subtree.stop) {
          // Multinomial acceptance
          double log_sum_weight_subtree = subtree.sum_log_weight;
          double new_sum_log_weight = nuts_log_sum_exp(sum_log_weight, log_sum_weight_subtree);

          double accept_prob_subtree;
          if (log_sum_weight_subtree > new_sum_log_weight) {
            accept_prob_subtree = 1.0;
          } else {
            accept_prob_subtree = std::exp(log_sum_weight_subtree - new_sum_log_weight);
          }
          if (!std::isfinite(accept_prob_subtree)) accept_prob_subtree = 0.0;

          std::uniform_real_distribution<double> unif01(0.0, 1.0);
          if (unif01(rng) < accept_prob_subtree) {
            std::memcpy(q_proposal_data.data(), nuts_ws.q_at(subtree.proposal_slot),
                        n_params * sizeof(double));
            std::memcpy(grad_proposal_data.data(), nuts_ws.grad_at(subtree.proposal_slot),
                        n_params * sizeof(double));
            log_prob_proposal = subtree.log_prob_proposal;
          }

          sum_log_weight = new_sum_log_weight;
        }

        // Update direction endpoints and rho half from subtree
        // Use memcpy instead of std::move to preserve pre-allocated buffers
        if (direction == 1) {
          nuts_ws.copy_node(NUTSWorkspace::NODE_RIGHT_SLOT, subtree.right_slot);
          std::memcpy(rho_fwd.data(), subtree.rho.data(), n_params * sizeof(double));
          std::memcpy(p_fwd_beg.data(), subtree.p_beg.data(), n_params * sizeof(double));
          std::memcpy(p_fwd_end.data(), subtree.p_end.data(), n_params * sizeof(double));
          std::memcpy(p_sharp_fwd_beg.data(), subtree.p_sharp_beg.data(), n_params * sizeof(double));
          std::memcpy(p_sharp_fwd_end.data(), subtree.p_sharp_end.data(), n_params * sizeof(double));
        } else {
          nuts_ws.copy_node(NUTSWorkspace::NODE_LEFT_SLOT, subtree.left_slot);
          std::memcpy(rho_bck.data(), subtree.rho.data(), n_params * sizeof(double));
          std::memcpy(p_bck_beg.data(), subtree.p_beg.data(), n_params * sizeof(double));
          std::memcpy(p_bck_end.data(), subtree.p_end.data(), n_params * sizeof(double));
          std::memcpy(p_sharp_bck_beg.data(), subtree.p_sharp_beg.data(), n_params * sizeof(double));
          std::memcpy(p_sharp_bck_end.data(), subtree.p_sharp_end.data(), n_params * sizeof(double));
        }

        // Combine rho = rho_bck + rho_fwd
        for (int i = 0; i < n_params; i++) {
          rho[i] = rho_bck[i] + rho_fwd[i];
        }

        iter_treedepth = j + 1;

        // Generalized U-turn check at top level (3 junctures)
        if (subtree.stop) break;

        // Check 1: Full trajectory — far endpoints vs total rho
        bool persist = compute_criterion(p_sharp_bck_end.data(), p_sharp_fwd_end.data(),
                                         rho.data(), n_params);

        // Check 2: Backward half + seam from forward (rho = rho_bck + p_fwd_beg)
        auto& rho_seam = nuts_ws.iter_rho_seam;
        for (int i = 0; i < n_params; i++) {
          rho_seam[i] = rho_bck[i] + p_fwd_beg[i];
        }
        persist &= compute_criterion(p_sharp_bck_end.data(), p_sharp_fwd_beg.data(),
                                      rho_seam.data(), n_params);

        // Check 3: Seam from backward + forward half (rho = rho_fwd + p_bck_beg)
        for (int i = 0; i < n_params; i++) {
          rho_seam[i] = rho_fwd[i] + p_bck_beg[i];
        }
        persist &= compute_criterion(p_sharp_bck_beg.data(), p_sharp_fwd_end.data(),
                                      rho_seam.data(), n_params);

        if (!persist) break;
      }

      // Accept proposal: copy from persistent proposal buffers (memcpy, no alloc)
      std::memcpy(q.data(), q_proposal_data.data(), n_params * sizeof(double));
      std::memcpy(current_grad.data(), grad_proposal_data.data(), n_params * sizeof(double));
      log_prob_current = log_prob_proposal;

      // Average acceptance statistic for dual averaging
      alpha = (total_leapfrog > 0) ? (sum_accept_prob / total_leapfrog) : 0.0;
      iter_n_leapfrog = total_leapfrog;
      }
      n_accept++;

      if (divergent) n_divergent++;
      if (iter_treedepth >= max_treedepth) result.n_max_treedepth++;

      // Adaptation during warmup
      if (is_warmup) {
        epsilon = da.update(alpha);

        // Early detection of catastrophic dense mass during terminal buffer.
        // Normal epsilon with dense mass is 0.1-0.5. If it exceeds 2.0, the mass
        // matrix eigenvalues are pathological. Fall back to DIAG immediately so
        // the remaining terminal buffer iterations (~48) properly adapt epsilon.
        // inv_mass_diag is always kept in sync with the dense diagonal (line 62).
        if (iter >= n_warmup - term_buffer && iter < n_warmup - 1 &&
            mass.type == MassMatrixType::DENSE && mass.adapted && epsilon > 2.0) {
          if (verbose) {
            REprintf("  [DENSE] WARNING at iter %d: epsilon=%.4f (catastrophic). "
                     "Falling back to DIAG mass.\n", iter, epsilon);
          }
          mass.type = MassMatrixType::DIAG;
          // inv_mass_diag already populated from dense diagonal (update_from_covariance)
          epsilon = find_reasonable_epsilon(q, data, layout, rng, mass.inv_mass_diag);
          da = DualAveraging(epsilon, n_params, target_boost);
          if (use_nuts) da.target_accept = nuts_target_accept;
        }

        // DIAG→DENSE recovery is checked at warmup end (after da.final_epsilon)
        // rather than during warmup — warmup divergences are normal for DIAG models
        // and resolve via dual averaging. Only catastrophic final epsilon matters.

        if (iter >= init_buffer && iter < n_warmup - term_buffer) {
          mass_stats.update(q);
          if (mass.type == MassMatrixType::DENSE) {
            cov_stats.update(q);
          }
          if (mass.type == MassMatrixType::BLOCK_DIAG) {
            for (auto& blk : mass.blocks) {
              blk.welford_update(q.data());
            }
          }
        }
        if (iter == n_warmup - 1) {
          epsilon = da.final_epsilon();

          // DIAG→BLOCK_DIAG→DENSE recovery at warmup end: if AUTO selected DIAG but the
          // final adapted epsilon is catastrophic (>2.0), DIAG can't capture the
          // posterior geometry. Try BLOCK_DIAG first if block_specs available,
          // otherwise fall back to DENSE with identity mass.
          if (auto_selected_diag && mass.type == MassMatrixType::DIAG &&
              epsilon > 2.0) {
            if (!block_specs.empty()) {
              // Try BLOCK_DIAG recovery first (cheaper than full DENSE)
              if (verbose) {
                REprintf("  [DIAG->BLOCK_DIAG] Warmup end: final epsilon=%.4f (catastrophic). "
                         "Switching to BLOCK_DIAG (adapted=false).\n", epsilon);
              }
              mass.init_block_diag(n_params, block_specs);
              effective_metric = MassMatrixType::BLOCK_DIAG;
              auto_selected_diag = false;
              epsilon = find_reasonable_epsilon(q, data, layout, rng);
              da = DualAveraging(epsilon, n_params, target_boost);
              if (use_nuts) da.target_accept = nuts_target_accept;
              epsilon = da.final_epsilon();
            } else if (n_params <= DENSE_MAX_PARAMS) {
              if (verbose) {
                REprintf("  [DIAG->DENSE] Warmup end: final epsilon=%.4f (catastrophic). "
                         "Switching to DENSE identity mass.\n", epsilon);
              }
              mass.init(n_params, MassMatrixType::DENSE);
              effective_metric = MassMatrixType::DENSE;
              auto_selected_diag = false;
              epsilon = find_reasonable_epsilon(q, data, layout, rng);
              da = DualAveraging(epsilon, n_params, target_boost);
              if (use_nuts) da.target_accept = nuts_target_accept;
              epsilon = da.final_epsilon();
            }
          }

          // BLOCK_DIAG→DIAG fallback: if epsilon still catastrophic after BLOCK_DIAG
          if (epsilon > 2.0 && mass.type == MassMatrixType::BLOCK_DIAG) {
            if (verbose) {
              REprintf("  [BLOCK_DIAG->DIAG] WARNING: epsilon=%.4f still catastrophic. "
                       "Falling back to DIAG.\n", epsilon);
            }
            mass.init(n_params, MassMatrixType::DIAG);
            effective_metric = MassMatrixType::DIAG;
            epsilon = find_reasonable_epsilon(q, data, layout, rng, mass.inv_mass_diag);
            da = DualAveraging(epsilon, n_params, target_boost);
            if (use_nuts) da.target_accept = nuts_target_accept;
            epsilon = da.final_epsilon();
          }

          // Final safety net: if epsilon is still > 1.0 with dense mass after
          // the full terminal buffer, fall back to DIAG. This catches cases
          // where the catastrophe develops slowly.
          if (epsilon > 1.0 && mass.type == MassMatrixType::DENSE && mass.adapted) {
            if (verbose) {
              REprintf("  [DENSE] WARNING: epsilon=%.4f after warmup (catastrophic). "
                       "Falling back to DIAG mass.\n", epsilon);
            }
            mass.type = MassMatrixType::DIAG;
            epsilon = find_reasonable_epsilon(q, data, layout, rng, mass.inv_mass_diag);
            da = DualAveraging(epsilon, n_params, target_boost);
            if (use_nuts) da.target_accept = nuts_target_accept;
            epsilon = da.final_epsilon();
          }

          // Precision-informed diagonal mass for ST_IV at warmup end.
          // Build Q_post = tau*(Q_s⊗Q_t) + diag(h_lik), factorize, extract diag(Q^{-1}).
          if (mass.sparse_gmrf.active && !mass.sparse_gmrf.factorized) {
            int st_S = data.spatiotemporal_data.n_spatial;
            int st_T = data.spatiotemporal_data.n_times;
            int ST = st_S * st_T;
            double tau_st = std::exp(q[layout.log_tau_st_idx]);

            // Compute likelihood Hessian diagonal for each ST cell
            std::vector<double> h_lik(ST, 0.0);
            for (int i = 0; i < data.N; i++) {
              if (data.spatiotemporal_data.st_flat[i] <= 0) continue;
              int k = data.spatiotemporal_data.st_flat[i] - 1;
              double eta_i = 0.0;
              for (int p2 = 0; p2 < data.p_num; p2++)
                eta_i += data.X_num_flat[static_cast<size_t>(i) * data.p_num + p2] * q[layout.beta_num_start + p2];
              if (layout.has_re && data.re_group[i] > 0)
                eta_i += re_value_for_eta(&q[layout.re_start], data.re_group[i] - 1,
                                           std::exp(q[layout.log_sigma_re_idx]), data.re_parameterization);
              if (layout.has_spatial && data.spatial_group[i] > 0)
                eta_i += q[layout.spatial_start + data.spatial_group[i] - 1];
              if (layout.has_temporal && !data.temporal_time_idx.empty() &&
                  i < (int)data.temporal_time_idx.size() && data.temporal_time_idx[i] > 0) {
                int t_idx = data.temporal_time_idx[i] - 1;
                int g_idx = data.temporal_group_idx[i] - 1;
                int t_base = g_idx * data.n_times + t_idx;
                if (t_base >= 0 && t_base < (int)q.size()) eta_i += q[layout.temporal_start + t_base];
              }
              eta_i += q[layout.st_delta_start + k];
              double mu_i = std::exp(eta_i);
              double h_i = 0.0;
              if (data.model_type == ModelType::POISSON_GAMMA) {
                h_i = mu_i;
              } else if (data.model_type == ModelType::BINOMIAL) {
                double p_i = 1.0 / (1.0 + std::exp(-eta_i));
                h_i = data.y_denom[i] * p_i * (1.0 - p_i);
              } else if (data.model_type == ModelType::NEGBIN_NEGBIN ||
                         data.model_type == ModelType::NEGBIN_GAMMA) {
                double phi = std::exp(q[layout.log_phi_num_idx]);
                h_i = mu_i / (1.0 + mu_i / phi);
              } else {
                h_i = mu_i;
              }
              if (data.spatiotemporal_data.shared) h_i *= 2.0;
              h_lik[k] += std::max(h_i, 1e-6);
            }

            mass.sparse_gmrf.build_and_factorize(
                data.spatiotemporal_data.adj_row_ptr,
                data.spatiotemporal_data.adj_col_idx,
                data.spatiotemporal_data.temporal_type,
                data.spatiotemporal_data.temporal_cyclic,
                tau_st, h_lik.data(), 0.001
            );

            if (mass.sparse_gmrf.factorized) {
              // Extract diag(Q^{-1}) and set diagonal mass for ST params
              int n_set = 0;
              double sum_var = 0.0;
              for (int k = 0; k < ST; k++) {
                Eigen::VectorXd ek = Eigen::VectorXd::Zero(ST);
                ek[k] = 1.0;
                Eigen::VectorXd col_k = mass.sparse_gmrf.llt.solve(ek);
                double var_k = col_k[k];
                if (var_k > 1e-10 && var_k < 100.0) {
                  mass.inv_mass_diag[layout.st_delta_start + k] = var_k;
                  n_set++;
                  sum_var += var_k;
                }
              }
              // Deactivate sparse GMRF — diagonal mass is now informed
              mass.sparse_gmrf.active = false;
              // Recompute epsilon with new mass
              epsilon = find_reasonable_epsilon(q, data, layout, rng, mass.inv_mass_diag);
              da = DualAveraging(epsilon, n_params, target_boost);
              if (use_nuts) da.target_accept = nuts_target_accept;
              epsilon = da.final_epsilon();
              if (verbose) {
                REprintf("  [SPARSE_GMRF] Diagonal mass set for %d/%d ST params, avg_var=%.6f, tau=%.4f, new epsilon=%.6f\n",
                         n_set, ST, n_set > 0 ? sum_var / n_set : 0.0, tau_st, epsilon);
              }
            } else if (verbose) {
              REprintf("  [SPARSE_GMRF] WARNING: Cholesky failed, keeping adapted diagonal mass\n");
            }
          }

          if (verbose) {
            REprintf("  [METRIC] Warmup done: epsilon=%.6f, mass.type=%s, mass.adapted=%d\n",
                     epsilon, metric_name(mass.type), (int)mass.adapted);
          }
        }
        // Print tree depth for last 10 warmup iterations
        if (verbose && iter >= n_warmup - 10) {
          REprintf("  [%s] warmup iter %d: treedepth=%d, epsilon=%.6f\n",
                   metric_name(mass.type),
                   iter, iter_treedepth, epsilon);
        }
      }

      // Adaptive NUTS probe: warn if most early iterations hit max treedepth
      // (Stan's approach: warn but keep NUTS running — truncated NUTS picks
      // from up to 2^depth candidates, far better than HMC(L=10) with tiny epsilon)
      if (nuts_probing && !is_warmup && sample_idx < nuts_probe_window) {
        if (iter_treedepth >= max_treedepth) nuts_probe_maxd++;
        if (sample_idx == nuts_probe_window - 1) {
          nuts_probing = false;  // Probe window complete
          if (nuts_probe_maxd >= (nuts_probe_window * 8 + 9) / 10) {
            result.n_max_treedepth += 0;  // Already counted above
            if (verbose) {
              REprintf("  [NUTS] %d/%d initial sampling iterations hit max treedepth (%d). "
                       "Consider increasing max_treedepth or reparameterizing.\n",
                       nuts_probe_maxd, nuts_probe_window, max_treedepth);
            }
          }
        }
      }
    } else {
      // -----------------------------------------------------------------
      // Fixed-trajectory HMC (original code)
      // -----------------------------------------------------------------

      // Sample momentum and compute kinetic energy
      std::vector<double> p(n_params);
      double kinetic_current = 0.0;
      double H_current;

      if (use_lbfgs && lbfgs_initialized && !lbfgs_warmup_done && lbfgs_state.d == n_params) {
        // L-BFGS: Sample p ~ N(0, B) where B ≈ 1/gamma * I (warmup only)
        std::vector<double> sqrt_diag = lbfgs_state.get_sqrt_B_diag();
        if ((int)sqrt_diag.size() == n_params) {
          for (int i = 0; i < n_params; i++) {
            p[i] = normal(rng) * sqrt_diag[i];
          }
          kinetic_current = lbfgs_state.kinetic_energy(p);
          H_current = -log_prob_current + kinetic_current;
        } else {
          mass.sample_momentum(p.data(), rng);
          kinetic_current = mass.kinetic_energy(p.data());
          H_current = -log_prob_current + kinetic_current;
        }
      } else {
        mass.sample_momentum(p.data(), rng);
        kinetic_current = mass.kinetic_energy(p.data());
        H_current = -log_prob_current + kinetic_current;
      }

      // Leapfrog integration
      std::vector<double> q_prop = q;
      std::vector<double> p_prop = p;

      // Determine effective L for this iteration
      int L_eff = L;
      if (use_nuts && use_lbfgs && !lbfgs_warmup_done) {
        // During L-BFGS warmup with NUTS mode, use fixed L=20
        L_eff = 20;
      }

      if (use_lbfgs && lbfgs_initialized && !lbfgs_warmup_done && lbfgs_state.d == n_params) {
        // L-BFGS leapfrog
        std::vector<double> grad(n_params);
        compute_gradient(q_prop, data, layout, grad);

        for (int l = 0; l < L_eff; l++) {
          for (int i = 0; i < n_params; i++) {
            p_prop[i] += 0.5 * epsilon * grad[i];
          }
          std::vector<double> Hp(n_params);
          lbfgs_state.multiply_H(p_prop, Hp);
          for (int i = 0; i < n_params; i++) {
            q_prop[i] += epsilon * Hp[i];
            if (!std::isfinite(q_prop[i])) {
              divergent = true;
              break;
            }
          }
          if (divergent) break;
          compute_gradient(q_prop, data, layout, grad);
          for (int i = 0; i < n_params; i++) {
            p_prop[i] += 0.5 * epsilon * grad[i];
          }
          for (int i = 0; i < n_params; i++) {
            if (!std::isfinite(p_prop[i]) || std::abs(p_prop[i]) > 1e10) {
              divergent = true;
              break;
            }
          }
          if (divergent) break;
        }
      } else {
        // Standard leapfrog
        for (int l = 0; l < L_eff; l++) {
          LeapfrogResult lf;
          if (use_mass_matrix) {
            lf = leapfrog_step_mass(q_prop, p_prop, epsilon, mass.inv_mass_diag, data, layout);
          } else {
            lf = leapfrog_step(q_prop, p_prop, epsilon, data, layout);
          }
          q_prop = lf.q;
          p_prop = lf.p;
          if (lf.divergent) {
            divergent = true;
            break;
          }
        }
      }

      // Compute proposed Hamiltonian (use same log-post as gradient mode)
      double log_prob_prop;
      if (g_gradient_mode == GradientMode::AUTODIFF_ARENA ||
          g_gradient_mode == GradientMode::AUTODIFF_FORWARD ||
          g_gradient_mode == GradientMode::AUTODIFF_TAPE) {
        log_prob_prop = ratiod::compute_log_post_impl(q_prop, data, layout);
      } else {
        log_prob_prop = compute_log_post(q_prop, data, layout);
      }
      double kinetic_prop = 0.0;

      if (use_lbfgs && lbfgs_initialized && !lbfgs_warmup_done && lbfgs_state.d == n_params) {
        kinetic_prop = lbfgs_state.kinetic_energy(p_prop);
      } else {
        kinetic_prop = mass.kinetic_energy(p_prop.data());
      }
      double H_prop = -log_prob_prop + kinetic_prop;

      // Metropolis accept/reject
      alpha = std::min(1.0, std::exp(H_current - H_prop));
      if (!std::isfinite(alpha)) alpha = 0.0;

      std::uniform_real_distribution<double> unif01(0.0, 1.0);
      bool accepted = (unif01(rng) < alpha) && !divergent;
      if (accepted) {
        q = q_prop;
        log_prob_current = log_prob_prop;
        n_accept++;
        // Update cached gradient for transition to NUTS after L-BFGS warmup
        if (use_nuts) {
          compute_gradient(q, data, layout, current_grad);
        }
      }
      if (divergent) n_divergent++;

      // Adaptation during warmup
      if (is_warmup) {
        epsilon = da.update(alpha);
        // Only collect mass stats during mass adaptation phase (A5)
        if (iter >= init_buffer && iter < n_warmup - term_buffer) {
          mass_stats.update(q);
          if (mass.type == MassMatrixType::DENSE) {
            cov_stats.update(q);
          }
        }
        // On last warmup iteration, use averaged step size for sampling (A1)
        if (iter == n_warmup - 1) {
          epsilon = da.final_epsilon();
        }
      }

      // L-BFGS update: collect (s, y) pairs from accepted samples (warmup only)
      if (use_lbfgs && !lbfgs_warmup_done) {
        std::vector<double> grad_current(n_params);
        compute_gradient(q, data, layout, grad_current);

        if (!lbfgs_initialized) {
          q_prev = q;
          grad_prev = grad_current;
          lbfgs_initialized = true;
        } else if (accepted) {
          std::vector<double> s(n_params), y(n_params);
          for (int i = 0; i < n_params; i++) {
            s[i] = q[i] - q_prev[i];
            y[i] = grad_current[i] - grad_prev[i];
          }
          lbfgs_state.add_pair(s, y);
          q_prev = q;
          grad_prev = grad_current;
        }
      }

      iter_n_leapfrog = L_eff;
    }  // end fixed-trajectory HMC

    // Store sample (flat row-major storage, single memcpy)
    if (!is_warmup) {
      std::memcpy(result.sample_row(sample_idx), q.data(),
                  n_params * sizeof(double));
      // A non-centred block is sampled in a coordinate that is not the effect
      // it stands for, so the stored row carries the effect. q keeps the
      // sampled coordinate.
      {
          double* row = result.sample_row(sample_idx);
          if (data.gp_parameterization == 1 && data.has_gp && layout.is_gp) {
              double sigma2_store = std::exp(q[layout.log_sigma2_gp_idx]);
              double phi_store = std::exp(q[layout.log_phi_gp_idx]);
              RATIOD_TLS_WORKSPACE(ratiod_gp::NNGPNCWorkspace, nc_ws_store);
              ratiod_gp::nngp_nc_forward(&q[layout.gp_w_start], sigma2_store, phi_store,
                                         data.gp_data, nc_ws_store);
              int N_gp = data.gp_data.n_obs;
              for (int i = 0; i < N_gp; i++) {
                  row[layout.gp_w_start + i] = nc_ws_store.w[i];
              }
          }
          // The SVC row carries the field eta reads: each term on the effect
          // scale whichever coordinate it was sampled in, centred over
          // locations as svc_center_eta enters it.
          if (layout.has_svc && data.has_svc && !data.svc_is_hsgp &&
              data.svc_data.n_svc > 0) {
              const int n_svc_store = data.svc_data.n_svc;
              std::vector<double> svc_s2_store(n_svc_store), svc_phi_store(n_svc_store);
              for (int j = 0; j < n_svc_store; j++) {
                  svc_s2_store[j] = std::exp(q[layout.log_sigma2_svc_start + j]);
                  svc_phi_store[j] = std::exp(q[layout.log_phi_svc_start + j]);
              }
              ratiod_svc::svc_field<ratiod_svc::SVC_DENSITY_SLOT>(
                  &q[layout.svc_w_start], svc_s2_store.data(), svc_phi_store.data(),
                  data.svc_data, data.svc_gp_view, data.svc_noncentered,
                  &row[layout.svc_w_start]);
              ratiod_svc::svc_center_terms(&row[layout.svc_w_start], data.svc_data);
          }
          if (msgp_nc(data, layout)) {
              MultiscaleGPView msgp_store;
              msgp_store.read(q, data, layout);
              const int N_msgp_store = data.multiscale_gp_data.n_obs;
              for (int i = 0; i < N_msgp_store; i++) {
                  row[layout.gp_local_start + i] = msgp_store.local()[i];
                  row[layout.gp_regional_start + i] = msgp_store.regional()[i];
              }
          }
          if (temporal_effects_transformed(data, layout)) {
              std::vector<double> phi_store_buf;
              const double* phi_store = temporal_effects(q, data, layout, phi_store_buf);
              const int T_store = layout.temporal_end - layout.temporal_start;
              for (int i = 0; i < T_store; i++) {
                  row[layout.temporal_start + i] = phi_store[i];
              }
          }
          // The intrinsic multiscale arms enter eta centred, and the R side
          // adds the stored field straight back into eta, so the stored draw
          // is the centred one: an uncentred draw would report -- and predict
          // from -- a different series than the one the likelihood saw.
          if (layout.has_multiscale_temporal) {
              const auto& mst_store = data.multiscale_temporal_data;
              const int n_trend_store = layout.trend_end - layout.trend_start;
              const int n_seasonal_store = layout.seasonal_end - layout.seasonal_start;
              const bool trend_intrinsic_store =
                  n_trend_store > 0 &&
                  (mst_store.trend_type == TemporalType::RW1 ||
                   mst_store.trend_type == TemporalType::RW2);
              const bool seasonal_store =
                  n_seasonal_store > 0 && mst_store.seasonal_period > 0;
              // In the non-centred coordinate the row still holds z, so the
              // effects have to be built before they can be centred.
              if (mst_store.noncentered) {
                  const double s2_trend_store = (n_trend_store > 0)
                      ? std::exp(q[layout.log_sigma2_trend_idx]) : 1.0;
                  const double s2_seasonal_store = (n_seasonal_store > 0)
                      ? std::exp(q[layout.log_sigma2_seasonal_idx]) : 1.0;
                  const auto arms_store = ratiod_temporal::ms_read_arms(
                      trend_intrinsic_store ? &q[layout.trend_start] : nullptr,
                      n_trend_store,
                      seasonal_store ? &q[layout.seasonal_start] : nullptr,
                      n_seasonal_store,
                      s2_trend_store, s2_seasonal_store, mst_store);
                  if (trend_intrinsic_store) {
                      for (int i = 0; i < n_trend_store; i++)
                          row[layout.trend_start + i] = arms_store.trend[i];
                  }
                  if (seasonal_store) {
                      for (int i = 0; i < n_seasonal_store; i++)
                          row[layout.seasonal_start + i] = arms_store.seasonal[i];
                  }
              }
              if (trend_intrinsic_store) {
                  (void)tulpa::s2z_centre_component(row, layout.trend_start,
                                                    n_trend_store);
              }
              if (seasonal_store) {
                  (void)tulpa::s2z_centre_component(row, layout.seasonal_start,
                                                    n_seasonal_store);
              }
          }
      }
      result.log_prob[sample_idx] = log_prob_current;
      result.accept_prob[sample_idx] = alpha;
      result.n_leapfrog[sample_idx] = iter_n_leapfrog;
      result.divergent[sample_idx] = divergent ? 1 : 0;
      result.treedepth[sample_idx] = iter_treedepth;

      // Collapsed: store mode values from the last gradient evaluation
      if (data.gp_collapsed && data.has_gp && result.n_gp_collapsed > 0) {
          collapsed_gp_store_sample(sample_idx, collapsed_gp_ws(),
              result.gp_w_star_flat, result.n_gp_collapsed);
      }
      if ((data.icar_collapsed || data.bym2_collapsed) && result.n_icar_collapsed > 0) {
          collapsed_icar_store_sample(sample_idx, data, collapsed_icar_ws(),
              result.icar_phi_star_flat, result.bym2_theta_star_flat,
              result.n_icar_collapsed);
      }

      sample_idx++;
    } else {
      warmup_total_leapfrog += iter_n_leapfrog;  // TEMP: diagnostic
    }

    // Note: verbose output disabled in parallel - not thread-safe
    // Progress will be reported after parallel region
  }

  result.epsilon = da.final_epsilon();

  // Diagnostic stats - only when verbose
  if (verbose) {
    int sampling_total_lf = 0;
    for (int i = 0; i < result.n_sample; i++) sampling_total_lf += result.n_leapfrog[i];
    REprintf("  [STATS] Chain %d: metric=%s, adapted=%d, warmup_LF=%d, sampling_LF=%d, total_LF=%d, epsilon=%.6f\n",
             chain_id + 1,
             metric_name(mass.type),
             (int)mass.adapted,
             warmup_total_leapfrog, sampling_total_lf,
             warmup_total_leapfrog + sampling_total_lf, result.epsilon);
  }

  return result;
}

// R wrapper version - for single chain or non-parallel use
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
    int max_treedepth,
    MassMatrixType metric_type,
    double adapt_delta,
    const tulpa::WalnutsConfig* walnuts
) {
  // Runtime gradient check: compare active gradient function against numerical
  verify_gradient_or_fallback(q_init, data, layout);

  // Run C++ version - pass verbose through for debugging
  HMCResultCpp cpp_result = run_hmc_chain_cpp(
    q_init, data, layout, n_iter, n_warmup, L, chain_id, seed, verbose, max_treedepth, metric_type, adapt_delta, walnuts
  );

  // Convert to R result
  int n_params = q_init.size();
  HMCResult result = cpp_to_r_result(cpp_result, n_params);

  if (verbose) {
    int n_div = 0;
    for (int i = 0; i < cpp_result.n_sample; i++) {
      n_div += cpp_result.divergent[i];
    }
    Rcpp::Rcerr << "Chain " << (chain_id + 1) << " complete. "
                << "Divergent: " << n_div << std::endl;
  }

  return result;
}

// =====================================================================
// Run multiple chains in parallel using OpenMP
// =====================================================================

std::vector<HMCResult> run_hmc_parallel_chains(
    const std::vector<double>& q_init,
    const ModelData& data,
    int n_iter,
    int n_warmup,
    int L,
    int n_chains,
    unsigned int seed,
    bool verbose,
    int max_treedepth,
    MassMatrixType metric_type,
    double adapt_delta,
    const tulpa::WalnutsConfig* walnuts
) {
  ParamLayout layout = compute_param_layout(data);
  int n_params = layout.total_params;

  // Runtime gradient check: compare active gradient function against numerical
  // BEFORE spawning parallel chains. This is single-threaded, so R API and
  // g_gradient_mode modification are safe.
  verify_gradient_or_fallback(q_init, data, layout);

  // Use pure C++ containers in parallel region
  std::vector<HMCResultCpp> cpp_results(n_chains);

  // Thread-safe autodiff: Each chain creates its own tape via TapeScope (RAII).
  // All gradient modes (N, A, A_t, H) are now thread-safe and can run in parallel.
  // The old global tape limitation has been removed.

  require_autodiff_supported(data, layout);

  if (n_chains > 1) {
    // The team stays at ratiod_omp::chain_team_size() for the whole session;
    // n_cores caps how many chains are in flight at once. Sizing the team from
    // n_cores instead let it shrink between fits, which corrupts the heap on
    // Windows (see omp_chain_team.h). Without OpenMP the helper runs the chains
    // in order. verbose is off inside the region either way; the per-chain
    // summary is printed below.
    const int max_concurrent = data.n_cores > 0 ? data.n_cores : n_chains;
    ratiod_omp::for_each_chain(n_chains, max_concurrent, [&](int c) {
      cpp_results[c] = run_hmc_chain_cpp(
        q_init, data, layout,
        n_iter, n_warmup, L, c, seed, false, max_treedepth, metric_type, adapt_delta, walnuts
      );
    });
  } else {
    // Single chain - run sequentially with verbose output
    cpp_results[0] = run_hmc_chain_cpp(
      q_init, data, layout,
      n_iter, n_warmup, L, 0, seed, verbose, max_treedepth, metric_type, adapt_delta, walnuts
    );
  }

  // Convert to R objects outside parallel region (single-threaded)
  std::vector<HMCResult> results(n_chains);
  for (int c = 0; c < n_chains; c++) {
    results[c] = cpp_to_r_result(cpp_results[c], n_params);

    if (verbose && n_chains > 1) {
      // Print summary if we ran in parallel (verbose was disabled during parallel run)
      int n_div = 0;
      for (int i = 0; i < cpp_results[c].n_sample; i++) {
        n_div += cpp_results[c].divergent[i];
      }
      Rcpp::Rcerr << "Chain " << (c + 1) << " complete. "
                  << "Divergent: " << n_div << std::endl;
    }
  }

  return results;
}

} // namespace ratiod_hmc

// =====================================================================
// R EXPORTS
// =====================================================================

// HMC sampler with bundled list arguments to avoid R's 65-arg limit for .Call
// Parameters are bundled into logical groups:
//   re_params: random effects (group, n_groups, n_terms, group_matrix, slopes, etc.)
//   spatial_params: spatial structure (type, group, adjacency, etc.)
//   temporal_params: temporal structure (type, time_idx, group_idx, etc.)
//   prior_params: prior hyperparameters
//   zi_params: zero-inflation (type, X_zi, prior_sd)
//   latent_params: latent factors
//   st_params: spatiotemporal interaction
// [[Rcpp::export]]
Rcpp::List cpp_hmc_fit(
    Rcpp::NumericVector q_init,
    Rcpp::IntegerVector y_num,
    Rcpp::IntegerVector y_denom,
    Rcpp::NumericVector y_num_cont,
    Rcpp::NumericVector y_denom_cont,
    Rcpp::NumericMatrix X_num,
    Rcpp::NumericMatrix X_denom,
    std::string model_type_str,
    Rcpp::List re_params,
    Rcpp::List spatial_params,
    Rcpp::List temporal_params,
    Rcpp::List prior_params,
    Rcpp::List zi_params,
    Rcpp::List latent_params,
    Rcpp::List st_params,
    Rcpp::List tvc_params,  // Time-varying coefficients
    Rcpp::List svc_params,  // Spatially-varying coefficients
    int n_iter,
    int n_warmup,
    int L,
    int n_chains,
    unsigned int seed,
    int n_threads,
    bool verbose,
    std::string gradient_mode_str = "auto",
    int max_treedepth = 10,
    std::string metric_str = "auto",
    double adapt_delta = -1.0,
    bool walnuts = false,
    int n_cores = 0
) {
  using namespace ratiod_hmc;

  // Set global gradient mode from R parameter
  GradientMode grad_mode = parse_gradient_mode(gradient_mode_str);
  set_gradient_mode(grad_mode);

  const tulpa::WalnutsConfig walnuts_cfg;
  const tulpa::WalnutsConfig* walnuts_ptr = walnuts ? &walnuts_cfg : nullptr;

  // Parse metric type
  MassMatrixType metric_type = parse_metric_type(metric_str);

  // =========================================================================
  // Extract bundled parameters from lists with defensive checks
  // =========================================================================

  // The random-effect bundle is read by apply_re_params() below, once data.N
  // is known: see hmc_model_data_blocks.h.

  // Spatial parameters (eager deep copies)
  std::string spatial_type_str = Rcpp::as<std::string>(spatial_params["type"]);
  std::vector<int> spatial_group = Rcpp::as<std::vector<int>>(spatial_params["group"]);
  int n_spatial_units = Rcpp::as<int>(spatial_params["n_units"]);
  std::vector<int> adj_row_ptr = Rcpp::as<std::vector<int>>(spatial_params["adj_row_ptr"]);
  std::vector<int> adj_col_idx = Rcpp::as<std::vector<int>>(spatial_params["adj_col_idx"]);
  std::vector<int> n_neighbors = Rcpp::as<std::vector<int>>(spatial_params["n_neighbors"]);
  double bym2_scale_factor = Rcpp::as<double>(spatial_params["bym2_scale"]);

  // Precision mass matrix data (Q_inv and L_Q for ICAR/BYM2)
  std::vector<double> spatial_Q_inv;
  std::vector<double> spatial_L_Q;
  if (spatial_params.containsElementNamed("Q_inv") &&
      spatial_params.containsElementNamed("L_Q")) {
    SEXP qi_sexp = spatial_params["Q_inv"];
    SEXP lq_sexp = spatial_params["L_Q"];
    if (!Rf_isNull(qi_sexp) && !Rf_isNull(lq_sexp)) {
      spatial_Q_inv = Rcpp::as<std::vector<double>>(qi_sexp);
      spatial_L_Q = Rcpp::as<std::vector<double>>(lq_sexp);
    }
  }

  // The temporal bundle is read by apply_temporal_params() below.

  // Prior parameters
  double sigma_beta = Rcpp::as<double>(prior_params["sigma_beta"]);
  double sigma_re_scale = Rcpp::as<double>(prior_params["sigma_re_scale"]);
  double phi_prior_shape = Rcpp::as<double>(prior_params["phi_shape"]);
  double phi_prior_rate = Rcpp::as<double>(prior_params["phi_rate"]);
  double tau_spatial_shape = Rcpp::as<double>(prior_params["tau_spatial_shape"]);
  double tau_spatial_rate = Rcpp::as<double>(prior_params["tau_spatial_rate"]);

  // Zero-inflation parameters
  std::string zi_type_str = Rcpp::as<std::string>(zi_params["type"]);

  // Handle X_zi which may be numeric matrix or NULL
  Rcpp::NumericMatrix X_zi;
  SEXP xzi_sexp = zi_params["X"];
  if (!Rf_isNull(xzi_sexp) && Rf_isMatrix(xzi_sexp)) {
    X_zi = Rcpp::as<Rcpp::NumericMatrix>(xzi_sexp);
  } else {
    // Create empty dummy matrix
    X_zi = Rcpp::NumericMatrix(1, 1);
    X_zi(0, 0) = 1.0;
  }
  double zi_prior_sd = Rcpp::as<double>(zi_params["prior_sd"]);

  // One-inflation parameters (for OI-binomial and ZOIB)
  Rcpp::NumericMatrix X_oi;
  SEXP xoi_sexp = zi_params["X_oi"];
  if (!Rf_isNull(xoi_sexp) && Rf_isMatrix(xoi_sexp)) {
    X_oi = Rcpp::as<Rcpp::NumericMatrix>(xoi_sexp);
  } else {
    // Create empty dummy matrix
    X_oi = Rcpp::NumericMatrix(1, 1);
    X_oi(0, 0) = 1.0;
  }
  int p_oi = 0;
  SEXP p_oi_sexp = zi_params["p_oi"];
  if (!Rf_isNull(p_oi_sexp)) {
    p_oi = Rcpp::as<int>(p_oi_sexp);
  }
  double oi_prior_sd = zi_prior_sd;  // Default to same as ZI
  SEXP oi_prior_sd_sexp = zi_params["oi_prior_sd"];
  if (!Rf_isNull(oi_prior_sd_sexp)) {
    oi_prior_sd = Rcpp::as<double>(oi_prior_sd_sexp);
  }


  // =========================================================================
  // Set up model data
  // =========================================================================
  ModelData data;

  // Copy response data
  data.y_num = std::vector<int>(y_num.begin(), y_num.end());
  data.y_denom = std::vector<int>(y_denom.begin(), y_denom.end());
  data.y_num_cont = std::vector<double>(y_num_cont.begin(), y_num_cont.end());
  data.y_denom_cont = std::vector<double>(y_denom_cont.begin(), y_denom_cont.end());

  // Flatten design matrices for cache efficiency
  data.p_num = X_num.ncol();
  data.p_denom = X_denom.ncol();
  data.N = y_num.size();

  data.X_num_flat.resize(data.N * data.p_num);
  for (int i = 0; i < data.N; i++) {
    for (int j = 0; j < data.p_num; j++) {
      data.X_num_flat[i * data.p_num + j] = X_num(i, j);
    }
  }

  data.X_denom_flat.resize(data.N * data.p_denom);
  for (int i = 0; i < data.N; i++) {
    for (int j = 0; j < data.p_denom; j++) {
      data.X_denom_flat[i * data.p_denom + j] = X_denom(i, j);
    }
  }

  // Random effects
  apply_re_params(data, re_params);

  // Model type
  if (model_type_str == "binomial") {
    data.model_type = ModelType::BINOMIAL;
  } else if (model_type_str == "negbin_negbin") {
    data.model_type = ModelType::NEGBIN_NEGBIN;
  } else if (model_type_str == "poisson_gamma") {
    data.model_type = ModelType::POISSON_GAMMA;
  } else if (model_type_str == "negbin_gamma") {
    data.model_type = ModelType::NEGBIN_GAMMA;
  } else if (model_type_str == "gamma_gamma") {
    data.model_type = ModelType::GAMMA_GAMMA;
  } else if (model_type_str == "lognormal") {
    data.model_type = ModelType::LOGNORMAL;
  } else if (model_type_str == "beta_binomial") {
    data.model_type = ModelType::BETA_BINOMIAL;
  } else {
    data.model_type = ModelType::POISSON_GAMMA;  // fallback
  }

  // Spatial structure
  if (spatial_type_str == "icar") {
    data.spatial_type = SpatialType::ICAR;
  } else if (spatial_type_str == "bym2") {
    data.spatial_type = SpatialType::BYM2;
  } else if (spatial_type_str == "car_proper") {
    data.spatial_type = SpatialType::CAR_PROPER;
  } else {
    data.spatial_type = SpatialType::NONE;
  }

  data.spatial_group = spatial_group;  // Already deep copied above
  data.n_spatial_units = n_spatial_units;
  data.adj_row_ptr = adj_row_ptr;
  data.adj_col_idx = adj_col_idx;
  data.n_neighbors = n_neighbors;
  data.bym2_scale_factor = bym2_scale_factor;
  data.spatial_Q_inv = std::move(spatial_Q_inv);
  data.spatial_L_Q = std::move(spatial_L_Q);

  if (data.spatial_type == SpatialType::CAR_PROPER) {
    if (spatial_params.containsElementNamed("rho_lower"))
      data.car_rho_lower = Rcpp::as<double>(spatial_params["rho_lower"]);
    if (spatial_params.containsElementNamed("rho_upper"))
      data.car_rho_upper = Rcpp::as<double>(spatial_params["rho_upper"]);
    if (spatial_params.containsElementNamed("rho_prior_a"))
      data.car_rho_prior_a = Rcpp::as<double>(spatial_params["rho_prior_a"]);
    if (spatial_params.containsElementNamed("rho_prior_b"))
      data.car_rho_prior_b = Rcpp::as<double>(spatial_params["rho_prior_b"]);
    if (spatial_params.containsElementNamed("center"))
      data.car_center = Rcpp::as<bool>(spatial_params["center"]);
  }

  // Collapsed ICAR/BYM2 parameterization
  data.icar_collapsed = false;
  data.bym2_collapsed = false;
  if (spatial_params.containsElementNamed("parameterization")) {
      std::string spatial_param_str = Rcpp::as<std::string>(spatial_params["parameterization"]);
      if (spatial_param_str == "collapsed") {
          if (data.spatial_type == SpatialType::ICAR) {
              data.icar_collapsed = true;
          } else if (data.spatial_type == SpatialType::BYM2) {
              data.bym2_collapsed = true;
          }
      }
  }
  if (spatial_params.containsElementNamed("rsr")) {
    apply_rsr_params(data, Rcpp::as<Rcpp::List>(spatial_params["rsr"]));
  }

  // Temporal structure
  apply_temporal_params(data, temporal_params);

  // Zero-inflation structure
  data.zi_type = ratiod_zi::parse_zi_type(zi_type_str);
  // Use explicit p_zi from R (not X_zi.ncol()) because OI-only models
  // pass a 1-column placeholder X_zi but p_zi=0
  {
    SEXP p_zi_sexp = zi_params["p_zi"];
    data.p_zi = (!Rf_isNull(p_zi_sexp)) ? Rcpp::as<int>(p_zi_sexp) : X_zi.ncol();
  }
  data.zi_prior_sd = zi_prior_sd;
  data.X_zi_flat.resize(data.N * data.p_zi);
  for (int i = 0; i < data.N; i++) {
    for (int j = 0; j < data.p_zi; j++) {
      data.X_zi_flat[i * data.p_zi + j] = X_zi(i, j);
    }
  }

  // One-inflation structure (for OI-binomial and ZOIB)
  data.p_oi = p_oi;
  data.oi_prior_sd = oi_prior_sd;
  if (p_oi > 0) {
    data.X_oi_flat.resize(data.N * data.p_oi);
    for (int i = 0; i < data.N; i++) {
      for (int j = 0; j < data.p_oi; j++) {
        data.X_oi_flat[i * data.p_oi + j] = X_oi(i, j);
      }
    }
  }

  // Priors
  data.sigma_beta = sigma_beta;
  data.sigma_re_scale = sigma_re_scale;
  data.phi_prior_shape = phi_prior_shape;
  data.phi_prior_rate = phi_prior_rate;
  data.tau_spatial_shape = tau_spatial_shape;
  data.tau_spatial_rate = tau_spatial_rate;

  // Parallelization
  data.n_threads = n_threads;
  // Across-chain team bound. n_cores == 0 means "size the team to the chain
  // count"; a positive value (the R backend passes the core budget) keeps the
  // multi-chain team the same size across fits.
  data.n_cores = (n_cores > 0) ? n_cores : n_chains;

  // Feature flags this entry never sets, which belong to cpp_hmc_fit_gp.
  // has_multiscale_temporal is not among them: the temporal bundle above sets
  // it for temporal_multiscale(), and compute_param_layout() allocates the
  // trend, seasonal and short-term blocks off it.
  data.has_gp = false;
  data.has_multiscale_gp = false;
  data.has_hsgp = false;

  apply_latent_params(data, latent_params);
  apply_st_params(data, st_params);
  apply_tvc_params(data, tvc_params);
  apply_svc_params(data, svc_params);

  // Initialize parameters
  std::vector<double> q0(q_init.begin(), q_init.end());

  // Memory barrier to ensure all copies complete before HMC execution
  // This prevents R GC from invalidating memory during sampling
  std::atomic_thread_fence(std::memory_order_seq_cst);

  require_autodiff_supported(data, compute_param_layout(data));

  // Run sampler
  if (n_chains == 1) {
    ParamLayout layout = compute_param_layout(data);
    HMCResult result = run_hmc_chain(
      q0, data, layout, n_iter, n_warmup, L, 0, seed, verbose, max_treedepth, metric_type, adapt_delta, walnuts_ptr
    );

    Rcpp::List ret = Rcpp::List::create(
      Rcpp::Named("samples") = result.samples,
      Rcpp::Named("log_prob") = result.log_prob,
      Rcpp::Named("accept_prob") = result.accept_prob,
      Rcpp::Named("n_leapfrog") = result.n_leapfrog,
      Rcpp::Named("treedepth") = result.treedepth,
      Rcpp::Named("divergent") = result.divergent,
      Rcpp::Named("epsilon") = result.epsilon,
      Rcpp::Named("n_warmup") = result.n_warmup,
      Rcpp::Named("n_sample") = result.n_sample,
      Rcpp::Named("n_chains") = 1,
      Rcpp::Named("sampler") = result.sampler.empty()
        ? ((L == 0) ? std::string("NUTS") : std::string("HMC"))
        : result.sampler
    );
    if (result.n_gp_collapsed > 0) {
      ret["gp_w_star"] = result.gp_w_star;
    }
    if (result.n_icar_collapsed > 0) {
      ret["icar_phi_star"] = result.icar_phi_star;
      if (result.bym2_theta_star.nrow() > 0) {
        ret["bym2_theta_star"] = result.bym2_theta_star;
      }
    }
    return ret;
  } else {
    // Multiple chains
    std::vector<HMCResult> results = run_hmc_parallel_chains(
      q0, data, n_iter, n_warmup, L, n_chains, seed, verbose, max_treedepth, metric_type, adapt_delta, walnuts_ptr
    );

    // Combine results
    int n_sample = results[0].n_sample;
    int n_params = results[0].samples.ncol();

    Rcpp::List samples_list(n_chains);
    Rcpp::List log_prob_list(n_chains);
    Rcpp::List accept_prob_list(n_chains);
    Rcpp::List n_leapfrog_list(n_chains);
    Rcpp::List treedepth_list(n_chains);
    Rcpp::List divergent_list(n_chains);
    Rcpp::NumericVector epsilon_vec(n_chains);

    // Determine sampler name: if any chain switched, report it
    std::string sampler_name = (L == 0) ? "NUTS" : "HMC";
    for (int c = 0; c < n_chains; c++) {
      samples_list[c] = results[c].samples;
      log_prob_list[c] = results[c].log_prob;
      accept_prob_list[c] = results[c].accept_prob;
      n_leapfrog_list[c] = results[c].n_leapfrog;
      treedepth_list[c] = results[c].treedepth;
      divergent_list[c] = results[c].divergent;
      epsilon_vec[c] = results[c].epsilon;
      if (!results[c].sampler.empty()) {
        sampler_name = results[c].sampler;
      }
    }

    return Rcpp::List::create(
      Rcpp::Named("samples") = samples_list,
      Rcpp::Named("log_prob") = log_prob_list,
      Rcpp::Named("accept_prob") = accept_prob_list,
      Rcpp::Named("n_leapfrog") = n_leapfrog_list,
      Rcpp::Named("treedepth") = treedepth_list,
      Rcpp::Named("divergent") = divergent_list,
      Rcpp::Named("epsilon") = epsilon_vec,
      Rcpp::Named("n_warmup") = n_warmup,
      Rcpp::Named("n_sample") = n_sample,
      Rcpp::Named("n_chains") = n_chains,
      Rcpp::Named("sampler") = sampler_name
    );
  }
}

// [[Rcpp::export]]
int cpp_get_max_threads() {
  #ifdef _OPENMP
  return omp_get_max_threads();
  #else
  return 1;
  #endif
}

// HMC sampler for GP-based spatial models
// Parameters are bundled into lists to avoid R's .Call argument limit:
//   gp_params: GP spatial parameters
//   ms_gp_params: multiscale GP parameters
//   ms_temporal_params: multiscale temporal parameters
//   rsr_params: RSR parameters
// [[Rcpp::export]]
Rcpp::List cpp_hmc_fit_gp(
    Rcpp::NumericVector q_init,
    Rcpp::IntegerVector y_num,
    Rcpp::IntegerVector y_denom,
    Rcpp::NumericVector y_denom_cont,
    Rcpp::NumericMatrix X_num,
    Rcpp::NumericMatrix X_denom,
    Rcpp::List re_params,
    std::string model_type_str,
    Rcpp::List gp_params,
    Rcpp::List ms_gp_params,
    Rcpp::List ms_temporal_params,
    Rcpp::List rsr_params,
    Rcpp::List temporal_params,  // Regular temporal (RW1/RW2/AR1)
    Rcpp::List latent_params,
    Rcpp::List st_params,
    Rcpp::List tvc_params,
    Rcpp::List svc_params,
    double sigma_beta,
    double sigma_re_scale,
    double phi_prior_shape,
    double phi_prior_rate,
    std::string zi_type_str,
    Rcpp::NumericMatrix X_zi,
    double zi_prior_sd,
    int n_iter,
    int n_warmup,
    int L,
    int n_chains,
    unsigned int seed,
    int n_threads,
    bool verbose,
    int max_treedepth = 10,
    double adapt_delta = -1.0,
    std::string metric_str = "auto",
    std::string gradient_mode_str = "auto"
) {
  using namespace ratiod_hmc;

  // Parse metric and gradient mode from string parameters
  GradientMode grad_mode = parse_gradient_mode(gradient_mode_str);
  set_gradient_mode(grad_mode);
  MassMatrixType metric_type = parse_metric_type(metric_str);
  // Force all Rcpp parameter extractions into eagerly-copied std::vectors FIRST
  // This prevents R garbage collection from invalidating lazy Rcpp views during C++ execution
  // The original debug output workaround worked because I/O forced R to sync; this achieves
  // the same effect through explicit eager copying without visible output.

  // Extract GP parameters - convert to native C++ types immediately
  std::string gp_type_str = Rcpp::as<std::string>(gp_params["gp_type"]);

  // Force eager copy into std::vectors (not Rcpp views that could be GC'd)
  std::vector<double> coords_vec = Rcpp::as<std::vector<double>>(gp_params["coords"]);
  std::vector<int> nn_idx_vec = Rcpp::as<std::vector<int>>(gp_params["nn_idx"]);
  std::vector<double> nn_dist_vec = Rcpp::as<std::vector<double>>(gp_params["nn_dist"]);
  std::vector<int> nn_order_vec = Rcpp::as<std::vector<int>>(gp_params["nn_order"]);
  std::vector<int> nn_order_inv_vec = Rcpp::as<std::vector<int>>(gp_params["nn_order_inv"]);
  std::vector<double> nn_neighbor_dist_vec = Rcpp::as<std::vector<double>>(gp_params["nn_neighbor_dist"]);  // Phase 1.3

  int nn = Rcpp::as<int>(gp_params["nn"]);
  std::string cov_type_str = Rcpp::as<std::string>(gp_params["cov_type"]);
  double nu = Rcpp::as<double>(gp_params["nu"]);
  bool gp_shared = Rcpp::as<bool>(gp_params["shared"]);
  double gp_sigma2_prior_U = Rcpp::as<double>(gp_params["sigma2_prior_U"]);
  double gp_sigma2_prior_alpha = Rcpp::as<double>(gp_params["sigma2_prior_alpha"]);
  double gp_phi_prior_lower = Rcpp::as<double>(gp_params["phi_prior_lower"]);
  double gp_phi_prior_upper = Rcpp::as<double>(gp_params["phi_prior_upper"]);

  // GP solver configuration

  // Observation-to-location mapping (1-based from R, convert to 0-based)
  std::vector<int> gp_obs_to_loc_r = Rcpp::as<std::vector<int>>(gp_params["gp_obs_to_loc"]);
  int gp_n_unique = Rcpp::as<int>(gp_params["n_unique"]);

  // Memory barrier to ensure all extractions complete before proceeding
  std::atomic_thread_fence(std::memory_order_seq_cst);

  // Extract multiscale GP parameters - eager copy to std::vectors
  std::vector<int> nn_idx_local_vec = Rcpp::as<std::vector<int>>(ms_gp_params["nn_idx_local"]);
  std::vector<double> nn_dist_local_vec = Rcpp::as<std::vector<double>>(ms_gp_params["nn_dist_local"]);
  std::vector<int> nn_order_local_vec = Rcpp::as<std::vector<int>>(ms_gp_params["nn_order_local"]);
  std::vector<int> nn_order_inv_local_vec = Rcpp::as<std::vector<int>>(ms_gp_params["nn_order_inv_local"]);
  int nn_local = Rcpp::as<int>(ms_gp_params["nn_local"]);
  std::vector<int> nn_idx_regional_vec = Rcpp::as<std::vector<int>>(ms_gp_params["nn_idx_regional"]);
  std::vector<double> nn_dist_regional_vec = Rcpp::as<std::vector<double>>(ms_gp_params["nn_dist_regional"]);
  std::vector<int> nn_order_regional_vec = Rcpp::as<std::vector<int>>(ms_gp_params["nn_order_regional"]);
  std::vector<int> nn_order_inv_regional_vec = Rcpp::as<std::vector<int>>(ms_gp_params["nn_order_inv_regional"]);
  int nn_regional = Rcpp::as<int>(ms_gp_params["nn_regional"]);
  std::vector<double> nn_neighbor_dist_local_vec = Rcpp::as<std::vector<double>>(ms_gp_params["nn_neighbor_dist_local"]);  // Phase 1.3
  std::vector<double> nn_neighbor_dist_regional_vec = Rcpp::as<std::vector<double>>(ms_gp_params["nn_neighbor_dist_regional"]);  // Phase 1.3
  double range_local_lower = Rcpp::as<double>(ms_gp_params["range_local_lower"]);
  double range_local_upper = Rcpp::as<double>(ms_gp_params["range_local_upper"]);
  double range_regional_lower = Rcpp::as<double>(ms_gp_params["range_regional_lower"]);
  double range_regional_upper = Rcpp::as<double>(ms_gp_params["range_regional_upper"]);
  double ms_sigma2_local_prior_U = Rcpp::as<double>(ms_gp_params["sigma2_local_prior_U"]);
  double ms_sigma2_local_prior_alpha = Rcpp::as<double>(ms_gp_params["sigma2_local_prior_alpha"]);
  double ms_sigma2_regional_prior_U = Rcpp::as<double>(ms_gp_params["sigma2_regional_prior_U"]);
  double ms_sigma2_regional_prior_alpha = Rcpp::as<double>(ms_gp_params["sigma2_regional_prior_alpha"]);
  std::string msgp_sampler_str = Rcpp::as<std::string>(ms_gp_params["sampler"]);

  // Extract multiscale temporal parameters - eager copy
  std::string ms_temporal_type_str = Rcpp::as<std::string>(ms_temporal_params["type"]);
  std::vector<int> ms_time_index_vec = Rcpp::as<std::vector<int>>(ms_temporal_params["time_index"]);
  std::vector<int> ms_group_index_vec = Rcpp::as<std::vector<int>>(ms_temporal_params["group_index"]);
  int ms_n_times = Rcpp::as<int>(ms_temporal_params["n_times"]);
  int ms_n_groups = Rcpp::as<int>(ms_temporal_params["n_groups"]);
  std::string trend_type_str = Rcpp::as<std::string>(ms_temporal_params["trend_type"]);
  int seasonal_period = Rcpp::as<int>(ms_temporal_params["seasonal_period"]);
  std::string short_term_type_str = Rcpp::as<std::string>(ms_temporal_params["short_term_type"]);
  bool ms_temporal_shared = Rcpp::as<bool>(ms_temporal_params["shared"]);
  double ms_sigma2_trend_prior_U = Rcpp::as<double>(ms_temporal_params["sigma2_trend_prior_U"]);
  double ms_sigma2_trend_prior_alpha = Rcpp::as<double>(ms_temporal_params["sigma2_trend_prior_alpha"]);
  double ms_sigma2_seasonal_prior_U = Rcpp::as<double>(ms_temporal_params["sigma2_seasonal_prior_U"]);
  double ms_sigma2_seasonal_prior_alpha = Rcpp::as<double>(ms_temporal_params["sigma2_seasonal_prior_alpha"]);
  double ms_sigma2_short_prior_U = Rcpp::as<double>(ms_temporal_params["sigma2_short_prior_U"]);
  double ms_sigma2_short_prior_alpha = Rcpp::as<double>(ms_temporal_params["sigma2_short_prior_alpha"]);

  // Extract RSR parameters - eager copy
  bool has_rsr = Rcpp::as<bool>(rsr_params["has_rsr"]);
  std::vector<double> rsr_basis_vec = Rcpp::as<std::vector<double>>(rsr_params["basis"]);
  int rsr_rank = Rcpp::as<int>(rsr_params["rank"]);

  // Second memory barrier after all Rcpp extractions
  std::atomic_thread_fence(std::memory_order_seq_cst);
  using namespace ratiod_hmc;

  // Set up model data
  ModelData data;

  // Copy response data
  data.y_num = std::vector<int>(y_num.begin(), y_num.end());
  data.y_denom = std::vector<int>(y_denom.begin(), y_denom.end());
  data.y_denom_cont = std::vector<double>(y_denom_cont.begin(), y_denom_cont.end());
  data.N = y_num.size();

  // Flatten design matrices
  data.p_num = X_num.ncol();
  data.p_denom = X_denom.ncol();

  data.X_num_flat.resize(data.N * data.p_num);
  for (int i = 0; i < data.N; i++) {
    for (int j = 0; j < data.p_num; j++) {
      data.X_num_flat[i * data.p_num + j] = X_num(i, j);
    }
  }

  data.X_denom_flat.resize(data.N * data.p_denom);
  for (int i = 0; i < data.N; i++) {
    for (int j = 0; j < data.p_denom; j++) {
      data.X_denom_flat[i * data.p_denom + j] = X_denom(i, j);
    }
  }

  // Random effects. The same bundle cpp_hmc_fit reads, through the same
  // builder: random slopes and crossed / nested terms are laid out by
  // compute_param_layout() off these fields, and R names the columns off the
  // same structure.
  apply_re_params(data, re_params);

  // Model type
  if (model_type_str == "binomial") {
    data.model_type = ModelType::BINOMIAL;
  } else if (model_type_str == "negbin_negbin") {
    data.model_type = ModelType::NEGBIN_NEGBIN;
  } else if (model_type_str == "poisson_gamma") {
    data.model_type = ModelType::POISSON_GAMMA;
  } else if (model_type_str == "negbin_gamma") {
    data.model_type = ModelType::NEGBIN_GAMMA;
  } else if (model_type_str == "gamma_gamma") {
    data.model_type = ModelType::GAMMA_GAMMA;
  } else if (model_type_str == "lognormal") {
    data.model_type = ModelType::LOGNORMAL;
  } else if (model_type_str == "beta_binomial") {
    data.model_type = ModelType::BETA_BINOMIAL;
  } else {
    data.model_type = ModelType::POISSON_GAMMA;  // fallback
  }

  // Covariance type
  ratiod_gp::CovType cov_type;
  if (cov_type_str == "exponential") {
    cov_type = ratiod_gp::CovType::EXPONENTIAL;
  } else if (cov_type_str == "matern") {
    cov_type = ratiod_gp::CovType::MATERN;
  } else if (cov_type_str == "gaussian") {
    cov_type = ratiod_gp::CovType::GAUSSIAN;
  } else if (cov_type_str == "spherical") {
    cov_type = ratiod_gp::CovType::SPHERICAL;
  } else {
    Rcpp::stop("Unknown covariance type '%s'. Expected one of exponential, "
               "matern, gaussian, spherical.", cov_type_str);
  }

  // GP spatial structure
  if (gp_type_str == "gp") {
    data.spatial_type = SpatialType::GP;
    data.has_gp = true;
    data.has_multiscale_gp = false;
    data.has_hsgp = false;

    data.gp_data.n_obs = gp_n_unique;  // Unique locations, not total observations
    data.gp_data.nn = nn;
    data.gp_data.coords = coords_vec;  // Already std::vector from eager copy
    data.gp_data.nn_idx = nn_idx_vec;
    data.gp_data.nn_dist = nn_dist_vec;
    data.gp_data.nn_neighbor_dist = nn_neighbor_dist_vec;  // Phase 1.3: cached pairwise distances
    // Convert obs_to_loc from R's 1-based to C++'s 0-based indexing
    data.gp_data.obs_to_loc.resize(gp_obs_to_loc_r.size());
    for (size_t i = 0; i < gp_obs_to_loc_r.size(); i++) {
      data.gp_data.obs_to_loc[i] = gp_obs_to_loc_r[i] - 1;
    }
    // Convert from R's 1-based to C++'s 0-based indexing
    data.gp_data.nn_order.resize(nn_order_vec.size());
    for (size_t i = 0; i < nn_order_vec.size(); i++) {
      data.gp_data.nn_order[i] = nn_order_vec[i] - 1;
    }
    data.gp_data.nn_order_inv.resize(nn_order_inv_vec.size());
    for (size_t i = 0; i < nn_order_inv_vec.size(); i++) {
      data.gp_data.nn_order_inv[i] = nn_order_inv_vec[i] - 1;
    }
    data.gp_data.cov_type = cov_type;
    data.gp_data.nu = nu;
    data.gp_data.shared = gp_shared;

    // Set solver configuration

    data.gp_sigma2_prior_U = gp_sigma2_prior_U;
    data.gp_sigma2_prior_alpha = gp_sigma2_prior_alpha;
    data.gp_phi_prior_lower = gp_phi_prior_lower;
    data.gp_phi_prior_upper = gp_phi_prior_upper;

    // GP parameterization: centered (default), noncentered, or collapsed
    if (gp_params.containsElementNamed("parameterization")) {
        std::string gp_param_str = Rcpp::as<std::string>(gp_params["parameterization"]);
        if (gp_param_str == "collapsed") {
            data.gp_parameterization = 0;  // Not relevant for collapsed
            data.gp_collapsed = true;
        } else {
            data.gp_parameterization = (gp_param_str == "centered") ? 0 : 1;
            data.gp_collapsed = false;
        }
    } else {
        data.gp_parameterization = 0;  // Default: centered
        data.gp_collapsed = false;
    }

  } else if (gp_type_str == "multiscale_gp") {
    data.spatial_type = SpatialType::MULTISCALE_GP;
    data.has_gp = false;
    data.has_multiscale_gp = true;
    data.has_hsgp = false;

    // Check if using HSGP approximation
    std::string msgp_approx = "nngp";
    if (gp_params.containsElementNamed("msgp_approx")) {
      msgp_approx = Rcpp::as<std::string>(gp_params["msgp_approx"]);
    }
    data.msgp_is_hsgp = (msgp_approx == "hsgp");

    if (data.msgp_is_hsgp) {
      // HSGP-MSGP: set up shared basis functions, no NNGP neighbor computation
      int hsgp_m = Rcpp::as<int>(gp_params["hsgp_m"]);
      double hsgp_c = Rcpp::as<double>(gp_params["hsgp_c"]);
      ratiod_hsgp::setup_hsgp_2d(coords_vec, data.N, hsgp_m, hsgp_c,
                                  gp_shared, data.msgp_hsgp_data);
      data.multiscale_gp_data.shared = gp_shared;
      // Set n_obs for consistency (used by param layout check)
      data.multiscale_gp_data.n_obs = data.N;

      // Lengthscale prior means from range bounds (geometric mean on log scale)
      data.ms_log_ls_local_mean = 0.5 * (std::log(range_local_lower) + std::log(range_local_upper));
      data.ms_log_ls_local_sd = 0.5;
      data.ms_log_ls_regional_mean = 0.5 * (std::log(range_regional_lower) + std::log(range_regional_upper));
      data.ms_log_ls_regional_sd = 0.5;

      if (verbose) {
        Rcpp::Rcout << "  HSGP-MSGP: m=" << hsgp_m << ", c=" << hsgp_c
                    << ", m_total=" << data.msgp_hsgp_data.m_total
                    << " (local+regional: " << 2 * data.msgp_hsgp_data.m_total << " basis coefficients)\n";
        Rcpp::Rcout << "  Lengthscale priors: local LogN(" << data.ms_log_ls_local_mean
                    << ", " << data.ms_log_ls_local_sd << "), regional LogN("
                    << data.ms_log_ls_regional_mean << ", " << data.ms_log_ls_regional_sd << ")\n";
      }
    } else {
      // NNGP-MSGP: standard neighbor-based computation
      data.multiscale_gp_data.n_obs = gp_n_unique;  // Unique locations, not total observations
      data.multiscale_gp_data.coords = coords_vec;  // Already std::vector from eager copy
      // Convert obs_to_loc from R's 1-based to C++'s 0-based indexing
      data.multiscale_gp_data.obs_to_loc.resize(gp_obs_to_loc_r.size());
      for (size_t i = 0; i < gp_obs_to_loc_r.size(); i++) {
        data.multiscale_gp_data.obs_to_loc[i] = gp_obs_to_loc_r[i] - 1;
      }

      // Local scale - use pre-copied std::vectors
      data.multiscale_gp_data.nn_local = nn_local;
      data.multiscale_gp_data.nn_idx_local = nn_idx_local_vec;
      data.multiscale_gp_data.nn_dist_local = nn_dist_local_vec;
      // Convert from R's 1-based to C++'s 0-based indexing
      data.multiscale_gp_data.nn_order_local.resize(nn_order_local_vec.size());
      for (size_t i = 0; i < nn_order_local_vec.size(); i++) {
        data.multiscale_gp_data.nn_order_local[i] = nn_order_local_vec[i] - 1;
      }
      data.multiscale_gp_data.nn_order_inv_local.resize(nn_order_inv_local_vec.size());
      for (size_t i = 0; i < nn_order_inv_local_vec.size(); i++) {
        data.multiscale_gp_data.nn_order_inv_local[i] = nn_order_inv_local_vec[i] - 1;
      }
      data.multiscale_gp_data.nn_neighbor_dist_local = nn_neighbor_dist_local_vec;  // Phase 1.3

      // Regional scale - use pre-copied std::vectors
      data.multiscale_gp_data.nn_regional = nn_regional;
      data.multiscale_gp_data.nn_idx_regional = nn_idx_regional_vec;
      data.multiscale_gp_data.nn_dist_regional = nn_dist_regional_vec;
      // Convert from R's 1-based to C++'s 0-based indexing
      data.multiscale_gp_data.nn_order_regional.resize(nn_order_regional_vec.size());
      for (size_t i = 0; i < nn_order_regional_vec.size(); i++) {
        data.multiscale_gp_data.nn_order_regional[i] = nn_order_regional_vec[i] - 1;
      }
      data.multiscale_gp_data.nn_order_inv_regional.resize(nn_order_inv_regional_vec.size());
      for (size_t i = 0; i < nn_order_inv_regional_vec.size(); i++) {
        data.multiscale_gp_data.nn_order_inv_regional[i] = nn_order_inv_regional_vec[i] - 1;
      }
      data.multiscale_gp_data.nn_neighbor_dist_regional = nn_neighbor_dist_regional_vec;  // Phase 1.3

      // Range constraints
      data.multiscale_gp_data.range_local_lower = range_local_lower;
      data.multiscale_gp_data.range_local_upper = range_local_upper;
      data.multiscale_gp_data.range_regional_lower = range_regional_lower;
      data.multiscale_gp_data.range_regional_upper = range_regional_upper;

      data.multiscale_gp_data.cov_type = cov_type;
      data.multiscale_gp_data.nu = nu;
      data.multiscale_gp_data.sampler = ratiod_gp::parse_msgp_sampler(msgp_sampler_str);
    }

    data.multiscale_gp_data.shared = gp_shared;

    // Multi-scale GP parameterization: centered (default) or noncentered. The
    // HSGP arm samples basis coefficients, which are already the non-centred
    // coordinate, so msgp_nc() reads this for the two-neighbour-set arm alone.
    {
        std::string msgp_param_str = "centered";
        if (gp_params.containsElementNamed("parameterization")) {
            msgp_param_str = Rcpp::as<std::string>(gp_params["parameterization"]);
        }
        data.msgp_parameterization = (msgp_param_str == "noncentered") ? 1 : 0;
    }

    data.ms_sigma2_local_prior_U = ms_sigma2_local_prior_U;
    data.ms_sigma2_local_prior_alpha = ms_sigma2_local_prior_alpha;
    data.ms_sigma2_regional_prior_U = ms_sigma2_regional_prior_U;
    data.ms_sigma2_regional_prior_alpha = ms_sigma2_regional_prior_alpha;

  } else if (gp_type_str == "hsgp") {
    data.spatial_type = SpatialType::HSGP;
    data.has_gp = false;
    data.has_multiscale_gp = false;
    data.has_hsgp = true;

    // HSGP parameters from gp_params
    int hsgp_m = Rcpp::as<int>(gp_params["hsgp_m"]);
    double hsgp_c = Rcpp::as<double>(gp_params["hsgp_c"]);
    bool hsgp_shared = gp_shared;

    // Setup HSGP data structure with precomputed basis functions
    ratiod_hsgp::setup_hsgp_2d(coords_vec, data.N, hsgp_m, hsgp_c,
                                hsgp_shared, data.hsgp_data);

    data.hsgp_m_per_dim = hsgp_m;
    data.hsgp_boundary_factor = hsgp_c;

  } else {
    data.spatial_type = SpatialType::NONE;
    data.has_gp = false;
    data.has_multiscale_gp = false;
    data.has_hsgp = false;
  }

  // Initialize adjacency for ICAR/BYM2 (not used with GP)
  data.n_spatial_units = 0;
  data.bym2_scale_factor = 1.0;

  // The temporal margin and its prior anchors, from the same builder
  // cpp_hmc_fit reads. It runs ahead of the multi-scale block below, which
  // carries its own dedicated bundle and keeps precedence.
  apply_temporal_params(data, temporal_params);

  // Multi-scale temporal structure
  if (ms_temporal_type_str == "multiscale") {
    data.has_multiscale_temporal = true;

    data.multiscale_temporal_data.n_times = ms_n_times;
    data.multiscale_temporal_data.n_groups = ms_n_groups;
    data.multiscale_temporal_data.n_obs = data.N;
    data.multiscale_temporal_data.time_index = ms_time_index_vec;  // Already std::vector from eager copy
    data.multiscale_temporal_data.group_index = ms_group_index_vec;
    data.multiscale_temporal_data.shared = ms_temporal_shared;
    data.multiscale_temporal_data.seasonal_period = seasonal_period;

    // Parse temporal component types
    data.multiscale_temporal_data.trend_type = ratiod_temporal::parse_temporal_type(trend_type_str);
    data.multiscale_temporal_data.short_term_type = ratiod_temporal::parse_temporal_type(short_term_type_str);

    data.ms_sigma2_trend_prior_U = ms_sigma2_trend_prior_U;
    data.ms_sigma2_trend_prior_alpha = ms_sigma2_trend_prior_alpha;
    data.ms_sigma2_seasonal_prior_U = ms_sigma2_seasonal_prior_U;
    data.ms_sigma2_seasonal_prior_alpha = ms_sigma2_seasonal_prior_alpha;
    data.ms_sigma2_short_prior_U = ms_sigma2_short_prior_U;
    data.ms_sigma2_short_prior_alpha = ms_sigma2_short_prior_alpha;

  } else {
    data.has_multiscale_temporal = false;
    data.multiscale_temporal_data.trend_type = ratiod_temporal::TemporalType::NONE;
    data.multiscale_temporal_data.short_term_type = ratiod_temporal::TemporalType::NONE;
    data.multiscale_temporal_data.seasonal_period = 0;
  }

  set_rsr_basis(data, has_rsr, std::move(rsr_basis_vec), rsr_rank);

  // Zero-inflation structure (GP interface: no OI support, use matrix directly)
  data.zi_type = ratiod_zi::parse_zi_type(zi_type_str);
  data.p_zi = X_zi.ncol();
  data.zi_prior_sd = zi_prior_sd;
  data.X_zi_flat.resize(data.N * data.p_zi);
  for (int i = 0; i < data.N; i++) {
    for (int j = 0; j < data.p_zi; j++) {
      data.X_zi_flat[i * data.p_zi + j] = X_zi(i, j);
    }
  }

  // Standard priors
  data.sigma_beta = sigma_beta;
  data.sigma_re_scale = sigma_re_scale;
  data.phi_prior_shape = phi_prior_shape;
  data.phi_prior_rate = phi_prior_rate;
  data.tau_spatial_shape = 1.0;
  data.tau_spatial_rate = 0.01;

  // The blocks a GP spatial main effect pairs with. compute_param_layout()
  // allocates each of them off these flags and the R backend lays the same
  // blocks over the returned draws, so a block declared inert here is one the
  // sampler never writes and R still names.
  apply_latent_params(data, latent_params);
  apply_st_params(data, st_params);
  apply_tvc_params(data, tvc_params);
  apply_svc_params(data, svc_params);

  // Parallelization
  data.n_threads = n_threads;

  // Final memory barrier before HMC execution
  std::atomic_thread_fence(std::memory_order_seq_cst);

  // Initialize parameters - use explicit std::vector copy from Rcpp
  std::vector<double> q0(q_init.begin(), q_init.end());

  require_autodiff_supported(data, compute_param_layout(data));

  // Run sampler
  if (n_chains == 1) {
    ParamLayout layout = compute_param_layout(data);
    HMCResult result = run_hmc_chain(
      q0, data, layout, n_iter, n_warmup, L, 0, seed, verbose, max_treedepth,
      metric_type, adapt_delta, nullptr
    );

    return Rcpp::List::create(
      Rcpp::Named("samples") = result.samples,
      Rcpp::Named("log_prob") = result.log_prob,
      Rcpp::Named("accept_prob") = result.accept_prob,
      Rcpp::Named("n_leapfrog") = result.n_leapfrog,
      Rcpp::Named("treedepth") = result.treedepth,
      Rcpp::Named("divergent") = result.divergent,
      Rcpp::Named("epsilon") = result.epsilon,
      Rcpp::Named("n_warmup") = result.n_warmup,
      Rcpp::Named("n_sample") = result.n_sample,
      Rcpp::Named("n_chains") = 1,
      Rcpp::Named("sampler") = result.sampler.empty()
        ? ((L == 0) ? std::string("NUTS") : std::string("HMC"))
        : result.sampler
    );
  } else {
    // Multiple chains
    std::vector<HMCResult> results = run_hmc_parallel_chains(
      q0, data, n_iter, n_warmup, L, n_chains, seed, verbose, max_treedepth,
      metric_type, adapt_delta, nullptr
    );

    // Combine results
    int n_sample = results[0].n_sample;
    int n_params = results[0].samples.ncol();

    Rcpp::List samples_list(n_chains);
    Rcpp::List log_prob_list(n_chains);
    Rcpp::List accept_prob_list(n_chains);
    Rcpp::List n_leapfrog_list(n_chains);
    Rcpp::List treedepth_list(n_chains);
    Rcpp::List divergent_list(n_chains);
    Rcpp::NumericVector epsilon_vec(n_chains);

    std::string sampler_name = (L == 0) ? "NUTS" : "HMC";
    for (int c = 0; c < n_chains; c++) {
      samples_list[c] = results[c].samples;
      log_prob_list[c] = results[c].log_prob;
      accept_prob_list[c] = results[c].accept_prob;
      n_leapfrog_list[c] = results[c].n_leapfrog;
      treedepth_list[c] = results[c].treedepth;
      divergent_list[c] = results[c].divergent;
      epsilon_vec[c] = results[c].epsilon;
      if (!results[c].sampler.empty()) {
        sampler_name = results[c].sampler;
      }
    }

    return Rcpp::List::create(
      Rcpp::Named("samples") = samples_list,
      Rcpp::Named("log_prob") = log_prob_list,
      Rcpp::Named("accept_prob") = accept_prob_list,
      Rcpp::Named("n_leapfrog") = n_leapfrog_list,
      Rcpp::Named("treedepth") = treedepth_list,
      Rcpp::Named("divergent") = divergent_list,
      Rcpp::Named("epsilon") = epsilon_vec,
      Rcpp::Named("n_warmup") = n_warmup,
      Rcpp::Named("n_sample") = n_sample,
      Rcpp::Named("n_chains") = n_chains,
      Rcpp::Named("sampler") = sampler_name
    );
  }
}

// [[Rcpp::export]]
Rcpp::List cpp_hmc_fit_gp_v2(Rcpp::List args) {
  // O2-safe interface: single List parameter to minimize Rcpp template instantiation
  // at ABI boundary. All parameter extraction happens inside function body where
  // compiler has full visibility.

  // Extract all parameters from the list - matching cpp_hmc_fit_gp signature
  Rcpp::NumericVector q_init = Rcpp::as<Rcpp::NumericVector>(args["q_init"]);
  Rcpp::IntegerVector y_num = Rcpp::as<Rcpp::IntegerVector>(args["y_num"]);
  Rcpp::IntegerVector y_denom = Rcpp::as<Rcpp::IntegerVector>(args["y_denom"]);
  Rcpp::NumericVector y_denom_cont = Rcpp::as<Rcpp::NumericVector>(args["y_denom_cont"]);
  Rcpp::NumericMatrix X_num = Rcpp::as<Rcpp::NumericMatrix>(args["X_num"]);
  Rcpp::NumericMatrix X_denom = Rcpp::as<Rcpp::NumericMatrix>(args["X_denom"]);
  Rcpp::List re_params = Rcpp::as<Rcpp::List>(args["re_params"]);
  std::string model_type_str = Rcpp::as<std::string>(args["model_type_str"]);
  Rcpp::List gp_params = Rcpp::as<Rcpp::List>(args["gp_params"]);
  Rcpp::List ms_gp_params = Rcpp::as<Rcpp::List>(args["ms_gp_params"]);
  Rcpp::List ms_temporal_params = Rcpp::as<Rcpp::List>(args["ms_temporal_params"]);
  Rcpp::List rsr_params = Rcpp::as<Rcpp::List>(args["rsr_params"]);
  Rcpp::List temporal_params = Rcpp::as<Rcpp::List>(args["temporal_params"]);
  Rcpp::List latent_params = Rcpp::as<Rcpp::List>(args["latent_params"]);
  Rcpp::List st_params = Rcpp::as<Rcpp::List>(args["st_params"]);
  Rcpp::List tvc_params = Rcpp::as<Rcpp::List>(args["tvc_params"]);
  Rcpp::List svc_params = Rcpp::as<Rcpp::List>(args["svc_params"]);
  double sigma_beta = Rcpp::as<double>(args["sigma_beta"]);
  double sigma_re_scale = Rcpp::as<double>(args["sigma_re_scale"]);
  double phi_prior_shape = Rcpp::as<double>(args["phi_prior_shape"]);
  double phi_prior_rate = Rcpp::as<double>(args["phi_prior_rate"]);
  std::string zi_type_str = Rcpp::as<std::string>(args["zi_type_str"]);
  Rcpp::NumericMatrix X_zi = Rcpp::as<Rcpp::NumericMatrix>(args["X_zi"]);
  double zi_prior_sd = Rcpp::as<double>(args["zi_prior_sd"]);
  int n_iter = Rcpp::as<int>(args["n_iter"]);
  int n_warmup = Rcpp::as<int>(args["n_warmup"]);
  int L = Rcpp::as<int>(args["L"]);
  int n_chains = Rcpp::as<int>(args["n_chains"]);
  unsigned int seed = Rcpp::as<unsigned int>(args["seed"]);
  int n_threads = Rcpp::as<int>(args["n_threads"]);
  bool verbose = Rcpp::as<bool>(args["verbose"]);
  int max_treedepth = 10;
  if (args.containsElementNamed("max_treedepth")) {
    max_treedepth = Rcpp::as<int>(args["max_treedepth"]);
  }
  double adapt_delta = -1.0;
  if (args.containsElementNamed("adapt_delta")) {
    adapt_delta = Rcpp::as<double>(args["adapt_delta"]);
  }
  std::string metric_str = "auto";
  if (args.containsElementNamed("metric_str")) {
    metric_str = Rcpp::as<std::string>(args["metric_str"]);
  }
  std::string gradient_mode_str = "auto";
  if (args.containsElementNamed("gradient_mode_str")) {
    gradient_mode_str = Rcpp::as<std::string>(args["gradient_mode_str"]);
  }

  // Delegate to the original implementation (metric/gradient parsed inside)
  return cpp_hmc_fit_gp(
    q_init, y_num, y_denom, y_denom_cont,
    X_num, X_denom, re_params,
    model_type_str, gp_params, ms_gp_params, ms_temporal_params, rsr_params,
    temporal_params, latent_params, st_params, tvc_params, svc_params,
    sigma_beta, sigma_re_scale, phi_prior_shape, phi_prior_rate,
    zi_type_str, X_zi, zi_prior_sd,
    n_iter, n_warmup, L, n_chains, seed, n_threads, verbose, max_treedepth, adapt_delta,
    metric_str, gradient_mode_str
  );
}
