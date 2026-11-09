#include <RcppEigen.h>
#include "hmc_sampler.h"

using namespace Rcpp;

namespace ratiod_hmc {

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

  // Overdispersion / shape / sigma parameters: a negative binomial
  // overdispersion, a gamma shape, a log-normal SD or a beta-binomial
  // precision, as model_type_info() lists them
  const ModelTypeInfo type_info = model_type_info(data.model_type);
  layout.has_phi_num = type_info.phi_num;
  layout.has_phi_denom = type_info.phi_denom;

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
      layout.logit_rho_car_idx = -1;
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
    // Basis coefficients per scale
    const int n_per_scale = data.msgp_hsgp_data.m_total;
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
    layout.range_svc_start = idx;
    idx += data.svc_data.n_svc;
    layout.range_svc_end = idx;

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
    layout.range_svc_start = layout.range_svc_end = -1;
    layout.svc_w_start = layout.svc_w_end = -1;
  }

  // Latent factors for unmeasured confounders
  layout.has_latent = data.has_latent;
  if (layout.has_latent && data.latent_n_factors > 0) {
    // Lower-triangular loadings (hmc_latent.h)
    layout.latent_loading_start = idx;
    idx += ratiod_latent::n_loadings(
        ratiod_latent::n_processes(data.latent_shared), data.latent_n_factors);
    layout.latent_loading_end = idx;

    // Factor scores (N x K)
    layout.latent_factor_start = idx;
    idx += data.N * data.latent_n_factors;
    layout.latent_factor_end = idx;
  } else {
    layout.latent_loading_start = layout.latent_loading_end = -1;
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

    // Spatiotemporal interaction effects. A collapsed GP interaction is
    // integrated out (st_gp_laplace.h) and has no slots.
    layout.st_collapsed = layout.is_st_gp && data.st_parameterization == 2;
    if (layout.st_collapsed) {
      layout.st_delta_start = layout.st_delta_end = -1;
    } else {
      layout.st_delta_start = idx;
      idx += data.spatiotemporal_data.n_params;
      layout.st_delta_end = idx;
    }
  } else {
    layout.st_collapsed = false;
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

ModelType parse_model_type(const std::string& model_type_str) {
  if (model_type_str == "binomial")      return ModelType::BINOMIAL;
  if (model_type_str == "negbin_negbin") return ModelType::NEGBIN_NEGBIN;
  if (model_type_str == "poisson_gamma") return ModelType::POISSON_GAMMA;
  if (model_type_str == "negbin_gamma")  return ModelType::NEGBIN_GAMMA;
  if (model_type_str == "gamma_gamma")   return ModelType::GAMMA_GAMMA;
  if (model_type_str == "lognormal")     return ModelType::LOGNORMAL;
  if (model_type_str == "beta_binomial") return ModelType::BETA_BINOMIAL;
  if (model_type_str == "poisson_fixed") return ModelType::POISSON_FIXED;
  if (model_type_str == "negbin_fixed")  return ModelType::NEGBIN_FIXED;
  if (model_type_str == "lognormal_fixed") return ModelType::LOGNORMAL_FIXED;
  if (model_type_str == "poisson_negbin") return ModelType::POISSON_NEGBIN;
  Rcpp::stop("Unknown model type: " + model_type_str);
}

void apply_family_params(ModelData& data, Rcpp::List family) {
  data.model_type = parse_model_type(Rcpp::as<std::string>(family["model_type"]));
  const bool prob_num = data.model_type == ModelType::BINOMIAL ||
                        data.model_type == ModelType::BETA_BINOMIAL;
  data.link_num = prob_num ? LinkType::LOGIT : LinkType::LOG;
  data.link_denom = LinkType::LOG;
  data.link_zi = LinkType::LOGIT;
  data.link_oi = LinkType::LOGIT;

  auto read_link = [&family](const char* name, LinkType& out) {
    if (!family.containsElementNamed(name)) return;
    SEXP value = family[name];
    if (Rf_isNull(value)) return;
    const std::string link = Rcpp::as<std::string>(value);
    if (!ratiod_link::parse_link(link, out)) {
      Rcpp::stop(std::string("Unknown link '") + link + "' for " + name);
    }
  };
  read_link("link_num", data.link_num);
  read_link("link_denom", data.link_denom);
  read_link("link_zi", data.link_zi);
  read_link("link_oi", data.link_oi);
}

void apply_response_data(ModelData& data,
                         const Rcpp::IntegerVector& y_num,
                         const Rcpp::IntegerVector& y_denom,
                         const Rcpp::NumericVector& y_num_cont,
                         const Rcpp::NumericVector& y_denom_cont,
                         const Rcpp::NumericMatrix& X_num,
                         const Rcpp::NumericMatrix& X_denom) {
  const int N = y_num.size();
  if (y_denom.size() != N || y_num_cont.size() != N ||
      y_denom_cont.size() != N || X_num.nrow() != N || X_denom.nrow() != N) {
    Rcpp::stop("Each response and design matrix needs one entry per observation.");
  }
  data.N = N;
  data.p_num = X_num.ncol();
  data.p_denom = X_denom.ncol();
  data.y_num.assign(y_num.begin(), y_num.end());
  data.y_denom.assign(y_denom.begin(), y_denom.end());
  data.y_num_cont.assign(y_num_cont.begin(), y_num_cont.end());
  data.y_denom_cont.assign(y_denom_cont.begin(), y_denom_cont.end());

  // Row-major, so one observation's covariates are contiguous
  data.X_num_flat.resize(static_cast<size_t>(N) * data.p_num);
  data.X_denom_flat.resize(static_cast<size_t>(N) * data.p_denom);
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < data.p_num; j++) {
      data.X_num_flat[i * data.p_num + j] = X_num(i, j);
    }
    for (int j = 0; j < data.p_denom; j++) {
      data.X_denom_flat[i * data.p_denom + j] = X_denom(i, j);
    }
  }
}

Rcpp::CharacterVector layout_param_names(const ModelData& data,
                                         const ParamLayout& layout) {
  if (layout.is_gp || layout.is_multiscale_gp || layout.is_hsgp ||
      layout.has_svc || layout.has_tvc || layout.has_latent ||
      layout.has_spatiotemporal || layout.has_multiscale_temporal ||
      layout.has_re_slopes || data.n_re_terms > 1) {
    Rcpp::stop("layout_param_names() names the blocks of the ESS, SGHMC, "
               "SGLD and VI samplers only");
  }
  // Each name is written at the index the layout assigns, so the names follow
  // the parameter vector whatever order the blocks take in it
  std::vector<std::string> names(layout.total_params);
  auto scalar = [&names](int idx, const char* name) {
    if (idx >= 0) names[idx] = name;
  };
  auto indexed = [&names](const char* stem, int start, int end) {
    for (int i = start; i >= 0 && i < end; i++) {
      names[i] = std::string(stem) + "[" + std::to_string(i - start + 1) + "]";
    }
  };

  indexed("beta_num", layout.beta_num_start, layout.beta_num_end);
  indexed("beta_denom", layout.beta_denom_start, layout.beta_denom_end);
  if (layout.has_re) {
    scalar(layout.log_sigma_re_idx, "log_sigma_re");
    indexed("re", layout.re_start, layout.re_end);
  }
  if (layout.has_phi_num) scalar(layout.log_phi_num_idx, "log_phi_num");
  if (layout.has_phi_denom) scalar(layout.log_phi_denom_idx, "log_phi_denom");
  if (layout.has_spatial) {
    scalar(layout.log_tau_spatial_idx, "log_tau_spatial");
    scalar(layout.logit_rho_car_idx, "logit_rho_car");
    scalar(layout.log_sigma_bym2_idx, "log_sigma_bym2");
    scalar(layout.logit_rho_bym2_idx, "logit_rho_bym2");
    indexed("phi_spatial", layout.spatial_start, layout.spatial_end);
    indexed("theta_bym2", layout.theta_bym2_start, layout.theta_bym2_end);
  }
  if (layout.has_temporal) {
    scalar(layout.log_tau_temporal_idx, "log_tau_temporal");
    scalar(layout.logit_rho_ar1_idx, "logit_rho_ar1");
    scalar(layout.log_sigma2_temporal_gp_idx, "log_sigma2_temporal_gp");
    scalar(layout.logit_phi_temporal_gp_idx, "logit_phi_temporal_gp");
    indexed("phi_temporal", layout.temporal_start, layout.temporal_end);
  }
  if (layout.has_zi) indexed("beta_zi", layout.beta_zi_start, layout.beta_zi_end);
  if (layout.has_oi) indexed("beta_oi", layout.beta_oi_start, layout.beta_oi_end);

  for (int i = 0; i < layout.total_params; i++) {
    if (names[i].empty()) {
      Rcpp::stop("layout_param_names(): parameter " + std::to_string(i + 1) +
                 " of " + std::to_string(layout.total_params) + " has no name");
    }
  }
  return Rcpp::wrap(names);
}

void apply_areal_structure(ModelData& data, Rcpp::List spatial_params) {
  data.areal_components = ratiod_constraints::ArealComponents();
  data.areal_node_prec.clear();
  if (spatial_params.containsElementNamed("component")) {
    SEXP comp = spatial_params["component"];
    if (!Rf_isNull(comp)) {
      const std::vector<int> block = Rcpp::as<std::vector<int>>(comp);
      int n_blocks = 0;
      for (int v : block) n_blocks = std::max(n_blocks, v + 1);
      data.areal_components.block_of = block;
      data.areal_components.members.assign(n_blocks, std::vector<int>());
      for (int u = 0; u < static_cast<int>(block.size()); u++) {
        if (block[u] >= 0) data.areal_components.members[block[u]].push_back(u);
      }
    }
  }
  if (spatial_params.containsElementNamed("node_prec")) {
    SEXP np = spatial_params["node_prec"];
    if (!Rf_isNull(np)) data.areal_node_prec = Rcpp::as<std::vector<double>>(np);
  }
}

int get_n_params(const ModelData& data) {
  ParamLayout layout = compute_param_layout(data);
  return layout.total_params;
}


}  // namespace ratiod_hmc
