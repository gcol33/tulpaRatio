// vi_sampler.cpp
// Variational Inference backend for tulpaRatio
// Main entry point with variant selection and Rcpp exports

#include "hmc_sampler.h"
#include "vi_types.h"
#include "vi_optimizer.h"
#include "vi_meanfield.h"
#include "vi_fullrank.h"
#include "vi_lowrank.h"
#include "autodiff.h"
#include "log_post_impl.h"
#include <Rcpp.h>

using namespace Rcpp;
using namespace ratiod_hmc;

namespace ratiod {
namespace vi {

// =====================================================================
// Bridge function: compute log joint and gradient
// =====================================================================

// This function bridges VI with the existing HMC gradient infrastructure
double compute_log_joint_grad(
    const Eigen::VectorXd& params_eigen,
    const ModelData& data,
    const ParamLayout& layout,
    Eigen::VectorXd& grad_eigen
) {
  int D = params_eigen.size();

  // Convert Eigen to std::vector for HMC interface
  std::vector<double> params(D);
  for (int i = 0; i < D; ++i) {
    params[i] = params_eigen(i);
  }

  // Compute log posterior
  double log_post = compute_log_post(params, data, layout);

  // Compute gradient using the HMC gradient machinery
  std::vector<double> grad(D);
  compute_gradient(params, data, layout, grad);

  // Convert back to Eigen
  grad_eigen.resize(D);
  for (int i = 0; i < D; ++i) {
    grad_eigen(i) = grad[i];
  }

  return log_post;
}

// =====================================================================
// VI Dispatcher
// =====================================================================

VIResult fit_vi(
    const ModelData& data,
    const ParamLayout& layout,
    int D,
    const VIConfig& config,
    const Eigen::VectorXd* init_mu = nullptr
) {
  // Select variant
  VIVariant variant = select_variant(D, config);

  if (config.verbose) {
    Rcpp::Rcout << "VI variant: " << variant_to_string(variant)
                << " (D=" << D << ")\n";
  }

  // Dispatch to appropriate implementation
  switch (variant) {
    case VIVariant::MEANFIELD:
      return fit_meanfield(data, layout, D, config, init_mu);

    case VIVariant::LOWRANK:
      return fit_lowrank(data, layout, D, config, init_mu);

    case VIVariant::FULLRANK:
      return fit_fullrank(data, layout, D, config, init_mu, nullptr);

    default:
      Rcpp::stop("Unknown VI variant");
  }

  return VIResult();  // Never reached
}

} // namespace vi
} // namespace ratiod

// =====================================================================
// Rcpp Exports
// =====================================================================

// [[Rcpp::export]]
Rcpp::List cpp_vi_fit(
    Rcpp::NumericVector q_init,
    Rcpp::IntegerVector y_num,
    Rcpp::IntegerVector y_denom,
    Rcpp::NumericVector y_num_cont,
    Rcpp::NumericVector y_denom_cont,
    Rcpp::NumericMatrix X_num,
    Rcpp::NumericMatrix X_denom,
    Rcpp::List family,
    Rcpp::List re_params,
    Rcpp::List spatial_params,
    Rcpp::List temporal_params,
    Rcpp::List prior_params,
    Rcpp::List zi_params,
    Rcpp::List latent_params,
    Rcpp::List st_params,
    Rcpp::List vi_options,
    bool verbose = true
) {
  using namespace ratiod_hmc;
  using namespace ratiod::vi;

  // Parse VI config
  VIConfig config = parse_vi_config(vi_options);
  config.verbose = verbose;

  ModelData data;
  apply_response_data(data, y_num, y_denom, y_num_cont, y_denom_cont,
                      X_num, X_denom);
  apply_family_params(data, family);

  // Random effects
  Rcpp::IntegerVector re_group = re_params["group"];
  data.n_re_groups = Rcpp::as<int>(re_params["n_groups"]);
  data.re_group.resize(data.N);
  for (int i = 0; i < data.N; i++) {
    data.re_group[i] = re_group[i];
  }

  // Multi-term RE support
  data.n_re_terms = Rcpp::as<int>(re_params["n_terms"]);
  data.total_re_groups = data.n_re_groups;
  data.has_re_slopes = false;
  data.has_re_correlated_slopes = false;
  data.total_re_params = data.n_re_groups;
  data.total_sigma_params = (data.n_re_groups > 0) ? 1 : 0;
  data.total_chol_params = 0;

  // Spatial structure
  std::string spatial_type = Rcpp::as<std::string>(spatial_params["type"]);
  if (spatial_type == "none") {
    data.spatial_type = SpatialType::NONE;
  } else if (spatial_type == "icar") {
    data.spatial_type = SpatialType::ICAR;
  } else if (spatial_type == "bym2") {
    data.spatial_type = SpatialType::BYM2;
  } else {
    data.spatial_type = SpatialType::NONE;
  }

  Rcpp::IntegerVector spatial_group = spatial_params["group"];
  data.n_spatial_units = Rcpp::as<int>(spatial_params["n_units"]);
  data.spatial_group.resize(data.N);
  for (int i = 0; i < data.N; i++) {
    data.spatial_group[i] = spatial_group[i];
  }

  if (data.spatial_type == SpatialType::ICAR || data.spatial_type == SpatialType::BYM2) {
    Rcpp::IntegerVector adj_row_ptr = spatial_params["adj_row_ptr"];
    Rcpp::IntegerVector adj_col_idx = spatial_params["adj_col_idx"];
    Rcpp::IntegerVector n_neighbors = spatial_params["n_neighbors"];

    data.adj_row_ptr.resize(adj_row_ptr.size());
    data.adj_col_idx.resize(adj_col_idx.size());
    data.n_neighbors.resize(n_neighbors.size());

    for (int i = 0; i < adj_row_ptr.size(); i++) data.adj_row_ptr[i] = adj_row_ptr[i];
    for (int i = 0; i < adj_col_idx.size(); i++) data.adj_col_idx[i] = adj_col_idx[i];
    for (int i = 0; i < n_neighbors.size(); i++) data.n_neighbors[i] = n_neighbors[i];

    data.bym2_scale_factor = Rcpp::as<double>(spatial_params["bym2_scale"]);
  }

  // Temporal structure
  std::string temporal_type_str = Rcpp::as<std::string>(temporal_params["type"]);
  if (temporal_type_str == "none") {
    data.temporal_type = TemporalType::NONE;
  } else if (temporal_type_str == "rw1") {
    data.temporal_type = TemporalType::RW1;
  } else if (temporal_type_str == "rw2") {
    data.temporal_type = TemporalType::RW2;
  } else if (temporal_type_str == "ar1") {
    data.temporal_type = TemporalType::AR1;
  } else {
    data.temporal_type = TemporalType::NONE;
  }

  Rcpp::IntegerVector temporal_time_idx = temporal_params["time_idx"];
  Rcpp::IntegerVector temporal_group_idx = temporal_params["group_idx"];
  data.n_times = Rcpp::as<int>(temporal_params["n_times"]);
  data.n_temporal_groups = Rcpp::as<int>(temporal_params["n_groups"]);
  data.n_temporal_params = Rcpp::as<int>(temporal_params["n_params"]);
  data.temporal_cyclic = Rcpp::as<bool>(temporal_params["cyclic"]);
  data.temporal_shared = Rcpp::as<bool>(temporal_params["shared"]);
  data.tau_temporal_shape = Rcpp::as<double>(temporal_params["tau_shape"]);
  data.tau_temporal_rate = Rcpp::as<double>(temporal_params["tau_rate"]);

  data.temporal_time_idx.resize(data.N);
  data.temporal_group_idx.resize(data.N);
  for (int i = 0; i < data.N; i++) {
    data.temporal_time_idx[i] = temporal_time_idx[i];
    data.temporal_group_idx[i] = temporal_group_idx[i];
  }

  // Zero-inflation: use the canonical parser (ratiod_zi::parse_zi_type) so
  // VI dispatches on the same fully-qualified strings ("zi_poisson",
  // "zi_negbin", "zi_binomial", "hurdle_binomial", "oi_binomial", "zoib",
  // ...) as the HMC/ESS/SGHMC backends, instead of a separate, lossy parser.
  std::string zi_type_str = Rcpp::as<std::string>(zi_params["type"]);
  data.zi_type = ratiod_zi::parse_zi_type(zi_type_str);

  if (data.zi_type != ratiod_zi::ZIType::NONE) {
    Rcpp::NumericMatrix X_zi = zi_params["X"];
    // Use explicit p_zi from R (not X_zi.ncol()) because OI-only models
    // pass a 1-column placeholder X_zi but p_zi=0
    SEXP p_zi_sexp = zi_params["p_zi"];
    data.p_zi = (!Rf_isNull(p_zi_sexp)) ? Rcpp::as<int>(p_zi_sexp) : X_zi.ncol();
    data.X_zi_flat.resize(data.N * data.p_zi);
    for (int i = 0; i < data.N; i++) {
      for (int j = 0; j < data.p_zi; j++) {
        data.X_zi_flat[i * data.p_zi + j] = X_zi(i, j);
      }
    }
  } else {
    data.p_zi = 0;
  }
  data.zi_prior_sd = Rcpp::as<double>(zi_params["prior_sd"]);

  // One-inflation (for OI-binomial and ZOIB)
  data.p_oi = 0;
  SEXP p_oi_sexp = zi_params["p_oi"];
  if (!Rf_isNull(p_oi_sexp)) {
    data.p_oi = Rcpp::as<int>(p_oi_sexp);
  }
  if (data.p_oi > 0) {
    Rcpp::NumericMatrix X_oi = zi_params["X_oi"];
    data.X_oi_flat.resize(data.N * data.p_oi);
    for (int i = 0; i < data.N; i++) {
      for (int j = 0; j < data.p_oi; j++) {
        data.X_oi_flat[i * data.p_oi + j] = X_oi(i, j);
      }
    }
  }
  data.oi_prior_sd = data.zi_prior_sd;
  SEXP oi_prior_sd_sexp = zi_params["oi_prior_sd"];
  if (!Rf_isNull(oi_prior_sd_sexp)) {
    data.oi_prior_sd = Rcpp::as<double>(oi_prior_sd_sexp);
  }

  // Latent factors
  data.has_latent = Rcpp::as<bool>(latent_params["has_latent"]);
  data.latent_n_factors = Rcpp::as<int>(latent_params["n_factors"]);
  data.latent_shared = Rcpp::as<bool>(latent_params["shared"]);
  data.latent_scale = Rcpp::as<bool>(latent_params["scale"]);
  data.latent_constraint = Rcpp::as<int>(latent_params["constraint"]);
  data.latent_sigma_prior_rate = Rcpp::as<double>(latent_params["sigma_prior_rate"]);

  // Spatiotemporal
  data.has_spatiotemporal = Rcpp::as<bool>(st_params["has_spatiotemporal"]);

  // Prior parameters
  data.sigma_beta = Rcpp::as<double>(prior_params["sigma_beta"]);
  data.sigma_re_scale = Rcpp::as<double>(prior_params["sigma_re_scale"]);
  data.phi_prior_shape = Rcpp::as<double>(prior_params["phi_shape"]);
  data.phi_prior_rate = Rcpp::as<double>(prior_params["phi_rate"]);
  data.tau_spatial_shape = Rcpp::as<double>(prior_params["tau_spatial_shape"]);
  data.tau_spatial_rate = Rcpp::as<double>(prior_params["tau_spatial_rate"]);

  // Initialize flags for non-supported features
  data.has_svc = false;
  data.has_gp = false;
  data.has_multiscale_gp = false;
  data.has_hsgp = false;
  data.has_rsr = false;
  data.has_multiscale_temporal = false;
  data.n_threads = 1;

  // Initialize SVC data (empty)
  data.svc_data.n_svc = 0;
  data.svc_data.n_obs = data.N;
  data.svc_data.nn = 0;
  data.svc_sigma2_prior_scale = 1.0;

  // Compute parameter layout
  ParamLayout layout = compute_param_layout(data);
  const int D = layout.total_params;

  // Initial values: q_init when it matches the layout, zeros otherwise
  Eigen::VectorXd init_mu = Eigen::VectorXd::Zero(D);
  if (q_init.size() == D) {
    for (int i = 0; i < D; ++i) init_mu(i) = q_init[i];
  }

  // Set gradient mode to forward autodiff (fastest general-purpose)
  set_gradient_mode(GradientMode::AUTO);

  // Fit VI
  VIResult result = fit_vi(data, layout, D, config, &init_mu);

  // Return results as R list
  Rcpp::List out = vi_result_to_list(result);
  out["param_names"] = layout_param_names(data, layout);
  return out;
}

