// vi_sampler.cpp
// Variational Inference backend for tulpaRatio
// Main entry point with variant selection and Rcpp exports

#include "hmc_sampler.h"
#include "hmc_model_data_blocks.h"
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
    Rcpp::List gp_params,
    Rcpp::List temporal_params,
    Rcpp::List prior_params,
    Rcpp::List zi_params,
    Rcpp::List latent_params,
    Rcpp::List st_params,
    Rcpp::List tvc_params,
    Rcpp::List svc_params,
    Rcpp::List vi_options,
    bool verbose = true
) {
  using namespace ratiod_hmc;
  using namespace ratiod::vi;

  // Parse VI config
  VIConfig config = parse_vi_config(vi_options);
  config.verbose = verbose;

  ModelData data = build_model_data(y_num, y_denom, y_num_cont, y_denom_cont,
                                    X_num, X_denom, family, re_params,
                                    spatial_params, gp_params, temporal_params,
                                    prior_params, zi_params, latent_params,
                                    st_params, tvc_params, svc_params);
  data.n_threads = 1;

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

