// ess_sampler.cpp
// Rcpp interface for Elliptical Slice Sampling
//
// NOTE: ESS is most efficient for simple models with random effects.
// For models with GP/HSGP spatial effects, HMC is recommended.

#include <Rcpp.h>
#include <RcppEigen.h>
#include "hmc_sampler.h"
#include "ess_sampler.h"
#include "log_post_impl.h"

using namespace Rcpp;
using namespace ratiod_hmc;
using namespace ratiod_ess;

// Implementation of log_post for ESS
namespace ratiod_ess {

double compute_log_post_double(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout
) {
    // Use the templated implementation from log_post_impl.h
    return ratiod::compute_log_post_impl<double>(params, data, layout);
}

} // namespace ratiod_ess

// [[Rcpp::export]]
Rcpp::List cpp_ess_fit(
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
    Rcpp::List tvc_params,
    Rcpp::List svc_params,
    int n_iter,
    int n_warmup,
    unsigned int seed,
    bool verbose
) {
    using namespace ratiod_hmc;

    // =========================================================================
    // Set up model data
    // =========================================================================

    ModelData data = build_model_data(y_num, y_denom, y_num_cont, y_denom_cont,
                                      X_num, X_denom, family, re_params,
                                      spatial_params, temporal_params,
                                      prior_params, zi_params, latent_params,
                                      st_params, tvc_params, svc_params);
    data.n_threads = 1;

    // =========================================================================
    // Compute parameter layout
    // =========================================================================
    ParamLayout layout = compute_param_layout(data);
    int n_params = layout.total_params;

    // =========================================================================
    // Initialize parameters
    // =========================================================================
    std::vector<double> init_params(n_params, 0.0);
    if (q_init.size() == n_params) {
        init_params = Rcpp::as<std::vector<double>>(q_init);
    }

    // =========================================================================
    // Configure ESS
    // =========================================================================
    ESSConfig config;
    config.n_iter = n_iter;
    config.n_warmup = n_warmup;
    config.n_thin = 1;
    config.verbose = verbose;
    config.print_every = 100;
    config.seed = seed;

    // =========================================================================
    // Run ESS sampler
    // =========================================================================
    ESSResult result = run_ess_sampler(init_params, data, layout, config);

    if (!result.success) {
        stop("ESS sampler failed: " + result.error_msg);
    }

    // =========================================================================
    // Build return list
    // =========================================================================
    int n_save = result.samples.rows();

    NumericMatrix samples_out(n_save, n_params);
    for (int i = 0; i < n_save; i++) {
        for (int j = 0; j < n_params; j++) {
            samples_out(i, j) = result.samples(i, j);
        }
    }

    CharacterVector param_names = layout_param_names(data, layout);
    colnames(samples_out) = param_names;

    return List::create(
        Named("samples") = samples_out,
        Named("log_lik") = wrap(result.log_lik),
        Named("n_slice_evals") = result.n_slice_evals,
        Named("avg_slice_evals") = result.avg_slice_evals,
        Named("n_params") = n_params,
        Named("param_names") = param_names
    );
}

