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
    int n_iter,
    int n_warmup,
    unsigned int seed,
    bool verbose
) {
    using namespace ratiod_hmc;

    // =========================================================================
    // Set up model data (similar to cpp_hmc_fit)
    // =========================================================================

    ModelData data;
    apply_response_data(data, y_num, y_denom, y_num_cont, y_denom_cont,
                        X_num, X_denom);
    apply_family_params(data, family);
    const int N = data.N;

    // Random effects
    data.re_group = Rcpp::as<std::vector<int>>(re_params["group"]);
    data.n_re_groups = Rcpp::as<int>(re_params["n_groups"]);
    data.n_re_terms = 0;  // ESS doesn't support multi-term RE yet
    data.total_re_groups = data.n_re_groups;
    data.has_re_slopes = false;
    data.has_re_correlated_slopes = false;
    data.total_re_params = data.n_re_groups;
    data.total_sigma_params = (data.n_re_groups > 0) ? 1 : 0;
    data.total_chol_params = 0;

    // Prior parameters
    data.sigma_beta = Rcpp::as<double>(prior_params["sigma_beta"]);
    data.sigma_re_scale = Rcpp::as<double>(prior_params["sigma_re_scale"]);
    data.phi_prior_shape = Rcpp::as<double>(prior_params["phi_shape"]);
    data.phi_prior_rate = Rcpp::as<double>(prior_params["phi_rate"]);

    // Spatial structure
    std::string spatial_type_str = Rcpp::as<std::string>(spatial_params["type"]);
    if (spatial_type_str == "icar") {
        data.spatial_type = SpatialType::ICAR;
        data.spatial_group = Rcpp::as<std::vector<int>>(spatial_params["group"]);
        data.n_spatial_units = Rcpp::as<int>(spatial_params["n_units"]);
        apply_areal_structure(data, spatial_params);
        data.adj_row_ptr = Rcpp::as<std::vector<int>>(spatial_params["adj_row_ptr"]);
        data.adj_col_idx = Rcpp::as<std::vector<int>>(spatial_params["adj_col_idx"]);
        data.n_neighbors = Rcpp::as<std::vector<int>>(spatial_params["n_neighbors"]);
        data.tau_spatial_shape = Rcpp::as<double>(prior_params["tau_spatial_shape"]);
        data.tau_spatial_rate = Rcpp::as<double>(prior_params["tau_spatial_rate"]);
    } else if (spatial_type_str == "bym2") {
        data.spatial_type = SpatialType::BYM2;
        data.spatial_group = Rcpp::as<std::vector<int>>(spatial_params["group"]);
        data.n_spatial_units = Rcpp::as<int>(spatial_params["n_units"]);
        apply_areal_structure(data, spatial_params);
        data.adj_row_ptr = Rcpp::as<std::vector<int>>(spatial_params["adj_row_ptr"]);
        data.adj_col_idx = Rcpp::as<std::vector<int>>(spatial_params["adj_col_idx"]);
        data.n_neighbors = Rcpp::as<std::vector<int>>(spatial_params["n_neighbors"]);
        data.bym2_scale_factor = Rcpp::as<double>(spatial_params["bym2_scale"]);
        data.tau_spatial_shape = Rcpp::as<double>(prior_params["tau_spatial_shape"]);
        data.tau_spatial_rate = Rcpp::as<double>(prior_params["tau_spatial_rate"]);
    } else {
        data.spatial_type = SpatialType::NONE;
        data.n_spatial_units = 0;
    }

    // Temporal structure
    std::string temporal_type_str = Rcpp::as<std::string>(temporal_params["type"]);
    if (temporal_type_str == "rw1") {
        data.temporal_type = TemporalType::RW1;
    } else if (temporal_type_str == "rw2") {
        data.temporal_type = TemporalType::RW2;
    } else if (temporal_type_str == "ar1") {
        data.temporal_type = TemporalType::AR1;
    } else {
        data.temporal_type = TemporalType::NONE;
    }

    if (data.temporal_type != TemporalType::NONE) {
        data.temporal_time_idx = Rcpp::as<std::vector<int>>(temporal_params["time_idx"]);
        data.temporal_group_idx = Rcpp::as<std::vector<int>>(temporal_params["group_idx"]);
        data.n_times = Rcpp::as<int>(temporal_params["n_times"]);
        data.n_temporal_groups = Rcpp::as<int>(temporal_params["n_groups"]);
        data.n_temporal_params = Rcpp::as<int>(temporal_params["n_params"]);
        data.temporal_cyclic = Rcpp::as<bool>(temporal_params["cyclic"]);
        data.temporal_shared = Rcpp::as<bool>(temporal_params["shared"]);
        data.tau_temporal_shape = Rcpp::as<double>(temporal_params["tau_shape"]);
        data.tau_temporal_rate = Rcpp::as<double>(temporal_params["tau_rate"]);
    } else {
        data.n_times = 0;
        data.n_temporal_groups = 0;
        data.n_temporal_params = 0;
        data.temporal_cyclic = false;
        data.temporal_shared = true;
    }

    // Zero-inflation
    std::string zi_type_str = Rcpp::as<std::string>(zi_params["type"]);
    data.zi_type = ratiod_zi::parse_zi_type(zi_type_str);

    if (data.zi_type != ZIType::NONE) {
        NumericMatrix X_zi = Rcpp::as<NumericMatrix>(zi_params["X"]);
        // Use explicit p_zi from R (not X_zi.ncol()) because OI-only models
        // pass a 1-column placeholder X_zi but p_zi=0
        SEXP p_zi_sexp = zi_params["p_zi"];
        data.p_zi = (!Rf_isNull(p_zi_sexp)) ? Rcpp::as<int>(p_zi_sexp) : X_zi.ncol();
        data.X_zi_flat.resize(N * data.p_zi);
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < data.p_zi; j++) {
                data.X_zi_flat[i * data.p_zi + j] = X_zi(i, j);
            }
        }
        data.zi_prior_sd = Rcpp::as<double>(zi_params["prior_sd"]);
    } else {
        data.p_zi = 0;
        data.zi_prior_sd = 10.0;
    }

    // One-inflation (for OI-binomial and ZOIB)
    data.p_oi = 0;
    SEXP p_oi_sexp = zi_params["p_oi"];
    if (!Rf_isNull(p_oi_sexp)) {
        data.p_oi = Rcpp::as<int>(p_oi_sexp);
    }
    if (data.p_oi > 0) {
        NumericMatrix X_oi = zi_params["X_oi"];
        data.X_oi_flat.resize(N * data.p_oi);
        for (int i = 0; i < N; i++) {
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

    // GP/HSGP not supported
    data.has_gp = false;
    data.has_multiscale_gp = false;
    data.has_hsgp = false;
    data.has_svc = false;
    data.has_rsr = false;

    // Initialize SVC data (empty)
    data.svc_data.n_svc = 0;
    data.svc_data.n_obs = N;
    data.svc_data.nn = 0;
    data.svc_sigma2_prior_scale = 1.0;

    // Multi-scale temporal not supported
    data.has_multiscale_temporal = false;

    // Latent factors not supported
    data.has_latent = false;
    data.latent_n_factors = 0;

    // Spatiotemporal not supported
    data.has_spatiotemporal = false;

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

