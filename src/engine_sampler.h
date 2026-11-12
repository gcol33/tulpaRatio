#ifndef TULPARATIO_ENGINE_SAMPLER_H
#define TULPARATIO_ENGINE_SAMPLER_H

#include <RcppEigen.h>
#include <tulpa/ess_api.h>
#include <tulpa/sghmc_api.h>
#include <tulpa/vi_api.h>
#include <tulpa/model_point_api.h>
#include <tulpa/field_range.h>
#include "covariance.h"

namespace tulpaRatio {

template<class Result>
struct EngineResult {
    Result value{};
    ~EngineResult() { value.free_buffers(); }
};

template<class T>
T sampler_option(const Rcpp::List& options, const char* name, T default_value) {
    return options.containsElementNamed(name) ? Rcpp::as<T>(options[name]) : default_value;
}

inline Rcpp::List optional_list(const Rcpp::Nullable<Rcpp::List>& value) {
    return value.isNotNull() ? Rcpp::List(value) : Rcpp::List::create();
}

inline Rcpp::NumericMatrix engine_matrix(const double* x, int nr, int nc) {
    Rcpp::NumericMatrix out(nr, nc);
    if (x) for (int i = 0; i < nr; ++i) for (int j = 0; j < nc; ++j)
        out(i,j) = x[static_cast<size_t>(i) * nc + j];
    return out;
}

inline Rcpp::NumericVector engine_vector(const double* x, int n) {
    return x ? Rcpp::NumericVector(x, x + n) : Rcpp::NumericVector(n);
}

inline void restore_engine_ranges(Rcpp::NumericMatrix& draws, const tulpa::ModelData& data,
                                  const tulpa::ParamLayout& layout) {
    const auto restore = [&](int index, double lower, double upper, double scale) {
        for (int i = 0; i < draws.nrow(); ++i)
            draws(i,index) = std::log(tulpa::field_range_from_coordinate(
                draws(i,index), lower, upper, true) / scale);
    };
    if (data.has_svc && data.svc_range_prior == 1)
        for (int j = 0; j < data.svc_data.n_svc; ++j)
            restore(layout.log_phi_svc_start + j, data.svc_range_lower, data.svc_range_upper,
                ratiod_cov::range_scale(data.svc_data.cov_type));
    if (layout.is_st_gp && data.st_uniform_ranges) {
        const auto& st = data.spatiotemporal_data;
        restore(layout.log_phi_st_space_idx, st.phi_space_prior_lower, st.phi_space_prior_upper, 1.0);
        restore(layout.log_phi_st_time_idx, st.phi_time_prior_lower, st.phi_time_prior_upper, 1.0);
    }
}

inline void store_engine_draws(Rcpp::NumericMatrix& samples, const tulpa::ModelData& data,
                               const tulpa::ParamLayout& layout) {
    std::vector<double> point(layout.total_params), stored(layout.total_params);
    const auto evaluate = tulpa::get_model_point_fn();
    for (int i = 0; i < samples.nrow(); ++i) {
        for (int j = 0; j < layout.total_params; ++j) point[j] = samples(i,j);
        tulpa::ModelPointRequest request;
        request.position = point.data(); request.n_params = layout.total_params;
        request.stored = stored.data();
        evaluate(&data, &layout, &request);
        for (int j = 0; j < layout.total_params; ++j) samples(i,j) = stored[j];
    }
    restore_engine_ranges(samples, data, layout);
}

inline Rcpp::List sample_engine(const std::string& method, const Rcpp::List& options,
    const tulpa::ModelData& data, const tulpa::ParamLayout& layout,
    const std::vector<double>& initial, int n_iter, int n_warmup, int seed, bool verbose) {
    const int p = layout.total_params;
    if (method == "evaluate") {
        double value = 0, prior = 0;
        std::vector<double> gradient(p), stored(p);
        tulpa::ModelPointRequest request;
        request.position = initial.data(); request.n_params = p;
        request.value = &value; request.gradient = gradient.data();
        request.stored = stored.data(); request.prior = &prior;
        tulpa::get_model_point_fn()(&data, &layout, &request);
        return Rcpp::List::create(Rcpp::Named("log_post") = value,
            Rcpp::Named("gradient") = gradient, Rcpp::Named("stored") = stored,
            Rcpp::Named("log_prior") = prior);
    }
    if (method == "ess") {
        auto config = tulpa::default_ess_config();
        config.n_iter = n_iter; config.n_warmup = n_warmup; config.verbose = verbose;
        config.joint_sigma_re = layout.has_re;
        Rcpp::Function set_seed = Rcpp::Environment::base_env()["set.seed"];
        set_seed(seed);
        EngineResult<tulpa::ESSShimResult> owner;
        auto& result = owner.value;
        tulpa::get_ess_fn()(initial.data(), p, &data, &layout, &config, &result);
        if (!result.success) Rcpp::stop("ESS: %s", result.error_msg);
        auto samples = engine_matrix(result.samples, result.n_save, p);
        store_engine_draws(samples, data, layout);
        return Rcpp::List::create(Rcpp::Named("samples") = samples,
            Rcpp::Named("log_lik") = engine_vector(result.log_lik, result.n_save),
            Rcpp::Named("n_slice_evals") = result.n_slice_evals,
            Rcpp::Named("avg_slice_evals") = result.avg_slice_evals,
            Rcpp::Named("n_params") = p);
    }
    if (method == "sghmc" || method == "sgld") {
        EngineResult<tulpa::SGSamplerShimResult> owner;
        auto& result = owner.value;
        const int batch = sampler_option(options, "batch_size", data.N);
        if (batch != data.N)
            Rcpp::stop("Minibatch %s is unsupported by the engine; batch_size must equal n_obs",method.c_str());
        const double epsilon = sampler_option(options, "epsilon", .01);
        if (method == "sghmc")
            tulpa::get_sghmc_fit_fn()(&data, &layout, initial.data(), p, n_iter, n_warmup,
              batch, epsilon, sampler_option(options, "alpha", .1), sampler_option(options, "L", 10),
              seed, 1, 100.0, verbose, &result);
        else
            tulpa::get_sgld_fit_fn()(&data, &layout, initial.data(), p, n_iter, n_warmup,
              batch, epsilon, sampler_option(options, "schedule_a", .01),
              sampler_option(options, "schedule_b", 1.0), sampler_option(options, "schedule_gamma", .55),
              sampler_option(options, "use_schedule", true), 100.0, seed, verbose, &result);
        if (!result.success) Rcpp::stop("%s: %s", method.c_str(), result.error_msg);
        auto samples = engine_matrix(result.samples, result.n_sample, p);
        store_engine_draws(samples, data, layout);
        return Rcpp::List::create(Rcpp::Named("samples") = samples,
            Rcpp::Named("log_lik") = engine_vector(result.log_lik, result.n_sample),
            Rcpp::Named("epsilon_history") = engine_vector(result.epsilon_history, result.n_eps_history),
            Rcpp::Named("final_epsilon") = result.final_epsilon,
            Rcpp::Named("batch_size") = batch, Rcpp::Named("n_params") = p);
    }
    if (method == "vi") {
        auto config = tulpa::default_vi_config();
        const std::string variant = sampler_option(options, "variant", std::string("auto"));
        config.variant = tulpa::vi::parse_variant(variant);
        config.max_iter = sampler_option(options, "max_iter", config.max_iter);
        config.mc_samples = sampler_option(options, "mc_samples", config.mc_samples);
        config.tol_grad = sampler_option(options, "tol_grad", config.tol_grad);
        config.tol_rel_elbo = sampler_option(options, "tol_rel_elbo", config.tol_rel_elbo);
        config.patience = sampler_option(options, "patience", config.patience);
        config.adam_alpha = sampler_option(options, "adam_alpha", config.adam_alpha);
        config.adam_beta1 = sampler_option(options, "adam_beta1", config.adam_beta1);
        config.adam_beta2 = sampler_option(options, "adam_beta2", config.adam_beta2);
        config.adam_eps = sampler_option(options, "adam_eps", config.adam_eps);
        config.rank = sampler_option(options, "rank", config.rank);
        config.use_laplace_init = sampler_option(options, "use_laplace_init", true);
        config.verbose = verbose; config.seed = seed;
        EngineResult<tulpa::VIShimResult> owner;
        auto& result = owner.value;
        tulpa::get_vi_fit_fn()(&data, &layout, p, &config, initial.data(), p, &result);
        auto samples = engine_matrix(result.samples, result.n_samples, p);
        store_engine_draws(samples, data, layout);
        return Rcpp::List::create(Rcpp::Named("variant") = tulpa::vi::variant_to_string(result.variant_used),
            Rcpp::Named("mu") = engine_vector(result.mu, p),
            Rcpp::Named("Sigma") = engine_matrix(result.Sigma, p, p),
            Rcpp::Named("L") = engine_matrix(result.L_factor, p,
                result.variant_used == tulpa::vi::VIVariant::FULLRANK ? p : result.rank_used),
            Rcpp::Named("d") = engine_vector(result.d_diag, result.d_diag ? p : 0),
            Rcpp::Named("rank") = result.rank_used, Rcpp::Named("elbo") = result.final_elbo,
            Rcpp::Named("iterations") = result.iterations,
            Rcpp::Named("converged") = bool(result.converged),
            Rcpp::Named("elbo_history") = engine_vector(result.elbo_history, result.iterations),
            Rcpp::Named("samples") = samples, Rcpp::Named("psis_k") = result.psis_k);
    }
    Rcpp::stop("Unknown engine sampler '%s'", method.c_str());
}

}
#endif
