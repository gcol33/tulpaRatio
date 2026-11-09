// lik_gamma_gamma.cpp
// =====================================================================
// MATH HEADER (per CLAUDE.md "Math ground first")
// =====================================================================
//
// Two linear predictors (two processes), log link
// -----------------------------------------------
//   eta_num_i   = x_num_i^T beta_num     + offset_num_i
//   eta_denom_i = x_denom_i^T beta_denom + offset_denom_i
//   mu_num_i    = exp(eta_num_i)         (Gamma mean parameterisation)
//   mu_denom_i  = exp(eta_denom_i)
//
// Per-observation observation model (matches log_post_impl.h gamma_gamma)
// -----------------------------------------------------------------------
//   y_num_i   ~ Gamma(shape_num,   rate=shape_num   / mu_num_i)
//   y_denom_i ~ Gamma(shape_denom, rate=shape_denom / mu_denom_i)
//
// Extra parameters (in layout):
//   log_shape_num   = layout.extra_offset + 0
//   log_shape_denom = layout.extra_offset + 1
//
// Closed-form per-obs gradients (recorded for B2)
// -----------------------------------------------
//   d ll_i / d eta_num_i   = shape_num   * (1 - y_num_i  / mu_num_i)
//   d ll_i / d eta_denom_i = shape_denom * (1 - y_denom_i / mu_denom_i)
//   d ll_i / d log_shape_*  = shape_*   * (1 + log(rate_*) - rate_* y_*/shape_*
//                                       - digamma(shape_*) + log(y_*))
//
// Priors
// ------
//   log_shape_{num,denom} ~ Gamma(phi_prior_shape, phi_prior_rate)
// =====================================================================

#include <type_traits>
#include <tulpa/likelihood.h>
#include <tulpa/model_data.h>
#include <tulpa/param_layout.h>
#include <tulpa/autodiff_arena.h>

#include "lik_specs/ratio_config.h"
#include "lik_specs/lik_helpers.h"
#include "lik_specs/lik_grad_h_kernel.h"

namespace tulpaRatio {

namespace {
struct GammaGammaCfg {
    double phi_prior_shape;
    double phi_prior_rate;
};
inline GammaGammaCfg& gamma_gamma_cfg() {
    static GammaGammaCfg c{1.0, 0.01};
    return c;
}
}  // namespace

template <typename T>
static T gamma_gamma_log_likelihood(
    int i,
    const T* eta,
    const T& /*logit_zi*/,
    const T& /*logit_oi*/,
    const std::vector<T>& params,
    const tulpa::ModelData& /*data*/,
    const tulpa::ParamLayout& layout,
    const void* model_data
) {
    using std::exp;

    if constexpr (std::is_same_v<T, tulpa::arena::Var>) {
        tulpa::arena::current_arena() = eta[0].arena_;
    }

    const auto* resp = static_cast<const lik::GammaGammaResponseData*>(model_data);
    const double y_n = resp->y_num_cont[i];
    const double y_d = resp->y_denom_cont[i];

    const T mu_num   = exp(eta[0]);
    const T mu_denom = exp(eta[1]);

    const T log_shape_num   = params[layout.extra_offset + 0];
    const T log_shape_denom = params[layout.extra_offset + 1];
    const T shape_num       = exp(log_shape_num);
    const T shape_denom     = exp(log_shape_denom);

    T ll = ratiod::math::log_lik_gamma(y_n, shape_num,   mu_num);
    ll = ll + ratiod::math::log_lik_gamma(y_d, shape_denom, mu_denom);

    if (i == 0) {
        const auto& cfg = gamma_gamma_cfg();
        ll = ll + lik::log_prior_gamma(log_shape_num,   cfg.phi_prior_shape, cfg.phi_prior_rate);
        ll = ll + lik::log_prior_gamma(log_shape_denom, cfg.phi_prior_shape, cfg.phi_prior_rate);
    }
    return ll;
}

tulpa::LikelihoodSpec build_gamma_gamma_spec(const RatioConfig& cfg) {
    gamma_gamma_cfg() = {cfg.phi_prior_shape, cfg.phi_prior_rate};

    tulpa::LikelihoodSpec spec;
    spec.name           = "tulpaRatio_gamma_gamma";
    spec.n_processes    = 2;
    spec.n_extra_params = 2;

    if (cfg.num_link != "log" || cfg.denom_link != "log") {
        spec.name = "tulpaRatio_gamma_gamma_unsupported_link";
        return spec;
    }
    spec.ll_double = gamma_gamma_log_likelihood<double>;
    spec.ll_arena  = gamma_gamma_log_likelihood<tulpa::arena::Var>;
    spec.gradient_fn = &grad_h_gamma_gamma;
    return spec;
}

}  // namespace tulpaRatio
