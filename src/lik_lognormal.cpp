// lik_lognormal.cpp
// =====================================================================
// MATH HEADER (per CLAUDE.md "Math ground first")
// =====================================================================
//
// Two linear predictors, identity link on the log-scale mean
// ---------------------------------------------------------
//   eta_num_i   = x_num_i^T beta_num     + offset_num_i  (mean of log y_num)
//   eta_denom_i = x_denom_i^T beta_denom + offset_denom_i
//
// Per-observation observation model (matches log_post_impl.h lognormal):
//   log(y_num_i)   ~ Normal(eta_num_i,   sigma_num)
//   log(y_denom_i) ~ Normal(eta_denom_i, sigma_denom)
//
// Extra parameters (in layout):
//   log_sigma_num   = layout.extra_offset + 0
//   log_sigma_denom = layout.extra_offset + 1
//
// Per-observation log-likelihood (drops -0.5*log(2*pi) constant for parity
// with the legacy kernel).
//
// Closed-form per-obs gradients (recorded for B2)
// -----------------------------------------------
//   d ll_i / d eta_num_i   = (log(y_num_i)  - eta_num_i)   / sigma_num^2
//   d ll_i / d eta_denom_i = (log(y_denom_i)- eta_denom_i) / sigma_denom^2
//   d ll_i / d log_sigma_* = z*^2 - 1  with z* = (log(y_*) - eta_*) / sigma_*
//
// Priors
// ------
//   log_sigma_{num,denom} ~ Half-Cauchy(0, sigma_prior_scale)
//   matches log_prior_half_cauchy in the legacy autodiff_utils.h.
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
struct LognormalCfg {
    double sigma_prior_scale;
};
inline LognormalCfg& lognormal_cfg() {
    static LognormalCfg c{1.0};
    return c;
}
}  // namespace

template <typename T>
static T lognormal_log_likelihood(
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

    const auto* resp = static_cast<const lik::LognormalResponseData*>(model_data);
    const double y_n = resp->y_num_cont[i];
    const double y_d = resp->y_denom_cont[i];

    const T log_sigma_num   = params[layout.extra_offset + 0];
    const T log_sigma_denom = params[layout.extra_offset + 1];
    const T sigma_num       = exp(log_sigma_num);
    const T sigma_denom     = exp(log_sigma_denom);

    T ll = ratiod::math::log_lik_lognormal(y_n, eta[0], sigma_num);
    ll = ll + ratiod::math::log_lik_lognormal(y_d, eta[1], sigma_denom);

    if (i == 0) {
        const auto& cfg = lognormal_cfg();
        ll = ll + lik::log_prior_half_cauchy(log_sigma_num,   cfg.sigma_prior_scale);
        ll = ll + lik::log_prior_half_cauchy(log_sigma_denom, cfg.sigma_prior_scale);
    }
    return ll;
}

tulpa::LikelihoodSpec build_lognormal_spec(const RatioConfig& cfg) {
    lognormal_cfg() = {cfg.sigma_prior_scale};

    tulpa::LikelihoodSpec spec;
    spec.name           = "tulpaRatio_lognormal";
    spec.n_processes    = 2;
    spec.n_extra_params = 2;

    // Lognormal uses an identity link on the log-scale mean — no link branch.
    spec.ll_double = lognormal_log_likelihood<double>;
    spec.ll_arena  = lognormal_log_likelihood<tulpa::arena::Var>;
    spec.gradient_fn = &grad_h_lognormal;
    return spec;
}

}  // namespace tulpaRatio
