// lik_beta_binomial.cpp
// =====================================================================
// MATH HEADER (per CLAUDE.md "Math ground first")
// =====================================================================
//
// Single linear predictor (one process), logit link
// -------------------------------------------------
//   eta_i = x_i^T beta + offset_i
//   p_i   = sigmoid(eta_i)
//
// Per-observation observation model (matches log_post_impl.h beta_binomial):
//   y_i | n_i ~ BetaBinomial(n_i, alpha_i, beta_i)
// with mean-precision parameterisation:
//   alpha_i = p_i  * phi
//   beta_i  = (1 - p_i) * phi
//
// Extra parameter (in layout):
//   log_phi = layout.extra_offset + 0   (precision parameter)
//
// Per-observation log-likelihood
// ------------------------------
//   ll_i = lgamma(y_i + alpha_i) + lgamma(n_i - y_i + beta_i) - lgamma(n_i + alpha_i + beta_i)
//        - lgamma(alpha_i) - lgamma(beta_i) + lgamma(alpha_i + beta_i) + lchoose(n_i, y_i)
//
// Closed-form per-obs gradients (recorded for B2)
// -----------------------------------------------
//   d ll_i / d eta_i  = phi * p_i * (1 - p_i) * [digamma(y_i + alpha_i)
//                       - digamma(n_i - y_i + beta_i)
//                       - digamma(alpha_i) + digamma(beta_i)]
//   d ll_i / d log_phi closed form via digamma differences (omitted here).
//
// Prior
// -----
//   log_phi ~ Gamma(phi_prior_shape, phi_prior_rate)
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
struct BetaBinomialCfg {
    double phi_prior_shape;
    double phi_prior_rate;
};
inline BetaBinomialCfg& beta_binomial_cfg() {
    static BetaBinomialCfg c{1.0, 0.01};
    return c;
}
}  // namespace

template <typename T>
static T beta_binomial_log_likelihood(
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

    const auto* resp = static_cast<const lik::BetaBinomialResponseData*>(model_data);
    const int y = resp->y[i];
    const int n = resp->n[i];

    const T log_phi = params[layout.extra_offset];
    const T phi     = exp(log_phi);

    T ll = lik::beta_binomial_log_pmf(y, n, eta[0], phi);

    if (i == 0) {
        const auto& cfg = beta_binomial_cfg();
        ll = ll + lik::log_prior_gamma(log_phi, cfg.phi_prior_shape, cfg.phi_prior_rate);
    }
    return ll;
}

tulpa::LikelihoodSpec build_beta_binomial_spec(const RatioConfig& cfg) {
    beta_binomial_cfg() = {cfg.phi_prior_shape, cfg.phi_prior_rate};

    tulpa::LikelihoodSpec spec;
    spec.name           = "tulpaRatio_beta_binomial";
    spec.n_processes    = 1;
    spec.n_extra_params = 1;

    if (cfg.num_link != "logit") {
        spec.name = "tulpaRatio_beta_binomial_unsupported_link";
        return spec;
    }
    spec.ll_double = beta_binomial_log_likelihood<double>;
    spec.ll_arena  = beta_binomial_log_likelihood<tulpa::arena::Var>;
    spec.gradient_fn = &grad_h_beta_binomial;
    return spec;
}

}  // namespace tulpaRatio
