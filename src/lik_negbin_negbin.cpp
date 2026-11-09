// lik_negbin_negbin.cpp
// =====================================================================
// MATH HEADER (per CLAUDE.md "Math ground first")
// =====================================================================
//
// Two linear predictors (two processes)
// -------------------------------------
//   eta_num_i   = x_num_i^T beta_num     + offset_num_i
//   eta_denom_i = x_denom_i^T beta_denom + offset_denom_i
//   mu_num_i    = exp(eta_num_i)
//   mu_denom_i  = exp(eta_denom_i)
//
// Per-observation observation model
// ---------------------------------
//   y_num_i   ~ NegBin(mu_num_i,   phi_num)
//   y_denom_i ~ NegBin(mu_denom_i, phi_denom)
//
// Extra parameters (in layout):
//   log_phi_num   = layout.extra_offset + 0
//   log_phi_denom = layout.extra_offset + 1
//
// Closed-form per-obs gradients (recorded for B2)
// -----------------------------------------------
//   d ll_i / d eta_num_i   = (y_num_i   - mu_num_i)   * phi_num   /(mu_num_i   + phi_num)
//   d ll_i / d eta_denom_i = (y_denom_i - mu_denom_i) * phi_denom /(mu_denom_i + phi_denom)
//
// Priors
// ------
//   log_phi_{num,denom} ~ Gamma(phi_prior_shape, phi_prior_rate)
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
struct NegbinNegbinCfg {
    double phi_prior_shape;
    double phi_prior_rate;
};
inline NegbinNegbinCfg& negbin_negbin_cfg() {
    static NegbinNegbinCfg c{1.0, 0.01};
    return c;
}
}  // namespace

template <typename T>
static T negbin_negbin_log_likelihood(
    int i,
    const T* eta,
    const T& logit_zi,
    const T& /*logit_oi*/,
    const std::vector<T>& params,
    const tulpa::ModelData& data,
    const tulpa::ParamLayout& layout,
    const void* model_data
) {
    using std::exp;

    if constexpr (std::is_same_v<T, tulpa::arena::Var>) {
        tulpa::arena::current_arena() = eta[0].arena_;
    }

    const auto* resp = static_cast<const lik::NegbinNegbinResponseData*>(model_data);
    const int y_n = resp->y_num[i];
    const int y_d = resp->y_denom[i];

    const T mu_num   = exp(eta[0]);
    const T mu_denom = exp(eta[1]);

    const T log_phi_num   = params[layout.extra_offset + 0];
    const T log_phi_denom = params[layout.extra_offset + 1];
    const T phi_num       = exp(log_phi_num);
    const T phi_denom     = exp(log_phi_denom);

    T ll;
    switch (data.zi_type) {
        case tulpa::ZIType::ZI_NEGBIN:
            ll = ratiod_zi::zi_negbin_lpmf_logit(y_n, mu_num, phi_num, logit_zi);
            break;
        case tulpa::ZIType::HURDLE_NEGBIN:
            ll = ratiod_zi::hurdle_negbin_lpmf_logit(y_n, mu_num, phi_num, logit_zi);
            break;
        case tulpa::ZIType::NONE:
        default:
            ll = ratiod::math::log_lik_negbin(y_n, mu_num, phi_num);
            break;
    }
    ll = ll + ratiod::math::log_lik_negbin(y_d, mu_denom, phi_denom);

    if (i == 0) {
        const auto& cfg = negbin_negbin_cfg();
        ll = ll + lik::log_prior_gamma(log_phi_num,   cfg.phi_prior_shape, cfg.phi_prior_rate);
        ll = ll + lik::log_prior_gamma(log_phi_denom, cfg.phi_prior_shape, cfg.phi_prior_rate);
    }
    return ll;
}

tulpa::LikelihoodSpec build_negbin_negbin_spec(const RatioConfig& cfg) {
    negbin_negbin_cfg() = {cfg.phi_prior_shape, cfg.phi_prior_rate};

    tulpa::LikelihoodSpec spec;
    spec.name           = "tulpaRatio_negbin_negbin";
    spec.n_processes    = 2;
    spec.n_extra_params = 2;

    if (cfg.num_link != "log" || cfg.denom_link != "log") {
        spec.name = "tulpaRatio_negbin_negbin_unsupported_link";
        return spec;
    }
    spec.ll_double = negbin_negbin_log_likelihood<double>;
    spec.ll_arena  = negbin_negbin_log_likelihood<tulpa::arena::Var>;
    if (cfg.zi == "none") {
        spec.gradient_fn = &grad_h_negbin_negbin;
    }
    return spec;
}

}  // namespace tulpaRatio
