// lik_negbin_gamma.cpp
// =====================================================================
// MATH HEADER (per CLAUDE.md "Math ground first")
// =====================================================================
//
// Two linear predictors (two processes), log link on each
// -------------------------------------------------------
//   eta_num_i   = x_num_i^T beta_num     + offset_num_i
//   eta_denom_i = x_denom_i^T beta_denom + offset_denom_i
//   mu_num_i    = exp(eta_num_i)         (NegBin numerator mean)
//   mu_denom_i  = exp(eta_denom_i)       (Gamma denominator mean)
//
// Per-observation observation model (matches log_post_impl.h negbin_gamma):
//   y_num_i   ~ NegBin(mu_num_i,   phi_num)
//   y_denom_i ~ Gamma(phi_denom,   rate=phi_denom / mu_denom_i)
//
// Extra parameters (in layout, in this order):
//   log_phi_num   = layout.extra_offset + 0     (NegBin overdispersion)
//   log_phi_denom = layout.extra_offset + 1     (Gamma shape)
//
// Per-observation log-likelihood
// ------------------------------
//   ll_i = log P_negbin(y_num_i | mu_num_i, phi_num)
//        + log p_gamma(y_denom_i | phi_denom, rate=phi_denom/mu_denom_i)
//
// Closed-form per-obs gradient (recorded for B2)
// ----------------------------------------------
//   d ll_i / d eta_num_i    = (y_num_i - mu_num_i) * phi_num/(mu_num_i + phi_num)
//   d ll_i / d eta_denom_i  = phi_denom * (1 - y_denom_i / mu_denom_i)
//   d ll_i / d log_phi_num  = phi_num * [digamma(y_num_i + phi_num)
//                              - digamma(phi_num) + log(phi_num/(mu_num_i+phi_num))
//                              + (mu_num_i - y_num_i)/(mu_num_i + phi_num)]
//   d ll_i / d log_phi_denom (Gamma shape derivative — see gamma_gamma)
//
// Priors
// ------
//   log_phi_num   ~ Gamma(phi_prior_shape, phi_prior_rate)
//   log_phi_denom ~ Gamma(phi_prior_shape, phi_prior_rate)
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
struct NegbinGammaCfg {
    double phi_prior_shape;
    double phi_prior_rate;
};
inline NegbinGammaCfg& negbin_gamma_cfg() {
    static NegbinGammaCfg c{1.0, 0.01};
    return c;
}
}  // namespace

template <typename T>
static T negbin_gamma_log_likelihood(
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

    const auto* resp = static_cast<const lik::NegbinGammaResponseData*>(model_data);
    const int    y_n = resp->y_num[i];
    const double y_d = resp->y_denom_cont[i];

    const T mu_num   = exp(eta[0]);
    const T mu_denom = exp(eta[1]);

    const T log_phi_num   = params[layout.extra_offset + 0];
    const T log_phi_denom = params[layout.extra_offset + 1];
    const T phi_num       = exp(log_phi_num);
    const T phi_denom     = exp(log_phi_denom);

    T ll;
    switch (data.zi_type) {
        case tulpa::ZIType::ZI_NEGBIN:
            ll = lik::zi_negbin_log_pmf(y_n, mu_num, phi_num, logit_zi);
            break;
        case tulpa::ZIType::HURDLE_NEGBIN:
            ll = lik::hurdle_negbin_log_pmf(y_n, mu_num, phi_num, logit_zi);
            break;
        case tulpa::ZIType::NONE:
        default:
            ll = lik::negbin_log_pmf(y_n, mu_num, phi_num);
            break;
    }
    ll = ll + lik::gamma_log_pdf(y_d, phi_denom, mu_denom);

    if (i == 0) {
        const auto& cfg = negbin_gamma_cfg();
        ll = ll + lik::log_prior_gamma(log_phi_num,   cfg.phi_prior_shape, cfg.phi_prior_rate);
        ll = ll + lik::log_prior_gamma(log_phi_denom, cfg.phi_prior_shape, cfg.phi_prior_rate);
    }
    return ll;
}

tulpa::LikelihoodSpec build_negbin_gamma_spec(const RatioConfig& cfg) {
    negbin_gamma_cfg() = {cfg.phi_prior_shape, cfg.phi_prior_rate};

    tulpa::LikelihoodSpec spec;
    spec.name           = "tulpaRatio_negbin_gamma";
    spec.n_processes    = 2;
    spec.n_extra_params = 2;  // log_phi_num, log_phi_denom

    if (cfg.num_link != "log" || cfg.denom_link != "log") {
        spec.name = "tulpaRatio_negbin_gamma_unsupported_link";
        return spec;
    }
    spec.ll_double = negbin_gamma_log_likelihood<double>;
    spec.ll_arena  = negbin_gamma_log_likelihood<tulpa::arena::Var>;
    if (cfg.zi == "none") {
        spec.gradient_fn = &grad_h_negbin_gamma;
    }
    return spec;
}

}  // namespace tulpaRatio
