// lik_poisson_gamma.cpp
// =====================================================================
// MATH HEADER (per CLAUDE.md "Math ground first")
// =====================================================================
//
// Two linear predictors (two processes, log link on each)
// -------------------------------------------------------
//   eta_num_i   = x_num_i^T beta_num     + offset_num_i      (offset = 0 in B1b)
//   eta_denom_i = x_denom_i^T beta_denom + offset_denom_i
//   mu_num_i    = exp(eta_num_i)         (Poisson numerator mean)
//   mu_denom_i  = exp(eta_denom_i)       (Gamma denominator mean)
//
// Per-observation observation model (matches log_post_impl.h poisson_gamma):
//   y_num_i      ~ Poisson(mu_num_i)
//   y_denom_i    ~ Gamma(shape, rate=shape/mu_denom_i)
// The Gamma shape (== "phi_denom" in legacy) is positive; parameterized as
// shape = exp(log_shape) at layout.extra_offset + 0. There is NO phi_num.
//
// Per-observation log-likelihood
// ------------------------------
//   ll_pois_i  = y_num_i * eta_num_i - mu_num_i - lgamma(y_num_i + 1)
//   ll_gamma_i = shape*log(rate_i) + (shape-1)*log(y_denom_i) - rate_i*y_denom_i
//                - lgamma(shape),  rate_i = shape / mu_denom_i
//   ll_i       = ll_pois_i + ll_gamma_i
//
// Closed-form per-obs gradient (recorded for B2; AD provides it now)
// ------------------------------------------------------------------
//   d ll_i / d eta_num_i   = y_num_i - mu_num_i
//   d ll_i / d eta_denom_i = shape * (y_denom_i / mu_denom_i - 1)
//   d ll_i / d log_shape   = shape * (1 + log(rate_i) - rate_i*y_denom_i/shape
//                                      - digamma(shape) + log(y_denom_i))
//
// Extra-parameter prior
// ---------------------
//   log_shape ~ Gamma(phi_prior_shape, phi_prior_rate) (Jacobian-adjusted)
//   matches log_prior_gamma in legacy code.
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

// Cached per-builder copy of the priors used by the templated callback. The
// LikelihoodSpec carries no per-family state, so we stash these in static
// thread-local storage at build time. B1b runs single-chain so a single set
// per process is sufficient (a future spec_ctx pointer would generalise).
namespace {
struct PoissonGammaCfg {
    double phi_prior_shape;
    double phi_prior_rate;
};
inline PoissonGammaCfg& poisson_gamma_cfg() {
    static PoissonGammaCfg c{1.0, 0.01};
    return c;
}
}  // namespace

template <typename T>
static T poisson_gamma_log_likelihood(
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

    const auto* resp = static_cast<const lik::PoissonGammaResponseData*>(model_data);
    const int    y_n = resp->y_num[i];
    const double y_d = resp->y_denom_cont[i];

    const T eta_num   = eta[0];
    const T eta_denom = eta[1];
    const T mu_num    = exp(eta_num);
    const T mu_denom  = exp(eta_denom);

    // Numerator: Poisson(y_num | mu_num) with optional ZI / hurdle.
    T ll;
    switch (data.zi_type) {
        case tulpa::ZIType::ZI_POISSON:
            ll = ratiod_zi::zi_poisson_lpmf_logit(y_n, mu_num, logit_zi);
            break;
        case tulpa::ZIType::HURDLE_POISSON:
            ll = ratiod_zi::hurdle_poisson_lpmf_logit(y_n, mu_num, logit_zi);
            break;
        case tulpa::ZIType::NONE:
        default:
            // Inline plain Poisson form to share `mu_num` with the denominator.
            ll = ratiod::math::poisson_eta_density(double(y_n), eta_num, mu_num, T(std::lgamma(double(y_n) + 1.0)));
            break;
    }

    // Denominator: Gamma(y_denom | shape, rate=shape/mu_denom).
    const T log_shape = params[layout.extra_offset];
    const T shape     = exp(log_shape);
    ll = ll + ratiod::math::log_lik_gamma(y_d, shape, mu_denom);

    // Anchor the extra-parameter prior at i==0 so it contributes exactly once
    // to the total log posterior while still flowing through autodiff.
    if (i == 0) {
        const auto& cfg = poisson_gamma_cfg();
        ll = ll + lik::log_prior_gamma(log_shape,
                                           cfg.phi_prior_shape,
                                           cfg.phi_prior_rate);
    }
    return ll;
}

tulpa::LikelihoodSpec build_poisson_gamma_spec(const RatioConfig& cfg) {
    poisson_gamma_cfg() = {cfg.phi_prior_shape, cfg.phi_prior_rate};

    tulpa::LikelihoodSpec spec;
    spec.name           = "tulpaRatio_poisson_gamma";
    spec.n_processes    = 2;
    spec.n_extra_params = 1;  // log_shape (Gamma denominator only)

    if (cfg.num_link != "log" || cfg.denom_link != "log") {
        spec.name = "tulpaRatio_poisson_gamma_unsupported_link";
        return spec;
    }
    spec.ll_double = poisson_gamma_log_likelihood<double>;
    spec.ll_arena  = poisson_gamma_log_likelihood<tulpa::arena::Var>;

    // H-mode handcoded gradient covers only plain Poisson-Gamma. ZI variants
    // route through the templated AD path.
    if (cfg.zi == "none") {
        spec.gradient_fn = &grad_h_poisson_gamma;
    }
    return spec;
}

}  // namespace tulpaRatio
