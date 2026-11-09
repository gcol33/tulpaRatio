// lik_binomial.cpp
// =====================================================================
// MATH HEADER (per CLAUDE.md "Math ground first")
// =====================================================================
//
// Per-observation model
// ---------------------
//   y_i | n_i ~ Binomial(n_i, p_i),         i = 1, ..., N
//   eta_i = x_i^T beta + offset_i           (offset_i = 0 in B1a)
//   p_i   = sigmoid(eta_i) = 1 / (1 + exp(-eta_i))   (logit link)
//
//   y_i = success count (numerator), integer in [0, n_i]
//   n_i = trial count (denominator), fixed integer not modelled
//
// Per-observation log-likelihood
// ------------------------------
//   log L_i(beta) = y_i * log(p_i) + (n_i - y_i) * log(1 - p_i) + lchoose(n_i, y_i)
//
// Numerically stable evaluation (used here):
//   log(p_i)     = eta_i - softplus(eta_i)   where softplus(x) = log(1 + exp(x))
//   log(1 - p_i) =       - softplus(eta_i)
//
// Closed-form per-obs gradient (recorded for B2; AD provides it in B1a):
//   d log L_i / d eta_i = y_i - n_i * p_i
//
// ZI / hurdle / OI / ZOIB (B1c)
// -----------------------------
// When cfg.zi != "none", this file routes to the corresponding mixture-form
// log-pmf in lik_helpers.h (zi_binom_log_pmf, hurdle_binom_log_pmf,
// oi_binom_log_pmf, zoib_log_pmf). The hand-coded H-mode gradient does NOT
// support ZI yet, so the spec builder leaves spec.gradient_fn = nullptr in
// the ZI case — autodiff (A_r/A/N) handles the inflation parameters via
// logit_zi / logit_oi which tulpa pre-computes from data.X_zi/X_oi.
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

// ============================================================================
// Templated per-observation log-likelihood for binomial (and ZI/hurdle/OI/
// ZOIB variants). Branches on data.zi_type so the same callback covers every
// inflation flavour without per-variant function pointers.
// Signature matches tulpa::LikelihoodFn<T>.
// ============================================================================
template <typename T>
static T binomial_log_likelihood(
    int i,
    const T* eta,
    const T& logit_zi,
    const T& logit_oi,
    const std::vector<T>& /*params*/,
    const tulpa::ModelData& data,
    const tulpa::ParamLayout& /*layout*/,
    const void* model_data
) {
    // Anchor the arena so any temporaries created below allocate against the
    // arena that produced eta[0]. Mirrors the convention used in tulpaObs.
    if constexpr (std::is_same_v<T, tulpa::arena::Var>) {
        tulpa::arena::current_arena() = eta[0].arena_;
    }

    const auto* resp = static_cast<const lik::BinomialResponseData*>(model_data);
    const int y_i = resp->y[i];
    const int n_i = resp->n[i];

    switch (data.zi_type) {
        case tulpa::ZIType::ZI_BINOMIAL:
            return ratiod::math::log_lik_zi_binomial(y_i, n_i, eta[0], logit_zi);
        case tulpa::ZIType::HURDLE_BINOMIAL:
            return ratiod::math::log_lik_hurdle_binomial(y_i, n_i, eta[0], logit_zi);
        case tulpa::ZIType::OI_BINOMIAL:
            return ratiod::math::log_lik_oi_binomial(y_i, n_i, eta[0], logit_oi);
        case tulpa::ZIType::ZOIB:
            return ratiod::math::log_lik_zoib(y_i, n_i, eta[0], logit_zi, logit_oi);
        case tulpa::ZIType::NONE:
        default:
            return ratiod::math::log_lik_binomial(y_i, n_i, eta[0]);
    }
}

// ============================================================================
// Public builder: assemble a LikelihoodSpec for the binomial-ratio family.
// Currently only the logit link is supported; new links should branch here.
// ============================================================================
tulpa::LikelihoodSpec build_binomial_spec(const RatioConfig& cfg) {
    tulpa::LikelihoodSpec spec;
    spec.name         = "tulpaRatio_binomial";
    spec.n_processes  = 1;
    spec.n_extra_params = 0;

    if (cfg.num_link != "logit") {
        // B1a registers only the logit link. Other links are deferred.
        spec.name = "tulpaRatio_binomial_unsupported_link";
        return spec;  // dispatcher will raise — see lik_dispatch.cpp
    }

    spec.ll_double = binomial_log_likelihood<double>;
    spec.ll_arena  = binomial_log_likelihood<tulpa::arena::Var>;

    // Hand-coded H-mode gradient (B2) covers only the plain Binomial. ZI/OI
    // variants flow through autodiff (A_r/A) — leave gradient_fn unset so the
    // dispatcher falls back to the templated AD path.
    if (cfg.zi == "none") {
        spec.gradient_fn = &grad_h_binomial;
    }

    // No ResidualFn, EtaWeightsFn, extend_layout, or extra_prior — the
    // gradient_fn carries the analytical derivative end-to-end.
    return spec;
}

}  // namespace tulpaRatio
