// ratio_obs_lik.h
// One observation's log-likelihood, written once for every scalar type.
//
// NOTE: include AFTER hmc_sampler.h (ModelData, ParamLayout, ModelType).

#ifndef RATIOD_RATIO_OBS_LIK_H
#define RATIOD_RATIO_OBS_LIK_H

#include <limits>
#include "autodiff_utils.h"
#include "hmc_zi.h"

namespace ratiod {

using namespace math;
using ratiod_hmc::ModelData;
using ratiod_hmc::ParamLayout;
using ratiod_hmc::ModelType;
using ratiod_zi::ZIType;

// One observation's log-likelihood given its two linear predictors, the
// dispersions and the inflation logits. Written once for every scalar type:
// the sampler evaluates it on doubles and autodiff variables, the collapsed
// interaction on Taylor jets to read its derivatives in the predictors.
template <typename T>
inline T ratio_obs_log_lik(const ModelData& data, const ParamLayout& layout,
                           int i, const T& eta_num, const T& eta_denom,
                           const T& phi_num, const T& phi_denom,
                           const T& logit_zi, const T& logit_oi) {
    // An identity or inverse link can map eta outside the positive
    // half-line, where a mean has no density
    const T neg_inf = T(-std::numeric_limits<double>::infinity());

    // Compute likelihood based on model type
    T ll_i = T(0.0);

    if (data.model_type == ModelType::BINOMIAL) {
        T p = inv_logit(prob_logit(eta_num, data.link_num));
        int y = data.y_num[i];
        int n = data.y_denom[i];

        // Handle different ZI types for binomial
        if (data.zi_type == ZIType::ZI_BINOMIAL) {
            ll_i = log_lik_zi_binomial(y, n, p, logit_zi);
        } else if (data.zi_type == ZIType::OI_BINOMIAL) {
            ll_i = log_lik_oi_binomial(y, n, p, logit_oi);
        } else if (data.zi_type == ZIType::ZOIB) {
            ll_i = log_lik_zoib(y, n, p, logit_zi, logit_oi);
        } else if (data.zi_type == ZIType::HURDLE_BINOMIAL) {
            ll_i = log_lik_hurdle_binomial(y, n, p, logit_zi);
        } else {
            ll_i = log_lik_binomial(y, n, p);
        }

    } else if (data.model_type == ModelType::NEGBIN_NEGBIN) {
        // Zero-inflation and hurdle structures attach to the numerator only;
        // the denominator count is always the plain family. Both densities
        // route through ratiod_zi::zi_log_likelihood so a count-response
        // ZI/hurdle term cannot be carried by one and dropped by the
        // other.
        T mu_num = link_mean(eta_num, data.link_num);
        T mu_denom = link_mean(eta_denom, data.link_denom);
        if (layout.has_zi) {
            ll_i = ratiod_zi::zi_log_likelihood(data.y_num[i], mu_num, phi_num,
                                                logit_zi, data.zi_type);
        } else {
            ll_i = log_lik_negbin(data.y_num[i], mu_num, phi_num);
        }
        ll_i = ll_i + log_lik_negbin(data.y_denom[i], mu_denom, phi_denom);

    } else if (data.model_type == ModelType::POISSON_GAMMA) {
        T mu_num = link_mean(eta_num, data.link_num);
        T mu_denom = link_mean(eta_denom, data.link_denom);
        if (get_value(mu_denom) <= 0.0) return neg_inf;
        if (layout.has_zi) {
            ll_i = ratiod_zi::zi_log_likelihood(data.y_num[i], mu_num, phi_num,
                                                logit_zi, data.zi_type);
        } else {
            ll_i = log_lik_poisson(data.y_num[i], mu_num);
        }
        ll_i = ll_i + log_lik_gamma(data.y_denom_cont[i], phi_denom, mu_denom);

    } else if (data.model_type == ModelType::NEGBIN_GAMMA) {
        T mu_num = link_mean(eta_num, data.link_num);
        T mu_denom = link_mean(eta_denom, data.link_denom);
        if (get_value(mu_denom) <= 0.0) return neg_inf;
        if (layout.has_zi) {
            ll_i = ratiod_zi::zi_log_likelihood(data.y_num[i], mu_num, phi_num,
                                                logit_zi, data.zi_type);
        } else {
            ll_i = log_lik_negbin(data.y_num[i], mu_num, phi_num);
        }
        ll_i = ll_i + log_lik_gamma(data.y_denom_cont[i], phi_denom, mu_denom);

    } else if (data.model_type == ModelType::GAMMA_GAMMA) {
        // Gamma-Gamma: both responses are continuous Gamma
        // phi_num = shape_num, phi_denom = shape_denom
        T mu_num = link_mean(eta_num, data.link_num);
        T mu_denom = link_mean(eta_denom, data.link_denom);
        if (get_value(mu_num) <= 0.0 || get_value(mu_denom) <= 0.0) return neg_inf;
        ll_i = log_lik_gamma_gamma(data.y_num_cont[i], data.y_denom_cont[i],
                                   mu_num, mu_denom, phi_num, phi_denom);

    } else if (data.model_type == ModelType::LOGNORMAL) {
        // Lognormal-Lognormal: both responses are continuous Lognormal
        // eta = mean on log scale, phi = sigma (std dev on log scale)
        ll_i = log_lik_lognormal_lognormal(data.y_num_cont[i], data.y_denom_cont[i],
                                           eta_num, eta_denom, phi_num, phi_denom);

    } else if (data.model_type == ModelType::BETA_BINOMIAL) {
        // Beta-Binomial: overdispersed binomial
        // phi_num = precision parameter (alpha + beta)
        T p = inv_logit(prob_logit(eta_num, data.link_num));
        int y = data.y_num[i];
        int n = data.y_denom[i];
        ll_i = log_lik_beta_binomial(y, n, p, phi_num);

    } else if (data.model_type == ModelType::POISSON_FIXED ||
               data.model_type == ModelType::NEGBIN_FIXED) {
        // A known exposure E scales the mean: mu = E * linkinv(eta), the
        // rate being linkinv(eta)
        T mu_num = link_mean(eta_num, data.link_num) * data.y_denom_cont[i];
        if (layout.has_zi) {
            ll_i = ratiod_zi::zi_log_likelihood(data.y_num[i], mu_num, phi_num,
                                                logit_zi, data.zi_type);
        } else if (data.model_type == ModelType::POISSON_FIXED) {
            ll_i = log_lik_poisson(data.y_num[i], mu_num);
        } else {
            ll_i = log_lik_negbin(data.y_num[i], mu_num, phi_num);
        }

    } else if (data.model_type == ModelType::LOGNORMAL_FIXED) {
        // log y ~ N(log E + eta, sigma): the median is E exp(eta)
        ll_i = log_lik_lognormal(data.y_num_cont[i],
                                 eta_num + std::log(data.y_denom_cont[i]), phi_num);

    } else if (data.model_type == ModelType::POISSON_NEGBIN) {
        T mu_num = link_mean(eta_num, data.link_num);
        T mu_denom = link_mean(eta_denom, data.link_denom);
        if (layout.has_zi) {
            ll_i = ratiod_zi::zi_log_likelihood(data.y_num[i], mu_num, phi_num,
                                                logit_zi, data.zi_type);
        } else {
            ll_i = log_lik_poisson(data.y_num[i], mu_num);
        }
        ll_i = ll_i + log_lik_negbin(data.y_denom[i], mu_denom, phi_denom);
    }

    return ll_i;
}

// An observation's log-likelihood and its first derivative in each argument
// ratio_obs_log_lik takes, read off that same function on forward duals: one
// pass per argument asked for. The hand-coded gradients that walk the
// observations one at a time take their residuals here, so the family, the
// links and any inflation are the ones the log posterior evaluates.
struct ObsDerivs {
    double ll = 0.0;
    double eta_num = 0.0, eta_denom = 0.0;
    double phi_num = 0.0, phi_denom = 0.0;
    double logit_zi = 0.0, logit_oi = 0.0;
};

enum ObsDerivWant : unsigned {
    OBS_D_PHI_NUM = 1u << 0,
    OBS_D_PHI_DENOM = 1u << 1,
    OBS_D_ZI = 1u << 2,
    OBS_D_OI = 1u << 3
};

inline ObsDerivs ratio_obs_derivs(const ModelData& data, const ParamLayout& layout,
                                  int i, double eta_num, double eta_denom,
                                  double phi_num, double phi_denom,
                                  double logit_zi, double logit_oi,
                                  unsigned want) {
    using D = fwd::Dual;
    // The argument seeded with slope 1 in this pass
    auto pass = [&](int seed) {
        return ratio_obs_log_lik<D>(
            data, layout, i, D(eta_num, seed == 0 ? 1.0 : 0.0),
            D(eta_denom, seed == 1 ? 1.0 : 0.0), D(phi_num, seed == 2 ? 1.0 : 0.0),
            D(phi_denom, seed == 3 ? 1.0 : 0.0), D(logit_zi, seed == 4 ? 1.0 : 0.0),
            D(logit_oi, seed == 5 ? 1.0 : 0.0));
    };
    ObsDerivs out;
    const D l_num = pass(0);
    out.ll = l_num.val;
    out.eta_num = l_num.grad;
    out.eta_denom = pass(1).grad;
    if (want & OBS_D_PHI_NUM) out.phi_num = pass(2).grad;
    if (want & OBS_D_PHI_DENOM) out.phi_denom = pass(3).grad;
    if (want & OBS_D_ZI) out.logit_zi = pass(4).grad;
    if (want & OBS_D_OI) out.logit_oi = pass(5).grad;
    return out;
}

}  // namespace ratiod

#endif  // RATIOD_RATIO_OBS_LIK_H
