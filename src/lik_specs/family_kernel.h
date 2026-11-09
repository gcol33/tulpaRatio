#ifndef TULPARATIO_FAMILY_KERNEL_H
#define TULPARATIO_FAMILY_KERNEL_H

#include "lik_helpers.h"
#include <tulpa/model_data.h>
#include <tulpa/param_layout.h>

namespace tulpaRatio { namespace lik {

template<class Margin>
constexpr bool continuous_response = std::is_same<Margin,margins::Gamma>::value ||
                                     std::is_same<Margin,margins::Lognormal>::value;

template<class Margin, class T>
T margin_likelihood(double y, int n, const T& eta, const T& phi,
                    tulpa::ZIType zi, const T& logit_zi, const T& logit_oi) {
    using namespace margins;
    using namespace ratiod::math;
    using std::exp;
    if constexpr (std::is_same<Margin,Binomial>::value) {
        switch (zi) {
            case tulpa::ZIType::ZI_BINOMIAL: return log_lik_zi_binomial(int(y),n,eta,logit_zi);
            case tulpa::ZIType::HURDLE_BINOMIAL: return log_lik_hurdle_binomial(int(y),n,eta,logit_zi);
            case tulpa::ZIType::OI_BINOMIAL: return log_lik_oi_binomial(int(y),n,eta,logit_oi);
            case tulpa::ZIType::ZOIB: return log_lik_zoib(int(y),n,eta,logit_zi,logit_oi);
            default: return log_lik_binomial(int(y),n,eta);
        }
    } else if constexpr (std::is_same<Margin,BetaBinomial>::value) {
        return log_lik_beta_binomial(int(y),n,eta,phi);
    } else if constexpr (std::is_same<Margin,Lognormal>::value) {
        return log_lik_lognormal(y,eta,phi);
    } else {
        const T mu = exp(eta);
        if constexpr (std::is_same<Margin,Poisson>::value) {
            if (zi == tulpa::ZIType::ZI_POISSON) return ratiod_zi::zi_poisson_lpmf_logit(int(y),mu,logit_zi);
            if (zi == tulpa::ZIType::HURDLE_POISSON) return ratiod_zi::hurdle_poisson_lpmf_logit(int(y),mu,logit_zi);
            return margins::evaluate<Poisson>(y,0,eta,phi,&mu);
        } else if constexpr (std::is_same<Margin,NegBin>::value) {
            if (zi == tulpa::ZIType::ZI_NEGBIN) return ratiod_zi::zi_negbin_lpmf_logit(int(y),mu,phi,logit_zi);
            if (zi == tulpa::ZIType::HURDLE_NEGBIN) return ratiod_zi::hurdle_negbin_lpmf_logit(int(y),mu,phi,logit_zi);
            return log_lik_negbin(int(y),mu,phi);
        } else return log_lik_gamma(y,phi,mu);
    }
}

template<class Margin, class T>
T extra_prior(const T& log_phi, const tulpa::ModelData& data) {
    if constexpr (Margin::prior == margins::Prior::Gamma)
        return log_prior_gamma(log_phi,data.phi_prior_shape,data.phi_prior_rate);
    else return log_prior_half_cauchy(log_phi,data.sigma_re_scale);
}

template<class Num, class Den, class T>
T family_likelihood(int i, const T* eta, const T& logit_zi, const T& logit_oi,
    const std::vector<T>& params, const tulpa::ModelData& data,
    const tulpa::ParamLayout& layout, const void* payload) {
    using std::exp;
    if constexpr (std::is_same_v<T,tulpa::arena::Var>)
        tulpa::arena::current_arena() = eta[0].arena_;
    const auto& r = *static_cast<const ResponseData*>(payload);
    const double yn = continuous_response<Num> ? r.y_num_cont[i] : r.y_num[i];
    const int n = std::is_same<Num,margins::Binomial>::value ||
                  std::is_same<Num,margins::BetaBinomial>::value ? r.y_denom[i] : 0;
    T pn(1.0), pd(1.0);
    if constexpr (Num::n_extra) pn = exp(params[layout.extra_offset]);
    if constexpr (Den::n_extra) pd = exp(params[layout.extra_offset+Num::n_extra]);
    T ll = margin_likelihood<Num>(yn,n,eta[0],pn,data.zi_type,logit_zi,logit_oi);
    if constexpr (!std::is_same<Den,margins::NoMargin>::value) {
        const double yd = continuous_response<Den> ? r.y_denom_cont[i] : r.y_denom[i];
        ll = ll + margin_likelihood<Den>(yd,0,eta[1],pd,tulpa::ZIType::NONE,logit_zi,logit_oi);
    }
    if (i == 0) {
        if constexpr (Num::n_extra) ll = ll + extra_prior<Num>(params[layout.extra_offset],data);
        if constexpr (Den::n_extra) ll = ll + extra_prior<Den>(params[layout.extra_offset+Num::n_extra],data);
    }
    return ll;
}

} }
#endif
