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
    T num_eta = eta[0];
    if (!r.exposure.empty()) {
        if constexpr (std::is_same<Num,margins::Lognormal>::value)
            num_eta = num_eta + T(std::log(r.exposure[i]));
        else if (r.num_link == ratiod_link::LinkType::LOG)
            num_eta = num_eta + T(std::log(r.exposure[i]));
        else {
            const T mean = ratiod::math::link_mean(num_eta, r.num_link) * T(r.exposure[i]);
            if (!(ratiod::math::get_value(mean) > 0)) return T(-INFINITY);
            num_eta = ratiod::math::safe_log(mean);
        }
    }
    const auto num_link = r.exposure.empty() ? r.num_link :
        (std::is_same<Num,margins::Lognormal>::value ? r.num_link : ratiod_link::LinkType::LOG);
    T ll = margins::linked_density<Num>(yn,n,num_eta,pn,data.zi_type,logit_zi,logit_oi,num_link,r.zi_link,r.oi_link);
    if constexpr (!std::is_same<Den,margins::NoMargin>::value) {
        const double yd = continuous_response<Den> ? r.y_denom_cont[i] : r.y_denom[i];
        ll = ll + margins::linked_density<Den>(yd,0,eta[1],pd,tulpa::ZIType::NONE,logit_zi,logit_oi,r.denom_link);
    }
    if (i == 0) {
        if constexpr (Num::n_extra) ll = ll + extra_prior<Num>(params[layout.extra_offset],data);
        if constexpr (Den::n_extra) ll = ll + extra_prior<Den>(params[layout.extra_offset+Num::n_extra],data);
    }
    return ll;
}

} }
#endif
