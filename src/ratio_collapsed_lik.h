#ifndef TULPARATIO_RATIO_COLLAPSED_LIK_H
#define TULPARATIO_RATIO_COLLAPSED_LIK_H

#include "ratio_margins.h"

namespace ratiod_hmc {

template<class Num, class Den = tulpaRatio::margins::NoMargin>
inline tulpaRatio::margins::Result collapsed_observation_pair(
    const ModelData& data, int i, double en, double ed, double pn, double pd,
    bool shared, bool trials = false) {
    using namespace tulpaRatio::margins;
    const Observation xn{std::is_same<Num,Gamma>::value || std::is_same<Num,Lognormal>::value
        ? data.y_num_cont[i] : double(data.y_num[i]), en,
        trials ? data.y_denom[i] : 0};
    auto out = Num(pn).template operator()<false,SecondOrder>(xn);
    if constexpr (!std::is_same<Den,NoMargin>::value) {
        if (shared) {
            const Observation xd{std::is_same<Den,Gamma>::value || std::is_same<Den,Lognormal>::value
                ? data.y_denom_cont[i] : double(data.y_denom[i]), ed};
            const auto b = Den(pd).template operator()<false,SecondOrder>(xd);
            out.ll += b.ll;
            out.residual += b.residual;
            out.neg_hess += b.neg_hess;
        }
    }
    return out;
}

inline tulpaRatio::margins::Result collapsed_observation(
    const ModelData& data, int i, double en, double ed, double pn, double pd, bool shared) {
    using namespace tulpaRatio::margins;
    switch (data.model_type) {
        case ModelType::BINOMIAL:
            return collapsed_observation_pair<Binomial>(data,i,en,ed,pn,pd,false,true);
        case ModelType::BETA_BINOMIAL:
            return collapsed_observation_pair<BetaBinomial>(data,i,en,ed,pn,pd,false,true);
        case ModelType::POISSON_GAMMA:
            return collapsed_observation_pair<Poisson,Gamma>(data,i,en,ed,pn,pd,shared);
        case ModelType::NEGBIN_GAMMA:
            return collapsed_observation_pair<NegBin,Gamma>(data,i,en,ed,pn,pd,shared);
        case ModelType::NEGBIN_NEGBIN:
            return collapsed_observation_pair<NegBin,NegBin>(data,i,en,ed,pn,pd,shared);
        case ModelType::GAMMA_GAMMA:
            return collapsed_observation_pair<Gamma,Gamma>(data,i,en,ed,pn,pd,shared);
        case ModelType::LOGNORMAL:
            return collapsed_observation_pair<Lognormal,Lognormal>(data,i,en,ed,pn,pd,shared);
        default:
            Rcpp::stop("Unsupported collapsed observation family");
    }
}

}
#endif
