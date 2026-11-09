#ifndef TULPARATIO_RATIO_COLLAPSED_LIK_H
#define TULPARATIO_RATIO_COLLAPSED_LIK_H

#include "ratio_margins.h"

namespace ratiod_hmc {

template<class Num, class Den = tulpaRatio::margins::NoMargin,
         unsigned Fields = tulpaRatio::margins::SecondOrder>
inline tulpaRatio::margins::Result collapsed_observation_pair(
    const ModelData& data, int i, double en, double ed, double pn, double pd,
    bool shared, bool trials = false, double* rn = nullptr, double* rd = nullptr) {
    using namespace tulpaRatio::margins;
    const Observation xn{std::is_same<Num,Gamma>::value || std::is_same<Num,Lognormal>::value
        ? data.y_num_cont[i] : double(data.y_num[i]), en,
        trials ? data.y_denom[i] : 0};
    auto out = Num(pn).template evaluate<false,Fields>(xn);
    if (rn) *rn=out.residual;
    if (rd) *rd=0.;
    if constexpr (!std::is_same<Den,NoMargin>::value) {
        if (shared) {
            const Observation xd{std::is_same<Den,Gamma>::value || std::is_same<Den,Lognormal>::value
                ? data.y_denom_cont[i] : double(data.y_denom[i]), ed};
            const auto b = Den(pd).template evaluate<false,Fields>(xd);
            if (rd) *rd=b.residual;
            out.ll += b.ll;
            out.residual += b.residual;
            out.neg_hess += b.neg_hess;
        }
    }
    return out;
}

template<class Num, class Den> struct CollapsedFamily {
    using numerator=Num;
    using denominator=Den;
};

template<class Fn>
auto visit_collapsed_family(ModelType type, Fn fn) {
    using namespace tulpaRatio::margins;
    switch (type) {
        case ModelType::BINOMIAL: return fn(CollapsedFamily<Binomial,NoMargin>{});
        case ModelType::BETA_BINOMIAL: return fn(CollapsedFamily<BetaBinomial,NoMargin>{});
        case ModelType::POISSON_GAMMA: return fn(CollapsedFamily<Poisson,Gamma>{});
        case ModelType::NEGBIN_GAMMA: return fn(CollapsedFamily<NegBin,Gamma>{});
        case ModelType::NEGBIN_NEGBIN: return fn(CollapsedFamily<NegBin,NegBin>{});
        case ModelType::GAMMA_GAMMA: return fn(CollapsedFamily<Gamma,Gamma>{});
        case ModelType::LOGNORMAL: return fn(CollapsedFamily<Lognormal,Lognormal>{});
        default: Rcpp::stop("Unsupported collapsed observation family");
    }
}

template<unsigned Fields = tulpaRatio::margins::SecondOrder>
inline tulpaRatio::margins::Result collapsed_observation(
    const ModelData& data, int i, double en, double ed, double pn, double pd, bool shared,
    double* rn = nullptr, double* rd = nullptr) {
    return visit_collapsed_family(data.model_type,[&](auto tag) {
        using Num=typename decltype(tag)::numerator;
        using Den=typename decltype(tag)::denominator;
        constexpr bool single=std::is_same<Den,tulpaRatio::margins::NoMargin>::value;
        return collapsed_observation_pair<Num,Den,Fields>(data,i,en,ed,pn,pd,shared,single,rn,rd);
    });
}

inline auto collapsed_extra_scores(const ModelData& data, int i, double en, double ed,
                                   double pn, double pd, bool shared) {
    using namespace tulpaRatio::margins;
    return visit_collapsed_family(data.model_type,[&](auto tag) {
        using Num=typename decltype(tag)::numerator;
        using Den=typename decltype(tag)::denominator;
        const Observation xn{std::is_same<Num,Gamma>::value || std::is_same<Num,Lognormal>::value
            ? data.y_num_cont[i] : double(data.y_num[i]), en,
            std::is_same<Den,NoMargin>::value ? data.y_denom[i] : 0};
        const auto a=Num(pn,false).template evaluate<false,Extra>(xn,false);
        double b=0.;
        if constexpr (!std::is_same<Den,NoMargin>::value) {
            if (shared) {
                const Observation xd{std::is_same<Den,Gamma>::value || std::is_same<Den,Lognormal>::value
                    ? data.y_denom_cont[i] : double(data.y_denom[i]), ed};
                b=Den(pd,false).template evaluate<false,Extra>(xd,false).extra;
            }
        }
        return std::make_pair(Num::log_extra ? a.extra : pn*a.extra,Den::log_extra ? b : pd*b);
    });
}

inline auto collapsed_curvature_pair(const ModelData& data, int i, double en, double ed,
                                     double pn, double pd, bool shared) {
    using namespace tulpaRatio::margins;
    return visit_collapsed_family(data.model_type,[&](auto tag) {
        using Num=typename decltype(tag)::numerator;
        using Den=typename decltype(tag)::denominator;
        const Observation xn{std::is_same<Num,Gamma>::value || std::is_same<Num,Lognormal>::value
            ? data.y_num_cont[i] : double(data.y_num[i]), en,
            std::is_same<Den,NoMargin>::value ? data.y_denom[i] : 0};
        auto a=curvature_derivatives<Num>(xn,pn);
        CurvatureDerivatives b;
        if constexpr (!std::is_same<Den,NoMargin>::value) {
            if (shared) {
                const Observation xd{std::is_same<Den,Gamma>::value || std::is_same<Den,Lognormal>::value
                    ? data.y_denom_cont[i] : double(data.y_denom[i]), ed};
                b=curvature_derivatives<Den>(xd,pd);
            }
        }
        return std::make_pair(a,b);
    });
}

}
#endif
