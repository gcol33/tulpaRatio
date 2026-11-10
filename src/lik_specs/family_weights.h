#ifndef TULPARATIO_FAMILY_WEIGHTS_H
#define TULPARATIO_FAMILY_WEIGHTS_H

#include "family_kernel.h"

namespace tulpaRatio { namespace lik {

template<class Margin>
std::pair<double, double> margin_weights(double y, int trials, double eta, double phi) {
    const auto result = margins::curvature_derivatives<Margin>({y, eta, trials}, phi);
    return {result.score, result.w};
}

template<class Margin>
std::pair<tulpa::arena::Var, tulpa::arena::Var> margin_weights(double y, int trials,
    const tulpa::arena::Var& eta, const tulpa::arena::Var& phi) {
    const auto result = margins::curvature_derivatives<Margin>({y, eta.val(), trials}, phi.val());
    auto* arena = eta.arena_;
    std::vector<int32_t> indices;
    arena->add_custom_backward({eta.idx_, phi.idx_}, {result.score, result.w},
        [result](const double*, int, const double*, int, const double* adjoints, double* gradient) {
            gradient[0] = -result.w * adjoints[0] + result.eta * adjoints[1];
            gradient[1] = result.score_phi * adjoints[0] + result.phi * adjoints[1];
        }, indices);
    tulpa::arena::Var score, weight;
    score.arena_ = weight.arena_ = arena;
    score.idx_ = indices[0]; weight.idx_ = indices[1];
    return {score, weight};
}

template<class Num, class Den, class T>
void family_weights(int i, const T* eta, tulpa::EtaWeightScalar<T>, tulpa::EtaWeightScalar<T>,
    const std::vector<T>& params, const tulpa::ModelData& data,
    const tulpa::ParamLayout& layout, const void* payload, T* score, T* weights) {
    using std::exp;
    if constexpr (std::is_same_v<T, tulpa::arena::Var>) tulpa::arena::current_arena() = eta[0].arena_;
    const auto& response = *static_cast<const ResponseData*>(payload);
    T phi_num(1), phi_den(1);
    if constexpr (Num::n_extra) phi_num = exp(params[layout.extra_offset]);
    if constexpr (Den::n_extra) phi_den = exp(params[layout.extra_offset + Num::n_extra]);
    const double y_num = continuous_response<Num> ? response.y_num_cont[i] : response.y_num[i];
    const int trials = std::is_same_v<Num, margins::Binomial> || std::is_same_v<Num, margins::BetaBinomial>
        ? response.y_denom[i] : 0;
    const auto numerator = margin_weights<Num>(y_num, trials, eta[0], phi_num);
    for (int k = 0; k < data.n_processes * data.n_processes; ++k) weights[k] = T(0);
    score[0] = numerator.first; weights[0] = numerator.second;
    if constexpr (!std::is_same_v<Den, margins::NoMargin>) {
        const double y_den = continuous_response<Den> ? response.y_denom_cont[i] : response.y_denom[i];
        const auto denominator = margin_weights<Den>(y_den, 0, eta[1], phi_den);
        score[1] = denominator.first; weights[3] = denominator.second;
    }
}

} }
#endif
