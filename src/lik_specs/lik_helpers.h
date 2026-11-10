#ifndef TULPARATIO_LIK_HELPERS_H
#define TULPARATIO_LIK_HELPERS_H
#include <vector>
#include "../ratio_margins.h"
namespace tulpaRatio { namespace lik {
using ratiod::math::log_prior_gamma;
using tulpa::math::log_prior_half_cauchy;

struct ResponseData {
    std::vector<int> y_num, y_denom;
    std::vector<double> y_num_cont, y_denom_cont;
    std::vector<double> exposure;
    ratiod_link::LinkType num_link = ratiod_link::LinkType::LOGIT;
    ratiod_link::LinkType denom_link = ratiod_link::LinkType::LOG;
    ratiod_link::LinkType zi_link = ratiod_link::LinkType::LOGIT;
    ratiod_link::LinkType oi_link = ratiod_link::LinkType::LOGIT;
};

} }
#endif
