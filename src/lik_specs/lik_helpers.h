#ifndef TULPARATIO_LIK_HELPERS_H
#define TULPARATIO_LIK_HELPERS_H
#include <vector>
#include "../ratio_margins.h"
#include "../hmc_zi.h"
namespace tulpaRatio { namespace lik {
using ratiod::math::log_prior_gamma;
using tulpa::math::log_prior_half_cauchy;

struct ResponseData {
    std::vector<int> y_num, y_denom;
    std::vector<double> y_num_cont, y_denom_cont;
};

} }
#endif
