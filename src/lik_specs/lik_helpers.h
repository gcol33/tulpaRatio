#ifndef TULPARATIO_LIK_HELPERS_H
#define TULPARATIO_LIK_HELPERS_H
#include <vector>
#include "../ratio_margins.h"
#include "../hmc_zi.h"
namespace tulpaRatio { namespace lik {
using ratiod::math::log_prior_gamma;
using tulpa::math::log_prior_half_cauchy;
using ratiod::math::log_lik_binomial;
using ratiod::math::log_lik_beta_binomial;
using ratiod::math::log_lik_poisson;
using ratiod::math::log_lik_negbin;
using ratiod::math::log_lik_gamma;
using ratiod::math::log_lik_lognormal;
using ratiod::math::log_lik_zi_binomial;
using ratiod::math::log_lik_hurdle_binomial;
using ratiod::math::log_lik_oi_binomial;
using ratiod::math::log_lik_zoib;
using ratiod_zi::zi_poisson_lpmf_logit;
using ratiod_zi::hurdle_poisson_lpmf_logit;
using ratiod_zi::zi_negbin_lpmf_logit;
using ratiod_zi::hurdle_negbin_lpmf_logit;
struct BinomialResponseData {
    std::vector<int> y;  // numerator successes, length N
    std::vector<int> n;  // trial counts (denominator), length N
};

struct BetaBinomialResponseData {
    std::vector<int> y;
    std::vector<int> n;
};

struct PoissonGammaResponseData {
    std::vector<int>    y_num;        // count numerator
    std::vector<double> y_denom_cont; // continuous denominator (Gamma response)
};

struct NegbinGammaResponseData {
    std::vector<int>    y_num;        // count numerator (NegBin)
    std::vector<double> y_denom_cont; // continuous denominator (Gamma)
};

struct NegbinNegbinResponseData {
    std::vector<int> y_num;
    std::vector<int> y_denom;
};

struct GammaGammaResponseData {
    std::vector<double> y_num_cont;
    std::vector<double> y_denom_cont;
};

struct LognormalResponseData {
    std::vector<double> y_num_cont;
    std::vector<double> y_denom_cont;
};
} }
#endif
