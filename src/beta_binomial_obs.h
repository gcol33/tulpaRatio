// beta_binomial_obs.h
// One beta-binomial observation's log-likelihood and its derivatives in the
// logit eta of the mean and in the precision phi, for every hand-coded
// gradient.

#ifndef RATIOD_BETA_BINOMIAL_OBS_H
#define RATIOD_BETA_BINOMIAL_OBS_H

#include <cmath>
#include "portable_math.h"
#include <tulpa/count_math.h>

namespace ratiod {
namespace math {

// The log density without its data-only binomial coefficient, and both scores.
struct BetaBinomialObs {
    double ll;
    double d_eta;
    double d_phi;
};

// y ~ BetaBinomial(n, p phi, (1 - p) phi), p = inv_logit(eta). p and 1 - p
// are formed separately so neither rounds to zero at large |eta|.
inline BetaBinomialObs beta_binomial_obs(int y, int n, double eta, double phi) {
    const double p = 1.0 / (1.0 + std::exp(-eta));
    const double one_m_p = 1.0 / (1.0 + std::exp(eta));
    const double alpha = p * phi;
    const double beta = one_m_p * phi;

    BetaBinomialObs out;
    out.ll = tulpa::math::beta_binomial_log_pmf(y,n,p,one_m_p,phi,0.0);
    out.d_eta = phi*p*one_m_p*(tulpa::math::polygamma_rising_difference(0,y,alpha)
        - tulpa::math::polygamma_rising_difference(0,n-y,beta));
    out.d_phi = tulpa::math::beta_binomial_phi_derivative(y,n,p,one_m_p,phi);
    return out;
}

}  // namespace math
}  // namespace ratiod

#endif  // RATIOD_BETA_BINOMIAL_OBS_H
