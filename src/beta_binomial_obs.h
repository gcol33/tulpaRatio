// beta_binomial_obs.h
// One beta-binomial observation's log-likelihood and its derivatives in the
// logit eta of the mean and in the precision phi, for every hand-coded
// gradient.

#ifndef RATIOD_BETA_BINOMIAL_OBS_H
#define RATIOD_BETA_BINOMIAL_OBS_H

#include <cmath>
#include "portable_math.h"

namespace ratiod {
namespace math {

// The parts of the log-likelihood and of d/d phi that vary across
// observations; a caller adds lgamma(phi) to ll, digamma(phi) to d_phi and
// lchoose(n, y), which it can hoist or cache.
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

    const auto [psi_y_alpha, lg_y_alpha] = portable_digamma_lgamma(y + alpha);
    const auto [psi_nmy_beta, lg_nmy_beta] = portable_digamma_lgamma(n - y + beta);
    const auto [psi_alpha, lg_alpha] = portable_digamma_lgamma(alpha);
    const auto [psi_beta, lg_beta] = portable_digamma_lgamma(beta);
    const auto [psi_n_phi, lg_n_phi] = portable_digamma_lgamma(n + phi);

    BetaBinomialObs out;
    out.ll = lg_y_alpha + lg_nmy_beta - lg_n_phi - lg_alpha - lg_beta;
    // d ll / d p = phi (psi(y + alpha) - psi(n - y + beta) - psi(alpha) + psi(beta)),
    // d p / d eta = p (1 - p)
    out.d_eta = phi * (psi_y_alpha - psi_nmy_beta - psi_alpha + psi_beta) * p * one_m_p;
    out.d_phi = p * (psi_y_alpha - psi_alpha) + one_m_p * (psi_nmy_beta - psi_beta)
        - psi_n_phi;
    return out;
}

}  // namespace math
}  // namespace ratiod

#endif  // RATIOD_BETA_BINOMIAL_OBS_H
