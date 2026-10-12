// lik_helpers.h
// Templated numerical helpers and per-family response payloads shared across
// tulpaRatio LikelihoodSpec implementations. Single source of truth for the
// link inverses, log-pmf / log-pdf kernels, and integer/continuous response
// data structs.
//
// All functions are `static inline` so each TU that includes this header sees
// its own copy with no ODR risk; the compiler inlines them at use sites.
//
// Templated on T so the same code path works for double (evaluation),
// tulpa::arena::Var (reverse AD), and fwd::Dual (forward AD).
//
// Numerical conventions
// ---------------------
// * exp / log / log1p / lgamma resolve via ADL on std:: for double, and via
//   the corresponding overload in `tulpa::arena` / `fwd` namespaces for the
//   AD types (provided by tulpa's autodiff_arena.h / autodiff_fwd.h).
// * Constant terms that depend only on integer counts (lgamma(y+1), lchoose,
//   log(y) for continuous y) are kept in the log-pmf for absolute parity with
//   the legacy log_post_impl.h code; the autodiff path treats them as zeros
//   in the derivative.

#ifndef TULPARATIO_LIK_HELPERS_H
#define TULPARATIO_LIK_HELPERS_H

#include <cmath>
#include <vector>
#include <tulpa/autodiff_arena.h>
#include <tulpa/autodiff_fwd.h>
#include <tulpa/portable_math.h>

namespace tulpaRatio {
namespace lik {

// ============================================================================
// Numerically stable scalar helpers
// ============================================================================

// log(1 + exp(x)) without overflow. Mirrors tulpaObs::safe_log1pexp.
template <typename T>
static inline T safe_log1pexp(const T& x) {
    using std::exp;
    using std::log1p;
    if (x > 35.0) return x;
    if (x < -10.0) return exp(x);
    return log1p(exp(x));
}

// log(1 - exp(x)) for x < 0, the log mass a zero-truncated distribution
// keeps. Near x = 0, 1 - exp(x) rounds to zero first; there
// -expm1(x) = -x (1 + x/2 + x^2/6 + x^3/24 + ...).
template <typename T>
static inline T log1m_exp(const T& x) {
    using std::log;
    using std::log1p;
    using std::exp;
    if (x > -1e-3) {
        return log(T(0.0) - x)
            + log1p(T(0.5) * x + T(1.0 / 6.0) * x * x + T(1.0 / 24.0) * x * x * x);
    }
    return log1p(T(0.0) - exp(x));
}

// log(sigmoid(x)) and log(1 - sigmoid(x)).
template <typename T>
static inline T log_inv_logit(const T& x) {
    return x - safe_log1pexp(x);
}
template <typename T>
static inline T log1m_inv_logit(const T& x) {
    return T(0.0) - safe_log1pexp(x);
}

// ----- log_lgamma ---------------------------------------------------------
// Wrapper around lgamma that resolves to std::lgamma for double and the
// AD-namespace lgamma for tulpa::arena::Var / fwd::Dual via ADL.
template <typename T>
static inline T log_lgamma(const T& x) {
    using std::lgamma;
    return lgamma(x);
}

// ============================================================================
// Per-family response payload structs (Pattern 2: per-family POD)
// ----------------------------------------------------------------------------
// One struct per family, each holding only the fields that family needs. The
// bridge (tulpa_bridge.cpp) constructs the right struct based on cfg.family,
// stashes a pointer in ModelData::model_response_data, and the templated
// likelihood callback reinterprets it. No tagged union, no junk fields.
//
// Continuous responses keep their own double vectors so families that mix
// integer / continuous (poisson_gamma, negbin_gamma) read each field cleanly.
// ============================================================================

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

// ============================================================================
// Log-pmf / log-pdf kernels shared across families
// ----------------------------------------------------------------------------
// Each family file calls these so the per-observation algebra appears in only
// one place. The mean parameterizations match the legacy autodiff_utils.h
// kernels; we restate them as comments for the math header inside each lik_*
// builder.
// ============================================================================

// ----- Binomial logit ------------------------------------------------------
// log P(y | n, p), p = sigmoid(eta).  Used for binomial.
template <typename T>
static inline T binom_logit_log_pmf(int y, int n, const T& eta) {
    T log_p   = log_inv_logit(eta);
    T log_1mp = log1m_inv_logit(eta);
    T lchoose = T(tulpa::math::portable_lchoose(n, y));
    return T(double(y)) * log_p + T(double(n - y)) * log_1mp + lchoose;
}

// ----- Beta-Binomial -------------------------------------------------------
// y ~ BetaBinomial(n, alpha, beta) with mean-precision parameterization:
//   p = sigmoid(eta), phi = precision.  alpha = p*phi, beta = (1-p)*phi.
// log P(y) = lgamma(y+alpha) + lgamma(n-y+beta) - lgamma(n+alpha+beta)
//           - lgamma(alpha) - lgamma(beta) + lgamma(alpha+beta)
//           + lchoose(n, y).
template <typename T>
static inline T beta_binomial_log_pmf(int y, int n, const T& eta, const T& phi) {
    using std::exp;
    // Use stable inv_logit via log_inv_logit for consistency, then exp once.
    T log_p   = log_inv_logit(eta);
    T log_1mp = log1m_inv_logit(eta);
    T p       = exp(log_p);
    T one_mp  = exp(log_1mp);

    T alpha = p      * phi;
    T beta  = one_mp * phi;

    T ll = log_lgamma(T(double(y)) + alpha)
         + log_lgamma(T(double(n - y)) + beta)
         - log_lgamma(T(double(n)) + alpha + beta);
    ll = ll - log_lgamma(alpha) - log_lgamma(beta) + log_lgamma(alpha + beta);
    ll = ll + T(tulpa::math::portable_lchoose(n, y));
    return ll;
}

// ----- Poisson with offset --------------------------------------------------
// Per-observation Poisson log-pmf with mean parameterization mu = exposure *
// exp(eta). Constant lgamma(y+1) kept for absolute parity.
//   log P(y | mu) = y*log(mu) - mu - lgamma(y+1).
// In poisson_gamma the legacy code parameterizes the *numerator* as
//   mu_num = exp(eta_num)        (no exposure multiplied through eta),
// so we expose two flavours: with-exposure and without.
template <typename T>
static inline T poisson_log_pmf(int y, const T& mu) {
    using std::log;
    T y_T = T(double(y));
    return y_T * log(mu) - mu - T(std::lgamma(double(y) + 1.0));
}

// ----- NegBin (mean / overdispersion) --------------------------------------
// y ~ NegBin(mu, phi) via Gamma-Poisson mixture:
//   log P(y) = lgamma(y + phi) - lgamma(y+1) - lgamma(phi)
//            + phi*log(phi/(mu+phi)) + y*log(mu/(mu+phi)).
template <typename T>
static inline T negbin_log_pmf(int y, const T& mu, const T& phi) {
    using std::log;
    T y_T = T(double(y));
    T ll  = log_lgamma(y_T + phi) - T(std::lgamma(double(y) + 1.0)) - log_lgamma(phi);
    T mp  = mu + phi;
    ll = ll + phi * log(phi / mp);
    ll = ll + y_T * log(mu / mp);
    return ll;
}

// ----- Gamma (shape / mean) ------------------------------------------------
// y ~ Gamma(shape, rate=shape/mu) (mean parameterization).
//   log p(y) = shape*log(rate) + (shape-1)*log(y) - rate*y - lgamma(shape).
// log(y) is constant w.r.t. parameters but kept for absolute parity.
template <typename T>
static inline T gamma_log_pdf(double y, const T& shape, const T& mu) {
    using std::log;
    T rate = shape / mu;
    return shape * log(rate)
         + (shape - T(1.0)) * T(std::log(y))
         - rate * y
         - log_lgamma(shape);
}

// ----- Lognormal -----------------------------------------------------------
// log(y) ~ Normal(mu_log, sigma).  mu_log is the linear predictor on the
// log scale, sigma is the std-dev (always positive — passed in as exp(log_sig)).
//   log p(y) = -log(y) - log(sigma) - 0.5*((log(y)-mu)/sigma)^2  (drops the
//             constant -0.5*log(2*pi) for parity with the legacy kernel).
template <typename T>
static inline T lognormal_log_pdf(double y, const T& mu, const T& sigma) {
    using std::log;
    double log_y = std::log(y);
    T z = (T(log_y) - mu) / sigma;
    return T(-log_y) - log(sigma) - T(0.5) * z * z;
}

// ============================================================================
// Zero-inflation / hurdle / one-inflation / zero-and-one-inflation kernels
// ----------------------------------------------------------------------------
// Mixture-form math from src/hmc_zi.h (the legacy hand-coded kernel) lifted to
// the templated AD path. Each kernel is parameterized by linear-predictor
// values (eta for the count process, logit_zi / logit_oi for the inflation
// processes) — tulpa's spec dispatcher pre-computes these from
// data.X_zi_flat / data.X_oi_flat at the per-observation level and feeds them
// to the LikelihoodFn, so families never need to do their own X_zi * beta_zi
// multiply.
//
// Stable log-sum-exp via max-shifted addition; no `if (small) -1e10` clamps —
// the AD path requires smooth derivatives across the whole support.
// ============================================================================

// log(exp(a) + exp(b)) = max(a,b) + log1p(exp(-|a-b|)) — smooth and AD-friendly.
template <typename T>
static inline T log_sum_exp_pair(const T& a, const T& b) {
    using std::abs;
    using std::log1p;
    using std::exp;
    using std::fabs;
    // Cannot use std::max on AD types without operator<; rewrite as a + log1p(exp(b-a))
    // when a >= b, b + log1p(exp(a-b)) otherwise. The branch is scalar-comparable
    // because tulpa::arena::Var and fwd::Dual support operator< against another T.
    // For numerical robustness use the symmetric form via fabs on the difference.
    T d  = a - b;
    T mx = (a > b) ? a : b;
    T ad = (d > T(0.0)) ? d : (T(0.0) - d);  // |a - b|
    return mx + log1p(exp(T(0.0) - ad));
}

// ----- ZI binomial ---------------------------------------------------------
// y ~ ZI-Binomial(n, p, pi):
//   P(Y=0) = pi + (1 - pi) * (1 - p)^n
//   P(Y=y) = (1 - pi) * Binomial(y; n, p),  y > 0
// p = sigmoid(eta), pi = sigmoid(logit_zi).
template <typename T>
static inline T zi_binom_log_pmf(int y, int n, const T& eta, const T& logit_zi) {
    using std::log;
    T log_zi   = log_inv_logit(logit_zi);
    T log_1mzi = log1m_inv_logit(logit_zi);
    if (y == 0) {
        T log_p0_count = T(double(n)) * log1m_inv_logit(eta);  // n * log(1 - p)
        return log_sum_exp_pair(log_zi, log_1mzi + log_p0_count);
    }
    return log_1mzi + binom_logit_log_pmf(y, n, eta);
}

// ----- Hurdle binomial -----------------------------------------------------
// theta = sigmoid(logit_zi)  (P(Y > 0)).
// P(Y=0)  = 1 - theta
// P(Y=y)  = theta * Binomial(y; n, p) / (1 - (1-p)^n),  y > 0
template <typename T>
static inline T hurdle_binom_log_pmf(int y, int n, const T& eta, const T& logit_zi) {
    using std::log;
    using std::log1p;
    using std::exp;
    if (y == 0) {
        return log1m_inv_logit(logit_zi);
    }
    T log_p0  = T(double(n)) * log1m_inv_logit(eta);  // log((1-p)^n)
    T log_norm = log1m_exp(log_p0);
    return log_inv_logit(logit_zi) + binom_logit_log_pmf(y, n, eta) - log_norm;
}

// ----- One-inflated binomial -----------------------------------------------
// psi = sigmoid(logit_oi)  (probability of structural one).
// P(Y=n) = psi + (1 - psi) * p^n
// P(Y=y) = (1 - psi) * Binomial(y; n, p),  y < n
template <typename T>
static inline T oi_binom_log_pmf(int y, int n, const T& eta, const T& logit_oi) {
    using std::log;
    T log_oi   = log_inv_logit(logit_oi);
    T log_1moi = log1m_inv_logit(logit_oi);
    if (y == n) {
        T log_pn_count = T(double(n)) * log_inv_logit(eta);  // n * log(p)
        return log_sum_exp_pair(log_oi, log_1moi + log_pn_count);
    }
    return log_1moi + binom_logit_log_pmf(y, n, eta);
}

// ----- Zero-and-one-inflated binomial --------------------------------------
// pi_0 = sigmoid(logit_zi), pi_1 = sigmoid(logit_oi).
// "Extended" ZOIB form (matches legacy autodiff_utils.h::log_lik_zoib): the
// binomial process runs conditional on neither structural zero nor structural
// one, and CAN itself emit 0 or n, so the boundary masses include both the
// structural and binomial-tail contributions:
//   P(Y=0) = pi_0 + (1 - pi_0) * (1 - pi_1) * (1 - p)^n
//   P(Y=n) = (1 - pi_0) * [pi_1 + (1 - pi_1) * p^n]
//   P(Y=y) = (1 - pi_0) * (1 - pi_1) * Binomial(y; n, p),  0 < y < n
template <typename T>
static inline T zoib_log_pmf(int y, int n, const T& eta,
                              const T& logit_zi, const T& logit_oi) {
    T log_pi0    = log_inv_logit(logit_zi);
    T log_1m_pi0 = log1m_inv_logit(logit_zi);
    T log_pi1    = log_inv_logit(logit_oi);
    T log_1m_pi1 = log1m_inv_logit(logit_oi);
    if (y == 0) {
        T log_p0_count = T(double(n)) * log1m_inv_logit(eta);  // n * log(1 - p)
        return log_sum_exp_pair(log_pi0,
                                log_1m_pi0 + log_1m_pi1 + log_p0_count);
    }
    if (y == n) {
        T log_pn_count = T(double(n)) * log_inv_logit(eta);    // n * log(p)
        return log_1m_pi0 + log_sum_exp_pair(log_pi1,
                                             log_1m_pi1 + log_pn_count);
    }
    return log_1m_pi0 + log_1m_pi1 + binom_logit_log_pmf(y, n, eta);
}

// ----- ZI Poisson ----------------------------------------------------------
// pi = sigmoid(logit_zi), mu = exp(eta).
// P(Y=0) = pi + (1 - pi) * exp(-mu)
// P(Y=y) = (1 - pi) * Poisson(y; mu),  y > 0
template <typename T>
static inline T zi_poisson_log_pmf(int y, const T& mu, const T& logit_zi) {
    T log_zi   = log_inv_logit(logit_zi);
    T log_1mzi = log1m_inv_logit(logit_zi);
    if (y == 0) {
        T log_p0_count = T(0.0) - mu;  // -mu
        return log_sum_exp_pair(log_zi, log_1mzi + log_p0_count);
    }
    return log_1mzi + poisson_log_pmf(y, mu);
}

// ----- Hurdle Poisson ------------------------------------------------------
// theta = sigmoid(logit_zi), mu = exp(eta).
// P(Y=0) = 1 - theta
// P(Y=y) = theta * Poisson(y; mu) / (1 - exp(-mu)),  y > 0
template <typename T>
static inline T hurdle_poisson_log_pmf(int y, const T& mu, const T& logit_zi) {
    using std::log1p;
    using std::exp;
    if (y == 0) {
        return log1m_inv_logit(logit_zi);
    }
    T log_norm = log1m_exp(T(0.0) - mu);
    return log_inv_logit(logit_zi) + poisson_log_pmf(y, mu) - log_norm;
}

// ----- ZI NegBin -----------------------------------------------------------
// pi = sigmoid(logit_zi).
// P_NB(0) = (phi / (phi + mu))^phi  =>  log_p0_count = phi * log(phi/(phi+mu))
template <typename T>
static inline T zi_negbin_log_pmf(int y, const T& mu, const T& phi,
                                   const T& logit_zi) {
    using std::log;
    T log_zi   = log_inv_logit(logit_zi);
    T log_1mzi = log1m_inv_logit(logit_zi);
    if (y == 0) {
        T log_p0_count = phi * log(phi / (phi + mu));
        return log_sum_exp_pair(log_zi, log_1mzi + log_p0_count);
    }
    return log_1mzi + negbin_log_pmf(y, mu, phi);
}

// ----- Hurdle NegBin -------------------------------------------------------
// theta = sigmoid(logit_zi).
template <typename T>
static inline T hurdle_negbin_log_pmf(int y, const T& mu, const T& phi,
                                       const T& logit_zi) {
    using std::log;
    using std::log1p;
    using std::exp;
    if (y == 0) {
        return log1m_inv_logit(logit_zi);
    }
    T log_norm = log1m_exp(phi * log(phi / (phi + mu)));
    return log_inv_logit(logit_zi) + negbin_log_pmf(y, mu, phi) - log_norm;
}

// ============================================================================
// Priors on extra parameters (added inside the per-obs likelihood at i==0)
// ----------------------------------------------------------------------------
// The B1b dispatcher does NOT register an `extra_prior` callback because that
// forces tulpa's gradient dispatcher onto numerical gradients (see
// hmc_gradient_dispatch.h:36). Instead, the per-obs templated likelihood adds
// the prior contribution exactly once at i==0 — autodiff differentiates
// through it cleanly. Single source of truth shared across every family that
// has extra parameters.
// ============================================================================

// Gamma prior on a phi parameter, with the parameterization used in
// log_post_impl.h:
//   p(log_phi)  ∝  (shape-1)*log_phi - rate*phi + log_phi   (Jacobian-adjusted)
template <typename T>
static inline T log_prior_gamma_log(const T& log_phi, double shape, double rate) {
    using std::exp;
    T phi = exp(log_phi);
    return T(shape - 1.0) * log_phi - T(rate) * phi + log_phi;
}

// Half-Cauchy prior on a sigma parameter on log-scale: matches
// log_prior_half_cauchy in autodiff_utils.h:
//   p(log_sigma) ∝ -log(1 + (sigma/scale)^2) + log_sigma
template <typename T>
static inline T log_prior_half_cauchy_log(const T& log_sigma, double scale) {
    using std::exp;
    using std::log;
    T sigma = exp(log_sigma);
    T ratio = sigma / T(scale);
    return T(0.0) - log(T(1.0) + ratio * ratio) + log_sigma;
}

}  // namespace lik
}  // namespace tulpaRatio

#endif  // TULPARATIO_LIK_HELPERS_H
