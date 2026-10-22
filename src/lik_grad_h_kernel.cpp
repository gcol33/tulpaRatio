// lik_grad_h_kernel.cpp
// =====================================================================
// MATH HEADER (per CLAUDE.md "Math ground first")
// =====================================================================
//
// All 7 ratio families share the same scaffolding for B2:
//   * params layout: [beta_0 | beta_1 | ... | extras]
//                    where beta_k starts at layout.process_beta_start[k] and
//                    has length layout.process_beta_count[k].
//                    Extras (log_phi / log_sigma / log_shape) live at
//                    layout.extra_offset, length layout.n_extra_params.
//   * eta_k[i] = X_k[i] . beta_k + offset_k[i]    (process k, obs i)
//   * Log posterior: log P(params|data) = log L(data|params) + log Pi(params)
//   * Log Pi(params) = sum_k sum_j  log N(beta_kj | 0, sigma_beta^2)
//                    + family-specific extra-parameter priors
//
// Per-observation gradient pattern
// --------------------------------
// For every family the per-obs likelihood depends on params only through the
// linear predictors eta_k[i] and a small set of extra parameters. Define
//   resid_k(i) := d log L_i / d eta_k[i]
//   extra_k(i) := d log L_i / d extra_k    (per-obs contribution)
// Then
//   d log L / d beta_kj = sum_i  resid_k(i) * X_k[i,j]
//   d log L / d extra_k = sum_i  extra_k(i)
//
// Each family supplies a per-obs functor that fills resid[k] and the extra
// gradients; the shared scaffolding does the X^T R accumulation, the
// Gaussian beta prior gradient, the extra-parameter priors, and the fused
// log-posterior accumulation.
//
// Per-family residuals (closed-form, ported from
// tulpa/src/hmc_gradient_analytical_lik_scalar.h)
// ---------------------------------------------------------------------
//   binomial:       resid_num = y - n*p,                 p = sigmoid(eta)
//   poisson_gamma:  resid_num   = y_num - mu_num,        mu_num = exp(eta_num)
//                   resid_denom = shape*(1 - y_d/mu_d),  mu_d  = exp(eta_d)
//                   d/d log_shape = sum_i [shape*(log(rate_i) + 1 + log(y_d_i)
//                                          - digamma(shape) - rate_i*y_d_i/shape)]
//   negbin_gamma:   numerator = NegBin scalar resid (see negbin block);
//                   denominator = Gamma resid (see gamma_gamma block).
//   negbin_negbin:  numerator + denominator both NegBin scalar resids
//     resid_k = y_k - mu_k*(y_k+phi_k)/(mu_k+phi_k)
//     d/dlog_phi_k = phi_k * [digamma(y+phi) - digamma(phi) + log(phi/(mu+phi))
//                              + (mu - y)/(mu+phi)]
//   gamma_gamma:    Gamma response on each process (only y > 0 contributes):
//     resid_k = phi_k*(y_k/mu_k - 1)
//     d/dlog_phi_k = phi_k * (log(rate) + 1 + log(y) - digamma(phi) - y/mu)
//   lognormal:      eta_k IS mu_k on log scale; sigma_k = exp(log_sigma_k):
//     resid_k = (log y_k - mu_k) / sigma_k^2
//     d/dlog_sigma_k = -1 + z_k^2,   z_k = (log y_k - mu_k)/sigma_k
//   beta_binomial:  p = sigmoid(eta), phi = exp(log_phi),
//                   alpha = p*phi, beta = (1-p)*phi:
//     dLL/dp = phi*(psi(y+a) - psi(n-y+b) - psi(a) + psi(b))
//     resid  = dLL/dp * p*(1-p)
//     dLL/dphi = p*psi(y+a) + (1-p)*psi(n-y+b) - psi(n+phi)
//                 - p*psi(a) - (1-p)*psi(b) + psi(phi)
//     d/dlog_phi = phi * dLL/dphi
//
// Extra-parameter priors (from B1b lik_helpers.h)
// ----------------------------------------------
//   Gamma(shape, rate) on log_phi:
//     log Pi(log_phi) = (shape-1)*log_phi - rate*phi + log_phi  (Jacobian)
//     d/dlog_phi = shape - rate * phi
//   Half-Cauchy(0, scale) on log_sigma:
//     log Pi(log_sigma) = -log(1 + (sigma/scale)^2) + log_sigma
//     d/dlog_sigma = -2*r^2/(1+r^2) + 1, r = sigma/scale
// =====================================================================

#include "lik_specs/lik_grad_h_kernel.h"
#include "lik_specs/lik_helpers.h"
#include "beta_binomial_obs.h"

#include <Rcpp.h>
#include <cmath>
#include <vector>
#include <tulpa/likelihood.h>
#include <tulpa/portable_math.h>

namespace tulpaRatio {

namespace {

// ----- Layout-feature guard ---------------------------------------------
// B2 covers the same scope as B1b: betas only, optional extras, NO spatial /
// temporal / RE / ZI / OI / SVC / TVC / latent / ST. Bail out loudly if any
// of those flags are set so we never silently produce a wrong gradient.
inline void assert_no_latent_structure(const tulpa::ParamLayout& layout) {
    const bool any_latent =
        layout.has_re || layout.has_re_slopes ||
        layout.has_spatial || layout.is_gp || layout.is_multiscale_gp ||
        layout.is_hsgp || layout.is_icar_collapsed || layout.is_bym2_collapsed ||
        layout.has_temporal || layout.has_multiscale_temporal ||
        layout.is_temporal_gp || layout.has_zi || layout.has_oi ||
        layout.has_svc || layout.has_tvc || layout.has_latent ||
        layout.has_spatiotemporal;
    if (any_latent) {
        Rcpp::stop("tulpaRatio H-kernel: latent structure (spatial/temporal/RE/"
                   "ZI/SVC/TVC/ST) not supported in B2 — use A_r mode for these.");
    }
}

// ----- Beta Gaussian prior + ll accumulator -----------------------------
// Adds d/dbeta_kj of N(0, sigma_beta^2) prior into grad[] and accumulates
// the prior log-density into log_post (when log_post_out != nullptr).
inline void apply_beta_prior(
    const std::vector<double>& params,
    const tulpa::ModelData& data,
    const tulpa::ParamLayout& layout,
    std::vector<double>& grad,
    double* log_post_out
) {
    const double sigma_beta = data.sigma_beta;
    const double tau_beta = 1.0 / (sigma_beta * sigma_beta);
    const double log_norm = -0.5 * std::log(2.0 * M_PI * sigma_beta * sigma_beta);

    for (int k = 0; k < data.n_processes; ++k) {
        const int start = layout.process_beta_start[k];
        const int p_k   = layout.process_beta_count[k];
        for (int j = 0; j < p_k; ++j) {
            const double b = params[start + j];
            grad[start + j] += -tau_beta * b;
            if (log_post_out) {
                *log_post_out += log_norm - 0.5 * tau_beta * b * b;
            }
        }
    }
}

// ----- Compute eta_k[i] for every process k, every obs i ----------------
// Returns a row-major [n_processes x N] matrix. Includes per-process offset
// when present. Pure double (no AD needed at H-mode).
inline std::vector<double> compute_eta_matrix(
    const std::vector<double>& params,
    const tulpa::ModelData& data,
    const tulpa::ParamLayout& layout
) {
    const int N  = data.N;
    const int np = data.n_processes;
    std::vector<double> eta(static_cast<size_t>(np) * N, 0.0);

    for (int k = 0; k < np; ++k) {
        const auto& proc = data.processes[k];
        const int p_k    = proc.p;
        const int bstart = layout.process_beta_start[k];
        double*       eta_k   = eta.data() + static_cast<size_t>(k) * N;
        const double* X_k     = proc.X_flat.data();
        const double* beta_k  = params.data() + bstart;

        if (p_k > 0) {
            for (int i = 0; i < N; ++i) {
                double s = 0.0;
                const double* row = X_k + static_cast<size_t>(i) * p_k;
                for (int j = 0; j < p_k; ++j) s += row[j] * beta_k[j];
                eta_k[i] = s;
            }
        }
        if (!proc.offset.empty()) {
            if (static_cast<int>(proc.offset.size()) != N) {
                Rcpp::stop("ProcessData[%d]: offset length %d != N=%d",
                           k, (int)proc.offset.size(), N);
            }
            for (int i = 0; i < N; ++i) eta_k[i] += proc.offset[i];
        }
    }
    return eta;
}

// ----- Scatter sum_i resid_k(i) * X_k[i,j]  into  grad_beta_k[] ---------
// Single source of truth for the X^T r reduction. Each family's Functor
// supplies resid_k(i) for k in [0, n_processes); the scaffold does the
// rest. extra_grad_out[e] receives sum_i (per-obs partial of extra_e).
template <class PerObsFn>
inline double run_obs_loop(
    const std::vector<double>& params,
    const tulpa::ModelData& data,
    const tulpa::ParamLayout& layout,
    const double* eta,                  // [np x N]
    PerObsFn fn,
    std::vector<double>& grad,
    int n_extra_grads                  // length of extra_grad_out
) {
    const int N  = data.N;
    const int np = data.n_processes;
    std::vector<double> resid(np, 0.0);
    std::vector<double> extra(n_extra_grads, 0.0);
    std::vector<double> extra_acc(n_extra_grads, 0.0);
    double log_lik = 0.0;

    for (int i = 0; i < N; ++i) {
        for (int k = 0; k < np; ++k) resid[k] = 0.0;
        for (int e = 0; e < n_extra_grads; ++e) extra[e] = 0.0;

        const double ll_i = fn(i, eta, resid.data(), extra.data());
        log_lik += ll_i;

        // Scatter beta gradients
        for (int k = 0; k < np; ++k) {
            const auto& proc = data.processes[k];
            const int p_k = proc.p;
            if (p_k == 0 || resid[k] == 0.0) continue;
            const int bstart = layout.process_beta_start[k];
            const double* row = &proc.X_flat[static_cast<size_t>(i) * p_k];
            for (int j = 0; j < p_k; ++j) {
                grad[bstart + j] += resid[k] * row[j];
            }
        }

        // Accumulate extra-grad contributions
        for (int e = 0; e < n_extra_grads; ++e) extra_acc[e] += extra[e];
    }

    // Write extra-grad sums into the parameter vector
    for (int e = 0; e < n_extra_grads; ++e) {
        grad[layout.extra_offset + e] += extra_acc[e];
    }
    return log_lik;
}

// ----- Helper: log Gamma(shape, rate) prior on log_phi (Jacobian-adjusted)
inline double log_gamma_prior_log(double log_phi, double shape, double rate) {
    const double phi = std::exp(log_phi);
    return (shape - 1.0) * log_phi - rate * phi + log_phi;
}
inline double dlog_gamma_prior_log(double log_phi, double shape, double rate) {
    // d/dlog_phi  = (shape-1) - rate*phi + 1 = shape - rate*phi.
    return shape - rate * std::exp(log_phi);
}

// ----- Helper: half-Cauchy(0, scale) prior on log_sigma (Jacobian-adjusted)
inline double log_half_cauchy_log(double log_sigma, double scale) {
    const double s = std::exp(log_sigma);
    const double r = s / scale;
    return -std::log(1.0 + r * r) + log_sigma;
}
inline double dlog_half_cauchy_log(double log_sigma, double scale) {
    const double s = std::exp(log_sigma);
    const double r = s / scale;
    const double r2 = r * r;
    return -2.0 * r2 / (1.0 + r2) + 1.0;
}

// =====================================================================
// Per-family functors. Each one captures the response payload, the
// shared cfg knobs, and any decoded extra parameters; operator()(i, eta,
// resid, extra) returns log L_i and writes per-obs derivatives.
// =====================================================================

struct BinomialFn {
    const lik::BinomialResponseData* resp;
    int N;
    double operator()(int i, const double* eta, double* resid, double* /*extra*/) const {
        const double eta_i = eta[i];   // np = 1, so eta[i] is the only one
        const int y = resp->y[i];
        const int n = resp->n[i];
        const double p = 1.0 / (1.0 + std::exp(-eta_i));
        resid[0] = static_cast<double>(y) - n * p;

        // Stable log-pmf
        double log_p, log_1mp;
        if (eta_i >= 0.0) {
            const double e = std::exp(-eta_i);
            log_p   = -std::log1p(e);
            log_1mp = -eta_i - std::log1p(e);
        } else {
            const double e = std::exp(eta_i);
            log_p   = eta_i - std::log1p(e);
            log_1mp = -std::log1p(e);
        }
        return y * log_p + (n - y) * log_1mp + tulpa::math::portable_lchoose(n, y);
    }
};

struct PoissonGammaFn {
    const lik::PoissonGammaResponseData* resp;
    double shape;       // == exp(log_shape)
    double log_shape;
    int N;
    // Cached digamma(shape) — independent of i.
    double dig_shape;
    double operator()(int i, const double* eta, double* resid, double* extra) const {
        const double eta_n = eta[i];          // process 0
        const double eta_d = eta[N + i];      // process 1
        const int    y_n   = resp->y_num[i];
        const double y_d   = resp->y_denom_cont[i];
        const double mu_n  = std::exp(eta_n);
        const double mu_d  = std::exp(eta_d);

        // Numerator: Poisson
        resid[0] = static_cast<double>(y_n) - mu_n;

        // Denominator: Gamma(shape, rate=shape/mu_d)
        // Skip if y_d <= 0 (matches legacy behaviour; gamma support).
        double ll_g = 0.0;
        double extra_i = 0.0;
        if (y_d > 0.0) {
            resid[1] = shape * (y_d / mu_d - 1.0);
            const double rate = shape / mu_d;
            // ll_gamma = shape*log(rate) + (shape-1)*log(y) - rate*y - lgamma(shape)
            const double log_y = std::log(y_d);
            ll_g = shape * std::log(rate) + (shape - 1.0) * log_y - rate * y_d
                   - std::lgamma(shape);
            // d/dshape per obs (chain-rule to d/dlog_shape applied outside):
            //   d/dshape = log(rate) + 1 + log(y) - digamma(shape) - rate*y/shape
            extra_i = std::log(rate) + 1.0 + log_y - dig_shape - rate * y_d / shape;
        } else {
            resid[1] = 0.0;
        }
        extra[0] += extra_i;

        const double ll_p = static_cast<double>(y_n) * eta_n - mu_n
                            - std::lgamma(static_cast<double>(y_n) + 1.0);
        return ll_p + ll_g;
    }
};

struct NegbinGammaFn {
    const lik::NegbinGammaResponseData* resp;
    double phi_num;       // == exp(log_phi_num)
    double phi_denom;     // == exp(log_phi_denom)   == "shape"
    int N;
    double dig_phi_num;
    double dig_phi_denom;
    double operator()(int i, const double* eta, double* resid, double* extra) const {
        const double eta_n = eta[i];
        const double eta_d = eta[N + i];
        const int    y_n   = resp->y_num[i];
        const double y_d   = resp->y_denom_cont[i];
        const double mu_n  = std::exp(eta_n);
        const double mu_d  = std::exp(eta_d);

        // Numerator: NegBin(mu_num, phi_num)
        const double denom_n = mu_n + phi_num;
        resid[0] = static_cast<double>(y_n)
                   - mu_n * (static_cast<double>(y_n) + phi_num) / denom_n;
        const double dlog_phi_num_i = tulpa::math::portable_digamma(y_n + phi_num)
                                       - dig_phi_num
                                       + std::log(phi_num / denom_n)
                                       + (mu_n - static_cast<double>(y_n)) / denom_n;

        // Denominator: Gamma(shape=phi_denom, rate=phi_denom/mu_d). Only if y_d > 0.
        double dlog_phi_denom_i = 0.0;
        double ll_g = 0.0;
        if (y_d > 0.0) {
            resid[1] = phi_denom * (y_d / mu_d - 1.0);
            const double rate = phi_denom / mu_d;
            const double log_y = std::log(y_d);
            ll_g = phi_denom * std::log(rate)
                   + (phi_denom - 1.0) * log_y - rate * y_d
                   - std::lgamma(phi_denom);
            dlog_phi_denom_i = std::log(rate) + 1.0 + log_y - dig_phi_denom
                               - rate * y_d / phi_denom;
        } else {
            resid[1] = 0.0;
        }
        // extra grads come back as d/dlog_phi (= phi * d/dphi). The chain rule
        // is applied at the scaffold level via multiplying by phi after the
        // sum — so accumulate the d/dphi pieces here, multiply once outside.
        extra[0] += dlog_phi_num_i;
        extra[1] += dlog_phi_denom_i;

        // Log-likelihood: NegBin numerator + Gamma denominator
        const double ll_n = lik::negbin_log_pmf<double>(y_n, mu_n, phi_num);
        return ll_n + ll_g;
    }
};

struct NegbinNegbinFn {
    const lik::NegbinNegbinResponseData* resp;
    double phi_num, phi_denom;
    int N;
    double dig_phi_num, dig_phi_denom;
    double operator()(int i, const double* eta, double* resid, double* extra) const {
        const double eta_n = eta[i];
        const double eta_d = eta[N + i];
        const int    y_n   = resp->y_num[i];
        const int    y_d   = resp->y_denom[i];
        const double mu_n  = std::exp(eta_n);
        const double mu_d  = std::exp(eta_d);

        const double denom_n = mu_n + phi_num;
        const double denom_d = mu_d + phi_denom;
        resid[0] = static_cast<double>(y_n)
                   - mu_n * (static_cast<double>(y_n) + phi_num) / denom_n;
        resid[1] = static_cast<double>(y_d)
                   - mu_d * (static_cast<double>(y_d) + phi_denom) / denom_d;

        extra[0] += tulpa::math::portable_digamma(y_n + phi_num) - dig_phi_num
                    + std::log(phi_num / denom_n)
                    + (mu_n - static_cast<double>(y_n)) / denom_n;
        extra[1] += tulpa::math::portable_digamma(y_d + phi_denom) - dig_phi_denom
                    + std::log(phi_denom / denom_d)
                    + (mu_d - static_cast<double>(y_d)) / denom_d;

        const double ll_n = lik::negbin_log_pmf<double>(y_n, mu_n, phi_num);
        const double ll_d = lik::negbin_log_pmf<double>(y_d, mu_d, phi_denom);
        return ll_n + ll_d;
    }
};

struct GammaGammaFn {
    const lik::GammaGammaResponseData* resp;
    double phi_num, phi_denom;
    int N;
    double dig_phi_num, dig_phi_denom;
    double operator()(int i, const double* eta, double* resid, double* extra) const {
        const double eta_n = eta[i];
        const double eta_d = eta[N + i];
        const double y_n   = resp->y_num_cont[i];
        const double y_d   = resp->y_denom_cont[i];
        const double mu_n  = std::exp(eta_n);
        const double mu_d  = std::exp(eta_d);

        double ll = 0.0;
        if (y_n > 0.0) {
            resid[0] = phi_num * (y_n / mu_n - 1.0);
            const double rate_n = phi_num / mu_n;
            const double log_yn = std::log(y_n);
            ll += phi_num * std::log(rate_n)
                  + (phi_num - 1.0) * log_yn - rate_n * y_n
                  - std::lgamma(phi_num);
            extra[0] += std::log(rate_n) + 1.0 + log_yn - dig_phi_num - y_n / mu_n;
        } else {
            resid[0] = 0.0;
        }
        if (y_d > 0.0) {
            resid[1] = phi_denom * (y_d / mu_d - 1.0);
            const double rate_d = phi_denom / mu_d;
            const double log_yd = std::log(y_d);
            ll += phi_denom * std::log(rate_d)
                  + (phi_denom - 1.0) * log_yd - rate_d * y_d
                  - std::lgamma(phi_denom);
            extra[1] += std::log(rate_d) + 1.0 + log_yd - dig_phi_denom - y_d / mu_d;
        } else {
            resid[1] = 0.0;
        }
        return ll;
    }
};

struct LognormalFn {
    const lik::LognormalResponseData* resp;
    double sigma_num, sigma_denom;
    int N;
    double operator()(int i, const double* eta, double* resid, double* extra) const {
        const double mu_n = eta[i];           // eta IS mu_log
        const double mu_d = eta[N + i];
        const double y_n  = resp->y_num_cont[i];
        const double y_d  = resp->y_denom_cont[i];
        const double log_yn = std::log(y_n);
        const double log_yd = std::log(y_d);
        const double s_n  = sigma_num;
        const double s_d  = sigma_denom;
        const double s_n2 = s_n * s_n;
        const double s_d2 = s_d * s_d;
        const double z_n  = (log_yn - mu_n) / s_n;
        const double z_d  = (log_yd - mu_d) / s_d;

        resid[0] = (log_yn - mu_n) / s_n2;
        resid[1] = (log_yd - mu_d) / s_d2;

        // d/dsigma_k = (-1 + z_k^2) / sigma_k
        // d/dlog_sigma_k = sigma_k * d/dsigma_k = -1 + z_k^2
        extra[0] += -1.0 + z_n * z_n;
        extra[1] += -1.0 + z_d * z_d;

        // Log-likelihood (matches legacy: drops -0.5*log(2*pi))
        return -log_yn - std::log(s_n) - 0.5 * z_n * z_n
             + -log_yd - std::log(s_d) - 0.5 * z_d * z_d;
    }
};

struct BetaBinomialFn {
    const lik::BetaBinomialResponseData* resp;
    double phi;            // == exp(log_phi)
    int N;
    double dig_phi;
    double operator()(int i, const double* eta, double* resid, double* extra) const {
        const int y = resp->y[i];
        const int n = resp->n[i];
        const ratiod::math::BetaBinomialObs bb =
            ratiod::math::beta_binomial_obs(y, n, eta[i], phi);
        resid[0] = bb.d_eta;
        // d/dphi (will be multiplied by phi at the end of the loop)
        extra[0] += bb.d_phi + dig_phi;
        return bb.ll + std::lgamma(phi) + tulpa::math::portable_lchoose(n, y);
    }
};

}  // namespace

// =====================================================================
// Public per-family entry points
// =====================================================================

void grad_h_binomial(
    const std::vector<double>& params,
    const tulpa::ModelData& data,
    const tulpa::ParamLayout& layout,
    std::vector<double>& grad,
    double* log_post_out
) {
    assert_no_latent_structure(layout);
    const int n = static_cast<int>(params.size());
    grad.assign(n, 0.0);
    if (log_post_out) *log_post_out = 0.0;

    apply_beta_prior(params, data, layout, grad, log_post_out);

    const std::vector<double> eta = compute_eta_matrix(params, data, layout);
    const auto* resp = static_cast<const lik::BinomialResponseData*>(
        data.model_response_data);
    BinomialFn fn{resp, data.N};

    const double log_lik = run_obs_loop(params, data, layout, eta.data(), fn,
                                        grad, /*n_extra_grads=*/0);
    if (log_post_out) *log_post_out += log_lik;
}

void grad_h_poisson_gamma(
    const std::vector<double>& params,
    const tulpa::ModelData& data,
    const tulpa::ParamLayout& layout,
    std::vector<double>& grad,
    double* log_post_out
) {
    assert_no_latent_structure(layout);
    const int n = static_cast<int>(params.size());
    grad.assign(n, 0.0);
    if (log_post_out) *log_post_out = 0.0;

    apply_beta_prior(params, data, layout, grad, log_post_out);

    const double log_shape = params[layout.extra_offset];
    const double shape     = std::exp(log_shape);
    const double dig_shape = tulpa::math::portable_digamma(shape);

    const std::vector<double> eta = compute_eta_matrix(params, data, layout);
    const auto* resp = static_cast<const lik::PoissonGammaResponseData*>(
        data.model_response_data);
    PoissonGammaFn fn{resp, shape, log_shape, data.N, dig_shape};

    const double log_lik = run_obs_loop(params, data, layout, eta.data(), fn,
                                        grad, /*n_extra_grads=*/1);

    // Convert d/dshape -> d/dlog_shape via chain rule (* shape).
    grad[layout.extra_offset] *= shape;

    // Extra-parameter prior on log_shape: Gamma(phi_prior_shape, phi_prior_rate)
    grad[layout.extra_offset] += dlog_gamma_prior_log(
        log_shape, data.phi_prior_shape, data.phi_prior_rate);
    if (log_post_out) {
        *log_post_out += log_lik
            + log_gamma_prior_log(log_shape, data.phi_prior_shape, data.phi_prior_rate);
    }
}

void grad_h_negbin_gamma(
    const std::vector<double>& params,
    const tulpa::ModelData& data,
    const tulpa::ParamLayout& layout,
    std::vector<double>& grad,
    double* log_post_out
) {
    assert_no_latent_structure(layout);
    const int n = static_cast<int>(params.size());
    grad.assign(n, 0.0);
    if (log_post_out) *log_post_out = 0.0;

    apply_beta_prior(params, data, layout, grad, log_post_out);

    const double log_phi_num   = params[layout.extra_offset + 0];
    const double log_phi_denom = params[layout.extra_offset + 1];
    const double phi_num       = std::exp(log_phi_num);
    const double phi_denom     = std::exp(log_phi_denom);
    const double dig_phi_num   = tulpa::math::portable_digamma(phi_num);
    const double dig_phi_denom = tulpa::math::portable_digamma(phi_denom);

    const std::vector<double> eta = compute_eta_matrix(params, data, layout);
    const auto* resp = static_cast<const lik::NegbinGammaResponseData*>(
        data.model_response_data);
    NegbinGammaFn fn{resp, phi_num, phi_denom, data.N, dig_phi_num, dig_phi_denom};

    const double log_lik = run_obs_loop(params, data, layout, eta.data(), fn,
                                        grad, /*n_extra_grads=*/2);

    // The functor wrote d/dphi; convert to d/dlog_phi via chain rule (* phi)
    grad[layout.extra_offset + 0] *= phi_num;
    grad[layout.extra_offset + 1] *= phi_denom;

    // Extra-parameter priors on log_phi_num, log_phi_denom
    grad[layout.extra_offset + 0] += dlog_gamma_prior_log(
        log_phi_num,   data.phi_prior_shape, data.phi_prior_rate);
    grad[layout.extra_offset + 1] += dlog_gamma_prior_log(
        log_phi_denom, data.phi_prior_shape, data.phi_prior_rate);

    if (log_post_out) {
        *log_post_out += log_lik
            + log_gamma_prior_log(log_phi_num,   data.phi_prior_shape, data.phi_prior_rate)
            + log_gamma_prior_log(log_phi_denom, data.phi_prior_shape, data.phi_prior_rate);
    }
}

void grad_h_negbin_negbin(
    const std::vector<double>& params,
    const tulpa::ModelData& data,
    const tulpa::ParamLayout& layout,
    std::vector<double>& grad,
    double* log_post_out
) {
    assert_no_latent_structure(layout);
    const int n = static_cast<int>(params.size());
    grad.assign(n, 0.0);
    if (log_post_out) *log_post_out = 0.0;

    apply_beta_prior(params, data, layout, grad, log_post_out);

    const double log_phi_num   = params[layout.extra_offset + 0];
    const double log_phi_denom = params[layout.extra_offset + 1];
    const double phi_num       = std::exp(log_phi_num);
    const double phi_denom     = std::exp(log_phi_denom);
    const double dig_phi_num   = tulpa::math::portable_digamma(phi_num);
    const double dig_phi_denom = tulpa::math::portable_digamma(phi_denom);

    const std::vector<double> eta = compute_eta_matrix(params, data, layout);
    const auto* resp = static_cast<const lik::NegbinNegbinResponseData*>(
        data.model_response_data);
    NegbinNegbinFn fn{resp, phi_num, phi_denom, data.N, dig_phi_num, dig_phi_denom};

    const double log_lik = run_obs_loop(params, data, layout, eta.data(), fn,
                                        grad, /*n_extra_grads=*/2);

    grad[layout.extra_offset + 0] *= phi_num;
    grad[layout.extra_offset + 1] *= phi_denom;

    grad[layout.extra_offset + 0] += dlog_gamma_prior_log(
        log_phi_num,   data.phi_prior_shape, data.phi_prior_rate);
    grad[layout.extra_offset + 1] += dlog_gamma_prior_log(
        log_phi_denom, data.phi_prior_shape, data.phi_prior_rate);

    if (log_post_out) {
        *log_post_out += log_lik
            + log_gamma_prior_log(log_phi_num,   data.phi_prior_shape, data.phi_prior_rate)
            + log_gamma_prior_log(log_phi_denom, data.phi_prior_shape, data.phi_prior_rate);
    }
}

void grad_h_gamma_gamma(
    const std::vector<double>& params,
    const tulpa::ModelData& data,
    const tulpa::ParamLayout& layout,
    std::vector<double>& grad,
    double* log_post_out
) {
    assert_no_latent_structure(layout);
    const int n = static_cast<int>(params.size());
    grad.assign(n, 0.0);
    if (log_post_out) *log_post_out = 0.0;

    apply_beta_prior(params, data, layout, grad, log_post_out);

    const double log_phi_num   = params[layout.extra_offset + 0];
    const double log_phi_denom = params[layout.extra_offset + 1];
    const double phi_num       = std::exp(log_phi_num);
    const double phi_denom     = std::exp(log_phi_denom);
    const double dig_phi_num   = tulpa::math::portable_digamma(phi_num);
    const double dig_phi_denom = tulpa::math::portable_digamma(phi_denom);

    const std::vector<double> eta = compute_eta_matrix(params, data, layout);
    const auto* resp = static_cast<const lik::GammaGammaResponseData*>(
        data.model_response_data);
    GammaGammaFn fn{resp, phi_num, phi_denom, data.N, dig_phi_num, dig_phi_denom};

    const double log_lik = run_obs_loop(params, data, layout, eta.data(), fn,
                                        grad, /*n_extra_grads=*/2);

    grad[layout.extra_offset + 0] *= phi_num;
    grad[layout.extra_offset + 1] *= phi_denom;

    grad[layout.extra_offset + 0] += dlog_gamma_prior_log(
        log_phi_num,   data.phi_prior_shape, data.phi_prior_rate);
    grad[layout.extra_offset + 1] += dlog_gamma_prior_log(
        log_phi_denom, data.phi_prior_shape, data.phi_prior_rate);

    if (log_post_out) {
        *log_post_out += log_lik
            + log_gamma_prior_log(log_phi_num,   data.phi_prior_shape, data.phi_prior_rate)
            + log_gamma_prior_log(log_phi_denom, data.phi_prior_shape, data.phi_prior_rate);
    }
}

void grad_h_lognormal(
    const std::vector<double>& params,
    const tulpa::ModelData& data,
    const tulpa::ParamLayout& layout,
    std::vector<double>& grad,
    double* log_post_out
) {
    assert_no_latent_structure(layout);
    const int n = static_cast<int>(params.size());
    grad.assign(n, 0.0);
    if (log_post_out) *log_post_out = 0.0;

    apply_beta_prior(params, data, layout, grad, log_post_out);

    const double log_sig_num   = params[layout.extra_offset + 0];
    const double log_sig_denom = params[layout.extra_offset + 1];
    const double sig_num       = std::exp(log_sig_num);
    const double sig_denom     = std::exp(log_sig_denom);

    const std::vector<double> eta = compute_eta_matrix(params, data, layout);
    const auto* resp = static_cast<const lik::LognormalResponseData*>(
        data.model_response_data);
    LognormalFn fn{resp, sig_num, sig_denom, data.N};

    const double log_lik = run_obs_loop(params, data, layout, eta.data(), fn,
                                        grad, /*n_extra_grads=*/2);
    // Functor writes d/dlog_sigma directly (= -1 + z^2), so no chain-rule pass.

    // Half-Cauchy(0, sigma_re_scale) prior on each log_sigma. tulpaRatio's
    // R-side bridge sets `data.sigma_re_scale = sigma_prior_scale` from the
    // RatioConfig (mirrors legacy backend). We mirror that contract here.
    grad[layout.extra_offset + 0] += dlog_half_cauchy_log(
        log_sig_num,   data.sigma_re_scale);
    grad[layout.extra_offset + 1] += dlog_half_cauchy_log(
        log_sig_denom, data.sigma_re_scale);

    if (log_post_out) {
        *log_post_out += log_lik
            + log_half_cauchy_log(log_sig_num,   data.sigma_re_scale)
            + log_half_cauchy_log(log_sig_denom, data.sigma_re_scale);
    }
}

void grad_h_beta_binomial(
    const std::vector<double>& params,
    const tulpa::ModelData& data,
    const tulpa::ParamLayout& layout,
    std::vector<double>& grad,
    double* log_post_out
) {
    assert_no_latent_structure(layout);
    const int n = static_cast<int>(params.size());
    grad.assign(n, 0.0);
    if (log_post_out) *log_post_out = 0.0;

    apply_beta_prior(params, data, layout, grad, log_post_out);

    const double log_phi = params[layout.extra_offset];
    const double phi     = std::exp(log_phi);
    const double dig_phi = tulpa::math::portable_digamma(phi);

    const std::vector<double> eta = compute_eta_matrix(params, data, layout);
    const auto* resp = static_cast<const lik::BetaBinomialResponseData*>(
        data.model_response_data);
    BetaBinomialFn fn{resp, phi, data.N, dig_phi};

    const double log_lik = run_obs_loop(params, data, layout, eta.data(), fn,
                                        grad, /*n_extra_grads=*/1);

    // Convert d/dphi -> d/dlog_phi via chain rule
    grad[layout.extra_offset] *= phi;

    grad[layout.extra_offset] += dlog_gamma_prior_log(
        log_phi, data.phi_prior_shape, data.phi_prior_rate);

    if (log_post_out) {
        *log_post_out += log_lik
            + log_gamma_prior_log(log_phi, data.phi_prior_shape, data.phi_prior_rate);
    }
}

}  // namespace tulpaRatio
