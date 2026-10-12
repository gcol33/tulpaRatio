// gibbs_spatial.h
// Component-wise Gibbs sampler for ICAR/BYM2 spatial models
// Designed for large S where HMC struggles with dimensionality
//
// Update scheme:
//   1. phi_s | phi_{-s}, beta, tau, y  (univariate MH per site)
//   2. tau | phi                        (conjugate Gamma for ICAR)
//   3. beta | phi, y                    (block MH with Gaussian proposal)
//   4. dispersion | rest                (univariate MH on log scale)
//   For BYM2: sigma, rho, phi, theta all updated

#pragma once

#include <Rcpp.h>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>

namespace gibbs {

// =========================================================================
// Site-level log-likelihood (sum over observations at a site)
// =========================================================================

enum class GibbsFamily { POISSON_GAMMA, NEGBIN_NEGBIN, BINOMIAL, NEGBIN_GAMMA };

struct GibbsData {
    int N;                              // Total observations
    int S;                              // Number of spatial units
    int p_num;                          // Numerator predictors
    int p_denom;                        // Denominator predictors
    GibbsFamily family;

    const double* X_num;                // N x p_num (row-major)
    const double* X_denom;              // N x p_denom (row-major)
    const int* y_num;                   // Integer numerator response
    const int* y_denom;                 // Integer denominator response (NB, Binomial)
    const double* y_denom_cont;         // Continuous denominator (PG, NB-Gamma)

    const int* spatial_group;           // Maps obs -> site (0-based)

    // Column holding the intercept in each design matrix, or -1 when the
    // formula has none. The spatial field's level is carried here.
    int icept_num = -1;
    int icept_denom = -1;

    // Priors, matching what compute_log_post() applies to the same model so
    // that a fit does not depend on which backend the router picks.
    double sigma_beta = 10.0;         // beta ~ N(0, sigma_beta^2)
    double sigma_re_scale = 2.5;      // BYM2 sigma_total ~ Half-Cauchy(0, .)
    double tau_spatial_shape = 1.0;   // ICAR tau ~ Gamma(shape, rate)
    double tau_spatial_rate = 0.01;

    // CSR obs-to-site mapping (built once)
    std::vector<int> site_obs_ptr;      // site_obs_ptr[s] = start index
    std::vector<int> site_obs_idx;      // observation indices for each site

    // Adjacency (CSR format)
    const int* adj_row_ptr;
    const int* adj_col_idx;
    const int* n_neighbors;

    // BYM2
    bool is_bym2;
    double bym2_scale;

    // TVC (Temporally-Varying Coefficients)
    bool has_tvc = false;
    int tvc_n_times = 0;
    int tvc_n_groups = 1;
    int tvc_n_terms = 0;
    std::vector<int> tvc_time_index;    // obs -> time (1-based)
    std::vector<int> tvc_group_index;   // obs -> group (1-based)
    std::vector<double> tvc_X;          // n_obs x n_terms (row-major)
    bool tvc_shared = true;
    int tvc_structure = 0;              // 0=RW1, 1=RW2, 2=AR1

    // Time-indexed obs mapping for TVC updates (built once)
    // time_obs_ptr[t] = start index in time_obs_idx for time t
    std::vector<int> time_obs_ptr;
    std::vector<int> time_obs_idx;
};

// Build CSR site->obs mapping
inline void build_site_obs_map(GibbsData& d) {
    d.site_obs_ptr.assign(d.S + 1, 0);
    for (int i = 0; i < d.N; i++) {
        d.site_obs_ptr[d.spatial_group[i] + 1]++;
    }
    for (int s = 0; s < d.S; s++) {
        d.site_obs_ptr[s + 1] += d.site_obs_ptr[s];
    }
    d.site_obs_idx.resize(d.N);
    std::vector<int> pos(d.site_obs_ptr.begin(), d.site_obs_ptr.end());
    for (int i = 0; i < d.N; i++) {
        int s = d.spatial_group[i];
        d.site_obs_idx[pos[s]++] = i;
    }
}

// Column of a row-major design matrix that is all ones, or -1 if there is none.
inline int find_intercept_col(const double* X, int N, int p) {
    if (X == nullptr || p <= 0 || N <= 0) return -1;
    for (int j = 0; j < p; j++) {
        bool all_ones = true;
        for (int i = 0; i < N; i++) {
            if (X[i * p + j] != 1.0) { all_ones = false; break; }
        }
        if (all_ones) return j;
    }
    return -1;
}

inline void build_intercept_index(GibbsData& d) {
    d.icept_num = find_intercept_col(d.X_num, d.N, d.p_num);
    d.icept_denom = (d.family == GibbsFamily::BINOMIAL)
                    ? -1
                    : find_intercept_col(d.X_denom, d.N, d.p_denom);
}

// The spatial effect enters eta_num always, and eta_denom for every family
// whose denominator is modelled. Holding the field at sum zero shifts each of
// those linear predictors by the same amount, so it is available only when
// every one of them has an intercept to take that shift on.
inline bool can_constrain_field(const GibbsData& d) {
    if (d.icept_num < 0) return false;
    if (d.family == GibbsFamily::BINOMIAL) return true;
    return d.icept_denom >= 0;
}

// Change in log prior density from moving every intercept by `shift`.
inline double intercept_shift_log_prior_ratio(const double* beta_num,
                                              const double* beta_denom,
                                              const GibbsData& d,
                                              double shift) {
    const double prec = 1.0 / (d.sigma_beta * d.sigma_beta);
    double lp = -0.5 * prec * (shift * shift
                               + 2.0 * beta_num[d.icept_num] * shift);
    if (d.icept_denom >= 0) {
        lp -= 0.5 * prec * (shift * shift
                            + 2.0 * beta_denom[d.icept_denom] * shift);
    }
    return lp;
}

// Build CSR time->obs mapping for TVC updates
inline void build_time_obs_map(GibbsData& d) {
    if (!d.has_tvc || d.tvc_n_times == 0) return;
    d.time_obs_ptr.assign(d.tvc_n_times + 1, 0);
    for (int i = 0; i < d.N; i++) {
        int t = d.tvc_time_index[i] - 1;  // 0-based
        if (t >= 0 && t < d.tvc_n_times)
            d.time_obs_ptr[t + 1]++;
    }
    for (int t = 0; t < d.tvc_n_times; t++)
        d.time_obs_ptr[t + 1] += d.time_obs_ptr[t];
    d.time_obs_idx.resize(d.N);
    std::vector<int> pos(d.time_obs_ptr.begin(), d.time_obs_ptr.end());
    for (int i = 0; i < d.N; i++) {
        int t = d.tvc_time_index[i] - 1;
        if (t >= 0 && t < d.tvc_n_times)
            d.time_obs_idx[pos[t]++] = i;
    }
}

// Compute TVC contribution to eta for observation i
inline double compute_tvc_eta_obs(int i, const double* tvc_w, const GibbsData& d) {
    if (!d.has_tvc) return 0.0;
    int t = d.tvc_time_index[i] - 1;
    int g = d.tvc_group_index[i] - 1;
    double eta = 0.0;
    for (int j = 0; j < d.tvc_n_terms; j++) {
        int w_idx = (g * d.tvc_n_terms + j) * d.tvc_n_times + t;
        eta += d.tvc_X[i * d.tvc_n_terms + j] * tvc_w[w_idx];
    }
    return eta;
}

// Log-likelihood for a single observation given eta_num, eta_denom
inline double obs_log_lik(int i, double eta_num, double eta_denom,
                          double phi_num, double phi_denom,
                          const GibbsData& d) {
    double ll = 0.0;

    switch (d.family) {
    case GibbsFamily::POISSON_GAMMA: {
        double mu = std::exp(std::min(eta_num, 20.0));
        ll += d.y_num[i] * eta_num - mu;                // Poisson (drop y! constant)
        double alpha = phi_denom;
        double mu_d = std::exp(std::min(eta_denom, 20.0));
        double y_d = d.y_denom_cont[i];
        ll += alpha * std::log(alpha) - std::lgamma(alpha)
            + (alpha - 1.0) * std::log(y_d) - alpha * (y_d / mu_d + eta_denom);
        // Gamma: alpha*log(alpha/mu) + (alpha-1)*log(y) - alpha*y/mu - lgamma(alpha)
        // = alpha*log(alpha) - alpha*log(mu) + (alpha-1)*log(y) - alpha*y/mu - lgamma(alpha)
        break;
    }
    case GibbsFamily::NEGBIN_NEGBIN: {
        double mu_n = std::exp(std::min(eta_num, 20.0));
        double r_n = phi_num;
        int y_n = d.y_num[i];
        ll += std::lgamma(y_n + r_n) - std::lgamma(r_n) - std::lgamma(y_n + 1.0)
            + r_n * std::log(r_n / (mu_n + r_n))
            + y_n * std::log(mu_n / (mu_n + r_n));

        double mu_d = std::exp(std::min(eta_denom, 20.0));
        double r_d = phi_denom;
        int y_d = d.y_denom[i];
        ll += std::lgamma(y_d + r_d) - std::lgamma(r_d) - std::lgamma(y_d + 1.0)
            + r_d * std::log(r_d / (mu_d + r_d))
            + y_d * std::log(mu_d / (mu_d + r_d));
        break;
    }
    case GibbsFamily::BINOMIAL: {
        double p = 1.0 / (1.0 + std::exp(-eta_num));
        int y = d.y_num[i];
        int n_trials = d.y_denom[i];
        ll += y * std::log(p + 1e-300) + (n_trials - y) * std::log(1.0 - p + 1e-300);
        break;
    }
    case GibbsFamily::NEGBIN_GAMMA: {
        double mu_n = std::exp(std::min(eta_num, 20.0));
        double r_n = phi_num;
        int y_n = d.y_num[i];
        ll += std::lgamma(y_n + r_n) - std::lgamma(r_n) - std::lgamma(y_n + 1.0)
            + r_n * std::log(r_n / (mu_n + r_n))
            + y_n * std::log(mu_n / (mu_n + r_n));

        double alpha = phi_denom;
        double mu_d = std::exp(std::min(eta_denom, 20.0));
        double y_d = d.y_denom_cont[i];
        ll += alpha * std::log(alpha) - std::lgamma(alpha)
            + (alpha - 1.0) * std::log(y_d) - alpha * (y_d / mu_d + eta_denom);
        break;
    }
    }
    return ll;
}

// Sum log-likelihood over all observations at site s, given spatial effect value
inline double site_log_lik(int s, double spatial_effect,
                           const double* beta_num, const double* beta_denom,
                           double phi_num, double phi_denom,
                           const GibbsData& d,
                           const double* tvc_w = nullptr) {
    double ll = 0.0;
    bool is_binomial = (d.family == GibbsFamily::BINOMIAL);

    for (int idx = d.site_obs_ptr[s]; idx < d.site_obs_ptr[s + 1]; idx++) {
        int i = d.site_obs_idx[idx];
        double eta_num = spatial_effect;
        double eta_denom = is_binomial ? 0.0 : spatial_effect;

        for (int p = 0; p < d.p_num; p++)
            eta_num += d.X_num[i * d.p_num + p] * beta_num[p];
        if (!is_binomial) {
            for (int p = 0; p < d.p_denom; p++)
                eta_denom += d.X_denom[i * d.p_denom + p] * beta_denom[p];
        }

        // TVC contribution
        if (tvc_w != nullptr && d.has_tvc) {
            double tvc_eff = compute_tvc_eta_obs(i, tvc_w, d);
            eta_num += tvc_eff;
            if (!is_binomial && d.tvc_shared) eta_denom += tvc_eff;
        }

        ll += obs_log_lik(i, eta_num, eta_denom, phi_num, phi_denom, d);
    }
    return ll;
}

// Full log-likelihood over all observations
inline double full_log_lik(const double* phi, const double* beta_num,
                           const double* beta_denom,
                           double phi_num, double phi_denom,
                           const GibbsData& d,
                           const double* tvc_w = nullptr) {
    double ll = 0.0;
    bool is_binomial = (d.family == GibbsFamily::BINOMIAL);

    for (int i = 0; i < d.N; i++) {
        int s = d.spatial_group[i];
        double eta_num = phi[s];
        double eta_denom = is_binomial ? 0.0 : phi[s];

        for (int p = 0; p < d.p_num; p++)
            eta_num += d.X_num[i * d.p_num + p] * beta_num[p];
        if (!is_binomial) {
            for (int p = 0; p < d.p_denom; p++)
                eta_denom += d.X_denom[i * d.p_denom + p] * beta_denom[p];
        }

        if (tvc_w != nullptr && d.has_tvc) {
            double tvc_eff = compute_tvc_eta_obs(i, tvc_w, d);
            eta_num += tvc_eff;
            if (!is_binomial && d.tvc_shared) eta_denom += tvc_eff;
        }

        ll += obs_log_lik(i, eta_num, eta_denom, phi_num, phi_denom, d);
    }
    return ll;
}

// =========================================================================
// ICAR prior log-density (up to normalization constant)
// =========================================================================

// phi' Q phi for the ICAR precision Q = diag(n_s) - A, in CSR form.
inline double icar_quadratic_form(const double* phi, int S,
                                  const int* adj_row_ptr,
                                  const int* adj_col_idx,
                                  const int* n_neighbors) {
    double quad = 0.0;
    for (int s = 0; s < S; s++) {
        quad += n_neighbors[s] * phi[s] * phi[s];
        for (int k = adj_row_ptr[s]; k < adj_row_ptr[s + 1]; k++) {
            quad -= phi[s] * phi[adj_col_idx[k]];
        }
    }
    return quad;
}

inline double icar_quadratic_form(const double* phi, const GibbsData& d) {
    return icar_quadratic_form(phi, d.S, d.adj_row_ptr, d.adj_col_idx,
                               d.n_neighbors);
}

// ICAR conditional prior for phi_s | phi_{-s}
// Returns (mean, precision) of the full conditional Gaussian
inline std::pair<double, double> icar_conditional(int s, const double* phi,
                                                   double tau,
                                                   const int* adj_row_ptr,
                                                   const int* adj_col_idx,
                                                   const int* n_neighbors) {
    int n_s = n_neighbors[s];
    if (n_s == 0) return {0.0, 0.001};  // Isolated node

    double neighbor_sum = 0.0;
    for (int k = adj_row_ptr[s]; k < adj_row_ptr[s + 1]; k++) {
        neighbor_sum += phi[adj_col_idx[k]];
    }
    double prec = tau * n_s;
    double mean = neighbor_sum / n_s;
    return {mean, prec};
}

// =========================================================================
// Hyperparameter priors, on the transformed scale the sampler moves on
// =========================================================================

// sigma ~ Half-Cauchy(0, scale), evaluated as a density in log(sigma).
inline double half_cauchy_log_dens_log_scale(double sigma, double scale) {
    const double ratio = sigma / scale;
    return -std::log(1.0 + ratio * ratio) + std::log(sigma);
}

// rho ~ Uniform(0, 1), evaluated as a density in logit(rho).
inline double unif01_log_dens_logit_scale(double rho) {
    return std::log(rho) + std::log(1.0 - rho);
}

// beta ~ N(0, sigma_beta^2), summed over a coefficient vector.
inline double beta_log_prior(const double* beta, int p, double sigma_beta) {
    const double prec = 1.0 / (sigma_beta * sigma_beta);
    double lp = 0.0;
    for (int j = 0; j < p; j++) lp -= 0.5 * prec * beta[j] * beta[j];
    return lp;
}

// =========================================================================
// Main Gibbs sampler
// =========================================================================

struct GibbsResult {
    std::vector<double> draws_flat;     // (n_save × n_params) row-major
    std::vector<double> phi_draws_flat; // (n_save × S)
    int n_params = 0;
    int n_save = 0;
    int S = 0;
    std::vector<std::string> param_names;

    // TVC draws
    std::vector<double> tvc_w_draws_flat;   // (n_save × n_tvc_w) row-major
    std::vector<double> tvc_tau_draws_flat;  // (n_save × n_tvc_terms)
    int tvc_n_w = 0;

    // Diagnostics
    std::vector<double> accept_phi;     // Acceptance rate per site
    double accept_beta = 0.0;
    double accept_disp = 0.0;
    double accept_tvc = 0.0;
};

inline GibbsResult run_gibbs_icar(
    const GibbsData& d,
    int n_iter,
    int n_warmup,
    int thin,
    unsigned int seed,
    bool verbose
) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> rnorm(0.0, 1.0);
    std::uniform_real_distribution<double> runif(0.0, 1.0);
    std::gamma_distribution<double> rgamma_unit(1.0, 1.0);

    int S = d.S;
    bool is_binomial = (d.family == GibbsFamily::BINOMIAL);
    bool has_phi_num = (d.family == GibbsFamily::NEGBIN_NEGBIN ||
                        d.family == GibbsFamily::NEGBIN_GAMMA);
    bool has_phi_denom = !is_binomial;
    const bool constrain = can_constrain_field(d);

    // Parameter dimensions
    int p_num = d.p_num;
    int p_denom = is_binomial ? 0 : d.p_denom;
    int n_hyper = 1;  // log_tau
    if (has_phi_num) n_hyper++;   // log_phi_num
    if (has_phi_denom) n_hyper++; // log_phi_denom
    int n_params = p_num + p_denom + n_hyper;

    // Initialize parameters
    std::vector<double> beta_num(p_num, 0.0);
    std::vector<double> beta_denom(p_denom, 0.0);
    std::vector<double> phi(S, 0.0);
    double log_tau = std::log(1.0);
    double log_phi_num = std::log(1.0);   // NB size
    double log_phi_denom = std::log(5.0); // Gamma shape or NB size

    // Initialize beta from data (rough OLS-like)
    if (p_num > 0) beta_num[0] = 1.0;
    if (p_denom > 0) beta_denom[0] = 1.0;

    // Adaptation: proposal scales (tuned during warmup)
    double beta_scale = 0.1;
    double disp_scale = 0.1;
    std::vector<double> phi_scale(S, 0.5);

    // Acceptance tracking
    std::vector<int> phi_accept(S, 0);
    std::vector<int> phi_total(S, 0);
    int beta_accept = 0, beta_total = 0;
    int disp_accept = 0, disp_total = 0;

    // TVC initialization
    int tvc_n_w = d.has_tvc ? (d.tvc_n_groups * d.tvc_n_terms * d.tvc_n_times) : 0;
    std::vector<double> tvc_w(tvc_n_w, 0.0);
    std::vector<double> tvc_tau(d.has_tvc ? d.tvc_n_terms : 0, 1.0);
    std::vector<double> tvc_rho(d.has_tvc ? d.tvc_n_terms : 0, 0.5);
    double tvc_w_scale = 0.2;  // MH proposal scale for TVC coefficients
    int tvc_accept = 0, tvc_total = 0;
    const double* tvc_w_ptr = d.has_tvc ? tvc_w.data() : nullptr;

    // Output storage
    int n_save = (n_iter - n_warmup) / thin;
    GibbsResult result;
    result.n_params = n_params;
    result.n_save = n_save;
    result.S = S;
    result.draws_flat.resize(n_save * n_params);
    result.phi_draws_flat.resize(n_save * S);
    result.accept_phi.resize(S, 0.0);
    result.tvc_n_w = tvc_n_w;
    if (d.has_tvc) {
        result.tvc_w_draws_flat.resize(n_save * tvc_n_w);
        result.tvc_tau_draws_flat.resize(n_save * d.tvc_n_terms);
    }

    int save_idx = 0;

    // ---- Main Gibbs loop ----
    for (int iter = 0; iter < n_iter; iter++) {
        double tau = std::exp(log_tau);
        double phi_num_val = has_phi_num ? std::exp(log_phi_num) : 1.0;
        double phi_denom_val = has_phi_denom ? std::exp(log_phi_denom) : 1.0;

        // ---- 1. Update phi (spatial effects) via MH ----
        // Q1 = 0 leaves the field's level unidentified against the intercept,
        // so the field is held at sum zero and the intercept carries the level
        // alone. Raising site s by delta lowers every site by delta/S and
        // raises the intercept by the same delta/S: eta away from site s is
        // then unchanged, which keeps the update site-local, and the field's
        // sum is unchanged. Q kills the constant part, so the ICAR conditional
        // is still the proposal that cancels the prior, and what is left in the
        // ratio is the likelihood at site s and the intercept's own prior.
        // `phi_off` accumulates the lowering so it costs O(1) per site rather
        // than a pass over the field.
        double phi_off = 0.0;
        for (int s = 0; s < S; s++) {
            auto [cond_mean_raw, cond_prec] = icar_conditional(
                s, phi.data(), tau, d.adj_row_ptr, d.adj_col_idx, d.n_neighbors);

            const double actual_s = phi[s] + phi_off;
            const double phi_prop = cond_mean_raw + phi_off
                                  + rnorm(rng) / std::sqrt(cond_prec + 1e-10);
            const double delta = phi_prop - actual_s;
            const double shift = constrain ? delta / S : 0.0;

            double ll_curr = site_log_lik(s, actual_s, beta_num.data(), beta_denom.data(),
                                          phi_num_val, phi_denom_val, d, tvc_w_ptr);
            double ll_prop = site_log_lik(s, actual_s + delta, beta_num.data(), beta_denom.data(),
                                          phi_num_val, phi_denom_val, d, tvc_w_ptr);

            double log_ratio = ll_prop - ll_curr;
            if (constrain) {
                log_ratio += intercept_shift_log_prior_ratio(
                    beta_num.data(), beta_denom.data(), d, shift);
            }

            if (std::log(runif(rng)) < log_ratio) {
                phi[s] += delta;
                if (constrain) {
                    phi_off -= shift;
                    beta_num[d.icept_num] += shift;
                    if (d.icept_denom >= 0) beta_denom[d.icept_denom] += shift;
                }
                phi_accept[s]++;
            }
            phi_total[s]++;
        }
        if (phi_off != 0.0) {
            for (int s = 0; s < S; s++) phi[s] += phi_off;
        }

        // ---- 2. Update tau (ICAR precision) via conjugate Gamma ----
        {
            // Q is rank S-1 on a connected graph, so the field contributes
            // tau^{(S-1)/2} exp(-tau * quad / 2) and Gamma(a, b) is conjugate.
            const double quad = icar_quadratic_form(phi.data(), d);
            const double shape = d.tau_spatial_shape + 0.5 * (S - 1);
            const double rate = d.tau_spatial_rate + 0.5 * quad;

            if (rate > 0.0) {
                std::gamma_distribution<double> gamma_dist(shape, 1.0 / rate);
                tau = gamma_dist(rng);
                log_tau = std::log(tau);
            }
        }

        // ---- 3. Update beta via block random-walk MH ----
        {
            // Propose: beta* = beta + scale * N(0, I)
            std::vector<double> beta_num_prop(beta_num);
            std::vector<double> beta_denom_prop(beta_denom);

            for (int j = 0; j < p_num; j++)
                beta_num_prop[j] += beta_scale * rnorm(rng);
            for (int j = 0; j < p_denom; j++)
                beta_denom_prop[j] += beta_scale * rnorm(rng);

            double ll_curr = full_log_lik(phi.data(), beta_num.data(), beta_denom.data(),
                                          phi_num_val, phi_denom_val, d, tvc_w_ptr);
            double ll_prop = full_log_lik(phi.data(), beta_num_prop.data(), beta_denom_prop.data(),
                                          phi_num_val, phi_denom_val, d, tvc_w_ptr);

            double lp_curr = beta_log_prior(beta_num.data(), p_num, d.sigma_beta)
                           + beta_log_prior(beta_denom.data(), p_denom, d.sigma_beta);
            double lp_prop = beta_log_prior(beta_num_prop.data(), p_num, d.sigma_beta)
                           + beta_log_prior(beta_denom_prop.data(), p_denom, d.sigma_beta);

            if (std::log(runif(rng)) < (ll_prop + lp_prop) - (ll_curr + lp_curr)) {
                beta_num = beta_num_prop;
                beta_denom = beta_denom_prop;
                beta_accept++;
            }
            beta_total++;
        }

        // ---- 4. Update dispersion params via MH on log scale ----
        if (has_phi_num || has_phi_denom) {
            double log_phi_num_prop = log_phi_num;
            double log_phi_denom_prop = log_phi_denom;

            if (has_phi_num)
                log_phi_num_prop += disp_scale * rnorm(rng);
            if (has_phi_denom)
                log_phi_denom_prop += disp_scale * rnorm(rng);

            double pn_curr = has_phi_num ? std::exp(log_phi_num) : 1.0;
            double pd_curr = has_phi_denom ? std::exp(log_phi_denom) : 1.0;
            double pn_prop = has_phi_num ? std::exp(log_phi_num_prop) : 1.0;
            double pd_prop = has_phi_denom ? std::exp(log_phi_denom_prop) : 1.0;

            double ll_curr = full_log_lik(phi.data(), beta_num.data(), beta_denom.data(),
                                          pn_curr, pd_curr, d, tvc_w_ptr);
            double ll_prop = full_log_lik(phi.data(), beta_num.data(), beta_denom.data(),
                                          pn_prop, pd_prop, d, tvc_w_ptr);

            // PC prior on dispersion: Gamma(2, 0.5) on the parameter
            double lp_curr = 0.0, lp_prop = 0.0;
            if (has_phi_num) {
                lp_curr += std::log(pn_curr) - 0.5 * pn_curr;  // Gamma(2,0.5) + Jacobian
                lp_prop += std::log(pn_prop) - 0.5 * pn_prop;
            }
            if (has_phi_denom) {
                lp_curr += std::log(pd_curr) - 0.5 * pd_curr;
                lp_prop += std::log(pd_prop) - 0.5 * pd_prop;
            }

            if (std::log(runif(rng)) < (ll_prop + lp_prop) - (ll_curr + lp_curr)) {
                log_phi_num = log_phi_num_prop;
                log_phi_denom = log_phi_denom_prop;
                disp_accept++;
            }
            disp_total++;
        }

        // ---- 5. Update TVC coefficients via univariate MH with RW1 conditional proposal ----
        if (d.has_tvc) {
            for (int j = 0; j < d.tvc_n_terms; j++) {
                for (int g = 0; g < d.tvc_n_groups; g++) {
                    for (int t = 0; t < d.tvc_n_times; t++) {
                        int w_idx = (g * d.tvc_n_terms + j) * d.tvc_n_times + t;

                        // RW1 conditional prior: N(neighbor_mean, 1/(tau_j * n_neighbors))
                        int n_nb = 0;
                        double nb_sum = 0.0;
                        if (t > 0) { nb_sum += tvc_w[(g * d.tvc_n_terms + j) * d.tvc_n_times + t - 1]; n_nb++; }
                        if (t < d.tvc_n_times - 1) { nb_sum += tvc_w[(g * d.tvc_n_terms + j) * d.tvc_n_times + t + 1]; n_nb++; }
                        double cond_prec = tvc_tau[j] * std::max(n_nb, 1);
                        double cond_mean = (n_nb > 0) ? nb_sum / n_nb : 0.0;

                        // Propose from RW1 conditional
                        double w_prop = cond_mean + rnorm(rng) / std::sqrt(cond_prec + 1e-10);

                        // Compute likelihood ratio at obs with this time point
                        double ll_diff = 0.0;
                        for (int idx = d.time_obs_ptr[t]; idx < d.time_obs_ptr[t + 1]; idx++) {
                            int i = d.time_obs_idx[idx];
                            int s = d.spatial_group[i];
                            double eta_num_base = phi[s];
                            double eta_denom_base = is_binomial ? 0.0 : phi[s];
                            for (int p = 0; p < d.p_num; p++)
                                eta_num_base += d.X_num[i * d.p_num + p] * beta_num[p];
                            if (!is_binomial)
                                for (int p = 0; p < d.p_denom; p++)
                                    eta_denom_base += d.X_denom[i * d.p_denom + p] * beta_denom[p];

                            // Compute TVC eta with current vs proposed
                            double x_jt = d.tvc_X[i * d.tvc_n_terms + j];
                            double tvc_other = compute_tvc_eta_obs(i, tvc_w.data(), d) - x_jt * tvc_w[w_idx];
                            double tvc_curr_eta = tvc_other + x_jt * tvc_w[w_idx];
                            double tvc_prop_eta = tvc_other + x_jt * w_prop;

                            double en_c = eta_num_base + tvc_curr_eta;
                            double ed_c = eta_denom_base + (d.tvc_shared ? tvc_curr_eta : 0.0);
                            double en_p = eta_num_base + tvc_prop_eta;
                            double ed_p = eta_denom_base + (d.tvc_shared ? tvc_prop_eta : 0.0);

                            ll_diff += obs_log_lik(i, en_p, ed_p, phi_num_val, phi_denom_val, d)
                                     - obs_log_lik(i, en_c, ed_c, phi_num_val, phi_denom_val, d);
                        }

                        if (std::log(runif(rng)) < ll_diff) {
                            tvc_w[w_idx] = w_prop;
                            tvc_accept++;
                        }
                        tvc_total++;
                    }
                }

                // Update tau[j] via conjugate Gamma (RW1)
                double qf = 0.0;
                for (int gg = 0; gg < d.tvc_n_groups; gg++) {
                    for (int tt = 1; tt < d.tvc_n_times; tt++) {
                        double diff = tvc_w[(gg * d.tvc_n_terms + j) * d.tvc_n_times + tt]
                                    - tvc_w[(gg * d.tvc_n_terms + j) * d.tvc_n_times + tt - 1];
                        qf += diff * diff;
                    }
                }
                int rank = (d.tvc_n_times - 1) * d.tvc_n_groups;
                double shape_t = 0.5 * rank;
                double rate_t = 0.5 * qf + 0.01;
                std::gamma_distribution<double> gd_t(shape_t, 1.0 / rate_t);
                tvc_tau[j] = gd_t(rng);
            }
        }

        // ---- Adaptation during warmup ----
        if (iter < n_warmup && iter > 0 && iter % 50 == 0) {
            double target_rate = 0.44;  // Optimal for univariate

            // Adapt beta scale
            double beta_rate = (double)beta_accept / beta_total;
            if (beta_rate > target_rate + 0.05) beta_scale *= 1.2;
            else if (beta_rate < target_rate - 0.05) beta_scale *= 0.8;
            beta_scale = std::max(0.001, std::min(beta_scale, 5.0));
            beta_accept = 0; beta_total = 0;

            // Adapt dispersion scale
            if (has_phi_num || has_phi_denom) {
                double disp_rate = (double)disp_accept / disp_total;
                if (disp_rate > target_rate + 0.05) disp_scale *= 1.2;
                else if (disp_rate < target_rate - 0.05) disp_scale *= 0.8;
                disp_scale = std::max(0.001, std::min(disp_scale, 5.0));
                disp_accept = 0; disp_total = 0;
            }
        }

        // ---- Store draws after warmup ----
        if (iter >= n_warmup && (iter - n_warmup) % thin == 0) {
            int row = save_idx * n_params;
            int col = 0;
            for (int j = 0; j < p_num; j++) result.draws_flat[row + col++] = beta_num[j];
            for (int j = 0; j < p_denom; j++) result.draws_flat[row + col++] = beta_denom[j];
            if (has_phi_num) result.draws_flat[row + col++] = log_phi_num;
            if (has_phi_denom) result.draws_flat[row + col++] = log_phi_denom;
            result.draws_flat[row + col++] = log_tau;

            // Store phi
            for (int s = 0; s < S; s++)
                result.phi_draws_flat[save_idx * S + s] = phi[s];

            // Store TVC draws
            if (d.has_tvc) {
                for (int k = 0; k < tvc_n_w; k++)
                    result.tvc_w_draws_flat[save_idx * tvc_n_w + k] = tvc_w[k];
                for (int j = 0; j < d.tvc_n_terms; j++)
                    result.tvc_tau_draws_flat[save_idx * d.tvc_n_terms + j] = tvc_tau[j];
            }

            save_idx++;
        }

        // Progress
        if (verbose && (iter + 1) % 200 == 0) {
            Rcpp::Rcout << "  Gibbs iter " << (iter + 1) << "/" << n_iter;
            if (iter < n_warmup) Rcpp::Rcout << " (warmup)";
            Rcpp::Rcout << std::endl;
        }
    }

    // Compute acceptance rates
    for (int s = 0; s < S; s++) {
        result.accept_phi[s] = phi_total[s] > 0 ?
            (double)phi_accept[s] / phi_total[s] : 0.0;
    }
    result.accept_beta = beta_total > 0 ? (double)beta_accept / beta_total : 0.0;
    result.accept_disp = disp_total > 0 ? (double)disp_accept / disp_total : 0.0;
    result.accept_tvc = tvc_total > 0 ? (double)tvc_accept / tvc_total : 0.0;

    // Build parameter names
    for (int j = 0; j < p_num; j++)
        result.param_names.push_back("beta_num[" + std::to_string(j + 1) + "]");
    for (int j = 0; j < p_denom; j++)
        result.param_names.push_back("beta_denom[" + std::to_string(j + 1) + "]");
    if (has_phi_num) result.param_names.push_back("log_phi_num");
    if (has_phi_denom) result.param_names.push_back("log_phi_denom");
    result.param_names.push_back("log_tau");

    return result;
}

// =========================================================================
// BYM2 Gibbs sampler
// Riebler parameterization: spatial_effect = sigma*(sqrt(rho)*scale*phi + sqrt(1-rho)*theta)
// phi = ICAR structured, theta = iid unstructured
// =========================================================================

// Compute total spatial effect for site s under BYM2
inline double bym2_spatial_effect(int s, const double* phi, const double* theta,
                                   double sigma_total, double rho,
                                   double scale_factor) {
    double sigma_s = sigma_total * std::sqrt(rho);
    double sigma_u = sigma_total * std::sqrt(1.0 - rho);
    return sigma_s * scale_factor * phi[s] + sigma_u * theta[s];
}

// Site log-lik for BYM2 (spatial effect computed from phi + theta)
inline double site_log_lik_bym2(int s, const double* phi, const double* theta,
                                 double sigma_total, double rho, double scale_factor,
                                 const double* beta_num, const double* beta_denom,
                                 double disp_num, double disp_denom,
                                 const GibbsData& d) {
    double ll = 0.0;
    bool is_binomial = (d.family == GibbsFamily::BINOMIAL);
    double spatial = bym2_spatial_effect(s, phi, theta, sigma_total, rho, scale_factor);

    for (int idx = d.site_obs_ptr[s]; idx < d.site_obs_ptr[s + 1]; idx++) {
        int i = d.site_obs_idx[idx];
        double eta_num = spatial;
        double eta_denom = is_binomial ? 0.0 : spatial;

        for (int p = 0; p < d.p_num; p++)
            eta_num += d.X_num[i * d.p_num + p] * beta_num[p];
        if (!is_binomial) {
            for (int p = 0; p < d.p_denom; p++)
                eta_denom += d.X_denom[i * d.p_denom + p] * beta_denom[p];
        }

        ll += obs_log_lik(i, eta_num, eta_denom, disp_num, disp_denom, d);
    }
    return ll;
}

// Full log-lik for BYM2
inline double full_log_lik_bym2(const double* phi, const double* theta,
                                 double sigma_total, double rho, double scale_factor,
                                 const double* beta_num, const double* beta_denom,
                                 double disp_num, double disp_denom,
                                 const GibbsData& d) {
    double ll = 0.0;
    bool is_binomial = (d.family == GibbsFamily::BINOMIAL);

    for (int i = 0; i < d.N; i++) {
        int s = d.spatial_group[i];
        double spatial = bym2_spatial_effect(s, phi, theta, sigma_total, rho, scale_factor);
        double eta_num = spatial;
        double eta_denom = is_binomial ? 0.0 : spatial;

        for (int p = 0; p < d.p_num; p++)
            eta_num += d.X_num[i * d.p_num + p] * beta_num[p];
        if (!is_binomial) {
            for (int p = 0; p < d.p_denom; p++)
                eta_denom += d.X_denom[i * d.p_denom + p] * beta_denom[p];
        }

        ll += obs_log_lik(i, eta_num, eta_denom, disp_num, disp_denom, d);
    }
    return ll;
}

inline GibbsResult run_gibbs_bym2(
    const GibbsData& d,
    int n_iter,
    int n_warmup,
    int thin,
    unsigned int seed,
    bool verbose
) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> rnorm(0.0, 1.0);
    std::uniform_real_distribution<double> runif(0.0, 1.0);

    int S = d.S;
    double scale_factor = d.bym2_scale;
    bool is_binomial = (d.family == GibbsFamily::BINOMIAL);
    bool has_disp_num = (d.family == GibbsFamily::NEGBIN_NEGBIN ||
                         d.family == GibbsFamily::NEGBIN_GAMMA);
    bool has_disp_denom = !is_binomial;
    const bool constrain = can_constrain_field(d);

    // Parameter dimensions
    int p_num = d.p_num;
    int p_denom = is_binomial ? 0 : d.p_denom;
    int n_hyper = 2;  // log_sigma_total, logit_rho
    if (has_disp_num) n_hyper++;
    if (has_disp_denom) n_hyper++;
    int n_params = p_num + p_denom + n_hyper;

    // Initialize
    std::vector<double> beta_num(p_num, 0.0);
    std::vector<double> beta_denom(p_denom, 0.0);
    std::vector<double> phi(S, 0.0);    // ICAR structured
    std::vector<double> theta(S, 0.0);  // iid unstructured
    double log_sigma_total = std::log(1.0);
    double logit_rho = 0.0;  // rho = 0.5
    double log_disp_num = std::log(1.0);
    double log_disp_denom = std::log(5.0);

    if (p_num > 0) beta_num[0] = 1.0;
    if (p_denom > 0) beta_denom[0] = 1.0;

    // Proposal scales
    double beta_scale = 0.1;
    double disp_scale = 0.1;
    double sigma_scale = 0.1;
    double rho_scale = 0.3;
    double level_scale = 0.2;
    double sigma_joint_scale = 0.2;
    double rho_joint_scale = 0.2;

    // Acceptance tracking
    std::vector<int> phi_accept(S, 0), phi_total(S, 0);
    std::vector<int> theta_accept(S, 0), theta_total(S, 0);
    int beta_accept_cnt = 0, beta_total_cnt = 0;
    int disp_accept_cnt = 0, disp_total_cnt = 0;
    int sigma_accept_cnt = 0, sigma_total_cnt = 0;
    int rho_accept_cnt = 0, rho_total_cnt = 0;
    int level_accept_cnt = 0, level_total_cnt = 0;
    int sigma_joint_accept_cnt = 0, sigma_joint_total_cnt = 0;
    int rho_joint_accept_cnt = 0, rho_joint_total_cnt = 0;

    // Output
    int n_save = (n_iter - n_warmup) / thin;
    GibbsResult result;
    result.n_params = n_params;
    result.n_save = n_save;
    result.S = S;
    result.draws_flat.resize(n_save * n_params);
    result.phi_draws_flat.resize(n_save * S);  // Store total spatial effect
    result.accept_phi.resize(S, 0.0);

    int save_idx = 0;

    // ---- Main Gibbs loop ----
    for (int iter = 0; iter < n_iter; iter++) {
        double sigma_total = std::exp(log_sigma_total);
        double rho = 1.0 / (1.0 + std::exp(-logit_rho));
        double disp_num_val = has_disp_num ? std::exp(log_disp_num) : 1.0;
        double disp_denom_val = has_disp_denom ? std::exp(log_disp_denom) : 1.0;

        // Scale the structured component contributes to eta; the Riebler
        // parameterization carries it here rather than in phi's own precision.
        double sigma_s = sigma_total * std::sqrt(rho);

        // ---- 1. Update phi (ICAR structured) via MH ----
        // phi is the standardised ICAR field: precision 1, scale carried by
        // sigma and rho. Drawing from its own full conditional is what makes
        // the acceptance ratio the likelihood ratio alone -- for the field
        // itself. The field reaches eta as sigma_s * scale_factor * phi, so
        // holding phi to sum zero moves sigma_s * scale_factor * phi_mean into
        // the intercept every sweep; that move has its own prior (the
        // intercept's N(0, sigma_beta^2)) and belongs in the same acceptance
        // test as the site update, exactly as run_gibbs_icar's phi_off does
        // above. Deciding it unconditionally after the sweep (no MH test)
        // ignored the intercept's prior and let the field's level random-walk
        // into the intercept without resistance.
        {
            const double c = sigma_s * scale_factor;
            double phi_off = 0.0;
            for (int s = 0; s < S; s++) {
                auto [cond_mean, cond_prec] = icar_conditional(
                    s, phi.data(), 1.0, d.adj_row_ptr, d.adj_col_idx, d.n_neighbors);

                const double actual_s = phi[s] + phi_off;
                const double phi_prop = cond_mean + phi_off
                                      + rnorm(rng) / std::sqrt(cond_prec + 1e-10);
                const double delta = phi_prop - actual_s;
                const double phi_shift = constrain ? delta / S : 0.0;
                const double eta_shift = c * phi_shift;

                phi[s] = actual_s;
                double ll_curr = site_log_lik_bym2(s, phi.data(), theta.data(),
                                                    sigma_total, rho, scale_factor,
                                                    beta_num.data(), beta_denom.data(),
                                                    disp_num_val, disp_denom_val, d);
                phi[s] = actual_s + delta;
                double ll_prop = site_log_lik_bym2(s, phi.data(), theta.data(),
                                                    sigma_total, rho, scale_factor,
                                                    beta_num.data(), beta_denom.data(),
                                                    disp_num_val, disp_denom_val, d);

                double log_ratio = ll_prop - ll_curr;
                if (constrain) {
                    log_ratio += intercept_shift_log_prior_ratio(
                        beta_num.data(), beta_denom.data(), d, eta_shift);
                }

                if (std::log(runif(rng)) < log_ratio) {
                    if (constrain) {
                        phi_off -= phi_shift;
                        beta_num[d.icept_num] += eta_shift;
                        if (d.icept_denom >= 0) beta_denom[d.icept_denom] += eta_shift;
                    }
                    phi_accept[s]++;
                } else {
                    phi[s] = actual_s;  // Reject
                }
                phi_total[s]++;
            }
            if (phi_off != 0.0) {
                for (int s = 0; s < S; s++) phi[s] += phi_off;
            }
        }

        // ---- 2. Update theta (iid unstructured) via MH ----
        // Prior: theta_s ~ N(0, 1)
        for (int s = 0; s < S; s++) {
            double theta_prop = rnorm(rng);  // Propose from prior N(0,1)

            double old_theta_s = theta[s];
            double ll_curr = site_log_lik_bym2(s, phi.data(), theta.data(),
                                                sigma_total, rho, scale_factor,
                                                beta_num.data(), beta_denom.data(),
                                                disp_num_val, disp_denom_val, d);
            theta[s] = theta_prop;
            double ll_prop = site_log_lik_bym2(s, phi.data(), theta.data(),
                                                sigma_total, rho, scale_factor,
                                                beta_num.data(), beta_denom.data(),
                                                disp_num_val, disp_denom_val, d);

            if (std::log(runif(rng)) < ll_prop - ll_curr) {
                theta_accept[s]++;
            } else {
                theta[s] = old_theta_s;
            }
            theta_total[s]++;
        }

        // ---- 2b. Move the unstructured component's level against the intercept ----
        // sigma_u * mean(theta) is a level the intercept can carry just as well,
        // and the per-site theta updates only reach it by agreeing S times over.
        // Translating the whole of theta at once walks that direction in one
        // step, at eta held fixed, so the acceptance ratio is the priors alone.
        if (constrain) {
            const double sigma_u = sigma_total * std::sqrt(1.0 - rho);
            const double c = level_scale * rnorm(rng);
            const double shift = sigma_u * c;

            double theta_sum = 0.0;
            for (int s = 0; s < S; s++) theta_sum += theta[s];

            // theta_s ~ N(0, 1), shifted by -c
            double log_ratio = -0.5 * (S * c * c - 2.0 * c * theta_sum);

            // beta ~ N(0, sigma_beta^2), shifted by +sigma_u * c
            const double prec_beta = 1.0 / (d.sigma_beta * d.sigma_beta);
            const double b_num = beta_num[d.icept_num];
            log_ratio -= 0.5 * prec_beta * (shift * shift + 2.0 * b_num * shift);
            if (d.icept_denom >= 0) {
                const double b_den = beta_denom[d.icept_denom];
                log_ratio -= 0.5 * prec_beta * (shift * shift + 2.0 * b_den * shift);
            }

            if (std::log(runif(rng)) < log_ratio) {
                for (int s = 0; s < S; s++) theta[s] -= c;
                beta_num[d.icept_num] += shift;
                if (d.icept_denom >= 0) beta_denom[d.icept_denom] += shift;
                level_accept_cnt++;
            }
            level_total_cnt++;
        }

        // ---- 3. Update sigma_total via MH on log scale ----
        {
            double log_sigma_prop = log_sigma_total + sigma_scale * rnorm(rng);
            double sigma_prop = std::exp(log_sigma_prop);

            double ll_curr = full_log_lik_bym2(phi.data(), theta.data(),
                                                sigma_total, rho, scale_factor,
                                                beta_num.data(), beta_denom.data(),
                                                disp_num_val, disp_denom_val, d);
            double ll_prop = full_log_lik_bym2(phi.data(), theta.data(),
                                                sigma_prop, rho, scale_factor,
                                                beta_num.data(), beta_denom.data(),
                                                disp_num_val, disp_denom_val, d);

            double lp_curr = half_cauchy_log_dens_log_scale(sigma_total, d.sigma_re_scale);
            double lp_prop = half_cauchy_log_dens_log_scale(sigma_prop, d.sigma_re_scale);

            if (std::log(runif(rng)) < (ll_prop + lp_prop) - (ll_curr + lp_curr)) {
                log_sigma_total = log_sigma_prop;
                sigma_accept_cnt++;
            }
            sigma_total_cnt++;
        }

        // ---- 3b. Move sigma_total against the field, at eta held fixed ----
        // sigma reaches eta only through sigma * (field), so scaling the field
        // by the reciprocal puts every linear predictor back where it was and
        // the likelihood drops out of the acceptance ratio. Step 3 can only
        // move sigma as far as the likelihood allows with the field pinned,
        // which at S sites is a width that shrinks with S; this move has no
        // such limit and leaves the priors and the Jacobian to decide.
        {
            sigma_total = std::exp(log_sigma_total);

            const double delta = sigma_joint_scale * rnorm(rng);
            const double a = std::exp(-delta);
            const double sigma_prop = sigma_total * std::exp(delta);

            const double quad_phi = icar_quadratic_form(phi.data(), d);
            double ss_theta = 0.0;
            for (int s = 0; s < S; s++) ss_theta += theta[s] * theta[s];

            // phi carries S-1 free dimensions under the sum-to-zero constraint
            // and theta carries S, so scaling both by a has log-Jacobian
            // (2S-1) * log(a).
            double log_ratio = -0.5 * (a * a - 1.0) * (quad_phi + ss_theta)
                             - (2.0 * S - 1.0) * delta
                             + half_cauchy_log_dens_log_scale(sigma_prop, d.sigma_re_scale)
                             - half_cauchy_log_dens_log_scale(sigma_total, d.sigma_re_scale);

            if (std::log(runif(rng)) < log_ratio) {
                for (int s = 0; s < S; s++) { phi[s] *= a; theta[s] *= a; }
                log_sigma_total = std::log(sigma_prop);
                sigma_joint_accept_cnt++;
            }
            sigma_joint_total_cnt++;
        }

        // ---- 4. Update rho via MH on logit scale ----
        {
            double logit_rho_prop = logit_rho + rho_scale * rnorm(rng);
            double rho_prop = 1.0 / (1.0 + std::exp(-logit_rho_prop));

            sigma_total = std::exp(log_sigma_total);  // Refresh

            double ll_curr = full_log_lik_bym2(phi.data(), theta.data(),
                                                sigma_total, rho, scale_factor,
                                                beta_num.data(), beta_denom.data(),
                                                disp_num_val, disp_denom_val, d);
            double ll_prop = full_log_lik_bym2(phi.data(), theta.data(),
                                                sigma_total, rho_prop, scale_factor,
                                                beta_num.data(), beta_denom.data(),
                                                disp_num_val, disp_denom_val, d);

            double lp_curr = unif01_log_dens_logit_scale(rho);
            double lp_prop = unif01_log_dens_logit_scale(rho_prop);

            if (std::log(runif(rng)) < (ll_prop + lp_prop) - (ll_curr + lp_curr)) {
                logit_rho = logit_rho_prop;
                rho_accept_cnt++;
            }
            rho_total_cnt++;
        }

        // ---- 4b. Move rho against the field, at eta held fixed ----
        // rho splits a fixed total between the structured and unstructured
        // components. Rescaling each by the reciprocal of its own share holds
        // eta where it was, so the split can be re-drawn without the
        // likelihood having any say, which is what step 4 lacks.
        {
            rho = 1.0 / (1.0 + std::exp(-logit_rho));

            const double logit_rho_prop = logit_rho + rho_joint_scale * rnorm(rng);
            const double rho_prop = 1.0 / (1.0 + std::exp(-logit_rho_prop));

            if (rho_prop > 0.0 && rho_prop < 1.0) {
                const double a_phi = std::sqrt(rho / rho_prop);
                const double a_th = std::sqrt((1.0 - rho) / (1.0 - rho_prop));

                const double quad_phi = icar_quadratic_form(phi.data(), d);
                double ss_theta = 0.0;
                for (int s = 0; s < S; s++) ss_theta += theta[s] * theta[s];

                double log_ratio = -0.5 * (a_phi * a_phi - 1.0) * quad_phi
                                 - 0.5 * (a_th * a_th - 1.0) * ss_theta
                                 + (S - 1.0) * std::log(a_phi) + S * std::log(a_th)
                                 + unif01_log_dens_logit_scale(rho_prop)
                                 - unif01_log_dens_logit_scale(rho);

                if (std::isfinite(log_ratio) && std::log(runif(rng)) < log_ratio) {
                    for (int s = 0; s < S; s++) {
                        phi[s] *= a_phi;
                        theta[s] *= a_th;
                    }
                    logit_rho = logit_rho_prop;
                    rho_joint_accept_cnt++;
                }
            }
            rho_joint_total_cnt++;
        }

        // ---- 5. Update beta via block MH ----
        {
            sigma_total = std::exp(log_sigma_total);
            rho = 1.0 / (1.0 + std::exp(-logit_rho));

            std::vector<double> beta_num_prop(beta_num);
            std::vector<double> beta_denom_prop(beta_denom);
            for (int j = 0; j < p_num; j++)
                beta_num_prop[j] += beta_scale * rnorm(rng);
            for (int j = 0; j < p_denom; j++)
                beta_denom_prop[j] += beta_scale * rnorm(rng);

            double ll_curr = full_log_lik_bym2(phi.data(), theta.data(),
                                                sigma_total, rho, scale_factor,
                                                beta_num.data(), beta_denom.data(),
                                                disp_num_val, disp_denom_val, d);
            double ll_prop = full_log_lik_bym2(phi.data(), theta.data(),
                                                sigma_total, rho, scale_factor,
                                                beta_num_prop.data(), beta_denom_prop.data(),
                                                disp_num_val, disp_denom_val, d);

            double lp_curr = beta_log_prior(beta_num.data(), p_num, d.sigma_beta)
                           + beta_log_prior(beta_denom.data(), p_denom, d.sigma_beta);
            double lp_prop = beta_log_prior(beta_num_prop.data(), p_num, d.sigma_beta)
                           + beta_log_prior(beta_denom_prop.data(), p_denom, d.sigma_beta);

            if (std::log(runif(rng)) < (ll_prop + lp_prop) - (ll_curr + lp_curr)) {
                beta_num = beta_num_prop;
                beta_denom = beta_denom_prop;
                beta_accept_cnt++;
            }
            beta_total_cnt++;
        }

        // ---- 6. Update dispersion via MH ----
        if (has_disp_num || has_disp_denom) {
            sigma_total = std::exp(log_sigma_total);
            rho = 1.0 / (1.0 + std::exp(-logit_rho));

            double log_dn_prop = log_disp_num;
            double log_dd_prop = log_disp_denom;
            if (has_disp_num) log_dn_prop += disp_scale * rnorm(rng);
            if (has_disp_denom) log_dd_prop += disp_scale * rnorm(rng);

            double dn_curr = has_disp_num ? std::exp(log_disp_num) : 1.0;
            double dd_curr = has_disp_denom ? std::exp(log_disp_denom) : 1.0;
            double dn_prop = has_disp_num ? std::exp(log_dn_prop) : 1.0;
            double dd_prop = has_disp_denom ? std::exp(log_dd_prop) : 1.0;

            double ll_curr = full_log_lik_bym2(phi.data(), theta.data(),
                                                sigma_total, rho, scale_factor,
                                                beta_num.data(), beta_denom.data(),
                                                dn_curr, dd_curr, d);
            double ll_prop = full_log_lik_bym2(phi.data(), theta.data(),
                                                sigma_total, rho, scale_factor,
                                                beta_num.data(), beta_denom.data(),
                                                dn_prop, dd_prop, d);

            double lp_curr = 0.0, lp_prop = 0.0;
            if (has_disp_num) {
                lp_curr += std::log(dn_curr) - 0.5 * dn_curr;
                lp_prop += std::log(dn_prop) - 0.5 * dn_prop;
            }
            if (has_disp_denom) {
                lp_curr += std::log(dd_curr) - 0.5 * dd_curr;
                lp_prop += std::log(dd_prop) - 0.5 * dd_prop;
            }

            if (std::log(runif(rng)) < (ll_prop + lp_prop) - (ll_curr + lp_curr)) {
                log_disp_num = log_dn_prop;
                log_disp_denom = log_dd_prop;
                disp_accept_cnt++;
            }
            disp_total_cnt++;
        }

        // ---- Adaptation during warmup ----
        if (iter < n_warmup && iter > 0 && iter % 50 == 0) {
            double target = 0.44;

            auto adapt = [&](double& scale, int& acc, int& tot) {
                if (tot == 0) return;
                double rate = (double)acc / tot;
                if (rate > target + 0.05) scale *= 1.2;
                else if (rate < target - 0.05) scale *= 0.8;
                scale = std::max(0.001, std::min(scale, 5.0));
                acc = 0; tot = 0;
            };

            adapt(beta_scale, beta_accept_cnt, beta_total_cnt);
            adapt(sigma_scale, sigma_accept_cnt, sigma_total_cnt);
            adapt(rho_scale, rho_accept_cnt, rho_total_cnt);
            adapt(level_scale, level_accept_cnt, level_total_cnt);
            adapt(sigma_joint_scale, sigma_joint_accept_cnt, sigma_joint_total_cnt);
            adapt(rho_joint_scale, rho_joint_accept_cnt, rho_joint_total_cnt);
            if (has_disp_num || has_disp_denom)
                adapt(disp_scale, disp_accept_cnt, disp_total_cnt);
        }

        // ---- Store draws ----
        if (iter >= n_warmup && (iter - n_warmup) % thin == 0) {
            sigma_total = std::exp(log_sigma_total);
            rho = 1.0 / (1.0 + std::exp(-logit_rho));

            int row = save_idx * n_params;
            int col = 0;
            for (int j = 0; j < p_num; j++) result.draws_flat[row + col++] = beta_num[j];
            for (int j = 0; j < p_denom; j++) result.draws_flat[row + col++] = beta_denom[j];
            if (has_disp_num) result.draws_flat[row + col++] = log_disp_num;
            if (has_disp_denom) result.draws_flat[row + col++] = log_disp_denom;
            result.draws_flat[row + col++] = log_sigma_total;
            result.draws_flat[row + col++] = logit_rho;

            // Store total spatial effect per site
            for (int s = 0; s < S; s++) {
                result.phi_draws_flat[save_idx * S + s] =
                    bym2_spatial_effect(s, phi.data(), theta.data(),
                                        sigma_total, rho, scale_factor);
            }

            save_idx++;
        }

        if (verbose && (iter + 1) % 200 == 0) {
            Rcpp::Rcout << "  Gibbs iter " << (iter + 1) << "/" << n_iter;
            if (iter < n_warmup) Rcpp::Rcout << " (warmup)";
            Rcpp::Rcout << std::endl;
        }
    }

    // Acceptance rates
    for (int s = 0; s < S; s++) {
        result.accept_phi[s] = phi_total[s] > 0 ?
            (double)phi_accept[s] / phi_total[s] : 0.0;
    }
    result.accept_beta = beta_total_cnt > 0 ? (double)beta_accept_cnt / beta_total_cnt : 0.0;
    result.accept_disp = disp_total_cnt > 0 ? (double)disp_accept_cnt / disp_total_cnt : 0.0;

    // Param names
    for (int j = 0; j < p_num; j++)
        result.param_names.push_back("beta_num[" + std::to_string(j + 1) + "]");
    for (int j = 0; j < p_denom; j++)
        result.param_names.push_back("beta_denom[" + std::to_string(j + 1) + "]");
    if (has_disp_num) result.param_names.push_back("log_disp_num");
    if (has_disp_denom) result.param_names.push_back("log_disp_denom");
    result.param_names.push_back("log_sigma_total");
    result.param_names.push_back("logit_rho");

    return result;
}

} // namespace gibbs
