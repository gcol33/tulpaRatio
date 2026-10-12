// gibbs_spatial.h
// Component-wise Gibbs sampler for ICAR/BYM2 spatial models
// Designed for large S where HMC struggles with dimensionality
//
// Update scheme, ICAR:
//   1. phi_s | phi_{-s}, beta, tau, y  (univariate MH per site)
//   2. tau | phi                        (conjugate Gamma for ICAR)
//   3. beta | phi, y                    (block MH with Gaussian proposal)
//   4. dispersion | rest                (univariate MH on log scale)
// BYM2 (Knorr-Held and Rue 2002, Scand. J. Stat. 29:597):
//   1. (beta, phi, theta) | sigma, rho  (independence MH, Gaussian proposal)
//   2. (sigma, rho, beta, phi, theta)   (one block: random walk on the two
//                                        hyperparameters, the latent field
//                                        drawn from its Gaussian approximation)
//   3. dispersion | rest                (MH on log scale)

#pragma once

#include <Rcpp.h>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>

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

// d(log-lik)/d(eta) and -d^2(log-lik)/d(eta)^2 of one observation, for the
// numerator and the denominator predictor. Every family here is log-concave
// in its predictor, so each curvature is non-negative:
//   Poisson   g = y - mu,                     h = mu
//   NB        g = y - (y + r) mu / (r + mu),  h = (y + r) mu r / (r + mu)^2
//   Gamma     g = alpha (y / mu - 1),         h = alpha y / mu
//   binomial  g = y - n p,                    h = n p (1 - p)
struct ObsDerivs {
    double g_num = 0.0, h_num = 0.0, g_den = 0.0, h_den = 0.0;
};

inline ObsDerivs obs_log_lik_derivs(int i, double eta_num, double eta_denom,
                                    double disp_num, double disp_denom,
                                    const GibbsData& d) {
    ObsDerivs out;
    auto negbin = [](double y, double eta, double r, double& g, double& h) {
        const double mu = std::exp(std::min(eta, 20.0));
        g = y - (y + r) * mu / (r + mu);
        h = (y + r) * mu * r / ((r + mu) * (r + mu));
    };
    auto gamma = [](double y, double eta, double alpha, double& g, double& h) {
        const double ratio = y * std::exp(-std::min(eta, 20.0));
        g = alpha * (ratio - 1.0);
        h = alpha * ratio;
    };
    switch (d.family) {
    case GibbsFamily::POISSON_GAMMA: {
        const double mu = std::exp(std::min(eta_num, 20.0));
        out.g_num = d.y_num[i] - mu;
        out.h_num = mu;
        gamma(d.y_denom_cont[i], eta_denom, disp_denom, out.g_den, out.h_den);
        break;
    }
    case GibbsFamily::NEGBIN_NEGBIN:
        negbin(d.y_num[i], eta_num, disp_num, out.g_num, out.h_num);
        negbin(d.y_denom[i], eta_denom, disp_denom, out.g_den, out.h_den);
        break;
    case GibbsFamily::BINOMIAL: {
        const double p = 1.0 / (1.0 + std::exp(-eta_num));
        out.g_num = d.y_num[i] - d.y_denom[i] * p;
        out.h_num = d.y_denom[i] * p * (1.0 - p);
        break;
    }
    case GibbsFamily::NEGBIN_GAMMA:
        negbin(d.y_num[i], eta_num, disp_num, out.g_num, out.h_num);
        gamma(d.y_denom_cont[i], eta_denom, disp_denom, out.g_den, out.h_den);
        break;
    }
    return out;
}

// The latent Gaussian of a BYM2 model, x = (beta_num, beta_denom, phi, theta),
// with prior precision blockdiag(I / sigma_beta^2, I / sigma_beta^2, Q, I)
// (Q the ICAR precision of the standardised field). The field reaches both
// predictors as c phi_s + u theta_s, with c = sigma sqrt(rho) scale and
// u = sigma sqrt(1 - rho).
struct Bym2Latent {
    int p_num = 0, p_den = 0, S = 0;
    int phi0() const { return p_num + p_den; }
    int theta0() const { return p_num + p_den + S; }
    int size() const { return p_num + p_den + 2 * S; }
};

// Predictors of every observation at x
inline void bym2_latent_eta(const Eigen::VectorXd& x, double c, double u,
                            const Bym2Latent& lay, const GibbsData& d,
                            std::vector<double>& eta_num,
                            std::vector<double>& eta_den) {
    const bool is_binomial = (d.family == GibbsFamily::BINOMIAL);
    eta_num.assign(d.N, 0.0);
    eta_den.assign(d.N, 0.0);
    for (int i = 0; i < d.N; i++) {
        const int s = d.spatial_group[i];
        const double field = c * x[lay.phi0() + s] + u * x[lay.theta0() + s];
        double en = field;
        for (int j = 0; j < lay.p_num; j++) en += d.X_num[i * d.p_num + j] * x[j];
        eta_num[i] = en;
        if (!is_binomial) {
            double ed = field;
            for (int j = 0; j < lay.p_den; j++)
                ed += d.X_denom[i * d.p_denom + j] * x[lay.p_num + j];
            eta_den[i] = ed;
        }
    }
}

// Log density of (x, sigma, rho) given the dispersions, on the coordinates the
// sampler moves: log sigma and logit rho, each prior's Jacobian included.
inline double bym2_log_target(const Eigen::VectorXd& x, double sigma, double rho,
                              double disp_num, double disp_den,
                              const Bym2Latent& lay, const GibbsData& d) {
    const double c = sigma * std::sqrt(rho) * d.bym2_scale;
    const double u = sigma * std::sqrt(1.0 - rho);
    std::vector<double> eta_num, eta_den;
    bym2_latent_eta(x, c, u, lay, d, eta_num, eta_den);
    double lp = 0.0;
    for (int i = 0; i < d.N; i++) {
        lp += obs_log_lik(i, eta_num[i], eta_den[i], disp_num, disp_den, d);
    }
    lp += beta_log_prior(x.data(), lay.p_num, d.sigma_beta);
    lp += beta_log_prior(x.data() + lay.p_num, lay.p_den, d.sigma_beta);
    lp -= 0.5 * icar_quadratic_form(x.data() + lay.phi0(), d);
    for (int s = 0; s < lay.S; s++) lp -= 0.5 * x[lay.theta0() + s] * x[lay.theta0() + s];
    lp += half_cauchy_log_dens_log_scale(sigma, d.sigma_re_scale);
    lp += unif01_log_dens_logit_scale(rho);
    return lp;
}

// A Gaussian approximation N(mode, P^-1) to the latent field's full
// conditional at given hyperparameters: Newton iterations from zero to the
// conditional mode, P the negative Hessian there. It depends on the
// hyperparameters and the data alone, never on the current field, so it is a
// valid proposal in both directions of the move. Along the ICAR null
// direction the likelihood sees phi and the intercept only through their sum,
// so the intercept's prior is what keeps P definite there; a small ridge on
// phi covers a graph whose components have no intercept to lean on.
struct Bym2Gauss {
    Eigen::VectorXd mode;
    Eigen::SparseMatrix<double> P;
    Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> chol;
    double log_det = 0.0;
    Eigen::VectorXd Pinv_a;   // P^-1 a, for the constraint
    double a_Pinv_a = 0.0;
    bool ok = false;
};

inline bool bym2_gaussian_approx(double sigma, double rho, double disp_num,
                                 double disp_den, bool constrain,
                                 const Bym2Latent& lay, const GibbsData& d,
                                 Bym2Gauss& out) {
    typedef Eigen::Triplet<double> Trip;
    const int K = lay.size();
    const int S = lay.S;
    const double c = sigma * std::sqrt(rho) * d.bym2_scale;
    const double u = sigma * std::sqrt(1.0 - rho);
    const double prec_beta = 1.0 / (d.sigma_beta * d.sigma_beta);
    const double ridge = 1e-6;
    const bool is_binomial = (d.family == GibbsFamily::BINOMIAL);

    // Objective with the ridge the proposal's precision carries
    auto objective = [&](const Eigen::VectorXd& x) {
        double ss_phi = 0.0;
        for (int s = 0; s < S; s++) ss_phi += x[lay.phi0() + s] * x[lay.phi0() + s];
        return bym2_log_target(x, sigma, rho, disp_num, disp_den, lay, d)
               - 0.5 * ridge * ss_phi;
    };

    // Gradient and precision at x
    auto build = [&](const Eigen::VectorXd& x, Eigen::VectorXd& grad,
                     Eigen::SparseMatrix<double>& P) {
        std::vector<double> eta_num, eta_den;
        bym2_latent_eta(x, c, u, lay, d, eta_num, eta_den);
        grad = Eigen::VectorXd::Zero(K);
        std::vector<double> h_site(S, 0.0), g_site(S, 0.0);
        Eigen::MatrixXd H_nn = Eigen::MatrixXd::Zero(lay.p_num, lay.p_num);
        Eigen::MatrixXd H_dd = Eigen::MatrixXd::Zero(lay.p_den, lay.p_den);
        Eigen::MatrixXd H_nf = Eigen::MatrixXd::Zero(lay.p_num, S);
        Eigen::MatrixXd H_df = Eigen::MatrixXd::Zero(lay.p_den, S);
        for (int i = 0; i < d.N; i++) {
            const int s = d.spatial_group[i];
            const ObsDerivs od = obs_log_lik_derivs(i, eta_num[i], eta_den[i],
                                                    disp_num, disp_den, d);
            const double* xn = d.X_num + i * d.p_num;
            for (int j = 0; j < lay.p_num; j++) {
                grad[j] += od.g_num * xn[j];
                for (int k = 0; k <= j; k++) H_nn(j, k) += od.h_num * xn[j] * xn[k];
                H_nf(j, s) += od.h_num * xn[j];
            }
            if (!is_binomial) {
                const double* xd = d.X_denom + i * d.p_denom;
                for (int j = 0; j < lay.p_den; j++) {
                    grad[lay.p_num + j] += od.g_den * xd[j];
                    for (int k = 0; k <= j; k++) H_dd(j, k) += od.h_den * xd[j] * xd[k];
                    H_df(j, s) += od.h_den * xd[j];
                }
            }
            g_site[s] += od.g_num + od.g_den;
            h_site[s] += od.h_num + od.h_den;
        }

        // Prior and penalty gradients
        for (int j = 0; j < lay.p_num + lay.p_den; j++) grad[j] -= prec_beta * x[j];
        for (int s = 0; s < S; s++) {
            const int ip = lay.phi0() + s, it = lay.theta0() + s;
            double q_phi = d.n_neighbors[s] * x[ip];
            for (int k = d.adj_row_ptr[s]; k < d.adj_row_ptr[s + 1]; k++)
                q_phi -= x[lay.phi0() + d.adj_col_idx[k]];
            grad[ip] += c * g_site[s] - q_phi - ridge * x[ip];
            grad[it] += u * g_site[s] - x[it];
        }

        std::vector<Trip> trip;
        trip.reserve(static_cast<size_t>(K) * 4 + d.adj_row_ptr[S] +
                     static_cast<size_t>(lay.p_num + lay.p_den) * (4 * S + 2 * (lay.p_num + lay.p_den)));
        auto add_sym = [&trip](int r, int col, double v) {
            if (v == 0.0) return;
            trip.emplace_back(r, col, v);
            if (r != col) trip.emplace_back(col, r, v);
        };
        for (int j = 0; j < lay.p_num; j++) {
            add_sym(j, j, prec_beta);
            for (int k = 0; k <= j; k++) add_sym(j, k, H_nn(j, k));
            for (int s = 0; s < S; s++) {
                add_sym(j, lay.phi0() + s, c * H_nf(j, s));
                add_sym(j, lay.theta0() + s, u * H_nf(j, s));
            }
        }
        for (int j = 0; j < lay.p_den; j++) {
            const int r = lay.p_num + j;
            add_sym(r, r, prec_beta);
            for (int k = 0; k <= j; k++) add_sym(r, lay.p_num + k, H_dd(j, k));
            for (int s = 0; s < S; s++) {
                add_sym(r, lay.phi0() + s, c * H_df(j, s));
                add_sym(r, lay.theta0() + s, u * H_df(j, s));
            }
        }
        for (int s = 0; s < S; s++) {
            const int ip = lay.phi0() + s, it = lay.theta0() + s;
            add_sym(ip, ip, d.n_neighbors[s] + ridge + c * c * h_site[s]);
            for (int k = d.adj_row_ptr[s]; k < d.adj_row_ptr[s + 1]; k++) {
                trip.emplace_back(ip, lay.phi0() + d.adj_col_idx[k], -1.0);
            }
            add_sym(it, it, 1.0 + u * u * h_site[s]);
            add_sym(ip, it, c * u * h_site[s]);
        }
        P.resize(K, K);
        P.setFromTriplets(trip.begin(), trip.end());
    };

    Eigen::VectorXd x = Eigen::VectorXd::Zero(K);
    Eigen::VectorXd grad;
    double f = objective(x);
    for (int it = 0; it < 50; it++) {
        build(x, grad, out.P);
        out.chol.compute(out.P);
        if (out.chol.info() != Eigen::Success) return out.ok = false;
        const Eigen::VectorXd step = out.chol.solve(grad);
        double t = 1.0, f_new = objective(x + step);
        while (!(f_new >= f - 1e-10) && t > 1e-6) {
            t *= 0.5;
            f_new = objective(x + t * step);
        }
        x += t * step;
        f = f_new;
        if ((t * step).cwiseAbs().maxCoeff() < 1e-8) break;
    }
    build(x, grad, out.P);
    out.chol.compute(out.P);
    if (out.chol.info() != Eigen::Success || !x.allFinite()) return out.ok = false;
    out.mode = x;

    Eigen::SparseMatrix<double> L = out.chol.matrixL();
    out.log_det = 2.0 * L.diagonal().array().log().sum();
    if (constrain) {
        Eigen::VectorXd a = Eigen::VectorXd::Zero(K);
        a.segment(lay.phi0(), S).setOnes();
        out.Pinv_a = out.chol.solve(a);
        out.a_Pinv_a = a.dot(out.Pinv_a);
    }
    return out.ok = std::isfinite(out.log_det);
}

// Log density of the proposal at x: N(mode, P^-1), conditioned on sum(phi) = 0
// when the field is constrained, where every x it is evaluated at satisfies it.
inline double bym2_proposal_log_density(const Bym2Gauss& g, const Eigen::VectorXd& x,
                                        bool constrain, const Bym2Latent& lay) {
    const Eigen::VectorXd r = x - g.mode;
    double lq = 0.5 * g.log_det - 0.5 * r.dot(g.P * r)
              - 0.5 * lay.size() * std::log(2.0 * M_PI);
    if (constrain) {
        const double a_mode = g.mode.segment(lay.phi0(), lay.S).sum();
        lq += 0.5 * std::log(2.0 * M_PI * g.a_Pinv_a)
              + 0.5 * a_mode * a_mode / g.a_Pinv_a;
    }
    return lq;
}

// A draw from the proposal: N(mode, P^-1), then the sum-to-zero correction
// x - P^-1 a (a'x) / (a' P^-1 a) (Rue and Held 2005, Sec. 2.3.3).
template <typename RNG>
inline Eigen::VectorXd bym2_proposal_draw(const Bym2Gauss& g, bool constrain,
                                          const Bym2Latent& lay, RNG& rng) {
    std::normal_distribution<double> rnorm(0.0, 1.0);
    Eigen::VectorXd z(lay.size());
    for (int k = 0; k < lay.size(); k++) z[k] = rnorm(rng);
    const Eigen::VectorXd w = g.chol.matrixU().solve(z);
    Eigen::VectorXd x = g.mode + g.chol.permutationPinv() * w;
    if (constrain) {
        const double ax = x.segment(lay.phi0(), lay.S).sum();
        x -= g.Pinv_a * (ax / g.a_Pinv_a);
    }
    return x;
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
    Bym2Latent lay;
    lay.p_num = d.p_num;
    lay.p_den = is_binomial ? 0 : d.p_denom;
    lay.S = S;
    int p_num = lay.p_num;
    int p_denom = lay.p_den;
    int n_hyper = 2;  // log_sigma_total, logit_rho
    if (has_disp_num) n_hyper++;
    if (has_disp_denom) n_hyper++;
    int n_params = p_num + p_denom + n_hyper;

    // Initialize: the latent field x = (beta_num, beta_denom, phi, theta)
    Eigen::VectorXd x = Eigen::VectorXd::Zero(lay.size());
    double log_sigma_total = std::log(1.0);
    double logit_rho = 0.0;  // rho = 0.5
    double log_disp_num = std::log(1.0);
    double log_disp_denom = std::log(5.0);

    // Proposal scales of the hyperparameter random walk and the dispersions
    double hyper_scale = 0.3;
    double disp_scale = 0.1;

    int field_accept_cnt = 0, field_total_cnt = 0;
    int hyper_accept_cnt = 0, hyper_total_cnt = 0;
    int disp_accept_cnt = 0, disp_total_cnt = 0;

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
    auto rho_of = [](double logit) { return 1.0 / (1.0 + std::exp(-logit)); };

    Bym2Gauss approx_curr, approx_prop;

    // ---- Main loop ----
    for (int iter = 0; iter < n_iter; iter++) {
        double disp_num_val = has_disp_num ? std::exp(log_disp_num) : 1.0;
        double disp_denom_val = has_disp_denom ? std::exp(log_disp_denom) : 1.0;
        double sigma_total = std::exp(log_sigma_total);
        double rho = rho_of(logit_rho);

        // ---- 1. The latent field at the current hyperparameters ----
        // An independence proposal from the field's Gaussian approximation,
        // so the acceptance ratio weighs the target against that
        // approximation at both points.
        const bool approx_ok = bym2_gaussian_approx(sigma_total, rho, disp_num_val,
                                                    disp_denom_val, constrain, lay,
                                                    d, approx_curr);
        double lp_curr = bym2_log_target(x, sigma_total, rho, disp_num_val,
                                         disp_denom_val, lay, d);
        if (approx_ok) {
            const double lq_curr = bym2_proposal_log_density(approx_curr, x, constrain, lay);
            const Eigen::VectorXd x_prop = bym2_proposal_draw(approx_curr, constrain, lay, rng);
            const double lp_prop = bym2_log_target(x_prop, sigma_total, rho, disp_num_val,
                                                   disp_denom_val, lay, d);
            const double lq_prop = bym2_proposal_log_density(approx_curr, x_prop,
                                                             constrain, lay);
            const double log_ratio = (lp_prop - lq_prop) - (lp_curr - lq_curr);
            if (std::isfinite(lp_prop) && std::log(runif(rng)) < log_ratio) {
                x = x_prop;
                lp_curr = lp_prop;
                field_accept_cnt++;
            }
            field_total_cnt++;

            // ---- 2. One block: (sigma, rho) by random walk, the field from
            // its Gaussian approximation at the proposed values ----
            const double log_sigma_prop = log_sigma_total + hyper_scale * rnorm(rng);
            const double logit_rho_prop = logit_rho + hyper_scale * rnorm(rng);
            const double sigma_prop = std::exp(log_sigma_prop);
            const double rho_prop = rho_of(logit_rho_prop);
            if (rho_prop > 0.0 && rho_prop < 1.0 &&
                bym2_gaussian_approx(sigma_prop, rho_prop, disp_num_val, disp_denom_val,
                                     constrain, lay, d, approx_prop)) {
                const double lq_back = bym2_proposal_log_density(approx_curr, x,
                                                                 constrain, lay);
                const Eigen::VectorXd x_new = bym2_proposal_draw(approx_prop, constrain,
                                                                 lay, rng);
                const double lq_fwd = bym2_proposal_log_density(approx_prop, x_new,
                                                                constrain, lay);
                const double lp_new = bym2_log_target(x_new, sigma_prop, rho_prop,
                                                      disp_num_val, disp_denom_val,
                                                      lay, d);
                const double log_ratio2 = (lp_new - lq_fwd) - (lp_curr - lq_back);
                if (std::isfinite(lp_new) && std::log(runif(rng)) < log_ratio2) {
                    x = x_new;
                    log_sigma_total = log_sigma_prop;
                    logit_rho = logit_rho_prop;
                    hyper_accept_cnt++;
                }
            }
            hyper_total_cnt++;
        }

        // ---- 3. Update dispersion via MH ----
        if (has_disp_num || has_disp_denom) {
            sigma_total = std::exp(log_sigma_total);
            rho = rho_of(logit_rho);

            double log_dn_prop = log_disp_num;
            double log_dd_prop = log_disp_denom;
            if (has_disp_num) log_dn_prop += disp_scale * rnorm(rng);
            if (has_disp_denom) log_dd_prop += disp_scale * rnorm(rng);

            double dn_curr = has_disp_num ? std::exp(log_disp_num) : 1.0;
            double dd_curr = has_disp_denom ? std::exp(log_disp_denom) : 1.0;
            double dn_prop = has_disp_num ? std::exp(log_dn_prop) : 1.0;
            double dd_prop = has_disp_denom ? std::exp(log_dd_prop) : 1.0;

            double ll_curr = bym2_log_target(x, sigma_total, rho, dn_curr, dd_curr, lay, d);
            double ll_prop = bym2_log_target(x, sigma_total, rho, dn_prop, dd_prop, lay, d);

            double lpr_curr = 0.0, lpr_prop = 0.0;
            if (has_disp_num) {
                lpr_curr += std::log(dn_curr) - 0.5 * dn_curr;
                lpr_prop += std::log(dn_prop) - 0.5 * dn_prop;
            }
            if (has_disp_denom) {
                lpr_curr += std::log(dd_curr) - 0.5 * dd_curr;
                lpr_prop += std::log(dd_prop) - 0.5 * dd_prop;
            }

            if (std::log(runif(rng)) < (ll_prop + lpr_prop) - (ll_curr + lpr_curr)) {
                log_disp_num = log_dn_prop;
                log_disp_denom = log_dd_prop;
                disp_accept_cnt++;
            }
            disp_total_cnt++;
        }

        // ---- Adaptation during warmup ----
        // The two-dimensional random walk aims at an acceptance near 0.3
        if (iter < n_warmup && iter > 0 && iter % 50 == 0) {
            auto adapt = [&](double& scale, int& acc, int& tot, double target) {
                if (tot == 0) return;
                double rate = (double)acc / tot;
                if (rate > target + 0.05) scale *= 1.2;
                else if (rate < target - 0.05) scale *= 0.8;
                scale = std::max(0.001, std::min(scale, 5.0));
                acc = 0; tot = 0;
            };
            adapt(hyper_scale, hyper_accept_cnt, hyper_total_cnt, 0.3);
            if (has_disp_num || has_disp_denom)
                adapt(disp_scale, disp_accept_cnt, disp_total_cnt, 0.44);
        }

        // ---- Store draws ----
        if (iter >= n_warmup && (iter - n_warmup) % thin == 0) {
            sigma_total = std::exp(log_sigma_total);
            rho = rho_of(logit_rho);

            int row = save_idx * n_params;
            int col = 0;
            for (int j = 0; j < p_num + p_denom; j++) result.draws_flat[row + col++] = x[j];
            if (has_disp_num) result.draws_flat[row + col++] = log_disp_num;
            if (has_disp_denom) result.draws_flat[row + col++] = log_disp_denom;
            result.draws_flat[row + col++] = log_sigma_total;
            result.draws_flat[row + col++] = logit_rho;

            // Store total spatial effect per site
            for (int s = 0; s < S; s++) {
                result.phi_draws_flat[save_idx * S + s] =
                    bym2_spatial_effect(s, x.data() + lay.phi0(), x.data() + lay.theta0(),
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

    // Acceptance rates: the field moves as one block, so every site reports
    // the block's rate
    const double field_rate = field_total_cnt > 0 ?
        (double)field_accept_cnt / field_total_cnt : 0.0;
    for (int s = 0; s < S; s++) result.accept_phi[s] = field_rate;
    result.accept_beta = field_rate;
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
