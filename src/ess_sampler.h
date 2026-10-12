// ess_sampler.h
// Elliptical Slice Sampling for models with Gaussian priors
//
// ESS is efficient for sampling parameters with Gaussian priors because it:
// 1. Uses the prior as the proposal distribution
// 2. Automatically adapts to the posterior geometry
// 3. Has no tuning parameters (unlike HMC step size)
//
// Reference: Murray, Adams, MacKay (2010) "Elliptical Slice Sampling"
// https://proceedings.mlr.press/v9/murray10a.html

#ifndef RATIOD_ESS_SAMPLER_H
#define RATIOD_ESS_SAMPLER_H

#include <Rcpp.h>
#include <RcppEigen.h>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include "hmc_sampler.h"

namespace ratiod_ess {

using ratiod_hmc::ModelData;
using ratiod_hmc::ParamLayout;
using ratiod_hmc::ModelType;

// ============================================================================
// ESS Configuration
// ============================================================================

struct ESSConfig {
    int n_iter;                 // Total iterations
    int n_warmup;               // Warmup iterations (for adaptation)
    int n_thin;                 // Thinning interval
    bool verbose;               // Print progress
    int print_every;            // Print every N iterations
    unsigned int seed;          // Random seed

    // ESS-specific options
    bool use_cholesky;          // Use Cholesky decomposition for multivariate normal
    bool adapt_during_warmup;   // Adapt covariance during warmup
    int adapt_interval;         // How often to update covariance estimate

    ESSConfig()
        : n_iter(2000), n_warmup(1000), n_thin(1), verbose(true),
          print_every(100), seed(12345), use_cholesky(true),
          adapt_during_warmup(true), adapt_interval(100) {}
};

// ============================================================================
// ESS Result
// ============================================================================

struct ESSResult {
    Eigen::MatrixXd samples;       // (n_save x n_params) posterior samples
    std::vector<double> log_lik;   // Log-likelihood at each sample
    int n_slice_evals;             // Total likelihood evaluations
    double avg_slice_evals;        // Average evaluations per ESS step
    bool success;
    std::string error_msg;
};

// ============================================================================
// Gaussian Prior specification
// ============================================================================

// The reference of one ESS block over the parameters in param_indices:
// N(mean, Sigma), or a multivariate t with `dof` degrees of freedom and
// scale matrix Sigma when dof > 0. Sigma is scale^2 I, diag(sd^2), L L^T,
// or (R R^T)^{-1} given the Cholesky factor R of the precision.
struct GaussianPrior {
    std::vector<int> param_indices;  // Indices of parameters in the block
    Eigen::VectorXd mean;            // Reference mean
    Eigen::MatrixXd chol_cov;        // Dense case: lower Cholesky factor L
    Eigen::MatrixXd chol_prec;       // Precision case: lower Cholesky factor R
    Eigen::VectorXd sd;              // Diagonal case: standard deviations
    bool is_identity_cov;            // Sigma = scale^2 I
    bool is_diag;                    // Sigma = diag(sd^2)
    bool is_precision;               // Sigma^{-1} = R R^T
    double scale;
    double dof;                      // 0: Gaussian reference

    GaussianPrior()
        : is_identity_cov(true), is_diag(false), is_precision(false),
          scale(1.0), dof(0.0) {}
};

// ============================================================================
// Helper: Sample from standard normal
// ============================================================================

inline Eigen::VectorXd sample_std_normal(int n, std::mt19937& rng) {
    std::normal_distribution<double> std_normal(0.0, 1.0);
    Eigen::VectorXd z(n);
    for (int i = 0; i < n; i++) {
        z(i) = std_normal(rng);
    }
    return z;
}

// ============================================================================
// Reference: draw and log-density
// ============================================================================

// Sigma^{1/2} z for z ~ N(0, I): a zero-mean draw from N(0, Sigma).
inline Eigen::VectorXd reference_draw(const GaussianPrior& prior,
                                      const Eigen::VectorXd& z) {
    if (prior.is_identity_cov) return prior.scale * z;
    if (prior.is_diag) return prior.sd.cwiseProduct(z);
    if (prior.is_precision) {
        return prior.chol_prec.transpose().triangularView<Eigen::Upper>().solve(z);
    }
    return prior.chol_cov * z;
}

// (x - mean)^T Sigma^{-1} (x - mean)
inline double reference_mahalanobis_sq(const GaussianPrior& prior,
                                       const Eigen::VectorXd& x) {
    const Eigen::VectorXd d = x - prior.mean;
    if (prior.is_identity_cov) return d.squaredNorm() / (prior.scale * prior.scale);
    if (prior.is_diag) return d.cwiseQuotient(prior.sd).squaredNorm();
    if (prior.is_precision) return (prior.chol_prec.transpose() * d).squaredNorm();
    return prior.chol_cov.triangularView<Eigen::Lower>().solve(d).squaredNorm();
}

// Log-density of the reference up to its constant.
inline double reference_log_density(const GaussianPrior& prior,
                                    const Eigen::VectorXd& x) {
    const double q = reference_mahalanobis_sq(prior, x);
    if (prior.dof <= 0.0) return -0.5 * q;
    const double d = static_cast<double>(x.size());
    return -0.5 * (prior.dof + d) * std::log1p(q / prior.dof);
}

// A t reference is the scale mixture N(mean, s Sigma), s ~ InvGamma(nu/2,
// nu/2). Generalised ESS (Nishihara, Murray and Adams 2014) draws s from its
// conditional given the current state, InvGamma((nu + d)/2, (nu + q)/2),
// then takes the ellipse from N(mean, s Sigma); the slice function, the
// target over the t density, does not depend on s. Returns sqrt(s), 1 for a
// Gaussian reference.
inline double reference_ellipse_scale(const GaussianPrior& prior,
                                      const Eigen::VectorXd& f,
                                      std::mt19937& rng) {
    if (prior.dof <= 0.0) return 1.0;
    const double d = static_cast<double>(f.size());
    const double q = reference_mahalanobis_sq(prior, f);
    std::gamma_distribution<double> g(0.5 * (prior.dof + d), 1.0);
    return std::sqrt(0.5 * (prior.dof + q) / g(rng));
}

// Degrees of freedom of the adapted reference. Heavy tails mean a state far
// from the fitted mean draws a wide ellipse, so a reference fitted to warmup
// draws that had not yet spread along a ridge does not hold the chain
// inside the region it was fitted to; four keeps its variance finite.
constexpr double ESS_REFERENCE_DOF = 4.0;

// Blocks up to this size get a dense adapted covariance; larger ones a
// diagonal one, which keeps the O(d^2) reference evaluation and the O(d^3)
// factorisation small beside the likelihood.
constexpr int ESS_DENSE_REFERENCE_MAX = 200;

// Running mean and covariance of a block over warmup draws (Welford).
struct ReferenceWelford {
    int n = 0;
    Eigen::VectorXd mean;
    Eigen::MatrixXd m2;

    void reset(int d) {
        n = 0;
        mean = Eigen::VectorXd::Zero(d);
        m2 = Eigen::MatrixXd::Zero(d, d);
    }
    void add(const Eigen::VectorXd& x) {
        n++;
        const Eigen::VectorXd delta = x - mean;
        mean += delta / n;
        m2.noalias() += delta * (x - mean).transpose();
    }
};

// The reference fitted to the warmup draws: their mean, and their covariance
// shrunk toward a small diagonal as the HMC mass matrix is
// (shrink_warmup_cov).
inline GaussianPrior adapted_reference(const std::vector<int>& indices,
                                       const ReferenceWelford& w) {
    const int d = static_cast<int>(indices.size());
    GaussianPrior prior;
    prior.param_indices = indices;
    prior.mean = w.mean;
    prior.is_identity_cov = false;
    prior.dof = ESS_REFERENCE_DOF;
    Eigen::MatrixXd cov = w.m2 / (w.n - 1.0);
    for (int i = 0; i < d; i++) {
        for (int j = 0; j < d; j++) {
            cov(i, j) = ratiod_hmc::shrink_warmup_cov(cov(i, j), w.n, i == j);
        }
    }
    if (d <= ESS_DENSE_REFERENCE_MAX) {
        Eigen::LLT<Eigen::MatrixXd> llt(cov);
        if (llt.info() == Eigen::Success) {
            prior.chol_cov = llt.matrixL();
            return prior;
        }
    }
    prior.is_diag = true;
    prior.sd = cov.diagonal().cwiseSqrt();
    return prior;
}

// ============================================================================
// Laplace reference
// ============================================================================

double compute_log_post_double(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout
);

// Gradient of the log-posterior restricted to the parameters in idx.
inline Eigen::VectorXd block_gradient(const std::vector<double>& params,
                                      const std::vector<int>& idx,
                                      const ModelData& data,
                                      const ParamLayout& layout) {
    std::vector<double> grad(params.size(), 0.0);
    ratiod_hmc::compute_gradient(params, data, layout, grad, nullptr);
    Eigen::VectorXd g(idx.size());
    for (size_t i = 0; i < idx.size(); i++) g(i) = grad[idx[i]];
    return g;
}

// Negative Hessian of the log-posterior over the parameters in idx, by
// central differences of the gradient, symmetrised.
inline Eigen::MatrixXd block_neg_hessian(std::vector<double> params,
                                         const std::vector<int>& idx,
                                         const ModelData& data,
                                         const ParamLayout& layout) {
    const int d = static_cast<int>(idx.size());
    Eigen::MatrixXd H(d, d);
    for (int j = 0; j < d; j++) {
        const double x0 = params[idx[j]];
        const double h = 1e-4 * std::max(1.0, std::abs(x0));
        params[idx[j]] = x0 + h;
        const Eigen::VectorXd g_plus = block_gradient(params, idx, data, layout);
        params[idx[j]] = x0 - h;
        const Eigen::VectorXd g_minus = block_gradient(params, idx, data, layout);
        params[idx[j]] = x0;
        H.col(j) = (g_minus - g_plus) / (2.0 * h);
    }
    return 0.5 * (H + H.transpose());
}

// Laplace approximation of the block idx with every other parameter held at
// its value in params: damped Newton to the conditional mode, then the
// curvature there as the precision. Given the variance parameters the
// random effects and coefficients of these models have a log-concave
// conditional posterior, so the mode exists. Its covariance carries the
// directions the data leave to the prior (a level shared by an intercept
// and every random effect) from the start, before any draw has moved along
// them. Returns false, leaving `out` untouched, when the block is too large
// for a dense factor or the curvature is not positive definite.
inline bool laplace_reference(std::vector<double> params,
                              const std::vector<int>& idx,
                              const ModelData& data,
                              const ParamLayout& layout,
                              GaussianPrior& out) {
    const int d = static_cast<int>(idx.size());
    if (d == 0 || d > ESS_DENSE_REFERENCE_MAX) return false;
    double lp = compute_log_post_double(params, data, layout);
    if (!std::isfinite(lp)) return false;
    const Eigen::MatrixXd I = Eigen::MatrixXd::Identity(d, d);

    for (int it = 0; it < 50; it++) {
        const Eigen::VectorXd g = block_gradient(params, idx, data, layout);
        const Eigen::MatrixXd A = block_neg_hessian(params, idx, data, layout);
        if (!g.allFinite() || !A.allFinite()) return false;
        Eigen::LLT<Eigen::MatrixXd> llt(A);
        double damp = 1e-6 * std::max(1.0, A.diagonal().cwiseAbs().maxCoeff());
        while (llt.info() != Eigen::Success) {
            if (damp > 1e12) return false;
            llt.compute(A + damp * I);
            damp *= 10.0;
        }
        const Eigen::VectorXd step = llt.solve(g);
        std::vector<double> trial = params;
        double t = 1.0;
        bool moved = false;
        for (int k = 0; k < 30 && !moved; k++, t *= 0.5) {
            for (int i = 0; i < d; i++) trial[idx[i]] = params[idx[i]] + t * step(i);
            const double lp_trial = compute_log_post_double(trial, data, layout);
            if (std::isfinite(lp_trial) && lp_trial >= lp) {
                params = trial;
                lp = lp_trial;
                moved = true;
            }
        }
        if (!moved || t * step.cwiseAbs().maxCoeff() < 1e-8) break;
    }

    Eigen::LLT<Eigen::MatrixXd> llt(block_neg_hessian(params, idx, data, layout));
    if (llt.info() != Eigen::Success) return false;
    out = GaussianPrior();
    out.param_indices = idx;
    out.mean.resize(d);
    for (int i = 0; i < d; i++) out.mean(i) = params[idx[i]];
    out.is_identity_cov = false;
    out.is_precision = true;
    out.chol_prec = llt.matrixL();
    out.dof = ESS_REFERENCE_DOF;
    return true;
}

// ============================================================================
// Core ESS step for a single block of parameters
// ============================================================================

// Performs one ESS step for parameters with a Gaussian prior
// Returns new parameter values for the block
//
// Algorithm:
// 1. Draw nu ~ N(0, s Sigma) from the reference, sqrt(s) = nu_scale
// 2. Set threshold: log_y = log_lik(f) + log(u), u ~ Uniform(0,1)
// 3. Draw initial angle: theta ~ Uniform(0, 2*pi)
// 4. Set bracket: [theta_min, theta_max] = [theta - 2*pi, theta]
// 5. Loop:
//    a. f' = f * cos(theta) + nu * sin(theta)
//    b. If log_lik(f') > log_y: accept and return f'
//    c. Else shrink bracket toward 0 and draw new theta
//
template<typename LogLikFn>
Eigen::VectorXd ess_step(
    const Eigen::VectorXd& f,           // Current parameter values (block)
    const GaussianPrior& prior,         // Gaussian prior for this block
    LogLikFn log_lik_fn,                // Function: f -> log_likelihood
    double current_log_lik,             // Current log-likelihood
    double nu_scale,                    // sqrt(s), see reference_ellipse_scale()
    std::mt19937& rng,
    int& n_evals                        // Output: number of likelihood evaluations
) {
    std::uniform_real_distribution<double> unif(0.0, 1.0);

    // Step 1: Draw nu from the reference (centered at 0)
    int n = f.size();
    const Eigen::VectorXd nu = nu_scale * reference_draw(prior, sample_std_normal(n, rng));

    // Step 2: Set threshold
    double log_u = std::log(unif(rng));
    double log_y = current_log_lik + log_u;

    // Step 3: Initial angle
    double theta = unif(rng) * 2.0 * M_PI;
    double theta_min = theta - 2.0 * M_PI;
    double theta_max = theta;

    // Center f around prior mean for correct geometry
    Eigen::VectorXd f_centered = f - prior.mean;

    n_evals = 0;
    const int max_shrinks = 1000;  // Prevent infinite loop

    for (int shrink = 0; shrink < max_shrinks; shrink++) {
        // Step 5a: Proposal on ellipse
        double cos_theta = std::cos(theta);
        double sin_theta = std::sin(theta);
        Eigen::VectorXd f_prime_centered = f_centered * cos_theta + nu * sin_theta;
        Eigen::VectorXd f_prime = f_prime_centered + prior.mean;

        // Step 5b: Evaluate log-likelihood
        double log_lik_prime = log_lik_fn(f_prime);
        n_evals++;

        if (log_lik_prime > log_y) {
            // Accept
            return f_prime;
        }

        // Step 5c: Shrink bracket
        if (theta < 0) {
            theta_min = theta;
        } else {
            theta_max = theta;
        }

        // Draw new theta from shrunk bracket
        theta = unif(rng) * (theta_max - theta_min) + theta_min;
    }

    // If we get here, something went wrong - return original
    Rcpp::warning("ESS reached max shrinks without accepting");
    return f;
}

// ============================================================================
// Build Gaussian priors from model structure
// ============================================================================

// Extract which parameters have Gaussian priors from the model
inline std::vector<GaussianPrior> build_gaussian_priors(
    const ModelData& data,
    const ParamLayout& layout,
    int n_params
) {
    std::vector<GaussianPrior> priors;

    // Fixed effects: beta ~ N(0, sigma_beta^2)
    // These have Gaussian priors but we sample them with ESS together
    {
        GaussianPrior prior;
        // Numerator betas
        for (int j = layout.beta_num_start; j < layout.beta_num_end; j++) {
            prior.param_indices.push_back(j);
        }
        // Denominator betas
        for (int j = layout.beta_denom_start; j < layout.beta_denom_end; j++) {
            prior.param_indices.push_back(j);
        }

        if (!prior.param_indices.empty()) {
            int n = prior.param_indices.size();
            prior.mean = Eigen::VectorXd::Zero(n);
            prior.is_identity_cov = true;
            prior.scale = data.sigma_beta;
            priors.push_back(prior);
        }
    }

    // Random effects: re ~ N(0, sigma_re^2)
    // Note: sigma_re is estimated, so we handle this specially
    if (layout.has_re && layout.re_end > layout.re_start) {
        GaussianPrior prior;
        for (int j = layout.re_start; j < layout.re_end; j++) {
            prior.param_indices.push_back(j);
        }

        int n = prior.param_indices.size();
        prior.mean = Eigen::VectorXd::Zero(n);
        prior.is_identity_cov = true;
        prior.scale = 1.0;  // Will be scaled by sigma_re during sampling
        priors.push_back(prior);
    }

    // Spatial effects (ICAR): phi ~ N(0, tau^-1 * Q^-)
    // For ICAR, the precision matrix is the graph Laplacian
    // For now, we treat these as standard normals (simplified)
    if (layout.has_spatial && layout.spatial_end > layout.spatial_start) {
        GaussianPrior prior;
        for (int j = layout.spatial_start; j < layout.spatial_end; j++) {
            prior.param_indices.push_back(j);
        }

        int n = prior.param_indices.size();
        prior.mean = Eigen::VectorXd::Zero(n);
        prior.is_identity_cov = true;  // Simplified - proper ICAR would use precision
        prior.scale = 1.0;
        priors.push_back(prior);
    }

    // Temporal effects: phi_temporal ~ N(0, tau_temporal^-1 * Q)
    if (layout.has_temporal && layout.temporal_end > layout.temporal_start) {
        GaussianPrior prior;
        for (int j = layout.temporal_start; j < layout.temporal_end; j++) {
            prior.param_indices.push_back(j);
        }

        int n = prior.param_indices.size();
        prior.mean = Eigen::VectorXd::Zero(n);
        prior.is_identity_cov = true;  // Simplified
        prior.scale = 1.0;
        priors.push_back(prior);
    }

    // GP spatial effects
    if (layout.is_gp && layout.gp_w_end > layout.gp_w_start) {
        GaussianPrior prior;
        for (int j = layout.gp_w_start; j < layout.gp_w_end; j++) {
            prior.param_indices.push_back(j);
        }

        int n = prior.param_indices.size();
        prior.mean = Eigen::VectorXd::Zero(n);
        prior.is_identity_cov = true;
        prior.scale = 1.0;  // GP effects use kernel covariance
        priors.push_back(prior);
    }

    // ZI coefficients: beta_zi ~ N(0, zi_prior_sd^2)
    if (layout.has_zi && layout.beta_zi_end > layout.beta_zi_start) {
        GaussianPrior prior;
        for (int j = layout.beta_zi_start; j < layout.beta_zi_end; j++) {
            prior.param_indices.push_back(j);
        }

        int n = prior.param_indices.size();
        prior.mean = Eigen::VectorXd::Zero(n);
        prior.is_identity_cov = true;
        prior.scale = data.zi_prior_sd;
        priors.push_back(prior);
    }

    // OI coefficients: beta_oi ~ N(0, oi_prior_sd^2)
    if (layout.has_oi && layout.beta_oi_end > layout.beta_oi_start) {
        GaussianPrior prior;
        for (int j = layout.beta_oi_start; j < layout.beta_oi_end; j++) {
            prior.param_indices.push_back(j);
        }

        int n = prior.param_indices.size();
        prior.mean = Eigen::VectorXd::Zero(n);
        prior.is_identity_cov = true;
        prior.scale = data.oi_prior_sd;
        priors.push_back(prior);
    }

    // Latent factors
    if (layout.has_latent && layout.latent_factor_end > layout.latent_factor_start) {
        GaussianPrior prior;
        for (int j = layout.latent_factor_start; j < layout.latent_factor_end; j++) {
            prior.param_indices.push_back(j);
        }

        int n = prior.param_indices.size();
        prior.mean = Eigen::VectorXd::Zero(n);
        prior.is_identity_cov = true;
        prior.scale = 1.0;
        priors.push_back(prior);
    }

    return priors;
}

// ============================================================================
// Identify non-Gaussian parameters (sampled with MH or other methods)
// ============================================================================

inline std::vector<int> get_non_gaussian_params(
    const ParamLayout& layout,
    int n_params
) {
    std::vector<int> non_gaussian;

    // Variance parameters (log scale) - use random walk MH
    if (layout.has_re && layout.log_sigma_re_idx >= 0) {
        non_gaussian.push_back(layout.log_sigma_re_idx);
    }

    if (layout.has_phi_num && layout.log_phi_num_idx >= 0) {
        non_gaussian.push_back(layout.log_phi_num_idx);
    }

    if (layout.has_phi_denom && layout.log_phi_denom_idx >= 0) {
        non_gaussian.push_back(layout.log_phi_denom_idx);
    }

    if (layout.has_spatial && layout.log_tau_spatial_idx >= 0) {
        non_gaussian.push_back(layout.log_tau_spatial_idx);
    }

    if (layout.is_bym2) {
        if (layout.log_sigma_bym2_idx >= 0) {
            non_gaussian.push_back(layout.log_sigma_bym2_idx);
        }
        if (layout.logit_rho_bym2_idx >= 0) {
            non_gaussian.push_back(layout.logit_rho_bym2_idx);
        }
    }

    if (layout.has_temporal && layout.log_tau_temporal_idx >= 0) {
        non_gaussian.push_back(layout.log_tau_temporal_idx);
    }

    if (layout.is_ar1 && layout.logit_rho_ar1_idx >= 0) {
        non_gaussian.push_back(layout.logit_rho_ar1_idx);
    }

    if (layout.is_gp) {
        if (layout.log_sigma2_gp_idx >= 0) {
            non_gaussian.push_back(layout.log_sigma2_gp_idx);
        }
        if (layout.log_phi_gp_idx >= 0) {
            non_gaussian.push_back(layout.log_phi_gp_idx);
        }
    }

    return non_gaussian;
}

// ============================================================================
// Random walk Metropolis-Hastings step for non-Gaussian parameters
// ============================================================================

template<typename LogPostFn>
double rwmh_step(
    std::vector<double>& params,
    int idx,
    double proposal_sd,
    LogPostFn log_post_fn,
    double current_log_post,
    std::mt19937& rng,
    bool& accepted
) {
    std::normal_distribution<double> normal(0.0, proposal_sd);
    std::uniform_real_distribution<double> unif(0.0, 1.0);

    double old_val = params[idx];
    double proposal = old_val + normal(rng);

    params[idx] = proposal;
    double proposed_log_post = log_post_fn(params);

    double log_alpha = proposed_log_post - current_log_post;

    if (std::log(unif(rng)) < log_alpha) {
        accepted = true;
        return proposed_log_post;
    } else {
        params[idx] = old_val;
        accepted = false;
        return current_log_post;
    }
}

// ============================================================================
// Adaptive proposal SD for RWMH
// ============================================================================

struct AdaptiveProposal {
    std::vector<double> proposal_sds;
    std::vector<int> n_accepted;
    std::vector<int> n_total;
    double target_rate;
    int adapt_interval;

    AdaptiveProposal(int n_params, double init_sd = 0.1, double target = 0.234)
        : proposal_sds(n_params, init_sd),
          n_accepted(n_params, 0),
          n_total(n_params, 0),
          target_rate(target),
          adapt_interval(50) {}

    void record(int idx, bool accepted) {
        n_total[idx]++;
        if (accepted) n_accepted[idx]++;
    }

    void adapt(int iter) {
        if (iter % adapt_interval != 0) return;

        for (size_t i = 0; i < proposal_sds.size(); i++) {
            if (n_total[i] > 0) {
                double rate = static_cast<double>(n_accepted[i]) / n_total[i];
                // Robbins-Monro type adaptation
                double gamma = 1.0 / std::sqrt(iter + 1.0);
                double log_sd = std::log(proposal_sds[i]);
                log_sd += gamma * (rate - target_rate);
                proposal_sds[i] = std::exp(log_sd);
                // Reset counters
                n_accepted[i] = 0;
                n_total[i] = 0;
            }
        }
    }
};

// ============================================================================
// Main ESS sampler
// ============================================================================

// Forward declaration of log_post computation
double compute_log_post_double(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout
);

inline ESSResult run_ess_sampler(
    const std::vector<double>& init_params,
    const ModelData& data,
    const ParamLayout& layout,
    const ESSConfig& config
) {
    ESSResult result;
    result.success = true;

    int n_params = init_params.size();
    int n_save = (config.n_iter - config.n_warmup) / config.n_thin;

    result.samples.resize(n_save, n_params);
    result.log_lik.resize(n_save);
    result.n_slice_evals = 0;

    // Initialize RNG
    std::mt19937 rng(config.seed);

    // Build Gaussian priors for ESS blocks
    std::vector<GaussianPrior> gaussian_priors = build_gaussian_priors(data, layout, n_params);

    // Get non-Gaussian parameters for RWMH
    std::vector<int> non_gaussian = get_non_gaussian_params(layout, n_params);

    // Adaptive proposals for RWMH parameters
    AdaptiveProposal adaptive(non_gaussian.size());

    // Current state
    std::vector<double> params = init_params;

    // Compute initial log-posterior
    double current_log_post = compute_log_post_double(params, data, layout);

    if (!std::isfinite(current_log_post)) {
        result.success = false;
        result.error_msg = "Initial log-posterior is not finite";
        return result;
    }

    // Progress
    if (config.verbose) {
        Rcpp::Rcout << "Running ESS sampler...\n";
        Rcpp::Rcout << "  Parameters: " << n_params << "\n";
        Rcpp::Rcout << "  Gaussian blocks: " << gaussian_priors.size() << "\n";
        Rcpp::Rcout << "  Non-Gaussian params: " << non_gaussian.size() << "\n";
        Rcpp::Rcout << "  Iterations: " << config.n_iter << " (warmup: " << config.n_warmup << ")\n";
    }

    int save_idx = 0;
    int total_ess_evals = 0;
    int total_ess_steps = 0;

    // The Gaussian blocks start as one block under its Laplace reference,
    // and the chain starts at that block's conditional mode.
    {
        std::vector<int> gaussian_indices;
        for (const auto& prior : gaussian_priors) {
            gaussian_indices.insert(gaussian_indices.end(), prior.param_indices.begin(),
                                    prior.param_indices.end());
        }
        GaussianPrior laplace;
        if (laplace_reference(params, gaussian_indices, data, layout, laplace)) {
            for (size_t i = 0; i < gaussian_indices.size(); i++) {
                params[gaussian_indices[i]] = laplace.mean(i);
            }
            current_log_post = compute_log_post_double(params, data, layout);
            gaussian_priors.assign(1, laplace);
        }
    }

    // Warmup fits the reference to the posterior from there. At the end of
    // each adaptation window every parameter is merged into one block whose reference is the mean
    // and covariance of that window's draws, which carries the correlation
    // between blocks (an intercept and its random effects, the effects and
    // their scale) that separate updates mix along slowly. Every coordinate
    // is unconstrained and the reference is divided out of the slice, so
    // the variance parameters can join the ellipse; their random-walk
    // updates continue alongside. Windows double in length, as Stan's
    // metric windows do, from a tenth of warmup (at least 50 iterations).
    // Each window fits only its own draws, made
    // after the previous reference moved the chain closer to the posterior;
    // the last ends before warmup does, so the reference is fixed while
    // draws are saved.
    std::vector<int> joint_indices;
    for (const auto& prior : gaussian_priors) {
        joint_indices.insert(joint_indices.end(), prior.param_indices.begin(),
                             prior.param_indices.end());
    }
    joint_indices.insert(joint_indices.end(), non_gaussian.begin(), non_gaussian.end());
    const int joint_dim = static_cast<int>(joint_indices.size());
    std::vector<int> window_ends;
    if (joint_dim > 0) {
        const int last = (9 * config.n_warmup) / 10;
        int len = std::max(50, config.n_warmup / 10);
        for (int end = len; end + 2 * len <= last; len *= 2, end += len) {
            window_ends.push_back(end);
        }
        if (last >= 25) window_ends.push_back(last);
    }
    size_t next_window = 0;
    ReferenceWelford welford;
    welford.reset(joint_dim);

    for (int iter = 0; iter < config.n_iter; iter++) {
        // Check for user interrupt
        if (iter % 100 == 0) {
            Rcpp::checkUserInterrupt();
        }

        // ----------------------------------------------------------------
        // ESS updates for Gaussian-prior blocks
        // ----------------------------------------------------------------
        for (auto& prior : gaussian_priors) {
            // Extract current values for this block
            int block_size = prior.param_indices.size();
            Eigen::VectorXd f(block_size);
            for (int i = 0; i < block_size; i++) {
                f(i) = params[prior.param_indices[i]];
            }

            // The ellipse is drawn from this block's Gaussian reference, so the
            // slice is taken on everything else: the log-posterior less the
            // reference's own log-density (up to its constant). The reference
            // need not be the model's prior for the block; dividing it out
            // leaves the posterior as the target either way.
            auto log_lik_fn = [&](const Eigen::VectorXd& f_new) -> double {
                std::vector<double> params_temp = params;
                for (int i = 0; i < block_size; i++) {
                    params_temp[prior.param_indices[i]] = f_new(i);
                }
                return compute_log_post_double(params_temp, data, layout)
                       - reference_log_density(prior, f_new);
            };

            // Perform ESS step, the threshold read at the current state
            int n_evals = 0;
            const double nu_scale = reference_ellipse_scale(prior, f, rng);
            Eigen::VectorXd f_new = ess_step(f, prior, log_lik_fn, log_lik_fn(f),
                                             nu_scale, rng, n_evals);

            // Update params
            for (int i = 0; i < block_size; i++) {
                params[prior.param_indices[i]] = f_new(i);
            }

            total_ess_evals += n_evals;
            total_ess_steps++;
        }

        // Update log-posterior after ESS updates
        current_log_post = compute_log_post_double(params, data, layout);

        // ----------------------------------------------------------------
        // RWMH updates for non-Gaussian parameters
        // ----------------------------------------------------------------
        for (size_t i = 0; i < non_gaussian.size(); i++) {
            int idx = non_gaussian[i];
            bool accepted = false;
            current_log_post = rwmh_step(
                params, idx, adaptive.proposal_sds[i],
                [&](const std::vector<double>& p) { return compute_log_post_double(p, data, layout); },
                current_log_post, rng, accepted
            );
            adaptive.record(i, accepted);
        }

        // Adapt RWMH proposals during warmup
        if (iter < config.n_warmup && config.adapt_during_warmup) {
            adaptive.adapt(iter);
        }

        // Adapt the ESS reference
        if (next_window < window_ends.size()) {
            Eigen::VectorXd x(joint_dim);
            for (int i = 0; i < joint_dim; i++) x(i) = params[joint_indices[i]];
            welford.add(x);
            if (iter + 1 == window_ends[next_window]) {
                if (welford.n >= 10) {
                    gaussian_priors.assign(1, adapted_reference(joint_indices, welford));
                }
                welford.reset(joint_dim);
                next_window++;
            }
        }

        // ----------------------------------------------------------------
        // Store sample
        // ----------------------------------------------------------------
        if (iter >= config.n_warmup && (iter - config.n_warmup) % config.n_thin == 0) {
            for (int j = 0; j < n_params; j++) {
                result.samples(save_idx, j) = params[j];
            }
            result.log_lik[save_idx] = current_log_post;
            save_idx++;
        }

        // Progress
        if (config.verbose && (iter + 1) % config.print_every == 0) {
            Rcpp::Rcout << "Iter " << (iter + 1) << "/" << config.n_iter;
            if (iter < config.n_warmup) {
                Rcpp::Rcout << " (warmup)";
            }
            Rcpp::Rcout << " log_post = " << current_log_post << "\n";
        }
    }

    result.n_slice_evals = total_ess_evals;
    result.avg_slice_evals = total_ess_steps > 0 ?
        static_cast<double>(total_ess_evals) / total_ess_steps : 0.0;

    if (config.verbose) {
        Rcpp::Rcout << "ESS complete. Avg slice evals per step: "
                    << result.avg_slice_evals << "\n";
    }

    return result;
}

} // namespace ratiod_ess

#endif // RATIOD_ESS_SAMPLER_H
