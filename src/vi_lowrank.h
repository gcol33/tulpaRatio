// vi_lowrank.h
// Low-rank + diagonal Gaussian variational inference
// q(θ) = N(θ; μ, LL' + D) where L is D×r and D = diag(d²)

#ifndef RATIOD_VI_LOWRANK_H
#define RATIOD_VI_LOWRANK_H

#include "vi_types.h"
#include <tulpa/vi_math.h>
#include "hmc_sampler.h"
#include <random>
#include <cmath>

namespace ratiod {
namespace vi {

// Forward declaration - implemented in vi_sampler.cpp
double compute_log_joint_grad(
    const Eigen::VectorXd& params,
    const ratiod_hmc::ModelData& data,
    const ratiod_hmc::ParamLayout& layout,
    Eigen::VectorXd& grad
);

// ---------------------------------------------------------------------
// Low-Rank ELBO and Gradient
// ---------------------------------------------------------------------

struct LowRankGradients {
  Eigen::VectorXd grad_mu;
  Eigen::MatrixXd grad_L;
  Eigen::VectorXd grad_log_d;
  double elbo;
};

// Compute ELBO and gradients for low-rank VI
// Uses reparameterization: θ = μ + L η + d ⊙ ε
// where η ~ N(0, Iᵣ), ε ~ N(0, I_D)
inline LowRankGradients compute_lowrank_elbo_grad(
    const LowRankParams& params,
    const ratiod_hmc::ModelData& data,
    const ratiod_hmc::ParamLayout& layout,
    int mc_samples,
    std::mt19937& rng
) {
  int D = params.dim();
  int r = params.rank();
  LowRankGradients result;
  result.grad_mu = Eigen::VectorXd::Zero(D);
  result.grad_L = Eigen::MatrixXd::Zero(D, r);
  result.grad_log_d = Eigen::VectorXd::Zero(D);
  result.elbo = 0.0;

  std::normal_distribution<double> N01(0.0, 1.0);
  Eigen::VectorXd d = exp(params.log_d.array());
  const tulpa::vi::LowRankPrecision precision(d,params.L);

  for (int m = 0; m < mc_samples; ++m) {
    // Sample η ~ N(0, Iᵣ) for low-rank part
    Eigen::VectorXd eta(r);
    for (int i = 0; i < r; ++i) {
      eta(i) = N01(rng);
    }

    // Sample ε ~ N(0, I_D) for diagonal part
    Eigen::VectorXd eps(D);
    for (int i = 0; i < D; ++i) {
      eps(i) = N01(rng);
    }

    // Reparameterize: θ = μ + L η + d ⊙ ε
    Eigen::VectorXd theta = params.mu + params.L * eta + d.asDiagonal() * eps;

    // Gradient of log p(y, θ) w.r.t. θ
    Eigen::VectorXd grad_theta;
    double log_joint = compute_log_joint_grad(theta, data, layout, grad_theta);
    tulpa::vi::path_gradient(grad_theta,precision.apply(theta-params.mu));

    // Derivative through the variational mean.
    result.grad_mu += grad_theta;

    // Chain rule for L: ∂θ/∂L = η' (outer product with grad_theta)
    result.grad_L.noalias() += grad_theta * eta.transpose();

    // Chain rule for log_d: ∂θ/∂log_dᵢ = εᵢ · dᵢ
    result.grad_log_d += (grad_theta.array() * eps.array() * d.array()).matrix();

    result.elbo += log_joint;
  }

  // Average over MC samples
  result.grad_mu /= mc_samples;
  result.grad_L /= mc_samples;
  result.grad_log_d /= mc_samples;
  result.elbo /= mc_samples;

  // Add entropy to ELBO
  result.elbo += params.entropy();

  return result;
}

// ---------------------------------------------------------------------
// Low-Rank VI Fitting
// ---------------------------------------------------------------------

inline VIResult fit_lowrank(
    const ratiod_hmc::ModelData& data,
    const ratiod_hmc::ParamLayout& layout,
    int D,
    const VIConfig& config,
    const Eigen::VectorXd* init_mu = nullptr
) {
  // Determine rank
  int r = select_rank(D, config);

  if (config.verbose) {
    Rcpp::Rcout << "Low-rank VI with D=" << D << ", rank=" << r << "\n";
  }

  // Initialize parameters
  LowRankParams params(D, r);
  const Eigen::MatrixXd convergence_reference=params.L;

  if (init_mu != nullptr && init_mu->size() == D) {
    params.mu = *init_mu;
  }

  // Number of variational parameters: D (mu) + D*r (L) + D (log_d)
  int n_var_params = D + D * r + D;

  // Initialize optimizer
  tulpa::vi::AdamOptimizer optimizer(config.adam_alpha, config.adam_beta1,
                          config.adam_beta2, config.adam_eps);
  tulpa::vi::AdamState state = optimizer.init_state(n_var_params);

  // Convergence checker
  tulpa::vi::ConvergenceChecker checker(config.tol_grad, config.tol_rel_elbo, config.patience);

  // Random number generator
  std::mt19937 rng(config.seed);

  // Optimization loop
  VIResult result;
  result.variant_used = VIVariant::LOWRANK;
  result.elbo_history.reserve(config.max_iter);

  for (int iter = 0; iter < config.max_iter; ++iter) {
    // Compute ELBO and gradients
    LowRankGradients grads = compute_lowrank_elbo_grad(
        params, data, layout, config.mc_samples, rng);

    result.elbo_history.push_back(grads.elbo);

    // Flatten gradients: [grad_mu, grad_L (column-major), grad_log_d]
    Eigen::VectorXd grad_flat(n_var_params);
    grad_flat.head(D) = grads.grad_mu;

    // Flatten L column by column
    Eigen::Map<Eigen::VectorXd>(grad_flat.data() + D, D * r) =
        Eigen::Map<const Eigen::VectorXd>(grads.grad_L.data(), D * r);

    grad_flat.tail(D) = grads.grad_log_d;

    // Gradient norm for convergence check
    double grad_norm = grad_flat.norm();

    // Check convergence
    Eigen::VectorXd convergence_gradient=grad_flat;
    const Eigen::MatrixXd aligned=tulpa::vi::align_factor_gradient(params.L,grads.grad_L,convergence_reference);
    Eigen::Map<Eigen::VectorXd>(convergence_gradient.data()+D,D*r)=
      Eigen::Map<const Eigen::VectorXd>(aligned.data(),D*r);
    std::string converged = checker.check(grads.elbo, convergence_gradient);
    if (!converged.empty()) {
      result.converged = true;
      result.iterations = iter + 1;
      if (config.verbose) {
        Rcpp::Rcout << "Converged at iteration " << iter + 1
                    << " (" << converged << ")\n";
      }
      break;
    }

    // Adam update
    Eigen::VectorXd params_flat = params.flatten();
    params_flat = optimizer.step_clipped(params_flat, grad_flat, state);
    params.unflatten(params_flat);

    // Progress reporting
    if (config.verbose && (iter + 1) % config.print_every == 0) {
      Rcpp::Rcout << "Iter " << iter + 1 << ": ELBO = " << grads.elbo
                  << ", |grad| = " << grad_norm << "\n";
    }

    // Check for user interrupt
    if ((iter + 1) % 100 == 0) {
      Rcpp::checkUserInterrupt();
    }

    result.iterations = iter + 1;
  }

  // Store final parameters
  result.mu = params.mu;
  result.Sigma = params.covariance();
  result.L_factor = params.L;
  result.d_diag = exp(params.log_d.array());
  result.rank_used = r;
  result.final_elbo = result.elbo_history.back();

  // Generate posterior samples for diagnostics
  int n_samples = 1000;
  result.samples.resize(n_samples, D);
  for (int s = 0; s < n_samples; ++s) {
    result.samples.row(s) = params.sample(rng);
  }

  return result;
}

} // namespace vi
} // namespace ratiod

#endif // RATIOD_VI_LOWRANK_H
