// laplace_core.cpp
// Core Laplace approximation engine for ratiod
// Implements nested Laplace approximation for latent Gaussian models

#include "laplace_core.h"
#include "linalg_fast.h"
#include "ar1_shared.h"
#include "omp_thread_scope.h"
#include "omp_sum.h"
#include <tulpa/soft_sum_to_zero.h>
#include <Rcpp.h>
#include <cmath>
#include <algorithm>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace Rcpp;

namespace ratiod {

// ---------------------------------------------------------------------
// Likelihood functions
// ---------------------------------------------------------------------

// Log-likelihood for binomial: y ~ Binomial(n, logit^{-1}(eta))
double log_lik_binomial(int y, int n, double eta) {
  // Numerically stable computation
  // log p = y * eta - n * log(1 + exp(eta)) + log(choose(n, y))
  double log_p;
  if (eta > 0) {
    log_p = y * eta - n * eta - n * std::log(1.0 + std::exp(-eta));
  } else {
    log_p = y * eta - n * std::log(1.0 + std::exp(eta));
  }
  // Omit log(choose(n,y)) as it's constant w.r.t. eta
  return log_p;
}

// Gradient: d/d(eta) log p(y | eta) = y - n * p where p = logit^{-1}(eta)
double grad_log_lik_binomial(int y, int n, double eta) {
  double p;
  if (eta > 0) {
    double exp_neg_eta = std::exp(-eta);
    p = 1.0 / (1.0 + exp_neg_eta);
  } else {
    double exp_eta = std::exp(eta);
    p = exp_eta / (1.0 + exp_eta);
  }
  return y - n * p;
}

// Negative Hessian: -d²/d(eta)² log p = n * p * (1-p)
double neg_hess_log_lik_binomial(int y, int n, double eta) {
  double p;
  if (eta > 0) {
    double exp_neg_eta = std::exp(-eta);
    p = 1.0 / (1.0 + exp_neg_eta);
  } else {
    double exp_eta = std::exp(eta);
    p = exp_eta / (1.0 + exp_eta);
  }
  return n * p * (1.0 - p);
}

// Log-likelihood for negative binomial: y ~ NegBin(mu = exp(eta), phi)
// Parameterization: E[Y] = mu, Var[Y] = mu + mu²/phi
double log_lik_negbin(int y, double eta, double phi) {
  double mu = ratiod_linalg::safe_exp(eta);
  // log p = lgamma(y + phi) - lgamma(phi) - lgamma(y+1)
  //       + phi * log(phi/(mu+phi)) + y * log(mu/(mu+phi))
  double log_p = R::lgammafn(y + phi) - R::lgammafn(phi) - R::lgammafn(y + 1.0)
               + phi * std::log(phi / (mu + phi))
               + y * std::log(mu / (mu + phi));
  return log_p;
}

// Gradient: d/d(eta) log p = y - (y + phi) * mu / (mu + phi)
//                          = y - (y + phi) * p where p = mu/(mu+phi)
double grad_log_lik_negbin(int y, double eta, double phi) {
  double mu = ratiod_linalg::safe_exp(eta);
  double p = mu / (mu + phi);
  return y - (y + phi) * p;
}

// Negative Hessian: (y + phi) * p * (1-p) * mu (chain rule factor)
// Actually: -d²/d(eta)² = (y + phi) * mu * phi / (mu + phi)²
double neg_hess_log_lik_negbin(int y, double eta, double phi) {
  double mu = ratiod_linalg::safe_exp(eta);
  double denom = mu + phi;
  return (y + phi) * mu * phi / (denom * denom);
}

// Log-likelihood for Poisson: y ~ Poisson(mu = exp(eta))
double log_lik_poisson(int y, double eta) {
  // log p = y * eta - exp(eta) - lgamma(y+1)
  return y * eta - ratiod_linalg::safe_exp(eta) - R::lgammafn(y + 1.0);
}

// Gradient: d/d(eta) log p = y - exp(eta)
double grad_log_lik_poisson(int y, double eta) {
  return y - ratiod_linalg::safe_exp(eta);
}

// Negative Hessian: exp(eta)
double neg_hess_log_lik_poisson(int y, double eta) {
  return ratiod_linalg::safe_exp(eta);
}

// ---------------------------------------------------------------------
// Helper functions for sparse matrix operations
// ---------------------------------------------------------------------

// Solve lower triangular system L x = b
void solve_lower(const std::vector<double>& L_vals,
                 const std::vector<int>& L_row,
                 const std::vector<int>& L_col_ptr,
                 const std::vector<double>& b,
                 std::vector<double>& x) {
  int n = b.size();
  x = b;  // Copy b to x

  for (int j = 0; j < n; j++) {
    x[j] /= L_vals[L_col_ptr[j]];  // Diagonal element
    for (int k = L_col_ptr[j] + 1; k < L_col_ptr[j + 1]; k++) {
      x[L_row[k]] -= L_vals[k] * x[j];
    }
  }
}

// Solve upper triangular system L' x = b
void solve_upper(const std::vector<double>& L_vals,
                 const std::vector<int>& L_row,
                 const std::vector<int>& L_col_ptr,
                 const std::vector<double>& b,
                 std::vector<double>& x) {
  int n = b.size();
  x = b;

  for (int j = n - 1; j >= 0; j--) {
    for (int k = L_col_ptr[j] + 1; k < L_col_ptr[j + 1]; k++) {
      x[j] -= L_vals[k] * x[L_row[k]];
    }
    x[j] /= L_vals[L_col_ptr[j]];
  }
}

// ---------------------------------------------------------------------
// Laplace mode finding
// ---------------------------------------------------------------------

// Simple dense implementation for now
// TODO: Add sparse implementation for large problems
LaplaceResult laplace_mode_dense(
    const IntegerVector& y,
    const IntegerVector& n,
    const NumericMatrix& X,
    const NumericVector& re_idx,
    int n_re_groups,
    double sigma_re,
    const std::string& family,
    double phi,
    int max_iter,
    double tol,
    int n_threads
) {
  int N = y.size();
  int p = X.ncol();
  int n_x = p + n_re_groups;  // Fixed effects + random effects

  LaplaceResult result;
  result.mode = NumericVector(n_x, 0.0);
  result.converged = false;
  result.n_iter = 0;

  // Initialize
  NumericVector x(n_x, 0.0);
  NumericVector grad(n_x);
  NumericMatrix H(n_x, n_x);  // Hessian (dense for now)

  double tau_re = 1.0 / (sigma_re * sigma_re + 1e-10);

  ratiod_omp::ScopedThreadCount thread_scope(n_threads);

  for (int iter = 0; iter < max_iter; iter++) {
    // Compute linear predictor (parallelized)
    NumericVector eta(N);
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int i = 0; i < N; i++) {
      eta[i] = 0.0;
      for (int j = 0; j < p; j++) {
        eta[i] += X(i, j) * x[j];
      }
      // Add random effect
      if (n_re_groups > 0) {
        int g = (int)re_idx[i] - 1;  // 0-based
        if (g >= 0 && g < n_re_groups) {
          eta[i] += x[p + g];
        }
      }
    }

    // Compute gradient and Hessian
    std::fill(grad.begin(), grad.end(), 0.0);
    for (int j = 0; j < n_x; j++) {
      for (int k = 0; k < n_x; k++) {
        H(j, k) = 0.0;
      }
    }

    // Likelihood contributions (parallelized with reduction)
    // Use thread-local storage for accumulation
    std::vector<double> grad_local(n_x, 0.0);
    std::vector<std::vector<double>> H_local(n_x, std::vector<double>(n_x, 0.0));

    #ifdef _OPENMP
    #pragma omp parallel
    {
      std::vector<double> grad_thread(n_x, 0.0);
      std::vector<std::vector<double>> H_thread(n_x, std::vector<double>(n_x, 0.0));

      #pragma omp for schedule(static)
      for (int i = 0; i < N; i++) {
        double g_i, h_i;

        if (family == "binomial") {
          g_i = grad_log_lik_binomial(y[i], n[i], eta[i]);
          h_i = neg_hess_log_lik_binomial(y[i], n[i], eta[i]);
        } else if (family == "negbin") {
          g_i = grad_log_lik_negbin(y[i], eta[i], phi);
          h_i = neg_hess_log_lik_negbin(y[i], eta[i], phi);
        } else {  // poisson
          g_i = grad_log_lik_poisson(y[i], eta[i]);
          h_i = neg_hess_log_lik_poisson(y[i], eta[i]);
        }

        // Gradient w.r.t. fixed effects
        for (int j = 0; j < p; j++) {
          grad_thread[j] += g_i * X(i, j);
          for (int k = 0; k < p; k++) {
            H_thread[j][k] += h_i * X(i, j) * X(i, k);
          }
        }

        // Gradient w.r.t. random effects
        if (n_re_groups > 0) {
          int g = (int)re_idx[i] - 1;
          if (g >= 0 && g < n_re_groups) {
            grad_thread[p + g] += g_i;
            H_thread[p + g][p + g] += h_i;

            // Cross terms
            for (int j = 0; j < p; j++) {
              H_thread[j][p + g] += h_i * X(i, j);
              H_thread[p + g][j] += h_i * X(i, j);
            }
          }
        }
      }

      // Reduce thread-local results
      #pragma omp critical
      {
        for (int j = 0; j < n_x; j++) {
          grad_local[j] += grad_thread[j];
          for (int k = 0; k < n_x; k++) {
            H_local[j][k] += H_thread[j][k];
          }
        }
      }
    }
    #else
    // Non-OpenMP fallback
    for (int i = 0; i < N; i++) {
      double g_i, h_i;

      if (family == "binomial") {
        g_i = grad_log_lik_binomial(y[i], n[i], eta[i]);
        h_i = neg_hess_log_lik_binomial(y[i], n[i], eta[i]);
      } else if (family == "negbin") {
        g_i = grad_log_lik_negbin(y[i], eta[i], phi);
        h_i = neg_hess_log_lik_negbin(y[i], eta[i], phi);
      } else {  // poisson
        g_i = grad_log_lik_poisson(y[i], eta[i]);
        h_i = neg_hess_log_lik_poisson(y[i], eta[i]);
      }

      // Gradient w.r.t. fixed effects
      for (int j = 0; j < p; j++) {
        grad_local[j] += g_i * X(i, j);
        for (int k = 0; k < p; k++) {
          H_local[j][k] += h_i * X(i, j) * X(i, k);
        }
      }

      // Gradient w.r.t. random effects
      if (n_re_groups > 0) {
        int g = (int)re_idx[i] - 1;
        if (g >= 0 && g < n_re_groups) {
          grad_local[p + g] += g_i;
          H_local[p + g][p + g] += h_i;

          // Cross terms
          for (int j = 0; j < p; j++) {
            H_local[j][p + g] += h_i * X(i, j);
            H_local[p + g][j] += h_i * X(i, j);
          }
        }
      }
    }
    #endif

    // Copy local results to output
    for (int j = 0; j < n_x; j++) {
      grad[j] = grad_local[j];
      for (int k = 0; k < n_x; k++) {
        H(j, k) = H_local[j][k];
      }
    }

    // Add prior contributions for random effects
    for (int g = 0; g < n_re_groups; g++) {
      grad[p + g] -= tau_re * x[p + g];
      H(p + g, p + g) += tau_re;
    }

    // Add weak prior for fixed effects (regularization)
    double tau_beta = 1e-4;
    for (int j = 0; j < p; j++) {
      grad[j] -= tau_beta * x[j];
      H(j, j) += tau_beta;
    }

    // Newton step: solve H * delta = grad
    // For now, use simple Cholesky (dense)

    // Cholesky factorization of H
    NumericMatrix L(n_x, n_x);
    for (int j = 0; j < n_x; j++) {
      for (int k = 0; k <= j; k++) {
        double sum = H(j, k);
        for (int i = 0; i < k; i++) {
          sum -= L(j, i) * L(k, i);
        }
        if (j == k) {
          if (sum <= 0) sum = 1e-6;  // Ensure positive definite
          L(j, k) = std::sqrt(sum);
        } else {
          L(j, k) = sum / L(k, k);
        }
      }
    }

    // Solve L * z = grad
    NumericVector z(n_x);
    for (int j = 0; j < n_x; j++) {
      double sum = grad[j];
      for (int k = 0; k < j; k++) {
        sum -= L(j, k) * z[k];
      }
      z[j] = sum / L(j, j);
    }

    // Solve L' * delta = z
    NumericVector delta(n_x);
    for (int j = n_x - 1; j >= 0; j--) {
      double sum = z[j];
      for (int k = j + 1; k < n_x; k++) {
        sum -= L(k, j) * delta[k];
      }
      delta[j] = sum / L(j, j);
    }

    // Update x with NaN/Inf checking
    double step_size = 1.0;
    double max_delta = 0.0;
    bool has_nan = false;
    for (int j = 0; j < n_x; j++) {
      if (!std::isfinite(delta[j])) {
        has_nan = true;
        break;
      }
      max_delta = std::max(max_delta, std::abs(delta[j]));
    }

    // If NaN detected, try smaller step or abort
    if (has_nan) {
      // Reset to previous iteration with smaller step
      step_size = 0.1;
      for (int j = 0; j < n_x; j++) {
        if (std::isfinite(delta[j])) {
          x[j] += step_size * delta[j];
        }
      }
      continue;  // Try next iteration
    }

    for (int j = 0; j < n_x; j++) {
      x[j] += step_size * delta[j];
    }

    // Check convergence
    if (max_delta < tol) {
      result.converged = true;
      result.n_iter = iter + 1;
      break;
    }

    result.n_iter = iter + 1;
  }

  // Store results
  result.mode = x;

  // Compute log determinant of H
  double log_det = 0.0;
  NumericMatrix L_final(n_x, n_x);

  // Recompute final Hessian at mode (parallelized)
  NumericVector eta_final(N);
  #ifdef _OPENMP
  #pragma omp parallel for schedule(static)
  #endif
  for (int i = 0; i < N; i++) {
    eta_final[i] = 0.0;
    for (int j = 0; j < p; j++) {
      eta_final[i] += X(i, j) * x[j];
    }
    if (n_re_groups > 0) {
      int g = (int)re_idx[i] - 1;
      if (g >= 0 && g < n_re_groups) {
        eta_final[i] += x[p + g];
      }
    }
  }

  for (int j = 0; j < n_x; j++) {
    for (int k = 0; k < n_x; k++) {
      H(j, k) = 0.0;
    }
  }

  // Parallelize final Hessian computation
  std::vector<std::vector<double>> H_final_local(n_x, std::vector<double>(n_x, 0.0));

  #ifdef _OPENMP
  #pragma omp parallel
  {
    std::vector<std::vector<double>> H_thread(n_x, std::vector<double>(n_x, 0.0));

    #pragma omp for schedule(static)
    for (int i = 0; i < N; i++) {
      double h_i;
      if (family == "binomial") {
        h_i = neg_hess_log_lik_binomial(y[i], n[i], eta_final[i]);
      } else if (family == "negbin") {
        h_i = neg_hess_log_lik_negbin(y[i], eta_final[i], phi);
      } else {
        h_i = neg_hess_log_lik_poisson(y[i], eta_final[i]);
      }

      for (int j = 0; j < p; j++) {
        for (int k = 0; k < p; k++) {
          H_thread[j][k] += h_i * X(i, j) * X(i, k);
        }
      }

      if (n_re_groups > 0) {
        int g = (int)re_idx[i] - 1;
        if (g >= 0 && g < n_re_groups) {
          H_thread[p + g][p + g] += h_i;
          for (int j = 0; j < p; j++) {
            H_thread[j][p + g] += h_i * X(i, j);
            H_thread[p + g][j] += h_i * X(i, j);
          }
        }
      }
    }

    #pragma omp critical
    {
      for (int j = 0; j < n_x; j++) {
        for (int k = 0; k < n_x; k++) {
          H_final_local[j][k] += H_thread[j][k];
        }
      }
    }
  }
  #else
  for (int i = 0; i < N; i++) {
    double h_i;
    if (family == "binomial") {
      h_i = neg_hess_log_lik_binomial(y[i], n[i], eta_final[i]);
    } else if (family == "negbin") {
      h_i = neg_hess_log_lik_negbin(y[i], eta_final[i], phi);
    } else {
      h_i = neg_hess_log_lik_poisson(y[i], eta_final[i]);
    }

    for (int j = 0; j < p; j++) {
      for (int k = 0; k < p; k++) {
        H_final_local[j][k] += h_i * X(i, j) * X(i, k);
      }
    }

    if (n_re_groups > 0) {
      int g = (int)re_idx[i] - 1;
      if (g >= 0 && g < n_re_groups) {
        H_final_local[p + g][p + g] += h_i;
        for (int j = 0; j < p; j++) {
          H_final_local[j][p + g] += h_i * X(i, j);
          H_final_local[p + g][j] += h_i * X(i, j);
        }
      }
    }
  }
  #endif

  // Copy to H matrix
  for (int j = 0; j < n_x; j++) {
    for (int k = 0; k < n_x; k++) {
      H(j, k) = H_final_local[j][k];
    }
  }

  // Add prior
  for (int g = 0; g < n_re_groups; g++) {
    H(p + g, p + g) += tau_re;
  }
  for (int j = 0; j < p; j++) {
    H(j, j) += 1e-4;
  }

  // Cholesky and log det
  for (int j = 0; j < n_x; j++) {
    for (int k = 0; k <= j; k++) {
      double sum = H(j, k);
      for (int i = 0; i < k; i++) {
        sum -= L_final(j, i) * L_final(k, i);
      }
      if (j == k) {
        if (sum <= 0) sum = 1e-6;
        L_final(j, k) = std::sqrt(sum);
        log_det += std::log(L_final(j, k));
      } else {
        L_final(j, k) = sum / L_final(k, k);
      }
    }
  }

  result.log_det_Q = 2.0 * log_det;  // log|H| = 2 * sum(log(diag(L)))

  // Compute log marginal likelihood approximation (parallelized)
  const double log_lik = ratiod_omp::sum_range(N, 0, [&](int i) {
    if (family == "binomial") return log_lik_binomial(y[i], n[i], eta_final[i]);
    if (family == "negbin") return log_lik_negbin(y[i], eta_final[i], phi);
    return log_lik_poisson(y[i], eta_final[i]);
  });

  // Log prior for RE
  double log_prior_re = 0.0;
  for (int g = 0; g < n_re_groups; g++) {
    log_prior_re += -0.5 * tau_re * x[p + g] * x[p + g];
  }
  log_prior_re += 0.5 * n_re_groups * std::log(tau_re / (2.0 * M_PI));

  // Laplace approximation: log p(y|theta) ≈ log p(y,x*|theta) - 0.5*log|H| + (n_x/2)*log(2π)
  result.log_marginal = log_lik + log_prior_re - 0.5 * result.log_det_Q + 0.5 * n_x * std::log(2.0 * M_PI);

  return result;
}

// ---------------------------------------------------------------------
// Laplace mode finding WITH SPATIAL (ICAR)
// ---------------------------------------------------------------------

LaplaceResult laplace_mode_spatial(
    const IntegerVector& y,
    const IntegerVector& n,
    const NumericMatrix& X,
    const NumericVector& re_idx,
    int n_re_groups,
    double sigma_re,
    const IntegerVector& spatial_idx,
    int n_spatial_units,
    const IntegerVector& adj_row_ptr,
    const IntegerVector& adj_col_idx,
    const IntegerVector& n_neighbors,
    double tau_spatial,
    const std::string& family,
    double phi,
    int max_iter,
    double tol,
    int n_threads
) {
  int N = y.size();
  int p = X.ncol();
  // Parameters: fixed effects + RE + spatial effects
  int n_x = p + n_re_groups + n_spatial_units;

  LaplaceResult result;
  result.mode = NumericVector(n_x, 0.0);
  result.converged = false;
  result.n_iter = 0;

  // Initialize
  NumericVector x(n_x, 0.0);
  NumericVector grad(n_x);
  NumericMatrix H(n_x, n_x);

  double tau_re = 1.0 / (sigma_re * sigma_re + 1e-10);

  // Indices
  int spatial_start = p + n_re_groups;

  ratiod_omp::ScopedThreadCount thread_scope(n_threads);

  for (int iter = 0; iter < max_iter; iter++) {
    // Compute linear predictor (parallelized)
    NumericVector eta(N);
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int i = 0; i < N; i++) {
      eta[i] = 0.0;
      // Fixed effects
      for (int j = 0; j < p; j++) {
        eta[i] += X(i, j) * x[j];
      }
      // Random effects
      if (n_re_groups > 0) {
        int g = (int)re_idx[i] - 1;
        if (g >= 0 && g < n_re_groups) {
          eta[i] += x[p + g];
        }
      }
      // Spatial effects
      if (n_spatial_units > 0) {
        int s = spatial_idx[i] - 1;  // 1-based to 0-based
        if (s >= 0 && s < n_spatial_units) {
          eta[i] += x[spatial_start + s];
        }
      }
    }

    // Initialize gradient and Hessian
    std::fill(grad.begin(), grad.end(), 0.0);
    for (int j = 0; j < n_x; j++) {
      for (int k = 0; k < n_x; k++) {
        H(j, k) = 0.0;
      }
    }

    // Likelihood contributions (parallelized with thread-local storage)
    std::vector<double> grad_local(n_x, 0.0);
    std::vector<std::vector<double>> H_local(n_x, std::vector<double>(n_x, 0.0));

    #ifdef _OPENMP
    #pragma omp parallel
    {
      std::vector<double> grad_thread(n_x, 0.0);
      std::vector<std::vector<double>> H_thread(n_x, std::vector<double>(n_x, 0.0));

      #pragma omp for schedule(static)
      for (int i = 0; i < N; i++) {
        double g_i, h_i;

        if (family == "binomial") {
          g_i = grad_log_lik_binomial(y[i], n[i], eta[i]);
          h_i = neg_hess_log_lik_binomial(y[i], n[i], eta[i]);
        } else if (family == "negbin") {
          g_i = grad_log_lik_negbin(y[i], eta[i], phi);
          h_i = neg_hess_log_lik_negbin(y[i], eta[i], phi);
        } else {
          g_i = grad_log_lik_poisson(y[i], eta[i]);
          h_i = neg_hess_log_lik_poisson(y[i], eta[i]);
        }

        // Fixed effects contributions
        for (int j = 0; j < p; j++) {
          grad_thread[j] += g_i * X(i, j);
          for (int k = 0; k < p; k++) {
            H_thread[j][k] += h_i * X(i, j) * X(i, k);
          }
        }

        // Random effects contributions
        if (n_re_groups > 0) {
          int g = (int)re_idx[i] - 1;
          if (g >= 0 && g < n_re_groups) {
            grad_thread[p + g] += g_i;
            H_thread[p + g][p + g] += h_i;
            for (int j = 0; j < p; j++) {
              H_thread[j][p + g] += h_i * X(i, j);
              H_thread[p + g][j] += h_i * X(i, j);
            }
          }
        }

        // Spatial effects contributions
        if (n_spatial_units > 0) {
          int s = spatial_idx[i] - 1;
          if (s >= 0 && s < n_spatial_units) {
            int sp_idx = spatial_start + s;
            grad_thread[sp_idx] += g_i;
            H_thread[sp_idx][sp_idx] += h_i;

            // Cross terms with fixed effects
            for (int j = 0; j < p; j++) {
              H_thread[j][sp_idx] += h_i * X(i, j);
              H_thread[sp_idx][j] += h_i * X(i, j);
            }

            // Cross terms with random effects
            if (n_re_groups > 0) {
              int g = (int)re_idx[i] - 1;
              if (g >= 0 && g < n_re_groups) {
                H_thread[p + g][sp_idx] += h_i;
                H_thread[sp_idx][p + g] += h_i;
              }
            }
          }
        }
      }

      // Reduce
      #pragma omp critical
      {
        for (int j = 0; j < n_x; j++) {
          grad_local[j] += grad_thread[j];
          for (int k = 0; k < n_x; k++) {
            H_local[j][k] += H_thread[j][k];
          }
        }
      }
    }
    #else
    // Non-OpenMP fallback
    for (int i = 0; i < N; i++) {
      double g_i, h_i;

      if (family == "binomial") {
        g_i = grad_log_lik_binomial(y[i], n[i], eta[i]);
        h_i = neg_hess_log_lik_binomial(y[i], n[i], eta[i]);
      } else if (family == "negbin") {
        g_i = grad_log_lik_negbin(y[i], eta[i], phi);
        h_i = neg_hess_log_lik_negbin(y[i], eta[i], phi);
      } else {
        g_i = grad_log_lik_poisson(y[i], eta[i]);
        h_i = neg_hess_log_lik_poisson(y[i], eta[i]);
      }

      for (int j = 0; j < p; j++) {
        grad_local[j] += g_i * X(i, j);
        for (int k = 0; k < p; k++) {
          H_local[j][k] += h_i * X(i, j) * X(i, k);
        }
      }

      if (n_re_groups > 0) {
        int g = (int)re_idx[i] - 1;
        if (g >= 0 && g < n_re_groups) {
          grad_local[p + g] += g_i;
          H_local[p + g][p + g] += h_i;
          for (int j = 0; j < p; j++) {
            H_local[j][p + g] += h_i * X(i, j);
            H_local[p + g][j] += h_i * X(i, j);
          }
        }
      }

      if (n_spatial_units > 0) {
        int s = spatial_idx[i] - 1;
        if (s >= 0 && s < n_spatial_units) {
          int sp_idx = spatial_start + s;
          grad_local[sp_idx] += g_i;
          H_local[sp_idx][sp_idx] += h_i;
          for (int j = 0; j < p; j++) {
            H_local[j][sp_idx] += h_i * X(i, j);
            H_local[sp_idx][j] += h_i * X(i, j);
          }
          if (n_re_groups > 0) {
            int g = (int)re_idx[i] - 1;
            if (g >= 0 && g < n_re_groups) {
              H_local[p + g][sp_idx] += h_i;
              H_local[sp_idx][p + g] += h_i;
            }
          }
        }
      }
    }
    #endif

    // Copy to output
    for (int j = 0; j < n_x; j++) {
      grad[j] = grad_local[j];
      for (int k = 0; k < n_x; k++) {
        H(j, k) = H_local[j][k];
      }
    }

    // Add prior contributions for random effects
    for (int g = 0; g < n_re_groups; g++) {
      grad[p + g] -= tau_re * x[p + g];
      H(p + g, p + g) += tau_re;
    }

    // Add ICAR prior contributions for spatial effects
    // ICAR prior: -tau/2 * sum_{i~j} (phi_i - phi_j)^2
    // Gradient: d/d(phi_i) = -tau * n_neighbors[i] * phi_i + tau * sum_{j~i} phi_j
    // Hessian diagonal: tau * n_neighbors[i]
    // Hessian off-diagonal (i~j): -tau
    for (int s = 0; s < n_spatial_units; s++) {
      int sp_idx = spatial_start + s;
      double phi_s = x[sp_idx];

      // Gradient from ICAR: -tau * (n_neighbors[s] * phi_s - sum of neighbors)
      double neighbor_sum = 0.0;
      for (int k = adj_row_ptr[s]; k < adj_row_ptr[s + 1]; k++) {
        int neighbor = adj_col_idx[k];  // Already 0-based
        neighbor_sum += x[spatial_start + neighbor];
      }
      grad[sp_idx] -= tau_spatial * (n_neighbors[s] * phi_s - neighbor_sum);

      // Hessian diagonal
      H(sp_idx, sp_idx) += tau_spatial * n_neighbors[s];

      // Hessian off-diagonal
      for (int k = adj_row_ptr[s]; k < adj_row_ptr[s + 1]; k++) {
        int neighbor = adj_col_idx[k];  // Already 0-based
        int nb_idx = spatial_start + neighbor;
        H(sp_idx, nb_idx) -= tau_spatial;
      }
    }

    // Small regularization for fixed effects
    double tau_beta = 1e-4;
    for (int j = 0; j < p; j++) {
      grad[j] -= tau_beta * x[j];
      H(j, j) += tau_beta;
    }

    // Newton step: solve H * delta = grad
    // Cholesky factorization
    NumericMatrix L(n_x, n_x);
    for (int j = 0; j < n_x; j++) {
      for (int k = 0; k <= j; k++) {
        double sum = H(j, k);
        for (int i = 0; i < k; i++) {
          sum -= L(j, i) * L(k, i);
        }
        if (j == k) {
          if (sum <= 0) sum = 1e-6;
          L(j, k) = std::sqrt(sum);
        } else {
          L(j, k) = sum / L(k, k);
        }
      }
    }

    // Solve L * z = grad
    NumericVector z(n_x);
    for (int j = 0; j < n_x; j++) {
      double sum = grad[j];
      for (int k = 0; k < j; k++) {
        sum -= L(j, k) * z[k];
      }
      z[j] = sum / L(j, j);
    }

    // Solve L' * delta = z
    NumericVector delta(n_x);
    for (int j = n_x - 1; j >= 0; j--) {
      double sum = z[j];
      for (int k = j + 1; k < n_x; k++) {
        sum -= L(k, j) * delta[k];
      }
      delta[j] = sum / L(j, j);
    }

    // Update x
    double max_delta = 0.0;
    for (int j = 0; j < n_x; j++) {
      max_delta = std::max(max_delta, std::abs(delta[j]));
      x[j] += delta[j];
    }

    // Check convergence
    if (max_delta < tol) {
      result.converged = true;
      result.n_iter = iter + 1;
      break;
    }

    result.n_iter = iter + 1;
  }

  // Center spatial effects (soft sum-to-zero constraint)
  if (n_spatial_units > 0) {
    double mean_phi = 0.0;
    for (int s = 0; s < n_spatial_units; s++) {
      mean_phi += x[spatial_start + s];
    }
    mean_phi /= n_spatial_units;
    for (int s = 0; s < n_spatial_units; s++) {
      x[spatial_start + s] -= mean_phi;
    }
  }

  result.mode = x;

  // Compute log determinant and log marginal (simplified)
  // Recompute final Hessian at mode for log determinant
  double log_det = 0.0;
  NumericMatrix L_final(n_x, n_x);

  // Recompute H at mode (simplified - just use last H)
  for (int j = 0; j < n_x; j++) {
    for (int k = 0; k <= j; k++) {
      double sum = H(j, k);
      for (int i = 0; i < k; i++) {
        sum -= L_final(j, i) * L_final(k, i);
      }
      if (j == k) {
        if (sum <= 0) sum = 1e-6;
        L_final(j, k) = std::sqrt(sum);
        log_det += std::log(L_final(j, k));
      } else {
        L_final(j, k) = sum / L_final(k, k);
      }
    }
  }

  result.log_det_Q = 2.0 * log_det;

  // Compute log marginal (simplified approximation)
  double log_lik = 0.0;
  NumericVector eta_final(N);
  #ifdef _OPENMP
  #pragma omp parallel for schedule(static)
  #endif
  for (int i = 0; i < N; i++) {
    eta_final[i] = 0.0;
    for (int j = 0; j < p; j++) {
      eta_final[i] += X(i, j) * x[j];
    }
    if (n_re_groups > 0) {
      int g = (int)re_idx[i] - 1;
      if (g >= 0 && g < n_re_groups) {
        eta_final[i] += x[p + g];
      }
    }
    if (n_spatial_units > 0) {
      int s = spatial_idx[i] - 1;
      if (s >= 0 && s < n_spatial_units) {
        eta_final[i] += x[spatial_start + s];
      }
    }
  }

  log_lik = ratiod_omp::sum_range(N, 0, [&](int i) {
    if (family == "binomial") return log_lik_binomial(y[i], n[i], eta_final[i]);
    if (family == "negbin") return log_lik_negbin(y[i], eta_final[i], phi);
    return log_lik_poisson(y[i], eta_final[i]);
  });

  // Log prior for RE
  double log_prior_re = 0.0;
  for (int g = 0; g < n_re_groups; g++) {
    log_prior_re += -0.5 * tau_re * x[p + g] * x[p + g];
  }
  if (n_re_groups > 0) {
    log_prior_re += 0.5 * n_re_groups * std::log(tau_re / (2.0 * M_PI));
  }

  // Log prior for spatial (ICAR) - simplified
  double log_prior_spatial = 0.0;
  if (n_spatial_units > 0) {
    double quad_form = 0.0;
    for (int s = 0; s < n_spatial_units; s++) {
      double phi_s = x[spatial_start + s];
      quad_form += n_neighbors[s] * phi_s * phi_s;
      for (int k = adj_row_ptr[s]; k < adj_row_ptr[s + 1]; k++) {
        int neighbor = adj_col_idx[k];  // Already 0-based
        if (neighbor > s) {  // Count each edge once
          double phi_n = x[spatial_start + neighbor];
          quad_form -= 2.0 * phi_s * phi_n;
        }
      }
    }
    log_prior_spatial = -0.5 * tau_spatial * quad_form;
    // Add normalizing constant approximation
    log_prior_spatial += 0.5 * (n_spatial_units - 1) * std::log(tau_spatial / (2.0 * M_PI));
  }

  result.log_marginal = log_lik + log_prior_re + log_prior_spatial
                        - 0.5 * result.log_det_Q + 0.5 * n_x * std::log(2.0 * M_PI);

  return result;
}

// ---------------------------------------------------------------------
// Laplace mode finding WITH SPATIAL (BYM2)
// BYM2: u = sigma * (sqrt(rho) * phi_scaled + sqrt(1-rho) * theta)
// phi_scaled ~ ICAR, theta ~ N(0, I)
// ---------------------------------------------------------------------

LaplaceResult laplace_mode_bym2(
    const IntegerVector& y,
    const IntegerVector& n,
    const NumericMatrix& X,
    const NumericVector& re_idx,
    int n_re_groups,
    double sigma_re,
    const IntegerVector& spatial_idx,
    int n_spatial_units,
    const IntegerVector& adj_row_ptr,
    const IntegerVector& adj_col_idx,
    const IntegerVector& n_neighbors,
    double sigma_spatial,    // Total spatial SD
    double rho,              // Mixing proportion (0 = all IID, 1 = all structured)
    double scale_factor,     // BYM2 scaling factor
    const std::string& family,
    double phi,
    int max_iter,
    double tol,
    int n_threads
) {
  int N = y.size();
  int p = X.ncol();
  // Parameters: fixed effects + RE + phi_scaled + theta
  int n_x = p + n_re_groups + 2 * n_spatial_units;

  LaplaceResult result;
  result.mode = NumericVector(n_x, 0.0);
  result.converged = false;
  result.n_iter = 0;

  // Initialize
  NumericVector x(n_x, 0.0);
  NumericVector grad(n_x);
  NumericMatrix H(n_x, n_x);

  double tau_re = 1.0 / (sigma_re * sigma_re + 1e-10);

  // Indices for BYM2 components
  int phi_start = p + n_re_groups;              // phi_scaled (structured)
  int theta_start = phi_start + n_spatial_units; // theta (unstructured)

  // Precompute BYM2 weights
  double sqrt_rho = std::sqrt(rho + 1e-10);
  double sqrt_1_rho = std::sqrt(1.0 - rho + 1e-10);

  ratiod_omp::ScopedThreadCount thread_scope(n_threads);

  for (int iter = 0; iter < max_iter; iter++) {
    // Compute spatial effects and linear predictor
    NumericVector eta(N);
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int i = 0; i < N; i++) {
      eta[i] = 0.0;
      // Fixed effects
      for (int j = 0; j < p; j++) {
        eta[i] += X(i, j) * x[j];
      }
      // Random effects
      if (n_re_groups > 0) {
        int g = (int)re_idx[i] - 1;
        if (g >= 0 && g < n_re_groups) {
          eta[i] += x[p + g];
        }
      }
      // BYM2 spatial effects: sigma * (sqrt(rho) * phi * scale + sqrt(1-rho) * theta)
      if (n_spatial_units > 0) {
        int s = spatial_idx[i] - 1;
        if (s >= 0 && s < n_spatial_units) {
          double phi_s = x[phi_start + s];
          double theta_s = x[theta_start + s];
          double spatial_effect = sigma_spatial * (
            sqrt_rho * phi_s * scale_factor + sqrt_1_rho * theta_s
          );
          eta[i] += spatial_effect;
        }
      }
    }

    // Initialize gradient and Hessian
    std::fill(grad.begin(), grad.end(), 0.0);
    for (int j = 0; j < n_x; j++) {
      for (int k = 0; k < n_x; k++) {
        H(j, k) = 0.0;
      }
    }

    // Likelihood contributions
    std::vector<double> grad_local(n_x, 0.0);
    std::vector<std::vector<double>> H_local(n_x, std::vector<double>(n_x, 0.0));

    #ifdef _OPENMP
    #pragma omp parallel
    {
      std::vector<double> grad_thread(n_x, 0.0);
      std::vector<std::vector<double>> H_thread(n_x, std::vector<double>(n_x, 0.0));

      #pragma omp for schedule(static)
      for (int i = 0; i < N; i++) {
        double g_i, h_i;

        if (family == "binomial") {
          g_i = grad_log_lik_binomial(y[i], n[i], eta[i]);
          h_i = neg_hess_log_lik_binomial(y[i], n[i], eta[i]);
        } else if (family == "negbin") {
          g_i = grad_log_lik_negbin(y[i], eta[i], phi);
          h_i = neg_hess_log_lik_negbin(y[i], eta[i], phi);
        } else {
          g_i = grad_log_lik_poisson(y[i], eta[i]);
          h_i = neg_hess_log_lik_poisson(y[i], eta[i]);
        }

        // Fixed effects
        for (int j = 0; j < p; j++) {
          grad_thread[j] += g_i * X(i, j);
          for (int k = 0; k < p; k++) {
            H_thread[j][k] += h_i * X(i, j) * X(i, k);
          }
        }

        // Random effects
        if (n_re_groups > 0) {
          int g = (int)re_idx[i] - 1;
          if (g >= 0 && g < n_re_groups) {
            grad_thread[p + g] += g_i;
            H_thread[p + g][p + g] += h_i;
            for (int j = 0; j < p; j++) {
              H_thread[j][p + g] += h_i * X(i, j);
              H_thread[p + g][j] += h_i * X(i, j);
            }
          }
        }

        // BYM2 spatial effects
        if (n_spatial_units > 0) {
          int s = spatial_idx[i] - 1;
          if (s >= 0 && s < n_spatial_units) {
            // Derivative factors for phi and theta
            double d_phi = sigma_spatial * sqrt_rho * scale_factor;
            double d_theta = sigma_spatial * sqrt_1_rho;

            int phi_idx = phi_start + s;
            int theta_idx = theta_start + s;

            // Gradient contributions
            grad_thread[phi_idx] += g_i * d_phi;
            grad_thread[theta_idx] += g_i * d_theta;

            // Hessian contributions
            H_thread[phi_idx][phi_idx] += h_i * d_phi * d_phi;
            H_thread[theta_idx][theta_idx] += h_i * d_theta * d_theta;
            H_thread[phi_idx][theta_idx] += h_i * d_phi * d_theta;
            H_thread[theta_idx][phi_idx] += h_i * d_phi * d_theta;

            // Cross terms with fixed effects
            for (int j = 0; j < p; j++) {
              H_thread[j][phi_idx] += h_i * X(i, j) * d_phi;
              H_thread[phi_idx][j] += h_i * X(i, j) * d_phi;
              H_thread[j][theta_idx] += h_i * X(i, j) * d_theta;
              H_thread[theta_idx][j] += h_i * X(i, j) * d_theta;
            }

            // Cross terms with random effects
            if (n_re_groups > 0) {
              int g = (int)re_idx[i] - 1;
              if (g >= 0 && g < n_re_groups) {
                H_thread[p + g][phi_idx] += h_i * d_phi;
                H_thread[phi_idx][p + g] += h_i * d_phi;
                H_thread[p + g][theta_idx] += h_i * d_theta;
                H_thread[theta_idx][p + g] += h_i * d_theta;
              }
            }
          }
        }
      }

      #pragma omp critical
      {
        for (int j = 0; j < n_x; j++) {
          grad_local[j] += grad_thread[j];
          for (int k = 0; k < n_x; k++) {
            H_local[j][k] += H_thread[j][k];
          }
        }
      }
    }
    #else
    // Non-OpenMP fallback (similar logic)
    for (int i = 0; i < N; i++) {
      double g_i, h_i;
      if (family == "binomial") {
        g_i = grad_log_lik_binomial(y[i], n[i], eta[i]);
        h_i = neg_hess_log_lik_binomial(y[i], n[i], eta[i]);
      } else if (family == "negbin") {
        g_i = grad_log_lik_negbin(y[i], eta[i], phi);
        h_i = neg_hess_log_lik_negbin(y[i], eta[i], phi);
      } else {
        g_i = grad_log_lik_poisson(y[i], eta[i]);
        h_i = neg_hess_log_lik_poisson(y[i], eta[i]);
      }

      for (int j = 0; j < p; j++) {
        grad_local[j] += g_i * X(i, j);
        for (int k = 0; k < p; k++) {
          H_local[j][k] += h_i * X(i, j) * X(i, k);
        }
      }

      if (n_re_groups > 0) {
        int g = (int)re_idx[i] - 1;
        if (g >= 0 && g < n_re_groups) {
          grad_local[p + g] += g_i;
          H_local[p + g][p + g] += h_i;
          for (int j = 0; j < p; j++) {
            H_local[j][p + g] += h_i * X(i, j);
            H_local[p + g][j] += h_i * X(i, j);
          }
        }
      }

      if (n_spatial_units > 0) {
        int s = spatial_idx[i] - 1;
        if (s >= 0 && s < n_spatial_units) {
          double d_phi = sigma_spatial * sqrt_rho * scale_factor;
          double d_theta = sigma_spatial * sqrt_1_rho;
          int phi_idx = phi_start + s;
          int theta_idx = theta_start + s;

          grad_local[phi_idx] += g_i * d_phi;
          grad_local[theta_idx] += g_i * d_theta;
          H_local[phi_idx][phi_idx] += h_i * d_phi * d_phi;
          H_local[theta_idx][theta_idx] += h_i * d_theta * d_theta;
          H_local[phi_idx][theta_idx] += h_i * d_phi * d_theta;
          H_local[theta_idx][phi_idx] += h_i * d_phi * d_theta;

          for (int j = 0; j < p; j++) {
            H_local[j][phi_idx] += h_i * X(i, j) * d_phi;
            H_local[phi_idx][j] += h_i * X(i, j) * d_phi;
            H_local[j][theta_idx] += h_i * X(i, j) * d_theta;
            H_local[theta_idx][j] += h_i * X(i, j) * d_theta;
          }

          if (n_re_groups > 0) {
            int g = (int)re_idx[i] - 1;
            if (g >= 0 && g < n_re_groups) {
              H_local[p + g][phi_idx] += h_i * d_phi;
              H_local[phi_idx][p + g] += h_i * d_phi;
              H_local[p + g][theta_idx] += h_i * d_theta;
              H_local[theta_idx][p + g] += h_i * d_theta;
            }
          }
        }
      }
    }
    #endif

    // Copy to output
    for (int j = 0; j < n_x; j++) {
      grad[j] = grad_local[j];
      for (int k = 0; k < n_x; k++) {
        H(j, k) = H_local[j][k];
      }
    }

    // Prior contributions for random effects
    for (int g = 0; g < n_re_groups; g++) {
      grad[p + g] -= tau_re * x[p + g];
      H(p + g, p + g) += tau_re;
    }

    // Prior for phi_scaled (ICAR with precision 1)
    for (int s = 0; s < n_spatial_units; s++) {
      int phi_idx = phi_start + s;
      double phi_s = x[phi_idx];

      double neighbor_sum = 0.0;
      for (int k = adj_row_ptr[s]; k < adj_row_ptr[s + 1]; k++) {
        int neighbor = adj_col_idx[k];  // Already 0-based
        neighbor_sum += x[phi_start + neighbor];
      }
      // ICAR with precision 1 (variance scaling handled by sigma_spatial)
      grad[phi_idx] -= (n_neighbors[s] * phi_s - neighbor_sum);
      H(phi_idx, phi_idx) += n_neighbors[s];
      for (int k = adj_row_ptr[s]; k < adj_row_ptr[s + 1]; k++) {
        int neighbor = adj_col_idx[k];  // Already 0-based
        H(phi_idx, phi_start + neighbor) -= 1.0;
      }
    }

    // Prior for theta (IID N(0, 1))
    for (int s = 0; s < n_spatial_units; s++) {
      int theta_idx = theta_start + s;
      grad[theta_idx] -= x[theta_idx];
      H(theta_idx, theta_idx) += 1.0;
    }

    // Regularization for fixed effects
    double tau_beta = 1e-4;
    for (int j = 0; j < p; j++) {
      grad[j] -= tau_beta * x[j];
      H(j, j) += tau_beta;
    }

    // Newton step
    NumericMatrix L(n_x, n_x);
    for (int j = 0; j < n_x; j++) {
      for (int k = 0; k <= j; k++) {
        double sum = H(j, k);
        for (int i = 0; i < k; i++) {
          sum -= L(j, i) * L(k, i);
        }
        if (j == k) {
          if (sum <= 0) sum = 1e-6;
          L(j, k) = std::sqrt(sum);
        } else {
          L(j, k) = sum / L(k, k);
        }
      }
    }

    NumericVector z(n_x);
    for (int j = 0; j < n_x; j++) {
      double sum = grad[j];
      for (int k = 0; k < j; k++) {
        sum -= L(j, k) * z[k];
      }
      z[j] = sum / L(j, j);
    }

    NumericVector delta(n_x);
    for (int j = n_x - 1; j >= 0; j--) {
      double sum = z[j];
      for (int k = j + 1; k < n_x; k++) {
        sum -= L(k, j) * delta[k];
      }
      delta[j] = sum / L(j, j);
    }

    double max_delta = 0.0;
    for (int j = 0; j < n_x; j++) {
      max_delta = std::max(max_delta, std::abs(delta[j]));
      x[j] += delta[j];
    }

    if (max_delta < tol) {
      result.converged = true;
      result.n_iter = iter + 1;
      break;
    }
    result.n_iter = iter + 1;
  }

  // Center phi_scaled (sum-to-zero constraint)
  if (n_spatial_units > 0) {
    double mean_phi = 0.0;
    for (int s = 0; s < n_spatial_units; s++) {
      mean_phi += x[phi_start + s];
    }
    mean_phi /= n_spatial_units;
    for (int s = 0; s < n_spatial_units; s++) {
      x[phi_start + s] -= mean_phi;
    }
  }

  result.mode = x;

  // Compute log determinant
  double log_det = 0.0;
  NumericMatrix L_final(n_x, n_x);
  for (int j = 0; j < n_x; j++) {
    for (int k = 0; k <= j; k++) {
      double sum = H(j, k);
      for (int i = 0; i < k; i++) {
        sum -= L_final(j, i) * L_final(k, i);
      }
      if (j == k) {
        if (sum <= 0) sum = 1e-6;
        L_final(j, k) = std::sqrt(sum);
        log_det += std::log(L_final(j, k));
      } else {
        L_final(j, k) = sum / L_final(k, k);
      }
    }
  }
  result.log_det_Q = 2.0 * log_det;

  // Compute log marginal (simplified)
  double log_lik = 0.0;
  NumericVector eta_final(N);
  for (int i = 0; i < N; i++) {
    eta_final[i] = 0.0;
    for (int j = 0; j < p; j++) {
      eta_final[i] += X(i, j) * x[j];
    }
    if (n_re_groups > 0) {
      int g = (int)re_idx[i] - 1;
      if (g >= 0 && g < n_re_groups) {
        eta_final[i] += x[p + g];
      }
    }
    if (n_spatial_units > 0) {
      int s = spatial_idx[i] - 1;
      if (s >= 0 && s < n_spatial_units) {
        double spatial_effect = sigma_spatial * (
          sqrt_rho * x[phi_start + s] * scale_factor +
          sqrt_1_rho * x[theta_start + s]
        );
        eta_final[i] += spatial_effect;
      }
    }
  }

  for (int i = 0; i < N; i++) {
    if (family == "binomial") {
      log_lik += log_lik_binomial(y[i], n[i], eta_final[i]);
    } else if (family == "negbin") {
      log_lik += log_lik_negbin(y[i], eta_final[i], phi);
    } else {
      log_lik += log_lik_poisson(y[i], eta_final[i]);
    }
  }

  // Log priors
  double log_prior_re = 0.0;
  for (int g = 0; g < n_re_groups; g++) {
    log_prior_re += -0.5 * tau_re * x[p + g] * x[p + g];
  }
  if (n_re_groups > 0) {
    log_prior_re += 0.5 * n_re_groups * std::log(tau_re / (2.0 * M_PI));
  }

  // Log prior for theta (IID)
  double log_prior_theta = 0.0;
  for (int s = 0; s < n_spatial_units; s++) {
    log_prior_theta -= 0.5 * x[theta_start + s] * x[theta_start + s];
  }
  log_prior_theta -= 0.5 * n_spatial_units * std::log(2.0 * M_PI);

  // Log prior for phi_scaled (ICAR) - simplified
  double log_prior_phi = 0.0;
  double quad_form = 0.0;
  for (int s = 0; s < n_spatial_units; s++) {
    double phi_s = x[phi_start + s];
    quad_form += n_neighbors[s] * phi_s * phi_s;
    for (int k = adj_row_ptr[s]; k < adj_row_ptr[s + 1]; k++) {
      int neighbor = adj_col_idx[k];  // Already 0-based
      if (neighbor > s) {
        quad_form -= 2.0 * phi_s * x[phi_start + neighbor];
      }
    }
  }
  log_prior_phi = -0.5 * quad_form;

  result.log_marginal = log_lik + log_prior_re + log_prior_theta + log_prior_phi
                        - 0.5 * result.log_det_Q + 0.5 * n_x * std::log(2.0 * M_PI);

  return result;
}

} // namespace ratiod

// ---------------------------------------------------------------------
// R exports
// ---------------------------------------------------------------------

// [[Rcpp::export]]
Rcpp::List cpp_laplace_fit(
    Rcpp::IntegerVector y,
    Rcpp::IntegerVector n,
    Rcpp::NumericMatrix X,
    Rcpp::NumericVector re_idx,
    int n_re_groups,
    double sigma_re,
    std::string family,
    double phi = 1.0,
    int max_iter = 100,
    double tol = 1e-6,
    int n_threads = 1
) {
  ratiod::LaplaceResult result = ratiod::laplace_mode_dense(
    y, n, X, re_idx, n_re_groups, sigma_re, family, phi, max_iter, tol, n_threads
  );

  return Rcpp::List::create(
    Rcpp::Named("mode") = result.mode,
    Rcpp::Named("log_det_Q") = result.log_det_Q,
    Rcpp::Named("log_marginal") = result.log_marginal,
    Rcpp::Named("n_iter") = result.n_iter,
    Rcpp::Named("converged") = result.converged
  );
}

// [[Rcpp::export]]
int cpp_laplace_get_max_threads() {
  #ifdef _OPENMP
  return omp_get_max_threads();
  #else
  return 1;
  #endif
}

// [[Rcpp::export]]
Rcpp::List cpp_laplace_fit_spatial(
    Rcpp::IntegerVector y,
    Rcpp::IntegerVector n,
    Rcpp::NumericMatrix X,
    Rcpp::NumericVector re_idx,
    int n_re_groups,
    double sigma_re,
    Rcpp::IntegerVector spatial_idx,
    int n_spatial_units,
    Rcpp::IntegerVector adj_row_ptr,
    Rcpp::IntegerVector adj_col_idx,
    Rcpp::IntegerVector n_neighbors,
    double tau_spatial,
    std::string family,
    double phi = 1.0,
    int max_iter = 100,
    double tol = 1e-6,
    int n_threads = 1
) {
  ratiod::LaplaceResult result = ratiod::laplace_mode_spatial(
    y, n, X, re_idx, n_re_groups, sigma_re,
    spatial_idx, n_spatial_units, adj_row_ptr, adj_col_idx, n_neighbors, tau_spatial,
    family, phi, max_iter, tol, n_threads
  );

  return Rcpp::List::create(
    Rcpp::Named("mode") = result.mode,
    Rcpp::Named("log_det_Q") = result.log_det_Q,
    Rcpp::Named("log_marginal") = result.log_marginal,
    Rcpp::Named("n_iter") = result.n_iter,
    Rcpp::Named("converged") = result.converged
  );
}

// [[Rcpp::export]]
Rcpp::NumericMatrix cpp_laplace_sample(
    Rcpp::NumericVector mode,
    Rcpp::NumericMatrix H,
    int n_samples
) {
  int n_x = mode.size();
  Rcpp::NumericMatrix samples(n_samples, n_x);

  // Cholesky of H^{-1} = L_inv L_inv'
  // We need to sample from N(mode, H^{-1})
  // Sample z ~ N(0, I), then x = mode + L_inv * z

  // First, Cholesky of H
  Rcpp::NumericMatrix L(n_x, n_x);
  for (int j = 0; j < n_x; j++) {
    for (int k = 0; k <= j; k++) {
      double sum = H(j, k);
      for (int i = 0; i < k; i++) {
        sum -= L(j, i) * L(k, i);
      }
      if (j == k) {
        if (sum <= 0) sum = 1e-6;
        L(j, k) = std::sqrt(sum);
      } else {
        L(j, k) = sum / L(k, k);
      }
    }
  }

  // Sample
  for (int s = 0; s < n_samples; s++) {
    // z ~ N(0, I)
    Rcpp::NumericVector z(n_x);
    for (int j = 0; j < n_x; j++) {
      z[j] = R::rnorm(0.0, 1.0);
    }

    // Solve L' x_centered = z (back substitution)
    Rcpp::NumericVector x_centered(n_x);
    for (int j = n_x - 1; j >= 0; j--) {
      double sum = z[j];
      for (int k = j + 1; k < n_x; k++) {
        sum -= L(k, j) * x_centered[k];
      }
      x_centered[j] = sum / L(j, j);
    }

    // x = mode + x_centered
    for (int j = 0; j < n_x; j++) {
      samples(s, j) = mode[j] + x_centered[j];
    }
  }

  return samples;
}

// [[Rcpp::export]]
Rcpp::List cpp_laplace_fit_bym2(
    Rcpp::IntegerVector y,
    Rcpp::IntegerVector n,
    Rcpp::NumericMatrix X,
    Rcpp::NumericVector re_idx,
    int n_re_groups,
    double sigma_re,
    Rcpp::IntegerVector spatial_idx,
    int n_spatial_units,
    Rcpp::IntegerVector adj_row_ptr,
    Rcpp::IntegerVector adj_col_idx,
    Rcpp::IntegerVector n_neighbors,
    double sigma_spatial,
    double rho,
    double scale_factor,
    std::string family,
    double phi = 1.0,
    int max_iter = 100,
    double tol = 1e-6,
    int n_threads = 1
) {
  ratiod::LaplaceResult result = ratiod::laplace_mode_bym2(
    y, n, X, re_idx, n_re_groups, sigma_re,
    spatial_idx, n_spatial_units, adj_row_ptr, adj_col_idx, n_neighbors,
    sigma_spatial, rho, scale_factor,
    family, phi, max_iter, tol, n_threads
  );

  return Rcpp::List::create(
    Rcpp::Named("mode") = result.mode,
    Rcpp::Named("log_det_Q") = result.log_det_Q,
    Rcpp::Named("log_marginal") = result.log_marginal,
    Rcpp::Named("n_iter") = result.n_iter,
    Rcpp::Named("converged") = result.converged
  );
}

// ---------------------------------------------------------------------
// GP Spatial Laplace (NNGP approximation)
// ---------------------------------------------------------------------

// Helper: compute squared exponential covariance
inline double compute_cov_exp_laplace(double d, double sigma2, double phi) {
  if (d < 1e-10) return sigma2;
  return sigma2 * std::exp(-d / phi);
}

// Helper: compute Matern covariance (nu = 1.5)
inline double compute_cov_matern15_laplace(double d, double sigma2, double phi) {
  if (d < 1e-10) return sigma2;
  double x = std::sqrt(3.0) * d / phi;
  return sigma2 * (1.0 + x) * std::exp(-x);
}

// Helper: compute Matern covariance (nu = 2.5)
inline double compute_cov_matern25_laplace(double d, double sigma2, double phi) {
  if (d < 1e-10) return sigma2;
  double x = std::sqrt(5.0) * d / phi;
  return sigma2 * (1.0 + x + x * x / 3.0) * std::exp(-x);
}

// Compute covariance based on type
inline double compute_cov_laplace(double d, double sigma2, double phi, int cov_type) {
  if (cov_type == 0) {
    return compute_cov_exp_laplace(d, sigma2, phi);
  } else if (cov_type == 1) {
    return compute_cov_matern15_laplace(d, sigma2, phi);
  } else {
    return compute_cov_matern25_laplace(d, sigma2, phi);
  }
}

// Compute NNGP conditional mean and variance for observation i
inline void nngp_conditional_laplace(
    int obs_idx,
    int i,
    const std::vector<double>& w,
    double sigma2,
    double phi,
    int cov_type,
    const Rcpp::NumericMatrix& coords,
    const Rcpp::IntegerMatrix& nn_idx,
    const Rcpp::NumericMatrix& nn_dist,
    const Rcpp::IntegerVector& nn_order,
    int nn,
    double& cond_mean,
    double& cond_var,
    // The Kriging weights and the field positions they multiply. A caller
    // assembling the NNGP precision needs both; one reading only the
    // conditional moments passes nullptr.
    std::vector<int>* nb_out = nullptr,
    std::vector<double>* alpha_out = nullptr
) {
  if (nb_out) nb_out->clear();
  if (alpha_out) alpha_out->clear();

  int n_neighbors = 0;
  for (int j = 0; j < nn; j++) {
    if (nn_idx(i, j) > 0) n_neighbors++;
  }

  if (n_neighbors == 0) {
    cond_mean = 0.0;
    cond_var = sigma2;
    return;
  }

  std::vector<double> c_vec(n_neighbors);
  std::vector<double> C_mat(n_neighbors * n_neighbors);

  for (int j = 0; j < n_neighbors; j++) {
    c_vec[j] = compute_cov_laplace(nn_dist(i, j), sigma2, phi, cov_type);
  }

  for (int j1 = 0; j1 < n_neighbors; j1++) {
    int nn_orig1 = nn_order[nn_idx(i, j1) - 1];
    for (int j2 = 0; j2 < n_neighbors; j2++) {
      int nn_orig2 = nn_order[nn_idx(i, j2) - 1];
      if (j1 == j2) {
        C_mat[j1 * n_neighbors + j2] = sigma2;
      } else {
        double d12 = std::sqrt(
          std::pow(coords(nn_orig1, 0) - coords(nn_orig2, 0), 2) +
          std::pow(coords(nn_orig1, 1) - coords(nn_orig2, 1), 2)
        );
        C_mat[j1 * n_neighbors + j2] = compute_cov_laplace(d12, sigma2, phi, cov_type);
      }
    }
  }

  std::vector<double> L(n_neighbors * n_neighbors, 0.0);
  for (int j = 0; j < n_neighbors; j++) {
    for (int k = 0; k <= j; k++) {
      double sum = C_mat[j * n_neighbors + k];
      for (int m = 0; m < k; m++) {
        sum -= L[j * n_neighbors + m] * L[k * n_neighbors + m];
      }
      if (j == k) {
        L[j * n_neighbors + j] = std::sqrt(std::max(1e-10, sum));
      } else {
        L[j * n_neighbors + k] = sum / L[k * n_neighbors + k];
      }
    }
  }

  std::vector<double> y_solve(n_neighbors);
  for (int j = 0; j < n_neighbors; j++) {
    double sum = c_vec[j];
    for (int k = 0; k < j; k++) {
      sum -= L[j * n_neighbors + k] * y_solve[k];
    }
    y_solve[j] = sum / L[j * n_neighbors + j];
  }

  std::vector<double> alpha(n_neighbors);
  for (int j = n_neighbors - 1; j >= 0; j--) {
    double sum = y_solve[j];
    for (int k = j + 1; k < n_neighbors; k++) {
      sum -= L[k * n_neighbors + j] * alpha[k];
    }
    alpha[j] = sum / L[j * n_neighbors + j];
  }

  cond_mean = 0.0;
  for (int j = 0; j < n_neighbors; j++) {
    int nn_orig = nn_order[nn_idx(i, j) - 1];
    cond_mean += alpha[j] * w[nn_orig];
    if (nb_out) nb_out->push_back(nn_orig);
    if (alpha_out) alpha_out->push_back(alpha[j]);
  }

  double c_Cinv_c = 0.0;
  for (int j = 0; j < n_neighbors; j++) {
    c_Cinv_c += c_vec[j] * alpha[j];
  }
  cond_var = std::max(1e-10, sigma2 - c_Cinv_c);
}

// One NNGP field block's prior, accumulated into the Newton system and
// returned as a log density at the current field values.
//
// The NNGP prior is Q = (I - A)' D^{-1} (I - A): each location's residual
// r_i = w_i - sum_j alpha_ij w_j carries precision d_i = 1 / cond_var_i. Both
// the gradient and the Hessian therefore reach the neighbours as well as the
// location itself -- reading only the diagonal leaves Newton converging to
// something other than the mode of the density log_marginal then reports.
//
// `block_start` is where this field begins in the parameter vector, so the
// same routine serves a single-scale GP and each scale of a multi-scale one.
inline double add_nngp_block_laplace(
    const std::vector<double>& w,
    int block_start,
    int n_spatial,
    double sigma2,
    double phi_range,
    int cov_type,
    const Rcpp::NumericMatrix& coords,
    const Rcpp::IntegerMatrix& nn_idx,
    const Rcpp::NumericMatrix& nn_dist,
    const Rcpp::IntegerVector& nn_order,
    int nn,
    Rcpp::NumericVector& grad,
    Rcpp::NumericMatrix& H
) {
  double log_prior = 0.0;
  std::vector<int> nb;
  std::vector<double> alpha;

  // nn_order maps a position in the NNGP ordering to a location, 0-based. A
  // caller handing over the 1-based ordering R builds would write one past the
  // block on the last location, which corrupts the heap rather than failing.
  if ((int)nn_order.size() < n_spatial) {
    Rcpp::stop("NNGP ordering holds %d entries for %d locations",
               (int)nn_order.size(), n_spatial);
  }
  for (int i = 0; i < n_spatial; i++) {
    if (nn_order[i] < 0 || nn_order[i] >= n_spatial) {
      Rcpp::stop("NNGP ordering entry %d is %d, outside 0..%d; it must be 0-based",
                 i, nn_order[i], n_spatial - 1);
    }
  }

  for (int i = 0; i < n_spatial; i++) {
    const int obs_idx = nn_order[i];
    double cond_mean, cond_var;
    nngp_conditional_laplace(obs_idx, i, w, sigma2, phi_range, cov_type,
                             coords, nn_idx, nn_dist, nn_order, nn,
                             cond_mean, cond_var, &nb, &alpha);

    const double d_i = 1.0 / cond_var;
    const double resid = w[obs_idx] - cond_mean;
    log_prior += -0.5 * std::log(2.0 * M_PI * cond_var) - 0.5 * resid * resid * d_i;

    const int self = block_start + obs_idx;
    grad[self] -= d_i * resid;
    H(self, self) += d_i;

    const int k = static_cast<int>(nb.size());
    for (int a = 0; a < k; a++) {
      const int ia = block_start + nb[a];
      grad[ia] += d_i * resid * alpha[a];
      const double cross = d_i * alpha[a];
      H(self, ia) -= cross;
      H(ia, self) -= cross;
      for (int b = 0; b < k; b++) {
        H(ia, block_start + nb[b]) += cross * alpha[b];
      }
    }
  }

  return log_prior;
}

// One NNGP field over the shared set of locations: its covariance parameters
// and the neighbour structure the ordering was built with. A single-scale GP
// contributes one, a multi-scale GP one per scale.
struct NNGPBlock {
  const Rcpp::IntegerMatrix* nn_idx;
  const Rcpp::NumericMatrix* nn_dist;
  const Rcpp::IntegerVector* nn_order;
  int nn;
  double sigma2;
  double phi_range;
};

// Newton mode-finder for a latent field of fixed effects, group random
// effects, and any number of additive NNGP blocks over the same locations.
//
// `coords` holds one row per location and `obs_to_loc` maps each observation
// to the location entering its linear predictor, so repeated coordinates share
// a field value instead of the trailing observations losing the spatial term.
ratiod::LaplaceResult laplace_mode_nngp(
    const Rcpp::IntegerVector& y,
    const Rcpp::IntegerVector& n,
    const Rcpp::NumericMatrix& X,
    const Rcpp::NumericVector& re_idx,
    int n_re_groups,
    double sigma_re,
    const Rcpp::NumericMatrix& coords,
    const Rcpp::IntegerVector& obs_to_loc,
    int n_spatial,
    int cov_type,
    const std::vector<NNGPBlock>& blocks,
    const std::string& family,
    double phi,
    int max_iter,
    double tol,
    int n_threads
) {
  const int N = y.size();
  const int p = X.ncol();
  const int n_blocks = static_cast<int>(blocks.size());
  const int gp_start = p + n_re_groups;
  const int n_x = gp_start + n_blocks * n_spatial;

  ratiod::LaplaceResult result;
  result.mode = Rcpp::NumericVector(n_x, 0.0);
  result.converged = false;
  result.n_iter = 0;

  Rcpp::NumericVector x(n_x, 0.0);
  Rcpp::NumericVector grad(n_x, 0.0);
  Rcpp::NumericMatrix H(n_x, n_x);
  Rcpp::NumericVector eta(N);

  const double tau_re = 1.0 / (sigma_re * sigma_re + 1e-10);
  const double tau_beta = 1e-4;

  std::vector<double> w(n_spatial);
  std::vector<std::pair<int, double> > loads;
  loads.reserve(p + 1 + n_blocks);

  double log_lik = 0.0;
  double log_prior_gp = 0.0;

  ratiod_omp::ScopedThreadCount thread_scope(n_threads);

  // Assemble eta, the gradient, and the negative Hessian at the current x.
  // Called once per Newton step and once more at the converged point, so the
  // Hessian, its log-determinant, and the log-marginal all describe the mode
  // rather than the step before it.
  auto assemble = [&]() {
    std::fill(grad.begin(), grad.end(), 0.0);
    std::fill(H.begin(), H.end(), 0.0);
    log_lik = 0.0;
    log_prior_gp = 0.0;

    for (int i = 0; i < N; i++) {
      double e = 0.0;
      for (int j = 0; j < p; j++) e += X(i, j) * x[j];
      if (n_re_groups > 0) {
        const int g = static_cast<int>(re_idx[i]) - 1;
        if (g >= 0 && g < n_re_groups) e += x[p + g];
      }
      const int s = obs_to_loc[i];
      if (s >= 0 && s < n_spatial) {
        for (int b = 0; b < n_blocks; b++) e += x[gp_start + b * n_spatial + s];
      }
      eta[i] = e;
    }

    for (int i = 0; i < N; i++) {
      double g_i, h_i;
      if (family == "binomial") {
        g_i = ratiod::grad_log_lik_binomial(y[i], n[i], eta[i]);
        h_i = ratiod::neg_hess_log_lik_binomial(y[i], n[i], eta[i]);
        log_lik += ratiod::log_lik_binomial(y[i], n[i], eta[i]);
      } else if (family == "negbin") {
        g_i = ratiod::grad_log_lik_negbin(y[i], eta[i], phi);
        h_i = ratiod::neg_hess_log_lik_negbin(y[i], eta[i], phi);
        log_lik += ratiod::log_lik_negbin(y[i], eta[i], phi);
      } else {
        g_i = ratiod::grad_log_lik_poisson(y[i], eta[i]);
        h_i = ratiod::neg_hess_log_lik_poisson(y[i], eta[i]);
        log_lik += ratiod::log_lik_poisson(y[i], eta[i]);
      }

      // Every coordinate this observation loads, with the weight it enters
      // eta by. The likelihood reaches the whole outer product of that set,
      // which is what carries the fixed-effect / random-effect / spatial
      // cross terms a block-diagonal assembly drops.
      loads.clear();
      for (int j = 0; j < p; j++) {
        if (X(i, j) != 0.0) loads.push_back(std::make_pair(j, X(i, j)));
      }
      if (n_re_groups > 0) {
        const int g = static_cast<int>(re_idx[i]) - 1;
        if (g >= 0 && g < n_re_groups) loads.push_back(std::make_pair(p + g, 1.0));
      }
      const int s = obs_to_loc[i];
      if (s >= 0 && s < n_spatial) {
        for (int b = 0; b < n_blocks; b++)
          loads.push_back(std::make_pair(gp_start + b * n_spatial + s, 1.0));
      }

      const int n_load = static_cast<int>(loads.size());
      for (int a = 0; a < n_load; a++) {
        grad[loads[a].first] += g_i * loads[a].second;
        const double ha = h_i * loads[a].second;
        for (int c = 0; c < n_load; c++) {
          H(loads[a].first, loads[c].first) += ha * loads[c].second;
        }
      }
    }

    for (int g = 0; g < n_re_groups; g++) {
      grad[p + g] -= tau_re * x[p + g];
      H(p + g, p + g) += tau_re;
    }

    for (int b = 0; b < n_blocks; b++) {
      const int bs = gp_start + b * n_spatial;
      for (int s = 0; s < n_spatial; s++) w[s] = x[bs + s];
      log_prior_gp += add_nngp_block_laplace(
          w, bs, n_spatial, blocks[b].sigma2, blocks[b].phi_range, cov_type,
          coords, *blocks[b].nn_idx, *blocks[b].nn_dist, *blocks[b].nn_order,
          blocks[b].nn, grad, H);
    }

    for (int j = 0; j < p; j++) {
      grad[j] -= tau_beta * x[j];
      H(j, j) += tau_beta;
    }
  };

  // Cholesky of the negative Hessian, reused for the Newton step and, at the
  // mode, for the log-determinant the Laplace correction needs.
  Rcpp::NumericMatrix L(n_x, n_x);
  auto factor = [&]() {
    std::fill(L.begin(), L.end(), 0.0);
    for (int j = 0; j < n_x; j++) {
      for (int k = 0; k <= j; k++) {
        double sum = H(j, k);
        for (int m = 0; m < k; m++) sum -= L(j, m) * L(k, m);
        if (j == k) {
          if (sum <= 0) sum = 1e-6;
          L(j, k) = std::sqrt(sum);
        } else {
          L(j, k) = sum / L(k, k);
        }
      }
    }
  };

  for (int iter = 0; iter < max_iter; iter++) {
    assemble();
    factor();

    Rcpp::NumericVector z(n_x);
    for (int j = 0; j < n_x; j++) {
      double sum = grad[j];
      for (int k = 0; k < j; k++) sum -= L(j, k) * z[k];
      z[j] = sum / L(j, j);
    }

    Rcpp::NumericVector delta(n_x);
    for (int j = n_x - 1; j >= 0; j--) {
      double sum = z[j];
      for (int k = j + 1; k < n_x; k++) sum -= L(k, j) * delta[k];
      delta[j] = sum / L(j, j);
    }

    double max_delta = 0.0;
    for (int j = 0; j < n_x; j++) {
      max_delta = std::max(max_delta, std::abs(delta[j]));
      x[j] += delta[j];
    }

    result.n_iter = iter + 1;
    if (max_delta < tol) {
      result.converged = true;
      break;
    }
  }

  // At the mode: the Hessian the sampler draws from, its log-determinant, and
  // the log-marginal built from the same terms this assembly just evaluated.
  assemble();
  factor();

  double log_det = 0.0;
  for (int j = 0; j < n_x; j++) log_det += std::log(L(j, j));

  double log_prior_re = 0.0;
  for (int g = 0; g < n_re_groups; g++) {
    log_prior_re += -0.5 * tau_re * x[p + g] * x[p + g];
  }
  if (n_re_groups > 0) {
    log_prior_re += 0.5 * n_re_groups * std::log(tau_re / (2.0 * M_PI));
  }

  result.mode = x;
  result.hessian = H;
  result.log_det_Q = 2.0 * log_det;
  result.log_marginal = log_lik + log_prior_re + log_prior_gp
                        - 0.5 * result.log_det_Q + 0.5 * n_x * std::log(2.0 * M_PI);

  return result;
}

// ---------------------------------------------------------------------
// Multiscale Temporal Laplace
// ---------------------------------------------------------------------

// Soft sum-to-zero pin for an intrinsic field block. The constant direction of
// an RW1 or RW2 precision carries no prior mass and is jointly unidentified
// with the intercept, which leaves the Newton system singular along it. The
// penalty -0.5 * lambda * (sum phi)^2 has Hessian lambda * 11', supported
// entirely on that direction; lambda is the constant tulpa/soft_sum_to_zero.h
// derives. The HMC path identifies the same fields by augmenting the precision
// and centring the effects instead (tulpa/sum_to_zero.h), which removes the
// direction from a linear predictor a sampler has to traverse; here the mode
// and its curvature are what the direction has to be resolved for, so the
// penalty stands in for the constraint that makes the Newton system solvable.
inline void add_sum_to_zero_pin_laplace(
    Rcpp::NumericVector& grad,
    Rcpp::NumericMatrix& H,
    const Rcpp::NumericVector& x,
    int start_idx,
    int n
) {
  if (n < 1) return;

  const double lambda = tulpa::s2z_precision(n);

  double sum = 0.0;
  for (int t = 0; t < n; t++) sum += x[start_idx + t];
  const double push = lambda * sum;

  for (int t = 0; t < n; t++) {
    grad[start_idx + t] -= push;
    for (int k = 0; k < n; k++) {
      H(start_idx + t, start_idx + k) += lambda;
    }
  }
}

inline void add_rw1_precision_laplace(
    Rcpp::NumericVector& grad,
    Rcpp::NumericMatrix& H,
    const Rcpp::NumericVector& x,
    int start_idx,
    int n_times,
    double tau,
    bool cyclic
) {
  for (int t = 0; t < n_times; t++) {
    int idx = start_idx + t;

    if (t > 0) {
      double diff = x[idx] - x[idx - 1];
      grad[idx] -= tau * diff;
      grad[idx - 1] += tau * diff;
      H(idx, idx) += tau;
      H(idx - 1, idx - 1) += tau;
      H(idx, idx - 1) -= tau;
      H(idx - 1, idx) -= tau;
    }
  }

  if (cyclic && n_times > 1) {
    int idx_first = start_idx;
    int idx_last = start_idx + n_times - 1;
    double diff = x[idx_first] - x[idx_last];
    grad[idx_first] -= tau * diff;
    grad[idx_last] += tau * diff;
    H(idx_first, idx_first) += tau;
    H(idx_last, idx_last) += tau;
    H(idx_first, idx_last) -= tau;
    H(idx_last, idx_first) -= tau;
  }

  add_sum_to_zero_pin_laplace(grad, H, x, start_idx, n_times);
}

inline void add_rw2_precision_laplace(
    Rcpp::NumericVector& grad,
    Rcpp::NumericMatrix& H,
    const Rcpp::NumericVector& x,
    int start_idx,
    int n_times,
    double tau,
    bool cyclic
) {
  if (n_times < 3) return;

  for (int t = 1; t < n_times - 1; t++) {
    int idx = start_idx + t;
    double diff2 = x[idx - 1] - 2.0 * x[idx] + x[idx + 1];

    grad[idx - 1] -= tau * diff2;
    grad[idx] += 2.0 * tau * diff2;
    grad[idx + 1] -= tau * diff2;

    H(idx - 1, idx - 1) += tau;
    H(idx, idx) += 4.0 * tau;
    H(idx + 1, idx + 1) += tau;
    H(idx - 1, idx) -= 2.0 * tau;
    H(idx, idx - 1) -= 2.0 * tau;
    H(idx, idx + 1) -= 2.0 * tau;
    H(idx + 1, idx) -= 2.0 * tau;
    H(idx - 1, idx + 1) += tau;
    H(idx + 1, idx - 1) += tau;
  }

  // Only the constant direction is pinned. A non-cyclic RW2 also annihilates a
  // linear ramp, which a temporal covariate carries rather than this pin.
  add_sum_to_zero_pin_laplace(grad, H, x, start_idx, n_times);
}

inline void add_ar1_precision_laplace(
    Rcpp::NumericVector& grad,
    Rcpp::NumericMatrix& H,
    const Rcpp::NumericVector& x,
    int start_idx,
    int n_times,
    double tau,
    double rho
) {
  if (n_times < 1) return;

  double tau_marginal = tau * ratiod_ar1::one_minus_rho2(rho);
  int idx0 = start_idx;
  grad[idx0] -= tau_marginal * x[idx0];
  H(idx0, idx0) += tau_marginal;

  for (int t = 1; t < n_times; t++) {
    int idx = start_idx + t;
    int idx_prev = start_idx + t - 1;
    double resid = x[idx] - rho * x[idx_prev];

    grad[idx] -= tau * resid;
    grad[idx_prev] += tau * rho * resid;

    H(idx, idx) += tau;
    H(idx_prev, idx_prev) += tau * rho * rho;
    H(idx, idx_prev) -= tau * rho;
    H(idx_prev, idx) -= tau * rho;
  }
}

ratiod::LaplaceResult laplace_mode_multiscale_temporal(
    const Rcpp::IntegerVector& y,
    const Rcpp::IntegerVector& n,
    const Rcpp::NumericMatrix& X,
    const Rcpp::NumericVector& re_idx,
    int n_re_groups,
    double sigma_re,
    const Rcpp::IntegerVector& time_idx,
    int n_times,
    int seasonal_period,
    int trend_type,
    int short_type,
    double sigma2_trend,
    double sigma2_seasonal,
    double sigma2_short,
    double rho_short,
    const std::string& family,
    double phi,
    int max_iter,
    double tol,
    int n_threads
) {
  int N = y.size();
  int p = X.ncol();

  int n_trend = (trend_type > 0) ? n_times : 0;
  int n_seasonal = (seasonal_period > 0) ? seasonal_period : 0;
  int n_short = (short_type > 0) ? n_times : 0;

  int n_x = p + n_re_groups + n_trend + n_seasonal + n_short;

  ratiod::LaplaceResult result;
  result.mode = Rcpp::NumericVector(n_x, 0.0);
  result.converged = false;
  result.n_iter = 0;

  Rcpp::NumericVector x(n_x, 0.0);
  Rcpp::NumericVector grad(n_x);
  Rcpp::NumericMatrix H(n_x, n_x);

  double tau_re = 1.0 / (sigma_re * sigma_re + 1e-10);
  double tau_trend = 1.0 / (sigma2_trend + 1e-10);
  double tau_seasonal = 1.0 / (sigma2_seasonal + 1e-10);
  double tau_short = 1.0 / (sigma2_short + 1e-10);

  int trend_start = p + n_re_groups;
  int seasonal_start = trend_start + n_trend;
  int short_start = seasonal_start + n_seasonal;

  ratiod_omp::ScopedThreadCount thread_scope(n_threads);

  // Gradient and negative Hessian of the log posterior at x_cur. Runs once per
  // Newton step, and once more at the final mode so the precision returned to
  // the sampler belongs to that mode.
  auto assemble = [&](const Rcpp::NumericVector& x_cur,
                      Rcpp::NumericVector& grad,
                      Rcpp::NumericMatrix& H) {
    Rcpp::NumericVector eta(N);
    for (int i = 0; i < N; i++) {
      eta[i] = 0.0;
      for (int j = 0; j < p; j++) {
        eta[i] += X(i, j) * x_cur[j];
      }
      if (n_re_groups > 0) {
        int g = (int)re_idx[i] - 1;
        if (g >= 0 && g < n_re_groups) {
          eta[i] += x_cur[p + g];
        }
      }

      int t = time_idx[i] - 1;
      if (t >= 0 && t < n_times) {
        if (n_trend > 0 && t < n_trend) {
          eta[i] += x_cur[trend_start + t];
        }
        if (n_seasonal > 0) {
          int s = t % seasonal_period;
          if (s < n_seasonal) {
            eta[i] += x_cur[seasonal_start + s];
          }
        }
        if (n_short > 0 && t < n_short) {
          eta[i] += x_cur[short_start + t];
        }
      }
    }

    std::fill(grad.begin(), grad.end(), 0.0);
    for (int j = 0; j < n_x; j++) {
      for (int k = 0; k < n_x; k++) {
        H(j, k) = 0.0;
      }
    }

    for (int i = 0; i < N; i++) {
      double g_i, h_i;

      if (family == "binomial") {
        g_i = ratiod::grad_log_lik_binomial(y[i], n[i], eta[i]);
        h_i = ratiod::neg_hess_log_lik_binomial(y[i], n[i], eta[i]);
      } else if (family == "negbin") {
        g_i = ratiod::grad_log_lik_negbin(y[i], eta[i], phi);
        h_i = ratiod::neg_hess_log_lik_negbin(y[i], eta[i], phi);
      } else {
        g_i = ratiod::grad_log_lik_poisson(y[i], eta[i]);
        h_i = ratiod::neg_hess_log_lik_poisson(y[i], eta[i]);
      }

      for (int j = 0; j < p; j++) {
        grad[j] += g_i * X(i, j);
        for (int k = 0; k < p; k++) {
          H(j, k) += h_i * X(i, j) * X(i, k);
        }
      }

      if (n_re_groups > 0) {
        int g = (int)re_idx[i] - 1;
        if (g >= 0 && g < n_re_groups) {
          grad[p + g] += g_i;
          H(p + g, p + g) += h_i;
        }
      }

      int t = time_idx[i] - 1;
      if (t >= 0 && t < n_times) {
        if (n_trend > 0 && t < n_trend) {
          int idx = trend_start + t;
          grad[idx] += g_i;
          H(idx, idx) += h_i;
        }
        if (n_seasonal > 0) {
          int s = t % seasonal_period;
          if (s < n_seasonal) {
            int idx = seasonal_start + s;
            grad[idx] += g_i;
            H(idx, idx) += h_i;
          }
        }
        if (n_short > 0 && t < n_short) {
          int idx = short_start + t;
          grad[idx] += g_i;
          H(idx, idx) += h_i;
        }
      }
    }

    for (int g = 0; g < n_re_groups; g++) {
      grad[p + g] -= tau_re * x_cur[p + g];
      H(p + g, p + g) += tau_re;
    }

    if (trend_type == 1) {
      add_rw1_precision_laplace(grad, H, x_cur, trend_start, n_trend, tau_trend, false);
    } else if (trend_type == 2) {
      add_rw2_precision_laplace(grad, H, x_cur, trend_start, n_trend, tau_trend, false);
    }

    if (n_seasonal > 0) {
      add_rw1_precision_laplace(grad, H, x_cur, seasonal_start, n_seasonal, tau_seasonal, true);
    }

    if (short_type == 1) {
      add_ar1_precision_laplace(grad, H, x_cur, short_start, n_short, tau_short, rho_short);
    } else if (short_type == 2) {
      for (int t = 0; t < n_short; t++) {
        int idx = short_start + t;
        grad[idx] -= tau_short * x_cur[idx];
        H(idx, idx) += tau_short;
      }
    }

    double tau_beta = 1e-4;
    for (int j = 0; j < p; j++) {
      grad[j] -= tau_beta * x_cur[j];
      H(j, j) += tau_beta;
    }
  };

  for (int iter = 0; iter < max_iter; iter++) {
    assemble(x, grad, H);

    Rcpp::NumericMatrix L(n_x, n_x);
    for (int j = 0; j < n_x; j++) {
      for (int k = 0; k <= j; k++) {
        double sum = H(j, k);
        for (int m = 0; m < k; m++) {
          sum -= L(j, m) * L(k, m);
        }
        if (j == k) {
          if (sum <= 0) sum = 1e-6;
          L(j, k) = std::sqrt(sum);
        } else {
          L(j, k) = sum / L(k, k);
        }
      }
    }

    Rcpp::NumericVector z(n_x);
    for (int j = 0; j < n_x; j++) {
      double sum = grad[j];
      for (int k = 0; k < j; k++) {
        sum -= L(j, k) * z[k];
      }
      z[j] = sum / L(j, j);
    }

    Rcpp::NumericVector delta(n_x);
    for (int j = n_x - 1; j >= 0; j--) {
      double sum = z[j];
      for (int k = j + 1; k < n_x; k++) {
        sum -= L(k, j) * delta[k];
      }
      delta[j] = sum / L(j, j);
    }

    double max_delta = 0.0;
    for (int j = 0; j < n_x; j++) {
      max_delta = std::max(max_delta, std::abs(delta[j]));
      x[j] += delta[j];
    }

    if (max_delta < tol) {
      result.converged = true;
      result.n_iter = iter + 1;
      break;
    }
    result.n_iter = iter + 1;
  }

  if (n_trend > 0) {
    double mean_trend = 0.0;
    for (int t = 0; t < n_trend; t++) mean_trend += x[trend_start + t];
    mean_trend /= n_trend;
    for (int t = 0; t < n_trend; t++) x[trend_start + t] -= mean_trend;
  }
  if (n_seasonal > 0) {
    double mean_seasonal = 0.0;
    for (int s = 0; s < n_seasonal; s++) mean_seasonal += x[seasonal_start + s];
    mean_seasonal /= n_seasonal;
    for (int s = 0; s < n_seasonal; s++) x[seasonal_start + s] -= mean_seasonal;
  }

  result.mode = x;

  // Centring the trend and seasonal blocks moves eta, so the precision the
  // samples are drawn from is formed after it.
  assemble(x, grad, H);
  result.hessian = H;

  double log_det = 0.0;
  Rcpp::NumericMatrix L_final(n_x, n_x);
  for (int j = 0; j < n_x; j++) {
    for (int k = 0; k <= j; k++) {
      double sum = H(j, k);
      for (int m = 0; m < k; m++) {
        sum -= L_final(j, m) * L_final(k, m);
      }
      if (j == k) {
        if (sum <= 0) sum = 1e-6;
        L_final(j, k) = std::sqrt(sum);
        log_det += std::log(L_final(j, k));
      } else {
        L_final(j, k) = sum / L_final(k, k);
      }
    }
  }
  result.log_det_Q = 2.0 * log_det;
  result.log_marginal = -0.5 * result.log_det_Q + 0.5 * n_x * std::log(2.0 * M_PI);

  return result;
}

// [[Rcpp::export]]
Rcpp::List cpp_laplace_fit_gp(
    Rcpp::IntegerVector y,
    Rcpp::IntegerVector n,
    Rcpp::NumericMatrix X,
    Rcpp::NumericVector re_idx,
    int n_re_groups,
    double sigma_re,
    Rcpp::NumericMatrix coords,
    Rcpp::IntegerVector obs_to_loc,
    Rcpp::IntegerMatrix nn_idx,
    Rcpp::NumericMatrix nn_dist,
    Rcpp::IntegerVector nn_order,
    int n_spatial,
    int nn,
    double sigma2_gp,
    double phi_gp,
    int cov_type,
    std::string family,
    double phi = 1.0,
    int max_iter = 100,
    double tol = 1e-6,
    int n_threads = 1
) {
  std::vector<NNGPBlock> blocks(1);
  blocks[0].nn_idx = &nn_idx;
  blocks[0].nn_dist = &nn_dist;
  blocks[0].nn_order = &nn_order;
  blocks[0].nn = nn;
  blocks[0].sigma2 = sigma2_gp;
  blocks[0].phi_range = phi_gp;

  ratiod::LaplaceResult result = laplace_mode_nngp(
    y, n, X, re_idx, n_re_groups, sigma_re,
    coords, obs_to_loc, n_spatial, cov_type, blocks,
    family, phi, max_iter, tol, n_threads
  );

  return Rcpp::List::create(
    Rcpp::Named("mode") = result.mode,
    Rcpp::Named("hessian") = result.hessian,
    Rcpp::Named("log_det_Q") = result.log_det_Q,
    Rcpp::Named("log_marginal") = result.log_marginal,
    Rcpp::Named("n_iter") = result.n_iter,
    Rcpp::Named("converged") = result.converged
  );
}

namespace ratiod {

// ---------------------------------------------------------------------
// Laplace mode finding with RSR (Restricted Spatial Regression)
// RSR projects spatial effects to be orthogonal to covariate space
// w_rsr = P_perp * w where P_perp = I - X(X'X)^{-1}X'
// ---------------------------------------------------------------------

LaplaceResult laplace_mode_rsr(
    const IntegerVector& y,
    const IntegerVector& n,
    const NumericMatrix& X,
    const NumericVector& re_idx,
    int n_re_groups,
    double sigma_re,
    const IntegerVector& spatial_idx,
    int n_spatial_units,
    const IntegerVector& adj_row_ptr,
    const IntegerVector& adj_col_idx,
    const IntegerVector& n_neighbors,
    double tau_spatial,
    const NumericVector& rsr_projection,  // P_perp matrix (n_spatial x n_spatial, row-major)
    int rsr_n,
    const std::string& family,
    double phi,
    int max_iter,
    double tol,
    int n_threads
) {
  int N = y.size();
  int p = X.ncol();
  // Parameters: fixed effects + RE + spatial effects
  int n_x = p + n_re_groups + n_spatial_units;

  LaplaceResult result;
  result.mode = NumericVector(n_x, 0.0);
  result.converged = false;
  result.n_iter = 0;

  // Initialize
  NumericVector x(n_x, 0.0);
  NumericVector grad(n_x);
  NumericMatrix H(n_x, n_x);

  double tau_re = 1.0 / (sigma_re * sigma_re + 1e-10);

  // Indices
  int spatial_start = p + n_re_groups;

  ratiod_omp::ScopedThreadCount thread_scope(n_threads);

  for (int iter = 0; iter < max_iter; iter++) {
    // Compute projected spatial effects: w_proj = P_perp * w
    NumericVector w_proj(n_spatial_units, 0.0);
    for (int i = 0; i < n_spatial_units; i++) {
      for (int j = 0; j < n_spatial_units; j++) {
        // P_perp is stored row-major
        w_proj[i] += rsr_projection[i * rsr_n + j] * x[spatial_start + j];
      }
    }

    // Compute linear predictor with projected spatial effects
    NumericVector eta(N);
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int i = 0; i < N; i++) {
      eta[i] = 0.0;
      // Fixed effects
      for (int j = 0; j < p; j++) {
        eta[i] += X(i, j) * x[j];
      }
      // Random effects
      if (n_re_groups > 0) {
        int g = (int)re_idx[i] - 1;
        if (g >= 0 && g < n_re_groups) {
          eta[i] += x[p + g];
        }
      }
      // RSR projected spatial effects
      if (n_spatial_units > 0) {
        int s = spatial_idx[i] - 1;  // 1-based to 0-based
        if (s >= 0 && s < n_spatial_units) {
          eta[i] += w_proj[s];  // Use projected spatial effect
        }
      }
    }

    // Initialize gradient and Hessian
    std::fill(grad.begin(), grad.end(), 0.0);
    for (int j = 0; j < n_x; j++) {
      for (int k = 0; k < n_x; k++) {
        H(j, k) = 0.0;
      }
    }

    // Likelihood contributions
    // The chain rule for RSR: d/dw = P_perp' * d/dw_proj = P_perp' * d/deta
    // (since P_perp is symmetric, P_perp' = P_perp)
    std::vector<double> grad_local(n_x, 0.0);
    std::vector<std::vector<double>> H_local(n_x, std::vector<double>(n_x, 0.0));

    // First compute gradient w.r.t. projected spatial effects
    std::vector<double> grad_w_proj(n_spatial_units, 0.0);
    std::vector<double> H_w_proj_diag(n_spatial_units, 0.0);

    for (int i = 0; i < N; i++) {
      double g_i, h_i;

      if (family == "binomial") {
        g_i = grad_log_lik_binomial(y[i], n[i], eta[i]);
        h_i = neg_hess_log_lik_binomial(y[i], n[i], eta[i]);
      } else if (family == "negbin") {
        g_i = grad_log_lik_negbin(y[i], eta[i], phi);
        h_i = neg_hess_log_lik_negbin(y[i], eta[i], phi);
      } else {
        g_i = grad_log_lik_poisson(y[i], eta[i]);
        h_i = neg_hess_log_lik_poisson(y[i], eta[i]);
      }

      // Fixed effects
      for (int j = 0; j < p; j++) {
        grad_local[j] += g_i * X(i, j);
        for (int k = 0; k < p; k++) {
          H_local[j][k] += h_i * X(i, j) * X(i, k);
        }
      }

      // Random effects
      if (n_re_groups > 0) {
        int g = (int)re_idx[i] - 1;
        if (g >= 0 && g < n_re_groups) {
          grad_local[p + g] += g_i;
          H_local[p + g][p + g] += h_i;
          for (int j = 0; j < p; j++) {
            H_local[j][p + g] += h_i * X(i, j);
            H_local[p + g][j] += h_i * X(i, j);
          }
        }
      }

      // Gradient w.r.t. projected spatial effect
      if (n_spatial_units > 0) {
        int s = spatial_idx[i] - 1;
        if (s >= 0 && s < n_spatial_units) {
          grad_w_proj[s] += g_i;
          H_w_proj_diag[s] += h_i;
        }
      }
    }

    // Transform gradients and Hessian through RSR projection
    // grad_w = P_perp' * grad_w_proj (P_perp is symmetric)
    for (int i = 0; i < n_spatial_units; i++) {
      for (int j = 0; j < n_spatial_units; j++) {
        grad_local[spatial_start + i] += rsr_projection[i * rsr_n + j] * grad_w_proj[j];
      }
    }

    // Hessian for spatial: H_w = P_perp' * diag(H_w_proj) * P_perp
    // Since H_w_proj is diagonal, this becomes:
    // H_w[i,j] = sum_k P_perp[k,i] * H_w_proj[k] * P_perp[k,j]
    for (int i = 0; i < n_spatial_units; i++) {
      for (int j = 0; j <= i; j++) {
        double sum = 0.0;
        for (int k = 0; k < n_spatial_units; k++) {
          // P_perp is symmetric, so P_perp[k,i] = P_perp[i,k]
          sum += rsr_projection[k * rsr_n + i] * H_w_proj_diag[k] * rsr_projection[k * rsr_n + j];
        }
        H_local[spatial_start + i][spatial_start + j] = sum;
        if (i != j) {
          H_local[spatial_start + j][spatial_start + i] = sum;
        }
      }
    }

    // Cross-terms between fixed effects and projected spatial effects
    for (int i = 0; i < N; i++) {
      double h_i;
      if (family == "binomial") {
        h_i = neg_hess_log_lik_binomial(y[i], n[i], eta[i]);
      } else if (family == "negbin") {
        h_i = neg_hess_log_lik_negbin(y[i], eta[i], phi);
      } else {
        h_i = neg_hess_log_lik_poisson(y[i], eta[i]);
      }

      if (n_spatial_units > 0) {
        int s = spatial_idx[i] - 1;
        if (s >= 0 && s < n_spatial_units) {
          // Cross-terms with fixed effects through projection
          for (int j = 0; j < p; j++) {
            for (int k = 0; k < n_spatial_units; k++) {
              double P_ks = rsr_projection[s * rsr_n + k];  // P_perp[s,k]
              H_local[j][spatial_start + k] += h_i * X(i, j) * P_ks;
              H_local[spatial_start + k][j] += h_i * X(i, j) * P_ks;
            }
          }

          // Cross-terms with random effects through projection
          if (n_re_groups > 0) {
            int g = (int)re_idx[i] - 1;
            if (g >= 0 && g < n_re_groups) {
              for (int k = 0; k < n_spatial_units; k++) {
                double P_ks = rsr_projection[s * rsr_n + k];
                H_local[p + g][spatial_start + k] += h_i * P_ks;
                H_local[spatial_start + k][p + g] += h_i * P_ks;
              }
            }
          }
        }
      }
    }

    // Copy to main gradient and Hessian
    for (int j = 0; j < n_x; j++) {
      grad[j] = grad_local[j];
      for (int k = 0; k < n_x; k++) {
        H(j, k) = H_local[j][k];
      }
    }

    // Prior contributions
    // Random effects prior
    for (int g = 0; g < n_re_groups; g++) {
      grad[p + g] -= tau_re * x[p + g];
      H(p + g, p + g) += tau_re;
    }

    // ICAR prior on unprojected spatial effects
    for (int s = 0; s < n_spatial_units; s++) {
      int sp_idx = spatial_start + s;
      double phi_s = x[sp_idx];

      double neighbor_sum = 0.0;
      for (int k = adj_row_ptr[s]; k < adj_row_ptr[s + 1]; k++) {
        int neighbor = adj_col_idx[k];  // Already 0-based
        neighbor_sum += x[spatial_start + neighbor];
      }
      grad[sp_idx] -= tau_spatial * (n_neighbors[s] * phi_s - neighbor_sum);
      H(sp_idx, sp_idx) += tau_spatial * n_neighbors[s];

      for (int k = adj_row_ptr[s]; k < adj_row_ptr[s + 1]; k++) {
        int neighbor = adj_col_idx[k];  // Already 0-based
        int nb_idx = spatial_start + neighbor;
        H(sp_idx, nb_idx) -= tau_spatial;
      }
    }

    // Regularization for fixed effects
    double tau_beta = 1e-4;
    for (int j = 0; j < p; j++) {
      grad[j] -= tau_beta * x[j];
      H(j, j) += tau_beta;
    }

    // Newton step
    NumericMatrix L(n_x, n_x);
    for (int j = 0; j < n_x; j++) {
      for (int k = 0; k <= j; k++) {
        double sum = H(j, k);
        for (int i = 0; i < k; i++) {
          sum -= L(j, i) * L(k, i);
        }
        if (j == k) {
          if (sum <= 0) sum = 1e-6;
          L(j, k) = std::sqrt(sum);
        } else {
          L(j, k) = sum / L(k, k);
        }
      }
    }

    // Forward substitution
    NumericVector z(n_x);
    for (int j = 0; j < n_x; j++) {
      double sum = grad[j];
      for (int k = 0; k < j; k++) {
        sum -= L(j, k) * z[k];
      }
      z[j] = sum / L(j, j);
    }

    // Back substitution
    NumericVector delta(n_x);
    for (int j = n_x - 1; j >= 0; j--) {
      double sum = z[j];
      for (int k = j + 1; k < n_x; k++) {
        sum -= L(k, j) * delta[k];
      }
      delta[j] = sum / L(j, j);
    }

    // Update
    double max_delta = 0.0;
    for (int j = 0; j < n_x; j++) {
      max_delta = std::max(max_delta, std::abs(delta[j]));
      x[j] += delta[j];
    }

    if (max_delta < tol) {
      result.converged = true;
      result.n_iter = iter + 1;
      break;
    }
    result.n_iter = iter + 1;
  }

  // Center spatial effects (soft sum-to-zero constraint)
  if (n_spatial_units > 0) {
    double mean_phi = 0.0;
    for (int s = 0; s < n_spatial_units; s++) {
      mean_phi += x[spatial_start + s];
    }
    mean_phi /= n_spatial_units;
    for (int s = 0; s < n_spatial_units; s++) {
      x[spatial_start + s] -= mean_phi;
    }
  }

  result.mode = x;

  // Compute log determinant and log marginal
  double log_det = 0.0;
  NumericMatrix L_final(n_x, n_x);

  for (int j = 0; j < n_x; j++) {
    for (int k = 0; k <= j; k++) {
      double sum = H(j, k);
      for (int i = 0; i < k; i++) {
        sum -= L_final(j, i) * L_final(k, i);
      }
      if (j == k) {
        if (sum <= 0) sum = 1e-6;
        L_final(j, k) = std::sqrt(sum);
        log_det += std::log(L_final(j, k));
      } else {
        L_final(j, k) = sum / L_final(k, k);
      }
    }
  }
  result.log_det_Q = 2.0 * log_det;

  // Compute final projected spatial effects
  NumericVector w_proj_final(n_spatial_units, 0.0);
  for (int i = 0; i < n_spatial_units; i++) {
    for (int j = 0; j < n_spatial_units; j++) {
      w_proj_final[i] += rsr_projection[i * rsr_n + j] * x[spatial_start + j];
    }
  }

  // Compute log likelihood at mode
  double log_lik = 0.0;
  NumericVector eta_final(N);
  for (int i = 0; i < N; i++) {
    eta_final[i] = 0.0;
    for (int j = 0; j < p; j++) {
      eta_final[i] += X(i, j) * x[j];
    }
    if (n_re_groups > 0) {
      int g = (int)re_idx[i] - 1;
      if (g >= 0 && g < n_re_groups) {
        eta_final[i] += x[p + g];
      }
    }
    if (n_spatial_units > 0) {
      int s = spatial_idx[i] - 1;
      if (s >= 0 && s < n_spatial_units) {
        eta_final[i] += w_proj_final[s];
      }
    }
  }

  for (int i = 0; i < N; i++) {
    if (family == "binomial") {
      log_lik += log_lik_binomial(y[i], n[i], eta_final[i]);
    } else if (family == "negbin") {
      log_lik += log_lik_negbin(y[i], eta_final[i], phi);
    } else {
      log_lik += log_lik_poisson(y[i], eta_final[i]);
    }
  }

  // Log prior for RE
  double log_prior_re = 0.0;
  for (int g = 0; g < n_re_groups; g++) {
    log_prior_re += -0.5 * tau_re * x[p + g] * x[p + g];
  }
  if (n_re_groups > 0) {
    log_prior_re += 0.5 * n_re_groups * std::log(tau_re / (2.0 * M_PI));
  }

  // Log prior for spatial (ICAR)
  double log_prior_spatial = 0.0;
  if (n_spatial_units > 0) {
    double quad_form = 0.0;
    for (int s = 0; s < n_spatial_units; s++) {
      double phi_s = x[spatial_start + s];
      quad_form += n_neighbors[s] * phi_s * phi_s;
      for (int k = adj_row_ptr[s]; k < adj_row_ptr[s + 1]; k++) {
        int neighbor = adj_col_idx[k];  // Already 0-based
        if (neighbor > s) {
          double phi_n = x[spatial_start + neighbor];
          quad_form -= 2.0 * phi_s * phi_n;
        }
      }
    }
    log_prior_spatial = -0.5 * tau_spatial * quad_form;
    log_prior_spatial += 0.5 * (n_spatial_units - 1) * std::log(tau_spatial / (2.0 * M_PI));
  }

  result.log_marginal = log_lik + log_prior_re + log_prior_spatial
                        - 0.5 * result.log_det_Q + 0.5 * n_x * std::log(2.0 * M_PI);

  return result;
}

} // namespace ratiod

// [[Rcpp::export]]
Rcpp::List cpp_laplace_fit_rsr(
    Rcpp::IntegerVector y,
    Rcpp::IntegerVector n,
    Rcpp::NumericMatrix X,
    Rcpp::NumericVector re_idx,
    int n_re_groups,
    double sigma_re,
    Rcpp::IntegerVector spatial_idx,
    int n_spatial_units,
    Rcpp::IntegerVector adj_row_ptr,
    Rcpp::IntegerVector adj_col_idx,
    Rcpp::IntegerVector n_neighbors,
    double tau_spatial,
    Rcpp::NumericVector rsr_projection,
    int rsr_n,
    std::string family,
    double phi = 1.0,
    int max_iter = 100,
    double tol = 1e-6,
    int n_threads = 1
) {
  ratiod::LaplaceResult result = ratiod::laplace_mode_rsr(
    y, n, X, re_idx, n_re_groups, sigma_re,
    spatial_idx, n_spatial_units, adj_row_ptr, adj_col_idx, n_neighbors,
    tau_spatial, rsr_projection, rsr_n,
    family, phi, max_iter, tol, n_threads
  );

  return Rcpp::List::create(
    Rcpp::Named("mode") = result.mode,
    Rcpp::Named("log_det_Q") = result.log_det_Q,
    Rcpp::Named("log_marginal") = result.log_marginal,
    Rcpp::Named("n_iter") = result.n_iter,
    Rcpp::Named("converged") = result.converged
  );
}

// [[Rcpp::export]]
Rcpp::List cpp_laplace_fit_multiscale_gp(
    Rcpp::IntegerVector y,
    Rcpp::IntegerVector n,
    Rcpp::NumericMatrix X,
    Rcpp::NumericVector re_idx,
    int n_re_groups,
    double sigma_re,
    Rcpp::NumericMatrix coords,
    Rcpp::IntegerVector obs_to_loc,
    Rcpp::IntegerMatrix nn_idx_local,
    Rcpp::NumericMatrix nn_dist_local,
    Rcpp::IntegerVector nn_order_local,
    int nn_local,
    Rcpp::IntegerMatrix nn_idx_regional,
    Rcpp::NumericMatrix nn_dist_regional,
    Rcpp::IntegerVector nn_order_regional,
    int nn_regional,
    int n_spatial,
    double sigma2_local,
    double phi_local,
    double sigma2_regional,
    double phi_regional,
    int cov_type,
    std::string family,
    double phi = 1.0,
    int max_iter = 100,
    double tol = 1e-6,
    int n_threads = 1
) {
  std::vector<NNGPBlock> blocks(2);
  blocks[0].nn_idx = &nn_idx_local;
  blocks[0].nn_dist = &nn_dist_local;
  blocks[0].nn_order = &nn_order_local;
  blocks[0].nn = nn_local;
  blocks[0].sigma2 = sigma2_local;
  blocks[0].phi_range = phi_local;
  blocks[1].nn_idx = &nn_idx_regional;
  blocks[1].nn_dist = &nn_dist_regional;
  blocks[1].nn_order = &nn_order_regional;
  blocks[1].nn = nn_regional;
  blocks[1].sigma2 = sigma2_regional;
  blocks[1].phi_range = phi_regional;

  ratiod::LaplaceResult result = laplace_mode_nngp(
    y, n, X, re_idx, n_re_groups, sigma_re,
    coords, obs_to_loc, n_spatial, cov_type, blocks,
    family, phi, max_iter, tol, n_threads
  );

  return Rcpp::List::create(
    Rcpp::Named("mode") = result.mode,
    Rcpp::Named("hessian") = result.hessian,
    Rcpp::Named("log_det_Q") = result.log_det_Q,
    Rcpp::Named("log_marginal") = result.log_marginal,
    Rcpp::Named("n_iter") = result.n_iter,
    Rcpp::Named("converged") = result.converged
  );
}

// [[Rcpp::export]]
Rcpp::List cpp_laplace_fit_multiscale_temporal(
    Rcpp::IntegerVector y,
    Rcpp::IntegerVector n,
    Rcpp::NumericMatrix X,
    Rcpp::NumericVector re_idx,
    int n_re_groups,
    double sigma_re,
    Rcpp::IntegerVector time_idx,
    int n_times,
    int seasonal_period,
    int trend_type,
    int short_type,
    double sigma2_trend,
    double sigma2_seasonal,
    double sigma2_short,
    double rho_short,
    std::string family,
    double phi = 1.0,
    int max_iter = 100,
    double tol = 1e-6,
    int n_threads = 1
) {
  ratiod::LaplaceResult result = laplace_mode_multiscale_temporal(
    y, n, X, re_idx, n_re_groups, sigma_re,
    time_idx, n_times, seasonal_period, trend_type, short_type,
    sigma2_trend, sigma2_seasonal, sigma2_short, rho_short,
    family, phi, max_iter, tol, n_threads
  );

  return Rcpp::List::create(
    Rcpp::Named("mode") = result.mode,
    Rcpp::Named("hessian") = result.hessian,
    Rcpp::Named("log_det_Q") = result.log_det_Q,
    Rcpp::Named("log_marginal") = result.log_marginal,
    Rcpp::Named("n_iter") = result.n_iter,
    Rcpp::Named("converged") = result.converged
  );
}
