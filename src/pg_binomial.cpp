// pg_binomial.cpp
// Pólya-Gamma Gibbs sampler for binomial models with random effects
// Based on Polson, Scott & Windle (2013) JASA

#include "pg_binomial.h"
#include "pg_rng.h"
#include "hmc_cov.h"
#include "cov_type_code.h"
#include "linalg_fast.h"
#include "omp_thread_scope.h"
#include <RcppEigen.h>
#include <Rcpp.h>
#include <cmath>
#include <algorithm>
#include <vector>

// OpenMP parallelization notes:
// - SAFE to parallelize: matrix-vector products (X*beta), linear predictor computation
// - NOT SAFE: loops calling R's RNG (R::rnorm, rpg_int) or modifying Rcpp objects
// The #pragma omp directives below are applied ONLY to safe arithmetic operations
#ifdef _OPENMP
#include <omp.h>
#endif

using namespace Rcpp;

namespace ratiod {

// ---------------------------------------------------------------------
// Helper: solve linear system with diagonal covariance
// For posterior: (X'WX + D^{-1})^{-1} X'W kappa
// where W = diag(omega), D = prior precision
// ---------------------------------------------------------------------

// Compute X'WX + prior_prec * I and X'W * (kappa - offset)
// Returns list with posterior mean and precision matrix
List compute_posterior_normal(
    const NumericMatrix& X,
    const NumericVector& omega,
    const NumericVector& kappa,
    const NumericVector& offset,
    double prior_prec
) {
  int n = X.nrow();
  int p = X.ncol();

  // X'WX + prior_prec * I
  NumericMatrix XWX(p, p);
  NumericVector XWkappa(p);

  for (int j = 0; j < p; j++) {
    for (int k = j; k < p; k++) {
      double sum = 0.0;
      for (int i = 0; i < n; i++) {
        sum += X(i, j) * omega[i] * X(i, k);
      }
      XWX(j, k) = sum;
      if (j != k) XWX(k, j) = sum;
    }
    // Add prior precision to diagonal
    XWX(j, j) += prior_prec;

    // X'W(kappa - offset)
    double sum_kappa = 0.0;
    for (int i = 0; i < n; i++) {
      sum_kappa += X(i, j) * omega[i] * (kappa[i] / omega[i] - offset[i]);
    }
    XWkappa[j] = sum_kappa;
  }

  return List::create(
    Named("XWX") = XWX,
    Named("XWkappa") = XWkappa
  );
}

// Cholesky solve: solve (L L') x = b
NumericVector chol_solve(const NumericMatrix& L, const NumericVector& b) {
  int p = L.ncol();
  NumericVector y(p), x(p);

  // Forward substitution: L y = b
  for (int i = 0; i < p; i++) {
    double sum = b[i];
    for (int j = 0; j < i; j++) {
      sum -= L(i, j) * y[j];
    }
    y[i] = sum / L(i, i);
  }

  // Back substitution: L' x = y
  for (int i = p - 1; i >= 0; i--) {
    double sum = y[i];
    for (int j = i + 1; j < p; j++) {
      sum -= L(j, i) * x[j];
    }
    x[i] = sum / L(i, i);
  }

  return x;
}

// Cholesky decomposition (lower triangular)
NumericMatrix chol_decomp(const NumericMatrix& A) {
  int n = A.nrow();
  NumericMatrix L(n, n);

  for (int i = 0; i < n; i++) {
    for (int j = 0; j <= i; j++) {
      double sum = 0.0;
      for (int k = 0; k < j; k++) {
        sum += L(i, k) * L(j, k);
      }
      if (i == j) {
        L(i, j) = std::sqrt(std::max(A(i, i) - sum, 1e-10));
      } else {
        L(i, j) = (A(i, j) - sum) / L(j, j);
      }
    }
  }
  return L;
}

// Sample from multivariate normal using Cholesky
NumericVector rmvnorm_chol(const NumericVector& mean, const NumericMatrix& L) {
  int p = mean.size();
  NumericVector z(p), x(p);

  // Sample standard normal
  for (int i = 0; i < p; i++) {
    z[i] = R::rnorm(0.0, 1.0);
  }

  // x = mean + L * z
  for (int i = 0; i < p; i++) {
    x[i] = mean[i];
    for (int j = 0; j <= i; j++) {
      x[i] += L(i, j) * z[j];
    }
  }

  return x;
}

// ---------------------------------------------------------------------
// Update functions
// ---------------------------------------------------------------------

// Update beta (fixed effects)
NumericVector update_beta(
    const NumericVector& kappa,
    const NumericVector& omega,
    const NumericMatrix& X,
    const NumericVector& re_contrib,
    double prior_sd
) {
  int n = X.nrow();
  int p = X.ncol();
  double prior_prec = 1.0 / (prior_sd * prior_sd);

  // Compute posterior parameters
  List post = compute_posterior_normal(X, omega, kappa, re_contrib, prior_prec);
  NumericMatrix XWX = post["XWX"];
  NumericVector XWkappa = post["XWkappa"];

  // Cholesky decomposition
  NumericMatrix L = chol_decomp(XWX);

  // Posterior mean: solve XWX * mean = XWkappa
  NumericVector post_mean = chol_solve(L, XWkappa);

  // Sample from posterior
  // Need to sample from N(post_mean, XWX^{-1})
  // XWX^{-1} = (L L')^{-1} = L'^{-1} L^{-1}
  // So sample z ~ N(0, I), compute L^{-1} z, add to mean

  // More efficient: solve L z_star = z for z_star, then x = mean + z_star
  NumericVector z(p);
  for (int i = 0; i < p; i++) {
    z[i] = R::rnorm(0.0, 1.0);
  }

  // Solve L z_star = z (forward substitution only)
  NumericVector z_star(p);
  for (int i = 0; i < p; i++) {
    double sum = z[i];
    for (int j = 0; j < i; j++) {
      sum -= L(i, j) * z_star[j];
    }
    z_star[i] = sum / L(i, i);
  }

  // x = mean + z_star
  NumericVector beta(p);
  for (int i = 0; i < p; i++) {
    beta[i] = post_mean[i] + z_star[i];
  }

  return beta;
}

// Update random effects (blocked by group)
NumericVector update_re(
    const NumericVector& kappa,
    const NumericVector& omega,
    const NumericVector& X_beta,
    const IntegerVector& group,
    int n_groups,
    double sigma_re
) {
  int n = kappa.size();
  NumericVector re(n_groups);

  // For each group, compute posterior
  // Observations in group g: Y_g = X_beta_g + b_g + error
  // With PG augmentation: kappa_g/omega_g = X_beta_g + b_g + N(0, 1/omega_g)
  // Prior: b_g ~ N(0, sigma_re^2)
  //
  // Posterior: b_g | ... ~ N(m_g, v_g)
  // v_g = (sum(omega_g) + 1/sigma_re^2)^{-1}
  // m_g = v_g * sum(kappa_g - omega_g * X_beta_g)

  double prior_prec = 1.0 / (sigma_re * sigma_re + 1e-10);

  // Accumulate sufficient statistics by group
  NumericVector sum_omega(n_groups);
  NumericVector sum_resid(n_groups);

  for (int i = 0; i < n; i++) {
    int g = group[i] - 1;  // Convert to 0-based
    sum_omega[g] += omega[i];
    sum_resid[g] += kappa[i] - omega[i] * X_beta[i];
  }

  // Sample from posterior
  for (int g = 0; g < n_groups; g++) {
    double post_var = 1.0 / (sum_omega[g] + prior_prec);
    double post_mean = post_var * sum_resid[g];
    re[g] = R::rnorm(post_mean, std::sqrt(post_var));
  }

  return re;
}

// Update sigma_re with half-Cauchy prior
// p(sigma | b) ∝ p(b | sigma) * p(sigma)
// Using auxiliary variable approach for half-Cauchy
double update_sigma_re(
    const NumericVector& re,
    double scale
) {
  int J = re.size();

  // Sum of squared random effects
  double ss = 0.0;
  for (int j = 0; j < J; j++) {
    ss += re[j] * re[j];
  }

  // Half-Cauchy prior: sample via inverse-gamma auxiliary
  // sigma^2 | b, aux ~ IG((J+1)/2, ss/2 + 1/aux)
  // aux | sigma^2 ~ IG(1, 1/scale^2 + 1/sigma^2)

  // For simplicity, use direct sampling with truncated normal on log scale
  // or inverse gamma approximation

  // Simple approach: use conjugate inverse-gamma with weakly informative params
  // sigma^2 ~ IG(0.5, 0.5 * scale^2) approximately matches half-Cauchy for moderate scale

  // Posterior: sigma^2 | b ~ IG((J + 1)/2, ss/2 + 0.5 * scale^2)
  double shape = (J + 1.0) / 2.0;
  double rate = ss / 2.0 + 0.5 * scale * scale;

  double sigma_sq = 1.0 / R::rgamma(shape, 1.0 / rate);
  return std::sqrt(sigma_sq);
}

// ---------------------------------------------------------------------
// Main Gibbs sampler
// ---------------------------------------------------------------------

List pg_binomial_gibbs_impl(
    IntegerVector y,
    IntegerVector n,
    NumericMatrix X,
    IntegerVector group,
    int n_groups,
    int n_iter,
    int n_warmup,
    int thin,
    double prior_beta_sd,
    double prior_sigma_scale,
    bool store_eta,
    bool verbose,
    int n_threads
) {
  int N = y.size();
  int p = X.ncol();
  int n_save = (n_iter - n_warmup) / thin;

  ratiod_omp::ScopedThreadCount thread_scope(n_threads);

  // Storage
  NumericMatrix beta_draws(n_save, p);
  NumericMatrix re_draws(n_save, n_groups);
  NumericVector sigma_draws(n_save);
  NumericMatrix eta_draws;
  if (store_eta) {
    eta_draws = NumericMatrix(n_save, N);
  }

  // Initialize
  NumericVector beta(p, 0.0);
  NumericVector re(n_groups, 0.0);
  double sigma_re = 1.0;
  NumericVector omega(N, 1.0);  // PG draws
  NumericVector kappa(N);       // y - n/2
  NumericVector eta(N);         // Linear predictor
  NumericVector X_beta(N);      // X * beta
  NumericVector re_contrib(N);  // Random effects contribution

  // Compute kappa = y - n/2
  for (int i = 0; i < N; i++) {
    kappa[i] = y[i] - n[i] / 2.0;
  }

  // Gibbs iterations
  int save_idx = 0;
  for (int iter = 0; iter < n_iter; iter++) {
    // 1. Compute linear predictor (parallelized)
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int i = 0; i < N; i++) {
      X_beta[i] = 0.0;
      for (int j = 0; j < p; j++) {
        X_beta[i] += X(i, j) * beta[j];
      }
      // Only access re if we have random effects
      if (n_groups > 0) {
        re_contrib[i] = re[group[i] - 1];  // group is 1-based
      } else {
        re_contrib[i] = 0.0;
      }
      eta[i] = X_beta[i] + re_contrib[i];
    }

    // 2. Sample omega ~ PG(n, eta)
    // Note: NOT parallelized - R's RNG is not thread-safe
    for (int i = 0; i < N; i++) {
      omega[i] = rpg_int(n[i], eta[i]);
    }

    // 3. Update beta | omega, re, y
    beta = update_beta(kappa, omega, X, re_contrib, prior_beta_sd);

    // 4. Recompute X_beta after beta update (parallelized)
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int i = 0; i < N; i++) {
      X_beta[i] = 0.0;
      for (int j = 0; j < p; j++) {
        X_beta[i] += X(i, j) * beta[j];
      }
    }

    // 5. Update random effects | omega, beta, sigma_re
    if (n_groups > 0) {
      re = update_re(kappa, omega, X_beta, group, n_groups, sigma_re);

      // 6. Update sigma_re | re
      sigma_re = update_sigma_re(re, prior_sigma_scale);
    }

    // Save draws (after warmup, respecting thinning)
    if (iter >= n_warmup && (iter - n_warmup) % thin == 0) {
      for (int j = 0; j < p; j++) {
        beta_draws(save_idx, j) = beta[j];
      }
      for (int g = 0; g < n_groups; g++) {
        re_draws(save_idx, g) = re[g];
      }
      sigma_draws[save_idx] = sigma_re;

      if (store_eta) {
        for (int i = 0; i < N; i++) {
          eta_draws(save_idx, i) = eta[i];
        }
      }
      save_idx++;
    }

    // Progress
    if (verbose && (iter + 1) % 500 == 0) {
      Rcpp::Rcout << "Iteration " << (iter + 1) << "/" << n_iter << std::endl;
    }

    // Check for user interrupt
    if ((iter + 1) % 100 == 0) {
      Rcpp::checkUserInterrupt();
    }
  }

  List result = List::create(
    Named("beta") = beta_draws,
    Named("re") = re_draws,
    Named("sigma_re") = sigma_draws
  );

  if (store_eta) {
    result["eta"] = eta_draws;
  }

  return result;
}

} // namespace ratiod

// ---------------------------------------------------------------------
// R exports
// ---------------------------------------------------------------------

// [[Rcpp::export]]
Rcpp::List cpp_pg_binomial_gibbs(
    Rcpp::IntegerVector y,
    Rcpp::IntegerVector n,
    Rcpp::NumericMatrix X,
    Rcpp::IntegerVector group,
    int n_groups,
    int n_iter = 2000,
    int n_warmup = 1000,
    int thin = 1,
    double prior_beta_sd = 10.0,
    double prior_sigma_scale = 2.5,
    bool store_eta = false,
    bool verbose = true,
    int n_threads = 1
) {
  // CRITICAL: Must call GetRNGstate/PutRNGstate when using R's RNG from C++
  GetRNGstate();
  Rcpp::List result = ratiod::pg_binomial_gibbs_impl(
    y, n, X, group, n_groups,
    n_iter, n_warmup, thin,
    prior_beta_sd, prior_sigma_scale,
    store_eta, verbose, n_threads
  );
  PutRNGstate();
  return result;
}

// [[Rcpp::export]]
int cpp_pg_get_max_threads() {
  #ifdef _OPENMP
  return omp_get_max_threads();
  #else
  return 1;
  #endif
}

// Forward declaration
namespace ratiod {
  Rcpp::NumericVector update_spatial_icar(
      const Rcpp::NumericVector& kappa,
      const Rcpp::NumericVector& omega,
      const Rcpp::NumericVector& offset,
      const Rcpp::IntegerVector& group,
      const Rcpp::List& adj_list,
      const Rcpp::IntegerVector& n_neighbors,
      double tau
  );

  double update_tau_icar(
      const Rcpp::NumericVector& phi,
      const Rcpp::List& adj_list,
      const Rcpp::IntegerVector& n_neighbors,
      double prior_shape,
      double prior_rate
  );
}

// Binomial Gibbs sampler with random effects AND spatial effects (ICAR)
// [[Rcpp::export]]
Rcpp::List cpp_pg_binomial_gibbs_spatial(
    Rcpp::IntegerVector y,
    Rcpp::IntegerVector n,
    Rcpp::NumericMatrix X,
    Rcpp::IntegerVector re_group,
    int n_re_groups,
    Rcpp::IntegerVector spatial_group,
    int n_spatial_units,
    Rcpp::List adj_list,
    Rcpp::IntegerVector n_neighbors,
    int n_iter = 2000,
    int n_warmup = 1000,
    int thin = 1,
    double prior_beta_sd = 10.0,
    double prior_sigma_re_scale = 2.5,
    double prior_tau_shape = 1.0,
    double prior_tau_rate = 0.01,
    bool store_eta = false,
    bool verbose = true,
    int n_threads = 1
) {
  // CRITICAL: Must call GetRNGstate/PutRNGstate when using R's RNG from C++
  GetRNGstate();

  int N = y.size();
  int p = X.ncol();
  int n_save = (n_iter - n_warmup) / thin;

  ratiod_omp::ScopedThreadCount thread_scope(n_threads);

  // Storage
  Rcpp::NumericMatrix beta_draws(n_save, p);
  Rcpp::NumericMatrix re_draws(n_save, n_re_groups);
  Rcpp::NumericVector sigma_re_draws(n_save);
  Rcpp::NumericMatrix spatial_draws(n_save, n_spatial_units);
  Rcpp::NumericVector tau_draws(n_save);
  Rcpp::NumericMatrix eta_draws;
  if (store_eta) {
    eta_draws = Rcpp::NumericMatrix(n_save, N);
  }

  // Initialize
  Rcpp::NumericVector beta(p, 0.0);
  Rcpp::NumericVector re(n_re_groups, 0.0);
  double sigma_re = 1.0;
  Rcpp::NumericVector phi(n_spatial_units, 0.0);  // Spatial effects
  double tau = 1.0;  // Spatial precision
  Rcpp::NumericVector omega(N, 1.0);
  Rcpp::NumericVector kappa(N);
  Rcpp::NumericVector eta(N);
  Rcpp::NumericVector X_beta(N);
  Rcpp::NumericVector re_contrib(N);
  Rcpp::NumericVector spatial_contrib(N);
  Rcpp::NumericVector offset(N);

  // Compute kappa = y - n/2
  for (int i = 0; i < N; i++) {
    kappa[i] = y[i] - n[i] / 2.0;
  }

  // Gibbs iterations
  int save_idx = 0;
  for (int iter = 0; iter < n_iter; iter++) {
    // 1. Compute linear predictor (parallelized)
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int i = 0; i < N; i++) {
      X_beta[i] = 0.0;
      for (int j = 0; j < p; j++) {
        X_beta[i] += X(i, j) * beta[j];
      }
      re_contrib[i] = (n_re_groups > 0) ? re[re_group[i] - 1] : 0.0;
      spatial_contrib[i] = phi[spatial_group[i] - 1];
      eta[i] = X_beta[i] + re_contrib[i] + spatial_contrib[i];
    }

    // 2. Sample omega ~ PG(n, eta)
    // Note: NOT parallelized - R's RNG is not thread-safe
    for (int i = 0; i < N; i++) {
      omega[i] = ratiod::rpg_int(n[i], eta[i]);
    }

    // 3. Update beta | omega, re, phi, y
    // Offset for beta update = re + phi (parallelized)
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int i = 0; i < N; i++) {
      offset[i] = re_contrib[i] + spatial_contrib[i];
    }
    beta = ratiod::update_beta(kappa, omega, X, offset, prior_beta_sd);

    // 4. Recompute X_beta (parallelized)
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int i = 0; i < N; i++) {
      X_beta[i] = 0.0;
      for (int j = 0; j < p; j++) {
        X_beta[i] += X(i, j) * beta[j];
      }
    }

    // 5. Update random effects | omega, beta, phi, sigma_re
    if (n_re_groups > 0) {
      // Offset for RE update = X*beta + phi (parallelized)
      #ifdef _OPENMP
      #pragma omp parallel for schedule(static)
      #endif
      for (int i = 0; i < N; i++) {
        offset[i] = X_beta[i] + spatial_contrib[i];
      }
      re = ratiod::update_re(kappa, omega, offset, re_group, n_re_groups, sigma_re);

      // Update sigma_re
      sigma_re = ratiod::update_sigma_re(re, prior_sigma_re_scale);

      // Update re_contrib (parallelized)
      #ifdef _OPENMP
      #pragma omp parallel for schedule(static)
      #endif
      for (int i = 0; i < N; i++) {
        re_contrib[i] = re[re_group[i] - 1];
      }
    }

    // 6. Update spatial effects | omega, beta, re, tau
    // Offset for spatial update = X*beta + re (parallelized)
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int i = 0; i < N; i++) {
      offset[i] = X_beta[i] + re_contrib[i];
    }
    phi = ratiod::update_spatial_icar(kappa, omega, offset, spatial_group, adj_list, n_neighbors, tau);

    // 7. Update tau (spatial precision)
    tau = ratiod::update_tau_icar(phi, adj_list, n_neighbors, prior_tau_shape, prior_tau_rate);

    // Save draws
    if (iter >= n_warmup && (iter - n_warmup) % thin == 0) {
      for (int j = 0; j < p; j++) {
        beta_draws(save_idx, j) = beta[j];
      }
      for (int g = 0; g < n_re_groups; g++) {
        re_draws(save_idx, g) = re[g];
      }
      sigma_re_draws[save_idx] = sigma_re;
      for (int s = 0; s < n_spatial_units; s++) {
        spatial_draws(save_idx, s) = phi[s];
      }
      tau_draws[save_idx] = tau;

      if (store_eta) {
        for (int i = 0; i < N; i++) {
          eta_draws(save_idx, i) = eta[i];
        }
      }
      save_idx++;
    }

    // Progress
    if (verbose && (iter + 1) % 500 == 0) {
      Rcpp::Rcout << "Iteration " << (iter + 1) << "/" << n_iter << std::endl;
    }

    // Check for user interrupt
    if ((iter + 1) % 100 == 0) {
      Rcpp::checkUserInterrupt();
    }
  }

  Rcpp::List result = Rcpp::List::create(
    Rcpp::Named("beta") = beta_draws,
    Rcpp::Named("re") = re_draws,
    Rcpp::Named("sigma_re") = sigma_re_draws,
    Rcpp::Named("spatial") = spatial_draws,
    Rcpp::Named("tau") = tau_draws
  );

  if (store_eta) {
    result["eta"] = eta_draws;
  }

  PutRNGstate();
  return result;
}

// ---------------------------------------------------------------------
// BYM2 Spatial Gibbs sampler
// ---------------------------------------------------------------------

// Forward declaration for BYM2 functions
namespace ratiod {
  Rcpp::NumericVector update_spatial_bym2(
      const Rcpp::NumericVector& kappa,
      const Rcpp::NumericVector& omega,
      const Rcpp::NumericVector& offset,
      const Rcpp::IntegerVector& group,
      const Rcpp::List& adj_list,
      const Rcpp::IntegerVector& n_neighbors,
      Rcpp::NumericVector& phi_scaled,
      Rcpp::NumericVector& theta,
      double sigma_spatial,
      double rho,
      double scale_factor
  );

  double update_sigma_spatial(
      const Rcpp::NumericVector& u,
      double scale
  );

  double update_rho_bym2(
      const Rcpp::NumericVector& phi_scaled,
      const Rcpp::NumericVector& theta,
      double sigma_spatial,
      double scale_factor,
      const Rcpp::NumericVector& sum_omega,
      const Rcpp::NumericVector& sum_resid,
      double alpha,
      double beta
  );
}

// Binomial Gibbs sampler with random effects AND spatial effects (BYM2)
// [[Rcpp::export]]
Rcpp::List cpp_pg_binomial_gibbs_bym2(
    Rcpp::IntegerVector y,
    Rcpp::IntegerVector n,
    Rcpp::NumericMatrix X,
    Rcpp::IntegerVector re_group,
    int n_re_groups,
    Rcpp::IntegerVector spatial_group,
    int n_spatial_units,
    Rcpp::List adj_list,
    Rcpp::IntegerVector n_neighbors,
    double scale_factor,
    int n_iter = 2000,
    int n_warmup = 1000,
    int thin = 1,
    double prior_beta_sd = 10.0,
    double prior_sigma_re_scale = 2.5,
    double prior_sigma_spatial_scale = 2.5,
    double prior_rho_alpha = 0.5,
    double prior_rho_beta = 0.5,
    bool store_eta = false,
    bool verbose = true,
    int n_threads = 1
) {
  // CRITICAL: Must call GetRNGstate/PutRNGstate when using R's RNG from C++
  GetRNGstate();

  int N = y.size();
  int p = X.ncol();
  int n_save = (n_iter - n_warmup) / thin;

  ratiod_omp::ScopedThreadCount thread_scope(n_threads);

  // Storage
  Rcpp::NumericMatrix beta_draws(n_save, p);
  Rcpp::NumericMatrix re_draws(n_save, n_re_groups);
  Rcpp::NumericVector sigma_re_draws(n_save);
  Rcpp::NumericMatrix phi_scaled_draws(n_save, n_spatial_units);
  Rcpp::NumericMatrix theta_draws(n_save, n_spatial_units);
  Rcpp::NumericMatrix u_draws(n_save, n_spatial_units);  // Combined spatial effect
  Rcpp::NumericVector sigma_spatial_draws(n_save);
  Rcpp::NumericVector rho_draws(n_save);
  Rcpp::NumericMatrix eta_draws;
  if (store_eta) {
    eta_draws = Rcpp::NumericMatrix(n_save, N);
  }

  // Initialize
  Rcpp::NumericVector beta(p, 0.0);
  Rcpp::NumericVector re(n_re_groups, 0.0);
  double sigma_re = 1.0;
  Rcpp::NumericVector phi_scaled(n_spatial_units, 0.0);  // Structured component
  Rcpp::NumericVector theta(n_spatial_units, 0.0);       // Unstructured component
  Rcpp::NumericVector u(n_spatial_units, 0.0);           // Combined effect
  double sigma_spatial = 1.0;
  double rho = 0.5;  // Start with equal mix
  Rcpp::NumericVector omega(N, 1.0);
  Rcpp::NumericVector kappa(N);
  Rcpp::NumericVector eta(N);
  Rcpp::NumericVector X_beta(N);
  Rcpp::NumericVector re_contrib(N);
  Rcpp::NumericVector spatial_contrib(N);
  Rcpp::NumericVector offset(N);

  // Compute kappa = y - n/2
  for (int i = 0; i < N; i++) {
    kappa[i] = y[i] - n[i] / 2.0;
  }

  // Gibbs iterations
  int save_idx = 0;
  for (int iter = 0; iter < n_iter; iter++) {
    // 1. Compute linear predictor (parallelized)
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int i = 0; i < N; i++) {
      X_beta[i] = 0.0;
      for (int j = 0; j < p; j++) {
        X_beta[i] += X(i, j) * beta[j];
      }
      re_contrib[i] = (n_re_groups > 0) ? re[re_group[i] - 1] : 0.0;
      spatial_contrib[i] = u[spatial_group[i] - 1];
      eta[i] = X_beta[i] + re_contrib[i] + spatial_contrib[i];
    }

    // 2. Sample omega ~ PG(n, eta)
    // Note: NOT parallelized - R's RNG is not thread-safe
    for (int i = 0; i < N; i++) {
      omega[i] = ratiod::rpg_int(n[i], eta[i]);
    }

    // 3. Update beta | omega, re, u, y
    // Offset for beta update = re + u (parallelized)
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int i = 0; i < N; i++) {
      offset[i] = re_contrib[i] + spatial_contrib[i];
    }
    beta = ratiod::update_beta(kappa, omega, X, offset, prior_beta_sd);

    // 4. Recompute X_beta (parallelized)
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int i = 0; i < N; i++) {
      X_beta[i] = 0.0;
      for (int j = 0; j < p; j++) {
        X_beta[i] += X(i, j) * beta[j];
      }
    }

    // 5. Update random effects | omega, beta, u, sigma_re
    if (n_re_groups > 0) {
      // Offset for RE update = X*beta + u (parallelized)
      #ifdef _OPENMP
      #pragma omp parallel for schedule(static)
      #endif
      for (int i = 0; i < N; i++) {
        offset[i] = X_beta[i] + spatial_contrib[i];
      }
      re = ratiod::update_re(kappa, omega, offset, re_group, n_re_groups, sigma_re);

      // Update sigma_re
      sigma_re = ratiod::update_sigma_re(re, prior_sigma_re_scale);

      // Update re_contrib (parallelized)
      #ifdef _OPENMP
      #pragma omp parallel for schedule(static)
      #endif
      for (int i = 0; i < N; i++) {
        re_contrib[i] = re[re_group[i] - 1];
      }
    }

    // 6. Update BYM2 spatial effects | omega, beta, re, sigma_spatial, rho
    // Offset for spatial update = X*beta + re (parallelized)
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int i = 0; i < N; i++) {
      offset[i] = X_beta[i] + re_contrib[i];
    }
    u = ratiod::update_spatial_bym2(kappa, omega, offset, spatial_group, adj_list, n_neighbors,
                                   phi_scaled, theta, sigma_spatial, rho, scale_factor);

    // 7. Update sigma_spatial
    sigma_spatial = ratiod::update_sigma_spatial(u, prior_sigma_spatial_scale);

    // 8. Update rho (mixing proportion)
    // Need to compute sum_omega and sum_resid for rho update
    Rcpp::NumericVector sum_omega_s(n_spatial_units, 0.0);
    Rcpp::NumericVector sum_resid_s(n_spatial_units, 0.0);
    for (int i = 0; i < N; i++) {
      int s = spatial_group[i] - 1;
      sum_omega_s[s] += omega[i];
      sum_resid_s[s] += kappa[i] - omega[i] * offset[i];
    }
    rho = ratiod::update_rho_bym2(phi_scaled, theta, sigma_spatial, scale_factor,
                                  sum_omega_s, sum_resid_s, prior_rho_alpha, prior_rho_beta);

    // Update spatial contributions (parallelized)
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int i = 0; i < N; i++) {
      spatial_contrib[i] = u[spatial_group[i] - 1];
    }

    // Save draws
    if (iter >= n_warmup && (iter - n_warmup) % thin == 0) {
      for (int j = 0; j < p; j++) {
        beta_draws(save_idx, j) = beta[j];
      }
      for (int g = 0; g < n_re_groups; g++) {
        re_draws(save_idx, g) = re[g];
      }
      sigma_re_draws[save_idx] = sigma_re;
      for (int s = 0; s < n_spatial_units; s++) {
        phi_scaled_draws(save_idx, s) = phi_scaled[s];
        theta_draws(save_idx, s) = theta[s];
        u_draws(save_idx, s) = u[s];
      }
      sigma_spatial_draws[save_idx] = sigma_spatial;
      rho_draws[save_idx] = rho;

      if (store_eta) {
        for (int i = 0; i < N; i++) {
          eta_draws(save_idx, i) = eta[i];
        }
      }
      save_idx++;
    }

    // Progress
    if (verbose && (iter + 1) % 500 == 0) {
      Rcpp::Rcout << "Iteration " << (iter + 1) << "/" << n_iter << std::endl;
    }

    // Check for user interrupt
    if ((iter + 1) % 100 == 0) {
      Rcpp::checkUserInterrupt();
    }
  }

  Rcpp::List result = Rcpp::List::create(
    Rcpp::Named("beta") = beta_draws,
    Rcpp::Named("re") = re_draws,
    Rcpp::Named("sigma_re") = sigma_re_draws,
    Rcpp::Named("phi_scaled") = phi_scaled_draws,
    Rcpp::Named("theta") = theta_draws,
    Rcpp::Named("spatial") = u_draws,
    Rcpp::Named("sigma_spatial") = sigma_spatial_draws,
    Rcpp::Named("rho") = rho_draws
  );

  if (store_eta) {
    result["eta"] = eta_draws;
  }

  PutRNGstate();
  return result;
}

// ---------------------------------------------------------------------
// GP Spatial Gibbs Sampler for Binomial Models
// Uses sequential NNGP updates
// ---------------------------------------------------------------------

// The neighbour structure of one NNGP field, in the convention
// compute_nngp_neighbors() writes and backend_pg.R passes:
//
//   nn_order[k]   the 0-based LOCATION index sitting at position k of the
//                 conditioning order, so w and the per-location likelihood
//                 accumulators are indexed by nn_order[k] and never by k;
//   nn_idx(k, j)  the 1-based POSITION in that order of neighbour j of the
//                 location at position k, 0 where it has fewer than nn
//                 predecessors. That neighbour's location is nn_order[nn_idx-1];
//   nn_dist(k, j) the distance to that neighbour;
//   coords        one row per LOCATION, so the unique coordinates rather than
//                 the observation-order matrix.
//
// Every consumer reads the same struct. A second entry re-deriving the
// convention is how the multiscale sweep came to index nn_idx by location and
// read its values as locations, two mismatches in one expression.
struct PgNngpGraph {
  const Rcpp::NumericMatrix& coords;
  const Rcpp::IntegerMatrix& nn_idx;
  const Rcpp::NumericMatrix& nn_dist;
  const Rcpp::IntegerVector& nn_order;
  int nn;
  int n_spatial;
};

// Refuse a graph whose shapes or index ranges do not match that convention, at
// the entry point, rather than reading past an array inside a sweep. This path
// once took nn_order 1-based and used it directly to index w, coords and the
// accumulators, all sized n_spatial.
inline void pg_check_nngp_graph(const PgNngpGraph& g, const char* what) {
  if (g.n_spatial <= 0) {
    Rcpp::stop("%s: n_spatial is %d; the NNGP needs at least one location.",
               what, g.n_spatial);
  }
  if (g.coords.nrow() != g.n_spatial) {
    Rcpp::stop("%s: coords has %d rows against n_spatial = %d. The neighbour "
               "structure is built on the unique locations, so coords carries "
               "one row per location.",
               what, static_cast<int>(g.coords.nrow()), g.n_spatial);
  }
  if (g.coords.ncol() < 2) {
    Rcpp::stop("%s: coords has %d columns; two are required.",
               what, static_cast<int>(g.coords.ncol()));
  }
  if (g.nn_order.size() != g.n_spatial) {
    Rcpp::stop("%s: nn_order has length %d against n_spatial = %d.",
               what, static_cast<int>(g.nn_order.size()), g.n_spatial);
  }
  if (g.nn < 0 || g.nn_idx.nrow() != g.n_spatial || g.nn_idx.ncol() < g.nn ||
      g.nn_dist.nrow() != g.n_spatial || g.nn_dist.ncol() < g.nn) {
    Rcpp::stop("%s: nn_idx and nn_dist must each be %d x %d.",
               what, g.n_spatial, g.nn);
  }

  std::vector<bool> seen(static_cast<size_t>(g.n_spatial), false);
  for (int k = 0; k < g.n_spatial; k++) {
    const int loc = g.nn_order[k];
    if (loc < 0 || loc >= g.n_spatial) {
      Rcpp::stop("%s: nn_order[%d] is %d, outside [0, %d). It is a 0-based "
                 "location index.", what, k, loc, g.n_spatial);
    }
    if (seen[static_cast<size_t>(loc)]) {
      Rcpp::stop("%s: nn_order visits location %d twice; it is a permutation "
                 "of the locations.", what, loc);
    }
    seen[static_cast<size_t>(loc)] = true;

    for (int j = 0; j < g.nn; j++) {
      const int pos = g.nn_idx(k, j);
      // 0 means "no neighbour here"; anything else is a position strictly
      // earlier in the order, which is what makes the factorization sequential.
      if (pos < 0 || pos > k) {
        Rcpp::stop("%s: nn_idx(%d, %d) is %d; a neighbour is a 1-based "
                   "position in [1, %d], or 0 for absent.", what, k, j, pos, k);
      }
    }
  }
}

// obs_to_loc[i] is the location observation i was measured at. It is what maps
// the two sides of the model to each other, and the only thing that can: a
// location carries as many observations as the design gave it, in no particular
// row order.
inline void pg_check_obs_to_loc(const Rcpp::IntegerVector& obs_to_loc, int N,
                                int n_spatial, const char* what) {
  if (obs_to_loc.size() != N) {
    Rcpp::stop("%s: obs_to_loc has length %d against N = %d.",
               what, static_cast<int>(obs_to_loc.size()), N);
  }
  for (int i = 0; i < N; i++) {
    const int loc = obs_to_loc[i];
    if (loc < 0 || loc >= n_spatial) {
      Rcpp::stop("%s: obs_to_loc[%d] is %d, outside [0, %d). It is a 0-based "
                 "location index.", what, i, loc, n_spatial);
    }
  }
}

// Helper: compute NNGP conditional mean and variance
inline void pg_nngp_conditional(
    int i,
    const std::vector<double>& w,
    double sigma2,
    double phi_gp,
    ratiod_cov::CovType cov_type,
    const PgNngpGraph& g,
    double& cond_mean,
    double& cond_var
) {
  const Rcpp::NumericMatrix& coords = g.coords;
  const Rcpp::IntegerMatrix& nn_idx = g.nn_idx;
  const Rcpp::NumericMatrix& nn_dist = g.nn_dist;
  const Rcpp::IntegerVector& nn_order = g.nn_order;
  const int nn = g.nn;

  int n_neighbors = 0;
  for (int j = 0; j < nn; j++) {
    if (nn_idx(i, j) > 0) n_neighbors++;
  }

  if (n_neighbors == 0) {
    cond_mean = 0.0;
    cond_var = sigma2;
    return;
  }

  // The shared covariance family, so this path answers to the same four
  // spatial_gp(cov = ) choices every other one does. Each kernel returns
  // sigma2 at d = 0, so the zero-distance case needs no special arm.
  auto compute_cov = [sigma2, phi_gp, cov_type](double d) {
    return ratiod_cov::compute_cov(d, sigma2, phi_gp, cov_type);
  };

  std::vector<double> c_vec(n_neighbors);
  std::vector<double> C_mat(n_neighbors * n_neighbors);

  for (int j = 0; j < n_neighbors; j++) {
    c_vec[j] = compute_cov(nn_dist(i, j));
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
        C_mat[j1 * n_neighbors + j2] = compute_cov(d12);
      }
    }
  }

  // The shared neighbour-block factorization. A block it cannot factor falls
  // back to the marginal, which is what the no-neighbour case above returns.
  std::vector<double> L, y_sol, alpha;
  if (!ratiod_cov::nngp_chol(C_mat, n_neighbors, L)) {
    cond_mean = 0.0;
    cond_var = sigma2;
    return;
  }
  ratiod_cov::nngp_forward_solve(L, n_neighbors, c_vec, y_sol);
  ratiod_cov::nngp_back_solve(L, n_neighbors, y_sol, alpha);

  cond_mean = 0.0;
  for (int j = 0; j < n_neighbors; j++) {
    int nn_orig = nn_order[nn_idx(i, j) - 1];
    cond_mean += alpha[j] * w[nn_orig];
  }

  double c_Cinv_c = 0.0;
  for (int j = 0; j < n_neighbors; j++) {
    c_Cinv_c += c_vec[j] * alpha[j];
  }
  cond_var = ratiod_cov::nngp_floor_cond_var(sigma2 - c_Cinv_c);
}

// log p(w | sigma2, phi) under the NNGP, taken in the sampler's own
// conditioning order and from the same conditionals its field sweep draws
// from, so the hyperparameters are conditioned on the prior the field was
// actually drawn under.
//
//   log p(w) = sum_k log N(w_{o(k)} ; mu_k, d_k)
//
// The log-determinant half of that, sum_k -0.5 log d_k, is what an independent
// N(0, sigma2) form leaves out: without it raising sigma2 only ever shrinks the
// quadratic penalty, so the acceptance ratio ratchets one way and sigma2 runs
// away.
inline double pg_nngp_log_density(
    const std::vector<double>& w,
    double sigma2,
    double phi_gp,
    ratiod_cov::CovType cov_type,
    const PgNngpGraph& g
) {
  const double log_2pi = std::log(2.0 * M_PI);
  double ll = 0.0;
  for (int k = 0; k < g.n_spatial; k++) {
    double cond_mean = 0.0, cond_var = 0.0;
    pg_nngp_conditional(k, w, sigma2, phi_gp, cov_type, g, cond_mean, cond_var);
    const double resid = w[g.nn_order[k]] - cond_mean;
    ll += -0.5 * (log_2pi + std::log(cond_var)) - 0.5 * resid * resid / cond_var;
  }
  return ll;
}

// The Polya-Gamma likelihood at one location is the sum over every observation
// measured THERE, which obs_to_loc is what names. Both this and the scatter
// below used to run over observation ROW POSITION and drop every row past
// n_spatial, so on a design with several observations per location most of the
// data reached no location and the rest reached the wrong one.
inline void pg_accumulate_by_location(
    int N,
    const Rcpp::IntegerVector& obs_to_loc,
    const Rcpp::NumericVector& omega,
    const Rcpp::NumericVector& kappa,
    const Rcpp::NumericVector& offset,
    std::vector<double>& sum_omega,
    std::vector<double>& sum_resid
) {
  std::fill(sum_omega.begin(), sum_omega.end(), 0.0);
  std::fill(sum_resid.begin(), sum_resid.end(), 0.0);
  for (int i = 0; i < N; i++) {
    const int loc = obs_to_loc[i];
    sum_omega[loc] += omega[i];
    sum_resid[loc] += kappa[i] - omega[i] * offset[i];
  }
}

// The field's contribution to every observation's linear predictor.
inline void pg_scatter_by_location(
    int N,
    const Rcpp::IntegerVector& obs_to_loc,
    const std::vector<double>& w,
    Rcpp::NumericVector& contrib
) {
  for (int i = 0; i < N; i++) contrib[i] = w[obs_to_loc[i]];
}

// One sequential NNGP Gibbs sweep over a field, conjugate given the
// Polya-Gamma weights: the conditional prior N(mu_k, d_k) meets the per-location
// Gaussian likelihood the accumulators carry.
inline void pg_nngp_field_sweep(
    std::vector<double>& w,
    double sigma2,
    double phi_gp,
    ratiod_cov::CovType cov_type,
    const PgNngpGraph& g,
    const std::vector<double>& sum_omega,
    const std::vector<double>& sum_resid
) {
  for (int k = 0; k < g.n_spatial; k++) {
    const int loc = g.nn_order[k];
    double cond_mean = 0.0, cond_var = 0.0;
    pg_nngp_conditional(k, w, sigma2, phi_gp, cov_type, g, cond_mean, cond_var);

    const double tau_prior = 1.0 / cond_var;
    const double tau_post = tau_prior + sum_omega[loc];
    const double mean_post = (tau_prior * cond_mean + sum_resid[loc]) / tau_post;
    w[loc] = R::rnorm(mean_post, 1.0 / std::sqrt(tau_post));
  }
}

// One Metropolis-Hastings sweep over an NNGP field's (sigma2, phi). Both are
// random walks on the log scale, so each acceptance ratio carries the
// log-Jacobian of its own transform alongside the density difference.
//
// sigma2's prior is the PC prior on the standard deviation sigma = sqrt(sigma2),
// P(sigma > U) = alpha, i.e. Exponential(lambda) with lambda = -log(alpha) / U.
// In log(sigma2) its density is
//
//   -lambda sqrt(s2) - 0.5 log(s2)   (change of variable to s2)
//   + log(s2)                        (the walk's own Jacobian)
//
// which is the +0.5 log(s2) below. The earlier form carried a full log(s2) --
// the walk's Jacobian on a prior read as if it were already on sigma2 -- and so
// pushed sigma2 up by half a log-density on top of the missing determinant.
//
// phi's prior is uniform on [lower, upper]; a proposal outside is rejected.
// The phi ratio used to be its Jacobian ALONE, which draws phi from its prior
// rather than from its posterior.
inline void pg_update_gp_hyper(
    double& sigma2,
    double& phi_gp,
    const std::vector<double>& w,
    ratiod_cov::CovType cov_type,
    const PgNngpGraph& g,
    double prior_sigma_U,
    double prior_sigma_alpha,
    double prior_phi_lower,
    double prior_phi_upper,
    double rw_sd
) {
  const double lambda = -std::log(prior_sigma_alpha) / prior_sigma_U;
  auto log_prior_sigma2 = [lambda](double s2) {
    return -lambda * std::sqrt(s2) + 0.5 * std::log(s2);
  };

  double ll_curr = pg_nngp_log_density(w, sigma2, phi_gp, cov_type, g);

  const double sigma2_prop =
    ratiod_linalg::safe_exp(std::log(sigma2) + R::rnorm(0, rw_sd));
  if (std::isfinite(sigma2_prop) && sigma2_prop > 0) {
    const double ll_prop =
      pg_nngp_log_density(w, sigma2_prop, phi_gp, cov_type, g);
    const double log_alpha = (ll_prop + log_prior_sigma2(sigma2_prop)) -
                             (ll_curr + log_prior_sigma2(sigma2));
    if (std::isfinite(log_alpha) && std::log(R::runif(0, 1)) < log_alpha) {
      sigma2 = sigma2_prop;
      // The phi step below conditions on whichever sigma2 survived.
      ll_curr = ll_prop;
    }
  }

  const double phi_prop =
    ratiod_linalg::safe_exp(std::log(phi_gp) + R::rnorm(0, rw_sd));
  if (std::isfinite(phi_prop) && phi_prop >= prior_phi_lower &&
      phi_prop <= prior_phi_upper) {
    const double ll_prop =
      pg_nngp_log_density(w, sigma2, phi_prop, cov_type, g);
    const double log_alpha = ll_prop - ll_curr +
                             std::log(phi_prop) - std::log(phi_gp);
    if (std::isfinite(log_alpha) && std::log(R::runif(0, 1)) < log_alpha) {
      phi_gp = phi_prop;
    }
  }
}

// Drive pg_nngp_conditional at one location, deterministically. The PG GP
// sampler cannot arbitrate which kernel it runs -- its sigma2_gp rails and its
// draws do not reproduce at a fixed seed -- so the conditional is checked
// directly instead, against an independent implementation in R.
// [[Rcpp::export]]
Rcpp::NumericVector cpp_pg_nngp_conditional_probe(
    int i,
    Rcpp::NumericVector w,
    double sigma2,
    double phi_gp,
    int cov_type,
    Rcpp::NumericMatrix coords,
    Rcpp::IntegerMatrix nn_idx,
    Rcpp::NumericMatrix nn_dist,
    Rcpp::IntegerVector nn_order,
    int nn
) {
  const ratiod_cov::CovType cov = ratiod_cov::cov_type_from_int(cov_type);
  const PgNngpGraph g{coords, nn_idx, nn_dist, nn_order, nn,
                      static_cast<int>(nn_order.size())};
  std::vector<double> w_vec(w.begin(), w.end());
  double cond_mean = 0.0, cond_var = 0.0;
  pg_nngp_conditional(i, w_vec, sigma2, phi_gp, cov, g, cond_mean, cond_var);
  return Rcpp::NumericVector::create(Rcpp::Named("cond_mean") = cond_mean,
                                     Rcpp::Named("cond_var") = cond_var);
}

// Drive pg_nngp_log_density, the quantity the (sigma2, phi) acceptance ratios
// are differences of. Same reason as the conditional probe: the sampler cannot
// arbitrate its own hyperparameter step, so the density is scored directly
// against an R reference.
// [[Rcpp::export]]
double cpp_pg_nngp_log_density_probe(
    Rcpp::NumericVector w,
    double sigma2,
    double phi_gp,
    int cov_type,
    Rcpp::NumericMatrix coords,
    Rcpp::IntegerMatrix nn_idx,
    Rcpp::NumericMatrix nn_dist,
    Rcpp::IntegerVector nn_order,
    int nn
) {
  const ratiod_cov::CovType cov = ratiod_cov::cov_type_from_int(cov_type);
  const PgNngpGraph g{coords, nn_idx, nn_dist, nn_order, nn,
                      static_cast<int>(nn_order.size())};
  pg_check_nngp_graph(g, "cpp_pg_nngp_log_density_probe");
  std::vector<double> w_vec(w.begin(), w.end());
  return pg_nngp_log_density(w_vec, sigma2, phi_gp, cov, g);
}

// [[Rcpp::export]]
Rcpp::List cpp_pg_binomial_gibbs_gp(
    Rcpp::IntegerVector y,
    Rcpp::IntegerVector n,
    Rcpp::NumericMatrix X,
    Rcpp::IntegerVector re_group,
    int n_re_groups,
    Rcpp::NumericMatrix coords,
    Rcpp::IntegerMatrix nn_idx,
    Rcpp::NumericMatrix nn_dist,
    Rcpp::IntegerVector nn_order,
    Rcpp::IntegerVector obs_to_loc,
    int n_spatial,
    int nn,
    double sigma2_gp_init,
    double phi_gp_init,
    int cov_type,
    int n_iter = 2000,
    int n_warmup = 1000,
    int thin = 1,
    double prior_beta_sd = 10.0,
    double prior_sigma_re_scale = 2.5,
    double prior_sigma_gp_U = 1.0,
    double prior_sigma_gp_alpha = 0.01,
    double prior_phi_lower = 0.01,
    double prior_phi_upper = 10.0,
    bool store_eta = false,
    bool verbose = true,
    int n_threads = 1
) {
  // CRITICAL: Must call GetRNGstate/PutRNGstate when using R's RNG from C++
  GetRNGstate();

  int N = y.size();
  int p = X.ncol();
  int n_save = (n_iter - n_warmup) / thin;

  const PgNngpGraph graph{coords, nn_idx, nn_dist, nn_order, nn, n_spatial};
  pg_check_nngp_graph(graph, "cpp_pg_binomial_gibbs_gp");
  pg_check_obs_to_loc(obs_to_loc, N, n_spatial, "cpp_pg_binomial_gibbs_gp");

  ratiod_omp::ScopedThreadCount thread_scope(n_threads);

  // Storage
  Rcpp::NumericMatrix beta_draws(n_save, p);
  Rcpp::NumericMatrix re_draws(n_save, n_re_groups);
  Rcpp::NumericVector sigma_re_draws(n_save);
  Rcpp::NumericMatrix gp_draws(n_save, n_spatial);
  Rcpp::NumericVector sigma2_gp_draws(n_save);
  const ratiod_cov::CovType cov = ratiod_cov::cov_type_from_int(cov_type);
  Rcpp::NumericVector phi_gp_draws(n_save);
  Rcpp::NumericMatrix eta_draws_gp;
  if (store_eta) eta_draws_gp = Rcpp::NumericMatrix(n_save, N);

  // Initialize
  Rcpp::NumericVector beta(p, 0.0);
  Rcpp::NumericVector re(n_re_groups, 0.0);
  double sigma_re = 1.0;
  std::vector<double> w(n_spatial, 0.0);
  double sigma2_gp = sigma2_gp_init;
  double phi_gp = phi_gp_init;

  // Working vectors
  Rcpp::NumericVector omega(N);
  Rcpp::NumericVector kappa(N);
  Rcpp::NumericVector eta(N);
  Rcpp::NumericVector X_beta(N);
  Rcpp::NumericVector re_contrib(N);
  Rcpp::NumericVector gp_contrib(N);
  Rcpp::NumericVector offset(N);
  std::vector<double> sum_omega_gp(n_spatial, 0.0);
  std::vector<double> sum_resid_gp(n_spatial, 0.0);

  for (int i = 0; i < N; i++) {
    omega[i] = 0.5;
    kappa[i] = y[i] - 0.5 * n[i];
    X_beta[i] = 0.0;
    re_contrib[i] = 0.0;
    gp_contrib[i] = 0.0;
  }

  int save_idx = 0;

  for (int iter = 0; iter < n_iter; iter++) {
    if (verbose && (iter + 1) % 200 == 0) {
      Rcpp::Rcout << "  Iteration " << (iter + 1) << "/" << n_iter << "\n";
    }

    // 1. Compute eta
    for (int i = 0; i < N; i++) {
      eta[i] = X_beta[i] + re_contrib[i] + gp_contrib[i];
    }

    // 2. Update omega | eta
    for (int i = 0; i < N; i++) {
      omega[i] = ratiod::rpg_int(n[i], eta[i]);
    }

    // 3. Update beta
    for (int i = 0; i < N; i++) {
      offset[i] = re_contrib[i] + gp_contrib[i];
    }
    beta = ratiod::update_beta(kappa, omega, X, offset, prior_beta_sd);

    for (int i = 0; i < N; i++) {
      X_beta[i] = 0.0;
      for (int j = 0; j < p; j++) {
        X_beta[i] += X(i, j) * beta[j];
      }
    }

    // 4. Update random effects
    if (n_re_groups > 0) {
      for (int i = 0; i < N; i++) {
        offset[i] = X_beta[i] + gp_contrib[i];
      }
      re = ratiod::update_re(kappa, omega, offset, re_group, n_re_groups, sigma_re);
      sigma_re = ratiod::update_sigma_re(re, prior_sigma_re_scale);

      for (int i = 0; i < N; i++) {
        re_contrib[i] = re[re_group[i] - 1];
      }
    }

    // 5. Update GP effects (sequential NNGP Gibbs)
    for (int i = 0; i < N; i++) {
      offset[i] = X_beta[i] + re_contrib[i];
    }

    // Aggregate likelihood info per spatial location
    pg_accumulate_by_location(N, obs_to_loc, omega, kappa, offset,
                              sum_omega_gp, sum_resid_gp);

    // Update each GP effect in NNGP order
    pg_nngp_field_sweep(w, sigma2_gp, phi_gp, cov, graph,
                        sum_omega_gp, sum_resid_gp);

    // Update GP contributions
    pg_scatter_by_location(N, obs_to_loc, w, gp_contrib);

    // 6. Update GP hyperparameters via MH
    pg_update_gp_hyper(sigma2_gp, phi_gp, w, cov, graph,
                       prior_sigma_gp_U, prior_sigma_gp_alpha,
                       prior_phi_lower, prior_phi_upper, 0.1);

    // Save draws
    if (iter >= n_warmup && (iter - n_warmup) % thin == 0) {
      for (int j = 0; j < p; j++) {
        beta_draws(save_idx, j) = beta[j];
      }
      for (int g = 0; g < n_re_groups; g++) {
        re_draws(save_idx, g) = re[g];
      }
      sigma_re_draws[save_idx] = sigma_re;
      for (int s = 0; s < n_spatial; s++) {
        gp_draws(save_idx, s) = w[s];
      }
      sigma2_gp_draws[save_idx] = sigma2_gp;
      phi_gp_draws[save_idx] = phi_gp;

      if (store_eta) {
        for (int i = 0; i < N; i++) {
          eta_draws_gp(save_idx, i) = eta[i];
        }
      }
      save_idx++;
    }

    if ((iter + 1) % 100 == 0) Rcpp::checkUserInterrupt();
  }

  Rcpp::List result = Rcpp::List::create(
    Rcpp::Named("beta") = beta_draws,
    Rcpp::Named("re") = re_draws,
    Rcpp::Named("sigma_re") = sigma_re_draws,
    Rcpp::Named("gp") = gp_draws,
    Rcpp::Named("sigma2_gp") = sigma2_gp_draws,
    Rcpp::Named("phi_gp") = phi_gp_draws
  );

  if (store_eta) {
    result["eta"] = eta_draws_gp;
  }

  PutRNGstate();
  return result;
}

// ---------------------------------------------------------------------
// Multiscale Temporal Gibbs Sampler for Binomial Models
// Supports trend (RW1/RW2) + seasonal (cyclic RW1) + short-term (AR1/IID)
// ---------------------------------------------------------------------

// [[Rcpp::export]]
Rcpp::List cpp_pg_binomial_gibbs_temporal(
    Rcpp::IntegerVector y,
    Rcpp::IntegerVector n,
    Rcpp::NumericMatrix X,
    Rcpp::IntegerVector re_group,
    int n_re_groups,
    Rcpp::IntegerVector time_idx,
    int n_times,
    int seasonal_period,
    int trend_type,
    int short_type,
    int n_iter = 2000,
    int n_warmup = 1000,
    int thin = 1,
    double prior_beta_sd = 10.0,
    double prior_sigma_re_scale = 2.5,
    double prior_sigma_trend_scale = 1.0,
    double prior_sigma_seasonal_scale = 1.0,
    double prior_sigma_short_scale = 1.0,
    double rho_short_init = 0.5,
    bool store_eta = false,
    bool verbose = true,
    int n_threads = 1
) {
  // CRITICAL: Must call GetRNGstate/PutRNGstate when using R's RNG from C++
  GetRNGstate();

  int N = y.size();
  int p = X.ncol();
  int n_save = (n_iter - n_warmup) / thin;

  int n_trend = (trend_type > 0) ? n_times : 0;
  int n_seasonal = (seasonal_period > 0) ? seasonal_period : 0;
  int n_short = (short_type > 0) ? n_times : 0;

  ratiod_omp::ScopedThreadCount thread_scope(n_threads);

  // Storage
  Rcpp::NumericMatrix beta_draws(n_save, p);
  Rcpp::NumericMatrix re_draws(n_save, n_re_groups);
  Rcpp::NumericVector sigma_re_draws(n_save);
  Rcpp::NumericMatrix trend_draws(n_save, n_trend);
  Rcpp::NumericMatrix seasonal_draws(n_save, n_seasonal);
  Rcpp::NumericMatrix short_draws(n_save, n_short);
  Rcpp::NumericVector sigma_trend_draws(n_save);
  Rcpp::NumericVector sigma_seasonal_draws(n_save);
  Rcpp::NumericVector sigma_short_draws(n_save);
  Rcpp::NumericVector rho_short_draws(n_save);
  Rcpp::NumericMatrix eta_draws_temp;
  if (store_eta) eta_draws_temp = Rcpp::NumericMatrix(n_save, N);

  // Initialize
  Rcpp::NumericVector beta(p, 0.0);
  Rcpp::NumericVector re(n_re_groups, 0.0);
  double sigma_re = 1.0;
  Rcpp::NumericVector trend(n_trend, 0.0);
  Rcpp::NumericVector seasonal(n_seasonal, 0.0);
  Rcpp::NumericVector short_term(n_short, 0.0);
  double sigma_trend = 1.0;
  double sigma_seasonal = 1.0;
  double sigma_short = 1.0;
  double rho_short = rho_short_init;

  // Working vectors
  Rcpp::NumericVector omega(N);
  Rcpp::NumericVector kappa(N);
  Rcpp::NumericVector eta(N);
  Rcpp::NumericVector X_beta(N);
  Rcpp::NumericVector re_contrib(N);
  Rcpp::NumericVector temp_contrib(N);
  Rcpp::NumericVector offset(N);

  for (int i = 0; i < N; i++) {
    omega[i] = 0.5;
    kappa[i] = y[i] - 0.5 * n[i];
    X_beta[i] = 0.0;
    re_contrib[i] = 0.0;
    temp_contrib[i] = 0.0;
  }

  int save_idx = 0;

  for (int iter = 0; iter < n_iter; iter++) {
    if (verbose && (iter + 1) % 200 == 0) {
      Rcpp::Rcout << "  Iteration " << (iter + 1) << "/" << n_iter << "\n";
    }

    // 1. Compute eta and temporal contributions
    for (int i = 0; i < N; i++) {
      int t = time_idx[i] - 1;
      double temp_eff = 0.0;
      if (t >= 0 && t < n_times) {
        if (n_trend > 0 && t < n_trend) temp_eff += trend[t];
        if (n_seasonal > 0) temp_eff += seasonal[t % seasonal_period];
        if (n_short > 0 && t < n_short) temp_eff += short_term[t];
      }
      temp_contrib[i] = temp_eff;
      eta[i] = X_beta[i] + re_contrib[i] + temp_contrib[i];
    }

    // 2. Update omega
    for (int i = 0; i < N; i++) {
      omega[i] = ratiod::rpg_int(n[i], eta[i]);
    }

    // 3. Update beta
    for (int i = 0; i < N; i++) {
      offset[i] = re_contrib[i] + temp_contrib[i];
    }
    beta = ratiod::update_beta(kappa, omega, X, offset, prior_beta_sd);

    for (int i = 0; i < N; i++) {
      X_beta[i] = 0.0;
      for (int j = 0; j < p; j++) {
        X_beta[i] += X(i, j) * beta[j];
      }
    }

    // 4. Update random effects
    if (n_re_groups > 0) {
      for (int i = 0; i < N; i++) {
        offset[i] = X_beta[i] + temp_contrib[i];
      }
      re = ratiod::update_re(kappa, omega, offset, re_group, n_re_groups, sigma_re);
      sigma_re = ratiod::update_sigma_re(re, prior_sigma_re_scale);

      for (int i = 0; i < N; i++) {
        re_contrib[i] = re[re_group[i] - 1];
      }
    }

    // 5. Update temporal effects
    for (int i = 0; i < N; i++) {
      offset[i] = X_beta[i] + re_contrib[i];
    }

    // Aggregate for trend
    std::vector<double> sum_omega_t(n_times, 0.0);
    std::vector<double> sum_resid_t(n_times, 0.0);
    for (int i = 0; i < N; i++) {
      int t = time_idx[i] - 1;
      if (t >= 0 && t < n_times) {
        sum_omega_t[t] += omega[i];
        double other_temp = 0.0;
        if (n_seasonal > 0) other_temp += seasonal[t % seasonal_period];
        if (n_short > 0 && t < n_short) other_temp += short_term[t];
        sum_resid_t[t] += kappa[i] - omega[i] * (offset[i] + other_temp);
      }
    }

    // Update trend (RW1)
    if (trend_type == 1) {
      double tau_trend = 1.0 / (sigma_trend * sigma_trend);
      for (int t = 0; t < n_trend; t++) {
        double tau_prior, mean_prior;
        if (t == 0) {
          tau_prior = tau_trend;
          mean_prior = (n_trend > 1) ? trend[1] : 0.0;
        } else if (t == n_trend - 1) {
          tau_prior = tau_trend;
          mean_prior = trend[t - 1];
        } else {
          tau_prior = 2.0 * tau_trend;
          mean_prior = 0.5 * (trend[t - 1] + trend[t + 1]);
        }

        double tau_post = tau_prior + sum_omega_t[t];
        double mean_post = (tau_prior * mean_prior + sum_resid_t[t]) / tau_post;
        trend[t] = R::rnorm(mean_post, 1.0 / std::sqrt(tau_post));
      }

      double ss = 0.0;
      for (int t = 1; t < n_trend; t++) {
        double diff = trend[t] - trend[t - 1];
        ss += diff * diff;
      }
      double shape = prior_sigma_trend_scale + 0.5 * (n_trend - 1);
      double rate = prior_sigma_trend_scale + 0.5 * ss;
      sigma_trend = 1.0 / std::sqrt(R::rgamma(shape, 1.0 / rate));
    }

    // Update seasonal (cyclic RW1)
    if (n_seasonal > 0) {
      std::vector<double> sum_omega_s(seasonal_period, 0.0);
      std::vector<double> sum_resid_s(seasonal_period, 0.0);
      for (int i = 0; i < N; i++) {
        int t = time_idx[i] - 1;
        if (t >= 0) {
          int s = t % seasonal_period;
          sum_omega_s[s] += omega[i];
          double other_temp = 0.0;
          if (n_trend > 0 && t < n_trend) other_temp += trend[t];
          if (n_short > 0 && t < n_short) other_temp += short_term[t];
          sum_resid_s[s] += kappa[i] - omega[i] * (offset[i] + other_temp);
        }
      }

      double tau_seasonal_val = 1.0 / (sigma_seasonal * sigma_seasonal);
      for (int s = 0; s < n_seasonal; s++) {
        int s_prev = (s == 0) ? n_seasonal - 1 : s - 1;
        int s_next = (s == n_seasonal - 1) ? 0 : s + 1;

        double tau_prior = 2.0 * tau_seasonal_val;
        double mean_prior = 0.5 * (seasonal[s_prev] + seasonal[s_next]);

        double tau_post = tau_prior + sum_omega_s[s];
        double mean_post = (tau_prior * mean_prior + sum_resid_s[s]) / tau_post;
        seasonal[s] = R::rnorm(mean_post, 1.0 / std::sqrt(tau_post));
      }

      double mean_s = 0.0;
      for (int s = 0; s < n_seasonal; s++) mean_s += seasonal[s];
      mean_s /= n_seasonal;
      for (int s = 0; s < n_seasonal; s++) seasonal[s] -= mean_s;

      double ss = 0.0;
      for (int s = 0; s < n_seasonal; s++) {
        int s_next = (s == n_seasonal - 1) ? 0 : s + 1;
        double diff = seasonal[s_next] - seasonal[s];
        ss += diff * diff;
      }
      double shape = prior_sigma_seasonal_scale + 0.5 * n_seasonal;
      double rate = prior_sigma_seasonal_scale + 0.5 * ss;
      sigma_seasonal = 1.0 / std::sqrt(R::rgamma(shape, 1.0 / rate));
    }

    // Update short-term (AR1 or IID)
    if (short_type > 0) {
      std::vector<double> sum_omega_sh(n_short, 0.0);
      std::vector<double> sum_resid_sh(n_short, 0.0);
      for (int i = 0; i < N; i++) {
        int t = time_idx[i] - 1;
        if (t >= 0 && t < n_short) {
          sum_omega_sh[t] += omega[i];
          double other_temp = 0.0;
          if (n_trend > 0 && t < n_trend) other_temp += trend[t];
          if (n_seasonal > 0) other_temp += seasonal[t % seasonal_period];
          sum_resid_sh[t] += kappa[i] - omega[i] * (offset[i] + other_temp);
        }
      }

      double tau_short_val = 1.0 / (sigma_short * sigma_short);

      if (short_type == 1) {  // AR1
        for (int t = 0; t < n_short; t++) {
          double tau_prior, mean_prior;
          if (t == 0) {
            tau_prior = tau_short_val * (1.0 - rho_short * rho_short);
            mean_prior = 0.0;
          } else {
            tau_prior = tau_short_val;
            mean_prior = rho_short * short_term[t - 1];
          }

          double tau_post = tau_prior + sum_omega_sh[t];
          double mean_post = (tau_prior * mean_prior + sum_resid_sh[t]) / tau_post;
          short_term[t] = R::rnorm(mean_post, 1.0 / std::sqrt(tau_post));
        }
      } else {  // IID
        for (int t = 0; t < n_short; t++) {
          double tau_post = tau_short_val + sum_omega_sh[t];
          double mean_post = sum_resid_sh[t] / tau_post;
          short_term[t] = R::rnorm(mean_post, 1.0 / std::sqrt(tau_post));
        }
      }

      double ss = 0.0;
      if (short_type == 1) {
        ss = short_term[0] * short_term[0] * (1.0 - rho_short * rho_short);
        for (int t = 1; t < n_short; t++) {
          double resid = short_term[t] - rho_short * short_term[t - 1];
          ss += resid * resid;
        }
      } else {
        for (int t = 0; t < n_short; t++) {
          ss += short_term[t] * short_term[t];
        }
      }
      double shape = prior_sigma_short_scale + 0.5 * n_short;
      double rate = prior_sigma_short_scale + 0.5 * ss;
      sigma_short = 1.0 / std::sqrt(R::rgamma(shape, 1.0 / rate));
    }

    // Save draws
    if (iter >= n_warmup && (iter - n_warmup) % thin == 0) {
      for (int j = 0; j < p; j++) {
        beta_draws(save_idx, j) = beta[j];
      }
      for (int g = 0; g < n_re_groups; g++) {
        re_draws(save_idx, g) = re[g];
      }
      sigma_re_draws[save_idx] = sigma_re;
      for (int t = 0; t < n_trend; t++) {
        trend_draws(save_idx, t) = trend[t];
      }
      for (int s = 0; s < n_seasonal; s++) {
        seasonal_draws(save_idx, s) = seasonal[s];
      }
      for (int t = 0; t < n_short; t++) {
        short_draws(save_idx, t) = short_term[t];
      }
      sigma_trend_draws[save_idx] = sigma_trend;
      sigma_seasonal_draws[save_idx] = sigma_seasonal;
      sigma_short_draws[save_idx] = sigma_short;
      rho_short_draws[save_idx] = rho_short;

      if (store_eta) {
        for (int i = 0; i < N; i++) {
          eta_draws_temp(save_idx, i) = eta[i];
        }
      }
      save_idx++;
    }

    if ((iter + 1) % 100 == 0) Rcpp::checkUserInterrupt();
  }

  Rcpp::List result = Rcpp::List::create(
    Rcpp::Named("beta") = beta_draws,
    Rcpp::Named("re") = re_draws,
    Rcpp::Named("sigma_re") = sigma_re_draws,
    Rcpp::Named("trend") = trend_draws,
    Rcpp::Named("seasonal") = seasonal_draws,
    Rcpp::Named("short_term") = short_draws,
    Rcpp::Named("sigma_trend") = sigma_trend_draws,
    Rcpp::Named("sigma_seasonal") = sigma_seasonal_draws,
    Rcpp::Named("sigma_short") = sigma_short_draws,
    Rcpp::Named("rho_short") = rho_short_draws
  );

  if (store_eta) {
    result["eta"] = eta_draws_temp;
  }

  PutRNGstate();
  return result;
}


// -----------------------------------------------------------------------------
// Multiscale GP Gibbs sampler (local + regional components)
// -----------------------------------------------------------------------------

// [[Rcpp::export]]
Rcpp::List cpp_pg_binomial_gibbs_multiscale_gp(
    Rcpp::IntegerVector y,
    Rcpp::IntegerVector n,
    Rcpp::NumericMatrix X,
    Rcpp::IntegerVector re_group,
    int n_re_groups,
    Rcpp::NumericMatrix coords,
    Rcpp::IntegerMatrix nn_idx_local,
    Rcpp::NumericMatrix nn_dist_local,
    Rcpp::IntegerVector nn_order_local,
    int nn_local,
    Rcpp::IntegerMatrix nn_idx_regional,
    Rcpp::NumericMatrix nn_dist_regional,
    Rcpp::IntegerVector nn_order_regional,
    int nn_regional,
    Rcpp::IntegerVector obs_to_loc,
    int n_spatial,
    double sigma2_local_init,
    double phi_local_init,
    double sigma2_regional_init,
    double phi_regional_init,
    int cov_type,
    int n_iter = 2000,
    int n_warmup = 1000,
    int thin = 1,
    double prior_beta_sd = 10.0,
    double prior_sigma_re_scale = 2.5,
    double prior_sigma_local_U = 1.0,
    double prior_sigma_local_alpha = 0.01,
    double prior_phi_local_lower = 0.01,
    double prior_phi_local_upper = 5.0,
    double prior_sigma_regional_U = 1.0,
    double prior_sigma_regional_alpha = 0.01,
    double prior_phi_regional_lower = 0.1,
    double prior_phi_regional_upper = 20.0,
    bool store_eta = false,
    bool verbose = true,
    int n_threads = 1
) {
  // CRITICAL: Must call GetRNGstate/PutRNGstate when using R's RNG from C++
  GetRNGstate();

  int N = y.size();
  int p = X.ncol();

  // Both scales are NNGP fields over the same locations, read through the same
  // struct and the same conditional as the single-scale entry. They used to
  // carry their own inline exponential kernel, ignoring cov_type outright,
  // with the raw covariance in place of the
  // conditional mean's C^-1 c weights and sigma2 in place of the conditional
  // variance, which is not the density either scale's prior claims.
  const ratiod_cov::CovType cov = ratiod_cov::cov_type_from_int(cov_type);
  const PgNngpGraph graph_local{coords, nn_idx_local, nn_dist_local,
                                nn_order_local, nn_local, n_spatial};
  const PgNngpGraph graph_regional{coords, nn_idx_regional, nn_dist_regional,
                                   nn_order_regional, nn_regional, n_spatial};
  pg_check_nngp_graph(graph_local, "cpp_pg_binomial_gibbs_multiscale_gp (local)");
  pg_check_nngp_graph(graph_regional,
                      "cpp_pg_binomial_gibbs_multiscale_gp (regional)");
  pg_check_obs_to_loc(obs_to_loc, N, n_spatial,
                      "cpp_pg_binomial_gibbs_multiscale_gp");

  if (verbose) {
    Rcpp::Rcout << "PG Binomial Gibbs sampler with multiscale GP spatial\n";
    Rcpp::Rcout << "  N = " << N << ", p = " << p << "\n";
    Rcpp::Rcout << "  n_spatial = " << n_spatial << "\n";
    Rcpp::Rcout << "  nn_local = " << nn_local << ", nn_regional = " << nn_regional << "\n";
  }

  // Initialize parameters
  Rcpp::NumericVector beta(p, 0.0);
  Rcpp::NumericVector re(n_re_groups, 0.0);
  double sigma_re = 1.0;

  // Local GP effects
  std::vector<double> w_local(n_spatial, 0.0);
  double sigma2_local = sigma2_local_init;
  double phi_local = phi_local_init;

  // Regional GP effects
  std::vector<double> w_regional(n_spatial, 0.0);
  double sigma2_regional = sigma2_regional_init;
  double phi_regional = phi_regional_init;

  // Working vectors (use Rcpp types for compatibility with update functions)
  Rcpp::NumericVector omega(N, 0.0);
  Rcpp::NumericVector kappa(N, 0.0);
  Rcpp::NumericVector eta_vec(N, 0.0);
  Rcpp::NumericVector X_beta(N, 0.0);
  Rcpp::NumericVector re_contrib(N, 0.0);
  Rcpp::NumericVector local_contrib(N, 0.0);
  Rcpp::NumericVector regional_contrib(N, 0.0);
  Rcpp::NumericVector offset(N, 0.0);
  std::vector<double> sum_omega_local(n_spatial, 0.0);
  std::vector<double> sum_resid_local(n_spatial, 0.0);
  std::vector<double> sum_omega_regional(n_spatial, 0.0);
  std::vector<double> sum_resid_regional(n_spatial, 0.0);

  // Compute kappa
  for (int i = 0; i < N; i++) {
    kappa[i] = (double)y[i] - 0.5 * (double)n[i];
  }

  // Storage for draws
  int n_save = (n_iter - n_warmup) / thin;
  Rcpp::NumericMatrix beta_draws(n_save, p);
  Rcpp::NumericMatrix re_draws(n_save, n_re_groups);
  Rcpp::NumericVector sigma_re_draws(n_save);
  Rcpp::NumericMatrix w_local_draws(n_save, n_spatial);
  Rcpp::NumericMatrix w_regional_draws(n_save, n_spatial);
  Rcpp::NumericVector sigma2_local_draws(n_save);
  Rcpp::NumericVector phi_local_draws(n_save);
  Rcpp::NumericVector sigma2_regional_draws(n_save);
  Rcpp::NumericVector phi_regional_draws(n_save);
  Rcpp::NumericMatrix eta_draws_temp;
  if (store_eta) {
    eta_draws_temp = Rcpp::NumericMatrix(n_save, N);
  }

  int save_idx = 0;

  for (int iter = 0; iter < n_iter; iter++) {
    if (verbose && (iter + 1) % 200 == 0) {
      Rcpp::Rcout << "  Iteration " << (iter + 1) << "/" << n_iter << "\n";
    }

    // 1. Compute eta
    for (int i = 0; i < N; i++) {
      eta_vec[i] = X_beta[i] + re_contrib[i] + local_contrib[i] + regional_contrib[i];
    }

    // 2. Update omega | eta
    for (int i = 0; i < N; i++) {
      omega[i] = ratiod::rpg_int(n[i], eta_vec[i]);
    }

    // 3. Update beta
    for (int i = 0; i < N; i++) {
      offset[i] = re_contrib[i] + local_contrib[i] + regional_contrib[i];
    }
    beta = ratiod::update_beta(kappa, omega, X, offset, prior_beta_sd);

    for (int i = 0; i < N; i++) {
      X_beta[i] = 0.0;
      for (int j = 0; j < p; j++) {
        X_beta[i] += X(i, j) * beta[j];
      }
    }

    // 4. Update random effects
    if (n_re_groups > 0) {
      for (int i = 0; i < N; i++) {
        offset[i] = X_beta[i] + local_contrib[i] + regional_contrib[i];
      }
      re = ratiod::update_re(kappa, omega, offset, re_group, n_re_groups, sigma_re);
      sigma_re = ratiod::update_sigma_re(re, prior_sigma_re_scale);

      for (int i = 0; i < N; i++) {
        re_contrib[i] = re[re_group[i] - 1];
      }
    }

    // 5. Update local GP effects
    for (int i = 0; i < N; i++) {
      offset[i] = X_beta[i] + re_contrib[i] + regional_contrib[i];
    }

    // Aggregate likelihood info per spatial location
    pg_accumulate_by_location(N, obs_to_loc, omega, kappa, offset,
                              sum_omega_local, sum_resid_local);

    // Sequential NNGP Gibbs update for local effects
    pg_nngp_field_sweep(w_local, sigma2_local, phi_local, cov, graph_local,
                        sum_omega_local, sum_resid_local);

    // Update local contributions
    pg_scatter_by_location(N, obs_to_loc, w_local, local_contrib);

    // 6. Update regional GP effects
    for (int i = 0; i < N; i++) {
      offset[i] = X_beta[i] + re_contrib[i] + local_contrib[i];
    }

    pg_accumulate_by_location(N, obs_to_loc, omega, kappa, offset,
                              sum_omega_regional, sum_resid_regional);

    pg_nngp_field_sweep(w_regional, sigma2_regional, phi_regional, cov,
                        graph_regional, sum_omega_regional, sum_resid_regional);

    pg_scatter_by_location(N, obs_to_loc, w_regional, regional_contrib);

    // 7. Update hyperparameters via MH, each scale against its own NNGP
    // density. sigma2 used to be a conjugate draw from an independent
    // N(0, sigma2) form -- ignoring the spatial correlation and reading the PC
    // prior's upper bound U as an inverse-gamma rate, with alpha unread -- and
    // phi's ratio was built from the same unnormalized quadratic form the field
    // sweep used.
    pg_update_gp_hyper(sigma2_local, phi_local, w_local, cov, graph_local,
                       prior_sigma_local_U, prior_sigma_local_alpha,
                       prior_phi_local_lower, prior_phi_local_upper, 0.1);

    pg_update_gp_hyper(sigma2_regional, phi_regional, w_regional, cov,
                       graph_regional,
                       prior_sigma_regional_U, prior_sigma_regional_alpha,
                       prior_phi_regional_lower, prior_phi_regional_upper, 0.1);

    // Store draws after warmup
    if (iter >= n_warmup && (iter - n_warmup + 1) % thin == 0) {
      for (int j = 0; j < p; j++) {
        beta_draws(save_idx, j) = beta[j];
      }
      for (int g = 0; g < n_re_groups; g++) {
        re_draws(save_idx, g) = re[g];
      }
      sigma_re_draws[save_idx] = sigma_re;

      for (int s = 0; s < n_spatial; s++) {
        w_local_draws(save_idx, s) = w_local[s];
        w_regional_draws(save_idx, s) = w_regional[s];
      }
      sigma2_local_draws[save_idx] = sigma2_local;
      phi_local_draws[save_idx] = phi_local;
      sigma2_regional_draws[save_idx] = sigma2_regional;
      phi_regional_draws[save_idx] = phi_regional;

      if (store_eta) {
        for (int i = 0; i < N; i++) {
          eta_draws_temp(save_idx, i) = eta_vec[i];
        }
      }
      save_idx++;
    }

    if ((iter + 1) % 100 == 0) Rcpp::checkUserInterrupt();
  }

  Rcpp::List result = Rcpp::List::create(
    Rcpp::Named("beta") = beta_draws,
    Rcpp::Named("re") = re_draws,
    Rcpp::Named("sigma_re") = sigma_re_draws,
    Rcpp::Named("w_local") = w_local_draws,
    Rcpp::Named("w_regional") = w_regional_draws,
    Rcpp::Named("sigma2_local") = sigma2_local_draws,
    Rcpp::Named("phi_local") = phi_local_draws,
    Rcpp::Named("sigma2_regional") = sigma2_regional_draws,
    Rcpp::Named("phi_regional") = phi_regional_draws
  );

  if (store_eta) {
    result["eta"] = eta_draws_temp;
  }

  PutRNGstate();
  return result;
}


// ---------------------------------------------------------------------
// RSR (Restricted Spatial Regression) Gibbs sampler
//
// The intrinsic CAR field phi reaches the predictor as A phi, A = (I - Q Q') S
// with S the observation-to-unit incidence and Q the orthonormal basis of the
// restricted covariates. Given the Polya-Gamma weights, phi is Gaussian with
// precision tau Q_icar + A' Omega A and linear term A' (kappa - Omega offset),
// and is drawn from that conditional in one block.
// ---------------------------------------------------------------------

// [[Rcpp::export]]
Rcpp::List cpp_pg_binomial_gibbs_rsr(
    Rcpp::IntegerVector y,
    Rcpp::IntegerVector n,
    Rcpp::NumericMatrix X,
    Rcpp::IntegerVector re_group,
    int n_re_groups,
    Rcpp::IntegerVector spatial_group,
    int n_spatial_units,
    Rcpp::List adj_list,
    Rcpp::IntegerVector n_neighbors,
    Rcpp::NumericMatrix rsr_design,  // A = (I - Q Q') S, N x n_spatial_units

    int n_iter = 2000,
    int n_warmup = 1000,
    int thin = 1,
    double prior_beta_sd = 10.0,
    double prior_sigma_re_scale = 2.5,
    double prior_tau_shape = 1.0,
    double prior_tau_rate = 0.01,
    bool store_eta = false,
    bool verbose = true,
    int n_threads = 1
) {
  // CRITICAL: Must call GetRNGstate/PutRNGstate when using R's RNG from C++
  GetRNGstate();

  int N = y.size();
  int p = X.ncol();
  int n_save = (n_iter - n_warmup) / thin;

  ratiod_omp::ScopedThreadCount thread_scope(n_threads);

  // Storage for draws
  Rcpp::NumericMatrix beta_draws(n_save, p);
  Rcpp::NumericMatrix re_draws(n_save, n_re_groups);
  Rcpp::NumericVector sigma_re_draws(n_save);
  Rcpp::NumericMatrix spatial_raw_draws(n_save, n_spatial_units);
  Rcpp::NumericMatrix spatial_proj_draws(n_save, N);
  Rcpp::NumericVector tau_draws(n_save);
  Rcpp::NumericMatrix eta_draws(n_save, N);

  // Current values
  Rcpp::NumericVector beta(p, 0.0);
  Rcpp::NumericVector re(n_re_groups, 0.0);
  double sigma_re = 1.0;
  Rcpp::NumericVector phi(n_spatial_units, 0.0);
  double tau = 1.0;

  const int S = n_spatial_units;
  if (rsr_design.nrow() != N || rsr_design.ncol() != S) {
    Rcpp::stop("The RSR design is %d x %d; expected %d x %d.",
               rsr_design.nrow(), rsr_design.ncol(), N, S);
  }
  Eigen::Map<const Eigen::MatrixXd> A(rsr_design.begin(), N, S);

  // Intrinsic CAR structure Q = D - W; adj_list holds 1-based neighbours.
  Eigen::MatrixXd Q_icar = Eigen::MatrixXd::Zero(S, S);
  for (int s = 0; s < S; s++) {
    Q_icar(s, s) = n_neighbors[s];
    Rcpp::IntegerVector nb = adj_list[s];
    for (int k = 0; k < nb.size(); k++) Q_icar(s, nb[k] - 1) -= 1.0;
  }

  // When the restricted span holds the constant, A 1 = 0: neither the
  // likelihood nor the intrinsic prior sees the field's level. Adding 1 1' to
  // the conditional precision gives that direction a proper law, independent
  // of the rest, and centring the draw removes it again, which is the draw
  // under the sum-to-zero constraint.
  const double a_scale = std::max(1.0, A.cwiseAbs().maxCoeff());
  const bool level_removed =
      (A * Eigen::VectorXd::Ones(S)).cwiseAbs().maxCoeff() < 1e-8 * a_scale;

  Eigen::VectorXd phi_e = Eigen::VectorXd::Zero(S);
  Eigen::VectorXd field_obs = Eigen::VectorXd::Zero(N);
  Eigen::VectorXd resid(N);
  Eigen::VectorXd z(S);
  Eigen::MatrixXd OA(N, S);

  // Compute kappa once
  Rcpp::NumericVector kappa(N);
  for (int i = 0; i < N; i++) {
    kappa[i] = y[i] - n[i] / 2.0;
  }

  // Working vectors
  Rcpp::NumericVector omega(N, 0.1);
  Rcpp::NumericVector offset(N);
  Rcpp::NumericVector eta(N);
  Rcpp::NumericVector X_beta(N);
  Rcpp::NumericVector re_contrib(N, 0.0);
  Rcpp::NumericVector spatial_contrib(N, 0.0);

  // Initialize X_beta
  for (int i = 0; i < N; i++) {
    X_beta[i] = 0.0;
    for (int j = 0; j < p; j++) {
      X_beta[i] += X(i, j) * beta[j];
    }
  }

  int save_idx = 0;

  // The restricted field at the observations and eta of the current state,
  // refreshed for the omega draw and again at the save so the stored field
  // and eta match the saved parameters.
  auto refresh_eta = [&]() {
    field_obs.noalias() = A * phi_e;
    for (int i = 0; i < N; i++) {
      spatial_contrib[i] = field_obs[i];
      eta[i] = X_beta[i] + re_contrib[i] + spatial_contrib[i];
    }
  };

  for (int iter = 0; iter < n_iter; iter++) {
    // 1-3. The restricted field and eta
    refresh_eta();

    // 4. Sample omega ~ PG(n, eta)
    // Note: NOT parallelized - R's RNG is not thread-safe
    for (int i = 0; i < N; i++) {
      omega[i] = ratiod::rpg_int(n[i], eta[i]);
    }

    // 5. Update beta | omega, re, phi_proj, y
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int i = 0; i < N; i++) {
      offset[i] = re_contrib[i] + spatial_contrib[i];
    }
    beta = ratiod::update_beta(kappa, omega, X, offset, prior_beta_sd);

    // 6. Recompute X_beta
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int i = 0; i < N; i++) {
      X_beta[i] = 0.0;
      for (int j = 0; j < p; j++) {
        X_beta[i] += X(i, j) * beta[j];
      }
    }

    // 7. Update random effects | omega, beta, phi_proj, sigma_re
    if (n_re_groups > 0) {
      #ifdef _OPENMP
      #pragma omp parallel for schedule(static)
      #endif
      for (int i = 0; i < N; i++) {
        offset[i] = X_beta[i] + spatial_contrib[i];
      }
      re = ratiod::update_re(kappa, omega, offset, re_group, n_re_groups, sigma_re);

      sigma_re = ratiod::update_sigma_re(re, prior_sigma_re_scale);

      #ifdef _OPENMP
      #pragma omp parallel for schedule(static)
      #endif
      for (int i = 0; i < N; i++) {
        re_contrib[i] = re[re_group[i] - 1];
      }
    }

    // 8. phi | omega, beta, re, tau from its Gaussian conditional.
    for (int i = 0; i < N; i++) {
      resid[i] = kappa[i] - omega[i] * (X_beta[i] + re_contrib[i]);
    }
    OA = A;
    for (int i = 0; i < N; i++) OA.row(i) *= omega[i];
    Eigen::MatrixXd M = tau * Q_icar;
    M.noalias() += A.transpose() * OA;
    if (level_removed) M.array() += 1.0;
    Eigen::LLT<Eigen::MatrixXd> llt(M);
    if (llt.info() != Eigen::Success) {
      PutRNGstate();
      Rcpp::stop("The conditional precision of the restricted field is not "
                 "positive definite; the restricted covariates leave a "
                 "direction of the field unidentified.");
    }
    Eigen::VectorXd mu = llt.solve(A.transpose() * resid);
    for (int s = 0; s < S; s++) z[s] = R::norm_rand();
    phi_e = mu + llt.matrixU().solve(z);
    if (level_removed) phi_e.array() -= phi_e.mean();
    for (int s = 0; s < S; s++) phi[s] = phi_e[s];

    // 9. Update tau (spatial precision)
    tau = ratiod::update_tau_icar(phi, adj_list, n_neighbors, prior_tau_shape, prior_tau_rate);

    // Save draws
    if (iter >= n_warmup && (iter - n_warmup) % thin == 0) {
      for (int j = 0; j < p; j++) {
        beta_draws(save_idx, j) = beta[j];
      }
      for (int g = 0; g < n_re_groups; g++) {
        re_draws(save_idx, g) = re[g];
      }
      sigma_re_draws[save_idx] = sigma_re;

      // The field, and the restricted field at the observations.
      refresh_eta();
      for (int s = 0; s < n_spatial_units; s++) {
        spatial_raw_draws(save_idx, s) = phi[s];
      }
      for (int i = 0; i < N; i++) {
        spatial_proj_draws(save_idx, i) = spatial_contrib[i];
      }
      tau_draws[save_idx] = tau;

      if (store_eta) {
        for (int i = 0; i < N; i++) {
          eta_draws(save_idx, i) = eta[i];
        }
      }
      save_idx++;
    }

    // Progress
    if (verbose && (iter + 1) % 500 == 0) {
      Rcpp::Rcout << "Iteration " << (iter + 1) << "/" << n_iter << std::endl;
    }

    // Check for user interrupt
    if ((iter + 1) % 100 == 0) {
      Rcpp::checkUserInterrupt();
    }
  }

  Rcpp::List result = Rcpp::List::create(
    Rcpp::Named("beta") = beta_draws,
    Rcpp::Named("re") = re_draws,
    Rcpp::Named("sigma_re") = sigma_re_draws,
    Rcpp::Named("spatial_raw") = spatial_raw_draws,
    Rcpp::Named("spatial") = spatial_proj_draws,
    Rcpp::Named("tau") = tau_draws
  );

  if (store_eta) {
    result["eta"] = eta_draws;
  }

  PutRNGstate();
  return result;
}
