#include "pg_common.h"
#include "pg_engine.h"
#include <tulpa/count_math.h>
#include "omp_thread_scope.h"

using namespace Rcpp;
namespace ratiod {
static double negbin_loglik_eta(
    const IntegerVector& y,
    const NumericVector& eta,
    double r
) {
  int n = y.size();
  double ll = 0.0;

  for (int i = 0; i < n; i++) {
    // mu = r * exp(eta) in Zhou's logit parameterization
    double eta_i = std::max(-20.0, std::min(20.0, eta[i]));
    double mu_i = r * std::exp(eta_i);
    mu_i = std::max(1e-10, mu_i);

    ll += tulpa::math::negbin_log_pmf(y[i], mu_i, r);
  }

  return ll;
}

// Metropolis update for the dispersion conditional.
double update_r_negbin(
    const IntegerVector& y,
    const NumericVector& eta,
    double r_current,
    double prior_shape,
    double prior_rate
) {
  double log_r_current = std::log(r_current);
  double proposal_sd = 0.15;

  double log_r_prop = R::rnorm(log_r_current, proposal_sd);
  double r_prop = std::exp(log_r_prop);

  if (r_prop < 0.1 || r_prop > 500.0) {
    return r_current;
  }

  double log_prior_current = (prior_shape - 1.0) * log_r_current - prior_rate * r_current;
  double log_prior_prop = (prior_shape - 1.0) * log_r_prop - prior_rate * r_prop;

  double log_lik_current = negbin_loglik_eta(y, eta, r_current);
  double log_lik_prop = negbin_loglik_eta(y, eta, r_prop);

  double log_alpha = (log_prior_prop + log_lik_prop + log_r_prop)
                   - (log_prior_current + log_lik_current + log_r_current);

  if (std::log(R::runif(0.0, 1.0)) < log_alpha) {
    return r_prop;
  }
  return r_current;
}

// One process conditional in a shared-random-effect negative-binomial pair.
static void update_negbin_arm(const IntegerVector& y, const NumericMatrix& X,
    NumericVector& beta, const NumericVector& re, const IntegerVector& group,
    bool shared, NumericVector& X_beta, NumericVector& eta, NumericVector& kappa,
    NumericVector& omega, double& r, double beta_sd, double r_shape, double r_rate) {
  const int N = y.size(), p = X.ncol();
  NumericVector re_contrib(N), probabilities(N);
  for (int i = 0; i < N; ++i) {
    X_beta[i] = 0.0;
    for (int j = 0; j < p; ++j) X_beta[i] += X(i, j) * beta[j];
    re_contrib[i] = shared && re.size() > 0 ? re[group[i] - 1] : 0.0;
    eta[i] = X_beta[i] + re_contrib[i];
    kappa[i] = (y[i] - r) / 2.0;
  }
  for (int i = 0; i < N; ++i) {
    const int b = static_cast<int>(std::round(y[i] + r));
    omega[i] = rpg_int(std::max(1, b), eta[i]);
  }
  beta = update_beta(kappa, omega, X, re_contrib, beta_sd);
  for (int i = 0; i < N; ++i) {
    double eta_new = 0.0;
    for (int j = 0; j < p; ++j) eta_new += X(i, j) * beta[j];
    eta_new += re_contrib[i];
    probabilities[i] = 1.0 / (1.0 + std::exp(-eta_new));
  }
  r = update_r_negbin(y, probabilities, r, r_shape, r_rate);
}

List pg_negbin_negbin_gibbs(
    IntegerVector y_num,
    IntegerVector y_denom,
    NumericMatrix X_num,
    NumericMatrix X_denom,
    IntegerVector group,
    int n_groups,
    int n_iter,
    int n_warmup,
    int thin,
    double prior_beta_sd,
    double prior_sigma_scale,
    double prior_r_shape,
    double prior_r_rate,
    double r_num_init,
    double r_denom_init,
    bool shared,
    bool store_eta,
    bool verbose,
    int n_threads
) {
  const int N = y_num.size(), p_num = X_num.ncol(), p_denom = X_denom.ncol();
  const int n_save = ratiod::pg_n_save(n_iter, n_warmup, thin);
  ratiod_omp::ScopedThreadCount thread_scope(n_threads);
  NumericMatrix beta_num_draws(n_save, p_num), beta_denom_draws(n_save, p_denom);
  NumericMatrix re_draws(n_save, n_groups), eta_num_draws, eta_denom_draws;
  if (store_eta) {
    eta_num_draws = NumericMatrix(n_save, N);
    eta_denom_draws = NumericMatrix(n_save, N);
  }
  NumericVector sigma_draws(n_save), r_num_draws(n_save), r_denom_draws(n_save);
  NumericVector beta_num(p_num), beta_denom(p_denom), re(n_groups);
  NumericVector omega_num(N, 1.0), omega_denom(N, 1.0), kappa_num(N), kappa_denom(N);
  NumericVector eta_num(N), eta_denom(N), X_beta_num(N), X_beta_denom(N);
  double sigma_re = 1.0, r_num = r_num_init, r_denom = r_denom_init;
  HalfCauchyScale sigma_re_prior;
  NumericVector no_contribution(N), driver_kappa(N), driver_omega(N);
  NumericMatrix no_eta;
  PGField numerator(no_contribution,
    [&](const NumericVector&, const NumericVector&, const NumericVector&) {
      update_negbin_arm(y_num, X_num, beta_num, re, group, shared, X_beta_num,
        eta_num, kappa_num, omega_num, r_num, prior_beta_sd, prior_r_shape, prior_r_rate);
    }, [&](int row) {
      for (int j = 0; j < p_num; ++j) beta_num_draws(row, j) = beta_num[j];
      r_num_draws[row] = r_num;
      if (store_eta) for (int i = 0; i < N; ++i) eta_num_draws(row, i) = eta_num[i];
    });
  PGField denominator(no_contribution,
    [&](const NumericVector&, const NumericVector&, const NumericVector&) {
      update_negbin_arm(y_denom, X_denom, beta_denom, re, group, shared, X_beta_denom,
        eta_denom, kappa_denom, omega_denom, r_denom, prior_beta_sd, prior_r_shape, prior_r_rate);
    }, [&](int row) {
      for (int j = 0; j < p_denom; ++j) beta_denom_draws(row, j) = beta_denom[j];
      r_denom_draws[row] = r_denom;
      if (store_eta) for (int i = 0; i < N; ++i) eta_denom_draws(row, i) = eta_denom[i];
    });
  PGField random(no_contribution,
    [&](const NumericVector&, const NumericVector&, const NumericVector&) {
      if (!shared || n_groups == 0) return;
      NumericVector kappa(2 * N), omega(2 * N), offset(2 * N);
      IntegerVector groups(2 * N);
      for (int i = 0; i < N; ++i) {
        X_beta_num[i] = X_beta_denom[i] = 0.0;
        for (int j = 0; j < p_num; ++j) X_beta_num[i] += X_num(i, j) * beta_num[j];
        for (int j = 0; j < p_denom; ++j) X_beta_denom[i] += X_denom(i, j) * beta_denom[j];
        kappa[i] = kappa_num[i]; kappa[N + i] = kappa_denom[i];
        omega[i] = omega_num[i]; omega[N + i] = omega_denom[i];
        offset[i] = X_beta_num[i]; offset[N + i] = X_beta_denom[i];
        groups[i] = groups[N + i] = group[i];
      }
      re = update_re(kappa, omega, offset, groups, n_groups, sigma_re);
      sigma_re = sigma_re_prior.update(re, prior_sigma_scale);
    }, [&](int row) {
      for (int g = 0; g < n_groups; ++g) re_draws(row, g) = re[g];
      sigma_draws[row] = sigma_re;
    });
  pg_run(n_iter, n_warmup, thin, verbose, {&numerator, &denominator, &random},
    driver_kappa, driver_omega, no_eta, false,
    [](const NumericVector&, NumericVector&, NumericVector&) {});
  List result = List::create(_["beta_num"] = beta_num_draws,
    _["beta_denom"] = beta_denom_draws, _["re"] = re_draws,
    _["sigma_re"] = sigma_draws, _["r_num"] = r_num_draws, _["r_denom"] = r_denom_draws);
  if (store_eta) {
    result["eta_num"] = eta_num_draws;
    result["eta_denom"] = eta_denom_draws;
  }
  return result;
}

}

// [[Rcpp::export]]
Rcpp::List cpp_pg_negbin_gibbs(
    Rcpp::IntegerVector y,
    Rcpp::NumericMatrix X,
    Rcpp::IntegerVector group,
    int n_groups,
    int n_iter,
    int n_warmup,
    int thin,
    double prior_beta_sd,
    double prior_sigma_scale,
    double prior_r_shape,
    double prior_r_rate,
    double r_init,
    bool store_eta,
    bool verbose,
    int n_threads
) {
  ratiod::PGEngineResult result;
  tulpa::get_pg_negbin_fn()(INTEGER(y), REAL(X), y.size(), X.ncol(),
    INTEGER(group), n_groups, n_iter, n_warmup, thin, prior_beta_sd,
    prior_sigma_scale, prior_r_shape, prior_r_rate, r_init,
    store_eta, verbose, n_threads, &result.value);
  return ratiod::pg_engine_fit(result.value, store_eta);
}

// [[Rcpp::export]]
Rcpp::List cpp_pg_negbin_negbin_gibbs(
    Rcpp::IntegerVector y_num,
    Rcpp::IntegerVector y_denom,
    Rcpp::NumericMatrix X_num,
    Rcpp::NumericMatrix X_denom,
    Rcpp::IntegerVector group,
    int n_groups,
    int n_iter,
    int n_warmup,
    int thin,
    double prior_beta_sd,
    double prior_sigma_scale,
    double prior_r_shape,
    double prior_r_rate,
    double r_num_init,
    double r_denom_init,
    bool shared,
    bool store_eta,
    bool verbose,
    int n_threads
) {
  return ratiod::pg_negbin_negbin_gibbs(
    y_num, y_denom, X_num, X_denom, group, n_groups,
    n_iter, n_warmup, thin, prior_beta_sd, prior_sigma_scale,
    prior_r_shape, prior_r_rate, r_num_init, r_denom_init, shared,
    store_eta, verbose, n_threads
  );
}

// [[Rcpp::export]]
Rcpp::List cpp_pg_negbin_gibbs_spatial(
    Rcpp::IntegerVector y,
    Rcpp::NumericMatrix X,
    Rcpp::IntegerVector re_group,
    int n_re_groups,
    Rcpp::IntegerVector spatial_group,
    int n_spatial_units,
    Rcpp::List adj_list,
    Rcpp::IntegerVector n_neighbors,
    int n_iter,
    int n_warmup,
    int thin,
    double prior_beta_sd,
    double prior_sigma_re_scale,
    double prior_sigma_spatial_U,
    double prior_sigma_spatial_alpha,
    double prior_r_shape,
    double prior_r_rate,
    double r_init,
    bool store_eta,
    bool verbose,
    int n_threads
) {
  std::vector<int> row(n_spatial_units + 1, 0), col;
  for (int i = 0; i < n_spatial_units; ++i) {
    Rcpp::IntegerVector neighbors = adj_list[i];
    col.insert(col.end(), neighbors.begin(), neighbors.end());
    row[i + 1] = col.size();
  }
  ratiod::PGEngineResult result;
  tulpa::get_pg_negbin_spatial_fn()(INTEGER(y), REAL(X), y.size(), X.ncol(),
    INTEGER(re_group), n_re_groups, INTEGER(spatial_group), n_spatial_units,
    row.data(), col.data(), INTEGER(n_neighbors), n_iter, n_warmup, thin,
    prior_beta_sd, prior_sigma_re_scale, prior_sigma_spatial_U, prior_sigma_spatial_alpha,
    prior_r_shape, prior_r_rate, r_init, store_eta, verbose, n_threads, &result.value);
  return ratiod::pg_engine_fit(result.value, store_eta);
}
