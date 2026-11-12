// test_helpers.cpp
// Rcpp wrappers to expose internal C++ functions for unit testing

#include <Rcpp.h>
#include <cmath>
#include <vector>
#include <random>
#include "pg_common.h"
#include "pg_temporal.h"
#include "omp_sum.h"
#include "taylor_jet.h"

#include "linalg_fast.h"
#include "ratio_margins.h"
#ifdef _OPENMP
#include <omp.h>
#endif

using namespace Rcpp;

// [[Rcpp::export]]
NumericVector cpp_test_negbin_jet(int y, double mu, double phi) {
  using J = ratiod::jet::Jet3<double>;
  const auto v = ratiod::math::log_lik_negbin(y,
      ratiod::jet::safe_exp(J::line(std::log(mu), 1.0)), J(phi));
  return NumericVector::create(v.v, v.d1, v.d2, v.d3);
}

template <typename T>
T negbin_probe_density(int y, const T& eta, const T& log_phi, const T& z,
                       const std::string& type) {
  using std::exp;
  const T mu = exp(eta), phi = exp(log_phi);
  if (type == "zi_negbin") return ratiod_zi::zi_negbin_lpmf_logit(y, mu, phi, z);
  if (type == "hurdle_negbin") return ratiod_zi::hurdle_negbin_lpmf_logit(y, mu, phi, z);
  return ratiod::math::log_lik_negbin(y, mu, phi);
}

// [[Rcpp::export]]
List cpp_test_negbin_paths(int y, double mu, double phi, double z,
                           std::string type = "zi_negbin") {
  const double eta = std::log(mu), lp = std::log(phi);
  NumericMatrix out(2, 4);
  rownames(out) = CharacterVector::create("scalar_forward", "scalar_arena");
  colnames(out) = CharacterVector::create("value", "eta", "log_phi", "zi");
  for (int j = 0; j < 3; ++j) {
    const auto val = negbin_probe_density(y, fwd::Dual(eta, j == 0),
        fwd::Dual(lp, j == 1), fwd::Dual(z, j == 2), type);
    out(0, 0) = val.val;
    out(0, j + 1) = val.grad;
  }
  {
    tulpa::arena::ArenaScope scope;
    tulpa::arena::Var a(eta), b(lp), c(z);
    auto val = negbin_probe_density(y, a, b, c, type);
    val.backward();
    out(1, 0) = val.val(); out(1, 1) = a.adj(); out(1, 2) = b.adj(); out(1, 3) = c.adj();
  }

  using namespace tulpaRatio::margins;
  const auto inflation=static_cast<tulpa::ZIType>(ratiod_zi::parse_zi_type(type=="plain" ? "none":type));
  const auto result=linked_result<NegBin>(y,0,eta,phi,inflation,z,0,ratiod_link::LinkType::LOG);
  Observation observation{double(y),eta};
  observation.log_factorial=std::lgamma(y+1.);
  observation.lgamma_difference=tulpa::math::lgamma_ratio_count(y,phi);
  observation.digamma_difference=tulpa::math::polygamma_rising_difference(0,y,phi);
  const auto cached=NegBin(phi).evaluate<true>(observation);
  return List::create(_["autodiff"]=out,
      _["scalar_double"]=negbin_probe_density(y,eta,lp,z,type),
      _["H"]=NumericVector::create(result.ll,result.residual,phi*result.extra,result.zi),
      _["H_table"]=NumericVector::create(cached.ll,cached.residual,phi*cached.extra));
}

static std::vector<double> flatten_row_major(const NumericMatrix& x) {
  std::vector<double> out(x.size());
  for (int i = 0; i < x.nrow(); ++i)
    for (int j = 0; j < x.ncol(); ++j) out[i * x.ncol() + j] = x(i, j);
  return out;
}

// [[Rcpp::export]]
NumericMatrix cpp_test_cholesky(NumericMatrix x) {
  return ratiod::chol_decomp(x);
}

// [[Rcpp::export]]
NumericVector cpp_test_inverse_logit(NumericVector x) {
  NumericVector out(x.size());
  for (int i = 0; i < x.size(); ++i) out[i] = ratiod::math::inv_logit(x[i]);
  return out;
}

// [[Rcpp::export]]
List cpp_test_autodiff_unary(std::string op, double value, double parameter = 1.0) {
  using namespace tulpa::arena;
  ArenaScope scope;
  Var x(value), out(0.0);
  if (op == "exp_chain") out = exp(x * x);
  else if (op == "lgamma") out = lgamma(x);
  else if (op == "softplus") out = softplus(x);
  else if (op == "inv_logit") out = inv_logit(x);
  else if (op == "log") out = log(x);
  else if (op == "sqrt") out = sqrt(x);
  else if (op == "log1p") out = log1p(x);
  else if (op == "logit") out = logit(x);
  else if (op == "pow") out = pow(x, parameter);
  else stop("Unknown unary operation '%s'.", op.c_str());
  out.backward();
  return List::create(_["value"] = out.val(), _["gradient"] = x.adj());
}

// [[Rcpp::export]]
List cpp_test_autodiff_binary(std::string op, double first, double second) {
  using namespace tulpa::arena;
  ArenaScope scope;
  Var a(first), b(second), out(0.0);
  if (op == "division") out = a / b;
  else if (op == "log_sum_exp") out = log_sum_exp(a, b);
  else stop("Unknown binary operation '%s'.", op.c_str());
  out.backward();
  return List::create(_["value"] = out.val(),
                      _["gradient"] = NumericVector::create(a.adj(), b.adj()));
}

// [[Rcpp::export]]
List cpp_test_autodiff_vector(std::string op, NumericVector values,
                              IntegerVector y, double phi = 1.0) {
  using namespace tulpa::arena;
  ArenaScope scope;
  std::vector<Var> x;
  for (double value : values) x.emplace_back(value);
  Var scale(phi), out(0.0);
  for (int i = 0; i < values.size(); ++i) {
    if (op == "squares") out = out + x[i] * x[i];
    else if (op == "poisson") out = out + ratiod::math::log_lik_poisson(y[i], exp(x[i]));
    else if (op == "negbin") out = out + ratiod::math::log_lik_negbin(y[i], x[i], scale);
    else stop("Unknown vector operation '%s'.", op.c_str());
  }
  out.backward();
  NumericVector gradient(values.size());
  for (int i = 0; i < values.size(); ++i) gradient[i] = x[i].adj();
  return List::create(_["value"] = out.val(), _["gradient"] = gradient,
                      _["gradient_phi"] = scale.adj());
}

// ---------------------------------------------------------------------------
// PG Binomial helper functions (from pg_binomial.cpp)
// ---------------------------------------------------------------------------

// [[Rcpp::export]]
NumericVector cpp_test_pg_update_beta(
    NumericVector kappa,
    NumericVector omega,
    NumericMatrix X,
    NumericVector re_contrib,
    double prior_sd
) {
  return ratiod::update_beta(kappa, omega, X, re_contrib, prior_sd);
}

// [[Rcpp::export]]
NumericVector cpp_test_pg_update_re(
    NumericVector kappa,
    NumericVector omega,
    NumericVector X_beta,
    IntegerVector group,
    int n_groups,
    double sigma_re
) {
  return ratiod::update_re(kappa, omega, X_beta, group, n_groups, sigma_re);
}

// [[Rcpp::export]]
NumericVector cpp_test_pg_sigma_halfcauchy_chain(NumericVector re, double scale,
                                                 int n_iter) {
  ratiod::HalfCauchyScale state;
  NumericVector out(n_iter);
  for (int t = 0; t < n_iter; t++) out[t] = state.update(re, scale);
  return out;
}

// ---------------------------------------------------------------------------
// linalg_fast.h test wrappers
// ---------------------------------------------------------------------------


// [[Rcpp::export]]
double cpp_test_dot_product(NumericVector x, NumericVector y) {
  return ratiod_linalg::dot_product(x.begin(), y.begin(), x.size());
}

// [[Rcpp::export]]
double cpp_test_norm_squared(NumericVector x) {
  return ratiod_linalg::norm_squared(x.begin(), x.size());
}

// [[Rcpp::export]]
double cpp_test_vector_sum(NumericVector x) {
  return ratiod_linalg::vector_sum(x.begin(), x.size());
}

// [[Rcpp::export]]
NumericVector cpp_test_axpy(double a, NumericVector x, NumericVector y) {
  NumericVector result = clone(y);
  ratiod_linalg::axpy(a, x.begin(), result.begin(), x.size());
  return result;
}

// [[Rcpp::export]]
NumericVector cpp_test_scale(double a, NumericVector x) {
  NumericVector result = clone(x);
  ratiod_linalg::scale(a, result.begin(), x.size());
  return result;
}

// [[Rcpp::export]]
NumericVector cpp_test_linalg_matvec(NumericMatrix X, NumericVector beta) {
  int N = X.nrow();
  int p = X.ncol();
  const auto X_flat = flatten_row_major(X);
  NumericVector y(N);
  ratiod_linalg::matvec(X_flat.data(), beta.begin(), y.begin(), N, p);
  return y;
}

// [[Rcpp::export]]
NumericVector cpp_test_linalg_matvec_add(NumericMatrix X, NumericVector beta, NumericVector y_init) {
  int N = X.nrow();
  int p = X.ncol();
  const auto X_flat = flatten_row_major(X);
  NumericVector y = clone(y_init);
  ratiod_linalg::matvec_add(X_flat.data(), beta.begin(), y.begin(), N, p);
  return y;
}

// [[Rcpp::export]]
NumericVector cpp_test_linalg_matvec_transpose(NumericMatrix X, NumericVector x) {
  int N = X.nrow();
  int p = X.ncol();
  const auto X_flat = flatten_row_major(X);
  NumericVector y(p);
  ratiod_linalg::matvec_transpose(X_flat.data(), x.begin(), y.begin(), N, p);
  return y;
}

// [[Rcpp::export]]
double cpp_test_sparse_laplacian_quadform(IntegerVector row_ptr, IntegerVector col_idx, NumericVector x) {
  return ratiod_linalg::sparse_laplacian_quadform(row_ptr.begin(), col_idx.begin(), x.begin(), x.size());
}

// [[Rcpp::export]]
double cpp_test_linalg_log_sum_exp(double a, double b) {
  return ratiod_linalg::log_sum_exp(a, b);
}

// [[Rcpp::export]]
double cpp_test_linalg_log_sum_exp_vec(NumericVector x) {
  return ratiod_linalg::log_sum_exp_vec(x.begin(), x.size());
}

// [[Rcpp::export]]
NumericVector cpp_test_softmax_inplace(NumericVector x) {
  NumericVector result = clone(x);
  ratiod_linalg::softmax_inplace(result.begin(), result.size());
  return result;
}

// [[Rcpp::export]]
List cpp_test_compute_linear_predictors(
    NumericMatrix X_num, NumericVector beta_num,
    NumericMatrix X_denom, NumericVector beta_denom,
    int n_threads
) {
  int N = X_num.nrow();
  int p_num = X_num.ncol();
  int p_denom = X_denom.ncol();

  const auto X_num_flat = flatten_row_major(X_num);
  const auto X_denom_flat = flatten_row_major(X_denom);

  NumericVector eta_num(N);
  NumericVector eta_denom(N);

  ratiod_linalg::compute_linear_predictors(
    X_num_flat.data(), beta_num.begin(), p_num,
    X_denom_flat.data(), beta_denom.begin(), p_denom,
    eta_num.begin(), eta_denom.begin(), N, n_threads
  );

  return List::create(
    Named("eta_num") = eta_num,
    Named("eta_denom") = eta_denom
  );
}

// ---------------------------------------------------------------------------
// hmc_zi.h test wrappers
// ---------------------------------------------------------------------------


// [[Rcpp::export]]
double cpp_test_log1pexp(double x) {
  return ratiod::math::log1p_exp(x);
}

// [[Rcpp::export]]
double cpp_test_log_logistic(double x) {
  return ratiod::math::log_inv_logit(x);
}

// [[Rcpp::export]]
double cpp_test_log1m_logistic(double x) {
  return ratiod::math::log1m_inv_logit(x);
}

// [[Rcpp::export]]
double cpp_test_zi_poisson_lpmf(int y, double mu) {
  return ratiod::math::log_lik_poisson(y, mu);
}

// [[Rcpp::export]]
double cpp_test_zi_negbin_lpmf(int y, double mu, double phi) {
  return ratiod::math::log_lik_negbin(y, mu, phi);
}

// [[Rcpp::export]]
double cpp_test_binomial_lpmf(int y, int n, double q, double logit_zi,
                              double logit_oi, std::string zi_type_str) {
  using namespace ratiod::math;
  switch (ratiod_zi::parse_zi_type(zi_type_str)) {
    case ratiod_zi::ZIType::ZI_BINOMIAL: return log_lik_zi_binomial(y, n, q, logit_zi);
    case ratiod_zi::ZIType::OI_BINOMIAL: return log_lik_oi_binomial(y, n, q, logit_oi);
    case ratiod_zi::ZIType::ZOIB: return log_lik_zoib(y, n, q, logit_zi, logit_oi);
    case ratiod_zi::ZIType::HURDLE_BINOMIAL:
      return log_lik_hurdle_binomial(y, n, q, logit_zi);
    default: return log_lik_binomial(y, n, q);
  }
}

// [[Rcpp::export]]
double cpp_test_beta_binomial_lpmf(int y, int n, double q, double phi) {
  return ratiod::math::log_lik_beta_binomial(y, n, q, phi);
}

// [[Rcpp::export]]
double cpp_test_zi_poisson_lpmf_logit(int y, double mu, double logit_zi) {
  return ratiod_zi::zi_poisson_lpmf_logit(y, mu, logit_zi);
}

// [[Rcpp::export]]
double cpp_test_zi_negbin_lpmf_logit(int y, double mu, double phi, double logit_zi) {
  return ratiod_zi::zi_negbin_lpmf_logit(y, mu, phi, logit_zi);
}

// [[Rcpp::export]]
double cpp_test_zi_negbin_grad_logit_zi(int y, double mu, double phi, double logit_zi) {
  return ratiod_zi::zi_negbin_lpmf_logit(y, fwd::Dual(mu), fwd::Dual(phi),
                                       fwd::Dual(logit_zi, 1.0)).grad;
}

// [[Rcpp::export]]
double cpp_test_truncated_poisson_lpmf(int y, double mu) {
  return ratiod_zi::truncated_poisson_lpmf(y, mu);
}

// [[Rcpp::export]]
double cpp_test_truncated_negbin_lpmf(int y, double mu, double phi) {
  return ratiod_zi::truncated_negbin_lpmf(y, mu, phi);
}

// [[Rcpp::export]]
double cpp_test_hurdle_poisson_lpmf_logit(int y, double mu, double logit_theta) {
  return ratiod_zi::hurdle_poisson_lpmf_logit(y, mu, logit_theta);
}

// [[Rcpp::export]]
double cpp_test_hurdle_negbin_lpmf_logit(int y, double mu, double phi, double logit_theta) {
  return ratiod_zi::hurdle_negbin_lpmf_logit(y, mu, phi, logit_theta);
}

// [[Rcpp::export]]
double cpp_test_zi_log_likelihood(int y, double mu, double phi, double logit_zi, std::string zi_type_str) {
  ratiod_zi::ZIType zi_type = ratiod_zi::parse_zi_type(zi_type_str);
  return ratiod_zi::zi_log_likelihood(y, mu, phi, logit_zi, zi_type);
}

// ---------------------------------------------------------------------------
// OpenMP parallel execution tests
// ---------------------------------------------------------------------------

// [[Rcpp::export]]
List cpp_test_parallel_dot_products(NumericMatrix X, NumericVector y, int n_threads) {
  // Test OpenMP parallel reduction with multiple dot products
  int N = X.nrow();
  int p = X.ncol();

  const auto X_flat = flatten_row_major(X);

  NumericVector results(N);
  const double total_sum = ratiod_omp::sum_range(N, n_threads, [&](int i) {
    const double dot = ratiod_linalg::dot_product(&X_flat[i * p], y.begin(), p);
    results[i] = dot;
    return dot;
  });

  return List::create(
    Named("results") = results,
    Named("total_sum") = total_sum
  );
}

// [[Rcpp::export]]
List cpp_test_parallel_likelihood(
    IntegerVector y,
    NumericVector mu,
    int n_threads
) {
  // Test OpenMP parallel reduction for likelihood computation
  int N = y.size();
  const double log_lik = ratiod_omp::sum_range(N, n_threads, [&](int i) {
    return ratiod::math::log_lik_poisson(y[i], mu[i]);
  });

  return List::create(
    Named("log_lik") = log_lik,
    Named("n_threads_requested") = n_threads
  );
}

// [[Rcpp::export]]
NumericVector cpp_test_pg_rw2_conditional(NumericVector omega, NumericVector residual,
                                         double tau) {
  return ratiod::pg_rw2_draw(Rcpp::as<std::vector<double>>(omega),
    Rcpp::as<std::vector<double>>(residual), tau);
}
