// test_helpers.cpp
// Rcpp wrappers to expose internal C++ functions for unit testing

#include <Rcpp.h>
#include <cmath>
#include <vector>
#include <random>
#include "pg_common.h"
#include "pg_temporal.h"
#include "omp_sum.h"
#include "hmc_sampler.h"
#include "ratio_obs_lik.h"
#include "taylor_jet.h"

#include "linalg_fast.h"
#include "hmc_temporal.h"
#include "hmc_temporal_multiscale.h"
#include "hmc_zi.h"
#include "hmc_spatiotemporal.h"
#ifdef _OPENMP
#include <omp.h>
#endif

using namespace Rcpp;

namespace ratiod_hmc {
Rcpp::NumericVector negbin_table_probe(int y, double mu, double phi);
Rcpp::NumericMatrix negbin_h_kernels(int y, double mu, double phi);
}

// [[Rcpp::export]]
NumericMatrix cpp_test_negbin_h_kernels(int y, double mu, double phi) {
  return ratiod_hmc::negbin_h_kernels(y, mu, phi);
}

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
  rownames(out) = CharacterVector::create("legacy_forward", "legacy_arena");
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

  ratiod::ModelData data;
  data.model_type = ratiod::ModelType::NEGBIN_NEGBIN;
  data.y_num = {y}; data.y_denom = {0};
  data.zi_type = ratiod_zi::parse_zi_type(type == "plain" ? "none" : type);
  ratiod::ParamLayout layout;
  layout.has_zi = type != "plain";
  const auto h = ratiod::ratio_obs_derivs(data, layout, 0, eta, 0.0, phi, 1.0, z, 0.0,
      ratiod::OBS_D_PHI_NUM | ratiod::OBS_D_ZI);
  return List::create(_["autodiff"] = out,
      _["legacy_double"] = negbin_probe_density(y, eta, lp, z, type),
      _["H"] = NumericVector::create(h.ll + std::log(2.0), h.eta_num, phi * h.phi_num, h.logit_zi),
      _["H_table"] = ratiod_hmc::negbin_table_probe(y, mu, phi));
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

// A binomial model without observations has independent N(0,1) coefficients.
// [[Rcpp::export]]
List cpp_test_integrator(NumericVector q, NumericVector p, double epsilon) {
  ratiod::ModelData data;
  data.N = 0;
  data.p_num = q.size();
  data.p_denom = 0;
  data.model_type = ratiod::ModelType::BINOMIAL;
  data.sigma_beta = 1.0;
  const auto layout = ratiod_hmc::compute_param_layout(data);
  std::vector<double> position(q.begin(), q.end()), momentum(p.begin(), p.end());
  const auto step = ratiod_hmc::leapfrog_step(position, momentum, epsilon, data, layout);
  ratiod_hmc::DenseMassMatrix mass;
  mass.init(p.size(), ratiod_hmc::MassMatrixType::DIAG);
  const double h = ratiod_hmc::nuts_compute_hamiltonian_fast(
      -0.5 * ratiod_linalg::norm_squared(position.data(), position.size()),
      momentum.data(), mass, p.size());
  return List::create(_["q"] = step.q, _["p"] = step.p, _["hamiltonian"] = h);
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
// hmc_temporal.h test wrappers
// ---------------------------------------------------------------------------


// [[Rcpp::export]]
double cpp_test_rw1_quadratic_form(NumericVector phi, bool cyclic) {
  return ratiod_temporal::rw1_quadratic_form(phi.begin(), phi.size(), cyclic);
}

// [[Rcpp::export]]
double cpp_test_rw2_quadratic_form(NumericVector phi, bool cyclic) {
  return ratiod_temporal::rw2_quadratic_form(phi.begin(), phi.size(), cyclic);
}

// [[Rcpp::export]]
double cpp_test_ar1_log_density(NumericVector phi, double rho, double tau) {
  return ratiod_ar1::ar1_log_density(phi.begin(), phi.size(), rho, tau);
}

// [[Rcpp::export]]
double cpp_test_temporal_log_prior(
    NumericVector phi,
    std::string type_str,
    double tau,
    double rho,
    bool cyclic,
    int n_groups = 1,
    bool augment = false,
    bool tvc = false
) {
  if (tvc) {
    ratiod_tvc::TVCData data;
    data.n_times = phi.size() / n_groups;
    data.n_tvc = 1; data.n_groups = n_groups;
    data.structure = ratiod_temporal::parse_temporal_type(type_str);
    data.cyclic = cyclic;
    return ratiod_tvc::tvc_log_prior(std::vector<double>(phi.begin(),phi.end()),
                                     data,std::vector<double>{tau},std::vector<double>{rho});
  }
  return ratiod_temporal::temporal_log_prior(
    phi.begin(), phi.size() / n_groups, ratiod_temporal::parse_temporal_type(type_str),
    std::log(tau), tau, rho, cyclic, n_groups, augment
  );
}

// [[Rcpp::export]]
double cpp_test_ms_rw_log_lik(NumericVector phi, double sigma2, int order,
                              bool cyclic, bool augment) {
  std::vector<double> v(phi.begin(), phi.end());
  return order == 2
      ? ratiod_temporal::rw2_log_lik(v, sigma2, cyclic, augment)
      : ratiod_temporal::rw1_log_lik(v, sigma2, cyclic, augment);
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

// ---------------------------------------------------------------------------
// hmc_spatiotemporal.h: the GP interaction's field density
//
// The NNGP over a field's index set is EXACT when every location conditions on
// all of its predecessors, so at nn = n - 1 this has to reproduce the dense
// multivariate normal built from the same kernel. That identity is what pins
// the covariance assembly, the ordering and the neighbour Cholesky against the
// definition rather than against themselves; test-st-gp-density.R drives it.
// ---------------------------------------------------------------------------


static ratiod_spatiotemporal::SpatiotemporalData test_st_gp_structure(
    int n, NumericMatrix coords, NumericVector time_values, int nn,
    IntegerVector nn_idx, NumericVector nn_dist_space, NumericVector nn_dist_time,
    IntegerVector nn_order, const std::string& nonsep_type,
    const std::string& cov_space, const std::string& cov_time) {
  ratiod_spatiotemporal::SpatiotemporalData st;
  st.type = ratiod_spatiotemporal::STType::NONSEP_GP;
  st.n_params = n;
  st.nn = nn;
  st.coords.resize(2 * (size_t)n);
  for (int i = 0; i < n; i++) {
    st.coords[(size_t)i * 2 + 0] = coords(i, 0);
    st.coords[(size_t)i * 2 + 1] = coords(i, 1);
  }
  st.time_values = Rcpp::as<std::vector<double>>(time_values);
  st.nn_idx = Rcpp::as<std::vector<int>>(nn_idx);
  st.nn_dist_space = Rcpp::as<std::vector<double>>(nn_dist_space);
  st.nn_dist_time = Rcpp::as<std::vector<double>>(nn_dist_time);
  std::vector<int> ord = Rcpp::as<std::vector<int>>(nn_order);
  st.nn_order.resize(ord.size());
  for (size_t i = 0; i < ord.size(); i++) st.nn_order[i] = ord[i] - 1;
  st.nn_order_inv.assign(ord.size(), 0);
  for (size_t i = 0; i < st.nn_order.size(); i++)
    st.nn_order_inv[st.nn_order[i]] = (int)i;

  if (nonsep_type == "product") {
    st.nonsep_type = ratiod_spatiotemporal::NonsepType::PRODUCT;
  } else if (nonsep_type == "gneiting") {
    st.nonsep_type = ratiod_spatiotemporal::NonsepType::GNEITING;
  } else {
    Rcpp::stop("unknown nonsep_type '%s'", nonsep_type.c_str());
  }
  st.cov_space = ratiod_cov::parse_cov_type(cov_space);
  st.cov_time = ratiod_cov::parse_cov_type(cov_time);
  return st;
}

// [[Rcpp::export]]
double cpp_test_st_gp_log_lik(
    NumericVector w,
    double sigma2,
    double phi_space,
    double phi_time,
    NumericMatrix coords,        // n x 2, by field index
    NumericVector time_values,   // n, by field index
    int nn,
    IntegerVector nn_idx,        // n x nn, row-major, 1-based within ordering
    NumericVector nn_dist_space,
    NumericVector nn_dist_time,
    IntegerVector nn_order,      // 1-based from R, stored 0-based
    std::string nonsep_type = "product",
    std::string cov_space = "exponential",
    std::string cov_time = "exponential"
) {
  const ratiod_spatiotemporal::SpatiotemporalData st = test_st_gp_structure(
      w.size(), coords, time_values, nn, nn_idx, nn_dist_space, nn_dist_time,
      nn_order, nonsep_type, cov_space, cov_time);
  return ratiod_spatiotemporal::st_gp_nngp_log_lik(
      w.begin(), sigma2, phi_space, phi_time, st);
}

// The margin penalty's normalizer on an S x T grid (field index s * T + t),
// with its derivatives in log tau and the two log ranges.
// [[Rcpp::export]]
Rcpp::List cpp_test_st_gp_margin_normalizer(
    double tau,
    double phi_space,
    double phi_time,
    int n_spatial,
    int n_times,
    NumericMatrix coords,
    NumericVector time_values,
    int nn,
    IntegerVector nn_idx,
    NumericVector nn_dist_space,
    NumericVector nn_dist_time,
    IntegerVector nn_order,
    std::string nonsep_type = "gneiting"
) {
  ratiod_spatiotemporal::SpatiotemporalData st = test_st_gp_structure(
      n_spatial * n_times, coords, time_values, nn, nn_idx, nn_dist_space,
      nn_dist_time, nn_order, nonsep_type, "exponential", "exponential");
  st.n_spatial = n_spatial;
  st.n_times = n_times;
  const ratiod_spatiotemporal::StGpMarginNormalizer nz =
      ratiod_spatiotemporal::st_gp_margin_normalizer_grad(tau, phi_space, phi_time, st);
  if (!nz.ok) Rcpp::stop("the normalizer could not be formed");
  return Rcpp::List::create(
      Rcpp::Named("value") = nz.value,
      Rcpp::Named("grad") = Rcpp::NumericVector::create(
          nz.log_tau, nz.log_phi_space, nz.log_phi_time));
}

// The rho term of a Knorr-Held interaction's margin-penalty normalizer on an
// S x T grid with an AR1 time margin, with its rho derivative.
// [[Rcpp::export]]
Rcpp::NumericVector cpp_test_st_ar1_margin_normalizer(
    double rho, int n_spatial, int n_times, std::string type = "II") {
  ratiod_spatiotemporal::SpatiotemporalData st;
  if (type == "II") {
    st.type = ratiod_spatiotemporal::STType::TYPE_II;
  } else if (type == "IV") {
    st.type = ratiod_spatiotemporal::STType::TYPE_IV;
  } else {
    Rcpp::stop("unknown type '%s'", type.c_str());
  }
  st.temporal_type = ratiod_temporal::TemporalType::AR1;
  st.n_spatial = n_spatial;
  st.n_times = n_times;
  return Rcpp::NumericVector::create(
      ratiod_spatiotemporal::st_ar1_margin_log_normalizer(rho, st),
      ratiod_spatiotemporal::st_ar1_margin_log_normalizer_drho(rho, st));
}

// [[Rcpp::export]]
NumericVector cpp_test_pg_rw2_conditional(NumericVector omega, NumericVector residual,
                                         double tau) {
  return ratiod::pg_rw2_draw(Rcpp::as<std::vector<double>>(omega),
    Rcpp::as<std::vector<double>>(residual), tau);
}
