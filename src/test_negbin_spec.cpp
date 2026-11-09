#include <Rcpp.h>
#include "lik_specs/lik_helpers.h"

// [[Rcpp::export]]
Rcpp::NumericVector cpp_test_negbin_math(int y, double phi) {
  Rcpp::NumericVector out(5);
  out[0] = tulpaRatio::count_math::lgamma_ratio_count(y, phi);
  for (int k = 0; k < 4; ++k) out[k + 1] = tulpaRatio::count_math::polygamma_difference(k, y, phi);
  return out;
}

template <typename T>
T negbin_spec_probe_density(int y, const T& eta, const T& log_phi, const T& z,
                            const std::string& type) {
  using std::exp;
  const T mu = exp(eta), phi = exp(log_phi);
  if (type == "zi_negbin") return tulpaRatio::lik::zi_negbin_log_pmf(y, mu, phi, z);
  if (type == "hurdle_negbin") return tulpaRatio::lik::hurdle_negbin_log_pmf(y, mu, phi, z);
  return tulpaRatio::lik::negbin_log_pmf(y, mu, phi);
}

// [[Rcpp::export]]
Rcpp::NumericMatrix cpp_test_negbin_spec(int y, double mu, double phi, double z,
                                       std::string type = "zi_negbin") {
  const double eta = std::log(mu), lp = std::log(phi);
  Rcpp::NumericMatrix out(3, 4);
  rownames(out) = Rcpp::CharacterVector::create("spec_double", "spec_forward", "spec_arena");
  colnames(out) = Rcpp::CharacterVector::create("value", "eta", "log_phi", "zi");
  out(0, 0) = negbin_spec_probe_density(y, eta, lp, z, type);
  for (int j = 0; j < 3; ++j) {
    const auto v = negbin_spec_probe_density(y, fwd::Dual(eta, j == 0),
        fwd::Dual(lp, j == 1), fwd::Dual(z, j == 2), type);
    out(1, 0) = v.val; out(1, j + 1) = v.grad;
  }
  tulpa::arena::ArenaScope scope;
  tulpa::arena::Var a(eta), b(lp), c(z);
  auto v = negbin_spec_probe_density(y, a, b, c, type);
  v.backward();
  out(2, 0) = v.val(); out(2, 1) = a.adj(); out(2, 2) = b.adj(); out(2, 3) = c.adj();
  return out;
}
