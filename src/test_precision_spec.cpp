#include <Rcpp.h>
#include "lik_specs/lik_helpers.h"

namespace tulpaRatio {
Rcpp::NumericMatrix precision_spec_h_probe(double y, double mu, double phi);
Rcpp::NumericVector precision_spec_beta_h_probe(int y, int n, double phi);
}

// [[Rcpp::export]]
Rcpp::NumericMatrix cpp_test_precision_spec_h(double y, double mu, double phi) {
    return tulpaRatio::precision_spec_h_probe(y, mu, phi);
}

// [[Rcpp::export]]
Rcpp::NumericMatrix cpp_test_precision_spec(double y, double mu, double phi) {
    Rcpp::NumericMatrix out(3, 2);
    Rcpp::rownames(out) = Rcpp::CharacterVector::create("double", "forward", "arena");
    Rcpp::colnames(out) = Rcpp::CharacterVector::create("value", "dphi");
    out(0, 0) = ratiod::math::log_lik_gamma(y, phi, mu);
    auto f = ratiod::math::log_lik_gamma(y, fwd::Dual(phi, 1), fwd::Dual(mu));
    out(1, 0) = f.val; out(1, 1) = f.grad;
    tulpa::arena::ArenaScope scope;
    tulpa::arena::Var p(phi), m(mu);
    auto a = ratiod::math::log_lik_gamma(y, p, m); a.backward();
    out(2, 0) = a.val(); out(2, 1) = p.adj();
    return out;
}

// [[Rcpp::export]]
Rcpp::NumericMatrix cpp_test_precision_beta_spec(int y, int n, double phi) {
    Rcpp::NumericMatrix out(3,2);
    out(0,0)=ratiod::math::log_lik_beta_binomial(y,n,0.,phi);
    auto f=ratiod::math::log_lik_beta_binomial(y,n,fwd::Dual(0),fwd::Dual(phi,1));
    out(1,0)=f.val;out(1,1)=f.grad;
    tulpa::arena::ArenaScope scope;
    tulpa::arena::Var p(phi), eta(0.);
    auto a=ratiod::math::log_lik_beta_binomial(y,n,eta,p);a.backward();
    out(2,0)=a.val();out(2,1)=p.adj();
    return out;
}

// [[Rcpp::export]]
Rcpp::NumericVector cpp_test_precision_beta_spec_h(int y, int n, double phi) {
    return tulpaRatio::precision_spec_beta_h_probe(y,n,phi);
}
