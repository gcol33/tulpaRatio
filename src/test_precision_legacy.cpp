#include <RcppEigen.h>
#include "autodiff_utils.h"
#include "gibbs_spatial.h"

namespace ratiod_hmc {
Rcpp::NumericMatrix precision_h_probe(double y, double mu, double phi, int family);
}

// [[Rcpp::export]]
Rcpp::NumericMatrix cpp_test_precision_h(double y, double mu, double phi, int family) {
    return ratiod_hmc::precision_h_probe(y, mu, phi, family);
}

template<class T> T precision_legacy_density(double y, const T& mu, const T& phi) {
    return ratiod::math::log_lik_gamma(y, phi, mu);
}

// [[Rcpp::export]]
Rcpp::NumericMatrix cpp_test_precision_legacy(double y, double mu, double phi) {
    Rcpp::NumericMatrix out(3, 2);
    Rcpp::rownames(out) = Rcpp::CharacterVector::create("double", "forward", "arena");
    Rcpp::colnames(out) = Rcpp::CharacterVector::create("value", "dphi");
    out(0, 0) = precision_legacy_density(y, mu, phi);
    auto f = precision_legacy_density(y, fwd::Dual(mu), fwd::Dual(phi, 1));
    out(1, 0) = f.val; out(1, 1) = f.grad;
    ratiod::arena::ArenaScope scope;
    ratiod::arena::Var m(mu), p(phi);
    auto a = precision_legacy_density(y, m, p); a.backward();
    out(2, 0) = a.val(); out(2, 1) = p.adj();
    return out;
}

// [[Rcpp::export]]
Rcpp::NumericVector cpp_test_precision_gibbs(double y, double mu, double phi) {
    int count = 0;
    gibbs::GibbsData data{};
    data.y_num = &count; data.y_denom_cont = &y;
    Rcpp::NumericVector out(2);
    data.family = gibbs::GibbsFamily::POISSON_GAMMA;
    out[0] = gibbs::obs_log_lik(0, 0, std::log(mu), 2, phi, data) + 1;
    data.family = gibbs::GibbsFamily::NEGBIN_GAMMA;
    out[1] = gibbs::obs_log_lik(0, 0, std::log(mu), 2, phi, data) + 2*std::log1p(.5);
    return out;
}

// [[Rcpp::export]]
Rcpp::NumericMatrix cpp_test_precision_beta_legacy(int y, int n, double phi) {
    Rcpp::NumericMatrix out(3,2);
    out(0,0)=ratiod::math::log_lik_beta_binomial(y,n,0.,phi);
    auto f=ratiod::math::log_lik_beta_binomial(y,n,fwd::Dual(0),fwd::Dual(phi,1));
    out(1,0)=f.val; out(1,1)=f.grad;
    ratiod::arena::ArenaScope scope;
    ratiod::arena::Var p(phi), eta(0.);
    auto a=ratiod::math::log_lik_beta_binomial(y,n,eta,p);a.backward();
    out(2,0)=a.val();out(2,1)=p.adj();
    return out;
}
