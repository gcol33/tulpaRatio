#include <Rcpp.h>
#include "ratio_margins.h"

namespace ratiod_hmc { Rcpp::NumericMatrix residual_accumulation_probe(); }

// [[Rcpp::export]]
Rcpp::NumericMatrix cpp_test_residual_accumulation() {
    return ratiod_hmc::residual_accumulation_probe();
}

// [[Rcpp::export]]
Rcpp::NumericVector cpp_test_ratio_margin(std::string family, double y, int n,
                                          double eta, double phi, bool cached) {
    using namespace tulpaRatio::margins;
    Observation x{y,eta,n};
    x.log_y = y > 0 ? std::log(y) : 0.0;
    x.log_factorial = std::lgamma(y+1.0);
    if (family == "binomial" || family == "beta_binomial")
        x.log_factorial = tulpa::math::portable_lchoose(n,int(y));
    x.digamma_difference = tulpa::math::polygamma_rising_difference(0,int(y),phi);
    x.lgamma_difference = tulpa::math::lgamma_ratio_count(int(y),phi);
    const auto evaluate = [&](const auto& m) {
        return cached ? m.template operator()<true>(x) : m.template operator()<false>(x);
    };
    Result r;
    if (family == "poisson") r = evaluate(Poisson());
    else if (family == "negbin") r = evaluate(NegBin(phi));
    else if (family == "gamma") r = evaluate(Gamma(phi));
    else if (family == "lognormal") r = evaluate(Lognormal(phi));
    else if (family == "binomial") r = evaluate(Binomial());
    else if (family == "beta_binomial") r = evaluate(BetaBinomial(phi));
    else Rcpp::stop("Unknown margin");
    return Rcpp::NumericVector::create(Rcpp::Named("ll")=r.ll,
        Rcpp::Named("residual")=r.residual,Rcpp::Named("extra")=r.extra);
}

template<class T>
T margin_density(const std::string& family, double y, int n, const T& eta, const T& phi) {
    using namespace ratiod::math;
    if (family == "poisson") return log_lik_poisson(int(y),safe_exp(eta));
    if (family == "negbin") return log_lik_negbin(int(y),safe_exp(eta),phi);
    if (family == "gamma") return log_lik_gamma(y,phi,safe_exp(eta));
    if (family == "lognormal") return log_lik_lognormal(y,eta,phi);
    if (family == "binomial") return log_lik_binomial(int(y),n,eta);
    return log_lik_beta_binomial(int(y),n,eta,phi);
}

// [[Rcpp::export]]
Rcpp::NumericMatrix cpp_test_ratio_margin_scalars(std::string family, double y, int n,
                                                 double eta, double phi) {
    Rcpp::NumericMatrix out(3,3);
    out(0,0)=margin_density(family,y,n,eta,phi);
    const auto e=margin_density(family,y,n,fwd::Dual(eta,1),fwd::Dual(phi));
    const auto p=margin_density(family,y,n,fwd::Dual(eta),fwd::Dual(phi,1));
    out(1,0)=e.val;out(1,1)=e.grad;out(1,2)=p.grad;
    tulpa::arena::ArenaScope scope;
    tulpa::arena::Var a(eta),b(phi);
    auto c=margin_density(family,y,n,a,b);c.backward();
    out(2,0)=c.val();out(2,1)=a.adj();out(2,2)=b.adj();
    return out;
}
