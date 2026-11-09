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
