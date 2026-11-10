#include <Rcpp.h>

#include "ratio_margins.h"

// [[Rcpp::export]]
Rcpp::NumericVector cpp_test_linked_margin(std::string family, double y, int n,
    double eta, double phi, std::string link, std::string inflation = "none",
    double zi = -0.6, double oi = -1.0, std::string zi_link = "logit",
    std::string oi_link = "logit") {
    using namespace tulpaRatio::margins;
    ratiod_link::LinkType l, zl, ol;
    if (!ratiod_link::parse_link(link, l) || !ratiod_link::parse_link(zi_link, zl) ||
        !ratiod_link::parse_link(oi_link, ol)) Rcpp::stop("Unknown link");
    const auto type = static_cast<tulpa::ZIType>(ratiod_zi::parse_zi_type(inflation));
    Result r;
    if (family == "binomial") r = linked_result<Binomial>(y,n,eta,phi,type,zi,oi,l,zl,ol);
    else if (family == "beta_binomial") r = linked_result<BetaBinomial>(y,n,eta,phi,type,zi,oi,l,zl,ol);
    else if (family == "poisson") r = linked_result<Poisson>(y,n,eta,phi,type,zi,oi,l,zl,ol);
    else if (family == "negbin") r = linked_result<NegBin>(y,n,eta,phi,type,zi,oi,l,zl,ol);
    else if (family == "gamma") r = linked_result<Gamma>(y,n,eta,phi,type,zi,oi,l,zl,ol);
    else if (family == "lognormal") r = linked_result<Lognormal>(y,n,eta,phi,type,zi,oi,l,zl,ol);
    else Rcpp::stop("Unknown margin");
    return Rcpp::NumericVector::create(Rcpp::Named("ll")=r.ll,
        Rcpp::Named("score")=r.residual,Rcpp::Named("extra")=r.extra,
        Rcpp::Named("curvature")=r.neg_hess,Rcpp::Named("zi")=r.zi,Rcpp::Named("oi")=r.oi);
}

namespace ratiod_hmc { Rcpp::NumericMatrix residual_accumulation_probe(); }
namespace ratiod_hmc {
Rcpp::NumericMatrix collapsed_margin_probe(int,int,double,double,double,double,double,bool);
Rcpp::NumericVector collapsed_gp_reference_probe(Rcpp::NumericVector,int,int,double);
}

// [[Rcpp::export]]
Rcpp::NumericVector cpp_test_collapsed_gp_reference(Rcpp::NumericVector beta,int y,int n,double sigma) {
    return ratiod_hmc::collapsed_gp_reference_probe(beta,y,n,sigma);
}

// [[Rcpp::export]]
Rcpp::NumericMatrix cpp_test_collapsed_margin(int family, int y, double yd, double en,
    double ed, double pn, double pd, bool shared) {
    return ratiod_hmc::collapsed_margin_probe(family,y,yd,en,ed,pn,pd,shared);
}

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
        return cached ? m.template evaluate<true>(x) : m.template evaluate<false>(x);
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

// [[Rcpp::export]]
Rcpp::NumericVector cpp_test_margin_fields(std::string family, double eta,
                                          double phi, int fields, bool cached) {
    using namespace tulpaRatio::margins;
    Observation x{3.,eta,8};
    x.log_y=std::log(3.);
    x.log_factorial=family=="binomial" || family=="beta_binomial"
        ? tulpa::math::portable_lchoose(8,3) : std::lgamma(4.);
    x.digamma_difference=tulpa::math::polygamma_rising_difference(0,3,phi);
    x.lgamma_difference=tulpa::math::lgamma_ratio_count(3,phi);
    const auto run=[&](const auto& m) {
        const auto select=[&](auto mask) {
            constexpr unsigned want=decltype(mask)::value;
            return cached ? m.template evaluate<true,want>(x) : m.template evaluate<false,want>(x);
        };
        switch (fields) {
            case Value: return select(std::integral_constant<unsigned,Value>{});
            case Score: return select(std::integral_constant<unsigned,Score>{});
            case Extra: return select(std::integral_constant<unsigned,Extra>{});
            case Curvature: return select(std::integral_constant<unsigned,Curvature>{});
            default: return select(std::integral_constant<unsigned,FirstOrder|Curvature>{});
        }
    };
    Result r;
    if (family=="poisson") r=run(Poisson());
    else if (family=="negbin") r=run(NegBin(phi));
    else if (family=="gamma") r=run(Gamma(phi));
    else if (family=="lognormal") r=run(Lognormal(phi));
    else if (family=="binomial") r=run(Binomial());
    else if (family=="beta_binomial") r=run(BetaBinomial(phi));
    else Rcpp::stop("Unknown margin");
    return Rcpp::NumericVector::create(r.ll,r.residual,r.extra,r.neg_hess);
}

// [[Rcpp::export]]
Rcpp::NumericVector cpp_test_margin_curvature(std::string family,double eta,double phi) {
    using namespace tulpaRatio::margins;
    const Observation x{3.,eta,8};
    CurvatureDerivatives d;
    if (family=="poisson") d=curvature_derivatives<Poisson>(x,phi);
    else if (family=="negbin") d=curvature_derivatives<NegBin>(x,phi);
    else if (family=="gamma") d=curvature_derivatives<Gamma>(x,phi);
    else if (family=="lognormal") d=curvature_derivatives<Lognormal>(x,phi);
    else if (family=="binomial") d=curvature_derivatives<Binomial>(x,phi);
    else if (family=="beta_binomial") d=curvature_derivatives<BetaBinomial>(x,phi);
    else Rcpp::stop("Unknown margin");
    return Rcpp::NumericVector::create(d.w,d.eta,d.phi,d.score_phi,d.score);
}
