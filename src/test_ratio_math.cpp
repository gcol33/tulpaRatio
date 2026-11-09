#include <Rcpp.h>
#include "ratio_math.h"

template<class T> T prior_density(const std::string& prior, const T& x) {
    using namespace ratiod::math;
    if (prior == "softplus") return log1p_exp(x);
    if (prior == "log_inv_logit") return log_inv_logit(x);
    if (prior == "log1m_inv_logit") return log1m_inv_logit(x);
    if (prior == "gamma") return log_prior_gamma(x, 2.4, 1.7);
    if (prior == "half_cauchy") return log_prior_half_cauchy(x, 1.7);
    if (prior == "pc_variance") return pc_log_density(x, 0.5, 1.7, 0.01);
    return pc_log_density(x, -0.5, 1.7, 0.01);
}

// [[Rcpp::export]]
Rcpp::NumericMatrix cpp_test_ratio_prior(std::string prior, double x) {
    Rcpp::NumericMatrix out(3,2);
    out(0,0) = prior_density(prior,x);
    if (prior == "softplus") out(0,1)=tulpa::math::inv_logit(x);
    else if (prior == "log_inv_logit") out(0,1)=tulpa::math::inv_logit(-x);
    else if (prior == "log1m_inv_logit") out(0,1)=-tulpa::math::inv_logit(x);
    else if (prior == "gamma") out(0,1)=ratiod::math::log_prior_gamma_grad(x,2.4,1.7);
    else if (prior == "half_cauchy") out(0,1)=ratiod::math::log_prior_half_cauchy_grad(x,1.7);
    else if (prior == "pc_variance") out(0,1)=ratiod::math::pc_prior_grad_log_sigma2(std::exp(x),1.7,0.01);
    else out(0,1)=ratiod::math::pc_log_prior_log_precision_grad(x,1.7,0.01);
    const auto f=prior_density(prior,fwd::Dual(x,1)); out(1,0)=f.val;out(1,1)=f.grad;
    tulpa::arena::ArenaScope scope;
    tulpa::arena::Var a(x); auto b=prior_density(prior,a); b.backward();
    out(2,0)=b.val();out(2,1)=a.adj();
    return out;
}
