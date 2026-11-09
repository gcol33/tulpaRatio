#include <Rcpp.h>
#include "autodiff_utils.h"

template<class T>
T scalar_operation(const std::string& op, const T& x) {
    using namespace tulpa::math;
    if (op == "exp") return safe_exp(x);
    if (op == "log") return safe_log(x);
    if (op == "sqrt") return safe_sqrt(x);
    if (op == "pow") { using std::pow; return pow(x, 0.5); }
    if (op == "pow_dual") {
        if constexpr (std::is_same<T,fwd::Dual>::value) return fwd::pow(x,fwd::Dual(2.0));
        else { using std::pow; return pow(x,2.0); }
    }
    if (op == "expm1") { using std::expm1; return expm1(x); }
    if (op == "lift") return lift_unary(x, 3.0, 7.0);
    Rcpp::stop("Unknown scalar operation");
}

// [[Rcpp::export]]
Rcpp::NumericMatrix cpp_test_scalar_math(std::string op, double value) {
    Rcpp::NumericMatrix out(3,2);
    Rcpp::rownames(out) = Rcpp::CharacterVector::create("double","forward","arena");
    Rcpp::colnames(out) = Rcpp::CharacterVector::create("value","derivative");
    out(0,0) = scalar_operation(op,value);
    out(0,1) = NA_REAL;
    const auto f = scalar_operation(op,fwd::Dual(value,1.0));
    out(1,0) = f.val; out(1,1) = f.grad;
    tulpa::arena::ArenaScope scope;
    tulpa::arena::Var x(value);
    auto a = scalar_operation(op,x); a.backward();
    out(2,0) = a.val(); out(2,1) = x.adj();
    return out;
}

// [[Rcpp::export]]
Rcpp::NumericMatrix cpp_test_scalar_boundaries(double probability) {
    Rcpp::NumericMatrix out(3,4);
    Rcpp::rownames(out) = Rcpp::CharacterVector::create("double","forward","arena");
    const double p = tulpa::math::clamp_prob(probability);
    const double ni = -std::numeric_limits<double>::infinity();
    out(0,0) = std::log(p/(1-p)); out(0,1) = NA_REAL;
    out(0,2) = ratiod::math::log_add_exp(ni,ni); out(0,3) = NA_REAL;
    const auto f = fwd::logit(fwd::Dual(probability,1));
    const auto g = fwd::log_sum_exp(fwd::Dual(ni,1),fwd::Dual(ni));
    out(1,0)=f.val; out(1,1)=f.grad; out(1,2)=g.val; out(1,3)=g.grad;
    tulpa::arena::ArenaScope scope;
    tulpa::arena::Var x(probability), z(ni);
    auto a=tulpa::arena::logit(x); a.backward();
    out(2,0)=a.val(); out(2,1)=x.adj();
    auto b=tulpa::arena::log_sum_exp(z,z); b.backward();
    out(2,2)=b.val(); out(2,3)=z.adj();
    return out;
}
