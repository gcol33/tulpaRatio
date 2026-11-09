#include <Rcpp.h>
#include "autodiff_utils.h"
#include "taylor_jet.h"

// [[Rcpp::export]]
Rcpp::NumericVector cpp_test_precision_jet(double y, double phi, bool beta) {
  using J=ratiod::jet::Jet3<double>;
  const J precision=J::line(phi,1.0);
  const J result=beta ? ratiod::math::log_lik_beta_binomial(static_cast<int>(y),40,J(0),precision)
                      : ratiod::math::log_lik_gamma(y,precision,J(1));
  return Rcpp::NumericVector::create(result.v,result.d1,result.d2);
}
