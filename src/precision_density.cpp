#include <Rcpp.h>
#include <tulpa/count_math.h>
#include <tulpa/precision_math.h>
#include <algorithm>

// Elementwise density with R's recycling of draw-level precisions over observations.
// [[Rcpp::export]]
Rcpp::NumericVector cpp_ratio_precision_density(const Rcpp::NumericVector& y,
    const Rcpp::NumericVector& mu, const Rcpp::NumericVector& phi,
    const Rcpp::NumericVector& trials) {
  if (y.size()==0 || mu.size()==0 || phi.size()==0) return Rcpp::NumericVector(0);
  const R_xlen_t length=std::max({y.size(),mu.size(),phi.size(),trials.size()});
  Rcpp::NumericVector out(length);
  for (R_xlen_t i=0; i<length; ++i) {
    const double response=y[i%y.size()], mean=mu[i%mu.size()], precision=phi[i%phi.size()];
    if (trials.size()) {
      const int n=trials[i%trials.size()], k=static_cast<int>(response);
      out[i]=tulpa::math::beta_binomial_log_pmf(k,n,mean,1-mean,precision,
        std::lgamma(n+1.)-std::lgamma(k+1.)-std::lgamma(n-k+1.));
    } else {
      out[i]=tulpa::math::gamma_log_density(response,mean,precision,std::log(response));
    }
  }
  return out;
}
