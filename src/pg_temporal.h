#ifndef RATIOD_PG_TEMPORAL_H
#define RATIOD_PG_TEMPORAL_H

#include "pg_common.h"
#include "scale_prior.h"

namespace ratiod {

// The noncyclic RW2 precision is D2' D2 and has rank n - 2.
inline Rcpp::NumericVector pg_rw2_draw(const std::vector<double>& weights,
                                      const std::vector<double>& residual,
                                      double tau) {
  const int n = weights.size();
  if (n < 3) Rcpp::stop("An RW2 trend requires at least three time points.");
  Eigen::MatrixXd precision = Eigen::MatrixXd::Zero(n, n);
  Eigen::VectorXd rhs(n);
  for (int t = 0; t < n; ++t) {
    precision(t, t) = weights[t];
    rhs[t] = residual[t];
  }
  const double stencil[] = {1.0, -2.0, 1.0};
  for (int t = 0; t < n - 2; ++t)
    for (int j = 0; j < 3; ++j)
      for (int k = 0; k < 3; ++k)
        precision(t + j, t + k) += tau * stencil[j] * stencil[k];
  auto factor = pg_factor(precision);
  Eigen::VectorXd z(n);
  for (int t = 0; t < n; ++t) z[t] = R::rnorm(0.0, 1.0);
  return Rcpp::wrap(Eigen::VectorXd(factor.solve(rhs) + factor.matrixU().solve(z)));
}

inline double pg_rw2_scale(const Rcpp::NumericVector& trend, double sigma,
                            double U, double alpha) {
  if (!(U > 0.0) || !(alpha > 0.0 && alpha < 1.0))
    Rcpp::stop("RW2 scale prior requires U > 0 and 0 < alpha < 1.");
  double quadratic = 0.0;
  for (int t = 2; t < trend.size(); ++t) {
    const double difference = trend[t] - 2.0 * trend[t - 1] + trend[t - 2];
    quadratic += difference * difference;
  }
  const double tau = ratiod_scale::sample_tau(1.0 / (sigma * sigma), quadratic,
    trend.size() - 2, U, alpha, []() { return R::unif_rand(); });
  return 1.0 / std::sqrt(tau);
}

}
#endif
