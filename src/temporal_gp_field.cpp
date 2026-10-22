// temporal_gp_field.cpp
// The temporal GP effects behind each posterior draw, through the density the
// sampler evaluates.

#include <Rcpp.h>
#include <RcppEigen.h>
#include "hmc_sampler.h"
#include "hmc_model_data_blocks.h"

// [[Rcpp::export]]
Rcpp::NumericMatrix cpp_temporal_gp_field(const Rcpp::NumericMatrix& x,
                                          const Rcpp::NumericVector& log_sigma2,
                                          const Rcpp::NumericVector& logit_phi,
                                          const Rcpp::List& temporal_params) {
  const ratiod_temporal_gp::TemporalGPData gp =
      ratiod_hmc::temporal_gp_data_from(temporal_params);
  const int n_draws = x.nrow();
  const int n_eff = x.ncol();
  if (n_eff != gp.n_groups * gp.n_times) {
    Rcpp::stop("Temporal GP draws carry %d effects for %d groups of %d times.",
               n_eff, gp.n_groups, gp.n_times);
  }
  Rcpp::NumericMatrix f_out(n_draws, n_eff);
  std::vector<double> row(n_eff), f;
  for (int s = 0; s < n_draws; s++) {
    for (int k = 0; k < n_eff; k++) row[k] = x(s, k);
    ratiod_temporal_gp::temporal_gp_log_prior(row.data(), log_sigma2[s],
                                              logit_phi[s], gp, f);
    for (int k = 0; k < n_eff; k++) f_out(s, k) = f[k];
  }
  return f_out;
}

// The block's log density at one state, with the effects it puts into eta.
// [[Rcpp::export]]
Rcpp::List cpp_temporal_gp_log_prior(const Rcpp::NumericVector& x, double log_sigma2,
                                     double logit_phi, const Rcpp::List& temporal_params) {
  const ratiod_temporal_gp::TemporalGPData gp =
      ratiod_hmc::temporal_gp_data_from(temporal_params);
  std::vector<double> xv(x.begin(), x.end()), f;
  const double lp = ratiod_temporal_gp::temporal_gp_log_prior(xv.data(), log_sigma2,
                                                              logit_phi, gp, f);
  return Rcpp::List::create(Rcpp::Named("log_prior") = lp,
                            Rcpp::Named("f") = Rcpp::wrap(f));
}
