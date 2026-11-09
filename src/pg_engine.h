#ifndef RATIOD_PG_ENGINE_H
#define RATIOD_PG_ENGINE_H

#include <Rcpp.h>
#include <tulpa/pg_api.h>

namespace ratiod {
struct PGEngineResult {
  tulpa::PGShimResult value{};
  ~PGEngineResult() { value.free_buffers(); }
};

inline Rcpp::NumericMatrix pg_engine_matrix(const double* values, int rows, int cols) {
  Rcpp::NumericMatrix out(rows, cols);
  if (values)
    for (int i = 0; i < rows; ++i)
      for (int j = 0; j < cols; ++j) out(i, j) = values[i * cols + j];
  return out;
}

inline Rcpp::NumericVector pg_engine_vector(const double* values, int rows) {
  Rcpp::NumericVector out(rows);
  if (values) std::copy(values, values + rows, out.begin());
  return out;
}

inline Rcpp::List pg_engine_fit(const tulpa::PGShimResult& r, bool store_eta) {
  Rcpp::List out = Rcpp::List::create(
    Rcpp::_["beta"] = pg_engine_matrix(r.beta, r.n_save, r.n_beta),
    Rcpp::_["re"] = pg_engine_matrix(r.re, r.n_save, r.n_re),
    Rcpp::_["sigma_re"] = pg_engine_vector(r.sigma_re, r.n_save));
  if (r.r_disp) out["r"] = pg_engine_vector(r.r_disp, r.n_save);
  if (r.spatial) {
    out["spatial"] = pg_engine_matrix(r.spatial, r.n_save, r.n_spatial_units);
    out["tau"] = pg_engine_vector(r.tau_spatial, r.n_save);
  }
  if (store_eta) out["eta"] = pg_engine_matrix(r.eta, r.n_save, r.n_obs);
  return out;
}
}
#endif
