// Kriging the GP field onto new locations, for every posterior draw at once.
//
// The kernel is the shared family in hmc_cov.h, the one every fitting path
// reads, so a prediction is made under the covariance its fit was made under.
//
// The neighbour set of a new location is fixed by the coordinates alone, so it
// is found once and reused across draws; only the kernel evaluations and the
// solve move with (sigma2, phi).

#include <Rcpp.h>
#include <RcppEigen.h>

#include <algorithm>
#include <cmath>
#include <vector>

#include "cov_type_code.h"
#include "covariance.h"

// [[Rcpp::depends(RcppEigen)]]

namespace {

// The jitter the R implementation added to the neighbour block before solving.
constexpr double KRIGE_JITTER = 1e-6;

double coord_dist(const Rcpp::NumericMatrix& a, int ia,
                  const Rcpp::NumericMatrix& b, int ib) {
  const double dx = a(ia, 0) - b(ib, 0);
  const double dy = a(ia, 1) - b(ib, 1);
  return std::sqrt(dx * dx + dy * dy);
}

}  // namespace

// [[Rcpp::export]]
Rcpp::NumericMatrix cpp_kriging_predict(
    Rcpp::NumericMatrix coords_train,
    Rcpp::NumericMatrix coords_new,
    Rcpp::NumericMatrix w_train,
    Rcpp::NumericVector sigma2,
    Rcpp::NumericVector phi,
    int cov_type,
    int nn
) {
  const int n_train = coords_train.nrow();
  const int n_new = coords_new.nrow();
  const int n_draws = w_train.nrow();

  if (coords_train.ncol() < 2 || coords_new.ncol() < 2) {
    Rcpp::stop("cpp_kriging_predict: coordinates need two columns; got %d "
               "training and %d prediction.",
               static_cast<int>(coords_train.ncol()),
               static_cast<int>(coords_new.ncol()));
  }
  if (n_train < 1) {
    Rcpp::stop("cpp_kriging_predict: no training locations.");
  }
  if (w_train.ncol() != n_train) {
    Rcpp::stop("cpp_kriging_predict: the field has %d columns against %d "
               "training locations. It is indexed by location, so its width is "
               "the number of unique coordinates and not the number of rows of "
               "data.", static_cast<int>(w_train.ncol()), n_train);
  }
  if (sigma2.size() != n_draws || phi.size() != n_draws) {
    Rcpp::stop("cpp_kriging_predict: %d field draws against %d sigma2 and %d "
               "phi. The three are row-aligned draws of one posterior.",
               n_draws, static_cast<int>(sigma2.size()),
               static_cast<int>(phi.size()));
  }

  const ratiod_cov::CovType cov = ratiod_cov::cov_type_from_int(cov_type);
  const int k = std::min(std::max(nn, 1), n_train);

  // The neighbour set and its geometry, once per new location.
  std::vector<int> nb_idx(static_cast<size_t>(n_new) * k);
  std::vector<double> nb_dist(static_cast<size_t>(n_new) * k);
  std::vector<double> nb_pair(static_cast<size_t>(n_new) * k * k);

  std::vector<int> order(static_cast<size_t>(n_train));
  std::vector<double> dist(static_cast<size_t>(n_train));

  for (int i = 0; i < n_new; i++) {
    for (int t = 0; t < n_train; t++) {
      order[t] = t;
      dist[t] = coord_dist(coords_new, i, coords_train, t);
    }
    // Ties broken by index, which is what R's order() does, so a fit whose
    // locations coincide picks the same neighbours here as anywhere else.
    std::partial_sort(order.begin(), order.begin() + k, order.end(),
                      [&dist](int a, int b) {
                        if (dist[a] != dist[b]) return dist[a] < dist[b];
                        return a < b;
                      });

    for (int j = 0; j < k; j++) {
      const size_t slot = static_cast<size_t>(i) * k + j;
      nb_idx[slot] = order[j];
      nb_dist[slot] = dist[order[j]];
    }
    for (int a = 0; a < k; a++) {
      for (int b = 0; b < k; b++) {
        const size_t slot = (static_cast<size_t>(i) * k + a) * k + b;
        nb_pair[slot] = (a == b)
          ? 0.0
          : coord_dist(coords_train, nb_idx[static_cast<size_t>(i) * k + a],
                       coords_train, nb_idx[static_cast<size_t>(i) * k + b]);
      }
    }
  }

  Rcpp::NumericMatrix out(n_draws, n_new);
  Eigen::MatrixXd C_SS(k, k);
  Eigen::VectorXd c_s0(k);

  for (int s = 0; s < n_draws; s++) {
    const double s2 = sigma2[s];
    const double ph = phi[s];

    for (int i = 0; i < n_new; i++) {
      for (int a = 0; a < k; a++) {
        const size_t slot = static_cast<size_t>(i) * k + a;
        c_s0(a) = ratiod_cov::compute_cov(nb_dist[slot], s2, ph, cov);
        for (int b = 0; b < k; b++) {
          const size_t pair = (static_cast<size_t>(i) * k + a) * k + b;
          C_SS(a, b) = (a == b)
            ? s2 + KRIGE_JITTER
            : ratiod_cov::compute_cov(nb_pair[pair], s2, ph, cov);
        }
      }

      const Eigen::VectorXd weights = C_SS.partialPivLu().solve(c_s0);

      double acc = 0.0;
      for (int a = 0; a < k; a++) {
        acc += weights(a) * w_train(s, nb_idx[static_cast<size_t>(i) * k + a]);
      }
      if (!std::isfinite(acc)) {
        Rcpp::stop("cpp_kriging_predict: the neighbour block at prediction "
                   "location %d is singular at draw %d (sigma2 = %g, phi = %g).",
                   i + 1, s + 1, s2, ph);
      }
      out(s, i) = acc;
    }
  }

  return out;
}
