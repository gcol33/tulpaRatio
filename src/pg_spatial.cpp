// pg_spatial.cpp
// Spatial random effects for PG Gibbs sampler
// Implements ICAR (Intrinsic CAR) prior for areal data

#include "pg_spatial.h"
#include <Rcpp.h>
#include <cmath>
#include <limits>
#include <vector>

using namespace Rcpp;

namespace ratiod {

// ---------------------------------------------------------------------
// ICAR (Intrinsic Conditional Autoregressive) prior
// ---------------------------------------------------------------------

// Update spatial effects with ICAR prior
// Each phi_i is updated given all other phi values and the data
//
// Full conditional for phi_i:
// phi_i | ... ~ N(m_i, v_i)
// v_i = 1 / (tau * n_i + sum(omega_j for j in group i))
// m_i = v_i * (tau * sum(phi_j for neighbors j) + sum(kappa_j - omega_j * offset_j for j in group i))
//
// For group-level spatial effects:
// - Each group j has a spatial effect phi_j
// - Observations in group j share phi_j
// - ICAR prior relates phi across spatial units

NumericVector update_spatial_icar(
    const NumericVector& kappa,
    const NumericVector& omega,
    const NumericVector& offset,
    const IntegerVector& group,
    const List& adj_list,
    const IntegerVector& n_neighbors,
    double tau
) {
  int N = kappa.size();
  int J = adj_list.size();  // Number of spatial units
  NumericVector phi(J);

  // Accumulate data sufficient statistics by group
  NumericVector sum_omega(J);
  NumericVector sum_resid(J);

  for (int i = 0; i < N; i++) {
    int g = group[i] - 1;  // Convert to 0-based
    sum_omega[g] += omega[i];
    sum_resid[g] += kappa[i] - omega[i] * offset[i];
  }

  // Single-site Gibbs updates for each spatial unit
  // Could be improved with block updates, but this is simple and works
  for (int j = 0; j < J; j++) {
    // Get neighbors - use eager copy to avoid GC issues with lazy views
    std::vector<int> neighbors = Rcpp::as<std::vector<int>>(adj_list[j]);
    int n_j = n_neighbors[j];

    // Sum of neighbor values
    double neighbor_sum = 0.0;
    for (int k = 0; k < n_j; k++) {
      int neighbor_idx = neighbors[k] - 1;  // Convert to 0-based
      neighbor_sum += phi[neighbor_idx];
    }

    // Full conditional parameters
    double prec = tau * n_j + sum_omega[j];
    double mean_num = tau * neighbor_sum + sum_resid[j];

    // Handle isolated units (no neighbors)
    if (n_j == 0) {
      // Fall back to exchangeable prior with the data
      if (sum_omega[j] > 0) {
        prec = sum_omega[j] + 0.001;  // Small prior precision
        mean_num = sum_resid[j];
      } else {
        phi[j] = 0.0;
        continue;
      }
    }

    double post_var = 1.0 / prec;
    double post_mean = mean_num / prec;

    phi[j] = R::rnorm(post_mean, std::sqrt(post_var));
  }

  // Center the spatial effects (soft sum-to-zero)
  double mean_phi = 0.0;
  for (int j = 0; j < J; j++) {
    mean_phi += phi[j];
  }
  mean_phi /= J;

  for (int j = 0; j < J; j++) {
    phi[j] -= mean_phi;
  }

  return phi;
}

// Update spatial precision tau with gamma prior
// ICAR precision structure: phi' Q phi where Q_ii = n_neighbors[i], Q_ij = -1 if neighbors
//
// Posterior for tau:
// tau | phi ~ Gamma(a + (J-1)/2, b + 0.5 * phi' Q phi)

double update_tau_icar(
    const NumericVector& phi,
    const List& adj_list,
    const IntegerVector& n_neighbors,
    double prior_shape,
    double prior_rate
) {
  int J = phi.size();

  // Compute phi' Q phi = sum_i n_i * phi_i^2 - 2 * sum_{i~j} phi_i * phi_j
  // = sum_i n_i * phi_i^2 - sum_i phi_i * sum_{j~i} phi_j
  double quad_form = 0.0;

  for (int i = 0; i < J; i++) {
    // Diagonal contribution: n_i * phi_i^2
    quad_form += n_neighbors[i] * phi[i] * phi[i];

    // Off-diagonal contribution: -2 * sum over neighbors (count each edge once)
    std::vector<int> neighbors = Rcpp::as<std::vector<int>>(adj_list[i]);
    for (int k = 0; k < n_neighbors[i]; k++) {
      int j = neighbors[k] - 1;
      if (j > i) {  // Count each edge once
        quad_form -= 2.0 * phi[i] * phi[j];
      }
    }
  }

  // Posterior parameters
  // Note: ICAR has rank J-1 (one constraint), so degrees of freedom is (J-1)/2
  double post_shape = prior_shape + (J - 1.0) / 2.0;
  double post_rate = prior_rate + quad_form / 2.0;

  return R::rgamma(post_shape, 1.0 / post_rate);
}

// ---------------------------------------------------------------------
// BYM2 field
// ---------------------------------------------------------------------

// Univariate slice sampler with stepping out and shrinkage (Neal 2003,
// Ann. Stat. 31:705, Figs. 3 and 5).
template <typename LogDensity>
static double slice_sample(double x0, LogDensity log_dens, double width) {
  const int max_steps = 50;
  const double log_y = log_dens(x0) + std::log(R::unif_rand());

  double left = x0 - width * R::unif_rand();
  double right = left + width;
  int j = static_cast<int>(std::floor(max_steps * R::unif_rand()));
  int k = max_steps - 1 - j;
  while (j-- > 0 && log_dens(left) > log_y) left -= width;
  while (k-- > 0 && log_dens(right) > log_y) right += width;

  for (;;) {
    const double x1 = left + R::unif_rand() * (right - left);
    if (log_dens(x1) > log_y) return x1;
    if (x1 < x0) left = x1; else right = x1;
  }
}

PGBym2::PGBym2(const List& adj_list, double scale_factor, double sigma_scale)
  : sigma(1.0), rho(0.5), J_(adj_list.size()),
    scale_factor_(scale_factor), sigma_scale_(sigma_scale) {
  phi = Eigen::VectorXd::Zero(J_);
  theta = Eigen::VectorXd::Zero(J_);

  std::vector<Eigen::Triplet<double>> trip;
  for (int j = 0; j < J_; j++) {
    IntegerVector nb = adj_list[j];
    trip.emplace_back(j, j, static_cast<double>(nb.size()));
    for (int k = 0; k < nb.size(); k++) trip.emplace_back(j, nb[k] - 1, -1.0);
  }
  Q_icar_.resize(J_, J_);
  Q_icar_.setFromTriplets(trip.begin(), trip.end());
  llt_.analyzePattern(Q_icar_);
}

Eigen::VectorXd PGBym2::effect() const {
  return sigma * (std::sqrt(rho) * scale_factor_ * phi + std::sqrt(1.0 - rho) * theta);
}

// phi | rest ~ N(Q^-1 r, Q^-1) restricted to sum(phi) = 0, with
// Q = (D - W) + a^2 diag(S), r_s = a (R_s - S_s b theta_s), drawn by
// conditioning by kriging (Rue and Held 2005, Sec. 2.3.3).
void PGBym2::update_phi(const Eigen::VectorXd& S, const Eigen::VectorXd& R) {
  const double a = sigma * std::sqrt(rho) * scale_factor_;
  const double b = sigma * std::sqrt(1.0 - rho);

  Eigen::SparseMatrix<double> Q = Q_icar_;
  for (int j = 0; j < J_; j++) Q.coeffRef(j, j) += a * a * S[j];
  llt_.factorize(Q);
  if (llt_.info() != Eigen::Success) {
    Rcpp::stop("BYM2 conditional precision is not positive definite.");
  }

  const Eigen::VectorXd r = a * (R - b * S.cwiseProduct(theta));
  Eigen::VectorXd z(J_);
  for (int j = 0; j < J_; j++) z[j] = R::norm_rand();
  const Eigen::VectorXd noise = llt_.permutationPinv() * Eigen::VectorXd(llt_.matrixU().solve(z));
  const Eigen::VectorXd x = llt_.solve(r) + noise;

  const Eigen::VectorXd v = llt_.solve(Eigen::VectorXd::Ones(J_));
  phi = x - v * (x.sum() / v.sum());
}

// theta_s | rest ~ N(b (R_s - S_s a phi_s) / (1 + b^2 S_s), 1 / (1 + b^2 S_s))
void PGBym2::update_theta(const Eigen::VectorXd& S, const Eigen::VectorXd& R) {
  const double a = sigma * std::sqrt(rho) * scale_factor_;
  const double b = sigma * std::sqrt(1.0 - rho);
  for (int j = 0; j < J_; j++) {
    const double prec = 1.0 + b * b * S[j];
    const double mean = b * (R[j] - S[j] * a * phi[j]) / prec;
    theta[j] = mean + R::norm_rand() / std::sqrt(prec);
  }
}

// log sigma | rest: the field is sigma * w, so the likelihood is
// -A sigma^2 / 2 + B sigma with A = sum S w^2, B = sum R w.
void PGBym2::update_sigma(const Eigen::VectorXd& S, const Eigen::VectorXd& R) {
  const Eigen::VectorXd w =
    std::sqrt(rho) * scale_factor_ * phi + std::sqrt(1.0 - rho) * theta;
  const double A = S.dot(w.cwiseProduct(w));
  const double B = R.dot(w);
  const double s2 = sigma_scale_ * sigma_scale_;
  auto log_dens = [&](double log_s) {
    const double s = std::exp(log_s);
    return -0.5 * A * s * s + B * s - std::log1p(s * s / s2) + log_s;
  };
  sigma = std::exp(slice_sample(std::log(sigma), log_dens, 1.0));
}

// logit rho | rest, uniform prior on rho with the logit Jacobian
void PGBym2::update_rho(const Eigen::VectorXd& S, const Eigen::VectorXd& R) {
  const Eigen::VectorXd sphi = sigma * scale_factor_ * phi;
  const Eigen::VectorXd stheta = sigma * theta;
  auto log_dens = [&](double lr) {
    const double r = 1.0 / (1.0 + std::exp(-lr));
    if (!(r > 0.0 && r < 1.0)) return -std::numeric_limits<double>::infinity();
    const Eigen::VectorXd u = std::sqrt(r) * sphi + std::sqrt(1.0 - r) * stheta;
    return -0.5 * S.dot(u.cwiseProduct(u)) + R.dot(u) + std::log(r) + std::log1p(-r);
  };
  const double lr0 = std::log(rho) - std::log1p(-rho);
  rho = 1.0 / (1.0 + std::exp(-slice_sample(lr0, log_dens, 2.0)));
}

// sigma and rho given the centred components v = a phi and w = b theta
// (ancillarity-sufficiency interweaving, Yu and Meng 2011, J. Comput. Graph.
// Stat. 20:531). The likelihood depends on v + w only, so the conditional is
// the prior: v ~ N(0, a^2 (D - W)^+) on the sum-to-zero subspace (rank J - 1),
// w ~ N(0, b^2 I). phi and theta are then rescaled to the new a and b.
void PGBym2::interweave() {
  const double a = sigma * std::sqrt(rho) * scale_factor_;
  const double b = sigma * std::sqrt(1.0 - rho);
  const Eigen::VectorXd v = a * phi;
  const Eigen::VectorXd w = b * theta;
  const double qv = v.dot(Q_icar_ * v) / (scale_factor_ * scale_factor_);
  const double qw = w.squaredNorm();
  const double s2 = sigma_scale_ * sigma_scale_;

  auto log_dens_sigma = [&](double log_s) {
    const double s = std::exp(log_s);
    return -(2.0 * J_ - 1.0) * log_s - 0.5 * (qv / rho + qw / (1.0 - rho)) / (s * s)
           - std::log1p(s * s / s2) + log_s;
  };
  sigma = std::exp(slice_sample(std::log(sigma), log_dens_sigma, 1.0));

  const double sig2 = sigma * sigma;
  auto log_dens_rho = [&](double lr) {
    const double r = 1.0 / (1.0 + std::exp(-lr));
    if (!(r > 0.0 && r < 1.0)) return -std::numeric_limits<double>::infinity();
    return -0.5 * (J_ - 1.0) * std::log(r) - 0.5 * J_ * std::log1p(-r)
           - 0.5 * (qv / r + qw / (1.0 - r)) / sig2 + std::log(r) + std::log1p(-r);
  };
  const double lr0 = std::log(rho) - std::log1p(-rho);
  rho = 1.0 / (1.0 + std::exp(-slice_sample(lr0, log_dens_rho, 2.0)));

  phi = v / (sigma * std::sqrt(rho) * scale_factor_);
  theta = w / (sigma * std::sqrt(1.0 - rho));
}

void PGBym2::update(const Eigen::VectorXd& S, const Eigen::VectorXd& R) {
  update_phi(S, R);
  update_theta(S, R);
  update_sigma(S, R);
  update_rho(S, R);
  interweave();
}

} // namespace ratiod
