// pg_spatial.h
// Spatial random effects for PG Gibbs sampler
// Implements ICAR (Intrinsic CAR) and BYM2 fields

#ifndef RATIOD_PG_SPATIAL_H
#define RATIOD_PG_SPATIAL_H

#include <RcppEigen.h>
#include <vector>

namespace ratiod {

// ---------------------------------------------------------------------
// ICAR (Intrinsic Conditional Autoregressive) prior
// ---------------------------------------------------------------------

// Update spatial effects with ICAR prior
// phi_i | phi_{-i} ~ N(mean of neighbors, 1/(n_neighbors * tau))
//
// With PG augmentation:
// kappa_i/omega_i = X_beta_i + re_i + phi_i + error
//
// @param kappa y - n/2
// @param omega PG draws
// @param offset X*beta + RE contribution (everything except spatial)
// @param adj_list Adjacency list: adj_list[i] contains indices of neighbors of i
// @param n_neighbors Number of neighbors for each location
// @param tau Spatial precision parameter
// @return Updated spatial effects
Rcpp::NumericVector update_spatial_icar(
    const Rcpp::NumericVector& kappa,
    const Rcpp::NumericVector& omega,
    const Rcpp::NumericVector& offset,
    const Rcpp::IntegerVector& group,       // Group index for each observation
    const Rcpp::List& adj_list,             // List of neighbor indices
    const Rcpp::IntegerVector& n_neighbors, // Number of neighbors per location
    double tau
);

// Update spatial precision tau with gamma prior
// tau ~ Gamma(a, b)
// Posterior: tau | phi ~ Gamma(a + (J-1)/2, b + 0.5 * sum(phi * Q * phi))
// where Q is the precision matrix of ICAR
double update_tau_icar(
    const Rcpp::NumericVector& phi,
    const Rcpp::List& adj_list,
    const Rcpp::IntegerVector& n_neighbors,
    double prior_shape,
    double prior_rate
);

// ---------------------------------------------------------------------
// BYM2 field (Riebler et al. 2016)
// ---------------------------------------------------------------------

// u_s = sigma * (sqrt(rho) * scale_factor * phi_s + sqrt(1 - rho) * theta_s),
// phi an intrinsic CAR field with precision D - W on a connected graph and
// sum(phi) = 0, theta iid N(0, 1). The priors are those of the HMC backend:
// sigma ~ half-Cauchy(0, sigma_scale), rho ~ Uniform(0, 1).
//
// Given the Polya-Gamma draws, the binomial likelihood is Gaussian in the
// linear predictor, exp(kappa_i eta_i - omega_i eta_i^2 / 2). With
// eta_i = offset_i + u_{s(i)} it reaches the field only through the per-unit
// sums S_s = sum omega_i and R_s = sum (kappa_i - omega_i offset_i).
class PGBym2 {
public:
  PGBym2(const Rcpp::List& adj_list, double scale_factor, double sigma_scale);

  // One sweep: phi (block draw under the sum-to-zero constraint), theta,
  // sigma and rho, each from its full conditional, then sigma and rho again
  // given the centred field components.
  void update(const Eigen::VectorXd& S, const Eigen::VectorXd& R);

  // Combined effect u_s of every unit
  Eigen::VectorXd effect() const;

  Eigen::VectorXd phi;
  Eigen::VectorXd theta;
  double sigma;
  double rho;

private:
  int J_;
  double scale_factor_;
  double sigma_scale_;
  Eigen::SparseMatrix<double> Q_icar_;
  Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> llt_;

  void update_phi(const Eigen::VectorXd& S, const Eigen::VectorXd& R);
  void update_theta(const Eigen::VectorXd& S, const Eigen::VectorXd& R);
  void update_sigma(const Eigen::VectorXd& S, const Eigen::VectorXd& R);
  void update_rho(const Eigen::VectorXd& S, const Eigen::VectorXd& R);
  void interweave();
};

} // namespace ratiod

#endif // RATIOD_PG_SPATIAL_H
