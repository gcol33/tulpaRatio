#include <RcppEigen.h>
#include "hmc_integrator.h"

using namespace Rcpp;

namespace ratiod_hmc {

void DenseMassMatrix::advance_position(double* q, const double* p, double epsilon,
                                       double* velocity) const {
  if (type == MassMatrixType::BLOCK_DIAG && adapted && !has_precision_overrides()) {
    inv_mass_times_p(p,velocity);
    ratiod_linalg::axpy_weighted(epsilon,inv_mass_diag.data(),p,q,n);
    for (const auto& block : blocks) if (block.adapted)
      for (int i = 0; i < block.size; ++i) {
        const int j = block.start+i;
        q[j] += epsilon*(velocity[j]-inv_mass_diag[j]*p[j]);
      }
  } else advance_metric_position(*this,q,p,n,epsilon,velocity);
}

// =====================================================================
// Dense mass matrix: Cholesky decomposition via Eigen
// =====================================================================

bool DenseMassMatrix::update_from_covariance(const double* cov, int n_samples) {
  // Map the covariance data into an Eigen matrix (column-major)
  Eigen::Map<const Eigen::MatrixXd> C(cov, n, n);

  // Eigendecomposition for condition number control.
  // Without conditioning, ill-conditioned mass matrices force epsilon to be
  // tiny (driven by the stiffest direction), making sampling extremely slow.
  // E.g., HSGP+RW1 gets epsilon=3.2e-5 unconditioned vs ~0.01 conditioned.
  //
  // Clip eigenvalue ratio to MAX_COND so the step size ratio between the
  // loosest and stiffest directions is at most sqrt(MAX_COND) ≈ 100:1.
  constexpr double MAX_COND = 1e4;

  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(C);
  if (eig.info() != Eigen::Success) {
    // Eigendecomposition failed — degrade to diagonal
    type = MassMatrixType::DIAG;
    adapted = true;
    for (int i = 0; i < n; i++) {
      double var_i = cov[static_cast<size_t>(i) * n + i];
      inv_mass_diag[i] = std::max(1e-3, std::min(var_i, 1e3));
      sqrt_mass_diag[i] = 1.0 / std::sqrt(inv_mass_diag[i]);
    }
    return false;
  }

  Eigen::VectorXd evals = eig.eigenvalues();
  double lambda_max = evals.maxCoeff();
  double lambda_floor = std::max(lambda_max / MAX_COND, 1e-8);
  bool clipped = false;
  for (int i = 0; i < n; i++) {
    if (evals[i] < lambda_floor) {
      evals[i] = lambda_floor;
      clipped = true;
    }
  }

  // Reconstruct conditioned covariance: V * diag(λ_clipped) * V^T
  Eigen::MatrixXd C_cond;
  if (clipped) {
    const Eigen::MatrixXd& V = eig.eigenvectors();
    C_cond = V * evals.asDiagonal() * V.transpose();
  } else {
    C_cond = C;
  }

  // Cholesky of (conditioned) covariance — guaranteed to succeed after clipping
  Eigen::LLT<Eigen::MatrixXd> llt(C_cond);
  if (llt.info() != Eigen::Success) {
    // Should not happen after eigenvalue clipping, but handle gracefully
    type = MassMatrixType::DIAG;
    adapted = true;
    for (int i = 0; i < n; i++) {
      double var_i = cov[static_cast<size_t>(i) * n + i];
      inv_mass_diag[i] = std::max(1e-3, std::min(var_i, 1e3));
      sqrt_mass_diag[i] = 1.0 / std::sqrt(inv_mass_diag[i]);
    }
    return false;
  }

  // Store conditioned covariance as inv_mass_dense
  std::memcpy(inv_mass_dense.data(), C_cond.data(),
              static_cast<size_t>(n) * n * sizeof(double));

  // Store Cholesky factor L
  Eigen::MatrixXd L_mat = llt.matrixL();
  std::memcpy(L_inv_mass.data(), L_mat.data(),
              static_cast<size_t>(n) * n * sizeof(double));

  // Also update diagonal for fallback and find_reasonable_epsilon compatibility
  for (int i = 0; i < n; i++) {
    double var_i = C_cond(i, i);
    inv_mass_diag[i] = std::max(1e-3, std::min(var_i, 1e3));
    sqrt_mass_diag[i] = 1.0 / std::sqrt(inv_mass_diag[i]);
  }

  adapted = true;
  return true;
}

// =====================================================================
// Parameter layout computation
// =====================================================================

DualAveraging::DualAveraging(double epsilon_init, int n_params, double target_boost)
  : mu(std::log(10.0 * epsilon_init)), log_epsilon_bar(std::log(epsilon_init)), H_bar(0.0),
    gamma(0.05), t0(10.0), kappa(0.75),
    target_accept(compute_target(n_params, target_boost)), m(0) {}

double DualAveraging::update(double alpha) {
  m++;
  double w = 1.0 / (m + t0);
  H_bar = (1.0 - w) * H_bar + w * (target_accept - alpha);
  double log_epsilon = mu - std::sqrt((double)m) / gamma * H_bar;
  // Clamp log_epsilon to reasonable range
  // Lower bound: exp(-14) ≈ 8e-7, Upper bound: exp(2) ≈ 7.4
  log_epsilon = std::max(-14.0, std::min(log_epsilon, 2.0));
  double epsilon = std::exp(log_epsilon);
  double m_w = std::pow((double)m, -kappa);
  log_epsilon_bar = m_w * log_epsilon + (1.0 - m_w) * log_epsilon_bar;
  return epsilon;
}

double DualAveraging::final_epsilon() const {
  return std::exp(log_epsilon_bar);
}


}  // namespace ratiod_hmc
