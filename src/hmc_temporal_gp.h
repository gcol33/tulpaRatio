// hmc_temporal_gp.h
// Gaussian-process temporal effect over the distinct time instants.
//
// Each group's effect is f_g ~ N(0, sigma2 R(phi)) on the time grid, with R the
// correlation of the chosen kernel. The exponential kernel is Markov, so its
// density factorizes into the AR(1) recursion rho_t = exp(-dt_t / phi) and
// costs O(T). Every other kernel goes through a dense Cholesky factor of
// R(phi) + TGP_NUGGET I, shared by the groups, at O(T^3).
//
// In the non-centred coordinate the sampled block is z_g ~ N(0, I) and
// f_g = sigma L(phi) z_g with L L' the correlation; the AR(1) recursion is that
// same product for the exponential kernel. In the centred coordinate the block
// is f_g itself.
//
// temporal_gp_log_prior() is the density for every scalar type the templated
// log posterior is evaluated in; TemporalGPBlock evaluates it at a double state
// and differentiates it analytically for the hand-coded gradients.

#ifndef RATIOD_HMC_TEMPORAL_GP_H
#define RATIOD_HMC_TEMPORAL_GP_H

#include <vector>
#include <cmath>
#include <limits>
#include <Eigen/Dense>
#include "ar1_shared.h"
#include "autodiff_utils.h"
#include "hmc_cov.h"

namespace ratiod_temporal_gp {

using namespace ratiod::math;

enum class TemporalCovType { EXPONENTIAL, MATERN32, MATERN52, GAUSSIAN, PERIODIC };

struct TemporalGPData {
  int n_times = 0;                    // Distinct time instants, the grid size
  int n_groups = 1;                   // One effect per group on the shared grid
  std::vector<double> time_values;    // Sorted, length n_times

  TemporalCovType cov_type = TemporalCovType::EXPONENTIAL;
  double period = 1.0;                // Periodic kernel only
  bool shared = true;                 // Effect enters both processes

  // PC prior on sigma, P(sigma > U) = alpha, and a uniform prior on phi.
  double sigma2_prior_U = 1.0;
  double sigma2_prior_alpha = 0.01;
  double phi_lower = 0.01;
  double phi_upper = 10.0;

  bool noncentered = true;
};

// The nugget, as a fraction of the marginal variance, that keeps the dense
// correlation positive definite: a periodic kernel repeats exactly after one
// period, and a smooth kernel on a fine grid is numerically singular.
constexpr double TGP_NUGGET = 1e-6;

// -----------------------------------------------------------------------------
// Kernels. phi is the lengthscale; the four shared with the spatial GP come from
// ratiod_cov, so the two use the same Gaussian and Matern forms.
// -----------------------------------------------------------------------------

template <typename T>
inline T temporal_corr(double d, const T& phi, const TemporalGPData& gp) {
  const T one(1.0);
  switch (gp.cov_type) {
    case TemporalCovType::EXPONENTIAL:
      return ratiod_cov::cov_exponential(d, one, phi);
    case TemporalCovType::MATERN32:
      return ratiod_cov::cov_matern32(d, one, phi);
    case TemporalCovType::MATERN52:
      return ratiod_cov::cov_matern52(d, one, phi);
    case TemporalCovType::GAUSSIAN:
      return ratiod_cov::cov_gaussian(d, one, phi);
    case TemporalCovType::PERIODIC: {
      // exp(-2 sin^2(pi d / period) / phi^2)
      const double s = std::sin(M_PI * d / gp.period);
      return safe_exp(T(-2.0 * s * s) / (phi * phi));
    }
  }
  return ratiod_cov::cov_exponential(d, one, phi);
}

// d corr / d phi at corr value k, written against temporal_corr() above.
inline double temporal_dcorr_dphi(double d, double phi, double k,
                                  const TemporalGPData& gp) {
  switch (gp.cov_type) {
    case TemporalCovType::EXPONENTIAL:
      return ratiod_cov::dcov_dphi(d, 1.0, phi, k, ratiod_cov::CovType::EXPONENTIAL);
    case TemporalCovType::MATERN32:
      return ratiod_cov::dcov_dphi(d, 1.0, phi, k, ratiod_cov::CovType::MATERN);
    case TemporalCovType::GAUSSIAN:
      return ratiod_cov::dcov_dphi(d, 1.0, phi, k, ratiod_cov::CovType::GAUSSIAN);
    case TemporalCovType::MATERN52: {
      // k = (1 + u + u^2/3) exp(-u), u = sqrt(5) d / phi;
      // dk/dphi = u^2 (1 + u) exp(-u) / (3 phi)
      const double u = std::sqrt(5.0) * d / phi;
      return u * u * (1.0 + u) * std::exp(-u) / (3.0 * phi);
    }
    case TemporalCovType::PERIODIC: {
      // k = exp(-2 s^2 / phi^2), s = sin(pi d / period); dk/dphi = 4 s^2 k / phi^3
      const double s = std::sin(M_PI * d / gp.period);
      return 4.0 * s * s * k / (phi * phi * phi);
    }
  }
  return 0.0;
}

// -----------------------------------------------------------------------------
// Hyperparameters, sampled as log sigma2 and logit-bounded phi
// -----------------------------------------------------------------------------

template <typename T>
inline T temporal_gp_phi(const T& logit_phi, const TemporalGPData& gp) {
  return T(gp.phi_lower) + T(gp.phi_upper - gp.phi_lower) * inv_logit(logit_phi);
}

// PC prior on sigma and uniform prior on phi, on the sampled coordinates with
// the Jacobians of both transforms.
template <typename T>
inline T temporal_gp_hyper_log_prior(const T& log_sigma2, const T& phi,
                                     const TemporalGPData& gp) {
  const double rate = -std::log(gp.sigma2_prior_alpha) / gp.sigma2_prior_U;
  const double range = gp.phi_upper - gp.phi_lower;
  const T sigma = safe_sqrt(safe_exp(log_sigma2));
  return T(std::log(rate)) - T(rate) * sigma - safe_log(T(2.0) * sigma) + log_sigma2
       + safe_log(phi - T(gp.phi_lower)) + safe_log(T(gp.phi_upper) - phi)
       - T(std::log(range));
}

// L L' = R(phi) + TGP_NUGGET I over the time grid, row-major, lower triangular.
template <typename T>
inline bool temporal_gp_dense_factor(const T& phi, const TemporalGPData& gp,
                                     std::vector<T>& L) {
  const int n = gp.n_times;
  std::vector<T> R(static_cast<size_t>(n) * n);
  for (int i = 0; i < n; i++) {
    for (int j = 0; j <= i; j++) {
      const T k = temporal_corr(std::abs(gp.time_values[i] - gp.time_values[j]), phi, gp);
      R[i * n + j] = k;
      R[j * n + i] = k;
    }
  }
  return ratiod_cov::nngp_chol(R, n, L, TGP_NUGGET);
}

// -----------------------------------------------------------------------------
// The density
// -----------------------------------------------------------------------------

// Log density of the block x (z non-centred, f centred) and of its
// hyperparameters, writing the effects f that enter eta. -inf when the dense
// correlation does not factor. `chol`, when given, receives the dense factor.
template <typename T>
inline T temporal_gp_log_prior(const T* x, const T& log_sigma2, const T& logit_phi,
                               const TemporalGPData& gp, std::vector<T>& f,
                               std::vector<T>* chol = nullptr) {
  const int n = gp.n_times;
  const int G = gp.n_groups;
  const T sigma2 = safe_exp(log_sigma2);
  const T sigma = safe_sqrt(sigma2);
  const T phi = temporal_gp_phi(logit_phi, gp);
  T lp = temporal_gp_hyper_log_prior(log_sigma2, phi, gp);
  f.assign(static_cast<size_t>(G) * n, T(0.0));

  if (gp.noncentered) {
    for (int k = 0; k < G * n; k++) lp = lp - T(0.5) * x[k] * x[k];
  }

  if (gp.cov_type == TemporalCovType::EXPONENTIAL) {
    std::vector<T> rho(n > 1 ? n - 1 : 0), omr2(n > 1 ? n - 1 : 0);
    for (int t = 1; t < n; t++) {
      const double dt = gp.time_values[t] - gp.time_values[t - 1];
      rho[t - 1] = safe_exp(T(-dt) / phi);
      omr2[t - 1] = ratiod_ar1::one_minus_rho2(rho[t - 1]);
    }
    for (int g = 0; g < G; g++) {
      const int o = g * n;
      if (gp.noncentered) {
        f[o] = sigma * x[o];
        for (int t = 1; t < n; t++) {
          f[o + t] = rho[t - 1] * f[o + t - 1] + sigma * safe_sqrt(omr2[t - 1]) * x[o + t];
        }
      } else {
        lp = lp - T(0.5) * safe_log(T(2.0 * M_PI) * sigma2)
                - T(0.5) * x[o] * x[o] / sigma2;
        for (int t = 1; t < n; t++) {
          const T cv = sigma2 * omr2[t - 1];
          const T r = x[o + t] - rho[t - 1] * x[o + t - 1];
          lp = lp - T(0.5) * safe_log(T(2.0 * M_PI) * cv) - T(0.5) * r * r / cv;
        }
        for (int t = 0; t < n; t++) f[o + t] = x[o + t];
      }
    }
    return lp;
  }

  std::vector<T> L_local;
  std::vector<T>& L = chol ? *chol : L_local;
  if (!temporal_gp_dense_factor(phi, gp, L)) {
    return T(-std::numeric_limits<double>::infinity());
  }

  if (gp.noncentered) {
    for (int g = 0; g < G; g++) {
      const int o = g * n;
      for (int i = 0; i < n; i++) {
        T s = T(0.0);
        for (int j = 0; j <= i; j++) s = s + L[i * n + j] * x[o + j];
        f[o + i] = sigma * s;
      }
    }
    return lp;
  }

  T log_det_L = T(0.0);
  for (int i = 0; i < n; i++) log_det_L = log_det_L + safe_log(L[i * n + i]);
  std::vector<T> y(n);
  for (int g = 0; g < G; g++) {
    const int o = g * n;
    ratiod_cov::nngp_forward_solve(L.data(), n, x + o, y.data());
    T q = T(0.0);
    for (int i = 0; i < n; i++) q = q + y[i] * y[i];
    lp = lp - T(0.5 * n * std::log(2.0 * M_PI)) - T(0.5 * n) * log_sigma2
            - log_det_L - T(0.5) * q / sigma2;
    for (int i = 0; i < n; i++) f[o + i] = x[o + i];
  }
  return lp;
}

// -----------------------------------------------------------------------------
// The block at a double state, for the hand-coded gradients
// -----------------------------------------------------------------------------

class TemporalGPBlock {
 public:
  // Evaluates the block at x (n_groups * n_times entries) and the two sampled
  // hyperparameters. Returns false when the dense correlation does not factor;
  // log_prior() is then -inf.
  bool forward(const double* x, double log_sigma2, double logit_phi,
               const TemporalGPData& gp) {
    gp_ = &gp;
    x_ = x;
    log_sigma2_ = log_sigma2;
    sigma2_ = std::exp(log_sigma2);
    sigma_ = std::sqrt(sigma2_);
    phi_ = temporal_gp_phi(logit_phi, gp);
    dphi_dlogit_ = (phi_ - gp.phi_lower) * (gp.phi_upper - phi_) /
                   (gp.phi_upper - gp.phi_lower);
    lp_ = temporal_gp_log_prior(x, log_sigma2, logit_phi, gp, f_, &L_);
    return std::isfinite(lp_);
  }

  const double* f() const { return f_.data(); }
  double log_prior() const { return lp_; }

  // The block's whole gradient given the likelihood's dL/df: grad_x is
  // assigned for every entry of the block, the two hyperparameter gradients
  // are assigned, prior and likelihood included.
  void backward(const double* dL_df, double* grad_x, double& grad_log_sigma2,
                double& grad_logit_phi) {
    const TemporalGPData& gp = *gp_;
    const int n = gp.n_times;
    const int G = gp.n_groups;
    const double rate = -std::log(gp.sigma2_prior_alpha) / gp.sigma2_prior_U;
    double g_s = 0.5 - 0.5 * rate * sigma_;
    double g_logit = (gp.phi_upper + gp.phi_lower - 2.0 * phi_) /
                     (gp.phi_upper - gp.phi_lower);
    if (!std::isfinite(lp_)) {
      for (int k = 0; k < G * n; k++) grad_x[k] = 0.0;
      grad_log_sigma2 = g_s;
      grad_logit_phi = g_logit;
      return;
    }
    double g_phi = (gp.cov_type == TemporalCovType::EXPONENTIAL)
        ? backward_markov(dL_df, grad_x, g_s)
        : backward_dense(dL_df, grad_x, g_s);
    grad_log_sigma2 = g_s;
    grad_logit_phi = g_logit + g_phi * dphi_dlogit_;
  }

 private:
  // Returns d/dphi; adds d/d(log sigma2) into g_s.
  double backward_markov(const double* dL_df, double* grad_x, double& g_s) {
    const TemporalGPData& gp = *gp_;
    const int n = gp.n_times;
    const int G = gp.n_groups;
    rho_.resize(n > 1 ? n - 1 : 0);
    drho_.resize(rho_.size());
    dlog_omr2_.resize(rho_.size());
    a_.resize(n);
    a_[0] = sigma_;
    for (int t = 1; t < n; t++) {
      const double dt = gp.time_values[t] - gp.time_values[t - 1];
      const double rho = std::exp(-dt / phi_);
      rho_[t - 1] = rho;
      drho_[t - 1] = rho * dt / (phi_ * phi_);
      dlog_omr2_[t - 1] = ratiod_ar1::dlog_one_minus_rho2_drho(rho);
      a_[t] = sigma_ * std::sqrt(ratiod_ar1::one_minus_rho2(rho));
    }

    double g_phi = 0.0;
    if (gp.noncentered) {
      adj_.resize(n);
      for (int g = 0; g < G; g++) {
        const int o = g * n;
        adj_[n - 1] = dL_df[o + n - 1];
        for (int t = n - 2; t >= 0; t--) adj_[t] = dL_df[o + t] + rho_[t] * adj_[t + 1];
        for (int t = 0; t < n; t++) {
          grad_x[o + t] = -x_[o + t] + a_[t] * adj_[t];
          g_s += 0.5 * adj_[t] * a_[t] * x_[o + t];
        }
        for (int t = 1; t < n; t++) {
          // f_t = rho f_{t-1} + a_t z_t, both rho and a_t moving with phi
          const double da_drho = 0.5 * a_[t] * dlog_omr2_[t - 1];
          g_phi += adj_[t] * drho_[t - 1] *
                   (f_[o + t - 1] + da_drho * x_[o + t]);
        }
      }
      return g_phi;
    }

    for (int g = 0; g < G; g++) {
      const int o = g * n;
      for (int t = 0; t < n; t++) grad_x[o + t] = dL_df[o + t];
      grad_x[o] += -x_[o] / sigma2_;
      g_s += -0.5 + 0.5 * x_[o] * x_[o] / sigma2_;
      for (int t = 1; t < n; t++) {
        const double cv = a_[t] * a_[t];
        const double r = x_[o + t] - rho_[t - 1] * x_[o + t - 1];
        grad_x[o + t] += -r / cv;
        grad_x[o + t - 1] += rho_[t - 1] * r / cv;
        g_s += -0.5 + 0.5 * r * r / cv;
        // d/drho of -0.5 log cv - 0.5 r^2 / cv, with cv = sigma2 (1 - rho^2)
        const double dlog = dlog_omr2_[t - 1];
        const double d_rho = -0.5 * dlog * (1.0 - r * r / cv) + r * x_[o + t - 1] / cv;
        g_phi += d_rho * drho_[t - 1];
      }
    }
    return g_phi;
  }

  // Returns d/dphi; adds d/d(log sigma2) into g_s.
  double backward_dense(const double* dL_df, double* grad_x, double& g_s) {
    using RowMat = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    const TemporalGPData& gp = *gp_;
    const int n = gp.n_times;
    const int G = gp.n_groups;
    Eigen::Map<const RowMat> L(L_.data(), n, n);
    const auto Lt = L.triangularView<Eigen::Lower>();

    Eigen::MatrixXd dR(n, n);
    for (int i = 0; i < n; i++) {
      for (int j = 0; j <= i; j++) {
        const double d = std::abs(gp.time_values[i] - gp.time_values[j]);
        const double k = temporal_corr(d, phi_, gp);
        dR(i, j) = dR(j, i) = temporal_dcorr_dphi(d, phi_, k, gp);
      }
    }
    // P = L^{-1} dR L^{-T}, symmetric
    Eigen::MatrixXd P = Lt.solve(dR);
    P = Lt.solve(P.transpose()).eval();

    double g_phi = 0.0;
    if (gp.noncentered) {
      // dL/dphi = L Phi(P), Phi the lower triangle with the diagonal halved
      Eigen::MatrixXd Phi = P.triangularView<Eigen::StrictlyLower>();
      Phi.diagonal() = 0.5 * P.diagonal();
      for (int g = 0; g < G; g++) {
        const int o = g * n;
        Eigen::Map<const Eigen::VectorXd> z(x_ + o, n);
        Eigen::Map<const Eigen::VectorXd> dl(dL_df + o, n);
        Eigen::Map<const Eigen::VectorXd> fg(f_.data() + o, n);
        Eigen::Map<Eigen::VectorXd> gz(grad_x + o, n);
        const Eigen::VectorXd Ltdl = Lt.transpose() * dl;
        gz = -z + sigma_ * Ltdl;
        g_s += 0.5 * dl.dot(fg);
        const Eigen::VectorXd w = L * (Phi * z);
        g_phi += sigma_ * dl.dot(w);
      }
      return g_phi;
    }

    const double tr_P = P.trace();
    for (int g = 0; g < G; g++) {
      const int o = g * n;
      Eigen::Map<const Eigen::VectorXd> xg(x_ + o, n);
      Eigen::Map<const Eigen::VectorXd> dl(dL_df + o, n);
      Eigen::Map<Eigen::VectorXd> gx(grad_x + o, n);
      const Eigen::VectorXd alpha = Lt.transpose().solve(Lt.solve(xg));
      gx = dl - alpha / sigma2_;
      g_s += -0.5 * n + 0.5 * xg.dot(alpha) / sigma2_;
      g_phi += 0.5 * alpha.dot(dR * alpha) / sigma2_ - 0.5 * tr_P;
    }
    return g_phi;
  }

  const TemporalGPData* gp_ = nullptr;
  const double* x_ = nullptr;
  double log_sigma2_ = 0.0, sigma2_ = 1.0, sigma_ = 1.0;
  double phi_ = 1.0, dphi_dlogit_ = 0.0;
  double lp_ = 0.0;
  std::vector<double> f_, L_;
  std::vector<double> rho_, drho_, dlog_omr2_, a_, adj_;
};

} // namespace ratiod_temporal_gp

#endif // RATIOD_HMC_TEMPORAL_GP_H
