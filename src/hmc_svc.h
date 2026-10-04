// hmc_svc.h
// Spatially-Varying Coefficients (SVC) with NNGP approximation
// Implements Nearest Neighbor Gaussian Process for scalable GP inference

#ifndef RATIOD_HMC_SVC_H
#define RATIOD_HMC_SVC_H

#define _USE_MATH_DEFINES  // For M_PI on Windows
#include <vector>
#include <cmath>
#include <algorithm>

#include <tulpa/sum_to_zero.h>  // s2z_centre_blocks / s2z_component_mean
#include "hmc_cov.h"

// Fallback definition of M_PI if not provided by <cmath>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace ratiod_svc {

// The covariance family lives in ratiod_cov, templated once for the double
// evaluation and the autodiff types alike. These names are the spelling every
// call site here and in hmc_gp.h already used.
using ratiod_cov::CovType;

// SVC data structure
struct SVCData {
  int n_obs;                          // Number of observations
  int n_svc;                          // Number of spatially-varying coefficients
  int nn;                             // Number of nearest neighbors

  std::vector<double> coords;         // Coordinates (n_obs x 2, flattened)
  std::vector<int> svc_indices;       // Which design matrix columns have SVCs
  std::vector<double> X_svc;          // Design matrix subset for SVC terms (n_obs x n_svc)

  // NNGP neighbor structure
  std::vector<int> nn_idx;            // Neighbor indices (n_obs x nn, flattened)
  std::vector<double> nn_dist;        // Distances to neighbors (n_obs x nn, flattened)
  std::vector<int> nn_order;          // Observation ordering for NNGP
  std::vector<int> nn_order_inv;      // Inverse ordering

  CovType cov_type;
  bool shared;                        // Whether SVC is shared between num/denom
};

// -----------------------------------------------------------------------------
// Covariance functions
// -----------------------------------------------------------------------------

using ratiod_cov::cov_exponential;
using ratiod_cov::cov_matern32;
using ratiod_cov::cov_gaussian;
using ratiod_cov::cov_spherical;
using ratiod_cov::compute_cov;

// -----------------------------------------------------------------------------
// NNGP likelihood computation
// -----------------------------------------------------------------------------

// Compute NNGP log-likelihood for a single SVC term
// w: vector of SVC values at each location (length n_obs)
// sigma2: spatial variance
// phi: spatial range parameter
// Returns log p(w | sigma2, phi) under NNGP approximation
inline double nngp_log_lik(
    const std::vector<double>& w,
    double sigma2,
    double phi,
    const SVCData& svc_data
) {
  int N = svc_data.n_obs;
  int nn = svc_data.nn;

  double log_lik = 0.0;

  // First observation: marginal N(0, sigma2)
  int first_idx = svc_data.nn_order[0];
  log_lik += -0.5 * std::log(2.0 * M_PI * sigma2) -
             0.5 * w[first_idx] * w[first_idx] / sigma2;

  // Remaining observations: conditional on neighbors
  for (int i = 1; i < N; i++) {
    int obs_idx = svc_data.nn_order[i];

    // Count actual neighbors (early observations have fewer)
    int n_neighbors = 0;
    for (int j = 0; j < nn; j++) {
      int nn_flat_idx = i * nn + j;
      if (svc_data.nn_idx[nn_flat_idx] > 0) {
        n_neighbors++;
      }
    }

    if (n_neighbors == 0) {
      // No neighbors: marginal
      log_lik += -0.5 * std::log(2.0 * M_PI * sigma2) -
                 0.5 * w[obs_idx] * w[obs_idx] / sigma2;
      continue;
    }

    // Build covariance vector c(s_i, s_{N(i)}) and matrix C(s_{N(i)}, s_{N(i)})
    std::vector<double> c_vec(n_neighbors);
    std::vector<double> C_mat(n_neighbors * n_neighbors);

    // c_vec: covariances between obs i and its neighbors
    for (int j = 0; j < n_neighbors; j++) {
      int nn_flat_idx = i * nn + j;
      double d = svc_data.nn_dist[nn_flat_idx];
      c_vec[j] = compute_cov(d, sigma2, phi, svc_data.cov_type);
    }

    // C_mat: covariances among neighbors
    for (int j1 = 0; j1 < n_neighbors; j1++) {
      int nn_idx1 = svc_data.nn_order[svc_data.nn_idx[i * nn + j1] - 1];
      for (int j2 = 0; j2 < n_neighbors; j2++) {
        int nn_idx2 = svc_data.nn_order[svc_data.nn_idx[i * nn + j2] - 1];

        if (j1 == j2) {
          C_mat[j1 * n_neighbors + j2] = sigma2;
        } else {
          // Compute distance between neighbors
          double d12 = std::sqrt(
            std::pow(svc_data.coords[nn_idx1 * 2] - svc_data.coords[nn_idx2 * 2], 2) +
            std::pow(svc_data.coords[nn_idx1 * 2 + 1] - svc_data.coords[nn_idx2 * 2 + 1], 2)
          );
          C_mat[j1 * n_neighbors + j2] = compute_cov(d12, sigma2, phi, svc_data.cov_type);
        }
      }
    }

    // Solve C_mat * alpha = c_vec through the shared neighbour-block Cholesky.
    std::vector<double> L, y, alpha;
    if (!ratiod_cov::nngp_chol(C_mat, n_neighbors, L)) return -INFINITY;
    ratiod_cov::nngp_forward_solve(L, n_neighbors, c_vec, y);
    ratiod_cov::nngp_back_solve(L, n_neighbors, y, alpha);

    // Conditional mean: mu_i = c^T * C^{-1} * w_{N(i)} = c^T * alpha
    double cond_mean = 0.0;
    for (int j = 0; j < n_neighbors; j++) {
      int nn_orig_idx = svc_data.nn_order[svc_data.nn_idx[i * nn + j] - 1];
      cond_mean += alpha[j] * w[nn_orig_idx];
    }

    // Conditional variance: var_i = sigma2 - c^T * C^{-1} * c
    double c_Cinv_c = 0.0;
    for (int j = 0; j < n_neighbors; j++) {
      c_Cinv_c += c_vec[j] * alpha[j];
    }
    double cond_var = ratiod_cov::nngp_floor_cond_var(sigma2 - c_Cinv_c);

    // Log-likelihood contribution
    double resid = w[obs_idx] - cond_mean;
    log_lik += -0.5 * std::log(2.0 * M_PI * cond_var) -
               0.5 * resid * resid / cond_var;
  }

  return log_lik;
}

// -----------------------------------------------------------------------------
// SVC contribution to linear predictor
// -----------------------------------------------------------------------------

// The linear predictor contribution and the level identification that goes
// with it are templated in hmc_svc_autodiff.h (compute_svc_eta /
// svc_center_eta): (sigma2, phi) are sampled, so every path through them
// carries an autodiff scalar and a plain-double copy would only be a second
// version of the same function.

// -----------------------------------------------------------------------------
// Identification of each term's global level
// -----------------------------------------------------------------------------

// beta_j and mean(w_j) are not separately identifiable: term j contributes
// eta_i += X_svc[i,j] * w_j(s_i), so w_j -> w_j + c together with
// beta_j -> beta_j - c leaves eta EXACTLY unchanged, for any covariate. The
// level is identified by CENTRING w_j on its way into eta (svc_center_eta in
// hmc_svc_autodiff.h), which removes that direction from the likelihood.
//
// An NNGP field is PROPER, so its constant direction already carries a prior
// -- precision 1' Sigma^-1 1 -- and needs no penalty supplying one. See
// tulpa/sum_to_zero.h for why a penalty on the sum is the wrong instrument on
// a proper field: it stiffens a direction the sampler still has to traverse,
// where centring removes it.
//
// The likelihood gradient follows the centring: with w_c = P w and
// P = I - 11'/n, dL/dw = P' dL/dw_c = dL/dw_c - mean(dL/dw_c), the SAME
// projection applied to the accumulated per-term block. The field prior's own
// gradient is NOT projected -- the NNGP density is placed on the uncentred w.
// So the likelihood contribution is accumulated apart from the prior one and
// projected on its own (svc_center_terms), which is what keeps the two from
// being projected together.
// Per-term means of a term-major field block: the quantity the centring
// removes. A gradient path keeps the UNCENTRED w for the field prior and
// subtracts these on its way into eta, so one buffer serves both readers.
inline void svc_term_means(
    const double* w_flat,
    const SVCData& svc_data,
    double* means_out
) {
  for (int j = 0; j < svc_data.n_svc; j++)
    means_out[j] = tulpa::s2z_component_mean(w_flat, j * svc_data.n_obs,
                                             svc_data.n_obs);
}

// P applied in place to a term-major block: each term less its mean over
// locations. On a field it is the field eta reads; on dL/dw_c it is dL/dw.
inline void svc_center_terms(double* block, const SVCData& svc_data) {
  const int n_obs = svc_data.n_obs;
  for (int j = 0; j < svc_data.n_svc; j++) {
    const int off = j * n_obs;
    const double m = tulpa::s2z_component_mean(block, off, n_obs);
    for (int i = 0; i < n_obs; i++) block[off + i] -= m;
  }
}

// -----------------------------------------------------------------------------
// Prior on GP hyperparameters
// -----------------------------------------------------------------------------

// Log prior for sigma2 (spatial variance): Half-Cauchy or exponential
inline double log_prior_sigma2(double sigma2, double scale) {
  // Half-Cauchy(0, scale): 2 / (pi * scale * (1 + (sigma2/scale)^2))
  // On log scale for sigma = sqrt(sigma2)
  double sigma = std::sqrt(sigma2);
  return std::log(2.0 / (M_PI * scale)) - std::log(1.0 + sigma * sigma / (scale * scale));
}

// Parse covariance type from string
inline CovType parse_cov_type(const std::string& cov_str) {
  if (cov_str == "exponential") return CovType::EXPONENTIAL;
  if (cov_str == "matern") return CovType::MATERN;
  if (cov_str == "gaussian") return CovType::GAUSSIAN;
  if (cov_str == "spherical") return CovType::SPHERICAL;
  return CovType::EXPONENTIAL;  // Default
}

// -----------------------------------------------------------------------------
// Gradient computation for SVC parameters (for hand-coded HMC gradients)
// -----------------------------------------------------------------------------

// Struct to hold SVC NNGP gradient results
struct SVCGradients {
  std::vector<double> grad_w;         // Gradient w.r.t. spatial effects (length n_obs)
  double grad_log_sigma2;             // Gradient w.r.t. log(sigma2)
  double grad_log_phi;                // Gradient w.r.t. log(phi)
};

// Scratch for a gradient pass over every SVC term. Held per-thread (via
// RATIOD_TLS_WORKSPACE at the call site), never as members of SVCData: SVCData
// hangs off the single ModelData that every chain thread shares, so buffers
// living there are the same memory for every chain and a concurrent write from
// one chain overwrites what another chain's gradient evaluation is reading.
struct SVCGradWorkspace {
  std::vector<double> sigma2;   // size n_svc
  std::vector<double> phi;      // size n_svc
  std::vector<double> w_flat;   // size n_obs * n_svc
  std::vector<double> w_j;      // size n_obs (reused per term)
  // dL/dw_c, accumulated apart from the field prior's gradient so
  // svc_center_project_lik_grad projects the likelihood part alone.
  std::vector<double> lik_grad; // size n_obs * n_svc
  std::vector<double> term_mean; // size n_svc
  SVCGradients grads;           // grad_w sized n_obs, reused per term

  void resize(int n_svc, int n_obs) {
    sigma2.resize(n_svc);
    phi.resize(n_svc);
    w_flat.resize(static_cast<size_t>(n_obs) * n_svc);
    w_j.resize(n_obs);
    lik_grad.assign(static_cast<size_t>(n_obs) * n_svc, 0.0);
    term_mean.assign(n_svc, 0.0);
    grads.grad_w.resize(n_obs);
  }
};

using ratiod_cov::dcov_dphi;

// Fully analytical NNGP gradients for SVC - single pass, no redundant function calls
// Complexity: O(N * nn²) - ~4x faster than numerical
inline void svc_nngp_gradients(
    const std::vector<double>& w,
    double sigma2,
    double phi,
    const SVCData& svc_data,
    SVCGradients& grads
) {
  int N = svc_data.n_obs;
  int nn = svc_data.nn;

  grads.grad_w.assign(N, 0.0);
  grads.grad_log_sigma2 = 0.0;
  grads.grad_log_phi = 0.0;

  // Size preconditions on the NNGP neighbour structure. This runs on a chain
  // worker thread inside the OpenMP region, so a violation cannot raise from
  // here; the gradients stay at the zero they were just assigned. Sizes come
  // from svc_nngp_precompute and do not vary across gradient evaluations, so
  // the check is a structural guard rather than a per-draw condition.
  if ((int)svc_data.nn_order.size() < N) return;
  if ((int)svc_data.nn_idx.size() < N * nn) return;
  if ((int)svc_data.nn_dist.size() < N * nn) return;
  if ((int)w.size() < N) return;
  if ((int)svc_data.coords.size() < 2 * N) return;

  // First observation: marginal N(0, sigma2)
  int first_idx = svc_data.nn_order[0];
  if (first_idx < 0 || first_idx >= N) return;
  double w0 = w[first_idx];
  grads.grad_w[first_idx] = -w0 / sigma2;
  grads.grad_log_sigma2 += 0.5 * (w0 * w0 / sigma2 - 1.0);  // Will multiply by sigma2 at end

  // Preallocate work arrays (reused across iterations)
  std::vector<double> c_vec(nn), dc_vec(nn), C_mat(nn * nn), L(nn * nn);
  std::vector<double> y_vec(nn), alpha(nn), y2(nn), beta(nn), w_nb(nn);
  std::vector<int> nb_idx(nn);

  for (int i = 1; i < N; i++) {
    int obs_idx = svc_data.nn_order[i];
    if (obs_idx < 0 || obs_idx >= N) continue;

    // Count neighbors
    int n_nb = 0;
    for (int j = 0; j < nn && svc_data.nn_idx[i * nn + j] > 0; j++) n_nb++;

    if (n_nb == 0) {
      double wi = w[obs_idx];
      grads.grad_w[obs_idx] += -wi / sigma2;
      grads.grad_log_sigma2 += 0.5 * (wi * wi / sigma2 - 1.0);
      continue;
    }

    // Build c_vec, dc_vec (covariances and derivatives)
    for (int j = 0; j < n_nb; j++) {
      double d = svc_data.nn_dist[i * nn + j];
      c_vec[j] = compute_cov(d, sigma2, phi, svc_data.cov_type);
      dc_vec[j] = dcov_dphi(d, sigma2, phi, c_vec[j], svc_data.cov_type);
    }

    // Build C_mat and get neighbor indices
    bool ok = true;
    for (int j1 = 0; j1 < n_nb && ok; j1++) {
      int raw1 = svc_data.nn_idx[i * nn + j1];
      if (raw1 - 1 < 0 || raw1 - 1 >= (int)svc_data.nn_order.size()) { ok = false; break; }
      int idx1 = svc_data.nn_order[raw1 - 1];
      if (idx1 < 0 || idx1 >= N) { ok = false; break; }
      nb_idx[j1] = idx1;

      for (int j2 = 0; j2 < n_nb; j2++) {
        if (j1 == j2) {
          C_mat[j1 * n_nb + j2] = sigma2;
        } else {
          int raw2 = svc_data.nn_idx[i * nn + j2];
          if (raw2 - 1 < 0 || raw2 - 1 >= (int)svc_data.nn_order.size()) { ok = false; break; }
          int idx2 = svc_data.nn_order[raw2 - 1];
          double dx = svc_data.coords[idx1 * 2] - svc_data.coords[idx2 * 2];
          double dy = svc_data.coords[idx1 * 2 + 1] - svc_data.coords[idx2 * 2 + 1];
          C_mat[j1 * n_nb + j2] = compute_cov(std::sqrt(dx*dx + dy*dy), sigma2, phi, svc_data.cov_type);
        }
      }
    }
    if (!ok) {
      double wi = w[obs_idx];
      grads.grad_w[obs_idx] += -wi / sigma2;
      grads.grad_log_sigma2 += 0.5 * (wi * wi / sigma2 - 1.0);
      continue;
    }

    // Cholesky: C = LL', through the same factorization the density uses.
    if (!ratiod_cov::nngp_chol(C_mat.data(), n_nb, L.data())) continue;

    // alpha = C^{-1} c
    ratiod_cov::nngp_forward_solve(L.data(), n_nb, c_vec.data(), y_vec.data());
    ratiod_cov::nngp_back_solve(L.data(), n_nb, y_vec.data(), alpha.data());

    // beta = C^{-1} w_nb
    for (int j = 0; j < n_nb; j++) w_nb[j] = w[nb_idx[j]];
    ratiod_cov::nngp_forward_solve(L.data(), n_nb, w_nb.data(), y2.data());
    ratiod_cov::nngp_back_solve(L.data(), n_nb, y2.data(), beta.data());

    // Conditional mean and variance
    double mu = 0.0, c_alpha = 0.0;
    for (int j = 0; j < n_nb; j++) { mu += alpha[j] * w_nb[j]; c_alpha += c_vec[j] * alpha[j]; }
    double v = ratiod_cov::nngp_floor_cond_var(sigma2 - c_alpha);
    double r = w[obs_idx] - mu;

    // Gradient w.r.t. w
    grads.grad_w[obs_idx] += -r / v;
    for (int j = 0; j < n_nb; j++) grads.grad_w[nb_idx[j]] += alpha[j] * r / v;

    // Gradient w.r.t. sigma2: dv/ds2 = 1 - c'α/s2
    double dll_dv = 0.5 * (r * r / v - 1.0) / v;
    grads.grad_log_sigma2 += dll_dv * (1.0 - c_alpha / sigma2) * sigma2;

    // Gradient w.r.t. phi: compute quadratic forms on-the-fly
    double alpha_dc = 0.0, dc_beta = 0.0;
    for (int j = 0; j < n_nb; j++) { alpha_dc += alpha[j] * dc_vec[j]; dc_beta += dc_vec[j] * beta[j]; }

    // alpha' * dC/dphi * alpha and alpha' * dC/dphi * beta (computed on-the-fly)
    double alpha_dC_alpha = 0.0, alpha_dC_beta = 0.0;
    for (int j1 = 0; j1 < n_nb; j1++) {
      for (int j2 = 0; j2 < n_nb; j2++) {
        double dC_jk = 0.0;
        if (j1 != j2) {
          double dx = svc_data.coords[nb_idx[j1] * 2] - svc_data.coords[nb_idx[j2] * 2];
          double dy = svc_data.coords[nb_idx[j1] * 2 + 1] - svc_data.coords[nb_idx[j2] * 2 + 1];
          double d12 = std::sqrt(dx*dx + dy*dy);
          dC_jk = dcov_dphi(d12, sigma2, phi, C_mat[j1 * n_nb + j2],
                            svc_data.cov_type);
        }
        alpha_dC_alpha += alpha[j1] * dC_jk * alpha[j2];
        alpha_dC_beta += alpha[j1] * dC_jk * beta[j2];
      }
    }

    double dv_dphi = -2.0 * alpha_dc + alpha_dC_alpha;
    double dr_dphi = -dc_beta + alpha_dC_beta;
    grads.grad_log_phi += (dll_dv * dv_dphi + (-r / v) * dr_dphi) * phi;
  }
}

// Accumulate the NNGP field prior's gradient for every SVC term: d/dw into
// `grad` at `w_start` in the j-major layout the field is stored in, and the two
// hyperparameter derivatives at their own slots. Every gradient path reads the
// field prior the same way, so it is written once and the per-term extraction
// buffer comes from the caller's per-thread workspace.
inline void svc_nngp_prior_grads(
    const double* w_flat,
    const double* sigma2,
    const double* phi,
    const SVCData& svc_data,
    int w_start,
    int log_sigma2_start,
    int log_phi_start,
    SVCGradWorkspace& ws,
    double* grad
) {
  const int n_obs = svc_data.n_obs;
  const int n_svc = svc_data.n_svc;

  for (int j = 0; j < n_svc; j++) {
    const double* w_j_src = w_flat + static_cast<size_t>(j) * n_obs;
    std::copy(w_j_src, w_j_src + n_obs, ws.w_j.begin());

    svc_nngp_gradients(ws.w_j, sigma2[j], phi[j], svc_data, ws.grads);

    for (int i = 0; i < n_obs; i++) {
      grad[w_start + j * n_obs + i] += ws.grads.grad_w[i];
    }
    grad[log_sigma2_start + j] += ws.grads.grad_log_sigma2;
    grad[log_phi_start + j] += ws.grads.grad_log_phi;
  }
}

} // namespace ratiod_svc

#endif // RATIOD_HMC_SVC_H
