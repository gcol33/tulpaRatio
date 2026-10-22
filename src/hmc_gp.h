// hmc_gp.h
// Gaussian Process spatial effects with NNGP approximation
// Supports single-scale GP and multi-scale (local + regional) GP

#ifndef RATIOD_HMC_GP_H
#define RATIOD_HMC_GP_H

#include <vector>
#include <cmath>
#include <random>
#include <type_traits>
#include <tuple>
#include "tls_workspace.h"
#include <RcppEigen.h>
#include "hmc_cov.h"  // Shared kernels and neighbour-block factorization
#include "hmc_svc.h"  // Reuse covariance functions and NNGP infrastructure

// Verbose debug output (set to false for production)
#define GP_DEBUG_BOUNDS false

namespace ratiod_gp {

using ratiod_svc::CovType;
using ratiod_svc::compute_cov;

// How the multi-scale GP blocks' mass matrix is adapted during warmup. This
// selects an adaptation strategy, not a parameterization of the field; the
// coordinate the field is sampled in is spatial_multiscale(parameterization =).
enum class MSGPSampler {
  AUTO,   // The sampler's own warmup adaptation
  LBFGS   // L-BFGS quasi-Newton mass matrix adaptation - O(md) per step
};

// The spatial_multiscale(sampler =) names
inline constexpr ratiod_enum::EnumName<MSGPSampler> MSGP_SAMPLER_NAMES[] = {
  {"auto", MSGPSampler::AUTO}, {"lbfgs", MSGPSampler::LBFGS}};

inline MSGPSampler parse_msgp_sampler(const std::string& s) {
  return ratiod_enum::parse_enum(MSGP_SAMPLER_NAMES, s, "multi-scale GP sampler");
}

// =============================================================================
// L-BFGS MASS MATRIX ADAPTATION
// =============================================================================
//
// L-BFGS approximates the inverse Hessian using limited memory:
//   H_k ≈ (I - ρ_k s_k y_k^T) H_{k-1} (I - ρ_k y_k s_k^T) + ρ_k s_k s_k^T
//
// where:
//   s_k = q_k - q_{k-1}     (position difference)
//   y_k = g_k - g_{k-1}     (gradient difference)
//   ρ_k = 1 / (y_k^T s_k)
//
// Storage: O(md) where m = memory size (typically 5-20), d = dimension
// Compute H*v: O(md) via two-loop recursion
// =============================================================================

struct LBFGSState {
    int m;                                    // Memory size (number of pairs to store)
    int d;                                    // Dimension
    int k;                                    // Current iteration count
    std::vector<std::vector<double>> s_list; // Position differences (circular buffer)
    std::vector<std::vector<double>> y_list; // Gradient differences (circular buffer)
    std::vector<double> rho_list;            // 1 / (y^T s) values
    double gamma;                            // Scaling factor for initial H_0

    LBFGSState() : m(0), d(0), k(0), gamma(1.0) {}

    LBFGSState(int memory_size, int dimension)
        : m(memory_size), d(dimension), k(0), gamma(1.0) {
        s_list.reserve(m);
        y_list.reserve(m);
        rho_list.reserve(m);
    }

    // Add a new (s, y) pair from position and gradient differences
    void add_pair(const std::vector<double>& s, const std::vector<double>& y) {
        double ys = 0.0;
        double yy = 0.0;
        for (int i = 0; i < d; i++) {
            ys += y[i] * s[i];
            yy += y[i] * y[i];
        }

        // Skip if curvature condition not satisfied (ensures positive definiteness)
        if (ys < 1e-10) return;

        double rho = 1.0 / ys;

        // Update scaling factor: gamma = (s^T y) / (y^T y)
        if (yy > 1e-10) {
            gamma = ys / yy;
        }

        // Add to circular buffer
        if ((int)s_list.size() < m) {
            s_list.push_back(s);
            y_list.push_back(y);
            rho_list.push_back(rho);
        } else {
            // Circular replacement
            int idx = k % m;
            s_list[idx] = s;
            y_list[idx] = y;
            rho_list[idx] = rho;
        }
        k++;
    }

    // Two-loop recursion: compute H_k * v in O(md) time
    void multiply_H(const std::vector<double>& v, std::vector<double>& result) const {
        if (d <= 0 || (int)v.size() != d) {
            result = v;
            return;
        }

        result.resize(d);
        for (int i = 0; i < d; i++) {
            result[i] = v[i];
        }

        int n_stored = std::min(k, (int)s_list.size());
        n_stored = std::min(n_stored, m);

        if (n_stored == 0) {
            for (int i = 0; i < d; i++) {
                result[i] *= gamma;
            }
            return;
        }

        std::vector<double> alpha(n_stored);

        // First loop: from newest to oldest
        for (int i = n_stored - 1; i >= 0; i--) {
            int idx = (k - n_stored + i) % m;
            if (idx < 0) idx += m;
            if (idx >= (int)s_list.size()) continue;

            double dot = 0.0;
            for (int j = 0; j < d && j < (int)s_list[idx].size(); j++) {
                dot += s_list[idx][j] * result[j];
            }
            alpha[i] = rho_list[idx] * dot;
            for (int j = 0; j < d && j < (int)y_list[idx].size(); j++) {
                result[j] -= alpha[i] * y_list[idx][j];
            }
        }

        // Apply initial Hessian: r = gamma * q
        for (int i = 0; i < d; i++) {
            result[i] *= gamma;
        }

        // Second loop: from oldest to newest
        for (int i = 0; i < n_stored; i++) {
            int idx = (k - n_stored + i) % m;
            if (idx < 0) idx += m;
            if (idx >= (int)s_list.size()) continue;

            double dot = 0.0;
            for (int j = 0; j < d && j < (int)y_list[idx].size(); j++) {
                dot += y_list[idx][j] * result[j];
            }
            double beta = rho_list[idx] * dot;
            for (int j = 0; j < d && j < (int)s_list[idx].size(); j++) {
                result[j] += (alpha[i] - beta) * s_list[idx][j];
            }
        }
    }

    // Kinetic energy: K = 0.5 * p^T * H * p
    double kinetic_energy(const std::vector<double>& p) const {
        if ((int)p.size() != d) return 0.0;
        std::vector<double> Hp;
        multiply_H(p, Hp);
        double ke = 0.0;
        for (int i = 0; i < d; i++) {
            ke += p[i] * Hp[i];
        }
        return 0.5 * ke;
    }

    // Get diagonal of B for momentum sampling: sqrt(1/gamma)
    std::vector<double> get_sqrt_B_diag() const {
        std::vector<double> result(d);
        double sqrt_inv_gamma = std::sqrt(1.0 / gamma);
        for (int i = 0; i < d; i++) {
            result[i] = sqrt_inv_gamma;
        }
        return result;
    }
};

// Single-scale GP data structure
struct GPData {
  int n_obs;                          // Number of unique locations
  int nn;                             // Number of nearest neighbors

  std::vector<double> coords;         // Coordinates (n_obs x 2, flattened) — unique locations only

  // NNGP neighbor structure (computed on unique locations)
  std::vector<int> nn_idx;            // Neighbor indices (n_obs x nn, flattened)
  std::vector<double> nn_dist;        // Distances to neighbors (n_obs x nn, flattened)
  std::vector<double> nn_neighbor_dist;  // Phase 1.3: pairwise distances among neighbors (n_obs x nn x nn)
  std::vector<int> nn_order;          // Observation ordering for NNGP
  std::vector<int> nn_order_inv;      // Inverse ordering

  // Observation-to-location mapping (0-based, length = total N)
  // Maps each observation to its unique location index
  std::vector<int> obs_to_loc;

  CovType cov_type;
  double nu;                          // Matern smoothness (if applicable)
  bool shared;                        // Whether GP is shared between num/denom
};

// Multi-scale GP data: the field's two scales share one Hilbert-space basis
// (ModelData::msgp_hsgp_data); this carries what the rest of the sampler reads.
struct MultiscaleGPData {
  int n_obs = 0;                      // Observations the basis is evaluated at
  bool shared = true;                 // Whether the field enters both processes
  MSGPSampler sampler = MSGPSampler::AUTO;
};

// -----------------------------------------------------------------------------
// Priors for GP hyperparameters
// -----------------------------------------------------------------------------

// d/d(theta) of the PC prior on a marginal sd, sigma ~ Exponential(rate),
// written on theta = log(sigma^2): log p(theta) = const - rate * sigma +
// theta / 2 (ratiod_gp::log_prior_sd_pc_logvar_t is the density).
inline double log_prior_sd_pc_logvar_grad(double theta, double rate) {
  return 0.5 - 0.5 * rate * std::exp(0.5 * theta);
}

// Log prior for range with PC-style (favor larger ranges = simpler models)
inline double log_prior_phi_pc(double phi, double U, double alpha) {
  // P(phi < U) = alpha => 1/phi follows Exponential
  if (phi <= 0) return -INFINITY;
  double rate = -std::log(1.0 - alpha) / U;
  // Prior favors larger phi (simpler, smoother spatial structure)
  return std::log(rate) - rate / phi - 2.0 * std::log(phi);
}

// -----------------------------------------------------------------------------
// Gradient computation for GP parameters (for HMC)
// -----------------------------------------------------------------------------

// Struct to hold NNGP gradient results (for hand-coded gradients)
struct NNGPGradients {
  std::vector<double> grad_w;         // Gradient w.r.t. spatial effects
  double grad_log_sigma2;             // Gradient w.r.t. log(sigma2)
  double grad_log_phi;                // Gradient w.r.t. log(phi)
  // The NNGP log-density the gradient differentiates, accumulated in the same
  // pass. `log_lik_valid` is false where gp_nngp_log_lik_t() would return its
  // -1e10 sentinel (a neighbour block that does not factor), so a caller
  // evaluates the density itself there.
  double log_lik = 0.0;
  bool log_lik_valid = true;
};

using ratiod_cov::dcov_dphi;

// Fully analytical NNGP gradients - single pass, no redundant function calls
// Complexity: O(N * nn²) vs 4 * O(N * nn²) for numerical => ~4x faster
inline void gp_nngp_gradients(
    const std::vector<double>& w,
    double sigma2,
    double phi,
    const GPData& gp_data,
    NNGPGradients& grads,
    double /* epsilon */ = 1e-6
) {
  int N = gp_data.n_obs;
  int nn = gp_data.nn;

  grads.grad_w.assign(N, 0.0);
  grads.grad_log_sigma2 = 0.0;
  grads.grad_log_phi = 0.0;
  grads.log_lik = 0.0;
  grads.log_lik_valid = false;

  // Validate
  if ((int)gp_data.nn_order.size() < N || (int)gp_data.nn_idx.size() < N * nn ||
      (int)gp_data.nn_dist.size() < N * nn || (int)w.size() < N ||
      (int)gp_data.nn_neighbor_dist.size() < N * nn * nn ||
      (int)gp_data.coords.size() < 2 * N) return;

  const double half_log_2pi = 0.5 * std::log(2.0 * M_PI);
  const double log_sigma2 = std::log(sigma2);

  // First observation: marginal N(0, sigma2)
  int first_idx = gp_data.nn_order[0];
  if (first_idx < 0 || first_idx >= N) return;
  grads.log_lik_valid = true;
  double w0 = w[first_idx];
  grads.grad_w[first_idx] = -w0 / sigma2;
  grads.grad_log_sigma2 += 0.5 * (w0 * w0 / sigma2 - 1.0);  // Will multiply by sigma2 at end
  grads.log_lik += -half_log_2pi - 0.5 * log_sigma2 - 0.5 * w0 * w0 / sigma2;

  // Preallocate work arrays (reused across iterations)
  std::vector<double> c_vec(nn), dc_vec(nn), C_mat(nn * nn), L(nn * nn);
  std::vector<double> y_vec(nn), alpha(nn), y2(nn), beta(nn), w_nb(nn);
  std::vector<int> nb_idx(nn);

  for (int i = 1; i < N; i++) {
    int obs_idx = gp_data.nn_order[i];
    if (obs_idx < 0 || obs_idx >= N) continue;

    // Count neighbors
    int n_nb = 0;
    for (int j = 0; j < nn && gp_data.nn_idx[i * nn + j] > 0; j++) n_nb++;

    if (n_nb == 0) {
      double wi = w[obs_idx];
      grads.grad_w[obs_idx] += -wi / sigma2;
      grads.grad_log_sigma2 += 0.5 * (wi * wi / sigma2 - 1.0);
      grads.log_lik += -half_log_2pi - 0.5 * log_sigma2 - 0.5 * wi * wi / sigma2;
      continue;
    }

    // Build c_vec, dc_vec (covariances and derivatives)
    for (int j = 0; j < n_nb; j++) {
      double d = gp_data.nn_dist[i * nn + j];
      c_vec[j] = compute_cov(d, sigma2, phi, gp_data.cov_type);
      dc_vec[j] = dcov_dphi(d, sigma2, phi, c_vec[j], gp_data.cov_type);
    }

    // Build C_mat (symmetric: one kernel evaluation per neighbour pair, on the
    // cached distances the density reads) and get neighbor indices
    const double* nb_dist = &gp_data.nn_neighbor_dist[static_cast<size_t>(i) * nn * nn];
    bool ok = true;
    for (int j1 = 0; j1 < n_nb && ok; j1++) {
      int raw1 = gp_data.nn_idx[i * nn + j1];
      if (raw1 - 1 < 0 || raw1 - 1 >= (int)gp_data.nn_order.size()) { ok = false; break; }
      int idx1 = gp_data.nn_order[raw1 - 1];
      if (idx1 < 0 || idx1 >= N) { ok = false; break; }
      nb_idx[j1] = idx1;

      C_mat[j1 * n_nb + j1] = sigma2;
      for (int j2 = 0; j2 < j1; j2++) {
        double c12 = compute_cov(nb_dist[j1 * nn + j2], sigma2, phi, gp_data.cov_type);
        C_mat[j1 * n_nb + j2] = c12;
        C_mat[j2 * n_nb + j1] = c12;
      }
    }
    if (!ok) {
      double wi = w[obs_idx];
      grads.grad_w[obs_idx] += -wi / sigma2;
      grads.grad_log_sigma2 += 0.5 * (wi * wi / sigma2 - 1.0);
      grads.log_lik_valid = false;
      continue;
    }

    // Cholesky: C = LL', through the same factorization the density uses.
    if (!ratiod_cov::nngp_chol(C_mat.data(), n_nb, L.data())) {
      grads.log_lik_valid = false;
      continue;
    }

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
    grads.log_lik += -half_log_2pi - 0.5 * std::log(v) - 0.5 * r * r / v;

    // Gradient w.r.t. w
    grads.grad_w[obs_idx] += -r / v;
    for (int j = 0; j < n_nb; j++) grads.grad_w[nb_idx[j]] += alpha[j] * r / v;

    // Gradient w.r.t. sigma2: dv/ds2 = 1 - c'α/s2
    double dll_dv = 0.5 * (r * r / v - 1.0) / v;
    grads.grad_log_sigma2 += dll_dv * (1.0 - c_alpha / sigma2) * sigma2;

    // Gradient w.r.t. phi: compute quadratic forms on-the-fly
    double alpha_dc = 0.0, dc_beta = 0.0;
    for (int j = 0; j < n_nb; j++) { alpha_dc += alpha[j] * dc_vec[j]; dc_beta += dc_vec[j] * beta[j]; }

    // alpha' * dC/dphi * alpha and alpha' * dC/dphi * beta. dC/dphi is
    // symmetric with a zero diagonal, so each pair is visited once.
    double alpha_dC_alpha = 0.0, alpha_dC_beta = 0.0;
    for (int j1 = 1; j1 < n_nb; j1++) {
      for (int j2 = 0; j2 < j1; j2++) {
        double dC_jk = dcov_dphi(nb_dist[j1 * nn + j2], sigma2, phi,
                                 C_mat[j1 * n_nb + j2], gp_data.cov_type);
        alpha_dC_alpha += 2.0 * alpha[j1] * dC_jk * alpha[j2];
        alpha_dC_beta += dC_jk * (alpha[j1] * beta[j2] + alpha[j2] * beta[j1]);
      }
    }

    double dv_dphi = -2.0 * alpha_dc + alpha_dC_alpha;
    double dr_dphi = -dc_beta + alpha_dC_beta;
    grads.grad_log_phi += (dll_dv * dv_dphi + (-r / v) * dr_dphi) * phi;
  }
}

// =============================================================================
// Non-centered NNGP parameterization
// =============================================================================
// Instead of sampling w ~ NNGP(0, sigma2, phi) directly (centered),
// sample z ~ N(0, I) and transform z -> w via the NNGP autoregressive structure:
//   w[order[0]] = sqrt(sigma2) * z[0]
//   w[order[i]] = sum_j B[i,j] * w[nb_j(i)] + sqrt(d_i) * z[i]
// where B[i,:] = C_nb^{-1} c_i (regression coefficients) and
//       d_i = sigma2 - c_i' C_nb^{-1} c_i (conditional variance).
//
// This improves posterior geometry for large N, reducing NUTS treedepth.
// The prior on z is N(0,I), and no Jacobian is needed since we sample in z-space.

// The reverse pass is written for double alone -- an autodiff scalar records
// its own -- so the adjoint accumulator and the cached Cholesky factors it
// reads are allocated only in the double instantiation.
template <typename T>
struct NNGPNCWorkspaceT {
    static constexpr bool caches_for_backward = std::is_same<T, double>::value;

    int N = 0, nn = 0;
    std::vector<T> w;               // Transformed spatial effects (N)
    std::vector<T> sqrt_d;          // sqrt(conditional variance) per obs (N)
    std::vector<T> B_flat;          // Regression coefficients (N * nn)
    std::vector<int> B_n_nb;        // Number of actual neighbors per obs (N)
    std::vector<int> nb_idx_flat;   // Neighbor indices per obs (N * nn), 0-based in w
    std::vector<double> adj;        // Adjoint accumulator (N)
    std::vector<T> L_flat;          // Cached Cholesky factors (N * nn * nn) for backward phi grad

    void init(int N_, int nn_) {
        if (N == N_ && nn == nn_) return;
        N = N_; nn = nn_;
        w.resize(N);
        sqrt_d.resize(N);
        B_flat.assign(static_cast<size_t>(N) * nn, T(0.0));
        B_n_nb.assign(N, 0);
        nb_idx_flat.assign(static_cast<size_t>(N) * nn, -1);
        if (caches_for_backward) {
            adj.assign(N, 0.0);
            L_flat.assign(static_cast<size_t>(N) * nn * nn, T(0.0));
        }
    }
};

using NNGPNCWorkspace = NNGPNCWorkspaceT<double>;

// Forward pass: z -> w via NNGP autoregressive structure
// z and w are both indexed by LOCATION (0-based), matching the parameter layout.
// Caches B, sqrt_d, nb_idx for backward pass.
// O(N * nn^3) due to per-observation Cholesky.
template <typename T>
inline void nngp_nc_forward(
    const T* z,               // z[loc_idx], indexed by location, length N
    const T& sigma2, const T& phi,
    const GPData& gp_data,
    NNGPNCWorkspaceT<T>& ws
) {
    int N = gp_data.n_obs;
    int nn = gp_data.nn;
    ws.init(N, nn);

    // Work arrays for Cholesky
    std::vector<T> c_vec(nn), C_mat(nn * nn), L(nn * nn);
    std::vector<T> y_vec(nn), alpha(nn);

    // First observation: marginal N(0, sigma2)
    int first_loc = gp_data.nn_order[0];  // Location index
    ws.sqrt_d[0] = ratiod::math::safe_sqrt(sigma2);
    ws.w[first_loc] = ws.sqrt_d[0] * z[first_loc];
    ws.B_n_nb[0] = 0;

    for (int i = 1; i < N; i++) {
        int obs_loc = gp_data.nn_order[i];  // Location index for NNGP order i
        if (obs_loc < 0 || obs_loc >= N) {
            ws.sqrt_d[i] = ratiod::math::safe_sqrt(sigma2);
            ws.w[obs_loc] = ws.sqrt_d[i] * z[obs_loc];
            ws.B_n_nb[i] = 0;
            continue;
        }

        // Count neighbors
        int n_nb = 0;
        for (int j = 0; j < nn && gp_data.nn_idx[i * nn + j] > 0; j++) n_nb++;

        if (n_nb == 0) {
            ws.sqrt_d[i] = ratiod::math::safe_sqrt(sigma2);
            ws.w[obs_loc] = ws.sqrt_d[i] * z[obs_loc];
            ws.B_n_nb[i] = 0;
            continue;
        }

        // Build c_vec (covariance between obs and its neighbors)
        for (int j = 0; j < n_nb; j++) {
            double d = gp_data.nn_dist[i * nn + j];
            c_vec[j] = ratiod_cov::compute_cov(d, sigma2, phi, gp_data.cov_type);
        }

        // Build C_mat (covariance among neighbors) and get neighbor location indices
        bool ok = true;
        for (int j1 = 0; j1 < n_nb && ok; j1++) {
            int raw1 = gp_data.nn_idx[i * nn + j1];
            if (raw1 - 1 < 0 || raw1 - 1 >= (int)gp_data.nn_order.size()) { ok = false; break; }
            int loc1 = gp_data.nn_order[raw1 - 1];
            if (loc1 < 0 || loc1 >= N) { ok = false; break; }
            ws.nb_idx_flat[i * nn + j1] = loc1;

            // C is symmetric: one kernel evaluation per neighbour pair.
            C_mat[j1 * n_nb + j1] = sigma2;
            for (int j2 = 0; j2 < j1; j2++) {
                double d12 = gp_data.nn_neighbor_dist[i * nn * nn + j1 * nn + j2];
                T c12 = ratiod_cov::compute_cov(d12, sigma2, phi, gp_data.cov_type);
                C_mat[j1 * n_nb + j2] = c12;
                C_mat[j2 * n_nb + j1] = c12;
            }
        }

        if (!ok) {
            ws.sqrt_d[i] = ratiod::math::safe_sqrt(sigma2);
            ws.w[obs_loc] = ws.sqrt_d[i] * z[obs_loc];
            ws.B_n_nb[i] = 0;
            continue;
        }

        // Cholesky: C = LL', and alpha = C^{-1} c (the regression coefficients
        // B[i,:]), through the same factorization the density uses.
        if (!ratiod_cov::nngp_chol(C_mat.data(), n_nb, L.data())) {
            ws.sqrt_d[i] = ratiod::math::safe_sqrt(sigma2);
            ws.w[obs_loc] = ws.sqrt_d[i] * z[obs_loc];
            ws.B_n_nb[i] = 0;
            continue;
        }
        ratiod_cov::nngp_forward_solve(L.data(), n_nb, c_vec.data(), y_vec.data());
        ratiod_cov::nngp_back_solve(L.data(), n_nb, y_vec.data(), alpha.data());

        // Cache Cholesky factor L for backward phi gradient
        if (ws.caches_for_backward) {
            for (int j1 = 0; j1 < n_nb; j1++) {
                for (int j2 = 0; j2 <= j1; j2++) {
                    ws.L_flat[i * nn * nn + j1 * nn + j2] = L[j1 * n_nb + j2];
                }
            }
        }

        // Store B and compute conditional variance d_i
        T c_alpha = T(0.0);
        for (int j = 0; j < n_nb; j++) {
            ws.B_flat[i * nn + j] = alpha[j];
            c_alpha = c_alpha + c_vec[j] * alpha[j];
        }
        ws.B_n_nb[i] = n_nb;

        T d_i = ratiod_cov::nngp_floor_cond_var(sigma2 - c_alpha);
        ws.sqrt_d[i] = ratiod::math::safe_sqrt(d_i);

        // Forward transform: w[loc] = B @ w_neighbors + sqrt(d_i) * z[loc]
        T mu = T(0.0);
        for (int j = 0; j < n_nb; j++) {
            mu = mu + alpha[j] * ws.w[ws.nb_idx_flat[i * nn + j]];
        }
        ws.w[obs_loc] = mu + ws.sqrt_d[i] * z[obs_loc];
    }
}

// This thread's workspace for the transform, one per scalar type and slot. The
// forward pass runs on every leapfrog step, and its buffers are O(N * nn^2),
// so a workspace is held rather than allocated. `Slot` separates fields that
// are live at the same time: the single-scale field holds slot 0, the
// multi-scale field's two scales hold one slot each, and a gradient pass holds
// slots of its own because it evaluates the density after its backward pass
// has read the cached factors.
template <typename T, int Slot = 0>
inline NNGPNCWorkspaceT<T>& nngp_nc_ws() {
    RATIOD_TLS_WORKSPACE(NNGPNCWorkspaceT<T>, ws);
    return ws;
}

// Backward pass: given dL/dw from likelihood, compute gradients for z, log_sigma2, log_phi.
// z and grad_z are indexed by LOCATION (matching parameter layout).
// adj is indexed by NNGP order (internal).
//
// Adjoint propagation (reverse NNGP order):
//   adj[i] = dL/dw[order[i]]   (direct likelihood contribution)
//   For i = N-1 down to 1:
//     For each neighbor j of obs i:
//       adj[NNGP_index_of(nb_j)] += B[i,j] * adj[i]
//
// Then: grad_z[loc] = -z[loc] + sqrt_d[nngp_i] * adj[nngp_i]
//
// Outputs:
//   grad_z: full gradient for z[loc] (prior + likelihood), location-indexed
//   grad_log_sigma2_lik: likelihood contribution to d/d(log_sigma2)
//   grad_log_phi_lik: likelihood contribution to d/d(log_phi)
//   grad_log_phi_jac: Jacobian contribution to d/d(log_phi) = 0.5 * sum(dd_dphi/d_i * phi)
inline void nngp_nc_backward(
    const double* z,            // z[loc_idx], location-indexed
    double sigma2, double phi,
    const GPData& gp_data,
    const NNGPNCWorkspace& ws,
    const double* dL_dw,        // Likelihood gradient w.r.t. w[loc] (location-indexed)
    double* grad_z,             // Output: full gradient for z[loc] (prior + likelihood)
    double& grad_log_sigma2_lik,// Output: likelihood contribution to sigma2 gradient
    double& grad_log_phi_lik,   // Output: likelihood contribution to phi gradient
    double& grad_log_phi_jac    // Output: Jacobian contribution to phi gradient
) {
    int N = gp_data.n_obs;
    int nn = gp_data.nn;
    const std::vector<int>& nn_order_inv = gp_data.nn_order_inv;

    // Initialize adjoint from direct likelihood contribution (NNGP-order indexed)
    std::vector<double>& adj = const_cast<NNGPNCWorkspace&>(ws).adj;
    for (int i = 0; i < N; i++) {
        int loc = gp_data.nn_order[i];
        adj[i] = dL_dw[loc];
    }

    // Backward adjoint propagation (reverse NNGP order)
    for (int i = N - 1; i >= 1; i--) {
        int n_nb = ws.B_n_nb[i];
        for (int j = 0; j < n_nb; j++) {
            int nb_loc = ws.nb_idx_flat[i * nn + j];
            if (nb_loc >= 0 && nb_loc < N) {
                int nb_nngp = nn_order_inv[nb_loc];
                if (nb_nngp >= 0 && nb_nngp < N) {
                    adj[nb_nngp] += ws.B_flat[i * nn + j] * adj[i];
                }
            }
        }
    }

    // z gradients: prior (-z) + likelihood (sqrt_d * adj)
    // Map from NNGP order to location order
    for (int i = 0; i < N; i++) {
        int loc = gp_data.nn_order[i];
        grad_z[loc] = -z[loc] + ws.sqrt_d[i] * adj[i];
    }

    // --- Hyperparameter gradients ---

    // sigma2 likelihood gradient:
    // B independent of sigma2, d_i = sigma2 * d_tilde_i
    // dw[loc]/d(log_sigma2) = 0.5 * sqrt(d_i) * z[loc]
    grad_log_sigma2_lik = 0.0;
    for (int i = 0; i < N; i++) {
        int loc = gp_data.nn_order[i];
        grad_log_sigma2_lik += adj[i] * 0.5 * ws.sqrt_d[i] * z[loc];
    }

    // phi gradients (likelihood + Jacobian):
    // Both B and d depend on phi. Need per-obs dB/dphi and dd/dphi.
    grad_log_phi_lik = 0.0;
    grad_log_phi_jac = 0.0;

    // Work arrays
    std::vector<double> c_vec(nn), dc_vec(nn), L(nn * nn);
    std::vector<double> y_vec(nn), alpha(nn), dalpha(nn), dC_alpha(nn), rhs(nn);

    // First observation: d[0] = sigma2, independent of phi → no contribution

    for (int i = 1; i < N; i++) {
        int obs_loc = gp_data.nn_order[i];
        int n_nb = ws.B_n_nb[i];
        if (n_nb == 0 || obs_loc < 0 || obs_loc >= N) continue;

        // c_vec and its phi derivative, the factor and alpha restored from the
        // forward pass, and dC/dphi * alpha accumulated pair by pair: dC/dphi
        // is symmetric with a zero diagonal, and each entry needs the
        // covariance at that pair, so one kernel evaluation serves both.
        for (int j = 0; j < n_nb; j++) {
            double d = gp_data.nn_dist[i * nn + j];
            c_vec[j] = compute_cov(d, sigma2, phi, gp_data.cov_type);
            dc_vec[j] = dcov_dphi(d, sigma2, phi, c_vec[j], gp_data.cov_type);
        }
        for (int j1 = 0; j1 < n_nb; j1++) {
            for (int j2 = 0; j2 <= j1; j2++) {
                L[j1 * n_nb + j2] = ws.L_flat[i * nn * nn + j1 * nn + j2];
            }
            alpha[j1] = ws.B_flat[i * nn + j1];  // B = C^{-1}c = alpha
            dC_alpha[j1] = 0.0;
        }
        const double* nb_dist = &gp_data.nn_neighbor_dist[static_cast<size_t>(i) * nn * nn];
        for (int j1 = 1; j1 < n_nb; j1++) {
            for (int j2 = 0; j2 < j1; j2++) {
                double d12 = nb_dist[j1 * nn + j2];
                double c12 = compute_cov(d12, sigma2, phi, gp_data.cov_type);
                double dC_jk = dcov_dphi(d12, sigma2, phi, c12, gp_data.cov_type);
                dC_alpha[j1] += dC_jk * alpha[j2];
                dC_alpha[j2] += dC_jk * alpha[j1];
            }
        }

        // dalpha/dphi = C^{-1} (dc/dphi - dC/dphi * alpha)
        for (int j = 0; j < n_nb; j++) rhs[j] = dc_vec[j] - dC_alpha[j];
        for (int j = 0; j < n_nb; j++) {
            double s = 0.0;
            for (int k = 0; k < j; k++) s += L[j * n_nb + k] * y_vec[k];
            y_vec[j] = (rhs[j] - s) / L[j * n_nb + j];
        }
        for (int j = n_nb - 1; j >= 0; j--) {
            double s = 0.0;
            for (int k = j + 1; k < n_nb; k++) s += L[k * n_nb + j] * dalpha[k];
            dalpha[j] = (y_vec[j] - s) / L[j * n_nb + j];
        }

        // dd/dphi = -2 * dc' * alpha + alpha' * dC * alpha
        double alpha_dc = 0.0, alpha_dC_alpha = 0.0;
        for (int j = 0; j < n_nb; j++) {
            alpha_dc += alpha[j] * dc_vec[j];
            alpha_dC_alpha += alpha[j] * dC_alpha[j];
        }
        double dd_dphi = -2.0 * alpha_dc + alpha_dC_alpha;

        // Likelihood: dw[loc]/dphi = sum_j dalpha[j]*w[nb_j] + dd_dphi/(2*sqrt(d_i))*z[loc]
        double dw_dphi = 0.0;
        for (int j = 0; j < n_nb; j++) {
            dw_dphi += dalpha[j] * ws.w[ws.nb_idx_flat[i * nn + j]];
        }
        double sqrt_di = ws.sqrt_d[i];
        if (sqrt_di > 1e-15) {
            dw_dphi += dd_dphi / (2.0 * sqrt_di) * z[obs_loc];
        }
        grad_log_phi_lik += adj[i] * dw_dphi * phi;

        // Jacobian: d/d(phi) [0.5*log(d_i)] = 0.5 * dd_dphi / d_i
        double d_i = sqrt_di * sqrt_di;
        if (d_i > 1e-15) {
            grad_log_phi_jac += 0.5 * dd_dphi / d_i * phi;
        }
    }
}

} // namespace ratiod_gp

#endif // RATIOD_HMC_GP_H
