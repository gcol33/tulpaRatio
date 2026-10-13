// hmc_gp_autodiff.h
// Templated GP/NNGP functions for autodiff support
// Works with both double (for evaluation) and ad::Var (for gradients)

#ifndef RATIOD_HMC_GP_AUTODIFF_H
#define RATIOD_HMC_GP_AUTODIFF_H

#include <vector>
#include <cmath>
#include "hmc_cov.h"
#include "hmc_gp.h"
#include "autodiff_utils.h"

namespace ratiod_gp {

using namespace ratiod::math;

// The kernels, the neighbour-block Cholesky and its two solves are the shared
// ones in ratiod_cov. The copies that used to sit here carried sqrt(3) rounded
// to a typed-out literal, no spherical case at all (it fell through to
// exponential), and a ridge applied only once a pivot had already gone
// non-positive, where the double path this is supposed to reproduce adds one
// unconditionally.
using ratiod_cov::compute_cov;
using ratiod_cov::nngp_chol;
using ratiod_cov::nngp_forward_solve;
using ratiod_cov::nngp_back_solve;
using ratiod_cov::nngp_floor_cond_var;

// =============================================================================
// Templated NNGP log-likelihood
// =============================================================================

// Debug flag for GP autodiff
#ifndef GP_AUTODIFF_DEBUG
#define GP_AUTODIFF_DEBUG false
#endif

// NOTE: This function has a known heisenbug with autodiff - use numerical gradients for GP
// The templated code is preserved for future optimization work
template<typename T>
T gp_nngp_log_lik_t(
    const std::vector<T>& w,
    const T& sigma2,
    const T& phi,
    const GPData& gp_data
) {
    int N = gp_data.n_obs;
    int nn = gp_data.nn;

#if AUTODIFF_DEBUG
    static int call_count = 0;
    call_count++;
    Rcpp::Rcout << "[NNGP] Call #" << call_count << ": N=" << N << ", nn=" << nn
                << ", sigma2=" << get_value(sigma2) << ", phi=" << get_value(phi)
                << ", w.size()=" << w.size() << "\n";
    R_FlushConsole();
#endif

#if GP_AUTODIFF_DEBUG
    if (call_count <= 5 || call_count % 100 == 0) {
        Rcpp::Rcout << "[GP_AD] Call #" << call_count << ": N=" << N << ", nn=" << nn
                    << ", sigma2=" << get_value(sigma2) << ", phi=" << get_value(phi) << "\n";
    }
#endif

    // Bounds validation
    if (gp_data.nn_order.size() < (size_t)N) return T(-1e10);
    if (gp_data.nn_idx.size() < (size_t)(N * nn)) return T(-1e10);
    if (gp_data.nn_dist.size() < (size_t)(N * nn)) return T(-1e10);
    if (gp_data.nn_neighbor_dist.size() < (size_t)(N * nn * nn)) return T(-1e10);  // Critical: prevents segfault
    if (w.size() < (size_t)N) return T(-1e10);
    if (gp_data.coords.size() < (size_t)(2 * N)) return T(-1e10);

    T log_lik = T(0.0);

    // First observation: marginal N(0, sigma2)
    int first_idx = gp_data.nn_order[0];
    T log_sigma2 = safe_log(sigma2);
    log_lik = log_lik - T(0.5) * safe_log(T(2.0 * M_PI)) - T(0.5) * log_sigma2;
    log_lik = log_lik - T(0.5) * w[first_idx] * w[first_idx] / sigma2;

    // Pre-allocate work vectors
    std::vector<T> c_vec(nn);
    std::vector<T> C_mat(nn * nn);
    std::vector<T> L(nn * nn);
    std::vector<T> y(nn);
    std::vector<T> alpha(nn);

    // Remaining observations: conditional on neighbors
    for (int i = 1; i < N; i++) {
#if AUTODIFF_DEBUG
        if (i <= 3 || i == N-1) {
            Rcpp::Rcout << "[NNGP] Processing obs i=" << i << "/" << N << "\n";
            R_FlushConsole();
        }
#endif
        int obs_idx = gp_data.nn_order[i];

        // Bounds check
        if (obs_idx < 0 || obs_idx >= N) return T(-1e10);

        // Count actual neighbors
        int n_neighbors = 0;
        for (int j = 0; j < nn; j++) {
            int nn_flat_idx = i * nn + j;
            if (nn_flat_idx >= (int)gp_data.nn_idx.size()) break;
            if (gp_data.nn_idx[nn_flat_idx] > 0) {
                n_neighbors++;
            }
        }

        if (n_neighbors == 0) {
            // No neighbors: marginal
            log_lik = log_lik - T(0.5) * safe_log(T(2.0 * M_PI)) - T(0.5) * log_sigma2;
            log_lik = log_lik - T(0.5) * w[obs_idx] * w[obs_idx] / sigma2;
            continue;
        }

        // c_vec: covariances between obs i and its neighbors
        for (int j = 0; j < n_neighbors; j++) {
            int nn_flat_idx = i * nn + j;
            double d = gp_data.nn_dist[nn_flat_idx];
            c_vec[j] = compute_cov(d, sigma2, phi, gp_data.cov_type);
        }

        // C_mat: covariances among neighbors
        for (int j1 = 0; j1 < n_neighbors; j1++) {
            int raw_nn_idx1 = gp_data.nn_idx[i * nn + j1];

            // Bounds check
            if (raw_nn_idx1 - 1 < 0 || raw_nn_idx1 - 1 >= (int)gp_data.nn_order.size()) {
                return T(-1e10);
            }

            int nn_idx1 = gp_data.nn_order[raw_nn_idx1 - 1];

            if (nn_idx1 < 0 || nn_idx1 * 2 + 1 >= (int)gp_data.coords.size()) {
                return T(-1e10);
            }

            for (int j2 = 0; j2 < n_neighbors; j2++) {
                int raw_nn_idx2 = gp_data.nn_idx[i * nn + j2];

                if (raw_nn_idx2 - 1 < 0 || raw_nn_idx2 - 1 >= (int)gp_data.nn_order.size()) {
                    return T(-1e10);
                }

                int nn_idx2 = gp_data.nn_order[raw_nn_idx2 - 1];

                if (j1 == j2) {
                    C_mat[j1 * n_neighbors + j2] = sigma2;
                } else {
                    // Phase 1.3: Use cached pairwise neighbor distances
                    double d12 = gp_data.nn_neighbor_dist[i * nn * nn + j1 * nn + j2];
                    C_mat[j1 * n_neighbors + j2] = compute_cov(d12, sigma2, phi, gp_data.cov_type);
                }
            }
        }

        // Cholesky decomposition
        std::vector<T> L_small;
        if (!nngp_chol(C_mat, n_neighbors, L_small)) {
            return T(-1e10);  // Not positive definite
        }

        std::vector<T> c_small(c_vec.begin(), c_vec.begin() + n_neighbors);
        std::vector<T> y_small, alpha_small;
        nngp_forward_solve(L_small, n_neighbors, c_small, y_small);
        nngp_back_solve(L_small, n_neighbors, y_small, alpha_small);

        // Conditional mean
        T cond_mean = T(0.0);
        for (int j = 0; j < n_neighbors; j++) {
            int raw_nn_idx = gp_data.nn_idx[i * nn + j];

            if (raw_nn_idx - 1 < 0 || raw_nn_idx - 1 >= (int)gp_data.nn_order.size()) {
                return T(-1e10);
            }

            int nn_orig_idx = gp_data.nn_order[raw_nn_idx - 1];

            if (nn_orig_idx < 0 || nn_orig_idx >= (int)w.size()) {
                return T(-1e10);
            }

            cond_mean = cond_mean + alpha_small[j] * w[nn_orig_idx];
        }

        // Conditional variance: sigma2 - c^T * C^{-1} * c
        T c_Cinv_c = T(0.0);
        for (int j = 0; j < n_neighbors; j++) {
            c_Cinv_c = c_Cinv_c + c_small[j] * alpha_small[j];
        }
        T cond_var = nngp_floor_cond_var(sigma2 - c_Cinv_c);

        // Log-likelihood contribution
        T resid = w[obs_idx] - cond_mean;
        log_lik = log_lik - T(0.5) * safe_log(T(2.0 * M_PI));
        log_lik = log_lik - T(0.5) * safe_log(cond_var);
        log_lik = log_lik - T(0.5) * resid * resid / cond_var;
    }

#if AUTODIFF_DEBUG
    Rcpp::Rcout << "[NNGP] Completed, log_lik=" << get_value(log_lik) << "\n";
    R_FlushConsole();
#endif

    return log_lik;
}

// =============================================================================
// Templated GP priors
// =============================================================================

// PC prior on sigma2: P(sigma > U) = alpha => sigma ~ Exp(rate = -log(alpha)/U)
template<typename T>
T log_prior_sigma2_pc_t(const T& sigma2, double U, double alpha) {
    double rate = -std::log(alpha) / U;
    T sigma = safe_sqrt(sigma2);
    // p(sigma) = rate * exp(-rate * sigma)
    // Jacobian: d(sigma)/d(sigma2) = 1/(2*sigma)
    return T(std::log(rate)) - rate * sigma - safe_log(T(2.0) * sigma);
}

// PC prior on a marginal sd, sigma ~ Exponential(rate), written on the
// coordinate theta = log(sigma^2) it is sampled in:
//   log p(theta) = log(rate) - rate * sigma + log(d sigma / d theta),
// with d sigma / d theta = sigma / 2.
template<typename T>
T log_prior_sd_pc_logvar_t(const T& theta, double rate) {
    T sigma = safe_exp(T(0.5) * theta);
    return T(std::log(rate) - std::log(2.0)) - T(rate) * sigma + T(0.5) * theta;
}

// Uniform prior on phi within bounds
template<typename T>
T log_prior_phi_uniform_t(const T& phi, double lower, double upper) {
    double phi_val = get_value(phi);
    if (phi_val < lower || phi_val > upper) {
        return T(-1e10);
    }
    return T(-std::log(upper - lower));
}

}  // namespace ratiod_gp

#endif  // RATIOD_HMC_GP_AUTODIFF_H
