// hmc_svc_autodiff.h
// Templated SVC (Spatially-Varying Coefficients) with NNGP approximation
// Works with both double and ad::Var for automatic differentiation

#ifndef RATIOD_HMC_SVC_AUTODIFF_H
#define RATIOD_HMC_SVC_AUTODIFF_H

#define _USE_MATH_DEFINES
#include <vector>
#include <cmath>
#include <algorithm>
#include "autodiff_utils.h"
#include "hmc_cov.h"  // Shared kernels and neighbour-block factorization
#include "hmc_svc.h"  // For SVCData and CovType

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace ratiod_svc_ad {

using ratiod_svc::SVCData;
using ratiod_svc::CovType;
using namespace ratiod::math;

// The kernels, the neighbour-block Cholesky and its two solves are the shared
// ones in ratiod_cov, templated on the scalar type exactly as this file needs
// them. The copies that used to sit here carried their own Gaussian
// parameterization, their own 1e-4 diagonal ridge and their own 1e-4 blended
// conditional-variance floor, none of which the double path it is supposed to
// reproduce applied.
using ratiod_cov::compute_cov;
using ratiod_cov::nngp_chol;
using ratiod_cov::nngp_forward_solve;
using ratiod_cov::nngp_back_solve;
using ratiod_cov::nngp_floor_cond_var;

// =============================================================================
// Templated NNGP log-likelihood
// =============================================================================

// Compute NNGP log-likelihood for a single SVC term
// w: vector of SVC values at each location (length n_obs)
// sigma2: spatial variance
// phi: spatial range parameter
// Returns log p(w | sigma2, phi) under NNGP approximation
template<typename T>
T nngp_log_lik(
    const std::vector<T>& w,
    const T& sigma2,
    const T& phi,
    const SVCData& svc_data
) {
    int N = svc_data.n_obs;
    int nn = svc_data.nn;

    T log_lik = T(0.0);

    // First observation: marginal N(0, sigma2)
    int first_idx = svc_data.nn_order[0];
    log_lik = log_lik - T(0.5) * safe_log(T(2.0 * M_PI) * sigma2);
    log_lik = log_lik - T(0.5) * w[first_idx] * w[first_idx] / sigma2;

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
            log_lik = log_lik - T(0.5) * safe_log(T(2.0 * M_PI) * sigma2);
            log_lik = log_lik - T(0.5) * w[obs_idx] * w[obs_idx] / sigma2;
            continue;
        }

        // Build covariance vector c(s_i, s_{N(i)}) and matrix C(s_{N(i)}, s_{N(i)})
        std::vector<T> c_vec(n_neighbors);
        std::vector<T> C_mat(n_neighbors * n_neighbors);

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
                    // Compute distance between neighbors (fixed, not dependent on params)
                    double d12 = std::sqrt(
                        std::pow(svc_data.coords[nn_idx1 * 2] - svc_data.coords[nn_idx2 * 2], 2) +
                        std::pow(svc_data.coords[nn_idx1 * 2 + 1] - svc_data.coords[nn_idx2 * 2 + 1], 2)
                    );
                    C_mat[j1 * n_neighbors + j2] = compute_cov(d12, sigma2, phi, svc_data.cov_type);
                }
            }
        }

        // Cholesky decomposition: C = L * L^T
        std::vector<T> L;
        if (!nngp_chol(C_mat, n_neighbors, L)) {
            // Decomposition failed - return -infinity
            return T(-INFINITY);
        }

        // Solve C * alpha = c_vec via L * L^T * alpha = c_vec
        std::vector<T> y, alpha;
        nngp_forward_solve(L, n_neighbors, c_vec, y);
        nngp_back_solve(L, n_neighbors, y, alpha);

        // Conditional mean: mu_i = c^T * C^{-1} * w_{N(i)} = alpha^T * w_{N(i)}
        T cond_mean = T(0.0);
        for (int j = 0; j < n_neighbors; j++) {
            int nn_orig_idx = svc_data.nn_order[svc_data.nn_idx[i * nn + j] - 1];
            cond_mean = cond_mean + alpha[j] * w[nn_orig_idx];
        }

        // Conditional variance: var_i = sigma2 - c^T * C^{-1} * c
        T c_Cinv_c = T(0.0);
        for (int j = 0; j < n_neighbors; j++) {
            c_Cinv_c = c_Cinv_c + c_vec[j] * alpha[j];
        }
        T cond_var = nngp_floor_cond_var(sigma2 - c_Cinv_c);

        // Log-likelihood contribution
        T resid = w[obs_idx] - cond_mean;
        log_lik = log_lik - T(0.5) * safe_log(T(2.0 * M_PI) * cond_var);
        log_lik = log_lik - T(0.5) * resid * resid / cond_var;
    }

    return log_lik;
}

// =============================================================================
// SVC contribution to linear predictor
// =============================================================================

// Compute SVC contribution to linear predictor for all observations
// eta_svc[i] = sum_j X_svc[i,j] * w_j[i]
template<typename T>
void compute_svc_eta(
    const std::vector<T>& w_flat,  // n_obs x n_svc flattened
    const SVCData& svc_data,
    std::vector<T>& eta_svc         // Output: length n_obs
) {
    int N = svc_data.n_obs;
    int n_svc = svc_data.n_svc;

    eta_svc.assign(N, T(0.0));

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < n_svc; j++) {
            // w_flat is stored as [w1[1..N], w2[1..N], ...]
            T w_ij = w_flat[j * N + i];
            double x_ij = svc_data.X_svc[i * n_svc + j];
            eta_svc[i] = eta_svc[i] + T(x_ij) * w_ij;
        }
    }
}

// Identify each term's global level by centring it on its way into eta, then
// build eta from the centred field. One function so no path can compute
// svc_eta from an uncentred field; the gradient counterpart is
// ratiod_svc::svc_center_project_lik_grad in hmc_svc.h, whose comment carries
// the derivation.
template<typename T>
void svc_center_eta(
    std::vector<T>& w_flat,        // n_obs x n_svc, term-major; centred in place
    const SVCData& svc_data,
    std::vector<T>& eta_svc        // Output: length n_obs
) {
    tulpa::s2z_centre_blocks(w_flat.data(), svc_data.n_svc, svc_data.n_obs);
    compute_svc_eta(w_flat, svc_data, eta_svc);
}

// =============================================================================
// SVC prior on hyperparameters
// =============================================================================

// Log prior for sigma2 (spatial variance): Half-Cauchy
template<typename T>
T log_prior_sigma2_svc(const T& sigma2, double scale) {
    T sigma = safe_sqrt(sigma2);
    // Half-Cauchy: 2 / (pi * scale * (1 + (sigma/scale)^2))
    // Log form: log(2) - log(pi) - log(scale) - log(1 + (sigma/scale)^2)
    return T(std::log(2.0 / M_PI / scale)) - safe_log(T(1.0) + sigma * sigma / T(scale * scale));
}

// The range of an NNGP term carries a uniform prior on (lower, upper) and is
// sampled on the logit scale of that interval: phi = lower + (upper - lower) *
// s(u), s the inverse logit. Every value of u lies inside the support, so no
// proposal meets the -Inf edge a hard bound on log(phi) puts in the sampled
// space. The uniform density times the Jacobian dphi/du is s(u) (1 - s(u)) up
// to a constant, which is u's own log density.
template<typename T>
T svc_range(const T& u, double lower, double upper) {
    return T(lower) + T(upper - lower) * inv_logit(u);
}

template<typename T>
T svc_range_log_density(const T& u) {
    const T s = inv_logit(u);
    return safe_log(s) + safe_log(T(1.0) - s);
}

// u for a given range: the inverse map, for a coordinate placed at a chosen phi.
inline double svc_range_coord(double phi, double lower, double upper) {
    const double s = (phi - lower) / (upper - lower);
    return std::log(s) - std::log1p(-s);
}

// A gradient accumulated with respect to log(phi), carried onto u, plus the
// derivative of u's own log density.
inline double svc_range_grad_to_coord(double grad_log_phi, double u,
                                      double lower, double upper) {
    const double s = inv_logit(u);
    const double phi = lower + (upper - lower) * s;
    return grad_log_phi / phi * (upper - lower) * s * (1.0 - s) +
           (1.0 - 2.0 * s);
}

} // namespace ratiod_svc_ad

#endif // RATIOD_HMC_SVC_AUTODIFF_H
