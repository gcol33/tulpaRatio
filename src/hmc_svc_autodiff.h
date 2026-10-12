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
