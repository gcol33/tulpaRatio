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

} // namespace ratiod_svc

#endif // RATIOD_HMC_SVC_H
