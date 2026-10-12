// hmc_re_blocks_grad.h
// The random-effect gradient, one block per RE term: the prior phase that
// builds each term's effective effects, and the chain rule that carries the
// likelihood gradient back to the parameters those effects were built from.
//
// Both compute_gradient_analytical() and compute_gradient_composite() call
// this. The Cholesky factor of a correlated term, its LKJ prior and the
// non-centred transform re = diag(sigma) L z are written once, so the effects
// a loop adds to eta and the ones the write-back differentiates cannot fall
// out of step.
//
// Include AFTER hmc_sampler.h, which defines ModelData and ParamLayout.

#ifndef RATIOD_HMC_RE_BLOCKS_GRAD_H
#define RATIOD_HMC_RE_BLOCKS_GRAD_H

#include <cmath>
#include <vector>
#include "hmc_sampler.h"
#include "hmc_re.h"

namespace ratiod_hmc {

// What the prior phase leaves for the observation loop and the write-back.
struct ReBlockGrad {
  std::vector<std::vector<double>> lik;      // [term][g * n_coefs + c]
  std::vector<double> re_nc_flat;            // diag(sigma) L z, indexed like params
  std::vector<std::vector<double>> L_flats;  // [term] Cholesky factor; empty unless correlated
  std::vector<std::vector<double>> sigmas;   // [term] per-coefficient scales
  int n_terms = 0;
  bool nc = false;                           // slopes stored non-centred

  void reset() {
    lik.clear();
    re_nc_flat.clear();
    L_flats.clear();
    sigmas.clear();
    n_terms = 0;
    nc = false;
  }

  // A term whose effective effects live in re_nc_flat rather than in params.
  bool is_correlated_nc(const ParamLayout& layout, int t) const {
    return !re_nc_flat.empty() &&
           static_cast<int>(layout.re_correlated_multi.size()) > t &&
           layout.re_correlated_multi[t] &&
           layout.re_n_coefs_multi[t] > 1;
  }
};

// Prior gradients for every RE term, and the effective effects the
// observation loop reads. Returns false when a term's Cholesky factor left
// the unit ball, which the caller answers by abandoning the gradient.
inline bool re_blocks_prior_grad(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    std::vector<double>& grad,
    ReBlockGrad& ws
) {
  ws.reset();
  ws.nc = (layout.has_re_slopes && data.re_parameterization == 1);

  if (layout.has_re && layout.has_re_slopes && !layout.has_re_correlated_slopes) {
    // ============ Uncorrelated random slopes prior gradients ============
    ws.n_terms = data.n_re_terms;
    ws.lik.resize(ws.n_terms);

    for (int t = 0; t < ws.n_terms; t++) {
      int n_groups = data.re_n_groups_multi[t];
      int n_coefs = layout.re_n_coefs_multi[t];
      int re_start_t = layout.re_start_multi[t];
      ws.lik[t].assign(n_groups * n_coefs, 0.0);

      // Extract sigma parameters and compute priors
      for (int c = 0; c < n_coefs; c++) {
        int log_sigma_idx = layout.log_sigma_re_slopes[t][c];
        double log_sigma_c = params[log_sigma_idx];
        double sigma_c = std::exp(log_sigma_c);

        // Half-Cauchy prior on sigma_c
        double ratio_c = sigma_c / data.sigma_re_scale;
        double ratio_c_sq = ratio_c * ratio_c;
        grad[log_sigma_idx] = -2.0 * ratio_c_sq / (1.0 + ratio_c_sq) + 1.0;

        if (ws.nc) {
          // Non-centered: params store z ~ N(0,1). Prior: -0.5*z^2
          // No sigma contribution from z prior (chain rule applied after obs loop)
          for (int g = 0; g < n_groups; g++) {
            double z_gc = params[re_start_t + g * n_coefs + c];
            grad[re_start_t + g * n_coefs + c] = -z_gc;
          }
        } else {
          // Centered: params store re ~ N(0, sigma_c^2)
          double tau_c = 1.0 / (sigma_c * sigma_c + 1e-10);
          double sigma_grad_c = 0.0;
          for (int g = 0; g < n_groups; g++) {
            double re_gc = params[re_start_t + g * n_coefs + c];
            grad[re_start_t + g * n_coefs + c] = -tau_c * re_gc;
            sigma_grad_c += tau_c * re_gc * re_gc - 1.0;
          }
          grad[log_sigma_idx] += sigma_grad_c;
        }
      }
    }
  } else if (layout.has_re && layout.has_re_slopes && layout.has_re_correlated_slopes) {
    // ============ Correlated random slopes prior gradients ============
    // Multivariate normal with Sigma = diag(sigma) * L * L' * diag(sigma)
    // where L is lower-triangular Cholesky factor with L[i,i] = sqrt(1 - sum_{j<i} L[i,j]^2)
    // LKJ(eta=2) prior on correlation matrix
    ws.n_terms = data.n_re_terms;
    ws.lik.resize(ws.n_terms);

    for (int t = 0; t < ws.n_terms; t++) {
      int n_groups = data.re_n_groups_multi[t];
      int n_coefs = layout.re_n_coefs_multi[t];
      int re_start_t = layout.re_start_multi[t];
      bool is_correlated = layout.re_correlated_multi[t];
      ws.lik[t].assign(n_groups * n_coefs, 0.0);

      // Extract sigma parameters
      std::vector<double> sigmas(n_coefs);
      for (int c = 0; c < n_coefs; c++) {
        int log_sigma_idx = layout.log_sigma_re_slopes[t][c];
        sigmas[c] = std::exp(params[log_sigma_idx]);

        // Half-Cauchy prior on sigma_c: d/d(log_sigma) = -2*(sigma/scale)^2/(1+(sigma/scale)^2) + 1
        double ratio_c = sigmas[c] / data.sigma_re_scale;
        double ratio_c_sq = ratio_c * ratio_c;
        grad[log_sigma_idx] = -2.0 * ratio_c_sq / (1.0 + ratio_c_sq) + 1.0;
      }

      if (is_correlated && n_coefs > 1) {
        // Cholesky factor of the correlation matrix, on the map the density uses
        int chol_start = layout.chol_re_start_multi[t];
        std::vector<double> L_flat(n_coefs * n_coefs, 0.0);
        ratiod_re::cpc_cholesky(&params[chol_start], n_coefs, L_flat.data());

        // ---- Non-centered parameterization ----
        // Params store z ~ N(0,1). Compute re = diag(sigma) * L * z for observation loop.

        // Allocate ws.re_nc_flat if needed
        if (ws.re_nc_flat.empty()) {
          ws.re_nc_flat.assign(params.size(), 0.0);
        }

        // Pre-compute re from z for all groups
        for (int g = 0; g < n_groups; g++) {
          for (int c = 0; c < n_coefs; c++) {
            double Lz_c = 0.0;
            for (int k = 0; k <= c; k++) {
              Lz_c += L_flat[c * n_coefs + k] * params[re_start_t + g * n_coefs + k];
            }
            ws.re_nc_flat[re_start_t + g * n_coefs + c] = sigmas[c] * Lz_c;
          }
        }

        // Save term data for write-back chain rule
        ws.L_flats.resize(ws.n_terms);
        ws.sigmas.resize(ws.n_terms);
        ws.L_flats[t] = L_flat;
        ws.sigmas[t] = sigmas;

        // Prior on z: N(0, I) -> grad[z_idx] = -z[g,c]
        for (int g = 0; g < n_groups; g++) {
          for (int c = 0; c < n_coefs; c++) {
            grad[re_start_t + g * n_coefs + c] = -params[re_start_t + g * n_coefs + c];
          }
        }

        // Sigma: Half-Cauchy prior already written above. No centered contribution.
        // In non-centered, the log-det of Jacobian (re = diag(sigma)*L*z)
        // cancels with the |Sigma|^{-1/2} normalization, so no -n_groups term.

        // Cholesky coordinates: the LKJ prior and the map's Jacobian. No
        // centered prior contribution in non-centered parameterization.
        ratiod_re::cpc_lkj_prior_grad(&params[chol_start], n_coefs,
                                      ratiod_re::RE_LKJ_ETA, &grad[chol_start]);
      } else {
        // Uncorrelated term within a mixed model (fallback)
        for (int c = 0; c < n_coefs; c++) {
          double tau_c = 1.0 / (sigmas[c] * sigmas[c] + 1e-10);
          double sigma_grad_c = 0.0;
          for (int g = 0; g < n_groups; g++) {
            double re_gc = params[re_start_t + g * n_coefs + c];
            grad[re_start_t + g * n_coefs + c] = -tau_c * re_gc;
            sigma_grad_c += tau_c * re_gc * re_gc - 1.0;
          }
          int log_sigma_idx = layout.log_sigma_re_slopes[t][c];
          grad[log_sigma_idx] += sigma_grad_c;
        }
      }
    }
  } else if (layout.has_re && !layout.has_re_slopes) {
    // Intercept-only RE (single or crossed terms)
    int n_terms = (data.n_re_terms > 1) ? data.n_re_terms : 1;

    for (int t = 0; t < n_terms; t++) {
      // Get term-specific parameters
      int log_sigma_idx = (n_terms > 1) ? layout.log_sigma_re_multi[t] : layout.log_sigma_re_idx;
      int re_start_t = (n_terms > 1) ? layout.re_start_multi[t] : layout.re_start;
      int n_groups_t = (n_terms > 1) ? data.re_n_groups_multi[t] : data.n_re_groups;

      double log_sigma_t = params[log_sigma_idx];
      double sigma_t = std::exp(log_sigma_t);

      // Half-Cauchy prior on sigma_t (same for both parameterizations)
      double ratio_t = sigma_t / data.sigma_re_scale;
      double ratio_t_sq = ratio_t * ratio_t;
      grad[log_sigma_idx] = -2.0 * ratio_t_sq / (1.0 + ratio_t_sq) + 1.0;

      if (data.re_parameterization == 1) {
        // Non-centered: z ~ N(0, 1), prior grad = -z
        // No sigma contribution from z prior (sigma gradient comes from likelihood chain rule)
        for (int g = 0; g < n_groups_t; g++) {
          double z_g = params[re_start_t + g];
          grad[re_start_t + g] = -z_g;
        }
      } else {
        // Centered: re ~ N(0, sigma_t^2)
        double tau_t = 1.0 / (sigma_t * sigma_t + 1e-10);
        double sigma_grad_t = 0.0;
        for (int g = 0; g < n_groups_t; g++) {
          double re_g = params[re_start_t + g];
          grad[re_start_t + g] = -tau_t * re_g;
          sigma_grad_t += tau_t * re_g * re_g - 1.0;
        }
        grad[log_sigma_idx] += sigma_grad_t;
      }
    }
  }
  return true;
}

// The block's contribution to observation i's linear predictor.
inline double re_blocks_eta(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    const ReBlockGrad& ws,
    int i
) {
  if (!layout.has_re) return 0.0;
  const int n_terms = (data.n_re_terms > 0) ? data.n_re_terms : 1;
  double eta = 0.0;
  for (int t = 0; t < n_terms; t++) {
    int g = -1;
    if (data.n_re_terms > 1 && !data.re_group_multi_flat.empty()) {
      g = data.re_group_multi_flat[i * data.n_re_terms + t] - 1;
    } else if (!data.re_group.empty() && data.re_group[i] > 0) {
      g = data.re_group[i] - 1;
    }
    if (g < 0) continue;

    if (!layout.has_re_slopes) {
      const int re_start_t = (data.n_re_terms > 1) ? layout.re_start_multi[t]
                                                   : layout.re_start;
      const int log_sigma_idx = (data.n_re_terms > 1) ? layout.log_sigma_re_multi[t]
                                                      : layout.log_sigma_re_idx;
      double v = params[re_start_t + g];
      if (data.re_parameterization == 1) v *= std::exp(params[log_sigma_idx]);
      eta += v;
      continue;
    }

    const int n_coefs = layout.re_n_coefs_multi[t];
    const int re_base = layout.re_start_multi[t] + g * n_coefs;
    const bool is_corr = ws.is_correlated_nc(layout, t);
    const bool is_uncorr_nc = !is_corr && ws.nc;

    double contrib;
    if (is_corr) {
      contrib = ws.re_nc_flat[re_base];
    } else if (is_uncorr_nc) {
      contrib = std::exp(params[layout.log_sigma_re_slopes[t][0]]) * params[re_base];
    } else {
      contrib = params[re_base];
    }

    const int n_slopes = n_coefs - 1;
    if (n_slopes > 0 && t < static_cast<int>(data.re_slope_matrices.size()) &&
        !data.re_slope_matrices[t].empty()) {
      for (int s = 0; s < n_slopes; s++) {
        const double x_slope = data.re_slope_matrices[t][i * n_slopes + s];
        double re_slope;
        if (is_corr) {
          re_slope = ws.re_nc_flat[re_base + 1 + s];
        } else if (is_uncorr_nc) {
          re_slope = std::exp(params[layout.log_sigma_re_slopes[t][1 + s]]) *
                     params[re_base + 1 + s];
        } else {
          re_slope = params[re_base + 1 + s];
        }
        contrib += re_slope * x_slope;
      }
    }
    eta += contrib;
  }
  return eta;
}

// Observation i's shared residual, into whichever accumulator the write-back
// reads: a slope block's own likelihood array, an intercept-only block's slot
// in grad.
inline void re_blocks_scatter(
    const ModelData& data,
    const ParamLayout& layout,
    ReBlockGrad& ws,
    int i,
    double dLL_shared,
    std::vector<double>& grad
) {
  if (!layout.has_re) return;
  const int n_terms = (data.n_re_terms > 0) ? data.n_re_terms : 1;
  for (int t = 0; t < n_terms; t++) {
    int g = -1;
    if (data.n_re_terms > 1 && !data.re_group_multi_flat.empty()) {
      g = data.re_group_multi_flat[i * data.n_re_terms + t] - 1;
    } else if (!data.re_group.empty() && data.re_group[i] > 0) {
      g = data.re_group[i] - 1;
    }
    if (g < 0) continue;

    if (!layout.has_re_slopes) {
      const int re_start_t = (data.n_re_terms > 1) ? layout.re_start_multi[t]
                                                   : layout.re_start;
      grad[re_start_t + g] += dLL_shared;
      continue;
    }

    const int n_coefs = layout.re_n_coefs_multi[t];
    ws.lik[t][g * n_coefs + 0] += dLL_shared;
    const int n_slopes = n_coefs - 1;
    if (n_slopes > 0 && t < static_cast<int>(data.re_slope_matrices.size()) &&
        !data.re_slope_matrices[t].empty()) {
      for (int s = 0; s < n_slopes; s++) {
        ws.lik[t][g * n_coefs + 1 + s] +=
            dLL_shared * data.re_slope_matrices[t][i * n_slopes + s];
      }
    }
  }
}

// The chain rule from the effective effects back to the sampled parameters.
inline void re_blocks_writeback(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    std::vector<double>& grad,
    const ReBlockGrad& ws
) {
  if (layout.has_re_slopes && ws.n_terms > 0) {
    for (int t = 0; t < ws.n_terms; t++) {
      int n_groups = data.re_n_groups_multi[t];
      int n_coefs = layout.re_n_coefs_multi[t];
      int re_start_t = layout.re_start_multi[t];
      bool is_nc = (t < (int)ws.L_flats.size() && !ws.L_flats[t].empty());

      if (is_nc) {
        // Non-centered correlated slopes: chain rule transformation
        // ws.lik[t] contains dLL/d(re), but params store z.
        // Need to transform to dLL/d(z) and add sigma/chol gradients.
        const auto& L_flat = ws.L_flats[t];
        const auto& sigmas = ws.sigmas[t];

        // 1. Transform grad_re_lik to grad_z via chain rule:
        //    dLL/dz[g,k] = sum_{c>=k} dLL/dre[g,c] * sigma[c] * L[c,k]
        for (int g = 0; g < n_groups; g++) {
          for (int k = 0; k < n_coefs; k++) {
            double grad_z_lik = 0.0;
            for (int c = k; c < n_coefs; c++) {
              grad_z_lik += ws.lik[t][g * n_coefs + c] *
                            sigmas[c] * L_flat[c * n_coefs + k];
            }
            grad[re_start_t + g * n_coefs + k] += grad_z_lik;
          }
        }

        // 2. Sigma gradient from likelihood:
        //    dLL/d(log_sigma[c]) = sum_g dLL/dre[g,c] * re_nc[g,c]
        for (int c = 0; c < n_coefs; c++) {
          double sigma_lik_grad = 0.0;
          for (int g = 0; g < n_groups; g++) {
            sigma_lik_grad += ws.lik[t][g * n_coefs + c] *
                              ws.re_nc_flat[re_start_t + g * n_coefs + c];
          }
          int log_sigma_idx = layout.log_sigma_re_slopes[t][c];
          grad[log_sigma_idx] += sigma_lik_grad;
        }

        // 3. Cholesky gradient from likelihood: re[g,i] = sigma[i] sum_k
        //    L[i,k] z[g,k], so dLL/dL[i,k] = sigma[i] sum_g dLL/dre[g,i] z[g,k],
        //    carried back to the unconstrained coordinates through the map.
        int chol_start = layout.chol_re_start_multi[t];
        std::vector<double> gL(n_coefs * n_coefs, 0.0);
        for (int ii = 1; ii < n_coefs; ii++) {
          for (int k = 0; k <= ii; k++) {
            double S_ik = 0.0;
            for (int g = 0; g < n_groups; g++) {
              S_ik += ws.lik[t][g * n_coefs + ii] *
                      params[re_start_t + g * n_coefs + k];  // z[g,k]
            }
            gL[ii * n_coefs + k] = sigmas[ii] * S_ik;
          }
        }
        ratiod_re::cpc_cholesky_backward(&params[chol_start], n_coefs,
                                         L_flat.data(), gL.data(),
                                         &grad[chol_start]);

      } else if (data.re_parameterization == 1) {
        // Uncorrelated non-centered: apply chain rule re = sigma * z
        // ws.lik[t][g*nc+c] = dLL/d(re[g,c])
        // grad[z_gc] += dLL/d(re_gc) * sigma_c
        // grad[log_sigma_c] += dLL/d(re_gc) * z_gc * sigma_c
        for (int c = 0; c < n_coefs; c++) {
          double sigma_c = std::exp(params[layout.log_sigma_re_slopes[t][c]]);
          double sigma_lik_grad = 0.0;
          for (int g = 0; g < n_groups; g++) {
            double lik_gc = ws.lik[t][g * n_coefs + c];
            double z_gc = params[re_start_t + g * n_coefs + c];
            grad[re_start_t + g * n_coefs + c] += lik_gc * sigma_c;
            sigma_lik_grad += lik_gc * z_gc * sigma_c;
          }
          grad[layout.log_sigma_re_slopes[t][c]] += sigma_lik_grad;
        }
      } else {
        // Uncorrelated centered: add ws.lik directly
        for (int g = 0; g < n_groups; g++) {
          for (int c = 0; c < n_coefs; c++) {
            grad[re_start_t + g * n_coefs + c] += ws.lik[t][g * n_coefs + c];
          }
        }
      }
    }
  }

  // ============ Non-centered RE post-processing ============
  // At this point, grad[re+g] = prior_grad + centered_lik_grad
  // For non-centered: prior_grad = -z_g, so centered_lik = grad[re+g] + z_g
  // Transform: grad[z_g] = -z_g + sigma * centered_lik (chain rule through re = sigma*z)
  //            grad[log_sigma] += sigma * sum(z_g * centered_lik)
  if (layout.has_re && !layout.has_re_slopes && data.re_parameterization == 1) {
    int n_terms = (data.n_re_terms > 1) ? data.n_re_terms : 1;
    for (int t = 0; t < n_terms; t++) {
      int re_start_t = (n_terms > 1) ? layout.re_start_multi[t] : layout.re_start;
      int n_groups_t = (n_terms > 1) ? data.re_n_groups_multi[t] : data.n_re_groups;
      int log_sigma_idx = (n_terms > 1) ? layout.log_sigma_re_multi[t] : layout.log_sigma_re_idx;
      double sigma_t = std::exp(params[log_sigma_idx]);

      double sigma_lik_grad = 0.0;
      for (int g = 0; g < n_groups_t; g++) {
        double z_g = params[re_start_t + g];
        // Extract centered lik grad: total - prior = (grad[re+g]) - (-z_g) = grad[re+g] + z_g
        double centered_lik = grad[re_start_t + g] + z_g;
        // z gradient = prior + chain rule through sigma*z
        grad[re_start_t + g] = -z_g + sigma_t * centered_lik;
        // sigma gradient from likelihood: z_g * d_ll/d_re_g
        sigma_lik_grad += z_g * centered_lik;
      }
      // d_ll/d_log_sigma = sigma * sum(z_g * d_ll/d_re_g)
      grad[log_sigma_idx] += sigma_t * sigma_lik_grad;
    }
  }
}

}  // namespace ratiod_hmc

#endif  // RATIOD_HMC_RE_BLOCKS_GRAD_H
