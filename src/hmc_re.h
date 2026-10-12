// hmc_re.h
// Random-effect log prior and effective effects, one templated implementation
// serving both the analytic log posterior and the autodiff one.
//
// Covers the whole surface the parameter layout allocates for: one or more
// crossed terms, each intercept-only or carrying slopes, correlated or not, in
// the centred or the non-centred parameterization. The effective effect the
// linear predictor sees is not the parameter itself under the non-centred and
// correlated parameterizations, so it is returned alongside the prior.
//
// Include AFTER hmc_sampler.h, which defines ModelData and ParamLayout.

#ifndef RATIOD_HMC_RE_H
#define RATIOD_HMC_RE_H

#include <cmath>
#include <vector>
#include "autodiff_utils.h"

namespace ratiod_re {

using ratiod_hmc::ModelData;
using ratiod_hmc::ParamLayout;
using namespace ratiod::math;

// tanh via the logistic, so the autodiff types need no tanh primitive:
// tanh(x) = 2 * sigmoid(2x) - 1.
template <typename T>
inline T re_tanh(const T& x) {
    return T(2.0) * inv_logit(T(2.0) * x) - T(1.0);
}

// Cholesky factor of a K x K correlation matrix from its K(K-1)/2
// unconstrained coordinates, row-major over the strictly lower triangle, through
// canonical partial correlations (Lewandowski, Kurowicka & Joe 2009; the map
// Stan uses). With z = tanh(raw) and rem the squared length row r has left,
//   L[r,c] = z_rc * sqrt(rem_c),  rem_{c+1} = rem_c * (1 - z_rc^2),
//   L[r,r] = sqrt(rem_r),
// every row has unit length for every raw value, so the space has no boundary.
// Writes L (row-major, K x K) and returns log|d offdiag(L) / d raw|.
template <typename T>
inline T cpc_cholesky(const T* raw, int K, T* L) {
    T log_jac = T(0.0);
    for (int i = 0; i < K * K; i++) L[i] = T(0.0);
    L[0] = T(1.0);
    int idx = 0;
    for (int r = 1; r < K; r++) {
        T rem = T(1.0);
        for (int c = 0; c < r; c++) {
            const T z = re_tanh(raw[idx++]);
            const T one_m_z2 = T(1.0) - z * z;
            L[r * K + c] = z * safe_sqrt(rem);
            log_jac = log_jac + safe_log(one_m_z2) + T(0.5) * safe_log(rem);
            rem = rem * one_m_z2;
        }
        L[r * K + r] = safe_sqrt(rem);
    }
    return log_jac;
}

// LKJ(eta) density of the correlation matrix L L', written on L
// (Stan's lkj_corr_cholesky): sum over rows r >= 1 of
// (K - r - 1 + 2 (eta - 1)) log L[r,r], rows 0-based.
template <typename T>
inline T lkj_corr_cholesky_log(const T* L, int K, double eta) {
    T lp = T(0.0);
    for (int r = 1; r < K; r++) {
        lp = lp + T(K - r - 1 + 2.0 * (eta - 1.0)) * safe_log(L[r * K + r]);
    }
    return lp;
}

// d/d raw of lkj_corr_cholesky_log + the cpc_cholesky Jacobian. In row r both
// are sums of log(1 - z_rc^2): the Jacobian counts each once plus a half for
// every later entry in the row, and log L[r,r] = 0.5 * sum_c log(1 - z_rc^2),
// so d/d raw_rc = -2 z_rc * (1 + 0.5 * (r - 1 - c) + 0.5 * coef_r).
inline void cpc_lkj_prior_grad(const double* raw, int K, double eta, double* g_raw) {
    int idx = 0;
    for (int r = 1; r < K; r++) {
        const double coef_r = K - r - 1 + 2.0 * (eta - 1.0);
        for (int c = 0; c < r; c++) {
            const double z = std::tanh(raw[idx]);
            g_raw[idx] = -2.0 * z * (1.0 + 0.5 * (r - 1 - c) + 0.5 * coef_r);
            idx++;
        }
    }
}

// Reverse pass of cpc_cholesky: from gL = d f / d L (row-major, lower triangle
// including the diagonal) to g_raw = d f / d raw, accumulated. With
// s_c = sqrt(rem_c), dL[r,c]/dz_rj = s_c for j = c and -L[r,c] z_j / (1 - z_j^2)
// for j < c, and the diagonal depends on every z of its row the second way.
inline void cpc_cholesky_backward(const double* raw, int K, const double* L,
                                  const double* gL, double* g_raw) {
    int idx = 0;
    for (int r = 1; r < K; r++) {
        // tail_j = sum over the entries after j in row r (diagonal included)
        // of gL * L, accumulated from the diagonal backwards.
        double tail = gL[r * K + r] * L[r * K + r];
        double rem = 1.0;
        std::vector<double> s(r), z(r);
        for (int c = 0; c < r; c++) {
            z[c] = std::tanh(raw[idx + c]);
            s[c] = std::sqrt(rem);
            rem *= 1.0 - z[c] * z[c];
        }
        for (int c = r - 1; c >= 0; c--) {
            const double one_m_z2 = 1.0 - z[c] * z[c];
            g_raw[idx + c] += gL[r * K + c] * s[c] * one_m_z2 - z[c] * tail;
            tail += gL[r * K + c] * L[r * K + c];
        }
        idx += r;
    }
}

// Shape of the LKJ prior on a correlated term's correlation matrix.
constexpr double RE_LKJ_ETA = 2.0;

// Effective random effects, laid out per term as [group * n_coefs + coef].
template <typename T>
using ReEffects = std::vector<std::vector<T> >;

// Number of coefficients this term carries (1 = intercept only).
inline int re_term_n_coefs(const ParamLayout& layout, int t) {
    return layout.has_re_slopes ? layout.re_n_coefs_multi[t] : 1;
}

// Index of the log_sigma parameter for coefficient c of term t.
inline int re_log_sigma_idx(const ParamLayout& layout, int n_terms, int t, int c) {
    if (layout.has_re_slopes) return layout.log_sigma_re_slopes[t][c];
    if (n_terms > 1) return layout.log_sigma_re_multi[t];
    return layout.log_sigma_re_idx;
}

// Log prior on every random-effect term, filling `re_eff` with the effects the
// linear predictor sees. Sets `ok` false when the Cholesky factor of a
// correlated term leaves no positive diagonal, which the caller reports as
// -infinity.
template <typename T>
T re_log_prior(const std::vector<T>& params,
               const ModelData& data,
               const ParamLayout& layout,
               ReEffects<T>& re_eff,
               bool& ok) {
    ok = true;
    T log_prior = T(0.0);
    if (!layout.has_re) {
        re_eff.clear();
        return log_prior;
    }

    const int n_terms = (data.n_re_terms > 0) ? data.n_re_terms : 1;
    re_eff.assign(n_terms, std::vector<T>());

    for (int t = 0; t < n_terms; t++) {
        const int n_groups = (data.n_re_terms > 0) ? data.re_n_groups_multi[t]
                                                   : data.n_re_groups;
        const int n_coefs = re_term_n_coefs(layout, t);
        const bool correlated = layout.has_re_slopes
                                && layout.re_correlated_multi[t] && n_coefs > 1;

        // Half-Cauchy(0, scale) on each standard deviation, with the Jacobian of
        // the log transform.
        std::vector<T> sigmas(n_coefs);
        for (int c = 0; c < n_coefs; c++) {
            const T log_sigma = params[re_log_sigma_idx(layout, n_terms, t, c)];
            sigmas[c] = safe_exp(log_sigma);
            log_prior = log_prior + log_prior_half_cauchy(log_sigma, data.sigma_re_scale);
        }

        // Correlated slopes: Sigma = diag(sigma) L L' diag(sigma), L the
        // Cholesky factor of an LKJ(2) correlation matrix, sampled on its
        // canonical partial correlations (cpc_cholesky).
        std::vector<T> L_flat;
        if (correlated) {
            const int chol_start = layout.chol_re_start_multi[t];
            const int n_chol = n_coefs * (n_coefs - 1) / 2;
            std::vector<T> raw(params.begin() + chol_start,
                               params.begin() + chol_start + n_chol);
            L_flat.assign(static_cast<size_t>(n_coefs) * n_coefs, T(0.0));
            // Two statements: the LKJ term reads the factor cpc_cholesky() writes.
            log_prior = log_prior + cpc_cholesky(raw.data(), n_coefs, L_flat.data());
            log_prior = log_prior + lkj_corr_cholesky_log(L_flat.data(), n_coefs, RE_LKJ_ETA);
        }

        const int re_start = (n_terms > 1 || layout.has_re_slopes)
            ? layout.re_start_multi[t] : layout.re_start;
        re_eff[t].assign(static_cast<size_t>(n_groups) * n_coefs, T(0.0));

        if (correlated) {
            // Non-centred: params hold z ~ N(0, I) and re = diag(sigma) L z. The
            // |det(diag(sigma) L)| of the change of variables cancels the
            // |Sigma|^{-1/2} of the density exactly, so no determinant appears.
            for (int g = 0; g < n_groups; g++) {
                for (int c = 0; c < n_coefs; c++) {
                    T Lz = T(0.0);
                    for (int k = 0; k <= c; k++) {
                        Lz = Lz + L_flat[c * n_coefs + k]
                                * params[re_start + g * n_coefs + k];
                    }
                    re_eff[t][g * n_coefs + c] = sigmas[c] * Lz;
                    const T z_gc = params[re_start + g * n_coefs + c];
                    log_prior = log_prior - T(0.5) * z_gc * z_gc;
                }
            }
        } else if (data.re_parameterization == 1) {
            for (int g = 0; g < n_groups; g++) {
                for (int c = 0; c < n_coefs; c++) {
                    const T z_gc = params[re_start + g * n_coefs + c];
                    re_eff[t][g * n_coefs + c] = sigmas[c] * z_gc;
                    log_prior = log_prior - T(0.5) * z_gc * z_gc;
                }
            }
        } else {
            for (int g = 0; g < n_groups; g++) {
                for (int c = 0; c < n_coefs; c++) {
                    const T re_val = params[re_start + g * n_coefs + c];
                    re_eff[t][g * n_coefs + c] = re_val;
                    const T tau_re = T(1.0) / (sigmas[c] * sigmas[c] + T(1e-10));
                    log_prior = log_prior - T(0.5) * tau_re * re_val * re_val
                                          + T(0.5) * safe_log(tau_re);
                }
            }
            log_prior = log_prior
                - T(0.5 * n_groups * n_coefs * std::log(2.0 * M_PI));
        }
    }

    return log_prior;
}

// Random-effect contribution to one observation's linear predictor.
template <typename T>
T re_eta(int i, const ModelData& data, const ParamLayout& layout,
         const ReEffects<T>& re_eff) {
    if (re_eff.empty()) return T(0.0);
    const int n_terms = static_cast<int>(re_eff.size());
    T contrib = T(0.0);

    if (layout.has_re_slopes) {
        for (int t = 0; t < n_terms; t++) {
            const int group_idx = data.re_group_multi_flat[i * n_terms + t];
            if (group_idx <= 0) continue;
            const int g = group_idx - 1;
            const int n_coefs = layout.re_n_coefs_multi[t];
            contrib = contrib + re_eff[t][g * n_coefs];
            const int n_slopes = n_coefs - 1;
            if (n_slopes > 0 && !data.re_slope_matrices[t].empty()) {
                for (int s = 0; s < n_slopes; s++) {
                    contrib = contrib + re_eff[t][g * n_coefs + 1 + s]
                        * T(data.re_slope_matrices[t][i * n_slopes + s]);
                }
            }
        }
    } else if (n_terms > 1) {
        for (int t = 0; t < n_terms; t++) {
            const int group_idx = data.re_group_multi_flat[i * n_terms + t];
            if (group_idx > 0) contrib = contrib + re_eff[t][group_idx - 1];
        }
    } else if (!data.re_group.empty() && data.re_group[i] > 0) {
        contrib = contrib + re_eff[0][data.re_group[i] - 1];
    }

    return contrib;
}

}  // namespace ratiod_re

#endif  // RATIOD_HMC_RE_H
