#ifndef TULPARATIO_HMC_GRAD_HELPERS_H
#define TULPARATIO_HMC_GRAD_HELPERS_H
#include <RcppEigen.h>
#include "hmc_grad_internal.h"
#include "tls_workspace.h"
#include "log_post_impl.h"
#include "hmc_gradient_vectorized.h"
#include "ar1_shared.h"
#include "hmc_car_proper.h"

namespace ratiod_hmc {
RATIOD_TLS_WORKSPACE_FN(vectorized::VecGradWorkspace, vec_grad_ws)
static inline double icar_quadratic_form_ptr(
    const double* phi, int J,
    const ModelData& data
) {
  return ratiod::icar_quadratic_form_t(phi, J, data);
}

static inline void re_gradient_prior(
    const ModelData& data,
    const ParamLayout& layout,
    const double* re,   // re[g] = params[re_start + g]
    double* grad,
    double sigma_re
) {
    if (!layout.has_re || data.n_re_groups <= 0) return;

    // Half-Cauchy prior on sigma_re (log-scale)
    double ratio = sigma_re / data.sigma_re_scale;
    double ratio_sq = ratio * ratio;
    grad[layout.log_sigma_re_idx] = -2.0 * ratio_sq / (1.0 + ratio_sq) + 1.0;

    if (data.re_parameterization == 1) {
        // Non-centered: z ~ N(0,1), prior grad = -z
        for (int g = 0; g < data.n_re_groups; g++) {
            grad[layout.re_start + g] = -re[g];
        }
    } else {
        // Centered: re ~ N(0, sigma^2)
        double tau_re = 1.0 / (sigma_re * sigma_re + 1e-10);
        for (int g = 0; g < data.n_re_groups; g++) {
            grad[layout.re_start + g] = -tau_re * re[g];
            grad[layout.log_sigma_re_idx] += tau_re * re[g] * re[g] - 1.0;
        }
    }
}

// Get RE value for observation (handles NC -> sigma*z transformation)
static inline double re_value_for_eta(
    const double* re,
    int g,
    double sigma_re,
    int re_parameterization
) {
    double val = re[g];
    if (re_parameterization == 1) val *= sigma_re;
    return val;
}

// Apply NC chain rule transformation after observation loop
// Must be called AFTER observation loop has accumulated likelihood gradients in grad[re+g]
static inline void re_gradient_nc_transform(
    const ModelData& data,
    const ParamLayout& layout,
    const double* params,
    double* grad,
    double sigma_re
) {
    if (!layout.has_re || data.n_re_groups <= 0 || data.re_parameterization != 1) return;

    double sigma_lik_grad = 0.0;
    for (int g = 0; g < data.n_re_groups; g++) {
        double z_g = params[layout.re_start + g];
        // Extract centered lik grad: total - prior = grad[re+g] - (-z_g) = grad[re+g] + z_g
        double centered_lik = grad[layout.re_start + g] + z_g;
        // z gradient = prior + chain rule through sigma*z
        grad[layout.re_start + g] = -z_g + sigma_re * centered_lik;
        // sigma gradient from likelihood: z_g * d_ll/d_re_g
        sigma_lik_grad += z_g * centered_lik;
    }
    // d_ll/d_log_sigma = sigma * sum(z_g * d_ll/d_re_g)
    grad[layout.log_sigma_re_idx] += sigma_re * sigma_lik_grad;
}


// Common parameters extracted from the parameter vector
struct CommonGradParams {
    const double* beta_num;
    const double* beta_denom;
    double sigma_re;
    const double* re;
    double phi_num;
    double phi_denom;
};

// Extract common parameters from the HMC parameter vector
static inline CommonGradParams extract_common_params(
    const std::vector<double>& params,
    const ParamLayout& layout
) {
    CommonGradParams cp;
    cp.beta_num = &params[layout.beta_num_start];
    cp.beta_denom = &params[layout.beta_denom_start];
    cp.sigma_re = layout.has_re ? std::exp(params[layout.log_sigma_re_idx]) : 1.0;
    cp.re = layout.has_re ? &params[layout.re_start] : nullptr;
    cp.phi_num = layout.has_phi_num ? std::exp(params[layout.log_phi_num_idx]) : 1.0;
    cp.phi_denom = layout.has_phi_denom ? std::exp(params[layout.log_phi_denom_idx]) : 1.0;
    return cp;
}

// Beta N(0, sigma_beta^2) prior gradient
// d/d(beta) = -tau_beta * beta where tau_beta = 1/sigma_beta^2
static inline void beta_gradient_prior(
    const ModelData& data, const ParamLayout& layout,
    const double* beta_num, const double* beta_denom,
    double* grad
) {
    double tau_beta = 1.0 / (data.sigma_beta * data.sigma_beta);
    for (int j = 0; j < data.p_num; j++) {
        grad[layout.beta_num_start + j] = -tau_beta * beta_num[j];
    }
    for (int j = 0; j < data.p_denom; j++) {
        grad[layout.beta_denom_start + j] = -tau_beta * beta_denom[j];
    }
}

// Phi Gamma(shape, rate) prior gradient on log-scale
// d/d(log_phi) = shape - rate*phi
// (equivalently: (shape-1) - rate*phi + 1 with Jacobian expanded)
static inline void phi_gradient_prior(
    const ModelData& data, const ParamLayout& layout,
    double phi_num, double phi_denom,
    double* grad
) {
    if (layout.has_phi_num) {
        grad[layout.log_phi_num_idx] = ratiod::math::log_prior_gamma_score(
            phi_num,data.phi_prior_shape,data.phi_prior_rate);
    }
    if (layout.has_phi_denom) {
        grad[layout.log_phi_denom_idx] = ratiod::math::log_prior_gamma_score(
            phi_denom,data.phi_prior_shape,data.phi_prior_rate);
    }
}

// The inflation logits of observation i as the likelihood reads them (the
// inflation link applied to the linear predictor), with their slopes in that
// predictor.
struct InflationLogits {
    double zi = 0.0, zi_slope = 1.0, oi = 0.0, oi_slope = 1.0;
};

static inline InflationLogits inflation_logits(const ModelData& data,
                                               const ParamLayout& layout,
                                               const double* params, int i) {
    InflationLogits out;
    if (layout.has_zi && data.p_zi > 0) {
        double lin = 0.0;
        for (int j = 0; j < data.p_zi; j++)
            lin += data.X_zi_flat[i * data.p_zi + j] * params[layout.beta_zi_start + j];
        const ratiod_link::ValueDeriv vd = ratiod_link::prob_logit_vd(lin, data.link_zi);
        out.zi = vd.value;
        out.zi_slope = vd.deriv;
    }
    if (layout.has_oi && data.p_oi > 0) {
        double lin = 0.0;
        for (int j = 0; j < data.p_oi; j++)
            lin += data.X_oi_flat[i * data.p_oi + j] * params[layout.beta_oi_start + j];
        const ratiod_link::ValueDeriv vd = ratiod_link::prob_logit_vd(lin, data.link_oi);
        out.oi = vd.value;
        out.oi_slope = vd.deriv;
    }
    return out;
}

// N(0, sd^2) priors on the inflation coefficients, assigned into grad.
static inline void inflation_prior_grad(const ModelData& data,
                                        const ParamLayout& layout,
                                        const double* params, double* grad) {
    if (layout.has_zi && data.p_zi > 0) {
        const double tau = 1.0 / (data.zi_prior_sd * data.zi_prior_sd + 1e-10);
        for (int j = 0; j < data.p_zi; j++)
            grad[layout.beta_zi_start + j] = -tau * params[layout.beta_zi_start + j];
    }
    if (layout.has_oi && data.p_oi > 0) {
        const double tau = 1.0 / (data.oi_prior_sd * data.oi_prior_sd + 1e-10);
        for (int j = 0; j < data.p_oi; j++)
            grad[layout.beta_oi_start + j] = -tau * params[layout.beta_oi_start + j];
    }
}

// The likelihood at observation i and its derivatives, from the log
// posterior's own per-observation density (ratiod::ratio_obs_derivs): the
// residuals in the two predictors and d/d(phi) of the two dispersions are
// returned, and the inflation coefficients' gradient is added to grad.
static inline ratiod::ObsDerivs obs_likelihood_derivs(
    const ModelData& data, const ParamLayout& layout, const double* params,
    int i, double eta_num, double eta_denom, double phi_num, double phi_denom,
    double* grad) {
    const InflationLogits inf = inflation_logits(data, layout, params, i);
    unsigned want = 0u;
    if (layout.has_phi_num) want |= ratiod::OBS_D_PHI_NUM;
    if (layout.has_phi_denom) want |= ratiod::OBS_D_PHI_DENOM;
    if (layout.has_zi && data.p_zi > 0) want |= ratiod::OBS_D_ZI;
    if (layout.has_oi && data.p_oi > 0) want |= ratiod::OBS_D_OI;
    const ratiod::ObsDerivs od = ratiod::ratio_obs_derivs(
        data, layout, i, eta_num, eta_denom, phi_num, phi_denom, inf.zi, inf.oi,
        want);
    if (want & ratiod::OBS_D_ZI) {
        const double g = od.logit_zi * inf.zi_slope;
        for (int j = 0; j < data.p_zi; j++)
            grad[layout.beta_zi_start + j] += g * data.X_zi_flat[i * data.p_zi + j];
    }
    if (want & ratiod::OBS_D_OI) {
        const double g = od.logit_oi * inf.oi_slope;
        for (int j = 0; j < data.p_oi; j++)
            grad[layout.beta_oi_start + j] += g * data.X_oi_flat[i * data.p_oi + j];
    }
    return od;
}

// Scatter residuals to beta gradient slots
static inline void scatter_beta_gradients(
    const ModelData& data, const ParamLayout& layout,
    int i, double dLL_deta_num, double dLL_deta_denom,
    double* grad
) {
    for (int j = 0; j < data.p_num; j++) {
        grad[layout.beta_num_start + j] += dLL_deta_num * data.X_num_flat[i * data.p_num + j];
    }
    for (int j = 0; j < data.p_denom; j++) {
        grad[layout.beta_denom_start + j] += dLL_deta_denom * data.X_denom_flat[i * data.p_denom + j];
    }
}

// Scatter residuals to RE gradient slot
static inline void scatter_re_gradient(
    const ModelData& data, const ParamLayout& layout,
    int i, double dLL_deta_num, double dLL_deta_denom,
    double* grad
) {
    if (layout.has_re && data.re_group[i] > 0) {
        int g = data.re_group[i] - 1;
        grad[layout.re_start + g] += dLL_deta_num + dLL_deta_denom;
    }
}

// PC prior on the temporal SD, differentiated on log precision.
static inline void tau_temporal_prior_grad(
    const ModelData& data, const ParamLayout& layout,
    double tau_temporal, double* grad
) {
    grad[layout.log_tau_temporal_idx] = ratiod::math::pc_scale_gradient(1.0 / std::sqrt(tau_temporal), -0.5, data.temporal_sigma_prior_U, data.temporal_sigma_prior_alpha);
}

// The temporal AR1 correlation at this parameter vector, on (-1, 1). A model
// without one reads 0, which leaves every term depending on it inert.
static inline double read_temporal_rho_ar1(
    const std::vector<double>& params, const ModelData& data,
    const ParamLayout& layout
) {
    if (data.temporal_type == TemporalType::AR1 && layout.logit_rho_ar1_idx >= 0)
        return ratiod_ar1::rho_from_logit(params[layout.logit_rho_ar1_idx]);
    return 0.0;
}

// Its Beta prior's gradient in the sampled coordinate. Assigns rather than
// accumulates, like tau_temporal_prior_grad beside it, so it runs before
// temporal_gmrf_prior_grad's +=.
static inline void temporal_rho_prior_grad(
    const ModelData& data, const ParamLayout& layout,
    double rho_ar1, double* grad
) {
    if (data.temporal_type == TemporalType::AR1 && layout.logit_rho_ar1_idx >= 0)
        grad[layout.logit_rho_ar1_idx] = ratiod_ar1::log_prior_rho_grad(
            rho_ar1, data.temporal_rho_prior_a, data.temporal_rho_prior_b);
}

// =====================================================================
// Temporal prior gradient helper (RW1 / RW2 / AR1 / IID)
// Shared by all gradient functions that include temporal effects.
// Writes the sampled block's gradients, the tau gradient, and the rho
// gradient (for AR1).
//
// `tview` carries the effects (`phi`, from temporal_effects()), the block the
// sampler moves in (`z`), and an intrinsic walk's pre-centring sum.
// `grad_temporal_lik[0..T_len-1]` holds the likelihood contributions with
// respect to the effects, and the gradient written out is always with respect
// to the sampled block.
//
// It identifies an intrinsic walk the way compute_log_post does -- centring
// plus the augmented constant direction, see tulpa/sum_to_zero.h -- so that the
// gradient and the density it is returned alongside describe the same target.
// =====================================================================
static inline void temporal_gmrf_prior_grad(
    const ModelData& data, const ParamLayout& layout,
    double tau_temporal, double rho_ar1,
    const TemporalView& tview, int T_len,
    const double* grad_temporal_lik,
    double* grad
) {
    const double* phi_temporal = tview.phi;
    const double* z_temporal = tview.z;
    int T = data.n_times;
    int n_groups = data.n_temporal_groups;

    if (temporal_ar1_nc(data, layout)) {
        // Non-centred AR1: the chain rule through phi = L(tau, rho) z carries
        // the likelihood to z and to both hyperparameters, and seeds z's own
        // N(0, I) prior, so nothing here initializes the block first.
        double grad_log_tau = 0.0, grad_logit_rho = 0.0;
        for (int gg = 0; gg < n_groups; gg++) {
            const int base = layout.temporal_start + gg * T;
            std::pair<double, double> hyper = ratiod_temporal::ar1_nc_gradient(
                z_temporal + gg * T, phi_temporal + gg * T,
                grad_temporal_lik + gg * T, &grad[base],
                T, rho_ar1, tau_temporal);
            grad_log_tau += hyper.first;
            grad_logit_rho += hyper.second;
        }
        grad[layout.log_tau_temporal_idx] += grad_log_tau;
        if (layout.logit_rho_ar1_idx >= 0) {
            grad[layout.logit_rho_ar1_idx] += grad_logit_rho;
        }
        return;
    }

    if (temporal_rw_nc(data, layout)) {
        // Non-centred walk: eta reads the centred effects, so the likelihood
        // gradient is projected as in the centred coordinate and then carried
        // through phi = sigma * A^{-1} z. tau reaches eta only through that
        // transform, and the arm's own prior in this coordinate is N(0, I) on
        // the scaled directions and flat on the rest.
        const int order = (data.temporal_type == TemporalType::RW2) ? 2 : 1;
        const double sigma = 1.0 / std::sqrt(tau_temporal);
        std::vector<double> g_proj(grad_temporal_lik,
                                   grad_temporal_lik + T_len);
        tulpa::s2z_centre_blocks(g_proj.data(), n_groups, T);

        std::vector<double> gz(T_len, 0.0);
        double g_log_sigma2 = 0.0;
        ratiod_temporal_nc::rw_nc_grouped_backward(
            g_proj.data(), z_temporal, data.n_times, data.n_temporal_groups,
            order, data.temporal_cyclic, sigma, gz.data(), &g_log_sigma2);
        ratiod_temporal_nc::rw_nc_grouped_log_prior_grad(
            z_temporal, data.n_times, data.n_temporal_groups, order,
            data.temporal_cyclic, gz.data());

        for (int t = 0; t < T_len; t++) {
            grad[layout.temporal_start + t] = gz[t];
        }
        // sigma2 = 1 / tau, so d/d(log tau) = -d/d(log sigma2).
        grad[layout.log_tau_temporal_idx] += -g_log_sigma2;
        return;
    }

    // Initialize temporal gradients with likelihood contribution. An intrinsic
    // walk enters eta centred group by group, so eta depends on each group's
    // block only through phi_g - mean(phi_g) and the gradient with respect to
    // that block is the likelihood gradient with the group's mean removed.
    for (int t = 0; t < T_len; t++) {
        grad[layout.temporal_start + t] = grad_temporal_lik[t];
    }
    if (tview.centred) {
        tulpa::s2z_centre_blocks(grad + layout.temporal_start, n_groups, T);
    }

    // Each group's freed constant direction: its gradient, the same for every
    // entry of the group, and the quadratic form summed over the groups.
    double aug_quad = 0.0;
    if (tview.centred) {
        for (int gg = 0; gg < n_groups; gg++) {
            const double s = tview.raw_sums[gg];
            const double g_aug =
                ratiod_constraints::free_direction_grad(s, T, tau_temporal);
            for (int t = 0; t < T; t++) grad[layout.temporal_start + gg * T + t] += g_aug;
            aug_quad += ratiod_constraints::free_direction_quad(s, T);
        }
    }
    const int n_aug_pins = tview.centred ? n_groups : 0;

    if (data.temporal_type == TemporalType::RW1) {
        double total_qf = 0.0;
        int total_rank = 0;
        for (int gg = 0; gg < n_groups; gg++) {
            const double* phi_g = phi_temporal + gg * T;
            int base = layout.temporal_start + gg * T;
            double qf = 0.0;
            for (int t = 0; t < T; t++) {
                double g = 0.0;
                if (t > 0) {
                    g += tau_temporal * (phi_g[t - 1] - phi_g[t]);
                    qf += (phi_g[t] - phi_g[t - 1]) * (phi_g[t] - phi_g[t - 1]);
                }
                if (t < T - 1) g += tau_temporal * (phi_g[t + 1] - phi_g[t]);
                grad[base + t] += g;
            }
            if (data.temporal_cyclic) {
                double dc = phi_g[0] - phi_g[T - 1];
                qf += dc * dc;
                grad[base + 0] -= tau_temporal * dc;
                grad[base + T - 1] += tau_temporal * dc;
            }
            total_qf += qf;
            total_rank += tulpa::rw1_rank(T, data.temporal_cyclic);
        }
        total_rank = tulpa::s2z_aug_rank(total_rank, n_aug_pins);
        grad[layout.log_tau_temporal_idx] +=
            0.5 * total_rank - 0.5 * tau_temporal * (total_qf + aug_quad);

    } else if (data.temporal_type == TemporalType::RW2) {
        double total_qf = 0.0;
        int total_rank = 0;
        for (int gg = 0; gg < n_groups; gg++) {
            const double* phi_g = phi_temporal + gg * T;
            int base = layout.temporal_start + gg * T;
            double qf = 0.0;
            for (int t = 0; t < T; t++) {
                double g = 0.0;
                if (t >= 2) g -= tau_temporal * (phi_g[t - 2] - 2.0 * phi_g[t - 1] + phi_g[t]);
                if (t >= 1 && t < T - 1) g += 2.0 * tau_temporal * (phi_g[t - 1] - 2.0 * phi_g[t] + phi_g[t + 1]);
                if (t < T - 2) g -= tau_temporal * (phi_g[t] - 2.0 * phi_g[t + 1] + phi_g[t + 2]);
                grad[base + t] += g;
            }
            for (int t = 2; t < T; t++) {
                double d2 = phi_g[t - 2] - 2.0 * phi_g[t - 1] + phi_g[t];
                qf += d2 * d2;
            }
            if (data.temporal_cyclic && T >= 3) {
                double d2_a = phi_g[T - 2] - 2.0 * phi_g[T - 1] + phi_g[0];
                double d2_b = phi_g[T - 1] - 2.0 * phi_g[0] + phi_g[1];
                qf += d2_a * d2_a + d2_b * d2_b;
                grad[base + T - 2] -= tau_temporal * d2_a;
                grad[base + T - 1] += 2.0 * tau_temporal * d2_a;
                grad[base + 0] -= tau_temporal * d2_a;
                grad[base + T - 1] -= tau_temporal * d2_b;
                grad[base + 0] += 2.0 * tau_temporal * d2_b;
                grad[base + 1] -= tau_temporal * d2_b;
            }
            total_qf += qf;
            total_rank += tulpa::rw2_rank(T, data.temporal_cyclic);
        }
        // A non-cyclic RW2 keeps its per-group LINEAR null direction, which the
        // augmentation does not touch; s2z_aug_rank adds only the constants.
        total_rank = tulpa::s2z_aug_rank(total_rank, n_aug_pins);
        grad[layout.log_tau_temporal_idx] +=
            0.5 * total_rank - 0.5 * tau_temporal * (total_qf + aug_quad);

    } else if (data.temporal_type == TemporalType::AR1) {
        double omr2 = ratiod_ar1::one_minus_rho2(rho_ar1);
        double total_qf = 0.0, total_gr = 0.0;
        for (int gg = 0; gg < n_groups; gg++) {
            const double* phi_g = phi_temporal + gg * T;
            int base = layout.temporal_start + gg * T;
            grad[base] += -tau_temporal * omr2 * phi_g[0];
            if (T > 1) grad[base] += tau_temporal * rho_ar1 * (phi_g[1] - rho_ar1 * phi_g[0]);
            double qf = omr2 * phi_g[0] * phi_g[0];
            for (int t = 1; t < T; t++) {
                double r = phi_g[t] - rho_ar1 * phi_g[t - 1];
                qf += r * r;
                double g = -tau_temporal * r;
                if (t < T - 1) g += tau_temporal * rho_ar1 * (phi_g[t + 1] - rho_ar1 * phi_g[t]);
                grad[base + t] += g;
            }
            total_qf += qf;
            total_gr += tau_temporal * rho_ar1 * phi_g[0] * phi_g[0];
            for (int t = 1; t < T; t++) {
                total_gr += tau_temporal * (phi_g[t] - rho_ar1 * phi_g[t - 1]) * phi_g[t - 1];
            }
        }
        grad[layout.log_tau_temporal_idx] += 0.5 * T_len - 0.5 * tau_temporal * total_qf;
        if (layout.logit_rho_ar1_idx >= 0) {
            double gr = -n_groups * rho_ar1 / omr2 + total_gr;
            grad[layout.logit_rho_ar1_idx] += gr * ratiod_ar1::drho_dlogit(rho_ar1);
        }

    } else if (data.temporal_type == TemporalType::IID) {
        double total_qf = 0.0;
        for (int gg = 0; gg < n_groups; gg++) {
            const double* phi_g = phi_temporal + gg * T;
            int base = layout.temporal_start + gg * T;
            for (int t = 0; t < T; t++) {
                grad[base + t] += -tau_temporal * phi_g[t];
                total_qf += phi_g[t] * phi_g[t];
            }
        }
        grad[layout.log_tau_temporal_idx] += 0.5 * T_len - 0.5 * tau_temporal * total_qf;
    }
}

// =====================================================================
// PC prior gradient on log(sigma2) for GP variances
// Returns d log_prior / d log_sigma2 INCLUDING Jacobian for exp transform.
// Formula: -0.5 * rate * sigma + 0.5  where rate = -log(alpha) / U
// =====================================================================



// =====================================================================
// Spatial ICAR/BYM2 GMRF prior gradient.
//
// The single source of truth for the intrinsic spatial field's gradient, used
// by every gradient path that carries one. It identifies the field the way
// compute_log_post does -- hard centring plus the freed constant direction,
// see spatial_field_constraint.h -- so that the gradient and the density it is
// returned alongside describe the same target.
//
// phi_spatial must be the centred field and phi_raw_sums its pre-centring
// sums per block, both as produced by CenteredSpatialParams. grad_spatial_lik holds the
// accumulated likelihood gradient with respect to the phi *parameters*, so for
// BYM2 the sigma_s * scale chain factor is already applied by the caller's
// scatter. theta_bym2 gradients stay with the caller, which sets the prior term
// and accumulates the likelihood term during its own scatter.
// =====================================================================
// The multi-feature gradient paths scatter the spatial field's likelihood
// gradient in eta units (a bare dLL per observation). spatial_gmrf_prior_grad
// works in phi-parameter units, which for BYM2 differ by the sigma_s * scale
// chain factor; the vectorized main path applies that during its own scatter.
// Settles the BYM2 theta block at the same time, which stays with the caller
// because its likelihood term is scattered per observation.
static inline void spatial_lik_grad_to_param_scale(
    const ModelData& data, const ParamLayout& layout,
    double sigma_s_bym2, double sigma_u_bym2,
    const double* theta_bym2, const double* grad_theta_lik,
    double* grad_spatial_lik, double* grad
) {
    if (data.spatial_type != SpatialType::BYM2) return;
    const int S = data.n_spatial_units;
    const double chain = sigma_s_bym2 * data.bym2_scale_factor;
    for (int s = 0; s < S; s++) {
        grad_spatial_lik[s] *= chain;
        grad[layout.theta_bym2_start + s] =
            grad_theta_lik[s] * sigma_u_bym2 - theta_bym2[s];
    }
}

static inline void spatial_gmrf_prior_grad(
    const ModelData& data, const ParamLayout& layout,
    const double* phi_spatial,
    const double* phi_spatial_prior,
    const std::vector<double>& phi_raw_sums,
    bool centered,
    double tau_spatial,
    double sigma_s_bym2, double sigma_u_bym2,
    double rho_bym2,
    double rho_car,
    const double* theta_bym2,
    const double* grad_spatial_lik,
    double* grad
) {
    const int S = data.n_spatial_units;
    const bool is_bym2 = (data.spatial_type == SpatialType::BYM2);
    const bool is_car_proper = (data.spatial_type == SpatialType::CAR_PROPER);

    // The likelihood sees each block centred, so its gradient with respect to
    // the raw parameters is the projection removing each block's mean.
    // Applied here only: d(eta)/d(sigma_bym2) is a function of the centred
    // field with no such projection, so grad_spatial_lik itself must stay
    // unprojected.
    for (int s = 0; s < S; s++) {
        grad[layout.spatial_start + s] = grad_spatial_lik[s];
    }
    if (centered) {
        ratiod_constraints::project_areal_gradient(&grad[layout.spatial_start], S,
                                                   data.areal_components);
    }

    // ICAR prior: -0.5 * tau * phi' * Q * phi, Q the ICAR precision with the
    // per-unit multipliers and island diagonal of ratiod::icar_precision_row.
    // Proper CAR: Q(rho)_ij = n_neighbors[i] if i=j, -rho if i~j; no freed direction.
    // Gradient of each block's freed constant direction; see
    // spatial_field_constraint.h.
    const double free_scale = is_bym2 ? 1.0 : tau_spatial;
    for (int i = 0; i < S; i++) {
        double Qphi_i;
        if (is_car_proper) {
            Qphi_i = data.n_neighbors[i] * phi_spatial_prior[i];
            for (int k = data.adj_row_ptr[i]; k < data.adj_row_ptr[i + 1]; k++) {
                Qphi_i -= rho_car * phi_spatial_prior[data.adj_col_idx[k]];
            }
        } else {
            Qphi_i = ratiod::icar_precision_times(phi_spatial_prior, i, data);
        }
        const double free_grad = is_car_proper ? 0.0 :
            ratiod_constraints::areal_free_direction_grad(phi_raw_sums, S,
                                                          data.areal_components,
                                                          i, free_scale);

        if (is_bym2) {
            // For BYM2, ICAR prior has no tau scaling (it's absorbed into sigma/rho)
            grad[layout.spatial_start + i] += -Qphi_i + free_grad;
        } else {
            grad[layout.spatial_start + i] += -tau_spatial * Qphi_i + free_grad;
        }
    }

    if (is_bym2) {
        // BYM2 Riebler: transform (grad_sigma_s, grad_sigma_u) to (grad_log_sigma, grad_logit_rho)
        // grad_sigma_s_lik = d(LL)/d(sigma_s) * sigma_s  (chain rule for log)
        // grad_sigma_u_lik = d(LL)/d(sigma_u) * sigma_u
        double grad_sigma_s_lik = 0.0;
        double grad_sigma_u_lik = 0.0;
        for (int s = 0; s < S; s++) {
            const double scaled_phi = phi_spatial[s] * data.bym2_scale_factor;
            const double d_LL_d_spatial =
                grad_spatial_lik[s] / (sigma_s_bym2 * data.bym2_scale_factor);
            grad_sigma_s_lik += d_LL_d_spatial * sigma_s_bym2 * scaled_phi;
            grad_sigma_u_lik += d_LL_d_spatial * sigma_u_bym2 * theta_bym2[s];
        }
        grad[layout.log_sigma_bym2_idx] += grad_sigma_s_lik + grad_sigma_u_lik;
        grad[layout.logit_rho_bym2_idx] += 0.5 * ((1.0 - rho_bym2) * grad_sigma_s_lik
                                                    - rho_bym2 * grad_sigma_u_lik);
    } else if (is_car_proper) {
        // tau gradient: same shape as plain ICAR, but the quadratic form is
        // rho-weighted (car_quadratic_form), not the rho=1 special case above.
        const double quad = ratiod_car_proper::car_quadratic_form(
            phi_spatial_prior, S, data.adj_row_ptr, data.adj_col_idx,
            data.n_neighbors, rho_car);
        grad[layout.log_tau_spatial_idx] += 0.5 * S - 0.5 * tau_spatial * quad;

        // rho gradient: 0.5*log|Q(rho)| has no closed form cheaper than a
        // dense Cholesky (trace(Q^-1 W) needs the inverse), so it is central-
        // differenced here, on this one scalar, and chain-ruled to logit
        // scale. The quadratic-form part is exact (it is linear in rho).
        auto gmrf_term = [&](double rho) {
            auto Q = ratiod_car_proper::compute_car_precision(
                S, data.adj_row_ptr, data.adj_col_idx, data.n_neighbors, rho);
            const double logdet = ratiod_car_proper::car_log_det(S, Q);
            const double q = ratiod_car_proper::car_quadratic_form(
                phi_spatial_prior, S, data.adj_row_ptr, data.adj_col_idx,
                data.n_neighbors, rho);
            return 0.5 * logdet - 0.5 * tau_spatial * q;
        };
        const double eps = 1e-5;
        const double rho_p = std::min(rho_car + eps, data.car_rho_upper - 1e-9);
        const double rho_m = std::max(rho_car - eps, data.car_rho_lower + 1e-9);
        const double d_gmrf_d_rho = (gmrf_term(rho_p) - gmrf_term(rho_m)) / (rho_p - rho_m);
        const double u = (rho_car - data.car_rho_lower) / (data.car_rho_upper - data.car_rho_lower);
        const double d_rho_d_logit = (data.car_rho_upper - data.car_rho_lower) * u * (1.0 - u);
        grad[layout.logit_rho_car_idx] += d_gmrf_d_rho * d_rho_d_logit;
    } else {
        // Plain ICAR: tau gradient. (Q + 11'/J) is full rank, so the exponent on
        // tau is n/2 and the quadratic form carries the freed direction.
        // log_post = 0.5*n*log(tau) - 0.5*tau*(quad + sum^2/n) + const
        const double quad_full =
            ratiod::icar_quadratic_form_t(phi_spatial_prior, S, data)
            + ratiod_constraints::areal_free_direction_quad(phi_raw_sums, S,
                                                            data.areal_components);
        grad[layout.log_tau_spatial_idx] += 0.5 * S - 0.5 * tau_spatial * quad_full;
    }
}

// =====================================================================
// GP gradient (hand-coded, ~3x faster than autodiff)
// Uses analytical gradients from gp_nngp_gradients for NNGP prior

static inline void fill_fixed_eta(vectorized::VecGradWorkspace& ws,
                                  const ModelData& data,
                                  const double* beta_num, const double* beta_denom) {
  const int N = data.N;
  ws.init(N);
  using RowMajorMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  using VectorXd = Eigen::VectorXd;
  Eigen::Map<const RowMajorMatrix> X_num(data.X_num_flat.data(), N, data.p_num);
  Eigen::Map<const VectorXd> b_num(beta_num, data.p_num);
  Eigen::Map<VectorXd> eta_n(ws.eta_num.data(), N);
  eta_n.noalias() = X_num * b_num;
  const bool is_binomial = data.model_type == ModelType::BINOMIAL ||
                           data.model_type == ModelType::BETA_BINOMIAL;
  if (!is_binomial && data.p_denom > 0) {
    Eigen::Map<const RowMajorMatrix> X_denom(data.X_denom_flat.data(), N, data.p_denom);
    Eigen::Map<const VectorXd> b_denom(beta_denom, data.p_denom);
    Eigen::Map<VectorXd> eta_d(ws.eta_denom.data(), N);
    eta_d.noalias() = X_denom * b_denom;
  } else if (!is_binomial) {
    std::memset(ws.eta_denom.data(), 0, N * sizeof(double));
  }
}

static inline void svc_range_grads_to_coord(const std::vector<double>& params,
                                            const ModelData& data,
                                            const ParamLayout& layout,
                                            std::vector<double>& grad) {
    for (int k = layout.range_svc_start; k < layout.range_svc_end; k++) {
        grad[k] = ratiod_svc_ad::svc_range_grad_to_coord(
            grad[k], params[k], data.svc_phi_prior_lower, data.svc_phi_prior_upper);
    }
}

}  // namespace ratiod_hmc
#endif
