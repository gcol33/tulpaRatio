// log_post_impl.h
// Templated compute_log_post that works with both double and ad::Var
// Single implementation for both evaluation and autodiff gradients
//
// NOTE: This header must be included AFTER hmc_sampler.h which defines
// ModelData, ParamLayout, ModelType, TemporalType

#ifndef RATIOD_LOG_POST_IMPL_H
#define RATIOD_LOG_POST_IMPL_H

#include <vector>
#include <limits>
#include <type_traits>
#include "autodiff_utils.h"
#include "ar1_shared.h"  // AR1 rho: the floored 1 - rho^2 and its prior
#include <tulpa/soft_sum_to_zero.h>
#include "spatial_field_constraint.h"
#include "hmc_re.h"  // Templated random-effect prior and effects
#include "hmc_spatiotemporal.h"  // Templated spatiotemporal interaction priors
#include "hmc_hsgp.h"  // Templated HSGP spectral density
#include "hmc_gp_autodiff.h"  // Templated GP/NNGP functions
#include "hmc_svc_autodiff.h"  // Templated SVC functions
#include "hmc_tvc_autodiff.h"  // Templated TVC functions
#include "hmc_latent.h"
#include "hmc_temporal_multiscale.h"  // Templated multiscale temporal functions
#include "tls_workspace.h"
#include "omp_sum.h"
#include "rsr.h"
#include "hmc_gp_collapsed.h"     // Collapsed GP marginal (double only)
#include "hmc_icar_collapsed.h"   // Collapsed ICAR/BYM2 marginal (double only)
#include "hmc_car_proper.h"       // Proper CAR precision and log-determinant
#include "ratio_obs_lik.h"         // One observation's log-likelihood
#include "st_gp_laplace.h"         // Collapsed GP interaction (double only)

// Expects these to be defined by including hmc_sampler.h first:
// - ratiod_hmc::ModelData
// - ratiod_hmc::ParamLayout
// - ratiod_hmc::ModelType
// - ratiod_hmc::TemporalType

using ratiod_hmc::ModelData;
using ratiod_hmc::ParamLayout;
using ratiod_hmc::ModelType;
using ratiod_hmc::TemporalType;
using ratiod_hmc::STType;
using ratiod_zi::ZIType;

namespace ratiod {

using namespace math;

// Structures this density does not express, named so a caller on the R thread
// can refuse rather than sample the wrong posterior.
//
// The arena / forward / tape modes differentiate this density, and for those
// modes verify_gradient_runtime differences this same density, so a structure
// missing here is invisible to both: the gradient and its numerical reference
// agree on the same wrong value. The collapsed parameterizations cannot be
// expressed here at all -- they marginalize the field by locating its mode with
// Newton and adding a Laplace correction, which is not a closed-form function of
// the parameters -- so they are reported instead.
inline const char* log_post_impl_gap(const ModelData& data,
                                     const ParamLayout& layout) {
    if (layout.is_icar_collapsed) return "collapsed ICAR";
    if (layout.is_bym2_collapsed) return "collapsed BYM2";
    if (data.gp_collapsed) return "collapsed GP";
    if (layout.st_collapsed) return "collapsed GP interaction";
    // Proper CAR's log-determinant needs a dense Cholesky; not written here,
    // and not verified differentiable through autodiff either.
    if (layout.is_car_proper) return "proper CAR";
    return nullptr;
}

// phi' Q phi for the ICAR precision Q = D - W over the CSR adjacency, for both
// arms of the spatial prior and for the analytic gradient paths.
template<typename T>
inline T icar_quadratic_form_t(const T* phi, int J, const ModelData& data) {
    // Q_ii = c_i n_i (c_i for an island), Q_ij = -c_i for i ~ j, with c_i the
    // unit's precision multiplier (one on a connected graph; constant within
    // a component, so Q stays symmetric)
    const bool weighted = !data.areal_node_prec.empty();
    T quad_form = T(0.0);
    for (int i = 0; i < J; i++) {
        const double c = weighted ? data.areal_node_prec[i] : 1.0;
        const int n_i = data.n_neighbors[i];
        const double diag = (n_i > 0 ? n_i : (weighted ? 1 : 0)) * c;
        quad_form = quad_form + T(diag) * phi[i] * phi[i];
        const int row_start = data.adj_row_ptr[i];
        const int row_end = data.adj_row_ptr[i + 1];
        for (int k = row_start; k < row_end; k++) {
            const int j = data.adj_col_idx[k];
            if (j > i) {
                quad_form = quad_form - T(2.0 * c) * phi[i] * phi[j];
            }
        }
    }
    return quad_form;
}

// (Q phi)_i for the ICAR precision of icar_quadratic_form_t
inline double icar_precision_times(const double* phi, int i, const ModelData& data) {
    const bool weighted = !data.areal_node_prec.empty();
    const double c = weighted ? data.areal_node_prec[i] : 1.0;
    const int n_i = data.n_neighbors[i];
    double out = (n_i > 0 ? n_i : (weighted ? 1 : 0)) * c * phi[i];
    for (int k = data.adj_row_ptr[i]; k < data.adj_row_ptr[i + 1]; k++) {
        out -= c * phi[data.adj_col_idx[k]];
    }
    return out;
}

// What a caller wants of the density that is not the density. A fused gradient
// pass has already walked the observations, and for the non-centred temporal GP
// has already accumulated that block's prior, so it asks for the rest rather
// than paying for either twice.
struct LogPostOptions {
    bool skip_obs_loop = false;
    const double* precomputed_tgp_log_prior = nullptr;
    // The centred spatial GP's NNGP log-density, from a gradient pass that
    // computed it. Only passed alongside skip_obs_loop, so the field itself
    // is not needed either.
    const double* precomputed_gp_log_prior = nullptr;
    // The centred SVC terms' summed NNGP log-density, likewise.
    const double* precomputed_svc_log_prior = nullptr;
};

// ============================================================================
// Log-posterior computation
// T = double for evaluation, T = ad::Var / fwd::Dual / arena::Var for gradients
//
// The double instantiation is the whole of compute_log_post: the structures no
// autodiff scalar can express (the collapsed marginals, proper CAR's dense
// log-determinant), the parallel observation loop and the per-thread
// workspaces sit behind `if constexpr`, and log_post_impl_gap names what the
// other instantiations therefore leave out.
// ============================================================================

template<typename T>
T compute_log_post_impl(
    const std::vector<T>& params_in,
    const ModelData& data,
    const ParamLayout& layout,
    const LogPostOptions& opts = LogPostOptions()
) {
    constexpr bool kDouble = std::is_same<T, double>::value;

    // Hard sum-to-zero: centre the spatial block once, matching the analytic
    // compute_log_post. See spatial_field_constraint.h.
    std::vector<T> params_centered;
    std::vector<T> phi_spatial_raw_sums;
    const bool center_spatial =
        ratiod_hmc::spatial_block_is_centred(data, layout);
    if (center_spatial) {
        params_centered = params_in;
        phi_spatial_raw_sums = ratiod_constraints::center_areal_blocks(
            params_centered, layout.spatial_start, data.n_spatial_units,
            data.areal_components);
    }
    const std::vector<T>& params = center_spatial ? params_centered : params_in;

    // A collapsed field is marginalized out and has no slot in the parameter
    // vector, so nothing below may index it.
    const bool spatial_in_params = layout.has_spatial && layout.spatial_start >= 0;

    T log_post = T(0.0);

    // ========================================================================
    // Extract parameters
    // ========================================================================

    // Fixed effects — pointer views into params (no copy, read-only access)
    const T* beta_num = &params[layout.beta_num_start];
    const T* beta_denom = &params[layout.beta_denom_start];

    // Random effects: prior and effective effects together, from the one
    // implementation both densities call. See hmc_re.h.
    ratiod_re::ReEffects<T> re_eff;
    {
        bool re_ok = true;
        log_post = log_post
            + ratiod_re::re_log_prior(params, data, layout, re_eff, re_ok);
        if (!re_ok) return T(-INFINITY);
    }

    // Overdispersion (phi for Gamma denominator in poisson_gamma)
    T phi_num = T(1.0);
    if (layout.has_phi_num) {
        T log_phi = params[layout.log_phi_num_idx];
        phi_num = safe_exp(log_phi);
    }

    T phi_denom = T(1.0);
    if (layout.has_phi_denom) {
        T log_phi = params[layout.log_phi_denom_idx];
        phi_denom = safe_exp(log_phi);
    }

    // Spatial effects — pointer views into params (no copy, read-only access).
    // phi_spatial is what eta reads; phi_spatial_prior is what the prior reads,
    // and the two differ only where the field is centred and its precision does
    // not annihilate the constant (spatial_prior_reads_raw).
    const T* phi_spatial = nullptr;
    const T* phi_spatial_prior = nullptr;
    T tau_spatial = T(1.0);
    T sigma_s_bym2 = T(1.0);
    T sigma_u_bym2 = T(1.0);
    const T* theta_bym2 = nullptr;

    // The scale parameters are read whether or not the field has a slot: a
    // marginalized field still carries its own prior on them.
    if (layout.has_spatial) {
        if (layout.is_bym2) {
            // Riebler reparameterization: sigma_total, rho -> sigma_s, sigma_u
            T sigma_total_bym2 = safe_exp(params[layout.log_sigma_bym2_idx]);
            T logit_rho_val = params[layout.logit_rho_bym2_idx];
            T rho_bym2 = T(1.0) / (T(1.0) + safe_exp(-logit_rho_val));
            sigma_s_bym2 = sigma_total_bym2 * sqrt(rho_bym2);
            sigma_u_bym2 = sigma_total_bym2 * sqrt(T(1.0) - rho_bym2);

            if (spatial_in_params) {
                phi_spatial = &params[layout.spatial_start];
                theta_bym2 = &params[layout.theta_bym2_start];
            }
        } else {
            T log_tau = params[layout.log_tau_spatial_idx];
            tau_spatial = safe_exp(log_tau);

            if (spatial_in_params) {
                phi_spatial = &params[layout.spatial_start];
            }
        }
        if (spatial_in_params) {
            phi_spatial_prior =
                ratiod_hmc::spatial_prior_reads_raw(data, layout)
                    ? &params_in[layout.spatial_start]
                    : phi_spatial;
        }
    }

    // Collapsed ICAR/BYM2: the field is marginalized out, so its mode is found
    // by an inner Newton and a Laplace correction added, and phi_spatial points
    // at that mode rather than at a parameter slot. Neither is a closed-form
    // function of the parameters, so only the double instantiation carries it;
    // log_post_impl_gap names the structure for the others.
    if (data.icar_collapsed || data.bym2_collapsed) {
        if constexpr (kDouble) {
            // Actual RE values for the mode finder (non-centred -> actual)
            std::vector<double> re_vals_collapsed;
            if (layout.has_re) {
                const int n_re = layout.re_end - layout.re_start;
                const double sigma_re = std::exp(params[layout.log_sigma_re_idx]);
                const double* re_ptr = &params[layout.re_start];
                re_vals_collapsed.resize(n_re);
                for (int g = 0; g < n_re; g++) {
                    re_vals_collapsed[g] = (data.re_parameterization == 1)
                        ? sigma_re * re_ptr[g] : re_ptr[g];
                }
            }

            CollapsedTemporalOffset collapsed_temporal_offset =
                collapsed_temporal_obs_offset(data, layout, params);

            auto cres = collapsed_icar_log_post_contribution(
                data.bym2_collapsed, tau_spatial,
                data.bym2_collapsed ? std::exp(params[layout.log_sigma_bym2_idx]) : 0.0,
                data.bym2_collapsed ? params[layout.logit_rho_bym2_idx] : 0.0,
                data.bym2_scale_factor, phi_num, phi_denom,
                &params[layout.beta_num_start], &params[layout.beta_denom_start],
                re_vals_collapsed.empty() ? nullptr : re_vals_collapsed.data(),
                data, collapsed_icar_ws(),
                collapsed_temporal_offset.num_ptr(),
                collapsed_temporal_offset.denom_ptr());

            phi_spatial = cres.phi_spatial;
            theta_bym2 = cres.theta_bym2;
            log_post = log_post + cres.log_post_contribution;
        }
    }

    // Temporal effects
    std::vector<T> phi_temporal;
    std::vector<T> z_temporal;  // Non-centred AR1's sampled block
    // Intrinsic walks are centred group by group on their way into eta and each
    // group's constant direction is augmented in the prior; the analytic side
    // decides the same thing in temporal_is_intrinsic().
    bool center_temporal = false;
    const bool rw_nc_temporal = ratiod_hmc::temporal_rw_nc(data, layout);
    std::vector<T> phi_temporal_mean;
    T tau_temporal = T(1.0);
    T rho_ar1 = T(0.0);
    T sigma2_temporal_gp = T(1.0);
    T phi_temporal_gp = T(1.0);  // lengthscale

    if (layout.has_temporal) {
        // Extract temporal effects (common to all temporal types)
        int n_temporal = layout.temporal_end - layout.temporal_start;
        phi_temporal.resize(n_temporal);
        for (int t = 0; t < n_temporal; t++) {
            phi_temporal[t] = params[layout.temporal_start + t];
        }

        if (layout.is_temporal_gp) {
            // Temporal GP: sigma2 and phi (lengthscale) parameters
            T log_sigma2 = params[layout.log_sigma2_temporal_gp_idx];
            T logit_phi = params[layout.logit_phi_temporal_gp_idx];
            sigma2_temporal_gp = safe_exp(log_sigma2);

            // Logit-bounded phi: phi = lower + range * sigmoid(logit_phi)
            T sigmoid_phi = inv_logit(logit_phi);
            double phi_lower_lp = data.temporal_gp_phi_prior_lower;
            double phi_range_lp = data.temporal_gp_phi_prior_upper - phi_lower_lp;
            phi_temporal_gp = T(phi_lower_lp) + T(phi_range_lp) * sigmoid_phi;
        } else {
            // RW1/RW2/AR1: tau-based parameterization
            T log_tau = params[layout.log_tau_temporal_idx];
            tau_temporal = safe_exp(log_tau);

            if (layout.is_ar1) {
                rho_ar1 = ratiod_ar1::rho_from_logit(
                    params[layout.logit_rho_ar1_idx]);
            }
        }

        // Non-centred AR1: the block holds z ~ N(0, I). Every reader below
        // wants the effects, so reconstruct them here and keep z for the
        // prior, mirroring temporal_effects() on the analytic side.
        if (data.temporal_type == TemporalType::AR1 &&
            data.temporal_parameterization == 1) {
            z_temporal = phi_temporal;
            const int T_nc = data.n_times;
            T inv_sqrt_tau = T(1.0) / safe_sqrt(tau_temporal);
            T inv_sqrt_tau_omr2 =
                inv_sqrt_tau / safe_sqrt(ratiod_ar1::one_minus_rho2(rho_ar1));
            for (int g = 0; g < data.n_temporal_groups; g++) {
                const int off = g * T_nc;
                phi_temporal[off] = z_temporal[off] * inv_sqrt_tau_omr2;
                for (int t = 1; t < T_nc; t++) {
                    phi_temporal[off + t] = rho_ar1 * phi_temporal[off + t - 1]
                                          + z_temporal[off + t] * inv_sqrt_tau;
                }
            }
        }

        // Non-centred intrinsic walk: the block holds z ~ N(0, I), so the
        // effects are reconstructed here and z kept for the prior, mirroring
        // temporal_effects() on the analytic side.
        if (rw_nc_temporal) {
            z_temporal = phi_temporal;
            const int n_rw = static_cast<int>(phi_temporal.size());
            const int order_rw =
                (data.temporal_type == TemporalType::RW2) ? 2 : 1;
            const T sigma_rw = T(1.0) / safe_sqrt(tau_temporal);
            std::vector<T> eff_rw(n_rw, T(0.0));
            ratiod_temporal_nc::rw_nc_grouped_forward(
                z_temporal.data(), data.n_times, data.n_temporal_groups,
                order_rw, data.temporal_cyclic, sigma_rw, eff_rw.data());
            phi_temporal = eff_rw;
        }

        center_temporal = !layout.is_temporal_gp && !phi_temporal.empty() &&
            (data.temporal_type == TemporalType::RW1 ||
             data.temporal_type == TemporalType::RW2);
        if (center_temporal) {
            phi_temporal_mean.resize(data.n_temporal_groups);
            for (int g = 0; g < data.n_temporal_groups; g++) {
                phi_temporal_mean[g] = tulpa::s2z_component_mean(
                    phi_temporal.data(), g * data.n_times, data.n_times);
            }
        }
    }

    // ========================================================================
    // PRIORS
    // ========================================================================

    // Fixed effects: N(0, sigma_beta^2)
    double tau_beta = 1.0 / (data.sigma_beta * data.sigma_beta);
    for (int j = 0; j < data.p_num; j++) {
        log_post = log_post + log_prior_normal(beta_num[j], tau_beta);
    }
    for (int j = 0; j < data.p_denom; j++) {
        log_post = log_post + log_prior_normal(beta_denom[j], tau_beta);
    }

    // Overdispersion: Gamma prior
    if (layout.has_phi_num) {
        T log_phi = params[layout.log_phi_num_idx];
        log_post = log_post + log_prior_gamma(log_phi, data.phi_prior_shape, data.phi_prior_rate);
    }
    if (layout.has_phi_denom) {
        T log_phi = params[layout.log_phi_denom_idx];
        log_post = log_post + log_prior_gamma(log_phi, data.phi_prior_shape, data.phi_prior_rate);
    }

    // Spatial priors. The scale parameters carry a prior whether or not the
    // field has a slot; the field's own quadratic form is inside the collapsed
    // contribution above when it does not.
    if (layout.has_spatial) {
        const int J = data.n_spatial_units;

        if (layout.is_bym2) {
            // BYM2 Riebler: Half-Cauchy on sigma_total
            T log_sigma = params[layout.log_sigma_bym2_idx];
            log_post = log_post + log_prior_half_cauchy(log_sigma, data.sigma_re_scale);

            // Uniform(0,1) = Beta(1,1) on rho with logit Jacobian:
            // log p(logit_rho) = log(rho) + log(1-rho)
            T logit_rho_val = params[layout.logit_rho_bym2_idx];
            T rho_bym2_prior = T(1.0) / (T(1.0) + safe_exp(-logit_rho_val));
            log_post = log_post + log(rho_bym2_prior)
                                + log(T(1.0) - rho_bym2_prior);

            if (spatial_in_params) {
                // phi_scaled ~ N(0, (Q + 11'/J)^{-1}); the likelihood sees phi
                // centred, so the constant direction is a free draw at the
                // field's own scale.
                T quad_form = icar_quadratic_form_t(phi_spatial, J, data);
                log_post = log_post - T(0.5) * (quad_form
                    + ratiod_constraints::areal_free_direction_quad(
                          phi_spatial_raw_sums, J, data.areal_components));

                // N(0, I) prior on theta
                for (int s = 0; s < J; s++) {
                    log_post = log_post - T(0.5) * theta_bym2[s] * theta_bym2[s];
                }
            }
        } else if (!layout.is_car_proper) {
            // ICAR: Gamma prior on tau
            T log_tau = params[layout.log_tau_spatial_idx];
            log_post = log_post + log_prior_gamma(log_tau, data.tau_spatial_shape, data.tau_spatial_rate);

            if (spatial_in_params) {
                T quad_form = icar_quadratic_form_t(phi_spatial, J, data);
                // (Q + 11'/J) is full rank, so the exponent on tau is J/2 rather
                // than (J-1)/2, and the quadratic form carries the freed direction.
                log_post = log_post + T(0.5 * J) * log_tau
                         - T(0.5) * tau_spatial * (quad_form
                             + ratiod_constraints::areal_free_direction_quad(
                          phi_spatial_raw_sums, J, data.areal_components));
            }
        } else {
            // Proper CAR: tau ~ Gamma(shape, rate); rho ~ Beta(a, b) scaled to
            // (car_rho_lower, car_rho_upper), with the logit-transform Jacobian
            // folded in (the untransformed density is car_proper::log_prior_rho;
            // the Jacobian log(upper-lower) + log(u) + log(1-u) combines with it
            // to a*log(u) + b*log(1-u)). Its log|Q(rho)| needs a dense Cholesky,
            // which has no templated form, so only the double instantiation
            // carries the block; log_post_impl_gap names it for the others.
            if constexpr (kDouble) {
                const double log_tau_sp = params[layout.log_tau_spatial_idx];
                log_post += (data.tau_spatial_shape - 1.0) * log_tau_sp
                          - data.tau_spatial_rate * tau_spatial + log_tau_sp;

                const double u_rho =
                    1.0 / (1.0 + std::exp(-params[layout.logit_rho_car_idx]));
                const double rho_car = data.car_rho_lower
                    + (data.car_rho_upper - data.car_rho_lower) * u_rho;
                log_post += data.car_rho_prior_a * std::log(u_rho)
                          + data.car_rho_prior_b * std::log(1.0 - u_rho);

                // phi ~ N(0, (tau * Q(rho))^{-1}), Q(rho) = D - rho*W; full rank
                // for |rho| < 1, so no sum-to-zero identification term is needed.
                auto Q = ratiod_car_proper::compute_car_precision(
                    J, data.adj_row_ptr, data.adj_col_idx, data.n_neighbors, rho_car);
                const double log_det_Q = ratiod_car_proper::car_log_det(J, Q);
                const double quad = ratiod_car_proper::car_quadratic_form(
                    phi_spatial_prior, J, data.adj_row_ptr, data.adj_col_idx,
                    data.n_neighbors, rho_car);
                log_post += 0.5 * log_det_Q + 0.5 * J * log_tau_sp
                          - 0.5 * tau_spatial * quad;
            }
        }
    }

    // GP spatial parameters and priors
    std::vector<T> gp_w;

    if (layout.is_gp && data.has_gp) {
        // Extract hyperparameters from log-scale
        T log_sigma2_gp = params[layout.log_sigma2_gp_idx];
        T log_phi_gp = params[layout.log_phi_gp_idx];
        T sigma2_gp = safe_exp(log_sigma2_gp);
        T phi_gp = safe_exp(log_phi_gp);

        // PC prior on sigma2 + Jacobian for log transform
        log_post = log_post + ratiod_gp::log_prior_sigma2_pc_t(
            sigma2_gp, data.gp_sigma2_prior_U, data.gp_sigma2_prior_alpha);
        log_post = log_post + log_sigma2_gp;  // Jacobian

        // Uniform prior on phi within bounds + Jacobian
        double phi_val = get_value(phi_gp);
        if (phi_val < data.gp_phi_prior_lower || phi_val > data.gp_phi_prior_upper) {
            return T(-INFINITY);
        }
        log_post = log_post + ratiod_gp::log_prior_phi_uniform_t(
            phi_gp, data.gp_phi_prior_lower, data.gp_phi_prior_upper);
        log_post = log_post + log_phi_gp;  // Jacobian

        if (data.gp_collapsed) {
            // The field is marginalized out: its mode is located with Newton and
            // a Laplace correction added, which is not a closed-form function of
            // the parameters. Only the double instantiation carries it;
            // log_post_impl_gap names the structure for the others.
            if constexpr (kDouble) {
                auto gp_res = collapsed_gp_log_post_contribution(
                    &params[layout.beta_num_start], &params[layout.beta_denom_start],
                    sigma2_gp, phi_gp, phi_num, phi_denom, data, collapsed_gp_ws());

                // w* is what the observation loop reads.
                const int n_star = data.gp_data.n_obs;
                gp_w.assign(collapsed_gp_ws().w_star.begin(),
                            collapsed_gp_ws().w_star.begin() + n_star);
                log_post = log_post + gp_res.log_post_contribution;
            }
        } else if (layout.gp_w_start < 0 ||
                   layout.gp_w_start + data.gp_data.n_obs > (int)params.size()) {
            return T(-INFINITY);  // Invalid parameter layout
        } else if (data.gp_parameterization == 1) {
            const int n_gp = layout.gp_w_end - layout.gp_w_start;
            // Non-centred: the slot holds z, and the field is w = L(sigma2, phi) z
            // built by the NNGP autoregression. The NNGP prior on w and the
            // Jacobian |dw/dz| cancel exactly, so what remains of the field's
            // own density is N(0, I) on z; both hyperparameters reach eta
            // through the transform instead, so a caller that has already
            // walked the observations needs no transform.
            std::vector<T> z_gp(n_gp);
            T z_sq_sum = T(0.0);
            for (int k = 0; k < n_gp; k++) {
                z_gp[k] = params[layout.gp_w_start + k];
                z_sq_sum = z_sq_sum + z_gp[k] * z_gp[k];
            }
            log_post = log_post - T(0.5) * z_sq_sum;

            if (!opts.skip_obs_loop) {
                ratiod_gp::NNGPNCWorkspaceT<T>& nc_ws = ratiod_gp::nngp_nc_ws<T>();
                ratiod_gp::nngp_nc_forward(z_gp.data(), sigma2_gp, phi_gp,
                                           data.gp_data, nc_ws);
                gp_w.assign(nc_ws.w.begin(), nc_ws.w.begin() + n_gp);
            }
        } else if (opts.precomputed_gp_log_prior) {
            log_post = log_post + T(*opts.precomputed_gp_log_prior);
        } else {
            // Centred: the slot holds w, which carries the NNGP prior directly.
            const int n_gp = layout.gp_w_end - layout.gp_w_start;
            gp_w.resize(n_gp);
            for (int k = 0; k < n_gp; k++) {
                gp_w[k] = params[layout.gp_w_start + k];
            }

            // NNGP log-likelihood on spatial effects
            log_post = log_post + ratiod_gp::gp_nngp_log_lik_t(
                gp_w, sigma2_gp, phi_gp, data.gp_data);
        }
    }

    // HSGP spatial parameters and priors
    std::vector<T> hsgp_f;

    if (layout.is_hsgp && data.has_hsgp) {
        T log_sigma2_hsgp = params[layout.log_sigma2_hsgp_idx];
        T log_ls_hsgp = params[layout.log_lengthscale_hsgp_idx];
        T sigma2_hsgp = safe_exp(log_sigma2_hsgp);
        T ls_hsgp = safe_exp(log_ls_hsgp);

        // PC prior on sigma: P(sigma > 1) = 0.01 -> rate 4.6
        log_post = log_post + ratiod_gp::log_prior_sd_pc_logvar_t(
            log_sigma2_hsgp, 4.6);

        // LogNormal(0, 1) on the lengthscale; the -log(ell) of the density and
        // the +log(ell) Jacobian of the log transform cancel.
        log_post = log_post - T(0.5) * log_ls_hsgp * log_ls_hsgp;

        // N(0, I) on the basis coefficients, and f = Phi * (sqrt(S) .* beta).
        const int M_hsgp = data.hsgp_data.m_total;
        std::vector<T> beta_hsgp(M_hsgp);
        for (int j = 0; j < M_hsgp; j++) {
            beta_hsgp[j] = params[layout.hsgp_beta_start + j];
            log_post = log_post - T(0.5) * beta_hsgp[j] * beta_hsgp[j];
        }
        ratiod_hsgp::hsgp_evaluate_t(beta_hsgp.data(), sigma2_hsgp, ls_hsgp,
                                     data.hsgp_data, hsgp_f);
    }

    // Multi-scale GP spatial parameters and priors
    std::vector<T> ms_gp_effect;

    if (layout.is_multiscale_gp && data.has_multiscale_gp) {
        // Both scales read one Hilbert-space basis, each a spectral expansion:
        // sigma carries a PC prior on its own scale, the lengthscale a
        // LogNormal on the log scale it is parameterized on, and the basis
        // coefficients N(0, I).
        T log_sigma2_local = params[layout.log_sigma2_gp_local_idx];
        T log_ls_local = params[layout.log_phi_gp_local_idx];
        T sigma2_local = safe_exp(log_sigma2_local);
        T ls_local = safe_exp(log_ls_local);

        T log_sigma2_regional = params[layout.log_sigma2_gp_regional_idx];
        T log_ls_regional = params[layout.log_phi_gp_regional_idx];
        T sigma2_regional = safe_exp(log_sigma2_regional);
        T ls_regional = safe_exp(log_ls_regional);

        // PC prior on each scale's sigma
        log_post = log_post + ratiod_gp::log_prior_sd_pc_logvar_t(
            log_sigma2_local, -std::log(data.ms_sigma2_local_prior_alpha)
                                  / data.ms_sigma2_local_prior_U);
        log_post = log_post + ratiod_gp::log_prior_sd_pc_logvar_t(
            log_sigma2_regional, -std::log(data.ms_sigma2_regional_prior_alpha)
                                     / data.ms_sigma2_regional_prior_U);

        // LogNormal on each lengthscale. No Jacobian: the prior is written on
        // the log scale the parameter already lives on.
        T z_ls_local = (log_ls_local - T(data.ms_log_ls_local_mean))
                       / T(data.ms_log_ls_local_sd);
        log_post = log_post - T(0.5) * z_ls_local * z_ls_local
                            - T(std::log(data.ms_log_ls_local_sd));

        T z_ls_regional = (log_ls_regional - T(data.ms_log_ls_regional_mean))
                          / T(data.ms_log_ls_regional_sd);
        log_post = log_post - T(0.5) * z_ls_regional * z_ls_regional
                            - T(std::log(data.ms_log_ls_regional_sd));

        const int m_total = data.msgp_hsgp_data.m_total;
        std::vector<T> beta_local(m_total), beta_regional(m_total);
        for (int j = 0; j < m_total; j++) {
            beta_local[j] = params[layout.gp_local_start + j];
            beta_regional[j] = params[layout.gp_regional_start + j];
            log_post = log_post - T(0.5) * beta_local[j] * beta_local[j];
            log_post = log_post - T(0.5) * beta_regional[j] * beta_regional[j];
        }

        // The basis is shared, so both expansions are observation-level.
        std::vector<T> f_local, f_regional;
        ratiod_hsgp::hsgp_evaluate_t(beta_local.data(), sigma2_local, ls_local,
                                     data.msgp_hsgp_data, f_local);
        ratiod_hsgp::hsgp_evaluate_t(beta_regional.data(), sigma2_regional,
                                     ls_regional, data.msgp_hsgp_data, f_regional);

        ms_gp_effect.resize(data.N);
        for (int ii = 0; ii < data.N; ii++) {
            ms_gp_effect[ii] = f_local[ii] + f_regional[ii];
        }
    }

    // Temporal priors
    if (layout.has_temporal) {
        int T_times = data.n_times;

        if (layout.is_temporal_gp && opts.precomputed_tgp_log_prior) {
            // Fused path: the gradient function has already accumulated every
            // temporal GP prior term. Only passed alongside skip_obs_loop, so
            // nothing reads the effects the branch below would reconstruct.
            log_post = log_post + T(*opts.precomputed_tgp_log_prior);
        } else if (layout.is_temporal_gp) {
            // Temporal GP: PC prior on sigma2, logit-bounded phi (lengthscale)
            T log_sigma2 = params[layout.log_sigma2_temporal_gp_idx];

            // PC prior on sigma2 (favor smaller variance)
            double rate = -std::log(data.temporal_gp_sigma2_prior_alpha) / data.temporal_gp_sigma2_prior_U;
            T sigma_gp = safe_sqrt(sigma2_temporal_gp);
            log_post = log_post + T(std::log(rate)) - T(rate) * sigma_gp - safe_log(T(2.0) * sigma_gp);
            log_post = log_post + log_sigma2;  // Jacobian for log transform

            // Uniform prior on phi: logit-bounded parameterization guarantees bounds
            // Jacobian: log(phi - lower) + log(upper - phi) - log(range)
            double phi_lower_pr = data.temporal_gp_phi_prior_lower;
            double phi_upper_pr = data.temporal_gp_phi_prior_upper;
            double phi_range_pr = phi_upper_pr - phi_lower_pr;
            log_post = log_post + safe_log(phi_temporal_gp - T(phi_lower_pr))
                     + safe_log(T(phi_upper_pr) - phi_temporal_gp)
                     - T(std::log(phi_range_pr));

            const bool use_nc = (data.temporal_gp_parameterization == 1);

            // Precompute shared rho[t] and derived quantities once (same dt for all groups)
                std::vector<T> rho_shared(T_times > 1 ? T_times - 1 : 0);
                std::vector<T> log_one_minus_rho2_shared(T_times > 1 ? T_times - 1 : 0);
                std::vector<T> a_shared(T_times > 1 ? T_times - 1 : 0);
                T sigma_t = safe_sqrt(sigma2_temporal_gp);
                for (int t = 1; t < T_times; t++) {
                    double dt = data.temporal_gp_data.time_values[t] - data.temporal_gp_data.time_values[t - 1];
                    rho_shared[t - 1] = safe_exp(T(-dt) / phi_temporal_gp);
                    T one_minus_rho2_safe = ratiod_ar1::one_minus_rho2(rho_shared[t - 1]);
                    log_one_minus_rho2_shared[t - 1] = safe_log(one_minus_rho2_safe);
                    a_shared[t - 1] = sigma_t * safe_sqrt(one_minus_rho2_safe);
                }

            if (use_nc) {
                // Non-centered: params store z ~ N(0,1)
                // Prior: z ~ N(0, I) for each temporal effect
                int n_temporal = layout.temporal_end - layout.temporal_start;
                for (int t = 0; t < n_temporal; t++) {
                    log_post = log_post - T(0.5) * phi_temporal[t] * phi_temporal[t];
                }

                // Jacobian of transform f = g(z, sigma2, phi):
                // log|det(df/dz)| = T*log(sigma) + 0.5*sum_{t>=1} log(1 - rho_t^2) per group
                T log_jac_per_group = T(T_times) * safe_log(sigma_t);
                for (int t = 1; t < T_times; t++) {
                    log_jac_per_group = log_jac_per_group + T(0.5) * log_one_minus_rho2_shared[t - 1];
                }
                log_post = log_post + T(data.n_temporal_groups) * log_jac_per_group;

                // Forward transform z -> f: overwrite phi_temporal for use in obs loop
                // f[0] = sigma * z[0]
                // f[t] = rho_t * f[t-1] + a_t * z[t]
                std::vector<T> f_reconstructed(n_temporal);
                for (int g = 0; g < data.n_temporal_groups; g++) {
                    int off = g * T_times;
                    f_reconstructed[off] = sigma_t * phi_temporal[off];
                    for (int t = 1; t < T_times; t++) {
                        f_reconstructed[off + t] = rho_shared[t - 1] * f_reconstructed[off + t - 1] + a_shared[t - 1] * phi_temporal[off + t];
                    }
                }
                // Replace phi_temporal with reconstructed f for observation loop
                phi_temporal = std::move(f_reconstructed);
            } else {
                // Centered: GP log-likelihood using state-space representation
                for (int g = 0; g < data.n_temporal_groups; g++) {
                    T f0 = phi_temporal[g * T_times];
                    log_post = log_post - T(0.5) * safe_log(T(2.0 * M_PI) * sigma2_temporal_gp);
                    log_post = log_post - T(0.5) * f0 * f0 / sigma2_temporal_gp;

                    for (int t = 1; t < T_times; t++) {
                        T f_prev = phi_temporal[g * T_times + t - 1];
                        T f_curr = phi_temporal[g * T_times + t];

                        T cond_var = sigma2_temporal_gp * (T(1.0) - rho_shared[t - 1] * rho_shared[t - 1]);
                        T cond_var_safe = safe_max(cond_var, T(1e-10));
                        T cond_mean = rho_shared[t - 1] * f_prev;
                        T resid = f_curr - cond_mean;

                        log_post = log_post - T(0.5) * safe_log(T(2.0 * M_PI) * cond_var_safe);
                        log_post = log_post - T(0.5) * resid * resid / cond_var_safe;
                    }
                }
            }
        } else {
            // RW1/RW2/AR1: tau-based parameterization
            T log_tau = params[layout.log_tau_temporal_idx];
            log_post = log_post + log_prior_gamma(log_tau, data.tau_temporal_shape, data.tau_temporal_rate);

            // The augmented constant directions' contribution to the quadratic
            // form, one per group; see temporal_is_intrinsic().
            const auto group_aug_quad = [&]() -> T {
                T q = T(0.0);
                if (!center_temporal) return q;
                for (int g = 0; g < data.n_temporal_groups; g++) {
                    const T s = tulpa::s2z_component_sum(
                        phi_temporal.data(), g * T_times, T_times);
                    q = q + tulpa::s2z_aug_coef(T(1.0), T_times) * s * s;
                }
                return q;
            };
            const int n_aug_pins = center_temporal ? data.n_temporal_groups : 0;

            if (rw_nc_temporal) {
                // The walk's quadratic form and the transform's Jacobian have
                // already cancelled, so what is left is N(0, I) on the
                // coordinates that carry it -- a non-cyclic RW2's free linear
                // direction keeps the flat prior it had -- and tau reaches the
                // field through the transform alone.
                const int order_rw =
                    (data.temporal_type == TemporalType::RW2) ? 2 : 1;
                log_post = log_post + ratiod_temporal_nc::rw_nc_grouped_log_prior(
                    z_temporal.data(), data.n_times, data.n_temporal_groups,
                    order_rw, data.temporal_cyclic);

            } else if (data.temporal_type == TemporalType::RW1) {
                // RW1: sum of (phi[t] - phi[t-1])^2
                T quad_form = T(0.0);
                for (int g = 0; g < data.n_temporal_groups; g++) {
                    for (int t = 1; t < T_times; t++) {
                        T diff = phi_temporal[g * T_times + t] - phi_temporal[g * T_times + t - 1];
                        quad_form = quad_form + diff * diff;
                    }
                    // Cyclic: add wrap-around edge (phi[0] - phi[T-1])
                    if (data.temporal_cyclic) {
                        T diff_cyclic = phi_temporal[g * T_times] - phi_temporal[g * T_times + T_times - 1];
                        quad_form = quad_form + diff_cyclic * diff_cyclic;
                    }
                }
                // Rank: T for cyclic, T-1 for non-cyclic, per group, plus each
                // group's constant the augmentation fills.
                quad_form = quad_form + group_aug_quad();
                int rank_rw1 = tulpa::s2z_aug_rank(
                    tulpa::rw1_rank(T_times, data.temporal_cyclic)
                        * data.n_temporal_groups, n_aug_pins);
                log_post = log_post + T(0.5 * rank_rw1) * log_tau;
                log_post = log_post - T(0.5) * tau_temporal * quad_form;

            } else if (data.temporal_type == TemporalType::RW2) {
                // RW2: sum of (phi[t] - 2*phi[t-1] + phi[t-2])^2
                T quad_form = T(0.0);
                for (int g = 0; g < data.n_temporal_groups; g++) {
                    for (int t = 2; t < T_times; t++) {
                        T diff = phi_temporal[g * T_times + t]
                               - T(2.0) * phi_temporal[g * T_times + t - 1]
                               + phi_temporal[g * T_times + t - 2];
                        quad_form = quad_form + diff * diff;
                    }
                    // Cyclic: add wrap-around second-order differences
                    if (data.temporal_cyclic && T_times >= 3) {
                        T d2_a = phi_temporal[g * T_times + T_times - 2]
                               - T(2.0) * phi_temporal[g * T_times + T_times - 1]
                               + phi_temporal[g * T_times];
                        T d2_b = phi_temporal[g * T_times + T_times - 1]
                               - T(2.0) * phi_temporal[g * T_times]
                               + phi_temporal[g * T_times + 1];
                        quad_form = quad_form + d2_a * d2_a + d2_b * d2_b;
                    }
                }
                // Rank: T for cyclic, T-2 for non-cyclic, per group, plus each
                // group's constant. A non-cyclic RW2 also keeps its per-group
                // LINEAR null direction, which this does not touch.
                quad_form = quad_form + group_aug_quad();
                int rank_rw2 = tulpa::s2z_aug_rank(
                    tulpa::rw2_rank(T_times, data.temporal_cyclic)
                        * data.n_temporal_groups, n_aug_pins);
                log_post = log_post + T(0.5 * rank_rw2) * log_tau;
                log_post = log_post - T(0.5) * tau_temporal * quad_form;

            } else if (data.temporal_type == TemporalType::AR1) {
                // Beta(a, b) on u = (rho + 1) / 2, logit Jacobian folded in
                log_post = log_post + ratiod_ar1::log_prior_rho(
                    rho_ar1, data.temporal_rho_prior_a, data.temporal_rho_prior_b);

                if (data.temporal_parameterization == 1) {
                    // Non-centred: the block holds the N(0, I) innovations the
                    // effects were reconstructed from, and the transform's
                    // Jacobian is what carries tau and rho.
                    const int n_z = static_cast<int>(z_temporal.size());
                    for (int i = 0; i < n_z; i++) {
                        log_post = log_post - T(0.5) * z_temporal[i] * z_temporal[i];
                    }
                    log_post = log_post - T(0.5 * n_z * std::log(2.0 * M_PI));
                } else {
                    // The normalized density, both the stationary first
                    // observation and the conditional residuals, since the
                    // gaussian normalizers carry the +0.5*T*log(tau) a
                    // gradient-based sampler needs.
                    for (int g = 0; g < data.n_temporal_groups; g++) {
                        log_post = log_post + ratiod_ar1::ar1_log_density(
                            &phi_temporal[g * T_times], T_times, rho_ar1,
                            tau_temporal);
                    }
                }

            } else if (data.temporal_type == TemporalType::IID) {
                // IID N(0, 1/tau). Full rank and proper, so no sum-to-zero
                // pin -- the same terms the analytic log posterior writes.
                T quad_form = T(0.0);
                for (int g = 0; g < data.n_temporal_groups; g++) {
                    for (int t = 0; t < T_times; t++) {
                        T phi_t = phi_temporal[g * T_times + t];
                        quad_form = quad_form + phi_t * phi_t;
                    }
                }
                const int n_iid = T_times * data.n_temporal_groups;
                log_post = log_post + T(0.5 * n_iid) * log_tau
                         - T(0.5 * n_iid * std::log(2.0 * M_PI));
                log_post = log_post - T(0.5) * tau_temporal * quad_form;
            }
        }
    }

    // Multiscale temporal parameters and priors
    // The blocks the sampler moves in, and the effects they contribute to eta.
    // The two are the same values in the centred coordinate and differ by the
    // transform in the non-centred one.
    std::vector<T> ms_trend_block;
    std::vector<T> ms_seasonal_block;
    std::vector<T> ms_trend;
    std::vector<T> ms_seasonal;
    std::vector<T> ms_short_term;
    std::vector<T> ms_temporal_eta;
    T ms_sigma2_trend = T(1.0);
    T ms_sigma2_seasonal = T(1.0);
    T ms_sigma2_short = T(1.0);
    T ms_rho_short = T(0.5);

    if (layout.has_multiscale_temporal) {
        const auto& ms_data = data.multiscale_temporal_data;

        // Trend component
        if (layout.log_sigma2_trend_idx >= 0) {
            T log_sigma2_trend = params[layout.log_sigma2_trend_idx];
            ms_sigma2_trend = safe_exp(log_sigma2_trend);

            int n_trend = layout.trend_end - layout.trend_start;
            ms_trend_block.resize(n_trend);
            for (int t = 0; t < n_trend; t++) {
                ms_trend_block[t] = params[layout.trend_start + t];
            }
            ratiod_temporal::ms_arm_effects(
                ms_trend_block, ratiod_temporal::ms_arm_order(ms_data.trend_type),
                false, ms_sigma2_trend, ms_data.noncentered, ms_trend);

            // PC prior on sigma2_trend + Jacobian for log transform
            log_post = log_post + ratiod_temporal::log_prior_sigma2_temporal_pc(
                ms_sigma2_trend, data.ms_sigma2_trend_prior_U, data.ms_sigma2_trend_prior_alpha);
            log_post = log_post + log_sigma2_trend;  // Jacobian
        }

        // Seasonal component
        if (layout.log_sigma2_seasonal_idx >= 0) {
            T log_sigma2_seasonal = params[layout.log_sigma2_seasonal_idx];
            ms_sigma2_seasonal = safe_exp(log_sigma2_seasonal);

            int n_seasonal = layout.seasonal_end - layout.seasonal_start;
            ms_seasonal_block.resize(n_seasonal);
            for (int t = 0; t < n_seasonal; t++) {
                ms_seasonal_block[t] = params[layout.seasonal_start + t];
            }
            ratiod_temporal::ms_arm_effects(
                ms_seasonal_block, 1, true, ms_sigma2_seasonal,
                ms_data.noncentered, ms_seasonal);

            // PC prior on sigma2_seasonal + Jacobian
            log_post = log_post + ratiod_temporal::log_prior_sigma2_temporal_pc(
                ms_sigma2_seasonal, data.ms_sigma2_seasonal_prior_U, data.ms_sigma2_seasonal_prior_alpha);
            log_post = log_post + log_sigma2_seasonal;  // Jacobian
        }

        // Short-term component
        if (layout.log_sigma2_short_idx >= 0) {
            T log_sigma2_short = params[layout.log_sigma2_short_idx];
            ms_sigma2_short = safe_exp(log_sigma2_short);

            int n_short = layout.short_term_end - layout.short_term_start;
            ms_short_term.resize(n_short);
            for (int t = 0; t < n_short; t++) {
                ms_short_term[t] = params[layout.short_term_start + t];
            }

            // PC prior on sigma2_short + Jacobian
            log_post = log_post + ratiod_temporal::log_prior_sigma2_temporal_pc(
                ms_sigma2_short, data.ms_sigma2_short_prior_U, data.ms_sigma2_short_prior_alpha);
            log_post = log_post + log_sigma2_short;  // Jacobian

            // AR1 rho parameter
            if (layout.logit_rho_short_idx >= 0) {
                T logit_rho_short = params[layout.logit_rho_short_idx];
                // Map logit to (-1, 1): rho = 2*invlogit(logit) - 1
                T u = inv_logit(logit_rho_short);
                ms_rho_short = T(2.0) * u - T(1.0);

                // Beta(2, 2) on u = (rho + 1) / 2, logit Jacobian folded in
                log_post = log_post + ratiod_ar1::log_prior_logit_rho(
                    u, ratiod_ar1::RHO_PRIOR_A, ratiod_ar1::RHO_PRIOR_B);
            }
        }

        // GMRF log-likelihood for all components
        log_post = log_post + ratiod_temporal::multiscale_temporal_log_lik(
            ms_trend_block, ms_seasonal_block, ms_short_term,
            ms_sigma2_trend, ms_sigma2_seasonal, ms_sigma2_short, ms_rho_short,
            ms_data);

        // Precompute multiscale temporal contribution to linear predictor
        ratiod_temporal::compute_temporal_eta(
            ms_trend, ms_seasonal, ms_short_term, ms_data, ms_temporal_eta);
    }

    // SVC (Spatially-Varying Coefficients) parameters and priors
    std::vector<T> svc_sigma2;
    std::vector<T> svc_phi;
    std::vector<T> svc_w_flat;
    std::vector<T> svc_eta;

    if (layout.has_svc && data.svc_data.n_svc > 0) {
        int n_svc = data.svc_data.n_svc;
        int n_obs = data.svc_data.n_obs;

        if (data.svc_is_hsgp) {
            // HSGP-based SVC: basis function approximation
            int m_total = data.svc_hsgp_data.m_total;

            // Per-term: sigma2, lengthscale, beta[m_total]
            svc_eta.resize(n_obs, T(0.0));
            for (int j = 0; j < n_svc; j++) {
                T log_sigma2 = params[layout.log_sigma2_svc_start + j];
                T sigma2_j = safe_exp(log_sigma2);

                // PC prior on sigma
                log_post = log_post + ratiod_gp::log_prior_sd_pc_logvar_t(
                    log_sigma2, 4.6);

                T log_ls = params[layout.range_svc_start + j];
                T ls_j = safe_exp(log_ls);

                // LogNormal(0,1) on lengthscale
                log_post = log_post - T(0.5) * log_ls * log_ls;

                // N(0, I) prior on beta_j, then f_j = Phi * (sqrt(S_j) .* beta_j)
                // through the one expansion every HSGP block reads.
                std::vector<T> beta_j(m_total);
                for (int k = 0; k < m_total; k++) {
                    beta_j[k] = params[layout.svc_w_start + j * m_total + k];
                    log_post = log_post - T(0.5) * beta_j[k] * beta_j[k];
                }

                std::vector<T> f_j;
                ratiod_hsgp::hsgp_evaluate_t(beta_j.data(), sigma2_j, ls_j,
                                             data.svc_hsgp_data, f_j);
                for (int i = 0; i < n_obs; i++) {
                    svc_eta[i] = svc_eta[i]
                        + f_j[i] * T(data.svc_data.X_svc[i * n_svc + j]);
                }
            }
        } else {
            // NNGP-based SVC (original path)

            // Extract sigma2 (spatial variance) parameters
            svc_sigma2.resize(n_svc);
            for (int j = 0; j < n_svc; j++) {
                T log_sigma2 = params[layout.log_sigma2_svc_start + j];
                svc_sigma2[j] = safe_exp(log_sigma2);

                T sigma = safe_sqrt(svc_sigma2[j]);
                double scale = data.svc_sigma2_prior_scale;
                log_post = log_post - safe_log(T(1.0) + sigma * sigma / T(scale * scale));
                log_post = log_post + log_sigma2;
            }

            // Range parameters: uniform on the prior interval, sampled on its
            // logit scale (svc_range)
            svc_phi.resize(n_svc);
            for (int j = 0; j < n_svc; j++) {
                const T u = params[layout.range_svc_start + j];
                svc_phi[j] = ratiod_svc_ad::svc_range(
                    u, data.svc_phi_prior_lower, data.svc_phi_prior_upper);
                log_post = log_post + ratiod_svc_ad::svc_range_log_density(u);
            }

            // The sampled block, and the field it stands for
            int n_svc_params = n_svc * n_obs;
            std::vector<T> svc_block(n_svc_params);
            for (int k = 0; k < n_svc_params; k++) {
                svc_block[k] = params[layout.svc_w_start + k];
            }
            // Non-centred, the field enters only through eta, so a caller that
            // has already walked the observations needs no transform at all.
            const bool need_field = !(data.svc_noncentered && opts.skip_obs_loop);
            if (need_field) {
                svc_w_flat.resize(n_svc_params);
                ratiod_svc::svc_field<ratiod_svc::SVC_DENSITY_SLOT>(svc_block.data(), svc_sigma2.data(),
                                      svc_phi.data(), data.svc_data,
                                      data.svc_gp_view, data.svc_noncentered,
                                      svc_w_flat.data());
            }

            if (data.svc_noncentered) {
                // The NNGP prior on w and the Jacobian |dw/dz| cancel, so the
                // field's own density is N(0, I) on z.
                T z_sq_sum = T(0.0);
                for (int k = 0; k < n_svc_params; k++)
                    z_sq_sum = z_sq_sum + svc_block[k] * svc_block[k];
                log_post = log_post - T(0.5) * z_sq_sum;
            } else if (opts.precomputed_svc_log_prior) {
                log_post = log_post + T(*opts.precomputed_svc_log_prior);
            } else {
                // NNGP prior on each SVC term: the single-scale GP density on
                // the SVC neighbour structure
                for (int j = 0; j < n_svc; j++) {
                    std::vector<T> w_j(n_obs);
                    for (int k = 0; k < n_obs; k++) {
                        w_j[k] = svc_w_flat[j * n_obs + k];
                    }
                    log_post = log_post + ratiod_gp::gp_nngp_log_lik_t(
                        w_j, svc_sigma2[j], svc_phi[j], data.svc_gp_view);
                }
            }

            // Identify each term's level by centring it on its way into eta:
            // the NNGP prior above is already proper on the constant
            // direction, so what the alias with beta_j needs is that direction
            // removed from the likelihood, not a second prior stiffening it.
            if (need_field) {
                ratiod_svc_ad::svc_center_eta(svc_w_flat, data.svc_data, svc_eta);
            }
        }
    }

    // TVC (Temporally-Varying Coefficients) parameters and priors
    std::vector<T> tvc_tau;
    std::vector<T> tvc_rho;
    std::vector<T> tvc_w_flat;
    std::vector<T> tvc_eta;

    if (layout.has_tvc && data.tvc_data.n_tvc > 0) {
        int n_tvc = data.tvc_data.n_tvc;
        int n_groups = data.tvc_data.n_groups;
        int n_times = data.tvc_data.n_times;
        int n_obs = data.tvc_data.n_obs;

        // Extract tau (precision) parameters
        tvc_tau.resize(n_tvc);
        for (int j = 0; j < n_tvc; j++) {
            T log_tau = params[layout.log_tau_tvc_start + j];
            tvc_tau[j] = safe_exp(log_tau);

            // Gamma(shape, rate) on tau + Jacobian for log transform
            log_post = log_post + log_prior_gamma(log_tau, data.tvc_tau_shape,
                                                  data.tvc_tau_rate);
        }

        // Extract rho (AR1 correlation) parameters if AR1 structure
        tvc_rho.resize(n_tvc, T(0.0));
        if (data.tvc_data.structure == ratiod_temporal::TemporalType::AR1) {
            for (int j = 0; j < n_tvc; j++) {
                T logit_rho = params[layout.logit_rho_tvc_start + j];
                // Map logit to (-1, 1): rho = 2*invlogit(logit) - 1
                T u = inv_logit(logit_rho);
                tvc_rho[j] = T(2.0) * u - T(1.0);

                // Beta(2, 2) on u = (rho + 1) / 2, logit Jacobian folded in
                log_post = log_post + ratiod_ar1::log_prior_logit_rho(
                    u, ratiod_ar1::RHO_PRIOR_A, ratiod_ar1::RHO_PRIOR_B);
            }
        }

        // Extract TVC values
        int n_tvc_params = n_groups * n_tvc * n_times;
        tvc_w_flat.resize(n_tvc_params);
        for (int k = 0; k < n_tvc_params; k++) {
            tvc_w_flat[k] = params[layout.tvc_w_start + k];
        }

        // TVC temporal prior (RW1, RW2, or AR1)
        log_post = log_post + ratiod_tvc_ad::tvc_log_prior(
            tvc_w_flat, data.tvc_data, tvc_tau, tvc_rho
        );

        // Soft sum-to-zero constraint for identifiability
        log_post = log_post + ratiod_tvc::tvc_sum_to_zero_penalty(
            tvc_w_flat, data.tvc_data
        );

        // Precompute TVC contribution to linear predictor
        tvc_eta.resize(n_obs, T(0.0));
        ratiod_tvc_ad::compute_tvc_eta(tvc_w_flat, data.tvc_data, tvc_eta);
    }

    // Latent factors: loadings and raw scores carry the priors, the linear
    // predictors read the constrained scores (hmc_latent.h)
    std::vector<T> latent_eta_num;
    std::vector<T> latent_eta_denom;

    if (layout.has_latent && data.latent_n_factors > 0) {
        const int K = data.latent_n_factors;
        const int N = data.N;
        const int P = ratiod_latent::n_processes(data.latent_shared);
        const auto constraint =
            static_cast<ratiod_latent::LatentConstraint>(data.latent_constraint);

        const T* theta = &params[layout.latent_loading_start];
        const T* raw = &params[layout.latent_factor_start];
        std::vector<T> lam, f;
        ratiod_latent::loadings(theta, P, K, lam);
        ratiod_latent::constrain_scores(raw, N, K, constraint, f);

        log_post = log_post + ratiod_latent::loadings_log_prior(
            theta, P, K, data.latent_sigma_prior_rate, data.latent_offdiag_sd);
        log_post = log_post + ratiod_latent::scores_log_prior(raw, N * K);

        ratiod_latent::process_effects(f, lam, N, P, K, latent_eta_num,
                                       latent_eta_denom);
    }

    // Spatiotemporal interaction priors
    const T* st_delta = nullptr;
    std::vector<T> st_delta_nc;

    if (layout.has_spatiotemporal &&
        data.spatiotemporal_data.type != STType::NONE) {
        const auto& st_data = data.spatiotemporal_data;
        const int S = st_data.n_spatial;
        const int T_st = st_data.n_times;
        const int ST = S * T_st;

        // Precision parameter
        T log_tau_st = params[layout.log_tau_st_idx];
        T tau_st = safe_exp(log_tau_st);
        T tau_st2 = T(1.0);
        T rho_st = T(0.0);
        T phi_st_space = T(1.0);
        T phi_st_time = T(1.0);

        // PC prior on sigma = 1/sqrt(tau), as a density of log tau
        log_post = log_post + ratiod::math::pc_log_prior_log_precision(
            log_tau_st, data.st_sigma2_prior_U, data.st_sigma2_prior_alpha);

        // AR1 rho parameter
        if (layout.logit_rho_st_idx >= 0) {
            rho_st = ratiod_ar1::rho_from_logit(params[layout.logit_rho_st_idx]);

            // Beta(a, b) on u = (rho + 1) / 2, logit Jacobian folded in
            log_post = log_post + ratiod_ar1::log_prior_rho(
                rho_st, st_data.rho_prior_a, st_data.rho_prior_b);
        }

        // GP range parameters
        if (layout.is_st_gp) {
            T log_phi_space = params[layout.log_phi_st_space_idx];
            T log_phi_time = params[layout.log_phi_st_time_idx];
            phi_st_space = safe_exp(log_phi_space);
            phi_st_time = safe_exp(log_phi_time);

            // Uniform prior within bounds
            if (get_value(phi_st_space) < data.st_phi_space_prior_lower ||
                get_value(phi_st_space) > data.st_phi_space_prior_upper) {
                return T(-std::numeric_limits<double>::infinity());
            }
            if (get_value(phi_st_time) < data.st_phi_time_prior_lower ||
                get_value(phi_st_time) > data.st_phi_time_prior_upper) {
                return T(-std::numeric_limits<double>::infinity());
            }
            log_post = log_post + log_phi_space + log_phi_time;  // Jacobians
        }

        if (layout.st_collapsed) {
            // Integrated out with the observations below; the margin
            // penalty's normalizer stays here, as in the centred coordinate.
            T st_norm;
            if (!ratiod_spatiotemporal::st_margin_normalizer(
                    tau_st, rho_st, phi_st_space, phi_st_time, st_data, &st_norm)) {
                return T(-std::numeric_limits<double>::infinity());
            }
            log_post = log_post + st_norm;
        } else {
            const T* z_or_delta = &params[layout.st_delta_start];

            // NC reparameterization: store z, reconstruct delta
            const bool st_use_nc = ratiod_spatiotemporal::st_noncentered_gmrf(
                data.st_parameterization, data.st_is_hsgp, st_data);
            const bool st_use_nc_gp = ratiod_spatiotemporal::st_noncentered_gp(
                data.st_parameterization, data.st_is_hsgp, st_data);

            if (st_use_nc_gp) {
                // The GP interaction from z ~ N(0, I) through the NNGP
                // autoregression: the field's density is N(0, I) on z, and the
                // soft sum-to-zero with the normalizer of the subspace it leaves
                // stays on delta, as in the centred coordinate.
                st_delta_nc.resize(ST);
                if (!ratiod_spatiotemporal::st_gp_nc_forward(
                        z_or_delta, T(1.0) / tau_st, phi_st_space, phi_st_time,
                        st_data, st_delta_nc.data())) {
                    return T(-std::numeric_limits<double>::infinity());
                }
                st_delta = st_delta_nc.data();

                T z_sq = T(0.0);
                for (int k = 0; k < ST; k++) z_sq = z_sq + z_or_delta[k] * z_or_delta[k];
                log_post = log_post - T(0.5) * z_sq - T(0.5 * ST * std::log(2.0 * M_PI));

                log_post = log_post + ratiod_spatiotemporal::st_sum_to_zero_penalty(
                    st_delta, S, T_st, true, true
                );
                T st_norm;
                if (!ratiod_spatiotemporal::st_margin_normalizer(
                        tau_st, rho_st, phi_st_space, phi_st_time, st_data, &st_norm)) {
                    return T(-std::numeric_limits<double>::infinity());
                }
                log_post = log_post + st_norm;
            } else if (st_use_nc) {
                // Forward transform: delta = z / sqrt(tau_st)
                T inv_scale = T(1.0) / safe_sqrt(tau_st);
                st_delta_nc.resize(ST);
                for (int k = 0; k < ST; k++) {
                    st_delta_nc[k] = z_or_delta[k] * inv_scale;
                }
                st_delta = st_delta_nc.data();

                // NC prior: -0.5 * z^T (Q_s (x) Q_t) z  (tau-free GMRF)
                log_post = log_post + ratiod_spatiotemporal::spatiotemporal_log_prior(
                    z_or_delta, T(1.0), T(1.0), rho_st, phi_st_space, phi_st_time,
                    st_data
                );

                // The GMRF normalizer and the NC Jacobian, less the null
                // directions the sum-to-zero penalty on z pins
                const int tau_power = ratiod_spatiotemporal::st_noncentered_tau_power(st_data);
                log_post = log_post + T(0.5 * tau_power) * safe_log(tau_st);

                // Sum-to-zero on z. In the hard limit this is the constraint on
                // delta = z / sqrt(tau); held on z, its strength does not fade as
                // tau grows, which would leave z free to drift along the
                // directions Q does not constrain.
                log_post = log_post + ratiod_spatiotemporal::st_sum_to_zero_penalty(
                    z_or_delta, S, T_st, true, true
                );
            } else if (data.st_is_hsgp) {
                // HSGP-ST: spectral basis interaction (centered). Each basis
                // function j gets an independent temporal GMRF with precision
                // tau_st / S(lambda_j).
                st_delta = z_or_delta;
                const int M = data.st_hsgp_data.m_total;

                T log_sigma2_st = params[layout.log_sigma2_st_hsgp_idx];
                T log_ls_st = params[layout.log_lengthscale_st_hsgp_idx];
                T sigma2_st_hsgp = safe_exp(log_sigma2_st);
                T lengthscale_st_hsgp = safe_exp(log_ls_st);

                // PC prior on sigma_st_hsgp: rate = 4.6
                log_post = log_post + ratiod_gp::log_prior_sd_pc_logvar_t(
                    log_sigma2_st, 4.6);

                // LogNormal(0,1) on lengthscale
                log_post = log_post - T(0.5) * log_ls_st * log_ls_st;

                int rank_t = ratiod_spatiotemporal::st_time_rank(
                    st_data.temporal_type, T_st, st_data.temporal_cyclic);
                const bool st_hsgp_ar1 =
                    (st_data.temporal_type == TemporalType::AR1 && T_st >= 2);
                if (st_hsgp_ar1) {
                    // One log|R(rho)| per basis function.
                    log_post = log_post + T(0.5 * M) *
                               safe_log(ratiod_ar1::one_minus_rho2(rho_st));
                }

                for (int j = 0; j < M; j++) {
                    T S_j = ratiod_hsgp::spectral_density_se(
                        data.st_hsgp_data.eigenvalues[j], sigma2_st_hsgp,
                        lengthscale_st_hsgp);
                    T S_j_floor = (get_value(S_j) < 1e-10) ? T(1e-10) : S_j;
                    T prec_j = tau_st / S_j_floor;

                    // GMRF quadratic form: -0.5 * prec_j * delta_j' Q_t delta_j
                    const T* dj = &st_delta[j * T_st];
                    T qf = T(0.0);
                    if (st_data.temporal_type == TemporalType::RW1) {
                        for (int t = 1; t < T_st; t++) {
                            T d1 = dj[t] - dj[t - 1];
                            qf = qf + d1 * d1;
                        }
                    } else if (st_data.temporal_type == TemporalType::RW2) {
                        for (int t = 2; t < T_st; t++) {
                            T d2 = dj[t] - T(2.0) * dj[t - 1] + dj[t - 2];
                            qf = qf + d2 * d2;
                        }
                    } else if (st_hsgp_ar1) {
                        qf = ratiod_ar1::ar1_quadratic_form(dj, T_st, rho_st);
                    }
                    log_post = log_post + T(0.5 * rank_t) * safe_log(prec_j)
                                        - T(0.5) * prec_j * qf;

                    // Soft sum-to-zero per basis function
                    T sum_j = T(0.0);
                    for (int t = 0; t < T_st; t++) sum_j = sum_j + dj[t];
                    log_post = log_post - T(0.5 * tulpa::s2z_precision(T_st)) * sum_j * sum_j;
                }
            } else {
                // Centered parameterization (ICAR/BYM2 spatial)
                st_delta = z_or_delta;

                log_post = log_post + ratiod_spatiotemporal::spatiotemporal_log_prior(
                    st_delta, tau_st, tau_st2, rho_st, phi_st_space, phi_st_time,
                    st_data
                );

                // Soft sum-to-zero constraint for identifiability, and the
                // normalizer of the subspace it leaves
                log_post = log_post + ratiod_spatiotemporal::st_sum_to_zero_penalty(
                    st_delta, S, T_st, true, true
                );
                T st_norm;
                if (!ratiod_spatiotemporal::st_margin_normalizer(
                        tau_st, rho_st, phi_st_space, phi_st_time, st_data, &st_norm)) {
                    return T(-std::numeric_limits<double>::infinity());
                }
                log_post = log_post + st_norm;
            }
        }
    }

    // Zero-inflation / One-inflation parameters
    std::vector<T> beta_zi;
    std::vector<T> beta_oi;

    if (layout.has_zi && data.p_zi > 0) {
        beta_zi.resize(data.p_zi);
        for (int j = 0; j < data.p_zi; j++) {
            beta_zi[j] = params[layout.beta_zi_start + j];
        }
        // Prior on beta_zi: N(0, zi_prior_sd^2)
        double tau_zi = 1.0 / (data.zi_prior_sd * data.zi_prior_sd);
        for (int j = 0; j < data.p_zi; j++) {
            log_post = log_post + log_prior_normal(beta_zi[j], tau_zi);
        }
    }

    if (layout.has_oi && data.p_oi > 0) {
        beta_oi.resize(data.p_oi);
        for (int j = 0; j < data.p_oi; j++) {
            beta_oi[j] = params[layout.beta_oi_start + j];
        }
        // Prior on beta_oi: N(0, oi_prior_sd^2)
        double tau_oi = 1.0 / (data.oi_prior_sd * data.oi_prior_sd);
        for (int j = 0; j < data.p_oi; j++) {
            log_post = log_post + log_prior_normal(beta_oi[j], tau_oi);
        }
    }

    // ========================================================================
    // LIKELIHOOD
    // ========================================================================

    // The spatial field at observation i: an areal field (a collapsed field
    // has no slot but does have a mode, which the collapsed contribution above
    // pointed phi_spatial at), an NNGP field read at the observation's
    // location, or a basis expansion evaluated at the observation. `shared`
    // enters both predictors, `num_only` the numerator's alone.
    const bool areal_field = layout.has_spatial && phi_spatial != nullptr
                             && !data.spatial_group.empty();
    const bool gp_field = layout.is_gp && !gp_w.empty();
    const bool hsgp_field = !hsgp_f.empty();
    const bool msgp_field = layout.is_multiscale_gp && !ms_gp_effect.empty();
    const bool has_spatial_field = areal_field || gp_field || hsgp_field || msgp_field;

    auto spatial_field_at = [&](int i, T& shared, T& num_only) {
        if (areal_field && data.spatial_group[i] > 0) {
            int s = data.spatial_group[i] - 1;
            if (layout.is_bym2) {
                T scaled_phi = phi_spatial[s] * T(data.bym2_scale_factor);
                shared = shared + (sigma_s_bym2 * scaled_phi + sigma_u_bym2 * theta_bym2[s]);
            } else {
                shared = shared + phi_spatial[s];
            }
        }
        if (gp_field) {
            T e = gp_w[data.gp_data.obs_to_loc[i]];
            if (data.gp_data.shared) shared = shared + e; else num_only = num_only + e;
        }
        if (hsgp_field) {
            if (data.hsgp_data.shared) shared = shared + hsgp_f[i];
            else num_only = num_only + hsgp_f[i];
        }
        if (msgp_field) {
            if (data.multiscale_gp_data.shared) shared = shared + ms_gp_effect[i];
            else num_only = num_only + ms_gp_effect[i];
        }
    };

    // Under restricted spatial regression the predictors read the field's
    // projection onto the complement of the restricted covariates (rsr.h).
    std::vector<T> rsr_shared, rsr_num_only;
    if (data.has_rsr && has_spatial_field) {
        rsr_shared.assign(data.N, T(0.0));
        rsr_num_only.assign(data.N, T(0.0));
        for (int i = 0; i < data.N; i++) {
            spatial_field_at(i, rsr_shared[i], rsr_num_only[i]);
        }
        ratiod_rsr::project_out(rsr_shared, data.rsr_basis, data.rsr_rank);
        ratiod_rsr::project_out(rsr_num_only, data.rsr_basis, data.rsr_rank);
    }

    // One observation's two linear predictors and inflation logits, written
    // once for every scalar type.
    struct ObsPredictors { T eta_num, eta_denom, logit_zi, logit_oi; };
    auto obs_predictors = [&](int i) -> ObsPredictors {
        // Compute linear predictors
        T eta_num = T(0.0);
        T eta_denom = T(0.0);

        for (int j = 0; j < data.p_num; j++) {
            eta_num = eta_num + data.X_num_flat[i * data.p_num + j] * beta_num[j];
        }
        for (int j = 0; j < data.p_denom; j++) {
            eta_denom = eta_denom + data.X_denom_flat[i * data.p_denom + j] * beta_denom[j];
        }

        // Add random effects (shared between num and denom)
        if (!re_eff.empty()) {
            T re_contrib = ratiod_re::re_eta(i, data, layout, re_eff);
            eta_num = eta_num + re_contrib;
            eta_denom = eta_denom + re_contrib;
        }

        // Add the spatial field.
        if (has_spatial_field) {
            T shared = T(0.0), num_only = T(0.0);
            if (data.has_rsr) {
                shared = rsr_shared[i];
                num_only = rsr_num_only[i];
            } else {
                spatial_field_at(i, shared, num_only);
            }
            eta_num = eta_num + shared + num_only;
            eta_denom = eta_denom + shared;
        }

        // Add temporal effects
        if (layout.has_temporal && !data.temporal_time_idx.empty() && data.temporal_time_idx[i] > 0) {
            int t = data.temporal_time_idx[i] - 1;
            int g = data.temporal_group_idx[i] - 1;
            // Intrinsic walks are centred on the way in; their augmented
            // constant carries tau, so leaving it in eta would free the level.
            T temporal_effect = center_temporal
                ? phi_temporal[g * data.n_times + t] - phi_temporal_mean[g]
                : phi_temporal[g * data.n_times + t];

            if (data.temporal_shared) {
                eta_num = eta_num + temporal_effect;
                eta_denom = eta_denom + temporal_effect;
            } else {
                eta_num = eta_num + temporal_effect;
            }
        }

        // Add multiscale temporal effect
        if (layout.has_multiscale_temporal && !ms_temporal_eta.empty()) {
            T ms_temp_effect = ms_temporal_eta[i];
            if (data.multiscale_temporal_data.shared) {
                eta_num = eta_num + ms_temp_effect;
                eta_denom = eta_denom + ms_temp_effect;
            } else {
                eta_num = eta_num + ms_temp_effect;
            }
        }

        // Add SVC (Spatially-Varying Coefficients) effect
        if (layout.has_svc && !svc_eta.empty()) {
            T svc_effect = svc_eta[i];
            if (data.svc_data.shared) {
                eta_num = eta_num + svc_effect;
                eta_denom = eta_denom + svc_effect;
            } else {
                eta_num = eta_num + svc_effect;
            }
        }

        // Add TVC (Temporally-Varying Coefficients) effect
        if (layout.has_tvc && !tvc_eta.empty()) {
            T tvc_effect = tvc_eta[i];
            if (data.tvc_data.shared) {
                eta_num = eta_num + tvc_effect;
                eta_denom = eta_denom + tvc_effect;
            } else {
                eta_num = eta_num + tvc_effect;
            }
        }

        // Add each process's latent factor effect
        if (!latent_eta_num.empty()) {
            eta_num = eta_num + latent_eta_num[i];
            if (!latent_eta_denom.empty()) eta_denom = eta_denom + latent_eta_denom[i];
        }

        // Add spatiotemporal interaction effect
        if (layout.has_spatiotemporal && st_delta != nullptr) {
            T st_effect = T(0.0);
            if (data.st_is_hsgp) {
                // HSGP-ST: sum_j Phi[i,j] * delta_st[j * T + t - 1]
                int t = data.spatiotemporal_data.t_idx[i] - 1;  // 0-based
                int M = data.st_hsgp_data.m_total;
                int T_st = data.spatiotemporal_data.n_times;
                for (int j = 0; j < M; j++) {
                    st_effect = st_effect +
                        data.st_hsgp_data.phi_flat[i * M + j] * st_delta[j * T_st + t];
                }
            } else {
                // ICAR-ST: direct index lookup
                int st_idx = data.spatiotemporal_data.st_flat[i];
                if (st_idx > 0) st_effect = st_delta[st_idx - 1];
            }
            if (data.spatiotemporal_data.shared) {
                eta_num = eta_num + st_effect;
                eta_denom = eta_denom + st_effect;
            } else {
                eta_num = eta_num + st_effect;
            }
        }

        // Compute ZI/OI linear predictors if needed
        T logit_zi = T(0.0);
        T logit_oi = T(0.0);

        if (layout.has_zi && data.p_zi > 0) {
            for (int j = 0; j < data.p_zi; j++) {
                logit_zi = logit_zi + data.X_zi_flat[i * data.p_zi + j] * beta_zi[j];
            }
        }

        if (layout.has_oi && data.p_oi > 0) {
            for (int j = 0; j < data.p_oi; j++) {
                logit_oi = logit_oi + data.X_oi_flat[i * data.p_oi + j] * beta_oi[j];
            }
        }
        // The inflation kernels take their probabilities on the logit scale
        logit_zi = prob_logit(logit_zi, data.link_zi);
        logit_oi = prob_logit(logit_oi, data.link_oi);

        return ObsPredictors{eta_num, eta_denom, logit_zi, logit_oi};
    };

    // One observation's contribution. How the sum over observations is taken
    // is what differs, below.
    auto obs_log_lik_i = [&](int i) -> T {
        const ObsPredictors pr = obs_predictors(i);
        return ratio_obs_log_lik(data, layout, i, pr.eta_num, pr.eta_denom,
                                 phi_num, phi_denom, pr.logit_zi, pr.logit_oi);
    };

    // A caller that accumulated the observation likelihood during a fused
    // gradient pass asks for the prior and structural terms alone.
    if (!opts.skip_obs_loop && layout.st_collapsed) {
        // The interaction integrated out: the observations enter through the
        // Laplace marginal, with every other block as an offset.
        if constexpr (kDouble) {
            std::vector<double> off_num(data.N), off_denom(data.N);
            for (int i = 0; i < data.N; i++) {
                const ObsPredictors pr = obs_predictors(i);
                off_num[i] = pr.eta_num;
                off_denom[i] = pr.eta_denom;
            }
            log_post = log_post + ratiod_st_laplace::evaluate(
                data, layout, params, off_num.data(), off_denom.data(),
                phi_num, phi_denom);
        } else {
            Rcpp::stop("The collapsed GP interaction has no autodiff density.");
        }
    } else if (!opts.skip_obs_loop) {
        if constexpr (kDouble) {
            // Only the double instantiation takes this in parallel: an autodiff
            // tape is not thread-safe. The NNGP paths read shared neighbour
            // structures, so they stay on one thread.
            const int use_threads =
                (layout.is_gp || layout.is_multiscale_gp) ? 1 : data.n_threads;
            const double obs_log_lik = ratiod_omp::sum_range(
                data.N, use_threads, [&](int i) { return obs_log_lik_i(i); });
            log_post = log_post + obs_log_lik;
        } else {
            T obs_log_lik = T(0.0);
            for (int i = 0; i < data.N; i++) {
                obs_log_lik = obs_log_lik + obs_log_lik_i(i);
            }
            log_post = log_post + obs_log_lik;
        }
    }

    return log_post;
}

}  // namespace ratiod

#endif  // RATIOD_LOG_POST_IMPL_H
