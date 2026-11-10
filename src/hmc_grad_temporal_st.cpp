#include <RcppEigen.h>
#include "hmc_grad_helpers.h"
#include "hmc_multiscale_temporal_grad.h"
#include "st_prior_grad.h"

using namespace Rcpp;

namespace ratiod_hmc {
void compute_gradient_ms_temporal_handcoded(
    const std::vector<double>& params_in,
    const ModelData& data,
    const ParamLayout& layout,
    std::vector<double>& grad,
    double* log_post_out
) {
    // Centre the spatial block once, matching compute_log_post; see
    // spatial_field_constraint.h. The returned gradient is with respect to the
    // raw parameters, which is what spatial_gmrf_prior_grad projects for.
    const bool center_spatial = spatial_block_is_centred(data, layout);
    ratiod_constraints::CenteredSpatialParams centering(
        params_in, center_spatial, layout.spatial_start, data.n_spatial_units,
        data.areal_components);
    const std::vector<double>& params = centering.params();

    const bool fuse_lp = (log_post_out != nullptr);
    double obs_log_lik = 0.0;
    int n_params = params.size();
    grad.assign(n_params, 0.0);

    // Extract common parameters
    auto cp = extract_common_params(params, layout);
    const double* beta_num = cp.beta_num;
    const double* beta_denom = cp.beta_denom;
    double sigma_re = cp.sigma_re;
    const double* re = cp.re;
    double phi_num = cp.phi_num;
    double phi_denom = cp.phi_denom;

    // Spatial parameters (ICAR/BYM2 if present)
    double tau_spatial = 0.0;
    const double* spatial_phi = nullptr;
    double sigma_s_bym2 = 0.0, sigma_u_bym2 = 0.0;
    double rho_bym2 = 0.5;
    const double* theta_bym2 = nullptr;
    if (layout.has_spatial) {
        if (!layout.is_bym2) tau_spatial = std::exp(params[layout.log_tau_spatial_idx]);
        spatial_phi = &params[layout.spatial_start];
        if (layout.is_bym2) {
            double sigma_total = std::exp(params[layout.log_sigma_bym2_idx]);
            double logit_rho = params[layout.logit_rho_bym2_idx];
            rho_bym2 = 1.0 / (1.0 + std::exp(-logit_rho));
            sigma_s_bym2 = sigma_total * std::sqrt(rho_bym2);
            sigma_u_bym2 = sigma_total * std::sqrt(1.0 - rho_bym2);
            theta_bym2 = &params[layout.theta_bym2_start];
        }
    }

    // Multiscale temporal parameters
    const auto& mst = data.multiscale_temporal_data;
    int n_trend = layout.trend_end - layout.trend_start;
    int n_seasonal = layout.seasonal_end - layout.seasonal_start;
    int n_short = layout.short_term_end - layout.short_term_start;

    const double* trend = (n_trend > 0) ? &params[layout.trend_start] : nullptr;
    const double* seasonal = (n_seasonal > 0) ? &params[layout.seasonal_start] : nullptr;
    const double* short_term = (n_short > 0) ? &params[layout.short_term_start] : nullptr;

    double sigma2_trend = (n_trend > 0) ? std::exp(params[layout.log_sigma2_trend_idx]) : 1.0;
    double sigma2_seasonal = (n_seasonal > 0) ? std::exp(params[layout.log_sigma2_seasonal_idx]) : 1.0;
    double sigma2_short = (n_short > 0) ? std::exp(params[layout.log_sigma2_short_idx]) : 1.0;
    double rho_short = 0.5;
    if (mst.short_term_type == TemporalType::AR1 && layout.logit_rho_short_idx >= 0) {
        double logit_rho = params[layout.logit_rho_short_idx];
        double u = 1.0 / (1.0 + std::exp(-logit_rho));
        rho_short = 2.0 * u - 1.0;
    }

    // =========================================================================
    // Prior gradients
    // =========================================================================
    beta_gradient_prior(data, layout, beta_num, beta_denom, grad.data());
    re_gradient_prior(data, layout, re, grad.data(), sigma_re);
    phi_gradient_prior(data, layout, phi_num, phi_denom, grad.data());

    // Spatial prior gradients (ICAR/BYM2)
    if (layout.has_spatial && !layout.is_bym2) {
        // PC prior on the ICAR SD.
        grad[layout.log_tau_spatial_idx] = ratiod::math::pc_scale_gradient(1.0 / std::sqrt(tau_spatial), -0.5, data.spatial_sigma_prior_U, data.spatial_sigma_prior_alpha);
    }
    if (layout.is_bym2) {
        // BYM2 Riebler prior gradients
        double sigma_total = sigma_s_bym2 / std::sqrt(rho_bym2);
        grad[layout.log_sigma_bym2_idx] = ratiod::math::pc_scale_gradient(sigma_total, 1.0, data.spatial_sigma_prior_U, data.spatial_sigma_prior_alpha);
        grad[layout.logit_rho_bym2_idx] = 1.0 - 2.0 * rho_bym2;
    }

    // PC priors on multiscale temporal variances + Jacobian (+1 for log_sigma2 = log(exp(·)))
    if (n_trend > 0) {
        grad[layout.log_sigma2_trend_idx] = ratiod::math::pc_prior_grad_log_sigma2(
            sigma2_trend, data.ms_sigma2_trend_prior_U, data.ms_sigma2_trend_prior_alpha);
    }
    if (n_seasonal > 0) {
        grad[layout.log_sigma2_seasonal_idx] = ratiod::math::pc_prior_grad_log_sigma2(
            sigma2_seasonal, data.ms_sigma2_seasonal_prior_U, data.ms_sigma2_seasonal_prior_alpha);
    }
    if (n_short > 0) {
        grad[layout.log_sigma2_short_idx] = ratiod::math::pc_prior_grad_log_sigma2(
            sigma2_short, data.ms_sigma2_short_prior_U, data.ms_sigma2_short_prior_alpha);
    }
    if (mst.short_term_type == TemporalType::AR1 && layout.logit_rho_short_idx >= 0) {
        // The rho prior in the coordinate the sampler moves
        double u = (rho_short + 1.0) / 2.0;
        grad[layout.logit_rho_short_idx] = ratiod_ar1::log_prior_logit_rho_grad(
            u, ratiod_ar1::RHO_PRIOR_A, ratiod_ar1::RHO_PRIOR_B);
    }

    // =========================================================================
    // Likelihood loop
    // =========================================================================
    std::vector<double> grad_trend_lik(n_trend, 0.0);
    std::vector<double> grad_seasonal_lik(n_seasonal, 0.0);
    std::vector<double> grad_short_lik(n_short, 0.0);
    std::vector<double> grad_spatial_lik;
    if (layout.has_spatial) grad_spatial_lik.assign(data.n_spatial_units, 0.0);
    std::vector<double> grad_theta_lik;
    if (layout.is_bym2) grad_theta_lik.assign(data.n_spatial_units, 0.0);

    // Vectorized eta computation
    const int N = data.N;
    const bool is_binomial = (data.model_type == ModelType::BINOMIAL ||
                              data.model_type == ModelType::BETA_BINOMIAL);
    using RowMajorMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using VectorXd = Eigen::VectorXd;

    fill_fixed_eta(vec_grad_ws(), data, beta_num, beta_denom);
    Eigen::Map<Eigen::VectorXd> eta_n(vec_grad_ws().eta_num.data(), N);

    // Pre-assemble temporal effect per time point (avoids per-obs modulo)
    const int n_times = mst.n_times;
    // eta reads the arms' effects, which are the blocks themselves unless the
    // intrinsic arms are sampled in their non-centred coordinate.
    const auto ms_arms = ratiod_temporal::ms_read_arms(
        trend, n_trend, seasonal, n_seasonal, sigma2_trend, sigma2_seasonal, mst);
    std::vector<double> ms_effect_by_time(n_times, 0.0);
    ratiod_temporal::compute_ms_effect_by_time(
        ms_arms.trend, n_trend, ms_arms.seasonal, n_seasonal, short_term, n_short,
        mst.seasonal_period, n_times, ms_effect_by_time.data());

    // Per-obs: add RE + spatial + multiscale temporal effects
    std::vector<int> obs_s_unit(N, -1);
    std::vector<int> obs_t_idx(N, -1);
    for (int i = 0; i < N; i++) {
        if (layout.has_re && data.re_group[i] > 0) {
            double re_eff = re_value_for_eta(re, data.re_group[i] - 1, sigma_re, data.re_parameterization);
            vec_grad_ws().eta_num[i] += re_eff;
            if (!is_binomial) vec_grad_ws().eta_denom[i] += re_eff;
        }
        if (layout.has_spatial && data.spatial_group[i] > 0) {
            int s_unit = data.spatial_group[i] - 1;
            obs_s_unit[i] = s_unit;
            double spatial_eff;
            if (layout.is_bym2) {
                spatial_eff = sigma_s_bym2 * data.bym2_scale_factor * spatial_phi[s_unit] + sigma_u_bym2 * theta_bym2[s_unit];
            } else {
                spatial_eff = spatial_phi[s_unit];
            }
            vec_grad_ws().eta_num[i] += spatial_eff;
            if (!is_binomial) vec_grad_ws().eta_denom[i] += spatial_eff;
        }
        if (!mst.time_index.empty() && i < (int)mst.time_index.size() && mst.time_index[i] > 0) {
            int t_idx = mst.time_index[i] - 1;
            obs_t_idx[i] = t_idx;
            vec_grad_ws().eta_num[i] += ms_effect_by_time[t_idx];
            if (!is_binomial && mst.shared) vec_grad_ws().eta_denom[i] += ms_effect_by_time[t_idx];
        }
    }

    {
        double grad_phi_num_lik = 0.0, grad_phi_denom_lik = 0.0;
        vectorized::dispatch_residuals_and_beta_grads(
            data, layout,
            vec_grad_ws().eta_num.data(), vec_grad_ws().eta_denom.data(),
            vec_grad_ws().resid_num.data(), vec_grad_ws().resid_denom.data(),
            grad.data(), grad_phi_num_lik, grad_phi_denom_lik,
            obs_log_lik, fuse_lp, phi_num, phi_denom, vec_grad_ws());
    }

    // Scatter residuals
    for (int i = 0; i < N; i++) {
        double dLL_num = vec_grad_ws().resid_num[i];
        double dLL_denom = vec_grad_ws().resid_denom[i];
        if (layout.has_re && data.re_group[i] > 0) {
            grad[layout.re_start + data.re_group[i] - 1] += dLL_num + dLL_denom;
        }
        double dLL_shared = dLL_num + dLL_denom;
        int s_unit = obs_s_unit[i];
        if (layout.has_spatial && s_unit >= 0) {
            grad_spatial_lik[s_unit] += dLL_shared;
            if (layout.is_bym2) grad_theta_lik[s_unit] += dLL_shared;
        }
        int t_idx = obs_t_idx[i];
        if (t_idx >= 0) {
            double dLL_temporal = mst.shared ? dLL_shared : dLL_num;
            if (trend != nullptr && t_idx < n_trend) grad_trend_lik[t_idx] += dLL_temporal;
            if (seasonal != nullptr && mst.seasonal_period > 0) {
                int s_idx = t_idx % mst.seasonal_period;
                if (s_idx < n_seasonal) grad_seasonal_lik[s_idx] += dLL_temporal;
            }
            if (short_term != nullptr && t_idx < n_short) grad_short_lik[t_idx] += dLL_temporal;
        }
    }

    // Spatial GMRF prior gradients (ICAR/BYM2)
    if (layout.has_spatial) {
        spatial_lik_grad_to_param_scale(data, layout, sigma_s_bym2, sigma_u_bym2,
                                        theta_bym2, grad_theta_lik.data(),
                                        grad_spatial_lik.data(), grad.data());
        spatial_gmrf_prior_grad(data, layout, spatial_phi, spatial_phi,
                                centering.raw_sums(),
                                centering.active(), tau_spatial,
                                sigma_s_bym2, sigma_u_bym2, rho_bym2, 0.0, theta_bym2,
                                grad_spatial_lik.data(), grad.data());
    }

    // =========================================================================
    // Multiscale temporal GMRF prior gradients
    // =========================================================================
    ratiod_temporal_grad::MultiscaleTemporalGradients ms_grads;
    ratiod_temporal_grad::multiscale_temporal_prior_gradients(
        trend, n_trend,
        seasonal, n_seasonal,
        short_term, n_short,
        sigma2_trend, sigma2_seasonal, sigma2_short, rho_short,
        mst, ms_grads);
    double nc_g_log_s2_trend = 0.0, nc_g_log_s2_seasonal = 0.0;
    ratiod_temporal_grad::project_ms_lik_gradients(
        grad_trend_lik.data(), n_trend, grad_seasonal_lik.data(), n_seasonal, mst,
        {trend, ms_arms.trend, sigma2_trend, &nc_g_log_s2_trend},
        {seasonal, ms_arms.seasonal, sigma2_seasonal, &nc_g_log_s2_seasonal});

    for (int t = 0; t < n_trend; t++) grad[layout.trend_start + t] = grad_trend_lik[t] + ms_grads.grad_trend[t];
    for (int t = 0; t < n_seasonal; t++) grad[layout.seasonal_start + t] = grad_seasonal_lik[t] + ms_grads.grad_seasonal[t];
    for (int t = 0; t < n_short; t++) grad[layout.short_term_start + t] = grad_short_lik[t] + ms_grads.grad_short_term[t];
    if (n_trend > 0)
        grad[layout.log_sigma2_trend_idx] += ms_grads.grad_log_sigma2_trend + nc_g_log_s2_trend;
    if (n_seasonal > 0)
        grad[layout.log_sigma2_seasonal_idx] += ms_grads.grad_log_sigma2_seasonal + nc_g_log_s2_seasonal;
    if (n_short > 0) grad[layout.log_sigma2_short_idx] += ms_grads.grad_log_sigma2_short;
    if (mst.short_term_type == TemporalType::AR1 && layout.logit_rho_short_idx >= 0) {
        grad[layout.logit_rho_short_idx] += ms_grads.grad_logit_rho_short;
    }

    // Non-centered RE chain rule transformation
    re_gradient_nc_transform(data, layout, params.data(), grad.data(), sigma_re);

    // params_in, not params: compute_log_post centres its own copy and recovers
    // phi_raw_sums from the raw field, which the centred copy has zeroed.
    if (fuse_lp) *log_post_out = compute_log_post(params_in, data, layout, /*skip_obs_loop=*/true) + obs_log_lik;
}

// =====================================================================
// Spatiotemporal interaction gradient (hand-coded, rows 28-29, 58-59, 90-91)
// Supports Knorr-Held Type I-IV with ICAR spatial + RW1/RW2 temporal
// =====================================================================

void compute_gradient_spatiotemporal_handcoded(
    const std::vector<double>& params_in,
    const ModelData& data,
    const ParamLayout& layout,
    std::vector<double>& grad,
    double* log_post_out
) {
    // Centre the spatial block once, matching compute_log_post; see
    // spatial_field_constraint.h. The returned gradient is with respect to the
    // raw parameters, which is what spatial_gmrf_prior_grad projects for.
    const bool center_spatial = spatial_block_is_centred(data, layout);
    ratiod_constraints::CenteredSpatialParams centering(
        params_in, center_spatial, layout.spatial_start, data.n_spatial_units,
        data.areal_components);
    const std::vector<double>& params = centering.params();

    const bool fuse_lp = (log_post_out != nullptr);
    double obs_log_lik = 0.0;
    int n_params = params.size();
    grad.assign(n_params, 0.0);

    // Extract common parameters
    auto cp = extract_common_params(params, layout);
    const double* beta_num = cp.beta_num;
    const double* beta_denom = cp.beta_denom;
    double sigma_re = cp.sigma_re;
    const double* re = cp.re;
    double phi_num = cp.phi_num;
    double phi_denom = cp.phi_denom;

    // Spatial parameters (ICAR/BYM2)
    double tau_spatial = 0.0;
    const double* spatial_phi = nullptr;
    double sigma_s_bym2 = 0.0, sigma_u_bym2 = 0.0;
    double rho_bym2 = 0.5;
    const double* theta_bym2 = nullptr;
    if (layout.has_spatial) {
        if (!layout.is_bym2) tau_spatial = std::exp(params[layout.log_tau_spatial_idx]);
        spatial_phi = &params[layout.spatial_start];
        if (layout.is_bym2) {
            double sigma_total = std::exp(params[layout.log_sigma_bym2_idx]);
            double logit_rho = params[layout.logit_rho_bym2_idx];
            rho_bym2 = 1.0 / (1.0 + std::exp(-logit_rho));
            sigma_s_bym2 = sigma_total * std::sqrt(rho_bym2);
            sigma_u_bym2 = sigma_total * std::sqrt(1.0 - rho_bym2);
            theta_bym2 = &params[layout.theta_bym2_start];
        }
    }

    // Temporal parameters
    double tau_temporal = 0.0;
    int T_temporal = 0;
    const double* phi_temporal = nullptr;
    TemporalView tview;
    double rho_ar1 = 0.0;
    if (layout.has_temporal) {
        tau_temporal = std::exp(params[layout.log_tau_temporal_idx]);
        T_temporal = layout.temporal_end - layout.temporal_start;
        tview.read(params, data, layout);
        phi_temporal = tview.phi;
        rho_ar1 = read_temporal_rho_ar1(params, data, layout);
    }

    // Spatiotemporal interaction parameters
    const auto& st = data.spatiotemporal_data;
    int ST = st.n_params;
    double tau_st = std::exp(params[layout.log_tau_st_idx]);
    // The interaction's own time-margin correlation. The layout carries one
    // only where the margin is AR1; elsewhere 0 leaves R(rho) at the identity
    // and every term reading it inert.
    double rho_st = (layout.logit_rho_st_idx >= 0)
        ? ratiod_ar1::rho_from_logit(params[layout.logit_rho_st_idx])
        : 0.0;
    double sigma2_st_hsgp = 1.0, lengthscale_st_hsgp = 1.0;
    if (data.st_is_hsgp) {
        sigma2_st_hsgp = std::exp(params[layout.log_sigma2_st_hsgp_idx]);
        lengthscale_st_hsgp = std::exp(params[layout.log_lengthscale_st_hsgp_idx]);
    }

    // The GP types' two ranges. The prior is uniform inside its bounds, so it
    // leaves only the log Jacobian on each; the field's own contribution to
    // both arrives with the interaction's prior below.
    double phi_st_space = 1.0, phi_st_time = 1.0;
    if (layout.is_st_gp) {
        phi_st_space = st_gp_range_value(params[layout.log_phi_st_space_idx], data, true);
        phi_st_time = st_gp_range_value(params[layout.log_phi_st_time_idx], data, false);
    }

    // NC reparameterization: params store z, reconstruct delta
    const bool st_use_nc = ratiod_spatiotemporal::st_noncentered_gmrf(
        data.st_parameterization, data.st_is_hsgp, st);
    const double* z_or_delta = &params[layout.st_delta_start];
    RATIOD_TLS_WORKSPACE(std::vector<double>, st_delta_buf);
    const double* delta;
    double inv_scale = 1.0;
    if (st_use_nc) {
        inv_scale = 1.0 / std::sqrt(tau_st);
        st_delta_buf.resize(ST);
        for (int k = 0; k < ST; k++) st_delta_buf[k] = z_or_delta[k] * inv_scale;
        delta = st_delta_buf.data();
    } else {
        delta = z_or_delta;
    }

    // =========================================================================
    // Prior gradients for base parameters
    // =========================================================================
    beta_gradient_prior(data, layout, beta_num, beta_denom, grad.data());
    re_gradient_prior(data, layout, re, grad.data(), sigma_re);
    phi_gradient_prior(data, layout, phi_num, phi_denom, grad.data());

    // PC prior on the ICAR SD.
    if (layout.has_spatial && !layout.is_bym2) {
        grad[layout.log_tau_spatial_idx] = ratiod::math::pc_scale_gradient(1.0 / std::sqrt(tau_spatial), -0.5, data.spatial_sigma_prior_U, data.spatial_sigma_prior_alpha);
    }
    if (layout.is_bym2) {
        // BYM2 Riebler prior gradients
        double sigma_total = sigma_s_bym2 / std::sqrt(rho_bym2);
        grad[layout.log_sigma_bym2_idx] = ratiod::math::pc_scale_gradient(sigma_total, 1.0, data.spatial_sigma_prior_U, data.spatial_sigma_prior_alpha);
        grad[layout.logit_rho_bym2_idx] = 1.0 - 2.0 * rho_bym2;
    }

    // PC prior on the temporal SD and Beta prior on rho.
    if (layout.has_temporal) {
        tau_temporal_prior_grad(data, layout, tau_temporal, grad.data());
        temporal_rho_prior_grad(data, layout, rho_ar1, grad.data());
    }

    // ST interaction prior on tau_st (PC prior on sigma_st)
    grad[layout.log_tau_st_idx] = ratiod::math::pc_log_prior_log_precision_grad(
        std::log(tau_st), data.st_sigma2_prior_U, data.st_sigma2_prior_alpha);
    // Every interaction type carries one precision, tau_st.

    // ST interaction ranges: uniform inside the bounds, so each carries only
    // its log Jacobian. Outside them the density is -inf and the state is
    // rejected on the value, not steered on the gradient.
    if (layout.is_st_gp) {
        grad[layout.log_phi_st_space_idx] = tulpa::uniform_range_coordinate_score(
            params[layout.log_phi_st_space_idx], data.st_parameterization != 2);
        grad[layout.log_phi_st_time_idx] = tulpa::uniform_range_coordinate_score(
            params[layout.log_phi_st_time_idx], data.st_parameterization != 2);
    }

    // ST interaction prior on rho_st (Beta on u = (rho + 1) / 2)
    if (layout.logit_rho_st_idx >= 0) {
        grad[layout.logit_rho_st_idx] = ratiod_ar1::log_prior_rho_grad(
            rho_st, st.rho_prior_a, st.rho_prior_b);
    }

    // HSGP-ST hyperparameter priors: PC on sigma, LogNormal(0, 1) on the
    // lengthscale. The field's own contribution to both arrives below.
    if (data.st_is_hsgp) {
        grad[layout.log_sigma2_st_hsgp_idx] = ratiod::math::pc_scale_gradient(std::exp(0.5 * params[layout.log_sigma2_st_hsgp_idx]),
            0.5, data.st_sigma2_prior_U, data.st_sigma2_prior_alpha);
        grad[layout.log_lengthscale_st_hsgp_idx] =
            -params[layout.log_lengthscale_st_hsgp_idx];
    }

    // =========================================================================
    // =========================================================================
    // Vectorized likelihood loop
    // =========================================================================
    std::vector<double> grad_spatial_lik;
    if (layout.has_spatial) grad_spatial_lik.assign(data.n_spatial_units, 0.0);
    std::vector<double> grad_theta_lik;
    if (layout.is_bym2) grad_theta_lik.assign(data.n_spatial_units, 0.0);
    std::vector<double> grad_temporal_lik(T_temporal, 0.0);
    std::vector<double> grad_delta_lik(ST, 0.0);

    const int N = data.N;
    const bool is_binomial = (data.model_type == ModelType::BINOMIAL ||
                              data.model_type == ModelType::BETA_BINOMIAL);

    // --- Pass 1: Vectorized eta computation (Eigen matvec + scalar expansion) ---
    using RowMajorMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using VectorXd = Eigen::VectorXd;


    fill_fixed_eta(vec_grad_ws(), data, beta_num, beta_denom);
    Eigen::Map<Eigen::VectorXd> eta_n(vec_grad_ws().eta_num.data(), N);

    // Add RE, spatial, temporal, and ST effects to eta (scalar expansion)
    for (int i = 0; i < N; i++) {
        // RE
        if (layout.has_re && data.re_group[i] > 0) {
            double re_eff = re_value_for_eta(re, data.re_group[i] - 1, sigma_re, data.re_parameterization);
            vec_grad_ws().eta_num[i] += re_eff;
            if (!is_binomial) vec_grad_ws().eta_denom[i] += re_eff;
        }
        // Spatial
        if (layout.has_spatial && data.spatial_group[i] > 0) {
            int s_unit = data.spatial_group[i] - 1;
            double spatial_eff;
            if (layout.is_bym2) {
                spatial_eff = sigma_s_bym2 * data.bym2_scale_factor * spatial_phi[s_unit] + sigma_u_bym2 * theta_bym2[s_unit];
            } else {
                spatial_eff = spatial_phi[s_unit];
            }
            vec_grad_ws().eta_num[i] += spatial_eff;
            if (!is_binomial) vec_grad_ws().eta_denom[i] += spatial_eff;
        }
        // Temporal
        if (layout.has_temporal && !data.temporal_time_idx.empty() &&
            i < (int)data.temporal_time_idx.size() && data.temporal_time_idx[i] > 0) {
            int t = data.temporal_time_idx[i] - 1;
            int g = data.temporal_group_idx[i] - 1;
            int t_base = g * data.n_times + t;
            if (t_base >= 0 && t_base < T_temporal) {
                vec_grad_ws().eta_num[i] += phi_temporal[t_base];
                if (!is_binomial && data.temporal_shared) vec_grad_ws().eta_denom[i] += phi_temporal[t_base];
            }
        }
        // Spatiotemporal interaction
        if (st.st_flat[i] > 0) {
            double st_effect = delta[st.st_flat[i] - 1];
            vec_grad_ws().eta_num[i] += st_effect;
            if (!is_binomial && st.shared) vec_grad_ws().eta_denom[i] += st_effect;
        }
    }

    // --- Pass 2+3: Vectorized residuals + beta grads (template-dispatched) ---
    {
        double grad_phi_num_lik = 0.0, grad_phi_denom_lik = 0.0;
        vectorized::dispatch_residuals_and_beta_grads(
            data, layout,
            vec_grad_ws().eta_num.data(), vec_grad_ws().eta_denom.data(),
            vec_grad_ws().resid_num.data(), vec_grad_ws().resid_denom.data(),
            grad.data(), grad_phi_num_lik, grad_phi_denom_lik,
            obs_log_lik, fuse_lp, phi_num, phi_denom, vec_grad_ws());
    }

    // Scatter residuals to RE, spatial, temporal, and ST gradient buffers
    for (int i = 0; i < N; i++) {
        double dLL_num = vec_grad_ws().resid_num[i];
        double dLL_denom = vec_grad_ws().resid_denom[i];
        double dLL_shared = dLL_num + dLL_denom;

        // RE
        if (layout.has_re && data.re_group[i] > 0)
            grad[layout.re_start + data.re_group[i] - 1] += dLL_shared;

        // Spatial
        if (layout.has_spatial && data.spatial_group[i] > 0) {
            int s_unit = data.spatial_group[i] - 1;
            if (layout.is_bym2) { grad_spatial_lik[s_unit] += dLL_shared; grad_theta_lik[s_unit] += dLL_shared; }
            else { grad_spatial_lik[s_unit] += dLL_shared; }
        }

        // Temporal
        if (layout.has_temporal && !data.temporal_time_idx.empty() &&
            i < (int)data.temporal_time_idx.size() && data.temporal_time_idx[i] > 0) {
            int t = data.temporal_time_idx[i] - 1;
            int g = data.temporal_group_idx[i] - 1;
            int t_base = g * data.n_times + t;
            if (t_base >= 0 && t_base < T_temporal)
                grad_temporal_lik[t_base] += data.temporal_shared ? dLL_shared : dLL_num;
        }

        // Spatiotemporal interaction
        if (st.st_flat[i] > 0) {
            int st_idx = st.st_flat[i] - 1;
            grad_delta_lik[st_idx] += st.shared ? dLL_shared : dLL_num;
        }
    }

    // Spatial GMRF prior gradients (ICAR/BYM2)
    if (layout.has_spatial) {
        spatial_lik_grad_to_param_scale(data, layout, sigma_s_bym2, sigma_u_bym2,
                                        theta_bym2, grad_theta_lik.data(),
                                        grad_spatial_lik.data(), grad.data());
        spatial_gmrf_prior_grad(data, layout, spatial_phi, spatial_phi,
                                centering.raw_sums(),
                                centering.active(), tau_spatial,
                                sigma_s_bym2, sigma_u_bym2, rho_bym2, 0.0, theta_bym2,
                                grad_spatial_lik.data(), grad.data());
    }

    // Temporal GMRF prior gradients
    if (layout.has_temporal && T_temporal > 0) {
        temporal_gmrf_prior_grad(data, layout, tau_temporal, rho_ar1,
                                 tview, T_temporal, grad_temporal_lik.data(), grad.data());
    }

    // =========================================================================
    // Spatiotemporal interaction prior gradients
    // =========================================================================
    // One implementation, shared with compute_gradient_composite
    // (st_prior_grad.h): the interaction's field prior, its sum-to-zero
    // margins, and what both contribute to tau, rho and the HSGP-ST spectral
    // hyperparameters. The log-prior comes back with the gradient so
    // compute_log_post below does not evaluate the Kronecker quadratic form a
    // second time.
    const ratiod_spatiotemporal::StPriorGrad st_pg =
        ratiod_spatiotemporal::st_interaction_prior_grad(
            st, data.st_is_hsgp, data.st_hsgp_data,
            z_or_delta, delta, grad_delta_lik.data(),
            tau_st, rho_st, sigma2_st_hsgp, lengthscale_st_hsgp,
            phi_st_space, phi_st_time,
            st_use_nc, &grad[layout.st_delta_start]);

    double st_lp_accum = st_pg.log_prior;
    grad[layout.log_tau_st_idx] += st_pg.log_tau;
    if (layout.logit_rho_st_idx >= 0) {
        grad[layout.logit_rho_st_idx] += st_pg.logit_rho;
    }
    if (data.st_is_hsgp) {
        grad[layout.log_sigma2_st_hsgp_idx] += st_pg.log_sigma2_hsgp;
        grad[layout.log_lengthscale_st_hsgp_idx] += st_pg.log_lengthscale_hsgp;
    }
    if (layout.is_st_gp) {
        grad[layout.log_phi_st_space_idx] += st_pg.log_phi_space * st_gp_range_slope(
            params[layout.log_phi_st_space_idx], data, true);
        grad[layout.log_phi_st_time_idx] += st_pg.log_phi_time * st_gp_range_slope(
            params[layout.log_phi_st_time_idx], data, false);
    }

    // Non-centered RE chain rule transformation
    re_gradient_nc_transform(data, layout, params.data(), grad.data(), sigma_re);

    // params_in, not params: compute_log_post centres its own copy and recovers
    // phi_raw_sums from the raw field, which the centred copy has zeroed.
    if (fuse_lp) *log_post_out = compute_log_post(params_in, data, layout, /*skip_obs_loop=*/true, &st_lp_accum) + obs_log_lik;
}


}  // namespace ratiod_hmc
