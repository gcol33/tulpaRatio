#include <RcppEigen.h>
#include "hmc_grad_helpers.h"
#include "hmc_gp_collapsed.h"
#include "hmc_icar_collapsed.h"
#include "hmc_tvc_grad.h"
#include "hmc_multiscale_temporal_grad.h"
#include "st_prior_grad.h"
#include "hmc_re_blocks_grad.h"

using namespace Rcpp;

namespace ratiod_hmc {
void compute_gradient_composite(
    const std::vector<double>& params_in,
    const ModelData& data,
    const ParamLayout& layout,
    std::vector<double>& grad,
    double* log_post_out
) {
    // Centre the spatial block once, matching compute_log_post; see
    // spatial_field_constraint.h. The returned gradient is with respect to the
    // raw parameters, which is what spatial_gmrf_prior_grad projects for.
    // Proper CAR is full rank (rho < 1) and already identified, so it is not
    // centered like ICAR/BYM2's rank-deficient Q.
    const bool center_spatial = spatial_block_is_centred(data, layout);
    ratiod_constraints::CenteredSpatialParams centering(
        params_in, center_spatial, layout.spatial_start, data.n_spatial_units,
        data.areal_components);
    const std::vector<double>& params = centering.params();

    const bool fuse_lp = (log_post_out != nullptr) && !layout.has_zi;
    // params_in, not params: compute_log_post centres its own copy and recovers
    // phi_raw_sums from the raw field, which the centred copy has zeroed.
    if (log_post_out && layout.has_zi) *log_post_out = compute_log_post(params_in, data, layout);
    double obs_log_lik = 0.0;
    int n_params = params.size();
    grad.assign(n_params, 0.0);

    const int N = data.N;
    const bool is_binomial = (data.model_type == ModelType::BINOMIAL ||
                              data.model_type == ModelType::BETA_BINOMIAL);

    // =========================================================================
    // Phase 1: Extract all parameters
    // =========================================================================
    auto cp = extract_common_params(params, layout);
    const double* beta_num = cp.beta_num;
    const double* beta_denom = cp.beta_denom;
    double sigma_re = cp.sigma_re;
    const double* re = cp.re;
    double phi_num = cp.phi_num;
    double phi_denom = cp.phi_denom;

    // --- Spatial (ICAR/BYM2/pCAR) ---
    double tau_spatial = 0.0;
    const double* spatial_phi = nullptr;
    const double* spatial_phi_prior = nullptr;
    double sigma_s_bym2 = 0.0, sigma_u_bym2 = 0.0, rho_bym2 = 0.5, rho_car = 0.0;
    const double* theta_bym2 = nullptr;
    // A collapsed field has no latent block for the layout to allocate, so
    // spatial_start is -1 and this arm would index in front of the parameter
    // vector. resolve_gradient_fn does not send a collapsed model here; the
    // condition says so rather than relying on it.
    const bool has_icar_bym2 = layout.has_spatial && !layout.is_hsgp && !layout.is_gp &&
                                !layout.is_multiscale_gp && !layout.has_svc &&
                                !data.icar_collapsed && !data.bym2_collapsed;
    if (has_icar_bym2) {
        if (!layout.is_bym2) tau_spatial = std::exp(params[layout.log_tau_spatial_idx]);
        spatial_phi = &params[layout.spatial_start];
        spatial_phi_prior = spatial_prior_reads_raw(data, layout)
                                ? &params_in[layout.spatial_start]
                                : spatial_phi;
        if (layout.is_car_proper) {
            const double u = 1.0 / (1.0 + std::exp(-params[layout.logit_rho_car_idx]));
            rho_car = data.car_rho_lower + (data.car_rho_upper - data.car_rho_lower) * u;
        }
        if (layout.is_bym2) {
            double sigma_total = std::exp(params[layout.log_sigma_bym2_idx]);
            double logit_rho = params[layout.logit_rho_bym2_idx];
            rho_bym2 = 1.0 / (1.0 + std::exp(-logit_rho));
            sigma_s_bym2 = sigma_total * std::sqrt(rho_bym2);
            sigma_u_bym2 = sigma_total * std::sqrt(1.0 - rho_bym2);
            theta_bym2 = &params[layout.theta_bym2_start];
        }
    }

    // --- GP spatial (NNGP) ---
    // The field, its two hyperparameters and the non-centred forward transform,
    // read off the same ratiod_gp entry points compute_gradient_gp_handcoded
    // uses. Without this arm the composite is not a catch-all for a GP main
    // effect: the block reaches eta in compute_log_post and comes back at zero
    // in the gradient, which is what a GP paired with an interaction, a latent
    // factor or a TVC falls through to.
    RATIOD_TLS_WORKSPACE(ratiod_gp::NNGPNCWorkspace, gp_nc_ws);
    const bool has_nngp_gp = layout.is_gp && data.has_gp && !data.gp_collapsed;
    const int N_gp_comp = has_nngp_gp ? data.gp_data.n_obs : 0;
    double gp_sigma2_comp = 0.0, gp_phi_comp = 0.0;
    bool gp_use_nc = false;
    std::vector<double> gp_w_comp;
    if (has_nngp_gp) {
        gp_sigma2_comp = std::exp(params[layout.log_sigma2_gp_idx]);
        gp_phi_comp = std::exp(params[layout.log_phi_gp_idx]);
        // Outside its uniform bounds the density is -Inf, so the point is
        // rejected whatever the gradient says; return the zero gradient the
        // dedicated GP path returns there rather than a number read off a
        // covariance built at a range the prior excludes.
        if (gp_phi_comp < data.gp_phi_prior_lower || gp_phi_comp > data.gp_phi_prior_upper) {
            if (log_post_out && !layout.has_zi)
                *log_post_out = compute_log_post(params_in, data, layout);
            return;
        }
        gp_use_nc = (data.gp_parameterization == 1);
        gp_w_comp.resize(N_gp_comp);
        if (gp_use_nc) {
            ratiod_gp::nngp_nc_forward(&params[layout.gp_w_start], gp_sigma2_comp,
                                       gp_phi_comp, data.gp_data, gp_nc_ws);
            std::memcpy(gp_w_comp.data(), gp_nc_ws.w.data(), N_gp_comp * sizeof(double));
        } else {
            for (int i = 0; i < N_gp_comp; i++)
                gp_w_comp[i] = params[layout.gp_w_start + i];
        }
    }

    // --- Multi-scale GP spatial (two scales on one HSGP basis) ---
    const bool has_msgp = layout.is_multiscale_gp && data.has_multiscale_gp;
    RATIOD_TLS_WORKSPACE(ratiod_hsgp::HSGPWorkspace, msgp_ws_local);
    RATIOD_TLS_WORKSPACE(ratiod_hsgp::HSGPWorkspace, msgp_ws_regional);
    MultiscaleHSGPView msgp;
    std::vector<double> grad_ms_f_lik;
    if (has_msgp) {
        msgp.read(params, data, layout, msgp_ws_local, msgp_ws_regional);
        grad_ms_f_lik.assign(N, 0.0);
    }

    // --- HSGP spatial ---
    RATIOD_TLS_WORKSPACE(ratiod_hsgp::HSGPWorkspace, hsgp_ws);
    double hsgp_sigma2 = 0.0, hsgp_lengthscale = 0.0;
    const double* hsgp_beta_ptr = nullptr;
    if (layout.is_hsgp && data.has_hsgp) {
        hsgp_sigma2 = std::exp(params[layout.log_sigma2_hsgp_idx]);
        hsgp_lengthscale = std::exp(params[layout.log_lengthscale_hsgp_idx]);
        hsgp_beta_ptr = &params[layout.hsgp_beta_start];
        hsgp_ws.init(data.hsgp_data.n_obs, data.hsgp_data.m_total);
        ratiod_hsgp::hsgp_evaluate_ws(hsgp_beta_ptr, hsgp_sigma2, hsgp_lengthscale,
                                       data.hsgp_data, hsgp_ws);
    }

    // --- Temporal (RW1/RW2/AR1 GMRF) ---
    double tau_temporal = 0.0;
    int T_temporal = 0;
    const double* phi_temporal = nullptr;
    TemporalView tview;
    double rho_ar1 = 0.0;
    const bool has_gmrf_temporal = layout.has_temporal && !layout.is_temporal_gp &&
                                   !layout.has_multiscale_temporal && !layout.has_tvc;
    if (has_gmrf_temporal) {
        tau_temporal = std::exp(params[layout.log_tau_temporal_idx]);
        T_temporal = layout.temporal_end - layout.temporal_start;
        tview.read(params, data, layout);
        phi_temporal = tview.phi;
        rho_ar1 = read_temporal_rho_ar1(params, data, layout);
    }

    // --- Temporal GP ---
    const bool has_temporal_gp = layout.is_temporal_gp && layout.has_temporal;
    int T_gp = 0;
    RATIOD_TLS_WORKSPACE(ratiod_temporal_gp::TemporalGPBlock, tgp_block_composite);
    RATIOD_TLS_WORKSPACE(std::vector<double>, temporal_gp_f);
    if (has_temporal_gp) {
        T_gp = data.n_times;
        tgp_block_composite.forward(&params[layout.temporal_start],
                                    params[layout.log_sigma2_temporal_gp_idx],
                                    params[layout.logit_phi_temporal_gp_idx],
                                    data.temporal_gp_data);
        const int total_gp = data.temporal_gp_data.n_groups * T_gp;
        temporal_gp_f.assign(tgp_block_composite.f(), tgp_block_composite.f() + total_gp);
    }

    // --- TVC ---
    RATIOD_TLS_WORKSPACE(std::vector<double>, tvc_eta_precomp);
    int n_tvc = 0, n_tvc_times = 0, n_tvc_groups = 1, n_w = 0;
    RATIOD_TLS_WORKSPACE(std::vector<double>, tvc_tau_buf);
    RATIOD_TLS_WORKSPACE(std::vector<double>, tvc_rho_buf);
    RATIOD_TLS_WORKSPACE(std::vector<double>, tvc_w_flat_buf);
    if (layout.has_tvc && data.has_tvc) {
        n_tvc = data.tvc_data.n_tvc;
        n_tvc_times = data.tvc_data.n_times;
        n_tvc_groups = data.tvc_data.n_groups;
        n_w = n_tvc_groups * n_tvc * n_tvc_times;

        tvc_tau_buf.resize(n_tvc);
        tvc_rho_buf.resize(n_tvc);
        tvc_w_flat_buf.resize(n_w);

        for (int j = 0; j < n_tvc; j++) {
            tvc_tau_buf[j] = std::exp(params[layout.log_tau_tvc_start + j]);
            if (data.tvc_data.structure == ratiod_temporal::TemporalType::AR1) {
                double logit_rho = params[layout.logit_rho_tvc_start + j];
                double u = 1.0 / (1.0 + std::exp(-logit_rho));
                tvc_rho_buf[j] = 2.0 * u - 1.0;
            } else {
                tvc_rho_buf[j] = 0.0;
            }
        }
        for (int k = 0; k < n_w; k++) tvc_w_flat_buf[k] = params[layout.tvc_w_start + k];

        // Precompute TVC eta contribution
        tvc_eta_precomp.assign(N, 0.0);
        for (int i = 0; i < N; i++) {
            int t = data.tvc_data.time_index[i] - 1;
            int g = data.tvc_data.group_index[i] - 1;
            for (int j = 0; j < n_tvc; j++) {
                int w_idx = (g * n_tvc + j) * n_tvc_times + t;
                tvc_eta_precomp[i] += data.tvc_data.X_tvc[i * n_tvc + j] * tvc_w_flat_buf[w_idx];
            }
        }
    }

    // --- SVC, on either approximation ---
    // The NNGP arm carries the field itself; the HSGP arm carries its basis
    // coefficients. Both reach eta through svc_eta_precomp, so the observation
    // loop reads one quantity either way.
    RATIOD_TLS_WORKSPACE(ratiod_hsgp::HSGPWorkspace, svc_hsgp_ws);
    int n_svc = 0, svc_m_total = 0;
    RATIOD_TLS_WORKSPACE(std::vector<double>, svc_eta_precomp);
    RATIOD_TLS_WORKSPACE(std::vector<double>, svc_f_all);
    const bool has_svc_hsgp = layout.has_svc && data.has_svc && data.svc_is_hsgp;
    const bool has_svc_nngp = layout.has_svc && data.has_svc && !data.svc_is_hsgp;
    const int N_svc = has_svc_nngp ? data.svc_data.n_obs : 0;
    std::vector<double> svc_w_nngp;
    std::vector<double> svc_term_mean;
    std::vector<double> svc_sigma2_c, svc_phi_c;
    if (has_svc_nngp) {
        n_svc = data.svc_data.n_svc;
        svc_w_nngp.resize(static_cast<size_t>(N_svc) * n_svc);
        svc_sigma2_c.resize(n_svc);
        svc_phi_c.resize(n_svc);
        for (int j = 0; j < n_svc; j++) {
            svc_sigma2_c[j] = std::exp(params[layout.log_sigma2_svc_start + j]);
            svc_phi_c[j] = ratiod_svc_ad::svc_range(params[layout.range_svc_start + j],
                                                    data.svc_phi_prior_lower,
                                                    data.svc_phi_prior_upper);
        }
        ratiod_svc::svc_field<ratiod_svc::SVC_GRAD_SLOT>(&params[layout.svc_w_start], svc_sigma2_c.data(),
                              svc_phi_c.data(), data.svc_data, data.svc_gp_view,
                              data.svc_noncentered, svc_w_nngp.data());
        // eta reads the CENTRED field, the NNGP prior below reads w itself.
        svc_term_mean.assign(n_svc, 0.0);
        ratiod_svc::svc_term_means(svc_w_nngp.data(), data.svc_data,
                                   svc_term_mean.data());
        svc_eta_precomp.assign(N, 0.0);
        for (int i = 0; i < N; i++) {
            double eff = 0.0;
            for (int j = 0; j < n_svc; j++)
                eff += data.svc_data.X_svc[i * n_svc + j] *
                       (svc_w_nngp[j * N_svc + i] - svc_term_mean[j]);
            svc_eta_precomp[i] = eff;
        }
    }
    if (has_svc_hsgp) {
        n_svc = data.svc_data.n_svc;
        svc_m_total = data.svc_hsgp_data.m_total;
        svc_hsgp_ws.init(data.svc_hsgp_data.n_obs, svc_m_total);
        svc_eta_precomp.assign(N, 0.0);
        svc_f_all.resize(n_svc * N);

        for (int j = 0; j < n_svc; j++) {
            double sigma2_j = std::exp(params[layout.log_sigma2_svc_start + j]);
            double lengthscale_j = std::exp(params[layout.range_svc_start + j]);
            const double* beta_j = &params[layout.svc_w_start + j * svc_m_total];

            ratiod_hsgp::hsgp_evaluate_ws(beta_j, sigma2_j, lengthscale_j,
                                           data.svc_hsgp_data, svc_hsgp_ws);

            for (int i = 0; i < N; i++) {
                double f_ji = svc_hsgp_ws.hsgp_f[i];
                svc_f_all[j * N + i] = f_ji;
                double x_ij = data.svc_data.X_svc[i * n_svc + j];
                svc_eta_precomp[i] += x_ij * f_ji;
            }
        }
    }

    // --- Latent factors ---
    ratiod_latent::LatentView latent_view;
    RATIOD_TLS_WORKSPACE(std::vector<double>, latent_d_num);
    RATIOD_TLS_WORKSPACE(std::vector<double>, latent_d_denom);
    if (layout.has_latent && data.latent_n_factors > 0) {
        latent_view.read(params.data(), layout.latent_loading_start,
                         layout.latent_factor_start, N, data.latent_n_factors,
                         data.latent_shared,
                         static_cast<ratiod_latent::LatentConstraint>(data.latent_constraint));
        latent_d_num.assign(N, 0.0);
        latent_d_denom.assign(N, 0.0);
    }

    // --- Spatiotemporal ---
    const bool has_st = layout.has_spatiotemporal &&
                        data.spatiotemporal_data.type != STType::NONE &&
                        layout.st_delta_start >= 0 && layout.log_tau_st_idx >= 0;
    double tau_st = 0.0;
    double rho_st = 0.0;
    double sigma2_st_hsgp = 1.0, lengthscale_st_hsgp = 1.0;
    double phi_st_space = 1.0, phi_st_time = 1.0;
    const double* st_delta = nullptr;
    RATIOD_TLS_WORKSPACE(std::vector<double>, st_delta_buf);
    bool st_use_nc = false;
    double inv_scale_st = 1.0;
    const double* z_or_delta_st = nullptr;
    int ST_n = 0;
    if (has_st) {
        const auto& st = data.spatiotemporal_data;
        ST_n = st.n_params;
        tau_st = std::exp(params[layout.log_tau_st_idx]);
        // The layout carries a time-margin correlation only where the margin
        // is AR1; elsewhere 0 leaves R(rho) at the identity.
        if (layout.logit_rho_st_idx >= 0)
            rho_st = ratiod_ar1::rho_from_logit(params[layout.logit_rho_st_idx]);
        if (data.st_is_hsgp) {
            sigma2_st_hsgp = std::exp(params[layout.log_sigma2_st_hsgp_idx]);
            lengthscale_st_hsgp = std::exp(params[layout.log_lengthscale_st_hsgp_idx]);
        }
        // The GP types' two ranges. The prior is uniform inside its bounds, so it
        // leaves only the log Jacobian on each; the field's own contribution to
        // both arrives with the interaction's prior below.
        if (layout.is_st_gp) {
            phi_st_space = std::exp(params[layout.log_phi_st_space_idx]);
            phi_st_time = std::exp(params[layout.log_phi_st_time_idx]);
        }
        st_use_nc = ratiod_spatiotemporal::st_noncentered_gmrf(
            data.st_parameterization, data.st_is_hsgp, st);
        z_or_delta_st = &params[layout.st_delta_start];
        if (st_use_nc) {
            inv_scale_st = 1.0 / std::sqrt(tau_st);
            st_delta_buf.resize(ST_n);
            for (int k = 0; k < ST_n; k++) st_delta_buf[k] = z_or_delta_st[k] * inv_scale_st;
            st_delta = st_delta_buf.data();
        } else {
            st_delta = z_or_delta_st;
        }
    }

    // A collapsed GP interaction keeps its three hyperparameters and no field
    const bool st_lap = layout.st_collapsed;
    if (st_lap) {
        tau_st = std::exp(params[layout.log_tau_st_idx]);
        phi_st_space = std::exp(params[layout.log_phi_st_space_idx]);
        phi_st_time = std::exp(params[layout.log_phi_st_time_idx]);
        if (phi_st_space < data.st_phi_space_prior_lower ||
            phi_st_space > data.st_phi_space_prior_upper ||
            phi_st_time < data.st_phi_time_prior_lower ||
            phi_st_time > data.st_phi_time_prior_upper) {
            if (log_post_out) *log_post_out = -INFINITY;
            return;
        }
    }

    // --- Random effects, one block per term ---
    // The same prior phase, eta contribution and chain rule
    // compute_gradient_analytical uses, from hmc_re_blocks_grad.h: crossed and
    // nested terms, random slopes, correlated or not, centred or not. The
    // legacy single-term block is the one-term case of it.
    ReBlockGrad re_ws;

    // --- Multiscale temporal ---
    const bool has_ms_temporal = layout.has_multiscale_temporal;
    int n_trend = 0, n_seasonal = 0, n_short = 0;
    const double* trend = nullptr;
    const double* seasonal = nullptr;
    const double* short_term = nullptr;
    double sigma2_trend = 1.0, sigma2_seasonal = 1.0, sigma2_short = 1.0;
    double rho_short = 0.5;
    RATIOD_TLS_WORKSPACE(std::vector<double>, ms_effect_by_time_c);
    ratiod_temporal::MSArmEffects ms_arms_c;
    RATIOD_TLS_WORKSPACE(std::vector<double>, grad_trend_lik_c);
    RATIOD_TLS_WORKSPACE(std::vector<double>, grad_seasonal_lik_c);
    RATIOD_TLS_WORKSPACE(std::vector<double>, grad_short_lik_c);
    RATIOD_TLS_WORKSPACE(std::vector<int>, obs_t_idx_ms_c);
    if (has_ms_temporal) {
        const auto& mst = data.multiscale_temporal_data;
        n_trend = layout.trend_end - layout.trend_start;
        n_seasonal = layout.seasonal_end - layout.seasonal_start;
        n_short = layout.short_term_end - layout.short_term_start;
        if (n_trend > 0) { trend = &params[layout.trend_start]; sigma2_trend = std::exp(params[layout.log_sigma2_trend_idx]); }
        if (n_seasonal > 0) { seasonal = &params[layout.seasonal_start]; sigma2_seasonal = std::exp(params[layout.log_sigma2_seasonal_idx]); }
        if (n_short > 0) { short_term = &params[layout.short_term_start]; sigma2_short = std::exp(params[layout.log_sigma2_short_idx]); }
        if (mst.short_term_type == TemporalType::AR1 && layout.logit_rho_short_idx >= 0) {
            double u = 1.0 / (1.0 + std::exp(-params[layout.logit_rho_short_idx]));
            rho_short = 2.0 * u - 1.0;
        }
        // Pre-assemble temporal effect per time point. eta reads the arms'
        // effects, which are the blocks unless the intrinsic arms are sampled
        // in their non-centred coordinate.
        ms_arms_c = ratiod_temporal::ms_read_arms(
            trend, n_trend, seasonal, n_seasonal, sigma2_trend, sigma2_seasonal, mst);
        ms_effect_by_time_c.assign(mst.n_times, 0.0);
        ratiod_temporal::compute_ms_effect_by_time(
            ms_arms_c.trend, n_trend, ms_arms_c.seasonal, n_seasonal,
            short_term, n_short,
            mst.seasonal_period, mst.n_times, ms_effect_by_time_c.data());
        grad_trend_lik_c.assign(n_trend, 0.0);
        grad_seasonal_lik_c.assign(n_seasonal, 0.0);
        grad_short_lik_c.assign(n_short, 0.0);
        obs_t_idx_ms_c.assign(N, -1);
    }

    // =========================================================================
    // Phase 2: Prior gradients (independent of data, computed first)
    // =========================================================================
    beta_gradient_prior(data, layout, beta_num, beta_denom, grad.data());
    if (!re_blocks_prior_grad(params, data, layout, grad, re_ws)) return;
    phi_gradient_prior(data, layout, phi_num, phi_denom, grad.data());
    inflation_prior_grad(data, layout, params.data(), grad.data());

    // ICAR/BYM2/pCAR spatial priors
    if (has_icar_bym2 && !layout.is_bym2) {
        grad[layout.log_tau_spatial_idx] = ratiod::math::pc_scale_gradient(1.0 / std::sqrt(tau_spatial), -0.5, data.spatial_sigma_prior_U, data.spatial_sigma_prior_alpha);
    }
    if (has_icar_bym2 && layout.is_bym2) {
        double sigma_total = sigma_s_bym2 / std::sqrt(rho_bym2);
        grad[layout.log_sigma_bym2_idx] = ratiod::math::pc_scale_gradient(sigma_total, 1.0, data.spatial_sigma_prior_U, data.spatial_sigma_prior_alpha);
        grad[layout.logit_rho_bym2_idx] = 1.0 - 2.0 * rho_bym2;
    }
    if (has_icar_bym2 && layout.is_car_proper) {
        // Beta(a, b) on rho with the logit-transform Jacobian folded in; see
        // the matching derivation in compute_log_post.
        const double u = (rho_car - data.car_rho_lower) / (data.car_rho_upper - data.car_rho_lower);
        grad[layout.logit_rho_car_idx] = data.car_rho_prior_a * (1.0 - u) - data.car_rho_prior_b * u;
    }

    // GP (NNGP) priors: PC on sigma2, uniform on phi (log-transform Jacobian
    // only). Under the non-centred parameterization the field's own prior is
    // N(0, I) on z and is seeded by nngp_nc_backward, so it is not added twice.
    if (has_nngp_gp) {
        grad[layout.log_sigma2_gp_idx] += ratiod::math::pc_prior_grad_log_sigma2(
            gp_sigma2_comp, data.gp_sigma2_prior_U, data.gp_sigma2_prior_alpha);
        grad[layout.log_phi_gp_idx] += 1.0;
        if (!gp_use_nc) {
            ratiod_gp::NNGPGradients nngp_grads;
            ratiod_gp::gp_nngp_gradients(gp_w_comp, gp_sigma2_comp, gp_phi_comp,
                                         data.gp_data, nngp_grads);
            for (int i = 0; i < N_gp_comp; i++)
                grad[layout.gp_w_start + i] += nngp_grads.grad_w[i];
            grad[layout.log_sigma2_gp_idx] += nngp_grads.grad_log_sigma2;
            grad[layout.log_phi_gp_idx] += nngp_grads.grad_log_phi;
        }
    }

    // Multi-scale GP priors
    if (has_msgp) msgp.add_prior_gradient(data, layout, grad.data());

    // SVC (NNGP) priors: half-Cauchy on each sigma, and the NNGP field prior.
    // Centred, that prior is placed on the UNCENTRED w and so is not
    // projected; non-centred, it is N(0, I) on z and is seeded by
    // svc_field_accumulate below. The likelihood's gradient is grad_svc_w_lik.
    // Each range's uniform prior is u's own log density, added where the
    // range slots are carried onto u (svc_range_grads_to_coord).
    if (has_svc_nngp) {
        for (int j = 0; j < n_svc; j++) {
            grad[layout.log_sigma2_svc_start + j] += ratiod::math::pc_scale_gradient(std::sqrt(svc_sigma2_c[j]), 0.5, data.svc_sigma_prior_U, data.svc_sigma_prior_alpha);
        }
        if (!data.svc_noncentered) {
            RATIOD_TLS_WORKSPACE(ratiod_svc::SVCGradWorkspace, svc_scratch);
            svc_scratch.resize(n_svc, N_svc);
            ratiod_svc::svc_nngp_prior_grads(
                svc_w_nngp.data(), svc_sigma2_c.data(), svc_phi_c.data(),
                data.svc_data, data.svc_gp_view, layout.svc_w_start, layout.log_sigma2_svc_start,
                layout.range_svc_start, svc_scratch, grad.data());
        }
    }

    // HSGP priors
    if (layout.is_hsgp && data.has_hsgp) {
        grad[layout.log_sigma2_hsgp_idx] = ratiod::math::pc_scale_gradient(std::exp(0.5 * params[layout.log_sigma2_hsgp_idx]),
            0.5, data.gp_sigma2_prior_U, data.gp_sigma2_prior_alpha);
        double log_ls = params[layout.log_lengthscale_hsgp_idx];
        grad[layout.log_lengthscale_hsgp_idx] = -log_ls;  // LogNormal(0,1) prior
        // N(0,I) prior on beta
        int M = data.hsgp_data.m_total;
        for (int j = 0; j < M; j++)
            grad[layout.hsgp_beta_start + j] = -hsgp_beta_ptr[j];
    }

    // Temporal GMRF priors
    if (has_gmrf_temporal) {
        tau_temporal_prior_grad(data, layout, tau_temporal, grad.data());
        temporal_rho_prior_grad(data, layout, rho_ar1, grad.data());
    }

    // PC priors on TVC SDs, differentiated on log precision.
    if (layout.has_tvc && data.has_tvc) {
        for (int j = 0; j < n_tvc; j++) {
            grad[layout.log_tau_tvc_start + j] = ratiod::math::pc_scale_gradient(1.0 / std::sqrt(tvc_tau_buf[j]), -0.5, data.tvc_sigma_prior_U, data.tvc_sigma_prior_alpha);
            if (data.tvc_data.structure == ratiod_temporal::TemporalType::AR1) {
                double u = (tvc_rho_buf[j] + 1.0) / 2.0;
                grad[layout.logit_rho_tvc_start + j] = ratiod_ar1::log_prior_logit_rho_grad(
                    u, ratiod_ar1::RHO_PRIOR_A, ratiod_ar1::RHO_PRIOR_B);
            }
        }
    }

    // SVC-HSGP priors: PC prior on sigma, LogNormal(0,1) on lengthscale, N(0,I) on beta
    if (layout.has_svc && data.has_svc && data.svc_is_hsgp) {
        for (int j = 0; j < n_svc; j++) {
            grad[layout.log_sigma2_svc_start + j] = ratiod::math::pc_scale_gradient(std::exp(0.5 * params[layout.log_sigma2_svc_start + j]),
                0.5, data.svc_sigma_prior_U, data.svc_sigma_prior_alpha);
            // LogNormal(0,1) on lengthscale: d/d(log_ls) [-0.5*log_ls^2] = -log_ls
            double log_ls_j = params[layout.range_svc_start + j];
            grad[layout.range_svc_start + j] = -log_ls_j;
            // N(0,I) prior on SVC beta
            for (int m = 0; m < svc_m_total; m++)
                grad[layout.svc_w_start + j * svc_m_total + m] = -params[layout.svc_w_start + j * svc_m_total + m];
        }
    }

    // Multiscale temporal priors (PC priors on variances + AR1 rho prior)
    if (has_ms_temporal) {
        const auto& mst = data.multiscale_temporal_data;
        if (n_trend > 0)
            grad[layout.log_sigma2_trend_idx] = ratiod::math::pc_prior_grad_log_sigma2(
                sigma2_trend, data.ms_sigma2_trend_prior_U, data.ms_sigma2_trend_prior_alpha);
        if (n_seasonal > 0)
            grad[layout.log_sigma2_seasonal_idx] = ratiod::math::pc_prior_grad_log_sigma2(
                sigma2_seasonal, data.ms_sigma2_seasonal_prior_U, data.ms_sigma2_seasonal_prior_alpha);
        if (n_short > 0)
            grad[layout.log_sigma2_short_idx] = ratiod::math::pc_prior_grad_log_sigma2(
                sigma2_short, data.ms_sigma2_short_prior_U, data.ms_sigma2_short_prior_alpha);
        if (mst.short_term_type == TemporalType::AR1 && layout.logit_rho_short_idx >= 0) {
            double u = (rho_short + 1.0) / 2.0;
            grad[layout.logit_rho_short_idx] = ratiod_ar1::log_prior_logit_rho_grad(
                u, ratiod_ar1::RHO_PRIOR_A, ratiod_ar1::RHO_PRIOR_B);
        }
    }

    // ST interaction hyperparameter priors: PC on sigma_st, Beta on the time
    // margin's correlation, and for HSGP-ST a PC on sigma and a LogNormal(0, 1)
    // on the lengthscale. The field's own contribution to each arrives below.
    if (st_lap) {
        // The same priors on the collapsed interaction's hyperparameters, and
        // the margin penalty's normalizer the density keeps
        const ratiod_spatiotemporal::StGpMarginNormalizer nz =
            ratiod_spatiotemporal::st_gp_margin_normalizer_grad(
                tau_st, phi_st_space, phi_st_time, data.spatiotemporal_data);
        grad[layout.log_tau_st_idx] = ratiod::math::pc_log_prior_log_precision_grad(
            std::log(tau_st), data.st_sigma2_prior_U, data.st_sigma2_prior_alpha)
            + nz.log_tau;
        grad[layout.log_phi_st_space_idx] = 1.0 + nz.log_phi_space;
        grad[layout.log_phi_st_time_idx] = 1.0 + nz.log_phi_time;
    }
    if (has_st) {
        grad[layout.log_tau_st_idx] = ratiod::math::pc_log_prior_log_precision_grad(
            std::log(tau_st), data.st_sigma2_prior_U, data.st_sigma2_prior_alpha);
        if (layout.is_st_gp) {
            // Uniform inside the bounds: each range carries only its log
            // Jacobian, the same term the density adds.
            grad[layout.log_phi_st_space_idx] = 1.0;
            grad[layout.log_phi_st_time_idx] = 1.0;
        }
        if (layout.logit_rho_st_idx >= 0) {
            grad[layout.logit_rho_st_idx] = ratiod_ar1::log_prior_rho_grad(
                rho_st, data.spatiotemporal_data.rho_prior_a,
                data.spatiotemporal_data.rho_prior_b);
        }
        if (data.st_is_hsgp) {
            grad[layout.log_sigma2_st_hsgp_idx] = ratiod::math::pc_scale_gradient(std::exp(0.5 * params[layout.log_sigma2_st_hsgp_idx]),
                0.5, data.st_sigma2_prior_U, data.st_sigma2_prior_alpha);
            grad[layout.log_lengthscale_st_hsgp_idx] =
                -params[layout.log_lengthscale_st_hsgp_idx];
        }
    }

    // =========================================================================
    // Phase 3: Observation loop — compute eta, residuals, scatter gradients
    // =========================================================================
    RATIOD_TLS_WORKSPACE(std::vector<double>, grad_spatial_lik);
    RATIOD_TLS_WORKSPACE(std::vector<double>, grad_theta_lik);
    RATIOD_TLS_WORKSPACE(std::vector<double>, grad_temporal_lik);
    RATIOD_TLS_WORKSPACE(std::vector<double>, grad_hsgp_f);
    RATIOD_TLS_WORKSPACE(std::vector<double>, grad_delta_lik);
    RATIOD_TLS_WORKSPACE(std::vector<double>, grad_tvc_w);
    RATIOD_TLS_WORKSPACE(std::vector<double>, grad_svc_f);
    RATIOD_TLS_WORKSPACE(std::vector<double>, grad_svc_w_lik);
    RATIOD_TLS_WORKSPACE(std::vector<double>, grad_gp_w_lik);

    if (has_icar_bym2) grad_spatial_lik.assign(data.n_spatial_units, 0.0);
    if (has_icar_bym2 && layout.is_bym2) grad_theta_lik.assign(data.n_spatial_units, 0.0);
    if (has_gmrf_temporal) grad_temporal_lik.assign(T_temporal, 0.0);
    if (has_temporal_gp) grad_temporal_lik.assign(data.temporal_gp_data.n_groups * T_gp, 0.0);
    if (layout.is_hsgp && data.has_hsgp) grad_hsgp_f.assign(N, 0.0);
    if (has_nngp_gp) grad_gp_w_lik.assign(N_gp_comp, 0.0);
    if (has_st) grad_delta_lik.assign(ST_n, 0.0);
    if (layout.has_tvc && data.has_tvc) grad_tvc_w.assign(n_w, 0.0);
    if (layout.has_svc && data.has_svc && data.svc_is_hsgp) grad_svc_f.assign(n_svc * N, 0.0);
    if (has_svc_nngp) grad_svc_w_lik.assign(static_cast<size_t>(n_svc) * N_svc, 0.0);

    // Phi likelihood gradient accumulators
    double grad_phi_num_lik_comp = 0.0;
    double grad_phi_denom_lik_comp = 0.0;

    // One observation's two linear predictors, every block included
    auto eta_at = [&](int i, double& eta_num_i, double& eta_denom_i) {
        eta_num_i = 0.0;
        eta_denom_i = 0.0;
        for (int j = 0; j < data.p_num; j++) eta_num_i += data.X_num_flat[i * data.p_num + j] * beta_num[j];
        if (!is_binomial)
            for (int j = 0; j < data.p_denom; j++) eta_denom_i += data.X_denom_flat[i * data.p_denom + j] * beta_denom[j];

        // Random effects
        {
            const double re_eta_i = re_blocks_eta(params, data, layout, re_ws, i);
            eta_num_i += re_eta_i;
            if (!is_binomial) eta_denom_i += re_eta_i;
        }

        // ICAR/BYM2 spatial
        if (has_icar_bym2 && data.spatial_group[i] > 0) {
            int s_unit = data.spatial_group[i] - 1;
            double spatial_eff;
            if (layout.is_bym2)
                spatial_eff = sigma_s_bym2 * data.bym2_scale_factor * spatial_phi[s_unit] + sigma_u_bym2 * theta_bym2[s_unit];
            else
                spatial_eff = spatial_phi[s_unit];
            eta_num_i += spatial_eff;
            if (!is_binomial) eta_denom_i += spatial_eff;
        }

        // HSGP spatial
        if (layout.is_hsgp && data.has_hsgp) {
            double hsgp_eff = hsgp_ws.hsgp_f[i];
            eta_num_i += hsgp_eff;
            if (!is_binomial && data.hsgp_data.shared) eta_denom_i += hsgp_eff;
        }

        // Multi-scale GP spatial
        if (has_msgp) {
            const double ms_eff = msgp.effect(i);
            eta_num_i += ms_eff;
            if (!is_binomial && msgp.shared) eta_denom_i += ms_eff;
        }

        // GP spatial (NNGP)
        if (has_nngp_gp) {
            double gp_eff = gp_w_comp[data.gp_data.obs_to_loc[i]];
            eta_num_i += gp_eff;
            if (!is_binomial && data.gp_data.shared) eta_denom_i += gp_eff;
        }

        // Temporal GMRF
        if (has_gmrf_temporal && !data.temporal_time_idx.empty() &&
            i < (int)data.temporal_time_idx.size() && data.temporal_time_idx[i] > 0) {
            int t = data.temporal_time_idx[i] - 1;
            int g = data.temporal_group_idx[i] - 1;
            int t_base = g * data.n_times + t;
            if (t_base >= 0 && t_base < T_temporal) {
                eta_num_i += phi_temporal[t_base];
                if (!is_binomial && data.temporal_shared) eta_denom_i += phi_temporal[t_base];
            }
        }

        // Temporal GP
        if (has_temporal_gp && !data.temporal_time_idx.empty() &&
            i < (int)data.temporal_time_idx.size() && data.temporal_time_idx[i] > 0) {
            int t = data.temporal_time_idx[i] - 1;
            int g = data.temporal_group_idx[i] - 1;
            int t_base = g * T_gp + t;
            if (t_base >= 0 && t_base < (int)temporal_gp_f.size()) {
                eta_num_i += temporal_gp_f[t_base];
                if (!is_binomial && data.temporal_shared) eta_denom_i += temporal_gp_f[t_base];
            }
        }

        // TVC
        if (layout.has_tvc && data.has_tvc) {
            eta_num_i += tvc_eta_precomp[i];
            if (!is_binomial) eta_denom_i += tvc_eta_precomp[i];
        }

        // SVC, on either approximation
        if (has_svc_hsgp || has_svc_nngp) {
            eta_num_i += svc_eta_precomp[i];
            if (!is_binomial) eta_denom_i += svc_eta_precomp[i];
        }

        // Latent factors
        if (latent_view.active()) {
            eta_num_i += latent_view.eta_num[i];
            eta_denom_i += latent_view.denom(i);
        }

        // Multiscale temporal
        if (has_ms_temporal) {
            const auto& mst = data.multiscale_temporal_data;
            if (!mst.time_index.empty() && i < (int)mst.time_index.size() && mst.time_index[i] > 0) {
                int t_idx = mst.time_index[i] - 1;
                obs_t_idx_ms_c[i] = t_idx;
                eta_num_i += ms_effect_by_time_c[t_idx];
                if (!is_binomial && mst.shared) eta_denom_i += ms_effect_by_time_c[t_idx];
            }
        }

        // Spatiotemporal
        if (has_st) {
            double st_eff = 0.0;
            if (data.st_is_hsgp) {
                // HSGP-ST: Phi-weighted basis-temporal interaction
                int t = data.spatiotemporal_data.t_idx[i] - 1;
                int M_st = data.st_hsgp_data.m_total;
                int T_st_c = data.spatiotemporal_data.n_times;
                for (int j = 0; j < M_st; j++)
                    st_eff += data.st_hsgp_data.phi_flat[i * M_st + j] * st_delta[j * T_st_c + t];
            } else if (data.spatiotemporal_data.st_flat[i] > 0) {
                st_eff = st_delta[data.spatiotemporal_data.st_flat[i] - 1];
            }
            eta_num_i += st_eff;
            if (!is_binomial && data.spatiotemporal_data.shared) eta_denom_i += st_eff;
        }
    };

    // The collapsed GP interaction: every other block reaches the Laplace
    // marginal as offsets, and its gradient comes back as per-observation
    // residuals in place of the family's (st_gp_laplace.h)
    ratiod_st_laplace::Gradient st_lap_grad;
    if (st_lap) {
        std::vector<double> off_num(N), off_denom(N);
        for (int i = 0; i < N; i++) eta_at(i, off_num[i], off_denom[i]);
        if (!ratiod_st_laplace::gradient(data, layout, params, off_num.data(),
                                         off_denom.data(), phi_num, phi_denom,
                                         layout.has_phi_num, layout.has_phi_denom,
                                         st_lap_grad)) {
            if (log_post_out) *log_post_out = -INFINITY;
            return;
        }
        grad_phi_num_lik_comp = st_lap_grad.phi_num;
        grad_phi_denom_lik_comp = st_lap_grad.phi_denom;
        grad[layout.log_tau_st_idx] -= st_lap_grad.log_sigma2;
        grad[layout.log_phi_st_space_idx] += st_lap_grad.log_phi_space;
        grad[layout.log_phi_st_time_idx] += st_lap_grad.log_phi_time;
    }

    // Scalar observation loop (handles all features uniformly)
    for (int i = 0; i < N; i++) {
        double eta_num_i = 0.0, eta_denom_i = 0.0;
        eta_at(i, eta_num_i, eta_denom_i);

        // Compute residuals
        double dLL_num = 0.0, dLL_denom = 0.0;
        if (st_lap) {
            dLL_num = st_lap_grad.e_num[i];
            dLL_denom = st_lap_grad.e_denom[i];
        } else {
            const ratiod::ObsDerivs od = obs_likelihood_derivs(
                data, layout, params.data(), i, eta_num_i, eta_denom_i, phi_num,
                phi_denom, grad.data());
            dLL_num = od.eta_num;
            dLL_denom = od.eta_denom;
            grad_phi_num_lik_comp += od.phi_num;
            grad_phi_denom_lik_comp += od.phi_denom;
        }

        double dLL_shared = dLL_num + dLL_denom;

        // Scatter to beta
        for (int j = 0; j < data.p_num; j++) grad[layout.beta_num_start + j] += dLL_num * data.X_num_flat[i * data.p_num + j];
        if (!is_binomial)
            for (int j = 0; j < data.p_denom; j++) grad[layout.beta_denom_start + j] += dLL_denom * data.X_denom_flat[i * data.p_denom + j];

        // Scatter to the RE blocks
        re_blocks_scatter(data, layout, re_ws, i, dLL_shared, grad);

        // Scatter to ICAR/BYM2 spatial
        if (has_icar_bym2 && data.spatial_group[i] > 0) {
            int s_unit = data.spatial_group[i] - 1;
            grad_spatial_lik[s_unit] += dLL_shared;
            if (layout.is_bym2) grad_theta_lik[s_unit] += dLL_shared;
        }

        // Scatter to HSGP
        if (layout.is_hsgp && data.has_hsgp) {
            grad_hsgp_f[i] = dLL_num + (data.hsgp_data.shared ? dLL_denom : 0.0);
        }

        // Scatter to the multi-scale GP field
        if (has_msgp) grad_ms_f_lik[i] = msgp.shared ? dLL_shared : dLL_num;

        // Scatter to the GP field, by location rather than by row
        if (has_nngp_gp) {
            grad_gp_w_lik[data.gp_data.obs_to_loc[i]] +=
                data.gp_data.shared ? dLL_shared : dLL_num;
        }

        // Scatter to temporal GMRF
        if (has_gmrf_temporal && !data.temporal_time_idx.empty() &&
            i < (int)data.temporal_time_idx.size() && data.temporal_time_idx[i] > 0) {
            int t = data.temporal_time_idx[i] - 1;
            int g = data.temporal_group_idx[i] - 1;
            int t_base = g * data.n_times + t;
            if (t_base >= 0 && t_base < T_temporal)
                grad_temporal_lik[t_base] += data.temporal_shared ? dLL_shared : dLL_num;
        }

        // Scatter to temporal GP
        if (has_temporal_gp && !data.temporal_time_idx.empty() &&
            i < (int)data.temporal_time_idx.size() && data.temporal_time_idx[i] > 0) {
            int t = data.temporal_time_idx[i] - 1;
            int g = data.temporal_group_idx[i] - 1;
            int t_base = g * T_gp + t;
            if (t_base >= 0 && t_base < (int)temporal_gp_f.size())
                grad_temporal_lik[t_base] += data.temporal_shared ? dLL_shared : dLL_num;
        }

        // Scatter to TVC
        if (layout.has_tvc && data.has_tvc) {
            int t = data.tvc_data.time_index[i] - 1;
            int g = data.tvc_data.group_index[i] - 1;
            for (int j = 0; j < n_tvc; j++) {
                int w_idx = (g * n_tvc + j) * n_tvc_times + t;
                grad_tvc_w[w_idx] += dLL_shared * data.tvc_data.X_tvc[i * n_tvc + j];
            }
        }

        // Scatter to SVC
        if (has_svc_hsgp) {
            for (int j = 0; j < n_svc; j++) {
                double x_ij = data.svc_data.X_svc[i * n_svc + j];
                grad_svc_f[j * N + i] = dLL_shared * x_ij;
            }
        }
        if (has_svc_nngp) {
            const double dLL_svc = data.svc_data.shared ? dLL_shared : dLL_num;
            for (int j = 0; j < n_svc; j++)
                grad_svc_w_lik[j * N_svc + i] +=
                    dLL_svc * data.svc_data.X_svc[i * n_svc + j];
        }

        if (latent_view.active()) {
            latent_d_num[i] = dLL_num;
            latent_d_denom[i] = dLL_denom;
        }

        // Scatter to multiscale temporal
        if (has_ms_temporal && obs_t_idx_ms_c[i] >= 0) {
            const auto& mst = data.multiscale_temporal_data;
            int t_idx = obs_t_idx_ms_c[i];
            double dLL_temporal = mst.shared ? dLL_shared : dLL_num;
            if (trend != nullptr && t_idx < n_trend) grad_trend_lik_c[t_idx] += dLL_temporal;
            if (seasonal != nullptr && mst.seasonal_period > 0) {
                int s_idx = t_idx % mst.seasonal_period;
                if (s_idx < n_seasonal) grad_seasonal_lik_c[s_idx] += dLL_temporal;
            }
            if (short_term != nullptr && t_idx < n_short) grad_short_lik_c[t_idx] += dLL_temporal;
        }

        // Scatter to ST interaction
        if (has_st) {
            double dLL_st = data.spatiotemporal_data.shared ? dLL_shared : dLL_num;
            if (data.st_is_hsgp) {
                int t = data.spatiotemporal_data.t_idx[i] - 1;
                int M_st = data.st_hsgp_data.m_total;
                int T_st_c = data.spatiotemporal_data.n_times;
                for (int j = 0; j < M_st; j++)
                    grad_delta_lik[j * T_st_c + t] += data.st_hsgp_data.phi_flat[i * M_st + j] * dLL_st;
            } else if (data.spatiotemporal_data.st_flat[i] > 0) {
                grad_delta_lik[data.spatiotemporal_data.st_flat[i] - 1] += dLL_st;
            }
        }
    }

    // Add phi likelihood gradient contributions (multiply by phi for log-scale Jacobian)
    if (layout.has_phi_num) {
        grad[layout.log_phi_num_idx] += grad_phi_num_lik_comp * phi_num;
    }
    if (layout.has_phi_denom) {
        grad[layout.log_phi_denom_idx] += grad_phi_denom_lik_comp * phi_denom;
    }

    // =========================================================================
    // Phase 4: Structural/GMRF prior gradients (post-observation loop)
    // =========================================================================

    // ICAR/BYM2 spatial GMRF
    if (has_icar_bym2) {
        spatial_lik_grad_to_param_scale(data, layout, sigma_s_bym2, sigma_u_bym2,
                                        theta_bym2,
                                        layout.is_bym2 ? grad_theta_lik.data() : nullptr,
                                        grad_spatial_lik.data(), grad.data());
        spatial_gmrf_prior_grad(data, layout, spatial_phi, spatial_phi_prior,
                                centering.raw_sums(),
                                centering.active(), tau_spatial,
                                sigma_s_bym2, sigma_u_bym2, rho_bym2, rho_car, theta_bym2,
                                grad_spatial_lik.data(), grad.data());
    }

    // Multi-scale GP field: the likelihood through both expansions
    if (has_msgp) msgp.add_lik_gradient(grad_ms_f_lik.data(), data, layout, grad.data());

    // GP (NNGP) field: the likelihood scatter reaches the sampled parameters
    // through the non-centred backward pass, or lands on them directly.
    if (has_nngp_gp) {
        if (gp_use_nc) {
            std::vector<double> grad_z(N_gp_comp, 0.0);
            double grad_log_sigma2_lik = 0.0, grad_log_phi_lik = 0.0, grad_log_phi_jac = 0.0;
            ratiod_gp::nngp_nc_backward(
                &params[layout.gp_w_start], gp_sigma2_comp, gp_phi_comp,
                data.gp_data, gp_nc_ws, grad_gp_w_lik.data(), grad_z.data(),
                grad_log_sigma2_lik, grad_log_phi_lik, grad_log_phi_jac);
            for (int i = 0; i < N_gp_comp; i++)
                grad[layout.gp_w_start + i] += grad_z[i];
            grad[layout.log_sigma2_gp_idx] += grad_log_sigma2_lik;
            grad[layout.log_phi_gp_idx] += grad_log_phi_lik;
        } else {
            for (int i = 0; i < N_gp_comp; i++)
                grad[layout.gp_w_start + i] += grad_gp_w_lik[i];
        }
    }

    if (has_svc_nngp) {
        ratiod_svc::svc_center_terms(grad_svc_w_lik.data(), data.svc_data);
        ratiod_svc::svc_field_accumulate<ratiod_svc::SVC_GRAD_SLOT>(
            &params[layout.svc_w_start], svc_sigma2_c.data(), svc_phi_c.data(),
            data.svc_data, data.svc_gp_view, data.svc_noncentered,
            grad_svc_w_lik.data(), layout.svc_w_start,
            layout.log_sigma2_svc_start, layout.range_svc_start, grad.data());
        svc_range_grads_to_coord(params, data, layout, grad);
    }

    // HSGP spectral density gradients
    if (layout.is_hsgp && data.has_hsgp) {
        int M = data.hsgp_data.m_total;
        std::copy(grad_hsgp_f.begin(), grad_hsgp_f.end(), hsgp_ws.grad_f.begin());
        double grad_log_sigma2 = 0.0, grad_log_lengthscale = 0.0;
        ratiod_hsgp::hsgp_compute_gradients_ws(
            hsgp_beta_ptr, hsgp_sigma2, hsgp_lengthscale,
            data.hsgp_data, hsgp_ws, grad_log_sigma2, grad_log_lengthscale);
        grad[layout.log_sigma2_hsgp_idx] += grad_log_sigma2;
        grad[layout.log_lengthscale_hsgp_idx] += grad_log_lengthscale;
        for (int j = 0; j < M; j++)
            grad[layout.hsgp_beta_start + j] += hsgp_ws.grad_beta_out[j];
    }

    // Temporal GMRF prior
    if (has_gmrf_temporal && T_temporal > 0) {
        temporal_gmrf_prior_grad(data, layout, tau_temporal, rho_ar1,
                                 tview, T_temporal, grad_temporal_lik.data(), grad.data());
    }

    // Temporal GP: the block's prior and the likelihood through its effects
    if (has_temporal_gp) {
        tgp_block_composite.backward(grad_temporal_lik.data(), &grad[layout.temporal_start],
                                     grad[layout.log_sigma2_temporal_gp_idx],
                                     grad[layout.logit_phi_temporal_gp_idx]);
    }

    // Multiscale temporal GMRF prior gradients
    if (has_ms_temporal) {
        ratiod_temporal_grad::MultiscaleTemporalGradients ms_grads;
        ratiod_temporal_grad::multiscale_temporal_prior_gradients(
            trend, n_trend, seasonal, n_seasonal, short_term, n_short,
            sigma2_trend, sigma2_seasonal, sigma2_short, rho_short,
            data.multiscale_temporal_data, ms_grads);
        double nc_g_log_s2_trend_c = 0.0, nc_g_log_s2_seasonal_c = 0.0;
        ratiod_temporal_grad::project_ms_lik_gradients(
            grad_trend_lik_c.data(), n_trend, grad_seasonal_lik_c.data(), n_seasonal,
            data.multiscale_temporal_data,
            {trend, ms_arms_c.trend, sigma2_trend, &nc_g_log_s2_trend_c},
            {seasonal, ms_arms_c.seasonal, sigma2_seasonal, &nc_g_log_s2_seasonal_c});
        for (int t = 0; t < n_trend; t++) grad[layout.trend_start + t] = grad_trend_lik_c[t] + ms_grads.grad_trend[t];
        for (int t = 0; t < n_seasonal; t++) grad[layout.seasonal_start + t] = grad_seasonal_lik_c[t] + ms_grads.grad_seasonal[t];
        for (int t = 0; t < n_short; t++) grad[layout.short_term_start + t] = grad_short_lik_c[t] + ms_grads.grad_short_term[t];
        if (n_trend > 0)
            grad[layout.log_sigma2_trend_idx] += ms_grads.grad_log_sigma2_trend + nc_g_log_s2_trend_c;
        if (n_seasonal > 0)
            grad[layout.log_sigma2_seasonal_idx] += ms_grads.grad_log_sigma2_seasonal + nc_g_log_s2_seasonal_c;
        if (n_short > 0) grad[layout.log_sigma2_short_idx] += ms_grads.grad_log_sigma2_short;
        if (data.multiscale_temporal_data.short_term_type == TemporalType::AR1 && layout.logit_rho_short_idx >= 0)
            grad[layout.logit_rho_short_idx] += ms_grads.grad_logit_rho_short;
    }

    // TVC structural prior gradients
    if (layout.has_tvc && data.has_tvc) {
        // Initialize TVC gradient workspace
        RATIOD_TLS_WORKSPACE(ratiod_tvc::TVCGradientWS, tvc_grad_ws);
        RATIOD_TLS_WORKSPACE(std::vector<double>, tvc_grad_w_buf);
        RATIOD_TLS_WORKSPACE(std::vector<double>, tvc_grad_log_tau_buf);
        RATIOD_TLS_WORKSPACE(std::vector<double>, tvc_grad_logit_rho_buf);
        RATIOD_TLS_WORKSPACE(std::vector<double>, tvc_grad_w_jg_buf);
        RATIOD_TLS_WORKSPACE(std::vector<double>, tvc_d_buf);
        tvc_grad_w_buf.assign(n_w, 0.0);
        tvc_grad_log_tau_buf.assign(n_tvc, 0.0);
        tvc_grad_logit_rho_buf.assign(n_tvc, 0.0);
        tvc_grad_w_jg_buf.resize(n_tvc_times);
        tvc_d_buf.resize(n_tvc_times);

        tvc_grad_ws.grad_w = tvc_grad_w_buf.data();
        tvc_grad_ws.grad_log_tau = tvc_grad_log_tau_buf.data();
        tvc_grad_ws.grad_logit_rho = tvc_grad_logit_rho_buf.data();
        tvc_grad_ws.grad_w_jg = tvc_grad_w_jg_buf.data();
        tvc_grad_ws.d_buf = tvc_d_buf.data();
        tvc_grad_ws.n_w = n_w;
        tvc_grad_ws.n_tvc = n_tvc;

        ratiod_tvc::tvc_prior_gradients_ws(
            tvc_w_flat_buf.data(), data.tvc_data,
            tvc_tau_buf.data(), tvc_rho_buf.data(), tvc_grad_ws);

        // Add likelihood + prior to main gradient
        for (int k = 0; k < n_w; k++)
            grad[layout.tvc_w_start + k] += grad_tvc_w[k] + tvc_grad_w_buf[k];
        for (int j = 0; j < n_tvc; j++) {
            grad[layout.log_tau_tvc_start + j] += tvc_grad_log_tau_buf[j];
            if (data.tvc_data.structure == ratiod_temporal::TemporalType::AR1)
                grad[layout.logit_rho_tvc_start + j] += tvc_grad_logit_rho_buf[j];
        }
    }

    // SVC spectral density gradients
    if (layout.has_svc && data.has_svc && data.svc_is_hsgp) {
        for (int j = 0; j < n_svc; j++) {
            double sigma2_j = std::exp(params[layout.log_sigma2_svc_start + j]);
            double lengthscale_j = std::exp(params[layout.range_svc_start + j]);
            const double* beta_j = &params[layout.svc_w_start + j * svc_m_total];

            // Re-evaluate to cache sqrt_S
            ratiod_hsgp::hsgp_evaluate_ws(beta_j, sigma2_j, lengthscale_j,
                                           data.svc_hsgp_data, svc_hsgp_ws);

            // Copy per-SVC-term grad_f
            for (int i = 0; i < N; i++) svc_hsgp_ws.grad_f[i] = grad_svc_f[j * N + i];

            double gls2 = 0.0, gll = 0.0;
            ratiod_hsgp::hsgp_compute_gradients_ws(
                beta_j, sigma2_j, lengthscale_j,
                data.svc_hsgp_data, svc_hsgp_ws, gls2, gll);

            grad[layout.log_sigma2_svc_start + j] += gls2;
            grad[layout.range_svc_start + j] += gll;
            for (int m = 0; m < svc_m_total; m++)
                grad[layout.svc_w_start + j * svc_m_total + m] += svc_hsgp_ws.grad_beta_out[m];
        }
    }

    if (latent_view.active()) {
        latent_view.add_gradient(params.data(), latent_d_num, latent_d_denom,
                                 data.latent_sigma_prior_rate,
                                 data.latent_offdiag_sd, grad.data());
    }

    // ST interaction prior gradients: the same implementation
    // compute_gradient_spatiotemporal_handcoded drives (st_prior_grad.h).
    if (has_st) {
        const ratiod_spatiotemporal::StPriorGrad st_pg =
            ratiod_spatiotemporal::st_interaction_prior_grad(
                data.spatiotemporal_data, data.st_is_hsgp, data.st_hsgp_data,
                z_or_delta_st, st_delta, grad_delta_lik.data(),
                tau_st, rho_st, sigma2_st_hsgp, lengthscale_st_hsgp,
                phi_st_space, phi_st_time,
                st_use_nc, &grad[layout.st_delta_start]);

        grad[layout.log_tau_st_idx] += st_pg.log_tau;
        if (layout.logit_rho_st_idx >= 0) {
            grad[layout.logit_rho_st_idx] += st_pg.logit_rho;
        }
        if (data.st_is_hsgp) {
            grad[layout.log_sigma2_st_hsgp_idx] += st_pg.log_sigma2_hsgp;
            grad[layout.log_lengthscale_st_hsgp_idx] += st_pg.log_lengthscale_hsgp;
        }
        if (layout.is_st_gp) {
            grad[layout.log_phi_st_space_idx] += st_pg.log_phi_space;
            grad[layout.log_phi_st_time_idx] += st_pg.log_phi_time;
        }
    }

    // The chain rule back onto the sampled parameters.
    re_blocks_writeback(params, data, layout, grad, re_ws);

    // Fused log-posterior. params_in, not params: compute_log_post centres its
    // own copy and recovers phi_raw_sums from the raw field, which the centred
    // copy has zeroed.
    if (fuse_lp && !layout.has_zi) {
        *log_post_out = compute_log_post(params_in, data, layout);
    }
}

}  // namespace ratiod_hmc
