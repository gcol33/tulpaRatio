#include <RcppEigen.h>
#include "hmc_grad_helpers.h"
#include "hmc_re_blocks_grad.h"
#include "omp_chain_team.h"

using namespace Rcpp;

namespace ratiod_hmc {
bool can_use_analytical_gradient(const ModelData& data, const ParamLayout& layout) {
  // Hand-coded gradients for basic models without complex structure
  bool is_basic_family = (data.model_type == ModelType::POISSON_GAMMA ||
                          data.model_type == ModelType::NEGBIN_NEGBIN ||
                          data.model_type == ModelType::NEGBIN_GAMMA ||
                          data.model_type == ModelType::BINOMIAL ||
                          data.model_type == ModelType::GAMMA_GAMMA ||
                          data.model_type == ModelType::BETA_BINOMIAL ||
                          data.model_type == ModelType::LOGNORMAL);

  // Check if spatial type is one we have hand-coded gradients for
  bool spatial_is_icar_bym2 = (data.spatial_type == SpatialType::ICAR ||
                               data.spatial_type == SpatialType::BYM2);

  // Temporal is OK alone or combined with ICAR/BYM2 spatial (no spatiotemporal interaction)
  // Note: Temporal GP is excluded - use autodiff for that
  bool temporal_ok = !layout.has_temporal ||
                     (layout.has_temporal && !layout.is_temporal_gp &&
                      !layout.has_spatiotemporal &&
                      (!layout.has_spatial || spatial_is_icar_bym2));

  // Spatial is OK for ICAR/BYM2 (alone or combined with temporal)
  bool spatial_ok = !layout.has_spatial ||
                    (layout.has_spatial && spatial_is_icar_bym2 && !layout.has_spatiotemporal);

  // ZI is OK for basic models, including with ICAR/BYM2 spatial and temporal
  // Components are additive in eta; ZI modifies residuals independently
  bool zi_ok = !layout.has_zi ||
               (layout.has_zi &&
                (!layout.has_spatial || spatial_is_icar_bym2));

  // Random slopes: both correlated (|) and uncorrelated (||) are supported
  // Can combine with ICAR/BYM2 spatial, temporal, and ZI
  // Components are additive in eta; gradient scatter is independent
  bool slopes_ok = !layout.has_re_slopes ||
                   (layout.has_re_slopes &&
                    (!layout.has_spatial || spatial_is_icar_bym2));

  return (is_basic_family &&
          !layout.is_gp && !layout.is_multiscale_gp && !layout.is_hsgp &&
          !layout.is_icar_collapsed && !layout.is_bym2_collapsed &&
          temporal_ok && spatial_ok && zi_ok && slopes_ok &&
          !layout.has_latent && !layout.has_spatiotemporal &&
          !layout.has_multiscale_temporal && !layout.has_tvc &&
          !layout.has_svc &&  // SVC has its own gradient function
          (data.n_re_terms <= 1 ||
           (data.n_re_terms > 1 && !layout.has_re_slopes)));  // Crossed RE (intercept-only) is OK
}

void compute_gradient_analytical(
    const std::vector<double>& params_in,
    const ModelData& data,
    const ParamLayout& layout,
    std::vector<double>& grad,
    double* log_post_out
) {
  // Centre the spatial block once, matching compute_log_post. The vectorized
  // fast paths index params directly, so centring here is what makes them see
  // the constrained field. The returned gradient is with respect to the raw
  // parameters, which is why the spatial likelihood gradient is projected below.
  std::vector<double> params_centered;
  std::vector<double> phi_raw_sums;
  const bool center_spatial = spatial_block_is_centred(data, layout);
  if (center_spatial) {
    params_centered = params_in;
    phi_raw_sums = ratiod_constraints::center_areal_blocks(
        params_centered, layout.spatial_start, data.n_spatial_units,
        data.areal_components);
  }
  const std::vector<double>& params = center_spatial ? params_centered : params_in;

  int n_params = params.size();
  grad.assign(n_params, 0.0);

  // Fused log-posterior computation: accumulate observation log-likelihood
  // alongside gradients to avoid a separate O(N) pass.
  const bool compute_lp = (log_post_out != nullptr);
  double obs_log_lik = 0.0;

  // Extract parameters
  const double* beta_num = &params[layout.beta_num_start];
  const double* beta_denom = &params[layout.beta_denom_start];

  double log_sigma_re = 0.0, sigma_re = 1.0, tau_re = 1.0;
  const double* re = nullptr;
  if (layout.has_re) {
    log_sigma_re = params[layout.log_sigma_re_idx];
    sigma_re = std::exp(log_sigma_re);
    tau_re = 1.0 / (sigma_re * sigma_re + 1e-10);
    re = &params[layout.re_start];
  }

  double phi_num = 1.0, log_phi_num = 0.0;
  double phi_denom = 1.0, log_phi_denom = 0.0;
  if (layout.has_phi_num) {
    log_phi_num = params[layout.log_phi_num_idx];
    phi_num = std::exp(log_phi_num);
  }
  if (layout.has_phi_denom) {
    log_phi_denom = params[layout.log_phi_denom_idx];
    phi_denom = std::exp(log_phi_denom);
  }

  // ============ Prior gradients (cheap) ============

  // Beta priors: N(0, sigma_beta^2)
  beta_gradient_prior(data, layout, beta_num, beta_denom, grad.data());

  // sigma_re: Half-Cauchy prior with scale = data.sigma_re_scale (via log transform)
  // log_post = -log(1 + (sigma/scale)^2) + log(sigma) (Jacobian)
  // d/d(log_sigma) = -2*(sigma/scale)^2/(1+(sigma/scale)^2) + 1
  if (layout.has_re) {
    double ratio = sigma_re / data.sigma_re_scale;
    double ratio_sq = ratio * ratio;
    grad[layout.log_sigma_re_idx] = -2.0 * ratio_sq / (1.0 + ratio_sq) + 1.0;
  }

  // phi priors: Gamma(shape, rate) via log transform
  phi_gradient_prior(data, layout, phi_num, phi_denom, grad.data());

  // RE priors, and the effective effects a non-centred correlated term's
  // loops read. See hmc_re_blocks_grad.h.
  ReBlockGrad re_ws;
  if (!re_blocks_prior_grad(params, data, layout, grad, re_ws)) return;
  std::vector<std::vector<double>>& grad_re_slopes_lik = re_ws.lik;
  const int n_re_terms_slopes = re_ws.n_terms;
  const bool slopes_nc = re_ws.nc;
  const std::vector<double>& re_nc_flat = re_ws.re_nc_flat;

  // ============ Temporal prior gradients ============
  double log_tau_temporal = 0.0, tau_temporal = 1.0;
  double logit_rho_ar1 = 0.0, rho_ar1 = 0.0;
  int T_len = 0;
  const double* phi_temporal = nullptr;
  std::vector<double> grad_temporal_lik;  // Likelihood contribution
  TemporalView tview;

  if (layout.has_temporal) {
    log_tau_temporal = params[layout.log_tau_temporal_idx];
    tau_temporal = std::exp(log_tau_temporal);
    T_len = layout.temporal_end - layout.temporal_start;
    tview.read(params, data, layout);
    phi_temporal = tview.phi;
    grad_temporal_lik.assign(T_len, 0.0);

    // PC prior on the temporal SD, including the log-precision Jacobian.
    tau_temporal_prior_grad(data, layout, tau_temporal, grad.data());

    // AR1: extract rho and add prior
    if (data.temporal_type == TemporalType::AR1 && layout.logit_rho_ar1_idx >= 0) {
      logit_rho_ar1 = params[layout.logit_rho_ar1_idx];
      rho_ar1 = ratiod_ar1::rho_from_logit(logit_rho_ar1);
      temporal_rho_prior_grad(data, layout, rho_ar1, grad.data());
    }
  }

  // ============ Spatial prior gradients (ICAR and BYM2) ============
  double log_tau_spatial = 0.0, tau_spatial = 1.0;
  double sigma_s_bym2 = 1.0, sigma_u_bym2 = 1.0;
  double rho_bym2 = 0.5;  // Riebler mixing parameter
  int n_spatial = 0;
  const double* phi_spatial = nullptr;
  const double* theta_bym2 = nullptr;
  std::vector<double> grad_spatial_lik;  // Likelihood contribution

  if (layout.has_spatial) {
    n_spatial = data.n_spatial_units;
    phi_spatial = &params[layout.spatial_start];
    grad_spatial_lik.assign(n_spatial, 0.0);

    if (data.spatial_type == SpatialType::BYM2) {
      // BYM2 Riebler: derive sigma_s, sigma_u from sigma_total, rho
      double sigma_total = std::exp(params[layout.log_sigma_bym2_idx]);
      double logit_rho = params[layout.logit_rho_bym2_idx];
      rho_bym2 = 1.0 / (1.0 + std::exp(-logit_rho));
      sigma_s_bym2 = sigma_total * std::sqrt(rho_bym2);
      sigma_u_bym2 = sigma_total * std::sqrt(1.0 - rho_bym2);
      theta_bym2 = &params[layout.theta_bym2_start];

      // PC prior on the total spatial SD.
      grad[layout.log_sigma_bym2_idx] = ratiod::math::pc_scale_gradient(sigma_total, 1.0, data.spatial_sigma_prior_U, data.spatial_sigma_prior_alpha);

      // Uniform(0,1) = Beta(1,1) on rho with logit Jacobian:
      // d/d(logit_rho) [log(rho) + log(1-rho)] = (1-rho) - rho = 1 - 2*rho
      grad[layout.logit_rho_bym2_idx] = 1.0 - 2.0 * rho_bym2;

      // Initialize theta gradients (N(0,1) prior: d/d(theta) = -theta)
      for (int s = 0; s < n_spatial; s++) {
        grad[layout.theta_bym2_start + s] = -theta_bym2[s];
      }
    } else {
      // ICAR: extract tau
      log_tau_spatial = params[layout.log_tau_spatial_idx];
      tau_spatial = std::exp(log_tau_spatial);

      // PC prior on the spatial SD, including the log-precision Jacobian.
      grad[layout.log_tau_spatial_idx] = ratiod::math::pc_scale_gradient(1.0 / std::sqrt(tau_spatial), -0.5, data.spatial_sigma_prior_U, data.spatial_sigma_prior_alpha);
    }
  }

  // ============ Zero-inflation prior gradients ============
  const double* beta_zi = nullptr;
  std::vector<double> grad_beta_zi;
  double tau_zi = 1.0;

  if (layout.has_zi && data.p_zi > 0) {
    beta_zi = &params[layout.beta_zi_start];
    tau_zi = 1.0 / (data.zi_prior_sd * data.zi_prior_sd + 1e-10);
    grad_beta_zi.assign(data.p_zi, 0.0);

    // N(0, zi_prior_sd^2) prior on ZI coefficients
    for (int j = 0; j < data.p_zi; j++) {
      grad[layout.beta_zi_start + j] = -tau_zi * beta_zi[j];
    }
  }

  // ============ One-inflation (OI) prior gradients ============
  const double* beta_oi = nullptr;
  std::vector<double> grad_beta_oi;
  double tau_oi = 1.0;

  if (layout.has_oi && data.p_oi > 0) {
    beta_oi = &params[layout.beta_oi_start];
    tau_oi = 1.0 / (data.oi_prior_sd * data.oi_prior_sd + 1e-10);
    grad_beta_oi.assign(data.p_oi, 0.0);

    // N(0, oi_prior_sd^2) prior on OI coefficients
    for (int j = 0; j < data.p_oi; j++) {
      grad[layout.beta_oi_start + j] = -tau_oi * beta_oi[j];
    }
  }

  // ============ Likelihood gradients (O(n)) ============

  // Try fused single-pass gradient first (best for small p <= 4).
  // Then fall back to 3-pass vectorized (better for larger p with Eigen).
  // Finally fall back to scalar loop for complex models (ZI, slopes, etc.).
  bool used_vectorized = vectorized::dispatch_fused_gradient(
      params, data, layout, grad, obs_log_lik,
      compute_lp, grad_temporal_lik, grad_spatial_lik, vec_grad_ws());

  // Fall back to 3-pass vectorized for p > 4 (Eigen matvec is beneficial)
  if (!used_vectorized) {
    used_vectorized = vectorized::dispatch_vectorized_gradient(
        params, data, layout, grad, obs_log_lik,
        compute_lp, grad_temporal_lik, grad_spatial_lik, vec_grad_ws());
  }

  // Hybrid path for slopes models (no ZI/OI): vectorize X*beta + residuals,
  // scalar loop for slopes RE expansion + gradient scatter
  if (!used_vectorized && (layout.has_re_slopes || layout.has_re_correlated_slopes) &&
      !layout.has_zi && !layout.has_oi &&
      !data.has_svc && !data.has_tvc && !data.has_latent &&
      !data.has_spatiotemporal && !data.has_temporal_gp && !data.has_multiscale_temporal &&
      data.spatial_type != SpatialType::GP &&
      data.spatial_type != SpatialType::MULTISCALE_GP &&
      data.spatial_type != SpatialType::HSGP) {

    const int N = data.N;
    const bool is_binomial = (data.model_type == ModelType::BINOMIAL ||
                              data.model_type == ModelType::BETA_BINOMIAL);

    // --- Pass 1: Vectorized eta base (Eigen matvec) ---
    using RowMajorMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using VectorXd = Eigen::VectorXd;

    fill_fixed_eta(vec_grad_ws(), data, beta_num, beta_denom);
    Eigen::Map<Eigen::VectorXd> eta_n(vec_grad_ws().eta_num.data(), N);

    // Pre-compute sigma values for NC slopes (avoid N * n_slopes exp() calls)
    std::vector<std::vector<double>> precomp_sigma(n_re_terms_slopes);
    if (slopes_nc) {
      for (int t = 0; t < n_re_terms_slopes; t++) {
        int n_coefs = layout.re_n_coefs_multi[t];
        precomp_sigma[t].resize(n_coefs);
        for (int c = 0; c < n_coefs; c++) {
          precomp_sigma[t][c] = std::exp(params[layout.log_sigma_re_slopes[t][c]]);
        }
      }
    }

    // Scalar loop: add slopes RE + spatial + temporal to eta
    // Track per-obs indices for scatter pass
    std::vector<int> obs_s_idx(N, -1);       // spatial group index
    std::vector<int> obs_t_idx(N, -1);       // temporal flat index
    for (int i = 0; i < N; i++) {
      // Slopes RE contribution
      if (layout.has_re_slopes && n_re_terms_slopes > 0) {
        int re_group_idx_i = data.re_group_multi_flat[i * data.n_re_terms + 0];
        if (re_group_idx_i > 0) {
          int g = re_group_idx_i - 1;
          int n_coefs = layout.re_n_coefs_multi[0];
          int re_base = layout.re_start_multi[0] + g * n_coefs;

          bool is_corr_t = !re_nc_flat.empty() &&
                           layout.re_correlated_multi.size() > 0 &&
                           layout.re_correlated_multi[0] && n_coefs > 1;
          bool is_uncorr_nc = !is_corr_t && slopes_nc;

          // Intercept
          double re_contrib;
          if (is_corr_t) {
            re_contrib = re_nc_flat[re_base];
          } else if (is_uncorr_nc) {
            re_contrib = precomp_sigma[0][0] * params[re_base];
          } else {
            re_contrib = params[re_base];
          }

          // Slope contributions
          int n_slopes = n_coefs - 1;
          if (n_slopes > 0 && !data.re_slope_matrices[0].empty()) {
            for (int s = 0; s < n_slopes; s++) {
              double x_slope = data.re_slope_matrices[0][i * n_slopes + s];
              double re_slope;
              if (is_corr_t) {
                re_slope = re_nc_flat[re_base + 1 + s];
              } else if (is_uncorr_nc) {
                re_slope = precomp_sigma[0][1 + s] * params[re_base + 1 + s];
              } else {
                re_slope = params[re_base + 1 + s];
              }
              re_contrib += re_slope * x_slope;
            }
          }

          vec_grad_ws().eta_num[i] += re_contrib;
          if (!is_binomial) vec_grad_ws().eta_denom[i] += re_contrib;
        }
      }

      // Spatial effect (ICAR or BYM2 only — GP/HSGP/MSGP excluded above)
      if (layout.has_spatial && !data.spatial_group.empty() && data.spatial_group[i] > 0) {
        int s = data.spatial_group[i] - 1;
        obs_s_idx[i] = s;
        double spatial_eff;
        if (data.spatial_type == SpatialType::BYM2) {
          spatial_eff = sigma_s_bym2 * data.bym2_scale_factor * phi_spatial[s] + sigma_u_bym2 * theta_bym2[s];
        } else {
          spatial_eff = phi_spatial[s];
        }
        vec_grad_ws().eta_num[i] += spatial_eff;
        if (!is_binomial) vec_grad_ws().eta_denom[i] += spatial_eff;
      }

      // Temporal effect
      if (layout.has_temporal && !data.temporal_time_idx.empty() && data.temporal_time_idx[i] > 0) {
        int t = data.temporal_time_idx[i] - 1;
        int g = data.temporal_group_idx[i] - 1;
        int t_flat = g * data.n_times + t;
        obs_t_idx[i] = t_flat;
        vec_grad_ws().eta_num[i] += phi_temporal[t_flat];
        if (!is_binomial) vec_grad_ws().eta_denom[i] += phi_temporal[t_flat];
      }
    }

    // --- Pass 2+3: Vectorized residuals + beta grads (template-dispatched) ---
    {
      double grad_phi_num_lik_v = 0.0, grad_phi_denom_lik_v = 0.0;
      vectorized::dispatch_residuals_and_beta_grads(
          data, layout,
          vec_grad_ws().eta_num.data(), vec_grad_ws().eta_denom.data(),
          vec_grad_ws().resid_num.data(), vec_grad_ws().resid_denom.data(),
          grad.data(), grad_phi_num_lik_v, grad_phi_denom_lik_v,
          obs_log_lik, compute_lp, phi_num, phi_denom, vec_grad_ws());
    }

    // Scatter residuals to slopes RE, spatial, temporal gradient buffers
    for (int i = 0; i < N; i++) {
      double dLL_num = vec_grad_ws().resid_num[i];
      double dLL_denom = vec_grad_ws().resid_denom[i];
      double dLL_shared = dLL_num + dLL_denom;

      // Slopes RE gradient scatter
      if (layout.has_re_slopes && n_re_terms_slopes > 0) {
        int re_group_idx_i = data.re_group_multi_flat[i * data.n_re_terms + 0];
        if (re_group_idx_i > 0) {
          int g = re_group_idx_i - 1;
          int n_coefs = layout.re_n_coefs_multi[0];

          // Intercept gradient
          grad_re_slopes_lik[0][g * n_coefs] += dLL_shared;

          // Slope gradients (chain rule: d(LL)/d(re_slope) = d(LL)/d(eta) * x_slope)
          int n_slopes = n_coefs - 1;
          if (n_slopes > 0 && !data.re_slope_matrices[0].empty()) {
            for (int s = 0; s < n_slopes; s++) {
              double x_slope = data.re_slope_matrices[0][i * n_slopes + s];
              grad_re_slopes_lik[0][g * n_coefs + 1 + s] += dLL_shared * x_slope;
            }
          }
        }
      }

      // Spatial gradient scatter
      if (obs_s_idx[i] >= 0) {
        int s = obs_s_idx[i];
        if (data.spatial_type == SpatialType::BYM2) {
          grad_spatial_lik[s] += dLL_shared * sigma_s_bym2 * data.bym2_scale_factor;
          grad[layout.theta_bym2_start + s] += dLL_shared * sigma_u_bym2;
        } else {
          grad_spatial_lik[s] += dLL_shared;
        }
      }

      // Temporal gradient scatter
      if (obs_t_idx[i] >= 0) {
        grad_temporal_lik[obs_t_idx[i]] += data.temporal_shared ? dLL_shared : dLL_num;
      }
    }

    used_vectorized = true;
  }

  if (!used_vectorized) {

  // Accumulators for beta gradients (will be added via X' * residual)
  std::vector<double> grad_beta_num(data.p_num, 0.0);
  std::vector<double> grad_beta_denom(data.p_denom, 0.0);
  double grad_phi_num_lik = 0.0;
  double grad_phi_denom_lik = 0.0;

  // Every likelihood sum the observation loop builds has a region in each
  // thread's slot of `red_partials`; the slots are added afterwards in
  // thread-index order. Nothing is combined by the runtime: libomp's atomic
  // and tree-reduction paths for doubles drop updates on aarch64-w64-mingw
  // (omp_sum.h), and a critical section or atomic sums in thread-arrival
  // order, which varies from run to run. The buffer is sized by
  // omp_get_max_threads() and the region below carries no num_threads clause,
  // so both read the same value and every thread of the team has a slot. The
  // backends that take a per-fit thread count scope it (omp_thread_scope.h),
  // which holds that value still for the length of a fit.
  #ifdef _OPENMP
  const int grad_team = std::max(1, omp_get_max_threads());
  #else
  const int grad_team = 1;
  #endif
  const bool acc_zi = layout.has_zi && data.p_zi > 0;
  const bool acc_oi = layout.has_oi && data.p_oi > 0;
  const bool acc_slopes = layout.has_re_slopes && n_re_terms_slopes > 0 &&
                          !grad_re_slopes_lik.empty();
  const bool acc_bym2 = layout.has_spatial &&
                        data.spatial_type == SpatialType::BYM2;
  const int n_re_acc = layout.has_re ? data.n_re_groups : 0;
  const int n_re_crossed_acc =
      (layout.has_re && data.n_re_terms > 1) ? data.total_re_groups : 0;
  const int n_slopes_acc =
      acc_slopes ? (int)grad_re_slopes_lik[0].size() : 0;
  struct {
    int beta_num, beta_denom, phi_num, phi_denom, re, re_crossed, ll,
        beta_zi, beta_oi, re_slopes, temporal, spatial, theta_bym2, stride;
  } acc_at;
  {
    int n = 0;
    auto region = [&n](int len) { const int at = n; n += len; return at; };
    acc_at.beta_num = region(data.p_num);
    acc_at.beta_denom = region(data.p_denom);
    acc_at.phi_num = region(1);
    acc_at.phi_denom = region(1);
    acc_at.re = region(n_re_acc);
    acc_at.re_crossed = region(n_re_crossed_acc);
    acc_at.ll = region(1);
    acc_at.beta_zi = region(acc_zi ? data.p_zi : 0);
    acc_at.beta_oi = region(acc_oi ? data.p_oi : 0);
    acc_at.re_slopes = region(n_slopes_acc);
    acc_at.temporal = region((int)grad_temporal_lik.size());
    acc_at.spatial = region((int)grad_spatial_lik.size());
    acc_at.theta_bym2 = region(acc_bym2 ? n_spatial : 0);
    // Whole cache lines per slot, so neighbouring threads share none.
    acc_at.stride = (n + 7) / 8 * 8;
  }
  std::vector<double> red_partials((size_t)grad_team * acc_at.stride, 0.0);

  #ifdef _OPENMP
  #pragma omp parallel
  #endif
  {
    #ifdef _OPENMP
    double* acc = &red_partials[(size_t)omp_get_thread_num() * acc_at.stride];
    #else
    double* acc = red_partials.data();
    #endif
    // Pre-allocated per-obs group index buffer for crossed RE (reused across iterations)
    std::vector<int> re_idx_multi_buf(
        (layout.has_re && data.n_re_terms > 1) ? data.n_re_terms : 0, -1);

    #ifdef _OPENMP
    #pragma omp for schedule(static)
    #endif
    for (int i = 0; i < data.N; i++) {
      // Compute linear predictors
      double eta_num = ratiod_linalg::dot_product(
          &data.X_num_flat[i * data.p_num], beta_num, data.p_num);
      double eta_denom = (data.p_denom > 0) ?
          ratiod_linalg::dot_product(
              &data.X_denom_flat[i * data.p_denom], beta_denom, data.p_denom) :
          0.0;

      // Add RE if present
      int re_idx = -1;
      int re_term_idx = -1;
      int re_group_idx = -1;
      int re_n_coefs_i = 1;
      std::vector<double> re_slope_x_i;  // Slope design values for this obs
      int n_crossed_terms = 0;
      if (layout.has_re) {
        if (layout.has_re_slopes && n_re_terms_slopes > 0) {
          // Random slopes case: handle first term only (single RE term for H gradients)
          re_term_idx = 0;
          re_group_idx = data.re_group_multi_flat[i * data.n_re_terms + 0];
          if (re_group_idx > 0) {
            int g = re_group_idx - 1;
            re_n_coefs_i = layout.re_n_coefs_multi[0];
            int re_base = layout.re_start_multi[0] + g * re_n_coefs_i;

            // For correlated slopes: use pre-computed non-centered re (re_nc_flat)
            // For uncorrelated NC slopes: compute re = sigma * z on the fly
            // For uncorrelated centered slopes: params store re directly
            bool is_corr_t = !re_nc_flat.empty() &&
                             layout.re_correlated_multi.size() > 0 &&
                             layout.re_correlated_multi[0] && re_n_coefs_i > 1;
            bool is_uncorr_nc = !is_corr_t && slopes_nc;

            // Intercept contribution
            double re_contrib;
            if (is_corr_t) {
              re_contrib = re_nc_flat[re_base];
            } else if (is_uncorr_nc) {
              double sigma_int = std::exp(params[layout.log_sigma_re_slopes[0][0]]);
              re_contrib = sigma_int * params[re_base];
            } else {
              re_contrib = params[re_base];
            }

            // Slope contributions
            int n_slopes = re_n_coefs_i - 1;
            re_slope_x_i.resize(n_slopes);
            if (n_slopes > 0 && !data.re_slope_matrices[0].empty()) {
              for (int s = 0; s < n_slopes; s++) {
                double x_slope = data.re_slope_matrices[0][i * n_slopes + s];
                re_slope_x_i[s] = x_slope;
                double re_slope;
                if (is_corr_t) {
                  re_slope = re_nc_flat[re_base + 1 + s];
                } else if (is_uncorr_nc) {
                  double sigma_s = std::exp(params[layout.log_sigma_re_slopes[0][1 + s]]);
                  re_slope = sigma_s * params[re_base + 1 + s];
                } else {
                  re_slope = params[re_base + 1 + s];
                }
                re_contrib += re_slope * x_slope;
              }
            }

            eta_num += re_contrib;
            if (data.model_type != ModelType::BINOMIAL) {
              eta_denom += re_contrib;
            }
          }
        } else if (data.n_re_terms > 1) {
          // Crossed RE (multiple intercept-only terms)
          // Non-centered: re_val = sigma * z; centered: re_val = params directly
          n_crossed_terms = data.n_re_terms;
          for (int t = 0; t < n_crossed_terms; t++) {
            int group_idx = data.re_group_multi_flat[i * n_crossed_terms + t];
            if (group_idx > 0) {
              int g = group_idx - 1;
              re_idx_multi_buf[t] = g;
              double z_or_re = params[layout.re_start_multi[t] + g];
              double re_val = z_or_re;
              if (data.re_parameterization == 1) {
                double sigma_t = std::exp(params[layout.log_sigma_re_multi[t]]);
                re_val = sigma_t * z_or_re;
              }
              eta_num += re_val;
              if (data.model_type != ModelType::BINOMIAL) {
                eta_denom += re_val;
              }
            } else {
              re_idx_multi_buf[t] = -1;
            }
          }
        } else if (data.re_group[i] > 0) {
          // Simple intercept-only RE (single term)
          // Non-centered: re = sigma * z; centered: re = params directly
          re_idx = data.re_group[i] - 1;
          double re_val = re[re_idx];
          if (data.re_parameterization == 1) {
            re_val = sigma_re * re_val;  // re_val was z, now sigma*z
          }
          eta_num += re_val;
          if (data.model_type != ModelType::BINOMIAL) {
            eta_denom += re_val;
          }
        }
      }
      // Add temporal effect if present
      int t_idx = -1;
      if (layout.has_temporal && !data.temporal_time_idx.empty() && data.temporal_time_idx[i] > 0) {
        int t = data.temporal_time_idx[i] - 1;
        int g = data.temporal_group_idx[i] - 1;
        t_idx = g * data.n_times + t;  // Panel temporal: flat index
        double temporal_effect = phi_temporal[t_idx];
        eta_num += temporal_effect;
        if (data.model_type != ModelType::BINOMIAL) {
          eta_denom += temporal_effect;
        }
      }

      // Add spatial effect if present
      int s_idx = -1;
      double d_spatial_d_phi = 0.0;  // Derivative of spatial_effect wrt phi_spatial
      double d_spatial_d_theta = 0.0;  // Derivative of spatial_effect wrt theta_bym2
      if (layout.has_spatial && !data.spatial_group.empty() && data.spatial_group[i] > 0) {
        s_idx = data.spatial_group[i] - 1;
        double spatial_effect;
        if (data.spatial_type == SpatialType::BYM2) {
          // BYM2: spatial_effect = sigma_s * scale * phi + sigma_u * theta
          double scaled_phi = phi_spatial[s_idx] * data.bym2_scale_factor;
          spatial_effect = sigma_s_bym2 * scaled_phi + sigma_u_bym2 * theta_bym2[s_idx];
          d_spatial_d_phi = sigma_s_bym2 * data.bym2_scale_factor;
          d_spatial_d_theta = sigma_u_bym2;
        } else {
          // ICAR: spatial_effect = phi_spatial
          spatial_effect = phi_spatial[s_idx];
          d_spatial_d_phi = 1.0;
        }
        eta_num += spatial_effect;
        if (data.model_type != ModelType::BINOMIAL) {
          eta_denom += spatial_effect;
        }
      }

      double resid_num = 0.0;
      double resid_denom = 0.0;
      double grad_phi_num_i = 0.0;
      double grad_phi_denom_i = 0.0;
      double grad_logit_zi_i = 0.0;
      double grad_logit_oi_i = 0.0;

      // Compute ZI linear predictor if applicable
      double logit_zi = 0.0;
      if (layout.has_zi && data.p_zi > 0) {
        logit_zi = ratiod_linalg::dot_product(
            &data.X_zi_flat[i * data.p_zi], beta_zi, data.p_zi);
      }

      // Compute OI linear predictor if applicable
      double logit_oi = 0.0;
      if (layout.has_oi && data.p_oi > 0) {
        logit_oi = ratiod_linalg::dot_product(
            &data.X_oi_flat[i * data.p_oi], beta_oi, data.p_oi);
      }

      if (layout.has_zi || layout.has_oi) {
        // An inflated or hurdle margin: the residuals are the derivatives of
        // the density the log posterior evaluates
        unsigned want = 0u;
        if (layout.has_phi_num) want |= ratiod::OBS_D_PHI_NUM;
        if (layout.has_phi_denom) want |= ratiod::OBS_D_PHI_DENOM;
        if (acc_zi) want |= ratiod::OBS_D_ZI;
        if (acc_oi) want |= ratiod::OBS_D_OI;
        const ratiod::ObsDerivs od = ratiod::ratio_obs_derivs(
            data, layout, i, eta_num, eta_denom, phi_num, phi_denom,
            logit_zi, logit_oi, want);
        resid_num = od.eta_num;
        resid_denom = od.eta_denom;
        grad_phi_num_i = od.phi_num;
        grad_phi_denom_i = od.phi_denom;
        grad_logit_zi_i = od.logit_zi;
        grad_logit_oi_i = od.logit_oi;

      } else {
        using namespace tulpaRatio::margins;
        const auto apply = [&](const auto& num, const auto& den,
                               const Observation& x, const Observation& y) {
          using Num = std::decay_t<decltype(num)>;
          using Den = std::decay_t<decltype(den)>;
          const auto n = num.template evaluate<std::is_same<Num,Binomial>::value>(x,false);
          const auto d = den.template evaluate<std::is_same<Den,Binomial>::value>(y,false);
          resid_num = n.residual;
          resid_denom = d.residual;
          grad_phi_num_i = n.extra;
          grad_phi_denom_i = d.extra;
          if constexpr (Num::log_extra) grad_phi_num_i /= phi_num;
          if constexpr (Den::log_extra) grad_phi_denom_i /= phi_denom;
        };
        const Observation count_num{data.y_num.empty() ? 0.0 : double(data.y_num[i]),eta_num};
        switch (data.model_type) {
          case ModelType::BINOMIAL:
            apply(Binomial(),NoMargin(),{double(data.y_num[i]),eta_num,data.y_denom[i]},{});
            break;
          case ModelType::POISSON_GAMMA:
            apply(Poisson(),Gamma(phi_denom,false),count_num,{data.y_denom_cont[i],eta_denom});
            break;
          case ModelType::NEGBIN_GAMMA:
            apply(NegBin(phi_num),Gamma(phi_denom,false),count_num,{data.y_denom_cont[i],eta_denom});
            break;
          case ModelType::NEGBIN_NEGBIN:
            apply(NegBin(phi_num),NegBin(phi_denom),count_num,{double(data.y_denom[i]),eta_denom});
            break;
          case ModelType::GAMMA_GAMMA:
            apply(Gamma(phi_num,false),Gamma(phi_denom,false),{data.y_num_cont[i],eta_num},
                  {data.y_denom_cont[i],eta_denom});
            break;
          case ModelType::LOGNORMAL:
            apply(Lognormal(phi_num,false),Lognormal(phi_denom,false),{data.y_num_cont[i],eta_num},
                  {data.y_denom_cont[i],eta_denom});
            break;
          case ModelType::BETA_BINOMIAL:
            apply(BetaBinomial(phi_num),NoMargin(),{double(data.y_num[i]),eta_num,data.y_denom[i]},{});
            break;
          default: break;
        }
      }

      // Accumulate ZI coefficient gradients
      if (acc_zi) {
        for (int j = 0; j < data.p_zi; j++) {
          acc[acc_at.beta_zi + j] +=
              data.X_zi_flat[i * data.p_zi + j] * grad_logit_zi_i;
        }
      }

      // Accumulate OI coefficient gradients
      if (acc_oi) {
        for (int j = 0; j < data.p_oi; j++) {
          acc[acc_at.beta_oi + j] +=
              data.X_oi_flat[i * data.p_oi + j] * grad_logit_oi_i;
        }
      }

      // Accumulate beta gradients: grad += X[i,:] * resid
      for (int j = 0; j < data.p_num; j++) {
        acc[acc_at.beta_num + j] += data.X_num_flat[i * data.p_num + j] * resid_num;
      }
      // For BINOMIAL, beta_denom doesn't affect likelihood
      if (data.model_type != ModelType::BINOMIAL) {
        for (int j = 0; j < data.p_denom; j++) {
          acc[acc_at.beta_denom + j] +=
              data.X_denom_flat[i * data.p_denom + j] * resid_denom;
        }
      }

      // Accumulate RE gradient
      if (layout.has_re_slopes && re_group_idx > 0) {
        // Random slopes case: gradient for intercept and each slope
        double re_grad_base = resid_num;
        if (data.model_type != ModelType::BINOMIAL) {
          re_grad_base += resid_denom;
        }
        int g = re_group_idx - 1;
        int n_coefs = re_n_coefs_i;

        // Intercept gradient: same as simple RE
        acc[acc_at.re_slopes + g * n_coefs] += re_grad_base;

        // Slope gradients: multiply by slope design value
        int n_slopes = n_coefs - 1;
        for (int s = 0; s < n_slopes; s++) {
          acc[acc_at.re_slopes + g * n_coefs + 1 + s] +=
              re_grad_base * re_slope_x_i[s];
        }
      } else if (n_crossed_terms > 0) {
        // Crossed RE (multiple intercept-only terms)
        double re_grad_i = resid_num;
        if (data.model_type != ModelType::BINOMIAL) {
          re_grad_i += resid_denom;
        }
        for (int t = 0; t < n_crossed_terms; t++) {
          if (re_idx_multi_buf[t] >= 0) {
            acc[acc_at.re_crossed + data.re_offsets[t] + re_idx_multi_buf[t]] +=
                re_grad_i;
          }
        }
      } else if (re_idx >= 0) {
        // Simple intercept-only RE (single term)
        double re_grad_i = resid_num;
        if (data.model_type != ModelType::BINOMIAL) {
          re_grad_i += resid_denom;  // Shared RE affects both processes
        }
        acc[acc_at.re + re_idx] += re_grad_i;
      }

      // Accumulate temporal gradient (from likelihood)
      if (t_idx >= 0) {
        double temp_grad_i = resid_num;
        if (data.model_type != ModelType::BINOMIAL) {
          temp_grad_i += resid_denom;
        }
        acc[acc_at.temporal + t_idx] += temp_grad_i;
      }

      // Accumulate spatial gradient (from likelihood)
      if (s_idx >= 0) {
        double lik_grad = resid_num;
        if (data.model_type != ModelType::BINOMIAL) {
          lik_grad += resid_denom;
        }
        // For ICAR: grad_spatial[s] += lik_grad (d_spatial_d_phi = 1)
        // For BYM2: grad_phi[s] += lik_grad * d_spatial_d_phi, grad_theta[s] += lik_grad * d_spatial_d_theta
        acc[acc_at.spatial + s_idx] += lik_grad * d_spatial_d_phi;

        if (acc_bym2) {
          acc[acc_at.theta_bym2 + s_idx] += lik_grad * d_spatial_d_theta;
        }
      }

      // Accumulate phi gradients
      acc[acc_at.phi_num] += grad_phi_num_i;
      acc[acc_at.phi_denom] += grad_phi_denom_i;

      // The observation's log-likelihood, fused into this pass rather than a
      // separate one through compute_log_post
      if (compute_lp) {
        acc[acc_at.ll] += ratiod::ratio_obs_log_lik(
            data, layout, i, eta_num, eta_denom, phi_num, phi_denom, logit_zi, logit_oi);
      }
    }
  }  // end parallel

  // Deterministic reduction: thread-index order, fixed team size.
  for (int th = 0; th < grad_team; th++) {
    const double* slot = &red_partials[(size_t)th * acc_at.stride];
    for (int j = 0; j < data.p_num; j++) grad_beta_num[j] += slot[acc_at.beta_num + j];
    for (int j = 0; j < data.p_denom; j++) grad_beta_denom[j] += slot[acc_at.beta_denom + j];
    grad_phi_num_lik += slot[acc_at.phi_num];
    grad_phi_denom_lik += slot[acc_at.phi_denom];
    for (int g = 0; g < n_re_acc; g++) {
      grad[layout.re_start + g] += slot[acc_at.re + g];
    }
    // Crossed intercept-only RE (n_re_terms > 1, not slopes) scatter to
    // non-contiguous grad positions.
    if (n_re_crossed_acc > 0) {
      for (int t = 0; t < (int)data.re_n_groups_multi.size(); t++) {
        int re_start_t = layout.re_start_multi[t];
        int offset_t = data.re_offsets[t];
        for (int g = 0; g < data.re_n_groups_multi[t]; g++) {
          grad[re_start_t + g] += slot[acc_at.re_crossed + offset_t + g];
        }
      }
    }
    obs_log_lik += slot[acc_at.ll];
    if (acc_zi) {
      for (int j = 0; j < data.p_zi; j++) grad_beta_zi[j] += slot[acc_at.beta_zi + j];
    }
    if (acc_oi) {
      for (int j = 0; j < data.p_oi; j++) grad_beta_oi[j] += slot[acc_at.beta_oi + j];
    }
    for (int k = 0; k < n_slopes_acc; k++) {
      grad_re_slopes_lik[0][k] += slot[acc_at.re_slopes + k];
    }
    for (int t = 0; t < (int)grad_temporal_lik.size(); t++) {
      grad_temporal_lik[t] += slot[acc_at.temporal + t];
    }
    for (int s = 0; s < (int)grad_spatial_lik.size(); s++) {
      grad_spatial_lik[s] += slot[acc_at.spatial + s];
    }
    if (acc_bym2) {
      for (int s = 0; s < n_spatial; s++) {
        grad[layout.theta_bym2_start + s] += slot[acc_at.theta_bym2 + s];
      }
    }
  }

  // Add likelihood gradients to total
  for (int j = 0; j < data.p_num; j++) {
    grad[layout.beta_num_start + j] += grad_beta_num[j];
  }
  for (int j = 0; j < data.p_denom; j++) {
    grad[layout.beta_denom_start + j] += grad_beta_denom[j];
  }

  // Phi gradients (with Jacobian for log transform)
  if (layout.has_phi_num) {
    grad[layout.log_phi_num_idx] += phi_num * grad_phi_num_lik;
  }
  if (layout.has_phi_denom) {
    grad[layout.log_phi_denom_idx] += phi_denom * grad_phi_denom_lik;
  }

  // ZI coefficient gradients (likelihood contribution)
  if (layout.has_zi && data.p_zi > 0) {
    for (int j = 0; j < data.p_zi; j++) {
      grad[layout.beta_zi_start + j] += grad_beta_zi[j];
    }
  }

  // OI coefficient gradients (likelihood contribution)
  if (layout.has_oi && data.p_oi > 0) {
    for (int j = 0; j < data.p_oi; j++) {
      grad[layout.beta_oi_start + j] += grad_beta_oi[j];
    }
  }

  } // end if (!used_vectorized)

  // The chain rule back onto the sampled parameters, for every RE term and
  // both parameterizations. Outside the used_vectorized check: the hybrid
  // slopes path sets used_vectorized = true and still fills the block's
  // likelihood accumulator through its own scatter pass.
  re_blocks_writeback(params, data, layout, grad, re_ws);

  // ============ Temporal GMRF prior gradients ============
  if (layout.has_temporal && T_len > 0) {
    temporal_gmrf_prior_grad(data, layout, tau_temporal, rho_ar1,
                             tview, T_len, grad_temporal_lik.data(), grad.data());
  }

  // ============ Spatial GMRF prior gradients (ICAR and BYM2) ============
  if (layout.has_spatial && n_spatial > 0) {
    spatial_gmrf_prior_grad(data, layout, phi_spatial, phi_spatial, phi_raw_sums,
                            center_spatial,
                            tau_spatial, sigma_s_bym2, sigma_u_bym2, rho_bym2, 0.0,
                            theta_bym2, grad_spatial_lik.data(), grad.data());
  }

  // Fused log-posterior output: combine prior/structural terms with observation log-lik.
  // Prior/structural terms are computed via compute_log_post with skip_obs_loop=true (O(p+S+T)).
  // Observation log-lik was accumulated inline during the gradient computation (O(N)).
  // Total: one O(N) pass instead of two.
  // compute_log_post centres its own copy and recovers phi_raw_sums from the raw
  // field, so it takes params_in; the centred copy would zero that sum and drop
  // the freed constant direction from the prior.
  if (log_post_out) {
    *log_post_out = compute_log_post(params_in, data, layout, /*skip_obs_loop=*/true) + obs_log_lik;
  }

}

}  // namespace ratiod_hmc
