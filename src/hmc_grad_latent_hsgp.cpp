#include <RcppEigen.h>
#include "hmc_grad_helpers.h"

using namespace Rcpp;

namespace ratiod_hmc {
void compute_gradient_latent_handcoded(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    std::vector<double>& grad,
    double* log_post_out
) {
    // Fused log-posterior: accumulate obs log-lik during gradient loop,
    // then add prior/structural terms via skip_obs_loop=true (avoids 2nd O(N) pass)
    const bool fuse_lp = (log_post_out != nullptr) && !layout.has_zi;
    if (log_post_out && layout.has_zi) *log_post_out = compute_log_post(params, data, layout);
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

    const int N = data.N;
    ratiod_latent::LatentView lv;
    lv.read(params.data(), layout.latent_loading_start, layout.latent_factor_start,
            N, data.latent_n_factors, data.latent_shared,
            static_cast<ratiod_latent::LatentConstraint>(data.latent_constraint));

    beta_gradient_prior(data, layout, beta_num, beta_denom, grad.data());
    re_gradient_prior(data, layout, re, grad.data(), sigma_re);
    phi_gradient_prior(data, layout, phi_num, phi_denom, grad.data());
    inflation_prior_grad(data, layout, params.data(), grad.data());

    std::vector<double> d_num(N, 0.0), d_denom(N, 0.0);
    for (int i = 0; i < N; i++) {
        double eta_num = 0.0, eta_denom = 0.0;
        for (int j = 0; j < data.p_num; j++) {
            eta_num += data.X_num_flat[i * data.p_num + j] * beta_num[j];
        }
        for (int j = 0; j < data.p_denom; j++) {
            eta_denom += data.X_denom_flat[i * data.p_denom + j] * beta_denom[j];
        }

        // Random effects (handles NC parameterization)
        if (layout.has_re && data.re_group[i] > 0) {
            int g = data.re_group[i] - 1;
            double re_eff = re_value_for_eta(re, g, sigma_re, data.re_parameterization);
            eta_num += re_eff;
            eta_denom += re_eff;
        }

        eta_num += lv.eta_num[i];
        eta_denom += lv.denom(i);

        const ratiod::ObsDerivs od = obs_likelihood_derivs(
            data, layout, params.data(), i, eta_num, eta_denom, phi_num, phi_denom,
            grad.data());
        if (fuse_lp) obs_log_lik += od.ll;
        d_num[i] = od.eta_num;
        d_denom[i] = od.eta_denom;

        scatter_beta_gradients(data, layout, i, od.eta_num, od.eta_denom, grad.data());
        scatter_re_gradient(data, layout, i, od.eta_num, od.eta_denom, grad.data());
        if (layout.has_phi_num) grad[layout.log_phi_num_idx] += od.phi_num * phi_num;
        if (layout.has_phi_denom) grad[layout.log_phi_denom_idx] += od.phi_denom * phi_denom;
    }

    lv.add_gradient(params.data(), d_num, d_denom, data.latent_sigma_prior_rate,
                    data.latent_offdiag_sd, grad.data());

    // Non-centered RE chain rule transformation
    re_gradient_nc_transform(data, layout, params.data(), grad.data(), sigma_re);

    if (fuse_lp) *log_post_out = compute_log_post(params, data, layout, /*skip_obs_loop=*/true) + obs_log_lik;
}

// =====================================================================
// GP gradient via autodiff (O(N*nn^3) - much faster than numerical O(N^2))
// Uses templated NNGP likelihood from hmc_gp_autodiff.h
// =====================================================================

// =====================================================================
// Common autodiff prior setup (beta, RE, phi priors)
// Shared by gp_autodiff, msgp_autodiff, gp_temporal_autodiff.
// Returns log_post, sigma_re, phi_num, phi_denom via output parameters.
// =====================================================================


// =====================================================================
// HSGP gradient (O(N*M^2) - analytical, ~50x faster than numerical)
// =====================================================================

void compute_gradient_hsgp(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    std::vector<double>& grad,
    double* log_post_out
) {
    // Thread-local workspace eliminates 9+ heap allocations per call
    RATIOD_TLS_WORKSPACE(ratiod_hsgp::HSGPWorkspace, hsgp_ws);
    hsgp_ws.init(data.N, data.hsgp_data.m_total);

    // Fused log-posterior: accumulate obs log-lik during gradient loop,
    // then add prior/structural terms via skip_obs_loop=true (avoids 2nd O(N) pass)
    const bool fuse_lp = (log_post_out != nullptr) && !layout.has_zi;
    if (log_post_out && layout.has_zi) *log_post_out = compute_log_post(params, data, layout);
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

  // HSGP parameters
  double log_sigma2 = params[layout.log_sigma2_hsgp_idx];
  double log_lengthscale = params[layout.log_lengthscale_hsgp_idx];
  double sigma2_hsgp = std::exp(log_sigma2);
  double lengthscale_hsgp = std::exp(log_lengthscale);

  int m_total = data.hsgp_data.m_total;
  const double* hsgp_beta_ptr = &params[layout.hsgp_beta_start];

  // Evaluate HSGP spatial effect (uses workspace, zero allocation)
  ratiod_hsgp::hsgp_evaluate_ws(hsgp_beta_ptr, sigma2_hsgp, lengthscale_hsgp,
                                 data.hsgp_data, hsgp_ws);

  // Temporal parameters (for HSGP + temporal combinations)
  double tau_temporal = 0.0;
  int T_len = 0;
  const double* phi_temporal = nullptr;
  TemporalView tview;
  double rho_ar1 = 0.0;
  if (layout.has_temporal) {
    tau_temporal = std::exp(params[layout.log_tau_temporal_idx]);
    T_len = layout.temporal_end - layout.temporal_start;
    tview.read(params, data, layout);
    phi_temporal = tview.phi;
    rho_ar1 = read_temporal_rho_ar1(params, data, layout);
  }

  // Zero the grad_f buffer (workspace, no allocation)
  std::memset(hsgp_ws.grad_f.data(), 0, data.N * sizeof(double));
  std::vector<double> grad_temporal_lik(T_len, 0.0);

  // --- Prior gradients ---

  beta_gradient_prior(data, layout, beta_num, beta_denom, grad.data());
  re_gradient_prior(data, layout, re, grad.data(), sigma_re);
  phi_gradient_prior(data, layout, phi_num, phi_denom, grad.data());

  // HSGP prior gradients (will be added to by hsgp_compute_gradients_ws)
  grad[layout.log_sigma2_hsgp_idx] =
      ratiod::math::pc_scale_gradient(std::sqrt(sigma2_hsgp), 0.5, data.gp_sigma2_prior_U, data.gp_sigma2_prior_alpha);

  // LogNormal(0,1) on lengthscale
  grad[layout.log_lengthscale_hsgp_idx] = -log_lengthscale;

  // N(0, I) prior on beta: d/d(beta_j) = -beta_j
  for (int j = 0; j < m_total; j++) {
    grad[layout.hsgp_beta_start + j] = -hsgp_beta_ptr[j];
  }

  // PC prior on the temporal SD and Beta prior on rho.
  if (layout.has_temporal) {
    tau_temporal_prior_grad(data, layout, tau_temporal, grad.data());
    temporal_rho_prior_grad(data, layout, rho_ar1, grad.data());
  }

  // --- Vectorized observation loop (3-pass: Eigen matvec, scalar residuals, Eigen scatter) ---
  const double* hsgp_f = hsgp_ws.hsgp_f.data();
  double* grad_f_ptr = hsgp_ws.grad_f.data();

  const int N = data.N;
  const bool is_binomial = (data.model_type == ModelType::BINOMIAL ||
                            data.model_type == ModelType::BETA_BINOMIAL);

  // Use thread-local workspace for eta/resid buffers (zero allocation)
  using RowMajorMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using VectorXd = Eigen::VectorXd;
    fill_fixed_eta(vec_grad_ws(), data, beta_num, beta_denom);
    Eigen::Map<Eigen::VectorXd> eta_n(vec_grad_ws().eta_num.data(), N);

  // Add RE (expand to dense and vectorized add)
  if (layout.has_re) {
    for (int i = 0; i < N; i++) {
      if (data.re_group[i] > 0) {
        int g = data.re_group[i] - 1;
        double re_eff = re_value_for_eta(re, g, sigma_re, data.re_parameterization);
        vec_grad_ws().eta_num[i] += re_eff;
        if (!is_binomial) vec_grad_ws().eta_denom[i] += re_eff;
      }
    }
  }

  // Add temporal (expand to observation level)
  if (layout.has_temporal && !data.temporal_time_idx.empty()) {
    for (int i = 0; i < N; i++) {
      if (i < (int)data.temporal_time_idx.size() && data.temporal_time_idx[i] > 0) {
        int t = data.temporal_time_idx[i] - 1;
        int g = data.temporal_group_idx[i] - 1;
        int t_idx = g * data.n_times + t;
        if (t_idx >= 0 && t_idx < T_len) {
          vec_grad_ws().eta_num[i] += phi_temporal[t_idx];
          if (!is_binomial && data.temporal_shared) vec_grad_ws().eta_denom[i] += phi_temporal[t_idx];
        }
      }
    }
  }

  // Add HSGP spatial effect (vectorized)
  Eigen::Map<const VectorXd> hsgp_fv(hsgp_f, N);
  eta_n += hsgp_fv;
  if (data.hsgp_data.shared && !is_binomial) {
    Eigen::Map<VectorXd>(vec_grad_ws().eta_denom.data(), N) += hsgp_fv;
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

  // Accumulate grad_f for HSGP from residuals
  for (int i = 0; i < N; i++) {
    grad_f_ptr[i] = data.hsgp_data.shared
        ? (vec_grad_ws().resid_num[i] + vec_grad_ws().resid_denom[i])
        : vec_grad_ws().resid_num[i];
  }

  // RE gradients (scatter from residuals to group-level)
  if (layout.has_re) {
    for (int i = 0; i < N; i++) {
      scatter_re_gradient(data, layout, i, vec_grad_ws().resid_num[i],
                          vec_grad_ws().resid_denom[i], grad.data());
    }
  }

  // Temporal likelihood gradients (scatter to temporal buffer)
  if (layout.has_temporal && !data.temporal_time_idx.empty()) {
    for (int i = 0; i < N; i++) {
      if (i < (int)data.temporal_time_idx.size() && data.temporal_time_idx[i] > 0) {
        int t = data.temporal_time_idx[i] - 1;
        int g = data.temporal_group_idx[i] - 1;
        int t_idx = g * data.n_times + t;
        if (t_idx >= 0 && t_idx < T_len) {
          double lik_grad = data.temporal_shared ?
            (vec_grad_ws().resid_num[i] + vec_grad_ws().resid_denom[i]) :
            vec_grad_ws().resid_num[i];
          grad_temporal_lik[t_idx] += lik_grad;
        }
      }
    }
  }

  // Compute HSGP parameter gradients using workspace (zero allocation)
  double hsgp_grad_log_sigma2, hsgp_grad_log_lengthscale;
  ratiod_hsgp::hsgp_compute_gradients_ws(hsgp_beta_ptr, sigma2_hsgp, lengthscale_hsgp,
                                          data.hsgp_data, hsgp_ws,
                                          hsgp_grad_log_sigma2, hsgp_grad_log_lengthscale);

  // Add likelihood contribution to HSGP gradients
  for (int j = 0; j < m_total; j++) {
    grad[layout.hsgp_beta_start + j] += hsgp_ws.grad_beta_out[j];
  }
  grad[layout.log_sigma2_hsgp_idx] += hsgp_grad_log_sigma2;
  grad[layout.log_lengthscale_hsgp_idx] += hsgp_grad_log_lengthscale;

  // Temporal GMRF gradients
  if (layout.has_temporal && T_len > 0) {
    temporal_gmrf_prior_grad(data, layout, tau_temporal, rho_ar1,
                             tview, T_len, grad_temporal_lik.data(), grad.data());
  }

    // Non-centered RE chain rule transformation
    re_gradient_nc_transform(data, layout, params.data(), grad.data(), sigma_re);

    if (fuse_lp) *log_post_out = compute_log_post(params, data, layout, /*skip_obs_loop=*/true) + obs_log_lik;
}

// =====================================================================
// HSGP-MSGP gradient (hand-coded)
// Two independent HSGP evaluations with shared basis matrix
// =====================================================================

void compute_gradient_msgp_hsgp(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    std::vector<double>& grad,
    double* log_post_out
) {
    // Two thread-local HSGP workspaces (one per scale)
    RATIOD_TLS_WORKSPACE(ratiod_hsgp::HSGPWorkspace, ws_local);
    RATIOD_TLS_WORKSPACE(ratiod_hsgp::HSGPWorkspace, ws_regional);

    const bool fuse_lp = (log_post_out != nullptr) && !layout.has_zi;
    if (log_post_out && layout.has_zi) *log_post_out = compute_log_post(params, data, layout);
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

    MultiscaleHSGPView msgp;
    msgp.read(params, data, layout, ws_local, ws_regional);

    // Temporal parameters (for HSGP-MSGP + temporal combinations)
    double tau_temporal = 0.0;
    int T_len = 0;
    const double* phi_temporal = nullptr;
    TemporalView tview;
    double rho_ar1 = 0.0;
    if (layout.has_temporal) {
        tau_temporal = std::exp(params[layout.log_tau_temporal_idx]);
        T_len = layout.temporal_end - layout.temporal_start;
        tview.read(params, data, layout);
        phi_temporal = tview.phi;
        rho_ar1 = read_temporal_rho_ar1(params, data, layout);
    }

    std::vector<double> grad_temporal_lik(T_len, 0.0);

    // --- Prior gradients ---
    beta_gradient_prior(data, layout, beta_num, beta_denom, grad.data());
    re_gradient_prior(data, layout, re, grad.data(), sigma_re);
    phi_gradient_prior(data, layout, phi_num, phi_denom, grad.data());

    msgp.add_prior_gradient(data, layout, grad.data());

    // Temporal prior on tau/rho
    if (layout.has_temporal) {
        tau_temporal_prior_grad(data, layout, tau_temporal, grad.data());
        temporal_rho_prior_grad(data, layout, rho_ar1, grad.data());
    }

    // --- Vectorized observation loop ---
    const int N = data.N;
    const bool is_binomial = (data.model_type == ModelType::BINOMIAL ||
                              data.model_type == ModelType::BETA_BINOMIAL);
    const bool shared = msgp.shared;

    using RowMajorMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using VectorXd = Eigen::VectorXd;
    fill_fixed_eta(vec_grad_ws(), data, beta_num, beta_denom);
    Eigen::Map<Eigen::VectorXd> eta_n(vec_grad_ws().eta_num.data(), N);

    // Add RE
    if (layout.has_re) {
        for (int i = 0; i < N; i++) {
            if (data.re_group[i] > 0) {
                int g = data.re_group[i] - 1;
                double re_eff = re_value_for_eta(re, g, sigma_re, data.re_parameterization);
                vec_grad_ws().eta_num[i] += re_eff;
                if (!is_binomial) vec_grad_ws().eta_denom[i] += re_eff;
            }
        }
    }

    // Add temporal
    if (layout.has_temporal && !data.temporal_time_idx.empty()) {
        for (int i = 0; i < N; i++) {
            if (i < (int)data.temporal_time_idx.size() && data.temporal_time_idx[i] > 0) {
                int t = data.temporal_time_idx[i] - 1;
                int g = data.temporal_group_idx[i] - 1;
                int t_idx = g * data.n_times + t;
                if (t_idx >= 0 && t_idx < T_len) {
                    vec_grad_ws().eta_num[i] += phi_temporal[t_idx];
                    if (!is_binomial && data.temporal_shared) vec_grad_ws().eta_denom[i] += phi_temporal[t_idx];
                }
            }
        }
    }

    // The multi-scale field: f_local + f_regional
    for (int i = 0; i < N; i++) {
        const double f_i = msgp.effect(i);
        vec_grad_ws().eta_num[i] += f_i;
        if (shared && !is_binomial) vec_grad_ws().eta_denom[i] += f_i;
    }

    // Pass 2+3: vectorized residuals + beta grads
    {
        double grad_phi_num_lik = 0.0, grad_phi_denom_lik = 0.0;
        vectorized::dispatch_residuals_and_beta_grads(
            data, layout,
            vec_grad_ws().eta_num.data(), vec_grad_ws().eta_denom.data(),
            vec_grad_ws().resid_num.data(), vec_grad_ws().resid_denom.data(),
            grad.data(), grad_phi_num_lik, grad_phi_denom_lik,
            obs_log_lik, fuse_lp, phi_num, phi_denom, vec_grad_ws());
    }

    // dL/df, shared by both scales
    std::vector<double> dL_df(N);
    for (int i = 0; i < N; i++) {
        dL_df[i] = shared ? (vec_grad_ws().resid_num[i] + vec_grad_ws().resid_denom[i])
                          : vec_grad_ws().resid_num[i];
    }

    // RE gradients
    if (layout.has_re) {
        for (int i = 0; i < N; i++) {
            scatter_re_gradient(data, layout, i, vec_grad_ws().resid_num[i],
                                vec_grad_ws().resid_denom[i], grad.data());
        }
    }

    // Temporal likelihood gradients
    if (layout.has_temporal && !data.temporal_time_idx.empty()) {
        for (int i = 0; i < N; i++) {
            if (i < (int)data.temporal_time_idx.size() && data.temporal_time_idx[i] > 0) {
                int t = data.temporal_time_idx[i] - 1;
                int g = data.temporal_group_idx[i] - 1;
                int t_idx = g * data.n_times + t;
                if (t_idx >= 0 && t_idx < T_len) {
                    double lik_grad = data.temporal_shared ?
                        (vec_grad_ws().resid_num[i] + vec_grad_ws().resid_denom[i]) :
                        vec_grad_ws().resid_num[i];
                    grad_temporal_lik[t_idx] += lik_grad;
                }
            }
        }
    }

    msgp.add_lik_gradient(dL_df.data(), data, layout, grad.data());

    // Temporal GMRF gradients
    if (layout.has_temporal && T_len > 0) {
        temporal_gmrf_prior_grad(data, layout, tau_temporal, rho_ar1,
                                 tview, T_len, grad_temporal_lik.data(), grad.data());
    }

    // Non-centered RE chain rule transformation
    re_gradient_nc_transform(data, layout, params.data(), grad.data(), sigma_re);

    if (fuse_lp) *log_post_out = compute_log_post(params, data, layout, /*skip_obs_loop=*/true) + obs_log_lik;
}


}  // namespace ratiod_hmc
