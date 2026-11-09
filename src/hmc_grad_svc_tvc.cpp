#include <RcppEigen.h>
#include "hmc_grad_helpers.h"
#include "hmc_tvc_grad.h"

using namespace Rcpp;

namespace ratiod_hmc {


void compute_gradient_svc_handcoded(
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

    // SVC parameters (per-thread scratch: data.svc_data is shared by every
    // chain thread, so its own buffers cannot be written from here)
    int n_svc = data.svc_data.n_svc;
    int N_obs = data.svc_data.n_obs;

    RATIOD_TLS_WORKSPACE(ratiod_svc::SVCGradWorkspace, svc_scratch);
    svc_scratch.resize(n_svc, N_obs);

    double* svc_sigma2 = svc_scratch.sigma2.data();
    double* svc_phi = svc_scratch.phi.data();
    for (int j = 0; j < n_svc; j++) {
        svc_sigma2[j] = std::exp(params[layout.log_sigma2_svc_start + j]);
        svc_phi[j] = ratiod_svc_ad::svc_range(params[layout.range_svc_start + j],
                                              data.svc_phi_prior_lower,
                                              data.svc_phi_prior_upper);
    }

    // The SVC field at the sampled coordinate, into per-thread scratch
    double* svc_w_flat = svc_scratch.w_flat.data();
    const double* svc_block = &params[layout.svc_w_start];
    ratiod_svc::svc_field<ratiod_svc::SVC_GRAD_SLOT>(svc_block, svc_sigma2, svc_phi, data.svc_data,
                          data.svc_gp_view, data.svc_noncentered, svc_w_flat);

    // =========================================================================
    // Prior gradients
    // =========================================================================

    beta_gradient_prior(data, layout, beta_num, beta_denom, grad.data());
    re_gradient_prior(data, layout, re, grad.data(), sigma_re);
    phi_gradient_prior(data, layout, phi_num, phi_denom, grad.data());

    // SVC hyperparameter priors
    for (int j = 0; j < n_svc; j++) {
        // PC prior on sigma, including the log-variance Jacobian.
        grad[layout.log_sigma2_svc_start + j] = ratiod::math::pc_scale_gradient(std::sqrt(svc_sigma2[j]), 0.5, data.svc_sigma_prior_U, data.svc_sigma_prior_alpha);
    }

    // =========================================================================
    // NNGP field prior (centred: on w; non-centred: N(0, I) on z, seeded by
    // svc_field_accumulate below)
    // =========================================================================
    double svc_log_prior = std::numeric_limits<double>::quiet_NaN();
    if (!data.svc_noncentered) {
        svc_log_prior = ratiod_svc::svc_nngp_prior_grads(
            svc_w_flat, svc_sigma2, svc_phi, data.svc_data, data.svc_gp_view,
            layout.svc_w_start, layout.log_sigma2_svc_start, layout.range_svc_start,
            svc_scratch, grad.data());
    }

    // The field prior's gradient above is placed on the uncentred w and is
    // NOT projected. The likelihood's is: eta reads the CENTRED field, so its
    // gradient is accumulated apart in svc_scratch.lik_grad and projected onto
    // the sum-to-zero subspace by svc_center_terms below.
    double* svc_lik_grad = svc_scratch.lik_grad.data();
    std::fill(svc_scratch.lik_grad.begin(), svc_scratch.lik_grad.end(), 0.0);

    // The field the likelihood reads is w minus its per-term mean; the field
    // the NNGP prior above reads is w itself. Holding the means rather than a
    // centred copy lets one buffer serve both.
    double* svc_term_mean = svc_scratch.term_mean.data();
    ratiod_svc::svc_term_means(svc_w_flat, data.svc_data, svc_term_mean);

    // =========================================================================
    // Data likelihood loop (vectorized eta + per-obs scatter)
    // =========================================================================
    const int N = data.N;
    const bool is_binomial = (data.model_type == ModelType::BINOMIAL ||
                              data.model_type == ModelType::BETA_BINOMIAL);

    using RowMajorMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using VectorXd = Eigen::VectorXd;

    fill_fixed_eta(vec_grad_ws(), data, beta_num, beta_denom);
    Eigen::Map<Eigen::VectorXd> eta_n(vec_grad_ws().eta_num.data(), N);

    for (int i = 0; i < N; i++) {
        if (layout.has_re && data.re_group[i] > 0) {
            double re_eff = re_value_for_eta(re, data.re_group[i] - 1, sigma_re, data.re_parameterization);
            vec_grad_ws().eta_num[i] += re_eff;
            if (!is_binomial) vec_grad_ws().eta_denom[i] += re_eff;
        }
        double svc_effect = 0.0;
        for (int j = 0; j < n_svc; j++) {
            svc_effect += data.svc_data.X_svc[i * n_svc + j] *
                          (svc_w_flat[j * N_obs + i] - svc_term_mean[j]);
        }
        vec_grad_ws().eta_num[i] += svc_effect;
        if (!is_binomial && data.svc_data.shared) vec_grad_ws().eta_denom[i] += svc_effect;
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

    for (int i = 0; i < N; i++) {
        double dLL_num = vec_grad_ws().resid_num[i];
        double dLL_denom = vec_grad_ws().resid_denom[i];
        if (layout.has_re && data.re_group[i] > 0) {
            grad[layout.re_start + data.re_group[i] - 1] += dLL_num + dLL_denom;
        }
        double dLL_dsvc = data.svc_data.shared ? (dLL_num + dLL_denom) : dLL_num;
        for (int j = 0; j < n_svc; j++) {
            svc_lik_grad[j * N_obs + i] += dLL_dsvc * data.svc_data.X_svc[i * n_svc + j];
        }
    }

    ratiod_svc::svc_center_terms(svc_lik_grad, data.svc_data);
    ratiod_svc::svc_field_accumulate<ratiod_svc::SVC_GRAD_SLOT>(
        svc_block, svc_sigma2, svc_phi, data.svc_data, data.svc_gp_view,
        data.svc_noncentered, svc_lik_grad, layout.svc_w_start,
        layout.log_sigma2_svc_start, layout.range_svc_start, grad.data());
    svc_range_grads_to_coord(params, data, layout, grad);

    // Non-centered RE chain rule transformation
    re_gradient_nc_transform(data, layout, params.data(), grad.data(), sigma_re);

    if (fuse_lp) {
        ratiod::LogPostOptions opts;
        opts.skip_obs_loop = true;
        if (std::isfinite(svc_log_prior)) opts.precomputed_svc_log_prior = &svc_log_prior;
        *log_post_out = ratiod::compute_log_post_impl<double>(params, data, layout, opts)
                        + obs_log_lik;
    }
}

// =====================================================================
// HSGP-SVC gradient (hand-coded)
// Uses HSGP basis function approximation for spatially-varying coefficients.
// Each SVC term k has its own GP with sigma2_k, lengthscale_k, beta_k[m^2].
// All terms share the same basis matrix Phi (same coordinates/eigenvalues).
// =====================================================================

void compute_gradient_svc_hsgp_handcoded(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    std::vector<double>& grad,
    double* log_post_out
) {
    // Thread-local HSGP workspace (one per SVC term, reused)
    RATIOD_TLS_WORKSPACE(ratiod_hsgp::HSGPWorkspace, hsgp_ws);
    hsgp_ws.init(data.N, data.svc_hsgp_data.m_total);

    const bool fuse_lp = (log_post_out != nullptr) && !layout.has_zi;
    if (log_post_out && layout.has_zi) *log_post_out = compute_log_post(params, data, layout);
    double obs_log_lik = 0.0;
    int n_params = params.size();
    grad.assign(n_params, 0.0);

    auto cp = extract_common_params(params, layout);
    const double* beta_num = cp.beta_num;
    const double* beta_denom = cp.beta_denom;
    double sigma_re = cp.sigma_re;
    const double* re = cp.re;
    double phi_num = cp.phi_num;
    double phi_denom = cp.phi_denom;

    const int N = data.N;
    const int n_svc = data.svc_data.n_svc;
    const int m_total = data.svc_hsgp_data.m_total;
    const bool is_binomial = (data.model_type == ModelType::BINOMIAL ||
                              data.model_type == ModelType::BETA_BINOMIAL);

    // Per-term HSGP parameters: sigma2_k, lengthscale_k, beta_k[m_total]
    // Stored in params as: [log_sigma2_svc[0..n_svc], log_lengthscale_svc[0..n_svc], beta[0..n_svc*m_total]]
    // (the range_svc slot holds the log lengthscale for HSGP-SVC)

    // Evaluate f_k(s_i) for each SVC term and accumulate SVC contribution to eta
    // svc_f[j][i] = Phi * (sqrt(S_j) * beta_j)  -- spatial function for SVC term j at obs i
    // We compute this iteratively, storing each f_k in hsgp_ws.hsgp_f
    std::vector<double> svc_eta(N, 0.0);
    // Store per-term f values for gradient backprop (n_svc * N)
    std::vector<double> svc_f_all(n_svc * N);

    for (int j = 0; j < n_svc; j++) {
        double sigma2_j = std::exp(params[layout.log_sigma2_svc_start + j]);
        double lengthscale_j = std::exp(params[layout.range_svc_start + j]);
        const double* beta_j = &params[layout.svc_w_start + j * m_total];

        // Evaluate f_j = Phi * (sqrt(S_j) ⊙ beta_j)
        ratiod_hsgp::hsgp_evaluate_ws(beta_j, sigma2_j, lengthscale_j,
                                       data.svc_hsgp_data, hsgp_ws);

        // Store f_j and accumulate into svc_eta
        std::memcpy(&svc_f_all[j * N], hsgp_ws.hsgp_f.data(), N * sizeof(double));
        for (int i = 0; i < N; i++) {
            svc_eta[i] += data.svc_data.X_svc[i * n_svc + j] * hsgp_ws.hsgp_f[i];
        }
    }

    // --- Prior gradients ---
    beta_gradient_prior(data, layout, beta_num, beta_denom, grad.data());
    re_gradient_prior(data, layout, re, grad.data(), sigma_re);
    phi_gradient_prior(data, layout, phi_num, phi_denom, grad.data());

    // Per-term HSGP hyperparameter priors
    for (int j = 0; j < n_svc; j++) {
        double sigma2_j = std::exp(params[layout.log_sigma2_svc_start + j]);
        double log_ls_j = params[layout.range_svc_start + j];

        // PC prior on sigma
        grad[layout.log_sigma2_svc_start + j] =
            ratiod::math::pc_scale_gradient(std::sqrt(sigma2_j), 0.5, data.svc_sigma_prior_U, data.svc_sigma_prior_alpha);

        // LogNormal(0,1) on lengthscale: d/d(log_ell) = -log_ell
        grad[layout.range_svc_start + j] = -log_ls_j;

        // N(0,I) prior on beta: d/d(beta_j_k) = -beta_j_k
        for (int k = 0; k < m_total; k++) {
            grad[layout.svc_w_start + j * m_total + k] = -params[layout.svc_w_start + j * m_total + k];
        }
    }

    // --- Vectorized observation loop ---
    using RowMajorMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using VectorXd = Eigen::VectorXd;
    fill_fixed_eta(vec_grad_ws(), data, beta_num, beta_denom);
    Eigen::Map<Eigen::VectorXd> eta_n(vec_grad_ws().eta_num.data(), N);

    // Add RE
    if (layout.has_re) {
        for (int i = 0; i < N; i++) {
            if (data.re_group[i] > 0) {
                double re_eff = re_value_for_eta(re, data.re_group[i] - 1, sigma_re, data.re_parameterization);
                vec_grad_ws().eta_num[i] += re_eff;
                if (!is_binomial) vec_grad_ws().eta_denom[i] += re_eff;
            }
        }
    }

    // Add SVC effect
    for (int i = 0; i < N; i++) {
        vec_grad_ws().eta_num[i] += svc_eta[i];
        if (!is_binomial && data.svc_data.shared) vec_grad_ws().eta_denom[i] += svc_eta[i];
    }

    // Compute residuals and beta grads
    {
        double grad_phi_num_lik = 0.0, grad_phi_denom_lik = 0.0;
        vectorized::dispatch_residuals_and_beta_grads(
            data, layout,
            vec_grad_ws().eta_num.data(), vec_grad_ws().eta_denom.data(),
            vec_grad_ws().resid_num.data(), vec_grad_ws().resid_denom.data(),
            grad.data(), grad_phi_num_lik, grad_phi_denom_lik,
            obs_log_lik, fuse_lp, phi_num, phi_denom, vec_grad_ws());
    }

    // RE gradients
    if (layout.has_re) {
        for (int i = 0; i < N; i++) {
            if (data.re_group[i] > 0) {
                grad[layout.re_start + data.re_group[i] - 1] += vec_grad_ws().resid_num[i] + vec_grad_ws().resid_denom[i];
            }
        }
    }

    // --- HSGP gradient backprop per SVC term ---
    // grad_f_k[i] = dLL/d(svc_eta_i) * X_svc[i,k]
    // where dLL/d(svc_eta_i) = resid_num[i] + (shared ? resid_denom[i] : 0)
    for (int j = 0; j < n_svc; j++) {
        double sigma2_j = std::exp(params[layout.log_sigma2_svc_start + j]);
        double lengthscale_j = std::exp(params[layout.range_svc_start + j]);
        const double* beta_j = &params[layout.svc_w_start + j * m_total];

        // Build grad_f for this SVC term
        double* grad_f_ptr = hsgp_ws.grad_f.data();
        for (int i = 0; i < N; i++) {
            double dLL = data.svc_data.shared
                ? (vec_grad_ws().resid_num[i] + vec_grad_ws().resid_denom[i])
                : vec_grad_ws().resid_num[i];
            grad_f_ptr[i] = dLL * data.svc_data.X_svc[i * n_svc + j];
        }

        // Re-evaluate sqrt_S for this term (needed by gradient computation)
        // hsgp_evaluate_ws caches sqrt_S in workspace
        ratiod_hsgp::hsgp_evaluate_ws(beta_j, sigma2_j, lengthscale_j,
                                       data.svc_hsgp_data, hsgp_ws);

        // Compute HSGP gradients for beta_j, sigma2_j, lengthscale_j
        double grad_log_sigma2_j, grad_log_lengthscale_j;
        ratiod_hsgp::hsgp_compute_gradients_ws(beta_j, sigma2_j, lengthscale_j,
                                                data.svc_hsgp_data, hsgp_ws,
                                                grad_log_sigma2_j, grad_log_lengthscale_j);

        // Accumulate into gradient vector
        for (int k = 0; k < m_total; k++) {
            grad[layout.svc_w_start + j * m_total + k] += hsgp_ws.grad_beta_out[k];
        }
        grad[layout.log_sigma2_svc_start + j] += grad_log_sigma2_j;
        grad[layout.range_svc_start + j] += grad_log_lengthscale_j;
    }

    re_gradient_nc_transform(data, layout, params.data(), grad.data(), sigma_re);

    if (fuse_lp) *log_post_out = compute_log_post(params, data, layout, /*skip_obs_loop=*/true) + obs_log_lik;
}

// =====================================================================
// TVC gradient (hand-coded, ~3x faster than autodiff)
// Uses analytical gradients from hmc_tvc_grad.h for RW1/RW2/AR1 priors
// =====================================================================

void compute_gradient_tvc_handcoded(
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

    // TVC parameters (per-thread scratch: data.tvc_data is shared by every
    // chain thread, so its own buffers cannot be written from here)
    int n_tvc = data.tvc_data.n_tvc;
    int n_times = data.tvc_data.n_times;
    int n_groups = data.tvc_data.n_groups;
    int n_w = n_groups * n_tvc * n_times;

    RATIOD_TLS_WORKSPACE(ratiod_tvc::TVCGradWorkspace, tvc_scratch);
    tvc_scratch.resize(n_tvc, n_times, n_groups, data.N);

    double* tvc_tau = tvc_scratch.tau.data();
    double* tvc_rho = tvc_scratch.rho.data();
    for (int j = 0; j < n_tvc; j++) {
        tvc_tau[j] = std::exp(params[layout.log_tau_tvc_start + j]);
        if (data.tvc_data.structure == ratiod_temporal::TemporalType::AR1) {
            double logit_rho = params[layout.logit_rho_tvc_start + j];
            double u = 1.0 / (1.0 + std::exp(-logit_rho));
            tvc_rho[j] = 2.0 * u - 1.0;
        } else {
            tvc_rho[j] = 0.0;
        }
    }

    // Extract TVC w values into per-thread scratch
    double* tvc_w_flat = tvc_scratch.w_flat.data();
    for (int k = 0; k < n_w; k++) {
        tvc_w_flat[k] = params[layout.tvc_w_start + k];
    }

    // =========================================================================
    // Prior gradients
    // =========================================================================

    beta_gradient_prior(data, layout, beta_num, beta_denom, grad.data());
    re_gradient_prior(data, layout, re, grad.data(), sigma_re);
    phi_gradient_prior(data, layout, phi_num, phi_denom, grad.data());

    // PC priors on TVC SDs, differentiated on log precision.
    for (int j = 0; j < n_tvc; j++) {
        grad[layout.log_tau_tvc_start + j] = ratiod::math::pc_scale_gradient(1.0 / std::sqrt(tvc_tau[j]), -0.5, data.tvc_sigma_prior_U, data.tvc_sigma_prior_alpha);
    }

    // AR1: the rho prior in the coordinate the sampler moves
    if (data.tvc_data.structure == ratiod_temporal::TemporalType::AR1) {
        for (int j = 0; j < n_tvc; j++) {
            double u = (tvc_rho[j] + 1.0) / 2.0;  // u in (0,1)
            grad[layout.logit_rho_tvc_start + j] = ratiod_ar1::log_prior_logit_rho_grad(
                u, ratiod_ar1::RHO_PRIOR_A, ratiod_ar1::RHO_PRIOR_B);
        }
    }

    // =========================================================================
    // Compute TVC prior gradients using zero-allocation workspace version
    // =========================================================================
    ratiod_tvc::TVCGradientWS tvc_ws;
    tvc_ws.grad_w = tvc_scratch.grad_w.data();
    tvc_ws.grad_log_tau = tvc_scratch.grad_log_tau.data();
    tvc_ws.grad_logit_rho = tvc_scratch.grad_logit_rho.data();
    tvc_ws.grad_w_jg = tvc_scratch.grad_w_jg.data();
    tvc_ws.d_buf = tvc_scratch.d_buf.data();
    tvc_ws.n_w = n_w;
    tvc_ws.n_tvc = n_tvc;
    ratiod_tvc::tvc_prior_gradients_ws(tvc_w_flat, data.tvc_data, tvc_tau, tvc_rho, tvc_ws);

    // Add TVC prior gradient contributions
    for (int k = 0; k < n_w; k++) {
        grad[layout.tvc_w_start + k] += tvc_ws.grad_w[k];
    }
    for (int j = 0; j < n_tvc; j++) {
        grad[layout.log_tau_tvc_start + j] += tvc_ws.grad_log_tau[j];
    }
    if (data.tvc_data.structure == ratiod_temporal::TemporalType::AR1) {
        for (int j = 0; j < n_tvc; j++) {
            grad[layout.logit_rho_tvc_start + j] += tvc_ws.grad_logit_rho[j];
        }
    }

    // =========================================================================
    // Precompute TVC contribution to linear predictor (pre-allocated buffer)
    // =========================================================================
    double* tvc_eta = tvc_scratch.eta.data();
    std::fill(tvc_eta, tvc_eta + data.N, 0.0);
    for (int i = 0; i < data.N; i++) {
        int t = data.tvc_data.time_index[i] - 1;  // 0-based
        int g = data.tvc_data.group_index[i] - 1;  // 0-based

        for (int j = 0; j < n_tvc; j++) {
            double x_ij = data.tvc_data.X_tvc[i * n_tvc + j];
            double w_jgt = tvc_w_flat[(g * n_tvc + j) * n_times + t];
            tvc_eta[i] += x_ij * w_jgt;
        }
    }

    // =========================================================================
    // Data likelihood loop (vectorized eta + per-obs scatter)
    // =========================================================================
    const int N = data.N;
    const bool is_binomial = (data.model_type == ModelType::BINOMIAL ||
                              data.model_type == ModelType::BETA_BINOMIAL);

    // --- Pass 1: Vectorized eta computation (Eigen matvec) ---
    using RowMajorMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using VectorXd = Eigen::VectorXd;


    fill_fixed_eta(vec_grad_ws(), data, beta_num, beta_denom);
    Eigen::Map<Eigen::VectorXd> eta_n(vec_grad_ws().eta_num.data(), N);

    // Add RE + TVC effects per-obs
    for (int i = 0; i < N; i++) {
        if (layout.has_re && data.re_group[i] > 0) {
            int g = data.re_group[i] - 1;
            double re_eff = re_value_for_eta(re, g, sigma_re, data.re_parameterization);
            vec_grad_ws().eta_num[i] += re_eff;
            if (!is_binomial) vec_grad_ws().eta_denom[i] += re_eff;
        }
        double tvc_effect = tvc_eta[i];
        vec_grad_ws().eta_num[i] += tvc_effect;
        if (!is_binomial && data.tvc_data.shared) vec_grad_ws().eta_denom[i] += tvc_effect;
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

    // Scatter residuals to RE and TVC gradients
    for (int i = 0; i < N; i++) {
        if (layout.has_re && data.re_group[i] > 0) {
            grad[layout.re_start + data.re_group[i] - 1] += vec_grad_ws().resid_num[i] + vec_grad_ws().resid_denom[i];
        }
        double dLL_dtvc = data.tvc_data.shared
            ? (vec_grad_ws().resid_num[i] + vec_grad_ws().resid_denom[i])
            : vec_grad_ws().resid_num[i];
        int t = data.tvc_data.time_index[i] - 1;
        int g = data.tvc_data.group_index[i] - 1;
        for (int j = 0; j < n_tvc; j++) {
            double x_ij = data.tvc_data.X_tvc[i * n_tvc + j];
            int w_idx = (g * n_tvc + j) * n_times + t;
            grad[layout.tvc_w_start + w_idx] += dLL_dtvc * x_ij;
        }
    }

    re_gradient_nc_transform(data, layout, params.data(), grad.data(), sigma_re);

    if (fuse_lp) *log_post_out = compute_log_post(params, data, layout, /*skip_obs_loop=*/true) + obs_log_lik;
}


}  // namespace ratiod_hmc
