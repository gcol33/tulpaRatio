#include <RcppEigen.h>
#include "hmc_grad_helpers.h"
#include "hmc_gp_autodiff.h"
#include "hmc_gp_collapsed.h"

using namespace Rcpp;

namespace ratiod_hmc {
void compute_gradient_gp_handcoded(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    std::vector<double>& grad,
    double* log_post_out
) {
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

    // GP parameters
    int N_gp = data.gp_data.n_obs;
    double sigma2_gp = std::exp(params[layout.log_sigma2_gp_idx]);
    double phi_gp = std::exp(params[layout.log_phi_gp_idx]);

    // Outside the range prior's support the density is zero: a caller reading
    // the fused log-posterior must see -Inf there, not the value it held.
    if (phi_gp < data.gp_phi_prior_lower || phi_gp > data.gp_phi_prior_upper) {
        if (log_post_out) *log_post_out = -INFINITY;
        return;
    }

    // Non-centered parameterization: params store z ~ N(0,1), reconstruct w
    const bool use_nc = (data.gp_parameterization == 1);
    RATIOD_TLS_WORKSPACE(ratiod_gp::NNGPNCWorkspace, nc_ws);
    const bool fuse_lp = (log_post_out != nullptr) && !layout.has_zi;
    if (log_post_out && layout.has_zi) *log_post_out = compute_log_post(params, data, layout);

    // Get spatial effects: either w directly (centered) or reconstruct from z (NC)
    std::vector<double> gp_w(N_gp);
    if (use_nc) {
        // Forward pass: z -> w
        const double* z_params = &params[layout.gp_w_start];
        ratiod_gp::nngp_nc_forward(z_params, sigma2_gp, phi_gp, data.gp_data, nc_ws);
        // Use reconstructed w
        std::memcpy(gp_w.data(), nc_ws.w.data(), N_gp * sizeof(double));
    } else {
        for (int i = 0; i < N_gp; i++) {
            gp_w[i] = params[layout.gp_w_start + i];
        }
    }

    // Prior gradients
    beta_gradient_prior(data, layout, beta_num, beta_denom, grad.data());
    re_gradient_prior(data, layout, re, grad.data(), sigma_re);
    phi_gradient_prior(data, layout, phi_num, phi_denom, grad.data());

    // PC prior on GP variance
    grad[layout.log_sigma2_gp_idx] = ratiod::math::pc_prior_grad_log_sigma2(
        sigma2_gp, data.gp_sigma2_prior_U, data.gp_sigma2_prior_alpha);

    // Uniform prior on phi - just Jacobian for log transform
    grad[layout.log_phi_gp_idx] = 1.0;

    // NC: no Jacobian gradient needed (cancels with NNGP normalizing constant)

    ratiod_gp::NNGPGradients nngp_grads;
    if (!use_nc) {
        // =====================================================================
        // Centered: NNGP prior gradients on w
        // =====================================================================
        ratiod_gp::gp_nngp_gradients(gp_w, sigma2_gp, phi_gp, data.gp_data, nngp_grads);

        for (int i = 0; i < N_gp; i++) {
            grad[layout.gp_w_start + i] += nngp_grads.grad_w[i];
        }
        grad[layout.log_sigma2_gp_idx] += nngp_grads.grad_log_sigma2;
        grad[layout.log_phi_gp_idx] += nngp_grads.grad_log_phi;
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

    // Add RE + GP effects per-obs (using reconstructed w for NC)
    for (int i = 0; i < N; i++) {
        if (layout.has_re && data.re_group[i] > 0) {
            int g = data.re_group[i] - 1;
            double re_eff = re_value_for_eta(re, g, sigma_re, data.re_parameterization);
            vec_grad_ws().eta_num[i] += re_eff;
            if (!is_binomial) vec_grad_ws().eta_denom[i] += re_eff;
        }
        int loc_i = data.gp_data.obs_to_loc[i];
        double gp_effect = gp_w[loc_i];
        vec_grad_ws().eta_num[i] += gp_effect;
        if (!is_binomial && data.gp_data.shared) vec_grad_ws().eta_denom[i] += gp_effect;
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

    // Scatter residuals to RE gradients
    for (int i = 0; i < N; i++) {
        if (layout.has_re && data.re_group[i] > 0) {
            grad[layout.re_start + data.re_group[i] - 1] += vec_grad_ws().resid_num[i] + vec_grad_ws().resid_denom[i];
        }
    }

    if (use_nc) {
        // =====================================================================
        // NC: Accumulate dL/dw from residuals, then backward pass for z gradients
        // =====================================================================
        std::vector<double> dL_dw(N_gp, 0.0);
        for (int i = 0; i < N; i++) {
            int loc_i = data.gp_data.obs_to_loc[i];
            double dLL = data.gp_data.shared
                ? (vec_grad_ws().resid_num[i] + vec_grad_ws().resid_denom[i])
                : vec_grad_ws().resid_num[i];
            dL_dw[loc_i] += dLL;
        }

        // Backward pass: dL/dw -> grad_z, grad_log_sigma2, grad_log_phi
        std::vector<double> grad_z(N_gp, 0.0);
        double grad_log_sigma2_lik = 0.0, grad_log_phi_lik = 0.0, grad_log_phi_jac = 0.0;
        const double* z_params = &params[layout.gp_w_start];
        ratiod_gp::nngp_nc_backward(
            z_params, sigma2_gp, phi_gp, data.gp_data, nc_ws,
            dL_dw.data(), grad_z.data(),
            grad_log_sigma2_lik, grad_log_phi_lik, grad_log_phi_jac);

        // nngp_nc_backward returns the FULL z gradient -- its own contract is
        // "prior + likelihood", and it seeds each entry with -z[loc] before
        // adding sqrt(d_i) * adj_i. Subtracting z_params here as well applied
        // the N(0,1) prior twice.
        for (int i = 0; i < N_gp; i++) {
            grad[layout.gp_w_start + i] += grad_z[i];
        }

        // Add likelihood contributions to hyperparameters
        // (no Jacobian — cancels with NNGP normalizing constant)
        grad[layout.log_sigma2_gp_idx] += grad_log_sigma2_lik;
        grad[layout.log_phi_gp_idx] += grad_log_phi_lik;
    } else {
        // Centered: scatter residuals directly to GP w gradients
        for (int i = 0; i < N; i++) {
            int loc_i = data.gp_data.obs_to_loc[i];
            double dLL_dspatial = data.gp_data.shared
                ? (vec_grad_ws().resid_num[i] + vec_grad_ws().resid_denom[i])
                : vec_grad_ws().resid_num[i];
            grad[layout.gp_w_start + loc_i] += dLL_dspatial;
        }
    }

    re_gradient_nc_transform(data, layout, params.data(), grad.data(), sigma_re);

    // Log-posterior: the observation likelihood was accumulated above, and the
    // centred field's NNGP density in the same pass as its gradient.
    if (fuse_lp) {
        ratiod::LogPostOptions opts;
        opts.skip_obs_loop = true;
        if (!use_nc && nngp_grads.log_lik_valid) {
            opts.precomputed_gp_log_prior = &nngp_grads.log_lik;
        }
        *log_post_out = ratiod::compute_log_post_impl<double>(params, data, layout, opts)
                        + obs_log_lik;
    }

}

// =====================================================================
// Collapsed GP gradient (hand-coded)
// GP effects marginalized via inner Laplace — only hyperparams in HMC
// =====================================================================

// collapsed_gp_ws() declared earlier (shared with compute_log_post)

void compute_gradient_gp_collapsed(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    std::vector<double>& grad,
    double* log_post_out
) {
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

    // GP hyperparameters
    double sigma2_gp = std::exp(params[layout.log_sigma2_gp_idx]);
    double phi_gp = std::exp(params[layout.log_phi_gp_idx]);

    if (phi_gp < data.gp_phi_prior_lower || phi_gp > data.gp_phi_prior_upper) {
        if (log_post_out) *log_post_out = -INFINITY;
        return;
    }

    int N_gp = data.gp_data.n_obs;
    int N = data.N;
    bool is_binomial = (data.model_type == ModelType::BINOMIAL ||
                        data.model_type == ModelType::BETA_BINOMIAL);

    // ---- Inner Laplace: find w* ----
    double collapsed_lp = collapsed_gp_find_mode(
        beta_num, beta_denom, sigma2_gp, phi_gp,
        phi_num, phi_denom, data, collapsed_gp_ws());

    // ---- Prior gradients (outer params only) ----
    beta_gradient_prior(data, layout, beta_num, beta_denom, grad.data());
    re_gradient_prior(data, layout, re, grad.data(), sigma_re);
    phi_gradient_prior(data, layout, phi_num, phi_denom, grad.data());

    // GP hyperparameter priors
    grad[layout.log_sigma2_gp_idx] = ratiod::math::pc_prior_grad_log_sigma2(
        sigma2_gp, data.gp_sigma2_prior_U, data.gp_sigma2_prior_alpha);
    grad[layout.log_phi_gp_idx] = 1.0;  // Uniform prior Jacobian

    // ---- Data likelihood gradient at w* ----
    // Compute residuals at the mode w*
    std::vector<double> resid_num(N), resid_denom(N);
    collapsed_gp_compute_residuals(
        collapsed_gp_ws().w_star.data(), beta_num, beta_denom,
        phi_num, phi_denom, data,
        resid_num.data(), resid_denom.data());

    // Scatter to beta gradients + phi likelihood gradient
    for (int i = 0; i < N; i++) {
        for (int p = 0; p < data.p_num; p++) {
            grad[layout.beta_num_start + p] += resid_num[i] * data.X_num_flat[i * data.p_num + p];
        }
        if (!is_binomial) {
            for (int p = 0; p < data.p_denom; p++) {
                grad[layout.beta_denom_start + p] += resid_denom[i] * data.X_denom_flat[i * data.p_denom + p];
            }
        }
        // Scatter to RE gradients
        if (layout.has_re && data.re_group.size() > (size_t)i && data.re_group[i] > 0) {
            grad[layout.re_start + data.re_group[i] - 1] += resid_num[i] + resid_denom[i];
        }

        // Phi (dispersion) likelihood gradient
        if (layout.has_phi_num || layout.has_phi_denom) {
            int loc_i = data.gp_data.obs_to_loc[i];
            double eta_num_i = 0.0, eta_denom_i = 0.0;
            for (int p = 0; p < data.p_num; p++)
                eta_num_i += data.X_num_flat[i * data.p_num + p] * beta_num[p];
            if (!is_binomial) {
                for (int p = 0; p < data.p_denom; p++)
                    eta_denom_i += data.X_denom_flat[i * data.p_denom + p] * beta_denom[p];
            }
            eta_num_i += collapsed_gp_ws().w_star[loc_i];
            if (!is_binomial && data.gp_data.shared) eta_denom_i += collapsed_gp_ws().w_star[loc_i];
            if (layout.has_re && data.re_group.size() > (size_t)i && data.re_group[i] > 0) {
                double re_val = (data.re_parameterization == 1) ?
                    sigma_re * re[data.re_group[i] - 1] : re[data.re_group[i] - 1];
                eta_num_i += re_val;
                if (!is_binomial) eta_denom_i += re_val;
            }
            const ratiod::ObsDerivs od = obs_likelihood_derivs(
                data, layout, params.data(), i, eta_num_i, eta_denom_i, phi_num,
                phi_denom, grad.data());
            if (layout.has_phi_num) grad[layout.log_phi_num_idx] += od.phi_num * phi_num;
            if (layout.has_phi_denom) grad[layout.log_phi_denom_idx] += od.phi_denom * phi_denom;
        }
    }

    // ---- GP hyperparameter gradients from NNGP prior at w* ----
    // d/d(log sigma2) and d/d(log phi) of log p_NNGP(w*|sigma2,phi)
    // Use the existing NNGP gradient function
    ratiod_gp::NNGPGradients nngp_grads;
    std::vector<double> w_star_vec(collapsed_gp_ws().w_star.begin(),
                                    collapsed_gp_ws().w_star.end());
    ratiod_gp::gp_nngp_gradients(w_star_vec, sigma2_gp, phi_gp,
                                  data.gp_data, nngp_grads);
    grad[layout.log_sigma2_gp_idx] += nngp_grads.grad_log_sigma2;
    grad[layout.log_phi_gp_idx] += nngp_grads.grad_log_phi;

    // ---- Laplace correction gradient via numerical differentiation ----
    // The Laplace log-det depends on w* which depends on ALL params implicitly.
    // We compute d/dθ [-0.5 log det(W+Q)] numerically for each outer param,
    // using warm-started Newton solves (1-2 iters each from current w*).
    // For non-GP params (beta, phi, RE), we reuse the NNGP structure from
    // collapsed_gp_ws() (Q doesn't change), skipping NNGP rebuild.
    {
        const double eps = 1e-5;
        std::vector<double> params_pert = params;
        const int gp_idx1 = layout.log_sigma2_gp_idx;
        const int gp_idx2 = layout.log_phi_gp_idx;

        for (int j = 0; j < n_params; j++) {
            double orig = params_pert[j];
            bool is_gp_hyperparam = (j == gp_idx1 || j == gp_idx2);

            // Forward perturbation
            params_pert[j] = orig + eps;
            double sigma2_p = std::exp(params_pert[gp_idx1]);
            double phi_p = std::exp(params_pert[gp_idx2]);
            const double* beta_num_p = &params_pert[layout.beta_num_start];
            const double* beta_denom_p = is_binomial ? beta_num_p : &params_pert[layout.beta_denom_start];
            double phi_num_p = layout.has_phi_num ? std::exp(params_pert[layout.log_phi_num_idx]) : phi_num;
            double phi_denom_p = layout.has_phi_denom ? std::exp(params_pert[layout.log_phi_denom_idx]) : phi_denom;
            double ld_plus = laplace_log_det_full(
                beta_num_p, beta_denom_p, sigma2_p, phi_p,
                phi_num_p, phi_denom_p, data, collapsed_gp_ws().w_star,
                is_gp_hyperparam ? nullptr : &collapsed_gp_ws(),
                is_gp_hyperparam);

            // Backward perturbation
            params_pert[j] = orig - eps;
            sigma2_p = std::exp(params_pert[gp_idx1]);
            phi_p = std::exp(params_pert[gp_idx2]);
            beta_num_p = &params_pert[layout.beta_num_start];
            beta_denom_p = is_binomial ? beta_num_p : &params_pert[layout.beta_denom_start];
            phi_num_p = layout.has_phi_num ? std::exp(params_pert[layout.log_phi_num_idx]) : phi_num;
            phi_denom_p = layout.has_phi_denom ? std::exp(params_pert[layout.log_phi_denom_idx]) : phi_denom;
            double ld_minus = laplace_log_det_full(
                beta_num_p, beta_denom_p, sigma2_p, phi_p,
                phi_num_p, phi_denom_p, data, collapsed_gp_ws().w_star,
                is_gp_hyperparam ? nullptr : &collapsed_gp_ws(),
                is_gp_hyperparam);

            grad[j] += (ld_plus - ld_minus) / (2.0 * eps);
            params_pert[j] = orig;
        }
    }

    // ---- NC transform for RE ----
    re_gradient_nc_transform(data, layout, params.data(), grad.data(), sigma_re);

    // ---- Log-posterior ----
    if (log_post_out) {
        // Compute full log-posterior including priors on outer params
        // collapsed_lp already has data_ll + nngp_prior
        double lp = collapsed_lp;

        // Laplace correction: -0.5 * log det(W + Q) via sparse Cholesky
        lp += collapsed_gp_ws().laplace_log_det;

        // Beta priors
        for (int p = 0; p < data.p_num; p++)
            lp += -0.5 * beta_num[p] * beta_num[p] / (data.sigma_beta * data.sigma_beta);
        for (int p = 0; p < data.p_denom; p++)
            lp += -0.5 * beta_denom[p] * beta_denom[p] / (data.sigma_beta * data.sigma_beta);

        // RE priors
        if (layout.has_re) {
            int n_re = layout.re_end - layout.re_start;
            double sigma_re2 = sigma_re * sigma_re;
            for (int g = 0; g < n_re; g++) {
                if (data.re_parameterization == 1) {
                    // NC: z ~ N(0,1)
                    lp += -0.5 * re[g] * re[g];
                } else {
                    lp += -0.5 * re[g] * re[g] / sigma_re2;
                }
            }
            // sigma_re half-Cauchy prior (log-scale)
            double ratio = sigma_re / data.sigma_re_scale;
            lp += -std::log(1.0 + ratio * ratio)
                  + params[layout.log_sigma_re_idx];
        }

        // GP hyperparameter priors
        double sigma_gp = std::sqrt(sigma2_gp);
        double rate = -std::log(data.gp_sigma2_prior_alpha) / data.gp_sigma2_prior_U;
        lp += std::log(rate) - rate * sigma_gp - std::log(2.0 * sigma_gp)
              + params[layout.log_sigma2_gp_idx];
        // phi uniform + Jacobian
        lp += params[layout.log_phi_gp_idx]
              - std::log(data.gp_phi_prior_upper - data.gp_phi_prior_lower);

        // Phi (dispersion) priors
        if (layout.has_phi_num) {
            double log_phi = params[layout.log_phi_num_idx];
            lp += log_phi;  // Jacobian for log transform (exponential prior)
        }
        if (layout.has_phi_denom) {
            double log_phi = params[layout.log_phi_denom_idx];
            lp += log_phi;
        }

        *log_post_out = lp;
    }
}

// =====================================================================
// Collapsed ICAR/BYM2 gradient
// ICAR: H-mode (analytical envelope + analytical Laplace via implicit fn thm)
// BYM2: numerical fallback (H-mode planned)
// =====================================================================

void compute_gradient_gp_plus_temporal_gmrf(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    std::vector<double>& grad,
    double* log_post_out
) {
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

    int N_gp = data.gp_data.n_obs;
    double sigma2_gp = std::exp(params[layout.log_sigma2_gp_idx]);
    double phi_gp = std::exp(params[layout.log_phi_gp_idx]);
    std::vector<double> gp_w(N_gp);
    for (int i = 0; i < N_gp; i++) gp_w[i] = params[layout.gp_w_start + i];

    double tau_temporal = std::exp(params[layout.log_tau_temporal_idx]);
    int T_len = layout.temporal_end - layout.temporal_start;
    TemporalView tview;
    tview.read(params, data, layout);
    const double* phi_temporal = tview.phi;
    double rho_ar1 = read_temporal_rho_ar1(params, data, layout);

    // Prior gradients
    beta_gradient_prior(data, layout, beta_num, beta_denom, grad.data());
    re_gradient_prior(data, layout, re, grad.data(), sigma_re);
    phi_gradient_prior(data, layout, phi_num, phi_denom, grad.data());

    grad[layout.log_sigma2_gp_idx] = ratiod::math::pc_prior_grad_log_sigma2(
        sigma2_gp, data.gp_sigma2_prior_U, data.gp_sigma2_prior_alpha);
    grad[layout.log_phi_gp_idx] = 1.0;
    tau_temporal_prior_grad(data, layout, tau_temporal, grad.data());
    temporal_rho_prior_grad(data, layout, rho_ar1, grad.data());

    // NNGP prior gradients
    ratiod_gp::NNGPGradients nngp_grads;
    ratiod_gp::gp_nngp_gradients(gp_w, sigma2_gp, phi_gp, data.gp_data, nngp_grads);
    for (int i = 0; i < N_gp; i++) grad[layout.gp_w_start + i] += nngp_grads.grad_w[i];
    grad[layout.log_sigma2_gp_idx] += nngp_grads.grad_log_sigma2;
    grad[layout.log_phi_gp_idx] += nngp_grads.grad_log_phi;

    // Likelihood (vectorized eta + per-obs scatter)
    const int N = data.N;
    const bool is_binomial = (data.model_type == ModelType::BINOMIAL ||
                              data.model_type == ModelType::BETA_BINOMIAL);
    std::vector<double> grad_temporal_lik(T_len, 0.0);
    std::vector<int> obs_t_idx(N, -1);

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
        int loc_i = data.gp_data.obs_to_loc[i];
        double gp_effect = gp_w[loc_i];
        vec_grad_ws().eta_num[i] += gp_effect;
        if (!is_binomial && data.gp_data.shared) vec_grad_ws().eta_denom[i] += gp_effect;

        if (!data.temporal_time_idx.empty() && i < (int)data.temporal_time_idx.size() && data.temporal_time_idx[i] > 0) {
            int t = data.temporal_time_idx[i] - 1;
            int g = data.temporal_group_idx[i] - 1;
            int t_idx = g * data.n_times + t;
            if (t_idx >= 0 && t_idx < T_len) {
                obs_t_idx[i] = t_idx;
                vec_grad_ws().eta_num[i] += phi_temporal[t_idx];
                if (!is_binomial && data.temporal_shared) vec_grad_ws().eta_denom[i] += phi_temporal[t_idx];
            }
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

    for (int i = 0; i < N; i++) {
        double dLL_num = vec_grad_ws().resid_num[i];
        double dLL_denom = vec_grad_ws().resid_denom[i];
        if (layout.has_re && data.re_group[i] > 0) {
            grad[layout.re_start + data.re_group[i] - 1] += dLL_num + dLL_denom;
        }
        int loc_i = data.gp_data.obs_to_loc[i];
        grad[layout.gp_w_start + loc_i] += data.gp_data.shared ? (dLL_num + dLL_denom) : dLL_num;
        if (obs_t_idx[i] >= 0) grad_temporal_lik[obs_t_idx[i]] += data.temporal_shared ? (dLL_num + dLL_denom) : dLL_num;
    }

    // Temporal GMRF gradients
    temporal_gmrf_prior_grad(data, layout, tau_temporal, rho_ar1,
                             tview, T_len, grad_temporal_lik.data(), grad.data());

    // Non-centered RE chain rule transformation
    re_gradient_nc_transform(data, layout, params.data(), grad.data(), sigma_re);

    if (fuse_lp) *log_post_out = compute_log_post(params, data, layout, /*skip_obs_loop=*/true) + obs_log_lik;
}

// =====================================================================
// Temporal GP (standalone) hand-coded gradients; the block's density and
// gradient come from ratiod_temporal_gp::TemporalGPBlock
// =====================================================================

void compute_gradient_temporal_gp(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    std::vector<double>& grad,
    double* log_post_out
) {
    // Fused log-posterior: accumulate obs log-lik during gradient loop,
    // then add prior/structural terms via skip_obs_loop=true (avoids 2nd O(N) pass).
    // Also fuse temporal GP prior to avoid redundant NC forward pass in compute_log_post.
    const bool fuse_lp = (log_post_out != nullptr) && !layout.has_zi;
    if (log_post_out && layout.has_zi) *log_post_out = compute_log_post(params, data, layout);
    double obs_log_lik = 0.0;
    double tgp_lp_accum = 0.0;  // Accumulates temporal GP prior terms
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

    // Temporal effects: n_temporal_groups * n_times parameters
    int T_times = data.n_times;
    int T_len = layout.temporal_end - layout.temporal_start;

    RATIOD_TLS_WORKSPACE(ratiod_temporal_gp::TemporalGPBlock, tgp_block);
    tgp_block.forward(&params[layout.temporal_start],
                      params[layout.log_sigma2_temporal_gp_idx],
                      params[layout.logit_phi_temporal_gp_idx],
                      data.temporal_gp_data);
    const double* f_temporal = tgp_block.f();
    tgp_lp_accum = tgp_block.log_prior();

    // ---- Prior gradients ----
    beta_gradient_prior(data, layout, beta_num, beta_denom, grad.data());
    re_gradient_prior(data, layout, re, grad.data(), sigma_re);
    phi_gradient_prior(data, layout, phi_num, phi_denom, grad.data());

    // ---- Likelihood gradients (vectorized) ----
    // Thread-local buffer avoids heap allocation per gradient call
    RATIOD_TLS_WORKSPACE(std::vector<double>, grad_temporal_lik);
    grad_temporal_lik.assign(T_len, 0.0);

    const int N = data.N;
    const bool is_binomial = (data.model_type == ModelType::BINOMIAL ||
                              data.model_type == ModelType::BETA_BINOMIAL);

    // --- Pass 1: Vectorized eta computation (Eigen matvec + scalar expansion) ---
    using RowMajorMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using VectorXd = Eigen::VectorXd;


    fill_fixed_eta(vec_grad_ws(), data, beta_num, beta_denom);
    Eigen::Map<Eigen::VectorXd> eta_n(vec_grad_ws().eta_num.data(), N);

    // Add RE to eta
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

    // Add temporal GP effect to eta
    for (int i = 0; i < N; i++) {
        if (!data.temporal_time_idx.empty() && data.temporal_time_idx[i] > 0) {
            int t = data.temporal_time_idx[i] - 1;
            int g = (i < (int)data.temporal_group_idx.size() && data.temporal_group_idx[i] > 0)
                    ? data.temporal_group_idx[i] - 1 : 0;
            int flat_idx = g * T_times + t;
            if (flat_idx >= 0 && flat_idx < T_len) {
                vec_grad_ws().eta_num[i] += f_temporal[flat_idx];
                if (!is_binomial && data.temporal_shared) vec_grad_ws().eta_denom[i] += f_temporal[flat_idx];
            }
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

    // Scatter residuals to RE gradient
    if (layout.has_re) {
        for (int i = 0; i < N; i++) {
            if (data.re_group[i] > 0)
                grad[layout.re_start + data.re_group[i] - 1] += vec_grad_ws().resid_num[i] + vec_grad_ws().resid_denom[i];
        }
    }

    // Scatter residuals to temporal lik gradient
    for (int i = 0; i < N; i++) {
        if (!data.temporal_time_idx.empty() && data.temporal_time_idx[i] > 0) {
            int t = data.temporal_time_idx[i] - 1;
            int g = (i < (int)data.temporal_group_idx.size() && data.temporal_group_idx[i] > 0)
                    ? data.temporal_group_idx[i] - 1 : 0;
            int flat_idx = g * T_times + t;
            if (flat_idx >= 0 && flat_idx < T_len)
                grad_temporal_lik[flat_idx] += data.temporal_shared
                    ? (vec_grad_ws().resid_num[i] + vec_grad_ws().resid_denom[i])
                    : vec_grad_ws().resid_num[i];
        }
    }

    tgp_block.backward(grad_temporal_lik.data(), &grad[layout.temporal_start],
                       grad[layout.log_sigma2_temporal_gp_idx],
                       grad[layout.logit_phi_temporal_gp_idx]);

    // Phi gradients already applied by dispatch_residuals_and_beta_grads() (log-transform Jacobian included)

    // Non-centered RE chain rule transformation
    re_gradient_nc_transform(data, layout, params.data(), grad.data(), sigma_re);

    if (fuse_lp) {
        // The block's density is already evaluated; compute_log_post adds it
        // instead of evaluating it a second time.
        *log_post_out = compute_log_post(params, data, layout, /*skip_obs_loop=*/true,
                                         &tgp_lp_accum) + obs_log_lik;
    }
}

// =====================================================================
// SVC gradient (hand-coded, ~3x faster than autodiff)
// Uses the spatial GP's analytical NNGP gradients for the field prior
// =====================================================================


}  // namespace ratiod_hmc
