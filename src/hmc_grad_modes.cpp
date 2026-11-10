#include <RcppEigen.h>
#include "hmc_grad_helpers.h"
#include "hmc_field_coordinates.h"

using namespace Rcpp;

namespace ratiod_hmc {
double icar_quadratic_form(
    const std::vector<double>& phi,
    const ModelData& data
) {
  return icar_quadratic_form_ptr(phi.data(), data.n_spatial_units, data);
}
// =====================================================================
// Log-posterior computation
// =====================================================================

// One density, in one place, so a term cannot reach the value the H path
// reports without also reaching the gradient the autodiff modes take of it.
// Collapsed marginals, proper CAR, parallel observation loops and fused
// likelihood values are selected by LogPostOptions and scalar type.
double compute_log_post(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    bool skip_obs_loop,
    const double* precomputed_tgp_log_prior
) {
  if (field_coordinates && !field_coordinates->empty()) {
    std::vector<double> x;
    const double jacobian = field_coordinates->forward(params, x);
    FieldCoordinateScope raw(nullptr);
    return compute_log_post(x, data, layout, skip_obs_loop, precomputed_tgp_log_prior) + jacobian;
  }
  ratiod::LogPostOptions opts;
  opts.skip_obs_loop = skip_obs_loop;
  opts.precomputed_tgp_log_prior = precomputed_tgp_log_prior;
  return ratiod::compute_log_post_impl<double>(params, data, layout, opts);
}

// =====================================================================
// Analytical gradient for simple Poisson-Gamma models
// O(n) instead of O(n*p) - huge speedup for typical models
// =====================================================================


// =====================================================================
// Numerical gradients
// =====================================================================

// Finite-difference gradient of `f` at `params`, step h = 1e-5. The
// two-point central difference has truncation error h^2 f^(3)/6; the
// five-point stencil (f(x-2h) - 8 f(x-h) + 8 f(x+h) - f(x+2h)) / 12h has
// h^4 f^(5)/30, which stays below the gradient check's tolerance when a
// covariate on a large scale makes the higher derivatives along its
// coefficient large.
template <typename LogPost>
void finite_difference_gradient(LogPost&& f, const std::vector<double>& params,
                                std::vector<double>& grad, bool five_point) {
  const int n = static_cast<int>(params.size());
  const double h = 1e-5;
  grad.resize(n);
  std::vector<double> p = params;
  for (int i = 0; i < n; i++) {
    const double x = params[i];
    p[i] = x + h; const double f1p = f(p);
    p[i] = x - h; const double f1m = f(p);
    if (five_point) {
      p[i] = x + 2.0 * h; const double f2p = f(p);
      p[i] = x - 2.0 * h; const double f2m = f(p);
      grad[i] = (f2m - 8.0 * f1m + 8.0 * f1p - f2p) / (12.0 * h);
    } else {
      grad[i] = (f1p - f1m) / (2.0 * h);
    }
    p[i] = x;
  }
}

// Gradient of compute_log_post (the density the hand-coded gradients
// differentiate), the sampler's gradient in mode 'N'.
void compute_gradient_numerical(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    std::vector<double>& grad,
    double* log_post_out
) {
  if (log_post_out) *log_post_out = compute_log_post(params, data, layout);
  finite_difference_gradient(
    [&](const std::vector<double>& q) { return compute_log_post(q, data, layout); },
    params, grad, false);
}

// Five-point reference gradient for the gradient checks: of
// compute_log_post_impl<double> when `impl`, the density the autodiff modes
// differentiate (its parameterization can differ from compute_log_post's,
// e.g. non-centred random effects), else of compute_log_post.
void reference_gradient(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    std::vector<double>& grad,
    bool impl
) {
  if (impl) {
    finite_difference_gradient(
      [&](const std::vector<double>& q) { return ratiod::compute_log_post_impl(q, data, layout); },
      params, grad, true);
  } else {
    finite_difference_gradient(
      [&](const std::vector<double>& q) { return compute_log_post(q, data, layout); },
      params, grad, true);
  }
}

// =====================================================================
// Unified gradient interface
// =====================================================================

// Move every structured field off zero, deterministically.
//
// initialize_hmc_params_full() starts each field block at exactly 0.0, and a
// quadratic-form prior -tau/2 phi' Q phi contributes nothing to any gradient
// there: a wrong Q, a sign error on it, a missing quadratic term and an absent
// sum-to-zero penalty are all invisible, because the analytic and the numerical
// gradient agree on the same zero. The offline harness evaluates away from the
// origin for exactly this reason (src/test_gradient_check.cpp). The offset is
// small, bounded and a function of the index alone, so the check point is
// reproducible and stays inside any prior's support.
std::vector<double> offset_structured_blocks(const std::vector<double>& q,
                                             const ParamLayout& layout) {
  std::vector<double> out = q;
  auto jitter = [&](int start, int end) {
    if (start < 0 || end <= start) return;
    for (int i = start; i < end && i < static_cast<int>(out.size()); i++) {
      out[i] += 0.1 * std::sin(1.7 * static_cast<double>(i) + 0.3);
    }
  };

  jitter(layout.spatial_start, layout.spatial_end);
  jitter(layout.theta_bym2_start, layout.theta_bym2_end);
  jitter(layout.temporal_start, layout.temporal_end);
  jitter(layout.hsgp_beta_start, layout.hsgp_beta_end);
  jitter(layout.tvc_w_start, layout.tvc_w_end);
  jitter(layout.svc_w_start, layout.svc_w_end);
  jitter(layout.gp_w_start, layout.gp_w_end);
  jitter(layout.gp_local_start, layout.gp_local_end);
  jitter(layout.gp_regional_start, layout.gp_regional_end);
  jitter(layout.trend_start, layout.trend_end);
  jitter(layout.seasonal_start, layout.seasonal_end);
  jitter(layout.short_term_start, layout.short_term_end);
  jitter(layout.st_delta_start, layout.st_delta_end);
  jitter(layout.latent_factor_start, layout.latent_factor_end);
  for (size_t t = 0; t < layout.re_start_multi.size(); t++) {
    jitter(layout.re_start_multi[t], layout.re_end_multi[t]);
  }
  return out;
}

// Runtime gradient check: compare compute_gradient() dispatcher output against
// numerical gradients of the density it is supposed to be the gradient of,
// once, before sampling starts. Covers every specialized gradient function the
// dispatcher can select (GP, HSGP, SVC, TVC, MSGP, spatiotemporal, ...), at a
// point where the structured fields are nonzero.
bool verify_gradient_runtime(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    double tol,
    std::string& report
) {
  std::vector<double> grad_active, grad_numerical;
  compute_gradient(params, data, layout, grad_active, nullptr);

  // For A_r/A modes: use compute_log_post_impl<double> as numerical reference,
  // since those modes differentiate compute_log_post_impl<T> (which may use
  // different parameterization than H-mode's compute_log_post, e.g. non-centered RE).
  // For H mode: use compute_log_post (H-mode function) as reference.
  if (g_gradient_mode == GradientMode::AUTODIFF_ARENA ||
      g_gradient_mode == GradientMode::AUTODIFF_FORWARD) {
    reference_gradient(params, data, layout, grad_numerical, true);
  } else {
    reference_gradient(params, data, layout, grad_numerical, false);
  }

  double max_diff = 0.0;
  int worst_idx = -1;
  for (size_t i = 0; i < grad_active.size(); i++) {
    double diff = std::abs(grad_active[i] - grad_numerical[i]);
    double scale = std::max(1.0, std::max(std::abs(grad_active[i]),
                                           std::abs(grad_numerical[i])));
    double rel_diff = diff / scale;
    if (rel_diff > max_diff) {
      max_diff = rel_diff;
      worst_idx = i;
    }
  }

  if (max_diff > tol) {
    char buf[200];
    std::snprintf(buf, sizeof(buf),
                  "parameter %d: gradient %.8e, finite difference %.8e "
                  "(relative difference %.2e)",
                  worst_idx + 1, grad_active[worst_idx], grad_numerical[worst_idx],
                  max_diff);
    report = buf;
    return false;
  }
  return true;
}

// Run the runtime check and fall back to numerical gradients if it fails.
// Both the single-chain and the multi-chain entry points call this, so a
// mismatch is reported the same way at either -- an R-level warning, which is
// what an expect_warning(..., NA) guard in the tests can see. Both call sites
// are on the R thread, before any parallel region opens.
void verify_gradient_or_fallback(const std::vector<double>& q_init,
                                 const ModelData& data,
                                 const ParamLayout& layout) {
  if (g_gradient_mode == GradientMode::NUMERICAL) return;

  const std::vector<double> probe = offset_structured_blocks(q_init, layout);
  std::string report;
  if (verify_gradient_runtime(probe, data, layout, 1e-4, report)) return;

  g_gradient_mode = GradientMode::NUMERICAL;
  Rcpp::warning(
    "The model's gradient disagrees with its finite difference at " + report +
    "; sampling with finite-difference gradients (gradient_mode = \"N\"), "
    "which is slow. Please report this at "
    "https://github.com/gcol33/tulpaRatio/issues"
  );
}

// =====================================================================
// Autodiff gradient (O(n) - works for ALL models)
// =====================================================================


// =====================================================================
// Arena-based reverse-mode autodiff gradient (O(N) - fast, all models)
// Uses contiguous SoA memory layout with pre-computed partials.
// ~10-30x faster than tape autodiff, within 50% of hand-coded speed.
// =====================================================================

void compute_gradient_arena(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    std::vector<double>& grad,
    double* log_post_out
) {
    using namespace tulpa::arena;

    int n_nodes_used = 0;
    {
        // Thread-safe: each call gets its own arena via RAII
        ArenaScope scope;
        Arena* arena = scope.arena();

        // Create autodiff variables from parameters
        std::vector<Var> params_ar = make_vars(arena, params);

        // Compute log posterior using templated implementation
        Var log_post = ratiod::compute_log_post_impl(params_ar, data, layout);

        // Extract log_post value before backward pass
        if (log_post_out) *log_post_out = log_post.val();

        // Backward pass to compute gradients
        log_post.backward();

        // Extract gradients
        grad = get_adjoints(params_ar);

        n_nodes_used = arena->size();
        // ArenaScope destructor handles cleanup
    }

    (void)n_nodes_used;  // suppress unused warning
}

// =====================================================================
// Forward-mode autodiff gradient (O(n×p) - but ~10x faster than tape)
// Uses dual numbers for efficient gradient computation without heap allocation
// =====================================================================

void compute_gradient_forward(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    std::vector<double>& grad,
    double* log_post_out
) {
    int n_params = static_cast<int>(params.size());
    grad.assign(n_params, 0.0);

    // Forward-mode: compute one gradient component per forward pass
    // Seed each parameter in turn and evaluate
    std::vector<fwd::Dual> params_dual(n_params);

    for (int i = 0; i < n_params; i++) {
        // Seed parameter i: value=params[i], gradient=1.0
        // All others: value=params[j], gradient=0.0
        for (int j = 0; j < n_params; j++) {
            params_dual[j].val = params[j];
            params_dual[j].grad = (j == i) ? 1.0 : 0.0;
        }

        // Compute log posterior with dual numbers
        fwd::Dual log_post = ratiod::compute_log_post_impl(params_dual, data, layout);

        // Extract gradient component
        grad[i] = log_post.grad;

        // Extract log_post value on first pass (free: already computed)
        if (i == 0 && log_post_out) *log_post_out = log_post.val;
    }
}

// =====================================================================
// RE gradient helpers for specialized gradient functions
// Handles both centered and non-centered parameterizations correctly
// =====================================================================


}  // namespace ratiod_hmc
