#include <RcppEigen.h>
#include "hmc_grad_helpers.h"
#include "hmc_field_coordinates.h"
#include "hmc_integrator.h"
#include "hmc_gp_collapsed.h"
#include "hmc_icar_collapsed.h"
#include <tulpa/walnuts.h>

using namespace Rcpp;

namespace ratiod_hmc {
class WelfordStats {
public:
  int n;
  std::vector<double> mean;
  std::vector<double> M2;  // Sum of squared differences from mean

  WelfordStats(int dim) : n(0), mean(dim, 0.0), M2(dim, 0.0) {}

  void update(const std::vector<double>& x) {
    n++;
    for (size_t i = 0; i < x.size(); i++) {
      double delta = x[i] - mean[i];
      mean[i] += delta / n;
      double delta2 = x[i] - mean[i];
      M2[i] += delta * delta2;
    }
  }

  std::vector<double> variance() const {
    std::vector<double> var(mean.size());
    if (n < 2) {
      // Return unit variance if not enough samples
      std::fill(var.begin(), var.end(), 1.0);
    } else {
      for (size_t i = 0; i < mean.size(); i++) {
        var[i] = M2[i] / (n - 1);
        // Ensure minimum variance to avoid numerical issues
        if (var[i] < 1e-6) var[i] = 1e-6;
      }
    }
    return var;
  }

  // Get inverse mass matrix (= variance, regularized for stability)
  // For HMC: M = diag(1/var), so M^{-1} = diag(var)
  // High variance parameters should move faster in position space
  // Uses Stan-style Bayesian shrinkage toward unit variance to prevent
  // extreme mass matrix entries from small sample sizes
  std::vector<double> inv_mass() const {
    auto var = variance();
    for (size_t i = 0; i < var.size(); i++) {
      var[i] = shrink_warmup_cov(var[i], n, true);
      // Safety clamp for extreme values
      var[i] = std::max(1e-3, std::min(var[i], 1e3));
    }
    return var;
  }

  // Get sqrt of mass matrix for momentum sampling
  // Since M = diag(1/var), sqrt(M) = diag(1/sqrt(var))
  // p ~ N(0, M), so p_i = z_i / sqrt(var_i)
  // Uses the same regularized variance as inv_mass()
  std::vector<double> sqrt_mass() const {
    auto inv_m = inv_mass();
    std::vector<double> sqrt_m(inv_m.size());
    for (size_t i = 0; i < inv_m.size(); i++) {
      sqrt_m[i] = 1.0 / std::sqrt(inv_m[i]);
    }
    return sqrt_m;
  }

  void reset() {
    n = 0;
    std::fill(mean.begin(), mean.end(), 0.0);
    std::fill(M2.begin(), M2.end(), 0.0);
  }
};

// =====================================================================
// Leapfrog integrator
// =====================================================================

// The WALNUTS model over this sampler's metric
// =====================================================================

// What tulpa::walnuts_transition() reads of a chain: the dispatched gradient,
// and the kinetic energy, metric product and momentum draw of the adapted
// mass matrix, with the drift theta += step * M^-1 rho through the same
// product the NUTS U-turn criterion uses.
struct RatioWalnutsModel {
  const DenseMassMatrix& mass;
  GradientFn gradient_fn;
  const ModelData& data;
  const ParamLayout& layout;
  int n;

  void gradient(const std::vector<double>& theta, std::vector<double>& grad,
                double* log_density) {
    gradient_fn(theta, data, layout, grad, log_density);
  }
  double kinetic_energy(const double* rho) const {
    return mass.kinetic_energy(rho);
  }
  void inv_mass_times_p(const double* rho, double* out) const {
    mass.inv_mass_times_p(rho, out);
  }
  void drift(double step, double* theta, const double* rho,
             double* scratch) const {
    mass.inv_mass_times_p(rho, scratch);
    for (int i = 0; i < n; i++) theta[i] += step * scratch[i];
  }
  template <class Rng>
  void sample_momentum(double* rho, Rng& rng) const {
    mass.sample_momentum(rho, rng);
  }
};

// =====================================================================
// Run single HMC chain
// =====================================================================

FieldCoordinates make_field_coordinates(const std::vector<double>& initial,
                                         const ModelData& data,
                                         const ParamLayout& layout) {
  FieldCoordinates out;
  out.gradient = resolve_gradient_fn(g_gradient_mode, data, layout);
  struct Block { int begin, end, scale; double sign; int width; bool constant, linear, standardized; };
  std::vector<Block> blocks;
  auto add = [&](int begin, int end, int scale, double sign, int width = 0,
                 bool constant = false, bool linear = false, bool standardized = false) {
    if (begin >= 0 && end > begin && scale >= 0)
      blocks.push_back({begin, end, scale, sign, width, constant, linear, standardized});
  };
  if (data.auto_spatial_coordinates && layout.has_spatial && !layout.is_bym2)
    add(layout.spatial_start, layout.spatial_end, layout.log_tau_spatial_idx, 1);
  if (data.auto_temporal_coordinates && layout.has_temporal && !layout.is_temporal_gp)
    add(layout.temporal_start, layout.temporal_end, layout.log_tau_temporal_idx, 1,
        data.n_times, false, data.temporal_type == TemporalType::RW2 && !data.temporal_cyclic);
  if (data.auto_gp_coordinates && layout.is_gp && !layout.is_gp_collapsed)
    add(layout.gp_w_start, layout.gp_w_end, layout.log_sigma2_gp_idx,
        1, 0, false, false, true);
  if (data.auto_temporal_gp_coordinates && layout.is_temporal_gp)
    add(layout.temporal_start, layout.temporal_end, layout.log_sigma2_temporal_gp_idx,
        1, 0, false, false, true);
  if (data.auto_svc_coordinates && layout.has_svc && !data.svc_is_hsgp) {
    const int width = (layout.svc_w_end - layout.svc_w_start) / data.svc_data.n_svc;
    for (int j = 0; j < data.svc_data.n_svc; ++j)
      add(layout.svc_w_start + j * width, layout.svc_w_start + (j + 1) * width,
          layout.log_sigma2_svc_start + j, 1, 0, false, false, true);
  }
  if (data.auto_multiscale_coordinates && layout.has_multiscale_temporal) {
    add(layout.trend_start, layout.trend_end, layout.log_sigma2_trend_idx, -1,
        data.multiscale_temporal_data.n_times, false,
        data.multiscale_temporal_data.trend_type == TemporalType::RW2);
    add(layout.seasonal_start, layout.seasonal_end, layout.log_sigma2_seasonal_idx, -1);
    add(layout.short_term_start, layout.short_term_end, layout.log_sigma2_short_idx, -1);
  }
  if (layout.has_tvc) {
    const auto& tvc = data.tvc_data;
    for (int g = 0; g < tvc.n_groups; ++g) for (int j = 0; j < tvc.n_tvc; ++j) {
      const int begin = layout.tvc_w_start + (g * tvc.n_tvc + j) * tvc.n_times;
      const bool intrinsic = tvc.structure == TemporalType::RW1 || tvc.structure == TemporalType::RW2;
      add(begin, begin + tvc.n_times, layout.log_tau_tvc_start + j, 1, tvc.n_times,
          intrinsic, tvc.structure == TemporalType::RW2 && !tvc.cyclic);
    }
  }
  if (blocks.empty()) return out;
  std::vector<double> origin = initial;
  for (const auto& block : blocks)
    std::fill(origin.begin() + block.begin, origin.begin() + block.end, 0.0);
  std::vector<double> score;
  out.gradient(origin, data, layout, score, nullptr);
  const double full = compute_log_post(origin, data, layout);
  const double prior = compute_log_post(origin, data, layout, true);
  // A balanced direction measures one Rayleigh quotient per block. Its fixed
  // curvature sets a coordinate only; the density keeps every prior and
  // likelihood term. Setup takes a constant number of density evaluations
  // per block, independent of its number of coefficients.
  for (const auto& block : blocks) {
    std::vector<double> plus = origin, minus = origin;
    const double h = std::pow(std::numeric_limits<double>::epsilon(), 0.25) *
        std::sqrt(1 + std::abs(full));
    const FieldCoordinate geometry{block.begin, block.end, block.scale, block.sign,
        1, 0, block.width, block.constant, block.linear};
    if (geometry.rank() == 0) continue;
    std::vector<double> direction(origin.size(), 0.0);
    for (int i = block.begin; i < block.end; ++i) direction[i] = i % 2 ? 1 : -1;
    double denominator = 0;
    for (int start = block.begin; start < block.end; start += geometry.width()) {
      const auto null = geometry.null_component(direction, start);
      for (int t = 0; t < geometry.width(); ++t) {
        const int i = start + t;
        const double shift = h * (direction[i] - null.at(t));
        plus[i] += shift;
        minus[i] -= shift;
        denominator += shift * shift;
      }
    }
    if (!(denominator > 0)) continue;
    const double prior_curve = -(compute_log_post(plus, data, layout, true) -
        2 * prior + compute_log_post(minus, data, layout, true)) / denominator;
    const double full_curve = -(compute_log_post(plus, data, layout) -
        2 * full + compute_log_post(minus, data, layout)) / denominator;
    if (!(prior_curve > 0) || !std::isfinite(prior_curve) || !std::isfinite(full_curve))
      continue;
    double score_power = 0.0;
    for (int i = block.begin; i < block.end; ++i) score_power += score[i] * score[i];
    const double likelihood_information = std::max(0.0, full_curve - prior_curve) +
        score_power / (block.end - block.begin);
    const double scaled = (block.standardized ? likelihood_information : prior_curve) *
        std::exp(-block.sign * initial[block.scale]);
    const double fixed = block.standardized ? prior_curve : likelihood_information;
    out.entries.push_back({block.begin, block.end, block.scale, block.sign, scaled, fixed,
        block.width, block.constant, block.linear});
  }
  return out;
}

// Pure C++ version - safe for OpenMP parallel regions
namespace {
class HmcChain {

  const std::vector<double>& q_init;
  const ModelData& data;
  const ParamLayout& layout;
  int n_iter, n_warmup, L, chain_id;
  unsigned int seed;
  bool verbose;
  int max_treedepth;
  MassMatrixType metric_type;
  double adapt_delta;
  const tulpa::WalnutsConfig* walnuts;
  int n_params, n_sample;
  bool use_nuts;
  HMCResultCpp result;
  std::mt19937 rng, st_draw_rng;
  std::normal_distribution<double> normal;
  std::uniform_real_distribution<double> unif;
  std::vector<double> q, current_grad;
  FieldCoordinates coordinates;
  double log_prob_current, epsilon, target_boost, nuts_target_accept;
  DualAveraging da{1.0, 0, 0.0};
  static constexpr int DENSE_MAX_PARAMS = 200;
  MassMatrixType effective_metric;
  bool auto_selected_diag;
  std::vector<std::pair<int,int>> block_specs;
  DenseMassMatrix mass;
  WelfordStats mass_stats{0};
  WelfordCovStats cov_stats{0};
  bool use_mass_matrix, use_lbfgs, lbfgs_initialized, lbfgs_warmup_done;
  ratiod_gp::LBFGSState lbfgs_state;
  std::vector<double> q_prev, grad_prev;
  int init_buffer, term_buffer, init_window, next_window_idx;
  std::vector<int> mass_window_ends;
  NUTSWorkspace nuts_ws;
  std::vector<double> _nuts_p, _nuts_q_proposal, _nuts_grad_proposal;
  int sample_idx, n_accept, n_divergent;
  int nuts_probe_window, nuts_probe_maxd;
  bool nuts_probing, use_walnuts;
  tulpa::WalnutsWorkspace walnuts_ws;
  int warmup_total_leapfrog;
  double alpha, iter_energy;
  bool divergent;
  int iter_n_leapfrog, iter_treedepth;

public:
  HmcChain(const std::vector<double>& initial, const ModelData& model,
           const ParamLayout& parameters, int iterations, int warmup_iterations,
           int leapfrog_steps, int id, unsigned int random_seed, bool print_progress,
           int tree_depth, MassMatrixType metric, double target_accept,
           const tulpa::WalnutsConfig* walnuts_config)
      : q_init(initial), data(model), layout(parameters), n_iter(iterations),
        n_warmup(warmup_iterations), L(leapfrog_steps), chain_id(id), seed(random_seed),
        verbose(print_progress), max_treedepth(tree_depth), metric_type(metric),
        adapt_delta(target_accept), walnuts(walnuts_config) {}

  HMCResultCpp run() {
    initialize_state();
    FieldCoordinateScope coordinate_scope(&coordinates);
    initialize_warmup();
    warmup();
    sample();


    result.epsilon = da.final_epsilon();

    // Diagnostic stats - only when verbose
    if (verbose) {
      int sampling_total_lf = 0;
      for (int i = 0; i < result.n_sample; i++) sampling_total_lf += result.n_leapfrog[i];
      REprintf("  [STATS] Chain %d: metric=%s, adapted=%d, warmup_LF=%d, sampling_LF=%d, total_LF=%d, epsilon=%.6f\n",
               chain_id + 1,
               metric_name(mass.type),
               (int)mass.adapted,
               warmup_total_leapfrog, sampling_total_lf,
               warmup_total_leapfrog + sampling_total_lf, result.epsilon);
    }

    return std::move(result);
  }

private:
  void initialize_state() {
    n_params = q_init.size();
    n_sample = n_iter - n_warmup;
    use_nuts = (L == 0);


    result.n_params_stored = n_params;
    result.samples_flat.resize(static_cast<size_t>(n_sample) * n_params);
    result.log_prob.resize(n_sample);
    result.energy.resize(n_sample, NA_REAL);
    result.accept_prob.resize(n_sample);
    result.n_leapfrog.resize(n_sample, L);
    result.divergent.resize(n_sample, 0);
    result.treedepth.resize(n_sample, 0);
    result.n_warmup = n_warmup;
    result.n_sample = n_sample;
    result.chain_id = chain_id;
    result.n_max_treedepth = 0;

    // Collapsed GP: allocate w* storage
    if (data.gp_collapsed && data.has_gp) {
        result.n_gp_collapsed = data.gp_data.n_obs;
        result.gp_w_star_flat.resize(static_cast<size_t>(n_sample) * data.gp_data.n_obs, 0.0);
    }

    // Collapsed GP interaction: one draw of the field per saved iteration
    if (layout.st_collapsed) {
        result.n_st_collapsed = data.spatiotemporal_data.n_params;
        result.st_delta_flat.resize(
            static_cast<size_t>(n_sample) * result.n_st_collapsed, 0.0);
    }

    // Collapsed ICAR/BYM2: allocate phi*/theta* storage
    if (data.icar_collapsed || data.bym2_collapsed) {
        int S = data.n_spatial_units;
        result.n_icar_collapsed = S;
        result.icar_phi_star_flat.resize(static_cast<size_t>(n_sample) * S, 0.0);
        if (data.bym2_collapsed) {
            result.bym2_theta_star_flat.resize(static_cast<size_t>(n_sample) * S, 0.0);
        }
    }

    rng = std::mt19937(seed + chain_id * 12345);
    st_draw_rng = std::mt19937(seed + chain_id * 12345 + 7919);
    normal = std::normal_distribution<double>(0.0, 1.0);
    unif = std::uniform_real_distribution<double>(0.0, 1.0);

    // Reset VecGradWorkspace cache for new model fit
    vec_grad_ws().cached_data_id = 0;

    q = q_init;
    coordinates = make_field_coordinates(q_init, data, layout);
    coordinates.inverse(q);
  }
  void initialize_warmup() {
    // For NUTS: fuse initial log_post + gradient into single O(N) pass
    current_grad.resize(n_params);

    if (use_nuts) {
      compute_gradient(q, data, layout, current_grad, &log_prob_current);
    } else {
      log_prob_current = compute_log_post(q, data, layout);
    }

    epsilon = find_reasonable_epsilon(q, data, layout, rng);

    // Compute target_boost for challenging model combinations
    // MSGP and GP with temporal are particularly challenging
    target_boost = 0.0;
    if (data.has_multiscale_gp) {
      target_boost += 0.10;  // MSGP models need higher target acceptance
      if (layout.has_temporal) {
        target_boost += 0.05;  // MSGP + temporal is even more challenging
      }
    } else if (data.spatial_type == SpatialType::GP) {
      target_boost += 0.05;  // GP models moderately challenging
      if (layout.has_temporal) {
        target_boost += 0.05;  // GP + temporal combination
      }
    }
    da = DualAveraging(epsilon, n_params, target_boost);

    nuts_target_accept = 0.80;
    if (use_nuts) {
      if (adapt_delta > 0) {
        // User override
        nuts_target_accept = adapt_delta;
      } else {
        // Auto-select based on model complexity
        nuts_target_accept = 0.80;  // Stan default base

        // BYM2: high correlation between ICAR phi + unstructured theta
        if (data.spatial_type == SpatialType::BYM2) {
          nuts_target_accept = 0.90;
        }
        // ICAR: correlated spatial params need slightly higher target
        else if (data.spatial_type == SpatialType::ICAR) {
          nuts_target_accept = 0.85;
        }

        // Correlated random slopes add funnel geometry
        if (data.has_re_correlated_slopes) {
          nuts_target_accept = std::max(nuts_target_accept, 0.90);
        }

        // Temporal GP NC: z ~ N(0,1) decorrelates parameters, lower target OK
        // Benchmarked: 0.70 gives 20% fewer LF steps and 50% less seed variance
        if (layout.is_temporal_gp && nuts_target_accept > 0.70) {
          nuts_target_accept = 0.70;
        }

        nuts_target_accept = std::min(0.99, nuts_target_accept);
      }
      da.target_accept = nuts_target_accept;
    }

    // current_grad already computed above (fused with log_prob for NUTS)


    effective_metric = metric_type;
    auto_selected_diag = false;
    // Block specs for BLOCK_DIAG: (start_index, block_size) pairs


    if (effective_metric == MassMatrixType::AUTO) {
      if (layout.is_bym2 && layout.log_sigma_bym2_idx >= 0 &&
          layout.logit_rho_bym2_idx == layout.log_sigma_bym2_idx + 1) {
        block_specs.push_back({layout.log_sigma_bym2_idx, 2});
      }
      if (layout.is_gp && layout.log_sigma2_gp_idx >= 0 &&
          layout.log_phi_gp_idx == layout.log_sigma2_gp_idx + 1) {
        block_specs.push_back({layout.log_sigma2_gp_idx, 2});
      }
      if (layout.is_multiscale_gp) {
        if (layout.log_sigma2_gp_local_idx >= 0 &&
            layout.log_phi_gp_local_idx == layout.log_sigma2_gp_local_idx + 1) {
          block_specs.push_back({layout.log_sigma2_gp_local_idx, 2});
        }
        if (layout.log_sigma2_gp_regional_idx >= 0 &&
            layout.log_phi_gp_regional_idx == layout.log_sigma2_gp_regional_idx + 1) {
          block_specs.push_back({layout.log_sigma2_gp_regional_idx, 2});
        }
      }
      if (layout.has_svc && layout.log_sigma2_svc_start >= 0 &&
          layout.range_svc_start >= 0) {
        int n_svc = layout.log_sigma2_svc_end - layout.log_sigma2_svc_start;
        for (int t = 0; t < n_svc; t++) {
          int sigma_idx = layout.log_sigma2_svc_start + t;
          int phi_idx = layout.range_svc_start + t;
          if (phi_idx == sigma_idx + n_svc) {
            // SVC sigma2 and phi are in separate contiguous blocks; can't form a 2x2 block
            // unless they're consecutive. Skip non-consecutive pairs.
          }
        }
        // SVC layout: [sigma2_0, sigma2_1, ..., phi_0, phi_1, ...]
        // These aren't consecutive pairs, so we'd need per-SVC blocks of non-contiguous params.
        // For now, only handle n_svc=1 where sigma2 and phi are adjacent:
        if (n_svc == 1 && layout.range_svc_start == layout.log_sigma2_svc_start + 1) {
          block_specs.push_back({layout.log_sigma2_svc_start, 2});
        }
      }
      if (layout.is_st_gp && layout.log_phi_st_space_idx >= 0 &&
          layout.log_phi_st_time_idx == layout.log_phi_st_space_idx + 1) {
        block_specs.push_back({layout.log_phi_st_space_idx, 2});
      }
      if (layout.has_multiscale_temporal) {
        // Multiscale temporal hyperparams form a natural block (3-4 params):
        // log_sigma2_trend, log_sigma2_seasonal, log_sigma2_short [, logit_rho_short]
        // These are correlated but the temporal effects themselves (phi) are not
        // strongly correlated with the hyperparams → BLOCK_DIAG, not full DENSE.
        int ms_block_start = -1;
        int ms_block_size = 0;
        if (layout.log_sigma2_trend_idx >= 0) {
          ms_block_start = layout.log_sigma2_trend_idx;
          ms_block_size = 1;
        }
        if (layout.log_sigma2_seasonal_idx >= 0) {
          if (ms_block_start < 0) ms_block_start = layout.log_sigma2_seasonal_idx;
          ms_block_size++;
        }
        if (layout.log_sigma2_short_idx >= 0) {
          if (ms_block_start < 0) ms_block_start = layout.log_sigma2_short_idx;
          ms_block_size++;
        }
        if (layout.logit_rho_short_idx >= 0) {
          ms_block_size++;
        }
        if (ms_block_start >= 0 && ms_block_size >= 2 && ms_block_size <= 4) {
          block_specs.push_back({ms_block_start, ms_block_size});
        }
      }
      // Correlated slopes: Cholesky params form a natural block
      if (layout.has_re_correlated_slopes) {
        for (size_t t = 0; t < layout.chol_re_start_multi.size(); t++) {
          if (layout.re_correlated_multi[t]) {
            int chol_start = layout.chol_re_start_multi[t];
            int chol_size = layout.chol_re_end_multi[t] - chol_start;
            // Also include the corresponding sigma params
            // For now, just handle the Cholesky block if size <= 4
            if (chol_size >= 2 && chol_size <= 4) {
              block_specs.push_back({chol_start, chol_size});
            }
          }
        }
      }

      // NB phi params: overdispersion params are often correlated
      // A 2×2 block captures their joint curvature cheaply (4 extra multiplies/step)
      if (layout.has_phi_num && layout.has_phi_denom &&
          layout.log_phi_denom_idx == layout.log_phi_num_idx + 1) {
        block_specs.push_back({layout.log_phi_num_idx, 2});
      }

      // First check if model needs full DENSE (before block decision)
      // NB+ICAR: NegBin's digamma curvature creates strong correlations between
      // spatial phi params and overdispersion that BLOCK_DIAG's small blocks can't
      // capture. DENSE mass doubles the step size, cutting treedepth from 8-9 to 5-6,
      // which more than pays for the O(n²) per-step cost at p~108.
      // GP_t: NC z-sigma2-phi funnel creates erratic treedepth (2-10) with DIAG.
      // At p≤50, DENSE overhead is negligible.
      bool is_nb_family = (data.model_type == ModelType::NEGBIN_NEGBIN ||
                           data.model_type == ModelType::NEGBIN_GAMMA);
      bool is_icar = (data.spatial_type == SpatialType::ICAR);
      bool is_binomial_family = (data.model_type == ModelType::BINOMIAL ||
                                data.model_type == ModelType::BETA_BINOMIAL);
      bool hsgp_temporal = layout.is_hsgp && data.has_hsgp && layout.has_temporal;

      bool needs_full_dense = layout.has_latent ||  // N×K latent factors
                              hsgp_temporal ||  // HSGP+temporal cross-correlations
                              (is_nb_family && is_icar && n_params <= DENSE_MAX_PARAMS) ||  // NB+ICAR
                              (is_binomial_family && is_icar && n_params <= DENSE_MAX_PARAMS);  // Bin+ICAR

      // HSGP-only (no temporal): DIAG outperforms BLOCK_DIAG (29k LF/6 div vs 39k LF/15 div).
      // HSGP-only (no temporal): DIAG outperforms BLOCK_DIAG (29k LF/6 div vs
      // 39k LF/15 div). The real correlations are between lengthscale and m^2 basis
      // coefficients, which small blocks can't capture. Block adaptation adds noise.
      // HSGP+temporal uses full DENSE (handled above in needs_full_dense).
      bool prefer_diag = layout.is_hsgp && data.has_hsgp && !layout.has_temporal;

      if (needs_full_dense) {
        effective_metric = MassMatrixType::DENSE;
        block_specs.clear();
        auto_selected_diag = false;
      } else if (!block_specs.empty() && !prefer_diag) {
        // BLOCK_DIAG: captures key correlations without full O(n²)
        effective_metric = MassMatrixType::BLOCK_DIAG;
        auto_selected_diag = false;
      } else {
        // No blocks detected, no DENSE needed — fall back to DIAG
        effective_metric = MassMatrixType::DIAG;
        auto_selected_diag = true;
      }

      if (verbose) {
        REprintf("  [METRIC] auto -> %s (p=%d", metric_name(effective_metric), n_params);
        if (effective_metric == MassMatrixType::BLOCK_DIAG) {
          REprintf(", %d blocks:", (int)block_specs.size());
          for (const auto& bs : block_specs) {
            REprintf(" [%d,%d)", bs.first, bs.first + bs.second);
          }
        }
        if (layout.has_re) REprintf(", re");
        if (layout.has_re_correlated_slopes) REprintf(", correlated_slopes");
        if (layout.has_temporal) REprintf(", temporal");
        if (layout.is_bym2) REprintf(", bym2");
        if (layout.is_hsgp) REprintf(", hsgp");
        if (layout.is_gp) REprintf(", gp");
        if (layout.is_multiscale_gp) REprintf(", msgp");
        if (layout.is_temporal_gp) REprintf(", temporal_gp");
        if (layout.has_multiscale_temporal) REprintf(", ms_temporal");
        if (layout.has_svc) REprintf(", svc");
        if (layout.has_spatiotemporal) REprintf(", spatiotemporal");
        if (layout.has_latent) REprintf(", latent");
        if (layout.has_tvc) REprintf(", tvc");
        REprintf(")\n");
      }
    }
    // Also build block_specs when user explicitly requests BLOCK_DIAG
    if (effective_metric == MassMatrixType::BLOCK_DIAG && block_specs.empty()) {
      // User forced block_diag but AUTO didn't run — detect blocks from layout
      if (layout.is_temporal_gp && layout.log_sigma2_temporal_gp_idx >= 0 &&
          layout.logit_phi_temporal_gp_idx == layout.log_sigma2_temporal_gp_idx + 1) {
        block_specs.push_back({layout.log_sigma2_temporal_gp_idx, 2});
      }
      if (layout.is_hsgp && layout.log_sigma2_hsgp_idx >= 0 &&
          layout.log_lengthscale_hsgp_idx == layout.log_sigma2_hsgp_idx + 1) {
        block_specs.push_back({layout.log_sigma2_hsgp_idx, 2});
      }
      if (layout.is_bym2 && layout.log_sigma_bym2_idx >= 0 &&
          layout.logit_rho_bym2_idx == layout.log_sigma_bym2_idx + 1) {
        block_specs.push_back({layout.log_sigma_bym2_idx, 2});
      }
      if (layout.is_gp && layout.log_sigma2_gp_idx >= 0 &&
          layout.log_phi_gp_idx == layout.log_sigma2_gp_idx + 1) {
        block_specs.push_back({layout.log_sigma2_gp_idx, 2});
      }
      if (layout.is_multiscale_gp) {
        if (layout.log_sigma2_gp_local_idx >= 0 &&
            layout.log_phi_gp_local_idx == layout.log_sigma2_gp_local_idx + 1) {
          block_specs.push_back({layout.log_sigma2_gp_local_idx, 2});
        }
        if (layout.log_sigma2_gp_regional_idx >= 0 &&
            layout.log_phi_gp_regional_idx == layout.log_sigma2_gp_regional_idx + 1) {
          block_specs.push_back({layout.log_sigma2_gp_regional_idx, 2});
        }
      }
      if (layout.is_st_gp && layout.log_phi_st_space_idx >= 0 &&
          layout.log_phi_st_time_idx == layout.log_phi_st_space_idx + 1) {
        block_specs.push_back({layout.log_phi_st_space_idx, 2});
      }
      if (layout.has_phi_num && layout.has_phi_denom &&
          layout.log_phi_denom_idx == layout.log_phi_num_idx + 1) {
        block_specs.push_back({layout.log_phi_num_idx, 2});
      }
      // If still no blocks found, fall back to DIAG
      if (block_specs.empty()) {
        effective_metric = MassMatrixType::DIAG;
        if (verbose) {
          REprintf("  [BLOCK_DIAG] No correlated hyperparameter pairs found, falling back to DIAG\n");
        }
      }
    }

    // Auto-downgrade dense to diagonal when n_params too large
    // Dense needs O(p^2) storage and O(p^3) Cholesky; also needs n_warmup >= p samples
    if (effective_metric == MassMatrixType::DENSE && n_params > DENSE_MAX_PARAMS) {
      if (verbose) {
        REprintf("  [DENSE] n_params=%d > %d: auto-downgrading to diagonal\n",
                 n_params, DENSE_MAX_PARAMS);
      }
      effective_metric = MassMatrixType::DIAG;
    }


    try {
      if (effective_metric == MassMatrixType::BLOCK_DIAG && !block_specs.empty()) {
        mass.init_block_diag(n_params, block_specs);
      } else {
        mass.init(n_params, effective_metric);
      }
    } catch (const std::bad_alloc&) {
      if (effective_metric == MassMatrixType::DENSE) {
        if (verbose) {
          REprintf("  [DENSE] Allocation failed for p=%d, falling back to diagonal\n", n_params);
        }
        effective_metric = MassMatrixType::DIAG;
        mass.init(n_params, effective_metric);
      } else {
        throw;
      }
    }

    // Warm-start mass matrix from model structure.
    // Sets informed diagonal entries for parameter groups with known posterior scale,
    // giving the step size tuner a reasonable starting point even before warmup
    // samples are collected. This is critical for HSGP (m^2 basis coefficients),
    // BYM2 (spatial + IID), and correlated slopes (z ~ N(0,1)) models where the
    // identity mass causes excessively small epsilon → deep NUTS trees.
    {
      std::vector<double> inv_m(n_params, 1.0);
      std::vector<double> sqrt_m(n_params, 1.0);
      bool any_informed = false;

      // HSGP basis coefficients: beta_j ~ N(0, 1) → posterior variance ≈ 1
      // Hyperparameters: log_sigma2 ~ prior with moderate variance,
      //                  log_lengthscale ~ LogNormal(0,1) → variance ≈ 1
      if (layout.is_hsgp) {
        for (int j = layout.hsgp_beta_start; j < layout.hsgp_beta_end; j++) {
          inv_m[j] = 1.0;  // N(0,1) prior → unit scale
        }
        inv_m[layout.log_sigma2_hsgp_idx] = 1.0;
        inv_m[layout.log_lengthscale_hsgp_idx] = 1.0;
        any_informed = true;
      }

      // ICAR: phi[s] precision ≈ degree (number of neighbors)
      // Higher degree → smaller variance → tighter mass
      if (layout.has_spatial && !layout.is_bym2 &&
          data.spatial_type == SpatialType::ICAR && !data.adj_row_ptr.empty()) {
        for (int s = 0; s < (layout.spatial_end - layout.spatial_start); s++) {
          int degree = data.adj_row_ptr[s + 1] - data.adj_row_ptr[s];
          // ICAR precision diagonal ≈ degree; variance ≈ 1/degree
          double var_est = 1.0 / std::max(1.0, (double)degree);
          inv_m[layout.spatial_start + s] = var_est;
        }
        any_informed = true;
      }

      // BYM2: spatial phi ~ ICAR (eigenvalue-scaled), theta ~ N(0, I)
      // Riebler parameterization: phi[s] ≈ scale_factor variance
      if (layout.is_bym2) {
        double sf = std::max(data.bym2_scale_factor, 0.1);
        for (int s = layout.spatial_start; s < layout.spatial_end; s++) {
          inv_m[s] = sf * sf;  // ICAR variance ~ scale_factor^2
        }
        for (int s = layout.theta_bym2_start; s < layout.theta_bym2_end; s++) {
          inv_m[s] = 1.0;  // IID: N(0,1)
        }
        inv_m[layout.log_sigma_bym2_idx] = 1.0;
        inv_m[layout.logit_rho_bym2_idx] = 4.0;  // logit scale: wider
        any_informed = true;
      }

      // Correlated slopes: z ~ N(0, 1) (non-centered), Cholesky raw ~ tanh
      if (layout.has_re_correlated_slopes) {
        // RE slopes z values
        for (int j = layout.re_start; j < layout.re_end; j++) {
          inv_m[j] = 1.0;  // z ~ N(0,1)
        }
        any_informed = true;
      }

      // Temporal effects: RW1/RW2 have known precision structure
      if (layout.has_temporal) {
        // Temporal effects: moderate scale (tau-dependent, start at 1.0)
        for (int j = layout.temporal_start; j < layout.temporal_end; j++) {
          inv_m[j] = 1.0;
        }
        // AR1 rho: logit scale variance ≈ 4
        if (layout.logit_rho_ar1_idx >= 0) {
          inv_m[layout.logit_rho_ar1_idx] = 4.0;
        }
        any_informed = true;
      }

      // Non-centered RE: z ~ N(0, 1) → unit scale
      if (layout.has_re && data.re_parameterization == 1) {  // 1 = non-centered
        for (int j = layout.re_start; j < layout.re_end; j++) {
          inv_m[j] = 1.0;
        }
        any_informed = true;
      }

      // GP/SVC/TVC hyperparameters: moderate scale
      if (layout.is_gp) {
        if (layout.log_sigma2_gp_idx >= 0) inv_m[layout.log_sigma2_gp_idx] = 1.0;
        if (layout.log_phi_gp_idx >= 0) inv_m[layout.log_phi_gp_idx] = 1.0;
        any_informed = true;
      }

      // Temporal GP: NC z ~ N(0,1), logit_phi has wider posterior scale
      if (layout.is_temporal_gp) {
        for (int j = layout.temporal_start; j < layout.temporal_end; j++) {
          inv_m[j] = 1.0;  // z ~ N(0,1) for NC
        }
        if (layout.log_sigma2_temporal_gp_idx >= 0)
          inv_m[layout.log_sigma2_temporal_gp_idx] = 1.0;
        if (layout.logit_phi_temporal_gp_idx >= 0)
          inv_m[layout.logit_phi_temporal_gp_idx] = 4.0;  // logit scale: wider
        any_informed = true;
      }

      for (const auto& entry : coordinates.entries) {
        const double a2 = std::exp(2 * entry.log_scale(q_init));
        for (int i = entry.begin; i < entry.end; ++i) inv_m[i] /= a2;
        any_informed = true;
      }
      if (any_informed) {
        // Compute sqrt_mass from inv_mass
        for (int i = 0; i < n_params; i++) {
          inv_m[i] = std::max(1e-3, std::min(inv_m[i], 1e3));
          sqrt_m[i] = 1.0 / std::sqrt(inv_m[i]);
        }
        mass.set_diagonal(inv_m, sqrt_m);
        if (verbose) {
          REprintf("  [WARMSTART] Initialized mass matrix from model structure\n");
        }
      }
    }

    // Initialize sparse GMRF block for ST_IV spatiotemporal interaction.
    // Uses sparse Cholesky of posterior precision Q = tau*(Q_s⊗Q_t) + diag(H_lik).
    // At warmup end, extracts diag(Q^{-1}) to set precision-informed diagonal mass.
    // NOTE: Factorization happens later (after warmup discovers tau and H_lik).
    // Centered-only: the raw parameter this reads at warmup end (q[st_delta_start+k],
    // see the [SPARSE_GMRF] block below) IS delta under the centered parameterization.
    // Under non-centered (data.st_parameterization == 1) that slot holds z = delta *
    // sqrt(tau) instead, which this Q = tau*(Qs⊗Qt) + diag(H_lik) construction does not
    // describe -- z's natural scale is tau-free and near-unit, which the ordinary
    // Welford-adapted mass below already handles.
    bool use_sparse_gmrf_mass = (data.st_parameterization == 0);
    if (use_sparse_gmrf_mass && data.has_spatiotemporal && data.spatiotemporal_data.type == STType::TYPE_IV) {
      int st_S = data.spatiotemporal_data.n_spatial;
      int st_T = data.spatiotemporal_data.n_times;
      mass.sparse_gmrf.init(layout.st_delta_start, st_S, st_T);
      if (verbose) {
        REprintf("  [SPARSE_GMRF] ST_IV block initialized: %dx%d=%d params at offset %d\n",
                 st_S, st_T, st_S * st_T, layout.st_delta_start);
      }
    }

    // Recompute epsilon with warm-start mass (if informed)
    // This gives the dual averaging a better starting point when mass is pre-set
    if (mass.type != MassMatrixType::DIAG && !mass.inv_mass_diag.empty()) {
      epsilon = find_reasonable_epsilon(q, data, layout, rng, mass.inv_mass_diag);
    }

    mass_stats = WelfordStats(n_params);              // Always track diagonal
    cov_stats = WelfordCovStats(n_params);            // Only used when dense
    use_mass_matrix = false;

    // L-BFGS mass matrix adaptation (warmup-only)
    // Uses L-BFGS to learn curvature during warmup, then switches to standard HMC
    use_lbfgs = data.has_multiscale_gp &&
                     data.multiscale_gp_data.sampler == ratiod_gp::MSGPSampler::LBFGS;


    lbfgs_initialized = false;
    lbfgs_warmup_done = false;  // After warmup, use standard HMC
    if (use_lbfgs) {
      lbfgs_state = ratiod_gp::LBFGSState(10, n_params);
      q_prev.resize(n_params);
      grad_prev.resize(n_params);
    }

    // Stan-style expanding warmup windows for mass matrix adaptation
    // Phase 1: [0, init_buffer) - step size adaptation only
    // Phase 2: [init_buffer, n_warmup - term_buffer) - mass matrix adaptation
    //   Windows double in size: 25, 50, 100, 200, ...
    //   Last window extends to fill remaining space
    // Phase 3: [n_warmup - term_buffer, n_warmup) - final step size tuning
    // Models with structured warm-start (ICAR degree, BYM2 scale, HSGP) already
    // have reasonable mass, so we can start adaptation earlier (init_buffer=25).
    // This saves ~50 iterations of deep-tree warmup. Temporal_gp warm-start is
    // trivial (identity-like) so it still needs the full 75 iterations.
    bool has_structured_warmstart = layout.is_hsgp || layout.is_bym2 ||
      (layout.has_spatial && !layout.is_bym2 && data.spatial_type == SpatialType::ICAR && !data.adj_row_ptr.empty());
    init_buffer = has_structured_warmstart ? 25 : 75;
    term_buffer = 50;
    // For high-dimensional models (p>80), a 25-sample first mass window gives
    // very noisy variance estimates (25 samples / 108 params = 0.23 samples/param).
    // Skip the tiny first window by using a larger init_window (=50), so the first
    // mass update has ~50 samples. This trades one less mass update for better quality.
    init_window = (n_params > 80) ? 50 : 25;

    if (effective_metric == MassMatrixType::DENSE && n_params > 100) {
      term_buffer = 60;  // Reduced from 75 — saves 15 iterations for mass adaptation
    }

    // Note: For p~24, first mass window (25 samples < 29 needed) fails,
    // but this is fine — better to wait for more samples than set a poor
    // mass estimate early. The second window (100+ samples) gives good mass.

    // Adjust for short warmup
    if (n_warmup < init_buffer + term_buffer + init_window) {
      init_buffer = std::max(1, n_warmup / 5);
      term_buffer = std::max(1, n_warmup / 10);
      init_window = std::max(1, n_warmup - init_buffer - term_buffer);
    }

    // Compute mass adaptation window endpoints

    {
      int adapt_end = n_warmup - term_buffer;
      if (adapt_end <= init_buffer) {
        // No room for mass adaptation windows
        mass_window_ends.push_back(std::max(1, adapt_end));
      } else {
        int next_end = init_buffer + init_window;
        int win_size = init_window;
        while (next_end < adapt_end) {
          int next_win = 2 * win_size;
          if (next_end + next_win > adapt_end) {
            // Extend current window to fill remaining space
            mass_window_ends.push_back(adapt_end);
            break;
          }
          mass_window_ends.push_back(next_end);
          win_size = next_win;
          next_end += win_size;
        }
        if (mass_window_ends.empty() || mass_window_ends.back() < adapt_end) {
          mass_window_ends.push_back(adapt_end);
        }
      }
    }
    next_window_idx = 0;

    // Pre-allocate NUTS workspace (zero-allocation tree building)

                  // Momentum sampling buffer
         // Persistent proposal (survives tree resets)
      // Persistent proposal gradient
    if (use_nuts) {
      nuts_ws.init(n_params, max_treedepth);
      nuts_ws.gradient_fn = resolve_gradient_fn(g_gradient_mode, data, layout);
      _nuts_p.resize(n_params);
      _nuts_q_proposal.resize(n_params);
      _nuts_grad_proposal.resize(n_params);
    }

    sample_idx = 0;
    n_accept = 0;
    n_divergent = 0;
    // Adaptive NUTS→fixed-L switching: monitor early sampling for max treedepth
    nuts_probe_window = std::min(20, n_sample);  // Check first 20 sampling iterations
    nuts_probe_maxd = 0;  // Count of maxd hits in probe window
    nuts_probing = use_nuts && (L == 0);  // Only probe when using NUTS by default

    // WALNUTS in place of the NUTS trajectory (tulpa/walnuts.h): an exact kernel
    // whose macro step is subdivided where the local curvature needs it.
    // `epsilon` is the macro step; dual averaging tunes it on the mean
    // acceptance of each macro step's coarsest subdivision.
    use_walnuts = (walnuts != nullptr) && use_nuts;

    if (use_walnuts) {
      walnuts_ws.init(n_params, max_treedepth);
      result.sampler = "WALNUTS";
    }

    warmup_total_leapfrog = 0;
    // Note: warmup divergences are normal for DIAG models and resolve via dual
    // averaging — only final epsilon matters (checked at warmup end).

    if (verbose && layout.is_temporal_gp) {
      REprintf("  [MASS-WINDOWS] n_warmup=%d, windows=[", n_warmup);
      for (size_t w = 0; w < mass_window_ends.size(); w++) {
        REprintf("%d%s", mass_window_ends[w], w + 1 < mass_window_ends.size() ? "," : "");
      }
      REprintf("]\n");
    }
  }
  void adapt_window(int iter) {
    // Check if we've reached a mass adaptation window boundary
    if (next_window_idx < (int)mass_window_ends.size() &&
        iter == mass_window_ends[next_window_idx]) {
      bool dense_covariance_set = false;  // Track if DENSE covariance (not just diagonal) succeeded this window
      int dense_threshold = std::min(n_params + 5,
                                     std::max(50, n_params / 2));
      if (mass.type == MassMatrixType::DENSE && cov_stats.n >= dense_threshold) {
        auto cov = cov_stats.covariance();
        if (mass.update_from_covariance(cov.data(), cov_stats.n)) {
          use_mass_matrix = true;
          dense_covariance_set = true;
          if (verbose) {
            REprintf("  [DENSE] Window %d (iter %d): dense mass SET (n=%d, p=%d, OAS shrinkage=%.3f)\n",
                     next_window_idx, iter, cov_stats.n, n_params,
                     cov_stats.shrinkage_intensity);
          }
        } else {
          // Cholesky failed — mass auto-degraded to DIAG, use diagonal stats
          if (verbose) {
            REprintf("  [DENSE] Window %d (iter %d): Cholesky FAILED (cov_stats.n=%d, p=%d)\n",
                     next_window_idx, iter, cov_stats.n, n_params);
          }
          if (mass_stats.n >= 10) {
            mass.set_diagonal(mass_stats.inv_mass(), mass_stats.sqrt_mass());
            use_mass_matrix = true;
          }
        }
      } else if (mass.type == MassMatrixType::DENSE) {
        // Not enough samples for dense yet — use diagonal as interim
        if (verbose) {
          REprintf("  [DENSE] Window %d (iter %d): not enough samples (cov_stats.n=%d, need=%d)\n",
                   next_window_idx, iter, cov_stats.n, dense_threshold);
        }
        if (mass_stats.n >= 10) {
          mass.set_diagonal(mass_stats.inv_mass(), mass_stats.sqrt_mass());
          use_mass_matrix = true;
        }
      } else if (mass.type == MassMatrixType::BLOCK_DIAG) {
        // Block-diagonal: set diagonal for all params, then adapt block covariances
        if (mass_stats.n >= 10) {
          mass.set_diagonal(mass_stats.inv_mass(), mass_stats.sqrt_mass());
          use_mass_matrix = true;
        }
        int n_adapted = 0;
        for (auto& blk : mass.blocks) {
          if (blk.update_from_welford()) {
            n_adapted++;
          }
        }
        if (verbose && n_adapted > 0) {
          REprintf("  [BLOCK_DIAG] Window %d (iter %d): %d/%d blocks adapted (n=%d)\n",
                   next_window_idx, iter, n_adapted, (int)mass.blocks.size(), mass_stats.n);
        }
        // Reset block Welford accumulators for next window
        for (auto& blk : mass.blocks) {
          blk.reset_welford();
        }
      } else if (mass_stats.n >= 10) {
        // Diagonal path
        mass.set_diagonal(mass_stats.inv_mass(), mass_stats.sqrt_mass());
        use_mass_matrix = true;
      }

      // Temporal GP NC: z ~ N(0,1) by construction → optimal diag mass ≈ 1.0.
      // With limited warmup samples, noisy variance estimates for 20 z params
      // create unbalanced mass → small epsilon. Fix z entries to 1.0 so the
      // step size is driven by the hyperparameters (beta, sigma2, phi) only.
      if (verbose && layout.is_temporal_gp) {
        REprintf("  [Z-DEBUG] Window %d (iter %d): use_mass=%d, tgp=%d, nc=%d, ts=%d, te=%d, mass_n=%d\n",
                 next_window_idx, iter, (int)use_mass_matrix,
                 (int)layout.is_temporal_gp, (int)data.temporal_gp_data.noncentered,
                 layout.temporal_start, layout.temporal_end, mass_stats.n);
      }
      if (use_mass_matrix && layout.is_temporal_gp &&
          data.temporal_gp_data.noncentered &&
          layout.temporal_start >= 0 && layout.temporal_end > layout.temporal_start) {
        if (verbose) {
          REprintf("  [Z-FREEZE] Window %d: z mass before=[", next_window_idx);
          for (int j = layout.temporal_start; j < std::min(layout.temporal_end, layout.temporal_start + 5); j++) {
            REprintf("%.3f%s", mass.inv_mass_diag[j], j < layout.temporal_start + 4 ? "," : "");
          }
          REprintf("...], hyper=[");
          // Print beta and hyperparams
          for (int j = 0; j < std::min(4, layout.temporal_start); j++) {
            REprintf("%.3f%s", mass.inv_mass_diag[j], j < 3 ? "," : "");
          }
          REprintf("], sigma2=%.3f, phi=%.3f\n",
                   layout.log_sigma2_temporal_gp_idx >= 0 ? mass.inv_mass_diag[layout.log_sigma2_temporal_gp_idx] : -1.0,
                   layout.logit_phi_temporal_gp_idx >= 0 ? mass.inv_mass_diag[layout.logit_phi_temporal_gp_idx] : -1.0);
        }
        for (int j = layout.temporal_start; j < layout.temporal_end; j++) {
          mass.inv_mass_diag[j] = 1.0;
          mass.sqrt_mass_diag[j] = 1.0;
        }
      }

      mass_stats.reset();
      // For dense: only reset cov_stats when full covariance was successfully
      // computed THIS window. Otherwise keep accumulating across windows until
      // we have enough samples. This prevents the chicken-and-egg problem
      // where short windows never collect enough.
      // NOTE: We use dense_covariance_set (not mass.adapted) because
      // set_diagonal() also sets adapted=true, which would incorrectly
      // trigger a reset when we're still building up covariance samples.
      if (mass.type != MassMatrixType::DENSE || dense_covariance_set) {
        cov_stats.reset();
      }
      // Re-initialize step size with current mass matrix (A3)
      // Use dense-aware version when dense mass is adapted, so the step size
      // is calibrated for the rotated phase space (not just the diagonal).
      if (use_mass_matrix && mass.type == MassMatrixType::DENSE && mass.adapted) {
        epsilon = find_reasonable_epsilon(q, data, layout, rng, mass);
      } else if (use_mass_matrix) {
        epsilon = find_reasonable_epsilon(q, data, layout, rng, mass.inv_mass_diag);
      } else {
        epsilon = find_reasonable_epsilon(q, data, layout, rng);
      }
      da = DualAveraging(epsilon, n_params, target_boost);
      if (use_nuts) da.target_accept = nuts_target_accept;  // Preserve model-adaptive target

      next_window_idx++;
    }

    // L-BFGS: transition from L-BFGS to standard HMC at end of warmup
    // Extract diagonal mass matrix from learned curvature
    if (use_lbfgs && !lbfgs_warmup_done && iter == n_warmup - 1 && lbfgs_initialized) {
      // Use gamma from L-BFGS as uniform scaling for mass matrix
      // gamma = (s^T y) / (y^T y) approximates average inverse Hessian scaling
      double gamma = lbfgs_state.gamma;
      if (gamma > 0.01 && gamma < 100.0) {
        // Set inv_mass = gamma * I (larger gamma = larger variance = larger step in that direction)
        std::vector<double> inv_m(n_params, gamma);
        std::vector<double> sqrt_m(n_params, 1.0 / std::sqrt(gamma));
        mass.set_diagonal(inv_m, sqrt_m);
        use_mass_matrix = true;
      }
      lbfgs_warmup_done = true;
    }

    // =========================================================================
    // NUTS or fixed-trajectory HMC
    // =========================================================================
  }
  void adapt_nuts(int iter) {
    epsilon = da.update(alpha);

    if (iter >= n_warmup - term_buffer && iter < n_warmup - 1 &&
        mass.type == MassMatrixType::DENSE && mass.adapted && epsilon > 2.0) {
      if (verbose) {
        REprintf("  [DENSE] WARNING at iter %d: epsilon=%.4f (catastrophic). "
                 "Falling back to DIAG mass.\n", iter, epsilon);
      }
      mass.type = MassMatrixType::DIAG;
      // inv_mass_diag already populated from dense diagonal (update_from_covariance)
      epsilon = find_reasonable_epsilon(q, data, layout, rng, mass.inv_mass_diag);
      da = DualAveraging(epsilon, n_params, target_boost);
      if (use_nuts) da.target_accept = nuts_target_accept;
    }

    // DIAG→DENSE recovery is checked at warmup end (after da.final_epsilon)
    // rather than during warmup — warmup divergences are normal for DIAG models
    // and resolve via dual averaging. Only catastrophic final epsilon matters.

    if (iter >= init_buffer && iter < n_warmup - term_buffer) {
      mass_stats.update(q);
      if (mass.type == MassMatrixType::DENSE) {
        cov_stats.update(q);
      }
      if (mass.type == MassMatrixType::BLOCK_DIAG) {
        for (auto& blk : mass.blocks) {
          blk.welford_update(q.data());
        }
      }
    }
    if (iter == n_warmup - 1) {
      epsilon = da.final_epsilon();

      // DIAG→BLOCK_DIAG→DENSE recovery at warmup end: if AUTO selected DIAG but the
      // final adapted epsilon is catastrophic (>2.0), DIAG can't capture the
      // posterior geometry. Try BLOCK_DIAG first if block_specs available,
      // otherwise fall back to DENSE with identity mass.
      if (auto_selected_diag && mass.type == MassMatrixType::DIAG &&
          epsilon > 2.0) {
        if (!block_specs.empty()) {
          // Try BLOCK_DIAG recovery first (cheaper than full DENSE)
          if (verbose) {
            REprintf("  [DIAG->BLOCK_DIAG] Warmup end: final epsilon=%.4f (catastrophic). "
                     "Switching to BLOCK_DIAG (adapted=false).\n", epsilon);
          }
          mass.init_block_diag(n_params, block_specs);
          effective_metric = MassMatrixType::BLOCK_DIAG;
          auto_selected_diag = false;
          epsilon = find_reasonable_epsilon(q, data, layout, rng);
          da = DualAveraging(epsilon, n_params, target_boost);
          if (use_nuts) da.target_accept = nuts_target_accept;
          epsilon = da.final_epsilon();
        } else if (n_params <= DENSE_MAX_PARAMS) {
          if (verbose) {
            REprintf("  [DIAG->DENSE] Warmup end: final epsilon=%.4f (catastrophic). "
                     "Switching to DENSE identity mass.\n", epsilon);
          }
          mass.init(n_params, MassMatrixType::DENSE);
          effective_metric = MassMatrixType::DENSE;
          auto_selected_diag = false;
          epsilon = find_reasonable_epsilon(q, data, layout, rng);
          da = DualAveraging(epsilon, n_params, target_boost);
          if (use_nuts) da.target_accept = nuts_target_accept;
          epsilon = da.final_epsilon();
        }
      }

      // BLOCK_DIAG→DIAG fallback: if epsilon still catastrophic after BLOCK_DIAG
      if (epsilon > 2.0 && mass.type == MassMatrixType::BLOCK_DIAG) {
        if (verbose) {
          REprintf("  [BLOCK_DIAG->DIAG] WARNING: epsilon=%.4f still catastrophic. "
                   "Falling back to DIAG.\n", epsilon);
        }
        mass.init(n_params, MassMatrixType::DIAG);
        effective_metric = MassMatrixType::DIAG;
        epsilon = find_reasonable_epsilon(q, data, layout, rng, mass.inv_mass_diag);
        da = DualAveraging(epsilon, n_params, target_boost);
        if (use_nuts) da.target_accept = nuts_target_accept;
        epsilon = da.final_epsilon();
      }

      // Final safety net: if epsilon is still > 1.0 with dense mass after
      // the full terminal buffer, fall back to DIAG. This catches cases
      // where the catastrophe develops slowly.
      if (epsilon > 1.0 && mass.type == MassMatrixType::DENSE && mass.adapted) {
        if (verbose) {
          REprintf("  [DENSE] WARNING: epsilon=%.4f after warmup (catastrophic). "
                   "Falling back to DIAG mass.\n", epsilon);
        }
        mass.type = MassMatrixType::DIAG;
        epsilon = find_reasonable_epsilon(q, data, layout, rng, mass.inv_mass_diag);
        da = DualAveraging(epsilon, n_params, target_boost);
        if (use_nuts) da.target_accept = nuts_target_accept;
        epsilon = da.final_epsilon();
      }

      // Precision-informed diagonal mass for ST_IV at warmup end.
      // Build Q_post = tau*(Q_s⊗Q_t) + diag(h_lik), factorize, extract diag(Q^{-1}).
      if (mass.sparse_gmrf.active && !mass.sparse_gmrf.factorized) {
        int st_S = data.spatiotemporal_data.n_spatial;
        int st_T = data.spatiotemporal_data.n_times;
        int ST = st_S * st_T;
        double tau_st = std::exp(q[layout.log_tau_st_idx]);

        // Compute likelihood Hessian diagonal for each ST cell
        std::vector<double> h_lik(ST, 0.0);
        for (int i = 0; i < data.N; i++) {
          if (data.spatiotemporal_data.st_flat[i] <= 0) continue;
          int k = data.spatiotemporal_data.st_flat[i] - 1;
          double eta_i = 0.0;
          for (int p2 = 0; p2 < data.p_num; p2++)
            eta_i += data.X_num_flat[static_cast<size_t>(i) * data.p_num + p2] * q[layout.beta_num_start + p2];
          if (layout.has_re && data.re_group[i] > 0)
            eta_i += re_value_for_eta(&q[layout.re_start], data.re_group[i] - 1,
                                       std::exp(q[layout.log_sigma_re_idx]), data.re_parameterization);
          if (layout.has_spatial && data.spatial_group[i] > 0)
            eta_i += q[layout.spatial_start + data.spatial_group[i] - 1];
          if (layout.has_temporal && !data.temporal_time_idx.empty() &&
              i < (int)data.temporal_time_idx.size() && data.temporal_time_idx[i] > 0) {
            int t_idx = data.temporal_time_idx[i] - 1;
            int g_idx = data.temporal_group_idx[i] - 1;
            int t_base = g_idx * data.n_times + t_idx;
            if (t_base >= 0 && t_base < (int)q.size()) eta_i += q[layout.temporal_start + t_base];
          }
          eta_i += q[layout.st_delta_start + k];
          double mu_i = std::exp(eta_i);
          double h_i = 0.0;
          if (data.model_type == ModelType::POISSON_GAMMA) {
            h_i = mu_i;
          } else if (data.model_type == ModelType::BINOMIAL) {
            double p_i = 1.0 / (1.0 + std::exp(-eta_i));
            h_i = data.y_denom[i] * p_i * (1.0 - p_i);
          } else if (data.model_type == ModelType::NEGBIN_NEGBIN ||
                     data.model_type == ModelType::NEGBIN_GAMMA) {
            double phi = std::exp(q[layout.log_phi_num_idx]);
            h_i = mu_i / (1.0 + mu_i / phi);
          } else {
            h_i = mu_i;
          }
          if (data.spatiotemporal_data.shared) h_i *= 2.0;
          h_lik[k] += std::max(h_i, 1e-6);
        }

        mass.sparse_gmrf.build_and_factorize(
            data.spatiotemporal_data.adj_row_ptr,
            data.spatiotemporal_data.adj_col_idx,
            data.spatiotemporal_data.temporal_type,
            data.spatiotemporal_data.temporal_cyclic,
            tau_st, h_lik.data(), 0.001
        );

        if (mass.sparse_gmrf.factorized) {
          // Extract diag(Q^{-1}) and set diagonal mass for ST params
          int n_set = 0;
          double sum_var = 0.0;
          for (int k = 0; k < ST; k++) {
            Eigen::VectorXd ek = Eigen::VectorXd::Zero(ST);
            ek[k] = 1.0;
            Eigen::VectorXd col_k = mass.sparse_gmrf.llt.solve(ek);
            double var_k = col_k[k];
            if (var_k > 1e-10 && var_k < 100.0) {
              mass.inv_mass_diag[layout.st_delta_start + k] = var_k;
              n_set++;
              sum_var += var_k;
            }
          }
          // Deactivate sparse GMRF — diagonal mass is now informed
          mass.sparse_gmrf.active = false;
          // Recompute epsilon with new mass
          epsilon = find_reasonable_epsilon(q, data, layout, rng, mass.inv_mass_diag);
          da = DualAveraging(epsilon, n_params, target_boost);
          if (use_nuts) da.target_accept = nuts_target_accept;
          epsilon = da.final_epsilon();
          if (verbose) {
            REprintf("  [SPARSE_GMRF] Diagonal mass set for %d/%d ST params, avg_var=%.6f, tau=%.4f, new epsilon=%.6f\n",
                     n_set, ST, n_set > 0 ? sum_var / n_set : 0.0, tau_st, epsilon);
          }
        } else if (verbose) {
          REprintf("  [SPARSE_GMRF] WARNING: Cholesky failed, keeping adapted diagonal mass\n");
        }
      }

      if (verbose) {
        REprintf("  [METRIC] Warmup done: epsilon=%.6f, mass.type=%s, mass.adapted=%d\n",
                 epsilon, metric_name(mass.type), (int)mass.adapted);
      }
    }
    // Print tree depth for last 10 warmup iterations
    if (verbose && iter >= n_warmup - 10) {
      REprintf("  [%s] warmup iter %d: treedepth=%d, epsilon=%.6f\n",
               metric_name(mass.type),
               iter, iter_treedepth, epsilon);
    }
  }
  void adapt_hmc(int iter) {
    epsilon = da.update(alpha);
    // Only collect mass stats during mass adaptation phase (A5)
    if (iter >= init_buffer && iter < n_warmup - term_buffer) {
      mass_stats.update(q);
      if (mass.type == MassMatrixType::DENSE) {
        cov_stats.update(q);
      }
    }
    // On last warmup iteration, use averaged step size for sampling (A1)
    if (iter == n_warmup - 1) {
      epsilon = da.final_epsilon();
    }
  }
  void transition(int iter, bool is_warmup) {
    if (is_warmup) adapt_window(iter);
    alpha = 0.0;
    divergent = false;
    iter_n_leapfrog = L;
    iter_treedepth = 0;
    iter_energy = NA_REAL;

    if (use_nuts && !(use_lbfgs && !lbfgs_warmup_done)) {
      if (use_walnuts) {
        RatioWalnutsModel walnuts_model{mass, nuts_ws.gradient_fn, data,
                                        layout, n_params};
        tulpa::WalnutsTransitionResult w = tulpa::walnuts_transition(
          q, current_grad, log_prob_current, epsilon, max_treedepth,
          *walnuts, walnuts_model, walnuts_ws, rng);
        alpha = w.mean_accept;
        divergent = w.divergent;
        iter_n_leapfrog = w.n_grad;
        iter_treedepth = w.depth;
        iter_energy = w.H0;
      } else {
      // -----------------------------------------------------------------
      // NUTS: No-U-Turn Sampler (optimized zero-allocation path)
      // -----------------------------------------------------------------

      auto& p = _nuts_p;

      // Step size jitter: ±20% random noise per trajectory
      // Prevents systematic step-size resonances that cause divergences.
      // Only during post-warmup sampling — warmup needs stable epsilon for adaptation.
      double eps_iter = epsilon;
      if (!is_warmup) {
        double jitter = 1.0 + 0.2 * (2.0 * unif(rng) - 1.0);  // U[0.8, 1.2]
        eps_iter = epsilon * jitter;
      }

      // Sample momentum p ~ N(0, M) where M = C^{-1}
      mass.sample_momentum(p.data(), rng);

      // Initial Hamiltonian (pointer-based, no vector overhead)
      double H0 = nuts_compute_hamiltonian_fast(
        log_prob_current, p.data(), mass, n_params
      );
      iter_energy = H0;
      double delta_max = 1000.0;

      // Load current state into workspace persistent slots
      nuts_ws.load_node(NUTSWorkspace::NODE_LEFT_SLOT,
                        q.data(), p.data(), current_grad.data(), log_prob_current);
      nuts_ws.load_node(NUTSWorkspace::NODE_RIGHT_SLOT,
                        q.data(), p.data(), current_grad.data(), log_prob_current);

      // Initialize persistent proposal buffers (pre-allocated, no per-iter malloc)
      auto& q_proposal_data = _nuts_q_proposal;
      auto& grad_proposal_data = _nuts_grad_proposal;
      std::memcpy(q_proposal_data.data(), q.data(), n_params * sizeof(double));
      std::memcpy(grad_proposal_data.data(), current_grad.data(), n_params * sizeof(double));
      double log_prob_proposal = log_prob_current;
      double sum_log_weight = 0.0;  // Relative weights: log(exp(H0 - H0)) = 0

      int total_leapfrog = 0;
      double sum_accept_prob = 0.0;
      divergent = false;

      // Generalized U-turn tracking at top level (Stan-style)
      // rho = total momentum sum. rho_bck/rho_fwd = halves for 3-juncture checks.
      // At each iteration the entire old trajectory becomes one half,
      // the new subtree becomes the other half (Stan's approach).
      // Uses pre-allocated workspace vectors (no per-iteration heap allocation).
      auto& rho = nuts_ws.iter_rho;
      std::memcpy(rho.data(), p.data(), n_params * sizeof(double));
      auto& rho_bck = nuts_ws.iter_rho_bck;
      auto& rho_fwd = nuts_ws.iter_rho_fwd;
      std::fill(rho_bck.begin(), rho_bck.end(), 0.0);
      std::fill(rho_fwd.begin(), rho_fwd.end(), 0.0);

      // p_sharp = M^{-1} * p at initial point — full mass for correct U-turn geometry
      auto& p_sharp_init = nuts_ws.iter_p_sharp_init;
      mass.inv_mass_times_p(p.data(), p_sharp_init.data());

      // Boundary momenta: _end = far endpoint, _beg = origin-facing boundary
      // Stan naming: bck_end=bck_bck, bck_beg=bck_fwd, fwd_beg=fwd_bck, fwd_end=fwd_fwd
      auto& p_fwd_beg = nuts_ws.iter_p_fwd_beg;
      auto& p_fwd_end = nuts_ws.iter_p_fwd_end;
      auto& p_bck_beg = nuts_ws.iter_p_bck_beg;
      auto& p_bck_end = nuts_ws.iter_p_bck_end;
      std::memcpy(p_fwd_beg.data(), p.data(), n_params * sizeof(double));
      std::memcpy(p_fwd_end.data(), p.data(), n_params * sizeof(double));
      std::memcpy(p_bck_beg.data(), p.data(), n_params * sizeof(double));
      std::memcpy(p_bck_end.data(), p.data(), n_params * sizeof(double));
      auto& p_sharp_fwd_beg = nuts_ws.iter_p_sharp_fwd_beg;
      auto& p_sharp_fwd_end = nuts_ws.iter_p_sharp_fwd_end;
      auto& p_sharp_bck_beg = nuts_ws.iter_p_sharp_bck_beg;
      auto& p_sharp_bck_end = nuts_ws.iter_p_sharp_bck_end;
      std::memcpy(p_sharp_fwd_beg.data(), p_sharp_init.data(), n_params * sizeof(double));
      std::memcpy(p_sharp_fwd_end.data(), p_sharp_init.data(), n_params * sizeof(double));
      std::memcpy(p_sharp_bck_beg.data(), p_sharp_init.data(), n_params * sizeof(double));
      std::memcpy(p_sharp_bck_end.data(), p_sharp_init.data(), n_params * sizeof(double));

      // Build tree until U-turn or max depth
      for (int j = 0; j < max_treedepth; j++) {
        std::uniform_int_distribution<int> dir_dist(0, 1);
        int direction = 2 * dir_dist(rng) - 1;

        nuts_ws.reset_tree();

        int start_slot = nuts_ws.alloc_slot();
        if (start_slot < 0) break;
        if (direction == 1) {
          nuts_ws.copy_node(start_slot, NUTSWorkspace::NODE_RIGHT_SLOT);
        } else {
          nuts_ws.copy_node(start_slot, NUTSWorkspace::NODE_LEFT_SLOT);
        }

        // Stan: relabel halves before building subtree
        // Entire old trajectory becomes one half; new subtree is the other
        if (direction == 1) {
          // Extending forward: old trajectory → backward half
          std::memcpy(rho_bck.data(), rho.data(), n_params * sizeof(double));
          std::memcpy(p_bck_beg.data(), p_fwd_end.data(), n_params * sizeof(double));
          std::memcpy(p_sharp_bck_beg.data(), p_sharp_fwd_end.data(), n_params * sizeof(double));
        } else {
          // Extending backward: old trajectory → forward half
          std::memcpy(rho_fwd.data(), rho.data(), n_params * sizeof(double));
          std::memcpy(p_fwd_beg.data(), p_bck_end.data(), n_params * sizeof(double));
          std::memcpy(p_sharp_fwd_beg.data(), p_sharp_bck_end.data(), n_params * sizeof(double));
        }

        TreeStats subtree = build_tree_fast(
          nuts_ws, start_slot, direction, j,
          eps_iter, mass, H0, delta_max,
          data, layout, rng
        );

        total_leapfrog += subtree.n_leapfrog;
        sum_accept_prob += subtree.sum_accept_prob;

        if (subtree.divergent) {
          divergent = true;
        }

        if (!subtree.stop) {
          // Multinomial acceptance
          double log_sum_weight_subtree = subtree.sum_log_weight;
          double new_sum_log_weight = ratiod_linalg::log_sum_exp(sum_log_weight, log_sum_weight_subtree);

          double accept_prob_subtree;
          if (log_sum_weight_subtree > new_sum_log_weight) {
            accept_prob_subtree = 1.0;
          } else {
            accept_prob_subtree = std::exp(log_sum_weight_subtree - new_sum_log_weight);
          }
          if (!std::isfinite(accept_prob_subtree)) accept_prob_subtree = 0.0;

          std::uniform_real_distribution<double> unif01(0.0, 1.0);
          if (unif01(rng) < accept_prob_subtree) {
            std::memcpy(q_proposal_data.data(), nuts_ws.q_at(subtree.proposal_slot),
                        n_params * sizeof(double));
            std::memcpy(grad_proposal_data.data(), nuts_ws.grad_at(subtree.proposal_slot),
                        n_params * sizeof(double));
            log_prob_proposal = subtree.log_prob_proposal;
          }

          sum_log_weight = new_sum_log_weight;
        }

        // Update direction endpoints and rho half from subtree
        // Use memcpy instead of std::move to preserve pre-allocated buffers
        if (direction == 1) {
          nuts_ws.copy_node(NUTSWorkspace::NODE_RIGHT_SLOT, subtree.right_slot);
          std::memcpy(rho_fwd.data(), subtree.rho.data(), n_params * sizeof(double));
          std::memcpy(p_fwd_beg.data(), subtree.p_beg.data(), n_params * sizeof(double));
          std::memcpy(p_fwd_end.data(), subtree.p_end.data(), n_params * sizeof(double));
          std::memcpy(p_sharp_fwd_beg.data(), subtree.p_sharp_beg.data(), n_params * sizeof(double));
          std::memcpy(p_sharp_fwd_end.data(), subtree.p_sharp_end.data(), n_params * sizeof(double));
        } else {
          nuts_ws.copy_node(NUTSWorkspace::NODE_LEFT_SLOT, subtree.left_slot);
          std::memcpy(rho_bck.data(), subtree.rho.data(), n_params * sizeof(double));
          std::memcpy(p_bck_beg.data(), subtree.p_beg.data(), n_params * sizeof(double));
          std::memcpy(p_bck_end.data(), subtree.p_end.data(), n_params * sizeof(double));
          std::memcpy(p_sharp_bck_beg.data(), subtree.p_sharp_beg.data(), n_params * sizeof(double));
          std::memcpy(p_sharp_bck_end.data(), subtree.p_sharp_end.data(), n_params * sizeof(double));
        }

        // Combine rho = rho_bck + rho_fwd
        for (int i = 0; i < n_params; i++) {
          rho[i] = rho_bck[i] + rho_fwd[i];
        }

        iter_treedepth = j + 1;

        // Generalized U-turn check at top level (3 junctures)
        if (subtree.stop) break;

        // Check 1: Full trajectory — far endpoints vs total rho
        bool persist = compute_criterion(p_sharp_bck_end.data(), p_sharp_fwd_end.data(),
                                         rho.data(), n_params);

        // Check 2: Backward half + seam from forward (rho = rho_bck + p_fwd_beg)
        auto& rho_seam = nuts_ws.iter_rho_seam;
        for (int i = 0; i < n_params; i++) {
          rho_seam[i] = rho_bck[i] + p_fwd_beg[i];
        }
        persist &= compute_criterion(p_sharp_bck_end.data(), p_sharp_fwd_beg.data(),
                                      rho_seam.data(), n_params);

        // Check 3: Seam from backward + forward half (rho = rho_fwd + p_bck_beg)
        for (int i = 0; i < n_params; i++) {
          rho_seam[i] = rho_fwd[i] + p_bck_beg[i];
        }
        persist &= compute_criterion(p_sharp_bck_beg.data(), p_sharp_fwd_end.data(),
                                      rho_seam.data(), n_params);

        if (!persist) break;
      }

      // Accept proposal: copy from persistent proposal buffers (memcpy, no alloc)
      std::memcpy(q.data(), q_proposal_data.data(), n_params * sizeof(double));
      std::memcpy(current_grad.data(), grad_proposal_data.data(), n_params * sizeof(double));
      log_prob_current = log_prob_proposal;

      // Average acceptance statistic for dual averaging
      alpha = (total_leapfrog > 0) ? (sum_accept_prob / total_leapfrog) : 0.0;
      iter_n_leapfrog = total_leapfrog;
      }
      n_accept++;

      if (divergent) n_divergent++;
      if (iter_treedepth >= max_treedepth) result.n_max_treedepth++;

      // Adaptation during warmup
      if (is_warmup) adapt_nuts(iter);

      // Adaptive NUTS probe: warn if most early iterations hit max treedepth
      // (Stan's approach: warn but keep NUTS running — truncated NUTS picks
      // from up to 2^depth candidates, far better than HMC(L=10) with tiny epsilon)
      if (nuts_probing && !is_warmup && sample_idx < nuts_probe_window) {
        if (iter_treedepth >= max_treedepth) nuts_probe_maxd++;
        if (sample_idx == nuts_probe_window - 1) {
          nuts_probing = false;  // Probe window complete
          if (nuts_probe_maxd >= (nuts_probe_window * 8 + 9) / 10) {
            result.n_max_treedepth += 0;  // Already counted above
            if (verbose) {
              REprintf("  [NUTS] %d/%d initial sampling iterations hit max treedepth (%d). "
                       "Consider increasing max_treedepth or reparameterizing.\n",
                       nuts_probe_maxd, nuts_probe_window, max_treedepth);
            }
          }
        }
      }
    } else {

      // Sample momentum and compute kinetic energy
      std::vector<double> p(n_params);
      double kinetic_current = 0.0;
      double H_current;

      if (use_lbfgs && lbfgs_initialized && !lbfgs_warmup_done && lbfgs_state.d == n_params) {
        // L-BFGS: Sample p ~ N(0, B) where B ≈ 1/gamma * I (warmup only)
        std::vector<double> sqrt_diag = lbfgs_state.get_sqrt_B_diag();
        if ((int)sqrt_diag.size() == n_params) {
          for (int i = 0; i < n_params; i++) {
            p[i] = normal(rng) * sqrt_diag[i];
          }
          kinetic_current = lbfgs_state.kinetic_energy(p);
          H_current = -log_prob_current + kinetic_current;
        } else {
          mass.sample_momentum(p.data(), rng);
          kinetic_current = mass.kinetic_energy(p.data());
          H_current = -log_prob_current + kinetic_current;
        }
      } else {
        mass.sample_momentum(p.data(), rng);
        kinetic_current = mass.kinetic_energy(p.data());
        H_current = -log_prob_current + kinetic_current;
      }

      // Leapfrog integration
      std::vector<double> q_prop = q;
      std::vector<double> p_prop = p;

      // Determine effective L for this iteration
      int L_eff = L;
      if (use_nuts && use_lbfgs && !lbfgs_warmup_done) {
        // During L-BFGS warmup with NUTS mode, use fixed L=20
        L_eff = 20;
      }

      if (use_lbfgs && lbfgs_initialized && !lbfgs_warmup_done && lbfgs_state.d == n_params) {
        // L-BFGS leapfrog
        std::vector<double> grad(n_params);
        compute_gradient(q_prop, data, layout, grad);

        for (int l = 0; l < L_eff; l++) {
          for (int i = 0; i < n_params; i++) {
            p_prop[i] += 0.5 * epsilon * grad[i];
          }
          std::vector<double> Hp(n_params);
          lbfgs_state.multiply_H(p_prop, Hp);
          for (int i = 0; i < n_params; i++) {
            q_prop[i] += epsilon * Hp[i];
            if (!std::isfinite(q_prop[i])) {
              divergent = true;
              break;
            }
          }
          if (divergent) break;
          compute_gradient(q_prop, data, layout, grad);
          for (int i = 0; i < n_params; i++) {
            p_prop[i] += 0.5 * epsilon * grad[i];
          }
          for (int i = 0; i < n_params; i++) {
            if (!std::isfinite(p_prop[i]) || std::abs(p_prop[i]) > 1e10) {
              divergent = true;
              break;
            }
          }
          if (divergent) break;
        }
      } else {
        // Standard leapfrog
        for (int l = 0; l < L_eff; l++) {
          LeapfrogResult lf;
          if (use_mass_matrix) {
            lf = leapfrog_step_mass(q_prop, p_prop, epsilon, mass.inv_mass_diag, data, layout);
          } else {
            lf = leapfrog_step(q_prop, p_prop, epsilon, data, layout);
          }
          q_prop = lf.q;
          p_prop = lf.p;
          if (lf.divergent) {
            divergent = true;
            break;
          }
        }
      }

      // Compute proposed Hamiltonian (use same log-post as gradient mode)
      const double log_prob_prop = compute_log_post(q_prop, data, layout);
      double kinetic_prop = 0.0;

      if (use_lbfgs && lbfgs_initialized && !lbfgs_warmup_done && lbfgs_state.d == n_params) {
        kinetic_prop = lbfgs_state.kinetic_energy(p_prop);
      } else {
        kinetic_prop = mass.kinetic_energy(p_prop.data());
      }
      double H_prop = -log_prob_prop + kinetic_prop;
      iter_energy = H_current;

      // Metropolis accept/reject
      alpha = metropolis_accept_prob(H_prop - H_current);

      std::uniform_real_distribution<double> unif01(0.0, 1.0);
      bool accepted = (unif01(rng) < alpha) && !divergent;
      if (accepted) {
        q = q_prop;
        log_prob_current = log_prob_prop;
        n_accept++;
        // Update cached gradient for transition to NUTS after L-BFGS warmup
        if (use_nuts) {
          compute_gradient(q, data, layout, current_grad);
        }
      }
      if (divergent) n_divergent++;

      // Adaptation during warmup
      if (is_warmup) adapt_hmc(iter);

      // L-BFGS update: collect (s, y) pairs from accepted samples (warmup only)
      if (use_lbfgs && !lbfgs_warmup_done) {
        std::vector<double> grad_current(n_params);
        compute_gradient(q, data, layout, grad_current);

        if (!lbfgs_initialized) {
          q_prev = q;
          grad_prev = grad_current;
          lbfgs_initialized = true;
        } else if (accepted) {
          std::vector<double> s(n_params), y(n_params);
          for (int i = 0; i < n_params; i++) {
            s[i] = q[i] - q_prev[i];
            y[i] = grad_current[i] - grad_prev[i];
          }
          lbfgs_state.add_pair(s, y);
          q_prev = q;
          grad_prev = grad_current;
        }
      }

      iter_n_leapfrog = L_eff;
    }  // end fixed-trajectory HMC
  }
  void store_sample() {
    std::vector<double> canonical;
    const double stored_jacobian = coordinates.entries.empty() ? 0.0 : coordinates.forward(q, canonical);
    const auto& stored = coordinates.entries.empty() ? q : canonical;
    std::memcpy(result.sample_row(sample_idx), stored.data(),
                n_params * sizeof(double));
    // A non-centred block is sampled in a coordinate that is not the effect
    // it stands for, so the stored row carries the effect. q keeps the
    // sampled coordinate.
    {
        double* row = result.sample_row(sample_idx);
        if (data.gp_parameterization == 1 && data.has_gp && layout.is_gp) {
            double sigma2_store = std::exp(q[layout.log_sigma2_gp_idx]);
            double phi_store = std::exp(q[layout.log_phi_gp_idx]);
            RATIOD_TLS_WORKSPACE(ratiod_gp::NNGPNCWorkspace, nc_ws_store);
            ratiod_gp::nngp_nc_forward(&stored[layout.gp_w_start], sigma2_store, phi_store,
                                       data.gp_data, nc_ws_store);
            int N_gp = data.gp_data.n_obs;
            for (int i = 0; i < N_gp; i++) {
                row[layout.gp_w_start + i] = nc_ws_store.w[i];
            }
        }
        // The SVC row carries the field eta reads: each term on the effect
        // scale whichever coordinate it was sampled in, centred over
        // locations as svc_center_eta enters it. Each range slot carries
        // log(phi), the unconstrained value the row has always held there.
        if (layout.has_svc && data.has_svc && !data.svc_is_hsgp &&
            data.svc_data.n_svc > 0) {
            const int n_svc_store = data.svc_data.n_svc;
            std::vector<double> svc_s2_store(n_svc_store), svc_phi_store(n_svc_store);
            for (int j = 0; j < n_svc_store; j++) {
                svc_s2_store[j] = std::exp(q[layout.log_sigma2_svc_start + j]);
                svc_phi_store[j] = ratiod_svc_ad::svc_range(
                    q[layout.range_svc_start + j], data.svc_phi_prior_lower,
                    data.svc_phi_prior_upper);
                row[layout.range_svc_start + j] = std::log(svc_phi_store[j]);
            }
            ratiod_svc::svc_field<ratiod_svc::SVC_DENSITY_SLOT>(
                &stored[layout.svc_w_start], svc_s2_store.data(), svc_phi_store.data(),
                data.svc_data, data.svc_gp_view, data.svc_noncentered,
                &row[layout.svc_w_start]);
            ratiod_svc::svc_center_terms(&row[layout.svc_w_start], data.svc_data);
        }
        if (layout.has_spatiotemporal &&
            ratiod_spatiotemporal::st_noncentered_gp(data.st_parameterization,
                                                     data.st_is_hsgp,
                                                     data.spatiotemporal_data)) {
            const double sigma2_st_store = std::exp(-q[layout.log_tau_st_idx]);
            const double phi_space_store = st_gp_range_value(q[layout.log_phi_st_space_idx], data, true);
            const double phi_time_store = st_gp_range_value(q[layout.log_phi_st_time_idx], data, false);
            (void)ratiod_spatiotemporal::st_gp_nc_forward(
                &q[layout.st_delta_start], sigma2_st_store, phi_space_store,
                phi_time_store, data.spatiotemporal_data,
                &row[layout.st_delta_start]);
        }
        if (layout.is_st_gp && !layout.st_collapsed) {
            row[layout.log_phi_st_space_idx] = std::log(st_gp_range_value(q[layout.log_phi_st_space_idx], data, true));
            row[layout.log_phi_st_time_idx] = std::log(st_gp_range_value(q[layout.log_phi_st_time_idx], data, false));
        }
        if (temporal_effects_transformed(data, layout)) {
            std::vector<double> phi_store_buf;
            const double* phi_store = temporal_effects(stored, data, layout, phi_store_buf);
            const int T_store = layout.temporal_end - layout.temporal_start;
            for (int i = 0; i < T_store; i++) {
                row[layout.temporal_start + i] = phi_store[i];
            }
        }
        if (layout.has_tvc)
            tulpa::s2z_centre_blocks(row + layout.tvc_w_start,
                data.tvc_data.n_groups * data.tvc_data.n_tvc, data.tvc_data.n_times);
        if (layout.has_spatiotemporal && !data.spatiotemporal_data.spatial_partition.trivial()) {
            const auto& st = data.spatiotemporal_data;
            std::vector<double> field(row + layout.st_delta_start, row + layout.st_delta_end);
            tulpa::center_st_margins(field, st.n_spatial, st.n_times, st.spatial_partition);
            std::copy(field.begin(), field.end(), row + layout.st_delta_start);
        }
        if (layout.has_temporal)
            data.temporal_restricted_slopes.remove(row + layout.temporal_start);
        // The intrinsic multiscale arms enter eta centred, and the R side
        // adds the stored field straight back into eta, so the stored draw
        // is the centred one: an uncentred draw would report -- and predict
        // from -- a different series than the one the likelihood saw.
        if (layout.has_multiscale_temporal) {
            const auto& mst_store = data.multiscale_temporal_data;
            const int n_trend_store = layout.trend_end - layout.trend_start;
            const int n_seasonal_store = layout.seasonal_end - layout.seasonal_start;
            const bool trend_intrinsic_store =
                n_trend_store > 0 &&
                (mst_store.trend_type == TemporalType::RW1 ||
                 mst_store.trend_type == TemporalType::RW2);
            const bool seasonal_store =
                n_seasonal_store > 0 && mst_store.seasonal_period > 0;
            // In the non-centred coordinate the row still holds z, so the
            // effects have to be built before they can be centred.
            if (mst_store.noncentered) {
                const double s2_trend_store = (n_trend_store > 0)
                    ? std::exp(q[layout.log_sigma2_trend_idx]) : 1.0;
                const double s2_seasonal_store = (n_seasonal_store > 0)
                    ? std::exp(q[layout.log_sigma2_seasonal_idx]) : 1.0;
                const auto arms_store = ratiod_temporal::ms_read_arms(
                    trend_intrinsic_store ? &q[layout.trend_start] : nullptr,
                    n_trend_store,
                    seasonal_store ? &q[layout.seasonal_start] : nullptr,
                    n_seasonal_store,
                    s2_trend_store, s2_seasonal_store, mst_store);
                if (trend_intrinsic_store) {
                    for (int i = 0; i < n_trend_store; i++)
                        row[layout.trend_start + i] = arms_store.trend[i];
                }
                if (seasonal_store) {
                    for (int i = 0; i < n_seasonal_store; i++)
                        row[layout.seasonal_start + i] = arms_store.seasonal[i];
                }
            }
            if (trend_intrinsic_store) {
                (void)tulpa::s2z_centre_component(row, layout.trend_start,
                                                  n_trend_store);
                data.ms_trend_restricted_slopes.remove(row + layout.trend_start);
            }
            if (seasonal_store) {
                (void)tulpa::s2z_centre_component(row, layout.seasonal_start,
                                                  n_seasonal_store);
            }
        }
    }
    result.log_prob[sample_idx] = log_prob_current - stored_jacobian;
    result.energy[sample_idx] = iter_energy;
    result.accept_prob[sample_idx] = alpha;
    result.n_leapfrog[sample_idx] = iter_n_leapfrog;
    result.divergent[sample_idx] = divergent ? 1 : 0;
    result.treedepth[sample_idx] = iter_treedepth;

    // Collapsed: store mode values from the last gradient evaluation
    if (data.gp_collapsed && data.has_gp && result.n_gp_collapsed > 0) {
        collapsed_gp_store_sample(sample_idx, collapsed_gp_ws(),
            result.gp_w_star_flat, result.n_gp_collapsed);
    }
    // The collapsed interaction at the saved state: the Laplace
    // approximation N(dh, H^-1) re-evaluated there, and one draw from it.
    // Its own stream, so storing a draw does not move the chain.
    if (result.n_st_collapsed > 0) {
        (void)compute_log_post(q, data, layout);
        ratiod_st_laplace::draw(
            data, ratiod_st_laplace::workspace(), st_draw_rng,
            &result.st_delta_flat[static_cast<size_t>(sample_idx) *
                                  result.n_st_collapsed]);
    }
    if ((data.icar_collapsed || data.bym2_collapsed) && result.n_icar_collapsed > 0) {
        collapsed_icar_store_sample(sample_idx, data, collapsed_icar_ws(),
            result.icar_phi_star_flat, result.bym2_theta_star_flat,
            result.n_icar_collapsed);
    }

    sample_idx++;
  }
  void warmup() {
    for (int iter = 0; iter < n_warmup; ++iter) {
      transition(iter, true);
      warmup_total_leapfrog += iter_n_leapfrog;
    }
  }
  void sample() {
    for (int iter = n_warmup; iter < n_iter; ++iter) {
      transition(iter, false);
      store_sample();
    }
  }
};
}  // namespace

HMCResultCpp run_hmc_chain_cpp(
    const std::vector<double>& q_init,
    const ModelData& data,
    const ParamLayout& layout,
    int n_iter,
    int n_warmup,
    int L,
    int chain_id,
    unsigned int seed,
    bool verbose,
    int max_treedepth,
    MassMatrixType metric_type,
    double adapt_delta,
    const tulpa::WalnutsConfig* walnuts
) {
  return HmcChain(q_init, data, layout, n_iter, n_warmup, L, chain_id, seed,
                  verbose, max_treedepth, metric_type, adapt_delta, walnuts).run();
}

}  // namespace ratiod_hmc
