// hmc_gradient_vectorized.h
// Vectorized gradient computation with template specialization per model type.
// Replaces scalar per-observation loop in compute_gradient_analytical() for
// non-ZI, non-slopes model configurations.
//
// Architecture (three-pass):
//   Pass 1: Vectorized eta = X * beta  (Eigen matvec)
//           + dense-expanded RE/spatial/temporal additions
//   Pass 2: Scalar model-specific residual kernel (templated per ModelType)
//   Pass 3: Vectorized grad_beta = X^T * resid  (Eigen matvec)
//
// Template dispatch eliminates dead branches at compile time, reducing
// instruction cache pressure from ~30-50KB to ~5KB per instantiation.
//
// IMPORTANT: This header must be included from hmc_sampler.cpp AFTER
// hmc_sampler.h.

#ifndef RATIOD_HMC_GRADIENT_VECTORIZED_H
#define RATIOD_HMC_GRADIENT_VECTORIZED_H

#include <vector>
#include <cmath>
#include <cstring>

// Assumes hmc_sampler.h and RcppEigen.h are already included by the .cpp file.
// IMPORTANT: This file is included from within namespace ratiod_hmc {} in hmc_sampler.cpp.
// Do NOT wrap contents in namespace ratiod_hmc — it would be doubly nested.

namespace vectorized {

// ============================================================================
// Pre-allocated workspace for vectorized gradient (avoids per-call allocation)
// ============================================================================
struct VecGradWorkspace {
  int N = 0;
  std::vector<double> eta_num;
  std::vector<double> eta_denom;
  std::vector<double> mu_num;     // exp(eta_num) — vectorized via Eigen
  std::vector<double> mu_denom;   // exp(eta_denom) — vectorized via Eigen
  std::vector<double> resid_num;
  std::vector<double> resid_denom;
  std::vector<double> effect_dense;  // Pre-expanded RE+spatial+temporal

  // Precomputed data-dependent constants (computed once per dataset)
  uint64_t cached_data_id = 0;
  std::vector<double> lgamma_y_num_p1;    // lgamma(y_num[i] + 1)
  std::vector<double> lgamma_y_denom_p1;  // lgamma(y_denom[i] + 1)
  std::vector<double> lchoose_cache;      // lchoose(n_trials[i], y[i]) for binomial
  std::vector<double> log_y_num_cont;     // log(y_num_cont[i]) for GG, LN
  std::vector<double> log_y_denom_cont;   // log(y_denom_cont[i]) for PG, GG, LN

  // Digamma/lgamma lookup tables for NB/PG: indexed by integer y value.
  // Rebuilt per gradient call (phi changes), but only max_y+1 entries vs N observations.
  // Typically ~50-100 entries vs 500+ observations = 5-10x fewer digamma calls.
  int max_y_num_val = -1;   // max(y_num) across dataset
  int max_y_denom_val = -1; // max(y_denom) across dataset
  std::vector<double> dig_table_num;   // digamma(y + phi_num) - digamma(phi_num) for y = 0..max_y_num
  std::vector<double> lg_table_num;    // log Gamma(y + phi_num) - log Gamma(phi_num) for y = 0..max_y_num
  std::vector<double> dig_table_denom; // digamma(y + phi_denom) - digamma(phi_denom) for y = 0..max_y_denom
  std::vector<double> lg_table_denom;  // log Gamma(y + phi_denom) - log Gamma(phi_denom) for y = 0..max_y_denom

  void init(int n) {
    if (n == N) return;  // Already sized
    N = n;
    eta_num.resize(n);
    eta_denom.resize(n);
    mu_num.resize(n);
    mu_denom.resize(n);
    resid_num.resize(n);
    resid_denom.resize(n);
    effect_dense.resize(n);
    lgamma_y_num_p1.resize(n);
    lgamma_y_denom_p1.resize(n);
    lchoose_cache.resize(n);
    log_y_num_cont.resize(n);
    log_y_denom_cont.resize(n);
    cached_data_id = 0;  // Force recompute on next use
  }

  // Precompute data-only constants (lgamma(y+1), lchoose, log(y)) — called once per dataset
  void precompute(const ModelData& data) {
    if (data.unique_id == cached_data_id) return;  // Already computed for this data
    cached_data_id = data.unique_id;
    max_y_num_val = 0;
    max_y_denom_val = 0;
    for (int i = 0; i < N; i++) {
      lgamma_y_num_p1[i] = std::lgamma(data.y_num[i] + 1.0);
      lgamma_y_denom_p1[i] = std::lgamma(data.y_denom[i] + 1.0);
      // lchoose for binomial/beta-binomial: lgamma(n+1) - lgamma(y+1) - lgamma(n-y+1)
      lchoose_cache[i] = std::lgamma(data.y_denom[i] + 1.0)
                        - std::lgamma(data.y_num[i] + 1.0)
                        - std::lgamma(data.y_denom[i] - data.y_num[i] + 1.0);
      // log(y) for PG/GG/LN families (data-only, never changes)
      log_y_num_cont[i] = (i < static_cast<int>(data.y_num_cont.size()) && data.y_num_cont[i] > 0.0)
                         ? std::log(data.y_num_cont[i]) : 0.0;
      log_y_denom_cont[i] = (i < static_cast<int>(data.y_denom_cont.size()) && data.y_denom_cont[i] > 0.0)
                           ? std::log(data.y_denom_cont[i]) : 0.0;
      // Track max y values for lookup table sizing
      if (data.y_num[i] > max_y_num_val) max_y_num_val = data.y_num[i];
      if (data.y_denom[i] > max_y_denom_val) max_y_denom_val = data.y_denom[i];
    }
  }

  // Build digamma/lgamma lookup tables for current phi values.
  // Called once per gradient evaluation (phi changes each leapfrog step).
  // Cost: ~(max_y_num + max_y_denom) digamma calls vs N*2 without table.
  void build_digamma_tables(double phi_num, double phi_denom, bool need_denom) {
    int tn = max_y_num_val + 1;
    if (static_cast<int>(dig_table_num.size()) < tn) {
      dig_table_num.resize(tn);
      lg_table_num.resize(tn);
    }
    for (int y = 0; y < tn; y++) {
      dig_table_num[y] = tulpa::math::polygamma_rising_difference(0, y, phi_num);
      lg_table_num[y] = tulpa::math::lgamma_ratio_count(y, phi_num);
    }
    if (need_denom) {
      int td = max_y_denom_val + 1;
      if (static_cast<int>(dig_table_denom.size()) < td) {
        dig_table_denom.resize(td);
        lg_table_denom.resize(td);
      }
      for (int y = 0; y < td; y++) {
        dig_table_denom[y] = tulpa::math::polygamma_rising_difference(0, y, phi_denom);
        lg_table_denom[y] = tulpa::math::lgamma_ratio_count(y, phi_denom);
      }
    }
  }
};

// ============================================================================
// Pass 1: Expand grouped effects to dense N-vectors
// ============================================================================

inline void expand_re_single(
    const ModelData& data,
    const ParamLayout& layout,
    const double* params,
    double* dense,
    int N,
    double sigma_re = 0.0  // >0 means non-centered: multiply z by sigma
) {
  const double* re = &params[layout.re_start];
  const auto& group = data.re_group;
  if (sigma_re > 0.0) {
    // Non-centered: dense[i] = sigma * z[g]
    for (int i = 0; i < N; i++) {
      dense[i] = (group[i] > 0) ? sigma_re * re[group[i] - 1] : 0.0;
    }
  } else {
    // Centered: dense[i] = re[g]
    for (int i = 0; i < N; i++) {
      dense[i] = (group[i] > 0) ? re[group[i] - 1] : 0.0;
    }
  }
}

inline void expand_re_crossed(
    const ModelData& data,
    const ParamLayout& layout,
    const double* params,
    double* dense,
    int N,
    const double* sigma_re_terms = nullptr  // non-null means non-centered: multiply z by sigma per term
) {
  std::memset(dense, 0, N * sizeof(double));
  const int n_terms = data.n_re_terms;
  for (int i = 0; i < N; i++) {
    for (int t = 0; t < n_terms; t++) {
      int gidx = data.re_group_multi_flat[i * n_terms + t];
      if (gidx > 0) {
        double scale = (sigma_re_terms != nullptr) ? sigma_re_terms[t] : 1.0;
        dense[i] += scale * params[layout.re_start_multi[t] + gidx - 1];
      }
    }
  }
}

inline void expand_spatial_icar(
    const ModelData& data,
    const double* phi_spatial,
    double* dense,
    int N
) {
  const auto& group = data.spatial_group;
  for (int i = 0; i < N; i++) {
    dense[i] = (group[i] > 0) ? phi_spatial[group[i] - 1] : 0.0;
  }
}

inline void expand_spatial_bym2(
    const ModelData& data,
    const double* phi_spatial,
    const double* theta_bym2,
    double sigma_s, double sigma_u,
    double* dense,
    int N
) {
  double scale = data.bym2_scale_factor;
  const auto& group = data.spatial_group;
  for (int i = 0; i < N; i++) {
    if (group[i] > 0) {
      int s = group[i] - 1;
      dense[i] = sigma_s * phi_spatial[s] * scale + sigma_u * theta_bym2[s];
    } else {
      dense[i] = 0.0;
    }
  }
}

inline void expand_temporal(
    const ModelData& data,
    const double* phi_temporal,
    double* dense,
    int N
) {
  const auto& tidx = data.temporal_time_idx;
  const auto& gidx = data.temporal_group_idx;
  int T = data.n_times;
  for (int i = 0; i < N; i++) {
    if (tidx[i] > 0) {
      int t = tidx[i] - 1;
      int g = gidx[i] - 1;
      dense[i] = phi_temporal[g * T + t];  // Panel temporal: flat index
    } else {
      dense[i] = 0.0;
    }
  }
}

// ============================================================================
// Pass 2: Templated residual kernels (one per ModelType)
// Each computes resid_num[i], resid_denom[i], and phi gradient contributions.
// ============================================================================

template<ModelType MT> struct ModelTag {};

using tulpaRatio::margins::NoMargin;
using tulpaRatio::margins::Poisson;
using tulpaRatio::margins::NegBin;
using tulpaRatio::margins::Gamma;
using tulpaRatio::margins::Lognormal;
using tulpaRatio::margins::Binomial;
using tulpaRatio::margins::BetaBinomial;

template<ModelType MT> struct ResidualFamily;
#define RATIO_RESIDUAL_FAMILY(MT, NUM, DEN) \
    template<> struct ResidualFamily<ModelType::MT> { using Num = NUM; using Den = DEN; };
RATIO_RESIDUAL_FAMILY(BINOMIAL, Binomial, NoMargin)
RATIO_RESIDUAL_FAMILY(NEGBIN_NEGBIN, NegBin, NegBin)
RATIO_RESIDUAL_FAMILY(POISSON_GAMMA, Poisson, Gamma)
RATIO_RESIDUAL_FAMILY(NEGBIN_GAMMA, NegBin, Gamma)
RATIO_RESIDUAL_FAMILY(GAMMA_GAMMA, Gamma, Gamma)
RATIO_RESIDUAL_FAMILY(LOGNORMAL, Lognormal, Lognormal)
RATIO_RESIDUAL_FAMILY(BETA_BINOMIAL, BetaBinomial, NoMargin)
#undef RATIO_RESIDUAL_FAMILY

template<class Margin>
inline tulpaRatio::margins::Observation margin_observation(
    int i, double eta, const ModelData& data, const VecGradWorkspace& ws, bool numerator) {
    tulpaRatio::margins::Observation x{0.0,eta};
    if constexpr (std::is_same<Margin,NoMargin>::value) return x;
    if constexpr (std::is_same<Margin,Gamma>::value || std::is_same<Margin,Lognormal>::value) {
        x.y = numerator ? data.y_num_cont[i] : data.y_denom_cont[i];
        x.log_y = numerator ? ws.log_y_num_cont[i] : ws.log_y_denom_cont[i];
    } else {
        x.y = numerator ? data.y_num[i] : data.y_denom[i];
        x.log_factorial = numerator ? ws.lgamma_y_num_p1[i] : ws.lgamma_y_denom_p1[i];
        if constexpr (std::is_same<Margin,NegBin>::value) {
            x.digamma_difference = numerator ? ws.dig_table_num[int(x.y)] : ws.dig_table_denom[int(x.y)];
            x.lgamma_difference = numerator ? ws.lg_table_num[int(x.y)] : ws.lg_table_denom[int(x.y)];
        }
        if constexpr (std::is_same<Margin,Binomial>::value || std::is_same<Margin,BetaBinomial>::value) {
            x.trials = data.y_denom[i];
            x.log_factorial = ws.lchoose_cache[i];
        }
    }
    return x;
}

template<class Margin>
inline double margin_extra_derivative(const Margin& m, double score) {
    if constexpr (Margin::log_extra) return score / m.phi;
    else return score;
}

template<ModelType MT>
inline void compute_residuals_impl(
    ModelTag<MT>, int N, const double* eta_num, const double* eta_denom,
    const ModelData& data, double phi_num, double phi_denom,
    double* resid_num, double* resid_denom,
    double& grad_phi_num_lik, double& grad_phi_denom_lik,
    double& obs_ll, bool compute_lp, const VecGradWorkspace& ws
) {
    using Num = typename ResidualFamily<MT>::Num;
    using Den = typename ResidualFamily<MT>::Den;
    const Num num(phi_num,compute_lp);
    const Den den(phi_denom,compute_lp);
    double gpn = 0.0, gpd = 0.0, ll = 0.0;
    for (int i = 0; i < N; ++i) {
        const auto x = margin_observation<Num>(i,eta_num[i],data,ws,true);
        const auto y = margin_observation<Den>(i,eta_denom ? eta_denom[i] : 0.0,data,ws,false);
        const auto a = num.template evaluate<true>(x,compute_lp);
        const auto b = den.template evaluate<true>(y,compute_lp);
        resid_num[i] = a.residual;
        resid_denom[i] = b.residual;
        gpn += margin_extra_derivative(num,a.extra);
        gpd += margin_extra_derivative(den,b.extra);
        if (compute_lp) {
            ll += a.ll;
            if constexpr (!std::is_same<Den,NoMargin>::value) ll += b.ll;
        }
    }
    grad_phi_num_lik += gpn;
    grad_phi_denom_lik += gpd;
    obs_ll += ll;
}
// ============================================================================
// Pass 2 dispatcher
// ============================================================================

template<ModelType MT>
inline void compute_residuals(
    int N,
    const double* eta_num, const double* eta_denom,
    const ModelData& data,
    double phi_num, double phi_denom,
    double* resid_num, double* resid_denom,
    double& grad_phi_num_lik, double& grad_phi_denom_lik,
    double& obs_ll, bool compute_lp,
    const VecGradWorkspace& ws
) {
  compute_residuals_impl(
    ModelTag<MT>{}, N, eta_num, eta_denom, data,
    phi_num, phi_denom, resid_num, resid_denom,
    grad_phi_num_lik, grad_phi_denom_lik, obs_ll, compute_lp, ws
  );
}

// ============================================================================
// Pass 3: Scatter-add residuals back to grouped effect gradients
// ============================================================================

inline void accumulate_re_gradient_single(
    const ModelData& data,
    const ParamLayout& layout,
    const double* resid_num, const double* resid_denom,
    bool is_binomial,
    double* grad,
    int N
) {
  for (int i = 0; i < N; i++) {
    int gidx = data.re_group[i];
    if (gidx > 0) {
      double re_grad_i = resid_num[i];
      if (!is_binomial) re_grad_i += resid_denom[i];
      grad[layout.re_start + gidx - 1] += re_grad_i;
    }
  }
}

inline void accumulate_re_gradient_crossed(
    const ModelData& data,
    const ParamLayout& layout,
    const double* resid_num, const double* resid_denom,
    bool is_binomial,
    double* grad,
    int N
) {
  const int n_terms = data.n_re_terms;
  for (int i = 0; i < N; i++) {
    double re_grad_i = resid_num[i];
    if (!is_binomial) re_grad_i += resid_denom[i];
    for (int t = 0; t < n_terms; t++) {
      int gidx = data.re_group_multi_flat[i * n_terms + t];
      if (gidx > 0) {
        grad[layout.re_start_multi[t] + gidx - 1] += re_grad_i;
      }
    }
  }
}

inline void accumulate_temporal_gradient(
    const ModelData& data,
    const double* resid_num, const double* resid_denom,
    bool is_binomial,
    double* grad_temporal_lik,
    int N
) {
  const auto& tidx = data.temporal_time_idx;
  const auto& gidx = data.temporal_group_idx;
  int T = data.n_times;
  for (int i = 0; i < N; i++) {
    if (tidx[i] > 0) {
      // An unshared temporal effect enters the numerator only, so the
      // denominator residual must not feed its gradient.
      double temp_grad_i = resid_num[i];
      if (!is_binomial && data.temporal_shared) temp_grad_i += resid_denom[i];
      int t = tidx[i] - 1;
      int g = gidx[i] - 1;
      grad_temporal_lik[g * T + t] += temp_grad_i;  // Panel temporal: flat index
    }
  }
}

// Accumulate BYM2 spatial gradient from dense residual vectors
inline void accumulate_spatial_gradient_bym2(
    const ModelData& data,
    double sigma_s, double sigma_u,
    const double* resid_num, const double* resid_denom,
    bool is_binomial,
    double* grad_spatial_lik,
    double* grad_theta,
    int N
) {
  double scale = data.bym2_scale_factor;
  const auto& group = data.spatial_group;
  for (int i = 0; i < N; i++) {
    if (group[i] > 0) {
      int s = group[i] - 1;
      double lik_grad = resid_num[i];
      if (!is_binomial) lik_grad += resid_denom[i];
      grad_spatial_lik[s] += lik_grad * sigma_s * scale;
      grad_theta[s] += lik_grad * sigma_u;
    }
  }
}

// ============================================================================
// FUSED SINGLE-PASS GRADIENT
// Computes eta + residuals + gradient accumulation in one observation loop,
// eliminating intermediate N-vectors (eta, resid, effect_dense) and Eigen
// overhead for small p. For p_num,p_denom <= 4, manual dot products keep
// beta accumulators in registers.
//
// Supports: base, +RE (single/crossed), +ICAR, +BYM2, +temporal
// Requires: same exclusions as vectorized (no ZI, slopes, GP, etc.)
// ============================================================================

inline bool can_use_fused(const ModelData& data, const ParamLayout& layout) {
  if (layout.has_zi || layout.has_oi) return false;
  if (layout.has_re_slopes || layout.has_re_correlated_slopes) return false;
  if (data.has_re_slopes) return false;
  if (data.spatial_type == SpatialType::GP ||
      data.spatial_type == SpatialType::MULTISCALE_GP ||
      data.spatial_type == SpatialType::HSGP) return false;
  if (data.has_svc || data.has_tvc || data.has_latent ||
      data.has_spatiotemporal || data.has_temporal_gp ||
      data.has_multiscale_temporal) return false;
  // Only fuse when Eigen overhead exceeds benefit (small p)
  if (data.p_num > 4 || data.p_denom > 4) return false;
  return true;
}

template<ModelType MT>
void compute_gradient_fused(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    std::vector<double>& grad,
    double& obs_ll,
    bool compute_lp,
    std::vector<double>& grad_temporal_lik_out,
    std::vector<double>& grad_spatial_lik_out,
    VecGradWorkspace& ws
) {
  const int N = data.N;
  const int p_num = data.p_num;
  const int p_denom = data.p_denom;
  const double* X_num = data.X_num_flat.data();
  const double* X_denom = data.X_denom_flat.data();
  const double* beta_num = &params[layout.beta_num_start];
  const double* beta_denom = &params[layout.beta_denom_start];

  constexpr bool is_binomial = (MT == ModelType::BINOMIAL) || (MT == ModelType::BETA_BINOMIAL);

  ws.init(N);
  ws.precompute(data);  // lgamma(y+1), lchoose — no-op if already done

  double phi_num = layout.has_phi_num ? std::exp(params[layout.log_phi_num_idx]) : 1.0;
  double phi_denom = layout.has_phi_denom ? std::exp(params[layout.log_phi_denom_idx]) : 1.0;

  // Build digamma/lgamma lookup tables for NB (indexed by integer y, rebuilt each call since phi changes)
  if constexpr (MT == ModelType::NEGBIN_NEGBIN) {
    ws.build_digamma_tables(phi_num, phi_denom, true);
  } else if constexpr (MT == ModelType::NEGBIN_GAMMA) {
    ws.build_digamma_tables(phi_num, phi_denom, false);  // numerator only (denom is Gamma, not NB)
  } else if constexpr (MT == ModelType::BETA_BINOMIAL) {
    // Beta-binomial needs digamma(y+alpha), digamma(n-y+beta) — not integer-indexed by y alone
    // (alpha = p*phi, beta = (1-p)*phi depend on eta which varies per obs). Skip table.
  }

  // Stack-allocated beta gradient accumulators (p <= 4, stay in registers)
  double gb_num[4] = {};
  double gb_denom[4] = {};
  double grad_phi_num_lik = 0.0;
  double grad_phi_denom_lik = 0.0;
  double ll = 0.0;

  // --- Hoisted phi-dependent constants (model-specific, computed once) ---
  using Num = typename ResidualFamily<MT>::Num;
  using Den = typename ResidualFamily<MT>::Den;
  const Num num(phi_num,compute_lp);
  const Den den(phi_denom,compute_lp);

  // BYM2 spatial parameters
  double sigma_s_bym2 = 0.0, sigma_u_bym2 = 0.0, bym2_scale = 0.0;
  const double* phi_spatial = nullptr;
  const double* theta_bym2 = nullptr;
  if (layout.has_spatial) {
    phi_spatial = &params[layout.spatial_start];
    if (data.spatial_type == SpatialType::BYM2) {
      double sigma_total = std::exp(params[layout.log_sigma_bym2_idx]);
      double logit_rho = params[layout.logit_rho_bym2_idx];
      double rho_bym2 = 1.0 / (1.0 + std::exp(-logit_rho));
      sigma_s_bym2 = sigma_total * std::sqrt(rho_bym2);
      sigma_u_bym2 = sigma_total * std::sqrt(1.0 - rho_bym2);
      theta_bym2 = &params[layout.theta_bym2_start];
      bym2_scale = data.bym2_scale_factor;
    }
  }
  const double* phi_temporal = nullptr;
  TemporalView tview;
  if (layout.has_temporal) {
    tview.read(params, data, layout);
    phi_temporal = tview.phi;
  }

  // Families using exp(eta) benefit from Eigen-vectorized exp (SIMD 2-4x faster).
  // Split into 3 passes: eta computation → vectorized exp → scalar kernel.
  // NB/GG: also enables vectorized log pass for log(mu+phi).
  // PG: SIMD exp + eliminates redundant scalar exp(log_rate) per obs.
  // Binomial/LN/BB stay single-pass (simpler kernels, overhead not worthwhile).
  constexpr bool use_multipass = (MT == ModelType::NEGBIN_NEGBIN) ||
                                 (MT == ModelType::NEGBIN_GAMMA) ||
                                 (MT == ModelType::GAMMA_GAMMA) ||
                                 (MT == ModelType::POISSON_GAMMA);

  // Non-centered RE: extract sigma for computing re = sigma * z
  double fused_sigma_re = 0.0;  // >0 when non-centered single term
  std::vector<double> fused_sigma_re_terms;  // per-term for crossed NC
  if (layout.has_re && data.re_parameterization == 1 && !layout.has_re_slopes) {
    if (data.n_re_terms > 1) {
      fused_sigma_re_terms.resize(data.n_re_terms);
      for (int t = 0; t < data.n_re_terms; t++) {
        fused_sigma_re_terms[t] = std::exp(params[layout.log_sigma_re_multi[t]]);
      }
    } else {
      fused_sigma_re = std::exp(params[layout.log_sigma_re_idx]);
    }
  }

  // --- Shared lambda: compute eta for observation i ---
  // (avoids code duplication between needs_exp and !needs_exp paths)
  auto compute_eta_i = [&](int i, double& eta_n, double& eta_d) {
    const double* xi_num = &X_num[i * p_num];
    eta_n = 0.0;
    for (int j = 0; j < p_num; j++)
      eta_n += xi_num[j] * beta_num[j];
    eta_d = 0.0;
    if constexpr (!is_binomial) {
      const double* xi_d = &X_denom[i * p_denom];
      for (int j = 0; j < p_denom; j++)
        eta_d += xi_d[j] * beta_denom[j];
    }
    if (layout.has_re) {
      double re_val;
      if (data.n_re_terms > 1) {
        re_val = 0.0;
        for (int t = 0; t < data.n_re_terms; t++) {
          int g = data.re_group_multi_flat[i * data.n_re_terms + t];
          if (g > 0) {
            double z_or_re = params[layout.re_start_multi[t] + g - 1];
            re_val += fused_sigma_re_terms.empty() ? z_or_re : fused_sigma_re_terms[t] * z_or_re;
          }
        }
      } else {
        int g = data.re_group[i];
        if (g > 0) {
          double z_or_re = params[layout.re_start + g - 1];
          re_val = (fused_sigma_re > 0.0) ? fused_sigma_re * z_or_re : z_or_re;
        } else {
          re_val = 0.0;
        }
      }
      eta_n += re_val;
      if constexpr (!is_binomial) eta_d += re_val;
    }
    if (layout.has_spatial) {
      int s = data.spatial_group[i];
      if (s > 0) {
        s -= 1;
        double sv;
        if (data.spatial_type == SpatialType::BYM2) {
          sv = sigma_s_bym2 * phi_spatial[s] * bym2_scale + sigma_u_bym2 * theta_bym2[s];
        } else {
          sv = phi_spatial[s];
        }
        eta_n += sv;
        if constexpr (!is_binomial) eta_d += sv;
      }
    }
    if (layout.has_temporal && !data.temporal_time_idx.empty()) {
      int t = data.temporal_time_idx[i];
      if (t > 0) {
        int g = data.temporal_group_idx[i] - 1;
        int flat = g * data.n_times + (t - 1);  // Panel temporal: flat index
        eta_n += phi_temporal[flat];
        // An unshared temporal effect enters the numerator only.
        if constexpr (!is_binomial) {
          if (data.temporal_shared) eta_d += phi_temporal[flat];
        }
      }
    }
  };

  // --- Shared lambda: scatter grouped effect gradients for obs i ---
  auto scatter_gradients_i = [&](int i, double resid_n, double resid_d) {
    const double* xi_num = &X_num[i * p_num];
    for (int j = 0; j < p_num; j++)
      gb_num[j] += xi_num[j] * resid_n;
    if constexpr (!is_binomial) {
      const double* xi_d = &X_denom[i * p_denom];
      for (int j = 0; j < p_denom; j++)
        gb_denom[j] += xi_d[j] * resid_d;
    }
    double total_lik_grad = resid_n;
    if constexpr (!is_binomial) total_lik_grad += resid_d;
    if (layout.has_re) {
      if (data.n_re_terms > 1) {
        for (int t = 0; t < data.n_re_terms; t++) {
          int g = data.re_group_multi_flat[i * data.n_re_terms + t];
          if (g > 0) grad[layout.re_start_multi[t] + g - 1] += total_lik_grad;
        }
      } else {
        int g = data.re_group[i];
        if (g > 0) grad[layout.re_start + g - 1] += total_lik_grad;
      }
    }
    if (layout.has_spatial) {
      int s = data.spatial_group[i];
      if (s > 0) {
        s -= 1;
        if (data.spatial_type == SpatialType::BYM2) {
          grad_spatial_lik_out[s] += total_lik_grad * sigma_s_bym2 * bym2_scale;
          grad[layout.theta_bym2_start + s] += total_lik_grad * sigma_u_bym2;
        } else {
          grad_spatial_lik_out[s] += total_lik_grad;
        }
      }
    }
    if (layout.has_temporal && !data.temporal_time_idx.empty()) {
      int t = data.temporal_time_idx[i];
      if (t > 0) {
        int g = data.temporal_group_idx[i] - 1;
        int flat = g * data.n_times + (t - 1);  // Panel temporal: flat index
        double temporal_lik_grad = resid_n;
        if constexpr (!is_binomial) {
          if (data.temporal_shared) temporal_lik_grad += resid_d;
        }
        grad_temporal_lik_out[flat] += temporal_lik_grad;
      }
    }
  };

  const auto process_margin_i = [&](int i, double en, double ed,
                                     const double* mu_n, const double* mu_d) {
    auto x=margin_observation<Num>(i,en,data,ws,true);
    auto y=margin_observation<Den>(i,ed,data,ws,false);
    x.mean=mu_n;
    y.mean=mu_d;
    const auto a=num.template evaluate<true>(x,compute_lp);
    const auto b=den.template evaluate<true>(y,compute_lp);
    grad_phi_num_lik += margin_extra_derivative(num,a.extra);
    grad_phi_denom_lik += margin_extra_derivative(den,b.extra);
    if (compute_lp) {
      ll += a.ll;
      if constexpr (!std::is_same<Den,NoMargin>::value) ll += b.ll;
    }
    scatter_gradients_i(i,a.residual,b.residual);
  };

  if constexpr (use_multipass) {
    // === 2.5-PASS: vectorized exp for NB/PG/GG ===
    using ArrayXd = Eigen::Array<double, Eigen::Dynamic, 1>;

    // Pass 1: compute etas into workspace
    for (int i = 0; i < N; i++) {
      compute_eta_i(i, ws.eta_num[i], ws.eta_denom[i]);
    }

    // Pass 1.5: Eigen-vectorized exp (uses SIMD: SSE2=2x, AVX=4x)
    Eigen::Map<ArrayXd>(ws.mu_num.data(), N) =
        Eigen::Map<const ArrayXd>(ws.eta_num.data(), N).exp();
    if constexpr (!is_binomial) {
      Eigen::Map<ArrayXd>(ws.mu_denom.data(), N) =
          Eigen::Map<const ArrayXd>(ws.eta_denom.data(), N).exp();
    }

    for (int i=0;i<N;++i) {
      process_margin_i(i,ws.eta_num[i],ws.eta_denom[i],&ws.mu_num[i],&ws.mu_denom[i]);
    }
  } else {
    for (int i=0;i<N;++i) {
      double en=0.0,ed=0.0;
      compute_eta_i(i,en,ed);
      process_margin_i(i,en,ed,nullptr,nullptr);
    }
  }

  // Write back beta gradients from stack accumulators
  for (int j = 0; j < p_num; j++)
    grad[layout.beta_num_start + j] += gb_num[j];
  if constexpr (!is_binomial) {
    for (int j = 0; j < p_denom; j++)
      grad[layout.beta_denom_start + j] += gb_denom[j];
  }

  // Phi gradients (with log-transform Jacobian)
  if (layout.has_phi_num) {
    grad[layout.log_phi_num_idx] += phi_num * grad_phi_num_lik;
  }
  if (layout.has_phi_denom) {
    grad[layout.log_phi_denom_idx] += phi_denom * grad_phi_denom_lik;
  }

  obs_ll += ll;
}

// Fused dispatcher: switches on ModelType
inline bool dispatch_fused_gradient(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    std::vector<double>& grad,
    double& obs_ll,
    bool compute_lp,
    std::vector<double>& grad_temporal_lik_out,
    std::vector<double>& grad_spatial_lik_out,
    VecGradWorkspace& ws
) {
  if (!can_use_fused(data, layout)) return false;

  switch (data.model_type) {
    case ModelType::BINOMIAL:
      compute_gradient_fused<ModelType::BINOMIAL>(
        params, data, layout, grad, obs_ll, compute_lp,
        grad_temporal_lik_out, grad_spatial_lik_out, ws);
      return true;
    case ModelType::NEGBIN_NEGBIN:
      compute_gradient_fused<ModelType::NEGBIN_NEGBIN>(
        params, data, layout, grad, obs_ll, compute_lp,
        grad_temporal_lik_out, grad_spatial_lik_out, ws);
      return true;
    case ModelType::POISSON_GAMMA:
      compute_gradient_fused<ModelType::POISSON_GAMMA>(
        params, data, layout, grad, obs_ll, compute_lp,
        grad_temporal_lik_out, grad_spatial_lik_out, ws);
      return true;
    case ModelType::NEGBIN_GAMMA:
      compute_gradient_fused<ModelType::NEGBIN_GAMMA>(
        params, data, layout, grad, obs_ll, compute_lp,
        grad_temporal_lik_out, grad_spatial_lik_out, ws);
      return true;
    case ModelType::GAMMA_GAMMA:
      compute_gradient_fused<ModelType::GAMMA_GAMMA>(
        params, data, layout, grad, obs_ll, compute_lp,
        grad_temporal_lik_out, grad_spatial_lik_out, ws);
      return true;
    case ModelType::LOGNORMAL:
      compute_gradient_fused<ModelType::LOGNORMAL>(
        params, data, layout, grad, obs_ll, compute_lp,
        grad_temporal_lik_out, grad_spatial_lik_out, ws);
      return true;
    case ModelType::BETA_BINOMIAL:
      compute_gradient_fused<ModelType::BETA_BINOMIAL>(
        params, data, layout, grad, obs_ll, compute_lp,
        grad_temporal_lik_out, grad_spatial_lik_out, ws);
      return true;
    default:
      // Model types without a hand-coded likelihood take the arena gradient
      break;
  }
  return false;
}

// ============================================================================
// Check whether a model configuration can use the vectorized path
// ============================================================================

inline bool can_use_vectorized(const ModelData& data, const ParamLayout& layout) {
  if (layout.has_zi || layout.has_oi) return false;
  if (layout.has_re_slopes || layout.has_re_correlated_slopes) return false;
  if (data.has_re_slopes) return false;
  if (data.spatial_type == SpatialType::GP ||
      data.spatial_type == SpatialType::MULTISCALE_GP ||
      data.spatial_type == SpatialType::HSGP) return false;
  if (data.has_svc || data.has_tvc || data.has_latent ||
      data.has_spatiotemporal || data.has_temporal_gp ||
      data.has_multiscale_temporal) return false;
  return true;
}

// ============================================================================
// Main vectorized gradient function (templated on ModelType)
// Computes observation-loop gradients for beta, RE, spatial, temporal, phi.
// Prior gradients and post-loop structural gradients (ICAR, BYM2, RW, AR1)
// are NOT computed here — they remain in compute_gradient_analytical().
//
// Temporal likelihood grads are written into grad_temporal_lik_out (caller
// handles the GMRF prior combination). Spatial likelihood grads are written
// into grad[layout.spatial_start + s] (ICAR) or both spatial and theta_bym2
// positions (BYM2).
// ============================================================================

template<ModelType MT>
void compute_obs_gradients_vectorized(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    std::vector<double>& grad,
    double& obs_ll,
    bool compute_lp,
    std::vector<double>& grad_temporal_lik_out,
    std::vector<double>& grad_spatial_lik_out,
    VecGradWorkspace& ws
) {
  const int N = data.N;
  ws.init(N);
  ws.precompute(data);  // Precompute lgamma(y+1), lchoose — no-op if already done

  const double* beta_num = &params[layout.beta_num_start];
  const double* beta_denom = &params[layout.beta_denom_start];
  double phi_num = layout.has_phi_num ? std::exp(params[layout.log_phi_num_idx]) : 1.0;
  double phi_denom = layout.has_phi_denom ? std::exp(params[layout.log_phi_denom_idx]) : 1.0;

  constexpr bool is_binomial = (MT == ModelType::BINOMIAL) || (MT == ModelType::BETA_BINOMIAL);

  // === Pass 1: Vectorized linear predictor computation (Eigen matvec) ===
  using RowMajorMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  using VectorXd = Eigen::VectorXd;

  Eigen::Map<const RowMajorMatrix> X_num(data.X_num_flat.data(), N, data.p_num);
  Eigen::Map<const VectorXd> b_num(beta_num, data.p_num);
  Eigen::Map<VectorXd> eta_n(ws.eta_num.data(), N);
  eta_n.noalias() = X_num * b_num;

  if constexpr (!is_binomial) {
    Eigen::Map<const RowMajorMatrix> X_denom(data.X_denom_flat.data(), N, data.p_denom);
    Eigen::Map<const VectorXd> b_denom(beta_denom, data.p_denom);
    Eigen::Map<VectorXd> eta_d(ws.eta_denom.data(), N);
    eta_d.noalias() = X_denom * b_denom;
  }

  // Add RE (expand to dense, then vectorized add)
  // Non-centered: multiply z by sigma to get actual RE values
  double sigma_re_nc = 0.0;  // >0 when non-centered
  std::vector<double> sigma_re_terms_nc;  // per-term sigmas for crossed NC
  if (layout.has_re && data.re_parameterization == 1 && !layout.has_re_slopes) {
    if (data.n_re_terms > 1) {
      sigma_re_terms_nc.resize(data.n_re_terms);
      for (int t = 0; t < data.n_re_terms; t++) {
        sigma_re_terms_nc[t] = std::exp(params[layout.log_sigma_re_multi[t]]);
      }
    } else {
      sigma_re_nc = std::exp(params[layout.log_sigma_re_idx]);
    }
  }
  if (layout.has_re) {
    if (data.n_re_terms > 1) {
      expand_re_crossed(data, layout, params.data(), ws.effect_dense.data(), N,
                        sigma_re_terms_nc.empty() ? nullptr : sigma_re_terms_nc.data());
    } else {
      expand_re_single(data, layout, params.data(), ws.effect_dense.data(), N, sigma_re_nc);
    }
    Eigen::Map<VectorXd>(ws.eta_num.data(), N) +=
        Eigen::Map<const VectorXd>(ws.effect_dense.data(), N);
    if constexpr (!is_binomial) {
      Eigen::Map<VectorXd>(ws.eta_denom.data(), N) +=
          Eigen::Map<const VectorXd>(ws.effect_dense.data(), N);
    }
  }

  // Add temporal
  if (layout.has_temporal && !data.temporal_time_idx.empty()) {
    TemporalView tview;
    tview.read(params, data, layout);
    const double* phi_temporal = tview.phi;
    expand_temporal(data, phi_temporal, ws.effect_dense.data(), N);
    Eigen::Map<VectorXd>(ws.eta_num.data(), N) +=
        Eigen::Map<const VectorXd>(ws.effect_dense.data(), N);
    if constexpr (!is_binomial) {
      // An unshared temporal effect enters the numerator only.
      if (data.temporal_shared) {
        Eigen::Map<VectorXd>(ws.eta_denom.data(), N) +=
            Eigen::Map<const VectorXd>(ws.effect_dense.data(), N);
      }
    }
  }

  // Add spatial
  double sigma_s_bym2 = 0.0, sigma_u_bym2 = 0.0;
  if (layout.has_spatial && !data.spatial_group.empty()) {
    const double* phi_spatial = &params[layout.spatial_start];
    if (data.spatial_type == SpatialType::BYM2) {
      double sigma_total = std::exp(params[layout.log_sigma_bym2_idx]);
      double logit_rho = params[layout.logit_rho_bym2_idx];
      double rho_bym2 = 1.0 / (1.0 + std::exp(-logit_rho));
      sigma_s_bym2 = sigma_total * std::sqrt(rho_bym2);
      sigma_u_bym2 = sigma_total * std::sqrt(1.0 - rho_bym2);
      const double* theta_bym2 = &params[layout.theta_bym2_start];
      expand_spatial_bym2(data, phi_spatial, theta_bym2,
                          sigma_s_bym2, sigma_u_bym2,
                          ws.effect_dense.data(), N);
    } else {
      expand_spatial_icar(data, phi_spatial, ws.effect_dense.data(), N);
    }
    Eigen::Map<VectorXd>(ws.eta_num.data(), N) +=
        Eigen::Map<const VectorXd>(ws.effect_dense.data(), N);
    if constexpr (!is_binomial) {
      Eigen::Map<VectorXd>(ws.eta_denom.data(), N) +=
          Eigen::Map<const VectorXd>(ws.effect_dense.data(), N);
    }
  }

  // Build digamma/lgamma lookup tables for NB (same optimization as fused path)
  if constexpr (MT == ModelType::NEGBIN_NEGBIN) {
    ws.build_digamma_tables(phi_num, phi_denom, true);
  } else if constexpr (MT == ModelType::NEGBIN_GAMMA) {
    ws.build_digamma_tables(phi_num, phi_denom, false);
  }

  // === Pass 2: Model-specific residuals (scalar, template-specialized) ===
  double grad_phi_num_lik = 0.0;
  double grad_phi_denom_lik = 0.0;

  compute_residuals<MT>(
    N, ws.eta_num.data(), ws.eta_denom.data(), data,
    phi_num, phi_denom,
    ws.resid_num.data(), ws.resid_denom.data(),
    grad_phi_num_lik, grad_phi_denom_lik,
    obs_ll, compute_lp, ws
  );

  // === Pass 3: Vectorized gradient accumulation ===

  // Beta gradients: grad_beta += X^T * resid
  {
    Eigen::Map<const VectorXd> rn(ws.resid_num.data(), N);
    Eigen::Map<VectorXd> gb_num(&grad[layout.beta_num_start], data.p_num);
    gb_num.noalias() += X_num.transpose() * rn;
  }

  if constexpr (!is_binomial) {
    Eigen::Map<const RowMajorMatrix> X_denom(data.X_denom_flat.data(), N, data.p_denom);
    Eigen::Map<const VectorXd> rd(ws.resid_denom.data(), N);
    Eigen::Map<VectorXd> gb_denom(&grad[layout.beta_denom_start], data.p_denom);
    gb_denom.noalias() += X_denom.transpose() * rd;
  }

  // Phi gradients (with log-transform Jacobian)
  if (layout.has_phi_num) {
    grad[layout.log_phi_num_idx] += phi_num * grad_phi_num_lik;
  }
  if (layout.has_phi_denom) {
    grad[layout.log_phi_denom_idx] += phi_denom * grad_phi_denom_lik;
  }

  // RE gradients (scatter from dense residuals to grouped params)
  // accumulate_re_* writes centered likelihood gradient to grad[re+g]
  // Non-centered post-processing (chain rule) happens in compute_gradient_analytical
  if (layout.has_re) {
    if (data.n_re_terms > 1) {
      accumulate_re_gradient_crossed(data, layout,
        ws.resid_num.data(), ws.resid_denom.data(),
        is_binomial, grad.data(), N);
    } else {
      accumulate_re_gradient_single(data, layout,
        ws.resid_num.data(), ws.resid_denom.data(),
        is_binomial, grad.data(), N);
    }
  }

  // Temporal likelihood gradients → separate buffer for GMRF combination
  if (layout.has_temporal && !data.temporal_time_idx.empty()) {
    accumulate_temporal_gradient(data,
      ws.resid_num.data(), ws.resid_denom.data(),
      is_binomial, grad_temporal_lik_out.data(), N);
  }

  // Spatial likelihood gradients
  if (layout.has_spatial && !data.spatial_group.empty()) {
    int n_spatial = data.n_spatial_units;
    if (data.spatial_type == SpatialType::BYM2) {
      accumulate_spatial_gradient_bym2(data,
        sigma_s_bym2, sigma_u_bym2,
        ws.resid_num.data(), ws.resid_denom.data(),
        is_binomial,
        grad_spatial_lik_out.data(),
        &grad[layout.theta_bym2_start],
        N);
    } else {
      // ICAR
      const auto& group = data.spatial_group;
      for (int i = 0; i < N; i++) {
        if (group[i] > 0) {
          int s = group[i] - 1;
          double lik_grad = ws.resid_num[i];
          if (!is_binomial) lik_grad += ws.resid_denom[i];
          grad_spatial_lik_out[s] += lik_grad;
        }
      }
    }
  }
}

// ============================================================================
// Top-level dispatcher: switches on ModelType, calls template instantiation
// ============================================================================

inline bool dispatch_vectorized_gradient(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    std::vector<double>& grad,
    double& obs_ll,
    bool compute_lp,
    std::vector<double>& grad_temporal_lik_out,
    std::vector<double>& grad_spatial_lik_out,
    VecGradWorkspace& ws
) {
  if (!can_use_vectorized(data, layout)) return false;

  switch (data.model_type) {
    case ModelType::BINOMIAL:
      compute_obs_gradients_vectorized<ModelType::BINOMIAL>(
        params, data, layout, grad, obs_ll, compute_lp,
        grad_temporal_lik_out, grad_spatial_lik_out, ws);
      return true;
    case ModelType::NEGBIN_NEGBIN:
      compute_obs_gradients_vectorized<ModelType::NEGBIN_NEGBIN>(
        params, data, layout, grad, obs_ll, compute_lp,
        grad_temporal_lik_out, grad_spatial_lik_out, ws);
      return true;
    case ModelType::POISSON_GAMMA:
      compute_obs_gradients_vectorized<ModelType::POISSON_GAMMA>(
        params, data, layout, grad, obs_ll, compute_lp,
        grad_temporal_lik_out, grad_spatial_lik_out, ws);
      return true;
    case ModelType::NEGBIN_GAMMA:
      compute_obs_gradients_vectorized<ModelType::NEGBIN_GAMMA>(
        params, data, layout, grad, obs_ll, compute_lp,
        grad_temporal_lik_out, grad_spatial_lik_out, ws);
      return true;
    case ModelType::GAMMA_GAMMA:
      compute_obs_gradients_vectorized<ModelType::GAMMA_GAMMA>(
        params, data, layout, grad, obs_ll, compute_lp,
        grad_temporal_lik_out, grad_spatial_lik_out, ws);
      return true;
    case ModelType::LOGNORMAL:
      compute_obs_gradients_vectorized<ModelType::LOGNORMAL>(
        params, data, layout, grad, obs_ll, compute_lp,
        grad_temporal_lik_out, grad_spatial_lik_out, ws);
      return true;
    case ModelType::BETA_BINOMIAL:
      compute_obs_gradients_vectorized<ModelType::BETA_BINOMIAL>(
        params, data, layout, grad, obs_ll, compute_lp,
        grad_temporal_lik_out, grad_spatial_lik_out, ws);
      return true;
    default:
      break;
  }
  return false;
}

// ============================================================================
// Shared vectorized residual + beta-grad kernel for specialized gradient fns
// ============================================================================
//
// These functions allow specialized gradient functions (HSGP, temporal GP,
// spatiotemporal, slopes) to delegate the expensive obs loop to the same
// template-specialized, Eigen-vectorized infrastructure used by
// compute_obs_gradients_vectorized(). The caller:
//   1. Builds eta vectors (X*beta + RE + custom effects) using Eigen matvec
//   2. Calls compute_residuals_and_beta_grads() for residuals + beta grads
//   3. Scatters residuals to custom effect gradient buffers
//
// This eliminates:
//   - Scalar dot products for eta (replaced by Eigen matvec in caller)
//   - Runtime if/else on model_type (replaced by template dispatch)
//   - Scalar scatter for X^T * resid (replaced by Eigen matvec here)

template<ModelType MT>
void compute_residuals_and_beta_grads(
    const ModelData& data,
    const ParamLayout& layout,
    const double* eta_num,             // INPUT: pre-built [N] with all effects
    const double* eta_denom,           // INPUT: pre-built [N] with all effects
    double* resid_num_out,             // OUTPUT: [N] residuals
    double* resid_denom_out,           // OUTPUT: [N] residuals
    double* grad,                      // beta grads accumulated here
    double& grad_phi_num_lik,          // OUTPUT: phi_num likelihood gradient
    double& grad_phi_denom_lik,        // OUTPUT: phi_denom likelihood gradient
    double& obs_ll,                    // OUTPUT: obs log-likelihood (if compute_lp)
    bool compute_lp,
    double phi_num,
    double phi_denom,
    VecGradWorkspace& ws
) {
    const int N = data.N;
    constexpr bool is_binomial = (MT == ModelType::BINOMIAL) || (MT == ModelType::BETA_BINOMIAL);

    // Ensure workspace is initialized and precomputed
    ws.init(N);
    ws.precompute(data);

    // Build digamma/lgamma lookup tables for NB families
    if constexpr (MT == ModelType::NEGBIN_NEGBIN) {
        ws.build_digamma_tables(phi_num, phi_denom, true);
    } else if constexpr (MT == ModelType::NEGBIN_GAMMA) {
        ws.build_digamma_tables(phi_num, phi_denom, false);
    }

    // === Pass 2: Template-specialized residual computation ===
    // Copy eta into workspace (residual kernels read from ws for multipass SIMD)
    // Skip if caller already built eta directly in ws (avoids UB from self-memcpy)
    if (eta_num != ws.eta_num.data()) {
        std::memcpy(ws.eta_num.data(), eta_num, N * sizeof(double));
    }
    if constexpr (!is_binomial) {
        if (eta_denom != ws.eta_denom.data()) {
            std::memcpy(ws.eta_denom.data(), eta_denom, N * sizeof(double));
        }
    }

    grad_phi_num_lik = 0.0;
    grad_phi_denom_lik = 0.0;

    compute_residuals<MT>(
        N, ws.eta_num.data(), ws.eta_denom.data(), data,
        phi_num, phi_denom,
        ws.resid_num.data(), ws.resid_denom.data(),
        grad_phi_num_lik, grad_phi_denom_lik,
        obs_ll, compute_lp, ws
    );

    // Copy residuals to output (skip if output IS the workspace buffer)
    if (resid_num_out != ws.resid_num.data()) {
        std::memcpy(resid_num_out, ws.resid_num.data(), N * sizeof(double));
    }
    if constexpr (!is_binomial) {
        if (resid_denom_out != ws.resid_denom.data()) {
            std::memcpy(resid_denom_out, ws.resid_denom.data(), N * sizeof(double));
        }
    } else {
        // Always zero resid_denom for binomial — memset has no UB with self-target
        // (unlike memcpy). Workspace may have stale data from prior non-binomial call.
        std::memset(resid_denom_out, 0, N * sizeof(double));
    }

    // === Pass 3: Vectorized beta gradient accumulation ===
    using RowMajorMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using VectorXd = Eigen::VectorXd;

    {
        Eigen::Map<const RowMajorMatrix> X_num(data.X_num_flat.data(), N, data.p_num);
        Eigen::Map<const VectorXd> rn(ws.resid_num.data(), N);
        Eigen::Map<VectorXd> gb_num(&grad[layout.beta_num_start], data.p_num);
        gb_num.noalias() += X_num.transpose() * rn;
    }

    if constexpr (!is_binomial) {
        Eigen::Map<const RowMajorMatrix> X_denom(data.X_denom_flat.data(), N, data.p_denom);
        Eigen::Map<const VectorXd> rd(ws.resid_denom.data(), N);
        Eigen::Map<VectorXd> gb_denom(&grad[layout.beta_denom_start], data.p_denom);
        gb_denom.noalias() += X_denom.transpose() * rd;
    }

    // Phi gradients (with log-transform Jacobian)
    if (layout.has_phi_num) {
        grad[layout.log_phi_num_idx] += phi_num * grad_phi_num_lik;
    }
    if (layout.has_phi_denom) {
        grad[layout.log_phi_denom_idx] += phi_denom * grad_phi_denom_lik;
    }
}

// Runtime dispatcher: switches on data.model_type, calls template instantiation
inline bool dispatch_residuals_and_beta_grads(
    const ModelData& data,
    const ParamLayout& layout,
    const double* eta_num,
    const double* eta_denom,
    double* resid_num_out,
    double* resid_denom_out,
    double* grad,
    double& grad_phi_num_lik,
    double& grad_phi_denom_lik,
    double& obs_ll,
    bool compute_lp,
    double phi_num,
    double phi_denom,
    VecGradWorkspace& ws
) {
    #define DISPATCH_CASE(MT) \
        case MT: \
            compute_residuals_and_beta_grads<MT>( \
                data, layout, eta_num, eta_denom, \
                resid_num_out, resid_denom_out, grad, \
                grad_phi_num_lik, grad_phi_denom_lik, \
                obs_ll, compute_lp, phi_num, phi_denom, ws); \
            return true;

    switch (data.model_type) {
        DISPATCH_CASE(ModelType::BINOMIAL)
        DISPATCH_CASE(ModelType::NEGBIN_NEGBIN)
        DISPATCH_CASE(ModelType::POISSON_GAMMA)
        DISPATCH_CASE(ModelType::NEGBIN_GAMMA)
        DISPATCH_CASE(ModelType::GAMMA_GAMMA)
        DISPATCH_CASE(ModelType::LOGNORMAL)
        DISPATCH_CASE(ModelType::BETA_BINOMIAL)
        default:
            break;
    }
    #undef DISPATCH_CASE
    return false;
}

}  // namespace vectorized
// Note: namespace ratiod_hmc is NOT closed here — the .cpp file handles it

#endif  // RATIOD_HMC_GRADIENT_VECTORIZED_H
