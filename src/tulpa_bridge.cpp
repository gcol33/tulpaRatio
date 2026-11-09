#include <Rcpp.h>
#include <functional>
#include <memory>
#include <vector>
#include <string>
#include <tulpa/nuts_api.h>
#include <tulpa/pg_api.h>
#include <tulpa/likelihood.h>
#include <tulpa/model_data.h>
#include <tulpa/param_layout.h>

#include "lik_specs/ratio_config.h"
#include "lik_specs/lik_helpers.h"

namespace tulpaRatio {
tulpa::LikelihoodSpec build_ratio_likelihood_spec(const RatioConfig& cfg);
}

namespace {

Rcpp::NumericMatrix row_major_matrix(const double* values, int nrow, int ncol) {
  Rcpp::NumericMatrix out(nrow, ncol);
  if (values == nullptr) {
    return out;
  }
  for (int i = 0; i < nrow; ++i) {
    for (int j = 0; j < ncol; ++j) {
      out(i, j) = values[i * ncol + j];
    }
  }
  return out;
}

Rcpp::NumericVector buffer_vector(const double* values, int n) {
  Rcpp::NumericVector out(n);
  if (values == nullptr) {
    return out;
  }
  for (int i = 0; i < n; ++i) {
    out[i] = values[i];
  }
  return out;
}

template <typename T>
Rcpp::IntegerVector buffer_int_vector(const T* values, int n) {
  Rcpp::IntegerVector out(n);
  if (values == nullptr) {
    return out;
  }
  for (int i = 0; i < n; ++i) {
    out[i] = static_cast<int>(values[i]);
  }
  return out;
}

// Copy a column-major Rcpp NumericMatrix into a row-major flat double vector
// with shape [N x p], matching tulpa::ProcessData::X_flat layout.
std::vector<double> rmajor_from_rmatrix(const Rcpp::NumericMatrix& X) {
  const int N = X.nrow();
  const int p = X.ncol();
  std::vector<double> out(static_cast<size_t>(N) * p, 0.0);
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < p; ++j) {
      out[i * p + j] = X(i, j);
    }
  }
  return out;
}

// ----- Per-family payload builder ------------------------------------------
// Each family has its own POD struct. We build the right one based on
// cfg.family, returning a `void*` plus a `unique_ptr<void, ...>` for cleanup.
// Centralising the construction here keeps the family files pure (no Rcpp
// in lik_*.cpp) and the bridge a single switch — no per-family allocation
// boilerplate sprinkled around.
struct ResponsePayload {
  void* ptr = nullptr;
  std::function<void(void*)> deleter;
  ~ResponsePayload() { if (ptr) deleter(ptr); }
  ResponsePayload() = default;
  ResponsePayload(const ResponsePayload&) = delete;
  ResponsePayload& operator=(const ResponsePayload&) = delete;
  ResponsePayload(ResponsePayload&& o) noexcept
      : ptr(o.ptr), deleter(std::move(o.deleter)) { o.ptr = nullptr; }
};

template <typename Payload>
ResponsePayload make_payload(Payload* obj) {
  ResponsePayload r;
  r.ptr = obj;
  r.deleter = [](void* p) { delete static_cast<Payload*>(p); };
  return r;
}

ResponsePayload build_response_payload(
    const std::string& family,
    const Rcpp::IntegerVector& y_num_int,
    const Rcpp::IntegerVector& y_denom_int,
    const Rcpp::NumericVector& y_num_cont,
    const Rcpp::NumericVector& y_denom_cont,
    int N
) {
  using namespace tulpaRatio::lik;

  if (family == "binomial") {
    auto* p = new BinomialResponseData;
    p->y.assign(N, 0); p->n.assign(N, 0);
    for (int i = 0; i < N; ++i) { p->y[i] = y_num_int[i]; p->n[i] = y_denom_int[i]; }
    return make_payload(p);
  }
  if (family == "beta_binomial") {
    auto* p = new BetaBinomialResponseData;
    p->y.assign(N, 0); p->n.assign(N, 0);
    for (int i = 0; i < N; ++i) { p->y[i] = y_num_int[i]; p->n[i] = y_denom_int[i]; }
    return make_payload(p);
  }
  if (family == "poisson_gamma") {
    auto* p = new PoissonGammaResponseData;
    p->y_num.assign(N, 0);
    p->y_denom_cont.assign(N, 0.0);
    for (int i = 0; i < N; ++i) {
      p->y_num[i]        = y_num_int[i];
      p->y_denom_cont[i] = y_denom_cont[i];
    }
    return make_payload(p);
  }
  if (family == "negbin_gamma") {
    auto* p = new NegbinGammaResponseData;
    p->y_num.assign(N, 0);
    p->y_denom_cont.assign(N, 0.0);
    for (int i = 0; i < N; ++i) {
      p->y_num[i]        = y_num_int[i];
      p->y_denom_cont[i] = y_denom_cont[i];
    }
    return make_payload(p);
  }
  if (family == "negbin_negbin") {
    auto* p = new NegbinNegbinResponseData;
    p->y_num.assign(N, 0); p->y_denom.assign(N, 0);
    for (int i = 0; i < N; ++i) { p->y_num[i] = y_num_int[i]; p->y_denom[i] = y_denom_int[i]; }
    return make_payload(p);
  }
  if (family == "gamma_gamma") {
    auto* p = new GammaGammaResponseData;
    p->y_num_cont.assign(N, 0.0);
    p->y_denom_cont.assign(N, 0.0);
    for (int i = 0; i < N; ++i) {
      p->y_num_cont[i]   = y_num_cont[i];
      p->y_denom_cont[i] = y_denom_cont[i];
    }
    return make_payload(p);
  }
  if (family == "lognormal") {
    auto* p = new LognormalResponseData;
    p->y_num_cont.assign(N, 0.0);
    p->y_denom_cont.assign(N, 0.0);
    for (int i = 0; i < N; ++i) {
      p->y_num_cont[i]   = y_num_cont[i];
      p->y_denom_cont[i] = y_denom_cont[i];
    }
    return make_payload(p);
  }

  Rcpp::stop("build_response_payload: unknown family '%s'", family.c_str());
  return ResponsePayload();  // unreachable
}

} // namespace

// [[Rcpp::export]]
int cpp_tulpa_abi_version() {
  tulpa::check_abi_version();
  auto fn = reinterpret_cast<tulpa::GetABIVersionFn>(
    R_GetCCallable("tulpa", "tulpa_get_abi_version")
  );
  return fn();
}

// The ABI constant baked into this package at compile time, taken from the
// tulpa headers seen via LinkingTo. Unguarded, so a mismatch can be reported
// as two numbers rather than thrown.
// [[Rcpp::export]]
int cpp_tulpa_compiled_abi_version() {
  return tulpa::TULPA_ABI_VERSION;
}

// The ABI constant carried by the tulpa DLL currently loaded. Unguarded, as
// above.
// [[Rcpp::export]]
int cpp_tulpa_runtime_abi_version() {
  auto fn = reinterpret_cast<tulpa::GetABIVersionFn>(
    R_GetCCallable("tulpa", "tulpa_get_abi_version")
  );
  return fn();
}

// [[Rcpp::export]]
Rcpp::List cpp_tulpa_pg_binomial_gibbs(
    Rcpp::IntegerVector y,
    Rcpp::IntegerVector n,
    Rcpp::NumericMatrix X,
    Rcpp::IntegerVector group,
    int n_groups,
    int n_iter = 2000,
    int n_warmup = 1000,
    int thin = 1,
    double prior_beta_sd = 10.0,
    double prior_sigma_scale = 2.5,
    bool store_eta = false,
    bool verbose = true,
    int n_threads = 1
) {
  tulpa::PGShimResult result{};
  result.beta = nullptr;
  result.re = nullptr;
  result.sigma_re = nullptr;
  result.eta = nullptr;
  result.r_disp = nullptr;
  result.spatial = nullptr;
  result.tau_spatial = nullptr;

  tulpa::get_pg_binomial_fn()(
    INTEGER(y),
    INTEGER(n),
    REAL(X),
    y.size(),
    X.ncol(),
    INTEGER(group),
    n_groups,
    n_iter,
    n_warmup,
    thin,
    prior_beta_sd,
    prior_sigma_scale,
    store_eta ? 1 : 0,
    verbose ? 1 : 0,
    n_threads,
    &result
  );

  Rcpp::List out = Rcpp::List::create(
    Rcpp::Named("beta") = row_major_matrix(result.beta, result.n_save, result.n_beta),
    Rcpp::Named("re") = row_major_matrix(result.re, result.n_save, result.n_re),
    Rcpp::Named("sigma_re") = buffer_vector(result.sigma_re, result.n_save)
  );

  if (store_eta) {
    out["eta"] = row_major_matrix(result.eta, result.n_save, result.n_obs);
  }

  result.free_buffers();
  return out;
}

// ============================================================================
// LikelihoodSpec-driven NUTS entry point (B1b: 7 ratio families).
//
// Takes a list of design matrices (one per process) so adding a 3+-process
// family is a dispatcher branch + one extra X in X_list, with no signature
// change here. Attaches a per-family response payload, asks the family
// dispatcher for a LikelihoodSpec, then routes through tulpa::get_nuts_fn().
//
// Returns a list shaped like the legacy `cpp_hmc_fit` output (samples,
// log_prob, accept_prob, n_leapfrog, treedepth, divergent, epsilon, sampler)
// for a single chain so backend_hmc.R's existing converter can consume it.
// ============================================================================
// [[Rcpp::export]]
Rcpp::List cpp_tulpaRatio_run_nuts_specs(
    Rcpp::IntegerVector y_num,
    Rcpp::IntegerVector y_denom,
    Rcpp::NumericVector y_num_cont,
    Rcpp::NumericVector y_denom_cont,
    Rcpp::List X_list,
    Rcpp::List cfg_list,
    Rcpp::NumericVector init,
    int n_iter,
    int n_warmup,
    int max_treedepth = 10,
    double adapt_delta = 0.8,
    int seed = 1,
    bool verbose = false,
    std::string gradient_mode = "A_r",
    double sigma_beta = 2.5,
    double phi_prior_shape = 1.0,
    double phi_prior_rate  = 0.01,
    double sigma_prior_scale = 1.0,
    Rcpp::List re_params = Rcpp::List::create(),
    Rcpp::List spatial_params = Rcpp::List::create(),
    Rcpp::List temporal_params = Rcpp::List::create()
) {
  const int n_proc_input = X_list.size();
  if (n_proc_input < 1) {
    Rcpp::stop("cpp_tulpaRatio_run_nuts_specs: X_list must contain at least one "
               "design matrix.");
  }
  Rcpp::NumericMatrix X0 = X_list[0];
  const int N = X0.nrow();

  // ---- Build RatioConfig from R-side list (with hyperprior overrides) -----
  tulpaRatio::RatioConfig cfg;
  cfg.family    = Rcpp::as<std::string>(cfg_list["family"]);
  if (cfg_list.containsElementNamed("zi"))         cfg.zi         = Rcpp::as<std::string>(cfg_list["zi"]);
  if (cfg_list.containsElementNamed("num_link"))   cfg.num_link   = Rcpp::as<std::string>(cfg_list["num_link"]);
  if (cfg_list.containsElementNamed("denom_link")) cfg.denom_link = Rcpp::as<std::string>(cfg_list["denom_link"]);
  cfg.phi_prior_shape   = phi_prior_shape;
  cfg.phi_prior_rate    = phi_prior_rate;
  cfg.sigma_prior_scale = sigma_prior_scale;

  // Map cfg.zi (string) to tulpa::ZIType. The legacy distribution-specific
  // values match what the engine uses to build the X_zi / X_oi blocks in the
  // layout. The templated LikelihoodFn dispatches on data.zi_type to pick the
  // right mixture-form log-pmf.
  auto parse_zi = [](const std::string& s) -> tulpa::ZIType {
    if (s == "none" || s.empty())       return tulpa::ZIType::NONE;
    if (s == "zi_poisson")              return tulpa::ZIType::ZI_POISSON;
    if (s == "zi_negbin")               return tulpa::ZIType::ZI_NEGBIN;
    if (s == "hurdle_poisson")          return tulpa::ZIType::HURDLE_POISSON;
    if (s == "hurdle_negbin")           return tulpa::ZIType::HURDLE_NEGBIN;
    if (s == "zi_binomial")             return tulpa::ZIType::ZI_BINOMIAL;
    if (s == "hurdle_binomial")         return tulpa::ZIType::HURDLE_BINOMIAL;
    if (s == "oi_binomial")             return tulpa::ZIType::OI_BINOMIAL;
    if (s == "zoib")                    return tulpa::ZIType::ZOIB;
    Rcpp::stop("tulpaRatio bridge: unknown zi='%s'.", s.c_str());
    return tulpa::ZIType::NONE;
  };
  const tulpa::ZIType zi_type = parse_zi(cfg.zi);
  const bool has_oi_block = (zi_type == tulpa::ZIType::OI_BINOMIAL) ||
                            (zi_type == tulpa::ZIType::ZOIB);
  // ZI block is present whenever zi_type != NONE EXCEPT pure OI (OI alone uses
  // X_oi only, no X_zi). ZOIB needs both. The other inflation types use X_zi.
  const bool has_zi_block = (zi_type != tulpa::ZIType::NONE) &&
                            (zi_type != tulpa::ZIType::OI_BINOMIAL);

  // ---- Validate response shape vs. family expectations -------------------
  const bool has_y_int  = (y_num.size()  == N) && (y_denom.size()  == N);
  const bool has_y_cont = (y_num_cont.size() == N) || (y_denom_cont.size() == N);
  if (!has_y_int && !has_y_cont) {
    Rcpp::stop("cpp_tulpaRatio_run_nuts_specs: response vectors all empty.");
  }

  // ---- Build LikelihoodSpec & response payload via dispatcher ------------
  tulpa::LikelihoodSpec spec = tulpaRatio::build_ratio_likelihood_spec(cfg);
  ResponsePayload response = build_response_payload(
      cfg.family, y_num, y_denom, y_num_cont, y_denom_cont, N);

  if (n_proc_input != spec.n_processes) {
    Rcpp::stop("cpp_tulpaRatio_run_nuts_specs: X_list has %d matrices but "
               "family='%s' expects %d processes.",
               n_proc_input, cfg.family.c_str(), spec.n_processes);
  }

  // ---- ModelData (n_processes from spec; no spatial/temporal/RE/ZI) -----
  tulpa::ModelData data;
  data.N = N;
  data.n_processes = spec.n_processes;
  data.sigma_beta  = sigma_beta;
  data.model_response_data = response.ptr;
  data.likelihood_spec     = &spec;

  // Hyperprior knobs read by the hand-coded H-kernel (B2). The AD path
  // doesn't read these — it gets the same values from the per-family
  // static cfg set inside each build_*_spec — but the H kernel uses
  // ModelData fields so the entire ratio_config is present in one place.
  data.phi_prior_shape = phi_prior_shape;
  data.phi_prior_rate  = phi_prior_rate;
  data.sigma_re_scale  = sigma_prior_scale;

  data.processes.resize(spec.n_processes);
  int total_p_design = 0;
  for (int k = 0; k < spec.n_processes; ++k) {
    Rcpp::NumericMatrix Xk = X_list[k];
    if (Xk.nrow() != N) {
      Rcpp::stop("cpp_tulpaRatio_run_nuts_specs: X_list[[%d]] has %d rows, "
                 "expected %d (rows of X_list[[1]]).",
                 k + 1, Xk.nrow(), N);
    }
    data.processes[k].p      = Xk.ncol();
    data.processes[k].X_flat = rmajor_from_rmatrix(Xk);
    total_p_design += Xk.ncol();
  }
  data.sharing.init(spec.n_processes);

  // ---- ZI / OI blocks (B1c) ---------------------------------------------
  // R-side passes X_zi (always for ZI/hurdle/ZOIB; mandatory) and X_oi (for
  // OI/ZOIB). The engine extends the layout with beta_zi / beta_oi blocks
  // based on these dimensions and pre-computes logit_zi[i] / logit_oi[i] for
  // each obs before calling the LikelihoodFn — the family kernels never have
  // to multiply X_zi * beta_zi themselves.
  data.zi_type     = zi_type;
  data.p_zi        = 0;
  data.p_oi        = 0;
  // Defaults match the legacy backend (R-side: priors$zi_prior_sd %||% 10.0,
  // priors$oi_prior_sd %||% priors$zi_prior_sd %||% 10.0). R passes overrides
  // via cfg_list$zi_prior_sd / cfg_list$oi_prior_sd so the engine stays
  // signature-stable while the prior survives a path switch.
  data.zi_prior_sd = cfg_list.containsElementNamed("zi_prior_sd")
      ? Rcpp::as<double>(cfg_list["zi_prior_sd"]) : 10.0;
  data.oi_prior_sd = cfg_list.containsElementNamed("oi_prior_sd")
      ? Rcpp::as<double>(cfg_list["oi_prior_sd"]) : data.zi_prior_sd;

  int p_zi_block = 0;
  int p_oi_block = 0;
  if (has_zi_block) {
    if (!cfg_list.containsElementNamed("X_zi")) {
      Rcpp::stop("cpp_tulpaRatio_run_nuts_specs: zi='%s' requires cfg_list$X_zi.",
                 cfg.zi.c_str());
    }
    Rcpp::NumericMatrix X_zi = cfg_list["X_zi"];
    if (X_zi.nrow() != N) {
      Rcpp::stop("cpp_tulpaRatio_run_nuts_specs: X_zi has %d rows, expected %d.",
                 X_zi.nrow(), N);
    }
    data.X_zi_flat = rmajor_from_rmatrix(X_zi);
    data.p_zi      = X_zi.ncol();
    p_zi_block     = X_zi.ncol();
  }
  if (has_oi_block) {
    if (!cfg_list.containsElementNamed("X_oi")) {
      Rcpp::stop("cpp_tulpaRatio_run_nuts_specs: zi='%s' requires cfg_list$X_oi.",
                 cfg.zi.c_str());
    }
    Rcpp::NumericMatrix X_oi = cfg_list["X_oi"];
    if (X_oi.nrow() != N) {
      Rcpp::stop("cpp_tulpaRatio_run_nuts_specs: X_oi has %d rows, expected %d.",
                 X_oi.nrow(), N);
    }
    data.X_oi_flat = rmajor_from_rmatrix(X_oi);
    data.p_oi      = X_oi.ncol();
    p_oi_block     = X_oi.ncol();
  }

  // ---- Random effects (B1d-1: single grouping factor, intercepts + slopes) ----
  // R passes the same re_params list shape used by legacy cpp_hmc_fit. The
  // spec path supports the single-grouping-factor case in B1d-1: random
  // intercepts (n_coefs == 1) and uncorrelated random slopes (n_coefs > 1,
  // correlated == FALSE). Multi-term (crossed/nested) and correlated slopes
  // (LKJ-Cholesky) still fall back to legacy.
  //
  // The engine's tulpa::compute_re_prior reads these fields:
  //   * intercept-only path:  n_re_terms == 0, n_re_groups, re_group
  //   * slopes path:          n_re_terms == 1, has_re_slopes,
  //                           re_n_coefs, re_n_groups_multi,
  //                           re_slope_matrices, re_group_multi_flat
  // tulpa::compute_param_layout matches tulpaRatio's legacy
  // compute_param_layout slot-for-slot, so q_init / sample columns agree
  // without any reordering of the RE block.
  int p_re_block = 0;
  if (re_params.containsElementNamed("n_groups")) {
    const int n_re_groups = Rcpp::as<int>(re_params["n_groups"]);
    const int n_re_terms  = Rcpp::as<int>(re_params["n_terms"]);
    const bool has_slopes = Rcpp::as<bool>(re_params["has_slopes"]);
    const bool has_correlated_slopes =
        re_params.containsElementNamed("has_correlated_slopes")
          ? Rcpp::as<bool>(re_params["has_correlated_slopes"])
          : false;

    if (n_re_groups > 0) {
      if (n_re_terms > 1) {
        Rcpp::stop("cpp_tulpaRatio_run_nuts_specs: multi-term RE is not yet "
                   "supported on the spec path. Use the legacy backend.");
      }
      if (has_correlated_slopes) {
        Rcpp::stop("cpp_tulpaRatio_run_nuts_specs: correlated random slopes "
                   "are not yet supported on the spec path. "
                   "Use the legacy backend.");
      }

      data.re_group = Rcpp::as<std::vector<int>>(re_params["group"]);
      data.n_re_groups = n_re_groups;
      data.re_parameterization =
          Rcpp::as<int>(re_params["parameterization"]);  // 0=centered, 1=NC
      if ((int)data.re_group.size() != N) {
        Rcpp::stop("cpp_tulpaRatio_run_nuts_specs: re_params$group length (%d) "
                   "must equal N (%d).", (int)data.re_group.size(), N);
      }

      if (has_slopes) {
        // Single-term slopes path. Engine reads:
        //   re_n_coefs[0], re_n_groups_multi[0], re_correlated[0],
        //   re_n_chol[0], re_slope_matrices[0], re_group_multi_flat
        //   (length N — same content as re_group when n_terms == 1).
        const int n_coefs = Rcpp::as<int>(re_params["n_coefs"]);
        if (n_coefs < 2) {
          Rcpp::stop("cpp_tulpaRatio_run_nuts_specs: has_slopes=TRUE but "
                     "n_coefs (%d) < 2.", n_coefs);
        }
        const int n_slopes = n_coefs - 1;
        Rcpp::NumericMatrix Xs = Rcpp::as<Rcpp::NumericMatrix>(
            re_params["slope_matrix"]);
        if (Xs.nrow() != N || Xs.ncol() != n_slopes) {
          Rcpp::stop("cpp_tulpaRatio_run_nuts_specs: slope_matrix shape "
                     "(%d x %d) does not match N x (n_coefs - 1) = %d x %d.",
                     Xs.nrow(), Xs.ncol(), N, n_slopes);
        }

        data.n_re_terms = 1;
        data.re_n_groups_multi.assign(1, n_re_groups);
        data.re_offsets.assign(1, 0);
        data.re_group_multi_flat.assign(N, 0);
        for (int i = 0; i < N; ++i) {
          data.re_group_multi_flat[i] = data.re_group[i];
        }

        data.has_re_slopes = true;
        data.has_re_correlated_slopes = false;
        data.re_n_coefs.assign(1, n_coefs);
        data.re_n_slopes.assign(1, n_slopes);
        data.re_correlated.assign(1, false);
        data.re_n_chol.assign(1, 0);
        data.re_slope_matrices.assign(1, std::vector<double>(
            static_cast<size_t>(N) * n_slopes, 0.0));
        for (int i = 0; i < N; ++i) {
          for (int s = 0; s < n_slopes; ++s) {
            data.re_slope_matrices[0][i * n_slopes + s] = Xs(i, s);
          }
        }

        data.total_re_groups   = n_re_groups;
        data.total_re_params   = n_re_groups * n_coefs;
        data.total_sigma_params = n_coefs;
        data.total_chol_params = 0;

        // Layout: n_coefs sigmas + n_groups * n_coefs RE effects.
        p_re_block = n_coefs + n_re_groups * n_coefs;
      } else {
        // Single-term intercept-only path. Mirrors the engine's legacy slot
        // selection: n_re_terms == 0 routes through compute_param_layout's
        // single-term branch and through generic_re_effect's re_group fallback.
        data.n_re_terms = 0;
        data.has_re_slopes = false;
        data.has_re_correlated_slopes = false;
        data.total_re_groups = n_re_groups;
        data.total_re_params = n_re_groups;
        data.total_sigma_params = 1;
        data.total_chol_params = 0;
        // 1 (log_sigma) + n_re_groups (z or re).
        p_re_block = 1 + n_re_groups;
      }

      // Disable the hand-coded H gradient when RE is present: the H-kernel
      // asserts no latent structure. The autodiff path picks up RE via
      // tulpa::compute_log_post_generic, which includes the RE prior and
      // adds the per-obs RE effect to eta. Lifting this requires extending
      // lik_grad_h_kernel.cpp with the RE prior gradient + Z^T residual
      // accumulation; deferred to a follow-up.
      spec.gradient_fn = nullptr;
    }
  }

  // ---- Spatial (B1d Step 2: ICAR / BYM2, non-collapsed) -----------------
  // R passes the same spatial_params list shape used by legacy cpp_hmc_fit.
  // The spec path supports:
  //   - "icar": log_tau + n_units phi values  (1 + n_units params)
  //   - "bym2": log_sigma + logit_rho + 2*n_units (phi_scaled, theta)
  //                                     (2 + 2*n_units params)
  // Collapsed parameterisations and CAR_PROPER / GP / HSGP / multiscale GP
  // are deferred. The engine's compute_layout + compute_log_post_generic
  // wires the prior and adds the per-obs spatial effect to eta when these
  // fields are set.
  int p_spatial_block = 0;
  if (spatial_params.containsElementNamed("type")) {
    const std::string spatial_type =
        Rcpp::as<std::string>(spatial_params["type"]);
    if (spatial_type != "none" && !spatial_type.empty()) {
      const std::string param =
          spatial_params.containsElementNamed("parameterization")
            ? Rcpp::as<std::string>(spatial_params["parameterization"])
            : std::string("standard");
      if (param == "collapsed") {
        Rcpp::stop("cpp_tulpaRatio_run_nuts_specs: collapsed spatial "
                   "parameterisations are not yet supported on the spec path. "
                   "Use the legacy backend.");
      }
      if (spatial_type == "icar") {
        data.spatial_type = tulpa::SpatialType::ICAR;
      } else if (spatial_type == "bym2") {
        data.spatial_type = tulpa::SpatialType::BYM2;
      } else {
        Rcpp::stop("cpp_tulpaRatio_run_nuts_specs: spatial type '%s' is not "
                   "yet supported on the spec path. Supported: 'icar', 'bym2'.",
                   spatial_type.c_str());
      }

      data.spatial_sigma_prior_U = Rcpp::as<double>(spatial_params["sigma_prior_U"]);
      data.spatial_sigma_prior_alpha = Rcpp::as<double>(spatial_params["sigma_prior_alpha"]);
      data.spatial_group   =
          Rcpp::as<std::vector<int>>(spatial_params["group"]);
      data.set_spatial_adjacency(
          Rcpp::as<int>(spatial_params["n_units"]),
          Rcpp::as<std::vector<int>>(spatial_params["adj_row_ptr"]),
          Rcpp::as<std::vector<int>>(spatial_params["adj_col_idx"]),
          Rcpp::as<std::vector<int>>(spatial_params["n_neighbors"]));
      if (spatial_params.containsElementNamed("bym2_scale")) {
        data.bym2_scale_factor = Rcpp::as<double>(spatial_params["bym2_scale"]);
      }
      if (spatial_params.containsElementNamed("Q_inv") &&
          spatial_params.containsElementNamed("L_Q")) {
        SEXP qi = spatial_params["Q_inv"];
        SEXP lq = spatial_params["L_Q"];
        if (!Rf_isNull(qi) && !Rf_isNull(lq)) {
          data.spatial_Q_inv = Rcpp::as<std::vector<double>>(qi);
          data.spatial_L_Q   = Rcpp::as<std::vector<double>>(lq);
        }
      }
      if ((int)data.spatial_group.size() != N) {
        Rcpp::stop("cpp_tulpaRatio_run_nuts_specs: spatial_params$group length "
                   "(%d) must equal N (%d).",
                   (int)data.spatial_group.size(), N);
      }

      p_spatial_block = (data.spatial_type == tulpa::SpatialType::BYM2)
          ? (2 + 2 * data.n_spatial_units)
          : (1 +     data.n_spatial_units);

      // Same rationale as RE: the H-kernel rejects spatial structure.
      spec.gradient_fn = nullptr;
    }
  }

  // ---- Temporal (B1d Step 3: RW1 / RW2 / AR1) ---------------------------
  // Layout sizes follow compute_param_layout exactly:
  //   - RW1/RW2:  log_tau + n_temporal_params           = 1 + n_temporal_params
  //   - AR1   :  log_tau + logit_rho + n_temporal_params = 2 + n_temporal_params
  //   - GP    : deferred (separate hyperprior surface)
  // n_temporal_params = n_times * n_temporal_groups (R-side already computed).
  int p_temporal_block = 0;
  if (temporal_params.containsElementNamed("type")) {
    const std::string temporal_type =
        Rcpp::as<std::string>(temporal_params["type"]);
    if (temporal_type != "none" && !temporal_type.empty()) {
      if (temporal_type == "rw1") {
        data.temporal_type = tulpa::TemporalType::RW1;
      } else if (temporal_type == "rw2") {
        data.temporal_type = tulpa::TemporalType::RW2;
      } else if (temporal_type == "ar1") {
        data.temporal_type = tulpa::TemporalType::AR1;
      } else {
        Rcpp::stop("cpp_tulpaRatio_run_nuts_specs: temporal type '%s' is not "
                   "yet supported on the spec path. Supported: 'rw1', 'rw2', "
                   "'ar1'.", temporal_type.c_str());
      }

      data.temporal_time_idx =
          Rcpp::as<std::vector<int>>(temporal_params["time_idx"]);
      data.temporal_group_idx =
          Rcpp::as<std::vector<int>>(temporal_params["group_idx"]);
      data.n_times             = Rcpp::as<int>(temporal_params["n_times"]);
      data.n_temporal_groups   = Rcpp::as<int>(temporal_params["n_groups"]);
      data.n_temporal_params   = Rcpp::as<int>(temporal_params["n_params"]);
      data.temporal_cyclic     = Rcpp::as<bool>(temporal_params["cyclic"]);
      data.temporal_shared     = Rcpp::as<bool>(temporal_params["shared"]);
      data.ar1_rho_prior_a = Rcpp::as<double>(temporal_params["rho_prior_a"]);
      data.ar1_rho_prior_b = Rcpp::as<double>(temporal_params["rho_prior_b"]);
      if (temporal_params.containsElementNamed("sigma_prior_U")) {
        data.temporal_sigma_prior_U =
            Rcpp::as<double>(temporal_params["sigma_prior_U"]);
      }
      if (temporal_params.containsElementNamed("sigma_prior_alpha")) {
        data.temporal_sigma_prior_alpha =
            Rcpp::as<double>(temporal_params["sigma_prior_alpha"]);
      }

      if ((int)data.temporal_time_idx.size() != N) {
        Rcpp::stop("cpp_tulpaRatio_run_nuts_specs: temporal_params$time_idx "
                   "length (%d) must equal N (%d).",
                   (int)data.temporal_time_idx.size(), N);
      }

      const int hyper = (data.temporal_type == tulpa::TemporalType::AR1) ? 2
                                                                          : 1;
      p_temporal_block = hyper + data.n_temporal_params;

      // H-kernel rejects temporal structure too.
      spec.gradient_fn = nullptr;
    }
  }

  // ---- Param layout (engine derives from data + spec) -------------------
  tulpa::ParamLayout layout = tulpa::compute_layout(data);

  int expected_total = total_p_design + p_re_block + p_spatial_block
                     + p_temporal_block + p_zi_block + p_oi_block
                     + spec.n_extra_params;
  if (layout.total_params != expected_total) {
    Rcpp::stop("cpp_tulpaRatio_run_nuts_specs: layout.total_params (%d) != "
               "expected (%d) for family='%s' [sum(p_k)=%d, p_re=%d, "
               "p_spatial=%d, p_temporal=%d, p_zi=%d, p_oi=%d, n_extra=%d]. "
               "Spec path expects no other latent structure.",
               layout.total_params, expected_total, cfg.family.c_str(),
               total_p_design, p_re_block, p_spatial_block, p_temporal_block,
               p_zi_block, p_oi_block, spec.n_extra_params);
  }
  if (init.size() != expected_total) {
    Rcpp::stop("cpp_tulpaRatio_run_nuts_specs: init length (%d) != expected (%d) "
               "for family='%s'.",
               (int)init.size(), expected_total, cfg.family.c_str());
  }

  // ---- Pin gradient mode -------------------------------------------------
  int grad_rc = tulpa::set_gradient_mode_str(gradient_mode.c_str());
  if (grad_rc != 0) {
    Rcpp::stop("cpp_tulpaRatio_run_nuts_specs: unrecognised gradient_mode '%s'.",
               gradient_mode.c_str());
  }
  // NB: tulpa's dispatcher (hmc_gradient_dispatch.h:33) routes a registered
  // `spec.gradient_fn` for every mode except NUMERICAL — so a caller passing
  // gradient_mode = "A_r" while the spec ships `grad_h_<family>` actually
  // runs the hand-coded H kernel. In practice the H kernel tracks the
  // legacy backend's analytical/AD gradients more tightly than tulpa's
  // generic arena AD does (matching summation order on the same closed-form
  // derivatives), so we leave the hook in place.

  // ---- Run NUTS via the registered tulpa entry point ---------------------
  std::vector<double> init_vec(init.begin(), init.end());
  tulpa::NUTSResult result{};
  tulpa::get_nuts_fn()(
    &data,
    &layout,
    init_vec.data(),
    layout.total_params,
    n_iter,
    n_warmup,
    max_treedepth,
    adapt_delta,
    static_cast<unsigned int>(seed),
    verbose ? 1 : 0,
    nullptr,            // inv_metric_diag (v11) — default mass-adaptation
    &result
  );

  const int n_save = result.n_sample;
  const int n_par  = result.n_params;

  Rcpp::NumericMatrix samples = row_major_matrix(result.samples, n_save, n_par);
  Rcpp::NumericVector log_prob    = buffer_vector(result.log_prob,    n_save);
  Rcpp::NumericVector accept_prob = buffer_vector(result.accept_prob, n_save);
  Rcpp::IntegerVector divergent   = buffer_int_vector(result.divergent, n_save);
  Rcpp::IntegerVector treedepth   = buffer_int_vector(result.treedepth, n_save);
  Rcpp::IntegerVector n_leapfrog(n_save);
  for (int s = 0; s < n_save; ++s) {
    int td = treedepth[s];
    n_leapfrog[s] = (td > 0) ? ((1 << td) - 1) : 1;
  }

  Rcpp::List out = Rcpp::List::create(
    Rcpp::Named("samples")     = samples,
    Rcpp::Named("log_prob")    = log_prob,
    Rcpp::Named("accept_prob") = accept_prob,
    Rcpp::Named("n_leapfrog")  = n_leapfrog,
    Rcpp::Named("treedepth")   = treedepth,
    Rcpp::Named("divergent")   = divergent,
    Rcpp::Named("epsilon")     = result.epsilon,
    Rcpp::Named("sampler")     = std::string(result.sampler[0] ? result.sampler : "NUTS")
  );

  result.free_buffers();
  return out;
}
