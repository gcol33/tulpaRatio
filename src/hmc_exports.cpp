#include <RcppEigen.h>
#include "hmc_grad_internal.h"
#include "hmc_model_data_blocks.h"
#include "omp_chain_team.h"
#include "omp_thread_scope.h"
#include "cov_type_code.h"

using namespace Rcpp;

namespace ratiod_hmc {
// R wrapper version - for single chain or non-parallel use
HMCResult run_hmc_chain(
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
  // Runtime gradient check: compare active gradient function against numerical
  verify_gradient_or_fallback(q_init, data, layout);

  // Run C++ version - pass verbose through for debugging
  HMCResultCpp cpp_result = run_hmc_chain_cpp(
    q_init, data, layout, n_iter, n_warmup, L, chain_id, seed, verbose, max_treedepth, metric_type, adapt_delta, walnuts
  );

  // Convert to R result
  int n_params = q_init.size();
  HMCResult result = cpp_to_r_result(cpp_result, n_params);

  if (verbose) {
    int n_div = 0;
    for (int i = 0; i < cpp_result.n_sample; i++) {
      n_div += cpp_result.divergent[i];
    }
    Rcpp::Rcerr << "Chain " << (chain_id + 1) << " complete. "
                << "Divergent: " << n_div << std::endl;
  }

  return result;
}

// =====================================================================
// Run multiple chains in parallel using OpenMP
// =====================================================================

std::vector<HMCResult> run_hmc_parallel_chains(
    const std::vector<double>& q_init,
    const ModelData& data,
    int n_iter,
    int n_warmup,
    int L,
    int n_chains,
    unsigned int seed,
    bool verbose,
    int max_treedepth,
    MassMatrixType metric_type,
    double adapt_delta,
    const tulpa::WalnutsConfig* walnuts
) {
  ParamLayout layout = compute_param_layout(data);
  int n_params = layout.total_params;

  // Runtime gradient check: compare active gradient function against numerical
  // BEFORE spawning parallel chains. This is single-threaded, so R API and
  // g_gradient_mode modification are safe.
  verify_gradient_or_fallback(q_init, data, layout);

  // Use pure C++ containers in parallel region
  std::vector<HMCResultCpp> cpp_results(n_chains);

  // Thread-safe autodiff: Each chain creates its own tape via TapeScope (RAII).

  require_gradient_supported(data, layout);

  if (n_chains > 1) {
    // The team stays at ratiod_omp::chain_team_size() for the whole session;
    // n_cores caps how many chains are in flight at once. Sizing the team from
    // n_cores instead let it shrink between fits, which corrupts the heap on
    // Windows (see omp_chain_team.h). Without OpenMP the helper runs the chains
    // in order. verbose is off inside the region either way; the per-chain
    // summary is printed below.
    const int max_concurrent = data.n_cores > 0 ? data.n_cores : n_chains;
    ratiod_omp::for_each_chain(n_chains, max_concurrent, [&](int c) {
      cpp_results[c] = run_hmc_chain_cpp(
        q_init, data, layout,
        n_iter, n_warmup, L, c, seed, false, max_treedepth, metric_type, adapt_delta, walnuts
      );
    });
  } else {
    // Single chain - run sequentially with verbose output
    cpp_results[0] = run_hmc_chain_cpp(
      q_init, data, layout,
      n_iter, n_warmup, L, 0, seed, verbose, max_treedepth, metric_type, adapt_delta, walnuts
    );
  }

  // Convert to R objects outside parallel region (single-threaded)
  std::vector<HMCResult> results(n_chains);
  for (int c = 0; c < n_chains; c++) {
    results[c] = cpp_to_r_result(cpp_results[c], n_params);

    if (verbose && n_chains > 1) {
      // Print summary if we ran in parallel (verbose was disabled during parallel run)
      int n_div = 0;
      for (int i = 0; i < cpp_results[c].n_sample; i++) {
        n_div += cpp_results[c].divergent[i];
      }
      Rcpp::Rcerr << "Chain " << (c + 1) << " complete. "
                  << "Divergent: " << n_div << std::endl;
    }
  }

  return results;
}

} // namespace ratiod_hmc

// =====================================================================
// R EXPORTS
// =====================================================================

// HMC sampler with bundled list arguments to avoid R's 65-arg limit for .Call
// Parameters are bundled into logical groups:
//   re_params: random effects (group, n_groups, n_terms, group_matrix, slopes, etc.)
//   spatial_params: spatial structure (type, group, adjacency, etc.)
//   temporal_params: temporal structure (type, time_idx, group_idx, etc.)
//   prior_params: prior hyperparameters
//   zi_params: zero-inflation (type, X_zi, prior_sd)
//   latent_params: latent factors
//   st_params: spatiotemporal interaction
// The chains' results as the list R reads: one chain's vectors directly, or
// a list per field with one entry per chain. The collapsed fields (the inner
// Laplace modes of a collapsed GP, ICAR or BYM2 field) follow the same shape,
// so each chain's draws stay paired with its own field.
Rcpp::List hmc_results_list(const std::vector<ratiod_hmc::HMCResult>& results, int L) {
  const int n_chains = static_cast<int>(results.size());
  std::string sampler_name = (L == 0) ? "NUTS" : "HMC";
  for (const auto& r : results) {
    if (!r.sampler.empty()) sampler_name = r.sampler;
  }
  using ratiod_hmc::HMCResult;
  auto field = [&](auto get) -> SEXP {
    if (n_chains == 1) return Rcpp::wrap(get(results[0]));
    Rcpp::List out(n_chains);
    for (int c = 0; c < n_chains; c++) out[c] = get(results[c]);
    return out;
  };
  Rcpp::NumericVector epsilon(n_chains);
  for (int c = 0; c < n_chains; c++) epsilon[c] = results[c].epsilon;

  Rcpp::List ret = Rcpp::List::create(
    Rcpp::Named("samples") = field([](const HMCResult& r) { return r.samples; }),
    Rcpp::Named("log_prob") = field([](const HMCResult& r) { return r.log_prob; }),
    Rcpp::Named("energy") = field([](const HMCResult& r) { return r.energy; }),
    Rcpp::Named("accept_prob") = field([](const HMCResult& r) { return r.accept_prob; }),
    Rcpp::Named("n_leapfrog") = field([](const HMCResult& r) { return r.n_leapfrog; }),
    Rcpp::Named("treedepth") = field([](const HMCResult& r) { return r.treedepth; }),
    Rcpp::Named("divergent") = field([](const HMCResult& r) { return r.divergent; }),
    Rcpp::Named("epsilon") = n_chains == 1 ? Rcpp::wrap(epsilon[0]) : Rcpp::wrap(epsilon),
    Rcpp::Named("n_warmup") = results[0].n_warmup,
    Rcpp::Named("n_sample") = results[0].n_sample,
    Rcpp::Named("n_chains") = n_chains,
    Rcpp::Named("sampler") = sampler_name
  );
  if (results[0].n_gp_collapsed > 0) {
    ret["gp_w_star"] = field([](const HMCResult& r) { return r.gp_w_star; });
  }
  if (results[0].n_st_collapsed > 0) {
    ret["st_delta_draws"] = field([](const HMCResult& r) { return r.st_delta; });
  }
  if (results[0].n_icar_collapsed > 0) {
    ret["icar_phi_star"] = field([](const HMCResult& r) { return r.icar_phi_star; });
    if (results[0].bym2_theta_star.nrow() > 0) {
      ret["bym2_theta_star"] = field([](const HMCResult& r) { return r.bym2_theta_star; });
    }
  }
  return ret;
}

namespace ratiod_hmc {

// The areal spatial field (ICAR, BYM2, proper CAR) and the restriction any
// spatial field may carry. A GP-type field arrives in the GP bundle instead.
static void apply_spatial_params(ModelData& data, const Rcpp::List& spatial_params) {
  data.auto_spatial_coordinates = spatial_params.containsElementNamed("parameterization") &&
      Rcpp::as<std::string>(spatial_params["parameterization"]) == "auto";
  const std::string type = Rcpp::as<std::string>(spatial_params["type"]);
  if (type == "icar") {
    data.spatial_type = SpatialType::ICAR;
  } else if (type == "bym2") {
    data.spatial_type = SpatialType::BYM2;
  } else if (type == "car_proper") {
    data.spatial_type = SpatialType::CAR_PROPER;
  } else {
    data.spatial_type = SpatialType::NONE;
  }

  data.spatial_group = Rcpp::as<std::vector<int>>(spatial_params["group"]);
  data.n_spatial_units = Rcpp::as<int>(spatial_params["n_units"]);
  apply_areal_structure(data, spatial_params);
  data.adj_row_ptr = Rcpp::as<std::vector<int>>(spatial_params["adj_row_ptr"]);
  data.adj_col_idx = Rcpp::as<std::vector<int>>(spatial_params["adj_col_idx"]);
  data.n_neighbors = Rcpp::as<std::vector<int>>(spatial_params["n_neighbors"]);
  data.bym2_scale_factor = Rcpp::as<double>(spatial_params["bym2_scale"]);

  // Precision mass matrix data (Q_inv and L_Q for ICAR/BYM2)
  if (spatial_params.containsElementNamed("Q_inv") &&
      spatial_params.containsElementNamed("L_Q")) {
    SEXP qi_sexp = spatial_params["Q_inv"];
    SEXP lq_sexp = spatial_params["L_Q"];
    if (!Rf_isNull(qi_sexp) && !Rf_isNull(lq_sexp)) {
      data.spatial_Q_inv = Rcpp::as<std::vector<double>>(qi_sexp);
      data.spatial_L_Q = Rcpp::as<std::vector<double>>(lq_sexp);
    }
  }

  if (data.spatial_type == SpatialType::CAR_PROPER) {
    if (spatial_params.containsElementNamed("rho_lower"))
      data.car_rho_lower = Rcpp::as<double>(spatial_params["rho_lower"]);
    if (spatial_params.containsElementNamed("rho_upper"))
      data.car_rho_upper = Rcpp::as<double>(spatial_params["rho_upper"]);
    if (spatial_params.containsElementNamed("rho_prior_a"))
      data.car_rho_prior_a = Rcpp::as<double>(spatial_params["rho_prior_a"]);
    if (spatial_params.containsElementNamed("rho_prior_b"))
      data.car_rho_prior_b = Rcpp::as<double>(spatial_params["rho_prior_b"]);
    if (spatial_params.containsElementNamed("center"))
      data.car_center = Rcpp::as<bool>(spatial_params["center"]);
  }

  // Collapsed ICAR/BYM2 parameterization
  data.icar_collapsed = false;
  data.bym2_collapsed = false;
  if (spatial_params.containsElementNamed("parameterization") &&
      Rcpp::as<std::string>(spatial_params["parameterization"]) == "collapsed") {
    data.icar_collapsed = (data.spatial_type == SpatialType::ICAR);
    data.bym2_collapsed = (data.spatial_type == SpatialType::BYM2);
  }
  if (spatial_params.containsElementNamed("rsr")) {
    data.rsr = restriction_from(Rcpp::as<Rcpp::List>(spatial_params["rsr"]), data.N);
  }
}

// A NNGP, HSGP or multi-scale GP spatial field. It replaces whatever areal
// state the spatial bundle wrote for the same field.
static void apply_gp_params(ModelData& data, const Rcpp::List& gp) {
  data.has_gp = false;
  data.has_multiscale_gp = false;
  data.has_hsgp = false;
  const std::string gp_type = Rcpp::as<std::string>(gp["gp_type"]);
  if (gp_type == "none") return;

  data.spatial_group.clear();
  data.n_spatial_units = 0;
  data.adj_row_ptr.clear();
  data.adj_col_idx.clear();
  data.n_neighbors.clear();
  data.bym2_scale_factor = 1.0;
  data.spatial_Q_inv.clear();
  data.spatial_L_Q.clear();
  data.areal_components = ratiod_constraints::ArealComponents();
  data.areal_node_prec.clear();

  const std::vector<double> coords = Rcpp::as<std::vector<double>>(gp["coords"]);
  const bool shared = Rcpp::as<bool>(gp["shared"]);
  data.gp_sigma2_prior_U = Rcpp::as<double>(gp["sigma2_prior_U"]);
  data.gp_sigma2_prior_alpha = Rcpp::as<double>(gp["sigma2_prior_alpha"]);

  if (gp_type == "gp") {
    data.spatial_type = SpatialType::GP;
    data.has_gp = true;
    // 1-based from R, 0-based here
    auto zero_based = [](const std::vector<int>& v) {
      std::vector<int> out(v.size());
      for (size_t i = 0; i < v.size(); i++) out[i] = v[i] - 1;
      return out;
    };
    data.gp_data.n_obs = Rcpp::as<int>(gp["n_unique"]);  // Unique locations
    data.gp_data.nn = Rcpp::as<int>(gp["nn"]);
    data.gp_data.coords = coords;
    data.gp_data.nn_idx = Rcpp::as<std::vector<int>>(gp["nn_idx"]);
    data.gp_data.nn_dist = Rcpp::as<std::vector<double>>(gp["nn_dist"]);
    data.gp_data.nn_neighbor_dist = Rcpp::as<std::vector<double>>(gp["nn_neighbor_dist"]);
    data.gp_data.obs_to_loc = zero_based(Rcpp::as<std::vector<int>>(gp["gp_obs_to_loc"]));
    data.gp_data.nn_order = zero_based(Rcpp::as<std::vector<int>>(gp["nn_order"]));
    data.gp_data.nn_order_inv = zero_based(Rcpp::as<std::vector<int>>(gp["nn_order_inv"]));
    data.gp_data.cov_type = ratiod_cov::parse_cov_type(
        Rcpp::as<std::string>(gp["cov_type"]), Rcpp::as<double>(gp["nu"]));
    data.gp_data.nu = Rcpp::as<double>(gp["nu"]);
    data.gp_data.shared = shared;

    data.gp_phi_prior_lower = Rcpp::as<double>(gp["phi_prior_lower"]);
    data.gp_phi_prior_upper = Rcpp::as<double>(gp["phi_prior_upper"]);

    // centred (0), non-centred (1) or collapsed
    const std::string param = Rcpp::as<std::string>(gp["parameterization"]);
    data.gp_collapsed = (param == "collapsed");
    data.gp_parameterization = (param == "noncentered" || param == "auto") ? 1 : 0;
    data.auto_gp_coordinates = param == "auto";

  } else if (gp_type == "multiscale_gp") {
    data.spatial_type = SpatialType::MULTISCALE_GP;
    data.has_multiscale_gp = true;

    // Both scales on one Hilbert-space basis
    ratiod_hsgp::setup_hsgp_2d(coords, data.N, Rcpp::as<int>(gp["hsgp_m"]),
                                Rcpp::as<double>(gp["c_boundary"]), shared,
                                data.msgp_hsgp_data);
    data.multiscale_gp_data.shared = shared;
    data.multiscale_gp_data.n_obs = data.N;
    data.multiscale_gp_data.sampler =
        ratiod_gp::parse_msgp_sampler(Rcpp::as<std::string>(gp["sampler"]));

    // Lengthscale prior means from range bounds (geometric mean on log scale)
    data.ms_log_ls_local_mean = 0.5 * (std::log(Rcpp::as<double>(gp["range_local_lower"])) +
                                       std::log(Rcpp::as<double>(gp["range_local_upper"])));
    data.ms_log_ls_local_sd = 0.5;
    data.ms_log_ls_regional_mean = 0.5 * (std::log(Rcpp::as<double>(gp["range_regional_lower"])) +
                                          std::log(Rcpp::as<double>(gp["range_regional_upper"])));
    data.ms_log_ls_regional_sd = 0.5;

    data.ms_sigma2_local_prior_U = Rcpp::as<double>(gp["sigma2_local_prior_U"]);
    data.ms_sigma2_local_prior_alpha = Rcpp::as<double>(gp["sigma2_local_prior_alpha"]);
    data.ms_sigma2_regional_prior_U = Rcpp::as<double>(gp["sigma2_regional_prior_U"]);
    data.ms_sigma2_regional_prior_alpha = Rcpp::as<double>(gp["sigma2_regional_prior_alpha"]);

  } else if (gp_type == "hsgp") {
    data.spatial_type = SpatialType::HSGP;
    data.has_hsgp = true;
    const int m = Rcpp::as<int>(gp["hsgp_m"]);
    const double c = Rcpp::as<double>(gp["c_boundary"]);
    ratiod_hsgp::setup_hsgp_2d(coords, data.N, m, c, shared, data.hsgp_data);
    data.hsgp_m_per_dim = m;
    data.hsgp_boundary_factor = c;

  } else {
    Rcpp::stop("Unknown GP spatial type '%s'. Expected one of 'none', 'gp', "
               "'multiscale_gp', 'hsgp'.", gp_type);
  }
}

// Zero inflation, hurdle and one inflation
static void apply_zi_params(ModelData& data, const Rcpp::List& zi_params) {
  data.zi_type = ratiod_zi::parse_zi_type(Rcpp::as<std::string>(zi_params["type"]));

  // A design matrix, or none: an OI-only model passes a placeholder X with
  // p_zi = 0, so the width is read from p_zi, not from the matrix
  auto flatten = [&](SEXP x, int p, std::vector<double>& out) {
    out.assign(static_cast<size_t>(data.N) * p, 0.0);
    if (p == 0) return;
    if (Rf_isNull(x) || !Rf_isMatrix(x)) {
      Rcpp::stop("Inflation design matrix missing for %d coefficient(s).", p);
    }
    const Rcpp::NumericMatrix X(x);
    for (int i = 0; i < data.N; i++)
      for (int j = 0; j < p; j++) out[i * p + j] = X(i, j);
  };

  SEXP x_zi = zi_params["X"];
  SEXP p_zi = zi_params["p_zi"];
  data.p_zi = !Rf_isNull(p_zi) ? Rcpp::as<int>(p_zi)
            : (Rf_isMatrix(x_zi) ? Rcpp::NumericMatrix(x_zi).ncol() : 0);
  data.zi_prior_sd = Rcpp::as<double>(zi_params["prior_sd"]);
  flatten(x_zi, data.p_zi, data.X_zi_flat);

  SEXP p_oi = zi_params["p_oi"];
  data.p_oi = Rf_isNull(p_oi) ? 0 : Rcpp::as<int>(p_oi);
  SEXP oi_sd = zi_params["oi_prior_sd"];
  data.oi_prior_sd = Rf_isNull(oi_sd) ? data.zi_prior_sd : Rcpp::as<double>(oi_sd);
  flatten(zi_params["X_oi"], data.p_oi, data.X_oi_flat);
}

static void apply_prior_params(ModelData& data, const Rcpp::List& prior_params) {
  data.sigma_beta = Rcpp::as<double>(prior_params["sigma_beta"]);
  data.sigma_re_scale = Rcpp::as<double>(prior_params["sigma_re_scale"]);
  data.phi_prior_shape = Rcpp::as<double>(prior_params["phi_shape"]);
  data.phi_prior_rate = Rcpp::as<double>(prior_params["phi_rate"]);
  data.spatial_sigma_prior_U = Rcpp::as<double>(prior_params["spatial_sigma_prior_U"]);
  data.spatial_sigma_prior_alpha = Rcpp::as<double>(prior_params["spatial_sigma_prior_alpha"]);
}

ModelData build_model_data(
    const Rcpp::IntegerVector& y_num,
    const Rcpp::IntegerVector& y_denom,
    const Rcpp::NumericVector& y_num_cont,
    const Rcpp::NumericVector& y_denom_cont,
    const Rcpp::NumericMatrix& X_num,
    const Rcpp::NumericMatrix& X_denom,
    const Rcpp::List& family,
    const Rcpp::List& re_params,
    const Rcpp::List& spatial_params,
    const Rcpp::List& gp_params,
    const Rcpp::List& temporal_params,
    const Rcpp::List& prior_params,
    const Rcpp::List& zi_params,
    const Rcpp::List& latent_params,
    const Rcpp::List& st_params,
    const Rcpp::List& tvc_params,
    const Rcpp::List& svc_params
) {
  ModelData data;
  apply_family_params(data, family);
  apply_response_data(data, y_num, y_denom, y_num_cont, y_denom_cont,
                      X_num, X_denom);
  apply_re_params(data, re_params);
  apply_spatial_params(data, spatial_params);
  apply_gp_params(data, gp_params);
  apply_temporal_params(data, temporal_params);
  apply_zi_params(data, zi_params);
  apply_prior_params(data, prior_params);
  apply_latent_params(data, latent_params);
  apply_st_params(data, st_params);
  apply_tvc_params(data, tvc_params);
  apply_svc_params(data, svc_params);
  return data;
}

} // namespace ratiod_hmc

// [[Rcpp::export]]
Rcpp::List cpp_hmc_fit(
    Rcpp::NumericVector q_init,
    Rcpp::IntegerVector y_num,
    Rcpp::IntegerVector y_denom,
    Rcpp::NumericVector y_num_cont,
    Rcpp::NumericVector y_denom_cont,
    Rcpp::NumericMatrix X_num,
    Rcpp::NumericMatrix X_denom,
    Rcpp::List family,
    Rcpp::List re_params,
    Rcpp::List spatial_params,
    Rcpp::List gp_params,
    Rcpp::List temporal_params,
    Rcpp::List prior_params,
    Rcpp::List zi_params,
    Rcpp::List latent_params,
    Rcpp::List st_params,
    Rcpp::List tvc_params,  // Time-varying coefficients
    Rcpp::List svc_params,  // Spatially-varying coefficients
    int n_iter,
    int n_warmup,
    int L,
    int n_chains,
    unsigned int seed,
    int n_threads,
    bool verbose,
    std::string gradient_mode_str = "auto",
    int max_treedepth = 10,
    std::string metric_str = "auto",
    double adapt_delta = -1.0,
    bool walnuts = false,
    int n_cores = 0
) {
  using namespace ratiod_hmc;

  // Set global gradient mode from R parameter
  GradientMode grad_mode = parse_gradient_mode(gradient_mode_str);
  set_gradient_mode(grad_mode);

  const tulpa::WalnutsConfig walnuts_cfg;
  const tulpa::WalnutsConfig* walnuts_ptr = walnuts ? &walnuts_cfg : nullptr;

  // Parse metric type
  MassMatrixType metric_type = parse_metric_type(metric_str);

  ModelData data = build_model_data(y_num, y_denom, y_num_cont, y_denom_cont,
                                    X_num, X_denom, family, re_params,
                                    spatial_params, gp_params, temporal_params,
                                    prior_params, zi_params, latent_params,
                                    st_params, tvc_params, svc_params);

  if (verbose && data.has_multiscale_gp) {
    Rcpp::Rcout << "  HSGP-MSGP: m_total=" << data.msgp_hsgp_data.m_total
                << " (local+regional: " << 2 * data.msgp_hsgp_data.m_total
                << " basis coefficients)\n";
    Rcpp::Rcout << "  Lengthscale priors: local LogN(" << data.ms_log_ls_local_mean
                << ", " << data.ms_log_ls_local_sd << "), regional LogN("
                << data.ms_log_ls_regional_mean << ", " << data.ms_log_ls_regional_sd << ")\n";
  }

  // Parallelization. The OpenMP regions inside a gradient read the
  // nthreads-var, which the chain team's workers inherit, so it is scoped to
  // this fit's within-chain budget; left at the process default every
  // gradient evaluation would start a team as wide as the machine.
  data.n_threads = n_threads;
  ratiod_omp::ScopedThreadCount omp_scope(n_threads);
  // Across-chain team bound. n_cores == 0 means "size the team to the chain
  // count"; a positive value (the R backend passes the core budget) keeps the
  // multi-chain team the same size across fits.
  data.n_cores = (n_cores > 0) ? n_cores : n_chains;

  // Initialize parameters
  std::vector<double> q0(q_init.begin(), q_init.end());

  // Memory barrier to ensure all copies complete before HMC execution
  // This prevents R GC from invalidating memory during sampling
  std::atomic_thread_fence(std::memory_order_seq_cst);

  const ParamLayout layout = compute_param_layout(data);
  if (q0.size() != static_cast<size_t>(layout.total_params))
    Rcpp::stop("Initial parameter vector length %d does not match layout size %d",
               static_cast<int>(q0.size()), layout.total_params);
  require_gradient_supported(data, layout);

  // Run sampler
  std::vector<HMCResult> results;
  if (n_chains == 1) {
    results.push_back(run_hmc_chain(
      q0, data, layout, n_iter, n_warmup, L, 0, seed, verbose, max_treedepth,
      metric_type, adapt_delta, walnuts_ptr
    ));
  } else {
    results = run_hmc_parallel_chains(
      q0, data, n_iter, n_warmup, L, n_chains, seed, verbose, max_treedepth,
      metric_type, adapt_delta, walnuts_ptr
    );
  }
  return hmc_results_list(results, L);
}

// [[Rcpp::export]]
int cpp_get_max_threads() {
  #ifdef _OPENMP
  return omp_get_max_threads();
  #else
  return 1;
  #endif
}

// [[Rcpp::export]]
Rcpp::NumericMatrix cpp_test_spatial_covariance(Rcpp::NumericVector distance,
    double variance, double range, std::string covariance, double nu = 1.5) {
    const auto type = ratiod_cov::parse_cov_type(covariance, nu);
    Rcpp::NumericMatrix out(distance.size(), 2);
    for (R_xlen_t i = 0; i < distance.size(); ++i) {
        out(i, 0) = ratiod_cov::compute_cov(distance[i], variance, range, type);
        out(i, 1) = ratiod_cov::dcov_dphi(distance[i], variance, range, out(i, 0), type);
    }
    return out;
}
