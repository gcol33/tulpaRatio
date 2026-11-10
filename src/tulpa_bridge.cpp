#include <Rcpp.h>
#include <functional>
#include <memory>
#include <vector>
#include <string>
#include <tulpa/nuts_api.h>
#include <tulpa/pg_api.h>
#include "pg_engine.h"
#include <tulpa/likelihood.h>
#include <tulpa/model_data.h>
#include <tulpa/param_layout.h>

#include "lik_specs/family_registry.h"
#include "lik_specs/lik_helpers.h"
#include "re_model_data.h"

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

std::unique_ptr<tulpaRatio::lik::ResponseData> build_response_payload(
    const std::string& family, const Rcpp::IntegerVector& yn,
    const Rcpp::IntegerVector& yd, const Rcpp::NumericVector& cn,
    const Rcpp::NumericVector& cd, int N) {
  const auto& entry = tulpaRatio::family_entry(family);
  auto p = std::make_unique<tulpaRatio::lik::ResponseData>();
  const auto copy = [N](tulpaRatio::ResponseShape shape, const Rcpp::IntegerVector& counts,
      const Rcpp::NumericVector& continuous, std::vector<int>& ints, std::vector<double>& reals) {
    if (shape == tulpaRatio::ResponseShape::Positive || shape == tulpaRatio::ResponseShape::Exposure) {
      if (continuous.size() != N) Rcpp::stop("Continuous response length must equal N");
      reals.assign(continuous.begin(),continuous.end());
    } else {
      if (counts.size() != N) Rcpp::stop("Count response length must equal N");
      ints.assign(counts.begin(),counts.end());
    }
  };
  copy(entry.num_response,yn,cn,p->y_num,p->y_num_cont);
  copy(entry.denom_response,yd,cd,p->y_denom,p->y_denom_cont);
  return p;
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

// [[Rcpp::export]]
int cpp_tulpa_compiled_abi_version() {
  return tulpa::TULPA_ABI_VERSION;
}

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
  ratiod::PGEngineResult result;
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
    &result.value
  );

  return ratiod::pg_engine_fit(result.value, store_eta);
}

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
    Rcpp::List temporal_params = Rcpp::List::create(),
    int n_chains = 1,
    int n_threads = 1,
    int L = 0,
    std::string metric = "auto",
    bool walnuts = false
) {
  const int n_proc_input = X_list.size();
  if (n_proc_input < 1) {
    Rcpp::stop("cpp_tulpaRatio_run_nuts_specs: X_list must contain at least one "
               "design matrix.");
  }
  Rcpp::NumericMatrix X0 = X_list[0];
  const int N = X0.nrow();

  tulpaRatio::RatioConfig cfg;
  cfg.family    = Rcpp::as<std::string>(cfg_list["family"]);
  if (cfg_list.containsElementNamed("zi"))         cfg.zi         = Rcpp::as<std::string>(cfg_list["zi"]);
  if (cfg_list.containsElementNamed("num_link"))   cfg.num_link   = Rcpp::as<std::string>(cfg_list["num_link"]);
  if (cfg_list.containsElementNamed("denom_link")) cfg.denom_link = Rcpp::as<std::string>(cfg_list["denom_link"]);

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
  const bool has_zi_block = (zi_type != tulpa::ZIType::NONE) &&
                            (zi_type != tulpa::ZIType::OI_BINOMIAL);

  const bool has_y_int  = (y_num.size()  == N) && (y_denom.size()  == N);
  const bool has_y_cont = (y_num_cont.size() == N) || (y_denom_cont.size() == N);
  if (!has_y_int && !has_y_cont) {
    Rcpp::stop("cpp_tulpaRatio_run_nuts_specs: response vectors all empty.");
  }

  tulpa::LikelihoodSpec spec = tulpaRatio::build_ratio_likelihood_spec(cfg);
  auto response = build_response_payload(
      cfg.family, y_num, y_denom, y_num_cont, y_denom_cont, N);

  if (n_proc_input != spec.n_processes) {
    Rcpp::stop("cpp_tulpaRatio_run_nuts_specs: X_list has %d matrices but "
               "family='%s' expects %d processes.",
               n_proc_input, cfg.family.c_str(), spec.n_processes);
  }

  tulpa::ModelData data;
  data.N = N;
  data.n_processes = spec.n_processes;
  data.sigma_beta  = sigma_beta;
  data.model_response_data = response.get();
  data.likelihood_spec     = &spec;

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
    if (cfg_list.containsElementNamed("offsets")) {
      Rcpp::List offsets=cfg_list["offsets"];
      if (offsets.size() != spec.n_processes) Rcpp::stop("One offset is required per process");
      data.processes[k].offset=Rcpp::as<std::vector<double>>(offsets[k]);
      if (data.processes[k].offset.size() != static_cast<size_t>(N))
        Rcpp::stop("Process offset length must equal N");
    }
    total_p_design += Xk.ncol();
  }
  data.sharing.init(spec.n_processes);

  data.zi_type     = zi_type;
  data.p_zi        = 0;
  data.p_oi        = 0;
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

  int p_re_block=0;
  if (re_params.size() > 0) {
    ratiod_hmc::apply_re_params(data,re_params);
    p_re_block=data.total_sigma_params+data.total_chol_params+data.total_re_params;
    if (data.n_re_groups > 0 || data.n_re_terms > 0) spec.gradient_fn=nullptr;
  }

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
        data.auto_spatial_coordinates = param == "auto";
      } else if (spatial_type == "bym2") {
        data.spatial_type = tulpa::SpatialType::BYM2;
      } else {
        Rcpp::stop("cpp_tulpaRatio_run_nuts_specs: spatial type '%s' is not "
                   "yet supported on the spec path. Supported: 'icar', 'bym2'.",
                   spatial_type.c_str());
      }
      data.spatial_component_centering = true;

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
        std::vector<double> precision;
        if (spatial_params.containsElementNamed("node_prec") &&
            !Rf_isNull(spatial_params["node_prec"]))
          precision = Rcpp::as<std::vector<double>>(spatial_params["node_prec"]);
        data.set_bym2_scaling(Rcpp::as<double>(spatial_params["bym2_scale"]),
                             std::move(precision));
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

      spec.gradient_fn = nullptr;
    }
  }

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
      const bool shared = Rcpp::as<bool>(temporal_params["shared"]);
      for (int k = 1; k < data.n_processes; ++k) data.sharing.temporal[k] = shared;
      if (temporal_params.containsElementNamed("parameterization"))
        data.auto_temporal_coordinates =
            Rcpp::as<std::string>(temporal_params["parameterization"]) == "auto";
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

      spec.gradient_fn = nullptr;
    }
  }

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

  int grad_rc = tulpa::set_gradient_mode_str(gradient_mode.c_str());
  if (grad_rc != 0) {
    Rcpp::stop("cpp_tulpaRatio_run_nuts_specs: unrecognised gradient_mode '%s'.",
               gradient_mode.c_str());
  }

  if (n_chains < 1) Rcpp::stop("n_chains must be positive");
  tulpa::SamplerControl control;
  control.L = L;
  control.walnuts = walnuts;
  if (metric == "auto") control.metric = tulpa::MassMatrixType::AUTO;
  else if (metric == "diag") control.metric = tulpa::MassMatrixType::DIAG;
  else if (metric == "dense") control.metric = tulpa::MassMatrixType::DENSE;
  else if (metric == "block_diag") control.metric = tulpa::MassMatrixType::BLOCK_DIAG;
  else Rcpp::stop("Unknown metric '%s'", metric.c_str());
  tulpa::OmpCapScope thread_cap(n_threads);
  std::vector<double> init_vec;
  for (int c=0;c<n_chains;++c) init_vec.insert(init_vec.end(),init.begin(),init.end());
  std::vector<tulpa::NUTSResult> results(n_chains);
  if (n_chains == 1) {
    tulpa::get_nuts_fn()(&data,&layout,init_vec.data(),layout.total_params,
      n_iter,n_warmup,max_treedepth,adapt_delta,static_cast<unsigned int>(seed),
      verbose ? 1 : 0,&control,nullptr,0.0,results.data());
  } else {
    tulpa::get_nuts_chains_fn()(&data,&layout,init_vec.data(),layout.total_params,n_chains,
      n_iter,n_warmup,max_treedepth,adapt_delta,static_cast<unsigned int>(seed),
      verbose ? 1 : 0,&control,nullptr,nullptr,results.data());
  }
  Rcpp::List samples(n_chains),log_prob(n_chains),accept_prob(n_chains),
             divergent(n_chains),treedepth(n_chains),n_leapfrog(n_chains),energy(n_chains);
  Rcpp::NumericVector epsilon(n_chains);
  Rcpp::CharacterVector samplers(n_chains);
  for (int c=0;c<n_chains;++c) {
    auto& result=results[c];
    const int n=result.n_sample;
    samples[c]=row_major_matrix(result.samples,n,result.n_params);
    log_prob[c]=buffer_vector(result.log_prob,n);
    accept_prob[c]=buffer_vector(result.accept_prob,n);
    divergent[c]=buffer_int_vector(result.divergent,n);
    treedepth[c]=buffer_int_vector(result.treedepth,n);
    n_leapfrog[c]=buffer_int_vector(result.n_leapfrog,n);
    energy[c]=buffer_vector(result.energy,n);
    const std::string sampler(result.sampler);
    samplers[c]=sampler == "nuts" ? "NUTS" : sampler;
    epsilon[c]=result.epsilon;
    result.free_buffers();
  }
  const auto pack = [n_chains](const Rcpp::List& x) -> SEXP {
    return n_chains == 1 ? SEXP(x[0]) : SEXP(x);
  };
  return Rcpp::List::create(Rcpp::Named("samples")=pack(samples),
    Rcpp::Named("log_prob")=pack(log_prob),Rcpp::Named("accept_prob")=pack(accept_prob),
    Rcpp::Named("n_leapfrog")=pack(n_leapfrog),Rcpp::Named("treedepth")=pack(treedepth),
    Rcpp::Named("divergent")=pack(divergent),Rcpp::Named("epsilon")=epsilon,
    Rcpp::Named("energy")=pack(energy),
    Rcpp::Named("sampler")=Rcpp::as<std::string>(samplers[0]));
}
