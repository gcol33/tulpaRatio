#include <RcppEigen.h>
#include "spatial_model_data.h"
#include <functional>
#include <memory>
#include <vector>
#include <string>
#include <tulpa/nuts_api.h>
#include <tulpa/field_setup_api.h>
#include <tulpa/field_range.h>
#include <tulpa/st_hsgp_kernel.h>
#include <tulpa/st_margins.h>
#include <tulpa/pg_api.h>
#include "pg_engine.h"
#include <tulpa/likelihood.h>
#include <tulpa/model_data.h>
#include <tulpa/param_layout.h>
#include <tulpa/field_coordinates.h>

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
    bool walnuts = false,
    Rcpp::Nullable<Rcpp::List> svc_spec = R_NilValue,
    Rcpp::Nullable<Rcpp::List> tvc_spec = R_NilValue,
    Rcpp::Nullable<Rcpp::List> latent_spec = R_NilValue,
    Rcpp::Nullable<Rcpp::List> st_spec = R_NilValue
) {
  const Rcpp::List svc_params = svc_spec.isNotNull() ? Rcpp::List(svc_spec) : Rcpp::List::create();
  const Rcpp::List tvc_params = tvc_spec.isNotNull() ? Rcpp::List(tvc_spec) : Rcpp::List::create();
  const Rcpp::List latent_params = latent_spec.isNotNull() ? Rcpp::List(latent_spec) : Rcpp::List::create();
  const Rcpp::List st_params = st_spec.isNotNull() ? Rcpp::List(st_spec) : Rcpp::List::create();
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
  if (cfg_list.containsElementNamed("zi_link")) cfg.zi_link = Rcpp::as<std::string>(cfg_list["zi_link"]);
  if (cfg_list.containsElementNamed("oi_link")) cfg.oi_link = Rcpp::as<std::string>(cfg_list["oi_link"]);
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
  if (cfg_list.containsElementNamed("exposure")) {
    response->exposure = Rcpp::as<std::vector<double>>(cfg_list["exposure"]);
    if (response->exposure.size() != static_cast<size_t>(N)) Rcpp::stop("Exposure length must equal N");
    for (double exposure : response->exposure)
      if (!(exposure > 0) || !std::isfinite(exposure)) Rcpp::stop("Exposures must be positive and finite");
  }
  ratiod_link::parse_link(cfg.num_link, response->num_link);
  ratiod_link::parse_link(cfg.denom_link, response->denom_link);
  ratiod_link::parse_link(cfg.zi_link, response->zi_link);
  ratiod_link::parse_link(cfg.oi_link, response->oi_link);

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
  }
  data.sharing.init(spec.n_processes);

  data.zi_type     = zi_type;
  data.p_zi        = 0;
  data.p_oi        = 0;
  data.zi_prior_sd = cfg_list.containsElementNamed("zi_prior_sd")
      ? Rcpp::as<double>(cfg_list["zi_prior_sd"]) : 10.0;
  data.oi_prior_sd = cfg_list.containsElementNamed("oi_prior_sd")
      ? Rcpp::as<double>(cfg_list["oi_prior_sd"]) : data.zi_prior_sd;

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
  }

  if (re_params.size() > 0) {
    ratiod_hmc::apply_re_params(data,re_params);
    if (data.n_re_groups > 0 || data.n_re_terms > 0) spec.gradient_fn=nullptr;
  }

  const int p_spatial_block = tulpaRatio::apply_spatial_model_data(data, spatial_params);
  if (p_spatial_block > 0) spec.gradient_fn = nullptr;

  if (temporal_params.containsElementNamed("type")) {
    const std::string temporal_type =
        Rcpp::as<std::string>(temporal_params["type"]);
    if (temporal_type != "none" && !temporal_type.empty()) {
      if (temporal_type == "rw1") {
        data.temporal_type = tulpa::TemporalType::RW1;
      } else if (temporal_type == "rw2") {
        data.temporal_type = tulpa::TemporalType::RW2;
      } else if (temporal_type == "gp") {
        data.temporal_type = tulpa::TemporalType::GP;
        data.has_temporal_gp = true;
      } else if (temporal_type == "multiscale") {
        data.temporal_type = tulpa::TemporalType::NONE;
        data.has_multiscale_temporal = true;
      } else if (temporal_type == "iid") {
        data.temporal_type = tulpa::TemporalType::IID;
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
      if (temporal_params.containsElementNamed("parameterization")) {
        const auto parameterization = Rcpp::as<std::string>(temporal_params["parameterization"]);
        data.auto_temporal_coordinates = parameterization == "auto";
        data.temporal_parameterization = parameterization == "noncentered" ? 1 : 0;
      }
      data.temporal_group_centering = true;
      if (temporal_params.containsElementNamed("rtr")) {
        const Rcpp::List restriction = temporal_params["rtr"];
        if (restriction.containsElementNamed("active") && Rcpp::as<bool>(restriction["active"])) {
          data.set_temporal_restriction(
              Rcpp::as<std::vector<double>>(restriction["basis"]),
              Rcpp::as<int>(restriction["rank"]));
        }
      }
      if (temporal_type == "gp") {
        auto& gp = data.temporal_gp_data;
        gp.n_obs = data.n_times;
        gp.n_groups = data.n_temporal_groups;
        gp.time_values = Rcpp::as<std::vector<double>>(temporal_params["time_values"]);
        gp.group_index = data.temporal_group_idx;
        const std::string covariance = Rcpp::as<std::string>(temporal_params["cov_type"]);
        if (covariance == "exponential") gp.cov_type = tulpa::TemporalCovType::EXPONENTIAL;
        else if (covariance == "matern") gp.cov_type = tulpa::TemporalCovType::MATERN;
        else if (covariance == "gaussian") gp.cov_type = tulpa::TemporalCovType::GAUSSIAN;
        else if (covariance == "periodic") gp.cov_type = tulpa::TemporalCovType::PERIODIC;
        else Rcpp::stop("Unknown temporal GP covariance");
        gp.nu = Rcpp::as<double>(temporal_params["nu"]);
        gp.period = Rcpp::as<double>(temporal_params["period"]);
        gp.shared = shared;
        data.temporal_gp_relative_nugget = 1e-6;
        data.temporal_gp_parameterization =
            Rcpp::as<std::string>(temporal_params["gp_parameterization"]) == "centered" ? 0 : 1;
        data.auto_temporal_gp_coordinates = data.auto_temporal_coordinates;
        data.auto_temporal_coordinates = false;
        data.temporal_gp_sigma2_prior_U = Rcpp::as<double>(temporal_params["gp_sigma2_prior_U"]);
        data.temporal_gp_sigma2_prior_alpha = Rcpp::as<double>(temporal_params["gp_sigma2_prior_alpha"]);
        data.temporal_gp_phi_prior_lower = Rcpp::as<double>(temporal_params["gp_phi_prior_lower"]);
        data.temporal_gp_phi_prior_upper = Rcpp::as<double>(temporal_params["gp_phi_prior_upper"]);
        if (gp.cov_type == tulpa::TemporalCovType::GAUSSIAN) {
          data.temporal_gp_phi_prior_lower *= std::sqrt(2.0);
          data.temporal_gp_phi_prior_upper *= std::sqrt(2.0);
        }
        if (gp.time_values.size() != static_cast<size_t>(data.n_times))
          Rcpp::stop("temporal GP needs one time value per distinct time");
      }
      if (temporal_type == "multiscale") {
        auto& ms = data.multiscale_temporal_data;
        ms.n_obs = N;
        ms.n_times = data.n_times;
        ms.n_groups = data.n_temporal_groups;
        ms.time_index = data.temporal_time_idx;
        ms.group_index = data.temporal_group_idx;
        ms.shared = shared;
        const auto parse = [](const std::string& type) {
          if (type == "none") return tulpa::TemporalType::NONE;
          if (type == "rw1") return tulpa::TemporalType::RW1;
          if (type == "rw2") return tulpa::TemporalType::RW2;
          if (type == "ar1") return tulpa::TemporalType::AR1;
          if (type == "iid") return tulpa::TemporalType::IID;
          Rcpp::stop("Unknown multiscale temporal component");
          return tulpa::TemporalType::NONE;
        };
        ms.trend_type = parse(Rcpp::as<std::string>(temporal_params["trend_type"]));
        ms.short_term_type = parse(Rcpp::as<std::string>(temporal_params["short_term_type"]));
        ms.seasonal_period = Rcpp::as<int>(temporal_params["seasonal_period"]);
        data.ms_sigma2_trend_prior_U = Rcpp::as<double>(temporal_params["sigma2_trend_prior_U"]);
        data.ms_sigma2_trend_prior_alpha = Rcpp::as<double>(temporal_params["sigma2_trend_prior_alpha"]);
        data.ms_sigma2_seasonal_prior_U = Rcpp::as<double>(temporal_params["sigma2_seasonal_prior_U"]);
        data.ms_sigma2_seasonal_prior_alpha = Rcpp::as<double>(temporal_params["sigma2_seasonal_prior_alpha"]);
        data.ms_sigma2_short_prior_U = Rcpp::as<double>(temporal_params["sigma2_short_prior_U"]);
        data.ms_sigma2_short_prior_alpha = Rcpp::as<double>(temporal_params["sigma2_short_prior_alpha"]);
        const std::string coordinate = Rcpp::as<std::string>(temporal_params["ms_parameterization"]);
        const bool nc = coordinate != "centered";
        data.auto_multiscale_coordinates = coordinate == "auto";
        data.ms_trend_parameterization = nc && ms.trend_type != tulpa::TemporalType::NONE;
        data.ms_seasonal_parameterization = nc && ms.seasonal_period > 0;
        data.ms_short_parameterization = nc && ms.short_term_type != tulpa::TemporalType::NONE;
        tulpa::prepare_field(data, tulpa::FieldSetup::MULTISCALE_TEMPORAL);
      }
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

      spec.gradient_fn = nullptr;
    }
  }

  tulpa::prepare_field(data, tulpa::FieldSetup::TEMPORAL_RESTRICTION);
  const auto share = [&](std::vector<bool>& mask, bool shared) {
    for (int p = 1; p < data.n_processes; ++p) mask[p] = shared;
  };
  if (svc_params.containsElementNamed("has_svc") && Rcpp::as<bool>(svc_params["has_svc"])) {
    auto& svc = data.svc_data;
    data.has_svc = true;
    svc.n_obs = N;
    svc.n_svc = Rcpp::as<int>(svc_params["n_svc"]);
    svc.nn = Rcpp::as<int>(svc_params["nn"]);
    svc.coords = Rcpp::as<std::vector<double>>(svc_params["coords"]);
    svc.svc_indices = Rcpp::as<std::vector<int>>(svc_params["svc_indices"]);
    svc.X_svc = Rcpp::as<std::vector<double>>(svc_params["X_svc"]);
    svc.shared = Rcpp::as<bool>(svc_params["shared"]);
    svc.neighbor_jitter = 1e-8;
    svc.marginal_nugget = 0.0;
    svc.variance_floor = 1e-10;
    svc.blend_variance_floor = false;
    share(data.sharing.svc, svc.shared);
    data.svc_sigma_prior_U = Rcpp::as<double>(svc_params["sigma_prior_U"]);
    data.svc_sigma_prior_alpha = Rcpp::as<double>(svc_params["sigma_prior_alpha"]);
    data.svc_is_hsgp = Rcpp::as<std::string>(svc_params["svc_approx"]) == "hsgp";
    if (data.svc_is_hsgp) {
      data.svc_range_prior = 2;
      data.svc_hsgp_m_per_dim = Rcpp::as<int>(svc_params["hsgp_m"]);
      data.svc_hsgp_boundary_factor = Rcpp::as<double>(svc_params["c_boundary"]);
      tulpa::prepare_field(data, tulpa::FieldSetup::SVC_HSGP);
    } else {
      data.svc_range_prior = 1;
      data.svc_range_lower = Rcpp::as<double>(svc_params["phi_prior_lower"]);
      data.svc_range_upper = Rcpp::as<double>(svc_params["phi_prior_upper"]);
      const std::string covariance = Rcpp::as<std::string>(svc_params["cov_type"]);
      if (covariance == "exponential") svc.cov_type = tulpa::CovType::EXPONENTIAL;
      else if (covariance == "matern") svc.cov_type = tulpa::CovType::MATERN32;
      else if (covariance == "gaussian") svc.cov_type = tulpa::CovType::GAUSSIAN;
      else if (covariance == "spherical") svc.cov_type = tulpa::CovType::SPHERICAL;
      else Rcpp::stop("Unknown SVC covariance");
      if (svc.cov_type == tulpa::CovType::GAUSSIAN) {
        data.svc_range_lower *= std::sqrt(2.0);
        data.svc_range_upper *= std::sqrt(2.0);
      }
      svc.nn_idx = Rcpp::as<std::vector<int>>(svc_params["nn_idx"]);
      svc.nn_dist = Rcpp::as<std::vector<double>>(svc_params["nn_dist"]);
      svc.nn_order = Rcpp::as<std::vector<int>>(svc_params["nn_order"]);
      svc.nn_order_inv = Rcpp::as<std::vector<int>>(svc_params["nn_order_inv"]);
      const std::string coordinate = Rcpp::as<std::string>(svc_params["parameterization"]);
      data.svc_parameterization = coordinate == "centered" ? 0 : 1;
      data.auto_svc_coordinates = coordinate == "auto";
    }
    spec.gradient_fn = nullptr;
  }
  if (tvc_params.containsElementNamed("has_tvc") && Rcpp::as<bool>(tvc_params["has_tvc"])) {
    auto& tvc = data.tvc_data;
    data.has_tvc = true;
    tvc.n_obs = N;
    tvc.n_tvc = Rcpp::as<int>(tvc_params["n_tvc"]);
    tvc.n_times = Rcpp::as<int>(tvc_params["n_times"]);
    tvc.n_groups = Rcpp::as<int>(tvc_params["n_groups"]);
    tvc.time_index = Rcpp::as<std::vector<int>>(tvc_params["time_index"]);
    tvc.group_index = Rcpp::as<std::vector<int>>(tvc_params["group_index"]);
    tvc.tvc_indices = Rcpp::as<std::vector<int>>(tvc_params["tvc_indices"]);
    tvc.X_tvc = Rcpp::as<std::vector<double>>(tvc_params["X_tvc"]);
    tvc.shared = Rcpp::as<bool>(tvc_params["shared"]);
    tvc.cyclic = Rcpp::as<bool>(tvc_params["cyclic"]);
    const std::string structure = Rcpp::as<std::string>(tvc_params["structure"]);
    if (structure == "rw1") tvc.structure = tulpa::TemporalType::RW1;
    else if (structure == "rw2") tvc.structure = tulpa::TemporalType::RW2;
    else if (structure == "ar1") tvc.structure = tulpa::TemporalType::AR1;
    else if (structure == "iid") tvc.structure = tulpa::TemporalType::IID;
    else Rcpp::stop("Unsupported TVC structure");
    data.tvc_sigma_prior_U = Rcpp::as<double>(tvc_params["sigma_prior_U"]);
    data.tvc_sigma_prior_alpha = Rcpp::as<double>(tvc_params["sigma_prior_alpha"]);
    data.tvc_rho_prior_a = 2.0;
    data.tvc_rho_prior_b = 2.0;
    share(data.sharing.tvc, tvc.shared);
    spec.gradient_fn = nullptr;
  }
  if (latent_params.containsElementNamed("has_latent") && Rcpp::as<bool>(latent_params["has_latent"])) {
    data.has_latent = true;
    data.latent_bilinear = true;
    data.latent_n_factors = Rcpp::as<int>(latent_params["n_factors"]);
    data.latent_shared = Rcpp::as<bool>(latent_params["shared"]);
    data.latent_loading_processes = data.latent_shared ? data.n_processes : 1;
    data.latent_constraint = Rcpp::as<int>(latent_params["constraint"]);
    data.latent_sigma_prior_rate = Rcpp::as<double>(latent_params["sigma_prior_rate"]);
    data.latent_offdiag_sd = Rcpp::as<double>(latent_params["offdiag_sd"]);
    spec.gradient_fn = nullptr;
  }

  if (st_params.containsElementNamed("has_spatiotemporal") && Rcpp::as<bool>(st_params["has_spatiotemporal"])) {
    auto& st = data.spatiotemporal_data;
    data.has_spatiotemporal = true;
    data.st_is_hsgp = Rcpp::as<bool>(st_params["st_is_hsgp"]);
    const std::string type = Rcpp::as<std::string>(st_params["type"]);
    if (type == "I") st.type = tulpa::STType::TYPE_I;
    else if (type == "II") st.type = tulpa::STType::TYPE_II;
    else if (type == "III") st.type = tulpa::STType::TYPE_III;
    else if (type == "IV") st.type = tulpa::STType::TYPE_IV;
    else if (type == "separable") st.type = tulpa::STType::SEPARABLE;
    else if (type == "nonsep_gp") st.type = tulpa::STType::NONSEP_GP;
    else Rcpp::stop("Unsupported spatiotemporal bridge type");
    const bool gp = st.type == tulpa::STType::SEPARABLE || st.type == tulpa::STType::NONSEP_GP;
    data.st_margin_centering = !gp && !data.st_is_hsgp;
    const std::string temporal = Rcpp::as<std::string>(st_params["temporal_type"]);
    if (!data.st_is_hsgp && !tulpa_st::st_reads_time_margin(st.type))
      st.temporal_type = tulpa::TemporalType::NONE;
    else if (temporal == "rw1") st.temporal_type = tulpa::TemporalType::RW1;
    else if (temporal == "rw2") st.temporal_type = tulpa::TemporalType::RW2;
    else if (temporal == "ar1") st.temporal_type = tulpa::TemporalType::AR1;
    else Rcpp::stop("Unsupported spatiotemporal time margin");
    st.n_spatial = Rcpp::as<int>(st_params["n_spatial"]);
    st.n_times = Rcpp::as<int>(st_params["n_times"]);
    st.n_params = Rcpp::as<int>(st_params["n_params"]);
    st.st_flat = Rcpp::as<std::vector<int>>(st_params["st_flat"]);
    st.s_idx = Rcpp::as<std::vector<int>>(st_params["s_idx"]);
    st.t_idx = Rcpp::as<std::vector<int>>(st_params["t_idx"]);
    st.shared = Rcpp::as<bool>(st_params["shared"]);
    st.temporal_cyclic = Rcpp::as<bool>(st_params["temporal_cyclic"]);
    st.adj_row_ptr = Rcpp::as<std::vector<int>>(st_params["adj_row_ptr"]);
    st.adj_col_idx = Rcpp::as<std::vector<int>>(st_params["adj_col_idx"]);
    if (st.adj_row_ptr.size() == static_cast<size_t>(st.n_spatial + 1)) {
      st.n_neighbors.resize(st.n_spatial);
      for (int s = 0; s < st.n_spatial; ++s)
        st.n_neighbors[s] = st.adj_row_ptr[s + 1] - st.adj_row_ptr[s];
    }
    if (gp) {
      data.st_uniform_ranges = true;
      st.nn = Rcpp::as<int>(st_params["nn"]);
      st.coords = Rcpp::as<std::vector<double>>(st_params["gp_coords"]);
      st.time_values = Rcpp::as<std::vector<double>>(st_params["gp_time_values"]);
      st.nn_idx = Rcpp::as<std::vector<int>>(st_params["nn_idx"]);
      st.nn_dist_space = Rcpp::as<std::vector<double>>(st_params["nn_dist_space"]);
      st.nn_dist_time = Rcpp::as<std::vector<double>>(st_params["nn_dist_time"]);
      st.nn_order = Rcpp::as<std::vector<int>>(st_params["nn_order"]);
      st.nn_order_inv = Rcpp::as<std::vector<int>>(st_params["nn_order_inv"]);
      for (int& index : st.nn_order) --index;
      for (int& index : st.nn_order_inv) --index;
      st.cov_space = static_cast<tulpa::CovType>(Rcpp::as<int>(st_params["cov_space"]));
      st.cov_time = static_cast<tulpa::CovType>(Rcpp::as<int>(st_params["cov_time"]));
      const std::string nonsep = Rcpp::as<std::string>(st_params["nonsep_type"]);
      st.nonsep_type = nonsep == "gneiting" ? tulpa::NonsepType::GNEITING : tulpa::NonsepType::PRODUCT;
      st.phi_space_prior_lower = Rcpp::as<double>(st_params["phi_space_prior_lower"]);
      st.phi_space_prior_upper = Rcpp::as<double>(st_params["phi_space_prior_upper"]);
      st.phi_time_prior_lower = Rcpp::as<double>(st_params["phi_time_prior_lower"]);
      st.phi_time_prior_upper = Rcpp::as<double>(st_params["phi_time_prior_upper"]);
    }
    if (data.st_is_hsgp) {
      data.st_hsgp_coords = Rcpp::as<std::vector<double>>(st_params["hsgp_coords"]);
      data.st_hsgp_m_per_dim = Rcpp::as<int>(st_params["hsgp_m"]);
      data.st_hsgp_boundary_factor = Rcpp::as<double>(st_params["c_boundary"]);
      data.st_hsgp_sigma2_prior_U = Rcpp::as<double>(st_params["sigma2_prior_U"]);
      data.st_hsgp_sigma2_prior_alpha = Rcpp::as<double>(st_params["sigma2_prior_alpha"]);
      tulpa::prepare_field(data, tulpa::FieldSetup::ST_HSGP);
    }
    data.st_parameterization = Rcpp::as<std::string>(st_params["parameterization"]) == "noncentered" ? 1 : 0;
    if (data.st_margin_centering) tulpa::setup_st_spatial_partition(st);
    data.st_sigma2_prior_U = Rcpp::as<double>(st_params["sigma2_prior_U"]);
    data.st_sigma2_prior_alpha = Rcpp::as<double>(st_params["sigma2_prior_alpha"]);
    data.st_rho_prior_a = Rcpp::as<double>(st_params["rho_prior_a"]);
    data.st_rho_prior_b = Rcpp::as<double>(st_params["rho_prior_b"]);
    share(data.sharing.st, st.shared);
    spec.gradient_fn = nullptr;
  }

  tulpa::ParamLayout layout = tulpa::compute_layout(data);

  if (init.size() != layout.total_params) {
    Rcpp::stop("cpp_tulpaRatio_run_nuts_specs: init length (%d) != engine layout (%d) for family='%s'.",
               static_cast<int>(init.size()), layout.total_params, cfg.family.c_str());
  }

  std::vector<double> initial(init.begin(), init.end());
  if (data.st_is_hsgp && data.st_parameterization == 1) {
    const double rho = layout.logit_rho_st_idx < 0 ? 0.0 :
      2.0 * tulpa::math::inv_logit(initial[layout.logit_rho_st_idx]) - 1.0;
    tulpa_st_hsgp::nc_inverse(initial.data() + layout.st_delta_start,
      tulpa::math::safe_exp(initial[layout.log_tau_st_idx]), rho,
      tulpa::math::safe_exp(initial[layout.log_sigma2_st_hsgp_idx]), tulpa::math::safe_exp(initial[layout.log_lengthscale_st_hsgp_idx]),
      data.spatiotemporal_data, data.st_hsgp_data, initial.data() + layout.st_delta_start);
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
  for (int c=0;c<n_chains;++c) init_vec.insert(init_vec.end(),initial.begin(),initial.end());
  if (cfg_list.containsElementNamed("initialize") && Rcpp::as<bool>(cfg_list["initialize"]) &&
      data.has_gp && data.gp_range_prior.type == tulpa::RangePriorType::UNIFORM) {
    const tulpa::LogRangeCoordinate range{layout.log_phi_gp_idx,
        data.gp_range_prior.lower, data.gp_range_prior.upper};
    for (int c = 0; c < n_chains; ++c) {
      double& value = init_vec[c * layout.total_params + range.index];
      value = range.interior(value);
    }
  }
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
  Rcpp::List spatial_fields(n_chains), unstructured_fields(n_chains);
  for (int c=0;c<n_chains;++c) {
    auto& result=results[c];
    const int n=result.n_sample;
    auto draws = row_major_matrix(result.samples,n,result.n_params);
    const auto restore_range = [&](int index, double lower, double upper, double scale) {
      for (int i = 0; i < n; ++i)
        draws(i,index) = std::log(tulpa::field_range_from_coordinate(draws(i,index), lower, upper, true) / scale);
    };
    if (data.has_svc && data.svc_range_prior == 1)
      for (int j = 0; j < data.svc_data.n_svc; ++j) {
        const int index = layout.log_phi_svc_start + j;
        const double scale = data.svc_data.cov_type == tulpa::CovType::GAUSSIAN ? std::sqrt(2.0) : 1.0;
        restore_range(index, data.svc_range_lower, data.svc_range_upper, scale);
      }
    if (layout.is_st_gp && data.st_uniform_ranges) {
      const auto& st = data.spatiotemporal_data;
      restore_range(layout.log_phi_st_space_idx, st.phi_space_prior_lower, st.phi_space_prior_upper, 1.0);
      restore_range(layout.log_phi_st_time_idx, st.phi_time_prior_lower, st.phi_time_prior_upper, 1.0);
    }
    samples[c]=draws;
    log_prob[c]=buffer_vector(result.log_prob,n);
    accept_prob[c]=buffer_vector(result.accept_prob,n);
    divergent[c]=buffer_int_vector(result.divergent,n);
    treedepth[c]=buffer_int_vector(result.treedepth,n);
    n_leapfrog[c]=buffer_int_vector(result.n_leapfrog,n);
    energy[c]=buffer_vector(result.energy,n);
    const std::string sampler(result.sampler);
    samplers[c]=sampler == "nuts" ? "NUTS" : sampler;
    epsilon[c]=result.epsilon;
    if (layout.is_icar_collapsed || layout.is_bym2_collapsed || layout.is_gp_collapsed) {
      const int units = layout.is_gp ? data.gp_data.n_obs : data.n_spatial_units;
      if (result.n_spatial_fields != units || (n > 0 && (!result.spatial_draws ||
          (layout.is_bym2_collapsed && !result.unstructured_draws))))
        Rcpp::stop("The engine must return conditional spatial fields for every retained draw");
      spatial_fields[c] = row_major_matrix(result.spatial_draws, n, units);
      if (layout.is_bym2_collapsed)
        unstructured_fields[c] = row_major_matrix(result.unstructured_draws, n, units);
    }
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
    Rcpp::Named("sampler")=Rcpp::as<std::string>(samplers[0]),
    Rcpp::Named("icar_phi_star")=(layout.is_icar_collapsed || layout.is_bym2_collapsed) ? pack(spatial_fields) : R_NilValue,
    Rcpp::Named("bym2_theta_star")=layout.is_bym2_collapsed ? pack(unstructured_fields) : R_NilValue,
    Rcpp::Named("gp_w_star")=layout.is_gp_collapsed ? pack(spatial_fields) : R_NilValue);
}
