#include "engine_fixture.h"
#include <tulpa/field_coordinates.h>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace {
using namespace engine_fixture;

struct GradientScope {
    explicit GradientScope(const std::string& mode) {
        if(tulpa::set_gradient_mode_str(mode.c_str()))
            Rcpp::stop("Unknown gradient mode '%s'",mode.c_str());
    }
    ~GradientScope() { tulpa::set_gradient_mode_str("auto"); }
};

double point_value(const std::vector<double>& q, const Fixture& data,
    const ParamLayout& layout, const double* anchor=nullptr) {
    double value=0;
    tulpa::ModelPointRequest request;
    request.position=q.data();request.n_params=q.size();request.value=&value;
    request.coordinate_anchor=anchor;
    tulpa::get_model_point_fn()(&data,&layout,&request);
    return value;
}

void point_gradient(const std::vector<double>& q,const Fixture& data,
    const ParamLayout& layout,std::vector<double>& gradient,double* value=nullptr,
    const double* anchor=nullptr) {
    gradient.resize(q.size());
    tulpa::ModelPointRequest request;
    request.position=q.data();request.n_params=q.size();request.gradient=gradient.data();
    request.value=value;request.coordinate_anchor=anchor;
    tulpa::get_model_point_fn()(&data,&layout,&request);
}

std::vector<double> stored_point(const std::vector<double>& q,const Fixture& data,
    const ParamLayout& layout) {
    std::vector<double> out(q.size());
    tulpa::ModelPointRequest request;
    request.position=q.data();request.n_params=q.size();request.stored=out.data();
    tulpa::get_model_point_fn()(&data,&layout,&request);
    return out;
}
}



// [[Rcpp::export]]
Rcpp::CharacterVector cpp_gradient_fields() {
  Rcpp::CharacterVector out(sizeof(KNOWN_FIELDS) / sizeof(KNOWN_FIELDS[0]));
  for (int i = 0; i < out.size(); i++) out[i] = KNOWN_FIELDS[i];
  return out;
}

// [[Rcpp::export]]
Rcpp::List cpp_gradient_chain_fixture(std::string field,std::string mode="H",
    unsigned int seed=17192,bool auto_coordinates=false) {
    auto data=make_model(field,20,4,4,seed,"binomial",true,"none");
    data.auto_spatial_coordinates=auto_coordinates;
    data.auto_temporal_coordinates=auto_coordinates;
    auto layout=fixture_layout(data);
    GradientScope gradient(mode);
    auto initial=draw_params(data,layout,layout.total_params,seed);
    tulpa::NUTSResult result{};
    tulpa::SamplerControl control;
    tulpa::get_nuts_fn()(&data,&layout,initial.data(),layout.total_params,
      160,120,4,-1,seed,0,&control,nullptr,0,&result);
    struct Cleanup { tulpa::NUTSResult& result; ~Cleanup(){result.free_buffers();} } cleanup{result};
    const auto vector=[](const double* p,int n) {
      return p ? Rcpp::NumericVector(p,p+n) : Rcpp::NumericVector();
    };
    return Rcpp::List::create(Rcpp::Named("samples")=vector(result.samples,result.n_sample*result.n_params),
      Rcpp::Named("log_prob")=vector(result.log_prob,result.n_sample),
      Rcpp::Named("energy")=vector(result.energy,result.n_sample),
      Rcpp::Named("accept_prob")=vector(result.accept_prob,result.n_sample),
      Rcpp::Named("epsilon")=result.epsilon);
}

// [[Rcpp::export]]
Rcpp::List cpp_scale_prior_check(std::string field, double x,
                                double U = 0.8, double alpha = 0.15) {
  Fixture data = make_model(field, 40, 8, 5, 42, "binomial", true, "none");
  const ParamLayout layout = fixture_layout(data);
  std::vector<double> params(layout.total_params, 0.0);
  std::vector<int> slots;
  std::vector<double> slopes;
  auto add = [&](int slot, double slope) {
    if (slot >= 0) { slots.push_back(slot); slopes.push_back(slope); params[slot] = x; }
  };
  if (layout.has_spatial) {
    if (layout.is_bym2) add(layout.log_sigma_bym2_idx, 1.0);
    else add(layout.log_tau_spatial_idx, -0.5);
  }
  if (layout.has_temporal) add(layout.log_tau_temporal_idx, -0.5);
  if (layout.has_tvc) {
    for (int j = layout.log_tau_tvc_start; j < layout.log_tau_tvc_end; ++j) add(j, -0.5);
  }
  if (layout.is_hsgp) add(layout.log_sigma2_hsgp_idx, 0.5);
  if (layout.has_svc) {
    for (int j = layout.log_sigma2_svc_start; j < layout.log_sigma2_svc_end; ++j) add(j, 0.5);
  }
  auto set = [](Fixture& model, double u, double a) {
    model.spatial_sigma_prior_U = model.temporal_sigma_prior_U = u;
    model.tvc_sigma_prior_U = model.svc_sigma_prior_U = u;
    model.gp_sigma2_prior_U = model.hsgp_sigma2_prior_U = u;
    model.spatial_sigma_prior_alpha = model.temporal_sigma_prior_alpha = a;
    model.tvc_sigma_prior_alpha = model.svc_sigma_prior_alpha = a;
    model.gp_sigma2_prior_alpha = model.hsgp_sigma2_prior_alpha = a;
  };
  set(data, 1.0, 0.01);
  const double base = point_value(params, data, layout);
  Fixture alternative = data;
  set(alternative, U, alpha);
  const auto alternative_layout = fixture_layout(alternative);
  if (alternative_layout.total_params != layout.total_params)
    Rcpp::stop("Prior anchors must preserve the parameter layout");
  const double changed = point_value(params, alternative, alternative_layout);
  Rcpp::NumericVector density(slopes.size()), gradient(slopes.size());
  for (size_t j = 0; j < slopes.size(); ++j) {
    density[j] = ratiod::math::pc_log_density(x, slopes[j], U, alpha);
    gradient[j] = ratiod::math::pc_scale_gradient(std::exp(slopes[j] * x), slopes[j], U, alpha);
  }
  return Rcpp::List::create(Rcpp::_["delta"] = changed - base,
      Rcpp::_["slopes"] = slopes, Rcpp::_["density"] = density,
      Rcpp::_["gradient"] = gradient);
}

// [[Rcpp::export]]
Rcpp::NumericVector cpp_scale_conditional_draws(int n, double quad, int rank,
                                               double U, double alpha) {
  Rcpp::NumericVector out(n);
  double tau = 1.0;
  for (int i = -1000; i < n; ++i) {
    tau = ratiod_scale::sample_tau(tau, quad, rank, U, alpha,
        []() { return R::unif_rand(); });
    if (i >= 0) out[i] = tau;
  }
  return out;
}

// [[Rcpp::export]]
Rcpp::List cpp_gradient_check(std::string field,
                              int n_obs = 400,
                              int n_units = 25,
                              int n_times = 10,
                              double eps = 1e-5,
                              unsigned int seed = 42,
                              std::string mode = "handcoded",
                              bool precenter = false,
                              std::string family = "binomial",
                              bool temporal_shared = true,
                              std::string zi = "none",
                              bool near_unit_rho = false,
                              int n_threads = 1,
                              std::string link_num = "",
                              std::string link_denom = "",
                              std::string link_inflation = "") {
  const bool auto_coordinates = field.size() > 5 && field.substr(field.size() - 5) == "_auto";
  if (auto_coordinates) field.resize(field.size() - 5);
  Fixture data = make_model(field, n_obs, n_units, n_times, seed,
                              family, temporal_shared, zi);
  data.auto_spatial_coordinates = auto_coordinates;
  data.auto_temporal_coordinates = auto_coordinates;
  data.auto_gp_coordinates = auto_coordinates;
  data.auto_multiscale_coordinates = auto_coordinates;
  data.auto_temporal_gp_coordinates = auto_coordinates;
  data.auto_svc_coordinates = auto_coordinates;
  if (auto_coordinates && data.has_svc && !data.svc_is_hsgp) data.svc_parameterization = 1;
  if (auto_coordinates && data.has_temporal_gp) data.temporal_gp_parameterization = 1;
  if (auto_coordinates && data.has_gp) data.gp_parameterization = 1;

  auto set_link = [](const std::string& name, LinkType& out) {
    if (!name.empty() && !ratiod_link::parse_link(name, out)) {
      Rcpp::stop("cpp_gradient_check: unknown link '" + name + "'");
    }
  };
  set_link(link_num, data.response.num_link);
  set_link(link_denom, data.response.denom_link);
  set_link(link_inflation, data.response.zi_link);
  set_link(link_inflation, data.response.oi_link);
  if(!link_num.empty()) data.config.num_link=link_num;
  if(!link_denom.empty()) data.config.denom_link=link_denom;
  if(!link_inflation.empty()) data.config.zi_link=data.config.oi_link=link_inflation;

  tulpa::OmpCapScope thread_cap(n_threads);
  ParamLayout layout = fixture_layout(data);
  const int n_params = layout.total_params;

  GradientScope gradient_scope(mode);

  std::vector<double> params = draw_params(data, layout, n_params, seed);

  if ((data.response.denom_link == LinkType::IDENTITY || data.response.denom_link == LinkType::INVERSE) &&
      data.processes.size() > 1 && data.processes[1].p > 0) {
    params[layout.process_beta_start[1]] = 5.0;
  }

  if (near_unit_rho) {

    const double logit_edge = 30.0;
    for (int idx : {layout.logit_rho_ar1_idx, layout.logit_rho_short_idx,
                    layout.logit_rho_st_idx}) {
      if (idx >= 0 && idx < n_params) params[idx] = logit_edge;
    }
    for (int j = layout.logit_rho_tvc_start; j >= 0 && j < layout.logit_rho_tvc_end; j++) {
      if (j < n_params) params[j] = logit_edge;
    }
  }

  if (precenter && layout.spatial_start >= 0) {
    double m = 0.0;
    for (int i = layout.spatial_start; i < layout.spatial_end; i++) m += params[i];
    m /= static_cast<double>(layout.spatial_end - layout.spatial_start);
    for (int i = layout.spatial_start; i < layout.spatial_end; i++) params[i] -= m;
  }

  const auto native=params;
  if(auto_coordinates) {
    tulpa::ModelPointRequest request;
    request.position=native.data();request.n_params=n_params;
    request.coordinate_anchor=native.data();request.sampling_position=params.data();
    tulpa::get_model_point_fn()(&data,&layout,&request);
  }
  const double* anchor=auto_coordinates ? native.data() : nullptr;
  std::vector<double> grad_analytic(n_params, 0.0);
  double lp_from_mode = 0.0;
  point_gradient(params, data, layout, grad_analytic, &lp_from_mode,anchor);

  auto lp_at = [&](const std::vector<double>& p) {
    return point_value(p, data, layout,anchor);
  };

  std::vector<double> grad_fd(n_params, 0.0);
  for (int i = 0; i < n_params; i++) {
    std::vector<double> up = params, dn = params;
    const double h = eps * std::max(1.0, std::fabs(params[i]));
    up[i] += h;
    dn[i] -= h;
    grad_fd[i] = (lp_at(up) - lp_at(dn)) / (2.0 * h);
  }

  Rcpp::CharacterVector block(n_params, "other");
  for (int i = 0; i < n_params; i++) {
    if (i >= layout.process_beta_start[0] && i < (layout.process_beta_start[0]+layout.process_beta_count[0])) block[i] = "beta_num";
    if (layout.beta_zi_start >= 0 && i >= layout.beta_zi_start && i < layout.beta_zi_end)
      block[i] = "beta_zi";
    if (layout.spatial_start >= 0 && i >= layout.spatial_start && i < layout.spatial_end)
      block[i] = "spatial";
    if (layout.theta_bym2_start >= 0 && i >= layout.theta_bym2_start && i < layout.theta_bym2_end)
      block[i] = "theta_bym2";
    if (layout.temporal_start >= 0 && i >= layout.temporal_start && i < layout.temporal_end)
      block[i] = "temporal";
    if (layout.hsgp_beta_start >= 0 && i >= layout.hsgp_beta_start && i < layout.hsgp_beta_end)
      block[i] = "hsgp_beta";
    if (layout.tvc_w_start >= 0 && i >= layout.tvc_w_start && i < layout.tvc_w_end)
      block[i] = "tvc_w";
    if (layout.log_tau_tvc_start >= 0 && i >= layout.log_tau_tvc_start && i < layout.log_tau_tvc_end)
      block[i] = "log_tau_tvc";
    if (i == layout.log_tau_spatial_idx) block[i] = "log_tau_spatial";
    if (i == layout.log_tau_temporal_idx) block[i] = "log_tau_temporal";
    if (i == layout.log_sigma2_hsgp_idx) block[i] = "log_sigma2_hsgp";
    if (i == layout.log_lengthscale_hsgp_idx) block[i] = "log_lengthscale_hsgp";
  }
  for (size_t t = 0; t < layout.re_start_multi.size(); t++) {
    for (int i = layout.re_start_multi[t]; i < layout.re_end_multi[t]; i++) block[i] = "re";
    if (layout.chol_re_start_multi[t] >= 0) {
      for (int i = layout.chol_re_start_multi[t]; i < layout.chol_re_end_multi[t]; i++)
        block[i] = "chol_re";
    }
    if (t < layout.log_sigma_re_slopes.size()) {
      for (int idx : layout.log_sigma_re_slopes[t]) block[idx] = "log_sigma_re";
    } else if (t < layout.log_sigma_re_multi.size()) {
      block[layout.log_sigma_re_multi[t]] = "log_sigma_re";
    }
  }

  return Rcpp::List::create(
    Rcpp::Named("analytic") = Rcpp::wrap(grad_analytic),
    Rcpp::Named("finite_diff") = Rcpp::wrap(grad_fd),
    Rcpp::Named("block") = block,
    Rcpp::Named("params") = Rcpp::wrap(params),
    Rcpp::Named("n_params") = n_params,
    Rcpp::Named("log_post") = point_value(params, data, layout,anchor),
    Rcpp::Named("log_post_mode") = lp_from_mode
  );
}

// [[Rcpp::export]]
double cpp_gradient_race(std::string field,
                         int n_points = 8,
                         int n_threads = 4,
                         int rounds = 8,
                         int n_obs = 200,
                         int n_units = 16,
                         int n_times = 8,
                         unsigned int seed = 42,
                         std::string family = "binomial",
                         bool prime_other_model = false) {
  if (!is_known_field(field)) {
    Rcpp::stop("Unknown field '%s'.", field.c_str());
  }

  Fixture data = make_model(field, n_obs, n_units, n_times, seed,
                              family, /*temporal_shared=*/true, "none");

  ParamLayout layout = fixture_layout(data);
  const int n_params = layout.total_params;
  GradientScope gradient_scope("H");
  tulpa::OmpCapScope thread_cap(1);
  tulpa::get_model_point_fn();

  std::vector<std::vector<double>> points(n_points);
  for (int k = 0; k < n_points; k++) {
    points[k] = draw_params(data, layout, n_params, seed + static_cast<unsigned int>(k));
  }

  if (prime_other_model) {
    Fixture other = make_model(field, n_obs, n_units, n_times, seed + 977u,
                                 family, /*temporal_shared=*/true, "none");

    ParamLayout other_layout = fixture_layout(other);
    if (other_layout.total_params != n_params) {
      Rcpp::stop("cpp_gradient_race: '%s' does not size its parameter vector "
                 "from the dimensions alone, so the two models cannot be "
                 "evaluated at the same point.", field.c_str());
    }
    std::vector<double> discard(n_params, 0.0);
    point_gradient(points[0],other,other_layout,discard);
  }

  std::vector<std::vector<double>> serial(n_points);
  for (int k = 0; k < n_points; k++) {
    serial[k].assign(n_params, 0.0);
    point_gradient(points[k], data, layout, serial[k]);
  }

  double max_dev = 0.0;
  for (int r = 0; r < rounds; r++) {
    std::vector<std::vector<double>> concurrent(n_points);
    for (int k = 0; k < n_points; k++) concurrent[k].assign(n_params, 0.0);

#ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(n_threads)
#endif
    for (int k = 0; k < n_points; k++) {
      point_gradient(points[k], data, layout, concurrent[k]);
    }

    for (int k = 0; k < n_points; k++) {
      for (int i = 0; i < n_params; i++) {
        const double d = std::fabs(concurrent[k][i] - serial[k][i]);
        if (d > max_dev) max_dev = d;
      }
    }
  }
  return max_dev;
}

// [[Rcpp::export]]
double cpp_logpost_at(std::string field,
                      int n_obs = 400,
                      int n_units = 25,
                      int n_times = 10,
                      unsigned int seed = 42,
                      std::string family = "binomial",
                      bool temporal_shared = true,
                      std::string zi = "none") {
  Fixture data = make_model(field, n_obs, n_units, n_times, seed,
                              family, temporal_shared, zi);
  ParamLayout layout = fixture_layout(data);
  const int n_params = layout.total_params;
  std::mt19937 rng(seed + 1000u);
  std::normal_distribution<double> rnorm(0.0, 0.35);
  std::vector<double> params(n_params);
  for (int i = 0; i < n_params; i++) params[i] = rnorm(rng);
  return point_value(params, data, layout);
}

// [[Rcpp::export]]
double cpp_st_gp_nc_coordinate_spread(std::string field = "stgp_nc",
                                      int n_draws = 5,
                                      unsigned int seed = 42) {
  const std::string centred_field = field.substr(0, field.size() - 3);
  Fixture data_nc = make_model(field, 400, 25, 10, seed);
  Fixture data_c = make_model(centred_field, 400, 25, 10, seed);
  ParamLayout layout = fixture_layout(data_nc);
  const ParamLayout centered_layout = fixture_layout(data_c);
  if (centered_layout.total_params != layout.total_params)
    Rcpp::stop("ST coordinate charts must have equal dimensions");
  const int n_params = layout.total_params;
  const int ST = data_nc.spatiotemporal_data.n_params;

  std::vector<double> params_nc = draw_params(data_nc, layout, n_params, seed);
  std::mt19937 rng(seed + 3000u);
  std::normal_distribution<double> rnorm(0.0, 1.0);

  double lo = R_PosInf, hi = R_NegInf;
  for (int d = 0; d < n_draws; d++) {
    for (int k = 0; k < ST; k++) params_nc[layout.st_delta_start + k] = rnorm(rng);
    const double lp_nc = point_value(params_nc, data_nc, layout);

    std::vector<double> params_c = stored_point(params_nc,data_nc,layout);
    const double diff = lp_nc - point_value(params_c, data_c, centered_layout);
    lo = std::min(lo, diff);
    hi = std::max(hi, diff);
  }
  return hi - lo;
}

// [[Rcpp::export]]
Rcpp::List cpp_st_ar1_zero_density(std::string field, Rcpp::NumericVector correlations) {
  Fixture data = make_model(field, 40, 9, 4, 19);
  const ParamLayout layout = fixture_layout(data);
  std::vector<double> q = draw_params(data, layout, layout.total_params, 19);
  const auto& st = data.spatiotemporal_data;
  std::fill(q.begin() + layout.st_delta_start, q.begin() + layout.st_delta_end, 0.0);
  q[layout.log_tau_st_idx] = 0.0;
  Rcpp::NumericVector values(correlations.size());
  for (int i = 0; i < correlations.size(); ++i) {
    const double rho = correlations[i];
    q[layout.logit_rho_st_idx] = std::log((1.0 + rho) / (1.0 - rho));
    values[i] = point_value(q, data, layout) - (data.st_rho_prior_a*std::log((1+rho)/2)+
         data.st_rho_prior_b*std::log((1-rho)/2)-std::log(2.0)-
         std::lgamma(data.st_rho_prior_a)-std::lgamma(data.st_rho_prior_b)+
         std::lgamma(data.st_rho_prior_a+data.st_rho_prior_b));
  }
  return Rcpp::List::create(Rcpp::Named("value") = values,
    Rcpp::Named("S") = st.n_spatial, Rcpp::Named("T") = st.n_times,
    Rcpp::Named("row_ptr") = st.adj_row_ptr, Rcpp::Named("col_idx") = st.adj_col_idx);
}

// [[Rcpp::export]]
double cpp_st_gaussian_conditional_exactness(std::string field="stgp",
    int n_draws=4,unsigned int seed=42,int n_obs=200,int n_units=9,int n_times=4) {
  auto data=make_model(field,n_obs,n_units,n_times,seed,"lognormal");
  data.st_parameterization=0;
  auto layout=fixture_layout(data);
  auto q=draw_params(data,layout,layout.total_params,seed);
  const int n=layout.st_delta_end-layout.st_delta_start;
  const int start=layout.st_delta_start;
  std::fill(q.begin()+start,q.begin()+start+n,0);
  std::vector<double> base_gradient;
  double base;
  point_gradient(q,data,layout,base_gradient,&base);
  Eigen::VectorXd score(n);
  for(int i=0;i<n;++i) score[i]=base_gradient[start+i];
  Eigen::MatrixXd precision(n,n);
  for(int j=0;j<n;++j) {
    auto at=q;at[start+j]=1;
    std::vector<double> gradient;
    point_gradient(at,data,layout,gradient);
    for(int i=0;i<n;++i) precision(i,j)=base_gradient[start+i]-gradient[start+i];
  }
  precision=(0.5*(precision+precision.transpose())).eval();
  Eigen::LLT<Eigen::MatrixXd> factor(precision);
  if(factor.info()!=Eigen::Success) Rcpp::stop("Gaussian conditional precision is not positive definite");
  const Eigen::VectorXd mean=factor.solve(score);
  const double logdet=2*factor.matrixL().toDenseMatrix().diagonal().array().log().sum();
  const double marginal=base+0.5*score.dot(mean)+0.5*n*std::log(2*M_PI)-0.5*logdet;
  std::mt19937 rng(seed+5000u);
  std::normal_distribution<double> normal(0,0.3);
  double worst=0;
  for(int d=0;d<n_draws;++d) {
    Eigen::VectorXd error(n);
    for(int i=0;i<n;++i) { error[i]=normal(rng);q[start+i]=mean[i]+error[i]; }
    const double conditional=-0.5*n*std::log(2*M_PI)+0.5*logdet-
      0.5*error.dot(precision*error);
    worst=std::max(worst,std::abs(point_value(q,data,layout)-conditional-marginal));
  }
  return worst;
}

inline double reference_rw_quad(const double* x,int n,int order,bool cyclic) {
  double quad=0;
  const int count=cyclic ? n : n-order;
  for(int i=0;i<count;++i) {
    const double difference=order==1 ? x[(i+1)%n]-x[i] :
      x[(i+2)%n]-2*x[(i+1)%n]+x[i];
    quad+=difference*difference;
  }
  return quad;
}

inline double nc_reference_log_prior(const double* phi, int n, int n_groups,
                                     int order, bool cyclic, double sigma2) {
  double quad = 0.0;
  for (int g = 0; g < n_groups; g++) {
    quad += (order == 2)
                ? reference_rw_quad(phi + g * n,n,2,cyclic)
                : reference_rw_quad(phi + g * n,n,1,cyclic);
    const double s = tulpa::s2z_component_sum(phi, g * n, n);
    quad += tulpa::s2z_aug_coef(1.0, n) * s * s;
  }

  const int rank_g = (order == 2) ? tulpa::rw2_rank(n, cyclic)
                                  : tulpa::rw1_rank(n, cyclic);
  const int rank = tulpa::s2z_aug_rank(rank_g * n_groups, n_groups);
  return -0.5 * quad / sigma2 - 0.5 * rank * std::log(2.0 * M_PI * sigma2);
}

// [[Rcpp::export]]
double cpp_ms_temporal_nc_coordinate_spread(std::string field = "ms_temporal_nc",
                                            int n = 12,
                                            double sigma2 = 0.25,
                                            int n_draws = 6,
                                            unsigned int seed = 42,
                                            int n_groups = 1) {
  int order = 1;
  bool cyclic = false;
  if (field == "ms_temporal_rw2_nc") {
    order = 2;
  } else if (field == "ms_temporal_seasonal_nc") {
    cyclic = true;
  } else if (field != "ms_temporal_nc") {
    Rcpp::stop("Unknown multi-scale temporal field '%s'.", field);
  }

  std::mt19937 rng(seed + 7000u);
  std::normal_distribution<double> rnorm(0.0, 1.0);
  const double sigma = std::sqrt(sigma2);
  const int n_total = n * n_groups;

  double first = 0.0;
  double worst = 0.0;
  for (int d = 0; d < n_draws; d++) {
    std::vector<double> z(n_total), phi(n_total);
    for (int k = 0; k < n_total; k++) z[k] = rnorm(rng);
    tulpa_temporal_nc::rw_nc_grouped_forward(z.data(), n, n_groups, order,
                                              cyclic, sigma, phi.data());

    const double lp_nc = tulpa_temporal_nc::rw_nc_grouped_log_prior(
        z.data(), n, n_groups, order, cyclic);
    const double lp_centred =
        nc_reference_log_prior(phi.data(), n, n_groups, order, cyclic, sigma2);
    const double gap = lp_nc - lp_centred;
    if (d == 0) first = gap;
    worst = std::max(worst, std::abs(gap - first));
  }
  return worst;
}

// [[Rcpp::export]]
Rcpp::NumericVector cpp_polygamma(int n, Rcpp::NumericVector x) {
  Rcpp::NumericVector out(x.size());
  for (R_xlen_t i = 0; i < x.size(); i++) {
    out[i] = ratiod::math::portable_polygamma(n, x[i]);
  }
  return out;
}
