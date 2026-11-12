#ifndef TULPARATIO_ENGINE_FIXTURE_H
#define TULPARATIO_ENGINE_FIXTURE_H
#include <RcppEigen.h>
#include <random>
#include <tulpa/model_point_api.h>
#include <tulpa/field_setup_api.h>
#include <tulpa/soft_sum_to_zero.h>
#include <tulpa/temporal_nc.h>
#include <tulpa/field_range.h>
#include <tulpa/st_margins.h>
#include "lik_specs/family_registry.h"
#include "lik_specs/lik_helpers.h"
#include "scale_prior.h"

namespace engine_fixture {
using tulpa::SpatialType;
using tulpa::TemporalType;
using tulpa::TemporalCovType;
using tulpa::ZIType;
using ratiod_link::LinkType;
using tulpa::ParamLayout;

struct Fixture : tulpa::ModelData {
    tulpaRatio::lik::ResponseData response;
    tulpaRatio::RatioConfig config;
    tulpa::LikelihoodSpec spec;
    Fixture() { processes.resize(2); sharing.init(2); }
    void bind() {
        spec = tulpaRatio::build_ratio_likelihood_spec(config);
        n_processes = spec.n_processes;
        processes.resize(n_processes);
        model_response_data = &response;
        likelihood_spec = &spec;
        const auto share = [this](std::vector<bool>& mask, bool shared) {
            mask.assign(n_processes, shared); mask[0] = true;
        };
        share(sharing.latent,latent_shared);
        share(sharing.svc,svc_data.shared);
        share(sharing.tvc,tvc_data.shared);
        share(sharing.st,spatiotemporal_data.shared);
        if (has_latent) {
            latent_bilinear = true;
            latent_loading_processes = latent_shared ? n_processes : 1;
        }
        if (n_re_groups || n_re_terms || has_re_slopes || has_temporal_gp || has_svc || has_tvc || has_spatiotemporal ||
            has_multiscale_temporal || has_latent || spatial_restriction.rank ||
            temporal_restriction_rank || has_gp || has_hsgp || has_multiscale_gp ||
            n_spatial_units || temporal_type != TemporalType::NONE)
            spec.gradient_fn = nullptr;
    }
};

inline ParamLayout fixture_layout(Fixture& data) {
    data.bind();
    return tulpa::compute_layout(data);
}

inline void hsgp_basis(const std::vector<double>& coords, int n, int m,
                        double boundary, bool shared, tulpa::HSGPData& out) {
    Rcpp::NumericMatrix x(n,2);
    for (int i=0;i<n;++i) { x(i,0)=coords[2*i]; x(i,1)=coords[2*i+1]; }
    Rcpp::Function build=Rcpp::Environment::namespace_env("tulpa")["cpp_hsgp_basis_2d"];
    Rcpp::List result=build(x,m,boundary);
    Rcpp::NumericMatrix basis=result["phi_basis"];
    out.n_obs=n; out.m_per_dim=m; out.m_total=basis.ncol(); out.shared=shared;
    out.phi_flat.resize(static_cast<size_t>(n)*out.m_total);
    for(int i=0;i<n;++i) for(int j=0;j<out.m_total;++j)
        out.phi_flat[i*out.m_total+j]=basis(i,j);
    out.eigenvalues=Rcpp::as<std::vector<double>>(result["lambda_eig"]);
}

inline ZIType inflation_type(const std::string& s) {
    const std::vector<std::pair<std::string,ZIType>> entries={
        {"none",ZIType::NONE},{"zi_poisson",ZIType::ZI_POISSON},
        {"zi_negbin",ZIType::ZI_NEGBIN},{"hurdle_poisson",ZIType::HURDLE_POISSON},
        {"hurdle_negbin",ZIType::HURDLE_NEGBIN},{"zi_binomial",ZIType::ZI_BINOMIAL},
        {"hurdle_binomial",ZIType::HURDLE_BINOMIAL},
        {"oi_binomial",ZIType::OI_BINOMIAL},{"zoib",ZIType::ZOIB}};
    for(const auto& entry:entries) if(entry.first==s) return entry.second;
    Rcpp::stop("Unknown inflation type '%s'",s.c_str());
}
void build_grid_adjacency(int n,
                          std::vector<int>& row_ptr,
                          std::vector<int>& col_idx,
                          std::vector<int>& n_neighbors) {
  const int side = static_cast<int>(std::ceil(std::sqrt(static_cast<double>(n))));
  std::vector<int> lon(n), lat(n);
  for (int i = 0; i < n; i++) {
    lon[i] = i % side;
    lat[i] = i / side;
  }
  row_ptr.assign(n + 1, 0);
  col_idx.clear();
  n_neighbors.assign(n, 0);
  for (int i = 0; i < n; i++) {
    row_ptr[i] = static_cast<int>(col_idx.size());
    for (int j = 0; j < n; j++) {
      if (i == j) continue;
      const int dx = lon[i] - lon[j];
      const int dy = lat[i] - lat[j];
      if (dx * dx + dy * dy <= 2) {
        col_idx.push_back(j);
        n_neighbors[i]++;
      }
    }
  }
  row_ptr[n] = static_cast<int>(col_idx.size());
}

struct NNGPNeighbors {
  int k = 0;
  std::vector<double> coords;
  std::vector<int> nn_idx;
  std::vector<double> nn_dist;
  std::vector<double> nn_neighbor_dist;
  std::vector<int> nn_order;
  std::vector<int> nn_order_inv;
};

inline NNGPNeighbors build_nngp(const std::vector<double>& coords, int n, int k) {
  NNGPNeighbors out;
  out.k = k;

  std::vector<int> order(n);
  for (int i = 0; i < n; i++) order[i] = i;
  std::sort(order.begin(), order.end(), [&](int a, int b) {
    if (coords[static_cast<size_t>(a) * 2] != coords[static_cast<size_t>(b) * 2])
      return coords[static_cast<size_t>(a) * 2] < coords[static_cast<size_t>(b) * 2];
    return coords[static_cast<size_t>(a) * 2 + 1] < coords[static_cast<size_t>(b) * 2 + 1];
  });

  out.coords.resize(static_cast<size_t>(n) * 2);
  for (int i = 0; i < n; i++) {
    out.coords[static_cast<size_t>(i) * 2 + 0] = coords[static_cast<size_t>(order[i]) * 2 + 0];
    out.coords[static_cast<size_t>(i) * 2 + 1] = coords[static_cast<size_t>(order[i]) * 2 + 1];
  }

  auto dist = [&](int a, int b) {
    const double dx = out.coords[static_cast<size_t>(a) * 2] - out.coords[static_cast<size_t>(b) * 2];
    const double dy = out.coords[static_cast<size_t>(a) * 2 + 1] - out.coords[static_cast<size_t>(b) * 2 + 1];
    return std::sqrt(dx * dx + dy * dy);
  };

  out.nn_idx.assign(static_cast<size_t>(n) * k, 0);
  out.nn_dist.assign(static_cast<size_t>(n) * k, 0.0);
  out.nn_neighbor_dist.assign(static_cast<size_t>(n) * k * k, 0.0);

  for (int i = 1; i < n; i++) {
    std::vector<std::pair<double, int>> cand;
    cand.reserve(i);
    for (int j = 0; j < i; j++) cand.emplace_back(dist(i, j), j);
    std::sort(cand.begin(), cand.end());
    const int m = std::min(static_cast<int>(cand.size()), k);
    for (int c = 0; c < m; c++) {
      out.nn_idx[static_cast<size_t>(i) * k + c] = cand[c].second + 1;
      out.nn_dist[static_cast<size_t>(i) * k + c] = cand[c].first;
    }
    for (int a = 0; a < m; a++) {
      for (int b = 0; b < m; b++) {
        out.nn_neighbor_dist[(static_cast<size_t>(i) * k + a) * k + b] =
            (a == b) ? 0.0 : dist(cand[a].second, cand[b].second);
      }
    }
  }

  out.nn_order.resize(n);
  out.nn_order_inv.resize(n);
  for (int i = 0; i < n; i++) {
    out.nn_order[i] = order[i];
    out.nn_order_inv[order[i]] = i;
  }
  return out;
}

const char* const KNOWN_FIELDS[] = {
  "none",
  "icar", "bym2", "car_proper", "car_proper_ms", "car_proper_st",
  "car_proper_raw",
  "rw1", "rw2", "rw1_nc", "rw2_nc", "rw1_cyclic", "rw1_cyclic_nc",
  "rw1_panel", "rw1_nc_panel", "rw2_nc_panel", "rw1_cyclic_nc_panel",
  "temporal_ar1", "temporal_ar1_nc", "temporal_iid", "icar_rw1",
  "icar_ms", "bym2_ms", "icar_st", "bym2_st",
  "icar_disconnected", "bym2_disconnected", "icar_disconnected_rw1", "st4", "st4_nc", "st4_ar1",
  "st4_ar1_nc", "st2", "st2_rw2", "st2_ar1", "st3", "icar_st_nc", "st2_nc",
  "st4_rw2", "st4_rw2_nc", "st4_rw2_cyclic", "st4_rw2_cyclic_nc",
  "st2_rw2_nc", "st2_ar1_nc", "st2_cyclic", "st2_cyclic_nc", "st3_nc",
  "stgp", "stgp_matern", "stgp_gneiting", "stgp_latent",
  "stgp_nc", "stgp_gneiting_nc", "stgp_collapsed", "stgp_gneiting_collapsed",
  "icar_collapsed", "bym2_collapsed",
  "icar_collapsed_rw1", "bym2_collapsed_rw1",
  "hsgp",
  "ms_temporal_nc", "ms_temporal_rw2", "ms_temporal_rw2_nc",
  "ms_temporal_seasonal", "ms_temporal_seasonal_nc",
  "tvc", "tvc_iid", "tvc_ar1", "tvc_rw2",
  "re", "re_crossed", "re_slopes", "re_slopes_corr",
  "gp", "gp_matern", "gp_gaussian", "gp_spherical",
  "gp_nc", "gp_collapsed", "gp_temporal",
  "msgp_temporal", "msgp_hsgp",
  "svc", "svc_hsgp", "svc_nc",
  "temporal_gp", "ms_temporal", "latent", "latent2", "latent_num",
  "latent2_first_zero", "stgp_latent2",

  "gp_st4", "gp_stgp", "gp_temporal_st4", "msgp_st4", "gp_collapsed_st4",
  "icar_collapsed_st4", "bym2_collapsed_st4", "tvc_st4", "temporal_gp_st4",
  "gp_tgp", "svc_ms", "svc_ms_nc", "gp_slopes", "gp_slopes_corr", "gp_crossed",

  "temporal_gp_centered", "temporal_gp_matern32", "temporal_gp_matern52",
  "temporal_gp_gaussian", "temporal_gp_periodic",
  "temporal_gp_matern32_centered", "temporal_gp_periodic_centered",
  "gp_tgp_matern32", "gp_tgp_centered", "gp_tgp_gaussian_centered"
};

const char* const RSR_FIELDS[] = {"icar", "bym2", "hsgp", "gp"};
const char* const RSR_SUFFIX = "_rsr";
const char* const RTR_FIELDS[] = {"rw1", "temporal_ar1", "temporal_gp",
                                  "temporal_gp_matern32", "ms_temporal"};
const char* const RTR_SUFFIX = "_rtr";

bool has_suffix(const std::string& field, const char* s) {
  const std::string suffix(s);
  return field.size() > suffix.size() &&
         field.compare(field.size() - suffix.size(), suffix.size(), suffix) == 0;
}

bool is_rsr_field(const std::string& field) { return has_suffix(field, RSR_SUFFIX); }
bool is_rtr_field(const std::string& field) { return has_suffix(field, RTR_SUFFIX); }

template <size_t K>
bool restricted_base_known(const std::string& field, const char* suffix,
                           const char* const (&bases)[K]) {
  const std::string base = field.substr(0, field.size() - std::string(suffix).size());
  for (const char* known : bases) {
    if (base == known) return true;
  }
  return false;
}

bool is_known_field(const std::string& field) {
  if (is_rsr_field(field)) return restricted_base_known(field, RSR_SUFFIX, RSR_FIELDS);
  if (is_rtr_field(field)) return restricted_base_known(field, RTR_SUFFIX, RTR_FIELDS);
  for (const char* known : KNOWN_FIELDS) {
    if (field == known) return true;
  }
  return false;
}

struct STNNGPNeighbors {
  int k = 0;
  std::vector<int> nn_idx;
  std::vector<double> nn_dist_space;
  std::vector<double> nn_dist_time;
  std::vector<int> nn_order;
  std::vector<int> nn_order_inv;
};

inline STNNGPNeighbors build_nngp_st(const std::vector<double>& coords,
                              const std::vector<double>& times, int n, int k) {
  STNNGPNeighbors out;
  out.k = k;

  std::vector<int> order(n);
  for (int i = 0; i < n; i++) order[i] = i;
  std::sort(order.begin(), order.end(), [&](int a, int b) {
    if (times[a] != times[b]) return times[a] < times[b];
    if (coords[static_cast<size_t>(a) * 2] != coords[static_cast<size_t>(b) * 2])
      return coords[static_cast<size_t>(a) * 2] < coords[static_cast<size_t>(b) * 2];
    return coords[static_cast<size_t>(a) * 2 + 1] < coords[static_cast<size_t>(b) * 2 + 1];
  });

  auto sep = [&](int a, int b, double* h, double* u) {
    const int ia = order[a], ib = order[b];
    const double dx = coords[static_cast<size_t>(ia) * 2] - coords[static_cast<size_t>(ib) * 2];
    const double dy = coords[static_cast<size_t>(ia) * 2 + 1] - coords[static_cast<size_t>(ib) * 2 + 1];
    *h = std::sqrt(dx * dx + dy * dy);
    *u = std::abs(times[ia] - times[ib]);
  };

  out.nn_idx.assign(static_cast<size_t>(n) * k, 0);
  out.nn_dist_space.assign(static_cast<size_t>(n) * k, 0.0);
  out.nn_dist_time.assign(static_cast<size_t>(n) * k, 0.0);

  for (int i = 1; i < n; i++) {
    std::vector<std::pair<double, int>> cand;
    cand.reserve(i);
    for (int j = 0; j < i; j++) {
      double h, u;
      sep(i, j, &h, &u);
      cand.emplace_back(std::sqrt(h * h + u * u), j);
    }
    std::sort(cand.begin(), cand.end());
    const int m = std::min(static_cast<int>(cand.size()), k);
    for (int c = 0; c < m; c++) {
      double h, u;
      sep(i, cand[c].second, &h, &u);
      out.nn_idx[static_cast<size_t>(i) * k + c] = cand[c].second + 1;
      out.nn_dist_space[static_cast<size_t>(i) * k + c] = h;
      out.nn_dist_time[static_cast<size_t>(i) * k + c] = u;
    }
  }

  out.nn_order.resize(n);
  out.nn_order_inv.resize(n);
  for (int i = 0; i < n; i++) {
    out.nn_order[i] = order[i];
    out.nn_order_inv[order[i]] = i;
  }
  return out;
}

inline tulpa::OrthogonalProjection numerator_design_restriction(const Fixture& data) {
  const int n = data.N;
  std::vector<double> q0(n), q1(n);
  double n0 = 0.0;
  for (int i = 0; i < n; i++) {
    q0[i] = data.processes[0].X_flat[static_cast<size_t>(i) * 2 + 0];
    n0 += q0[i] * q0[i];
  }
  n0 = std::sqrt(n0);
  double dot = 0.0;
  for (int i = 0; i < n; i++) {
    q0[i] /= n0;
    dot += q0[i] * data.processes[0].X_flat[static_cast<size_t>(i) * 2 + 1];
  }
  double n1 = 0.0;
  for (int i = 0; i < n; i++) {
    q1[i] = data.processes[0].X_flat[static_cast<size_t>(i) * 2 + 1] - dot * q0[i];
    n1 += q1[i] * q1[i];
  }
  n1 = std::sqrt(n1);
  tulpa::OrthogonalProjection r;
  r.rank = 2;
  r.basis.resize(static_cast<size_t>(n) * 2);
  for (int i = 0; i < n; i++) {
    r.basis[static_cast<size_t>(i) * 2 + 0] = q0[i];
    r.basis[static_cast<size_t>(i) * 2 + 1] = q1[i] / n1;
  }
  return r;
}

inline Fixture make_model(const std::string& field_in, int n_obs, int n_units,
                     int n_times, unsigned int seed,
                     const std::string& family = "binomial",
                     bool temporal_shared = true,
                     const std::string& zi = "none") {
  const bool want_rsr = is_rsr_field(field_in);
  const bool want_rtr = is_rtr_field(field_in);
  const std::string field = want_rsr
      ? field_in.substr(0, field_in.size() - std::string(RSR_SUFFIX).size())
      : want_rtr
      ? field_in.substr(0, field_in.size() - std::string(RTR_SUFFIX).size())
      : field_in;
  if (!is_known_field(field_in)) {
    Rcpp::stop("cpp_gradient_check: unknown field '" + field_in +
               "'. A name make_model() does not build a structure for would "
               "yield a model with no structured field, and the check would "
               "pass while testing nothing.");
  }

  Fixture data;
  std::mt19937 rng(seed);
  std::normal_distribution<double> rnorm(0.0, 1.0);
  std::uniform_int_distribution<int> rtrials(10, 50);

  const bool is_binomial = (family == "binomial" || family == "beta_binomial");

  const bool denom_is_count = (family == "negbin_negbin" || family == "poisson_negbin");
  const bool fixed_exposure = (family == "poisson_fixed" || family == "negbin_fixed" ||
                               family == "lognormal_fixed");

  const bool num_is_continuous = (family == "lognormal_fixed" ||
                                  family == "gamma_gamma" ||
                                  family == "lognormal");
  const ZIType zi_type = inflation_type(zi);
  const bool has_zi = (zi_type != ZIType::NONE);
  const bool has_oi = (zi_type == ZIType::OI_BINOMIAL || zi_type == ZIType::ZOIB);

  const bool want_ms = (field == "icar_ms" || field == "bym2_ms" ||
                        field == "ms_temporal" || field == "svc_ms" ||
                        field == "svc_ms_nc" ||
                        field == "car_proper_ms" ||
                        field == "ms_temporal_nc" ||
                        field == "ms_temporal_rw2" ||
                        field == "ms_temporal_rw2_nc" ||
                        field == "ms_temporal_seasonal" ||
                        field == "ms_temporal_seasonal_nc");
  const bool want_st = (field == "icar_st" || field == "bym2_st" ||
                        field == "car_proper_st" || field == "icar_st_nc");

  const bool want_st4 = (field == "st4" || field == "st4_nc" ||
                         field == "st4_rw2" || field == "st4_rw2_nc" ||
                         field == "st4_rw2_cyclic" || field == "st4_rw2_cyclic_nc" ||
                         field == "st4_ar1" || field == "st4_ar1_nc" ||
                         field == "gp_st4" || field == "gp_temporal_st4" ||
                         field == "msgp_st4" || field == "gp_collapsed_st4" ||
                         field == "icar_collapsed_st4" ||
                         field == "bym2_collapsed_st4" ||
                         field == "tvc_st4" || field == "temporal_gp_st4");

  const bool want_st2 = (field == "st2" || field == "st2_rw2" ||
                         field == "st2_ar1" || field == "st2_nc" ||
                         field == "st2_rw2_nc" || field == "st2_ar1_nc" ||
                         field == "st2_cyclic" || field == "st2_cyclic_nc");
  const bool want_st3 = (field == "st3" || field == "st3_nc");

  const bool want_stgp = (field == "stgp" || field == "stgp_matern" ||
                          field == "stgp_gneiting" || field == "stgp_latent" ||
                          field == "stgp_latent2" ||
                          field == "stgp_nc" || field == "stgp_gneiting_nc" ||
                          field == "stgp_collapsed" ||
                          field == "stgp_gneiting_collapsed" ||
                          field == "gp_stgp");

  const bool want_disconnected = (field == "icar_disconnected" ||
                                  field == "bym2_disconnected" ||
                                  field == "icar_disconnected_rw1");
  const bool want_icar = (field == "icar" || field == "icar_rw1" ||
                          field == "icar_disconnected" ||
                          field == "icar_disconnected_rw1" ||
                          field == "icar_ms" || field == "icar_st" ||
                          field == "icar_collapsed" || field == "icar_collapsed_rw1" ||
                          field == "icar_collapsed_st4");
  const bool want_bym2 = (field == "bym2" || field == "bym2_ms" ||
                          field == "bym2_disconnected" ||
                          field == "bym2_st" || field == "bym2_collapsed" ||
                          field == "bym2_collapsed_rw1" ||
                          field == "bym2_collapsed_st4");

  const bool want_car_proper = (field == "car_proper" ||
                                field == "car_proper_ms" ||
                                field == "car_proper_st" ||
                                field == "car_proper_raw");
  const bool want_rw1  = (field == "rw1"  || field == "rw1_nc" ||
                          field == "rw1_cyclic" || field == "rw1_cyclic_nc" ||
                          field == "rw1_panel" || field == "rw1_nc_panel" ||
                          field == "rw1_cyclic_nc_panel" ||
                          field == "icar_rw1" || field == "icar_disconnected_rw1" || want_st ||
                          field == "icar_collapsed_rw1" || field == "bym2_collapsed_rw1" ||
                          field == "gp_temporal" || field == "msgp_temporal" ||
                          field == "gp_temporal_st4");
  const bool want_rw2  = (field == "rw2" || field == "rw2_nc" ||
                          field == "rw2_nc_panel");

  const bool want_temporal_ar1 = (field == "temporal_ar1" ||
                                  field == "temporal_ar1_nc");

  const bool want_temporal_iid = (field == "temporal_iid");
  const bool want_hsgp = (field == "hsgp");
  const bool want_tvc  = (field == "tvc" || field == "tvc_iid" || field == "tvc_rw2" ||
                          field == "tvc_ar1" || field == "tvc_st4");

  const bool want_gp_collapsed = (field == "gp_collapsed" ||
                                  field == "gp_collapsed_st4");

  const bool want_gp_nc = (field == "gp_nc");
  const bool want_gp = (field == "gp" || field == "gp_matern" ||
                        field == "gp_gaussian" || field == "gp_spherical" ||
                        field == "gp_temporal" || want_gp_nc ||
                        want_gp_collapsed ||
                        field == "gp_st4" || field == "gp_stgp" ||
                        field == "gp_temporal_st4" ||
                        field.rfind("gp_tgp", 0) == 0 ||
                        field == "gp_slopes" || field == "gp_slopes_corr" ||
                        field == "gp_crossed");

  const bool want_msgp = (field == "msgp_temporal" || field == "msgp_st4" ||
                          field == "msgp_hsgp");
  const bool want_svc_hsgp = (field == "svc_hsgp");

  const bool want_svc_nc = (field == "svc_nc" || field == "svc_ms_nc");
  const bool want_svc = (field == "svc" || want_svc_hsgp || field == "svc_ms" ||
                         want_svc_nc);

  const bool want_temporal_gp = (field.rfind("temporal_gp", 0) == 0 ||
                                 field.rfind("gp_tgp", 0) == 0);
  const bool want_latent = (field == "latent" || field == "stgp_latent" ||
                             field == "latent2" || field == "latent_num" ||
                             field == "latent2_first_zero" ||
                             field == "stgp_latent2");

  const bool want_re_slopes = (field == "re_slopes" || field == "re_slopes_corr" ||
                               field == "gp_slopes" || field == "gp_slopes_corr");
  const bool want_re = (field == "re" || field == "re_crossed" ||
                        field == "gp_crossed" || want_re_slopes);

  data.N = n_obs;
  data.config.family = family; data.config.zi = zi;
  const auto& entry=tulpaRatio::family_entry(family);
  data.config.num_link=entry.num_link; data.config.denom_link=entry.denom_link;
  ratiod_link::parse_link(entry.num_link,data.response.num_link);
  ratiod_link::parse_link(entry.denom_link,data.response.denom_link);
  data.processes[0].p = 2;
  data.processes[1].p = (is_binomial || fixed_exposure) ? 0 : 2;



  data.sigma_beta = 10.0;
  data.sigma_re_scale = 1.0;
  data.phi_prior_shape = 1.0;
  data.phi_prior_rate = 0.01;
  data.spatial_sigma_prior_U = 3.0;
  data.spatial_sigma_prior_alpha = 0.01;
  data.temporal_sigma_prior_U = 3.0;
  data.temporal_sigma_prior_alpha = 0.01;

  data.processes[0].X_flat.resize(static_cast<size_t>(n_obs) * 2);
  data.response.y_num.resize(n_obs);

  data.response.y_denom.assign(n_obs, 1);
  if (num_is_continuous) data.response.y_num_cont.resize(n_obs);
  if (fixed_exposure) {
    data.response.y_denom_cont.resize(n_obs);
  } else if (!is_binomial) {
    data.processes[1].X_flat.resize(static_cast<size_t>(n_obs) * 2);
    if (denom_is_count) {
      data.response.y_denom.assign(n_obs, 0);
    } else {
      data.response.y_denom_cont.resize(n_obs);
    }
  }
  if (has_zi) data.X_zi_flat.resize(static_cast<size_t>(n_obs) * 2);
  if (has_oi) data.X_oi_flat.resize(static_cast<size_t>(n_obs) * 2);
  std::uniform_real_distribution<double> runif(0.0, 1.0);
  for (int i = 0; i < n_obs; i++) {
    const double x = rnorm(rng);
    data.processes[0].X_flat[static_cast<size_t>(i) * 2 + 0] = 1.0;
    data.processes[0].X_flat[static_cast<size_t>(i) * 2 + 1] = x;
    if (is_binomial) {
      const int trials = rtrials(rng);
      const double p = 1.0 / (1.0 + std::exp(-(0.5 + 0.3 * x)));
      std::binomial_distribution<int> rbinom(trials, p);
      data.response.y_denom[i] = trials;
      data.response.y_num[i] = rbinom(rng);
    } else if (fixed_exposure) {
      const double exposure = 0.5 + 2.5 * runif(rng);
      data.response.y_denom_cont[i] = exposure;
      if (num_is_continuous) {
        data.response.y_num_cont[i] = exposure * std::exp(0.5 + 0.3 * x + 0.4 * rnorm(rng));
        data.response.y_num[i] = 0;
      } else {
        std::poisson_distribution<int> rpois(exposure * std::exp(0.5 + 0.3 * x));
        data.response.y_num[i] = rpois(rng);
      }
    } else {
      const double z = rnorm(rng);
      data.processes[1].X_flat[static_cast<size_t>(i) * 2 + 0] = 1.0;
      data.processes[1].X_flat[static_cast<size_t>(i) * 2 + 1] = z;
      if (num_is_continuous) {
        data.response.y_num_cont[i] = std::exp(0.5 + 0.3 * x + 0.4 * rnorm(rng));
        data.response.y_num[i] = 0;
      } else {
        std::poisson_distribution<int> rpois(std::exp(0.5 + 0.3 * x));
        data.response.y_num[i] = rpois(rng);
      }
      if (denom_is_count) {
        std::poisson_distribution<int> rpois_d(std::exp(0.4 + 0.2 * z));
        data.response.y_denom[i] = rpois_d(rng);
      } else {
        std::gamma_distribution<double> rgamma(2.0, std::exp(0.4 + 0.2 * z) / 2.0);
        data.response.y_denom_cont[i] = std::max(1e-3, rgamma(rng));
      }
    }
    if (has_oi) {
      data.X_oi_flat[static_cast<size_t>(i) * 2 + 0] = 1.0;
      data.X_oi_flat[static_cast<size_t>(i) * 2 + 1] = x;
    }
    if (has_zi) {
      data.X_zi_flat[static_cast<size_t>(i) * 2 + 0] = 1.0;
      data.X_zi_flat[static_cast<size_t>(i) * 2 + 1] = x;

      if (runif(rng) < 1.0 / (1.0 + std::exp(-(-0.4 + 0.6 * x)))) data.response.y_num[i] = 0;
    }
  }

  data.re_parameterization = 0;
  if (want_re) {
    const int n_terms = (field == "re_crossed" || field == "gp_crossed") ? 2 : 1;
    const int n_coefs = want_re_slopes ? 2 : 1;
    const bool correlated = (field == "re_slopes_corr" ||
                             field == "gp_slopes_corr");
    const int n_groups = 5;

    data.n_re_terms = n_terms;
    data.n_re_groups = n_groups;
    data.re_n_groups_multi.assign(n_terms, n_groups);
    data.total_re_groups = n_terms * n_groups;
    data.re_group.resize(n_obs);
    data.re_group_multi_flat.resize(static_cast<size_t>(n_obs) * n_terms);
    for (int i = 0; i < n_obs; i++) {
      data.re_group[i] = (i % n_groups) + 1;
      for (int t = 0; t < n_terms; t++) {

        data.re_group_multi_flat[static_cast<size_t>(i) * n_terms + t] =
            ((i + t) % n_groups) + 1;
      }
    }

    data.has_re_slopes = want_re_slopes;
    data.has_re_correlated_slopes = correlated;
    data.re_n_coefs.assign(n_terms, n_coefs);
    data.re_n_slopes.assign(n_terms, n_coefs - 1);
    data.re_correlated.assign(n_terms, correlated);
    data.re_n_chol.assign(n_terms, correlated ? n_coefs * (n_coefs - 1) / 2 : 0);
    data.re_slope_matrices.assign(n_terms, std::vector<double>());
    if (n_coefs > 1) {
      for (int t = 0; t < n_terms; t++) {
        data.re_slope_matrices[t].resize(static_cast<size_t>(n_obs) * (n_coefs - 1));
        for (int i = 0; i < n_obs; i++) {
          data.re_slope_matrices[t][i] =
              data.processes[0].X_flat[static_cast<size_t>(i) * 2 + 1];
        }
      }
    }
  } else {

    data.n_re_terms = 0;
    data.n_re_groups = 0;
    data.total_re_groups = 0;
  }

  if (want_icar || want_bym2 || want_car_proper) {
    data.spatial_type = want_bym2 ? SpatialType::BYM2 :
                         want_car_proper ? SpatialType::CAR_PROPER : SpatialType::ICAR;
    data.n_spatial_units = n_units;
    build_grid_adjacency(n_units, data.adj_row_ptr, data.adj_col_idx, data.n_neighbors);
    if (want_disconnected) {

      const int h = (n_units - 1) / 2;
      std::vector<int> ptr_a, idx_a, nn_a, ptr_b, idx_b, nn_b;
      build_grid_adjacency(h, ptr_a, idx_a, nn_a);
      build_grid_adjacency(n_units - 1 - h, ptr_b, idx_b, nn_b);
      data.adj_row_ptr.assign(n_units + 1, 0);
      data.adj_col_idx.clear();
      data.n_neighbors.assign(n_units, 0);
      for (int i = 0; i < n_units; i++) {
        data.adj_row_ptr[i] = static_cast<int>(data.adj_col_idx.size());
        if (i < h) {
          for (int k = ptr_a[i]; k < ptr_a[i + 1]; k++) data.adj_col_idx.push_back(idx_a[k]);
          data.n_neighbors[i] = nn_a[i];
        } else if (i < n_units - 1) {
          for (int k = ptr_b[i - h]; k < ptr_b[i - h + 1]; k++)
            data.adj_col_idx.push_back(idx_b[k] + h);
          data.n_neighbors[i] = nn_b[i - h];
        }
      }
      data.adj_row_ptr[n_units] = static_cast<int>(data.adj_col_idx.size());
      data.bym2_node_prec.assign(n_units, 1.0);
      if (want_bym2) {
        for (int i = h; i < n_units - 1; i++) data.bym2_node_prec[i] = 1.7;
        data.bym2_node_prec[n_units - 1] = 0.6;
      }
    }
    data.spatial_group.resize(n_obs);
    for (int i = 0; i < n_obs; i++) data.spatial_group[i] = (i % n_units) + 1;
    data.bym2_scale_factor = 1.0;

    data.icar_collapsed = (field == "icar_collapsed" || field == "icar_collapsed_rw1" ||
                           field == "icar_collapsed_st4");
    data.bym2_collapsed = (field == "bym2_collapsed" || field == "bym2_collapsed_rw1" ||
                           field == "bym2_collapsed_st4");

    data.car_rho_lower = 0.0;
    data.car_rho_upper = 1.0;
    data.car_rho_prior_a = 1.0;
    data.car_rho_prior_b = 1.0;

    data.car_center = (field != "car_proper_raw");
  } else if (want_hsgp) {

    data.spatial_type = SpatialType::HSGP;
    data.has_hsgp = true;
    data.hsgp_lengthscale_prior_U = 0.1;
    data.hsgp_lengthscale_prior_alpha = 0.05;
    std::vector<double> coords(static_cast<size_t>(n_obs) * 2);
    std::uniform_real_distribution<double> runif(0.0, 1.0);
    for (int i = 0; i < n_obs; i++) {
      coords[static_cast<size_t>(i) * 2 + 0] = runif(rng);
      coords[static_cast<size_t>(i) * 2 + 1] = runif(rng);
    }
    hsgp_basis(coords, n_obs, 3, 1.5, /*shared=*/true,
                               data.hsgp_data);
    data.n_spatial_units = 0;
    data.bym2_scale_factor = 1.0;
  } else if (want_gp || want_msgp) {

    std::vector<double> gp_coords(static_cast<size_t>(n_units) * 2);
    std::uniform_real_distribution<double> runif_c(0.0, 1.0);
    for (int i = 0; i < n_units; i++) {
      gp_coords[static_cast<size_t>(i) * 2 + 0] = runif_c(rng);
      gp_coords[static_cast<size_t>(i) * 2 + 1] = runif_c(rng);
    }
    const int nn = std::min(5, n_units - 1);
    NNGPNeighbors nb = build_nngp(gp_coords, n_units, nn);

    tulpa::CovType cov = tulpa::CovType::EXPONENTIAL;
    if (field == "gp_matern")    cov = tulpa::CovType::MATERN32;
    if (field == "gp_gaussian")  cov = tulpa::CovType::GAUSSIAN;
    if (field == "gp_spherical") cov = tulpa::CovType::SPHERICAL;

    std::vector<int> obs_to_loc(n_obs);
    for (int i = 0; i < n_obs; i++) obs_to_loc[i] = i % n_units;

    if (want_gp) {
      data.spatial_type = SpatialType::GP;
      data.has_gp = true;
      data.gp_collapsed = want_gp_collapsed;
      auto& gp = data.gp_data;
      gp.n_obs = n_units;
      gp.nn = nn;
      gp.coords = gp_coords;
      gp.nn_idx = nb.nn_idx;
      gp.nn_dist = nb.nn_dist;
      gp.nn_neighbor_dist = nb.nn_neighbor_dist;
      gp.nn_order = nb.nn_order;
      gp.nn_order_inv = nb.nn_order_inv;
      gp.obs_to_loc = obs_to_loc;
      gp.cov_type = cov;
      gp.nu = 1.5;
      gp.shared = true;

      data.gp_parameterization = want_gp_nc ? 1 : 0;
      data.gp_sigma2_prior_U = 1.0;
      data.gp_sigma2_prior_alpha = 0.01;
      data.gp_range_prior.type = tulpa::RangePriorType::UNIFORM;
      data.gp_range_prior.lower = 0.01;
      data.gp_range_prior.upper = 100.0;
    } else {
      data.spatial_type = SpatialType::MULTISCALE_GP;
      data.has_multiscale_gp = true;
      data.msgp_is_hsgp = true;
      auto& ms = data.multiscale_gp_data;
      ms.n_obs = n_obs;
      ms.shared = true;
      ms.sampler = tulpa::MSGPSampler::AUTO;

      std::vector<double> hs_coords(static_cast<size_t>(n_obs) * 2);
      std::uniform_real_distribution<double> runif_h(0.0, 1.0);
      for (int i = 0; i < n_obs; i++) {
        hs_coords[static_cast<size_t>(i) * 2 + 0] = runif_h(rng);
        hs_coords[static_cast<size_t>(i) * 2 + 1] = runif_h(rng);
      }
      hsgp_basis(hs_coords, n_obs, 3, 1.5, /*shared=*/true,
                                 data.msgp_hsgp_data);
      data.ms_log_ls_local_mean = std::log(0.1);
      data.ms_log_ls_local_sd = 0.5;
      data.ms_log_ls_regional_mean = std::log(1.0);
      data.ms_log_ls_regional_sd = 0.5;
    }
    data.n_spatial_units = 0;
    data.bym2_scale_factor = 1.0;
  } else {
    data.spatial_type = SpatialType::NONE;
    data.n_spatial_units = 0;
    data.bym2_scale_factor = 1.0;
  }

  if (want_svc) {
    std::vector<double> coords(static_cast<size_t>(n_obs) * 2);
    std::uniform_real_distribution<double> runif_c(0.0, 1.0);
    for (int i = 0; i < n_obs; i++) {
      coords[static_cast<size_t>(i) * 2 + 0] = runif_c(rng);
      coords[static_cast<size_t>(i) * 2 + 1] = runif_c(rng);
    }
    auto& svc = data.svc_data;
    const int n_svc = (field == "svc_nc") ? 2 : 1;
    svc.n_obs = n_obs;
    svc.n_svc = n_svc;
    svc.shared = true;
    svc.coords = coords;
    svc.svc_indices.clear();
    for (int j = 0; j < n_svc; j++) svc.svc_indices.push_back(2 - n_svc + j);
    svc.X_svc.resize(static_cast<size_t>(n_obs) * n_svc);
    for (int i = 0; i < n_obs; i++) {
      for (int j = 0; j < n_svc; j++) {
        svc.X_svc[static_cast<size_t>(i) * n_svc + j] =
            data.processes[0].X_flat[static_cast<size_t>(i) * 2 + svc.svc_indices[j]];
      }
    }
    svc.cov_type = tulpa::CovType::EXPONENTIAL;

    data.has_svc = true;
    data.svc_is_hsgp = want_svc_hsgp;
    if (want_svc_hsgp) {
      svc.nn = 0;
      data.svc_hsgp_m_per_dim = 3;
      data.svc_hsgp_boundary_factor = 1.5;
      hsgp_basis(coords, n_obs, 3, 1.5, /*shared=*/true,
                                 data.svc_hsgp_data);
    } else {
      const int nn = std::min(5, n_obs - 1);
      NNGPNeighbors nb = build_nngp(coords, n_obs, nn);
      svc.nn = nn;

      svc.nn_idx = nb.nn_idx;
      svc.nn_dist = nb.nn_dist;
      svc.nn_order = nb.nn_order;
      svc.nn_order_inv = nb.nn_order_inv;
      data.svc_parameterization = want_svc_nc ? 1 : 0;

    }
    data.svc_sigma_prior_U = 1.0;
    data.svc_sigma_prior_alpha = 0.01;
    data.svc_range_prior = 1;
    data.svc_range_lower = 0.001;
    data.svc_range_upper = 100.0;
  }

  if (want_latent) {
    const bool two = (field == "latent2" || field == "latent2_first_zero" ||
                      field == "stgp_latent2");
    data.has_latent = true;
    data.latent_n_factors = two ? 2 : 1;
    data.latent_shared = (field != "latent_num");
    data.latent_constraint = (field == "latent2_first_zero") ? 1 : 0;
    data.latent_sigma_prior_rate = 1.0;
    data.latent_offdiag_sd = 0.7;
  }

  if (want_tvc) {
    auto& tvc = data.tvc_data;
    tvc.n_obs = n_obs;
    tvc.n_times = n_times;
    tvc.n_tvc = 1;
    tvc.n_groups = 1;

    tvc.structure = (field == "tvc_iid") ? tulpa::TemporalType::IID
                  : (field == "tvc_rw2") ? tulpa::TemporalType::RW2
                  : (field == "tvc_ar1") ? tulpa::TemporalType::AR1
                                          : tulpa::TemporalType::RW1;
    tvc.shared = true;
    tvc.cyclic = false;
    tvc.time_index.resize(n_obs);
    tvc.group_index.assign(n_obs, 1);
    tvc.tvc_indices.assign(1, 1);
    tvc.X_tvc.resize(n_obs);
    for (int i = 0; i < n_obs; i++) {
      tvc.time_index[i] = (i % n_times) + 1;
      tvc.X_tvc[i] = data.processes[0].X_flat[static_cast<size_t>(i) * 2 + 1];
    }
    data.has_tvc = true;
    data.tvc_sigma_prior_U = 1.0;
    data.tvc_sigma_prior_alpha = 0.01;
  }

  if (want_temporal_gp) {

    data.temporal_type = TemporalType::GP;
    data.has_temporal_gp = true;
    data.n_times = n_times;
    data.n_temporal_groups = 1;
    data.n_temporal_params = n_times;
    data.temporal_cyclic = false;
    data.sharing.temporal[1] = temporal_shared;
    data.temporal_time_idx.resize(n_obs);
    data.temporal_group_idx.assign(n_obs, 1);
    for (int i = 0; i < n_obs; i++) data.temporal_time_idx[i] = (i % n_times) + 1;

    auto& tgp = data.temporal_gp_data;
    tgp.n_obs = n_times;
    tgp.n_groups = 1;
    tgp.time_values.resize(n_times);
    double t_val = 0.0;
    for (int t = 0; t < n_times; t++) {
      t_val += 0.6 + 0.4 * static_cast<double>((t * 7) % 5) / 4.0;
      tgp.time_values[t] = t_val;
    }
    using tulpa::TemporalCovType;
    const auto has = [&](const char* s) { return field.find(s) != std::string::npos; };
    tgp.cov_type = has("matern32") ? TemporalCovType::MATERN
                 : has("matern52") ? TemporalCovType::MATERN
                 : has("gaussian") ? TemporalCovType::GAUSSIAN
                 : has("periodic") ? TemporalCovType::PERIODIC
                 : TemporalCovType::EXPONENTIAL;

    tgp.nu = has("matern52") ? 2.5 : 1.5;
    tgp.period = 2.5;
    tgp.shared = temporal_shared;
    data.temporal_gp_sigma2_prior_U = 1.0;
    data.temporal_gp_sigma2_prior_alpha = 0.01;
    data.temporal_gp_phi_prior_lower = 0.01;
    data.temporal_gp_phi_prior_upper = 10.0;
    data.temporal_gp_parameterization = has("centered") ? 0 : 1;
  } else if (want_rw1 || want_rw2 || want_temporal_ar1 || want_temporal_iid) {
    data.temporal_type = want_temporal_ar1  ? TemporalType::AR1
                       : want_temporal_iid  ? TemporalType::IID
                       : want_rw2           ? TemporalType::RW2
                                            : TemporalType::RW1;
    data.temporal_parameterization =
        (field == "temporal_ar1_nc" || field == "rw1_nc" || field == "rw2_nc" ||
         field == "rw1_cyclic_nc" || field == "rw1_nc_panel" ||
         field == "rw2_nc_panel" || field == "rw1_cyclic_nc_panel") ? 1 : 0;
    data.n_times = n_times;

    const bool temporal_panel =
        (field == "rw1_panel" || field == "rw1_nc_panel" ||
         field == "rw2_nc_panel" || field == "rw1_cyclic_nc_panel");
    data.n_temporal_groups = temporal_panel ? 3 : 1;
    data.n_temporal_params = n_times * data.n_temporal_groups;

    data.temporal_cyclic =
        (field == "rw1_cyclic" || field == "rw1_cyclic_nc" ||
         field == "rw1_cyclic_nc_panel");
    data.sharing.temporal[1] = temporal_shared;
    data.temporal_time_idx.resize(n_obs);
    data.temporal_group_idx.assign(n_obs, 1);
    for (int i = 0; i < n_obs; i++) {
      data.temporal_time_idx[i] = (i % n_times) + 1;
      data.temporal_group_idx[i] =
          (i / n_times) % data.n_temporal_groups + 1;
    }
  } else {
    data.temporal_type = TemporalType::NONE;
    data.n_times = 0;
    data.n_temporal_groups = 0;
    data.n_temporal_params = 0;
  }

  if (want_ms) {
    auto& mst = data.multiscale_temporal_data;
    mst.n_times = n_times;
    mst.n_groups = 1;
    mst.n_obs = n_obs;

    const bool ms_rw2 = (field == "ms_temporal_rw2" ||
                         field == "ms_temporal_rw2_nc");
    const bool ms_seasonal = (field == "ms_temporal_seasonal" ||
                              field == "ms_temporal_seasonal_nc");
    mst.trend_type = ms_rw2 ? tulpa::TemporalType::RW2
                            : tulpa::TemporalType::RW1;
    mst.seasonal_period = ms_seasonal ? 4 : 0;
    mst.short_term_type = tulpa::TemporalType::AR1;
    mst.shared = true;
    const bool noncentered = (field == "ms_temporal_nc" ||
                       field == "ms_temporal_rw2_nc" ||
                       field == "ms_temporal_seasonal_nc");
    data.ms_trend_parameterization = data.ms_seasonal_parameterization =
      data.ms_short_parameterization = noncentered ? 1 : 0;
    mst.time_index.resize(n_obs);
    mst.group_index.assign(n_obs, 1);
    for (int i = 0; i < n_obs; i++) mst.time_index[i] = (i % n_times) + 1;
    data.has_multiscale_temporal = true;
    tulpa::prepare_field(data,tulpa::FieldSetup::MULTISCALE_TEMPORAL);
  }

  if (want_st || want_st4 || want_st2 || want_st3) {
    auto& st = data.spatiotemporal_data;
    st.type = want_st4 ? tulpa::STType::TYPE_IV
            : want_st2 ? tulpa::STType::TYPE_II
            : want_st3 ? tulpa::STType::TYPE_III
                       : tulpa::STType::TYPE_I;
    st.shared = true;
    st.n_spatial = n_units;
    st.n_times = n_times;
    st.n_params = n_units * n_times;
    st.s_idx.resize(n_obs);
    st.t_idx.resize(n_obs);
    st.st_flat.resize(n_obs);
    for (int i = 0; i < n_obs; i++) {
      const int s = (i % n_units) + 1;
      const int t = (i % n_times) + 1;
      st.s_idx[i] = s;
      st.t_idx[i] = t;
      st.st_flat[i] = (s - 1) * n_times + t;
    }
    st.spatial_is_gp = false;
    st.spatial_proper = false;
    st.bym2_scale = 1.0;
    if (want_st4 || want_st3) {

      build_grid_adjacency(n_units, st.adj_row_ptr, st.adj_col_idx, st.n_neighbors);
      for (int& j : st.adj_col_idx) j += 1;

      st.temporal_type = (field == "st4_ar1" || field == "st4_ar1_nc")
                            ? tulpa::TemporalType::AR1
                        : (field == "st4_rw2" || field == "st4_rw2_nc" ||
                           field == "st4_rw2_cyclic" || field == "st4_rw2_cyclic_nc")
                            ? tulpa::TemporalType::RW2
                            : tulpa::TemporalType::RW1;
      st.temporal_cyclic = field == "st4_rw2_cyclic" || field == "st4_rw2_cyclic_nc";
    } else if (want_st2) {

      st.temporal_type = (field == "st2_ar1" || field == "st2_ar1_nc")
                             ? tulpa::TemporalType::AR1
                       : (field == "st2_rw2" || field == "st2_rw2_nc")
                             ? tulpa::TemporalType::RW2
                             : tulpa::TemporalType::RW1;
      st.temporal_cyclic = (field == "st2_cyclic" || field == "st2_cyclic_nc");
    } else {
      st.temporal_type = tulpa::TemporalType::NONE;
      st.temporal_cyclic = false;
    }
    st.sigma2_prior_U = 1.0;
    st.sigma2_prior_alpha = 0.01;
    data.has_spatiotemporal = true;
    data.st_is_hsgp = false;

    data.st_parameterization =
        (field.size() > 3 && field.compare(field.size() - 3, 3, "_nc") == 0) ? 1 : 0;
  }

  if (want_stgp) {
    auto& st = data.spatiotemporal_data;
    st.type = (field == "stgp_gneiting" || field == "stgp_gneiting_nc" ||
               field == "stgp_gneiting_collapsed")
                  ? tulpa::STType::NONSEP_GP
                  : tulpa::STType::SEPARABLE;
    st.shared = true;
    st.n_spatial = n_units;
    st.n_times = n_times;
    st.n_params = n_units * n_times;
    st.s_idx.resize(n_obs);
    st.t_idx.resize(n_obs);
    st.st_flat.resize(n_obs);
    for (int i = 0; i < n_obs; i++) {
      const int s = (i % n_units) + 1;
      const int t = (i % n_times) + 1;
      st.s_idx[i] = s;
      st.t_idx[i] = t;
      st.st_flat[i] = (s - 1) * n_times + t;
    }
    st.spatial_is_gp = true;
    st.spatial_proper = false;
    st.bym2_scale = 1.0;

    st.temporal_type = tulpa::TemporalType::NONE;
    st.temporal_cyclic = false;
    st.sigma2_prior_U = 1.0;
    st.sigma2_prior_alpha = 0.01;

    std::vector<double> unit_coords(static_cast<size_t>(n_units) * 2);
    std::uniform_real_distribution<double> runif_st(0.0, 1.0);
    for (int s = 0; s < n_units; s++) {
      unit_coords[static_cast<size_t>(s) * 2 + 0] = runif_st(rng);
      unit_coords[static_cast<size_t>(s) * 2 + 1] = runif_st(rng);
    }
    const int ST_n = st.n_params;
    st.coords.assign(static_cast<size_t>(ST_n) * 2, 0.0);
    st.time_values.assign(ST_n, 0.0);
    const double t_scale = (n_times > 1) ? 1.0 / (n_times - 1) : 0.0;
    for (int s = 0; s < n_units; s++) {
      for (int t = 0; t < n_times; t++) {
        const size_t k = static_cast<size_t>(s) * n_times + t;
        st.coords[k * 2 + 0] = unit_coords[static_cast<size_t>(s) * 2 + 0];
        st.coords[k * 2 + 1] = unit_coords[static_cast<size_t>(s) * 2 + 1];
        st.time_values[k] = t * t_scale;
      }
    }

    st.nn = std::min(5, ST_n - 1);
    STNNGPNeighbors stnb = build_nngp_st(st.coords, st.time_values, ST_n, st.nn);
    st.nn_idx = stnb.nn_idx;
    st.nn_dist_space = stnb.nn_dist_space;
    st.nn_dist_time = stnb.nn_dist_time;
    st.nn_order = stnb.nn_order;
    st.nn_order_inv = stnb.nn_order_inv;

    st.cov_space = (field == "stgp_matern") ? tulpa::CovType::MATERN32
                                            : tulpa::CovType::EXPONENTIAL;
    st.cov_time = st.cov_space;
    st.nonsep_type = (field == "stgp_gneiting" || field == "stgp_gneiting_nc" ||
                      field == "stgp_gneiting_collapsed")
                         ? tulpa::NonsepType::GNEITING
                         : tulpa::NonsepType::PRODUCT;

    data.has_spatiotemporal = true;
    data.st_is_hsgp = false;

    const auto ends_with = [&](const char* suffix) {
      const std::string sfx(suffix);
      return field.size() > sfx.size() &&
             field.compare(field.size() - sfx.size(), sfx.size(), sfx) == 0;
    };
    data.st_parameterization = ends_with("_nc") ? 1 : ends_with("_collapsed") ? 2 : 0;
  }

  data.zi_type = zi_type;
  data.p_zi = has_zi ? 2 : 0;
  data.zi_prior_sd = 1.0;
  data.p_oi = has_oi ? 2 : 0;
  data.oi_prior_sd = 1.0;

  if (want_rsr) data.spatial_restriction = numerator_design_restriction(data);
  if (want_rtr) { auto restriction=numerator_design_restriction(data); data.set_temporal_restriction(restriction.basis,restriction.rank); }
  tulpa::prepare_field(data,tulpa::FieldSetup::TEMPORAL_RESTRICTION);
  if (data.has_spatiotemporal && !data.st_is_hsgp && data.st_parameterization != 2)
    tulpa::setup_st_spatial_partition(data.spatiotemporal_data);

  if (data.n_spatial_units) {
    data.set_spatial_adjacency(data.n_spatial_units,data.adj_row_ptr,data.adj_col_idx,data.n_neighbors);
    data.spatial_component_centering = true;
  }
  data.temporal_group_centering = true;
  data.temporal_gp_relative_nugget = 1e-6;
  data.st_margin_centering = data.has_spatiotemporal &&
    data.spatiotemporal_data.type != tulpa::STType::SEPARABLE &&
    data.spatiotemporal_data.type != tulpa::STType::NONSEP_GP && !data.st_is_hsgp;
  if(data.st_parameterization==2) data.st_parameterization=1;
  if (fixed_exposure) {
    data.response.exposure=data.response.y_denom_cont;
    data.processes[0].offset.resize(n_obs);
  }
  return data;
}

inline std::vector<double> draw_params(const Fixture& data, const ParamLayout& layout,
                                int n_params,
                                unsigned int seed) {
  std::mt19937 rng(seed + 1000u);
  std::normal_distribution<double> rnorm(0.0, 0.35);
  std::vector<double> params(n_params);
  for (int i = 0; i < n_params; i++) params[i] = rnorm(rng);

  const double log_phi_center = std::log(0.1);
  for (int idx : {layout.log_phi_gp_idx, layout.log_phi_gp_local_idx,
                  layout.log_phi_gp_regional_idx,
                  layout.log_phi_st_space_idx, layout.log_phi_st_time_idx}) {
    if (idx >= 0 && idx < n_params) params[idx] += log_phi_center;
  }

  if (layout.log_phi_svc_start >= 0) {
    for (int j = layout.log_phi_svc_start; j < layout.log_phi_svc_end; j++) {
      if (j >= n_params) continue;
      if (data.svc_is_hsgp) {
        params[j] += log_phi_center;
      } else {
        const double range=std::exp(params[j]+log_phi_center);
        params[j]=std::log(range-data.svc_range_lower)-std::log(data.svc_range_upper-range);
      }
    }
  }
  return params;
}


}
#endif
