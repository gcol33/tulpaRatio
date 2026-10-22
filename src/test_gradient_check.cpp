// Finite-difference verification for the analytic gradient.
//
// The log posterior and the analytic gradient are written out separately, in
// several copies, for each field type. When a prior term is added to one and
// not the other, NUTS samples a density whose gradient points elsewhere; the
// chain still runs and still returns numbers, so nothing downstream fails.
// These helpers build a small model, evaluate both, and compare.

#include <Rcpp.h>
#include <vector>
#include <chrono>
#include <cmath>
#include <random>
#include <algorithm>
#include <utility>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "hmc_sampler.h"
#include "hmc_temporal.h"
#include "log_post_impl.h"

using namespace ratiod_hmc;

namespace {

// Rook-and-diagonal adjacency on the smallest square grid holding n units,
// emitted as CSR. Matches the structure the benchmark builds in R.
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

// Nearest-neighbour sets for an NNGP, mirroring compute_nngp_neighbors() in
// R/spatial.R: order the locations by coordinate, then give each one the k
// nearest among its predecessors in that order. nn_idx is 1-based within the
// ordering with 0 marking an absent neighbour, and nn_order / nn_order_inv are
// 0-based, which is the convention the sampler entry converts R's inputs to.
struct NNGPNeighbors {
  int k = 0;
  std::vector<double> coords;          // reordered coordinates
  std::vector<int> nn_idx;
  std::vector<double> nn_dist;
  std::vector<double> nn_neighbor_dist;
  std::vector<int> nn_order;
  std::vector<int> nn_order_inv;
};

NNGPNeighbors build_nngp(const std::vector<double>& coords, int n, int k) {
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
      out.nn_idx[static_cast<size_t>(i) * k + c] = cand[c].second + 1;  // 1-based
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
    out.nn_order[i] = order[i];       // 0-based
    out.nn_order_inv[order[i]] = i;
  }
  return out;
}

// Every field name make_model() builds a structure for. A name outside this
// list would fall through every branch and yield a model with no structured
// field at all, so the check would pass while testing nothing.
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
  "st2_rw2_nc", "st2_ar1_nc", "st2_cyclic", "st2_cyclic_nc", "st3_nc",
  "stgp", "stgp_matern", "stgp_gneiting", "stgp_latent",
  "stgp_nc", "stgp_gneiting_nc", "stgp_collapsed", "stgp_gneiting_collapsed",
  "icar_collapsed", "bym2_collapsed",
  "icar_collapsed_rw1", "bym2_collapsed_rw1",
  "hsgp",
  "ms_temporal_nc", "ms_temporal_rw2", "ms_temporal_rw2_nc",
  "ms_temporal_seasonal", "ms_temporal_seasonal_nc",
  "tvc", "tvc_iid", "tvc_ar1",
  "re", "re_crossed", "re_slopes", "re_slopes_corr",
  "gp", "gp_matern", "gp_gaussian", "gp_spherical",
  "gp_nc", "gp_collapsed", "gp_temporal",
  "msgp_temporal", "msgp_hsgp",
  "svc", "svc_hsgp", "svc_nc",
  "temporal_gp", "ms_temporal", "latent", "latent2", "latent_num",
  "latent2_first_zero", "stgp_latent2",
  // Combinations a specialized gradient used to be selected for while writing
  // none of the second block. Each is routed by its feature mask now, and each
  // of these is what says the function it lands on writes both blocks.
  "gp_st4", "gp_stgp", "gp_temporal_st4", "msgp_st4", "gp_collapsed_st4",
  "icar_collapsed_st4", "bym2_collapsed_st4", "tvc_st4", "temporal_gp_st4",
  "gp_tgp", "svc_ms", "svc_ms_nc", "gp_slopes", "gp_slopes_corr", "gp_crossed",
  // The temporal GP under each kernel and in either coordinate, alone and
  // beside a spatial GP (which sends it to the composite).
  "temporal_gp_centered", "temporal_gp_matern32", "temporal_gp_matern52",
  "temporal_gp_gaussian", "temporal_gp_periodic",
  "temporal_gp_matern32_centered", "temporal_gp_periodic_centered",
  "gp_tgp_matern32", "gp_tgp_centered", "gp_tgp_gaussian_centered"
};

// The spatial fields make_model() also builds under restricted spatial
// regression, named with the suffix `_rsr`.
const char* const RSR_FIELDS[] = {"icar", "bym2", "hsgp", "gp"};
const char* const RSR_SUFFIX = "_rsr";

bool is_rsr_field(const std::string& field) {
  const std::string suffix(RSR_SUFFIX);
  return field.size() > suffix.size() &&
         field.compare(field.size() - suffix.size(), suffix.size(), suffix) == 0;
}

bool is_known_field(const std::string& field) {
  if (is_rsr_field(field)) {
    const std::string base = field.substr(0, field.size() - std::string(RSR_SUFFIX).size());
    for (const char* known : RSR_FIELDS) {
      if (base == known) return true;
    }
    return false;
  }
  for (const char* known : KNOWN_FIELDS) {
    if (field == known) return true;
  }
  return false;
}

// The spatiotemporal counterpart of build_nngp, mirroring
// compute_st_nngp_neighbors() in R/spatiotemporal.R: order the field's entries
// by time and then by coordinate, and give each one the k nearest among its
// predecessors under the joint space-time distance. The two component
// distances are kept apart, the covariance reading each on its own range.
struct STNNGPNeighbors {
  int k = 0;
  std::vector<int> nn_idx;
  std::vector<double> nn_dist_space;
  std::vector<double> nn_dist_time;
  std::vector<int> nn_order;
  std::vector<int> nn_order_inv;
};

STNNGPNeighbors build_nngp_st(const std::vector<double>& coords,
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

  // Separation between two positions IN THE ORDERING, read off the original
  // entries the way the density does.
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
      out.nn_idx[static_cast<size_t>(i) * k + c] = cand[c].second + 1;  // 1-based
      out.nn_dist_space[static_cast<size_t>(i) * k + c] = h;
      out.nn_dist_time[static_cast<size_t>(i) * k + c] = u;
    }
  }

  out.nn_order.resize(n);
  out.nn_order_inv.resize(n);
  for (int i = 0; i < n; i++) {
    out.nn_order[i] = order[i];       // 0-based
    out.nn_order_inv[order[i]] = i;
  }
  return out;
}

// Synthetic model with an intercept, one covariate, and whichever structured
// field is under test. `family` selects binomial (no denominator predictors)
// or one of the count families (a denominator with its own predictors), which
// is what exercises the paths that treat the two linear predictors
// differently, `temporal_shared` among them. `zi` attaches a zero-inflation or
// hurdle structure to the numerator, with its own two-column design matrix.
// Restricted spatial regression on the numerator's own design: the basis is
// the two orthonormalized columns (intercept, x), so the field reaches eta
// with both directions removed.
void restrict_to_numerator_design(ModelData& data) {
  const int n = data.N;
  std::vector<double> q0(n), q1(n);
  double n0 = 0.0;
  for (int i = 0; i < n; i++) {
    q0[i] = data.X_num_flat[static_cast<size_t>(i) * 2 + 0];
    n0 += q0[i] * q0[i];
  }
  n0 = std::sqrt(n0);
  double dot = 0.0;
  for (int i = 0; i < n; i++) {
    q0[i] /= n0;
    dot += q0[i] * data.X_num_flat[static_cast<size_t>(i) * 2 + 1];
  }
  double n1 = 0.0;
  for (int i = 0; i < n; i++) {
    q1[i] = data.X_num_flat[static_cast<size_t>(i) * 2 + 1] - dot * q0[i];
    n1 += q1[i] * q1[i];
  }
  n1 = std::sqrt(n1);
  data.has_rsr = true;
  data.rsr_rank = 2;
  data.rsr_basis.resize(static_cast<size_t>(n) * 2);
  for (int i = 0; i < n; i++) {
    data.rsr_basis[static_cast<size_t>(i) * 2 + 0] = q0[i];
    data.rsr_basis[static_cast<size_t>(i) * 2 + 1] = q1[i] / n1;
  }
}

ModelData make_model(const std::string& field_in, int n_obs, int n_units,
                     int n_times, unsigned int seed,
                     const std::string& family = "binomial",
                     bool temporal_shared = true,
                     const std::string& zi = "none") {
  const bool want_rsr = is_rsr_field(field_in);
  const std::string field = want_rsr
      ? field_in.substr(0, field_in.size() - std::string(RSR_SUFFIX).size())
      : field_in;
  if (!is_known_field(field_in)) {
    Rcpp::stop("cpp_gradient_check: unknown field '" + field_in +
               "'. A name make_model() does not build a structure for would "
               "yield a model with no structured field, and the check would "
               "pass while testing nothing.");
  }

  ModelData data;
  std::mt19937 rng(seed);
  std::normal_distribution<double> rnorm(0.0, 1.0);
  std::uniform_int_distribution<int> rtrials(10, 50);

  // beta_binomial reads the same trials and successes
  const bool is_binomial = (family == "binomial" || family == "beta_binomial");
  // The negative binomial denominators are counts; a known exposure is data
  // and carries no coefficients; the other families carry a continuous Gamma
  // denominator.
  const bool denom_is_count = (family == "negbin_negbin" || family == "poisson_negbin");
  const bool fixed_exposure = (family == "poisson_fixed" || family == "negbin_fixed" ||
                               family == "lognormal_fixed");
  // gamma_gamma and lognormal read a continuous numerator beside their
  // continuous denominator
  const bool num_is_continuous = (family == "lognormal_fixed" ||
                                  family == "gamma_gamma" ||
                                  family == "lognormal");
  const ZIType zi_type = ratiod_zi::parse_zi_type(zi);
  const bool has_zi = (zi_type != ZIType::NONE);
  const bool has_oi = (zi_type == ZIType::OI_BINOMIAL || zi_type == ZIType::ZOIB);

  // icar_ms / icar_st / bym2_ms / bym2_st reach the multi-feature gradient
  // paths, which carry the spatial field alongside another structure and are
  // therefore the ones that must agree with the log posterior about how that
  // field is identified.
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
  // st4: a spatiotemporal Type IV (Kronecker) interaction with NO accompanying
  // additive spatial or temporal field -- the only structured term is the
  // interaction itself. icar_st/bym2_st above exercise
  // compute_gradient_spatiotemporal_handcoded's multi-feature path but only
  // ever build a Type I interaction; this is the field that checks the
  // handcoded Type IV Kronecker block against finite differences.
  const bool want_st4 = (field == "st4" || field == "st4_nc" ||
                         field == "st4_ar1" || field == "st4_ar1_nc" ||
                         field == "gp_st4" || field == "gp_temporal_st4" ||
                         field == "msgp_st4" || field == "gp_collapsed_st4" ||
                         field == "icar_collapsed_st4" ||
                         field == "bym2_collapsed_st4" ||
                         field == "tvc_st4" || field == "temporal_gp_st4");
  // Type II (structured time within each spatial unit) and Type III
  // (structured space at each time point). Both reach the shared interaction
  // gradient through a different margin than Type IV does, and Type II is the
  // one whose RW2 and AR1 arms the composite gradient used to be missing.
  const bool want_st2 = (field == "st2" || field == "st2_rw2" ||
                         field == "st2_ar1" || field == "st2_nc" ||
                         field == "st2_rw2_nc" || field == "st2_ar1_nc" ||
                         field == "st2_cyclic" || field == "st2_cyclic_nc");
  const bool want_st3 = (field == "st3" || field == "st3_nc");
  // The GP interaction types, over the same S x T grid the Knorr-Held types
  // use. Each is a continuous covariance over that grid rather than a GMRF
  // stencil on it, so it reaches st_prior_grad.h's GP branch and the two
  // ranges the layout allocates alongside tau. stgp_latent pairs it with a
  // latent factor, which is what routes an interaction away from
  // compute_gradient_spatiotemporal_handcoded and into the composite -- the
  // two callers of that branch, both arbitrated.
  const bool want_stgp = (field == "stgp" || field == "stgp_matern" ||
                          field == "stgp_gneiting" || field == "stgp_latent" ||
                          field == "stgp_latent2" ||
                          field == "stgp_nc" || field == "stgp_gneiting_nc" ||
                          field == "stgp_collapsed" ||
                          field == "stgp_gneiting_collapsed" ||
                          field == "gp_stgp");
  // icar_collapsed_rw1 / bym2_collapsed_rw1 pair a collapsed spatial field
  // with a companion temporal RW1 term -- the combination the specialized
  // collapsed gradient has to carry into its inner Laplace.
  // Two grid components and an island: centred per component, the island on
  // a proper prior, BYM2 with a different precision multiplier per component.
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
  // Proper CAR (rho estimated). can_use_analytical_gradient() excludes it, so
  // it never reaches compute_gradient_analytical, but GF_AREAL covers a proper
  // CAR field as much as an intrinsic one: paired with a multi-scale temporal
  // block or a spatiotemporal interaction it reaches the two specialized
  // functions those masks admit, not only the composite. Those are the paths
  // that have to agree with the log posterior about whether the field's mean
  // is removed, and car_proper_ms / car_proper_st are what checks them.
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
  // The temporal block's own AR1 arm, which carries a rho of its own. rw1 and
  // rw2 reach the same kernels without it. temporal_ar1_nc samples the same
  // field in its non-centred coordinate, where the block holds the N(0, I)
  // innovations and both hyperparameters reach the effects through the
  // transform rather than through a quadratic form.
  const bool want_temporal_ar1 = (field == "temporal_ar1" ||
                                  field == "temporal_ar1_nc");
  // The unstructured arm: a diagonal, full-rank, proper temporal prior, which
  // is the one branch with no differencing stencil and no sum-to-zero pin.
  const bool want_temporal_iid = (field == "temporal_iid");
  const bool want_hsgp = (field == "hsgp");
  const bool want_tvc  = (field == "tvc" || field == "tvc_iid" ||
                          field == "tvc_ar1" || field == "tvc_st4");
  // The NNGP / spectral fields. Each reaches a gradient kernel of its own that
  // no areal field visits: see resolve_gradient_fn in hmc_sampler.cpp.
  const bool want_gp_collapsed = (field == "gp_collapsed" ||
                                  field == "gp_collapsed_st4");
  // spatial_gp(parameterization = "noncentered"): the sampled parameters are
  // z ~ N(0, I) and the field is w = L(sigma2, phi) z, so both hyperparameters
  // reach eta through the transform as well as through the prior.
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
  // The multi-scale field: two scales on one shared spectral basis, with a PC
  // prior on each scale's sigma and a LogNormal on its lengthscale.
  // msgp_temporal pairs it with a temporal walk, msgp_st4 with a space-time
  // interaction, which no specialized function carries.
  const bool want_msgp = (field == "msgp_temporal" || field == "msgp_st4" ||
                          field == "msgp_hsgp");
  const bool want_svc_hsgp = (field == "svc_hsgp");
  // svc_nc / svc_ms_nc sample each term in the non-centred coordinate;
  // svc_nc carries two terms so each one's transform reads its own workspace.
  const bool want_svc_nc = (field == "svc_nc" || field == "svc_ms_nc");
  const bool want_svc = (field == "svc" || want_svc_hsgp || field == "svc_ms" ||
                         want_svc_nc);
  // temporal_gp* and gp_tgp*: the kernel and the coordinate are read off the
  // name's suffixes below.
  const bool want_temporal_gp = (field.rfind("temporal_gp", 0) == 0 ||
                                 field.rfind("gp_tgp", 0) == 0);
  const bool want_latent = (field == "latent" || field == "stgp_latent" ||
                             field == "latent2" || field == "latent_num" ||
                             field == "latent2_first_zero" ||
                             field == "stgp_latent2");
  // Random slopes route through a parameter layout the single-term path does
  // not use: one sigma per coefficient, and for the correlated variant a
  // tanh-Cholesky factor with an LKJ prior and re = diag(sigma) L z.
  const bool want_re_slopes = (field == "re_slopes" || field == "re_slopes_corr" ||
                               field == "gp_slopes" || field == "gp_slopes_corr");
  const bool want_re = (field == "re" || field == "re_crossed" ||
                        field == "gp_crossed" || want_re_slopes);

  data.N = n_obs;
  data.model_type = parse_model_type(family);
  data.p_num = 2;
  data.p_denom = (is_binomial || fixed_exposure) ? 0 : 2;
  data.n_threads = 1;
  data.n_cores = 1;

  data.sigma_beta = 10.0;
  data.sigma_re_scale = 1.0;
  data.phi_prior_shape = 1.0;
  data.phi_prior_rate = 0.01;
  data.tau_spatial_shape = 1.0;
  data.tau_spatial_rate = 0.01;
  data.tau_temporal_shape = 1.0;
  data.tau_temporal_rate = 0.01;

  data.X_num_flat.resize(static_cast<size_t>(n_obs) * 2);
  data.y_num.resize(n_obs);
  // y_denom is sized for every family: the continuous-denominator families
  // carry a placeholder of 1, matching what the backend passes down, and the
  // lgamma/lchoose precompute reads it unconditionally.
  data.y_denom.assign(n_obs, 1);
  if (num_is_continuous) data.y_num_cont.resize(n_obs);
  if (fixed_exposure) {
    data.y_denom_cont.resize(n_obs);
  } else if (!is_binomial) {
    data.X_denom_flat.resize(static_cast<size_t>(n_obs) * 2);
    if (denom_is_count) {
      data.y_denom.assign(n_obs, 0);
    } else {
      data.y_denom_cont.resize(n_obs);
    }
  }
  if (has_zi) data.X_zi_flat.resize(static_cast<size_t>(n_obs) * 2);
  if (has_oi) data.X_oi_flat.resize(static_cast<size_t>(n_obs) * 2);
  std::uniform_real_distribution<double> runif(0.0, 1.0);
  for (int i = 0; i < n_obs; i++) {
    const double x = rnorm(rng);
    data.X_num_flat[static_cast<size_t>(i) * 2 + 0] = 1.0;
    data.X_num_flat[static_cast<size_t>(i) * 2 + 1] = x;
    if (is_binomial) {
      const int trials = rtrials(rng);
      const double p = 1.0 / (1.0 + std::exp(-(0.5 + 0.3 * x)));
      std::binomial_distribution<int> rbinom(trials, p);
      data.y_denom[i] = trials;
      data.y_num[i] = rbinom(rng);
    } else if (fixed_exposure) {
      const double exposure = 0.5 + 2.5 * runif(rng);
      data.y_denom_cont[i] = exposure;
      if (num_is_continuous) {
        data.y_num_cont[i] = exposure * std::exp(0.5 + 0.3 * x + 0.4 * rnorm(rng));
        data.y_num[i] = 0;
      } else {
        std::poisson_distribution<int> rpois(exposure * std::exp(0.5 + 0.3 * x));
        data.y_num[i] = rpois(rng);
      }
    } else {
      const double z = rnorm(rng);
      data.X_denom_flat[static_cast<size_t>(i) * 2 + 0] = 1.0;
      data.X_denom_flat[static_cast<size_t>(i) * 2 + 1] = z;
      if (num_is_continuous) {
        data.y_num_cont[i] = std::exp(0.5 + 0.3 * x + 0.4 * rnorm(rng));
        data.y_num[i] = 0;
      } else {
        std::poisson_distribution<int> rpois(std::exp(0.5 + 0.3 * x));
        data.y_num[i] = rpois(rng);
      }
      if (denom_is_count) {
        std::poisson_distribution<int> rpois_d(std::exp(0.4 + 0.2 * z));
        data.y_denom[i] = rpois_d(rng);
      } else {
        std::gamma_distribution<double> rgamma(2.0, std::exp(0.4 + 0.2 * z) / 2.0);
        data.y_denom_cont[i] = std::max(1e-3, rgamma(rng));
      }
    }
    if (has_oi) {
      data.X_oi_flat[static_cast<size_t>(i) * 2 + 0] = 1.0;
      data.X_oi_flat[static_cast<size_t>(i) * 2 + 1] = x;
    }
    if (has_zi) {
      data.X_zi_flat[static_cast<size_t>(i) * 2 + 0] = 1.0;
      data.X_zi_flat[static_cast<size_t>(i) * 2 + 1] = x;
      // Both arms of every ZI and hurdle density have to be represented in the
      // data, or a term that only appears at y = 0 (or only at y > 0) never
      // enters the sum and the check cannot see it. Zeroing at a rate of about
      // 0.4 leaves both well populated.
      if (runif(rng) < 1.0 / (1.0 + std::exp(-(-0.4 + 0.6 * x)))) data.y_num[i] = 0;
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
        // Offset the second term so the two groupings cross rather than coincide.
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
              data.X_num_flat[static_cast<size_t>(i) * 2 + 1];
        }
      }
    }
  } else {
    // The constrained field is what is under test.
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
      // Units [0, h) and [h, n - 1) are two grids, unit n - 1 an island
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
      auto& comps = data.areal_components;
      comps.block_of.assign(n_units, -1);
      comps.members.assign(2, std::vector<int>());
      for (int i = 0; i < n_units - 1; i++) {
        comps.block_of[i] = (i < h) ? 0 : 1;
        comps.members[comps.block_of[i]].push_back(i);
      }
      data.areal_node_prec.assign(n_units, 1.0);
      if (want_bym2) {
        for (int i = h; i < n_units - 1; i++) data.areal_node_prec[i] = 1.7;
        data.areal_node_prec[n_units - 1] = 0.6;
      }
    }
    data.spatial_group.resize(n_obs);
    for (int i = 0; i < n_obs; i++) data.spatial_group[i] = (i % n_units) + 1;
    data.bym2_scale_factor = 1.0;
    // Collapsed: the field is marginalized out, so the layout allocates no
    // phi/theta and the density carries a Laplace correction at the mode.
    data.icar_collapsed = (field == "icar_collapsed" || field == "icar_collapsed_rw1" ||
                           field == "icar_collapsed_st4");
    data.bym2_collapsed = (field == "bym2_collapsed" || field == "bym2_collapsed_rw1" ||
                           field == "bym2_collapsed_st4");
    // rho in the interior of (0, 1), away from both boundaries the
    // finite-difference step could clip against.
    data.car_rho_lower = 0.0;
    data.car_rho_upper = 1.0;
    data.car_rho_prior_a = 1.0;
    data.car_rho_prior_b = 1.0;
    // eta reads the centred field and the prior reads the raw one, which for a
    // full-rank Q(rho) are two different quadratic forms; car_proper_raw is the
    // same model with the mean left in the likelihood.
    data.car_center = (field != "car_proper_raw");
  } else if (want_hsgp) {
    // HSGP is an observation-level spectral field: m_per_dim^2 basis
    // coefficients plus a variance and a lengthscale. Kept at m = 3 so the
    // finite-difference sweep stays cheap.
    data.spatial_type = SpatialType::HSGP;
    data.has_hsgp = true;
    std::vector<double> coords(static_cast<size_t>(n_obs) * 2);
    std::uniform_real_distribution<double> runif(0.0, 1.0);
    for (int i = 0; i < n_obs; i++) {
      coords[static_cast<size_t>(i) * 2 + 0] = runif(rng);
      coords[static_cast<size_t>(i) * 2 + 1] = runif(rng);
    }
    ratiod_hsgp::setup_hsgp_2d(coords, n_obs, 3, 1.5, /*shared=*/true,
                               data.hsgp_data);
    data.n_spatial_units = 0;
    data.bym2_scale_factor = 1.0;
  } else if (want_gp || want_msgp) {
    // NNGP field over n_units unique locations, with several observations per
    // location so obs_to_loc is exercised rather than being the identity.
    std::vector<double> gp_coords(static_cast<size_t>(n_units) * 2);
    std::uniform_real_distribution<double> runif_c(0.0, 1.0);
    for (int i = 0; i < n_units; i++) {
      gp_coords[static_cast<size_t>(i) * 2 + 0] = runif_c(rng);
      gp_coords[static_cast<size_t>(i) * 2 + 1] = runif_c(rng);
    }
    const int nn = std::min(5, n_units - 1);
    NNGPNeighbors nb = build_nngp(gp_coords, n_units, nn);

    ratiod_svc::CovType cov = ratiod_svc::CovType::EXPONENTIAL;
    if (field == "gp_matern")    cov = ratiod_svc::CovType::MATERN;
    if (field == "gp_gaussian")  cov = ratiod_svc::CovType::GAUSSIAN;
    if (field == "gp_spherical") cov = ratiod_svc::CovType::SPHERICAL;

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
      // spatial_gp() offers "centered", "noncentered" and "collapsed" and
      // defaults to centered, which is the parameterization these fields
      // build; gp_nc is the non-centred one.
      data.gp_parameterization = want_gp_nc ? 1 : 0;
      data.gp_sigma2_prior_U = 1.0;
      data.gp_sigma2_prior_alpha = 0.01;
      data.gp_phi_prior_lower = 0.01;
      data.gp_phi_prior_upper = 100.0;
    } else {
      data.spatial_type = SpatialType::MULTISCALE_GP;
      data.has_multiscale_gp = true;
      auto& ms = data.multiscale_gp_data;
      ms.n_obs = n_obs;
      ms.shared = true;
      ms.sampler = ratiod_gp::MSGPSampler::AUTO;

      // Both scales read one basis built over the observation locations, so
      // the field is m_total coefficients per scale.
      std::vector<double> hs_coords(static_cast<size_t>(n_obs) * 2);
      std::uniform_real_distribution<double> runif_h(0.0, 1.0);
      for (int i = 0; i < n_obs; i++) {
        hs_coords[static_cast<size_t>(i) * 2 + 0] = runif_h(rng);
        hs_coords[static_cast<size_t>(i) * 2 + 1] = runif_h(rng);
      }
      ratiod_hsgp::setup_hsgp_2d(hs_coords, n_obs, 3, 1.5, /*shared=*/true,
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

  // SVC: a spatially-varying slope on the covariate (and, for svc_nc, on the
  // intercept too), either as an NNGP field over the observation locations or
  // on an HSGP spectral basis.
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
            data.X_num_flat[static_cast<size_t>(i) * 2 + svc.svc_indices[j]];
      }
    }
    svc.cov_type = ratiod_svc::CovType::EXPONENTIAL;

    data.has_svc = true;
    data.svc_is_hsgp = want_svc_hsgp;
    if (want_svc_hsgp) {
      svc.nn = 0;
      data.svc_hsgp_m_per_dim = 3;
      data.svc_hsgp_boundary_factor = 1.5;
      ratiod_hsgp::setup_hsgp_2d(coords, n_obs, 3, 1.5, /*shared=*/true,
                                 data.svc_hsgp_data);
    } else {
      const int nn = std::min(5, n_obs - 1);
      NNGPNeighbors nb = build_nngp(coords, n_obs, nn);
      svc.nn = nn;
      // coords stay in their original order; nn_order maps into them
      svc.nn_idx = nb.nn_idx;
      svc.nn_dist = nb.nn_dist;
      svc.nn_order = nb.nn_order;
      svc.nn_order_inv = nb.nn_order_inv;
      data.svc_noncentered = want_svc_nc;
      data.svc_gp_view = ratiod_svc::make_svc_gp_view(svc);
    }
    data.svc_sigma2_prior_scale = 1.0;
    data.svc_phi_prior_lower = 0.001;
    data.svc_phi_prior_upper = 100.0;
  }

  // Latent factors: observation-level scores with process-specific loadings.
  // latent2 and stgp_latent2 fill the whole lower-triangular loading matrix,
  // latent_num loads on the numerator alone, and latent2_first_zero reads the
  // scores through the first-zero shift instead of centring.
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

  // TVC: one RW1 trajectory per (group, term) over the time axis.
  if (want_tvc) {
    auto& tvc = data.tvc_data;
    tvc.n_obs = n_obs;
    tvc.n_times = n_times;
    tvc.n_tvc = 1;
    tvc.n_groups = 1;
    // tvc_iid: TemporalType::IID is implemented in tvc_term_log_prior /
    // tvc_prior_gradients_ws and reachable from the C++ string parser.
    // temporal_tvc()'s match.arg in R does not offer it, so this harness is
    // where it is exercised.
    tvc.structure = (field == "tvc_iid") ? ratiod_temporal::TemporalType::IID
                  : (field == "tvc_ar1") ? ratiod_temporal::TemporalType::AR1
                                          : ratiod_temporal::TemporalType::RW1;
    tvc.shared = true;
    tvc.cyclic = false;
    tvc.time_index.resize(n_obs);
    tvc.group_index.assign(n_obs, 1);
    tvc.tvc_indices.assign(1, 1);
    tvc.X_tvc.resize(n_obs);
    for (int i = 0; i < n_obs; i++) {
      tvc.time_index[i] = (i % n_times) + 1;
      tvc.X_tvc[i] = data.X_num_flat[static_cast<size_t>(i) * 2 + 1];
    }
    data.has_tvc = true;
    data.tvc_tau_shape = 1.0;
    data.tvc_tau_rate = 0.01;
  }

  if (want_temporal_gp) {
    // A continuous-time GP over the distinct time instants, at irregular
    // spacing so the state-space recursion's dt varies from step to step.
    data.temporal_type = TemporalType::GP;
    data.has_temporal_gp = true;
    data.n_times = n_times;
    data.n_temporal_groups = 1;
    data.n_temporal_params = n_times;
    data.temporal_cyclic = false;
    data.temporal_shared = temporal_shared;
    data.temporal_time_idx.resize(n_obs);
    data.temporal_group_idx.assign(n_obs, 1);
    for (int i = 0; i < n_obs; i++) data.temporal_time_idx[i] = (i % n_times) + 1;

    auto& tgp = data.temporal_gp_data;
    tgp.n_times = n_times;
    tgp.n_groups = 1;
    tgp.time_values.resize(n_times);
    double t_val = 0.0;
    for (int t = 0; t < n_times; t++) {
      t_val += 0.6 + 0.4 * static_cast<double>((t * 7) % 5) / 4.0;
      tgp.time_values[t] = t_val;
    }
    using ratiod_temporal_gp::TemporalCovType;
    const auto has = [&](const char* s) { return field.find(s) != std::string::npos; };
    tgp.cov_type = has("matern32") ? TemporalCovType::MATERN32
                 : has("matern52") ? TemporalCovType::MATERN52
                 : has("gaussian") ? TemporalCovType::GAUSSIAN
                 : has("periodic") ? TemporalCovType::PERIODIC
                 : TemporalCovType::EXPONENTIAL;
    // Shorter than the span of the grid, so the periodic kernel wraps.
    tgp.period = 2.5;
    tgp.shared = temporal_shared;
    tgp.sigma2_prior_U = 1.0;
    tgp.sigma2_prior_alpha = 0.01;
    tgp.phi_lower = 0.01;
    tgp.phi_upper = 10.0;
    tgp.noncentered = !has("centered");
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
    // Panel data: G disconnected walks, each centred on its own with its own
    // augmented constant. The centred and non-centred coordinates are the same
    // model and pair against each other here.
    const bool temporal_panel =
        (field == "rw1_panel" || field == "rw1_nc_panel" ||
         field == "rw2_nc_panel" || field == "rw1_cyclic_nc_panel");
    data.n_temporal_groups = temporal_panel ? 3 : 1;
    data.n_temporal_params = n_times * data.n_temporal_groups;
    // The cyclic walk inverts B = C + 11'/n rather than a triangular stack, so
    // it is its own branch of the transform.
    data.temporal_cyclic =
        (field == "rw1_cyclic" || field == "rw1_cyclic_nc" ||
         field == "rw1_cyclic_nc_panel");
    data.temporal_shared = temporal_shared;
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

  // Multiscale temporal: trend plus a short-term AR1, which is what routes to
  // compute_gradient_ms_temporal_handcoded.
  if (want_ms) {
    auto& mst = data.multiscale_temporal_data;
    mst.n_times = n_times;
    mst.n_groups = 1;
    mst.n_obs = n_obs;
    // The trend's order and the seasonal arm decide which branch of the
    // transform runs: RW1 integrates once, RW2 twice and keeps a free linear
    // direction, and the seasonal arm is the cyclic case.
    const bool ms_rw2 = (field == "ms_temporal_rw2" ||
                         field == "ms_temporal_rw2_nc");
    const bool ms_seasonal = (field == "ms_temporal_seasonal" ||
                              field == "ms_temporal_seasonal_nc");
    mst.trend_type = ms_rw2 ? ratiod_temporal::TemporalType::RW2
                            : ratiod_temporal::TemporalType::RW1;
    mst.seasonal_period = ms_seasonal ? 4 : 0;
    mst.short_term_type = ratiod_temporal::TemporalType::AR1;
    mst.shared = true;
    mst.noncentered = (field == "ms_temporal_nc" ||
                       field == "ms_temporal_rw2_nc" ||
                       field == "ms_temporal_seasonal_nc");
    mst.time_index.resize(n_obs);
    mst.group_index.assign(n_obs, 1);
    for (int i = 0; i < n_obs; i++) mst.time_index[i] = (i % n_times) + 1;
    data.has_multiscale_temporal = true;
  }

  // Spatiotemporal interaction over the S x T grid, which routes to
  // compute_gradient_spatiotemporal_handcoded. want_st is Type I (IID) paired
  // with a separate additive ICAR/BYM2 field; want_st4 is Type IV (Kronecker)
  // on its own, with the interaction's own space margin carrying the CSR
  // adjacency and its own time margin carrying a real RW1 structure.
  if (want_st || want_st4 || want_st2 || want_st3) {
    auto& st = data.spatiotemporal_data;
    st.type = want_st4 ? ratiod_spatiotemporal::STType::TYPE_IV
            : want_st2 ? ratiod_spatiotemporal::STType::TYPE_II
            : want_st3 ? ratiod_spatiotemporal::STType::TYPE_III
                       : ratiod_spatiotemporal::STType::TYPE_I;
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
      // build_grid_adjacency emits 0-based col_idx (the convention data.adj_col_idx
      // uses); the interaction's own adjacency is read with adj_col_idx[idx]-1
      // throughout (hmc_spatiotemporal.h, hmc_sampler.cpp), i.e. it is 1-based,
      // matching what R's spatiotemporal.R comments as "0-based row_ptr, 1-based
      // col_idx -- C++ does -1". Shift it here rather than reusing the 0-based
      // helper as-is, which would read adj_col_idx[idx]-1 == -1 out of bounds.
      build_grid_adjacency(n_units, st.adj_row_ptr, st.adj_col_idx, st.n_neighbors);
      for (int& j : st.adj_col_idx) j += 1;
      // st4_ar1 gives the interaction an AR1 time margin, whose precision
      // R(rho) is what the layout allocates logit_rho_st_idx for.
      st.temporal_type = (field == "st4_ar1" || field == "st4_ar1_nc")
                            ? ratiod_temporal::TemporalType::AR1
                            : ratiod_temporal::TemporalType::RW1;
      st.temporal_cyclic = false;
    } else if (want_st2) {
      // Type II reads only the time margin, so it carries no adjacency. Its
      // three arms are the three temporal precisions.
      st.temporal_type = (field == "st2_ar1" || field == "st2_ar1_nc")
                             ? ratiod_temporal::TemporalType::AR1
                       : (field == "st2_rw2" || field == "st2_rw2_nc")
                             ? ratiod_temporal::TemporalType::RW2
                             : ratiod_temporal::TemporalType::RW1;
      st.temporal_cyclic = (field == "st2_cyclic" || field == "st2_cyclic_nc");
    } else {
      st.temporal_type = ratiod_temporal::TemporalType::NONE;
      st.temporal_cyclic = false;
    }
    st.sigma2_prior_U = 1.0;
    st.sigma2_prior_alpha = 0.01;
    data.has_spatiotemporal = true;
    data.st_is_hsgp = false;
    // The *_nc fields exercise the non-centered interaction: params store z,
    // delta = z/sqrt(tau_st).
    data.st_parameterization =
        (field.size() > 3 && field.compare(field.size() - 3, 3, "_nc") == 0) ? 1 : 0;
  }

  // GP interaction over the S x T grid: one spatial coordinate pair per unit,
  // repeated across that unit's times, and the time axis on the same [0, 1]
  // scale as the unit square so the joint neighbour metric weighs the two
  // alike. Field index k = (s - 1) * T + (t - 1), the same flattening st_flat
  // carries.
  if (want_stgp) {
    auto& st = data.spatiotemporal_data;
    st.type = (field == "stgp_gneiting" || field == "stgp_gneiting_nc" ||
               field == "stgp_gneiting_collapsed")
                  ? ratiod_spatiotemporal::STType::NONSEP_GP
                  : ratiod_spatiotemporal::STType::SEPARABLE;
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
    // The GP types read no time margin, so the layout allocates no rho for
    // them; NONE is what says so.
    st.temporal_type = ratiod_temporal::TemporalType::NONE;
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

    st.cov_space = (field == "stgp_matern") ? ratiod_svc::CovType::MATERN
                                            : ratiod_svc::CovType::EXPONENTIAL;
    st.cov_time = st.cov_space;
    st.nonsep_type = (field == "stgp_gneiting" || field == "stgp_gneiting_nc" ||
                      field == "stgp_gneiting_collapsed")
                         ? ratiod_spatiotemporal::NonsepType::GNEITING
                         : ratiod_spatiotemporal::NonsepType::PRODUCT;

    data.has_spatiotemporal = true;
    data.st_is_hsgp = false;
    // The *_nc fields sample z, delta the NNGP autoregression run from it;
    // the *_collapsed fields integrate delta out (st_gp_laplace.h)
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

  if (want_rsr) restrict_to_numerator_design(data);

  return data;
}

// A point away from the origin, so that terms which vanish at zero (a
// sum-to-zero penalty among them) still contribute.
std::vector<double> draw_params(const ModelData& data, const ParamLayout& layout,
                                int n_params,
                                unsigned int seed) {
  std::mt19937 rng(seed + 1000u);
  std::normal_distribution<double> rnorm(0.0, 0.35);
  std::vector<double> params(n_params);
  for (int i = 0; i < n_params; i++) params[i] = rnorm(rng);

  // The GP locations sit in the unit square, so a draw centred on log phi = 0
  // asks for a correlation range the size of the whole domain. Every neighbour
  // block is then numerically singular, the conditional variance sits on its
  // floor, and the density is not differentiable in sigma2 there -- the
  // comparison would measure the floor rather than the gradient. Centre the
  // range on a tenth of the domain instead, which the random draw still moves.
  // The floor binds harder the smoother the kernel: measured relative deviation
  // on log_sigma2 at log phi = 0 was 2e-07 exponential, 2.4e-05 Matern 3/2 and
  // 7.0e-02 Gaussian, all flat in the difference step.
  const double log_phi_center = std::log(0.1);
  for (int idx : {layout.log_phi_gp_idx, layout.log_phi_gp_local_idx,
                  layout.log_phi_gp_regional_idx,
                  layout.log_phi_st_space_idx, layout.log_phi_st_time_idx}) {
    if (idx >= 0 && idx < n_params) params[idx] += log_phi_center;
  }
  // An NNGP SVC range slot holds the logit of phi on its prior interval, so
  // the same centring is placed through svc_range_coord; the HSGP slot is a
  // log lengthscale and shifts as the others do.
  if (layout.range_svc_start >= 0) {
    for (int j = layout.range_svc_start; j < layout.range_svc_end; j++) {
      if (j >= n_params) continue;
      if (data.svc_is_hsgp) {
        params[j] += log_phi_center;
      } else {
        params[j] = ratiod_svc_ad::svc_range_coord(
            std::exp(params[j] + log_phi_center), data.svc_phi_prior_lower,
            data.svc_phi_prior_upper);
      }
    }
  }
  return params;
}

}  // namespace

// Every field make_model() builds a structure for, so a sweep over the harness
// reaches a newly added one without a second list to keep in step.
// [[Rcpp::export]]
Rcpp::CharacterVector cpp_gradient_fields() {
  Rcpp::CharacterVector out(sizeof(KNOWN_FIELDS) / sizeof(KNOWN_FIELDS[0]));
  for (int i = 0; i < out.size(); i++) out[i] = KNOWN_FIELDS[i];
  return out;
}

// Returns the analytic gradient alongside a central-difference gradient of the
// log posterior, both evaluated at the same random point.
// [[Rcpp::export]]
std::string cpp_gradient_dispatch(std::string field,
                                  int n_obs = 40,
                                  int n_units = 8,
                                  int n_times = 5,
                                  unsigned int seed = 42,
                                  std::string family = "binomial") {
  if (!is_known_field(field)) {
    Rcpp::stop("Unknown field '%s'.", field.c_str());
  }
  ModelData data = make_model(field, n_obs, n_units, n_times, seed,
                              family, /*temporal_shared=*/true, "none");
  ParamLayout layout = compute_param_layout(data);
  return std::string(
      gradient_fn_name(resolve_gradient_fn(GradientMode::HANDCODED, data, layout)));
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
  ModelData data = make_model(field, n_obs, n_units, n_times, seed,
                              family, temporal_shared, zi);
  // A non-canonical link replaces the canonical one of its predictor; the
  // inflation link serves the zero- and one-inflation probabilities alike.
  auto set_link = [](const std::string& name, LinkType& out) {
    if (!name.empty() && !ratiod_link::parse_link(name, out)) {
      Rcpp::stop("cpp_gradient_check: unknown link '" + name + "'");
    }
  };
  set_link(link_num, data.link_num);
  set_link(link_denom, data.link_denom);
  set_link(link_inflation, data.link_zi);
  set_link(link_inflation, data.link_oi);
  // The observation loop is the one part of the density that runs in parallel,
  // and only in the double instantiation. make_model() builds a single-threaded
  // model, so a caller that wants that arm reached has to ask for it.
  data.n_threads = n_threads;
  ParamLayout layout = compute_param_layout(data);
  const int n_params = get_n_params(data);

  GradientMode gm = GradientMode::HANDCODED;
  if (mode == "arena")    gm = GradientMode::AUTODIFF_ARENA;
  if (mode == "forward")  gm = GradientMode::AUTODIFF_FORWARD;
  if (mode == "tape")     gm = GradientMode::AUTODIFF_TAPE;
  set_gradient_mode(gm);
  // Not fatal here: the harness has to be able to measure how far the templated
  // density is from the analytic one on a structure it cannot express, which is
  // what the gap is. Report it so a caller can assert on it.
  const char* impl_gap = ratiod::log_post_impl_gap(data, layout);

  std::vector<double> params = draw_params(data, layout, n_params, seed);

  // An identity or inverse link has a mean only where eta > 0, so the
  // denominator predictor is moved well inside that half-line.
  if ((data.link_denom == LinkType::IDENTITY || data.link_denom == LinkType::INVERSE) &&
      data.p_denom > 0) {
    params[layout.beta_denom_start] = 5.0;
  }

  // Every AR1 correlation is drawn centred on rho = 0, so nothing in the
  // sweep reaches the edge of its range. near_unit_rho places each sampled
  // one where 1 - rho^2 underflows to exactly zero in double, which is the
  // regime ratiod_ar1::one_minus_rho2's floor exists for and the regime a
  // path that floors a different quantity, or none, parts company with the
  // others in. The temporal GP's correlation is exp(-dt / phi) on a phi the
  // layout bounds, so it cannot be taken there this way.
  if (near_unit_rho) {
    // rho = 2u - 1 gives 1 - rho^2 = 4u(1 - u) and rho = u gives
    // (1 - u)(1 + u), so the floor binds in both mappings once the logit
    // passes about 24. Past about 37, u rounds to exactly 1 and the prior's
    // own log(1 - u) is -inf, which is the edge of its support rather than
    // anything about 1 - rho^2.
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

  std::vector<double> grad_analytic(n_params, 0.0);
  double lp_from_mode = 0.0;
  compute_gradient(params, data, layout, grad_analytic, &lp_from_mode);

  // compute_log_post is compute_log_post_impl<double>, so differencing it is
  // differencing the density every mode reports a value for.
  auto lp_at = [&](const std::vector<double>& p) {
    return compute_log_post(p, data, layout);
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
    if (i >= layout.beta_num_start && i < layout.beta_num_end) block[i] = "beta_num";
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
    Rcpp::Named("log_post") = compute_log_post(params, data, layout),
    Rcpp::Named("log_post_mode") = lp_from_mode,
    Rcpp::Named("impl_gap") = (impl_gap == nullptr)
      ? Rcpp::CharacterVector::create(NA_STRING)
      : Rcpp::CharacterVector::create(impl_gap)
  );
}

// Evaluates one gradient function at several points, first one at a time and
// then concurrently over a single shared ModelData, and returns how far the two
// results are apart.
//
// This is the shape a fit takes: chains run in parallel over one ModelData,
// each at its own position, and a gradient function that keeps intermediates in
// that shared object has every chain writing the same memory. The gradient is
// then a function of what the other chains happened to be doing, so a
// single-point finite-difference check agrees while the sampler follows a
// density its gradient does not describe. Per-thread scratch makes the two
// passes agree exactly, so this returns 0 and any nonzero value is a shared
// buffer.
//
// The collapsed fields warm-start their inner Newton search from the mode of
// the thread's previous call, which is a second way for two passes to disagree
// and not the one being measured: the two passes visit the points in different
// orders on different threads, so each point is reached from a different start.
// Both passes therefore clear the warm start before every evaluation, leaving
// the gradient a function of the point alone.
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
  // One ModelData for every point, exactly as one fit shares it across chains.
  ModelData data = make_model(field, n_obs, n_units, n_times, seed,
                              family, /*temporal_shared=*/true, "none");
  data.n_threads = 1;  // the region below is the parallelism under test
  ParamLayout layout = compute_param_layout(data);
  const int n_params = get_n_params(data);
  set_gradient_mode(GradientMode::HANDCODED);
  GradientFn fn = resolve_gradient_fn(GradientMode::HANDCODED, data, layout);

  std::vector<std::vector<double>> points(n_points);
  for (int k = 0; k < n_points; k++) {
    points[k] = draw_params(data, layout, n_params, seed + static_cast<unsigned int>(k));
  }

  // Hands the calling thread a second model of the same dimensions at the same
  // point first, so it enters the race holding that model's cached structure
  // while the workers of the concurrent pass build the current one's. The two
  // models differ only in their data, so a workspace keyed on size alone finds
  // its cache valid -- same number of locations, same (sigma2, phi) -- and
  // returns the other model's NNGP coefficients. A workspace keyed on the
  // model rebuilds instead, and the two passes agree.
  if (prime_other_model) {
    ModelData other = make_model(field, n_obs, n_units, n_times, seed + 977u,
                                 family, /*temporal_shared=*/true, "none");
    other.n_threads = 1;
    ParamLayout other_layout = compute_param_layout(other);
    if (get_n_params(other) != n_params) {
      Rcpp::stop("cpp_gradient_race: '%s' does not size its parameter vector "
                 "from the dimensions alone, so the two models cannot be "
                 "evaluated at the same point.", field.c_str());
    }
    std::vector<double> discard(n_params, 0.0);
    GradientFn other_fn = resolve_gradient_fn(GradientMode::HANDCODED, other,
                                              other_layout);
    other_fn(points[0], other, other_layout, discard, nullptr);
  }

  std::vector<std::vector<double>> serial(n_points);
  for (int k = 0; k < n_points; k++) {
    serial[k].assign(n_params, 0.0);
    collapsed_gp_reset_warm_start();
    collapsed_icar_reset_warm_start();
    ratiod_st_laplace::reset_warm_start();
    fn(points[k], data, layout, serial[k], nullptr);
  }

  double max_dev = 0.0;
  for (int r = 0; r < rounds; r++) {
    std::vector<std::vector<double>> concurrent(n_points);
    for (int k = 0; k < n_points; k++) concurrent[k].assign(n_params, 0.0);

#ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(n_threads)
#endif
    for (int k = 0; k < n_points; k++) {
      collapsed_gp_reset_warm_start();
      collapsed_icar_reset_warm_start();
      ratiod_st_laplace::reset_warm_start();
      fn(points[k], data, layout, concurrent[k], nullptr);
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

// Milliseconds per gradient evaluation (with its fused log posterior) of the
// default gradient path, over `n_rep` evaluations at points a small step
// apart, as a leapfrog trajectory visits them.
// [[Rcpp::export]]
double cpp_time_gradient(std::string field, int n_rep = 50, int n_obs = 400,
                         int n_units = 25, int n_times = 10,
                         unsigned int seed = 42,
                         std::string family = "poisson_gamma") {
  ModelData data = make_model(field, n_obs, n_units, n_times, seed, family);
  ParamLayout layout = compute_param_layout(data);
  const int n_params = get_n_params(data);
  set_gradient_mode(GradientMode::HANDCODED);
  std::vector<double> params = draw_params(data, layout, n_params, seed);
  std::vector<double> grad(n_params, 0.0);
  double lp = 0.0;
  compute_gradient(params, data, layout, grad, &lp);
  const auto t0 = std::chrono::steady_clock::now();
  for (int r = 0; r < n_rep; r++) {
    for (int k = 0; k < n_params; k++) params[k] += 1e-3 * grad[k] / (1.0 + std::abs(grad[k]));
    compute_gradient(params, data, layout, grad, &lp);
  }
  const auto t1 = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::milli>(t1 - t0).count() / n_rep;
}

// Evaluates the log posterior at a shared point so the analytic and autodiff
// implementations, which are written out separately, can be compared directly.
// [[Rcpp::export]]
double cpp_logpost_at(std::string field,
                      int n_obs = 400,
                      int n_units = 25,
                      int n_times = 10,
                      unsigned int seed = 42,
                      std::string family = "binomial",
                      bool temporal_shared = true,
                      std::string zi = "none") {
  ModelData data = make_model(field, n_obs, n_units, n_times, seed,
                              family, temporal_shared, zi);
  ParamLayout layout = compute_param_layout(data);
  const int n_params = get_n_params(data);
  std::mt19937 rng(seed + 1000u);
  std::normal_distribution<double> rnorm(0.0, 0.35);
  std::vector<double> params(n_params);
  for (int i = 0; i < n_params; i++) params[i] = rnorm(rng);
  return compute_log_post(params, data, layout);
}

// The two coordinates of the GP space-time interaction describe one
// posterior: at fixed hyperparameters the non-centred density at z minus the
// centred density at delta(z) is log|d delta / dz|, a function of the
// hyperparameters alone, so it is the same number for every z. Returns its spread across draws of z.
// [[Rcpp::export]]
double cpp_st_gp_nc_coordinate_spread(std::string field = "stgp_nc",
                                      int n_draws = 5,
                                      unsigned int seed = 42) {
  const std::string centred_field = field.substr(0, field.size() - 3);
  ModelData data_nc = make_model(field, 400, 25, 10, seed);
  ModelData data_c = make_model(centred_field, 400, 25, 10, seed);
  ParamLayout layout = compute_param_layout(data_nc);
  const int n_params = get_n_params(data_nc);
  const int ST = data_nc.spatiotemporal_data.n_params;

  std::vector<double> params_nc = draw_params(data_nc, layout, n_params, seed);
  std::mt19937 rng(seed + 3000u);
  std::normal_distribution<double> rnorm(0.0, 1.0);

  double lo = R_PosInf, hi = R_NegInf;
  for (int d = 0; d < n_draws; d++) {
    for (int k = 0; k < ST; k++) params_nc[layout.st_delta_start + k] = rnorm(rng);
    const double lp_nc = compute_log_post(params_nc, data_nc, layout);

    std::vector<double> params_c = params_nc;
    const bool ok = ratiod_spatiotemporal::st_gp_nc_forward(
        &params_nc[layout.st_delta_start],
        std::exp(-params_nc[layout.log_tau_st_idx]),
        std::exp(params_nc[layout.log_phi_st_space_idx]),
        std::exp(params_nc[layout.log_phi_st_time_idx]),
        data_nc.spatiotemporal_data, &params_c[layout.st_delta_start]);
    if (!ok) Rcpp::stop("cpp_st_gp_nc_coordinate_spread: the NNGP scan failed");
    const double diff = lp_nc - compute_log_post(params_c, data_c, layout);
    lo = std::min(lo, diff);
    hi = std::max(hi, diff);
  }
  return hi - lo;
}

// For a likelihood Gaussian in the interaction (lognormal), the collapsed
// interaction's Laplace step is exact: p(y, delta | rest) =
// p(y | rest) N(delta; dh, H^-1) for every delta. So the centred density at
// any delta, less log N(delta; dh, H^-1), is the collapsed density whatever
// delta is; a missing normalizer, penalty or correction term shows as a gap.
// Returns the largest absolute gap over draws of delta.
// [[Rcpp::export]]
double cpp_st_collapsed_exactness(std::string field = "stgp_collapsed",
                                  int n_draws = 4, unsigned int seed = 42,
                                  int n_obs = 200, int n_units = 9,
                                  int n_times = 4) {
  const std::string suffix = "_collapsed";
  const std::string centred_field = field.substr(0, field.size() - suffix.size());
  ModelData data_l = make_model(field, n_obs, n_units, n_times, seed, "lognormal");
  ModelData data_c = make_model(centred_field, n_obs, n_units, n_times, seed,
                                "lognormal");
  ParamLayout lay_l = compute_param_layout(data_l);
  ParamLayout lay_c = compute_param_layout(data_c);
  std::vector<double> p_l = draw_params(data_l, lay_l, get_n_params(data_l), seed);
  const double lp_l = compute_log_post(p_l, data_l, lay_l);
  if (!std::isfinite(lp_l)) Rcpp::stop("cpp_st_collapsed_exactness: no mode");

  const auto& ws = ratiod_st_laplace::workspace();
  const int n = data_l.spatiotemporal_data.n_params;
  const double log_det_H = ratiod_st_laplace::log_det_factor(ws);

  std::mt19937 rng(seed + 5000u);
  std::normal_distribution<double> rnorm(0.0, 0.3);
  std::vector<double> e(n), buf(n);
  double worst = 0.0;
  for (int d = 0; d < n_draws; d++) {
    std::vector<double> p_c(p_l.begin(), p_l.begin() + lay_c.st_delta_start);
    for (int k = 0; k < n; k++) {
      e[k] = rnorm(rng);
      p_c.push_back(ws.delta[k] + e[k]);
    }
    p_c.insert(p_c.end(), p_l.begin() + lay_c.st_delta_start, p_l.end());
    double eHe = ratiod_st_laplace::prior_quad(data_l, ws.fac, e.data(), buf.data());
    for (int k = 0; k < n; k++) eHe += ws.W[k] * e[k] * e[k];
    const double log_normal = -0.5 * n * std::log(2.0 * M_PI) + 0.5 * log_det_H -
                              0.5 * eHe;
    const double gap = compute_log_post(p_c, data_c, lay_c) - log_normal - lp_l;
    worst = std::max(worst, std::abs(gap));
  }
  return worst;
}

// The centred density the transform claims to carry, for G walks each with
// its own augmented constant: each group's quadratic form and augmented
// direction, and a rank that counts the per-group ranks plus one direction per
// group. G = 1 is rw1_log_lik / rw2_log_lik with `augment`.
inline double nc_reference_log_prior(const double* phi, int n, int n_groups,
                                     int order, bool cyclic, double sigma2) {
  double quad = 0.0;
  for (int g = 0; g < n_groups; g++) {
    quad += (order == 2)
                ? ratiod_temporal::rw2_quadratic_form(phi + g * n, n, cyclic)
                : ratiod_temporal::rw1_quadratic_form(phi + g * n, n, cyclic);
    const double s = tulpa::s2z_component_sum(phi, g * n, n);
    quad += tulpa::s2z_aug_coef(1.0, n) * s * s;
  }

  const int rank_g = (order == 2) ? tulpa::rw2_rank(n, cyclic)
                                  : tulpa::rw1_rank(n, cyclic);
  const int rank = tulpa::s2z_aug_rank(rank_g * n_groups, n_groups);
  return -0.5 * quad / sigma2 - 0.5 * rank * std::log(2.0 * M_PI * sigma2);
}

// The invariant a finite difference cannot see. A transform that builds the
// WRONG field still leaves the density self-consistent and its gradient still
// matches its own difference quotient, so the gradient sweep passes on it. What
// it cannot leave alone is the relationship between the two coordinates:
//
//     log N(z; 0, I) = log p_centred(phi(z)) + log|d phi / d z|,
//
// and the Jacobian is 0.5 * rank * log(sigma2), a function of the
// hyperparameter alone. So at fixed sigma2 the two densities differ by ONE
// number for every z, and the spread of that difference over draws is what this
// returns. Anything but zero is a transform that does not carry the density it
// claims to.
//
// With `n_groups` above 1 the centred side is the grouped density: each
// group's own quadratic form and its own augmented constant.
//
// [[Rcpp::export]]
double cpp_ms_temporal_nc_coordinate_spread(std::string field = "ms_temporal_nc",
                                            int n = 12,
                                            double sigma2 = 0.25,
                                            int n_draws = 6,
                                            unsigned int seed = 42,
                                            int n_groups = 1) {
  int order = 1;
  bool cyclic = false;
  if (field == "ms_temporal_rw2_nc") order = 2;
  if (field == "ms_temporal_seasonal_nc") cyclic = true;

  std::mt19937 rng(seed + 7000u);
  std::normal_distribution<double> rnorm(0.0, 1.0);
  const double sigma = std::sqrt(sigma2);
  const int n_total = n * n_groups;

  double first = 0.0;
  double worst = 0.0;
  for (int d = 0; d < n_draws; d++) {
    std::vector<double> z(n_total), phi(n_total);
    for (int k = 0; k < n_total; k++) z[k] = rnorm(rng);
    ratiod_temporal_nc::rw_nc_grouped_forward(z.data(), n, n_groups, order,
                                              cyclic, sigma, phi.data());

    const double lp_nc = ratiod_temporal_nc::rw_nc_grouped_log_prior(
        z.data(), n, n_groups, order, cyclic);
    const double lp_centred =
        nc_reference_log_prior(phi.data(), n, n_groups, order, cyclic, sigma2);
    const double gap = lp_nc - lp_centred;
    if (d == 0) first = gap;
    worst = std::max(worst, std::abs(gap - first));
  }
  return worst;
}


// The portable polygamma psi^(n)(x), for comparison with R's psigamma().
// [[Rcpp::export]]
Rcpp::NumericVector cpp_polygamma(int n, Rcpp::NumericVector x) {
  Rcpp::NumericVector out(x.size());
  for (R_xlen_t i = 0; i < x.size(); i++) {
    out[i] = ratiod::math::portable_polygamma(n, x[i]);
  }
  return out;
}
