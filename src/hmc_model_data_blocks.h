// hmc_model_data_blocks.h
// ModelData assembly for the parameter bundles the R backend passes as
// named lists. One builder per block, so an entry point carries a block by
// calling the builder rather than by repeating its extraction.
//
// An absent (zero-length) list means the model does not carry that block,
// and the builder writes the same "off" state the block had when it was
// declared inert.

#ifndef RATIOD_HMC_MODEL_DATA_BLOCKS_H
#define RATIOD_HMC_MODEL_DATA_BLOCKS_H

#include <Rcpp.h>
#include <string>
#include <vector>
#include "hmc_sampler.h"
#include "ar1_shared.h"
#include "cov_type_code.h"
#include "hmc_temporal_gp.h"
#include "hmc_temporal_multiscale.h"

namespace ratiod_hmc {

inline void apply_latent_params(ModelData& data, const Rcpp::List& latent_params) {
  if (latent_params.size() == 0) {
    data.has_latent = false;
    data.latent_n_factors = 0;
    return;
  }
  data.has_latent = Rcpp::as<bool>(latent_params["has_latent"]);
  data.latent_n_factors = Rcpp::as<int>(latent_params["n_factors"]);
  data.latent_shared = Rcpp::as<bool>(latent_params["shared"]);
  data.latent_scale = Rcpp::as<bool>(latent_params["scale"]);
  data.latent_constraint = Rcpp::as<int>(latent_params["constraint"]);
  data.latent_sigma_prior_rate = Rcpp::as<double>(latent_params["sigma_prior_rate"]);
}

inline void apply_st_params(ModelData& data, const Rcpp::List& st_params) {
  // Spatiotemporal interaction - extract from list
  bool has_spatiotemporal = st_params.size() > 0 && Rcpp::as<bool>(st_params["has_spatiotemporal"]);
  data.has_spatiotemporal = has_spatiotemporal;
  if (has_spatiotemporal) {
    // Extract parameters from list (eager deep copies to prevent R GC issues)
    std::string st_type_str = Rcpp::as<std::string>(st_params["type"]);
    bool st_shared = Rcpp::as<bool>(st_params["shared"]);
    int st_n_spatial = Rcpp::as<int>(st_params["n_spatial"]);
    int st_n_times = Rcpp::as<int>(st_params["n_times"]);
    int st_n_params = Rcpp::as<int>(st_params["n_params"]);
    std::vector<int> st_s_idx = Rcpp::as<std::vector<int>>(st_params["s_idx"]);
    std::vector<int> st_t_idx = Rcpp::as<std::vector<int>>(st_params["t_idx"]);
    std::vector<int> st_flat = Rcpp::as<std::vector<int>>(st_params["st_flat"]);
    std::string st_temporal_type_str = Rcpp::as<std::string>(st_params["temporal_type"]);
    bool st_temporal_cyclic = Rcpp::as<bool>(st_params["temporal_cyclic"]);
    std::vector<int> st_adj_row_ptr = Rcpp::as<std::vector<int>>(st_params["adj_row_ptr"]);
    std::vector<int> st_adj_col_idx = Rcpp::as<std::vector<int>>(st_params["adj_col_idx"]);
    double st_sigma2_prior_U = Rcpp::as<double>(st_params["sigma2_prior_U"]);
    double st_sigma2_prior_alpha = Rcpp::as<double>(st_params["sigma2_prior_alpha"]);
    double st_rho_prior_a = st_params.containsElementNamed("rho_prior_a")
        ? Rcpp::as<double>(st_params["rho_prior_a"]) : ratiod_ar1::RHO_PRIOR_A;
    double st_rho_prior_b = st_params.containsElementNamed("rho_prior_b")
        ? Rcpp::as<double>(st_params["rho_prior_b"]) : ratiod_ar1::RHO_PRIOR_B;

    // Parse ST type (accept both R-side "I"/"IV" and legacy "type_i"/"type_iv")
    if (st_type_str == "I" || st_type_str == "type_i") {
      data.spatiotemporal_data.type = STType::TYPE_I;
    } else if (st_type_str == "II" || st_type_str == "type_ii") {
      data.spatiotemporal_data.type = STType::TYPE_II;
    } else if (st_type_str == "III" || st_type_str == "type_iii") {
      data.spatiotemporal_data.type = STType::TYPE_III;
    } else if (st_type_str == "IV" || st_type_str == "type_iv") {
      data.spatiotemporal_data.type = STType::TYPE_IV;
    } else if (st_type_str == "separable") {
      data.spatiotemporal_data.type = STType::SEPARABLE;
    } else if (st_type_str == "nonsep_gp") {
      data.spatiotemporal_data.type = STType::NONSEP_GP;
    } else {
      Rcpp::stop("Unknown spatiotemporal type: '%s'. Expected one of: I, II, III, IV, separable, nonsep_gp",
                 st_type_str.c_str());
    }

    data.spatiotemporal_data.shared = st_shared;
    data.spatiotemporal_data.n_spatial = st_n_spatial;
    data.spatiotemporal_data.n_times = st_n_times;
    data.spatiotemporal_data.n_params = st_n_params;

    // Observation indexing (already deep copied above)
    data.spatiotemporal_data.s_idx = st_s_idx;
    data.spatiotemporal_data.t_idx = st_t_idx;
    data.spatiotemporal_data.st_flat = st_flat;

    // Temporal type for Type II/IV and HSGP-ST, the interactions whose own
    // time margin carries a precision.
    const bool st_is_hsgp_flag =
        st_params.containsElementNamed("st_is_hsgp") &&
        Rcpp::as<bool>(st_params["st_is_hsgp"]);
    const bool st_reads_time_margin =
        ratiod_spatiotemporal::st_time_margin_is_structured(
            data.spatiotemporal_data.type) || st_is_hsgp_flag;
    if (st_temporal_type_str == "rw1") {
      data.spatiotemporal_data.temporal_type = TemporalType::RW1;
    } else if (st_temporal_type_str == "rw2") {
      data.spatiotemporal_data.temporal_type = TemporalType::RW2;
    } else if (st_temporal_type_str == "ar1") {
      data.spatiotemporal_data.temporal_type = TemporalType::AR1;
    } else if (st_reads_time_margin) {
      // The interaction reads this margin, and there is no precision to read
      // it with. Defaulting to RW1 would fit a structure the caller did not
      // ask for; contributing nothing would leave the field with no prior.
      Rcpp::stop("Spatiotemporal temporal margin '%s' is not supported for this "
                 "interaction type. Use temporal_rw1(), temporal_rw2() or "
                 "temporal_ar1().", st_temporal_type_str.c_str());
    } else {
      data.spatiotemporal_data.temporal_type = TemporalType::RW1;  // Unread
    }
    data.spatiotemporal_data.temporal_cyclic = st_temporal_cyclic;

    // Spatial adjacency for Type III/IV (already deep copied above)
    data.spatiotemporal_data.adj_row_ptr = st_adj_row_ptr;
    data.spatiotemporal_data.adj_col_idx = st_adj_col_idx;

    // The GP types' NNGP structure, indexed by the interaction's own field
    // index -- the S x T grid for spatiotemporal(), one entry per observation
    // for spatiotemporal_gp(). nn_order / nn_order_inv arrive 1-based from R
    // and are stored 0-based, the convention every other NNGP path uses;
    // nn_idx stays 1-based within the ordering, with 0 for an absent
    // neighbour. Nothing downstream can tell a half-filled structure from a
    // filled one once a density is reading it, so it is checked here.
    if (data.spatiotemporal_data.type == STType::SEPARABLE ||
        data.spatiotemporal_data.type == STType::NONSEP_GP) {
      auto& stg = data.spatiotemporal_data;
      stg.nn = Rcpp::as<int>(st_params["nn"]);
      stg.coords = Rcpp::as<std::vector<double>>(st_params["gp_coords"]);
      stg.time_values = Rcpp::as<std::vector<double>>(st_params["gp_time_values"]);
      stg.nn_idx = Rcpp::as<std::vector<int>>(st_params["nn_idx"]);
      stg.nn_dist_space = Rcpp::as<std::vector<double>>(st_params["nn_dist_space"]);
      stg.nn_dist_time = Rcpp::as<std::vector<double>>(st_params["nn_dist_time"]);
      std::vector<int> stg_order = Rcpp::as<std::vector<int>>(st_params["nn_order"]);
      std::vector<int> stg_order_inv = Rcpp::as<std::vector<int>>(st_params["nn_order_inv"]);
      stg.nn_order.resize(stg_order.size());
      for (size_t i = 0; i < stg_order.size(); i++) stg.nn_order[i] = stg_order[i] - 1;
      stg.nn_order_inv.resize(stg_order_inv.size());
      for (size_t i = 0; i < stg_order_inv.size(); i++) stg.nn_order_inv[i] = stg_order_inv[i] - 1;
      stg.cov_space = ratiod_cov::cov_type_from_int(
          Rcpp::as<int>(st_params["cov_space"]));
      stg.cov_time = ratiod_cov::cov_type_from_int(
          Rcpp::as<int>(st_params["cov_time"]));
      const std::string st_nonsep = Rcpp::as<std::string>(st_params["nonsep_type"]);
      if (st_nonsep == "product") {
        stg.nonsep_type = NonsepType::PRODUCT;
      } else if (st_nonsep == "gneiting") {
        stg.nonsep_type = NonsepType::GNEITING;
      } else {
        Rcpp::stop("Unknown spatiotemporal nonsep_type: '%s'. Expected one of: "
                   "product, gneiting", st_nonsep.c_str());
      }
      data.st_phi_space_prior_lower = Rcpp::as<double>(st_params["phi_space_prior_lower"]);
      data.st_phi_space_prior_upper = Rcpp::as<double>(st_params["phi_space_prior_upper"]);
      data.st_phi_time_prior_lower = Rcpp::as<double>(st_params["phi_time_prior_lower"]);
      data.st_phi_time_prior_upper = Rcpp::as<double>(st_params["phi_time_prior_upper"]);
      if (!ratiod_spatiotemporal::st_gp_structure_filled(stg)) {
        Rcpp::stop("Spatiotemporal GP interaction: the NNGP structure does not "
                   "describe its %d field entries (nn = %d). Every one of "
                   "gp_coords, gp_time_values, nn_idx, nn_dist_space, "
                   "nn_dist_time and nn_order has to be filled over the "
                   "interaction's own index.",
                   stg.n_params, stg.nn);
      }
    }

    // Prior parameters
    data.st_sigma2_prior_U = st_sigma2_prior_U;
    data.st_sigma2_prior_alpha = st_sigma2_prior_alpha;
    data.spatiotemporal_data.rho_prior_a = st_rho_prior_a;
    data.spatiotemporal_data.rho_prior_b = st_rho_prior_b;

    // Parameterization: centered by default. NC requires spectral decomposition
    // (Kronecker eigenvectors of Q_s ⊗ Q_t) which is not yet implemented.
    // Simple scaling NC (z = delta * sqrt(tau)) preserves GMRF anisotropy
    // and makes performance worse (eps=0.003, td=11.5 vs eps=0.006, td=10).
    data.st_parameterization = 0;  // Always centered for now
    if (st_params.containsElementNamed("parameterization")) {
      std::string st_param_str = Rcpp::as<std::string>(st_params["parameterization"]);
      data.st_parameterization = (st_param_str == "centered") ? 0 : 1;
    }

    // Kronecker precision data for ST_IV (precomputed in R)
    if (st_params.containsElementNamed("Qs_inv") &&
        st_params.containsElementNamed("Qt_inv")) {
      SEXP qs_sexp = st_params["Qs_inv"];
      SEXP ls_sexp = st_params["Ls"];
      SEXP qt_sexp = st_params["Qt_inv"];
      SEXP lt_sexp = st_params["Lt"];
      if (!Rf_isNull(qs_sexp) && !Rf_isNull(qt_sexp) &&
          !Rf_isNull(ls_sexp) && !Rf_isNull(lt_sexp)) {
        data.st_Qs_inv = Rcpp::as<std::vector<double>>(qs_sexp);
        data.st_Ls = Rcpp::as<std::vector<double>>(ls_sexp);
        data.st_Qt_inv = Rcpp::as<std::vector<double>>(qt_sexp);
        data.st_Lt = Rcpp::as<std::vector<double>>(lt_sexp);
      }
    }

    // HSGP-ST: spectral basis spatiotemporal interaction
    data.st_is_hsgp = false;
    if (st_params.containsElementNamed("st_is_hsgp") &&
        Rcpp::as<bool>(st_params["st_is_hsgp"])) {
      data.st_is_hsgp = true;
      data.spatiotemporal_data.is_hsgp = true;
      int st_hsgp_m = Rcpp::as<int>(st_params["hsgp_m"]);
      double st_hsgp_c = Rcpp::as<double>(st_params["hsgp_c"]);
      std::vector<double> st_hsgp_coords = Rcpp::as<std::vector<double>>(st_params["hsgp_coords"]);
      bool st_hsgp_scale = true;
      if (st_params.containsElementNamed("hsgp_scale_coords"))
        st_hsgp_scale = Rcpp::as<bool>(st_params["hsgp_scale_coords"]);

      // Setup HSGP basis (Phi matrix + eigenvalues)
      ratiod_hsgp::setup_hsgp_2d(
        st_hsgp_coords, data.N,
        st_hsgp_m, st_hsgp_c, st_hsgp_scale,
        data.st_hsgp_data);
      data.spatiotemporal_data.hsgp_m_total = data.st_hsgp_data.m_total;
      // Override n_spatial and n_params for HSGP-ST
      data.spatiotemporal_data.n_spatial = data.st_hsgp_data.m_total;
      data.spatiotemporal_data.n_params = data.st_hsgp_data.m_total * st_n_times;
    }
  } else {
    data.spatiotemporal_data.type = STType::NONE;
  }
}

inline void apply_tvc_params(ModelData& data, const Rcpp::List& tvc_params) {
  // TVC (Temporally-Varying Coefficients) parameters
  bool has_tvc = tvc_params.size() > 0 && Rcpp::as<bool>(tvc_params["has_tvc"]);
  data.has_tvc = has_tvc;
  if (has_tvc) {
    // Extract TVC parameters (eager deep copies to prevent R GC issues)
    int tvc_n_tvc = Rcpp::as<int>(tvc_params["n_tvc"]);
    int tvc_n_times = Rcpp::as<int>(tvc_params["n_times"]);
    int tvc_n_groups = Rcpp::as<int>(tvc_params["n_groups"]);
    std::string tvc_structure_str = Rcpp::as<std::string>(tvc_params["structure"]);
    bool tvc_shared = Rcpp::as<bool>(tvc_params["shared"]);
    bool tvc_cyclic = Rcpp::as<bool>(tvc_params["cyclic"]);
    std::vector<int> tvc_indices = Rcpp::as<std::vector<int>>(tvc_params["tvc_indices"]);
    std::vector<int> tvc_time_index = Rcpp::as<std::vector<int>>(tvc_params["time_index"]);
    std::vector<int> tvc_group_index = Rcpp::as<std::vector<int>>(tvc_params["group_index"]);
    std::vector<double> tvc_X_tvc = Rcpp::as<std::vector<double>>(tvc_params["X_tvc"]);
    double tvc_tau_shape = Rcpp::as<double>(tvc_params["tau_shape"]);
    double tvc_tau_rate = Rcpp::as<double>(tvc_params["tau_rate"]);

    // Populate TVC data structure
    data.tvc_data.n_obs = data.N;
    data.tvc_data.n_tvc = tvc_n_tvc;
    data.tvc_data.n_times = tvc_n_times;
    data.tvc_data.n_groups = tvc_n_groups;
    data.tvc_data.shared = tvc_shared;
    data.tvc_data.cyclic = tvc_cyclic;
    data.tvc_data.tvc_indices = tvc_indices;
    data.tvc_data.time_index = tvc_time_index;
    data.tvc_data.group_index = tvc_group_index;
    data.tvc_data.X_tvc = tvc_X_tvc;

    // Parse TVC temporal structure. R validates "gp" out at temporal_tvc()
    // and rejects anything but "rw1" for the Gibbs backend, so every string
    // reaching here is rw1/rw2/ar1/iid; the RW1 fallback below is for a
    // literally unrecognized string, not a real, unimplemented choice.
    data.tvc_data.structure = ratiod_tvc::parse_tvc_structure(tvc_structure_str);

    // Prior parameters
    data.tvc_tau_shape = tvc_tau_shape;
    data.tvc_tau_rate = tvc_tau_rate;
  } else {
    data.tvc_data.n_tvc = 0;
    data.tvc_data.n_times = 0;
    data.tvc_data.n_groups = 1;
  }
}

inline void apply_svc_params(ModelData& data, const Rcpp::List& svc_params) {
  // SVC (Spatially-Varying Coefficients) parameters
  bool has_svc = svc_params.size() > 0 && Rcpp::as<bool>(svc_params["has_svc"]);
  data.has_svc = has_svc;
  if (has_svc) {
    // Extract SVC parameters (eager deep copies to prevent R GC issues)
    int svc_n_svc = Rcpp::as<int>(svc_params["n_svc"]);
    int svc_nn = Rcpp::as<int>(svc_params["nn"]);
    bool svc_shared = Rcpp::as<bool>(svc_params["shared"]);
    std::string svc_cov_type_str = Rcpp::as<std::string>(svc_params["cov_type"]);
    std::vector<double> svc_coords = Rcpp::as<std::vector<double>>(svc_params["coords"]);
    std::vector<int> svc_indices = Rcpp::as<std::vector<int>>(svc_params["svc_indices"]);
    std::vector<double> svc_X_svc = Rcpp::as<std::vector<double>>(svc_params["X_svc"]);
    double svc_sigma2_scale = Rcpp::as<double>(svc_params["sigma2_prior_scale"]);
    double svc_phi_lower = Rcpp::as<double>(svc_params["phi_prior_lower"]);
    double svc_phi_upper = Rcpp::as<double>(svc_params["phi_prior_upper"]);

    // Check if this is HSGP-based SVC
    std::string svc_approx = "nngp";
    if (svc_params.containsElementNamed("svc_approx")) {
      svc_approx = Rcpp::as<std::string>(svc_params["svc_approx"]);
    }
    data.svc_is_hsgp = (svc_approx == "hsgp");

    // Populate SVC data structure (shared fields)
    data.svc_data.n_obs = data.N;
    data.svc_data.n_svc = svc_n_svc;
    data.svc_data.shared = svc_shared;
    data.svc_data.coords = svc_coords;
    data.svc_data.svc_indices = svc_indices;
    data.svc_data.X_svc = svc_X_svc;

    if (data.svc_is_hsgp) {
      // HSGP-based SVC: set up basis functions
      int hsgp_m = Rcpp::as<int>(svc_params["hsgp_m"]);
      double hsgp_c = Rcpp::as<double>(svc_params["hsgp_c"]);
      data.svc_hsgp_m_per_dim = hsgp_m;
      data.svc_hsgp_boundary_factor = hsgp_c;

      // Set up HSGP basis (shared across all SVC terms)
      ratiod_hsgp::setup_hsgp_2d(svc_coords, data.N, hsgp_m, hsgp_c,
                                  svc_shared, data.svc_hsgp_data);

      // No NNGP data needed
      data.svc_data.nn = 0;
    } else {
      // NNGP-based SVC: set up neighbor structure
      data.svc_data.nn = svc_nn;
      data.svc_data.nn_idx = Rcpp::as<std::vector<int>>(svc_params["nn_idx"]);
      data.svc_data.nn_dist = Rcpp::as<std::vector<double>>(svc_params["nn_dist"]);
      data.svc_data.nn_order = Rcpp::as<std::vector<int>>(svc_params["nn_order"]);
      data.svc_data.nn_order_inv = Rcpp::as<std::vector<int>>(svc_params["nn_order_inv"]);

      // Parse SVC covariance type
      if (svc_cov_type_str == "exponential") {
        data.svc_data.cov_type = ratiod_svc::CovType::EXPONENTIAL;
      } else if (svc_cov_type_str == "matern") {
        data.svc_data.cov_type = ratiod_svc::CovType::MATERN;
      } else if (svc_cov_type_str == "gaussian") {
        data.svc_data.cov_type = ratiod_svc::CovType::GAUSSIAN;
      } else if (svc_cov_type_str == "spherical") {
        data.svc_data.cov_type = ratiod_svc::CovType::SPHERICAL;
      } else {
        Rcpp::stop("Unknown SVC covariance type '%s'. Expected one of "
                   "exponential, matern, gaussian, spherical.",
                   svc_cov_type_str);
      }
    }

    // Prior parameters
    data.svc_sigma2_prior_scale = svc_sigma2_scale;
    data.svc_phi_prior_lower = svc_phi_lower;
    data.svc_phi_prior_upper = svc_phi_upper;
  } else {
    data.svc_data.n_svc = 0;
    data.svc_data.n_obs = data.N;
    data.svc_data.nn = 0;
    data.svc_is_hsgp = false;
  }
}


// The random-effect block. compute_param_layout() sizes the sigma, Cholesky
// and effect blocks off these fields, and the R backend lays the same blocks
// over the returned draws, so an entry point that declares the block
// single-term samples a shorter vector than the one R names and every column
// after it is read at the wrong offset.
//
// data.N has to be set before this runs: the per-term group column and the
// slope design are read over it.
inline void apply_re_params(ModelData& data, const Rcpp::List& re_params) {
  // Eager deep copies, so R's GC cannot invalidate a view during sampling.
  std::vector<int> re_group = Rcpp::as<std::vector<int>>(re_params["group"]);
  int n_re_groups = Rcpp::as<int>(re_params["n_groups"]);
  int n_re_terms = Rcpp::as<int>(re_params["n_terms"]);

  // group_matrix arrives numeric or integer depending on how R built it.
  Rcpp::IntegerMatrix re_group_matrix;
  SEXP group_mat_sexp = re_params["group_matrix"];
  if (Rf_isMatrix(group_mat_sexp)) {
    if (TYPEOF(group_mat_sexp) == INTSXP) {
      re_group_matrix = Rcpp::as<Rcpp::IntegerMatrix>(group_mat_sexp);
    } else {
      Rcpp::NumericMatrix nm(group_mat_sexp);
      re_group_matrix = Rcpp::IntegerMatrix(nm.nrow(), nm.ncol());
      for (int i = 0; i < nm.nrow(); i++) {
        for (int j = 0; j < nm.ncol(); j++) {
          re_group_matrix(i, j) = static_cast<int>(nm(i, j));
        }
      }
    }
  } else {
    re_group_matrix = Rcpp::IntegerMatrix(1, 1);
    re_group_matrix(0, 0) = 0;
  }

  std::vector<int> re_n_groups_vec = Rcpp::as<std::vector<int>>(re_params["n_groups_vec"]);
  bool has_re_slopes = Rcpp::as<bool>(re_params["has_slopes"]);
  bool has_re_correlated_slopes = Rcpp::as<bool>(re_params["has_correlated_slopes"]);
  std::vector<int> re_n_coefs_vec = Rcpp::as<std::vector<int>>(re_params["n_coefs_vec"]);
  std::vector<int> re_correlated_vec = Rcpp::as<std::vector<int>>(re_params["correlated_vec"]);
  std::vector<int> re_n_chol_vec = Rcpp::as<std::vector<int>>(re_params["n_chol_vec"]);
  Rcpp::List slope_matrices_list = re_params["slope_matrices"];

  data.re_group = re_group;
  data.n_re_groups = n_re_groups;
  data.n_re_terms = n_re_terms;

  data.has_re_slopes = has_re_slopes;
  data.has_re_correlated_slopes = has_re_correlated_slopes;

  // RE parameterization: 0 = centered, 1 = non-centered
  data.re_parameterization = Rcpp::as<int>(re_params["parameterization"]);

  if (n_re_terms > 0) {
    // Multi-term RE structure (with or without slopes)
    data.re_group_multi.resize(n_re_terms);
    data.re_n_groups_multi.resize(n_re_terms);
    data.re_offsets.resize(n_re_terms);
    data.re_n_coefs.resize(n_re_terms);
    data.re_correlated.resize(n_re_terms);
    data.re_n_chol.resize(n_re_terms);
    data.re_n_slopes.resize(n_re_terms);
    data.re_slope_matrices.resize(n_re_terms);

    int offset = 0;
    int total_re_params = 0;
    int total_sigma_params = 0;
    int total_chol_params = 0;

    for (int t = 0; t < n_re_terms; t++) {
      data.re_n_groups_multi[t] = re_n_groups_vec[t];
      data.re_offsets[t] = offset;

      int n_coefs = has_re_slopes ? re_n_coefs_vec[t] : 1;
      data.re_n_coefs[t] = n_coefs;
      data.re_correlated[t] = has_re_slopes ? (re_correlated_vec[t] != 0) : false;
      data.re_n_chol[t] = has_re_slopes ? re_n_chol_vec[t] : 0;
      data.re_n_slopes[t] = n_coefs - 1;

      if (has_re_slopes && data.re_n_slopes[t] > 0 && slope_matrices_list.size() > t) {
        SEXP mat_sexp = slope_matrices_list[t];
        if (!Rf_isNull(mat_sexp)) {
          Rcpp::NumericMatrix slope_mat(mat_sexp);
          int n_rows = slope_mat.nrow();
          int n_cols = slope_mat.ncol();
          data.re_slope_matrices[t].resize(n_rows * n_cols);
          for (int i = 0; i < n_rows; i++) {
            for (int j = 0; j < n_cols; j++) {
              data.re_slope_matrices[t][i * n_cols + j] = slope_mat(i, j);
            }
          }
        }
      }

      offset += re_n_groups_vec[t];
      total_re_params += re_n_groups_vec[t] * n_coefs;
      total_sigma_params += n_coefs;
      total_chol_params += data.re_n_chol[t];

      data.re_group_multi[t].resize(data.N);
      for (int i = 0; i < data.N; i++) {
        data.re_group_multi[t][i] = re_group_matrix(i, t);
      }
    }
    data.total_re_groups = offset;

    // Obs-major flat array: inner loop over terms for each observation.
    data.re_group_multi_flat.resize(n_re_terms * data.N);
    for (int i = 0; i < data.N; i++) {
      for (int t = 0; t < n_re_terms; t++) {
        data.re_group_multi_flat[i * n_re_terms + t] = data.re_group_multi[t][i];
      }
    }
    data.total_re_params = total_re_params;
    data.total_sigma_params = total_sigma_params;
    data.total_chol_params = total_chol_params;
  } else {
    data.total_re_groups = n_re_groups;
    data.total_re_params = n_re_groups;
    data.total_sigma_params = (n_re_groups > 0) ? 1 : 0;
    data.total_chol_params = 0;
  }
}

// The restricted-spatial-regression block, as hmc_rsr_params() builds it:
// `has_rsr`, the row-major basis `basis` and its column count `rank`.
inline void set_rsr_basis(ModelData& data, bool has_rsr,
                          std::vector<double> basis, int rank) {
  data.has_rsr = has_rsr;
  data.rsr_basis = std::move(basis);
  data.rsr_rank = rank;
  if (!data.has_rsr) {
    data.rsr_basis.clear();
    data.rsr_rank = 0;
  } else if (data.rsr_rank <= 0 ||
             static_cast<int>(data.rsr_basis.size()) != data.N * data.rsr_rank) {
    Rcpp::stop("The restricted spatial regression basis has %d entries for %d "
               "observations and rank %d.",
               static_cast<int>(data.rsr_basis.size()), data.N, data.rsr_rank);
  }
}

inline void apply_rsr_params(ModelData& data, const Rcpp::List& rsr_params) {
  set_rsr_basis(data, Rcpp::as<bool>(rsr_params["has_rsr"]),
                Rcpp::as<std::vector<double>>(rsr_params["basis"]),
                Rcpp::as<int>(rsr_params["rank"]));
}

// The temporal block: the RW1 / RW2 / AR1 margin, the continuous-time GP
// margin, the multi-scale decomposition, and the prior anchors each of them
// reads. Every field the bundle carries is assigned here, so a knob the R
// backend puts on the list is one both entry points honour rather than one
// whichever extraction happens to name it.
inline void apply_temporal_params(ModelData& data, const Rcpp::List& temporal_params) {
  std::string temporal_type_str = Rcpp::as<std::string>(temporal_params["type"]);
  std::vector<int> temporal_time_idx = Rcpp::as<std::vector<int>>(temporal_params["time_idx"]);
  std::vector<int> temporal_group_idx = Rcpp::as<std::vector<int>>(temporal_params["group_idx"]);
  int n_times = Rcpp::as<int>(temporal_params["n_times"]);
  int n_temporal_groups = Rcpp::as<int>(temporal_params["n_groups"]);
  int n_temporal_params = Rcpp::as<int>(temporal_params["n_params"]);
  bool temporal_cyclic = Rcpp::as<bool>(temporal_params["cyclic"]);
  bool temporal_shared = Rcpp::as<bool>(temporal_params["shared"]);
  double tau_temporal_shape = Rcpp::as<double>(temporal_params["tau_shape"]);
  double tau_temporal_rate = Rcpp::as<double>(temporal_params["tau_rate"]);
  double temporal_rho_prior_a = temporal_params.containsElementNamed("rho_prior_a")
      ? Rcpp::as<double>(temporal_params["rho_prior_a"]) : ratiod_ar1::RHO_PRIOR_A;
  double temporal_rho_prior_b = temporal_params.containsElementNamed("rho_prior_b")
      ? Rcpp::as<double>(temporal_params["rho_prior_b"]) : ratiod_ar1::RHO_PRIOR_B;

  // The single-component types read straight off the shared parser; the two
  // names below carry data of their own on top of it.
  data.temporal_type = ratiod_temporal::parse_temporal_type(temporal_type_str);
  data.has_temporal_gp = false;

  // AR1 alone carries a second coordinate; the string is read for every type
  // so a model that never reaches AR1 still leaves the flag at centred.
  {
    std::string ar1_param_str = "centered";
    if (temporal_params.containsElementNamed("ar1_parameterization")) {
      ar1_param_str = Rcpp::as<std::string>(temporal_params["ar1_parameterization"]);
    }
    data.temporal_parameterization = (ar1_param_str == "noncentered") ? 1 : 0;
  }

  if (temporal_type_str == "gp") {
    data.has_temporal_gp = true;
    data.temporal_gp_data.n_obs = data.n_times;  // Use n_times, not N (total obs)
    data.temporal_gp_data.n_groups = n_temporal_groups;
    data.temporal_gp_data.time_values = Rcpp::as<std::vector<double>>(temporal_params["time_values"]);
    data.temporal_gp_data.group_index = temporal_group_idx;

    std::string cov_type_str = Rcpp::as<std::string>(temporal_params["cov_type"]);
    data.temporal_gp_data.cov_type = ratiod_temporal_gp::parse_temporal_cov_type(cov_type_str);
    data.temporal_gp_data.nu = Rcpp::as<double>(temporal_params["nu"]);
    data.temporal_gp_data.period = Rcpp::as<double>(temporal_params["period"]);
    data.temporal_gp_data.shared = temporal_shared;

    data.temporal_gp_sigma2_prior_U = Rcpp::as<double>(temporal_params["gp_sigma2_prior_U"]);
    data.temporal_gp_sigma2_prior_alpha = Rcpp::as<double>(temporal_params["gp_sigma2_prior_alpha"]);
    data.temporal_gp_phi_prior_lower = Rcpp::as<double>(temporal_params["gp_phi_prior_lower"]);
    data.temporal_gp_phi_prior_upper = Rcpp::as<double>(temporal_params["gp_phi_prior_upper"]);

    // Parameterization: 0=centered, 1=non-centered (default NC)
    std::string gp_param_str = "noncentered";
    if (temporal_params.containsElementNamed("gp_parameterization")) {
        gp_param_str = Rcpp::as<std::string>(temporal_params["gp_parameterization"]);
    }
    data.temporal_gp_parameterization = (gp_param_str == "centered") ? 0 : 1;
  } else if (temporal_type_str == "multiscale") {
    data.temporal_type = TemporalType::NONE;  // MS_t uses its own data path
    data.has_multiscale_temporal = true;

    data.multiscale_temporal_data.n_times = n_times;
    data.multiscale_temporal_data.n_groups = n_temporal_groups;
    data.multiscale_temporal_data.n_obs = data.N;
    data.multiscale_temporal_data.time_index = temporal_time_idx;
    data.multiscale_temporal_data.group_index = temporal_group_idx;
    data.multiscale_temporal_data.shared = temporal_shared;

    int ms_seasonal_period = 0;
    if (temporal_params.containsElementNamed("seasonal_period"))
      ms_seasonal_period = Rcpp::as<int>(temporal_params["seasonal_period"]);
    data.multiscale_temporal_data.seasonal_period = ms_seasonal_period;

    std::string ms_trend_type = "rw1";
    if (temporal_params.containsElementNamed("trend_type"))
      ms_trend_type = Rcpp::as<std::string>(temporal_params["trend_type"]);
    std::string ms_short_type = "ar1";
    if (temporal_params.containsElementNamed("short_term_type"))
      ms_short_type = Rcpp::as<std::string>(temporal_params["short_term_type"]);
    data.multiscale_temporal_data.trend_type = ratiod_temporal::parse_temporal_type(ms_trend_type);
    data.multiscale_temporal_data.short_term_type = ratiod_temporal::parse_temporal_type(ms_short_type);
    // Its own key, beside gp_parameterization and ar1_parameterization: a bare
    // "parameterization" in this bundle belongs to whichever structure the
    // model actually carries, and the temporal block carries several.
    if (temporal_params.containsElementNamed("ms_parameterization")) {
      data.multiscale_temporal_data.noncentered =
          (Rcpp::as<std::string>(temporal_params["ms_parameterization"]) == "noncentered");
    }

    data.ms_sigma2_trend_prior_U = 1.0;
    data.ms_sigma2_trend_prior_alpha = 0.01;
    data.ms_sigma2_seasonal_prior_U = 1.0;
    data.ms_sigma2_seasonal_prior_alpha = 0.01;
    data.ms_sigma2_short_prior_U = 1.0;
    data.ms_sigma2_short_prior_alpha = 0.01;
    if (temporal_params.containsElementNamed("sigma2_trend_prior_U"))
      data.ms_sigma2_trend_prior_U = Rcpp::as<double>(temporal_params["sigma2_trend_prior_U"]);
    if (temporal_params.containsElementNamed("sigma2_trend_prior_alpha"))
      data.ms_sigma2_trend_prior_alpha = Rcpp::as<double>(temporal_params["sigma2_trend_prior_alpha"]);
    if (temporal_params.containsElementNamed("sigma2_short_prior_U"))
      data.ms_sigma2_short_prior_U = Rcpp::as<double>(temporal_params["sigma2_short_prior_U"]);
    if (temporal_params.containsElementNamed("sigma2_short_prior_alpha"))
      data.ms_sigma2_short_prior_alpha = Rcpp::as<double>(temporal_params["sigma2_short_prior_alpha"]);
  }

  data.temporal_time_idx = temporal_time_idx;
  data.temporal_group_idx = temporal_group_idx;
  data.n_times = n_times;
  data.n_temporal_groups = n_temporal_groups;
  data.n_temporal_params = n_temporal_params;
  data.temporal_cyclic = temporal_cyclic;
  data.temporal_shared = temporal_shared;
  data.tau_temporal_shape = tau_temporal_shape;
  data.tau_temporal_rate = tau_temporal_rate;
  // Beta anchors on u = (rho + 1) / 2 for an AR1 correlation.
  data.temporal_rho_prior_a = temporal_rho_prior_a;
  data.temporal_rho_prior_b = temporal_rho_prior_b;
}

}  // namespace ratiod_hmc

#endif  // RATIOD_HMC_MODEL_DATA_BLOCKS_H
