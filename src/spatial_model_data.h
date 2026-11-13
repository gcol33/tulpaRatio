#ifndef TULPARATIO_SPATIAL_MODEL_DATA_H
#define TULPARATIO_SPATIAL_MODEL_DATA_H

#include <RcppEigen.h>
#include <tulpa/model_data.h>
#include "covariance.h"
#include "engine_options.h"

namespace tulpaRatio {

inline std::vector<int> zero_based_indices(SEXP values) {
    auto out = Rcpp::as<std::vector<int>>(values);
    for (int& index : out) --index;
    return out;
}

inline tulpa::HSGPData spectral_basis(const Rcpp::List& sp, int N) {
    Rcpp::NumericMatrix phi = sp["basis"];
    tulpa::HSGPData out;
    out.n_obs = N;
    out.m_total = phi.ncol();
    out.m_per_dim = Rcpp::as<int>(sp["hsgp_m"]);
    out.shared = Rcpp::as<bool>(sp["shared"]);
    if (phi.nrow() != N) Rcpp::stop("HSGP basis must have N rows");
    out.phi_flat.resize(static_cast<size_t>(N) * out.m_total);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < out.m_total; ++j)
            out.phi_flat[i * out.m_total + j] = phi(i, j);
    out.eigenvalues = Rcpp::as<std::vector<double>>(sp["eigenvalues"]);
    if (out.eigenvalues.size() != static_cast<size_t>(out.m_total))
        Rcpp::stop("HSGP eigenvalues must match the basis width");
    return out;
}

inline int apply_spatial_model_data(tulpa::ModelData& data, const Rcpp::List& sp) {
    if (!sp.containsElementNamed("type")) return 0;
    const std::string type = Rcpp::as<std::string>(sp["type"]);
    if (type == "none") return 0;
    const std::string param = Rcpp::as<std::string>(sp["parameterization"]);
    if (param == "collapsed")
        Rcpp::stop("tulpaRatio native collapsed spatial integration is unavailable. Use full-field NUTS through tratio() or parameterization = 'auto'.");
    const int n = Rcpp::as<int>(sp["n_units"]);
    const bool shared = Rcpp::as<bool>(sp["shared"]);
    for (int k = 1; k < data.n_processes; ++k) data.sharing.spatial[k] = shared;
    const Rcpp::List restriction = sp["rsr"];
    if (Rcpp::as<bool>(restriction["active"])) {
        data.spatial_restriction.rank = Rcpp::as<int>(restriction["rank"]);
        data.spatial_restriction.basis = Rcpp::as<std::vector<double>>(restriction["basis"]);
    }
    if (type == "icar" || type == "bym2" || type == "car_proper") {
        data.spatial_type = type == "icar" ? tulpa::SpatialType::ICAR
            : type == "bym2" ? tulpa::SpatialType::BYM2 : tulpa::SpatialType::CAR_PROPER;
        data.spatial_component_centering = true;
        data.auto_spatial_coordinates = param == "auto";
        data.spatial_sigma_prior_U = Rcpp::as<double>(sp["sigma_prior_U"]);
        data.spatial_sigma_prior_alpha = Rcpp::as<double>(sp["sigma_prior_alpha"]);
        data.spatial_group = Rcpp::as<std::vector<int>>(sp["group"]);
        data.set_spatial_adjacency(n, Rcpp::as<std::vector<int>>(sp["adj_row_ptr"]),
            Rcpp::as<std::vector<int>>(sp["adj_col_idx"]),
            Rcpp::as<std::vector<int>>(sp["n_neighbors"]));
        if (data.spatial_group.size() != static_cast<size_t>(data.N))
            Rcpp::stop("Spatial group length must equal N");
        for (int g : data.spatial_group)
            if (g < 1 || g > n) Rcpp::stop("Spatial group index outside the field");
        std::vector<double> precision;
        if (!Rf_isNull(sp["node_prec"])) precision = Rcpp::as<std::vector<double>>(sp["node_prec"]);
        data.set_bym2_scaling(Rcpp::as<double>(sp["bym2_scale"]), std::move(precision));
        if (!Rf_isNull(sp["Q_inv"]) && !Rf_isNull(sp["L_Q"])) {
            data.spatial_Q_inv = Rcpp::as<std::vector<double>>(sp["Q_inv"]);
            data.spatial_L_Q = Rcpp::as<std::vector<double>>(sp["L_Q"]);
        }
        if (type == "car_proper") {
            data.car_rho_lower = Rcpp::as<double>(sp["rho_lower"]);
            data.car_rho_upper = Rcpp::as<double>(sp["rho_upper"]);
            data.car_rho_prior_a = Rcpp::as<double>(sp["rho_prior_a"]);
            data.car_rho_prior_b = Rcpp::as<double>(sp["rho_prior_b"]);
            data.car_center = Rcpp::as<bool>(sp["center"]);
            Eigen::MatrixXd A = Eigen::MatrixXd::Zero(n, n);
            for (int i = 0; i < n; ++i) {
                if (data.n_neighbors[i] < 1) Rcpp::stop("Proper CAR requires positive node degrees");
                for (int k = data.adj_row_ptr[i]; k < data.adj_row_ptr[i + 1]; ++k) {
                    const int j = data.adj_col_idx[k];
                    A(i, j) = 1.0 / std::sqrt(double(data.n_neighbors[i]) * data.n_neighbors[j]);
                }
            }
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(A);
            if (solver.info() != Eigen::Success) Rcpp::stop("CAR adjacency eigendecomposition failed");
            data.car_adj_eigenvalues.assign(solver.eigenvalues().data(), solver.eigenvalues().data() + n);
        }
        return type == "bym2" ? 2 + 2 * n : (type == "car_proper" ? 2 : 1) + n;
    }
    Rcpp::List gp = sp["gp"];
    data.gp_sigma2_prior_U = Rcpp::as<double>(gp["sigma2_prior_U"]);
    data.gp_sigma2_prior_alpha = Rcpp::as<double>(gp["sigma2_prior_alpha"]);
    if (type == "gp") {
        data.spatial_type = tulpa::SpatialType::GP;
        data.has_gp = true;
        data.gp_parameterization = (param == "noncentered" || param == "auto") ? 1 : 0;
        data.auto_gp_coordinates = param == "auto";
        data.conditional_gp_coordinates = gp.containsElementNamed("conditional_coordinates") &&
            Rcpp::as<bool>(gp["conditional_coordinates"]);
        auto& g = data.gp_data;
        g.n_obs = n;
        g.nn = Rcpp::as<int>(gp["nn"]);
        g.coords = Rcpp::as<std::vector<double>>(gp["coords"]);
        g.nn_idx = Rcpp::as<std::vector<int>>(gp["nn_idx"]);
        g.nn_dist = Rcpp::as<std::vector<double>>(gp["nn_dist"]);
        g.nn_neighbor_dist = Rcpp::as<std::vector<double>>(gp["nn_neighbor_dist"]);
        g.nn_order = zero_based_indices(gp["nn_order"]);
        g.nn_order_inv = zero_based_indices(gp["nn_order_inv"]);
        g.obs_to_loc = zero_based_indices(gp["gp_obs_to_loc"]);
        if (g.obs_to_loc.size() != static_cast<size_t>(data.N)) Rcpp::stop("GP location map must have N entries");
        for (int i : g.obs_to_loc) if (i < 0 || i >= n) Rcpp::stop("GP location index outside field");
        const std::string cov = Rcpp::as<std::string>(gp["cov_type"]);
        g.cov_type=ratiod_cov::parse_cov_type(cov,Rcpp::as<double>(gp["nu"]));
        g.nu = Rcpp::as<double>(gp["nu"]);
        g.shared = shared;
        if (cov == "gaussian") {
            const double factor = 1.0 / ratiod_cov::range_scale(g.cov_type);
            for (double& x : g.coords) x *= factor;
            for (double& x : g.nn_dist) x *= factor;
            for (double& x : g.nn_neighbor_dist) x *= factor;
        }
        data.gp_range_prior.type = tulpa::RangePriorType::UNIFORM;
        data.gp_range_prior.lower = Rcpp::as<double>(gp["phi_prior_lower"]);
        data.gp_range_prior.upper = Rcpp::as<double>(gp["phi_prior_upper"]);
        return 2 + n;
    }
    if (type == "hsgp") {
        data.spatial_type = tulpa::SpatialType::HSGP;
        data.has_hsgp = true;
        data.hsgp_data = spectral_basis(sp, data.N);
        data.hsgp_range_prior.type = tulpa::RangePriorType::LOGNORMAL;
        data.hsgp_sigma2_prior_U = data.gp_sigma2_prior_U;
        data.hsgp_sigma2_prior_alpha = data.gp_sigma2_prior_alpha;
        return 2 + data.hsgp_data.m_total;
    }
    if (type != "multiscale_gp") Rcpp::stop("Unknown spatial type: %s", type);
    data.spatial_type = tulpa::SpatialType::MULTISCALE_GP;
    data.has_multiscale_gp = true;
    data.msgp_is_hsgp = true;
    data.msgp_hsgp_data = spectral_basis(sp, data.N);
    data.multiscale_gp_data.shared = shared;
    const std::string sampler = Rcpp::as<std::string>(gp["sampler"]);
    data.multiscale_gp_data.sampler=ratiod_enum::parse_enum(MSGP_NAMES,sampler,"multiscale sampler");
    data.ms_sigma2_local_prior_U = Rcpp::as<double>(gp["sigma2_local_prior_U"]);
    data.ms_sigma2_local_prior_alpha = Rcpp::as<double>(gp["sigma2_local_prior_alpha"]);
    data.ms_sigma2_regional_prior_U = Rcpp::as<double>(gp["sigma2_regional_prior_U"]);
    data.ms_sigma2_regional_prior_alpha = Rcpp::as<double>(gp["sigma2_regional_prior_alpha"]);
    auto lognormal = [&](const char* lower, const char* upper) {
        tulpa::RangePrior p;
        p.type = tulpa::RangePriorType::LOGNORMAL;
        p.meanlog = 0.5 * (std::log(Rcpp::as<double>(gp[lower])) + std::log(Rcpp::as<double>(gp[upper])));
        p.sdlog = 0.5;
        return p;
    };
    data.ms_local_range_prior = lognormal("range_local_lower", "range_local_upper");
    data.ms_regional_range_prior = lognormal("range_regional_lower", "range_regional_upper");
    return 4 + 2 * data.msgp_hsgp_data.m_total;
}

} // namespace tulpaRatio
#endif
