// gibbs_spatial.cpp
// Rcpp interface for the Gibbs spatial sampler

#include "gibbs_spatial.h"
#include "omp_chain_team.h"
#include <Rcpp.h>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

// Convert one chain's pure-C++ result into the R list the backend expects.
// Called only from the serial section: it allocates SEXPs, which is not
// permitted on an OpenMP worker thread.
Rcpp::List gibbs_result_to_r(const gibbs::GibbsResult& res,
                             const gibbs::GibbsData& d,
                             int S) {
    Rcpp::CharacterVector param_names(res.param_names.size());
    for (size_t i = 0; i < res.param_names.size(); i++)
        param_names[i] = res.param_names[i];

    Rcpp::List result = Rcpp::List::create(
        Rcpp::Named("draws") = Rcpp::NumericMatrix(res.n_save, res.n_params,
                                                     res.draws_flat.begin()),
        Rcpp::Named("phi_draws") = Rcpp::NumericMatrix(res.n_save, S,
                                                        res.phi_draws_flat.begin()),
        Rcpp::Named("param_names") = param_names,
        Rcpp::Named("n_save") = res.n_save,
        Rcpp::Named("n_params") = res.n_params,
        Rcpp::Named("S") = S,
        Rcpp::Named("accept_phi") = Rcpp::wrap(res.accept_phi),
        Rcpp::Named("accept_beta") = res.accept_beta,
        Rcpp::Named("accept_disp") = res.accept_disp
    );

    if (d.has_tvc && res.tvc_n_w > 0) {
        result["tvc_w_draws"] = Rcpp::NumericMatrix(res.n_save, res.tvc_n_w,
                                                     res.tvc_w_draws_flat.begin());
        result["tvc_tau_draws"] = Rcpp::NumericMatrix(res.n_save, d.tvc_n_terms,
                                                       res.tvc_tau_draws_flat.begin());
        result["accept_tvc"] = res.accept_tvc;
    }

    return result;
}

}  // namespace

// [[Rcpp::export]]
Rcpp::List cpp_gibbs_spatial(Rcpp::List data_list) {
    using namespace gibbs;

    // Extract data from R list
    Rcpp::IntegerVector y_num_r = data_list["y_num"];
    Rcpp::NumericVector X_num_r = data_list["X_num"];
    Rcpp::IntegerVector spatial_group_r = data_list["spatial_group"];
    Rcpp::IntegerVector adj_row_ptr_r = data_list["adj_row_ptr"];
    Rcpp::IntegerVector adj_col_idx_r = data_list["adj_col_idx"];
    Rcpp::IntegerVector n_neighbors_r = data_list["n_neighbors"];

    int N = Rcpp::as<int>(data_list["N"]);
    int S = Rcpp::as<int>(data_list["S"]);
    int p_num = Rcpp::as<int>(data_list["p_num"]);
    int p_denom = Rcpp::as<int>(data_list["p_denom"]);
    int n_iter = Rcpp::as<int>(data_list["n_iter"]);
    int n_warmup = Rcpp::as<int>(data_list["n_warmup"]);
    int thin = Rcpp::as<int>(data_list["thin"]);
    unsigned int seed = Rcpp::as<unsigned int>(data_list["seed"]);
    bool verbose = Rcpp::as<bool>(data_list["verbose"]);
    std::string family_str = Rcpp::as<std::string>(data_list["family"]);

    // One seed per chain, derived on the R side so the fit stays reproducible.
    std::vector<unsigned int> chain_seeds;
    if (data_list.containsElementNamed("chain_seeds")) {
        Rcpp::NumericVector cs = Rcpp::as<Rcpp::NumericVector>(data_list["chain_seeds"]);
        chain_seeds.reserve(cs.size());
        for (int c = 0; c < cs.size(); c++)
            chain_seeds.push_back(static_cast<unsigned int>(cs[c]));
    }
    if (chain_seeds.empty()) chain_seeds.push_back(seed);
    const int n_chains = static_cast<int>(chain_seeds.size());

    // Across-chain core budget. Bounds the OpenMP team so it stays the same
    // size across successive fits in one session; a team that grows then
    // shrinks corrupts the GOMP thread pool and faults at process teardown
    // on Windows.
    int n_cores = data_list.containsElementNamed("n_cores") ?
                  Rcpp::as<int>(data_list["n_cores"]) : 1;
    if (n_cores < 1) n_cores = 1;

    const GibbsFamily family =
        ratiod_enum::parse_enum(GIBBS_FAMILY_NAMES, family_str, "Gibbs sampler family");

    bool is_binomial = (family == GibbsFamily::BINOMIAL);

    // Build GibbsData struct
    GibbsData d;
    d.N = N;
    d.S = S;
    d.p_num = p_num;
    d.p_denom = is_binomial ? 0 : p_denom;
    d.family = family;
    // BYM2 settings
    bool is_bym2 = data_list.containsElementNamed("is_bym2") &&
                    Rcpp::as<bool>(data_list["is_bym2"]);
    d.is_bym2 = is_bym2;
    d.bym2_scale = data_list.containsElementNamed("bym2_scale") ?
                   Rcpp::as<double>(data_list["bym2_scale"]) : 1.0;

    // Priors. The R side resolves these from the same defaults the HMC
    // backend uses, so the two agree on the model for a given call.
    auto prior_or = [&](const char* nm, double fallback) {
        return data_list.containsElementNamed(nm) ?
               Rcpp::as<double>(data_list[nm]) : fallback;
    };
    d.sigma_beta = prior_or("sigma_beta", 10.0);
    d.sigma_re_scale = prior_or("sigma_re_scale", 2.5);
    d.tau_spatial_shape = prior_or("tau_spatial_shape", 1.0);
    d.tau_spatial_rate = prior_or("tau_spatial_rate", 0.01);

    // Convert spatial_group from 1-based to 0-based
    std::vector<int> sg(N);
    for (int i = 0; i < N; i++) sg[i] = spatial_group_r[i] - 1;
    d.spatial_group = sg.data();

    d.y_num = y_num_r.begin();
    d.X_num = REAL(X_num_r);

    // Denominator data (depends on family)
    Rcpp::NumericVector X_denom_r;
    Rcpp::IntegerVector y_denom_int_r;
    Rcpp::NumericVector y_denom_cont_r;

    if (!is_binomial) {
        X_denom_r = Rcpp::as<Rcpp::NumericVector>(data_list["X_denom"]);
        d.X_denom = REAL(X_denom_r);
    } else {
        d.X_denom = nullptr;
    }

    if (family == GibbsFamily::NEGBIN_NEGBIN) {
        y_denom_int_r = Rcpp::as<Rcpp::IntegerVector>(data_list["y_denom"]);
        d.y_denom = y_denom_int_r.begin();
        d.y_denom_cont = nullptr;
    } else if (family == GibbsFamily::BINOMIAL) {
        y_denom_int_r = Rcpp::as<Rcpp::IntegerVector>(data_list["y_denom"]);
        d.y_denom = y_denom_int_r.begin();
        d.y_denom_cont = nullptr;
    } else {
        // Poisson-Gamma, NegBin-Gamma: continuous denominator
        y_denom_cont_r = Rcpp::as<Rcpp::NumericVector>(data_list["y_denom_cont"]);
        d.y_denom_cont = REAL(y_denom_cont_r);
        d.y_denom = nullptr;
    }

    // Adjacency (convert from 1-based to 0-based)
    std::vector<int> adj_rp(adj_row_ptr_r.begin(), adj_row_ptr_r.end());
    std::vector<int> adj_ci(adj_col_idx_r.size());
    for (int k = 0; k < (int)adj_col_idx_r.size(); k++)
        adj_ci[k] = adj_col_idx_r[k] - 1;  // 0-based
    std::vector<int> nn(n_neighbors_r.begin(), n_neighbors_r.end());
    d.adj_row_ptr = adj_rp.data();
    d.adj_col_idx = adj_ci.data();
    d.n_neighbors = nn.data();

    // Build site->obs mapping
    build_site_obs_map(d);

    // Locate the intercepts the spatial field's level is moved into.
    build_intercept_index(d);

    // TVC data
    std::vector<int> tvc_ti, tvc_gi;
    std::vector<double> tvc_X;
    if (data_list.containsElementNamed("has_tvc") && Rcpp::as<bool>(data_list["has_tvc"])) {
        d.has_tvc = true;
        d.tvc_n_times = Rcpp::as<int>(data_list["tvc_n_times"]);
        d.tvc_n_groups = Rcpp::as<int>(data_list["tvc_n_groups"]);
        d.tvc_n_terms = Rcpp::as<int>(data_list["tvc_n_terms"]);
        d.tvc_shared = Rcpp::as<bool>(data_list["tvc_shared"]);
        tvc_ti = Rcpp::as<std::vector<int>>(data_list["tvc_time_index"]);
        tvc_gi = Rcpp::as<std::vector<int>>(data_list["tvc_group_index"]);
        tvc_X = Rcpp::as<std::vector<double>>(data_list["tvc_X"]);
        d.tvc_time_index = tvc_ti;
        d.tvc_group_index = tvc_gi;
        d.tvc_X = tvc_X;
        // Build time->obs mapping
        build_time_obs_map(d);
        if (verbose) {
            Rcpp::Rcout << "  TVC: " << d.tvc_n_terms << " term(s), "
                        << d.tvc_n_times << " time points" << std::endl;
        }
    }

    // Run Gibbs sampler (ICAR or BYM2)
    if (verbose) {
        Rcpp::Rcout << "Running Gibbs sampler (" << (is_bym2 ? "BYM2" : "ICAR") << ")..." << std::endl;
        Rcpp::Rcout << "  Sites: " << S << ", Observations: " << N << std::endl;
        Rcpp::Rcout << "  Iterations: " << n_iter << " (warmup: " << n_warmup << ")" << std::endl;
        Rcpp::Rcout << "  Chains: " << n_chains << std::endl;
        if (is_bym2) Rcpp::Rcout << "  BYM2 scale factor: " << d.bym2_scale << std::endl;
    }

    // `d` is read-only for the duration of the sampler: build_site_obs_map and
    // build_time_obs_map are the only writers and both ran above, so a single
    // const view is shared safely across chains. Every other piece of sampler
    // state, including the RNG, is function-local to run_gibbs_*.
    const GibbsData& d_shared = d;
    std::vector<GibbsResult> chain_res(n_chains);

    if (n_chains > 1) {
        // verbose is forced off inside the region: the progress blocks in
        // run_gibbs_* write to Rcpp::Rcout, and R's API must not be touched
        // from a worker thread. The team is session-stable and n_cores caps
        // how many chains run at once; see omp_chain_team.h. Without OpenMP
        // the helper runs the chains in order.
        ratiod_omp::for_each_chain(n_chains, n_cores, [&](int c) {
            chain_res[c] = is_bym2 ?
                run_gibbs_bym2(d_shared, n_iter, n_warmup, thin, chain_seeds[c], false) :
                run_gibbs_icar(d_shared, n_iter, n_warmup, thin, chain_seeds[c], false);
        });
    } else {
        chain_res[0] = is_bym2 ?
            run_gibbs_bym2(d_shared, n_iter, n_warmup, thin, chain_seeds[0], verbose) :
            run_gibbs_icar(d_shared, n_iter, n_warmup, thin, chain_seeds[0], verbose);
    }

    if (verbose) {
        Rcpp::Rcout << "Gibbs complete." << std::endl;
        for (int c = 0; c < n_chains; c++) {
            const GibbsResult& res = chain_res[c];
            double avg_phi_accept = 0.0;
            for (int s = 0; s < S; s++) avg_phi_accept += res.accept_phi[s];
            avg_phi_accept /= S;
            Rcpp::Rcout << "  Chain " << (c + 1)
                        << ": avg phi acceptance " << avg_phi_accept
                        << ", beta " << res.accept_beta
                        << ", dispersion " << res.accept_disp;
            if (d.has_tvc) Rcpp::Rcout << ", TVC " << res.accept_tvc;
            Rcpp::Rcout << std::endl;
        }
    }

    // Serial conversion: allocating R objects is only safe here.
    Rcpp::List out(n_chains);
    for (int c = 0; c < n_chains; c++)
        out[c] = gibbs_result_to_r(chain_res[c], d, S);

    return out;
}
