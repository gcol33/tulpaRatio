#ifndef TULPARATIO_RATIO_H_GRADIENT_H
#define TULPARATIO_RATIO_H_GRADIENT_H
#include "lik_helpers.h"
#include <tulpa/scalar_priors.h>
#include <Rcpp.h>
#include <tulpa/likelihood.h>
namespace tulpaRatio { namespace h_kernel {
inline void assert_no_latent_structure(const tulpa::ParamLayout& layout) {
    const bool any_latent =
        layout.has_re || layout.has_re_slopes ||
        layout.has_spatial || layout.is_gp || layout.is_multiscale_gp ||
        layout.is_hsgp ||
        layout.has_temporal || layout.has_multiscale_temporal ||
        layout.is_temporal_gp || layout.has_zi || layout.has_oi ||
        layout.has_svc || layout.has_tvc || layout.has_latent ||
        layout.has_spatiotemporal;
    if (any_latent) {
        Rcpp::stop("tulpaRatio H-kernel: latent structure (spatial/temporal/RE/"
                   "ZI/SVC/TVC/ST) requires the engine gradient.");
    }
}

inline void apply_beta_prior(
    const std::vector<double>& params,
    const tulpa::ModelData& data,
    const tulpa::ParamLayout& layout,
    std::vector<double>& grad,
    double* log_post_out
) {
    const double sigma_beta = data.sigma_beta;
    const double tau_beta = 1.0 / (sigma_beta * sigma_beta);


    for (int k = 0; k < data.n_processes; ++k) {
        const int start = layout.process_beta_start[k];
        const int p_k   = layout.process_beta_count[k];
        for (int j = 0; j < p_k; ++j) {
            const double b = params[start + j];
            grad[start + j] += tulpa::math::log_prior_normal_gradient(b,tau_beta);
            if (log_post_out) {
                *log_post_out += tulpa::math::log_prior_normal(b,tau_beta);
            }
        }
    }
}

inline std::vector<double> compute_eta_matrix(
    const std::vector<double>& params,
    const tulpa::ModelData& data,
    const tulpa::ParamLayout& layout
) {
    const int N  = data.N;
    const int np = data.n_processes;
    std::vector<double> eta(static_cast<size_t>(np) * N, 0.0);

    for (int k = 0; k < np; ++k) {
        const auto& proc = data.processes[k];
        const int p_k    = proc.p;
        const int bstart = layout.process_beta_start[k];
        double*       eta_k   = eta.data() + static_cast<size_t>(k) * N;
        const double* X_k     = proc.X_flat.data();
        const double* beta_k  = params.data() + bstart;

        if (p_k > 0) {
            for (int i = 0; i < N; ++i) {
                double s = 0.0;
                const double* row = X_k + static_cast<size_t>(i) * p_k;
                for (int j = 0; j < p_k; ++j) s += row[j] * beta_k[j];
                eta_k[i] = s;
            }
        }
        if (!proc.offset.empty()) {
            if (static_cast<int>(proc.offset.size()) != N) {
                Rcpp::stop("ProcessData[%d]: offset length %d != N=%d",
                           k, (int)proc.offset.size(), N);
            }
            for (int i = 0; i < N; ++i) eta_k[i] += proc.offset[i];
        }
    }
    return eta;
}

template <class PerObsFn>
inline double run_obs_loop(
    const std::vector<double>& params,
    const tulpa::ModelData& data,
    const tulpa::ParamLayout& layout,
    const double* eta,
    PerObsFn fn,
    std::vector<double>& grad,
    int n_extra_grads
) {
    const int N  = data.N;
    const int np = data.n_processes;
    std::vector<double> resid(np, 0.0);
    std::vector<double> extra(n_extra_grads, 0.0);
    std::vector<double> extra_acc(n_extra_grads, 0.0);
    double log_lik = 0.0;

    for (int i = 0; i < N; ++i) {
        for (int k = 0; k < np; ++k) resid[k] = 0.0;
        for (int e = 0; e < n_extra_grads; ++e) extra[e] = 0.0;

        const double ll_i = fn(i, eta, resid.data(), extra.data());
        log_lik += ll_i;

        for (int k = 0; k < np; ++k) {
            const auto& proc = data.processes[k];
            const int p_k = proc.p;
            if (p_k == 0 || resid[k] == 0.0) continue;
            const int bstart = layout.process_beta_start[k];
            const double* row = &proc.X_flat[static_cast<size_t>(i) * p_k];
            for (int j = 0; j < p_k; ++j) {
                grad[bstart + j] += resid[k] * row[j];
            }
        }

        for (int e = 0; e < n_extra_grads; ++e) extra_acc[e] += extra[e];
    }

    for (int e = 0; e < n_extra_grads; ++e) {
        grad[layout.extra_offset + e] += extra_acc[e];
    }
    return log_lik;
}

using ratiod::math::log_prior_gamma;
using ratiod::math::log_prior_gamma_grad;
using tulpa::math::log_prior_half_cauchy;
using ratiod::math::log_prior_half_cauchy_grad;

using namespace margins;

template<class Num, class Den, class YN, class YD>
Ratio<Num,Den> bind_ratio(const YN& yn, const YD& yd, int N,
                         const std::vector<double>& params, int offset) {
    Response n, d;
    if constexpr (std::is_same<typename YN::value_type,int>::value) n.counts = yn.data();
    else n.continuous = yn.data();
    if constexpr (std::is_same<typename YD::value_type,int>::value) d.counts = yd.data();
    else d.continuous = yd.data();
    double pn = 1.0, pd = 1.0;
    if constexpr (Num::n_extra) pn = std::exp(params[offset]);
    if constexpr (Den::n_extra) pd = std::exp(params[offset + Num::n_extra]);
    return {Num(pn),Den(pd),n,d,N};
}

template<class Margin>
inline double apply_extra(const Margin& margin, int e,
                           const std::vector<double>& params,
                           const tulpa::ModelData& data,
                           const tulpa::ParamLayout& layout,
                           std::vector<double>& grad) {
    if constexpr (Margin::n_extra == 0) return 0.0;
    else {
        const int j = layout.extra_offset + e;
        const double log_p = params[j];
        if constexpr (!Margin::log_extra) grad[j] *= margin.phi;
        if constexpr (Margin::prior == Prior::Gamma) {
            grad[j] += log_prior_gamma_grad(log_p,data.phi_prior_shape,data.phi_prior_rate);
            return log_prior_gamma(log_p,data.phi_prior_shape,data.phi_prior_rate);
        } else {
            grad[j] += log_prior_half_cauchy_grad(log_p,data.sigma_re_scale);
            return log_prior_half_cauchy(log_p,data.sigma_re_scale);
        }
    }
}

template<class Num, class Den = NoMargin>
void gradient(const std::vector<double>& params, const tulpa::ModelData& data,
             const tulpa::ParamLayout& layout, std::vector<double>& grad,
             double* log_post_out) {
    assert_no_latent_structure(layout);
    grad.assign(params.size(),0.0);
    if (log_post_out) *log_post_out = 0.0;
    apply_beta_prior(params,data,layout,grad,log_post_out);
    const auto& r = *static_cast<const lik::ResponseData*>(data.model_response_data);
    const auto fn = [&]() {
        if constexpr (std::is_same<Den,NoMargin>::value) {
            double p = 1.0;
            if constexpr (Num::n_extra) p = std::exp(params[layout.extra_offset]);
            return Ratio<Num>{Num(p),NoMargin(),{r.y_num.data(),nullptr,r.y_denom.data()},{},data.N};
        } else if constexpr (std::is_same<Num,Gamma>::value || std::is_same<Num,Lognormal>::value) {
            return bind_ratio<Num,Den>(r.y_num_cont,r.y_denom_cont,data.N,params,layout.extra_offset);
        } else if constexpr (std::is_same<Den,Gamma>::value) {
            return bind_ratio<Num,Den>(r.y_num,r.y_denom_cont,data.N,params,layout.extra_offset);
        } else return bind_ratio<Num,Den>(r.y_num,r.y_denom,data.N,params,layout.extra_offset);
    }();
    const auto eta = compute_eta_matrix(params,data,layout);
    const double ll = run_obs_loop(params,data,layout,eta.data(),fn,grad,decltype(fn)::n_extra);
    double lp = ll;
    lp += apply_extra(fn.num,0,params,data,layout,grad);
    lp += apply_extra(fn.den,decltype(fn.num)::n_extra,params,data,layout,grad);
    if (log_post_out) *log_post_out += lp;
}


} }
#endif
