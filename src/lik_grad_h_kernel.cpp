// lik_grad_h_kernel.cpp
// =====================================================================
// MATH HEADER (per CLAUDE.md "Math ground first")
// =====================================================================
//
// All 7 ratio families share the same scaffolding for B2:
//   * params layout: [beta_0 | beta_1 | ... | extras]
//                    where beta_k starts at layout.process_beta_start[k] and
//                    has length layout.process_beta_count[k].
//                    Extras (log_phi / log_sigma / log_shape) live at
//                    layout.extra_offset, length layout.n_extra_params.
//   * eta_k[i] = X_k[i] . beta_k + offset_k[i]    (process k, obs i)
//   * Log posterior: log P(params|data) = log L(data|params) + log Pi(params)
//   * Log Pi(params) = sum_k sum_j  log N(beta_kj | 0, sigma_beta^2)
//                    + family-specific extra-parameter priors
//
// Per-observation gradient pattern
// --------------------------------
// For every family the per-obs likelihood depends on params only through the
// linear predictors eta_k[i] and a small set of extra parameters. Define
//   resid_k(i) := d log L_i / d eta_k[i]
//   extra_k(i) := d log L_i / d extra_k    (per-obs contribution)
// Then
//   d log L / d beta_kj = sum_i  resid_k(i) * X_k[i,j]
//   d log L / d extra_k = sum_i  extra_k(i)
//
// Each family supplies a per-obs functor that fills resid[k] and the extra
// gradients; the shared scaffolding does the X^T R accumulation, the
// Gaussian beta prior gradient, the extra-parameter priors, and the fused
// log-posterior accumulation.
//
// Per-family residuals (closed-form, ported from
// tulpa/src/hmc_gradient_analytical_lik_scalar.h)
// ---------------------------------------------------------------------
//   binomial:       resid_num = y - n*p,                 p = sigmoid(eta)
//   poisson_gamma:  resid_num   = y_num - mu_num,        mu_num = exp(eta_num)
//                   resid_denom = shape*(y_d/mu_d - 1),  mu_d  = exp(eta_d)
//                   d/d log_shape = sum_i [shape*(log(rate_i) + 1 + log(y_d_i)
//                                          - digamma(shape) - rate_i*y_d_i/shape)]
//   negbin_gamma:   numerator = NegBin scalar resid (see negbin block);
//                   denominator = Gamma resid (see gamma_gamma block).
//   negbin_negbin:  numerator + denominator both NegBin scalar resids
//     resid_k = y_k - mu_k*(y_k+phi_k)/(mu_k+phi_k)
//     d/dlog_phi_k = phi_k * [digamma(y+phi) - digamma(phi) + log(phi/(mu+phi))
//                              + (mu - y)/(mu+phi)]
//   gamma_gamma:    Gamma response on each process (only y > 0 contributes):
//     resid_k = phi_k*(y_k/mu_k - 1)
//     d/dlog_phi_k = phi_k * (log(rate) + 1 + log(y) - digamma(phi) - y/mu)
//   lognormal:      eta_k IS mu_k on log scale; sigma_k = exp(log_sigma_k):
//     resid_k = (log y_k - mu_k) / sigma_k^2
//     d/dlog_sigma_k = -1 + z_k^2,   z_k = (log y_k - mu_k)/sigma_k
//   beta_binomial:  p = sigmoid(eta), phi = exp(log_phi),
//                   alpha = p*phi, beta = (1-p)*phi:
//     dLL/dp = phi*(psi(y+a) - psi(n-y+b) - psi(a) + psi(b))
//     resid  = dLL/dp * p*(1-p)
//     dLL/dphi = p*psi(y+a) + (1-p)*psi(n-y+b) - psi(n+phi)
//                 - p*psi(a) - (1-p)*psi(b) + psi(phi)
//     d/dlog_phi = phi * dLL/dphi
//
// Extra-parameter priors (from B1b lik_helpers.h)
// ----------------------------------------------
//   Gamma(shape, rate) on log_phi:
//     log Pi(log_phi) = (shape-1)*log_phi - rate*phi + log_phi  (Jacobian)
//     d/dlog_phi = shape - rate * phi
//   Half-Cauchy(0, scale) on log_sigma:
//     log Pi(log_sigma) = -log(1 + (sigma/scale)^2) + log_sigma
//     d/dlog_sigma = -2*r^2/(1+r^2) + 1, r = sigma/scale
// =====================================================================

#include "lik_specs/lik_grad_h_kernel.h"
#include "lik_specs/lik_helpers.h"
#include "ratio_margins.h"

#include <Rcpp.h>
#include <cmath>
#include <vector>
#include <tulpa/likelihood.h>
#include <tulpa/portable_math.h>

namespace tulpaRatio {

namespace {

// ----- Layout-feature guard ---------------------------------------------
// B2 covers the same scope as B1b: betas only, optional extras, NO spatial /
// temporal / RE / ZI / OI / SVC / TVC / latent / ST. Bail out loudly if any
// of those flags are set so we never silently produce a wrong gradient.
inline void assert_no_latent_structure(const tulpa::ParamLayout& layout) {
    const bool any_latent =
        layout.has_re || layout.has_re_slopes ||
        layout.has_spatial || layout.is_gp || layout.is_multiscale_gp ||
        layout.is_hsgp || layout.is_icar_collapsed || layout.is_bym2_collapsed ||
        layout.has_temporal || layout.has_multiscale_temporal ||
        layout.is_temporal_gp || layout.has_zi || layout.has_oi ||
        layout.has_svc || layout.has_tvc || layout.has_latent ||
        layout.has_spatiotemporal;
    if (any_latent) {
        Rcpp::stop("tulpaRatio H-kernel: latent structure (spatial/temporal/RE/"
                   "ZI/SVC/TVC/ST) not supported in B2 — use A_r mode for these.");
    }
}

// ----- Beta Gaussian prior + ll accumulator -----------------------------
// Adds d/dbeta_kj of N(0, sigma_beta^2) prior into grad[] and accumulates
// the prior log-density into log_post (when log_post_out != nullptr).
inline void apply_beta_prior(
    const std::vector<double>& params,
    const tulpa::ModelData& data,
    const tulpa::ParamLayout& layout,
    std::vector<double>& grad,
    double* log_post_out
) {
    const double sigma_beta = data.sigma_beta;
    const double tau_beta = 1.0 / (sigma_beta * sigma_beta);
    const double log_norm = -0.5 * std::log(2.0 * M_PI * sigma_beta * sigma_beta);

    for (int k = 0; k < data.n_processes; ++k) {
        const int start = layout.process_beta_start[k];
        const int p_k   = layout.process_beta_count[k];
        for (int j = 0; j < p_k; ++j) {
            const double b = params[start + j];
            grad[start + j] += -tau_beta * b;
            if (log_post_out) {
                *log_post_out += log_norm - 0.5 * tau_beta * b * b;
            }
        }
    }
}

// ----- Compute eta_k[i] for every process k, every obs i ----------------
// Returns a row-major [n_processes x N] matrix. Includes per-process offset
// when present. Pure double (no AD needed at H-mode).
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

// ----- Scatter sum_i resid_k(i) * X_k[i,j]  into  grad_beta_k[] ---------
// Single source of truth for the X^T r reduction. Each family's Functor
// supplies resid_k(i) for k in [0, n_processes); the scaffold does the
// rest. extra_grad_out[e] receives sum_i (per-obs partial of extra_e).
template <class PerObsFn>
inline double run_obs_loop(
    const std::vector<double>& params,
    const tulpa::ModelData& data,
    const tulpa::ParamLayout& layout,
    const double* eta,                  // [np x N]
    PerObsFn fn,
    std::vector<double>& grad,
    int n_extra_grads                  // length of extra_grad_out
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

        // Scatter beta gradients
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

        // Accumulate extra-grad contributions
        for (int e = 0; e < n_extra_grads; ++e) extra_acc[e] += extra[e];
    }

    // Write extra-grad sums into the parameter vector
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
void grad_h(const std::vector<double>& params, const tulpa::ModelData& data,
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

} // namespace

Rcpp::NumericMatrix precision_spec_h_probe(double y, double mu, double phi) {
    const Gamma gamma(phi);
    const Observation x{y,std::log(mu)};
    const auto g = gamma.operator()<false>(x);
    Rcpp::NumericMatrix out(3,2);
    out(0,0) = (-1.0 + g.ll) + 1.0;
    out(0,1) = g.extra;
    out(1,0) = (NegBin(2).operator()<false>({0,0}).ll + g.ll) + 2*std::log1p(.5);
    out(1,1) = g.extra;
    out(2,0) = g.ll + g.ll;
    out(2,1) = g.extra;
    return out;
}

Rcpp::NumericVector precision_spec_beta_h_probe(int y, int n, double phi) {
    const auto b = BetaBinomial(phi).operator()<false>({double(y),0.0,n});
    return Rcpp::NumericVector::create(b.ll,b.extra);
}
void grad_h_beta_binomial(const std::vector<double>& params, const tulpa::ModelData& data,
 const tulpa::ParamLayout& layout, std::vector<double>& grad, double* lp) {
 grad_h<BetaBinomial>(params,data,layout,grad,lp);
}

void grad_h_binomial(const std::vector<double>& params, const tulpa::ModelData& data,
 const tulpa::ParamLayout& layout, std::vector<double>& grad, double* lp) {
 grad_h<Binomial>(params,data,layout,grad,lp);
}

void grad_h_negbin_gamma(const std::vector<double>& params, const tulpa::ModelData& data,
 const tulpa::ParamLayout& layout, std::vector<double>& grad, double* lp) {
 grad_h<NegBin,Gamma>(params,data,layout,grad,lp);
}

void grad_h_lognormal(const std::vector<double>& params, const tulpa::ModelData& data,
 const tulpa::ParamLayout& layout, std::vector<double>& grad, double* lp) {
 grad_h<Lognormal,Lognormal>(params,data,layout,grad,lp);
}

void grad_h_gamma_gamma(const std::vector<double>& params, const tulpa::ModelData& data,
 const tulpa::ParamLayout& layout, std::vector<double>& grad, double* lp) {
 grad_h<Gamma,Gamma>(params,data,layout,grad,lp);
}

void grad_h_poisson_gamma(const std::vector<double>& params, const tulpa::ModelData& data,
 const tulpa::ParamLayout& layout, std::vector<double>& grad, double* lp) {
 grad_h<Poisson,Gamma>(params,data,layout,grad,lp);
}

void grad_h_negbin_negbin(const std::vector<double>& params, const tulpa::ModelData& data,
 const tulpa::ParamLayout& layout, std::vector<double>& grad, double* lp) {
 grad_h<NegBin,NegBin>(params,data,layout,grad,lp);
}

} // namespace tulpaRatio
