#ifndef TULPARATIO_RATIO_MARGINS_H
#define TULPARATIO_RATIO_MARGINS_H

#include "ratio_math.h"
#include <tulpa/ad_scalar_math.h>
#include <cmath>
#include <type_traits>
#include <tulpa/count_math.h>
#include <tulpa/precision_math.h>
#include <tulpa/portable_math.h>
#include "beta_binomial_obs.h"

namespace ratiod { namespace math {

template<class T>
inline T poisson_eta_density(double y, const T& eta, const T& mu, const T& constant) {
    return y * eta - mu - constant;
}

template<class T>
inline T binomial_eta_density(double y, int n, const T& log_p,
                               const T& log_1mp, double constant) {
    return y * log_p + (n-y) * log_1mp + constant;
}

template<class T>
inline T lognormal_eta_density(double log_y, const T& eta, const T& sigma,
                                const T& log_sigma, const T& initial = T(0.0)) {
    const T z = (log_y - eta) / sigma;
    return initial + -log_y - log_sigma - T(0.5)*z*z;
}
struct NegbinMath {
    template <typename T> static double value(const T& x) { return get_value(x); }
    template <typename T> static T log(const T& x) { return safe_log(x); }
    template <typename T> static T log1p(const T& x) { return log1p_fn(x); }
    template <typename T> static T log1pmx(const T& x) { using tulpa::math::log1pmx; return log1pmx(x); }
    template <typename T> static T lgamma(const T& x) { return lgamma_fn(x); }
};

template<typename T>
inline T negbin_log_p0(const T& mu, const T& phi) {
    return tulpa::math::negbin_log_p0<T, NegbinMath>(mu, phi);
}

template<typename T>
inline T log_lik_poisson(int y, const T& mu) {
    // y * log(mu) - mu - lgamma(y+1)
    // Note: lgamma(y+1) is constant w.r.t. parameters, but we include it for correctness
    return poisson_eta_density(double(y), safe_log(mu), mu, lgamma_fn(T(y + 1.0)));
}

template<typename T>
inline T log_lik_gamma(double y, const T& shape, const T& mu) {
    return tulpa::math::gamma_log_density(y, mu, shape, std::log(y));
}

template<typename T>
inline T log_lik_negbin(int y, const T& mu, const T& phi) {
    return tulpa::math::negbin_log_pmf<T, NegbinMath>(y, mu, phi);
}

template<typename T>
inline T log_lik_binomial(int y, int n, const T& q) {
    return binomial_eta_density(double(y), n, log_inv_logit(q), log1m_inv_logit(q), portable_lchoose(n,y));
}

template<typename T>
inline T log_lik_zi_binomial(int y, int n, const T& q, const T& logit_zi) {
    if (y == 0) {
        return log_add_exp(log_inv_logit(logit_zi),
                           log1m_inv_logit(logit_zi) + n * log1m_inv_logit(q));
    }
    return log1m_inv_logit(logit_zi) + log_lik_binomial(y, n, q);
}

template<typename T>
inline T log_lik_oi_binomial(int y, int n, const T& q, const T& logit_oi) {
    if (y == n) {
        return log_add_exp(log_inv_logit(logit_oi),
                           log1m_inv_logit(logit_oi) + n * log_inv_logit(q));
    }
    return log1m_inv_logit(logit_oi) + log_lik_binomial(y, n, q);
}

template<typename T>
inline T log_lik_zoib(int y, int n, const T& q, const T& logit_zi, const T& logit_oi) {
    const T log_1m_zi = log1m_inv_logit(logit_zi);
    const T log_1m_oi = log1m_inv_logit(logit_oi);
    if (y == 0) {
        return log_add_exp(log_inv_logit(logit_zi),
                           log_1m_zi + log_1m_oi + n * log1m_inv_logit(q));
    }
    if (y == n) {
        return log_1m_zi + log_add_exp(log_inv_logit(logit_oi),
                                       log_1m_oi + n * log_inv_logit(q));
    }
    return log_1m_zi + log_1m_oi + log_lik_binomial(y, n, q);
}

template<typename T>
inline T log_lik_hurdle_binomial(int y, int n, const T& q, const T& logit_theta) {
    if (y == 0) return log1m_inv_logit(logit_theta);
    // theta P(Y=y|Y>0), P(Y=y|Y>0) = Binom(y|n,p) / (1 - (1-p)^n)
    return log_inv_logit(logit_theta) + log_lik_binomial(y, n, q)
        - log1m_exp(n * log1m_inv_logit(q));
}

template<typename T>
inline T log_lik_gamma_gamma(double y_num, double y_denom,
                             const T& mu_num, const T& mu_denom,
                             const T& shape_num, const T& shape_denom) {
    return log_lik_gamma(y_num, shape_num, mu_num)
         + log_lik_gamma(y_denom, shape_denom, mu_denom);
}

template<typename T>
inline T log_lik_lognormal(double y, const T& mu, const T& sigma) {
    return lognormal_eta_density(std::log(y), mu, sigma, safe_log(sigma));
}

template<typename T>
inline T log_lik_lognormal_lognormal(double y_num, double y_denom,
                                     const T& mu_num, const T& mu_denom,
                                     const T& sigma_num, const T& sigma_denom) {
    return log_lik_lognormal(y_num, mu_num, sigma_num) +
           log_lik_lognormal(y_denom, mu_denom, sigma_denom);
}

template<typename T>
inline T log_lik_beta_binomial(int y, int n, const T& q, const T& phi) {
    return tulpa::math::beta_binomial_log_pmf<T, NegbinMath>(y, n,
        safe_exp(log_inv_logit(q)), safe_exp(log1m_inv_logit(q)), phi, portable_lchoose(n,y));
}
} }

namespace tulpaRatio { namespace margins {

enum class Prior { None, Gamma, HalfCauchy };
enum Field : unsigned { Value = 1, Score = 2, Extra = 4, Curvature = 8,
                        FirstOrder = Value | Score | Extra, SecondOrder = Value | Score | Curvature };

struct Observation {
    double y, eta;
    int trials = 0;
    double log_y = 0.0, log_factorial = 0.0;
    double digamma_difference = 0.0, lgamma_difference = 0.0;
};

struct Result {
    double ll = 0.0, residual = 0.0, extra = 0.0, neg_hess = 0.0;
};

struct NoMargin {
    static constexpr int n_extra = 0;
    static constexpr Prior prior = Prior::None;
    static constexpr bool log_extra = false;
    explicit NoMargin(double = 1.0, bool = true) {}
    template<bool Cached, unsigned Fields = FirstOrder> Result operator()(const Observation&, bool = true) const { return {}; }
};

struct Poisson : NoMargin {
    using NoMargin::NoMargin;
    template<bool Cached, unsigned Fields = FirstOrder> Result operator()(const Observation& x, bool lp = true) const {
        const double mu = std::exp(x.eta);
        return {(Fields & Value) && lp ? ratiod::math::poisson_eta_density(x.y, x.eta, mu, Cached ? x.log_factorial : std::lgamma(x.y + 1.0)) : 0.0,
                (Fields & Score) ? x.y - mu : 0.0, 0.0, (Fields & Curvature) ? mu : 0.0};
    }
};

struct NegBin {
    static constexpr int n_extra = 1;
    static constexpr Prior prior = Prior::Gamma;
    static constexpr bool log_extra = false;
    double phi;
    explicit NegBin(double p, bool = true) : phi(p) {}
    template<bool Cached, unsigned Fields = FirstOrder> Result operator()(const Observation& x, bool lp = true) const {
        const double mu = std::exp(x.eta);
        Result out;
        const double denom = mu + phi;
        if constexpr (Cached) {
            if constexpr (Fields & Score) out.residual = x.y - mu * (x.y + phi) * (1.0 / denom);
            if constexpr (Fields & Extra) out.extra = tulpa::math::negbin_size_score(int(x.y), mu, phi, x.digamma_difference);
            if ((Fields & Value) && lp) out.ll = tulpa::math::negbin_log_pmf(int(x.y), mu, phi, x.lgamma_difference, x.log_factorial);
        } else {
            if constexpr (Fields & Score) out.residual = x.y - mu * (x.y + phi) / denom;
            if constexpr (Fields & Extra) out.extra = tulpa::math::negbin_size_score(int(x.y), mu, phi);
            if ((Fields & Value) && lp) out.ll = tulpa::math::negbin_log_pmf(int(x.y), mu, phi);
        }
        if constexpr (Fields & Curvature) out.neg_hess = (mu / denom) * (phi / denom) * (x.y + phi);
        return out;
    }
};

struct Gamma {
    static constexpr int n_extra = 1;
    static constexpr Prior prior = Prior::Gamma;
    static constexpr bool log_extra = false;
    double phi;
    tulpa::math::GammaPrecision precision;
    explicit Gamma(double p, bool lp = true) : phi(p), precision(p, lp) {}
    template<bool Cached, unsigned Fields = FirstOrder> Result operator()(const Observation& x, bool lp = true) const {
        if (x.y <= 0.0) return {Cached && lp ? -1e10 : 0.0, 0.0, 0.0};
        const double mu = std::exp(x.eta);
        const double ratio = Cached ? x.y * (1.0 / mu) : x.y / mu;
        const double log_y = Cached ? x.log_y : std::log(x.y);
        return {(Fields & Value) && lp ? precision.log_density(x.y, mu, log_y) : 0.0,
                (Fields & Score) ? phi * (ratio - 1.0) : 0.0,
                (Fields & Extra) ? precision.derivative(x.y, mu, log_y) : 0.0,
                (Fields & Curvature) ? phi * ratio : 0.0};
    }
};

struct Lognormal {
    static constexpr int n_extra = 1;
    static constexpr Prior prior = Prior::HalfCauchy;
    static constexpr bool log_extra = true;
    double phi, variance, log_sigma;
    explicit Lognormal(double p, bool lp = true) : phi(p), variance(p*p), log_sigma(lp ? std::log(p) : 0.0) {}
    template<bool Cached, unsigned Fields = FirstOrder> Result operator()(const Observation& x, bool lp = true) const {
        const double log_y = Cached ? x.log_y : std::log(x.y);
        const double z = (log_y - x.eta) / phi;
        return {(Fields & Value) && lp ? ratiod::math::lognormal_eta_density(log_y, x.eta, phi, log_sigma) : 0.0,
                (Fields & Score) ? (log_y - x.eta) / variance : 0.0,
                (Fields & Extra) ? -1.0 + z*z : 0.0, (Fields & Curvature) ? 1.0 / variance : 0.0};
    }
    double add_density(double ll, const Observation& x) const {
        const double log_y = std::log(x.y);
        const double z = (log_y - x.eta) / phi;
        return ratiod::math::lognormal_eta_density(log_y, x.eta, phi, log_sigma, ll);
    }
};

struct Binomial : NoMargin {
    using NoMargin::NoMargin;
    template<bool Cached, unsigned Fields = FirstOrder> Result operator()(const Observation& x, bool lp = true) const {
        double p, log_p, log_1mp;
        if (x.eta >= 0.0) {
            const double e = std::exp(-x.eta);
            p = 1.0 / (1.0 + e);
            log_p = Cached ? -std::log(1.0 + e) : -std::log1p(e);
            log_1mp = -x.eta + log_p;
        } else {
            const double e = std::exp(x.eta);
            p = Cached ? e / (1.0 + e) : 1.0 / (1.0 + std::exp(-x.eta));
            log_1mp = Cached ? -std::log(1.0 + e) : -std::log1p(e);
            log_p = x.eta + log_1mp;
        }
        const double constant = Cached ? x.log_factorial : tulpa::math::portable_lchoose(x.trials, int(x.y));
        const double q = x.eta >= 0.0 ? std::exp(log_1mp) : 1.0 - p;
        return {(Fields & Value) && lp ? ratiod::math::binomial_eta_density(x.y, x.trials, log_p, log_1mp, constant) : 0.0,
                (Fields & Score) ? x.y - x.trials*p : 0.0, 0.0,
                (Fields & Curvature) ? x.trials*p*q : 0.0};
    }
};

struct BetaBinomial {
    static constexpr int n_extra = 1;
    static constexpr Prior prior = Prior::Gamma;
    static constexpr bool log_extra = false;
    double phi;
    explicit BetaBinomial(double p, bool = true) : phi(p) {}
    template<bool Cached, unsigned Fields = FirstOrder> Result operator()(const Observation& x, bool lp = true) const {
        const auto bb = ratiod::math::beta_binomial_obs(int(x.y), x.trials, x.eta, phi);
        const double constant = Cached ? x.log_factorial : tulpa::math::portable_lchoose(x.trials, int(x.y));
        Result out{(Fields & Value) && lp ? bb.ll + constant : 0.0,
                   (Fields & Score) ? bb.d_eta : 0.0, (Fields & Extra) ? bb.d_phi : 0.0};
        if constexpr (Fields & Curvature) {
            const double p = 1.0 / (1.0 + std::exp(-x.eta));
            const double q = 1.0 / (1.0 + std::exp(x.eta));
            const double d = phi*p*q;
            out.neg_hess = -(q-p)*bb.d_eta - d*d*(
                tulpa::math::polygamma_rising_difference(1,int(x.y),p*phi) +
                tulpa::math::polygamma_rising_difference(1,x.trials-int(x.y),q*phi));
        }
        return out;
    }
};

struct Response {
    const int* counts = nullptr;
    const double* continuous = nullptr;
    const int* trials = nullptr;
    Observation at(int i, double eta) const {
        return {counts ? double(counts[i]) : (continuous ? continuous[i] : 0.0),
                eta, trials ? trials[i] : 0};
    }
};

template<class Num, class Den = NoMargin> struct Ratio {
    static constexpr int n_extra = Num::n_extra + Den::n_extra;
    Num num;
    Den den;
    Response num_response, den_response;
    int N;
    double operator()(int i, const double* eta, double* residual, double* extra) const {
        const auto x = num_response.at(i, eta[i]);
        const auto y = den_response.at(i, Den::n_extra || den_response.counts || den_response.continuous ? eta[N+i] : 0.0);
        const auto a = num.template operator()<false>(x);
        const auto b = den.template operator()<false>(y);
        residual[0] = a.residual;
        if constexpr (!std::is_same<Den,NoMargin>::value) residual[1] = b.residual;
        if constexpr (Num::n_extra) extra[0] += a.extra;
        if constexpr (Den::n_extra) extra[Num::n_extra] += b.extra;
        if constexpr (std::is_same<Den,Lognormal>::value) return den.add_density(a.ll, y);
        return a.ll + b.ll;
    }
};

} }
#endif
