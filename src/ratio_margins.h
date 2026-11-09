#ifndef TULPARATIO_RATIO_MARGINS_H
#define TULPARATIO_RATIO_MARGINS_H

#include <tulpa/ad_scalar_math.h>
#include <cmath>
#include <type_traits>
#include <tulpa/count_math.h>
#include <tulpa/precision_math.h>
#include <tulpa/portable_math.h>
#include "beta_binomial_obs.h"

namespace tulpaRatio { namespace margins {

enum class Prior { None, Gamma, HalfCauchy };

struct Observation {
    double y, eta;
    int trials = 0;
    double log_y = 0.0, log_factorial = 0.0;
    double digamma_difference = 0.0, lgamma_difference = 0.0;
};

struct Result {
    double ll = 0.0, residual = 0.0, extra = 0.0;
};

struct NoMargin {
    static constexpr int n_extra = 0;
    static constexpr Prior prior = Prior::None;
    static constexpr bool log_extra = false;
    explicit NoMargin(double = 1.0, bool = true) {}
    template<bool Cached> Result operator()(const Observation&, bool = true) const { return {}; }
};

struct Poisson : NoMargin {
    using NoMargin::NoMargin;
    template<bool Cached> Result operator()(const Observation& x, bool lp = true) const {
        const double mu = std::exp(x.eta);
        return {lp ? x.y * x.eta - mu - (Cached ? x.log_factorial : std::lgamma(x.y + 1.0)) : 0.0,
                x.y - mu, 0.0};
    }
};

struct NegBin {
    static constexpr int n_extra = 1;
    static constexpr Prior prior = Prior::Gamma;
    static constexpr bool log_extra = false;
    double phi;
    explicit NegBin(double p, bool = true) : phi(p) {}
    template<bool Cached> Result operator()(const Observation& x, bool lp = true) const {
        const double mu = std::exp(x.eta);
        Result out;
        const double denom = mu + phi;
        if constexpr (Cached) {
            out.residual = x.y - mu * (x.y + phi) * (1.0 / denom);
            out.extra = tulpa::math::negbin_size_score(int(x.y), mu, phi, x.digamma_difference);
            if (lp) out.ll = tulpa::math::negbin_log_pmf(int(x.y), mu, phi, x.lgamma_difference, x.log_factorial);
        } else {
            out.residual = x.y - mu * (x.y + phi) / denom;
            out.extra = tulpa::math::negbin_size_score(int(x.y), mu, phi);
            if (lp) out.ll = tulpa::math::negbin_log_pmf(int(x.y), mu, phi);
        }
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
    template<bool Cached> Result operator()(const Observation& x, bool lp = true) const {
        if (x.y <= 0.0) return {Cached && lp ? -1e10 : 0.0, 0.0, 0.0};
        const double mu = std::exp(x.eta);
        const double ratio = Cached ? x.y * (1.0 / mu) : x.y / mu;
        const double log_y = Cached ? x.log_y : std::log(x.y);
        return {lp ? precision.log_density(x.y, mu, log_y) : 0.0,
                phi * (ratio - 1.0), precision.derivative(x.y, mu, log_y)};
    }
};

struct Lognormal {
    static constexpr int n_extra = 1;
    static constexpr Prior prior = Prior::HalfCauchy;
    static constexpr bool log_extra = true;
    double phi, variance, log_sigma;
    explicit Lognormal(double p, bool lp = true) : phi(p), variance(p*p), log_sigma(lp ? std::log(p) : 0.0) {}
    template<bool Cached> Result operator()(const Observation& x, bool lp = true) const {
        const double log_y = Cached ? x.log_y : std::log(x.y);
        const double z = (log_y - x.eta) / phi;
        return {lp ? -log_y - log_sigma - 0.5*z*z : 0.0,
                (log_y - x.eta) / variance, -1.0 + z*z};
    }
    double add_density(double ll, const Observation& x) const {
        const double log_y = std::log(x.y);
        const double z = (log_y - x.eta) / phi;
        return ll + -log_y - log_sigma - 0.5*z*z;
    }
};

struct Binomial : NoMargin {
    using NoMargin::NoMargin;
    template<bool Cached> Result operator()(const Observation& x, bool lp = true) const {
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
        return {lp ? x.y*log_p + (x.trials-x.y)*log_1mp + constant : 0.0,
                x.y - x.trials*p, 0.0};
    }
};

struct BetaBinomial {
    static constexpr int n_extra = 1;
    static constexpr Prior prior = Prior::Gamma;
    static constexpr bool log_extra = false;
    double phi;
    explicit BetaBinomial(double p, bool = true) : phi(p) {}
    template<bool Cached> Result operator()(const Observation& x, bool lp = true) const {
        const auto bb = ratiod::math::beta_binomial_obs(int(x.y), x.trials, x.eta, phi);
        const double constant = Cached ? x.log_factorial : tulpa::math::portable_lchoose(x.trials, int(x.y));
        return {lp ? bb.ll + constant : 0.0, bb.d_eta, bb.d_phi};
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
