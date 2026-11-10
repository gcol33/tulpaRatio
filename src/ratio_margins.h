#ifndef TULPARATIO_RATIO_MARGINS_H
#define TULPARATIO_RATIO_MARGINS_H

#include "ratio_math.h"
#include "enum_names.h"
#include "taylor_jet.h"
#include <tulpa/types.h>
#include <tulpa/ad_scalar_math.h>
#include <cmath>
#include <type_traits>
#include <tulpa/count_math.h>
#include <tulpa/precision_math.h>
#include <tulpa/portable_math.h>

namespace tulpaRatio { namespace margins {
struct Poisson; struct NegBin; struct Gamma; struct Lognormal;
struct Binomial; struct BetaBinomial;
template<class Margin, class T>
T evaluate(double y, int n, const T& eta, const T& phi, const T* mean = nullptr);
} }

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
    return tulpaRatio::margins::evaluate<tulpaRatio::margins::Poisson>(double(y),0,safe_log(mu),T(1.0),&mu);
}

template<typename T>
inline T log_lik_gamma(double y, const T& shape, const T& mu) {
    return tulpaRatio::margins::evaluate<tulpaRatio::margins::Gamma>(y,0,T(0.0),shape,&mu);
}

template<typename T>
inline T log_lik_negbin(int y, const T& mu, const T& phi) {
    return tulpaRatio::margins::evaluate<tulpaRatio::margins::NegBin>(double(y),0,T(0.0),phi,&mu);
}

template<typename T>
inline T log_lik_binomial(int y, int n, const T& q) {
    return tulpaRatio::margins::evaluate<tulpaRatio::margins::Binomial>(double(y),n,q,T(1.0));
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
    return tulpaRatio::margins::evaluate<tulpaRatio::margins::Lognormal>(y,0,mu,sigma);
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
    return tulpaRatio::margins::evaluate<tulpaRatio::margins::BetaBinomial>(double(y),n,q,phi);
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
    const double* mean = nullptr;
};

struct Result {
    double ll = 0.0, residual = 0.0, extra = 0.0, neg_hess = 0.0;
    double zi = 0.0, oi = 0.0;
};

struct NoMargin {
    static constexpr int n_extra = 0;
    static constexpr Prior prior = Prior::None;
    static constexpr bool log_extra = false;
    explicit NoMargin(double = 1.0, bool = true) {}
    template<bool Cached, unsigned Fields = FirstOrder> Result evaluate(const Observation&, bool = true) const { return {}; }
};

struct Poisson : NoMargin {
    using NoMargin::NoMargin;
    template<bool Cached, unsigned Fields = FirstOrder> Result evaluate(const Observation& x, bool lp = true) const {
        const double mu = x.mean ? *x.mean : std::exp(x.eta);
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
    template<bool Cached, unsigned Fields = FirstOrder> Result evaluate(const Observation& x, bool lp = true) const {
        const double mu = x.mean ? *x.mean : std::exp(x.eta);
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
    template<bool Cached, unsigned Fields = FirstOrder> Result evaluate(const Observation& x, bool lp = true) const {
        if (x.y <= 0.0) return {Cached && lp ? -1e10 : 0.0, 0.0, 0.0};
        const double mu = x.mean ? *x.mean : std::exp(x.eta);
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
    template<bool Cached, unsigned Fields = FirstOrder> Result evaluate(const Observation& x, bool lp = true) const {
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
    template<bool Cached, unsigned Fields = FirstOrder> Result evaluate(const Observation& x, bool lp = true) const {
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
        const double constant = (Fields & Value) && lp ?
            (Cached ? x.log_factorial : tulpa::math::portable_lchoose(x.trials, int(x.y))) : 0.0;
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
    template<bool Cached, unsigned Fields = FirstOrder> Result evaluate(const Observation& x, bool lp = true) const {
        const double p = 1.0 / (1.0 + std::exp(-x.eta));
        const double q = 1.0 / (1.0 + std::exp(x.eta));
        Result out;
        if ((Fields & Value) && lp) {
            const double constant = Cached ? x.log_factorial : tulpa::math::portable_lchoose(x.trials,int(x.y));
            out.ll = tulpa::math::beta_binomial_log_pmf(int(x.y),x.trials,p,q,phi,0.0) + constant;
        }
        double score = 0.0;
        if constexpr (Fields & (Score | Curvature)) {
            score = phi*p*q*(tulpa::math::polygamma_rising_difference(0,int(x.y),p*phi)
                - tulpa::math::polygamma_rising_difference(0,x.trials-int(x.y),q*phi));
            if constexpr (Fields & Score) out.residual = score;
        }
        if constexpr (Fields & Extra)
            out.extra = tulpa::math::beta_binomial_phi_derivative(int(x.y),x.trials,p,q,phi);
        if constexpr (Fields & Curvature) {
            const double d = phi*p*q;
            out.neg_hess = -(q-p)*score - d*d*(
                tulpa::math::polygamma_rising_difference(1,int(x.y),p*phi) +
                tulpa::math::polygamma_rising_difference(1,x.trials-int(x.y),q*phi));
        }
        return out;
    }
};

template<class Margin, class T>
T evaluate(double y, int n, const T& eta, const T& phi, const T* mean) {
    using namespace ratiod::math;
    if constexpr (std::is_same<Margin,Poisson>::value) {
        const T mu=mean ? *mean : safe_exp(eta);
        return poisson_eta_density(y,eta,mu,lgamma_fn(T(y+1.0)));
    } else if constexpr (std::is_same<Margin,NegBin>::value) {
        const T mu=mean ? *mean : safe_exp(eta);
        return tulpa::math::negbin_log_pmf<T,NegbinMath>(int(y),mu,phi);
    } else if constexpr (std::is_same<Margin,Gamma>::value) {
        const T mu=mean ? *mean : safe_exp(eta);
        return tulpa::math::gamma_log_density(y,mu,phi,std::log(y));
    } else if constexpr (std::is_same<Margin,Lognormal>::value) {
        return lognormal_eta_density(std::log(y),eta,phi,safe_log(phi));
    } else if constexpr (std::is_same<Margin,Binomial>::value) {
        return binomial_eta_density(y,n,log_inv_logit(eta),log1m_inv_logit(eta),portable_lchoose(n,int(y)));
    } else {
        return tulpa::math::beta_binomial_log_pmf<T,NegbinMath>(int(y),n,
            safe_exp(log_inv_logit(eta)),safe_exp(log1m_inv_logit(eta)),phi,portable_lchoose(n,int(y)));
    }
}

struct CurvatureDerivatives {
    double w=0.0, eta=0.0, phi=0.0, score_phi=0.0, score=0.0;
};

template<class Margin>
CurvatureDerivatives curvature_derivatives(const Observation& x, double phi) {
    const auto a=Margin(phi,false).template evaluate<false,Score|Curvature>(x,false);
    CurvatureDerivatives out{a.neg_hess,0.,0.,0.,a.residual};
    if constexpr (std::is_same<Margin,Poisson>::value) out.eta=out.w;
    else if constexpr (std::is_same<Margin,Gamma>::value) {
        const double ratio=x.y/std::exp(x.eta);
        out.eta=-out.w; out.phi=ratio; out.score_phi=ratio-1.;
    } else if constexpr (std::is_same<Margin,NegBin>::value) {
        const double mu=std::exp(x.eta), A=mu+phi;
        out.eta=out.w*(phi-mu)/A;
        out.phi=mu*(mu*(x.y+2*phi)-x.y*phi)/(A*A*A);
        out.score_phi=mu*(x.y-mu)/(A*A);
    } else if constexpr (std::is_same<Margin,Binomial>::value) {
        const double p=1./(1.+std::exp(-x.eta));
        out.eta=out.w*(1.-2.*p);
    } else if constexpr (std::is_same<Margin,Lognormal>::value) {
        out.phi=-2./(phi*phi*phi);
        out.score_phi=-2.*(std::log(x.y)-x.eta)/(phi*phi*phi);
    } else if constexpr (std::is_same<Margin,BetaBinomial>::value) {
        const double p=1./(1.+std::exp(-x.eta)),q=1./(1.+std::exp(x.eta)),d=phi*p*q;
        const double A=tulpa::math::polygamma_rising_difference(0,int(x.y),p*phi);
        const double B=tulpa::math::polygamma_rising_difference(0,x.trials-int(x.y),q*phi);
        const double C=tulpa::math::polygamma_rising_difference(1,int(x.y),p*phi);
        const double D=tulpa::math::polygamma_rising_difference(1,x.trials-int(x.y),q*phi);
        const double E=tulpa::math::polygamma_rising_difference(2,int(x.y),p*phi);
        const double F=tulpa::math::polygamma_rising_difference(2,x.trials-int(x.y),q*phi);
        out.score_phi=p*q*(A-B)+d*(p*C-q*D);
        out.eta=2.*p*q*out.score+(q-p)*out.w-2.*d*d*(q-p)*(C+D)-d*d*d*(E-F);
        out.phi=-(q-p)*out.score_phi-2.*phi*p*p*q*q*(C+D)-d*d*(p*E+q*F);
    }
    return out;
}

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
        const auto a = num.template evaluate<false>(x);
        const auto b = den.template evaluate<false>(y);
        residual[0] = a.residual;
        if constexpr (!std::is_same<Den,NoMargin>::value) residual[1] = b.residual;
        if constexpr (Num::n_extra) extra[0] += a.extra;
        if constexpr (Den::n_extra) extra[Num::n_extra] += b.extra;
        if constexpr (std::is_same<Den,Lognormal>::value) return den.add_density(a.ll, y);
        return a.ll + b.ll;
    }
};

} }

namespace ratiod_zi {

using ratiod::math::get_value;
using ratiod::math::log1m_exp;
using ratiod::math::log_add_exp;
using ratiod::math::log_inv_logit;
using ratiod::math::log1m_inv_logit;
using ratiod::math::log_lik_poisson;
using ratiod::math::log_lik_negbin;
using ratiod::math::negbin_log_p0;


enum class ZIType {
  NONE,
  ZI_POISSON,
  ZI_NEGBIN,
  HURDLE_POISSON,
  HURDLE_NEGBIN,
  ZI_BINOMIAL,
  HURDLE_BINOMIAL,
  OI_BINOMIAL,
  ZOIB
};


inline constexpr ratiod_enum::EnumName<ZIType> ZI_TYPE_NAMES[] = {
  {"none", ZIType::NONE},
  {"zi_poisson", ZIType::ZI_POISSON},
  {"zi_negbin", ZIType::ZI_NEGBIN},
  {"hurdle_poisson", ZIType::HURDLE_POISSON},
  {"hurdle_negbin", ZIType::HURDLE_NEGBIN},
  {"zi_binomial", ZIType::ZI_BINOMIAL},
  {"hurdle_binomial", ZIType::HURDLE_BINOMIAL},
  {"oi_binomial", ZIType::OI_BINOMIAL},
  {"zoib", ZIType::ZOIB}};

inline ZIType parse_zi_type(const std::string& s) {
  return ratiod_enum::parse_enum(ZI_TYPE_NAMES, s, "zero/one inflation type");
}







template<typename T>
inline T zi_poisson_lpmf_logit(int y, const T& mu, const T& logit_zi) {
  if (y == 0) {

    return log_add_exp(log_inv_logit(logit_zi), log1m_inv_logit(logit_zi) - mu);
  }
  return log1m_inv_logit(logit_zi) + log_lik_poisson(y, mu);
}


template<typename T>
inline T zi_negbin_lpmf_logit(int y, const T& mu, const T& phi, const T& logit_zi) {
  if (y == 0) {

    return log_add_exp(log_inv_logit(logit_zi),
                       log1m_inv_logit(logit_zi) + negbin_log_p0(mu, phi));
  }
  return log1m_inv_logit(logit_zi) + log_lik_negbin(y, mu, phi);
}







template<typename T>
inline T truncated_poisson_lpmf(int y, const T& mu) {
  if (y <= 0) return T(-std::numeric_limits<double>::infinity());
  return log_lik_poisson(y, mu) - log1m_exp(-mu);
}


template<typename T>
inline T truncated_negbin_lpmf(int y, const T& mu, const T& phi) {
  if (y <= 0) return T(-std::numeric_limits<double>::infinity());
  return log_lik_negbin(y, mu, phi) - log1m_exp(negbin_log_p0(mu, phi));
}



template<typename T>
inline T hurdle_poisson_lpmf_logit(int y, const T& mu, const T& logit_theta) {

  if (y == 0) return log1m_inv_logit(logit_theta);
  return log_inv_logit(logit_theta) + truncated_poisson_lpmf(y, mu);
}


template<typename T>
inline T hurdle_negbin_lpmf_logit(int y, const T& mu, const T& phi, const T& logit_theta) {
  if (y == 0) return log1m_inv_logit(logit_theta);
  return log_inv_logit(logit_theta) + truncated_negbin_lpmf(y, mu, phi);
}








template<typename T>
inline T zi_log_likelihood(
    int y, const T& mu, const T& phi, const T& logit_zi_or_theta,
    ZIType zi_type) {

  switch (zi_type) {
    case ZIType::ZI_POISSON:
      return zi_poisson_lpmf_logit(y, mu, logit_zi_or_theta);

    case ZIType::ZI_NEGBIN:
      return zi_negbin_lpmf_logit(y, mu, phi, logit_zi_or_theta);

    case ZIType::HURDLE_POISSON:
      return hurdle_poisson_lpmf_logit(y, mu, logit_zi_or_theta);

    case ZIType::HURDLE_NEGBIN:
      return hurdle_negbin_lpmf_logit(y, mu, phi, logit_zi_or_theta);

    case ZIType::NONE:
    default:

      if (get_value(phi) > 0) {
        return log_lik_negbin(y, mu, phi);
      } else {
        return log_lik_poisson(y, mu);
      }
  }
}

}

namespace tulpaRatio { namespace margins {
template<class Margin, class T>
T linked_density(double y, int n, const T& predictor, const T& phi,
                    tulpa::ZIType zi, const T& zi_predictor, const T& oi_predictor,
                    ratiod_link::LinkType link = ratiod_link::LinkType::LOG,
                    ratiod_link::LinkType zi_link = ratiod_link::LinkType::LOGIT,
                    ratiod_link::LinkType oi_link = ratiod_link::LinkType::LOGIT) {
    using namespace margins;
    using namespace ratiod::math;
    using std::exp;
    const T logit_zi = prob_logit(zi_predictor, zi_link);
    const T logit_oi = prob_logit(oi_predictor, oi_link);
    T eta = predictor;
    if constexpr (std::is_same<Margin,Binomial>::value || std::is_same<Margin,BetaBinomial>::value)
        eta = prob_logit(predictor, link);
    else if constexpr (!std::is_same<Margin,Lognormal>::value) {
        if (link != ratiod_link::LinkType::LOG) {
            const T mean = link_mean(predictor, link);
            if (!(get_value(mean) > 0)) return T(-std::numeric_limits<double>::infinity());
            eta = safe_log(mean);
        }
    }
    if constexpr (std::is_same<Margin,Binomial>::value) {
        switch (zi) {
            case tulpa::ZIType::ZI_BINOMIAL: return log_lik_zi_binomial(int(y),n,eta,logit_zi);
            case tulpa::ZIType::HURDLE_BINOMIAL: return log_lik_hurdle_binomial(int(y),n,eta,logit_zi);
            case tulpa::ZIType::OI_BINOMIAL: return log_lik_oi_binomial(int(y),n,eta,logit_oi);
            case tulpa::ZIType::ZOIB: return log_lik_zoib(int(y),n,eta,logit_zi,logit_oi);
            default: return log_lik_binomial(int(y),n,eta);
        }
    } else if constexpr (std::is_same<Margin,BetaBinomial>::value) {
        return log_lik_beta_binomial(int(y),n,eta,phi);
    } else if constexpr (std::is_same<Margin,Lognormal>::value) {
        return log_lik_lognormal(y,eta,phi);
    } else {
        const T mu = safe_exp(eta);
        if constexpr (std::is_same<Margin,Poisson>::value) {
            if (zi == tulpa::ZIType::ZI_POISSON) return ratiod_zi::zi_poisson_lpmf_logit(int(y),mu,logit_zi);
            if (zi == tulpa::ZIType::HURDLE_POISSON) return ratiod_zi::hurdle_poisson_lpmf_logit(int(y),mu,logit_zi);
            return margins::evaluate<Poisson>(y,0,eta,phi,&mu);
        } else if constexpr (std::is_same<Margin,NegBin>::value) {
            if (zi == tulpa::ZIType::ZI_NEGBIN) return ratiod_zi::zi_negbin_lpmf_logit(int(y),mu,phi,logit_zi);
            if (zi == tulpa::ZIType::HURDLE_NEGBIN) return ratiod_zi::hurdle_negbin_lpmf_logit(int(y),mu,phi,logit_zi);
            return log_lik_negbin(int(y),mu,phi);
        } else return log_lik_gamma(y,phi,mu);
    }
}

template<class Margin>
Result linked_result(double y, int n, double eta, double phi,
    tulpa::ZIType inflation, double zi, double oi, ratiod_link::LinkType link,
    ratiod_link::LinkType zi_link = ratiod_link::LinkType::LOGIT,
    ratiod_link::LinkType oi_link = ratiod_link::LinkType::LOGIT) {
    using J = ratiod::jet::Jet3<double>;
    const J jet = linked_density<Margin>(y, n, J::line(eta, 1), J(phi),
        inflation, J(zi), J(oi), link, zi_link, oi_link);
    Result out;
    out.ll = jet.v;
    out.residual = jet.d1;
    out.neg_hess = -jet.d2;
    using D = fwd::Dual;
    const auto derivative = [&](int seed) {
        return linked_density<Margin>(y, n, D(eta), D(phi, seed == 0),
            inflation, D(zi, seed == 1), D(oi, seed == 2), link, zi_link, oi_link).grad;
    };
    if constexpr (Margin::n_extra) out.extra = derivative(0) * (Margin::log_extra ? phi : 1.0);
    out.zi = derivative(1);
    out.oi = derivative(2);
    return out;
}

} }
#endif
