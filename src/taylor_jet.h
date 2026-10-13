// taylor_jet.h
// Univariate Taylor jets of order three.
//
// A Jet3 carries a function f(t) along a line through the evaluation point as
// its value and first three derivatives at t = 0. Every operation is the
// product or chain rule up to third order (Faa di Bruno), so a log-density
// written once over a scalar type T and evaluated on jets returns its first
// three directional derivatives exactly, from the same code the sampler
// evaluates. The coefficient type S is double, or fwd::Dual when those
// derivatives are themselves differentiated with respect to one more
// parameter (a dispersion, say).

#ifndef RATIOD_TAYLOR_JET_H
#define RATIOD_TAYLOR_JET_H

#include <cmath>
#include <limits>
#include <Rcpp.h>

#include "autodiff_utils.h"
#include "portable_math.h"

namespace ratiod {
namespace jet {

// ---------------------------------------------------------------------------
// The coefficient type's elementary functions
// ---------------------------------------------------------------------------

inline double coef_value(double x) { return x; }
inline double coef_value(const fwd::Dual& x) { return x.val; }

inline bool coef_is_zero(double x) { return x == 0.0; }
inline bool coef_is_zero(const fwd::Dual& x) { return x.val == 0.0 && x.grad == 0.0; }

inline double coef_exp(double x) {
  return std::exp(std::min(std::max(x, -700.0), 700.0));
}
inline fwd::Dual coef_exp(const fwd::Dual& x) { return math::safe_exp(x); }

inline double coef_log(double x) { return std::log(x); }
inline fwd::Dual coef_log(const fwd::Dual& x) { return fwd::log(x); }

inline double coef_sqrt(double x) { return std::sqrt(x); }
inline fwd::Dual coef_sqrt(const fwd::Dual& x) { return fwd::sqrt(x); }

inline double coef_lgamma(double x) { return std::lgamma(x); }
inline fwd::Dual coef_lgamma(const fwd::Dual& x) {
  return fwd::Dual(std::lgamma(x.val), math::portable_digamma(x.val) * x.grad);
}

// psi^(n), n >= 0
inline double polygamma_value(int n, double x) {
  return n == 0 ? math::portable_digamma(x) : math::portable_polygamma(n, x);
}
inline double coef_polygamma(int n, double x) { return polygamma_value(n, x); }
inline fwd::Dual coef_polygamma(int n, const fwd::Dual& x) {
  return fwd::Dual(polygamma_value(n, x.val), polygamma_value(n + 1, x.val) * x.grad);
}

// ---------------------------------------------------------------------------
// The jet
// ---------------------------------------------------------------------------

template <typename S>
struct Jet3 {
  S v, d1, d2, d3;

  Jet3() : v(0.0), d1(0.0), d2(0.0), d3(0.0) {}
  Jet3(double c) : v(c), d1(0.0), d2(0.0), d3(0.0) {}
  Jet3(const S& v_, const S& d1_, const S& d2_, const S& d3_)
      : v(v_), d1(d1_), d2(d2_), d3(d3_) {}

  // A constant: the value, with no dependence on t
  static Jet3 constant(const S& c) { return Jet3(c, S(0.0), S(0.0), S(0.0)); }
  // The line itself through x with slope `slope`: x + slope * t
  static Jet3 line(const S& x, double slope) {
    return Jet3(x, S(slope), S(0.0), S(0.0));
  }

  Jet3& operator+=(const Jet3& o) { *this = *this + o; return *this; }
  Jet3& operator-=(const Jet3& o) { *this = *this - o; return *this; }
  Jet3& operator*=(const Jet3& o) { *this = *this * o; return *this; }
  Jet3& operator/=(const Jet3& o) { *this = *this / o; return *this; }
};

// h(f) from h and its first three derivatives at f's value
template <typename S>
inline Jet3<S> compose(const Jet3<S>& f, const S& h0, const S& h1, const S& h2,
                       const S& h3) {
  const S f1sq = f.d1 * f.d1;
  return Jet3<S>(h0, h1 * f.d1, h2 * f1sq + h1 * f.d2,
                 h3 * f1sq * f.d1 + 3.0 * h2 * f.d1 * f.d2 + h1 * f.d3);
}

template <typename S>
inline Jet3<S> operator+(const Jet3<S>& a, const Jet3<S>& b) {
  return Jet3<S>(a.v + b.v, a.d1 + b.d1, a.d2 + b.d2, a.d3 + b.d3);
}
template <typename S>
inline Jet3<S> operator-(const Jet3<S>& a, const Jet3<S>& b) {
  return Jet3<S>(a.v - b.v, a.d1 - b.d1, a.d2 - b.d2, a.d3 - b.d3);
}
template <typename S>
inline Jet3<S> operator-(const Jet3<S>& a) {
  return Jet3<S>(0.0 - a.v, 0.0 - a.d1, 0.0 - a.d2, 0.0 - a.d3);
}
template <typename S>
inline Jet3<S> operator*(const Jet3<S>& a, const Jet3<S>& b) {
  return Jet3<S>(a.v * b.v,
                 a.d1 * b.v + a.v * b.d1,
                 a.d2 * b.v + 2.0 * a.d1 * b.d1 + a.v * b.d2,
                 a.d3 * b.v + 3.0 * a.d2 * b.d1 + 3.0 * a.d1 * b.d2 + a.v * b.d3);
}
template <typename S>
inline Jet3<S> reciprocal(const Jet3<S>& a) {
  const S r = 1.0 / a.v;
  const S r2 = r * r;
  return compose(a, r, 0.0 - r2, 2.0 * r2 * r, -6.0 * r2 * r2);
}
template <typename S>
inline Jet3<S> operator/(const Jet3<S>& a, const Jet3<S>& b) {
  return a * reciprocal(b);
}

template <typename S>
inline Jet3<S> operator+(const Jet3<S>& a, double b) { return a + Jet3<S>(b); }
template <typename S>
inline Jet3<S> operator+(double a, const Jet3<S>& b) { return Jet3<S>(a) + b; }
template <typename S>
inline Jet3<S> operator-(const Jet3<S>& a, double b) { return a - Jet3<S>(b); }
template <typename S>
inline Jet3<S> operator-(double a, const Jet3<S>& b) { return Jet3<S>(a) - b; }
template <typename S>
inline Jet3<S> operator*(const Jet3<S>& a, double b) {
  return Jet3<S>(a.v * b, a.d1 * b, a.d2 * b, a.d3 * b);
}
template <typename S>
inline Jet3<S> operator*(double a, const Jet3<S>& b) { return b * a; }
template <typename S>
inline Jet3<S> operator/(const Jet3<S>& a, double b) { return a * (1.0 / b); }
template <typename S>
inline Jet3<S> operator/(double a, const Jet3<S>& b) { return a * reciprocal(b); }

// The elementary functions on jets. They live beside the type so that
// argument-dependent lookup finds them from inside the templated
// log-densities, which call safe_exp, lgamma_fn and the rest unqualified.

template <typename S>
inline double get_value(const Jet3<S>& x) { return coef_value(x.v); }

template <typename S>
inline Jet3<S> safe_exp(const Jet3<S>& x) {
  const S e = coef_exp(x.v);
  return compose(x, e, e, e, e);
}

template <typename S>
inline Jet3<S> safe_log(const Jet3<S>& x) {
  if (coef_value(x.v) <= 0.0) return Jet3<S>(-1e10);
  const S r = 1.0 / x.v;
  const S r2 = r * r;
  return compose(x, coef_log(x.v), r, 0.0 - r2, 2.0 * r2 * r);
}

template <typename S>
inline Jet3<S> safe_sqrt(const Jet3<S>& x) {
  if (coef_value(x.v) <= 0.0) return Jet3<S>(0.0);
  const S s = coef_sqrt(x.v);
  const S r = 1.0 / x.v;
  return compose(x, s, 0.5 / s, -0.25 * r / s, 0.375 * r * r / s);
}

template <typename S>
inline Jet3<S> log1p_fn(const Jet3<S>& x) {
  const S u = 1.0 + x.v;
  const S r = 1.0 / u;
  const S r2 = r * r;
  return compose(x, coef_log(u), r, 0.0 - r2, 2.0 * r2 * r);
}

// Whether the jet is constant along the line: its derivatives are all zero,
// so a function of it needs only its value.
template <typename S>
inline bool is_constant(const Jet3<S>& x) {
  return coef_is_zero(x.d1) && coef_is_zero(x.d2) && coef_is_zero(x.d3);
}

template <typename S>
inline Jet3<S> lgamma_fn(const Jet3<S>& x) {
  if (is_constant(x)) return Jet3<S>::constant(coef_lgamma(x.v));
  return compose(x, coef_lgamma(x.v), coef_polygamma(0, x.v),
                      coef_polygamma(1, x.v), coef_polygamma(2, x.v));
}

template <typename S>
inline Jet3<S> digamma_fn(const Jet3<S>& x) {
  return compose(x, coef_polygamma(0, x.v), coef_polygamma(1, x.v),
                      coef_polygamma(2, x.v), coef_polygamma(3, x.v));
}

template <typename S>
inline Jet3<S> inv_logit(const Jet3<S>& x) {
  const double xv = coef_value(x.v);
  // p computed on the stable side of zero
  const S p = (xv > 0.0) ? 1.0 / (1.0 + coef_exp(0.0 - x.v))
                         : coef_exp(x.v) / (1.0 + coef_exp(x.v));
  const S q = 1.0 - p;
  const S pq = p * q;
  return compose(x, p, pq, pq * (1.0 - 2.0 * p), pq * (1.0 - 6.0 * pq));
}

template <typename S>
inline Jet3<S> safe_max(const Jet3<S>& a, const Jet3<S>& b) {
  return get_value(a) >= get_value(b) ? a : b;
}

// The links a third-order jet can carry exactly. prob_logit maps a predictor
// to the logit of its probability, the identity under every link but probit
// and cloglog; link_mean maps it to a mean, exp under the log link. The other
// links are known here only through their value and first derivative.
template <typename S>
inline Jet3<S> prob_logit(const Jet3<S>& eta, ratiod_link::LinkType link) {
  if (link == ratiod_link::LinkType::PROBIT ||
      link == ratiod_link::LinkType::CLOGLOG) {
    Rcpp::stop("the probit and cloglog links have no third-order expansion here");
  }
  return eta;
}

template <typename S>
inline Jet3<S> link_mean(const Jet3<S>& eta, ratiod_link::LinkType link) {
  if (link != ratiod_link::LinkType::LOG) {
    Rcpp::stop("only the log link of a mean has a third-order expansion here");
  }
  return safe_exp(eta);
}

}  // namespace jet

namespace math {
template <typename S>
struct is_jet<jet::Jet3<S>> : std::true_type {};
}  // namespace math

}  // namespace ratiod

#endif  // RATIOD_TAYLOR_JET_H
