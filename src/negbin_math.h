#ifndef TULPARATIO_NEGBIN_MATH_H
#define TULPARATIO_NEGBIN_MATH_H

#include <cmath>

namespace tulpaRatio {
namespace count_math {

struct ScalarMath {
  static double value(double x) { return x; }
  static double log(double x) { return std::log(x); }
  static double log1p(double x) { return std::log1p(x); }
  static double lgamma(double x) { return std::lgamma(x); }
};

// log Gamma(phi + y) - log Gamma(phi), using the finite recurrence for small
// counts and the Stirling difference at large phi. The omitted Stirling term
// is bounded by 1 / (1680 phi^7) on the asymptotic branch.
template <typename T, class Math = ScalarMath>
inline T lgamma_ratio_count(int y, const T& phi) {
  if (y <= 32) {
    T sum(0.0);
    for (int j = 0; j < y; ++j) sum = sum + Math::log(phi + T(j));
    return sum;
  }
  if (Math::value(phi) < 1e3) return Math::lgamma(phi + T(y)) - Math::lgamma(phi);
  const T yy(y), l1 = Math::log1p(yy / phi);
  auto stirling = [](const T& x) {
    const T inv = T(1.0) / x, inv2 = inv * inv;
    return inv * (T(1.0 / 12.0) + inv2 * (T(-1.0 / 360.0) + inv2 * T(1.0 / 1260.0)));
  };
  return phi * l1 - yy + yy * (Math::log(phi) + l1) - T(0.5) * l1
       + stirling(phi + yy) - stirling(phi);
}

// psi^(k)(phi + y) - psi^(k)(phi). Each asymptotic power difference is
// phi^-q expm1(-q log1p(y / phi)), preserving its small relative increment.
inline double polygamma_difference(int k, int y, double phi) {
  double factorial = 1.0;
  for (int i = 2; i <= k; ++i) factorial *= i;
  const double sign = k % 2 == 0 ? 1.0 : -1.0;
  double sum = 0.0;
  if (y <= 32) {
    for (int j = 0; j < y; ++j) sum += sign * factorial / std::pow(phi + j, k + 1);
    return sum;
  }
  while (phi < 10.0 && y > 0) {
    sum += sign * factorial / std::pow(phi, k + 1);
    phi += 1.0;
    --y;
  }
  const double L = std::log1p(y / phi);
  auto power_difference = [&](int q) {
    return std::pow(phi, -q) * std::expm1(-q * L);
  };
  if (k == 0) sum += L;
  else sum -= sign * (factorial / k) * power_difference(k);
  static const double coef[] = {-0.5, -1.0 / 12.0, 1.0 / 120.0, -1.0 / 252.0,
      1.0 / 240.0, -1.0 / 132.0, 691.0 / 32760.0, -1.0 / 12.0};
  static const int power[] = {1, 2, 4, 6, 8, 10, 12, 14};
  for (int t = 0; t < 8; ++t) {
    double rising = 1.0;
    for (int i = 0; i < k; ++i) rising *= power[t] + i;
    sum += coef[t] * sign * rising * power_difference(power[t] + k);
  }
  return sum;
}

template <typename T, class Math = ScalarMath>
inline T log_p0(const T& mu, const T& phi) {
  return -phi * Math::log1p(mu / phi);
}

// The count normalizer can arrive from a per-count lookup table.
template <typename T, class Math = ScalarMath>
inline T log_pmf(int y, const T& mu, const T& phi, const T& rising, double log_factorial) {
  const T s = Math::log1p(mu / phi);
  return rising - T(log_factorial) + T(y) * (Math::log(mu) - Math::log(phi) - s) - phi * s;
}

template <typename T, class Math = ScalarMath>
inline T log_pmf(int y, const T& mu, const T& phi) {
  return log_pmf<T, Math>(y, mu, phi, lgamma_ratio_count<T, Math>(y, phi), std::lgamma(y + 1.0));
}

inline double size_score(int y, double mu, double phi, double digamma_difference) {
  return digamma_difference - std::log1p(mu / phi) + (mu - y) / (phi + mu);
}

inline double size_score(int y, double mu, double phi) {
  return size_score(y, mu, phi, polygamma_difference(0, y, phi));
}

}  // namespace count_math
}  // namespace tulpaRatio

#endif
