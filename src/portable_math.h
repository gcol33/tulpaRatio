// portable_math.h
// Thread-safe C++ math functions that replace R::lgammafn, R::digamma, etc.
// No R dependency in hot paths - avoids R's C stack checking and is OpenMP-safe.

#ifndef RATIOD_PORTABLE_MATH_H
#define RATIOD_PORTABLE_MATH_H

#include <cmath>
#include <limits>

namespace ratiod {
namespace math {

// Portable digamma using asymptotic expansion + recurrence
// For x > 0 only (sufficient for all our use cases).
// Unrolled recurrence (branchless for x >= 1) eliminates while-loop overhead.
inline double portable_digamma(double x) {
    if (x <= 0.0) return -1e10;  // guard (should not happen in practice)

    // Unrolled recurrence: always shift x to x+N where x+N >= 7.
    // For our use cases x >= 1 (counts + dispersion), so N=6 always suffices.
    // Each recurrence step: digamma(x) = digamma(x+1) - 1/x
    // Accumulate result = -1/x - 1/(x+1) - ... - 1/(x+N-1), then asymptotic at x+N.
    //
    // Conditional version for x < 7: at most 6 steps, unrolled.
    double result = 0.0;
    if (x < 7.0) {
        // Unrolled: up to 6 recurrence steps (handles x >= 1)
        if (x < 1.0) { result -= 1.0 / x; x += 1.0; }
        if (x < 2.0) { result -= 1.0 / x; x += 1.0; }
        if (x < 3.0) { result -= 1.0 / x; x += 1.0; }
        if (x < 4.0) { result -= 1.0 / x; x += 1.0; }
        if (x < 5.0) { result -= 1.0 / x; x += 1.0; }
        if (x < 6.0) { result -= 1.0 / x; x += 1.0; }
        if (x < 7.0) { result -= 1.0 / x; x += 1.0; }
    }
    // Asymptotic expansion: digamma(x) ~ log(x) - 1/(2x) - sum B_{2k}/(2k*x^{2k})
    double inv_x = 1.0 / x;
    double inv_x2 = inv_x * inv_x;
    // Bernoulli coefficients: B_2/(2*x^2), B_4/(4*x^4), ...
    result += std::log(x) - 0.5 * inv_x
              - inv_x2 * (1.0/12.0
              - inv_x2 * (1.0/120.0
              - inv_x2 * (1.0/252.0
              - inv_x2 * (1.0/240.0
              - inv_x2 * (5.0/660.0
              - inv_x2 * (691.0/32760.0
              - inv_x2 * (1.0/12.0)))))));
    return result;
}

// Fused digamma + lgamma: compute both from the same recurrence shift.
// Saves log() calls by accumulating product in recurrence (single log vs up to 7).
// Returns {digamma(x), lgamma(x)}.
inline std::pair<double, double> portable_digamma_lgamma(double x) {
    if (x <= 0.0) return {-1e10, 1e10};

    double dig_result = 0.0;

    // Recurrence: digamma(x) = digamma(x+1) - 1/x
    //             lgamma(x)  = lgamma(x+1) - log(x)
    // Product trick: accumulate x*(x+1)*...*(x+k-1), take single log at end.
    // Saves up to 6 log() calls (each ~20-30 cycles).
    double product = 1.0;
    if (x < 7.0) {
        if (x < 1.0) { dig_result -= 1.0 / x; product *= x; x += 1.0; }
        if (x < 2.0) { dig_result -= 1.0 / x; product *= x; x += 1.0; }
        if (x < 3.0) { dig_result -= 1.0 / x; product *= x; x += 1.0; }
        if (x < 4.0) { dig_result -= 1.0 / x; product *= x; x += 1.0; }
        if (x < 5.0) { dig_result -= 1.0 / x; product *= x; x += 1.0; }
        if (x < 6.0) { dig_result -= 1.0 / x; product *= x; x += 1.0; }
        if (x < 7.0) { dig_result -= 1.0 / x; product *= x; x += 1.0; }
    }
    double lg_result = -std::log(product);  // Single log replaces up to 7

    double inv_x = 1.0 / x;
    double inv_x2 = inv_x * inv_x;
    double log_x = std::log(x);

    // Digamma asymptotic
    dig_result += log_x - 0.5 * inv_x
                  - inv_x2 * (1.0/12.0
                  - inv_x2 * (1.0/120.0
                  - inv_x2 * (1.0/252.0
                  - inv_x2 * (1.0/240.0
                  - inv_x2 * (5.0/660.0
                  - inv_x2 * (691.0/32760.0
                  - inv_x2 * (1.0/12.0)))))));

    // Lgamma asymptotic (Stirling): lgamma(x) = (x-0.5)*log(x) - x + 0.5*log(2*pi) + sum
    lg_result += (x - 0.5) * log_x - x + 0.9189385332046727  // 0.5*log(2*pi)
                 + inv_x * (1.0/12.0
                 - inv_x2 * (1.0/360.0
                 - inv_x2 * (1.0/1260.0
                 - inv_x2 * (1.0/1680.0
                 - inv_x2 * (1.0/1188.0)))));

    return {dig_result, lg_result};
}

// Portable trigamma using asymptotic expansion + recurrence
inline double portable_trigamma(double x) {
    if (x <= 0.0) return 1e10;  // guard
    double result = 0.0;
    // Use recurrence to shift x >= 7
    while (x < 7.0) {
        result += 1.0 / (x * x);
        x += 1.0;
    }
    // Asymptotic expansion: trigamma(x) ~ 1/x + 1/(2x^2) + sum B_{2k}/(x^{2k+1})
    double inv_x = 1.0 / x;
    double inv_x2 = inv_x * inv_x;
    result += inv_x + 0.5 * inv_x2
              + inv_x2 * inv_x * (1.0/6.0
              - inv_x2 * (1.0/30.0
              - inv_x2 * (1.0/42.0
              - inv_x2 * (1.0/30.0
              - inv_x2 * (5.0/66.0)))));
    return result;
}

// Polygamma psi^(n)(x) for n >= 1, x > 0: the recurrence
// psi^(n)(x) = psi^(n)(x + 1) + (-1)^(n+1) n! / x^(n+1) shifts x to 12 or
// above, where the asymptotic series
// (-1)^(n+1) [ (n-1)!/x^n + n!/(2 x^(n+1)) + sum_k B_2k (2k+n-1)! / ((2k)! x^(2k+n)) ]
// is accurate to double precision with five Bernoulli terms.
inline double portable_polygamma(int n, double x) {
    if (x <= 0.0) return std::numeric_limits<double>::quiet_NaN();
    double n_fact = 1.0;
    for (int k = 2; k <= n; k++) n_fact *= k;
    const double sign = (n % 2 == 1) ? 1.0 : -1.0;  // (-1)^(n+1)
    double result = 0.0;
    while (x < 12.0) {
        result += sign * n_fact / std::pow(x, n + 1);
        x += 1.0;
    }
    static const double bernoulli[5] = {1.0 / 6.0, -1.0 / 30.0, 1.0 / 42.0,
                                        -1.0 / 30.0, 5.0 / 66.0};
    double series = (n_fact / n) / std::pow(x, n) + 0.5 * n_fact / std::pow(x, n + 1);
    double fact_num = n_fact / n;  // (n-1)!, grown to (2k+n-1)!
    double fact_den = 1.0;         // (2k)!
    for (int k = 1; k <= 5; k++) {
        fact_num *= static_cast<double>(2 * k + n - 2) * (2 * k + n - 1);
        fact_den *= static_cast<double>(2 * k - 1) * (2 * k);
        series += bernoulli[k - 1] * fact_num / (fact_den * std::pow(x, 2 * k + n));
    }
    return result + sign * series;
}

// Portable lchoose: log(C(n,k)) = lgamma(n+1) - lgamma(k+1) - lgamma(n-k+1)
inline double portable_lchoose(int n, int k) {
    if (k < 0 || k > n) return -std::numeric_limits<double>::infinity();
    return std::lgamma(n + 1.0) - std::lgamma(k + 1.0) - std::lgamma(n - k + 1.0);
}

}  // namespace math
}  // namespace ratiod

#endif  // RATIOD_PORTABLE_MATH_H
