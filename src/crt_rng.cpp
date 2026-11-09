// crt_rng.cpp
// Chinese Restaurant Table (CRT) distribution sampler implementation
// Reference: Zhou et al. (2012) "Lognormal and Gamma Mixed Negative Binomial Regression" ICML

#include "crt_rng.h"
#include <tulpa/count_math.h>
#include <Rcpp.h>
#include <cmath>

using namespace Rcpp;

namespace ratiod {

// Sample L ~ CRT(y, r) using sequential Bernoulli trials
// This is the standard O(y) algorithm from Zhou et al. (2012)
//
// The algorithm simulates the Chinese Restaurant Process:
// For customer i = 1, ..., y:
//   - Customer i starts a new table with probability r / (r + i - 1)
//   - Otherwise joins an existing table
// L = total number of tables created
int sample_crt(int y, double r) {
  if (y <= 0) return 0;
  if (r <= 0.0) return 0;  // Degenerate case: all at one table

  int L = 0;

  for (int i = 1; i <= y; i++) {
    // Probability that customer i starts a new table
    double p_new_table = r / (r + static_cast<double>(i) - 1.0);

    // Bernoulli trial
    if (R::runif(0.0, 1.0) < p_new_table) {
      L++;
    }
  }

  return L;
}

// Expected value of CRT(y, r)
// E[L] = r * (psi(y + r) - psi(r))
// where psi is the digamma function
double crt_mean(int y, double r) {
  if (y <= 0 || r <= 0.0) return 0.0;

  return r * tulpa::math::polygamma_rising_difference(0, y, r);
}

} // namespace ratiod

// ---------------------------------------------------------------------
// R exports
// ---------------------------------------------------------------------

// [[Rcpp::export]]
double cpp_crt_mean(int y, double r) {
  return ratiod::crt_mean(y, r);
}
