// pg_rng.h
// Pólya-Gamma random number generator
// Based on Polson, Scott & Windle (2013) JASA

#ifndef RATIOD_PG_RNG_H
#define RATIOD_PG_RNG_H

#include <Rcpp.h>
#include <cmath>

namespace ratiod {

// Sample from PG(1, z) using the Devroye method
// This is the core sampler for Pólya-Gamma augmentation
double rpg1(double z);

// Sample from PG(b, z) for integer b
// Uses sum of b independent PG(1, z) draws
double rpg_int(int b, double z);

// Real-shape PG draw using integer additivity and the fractional inverse CDF.
double rpg_real(double b, double z);

// Vectorized PG(1, z) sampler
Rcpp::NumericVector rpg1_vec(Rcpp::NumericVector z);

// Vectorized PG(b, z) sampler for integer b vector
Rcpp::NumericVector rpg_vec(Rcpp::IntegerVector b, Rcpp::NumericVector z);

// ---------------------------------------------------------------------
// Internal helper functions
// ---------------------------------------------------------------------

// Sample from J*(1, z) - the tilted Jacobi distribution
// Used in the Devroye method for PG(1, z)
double sample_jacobi_tilted(double z);

// Inverse Gaussian sampler (needed for large z)
double rinvgauss(double mu, double lambda);

// Exponential tilting constant
inline double cosh_safe(double x) {
  if (std::abs(x) > 500) {
    return std::exp(std::abs(x)) / 2.0;
  }
  return std::cosh(x);
}

} // namespace ratiod

#endif // QUOTR_PG_RNG_H
