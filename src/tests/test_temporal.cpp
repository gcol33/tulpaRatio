// test_temporal.cpp
// Catch2 tests for hmc_temporal.h

#define _USE_MATH_DEFINES  // For M_PI on Windows
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "catch.hpp"
#include "../hmc_temporal.h"
#include <vector>

using namespace ratiod_temporal;

// ============================================================================
// RW1 quadratic form
// ============================================================================

TEST_CASE("rw1_quadratic_form computes sum of squared differences", "[temporal]") {
  SECTION("constant vector has zero quadform") {
    double phi[] = {5.0, 5.0, 5.0, 5.0, 5.0};
    REQUIRE(rw1_quadratic_form(phi, 5, false) == Approx(0.0));
  }

  SECTION("linear increasing") {
    double phi[] = {0.0, 1.0, 2.0, 3.0, 4.0};
    // (1-0)^2 + (2-1)^2 + (3-2)^2 + (4-3)^2 = 4
    REQUIRE(rw1_quadratic_form(phi, 5, false) == Approx(4.0));
  }

  SECTION("two points") {
    double phi[] = {0.0, 3.0};
    REQUIRE(rw1_quadratic_form(phi, 2, false) == Approx(9.0));
  }

  SECTION("cyclic adds wrap-around edge") {
    double phi[] = {0.0, 1.0, 2.0};
    // Non-cyclic: (1-0)^2 + (2-1)^2 = 2
    // Cyclic: add (0-2)^2 = 4, total = 6
    REQUIRE(rw1_quadratic_form(phi, 3, false) == Approx(2.0));
    REQUIRE(rw1_quadratic_form(phi, 3, true) == Approx(6.0));
  }

  SECTION("cyclic constant still zero") {
    double phi[] = {3.0, 3.0, 3.0, 3.0};
    REQUIRE(rw1_quadratic_form(phi, 4, true) == Approx(0.0));
  }
}

// ============================================================================
// RW2 quadratic form
// ============================================================================

TEST_CASE("rw2_quadratic_form penalizes curvature", "[temporal]") {
  SECTION("constant vector has zero quadform") {
    double phi[] = {2.0, 2.0, 2.0, 2.0, 2.0};
    REQUIRE(rw2_quadratic_form(phi, 5, false) == Approx(0.0));
  }

  SECTION("linear trend has zero quadform") {
    double phi[] = {0.0, 1.0, 2.0, 3.0, 4.0};
    // Second differences: 0-2+2=0, 1-4+3=0, 2-6+4=0
    REQUIRE(rw2_quadratic_form(phi, 5, false) == Approx(0.0));
  }

  SECTION("quadratic trend has non-zero quadform") {
    double phi[] = {0.0, 1.0, 4.0, 9.0, 16.0};  // t^2
    // Second diffs: 0-2+4=2, 1-8+9=2, 4-18+16=2
    // Sum of squares: 3 * 4 = 12
    REQUIRE(rw2_quadratic_form(phi, 5, false) == Approx(12.0));
  }

  SECTION("short vector") {
    double phi[] = {1.0, 2.0};
    REQUIRE(rw2_quadratic_form(phi, 2, false) == Approx(0.0));  // T < 3
  }

  SECTION("three points") {
    double phi[] = {0.0, 1.0, 3.0};
    // Second diff: 0 - 2 + 3 = 1
    REQUIRE(rw2_quadratic_form(phi, 3, false) == Approx(1.0));
  }

  SECTION("cyclic linear is no longer zero") {
    double phi[] = {0.0, 1.0, 2.0, 3.0};
    // Non-cyclic: 0-2+2=0, 1-4+3=0 -> quadform = 0
    // Cyclic adds: (2-6+0)=-4 squared=16, (3-0+1)=4 squared=16 -> total = 32
    REQUIRE(rw2_quadratic_form(phi, 4, false) == Approx(0.0));
    REQUIRE(rw2_quadratic_form(phi, 4, true) == Approx(32.0));
  }
}

// ============================================================================
// AR1 log-density
// ============================================================================

TEST_CASE("ar1_log_density computes AR(1) likelihood", "[temporal]") {
  SECTION("rho=0 is IID normal") {
    double phi[] = {1.0, 0.5, -0.3};
    double tau = 2.0;
    double rho = 0.0;

    double log_dens = ratiod_ar1::ar1_log_density(phi, 3, rho, tau);

    // IID N(0, 1/tau): sum of -0.5*tau*phi^2 - 0.5*log(2*pi/tau)
    double expected = 0.0;
    double sigma2 = 1.0 / tau;
    for (int i = 0; i < 3; i++) {
      expected -= 0.5 * phi[i] * phi[i] / sigma2;
      expected -= 0.5 * std::log(2.0 * M_PI * sigma2);
    }
    REQUIRE(log_dens == Approx(expected).margin(1e-10));
  }

  SECTION("positive rho increases density for correlated series") {
    // A series where consecutive values are similar should have higher
    // density under high rho than low rho
    double phi[] = {1.0, 0.9, 0.8, 0.7};
    double tau = 1.0;

    double dens_low_rho = ratiod_ar1::ar1_log_density(phi, 4, 0.1, tau);
    double dens_high_rho = ratiod_ar1::ar1_log_density(phi, 4, 0.9, tau);

    REQUIRE(dens_high_rho > dens_low_rho);
  }

  SECTION("short series") {
    double phi[] = {1.0};
    REQUIRE(ratiod_ar1::ar1_log_density(phi, 1, 0.5, 1.0) == Approx(0.0));
  }

  SECTION("two points") {
    double phi[] = {1.0, 0.5};
    double rho = 0.8;
    double tau = 2.0;

    double log_dens = ratiod_ar1::ar1_log_density(phi, 2, rho, tau);

    // Manual calculation
    double marginal_var = 1.0 / (tau * (1.0 - rho * rho));
    double sigma2 = 1.0 / tau;
    double expected = -0.5 * phi[0] * phi[0] / marginal_var;
    expected -= 0.5 * std::log(2.0 * M_PI * marginal_var);
    double resid = phi[1] - rho * phi[0];
    expected -= 0.5 * resid * resid / sigma2;
    expected -= 0.5 * std::log(2.0 * M_PI * sigma2);

    REQUIRE(log_dens == Approx(expected));
  }
}

// ============================================================================
// Temporal log-prior
// ============================================================================

TEST_CASE("temporal_log_prior dispatches to correct model", "[temporal]") {
  double phi[] = {0.0, 1.0, 1.5, 1.2, 0.8};
  double tau = 2.0;
  double rho = 0.7;

  SECTION("NONE returns zero") {
    REQUIRE(temporal_log_prior(phi, 5, TemporalType::NONE, std::log(tau), tau, rho, false) == Approx(0.0));
  }

  SECTION("RW1 uses rw1_quadratic_form") {
    double log_prior = temporal_log_prior(phi, 5, TemporalType::RW1, std::log(tau), tau, rho, false);

    // Expected: 0.5 * (T-1) * log(tau) - 0.5 * tau * quadform
    double quad = rw1_quadratic_form(phi, 5, false);
    int rank = 4;  // T - 1
    double expected = 0.5 * rank * std::log(tau) - 0.5 * tau * quad;

    REQUIRE(log_prior == Approx(expected));
  }

  SECTION("RW2 uses rw2_quadratic_form") {
    double log_prior = temporal_log_prior(phi, 5, TemporalType::RW2, std::log(tau), tau, rho, false);

    double quad = rw2_quadratic_form(phi, 5, false);
    int rank = 3;  // T - 2
    double expected = 0.5 * rank * std::log(tau) - 0.5 * tau * quad;

    REQUIRE(log_prior == Approx(expected));
  }

  SECTION("AR1 uses ar1_log_density") {
    double log_prior = temporal_log_prior(phi, 5, TemporalType::AR1, std::log(tau), tau, rho, false);
    double expected = ratiod_ar1::ar1_log_density(phi, 5, rho, tau);
    REQUIRE(log_prior == Approx(expected));
  }

  SECTION("IID is sum of normal log-densities") {
    double log_prior = temporal_log_prior(phi, 5, TemporalType::IID, std::log(tau), tau, rho, false);

    double expected = 0.0;
    double sigma2 = 1.0 / tau;
    for (int i = 0; i < 5; i++) {
      expected -= 0.5 * phi[i] * phi[i] / sigma2;
      expected -= 0.5 * std::log(2.0 * M_PI * sigma2);
    }

    REQUIRE(log_prior == Approx(expected));
  }
}

// ============================================================================
// Cyclic RW properties
// ============================================================================

TEST_CASE("RW1 rank excludes the constant direction", "[temporal]") {
  double phi[] = {0.0, 1.0, 2.0, 1.0};  // 4 points

  SECTION("non-cyclic has rank T-1") {
    double log_prior = temporal_log_prior(phi, 4, TemporalType::RW1, std::log(2.0), 2.0, 0.0, false);
    // Check the log(tau) coefficient is (T-1)/2 = 1.5
    double quad = rw1_quadratic_form(phi, 4, false);
    double expected = 0.5 * 3 * std::log(2.0) - quad;
    REQUIRE(log_prior == Approx(expected));
  }

  SECTION("cyclic has rank T-1") {
    double log_prior = temporal_log_prior(phi, 4, TemporalType::RW1, std::log(2.0), 2.0, 0.0, true);
    double quad = rw1_quadratic_form(phi, 4, true);
    double expected = 0.5 * 3 * std::log(2.0) - quad;
    REQUIRE(log_prior == Approx(expected));
  }
}
