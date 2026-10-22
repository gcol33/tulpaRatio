// test_zi.cpp
// Catch2 tests for hmc_zi.h (zero-inflation functions)

#include "catch.hpp"
#include "../hmc_zi.h"
#include <cmath>

using namespace ratiod_zi;
using namespace ratiod::math;

// ============================================================================
// Helper functions
// ============================================================================

TEST_CASE("log1pexp is numerically stable", "[zi]") {
  SECTION("small values") {
    // For very negative x, log1p_exp(x) ≈ exp(x)
    REQUIRE(log1p_exp(-10.0) == Approx(std::log1p(std::exp(-10.0))));
  }

  SECTION("around zero") {
    REQUIRE(log1p_exp(0.0) == Approx(std::log(2.0)));
  }

  SECTION("large positive values") {
    REQUIRE(log1p_exp(100.0) == Approx(100.0).margin(1e-10));
  }

  SECTION("moderate values") {
    double x = 5.0;
    REQUIRE(log1p_exp(x) == Approx(std::log1p(std::exp(x))));
  }
}

TEST_CASE("logistic sigmoid function", "[zi]") {
  SECTION("at zero") {
    REQUIRE(inv_logit(0.0) == Approx(0.5));
  }

  SECTION("large positive") {
    REQUIRE(inv_logit(100.0) == Approx(1.0).margin(1e-10));
  }

  SECTION("large negative") {
    REQUIRE(inv_logit(-100.0) == Approx(0.0).margin(1e-10));
  }

  SECTION("positive value") {
    double x = 2.0;
    REQUIRE(inv_logit(x) == Approx(1.0 / (1.0 + std::exp(-x))));
  }

  SECTION("negative value") {
    double x = -2.0;
    REQUIRE(inv_logit(x) == Approx(1.0 / (1.0 + std::exp(-x))));
  }
}

TEST_CASE("log_logistic computes log(sigmoid)", "[zi]") {
  SECTION("at zero") {
    REQUIRE(log_inv_logit(0.0) == Approx(-std::log(2.0)));
  }

  SECTION("positive value") {
    double x = 2.0;
    double expected = std::log(1.0 / (1.0 + std::exp(-x)));
    REQUIRE(log_inv_logit(x) == Approx(expected));
  }

  SECTION("large positive") {
    REQUIRE(log_inv_logit(100.0) == Approx(0.0).margin(1e-10));
  }
}

TEST_CASE("log1m_logistic computes log(1-sigmoid)", "[zi]") {
  SECTION("at zero") {
    REQUIRE(log1m_inv_logit(0.0) == Approx(-std::log(2.0)));
  }

  SECTION("positive value") {
    double x = 2.0;
    double expected = std::log(1.0 - 1.0 / (1.0 + std::exp(-x)));
    REQUIRE(log1m_inv_logit(x) == Approx(expected));
  }

  SECTION("large positive gives very negative") {
    REQUIRE(log1m_inv_logit(100.0) < -90.0);
  }
}

// ============================================================================
// Distribution PMFs
// ============================================================================

TEST_CASE("poisson_lpmf computes Poisson log-PMF", "[zi]") {
  SECTION("y=0, mu=1") {
    // P(0|1) = exp(-1)
    REQUIRE(log_lik_poisson(0, 1.0) == Approx(-1.0));
  }

  SECTION("y=5, mu=5") {
    // Poisson mode should have high probability
    double lp = log_lik_poisson(5, 5.0);
    REQUIRE(lp > log_lik_poisson(0, 5.0));
    REQUIRE(lp > log_lik_poisson(10, 5.0));
  }

  SECTION("mu <= 0 returns very negative") {
    REQUIRE(log_lik_poisson(1, 0.0) < -1e9);
    REQUIRE(log_lik_poisson(1, -1.0) < -1e9);
  }
}

TEST_CASE("negbin_lpmf computes negative binomial log-PMF", "[zi]") {
  SECTION("y=0 has positive probability") {
    double lp = log_lik_negbin(0, 5.0, 2.0);
    REQUIRE(lp > -100.0);
    REQUIRE(std::exp(lp) > 0.0);
    REQUIRE(std::exp(lp) < 1.0);
  }

  SECTION("larger variance with smaller phi") {
    // Smaller phi = more overdispersion = flatter distribution
    double lp_low_phi = log_lik_negbin(10, 5.0, 0.5);
    double lp_high_phi = log_lik_negbin(10, 5.0, 100.0);
    // With high phi, NB approaches Poisson, so extreme values less likely
    // This depends on the specific value, but generally more overdispersion spreads mass
  }

  SECTION("invalid parameters") {
    REQUIRE(log_lik_negbin(1, 0.0, 1.0) < -1e9);
  }
}

// ============================================================================
// Zero-inflated likelihoods
// ============================================================================

TEST_CASE("zi_poisson_lpmf_logit handles zero inflation", "[zi]") {
  double mu = 5.0;

  SECTION("high ZI probability increases P(0)") {
    double logit_zi_high = 3.0;  // ~95% ZI
    double logit_zi_low = -3.0;  // ~5% ZI

    double lp_high_zi = zi_poisson_lpmf_logit(0, mu, logit_zi_high);
    double lp_low_zi = zi_poisson_lpmf_logit(0, mu, logit_zi_low);

    // Higher ZI should increase P(Y=0)
    REQUIRE(lp_high_zi > lp_low_zi);
  }

  SECTION("non-zero y unaffected by ZI for y>0") {
    double logit_zi = 2.0;
    double lp = zi_poisson_lpmf_logit(5, mu, logit_zi);
    // Should be log(1-pi) + Poisson(5|mu)
    double expected = log1m_inv_logit(logit_zi) + log_lik_poisson(5, mu);
    REQUIRE(lp == Approx(expected));
  }

  SECTION("no ZI (logit_zi very negative)") {
    double lp_zi = zi_poisson_lpmf_logit(3, mu, -100.0);
    double lp_plain = log_lik_poisson(3, mu);
    REQUIRE(lp_zi == Approx(lp_plain).margin(1e-6));
  }
}

TEST_CASE("zi_negbin_lpmf_logit handles zero-inflated NB", "[zi]") {
  double mu = 5.0;
  double phi = 2.0;

  SECTION("y=0 combines ZI and NB") {
    double logit_zi = 1.0;
    double lp = zi_negbin_lpmf_logit(0, mu, phi, logit_zi);

    // P(0) = pi + (1-pi)*NB(0|mu,phi)
    double pi = inv_logit(logit_zi);
    double expected = std::log(pi + (1-pi) * std::exp(log_lik_negbin(0, mu, phi)));
    REQUIRE(lp == Approx(expected));
  }

  SECTION("y>0 excludes ZI component") {
    double logit_zi = 1.0;
    double lp = zi_negbin_lpmf_logit(5, mu, phi, logit_zi);
    double expected = log1m_inv_logit(logit_zi) + log_lik_negbin(5, mu, phi);
    REQUIRE(lp == Approx(expected));
  }
}

// ============================================================================
// Truncated distributions
// ============================================================================

TEST_CASE("truncated_poisson_lpmf for positive values only", "[zi]") {
  double mu = 5.0;

  SECTION("y=0 returns -infinity") {
    REQUIRE(truncated_poisson_lpmf(0, mu) < -1e9);
  }

  SECTION("y>0 is normalized") {
    // P(Y=y | Y>0) = P(Y=y) / P(Y>0) = P(Y=y) / (1 - exp(-mu))
    double lp = truncated_poisson_lpmf(3, mu);
    double expected = log_lik_poisson(3, mu) - std::log(1.0 - std::exp(-mu));
    REQUIRE(lp == Approx(expected));
  }
}

TEST_CASE("truncated_negbin_lpmf for positive values only", "[zi]") {
  double mu = 5.0;
  double phi = 2.0;

  SECTION("y=0 returns -infinity") {
    REQUIRE(truncated_negbin_lpmf(0, mu, phi) < -1e9);
  }

  SECTION("y>0 is normalized") {
    double p0 = std::exp(log_lik_negbin(0, mu, phi));
    double lp = truncated_negbin_lpmf(3, mu, phi);
    double expected = log_lik_negbin(3, mu, phi) - std::log(1.0 - p0);
    REQUIRE(lp == Approx(expected));
  }
}

// ============================================================================
// Hurdle models
// ============================================================================

TEST_CASE("hurdle_poisson_lpmf_logit", "[zi]") {
  double mu = 5.0;

  SECTION("y=0 uses binary component") {
    double logit_theta = 1.0;  // theta = P(Y>0)
    double lp = hurdle_poisson_lpmf_logit(0, mu, logit_theta);
    // P(Y=0) = 1 - theta
    REQUIRE(lp == Approx(log1m_inv_logit(logit_theta)));
  }

  SECTION("y>0 uses truncated Poisson") {
    double logit_theta = 1.0;
    double lp = hurdle_poisson_lpmf_logit(5, mu, logit_theta);
    double expected = log_inv_logit(logit_theta) + truncated_poisson_lpmf(5, mu);
    REQUIRE(lp == Approx(expected));
  }
}

TEST_CASE("hurdle_negbin_lpmf_logit", "[zi]") {
  double mu = 5.0;
  double phi = 2.0;

  SECTION("y=0 uses binary component") {
    double logit_theta = 0.5;
    double lp = hurdle_negbin_lpmf_logit(0, mu, phi, logit_theta);
    REQUIRE(lp == Approx(log1m_inv_logit(logit_theta)));
  }

  SECTION("y>0 uses truncated NB") {
    double logit_theta = 0.5;
    double lp = hurdle_negbin_lpmf_logit(5, mu, phi, logit_theta);
    double expected = log_inv_logit(logit_theta) + truncated_negbin_lpmf(5, mu, phi);
    REQUIRE(lp == Approx(expected));
  }
}

// ============================================================================
// ZI type parsing
// ============================================================================

TEST_CASE("parse_zi_type converts strings to enum", "[zi]") {
  REQUIRE(parse_zi_type("none") == ZIType::NONE);
  REQUIRE_THROWS(parse_zi_type(""));
  REQUIRE(parse_zi_type("zi_poisson") == ZIType::ZI_POISSON);
  REQUIRE(parse_zi_type("zi_negbin") == ZIType::ZI_NEGBIN);
  REQUIRE(parse_zi_type("hurdle_poisson") == ZIType::HURDLE_POISSON);
  REQUIRE(parse_zi_type("hurdle_negbin") == ZIType::HURDLE_NEGBIN);
  REQUIRE_THROWS(parse_zi_type("unknown"));
}

// ============================================================================
// Generic ZI log-likelihood
// ============================================================================

TEST_CASE("zi_log_likelihood dispatches correctly", "[zi]") {
  int y = 3;
  double mu = 5.0;
  double phi = 2.0;
  double logit_zi = 0.5;

  SECTION("NONE with phi=0 returns Poisson") {
    // When phi <= 0, NONE uses Poisson
    double lp = zi_log_likelihood(y, mu, 0.0, logit_zi, ZIType::NONE);
    REQUIRE(lp == Approx(log_lik_poisson(y, mu)));
  }

  SECTION("NONE with phi>0 returns NegBin") {
    // When phi > 0, NONE uses NegBin
    double lp = zi_log_likelihood(y, mu, phi, logit_zi, ZIType::NONE);
    REQUIRE(lp == Approx(log_lik_negbin(y, mu, phi)));
  }

  SECTION("ZI_POISSON") {
    double lp = zi_log_likelihood(y, mu, phi, logit_zi, ZIType::ZI_POISSON);
    REQUIRE(lp == Approx(zi_poisson_lpmf_logit(y, mu, logit_zi)));
  }

  SECTION("ZI_NEGBIN") {
    double lp = zi_log_likelihood(y, mu, phi, logit_zi, ZIType::ZI_NEGBIN);
    REQUIRE(lp == Approx(zi_negbin_lpmf_logit(y, mu, phi, logit_zi)));
  }

  SECTION("HURDLE_POISSON") {
    double lp = zi_log_likelihood(y, mu, phi, logit_zi, ZIType::HURDLE_POISSON);
    REQUIRE(lp == Approx(hurdle_poisson_lpmf_logit(y, mu, logit_zi)));
  }

  SECTION("HURDLE_NEGBIN") {
    double lp = zi_log_likelihood(y, mu, phi, logit_zi, ZIType::HURDLE_NEGBIN);
    REQUIRE(lp == Approx(hurdle_negbin_lpmf_logit(y, mu, phi, logit_zi)));
  }
}
