test_that("ratiod_negbin_negbin creates valid family", {
  fam <- ratiod_negbin_negbin()

  expect_s3_class(fam, "ratiod_family")
  expect_equal(fam$name, "negbin_negbin")
  expect_equal(fam$numerator$distribution, "neg_binomial_2")
  expect_equal(fam$denominator$distribution, "neg_binomial_2")
})

test_that("ratiod_binomial creates valid family", {
  fam <- ratiod_binomial()

  expect_s3_class(fam, "ratiod_family")
  expect_equal(fam$numerator$distribution, "binomial")
  expect_equal(fam$denominator$distribution, "fixed")
})

test_that("ratiod_poisson_gamma creates valid family", {
  fam <- ratiod_poisson_gamma()

  expect_s3_class(fam, "ratiod_family")
  expect_equal(fam$numerator$distribution, "poisson")
  expect_equal(fam$denominator$distribution, "gamma")
})

test_that("ratiod_negbin_gamma creates valid family", {
  fam <- ratiod_negbin_gamma()

  expect_s3_class(fam, "ratiod_family")
  expect_equal(fam$name, "negbin_gamma")
  expect_equal(fam$numerator$distribution, "neg_binomial_2")
  expect_equal(fam$denominator$distribution, "gamma")
  expect_equal(fam$numerator$link, "log")
  expect_equal(fam$denominator$link, "log")
})

test_that("invalid link functions are rejected", {
  expect_error(
    ratiod_negbin_negbin(link_num = "identity"),
    "not supported"
  )

  expect_error(
    ratiod_binomial(link = "log"),
    "not supported"
  )
})

# Tests for v0.9.0 extended families

test_that("ratiod_beta_binomial creates valid family", {
  fam <- ratiod_beta_binomial()

  expect_s3_class(fam, "ratiod_family")
  expect_equal(fam$name, "beta_binomial_fixed")
  expect_equal(fam$numerator$distribution, "beta_binomial")
  expect_equal(fam$numerator$link, "logit")
  expect_true(fam$overdispersed)
})

test_that("ratiod_gamma_gamma creates valid family", {
  fam <- ratiod_gamma_gamma()

  expect_s3_class(fam, "ratiod_family")
  expect_equal(fam$name, "gamma_gamma")
  expect_equal(fam$numerator$distribution, "gamma")
  expect_equal(fam$denominator$distribution, "gamma")
  expect_equal(fam$numerator$link, "log")
  expect_equal(fam$denominator$link, "log")
})

test_that("a family accepts only the links the samplers fit", {
  # Every likelihood kernel uses the canonical link, so any other link would
  # be stored on the family and silently not fitted.
  expect_error(ratiod_binomial(link = "probit"), "not supported")
  expect_error(ratiod_beta_binomial(link = "cloglog"), "not supported")
  expect_error(ratiod_gamma_gamma(link_num = "identity"), "not supported")
  expect_error(ratiod_lognormal(link_num = "identity"), "not supported")
  expect_error(ratiod_zibinomial(link_zi = "probit"), "not supported")
  expect_error(ratiod_hurdle_pois(link_hurdle = "cloglog"), "not supported")
})

test_that("inflated count families fit the denominators they name", {
  expect_equal(get_hmc_model_type(ratiod_zipois()), "poisson_gamma")
  expect_equal(get_hmc_model_type(ratiod_zinegbin()), "negbin_negbin")
  expect_equal(get_hmc_model_type(ratiod_zinegbin(denom_family = "gamma")),
               "negbin_gamma")
  expect_equal(get_hmc_model_type(ratiod_hurdle_negbin(denom_family = "gamma")),
               "negbin_gamma")
  expect_equal(get_hmc_model_type(ratiod_hurdle_pois()), "poisson_gamma")
  expect_error(ratiod_zipois(denom_family = "negbin"), "must be one of")
  expect_error(ratiod_zinegbin(denom_family = "fixed"), "must be one of")
})

test_that("ratiod_lognormal creates valid family", {
  fam <- ratiod_lognormal()

  expect_s3_class(fam, "ratiod_family")
  expect_equal(fam$name, "lognormal_lognormal")
  expect_equal(fam$numerator$distribution, "lognormal")
  expect_equal(fam$denominator$distribution, "lognormal")
})

test_that("family print methods work", {
  expect_output(print(ratiod_beta_binomial()), "beta_binomial")
  expect_output(print(ratiod_gamma_gamma()), "gamma_gamma")
  expect_output(print(ratiod_lognormal()), "lognormal")
})
