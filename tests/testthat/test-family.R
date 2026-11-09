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

test_that("a family accepts only the links its likelihood implements", {
  # Probabilities take logit, probit or cloglog; gamma means log, identity or
  # inverse; count means and the log-normal location the log link only.
  expect_equal(ratiod_binomial(link = "probit")$numerator$link, "probit")
  expect_equal(ratiod_beta_binomial(link = "cloglog")$numerator$link, "cloglog")
  expect_equal(ratiod_gamma_gamma(link_num = "identity")$numerator$link, "identity")
  expect_equal(ratiod_zibinomial(link_zi = "probit")$numerator$link_zi, "probit")
  expect_equal(ratiod_hurdle_pois(link_hurdle = "cloglog")$numerator$link_hurdle,
               "cloglog")
  expect_error(ratiod_binomial(link = "log"), "not supported")
  expect_error(ratiod_lognormal(link_num = "identity"), "not supported")
  expect_error(ratiod_negbin_negbin(link_num = "identity"), "not supported")
  expect_error(ratiod_poisson_gamma(link_num = "inverse"), "not supported")
  expect_error(ratiod_zibinomial(link_zi = "inverse"), "not supported")
})

test_that("inflated count families fit the denominators they name", {
  expect_equal(get_hmc_model_type(ratiod_zipois()), "poisson_gamma")
  expect_equal(get_hmc_model_type(ratiod_zinegbin()), "negbin_negbin")
  expect_equal(get_hmc_model_type(ratiod_zinegbin(denom_family = "gamma")),
               "negbin_gamma")
  expect_equal(get_hmc_model_type(ratiod_hurdle_negbin(denom_family = "gamma")),
               "negbin_gamma")
  expect_equal(get_hmc_model_type(ratiod_hurdle_pois()), "poisson_gamma")
  expect_equal(get_hmc_model_type(ratiod_zipois(denom_family = "negbin")),
               "poisson_negbin")
  expect_equal(get_hmc_model_type(ratiod_zipois(denom_family = "fixed")),
               "poisson_fixed")
  expect_equal(get_hmc_model_type(ratiod_zinegbin(denom_family = "fixed")),
               "negbin_fixed")
  expect_equal(get_hmc_model_type(ratiod_hurdle_negbin(denom_family = "fixed")),
               "negbin_fixed")
  expect_equal(get_hmc_model_type(ratiod_lognormal(denom_fixed = TRUE)),
               "lognormal_fixed")
  expect_error(ratiod_zipois(denom_family = "binomial"), "must be one of")
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

test_that("all main families create valid objects", {
  fam1 <- ratiod_negbin_negbin()
  expect_s3_class(fam1, "ratiod_family")

  fam2 <- ratiod_binomial()
  expect_s3_class(fam2, "ratiod_family")

  fam3 <- ratiod_poisson_gamma()
  expect_s3_class(fam3, "ratiod_family")
})

test_that("print.ratiod_family works", {
  fam <- ratiod_negbin_negbin()
  output <- capture.output(print(fam))
  expect_true(length(output) > 0)
})

test_that("zero inflation families are created correctly", {
  zi <- ratiod_zipois()
  expect_s3_class(zi, "ratiod_family")
  expect_true(zi$zero_inflated)

  zi <- ratiod_zinegbin()
  expect_s3_class(zi, "ratiod_family")
  expect_true(zi$zero_inflated)

  hurdle <- ratiod_hurdle_pois()
  expect_s3_class(hurdle, "ratiod_family")
  expect_equal(hurdle$zi_type, "hurdle")
})

test_that("print.ratiod_family shows ZI info", {
  zi <- ratiod_zipois()
  output <- capture.output(print(zi))
  expect_true(any(grepl("zero", output, ignore.case = TRUE)) ||
              any(grepl("Zero", output)))
})

test_that("additional ZI families work correctly", {
  # Test binomial variants
  zi_bin <- ratiod_zibinomial()
  expect_s3_class(zi_bin, "ratiod_family")
  expect_true(zi_bin$zero_inflated)

  hurdle_bin <- ratiod_hurdle_binomial()
  expect_s3_class(hurdle_bin, "ratiod_family")
  expect_equal(hurdle_bin$zi_type, "hurdle")

  hurdle_nb <- ratiod_hurdle_negbin()
  expect_s3_class(hurdle_nb, "ratiod_family")
  expect_equal(hurdle_nb$zi_type, "hurdle")
})

test_that("can_use_pg_backend returns TRUE for binomial family", {
  fam <- ratiod_binomial()
  expect_true(tulpaRatio:::can_use_pg_backend(fam))
})

test_that("can_use_pg_backend returns FALSE for negbin family", {
  fam <- ratiod_negbin_negbin()
  expect_false(tulpaRatio:::can_use_pg_backend(fam))
})

test_that("mode = 'pg' refuses a negative binomial family", {
  set.seed(1)
  d <- data.frame(y_num = rpois(20, 5), y_denom = rpois(20, 3) + 1,
                  x1 = rnorm(20))
  expect_error(
    tratio(y_num | y_denom ~ x1, data = d, family = ratiod_negbin_negbin(),
           mode = "pg", control = list(iter = 20, chains = 1, verbose = FALSE)),
    "PG backend fits ratiod_binomial\\(\\) only"
  )
})

test_that("can_use_pg_backend returns FALSE for poisson_gamma family", {
  fam <- ratiod_poisson_gamma()
  expect_false(tulpaRatio:::can_use_pg_backend(fam))
})
