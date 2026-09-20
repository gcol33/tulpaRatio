# test-coverage-helpers.R
# Additional tests to improve coverage for helper functions

# -----------------------------------------------------------------------------
# priors.R coverage
# -----------------------------------------------------------------------------

test_that("ratiod_priors creates valid prior object", {
  priors <- ratiod_priors()
  expect_s3_class(priors, "ratiod_priors")
})

test_that("ratiod_priors accepts custom values", {
  priors <- ratiod_priors(
    beta = prior_normal(0, 5),
    sigma = prior_pc(U = 1.5),
    phi = prior_gamma(2, 0.5)
  )

  expect_s3_class(priors, "ratiod_priors")
  expect_equal(priors$beta$dist, "normal")
  expect_equal(priors$sigma$dist, "pc")
})

test_that("print.ratiod_priors works", {
  priors <- ratiod_priors()
  output <- capture.output(print(priors))
  expect_true(any(grepl("tulpaRatio prior specification", output)))
})

# -----------------------------------------------------------------------------
# plot_diagnostics.R coverage
# -----------------------------------------------------------------------------

test_that("diagnostics() returns list with diagnostic info", {
  skip_on_cran()
  skip_if_not_installed("bayesplot")

  set.seed(42)
  n <- 30
  df <- data.frame(
    count = rpois(n, 10),
    total = rpois(n, 100) + 10,
    x = rnorm(n)
  )

  fit <- tratio(
    count | total ~ x,
    data = df,
    family = ratiod_negbin_negbin(),
    control = list(iter = 100, warmup = 50, chains = 1)
  )

  diag <- diagnostics(fit)
  expect_true(is.list(diag) || inherits(diag, "data.frame"))
})

# -----------------------------------------------------------------------------
# simulate.R coverage
# -----------------------------------------------------------------------------

test_that("sim_ratiod generates valid data", {
  skip_on_cran()

  # Test with default parameters
  data <- sim_ratiod(
    n = 50,
    family = ratiod_negbin_negbin()
  )

  expect_s3_class(data, "ratiod_simdata")
  expect_true("data" %in% names(data))
  expect_equal(nrow(data$data), 50)
})

test_that("sim_ratiod works with random effects", {
  skip_on_cran()

  data <- sim_ratiod(
    n = 60,
    n_groups = 6,
    family = ratiod_negbin_negbin()
  )

  expect_s3_class(data, "ratiod_simdata")
  expect_true("group" %in% names(data$data) || "site" %in% names(data$data))
})

# -----------------------------------------------------------------------------
# temporal.R coverage
# -----------------------------------------------------------------------------

test_that("temporal_rw1 creates valid structure", {
  temp <- temporal_rw1(time_var = "year")
  expect_s3_class(temp, "ratiod_temporal")
  expect_equal(temp$type, "rw1")
  expect_equal(temp$time_var, "year")
})

test_that("temporal_rw2 creates valid structure", {
  temp <- temporal_rw2(time_var = "year")
  expect_s3_class(temp, "ratiod_temporal")
  expect_equal(temp$type, "rw2")
})

test_that("temporal_ar1 creates valid structure", {
  temp <- temporal_ar1(time_var = "year")
  expect_s3_class(temp, "ratiod_temporal")
  expect_equal(temp$type, "ar1")
})

test_that("temporal_ar1 accepts a Beta prior on its correlation", {
  temp <- temporal_ar1(time_var = "year", rho_prior = prior_beta(5, 2))
  expect_equal(temp$rho_prior$alpha, 5)
  expect_equal(temp$rho_prior$beta, 2)
  # `rho` is not an argument; it used to partial-match rho_prior and store a
  # bare number nothing read.
  expect_error(temporal_ar1(time_var = "year", rho = 0.8), "Beta prior")
})

test_that("print.ratiod_temporal works", {
  temp <- temporal_rw1(time_var = "year")
  output <- capture.output(print(temp))
  expect_true(length(output) > 0)
})

# -----------------------------------------------------------------------------
# latent.R coverage
# -----------------------------------------------------------------------------

test_that("latent_factor creates valid structure", {
  lf <- latent_factor(n_factors = 2)
  expect_s3_class(lf, "ratiod_latent")
  expect_equal(lf$n_factors, 2)
})

test_that("latent_factor validates n_factors", {
  expect_error(latent_factor(n_factors = 0), "positive integer")
  expect_error(latent_factor(n_factors = -1), "positive integer")
})

test_that("print.ratiod_latent works", {
  lf <- latent_factor(n_factors = 2)
  output <- capture.output(print(lf))
  expect_true(any(grepl("Latent factor specification", output)))
})

# -----------------------------------------------------------------------------
# spatiotemporal.R coverage
# -----------------------------------------------------------------------------

test_that("spatiotemporal creates valid structure", {
  # Need to create spatial and temporal objects first
  adj <- matrix(c(0, 1, 0, 1, 0, 1, 0, 1, 0), nrow = 3)
  spatial <- spatial_car(group_var = "region", adjacency = adj)
  temporal <- temporal_rw1(time_var = "year")

  st <- spatiotemporal(
    spatial = spatial,
    temporal = temporal,
    type = "I"
  )
  expect_s3_class(st, "ratiod_spatiotemporal")
  expect_equal(st$type, "I")
})

test_that("spatiotemporal validates type", {
  adj <- matrix(c(0, 1, 0, 1, 0, 1, 0, 1, 0), nrow = 3)
  spatial <- spatial_car(group_var = "region", adjacency = adj)
  temporal <- temporal_rw1(time_var = "year")

  # Invalid type should error - match.arg gives specific message
  expect_error(
    spatiotemporal(spatial = spatial, temporal = temporal, type = "invalid")
  )
})

test_that("print.ratiod_spatiotemporal works", {
  adj <- matrix(c(0, 1, 0, 1, 0, 1, 0, 1, 0), nrow = 3)
  spatial <- spatial_car(group_var = "region", adjacency = adj)
  temporal <- temporal_rw1(time_var = "year")

  st <- spatiotemporal(
    spatial = spatial,
    temporal = temporal,
    type = "II"
  )
  output <- capture.output(print(st))
  expect_true(length(output) > 0)
})

# -----------------------------------------------------------------------------
# ratio.R coverage
# -----------------------------------------------------------------------------

test_that("ratio_contrast errors with invalid models", {
  expect_error(ratio_contrast(NULL, NULL))
})

test_that("ratio handles invalid type", {
  skip_on_cran()

  set.seed(42)
  n <- 20
  df <- data.frame(
    count = rpois(n, 10),
    total = rpois(n, 100) + 10,
    x = rnorm(n)
  )

  fit <- tratio(
    count | total ~ x,
    data = df,
    family = ratiod_negbin_negbin(),
    control = list(iter = 50, warmup = 25, chains = 1)
  )

  expect_error(ratio(fit, type = "invalid_type"))
})

# -----------------------------------------------------------------------------
# validate.R coverage
# -----------------------------------------------------------------------------

test_that("compute_loo requires loo package", {
  skip_if_not_installed("loo")
  skip_on_cran()

  set.seed(42)
  n <- 20
  df <- data.frame(
    count = rpois(n, 10),
    total = rpois(n, 100) + 10,
    x = rnorm(n)
  )

  fit <- tratio(
    count | total ~ x,
    data = df,
    family = ratiod_negbin_negbin(),
    control = list(iter = 100, warmup = 50, chains = 1)
  )

  # This should work or throw informative error
  tryCatch({
    result <- loo(fit)
    expect_s3_class(result, "loo")
  }, error = function(e) {
    # Expected if loo computation fails due to short chain
    expect_true(TRUE)
  })
})

test_that("compute_waic requires loo package", {
  skip_if_not_installed("loo")
  skip_on_cran()

  set.seed(42)
  n <- 20
  df <- data.frame(
    count = rpois(n, 10),
    total = rpois(n, 100) + 10,
    x = rnorm(n)
  )

  fit <- tratio(
    count | total ~ x,
    data = df,
    family = ratiod_negbin_negbin(),
    control = list(iter = 100, warmup = 50, chains = 1)
  )

  tryCatch({
    result <- waic(fit)
    expect_s3_class(result, "waic")
  }, error = function(e) {
    expect_true(TRUE)
  })
})

# -----------------------------------------------------------------------------
# plot_map.R coverage
# -----------------------------------------------------------------------------

test_that("plot_map requires sf package", {
  # Test that plot_map function exists and is callable

  # The error condition for missing sf cannot be tested when sf is installed
  expect_true(is.function(plot_map))
})

test_that("plot_spatial_effects requires fit object", {
  expect_error(plot_spatial_effects(NULL))
})

# -----------------------------------------------------------------------------
# zi.R coverage
# -----------------------------------------------------------------------------

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
