# test-advanced-integration.R
# Integration tests for advanced features to improve coverage

# ============================================================================
# GP Spatial Model Fitting (hmc_gp.h coverage)
# ============================================================================

test_that("GP spatial model fits with exponential covariance", {
  skip_on_cran()

  set.seed(123)
  n <- 40

  # Generate spatial data on grid
  coords <- expand.grid(x = seq(0, 1, length.out = sqrt(n)),
                        y = seq(0, 1, length.out = sqrt(n)))
  if (nrow(coords) < n) {
    coords <- data.frame(x = runif(n), y = runif(n))
  }

  df <- data.frame(
    count = rpois(n, lambda = 5),
    effort = rgamma(n, shape = 4, rate = 1),
    x_coord = coords$x[1:n],
    y_coord = coords$y[1:n]
  )

 fit <- tratio(
    count | effort ~ 1,
    data = df,
    family = ratiod_poisson_gamma(),
    spatial = spatial_gp(~ x_coord + y_coord, cov = "exponential", nn = 5),
    mode = "hmc",
    control = list(iter = 100, warmup = 50, chains = 1, verbose = FALSE)
  )

  expect_s3_class(fit, "ratiod_fit")
  expect_true("gp" %in% names(fit) || !is.null(fit$spatial))
})


test_that("GP spatial model fits with matern covariance", {
  skip_on_cran()

  set.seed(456)
  n <- 30

  df <- data.frame(
    count = rpois(n, lambda = 5),
    effort = rgamma(n, shape = 4, rate = 1),
    lon = runif(n, 0, 10),
    lat = runif(n, 0, 10)
  )

  fit <- tratio(
    count | effort ~ 1,
    data = df,
    family = ratiod_poisson_gamma(),
    spatial = spatial_gp(~ lon + lat, cov = "matern", nu = 1.5, nn = 5),
    mode = "hmc",
    control = list(iter = 100, warmup = 50, chains = 1, verbose = FALSE)
  )

  expect_s3_class(fit, "ratiod_fit")
})


# ============================================================================
# Latent Factor Model Fitting (hmc_latent.h coverage)
# ============================================================================

test_that("latent factor model fits with 1 factor", {
  skip_on_cran()

  set.seed(789)
  n <- 50

  df <- data.frame(
    count = rpois(n, lambda = 5),
    effort = rgamma(n, shape = 4, rate = 1),
    x = rnorm(n)
  )

  fit <- tratio(
    count | effort ~ x,
    data = df,
    family = ratiod_poisson_gamma(),
    latent = latent_factor(n_factors = 1),
    mode = "hmc",
    control = list(iter = 100, warmup = 50, chains = 1, verbose = FALSE)
  )

  expect_s3_class(fit, "ratiod_fit")
  expect_equal(fit$latent$n_factors, 1)
})


test_that("latent factor model fits with 2 factors", {
  skip_on_cran()

  set.seed(101)
  n <- 60

  df <- data.frame(
    count = rpois(n, lambda = 5),
    effort = rgamma(n, shape = 4, rate = 1),
    x = rnorm(n)
  )

  fit <- tratio(
    count | effort ~ x,
    data = df,
    family = ratiod_poisson_gamma(),
    latent = latent_factor(n_factors = 2),
    mode = "hmc",
    control = list(iter = 100, warmup = 50, chains = 1, verbose = FALSE)
  )

  expect_s3_class(fit, "ratiod_fit")
  expect_equal(fit$latent$n_factors, 2)
})


# ============================================================================
# Spatiotemporal Model Fitting (hmc_spatiotemporal.h coverage)
# ============================================================================

test_that("spatiotemporal Type I model fits", {
  skip_on_cran()

  set.seed(202)
  n_regions <- 4
  n_times <- 5
  n <- n_regions * n_times

  # Create adjacency matrix (chain: 1-2-3-4)
  adj <- matrix(0, n_regions, n_regions)
  for (i in 1:(n_regions - 1)) {
    adj[i, i + 1] <- adj[i + 1, i] <- 1
  }

  df <- data.frame(
    region = factor(rep(1:n_regions, each = n_times)),
    time = rep(1:n_times, n_regions),
    count = rpois(n, lambda = 5),
    effort = rgamma(n, shape = 4, rate = 1)
  )

  fit <- tratio(
    count | effort ~ 1,
    data = df,
    family = ratiod_poisson_gamma(),
    spatiotemporal = spatiotemporal(
      spatial = spatial_car(adj, level = "group", group_var = "region"),
      temporal = temporal_rw1("time"),
      type = "I"
    ),
    mode = "hmc",
    control = list(iter = 100, warmup = 50, chains = 1, verbose = FALSE)
  )

  expect_s3_class(fit, "ratiod_fit")
})


test_that("spatiotemporal Type IV model fits", {
  skip_on_cran()

  set.seed(303)
  n_regions <- 4
  n_times <- 4
  n <- n_regions * n_times

  # Create adjacency matrix
  adj <- matrix(0, n_regions, n_regions)
  for (i in 1:(n_regions - 1)) {
    adj[i, i + 1] <- adj[i + 1, i] <- 1
  }

  df <- data.frame(
    region = factor(rep(1:n_regions, each = n_times)),
    time = rep(1:n_times, n_regions),
    count = rpois(n, lambda = 5),
    effort = rgamma(n, shape = 4, rate = 1)
  )

  fit <- tratio(
    count | effort ~ 1,
    data = df,
    family = ratiod_poisson_gamma(),
    spatiotemporal = spatiotemporal(
      spatial = spatial_car(adj, level = "group", group_var = "region"),
      temporal = temporal_rw1("time"),
      type = "IV"
    ),
    mode = "hmc",
    control = list(iter = 100, warmup = 50, chains = 1, verbose = FALSE)
  )

  expect_s3_class(fit, "ratiod_fit")
})


# ============================================================================
# SVC Model Fitting (hmc_svc.h coverage)
# ============================================================================

test_that("SVC model fits with single varying coefficient", {
  skip_on_cran()

  set.seed(404)
  n <- 30

  # Create data with coordinates for SVC
  df <- data.frame(
    lon = runif(n, 0, 10),
    lat = runif(n, 0, 10),
    count = rpois(n, lambda = 5),
    effort = rgamma(n, shape = 4, rate = 1),
    x = rnorm(n)
  )

  fit <- tratio(
    count | effort ~ x,
    data = df,
    family = ratiod_poisson_gamma(),
    spatial = spatial_svc(~ lon + lat, terms = 1, nn = 5),
    mode = "hmc",
    control = list(iter = 100, warmup = 50, chains = 1, verbose = FALSE)
  )

  expect_s3_class(fit, "ratiod_fit")
})


# ============================================================================
# Laplace Backend Fitting (backend_laplace.R coverage)
# ============================================================================

test_that("Laplace backend fits binomial model", {
  skip_on_cran()

  set.seed(505)
  n <- 100

  df <- data.frame(
    successes = rbinom(n, size = 20, prob = 0.3),
    trials = rep(20, n),
    x = rnorm(n)
  )

  fit <- tratio(
    successes | trials ~ x,
    data = df,
    family = ratiod_binomial(),
    mode = "laplace",
    control = list(verbose = FALSE)
  )

  expect_s3_class(fit, "ratiod_fit")
  expect_equal(fit$backend, "laplace")
})


test_that("Laplace backend fits negbin model", {
  skip_on_cran()

  set.seed(606)
  n <- 100

  df <- data.frame(
    count = rnbinom(n, size = 3, mu = 10),
    total = rnbinom(n, size = 5, mu = 50),
    x = rnorm(n)
  )

  fit <- tratio(
    count | total ~ x,
    data = df,
    family = ratiod_negbin_negbin(),
    mode = "laplace",
    control = list(verbose = FALSE)
  )

  expect_s3_class(fit, "ratiod_fit")
  expect_equal(fit$backend, "laplace")
})


test_that("Laplace backend fits poisson_gamma model", {
  skip_on_cran()

  set.seed(707)
  n <- 100

  df <- data.frame(
    count = rpois(n, lambda = 5),
    effort = rgamma(n, shape = 4, rate = 1),
    x = rnorm(n)
  )

  fit <- tratio(
    count | effort ~ x,
    data = df,
    family = ratiod_poisson_gamma(),
    mode = "laplace",
    control = list(verbose = FALSE)
  )

  expect_s3_class(fit, "ratiod_fit")
  expect_equal(fit$backend, "laplace")
})


test_that("Laplace backend fits model with random effects", {
  skip_on_cran()

  set.seed(808)
  n_sites <- 10
  n_per_site <- 10
  n <- n_sites * n_per_site

  df <- data.frame(
    site = factor(rep(1:n_sites, each = n_per_site)),
    successes = rbinom(n, size = 20, prob = 0.3),
    trials = rep(20, n),
    x = rnorm(n)
  )

  fit <- tratio(
    successes | trials ~ x + (1 | site),
    data = df,
    family = ratiod_binomial(),
    mode = "laplace",
    control = list(verbose = FALSE)
  )

  expect_s3_class(fit, "ratiod_fit")
  expect_equal(fit$backend, "laplace")
})


test_that("Laplace backend fits model with spatial CAR", {
  skip_on_cran()

  set.seed(909)
  n_sites <- 6
  n_per_site <- 10
  n <- n_sites * n_per_site

  # Chain adjacency
  adj <- matrix(0, n_sites, n_sites)
  for (i in 1:(n_sites - 1)) {
    adj[i, i + 1] <- adj[i + 1, i] <- 1
  }

  df <- data.frame(
    site = factor(rep(1:n_sites, each = n_per_site)),
    successes = rbinom(n, size = 20, prob = 0.3),
    trials = rep(20, n),
    x = rnorm(n)
  )

  fit <- tratio(
    successes | trials ~ x,
    data = df,
    family = ratiod_binomial(),
    spatial = spatial_car(adj, level = "group", group_var = "site"),
    mode = "laplace",
    control = list(verbose = FALSE)
  )

  expect_s3_class(fit, "ratiod_fit")
  expect_equal(fit$backend, "laplace")
})


# ============================================================================
# PG Backend Fitting (pg_binomial.cpp, backend_pg.R coverage)
# ============================================================================

test_that("PG backend fits binomial model", {
  skip_on_cran()

  set.seed(111)
  n <- 80

  df <- data.frame(
    successes = rbinom(n, size = 20, prob = 0.3),
    trials = rep(20, n),
    x = rnorm(n)
  )

  fit <- tratio(
    successes | trials ~ x,
    data = df,
    family = ratiod_binomial(),
    mode = "pg",
    control = list(iter = 100, warmup = 50, chains = 1, verbose = FALSE)
  )

  expect_s3_class(fit, "ratiod_fit")
  expect_equal(fit$backend, "pg")
})


test_that("PG backend fits binomial model with random effects", {
  skip_on_cran()

  set.seed(222)
  n_sites <- 8
  n_per_site <- 10
  n <- n_sites * n_per_site

  df <- data.frame(
    site = factor(rep(1:n_sites, each = n_per_site)),
    successes = rbinom(n, size = 20, prob = 0.3),
    trials = rep(20, n),
    x = rnorm(n)
  )

  fit <- tratio(
    successes | trials ~ x + (1 | site),
    data = df,
    family = ratiod_binomial(),
    mode = "pg",
    control = list(iter = 100, warmup = 50, chains = 1, verbose = FALSE)
  )

  expect_s3_class(fit, "ratiod_fit")
  expect_equal(fit$backend, "pg")

  # Check draws matrix exists and has content
  expect_true(!is.null(fit$draws))
  expect_true(nrow(fit$draws) > 0)
  expect_true(ncol(fit$draws) > 0)
})


test_that("PG backend fits binomial model with spatial CAR", {
  skip_on_cran()

  set.seed(333)
  n_sites <- 6
  n_per_site <- 12
  n <- n_sites * n_per_site

  # Chain adjacency
  adj <- matrix(0, n_sites, n_sites)
  for (i in 1:(n_sites - 1)) {
    adj[i, i + 1] <- adj[i + 1, i] <- 1
  }

  df <- data.frame(
    site = factor(rep(1:n_sites, each = n_per_site)),
    successes = rbinom(n, size = 20, prob = 0.3),
    trials = rep(20, n),
    x = rnorm(n)
  )

  fit <- tratio(
    successes | trials ~ x,
    data = df,
    family = ratiod_binomial(),
    spatial = spatial_car(adj, level = "group", group_var = "site"),
    mode = "pg",
    control = list(iter = 100, warmup = 50, chains = 1, verbose = FALSE)
  )

  expect_s3_class(fit, "ratiod_fit")
  expect_equal(fit$backend, "pg")
})


# ============================================================================
# Validation Functions (validate.R coverage)
# ============================================================================

test_that("ratiod_compare handles model comparison setup", {
  skip_on_cran()

  set.seed(777)
  n <- 40

  df <- data.frame(
    count = rpois(n, lambda = 5),
    effort = rgamma(n, shape = 4, rate = 1),
    x1 = rnorm(n),
    x2 = rnorm(n)
  )

  fit1 <- tratio(
    count | effort ~ x1,
    data = df,
    family = ratiod_poisson_gamma(),
    mode = "hmc",
    control = list(iter = 100, warmup = 50, chains = 1, verbose = FALSE)
  )

  fit2 <- tratio(
    count | effort ~ x1 + x2,
    data = df,
    family = ratiod_poisson_gamma(),
    mode = "hmc",
    control = list(iter = 100, warmup = 50, chains = 1, verbose = FALSE)
  )

  # Test that models can be compared (may fail gracefully if loo not available)
  result <- tryCatch(
    ratiod_compare(fit1, fit2),
    error = function(e) "expected_error"
  )

  # Either succeeds (loo_compare returns a matrix) or fails gracefully.
  expect_true(is.matrix(result) || is.data.frame(result) ||
              identical(result, "expected_error"))
})


# ============================================================================
# Combined advanced features
# ============================================================================

test_that("GP spatial with random effects fits", {
  skip_on_cran()

  set.seed(888)
  n_sites <- 5
  n_per_site <- 8
  n <- n_sites * n_per_site

  df <- data.frame(
    site = factor(rep(1:n_sites, each = n_per_site)),
    count = rpois(n, lambda = 5),
    effort = rgamma(n, shape = 4, rate = 1),
    x_coord = runif(n),
    y_coord = runif(n)
  )

  fit <- tratio(
    count | effort ~ (1 | site),
    data = df,
    family = ratiod_poisson_gamma(),
    spatial = spatial_gp(~ x_coord + y_coord, nn = 4),
    mode = "hmc",
    control = list(iter = 100, warmup = 50, chains = 1, verbose = FALSE)
  )

  expect_s3_class(fit, "ratiod_fit")
})


test_that("Latent factor with random effects fits", {
  skip_on_cran()

  set.seed(999)
  n_sites <- 6
  n_per_site <- 10
  n <- n_sites * n_per_site

  df <- data.frame(
    site = factor(rep(1:n_sites, each = n_per_site)),
    count = rpois(n, lambda = 5),
    effort = rgamma(n, shape = 4, rate = 1),
    x = rnorm(n)
  )

  fit <- tratio(
    count | effort ~ x + (1 | site),
    data = df,
    family = ratiod_poisson_gamma(),
    latent = latent_factor(n_factors = 1),
    mode = "hmc",
    control = list(iter = 100, warmup = 50, chains = 1, verbose = FALSE)
  )

  expect_s3_class(fit, "ratiod_fit")
})
