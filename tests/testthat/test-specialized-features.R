# Minimal integration tests for specialized C++ features
# Each test uses small data and few iterations to run under 1 second

# ---------------------------------------------------------------------------
# Spatial GP tests (exercises hmc_gp.h)
# ---------------------------------------------------------------------------

test_that("spatial_gp fits with HMC backend", {
  skip_on_cran()

  set.seed(123)
  n <- 15

  df <- data.frame(
    count = rpois(n, lambda = 8),
    effort = rgamma(n, shape = 4, rate = 1),
    lon = runif(n, 0, 10),
    lat = runif(n, 0, 10)
  )

  # Minimal GP fit - coords as formula
  fit <- tratio(
    count | effort ~ 1,
    data = df,
    family = ratiod_poisson_gamma(),
    spatial = spatial_gp(coords = ~ lon + lat, cov = "exponential", nn = 5),
    mode = "hmc",
    control = list(iter = 50, warmup = 25, chains = 1, verbose = FALSE)
  )

  expect_s3_class(fit, "ratiod_fit")
  expect_equal(fit$backend, "hmc")
})

# ---------------------------------------------------------------------------
# Spatiotemporal tests (exercises hmc_spatiotemporal.h)
# ---------------------------------------------------------------------------

test_that("spatiotemporal Type I fits with HMC backend", {
  skip_on_cran()

  set.seed(456)
  n_sites <- 4
  n_times <- 3
  n <- n_sites * n_times

  # Simple adjacency (chain)
  adj <- matrix(0, n_sites, n_sites)
  for (i in 1:(n_sites - 1)) adj[i, i + 1] <- adj[i + 1, i] <- 1

  df <- data.frame(
    count = rpois(n, lambda = 10),
    effort = rgamma(n, shape = 5, rate = 1),
    site = factor(rep(1:n_sites, times = n_times)),
    time = rep(1:n_times, each = n_sites)
  )

  st <- spatiotemporal(
    spatial = spatial_car(adj, spatial_level = "group", group_var = "site"),
    temporal = temporal_rw1("time"),
    type = "I"
  )

  fit <- tratio(
    count | effort ~ 1,
    data = df,
    family = ratiod_poisson_gamma(),
    spatiotemporal = st,
    mode = "hmc",
    control = list(iter = 50, warmup = 25, chains = 1, verbose = FALSE)
  )

  expect_s3_class(fit, "ratiod_fit")
})

test_that("spatiotemporal Type IV fits with HMC backend", {
  skip_on_cran()

  set.seed(789)
  n_sites <- 3
  n_times <- 4
  n <- n_sites * n_times

  adj <- matrix(0, n_sites, n_sites)
  adj[1, 2] <- adj[2, 1] <- 1
  adj[2, 3] <- adj[3, 2] <- 1

  df <- data.frame(
    count = rpois(n, lambda = 8),
    effort = rgamma(n, shape = 4, rate = 1),
    site = factor(rep(1:n_sites, times = n_times)),
    time = rep(1:n_times, each = n_sites)
  )

  st <- spatiotemporal(
    spatial = spatial_car(adj, spatial_level = "group", group_var = "site"),
    temporal = temporal_rw1("time"),
    type = "IV"
  )

  fit <- tratio(
    count | effort ~ 1,
    data = df,
    family = ratiod_poisson_gamma(),
    spatiotemporal = st,
    mode = "hmc",
    control = list(iter = 50, warmup = 25, chains = 1, verbose = FALSE)
  )

  expect_s3_class(fit, "ratiod_fit")
})

# ---------------------------------------------------------------------------
# SVC tests (exercises hmc_svc.h)
# ---------------------------------------------------------------------------

test_that("spatial_svc fits with HMC backend", {
  skip_on_cran()

  set.seed(111)
  n <- 15

  df <- data.frame(
    count = rpois(n, lambda = 10),
    effort = rgamma(n, shape = 5, rate = 1),
    covar = rnorm(n),
    lon = runif(n, 0, 10),
    lat = runif(n, 0, 10)
  )

  svc <- spatial_svc(
    coords = ~ lon + lat,
    terms = 1,  # Number of SVC terms
    cov = "exponential",
    nn = 5
  )

  fit <- tratio(
    count | effort ~ covar,
    data = df,
    family = ratiod_poisson_gamma(),
    spatial = svc,
    mode = "hmc",
    control = list(iter = 50, warmup = 25, chains = 1, verbose = FALSE)
  )

  expect_s3_class(fit, "ratiod_fit")
})

# ---------------------------------------------------------------------------
# Latent factor tests (exercises hmc_latent.h)
# ---------------------------------------------------------------------------

test_that("latent_factor fits with HMC backend", {
  skip_on_cran()

  set.seed(222)
  n <- 20

  df <- data.frame(
    count = rpois(n, lambda = 10),
    effort = rgamma(n, shape = 5, rate = 1),
    x = rnorm(n)
  )

  fit <- tratio(
    count | effort ~ x,
    data = df,
    family = ratiod_poisson_gamma(),
    latent = latent_factor(n_factors = 1),
    mode = "hmc",
    control = list(iter = 50, warmup = 25, chains = 1, verbose = FALSE)
  )

  expect_s3_class(fit, "ratiod_fit")
})

test_that("latent_factor with 2 factors fits", {
  skip_on_cran()

  set.seed(333)
  n <- 25

  df <- data.frame(
    count = rpois(n, lambda = 10),
    effort = rgamma(n, shape = 5, rate = 1),
    x = rnorm(n)
  )

  fit <- tratio(
    count | effort ~ x,
    data = df,
    family = ratiod_poisson_gamma(),
    latent = latent_factor(n_factors = 2),
    mode = "hmc",
    control = list(iter = 50, warmup = 25, chains = 1, verbose = FALSE)
  )

  expect_s3_class(fit, "ratiod_fit")
})

# ---------------------------------------------------------------------------
# Proper CAR tests (exercises additional spatial paths)
# ---------------------------------------------------------------------------

test_that("proper CAR fits with HMC backend", {
  skip_on_cran()

  set.seed(444)
  n_sites <- 4
  n_per_site <- 4
  n <- n_sites * n_per_site

  adj <- matrix(0, n_sites, n_sites)
  for (i in 1:(n_sites - 1)) adj[i, i + 1] <- adj[i + 1, i] <- 1

  df <- data.frame(
    count = rpois(n, lambda = 10),
    effort = rgamma(n, shape = 5, rate = 1),
    site = factor(rep(1:n_sites, each = n_per_site))
  )

  fit <- tratio(
    count | effort ~ 1 + (1 | site),
    data = df,
    family = ratiod_poisson_gamma(),
    spatial = spatial_car(adj, spatial_level = "group", group_var = "site", proper = TRUE),
    mode = "hmc",
    control = list(iter = 50, warmup = 25, chains = 1, verbose = FALSE)
  )

  expect_s3_class(fit, "ratiod_fit")
})

# ---------------------------------------------------------------------------
# BYM2 spatial tests
# ---------------------------------------------------------------------------

test_that("BYM2 spatial fits with HMC backend", {
  skip_on_cran()

  set.seed(555)
  n_sites <- 4
  n_per_site <- 3
  n <- n_sites * n_per_site

  adj <- matrix(0, n_sites, n_sites)
  for (i in 1:(n_sites - 1)) adj[i, i + 1] <- adj[i + 1, i] <- 1

  df <- data.frame(
    count = rpois(n, lambda = 10),
    effort = rgamma(n, shape = 5, rate = 1),
    site = factor(rep(1:n_sites, each = n_per_site))
  )

  fit <- tratio(
    count | effort ~ 1 + (1 | site),
    data = df,
    family = ratiod_poisson_gamma(),
    spatial = spatial_bym2(adj, spatial_level = "group", group_var = "site"),
    mode = "hmc",
    control = list(iter = 50, warmup = 25, chains = 1, verbose = FALSE)
  )

  expect_s3_class(fit, "ratiod_fit")
})

# ---------------------------------------------------------------------------
# Temporal RW2 tests
# ---------------------------------------------------------------------------

test_that("temporal RW2 fits with HMC backend", {
  skip_on_cran()

  set.seed(666)
  n_times <- 8
  n_per_time <- 3
  n <- n_times * n_per_time

  df <- data.frame(
    count = rpois(n, lambda = 10),
    effort = rgamma(n, shape = 5, rate = 1),
    time = rep(1:n_times, each = n_per_time)
  )

  fit <- tratio(
    count | effort ~ 1,
    data = df,
    family = ratiod_poisson_gamma(),
    temporal = temporal_rw2("time"),
    mode = "hmc",
    control = list(iter = 50, warmup = 25, chains = 1, verbose = FALSE)
  )

  expect_s3_class(fit, "ratiod_fit")
})

# ---------------------------------------------------------------------------
# AR1 temporal tests
# ---------------------------------------------------------------------------

test_that("temporal AR1 fits with HMC backend", {
  skip_on_cran()

  set.seed(777)
  n_times <- 6
  n_per_time <- 4
  n <- n_times * n_per_time

  df <- data.frame(
    count = rpois(n, lambda = 10),
    effort = rgamma(n, shape = 5, rate = 1),
    time = rep(1:n_times, each = n_per_time)
  )

  fit <- tratio(
    count | effort ~ 1,
    data = df,
    family = ratiod_poisson_gamma(),
    temporal = temporal_ar1("time"),
    mode = "hmc",
    control = list(iter = 50, warmup = 25, chains = 1, verbose = FALSE)
  )

  expect_s3_class(fit, "ratiod_fit")
})

# ---------------------------------------------------------------------------
# Cyclic temporal tests
# ---------------------------------------------------------------------------

test_that("cyclic temporal RW1 fits with HMC backend", {
  skip_on_cran()

  set.seed(888)
  n_times <- 6
  n_per_time <- 3
  n <- n_times * n_per_time

  df <- data.frame(
    count = rpois(n, lambda = 10),
    effort = rgamma(n, shape = 5, rate = 1),
    month = rep(1:n_times, each = n_per_time)
  )

  fit <- tratio(
    count | effort ~ 1,
    data = df,
    family = ratiod_poisson_gamma(),
    temporal = temporal_rw1("month", cyclic = TRUE),
    mode = "hmc",
    control = list(iter = 50, warmup = 25, chains = 1, verbose = FALSE)
  )

  expect_s3_class(fit, "ratiod_fit")
})
