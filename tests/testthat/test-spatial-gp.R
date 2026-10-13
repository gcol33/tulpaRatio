# Tests for spatial_gp() and spatial_multiscale()

test_that("spatial_gp creates valid object with formula coordinates", {
  gp <- spatial_gp(~ lon + lat)

  expect_s3_class(gp, "ratiod_gp")
  expect_s3_class(gp, "ratiod_spatial")
  expect_equal(gp$type, "gp")
  expect_equal(gp$coord_vars, c("lon", "lat"))
  expect_equal(gp$cov, "exponential")
  expect_equal(gp$nn, 15L)
  expect_true(gp$shared)
  expect_true(gp$scale)
})

test_that("spatial_gp creates valid object with character coordinates", {
  gp <- spatial_gp(c("x", "y"))

  expect_s3_class(gp, "ratiod_gp")
  expect_equal(gp$coord_vars, c("x", "y"))
})

test_that("spatial_gp accepts different covariance functions", {
  gp_exp <- spatial_gp(~ lon + lat, cov = "exponential")
  gp_mat <- spatial_gp(~ lon + lat, cov = "matern", nu = 2.5)
  gp_gau <- spatial_gp(~ lon + lat, cov = "gaussian")
  gp_sph <- spatial_gp(~ lon + lat, cov = "spherical")

  expect_equal(gp_exp$cov, "exponential")
  expect_equal(gp_mat$cov, "matern")
  expect_equal(gp_mat$nu, 2.5)
  expect_equal(gp_gau$cov, "gaussian")
  expect_equal(gp_sph$cov, "spherical")
})

test_that("spatial_gp rejects invalid coordinate specification", {
  expect_error(spatial_gp(~ lon), "exactly 2 coordinate variables")
  expect_error(spatial_gp(~ lon + lat + alt), "exactly 2 coordinate variables")
  expect_error(spatial_gp(c("x")), "character vector of length 2")
  expect_error(spatial_gp(123), "formula.*or character vector")
})

test_that("spatial_gp rejects invalid nu for Matern", {
  expect_error(spatial_gp(~ lon + lat, cov = "matern", nu = -1),
               "positive number")
  expect_error(spatial_gp(~ lon + lat, cov = "matern", nu = 0),
               "positive number")
})

test_that("spatial_gp rejects invalid nn", {
  expect_error(spatial_gp(~ lon + lat, nn = 0), "positive integer")
  expect_error(spatial_gp(~ lon + lat, nn = -5), "positive integer")
})

test_that("spatial_gp warns for non-shared effects", {
  expect_warning(spatial_gp(~ lon + lat, shared = FALSE), "confounded")
})

test_that("spatial_gp print method works", {
  gp <- spatial_gp(~ lon + lat, cov = "matern", nu = 1.5)
  output <- capture.output(print(gp))
  expect_true(any(grepl("Gaussian Process", output)))
  expect_true(any(grepl("matern", output)))
  expect_true(any(grepl("1.5", output)))
})

# Tests for spatial_multiscale

test_that("spatial_multiscale creates valid object", {
  ms <- spatial_multiscale(~ lon + lat)

  expect_s3_class(ms, "ratiod_multiscale")
  expect_s3_class(ms, "ratiod_spatial")
  expect_equal(ms$type, "multiscale")
  expect_equal(ms$coord_vars, c("lon", "lat"))
  expect_equal(ms$scales, c("local", "regional"))
  expect_equal(ms$m, 10L)
  expect_equal(ms$c_boundary, 1.5)
})

test_that("spatial_multiscale accepts custom range priors", {
  ms <- spatial_multiscale(
    ~ lon + lat,
    range_local = c(0.5, 2),
    range_regional = c(5, 20)
  )

  expect_equal(ms$range_local, c(0.5, 2))
  expect_equal(ms$range_regional, c(5, 20))
})

test_that("spatial_multiscale rejects invalid range specifications", {
  expect_error(
    spatial_multiscale(~ lon + lat, range_local = c(2, 1)),
    "lower < upper"
  )
  expect_error(
    spatial_multiscale(~ lon + lat, range_regional = c(10, 5)),
    "lower < upper"
  )
})

test_that("spatial_multiscale warns for overlapping ranges", {
  expect_warning(
    spatial_multiscale(~ lon + lat, range_local = c(0.1, 10), range_regional = c(5, 50)),
    "overlap"
  )
})

test_that("spatial_multiscale warns for non-shared effects", {
  expect_warning(
    spatial_multiscale(~ lon + lat, shared = FALSE),
    "confounded"
  )
})

test_that("spatial_multiscale print method works", {
  ms <- spatial_multiscale(~ lon + lat)
  output <- capture.output(print(ms))
  expect_true(any(grepl("Multi-Scale", output)))
  expect_true(any(grepl("local", output)))
  expect_true(any(grepl("regional", output)))
})

# Tests for validate_gp

test_that("validate_gp computes neighbor structure for GP", {
  gp <- spatial_gp(~ x + y, nn = 5)

  # Create test data
  set.seed(123)
  df <- data.frame(
    x = runif(20),
    y = runif(20)
  )

  validated <- validate_gp(gp, df)

  expect_equal(validated$n_obs, 20)
  expect_equal(validated$n_spatial, 20)
  expect_true(!is.null(validated$coords_matrix))
  expect_true(!is.null(validated$neighbor_info))
  expect_equal(validated$neighbor_info$k, 5)
})

test_that("validate_gp errors for missing coordinate columns", {
  gp <- spatial_gp(~ lon + lat)
  df <- data.frame(x = 1:10, y = 1:10)

  expect_error(validate_gp(gp, df), "lon.*not found")
})

test_that("validate_gp errors for NA coordinates", {
  gp <- spatial_gp(~ x + y)
  df <- data.frame(x = c(1:9, NA), y = 1:10)

  expect_error(validate_gp(gp, df), "missing values")
})

test_that("validate_gp scales coordinates when requested", {
  gp <- spatial_gp(~ x + y, scale = TRUE)

  df <- data.frame(
    x = c(0, 100, 200, 300),
    y = c(0, 1000, 2000, 3000)
  )

  validated <- validate_gp(gp, df)

  # Scaled coordinates should have mean ~0 and sd ~1
  expect_equal(mean(validated$coords_matrix[, 1]), 0, tolerance = 0.01)
  expect_equal(sd(validated$coords_matrix[, 1]), 1, tolerance = 0.01)
})

test_that("validate_gp preserves coordinates when scale = FALSE", {
  gp <- spatial_gp(~ x + y, scale = FALSE)

  df <- data.frame(
    x = c(100, 200, 300, 400),
    y = c(1000, 2000, 3000, 4000)
  )

  validated <- validate_gp(gp, df)

  expect_equal(validated$coords_matrix[, 1], c(100, 200, 300, 400))
  expect_equal(validated$coords_matrix[, 2], c(1000, 2000, 3000, 4000))
})

# --- Duplicate coordinate handling ---

test_that("validate_gp detects unique coordinates and builds obs_to_loc", {
  gp <- spatial_gp(~ lon + lat, nn = 2, scale_coords = FALSE)

  # 3 sites, 2 obs per site = 6 obs
  df <- data.frame(
    lon = c(1, 1, 2, 2, 3, 3),
    lat = c(10, 10, 20, 20, 30, 30)
  )

  expect_message(validated <- validate_gp(gp, df), "3 unique locations from 6")

  expect_equal(validated$n_unique, 3L)
  expect_equal(validated$n_obs, 3L)
  expect_equal(validated$n_spatial, 3L)
  expect_equal(validated$obs_to_loc, c(1L, 1L, 2L, 2L, 3L, 3L))
  expect_equal(nrow(validated$unique_coords), 3L)
  expect_equal(validated$unique_coords[1, ], c(1, 10))
  expect_equal(validated$unique_coords[2, ], c(2, 20))
  expect_equal(validated$unique_coords[3, ], c(3, 30))
})

test_that("validate_gp is identity mapping when all coordinates are unique", {
  gp <- spatial_gp(~ lon + lat, nn = 3, scale_coords = FALSE)

  df <- data.frame(lon = c(1, 2, 3, 4), lat = c(10, 20, 30, 40))

  # Should not emit message when all unique
  validated <- validate_gp(gp, df)

  expect_equal(validated$n_unique, 4L)
  expect_equal(validated$obs_to_loc, 1:4)
  expect_equal(nrow(validated$unique_coords), 4L)
})

test_that("GP model fits with duplicate coordinates", {
  skip_on_cran()
  set.seed(42)
  n_sites <- 10
  n_per_site <- 5
  N <- n_sites * n_per_site

  site_x <- runif(n_sites)
  site_y <- runif(n_sites)
  site_id <- rep(1:n_sites, each = n_per_site)

  df <- data.frame(
    y = rbinom(N, 20, 0.5),
    trials = rep(20L, N),
    x = rnorm(N),
    lon = site_x[site_id],
    lat = site_y[site_id]
  )

  fit <- tratio(
    y | trials ~ x,
    data = df,
    family = ratiod_binomial(),
    spatial = spatial_gp(~ lon + lat, nn = 5),
    mode = "hmc",
    control = list(iter = 100, warmup = 50, chains = 1, verbose = FALSE)
  )

  expect_s3_class(fit, "ratiod_fit")

  # Verify sampler isn't stuck (most parameters have SD > 0)
  draws <- fit$draws
  sds <- apply(draws, 2, sd, na.rm = TRUE)
  sds <- sds[is.finite(sds)]
  expect_true(length(sds) > 0 && mean(sds > 1e-10) > 0.9,
              info = "Most parameters should move")
})
