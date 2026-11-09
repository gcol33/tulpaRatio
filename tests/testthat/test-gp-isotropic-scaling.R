test_that("GP scaling preserves Euclidean distances and rotations", {
  coords <- cbind(c(-3, -1, 0, 2, 7), c(0, 1, 0, 2, -1))
  scaled <- tulpaRatio:::gp_scale_coords(coords, TRUE)
  unit <- sqrt(mean(apply(coords, 2, var)))
  expect_equal(scaled$scale, rep(unit, 2))
  expect_equal(as.numeric(dist(scaled$coords)), as.numeric(dist(coords)) / unit)
  rotation <- matrix(c(cos(.7), sin(.7), -sin(.7), cos(.7)), 2)
  rotated <- tulpaRatio:::gp_scale_coords(coords %*% rotation, TRUE)
  expect_equal(rotated$scale, scaled$scale)
  expect_equal(as.numeric(dist(rotated$coords)), as.numeric(dist(scaled$coords)))
  gp <- list(coord_center = scaled$center, coord_scale = scaled$scale)
  expect_equal(tulpaRatio:::gp_scale_new_coords(gp, coords), scaled$coords)
  expect_equal(tulpaRatio:::gp_scale_new_coords(gp, coords + 10),
               scaled$coords + 10 / unit)
  phi <- .8
  expect_equal(exp(-as.numeric(dist(scaled$coords)) / phi),
               exp(-as.numeric(dist(coords)) / (phi * unit)))
  expect_identical(tulpaRatio:::gp_scale_coords(coords, FALSE)$coords, coords)
})

test_that("constant coordinate columns remain finite for every spatial GP spec", {
  data <- data.frame(lon = seq_len(8), lat = 3)
  gp <- tulpaRatio:::validate_gp(spatial_gp(~ lon + lat, nn = 3), data)
  hsgp <- tulpaRatio:::validate_hsgp(spatial_hsgp(~ lon + lat), data)
  ms <- tulpaRatio:::validate_gp(spatial_multiscale(~ lon + lat), data)
  svc <- tulpaRatio:::validate_svc(spatial_svc(~ lon + lat, terms = 1),
                                  data, matrix(1, 8, 1))
  data$time <- rep(1:2, 4)
  st <- tulpaRatio:::validate_spatiotemporal(
    spatiotemporal(spatial_hsgp(~ lon + lat), temporal_rw1("time")), data)
  for (spec in list(gp, hsgp, ms, svc, st$spatial)) {
    expect_true(all(is.finite(spec$coords_matrix)))
    expect_identical(spec$coords_matrix[, 2], rep(0, 8))
    expect_equal(spec$coord_scale, rep(sd(data$lon) / sqrt(2), 2))
    expect_equal(unname(tulpaRatio:::gp_scale_new_coords(spec, as.matrix(data[c("lon", "lat")]))),
                 spec$coords_matrix)
  }
  expect_identical(gp$n_unique, 8L)
  expect_true(all(is.finite(gp$neighbor_info$nn_dist[gp$neighbor_info$nn_idx > 0])))
  for (coords in list(matrix(3, 8, 2), matrix(c(2, 3), 1, 2))) {
    scaled <- tulpaRatio:::gp_scale_coords(coords, TRUE)
    expect_identical(scaled$coords, matrix(0, nrow(coords), ncol(coords)))
    expect_identical(scaled$scale, c(1, 1))
  }
  single <- suppressMessages(tulpaRatio:::validate_gp(
    spatial_gp(~ lon + lat), data.frame(lon = rep(2, 8), lat = 3)))
  expect_identical(single$n_unique, 1L)
  expect_identical(single$unique_coords, matrix(0, 1, 2))
  expect_identical(dim(single$neighbor_info$nn_idx), c(1L, 0L))
})

test_that("lengthscale prior parameters use the prepared distance units", {
  data <- data.frame(lon = seq_len(8), lat = 3)
  gp <- tulpaRatio:::validate_gp(spatial_gp(~ lon + lat, nn = 3), data)
  info <- tulpaRatio:::prepare_gp_for_hmc(gp, data, 8)
  priors <- list(gp_phi_prior_lower = .2, gp_phi_prior_upper = 3)
  bundle <- tulpaRatio:::gp_bundle(info, priors)
  expect_equal(bundle$phi_prior_lower, .2)
  expect_equal(bundle$phi_prior_upper, 3)
  expect_equal(info$coords, as.vector(t(gp$unique_coords)))
  unit <- gp$coord_scale[1]
  raw_distance <- as.numeric(dist(as.matrix(data)))
  scaled_distance <- as.numeric(dist(gp$unique_coords))
  for (phi in c(bundle$phi_prior_lower, bundle$phi_prior_upper)) {
    expect_equal(exp(-scaled_distance / phi), exp(-raw_distance / (phi * unit)))
  }
})

test_that("space-time GP grids preserve the spatial distance units", {
  data <- expand.grid(site = seq_len(4), time = 1:2)
  data$lon <- c(0, 1, 3, 6)[data$site]
  data$lat <- 3
  adjacency <- 1 * (as.matrix(dist(seq_len(4))) == 1)
  spec <- spatiotemporal(spatial_car(adjacency, group_var = "site"),
                         temporal_rw1("time"), type = "separable",
                         coords = ~ lon + lat, nn = 3)
  validated <- tulpaRatio:::validate_spatiotemporal(spec, data)
  unit_coords <- cbind(c(0, 1, 3, 6), 3)
  scaled <- tulpaRatio:::gp_scale_coords(unit_coords, TRUE)
  expect_equal(validated$gp_grid$coords,
               scaled$coords[rep(seq_len(4), each = 2), , drop = FALSE])
  expect_true(all(is.finite(validated$gp_grid$coords)))
  expect_equal(sd(unique(validated$gp_grid$time_values)), 1)
})

test_that("constant-column GP fits use the same scaled geometry end to end", {
  skip_on_cran()
  set.seed(221)
  data <- data.frame(lon = rep(seq_len(8), 3), lat = 3)
  data$count <- rpois(24, exp(1 + .1 * data$lon))
  data$effort <- rgamma(24, shape = 3, rate = 1)
  scaled <- tulpaRatio:::gp_scale_coords(as.matrix(data[c("lon", "lat")]), TRUE)
  manual <- data
  manual[c("lon", "lat")] <- as.data.frame(scaled$coords)
  control <- list(iter = 100, warmup = 50, chains = 1, cores = 1, seed = 221)
  a <- suppressWarnings(tratio(count | effort ~ 1, data,
    family = ratiod_poisson_gamma(), spatial = spatial_gp(~ lon + lat, nn = 3),
    mode = "hmc", control = control))
  b <- suppressWarnings(tratio(count | effort ~ 1, manual,
    family = ratiod_poisson_gamma(),
    spatial = spatial_gp(~ lon + lat, nn = 3, scale_coords = FALSE),
    mode = "hmc", control = control))
  expect_true(all(is.finite(a$draws)))
  expect_identical(a$draws, b$draws)
  expect_identical(fitted(a), fitted(b))
})
