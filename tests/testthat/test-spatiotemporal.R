# test-spatiotemporal.R
# Tests for spatiotemporal interaction specifications

test_that("spatiotemporal creates Type I specification", {
  adj <- matrix(0, 5, 5)
  for (i in 1:4) adj[i, i+1] <- adj[i+1, i] <- 1

  st <- spatiotemporal(
    spatial = spatial_car(adj, level = "group", group_var = "region"),
    temporal = temporal_rw1("year"),
    type = "I"
  )

  expect_s3_class(st, "ratiod_spatiotemporal")
  expect_equal(st$type, "I")
  expect_true(st$shared)
})

test_that("spatiotemporal creates Type IV specification", {
  adj <- matrix(0, 5, 5)
  for (i in 1:4) adj[i, i+1] <- adj[i+1, i] <- 1

  st <- spatiotemporal(
    spatial = spatial_car(adj, level = "group", group_var = "region"),
    temporal = temporal_rw1("year"),
    type = "IV"
  )

  expect_s3_class(st, "ratiod_spatiotemporal")
  expect_equal(st$type, "IV")
})

test_that("spatiotemporal 'iid' normalizes to 'I'", {
  adj <- matrix(0, 3, 3)
  adj[1, 2] <- adj[2, 1] <- 1
  adj[2, 3] <- adj[3, 2] <- 1

  st <- spatiotemporal(
    spatial = spatial_car(adj, level = "group", group_var = "site"),
    temporal = temporal_ar1("time"),
    type = "iid"
  )

  expect_equal(st$type, "I")
})

test_that("spatiotemporal validates spatial specification", {
  expect_error(
    spatiotemporal(
      spatial = list(type = "invalid"),
      temporal = temporal_rw1("year"),
      type = "I"
    ),
    "tulpaRatio spatial specification"
  )
})

test_that("spatiotemporal validates temporal specification", {
  adj <- matrix(0, 3, 3)
  adj[1, 2] <- adj[2, 1] <- 1

  expect_error(
    spatiotemporal(
      spatial = spatial_car(adj, level = "group", group_var = "site"),
      temporal = list(type = "invalid"),
      type = "I"
    ),
    "tulpaRatio temporal specification"
  )
})

test_that("spatiotemporal warns for non-shared effects", {
  adj <- matrix(0, 3, 3)
  adj[1, 2] <- adj[2, 1] <- 1

  expect_warning(
    spatiotemporal(
      spatial = spatial_car(adj, level = "group", group_var = "site"),
      temporal = temporal_rw1("year"),
      type = "I",
      shared = FALSE
    ),
    "Non-shared"
  )
})

test_that("spatiotemporal warns for proper CAR with Type IV", {
  adj <- matrix(0, 3, 3)
  adj[1, 2] <- adj[2, 1] <- 1
  adj[2, 3] <- adj[3, 2] <- 1

  expect_warning(
    spatiotemporal(
      spatial = spatial_car(adj, level = "group", group_var = "site", proper = TRUE),
      temporal = temporal_rw1("year"),
      type = "IV"
    ),
    "identifiability"
  )
})

test_that("print.ratiod_spatiotemporal works", {
  adj <- matrix(0, 4, 4)
  for (i in 1:3) adj[i, i+1] <- adj[i+1, i] <- 1

  st <- spatiotemporal(
    spatial = spatial_car(adj, level = "group", group_var = "region"),
    temporal = temporal_rw2("year"),
    type = "III"
  )

  output <- capture.output(print(st))
  expect_true(any(grepl("Spatiotemporal", output)))
  expect_true(any(grepl("Type III", output)))
})

test_that("validate_spatiotemporal computes correct dimensions", {
  adj <- matrix(0, 4, 4)
  for (i in 1:3) adj[i, i+1] <- adj[i+1, i] <- 1

  st <- spatiotemporal(
    spatial = spatial_car(adj, level = "group", group_var = "region"),
    temporal = temporal_rw1("year"),
    type = "I"
  )

  df <- expand.grid(
    region = 1:4,
    year = 2020:2024
  )
  df$x <- rnorm(nrow(df))

  st_valid <- tulpaRatio:::validate_spatiotemporal(st, df)

  expect_equal(st_valid$n_spatial, 4)
  expect_equal(st_valid$n_times, 5)
  expect_equal(st_valid$n_params, 20)  # 4 * 5
})

test_that("build_st_index creates correct mappings", {
  adj <- matrix(0, 3, 3)
  adj[1, 2] <- adj[2, 1] <- 1
  adj[2, 3] <- adj[3, 2] <- 1

  st <- spatiotemporal(
    spatial = spatial_car(adj, level = "group", group_var = "site"),
    temporal = temporal_rw1("time"),
    type = "I"
  )

  df <- data.frame(
    site = c(1, 1, 2, 2, 3, 3),
    time = c(1, 2, 1, 2, 1, 2),
    x = rnorm(6)
  )

  st_valid <- tulpaRatio:::validate_spatiotemporal(st, df)
  idx <- st_valid$st_index

  expect_equal(idx$S, 3)
  expect_equal(idx$T, 2)
  expect_equal(idx$N, 6)
  expect_equal(length(idx$s_idx), 6)
  expect_equal(length(idx$t_idx), 6)
})

test_that("prepare_spatiotemporal_for_hmc returns correct structure", {
  adj <- matrix(0, 3, 3)
  adj[1, 2] <- adj[2, 1] <- 1
  adj[2, 3] <- adj[3, 2] <- 1

  st <- spatiotemporal(
    spatial = spatial_car(adj, level = "group", group_var = "site"),
    temporal = temporal_rw1("time"),
    type = "II"
  )

  df <- expand.grid(site = 1:3, time = 1:4)
  st_valid <- tulpaRatio:::validate_spatiotemporal(st, df)

  hmc_data <- tulpaRatio:::prepare_spatiotemporal_for_hmc(st_valid, df)

  expect_true(hmc_data$has_spatiotemporal)
  expect_equal(hmc_data$type, "II")
  expect_true(hmc_data$shared)
  expect_equal(hmc_data$n_spatial, 3)
  expect_equal(hmc_data$n_times, 4)
})

test_that("prepare_spatiotemporal_for_hmc handles NULL", {
  hmc_data <- tulpaRatio:::prepare_spatiotemporal_for_hmc(NULL, data.frame())

  expect_false(hmc_data$has_spatiotemporal)
  expect_equal(hmc_data$type, "none")
})

st_gp_adj <- function(S) {
  adj <- matrix(0, S, S)
  for (i in seq_len(S - 1L)) adj[i, i + 1L] <- adj[i + 1L, i] <- 1
  adj
}

st_gp_df <- function(S = 5L, T = 4L, seed = 123) {
  set.seed(seed)
  xy <- cbind(runif(S, 0, 10), runif(S, 0, 10))
  data.frame(
    s = factor(rep(seq_len(S), each = T)),
    lon = rep(xy[, 1], each = T),
    lat = rep(xy[, 2], each = T),
    year = rep(seq_len(T), times = S)
  )
}

st_gp_spec <- function(S = 5L, coords = ~ lon + lat, ...) {
  spatiotemporal(
    spatial = spatial_car(st_gp_adj(S), level = "group", group_var = "s"),
    temporal = temporal_rw1("year"),
    coords = coords,
    ...
  )
}

test_that("nonsep_gp carries its kernels and its non-separability", {
  st <- st_gp_spec(type = "nonsep_gp", nonsep_type = "gneiting",
                   cov_space = "matern", nn = 4)

  expect_s3_class(st, "ratiod_spatiotemporal")
  expect_equal(st$type, "nonsep_gp")
  expect_equal(st$nonsep_type, "gneiting")
  expect_equal(st$cov_space, "matern")
  expect_equal(st$cov_time, "exponential")
  expect_equal(st$nn, 4L)
  expect_equal(st$coord_vars, c("lon", "lat"))
})

test_that("the GP types need coordinates of their own", {
  for (type in c("separable", "nonsep_gp")) {
    expect_error(
      spatiotemporal(
        spatial = spatial_car(st_gp_adj(3), level = "group", group_var = "s"),
        temporal = temporal_rw1("t"),
        type = type
      ),
      "coords"
    )
  }
  expect_error(st_gp_spec(type = "separable", coords = ~ lon), "exactly 2")
  expect_error(st_gp_spec(type = "separable", coords = 42), "formula")
})

test_that("a GP main effect pairs with a GP interaction", {
  # gcol33/tulpaRatio#70: the GP sampler entry is handed the interaction now,
  # so the pairing builds instead of being refused.
  st <- spatiotemporal(
    spatial = spatial_gp(~ lon + lat),
    temporal = temporal_rw1("year"),
    type = "separable",
    coords = ~ lon + lat
  )
  expect_s3_class(st, "ratiod_spatiotemporal")
  expect_equal(st$type, "separable")
  expect_equal(st$coord_vars, c("lon", "lat"))
})

test_that("separable is the product arm, and says so", {
  expect_error(
    st_gp_spec(type = "separable", nonsep_type = "gneiting"),
    "nonsep_gp"
  )
  expect_equal(st_gp_spec(type = "separable")$nonsep_type, "product")
})

test_that("an unknown kernel is refused where the argument is nameable", {
  expect_error(st_gp_spec(type = "separable", cov_space = "cauchy"),
               "Unknown covariance")
})

test_that("validation builds the grid the interaction is defined on", {
  S <- 5L; T <- 4L
  st <- st_gp_spec(S = S, type = "separable", nn = 3)
  st <- tulpaRatio:::validate_spatiotemporal(st, st_gp_df(S, T))

  expect_equal(st$n_spatial, S)
  expect_equal(st$n_times, T)
  expect_equal(st$n_params, S * T)

  g <- st$gp_grid
  expect_equal(nrow(g$coords), S * T)
  expect_equal(length(g$time_values), S * T)
  expect_equal(sort(g$nn_order), seq_len(S * T))
  expect_equal(g$nn_order_inv[g$nn_order], seq_len(S * T))
  # Cell k = (s - 1) * T + t: a unit's coordinates repeat across its times,
  # and its times repeat across units.
  expect_equal(g$coords[1, ], g$coords[T, ])
  expect_false(isTRUE(all.equal(g$coords[1, ], g$coords[T + 1L, ])))
  expect_equal(g$time_values[seq_len(T)], g$time_values[T + seq_len(T)])
  # Zero, not Inf, where nn_idx says there is no neighbour.
  expect_true(all(g$nn_dist_space[g$nn_idx == 0L] == 0))
  expect_true(all(is.finite(g$nn_dist_time)))
})

test_that("a unit whose rows disagree gets its centroid, and is told so", {
  df <- st_gp_df(4L, 3L)
  df$lon <- df$lon + rep(c(-1, 0, 1), times = 4)  # spread within each unit
  st <- st_gp_spec(S = 4L, type = "separable", nn = 3)
  expect_message(
    tulpaRatio:::validate_spatiotemporal(st, df),
    "centroid"
  )
})

test_that("prepare_spatiotemporal_for_hmc emits the GP grid", {
  S <- 4L; T <- 3L
  df <- st_gp_df(S, T)
  st <- st_gp_spec(S = S, type = "nonsep_gp", nonsep_type = "gneiting", nn = 3)
  st <- tulpaRatio:::validate_spatiotemporal(st, df)
  info <- tulpaRatio:::prepare_spatiotemporal_for_hmc(st, df)

  expect_equal(info$type, "nonsep_gp")
  expect_equal(info$n_params, S * T)
  expect_equal(info$gp$nonsep_type, "gneiting")
  # Flattened row-major over (grid cell, neighbour) for C++.
  expect_equal(length(info$gp$coords), 2L * S * T)
  expect_equal(length(info$gp$time_values), S * T)
  expect_equal(length(info$gp$nn_idx), S * T * info$gp$nn)
  expect_equal(length(info$gp$nn_dist_space), S * T * info$gp$nn)
  expect_equal(info$gp$cov_space, 0L)  # exponential
  expect_equal(info$gp$cov_time, 0L)
})

test_that("a spatial GP indexes an interaction by location, not by row", {
  # Two observations per location: s_idx has to be the location, or st_flat
  # runs past the S x T grid it indexes.
  df <- data.frame(
    lon = rep(c(0, 1, 2), each = 4),
    lat = rep(c(0, 1, 2), each = 4),
    year = rep(1:2, times = 6)
  )
  st <- spatiotemporal(
    spatial = spatial_gp(~ lon + lat, nn = 2),
    temporal = temporal_rw1("year"),
    type = "I"
  )
  st <- tulpaRatio:::validate_spatiotemporal(st, df)

  expect_equal(st$n_spatial, 3L)
  expect_equal(st$n_params, 6L)
  expect_true(max(st$st_index$st_flat) <= st$n_params)
  expect_equal(sort(unique(st$st_index$s_idx)), 1:3)
})

test_that("compute_st_nngp_neighbors creates valid structure", {
  set.seed(42)
  coords <- cbind(runif(15), runif(15))
  time <- rep(1:3, each = 5)

  result <- tulpaRatio:::compute_st_nngp_neighbors(coords, time, k = 3)

  expect_equal(nrow(result$nn_idx), 15)
  expect_equal(ncol(result$nn_idx), 3)
  expect_equal(length(result$nn_order), 15)
  expect_equal(length(result$nn_order_inv), 15)

  # Check ordering is valid permutation
  expect_equal(sort(result$nn_order), 1:15)
  expect_equal(result$nn_order_inv[result$nn_order], 1:15)
})

test_that("all interaction types are supported", {
  adj <- matrix(0, 4, 4)
  for (i in 1:3) adj[i, i+1] <- adj[i+1, i] <- 1

  spatial <- spatial_car(adj, level = "group", group_var = "s")
  temporal <- temporal_rw1("t")

  for (type in c("I", "II", "III", "IV")) {
    st <- spatiotemporal(
      spatial = spatial,
      temporal = temporal,
      type = type
    )
    expect_equal(st$type, type)
  }
})

test_that("separable builds cleanly on an areal main effect", {
  st <- st_gp_spec(type = "separable")
  expect_equal(st$type, "separable")
  expect_equal(st$nonsep_type, "product")
})

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
