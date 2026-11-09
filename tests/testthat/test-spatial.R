test_that("spatial_car validates adjacency matrix", {
  # Non-square matrix
  bad_adj <- matrix(1, 3, 4)
  expect_error(spatial_car(bad_adj), "square")

  # Non-symmetric matrix
  bad_adj <- matrix(c(0, 1, 0, 0), 2, 2)
  expect_error(spatial_car(bad_adj), "symmetric")

  # Valid adjacency
  good_adj <- matrix(c(0, 1, 1, 0), 2, 2)
  expect_s3_class(spatial_car(good_adj, spatial_level = "obs"), "ratiod_spatial")
})

test_that("spatial_car requires group_var for group level", {
  adj <- matrix(c(0, 1, 1, 0), 2, 2)

  expect_error(
    spatial_car(adj, spatial_level = "group"),
    "group_var"
  )

  expect_s3_class(
    spatial_car(adj, spatial_level = "group", group_var = "site"),
    "ratiod_spatial"
  )
})

test_that("spatial_bym2 computes scale factor", {
  # Simple chain graph
  adj <- matrix(0, 4, 4)
  adj[1, 2] <- adj[2, 1] <- 1
  adj[2, 3] <- adj[3, 2] <- 1
  adj[3, 4] <- adj[4, 3] <- 1

  sp <- spatial_bym2(adj, spatial_level = "obs")

  expect_s3_class(sp, "ratiod_spatial")
  expect_equal(sp$type, "bym2")
  expect_true(sp$scale_factor > 0)
})

test_that("is_connected detects disconnected graphs", {
  # Connected graph
  adj_conn <- matrix(c(0, 1, 0, 1, 0, 1, 0, 1, 0), 3, 3)
  expect_true(tulpaRatio:::is_connected(adj_conn))

  # Disconnected graph (two isolated nodes)
  adj_disc <- matrix(c(0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0), 4, 4)
  expect_false(tulpaRatio:::is_connected(adj_disc))
})

# =============================================================================
# Tests for proper CAR (spatial_car with proper = TRUE)
# =============================================================================

test_that("spatial_car creates ICAR by default", {
  adj <- matrix(0, 4, 4)
  adj[1, 2] <- adj[2, 1] <- 1
  adj[2, 3] <- adj[3, 2] <- 1
  adj[3, 4] <- adj[4, 3] <- 1

  sp <- spatial_car(adj, spatial_level = "obs")

  expect_s3_class(sp, "ratiod_spatial")
  expect_equal(sp$type, "car")
  expect_false(sp$proper)
  expect_null(sp$rho_bounds)
})

test_that("spatial_car with proper = TRUE creates proper CAR", {
  adj <- matrix(0, 4, 4)
  adj[1, 2] <- adj[2, 1] <- 1
  adj[2, 3] <- adj[3, 2] <- 1
  adj[3, 4] <- adj[4, 3] <- 1

  sp <- spatial_car(adj, spatial_level = "obs", proper = TRUE)

  expect_s3_class(sp, "ratiod_spatial")
  expect_equal(sp$type, "car_proper")
  expect_true(sp$proper)
  expect_true(!is.null(sp$rho_bounds))
  expect_true(sp$rho_bounds["lower"] >= 0)
  expect_true(sp$rho_bounds["upper"] <= 1)
  expect_true(sp$rho_bounds["lower"] < sp$rho_bounds["upper"])
})

test_that("proper CAR print method shows rho info", {
  adj <- matrix(0, 4, 4)
  adj[1, 2] <- adj[2, 1] <- 1
  adj[2, 3] <- adj[3, 2] <- 1
  adj[3, 4] <- adj[4, 3] <- 1

  sp <- spatial_car(adj, spatial_level = "obs", proper = TRUE)
  output <- capture.output(print(sp))

  expect_true(any(grepl("Proper CAR", output)))
  expect_true(any(grepl("Rho bounds", output)))
  expect_true(any(grepl("estimated", output, ignore.case = TRUE)))
})

test_that("ICAR print method shows fixed rho info", {
  adj <- matrix(0, 4, 4)
  adj[1, 2] <- adj[2, 1] <- 1
  adj[2, 3] <- adj[3, 2] <- 1
  adj[3, 4] <- adj[4, 3] <- 1

  sp <- spatial_car(adj, spatial_level = "obs", proper = FALSE)
  output <- capture.output(print(sp))

  expect_true(any(grepl("ICAR", output)))
  expect_true(any(grepl("fixed at 1", output)))
})

test_that("compute_car_rho_bounds returns valid range", {
  # Simple chain graph
  adj <- matrix(0, 5, 5)
  for (i in 1:4) {
    adj[i, i + 1] <- adj[i + 1, i] <- 1
  }

  bounds <- tulpaRatio:::compute_car_rho_bounds(adj)

  expect_true(is.numeric(bounds))
  expect_equal(length(bounds), 2)
  expect_true("lower" %in% names(bounds))
  expect_true("upper" %in% names(bounds))
  expect_true(bounds["lower"] < bounds["upper"])
  expect_true(bounds["lower"] >= 0)
  expect_true(bounds["upper"] <= 1)
})

test_that("compute_car_rho_bounds handles isolated nodes", {
  # Graph with one isolated node
  adj <- matrix(0, 4, 4)
  adj[1, 2] <- adj[2, 1] <- 1
  adj[2, 3] <- adj[3, 2] <- 1
  # Node 4 has no neighbors (isolated)

  expect_warning(
    bounds <- tulpaRatio:::compute_car_rho_bounds(adj),
    regexp = "isolated"
  )

  # Should still return valid bounds
  expect_true(bounds["lower"] < bounds["upper"])
})

test_that("proper CAR works with group level", {
  adj <- matrix(0, 3, 3)
  adj[1, 2] <- adj[2, 1] <- 1
  adj[2, 3] <- adj[3, 2] <- 1

  sp <- spatial_car(adj, spatial_level = "group", group_var = "region", proper = TRUE)

  expect_equal(sp$type, "car_proper")
  expect_equal(sp$group_var, "region")
  expect_equal(sp$spatial_level, "group")
  expect_true(sp$proper)
})

test_that("proper CAR validates adjacency like ICAR", {
  # Non-square matrix
  bad_adj <- matrix(1, 3, 4)
  expect_error(spatial_car(bad_adj, proper = TRUE), "square")

  # Non-symmetric matrix
  bad_adj <- matrix(c(0, 1, 0, 0), 2, 2)
  expect_error(spatial_car(bad_adj, proper = TRUE), "symmetric")
})

test_that("spatial_car shared argument works for both variants", {
  adj <- matrix(c(0, 1, 1, 0), 2, 2)

  # ICAR - expect warning about non-shared effects
  expect_warning(
    sp_icar <- spatial_car(adj, spatial_level = "obs", shared = FALSE),
    "confounded"
  )
  expect_false(sp_icar$shared)

  # Proper CAR - expect warning about non-shared effects
  expect_warning(
    sp_proper <- spatial_car(adj, spatial_level = "obs", proper = TRUE, shared = FALSE),
    "confounded"
  )
  expect_false(sp_proper$shared)
})

test_that("spatial_car creates valid structure", {
  adj <- matrix(c(0, 1, 0, 1, 0, 1, 0, 1, 0), nrow = 3)
  spatial <- spatial_car(group_var = "site", adjacency = adj)

  expect_s3_class(spatial, "ratiod_spatial")
  expect_equal(spatial$type, "car")
  expect_equal(spatial$group_var, "site")
})

test_that("spatial_bym2 creates valid structure", {
  adj <- matrix(c(0, 1, 0, 1, 0, 1, 0, 1, 0), nrow = 3)
  spatial <- spatial_bym2(group_var = "site", adjacency = adj)

  expect_s3_class(spatial, "ratiod_spatial")
  expect_equal(spatial$type, "bym2")
})

test_that("spatial_gp creates valid structure", {
  gp <- spatial_gp(coords = c("lon", "lat"))

  expect_s3_class(gp, "ratiod_gp")
  # coords may be stored as coord_vars or similar
  expect_true("coord_vars" %in% names(gp) || "coords" %in% names(gp))
})

test_that("print.ratiod_spatial works", {
  adj <- matrix(c(0, 1, 0, 1, 0, 1, 0, 1, 0), nrow = 3)
  spatial <- spatial_car(group_var = "site", adjacency = adj)

  output <- capture.output(print(spatial))
  expect_true(length(output) > 0)
})

test_that("prepare_spatial_for_laplace errors without group_var", {
  spatial <- list(type = "car", group_var = NULL)
  data <- data.frame(x = 1:5)
  formula <- list()

  expect_error(
    tulpaRatio:::prepare_spatial_for_laplace(spatial, data, formula),
    "group_var"
  )
})

test_that("prepare_spatial_for_laplace errors when group_var not in data", {
  spatial <- list(type = "car", group_var = "site")
  data <- data.frame(x = 1:5)
  formula <- list()

  expect_error(
    tulpaRatio:::prepare_spatial_for_laplace(spatial, data, formula),
    "not found in data"
  )
})

test_that("prepare_spatial_for_laplace errors without adjacency", {
  spatial <- list(type = "car", group_var = "site", adjacency = NULL, adj_matrix = NULL)
  data <- data.frame(site = factor(rep(1:3, each = 2)))
  formula <- list()

  expect_error(
    tulpaRatio:::prepare_spatial_for_laplace(spatial, data, formula),
    "adj_matrix"
  )
})

test_that("prepare_spatial_for_laplace works correctly", {
  # Create simple adjacency matrix (chain: 1-2-3)
  adj <- matrix(c(0, 1, 0, 1, 0, 1, 0, 1, 0), nrow = 3)

  spatial <- list(
    type = "car",
    group_var = "site",
    adjacency = adj
  )
  data <- data.frame(site = factor(rep(1:3, each = 2)))
  formula <- list()

  result <- tulpaRatio:::prepare_spatial_for_laplace(spatial, data, formula)

  expect_equal(result$n_units, 3L)
  expect_equal(length(result$group_idx), 6)
  expect_equal(result$n_neighbors, c(1L, 2L, 1L))
})

test_that("prepare_spatial_for_pg errors without group_var for group level", {
  spatial <- list(spatial_level = "group", group_var = NULL, adjacency = diag(3))

  expect_error(
    tulpaRatio:::prepare_spatial_for_pg(spatial, data.frame(x = 1:3), list()),
    "group_var"
  )
})
