# test-backend-coverage.R
# Additional tests to improve backend coverage

# -----------------------------------------------------------------------------
# backend_pg.R coverage
# -----------------------------------------------------------------------------

test_that("can_use_pg_backend returns TRUE for binomial family", {
  fam <- ratiod_binomial()
  expect_true(tulpaRatio:::can_use_pg_backend(fam))
})

test_that("can_use_pg_backend returns TRUE for negbin family (PG+CRT)", {
  fam <- ratiod_negbin_negbin()
  expect_true(tulpaRatio:::can_use_pg_backend(fam))
})

test_that("can_use_pg_backend returns FALSE for poisson_gamma family", {
  fam <- ratiod_poisson_gamma()
  expect_false(tulpaRatio:::can_use_pg_backend(fam))
})

test_that("extract_re_from_data handles no RE", {
  # Create mock formula object
  mock_formula <- list(
    numerator = list(
      random_effects = NULL
    )
  )
  data <- data.frame(y = 1:10)

  result <- tulpaRatio:::extract_re_from_data(mock_formula, data)

  expect_equal(result$n_groups, 0L)
  expect_equal(result$n_re_terms, 0L)
  expect_equal(length(result$group_idx), 10)
  expect_false(result$has_slopes)
})

test_that("extract_re_from_data handles single RE", {
  mock_formula <- list(
    numerator = list(
      random_effects = list(
        list(
          group_var = "site",
          group = c(1, 1, 2, 2, 3, 3),
          n_groups = 3L,
          slope_vars = character(0)
        )
      )
    )
  )
  data <- data.frame(y = 1:6)

  result <- tulpaRatio:::extract_re_from_data(mock_formula, data)

  expect_equal(result$n_groups, 3L)
  expect_equal(result$n_re_terms, 1L)
  expect_equal(result$group_var, "site")
  expect_equal(result$total_groups, 3L)
})

test_that("extract_re_from_data handles multiple RE terms", {
  mock_formula <- list(
    numerator = list(
      random_effects = list(
        list(
          group_var = "site",
          group = c(1, 1, 2, 2),
          n_groups = 2L,
          slope_vars = character(0)
        ),
        list(
          group_var = "year",
          group = c(1, 2, 1, 2),
          n_groups = 2L,
          slope_vars = character(0)
        )
      )
    )
  )
  data <- data.frame(y = 1:4)

  result <- tulpaRatio:::extract_re_from_data(mock_formula, data)

  expect_equal(result$n_re_terms, 2L)
  expect_equal(result$total_groups, 4L)
  expect_equal(nrow(result$group_idx_matrix), 4)
  expect_equal(ncol(result$group_idx_matrix), 2)
})

test_that("extract_re_from_data warns about slopes", {
  mock_formula <- list(
    numerator = list(
      random_effects = list(
        list(
          group_var = "site",
          group = c(1, 1, 2, 2),
          n_groups = 2L,
          slope_vars = c("x1")
        )
      )
    )
  )
  data <- data.frame(y = 1:4)

  expect_warning(
    result <- tulpaRatio:::extract_re_from_data(mock_formula, data),
    "Random slopes not yet fully supported"
  )
  expect_true(result$has_slopes)
})

# -----------------------------------------------------------------------------
# backend_laplace.R coverage
# -----------------------------------------------------------------------------

test_that("extract_re_for_laplace handles no RE", {
  mock_formula <- list(
    numerator = list(
      response = 1:10,
      random_effects = NULL
    )
  )

  result <- tulpaRatio:::extract_re_for_laplace(mock_formula)

  expect_equal(result$n_groups, 0L)
  expect_equal(result$n_re_terms, 0L)
  expect_equal(length(result$group_idx), 10)
})

test_that("extract_re_for_laplace handles single RE", {
  mock_formula <- list(
    numerator = list(
      response = 1:6,
      random_effects = list(
        list(
          group_var = "site",
          group = c(1, 1, 2, 2, 3, 3),
          n_groups = 3L,
          slope_vars = character(0)
        )
      )
    )
  )

  result <- tulpaRatio:::extract_re_for_laplace(mock_formula)

  expect_equal(result$n_groups, 3L)
  expect_equal(result$n_re_terms, 1L)
  expect_equal(result$group_var, "site")
})

test_that("extract_re_for_laplace handles multiple RE with warning for slopes", {
  mock_formula <- list(
    numerator = list(
      response = 1:4,
      random_effects = list(
        list(
          group_var = "site",
          group = c(1, 1, 2, 2),
          n_groups = 2L,
          slope_vars = c("x1")
        ),
        list(
          group_var = "year",
          group = c(1, 2, 1, 2),
          n_groups = 2L,
          slope_vars = character(0)
        )
      )
    )
  )

  expect_warning(
    result <- tulpaRatio:::extract_re_for_laplace(mock_formula),
    "Random slopes are not carried"
  )
  expect_equal(result$n_re_terms, 2)
  expect_true(result$has_slopes)
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

# -----------------------------------------------------------------------------
# benchmark.R coverage
# -----------------------------------------------------------------------------

test_that("ratiod_benchmark_compare runs with configs", {
  skip_on_cran()

  configs <- list(
    list(N = 20, p = 2, n_groups = 0)
  )

  result <- ratiod_benchmark_compare(
    configs,
    n_iter = 50,
    n_warmup = 25,
    n_threads = 1
  )

  expect_true(is.data.frame(result))
  expect_true("N" %in% names(result))
  expect_true("elapsed_sec" %in% names(result))
  expect_equal(nrow(result), 1)
})

test_that("ratiod_benchmark_compare handles n_groups default", {
  skip_on_cran()

  # Config without explicit n_groups should default to 0
  configs <- list(
    list(N = 20, p = 2)
  )

  result <- ratiod_benchmark_compare(
    configs,
    n_iter = 50,
    n_warmup = 25
  )

  expect_equal(result$n_groups, 0)
})

# -----------------------------------------------------------------------------
# Convert functions coverage - tested via integration tests
# -----------------------------------------------------------------------------

# Note: convert_pg_to_ratiod_fit is tested through integration tests
# in test-backends.R when running with backend = "pg"

# -----------------------------------------------------------------------------
# Error paths
# -----------------------------------------------------------------------------

test_that("fit_pg_binomial errors without trials", {
  mock_formula <- list(
    numerator = list(
      response = 1:5,
      X = cbind(1, rnorm(5)),
      random_effects = NULL
    ),
    denominator = list(response = NULL)
  )

  expect_error(
    tulpaRatio:::fit_pg_binomial(
      formula = mock_formula,
      data = data.frame(x = 1:5),
      family = ratiod_binomial(),
      verbose = FALSE
    ),
    "trials"
  )
})

test_that("prepare_spatial_for_pg errors without group_var for group level", {
  spatial <- list(level = "group", group_var = NULL, adjacency = diag(3))

  expect_error(
    tulpaRatio:::prepare_spatial_for_pg(spatial, data.frame(x = 1:3), list()),
    "group_var"
  )
})
