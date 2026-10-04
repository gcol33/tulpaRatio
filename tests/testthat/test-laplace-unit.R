# Unit tests for Laplace backend helper functions

# ---------------------------------------------------------------------------
# get_laplace_family tests
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# extract_re_for_laplace tests
# ---------------------------------------------------------------------------

test_that("extract_re_for_laplace with no random effects", {
  set.seed(111)
  n <- 10
  df <- data.frame(
    count = rpois(n, 5),
    effort = rgamma(n, 3, 1),
    x = rnorm(n)
  )

  f <- ratiod_formula(count | effort ~ x, data = df)
  re_info <- tulpaRatio:::extract_re_for_laplace(f)

  expect_equal(re_info$n_groups, 0L)
  expect_equal(re_info$n_re_terms, 0L)
  expect_null(re_info$group_var)
  expect_false(re_info$has_slopes)
  expect_equal(length(re_info$group_idx), n)
})

test_that("extract_re_for_laplace with single RE term", {
  set.seed(222)
  n <- 15
  df <- data.frame(
    count = rpois(n, 5),
    effort = rgamma(n, 3, 1),
    x = rnorm(n),
    site = factor(rep(1:3, each = 5))
  )

  f <- ratiod_formula(count | effort ~ x + (1 | site), data = df)
  re_info <- tulpaRatio:::extract_re_for_laplace(f)

  expect_equal(re_info$n_groups, 3L)
  expect_equal(re_info$n_re_terms, 1L)
  expect_equal(re_info$group_var, "site")
  expect_false(re_info$has_slopes)
  expect_equal(length(re_info$group_idx), n)
})

test_that("extract_re_for_laplace with slopes warns", {
  set.seed(333)
  n <- 15
  df <- data.frame(
    count = rpois(n, 5),
    effort = rgamma(n, 3, 1),
    x = rnorm(n),
    site = factor(rep(1:3, each = 5))
  )

  f <- ratiod_formula(count | effort ~ x + (x | site), data = df)

  expect_warning(
    re_info <- tulpaRatio:::extract_re_for_laplace(f),
    "Random slopes are not carried"
  )

  expect_true(re_info$has_slopes)
})

test_that("extract_re_for_laplace with multiple RE terms", {
  set.seed(444)
  n <- 18
  df <- data.frame(
    count = rpois(n, 5),
    effort = rgamma(n, 3, 1),
    x = rnorm(n),
    site = factor(rep(1:3, each = 6)),
    year = factor(rep(1:2, times = 9))
  )

  f <- ratiod_formula(count | effort ~ x + (1 | site) + (1 | year), data = df)
  re_info <- tulpaRatio:::extract_re_for_laplace(f)

  expect_equal(re_info$n_re_terms, 2L)
  expect_equal(re_info$total_groups, 5L)  # 3 sites + 2 years
  expect_true(!is.null(re_info$re_terms))
  expect_equal(length(re_info$re_terms), 2)
})

# ---------------------------------------------------------------------------
# can_use_laplace_backend tests
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# prepare_spatial_for_laplace tests
# ---------------------------------------------------------------------------

test_that("prepare_spatial_for_laplace extracts adjacency structure", {
  set.seed(555)
  n_sites <- 4
  n_per_site <- 3
  n <- n_sites * n_per_site

  adj <- matrix(0, n_sites, n_sites)
  for (i in 1:(n_sites - 1)) adj[i, i + 1] <- adj[i + 1, i] <- 1

  df <- data.frame(
    count = rpois(n, 5),
    effort = rgamma(n, 3, 1),
    site = factor(rep(1:n_sites, each = n_per_site))
  )

  spatial <- spatial_car(adj, level = "group", group_var = "site")

  # Need a formula for prepare_spatial_for_laplace
  formula <- ratiod_formula(count | effort ~ 1, data = df)

  result <- tulpaRatio:::prepare_spatial_for_laplace(spatial, df, formula)

  expect_equal(result$n_units, n_sites)
  expect_equal(length(result$group_idx), n)
  expect_true(all(result$group_idx >= 1 & result$group_idx <= n_sites))
  expect_equal(length(result$n_neighbors), n_sites)
})

test_that("prepare_spatial_for_laplace errors without group_var", {
  spatial <- list(adj_matrix = matrix(0, 3, 3))
  df <- data.frame(x = 1:10)
  formula <- list()

  expect_error(
    tulpaRatio:::prepare_spatial_for_laplace(spatial, df, formula),
    "group_var"
  )
})

test_that("prepare_spatial_for_laplace errors when group_var not in data", {
  spatial <- list(
    adj_matrix = matrix(0, 3, 3),
    group_var = "missing_var"
  )
  df <- data.frame(x = 1:10, site = factor(1:10))
  formula <- list()

  expect_error(
    tulpaRatio:::prepare_spatial_for_laplace(spatial, df, formula),
    "not found in data"
  )
})

test_that("prepare_spatial_for_laplace errors without adjacency matrix", {
  spatial <- list(group_var = "site")
  df <- data.frame(site = factor(1:5))
  formula <- list()

  expect_error(
    tulpaRatio:::prepare_spatial_for_laplace(spatial, df, formula),
    "adj_matrix"
  )
})

# ---------------------------------------------------------------------------
# compute_hessian_at_mode tests
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# convert_laplace_to_ratiod_fit tests
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# compute_hessian_spatial tests
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# convert_laplace_spatial_to_ratiod_fit tests
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# ratiod_compare error handling tests
# ---------------------------------------------------------------------------

test_that("ratiod_compare errors with single model", {
  fit <- list()
  class(fit) <- "ratiod_fit"

  expect_error(
    ratiod_compare(fit),
    "At least two models"
  )
})

test_that("ratiod_compare errors with non-ratiod objects", {
  expect_error(
    ratiod_compare(list(a = 1), list(b = 2)),
    "must be ratiod_fit"
  )
})

# ---------------------------------------------------------------------------
# ratiod_average error handling tests
# ---------------------------------------------------------------------------

test_that("ratiod_average errors with single model", {
  fit <- list()
  class(fit) <- "ratiod_fit"

  expect_error(
    ratiod_average(fit),
    "At least two models"
  )
})

test_that("ratiod_average errors with non-ratiod objects", {
  expect_error(
    ratiod_average(list(a = 1), list(b = 2)),
    "must be ratiod_fit"
  )
})

# ---------------------------------------------------------------------------
# print.ratiod_average tests
# ---------------------------------------------------------------------------

test_that("print.ratiod_average works", {
  avg_obj <- list(
    weights = c(m1 = 0.6, m2 = 0.4),
    predictions = data.frame(
      mean = 1:3,
      sd = 0.1
    ),
    models = c("m1", "m2"),
    type = "ratio",
    weights_method = "loo",
    n_models = 2
  )
  class(avg_obj) <- "ratiod_average"

  output <- capture.output(print(avg_obj))
  expect_true(any(grepl("ratio model averaging", output)))
  expect_true(any(grepl("Model weights", output)))
  expect_true(any(grepl("m1", output)))
})

test_that("print.ratiod_average handles matrix predictions", {
  avg_obj <- list(
    weights = c(m1 = 0.6, m2 = 0.4),
    predictions = matrix(1:20, nrow = 10, ncol = 2),
    models = c("m1", "m2"),
    type = "ratio",
    weights_method = "loo",
    n_models = 2
  )
  class(avg_obj) <- "ratiod_average"

  output <- capture.output(print(avg_obj))
  expect_true(any(grepl("draws x", output)))
})

# ---------------------------------------------------------------------------
# fitted.ratiod_average tests
# ---------------------------------------------------------------------------

test_that("fitted.ratiod_average extracts predictions", {
  preds <- data.frame(mean = 1:5, sd = 0.1)
  avg_obj <- list(predictions = preds)
  class(avg_obj) <- "ratiod_average"

  result <- fitted(avg_obj)
  expect_equal(result, preds)
})

# ---------------------------------------------------------------------------
# weights.ratiod_average tests
# ---------------------------------------------------------------------------

test_that("weights.ratiod_average extracts weights", {
  w <- c(m1 = 0.7, m2 = 0.3)
  avg_obj <- list(weights = w)
  class(avg_obj) <- "ratiod_average"

  result <- weights(avg_obj)
  expect_equal(result, w)
})
