# Tests for alternative backends (PG and Laplace)

test_that("PG sampler produces valid PG(1, z) samples", {
  skip_on_cran()

  set.seed(123)

  # Test PG(1, 0) - should have mean ~0.25
  x <- tulpaRatio:::cpp_rpg1(rep(0, 5000))
  expect_true(all(x > 0))
  expect_equal(mean(x), 0.25, tolerance = 0.02)

  # Test PG(1, 2) - should have mean ~0.19
  x <- tulpaRatio:::cpp_rpg1(rep(2, 5000))
  mean_theory <- tanh(1) / 4  # tanh(z/2) / (2z)

  expect_equal(mean(x), mean_theory, tolerance = 0.02)
})

test_that("PG sampler works for PG(b, z)", {
  skip_on_cran()

  set.seed(456)

  # PG(5, 1) should be sum of 5 PG(1, 1)
  b <- rep(5L, 1000)
  z <- rep(1, 1000)
  x <- tulpaRatio:::cpp_rpg(b, z)

  expect_true(all(x > 0))
  # Mean should be approximately 5 * PG(1, 1) mean
  expected_mean <- 5 * tanh(0.5) / 2
  expect_equal(mean(x), expected_mean, tolerance = 0.1)
})

test_that("PG Gibbs sampler recovers parameters", {
  skip_on_cran()

  set.seed(42)

  # Generate data
  n_groups <- 10
  n_per_group <- 20
  N <- n_groups * n_per_group

  true_beta <- c(0.5, -0.3)
  true_sigma <- 0.6

  x <- rnorm(N)
  X <- cbind(1, x)
  group <- rep(1:n_groups, each = n_per_group)
  true_re <- rnorm(n_groups, 0, true_sigma)

  eta <- X %*% true_beta + true_re[group]
  trials <- rep(10L, N)
  y <- rbinom(N, trials, plogis(eta))

  # Fit
  fit <- tulpaRatio:::cpp_pg_binomial_gibbs(
    y = as.integer(y),
    n = trials,
    X = X,
    group = as.integer(group),
    n_groups = n_groups,
    n_iter = 500,
    n_warmup = 200,
    thin = 1,
    verbose = FALSE
  )

  # Check recovery (within reasonable tolerance for stochastic data)
  beta_post <- colMeans(fit$beta)
  expect_equal(beta_post[1], true_beta[1], tolerance = 0.5)
  expect_equal(beta_post[2], true_beta[2], tolerance = 0.3)

  # RE correlation should be high
  re_post <- colMeans(fit$re)
  expect_gt(cor(re_post, true_re), 0.8)
})

test_that("can_use_pg_backend correctly identifies supported families", {
  expect_true(tulpaRatio:::can_use_pg_backend(ratiod_binomial()))
  expect_true(tulpaRatio:::can_use_pg_backend(ratiod_negbin_negbin()))  # PG supports negbin via CRT augmentation
  expect_false(tulpaRatio:::can_use_pg_backend(ratiod_poisson_gamma()))
})

# ---------------------------------------------------------------------------
# PG backend helper function tests
# ---------------------------------------------------------------------------

test_that("extract_re_from_data handles no random effects", {
  # Create minimal formula object with no RE
  formula_obj <- list(
    numerator = list(
      random_effects = list(),
      response = 1:10
    )
  )
  df <- data.frame(y = 1:10, x = rnorm(10))

  result <- tulpaRatio:::extract_re_from_data(formula_obj, df)

  expect_equal(result$n_groups, 0L)
  expect_equal(result$n_re_terms, 0L)
  expect_equal(length(result$group_idx), 10)
  expect_false(result$has_slopes)
})

test_that("extract_re_from_data handles single random intercept", {
  n_per_group <- 5L
  n_groups <- 4L
  N <- n_per_group * n_groups

  # Create formula object with single RE
  formula_obj <- list(
    numerator = list(
      random_effects = list(
        list(
          group_var = "site",
          n_groups = n_groups,
          group = rep(1L:n_groups, each = n_per_group),
          slope_vars = character(0),
          has_intercept = TRUE
        )
      ),
      response = 1:N
    )
  )
  df <- data.frame(
    y = 1:N,
    x = rnorm(N),
    site = factor(rep(1:n_groups, each = n_per_group))
  )

  result <- tulpaRatio:::extract_re_from_data(formula_obj, df)

  expect_equal(result$n_groups, n_groups)
  expect_equal(result$n_re_terms, 1L)
  expect_equal(length(result$group_idx), N)
  expect_false(result$has_slopes)
})

test_that("extract_re_from_data handles multiple random effects", {
  n <- 20L

  # Create formula object with two RE terms
  formula_obj <- list(
    numerator = list(
      random_effects = list(
        list(
          group_var = "site",
          n_groups = 4L,
          group = rep(1L:4L, each = 5L),
          slope_vars = character(0),
          has_intercept = TRUE
        ),
        list(
          group_var = "year",
          n_groups = 5L,
          group = rep(1L:5L, 4L),
          slope_vars = character(0),
          has_intercept = TRUE
        )
      ),
      response = 1:n
    )
  )
  df <- data.frame(
    y = 1:n,
    site = factor(rep(1:4, each = 5)),
    year = factor(rep(1:5, 4))
  )

  result <- tulpaRatio:::extract_re_from_data(formula_obj, df)

  expect_equal(result$n_re_terms, 2L)
  expect_equal(result$total_groups, 9L)  # 4 + 5
  expect_true(!is.null(result$group_idx_matrix))
  expect_equal(ncol(result$group_idx_matrix), 2L)
})

test_that("extract_re_from_data warns about unsupported random slopes in PG", {
  n <- 20L

  # Create formula object with random slopes
  formula_obj <- list(
    numerator = list(
      random_effects = list(
        list(
          group_var = "site",
          n_groups = 4L,
          group = rep(1L:4L, each = 5L),
          slope_vars = c("x"),
          has_intercept = TRUE
        )
      ),
      response = 1:n
    )
  )
  df <- data.frame(
    y = 1:n,
    x = rnorm(n),
    site = factor(rep(1:4, each = 5))
  )

  expect_warning(
    result <- tulpaRatio:::extract_re_from_data(formula_obj, df),
    "Random slopes not yet fully supported"
  )

  expect_true(result$has_slopes)
})

test_that("prepare_spatial_for_pg handles group-level spatial", {
  n_sites <- 5
  n_per_site <- 4
  N <- n_sites * n_per_site

  # Create adjacency matrix (linear chain)
  adj <- matrix(0, n_sites, n_sites)
  adj[1, 2] <- adj[2, 1] <- 1
  adj[2, 3] <- adj[3, 2] <- 1
  adj[3, 4] <- adj[4, 3] <- 1
  adj[4, 5] <- adj[5, 4] <- 1

  spatial <- list(
    adjacency = adj,
    level = "group",
    group_var = "site"
  )

  df <- data.frame(
    y = 1:N,
    site = factor(rep(1:n_sites, each = n_per_site))
  )

  formula_obj <- list(
    numerator = list(
      random_effects = list()
    )
  )

  result <- tulpaRatio:::prepare_spatial_for_pg(spatial, df, formula_obj)

  expect_equal(result$n_units, n_sites)
  expect_equal(length(result$group_idx), N)
  expect_equal(length(result$adj_list), n_sites)
  expect_equal(result$n_neighbors[3], 2L)  # Middle site has 2 neighbors
})

test_that("prepare_spatial_for_pg handles observation-level spatial", {
  N <- 4

  # Simple 4-node adjacency
  adj <- matrix(0, 4, 4)
  adj[1, 2] <- adj[2, 1] <- 1
  adj[2, 3] <- adj[3, 2] <- 1
  adj[3, 4] <- adj[4, 3] <- 1

  spatial <- list(
    adjacency = adj,
    level = "obs",
    group_var = NULL
  )

  df <- data.frame(y = 1:4)
  formula_obj <- list(numerator = list(random_effects = list()))

  result <- tulpaRatio:::prepare_spatial_for_pg(spatial, df, formula_obj)

  expect_equal(result$n_units, 4)
  expect_equal(result$group_idx, 1:4)
})

test_that("prepare_spatial_for_pg errors without group_var for group level", {
  adj <- matrix(0, 3, 3)
  adj[1, 2] <- adj[2, 1] <- 1
  adj[2, 3] <- adj[3, 2] <- 1

  spatial <- list(
    adjacency = adj,
    level = "group",
    group_var = NULL
  )

  df <- data.frame(y = 1:6)
  formula_obj <- list(numerator = list(random_effects = list()))

  expect_error(
    tulpaRatio:::prepare_spatial_for_pg(spatial, df, formula_obj),
    "group_var"
  )
})

test_that("convert_pg_to_ratiod_fit creates valid ratiod_fit object", {
  # Create minimal chain result structure
  n_save <- 50
  p <- 2
  n_re <- 5
  N <- 20

  chain_result <- list(
    beta = matrix(rnorm(n_save * p), nrow = n_save, ncol = p),
    re = matrix(rnorm(n_save * n_re), nrow = n_save, ncol = n_re),
    eta = matrix(rnorm(n_save * N), nrow = n_save, ncol = N),
    sigma_re = abs(rnorm(n_save))
  )

  X <- cbind(1, rnorm(N))
  colnames(X) <- c("(Intercept)", "x")

  formula_obj <- list(
    numerator = list(
      random_effects = list(
        list(group_var = "site", n_groups = n_re, group = rep(1:n_re, length.out = N))
      )
    )
  )

  re_info <- list(
    group_idx = rep(1:n_re, length.out = N),
    n_groups = n_re,
    group_var = "site",
    n_re_terms = 1L,
    re_terms = list(list(
      group_var = "site",
      group_idx = rep(1:n_re, length.out = N),
      n_groups = n_re,
      offset = 0L
    ))
  )

  df <- data.frame(y = 1:N, x = rnorm(N), site = factor(rep(1:n_re, length.out = N)))
  family <- ratiod_binomial()

  fit <- tulpaRatio:::convert_pg_to_ratiod_fit(
    fit_raw = list(chain_result),  # Single chain
    formula = formula_obj,
    data = df,
    family = family,
    spatial = NULL,
    X = X,
    re_info = re_info,
    iter = 100,
    warmup = 50,
    thin = 1,
    chains = 1
  )

  expect_s3_class(fit, "ratiod_fit")
  expect_equal(fit$backend, "pg")
  expect_equal(fit$chains, 1)
  expect_equal(fit$n_save_per_chain, n_save)
  expect_true(is.array(fit$draws))
})
