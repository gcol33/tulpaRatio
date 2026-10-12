# Test random slopes support

test_that("formula parsing detects correlated vs uncorrelated slopes", {
  # Create test data
  set.seed(123)
  n <- 100
  df <- data.frame(
    y = rpois(n, 10),
    n = rep(50, n),
    x = rnorm(n),
    z = rnorm(n),
    site = rep(1:10, each = 10),
    year = rep(1:5, each = 20)
  )

  # Test correlated slope parsing (single bar)
  re1 <- tulpaRatio:::extract_random_effects("(1 + x | site)", df)
  expect_length(re1, 1)
  expect_equal(re1[[1]]$group_var, "site")
  expect_true(re1[[1]]$has_intercept)
  expect_equal(re1[[1]]$slope_vars, "x")
  expect_true(isTRUE(re1[[1]]$correlated))

  # Test uncorrelated slope parsing (double bar)
  re2 <- tulpaRatio:::extract_random_effects("(1 + x || site)", df)
  expect_length(re2, 1)
  expect_equal(re2[[1]]$group_var, "site")
  expect_true(re2[[1]]$has_intercept)
  expect_equal(re2[[1]]$slope_vars, "x")
  expect_false(isTRUE(re2[[1]]$correlated))

  # Test multiple slopes
  re3 <- tulpaRatio:::extract_random_effects("(1 + x + z | site)", df)
  expect_length(re3, 1)
  expect_equal(re3[[1]]$slope_vars, c("x", "z"))
  expect_true(isTRUE(re3[[1]]$correlated))

  # Test intercept only (should have correlated = TRUE by default, but no slopes)
  re4 <- tulpaRatio:::extract_random_effects("(1 | site)", df)
  expect_length(re4, 1)
  expect_true(re4[[1]]$has_intercept)
  expect_null(re4[[1]]$slope_vars)
})


test_that("formula expansion handles interactions (x*z)", {
  set.seed(456)
  n <- 100
  df <- data.frame(
    y = rpois(n, 10),
    n = rep(50, n),
    x = rnorm(n),
    z = rnorm(n),
    site = rep(1:10, each = 10)
  )

  # x*z should expand to x + z + x:z (3 slopes)
  re <- tulpaRatio:::extract_random_effects("(1 + x*z | site)", df)
  expect_length(re, 1)
  expect_true(re[[1]]$has_intercept)

  # Check that interaction was expanded
  expect_equal(length(re[[1]]$slope_vars), 3)
  expect_true("x" %in% re[[1]]$slope_vars)
  expect_true("z" %in% re[[1]]$slope_vars)
  expect_true("x:z" %in% re[[1]]$slope_vars)

  # Check slope_matrix was created with correct dimensions
  expect_equal(nrow(re[[1]]$slope_matrix), n)
  expect_equal(ncol(re[[1]]$slope_matrix), 3)

  # Check cleaned names
  expect_equal(length(re[[1]]$slope_vars_clean), 3)
  expect_true("x_z" %in% re[[1]]$slope_vars_clean)  # x:z -> x_z
})


test_that("formula expansion handles polynomials I(x^2)", {
  set.seed(789)
  n <- 100
  df <- data.frame(
    y = rpois(n, 10),
    n = rep(50, n),
    x = rnorm(n),
    site = rep(1:10, each = 10)
  )

  # I(x^2) should create a single squared term
  re <- tulpaRatio:::extract_random_effects("(1 + I(x^2) | site)", df)
  expect_length(re, 1)
  expect_true(re[[1]]$has_intercept)

  # Check slope was created
  expect_equal(length(re[[1]]$slope_vars), 1)
  expect_equal(re[[1]]$slope_vars, "I(x^2)")

  # Check slope_matrix contains squared values
  expect_equal(nrow(re[[1]]$slope_matrix), n)
  expect_equal(ncol(re[[1]]$slope_matrix), 1)
  expect_equal(as.numeric(re[[1]]$slope_matrix[, 1]), df$x^2)

  # Check cleaned name
  expect_equal(re[[1]]$slope_vars_clean, "x_pow2")
})


test_that("formula expansion handles x + I(x^2)", {
  set.seed(101)
  n <- 100
  df <- data.frame(
    y = rpois(n, 10),
    n = rep(50, n),
    x = rnorm(n),
    site = rep(1:10, each = 10)
  )

  # x + I(x^2) creates both linear and squared terms
  re <- tulpaRatio:::extract_random_effects("(1 + x + I(x^2) | site)", df)
  expect_length(re, 1)
  expect_equal(length(re[[1]]$slope_vars), 2)

  # Check slope_matrix dimensions
  expect_equal(ncol(re[[1]]$slope_matrix), 2)

  # Verify values (strip names from model.matrix output)
  expect_equal(as.numeric(re[[1]]$slope_matrix[, 1]), df$x)
  expect_equal(as.numeric(re[[1]]$slope_matrix[, 2]), df$x^2)
})


test_that("formula expansion handles poly(x, 2)", {
  set.seed(202)
  n <- 100
  df <- data.frame(
    y = rpois(n, 10),
    n = rep(50, n),
    x = rnorm(n),
    site = rep(1:10, each = 10)
  )

  # poly(x, 2) creates orthogonal polynomial terms
  re <- tulpaRatio:::extract_random_effects("(1 + poly(x, 2) | site)", df)
  expect_length(re, 1)

  # poly(x, 2) should expand to 2 columns
  expect_equal(length(re[[1]]$slope_vars), 2)
  expect_equal(ncol(re[[1]]$slope_matrix), 2)

  # Verify orthogonal polynomials match R's poly() (strip names for comparison)
  expected <- poly(df$x, 2)
  expect_equal(as.numeric(re[[1]]$slope_matrix[, 1]), as.numeric(expected[, 1]), tolerance = 1e-10)
  expect_equal(as.numeric(re[[1]]$slope_matrix[, 2]), as.numeric(expected[, 2]), tolerance = 1e-10)
})


test_that("expand_re_slopes handles edge cases", {
  n <- 50
  df <- data.frame(
    x = rnorm(n),
    z = rnorm(n),
    f = factor(sample(letters[1:3], n, replace = TRUE))
  )

  # Empty slopes
  result <- tulpaRatio:::expand_re_slopes(NULL, df)
  expect_equal(length(result$slope_vars), 0)
  expect_equal(ncol(result$slope_matrix), 0)

  # Single variable
  result <- tulpaRatio:::expand_re_slopes("x", df)
  expect_equal(result$slope_vars, "x")
  expect_equal(ncol(result$slope_matrix), 1)

  # Factor variable (creates dummy variables)
  result <- tulpaRatio:::expand_re_slopes("f", df)
  expect_true(length(result$slope_vars) >= 2)  # At least 2 dummy columns for 3-level factor
})


test_that("clean_slope_names produces expected output", {
  clean <- tulpaRatio:::clean_slope_names

  # Interactions
  expect_equal(clean("x:z"), "x_z")
  expect_equal(clean("a:b:c"), "a_b_c")

  # Powers
  expect_equal(clean("I(x^2)"), "x_pow2")
  expect_equal(clean("I(x^3)"), "x_pow3")

  # poly() columns
  expect_equal(clean("poly(x, 2)1"), "x_poly1")
  expect_equal(clean("poly(x, 2)2"), "x_poly2")

  # log/sqrt transformations
  expect_equal(clean("log(x)"), "log_x")
  expect_equal(clean("sqrt(x)"), "sqrt_x")

  # scale()
  expect_equal(clean("scale(x)"), "x_scaled")
})

test_that("random slopes model compiles and runs without error", {
  skip_on_cran()
  skip_if_not_installed("tulpaRatio")

  set.seed(42)
  n_sites <- 5
  n_per_site <- 10
  n <- n_sites * n_per_site

  # Simulate data with random slopes
  site <- rep(1:n_sites, each = n_per_site)
  x <- rnorm(n)

  # True random intercepts and slopes (uncorrelated for simplicity)
  true_intercept <- rnorm(n_sites, 0, 0.5)
  true_slope <- rnorm(n_sites, 0, 0.3)

  # Linear predictor with group-specific intercept and slope
  eta_num <- 2 + 0.5 * x + true_intercept[site] + true_slope[site] * x
  eta_denom <- 2.5 + true_intercept[site] + true_slope[site] * x

  y_num <- rpois(n, exp(eta_num))
  y_denom <- rpois(n, exp(eta_denom))

  df <- data.frame(
    y_num = y_num,
    y_denom = y_denom,
    x = x,
    site = factor(site)
  )

  # Fit model with uncorrelated random slopes (simpler, faster)
  # Use very short chain for testing
  fit <- tratio(
    y_num | y_denom ~ x + (1 + x || site),
    data = df,
    family = ratiod_negbin_negbin(),
    mode = "hmc",
    control = list(iter = 100, warmup = 50, chains = 1)
  )

  expect_s3_class(fit, "ratiod_fit")

  # Check that we have slope parameters in the output
  draws <- as.data.frame(fit$draws)
  param_names <- names(draws)

  # Should have sigma for intercept and slope
  expect_true(any(grepl("sigma_re\\[1,intercept\\]", param_names)))
  expect_true(any(grepl("sigma_re\\[1,x\\]", param_names)))

  # Should have RE for each group and coefficient
  expect_true(any(grepl("re\\[1,1,intercept\\]", param_names)))
  expect_true(any(grepl("re\\[1,1,x\\]", param_names)))
})

test_that("correlated random slopes model runs without error", {
  skip_on_cran()
  skip_if_not_installed("tulpaRatio")
  skip_if_not_installed("MASS")

  set.seed(42)
  n_sites <- 4
  n_per_site <- 15
  n <- n_sites * n_per_site

  site <- rep(1:n_sites, each = n_per_site)
  x <- rnorm(n)

  # Simulate with correlated intercept and slope
  # Covariance matrix: var_int=0.5, var_slope=0.3, cor=0.6
  sigma_int <- sqrt(0.5)
  sigma_slope <- sqrt(0.3)
  rho <- 0.6

  Sigma <- matrix(c(
    sigma_int^2, rho * sigma_int * sigma_slope,
    rho * sigma_int * sigma_slope, sigma_slope^2
  ), 2, 2)

  re <- MASS::mvrnorm(n_sites, c(0, 0), Sigma)
  true_intercept <- re[, 1]
  true_slope <- re[, 2]

  eta <- 2 + 0.5 * x + true_intercept[site] + true_slope[site] * x
  y <- rpois(n, exp(eta))
  trials <- rep(100, n)

  df <- data.frame(
    y = y,
    trials = trials,
    x = x,
    site = factor(site)
  )

  # Fit with correlated slopes (single bar)
  fit <- tratio(
    y | trials ~ x + (1 + x | site),
    data = df,
    family = ratiod_binomial(),
    mode = "hmc",
    control = list(iter = 100, warmup = 50, chains = 1)
  )

  expect_s3_class(fit, "ratiod_fit")

  # Check for Cholesky parameters
  draws <- as.data.frame(fit$draws)
  param_names <- names(draws)

  # The correlation of the intercept and the slope is reported
  expect_true("cor_re[1,intercept:x]" %in% param_names)
})

test_that("the correlation Cholesky map gives unit rows for any raw value", {
  raw <- rbind(c(0, 0, 0), c(3, -5, 8), c(-0.4, 1.2, 0.3))
  L <- tulpaRatio:::cpc_cholesky_draws(raw, 3)
  for (r in 1:3) {
    row_sq <- Reduce(`+`, lapply(seq_len(r), function(c) L[[r]][[c]]^2))
    expect_equal(row_sq, rep(1, 3))
  }
  expect_equal(L[[2]][[1]], tanh(raw[, 1]))
  expect_equal(L[[3]][[2]], tanh(raw[, 3]) * sqrt(1 - tanh(raw[, 2])^2))
})
