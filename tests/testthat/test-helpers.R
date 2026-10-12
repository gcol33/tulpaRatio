# Tests for helper functions (pure unit tests - no model fitting)

# Test select_main_params
test_that("select_main_params excludes high-dimensional parameters", {
  all_pars <- c(
    "beta_num[1]", "beta_num[2]",
    "beta_denom[1]", "beta_denom[2]",
    "sigma_re", "phi_dispersion",
    "re[1]", "re[2]", "re[3]", "re[4]", "re[5]",
    "spatial[1]", "spatial[2]",
    "phi_spatial[1]", "phi_scaled[1]",
    "theta[1]", "theta[2]"
  )

  result <- tulpaRatio:::select_main_params(all_pars)

  # Should include main params
  expect_true("beta_num[1]" %in% result)
  expect_true("sigma_re" %in% result)

  # Should exclude high-dimensional params
  expect_false("re[1]" %in% result)
  expect_false("spatial[1]" %in% result)
  expect_false("phi_spatial[1]" %in% result)
  expect_false("theta[1]" %in% result)
})

test_that("select_main_params keeps every low-dimensional parameter", {
  all_pars <- c(paste0("param", 1:20), "re[1]", "spatial[2]")
  result <- tulpaRatio:::select_main_params(all_pars)

  expect_identical(result, paste0("param", 1:20))
})

test_that("select_main_params handles empty input", {
  result <- tulpaRatio:::select_main_params(character(0))
  expect_equal(length(result), 0)
})

# Test grep_params
test_that("grep_params selects exact matches", {
  all_pars <- c("beta_num[1]", "beta_num[2]", "beta_denom[1]", "sigma_re")
  result <- tulpaRatio:::grep_params("sigma_re", all_pars)

  expect_equal(result, "sigma_re")
})

test_that("grep_params selects regex matches", {
  all_pars <- c("beta_num[1]", "beta_num[2]", "beta_denom[1]", "sigma_re")
  result <- tulpaRatio:::grep_params("beta_num", all_pars)

  expect_true("beta_num[1]" %in% result)
  expect_true("beta_num[2]" %in% result)
  expect_false("beta_denom[1]" %in% result)
})

test_that("grep_params handles multiple patterns", {
  all_pars <- c("beta_num[1]", "beta_denom[1]", "sigma_re", "phi")
  result <- tulpaRatio:::grep_params(c("beta", "sigma"), all_pars)

  expect_true("beta_num[1]" %in% result)
  expect_true("beta_denom[1]" %in% result)
  expect_true("sigma_re" %in% result)
  expect_false("phi" %in% result)
})

test_that("grep_params returns unique values", {
  all_pars <- c("beta_num[1]", "beta_num[2]")
  result <- tulpaRatio:::grep_params(c("beta_num[1]", "beta_num"), all_pars)

  expect_equal(length(result), 2)
  expect_equal(length(unique(result)), 2)
})

# Test print methods with mock objects
test_that("print.ratiod_ratio works with mock object", {
  mock_ratio <- structure(
    list(
      draws = matrix(rnorm(100), nrow = 10, ncol = 10),
      type = "response",
      by = NULL,
      n_obs = 10,
      n_draws = 10,
      data = data.frame(x = 1:10)
    ),
    class = "ratiod_ratio"
  )

  output <- capture.output(print(mock_ratio))
  expect_true(any(grepl("tulpaRatio ratio posterior", output)))
  expect_true(any(grepl("Scale: response", output)))
  expect_true(any(grepl("Observations: 10", output)))
  expect_true(any(grepl("Posterior draws: 10", output)))
})

test_that("print.ratiod_ratio shows grouping variable", {
  mock_ratio <- structure(
    list(
      draws = matrix(rnorm(30), nrow = 10, ncol = 3),
      type = "log",
      by = "site",
      n_obs = 3,
      n_draws = 10,
      data = data.frame(site = rep(c("A", "B", "C"), each = 3))
    ),
    class = "ratiod_ratio"
  )

  output <- capture.output(print(mock_ratio))
  expect_true(any(grepl("Grouped by: site", output)))
})

test_that("summary.ratiod_ratio computes statistics", {
  set.seed(123)
  draws <- matrix(rnorm(500, mean = 2, sd = 0.5), nrow = 50, ncol = 10)
  colnames(draws) <- paste0("ratio[", 1:10, "]")

  mock_ratio <- structure(
    list(
      draws = draws,
      type = "response",
      by = NULL,
      n_obs = 10,
      n_draws = 50,
      data = data.frame(x = 1:10)
    ),
    class = "ratiod_ratio"
  )

  summ <- summary(mock_ratio)

  expect_s3_class(summ, "ratiod_ratio_summary")
  expect_true("mean" %in% names(summ))
  expect_true("sd" %in% names(summ))
  expect_true("q2.5" %in% names(summ))
  expect_true("q50" %in% names(summ))
  expect_true("q97.5" %in% names(summ))
  expect_equal(nrow(summ), 10)

  # Check that means are approximately correct
  expect_equal(as.numeric(summ$mean), as.numeric(colMeans(draws)), tolerance = 0.001)
})

test_that("summary.ratiod_ratio respects custom probs", {
  draws <- matrix(rnorm(100), nrow = 10, ncol = 10)
  colnames(draws) <- paste0("ratio[", 1:10, "]")

  mock_ratio <- structure(
    list(
      draws = draws,
      type = "response",
      by = NULL,
      n_obs = 10,
      n_draws = 10,
      data = data.frame(x = 1:10)
    ),
    class = "ratiod_ratio"
  )

  summ <- summary(mock_ratio, probs = c(0.1, 0.9))

  expect_true("q10" %in% names(summ))
  expect_true("q90" %in% names(summ))
  expect_false("q2.5" %in% names(summ))
})

test_that("summary.ratiod_ratio handles grouped data", {
  draws <- matrix(rnorm(30), nrow = 10, ncol = 3)
  colnames(draws) <- paste0("ratio[", c("A", "B", "C"), "]")

  mock_ratio <- structure(
    list(
      draws = draws,
      type = "log",
      by = "site",
      n_obs = 3,
      n_draws = 10,
      data = data.frame(site = rep(c("A", "B", "C"), each = 3))
    ),
    class = "ratiod_ratio"
  )

  summ <- summary(mock_ratio)

  expect_true("group" %in% names(summ))
  expect_equal(nrow(summ), 3)
})

test_that("print.ratiod_ratio_summary works", {
  summ <- structure(
    data.frame(
      obs = 1:5,
      mean = c(1.2, 1.5, 1.8, 2.0, 2.2),
      sd = rep(0.1, 5),
      q2.5 = c(1.0, 1.3, 1.6, 1.8, 2.0),
      q97.5 = c(1.4, 1.7, 2.0, 2.2, 2.4)
    ),
    type = "response",
    n_draws = 100,
    class = c("ratiod_ratio_summary", "data.frame")
  )

  output <- capture.output(print(summ))
  expect_true(any(grepl("tulpaRatio ratio summary", output)))
})

test_that("print.ratiod_ratio_summary truncates long output", {
  summ <- structure(
    data.frame(
      obs = 1:20,
      mean = rnorm(20, mean = 1.5),
      sd = rep(0.1, 20),
      q2.5 = rnorm(20, mean = 1.3),
      q97.5 = rnorm(20, mean = 1.7)
    ),
    type = "response",
    n_draws = 100,
    class = c("ratiod_ratio_summary", "data.frame")
  )

  output <- capture.output(print(summ, n = 5))
  expect_true(any(grepl("and 15 more rows", output)))
})

test_that("print.ratiod_contrast works with mock object", {
  mock_contrast <- structure(
    data.frame(
      contrast = "winter vs summer",
      mean = 0.5,
      sd = 0.1,
      q2.5 = 0.3,
      q50 = 0.5,
      q97.5 = 0.7,
      prob_positive = 0.99
    ),
    type = "difference",
    contrast_var = "season",
    ref = "summer",
    class = c("ratiod_contrast", "data.frame")
  )

  output <- capture.output(print(mock_contrast))
  expect_true(any(grepl("tulpaRatio ratio contrasts", output)))
  expect_true(any(grepl("Type: difference", output)))
  expect_true(any(grepl("Variable: season", output)))
  expect_true(any(grepl("Reference: summer", output)))
})

test_that("print.ratiod_fitted works with mock object", {
  mock_fitted <- structure(
    data.frame(
      obs = rep(1:3, 3),
      component = rep(c("numerator", "denominator", "ratio"), each = 3),
      mean = rnorm(9, mean = 10),
      sd = rep(1, 9),
      q2.5 = rnorm(9, mean = 8),
      q97.5 = rnorm(9, mean = 12)
    ),
    n_draws = 100,
    class = c("ratiod_fitted", "data.frame")
  )

  output <- capture.output(print(mock_fitted))
  expect_true(any(grepl("tulpaRatio fitted values", output)))
  expect_true(any(grepl("Posterior draws: 100", output)))
})

test_that("print.ratiod_prediction works with mock object", {
  mock_pred <- structure(
    data.frame(
      obs = 1:3,
      component = "ratio",
      mean = c(1.2, 1.5, 1.8),
      sd = c(0.1, 0.15, 0.2),
      q2.5 = c(1.0, 1.2, 1.4),
      q97.5 = c(1.4, 1.8, 2.2)
    ),
    n_draws = 100,
    newdata = data.frame(x = c(-1, 0, 1)),
    class = c("ratiod_prediction", "data.frame")
  )

  output <- capture.output(print(mock_pred))
  expect_true(any(grepl("tulpaRatio predictions", output)))
  expect_true(any(grepl("Posterior draws: 100", output)))
  expect_true(any(grepl("New observations: 3", output)))
})

# Test aggregate_by_group
test_that("aggregate_by_group computes geometric mean", {
  set.seed(999)
  # Two observations, 10 draws each
  draws <- matrix(
    c(rep(2, 10), rep(8, 10)),
    nrow = 10, ncol = 2
  )
  data <- data.frame(obs = 1:2, group = c("A", "A"))

  result <- tulpaRatio:::aggregate_by_group(draws, data, "group")

  # Geometric mean of 2 and 8 = sqrt(16) = 4
  expect_equal(ncol(result), 1)
  expect_equal(as.numeric(result[1, 1]), 4, tolerance = 0.001)
})

test_that("aggregate_by_group handles multiple groups", {
  set.seed(123)
  draws <- matrix(
    c(rep(1, 10), rep(4, 10), rep(9, 10)),
    nrow = 10, ncol = 3
  )
  data <- data.frame(obs = 1:3, group = c("A", "A", "B"))

  result <- tulpaRatio:::aggregate_by_group(draws, data, "group")

  expect_equal(ncol(result), 2)  # 2 groups

  # Group A: geometric mean of 1 and 4 = 2
  idx_A <- which(grepl("A", colnames(result)))
  expect_equal(as.numeric(result[1, idx_A]), 2, tolerance = 0.001)

  # Group B: just 9
  idx_B <- which(grepl("B", colnames(result)))
  expect_equal(as.numeric(result[1, idx_B]), 9, tolerance = 0.001)
})

test_that("aggregate_by_group errors on missing variable", {
  draws <- matrix(rnorm(20), nrow = 10, ncol = 2)
  data <- data.frame(obs = 1:2, x = c("a", "b"))

  expect_error(
    tulpaRatio:::aggregate_by_group(draws, data, "nonexistent"),
    "not found"
  )
})

test_that("aggregate_by_group handles single observation per group", {
  draws <- matrix(c(rep(5, 10), rep(10, 10)), nrow = 10, ncol = 2)
  data <- data.frame(obs = 1:2, group = c("A", "B"))

  result <- tulpaRatio:::aggregate_by_group(draws, data, "group")

  expect_equal(ncol(result), 2)
  # Single observation means no averaging
  expect_equal(as.numeric(result[1, 1]), 5, tolerance = 0.001)
  expect_equal(as.numeric(result[1, 2]), 10, tolerance = 0.001)
})

# Test validation functions
test_that("validate_response accepts valid count data", {
  valid_counts <- c(0L, 1L, 5L, 10L, 100L)

  expect_silent(tulpaRatio:::validate_response(valid_counts, "poisson", "test"))
  expect_silent(tulpaRatio:::validate_response(valid_counts, "neg_binomial_2", "test"))
})

test_that("validate_response rejects negative counts", {
  invalid_counts <- c(-1, 0, 5, 10)

  expect_error(
    tulpaRatio:::validate_response(invalid_counts, "poisson", "numerator"),
    "non-negative"
  )
})

test_that("validate_response accepts valid gamma data", {
  valid_gamma <- c(0.1, 1.5, 2.3, 10.0)

  expect_silent(tulpaRatio:::validate_response(valid_gamma, "gamma", "denominator"))
})

test_that("validate_response rejects non-positive gamma data", {
  invalid_gamma <- c(0, 1.5, 2.3)

  expect_error(
    tulpaRatio:::validate_response(invalid_gamma, "gamma", "denominator"),
    "positive"
  )
})

# Test family validation
test_that("family objects have required components", {
  fam_pg <- ratiod_poisson_gamma()
  expect_true(is.list(fam_pg))
  expect_equal(fam_pg$name, "poisson_gamma")
  expect_equal(fam_pg$numerator$distribution, "poisson")
  expect_equal(fam_pg$denominator$distribution, "gamma")

  fam_nb <- ratiod_negbin_negbin()
  expect_equal(fam_nb$name, "negbin_negbin")
  expect_equal(fam_nb$numerator$distribution, "neg_binomial_2")
  expect_equal(fam_nb$denominator$distribution, "neg_binomial_2")

  fam_bin <- ratiod_binomial()
  expect_equal(fam_bin$name, "binomial_fixed")
})

# Test HDI computation (from draws_interval)
test_that("HDI computation works correctly", {
  set.seed(42)
  # Test with normal distribution - HDI should be centered around mean
  x <- rnorm(10000, mean = 5, sd = 1)

  # Simple HDI implementation test
  w <- 0.95
  sorted <- sort(x)
  n <- length(sorted)
  ci_n <- ceiling(w * n)
  widths <- sorted[(ci_n + 1):n] - sorted[1:(n - ci_n)]
  min_idx <- which.min(widths)
  lower <- sorted[min_idx]
  upper <- sorted[min_idx + ci_n]

  # HDI should be approximately 5 +/- 2 for 95% interval of N(5,1)
  expect_true(lower > 2 && lower < 4)
  expect_true(upper > 6 && upper < 8)
  expect_true((upper - lower) < 4.5)  # Width should be less than 4*sd
})
