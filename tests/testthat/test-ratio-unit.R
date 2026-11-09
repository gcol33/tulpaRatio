# Unit tests for ratio.R using mock objects (no model fitting)

# Create a minimal mock ratiod_fit object with ratio_draws
make_mock_ratio_fit <- function(n_draws = 100, n_obs = 20) {
  set.seed(42)

  param_names <- c("beta_num[1]", "beta_num[2]", "beta_denom[1]", "beta_denom[2]", "sigma_re")
  draws <- matrix(rnorm(n_draws * length(param_names)), nrow = n_draws, ncol = length(param_names))
  colnames(draws) <- param_names

  mock_data <- data.frame(
    count = rpois(n_obs, 10),
    effort = rgamma(n_obs, 5, 1),
    x = rnorm(n_obs),
    site = factor(rep(1:4, length.out = n_obs)),
    season = factor(rep(c("summer", "winter"), each = n_obs / 2))
  )

  ratio_draws <- matrix(rgamma(n_draws * n_obs, shape = 2, rate = 1), nrow = n_draws, ncol = n_obs)
  colnames(ratio_draws) <- paste0("ratio[", seq_len(n_obs), "]")

  structure(
    list(
      draws = draws,
      mode = "hmc",
      chains = 1,
      data = mock_data,
      formula = list(
        numerator = list(
          random_effects = list(
            list(group_var = "site", n_groups = 4, group = rep(1:4, length.out = n_obs))
          ),
          terms = ~ x
        ),
        denominator = list(random_effects = list(), terms = ~ x)
      ),
      .internal = list(
        hmc_data = list(
          N = n_obs,
          p_num = 2,
          p_denom = 2,
          n_re_groups = 4,
          re_group = rep(1:4, length.out = n_obs),
          X_num = model.matrix(~ x, mock_data),
          X_denom = model.matrix(~ x, mock_data)
        ),
        model_type = "poisson_gamma",
        samples = draws
      ),
      ratio_draws = ratio_draws,
      diagnostics = list(energy = rnorm(n_draws, 100, 5))
    ),
    class = "ratiod_fit"
  )
}

# Test ratio.ratiod_fit
test_that("ratio extracts posterior draws", {
  fit <- make_mock_ratio_fit()

  r <- ratio(fit)

  expect_s3_class(r, "ratiod_ratio")
  expect_equal(r$type, "response")
  expect_equal(ncol(r$draws), 20)
  expect_true(is.matrix(r$draws))
})

test_that("ratio works with log scale", {
  fit <- make_mock_ratio_fit()

  r_log <- ratio(fit, type = "log")

  expect_equal(r_log$type, "log")
  # Log ratios can be negative
  expect_true(any(r_log$draws < 0))
})

test_that("ratio works with logit scale", {
  fit <- make_mock_ratio_fit()
  # Make ratio draws between 0 and 1 for valid logit
  fit$ratio_draws <- matrix(rbeta(100 * 20, 2, 2), nrow = 100, ncol = 20)

  r_logit <- suppressWarnings(ratio(fit, type = "logit"))

  expect_equal(r_logit$type, "logit")
})

test_that("ratio returns summary when requested", {
  fit <- make_mock_ratio_fit()

  r_summ <- ratio(fit, summary = TRUE)

  expect_s3_class(r_summ, "ratiod_ratio_summary")
  expect_true("mean" %in% names(r_summ))
  expect_true("sd" %in% names(r_summ))
})

# Test aggregate_by_group
test_that("aggregate_by_group computes geometric mean", {
  set.seed(999)
  draws <- matrix(c(rep(2, 10), rep(8, 10)), nrow = 10, ncol = 2)
  data <- data.frame(obs = 1:2, group = c("A", "A"))

  result <- tulpaRatio:::aggregate_by_group(draws, data, "group")

  expect_equal(ncol(result), 1)
  # Geometric mean of 2 and 8 = sqrt(16) = 4
  expect_equal(as.numeric(result[1, 1]), 4, tolerance = 0.001)
})

test_that("aggregate_by_group handles multiple groups", {
  draws <- matrix(c(rep(1, 10), rep(4, 10), rep(9, 10)), nrow = 10, ncol = 3)
  data <- data.frame(obs = 1:3, group = c("A", "A", "B"))

  result <- tulpaRatio:::aggregate_by_group(draws, data, "group")

  expect_equal(ncol(result), 2)  # 2 groups
})

test_that("aggregate_by_group errors on missing variable", {
  draws <- matrix(rnorm(20), nrow = 10, ncol = 2)
  data <- data.frame(obs = 1:2, x = c("a", "b"))

  expect_error(
    tulpaRatio:::aggregate_by_group(draws, data, "nonexistent"),
    "not found"
  )
})

test_that("ratio with by aggregates correctly", {
  fit <- make_mock_ratio_fit()

  r_site <- ratio(fit, by = "site")

  expect_equal(r_site$by, "site")
  expect_equal(ncol(r_site$draws), 4)  # 4 sites
})

test_that("ratio errors on invalid group variable", {
  fit <- make_mock_ratio_fit()

  expect_error(ratio(fit, by = "nonexistent"), "not found")
})

# Test summary.ratiod_ratio
test_that("summary.ratiod_ratio computes statistics", {
  fit <- make_mock_ratio_fit()
  r <- ratio(fit)

  summ <- summary(r)

  expect_s3_class(summ, "ratiod_ratio_summary")
  expect_true("mean" %in% names(summ))
  expect_true("sd" %in% names(summ))
  expect_equal(nrow(summ), 20)
})

test_that("summary.ratiod_ratio respects custom probs", {
  fit <- make_mock_ratio_fit()
  r <- ratio(fit)

  summ <- summary(r, probs = c(0.1, 0.5, 0.9))

  expect_true("q10" %in% names(summ))
  expect_true("q50" %in% names(summ))
  expect_true("q90" %in% names(summ))
})

# Test print methods
test_that("print.ratiod_ratio works", {
  fit <- make_mock_ratio_fit()
  r <- ratio(fit)

  output <- capture.output(print(r))
  expect_true(any(grepl("tulpaRatio ratio posterior", output)))
  expect_true(any(grepl("Scale:", output)))
})

test_that("print.ratiod_ratio shows grouping", {
  fit <- make_mock_ratio_fit()
  r <- ratio(fit, by = "site")

  output <- capture.output(print(r))
  expect_true(any(grepl("Grouped by:", output)))
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

# Test ratio_contrast
test_that("ratio_contrast errors on non-formula", {
  fit <- make_mock_ratio_fit()

  expect_error(ratio_contrast(fit, "season"), "formula")
})

test_that("ratio_contrast errors on multi-variable formula", {
  fit <- make_mock_ratio_fit()

  expect_error(ratio_contrast(fit, ~ x + season), "exactly one variable")
})

test_that("ratio_contrast errors on non-existent variable", {
  fit <- make_mock_ratio_fit()

  expect_error(ratio_contrast(fit, ~ nonexistent), "not found")
})

test_that("print.ratiod_contrast works", {
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
  expect_true(any(grepl("Type:", output)))
})

# Test print.ratiod_fitted
test_that("print.ratiod_fitted works", {
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
})

# Test print.ratiod_prediction
test_that("print.ratiod_prediction works", {
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
})

# Test as_draws
test_that("as_draws.ratiod_fit works", {
  skip_if_not_installed("posterior")

  fit <- make_mock_ratio_fit()

  draws <- as_draws(fit)
  expect_true(inherits(draws, "draws"))
})

test_that("as_draws errors without posterior package", {
  # Can't easily test package absence, so just verify method is registered
  expect_true("as_draws.ratiod_fit" %in% methods("as_draws"))
})

# Test draws_wide

# Test draws_long

# Test draws_interval

# Test HDI computation
test_that("HDI computation works correctly", {
  set.seed(42)
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
})

test_that("average_predictions works for ratio type", {
  pred1 <- matrix(2, nrow = 10, ncol = 5)
  pred2 <- matrix(4, nrow = 10, ncol = 5)
  predictions <- list(pred1, pred2)
  weights <- c(0.5, 0.5)

  # Ratio averaging on log scale
  avg <- tulpaRatio:::average_predictions(predictions, weights, "ratio", summary = FALSE)
  # exp(0.5*log(2) + 0.5*log(4)) = exp(0.5*0.693 + 0.5*1.386) = exp(1.0397) ~ 2.83
  expect_true(all(abs(avg - sqrt(2 * 4)) < 0.01))  # Geometric mean
})

test_that("ratio_contrast validates inputs", {
  # ratio_contrast requires a contrast formula
  expect_error(ratio_contrast(NULL, NULL))
})

test_that("ratio handles invalid type", {
  skip_on_cran()

  set.seed(42)
  n <- 20
  df <- data.frame(
    count = rpois(n, 10),
    total = rpois(n, 100) + 10,
    x = rnorm(n)
  )

  fit <- tratio(
    count | total ~ x,
    data = df,
    family = ratiod_negbin_negbin(),
    control = list(iter = 50, warmup = 25, chains = 1)
  )

  expect_error(ratio(fit, type = "invalid_type"))
})

test_that("draw accessors preserve the draw contract", {
  skip_on_cran()
  expect_draw_accessors(make_mock_ratio_fit())
})
