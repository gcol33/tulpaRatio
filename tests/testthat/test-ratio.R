# Tests for ratio extraction functions (R/ratio.R)

# Helper to create a minimal fitted model for testing
make_test_fit <- function() {
  set.seed(12345)
  n <- 30
  n_groups <- 3
  df <- data.frame(
    count = rpois(n, lambda = 10),
    effort = rgamma(n, shape = 5, rate = 1),
    x = rnorm(n),
    season = factor(rep(c("summer", "winter"), each = n / 2)),
    site = factor(rep(1:n_groups, each = n / n_groups))
  )

  fit <- tratio(
    count | effort ~ x + (1 | site),
    data = df,
    family = ratiod_poisson_gamma(),
    mode = "hmc",
    control = list(iter = 100, warmup = 50, chains = 1)
  )
  fit
}

test_that("ratio extracts posterior draws", {
  skip_on_cran()

  fit <- make_test_fit()
  r <- ratio(fit)

  expect_s3_class(r, "ratiod_ratio")
  expect_equal(r$type, "response")
  expect_equal(r$n_obs, nrow(fit$data))
  expect_true(r$n_draws > 0)
  expect_true(is.matrix(r$draws))
  expect_equal(ncol(r$draws), r$n_obs)
})

test_that("ratio works with log scale", {
  skip_on_cran()

  fit <- make_test_fit()
  r_log <- ratio(fit, type = "log")

  expect_equal(r_log$type, "log")

  # Log scale should have negative values for ratios < 1
  r_resp <- ratio(fit, type = "response")
  expect_true(any(r_log$draws < 0) || all(r_resp$draws > 1))
})

test_that("ratio aggregates by group", {
  skip_on_cran()

  fit <- make_test_fit()
  r_site <- ratio(fit, by = "site")

  expect_equal(r_site$by, "site")
  expect_equal(r_site$n_obs, length(unique(fit$data$site)))
  expect_equal(ncol(r_site$draws), 3)  # 3 sites
})

test_that("ratio errors on invalid group variable", {
  skip_on_cran()

  fit <- make_test_fit()

  expect_error(
    ratio(fit, by = "nonexistent"),
    "not found"
  )
})

test_that("ratio returns summary when requested", {
  skip_on_cran()

  fit <- make_test_fit()
  r_summ <- ratio(fit, summary = TRUE)

  expect_s3_class(r_summ, "ratiod_ratio_summary")
  expect_true("mean" %in% names(r_summ))
  expect_true("sd" %in% names(r_summ))
  expect_true("q2.5" %in% names(r_summ))
  expect_true("q97.5" %in% names(r_summ))
})

test_that("summary.ratiod_ratio works", {
  skip_on_cran()

  fit <- make_test_fit()
  r <- ratio(fit)
  summ <- summary(r, probs = c(0.1, 0.5, 0.9))

  expect_s3_class(summ, "ratiod_ratio_summary")
  expect_true("q10" %in% names(summ))
  expect_true("q50" %in% names(summ))
  expect_true("q90" %in% names(summ))
  expect_equal(nrow(summ), nrow(fit$data))
})

test_that("print.ratiod_ratio works", {
  skip_on_cran()

  fit <- make_test_fit()
  r <- ratio(fit)

  output <- capture.output(print(r))
  expect_true(any(grepl("tulpaRatio ratio posterior", output)))
  expect_true(any(grepl("Scale:", output)))
  expect_true(any(grepl("Observations:", output)))
})

test_that("print.ratiod_ratio_summary works", {
  skip_on_cran()

  fit <- make_test_fit()
  r <- ratio(fit, summary = TRUE)

  output <- capture.output(print(r))
  expect_true(any(grepl("tulpaRatio ratio summary", output)))
})

test_that("ratio_contrast computes differences", {
  skip_on_cran()

  set.seed(54321)
  n <- 40
  df <- data.frame(
    count = rpois(n, lambda = 10),
    effort = rgamma(n, shape = 5, rate = 1),
    season = factor(rep(c("summer", "winter"), each = n / 2))
  )

  fit <- tratio(
    count | effort ~ season,
    data = df,
    family = ratiod_poisson_gamma(),
    mode = "hmc",
    control = list(iter = 100, warmup = 50, chains = 1)
  )

  contrast <- ratio_contrast(fit, ~ season)

  expect_s3_class(contrast, "ratiod_contrast")
  expect_true("contrast" %in% names(contrast))
  expect_true("mean" %in% names(contrast))
  expect_true("prob_positive" %in% names(contrast))
  expect_equal(nrow(contrast), 1)  # one contrast (winter vs summer)
})

test_that("ratio_contrast computes ratio of ratios", {
  skip_on_cran()

  set.seed(54321)
  n <- 40
  df <- data.frame(
    count = rpois(n, lambda = 10),
    effort = rgamma(n, shape = 5, rate = 1),
    season = factor(rep(c("summer", "winter"), each = n / 2))
  )

  fit <- tratio(
    count | effort ~ season,
    data = df,
    family = ratiod_poisson_gamma(),
    mode = "hmc",
    control = list(iter = 100, warmup = 50, chains = 1)
  )

  contrast <- ratio_contrast(fit, ~ season, type = "ratio")

  expect_equal(attr(contrast, "type"), "ratio")
  # Ratio of ratios should be positive
  expect_true(contrast$mean[1] > 0)
})

test_that("ratio_contrast errors on invalid inputs", {
  skip_on_cran()

  fit <- make_test_fit()

  # Non-formula
  expect_error(
    ratio_contrast(fit, "season"),
    "formula"
  )

  # Multi-variable formula
  expect_error(
    ratio_contrast(fit, ~ x + season),
    "exactly one variable"
  )

  # Non-existent variable
  expect_error(
    ratio_contrast(fit, ~ nonexistent),
    "not found"
  )
})

test_that("print.ratiod_contrast works", {
  skip_on_cran()

  set.seed(11111)
  n <- 40
  df <- data.frame(
    count = rpois(n, lambda = 10),
    effort = rgamma(n, shape = 5, rate = 1),
    season = factor(rep(c("summer", "winter"), each = n / 2))
  )

  fit <- tratio(
    count | effort ~ season,
    data = df,
    family = ratiod_poisson_gamma(),
    mode = "hmc",
    control = list(iter = 100, warmup = 50, chains = 1)
  )

  contrast <- ratio_contrast(fit, ~ season)
  output <- capture.output(print(contrast))

  expect_true(any(grepl("tulpaRatio ratio contrasts", output)))
  expect_true(any(grepl("Type:", output)))
  expect_true(any(grepl("Reference:", output)))
})

test_that("fitted.ratiod_fit returns fitted values", {
  skip_on_cran()

  fit <- make_test_fit()
  fv <- fitted(fit)

  expect_s3_class(fv, "ratiod_fitted")
  expect_true("component" %in% names(fv))
  expect_true("mean" %in% names(fv))
  expect_true("sd" %in% names(fv))

  # Should have all three components
  components <- unique(fv$component)
  expect_true("numerator" %in% components)
  expect_true("denominator" %in% components)
  expect_true("ratio" %in% components)
})

test_that("fitted.ratiod_fit filters by component", {
  skip_on_cran()

  fit <- make_test_fit()

  fv_num <- fitted(fit, component = "numerator")
  expect_true(all(fv_num$component == "numerator"))

  fv_ratio <- fitted(fit, component = "ratio")
  expect_true(all(fv_ratio$component == "ratio"))
})

test_that("fitted.ratiod_fit returns draws when summary=FALSE", {
  skip_on_cran()

  fit <- make_test_fit()
  fv_draws <- fitted(fit, summary = FALSE)

  expect_s3_class(fv_draws, "ratiod_fitted_draws")
  expect_true(is.list(fv_draws))
  expect_true("numerator" %in% names(fv_draws))
  expect_true("denominator" %in% names(fv_draws))
  expect_true("ratio" %in% names(fv_draws))
  expect_true(is.matrix(fv_draws$ratio))
})

test_that("print.ratiod_fitted works", {
  skip_on_cran()

  fit <- make_test_fit()
  fv <- fitted(fit)

  output <- capture.output(print(fv))
  expect_true(any(grepl("tulpaRatio fitted values", output)))
  expect_true(any(grepl("Posterior draws:", output)))
})

test_that("as_draws.ratiod_fit works", {
  skip_on_cran()
  skip_if_not_installed("posterior")

  fit <- make_test_fit()
  draws <- as_draws(fit)

  expect_true(inherits(draws, "draws_df") || inherits(draws, "draws"))
})

test_that("aggregate_by_group uses geometric mean", {
  # Test the internal aggregation function
  set.seed(999)
  draws <- matrix(
    c(rep(2, 10), rep(8, 10)),  # Two observations, 10 draws each
    nrow = 10, ncol = 2
  )
  data <- data.frame(obs = 1:2, group = c("A", "A"))

  result <- tulpaRatio:::aggregate_by_group(draws, data, "group")

  # Geometric mean of 2 and 8 = sqrt(16) = 4
  expect_equal(ncol(result), 1)
  expect_equal(as.numeric(result[1, 1]), 4, tolerance = 0.001)
})

# Additional tests for more coverage

test_that("ratio with logit scale warns for non-proportion ratios", {
  skip_on_cran()

  set.seed(12345)
  n <- 20
  df <- data.frame(
    count = rpois(n, lambda = 15),  # Large counts
    effort = rgamma(n, shape = 2, rate = 1),  # Small effort = high ratio
    x = rnorm(n)
  )

  fit <- tratio(
    count | effort ~ x,
    data = df,
    family = ratiod_poisson_gamma(),
    mode = "hmc",
    control = list(iter = 100, warmup = 50, chains = 1, verbose = FALSE)
  )

  # Should warn since ratios will be > 1
  # Wrap in suppressWarnings to handle multiple warnings (NaN and "not in (0,1)")
  suppressWarnings(expect_warning(ratio(fit, type = "logit")))
})

test_that("print.ratiod_ratio with by grouping", {
  skip_on_cran()

  fit <- make_test_fit()
  r <- ratio(fit, by = "site")

  output <- capture.output(print(r))
  expect_true(any(grepl("Grouped by:", output)))
})

test_that("summary.ratiod_ratio with by grouping", {
  skip_on_cran()

  fit <- make_test_fit()
  r <- ratio(fit, by = "site")
  summ <- summary(r)

  expect_true("group" %in% names(summ))
  expect_equal(nrow(summ), 3)  # 3 sites
})

test_that("print.ratiod_ratio_summary with many rows", {
  skip_on_cran()

  # Create a fit with many observations for testing truncated print
  set.seed(99999)
  n <- 30
  df <- data.frame(
    count = rpois(n, lambda = 10),
    effort = rgamma(n, shape = 5, rate = 1),
    x = rnorm(n)
  )

  fit <- tratio(
    count | effort ~ x,
    data = df,
    family = ratiod_poisson_gamma(),
    mode = "hmc",
    control = list(iter = 100, warmup = 50, chains = 1, verbose = FALSE)
  )

  r <- ratio(fit, summary = TRUE)
  output <- capture.output(print(r, n = 5))

  expect_true(any(grepl("and .* more rows", output)))
})

test_that("aggregate_by_group handles single observation per group", {
  draws <- matrix(c(2, 8), nrow = 1, ncol = 2)
  data <- data.frame(obs = 1:2, group = c("A", "B"))

  result <- tulpaRatio:::aggregate_by_group(draws, data, "group")

  expect_equal(ncol(result), 2)
  expect_equal(as.numeric(result[1, 1]), 2)
  expect_equal(as.numeric(result[1, 2]), 8)
})

test_that("draw accessors preserve the draw contract", {
  skip_on_cran()
  expect_draw_accessors(make_test_fit())
})
