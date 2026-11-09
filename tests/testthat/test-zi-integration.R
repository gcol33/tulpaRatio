# test-zi-integration.R
# Integration tests for zero-inflation and hurdle model fitting

# ============================================================================
# Zero-Inflated Poisson Model Fitting
# ============================================================================

test_that("zi_poisson fits with poisson_gamma family", {
  skip_on_cran()

  set.seed(123)
  n <- 50
  zi_prob <- 0.3

  # Simulate zero-inflated count data
  df <- data.frame(
    count = ifelse(runif(n) < zi_prob, 0, rpois(n, lambda = 5)),
    effort = rgamma(n, shape = 4, rate = 1),
    x = rnorm(n)
  )

  # Fit model with zero-inflation

  fit <- tratio(count | effort ~ x, data = df, family = ratiod_zi_poisson(denom_family = "gamma"), mode = "hmc",
    control = list(iter = 100, warmup = 50, chains = 1, verbose = FALSE))

  expect_s3_class(fit, "ratiod_fit")
  expect_equal(fit$family$name, "zi_poisson_gamma")

  # Should have ZI parameters in the output (named beta_zi[...])
  param_names <- colnames(fit$draws)
  expect_true(any(grepl("beta_zi", param_names)))
})


test_that("zi_poisson with formula fits covariates for ZI probability", {
  skip_on_cran()

  set.seed(456)
  n <- 60

  # Simulate data where ZI probability depends on habitat
  df <- data.frame(
    habitat = factor(rep(c("forest", "grassland"), each = n/2)),
    x = rnorm(n)
  )
  df$zi_prob <- ifelse(df$habitat == "forest", 0.4, 0.1)
  df$count <- ifelse(runif(n) < df$zi_prob, 0, rpois(n, lambda = 8))
  df$effort <- rgamma(n, shape = 4, rate = 1)

  # Fit with ZI covariate
  fit <- tratio(count | effort ~ x, data = df, family = ratiod_zi_poisson(denom_family = "gamma", zi_formula = ~habitat),
    mode = "hmc", control = list(iter = 100, warmup = 50, chains = 1, verbose = FALSE))

  expect_s3_class(fit, "ratiod_fit")

  # Check ZI coefficients exist (intercept + habitat effect)
  param_names <- colnames(fit$draws)
  zi_params <- param_names[grepl("beta_zi", param_names)]
  expect_gte(length(zi_params), 1)
})


# ============================================================================
# Zero-Inflated Negative Binomial Model Fitting
# ============================================================================

test_that("zi_negbin fits with negbin_negbin family", {
  skip_on_cran()

  set.seed(789)
  n <- 50
  zi_prob <- 0.25

  # Simulate zero-inflated overdispersed count data
  df <- data.frame(
    count = ifelse(runif(n) < zi_prob, 0, rnbinom(n, size = 3, mu = 10)),
    total = rnbinom(n, size = 5, mu = 50),
    x = rnorm(n)
  )

  fit <- tratio(count | total ~ x, data = df, family = ratiod_zi_negbin(denom_family = "negbin"), mode = "hmc",
    control = list(iter = 100, warmup = 50, chains = 1, verbose = FALSE))

  expect_s3_class(fit, "ratiod_fit")
  expect_equal(fit$family$name, "zi_negbin_negbin")
})


# ============================================================================
# Hurdle Poisson Model Fitting
# ============================================================================

test_that("hurdle_poisson fits correctly", {
  skip_on_cran()

  set.seed(101)
  n <- 50
  presence_prob <- 0.6

  # Simulate hurdle data: presence/absence + truncated Poisson
  df <- data.frame(
    count = ifelse(runif(n) > presence_prob, 0, rpois(n, lambda = 4) + 1),
    effort = rgamma(n, shape = 4, rate = 1),
    depth = rnorm(n)
  )
  # Ensure no zeros from the truncated Poisson (could still have zeros from hurdle)
  df$count[df$count > 0 & runif(n) < 0.1] <- 0  # Add some more zeros

  fit <- tratio(count | effort ~ depth, data = df, family = ratiod_hurdle_poisson(denom_family = "gamma"), mode = "hmc",
    control = list(iter = 100, warmup = 50, chains = 1, verbose = FALSE))

  expect_s3_class(fit, "ratiod_fit")
})


test_that("hurdle_poisson with formula for hurdle probability", {
  skip_on_cran()

  set.seed(202)
  n <- 60

  df <- data.frame(
    habitat = factor(rep(c("suitable", "unsuitable"), each = n/2)),
    x = rnorm(n)
  )
  # Higher presence probability in suitable habitat
  df$pres_prob <- ifelse(df$habitat == "suitable", 0.8, 0.3)
  df$count <- ifelse(runif(n) < df$pres_prob, rpois(n, lambda = 5) + 1, 0)
  df$effort <- rgamma(n, shape = 4, rate = 1)

  fit <- tratio(count | effort ~ x, data = df, family = ratiod_hurdle_poisson(denom_family = "gamma", zi_formula = ~habitat),
    mode = "hmc", control = list(iter = 100, warmup = 50, chains = 1, verbose = FALSE))

  expect_s3_class(fit, "ratiod_fit")
})


# ============================================================================
# Hurdle Negative Binomial Model Fitting
# ============================================================================

test_that("hurdle_negbin fits correctly", {
  skip_on_cran()

  set.seed(303)
  n <- 50
  presence_prob <- 0.7

  # Simulate hurdle NB data
  df <- data.frame(
    count = ifelse(runif(n) > presence_prob, 0, rnbinom(n, size = 2, mu = 8) + 1),
    total = rnbinom(n, size = 5, mu = 50),
    x = rnorm(n)
  )

  fit <- tratio(count | total ~ x, data = df, family = ratiod_hurdle_negbin(denom_family = "negbin"), mode = "hmc",
    control = list(iter = 100, warmup = 50, chains = 1, verbose = FALSE))

  expect_s3_class(fit, "ratiod_fit")
  expect_equal(fit$family$name, "hurdle_negbin_negbin")
})


# ============================================================================
# ZI with Random Effects
# ============================================================================

test_that("zi_poisson works with random effects", {
  skip_on_cran()

  set.seed(404)
  n_sites <- 8
  n_per_site <- 10
  n <- n_sites * n_per_site

  df <- data.frame(
    site = factor(rep(1:n_sites, each = n_per_site)),
    x = rnorm(n)
  )

  # Site-level random effect on ZI probability
  site_zi <- rep(rnorm(n_sites, mean = -1, sd = 0.5), each = n_per_site)
  df$zi_prob <- plogis(site_zi)
  df$count <- ifelse(runif(n) < df$zi_prob, 0, rpois(n, lambda = 6))
  df$effort <- rgamma(n, shape = 4, rate = 1)

  fit <- tratio(count | effort ~ x + (1 | site), data = df, family = ratiod_zi_poisson(denom_family = "gamma"),
    mode = "hmc", control = list(iter = 100, warmup = 50, chains = 1, verbose = FALSE))

  expect_s3_class(fit, "ratiod_fit")

  # Check random effects exist (sigma_re and re[...])
  param_names <- colnames(fit$draws)
  expect_true(any(grepl("re\\[", param_names)) || any(grepl("sigma_re", param_names)))
})


# ============================================================================
# Edge Cases and Validation
# ============================================================================

test_that("ZI model handles data with many zeros", {
  skip_on_cran()

  set.seed(505)
  n <- 60
  zi_prob <- 0.6  # High zero-inflation

  df <- data.frame(
    count = ifelse(runif(n) < zi_prob, 0, rpois(n, lambda = 3)),
    effort = rgamma(n, shape = 4, rate = 1),
    x = rnorm(n)
  )

  # Should handle ~60% zeros
  expect_true(mean(df$count == 0) > 0.5)

  fit <- tratio(count | effort ~ x, data = df, family = ratiod_zi_poisson(denom_family = "gamma"), mode = "hmc",
    control = list(iter = 100, warmup = 50, chains = 1, verbose = FALSE))

  expect_s3_class(fit, "ratiod_fit")
})


test_that("ZI model handles data with few zeros", {
  skip_on_cran()

  set.seed(606)
  n <- 60
  zi_prob <- 0.05  # Very low zero-inflation

  df <- data.frame(
    count = ifelse(runif(n) < zi_prob, 0, rpois(n, lambda = 10)),
    effort = rgamma(n, shape = 4, rate = 1),
    x = rnorm(n)
  )

  # Very few structural zeros
  expect_lt(mean(df$count == 0), 0.2)

  fit <- tratio(count | effort ~ x, data = df, family = ratiod_zi_poisson(denom_family = "gamma"), mode = "hmc",
    control = list(iter = 100, warmup = 50, chains = 1, verbose = FALSE))

  expect_s3_class(fit, "ratiod_fit")
})


test_that("ratio() extraction works with ZI models", {
  skip_on_cran()

  set.seed(707)
  n <- 40
  zi_prob <- 0.3

  df <- data.frame(
    count = ifelse(runif(n) < zi_prob, 0, rpois(n, lambda = 5)),
    effort = rgamma(n, shape = 4, rate = 1),
    x = rnorm(n)
  )

  fit <- tratio(count | effort ~ x, data = df, family = ratiod_zi_poisson(denom_family = "gamma"), mode = "hmc",
    control = list(iter = 100, warmup = 50, chains = 1, verbose = FALSE))

  # Extract ratios - should work same as non-ZI models
  ratios <- ratio(fit)
  expect_s3_class(ratios, "ratiod_ratio")
  expect_true(!is.null(ratios$draws))
})


# ============================================================================
# Summary and Print Methods with ZI
# ============================================================================

test_that("summary works for ZI models", {
  skip_on_cran()

  set.seed(808)
  n <- 40

  df <- data.frame(
    count = ifelse(runif(n) < 0.3, 0, rpois(n, lambda = 5)),
    effort = rgamma(n, shape = 4, rate = 1),
    x = rnorm(n)
  )

  fit <- tratio(count | effort ~ x, data = df, family = ratiod_zi_poisson(denom_family = "gamma"), mode = "hmc",
    control = list(iter = 100, warmup = 50, chains = 1, verbose = FALSE))

  # Summary should work
  summ <- summary(fit)
  expect_true(!is.null(summ))

  # Print should work
  output <- capture.output(print(fit))
  expect_true(length(output) > 0)
})
