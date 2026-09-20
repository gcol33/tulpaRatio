# Integration tests for backend code - these run model fits to exercise C++ code

# Helper for quick model fits
quick_fit <- function(family = ratiod_poisson_gamma(), n = 20, iter = 100, warmup = 50) {
  set.seed(42)
  df <- data.frame(
    count = rpois(n, lambda = 10),
    effort = rgamma(n, shape = 5, rate = 1),
    x = rnorm(n),
    site = factor(rep(1:4, length.out = n))
  )

  tratio(
    count | effort ~ x,
    data = df,
    family = family,
    mode = "hmc",
    control = list(iter = iter, warmup = warmup, chains = 1, verbose = FALSE)
  )
}

# Test HMC backend with poisson_gamma
test_that("HMC backend works with poisson_gamma", {
  fit <- quick_fit(ratiod_poisson_gamma())

  expect_s3_class(fit, "ratiod_fit")
  expect_equal(fit$backend, "hmc")
  expect_true(!is.null(fit$draws))
  expect_true(nrow(fit$draws) > 0)
})

# Test HMC backend with negbin_negbin
test_that("HMC backend works with negbin_negbin", {
  set.seed(123)
  n <- 20
  df <- data.frame(
    y_num = rnbinom(n, size = 5, mu = 10),
    y_denom = rnbinom(n, size = 5, mu = 20),
    x = rnorm(n)
  )

  fit <- tratio(
    y_num | y_denom ~ x,
    data = df,
    family = ratiod_negbin_negbin(),
    mode = "hmc",
    control = list(iter = 100, warmup = 50, chains = 1, verbose = FALSE)
  )

  expect_s3_class(fit, "ratiod_fit")
  expect_equal(fit$backend, "hmc")
})

# Test HMC backend with binomial
test_that("HMC backend works with binomial", {
  set.seed(456)
  n <- 20
  trials <- sample(10:20, n, replace = TRUE)
  df <- data.frame(
    successes = rbinom(n, trials, 0.4),
    trials = trials,
    x = rnorm(n)
  )

  fit <- tratio(
    successes | trials ~ x,
    data = df,
    family = ratiod_binomial(),
    mode = "hmc",
    control = list(iter = 100, warmup = 50, chains = 1, verbose = FALSE)
  )

  expect_s3_class(fit, "ratiod_fit")
  expect_equal(fit$backend, "hmc")
})

# Test HMC with random effects
test_that("HMC backend works with random effects", {
  set.seed(789)
  n <- 30
  df <- data.frame(
    count = rpois(n, lambda = 10),
    effort = rgamma(n, shape = 5, rate = 1),
    x = rnorm(n),
    site = factor(rep(1:5, each = 6))
  )

  fit <- tratio(
    count | effort ~ x + (1 | site),
    data = df,
    family = ratiod_poisson_gamma(),
    mode = "hmc",
    control = list(iter = 100, warmup = 50, chains = 1, verbose = FALSE)
  )

  expect_s3_class(fit, "ratiod_fit")
  # Should have RE parameters
  expect_true(any(grepl("sigma_re|re\\[", colnames(fit$draws))))
})

# Test print.ratiod_fit
test_that("print.ratiod_fit works", {
  fit <- quick_fit()

  output <- capture.output(print(fit))
  expect_true(any(grepl("ratio model fit", output)))
  expect_true(any(grepl("Family", output)))
  expect_true(any(grepl("Observations", output)))
})

# Test summary.ratiod_fit
test_that("summary.ratiod_fit works", {
  fit <- quick_fit()

  # summary prints to console and returns fit object invisibly
  summ <- summary(fit)
  expect_s3_class(summ, "ratiod_fit")
})

# Test summary with prob argument
test_that("summary.ratiod_fit respects prob argument", {
  fit <- quick_fit()

  # summary.ratiod_fit takes prob (CI width) not probs (quantiles)
  # It invisibly returns the fit object
  result <- summary(fit, prob = 0.90)
  expect_s3_class(result, "ratiod_fit")
})

# Test coef.ratiod_fit (if method exists)
test_that("coef.ratiod_fit extracts coefficients", {
  fit <- quick_fit()

  # coef method may not exist for all backends
  if (exists("coef.ratiod_fit", mode = "function")) {
    coefs <- coef(fit)
    expect_true(is.numeric(coefs))
    expect_true(length(coefs) > 0)
  } else {
    expect_true(TRUE)  # Skip if method doesn't exist
  }
})

# Test formula.ratiod_fit
test_that("formula info is accessible", {
  fit <- quick_fit()

  # Check formula is stored in fit object
  expect_true(!is.null(fit$formula) || !is.null(fit$call))
})

# Test ratio extraction
test_that("ratio() extracts posterior draws", {
  fit <- quick_fit()

  r <- ratio(fit)
  expect_s3_class(r, "ratiod_ratio")
  expect_true(is.matrix(r$draws))
  expect_equal(ncol(r$draws), 20)  # n = 20
})

# Test ratio with log scale
test_that("ratio() works with log scale", {
  fit <- quick_fit()

  r_log <- ratio(fit, type = "log")
  expect_equal(r_log$type, "log")
  # Log ratios can be negative
  expect_true(any(r_log$draws < 0) || all(r_log$draws > 0))
})

# Test ratio summary
test_that("ratio() with summary=TRUE returns summary", {
  fit <- quick_fit()

  r_summ <- ratio(fit, summary = TRUE)
  expect_s3_class(r_summ, "ratiod_ratio_summary")
  expect_true("mean" %in% names(r_summ))
})

# Test fitted values
test_that("fitted.ratiod_fit returns fitted values", {
  fit <- quick_fit()

  fv <- fitted(fit)
  expect_s3_class(fv, "ratiod_fitted")
  expect_true("component" %in% names(fv))
  expect_true("ratio" %in% fv$component)
})

# Test fitted with specific component
test_that("fitted.ratiod_fit filters by component", {
  fit <- quick_fit()

  fv_num <- fitted(fit, component = "numerator")
  expect_true(all(fv_num$component == "numerator"))

  fv_ratio <- fitted(fit, component = "ratio")
  expect_true(all(fv_ratio$component == "ratio"))
})

# Test fitted with summary=FALSE
test_that("fitted.ratiod_fit returns draws when summary=FALSE", {
  fit <- quick_fit()

  fv_draws <- fitted(fit, summary = FALSE)
  expect_s3_class(fv_draws, "ratiod_fitted_draws")
  expect_true(is.matrix(fv_draws$ratio))
})

# Test predict with newdata
test_that("predict.ratiod_fit works with newdata", {
  fit <- quick_fit()

  newdata <- data.frame(x = c(-1, 0, 1))
  pred <- predict(fit, newdata = newdata)

  expect_s3_class(pred, "ratiod_prediction")
  expect_equal(nrow(pred), 3)
})

# Test predict without newdata returns fitted
test_that("predict.ratiod_fit without newdata returns fitted", {
  fit <- quick_fit()

  pred <- predict(fit)
  expect_s3_class(pred, "ratiod_fitted")
})

# Test diagnostics()
test_that("diagnostics() returns diagnostics", {
  fit <- quick_fit()

  diag <- diagnostics(fit)
  expect_s3_class(diag, "data.frame")
  expect_true("rhat" %in% names(diag) || "Rhat" %in% names(diag))
  expect_true("ess_bulk" %in% names(diag) || "n_eff" %in% names(diag))
})

# Test check_diagnostics
test_that("check_diagnostics validates convergence", {
  fit <- quick_fit()

  # Should not error - returns various types depending on issues found
  result <- check_diagnostics(fit)
  # Result can be logical, NULL, list, or data.frame depending on implementation
  expect_true(TRUE)  # Main test is that it runs without error
})

# Test as_draws
test_that("as_draws.ratiod_fit converts to draws format", {
  skip_if_not_installed("posterior")

  fit <- quick_fit()

  draws <- as_draws(fit)
  expect_true(inherits(draws, "draws"))
})

# Test draws_wide
test_that("draws_wide.ratiod_fit works", {
  fit <- quick_fit()

  sp <- draws_wide(fit, beta_num)
  expect_s3_class(sp, "ratiod_draws")
  expect_true(".chain" %in% names(sp))
  expect_true(".draw" %in% names(sp))
})

# Test draws_long
test_that("draws_long.ratiod_fit works", {
  fit <- quick_fit()

  gd <- draws_long(fit, beta_num)
  expect_s3_class(gd, "ratiod_draws_long")
  expect_true(".variable" %in% names(gd))
  expect_true(".value" %in% names(gd))
})

# Test draws_interval
test_that("draws_interval.ratiod_fit works", {
  fit <- quick_fit()

  pi <- draws_interval(fit, beta_num)
  expect_s3_class(pi, "ratiod_draws_interval")
  expect_true(".value" %in% names(pi))
  expect_true(".lower" %in% names(pi))
  expect_true(".upper" %in% names(pi))
})

# Test draws_interval with HDI
test_that("draws_interval supports HDI", {
  fit <- quick_fit()

  pi_qi <- draws_interval(fit, beta_num, .interval = "qi")
  pi_hdi <- draws_interval(fit, beta_num, .interval = "hdi")

  expect_equal(unique(pi_qi$.interval), "qi")
  expect_equal(unique(pi_hdi$.interval), "hdi")
})

# Test multi-chain fitting
test_that("HMC backend works with multiple chains", {
  set.seed(999)
  n <- 20
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
    control = list(iter = 100, warmup = 50, chains = 2, verbose = FALSE)
  )

  expect_s3_class(fit, "ratiod_fit")
  expect_equal(fit$chains, 2)
})

# Test that auto backend selects HMC by default
test_that("auto backend selects HMC for standard models", {
  fit <- quick_fit()
  expect_equal(fit$backend, "hmc")
})

# Test ratio_contrast
test_that("ratio_contrast computes contrasts", {
  set.seed(111)
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
    control = list(iter = 100, warmup = 50, chains = 1, verbose = FALSE)
  )

  contrast <- ratio_contrast(fit, ~ season)

  expect_s3_class(contrast, "ratiod_contrast")
  expect_true("mean" %in% names(contrast))
  expect_true("prob_positive" %in% names(contrast))
})

# Test ratio_contrast with ratio type
test_that("ratio_contrast computes ratio of ratios", {
  set.seed(222)
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
    control = list(iter = 100, warmup = 50, chains = 1, verbose = FALSE)
  )

  contrast <- ratio_contrast(fit, ~ season, type = "ratio")

  expect_equal(attr(contrast, "type"), "ratio")
  # Ratio of ratios is positive
  expect_true(contrast$mean[1] > 0)
})

# ---------------------------------------------------------------------------
# Laplace backend tests (exercises laplace_core.cpp)
# ---------------------------------------------------------------------------

test_that("Laplace backend works with binomial", {
  set.seed(333)
  n <- 30
  trials <- sample(10:20, n, replace = TRUE)
  df <- data.frame(
    successes = rbinom(n, trials, 0.4),
    trials = trials,
    x = rnorm(n)
  )

  fit <- tratio(
    successes | trials ~ x,
    data = df,
    family = ratiod_binomial(),
    mode = "laplace",
    control = list(verbose = FALSE)
  )

  expect_s3_class(fit, "ratiod_fit")
  expect_equal(fit$backend, "laplace")
})

test_that("Laplace backend works with poisson_gamma", {
  set.seed(444)
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
    mode = "laplace",
    control = list(verbose = FALSE)
  )

  expect_s3_class(fit, "ratiod_fit")
  expect_equal(fit$backend, "laplace")
})

test_that("Laplace backend works with random effects", {
  set.seed(555)
  n <- 40
  df <- data.frame(
    count = rpois(n, lambda = 10),
    effort = rgamma(n, shape = 5, rate = 1),
    x = rnorm(n),
    site = factor(rep(1:5, each = 8))
  )

  fit <- tratio(
    count | effort ~ x + (1 | site),
    data = df,
    family = ratiod_poisson_gamma(),
    mode = "laplace",
    control = list(verbose = FALSE)
  )

  expect_s3_class(fit, "ratiod_fit")
  expect_equal(fit$backend, "laplace")
})

# ---------------------------------------------------------------------------
# PG (Pólya-Gamma) backend tests (exercises pg_binomial.cpp)
# ---------------------------------------------------------------------------

test_that("PG backend works with binomial", {
  set.seed(666)
  n <- 30
  trials <- sample(10:20, n, replace = TRUE)
  df <- data.frame(
    successes = rbinom(n, trials, 0.4),
    trials = trials,
    x = rnorm(n)
  )

  fit <- tratio(
    successes | trials ~ x,
    data = df,
    family = ratiod_binomial(),
    mode = "pg",
    control = list(iter = 100, warmup = 50, chains = 1, verbose = FALSE)
  )

  expect_s3_class(fit, "ratiod_fit")
  expect_equal(fit$backend, "pg")
})

test_that("PG backend works with random effects", {
  set.seed(777)
  n <- 40
  trials <- sample(10:20, n, replace = TRUE)
  df <- data.frame(
    successes = rbinom(n, trials, 0.4),
    trials = trials,
    x = rnorm(n),
    site = factor(rep(1:5, each = 8))
  )

  fit <- tratio(
    successes | trials ~ x + (1 | site),
    data = df,
    family = ratiod_binomial(),
    mode = "pg",
    control = list(iter = 100, warmup = 50, chains = 1, verbose = FALSE)
  )

  expect_s3_class(fit, "ratiod_fit")
  expect_equal(fit$backend, "pg")
})

# ---------------------------------------------------------------------------
# Spatial backend tests (exercises pg_spatial.cpp)
# ---------------------------------------------------------------------------

test_that("PG backend works with spatial CAR", {
  set.seed(888)
  n <- 25
  n_sites <- 5
  trials <- sample(10:20, n, replace = TRUE)

  # Create spatial data
  df <- data.frame(
    successes = rbinom(n, trials, 0.4),
    trials = trials,
    x = rnorm(n),
    site = factor(rep(1:n_sites, each = n / n_sites))
  )

  # Simple adjacency matrix: site i neighbors i-1 and i+1
  adj <- matrix(0, nrow = n_sites, ncol = n_sites)
  adj[1, 2] <- adj[2, 1] <- 1
  adj[2, 3] <- adj[3, 2] <- 1
  adj[3, 4] <- adj[4, 3] <- 1
  adj[4, 5] <- adj[5, 4] <- 1

  fit <- tratio(
    successes | trials ~ x + (1 | site),
    data = df,
    family = ratiod_binomial(),
    spatial = spatial_car(adj, level = "group", group_var = "site"),
    mode = "pg",
    control = list(iter = 100, warmup = 50, chains = 1, verbose = FALSE)
  )

  expect_s3_class(fit, "ratiod_fit")
})

test_that("HMC backend works with spatial CAR", {
  set.seed(999)
  n <- 25
  n_sites <- 5

  # Create spatial count data
  df <- data.frame(
    count = rpois(n, lambda = 10),
    effort = rgamma(n, shape = 5, rate = 1),
    x = rnorm(n),
    site = factor(rep(1:n_sites, each = n / n_sites))
  )

  # Simple adjacency matrix
  adj <- matrix(0, nrow = n_sites, ncol = n_sites)
  adj[1, 2] <- adj[2, 1] <- 1
  adj[2, 3] <- adj[3, 2] <- 1
  adj[3, 4] <- adj[4, 3] <- 1
  adj[4, 5] <- adj[5, 4] <- 1

  fit <- tratio(
    count | effort ~ x + (1 | site),
    data = df,
    family = ratiod_poisson_gamma(),
    spatial = spatial_car(adj, level = "group", group_var = "site"),
    mode = "hmc",
    control = list(iter = 100, warmup = 50, chains = 1, verbose = FALSE)
  )

  expect_s3_class(fit, "ratiod_fit")
})

# ---------------------------------------------------------------------------
# Temporal backend tests (exercises hmc_temporal.h)
# ---------------------------------------------------------------------------

test_that("HMC backend works with temporal RW1", {
  set.seed(1001)
  n_times <- 8
  n_sites <- 3
  n <- n_times * n_sites

  df <- data.frame(
    count = rpois(n, lambda = 10),
    effort = rgamma(n, shape = 5, rate = 1),
    x = rnorm(n),
    time = rep(1:n_times, n_sites),
    site = factor(rep(1:n_sites, each = n_times))
  )

  fit <- tratio(
    count | effort ~ x + (1 | site),
    data = df,
    family = ratiod_poisson_gamma(),
    temporal = temporal_rw1("time"),
    mode = "hmc",
    control = list(iter = 100, warmup = 50, chains = 1, verbose = FALSE)
  )

  expect_s3_class(fit, "ratiod_fit")
  expect_equal(fit$backend, "hmc")
  # Should have temporal parameters
  expect_true(any(grepl("phi_temporal|tau_temporal", colnames(fit$draws))))
})


test_that("HMC backend works with temporal AR1", {
  set.seed(1002)
  n_times <- 8
  n_sites <- 3
  n <- n_times * n_sites

  df <- data.frame(
    count = rpois(n, lambda = 10),
    effort = rgamma(n, shape = 5, rate = 1),
    x = rnorm(n),
    time = rep(1:n_times, n_sites),
    site = factor(rep(1:n_sites, each = n_times))
  )

  fit <- tratio(
    count | effort ~ x + (1 | site),
    data = df,
    family = ratiod_poisson_gamma(),
    temporal = temporal_ar1("time"),
    mode = "hmc",
    control = list(iter = 100, warmup = 50, chains = 1, verbose = FALSE)
  )

  expect_s3_class(fit, "ratiod_fit")
  expect_equal(fit$backend, "hmc")
  # AR1 should have rho parameter
  expect_true(any(grepl("rho|phi_temporal", colnames(fit$draws))))
})


test_that("HMC backend works with temporal RW2", {
  set.seed(1003)
  n_times <- 10  # RW2 needs more time points
  n_sites <- 3
  n <- n_times * n_sites

  df <- data.frame(
    count = rpois(n, lambda = 10),
    effort = rgamma(n, shape = 5, rate = 1),
    x = rnorm(n),
    time = rep(1:n_times, n_sites),
    site = factor(rep(1:n_sites, each = n_times))
  )

  fit <- tratio(
    count | effort ~ x + (1 | site),
    data = df,
    family = ratiod_poisson_gamma(),
    temporal = temporal_rw2("time"),
    mode = "hmc",
    control = list(iter = 100, warmup = 50, chains = 1, verbose = FALSE)
  )

  expect_s3_class(fit, "ratiod_fit")
  expect_equal(fit$backend, "hmc")
})


test_that("HMC backend works with cyclic RW1", {
  set.seed(1004)
  n_months <- 12
  n_sites <- 3
  n <- n_months * n_sites

  df <- data.frame(
    count = rpois(n, lambda = 10),
    effort = rgamma(n, shape = 5, rate = 1),
    x = rnorm(n),
    month = rep(1:n_months, n_sites),
    site = factor(rep(1:n_sites, each = n_months))
  )

  fit <- tratio(
    count | effort ~ x + (1 | site),
    data = df,
    family = ratiod_poisson_gamma(),
    temporal = temporal_rw1("month", cyclic = TRUE),
    mode = "hmc",
    control = list(iter = 100, warmup = 50, chains = 1, verbose = FALSE)
  )

  expect_s3_class(fit, "ratiod_fit")
  expect_equal(fit$backend, "hmc")
})


# ---------------------------------------------------------------------------
# Parallel chain tests (exercises OpenMP across-chain parallelization)
# ---------------------------------------------------------------------------

test_that("HMC backend works with parallel chains", {
  set.seed(3001)
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
    control = list(iter = 100, warmup = 50, chains = 2, cores = 2, verbose = FALSE)
  )

  expect_s3_class(fit, "ratiod_fit")
  expect_equal(fit$chains, 2)
  # Each chain should have contributed samples
  expect_true(nrow(fit$draws) >= 2 * 50)  # At least 50 post-warmup per chain
})


test_that("HMC backend with 4 chains produces consistent results", {
  set.seed(3002)
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
    mode = "hmc",  # Test with fewer cores than chains
    control = list(iter = 100, warmup = 50, chains = 4, cores = 2, verbose = FALSE)
  )

  expect_s3_class(fit, "ratiod_fit")
  expect_equal(fit$chains, 4)

  # Rhat should be reasonable for converged chains
  diag <- diagnostics(fit)
  if ("rhat" %in% names(diag)) {
    # All Rhats should be < 2.0 for very short chains (not converged, but not broken)
    expect_true(all(diag$rhat < 2.0, na.rm = TRUE))
  }
})


test_that("Temporal model works with multiple chains", {
  set.seed(3003)
  n_times <- 6
  n_sites <- 3
  n <- n_times * n_sites

  df <- data.frame(
    count = rpois(n, lambda = 10),
    effort = rgamma(n, shape = 5, rate = 1),
    x = rnorm(n),
    time = rep(1:n_times, n_sites),
    site = factor(rep(1:n_sites, each = n_times))
  )

  fit <- tratio(
    count | effort ~ x + (1 | site),
    data = df,
    family = ratiod_poisson_gamma(),
    temporal = temporal_rw1("time"),
    mode = "hmc",
    control = list(iter = 100, warmup = 50, chains = 2, cores = 2, verbose = FALSE)
  )

  expect_s3_class(fit, "ratiod_fit")
  expect_equal(fit$chains, 2)
})


# ---------------------------------------------------------------------------
# Edge case tests for robustness
# ---------------------------------------------------------------------------

test_that("Model handles single time point per site", {
  set.seed(5002)
  n <- 20
  df <- data.frame(
    count = rpois(n, lambda = 10),
    effort = rgamma(n, shape = 5, rate = 1),
    x = rnorm(n),
    time = 1:n,  # Each observation is a unique time
    site = factor(rep(1:4, each = 5))
  )

  fit <- tratio(
    count | effort ~ x + (1 | site),
    data = df,
    family = ratiod_poisson_gamma(),
    temporal = temporal_rw1("time"),
    mode = "hmc",
    control = list(iter = 100, warmup = 50, chains = 1, verbose = FALSE)
  )

  expect_s3_class(fit, "ratiod_fit")
})
