# Tests for validation and model comparison functions (R/validate.R)

test_that("pp_check generic dispatches correctly", {
  # pp_check is a generic
  expect_true(is.function(pp_check))
  expect_true("pp_check.ratiod_fit" %in% methods("pp_check"))
})

test_that("ratiod_compare requires at least two models", {
  skip_on_cran()
  skip_if_not_installed("loo")

  set.seed(123)
  n <- 30
  df <- data.frame(
    count = rpois(n, lambda = 10),
    effort = rgamma(n, shape = 3, rate = 1),
    x = rnorm(n)
  )

  fit <- tratio(
    count | effort ~ x,
    data = df,
    family = ratiod_poisson_gamma(),
    mode = "hmc",
    control = list(iter = 100, warmup = 50, chains = 1)
  )

  expect_error(
    ratiod_compare(fit),
    "At least two models"
  )
})

test_that("ratiod_compare validates model class", {
  skip_if_not_installed("loo")

  expect_error(
    ratiod_compare("not_a_fit", "also_not_a_fit"),
    "ratiod_fit"
  )
})

test_that("ratiod_average requires at least two models", {
  skip_on_cran()
  skip_if_not_installed("loo")

  set.seed(456)
  n <- 30
  df <- data.frame(
    count = rpois(n, lambda = 10),
    effort = rgamma(n, shape = 3, rate = 1),
    x = rnorm(n)
  )

  fit <- tratio(
    count | effort ~ x,
    data = df,
    family = ratiod_poisson_gamma(),
    mode = "hmc",
    control = list(iter = 100, warmup = 50, chains = 1)
  )

  expect_error(
    ratiod_average(fit),
    "At least two models"
  )
})

test_that("ratiod_average validates model class", {
  skip_if_not_installed("loo")

  expect_error(
    ratiod_average("not_a_fit", "also_not_a_fit"),
    "ratiod_fit"
  )
})

test_that("average_predictions works on log scale for ratios", {
  set.seed(789)
  # Create fake prediction matrices
  predictions <- list(
    matrix(rep(2, 50), nrow = 10, ncol = 5),
    matrix(rep(8, 50), nrow = 10, ncol = 5)
  )
  weights <- c(0.5, 0.5)

  result <- tulpaRatio:::average_predictions(predictions, weights, type = "ratio", summary = FALSE)

  # Geometric mean of 2 and 8 with equal weights = 4
  expect_equal(result[1, 1], 4, tolerance = 0.001)
})

test_that("average_predictions works for numerator/denominator", {
  set.seed(101)
  predictions <- list(
    matrix(rep(2, 50), nrow = 10, ncol = 5),
    matrix(rep(8, 50), nrow = 10, ncol = 5)
  )
  weights <- c(0.5, 0.5)

  result <- tulpaRatio:::average_predictions(predictions, weights, type = "numerator", summary = FALSE)

  # Arithmetic mean of 2 and 8 with equal weights = 5
  expect_equal(result[1, 1], 5, tolerance = 0.001)
})

test_that("average_predictions returns summary when requested", {
  set.seed(102)
  predictions <- list(
    matrix(rnorm(50, mean = 10), nrow = 10, ncol = 5),
    matrix(rnorm(50, mean = 12), nrow = 10, ncol = 5)
  )
  weights <- c(0.6, 0.4)

  result <- tulpaRatio:::average_predictions(predictions, weights, type = "numerator", summary = TRUE)

  expect_true(is.data.frame(result))
  expect_true("mean" %in% names(result))
  expect_true("sd" %in% names(result))
  expect_true("q2.5" %in% names(result))
  expect_true("q97.5" %in% names(result))
  expect_equal(nrow(result), 5)
})

test_that("print.ratiod_average works", {
  # Create mock ratiod_average object
  avg <- structure(
    list(
      weights = c(model1 = 0.7, model2 = 0.3),
      predictions = data.frame(
        mean = c(1.2, 1.5, 1.8),
        sd = c(0.1, 0.15, 0.2),
        q2.5 = c(1.0, 1.2, 1.4),
        q97.5 = c(1.4, 1.8, 2.2)
      ),
      models = c("model1", "model2"),
      type = "ratio",
      weights_method = "loo",
      n_models = 2
    ),
    class = "ratiod_average"
  )

  output <- capture.output(print(avg))
  expect_true(any(grepl("ratio model averaging", output)))
  expect_true(any(grepl("Method:", output)))
  expect_true(any(grepl("Model weights:", output)))
})

test_that("fitted.ratiod_average extracts predictions", {
  avg <- structure(
    list(
      predictions = data.frame(mean = 1:3, sd = rep(0.1, 3))
    ),
    class = "ratiod_average"
  )

  result <- fitted(avg)
  expect_equal(result, avg$predictions)
})

test_that("weights.ratiod_average extracts weights", {
  avg <- structure(
    list(
      weights = c(m1 = 0.6, m2 = 0.4)
    ),
    class = "ratiod_average"
  )

  result <- weights(avg)
  expect_equal(result, c(m1 = 0.6, m2 = 0.4))
})

test_that("loo.ratiod_fit method exists", {
  # Just verify the method exists
  expect_true(exists("loo.ratiod_fit", mode = "function"))
})

test_that("waic.ratiod_fit method exists", {
  # Just verify the method exists
  expect_true(exists("waic.ratiod_fit", mode = "function"))
})

test_that("pp_check_single requires bayesplot", {
  skip_if_not_installed("bayesplot")

  # This is an internal function - just verify it exists
  expect_true(exists("pp_check_single", where = asNamespace("tulpaRatio"), mode = "function"))
})

# ---------------------------------------------------------------------------
# Integration tests for loo.ratiod_fit and waic.ratiod_fit
# The pointwise log-likelihood is reconstructed centrally from the posterior
# draws + design + family, so WAIC / LOO work on every matrix-draw backend
# (hmc, ess, sghmc, vi) without the backend storing log_lik (gcol33/tulpaRatio#3).
# ---------------------------------------------------------------------------

test_that("loo.ratiod_fit works on the HMC backend", {
  skip_on_cran()
  skip_if_not_installed("loo")

  set.seed(123)
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
    control = list(iter = 200, warmup = 100, chains = 1, verbose = FALSE)
  )

  l <- suppressWarnings(loo::loo(fit))
  expect_s3_class(l, "loo")
  expect_true(is.finite(l$estimates["looic", "Estimate"]))
})

test_that("waic.ratiod_fit works on the HMC backend", {
  skip_on_cran()
  skip_if_not_installed("loo")

  set.seed(456)
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
    control = list(iter = 200, warmup = 100, chains = 1, verbose = FALSE)
  )

  w <- loo::waic(fit)
  expect_s3_class(w, "waic")
  expect_true(is.finite(w$estimates["waic", "Estimate"]))
})

test_that("pointwise log-likelihood matches an independent recomputation", {
  skip_on_cran()

  set.seed(11)
  n <- 40
  df <- data.frame(
    y_num = rnbinom(n, size = 5, mu = 8),
    y_denom = rnbinom(n, size = 8, mu = 60),
    x = rnorm(n)
  )
  fit <- tratio(
    y_num | y_denom ~ x,
    data = df,
    family = ratiod_negbin_negbin(),
    mode = "hmc",
    control = list(iter = 150, warmup = 75, chains = 1, verbose = FALSE)
  )

  ll <- tulpaRatio:::.ratiod_pointwise_loglik(fit)
  expect_equal(dim(ll), c(nrow(fit$draws), n))
  expect_true(all(is.finite(ll)))

  # Recompute from scratch: eta = X %*% beta per draw, summed NegBin densities.
  d  <- fit$draws
  hd <- fit$.internal$hmc_data
  bn <- d[, grep("^beta_num\\[",   colnames(d)), drop = FALSE]
  bd <- d[, grep("^beta_denom\\[", colnames(d)), drop = FALSE]
  phi_n <- d[, "phi_num"]
  phi_d <- d[, "phi_denom"]
  ref <- matrix(NA_real_, nrow(d), n)
  for (s in seq_len(nrow(d))) {
    mu_n <- exp(as.numeric(hd$X_num   %*% bn[s, ]))
    mu_d <- exp(as.numeric(hd$X_denom %*% bd[s, ]))
    ref[s, ] <- dnbinom(hd$y_num,   size = phi_n[s], mu = mu_n, log = TRUE) +
                dnbinom(hd$y_denom, size = phi_d[s], mu = mu_d, log = TRUE)
  }
  expect_equal(ll, ref, tolerance = 1e-10)
})

test_that("WAIC / LOO work on the ESS backend (no stored log_lik)", {
  skip_on_cran()
  skip_if_not_installed("loo")

  set.seed(202)
  n <- 30
  df <- data.frame(
    y_num = rnbinom(n, size = 5, mu = 8),
    y_denom = rnbinom(n, size = 8, mu = 60),
    x = rnorm(n)
  )
  fit <- tratio(
    y_num | y_denom ~ x,
    data = df,
    family = ratiod_negbin_negbin(),
    mode = "ess",
    control = list(iter = 200, warmup = 100, chains = 1, verbose = FALSE)
  )

  expect_true(is.matrix(fit$draws))
  w <- loo::waic(fit)
  expect_true(is.finite(w$estimates["waic", "Estimate"]))
})

test_that("ratiod_compare ranks two HMC fits", {
  skip_on_cran()
  skip_if_not_installed("loo")

  set.seed(303)
  n <- 40
  df <- data.frame(
    count = rpois(n, lambda = 10),
    effort = rgamma(n, shape = 5, rate = 1),
    x = rnorm(n),
    z = rnorm(n)
  )
  fit1 <- tratio(count | effort ~ x, data = df,
                 family = ratiod_poisson_gamma(), mode = "hmc",
                 control = list(iter = 200, warmup = 100, chains = 1, verbose = FALSE))
  fit2 <- tratio(count | effort ~ x + z, data = df,
                 family = ratiod_poisson_gamma(), mode = "hmc",
                 control = list(iter = 200, warmup = 100, chains = 1, verbose = FALSE))

  cmp <- suppressWarnings(ratiod_compare(fit1, fit2, criterion = "waic"))
  expect_true(nrow(cmp) == 2)
  expect_true("elpd_diff" %in% colnames(cmp))
})

test_that("ratiod_average works end-to-end across backends", {
  skip_on_cran()
  skip_if_not_installed("loo")

  set.seed(505)
  n <- 40
  df <- data.frame(
    y_num = rnbinom(n, size = 5, mu = 8),
    y_denom = rnbinom(n, size = 8, mu = 60),
    x = rnorm(n),
    z = rnorm(n)
  )

  for (m in c("hmc", "ess")) {
    f1 <- tratio(y_num | y_denom ~ x, data = df, family = ratiod_negbin_negbin(),
                 mode = m,
                 control = list(iter = 200, warmup = 100, chains = 1, verbose = FALSE))
    f2 <- tratio(y_num | y_denom ~ x + z, data = df, family = ratiod_negbin_negbin(),
                 mode = m,
                 control = list(iter = 200, warmup = 100, chains = 1, verbose = FALSE))

    avg <- suppressWarnings(ratiod_average(f1, f2))
    expect_s3_class(avg, "ratiod_average")
    expect_length(weights(avg), 2)
    expect_equal(sum(weights(avg)), 1, tolerance = 1e-6)
    expect_true(is.data.frame(avg$predictions))
    expect_equal(nrow(avg$predictions), n)
  }
})

test_that("WAIC errors informatively on a backend without a draw matrix", {
  skip_on_cran()

  set.seed(404)
  n <- 30
  df <- data.frame(
    succ = rbinom(n, 10, 0.4),
    trials = rep(10L, n),
    x = rnorm(n)
  )
  fit <- suppressWarnings(tratio(
    succ | trials ~ x, data = df, family = ratiod_binomial(),
    mode = "pg",
    control = list(iter = 100, warmup = 50, chains = 1, verbose = FALSE)
  ))

  if (!is.matrix(fit$draws)) {
    expect_error(tulpaRatio:::.ratiod_pointwise_loglik(fit), "does not store")
  } else {
    succeed()
  }
})

test_that("ratio() extracts ratio posterior draws", {
  skip_on_cran()

  set.seed(333)
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
    control = list(iter = 100, warmup = 50, chains = 1, verbose = FALSE)
  )

  # ratio() should work with HMC fits
  r <- ratio(fit)
  expect_s3_class(r, "ratiod_ratio")
  expect_true(is.matrix(r$draws))
  expect_equal(ncol(r$draws), n)
})

test_that("print.ratiod_average handles matrix predictions", {
  # Create mock ratiod_average object with matrix predictions
  avg <- structure(
    list(
      weights = c(model1 = 0.6, model2 = 0.4),
      predictions = matrix(rnorm(30), nrow = 10, ncol = 3),
      models = c("model1", "model2"),
      type = "ratio",
      weights_method = "loo",
      n_models = 2
    ),
    class = "ratiod_average"
  )

  output <- capture.output(print(avg))
  expect_true(any(grepl("draws x", output)))
  expect_true(any(grepl("observations", output)))
})

test_that("the pointwise log-likelihood is evaluated at the full linear predictor", {
  skip_on_cran()
  sim <- sim_ratiod(n = 80, family = ratiod_poisson_gamma(), sigma_re = 0.8,
                    seed = 4)
  fit <- tratio(y_num | y_denom ~ x1 + (1 | group), data = sim$data,
                family = ratiod_poisson_gamma(),
                control = list(iter = 300, chains = 1, seed = 1))
  ll <- tulpaRatio:::.ratiod_pointwise_loglik(fit)

  fv <- fitted(fit, component = "all", summary = FALSE)
  S <- nrow(fv$numerator)
  y_num <- matrix(sim$data$y_num, S, ncol(ll), byrow = TRUE)
  y_den <- matrix(sim$data$y_denom, S, ncol(ll), byrow = TRUE)
  shape <- fit$draws[, "shape"]
  manual <- dpois(y_num, fv$numerator, log = TRUE) +
    dgamma(y_den, shape = shape, rate = shape / fv$denominator, log = TRUE)
  expect_equal(unname(ll), unname(manual))
})
