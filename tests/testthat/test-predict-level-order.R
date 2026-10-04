# A fit numbers the units of a grouping variable by factor level. A prediction
# that numbered them in order of first appearance would hand each row another
# unit's effect whenever the two orders differ, so these fixtures list the
# units in reverse level order.

level_order_data <- function(n = 60, seed = 5) {
  set.seed(seed)
  S <- 6L
  unit <- rep(rev(seq_len(S)), each = n / S)
  data.frame(
    y = rbinom(n, 20L, plogis(-0.3 + 0.4 * (unit - 3.5) / 2)),
    trials = 20L,
    x = rnorm(n),
    region = factor(unit),
    site = factor(sprintf("s%02d", unit))
  )
}

level_order_adjacency <- function(S = 6L) {
  A <- matrix(0L, S, S)
  for (i in seq_len(S - 1L)) {
    A[i, i + 1L] <- 1L
    A[i + 1L, i] <- 1L
  }
  A
}

expect_prediction_matches_fit <- function(fit, df) {
  f <- fitted(fit, component = "ratio", summary = FALSE)$ratio
  p <- suppressWarnings(predict(fit, newdata = df, component = "ratio",
                                summary = FALSE))$ratio
  expect_equal(unname(p), unname(f))
}

test_that("an areal prediction reads each row's own unit (HMC)", {
  skip_on_cran()
  df <- level_order_data()
  fit <- suppressWarnings(tratio(
    y | trials ~ x, data = df, family = ratiod_binomial(),
    spatial = spatial_car(level_order_adjacency(), group_var = "region"),
    mode = "hmc",
    control = list(iter = 80, warmup = 40, chains = 1, verbose = FALSE)
  ))
  expect_prediction_matches_fit(fit, df)
})

test_that("a PG fit stores the predictor of the draw it saves", {
  skip_on_cran()
  skip("blocked on gcol33/tulpa#941: the field-free PG sampler is tulpa's")
  df <- level_order_data()
  fit <- suppressWarnings(tratio(
    y | trials ~ x, data = df, family = ratiod_binomial(), mode = "pg",
    control = list(iter = 80, warmup = 40, chains = 1, verbose = FALSE)
  ))
  eta <- fit$.internal$eta
  beta <- fit$.internal$beta
  expect_equal(unname(eta), unname(beta %*% t(cbind(1, df$x))))
})

test_that("a PG spatial fit stores the predictor of the draw it saves", {
  skip_on_cran()
  df <- level_order_data()
  fit <- suppressWarnings(tratio(
    y | trials ~ x, data = df, family = ratiod_binomial(),
    spatial = spatial_car(level_order_adjacency(), group_var = "region"),
    mode = "pg",
    control = list(iter = 80, warmup = 40, chains = 1, verbose = FALSE)
  ))
  internal <- fit$.internal
  field <- internal$spatial[, as.integer(df$region), drop = FALSE]
  expect_equal(unname(internal$eta),
               unname(internal$beta %*% t(cbind(1, df$x)) + field))
})

test_that("an areal and a random-effect prediction read each row's own unit (PG)", {
  skip_on_cran()
  df <- level_order_data()
  fit <- suppressWarnings(tratio(
    y | trials ~ x + (1 | site), data = df, family = ratiod_binomial(),
    spatial = spatial_car(level_order_adjacency(), group_var = "region"),
    mode = "pg",
    control = list(iter = 80, warmup = 40, chains = 1, verbose = FALSE)
  ))
  expect_prediction_matches_fit(fit, df)
})
