test_that("engine binomial posterior recovers planted coefficients", {
  skip_on_cran()

  set.seed(20260503)
  n <- 200
  x1 <- rnorm(n); x2 <- rnorm(n)
  eta <- 0.4 + 0.8 * x1 - 0.5 * x2
  p <- plogis(eta)
  n_trials <- sample(5:20, n, replace = TRUE)
  y <- rbinom(n, n_trials, p)
  dat <- data.frame(y = y, n_trials = n_trials, x1 = x1, x2 = x2)

  fit_one <- function(seed_val) {
    fit <- tulpaRatio::tratio(
      formula = y | n_trials ~ x1 + x2, data = dat,
      family = tulpaRatio::ratiod_binomial(), mode = "hmc",
      control = list(iter = 2000L, warmup = 500L, chains = 1L, seed = seed_val, verbose = FALSE, gradient_mode = "A_r")
    )
    draws <- fit$draws
    if (is.matrix(draws)) colMeans(draws) else stop("unexpected draws shape")
  }

  for (seed in c(42L, 43L)) {
    estimates <- fit_one(seed)
    expect_lt(max(abs(estimates - c(.4, .8, -.5))), .2)
  }
})
