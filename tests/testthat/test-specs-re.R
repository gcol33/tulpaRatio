simulate_re_binomial <- function(n = 200, n_groups = 8) {
  set.seed(20260601)
  group <- sample.int(n_groups, n, replace = TRUE)
  re_true <- rnorm(n_groups, sd = 0.6)
  x1 <- rnorm(n)
  eta <- 0.4 + 0.8 * x1 + re_true[group]
  p   <- plogis(eta)
  nt  <- sample(5:20, n, replace = TRUE)
  y   <- rbinom(n, nt, p)
  list(
    formula = y | n_trials ~ x1 + (1 | grp),
    data    = data.frame(y = y, n_trials = nt, x1 = x1,
                          grp = factor(group)),
    truth = p,
    family  = tulpaRatio::ratiod_binomial()
  )
}

fit_one_re <- function(spec, seed_val) {
  fit <- tulpaRatio::tratio(
    formula = spec$formula, data = spec$data, family = spec$family,
    mode = "hmc",
    control = list(iter = 2000L, warmup = 500L, chains = 1L, seed = seed_val, verbose = FALSE, gradient_mode = "A_r")
  )
  fit
}

test_that("engine random intercepts recover slope and probabilities", {
  skip_on_cran()
  spec <- simulate_re_binomial()
  for (seed in c(42L, 43L)) {
    fit <- fit_one_re(spec, seed)
    expect_lt(abs(mean(fit$draws[, "beta_num[2]"]) - .8), .2)
    probability <- colMeans(fitted(fit, summary = FALSE)$numerator)
    expect_lt(sqrt(mean((probability - spec$truth)^2)), .1)
  }
})
