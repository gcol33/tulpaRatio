# Predictive and simulated objects are read through as.matrix(), as.data.frame()
# and coef(), each returning the same values the object holds.

test_that("as.matrix() returns each component of the prior predictive draws", {
  set.seed(1)
  df <- data.frame(count = 0L, total = 0L, x = rnorm(12))
  pp <- prior_predict(count | total ~ x, family = ratiod_negbin_negbin(),
                      data = df, n = 7)
  expect_identical(as.matrix(pp), unclass(pp)$ratio)
  expect_identical(as.matrix(pp, component = "numerator"), unclass(pp)$y_num)
  expect_identical(as.matrix(pp, component = "denominator"), unclass(pp)$y_denom)
  expect_equal(dim(as.matrix(pp)), c(7L, 12L))
})

test_that("posterior predictive replicates carry a class and their two components", {
  skip_on_cran()
  set.seed(4)
  df <- data.frame(count = rpois(30, 12), total = rpois(30, 80), x = rnorm(30))
  fit <- tratio(count | total ~ x, data = df, family = ratiod_poisson_gamma(),
                control = list(iter = 300, chains = 1, seed = 1, verbose = FALSE))
  rep <- posterior_predict(fit, ndraws = 20)
  expect_s3_class(rep, "ratiod_ppred")
  expect_equal(dim(as.matrix(rep)), c(20L, 30L))
  expect_identical(as.matrix(rep, component = "denominator"), unclass(rep)$y_denom_rep)
  expect_error(as.matrix(rep, component = "ratio"), "not available")
  expect_output(print(rep), "Posterior predictive replicates")

  pd <- predict(fit, newdata = data.frame(x = c(-1, 1)), summary = FALSE)
  expect_equal(dim(as.matrix(pd)), c(nrow(as.data.frame(fit)), 2L))
  expect_output(print(pd), "Predictive draws at new data")
})

test_that("simulated data read back as a data frame and true coefficients", {
  sim <- sim_ratiod(n = 40, family = ratiod_poisson_gamma(),
                    beta_num = c(2, 0.5), beta_denom = c(1, 0.2), seed = 3)
  expect_s3_class(as.data.frame(sim), "data.frame")
  expect_named(as.data.frame(sim), c("y_num", "y_denom", "x1", "x2", "group"))
  expect_equal(coef(sim), c(`num:(Intercept)` = 2, `num:x1` = 0.5,
                            `denom:(Intercept)` = 1, `denom:x1` = 0.2))
  sim_b <- sim_ratiod(n = 40, family = ratiod_binomial(), beta_num = c(0, 1), seed = 3)
  expect_named(coef(sim_b), c("num:(Intercept)", "num:x1"))
})

test_that("a count above the 32-bit integer range is refused at the front door (#150)", {
  df <- data.frame(count = c(5, 3e9, 7), total = c(10, 10, 10), x = c(0, 1, 2))
  expect_error(
    tratio(count | total ~ x, data = df, family = ratiod_negbin_negbin(),
           control = list(iter = 10, chains = 1, verbose = FALSE)),
    "1 count\\(s\\) above 2147483647"
  )
})
