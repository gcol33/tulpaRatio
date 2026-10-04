laplace_summary_fit <- function(seed = 7) {
  set.seed(seed)
  n <- 40
  df <- data.frame(
    count = rpois(n, lambda = 10),
    effort = rgamma(n, shape = 5, rate = 1),
    x = rnorm(n),
    site = factor(rep(1:5, each = 8))
  )
  tratio(count | effort ~ x + (1 | site), data = df,
         family = ratiod_poisson_gamma(), mode = "laplace",
         control = list(verbose = FALSE))
}

test_that("the fixed-effect rows of a Laplace summary are the engine's mixture read", {
  fit <- laplace_summary_fit()
  probs <- c(0.025, 0.5, 0.975)
  summ <- apply_laplace_engine_summary(
    compute_param_summary(fit$draws, probs), fit, level = 0.95
  )
  rows <- match(fit$fixed_columns, summ$parameter)
  expect_equal(summ$mean[rows], unname(stats::coef(fit$engine)))
  expect_equal(summ$sd[rows], unname(sqrt(diag(stats::vcov(fit$engine)))))
  expect_equal(summ$q_lower[rows],
               unname(stats::confint(fit$engine, level = 0.95)[, 1L]))
})

test_that("the draws of a Laplace fit centre on the engine's fixed effects", {
  fit <- laplace_summary_fit()
  est <- unname(stats::coef(fit$engine))
  draws <- colMeans(fit$draws[, fit$fixed_columns, drop = FALSE])
  sd <- unname(sqrt(diag(stats::vcov(fit$engine))))
  expect_lt(max(abs(draws - est) / sd), 0.2)
})

test_that("hyperparameters are reported as posteriors, not point values", {
  fit <- laplace_summary_fit()
  for (nm in c("sigma_re", "shape")) {
    expect_true(nm %in% colnames(fit$draws))
    expect_gt(stats::sd(fit$draws[, nm]), 0)
  }
})

test_that("fitted, ratio and predict read one linear predictor", {
  fit <- laplace_summary_fit()
  pred <- predict(fit, newdata = fit$data[1:3, ])
  expect_equal(nrow(pred), 3L)
  expect_s3_class(ratio(fit), "ratiod_ratio")
})

test_that("a Laplace fit with a field refuses to predict at new rows", {
  set.seed(2)
  n <- 40
  adj <- matrix(0L, 5, 5)
  for (i in 1:4) adj[i, i + 1] <- adj[i + 1, i] <- 1L
  df <- data.frame(s = rbinom(n, 10, 0.4), n = 10L, x = rnorm(n),
                   site = factor(rep(1:5, 8)))
  fit <- tratio(s | n ~ x, data = df, family = ratiod_binomial(), mode = "laplace",
                spatial = spatial_car(adjacency = adj, group_var = "site"),
                control = list(verbose = FALSE))
  expect_error(predict(fit, newdata = df[1:2, ]), "not implemented")
})
