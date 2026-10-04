laplace_summary_fit <- function(seed) {
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

test_that("a Laplace fit keeps the closed-form Gaussian of its fixed effects", {
  fit <- laplace_summary_fit(7)
  g <- fit$fixed_gaussian
  expect_identical(g$names, colnames(fit$.internal$X))
  expect_equal(unname(g$mean), unname(fit$.internal$mode[seq_along(g$names)]))
  expect_equal(g$cov, t(g$cov))
  expect_true(all(eigen(g$cov, symmetric = TRUE, only.values = TRUE)$values > 0))
  expect_equal(unname(sqrt(diag(g$cov))),
               unname(apply(fit$draws[, g$names, drop = FALSE], 2, stats::sd)),
               tolerance = 0.15)
})

test_that("summary rows of a Laplace fit do not move with the draw seed", {
  fit <- laplace_summary_fit(7)
  probs <- c(0.025, 0.5, 0.975)
  a <- apply_laplace_gaussian_summary(compute_param_summary(fit$draws, probs), fit, probs)
  fit$draws <- fit$draws[rev(seq_len(nrow(fit$draws))), ]
  b <- apply_laplace_gaussian_summary(compute_param_summary(fit$draws, probs), fit, probs)
  g <- fit$fixed_gaussian
  rows <- match(g$names, a$parameter)
  expect_equal(a[rows, -1], b[rows, -1])
  expect_equal(a$mean[rows], unname(g$mean))
  expect_equal(a$sd[rows], unname(sqrt(diag(g$cov))))
  expect_equal(a$q_lower[rows], unname(g$mean - stats::qnorm(0.975) * sqrt(diag(g$cov))))
})

test_that("a plug-in hyperparameter is reported as a point value", {
  fit <- laplace_summary_fit(7)
  probs <- c(0.025, 0.5, 0.975)
  summ <- apply_laplace_gaussian_summary(compute_param_summary(fit$draws, probs), fit, probs)
  row <- summ[summ$parameter == "sigma_re", ]
  expect_equal(row$mean, fit$.internal$sigma_re)
  expect_true(all(is.na(row[, c("sd", "q_lower", "q_median", "q_upper")])))
})
