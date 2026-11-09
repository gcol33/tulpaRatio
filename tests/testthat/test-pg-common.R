test_that("PG Cholesky reports a non-positive-definite precision", {
  expect_error(tulpaRatio:::cpp_test_cholesky(matrix(c(1, 2, 2, 1), 2)),
               "not positive definite")
  expect_error(tulpaRatio:::cpp_test_cholesky(matrix(0, 2, 2)),
               "not positive definite")
})

test_that("PG coefficient conditional has the exact Gaussian mean and covariance", {
  skip_on_cran()
  X <- cbind(1, seq(0, 2, length.out = 12))
  omega <- seq(0.4, 1.6, length.out = 12)
  kappa <- seq(-0.3, 0.5, length.out = 12)
  offset <- rep(0.2, 12)
  Q <- crossprod(X, omega * X) + diag(1 / 2^2, 2)
  V <- solve(Q)
  mu <- drop(solve(Q, crossprod(X, kappa - omega * offset)))
  set.seed(191)
  draws <- t(replicate(30000, tulpaRatio:::cpp_test_pg_update_beta(
    kappa, omega, X, offset, 2)))
  expect_lt(max(abs(colMeans(draws) - mu) / sqrt(diag(V) / nrow(draws))), 5)
  covariance_se <- sqrt((V^2 + outer(diag(V), diag(V))) / (nrow(draws) - 1))
  expect_lt(max(abs(cov(draws) - V) / covariance_se), 5)
})

test_that("PG storage includes the last retained iteration under uneven thinning", {
  set.seed(19)
  args <- list(y = rep(1L, 6), n = rep(3L, 6), X = matrix(1, 6, 1),
    re_group = rep(1L, 6), n_re_groups = 0L,
    spatial_group = rep(1:3, 2), n_spatial_units = 3L,
    adj_list = list(2L, c(1L, 3L), 2L), n_neighbors = c(1L, 2L, 1L),
    n_iter = 9L, n_warmup = 4L, thin = 2L, store_eta = TRUE, verbose = FALSE)
  fit <- do.call(tulpaRatio:::cpp_pg_binomial_gibbs_spatial, args)
  expect_equal(nrow(fit$beta), 3L)
  expect_equal(nrow(fit$eta), 3L)
  expect_true(all(is.finite(fit$eta)))
})

test_that("zero Polya-Gamma weights leave the Gaussian coefficient prior", {
  set.seed(19)
  expected <- rnorm(2, sd = 2)
  set.seed(19)
  actual <- tulpaRatio:::cpp_test_pg_update_beta(rep(0, 2), rep(0, 2),
    diag(2), rep(0, 2), 2)
  expect_identical(as.numeric(actual), expected)
})
