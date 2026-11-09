test_that("RW2 PG conditional matches the dense second-difference precision", {
  n <- 6L
  D <- diff(diag(n), differences = 2)
  w <- seq(0.4, 1.5, length.out = n)
  b <- seq(-0.2, 0.7, length.out = n)
  Q <- diag(w) + 2.3 * crossprod(D)
  V <- solve(Q)
  mu <- drop(V %*% b)
  set.seed(219)
  z <- rnorm(n)
  expected <- mu + drop(solve(chol(Q), z))
  set.seed(219)
  actual <- tulpaRatio:::cpp_test_pg_rw2_conditional(w, b, 2.3)
  expect_equal(as.numeric(actual), expected, tolerance = 1e-12)
  expect_equal(as.numeric(D %*% rep(1, n)), rep(0, n - 2L))
  expect_equal(as.numeric(D %*% seq_len(n)), rep(0, n - 2L))
  skip_on_cran()
  set.seed(219)
  draws <- t(replicate(20000L,
    tulpaRatio:::cpp_test_pg_rw2_conditional(w, b, 2.3)))
  expect_lt(max(abs(colMeans(draws) - mu) / sqrt(diag(V) / nrow(draws))), 5)
  covariance_se <- sqrt((V^2 + outer(diag(V), diag(V))) / (nrow(draws) - 1))
  expect_lt(max(abs(cov(draws) - V) / covariance_se), 5)
})

test_that("PG RW2 posterior covers a simulated trend and innovation scale", {
  skip_on_cran()
  skip_unless_slow()
  set.seed(219)
  n <- 12L
  sigma <- 0.12
  truth <- c(0, 0, cumsum(cumsum(rnorm(n - 2L, sd = sigma))))
  truth <- truth - mean(truth)
  time <- rep(seq_len(n), each = 8L)
  d <- data.frame(time = time, trials = 30L)
  d$y <- rbinom(nrow(d), d$trials, plogis(truth[time]))
  fit <- tratio(y | trials ~ 1, data = d, family = ratiod_binomial(), mode = "pg",
    temporal = temporal_multiscale("time", trend = "rw2", seasonal = NULL,
                                    short_term = "none"),
    priors = ratiod_priors(beta = prior_normal(0, 0.05),
                           sigma_temporal = prior_pc(0.5, 0.1)),
    control = list(iter = 7000L, warmup = 2000L, chains = 1L, cores = 1L,
                    seed = 219L, verbose = FALSE))
  trend <- fit$.internal$temporal_draws$trend
  intervals <- apply(trend, 2, quantile, c(0.005, 0.995))
  expect_true(all(intervals[1, ] < truth & truth < intervals[2, ]))
  scale <- fit$.internal$chain_results[[1]]$sigma_trend
  if (is.null(scale)) scale <- as.numeric(fit$draws[, , "sigma_trend"])
  interval <- quantile(scale, c(0.025, 0.975))
  expect_true(interval[1] < sigma && sigma < interval[2])
  expect_gt(sd(scale), 0)
  eta <- fit$.internal$eta
  beta <- as.numeric(fit$draws[, , "beta_num[1]"])
  expect_equal(eta, trend[, time, drop = FALSE] + beta, tolerance = 1e-12)
})
