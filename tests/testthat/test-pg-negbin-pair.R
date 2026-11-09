test_that("paired NB dispersion likelihood uses its declared predictor scale", {
  y <- c(0L, 2L, 7L)
  eta <- c(-1, 0.2, 1.5)
  for (r in c(0.03, 0.3, 2, 900)) {
    expected <- sum(dnbinom(y, size = r, mu = r * exp(eta), log = TRUE))
    expect_equal(tulpaRatio:::cpp_test_pg_negbin_loglik(y, eta, r),
                 expected, tolerance = 1e-11)
    expected <- sum(dnbinom(y, size = r, mu = exp(eta), log = TRUE))
    expect_equal(tulpaRatio:::cpp_test_pg_negbin_loglik(y, eta, r, TRUE),
                 expected, tolerance = 1e-11)
  }
  for (eta in c(-100, 100))
    expect_true(is.finite(tulpaRatio:::cpp_test_pg_negbin_loglik(y, rep(eta, 3), 0.3)))
})

test_that("paired NB dispersion conditional agrees with normalized quadrature", {
  skip_on_cran()
  y <- c(0L, 0L, 1L, 7L, 0L, 3L)
  eta <- log(c(0.2, 0.2, 3, 7, 2, 2))
  x <- seq(-12, 8, length.out = 20001L)
  log_density <- vapply(exp(x), function(r)
    sum(dnbinom(y, mu = exp(eta), size = r, log = TRUE)) + dgamma(r, 2, 1, log = TRUE), 0) + x
  weights <- exp(log_density - max(log_density))
  cdf <- cumsum(weights) / sum(weights)
  probabilities <- c(0.1, 0.5, 0.9)
  keep <- !duplicated(cdf)
  reference <- approx(cdf[keep], exp(x)[keep], probabilities)$y
  set.seed(220)
  draws <- tulpaRatio:::cpp_test_pg_negbin_r_conditional(y, eta, 2, 1, 100000L)[-(1:10000)]
  expect_lt(max(abs(quantile(draws, probabilities) - reference)) / sd(draws), 0.08)
})

test_that("a paired NB sweep equals the exact Gaussian and marginal dispersion calculations", {
  X <- cbind(1, c(-1, 0.2, 1.3))
  yn <- c(0L, 2L, 7L)
  yd <- c(2L, 0L, 4L)
  rn <- 0.3
  rd <- 1.7
  beta_draw <- function(y, r) {
    omega <- tulpaRatio:::cpp_rpg_real(y + r, rep(-log(r), length(y)))
    Q <- crossprod(X, omega * X) + diag(1 / 2^2, ncol(X))
    rhs <- crossprod(X, (y - r) / 2 + omega * log(r))
    drop(solve(Q, rhs) + solve(chol(Q), rnorm(ncol(X))))
  }
  r_draw <- function(y, eta, r) {
    proposed <- exp(rnorm(1, log(r), 0.15))
    target <- function(r) sum(dnbinom(y, mu = exp(eta), size = r, log = TRUE)) +
      dgamma(r, 2, 0.5, log = TRUE) + log(r)
    if (log(runif(1)) < target(proposed) - target(r)) proposed else r
  }
  set.seed(220)
  bn <- beta_draw(yn, rn)
  bd <- beta_draw(yd, rd)
  expected_rn <- r_draw(yn, drop(X %*% bn), rn)
  expected_rd <- r_draw(yd, drop(X %*% bd), rd)
  set.seed(220)
  actual <- tulpaRatio:::cpp_pg_negbin_negbin_gibbs(yn, yd, X, X, rep(1L, 3), 0L,
    n_iter = 1L, n_warmup = 0L, thin = 1L, prior_beta_sd = 2,
    prior_sigma_scale = 1, prior_r_shape = 2, prior_r_rate = 0.5,
    r_num_init = rn, r_denom_init = rd, shared = FALSE, store_eta = TRUE,
    verbose = FALSE, n_threads = 1L)
  expect_equal(as.numeric(actual$beta_num), bn, tolerance = 1e-12)
  expect_equal(as.numeric(actual$beta_denom), bd, tolerance = 1e-12)
  expect_equal(actual$r_num, expected_rn, tolerance = 1e-12)
  expect_equal(actual$r_denom, expected_rd, tolerance = 1e-12)
  expect_equal(as.numeric(actual$eta_num), drop(X %*% bn), tolerance = 1e-12)
  expect_equal(as.numeric(actual$eta_denom), drop(X %*% bd), tolerance = 1e-12)
})

pg_pair_fixture <- function(iter = 6000L, warmup = 2000L, thin = 2L,
                            seed = 220L, n = 240L, groups = 12L) {
  set.seed(seed)
  g <- rep(seq_len(groups), each = n / groups)
  x <- rnorm(n, sd = 0.6)
  X <- cbind(1, x)
  truth <- list(beta_num = c(0.3, 0.5), beta_denom = c(0.8, -0.35),
                r_num = 0.8, r_denom = 1.7, sigma = 0.35)
  truth$re <- rnorm(groups, sd = truth$sigma)
  eta_num <- drop(X %*% truth$beta_num) + truth$re[g]
  eta_denom <- drop(X %*% truth$beta_denom) + truth$re[g]
  y_num <- rnbinom(n, mu = exp(eta_num), size = truth$r_num)
  y_denom <- rnbinom(n, mu = exp(eta_denom), size = truth$r_denom)
  set.seed(seed + 1L)
  fit <- tulpaRatio:::cpp_pg_negbin_negbin_gibbs(y_num, y_denom, X, X, g, groups,
    n_iter = iter, n_warmup = warmup, thin = thin, prior_beta_sd = 2,
    prior_sigma_scale = 1, prior_r_shape = 2, prior_r_rate = 0.5,
    r_num_init = 0.8, r_denom_init = 1.7, shared = TRUE, store_eta = TRUE,
    verbose = FALSE, n_threads = 1L)
  list(fit = fit, truth = truth, X = X, group = g, y_num = y_num, y_denom = y_denom)
}

test_that("paired NB saves the predictor of the stored coefficients and effects", {
  fixture <- pg_pair_fixture(iter = 50L, warmup = 20L)
  z <- fixture$fit
  for (arm in c("num", "denom")) {
    expected <- z[[paste0("beta_", arm)]] %*% t(fixture$X) + z$re[, fixture$group]
    expect_equal(z[[paste0("eta_", arm)]], expected, tolerance = 1e-12)
  }
})

test_that("paired NB posterior intervals cover simulated parameters across seeds", {
  skip_on_cran()
  skip_unless_slow()
  records <- list()
  for (seed in 220:239) {
    elapsed <- system.time(fixture <- pg_pair_fixture(iter = 4000L, warmup = 1000L,
      seed = seed, n = 120L, groups = 8L))[["elapsed"]]
    for (field in c("beta_num", "beta_denom", "r_num", "r_denom", "sigma_re", "re")) {
      draws <- as.matrix(fixture$fit[[field]])
      interval <- apply(draws, 2, quantile, c(0.025, 0.5, 0.975))
      truth <- if (field == "sigma_re") fixture$truth$sigma else fixture$truth[[field]]
      records[[paste(seed, field)]] <- data.frame(seed = seed, field = field,
        index = seq_along(truth), truth = truth, lower = interval[1, ],
        median = interval[2, ], upper = interval[3, ], seconds = elapsed,
        covered = interval[1, ] < truth & truth < interval[3, ])
    }
    record_path <- getOption("tulpa.pg.recovery.record")
    if (!is.null(record_path)) saveRDS(do.call(rbind, records), record_path)
  }
  coverage <- do.call(rbind, records)
  parameters <- split(subset(coverage, field != "re"),
    with(subset(coverage, field != "re"), paste(field, index)))
  for (parameter in parameters)
    expect_true(mean(parameter$covered) >= 0.85,
               info = paste(parameter$field[1], parameter$index[1]))
  effects <- subset(coverage, field == "re")
  expect_true(mean(effects$covered) >= 0.85)
})
