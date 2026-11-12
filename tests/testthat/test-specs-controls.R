test_that("engine sampler controls recover the binomial posterior across seeds", {
  skip_on_cran()
  d <- data.frame(y = c(4L, 8L, 10L, 12L, 7L, 9L, 6L, 8L), n = 15L)
  prior_sd <- 2.5
  log_density <- function(beta) sum(d$y) * beta -
    sum(d$n) * log1p(exp(beta)) - beta^2 / (2 * prior_sd^2)
  peak <- optimize(log_density, c(-8, 8), maximum = TRUE)$objective
  density <- function(beta) exp(log_density(beta) - peak)
  norm <- integrate(density, -15, 15, rel.tol = 1e-11)$value
  mean <- integrate(function(x) x * density(x), -15, 15)$value / norm
  variance <- integrate(function(x) (x - mean)^2 * density(x), -15, 15)$value / norm
  quantiles <- vapply(c(.1, .5, .9), function(p)
    uniroot(function(x) integrate(density, -15, x)$value / norm - p,
      c(-8, 8), tol = 1e-9)$root, numeric(1))
  calls <- 0L
  trace('cpp_tulpaRatio_run_nuts_specs', where = asNamespace('tulpaRatio'),
    tracer = function() calls <<- calls + 1L, print = FALSE)
  on.exit(untrace('cpp_tulpaRatio_run_nuts_specs', where = asNamespace('tulpaRatio')))
  for (seed in c(18L, 181L, 182L)) for (metric in c('auto', 'diag', 'dense', 'block_diag')) {
    for (algorithm in c('NUTS', 'HMC', 'WALNUTS')) {
      control <- list(iter = 3000L, warmup = 1000L, chains = 1L, cores = 1L,
        seed = seed, verbose = FALSE, adapt_delta = .9, metric = metric,
        L = if (algorithm == 'HMC') 7L else 0L, walnuts = algorithm == 'WALNUTS')
      fit <- tratio(y | n ~ 1, d, family = ratiod_binomial(), mode = 'hmc',
        priors = ratiod_priors(beta = prior_normal(0, prior_sd)), control = control)
      b <- as.matrix(fit$draws)[, 1]
      expect_identical(fit$algorithm, algorithm)
      expect_lt(abs(mean(b) - mean) / sqrt(variance), .2)
      expect_lt(max(abs(quantile(b, c(.1, .5, .9)) - quantiles)) / sqrt(variance), .3)
    }
  }
  expect_equal(calls, 36L)
})

test_that("WALNUTS requires a NUTS trajectory length", {
  d <- data.frame(y = 2L, n = 3L)
  expect_error(tratio(y | n ~ 1, d, family = ratiod_binomial(), mode = "hmc",
    control = list(iter = 20L, warmup = 10L, chains = 1L, L = 5L, walnuts = TRUE)),
    "walnuts requires control\\$L = 0")
})
