test_that("numerator-only temporal fields preserve process inclusion", {
  skip_on_cran()
  op <- options(tulpaRatio.use_specs = FALSE)
  on.exit(options(op), add = TRUE)
  calls <- 0L
  trace('cpp_tulpaRatio_run_nuts_specs', where = asNamespace('tulpaRatio'),
    tracer = function() calls <<- calls + 1L, print = FALSE)
  on.exit(untrace('cpp_tulpaRatio_run_nuts_specs', where = asNamespace('tulpaRatio')),
    add = TRUE)
  for (seed in c(18L, 181L, 182L)) {
    set.seed(seed)
    d <- data.frame(time = rep(1:4, each = 30L), x = rnorm(120L))
    field <- c(-.6, -.2, .2, .6)
    d$y <- rgamma(nrow(d), shape = 10,
      rate = 10 / exp(.2 + .4 * d$x + field[d$time]))
    d$z <- rgamma(nrow(d), shape = 12, rate = 12 / exp(.3 + .2 * d$x))
    args <- list(formula = y | z ~ x, data = d, family = ratiod_gamma_gamma(),
      temporal = temporal_rw1('time', shared = FALSE), mode = 'hmc',
      control = list(iter = 4000L, warmup = 1500L, chains = 1L, cores = 1L,
        seed = seed, verbose = FALSE, metric = 'dense', adapt_delta = .95,
        gradient_mode = 'A_r'))
    options(tulpaRatio.use_specs = FALSE)
    legacy <- do.call(tratio, args)
    options(tulpaRatio.use_specs = TRUE)
    before <- calls
    fit <- do.call(tratio, args)
    expect_equal(calls, before + 1L)
    a <- as.matrix(legacy$draws); b <- as.matrix(fit$draws)
    expect_identical(colnames(a), colnames(b))
    means <- abs(colMeans(a) - colMeans(b)) / apply(a, 2, sd)
    quantiles <- max(abs(sweep(apply(a, 2, quantile, c(.1, .5, .9)) -
      apply(b, 2, quantile, c(.1, .5, .9)), 2, apply(a, 2, sd), '/')))
    message(sprintf('SHARING seed=%d mean_legacy_sd=%.6f quantile_legacy_sd=%.6f',
      seed, max(means), quantiles))
    expect_lt(max(means), .35, label = paste(seed, names(which.max(means))))
    expect_lt(quantiles, .65)
    expect_true(all(abs(colMeans(b)[1:4] - c(.2, .4, .3, .2)) < .15))
    expect_equal(fit$diagnostics$n_divergent, 0L)
  }
})
