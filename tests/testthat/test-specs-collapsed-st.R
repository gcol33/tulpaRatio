test_that("collapsed ST GP requests use joint engine coordinates and recover planted fields", {
  skip_on_cran()
  calls <- 0L
  trace('cpp_tulpaRatio_sample', where = asNamespace('tulpaRatio'),
    tracer = function() calls <<- calls + 1L, print = FALSE)
  on.exit(untrace('cpp_tulpaRatio_sample', where = asNamespace('tulpaRatio')), add = TRUE)
  adjacency <- matrix(c(0, 1, 0, 1, 0, 1, 0, 1, 0), 3)
  for (seed in c(18L, 181L, 182L)) for (type in c('separable', 'nonsep_gp')) {
    set.seed(seed)
    raw <- expand.grid(unit = 1:3, time = 1:4, x = c(-1.2, -.4, .4, 1.2), replicate = 1:3)
    raw$n <- 40L
    raw_field <- c(-.8, 0, .8)[raw$unit] * c(-1.2, -.4, .4, 1.2)[raw$time]
    raw$y <- rbinom(nrow(raw), raw$n, plogis(.2 + .5 * raw$x + raw_field))
    d <- aggregate(cbind(y, n) ~ unit + time + x, data = raw, FUN = sum)
    stopifnot(nrow(raw) == 144L, nrow(d) == 48L, sum(d$n) == 5760L, sum(d$y) == sum(raw$y))
    d$lon <- c(.1, .5, .9)[d$unit]; d$lat <- c(.2, -.3, .6)[d$unit]
    field <- c(-.8, 0, .8)[d$unit] * c(-1.2, -.4, .4, 1.2)[d$time]
    eta <- .2 + .5 * d$x + field
    expect_gt(sqrt(mean(field^2)), .2)
    main <- spatial_car(adjacency, group_var = 'unit')
    st <- spatiotemporal(main, temporal_rw1('time'), type = type,
      parameterization = 'collapsed', coords = ~ lon + lat, nn = 4L,
      nonsep_type = if (type == 'nonsep_gp') 'gneiting' else 'product')
    fit_one <- function() {
      before <- calls
      timing <- system.time(fit <- suppressWarnings(tratio(y | n ~ x, data = d,
        family = ratiod_binomial(), spatial = main, spatiotemporal = st, mode = 'hmc',
        control = list(iter = 8000L, warmup = 2500L, chains = 1L, cores = 1L,
          seed = seed, verbose = FALSE, adapt_delta = .99,
          gradient_mode = 'A_r'))))
      expect_equal(calls, before + 1L, info = type)
      out <- list(fit = fit, seconds = timing[['elapsed']])
      evidence <- Sys.getenv('TULPA_BRIDGE_EVIDENCE', '')
      if (nzchar(evidence)) saveRDS(out, file.path(evidence,
        paste0('st-fit-engine-', type, '-', seed, '.rds')))
      out
    }
    b <- fit_one()
    y <- spatial_comparison_draws(b$fit$draws)
    expect_true(all(abs(colMeans(y)[1:2] - c(.2, .5)) < .2))
    fitted_eta <- qlogis(fitted(b$fit, summary = FALSE)$numerator)
    expect_lt(sqrt(mean((colMeans(fitted_eta) - eta)^2)), .2)
    expect_equal(b$fit$diagnostics$n_divergent, 0L)
  }
})
