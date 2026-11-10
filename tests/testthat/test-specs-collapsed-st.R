test_that("collapsed ST GP requests use joint engine coordinates and recover planted fields", {
  skip_on_cran()
  if (!'st_params' %in% names(formals(tulpaRatio:::cpp_tulpaRatio_run_nuts_specs)))
    skip('Requires the ST assembly in the concurrent temporal bridge for gcol33/tulpaRatio#176')
  op <- options(tulpaRatio.use_specs = FALSE)
  on.exit(options(op))
  calls <- 0L
  trace('cpp_tulpaRatio_run_nuts_specs', where = asNamespace('tulpaRatio'),
    tracer = function() calls <<- calls + 1L, print = FALSE)
  on.exit(untrace('cpp_tulpaRatio_run_nuts_specs', where = asNamespace('tulpaRatio')), add = TRUE)
  adjacency <- matrix(c(0, 1, 0, 1, 0, 1, 0, 1, 0), 3)
  for (seed in c(18L, 181L, 182L)) for (type in c('separable', 'nonsep_gp')) {
    set.seed(seed)
    d <- data.frame(unit = rep(1:3, each = 48L), time = rep(rep(1:4, each = 12L), 3), n = 40L)
    d$x <- rnorm(nrow(d)); d$lon <- c(.1, .5, .9)[d$unit]; d$lat <- c(.2, -.3, .6)[d$unit]
    field <- c(-.5, 0, .5)[d$unit] * c(-.6, -.2, .2, .6)[d$time]
    eta <- .2 + .5 * d$x + field
    d$y <- rbinom(nrow(d), d$n, plogis(eta))
    main <- spatial_car(adjacency, group_var = 'unit')
    st <- spatiotemporal(main, temporal_rw1('time'), type = type,
      parameterization = 'collapsed', coords = ~ lon + lat, nn = 4L,
      nonsep_type = if (type == 'nonsep_gp') 'gneiting' else 'product')
    fit_one <- function(engine) {
      options(tulpaRatio.use_specs = engine)
      before <- calls
      timing <- system.time(fit <- suppressWarnings(tratio(y | n ~ x, data = d,
        family = ratiod_binomial(), spatial = main, spatiotemporal = st, mode = 'hmc',
        control = list(iter = 8000L, warmup = 2500L, chains = 1L, cores = 1L,
          seed = seed, verbose = FALSE, adapt_delta = .99,
          gradient_mode = if (engine) 'A_r' else 'H'))))
      expect_equal(calls, before + as.integer(engine), info = type)
      list(fit = fit, seconds = timing[['elapsed']])
    }
    a <- fit_one(FALSE); b <- fit_one(TRUE)
    x <- spatial_comparison_draws(a$fit$draws); y <- spatial_comparison_draws(b$fit$draws)
    columns <- intersect(colnames(x), colnames(y))
    x <- x[, columns, drop = FALSE]; y <- y[, columns, drop = FALSE]
    mean_gap <- max(abs(colMeans(x) - colMeans(y)) / apply(x, 2, sd))
    quantile_gap <- max(abs(sweep(apply(x, 2, quantile, c(.1, .5, .9)) -
      apply(y, 2, quantile, c(.1, .5, .9)), 2, apply(x, 2, sd), '/')))
    ess <- function(z) min(vapply(seq_len(ncol(z)), function(j) posterior::ess_bulk(z[, j]), numeric(1)))
    metrics <- data.frame(type = type, seed = seed, legacy_seconds = a$seconds,
      joint_seconds = b$seconds, legacy_ess_sec = ess(x) / a$seconds,
      joint_ess_sec = ess(y) / b$seconds, mean_gap = mean_gap, quantile_gap = quantile_gap,
      legacy_divergences = a$fit$diagnostics$n_divergent, joint_divergences = b$fit$diagnostics$n_divergent)
    message(paste(capture.output(print(metrics, row.names = FALSE)), collapse = '\n'))
    evidence <- Sys.getenv('TULPA_BRIDGE_EVIDENCE', '')
    if (nzchar(evidence)) saveRDS(list(legacy = a, engine = b, metrics = metrics),
      file.path(evidence, paste0('collapsed-st-', type, '-', seed, '.rds')))
    expect_lt(mean_gap, .35, label = paste(type, seed))
    expect_lt(quantile_gap, .65, label = paste(type, seed))
    expect_true(all(abs(colMeans(y)[1:2] - c(.2, .5)) < .2))
    fitted_eta <- qlogis(fitted(b$fit, summary = FALSE)$numerator)
    expect_lt(sqrt(mean((colMeans(fitted_eta) - eta)^2)), .2)
    expect_equal(b$fit$diagnostics$n_divergent, 0L)
  }
})
