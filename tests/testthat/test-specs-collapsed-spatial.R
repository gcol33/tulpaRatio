test_that("joint engine fields and collapsed spatial posteriors agree across seeds", {
  skip_on_cran()
  op <- options(tulpaRatio.use_specs = FALSE)
  on.exit(options(op))
  calls <- 0L
  trace('cpp_tulpaRatio_run_nuts_specs', where = asNamespace('tulpaRatio'),
    tracer = function() calls <<- calls + 1L, print = FALSE)
  on.exit(untrace('cpp_tulpaRatio_run_nuts_specs', where = asNamespace('tulpaRatio')), add = TRUE)
  fixture <- spatial_bridge_fixture(1919L)
  adjacency <- fixture$adjacency
  truth <- fixture$truth
  d <- fixture$data
  fields <- list(
    icar = function(parameterization) spatial_car(adjacency, group_var = 'unit',
      parameterization = parameterization),
    bym2 = function(parameterization) spatial_bym2(adjacency, group_var = 'unit',
      parameterization = parameterization),
    gp = function(parameterization) spatial_gp(~ lon + lat, scale_coords = FALSE, nn = 5L,
      parameterization = parameterization))
  for (seed in c(18L, 181L, 182L)) for (name in names(fields)) {
    fit_one <- function(collapsed) {
      options(tulpaRatio.use_specs = !collapsed)
      parameterization <- 'collapsed'
      before <- calls
      timing <- system.time(fit <- suppressWarnings(tratio(y | n ~ x, data = d,
        family = ratiod_binomial(), spatial = fields[[name]](parameterization), mode = 'hmc',
        control = list(iter = 8000L, warmup = 2500L, chains = 1L, cores = 1L,
          seed = seed, verbose = FALSE, adapt_delta = .99,
          gradient_mode = if (collapsed) 'H' else 'A_r'))))
      expect_equal(calls, before + as.integer(!collapsed), info = name)
      list(fit = fit, seconds = timing[['elapsed']])
    }
    a <- fit_one(TRUE); b <- fit_one(FALSE)
    collapsed <- spatial_comparison_draws(a$fit$draws); sampled <- spatial_comparison_draws(b$fit$draws)
    columns <- intersect(colnames(collapsed), colnames(sampled))
    x <- collapsed[, columns, drop = FALSE]; y <- sampled[, columns, drop = FALSE]
    mean_gap <- max(abs(colMeans(x) - colMeans(y)) / apply(x, 2, sd))
    quantile_gap <- max(abs(sweep(apply(x, 2, quantile, c(.1, .5, .9)) -
      apply(y, 2, quantile, c(.1, .5, .9)), 2, apply(x, 2, sd), '/')))
    ess <- function(draws) min(vapply(seq_len(ncol(draws)), function(j)
      posterior::ess_bulk(draws[, j]), numeric(1)))
    ea <- ess(x); eb <- ess(y)
    expect_true(is.finite(ea) && ea > 0 && is.finite(eb) && eb > 0)
    message(sprintf(paste('COLLAPSED field=%s seed=%d mean_sd=%.6f quantile_sd=%.6f',
      'legacy_seconds=%.3f engine_seconds=%.3f legacy_ess_sec=%.3f engine_ess_sec=%.3f',
      'legacy_divergences=%d engine_divergences=%d'), name, seed, mean_gap, quantile_gap,
      a$seconds, b$seconds, ea / a$seconds, eb / b$seconds,
      a$fit$diagnostics$n_divergent, b$fit$diagnostics$n_divergent))
    evidence <- Sys.getenv('TULPA_BRIDGE_EVIDENCE', '')
    if (nzchar(evidence)) saveRDS(list(legacy = a, engine = b, mean_gap = mean_gap,
      quantile_gap = quantile_gap, legacy_ess = ea, engine_ess = eb),
      file.path(evidence, paste0('collapsed-', name, '-', seed, '.rds')))
    expect_lt(mean_gap, .35, label = paste(name, seed))
    expect_lt(quantile_gap, .65, label = paste(name, seed))
    expect_lt(abs(mean(y[, 'beta_num[2]']) - .5), .15, label = name)
    eta <- .2 + .5 * d$x + truth[d$unit]
    fitted_eta <- qlogis(fitted(b$fit, summary = FALSE)$numerator)
    expect_lt(sqrt(mean((colMeans(fitted_eta) - eta)^2)), .2, label = name)
  }
  options(tulpaRatio.use_specs = TRUE)
  for (name in names(fields)) {
    before <- calls
    fit <- suppressWarnings(tratio(y | n ~ x, data = d, family = ratiod_binomial(),
      spatial = fields[[name]]('collapsed'), mode = 'hmc', control = list(iter = 2000L,
        warmup = 750L, chains = 2L, cores = 2L, seed = 1919L, adapt_delta = .99,
        gradient_mode = 'A_r', verbose = FALSE)))
    expect_equal(calls, before + 1L, info = name)
    expect_equal(nrow(as.matrix(fit$draws)), 2500L)
    eta <- .2 + .5 * d$x + truth[d$unit]
    fitted_eta <- qlogis(fitted(fit, summary = FALSE)$numerator)
    expect_lt(sqrt(mean((colMeans(fitted_eta) - eta)^2)), .2, label = name)
  }
})
