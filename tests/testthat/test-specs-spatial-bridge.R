test_that("spatial bridge posteriors agree with the model sampler and simulated predictors", {
  skip_on_cran()
  calls <- 0L
  trace('cpp_tulpaRatio_run_nuts_specs', where = asNamespace('tulpaRatio'),
    tracer = function() calls <<- calls + 1L, print = FALSE)
  on.exit(untrace('cpp_tulpaRatio_run_nuts_specs', where = asNamespace('tulpaRatio')))
  op <- options(tulpaRatio.use_specs = FALSE)
  on.exit(options(op), add = TRUE)
  for (seed in c(18L, 181L, 182L)) {
    fixture <- spatial_bridge_fixture(seed)
    adjacency <- fixture$adjacency; coords <- fixture$coords
    truth <- fixture$truth; raw <- fixture$raw
    d <- fixture$data; eta <- fixture$eta
    raw_r <- raw
    raw_r$x <- raw_r$x + 2 * truth[raw_r$unit]
    field_r <- qr.resid(qr(model.matrix(~ x, raw_r)), truth[raw_r$unit])
    set.seed(seed + 1919L)
    raw_r$y <- rbinom(nrow(raw_r), raw_r$n, plogis(.2 + .5 * raw_r$x + field_r))
    d_r <- aggregate(cbind(y, n) ~ unit + x, data = raw_r, FUN = sum)
    d_r$lon <- coords[d_r$unit, 1]; d_r$lat <- coords[d_r$unit, 2]
    eta_r <- .2 + .5 * d_r$x + qr.resid(qr(model.matrix(~ x, d_r)), truth[d_r$unit])
    stopifnot(nrow(d_r) == 24L, sum(d_r$n) == 7200L)
    fields <- list(
      car = spatial_car(adjacency, group_var = 'unit', proper = TRUE),
      car_uncentered = spatial_car(adjacency, group_var = 'unit', proper = TRUE, center = FALSE),
      gp_centered = spatial_gp(~ lon + lat, scale_coords = FALSE, nn = 5L, parameterization = 'centered'),
      gp_noncentered = spatial_gp(~ lon + lat, scale_coords = FALSE, nn = 5L, parameterization = 'noncentered'),
      gp_auto = spatial_gp(~ lon + lat, scale_coords = FALSE, nn = 5L, parameterization = 'auto'),
      gp_matern_half = spatial_gp(~ lon + lat, scale_coords = FALSE, nn = 5L, cov = 'matern', nu = .5),
      gp_matern32 = spatial_gp(~ lon + lat, scale_coords = FALSE, nn = 5L, cov = 'matern', nu = 1.5),
      gp_matern52 = spatial_gp(~ lon + lat, scale_coords = FALSE, nn = 5L, cov = 'matern', nu = 2.5),
      gp_gaussian = spatial_gp(~ lon + lat, scale_coords = FALSE, nn = 5L, cov = 'gaussian'),
      gp_spherical = spatial_gp(~ lon + lat, scale_coords = FALSE, nn = 5L, cov = 'spherical'),
      hsgp = spatial_hsgp(~ lon + lat, scale_coords = FALSE, m = 3L),
      multiscale = spatial_multiscale(~ lon + lat, scale_coords = FALSE, m = 3L),
      multiscale_lbfgs = spatial_multiscale(~ lon + lat, scale_coords = FALSE, m = 3L,
        sampler = 'lbfgs'),
      rsr_icar = spatial_rsr(spatial_car(adjacency, group_var = 'unit'), ~ x),
      rsr_bym2 = spatial_rsr(spatial_bym2(adjacency, group_var = 'unit'), ~ x),
      rsr_car = spatial_rsr(spatial_car(adjacency, group_var = 'unit', proper = TRUE), ~ x),
      rsr_gp = spatial_rsr(spatial_gp(~ lon + lat, scale_coords = FALSE, nn = 5L,
        parameterization = 'noncentered'), ~ x),
      rsr_hsgp = spatial_rsr(spatial_hsgp(~ lon + lat, scale_coords = FALSE, m = 3L), ~ x),
      rsr_multiscale = spatial_rsr(spatial_multiscale(~ lon + lat, scale_coords = FALSE, m = 3L), ~ x))
    for (name in names(fields)) {
      restricted <- startsWith(name, 'rsr_')
      args <- list(formula = y | n ~ x, data = if (restricted) d_r else d, family = ratiod_binomial(),
        spatial = fields[[name]], mode = 'hmc', control = list(iter = 8000L,
          warmup = 2500L, chains = 1L, cores = 1L, seed = seed,
          verbose = FALSE, adapt_delta = .99,
          gradient_mode = if (identical(fields[[name]]$type, 'car_proper')) 'H' else 'A_r'))
      options(tulpaRatio.use_specs = FALSE)
      legacy <- suppressWarnings(do.call(tratio, args))
      options(tulpaRatio.use_specs = TRUE)
      args$control$gradient_mode <- 'A_r'
      before <- calls
      fit <- suppressWarnings(do.call(tratio, args))
      expect_equal(calls, before + 1L, info = name)
      a <- spatial_comparison_draws(legacy$draws); b <- spatial_comparison_draws(fit$draws)
      expect_identical(colnames(a), colnames(b), info = name)
      mean_gaps <- abs(colMeans(a) - colMeans(b)) / apply(a, 2, sd)
      quantile_gaps <- abs(sweep(apply(a, 2, quantile, c(.1, .5, .9)) -
        apply(b, 2, quantile, c(.1, .5, .9)), 2, apply(a, 2, sd), '/'))
      mean_gap <- max(mean_gaps)
      quantile_gap <- max(quantile_gaps)
      evidence <- Sys.getenv('TULPA_BRIDGE_EVIDENCE', '')
      if (nzchar(evidence)) saveRDS(list(legacy = a, engine = b,
        mean_gaps = mean_gaps, quantile_gaps = quantile_gaps,
        legacy_diagnostics = legacy$diagnostics, engine_diagnostics = fit$diagnostics),
        file.path(evidence, paste0('spatial-', name, '-', seed, '.rds')))
      message(sprintf('SPATIAL field=%s seed=%d mean_sd=%.6f quantile_sd=%.6f divergences=%d',
        name, seed, mean_gap, quantile_gap, fit$diagnostics$n_divergent))
      expect_lt(mean_gap, .35, label = paste(name, seed, names(which.max(mean_gaps))))
      expect_lt(quantile_gap, .65, label = paste(name, seed))
      expect_lt(abs(mean(b[, 'beta_num[2]']) - .5), .15, label = name)
      fitted_eta <- qlogis(fitted(fit, summary = FALSE)$numerator)
      expect_lt(sqrt(mean((colMeans(fitted_eta) - if (restricted) eta_r else eta)^2)), .2, label = name)
    }
  }
})
