test_that("spatial bridge preserves numerator-only fields and dispersion coordinates", {
  skip_on_cran()
  op <- options(tulpaRatio.use_specs = FALSE)
  on.exit(options(op))
  S <- 6L
  adjacency <- matrix(0, S, S)
  for (j in seq_len(S - 1L)) adjacency[j, j + 1L] <- adjacency[j + 1L, j] <- 1
  coords <- cbind(c(0, .2, .4, .7, .8, 1), c(.2, .8, .1, .6, 1, .3))
  for (seed in c(18L, 181L, 182L)) {
    set.seed(seed)
    d <- expand.grid(unit = seq_len(S), x = c(-1.2, -.4, .4, 1.2), replicate = 1:4)
    truth <- c(-.6, -.3, -.1, .1, .3, .6)
    d$lon <- coords[d$unit, 1]; d$lat <- coords[d$unit, 2]
    d$y <- rgamma(nrow(d), 30, scale = exp(.2 + .5 * d$x + truth[d$unit]) / 30)
    d$denom <- rgamma(nrow(d), 40, scale = exp(-.1 - .2 * d$x) / 40)
    fields <- list(
      car = spatial_car(adjacency, group_var = 'unit', proper = TRUE, shared = FALSE),
      gp = spatial_gp(~ lon + lat, scale_coords = FALSE, nn = 5L, shared = FALSE),
      hsgp = spatial_hsgp(~ lon + lat, scale_coords = FALSE, m = 3L, shared = FALSE),
      multiscale = spatial_multiscale(~ lon + lat, scale_coords = FALSE, m = 3L,
        shared = FALSE))
    for (name in names(fields)) {
      fit_one <- function(bridge) {
        options(tulpaRatio.use_specs = bridge)
        suppressWarnings(tratio(y | denom ~ x, data = d, family = ratiod_gamma_gamma(),
          spatial = fields[[name]], mode = 'hmc', control = list(iter = 8000L,
            warmup = 2500L, chains = 1L, cores = 1L, seed = seed, adapt_delta = .99,
            verbose = FALSE, gradient_mode = if (!bridge && identical(fields[[name]]$type, 'car_proper')) 'H' else 'A_r')))
      }
      legacy <- fit_one(FALSE); engine <- fit_one(TRUE)
      a <- as.matrix(legacy$draws); b <- as.matrix(engine$draws)
      expect_identical(colnames(a), colnames(b), info = name)
      mean_gap <- max(abs(colMeans(a) - colMeans(b)) / apply(a, 2, sd))
      quantile_gap <- max(abs(sweep(apply(a, 2, quantile, c(.1, .5, .9)) -
        apply(b, 2, quantile, c(.1, .5, .9)), 2, apply(a, 2, sd), '/')))
      evidence <- Sys.getenv('TULPA_BRIDGE_EVIDENCE', '')
      if (nzchar(evidence)) saveRDS(list(legacy = a, engine = b,
        mean_gap = mean_gap, quantile_gap = quantile_gap,
        legacy_diagnostics = legacy$diagnostics, engine_diagnostics = engine$diagnostics),
        file.path(evidence, paste0('pair-', name, '-', seed, '.rds')))
      message(sprintf('PAIR field=%s seed=%d mean_sd=%.6f quantile_sd=%.6f divergences=%d',
        name, seed, mean_gap, quantile_gap, engine$diagnostics$n_divergent))
      expect_lt(mean_gap, .35, label = paste(name, seed))
      expect_lt(quantile_gap, .65, label = paste(name, seed))
      expect_lt(abs(mean(b[, 'beta_num[2]']) - .5), .15, label = name)
      expect_lt(abs(mean(b[, 'beta_denom[2]']) + .2), .1, label = name)
    }
  }
})
