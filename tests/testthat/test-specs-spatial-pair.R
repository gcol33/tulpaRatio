test_that("spatial bridge preserves numerator-only fields and dispersion coordinates", {
  skip_on_cran()
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
      fit_one <- function() {
        suppressWarnings(tratio(y | denom ~ x, data = d, family = ratiod_gamma_gamma(),
          spatial = fields[[name]], mode = 'hmc', control = list(iter = 8000L,
            warmup = 2500L, chains = 1L, cores = 1L, seed = seed, adapt_delta = .99,
            verbose = FALSE, gradient_mode = 'A_r')))
      }
      engine <- fit_one()
      b <- spatial_comparison_draws(engine$draws)
      expect_lt(abs(mean(b[, 'beta_num[2]']) - .5), .15, label = name)
      expect_lt(abs(mean(b[, 'beta_denom[2]']) + .2), .1, label = name)
    }
  }
})
