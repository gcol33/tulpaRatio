spatial_covariance_kernels <- list(
  exponential = function(d, s2, phi) s2 * exp(-d / phi),
  matern = function(d, s2, phi) {
    u <- sqrt(3) * d / phi
    s2 * (1 + u) * exp(-u)
  },
  gaussian = function(d, s2, phi) s2 * exp(-0.5 * (d / phi)^2),
  spherical = function(d, s2, phi) {
    r <- d / phi
    ifelse(d >= phi, 0, s2 * (1 - 1.5 * r + 0.5 * r^3))
  },
  matern52 = function(d, s2, phi) {
    u <- sqrt(5) * d / phi
    s2 * (1 + u + u * u / 3) * exp(-u)
  }
)

SPATIAL_COVARIANCE_CODES <- setNames(seq_along(spatial_covariance_kernels) - 1L,
                                   names(spatial_covariance_kernels))

spatial_bridge_fixture <- function(seed) {
  set.seed(seed)
  S <- 6L
  adjacency <- matrix(0, S, S)
  for (s in seq_len(S - 1L)) adjacency[s, s + 1L] <- adjacency[s + 1L, s] <- 1
  coords <- cbind(c(0, .2, .4, .7, .8, 1), c(.2, .8, .1, .6, 1, .3))
  truth <- c(-.6, -.3, -.1, .1, .3, .6)
  raw <- expand.grid(unit = seq_len(S), x = c(-1.2, -.4, .4, 1.2), replicate = 1:10)
  raw$n <- 30L
  raw$y <- rbinom(nrow(raw), raw$n, plogis(.2 + .5 * raw$x + truth[raw$unit]))
  d <- aggregate(cbind(y, n) ~ unit + x, data = raw, FUN = sum)
  stopifnot(nrow(raw) == 240L, nrow(d) == 24L, sum(d$n) == 7200L, sum(d$y) == sum(raw$y))
  d$lon <- coords[d$unit, 1]; d$lat <- coords[d$unit, 2]
  list(S = S, adjacency = adjacency, coords = coords, truth = truth, raw = raw,
    data = d, eta = .2 + .5 * d$x + truth[d$unit])
}

spatial_comparison_draws <- function(draws) {
  out <- as.matrix(draws)
  precision <- startsWith(colnames(out), 'tau_')
  out[, precision] <- log(out[, precision, drop = FALSE])
  colnames(out)[precision] <- paste0('log_', colnames(out)[precision])
  out
}

with_legacy_sampler <- function(code) {
  old_options <- options(tulpaRatio.use_specs = FALSE)
  old_environment <- Sys.getenv('TULPARATIO_USE_SPECS', unset = NA_character_)
  on.exit({
    options(old_options)
    if (is.na(old_environment)) Sys.unsetenv('TULPARATIO_USE_SPECS') else
      Sys.setenv(TULPARATIO_USE_SPECS = old_environment)
  }, add = TRUE)
  Sys.unsetenv('TULPARATIO_USE_SPECS')
  eval.parent(substitute(code))
}
