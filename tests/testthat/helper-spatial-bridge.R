spatial_covariance_kernels <- list(exponential = function(d, s2, phi) s2 * exp(-d/phi), matern = function(d,
    s2, phi) {
    u <- sqrt(3) * d/phi
    s2 * (1 + u) * exp(-u)
}, gaussian = function(d, s2, phi) s2 * exp(-0.5 * (d/phi)^2), spherical = function(d, s2, phi) {
    r <- d/phi
    ifelse(d >= phi, 0, s2 * (1 - 1.5 * r + 0.5 * r^3))
}, matern52 = function(d, s2, phi) {
    u <- sqrt(5) * d/phi
    s2 * (1 + u + u * u/3) * exp(-u)
})

SPATIAL_COVARIANCE_CODES <- setNames(seq_along(spatial_covariance_kernels) - 1L, names(spatial_covariance_kernels))

spatial_bridge_fixture <- function(seed) {
    set.seed(seed)
    S <- 6L
    adjacency <- matrix(0, S, S)
    for (s in seq_len(S - 1L)) adjacency[s, s + 1L] <- adjacency[s + 1L, s] <- 1
    coords <- cbind(c(0, 0.2, 0.4, 0.7, 0.8, 1), c(0.2, 0.8, 0.1, 0.6, 1, 0.3))
    truth <- c(-0.6, -0.3, -0.1, 0.1, 0.3, 0.6)
    raw <- expand.grid(unit = seq_len(S), x = c(-1.2, -0.4, 0.4, 1.2), replicate = 1:10)
    raw$n <- 30L
    raw$y <- rbinom(nrow(raw), raw$n, plogis(0.2 + 0.5 * raw$x + truth[raw$unit]))
    d <- aggregate(cbind(y, n) ~ unit + x, data = raw, FUN = sum)
    stopifnot(nrow(raw) == 240L, nrow(d) == 24L, sum(d$n) == 7200L, sum(d$y) == sum(raw$y))
    d$lon <- coords[d$unit, 1]
    d$lat <- coords[d$unit, 2]
    list(S = S, adjacency = adjacency, coords = coords, truth = truth, raw = raw, data = d, eta = 0.2 +
        0.5 * d$x + truth[d$unit])
}

spatial_comparison_draws <- function(draws) {
    out <- as.matrix(draws)
    precision <- startsWith(colnames(out), "tau_")
    out[, precision] <- log(out[, precision, drop = FALSE])
    colnames(out)[precision] <- paste0("log_", colnames(out)[precision])
    out
}
