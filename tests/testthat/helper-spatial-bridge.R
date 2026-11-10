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
