# A binomial ratio model over a chain of areal units: one structured field,
# a fixed intercept and slope, `reps` observations per unit.

build_chain_adjacency <- function(n_units, integer = TRUE) {
  adj <- matrix(if (integer) 0L else 0, n_units, n_units)
  if (n_units > 1L) {
    i <- seq_len(n_units - 1L)
    adj[cbind(i, i + 1L)] <- if (integer) 1L else 1
    adj[cbind(i + 1L, i)] <- if (integer) 1L else 1
  }
  adj
}

simulate_spatial_binomial <- function(n_units = 8L, reps = 16L) {
  set.seed(20260602)
  spatial_re <- rnorm(n_units, sd = 0.4)
  spatial_re <- spatial_re - mean(spatial_re)  # ICAR sum-to-zero

  unit <- rep(seq_len(n_units), each = reps)
  n    <- length(unit)
  x1   <- rnorm(n)
  eta  <- 0.5 + 0.6 * x1 + spatial_re[unit]
  p    <- plogis(eta)
  nt   <- sample(15:30, n, replace = TRUE)
  y    <- rbinom(n, nt, p)

  list(
    formula = y | n_trials ~ x1,
    data    = data.frame(y = y, n_trials = nt, x1 = x1,
                          unit = factor(unit, levels = seq_len(n_units))),
    family  = tulpaRatio::ratiod_binomial(),
    truth = p,
    adj     = build_chain_adjacency(n_units)
  )
}

