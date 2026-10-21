# The Polya-Gamma BYM2 sampler and HMC target the same posterior: same field,
# same priors (half-Cauchy on sigma, uniform on rho).

grid_adjacency <- function(k) {
  n <- k * k
  adj <- matrix(0L, n, n)
  id <- function(r, c) (r - 1L) * k + c
  for (r in seq_len(k)) for (c in seq_len(k)) {
    if (c < k) adj[id(r, c), id(r, c + 1L)] <- adj[id(r, c + 1L), id(r, c)] <- 1L
    if (r < k) adj[id(r, c), id(r + 1L, c)] <- adj[id(r + 1L, c), id(r, c)] <- 1L
  }
  adj
}

test_that("PG and HMC agree on a BYM2 binomial model", {
  skip_on_cran()

  set.seed(42)
  k <- 5L
  J <- k * k
  reps <- 6L
  field <- as.numeric(scale(cumsum(rnorm(J)))) * 0.6
  field <- field - mean(field)
  unit <- rep(seq_len(J), each = reps)
  x1 <- rnorm(length(unit))
  nt <- sample(10:25, length(unit), replace = TRUE)
  y <- rbinom(length(unit), nt, plogis(-0.3 + 0.5 * x1 + field[unit]))
  d <- data.frame(y = y, n_trials = nt, x1 = x1,
                  unit = factor(unit, levels = seq_len(J)))
  sp <- spatial_bym2(grid_adjacency(k), level = "group", group_var = "unit")
  ctrl <- list(iter = 3000, warmup = 1000, chains = 2, verbose = FALSE, seed = 7)

  f_pg <- tratio(y | n_trials ~ x1, data = d, family = ratiod_binomial(),
                 spatial = sp, mode = "pg", control = ctrl)
  f_hmc <- tratio(y | n_trials ~ x1, data = d, family = ratiod_binomial(),
                  spatial = sp, mode = "hmc", control = ctrl)

  draw <- function(fit, v) posterior::as_draws_df(fit$draws)[[v]]
  pairs <- list(
    c("(Intercept)", "beta_num[1]"),
    c("x1", "beta_num[2]"),
    c("sigma_spatial", "sigma_spatial"),
    c("rho", "rho_spatial")
  )
  for (p in pairs) {
    a <- draw(f_pg, p[1])
    b <- draw(f_hmc, p[2])
    expect_lt(abs(mean(a) - mean(b)), 0.35 * sd(b), label = p[1])
    expect_lt(abs(log(sd(a) / sd(b))), log(1.3), label = p[1])
  }
})
