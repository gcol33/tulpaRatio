# test-specs-temporal.R
# B1d Step 3 parity test: RW1 + AR1 temporal effects routed through the
# LikelihoodSpec path produce posterior means within 4x within-MC noise (in posterior-sd units) of
# the legacy backend at the same seed.

simulate_temporal_binomial <- function(n_times = 16L, reps = 12L, type = "rw1") {
  set.seed(20260603)
  if (type == "ar1") {
    rho <- 0.6
    phi <- numeric(n_times)
    phi[1] <- rnorm(1, sd = 0.4)
    for (t in 2:n_times) {
      phi[t] <- rho * phi[t - 1] + rnorm(1, sd = 0.3)
    }
  } else {
    eps <- rnorm(n_times, sd = 0.25)
    phi <- cumsum(eps)
    phi <- phi - mean(phi)
  }

  time <- rep(seq_len(n_times), each = reps)
  n    <- length(time)
  x1   <- rnorm(n)
  eta  <- 0.4 + 0.5 * x1 + phi[time]
  p    <- plogis(eta)
  nt   <- sample(15:30, n, replace = TRUE)
  y    <- rbinom(n, nt, p)

  list(
    formula = y | n_trials ~ x1,
    data    = data.frame(y = y, n_trials = nt, x1 = x1,
                          t = factor(time, levels = seq_len(n_times))),
    family  = tulpaRatio::ratiod_binomial()
  )
}

fit_one_temporal <- function(sim, use_specs, seed_val, temporal_struct) {
  op <- options(tulpaRatio.use_specs = use_specs); on.exit(options(op), add = TRUE)
  fit <- tulpaRatio::tratio(
    formula = sim$formula, data = sim$data, family = sim$family,
    temporal = temporal_struct,
    mode = "hmc",
    control = list(iter = 3000L, warmup = 1000L, chains = 1L, seed = seed_val, verbose = FALSE, gradient_mode = "A_r")
  )
  fit$draws
}

# Largest posterior-mean difference between two fits, in units of the pooled
# posterior sd of each parameter.
max_standardised_diff <- function(a, b) {
  sd_pool <- sqrt((apply(a, 2, var) + apply(b, 2, var)) / 2)
  max(abs(colMeans(a) - colMeans(b)) / sd_pool)
}

temporal_constructor <- function(type_v) {
  switch(type_v,
    rw1 = tulpaRatio::temporal_rw1(time_var = "t"),
    ar1 = tulpaRatio::temporal_ar1(time_var = "t"),
    stop("unknown temporal type: ", type_v)
  )
}

for (t_type in c("rw1", "ar1")) {
  local({
    type_v <- t_type
    test_that(sprintf("B1d spec path matches legacy for %s temporal", type_v), {
      skip_on_cran()
      # RW1/RW2 carry a sum-to-zero penalty whose constant the legacy path now
      # scales as a standard deviation; the spec path delegates to the tulpa
      # engine, which still reads 0.001 as a precision. AR1 is proper and
      # carries no such penalty, so it stays comparable.
      if (identical(type_v, "rw1")) skip("blocked on gcol33/tulpa#241")
      sim <- simulate_temporal_binomial(type = type_v)
      tp  <- temporal_constructor(type_v)
      legacy42 <- fit_one_temporal(sim, FALSE, 42L, tp)
      specs42  <- fit_one_temporal(sim, TRUE,  42L, tp)
      legacy43 <- fit_one_temporal(sim, FALSE, 43L, tp)

      expect_identical(colnames(specs42), colnames(legacy42))
      cross  <- max_standardised_diff(legacy42, specs42)
      within <- max_standardised_diff(legacy42, legacy43)
      expect_lt(cross, max(4 * within, 0.5))
    })
  })
}
