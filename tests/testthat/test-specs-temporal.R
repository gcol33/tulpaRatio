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
    truth = p,
    family  = tulpaRatio::ratiod_binomial()
  )
}

fit_one_temporal <- function(sim, seed_val, temporal_struct) {
  fit <- tulpaRatio::tratio(
    formula = sim$formula, data = sim$data, family = sim$family,
    temporal = temporal_struct,
    mode = "hmc",
    control = list(iter = 3000L, warmup = 1000L, chains = 1L, seed = seed_val, verbose = FALSE, gradient_mode = "A_r")
  )
  fit
}

temporal_constructor <- function(type_v) {
  switch(type_v,
    rw1 = tulpaRatio::temporal_rw1(time_var = "t", parameterization = "centered"),
    ar1 = tulpaRatio::temporal_ar1(time_var = "t", parameterization = "centered"),
    stop("unknown temporal type: ", type_v)
  )
}

for (t_type in c("rw1", "ar1")) {
  local({
    type_v <- t_type
    test_that(sprintf("engine recovers %s temporal predictors", type_v), {
      skip_on_cran()
      sim <- simulate_temporal_binomial(type = type_v)
      tp  <- temporal_constructor(type_v)
      fit <- fit_one_temporal(sim, 42L, tp)
      expect_lt(abs(mean(fit$draws[, "beta_num[2]"]) - .5), .2)
      expect_lt(sqrt(mean((colMeans(fitted(fit, summary = FALSE)$numerator) - sim$truth)^2)), .1)
    })
  })
}
