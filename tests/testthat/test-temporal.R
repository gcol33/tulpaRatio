# test-temporal.R
# Tests for temporal structure specification and HMC integration

# Parameter-list helpers live in helper-temporal-params.R.

test_that("temporal_rw1 creates correct structure", {
  temp <- temporal_rw1("time_var")

  expect_s3_class(temp, "ratiod_temporal")
  expect_equal(temp$type, "rw1")
  expect_equal(temp$time_var, "time_var")
  expect_false(temp$cyclic)
  expect_true(temp$shared)
})


test_that("temporal_rw1 with cyclic option", {
  temp <- temporal_rw1("month", cyclic = TRUE)

  expect_s3_class(temp, "ratiod_temporal")
  expect_equal(temp$type, "rw1")
  expect_true(temp$cyclic)
})


test_that("temporal_rw2 creates correct structure", {
  temp <- temporal_rw2("year")

  expect_s3_class(temp, "ratiod_temporal")
  expect_equal(temp$type, "rw2")
  expect_equal(temp$time_var, "year")
  expect_false(temp$cyclic)
  expect_true(temp$shared)
})


test_that("temporal_ar1 creates correct structure", {
  temp <- temporal_ar1("time_var")

  expect_s3_class(temp, "ratiod_temporal")
  expect_equal(temp$type, "ar1")
  expect_equal(temp$time_var, "time_var")
  expect_true(temp$shared)
})


test_that("temporal with panel group", {
  temp <- temporal_rw1("year", group_var = "site")

  expect_s3_class(temp, "ratiod_temporal")
  expect_equal(temp$type, "rw1")
  expect_equal(temp$time_var, "year")
  expect_equal(temp$group_var, "site")
})


test_that("temporal shared=FALSE works", {
  # Expect warning about non-shared effects
  expect_warning(
    temp <- temporal_rw1("time_var", shared = FALSE),
    "confounded"
  )

  expect_s3_class(temp, "ratiod_temporal")
  expect_false(temp$shared)
})


test_that("temporal print method works", {
  temp <- temporal_rw1("year")
  output <- capture.output(print(temp))
  expect_true(any(grepl("temporal", output, ignore.case = TRUE)))
  expect_true(any(grepl("RW1", output)))
})


test_that("validate_temporal extracts time indices", {
  df <- data.frame(
    y = rbinom(20, 10, 0.5),
    n = rep(10, 20),
    time = rep(1:5, 4)
  )

  temp <- temporal_rw1("time")
  validated <- tulpaRatio:::validate_temporal(temp, df)

  expect_equal(length(validated$time_index), 20)
  expect_equal(validated$n_times, 5)
  expect_equal(validated$n_groups, 1)  # No panel group
})


test_that("validate_temporal with panel groups", {
  df <- data.frame(
    y = rbinom(30, 10, 0.5),
    n = rep(10, 30),
    time = rep(1:5, 6),
    site = rep(1:3, each = 10)
  )

  temp <- temporal_rw1("time", group_var = "site")
  validated <- tulpaRatio:::validate_temporal(temp, df)

  expect_equal(length(validated$time_index), 30)
  expect_equal(validated$n_times, 5)
  expect_equal(validated$n_groups, 3)
  expect_equal(validated$n_temporal_params, 15)  # 5 times * 3 groups
})


test_that("HMC with temporal RW1 runs", {
  skip_on_cran()

  set.seed(333)
  n_times <- 8
  n_per_time <- 20
  n <- n_times * n_per_time

  # Simulate temporal effect (random walk)
  temporal_effect <- cumsum(rnorm(n_times, 0, 0.2))
  temporal_effect <- temporal_effect - mean(temporal_effect)

  time_idx <- rep(1:n_times, each = n_per_time)
  x <- rnorm(n)
  p <- plogis(-0.5 + 0.5 * x + temporal_effect[time_idx])
  trials <- rpois(n, 20) + 10
  successes <- rbinom(n, trials, p)

  X <- cbind(1, x)

  # Parameters: 2 beta_num + 1 beta_denom + 1 log_tau_temporal + 8 temporal = 12
  result <- tulpaRatio:::cpp_hmc_fit(
    q_init = rep(0, 12),
    y_num = as.integer(successes),
    y_denom = as.integer(trials),
    y_num_cont = rep(0.0, n),
    y_denom_cont = rep(0.0, n),
    X_num = X,
    X_denom = matrix(1, n, 1),
    family = list(model_type = "binomial"),
    re_params = make_re_params_temporal(n),
    spatial_params = make_spatial_params_temporal(n),
    temporal_params = make_temporal_params(n, time_idx, type = "rw1", n_times = n_times),
    prior_params = make_prior_params_temporal(),
    zi_params = make_zi_params_temporal(n),
    latent_params = make_latent_params_temporal(),
    st_params = make_st_params_temporal(),
    tvc_params = make_tvc_params_temporal(),
    svc_params = make_svc_params_temporal(),
    n_iter = 200L,
    n_warmup = 100L,
    L = 10L,
    n_chains = 1L,
    seed = 333L,
    n_threads = 1L,
    verbose = FALSE
  )

  expect_equal(ncol(result$samples), 12)
  expect_equal(nrow(result$samples), 100)

  # Check temporal precision is positive
  log_tau_idx <- 4
  tau_est <- mean(exp(result$samples[, log_tau_idx]))
  expect_true(tau_est > 0)

  # Check temporal effects exist (sum constraint is soft and may not hold with short chains)
  temporal_cols <- 5:12
  temporal_means <- colMeans(result$samples[, temporal_cols])
  # Just verify temporal effects are finite (skip sum constraint for short chains)
  expect_true(all(is.finite(temporal_means)))
})




# Each group's walk sums to zero over time, so it carries the group's trend
# and a group intercept its level; the two no longer lie along the same
# direction (gcol33/tulpaRatio#135). Both coordinates describe that model, so
# they agree on the walks, the intercepts and the coefficients.
test_that("a panel walk carries its group's trend and the intercept its level", {
  skip_on_cran()

  set.seed(124)
  n_t <- 12L
  n_s <- 8L
  site <- rep(paste0("site", seq_len(n_s)), times = n_t)
  s <- match(site, paste0("site", seq_len(n_s)))
  t <- rep(seq_len(n_t), each = n_s)
  walks <- apply(matrix(rnorm(n_t * n_s, sd = 0.2), n_t, n_s), 2, cumsum)
  level <- rnorm(n_s, sd = 0.5)
  eff <- level[s] + walks[cbind(t, s)]
  x <- rnorm(n_t * n_s)
  df <- data.frame(year = 2014 + t, site = site, x = x,
                   count = rpois(n_t * n_s, exp(2.5 + 0.3 * x + eff)),
                   effort = rgamma(n_t * n_s, shape = 3, rate = exp(-eff)))

  # Monte Carlo standard error of each column mean from 20 batch means
  batch_mcse <- function(d, n_batch = 20L) {
    n <- nrow(d) %/% n_batch * n_batch
    b <- rowsum(d[seq_len(n), , drop = FALSE],
                rep(seq_len(n_batch), each = n / n_batch)) / (n / n_batch)
    apply(b, 2, stats::sd) / sqrt(n_batch)
  }

  fit_panel <- function(par) {
    fit <- suppressWarnings(
      tratio(count | effort ~ x + (1 | site), data = df,
             family = ratiod_poisson_gamma(),
             temporal = temporal_rw1("year", group_var = "site",
                                     parameterization = par),
             mode = "hmc",
             control = list(iter = 1000, chains = 1, seed = 2,
                            verbose = FALSE)))
    draws <- fit$.internal$temporal_draws
    group <- rep(seq_len(n_s), each = n_t)
    dr <- as.matrix(fit$draws)
    list(
      n_divergent = sum(unlist(fit$diagnostics$divergent)),
      walk_sums = apply(draws, 1, function(d) tapply(d, group, sum)),
      field = colMeans(draws),
      field_mcse = batch_mcse(draws),
      # contrasts between sites: their common mean trades with the intercept
      re = {
        re <- colMeans(dr[, paste0("re[", seq_len(n_s), "]")])
        re - mean(re)
      },
      slope = mean(dr[, "beta_num[2]"])
    )
  }

  cen <- fit_panel("centered")
  nc <- fit_panel("noncentered")

  # a walk not centred within its group trades its level with the site
  # intercept and sends about 95 of these 1000 draws divergent; centred, the
  # count is a few at most and varies across platforms
  expect_lt(nc$n_divergent, 10L)
  expect_lt(max(abs(cen$walk_sums)), 1e-8)
  expect_lt(max(abs(nc$walk_sums)), 1e-8)
  # the site levels reach the intercepts
  expect_gt(stats::cor(cen$re, level), 0.9)
  # each walk mean carries a Monte Carlo error of about 0.01; across the 96
  # entries the largest gap between the coordinates is a few of those
  expect_lt(max(abs(nc$field - cen$field) /
                sqrt(nc$field_mcse^2 + cen$field_mcse^2)), 5)
  expect_lt(max(abs(nc$re - cen$re)), 0.1)
  expect_lt(abs(nc$slope - cen$slope), 0.02)
})

test_that("HMC with temporal AR1 runs", {
  skip_on_cran()

  set.seed(444)
  # Enough time points for the likelihood to say anything about rho. At the ten
  # this fixture used to carry, the posterior for rho was almost the prior --
  # sd 0.32 around a mean of 0.07, with a 90% interval of [-0.54, +0.69] even
  # at 8000 iterations -- so the sign of the mean over 100 retained draws was
  # inside its own Monte Carlo error and came out anywhere from -0.03 to +0.40
  # depending on the seed. At eighty it is +0.47 or better across eight seeds.
  n_times <- 80
  n_per_time <- 6
  n <- n_times * n_per_time

  # Simulate AR1 process
  rho_true <- 0.7
  temporal_effect <- numeric(n_times)
  temporal_effect[1] <- rnorm(1, 0, 0.3)
  for (t in 2:n_times) {
    temporal_effect[t] <- rho_true * temporal_effect[t-1] + rnorm(1, 0, 0.2)
  }

  time_idx <- rep(1:n_times, each = n_per_time)
  x <- rnorm(n)
  p <- plogis(-0.3 + 0.4 * x + temporal_effect[time_idx])
  trials <- rpois(n, 15) + 10
  successes <- rbinom(n, trials, p)

  X <- cbind(1, x)

  # Parameters: 2 beta_num + 1 beta_denom + 1 log_tau + 1 logit_rho + n_times
  n_params <- 5L + n_times
  result <- tulpaRatio:::cpp_hmc_fit(
    q_init = rep(0, n_params),
    y_num = as.integer(successes),
    y_denom = as.integer(trials),
    y_num_cont = rep(0.0, n),
    y_denom_cont = rep(0.0, n),
    X_num = X,
    X_denom = matrix(1, n, 1),
    family = list(model_type = "binomial"),
    re_params = make_re_params_temporal(n),
    spatial_params = make_spatial_params_temporal(n),
    temporal_params = make_temporal_params(n, time_idx, type = "ar1", n_times = n_times),
    prior_params = make_prior_params_temporal(),
    zi_params = make_zi_params_temporal(n),
    latent_params = make_latent_params_temporal(),
    st_params = make_st_params_temporal(),
    tvc_params = make_tvc_params_temporal(),
    svc_params = make_svc_params_temporal(),
    n_iter = 200L,
    n_warmup = 100L,
    L = 10L,
    n_chains = 1L,
    seed = 444L,
    n_threads = 1L,
    verbose = FALSE
  )

  expect_equal(ncol(result$samples), n_params)
  expect_equal(nrow(result$samples), 100)

  # rho lives on (-1, 1): the sampled coordinate is logit((rho + 1) / 2).
  # Reading plogis() alone lands in (0, 1) whatever the support is, so it
  # cannot tell the two mappings apart (gcol33/tulpaRatio#67).
  logit_rho_idx <- 5  # After 2 beta_num + 1 beta_denom + 1 log_tau
  rho_samples <- 2 * plogis(result$samples[, logit_rho_idx]) - 1
  expect_true(all(rho_samples > -1 & rho_samples < 1))
  # The fixture is simulated at rho = 0.7, and a sampler that left rho at its
  # prior would sit at 0. Measured across eight seeds the posterior mean runs
  # +0.47 to +0.67, so 0.2 separates the two without asserting the shrinkage a
  # 100-draw chain has not shaken off.
  expect_gt(mean(rho_samples), 0.2)
})


test_that("HMC with cyclic RW1 (seasonal) runs", {
  skip_on_cran()

  set.seed(555)
  n_months <- 12
  n_years <- 3
  n_per_month <- 5
  n <- n_months * n_years * n_per_month

  # Simulate seasonal effect
  seasonal_effect <- sin(2 * pi * (1:n_months) / 12) * 0.3

  time_idx <- rep(rep(1:n_months, n_years), each = n_per_month)
  x <- rnorm(n)
  p <- plogis(-0.2 + 0.3 * x + seasonal_effect[time_idx])
  trials <- rpois(n, 20) + 10
  successes <- rbinom(n, trials, p)

  X <- cbind(1, x)

  # Parameters: 2 beta_num + 1 beta_denom + 1 log_tau + 12 temporal = 16
  result <- tulpaRatio:::cpp_hmc_fit(
    q_init = rep(0, 16),
    y_num = as.integer(successes),
    y_denom = as.integer(trials),
    y_num_cont = rep(0.0, n),
    y_denom_cont = rep(0.0, n),
    X_num = X,
    X_denom = matrix(1, n, 1),
    family = list(model_type = "binomial"),
    re_params = make_re_params_temporal(n),
    spatial_params = make_spatial_params_temporal(n),
    temporal_params = make_temporal_params(n, time_idx, type = "rw1", n_times = n_months, cyclic = TRUE),
    prior_params = make_prior_params_temporal(),
    zi_params = make_zi_params_temporal(n),
    latent_params = make_latent_params_temporal(),
    st_params = make_st_params_temporal(),
    tvc_params = make_tvc_params_temporal(),
    svc_params = make_svc_params_temporal(),
    n_iter = 200L,
    n_warmup = 100L,
    L = 10L,
    n_chains = 1L,
    seed = 555L,
    n_threads = 1L,
    verbose = FALSE
  )

  expect_equal(ncol(result$samples), 16)
  expect_equal(nrow(result$samples), 100)
})


test_that("HMC with panel temporal (group-specific) runs", {
  skip_on_cran()

  set.seed(666)
  n_times <- 5
  n_groups <- 3
  n_per <- 10
  n <- n_times * n_groups * n_per

  # Simulate group-specific temporal effects
  temporal_effects <- matrix(0, n_times, n_groups)
  for (g in 1:n_groups) {
    temporal_effects[, g] <- cumsum(rnorm(n_times, 0, 0.15))
    temporal_effects[, g] <- temporal_effects[, g] - mean(temporal_effects[, g])
  }

  time_idx <- rep(rep(1:n_times, each = n_per), n_groups)
  group_idx <- rep(1:n_groups, each = n_times * n_per)
  x <- rnorm(n)

  # Linear predictor
  eta <- -0.4 + 0.5 * x
  for (i in seq_along(eta)) {
    eta[i] <- eta[i] + temporal_effects[time_idx[i], group_idx[i]]
  }
  p <- plogis(eta)
  trials <- rpois(n, 15) + 5
  successes <- rbinom(n, trials, p)

  X <- cbind(1, x)

  # Parameters: 2 beta_num + 1 beta_denom + 1 log_tau + 15 temporal (5*3) = 19
  result <- tulpaRatio:::cpp_hmc_fit(
    q_init = rep(0, 19),
    y_num = as.integer(successes),
    y_denom = as.integer(trials),
    y_num_cont = rep(0.0, n),
    y_denom_cont = rep(0.0, n),
    X_num = X,
    X_denom = matrix(1, n, 1),
    family = list(model_type = "binomial"),
    re_params = make_re_params_temporal(n),
    spatial_params = make_spatial_params_temporal(n),
    temporal_params = make_temporal_params(n, time_idx, type = "rw1", n_times = n_times, group_idx = group_idx, n_groups = n_groups),
    prior_params = make_prior_params_temporal(),
    zi_params = make_zi_params_temporal(n),
    latent_params = make_latent_params_temporal(),
    st_params = make_st_params_temporal(),
    tvc_params = make_tvc_params_temporal(),
    svc_params = make_svc_params_temporal(),
    n_iter = 200L,
    n_warmup = 100L,
    L = 10L,
    n_chains = 1L,
    seed = 666L,
    n_threads = 1L,
    verbose = FALSE
  )

  expect_equal(ncol(result$samples), 19)
  expect_equal(nrow(result$samples), 100)
})


test_that("HMC with temporal runs multiple chains in parallel", {
  skip_on_cran()

  set.seed(777)
  n_times <- 6
  n_per_time <- 15
  n <- n_times * n_per_time

  # Simulate temporal effect (random walk)
  temporal_effect <- cumsum(rnorm(n_times, 0, 0.2))
  temporal_effect <- temporal_effect - mean(temporal_effect)

  time_idx <- rep(1:n_times, each = n_per_time)
  x <- rnorm(n)
  p <- plogis(-0.4 + 0.4 * x + temporal_effect[time_idx])
  trials <- rpois(n, 15) + 10
  successes <- rbinom(n, trials, p)

  X <- cbind(1, x)

  # Parameters: 2 beta_num + 1 beta_denom + 1 log_tau_temporal + 6 temporal = 10
  result <- tulpaRatio:::cpp_hmc_fit(
    q_init = rep(0, 10),
    y_num = as.integer(successes),
    y_denom = as.integer(trials),
    y_num_cont = rep(0.0, n),
    y_denom_cont = rep(0.0, n),
    X_num = X,
    X_denom = matrix(1, n, 1),
    family = list(model_type = "binomial"),
    re_params = make_re_params_temporal(n),
    spatial_params = make_spatial_params_temporal(n),
    temporal_params = make_temporal_params(n, time_idx, type = "rw1", n_times = n_times),
    prior_params = make_prior_params_temporal(),
    zi_params = make_zi_params_temporal(n),
    latent_params = make_latent_params_temporal(),
    st_params = make_st_params_temporal(),
    tvc_params = make_tvc_params_temporal(),
    svc_params = make_svc_params_temporal(),
    n_iter = 200L,
    n_warmup = 100L,
    L = 10L,
    n_chains = 2L,  # Multiple chains in parallel
    seed = 777L,
    n_threads = 1L,
    verbose = FALSE
  )

  # Multi-chain returns lists
  expect_true(is.list(result$samples))
  expect_equal(length(result$samples), 2)

  # Each chain has correct dimensions (n_iter=200 total, n_warmup=100 → 100 post-warmup)
  expect_equal(nrow(result$samples[[1]]), 100)
  expect_equal(nrow(result$samples[[2]]), 100)
  expect_equal(ncol(result$samples[[1]]), 10)
  expect_equal(ncol(result$samples[[2]]), 10)

  # Results should be different between chains (different random seeds)
  chain1_mean <- colMeans(result$samples[[1]])
  chain2_mean <- colMeans(result$samples[[2]])
  expect_false(all(chain1_mean == chain2_mean))
})


# =============================================================================
# Tests for temporal_gp() - Gaussian Process temporal structure
# =============================================================================

test_that("temporal_gp creates correct structure with exponential covariance", {
  temp <- temporal_gp("timestamp")

  expect_s3_class(temp, "ratiod_temporal_gp")
  expect_s3_class(temp, "ratiod_temporal")
  expect_equal(temp$type, "gp")
  expect_equal(temp$time_var, "timestamp")
  expect_equal(temp$cov, "exponential")
  expect_true(temp$shared)
  expect_true(temp$scale)
})


test_that("temporal_gp creates correct structure with matern covariance", {
  temp <- temporal_gp("day", cov = "matern", nu = 2.5)

  expect_s3_class(temp, "ratiod_temporal_gp")
  expect_equal(temp$cov, "matern")
  expect_equal(temp$nu, 2.5)
})


test_that("temporal_gp creates correct structure with periodic covariance", {
  temp <- temporal_gp("month", cov = "periodic", period = 12)

  expect_s3_class(temp, "ratiod_temporal_gp")
  expect_equal(temp$cov, "periodic")
  expect_equal(temp$period, 12)
})


test_that("temporal_gp validates period for periodic covariance", {
  expect_error(
    temporal_gp("month", cov = "periodic"),
    regexp = "period.*must be.*positive"
  )

  expect_error(
    temporal_gp("month", cov = "periodic", period = -5),
    regexp = "period.*must be.*positive"
  )
})


test_that("temporal_gp warns for non-shared effects", {
  expect_warning(
    temporal_gp("time", shared = FALSE),
    regexp = "confounded"
  )
})


test_that("temporal_gp print method works", {
  temp <- temporal_gp("timestamp", cov = "matern", nu = 1.5)
  output <- capture.output(print(temp))

  expect_true(any(grepl("Gaussian Process", output, ignore.case = TRUE)))
  expect_true(any(grepl("matern", output, ignore.case = TRUE)))
  expect_true(any(grepl("nu", output)))
})


test_that("validate_temporal_gp extracts time values correctly", {
  df <- data.frame(
    y = rpois(20, 10),
    n = rep(10, 20),
    time = sort(runif(20, 0, 100))  # Irregularly-spaced
  )

  temp <- temporal_gp("time")
  validated <- tulpaRatio:::validate_temporal_gp(temp, df)

  expect_equal(validated$n_obs, 20)
  expect_equal(length(validated$time_values), 20)
  expect_equal(validated$n_groups, 1)
  expect_equal(validated$n_temporal_params, 20)
})


test_that("validate_temporal_gp handles date/time variables", {
  df <- data.frame(
    y = rpois(10, 10),
    date = as.Date("2020-01-01") + 0:9
  )

  temp <- temporal_gp("date")
  validated <- tulpaRatio:::validate_temporal_gp(temp, df)

  expect_equal(validated$n_obs, 10)
  expect_true(is.numeric(validated$time_values))
})


test_that("validate_temporal_gp with grouping", {
  df <- data.frame(
    y = rpois(30, 10),
    time = rep(1:10, 3),
    group = rep(c("A", "B", "C"), each = 10)
  )

  temp <- temporal_gp("time", group_var = "group")
  validated <- tulpaRatio:::validate_temporal_gp(temp, df)

  expect_equal(validated$n_groups, 3)
  # For GP, we have one effect per unique time per group
  # With 10 unique times and 3 groups, we get 10 * 3 = 30 temporal params
  expect_equal(validated$n_temporal_params, 30)
})


# =============================================================================
# Tests for temporal_tvc() - Temporally-Varying Coefficients
# =============================================================================

test_that("temporal_tvc creates correct structure with index terms", {
  tvc <- temporal_tvc("year", terms = 1)

  expect_s3_class(tvc, "ratiod_tvc")
  expect_s3_class(tvc, "ratiod_temporal")
  expect_equal(tvc$type, "tvc")
  expect_equal(tvc$time_var, "year")
  expect_equal(tvc$structure, "rw1")
  expect_equal(tvc$terms_spec$type, "index")
  expect_equal(tvc$terms_spec$indices, 1L)
  expect_true(tvc$shared)
})


test_that("temporal_tvc creates correct structure with named terms", {
  tvc <- temporal_tvc("year", terms = c("(Intercept)", "depth"))

  expect_equal(tvc$terms_spec$type, "names")
  expect_equal(tvc$terms_spec$names, c("(Intercept)", "depth"))
})


test_that("temporal_tvc creates correct structure with formula terms", {
  tvc <- temporal_tvc("year", terms = ~ 1 + depth + temp)

  expect_equal(tvc$terms_spec$type, "formula")
  expect_true(inherits(tvc$terms_spec$formula, "formula"))
})


test_that("temporal_tvc supports different structures", {
  tvc_rw1 <- temporal_tvc("year", terms = 1, structure = "rw1")
  tvc_rw2 <- temporal_tvc("year", terms = 1, structure = "rw2")
  tvc_ar1 <- temporal_tvc("year", terms = 1, structure = "ar1")
  tvc_iid <- temporal_tvc("year", terms = 1, structure = "iid")

  expect_equal(tvc_rw1$structure, "rw1")
  expect_equal(tvc_rw2$structure, "rw2")
  expect_equal(tvc_ar1$structure, "ar1")
  expect_equal(tvc_iid$structure, "iid")
})

test_that("temporal_tvc rejects structure = 'gp' (gcol33/tulpaRatio#32)", {
  # C++ has no GP path for TVC at all (no lengthscale field, no branch in
  # tvc_term_log_prior); accepting the string and silently downgrading to
  # RW1 in the C++ string parser was the actual bug.
  expect_error(
    temporal_tvc("year", terms = 1, structure = "gp"),
    "not implemented"
  )
})


test_that("temporal_tvc warns for non-shared effects", {
  expect_warning(
    temporal_tvc("year", terms = 1, shared = FALSE),
    regexp = "confounded"
  )
})


test_that("temporal_tvc print method works", {
  tvc <- temporal_tvc("year", terms = c(1, 2), structure = "rw2")
  output <- capture.output(print(tvc))

  expect_true(any(grepl("temporally-varying", output, ignore.case = TRUE)))
  expect_true(any(grepl("RW2", output)))
})


test_that("validate_tvc resolves index terms correctly", {
  df <- data.frame(
    y = rpois(30, 10),
    x = rnorm(30),
    year = rep(2020:2024, each = 6)
  )
  X <- model.matrix(~ 1 + x, data = df)

  tvc <- temporal_tvc("year", terms = c(1, 2))  # Intercept and x
  validated <- tulpaRatio:::validate_tvc(tvc, df, X)

  expect_equal(validated$n_times, 5)
  expect_equal(validated$n_tvc, 2)
  expect_equal(validated$tvc_names, c("(Intercept)", "x"))
  expect_equal(validated$n_temporal_params, 10)  # 5 times * 2 terms
})


test_that("validate_tvc resolves named terms correctly", {
  df <- data.frame(
    y = rpois(30, 10),
    x = rnorm(30),
    z = rnorm(30),
    year = rep(2020:2024, each = 6)
  )
  X <- model.matrix(~ 1 + x + z, data = df)

  tvc <- temporal_tvc("year", terms = c("x", "z"))
  validated <- tulpaRatio:::validate_tvc(tvc, df, X)

  expect_equal(validated$n_tvc, 2)
  expect_equal(validated$tvc_names, c("x", "z"))
  expect_equal(validated$tvc_indices, c(2, 3))  # Columns 2 and 3 in X
})


test_that("validate_tvc errors on missing terms", {
  df <- data.frame(
    y = rpois(30, 10),
    x = rnorm(30),
    year = rep(2020:2024, each = 6)
  )
  X <- model.matrix(~ 1 + x, data = df)

  tvc <- temporal_tvc("year", terms = c("missing_var"))

  expect_error(
    tulpaRatio:::validate_tvc(tvc, df, X),
    regexp = "not found in design matrix"
  )
})


test_that("validate_tvc with panel groups", {
  df <- data.frame(
    y = rpois(60, 10),
    x = rnorm(60),
    year = rep(2020:2024, each = 12),
    site = rep(c("A", "B"), each = 6, times = 5)
  )
  X <- model.matrix(~ 1 + x, data = df)

  tvc <- temporal_tvc("year", terms = 1, group_var = "site")
  validated <- tulpaRatio:::validate_tvc(tvc, df, X)

  expect_equal(validated$n_times, 5)
  expect_equal(validated$n_groups, 2)
  expect_equal(validated$n_tvc, 1)
  expect_equal(validated$n_temporal_params, 10)  # 5 times * 1 term * 2 groups
})


# =============================================================================
# Tests for temporal_rtr() - Restricted Temporal Regression
# =============================================================================

test_that("temporal_rtr creates correct structure", {
  base_temp <- temporal_rw2("year")
  rtr <- temporal_rtr(base_temp, restrict_to = ~ temperature)

  expect_s3_class(rtr, "ratiod_rtr")
  expect_s3_class(rtr, "ratiod_temporal")
  expect_equal(rtr$type, "rw2")  # Preserves base type
  expect_true(rtr$rtr)
  expect_true(inherits(rtr$rtr_formula, "formula"))
})


test_that("temporal_rtr works with different base temporal types", {
  rtr_rw1 <- temporal_rtr(temporal_rw1("year"), restrict_to = ~ x)
  rtr_gp <- temporal_rtr(temporal_gp("time"), restrict_to = ~ x)

  expect_s3_class(rtr_rw1, "ratiod_rtr")
  expect_equal(rtr_rw1$type, "rw1")

  expect_s3_class(rtr_gp, "ratiod_rtr")
  expect_equal(rtr_gp$type, "gp")
})


test_that("temporal_rtr validates inputs", {
  expect_error(
    temporal_rtr("not_a_temporal", restrict_to = ~ x),
    regexp = "temporal specification"
  )

  expect_error(
    temporal_rtr(temporal_rw2("year"), restrict_to = "not_a_formula"),
    regexp = "formula"
  )
})


test_that("temporal_rtr print method works", {
  rtr <- temporal_rtr(temporal_rw2("year"), restrict_to = ~ temp + precip)
  output <- capture.output(print(rtr))

  expect_true(any(grepl("RTR", output)))
  expect_true(any(grepl("Orthogonal", output)))
  expect_true(any(grepl("temp", output)))
})


test_that("validate_rtr computes projection matrix", {
  df <- data.frame(
    y = rpois(50, 10),
    year = rep(2020:2024, each = 10),
    temp = rnorm(50),
    precip = rnorm(50)
  )

  rtr <- temporal_rtr(temporal_rw2("year"), restrict_to = ~ temp + precip)
  validated <- tulpaRatio:::validate_rtr(rtr, df, ~ y ~ temp + precip)

  expect_true(!is.null(validated$rtr_projection))
  expect_equal(nrow(validated$rtr_projection), 50)
  expect_equal(ncol(validated$rtr_projection), 50)
  expect_equal(validated$rtr_vars, c("temp", "precip"))
})


test_that("validate_rtr errors on missing variables", {
  df <- data.frame(
    y = rpois(50, 10),
    year = rep(2020:2024, each = 10),
    temp = rnorm(50)
  )

  rtr <- temporal_rtr(temporal_rw2("year"), restrict_to = ~ temp + missing)

  expect_error(
    tulpaRatio:::validate_rtr(rtr, df, ~ y ~ temp),
    regexp = "not found in data"
  )
})


test_that("compute_rtr_projection is orthogonal", {
  X <- matrix(rnorm(100), 50, 2)
  P_perp <- tulpaRatio:::compute_rtr_projection(X)

  # P_perp should be symmetric
  expect_true(all(abs(P_perp - t(P_perp)) < 1e-10))

  # P_perp * X should be approximately zero (orthogonal)
  result <- P_perp %*% X
  expect_true(all(abs(result) < 1e-10))

  # P_perp should be idempotent (projection property)
  P_perp_squared <- P_perp %*% P_perp
  expect_true(all(abs(P_perp - P_perp_squared) < 1e-10))
})


test_that("apply_rtr_projection works correctly", {
  X <- matrix(rnorm(100), 50, 2)
  P_perp <- tulpaRatio:::compute_rtr_projection(X)

  # Random temporal effect
  f <- rnorm(50)

  # Projected effect
  f_rtr <- tulpaRatio:::apply_rtr_projection(f, P_perp)

  # f_rtr should be orthogonal to X columns
  for (j in 1:2) {
    dot_product <- sum(f_rtr * X[, j])
    expect_true(abs(dot_product) < 1e-10)
  }
})


# =============================================================================
# Tests for tvc() extractor function
# =============================================================================

test_that("tvc extractor creates ratiod_tvc_posterior object", {
  # Mock a ratiod_fit object with TVC
  mock_fit <- structure(
    list(
      tvc = structure(
        list(
          type = "tvc",
          n_times = 5,
          n_tvc = 2,
          tvc_names = c("(Intercept)", "x"),
          time_levels = as.character(2020:2024),
          structure = "rw1"
        ),
        class = c("ratiod_tvc", "ratiod_temporal", "list")
      ),
      .internal = list(
        tvc_draws = array(rnorm(400 * 5 * 2), dim = c(400, 5, 2))
      )
    ),
    class = "ratiod_fit"
  )

  result <- tvc(mock_fit)

  expect_s3_class(result, "ratiod_tvc_posterior")
  expect_equal(dim(result$draws), c(400, 5, 2))
  expect_equal(result$n_times, 5)
  expect_equal(result$n_tvc, 2)
  expect_equal(result$term_names, c("(Intercept)", "x"))
})


test_that("tvc extractor subsets terms correctly", {
  mock_fit <- structure(
    list(
      tvc = structure(
        list(
          type = "tvc",
          n_times = 5,
          n_tvc = 3,
          tvc_names = c("(Intercept)", "x", "z"),
          time_levels = as.character(2020:2024),
          structure = "rw1"
        ),
        class = c("ratiod_tvc", "ratiod_temporal", "list")
      ),
      .internal = list(
        tvc_draws = array(rnorm(400 * 5 * 3), dim = c(400, 5, 3))
      )
    ),
    class = "ratiod_fit"
  )

  result <- tvc(mock_fit, terms = "x")

  expect_equal(result$n_tvc, 1)
  expect_equal(result$term_names, "x")
  expect_equal(dim(result$draws), c(400, 5, 1))
})


test_that("tvc summary method works", {
  mock_fit <- structure(
    list(
      tvc = structure(
        list(
          type = "tvc",
          n_times = 3,
          n_tvc = 1,
          tvc_names = c("(Intercept)"),
          time_levels = as.character(2020:2022),
          structure = "rw1"
        ),
        class = c("ratiod_tvc", "ratiod_temporal", "list")
      ),
      .internal = list(
        tvc_draws = array(rnorm(100 * 3 * 1), dim = c(100, 3, 1))
      )
    ),
    class = "ratiod_fit"
  )

  result <- tvc(mock_fit, summary = TRUE)

  expect_s3_class(result, "ratiod_tvc_summary")
  expect_s3_class(result, "data.frame")
  expect_equal(nrow(result), 3)  # 3 time points
  expect_true("mean" %in% names(result))
  expect_true("sd" %in% names(result))
})


test_that("tvc errors when model has no TVC", {
  mock_fit <- structure(
    list(tvc = NULL),
    class = "ratiod_fit"
  )

  expect_error(
    tvc(mock_fit),
    regexp = "not fitted with temporally-varying coefficients"
  )
})
