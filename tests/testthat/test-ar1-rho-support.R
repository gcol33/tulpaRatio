# The AR1 correlation's support and its prior (gcol33/tulpaRatio#67).
#
# temporal_ar1()'s rho used to be mapped onto (0, 1) while every other AR1
# correlation in the package -- the TVC arm, the multi-scale short-term arm and
# the spatiotemporal interaction -- lived on (-1, 1), so a negative
# autocorrelation was not reachable from that block and the documented
# Uniform(-1, 1) described neither the family nor the support the code used.
# `rho_prior` was stored on the spec and never read again.

# -----------------------------------------------------------------------------
# The prior a block carries
# -----------------------------------------------------------------------------

test_that("a correlation prior must be a Beta", {
  expect_error(temporal_ar1("year", rho_prior = prior_gamma(2, 1)), "Beta prior")
  expect_error(temporal_ar1("year", rho_prior = 0.8), "Beta prior")
  expect_null(temporal_ar1("year")$rho_prior)
})

test_that("the block's prior wins over the model-wide one", {
  block <- prior_beta(9, 1)
  model <- prior_beta(1, 4)
  expect_equal(unname(tulpaRatio:::rho_prior_anchors(block, model)), c(9, 1))
  expect_equal(unname(tulpaRatio:::rho_prior_anchors(NULL, model)), c(1, 4))
  # Beta(2, 2) on (rho + 1) / 2 is what priors_default() documents.
  expect_equal(unname(tulpaRatio:::rho_prior_anchors(NULL, NULL)), c(2, 2))
})

# -----------------------------------------------------------------------------
# The support
# -----------------------------------------------------------------------------

test_that("a negatively autocorrelated series recovers a negative rho", {
  skip_on_cran()

  # An alternating temporal effect: the configuration a rho pinned to (0, 1)
  # cannot express at all. Under the old mapping the posterior piles up at the
  # lower edge of (0, 1) and the recovered rho is positive by construction.
  set.seed(20260824)
  n_times <- 24
  n_per_time <- 40
  n <- n_times * n_per_time

  rho_true <- -0.75
  temporal_effect <- numeric(n_times)
  temporal_effect[1] <- rnorm(1, 0, 0.6)
  for (t in 2:n_times) {
    temporal_effect[t] <- rho_true * temporal_effect[t - 1] + rnorm(1, 0, 0.4)
  }

  time_idx <- rep(seq_len(n_times), each = n_per_time)
  x <- rnorm(n)
  p <- plogis(-0.2 + 0.4 * x + temporal_effect[time_idx])
  trials <- rpois(n, 20) + 10L
  successes <- rbinom(n, trials, p)
  X <- cbind(1, x)

  result <- tulpaRatio:::cpp_hmc_fit(
    q_init = rep(0, 2 + 1 + 1 + 1 + n_times),
    y_num = as.integer(successes),
    y_denom = as.integer(trials),
    y_num_cont = rep(0.0, n),
    y_denom_cont = rep(0.0, n),
    X_num = X,
    X_denom = matrix(1, n, 1),
    family = list(model_type = "binomial"),
    re_params = make_re_params_temporal(n),
    spatial_params = make_spatial_params_temporal(n),
    temporal_params = make_temporal_params(n, time_idx, type = "ar1",
                                           n_times = n_times),
    prior_params = make_prior_params_temporal(),
    zi_params = make_zi_params_temporal(n),
    latent_params = make_latent_params_temporal(),
    st_params = make_st_params_temporal(),
    tvc_params = make_tvc_params_temporal(),
    svc_params = make_svc_params_temporal(),
    n_iter = 800L,
    n_warmup = 400L,
    L = 10L,
    n_chains = 1L,
    seed = 7L,
    n_threads = 1L,
    verbose = FALSE
  )

  # logit_rho sits after the two numerator coefficients, the denominator
  # intercept and log_tau.
  draws <- result$samples[, 5]
  rho <- 2 * plogis(draws) - 1
  expect_true(all(rho > -1 & rho < 1))
  expect_lt(mean(rho), 0)
  expect_lt(quantile(rho, 0.9), 0)
})

test_that("the correlation prior reaches the density", {
  skip_on_cran()

  # Two fits on the same data differing only in the Beta anchors. The prior is
  # placed on u = (rho + 1) / 2, so Beta(9, 1) pushes rho up and Beta(1, 9)
  # pushes it down; with a short series and a weak signal the posterior follows.
  # An inert rho_prior leaves the two identical.
  set.seed(20260825)
  n_times <- 6
  n_per_time <- 8
  n <- n_times * n_per_time
  time_idx <- rep(seq_len(n_times), each = n_per_time)
  x <- rnorm(n)
  p <- plogis(-0.2 + 0.3 * x)
  trials <- rpois(n, 8) + 5L
  successes <- rbinom(n, trials, p)
  X <- cbind(1, x)

  fit_at <- function(a, b) {
    tp <- make_temporal_params(n, time_idx, type = "ar1", n_times = n_times)
    tp$rho_prior_a <- a
    tp$rho_prior_b <- b
    res <- tulpaRatio:::cpp_hmc_fit(
      q_init = rep(0, 2 + 1 + 1 + 1 + n_times),
      y_num = as.integer(successes),
      y_denom = as.integer(trials),
      y_num_cont = rep(0.0, n),
      y_denom_cont = rep(0.0, n),
      X_num = X,
      X_denom = matrix(1, n, 1),
      family = list(model_type = "binomial"),
      re_params = make_re_params_temporal(n),
      spatial_params = make_spatial_params_temporal(n),
      temporal_params = tp,
      prior_params = make_prior_params_temporal(),
      zi_params = make_zi_params_temporal(n),
      latent_params = make_latent_params_temporal(),
      st_params = make_st_params_temporal(),
      tvc_params = make_tvc_params_temporal(),
      svc_params = make_svc_params_temporal(),
      n_iter = 800L,
      n_warmup = 400L,
      L = 10L,
      n_chains = 1L,
      seed = 11L,
      n_threads = 1L,
      verbose = FALSE
    )
    mean(2 * plogis(res$samples[, 5]) - 1)
  }

  up <- fit_at(9, 1)
  down <- fit_at(1, 9)
  expect_gt(up, down)
})
