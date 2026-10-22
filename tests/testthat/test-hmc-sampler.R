# test-hmc-sampler.R
# Tests for HMC backend with bundled list arguments

# Helper function to create default parameter bundles
make_re_params <- function(n, group = rep(0L, n), n_groups = 0L, n_terms = 0L) {
  list(
    group = as.integer(group),
    n_groups = as.integer(n_groups),
    n_terms = as.integer(n_terms),
    group_matrix = matrix(0L, nrow = n, ncol = 1),
    n_groups_vec = as.integer(n_groups),
    has_slopes = FALSE,
    has_correlated_slopes = FALSE,
    n_coefs_vec = integer(0),
    correlated_vec = logical(0),
    n_chol_vec = integer(0),
    slope_matrices = list(),
    parameterization = 0L  # 0 = centered (default)
  )
}

make_spatial_params <- function(n, type = "none") {
  list(
    type = type,
    group = rep(0L, n),
    n_units = 0L,
    adj_row_ptr = 0L,
    adj_col_idx = integer(0),
    n_neighbors = integer(0),
    bym2_scale = 1.0
  )
}

make_temporal_params <- function(n) {
  list(
    type = "none",
    time_idx = rep(0L, n),
    group_idx = rep(0L, n),
    n_times = 0L,
    n_groups = 0L,
    n_params = 0L,
    cyclic = FALSE,
    shared = TRUE,
    tau_shape = 2.0,
    tau_rate = 0.5,
    # GP-specific fields (required even for non-GP types)
    time_values = numeric(0),
    cov_type = "exponential",
    nu = 1.5,
    period = 1.0,
    gp_sigma2_prior_U = 1.0,
    gp_sigma2_prior_alpha = 0.01,
    gp_phi_prior_lower = 0.01,
    gp_phi_prior_upper = 10.0
  )
}

make_prior_params <- function() {
  list(
    sigma_beta = 10.0,
    sigma_re_scale = 2.5,
    phi_shape = 2.0,
    phi_rate = 0.1,
    tau_spatial_shape = 1.0,
    tau_spatial_rate = 0.01
  )
}

make_zi_params <- function(n) {
  list(
    type = "none",
    X = matrix(0, nrow = n, ncol = 1),
    p_zi = 1L,
    prior_sd = 10.0,
    # OI fields (required even for non-OI types)
    X_oi = NULL,
    p_oi = 0L,
    oi_prior_sd = 10.0
  )
}

make_latent_params <- function() {
  list(
    has_latent = FALSE,
    n_factors = 0L,
    shared = FALSE,
    scale = TRUE,
    constraint = 0L,
    sigma_prior_rate = 0.0
  )
}

make_st_params <- function() {
  list(
    has_spatiotemporal = FALSE,
    type = "none",
    shared = TRUE,
    n_spatial = 0L,
    n_times = 0L,
    n_params = 0L,
    s_idx = integer(0),
    t_idx = integer(0),
    st_flat = integer(0),
    temporal_type = "rw1",
    temporal_cyclic = FALSE,
    adj_row_ptr = integer(0),
    adj_col_idx = integer(0),
    sigma2_prior_U = 1.0,
    sigma2_prior_alpha = 0.01
  )
}

make_tvc_params <- function() {
  list(
    has_tvc = FALSE,
    n_tvc = 0L,
    n_times = 0L,
    n_groups = 1L,
    structure = "rw1",
    time_idx = integer(0),
    group_idx = integer(0),
    X_tvc = matrix(0, nrow = 1, ncol = 1),
    sigma2_prior_U = 1.0,
    sigma2_prior_alpha = 0.01
  )
}

make_svc_params <- function() {
  list(
    has_svc = FALSE,
    n_svc = 0L,
    nn = 0L,
    shared = TRUE,
    cov_type = "exponential",
    spatial_idx = integer(0),
    X_svc = matrix(0, nrow = 1, ncol = 1),
    coords = matrix(0, nrow = 1, ncol = 2),
    sigma2_prior_U = 1.0,
    sigma2_prior_alpha = 0.01,
    phi_prior_shape = 3.0,
    phi_prior_rate = 1.0
  )
}

test_that("HMC backend runs for binomial model", {
  skip_on_cran()

  set.seed(123)
  n <- 200  # More data for better estimation
  x <- rnorm(n)
  p <- plogis(-0.5 + 0.8 * x)
  trials <- rpois(n, 30) + 15  # More trials per observation
  successes <- rbinom(n, trials, p)

  df <- data.frame(
    successes = successes,
    trials = trials,
    x = x
  )

  X <- cbind(1, x)

  result <- tulpaRatio:::cpp_hmc_fit(
    q_init = rep(0, 3),  # 2 beta_num + 1 beta_denom
    y_num = as.integer(successes),
    y_denom = as.integer(trials),
    y_num_cont = rep(0.0, n),
    y_denom_cont = rep(0.0, n),
    X_num = X,
    X_denom = matrix(1, n, 1),
    family = list(model_type = "binomial"),
    re_params = make_re_params(n),
    spatial_params = make_spatial_params(n),
    gp_params = list(gp_type = "none"),
    temporal_params = make_temporal_params(n),
    prior_params = make_prior_params(),
    zi_params = make_zi_params(n),
    latent_params = make_latent_params(),
    st_params = make_st_params(),
    tvc_params = make_tvc_params(),
    svc_params = make_svc_params(),
    n_iter = 200L,
    n_warmup = 100L,
    L = 10L,
    n_chains = 1L,
    seed = 123L,
    n_threads = 1L,
    verbose = FALSE
  )

  expect_true(is.matrix(result$samples))
  expect_equal(nrow(result$samples), 100)  # n_iter - n_warmup
  expect_equal(ncol(result$samples), 3)    # 2 beta_num + 1 beta_denom

  # Check intercept is approximately correct (wider tolerance)
  beta0_mean <- mean(result$samples[, 1])
  expect_true(abs(beta0_mean - (-0.5)) < 1.0)

  # Check slope is approximately correct (wider tolerance)
  beta1_mean <- mean(result$samples[, 2])
  expect_true(abs(beta1_mean - 0.8) < 1.0)
})


test_that("HMC backend runs for negbin model", {
  skip_on_cran()

  set.seed(456)
  n <- 100
  x <- rnorm(n)
  mu_num <- exp(2 + 0.5 * x)
  mu_denom <- exp(3 - 0.3 * x)
  phi <- 5

  y_num <- rnbinom(n, mu = mu_num, size = phi)
  y_denom <- rnbinom(n, mu = mu_denom, size = phi)

  X <- cbind(1, x)

  result <- tulpaRatio:::cpp_hmc_fit(
    q_init = rep(0, 6),  # 2 + 2 fixed + 2 log_phi
    y_num = as.integer(y_num),
    y_denom = as.integer(y_denom),
    y_num_cont = rep(0.0, n),
    y_denom_cont = rep(0.0, n),
    X_num = X,
    X_denom = X,
    family = list(model_type = "negbin_negbin"),
    re_params = make_re_params(n),
    spatial_params = make_spatial_params(n),
    gp_params = list(gp_type = "none"),
    temporal_params = make_temporal_params(n),
    prior_params = make_prior_params(),
    zi_params = make_zi_params(n),
    latent_params = make_latent_params(),
    st_params = make_st_params(),
    tvc_params = make_tvc_params(),
    svc_params = make_svc_params(),
    n_iter = 200L,
    n_warmup = 100L,
    L = 10L,
    n_chains = 1L,
    seed = 456L,
    n_threads = 1L,
    verbose = FALSE
  )

  expect_true(is.matrix(result$samples))
  expect_equal(ncol(result$samples), 6)  # 2 + 2 + 2
})


test_that("HMC backend runs with random effects", {
  skip_on_cran()

  set.seed(789)
  n_groups <- 10
  n_per_group <- 30  # More per group
  n <- n_groups * n_per_group

  group <- rep(1:n_groups, each = n_per_group)
  re_true <- rnorm(n_groups, 0, 0.5)

  x <- rnorm(n)
  p <- plogis(-0.5 + 0.8 * x + re_true[group])
  trials <- rpois(n, 30) + 15  # More trials
  successes <- rbinom(n, trials, p)

  X <- cbind(1, x)

  # Parameters: 2 beta_num + 1 beta_denom + 1 log_sigma_re + 10 RE = 14
  result <- tulpaRatio:::cpp_hmc_fit(
    q_init = rep(0, 14),
    y_num = as.integer(successes),
    y_denom = as.integer(trials),
    y_num_cont = rep(0.0, n),
    y_denom_cont = rep(0.0, n),
    X_num = X,
    X_denom = matrix(1, n, 1),
    family = list(model_type = "binomial"),
    re_params = make_re_params(n, group = group, n_groups = n_groups),
    spatial_params = make_spatial_params(n),
    gp_params = list(gp_type = "none"),
    temporal_params = make_temporal_params(n),
    prior_params = make_prior_params(),
    zi_params = make_zi_params(n),
    latent_params = make_latent_params(),
    st_params = make_st_params(),
    tvc_params = make_tvc_params(),
    svc_params = make_svc_params(),
    n_iter = 200L,
    n_warmup = 100L,
    L = 10L,
    n_chains = 1L,
    seed = 789L,
    n_threads = 1L,
    verbose = FALSE
  )

  expect_equal(ncol(result$samples), 14)

  # Check sigma_re is estimated (with wider bounds)
  log_sigma_idx <- 4  # After 2 beta_num + 1 beta_denom
  sigma_re_est <- mean(exp(result$samples[, log_sigma_idx]))
  expect_true(sigma_re_est > 0)
  expect_true(sigma_re_est < 5)  # Wider tolerance
})


test_that("HMC backend runs with ICAR spatial effects", {
  skip_on_cran()

  set.seed(111)

  # Create a simple 4x4 grid
  n_spatial <- 16
  n_per_spatial <- 10
  n <- n_spatial * n_per_spatial

  # Simple row-wise adjacency (each cell connected to horizontal neighbors)
  adj_row_ptr <- integer(n_spatial + 1)
  adj_col_idx <- integer(0)
  n_neighbors <- integer(n_spatial)

  for (i in 1:n_spatial) {
    neighbors <- integer(0)
    row <- (i - 1) %/% 4 + 1
    col <- (i - 1) %% 4 + 1

    # Left neighbor (0-based index for C++)
    if (col > 1) neighbors <- c(neighbors, (i - 1) - 1L)
    # Right neighbor (0-based)
    if (col < 4) neighbors <- c(neighbors, (i + 1) - 1L)
    # Top neighbor (0-based)
    if (row > 1) neighbors <- c(neighbors, (i - 4) - 1L)
    # Bottom neighbor (0-based)
    if (row < 4) neighbors <- c(neighbors, (i + 4) - 1L)

    n_neighbors[i] <- length(neighbors)
    adj_col_idx <- c(adj_col_idx, neighbors)
    adj_row_ptr[i + 1] <- adj_row_ptr[i] + length(neighbors)
  }

  spatial_group <- rep(1:n_spatial, each = n_per_spatial)

  # Simulate data
  x <- rnorm(n)
  spatial_effect <- rnorm(n_spatial, 0, 0.3)
  p <- plogis(-0.5 + 0.5 * x + spatial_effect[spatial_group])
  trials <- rpois(n, 15) + 5
  successes <- rbinom(n, trials, p)

  X <- cbind(1, x)

  # Build spatial params with adjacency
  spatial_params <- list(
    type = "icar",
    group = as.integer(spatial_group),
    n_units = as.integer(n_spatial),
    adj_row_ptr = as.integer(adj_row_ptr),
    adj_col_idx = as.integer(adj_col_idx),
    n_neighbors = as.integer(n_neighbors),
    bym2_scale = 1.0
  )

  # Parameters: 2 beta_num + 1 beta_denom + 1 log_tau + 16 spatial = 20
  result <- tulpaRatio:::cpp_hmc_fit(
    q_init = rep(0, 20),
    y_num = as.integer(successes),
    y_denom = as.integer(trials),
    y_num_cont = rep(0.0, n),
    y_denom_cont = rep(0.0, n),
    X_num = X,
    X_denom = matrix(1, n, 1),
    family = list(model_type = "binomial"),
    re_params = make_re_params(n),
    spatial_params = spatial_params,
    gp_params = list(gp_type = "none"),
    temporal_params = make_temporal_params(n),
    prior_params = make_prior_params(),
    zi_params = make_zi_params(n),
    latent_params = make_latent_params(),
    st_params = make_st_params(),
    tvc_params = make_tvc_params(),
    svc_params = make_svc_params(),
    n_iter = 200L,
    n_warmup = 100L,
    L = 10L,
    n_chains = 1L,
    seed = 111L,
    n_threads = 1L,
    verbose = FALSE
  )

  expect_equal(ncol(result$samples), 20)
  expect_equal(nrow(result$samples), 100)

  # Check tau_spatial is positive
  log_tau_idx <- 4  # After 2 beta_num + 1 beta_denom
  tau_est <- mean(exp(result$samples[, log_tau_idx]))
  expect_true(tau_est > 0)
})


test_that("HMC backend runs multiple chains in parallel", {
  skip_on_cran()

  set.seed(222)
  n <- 50
  x <- rnorm(n)
  p <- plogis(-0.3 + 0.6 * x)
  trials <- rpois(n, 15) + 5
  successes <- rbinom(n, trials, p)

  X <- cbind(1, x)

  # Parameters: 2 beta_num + 1 beta_denom = 3
  result <- tulpaRatio:::cpp_hmc_fit(
    q_init = rep(0, 3),
    y_num = as.integer(successes),
    y_denom = as.integer(trials),
    y_num_cont = rep(0.0, n),
    y_denom_cont = rep(0.0, n),
    X_num = X,
    X_denom = matrix(1, n, 1),
    family = list(model_type = "binomial"),
    re_params = make_re_params(n),
    spatial_params = make_spatial_params(n),
    gp_params = list(gp_type = "none"),
    temporal_params = make_temporal_params(n),
    prior_params = make_prior_params(),
    zi_params = make_zi_params(n),
    latent_params = make_latent_params(),
    st_params = make_st_params(),
    tvc_params = make_tvc_params(),
    svc_params = make_svc_params(),
    n_iter = 200L,
    n_warmup = 100L,
    L = 10L,
    n_chains = 2L,
    seed = 222L,
    n_threads = 1L,  # 1 thread within, parallel across chains
    verbose = FALSE
  )

  # Multi-chain returns lists
  expect_true(is.list(result$samples))
  expect_equal(length(result$samples), 2)

  # Each chain has samples
  expect_equal(nrow(result$samples[[1]]), 100)
  expect_equal(nrow(result$samples[[2]]), 100)
})


test_that("cpp_get_max_threads returns positive integer", {
  n_threads <- tulpaRatio:::cpp_get_max_threads()
  expect_true(is.numeric(n_threads))
  expect_true(n_threads >= 1)
})

test_that("a non-finite energy counts as a divergence, not as acceptance", {
  # std::min(1, exp(-NaN)) is 1, so a leaf whose step left the support used to
  # report perfect acceptance: dual averaging grew the step size to its cap
  # and the chain froze (364 of 500 draws divergent on this fixture at
  # iter = 1000, against 0 at iter = 400).
  skip_on_cran()
  G <- 12
  set.seed(5)
  n <- 48
  site <- rep(letters[1:G], length.out = n)
  u <- rnorm(G, sd = 0.5)[match(site, letters[1:G])]
  depth <- rnorm(n)
  df <- data.frame(count = rpois(n, exp(3 + 0.3 * depth + u)),
                   effort = rgamma(n, shape = 10, rate = exp(-u)),
                   depth = depth, site = site)
  fit <- tratio(count | effort ~ depth + (1 | site), data = df,
                family = ratiod_poisson_gamma(), mode = "hmc",
                control = list(iter = 1000, chains = 1, seed = 1))
  expect_equal(sum(fit$diagnostics$divergent), 0)
  expect_lt(fit$epsilon, 1)
})
