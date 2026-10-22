# test-gradient-coverage.R
# Tests for gradient method coverage across all family × spatial × temporal combinations
# Validates rows 42-66 in gradient_methods.md

# =============================================================================
# Helper functions
# =============================================================================

make_re_params <- function(n) {
  list(
    group = rep(0L, n),
    n_groups = 0L,
    n_terms = 0L,
    group_matrix = matrix(0L, nrow = n, ncol = 1),
    n_groups_vec = 0L,
    has_slopes = FALSE,
    has_correlated_slopes = FALSE,
    n_coefs_vec = integer(0),
    correlated_vec = logical(0),
    n_chol_vec = integer(0),
    slope_matrices = list(),
    parameterization = 0L  # 0 = centered (default)
  )
}

make_re_params_with_groups <- function(n, group_vec) {
  n_groups <- length(unique(group_vec[group_vec > 0]))
  list(
    group = as.integer(group_vec),
    n_groups = as.integer(n_groups),
    n_terms = 1L,
    group_matrix = matrix(as.integer(group_vec), nrow = n, ncol = 1),
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

make_spatial_params <- function(n) {
  list(
    type = "none",
    group = rep(0L, n),
    n_units = 0L,
    adj_row_ptr = 0L,
    adj_col_idx = integer(0),
    n_neighbors = integer(0),
    bym2_scale = 1.0
  )
}

make_temporal_params <- function(n, time_idx, type = "rw1", n_times, cyclic = FALSE,
                                  group_idx = NULL, n_groups = 1L) {
  if (is.null(group_idx)) group_idx <- rep(1L, n)
  list(
    type = type,
    time_idx = as.integer(time_idx),
    group_idx = as.integer(group_idx),
    n_times = as.integer(n_times),
    n_groups = as.integer(n_groups),
    n_params = as.integer(n_times * n_groups),
    cyclic = cyclic,
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

# =============================================================================
# Phase 2.1: RW2 temporal for negbin_negbin and binomial (rows 42-43)
# =============================================================================

test_that("negbin_negbin with RW2 temporal runs correctly (row 42)", {
  skip_on_cran()

  set.seed(4201)
  n_times <- 8
  n_per_time <- 15
  n <- n_times * n_per_time

  # Simulate RW2 temporal effect (smooth trend)
  temporal_effect <- numeric(n_times)
  temporal_effect[1:2] <- rnorm(2, 0, 0.1)
  for (t in 3:n_times) {
    temporal_effect[t] <- 2 * temporal_effect[t-1] - temporal_effect[t-2] + rnorm(1, 0, 0.1)
  }
  temporal_effect <- temporal_effect - mean(temporal_effect)

  time_idx <- rep(1:n_times, each = n_per_time)
  x <- rnorm(n)

  # Generate negbin_negbin data
  mu_num <- exp(2 + 0.3 * x + temporal_effect[time_idx])
  mu_denom <- exp(3 + 0.2 * x + 0.5 * temporal_effect[time_idx])

  y_num <- rnbinom(n, size = 5, mu = mu_num)
  y_denom <- rnbinom(n, size = 10, mu = mu_denom)

  X <- cbind(1, x)

  # Parameters: 2 beta_num + 2 beta_denom + 2 log_phi + 1 log_tau + 8 temporal = 15
  n_params <- 2 + 2 + 2 + 1 + n_times

  result <- tulpaRatio:::cpp_hmc_fit(
    q_init = rep(0, n_params),
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
    temporal_params = make_temporal_params(n, time_idx, type = "rw2", n_times = n_times),
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
    seed = 4201L,
    n_threads = 1L,
    verbose = FALSE
  )

  expect_equal(ncol(result$samples), n_params)
  # n_iter=200 total, n_warmup=100 → 100 post-warmup samples
  expect_equal(nrow(result$samples), 100)

  # Check temporal effects are finite
  temporal_cols <- (n_params - n_times + 1):n_params
  temporal_means <- colMeans(result$samples[, temporal_cols])
  expect_true(all(is.finite(temporal_means)))

  # Acceptance rate should be reasonable
  expect_true(mean(result$accept_prob) > 0.1)
})


test_that("binomial with RW2 temporal runs correctly (row 43)", {
  skip_on_cran()

  set.seed(4301)
  n_times <- 10
  n_per_time <- 12
  n <- n_times * n_per_time

  # Simulate RW2 temporal effect
  temporal_effect <- numeric(n_times)
  temporal_effect[1:2] <- rnorm(2, 0, 0.15)
  for (t in 3:n_times) {
    temporal_effect[t] <- 2 * temporal_effect[t-1] - temporal_effect[t-2] + rnorm(1, 0, 0.1)
  }
  temporal_effect <- temporal_effect - mean(temporal_effect)

  time_idx <- rep(1:n_times, each = n_per_time)
  x <- rnorm(n)

  # Generate binomial data
  p <- plogis(-0.3 + 0.4 * x + temporal_effect[time_idx])
  trials <- rpois(n, 20) + 10
  successes <- rbinom(n, trials, p)

  X <- cbind(1, x)

  # Parameters: 2 beta_num + 1 beta_denom + 1 log_tau + 10 temporal = 14
  n_params <- 2 + 1 + 1 + n_times

  result <- tulpaRatio:::cpp_hmc_fit(
    q_init = rep(0, n_params),
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
    temporal_params = make_temporal_params(n, time_idx, type = "rw2", n_times = n_times),
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
    seed = 4301L,
    n_threads = 1L,
    verbose = FALSE
  )

  expect_equal(ncol(result$samples), n_params)
  # n_iter=200 total, n_warmup=100 → 100 post-warmup samples
  expect_equal(nrow(result$samples), 100)

  # Check temporal effects are finite
  temporal_cols <- (n_params - n_times + 1):n_params
  temporal_means <- colMeans(result$samples[, temporal_cols])
  expect_true(all(is.finite(temporal_means)))

  # Acceptance rate should be reasonable
  expect_true(mean(result$accept_prob) > 0.1)
})

# =============================================================================
# Phase 2.2: MSGP for negbin_negbin and binomial (rows 44-45)
# =============================================================================

generate_msgp_data_family <- function(n, seed, family = "poisson_gamma") {
  set.seed(seed)

  # Generate coordinates on unit square
  coords <- data.frame(x = runif(n), y = runif(n))

  # Generate true spatial effects
  dist_mat <- as.matrix(dist(coords))

  # Local effect (short range)
  sigma_local <- 0.4
  phi_local <- 0.1
  K_local <- sigma_local^2 * exp(-dist_mat / phi_local)
  diag(K_local) <- diag(K_local) + 1e-6
  L_local <- chol(K_local)
  w_local <- as.numeric(t(L_local) %*% rnorm(n))

  # Regional effect (long range)
  sigma_regional <- 0.3
  phi_regional <- 0.4
  K_regional <- sigma_regional^2 * exp(-dist_mat / phi_regional)
  diag(K_regional) <- diag(K_regional) + 1e-6
  L_regional <- chol(K_regional)
  w_regional <- as.numeric(t(L_regional) %*% rnorm(n))

  w_total <- w_local + w_regional

  if (family == "negbin_negbin") {
    mu_num <- exp(2 + w_total)
    mu_denom <- exp(2.5 + 0.5 * w_total)
    y_num <- rnbinom(n, size = 5, mu = mu_num)
    y_denom <- rnbinom(n, size = 8, mu = mu_denom)
    y_denom_cont <- rep(0.0, n)
  } else if (family == "binomial") {
    p <- plogis(w_total)
    trials <- rpois(n, 30) + 10
    y_num <- rbinom(n, trials, p)
    y_denom <- trials
    y_denom_cont <- rep(0.0, n)
  } else {  # poisson_gamma
    y_num <- rpois(n, exp(2 + w_total))
    y_denom <- rep(0L, n)
    y_denom_cont <- rgamma(n, shape = 5, rate = 5 / exp(2 + 0.5 * w_total))
  }

  list(
    x = coords$x,
    y = coords$y,
    y_num = y_num,
    y_denom = y_denom,
    y_denom_cont = y_denom_cont
  )
}

test_that("negbin_negbin with MSGP runs correctly (row 44)", {
  skip_on_cran()
  skip_if_not(Sys.getenv("RUN_SLOW_TESTS") == "true",
              "Skipping slow MSGP test (set RUN_SLOW_TESTS=true to run)")

  df <- generate_msgp_data_family(n = 40, seed = 4401, family = "negbin_negbin")

  ms <- spatial_multiscale(
    ~ x + y,
    m = 6
  )

  fit <- suppressWarnings(
    tratio(
      y_num | y_denom ~ 1,
      data = as.data.frame(df),
      spatial = ms,
      family = ratiod_negbin_negbin(),
      mode = "hmc",
      control = list(chains = 1, iter = 150, warmup = 75, verbose = FALSE)
    )
  )

  expect_s3_class(fit, "ratiod_fit")
  expect_false(is.null(fit$diagnostics))
})


test_that("binomial with MSGP runs correctly (row 45)", {
  skip_on_cran()
  skip_if_not(Sys.getenv("RUN_SLOW_TESTS") == "true",
              "Skipping slow MSGP test (set RUN_SLOW_TESTS=true to run)")

  df <- generate_msgp_data_family(n = 40, seed = 4501, family = "binomial")

  ms <- spatial_multiscale(
    ~ x + y,
    m = 6
  )

  fit <- suppressWarnings(
    tratio(
      y_num | y_denom ~ 1,
      data = as.data.frame(df),
      spatial = ms,
      family = ratiod_binomial(),
      mode = "hmc",
      control = list(chains = 1, iter = 150, warmup = 75, verbose = FALSE)
    )
  )

  expect_s3_class(fit, "ratiod_fit")
})

# =============================================================================
# Phase 2.3: GP + temporal for all families (rows 46-51)
# =============================================================================

generate_gp_temporal_data <- function(n_spatial, n_times, seed, family = "poisson_gamma") {
  set.seed(seed)

  n <- n_spatial * n_times

  # Spatial coordinates
  coords <- data.frame(
    x = rep(runif(n_spatial), n_times),
    y = rep(runif(n_spatial), n_times)
  )

  # GP spatial effect
  dist_mat <- as.matrix(dist(coords[1:n_spatial, ]))
  sigma_gp <- 0.5
  phi_gp <- 0.2
  K <- sigma_gp^2 * exp(-dist_mat / phi_gp)
  diag(K) <- diag(K) + 1e-6
  L <- chol(K)
  w_spatial <- as.numeric(t(L) %*% rnorm(n_spatial))
  w_spatial_full <- rep(w_spatial, n_times)

  # Temporal effect (RW1)
  temporal_effect <- cumsum(rnorm(n_times, 0, 0.2))
  temporal_effect <- temporal_effect - mean(temporal_effect)
  temporal_full <- rep(temporal_effect, each = n_spatial)

  time_idx <- rep(1:n_times, each = n_spatial)

  if (family == "negbin_negbin") {
    mu_num <- exp(2 + w_spatial_full + temporal_full)
    mu_denom <- exp(2.5 + 0.5 * w_spatial_full + 0.5 * temporal_full)
    y_num <- rnbinom(n, size = 5, mu = mu_num)
    y_denom <- rnbinom(n, size = 8, mu = mu_denom)
    y_denom_cont <- rep(0.0, n)
  } else if (family == "binomial") {
    p <- plogis(w_spatial_full + temporal_full)
    trials <- rpois(n, 25) + 10
    y_num <- rbinom(n, trials, p)
    y_denom <- trials
    y_denom_cont <- rep(0.0, n)
  } else {  # poisson_gamma
    y_num <- rpois(n, exp(2 + w_spatial_full + temporal_full))
    y_denom <- rep(0L, n)
    y_denom_cont <- rgamma(n, shape = 5, rate = 5 / exp(2 + 0.5 * w_spatial_full))
  }

  list(
    x = coords$x,
    y = coords$y,
    y_num = y_num,
    y_denom = y_denom,
    y_denom_cont = y_denom_cont,
    time = time_idx,
    n_times = n_times
  )
}

test_that("negbin_negbin with GP + RW1 runs correctly (row 46)", {
  skip_on_cran()
  skip_if_not(Sys.getenv("RUN_SLOW_TESTS") == "true",
              "Skipping slow GP+temporal test (set RUN_SLOW_TESTS=true to run)")

  df <- generate_gp_temporal_data(n_spatial = 25, n_times = 5, seed = 4601, family = "negbin_negbin")

  fit <- suppressWarnings(
    tratio(
      y_num | y_denom ~ 1,
      data = as.data.frame(df),
      spatial = spatial_gp(~ x + y, nn = 8),
      temporal = temporal_rw1("time"),
      family = ratiod_negbin_negbin(),
      mode = "hmc",
      control = list(chains = 1, iter = 150, warmup = 75, verbose = FALSE)
    )
  )

  expect_s3_class(fit, "ratiod_fit")
})


test_that("binomial with GP + RW1 runs correctly (row 47)", {
  skip_on_cran()
  skip_if_not(Sys.getenv("RUN_SLOW_TESTS") == "true",
              "Skipping slow GP+temporal test (set RUN_SLOW_TESTS=true to run)")

  df <- generate_gp_temporal_data(n_spatial = 25, n_times = 5, seed = 4701, family = "binomial")

  fit <- suppressWarnings(
    tratio(
      y_num | y_denom ~ 1,
      data = as.data.frame(df),
      spatial = spatial_gp(~ x + y, nn = 8),
      temporal = temporal_rw1("time"),
      family = ratiod_binomial(),
      mode = "hmc",
      control = list(chains = 1, iter = 150, warmup = 75, verbose = FALSE)
    )
  )

  expect_s3_class(fit, "ratiod_fit")
})


test_that("poisson_gamma with GP + RW2 runs correctly (row 48)", {
  skip_on_cran()
  skip_if_not(Sys.getenv("RUN_SLOW_TESTS") == "true",
              "Skipping slow GP+temporal test (set RUN_SLOW_TESTS=true to run)")

  df <- generate_gp_temporal_data(n_spatial = 25, n_times = 6, seed = 4801, family = "poisson_gamma")

  fit <- suppressWarnings(
    tratio(
      y_num | y_denom_cont ~ 1,
      data = as.data.frame(df),
      spatial = spatial_gp(~ x + y, nn = 8),
      temporal = temporal_rw2("time"),
      family = ratiod_poisson_gamma(),
      mode = "hmc",
      control = list(chains = 1, iter = 150, warmup = 75, verbose = FALSE)
    )
  )

  expect_s3_class(fit, "ratiod_fit")
})


test_that("poisson_gamma with GP + AR1 runs correctly (row 49)", {
  skip_on_cran()
  skip_if_not(Sys.getenv("RUN_SLOW_TESTS") == "true",
              "Skipping slow GP+temporal test (set RUN_SLOW_TESTS=true to run)")

  df <- generate_gp_temporal_data(n_spatial = 25, n_times = 6, seed = 4901, family = "poisson_gamma")

  fit <- suppressWarnings(
    tratio(
      y_num | y_denom_cont ~ 1,
      data = as.data.frame(df),
      spatial = spatial_gp(~ x + y, nn = 8),
      temporal = temporal_ar1("time"),
      family = ratiod_poisson_gamma(),
      mode = "hmc",
      control = list(chains = 1, iter = 150, warmup = 75, verbose = FALSE)
    )
  )

  expect_s3_class(fit, "ratiod_fit")
})


test_that("negbin_negbin with GP + AR1 runs correctly (row 50)", {
  skip_on_cran()
  skip_if_not(Sys.getenv("RUN_SLOW_TESTS") == "true",
              "Skipping slow GP+temporal test (set RUN_SLOW_TESTS=true to run)")

  df <- generate_gp_temporal_data(n_spatial = 25, n_times = 6, seed = 5001, family = "negbin_negbin")

  fit <- suppressWarnings(
    tratio(
      y_num | y_denom ~ 1,
      data = as.data.frame(df),
      spatial = spatial_gp(~ x + y, nn = 8),
      temporal = temporal_ar1("time"),
      family = ratiod_negbin_negbin(),
      mode = "hmc",
      control = list(chains = 1, iter = 150, warmup = 75, verbose = FALSE)
    )
  )

  expect_s3_class(fit, "ratiod_fit")
})


test_that("binomial with GP + AR1 runs correctly (row 51)", {
  skip_on_cran()
  skip_if_not(Sys.getenv("RUN_SLOW_TESTS") == "true",
              "Skipping slow GP+temporal test (set RUN_SLOW_TESTS=true to run)")

  df <- generate_gp_temporal_data(n_spatial = 25, n_times = 6, seed = 5101, family = "binomial")

  fit <- suppressWarnings(
    tratio(
      y_num | y_denom ~ 1,
      data = as.data.frame(df),
      spatial = spatial_gp(~ x + y, nn = 8),
      temporal = temporal_ar1("time"),
      family = ratiod_binomial(),
      mode = "hmc",
      control = list(chains = 1, iter = 150, warmup = 75, verbose = FALSE)
    )
  )

  expect_s3_class(fit, "ratiod_fit")
})

# =============================================================================
# Quick smoke tests (run on CRAN)
# =============================================================================

test_that("RW2 temporal basic functionality works", {
  # Very minimal test that RW2 specification works
  temp <- temporal_rw2("time_var")

  expect_s3_class(temp, "ratiod_temporal")
  expect_equal(temp$type, "rw2")
})

# =============================================================================
# Phase 3.1: Spatial + ZI combinations (rows 52-56)
# =============================================================================

generate_spatial_zi_data <- function(n_sites, seed, spatial_type = "icar", family = "poisson_gamma") {
  set.seed(seed)

  # Generate adjacency for ICAR/BYM2 (grid layout)
  side <- ceiling(sqrt(n_sites))
  coords <- expand.grid(x = 1:side, y = 1:side)[1:n_sites, ]

  # Build adjacency list
  adj_list <- list()
  for (i in 1:n_sites) {
    neighbors <- c()
    for (j in 1:n_sites) {
      if (i != j) {
        dist_ij <- abs(coords$x[i] - coords$x[j]) + abs(coords$y[i] - coords$y[j])
        if (dist_ij == 1) neighbors <- c(neighbors, j)
      }
    }
    adj_list[[i]] <- neighbors
  }

  # Spatial effect (simplified)
  w_spatial <- rnorm(n_sites, 0, 0.5)

  # Zero-inflation probability
  pi_zi <- 0.2

  if (family == "poisson_gamma") {
    mu <- exp(2 + w_spatial)
    y_num <- rpois(n_sites, mu)
    # Apply ZI
    zi_mask <- rbinom(n_sites, 1, pi_zi)
    y_num[zi_mask == 1] <- 0

    y_denom <- rep(0L, n_sites)
    y_denom_cont <- rgamma(n_sites, shape = 5, rate = 5 / exp(2))
  } else if (family == "negbin_negbin") {
    mu_num <- exp(2 + w_spatial)
    mu_denom <- exp(2.5 + 0.5 * w_spatial)
    y_num <- rnbinom(n_sites, size = 5, mu = mu_num)
    y_denom <- rnbinom(n_sites, size = 8, mu = mu_denom)

    # Apply ZI to numerator
    zi_mask <- rbinom(n_sites, 1, pi_zi)
    y_num[zi_mask == 1] <- 0

    y_denom_cont <- rep(0.0, n_sites)
  } else {  # binomial
    p <- plogis(w_spatial)
    trials <- rpois(n_sites, 30) + 10
    y_num <- rbinom(n_sites, trials, p)

    # Apply ZI
    zi_mask <- rbinom(n_sites, 1, pi_zi)
    y_num[zi_mask == 1] <- 0

    y_denom <- trials
    y_denom_cont <- rep(0.0, n_sites)
  }

  list(
    x = coords$x / side,
    y = coords$y / side,
    y_num = y_num,
    y_denom = y_denom,
    y_denom_cont = y_denom_cont,
    site = 1:n_sites,
    adj_list = adj_list
  )
}

test_that("poisson_gamma with BYM2 + ZI runs correctly (row 52)", {
  skip_on_cran()
  skip_if_not(Sys.getenv("RUN_SLOW_TESTS") == "true",
              "Skipping slow spatial+ZI test (set RUN_SLOW_TESTS=true to run)")

  df <- generate_spatial_zi_data(n_sites = 25, seed = 5201, spatial_type = "bym2", family = "poisson_gamma")

  # Create adjacency matrix
  adj_mat <- matrix(0, 25, 25)
  for (i in 1:25) {
    for (j in df$adj_list[[i]]) {
      adj_mat[i, j] <- 1
    }
  }

  fit <- suppressWarnings(
    tratio(
      y_num | y_denom_cont ~ 1 + (1 | site),
      data = as.data.frame(df),
      spatial = spatial_bym2(adj_mat, level = "group", group_var = "site"),
      zi = ~ 1,
      family = ratiod_poisson_gamma(),
      mode = "hmc",
      control = list(chains = 1, iter = 150, warmup = 75, verbose = FALSE)
    )
  )

  expect_s3_class(fit, "ratiod_fit")
})


test_that("poisson_gamma with GP + ZI runs correctly (row 53)", {
  skip_on_cran()
  skip_if_not(Sys.getenv("RUN_SLOW_TESTS") == "true",
              "Skipping slow spatial+ZI test (set RUN_SLOW_TESTS=true to run)")

  df <- generate_spatial_zi_data(n_sites = 30, seed = 5301, family = "poisson_gamma")

  fit <- suppressWarnings(
    tratio(
      y_num | y_denom_cont ~ 1,
      data = as.data.frame(df),
      spatial = spatial_gp(~ x + y, nn = 8),
      zi = ~ 1,
      family = ratiod_poisson_gamma(),
      mode = "hmc",
      control = list(chains = 1, iter = 150, warmup = 75, verbose = FALSE)
    )
  )

  expect_s3_class(fit, "ratiod_fit")
})


test_that("negbin_negbin with ICAR + ZI runs correctly (row 54)", {
  skip_on_cran()
  skip_if_not(Sys.getenv("RUN_SLOW_TESTS") == "true",
              "Skipping slow spatial+ZI test (set RUN_SLOW_TESTS=true to run)")

  df <- generate_spatial_zi_data(n_sites = 25, seed = 5401, family = "negbin_negbin")

  # Create adjacency matrix
  adj_mat <- matrix(0, 25, 25)
  for (i in 1:25) {
    for (j in df$adj_list[[i]]) {
      adj_mat[i, j] <- 1
    }
  }

  fit <- suppressWarnings(
    tratio(
      y_num | y_denom ~ 1 + (1 | site),
      data = as.data.frame(df),
      spatial = spatial_icar(adj_mat, level = "group", group_var = "site"),
      zi = ~ 1,
      family = ratiod_negbin_negbin(),
      mode = "hmc",
      control = list(chains = 1, iter = 150, warmup = 75, verbose = FALSE)
    )
  )

  expect_s3_class(fit, "ratiod_fit")
})


# =============================================================================
# Phase 3.2: Crossed RE + spatial (rows 57-59)
# =============================================================================

test_that("poisson_gamma with crossed RE + ICAR runs correctly (row 57)", {
  skip_on_cran()
  skip_if_not(Sys.getenv("RUN_SLOW_TESTS") == "true",
              "Skipping slow crossed+spatial test (set RUN_SLOW_TESTS=true to run)")

  set.seed(5701)
  n_sites <- 16
  n_species <- 5
  n <- n_sites * n_species

  # Generate data with crossed RE (site × species)
  site <- rep(1:n_sites, each = n_species)
  species <- rep(1:n_species, n_sites)

  # Create grid adjacency for sites
  side <- 4
  adj_mat <- matrix(0, n_sites, n_sites)
  for (i in 1:n_sites) {
    row_i <- (i - 1) %/% side + 1
    col_i <- (i - 1) %% side + 1
    for (j in 1:n_sites) {
      row_j <- (j - 1) %/% side + 1
      col_j <- (j - 1) %% side + 1
      if (i != j && abs(row_i - row_j) + abs(col_i - col_j) == 1) {
        adj_mat[i, j] <- 1
      }
    }
  }

  # Site spatial effect
  w_site <- rnorm(n_sites, 0, 0.5)
  # Species RE
  re_species <- rnorm(n_species, 0, 0.3)

  mu <- exp(2 + w_site[site] + re_species[species])
  y_num <- rpois(n, mu)
  y_denom_cont <- rgamma(n, shape = 5, rate = 5 / exp(2))

  df <- data.frame(
    y_num = y_num,
    y_denom_cont = y_denom_cont,
    site = factor(site),
    species = factor(species)
  )

  fit <- suppressWarnings(
    tratio(
      y_num | y_denom_cont ~ 1 + (1 | site) + (1 | species),
      data = df,
      spatial = spatial_icar(adj_mat, level = "group", group_var = "site"),
      family = ratiod_poisson_gamma(),
      mode = "hmc",
      control = list(chains = 1, iter = 150, warmup = 75, verbose = FALSE)
    )
  )

  expect_s3_class(fit, "ratiod_fit")
})


# =============================================================================
# Phase 3.3: Random slopes + temporal (rows 60-63)
# =============================================================================

test_that("poisson_gamma with random slopes + RW1 runs correctly (row 60)", {
  skip_on_cran()
  skip_if_not(Sys.getenv("RUN_SLOW_TESTS") == "true",
              "Skipping slow slopes+temporal test (set RUN_SLOW_TESTS=true to run)")

  set.seed(6001)
  n_sites <- 10
  n_times <- 6
  n_per <- 5
  n <- n_sites * n_times * n_per

  # Random slopes for x by site
  site <- rep(1:n_sites, each = n_times * n_per)
  time <- rep(rep(1:n_times, each = n_per), n_sites)

  re_intercept <- rnorm(n_sites, 0, 0.3)
  re_slope <- rnorm(n_sites, 0, 0.2)

  # Temporal effect (RW1)
  temporal_effect <- cumsum(rnorm(n_times, 0, 0.15))
  temporal_effect <- temporal_effect - mean(temporal_effect)

  x <- rnorm(n)
  mu <- exp(2 + re_intercept[site] + (0.3 + re_slope[site]) * x + temporal_effect[time])
  y_num <- rpois(n, mu)
  y_denom_cont <- rgamma(n, shape = 5, rate = 5 / exp(2))

  df <- data.frame(
    y_num = y_num,
    y_denom_cont = y_denom_cont,
    x = x,
    site = factor(site),
    time = time
  )

  fit <- suppressWarnings(
    tratio(
      y_num | y_denom_cont ~ x + (x | site),
      data = df,
      temporal = temporal_rw1("time"),
      family = ratiod_poisson_gamma(),
      mode = "hmc",
      control = list(chains = 1, iter = 150, warmup = 75, verbose = FALSE)
    )
  )

  expect_s3_class(fit, "ratiod_fit")
})


# =============================================================================
# Phase 3.4: Three-way: spatial + temporal + ZI (rows 64-66)
# =============================================================================

test_that("poisson_gamma with ICAR + RW1 + ZI runs correctly (row 64)", {
  skip_on_cran()
  skip_if_not(Sys.getenv("RUN_SLOW_TESTS") == "true",
              "Skipping slow three-way test (set RUN_SLOW_TESTS=true to run)")

  set.seed(6401)
  n_sites <- 16
  n_times <- 5
  n <- n_sites * n_times

  # Grid adjacency
  side <- 4
  adj_mat <- matrix(0, n_sites, n_sites)
  for (i in 1:n_sites) {
    row_i <- (i - 1) %/% side + 1
    col_i <- (i - 1) %% side + 1
    for (j in 1:n_sites) {
      row_j <- (j - 1) %/% side + 1
      col_j <- (j - 1) %% side + 1
      if (i != j && abs(row_i - row_j) + abs(col_i - col_j) == 1) {
        adj_mat[i, j] <- 1
      }
    }
  }

  # Data structure
  site <- rep(1:n_sites, n_times)
  time <- rep(1:n_times, each = n_sites)

  # Spatial effect
  w_site <- rnorm(n_sites, 0, 0.4)

  # Temporal effect (RW1)
  temporal_effect <- cumsum(rnorm(n_times, 0, 0.2))
  temporal_effect <- temporal_effect - mean(temporal_effect)

  mu <- exp(2 + w_site[site] + temporal_effect[time])
  y_num <- rpois(n, mu)

  # Apply ZI (20% structural zeros)
  pi_zi <- 0.2
  zi_mask <- rbinom(n, 1, pi_zi)
  y_num[zi_mask == 1] <- 0

  y_denom_cont <- rgamma(n, shape = 5, rate = 5 / exp(2))

  df <- data.frame(
    y_num = y_num,
    y_denom_cont = y_denom_cont,
    site = factor(site),
    time = time
  )

  fit <- suppressWarnings(
    tratio(
      y_num | y_denom_cont ~ 1 + (1 | site),
      data = df,
      spatial = spatial_icar(adj_mat, level = "group", group_var = "site"),
      temporal = temporal_rw1("time"),
      zi = ~ 1,
      family = ratiod_poisson_gamma(),
      mode = "hmc",
      control = list(chains = 1, iter = 150, warmup = 75, verbose = FALSE)
    )
  )

  expect_s3_class(fit, "ratiod_fit")
})


# =============================================================================
# Additional tests for rows 55, 56, 58, 59, 61, 62, 63, 65, 66
# =============================================================================

test_that("negbin_negbin with BYM2 + ZI runs correctly (row 55)", {
  skip_on_cran()
  skip_if_not(Sys.getenv("RUN_SLOW_TESTS") == "true",
              "Skipping slow spatial+ZI test (set RUN_SLOW_TESTS=true to run)")

  df <- generate_spatial_zi_data(n_sites = 25, seed = 5501, spatial_type = "bym2", family = "negbin_negbin")

  # Create adjacency matrix
  adj_mat <- matrix(0, 25, 25)
  for (i in 1:25) {
    for (j in df$adj_list[[i]]) {
      adj_mat[i, j] <- 1
    }
  }

  fit <- suppressWarnings(
    tratio(
      y_num | y_denom ~ 1 + (1 | site),
      data = as.data.frame(df),
      spatial = spatial_bym2(adj_mat, level = "group", group_var = "site"),
      zi = ~ 1,
      family = ratiod_negbin_negbin(),
      mode = "hmc",
      control = list(chains = 1, iter = 150, warmup = 75, verbose = FALSE)
    )
  )

  expect_s3_class(fit, "ratiod_fit")
})


test_that("binomial with ICAR + ZI runs correctly (row 56)", {
  skip_on_cran()
  skip_if_not(Sys.getenv("RUN_SLOW_TESTS") == "true",
              "Skipping slow spatial+ZI test (set RUN_SLOW_TESTS=true to run)")

  df <- generate_spatial_zi_data(n_sites = 25, seed = 5601, family = "binomial")

  # Create adjacency matrix
  adj_mat <- matrix(0, 25, 25)
  for (i in 1:25) {
    for (j in df$adj_list[[i]]) {
      adj_mat[i, j] <- 1
    }
  }

  fit <- suppressWarnings(
    tratio(
      y_num | y_denom ~ 1 + (1 | site),
      data = as.data.frame(df),
      spatial = spatial_icar(adj_mat, level = "group", group_var = "site"),
      zi = ~ 1,
      family = ratiod_binomial(),
      mode = "hmc",
      control = list(chains = 1, iter = 150, warmup = 75, verbose = FALSE)
    )
  )

  expect_s3_class(fit, "ratiod_fit")
})


test_that("poisson_gamma with crossed RE + BYM2 runs correctly (row 58)", {
  skip_on_cran()
  skip_if_not(Sys.getenv("RUN_SLOW_TESTS") == "true",
              "Skipping slow crossed+spatial test (set RUN_SLOW_TESTS=true to run)")

  set.seed(5801)
  n_sites <- 16
  n_species <- 5
  n <- n_sites * n_species

  # Generate data with crossed RE (site × species)
  site <- rep(1:n_sites, each = n_species)
  species <- rep(1:n_species, n_sites)

  # Create grid adjacency for sites
  side <- 4
  adj_mat <- matrix(0, n_sites, n_sites)
  for (i in 1:n_sites) {
    row_i <- (i - 1) %/% side + 1
    col_i <- (i - 1) %% side + 1
    for (j in 1:n_sites) {
      row_j <- (j - 1) %/% side + 1
      col_j <- (j - 1) %% side + 1
      if (i != j && abs(row_i - row_j) + abs(col_i - col_j) == 1) {
        adj_mat[i, j] <- 1
      }
    }
  }

  # Site spatial effect
  w_site <- rnorm(n_sites, 0, 0.5)
  # Species RE
  re_species <- rnorm(n_species, 0, 0.3)

  mu <- exp(2 + w_site[site] + re_species[species])
  y_num <- rpois(n, mu)
  y_denom_cont <- rgamma(n, shape = 5, rate = 5 / exp(2))

  df <- data.frame(
    y_num = y_num,
    y_denom_cont = y_denom_cont,
    site = factor(site),
    species = factor(species)
  )

  fit <- suppressWarnings(
    tratio(
      y_num | y_denom_cont ~ 1 + (1 | site) + (1 | species),
      data = df,
      spatial = spatial_bym2(adj_mat, level = "group", group_var = "site"),
      family = ratiod_poisson_gamma(),
      mode = "hmc",
      control = list(chains = 1, iter = 150, warmup = 75, verbose = FALSE)
    )
  )

  expect_s3_class(fit, "ratiod_fit")
})


test_that("negbin_negbin with crossed RE + ICAR runs correctly (row 59)", {
  skip_on_cran()
  skip_if_not(Sys.getenv("RUN_SLOW_TESTS") == "true",
              "Skipping slow crossed+spatial test (set RUN_SLOW_TESTS=true to run)")

  set.seed(5901)
  n_sites <- 16
  n_species <- 5
  n <- n_sites * n_species

  # Generate data with crossed RE (site × species)
  site <- rep(1:n_sites, each = n_species)
  species <- rep(1:n_species, n_sites)

  # Create grid adjacency for sites
  side <- 4
  adj_mat <- matrix(0, n_sites, n_sites)
  for (i in 1:n_sites) {
    row_i <- (i - 1) %/% side + 1
    col_i <- (i - 1) %% side + 1
    for (j in 1:n_sites) {
      row_j <- (j - 1) %/% side + 1
      col_j <- (j - 1) %% side + 1
      if (i != j && abs(row_i - row_j) + abs(col_i - col_j) == 1) {
        adj_mat[i, j] <- 1
      }
    }
  }

  # Site spatial effect
  w_site <- rnorm(n_sites, 0, 0.5)
  # Species RE
  re_species <- rnorm(n_species, 0, 0.3)

  mu_num <- exp(2 + w_site[site] + re_species[species])
  mu_denom <- exp(2.5 + 0.5 * w_site[site])
  y_num <- rnbinom(n, size = 5, mu = mu_num)
  y_denom <- rnbinom(n, size = 8, mu = mu_denom)

  df <- data.frame(
    y_num = y_num,
    y_denom = y_denom,
    site = factor(site),
    species = factor(species)
  )

  fit <- suppressWarnings(
    tratio(
      y_num | y_denom ~ 1 + (1 | site) + (1 | species),
      data = df,
      spatial = spatial_icar(adj_mat, level = "group", group_var = "site"),
      family = ratiod_negbin_negbin(),
      mode = "hmc",
      control = list(chains = 1, iter = 150, warmup = 75, verbose = FALSE)
    )
  )

  expect_s3_class(fit, "ratiod_fit")
})


test_that("poisson_gamma with random slopes + AR1 runs correctly (row 61)", {
  skip_on_cran()
  skip_if_not(Sys.getenv("RUN_SLOW_TESTS") == "true",
              "Skipping slow slopes+temporal test (set RUN_SLOW_TESTS=true to run)")

  set.seed(6101)
  n_sites <- 10
  n_times <- 6
  n_per <- 5
  n <- n_sites * n_times * n_per

  # Random slopes for x by site
  site <- rep(1:n_sites, each = n_times * n_per)
  time <- rep(rep(1:n_times, each = n_per), n_sites)

  re_intercept <- rnorm(n_sites, 0, 0.3)
  re_slope <- rnorm(n_sites, 0, 0.2)

  # AR1 temporal effect
  rho <- 0.7
  temporal_effect <- numeric(n_times)
  temporal_effect[1] <- rnorm(1, 0, 0.3)
  for (t in 2:n_times) {
    temporal_effect[t] <- rho * temporal_effect[t-1] + rnorm(1, 0, 0.2)
  }

  x <- rnorm(n)
  mu <- exp(2 + re_intercept[site] + (0.3 + re_slope[site]) * x + temporal_effect[time])
  y_num <- rpois(n, mu)
  y_denom_cont <- rgamma(n, shape = 5, rate = 5 / exp(2))

  df <- data.frame(
    y_num = y_num,
    y_denom_cont = y_denom_cont,
    x = x,
    site = factor(site),
    time = time
  )

  fit <- suppressWarnings(
    tratio(
      y_num | y_denom_cont ~ x + (x | site),
      data = df,
      temporal = temporal_ar1("time"),
      family = ratiod_poisson_gamma(),
      mode = "hmc",
      control = list(chains = 1, iter = 150, warmup = 75, verbose = FALSE)
    )
  )

  expect_s3_class(fit, "ratiod_fit")
})


test_that("negbin_negbin with random slopes + RW1 runs correctly (row 62)", {
  skip_on_cran()
  skip_if_not(Sys.getenv("RUN_SLOW_TESTS") == "true",
              "Skipping slow slopes+temporal test (set RUN_SLOW_TESTS=true to run)")

  set.seed(6201)
  n_sites <- 10
  n_times <- 6
  n_per <- 5
  n <- n_sites * n_times * n_per

  # Random slopes for x by site
  site <- rep(1:n_sites, each = n_times * n_per)
  time <- rep(rep(1:n_times, each = n_per), n_sites)

  re_intercept <- rnorm(n_sites, 0, 0.3)
  re_slope <- rnorm(n_sites, 0, 0.2)

  # RW1 temporal effect
  temporal_effect <- cumsum(rnorm(n_times, 0, 0.15))
  temporal_effect <- temporal_effect - mean(temporal_effect)

  x <- rnorm(n)
  mu_num <- exp(2 + re_intercept[site] + (0.3 + re_slope[site]) * x + temporal_effect[time])
  mu_denom <- exp(2.5 + 0.5 * temporal_effect[time])
  y_num <- rnbinom(n, size = 5, mu = mu_num)
  y_denom <- rnbinom(n, size = 8, mu = mu_denom)

  df <- data.frame(
    y_num = y_num,
    y_denom = y_denom,
    x = x,
    site = factor(site),
    time = time
  )

  fit <- suppressWarnings(
    tratio(
      y_num | y_denom ~ x + (x | site),
      data = df,
      temporal = temporal_rw1("time"),
      family = ratiod_negbin_negbin(),
      mode = "hmc",
      control = list(chains = 1, iter = 150, warmup = 75, verbose = FALSE)
    )
  )

  expect_s3_class(fit, "ratiod_fit")
})


test_that("binomial with random slopes + AR1 runs correctly (row 63)", {
  skip_on_cran()
  skip_if_not(Sys.getenv("RUN_SLOW_TESTS") == "true",
              "Skipping slow slopes+temporal test (set RUN_SLOW_TESTS=true to run)")

  set.seed(6301)
  n_sites <- 10
  n_times <- 6
  n_per <- 5
  n <- n_sites * n_times * n_per

  # Random slopes for x by site
  site <- rep(1:n_sites, each = n_times * n_per)
  time <- rep(rep(1:n_times, each = n_per), n_sites)

  re_intercept <- rnorm(n_sites, 0, 0.3)
  re_slope <- rnorm(n_sites, 0, 0.2)

  # AR1 temporal effect
  rho <- 0.7
  temporal_effect <- numeric(n_times)
  temporal_effect[1] <- rnorm(1, 0, 0.3)
  for (t in 2:n_times) {
    temporal_effect[t] <- rho * temporal_effect[t-1] + rnorm(1, 0, 0.2)
  }

  x <- rnorm(n)
  p <- plogis(re_intercept[site] + (0.3 + re_slope[site]) * x + temporal_effect[time])
  trials <- rpois(n, 25) + 10
  y_num <- rbinom(n, trials, p)

  df <- data.frame(
    y_num = y_num,
    y_denom = trials,
    x = x,
    site = factor(site),
    time = time
  )

  fit <- suppressWarnings(
    tratio(
      y_num | y_denom ~ x + (x | site),
      data = df,
      temporal = temporal_ar1("time"),
      family = ratiod_binomial(),
      mode = "hmc",
      control = list(chains = 1, iter = 150, warmup = 75, verbose = FALSE)
    )
  )

  expect_s3_class(fit, "ratiod_fit")
})


test_that("poisson_gamma with BYM2 + RW1 + ZI runs correctly (row 65)", {
  skip_on_cran()
  skip_if_not(Sys.getenv("RUN_SLOW_TESTS") == "true",
              "Skipping slow three-way test (set RUN_SLOW_TESTS=true to run)")

  set.seed(6501)
  n_sites <- 16
  n_times <- 5
  n <- n_sites * n_times

  # Grid adjacency
  side <- 4
  adj_mat <- matrix(0, n_sites, n_sites)
  for (i in 1:n_sites) {
    row_i <- (i - 1) %/% side + 1
    col_i <- (i - 1) %% side + 1
    for (j in 1:n_sites) {
      row_j <- (j - 1) %/% side + 1
      col_j <- (j - 1) %% side + 1
      if (i != j && abs(row_i - row_j) + abs(col_i - col_j) == 1) {
        adj_mat[i, j] <- 1
      }
    }
  }

  # Data structure
  site <- rep(1:n_sites, n_times)
  time <- rep(1:n_times, each = n_sites)

  # Spatial effect (BYM2)
  w_site <- rnorm(n_sites, 0, 0.4)

  # Temporal effect (RW1)
  temporal_effect <- cumsum(rnorm(n_times, 0, 0.2))
  temporal_effect <- temporal_effect - mean(temporal_effect)

  mu <- exp(2 + w_site[site] + temporal_effect[time])
  y_num <- rpois(n, mu)

  # Apply ZI (20% structural zeros)
  pi_zi <- 0.2
  zi_mask <- rbinom(n, 1, pi_zi)
  y_num[zi_mask == 1] <- 0

  y_denom_cont <- rgamma(n, shape = 5, rate = 5 / exp(2))

  df <- data.frame(
    y_num = y_num,
    y_denom_cont = y_denom_cont,
    site = factor(site),
    time = time
  )

  fit <- suppressWarnings(
    tratio(
      y_num | y_denom_cont ~ 1 + (1 | site),
      data = df,
      spatial = spatial_bym2(adj_mat, level = "group", group_var = "site"),
      temporal = temporal_rw1("time"),
      zi = ~ 1,
      family = ratiod_poisson_gamma(),
      mode = "hmc",
      control = list(chains = 1, iter = 150, warmup = 75, verbose = FALSE)
    )
  )

  expect_s3_class(fit, "ratiod_fit")
})


test_that("negbin_negbin with ICAR + RW1 + ZI runs correctly (row 66)", {
  skip_on_cran()
  skip_if_not(Sys.getenv("RUN_SLOW_TESTS") == "true",
              "Skipping slow three-way test (set RUN_SLOW_TESTS=true to run)")

  set.seed(6601)
  n_sites <- 16
  n_times <- 5
  n <- n_sites * n_times

  # Grid adjacency
  side <- 4
  adj_mat <- matrix(0, n_sites, n_sites)
  for (i in 1:n_sites) {
    row_i <- (i - 1) %/% side + 1
    col_i <- (i - 1) %% side + 1
    for (j in 1:n_sites) {
      row_j <- (j - 1) %/% side + 1
      col_j <- (j - 1) %% side + 1
      if (i != j && abs(row_i - row_j) + abs(col_i - col_j) == 1) {
        adj_mat[i, j] <- 1
      }
    }
  }

  # Data structure
  site <- rep(1:n_sites, n_times)
  time <- rep(1:n_times, each = n_sites)

  # Spatial effect
  w_site <- rnorm(n_sites, 0, 0.4)

  # Temporal effect (RW1)
  temporal_effect <- cumsum(rnorm(n_times, 0, 0.2))
  temporal_effect <- temporal_effect - mean(temporal_effect)

  mu_num <- exp(2 + w_site[site] + temporal_effect[time])
  mu_denom <- exp(2.5 + 0.5 * w_site[site])
  y_num <- rnbinom(n, size = 5, mu = mu_num)
  y_denom <- rnbinom(n, size = 8, mu = mu_denom)

  # Apply ZI to numerator (20% structural zeros)
  pi_zi <- 0.2
  zi_mask <- rbinom(n, 1, pi_zi)
  y_num[zi_mask == 1] <- 0

  df <- data.frame(
    y_num = y_num,
    y_denom = y_denom,
    site = factor(site),
    time = time
  )

  fit <- suppressWarnings(
    tratio(
      y_num | y_denom ~ 1 + (1 | site),
      data = df,
      spatial = spatial_icar(adj_mat, level = "group", group_var = "site"),
      temporal = temporal_rw1("time"),
      zi = ~ 1,
      family = ratiod_negbin_negbin(),
      mode = "hmc",
      control = list(chains = 1, iter = 150, warmup = 75, verbose = FALSE)
    )
  )

  expect_s3_class(fit, "ratiod_fit")
})
