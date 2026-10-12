# Tests for HMC backend
# Note: Tests for deprecated cpp_hmc_simple and cpp_nuts_fit have been removed
# as those functions were moved to src/deprecated/ during C++ refactoring.

# Helper functions for creating bundled parameters
make_re_params_hmc <- function(n, group = rep(0L, n), n_groups = 0L, n_terms = 0L) {
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

make_spatial_params_hmc <- function(n) {
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

make_temporal_params_hmc <- function(n) {
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

make_prior_params_hmc <- function() {
  list(
    sigma_beta = 10.0,
    sigma_re_scale = 2.5,
    phi_shape = 2.0,
    phi_rate = 0.1,
    tau_spatial_shape = 1.0,
    tau_spatial_rate = 0.01
  )
}

make_zi_params_hmc <- function(n) {
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

make_latent_params_hmc <- function() {
  list(
    has_latent = FALSE,
    n_factors = 0L,
    shared = FALSE,
    scale = TRUE,
    constraint = 0L,
    sigma_prior_rate = 0.0
  )
}

make_st_params_hmc <- function() {
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

make_tvc_params_hmc <- function() {
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

make_svc_params_hmc <- function() {
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

test_that("HMC backend works for negbin_negbin model", {
  skip_on_cran()

  set.seed(456)

  # Generate NegBin-NegBin data
  N <- 80
  x <- rnorm(N)
  X <- cbind(1, x)
  true_beta_num <- c(2.0, 0.3)
  true_beta_denom <- c(2.5, -0.2)
  true_phi <- 5.0

  eta_num <- X %*% true_beta_num
  eta_denom <- X %*% true_beta_denom
  mu_num <- exp(eta_num)
  mu_denom <- exp(eta_denom)

  y_num <- rnbinom(N, size = true_phi, mu = mu_num)
  y_denom <- rnbinom(N, size = true_phi, mu = mu_denom)

  # Parameters: beta_num(2) + beta_denom(2) + log_phi_num + log_phi_denom = 6
  q_init <- c(rep(0.0, 4), log(5), log(5))  # Initialize phi at 5

  # Use cpp_hmc_fit with bundled list arguments
  fit <- tulpaRatio:::cpp_hmc_fit(
    q_init = q_init,
    y_num = as.integer(y_num),
    y_denom = as.integer(y_denom),
    y_num_cont = rep(0.0, N),
    y_denom_cont = rep(0.0, N),
    X_num = X,
    X_denom = X,
    family = list(model_type = "negbin_negbin"),
    re_params = make_re_params_hmc(N),
    spatial_params = make_spatial_params_hmc(N),
    temporal_params = make_temporal_params_hmc(N),
    prior_params = make_prior_params_hmc(),
    zi_params = make_zi_params_hmc(N),
    latent_params = make_latent_params_hmc(),
    st_params = make_st_params_hmc(),
    tvc_params = make_tvc_params_hmc(),
    svc_params = make_svc_params_hmc(),
    n_iter = 200L,
    n_warmup = 100L,
    L = 10L,
    n_chains = 1L,
    seed = 456L,
    n_threads = 1L,
    verbose = FALSE
  )

  expect_true(all(is.finite(fit$samples)))
  expect_equal(nrow(fit$samples), 100)
  expect_equal(ncol(fit$samples), 6)

  # Check intercepts are in right ballpark
  beta_num_post <- colMeans(fit$samples[, 1:2])
  expect_equal(beta_num_post[1], true_beta_num[1], tolerance = 0.5)
})

test_that("HMC backend works for poisson_gamma model", {
  skip_on_cran()

  set.seed(789)

  # Generate Poisson-Gamma data (CPUE-like)
  N <- 60
  x <- rnorm(N)
  X <- cbind(1, x)
  true_beta_num <- c(2.0, 0.4)
  true_beta_denom <- c(1.5, 0.2)
  true_shape <- 3.0

  eta_num <- X %*% true_beta_num
  eta_denom <- X %*% true_beta_denom

  y_num <- rpois(N, exp(eta_num))
  y_denom <- rgamma(N, shape = true_shape, rate = true_shape / exp(eta_denom))

  # Parameters: beta_num(2) + beta_denom(2) + log_shape = 5
  q_init <- c(rep(0.0, 4), log(3))

  # Use cpp_hmc_fit with bundled list arguments
  fit <- tulpaRatio:::cpp_hmc_fit(
    q_init = q_init,
    y_num = as.integer(y_num),
    y_denom = as.integer(rep(1, N)),  # placeholder
    y_num_cont = rep(0.0, N),
    y_denom_cont = y_denom,
    X_num = X,
    X_denom = X,
    family = list(model_type = "poisson_gamma"),
    re_params = make_re_params_hmc(N),
    spatial_params = make_spatial_params_hmc(N),
    temporal_params = make_temporal_params_hmc(N),
    prior_params = make_prior_params_hmc(),
    zi_params = make_zi_params_hmc(N),
    latent_params = make_latent_params_hmc(),
    st_params = make_st_params_hmc(),
    tvc_params = make_tvc_params_hmc(),
    svc_params = make_svc_params_hmc(),
    n_iter = 200L,
    n_warmup = 100L,
    L = 10L,
    n_chains = 1L,
    seed = 789L,
    n_threads = 1L,
    verbose = FALSE
  )

  expect_true(all(is.finite(fit$samples)))
  expect_equal(nrow(fit$samples), 100)
  expect_equal(ncol(fit$samples), 5)
})

# ---------------------------------------------------------------------------
# Additional HMC integration tests with tratio()
# ---------------------------------------------------------------------------

test_that("tratio() fits poisson_gamma model with HMC backend", {
  skip_on_cran()

  set.seed(123)
  n <- 30
  df <- data.frame(
    count = rpois(n, lambda = 10),
    effort = rgamma(n, shape = 5, rate = 1),
    x = rnorm(n)
  )

  fit <- tratio(
    count | effort ~ x,
    data = df,
    family = ratiod_poisson_gamma(),
    mode = "hmc",
    control = list(iter = 100, warmup = 50, chains = 1, verbose = FALSE)
  )

  expect_s3_class(fit, "ratiod_fit")
  expect_equal(fit$backend, "hmc")
  expect_true(is.array(fit$draws))
})

test_that("tratio() fits binomial model with HMC backend", {
  skip_on_cran()

  set.seed(456)
  n <- 30
  trials <- sample(10:20, n, replace = TRUE)
  df <- data.frame(
    successes = rbinom(n, trials, 0.4),
    trials = trials,
    x = rnorm(n)
  )

  fit <- tratio(
    successes | trials ~ x,
    data = df,
    family = ratiod_binomial(),
    mode = "hmc",
    control = list(iter = 100, warmup = 50, chains = 1, verbose = FALSE)
  )

  expect_s3_class(fit, "ratiod_fit")
  expect_equal(fit$backend, "hmc")
  expect_true(is.array(fit$draws))
})

test_that("tratio() fits negbin_negbin model with HMC backend", {
  skip_on_cran()

  set.seed(789)
  n <- 30
  df <- data.frame(
    y_num = rnbinom(n, size = 5, mu = 10),
    y_denom = rnbinom(n, size = 5, mu = 15),
    x = rnorm(n)
  )

  fit <- tratio(
    y_num | y_denom ~ x,
    data = df,
    family = ratiod_negbin_negbin(),
    mode = "hmc",
    control = list(iter = 100, warmup = 50, chains = 1, verbose = FALSE)
  )

  expect_s3_class(fit, "ratiod_fit")
  expect_equal(fit$backend, "hmc")
})

test_that("tratio() with random effects using HMC backend", {
  skip_on_cran()

  set.seed(111)
  n_groups <- 5
  n_per_group <- 6
  n <- n_groups * n_per_group

  df <- data.frame(
    count = rpois(n, lambda = 10),
    effort = rgamma(n, shape = 5, rate = 1),
    x = rnorm(n),
    site = factor(rep(1:n_groups, each = n_per_group))
  )

  fit <- tratio(
    count | effort ~ x + (1 | site),
    data = df,
    family = ratiod_poisson_gamma(),
    mode = "hmc",
    control = list(iter = 100, warmup = 50, chains = 1, verbose = FALSE)
  )

  expect_s3_class(fit, "ratiod_fit")
  # Check that draws have sigma parameters (3D array or matrix)
  draws <- fit$draws
  if (is.array(draws) && length(dim(draws)) == 3) {
    param_names <- dimnames(draws)[[3]]
  } else if (is.matrix(draws)) {
    param_names <- colnames(draws)
  } else {
    param_names <- names(draws)
  }
  expect_true(any(grepl("sigma", param_names, ignore.case = TRUE)) ||
              length(param_names) > 5)  # Has RE parameters
})

test_that("summary.ratiod_fit returns correct structure", {
  skip_on_cran()

  set.seed(222)
  n <- 20
  df <- data.frame(
    count = rpois(n, lambda = 10),
    effort = rgamma(n, shape = 5, rate = 1),
    x = rnorm(n)
  )

  fit <- tratio(
    count | effort ~ x,
    data = df,
    family = ratiod_poisson_gamma(),
    mode = "hmc",
    control = list(iter = 100, warmup = 50, chains = 1, verbose = FALSE)
  )

  summ <- summary(fit)

  expect_true(is.data.frame(summ) || is.list(summ))
})

test_that("print.ratiod_fit works without error", {
  skip_on_cran()

  set.seed(333)
  n <- 20
  df <- data.frame(
    count = rpois(n, lambda = 10),
    effort = rgamma(n, shape = 5, rate = 1),
    x = rnorm(n)
  )

  fit <- tratio(
    count | effort ~ x,
    data = df,
    family = ratiod_poisson_gamma(),
    mode = "hmc",
    control = list(iter = 100, warmup = 50, chains = 1, verbose = FALSE)
  )

  output <- capture.output(print(fit))
  expect_true(length(output) > 0)
})

test_that("ratio() works with HMC fit", {
  skip_on_cran()

  set.seed(444)
  n <- 20
  df <- data.frame(
    count = rpois(n, lambda = 10),
    effort = rgamma(n, shape = 5, rate = 1),
    x = rnorm(n)
  )

  fit <- tratio(
    count | effort ~ x,
    data = df,
    family = ratiod_poisson_gamma(),
    mode = "hmc",
    control = list(iter = 100, warmup = 50, chains = 1, verbose = FALSE)
  )

  r <- ratio(fit)

  expect_s3_class(r, "ratiod_ratio")
  expect_true(is.matrix(r$draws))
  expect_equal(ncol(r$draws), n)
})

test_that("ratio_contrast() works with HMC fit", {
  skip_on_cran()

  set.seed(555)
  n <- 20
  df <- data.frame(
    count = rpois(n, lambda = 10),
    effort = rgamma(n, shape = 5, rate = 1),
    x = rnorm(n),
    group = factor(rep(c("A", "B"), each = n/2))
  )

  fit <- tratio(
    count | effort ~ x + group,
    data = df,
    family = ratiod_poisson_gamma(),
    mode = "hmc",
    control = list(iter = 100, warmup = 50, chains = 1, verbose = FALSE)
  )

  # ratio_contrast takes a formula for contrast
  contrast <- ratio_contrast(fit, contrast = ~ group)

  expect_true(is.numeric(contrast) || is.list(contrast) || inherits(contrast, "ratiod_contrast"))
})
