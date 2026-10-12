#' Stochastic Gradient MCMC Backends for ratio Models
#'
#' @description
#' Stochastic gradient MCMC methods enable Tier 1 (Exact) inference on
#' large datasets by using minibatch gradients.
#'
#' @details
#' Two methods are provided:
#'
#' **SGHMC (Stochastic Gradient Hamiltonian Monte Carlo)**
#' Uses momentum with friction to correct for gradient noise.
#' Reference: Chen, Fox, Guestrin (2014) "Stochastic Gradient Hamiltonian Monte Carlo"
#'
#' **SGLD (Stochastic Gradient Langevin Dynamics)**
#' Simpler method without momentum, uses decaying step size schedule.
#' Reference: Welling & Teh (2011) "Bayesian Learning via Stochastic Gradient Langevin Dynamics"
#'
#' Both methods are asymptotically exact (Tier 1) - the continuous-time limit
#' samples from the true posterior. This distinguishes them from VI (Tier 3).
#'
#' @section When to Use:
#' - N > 10,000 observations
#' - Full HMC is too slow
#' - You need exact (Tier 1) inference, not approximations
#' - Model structure is relatively simple (RE, basic spatial/temporal)
#'
#' @name sghmc_backend
#' @keywords internal
NULL


#' Fit ratio model using Stochastic Gradient HMC
#'
#' @param formula A ratiod_formula object
#' @param data Data frame
#' @param family Model family
#' @param spatial Optional spatial structure
#' @param temporal Optional temporal structure
#' @param zi Optional zero-inflation specification
#' @param priors Prior specification
#' @param iter Total iterations
#' @param warmup Warmup iterations
#' @param batch_size Minibatch size (default: min(256, N/10))
#' @param epsilon Step size / learning rate (default: 0.01)
#' @param alpha Friction coefficient (default: 0.1)
#' @param L Leapfrog steps per iteration (default: 10)
#' @param seed Random seed
#' @param verbose Print progress
#'
#' @return A ratiod_fit object with class "ratiod_sghmc"
#' @keywords internal
fit_sghmc <- function(formula,
                      data,
                      family,
                      spatial = NULL,
                      temporal = NULL,
                      zi = NULL,
                      priors = NULL,
                      iter = 2000,
                      warmup = 1000,
                      batch_size = NULL,
                      epsilon = 0.01,
                      alpha = 0.1,
                      L = 10,
                      seed = NULL,
                      verbose = TRUE) {

  # Set seed
  if (is.null(seed)) {
    seed <- sample.int(.Machine$integer.max, 1)
  }

  # Get model type
  model_type <- get_hmc_model_type(family)

  # Extract data (reuse HMC preparation)
  hmc_data <- prepare_hmc_data(formula, data, family, model_type)

  # Set default batch size
  N <- hmc_data$N
  if (is.null(batch_size)) {
    batch_size <- min(256L, max(32L, as.integer(N / 10)))
  }
  batch_size <- as.integer(min(batch_size, N))

  # Prepare spatial structure
  spatial_info <- prepare_spatial_for_hmc(spatial, data, N)
  if (identical(spatial_info$type, "car_proper")) {
    stop("Proper CAR (`spatial_car(..., proper = TRUE)`) is not supported by ",
         "the SGHMC backend. Use `mode = \"hmc\"`.", call. = FALSE)
  }

  # Prepare temporal structure
  temporal_info <- prepare_temporal_for_hmc(temporal, data, N)

  # Prepare zero-inflation structure. Handles both an explicit `zi = `
  # argument and auto-detection from a ZI/hurdle/OI/ZOIB family when
  # `zi` is not given.
  zi_info <- prepare_zi_for_hmc(zi, data, N, family)

  # Prior parameters
  sp <- sampler_priors(priors, model_type)
  priors <- fill_priors(priors, model_type)
  sigma_beta <- sp$sigma_beta
  sigma_re_scale <- sp$sigma_re_scale
  phi_shape <- sp$phi_shape
  phi_rate <- sp$phi_rate
  tau_temporal_shape <- sp$tau_temporal_shape
  tau_temporal_rate <- sp$tau_temporal_rate
  tau_spatial_shape <- sp$tau_spatial_shape
  tau_spatial_rate <- sp$tau_spatial_rate

  # Bundle parameters
  re_params <- list(
    group = as.integer(hmc_data$re_group),
    n_groups = as.integer(hmc_data$n_re_groups),
    n_terms = as.integer(0)
  )

  spatial_params <- list(
    type = spatial_info$type,
    group = as.integer(spatial_info$group),
    n_units = as.integer(spatial_info$n_units %||% 0),
    adj_row_ptr = as.integer(spatial_info$adj_row_ptr %||% 0L),
    adj_col_idx = as.integer(spatial_info$adj_col_idx %||% integer(0)),
    n_neighbors = as.integer(spatial_info$n_neighbors %||% integer(0)),
    bym2_scale = spatial_info$bym2_scale %||% 1.0,
    component = spatial_info$component,
    node_prec = spatial_info$node_prec
  )

  temporal_params <- list(
    type = temporal_info$type %||% "none",
    time_idx = as.integer(temporal_info$time_index %||% rep(0L, N)),
    group_idx = as.integer(temporal_info$group_index %||% rep(0L, N)),
    n_times = as.integer(temporal_info$n_times %||% 0L),
    n_groups = as.integer(temporal_info$n_groups %||% 0L),
    n_params = as.integer(temporal_info$n_temporal_params %||% 0L),
    cyclic = temporal_info$precision_structure$cyclic %||% FALSE,
    shared = temporal_info$shared %||% TRUE,
    tau_shape = tau_temporal_shape,
    tau_rate = tau_temporal_rate
  )

  prior_params <- list(
    sigma_beta = sigma_beta,
    sigma_re_scale = sigma_re_scale,
    phi_shape = phi_shape,
    phi_rate = phi_rate,
    tau_spatial_shape = tau_spatial_shape,
    tau_spatial_rate = tau_spatial_rate
  )

  zi_params <- list(
    type = zi_info$type,
    X = zi_info$X_zi,
    p_zi = zi_info$p_zi %||% as.integer(ncol(zi_info$X_zi)),  # Explicit p_zi for OI-only models
    prior_sd = zi_info$prior_sd %||% 10.0,
    # OI info for OI-binomial and ZOIB models
    X_oi = zi_info$X_oi,
    p_oi = zi_info$p_oi %||% 0L,
    oi_prior_sd = sp$zi_prior_sd
  )

  latent_params <- list(
    has_latent = FALSE,
    n_factors = 0L,
    shared = FALSE,
    scale = TRUE,
    constraint = 0L,
    sigma_prior_rate = 0.0
  )

  st_params <- list(
    has_spatiotemporal = FALSE,
    type = "none"
  )

  # The sampler starts at zero on the layout it computes
  q_init <- numeric(0)

  if (verbose) {
    message("Stochastic Gradient HMC")
    message("  Observations: ", N)
    message("  Batch size: ", batch_size, " (", round(100 * batch_size / N, 1), "%)")
    message("  Iterations: ", iter, " (warmup: ", warmup, ")")
  }

  # Fit SGHMC model
  fit_raw <- cpp_sghmc_fit(
    q_init = q_init,
    y_num = as.integer(hmc_data$y_num),
    y_denom = as.integer(hmc_data$y_denom),
    y_num_cont = as.numeric(hmc_data$y_num_cont),
    y_denom_cont = hmc_data$y_denom_cont,
    X_num = hmc_data$X_num,
    X_denom = hmc_data$X_denom,
    family = family_params(family, model_type),
    re_params = re_params,
    spatial_params = spatial_params,
    temporal_params = temporal_params,
    prior_params = prior_params,
    zi_params = zi_params,
    latent_params = latent_params,
    st_params = st_params,
    n_iter = as.integer(iter),
    n_warmup = as.integer(warmup),
    batch_size = as.integer(batch_size),
    epsilon = epsilon,
    alpha = alpha,
    L = as.integer(L),
    seed = as.integer(seed),
    verbose = verbose
  )

  # Build result object
  result <- structure(list(
    # Samples
    draws = fit_raw$samples,
    log_lik = fit_raw$log_lik,

    # SGHMC-specific diagnostics
    epsilon_history = fit_raw$epsilon_history,
    final_epsilon = fit_raw$final_epsilon,
    batch_size = fit_raw$batch_size,

    # Model information
    formula = formula,
    data = data,
    family = family,
    model_type = model_type,
    n_obs = N,
    n_params = fit_raw$n_params,
    p_num = ncol(hmc_data$X_num),
    p_denom = ncol(hmc_data$X_denom),
    param_names = fit_raw$param_names,

    # Sampler settings
    iter = iter,
    warmup = warmup,
    chains = 1,

    # Backend info
    backend = "sghmc",
    seed = seed
  ), class = c("ratiod_sghmc", "ratiod_fit"))

  if (verbose) {
    message("SGHMC complete. Final epsilon: ", round(fit_raw$final_epsilon, 6))
  }

  result
}


#' Fit ratio model using Stochastic Gradient Langevin Dynamics
#'
#' @param formula A ratiod_formula object
#' @param data Data frame
#' @param family Model family
#' @param spatial Optional spatial structure
#' @param temporal Optional temporal structure
#' @param zi Optional zero-inflation specification
#' @param priors Prior specification
#' @param iter Total iterations
#' @param warmup Warmup iterations
#' @param batch_size Minibatch size
#' @param epsilon Initial step size
#' @param schedule_a Step size schedule parameter a
#' @param schedule_b Step size schedule parameter b
#' @param schedule_gamma Step size schedule decay rate
#' @param use_schedule Whether to use step size schedule
#' @param seed Random seed
#' @param verbose Print progress
#'
#' @return A ratiod_fit object with class "ratiod_sgld"
#' @keywords internal
fit_sgld <- function(formula,
                     data,
                     family,
                     spatial = NULL,
                     temporal = NULL,
                     zi = NULL,
                     priors = NULL,
                     iter = 2000,
                     warmup = 1000,
                     batch_size = NULL,
                     epsilon = 0.01,
                     schedule_a = 0.01,
                     schedule_b = 1.0,
                     schedule_gamma = 0.55,
                     use_schedule = TRUE,
                     seed = NULL,
                     verbose = TRUE) {

  # Set seed
  if (is.null(seed)) {
    seed <- sample.int(.Machine$integer.max, 1)
  }

  # Get model type
  model_type <- get_hmc_model_type(family)

  # Extract data
  hmc_data <- prepare_hmc_data(formula, data, family, model_type)

  # Set default batch size
  N <- hmc_data$N
  if (is.null(batch_size)) {
    batch_size <- min(256L, max(32L, as.integer(N / 10)))
  }
  batch_size <- as.integer(min(batch_size, N))

  # Prepare structures (same as SGHMC)
  spatial_info <- prepare_spatial_for_hmc(spatial, data, N)
  if (identical(spatial_info$type, "car_proper")) {
    stop("Proper CAR (`spatial_car(..., proper = TRUE)`) is not supported by ",
         "the SGHMC backend. Use `mode = \"hmc\"`.", call. = FALSE)
  }
  temporal_info <- prepare_temporal_for_hmc(temporal, data, N)

  # Prepare zero-inflation structure. Handles both an explicit `zi = `
  # argument and auto-detection from a ZI/hurdle/OI/ZOIB family when
  # `zi` is not given.
  zi_info <- prepare_zi_for_hmc(zi, data, N, family)

  # Priors
  sp <- sampler_priors(priors, model_type)
  priors <- fill_priors(priors, model_type)
  sigma_beta <- sp$sigma_beta
  sigma_re_scale <- sp$sigma_re_scale
  phi_shape <- sp$phi_shape
  phi_rate <- sp$phi_rate
  tau_temporal_shape <- sp$tau_temporal_shape
  tau_temporal_rate <- sp$tau_temporal_rate
  tau_spatial_shape <- sp$tau_spatial_shape
  tau_spatial_rate <- sp$tau_spatial_rate

  re_params <- list(
    group = as.integer(hmc_data$re_group),
    n_groups = as.integer(hmc_data$n_re_groups),
    n_terms = 0L
  )

  spatial_params <- list(
    type = spatial_info$type,
    group = as.integer(spatial_info$group),
    n_units = as.integer(spatial_info$n_units %||% 0),
    adj_row_ptr = as.integer(spatial_info$adj_row_ptr %||% 0L),
    adj_col_idx = as.integer(spatial_info$adj_col_idx %||% integer(0)),
    n_neighbors = as.integer(spatial_info$n_neighbors %||% integer(0)),
    bym2_scale = spatial_info$bym2_scale %||% 1.0,
    component = spatial_info$component,
    node_prec = spatial_info$node_prec
  )

  temporal_params <- list(
    type = temporal_info$type %||% "none",
    time_idx = as.integer(temporal_info$time_index %||% rep(0L, N)),
    group_idx = as.integer(temporal_info$group_index %||% rep(0L, N)),
    n_times = as.integer(temporal_info$n_times %||% 0L),
    n_groups = as.integer(temporal_info$n_groups %||% 0L),
    n_params = as.integer(temporal_info$n_temporal_params %||% 0L),
    cyclic = temporal_info$precision_structure$cyclic %||% FALSE,
    shared = temporal_info$shared %||% TRUE,
    tau_shape = tau_temporal_shape,
    tau_rate = tau_temporal_rate
  )

  prior_params <- list(
    sigma_beta = sigma_beta,
    sigma_re_scale = sigma_re_scale,
    phi_shape = phi_shape,
    phi_rate = phi_rate,
    tau_spatial_shape = tau_spatial_shape,
    tau_spatial_rate = tau_spatial_rate
  )

  zi_params <- list(
    type = zi_info$type,
    X = zi_info$X_zi,
    p_zi = zi_info$p_zi %||% as.integer(ncol(zi_info$X_zi)),  # Explicit p_zi for OI-only models
    prior_sd = zi_info$prior_sd %||% 10.0,
    # OI info for OI-binomial and ZOIB models
    X_oi = zi_info$X_oi,
    p_oi = zi_info$p_oi %||% 0L,
    oi_prior_sd = sp$zi_prior_sd
  )

  latent_params <- list(has_latent = FALSE, n_factors = 0L, shared = FALSE,
                        scale = TRUE, constraint = 0L, sigma_prior_rate = 0.0)

  st_params <- list(has_spatiotemporal = FALSE, type = "none")

  # The sampler starts at zero on the layout it computes
  q_init <- numeric(0)

  if (verbose) {
    message("Stochastic Gradient Langevin Dynamics")
    message("  Observations: ", N)
    message("  Batch size: ", batch_size, " (", round(100 * batch_size / N, 1), "%)")
    message("  Iterations: ", iter, " (warmup: ", warmup, ")")
    if (use_schedule) {
      message("  Step schedule: ", schedule_a, " * (", schedule_b, " + t)^(-", schedule_gamma, ")")
    }
  }

  # Fit SGLD model
  fit_raw <- cpp_sgld_fit(
    q_init = q_init,
    y_num = as.integer(hmc_data$y_num),
    y_denom = as.integer(hmc_data$y_denom),
    y_num_cont = as.numeric(hmc_data$y_num_cont),
    y_denom_cont = hmc_data$y_denom_cont,
    X_num = hmc_data$X_num,
    X_denom = hmc_data$X_denom,
    family = family_params(family, model_type),
    re_params = re_params,
    spatial_params = spatial_params,
    temporal_params = temporal_params,
    prior_params = prior_params,
    zi_params = zi_params,
    latent_params = latent_params,
    st_params = st_params,
    n_iter = as.integer(iter),
    n_warmup = as.integer(warmup),
    batch_size = as.integer(batch_size),
    epsilon = epsilon,
    schedule_a = schedule_a,
    schedule_b = schedule_b,
    schedule_gamma = schedule_gamma,
    use_schedule = use_schedule,
    seed = as.integer(seed),
    verbose = verbose
  )

  # Build result
  result <- structure(list(
    draws = fit_raw$samples,
    log_lik = fit_raw$log_lik,
    epsilon_history = fit_raw$epsilon_history,
    batch_size = fit_raw$batch_size,

    formula = formula,
    data = data,
    family = family,
    model_type = model_type,
    n_obs = N,
    n_params = fit_raw$n_params,
    p_num = ncol(hmc_data$X_num),
    p_denom = ncol(hmc_data$X_denom),
    param_names = fit_raw$param_names,

    iter = iter,
    warmup = warmup,
    chains = 1,

    backend = "sgld",
    seed = seed
  ), class = c("ratiod_sgld", "ratiod_fit"))

  if (verbose) {
    message("SGLD complete.")
  }

  result
}


#' @rdname summary.ratiod_fit
#'
#' @export
print.ratiod_sghmc <- function(x, digits = 3, ...) {
  cat("Stochastic Gradient HMC Fit\n")
  cat("Family:", class(x$family)[1], "\n")
  cat("Observations:", x$n_obs, "\n")
  cat("Parameters:", x$n_params, "\n")
  cat("Batch size:", x$batch_size, "(", round(100 * x$batch_size / x$n_obs, 1), "%)\n\n")

  cat("Sampling:\n")
  cat("  Iterations:", x$iter, "(warmup:", x$warmup, ")\n")
  cat("  Final epsilon:", round(x$final_epsilon, 6), "\n\n")

  # Coefficient summary
  cat("Posterior means:\n")
  draws <- x$draws
  means <- colMeans(draws)
  sds <- apply(draws, 2, sd)

  coef_summary <- data.frame(
    parameter = colnames(draws),
    mean = means,
    sd = sds
  )

  n_show <- min(10, nrow(coef_summary))
  print(coef_summary[1:n_show, ], digits = digits, row.names = FALSE)
  if (nrow(coef_summary) > n_show) {
    cat("... and", nrow(coef_summary) - n_show, "more parameters\n")
  }

  invisible(x)
}


#' @rdname summary.ratiod_fit
#'
#' @export
print.ratiod_sgld <- function(x, digits = 3, ...) {
  cat("Stochastic Gradient Langevin Dynamics Fit\n")
  cat("Family:", class(x$family)[1], "\n")
  cat("Observations:", x$n_obs, "\n")
  cat("Parameters:", x$n_params, "\n")
  cat("Batch size:", x$batch_size, "(", round(100 * x$batch_size / x$n_obs, 1), "%)\n\n")

  cat("Sampling:\n")
  cat("  Iterations:", x$iter, "(warmup:", x$warmup, ")\n\n")

  # Coefficient summary
  cat("Posterior means:\n")
  draws <- x$draws
  means <- colMeans(draws)
  sds <- apply(draws, 2, sd)

  coef_summary <- data.frame(
    parameter = colnames(draws),
    mean = means,
    sd = sds
  )

  n_show <- min(10, nrow(coef_summary))
  print(coef_summary[1:n_show, ], digits = digits, row.names = FALSE)
  if (nrow(coef_summary) > n_show) {
    cat("... and", nrow(coef_summary) - n_show, "more parameters\n")
  }

  invisible(x)
}

