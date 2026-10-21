#' Stochastic Gradient MCMC Backends for ratio Models
#'
#' @description
#' Stochastic gradient MCMC methods sample large datasets from minibatch
#' gradients. They are Tier 3 (Optimized): see Details.
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
#' Their continuous-time limit samples the posterior, but both run a fixed
#' step size on minibatch gradients with no Metropolis correction, so a run
#' samples a distribution perturbed by the discretization and the gradient
#' noise. They share Tier 3 with VI for that reason.
#'
#' @section When to Use:
#' - N > 10,000 observations
#' - Full HMC is too slow
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

  inp <- sampler_inputs(formula, data, family, spatial, temporal,
                        NULL, zi, NULL, priors, re_param = "centered")
  hmc_data <- inp$hmc_data
  model_type <- inp$model_type
  if (identical(inp$spatial_info$type, "car_proper")) {
    stop("Proper CAR (`spatial_car(..., proper = TRUE)`) is not supported by ",
         "the SGHMC backend. Use `mode = \"hmc\"`.", call. = FALSE)
  }
  assert_single_random_intercept(hmc_data, "sghmc")

  # Set default batch size
  N <- hmc_data$N
  if (is.null(batch_size)) {
    batch_size <- min(256L, max(32L, as.integer(N / 10)))
  }
  batch_size <- as.integer(min(batch_size, N))

  # The sampler starts at zero on the layout it computes
  q_init <- numeric(0)

  if (verbose) {
    message("Stochastic Gradient HMC")
    message("  Observations: ", N)
    message("  Batch size: ", batch_size, " (", round(100 * batch_size / N, 1), "%)")
    message("  Iterations: ", iter, " (warmup: ", warmup, ")")
  }

  # Fit SGHMC model
  fit_raw <- do.call(cpp_sghmc_fit, c(
    list(q_init = q_init),
    model_data_args(inp, family),
    list(n_iter = as.integer(iter), n_warmup = as.integer(warmup),
         batch_size = as.integer(batch_size), epsilon = epsilon,
         alpha = alpha, L = as.integer(L), seed = as.integer(seed),
         verbose = verbose)
  ))

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

  inp <- sampler_inputs(formula, data, family, spatial, temporal,
                        NULL, zi, NULL, priors, re_param = "centered")
  hmc_data <- inp$hmc_data
  model_type <- inp$model_type
  if (identical(inp$spatial_info$type, "car_proper")) {
    stop("Proper CAR (`spatial_car(..., proper = TRUE)`) is not supported by ",
         "the SGLD backend. Use `mode = \"hmc\"`.", call. = FALSE)
  }
  assert_single_random_intercept(hmc_data, "sgld")

  # Set default batch size
  N <- hmc_data$N
  if (is.null(batch_size)) {
    batch_size <- min(256L, max(32L, as.integer(N / 10)))
  }
  batch_size <- as.integer(min(batch_size, N))

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
  fit_raw <- do.call(cpp_sgld_fit, c(
    list(q_init = q_init),
    model_data_args(inp, family),
    list(n_iter = as.integer(iter), n_warmup = as.integer(warmup),
         batch_size = as.integer(batch_size), epsilon = epsilon,
         schedule_a = schedule_a, schedule_b = schedule_b,
         schedule_gamma = schedule_gamma, use_schedule = use_schedule,
         seed = as.integer(seed), verbose = verbose)
  ))

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

