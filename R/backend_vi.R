#' Variational Inference Backend for ratio Models
#'
#' @description
#' Automatic Differentiation Variational Inference (ADVI) backend.
#' Provides fast approximate Bayesian inference using three Gaussian
#' approximation variants.
#'
#' @details
#' This backend implements:
#' - **Mean-field**: Diagonal covariance, fastest but ignores correlations
#' - **Low-rank**: LL' + D covariance, good balance of speed and quality
#' - **Full-rank**: Full Cholesky covariance, best quality for small D
#'
#' **Automatic variant selection (default):**
#' - D < 200: Full-rank (best quality, feasible speed)
#' - 200 <= D < 2000: Low-rank with rank = D/10
#' - D >= 2000: Mean-field (only feasible option)
#'
#' **Optimization:**
#' - Adam optimizer with gradient clipping
#' - Reparameterization trick for gradient estimation
#' - Monte Carlo ELBO approximation
#'
#' **When to use:**
#' - Interactive exploration (10-100x faster than MCMC)
#' - Large datasets where MCMC is too slow
#' - Quick model comparison and diagnostics
#' - When approximate posteriors are acceptable
#'
#' **Limitations:**
#' - Approximation quality varies by model complexity
#' - May underestimate posterior variance (especially mean-field
#' - Check PSIS k-hat diagnostic for approximation quality
#'
#' @name vi_backend
#' @keywords internal
NULL


#' Fit ratio model using Variational Inference
#'
#' @param formula A ratiod_formula object
#' @param data Data frame
#' @param family Model family
#' @param spatial Optional spatial structure (limited support)
#' @param temporal Optional temporal structure (limited support)
#' @param zi Optional zero-inflation specification
#' @param priors Prior specification
#' @param variant VI variant: "auto", "meanfield", "lowrank", "fullrank"
#' @param max_iter Maximum optimization iterations
#' @param mc_samples Monte Carlo samples for gradient estimation
#' @param tol Convergence tolerance
#' @param rank Low-rank dimension (NULL = auto)
#' @param seed Random seed
#' @param verbose Print progress
#'
#' @return A ratiod_fit object with class "ratiod_vi"
#' @keywords internal
fit_vi <- function(formula,
                   data,
                   family,
                   spatial = NULL,
                   temporal = NULL,
                   spatiotemporal = NULL,
                   zi = NULL,
                   latent = NULL,
                   priors = NULL,
                   variant = "auto",
                   max_iter = 10000,
                   mc_samples = 10,
                   tol = 1e-4,
                   rank = NULL,
                   seed = NULL,
                   verbose = TRUE) {

  # Set seed
  if (is.null(seed)) {
    seed <- sample.int(.Machine$integer.max, 1)
  }

  assert_backend_fits_structures(
    "vi",
    list(spatial = spatial, temporal = temporal,
         spatiotemporal = spatiotemporal, zi = zi, latent = latent)
  )

  inp <- sampler_inputs(formula, data, family, spatial, temporal,
                        spatiotemporal, zi, latent, priors, re_param = "centered")
  hmc_data <- inp$hmc_data
  model_type <- inp$model_type
  if (identical(inp$spatial_info$type, "car_proper")) {
    stop("Proper CAR (`spatial_car(..., proper = TRUE)`) is not supported by ",
         "the VI backend. Use `mode = \"hmc\"`.", call. = FALSE)
  }
  assert_single_random_intercept(hmc_data, "vi")

  # VI options
  vi_options <- list(
    variant = variant,
    max_iter = as.integer(max_iter),
    mc_samples = as.integer(mc_samples),
    tol_grad = tol,
    tol_rel_elbo = 0.01,
    patience = 50L,
    adam_alpha = 0.01,
    adam_beta1 = 0.9,
    adam_beta2 = 0.999,
    adam_eps = 1e-8,
    rank = if (is.null(rank)) -1L else as.integer(rank),
    use_laplace_init = TRUE,
    verbose = verbose,
    print_every = 100L,
    seed = as.integer(seed)
  )

  # The variational mean starts at zero on the layout the fit computes
  q_init <- numeric(0)

  if (verbose) {
    message("Variational Inference")
    message("  Variant: ", variant)
    message("  Max iterations: ", max_iter)
  }

  # Fit VI model
  fit_raw <- do.call(cpp_vi_fit, c(
    list(q_init = q_init),
    model_data_args(inp, family),
    list(vi_options = vi_options, verbose = verbose)
  ))

  # Build result object
  result <- structure(list(
    # VI-specific results
    mu = fit_raw$mu,
    Sigma = fit_raw$Sigma,
    L = fit_raw$L,
    d = fit_raw$d,
    vi_variant = fit_raw$variant,
    vi_rank = fit_raw$rank,
    elbo = fit_raw$elbo,
    elbo_history = fit_raw$elbo_history,
    vi_iterations = fit_raw$iterations,
    vi_converged = fit_raw$converged,
    psis_k = fit_raw$psis_k,

    # Samples for compatibility
    samples = fit_raw$samples,
    n_samples = nrow(fit_raw$samples),
    draws = NULL,  # Will be set below after param_names

    # Model information
    formula = formula,
    data = data,
    family = family,
    model_type = model_type,
    n_obs = hmc_data$N,
    n_params = length(fit_raw$param_names),
    p_num = ncol(hmc_data$X_num),
    p_denom = ncol(hmc_data$X_denom),
    param_names = fit_raw$param_names,

    # Backend info
    backend = "vi",
    seed = seed
  ), class = c("ratiod_vi", "ratiod_fit"))

  # Name the samples matrix and create draws field
  if (!is.null(result$samples) && ncol(result$samples) == length(result$param_names)) {
    colnames(result$samples) <- result$param_names
    result$draws <- result$samples
  } else if (!is.null(result$samples) && ncol(result$samples) <= length(result$param_names)) {
    # Partial match: use first n param names
    colnames(result$samples) <- result$param_names[seq_len(ncol(result$samples))]
    result$draws <- result$samples
  }

  # Add diagnostic warnings
  if (!fit_raw$converged) {
    warning("VI did not converge within ", max_iter, " iterations")
  }

  if (fit_raw$psis_k > 0.7) {
    warning("PSIS k-hat = ", round(fit_raw$psis_k, 2),
            " > 0.7: VI approximation may be unreliable. ",
            "mode = \"hmc\" samples the posterior itself.")
  } else if (fit_raw$psis_k > 0.5) {
    message("PSIS k-hat = ", round(fit_raw$psis_k, 2),
            " (marginal): approximation quality is acceptable but not ideal.")
  }

  result
}


#' @rdname summary.ratiod_fit
#'
#' @export
print.ratiod_vi <- function(x, digits = 3, ...) {
  variant_names <- c(
    meanfield = "Mean-Field",
    lowrank = "Low-Rank",
    fullrank = "Full-Rank"
  )

  cat("Variational Inference (", variant_names[x$vi_variant], ")\n", sep = "")
  cat("Family:", class(x$family)[1], "\n")
  cat("Observations:", x$n_obs, "\n")
  cat("Parameters:", x$n_params)
  if (x$vi_variant == "lowrank") {
    cat(" (rank =", x$vi_rank, ")")
  }
  cat("\n\n")

  cat("ELBO:", round(x$elbo, 2), "\n")
  cat("Iterations:", x$vi_iterations)
  if (x$vi_converged) cat(" (converged)") else cat(" (not converged)")
  cat("\n\n")

  # PSIS diagnostic
  cat("Diagnostic: ")
  if (x$psis_k < 0) {
    cat("PSIS k-hat not computed\n")
  } else if (x$psis_k < 0.5) {
    cat("PSIS k-hat =", round(x$psis_k, 2), "(good)\n")
  } else if (x$psis_k < 0.7) {
    cat("PSIS k-hat =", round(x$psis_k, 2), "(marginal)\n")
  } else {
    cat("PSIS k-hat =", round(x$psis_k, 2), "(WARNING: poor approximation)\n")
  }
  cat("\n")

  # Coefficient summary
  cat("Posterior means:\n")
  coef_summary <- data.frame(
    parameter = x$param_names,
    mean = x$mu,
    sd = sqrt(diag(x$Sigma))
  )
  # Show first 10 parameters
  n_show <- min(10, nrow(coef_summary))
  print(coef_summary[1:n_show, ], digits = digits, row.names = FALSE)
  if (nrow(coef_summary) > n_show) {
    cat("... and", nrow(coef_summary) - n_show, "more parameters\n")
  }

  invisible(x)
}


#' @rdname summary.ratiod_fit
#' @return An object of class `summary.ratiod_vi`.
#' @export
summary.ratiod_vi <- function(object, prob = 0.95, ...) {
  alpha <- (1 - prob) / 2

  # Compute quantiles from samples
  q_lower <- apply(object$samples, 2, quantile, probs = alpha)
  q_upper <- apply(object$samples, 2, quantile, probs = 1 - alpha)

  summary_df <- data.frame(
    parameter = object$param_names,
    mean = object$mu,
    sd = sqrt(diag(object$Sigma)),
    q_lower = q_lower,
    q_upper = q_upper
  )
  names(summary_df)[4:5] <- paste0(c("q", "q"), c(alpha * 100, (1 - alpha) * 100))

  structure(list(
    summary = summary_df,
    elbo = object$elbo,
    vi_variant = object$vi_variant,
    vi_rank = object$vi_rank,
    vi_converged = object$vi_converged,
    psis_k = object$psis_k,
    prob = prob
  ), class = "summary.ratiod_vi")
}


#' @rdname summary.ratiod_fit
#' @return The input `x`, invisibly.
#' @export
print.summary.ratiod_vi <- function(x, digits = 3, ...) {
  cat("Variational Inference Summary\n")
  cat("ELBO:", round(x$elbo, 2), "\n")
  cat("PSIS k-hat:", round(x$psis_k, 2), "\n\n")

  cat("Posterior summary (", x$prob * 100, "% credible intervals):\n", sep = "")
  print(x$summary, digits = digits, row.names = FALSE)

  invisible(x)
}
