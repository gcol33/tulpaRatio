#' Elliptical Slice Sampling Backend for ratio Models
#'
#' @description
#' Elliptical Slice Sampling (ESS) backend for models with Gaussian priors.
#' ESS is particularly efficient for:
#' - Random effects (which have N(0, sigma^2) priors)
#' - Spatial effects (which have GMRF priors)
#' - GP effects (which have multivariate normal priors)
#'
#' @details
#' ESS has several advantages over HMC for models with Gaussian priors:
#' - **No tuning parameters**: Unlike HMC step size, ESS adapts automatically
#' - **Efficient for high-dimensional Gaussians**: Exploits prior structure
#' - **Exact transitions**: No discretization error (unlike leapfrog)
#'
#' ESS works by:
#' 1. Drawing a proposal from the prior (Gaussian)
#' 2. Defining an ellipse through current state and proposal
#' 3. Slice sampling along this ellipse
#'
#' Non-Gaussian parameters (variance components, correlations) are updated
#' using adaptive random-walk Metropolis-Hastings within each ESS iteration.
#'
#' @references
#' Murray, I., Adams, R., & MacKay, D. (2010). Elliptical slice sampling.
#' Proceedings of the 13th International Conference on Artificial Intelligence
#' and Statistics (AISTATS).
#'
#' @name ess_backend
#' @keywords internal
NULL


#' Fit ratio model using Elliptical Slice Sampling
#'
#' @param formula A ratiod_formula object
#' @param data Data frame
#' @param family Model family
#' @param spatial Optional spatial structure
#' @param temporal Optional temporal structure
#' @param spatiotemporal Optional spatiotemporal structure
#' @param zi Optional zero-inflation specification
#' @param latent Optional latent factor specification
#' @param priors Prior specification
#' @param iter Total iterations
#' @param warmup Warmup iterations
#' @param seed Random seed
#' @param verbose Print progress
#'
#' @return A ratiod_fit object with class "ratiod_ess"
#' @keywords internal
fit_ess <- function(formula,
                    data,
                    family,
                    spatial = NULL,
                    temporal = NULL,
                    spatiotemporal = NULL,
                    zi = NULL,
                    latent = NULL,
                    priors = NULL,
                    iter = 2000,
                    warmup = 1000,
                    seed = NULL,
                    verbose = TRUE) {

  # Set seed
  if (is.null(seed)) {
    seed <- sample.int(.Machine$integer.max, 1)
  }

  assert_backend_fits_structures(
    "ess",
    list(spatial = spatial, temporal = temporal,
         spatiotemporal = spatiotemporal, zi = zi, latent = latent)
  )

  # Get model type
  model_type <- get_hmc_model_type(family)

  # Extract data (reuse HMC preparation)
  hmc_data <- prepare_hmc_data(formula, data, family, model_type)

  # Prepare spatial structure
  spatial_info <- prepare_spatial_for_hmc(spatial, data, hmc_data$N)
  if (identical(spatial_info$type, "car_proper")) {
    stop("Proper CAR (`spatial_car(..., proper = TRUE)`) is not supported by ",
         "the ESS backend. Use `mode = \"hmc\"`.", call. = FALSE)
  }

  # Prepare temporal structure
  temporal_info <- prepare_temporal_for_hmc(temporal, data, hmc_data$N)

  # Prepare zero-inflation structure. Handles both an explicit `zi = `
  # argument and auto-detection from a ZI/hurdle/OI/ZOIB family when
  # `zi` is not given.
  zi_info <- prepare_zi_for_hmc(zi, data, hmc_data$N, family)

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
  rho_temporal_ab <- rho_prior_anchors(temporal_info$rho_prior,
                                       priors$rho_temporal)

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
    bym2_scale = spatial_info$bym2_scale %||% 1.0
  )

  temporal_params <- list(
    type = temporal_info$type %||% "none",
    time_idx = as.integer(temporal_info$time_index %||% rep(0L, hmc_data$N)),
    group_idx = as.integer(temporal_info$group_index %||% rep(0L, hmc_data$N)),
    n_times = as.integer(temporal_info$n_times %||% 0L),
    n_groups = as.integer(temporal_info$n_groups %||% 0L),
    n_params = as.integer(temporal_info$n_temporal_params %||% 0L),
    cyclic = temporal_info$precision_structure$cyclic %||% FALSE,
    shared = temporal_info$shared %||% TRUE,
    tau_shape = tau_temporal_shape,
    tau_rate = tau_temporal_rate,
    rho_prior_a = rho_temporal_ab[["a"]],
    rho_prior_b = rho_temporal_ab[["b"]]
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

  # Compute number of parameters
  n_params <- cpp_ess_get_n_params(
    hmc_data$X_num,
    hmc_data$X_denom,
    model_type,
    re_params,
    spatial_params,
    temporal_params,
    zi_params
  )

  # Initialize from zeros
  q_init <- rep(0.0, n_params)

  if (verbose) {
    message("Elliptical Slice Sampling")
    message("  Parameters: ", n_params)
    message("  Iterations: ", iter, " (warmup: ", warmup, ")")
  }

  # Fit ESS model
  fit_raw <- cpp_ess_fit(
    q_init = q_init,
    y_num = as.integer(hmc_data$y_num),
    y_denom = as.integer(hmc_data$y_denom),
    y_denom_cont = hmc_data$y_denom_cont,
    X_num = hmc_data$X_num,
    X_denom = hmc_data$X_denom,
    model_type_str = model_type,
    re_params = re_params,
    spatial_params = spatial_params,
    temporal_params = temporal_params,
    prior_params = prior_params,
    zi_params = zi_params,
    latent_params = latent_params,
    st_params = st_params,
    n_iter = as.integer(iter),
    n_warmup = as.integer(warmup),
    seed = as.integer(seed),
    verbose = verbose
  )

  # Build result object
  result <- structure(list(
    # Samples
    draws = fit_raw$samples,
    log_lik = fit_raw$log_lik,

    # ESS-specific diagnostics
    n_slice_evals = fit_raw$n_slice_evals,
    avg_slice_evals = fit_raw$avg_slice_evals,

    # Model information
    formula = formula,
    data = data,
    family = family,
    model_type = model_type,
    n_obs = hmc_data$N,
    n_params = fit_raw$n_params,
    p_num = ncol(hmc_data$X_num),
    p_denom = ncol(hmc_data$X_denom),
    param_names = fit_raw$param_names,

    # Sampler settings
    iter = iter,
    warmup = warmup,
    chains = 1,  # ESS runs single chain

    # Backend info
    backend = "ess",
    seed = seed
  ), class = c("ratiod_ess", "ratiod_fit"))

  if (verbose) {
    message("ESS complete. Avg slice evaluations per step: ",
            round(fit_raw$avg_slice_evals, 2))
  }

  result
}


#' @rdname summary.ratiod_fit
#' @param x A fit returned by [tratio()]: a `ratiod_fit`, or one of the
#'   backend subclasses `ratiod_ess`, `ratiod_vi`, `ratiod_sghmc` and
#'   `ratiod_sgld`, or the summary of one.
#' @param digits Number of significant digits to print (default 3).
#'
#' @export
print.ratiod_ess <- function(x, digits = 3, ...) {
  cat("Elliptical Slice Sampling Fit\n")
  cat("Family:", class(x$family)[1], "\n")
  cat("Observations:", x$n_obs, "\n")
  cat("Parameters:", x$n_params, "\n\n")

  cat("Sampling:\n")
  cat("  Iterations:", x$iter, "(warmup:", x$warmup, ")\n")
  cat("  Avg slice evals:", round(x$avg_slice_evals, 2), "\n\n")

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

  # Show first 10 parameters
  n_show <- min(10, nrow(coef_summary))
  print(coef_summary[1:n_show, ], digits = digits, row.names = FALSE)
  if (nrow(coef_summary) > n_show) {
    cat("... and", nrow(coef_summary) - n_show, "more parameters\n")
  }

  invisible(x)
}


#' @rdname summary.ratiod_fit
#' @return An object of class `summary.ratiod_ess`.
#' @export
summary.ratiod_ess <- function(object, prob = 0.95, ...) {
  alpha <- (1 - prob) / 2

  draws <- object$draws
  q_lower <- apply(draws, 2, quantile, probs = alpha)
  q_upper <- apply(draws, 2, quantile, probs = 1 - alpha)

  summary_df <- data.frame(
    parameter = colnames(draws),
    mean = colMeans(draws),
    sd = apply(draws, 2, sd),
    q_lower = q_lower,
    q_upper = q_upper
  )
  names(summary_df)[4:5] <- paste0(c("q", "q"), c(alpha * 100, (1 - alpha) * 100))

  structure(list(
    summary = summary_df,
    n_obs = object$n_obs,
    iter = object$iter,
    warmup = object$warmup,
    avg_slice_evals = object$avg_slice_evals,
    prob = prob
  ), class = "summary.ratiod_ess")
}


#' @rdname summary.ratiod_fit
#' @return The input `x`, invisibly.
#' @export
print.summary.ratiod_ess <- function(x, digits = 3, ...) {
  cat("Elliptical Slice Sampling Summary\n")
  cat("Observations:", x$n_obs, "\n")
  cat("Iterations:", x$iter, "(warmup:", x$warmup, ")\n")
  cat("Avg slice evals:", round(x$avg_slice_evals, 2), "\n\n")

  cat("Posterior summary (", x$prob * 100, "% credible intervals):\n", sep = "")
  print(x$summary, digits = digits, row.names = FALSE)

  invisible(x)
}

