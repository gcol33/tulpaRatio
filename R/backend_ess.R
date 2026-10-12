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
#' @param chains Number of chains; chain c is seeded with `seed + c - 1`
#' @param seed Random seed
#' @param verbose Print progress
#'
#' @return A `ratiod_fit` whose draws are unpacked to the natural scale
#'   exactly as an HMC fit's are, with `backend = "ess"`.
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
                    chains = 1,
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

  if ((hmc_data$n_re_terms %||% 0L) > 1L || isTRUE(hmc_data$has_slopes)) {
    stop("The ESS backend fits one random intercept. Use `mode = \"hmc\"` ",
         "for random slopes or several grouping factors.", call. = FALSE)
  }

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

  # The sampler starts every chain at zero on the layout it computes
  q_init <- numeric(0)

  if (verbose) {
    message("Elliptical Slice Sampling")
    message("  Iterations: ", iter, " (warmup: ", warmup, "), chains: ", chains)
  }

  runs <- lapply(seq_len(chains), function(c) {
    cpp_ess_fit(
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
      seed = as.integer(seed + c - 1L),
      verbose = verbose
    )
  })

  # ESS samples the parameter vector of the HMC sampler (one layout,
  # compute_param_layout()), with random effects in the centred coordinate,
  # so its draws go through the same unpacking. It has no trajectories: no
  # divergences, tree depth or energy.
  per_chain <- function(field) {
    out <- lapply(runs, function(r) r[[field]])
    if (chains == 1L) out[[1L]] else out
  }
  n_saved <- nrow(runs[[1L]]$samples)
  fill <- function(value) {
    out <- rep(list(rep(value, n_saved)), chains)
    if (chains == 1L) out[[1L]] else out
  }
  fit_raw <- list(
    samples = per_chain("samples"),
    log_prob = per_chain("log_lik"),
    accept_prob = fill(1),
    n_leapfrog = fill(0L),
    treedepth = fill(0L),
    divergent = fill(FALSE),
    epsilon = rep(NA_real_, chains),
    sampler = "ESS"
  )

  fit <- convert_hmc_to_ratiod_fit_full(
    fit_raw = fit_raw,
    hmc_data = hmc_data,
    spatial_info = spatial_info,
    temporal_info = temporal_info,
    zi_info = zi_info,
    formula = formula,
    data = data,
    family = family,
    model_type = model_type,
    iter = iter,
    warmup = warmup,
    chains = chains,
    re_param = "centered"
  )
  fit$backend <- "ess"
  fit$seed <- seed
  fit$diagnostics$avg_slice_evals <- mean(vapply(runs, function(r) r$avg_slice_evals, numeric(1)))

  if (verbose) {
    message("ESS complete. Avg slice evaluations per step: ",
            round(fit$diagnostics$avg_slice_evals, 2))
  }

  fit
}
