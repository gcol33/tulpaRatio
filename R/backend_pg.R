#' Polya-Gamma Backend for Binomial Models
#'
#' @description
#' Fast Gibbs sampling for binomial models using Polya-Gamma data augmentation.
#' This backend provides efficient inference for `ratiod_binomial()` family models
#' with random effects and spatial structure (ICAR or BYM2).
#'
#' @details
#' The Polya-Gamma method (Polson, Scott & Windle, 2013) enables efficient
#' Gibbs sampling for binomial regression by introducing auxiliary variables
#' that make the full conditionals Gaussian.
#'
#' **Supported model structures:**
#' - Fixed effects (intercept, covariates)
#' - Group-level random intercepts
#' - ICAR (Intrinsic CAR) spatial effects
#' - BYM2 spatial effects (structured + unstructured components)
#'
#' **When to use this backend:**
#' - Binomial family models with moderate to large data

#' - When Stan sampling is too slow
#' - When you need fast approximate inference
#'
#' @references
#' Polson, N.G., Scott, J.G., and Windle, J. (2013).
#' "Bayesian Inference for Logistic Models Using Polya-Gamma Latent Variables."
#' Journal of the American Statistical Association, 108(504), 1339-1349.
#'
#' @name pg_backend
#' @keywords internal
NULL


#' Fit binomial model using Polya-Gamma Gibbs sampling
#'
#' @param formula A ratiod_formula object
#' @param data Data frame
#' @param family Must be ratiod_binomial()
#' @param spatial Optional spatial structure from spatial_car() or spatial_bym2()
#' @param iter Total iterations per chain
#' @param warmup Warmup iterations per chain
#' @param chains Number of chains to run
#' @param thin Thinning interval
#' @param cores Number of cores for within-chain parallelism
#' @param seed Random seed for reproducibility
#' @param prior_beta_sd Prior SD for fixed effects (default 10)
#' @param prior_sigma_scale Half-Cauchy scale for RE SD (default 2.5)
#' @param prior_tau_shape Gamma shape for spatial precision (default 1)
#' @param prior_tau_rate Gamma rate for spatial precision (default 0.01)
#' @param verbose Print progress
#'
#' @return A ratiod_fit object
#' @keywords internal
fit_pg_binomial <- function(formula,
                            data,
                            family,
                            spatial = NULL,
                            temporal = NULL,
                            iter = 2000,
                            warmup = floor(iter / 2),
                            chains = 4,
                            thin = 1,
                            cores = NULL,
                            seed = NULL,
                            prior_beta_sd = 10,
                            prior_sigma_scale = 2.5,
                            prior_tau_shape = 1,
                            prior_tau_rate = 0.01,
                            verbose = TRUE) {

  assert_backend_fits_structures(
    "pg",
    list(spatial = spatial, temporal = temporal)
  )

  # Set cores for within-chain parallelism
  if (is.null(cores)) {
    cores <- cpp_pg_get_max_threads()
  }

  # Set base seed for reproducibility
  if (!is.null(seed)) {
    set.seed(seed)
  }

  # Extract response (numerator = successes, denominator = trials)
  y <- formula$numerator$response
  trials <- formula$denominator$response

  if (is.null(trials)) {
    stop("Binomial models require trials (denominator). ",
         "Use formula: successes | trials ~ predictors")
  }

  # Use the design matrix already built by ratiod_formula
  X <- formula$numerator$X

  # Extract random effects structure
  re_info <- extract_re_from_data(formula, data)

  # Check for spatial structure
  has_spatial <- !is.null(spatial)

  if (has_spatial && !is.null(spatial$type) && spatial$type == "car_proper") {
    stop("Proper CAR (`spatial_car(..., proper = TRUE)`) is not supported by ",
         "the PG backend. Use `mode = \"hmc\"`.", call. = FALSE)
  }

  # Dispatch to GP if spatial is GP type
  if (has_spatial && (spatial$type == "gp" || inherits(spatial, "ratiod_gp"))) {
    return(fit_pg_binomial_gp(
      formula = formula,
      data = data,
      family = family,
      spatial = spatial,
      iter = iter,
      warmup = warmup,
      chains = chains,
      thin = thin,
      cores = cores,
      seed = seed,
      verbose = verbose
    ))
  }

  # Dispatch to RSR if spatial is RSR type
  if (has_spatial && inherits(spatial, "ratiod_rsr")) {
    return(fit_pg_binomial_rsr(
      formula = formula,
      data = data,
      family = family,
      spatial = spatial,
      iter = iter,
      warmup = warmup,
      chains = chains,
      thin = thin,
      cores = cores,
      seed = seed,
      prior_beta_sd = prior_beta_sd,
      prior_sigma_scale = prior_sigma_scale,
      prior_tau_shape = prior_tau_shape,
      prior_tau_rate = prior_tau_rate,
      verbose = verbose
    ))
  }

  # Check for temporal structure and dispatch to multiscale temporal if applicable
  has_temporal <- !is.null(temporal)
  if (has_temporal && inherits(temporal, "ratiod_temporal_multiscale")) {
    return(fit_pg_binomial_temporal(
      formula = formula,
      data = data,
      family = family,
      temporal = temporal,
      iter = iter,
      warmup = warmup,
      chains = chains,
      thin = thin,
      cores = cores,
      seed = seed,
      verbose = verbose
    ))
  }

  # Prepare arguments
  n_iter <- as.integer(iter)
  n_warmup <- as.integer(warmup)

  # Prepare spatial info if needed (do once, not per chain)
  spatial_info <- NULL
  is_bym2 <- FALSE
  if (has_spatial) {
    spatial_info <- prepare_spatial_for_pg(spatial, data, formula)
    is_bym2 <- !is.null(spatial$type) && tolower(spatial$type) == "bym2"
  }

  # Run multiple chains sequentially (R's RNG is not thread-safe)
  chain_results <- vector("list", chains)

  for (chain in seq_len(chains)) {
    if (verbose && chains > 1) {
      cat(sprintf("\n=== Chain %d/%d ===\n", chain, chains))
    }

    # Set different seed for each chain
    if (!is.null(seed)) {
      set.seed(seed + chain - 1)
    }

    if (has_spatial) {
      if (is_bym2) {
        # BYM2 spatial model
        scale_factor <- spatial$scale_factor %||% 1.0
        prior_rho_alpha <- spatial$prior_rho_alpha %||% 0.5
        prior_rho_beta <- spatial$prior_rho_beta %||% 0.5

        chain_results[[chain]] <- cpp_pg_binomial_gibbs_bym2(
          y = as.integer(y),
          n = as.integer(trials),
          X = X,
          re_group = re_info$group_idx,
          n_re_groups = re_info$n_groups,
          spatial_group = spatial_info$group_idx,
          n_spatial_units = spatial_info$n_units,
          adj_list = spatial_info$adj_list,
          n_neighbors = spatial_info$n_neighbors,
          scale_factor = scale_factor,
          n_iter = n_iter,
          n_warmup = n_warmup,
          thin = as.integer(thin),
          prior_beta_sd = prior_beta_sd,
          prior_sigma_re_scale = prior_sigma_scale,
          prior_sigma_spatial_scale = prior_sigma_scale,
          prior_rho_alpha = prior_rho_alpha,
          prior_rho_beta = prior_rho_beta,
          store_eta = TRUE,
          verbose = verbose,
          n_threads = as.integer(cores)
        )
      } else {
        # ICAR spatial model
        chain_results[[chain]] <- cpp_pg_binomial_gibbs_spatial(
          y = as.integer(y),
          n = as.integer(trials),
          X = X,
          re_group = re_info$group_idx,
          n_re_groups = re_info$n_groups,
          spatial_group = spatial_info$group_idx,
          n_spatial_units = spatial_info$n_units,
          adj_list = spatial_info$adj_list,
          n_neighbors = spatial_info$n_neighbors,
          n_iter = n_iter,
          n_warmup = n_warmup,
          thin = as.integer(thin),
          prior_beta_sd = prior_beta_sd,
          prior_sigma_re_scale = prior_sigma_scale,
          prior_tau_shape = prior_tau_shape,
          prior_tau_rate = prior_tau_rate,
          store_eta = TRUE,
          verbose = verbose,
          n_threads = as.integer(cores)
        )
      }

    } else if (re_info$n_groups > 0) {
      # Fit with random effects only
      chain_results[[chain]] <- cpp_tulpa_pg_binomial_gibbs(
        y = as.integer(y),
        n = as.integer(trials),
        X = X,
        group = re_info$group_idx,
        n_groups = re_info$n_groups,
        n_iter = n_iter,
        n_warmup = n_warmup,
        thin = as.integer(thin),
        prior_beta_sd = prior_beta_sd,
        prior_sigma_scale = prior_sigma_scale,
        store_eta = TRUE,
        verbose = verbose,
        n_threads = as.integer(cores)
      )

    } else {
      # No random effects - add dummy group
      chain_results[[chain]] <- cpp_tulpa_pg_binomial_gibbs(
        y = as.integer(y),
        n = as.integer(trials),
        X = X,
        group = as.integer(rep(1L, length(y))),
        n_groups = 0L,
        n_iter = n_iter,
        n_warmup = n_warmup,
        thin = as.integer(thin),
        prior_beta_sd = prior_beta_sd,
        prior_sigma_scale = prior_sigma_scale,
        store_eta = TRUE,
        verbose = verbose,
        n_threads = as.integer(cores)
      )
    }
  }

  # Convert to ratiod_fit format with multi-chain support
  fit <- convert_pg_to_ratiod_fit(
    fit_raw = chain_results,
    formula = formula,
    data = data,
    family = family,
    spatial = spatial,
    X = X,
    re_info = re_info,
    iter = iter,
    warmup = warmup,
    thin = thin,
    chains = chains
  )

  return(fit)
}


#' Extract random effects information from data
#'
#' @description
#' Extracts random effects information for the PG backend.
#' Supports multiple crossed random effects (e.g., `(1|site) + (1|year)`).
#'
#' @keywords internal
extract_re_from_data <- function(formula, data) {
  # Check if there are random effects in the formula
  # ratiod_formula stores RE in $numerator$random_effects
  re_terms <- formula$numerator$random_effects
  n_obs <- nrow(data)

  if (is.null(re_terms) || length(re_terms) == 0) {
    return(list(
      group_idx = as.integer(rep(1L, n_obs)),
      n_groups = 0L,
      group_var = NULL,
      group_levels = NULL,
      # Multi-term fields
      n_re_terms = 0L,
      re_terms = list(),
      has_slopes = FALSE
    ))
  }

  n_re_terms <- length(re_terms)

  # Check for unsupported features and warn
  has_slopes <- FALSE
  for (term in re_terms) {
    if (length(term$slope_vars) > 0) {
      has_slopes <- TRUE
      warning(
        "Random slopes not yet fully supported in PG backend. Slope terms will be ignored: ",
        paste(term$slope_vars, collapse = ", "),
        "\nOnly random intercepts (1 | group) are currently implemented.",
        call. = FALSE
      )
    }
  }

  if (n_re_terms > 1) {
    # Multiple RE terms
    re_terms_processed <- vector("list", n_re_terms)
    total_groups <- 0L

    for (t in seq_len(n_re_terms)) {
      term <- re_terms[[t]]
      re_terms_processed[[t]] <- list(
        group_var = term$group_var,
        group_idx = as.integer(term$group),
        n_groups = as.integer(term$n_groups),
        offset = total_groups
      )
      total_groups <- total_groups + term$n_groups
    }

    # Build group index matrix
    group_idx_matrix <- matrix(0L, nrow = n_obs, ncol = n_re_terms)
    for (t in seq_len(n_re_terms)) {
      group_idx_matrix[, t] <- as.integer(re_terms[[t]]$group)
    }

    return(list(
      # Legacy fields (for backwards compatibility)
      group_idx = as.integer(re_terms[[1]]$group),
      n_groups = as.integer(re_terms[[1]]$n_groups),
      group_var = re_terms[[1]]$group_var,
      group_levels = NULL,
      # Multi-term fields
      n_re_terms = n_re_terms,
      re_terms = re_terms_processed,
      group_idx_matrix = group_idx_matrix,
      total_groups = total_groups,
      has_slopes = has_slopes
    ))
  }

  # Single RE term
  re_info <- re_terms[[1]]
  re_var <- re_info$group_var
  group_idx <- as.integer(re_info$group)
  n_groups <- re_info$n_groups

  return(list(
    group_idx = group_idx,
    n_groups = n_groups,
    group_var = re_var,
    group_levels = NULL,
    # Multi-term fields
    n_re_terms = 1L,
    re_terms = list(list(
      group_var = re_var,
      group_idx = group_idx,
      n_groups = as.integer(n_groups),
      offset = 0L
    )),
    total_groups = as.integer(n_groups),
    has_slopes = has_slopes
  ))
}


#' Prepare spatial structure for PG backend
#'
#' @keywords internal
prepare_spatial_for_pg <- function(spatial, data, formula) {
  # Get spatial group variable
  if (spatial$level == "group") {
    spatial_var <- spatial$group_var
    if (is.null(spatial_var)) {
      stop("spatial_car() with level='group' requires group_var")
    }
  } else {
    # Observation level - use row indices
    spatial_var <- NULL
  }

  if (!is.null(spatial_var)) {
    group_factor <- as.factor(data[[spatial_var]])
    group_idx <- as.integer(group_factor)
    n_units <- nlevels(group_factor)
  } else {
    # Observation level
    group_idx <- seq_len(nrow(data))
    n_units <- nrow(data)
  }

  # Convert adjacency matrix to list format
  adj_matrix <- spatial$adjacency
  adj_list <- vector("list", n_units)
  n_neighbors <- integer(n_units)

  for (i in seq_len(n_units)) {
    neighbors <- which(adj_matrix[i, ] > 0)
    adj_list[[i]] <- as.integer(neighbors)
    n_neighbors[i] <- length(neighbors)
  }

  return(list(
    group_idx = as.integer(group_idx),
    n_units = as.integer(n_units),
    adj_list = adj_list,
    n_neighbors = as.integer(n_neighbors)
  ))
}


#' Convert PG fit to ratiod_fit object
#'
#' @param fit_raw List of chain results (each element is a single chain's output)
#' @param chains Number of chains
#' @keywords internal
convert_pg_to_ratiod_fit <- function(fit_raw, formula, data, family,
                                    spatial, X, re_info, iter, warmup, thin,
                                    chains = 1) {

  # fit_raw is now a list of chain results
  n_chains <- length(fit_raw)
  n_save_per_chain <- nrow(fit_raw[[1]]$beta)
  n_save_total <- n_save_per_chain * n_chains

  # Create draws in format compatible with posterior package
  beta_names <- colnames(X)
  if (is.null(beta_names)) {
    beta_names <- paste0("beta[", seq_len(ncol(X)), "]")
  }

  # Determine if spatial model
  is_bym2 <- !is.null(spatial) && !is.null(spatial$type) &&
             tolower(spatial$type) == "bym2"

  # Build draws array with chain dimension: [iteration, chain, parameter]
  # For compatibility with posterior package
  n_params <- ncol(fit_raw[[1]]$beta)
  if (re_info$n_groups > 0) n_params <- n_params + 1  # sigma_re
  if (!is.null(spatial)) {
    if (is_bym2) {
      n_params <- n_params + 2  # sigma_spatial, rho
    } else {
      n_params <- n_params + 1  # tau_spatial
    }
  }

  # Create 3D array for draws: [iteration, chain, variable]
  draws_array <- array(NA_real_,
                       dim = c(n_save_per_chain, n_chains, n_params))

  # Build parameter names
  param_names <- beta_names
  if (re_info$n_groups > 0) {
    param_names <- c(param_names, "sigma_re")
  }
  if (!is.null(spatial)) {
    if (is_bym2) {
      param_names <- c(param_names, "sigma_spatial", "rho")
    } else {
      param_names <- c(param_names, "tau_spatial")
    }
  }
  dimnames(draws_array) <- list(
    iteration = seq_len(n_save_per_chain),
    chain = seq_len(n_chains),
    variable = param_names
  )

  # Fill in the draws from each chain
  for (c in seq_len(n_chains)) {
    chain_fit <- fit_raw[[c]]
    col_idx <- 1

    # Fixed effects
    for (j in seq_len(ncol(chain_fit$beta))) {
      draws_array[, c, col_idx] <- chain_fit$beta[, j]
      col_idx <- col_idx + 1
    }

    # Random effects SD
    if (re_info$n_groups > 0) {
      draws_array[, c, col_idx] <- chain_fit$sigma_re
      col_idx <- col_idx + 1
    }

    # Spatial parameters
    if (!is.null(spatial)) {
      if (is_bym2) {
        draws_array[, c, col_idx] <- chain_fit$sigma_spatial
        col_idx <- col_idx + 1
        draws_array[, c, col_idx] <- chain_fit$rho
        col_idx <- col_idx + 1
      } else {
        draws_array[, c, col_idx] <- chain_fit$tau
        col_idx <- col_idx + 1
      }
    }
  }

  # Also create combined eta array for ratio computation: [iteration, chain, obs]
  n_obs <- ncol(fit_raw[[1]]$eta)
  eta_array <- array(NA_real_,
                     dim = c(n_save_per_chain, n_chains, n_obs))
  for (c in seq_len(n_chains)) {
    eta_array[, c, ] <- fit_raw[[c]]$eta
  }

  # Combine internal components from all chains for backwards compatibility
  # Stack matrices row-wise for methods that expect 2D format
  combined_beta <- do.call(rbind, lapply(fit_raw, `[[`, "beta"))
  combined_re <- do.call(rbind, lapply(fit_raw, `[[`, "re"))
  combined_eta <- do.call(rbind, lapply(fit_raw, `[[`, "eta"))
  combined_spatial <- if (!is.null(fit_raw[[1]]$spatial)) {
    do.call(rbind, lapply(fit_raw, `[[`, "spatial"))
  } else NULL

  # Create ratiod_fit object
  fit <- list(
    draws = draws_array,
    formula = formula,
    data = data,
    family = family,
    spatial = spatial,
    backend = "pg",
    iter = iter,
    warmup = warmup,
    thin = thin,
    chains = n_chains,
    n_save = n_save_total,
    n_save_per_chain = n_save_per_chain,
    # Store raw components for ratio computation
    .internal = list(
      eta = combined_eta,
      eta_array = eta_array,  # 3D array version
      beta = combined_beta,
      re = combined_re,
      spatial = combined_spatial,
      phi_scaled = if (!is.null(fit_raw[[1]]$phi_scaled)) {
        do.call(rbind, lapply(fit_raw, `[[`, "phi_scaled"))
      } else NULL,
      theta = if (!is.null(fit_raw[[1]]$theta)) {
        do.call(rbind, lapply(fit_raw, `[[`, "theta"))
      } else NULL,
      X = X,
      re_info = re_info,
      chain_results = fit_raw  # Keep original chain results for diagnostics
    )
  )

  class(fit) <- "ratiod_fit"
  return(fit)
}


#' Check if PG backend is appropriate
#'
#' @param family A tulpaRatio family object
#' @return Logical indicating if PG backend can be used
#' @keywords internal
can_use_pg_backend <- function(family) {
  # PG backend supports:
  # 1. Binomial family (standard PG augmentation)
  # 2. Negative binomial family (PG + CRT augmentation)
  dist <- family$numerator$distribution

  # Binomial family
  if (isTRUE(dist == "binomial")) return(TRUE)

  # Negative binomial family (NB2 parameterization)
  if (isTRUE(dist == "neg_binomial_2")) return(TRUE)

  FALSE
}


#' Fit binomial model with GP spatial using Polya-Gamma Gibbs sampling
#'
#' @param formula A ratiod_formula object
#' @param data Data frame
#' @param family Must be ratiod_binomial()
#' @param spatial GP spatial structure from spatial_gp()
#' @param iter Total iterations per chain
#' @param warmup Warmup iterations per chain
#' @param chains Number of chains to run
#' @param thin Thinning interval
#' @param cores Number of cores for within-chain parallelism
#' @param seed Random seed for reproducibility
#' @param verbose Print progress
#'
#' @return A ratiod_fit object
#' @keywords internal
fit_pg_binomial_gp <- function(formula,
                                data,
                                family,
                                spatial,
                                iter = 2000,
                                warmup = floor(iter / 2),
                                chains = 4,
                                thin = 1,
                                cores = NULL,
                                seed = NULL,
                                verbose = TRUE) {

  # Set cores for within-chain parallelism
  if (is.null(cores)) {
    cores <- cpp_pg_get_max_threads()
  }

  # Set base seed for reproducibility
  if (!is.null(seed)) {
    set.seed(seed)
  }

  # Extract response
  y <- formula$numerator$response
  trials <- formula$denominator$response

  if (is.null(trials)) {
    stop("Binomial models require trials (denominator).")
  }

  # Design matrix
  X <- formula$numerator$X

  # Random effects
  re_info <- extract_re_from_data(formula, data)

  # Validate GP specification
  spatial <- validate_gp(spatial, data)

  if (verbose) {
    message("Fitting GP spatial model with Polya-Gamma Gibbs sampling...")
    message(sprintf("  Observations: %d", spatial$n_obs))
    message(sprintf("  Neighbors: %d", spatial$nn))
    message(sprintf("  Covariance: %s", spatial$cov))
  }

  cov_type <- cov_type_code(spatial$cov)

  # Get neighbor info
  nn_info <- spatial$neighbor_info

  # Initial hyperparameters
  sigma2_gp_init <- spatial$sigma2 %||% 1.0
  phi_gp_init <- spatial$phi %||% 1.0

  # Prepare arguments
  n_iter <- as.integer(iter)
  n_warmup <- as.integer(warmup)

  # Run chains
  chain_results <- vector("list", chains)

  for (chain in seq_len(chains)) {
    if (verbose && chains > 1) {
      cat(sprintf("\n=== Chain %d/%d ===\n", chain, chains))
    }

    if (!is.null(seed)) {
      set.seed(seed + chain - 1)
    }

    chain_results[[chain]] <- cpp_pg_binomial_gibbs_gp(
      y = as.integer(y),
      n = as.integer(trials),
      X = X,
      re_group = re_info$group_idx,
      n_re_groups = re_info$n_groups,
      # The unique locations, which is what the neighbour structure was built
      # on and what nn_order indexes. coords_matrix is the observation-order
      # matrix, one row per row of data.
      coords = spatial$unique_coords,
      nn_idx = nn_info$nn_idx,
      nn_dist = nn_info$nn_dist,
      # 0-based: the C++ uses these values directly as indices into w, coords
      # and the per-location likelihood accumulators, all sized n_spatial.
      # backend_hmc.R converts the same vector the same way.
      nn_order = as.integer(nn_info$nn_order - 1L),
      # 0-based, one entry per observation: which location it was measured at.
      # Without it the sampler can only pair the two by row position.
      obs_to_loc = as.integer(spatial$obs_to_loc - 1L),
      n_spatial = spatial$n_spatial,
      nn = spatial$nn,
      sigma2_gp_init = sigma2_gp_init,
      phi_gp_init = phi_gp_init,
      cov_type = cov_type,
      n_iter = n_iter,
      n_warmup = n_warmup,
      thin = as.integer(thin),
      store_eta = TRUE,
      verbose = verbose,
      n_threads = as.integer(cores)
    )
  }

  # Convert to ratiod_fit format
  fit <- convert_pg_gp_to_ratiod_fit(
    fit_raw = chain_results,
    formula = formula,
    data = data,
    family = family,
    spatial = spatial,
    X = X,
    re_info = re_info,
    iter = iter,
    warmup = warmup,
    thin = thin,
    chains = chains
  )

  return(fit)
}


#' Convert PG GP fit to ratiod_fit object
#' @keywords internal
convert_pg_gp_to_ratiod_fit <- function(fit_raw, formula, data, family,
                                         spatial, X, re_info, iter, warmup, thin,
                                         chains = 1) {

  n_chains <- length(fit_raw)
  n_save_per_chain <- nrow(fit_raw[[1]]$beta)
  n_save_total <- n_save_per_chain * n_chains

  # Create draws in format compatible with posterior package
  beta_names <- colnames(X)
  if (is.null(beta_names)) {
    beta_names <- paste0("beta[", seq_len(ncol(X)), "]")
  }

  # Count parameters
  n_params <- ncol(fit_raw[[1]]$beta) + 2  # beta + sigma2_gp + phi_gp
  if (re_info$n_groups > 0) n_params <- n_params + 1  # sigma_re

  # Create 3D array for draws: [iteration, chain, variable]
  draws_array <- array(NA_real_,
                       dim = c(n_save_per_chain, n_chains, n_params))

  # Build parameter names
  param_names <- c(beta_names, "sigma2_gp", "phi_gp")
  if (re_info$n_groups > 0) {
    param_names <- c(param_names, "sigma_re")
  }

  dimnames(draws_array) <- list(
    iteration = seq_len(n_save_per_chain),
    chain = seq_len(n_chains),
    variable = param_names
  )

  # Fill in the draws from each chain
  for (c in seq_len(n_chains)) {
    chain_fit <- fit_raw[[c]]
    col_idx <- 1

    # Fixed effects
    for (j in seq_len(ncol(chain_fit$beta))) {
      draws_array[, c, col_idx] <- chain_fit$beta[, j]
      col_idx <- col_idx + 1
    }

    # GP hyperparameters
    draws_array[, c, col_idx] <- chain_fit$sigma2_gp
    col_idx <- col_idx + 1
    draws_array[, c, col_idx] <- chain_fit$phi_gp
    col_idx <- col_idx + 1

    # Random effects SD
    if (re_info$n_groups > 0) {
      draws_array[, c, col_idx] <- chain_fit$sigma_re
      col_idx <- col_idx + 1
    }
  }

  # Combine internal components
  combined_beta <- do.call(rbind, lapply(fit_raw, `[[`, "beta"))
  combined_re <- do.call(rbind, lapply(fit_raw, `[[`, "re"))
  combined_eta <- do.call(rbind, lapply(fit_raw, `[[`, "eta"))
  # The field draws, one column per unique LOCATION, stacked in the same chain
  # order as the fixed effects beside them. The per-chain hyperparameters are
  # stacked the same way, so a draw of the field and the (sigma2, phi) it was
  # drawn under share a row -- what kriging it onto new coordinates needs.
  combined_w_gp <- do.call(rbind, lapply(fit_raw, `[[`, "gp"))
  combined_gp_hyper <- data.frame(
    sigma2 = unlist(lapply(fit_raw, `[[`, "sigma2_gp"), use.names = FALSE),
    phi = unlist(lapply(fit_raw, `[[`, "phi_gp"), use.names = FALSE)
  )

  fit <- list(
    draws = draws_array,
    formula = formula,
    data = data,
    family = family,
    spatial = spatial,
    spatial_type = "gp",
    backend = "pg",
    iter = iter,
    warmup = warmup,
    thin = thin,
    chains = n_chains,
    n_save = n_save_total,
    n_save_per_chain = n_save_per_chain,
    .internal = list(
      eta = combined_eta,
      beta = combined_beta,
      re = combined_re,
      w_gp = combined_w_gp,
      gp_hyper = combined_gp_hyper,
      gp_coords = spatial$unique_coords,
      gp_obs_to_loc = spatial$obs_to_loc,
      X = X,
      re_info = re_info,
      chain_results = fit_raw
    )
  )

  class(fit) <- "ratiod_fit"
  return(fit)
}


#' Fit binomial model with multiscale temporal using Polya-Gamma Gibbs sampling
#'
#' @param formula A ratiod_formula object
#' @param data Data frame
#' @param family Must be ratiod_binomial()
#' @param temporal Multiscale temporal structure from temporal_multiscale()
#' @param iter Total iterations per chain
#' @param warmup Warmup iterations per chain
#' @param chains Number of chains to run
#' @param thin Thinning interval
#' @param cores Number of cores for within-chain parallelism
#' @param seed Random seed for reproducibility
#' @param verbose Print progress
#'
#' @return A ratiod_fit object
#' @keywords internal
fit_pg_binomial_temporal <- function(formula,
                                      data,
                                      family,
                                      temporal,
                                      iter = 2000,
                                      warmup = floor(iter / 2),
                                      chains = 4,
                                      thin = 1,
                                      cores = NULL,
                                      seed = NULL,
                                      verbose = TRUE) {

  # Set cores for within-chain parallelism
  if (is.null(cores)) {
    cores <- cpp_pg_get_max_threads()
  }

  # Set base seed for reproducibility
  if (!is.null(seed)) {
    set.seed(seed)
  }

  # Extract response
  y <- formula$numerator$response
  trials <- formula$denominator$response

  if (is.null(trials)) {
    stop("Binomial models require trials (denominator).")
  }

  # Design matrix
  X <- formula$numerator$X

  # Random effects
  re_info <- extract_re_from_data(formula, data)

  # Validate temporal specification
  temporal <- validate_temporal_multiscale(temporal, data)

  if (verbose) {
    message("Fitting multiscale temporal model with Polya-Gamma Gibbs sampling...")
    message(sprintf("  Time points: %d", temporal$n_times))
    message(sprintf("  Components: %s", paste(temporal$components, collapse = ", ")))
  }

  # Map temporal types to C++ integers
  trend_type <- switch(temporal$trend,
    "none" = 0L,
    "rw1" = 1L,
    "rw2" = 2L,
    0L
  )

  short_type <- switch(temporal$short_term,
    "none" = 0L,
    "ar1" = 1L,
    "iid" = 2L,
    0L
  )

  seasonal_period <- temporal$seasonal %||% 0L

  # Prepare arguments
  n_iter <- as.integer(iter)
  n_warmup <- as.integer(warmup)

  # Run chains
  chain_results <- vector("list", chains)

  for (chain in seq_len(chains)) {
    if (verbose && chains > 1) {
      cat(sprintf("\n=== Chain %d/%d ===\n", chain, chains))
    }

    if (!is.null(seed)) {
      set.seed(seed + chain - 1)
    }

    chain_results[[chain]] <- cpp_pg_binomial_gibbs_temporal(
      y = as.integer(y),
      n = as.integer(trials),
      X = X,
      re_group = re_info$group_idx,
      n_re_groups = re_info$n_groups,
      time_idx = temporal$time_index,
      n_times = temporal$n_times,
      seasonal_period = as.integer(seasonal_period),
      trend_type = trend_type,
      short_type = short_type,
      n_iter = n_iter,
      n_warmup = n_warmup,
      thin = as.integer(thin),
      store_eta = TRUE,
      verbose = verbose,
      n_threads = as.integer(cores)
    )
  }

  # Convert to ratiod_fit format
  fit <- convert_pg_temporal_to_ratiod_fit(
    fit_raw = chain_results,
    formula = formula,
    data = data,
    family = family,
    temporal = temporal,
    X = X,
    re_info = re_info,
    iter = iter,
    warmup = warmup,
    thin = thin,
    chains = chains
  )

  return(fit)
}


#' Convert PG temporal fit to ratiod_fit object
#' @keywords internal
convert_pg_temporal_to_ratiod_fit <- function(fit_raw, formula, data, family,
                                               temporal, X, re_info, iter, warmup, thin,
                                               chains = 1) {

  n_chains <- length(fit_raw)
  n_save_per_chain <- nrow(fit_raw[[1]]$beta)
  n_save_total <- n_save_per_chain * n_chains

  # Create draws in format compatible with posterior package
  beta_names <- colnames(X)
  if (is.null(beta_names)) {
    beta_names <- paste0("beta[", seq_len(ncol(X)), "]")
  }

  # Count hyperparameters
  n_hyper <- 0
  hyper_names <- character(0)

  if (temporal$trend != "none") {
    n_hyper <- n_hyper + 1
    hyper_names <- c(hyper_names, "sigma2_trend")
  }
  if (!is.null(temporal$seasonal) && temporal$seasonal > 0) {
    n_hyper <- n_hyper + 1
    hyper_names <- c(hyper_names, "sigma2_seasonal")
  }
  if (temporal$short_term != "none") {
    n_hyper <- n_hyper + 1
    hyper_names <- c(hyper_names, "sigma2_short")
    if (temporal$short_term == "ar1") {
      n_hyper <- n_hyper + 1
      hyper_names <- c(hyper_names, "rho_short")
    }
  }

  n_params <- ncol(fit_raw[[1]]$beta) + n_hyper
  if (re_info$n_groups > 0) n_params <- n_params + 1

  # Create 3D array for draws
  draws_array <- array(NA_real_,
                       dim = c(n_save_per_chain, n_chains, n_params))

  # Build parameter names
  param_names <- c(beta_names, hyper_names)
  if (re_info$n_groups > 0) {
    param_names <- c(param_names, "sigma_re")
  }

  dimnames(draws_array) <- list(
    iteration = seq_len(n_save_per_chain),
    chain = seq_len(n_chains),
    variable = param_names
  )

  # Fill in the draws from each chain
  for (c in seq_len(n_chains)) {
    chain_fit <- fit_raw[[c]]
    col_idx <- 1

    # Fixed effects
    for (j in seq_len(ncol(chain_fit$beta))) {
      draws_array[, c, col_idx] <- chain_fit$beta[, j]
      col_idx <- col_idx + 1
    }

    # Temporal hyperparameters
    if (temporal$trend != "none" && !is.null(chain_fit$sigma2_trend)) {
      draws_array[, c, col_idx] <- chain_fit$sigma2_trend
      col_idx <- col_idx + 1
    }
    if (!is.null(temporal$seasonal) && temporal$seasonal > 0 && !is.null(chain_fit$sigma2_seasonal)) {
      draws_array[, c, col_idx] <- chain_fit$sigma2_seasonal
      col_idx <- col_idx + 1
    }
    if (temporal$short_term != "none" && !is.null(chain_fit$sigma2_short)) {
      draws_array[, c, col_idx] <- chain_fit$sigma2_short
      col_idx <- col_idx + 1
      if (temporal$short_term == "ar1" && !is.null(chain_fit$rho_short)) {
        draws_array[, c, col_idx] <- chain_fit$rho_short
        col_idx <- col_idx + 1
      }
    }

    # Random effects SD
    if (re_info$n_groups > 0) {
      draws_array[, c, col_idx] <- chain_fit$sigma_re
      col_idx <- col_idx + 1
    }
  }

  # Combine internal components
  combined_beta <- do.call(rbind, lapply(fit_raw, `[[`, "beta"))
  combined_re <- do.call(rbind, lapply(fit_raw, `[[`, "re"))
  combined_eta <- do.call(rbind, lapply(fit_raw, `[[`, "eta"))

  # Extract temporal draws
  temporal_draws <- list()
  if (!is.null(fit_raw[[1]]$trend)) {
    temporal_draws$trend <- do.call(rbind, lapply(fit_raw, `[[`, "trend"))
  }
  if (!is.null(fit_raw[[1]]$seasonal)) {
    temporal_draws$seasonal <- do.call(rbind, lapply(fit_raw, `[[`, "seasonal"))
  }
  if (!is.null(fit_raw[[1]]$short_term)) {
    temporal_draws$short_term <- do.call(rbind, lapply(fit_raw, `[[`, "short_term"))
  }

  fit <- list(
    draws = draws_array,
    formula = formula,
    data = data,
    family = family,
    temporal = temporal,
    backend = "pg",
    iter = iter,
    warmup = warmup,
    thin = thin,
    chains = n_chains,
    n_save = n_save_total,
    n_save_per_chain = n_save_per_chain,
    .internal = list(
      eta = combined_eta,
      beta = combined_beta,
      re = combined_re,
      temporal_draws = temporal_draws,
      X = X,
      re_info = re_info,
      chain_results = fit_raw
    )
  )

  class(fit) <- "ratiod_fit"
  return(fit)
}


#' Fit PG binomial with RSR (Restricted Spatial Regression)
#'
#' @description
#' Polya-Gamma Gibbs sampling with RSR to orthogonalize spatial effects
#' to covariates, preventing spatial confounding.
#'
#' @keywords internal
fit_pg_binomial_rsr <- function(formula,
                                 data,
                                 family,
                                 spatial,
                                 iter = 2000,
                                 warmup = floor(iter / 2),
                                 chains = 4,
                                 thin = 1,
                                 cores = NULL,
                                 seed = NULL,
                                 prior_beta_sd = 10,
                                 prior_sigma_scale = 2.5,
                                 prior_tau_shape = 1,
                                 prior_tau_rate = 0.01,
                                 verbose = TRUE) {

  if (verbose) {
    message("Fitting RSR spatial model with Polya-Gamma Gibbs sampling...")
  }

  # Set cores
  if (is.null(cores)) {
    cores <- cpp_pg_get_max_threads()
  }

  # Extract response and design matrix
  y <- formula$numerator$response
  trials <- formula$denominator$response
  X <- formula$numerator$X

  # Extract RE info
  re_info <- extract_re_from_data(formula, data)

  # Prepare spatial info
  spatial_info <- prepare_spatial_for_pg(spatial, data, formula)

  # The field reaches the observations as A phi, A = (I - Q Q') S.
  Q <- rsr_basis(spatial, data)
  N <- length(spatial_info$group_idx)
  incidence <- matrix(0, N, spatial_info$n_units)
  incidence[cbind(seq_len(N), spatial_info$group_idx)] <- 1
  rsr_design <- incidence - Q %*% crossprod(Q, incidence)
  spatial_info$rsr_basis <- Q

  if (verbose) {
    message(sprintf("  Orthogonal to: %s",
                    paste(all.vars(spatial$rsr_formula), collapse = ", ")))
  }


  n_iter <- as.integer(iter)
  n_warmup <- as.integer(warmup)

  # Run chains
  chain_results <- vector("list", chains)

  for (chain in seq_len(chains)) {
    if (verbose && chains > 1) {
      cat(sprintf("\n=== Chain %d/%d ===\n", chain, chains))
    }

    if (!is.null(seed)) {
      set.seed(seed + chain - 1)
    }

    chain_results[[chain]] <- cpp_pg_binomial_gibbs_rsr(
      y = as.integer(y),
      n = as.integer(trials),
      X = X,
      re_group = re_info$group_idx,
      n_re_groups = re_info$n_groups,
      spatial_group = spatial_info$group_idx,
      n_spatial_units = spatial_info$n_units,
      adj_list = spatial_info$adj_list,
      n_neighbors = spatial_info$n_neighbors,
      rsr_design = rsr_design,
      n_iter = n_iter,
      n_warmup = n_warmup,
      thin = as.integer(thin),
      prior_beta_sd = prior_beta_sd,
      prior_sigma_re_scale = prior_sigma_scale,
      prior_tau_shape = prior_tau_shape,
      prior_tau_rate = prior_tau_rate,
      store_eta = TRUE,
      verbose = verbose,
      n_threads = as.integer(cores)
    )
  }

  # Convert to ratiod_fit
  fit <- convert_pg_rsr_to_ratiod_fit(
    chain_results = chain_results,
    formula = formula,
    data = data,
    family = family,
    X = X,
    re_info = re_info,
    spatial_info = spatial_info,
    chains = chains,
    iter = iter,
    warmup = warmup,
    thin = thin
  )

  return(fit)
}


#' Convert PG RSR results to ratiod_fit
#' @keywords internal
convert_pg_rsr_to_ratiod_fit <- function(chain_results, formula, data, family,
                                          X, re_info, spatial_info,
                                          chains, iter, warmup, thin) {
  n_chains <- length(chain_results)
  p <- ncol(X)
  n_re <- re_info$n_groups
  n_spatial <- spatial_info$n_units
  n_save_per_chain <- nrow(chain_results[[1]]$beta)
  n_save_total <- n_save_per_chain * n_chains

  # Parameter names
  beta_names <- colnames(X)
  if (is.null(beta_names)) {
    beta_names <- paste0("beta[", seq_len(p), "]")
  }

  param_names <- beta_names
  if (n_re > 0) {
    param_names <- c(param_names, "sigma_re")
  }
  if (n_spatial > 0) {
    param_names <- c(param_names, "tau_spatial")
  }

  # Create 3D array: [iterations, chains, parameters]
  n_params <- length(param_names)
  draws_array <- array(dim = c(n_save_per_chain, n_chains, n_params),
                       dimnames = list(NULL, paste0("chain:", seq_len(n_chains)), param_names))

  for (c in seq_len(n_chains)) {
    chain_fit <- chain_results[[c]]

    col_idx <- 1
    # Fixed effects
    for (j in seq_len(p)) {
      draws_array[, c, col_idx] <- chain_fit$beta[, j]
      col_idx <- col_idx + 1
    }
    # Sigma_re
    if (n_re > 0) {
      draws_array[, c, col_idx] <- chain_fit$sigma_re
      col_idx <- col_idx + 1
    }
    # Tau (convert to SD for consistency)
    if (n_spatial > 0) {
      draws_array[, c, col_idx] <- chain_fit$tau
      col_idx <- col_idx + 1
    }
  }

  # Combine spatial draws
  combined_spatial <- do.call(rbind, lapply(chain_results, `[[`, "spatial"))
  combined_spatial_raw <- do.call(rbind, lapply(chain_results, `[[`, "spatial_raw"))
  combined_beta <- do.call(rbind, lapply(chain_results, `[[`, "beta"))
  combined_re <- do.call(rbind, lapply(chain_results, `[[`, "re"))
  combined_eta <- do.call(rbind, lapply(chain_results, `[[`, "eta"))

  fit <- list(
    draws = draws_array,
    formula = formula,
    data = data,
    family = family,
    spatial = spatial_info,
    spatial_type = "rsr",
    backend = "pg",
    iter = iter,
    warmup = warmup,
    thin = thin,
    chains = n_chains,
    n_save = n_save_total,
    n_save_per_chain = n_save_per_chain,
    .internal = list(
      beta = combined_beta,
      re = combined_re,
      eta = combined_eta,
      spatial = combined_spatial,
      spatial_raw = combined_spatial_raw,
      X = X,
      re_info = re_info,
      spatial_info = spatial_info,
      chain_results = chain_results
    )
  )

  class(fit) <- "ratiod_fit"
  return(fit)
}


# =============================================================================
# Negative Binomial PG+CRT Backend
# =============================================================================

#' Fit negative binomial model using Polya-Gamma + CRT Gibbs sampling
#'
#' @description
#' Fast Gibbs sampling for negative binomial models using Polya-Gamma data
#' augmentation for the logit component and Chinese Restaurant Table (CRT)
#' augmentation for the dispersion parameter.
#'
#' @details
#' The PG+CRT method (Zhou et al., 2012) enables efficient conjugate Gibbs
#' sampling for negative binomial regression. This extends the standard PG
#' augmentation to count data with overdispersion.
#'
#' **Model (NB2 parameterization):**
#' - \eqn{Y_i \sim NB(r, p_i)}
#' - \eqn{E[Y_i] = r \cdot p_i / (1 - p_i)}
#' - \eqn{logit(p_i) = X_i \beta + Z_i b}
#'
#' **Augmentation scheme:**
#' - omega_i ~ PG(y_i + r, eta_i) for the logit component
#' - L_i ~ CRT(y_i, r) for the dispersion update
#'
#' @references
#' Zhou, M., Li, L., Dunson, D., and Carin, L. (2012).
#' "Lognormal and Gamma Mixed Negative Binomial Regression." ICML.
#'
#' @name pg_negbin_backend
#' @keywords internal
NULL


#' Fit negative binomial model using PG+CRT Gibbs sampling
#'
#' @param formula A ratiod_formula object
#' @param data Data frame
#' @param family Must be ratiod_negbin_negbin() or similar NB family
#' @param spatial Optional spatial structure
#' @param iter Total iterations per chain
#' @param warmup Warmup iterations per chain
#' @param chains Number of chains
#' @param thin Thinning interval
#' @param cores Number of cores
#' @param seed Random seed
#' @param prior_beta_sd Prior SD for fixed effects
#' @param prior_sigma_scale Half-Cauchy scale for RE SD
#' @param prior_r_shape Gamma shape for dispersion prior
#' @param prior_r_rate Gamma rate for dispersion prior
#' @param r_init Initial dispersion value
#' @param verbose Print progress
#'
#' @return A ratiod_fit object
#' @keywords internal
fit_pg_negbin <- function(formula,
                          data,
                          family,
                          spatial = NULL,
                          iter = 2000,
                          warmup = floor(iter / 2),
                          chains = 4,
                          thin = 1,
                          cores = NULL,
                          seed = NULL,
                          prior_beta_sd = 10,
                          prior_sigma_scale = 2.5,
                          prior_r_shape = 1,
                          prior_r_rate = 0.1,
                          r_init = 1.0,
                          verbose = TRUE) {

  # Set cores
  if (is.null(cores)) {
    cores <- cpp_pg_get_max_threads()
  }

  if (!is.null(seed)) {
    set.seed(seed)
  }

  # Extract response (numerator counts)
  y <- formula$numerator$response

  # Design matrix
  X <- formula$numerator$X

  # Random effects
  re_info <- extract_re_from_data(formula, data)

  # Check for spatial
  has_spatial <- !is.null(spatial)

  if (has_spatial && !is.null(spatial$type) && spatial$type == "car_proper") {
    stop("Proper CAR (`spatial_car(..., proper = TRUE)`) is not supported by ",
         "the PG backend. Use `mode = \"hmc\"`.", call. = FALSE)
  }

  n_iter <- as.integer(iter)
  n_warmup <- as.integer(warmup)

  # Run chains
  chain_results <- vector("list", chains)

  for (chain in seq_len(chains)) {
    if (verbose && chains > 1) {
      cat(sprintf("\n=== Chain %d/%d ===\n", chain, chains))
    }

    if (!is.null(seed)) {
      set.seed(seed + chain - 1)
    }

    if (has_spatial) {
      spatial_info <- prepare_spatial_for_pg(spatial, data, formula)

      chain_results[[chain]] <- cpp_pg_negbin_gibbs_spatial(
        y = as.integer(y),
        X = X,
        re_group = re_info$group_idx,
        n_re_groups = re_info$n_groups,
        spatial_group = spatial_info$group_idx,
        n_spatial_units = spatial_info$n_units,
        adj_list = spatial_info$adj_list,
        n_neighbors = spatial_info$n_neighbors,
        n_iter = n_iter,
        n_warmup = n_warmup,
        thin = as.integer(thin),
        prior_beta_sd = prior_beta_sd,
        prior_sigma_re_scale = prior_sigma_scale,
        prior_tau_shape = 1,
        prior_tau_rate = 0.01,
        prior_r_shape = prior_r_shape,
        prior_r_rate = prior_r_rate,
        r_init = r_init,
        store_eta = TRUE,
        verbose = verbose,
        n_threads = as.integer(cores)
      )
    } else {
      chain_results[[chain]] <- cpp_pg_negbin_gibbs(
        y = as.integer(y),
        X = X,
        group = re_info$group_idx,
        n_groups = re_info$n_groups,
        n_iter = n_iter,
        n_warmup = n_warmup,
        thin = as.integer(thin),
        prior_beta_sd = prior_beta_sd,
        prior_sigma_scale = prior_sigma_scale,
        prior_r_shape = prior_r_shape,
        prior_r_rate = prior_r_rate,
        r_init = r_init,
        store_eta = TRUE,
        verbose = verbose,
        n_threads = as.integer(cores)
      )
    }
  }

  # Convert to ratiod_fit
  fit <- convert_pg_negbin_to_ratiod_fit(
    fit_raw = chain_results,
    formula = formula,
    data = data,
    family = family,
    spatial = spatial,
    X = X,
    re_info = re_info,
    iter = iter,
    warmup = warmup,
    thin = thin,
    chains = chains
  )

  return(fit)
}


#' Fit two-process NB ratio model using PG+CRT Gibbs sampling
#'
#' @description
#' Fits the ratiod_negbin_negbin() family using pure Gibbs sampling.
#' Both numerator and denominator are NB distributed with shared random effects.
#'
#' @param formula A ratiod_formula object
#' @param data Data frame
#' @param family Must be ratiod_negbin_negbin()
#' @param iter Total iterations
#' @param warmup Warmup iterations
#' @param chains Number of chains
#' @param thin Thinning interval
#' @param cores Number of cores
#' @param seed Random seed
#' @param prior_beta_sd Prior SD for fixed effects
#' @param prior_sigma_scale Half-Cauchy scale for RE SD
#' @param prior_r_shape Gamma shape for dispersion priors
#' @param prior_r_rate Gamma rate for dispersion priors
#' @param r_num_init Initial dispersion for numerator
#' @param r_denom_init Initial dispersion for denominator
#' @param verbose Print progress
#'
#' @return A ratiod_fit object
#' @keywords internal
fit_pg_negbin_negbin <- function(formula,
                                  data,
                                  family,
                                  iter = 2000,
                                  warmup = floor(iter / 2),
                                  chains = 4,
                                  thin = 1,
                                  cores = NULL,
                                  seed = NULL,
                                  prior_beta_sd = 10,
                                  prior_sigma_scale = 2.5,
                                  prior_r_shape = 1,
                                  prior_r_rate = 0.1,
                                  r_num_init = 1.0,
                                  r_denom_init = 1.0,
                                  verbose = TRUE) {

  if (is.null(cores)) {
    cores <- cpp_pg_get_max_threads()
  }

  if (!is.null(seed)) {
    set.seed(seed)
  }

  # Extract responses
  y_num <- formula$numerator$response
  y_denom <- formula$denominator$response

  if (is.null(y_denom)) {
    stop("NegBin-NegBin models require both numerator and denominator counts.")
  }

  # Design matrices
  X_num <- formula$numerator$X
  X_denom <- formula$denominator$X

  # If denominator has no predictors, use intercept only
  if (is.null(X_denom)) {
    X_denom <- matrix(1, nrow = nrow(data), ncol = 1)
    colnames(X_denom) <- "(Intercept)"
  }

  # Random effects
  re_info <- extract_re_from_data(formula, data)

  # Check for shared structure (default in tulpaRatio)
  shared <- is.null(formula$shared) || !isFALSE(formula$shared)

  n_iter <- as.integer(iter)
  n_warmup <- as.integer(warmup)

  # Run chains
  chain_results <- vector("list", chains)

  for (chain in seq_len(chains)) {
    if (verbose && chains > 1) {
      cat(sprintf("\n=== Chain %d/%d ===\n", chain, chains))
    }

    if (!is.null(seed)) {
      set.seed(seed + chain - 1)
    }

    chain_results[[chain]] <- cpp_pg_negbin_negbin_gibbs(
      y_num = as.integer(y_num),
      y_denom = as.integer(y_denom),
      X_num = X_num,
      X_denom = X_denom,
      group = re_info$group_idx,
      n_groups = re_info$n_groups,
      n_iter = n_iter,
      n_warmup = n_warmup,
      thin = as.integer(thin),
      prior_beta_sd = prior_beta_sd,
      prior_sigma_scale = prior_sigma_scale,
      prior_r_shape = prior_r_shape,
      prior_r_rate = prior_r_rate,
      r_num_init = r_num_init,
      r_denom_init = r_denom_init,
      shared = shared,
      store_eta = TRUE,
      verbose = verbose,
      n_threads = as.integer(cores)
    )
  }

  # Convert to ratiod_fit
  fit <- convert_pg_negbin_negbin_to_ratiod_fit(
    fit_raw = chain_results,
    formula = formula,
    data = data,
    family = family,
    X_num = X_num,
    X_denom = X_denom,
    re_info = re_info,
    iter = iter,
    warmup = warmup,
    thin = thin,
    chains = chains
  )

  return(fit)
}


#' Convert PG NegBin fit to ratiod_fit object
#' @keywords internal
convert_pg_negbin_to_ratiod_fit <- function(fit_raw, formula, data, family,
                                             spatial, X, re_info, iter, warmup, thin,
                                             chains = 1) {

  n_chains <- length(fit_raw)
  n_save_per_chain <- nrow(fit_raw[[1]]$beta)
  n_save_total <- n_save_per_chain * n_chains
  has_spatial <- !is.null(spatial)

  # Parameter names
  beta_names <- colnames(X)
  if (is.null(beta_names)) {
    beta_names <- paste0("beta[", seq_len(ncol(X)), "]")
  }

  # Count parameters
  n_params <- ncol(fit_raw[[1]]$beta) + 1  # beta + r
  if (re_info$n_groups > 0) n_params <- n_params + 1  # sigma_re
  if (has_spatial) n_params <- n_params + 1  # tau

  # Create 3D array for draws
  draws_array <- array(NA_real_,
                       dim = c(n_save_per_chain, n_chains, n_params))

  # Build parameter names
  param_names <- c(beta_names, "r")
  if (re_info$n_groups > 0) {
    param_names <- c(param_names, "sigma_re")
  }
  if (has_spatial) {
    param_names <- c(param_names, "tau_spatial")
  }

  dimnames(draws_array) <- list(
    iteration = seq_len(n_save_per_chain),
    chain = seq_len(n_chains),
    variable = param_names
  )

  # Fill draws
  for (c in seq_len(n_chains)) {
    chain_fit <- fit_raw[[c]]
    col_idx <- 1

    # Fixed effects
    for (j in seq_len(ncol(chain_fit$beta))) {
      draws_array[, c, col_idx] <- chain_fit$beta[, j]
      col_idx <- col_idx + 1
    }

    # Dispersion
    draws_array[, c, col_idx] <- chain_fit$r
    col_idx <- col_idx + 1

    # Sigma_re
    if (re_info$n_groups > 0) {
      draws_array[, c, col_idx] <- chain_fit$sigma_re
      col_idx <- col_idx + 1
    }

    # Tau
    if (has_spatial) {
      draws_array[, c, col_idx] <- chain_fit$tau
      col_idx <- col_idx + 1
    }
  }

  # Combine internals
  combined_beta <- do.call(rbind, lapply(fit_raw, `[[`, "beta"))
  combined_re <- do.call(rbind, lapply(fit_raw, `[[`, "re"))
  combined_eta <- do.call(rbind, lapply(fit_raw, `[[`, "eta"))
  combined_spatial <- if (has_spatial) {
    do.call(rbind, lapply(fit_raw, `[[`, "spatial"))
  } else NULL

  fit <- list(
    draws = draws_array,
    formula = formula,
    data = data,
    family = family,
    spatial = spatial,
    backend = "pg",
    iter = iter,
    warmup = warmup,
    thin = thin,
    chains = n_chains,
    n_save = n_save_total,
    n_save_per_chain = n_save_per_chain,
    .internal = list(
      eta = combined_eta,
      beta = combined_beta,
      re = combined_re,
      spatial = combined_spatial,
      X = X,
      re_info = re_info,
      chain_results = fit_raw
    )
  )

  class(fit) <- "ratiod_fit"
  return(fit)
}


#' Convert PG NegBin-NegBin fit to ratiod_fit object
#' @keywords internal
convert_pg_negbin_negbin_to_ratiod_fit <- function(fit_raw, formula, data, family,
                                                    X_num, X_denom, re_info,
                                                    iter, warmup, thin, chains = 1) {

  n_chains <- length(fit_raw)
  n_save_per_chain <- nrow(fit_raw[[1]]$beta_num)
  n_save_total <- n_save_per_chain * n_chains

  # Parameter names
  p_num <- ncol(X_num)
  p_denom <- ncol(X_denom)

  beta_num_names <- colnames(X_num)
  if (is.null(beta_num_names)) {
    beta_num_names <- paste0("beta_num[", seq_len(p_num), "]")
  } else {
    beta_num_names <- paste0("num_", beta_num_names)
  }

  beta_denom_names <- colnames(X_denom)
  if (is.null(beta_denom_names)) {
    beta_denom_names <- paste0("beta_denom[", seq_len(p_denom), "]")
  } else {
    beta_denom_names <- paste0("denom_", beta_denom_names)
  }

  # Count parameters
  n_params <- p_num + p_denom + 2  # betas + r_num + r_denom
  if (re_info$n_groups > 0) n_params <- n_params + 1  # sigma_re

  # Create 3D array
  draws_array <- array(NA_real_,
                       dim = c(n_save_per_chain, n_chains, n_params))

  # Build parameter names
  param_names <- c(beta_num_names, beta_denom_names, "r_num", "r_denom")
  if (re_info$n_groups > 0) {
    param_names <- c(param_names, "sigma_re")
  }

  dimnames(draws_array) <- list(
    iteration = seq_len(n_save_per_chain),
    chain = seq_len(n_chains),
    variable = param_names
  )

  # Fill draws
  for (c in seq_len(n_chains)) {
    chain_fit <- fit_raw[[c]]
    col_idx <- 1

    # Numerator betas
    for (j in seq_len(p_num)) {
      draws_array[, c, col_idx] <- chain_fit$beta_num[, j]
      col_idx <- col_idx + 1
    }

    # Denominator betas
    for (j in seq_len(p_denom)) {
      draws_array[, c, col_idx] <- chain_fit$beta_denom[, j]
      col_idx <- col_idx + 1
    }

    # Dispersions
    draws_array[, c, col_idx] <- chain_fit$r_num
    col_idx <- col_idx + 1
    draws_array[, c, col_idx] <- chain_fit$r_denom
    col_idx <- col_idx + 1

    # Sigma_re
    if (re_info$n_groups > 0) {
      draws_array[, c, col_idx] <- chain_fit$sigma_re
      col_idx <- col_idx + 1
    }
  }

  # Combine internals
  combined_beta_num <- do.call(rbind, lapply(fit_raw, `[[`, "beta_num"))
  combined_beta_denom <- do.call(rbind, lapply(fit_raw, `[[`, "beta_denom"))
  combined_re <- do.call(rbind, lapply(fit_raw, `[[`, "re"))
  combined_eta_num <- do.call(rbind, lapply(fit_raw, `[[`, "eta_num"))
  combined_eta_denom <- do.call(rbind, lapply(fit_raw, `[[`, "eta_denom"))

  fit <- list(
    draws = draws_array,
    formula = formula,
    data = data,
    family = family,
    backend = "pg",
    iter = iter,
    warmup = warmup,
    thin = thin,
    chains = n_chains,
    n_save = n_save_total,
    n_save_per_chain = n_save_per_chain,
    .internal = list(
      eta_num = combined_eta_num,
      eta_denom = combined_eta_denom,
      beta_num = combined_beta_num,
      beta_denom = combined_beta_denom,
      re = combined_re,
      X_num = X_num,
      X_denom = X_denom,
      re_info = re_info,
      chain_results = fit_raw
    )
  )

  class(fit) <- "ratiod_fit"
  return(fit)
}
