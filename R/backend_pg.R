pg_run_chains <- function(n, seed, verbose, call) {
  lapply(seq_len(n), function(chain) {
    if (verbose && n > 1L) cat(sprintf("\n=== Chain %d/%d ===\n", chain, n))
    if (!is.null(seed)) set.seed(seed + chain - 1L)
    call(chain)
  })
}

pg_beta_block <- function(X, field = "beta", arm = "num") {
  stats::setNames(rep(field, ncol(X)), fixed_effect_variables(X, arm))
}

pg_re_block <- function(re_info) {
  if (re_info$n_groups > 0L) c(sigma_re = "sigma_re") else character(0)
}

pg_stack <- function(chain_results, field) {
  values <- lapply(chain_results, `[[`, field)
  if (is.null(values[[1L]])) return(NULL)
  if (is.null(dim(values[[1L]]))) unlist(values, use.names = FALSE)
  else do.call(rbind, values)
}

pg_assemble_fit <- function(chain_results, formula, data, family, iter, warmup,
                            thin, blocks, combined_fields = c("eta", "beta", "re"),
                            internal_extra = list(), ...) {
  n_chains <- length(chain_results)
  first <- chain_results[[1L]][[unname(blocks[1L])]]
  n_save <- NROW(first)
  draws <- array(NA_real_, c(n_save, n_chains, length(blocks)),
    dimnames = list(iteration = seq_len(n_save), chain = seq_len(n_chains),
                    variable = names(blocks)))
  for (field in unique(unname(blocks))) {
    columns <- which(blocks == field)
    for (chain in seq_len(n_chains)) {
      values <- chain_results[[chain]][[field]]
      if (length(values) != n_save * length(columns))
        stop("PG draw block has an inconsistent size: ", field, call. = FALSE)
      draws[, chain, columns] <- values
    }
  }
  internal <- setNames(lapply(combined_fields, function(field)
    pg_stack(chain_results, field)), combined_fields)
  if (!is.null(chain_results[[1L]]$eta)) {
    eta <- array(NA_real_, c(n_save, n_chains, ncol(chain_results[[1L]]$eta)))
    for (chain in seq_len(n_chains)) eta[, chain, ] <- chain_results[[chain]]$eta
    internal$eta_array <- eta
  }
  internal <- c(internal, internal_extra, list(chain_results = chain_results))
  structure(c(list(draws = draws, formula = formula, data = data, family = family,
    backend = "pg", iter = iter, warmup = warmup, thin = thin, chains = n_chains,
    n_save = n_save * n_chains, n_save_per_chain = n_save, .internal = internal),
    list(...)), class = "ratiod_fit")
}

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

#' - When you need posterior draws from a Gibbs sampler
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
#' @param prior_sigma_spatial_U PC prior SD anchor (default 3)
#' @param prior_sigma_spatial_alpha PC prior tail probability (default 0.01)
#' @param prior_sigma_temporal_U,prior_sigma_temporal_alpha PC prior anchors for the RW2 trend SD.
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
                            prior_sigma_spatial_U = 3,
                            prior_sigma_spatial_alpha = 0.01,
                            prior_sigma_temporal_U = 3,
                            prior_sigma_temporal_alpha = 0.01,
                            verbose = TRUE) {

  assert_backend_fits_structures(
    "pg",
    list(spatial = spatial, temporal = temporal)
  )

  # Set cores for within-chain parallelism
  cores <- clamp_check_cores(cores %||% cpp_get_max_threads(),
                             check = is.null(cores))

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
      prior_sigma_spatial_U = prior_sigma_spatial_U,
      prior_sigma_spatial_alpha = prior_sigma_spatial_alpha,
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
      prior_beta_sd = prior_beta_sd,
      prior_sigma_scale = prior_sigma_scale,
      prior_sigma_trend_U = prior_sigma_temporal_U,
      prior_sigma_trend_alpha = prior_sigma_temporal_alpha,
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
  chain_results <- pg_run_chains(chains, seed, verbose, function(chain) {
    if (has_spatial) {
      if (is_bym2) {
        # BYM2 spatial model
        scale_factor <- spatial$scale_factor %||% 1.0

        cpp_pg_binomial_gibbs_bym2(
          y = as.integer(y),
          n = as.integer(trials),
          X = X,
          re_group = re_info$group_idx,
          n_re_groups = re_info$n_groups,
          spatial_group = spatial_info$group_idx,
          n_spatial_units = spatial_info$n_units,
          adj_list = spatial_info$adj_list,
          scale_factor = scale_factor,
          n_iter = n_iter,
          n_warmup = n_warmup,
          thin = as.integer(thin),
          prior_beta_sd = prior_beta_sd,
          prior_sigma_re_scale = prior_sigma_scale,
          prior_sigma_spatial_U = prior_sigma_spatial_U,
          prior_sigma_spatial_alpha = prior_sigma_spatial_alpha,
          store_eta = TRUE,
          verbose = verbose,
          n_threads = as.integer(cores)
        )
      } else {
        # ICAR spatial model
        cpp_pg_binomial_gibbs_spatial(
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
          prior_sigma_spatial_U = prior_sigma_spatial_U,
          prior_sigma_spatial_alpha = prior_sigma_spatial_alpha,
          store_eta = TRUE,
          verbose = verbose,
          n_threads = as.integer(cores)
        )
      }

    } else {
      # Fit with random effects only
      cpp_tulpa_pg_binomial_gibbs(
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

    }

  })

  # Convert to ratiod_fit format with multi-chain support
  fit <- pg_assemble_fit(chain_results, formula, data, family, iter, warmup, thin,
    blocks = c(pg_beta_block(X), pg_re_block(re_info),
      if (has_spatial) {
        if (is_bym2) c(sigma_spatial = "sigma_spatial", rho = "rho")
        else c(tau_spatial = "tau")
      }),
    combined_fields = c("eta", "beta", "re", "spatial", "phi_scaled", "theta"),
    internal_extra = list(X = X, re_info = re_info), spatial = spatial)

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

  re_terms_processed <- vector("list", n_re_terms)
  group_idx_matrix <- matrix(0L, nrow = n_obs, ncol = n_re_terms)
  total_groups <- 0L
  for (t in seq_len(n_re_terms)) {
    term <- re_terms[[t]]
    group_idx_matrix[, t] <- as.integer(term$group)
    re_terms_processed[[t]] <- list(
      group_var = term$group_var,
      group_idx = group_idx_matrix[, t],
      n_groups = as.integer(term$n_groups),
      offset = total_groups
    )
    total_groups <- total_groups + term$n_groups
  }

  c(list(
    group_idx = re_terms_processed[[1]]$group_idx,
    n_groups = re_terms_processed[[1]]$n_groups,
    group_var = re_terms_processed[[1]]$group_var,
    group_levels = NULL,
    n_re_terms = n_re_terms,
    re_terms = re_terms_processed),
    if (n_re_terms > 1L) list(group_idx_matrix = group_idx_matrix),
    list(total_groups = total_groups,
    has_slopes = has_slopes
  ))
}


#' Prepare spatial structure for PG backend
#'
#' @keywords internal
prepare_spatial_for_pg <- function(spatial, data, formula) {
  # Get spatial group variable
  if (spatial$spatial_level == "group") {
    spatial_var <- spatial$group_var
    if (is.null(spatial_var)) {
      stop("spatial_car() with spatial_level = 'group' requires group_var")
    }
  } else {
    # Observation level - use row indices
    spatial_var <- NULL
  }

  if (!is.null(spatial_var)) {
    group_idx <- areal_unit_index(spatial$adjacency, data[[spatial_var]])
    n_units <- nrow(spatial$adjacency)
  } else {
    # Observation level
    group_idx <- seq_len(nrow(data))
    n_units <- nrow(data)
  }

  csr <- adjacency_csr(spatial$adjacency)

  return(list(
    group_idx = as.integer(group_idx),
    n_units = as.integer(n_units),
    adj_list = csr$adj_list,
    n_neighbors = csr$n_neighbors
  ))
}



#' Check if PG backend is appropriate
#'
#' @param family A tulpaRatio family object
#' @return Logical indicating if PG backend can be used
#' @keywords internal
can_use_pg_backend <- function(family) {
  model_type <- get_hmc_model_type(family)
  !is.null(PG_MODELS[[model_type]]) && family_canonical_links(family, model_type) &&
    !isTRUE(family$zero_inflated) && !isTRUE(family$one_inflated)
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
  cores <- clamp_check_cores(cores %||% cpp_get_max_threads(),
                             check = is.null(cores))

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
  chain_results <- pg_run_chains(chains, seed, verbose, function(chain) {
    cpp_pg_binomial_gibbs_gp(
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

  })

  # Convert to ratiod_fit format
  fit <- pg_assemble_fit(chain_results, formula, data, family, iter, warmup, thin,
    blocks = c(pg_beta_block(X), c(sigma2_gp = "sigma2_gp", phi_gp = "phi_gp"),
               pg_re_block(re_info)),
    combined_fields = c("eta", "beta", "re"),
    internal_extra = list(
      w_gp = pg_stack(chain_results, "gp"),
      gp_hyper = data.frame(sigma2 = pg_stack(chain_results, "sigma2_gp"),
                            phi = pg_stack(chain_results, "phi_gp")),
      gp_coords = spatial$unique_coords, gp_obs_to_loc = spatial$obs_to_loc,
      X = X, re_info = re_info), spatial = spatial, spatial_type = "gp")

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
#' @param prior_beta_sd Prior SD for fixed effects.
#' @param prior_sigma_scale Half-Cauchy scale for random-effect SDs.
#' @param prior_sigma_trend_U,prior_sigma_trend_alpha PC prior anchors for the RW2 trend SD.
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
                                      prior_beta_sd = 10,
                                      prior_sigma_scale = 2.5,
                                      prior_sigma_trend_U = 3,
                                      prior_sigma_trend_alpha = 0.01,
                                      verbose = TRUE) {

  # Set cores for within-chain parallelism
  cores <- clamp_check_cores(cores %||% cpp_get_max_threads(),
                             check = is.null(cores))

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
  chain_results <- pg_run_chains(chains, seed, verbose, function(chain) {
    cpp_pg_binomial_gibbs_temporal(
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
      prior_beta_sd = prior_beta_sd,
      prior_sigma_re_scale = prior_sigma_scale,
      prior_sigma_trend_U = prior_sigma_trend_U,
      prior_sigma_trend_alpha = prior_sigma_trend_alpha,
      store_eta = TRUE,
      verbose = verbose,
      n_threads = as.integer(cores)
    )

  })

  # Convert to ratiod_fit format
  fit <- pg_assemble_fit(chain_results, formula, data, family, iter, warmup, thin,
    blocks = c(pg_beta_block(X),
      if (temporal$trend != "none") c(sigma_trend = "sigma_trend"),
      if (!is.null(temporal$seasonal) && temporal$seasonal > 0)
        c(sigma_seasonal = "sigma_seasonal"),
      if (temporal$short_term != "none") c(sigma_short = "sigma_short"),
      if (temporal$short_term == "ar1") c(rho_short = "rho_short"),
      pg_re_block(re_info)),
    internal_extra = list(
      temporal_draws = setNames(lapply(c("trend", "seasonal", "short_term"),
        function(field) pg_stack(chain_results, field)),
        c("trend", "seasonal", "short_term")), X = X, re_info = re_info),
    temporal = temporal)

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
                                 prior_sigma_spatial_U = 3,
                                 prior_sigma_spatial_alpha = 0.01,
                                 verbose = TRUE) {

  if (verbose) {
    message("Fitting RSR spatial model with Polya-Gamma Gibbs sampling...")
  }

  # Set cores
  cores <- clamp_check_cores(cores %||% cpp_get_max_threads(),
                             check = is.null(cores))

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
  chain_results <- pg_run_chains(chains, seed, verbose, function(chain) {
    cpp_pg_binomial_gibbs_rsr(
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
      prior_sigma_spatial_U = prior_sigma_spatial_U,
      prior_sigma_spatial_alpha = prior_sigma_spatial_alpha,
      store_eta = TRUE,
      verbose = verbose,
      n_threads = as.integer(cores)
    )

  })

  # Convert to ratiod_fit
  fit <- pg_assemble_fit(chain_results, formula, data, family, iter, warmup, thin,
    blocks = c(pg_beta_block(X), pg_re_block(re_info), c(tau_spatial = "tau")),
    combined_fields = c("eta", "beta", "re", "spatial", "spatial_raw"),
    internal_extra = list(X = X, re_info = re_info, spatial_info = spatial_info),
    spatial = spatial_info, spatial_type = "rsr")

  return(fit)
}



PG_MODELS <- list(
  binomial = list(fit = "fit_pg_binomial", fields = c(spatial = TRUE, temporal = TRUE),
    priors = c(
    prior_beta_sd = "sigma_beta", prior_sigma_scale = "sigma_re_scale",
    prior_sigma_spatial_U = "spatial_sigma_prior_U",
    prior_sigma_spatial_alpha = "spatial_sigma_prior_alpha",
    prior_sigma_temporal_U = "temporal_sigma_prior_U",
    prior_sigma_temporal_alpha = "temporal_sigma_prior_alpha")),
  negbin_negbin = list(fit = "fit_pg_negbin_negbin",
    fields = c(spatial = FALSE, temporal = FALSE), priors = c(
    prior_beta_sd = "sigma_beta", prior_sigma_scale = "sigma_re_scale",
    prior_r_shape = "phi_shape", prior_r_rate = "phi_rate"))
)

pg_family_field_support <- function(field, structures) {
  if (is.null(structures$family)) return(TRUE)
  model <- PG_MODELS[[get_hmc_model_type(structures$family)]]
  if (isTRUE(model$fields[[field]])) return(TRUE)
  "this PG likelihood fits fixed effects and a shared random intercept"
}

fit_pg <- function(formula, data, family, spatial = NULL, temporal = NULL,
                   priors = NULL, iter = 2000, warmup = floor(iter / 2),
                   chains = 4, thin = 1, cores = NULL, seed = NULL,
                   verbose = TRUE) {
  model_type <- get_hmc_model_type(family)
  model <- PG_MODELS[[model_type]]
  if (is.null(model) || !can_use_pg_backend(family))
    stop("The PG backend fits logit binomial and log-link NB-NB models.", call. = FALSE)
  assert_backend_fits_structures("pg", list(family = family,
    spatial = spatial, temporal = temporal))
  p <- sampler_priors(priors, model_type)
  args <- list(formula = formula, data = data, family = family,
    iter = iter, warmup = warmup, chains = chains, thin = thin,
    cores = cores, seed = seed, verbose = verbose)
  if (model_type == "binomial") {
    args$spatial <- spatial
    args$temporal <- temporal
  }
  args <- c(args, stats::setNames(p[unname(model$priors)], names(model$priors)))
  do.call(get(model$fit, mode = "function"), args)
}

#' Fit an NB-NB ratio with Polya-Gamma augmentation
#'
#' Both coefficients and saved predictors use log means. Dispersion has a
#' Gamma(shape, rate) prior on the positive real line. A common random
#' intercept enters both processes, with a Half-Cauchy prior on its SD.
#'
#' @inheritParams fit_pg_binomial
#' @param family An `ratiod_negbin_negbin()` family.
#' @param prior_r_shape,prior_r_rate Gamma shape and rate for each dispersion.
#' @param r_num_init,r_denom_init Initial dispersions.
#' @keywords internal
fit_pg_negbin_negbin <- function(formula, data, family,
    iter = 2000, warmup = floor(iter / 2), chains = 4, thin = 1,
    cores = NULL, seed = NULL, prior_beta_sd = 10, prior_sigma_scale = 2.5,
    prior_r_shape = 2, prior_r_rate = 0.1, r_num_init = 1, r_denom_init = 1,
    verbose = TRUE) {
  cores <- clamp_check_cores(cores %||% cpp_get_max_threads(), check = is.null(cores))
  shared_terms <- formula$shared$random_effects
  terms <- c(formula$numerator$random_effects, formula$denominator$random_effects)
  shared_names <- vapply(shared_terms, `[[`, "", "group_var")
  if (any(!vapply(terms, function(term) term$group_var %in% shared_names, FALSE)))
    stop("The NB-NB PG sampler fits a shared random intercept. Use mode = 'hmc' for process-specific random effects.", call. = FALSE)
  if (length(shared_terms) > 1L || any(vapply(shared_terms, function(term)
      length(term$slope_vars) > 0L || !isTRUE(term$has_intercept), FALSE)))
    stop("The NB-NB PG sampler fits one shared random intercept. Use mode = 'hmc' for random slopes or multiple grouping factors.", call. = FALSE)
  re_formula <- formula
  re_formula$numerator$random_effects <- shared_terms
  re_info <- extract_re_from_data(re_formula, data)
  X_num <- formula$numerator$X
  X_denom <- formula$denominator$X
  chain_results <- pg_run_chains(chains, seed, verbose, function(chain) {
    cpp_pg_negbin_negbin_gibbs(
      y_num = as.integer(formula$numerator$response),
      y_denom = as.integer(formula$denominator$response),
      X_num = X_num, X_denom = X_denom, group = re_info$group_idx,
      n_groups = re_info$n_groups, n_iter = as.integer(iter),
      n_warmup = as.integer(warmup), thin = as.integer(thin),
      prior_beta_sd = prior_beta_sd, prior_sigma_scale = prior_sigma_scale,
      prior_r_shape = prior_r_shape, prior_r_rate = prior_r_rate,
      r_num_init = r_num_init, r_denom_init = r_denom_init,
      shared = re_info$n_groups > 0L, store_eta = TRUE,
      verbose = verbose, n_threads = as.integer(cores))
  })
  pg_assemble_fit(chain_results, formula, data, family, iter, warmup, thin,
    blocks = c(pg_beta_block(X_num, "beta_num", "num"),
               pg_beta_block(X_denom, "beta_denom", "denom"),
               c(phi_num = "r_num", phi_denom = "r_denom"), pg_re_block(re_info)),
    combined_fields = c("eta_num", "eta_denom", "beta_num", "beta_denom", "re"),
    internal_extra = list(X_num = X_num, X_denom = X_denom,
                          X = X_num, re_info = re_info))
}
