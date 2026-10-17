#' Gibbs Sampling Backend for Spatial Models
#'
#' @description
#' Component-wise Gibbs sampler for ICAR spatial models. Uses univariate
#' MH updates for spatial effects with the ICAR conditional prior as proposal,
#' conjugate Gamma update for spatial precision, and block MH for fixed effects.
#'
#' @details
#' This backend is designed for models where spatial effects dominate the
#' parameter space (large S). Unlike HMC, the Gibbs sampler updates spatial
#' effects one at a time, avoiding the curse of dimensionality.
#'
#' **Supported families:**
#' - `ratiod_poisson_gamma()` (Poisson numerator, Gamma denominator)
#' - `ratiod_negbin_negbin()` (NegBin numerator, NegBin denominator)
#' - `ratiod_binomial()` (Binomial)
#' - `ratiod_negbin_gamma()` (NegBin numerator, Gamma denominator)
#'
#' **Supported spatial structures:**
#' - ICAR (intrinsic CAR) via `spatial_car()`
#'
#' @name gibbs_backend
#' @keywords internal
NULL


#' Fit ratio model using Gibbs sampling
#'
#' @param formula A ratiod_formula object
#' @param data Data frame
#' @param family Model family
#' @param spatial Spatial structure from spatial_car()
#' @param priors Prior specification. See [ratiod_priors()].
#' @param iter Total iterations
#' @param warmup Warmup iterations
#' @param thin Thinning interval
#' @param chains Number of independent chains
#' @param cores Number of cores used to run chains in parallel
#' @param seed Random seed
#' @param verbose Print progress
#'
#' @return A ratiod_fit object
#' @keywords internal
fit_gibbs <- function(formula,
                      data,
                      family,
                      spatial,
                      temporal = NULL,
                      priors = NULL,
                      iter = 2000,
                      warmup = floor(iter / 2),
                      thin = 1,
                      chains = 4,
                      cores = NULL,
                      seed = NULL,
                      verbose = TRUE) {

  sp <- sampler_priors(priors, get_hmc_model_type(family))
  chains <- as.integer(chains)
  if (is.na(chains) || chains < 1L) {
    stop("`chains` must be a positive integer.", call. = FALSE)
  }

  # Bound the OpenMP team by both the chain count and the machine, so the team
  # size is stable across successive fits in one session.
  if (is.null(cores)) cores <- chains
  cores <- as.integer(cores)
  if (is.na(cores) || cores < 1L) cores <- 1L
  cores <- min(cores, chains, cpp_get_max_threads())

  if (is.null(spatial)) {
    stop("Gibbs backend requires spatial structure. Use spatial_car().",
         call. = FALSE)
  }

  if (!is.null(spatial$type) && !spatial$type %in% c("icar", "car", "bym2")) {
    stop(sprintf("Gibbs backend supports ICAR spatial only, got '%s'.",
                 spatial$type), call. = FALSE)
  }

  assert_backend_fits_structures(
    "gibbs",
    list(spatial = spatial, temporal = temporal)
  )

  if (is.null(seed)) {
    seed <- sample.int(.Machine$integer.max, 1)
  }

  # Map family to Gibbs family string
  family_str <- get_gibbs_family_string(family)

  # Get spatial group mapping
  spatial_var <- spatial$group_var
  if (is.null(spatial_var)) {
    stop("Gibbs backend requires group-level spatial effects (group_var).",
         call. = FALSE)
  }

  spatial_group <- areal_unit_index(spatial$adjacency, data[[spatial_var]])  # 1-based

  S <- nrow(spatial$adjacency)
  N <- nrow(data)

  # Build design matrices from formula
  X_num <- formula$numerator$X
  p_num <- ncol(X_num)

  is_binomial <- (family_str == "binomial")
  if (is_binomial) {
    X_denom <- matrix(0, nrow = 0, ncol = 0)
    p_denom <- 0L
  } else {
    X_denom <- formula$denominator$X
    p_denom <- ncol(X_denom)
  }

  # Extract responses
  y_num <- as.integer(formula$numerator$response)

  # Denominator response depends on family
  y_denom <- NULL
  y_denom_cont <- NULL

  if (family_str == "binomial") {
    y_denom <- as.integer(formula$denominator$response)
  } else if (family_str == "negbin_negbin") {
    y_denom <- as.integer(formula$denominator$response)
  } else {
    # poisson_gamma, negbin_gamma: continuous denominator
    y_denom_cont <- as.numeric(formula$denominator$response)
  }

  # Build adjacency in CSR format (1-based for R, C++ converts to 0-based)
  adj_matrix <- spatial$adjacency
  adj_row_ptr <- integer(S + 1)
  adj_col_idx <- integer(0)
  n_neighbors <- integer(S)

  for (i in seq_len(S)) {
    neighbors <- which(adj_matrix[i, ] > 0)
    n_neighbors[i] <- length(neighbors)
    adj_col_idx <- c(adj_col_idx, neighbors)  # 1-based
    adj_row_ptr[i + 1] <- adj_row_ptr[i] + length(neighbors)
  }

  # Build data list for C++
  data_list <- list(
    y_num = y_num,
    X_num = as.numeric(t(X_num)),  # Row-major flat vector
    spatial_group = spatial_group,  # 1-based (C++ converts)
    adj_row_ptr = adj_row_ptr,     # Already 0-indexed start
    adj_col_idx = as.integer(adj_col_idx),  # 1-based (C++ converts)
    n_neighbors = n_neighbors,
    N = as.integer(N),
    S = as.integer(S),
    p_num = as.integer(p_num),
    p_denom = as.integer(p_denom),
    n_iter = as.integer(iter),
    n_warmup = as.integer(warmup),
    thin = as.integer(thin),
    seed = as.integer(seed %% .Machine$integer.max),
    verbose = verbose,
    family = family_str,
    # The priors the HMC backend reads, so the spatial model is the one
    # compute_log_post() scores regardless of which backend runs.
    sigma_beta = sp$sigma_beta,
    sigma_re_scale = sp$sigma_re_scale,
    tau_spatial_shape = sp$tau_spatial_shape,
    tau_spatial_rate = sp$tau_spatial_rate
  )

  # Add denominator data
  if (!is_binomial) {
    data_list$X_denom <- as.numeric(t(X_denom))  # Row-major flat vector
  }

  if (family_str == "negbin_negbin" || family_str == "binomial") {
    data_list$y_denom <- as.integer(y_denom)
  } else {
    data_list$y_denom_cont <- as.numeric(y_denom_cont)
  }

  # BYM2 settings
  is_bym2 <- !is.null(spatial$type) && tolower(spatial$type) == "bym2"
  data_list$is_bym2 <- is_bym2
  if (is_bym2) {
    data_list$bym2_scale <- spatial$scale_factor %||% 1.0
  }

  # TVC settings
  has_tvc <- !is.null(temporal) && inherits(temporal, "ratiod_tvc")
  data_list$has_tvc <- has_tvc
  if (has_tvc) {
    tvc <- validate_tvc(temporal, data, X_num)
    if (!identical(tvc$structure %||% "rw1", "rw1")) {
      stop(sprintf(
        "Gibbs backend supports TVC structure = \"rw1\" only, got \"%s\". ",
        tvc$structure %||% "rw1"
      ), "Use `mode = \"hmc\"` for \"rw2\"/\"ar1\"/\"iid\".", call. = FALSE)
    }
    data_list$tvc_time_index <- as.integer(tvc$time_index)
    data_list$tvc_group_index <- as.integer(tvc$group_index)
    data_list$tvc_X <- as.numeric(t(tvc$X_tvc))  # Row-major
    data_list$tvc_n_times <- as.integer(tvc$n_times)
    data_list$tvc_n_groups <- as.integer(tvc$n_groups)
    data_list$tvc_n_terms <- as.integer(tvc$n_tvc)
    data_list$tvc_shared <- tvc$shared %||% TRUE
    data_list$tvc_structure <- tvc$structure %||% "rw1"
  }

  # Each chain is an independent run of the sampler under its own seed. The
  # seeds are drawn once from `seed` so the whole fit stays reproducible.
  data_list$chain_seeds <- (as.numeric(seed) + seq_len(chains) * 7919) %%
    .Machine$integer.max
  data_list$n_cores <- as.integer(cores)

  # The chains run under OpenMP inside cpp_gibbs_spatial(), which returns one
  # result list per chain.
  cpp_chains <- cpp_gibbs_spatial(data_list)

  convert_chain <- function(cpp_result) {
    n_save <- cpp_result$n_save
    # C++ stores row-major but Rcpp::NumericMatrix fills column-major, so the
    # layout is reconstructed with byrow = TRUE.
    draws_mat <- matrix(as.numeric(cpp_result$draws),
                        nrow = n_save, ncol = cpp_result$n_params, byrow = TRUE)
    colnames(draws_mat) <- as.character(cpp_result$param_names)
    phi_mat <- matrix(as.numeric(cpp_result$phi_draws),
                      nrow = n_save, ncol = cpp_result$S, byrow = TRUE)
    colnames(phi_mat) <- paste0("phi_spatial[", seq_len(cpp_result$S), "]")
    list(draws = draws_mat, phi = phi_mat, cpp = cpp_result)
  }

  chain_fits <- lapply(cpp_chains, convert_chain)

  # Chains are stacked contiguously (chain 1 first), matching the HMC backend
  # so that a [iteration, chain] reshape is valid on either.
  samples_list <- lapply(chain_fits, function(cf) cbind(cf$draws, cf$phi))
  draws_mat <- do.call(rbind, lapply(chain_fits, `[[`, "draws"))
  phi_draws <- do.call(rbind, lapply(chain_fits, `[[`, "phi"))
  draws_full <- do.call(rbind, samples_list)

  cpp_result <- chain_fits[[1]]$cpp
  n_save_per_chain <- cpp_result$n_save
  n_save <- nrow(draws_mat)
  n_params <- ncol(draws_mat)
  param_names <- colnames(draws_mat)

  # eta_num[draw, i] = phi[draw, spatial_group[i]] + X_num[i, ] %*% beta_num[draw, ]
  beta_num_cols <- seq_len(p_num)
  beta_denom_cols <- if (p_denom > 0) (p_num + 1):(p_num + p_denom) else integer(0)

  phi_by_obs <- phi_draws[, spatial_group, drop = FALSE]
  eta_num <- draws_mat[, beta_num_cols, drop = FALSE] %*% t(X_num) + phi_by_obs
  eta_denom <- if (is_binomial) {
    matrix(0.0, nrow = n_save, ncol = N)
  } else {
    draws_mat[, beta_denom_cols, drop = FALSE] %*% t(X_denom) + phi_by_obs
  }

  model_type <- get_hmc_model_type(family)
  draws_full <- gibbs_natural_draws(draws_full, model_type)
  samples_list <- lapply(samples_list, gibbs_natural_draws, model_type = model_type)
  ratio_draws <- hmc_response_draws(list(eta_num = eta_num, eta_denom = eta_denom),
                                    family_links(family, model_type))$ratio
  colnames(ratio_draws) <- paste0("ratio[", seq_len(N), "]")

  # Build ratiod_fit
  fit <- list(
    draws = draws_full,
    samples = samples_list,
    ratio_draws = ratio_draws,
    formula = formula,
    data = data,
    family = family,
    spatial = spatial,
    backend = "gibbs",
    iter = iter,
    warmup = warmup,
    thin = thin,
    chains = chains,
    n_save = n_save,
    n_save_per_chain = n_save_per_chain,
    .internal = list(
      eta_num = eta_num,
      eta_denom = eta_denom,
      eta = eta_num,  # For methods expecting single eta
      eta_array = array(eta_num, dim = c(n_save_per_chain, chains, N)),
      phi_draws = phi_draws,
      X = X_num,
      # One row per chain; acceptance rates are per-site for phi.
      accept_phi = do.call(rbind, lapply(chain_fits, function(cf) cf$cpp$accept_phi)),
      accept_beta = do.call(rbind, lapply(chain_fits, function(cf) cf$cpp$accept_beta)),
      accept_disp = do.call(rbind, lapply(chain_fits, function(cf) cf$cpp$accept_disp))
    )
  )

  class(fit) <- "ratiod_fit"
  return(fit)
}


#' Gibbs draws on the natural scale
#'
#' The sampler works on log precisions, log scales and a logit mixing weight;
#' the fit reports them under the names the HMC backend gives the same
#' parameters.
#'
#' @param draws Draw matrix with the sampler's column names.
#' @param model_type Model type string of the family.
#' @return The draw matrix with transformed, renamed columns.
#' @keywords internal
gibbs_natural_draws <- function(draws, model_type) {
  disp <- MODEL_DISPERSION_NAMES[[model_type]]
  disp_name <- function(arm) if (arm %in% names(disp)) disp[[arm]] else NA
  map <- list(
    log_phi_num = c(disp_name("num"), "exp"),
    log_disp_num = c(disp_name("num"), "exp"),
    log_phi_denom = c(disp_name("denom"), "exp"),
    log_disp_denom = c(disp_name("denom"), "exp"),
    log_tau = c("tau_spatial", "exp"),
    log_sigma_total = c("sigma_spatial", "exp"),
    logit_rho = c("rho_spatial", "plogis")
  )
  cn <- colnames(draws)
  for (raw in intersect(names(map), cn)) {
    target <- map[[raw]]
    if (is.na(target[1])) next
    j <- match(raw, cn)
    draws[, j] <- if (target[2] == "exp") exp(draws[, j]) else stats::plogis(draws[, j])
    cn[j] <- target[1]
  }
  colnames(draws) <- cn
  draws
}


# The numerator and denominator pairs the Gibbs sampler has a likelihood for.
GIBBS_MODEL_TYPES <- c("poisson_gamma", "negbin_negbin", "negbin_gamma", "binomial")

#' The model type the Gibbs sampler fits for a family
#'
#' The same pair the HMC samplers read (`get_hmc_model_type()`), or `NA` when
#' the Gibbs sampler has no likelihood for it, as for an inflated count or a
#' known exposure.
#' @keywords internal
gibbs_model_type <- function(family) {
  if (is_zi_family(family)) return(NA_character_)
  type <- tryCatch(get_hmc_model_type(family), error = function(e) NA_character_)
  if (type %in% GIBBS_MODEL_TYPES) type else NA_character_
}

#' Map family to Gibbs family string
#' @keywords internal
get_gibbs_family_string <- function(family) {
  type <- gibbs_model_type(family)
  if (is.na(type)) {
    stop(sprintf("Gibbs backend does not support family '%s'. Use mode='hmc'.",
                 family$name), call. = FALSE)
  }
  type
}
