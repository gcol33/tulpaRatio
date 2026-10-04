#' Laplace Approximation Backend
#'
#' @description
#' Fast approximate inference using nested Laplace approximation.
#' Suitable for large areal datasets where full MCMC is too slow.
#'
#' @details
#' The Laplace backend finds the mode of the posterior and uses a Gaussian
#' approximation centered at the mode. This is much faster than MCMC but
#' provides approximate inference.
#'
#' **Supported model structures:**
#' - Fixed effects
#' - Group-level random intercepts
#' - All three families (binomial, negbin, poisson_gamma)
#'
#' **When to use:**
#' - Large datasets (N > 10,000)
#' - Screening models before full MCMC
#' - When approximate posteriors are acceptable
#'
#' @name laplace_backend
#' @keywords internal
NULL


#' Fit model using Laplace approximation
#'
#' @param formula A ratiod_formula object
#' @param data Data frame
#' @param family Model family
#' @param spatial Optional spatial structure
#' @param temporal Optional multi-scale temporal structure. See
#'   `BACKEND_STRUCTURE_SUPPORT` for what this backend carries.
#' @param priors Prior specification
#' @param n_samples Number of posterior samples to draw
#' @param cores Number of cores for parallel computation
#' @param verbose Print progress
#'
#' @return A ratiod_fit object
#' @keywords internal
fit_laplace <- function(formula,
                        data,
                        family,
                        spatial = NULL,
                        temporal = NULL,
                        priors = NULL,
                        n_samples = 1000,
                        cores = NULL,
                        verbose = TRUE) {

  assert_backend_fits_structures(
    "laplace",
    list(spatial = spatial, temporal = temporal)
  )

  # Set cores
  if (is.null(cores)) {
    cores <- cpp_laplace_get_max_threads()
  }

  # Determine which family/likelihood to use
  family_type <- get_laplace_family(family)

  # Extract response based on family
  if (family_type == "binomial") {
    y <- formula$numerator$response
    n_trials <- formula$denominator$response
    phi <- 1.0  # Not used for binomial
  } else if (family_type == "negbin") {
    y <- formula$numerator$response
    n_trials <- rep(1L, length(y))  # Placeholder
    # Extract phi: may be a prior object or scalar
    phi <- extract_prior_default(priors$phi, default = 1.0)
  } else {
    y <- formula$numerator$response
    n_trials <- rep(1L, length(y))
    phi <- 1.0
  }

  # Design matrix
  X <- formula$numerator$X

  # Random effects
  re_info <- extract_re_for_laplace(formula)

  # Hyperparameters (extract from prior objects if needed)
  sigma_re <- extract_prior_default(priors$sigma, default = 1.0)
  tau_spatial <- extract_prior_default(priors$tau_spatial, default = 1.0)

  # Check if spatial model
  has_spatial <- !is.null(spatial)

  if (has_spatial) {
    # Dispatch to spatial Laplace
    return(fit_laplace_spatial(
      formula = formula,
      data = data,
      family = family,
      family_type = family_type,
      spatial = spatial,
      y = y,
      n_trials = n_trials,
      X = X,
      re_info = re_info,
      phi = phi,
      sigma_re = sigma_re,
      tau_spatial = tau_spatial,
      n_samples = n_samples,
      cores = cores,
      verbose = verbose
    ))
  }

  if (!is.null(temporal)) {
    # Dispatch to multi-scale temporal Laplace
    return(fit_laplace_temporal(
      formula = formula,
      data = data,
      family = family,
      family_type = family_type,
      temporal = temporal,
      y = y,
      n_trials = n_trials,
      X = X,
      re_info = re_info,
      phi = phi,
      sigma_re = sigma_re,
      n_samples = n_samples,
      cores = cores,
      verbose = verbose
    ))
  }

  # For Laplace, we optimize sigma_re (empirical Bayes)
  if (verbose) {
    message("Finding mode of posterior...")
  }

  # First pass: find mode with initial sigma_re
  result <- optimize_laplace_hyperparams(
    y = y,
    n_trials = n_trials,
    X = X,
    re_idx = re_info$group_idx,
    n_re_groups = re_info$n_groups,
    family = family_type,
    phi = phi,
    sigma_re_init = sigma_re,
    n_threads = cores,
    verbose = verbose
  )

  # Now sample from Laplace approximation
  if (verbose) {
    message("Sampling from Laplace approximation...")
  }

  # Get the Hessian at the mode for sampling
  hess_result <- compute_hessian_at_mode(
    y = y,
    n_trials = n_trials,
    X = X,
    re_idx = re_info$group_idx,
    n_re_groups = re_info$n_groups,
    mode = result$mode,
    family = family_type,
    phi = phi,
    sigma_re = result$sigma_re_opt
  )

  # Sample from N(mode, H^{-1})
  samples <- sample_laplace_gaussian(
    mode = result$mode,
    H = hess_result$H,
    n_samples = as.integer(n_samples),
    X = X
  )

  # Convert to ratiod_fit format
  fit <- convert_laplace_to_ratiod_fit(
    samples = samples,
    result = result,
    formula = formula,
    data = data,
    family = family,
    X = X,
    re_info = re_info,
    n_samples = n_samples
  )

  return(fit)
}


#' Extract default value from prior specification
#'
#' @description
#' Handles prior objects (e.g., from `prior_pc()`) by extracting a reasonable
#' default value, or returns numeric values unchanged.
#'
#' @param prior A prior object or numeric value
#' @param default Default value to use if prior is NULL
#' @return A numeric scalar
#' @keywords internal
extract_prior_default <- function(prior, default = 1.0) {
  if (is.null(prior)) {
    return(default)
  }

  if (is.numeric(prior) && length(prior) == 1) {
    return(prior)
  }

  if (inherits(prior, "ratiod_prior")) {
    # For prior objects, use a reasonable default based on the distribution
    # PC prior: use the prior mean (1/rate) or just use the default
    # Other priors: use their location parameter if available
    if (prior$distribution == "pc") {
      # PC prior mean = 1/rate, but for Laplace we often want a fixed value
      # Use the upper bound U as a conservative estimate
      return(prior$U %||% default)
    } else if (prior$distribution == "normal") {
      return(prior$mean %||% default)
    } else if (prior$distribution == "half_normal" ||
               prior$distribution == "half_cauchy") {
      return(prior$scale %||% default)
    } else if (prior$distribution == "gamma") {
      # Gamma mean = shape/rate
      return((prior$shape / prior$rate) %||% default)
    }
    # Fallback for other prior types
    return(default)
  }

  # Fallback: return default for unrecognized types
  return(default)
}


#' Get Laplace family string
#' @keywords internal
get_laplace_family <- function(family) {
  dist <- family$numerator$distribution
  if (dist == "binomial") return("binomial")
  if (dist %in% c("negbin", "negative_binomial", "neg_binomial_2")) return("negbin")
  if (dist == "poisson") return("poisson")
  stop("Unsupported family for Laplace backend: ", dist)
}


#' Extract RE info for Laplace
#'
#' @description
#' Extracts random effects information for the Laplace backend.
#' Supports multiple crossed random effects (e.g., `(1|site) + (1|year)`).
#'
#' @param formula A ratiod_formula object
#' @return List with RE structure for Laplace backend
#' @keywords internal
extract_re_for_laplace <- function(formula) {
  re_terms <- formula$numerator$random_effects
  n_obs <- length(formula$numerator$response)

  if (is.null(re_terms) || length(re_terms) == 0) {
    return(list(
      group_idx = as.numeric(rep(1, n_obs)),
      n_groups = 0L,
      group_var = NULL,
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
        "Random slopes not yet fully supported in Laplace backend. Slope terms will be ignored: ",
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
        group_idx = as.numeric(term$group),
        n_groups = as.integer(term$n_groups),
        offset = total_groups
      )
      total_groups <- total_groups + term$n_groups
    }

    # Build group index matrix
    group_idx_matrix <- matrix(0, nrow = n_obs, ncol = n_re_terms)
    for (t in seq_len(n_re_terms)) {
      group_idx_matrix[, t] <- as.numeric(re_terms[[t]]$group)
    }

    return(list(
      # Legacy fields
      group_idx = as.numeric(re_terms[[1]]$group),
      n_groups = as.integer(re_terms[[1]]$n_groups),
      group_var = re_terms[[1]]$group_var,
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
  return(list(
    group_idx = as.numeric(re_info$group),
    n_groups = as.integer(re_info$n_groups),
    group_var = re_info$group_var,
    # Multi-term fields
    n_re_terms = 1L,
    re_terms = list(list(
      group_var = re_info$group_var,
      group_idx = as.numeric(re_info$group),
      n_groups = as.integer(re_info$n_groups),
      offset = 0L
    )),
    total_groups = as.integer(re_info$n_groups),
    has_slopes = has_slopes
  ))
}


#' Optimize hyperparameters via marginal likelihood
#' @keywords internal
optimize_laplace_hyperparams <- function(y, n_trials, X, re_idx, n_re_groups,
                                         family, phi, sigma_re_init, n_threads,
                                         verbose) {

  if (n_re_groups == 0) {
    # No random effects - just find mode
    result <- cpp_laplace_fit(
      y = as.integer(y),
      n = as.integer(n_trials),
      X = X,
      re_idx = re_idx,
      n_re_groups = 0L,
      sigma_re = 1.0,
      family = family,
      phi = phi,
      n_threads = as.integer(n_threads)
    )

    return(list(
      mode = result$mode,
      log_marginal = result$log_marginal,
      sigma_re_opt = 1.0,
      converged = result$converged
    ))
  }

  # Optimize sigma_re by maximizing marginal likelihood
  obj_fn <- function(log_sigma) {
    sigma <- exp(log_sigma)
    result <- cpp_laplace_fit(
      y = as.integer(y),
      n = as.integer(n_trials),
      X = X,
      re_idx = re_idx,
      n_re_groups = n_re_groups,
      sigma_re = sigma,
      family = family,
      phi = phi,
      n_threads = as.integer(n_threads)
    )
    # Return negative log marginal (for minimization)
    if (!result$converged) return(1e10)
    -result$log_marginal
  }

  # Optimize
  opt <- tryCatch({
    stats::optim(
      par = log(sigma_re_init),
      fn = obj_fn,
      method = "Brent",
      lower = log(0.01),
      upper = log(10)
    )
  }, error = function(e) {
    list(par = log(sigma_re_init), convergence = 1)
  })

  sigma_re_opt <- exp(opt$par)

  if (verbose) {
    message(sprintf("  Optimal sigma_re: %.3f", sigma_re_opt))
  }

  # Final fit at optimal sigma_re
  result <- cpp_laplace_fit(
    y = as.integer(y),
    n = as.integer(n_trials),
    X = X,
    re_idx = re_idx,
    n_re_groups = n_re_groups,
    sigma_re = sigma_re_opt,
    family = family,
    phi = phi,
    n_threads = as.integer(n_threads)
  )

  return(list(
    mode = result$mode,
    log_marginal = result$log_marginal,
    sigma_re_opt = sigma_re_opt,
    converged = result$converged
  ))
}


#' Compute Hessian at mode for sampling
#' @keywords internal
compute_hessian_at_mode <- function(y, n_trials, X, re_idx, n_re_groups,
                                    mode, family, phi, sigma_re) {
  N <- length(y)
  p <- ncol(X)
  n_x <- p + n_re_groups

  # Compute eta at mode
  eta <- as.numeric(X %*% mode[1:p])
  if (n_re_groups > 0) {
    for (i in seq_len(N)) {
      g <- as.integer(re_idx[i])
      if (g > 0 && g <= n_re_groups) {
        eta[i] <- eta[i] + mode[p + g]
      }
    }
  }

  # Build Hessian
  H <- matrix(0, n_x, n_x)

  # Likelihood contributions
  for (i in seq_len(N)) {
    if (family == "binomial") {
      p_i <- 1 / (1 + exp(-eta[i]))
      h_i <- n_trials[i] * p_i * (1 - p_i)
    } else if (family == "negbin") {
      mu_i <- exp(eta[i])
      h_i <- (y[i] + phi) * mu_i * phi / (mu_i + phi)^2
    } else {
      h_i <- exp(eta[i])
    }

    # Fixed effects block
    for (j in seq_len(p)) {
      for (k in seq_len(p)) {
        H[j, k] <- H[j, k] + h_i * X[i, j] * X[i, k]
      }
    }

    # Random effects
    if (n_re_groups > 0) {
      g <- as.integer(re_idx[i])
      if (g > 0 && g <= n_re_groups) {
        H[p + g, p + g] <- H[p + g, p + g] + h_i

        for (j in seq_len(p)) {
          H[j, p + g] <- H[j, p + g] + h_i * X[i, j]
          H[p + g, j] <- H[p + g, j] + h_i * X[i, j]
        }
      }
    }
  }

  # Add prior precision
  tau_re <- 1 / (sigma_re^2 + 1e-10)
  for (g in seq_len(n_re_groups)) {
    H[p + g, p + g] <- H[p + g, p + g] + tau_re
  }

  # Small regularization for fixed effects
  for (j in seq_len(p)) {
    H[j, j] <- H[j, j] + 1e-4
  }

  return(list(H = H))
}


#' Column names of the fixed effects in a Laplace fit
#' @keywords internal
laplace_beta_names <- function(X) {
  colnames(X) %||% paste0("beta[", seq_len(ncol(X)), "]")
}


#' Draw from the Laplace Gaussian and keep its fixed-effect block
#'
#' The Laplace approximation is `N(mode, H^{-1})`. The draws serve nonlinear
#' derived quantities; the fixed-effect rows of `summary()` read the closed
#' form, so the leading `p x p` block of `H^{-1}` travels with the draws.
#'
#' @param mode Posterior mode.
#' @param H Precision (negative Hessian) at the mode.
#' @param n_samples Number of draws.
#' @param X Design matrix; its columns are the leading entries of `mode`.
#' @return The `n_samples x length(mode)` draw matrix, with the attribute
#'   `fixed_gaussian` holding `mean`, `cov` and `names` of the fixed effects.
#' @keywords internal
sample_laplace_gaussian <- function(mode, H, n_samples, X) {
  samples <- cpp_laplace_sample(mode = mode, H = H, n_samples = n_samples)
  idx <- seq_len(ncol(X))
  selector <- diag(1, nrow(H), length(idx))
  cov <- solve(H, selector)[idx, , drop = FALSE]
  names <- laplace_beta_names(X)
  attr(samples, "fixed_gaussian") <- list(
    mean = stats::setNames(mode[idx], names),
    cov = matrix((cov + t(cov)) / 2, length(idx), length(idx),
                 dimnames = list(names, names)),
    names = names
  )
  samples
}


#' Convert Laplace results to ratiod_fit
#' @keywords internal
convert_laplace_to_ratiod_fit <- function(samples, result, formula, data,
                                         family, X, re_info, n_samples) {

  p <- ncol(X)
  n_re <- re_info$n_groups

  # Extract fixed effects samples
  beta_names <- laplace_beta_names(X)

  draws_list <- list()
  for (j in seq_len(p)) {
    draws_list[[beta_names[j]]] <- samples[, j]
  }

  # Add sigma_re as point estimate (could improve by sampling)
  if (n_re > 0) {
    draws_list[["sigma_re"]] <- rep(result$sigma_re_opt, n_samples)
  }

  draws <- do.call(cbind, draws_list)
  colnames(draws) <- names(draws_list)

  fit <- list(
    draws = draws,
    formula = formula,
    data = data,
    family = family,
    backend = "laplace",
    n_save = n_samples,
    laplace_result = result,
    fixed_gaussian = attr(samples, "fixed_gaussian"),
    plug_in = if (n_re > 0) "sigma_re",
    .internal = list(
      mode = result$mode,
      sigma_re = result$sigma_re_opt,
      log_marginal = result$log_marginal,
      X = X,
      re_info = re_info,
      samples = samples
    )
  )

  class(fit) <- "ratiod_fit"
  return(fit)
}


#' Check if Laplace backend can be used
#' @keywords internal
can_use_laplace_backend <- function(family) {
  # Laplace works for all families (including spatial now)
  TRUE
}


#' Fit spatial model using Laplace approximation
#'
#' @param formula A ratiod_formula object
#' @param data Data frame
#' @param family Model family
#' @param family_type String family type for C++
#' @param spatial Spatial structure specification
#' @param y Response vector
#' @param n_trials Trials vector
#' @param X Design matrix
#' @param re_info Random effects info
#' @param phi Overdispersion parameter
#' @param sigma_re RE standard deviation
#' @param tau_spatial Spatial precision
#' @param n_samples Number of posterior samples
#' @param cores Number of cores
#' @param verbose Print progress
#'
#' @return A ratiod_fit object
#' @keywords internal
fit_laplace_spatial <- function(formula,
                                 data,
                                 family,
                                 family_type,
                                 spatial,
                                 y,
                                 n_trials,
                                 X,
                                 re_info,
                                 phi,
                                 sigma_re,
                                 tau_spatial,
                                 n_samples,
                                 cores,
                                 verbose) {

  if (verbose) {
    message("Fitting spatial model with Laplace approximation...")
  }

  spatial_type <- spatial$type %||% "car"

  if (spatial_type == "car_proper") {
    stop("Proper CAR (`spatial_car(..., proper = TRUE)`) is not supported by ",
         "the Laplace backend. Use `mode = \"hmc\"`.", call. = FALSE)
  }

  if (spatial_type == "hsgp" || spatial_type == "svc") {
    stop(sprintf(
      "Spatial structure `%s` is not supported by the Laplace backend. Use `mode = \"hmc\"`.",
      spatial_type), call. = FALSE)
  }

  # Coordinate-based fields carry coordinates and neighbour sets rather than an
  # areal grouping, so they dispatch ahead of the adjacency preparation the
  # areal paths need.

  # GP spatial
  if (spatial_type == "gp" || inherits(spatial, "ratiod_gp")) {
    return(fit_laplace_gp(
      formula = formula,
      data = data,
      family = family,
      family_type = family_type,
      spatial = spatial,
      y = y,
      n_trials = n_trials,
      X = X,
      re_info = re_info,
      phi = phi,
      sigma_re = sigma_re,
      n_samples = n_samples,
      cores = cores,
      verbose = verbose
    ))
  }

  # Multi-scale GP (local + regional)
  if (spatial_type == "multiscale" || inherits(spatial, "ratiod_multiscale")) {
    return(fit_laplace_multiscale_gp(
      formula = formula,
      data = data,
      family = family,
      family_type = family_type,
      spatial = spatial,
      y = y,
      n_trials = n_trials,
      X = X,
      re_info = re_info,
      phi = phi,
      sigma_re = sigma_re,
      n_samples = n_samples,
      cores = cores,
      verbose = verbose
    ))
  }

  # Prepare spatial structure (areal fields: ICAR, BYM2, RSR)
  spatial_info <- prepare_spatial_for_laplace(spatial, data, formula)

  if (verbose) {
    message(sprintf("  Spatial units: %d", spatial_info$n_units))
    message(sprintf("  Spatial type: %s", spatial$type %||% "ICAR"))
  }

  # RSR (Restricted Spatial Regression)
  if (inherits(spatial, "ratiod_rsr")) {
    return(fit_laplace_rsr(
      formula = formula,
      data = data,
      family = family,
      family_type = family_type,
      spatial = spatial,
      spatial_info = spatial_info,
      y = y,
      n_trials = n_trials,
      X = X,
      re_info = re_info,
      phi = phi,
      sigma_re = sigma_re,
      tau_spatial = tau_spatial,
      n_samples = n_samples,
      cores = cores,
      verbose = verbose
    ))
  }

  # BYM2 model
  is_bym2 <- tolower(spatial_type) == "bym2"

  if (is_bym2) {
    return(fit_laplace_bym2(
      formula = formula,
      data = data,
      family = family,
      family_type = family_type,
      spatial = spatial,
      spatial_info = spatial_info,
      y = y,
      n_trials = n_trials,
      X = X,
      re_info = re_info,
      phi = phi,
      sigma_re = sigma_re,
      n_samples = n_samples,
      cores = cores,
      verbose = verbose
    ))
  }

  # Fit with fixed hyperparameters first (ICAR)
  if (verbose) {
    message("Finding mode of posterior...")
  }

  result <- cpp_laplace_fit_spatial(
    y = as.integer(y),
    n = as.integer(n_trials),
    X = X,
    re_idx = re_info$group_idx,
    n_re_groups = re_info$n_groups,
    sigma_re = sigma_re,
    spatial_idx = spatial_info$group_idx,
    n_spatial_units = spatial_info$n_units,
    adj_row_ptr = spatial_info$adj_row_ptr,
    adj_col_idx = spatial_info$adj_col_idx,
    n_neighbors = spatial_info$n_neighbors,
    tau_spatial = tau_spatial,
    family = family_type,
    phi = phi,
    n_threads = as.integer(cores)
  )

  if (!result$converged) {
    warning("Laplace approximation did not converge")
  }

  if (verbose) {
    message(sprintf("  Converged in %d iterations", result$n_iter))
    message("Sampling from Laplace approximation...")
  }

  # Compute Hessian at mode for sampling
  hess_result <- compute_hessian_spatial(
    y = y,
    n_trials = n_trials,
    X = X,
    re_idx = re_info$group_idx,
    n_re_groups = re_info$n_groups,
    spatial_idx = spatial_info$group_idx,
    n_spatial_units = spatial_info$n_units,
    adj_row_ptr = spatial_info$adj_row_ptr,
    adj_col_idx = spatial_info$adj_col_idx,
    n_neighbors = spatial_info$n_neighbors,
    mode = result$mode,
    family = family_type,
    phi = phi,
    sigma_re = sigma_re,
    tau_spatial = tau_spatial
  )

  # Sample from Laplace approximation
  samples <- sample_laplace_gaussian(
    mode = result$mode,
    H = hess_result$H,
    n_samples = as.integer(n_samples),
    X = X
  )

  # Convert to ratiod_fit
  fit <- convert_laplace_spatial_to_ratiod_fit(
    samples = samples,
    result = result,
    formula = formula,
    data = data,
    family = family,
    X = X,
    re_info = re_info,
    spatial_info = spatial_info,
    n_samples = n_samples
  )

  return(fit)
}


#' Prepare spatial structure for Laplace
#' @keywords internal
prepare_spatial_for_laplace <- function(spatial, data, formula) {
  # Get the spatial grouping variable
  group_var <- spatial$group_var
  if (is.null(group_var)) {
    stop(sprintf(
      paste0("Areal spatial structures need `group_var`; `%s` carries coordinates ",
             "instead and is fitted by its own Laplace path."),
      spatial$type %||% "car"), call. = FALSE)
  }

  # Get group indices
  if (!group_var %in% names(data)) {
    stop(sprintf("Spatial group variable '%s' not found in data", group_var))
  }

  group_factor <- as.factor(data[[group_var]])
  group_idx <- as.integer(group_factor)
  n_units <- nlevels(group_factor)

  # Get adjacency structure (may be stored as adj_matrix or adjacency)
  adj_matrix <- spatial$adj_matrix %||% spatial$adjacency
  if (is.null(adj_matrix)) {
    stop("Spatial structure must include adj_matrix or adjacency")
  }

  # Convert adjacency matrix to CSR format (0-based indexing for C++)
  n_neighbors <- integer(n_units)
  adj_row_ptr <- integer(n_units + 1)
  adj_col_idx <- integer(0)

  adj_row_ptr[1] <- 0L  # 0-based for C++
  for (i in seq_len(n_units)) {
    neighbors <- which(adj_matrix[i, ] != 0)
    n_neighbors[i] <- length(neighbors)
    adj_col_idx <- c(adj_col_idx, neighbors - 1L)  # Convert to 0-based for C++
    adj_row_ptr[i + 1] <- adj_row_ptr[i] + n_neighbors[i]
  }

  list(
    group_idx = group_idx,
    n_units = as.integer(n_units),
    n_neighbors = as.integer(n_neighbors),
    adj_row_ptr = as.integer(adj_row_ptr),
    adj_col_idx = as.integer(adj_col_idx)
  )
}


#' Compute Hessian at mode for spatial model
#' @keywords internal
compute_hessian_spatial <- function(y, n_trials, X, re_idx, n_re_groups,
                                     spatial_idx, n_spatial_units,
                                     adj_row_ptr, adj_col_idx, n_neighbors,
                                     mode, family, phi, sigma_re, tau_spatial) {
  N <- length(y)
  p <- ncol(X)
  n_x <- p + n_re_groups + n_spatial_units
  spatial_start <- p + n_re_groups

  # Compute eta at mode
  eta <- as.numeric(X %*% mode[1:p])
  if (n_re_groups > 0) {
    for (i in seq_len(N)) {
      g <- as.integer(re_idx[i])
      if (g > 0 && g <= n_re_groups) {
        eta[i] <- eta[i] + mode[p + g]
      }
    }
  }
  if (n_spatial_units > 0) {
    for (i in seq_len(N)) {
      s <- as.integer(spatial_idx[i])
      if (s > 0 && s <= n_spatial_units) {
        eta[i] <- eta[i] + mode[spatial_start + s]
      }
    }
  }

  # Build Hessian
  H <- matrix(0, n_x, n_x)
  tau_re <- 1 / (sigma_re^2 + 1e-10)

  # Likelihood contributions
  for (i in seq_len(N)) {
    if (family == "binomial") {
      p_i <- 1 / (1 + exp(-eta[i]))
      h_i <- n_trials[i] * p_i * (1 - p_i)
    } else if (family == "negbin") {
      mu_i <- exp(eta[i])
      h_i <- (y[i] + phi) * mu_i * phi / (mu_i + phi)^2
    } else {
      h_i <- exp(eta[i])
    }

    # Fixed effects block
    for (j in seq_len(p)) {
      for (k in seq_len(p)) {
        H[j, k] <- H[j, k] + h_i * X[i, j] * X[i, k]
      }
    }

    # Random effects
    if (n_re_groups > 0) {
      g <- as.integer(re_idx[i])
      if (g > 0 && g <= n_re_groups) {
        H[p + g, p + g] <- H[p + g, p + g] + h_i
        for (j in seq_len(p)) {
          H[j, p + g] <- H[j, p + g] + h_i * X[i, j]
          H[p + g, j] <- H[p + g, j] + h_i * X[i, j]
        }
      }
    }

    # Spatial effects
    if (n_spatial_units > 0) {
      s <- as.integer(spatial_idx[i])
      if (s > 0 && s <= n_spatial_units) {
        sp_idx <- spatial_start + s
        H[sp_idx, sp_idx] <- H[sp_idx, sp_idx] + h_i
        for (j in seq_len(p)) {
          H[j, sp_idx] <- H[j, sp_idx] + h_i * X[i, j]
          H[sp_idx, j] <- H[sp_idx, j] + h_i * X[i, j]
        }
        if (n_re_groups > 0) {
          g <- as.integer(re_idx[i])
          if (g > 0 && g <= n_re_groups) {
            H[p + g, sp_idx] <- H[p + g, sp_idx] + h_i
            H[sp_idx, p + g] <- H[sp_idx, p + g] + h_i
          }
        }
      }
    }
  }

  # Add prior precision for RE
  for (g in seq_len(n_re_groups)) {
    H[p + g, p + g] <- H[p + g, p + g] + tau_re
  }

  # Add ICAR precision for spatial
  # Note: adj_row_ptr and adj_col_idx use 0-based indexing (for C++ compatibility)
  for (s in seq_len(n_spatial_units)) {
    sp_idx <- spatial_start + s
    H[sp_idx, sp_idx] <- H[sp_idx, sp_idx] + tau_spatial * n_neighbors[s]
    # Convert 0-based CSR indices to R's 1-based indexing
    start_k <- adj_row_ptr[s] + 1L
    end_k <- adj_row_ptr[s + 1]
    if (end_k >= start_k) {
      for (k in start_k:end_k) {
        neighbor <- adj_col_idx[k] + 1L  # 0-based to 1-based
        nb_idx <- spatial_start + neighbor
        H[sp_idx, nb_idx] <- H[sp_idx, nb_idx] - tau_spatial
      }
    }
  }

  # Regularization for fixed effects
  for (j in seq_len(p)) {
    H[j, j] <- H[j, j] + 1e-4
  }

  list(H = H)
}


#' Convert spatial Laplace results to ratiod_fit
#' @keywords internal
convert_laplace_spatial_to_ratiod_fit <- function(samples, result, formula, data,
                                                   family, X, re_info, spatial_info,
                                                   n_samples) {
  p <- ncol(X)
  n_re <- re_info$n_groups
  n_spatial <- spatial_info$n_units
  spatial_start <- p + n_re

  # Extract fixed effects samples
  beta_names <- laplace_beta_names(X)

  draws_list <- list()
  for (j in seq_len(p)) {
    draws_list[[beta_names[j]]] <- samples[, j]
  }

  # Add spatial effects
  if (n_spatial > 0) {
    for (s in seq_len(n_spatial)) {
      draws_list[[paste0("spatial[", s, "]")]] <- samples[, spatial_start + s]
    }
  }

  draws <- do.call(cbind, draws_list)
  colnames(draws) <- names(draws_list)

  fit <- list(
    draws = draws,
    formula = formula,
    data = data,
    family = family,
    backend = "laplace",
    n_save = n_samples,
    laplace_result = result,
    fixed_gaussian = attr(samples, "fixed_gaussian"),
    .internal = list(
      mode = result$mode,
      log_marginal = result$log_marginal,
      X = X,
      re_info = re_info,
      spatial_info = spatial_info,
      samples = samples
    )
  )

  class(fit) <- "ratiod_fit"
  return(fit)
}


#' Fit BYM2 spatial model using Laplace approximation
#'
#' @param formula A ratiod_formula object
#' @param data Data frame
#' @param family Model family
#' @param family_type String family type for C++
#' @param spatial Spatial structure specification
#' @param spatial_info Prepared spatial info
#' @param y Response vector
#' @param n_trials Trials vector
#' @param X Design matrix
#' @param re_info Random effects info
#' @param phi Overdispersion parameter
#' @param sigma_re RE standard deviation
#' @param n_samples Number of posterior samples
#' @param cores Number of cores
#' @param verbose Print progress
#'
#' @return A ratiod_fit object
#' @keywords internal
fit_laplace_bym2 <- function(formula,
                              data,
                              family,
                              family_type,
                              spatial,
                              spatial_info,
                              y,
                              n_trials,
                              X,
                              re_info,
                              phi,
                              sigma_re,
                              n_samples,
                              cores,
                              verbose) {

  # Extract BYM2 hyperparameters
  sigma_spatial <- spatial$sigma %||% 1.0
  rho <- spatial$rho %||% 0.5
  scale_factor <- spatial$scale_factor %||% 1.0

  if (verbose) {
    message("Fitting BYM2 spatial model with Laplace approximation...")
    message(sprintf("  sigma_spatial: %.3f, rho: %.3f", sigma_spatial, rho))
  }

  # Fit with fixed hyperparameters
  if (verbose) {
    message("Finding mode of posterior...")
  }

  result <- cpp_laplace_fit_bym2(
    y = as.integer(y),
    n = as.integer(n_trials),
    X = X,
    re_idx = re_info$group_idx,
    n_re_groups = re_info$n_groups,
    sigma_re = sigma_re,
    spatial_idx = spatial_info$group_idx,
    n_spatial_units = spatial_info$n_units,
    adj_row_ptr = spatial_info$adj_row_ptr,
    adj_col_idx = spatial_info$adj_col_idx,
    n_neighbors = spatial_info$n_neighbors,
    sigma_spatial = sigma_spatial,
    rho = rho,
    scale_factor = scale_factor,
    family = family_type,
    phi = phi,
    n_threads = as.integer(cores)
  )

  if (!result$converged) {
    warning("Laplace approximation did not converge")
  }

  if (verbose) {
    message(sprintf("  Converged in %d iterations", result$n_iter))
    message("Sampling from Laplace approximation...")
  }

  # Compute Hessian at mode for sampling
  hess_result <- compute_hessian_bym2(
    y = y,
    n_trials = n_trials,
    X = X,
    re_idx = re_info$group_idx,
    n_re_groups = re_info$n_groups,
    spatial_idx = spatial_info$group_idx,
    n_spatial_units = spatial_info$n_units,
    adj_row_ptr = spatial_info$adj_row_ptr,
    adj_col_idx = spatial_info$adj_col_idx,
    n_neighbors = spatial_info$n_neighbors,
    mode = result$mode,
    family = family_type,
    phi = phi,
    sigma_re = sigma_re,
    sigma_spatial = sigma_spatial,
    rho = rho,
    scale_factor = scale_factor
  )

  # Sample from Laplace approximation
  samples <- sample_laplace_gaussian(
    mode = result$mode,
    H = hess_result$H,
    n_samples = as.integer(n_samples),
    X = X
  )

  # Convert to ratiod_fit
  fit <- convert_laplace_bym2_to_ratiod_fit(
    samples = samples,
    result = result,
    formula = formula,
    data = data,
    family = family,
    X = X,
    re_info = re_info,
    spatial_info = spatial_info,
    sigma_spatial = sigma_spatial,
    rho = rho,
    scale_factor = scale_factor,
    n_samples = n_samples
  )

  return(fit)
}


#' Compute Hessian at mode for BYM2 model
#' @keywords internal
compute_hessian_bym2 <- function(y, n_trials, X, re_idx, n_re_groups,
                                  spatial_idx, n_spatial_units,
                                  adj_row_ptr, adj_col_idx, n_neighbors,
                                  mode, family, phi, sigma_re,
                                  sigma_spatial, rho, scale_factor) {
  N <- length(y)
  p <- ncol(X)
  # Parameters: fixed effects + RE + phi_scaled + theta
  n_x <- p + n_re_groups + 2 * n_spatial_units
  phi_start <- p + n_re_groups
  theta_start <- phi_start + n_spatial_units

  sqrt_rho <- sqrt(rho + 1e-10)
  sqrt_1_rho <- sqrt(1.0 - rho + 1e-10)

  # Compute eta at mode
  eta <- as.numeric(X %*% mode[1:p])
  if (n_re_groups > 0) {
    for (i in seq_len(N)) {
      g <- as.integer(re_idx[i])
      if (g > 0 && g <= n_re_groups) {
        eta[i] <- eta[i] + mode[p + g]
      }
    }
  }
  if (n_spatial_units > 0) {
    for (i in seq_len(N)) {
      s <- as.integer(spatial_idx[i])
      if (s > 0 && s <= n_spatial_units) {
        phi_s <- mode[phi_start + s]
        theta_s <- mode[theta_start + s]
        spatial_effect <- sigma_spatial * (sqrt_rho * phi_s * scale_factor + sqrt_1_rho * theta_s)
        eta[i] <- eta[i] + spatial_effect
      }
    }
  }

  # Build Hessian
  H <- matrix(0, n_x, n_x)
  tau_re <- 1 / (sigma_re^2 + 1e-10)

  # Likelihood contributions
  for (i in seq_len(N)) {
    if (family == "binomial") {
      p_i <- 1 / (1 + exp(-eta[i]))
      h_i <- n_trials[i] * p_i * (1 - p_i)
    } else if (family == "negbin") {
      mu_i <- exp(eta[i])
      h_i <- (y[i] + phi) * mu_i * phi / (mu_i + phi)^2
    } else {
      h_i <- exp(eta[i])
    }

    # Fixed effects block
    for (j in seq_len(p)) {
      for (k in seq_len(p)) {
        H[j, k] <- H[j, k] + h_i * X[i, j] * X[i, k]
      }
    }

    # Random effects
    if (n_re_groups > 0) {
      g <- as.integer(re_idx[i])
      if (g > 0 && g <= n_re_groups) {
        H[p + g, p + g] <- H[p + g, p + g] + h_i
        for (j in seq_len(p)) {
          H[j, p + g] <- H[j, p + g] + h_i * X[i, j]
          H[p + g, j] <- H[p + g, j] + h_i * X[i, j]
        }
      }
    }

    # BYM2 spatial effects
    if (n_spatial_units > 0) {
      s <- as.integer(spatial_idx[i])
      if (s > 0 && s <= n_spatial_units) {
        phi_idx <- phi_start + s
        theta_idx <- theta_start + s

        coef_phi <- sigma_spatial * sqrt_rho * scale_factor
        coef_theta <- sigma_spatial * sqrt_1_rho

        # phi_scaled contributions
        H[phi_idx, phi_idx] <- H[phi_idx, phi_idx] + h_i * coef_phi^2
        # theta contributions
        H[theta_idx, theta_idx] <- H[theta_idx, theta_idx] + h_i * coef_theta^2
        # phi-theta cross
        H[phi_idx, theta_idx] <- H[phi_idx, theta_idx] + h_i * coef_phi * coef_theta
        H[theta_idx, phi_idx] <- H[theta_idx, phi_idx] + h_i * coef_phi * coef_theta

        # Cross with fixed effects
        for (j in seq_len(p)) {
          H[j, phi_idx] <- H[j, phi_idx] + h_i * X[i, j] * coef_phi
          H[phi_idx, j] <- H[phi_idx, j] + h_i * X[i, j] * coef_phi
          H[j, theta_idx] <- H[j, theta_idx] + h_i * X[i, j] * coef_theta
          H[theta_idx, j] <- H[theta_idx, j] + h_i * X[i, j] * coef_theta
        }
      }
    }
  }

  # Add prior precision for RE
  for (g in seq_len(n_re_groups)) {
    H[p + g, p + g] <- H[p + g, p + g] + tau_re
  }

  # Add ICAR precision for phi_scaled
  # Note: adj_row_ptr and adj_col_idx use 0-based indexing (for C++ compatibility)
  for (s in seq_len(n_spatial_units)) {
    phi_idx <- phi_start + s
    H[phi_idx, phi_idx] <- H[phi_idx, phi_idx] + n_neighbors[s]
    # Convert 0-based CSR indices to R's 1-based indexing
    start_k <- adj_row_ptr[s] + 1L
    end_k <- adj_row_ptr[s + 1]
    if (end_k >= start_k) {
      for (k in start_k:end_k) {
        neighbor <- adj_col_idx[k] + 1L  # 0-based to 1-based
        nb_idx <- phi_start + neighbor
        H[phi_idx, nb_idx] <- H[phi_idx, nb_idx] - 1.0
      }
    }
  }

  # Add N(0,1) prior precision for theta
  for (s in seq_len(n_spatial_units)) {
    theta_idx <- theta_start + s
    H[theta_idx, theta_idx] <- H[theta_idx, theta_idx] + 1.0
  }

  # Regularization for fixed effects
  for (j in seq_len(p)) {
    H[j, j] <- H[j, j] + 1e-4
  }

  list(H = H)
}


#' Fit GP spatial model using Laplace approximation
#'
#' @param formula A ratiod_formula object
#' @param data Data frame
#' @param family Model family
#' @param family_type String family type for C++
#' @param spatial GP spatial structure specification
#' @param y Response vector
#' @param n_trials Trials vector
#' @param X Design matrix
#' @param re_info Random effects info
#' @param phi Overdispersion parameter
#' @param sigma_re RE standard deviation
#' @param n_samples Number of posterior samples
#' @param cores Number of cores
#' @param verbose Print progress
#'
#' @return A ratiod_fit object
#' @keywords internal
fit_laplace_gp <- function(formula,
                           data,
                           family,
                           family_type,
                           spatial,
                           y,
                           n_trials,
                           X,
                           re_info,
                           phi,
                           sigma_re,
                           n_samples,
                           cores,
                           verbose) {

  if (verbose) {
    message("Fitting GP spatial model with Laplace approximation...")
  }

  # Validate GP specification against data
  spatial <- validate_gp(spatial, data)

  if (verbose) {
    message(sprintf("  Observations: %d", spatial$n_obs))
    message(sprintf("  Neighbors: %d", spatial$nn))
    message(sprintf("  Covariance: %s", spatial$cov))
  }

  # Get GP hyperparameters from priors or defaults
  sigma2_gp <- spatial$sigma2 %||% 1.0
  phi_gp <- spatial$phi %||% 1.0

  cov_type <- cov_type_code(spatial$cov)

  # Get neighbor info
  nn_info <- spatial$neighbor_info

  if (verbose) {
    message("Finding mode of posterior...")
  }

  result <- cpp_laplace_fit_gp(
    y = as.integer(y),
    n = as.integer(n_trials),
    X = X,
    re_idx = re_info$group_idx,
    n_re_groups = re_info$n_groups,
    sigma_re = sigma_re,
    coords = spatial$unique_coords,
    obs_to_loc = as.integer(spatial$obs_to_loc) - 1L,
    nn_idx = nn_info$nn_idx,
    nn_dist = nn_info$nn_dist,
    # The sampler entry reads nn_order 0-based; compute_nngp_neighbors()
    # returns it 1-based, the same conversion prepare_gp_for_hmc() leaves
    # to C++.
    nn_order = as.integer(nn_info$nn_order) - 1L,
    n_spatial = spatial$n_spatial,
    nn = spatial$nn,
    sigma2_gp = sigma2_gp,
    phi_gp = phi_gp,
    cov_type = cov_type,
    family = family_type,
    phi = phi,
    max_iter = 100L,
    tol = 1e-6,
    n_threads = as.integer(cores)
  )

  if (!result$converged) {
    warning("Laplace approximation did not converge")
  }

  if (verbose) {
    message(sprintf("  Converged in %d iterations", result$n_iter))
    message("Sampling from Laplace approximation...")
  }

  # Sample from Laplace approximation
  samples <- sample_laplace_gaussian(
    mode = result$mode,
    H = result$hessian,
    n_samples = as.integer(n_samples),
    X = X
  )

  # Convert to ratiod_fit
  fit <- convert_laplace_gp_to_ratiod_fit(
    samples = samples,
    result = result,
    formula = formula,
    data = data,
    family = family,
    X = X,
    re_info = re_info,
    spatial = spatial,
    n_samples = n_samples
  )

  return(fit)
}


#' Fit multi-scale GP spatial model using Laplace approximation
#'
#' @inheritParams fit_laplace_gp
#' @param spatial A `ratiod_multiscale` specification
#'
#' @return A ratiod_fit object
#' @keywords internal
fit_laplace_multiscale_gp <- function(formula,
                                      data,
                                      family,
                                      family_type,
                                      spatial,
                                      y,
                                      n_trials,
                                      X,
                                      re_info,
                                      phi,
                                      sigma_re,
                                      n_samples,
                                      cores,
                                      verbose) {

  if (verbose) {
    message("Fitting multi-scale GP spatial model with Laplace approximation...")
  }

  spatial <- validate_gp(spatial, data)

  local_info <- spatial$neighbor_info_local
  regional_info <- spatial$neighbor_info_regional

  # The Laplace backend conditions on fixed covariance hyperparameters. Each
  # scale takes the geometric centre of its range prior, the value equidistant
  # from both bounds on the log scale a range parameter lives on.
  phi_local <- spatial$phi_local %||% sqrt(prod(spatial$range_local))
  phi_regional <- spatial$phi_regional %||% sqrt(prod(spatial$range_regional))
  sigma2_local <- spatial$sigma2_local %||% 1.0
  sigma2_regional <- spatial$sigma2_regional %||% 1.0

  if (verbose) {
    message(sprintf("  Locations: %d", spatial$n_spatial))
    message(sprintf("  Neighbors: %d local, %d regional",
                    spatial$nn_local, spatial$nn_regional))
    message(sprintf("  Ranges held at: %.3g local, %.3g regional",
                    phi_local, phi_regional))
    message("Finding mode of posterior...")
  }

  result <- cpp_laplace_fit_multiscale_gp(
    y = as.integer(y),
    n = as.integer(n_trials),
    X = X,
    re_idx = re_info$group_idx,
    n_re_groups = re_info$n_groups,
    sigma_re = sigma_re,
    coords = spatial$unique_coords,
    obs_to_loc = as.integer(spatial$obs_to_loc) - 1L,
    nn_idx_local = local_info$nn_idx,
    nn_dist_local = local_info$nn_dist,
    nn_order_local = as.integer(local_info$nn_order) - 1L,
    nn_local = spatial$nn_local,
    nn_idx_regional = regional_info$nn_idx,
    nn_dist_regional = regional_info$nn_dist,
    nn_order_regional = as.integer(regional_info$nn_order) - 1L,
    nn_regional = spatial$nn_regional,
    n_spatial = spatial$n_spatial,
    sigma2_local = sigma2_local,
    phi_local = phi_local,
    sigma2_regional = sigma2_regional,
    phi_regional = phi_regional,
    cov_type = cov_type_code(spatial$cov),
    family = family_type,
    phi = phi,
    max_iter = 100L,
    tol = 1e-6,
    n_threads = as.integer(cores)
  )

  if (!result$converged) {
    warning("Laplace approximation did not converge")
  }

  if (verbose) {
    message(sprintf("  Converged in %d iterations", result$n_iter))
    message("Sampling from Laplace approximation...")
  }

  samples <- sample_laplace_gaussian(
    mode = result$mode,
    H = result$hessian,
    n_samples = as.integer(n_samples),
    X = X
  )

  convert_laplace_gp_to_ratiod_fit(
    samples = samples,
    result = result,
    formula = formula,
    data = data,
    family = family,
    X = X,
    re_info = re_info,
    spatial = spatial,
    n_samples = n_samples,
    field_names = c("w_local", "w_regional"),
    spatial_type = "multiscale"
  )
}


#' Convert GP Laplace results to ratiod_fit
#' @keywords internal
convert_laplace_gp_to_ratiod_fit <- function(samples, result, formula, data,
                                              family, X, re_info, spatial,
                                              n_samples,
                                              field_names = "w_gp",
                                              spatial_type = "gp") {
  p <- ncol(X)
  n_re <- re_info$n_groups
  n_spatial <- spatial$n_spatial
  spatial_start <- p + n_re

  # Extract fixed effects samples
  beta_names <- laplace_beta_names(X)

  draws_list <- list()
  for (j in seq_len(p)) {
    draws_list[[beta_names[j]]] <- samples[, j]
  }

  # One block of n_spatial columns per field, in the order the mode vector
  # lays them out. The summary carries the leading entries of each; the whole
  # block goes to .internal.
  n_show <- min(10, n_spatial)
  fields <- list()
  for (b in seq_along(field_names)) {
    block_start <- spatial_start + (b - 1L) * n_spatial
    for (s in seq_len(n_show)) {
      draws_list[[paste0(field_names[b], "[", s, "]")]] <- samples[, block_start + s]
    }
    fields[[field_names[b]]] <-
      samples[, (block_start + 1):(block_start + n_spatial), drop = FALSE]
  }

  draws <- do.call(cbind, draws_list)
  colnames(draws) <- names(draws_list)

  fit <- list(
    draws = draws,
    formula = formula,
    data = data,
    family = family,
    backend = "laplace",
    n_save = n_samples,
    laplace_result = result,
    fixed_gaussian = attr(samples, "fixed_gaussian"),
    spatial_type = spatial_type,
    .internal = c(
      list(
        mode = result$mode,
        log_marginal = result$log_marginal,
        X = X,
        re_info = re_info,
        spatial = spatial,
        samples = samples
      ),
      fields
    )
  )

  class(fit) <- "ratiod_fit"
  return(fit)
}


#' Fit multiscale temporal model using Laplace approximation
#'
#' @param formula A ratiod_formula object
#' @param data Data frame
#' @param family Model family
#' @param family_type String family type for C++
#' @param temporal Multiscale temporal structure specification
#' @param y Response vector
#' @param n_trials Trials vector
#' @param X Design matrix
#' @param re_info Random effects info
#' @param phi Overdispersion parameter
#' @param sigma_re RE standard deviation
#' @param n_samples Number of posterior samples
#' @param cores Number of cores
#' @param verbose Print progress
#'
#' @return A ratiod_fit object
#' @keywords internal
fit_laplace_temporal <- function(formula,
                                  data,
                                  family,
                                  family_type,
                                  temporal,
                                  y,
                                  n_trials,
                                  X,
                                  re_info,
                                  phi,
                                  sigma_re,
                                  n_samples,
                                  cores,
                                  verbose) {

  if (verbose) {
    message("Fitting multiscale temporal model with Laplace approximation...")
  }

  # Validate temporal specification
  temporal <- validate_temporal_multiscale(temporal, data)

  if (verbose) {
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

  # Default hyperparameters (could be made configurable)
  sigma2_trend <- 1.0
  sigma2_seasonal <- 1.0
  sigma2_short <- 1.0
  rho_short <- 0.5

  if (verbose) {
    message("Finding mode of posterior...")
  }

  result <- cpp_laplace_fit_multiscale_temporal(
    y = as.integer(y),
    n = as.integer(n_trials),
    X = X,
    re_idx = re_info$group_idx,
    n_re_groups = re_info$n_groups,
    sigma_re = sigma_re,
    time_idx = temporal$time_index,
    n_times = temporal$n_times,
    seasonal_period = as.integer(seasonal_period),
    trend_type = trend_type,
    short_type = short_type,
    sigma2_trend = sigma2_trend,
    sigma2_seasonal = sigma2_seasonal,
    sigma2_short = sigma2_short,
    rho_short = rho_short,
    family = family_type,
    phi = phi,
    max_iter = 100L,
    tol = 1e-6,
    n_threads = as.integer(cores)
  )

  if (!result$converged) {
    warning("Laplace approximation did not converge")
  }

  if (verbose) {
    message(sprintf("  Converged in %d iterations", result$n_iter))
    message("Sampling from Laplace approximation...")
  }

  # Sample from Laplace approximation
  samples <- sample_laplace_gaussian(
    mode = result$mode,
    H = result$hessian,
    n_samples = as.integer(n_samples),
    X = X
  )

  # Convert to ratiod_fit
  fit <- convert_laplace_temporal_to_ratiod_fit(
    samples = samples,
    result = result,
    formula = formula,
    data = data,
    family = family,
    X = X,
    re_info = re_info,
    temporal = temporal,
    n_samples = n_samples
  )

  return(fit)
}


#' Convert temporal Laplace results to ratiod_fit
#' @keywords internal
convert_laplace_temporal_to_ratiod_fit <- function(samples, result, formula, data,
                                                    family, X, re_info, temporal,
                                                    n_samples) {
  p <- ncol(X)
  n_re <- re_info$n_groups
  n_times <- temporal$n_times
  seasonal_period <- temporal$seasonal %||% 0

  # Parameter layout:
  # [0:p-1] = beta
  # [p:p+n_re-1] = random effects
  # [p+n_re:...] = temporal components (trend, seasonal, short_term)

  temporal_start <- p + n_re

  # Extract fixed effects samples
  beta_names <- laplace_beta_names(X)

  draws_list <- list()
  for (j in seq_len(p)) {
    draws_list[[beta_names[j]]] <- samples[, j]
  }

  # Track position for temporal components
  pos <- temporal_start

  # Extract temporal components
  temporal_draws <- list()

  if (temporal$trend != "none") {
    trend_samples <- samples[, (pos + 1):(pos + n_times), drop = FALSE]
    temporal_draws$trend <- trend_samples
    for (t in seq_len(n_times)) {
      draws_list[[paste0("trend[", t, "]")]] <- trend_samples[, t]
    }
    pos <- pos + n_times
  }

  if (seasonal_period > 0) {
    seasonal_samples <- samples[, (pos + 1):(pos + seasonal_period), drop = FALSE]
    temporal_draws$seasonal <- seasonal_samples
    for (s in seq_len(seasonal_period)) {
      draws_list[[paste0("seasonal[", s, "]")]] <- seasonal_samples[, s]
    }
    pos <- pos + seasonal_period
  }

  if (temporal$short_term != "none") {
    short_samples <- samples[, (pos + 1):(pos + n_times), drop = FALSE]
    temporal_draws$short_term <- short_samples
    for (t in seq_len(n_times)) {
      draws_list[[paste0("short_term[", t, "]")]] <- short_samples[, t]
    }
    pos <- pos + n_times
  }

  draws <- do.call(cbind, draws_list)
  colnames(draws) <- names(draws_list)

  fit <- list(
    draws = draws,
    formula = formula,
    data = data,
    family = family,
    backend = "laplace",
    n_save = n_samples,
    laplace_result = result,
    fixed_gaussian = attr(samples, "fixed_gaussian"),
    temporal = temporal,
    .internal = list(
      mode = result$mode,
      log_marginal = result$log_marginal,
      X = X,
      re_info = re_info,
      temporal_draws = temporal_draws,
      samples = samples
    )
  )

  class(fit) <- "ratiod_fit"
  return(fit)
}


#' Convert BYM2 Laplace results to ratiod_fit
#' @keywords internal
convert_laplace_bym2_to_ratiod_fit <- function(samples, result, formula, data,
                                                family, X, re_info, spatial_info,
                                                sigma_spatial, rho, scale_factor,
                                                n_samples) {
  p <- ncol(X)
  n_re <- re_info$n_groups
  n_spatial <- spatial_info$n_units
  phi_start <- p + n_re
  theta_start <- phi_start + n_spatial

  sqrt_rho <- sqrt(rho + 1e-10)
  sqrt_1_rho <- sqrt(1.0 - rho + 1e-10)

  # Extract fixed effects samples
  beta_names <- laplace_beta_names(X)

  draws_list <- list()
  for (j in seq_len(p)) {
    draws_list[[beta_names[j]]] <- samples[, j]
  }

  # Compute combined spatial effects u = sigma * (sqrt(rho) * phi * scale + sqrt(1-rho) * theta)
  if (n_spatial > 0) {
    for (s in seq_len(n_spatial)) {
      phi_s <- samples[, phi_start + s]
      theta_s <- samples[, theta_start + s]
      u_s <- sigma_spatial * (sqrt_rho * phi_s * scale_factor + sqrt_1_rho * theta_s)
      draws_list[[paste0("spatial[", s, "]")]] <- u_s
      draws_list[[paste0("phi_scaled[", s, "]")]] <- phi_s
      draws_list[[paste0("theta[", s, "]")]] <- theta_s
    }
  }

  # Add BYM2 hyperparameters as point estimates
  draws_list[["sigma_spatial"]] <- rep(sigma_spatial, n_samples)
  draws_list[["rho"]] <- rep(rho, n_samples)

  draws <- do.call(cbind, draws_list)
  colnames(draws) <- names(draws_list)

  fit <- list(
    draws = draws,
    formula = formula,
    data = data,
    family = family,
    backend = "laplace",
    n_save = n_samples,
    laplace_result = result,
    fixed_gaussian = attr(samples, "fixed_gaussian"),
    plug_in = c("sigma_spatial", "rho"),
    spatial_type = "bym2",
    .internal = list(
      mode = result$mode,
      log_marginal = result$log_marginal,
      X = X,
      re_info = re_info,
      spatial_info = spatial_info,
      sigma_spatial = sigma_spatial,
      rho = rho,
      scale_factor = scale_factor,
      samples = samples
    )
  )

  class(fit) <- "ratiod_fit"
  return(fit)
}


#' Fit Laplace model with RSR (Restricted Spatial Regression)
#'
#' @description
#' Laplace approximation with RSR to orthogonalize spatial effects
#' to covariates, preventing spatial confounding.
#'
#' @keywords internal
fit_laplace_rsr <- function(formula,
                            data,
                            family,
                            family_type,
                            spatial,
                            spatial_info,
                            y,
                            n_trials,
                            X,
                            re_info,
                            phi,
                            sigma_re,
                            tau_spatial,
                            n_samples,
                            cores,
                            verbose) {

  if (verbose) {
    message("Fitting RSR spatial model with Laplace approximation...")
  }

  # Validate and compute RSR projection matrix
  spatial <- validate_rsr(spatial, data, formula)

  if (is.null(spatial$rsr_projection)) {
    stop("Failed to compute RSR projection matrix", call. = FALSE)
  }

  rsr_projection <- spatial$rsr_projection
  rsr_n <- nrow(rsr_projection)

  if (verbose) {
    message(sprintf("  RSR projection dimension: %d x %d", rsr_n, rsr_n))
    message(sprintf("  Orthogonal to: %s", paste(spatial$rsr_vars, collapse = ", ")))
  }

  # Fit with RSR
  if (verbose) {
    message("Finding mode of posterior with RSR projection...")
  }

  result <- cpp_laplace_fit_rsr(
    y = as.integer(y),
    n = as.integer(n_trials),
    X = X,
    re_idx = re_info$group_idx,
    n_re_groups = re_info$n_groups,
    sigma_re = sigma_re,
    spatial_idx = spatial_info$group_idx,
    n_spatial_units = spatial_info$n_units,
    adj_row_ptr = spatial_info$adj_row_ptr,
    adj_col_idx = spatial_info$adj_col_idx,
    n_neighbors = spatial_info$n_neighbors,
    tau_spatial = tau_spatial,
    rsr_projection = as.vector(t(rsr_projection)),  # Row-major flatten
    rsr_n = as.integer(rsr_n),
    family = family_type,
    phi = phi,
    n_threads = as.integer(cores)
  )

  if (!result$converged) {
    warning("Laplace approximation with RSR did not converge")
  }

  if (verbose) {
    message(sprintf("  Converged in %d iterations", result$n_iter))
    message("Sampling from Laplace approximation...")
  }

  # Compute Hessian at mode for sampling
  hess_result <- compute_hessian_rsr(
    y = y,
    n_trials = n_trials,
    X = X,
    re_idx = re_info$group_idx,
    n_re_groups = re_info$n_groups,
    spatial_idx = spatial_info$group_idx,
    n_spatial_units = spatial_info$n_units,
    adj_row_ptr = spatial_info$adj_row_ptr,
    adj_col_idx = spatial_info$adj_col_idx,
    n_neighbors = spatial_info$n_neighbors,
    mode = result$mode,
    family = family_type,
    phi = phi,
    sigma_re = sigma_re,
    tau_spatial = tau_spatial,
    rsr_projection = rsr_projection
  )

  # Sample from Laplace approximation
  samples <- sample_laplace_gaussian(
    mode = result$mode,
    H = hess_result$H,
    n_samples = as.integer(n_samples),
    X = X
  )

  # Convert to ratiod_fit
  fit <- convert_laplace_rsr_to_ratiod_fit(
    samples = samples,
    result = result,
    formula = formula,
    data = data,
    family = family,
    X = X,
    re_info = re_info,
    spatial_info = spatial_info,
    rsr_projection = rsr_projection,
    n_samples = n_samples
  )

  return(fit)
}


#' Compute Hessian at mode for RSR model
#' @keywords internal
compute_hessian_rsr <- function(y, n_trials, X, re_idx, n_re_groups,
                                spatial_idx, n_spatial_units,
                                adj_row_ptr, adj_col_idx, n_neighbors,
                                mode, family, phi, sigma_re, tau_spatial,
                                rsr_projection) {
  N <- length(y)
  p <- ncol(X)
  n_x <- p + n_re_groups + n_spatial_units
  spatial_start <- p + n_re_groups

  # Compute projected spatial effects
  w_proj <- as.vector(rsr_projection %*% mode[(spatial_start + 1):n_x])

  # Compute eta at mode with projected spatial effects
  eta <- as.numeric(X %*% mode[1:p])
  if (n_re_groups > 0) {
    for (i in seq_len(N)) {
      g <- as.integer(re_idx[i])
      if (g > 0 && g <= n_re_groups) {
        eta[i] <- eta[i] + mode[p + g]
      }
    }
  }
  if (n_spatial_units > 0) {
    for (i in seq_len(N)) {
      s <- as.integer(spatial_idx[i])
      if (s > 0 && s <= n_spatial_units) {
        eta[i] <- eta[i] + w_proj[s]
      }
    }
  }

  # Build Hessian with RSR transformation
  H <- matrix(0, n_x, n_x)

  # Compute negative Hessian of log-likelihood at each observation
  h_diag <- numeric(N)
  for (i in seq_len(N)) {
    if (family == "binomial") {
      p_i <- 1 / (1 + exp(-eta[i]))
      h_diag[i] <- n_trials[i] * p_i * (1 - p_i)
    } else if (family == "negbin") {
      mu_i <- exp(eta[i])
      h_diag[i] <- (y[i] + phi) * mu_i * phi / (mu_i + phi)^2
    } else {
      h_diag[i] <- exp(eta[i])
    }
  }

  # Fixed effects block
  for (j in seq_len(p)) {
    for (k in seq_len(p)) {
      H[j, k] <- sum(h_diag * X[, j] * X[, k])
    }
  }

  # Random effects
  if (n_re_groups > 0) {
    for (i in seq_len(N)) {
      g <- as.integer(re_idx[i])
      if (g > 0 && g <= n_re_groups) {
        H[p + g, p + g] <- H[p + g, p + g] + h_diag[i]
        for (j in seq_len(p)) {
          H[j, p + g] <- H[j, p + g] + h_diag[i] * X[i, j]
          H[p + g, j] <- H[p + g, j] + h_diag[i] * X[i, j]
        }
      }
    }
  }

  # Spatial effects with RSR transformation
  # H_w = P' * diag(h_spatial) * P
  if (n_spatial_units > 0) {
    # First compute h for each spatial unit
    h_spatial <- numeric(n_spatial_units)
    for (i in seq_len(N)) {
      s <- as.integer(spatial_idx[i])
      if (s > 0 && s <= n_spatial_units) {
        h_spatial[s] <- h_spatial[s] + h_diag[i]
      }
    }

    # H_w = P' * diag(h_spatial) * P (P is symmetric, so P' = P)
    H_spatial <- t(rsr_projection) %*% diag(h_spatial) %*% rsr_projection
    H[(spatial_start + 1):n_x, (spatial_start + 1):n_x] <- H_spatial

    # Cross-terms with fixed effects: through P
    for (i in seq_len(N)) {
      s <- as.integer(spatial_idx[i])
      if (s > 0 && s <= n_spatial_units) {
        for (j in seq_len(p)) {
          for (k in seq_len(n_spatial_units)) {
            P_sk <- rsr_projection[s, k]
            H[j, spatial_start + k] <- H[j, spatial_start + k] + h_diag[i] * X[i, j] * P_sk
            H[spatial_start + k, j] <- H[spatial_start + k, j] + h_diag[i] * X[i, j] * P_sk
          }
        }
        # Cross-terms with random effects
        if (n_re_groups > 0) {
          g <- as.integer(re_idx[i])
          if (g > 0 && g <= n_re_groups) {
            for (k in seq_len(n_spatial_units)) {
              P_sk <- rsr_projection[s, k]
              H[p + g, spatial_start + k] <- H[p + g, spatial_start + k] + h_diag[i] * P_sk
              H[spatial_start + k, p + g] <- H[spatial_start + k, p + g] + h_diag[i] * P_sk
            }
          }
        }
      }
    }
  }

  # Add prior precision
  tau_re <- 1 / (sigma_re^2 + 1e-10)
  for (g in seq_len(n_re_groups)) {
    H[p + g, p + g] <- H[p + g, p + g] + tau_re
  }

  # ICAR prior on spatial effects
  # Note: adj_row_ptr and adj_col_idx use 0-based indexing (for C++ compatibility)
  if (n_spatial_units > 0) {
    for (s in seq_len(n_spatial_units)) {
      H[spatial_start + s, spatial_start + s] <- H[spatial_start + s, spatial_start + s] +
        tau_spatial * n_neighbors[s]

      # Convert 0-based CSR indices to R's 1-based indexing
      start_k <- adj_row_ptr[s] + 1L
      end_k <- adj_row_ptr[s + 1]
      if (end_k >= start_k) {
        for (k in start_k:end_k) {
          neighbor <- adj_col_idx[k] + 1L  # 0-based to 1-based
          H[spatial_start + s, spatial_start + neighbor] <-
            H[spatial_start + s, spatial_start + neighbor] - tau_spatial
        }
      }
    }
  }

  # Small regularization for fixed effects
  for (j in seq_len(p)) {
    H[j, j] <- H[j, j] + 1e-4
  }

  return(list(H = H))
}


#' Convert RSR Laplace results to ratiod_fit
#' @keywords internal
convert_laplace_rsr_to_ratiod_fit <- function(samples, result, formula, data,
                                               family, X, re_info, spatial_info,
                                               rsr_projection, n_samples) {
  p <- ncol(X)
  n_re <- re_info$n_groups
  n_spatial <- spatial_info$n_units
  spatial_start <- p + n_re

  # Extract fixed effects samples
  beta_names <- laplace_beta_names(X)

  draws_list <- list()
  for (j in seq_len(p)) {
    draws_list[[beta_names[j]]] <- samples[, j]
  }

  # Compute projected spatial effects for each sample
  if (n_spatial > 0) {
    for (s in seq_len(n_spatial)) {
      # Raw (unprojected) spatial effects
      draws_list[[paste0("spatial_raw[", s, "]")]] <- samples[, spatial_start + s]
    }

    # Compute projected spatial effects
    w_raw <- samples[, (spatial_start + 1):(spatial_start + n_spatial), drop = FALSE]
    w_proj <- t(rsr_projection %*% t(w_raw))

    for (s in seq_len(n_spatial)) {
      draws_list[[paste0("spatial[", s, "]")]] <- w_proj[, s]
    }
  }

  # Add hyperparameters as point estimates
  if (n_re > 0) {
    draws_list[["sigma_re"]] <- rep(1.0, n_samples)
  }

  draws <- do.call(cbind, draws_list)
  colnames(draws) <- names(draws_list)

  fit <- list(
    draws = draws,
    formula = formula,
    data = data,
    family = family,
    backend = "laplace",
    spatial_type = "rsr",
    n_save = n_samples,
    laplace_result = result,
    fixed_gaussian = attr(samples, "fixed_gaussian"),
    plug_in = if (n_re > 0) "sigma_re",
    .internal = list(
      mode = result$mode,
      log_marginal = result$log_marginal,
      X = X,
      re_info = re_info,
      spatial_info = spatial_info,
      rsr_projection = rsr_projection,
      samples = samples
    )
  )

  class(fit) <- "ratiod_fit"
  return(fit)
}
