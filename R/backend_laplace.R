#' Laplace Approximation Backend
#'
#' @description
#' Approximate inference on the nested-Laplace engine. The model is handed to
#' [tulpa::tulpa_nested_laplace_joint()] as one likelihood arm per process and
#' one latent block per random effect or structured field. The engine
#' integrates the hyperparameters of every block, and of every dispersion, over
#' an outer grid and solves the latent Gaussian at each grid cell, so the
#' fixed-effect posterior is the grid mixture rather than a Gaussian
#' conditioned on one value of each hyperparameter.
#'
#' @details
#' **Supported model structures:**
#' - Fixed effects
#' - Group-level random intercepts, crossed or nested terms (one `iid` block per
#'   term)
#' - Areal fields: ICAR, BYM2, proper CAR
#' - Gaussian-process fields (single scale, two scales) on a Hilbert-space basis
#' - Multi-scale temporal: trend (`rw1` / `rw2`), cyclic seasonal, short-term
#'   (`iid` / `ar1`)
#'
#' **Supported families:** binomial (one process), and the two-process families
#' `poisson_gamma`, `negbin_gamma`, `negbin_negbin` and `gamma_gamma`. A
#' two-process model is two likelihood arms, numerator and denominator, each
#' with its own fixed effects and dispersion, and every latent block enters both
#' arms.
#'
#' @name laplace_backend
#' @keywords internal
NULL


# One engine family per process. A process whose family carries a dispersion
# (a negative-binomial size, a gamma shape) integrates it on the outer grid.
LAPLACE_ARMS <- list(
  binomial      = list(num = "binomial"),
  poisson_gamma = list(num = "poisson", denom = "gamma"),
  negbin_gamma  = list(num = "neg_binomial_2", denom = "gamma"),
  negbin_negbin = list(num = "neg_binomial_2", denom = "neg_binomial_2"),
  gamma_gamma   = list(num = "gamma", denom = "gamma")
)

LAPLACE_DISPERSION <- c(neg_binomial_2 = "phi", gamma = "shape")


#' Fit model using Laplace approximation
#'
#' @param formula A ratiod_formula object
#' @param data Data frame
#' @param family Model family
#' @param spatial Optional spatial structure
#' @param temporal Optional multi-scale temporal structure. See
#'   `BACKEND_STRUCTURE_SUPPORT` for what this backend carries.
#' @param priors Prior specification. A `prior_pc()` or `prior_half_normal()` on
#'   `sigma` is carried to the random-effect standard deviations.
#' @param n_samples Number of posterior samples to draw
#' @param cores Number of threads for the engine
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

  model <- laplace_model(formula, data, family, spatial, temporal)

  if (verbose) {
    message(sprintf("Fitting %s model on the nested-Laplace engine (%d latent block%s)...",
                    model$arms$model_type, length(model$parts),
                    if (length(model$parts) == 1L) "" else "s"))
  }

  engine <- laplace_engine(model, priors, cores)

  if (verbose) message("Sampling the outer-grid mixture...")
  latent <- tulpa::tulpa_posterior_draws(engine, n = as.integer(n_samples))

  convert_laplace_fit(
    engine = engine, latent = latent, parts = model$parts, arms = model$arms,
    formula = formula, data = data, family = family,
    re_info = extract_re_for_laplace(formula), spatial = spatial,
    temporal = model$temporal, n_samples = n_samples
  )
}


# The arms and latent blocks a model hands to the engine.
laplace_model <- function(formula, data, family, spatial, temporal) {
  model_type <- get_hmc_model_type(family)
  arm_families <- LAPLACE_ARMS[[model_type]]
  if (is.null(arm_families)) {
    stop(sprintf("The Laplace backend does not fit the `%s` family. Use `mode = \"hmc\"`.",
                 model_type), call. = FALSE)
  }

  N <- length(formula$numerator$response)
  n_arms <- length(arm_families)

  parts <- laplace_re_parts(formula, first_block = 1L)
  temporal_spec <- NULL
  if (!is.null(spatial)) {
    parts <- c(parts, laplace_spatial_parts(
      spatial, data, formula, first_block = length(parts) + 1L, n_arms = n_arms
    ))
  }
  if (!is.null(temporal)) {
    built <- laplace_temporal_parts(temporal, data,
                                    first_block = length(parts) + 1L)
    parts <- c(parts, built$parts)
    temporal_spec <- built$temporal
  }
  list(parts = parts, arms = laplace_arms(formula, model_type),
       temporal = temporal_spec)
}


laplace_engine <- function(model, priors, cores) {
  engine <- tulpa::tulpa_nested_laplace_joint(
    responses = model$arms$responses,
    prior = lapply(model$parts, `[[`, "block"),
    phi_grid = model$arms$phi_grid,
    prior_sigma = laplace_sigma_prior(priors$sigma, model$parts),
    control = list(store_Q = TRUE, progress = FALSE,
                   integration = laplace_integration(model$parts),
                   n_threads = as.integer(cores %||% 1L))
  )

  unconverged <- engine$nonconverged_mass %||% 0
  if (unconverged > 0) {
    warning(sprintf(
      "The inner Laplace solve did not converge in outer-grid cells that carry %.1f%% of the posterior mass",
      100 * unconverged), call. = FALSE)
  }
  engine
}


# A central composite design needs every axis to have a support the engine can
# transform; the correlation of an `ar1` or proper-CAR block has none, and a
# model carrying one integrates on the adaptive lattice instead.
laplace_integration <- function(parts) {
  if (all(vapply(parts, `[[`, logical(1), "ccd_axes"))) "ccd" else "grid_adaptive"
}


# Likelihood arms of the engine call, with the dispersion axes of the arms that
# carry one.
laplace_arms <- function(formula, model_type) {
  spec <- LAPLACE_ARMS[[model_type]]
  N <- length(formula$numerator$response)
  binomial <- identical(model_type, "binomial")

  responses <- list(num = list(
    y = as.numeric(formula$numerator$response),
    n_trials = if (binomial) as.integer(formula$denominator$response)
               else rep(1L, N),
    X = formula$numerator$X,
    family = spec$num,
    phi = 1
  ))
  if (!is.null(spec$denom)) {
    responses$denom <- list(
      y = as.numeric(formula$denominator$response),
      n_trials = rep(1L, N),
      X = formula$denominator$X,
      family = spec$denom,
      phi = 1
    )
  }

  nodes <- exp(seq(log(0.1), log(200), length.out = 9L))
  phi_grid <- list()
  for (a in names(responses)) {
    if (responses[[a]]$family %in% names(LAPLACE_DISPERSION)) {
      phi_grid[[a]] <- tulpa::auto_grid(nodes)
    }
  }

  list(responses = responses, phi_grid = if (length(phi_grid)) phi_grid,
       model_type = model_type)
}


# A prior_pc() or prior_half_normal() on sigma is the engine's own pair of
# hyperprior families; it is attached to each random-effect block, keyed by
# block. Any other prior leaves the engine's default in place and says so.
laplace_sigma_prior <- function(prior, parts) {
  if (is.null(prior) || !inherits(prior, "ratiod_prior")) return(NULL)

  spec <- switch(
    prior$distribution,
    pc = list("pc.prec", c(U = prior$U, alpha = prior$alpha)),
    half_normal = list("half_normal", prior$sd),
    NULL
  )
  if (is.null(spec)) {
    warning(sprintf(
      "The Laplace backend carries `prior_pc()` and `prior_half_normal()` on sigma; the `%s` prior is not used and the engine's default applies.",
      prior$distribution), call. = FALSE)
    return(NULL)
  }

  re <- which(vapply(parts, function(p) identical(p$role, "re"), logical(1)))
  if (length(re) == 0L) return(NULL)
  lapply(re, function(k) list(block = k, prior = spec))
}


#' Prepare spatial structure for the Laplace backend
#' @keywords internal
prepare_spatial_for_laplace <- function(spatial, data, formula) {
  group_var <- spatial$group_var
  if (is.null(group_var)) {
    stop(sprintf(
      paste0("Areal spatial structures need `group_var`; `%s` carries coordinates ",
             "instead and is fitted by its own Laplace path."),
      spatial$type %||% "car"), call. = FALSE)
  }

  if (!group_var %in% names(data)) {
    stop(sprintf("Spatial group variable '%s' not found in data", group_var))
  }

  group_factor <- as.factor(data[[group_var]])
  group_idx <- as.integer(group_factor)
  n_units <- nlevels(group_factor)

  adj_matrix <- spatial$adj_matrix %||% spatial$adjacency
  if (is.null(adj_matrix)) {
    stop("Spatial structure must include adj_matrix or adjacency")
  }

  # CSR with 0-based column indices
  n_neighbors <- integer(n_units)
  adj_row_ptr <- integer(n_units + 1)
  adj_col_idx <- integer(0)

  adj_row_ptr[1] <- 0L
  for (i in seq_len(n_units)) {
    neighbors <- which(adj_matrix[i, ] != 0)
    n_neighbors[i] <- length(neighbors)
    adj_col_idx <- c(adj_col_idx, neighbors - 1L)
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


#' Column names of the fixed effects of one process
#' @keywords internal
laplace_beta_names <- function(X, prefix = NULL) {
  if (!is.null(prefix)) return(paste0(prefix, "[", seq_len(ncol(X)), "]"))
  colnames(X) %||% paste0("beta[", seq_len(ncol(X)), "]")
}


#' Extract RE info for Laplace
#'
#' @description
#' Random-effect structure of the numerator formula: one entry per term, with
#' the group index and the number of groups. A random slope is not carried and
#' is reported.
#'
#' @param formula A ratiod_formula object
#' @return List with the RE structure
#' @keywords internal
extract_re_for_laplace <- function(formula) {
  re_terms <- formula$numerator$random_effects
  n_obs <- length(formula$numerator$response)

  if (is.null(re_terms) || length(re_terms) == 0) {
    return(list(
      group_idx = as.numeric(rep(1, n_obs)),
      n_groups = 0L,
      group_var = NULL,
      n_re_terms = 0L,
      re_terms = list(),
      has_slopes = FALSE
    ))
  }

  n_re_terms <- length(re_terms)

  has_slopes <- FALSE
  for (term in re_terms) {
    if (length(term$slope_vars) > 0) {
      has_slopes <- TRUE
      warning(
        "Random slopes are not carried by the Laplace backend. Slope terms are ignored: ",
        paste(term$slope_vars, collapse = ", "),
        "\nOnly random intercepts (1 | group) are fitted.",
        call. = FALSE
      )
    }
  }

  if (n_re_terms > 1) {
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

    group_idx_matrix <- matrix(0, nrow = n_obs, ncol = n_re_terms)
    for (t in seq_len(n_re_terms)) {
      group_idx_matrix[, t] <- as.numeric(re_terms[[t]]$group)
    }

    return(list(
      group_idx = as.numeric(re_terms[[1]]$group),
      n_groups = as.integer(re_terms[[1]]$n_groups),
      group_var = re_terms[[1]]$group_var,
      n_re_terms = n_re_terms,
      re_terms = re_terms_processed,
      group_idx_matrix = group_idx_matrix,
      total_groups = total_groups,
      has_slopes = has_slopes
    ))
  }

  re_info <- re_terms[[1]]
  list(
    group_idx = as.numeric(re_info$group),
    n_groups = as.integer(re_info$n_groups),
    group_var = re_info$group_var,
    n_re_terms = 1L,
    re_terms = list(list(
      group_var = re_info$group_var,
      group_idx = as.numeric(re_info$group),
      n_groups = as.integer(re_info$n_groups),
      offset = 0L
    )),
    total_groups = as.integer(re_info$n_groups),
    has_slopes = has_slopes
  )
}


#' Assemble a ratiod_fit from an engine fit and its latent draws
#'
#' The engine's draws are on its own latent scale. Each part turns its slice
#' into the effect that enters the linear predictor, with the hyperparameter
#' values of the grid cell each draw came from, so the stored draws hold the
#' fixed effects followed by the effect of every part.
#'
#' @keywords internal
convert_laplace_fit <- function(engine, latent, parts, arms, formula, data,
                                family, re_info, spatial, temporal, n_samples) {
  layout <- engine$arm_layout
  responses <- arms$responses
  two_process <- length(responses) > 1L

  n_draw <- nrow(latent)
  theta_cell <- engine$theta_grid[attr(latent, "cells"), , drop = FALSE]

  p_arm <- as.integer(layout$p)
  beta_from <- cumsum(c(0L, p_arm))[seq_along(p_arm)]
  beta_cols <- lapply(seq_along(p_arm), function(a) beta_from[a] + seq_len(p_arm[a]))
  beta <- do.call(cbind, lapply(beta_cols, function(cols) latent[, cols, drop = FALSE]))

  block_start <- as.integer(layout$block_start)
  block_size <- as.integer(layout$block_size)
  effects <- lapply(seq_along(parts), function(b) {
    Z <- latent[, block_start[b] + seq_len(block_size[b]), drop = FALSE]
    effect <- parts[[b]]$effect(Z, theta_cell)
    if (ncol(effect) != parts[[b]]$n) {
      stop(sprintf("Latent block %d (%s) holds %d effects, expected %d.",
                   b, parts[[b]]$label, ncol(effect), parts[[b]]$n),
           call. = FALSE)
    }
    effect
  })

  samples <- cbind(beta, do.call(cbind, effects))
  colnames(samples) <- NULL

  n_beta <- ncol(beta)
  col_to <- n_beta + cumsum(vapply(parts, function(p) p$n, integer(1)))
  col_from <- c(n_beta, col_to[-length(col_to)])
  layout_out <- list(
    beta = stats::setNames(beta_cols, names(responses)),
    parts = lapply(seq_along(parts), function(b) {
      list(role = parts[[b]]$role, label = parts[[b]]$label,
           cols = col_from[b] + seq_len(parts[[b]]$n),
           index = parts[[b]]$index)
    })
  )

  fixed_names <- if (two_process) {
    c(laplace_beta_names(responses$num$X, "beta_num"),
      laplace_beta_names(responses$denom$X, "beta_denom"))
  } else {
    laplace_beta_names(responses$num$X)
  }

  draws_list <- stats::setNames(
    lapply(seq_len(n_beta), function(j) samples[, j]), fixed_names
  )

  hyper_names <- laplace_hyper_names(parts, responses)
  theta_draw <- attr(latent, "theta")
  for (axis in intersect(names(hyper_names), colnames(theta_draw))) {
    draws_list[[hyper_names[[axis]]]] <- theta_draw[, axis]
  }

  fields <- list()
  for (b in seq_along(parts)) {
    part <- parts[[b]]
    if (length(part$columns) == 0L) next
    eff <- effects[[b]]
    for (s in seq_len(min(part$n_show, ncol(eff)))) {
      draws_list[[sprintf("%s[%d]", part$columns, s)]] <- eff[, s]
    }
    fields[[part$columns]] <- eff
  }

  draws <- do.call(cbind, draws_list)
  colnames(draws) <- names(draws_list)

  spatial_part <- Filter(function(p) !is.null(p$spatial_info), parts)
  temporal_draws <- fields[intersect(c("trend", "seasonal", "short_term"),
                                     names(fields))]

  internal <- c(
    list(
      samples = samples,
      layout = layout_out,
      X = responses$num$X,
      X_num = responses$num$X,
      X_denom = if (two_process) responses$denom$X,
      log_marginal = engine$log_evidence,
      converged = !(engine$nonconverged_mass %||% 0 > 0),
      re_info = re_info,
      model_type = arms$model_type
    ),
    if (length(spatial_part)) list(spatial_info = spatial_part[[1L]]$spatial_info),
    if (length(temporal_draws)) list(temporal_draws = temporal_draws),
    if (!is.null(spatial)) list(spatial = laplace_spatial_record(spatial, data)),
    fields[intersect(c("w_gp", "w_local", "w_regional"), names(fields))]
  )

  keep <- setdiff(names(engine), c("Q_csc_p_per_grid", "Q_csc_i_per_grid",
                                    "Q_csc_x_per_grid"))
  fit <- list(
    draws = draws,
    formula = formula,
    data = data,
    family = family,
    backend = "laplace",
    n_save = n_draw,
    engine = engine[keep],
    fixed_columns = fixed_names,
    laplace_result = list(converged = !(engine$nonconverged_mass %||% 0 > 0),
                          log_marginal = engine$log_evidence),
    spatial_type = if (!is.null(spatial)) spatial$type %||% "car",
    temporal = temporal,
    .internal = internal
  )
  class(fit$engine) <- class(engine)

  class(fit) <- "ratiod_fit"
  fit
}


# Reported name of every hyperparameter axis: block axes by part, dispersion
# axes by the process they belong to.
laplace_hyper_names <- function(parts, responses) {
  names_by_axis <- unlist(lapply(parts, `[[`, "hyper"))
  gamma_arms <- names(responses)[vapply(responses, function(r) identical(r$family, "gamma"),
                                        logical(1))]
  for (a in names(responses)) {
    fam <- responses[[a]]$family
    if (!fam %in% names(LAPLACE_DISPERSION)) next
    stem <- if (identical(fam, "gamma") && length(gamma_arms) == 1L) "shape"
            else paste0(LAPLACE_DISPERSION[[fam]], "_", if (a == "num") "num" else "denom")
    names_by_axis[[paste0("phi_", a)]] <- stem
  }
  names_by_axis
}


# What a prediction needs of the spatial specification: the validated field for
# coordinate-based structures, the specification itself otherwise.
laplace_spatial_record <- function(spatial, data) {
  if (inherits(spatial, c("ratiod_gp", "ratiod_multiscale", "ratiod_hsgp"))) {
    return(validate_gp(spatial, data))
  }
  spatial
}


# Linear predictor of the first process at one outer-grid cell, rebuilt from the
# cell's latent mode through the parts' own transforms. The engine reports the
# same quantity as `fitted_eta`, which is what ties the transforms to its
# latent scale.
laplace_cell_eta <- function(engine, model, cell) {
  layout <- engine$arm_layout
  mode <- engine$modes[cell, ]
  theta <- engine$theta_grid[cell, , drop = FALSE]
  X <- model$arms$responses[[1L]]$X

  eta <- as.numeric(X %*% mode[seq_len(layout$p[1L])])
  for (b in seq_along(model$parts)) {
    part <- model$parts[[b]]
    Z <- matrix(mode[as.integer(layout$block_start[b]) + seq_len(layout$block_size[b])],
                nrow = 1L)
    effect <- part$effect(Z, theta)
    eta <- eta + as.numeric(spread_effect(effect, part$index, length(eta)))
  }
  eta
}
