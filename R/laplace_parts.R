#' Latent blocks of the Laplace tier
#'
#' @description
#' The Laplace tier hands a model to the nested-Laplace engine
#' ([tulpa::tulpa_nested_laplace_joint()]) as one likelihood arm per process and
#' one latent block per random effect or structured field. A *part* is a block
#' together with what the tier needs to read its latent draws back: the
#' per-observation unit index, the transform from the engine's latent scale to
#' the linear predictor, and the names the hyperparameter axes are reported
#' under.
#'
#' The latent vector of a joint fit is the fixed effects of every arm followed by
#' the blocks in order. A block's latent coordinates are unit-precision: an
#' `iid` block enters the predictor as `sigma * z`, a `bym2` block as
#' `sigma * (sqrt(rho) * s * a + sqrt(1 - rho) * b)` with the scale factor `s`, an `hsgp` block through
#' `sqrt(S_j) * z_j` with the squared-exponential spectral density `S`, whose
#' amplitude and lengthscale axes are held on the log scale; the
#' precision of `icar`, `car_proper`, `rw1`, `rw2` and `ar1` blocks carries their
#' scale, so their coordinates are the effect.
#'
#' @name laplace_parts
#' @keywords internal
NULL


# A part: `block` is the engine block, `n` the number of effect columns it
# reports, `index` the unit each observation reads, `effect(Z, theta)` the
# transform of an `n_draw x block_size` latent slice (with the outer-grid cell
# values `theta`, columns named `b<k>.<axis>`) to an `n_draw x n` effect matrix,
# and `hyper` the engine axis -> reported name map.
laplace_part <- function(role, label, block, n, index, effect, hyper,
                         columns = character(0), n_show = n) {
  list(role = role, label = label, block = block, n = as.integer(n),
       index = as.integer(index), effect = effect, hyper = hyper,
       columns = columns, n_show = as.integer(n_show),
       ccd_axes = !block$type %in% c("ar1", "car_proper"))
}


# The effect is the latent coordinate itself.
laplace_identity_effect <- function(Z, theta) Z


# Latent coordinates of a scale-free block are multiplied by the cell's sigma.
laplace_sigma_effect <- function(axis) {
  force(axis)
  function(Z, theta) Z * theta[, axis]
}


laplace_re_parts <- function(formula, first_block) {
  terms <- formula$numerator$random_effects
  if (is.null(terms) || length(terms) == 0L) return(list())

  multi <- length(terms) > 1L
  lapply(seq_along(terms), function(t) {
    term <- terms[[t]]
    k <- first_block + t - 1L
    axis <- paste0("b", k, ".sigma")
    name <- if (multi) sprintf("sigma_re[%s]", term$group_var) else "sigma_re"
    laplace_part(
      role = "re", label = term$group_var,
      block = list(type = "iid", obs_idx = as.integer(term$group),
                   n_units = as.integer(term$n_groups)),
      n = term$n_groups, index = term$group,
      effect = laplace_sigma_effect(axis),
      hyper = stats::setNames(name, axis)
    )
  })
}


# Areal field. Under restricted spatial regression the field reaches the
# observations as `P S z`, with `S` the observation-to-unit incidence and `P`
# the projection onto the orthogonal complement of the restricted covariates;
# the engine reads that matrix as the block's projector, and the part reports
# the projected effect per observation.
laplace_areal_parts <- function(spatial, spatial_info, first_block,
                                projection = NULL) {
  type <- spatial$type %||% "car"
  k <- first_block
  n <- spatial_info$n_units
  N <- length(spatial_info$group_idx)
  adjacency <- list(
    n_spatial_units = n,
    adj_row_ptr = spatial_info$adj_row_ptr,
    adj_col_idx = spatial_info$adj_col_idx,
    n_neighbors = spatial_info$n_neighbors
  )

  if (identical(tolower(type), "bym2")) {
    sf <- spatial$scale_factor %||% 1.0
    block <- list(type = "bym2", scale_factor = sf)
    effect <- function(Z, theta) {
      sigma <- theta[, paste0("b", k, ".sigma")]
      rho <- theta[, paste0("b", k, ".rho")]
      sigma * (sqrt(rho) * sf * Z[, seq_len(n), drop = FALSE] +
               sqrt(1 - rho) * Z[, n + seq_len(n), drop = FALSE])
    }
    hyper <- stats::setNames(c("sigma_spatial", "rho"),
                             paste0("b", k, c(".sigma", ".rho")))
  } else if (identical(type, "car_proper")) {
    bounds <- spatial$rho_bounds
    block <- list(type = "car_proper",
                  rho_bounds = c(bounds[["lower"]], bounds[["upper"]]))
    effect <- laplace_identity_effect
    hyper <- stats::setNames(c("tau_spatial", "rho"),
                             paste0("b", k, c(".tau", ".rho")))
  } else {
    block <- list(type = "icar")
    effect <- laplace_identity_effect
    hyper <- stats::setNames("tau_spatial", paste0("b", k, ".tau"))
  }

  if (is.null(projection)) {
    return(list(laplace_part(
      role = "spatial", label = block$type,
      block = c(block, list(spatial_idx = spatial_info$group_idx), adjacency),
      n = n, index = spatial_info$group_idx, effect = effect, hyper = hyper,
      columns = "spatial"
    )))
  }

  incidence <- matrix(0, N, n)
  incidence[cbind(seq_len(N), spatial_info$group_idx)] <- 1
  A <- projection %*% incidence
  list(laplace_part(
    role = "spatial", label = block$type,
    block = c(block, list(projector = A), adjacency),
    n = N, index = seq_len(N),
    effect = function(Z, theta) effect(Z, theta) %*% t(A),
    hyper = hyper, columns = "spatial", n_show = 10L
  ))
}


# Projection of a restricted spatial regression, or NULL for an unrestricted
# field.
laplace_rsr_projection <- function(spatial, data) {
  if (!inherits(spatial, "ratiod_rsr")) return(NULL)
  validate_rsr(spatial, data)$rsr_projection
}


# Coordinates a Hilbert-space basis is built on. A specification that is itself
# a Hilbert-space field carries one basis row per observation, as the HMC fit
# does; a neighbour-based specification is evaluated at its unique locations.
laplace_field_coords <- function(spatial, data) {
  on_observations <- inherits(spatial, "ratiod_hsgp") ||
    (inherits(spatial, "ratiod_multiscale") && identical(spatial$approx, "hsgp"))
  if (on_observations) {
    coords <- validate_hsgp(spatial, data)$coords_matrix
    return(list(coords = coords, obs_to_loc = seq_len(nrow(coords))))
  }
  validated <- validate_gp(spatial, data)
  list(coords = validated$unique_coords, obs_to_loc = validated$obs_to_loc)
}


# Squared-exponential HSGP block. Observation i reads basis row `obs_to_loc[i]`;
# under restricted spatial regression the observation rows are projected, and
# the part reports the projected effect per observation.
laplace_hsgp_part <- function(k, label, columns, basis, obs_to_loc, n_arms,
                              lengthscale = NULL, projection = NULL) {
  lam <- basis$eigenvalues
  Phi <- basis$phi
  Phi_obs <- Phi[obs_to_loc, , drop = FALSE]
  if (!is.null(projection)) {
    Phi_obs <- projection %*% Phi_obs
    Phi <- Phi_obs
    obs_to_loc <- seq_len(nrow(Phi_obs))
  }
  block <- list(type = "hsgp", m_total = basis$m_total,
                phi = rep(list(Phi_obs), n_arms),
                n_obs_per_arm = rep(nrow(Phi_obs), n_arms),
                eigenvalues = lam)
  if (!is.null(lengthscale)) block$lengthscale_grid <- lengthscale

  laplace_part(
    role = "spatial", label = label, block = block,
    n = nrow(Phi), index = obs_to_loc,
    effect = function(Z, theta) {
      sigma2 <- exp(theta[, paste0("b", k, ".sigma2")])
      ell2 <- exp(2 * theta[, paste0("b", k, ".lengthscale")])
      sqrt_S <- sqrt(sigma2 * 2 * pi * ell2 * exp(-0.5 * outer(ell2, lam)))
      (Z * sqrt_S) %*% t(Phi)
    },
    hyper = stats::setNames(
      paste0(c("sigma2_", "lengthscale_"), label),
      paste0("b", k, c(".sigma2", ".lengthscale"))
    ),
    columns = columns, n_show = 10L
  )
}


laplace_gp_parts <- function(spatial, data, first_block, n_arms,
                             projection = NULL) {
  field <- laplace_field_coords(spatial, data)
  basis <- hsgp_basis_2d(field$coords, m = as.integer(spatial$m %||% 6L),
                         c_boundary = spatial$c %||% 1.5)
  list(laplace_hsgp_part(first_block, "gp", "w_gp", basis, field$obs_to_loc,
                         n_arms, projection = projection))
}


laplace_multiscale_gp_parts <- function(spatial, data, first_block, n_arms,
                                        projection = NULL) {
  field <- laplace_field_coords(spatial, data)
  basis <- hsgp_basis_2d(field$coords, m = as.integer(spatial$m %||% 10L),
                         c_boundary = spatial$c_boundary %||% 1.5)
  scales <- list(local = spatial$range_local, regional = spatial$range_regional)
  lapply(seq_along(scales), function(s) {
    range <- scales[[s]]
    laplace_hsgp_part(
      first_block + s - 1L, names(scales)[s], paste0("w_", names(scales)[s]),
      basis, field$obs_to_loc, n_arms,
      lengthscale = exp(seq(log(range[1L]), log(range[2L]), length.out = 4L)),
      projection = projection
    )
  })
}


laplace_spatial_parts <- function(spatial, data, formula, first_block, n_arms) {
  projection <- laplace_rsr_projection(spatial, data)
  if (inherits(spatial, "ratiod_multiscale")) {
    return(laplace_multiscale_gp_parts(spatial, data, first_block, n_arms,
                                       projection))
  }
  if (inherits(spatial, c("ratiod_gp", "ratiod_hsgp"))) {
    return(laplace_gp_parts(spatial, data, first_block, n_arms, projection))
  }
  spatial_info <- prepare_spatial_for_laplace(spatial, data, formula)
  parts <- laplace_areal_parts(spatial, spatial_info, first_block, projection)
  parts[[1L]]$spatial_info <- spatial_info
  parts
}


laplace_temporal_parts <- function(temporal, data, first_block) {
  temporal <- validate_temporal_multiscale(temporal, data)
  n_times <- temporal$n_times
  time_index <- temporal$time_index
  parts <- list()
  k <- first_block - 1L

  add <- function(part) parts[[length(parts) + 1L]] <<- part

  if (temporal$trend != "none") {
    k <- k + 1L
    add(laplace_part(
      role = "temporal", label = "trend",
      block = list(type = temporal$trend, temporal_idx = time_index,
                   n_times = n_times),
      n = n_times, index = time_index, effect = laplace_identity_effect,
      hyper = stats::setNames("tau_trend", paste0("b", k, ".tau")),
      columns = "trend", n_show = n_times
    ))
  }

  period <- temporal$seasonal %||% 0L
  if (period > 0L) {
    k <- k + 1L
    season <- ((time_index - 1L) %% period) + 1L
    add(laplace_part(
      role = "temporal", label = "seasonal",
      block = list(type = "rw1", temporal_idx = season, n_times = period,
                   cyclic = TRUE),
      n = period, index = season, effect = laplace_identity_effect,
      hyper = stats::setNames("tau_seasonal", paste0("b", k, ".tau")),
      columns = "seasonal", n_show = period
    ))
  }

  if (temporal$short_term != "none") {
    k <- k + 1L
    if (identical(temporal$short_term, "ar1")) {
      add(laplace_part(
        role = "temporal", label = "short_term",
        block = list(type = "ar1", temporal_idx = time_index,
                     n_times = n_times),
        n = n_times, index = time_index, effect = laplace_identity_effect,
        hyper = stats::setNames(c("tau_short", "rho_short"),
                                paste0("b", k, c(".tau", ".rho"))),
        columns = "short_term", n_show = n_times
      ))
    } else {
      add(laplace_part(
        role = "temporal", label = "short_term",
        block = list(type = "iid", obs_idx = time_index, n_units = n_times),
        n = n_times, index = time_index,
        effect = laplace_sigma_effect(paste0("b", k, ".sigma")),
        hyper = stats::setNames("sigma_short", paste0("b", k, ".sigma")),
        columns = "short_term", n_show = n_times
      ))
    }
  }

  list(parts = parts, temporal = temporal)
}
