#' Spatiotemporal Interaction Specifications
#'
#' @description
#' Functions to specify spatiotemporal interaction effects for ratio models.
#' These capture dependencies that arise when spatial patterns vary over time,
#' or when temporal trends differ across space.
#'
#' @details
#' Spatiotemporal interactions extend the basic additive model:
#'
#' \deqn{\eta_{st} = X\beta + f_s(space) + f_t(time)}
#'
#' to include interactions:
#'
#' \deqn{\eta_{st} = X\beta + f_s(space) + f_t(time) + \delta_{st}}
#'
#' where \eqn{\delta_{st}} captures space-time interactions.
#'
#' **Interaction Types (following Knorr-Held, 2000):**
#'
#' - **Type I**: Unstructured interaction - IID \eqn{\delta_{st} \sim N(0, \sigma^2)}
#' - **Type II**: Structured time, unstructured space - temporal structure at each location
#' - **Type III**: Structured space, unstructured time - spatial structure at each time
#' - **Type IV**: Structured space AND time - full Kronecker interaction
#'
#' **Separable Models:**
#'
#' - **Separable**: Covariance is Kronecker product \eqn{C_{st} = C_s \otimes C_t}
#' - **Non-separable**: GP with joint space-time metric
#'
#' @name spatiotemporal
#' @references
#' Knorr-Held, L. (2000). Bayesian modelling of inseparable space-time variation
#' in disease risk. Statistics in Medicine, 19(17-18), 2555-2567.
#'
#' @keywords models
NULL


#' Spatiotemporal interaction specification
#'
#' @description
#' Specify a spatiotemporal interaction effect for ratio models.
#' The interaction captures structured or unstructured deviation from
#' the additive spatial + temporal model.
#'
#' @param spatial A spatial specification from [spatial_car()], [spatial_bym2()],
#'   or [spatial_gp()].
#' @param temporal A temporal specification from [temporal_rw1()], [temporal_rw2()],
#'   [temporal_ar1()], or [temporal_gp()].
#' @param type Interaction type:
#'   - `"I"` or `"iid"`: Unstructured interaction (IID)
#'   - `"II"`: Structured time at each location
#'   - `"III"`: Structured space at each time point
#'   - `"IV"`: Fully structured (Kronecker product of spatial and temporal)
#'   - `"separable"`: Separable space-time GP, the product covariance
#'     `C_s(h) C_t(u)` over the grid
#'   - `"nonsep_gp"`: Non-separable space-time GP, see `nonsep_type`
#' @param shared Logical; if TRUE (default), spatiotemporal effect enters both
#'   numerator and denominator. Set to FALSE for process-specific effects
#'   (triggers warning about potential confounding).
#' @param parameterization With the HMC backend: `"centered"` (default)
#'   samples the interaction directly; `"noncentered"` samples a standard
#'   normal `z` and reconstructs the interaction from it, as `z / sqrt(tau)`
#'   for the Knorr-Held types (`"I"` to `"IV"`) and through the
#'   nearest-neighbour GP autoregression, `delta_i = m_i + sqrt(d_i) z_i`, for
#'   the GP types. When the data carry little or no space-time interaction,
#'   the posterior for `tau` drifts toward large values and the centered
#'   field's conditional scale shrinks with it (a funnel), which NUTS cannot
#'   span with a single step size; `"noncentered"` decouples the two and should
#'   be tried first when `rhat` for the interaction (or an intercept sharing
#'   its geometry) will not drop with more warmup. An HSGP spatial margin is
#'   sampled centered. Ignored outside the HMC backend.
#' @param nonsep_type For `type = "nonsep_gp"`: `"product"` (the separable
#'   covariance) or `"gneiting"` (Gneiting 2002). An additive covariance is not
#'   among them: on a complete `S x T` grid its rank is `S + T - 1`, and the
#'   directions it drops are the space-time interactions the block carries.
#' @param coords Required for `"separable"` and `"nonsep_gp"`: a formula
#'   (`~ lon + lat`) or a character vector of two column names, giving the
#'   coordinates the interaction's covariance is built on. Where a spatial
#'   unit's rows carry different positions, the unit's coordinate is their
#'   centroid. The interaction's geometry is its own, so an areal spatial main
#'   effect pairs with a continuous interaction.
#' @param cov_space,cov_time For the GP types: the spatial and temporal
#'   kernels, one of `"exponential"` (default), `"matern"`, `"gaussian"`,
#'   `"spherical"`. `phi` is a lengthscale in all four.
#' @param nn For the GP types: NNGP neighbours per grid cell (default 15,
#'   capped at `S * T - 1`), chosen on a joint space-time metric.
#'
#' @return A `ratiod_spatiotemporal` object
#'
#' @details
#' **Type I (IID)**
#'
#' Independent random effect for each space-time combination:
#' \deqn{\delta_{st} \stackrel{iid}{\sim} N(0, \sigma^2_\delta)}
#'
#' This is the simplest form, requiring S*T parameters but capturing no
#' structured interaction.
#'
#' **Type II (Temporal structure per location)**
#'
#' Each location has its own temporal random effect:
#' \deqn{\delta_{\cdot t}^{(s)} \sim RW(\sigma^2)}
#'
#' This captures location-specific temporal trends but assumes independence
#' across locations.
#'
#' **Type III (Spatial structure per time point)**
#'
#' Each time point has its own spatial random effect:
#' \deqn{\delta_{s \cdot}^{(t)} \sim ICAR(\tau)}
#'
#' This captures time-specific spatial patterns but assumes independence
#' across time points.
#'
#' **Type IV (Full structure)**
#'
#' Kronecker product of spatial and temporal precision matrices:
#' \deqn{Q_\delta = Q_s \otimes Q_t}
#'
#' This is the most constrained model, assuming the interaction has the
#' same structure as the marginal effects.
#'
#' **Separable**
#'
#' For GP-based effects, assumes separable covariance:
#' \deqn{C(\mathbf{s}_1, t_1; \mathbf{s}_2, t_2) = C_s(\mathbf{s}_1, \mathbf{s}_2) \cdot C_t(t_1, t_2)}
#'
#' @examples
#' # Create adjacency matrix for 10 regions
#' adj <- matrix(0, 10, 10)
#' for (i in 1:9) adj[i, i+1] <- adj[i+1, i] <- 1
#'
#' # Type I: Unstructured interaction
#' st1 <- spatiotemporal(
#'   spatial = spatial_car(adj, level = "group", group_var = "region"),
#'   temporal = temporal_rw1("year"),
#'   type = "I"
#' )
#' print(st1)
#'
#' # Type IV: Fully structured interaction
#' st4 <- spatiotemporal(
#'   spatial = spatial_car(adj, level = "group", group_var = "region"),
#'   temporal = temporal_rw1("year"),
#'   type = "IV"
#' )
#' print(st4)
#'
#' \donttest{
#' # Generate synthetic spatiotemporal data
#' set.seed(123)
#' n_regions <- 10
#' n_years <- 8
#' df <- expand.grid(
#'   region = 1:n_regions,
#'   year = 2015:(2015 + n_years - 1)
#' )
#' df$x <- rnorm(nrow(df))
#' df$count <- rpois(nrow(df), lambda = 20)
#' df$effort <- rgamma(nrow(df), shape = 4, rate = 1)
#'
#' # Fit model with spatiotemporal interaction
#' fit <- tratio(
#'   count | effort ~ x,
#'   data = df,
#'   family = ratiod_poisson_gamma(),
#'   spatiotemporal = spatiotemporal(
#'     spatial = spatial_car(adj, level = "group", group_var = "region"),
#'     temporal = temporal_rw1("year"),
#'     type = "IV"
#'   ),
#'   control = list(iter = 400, chains = 1, seed = 1)
#' )
#' summary(fit)
#' }
#'
#' @seealso [spatial_car()], [spatial_gp()], [temporal_rw1()], [temporal_ar1()]
#'
#' @export
spatiotemporal <- function(spatial,
                           temporal,
                           type = c("I", "II", "III", "IV", "iid", "separable",
                                    "nonsep_gp"),
                           shared = TRUE,
                           parameterization = c("centered", "noncentered"),
                           nonsep_type = c("product", "gneiting"),
                           coords = NULL,
                           cov_space = NULL,
                           cov_time = NULL,
                           nn = NULL) {

  # Validate spatial specification

  if (!inherits(spatial, c("ratiod_spatial", "ratiod_gp", "ratiod_multiscale"))) {
    stop("`spatial` must be a tulpaRatio spatial specification ",
         "(from spatial_car, spatial_bym2, spatial_gp, etc.)", call. = FALSE)
  }

  # Validate temporal specification
  if (!inherits(temporal, "ratiod_temporal")) {
    stop("`temporal` must be a tulpaRatio temporal specification ",
         "(from temporal_rw1, temporal_rw2, temporal_ar1, temporal_gp, etc.)",
         call. = FALSE)
  }

  # Normalize type
  type <- match.arg(type)
  if (type == "iid") type <- "I"
  parameterization <- match.arg(parameterization)
  nonsep_type <- match.arg(nonsep_type)
  is_gp_type <- type %in% c("separable", "nonsep_gp")

  # The GP types put a continuous covariance on the space-time grid, so the
  # interaction carries its own coordinates. They are the interaction's, not
  # the spatial main effect's: an areal main effect and a continuous
  # interaction over the same units is the usual combination, and tying the
  # two would make the interaction's geometry a property of how the main
  # effect happens to be parameterized.
  coord_vars <- NULL
  if (is_gp_type) {
    if (type == "separable" && nonsep_type != "product") {
      stop("type = \"separable\" IS the product covariance. ",
           "For nonsep_type = \"", nonsep_type, "\" use type = \"nonsep_gp\".",
           call. = FALSE)
    }
    if (type == "separable") nonsep_type <- "product"

    coord_vars <- st_parse_coord_vars(coords, type)

    cov_space <- cov_space %||% "exponential"
    cov_time <- cov_time %||% "exponential"
    # Refuse an unknown name here rather than at the .Call boundary, where the
    # argument it came from is no longer nameable.
    cov_type_code(cov_space)
    cov_type_code(cov_time)
    nn <- as.integer(nn %||% 15L)
    if (is.na(nn) || nn < 1L) {
      stop("`nn` must be a positive integer", call. = FALSE)
    }
  }

  # Check for Kronecker interaction with proper CAR
  if (type == "IV" && inherits(spatial, "ratiod_spatial")) {
    if (!is.null(spatial$proper) && spatial$proper) {
      warning(
        "Type IV interaction with proper CAR may have identifiability issues.\n",
        "Consider using ICAR (proper = FALSE) instead.",
        call. = FALSE
      )
    }
  }

  # Warning for non-shared effects
  if (!shared) {
    warning(
      "Non-shared spatiotemporal effects (shared = FALSE) may lead to confounded ratio estimates.\n",
      "Consider whether space-time interactions should be shared between\n",
      "numerator and denominator to prevent spurious patterns in ratios.",
      call. = FALSE
    )
  }

  structure(
    list(
      type = type,
      spatial = spatial,
      temporal = temporal,
      shared = shared,
      parameterization = parameterization,
      # Read by the GP types only
      nonsep_type = if (is_gp_type) nonsep_type else NULL,
      coord_vars = coord_vars,
      cov_space = cov_space,
      cov_time = cov_time,
      nn = nn,
      # Filled in during validation
      n_spatial = NULL,
      n_times = NULL,
      n_params = NULL,
      spatial_group_var = spatial$group_var,
      temporal_time_var = temporal$time_var
    ),
    class = c("ratiod_spatiotemporal", "list")
  )
}


#' @rdname spatiotemporal
#' @param x A ratiod_spatiotemporal object
#' @param ... Ignored
#'
#' @export
print.ratiod_spatiotemporal <- function(x, ...) {
  cat("tulpaRatio Spatiotemporal Interaction Specification\n")
  cat("===============================================\n\n")

  type_desc <- switch(x$type,
    "I" = "Type I: Unstructured (IID)",
    "II" = "Type II: Structured time, unstructured space",
    "III" = "Type III: Structured space, unstructured time",
    "IV" = "Type IV: Fully structured (Kronecker)",
    "separable" = "Separable space-time GP (product covariance)",
    "nonsep_gp" = paste0("Non-separable space-time GP (", x$nonsep_type, ")")
  )
  cat("Interaction type:", type_desc, "\n\n")

  cat("Spatial component:\n")
  cat("  Type:", class(x$spatial)[1], "\n")
  if (!is.null(x$spatial$group_var)) {
    cat("  Group variable:", x$spatial$group_var, "\n")
  }
  if (!is.null(x$n_spatial)) {
    cat("  Spatial units:", x$n_spatial, "\n")
  }

  cat("\nTemporal component:\n")
  cat("  Type:", x$temporal$type, "\n")
  cat("  Time variable:", x$temporal$time_var, "\n")
  if (!is.null(x$n_times)) {
    cat("  Time points:", x$n_times, "\n")
  }

  cat("\nShared:", if (x$shared) "Yes (enters both processes)" else "No", "\n")

  if (!is.null(x$n_params)) {
    cat("Total interaction parameters:", x$n_params, "\n")
  }

  invisible(x)
}


#' Validate spatiotemporal specification against data
#'
#' @param st ratiod_spatiotemporal object
#' @param data Data frame
#'
#' @return Updated ratiod_spatiotemporal object with validated components
#' @keywords internal
validate_spatiotemporal <- function(st, data) {
  if (is.null(st)) return(NULL)

  # Validate spatial component
  if (inherits(st$spatial, "ratiod_hsgp")) {
    # HSGP spatial: validate coords, set n_spatial = m^2 (basis functions)
    st$spatial_is_hsgp <- TRUE
    coord_vars <- st$spatial$coord_vars
    coords <- as.matrix(data[, coord_vars, drop = FALSE])
    st$spatial$coords_matrix <- coords
    st$spatial$n_obs <- nrow(coords)
    m <- st$spatial$m %||% 6L
    st$n_spatial <- as.integer(m)^2  # m^2 basis functions replace S spatial units
  } else if (inherits(st$spatial, "ratiod_gp")) {
    st$spatial <- validate_gp(st$spatial, data)
    st$n_spatial <- st$spatial$n_spatial
  } else if (inherits(st$spatial, "ratiod_multiscale")) {
    st$spatial <- validate_gp(st$spatial, data)
    st$n_spatial <- st$spatial$n_spatial
  } else {
    validate_spatial(st$spatial, data)
    st$n_spatial <- st$spatial$n_spatial
  }

  # Validate temporal component
  if (inherits(st$temporal, "ratiod_temporal_gp")) {
    st$temporal <- validate_temporal_gp(st$temporal, data)
    st$n_times <- st$temporal$n_times
  } else if (inherits(st$temporal, "ratiod_temporal_multiscale")) {
    st$temporal <- validate_temporal_multiscale(st$temporal, data)
    st$n_times <- st$temporal$n_times
  } else {
    st$temporal <- validate_temporal(st$temporal, data)
    st$n_times <- st$temporal$n_times
  }

  # Compute number of interaction parameters based on type
  S <- st$n_spatial
  T <- st$n_times

  st$n_params <- switch(st$type,
    "I" = S * T,                    # IID: one per space-time combo
    "II" = S * T,                   # Temporal structure per location
    "III" = S * T,                  # Spatial structure per time point
    "IV" = S * T,                   # Fully structured
    "separable" = S * T,            # Separable space-time GP
    "nonsep_gp" = S * T             # Non-separable space-time GP
  )

  # Build space-time indexing
  st$st_index <- build_st_index(st, data)

  # The GP types carry a covariance over that grid, so they carry its geometry
  # and the NNGP neighbour sets built on it.
  if (st$type %in% c("separable", "nonsep_gp")) {
    st$gp_grid <- build_st_gp_grid(st, data)
  }

  st
}


#' The space-time grid a GP interaction is defined on
#'
#' One entry per grid cell, in the same flattening `st_flat` carries
#' (`k = (s - 1) * T + t`): the spatial unit's coordinates repeated across its
#' times, and the distinct time values repeated across units. Both axes are
#' standardized, since the neighbour metric adds a spatial distance to a
#' temporal one and the two ranges share a prior.
#'
#' @param st Validated `ratiod_spatiotemporal` object of a GP type.
#' @param data Data frame the coordinate columns are read from.
#'
#' @return List with the grid geometry and its NNGP neighbour sets.
#' @keywords internal
build_st_gp_grid <- function(st, data) {
  S <- st$n_spatial
  T <- st$n_times

  coords_unit <- st_unit_coords(st, data, S)
  times <- st_grid_time_values(st$temporal, T)

  # Standardize each axis. validate_gp() may already have scaled the
  # coordinates, in which case this is a no-op up to rounding.
  coords_unit <- st_scale_axis(coords_unit)
  times <- as.numeric(st_scale_axis(matrix(times, ncol = 1L)))

  coords_grid <- coords_unit[rep(seq_len(S), each = T), , drop = FALSE]
  time_grid <- rep(times, times = S)

  n_grid <- S * T
  nn <- min(as.integer(st$nn %||% 15L), n_grid - 1L)
  if (nn < 1L) {
    stop("Spatiotemporal GP: the ", S, " x ", T, " grid has too few cells ",
         "to build a neighbour set on.", call. = FALSE)
  }
  nb <- compute_st_nngp_neighbors(coords_grid, time_grid, nn)

  list(
    nn = nn,
    coords = coords_grid,
    time_values = time_grid,
    nn_idx = nb$nn_idx,
    nn_dist_space = nb$nn_dist_space,
    nn_dist_time = nb$nn_dist_time,
    nn_order = nb$nn_order,
    nn_order_inv = nb$nn_order_inv
  )
}


#' The coordinate columns a GP interaction is built on
#'
#' A formula (`~ lon + lat`) or a character vector of two column names.
#' @keywords internal
st_parse_coord_vars <- function(coords, type) {
  if (is.null(coords)) {
    stop("type = \"", type, "\" is a Gaussian process over the space-time ",
         "grid, so it needs coordinates: pass coords = ~ lon + lat. ",
         "For an adjacency-based interaction use type = \"IV\".",
         call. = FALSE)
  }
  if (inherits(coords, "formula")) {
    coord_vars <- all.vars(coords)
  } else if (is.character(coords)) {
    coord_vars <- coords
  } else {
    stop("`coords` must be a formula (~ lon + lat) or a character vector of ",
         "two column names.", call. = FALSE)
  }
  if (length(coord_vars) != 2L) {
    stop("`coords` must name exactly 2 coordinate variables, not ",
         length(coord_vars), ".", call. = FALSE)
  }
  coord_vars
}


#' One coordinate pair per spatial unit
#'
#' The interaction is indexed by unit, so each unit needs one position. Where a
#' unit's rows carry different positions -- point observations inside an areal
#' unit -- that position is their centroid, and saying so is the point of the
#' message.
#' @keywords internal
st_unit_coords <- function(st, data, S) {
  cv <- st$coord_vars
  missing_cols <- setdiff(cv, names(data))
  if (length(missing_cols) > 0) {
    stop("Coordinate variable(s) not found in data: ",
         paste(missing_cols, collapse = ", "), call. = FALSE)
  }
  xy <- cbind(as.numeric(data[[cv[1]]]), as.numeric(data[[cv[2]]]))
  if (anyNA(xy)) {
    stop("Coordinate columns contain missing values", call. = FALSE)
  }
  s_idx <- st$st_index$s_idx
  if (length(s_idx) != nrow(xy)) {
    stop("Spatiotemporal GP: ", length(s_idx), " spatial indices for ",
         nrow(xy), " rows of coordinates.", call. = FALSE)
  }

  out <- matrix(NA_real_, S, 2L)
  spread <- 0
  for (j in 1:2) {
    out[, j] <- as.numeric(tapply(xy[, j], factor(s_idx, levels = seq_len(S)),
                                  mean))
    rng <- tapply(xy[, j], factor(s_idx, levels = seq_len(S)),
                  function(z) if (length(z) > 1L) diff(range(z)) else 0)
    spread <- max(spread, max(as.numeric(rng), na.rm = TRUE))
  }
  if (anyNA(out)) {
    stop("Spatiotemporal GP: ", sum(!stats::complete.cases(out)),
         " of ", S, " spatial units have no coordinates in the data.",
         call. = FALSE)
  }
  if (spread > 0) {
    message(sprintf(
      "Spatiotemporal GP: coordinates vary within spatial unit (up to %.4g); using unit centroids.",
      spread))
  }
  out
}


#' Centre and scale one axis, leaving a constant axis alone
#' @keywords internal
st_scale_axis <- function(x) {
  x <- as.matrix(x)
  for (j in seq_len(ncol(x))) {
    s <- stats::sd(x[, j])
    x[, j] <- x[, j] - mean(x[, j])
    if (is.finite(s) && s > 0) x[, j] <- x[, j] / s
  }
  x
}


#' The distinct time values a temporal component indexes
#'
#' `validate_temporal()` keeps the factor levels it built the time index from,
#' and `validate_temporal_gp()` keeps the values themselves. A time variable
#' whose levels are not numeric has no spacing to read, so its positions stand
#' in for it.
#' @keywords internal
st_grid_time_values <- function(temporal, T) {
  tv <- temporal$unique_time_values
  if (is.null(tv)) {
    tv <- suppressWarnings(as.numeric(temporal$time_levels))
  }
  if (length(tv) != T || anyNA(tv)) tv <- seq_len(T)
  as.numeric(tv)
}


#' Build space-time index mapping
#'
#' @param st Validated spatiotemporal object
#' @param data Data frame
#'
#' @return List with space-time indexing information
#' @keywords internal
build_st_index <- function(st, data) {
  N <- nrow(data)
  S <- st$n_spatial
  T <- st$n_times

  # Get spatial index for each observation
  if (isTRUE(st$spatial_is_hsgp)) {
    # HSGP-ST: no spatial grouping, spatial mapping via Phi basis matrix
    s_idx <- rep(1L, N)  # placeholder — C++ uses Phi instead
  } else if (!is.null(st$spatial$group_var)) {
    spatial_var <- st$spatial$group_var
    if (!(spatial_var %in% names(data))) {
      stop(sprintf("Spatial group variable '%s' not found in data", spatial_var),
           call. = FALSE)
    }
    s_vals <- data[[spatial_var]]
    if (is.factor(s_vals)) {
      s_idx <- as.integer(s_vals)
    } else {
      s_factor <- as.factor(s_vals)
      s_idx <- as.integer(s_factor)
    }
  } else if (!is.null(st$spatial$obs_to_loc)) {
    # Spatial GP: n_spatial counts UNIQUE locations, so the interaction's
    # spatial index is the observation's location, not its row. seq_len(N)
    # here runs past the S x T grid the moment two observations share a
    # location, and st_flat indexes the interaction with it.
    s_idx <- as.integer(st$spatial$obs_to_loc)
  } else {
    # Observation-level spatial with one location per row
    s_idx <- seq_len(N)
  }

  # Get temporal index for each observation
  time_var <- st$temporal$time_var
  if (!(time_var %in% names(data))) {
    stop(sprintf("Time variable '%s' not found in data", time_var),
         call. = FALSE)
  }
  t_vals <- data[[time_var]]
  if (is.factor(t_vals)) {
    t_idx <- as.integer(t_vals)
  } else {
    unique_times <- sort(unique(t_vals))
    t_factor <- factor(t_vals, levels = unique_times)
    t_idx <- as.integer(t_factor)
  }

  # Compute flattened index: st_flat[n] = (s_idx[n] - 1) * T + t_idx[n]
  # This gives column-major ordering (s varies slowest)
  st_flat <- (s_idx - 1L) * T + t_idx

  list(
    s_idx = s_idx,
    t_idx = t_idx,
    st_flat = st_flat,
    N = N,
    S = S,
    T = T
  )
}


#' Prepare spatiotemporal data for HMC backend
#'
#' @param st Validated ratiod_spatiotemporal object
#' @param data Data frame
#'
#' @return List with C++ compatible spatiotemporal data
#' @keywords internal
prepare_spatiotemporal_for_hmc <- function(st, data) {
  if (is.null(st)) {
    return(list(
      has_spatiotemporal = FALSE,
      type = "none"
    ))
  }

  idx <- st$st_index
  S <- idx$S
  T <- idx$T

  # Prepare spatial precision structure
  spatial_Q_info <- prepare_spatial_precision(st$spatial)

  # Prepare temporal precision structure
  temporal_Q_info <- prepare_temporal_precision(st$temporal)

  result <- list(
    has_spatiotemporal = TRUE,
    type = st$type,
    shared = st$shared,
    parameterization = st$parameterization %||% "centered",
    n_spatial = S,
    n_times = T,
    n_params = st$n_params,

    # Observation indexing
    s_idx = idx$s_idx,
    t_idx = idx$t_idx,
    st_flat = idx$st_flat,

    # Spatial structure
    spatial_type = class(st$spatial)[1],
    spatial_Q = spatial_Q_info,

    # Temporal structure
    temporal_type = st$temporal$type,
    rho_prior = st$temporal$rho_prior,
    temporal_Q = temporal_Q_info,
    temporal_cyclic = isTRUE(st$temporal$cyclic)
  )

  # GP interaction fields: the grid geometry, its neighbour sets, and the two
  # kernels. Matrices are transposed on the way out because C++ reads them
  # row-major over (grid cell, neighbour).
  if (st$type %in% c("separable", "nonsep_gp")) {
    g <- st$gp_grid
    result$gp <- list(
      nn = as.integer(g$nn),
      coords = as.numeric(t(g$coords)),
      time_values = as.numeric(g$time_values),
      nn_idx = as.integer(t(g$nn_idx)),
      nn_dist_space = as.numeric(t(g$nn_dist_space)),
      nn_dist_time = as.numeric(t(g$nn_dist_time)),
      nn_order = as.integer(g$nn_order),
      nn_order_inv = as.integer(g$nn_order_inv),
      cov_space = cov_type_code(st$cov_space),
      cov_time = cov_type_code(st$cov_time),
      nonsep_type = st$nonsep_type %||% "product"
    )
  }

  # HSGP-ST fields
  if (isTRUE(st$spatial_is_hsgp)) {
    result$spatial_is_hsgp <- TRUE
    result$hsgp_m <- st$spatial$m %||% 6L
    result$hsgp_c <- st$spatial$c %||% 1.5
    result$hsgp_coords <- st$spatial$coords_matrix
    result$hsgp_scale_coords <- st$spatial$scale_coords %||% TRUE
  }

  result
}


#' Does a spatiotemporal interaction estimate a time-margin correlation?
#'
#' Type II applies the temporal precision within each spatial unit, Type IV
#' carries it as a Kronecker margin and HSGP-ST reads it per basis function, so
#' all three estimate an AR1 `rho`. Type I is iid over the whole grid and Type
#' III's time margin is unstructured, so neither has one; the GP types carry a
#' range instead. This mirrors `st_time_margin_is_structured()` in
#' `src/hmc_spatiotemporal.h`, which the parameter layout reads -- a parameter
#' allocated on one side and not the other shifts every index after it.
#'
#' @param st_info Output of [prepare_spatiotemporal_for_hmc()].
#'
#' @return `TRUE` if the interaction carries a `logit_rho_st` parameter.
#' @keywords internal
st_has_rho <- function(st_info) {
  if (is.null(st_info)) return(FALSE)
  if (!identical(st_info$temporal_type %||% "rw1", "ar1")) return(FALSE)
  (st_info$type %||% "none") %in% c("II", "IV") ||
    isTRUE(st_info$spatial_is_hsgp)
}


#' Prepare spatial precision matrix info
#'
#' @param spatial ratiod_spatial object
#'
#' @return List with spatial precision structure
#' @keywords internal
prepare_spatial_precision <- function(spatial) {
  if (is.null(spatial)) return(NULL)

  if (inherits(spatial, "ratiod_hsgp")) {
    # HSGP: no adjacency or neighbor structure needed — spatial precision is spectral
    return(list(
      type = "hsgp",
      n = spatial$m^2,
      adj_row_ptr = integer(0),
      adj_col_idx = integer(0)
    ))
  }

  if (inherits(spatial, c("ratiod_gp", "ratiod_multiscale"))) {
    # GP: return neighbor info for NNGP
    list(
      type = "gp",
      n = spatial$n_spatial,
      nn = spatial$nn,
      coords = as.vector(t(spatial$coords_matrix)),
      nn_idx = as.vector(t(spatial$neighbor_info$nn_idx)),
      nn_dist = as.vector(t(spatial$neighbor_info$nn_dist)),
      nn_order = spatial$neighbor_info$nn_order,
      nn_order_inv = spatial$neighbor_info$nn_order_inv
    )
  } else {
    # CAR/BYM2: return adjacency structure
    adj <- as.matrix(spatial$adjacency)
    n <- nrow(adj)

    # Build CSR format (0-based row_ptr, 1-based col_idx — C++ does -1)
    row_ptr <- integer(n + 1)  # initialized to 0 (0-based CSR)
    col_idx <- integer(0)

    for (i in seq_len(n)) {
      neighbors <- which(adj[i, ] > 0)
      col_idx <- c(col_idx, neighbors)
      row_ptr[i + 1] <- row_ptr[i] + length(neighbors)
    }

    n_neighbors <- diff(row_ptr)

    # Compute Q_inv and L_Q for precision mass matrix
    mass <- ridged_precision_factors(diag(n_neighbors) - adj)

    list(
      type = if (spatial$type == "bym2") "bym2" else "car",
      n = n,
      adj_row_ptr = row_ptr,
      adj_col_idx = col_idx,
      n_neighbors = n_neighbors,
      proper = isTRUE(spatial$proper),
      scale_factor = spatial$scale_factor,
      Q_inv = mass$Q_inv,
      L_Q = mass$L
    )
  }
}


#' Prepare temporal precision matrix info
#'
#' @param temporal ratiod_temporal object
#'
#' @return List with temporal precision structure
#' @keywords internal
prepare_temporal_precision <- function(temporal) {
  if (is.null(temporal)) return(NULL)

  T <- temporal$n_times
  type <- temporal$type
  cyclic <- isTRUE(temporal$cyclic)

  # Build temporal precision matrix Q_time for Kronecker mass
  Q_time <- NULL
  Q_time_inv_flat <- NULL
  L_time_flat <- NULL
  if (T >= 3) {
    if (type == "rw1") {
      # RW1: Q[i,i] = 2 (interior), 1 (boundary), Q[i,i+1] = Q[i+1,i] = -1
      Q_time <- matrix(0, T, T)
      for (i in 1:(T-1)) {
        Q_time[i, i] <- Q_time[i, i] + 1
        Q_time[i+1, i+1] <- Q_time[i+1, i+1] + 1
        Q_time[i, i+1] <- -1
        Q_time[i+1, i] <- -1
      }
    } else if (type == "rw2") {
      # RW2: second-order differences
      D <- matrix(0, T-2, T)
      for (i in 1:(T-2)) {
        D[i, i] <- 1; D[i, i+1] <- -2; D[i, i+2] <- 1
      }
      Q_time <- t(D) %*% D
    } else if (type == "ar1") {
      # AR1 with rho=0.5 as default (approximate)
      Q_time <- diag(T)
      for (i in 1:(T-1)) {
        Q_time[i, i+1] <- -0.5
        Q_time[i+1, i] <- -0.5
      }
    }
    if (!is.null(Q_time)) {
      mass <- ridged_precision_factors(Q_time)
      Q_time_inv_flat <- mass$Q_inv
      L_time_flat <- mass$L
    }
  }

  list(
    type = type,
    T = T,
    cyclic = cyclic,
    Q_time_inv = Q_time_inv_flat,
    L_time = L_time_flat
  )
}


#' Compute nearest neighbors for spatiotemporal NNGP
#'
#' @param coords N x 2 matrix of coordinates
#' @param time N-vector of time values
#' @param k Number of nearest neighbors
#'
#' @return List with neighbor structure
#' @keywords internal
compute_st_nngp_neighbors <- function(coords, time, k) {
  N <- nrow(coords)

  # Order by time first (ensures temporal causality in conditioning)
  time_order <- order(time, coords[, 1], coords[, 2])

  coords_ordered <- coords[time_order, , drop = FALSE]
  time_ordered <- time[time_order]

  # Compute neighbors in space-time
  # Zero where there is no neighbour, matching nn_idx's own 0. An Inf would
  # be a distance the covariance could be evaluated at if anything ever read
  # past the neighbour count.
  nn_idx <- matrix(0L, nrow = N, ncol = k)
  nn_dist_space <- matrix(0, nrow = N, ncol = k)
  nn_dist_time <- matrix(0, nrow = N, ncol = k)

  for (i in seq_len(N)[-1]) {
    n_candidates <- min(i - 1, k)

    if (n_candidates > 0) {
      # Compute space-time distances to previous observations
      # Using simple Euclidean distance in scaled space-time
      space_dists <- sqrt(
        (coords_ordered[1:(i-1), 1] - coords_ordered[i, 1])^2 +
        (coords_ordered[1:(i-1), 2] - coords_ordered[i, 2])^2
      )
      time_dists <- abs(time_ordered[1:(i-1)] - time_ordered[i])

      # Combined distance (can be weighted)
      combined_dists <- sqrt(space_dists^2 + time_dists^2)

      if (length(combined_dists) <= k) {
        nn_order <- order(combined_dists)
        nn_idx[i, seq_along(combined_dists)] <- nn_order
        nn_dist_space[i, seq_along(combined_dists)] <- space_dists[nn_order]
        nn_dist_time[i, seq_along(combined_dists)] <- time_dists[nn_order]
      } else {
        nn_order <- order(combined_dists)[1:k]
        nn_idx[i, ] <- nn_order
        nn_dist_space[i, ] <- space_dists[nn_order]
        nn_dist_time[i, ] <- time_dists[nn_order]
      }
    }
  }

  list(
    nn_idx = nn_idx,
    nn_dist_space = nn_dist_space,
    nn_dist_time = nn_dist_time,
    nn_order = time_order,
    nn_order_inv = order(time_order),
    k = k
  )
}
