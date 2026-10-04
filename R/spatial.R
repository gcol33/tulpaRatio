#' Spatial structure specifications for tulpaRatio
#'
#' @description
#' Functions to specify spatial random effects for ratio models.
#' Spatial effects are shared between numerator and denominator by default,
#' which helps prevent spurious ratio effects from spatially-structured
#' unmeasured confounders.
#'
#' @name ratiod_spatial
NULL

#' CAR spatial structure
#'
#' @description
#' Specify a conditional autoregressive (CAR) spatial random effect.
#' Supports both intrinsic CAR (ICAR) and proper CAR variants.
#'
#' @param adjacency Adjacency matrix (sparse or dense). A symmetric matrix
#'   where entry (i,j) is 1 if areas i and j are neighbors, 0 otherwise.
#' @param level Level at which spatial structure applies:
#'   - `"group"`: Spatial effect at the grouping variable level (e.g., sites).
#'     Requires `group_var` to be specified.
#'   - `"obs"`: Spatial effect at the observation level.
#' @param group_var Name of the grouping variable in data (required if
#'   `level = "group"`).
#' @param proper Logical; if FALSE (default), uses Intrinsic CAR (ICAR) with
#'   rho = 1 fixed. If TRUE, uses proper CAR with rho estimated. See Details.
#' @param shared Logical; if TRUE (default), spatial effect enters both
#'   numerator and denominator linear predictors identically.
#' @param parameterization Parameterization of the spatial random effect.
#'   Either `"standard"` (default) or `"collapsed"` to marginalize the spatial
#'   variance for improved mixing in sparse-graph regimes.
#' @param center Logical; if TRUE (default), the field's mean is removed on its
#'   way into the linear predictor, so the intercept carries the level and the
#'   field carries deviations from it. The prior still reads the uncentred
#'   field. `center = FALSE` requires `proper = TRUE` and leaves the mean in the
#'   linear predictor, where it is jointly identified with the intercept: on a
#'   36-unit fixture the two then correlate at -0.977 and the intercept's bulk
#'   ESS is 71 of 2000 against 1953 under the default, for the same posterior on
#'   the level. An intrinsic field (`proper = FALSE`) has no such choice, since
#'   its mean is unidentified until it is removed.
#'
#' @return A `ratiod_spatial` object
#'
#' @details
#' The CAR model specifies that:
#'
#' \deqn{\phi_i | \phi_{-i} \sim N\left(\rho \frac{\sum_{j \sim i} \phi_j}{n_i},
#'   \frac{\sigma^2}{n_i}\right)}
#'
#' where \eqn{n_i} is the number of neighbors of area i and \eqn{j \sim i}
#' denotes that j is a neighbor of i.
#'
#' This leads to a precision matrix:
#' \deqn{Q = \tau (D - \rho W)}
#'
#' where D is the diagonal matrix of neighbor counts and W is the adjacency
#' matrix.
#'
#' **ICAR (proper = FALSE, default)**
#'
#' - Sets rho = 1 (fixed)
#' - Improper prior (rank-deficient Q)
#' - Requires sum-to-zero constraint for identifiability
#' - Simpler: one fewer parameter to estimate
#' - Standard choice for disease mapping
#' - Equivalent to RW1 on a graph
#'
#' **Proper CAR (proper = TRUE)**
#'
#' - Estimates rho in (0, 1) from data
#' - Proper prior (integrates to 1)
#' - rho measures spatial autocorrelation strength
#' - rho -> 0: approaches independence (IID)
#' - rho -> 1: approaches ICAR
#' - More flexible but additional parameter to estimate
#' - Prior: rho ~ Beta(1, 1) (uniform on (0, 1))
#'
#' @examples
#' # Create adjacency matrix for 10 regions (chain structure)
#' adj <- matrix(0, 10, 10)
#' for (i in 1:9) {
#'   adj[i, i+1] <- adj[i+1, i] <- 1
#' }
#'
#' # ICAR (default, rho = 1 fixed)
#' icar <- spatial_car(adj, level = "group", group_var = "site")
#' print(icar)
#'
#' # Proper CAR (rho estimated)
#' proper_car <- spatial_car(adj, level = "group", group_var = "site",
#'                           proper = TRUE)
#' print(proper_car)
#'
#' \donttest{
#' # Generate synthetic data with spatial structure
#' set.seed(123)
#' n_sites <- 10
#' n_per_site <- 5
#' df <- data.frame(
#'   site = rep(1:n_sites, each = n_per_site),
#'   x = rnorm(n_sites * n_per_site),
#'   count = rpois(n_sites * n_per_site, 20),
#'   effort = rgamma(n_sites * n_per_site, shape = 5, rate = 1)
#' )
#'
#' # ICAR model (standard disease mapping approach)
#' fit_icar <- tratio(
#'   count | effort ~ x,
#'   data = df,
#'   family = ratiod_poisson_gamma(),
#'   spatial = spatial_car(adj, level = "group", group_var = "site"),
#'   mode = "hmc",
#'   control = list(iter = 200, warmup = 100, chains = 1)
#' )
#'
#' # Proper CAR model (estimate spatial autocorrelation)
#' fit_car <- tratio(
#'   count | effort ~ x,
#'   data = df,
#'   family = ratiod_poisson_gamma(),
#'   spatial = spatial_car(adj, level = "group", group_var = "site",
#'                         proper = TRUE),
#'   mode = "hmc",
#'   control = list(iter = 200, warmup = 100, chains = 1)
#' )
#'
#' # Extract rho from proper CAR
#' summary(fit_car)  # Shows rho_spatial parameter
#' }
#'
#' @seealso [spatial_bym2()] for decomposed spatial + IID effects,
#'   [spatial_gp()] for continuous spatial effects
#'
#' @export
spatial_car <- function(adjacency, level = c("group", "obs"),
                        group_var = NULL, proper = FALSE, shared = TRUE,
                        parameterization = c("standard", "collapsed"),
                        center = TRUE) {

  level <- match.arg(level)
  parameterization <- match.arg(parameterization)

  if (!is.logical(center) || length(center) != 1 || is.na(center)) {
    stop("`center` must be TRUE or FALSE", call. = FALSE)
  }
  # An intrinsic Q is rank-deficient on the constant direction, so the field's
  # mean is not identified at all until it is removed from the likelihood.
  # There is no model on the other side of that switch.
  if (!center && !proper) {
    stop("`center = FALSE` needs `proper = TRUE`. An intrinsic CAR field's
",
         "mean is unidentified against the intercept, so removing it from the
",
         "linear predictor is what identifies the model rather than a choice.",
         call. = FALSE)
  }

  # Validate adjacency matrix
  if (!is.matrix(adjacency) && !inherits(adjacency, "Matrix")) {
    stop("`adjacency` must be a matrix", call. = FALSE)
  }

  if (nrow(adjacency) != ncol(adjacency)) {
    stop("`adjacency` must be square", call. = FALSE)
  }

  # Check symmetry
  if (!isSymmetric(unname(as.matrix(adjacency)))) {
    stop("`adjacency` must be symmetric", call. = FALSE)
  }

  # Check for group_var if level = "group"
  if (level == "group" && is.null(group_var)) {
    stop("`group_var` is required when level = 'group'", call. = FALSE)
  }

  # Collapsed not supported with proper CAR
  if (parameterization == "collapsed" && proper) {
    stop("Collapsed parameterization is not supported with proper CAR", call. = FALSE)
  }

  # Compute eigenvalue bounds for proper CAR (needed for valid ρ range)
  rho_bounds <- NULL
  if (proper) {
    rho_bounds <- compute_car_rho_bounds(adjacency)
  }

  # Warning for non-shared spatial effects
  if (!shared) {
    warning(
      "Non-shared spatial effects (shared = FALSE) may lead to confounded ratio estimates.\n",
      "Consider whether spatial effects should be shared between\n",
      "numerator and denominator to prevent spurious spatial patterns in ratios.",
      call. = FALSE
    )
  }

  structure(
    list(
      type = if (proper) "car_proper" else "car",
      adjacency = adjacency,
      level = level,
      group_var = group_var,
      proper = proper,
      rho_bounds = rho_bounds,
      shared = shared,
      parameterization = parameterization,
      center = center,
      n_spatial = nrow(adjacency)
    ),
    class = c("ratiod_spatial", "list")
  )
}


#' Compute valid bounds for rho in proper CAR
#'
#' @description
#' For proper CAR, rho must be in the range (1/lambda_min, 1/lambda_max) where lambda are the
#' eigenvalues of D^(-1)W. In practice, we typically restrict to (0, 1) for
#' interpretability (positive spatial autocorrelation).
#'
#' @param adjacency Adjacency matrix
#'
#' @return Named vector with `lower` and `upper` bounds for rho
#' @keywords internal
compute_car_rho_bounds <- function(adjacency) {
  adj <- as.matrix(adjacency)
  diag(adj) <- 0

  n <- nrow(adj)
  n_neighbors <- rowSums(adj)

  # Check for isolated nodes (no neighbors)
  if (any(n_neighbors == 0)) {
    warning("Adjacency matrix contains isolated nodes (no neighbors).\n",
            "These will be treated as independent.", call. = FALSE)
    # Remove isolated nodes for eigenvalue computation
    keep <- n_neighbors > 0
    adj_sub <- adj[keep, keep]
    n_neighbors_sub <- n_neighbors[keep]

    if (sum(keep) < 2) {
      # Not enough connected nodes
      return(c(lower = 0, upper = 1))
    }

    D_inv <- diag(1 / n_neighbors_sub)
    D_inv_W <- D_inv %*% adj_sub
  } else {
    D_inv <- diag(1 / n_neighbors)
    D_inv_W <- D_inv %*% adj
  }

  # Compute eigenvalues
  eig <- eigen(D_inv_W, symmetric = FALSE, only.values = TRUE)$values
  eig_real <- Re(eig)

  # Theoretical bounds: 1/λ_min < ρ < 1/λ_max
  lambda_min <- min(eig_real)
  lambda_max <- max(eig_real)

  # For positive autocorrelation, restrict to (0, 1)
  # The upper bound is typically 1/λ_max ≈ 1 for connected graphs
  lower <- max(0, 1 / lambda_max)
  upper <- min(1, 1 / lambda_min)

  # Ensure valid range
  if (lower >= upper) {
    lower <- 0
    upper <- 1
  }

  c(lower = lower, upper = upper)
}

#' BYM2 spatial structure
#'
#' @description
#' Specify a Besag-York-Mollie 2 (BYM2) spatial random effect.
#' BYM2 decomposes the spatial effect into a structured (ICAR) component
#' and an unstructured (IID) component, with a mixing parameter controlling
#' the proportion of variance attributable to spatial structure.
#'
#' BYM2 is preferred over plain CAR when you want to:
#' - Distinguish structured vs unstructured spatial variation
#' - Have an interpretable spatial fraction parameter
#' - Use the scaling from Riebler et al. (2016)
#'
#' @inheritParams spatial_car
#' @param scale_factor Scaling factor for the ICAR component. If NULL
#'   (default), computed from the adjacency matrix following Riebler et al.
#'
#' @return A `ratiod_spatial` object
#'
#' @references
#' Riebler, A., Sorbye, S. H., Simpson, D., & Rue, H. (2016). An intuitive
#' Bayesian spatial model for disease mapping that accounts for scaling.
#' Statistical Methods in Medical Research, 25(4), 1145-1165.
#'
#' @examples
#' # Create adjacency matrix for 10 regions (chain structure)
#' adj <- matrix(0, 10, 10)
#' for (i in 1:9) {
#'   adj[i, i+1] <- adj[i+1, i] <- 1
#' }
#'
#' # Create BYM2 spatial structure
#' bym2 <- spatial_bym2(adj, level = "group", group_var = "region")
#' print(bym2)
#'
#' \dontrun{
#' # Generate synthetic epidemiological data (not run - requires laplace backend)
#' set.seed(456)
#' n_regions <- 10
#' epi_data <- data.frame(
#'   region = factor(1:n_regions),
#'   age = rnorm(n_regions, 50, 10),
#'   cases = rbinom(n_regions, size = 100, prob = 0.15),
#'   population = rep(100L, n_regions)
#' )
#'
#' # Disease mapping with BYM2 spatial smoothing
#' fit <- tratio(
#'   cases | population ~ age,
#'   spatial = spatial_bym2(adj, level = "group", group_var = "region"),
#'   data = epi_data,
#'   family = ratiod_binomial(),
#'   mode = "laplace"
#' )
#' summary(fit)
#' }
#'
#' @export
spatial_bym2 <- function(adjacency, level = c("group", "obs"),
                         group_var = NULL, shared = TRUE,
                         scale_factor = NULL,
                         parameterization = c("standard", "collapsed")) {

  level <- match.arg(level)
  parameterization <- match.arg(parameterization)

  # Validate adjacency matrix
  if (!is.matrix(adjacency) && !inherits(adjacency, "Matrix")) {
    stop("`adjacency` must be a matrix", call. = FALSE)
  }

  if (nrow(adjacency) != ncol(adjacency)) {
    stop("`adjacency` must be square", call. = FALSE)
  }

  if (!isSymmetric(unname(as.matrix(adjacency)))) {
    stop("`adjacency` must be symmetric", call. = FALSE)
  }

  if (level == "group" && is.null(group_var)) {
    stop("`group_var` is required when level = 'group'", call. = FALSE)
  }

  # Compute scale factor if not provided
  if (is.null(scale_factor)) {
    scale_factor <- compute_bym2_scale(adjacency)
  }

  # Warning for non-shared spatial effects
  if (!shared) {
    warning(
      "Non-shared spatial effects (shared = FALSE) may lead to confounded ratio estimates.\n",
      "Consider whether spatial effects should be shared between\n",
      "numerator and denominator to prevent spurious spatial patterns in ratios.",
      call. = FALSE
    )
  }

  structure(
    list(
      type = "bym2",
      adjacency = adjacency,
      level = level,
      group_var = group_var,
      shared = shared,
      parameterization = parameterization,
      n_spatial = nrow(adjacency),
      scale_factor = scale_factor
    ),
    class = c("ratiod_spatial", "list")
  )
}

#' Compute BYM2 scaling factor
#'
#' @description
#' Compute the scaling factor for BYM2 following Riebler et al. (2016).
#' This makes the spatial fraction parameter interpretable.
#'
#' @param adjacency Adjacency matrix
#'
#' @return Scaling factor (scalar)
#' @keywords internal
compute_bym2_scale <- function(adjacency) {
  # Build precision matrix Q for ICAR
  n <- nrow(adjacency)
  adj <- as.matrix(adjacency)
  diag(adj) <- 0

  # Number of neighbors for each area
  n_neighbors <- rowSums(adj)

  # ICAR precision matrix: Q_ii = n_neighbors[i], Q_ij = -1 if neighbors
  Q <- diag(n_neighbors) - adj

  # Compute generalized inverse (Q is rank-deficient)
  # Use eigendecomposition
  eig <- eigen(Q, symmetric = TRUE)

  # Remove the zero eigenvalue (rank deficiency)
  non_zero <- abs(eig$values) > 1e-10
  lambda <- eig$values[non_zero]

  # Geometric mean of non-zero eigenvalues
  # This is the scaling factor following INLA convention
  scale <- exp(mean(log(lambda)))

  scale
}

#' Print method for ratiod_spatial
#'
#' @param x A ratiod_spatial object
#' @param ... Ignored
#'
#' @export
print.ratiod_spatial <- function(x, ...) {
  cat("tulpaRatio spatial specification\n")
  cat("===========================\n\n")

  # Format type name
  type_name <- switch(x$type,
    car = "ICAR (Intrinsic CAR)",
    car_proper = "Proper CAR",
    bym2 = "BYM2",
    toupper(x$type)
  )
  cat("Type:", type_name, "\n")
  cat("Level:", x$level, "\n")
  cat("Spatial units:", x$n_spatial, "\n")
  cat("Shared:", if (x$shared) "Yes (enters both processes)" else "No", "\n")

  if (!is.null(x$group_var)) {
    cat("Group variable:", x$group_var, "\n")
  }

  if (x$type == "car_proper" && !is.null(x$rho_bounds)) {
    cat("Rho bounds: [", round(x$rho_bounds["lower"], 4), ", ",
        round(x$rho_bounds["upper"], 4), "]\n", sep = "")
    cat("  (spatial autocorrelation parameter, estimated from data)\n")
  }

  if (x$type == "car") {
    cat("  (rho fixed at 1, sum-to-zero constraint applied)\n")
  }

  if (x$type == "bym2") {
    cat("Scale factor:", round(x$scale_factor, 4), "\n")
  }

  invisible(x)
}

#' Check if adjacency matrix is connected
#'
#' @description
#' Check if the spatial graph defined by the adjacency matrix is fully
#' connected. A disconnected graph can cause identifiability issues.
#'
#' @param adjacency Adjacency matrix
#'
#' @return Logical; TRUE if connected
#' @keywords internal
is_connected <- function(adjacency) {
  n <- nrow(adjacency)
  if (n == 0) return(TRUE)

  adj <- as.matrix(adjacency)
  diag(adj) <- 0

  # BFS to check connectivity
  visited <- logical(n)
  queue <- 1L
  visited[1] <- TRUE

  while (length(queue) > 0) {
    current <- queue[1]
    queue <- queue[-1]

    neighbors <- which(adj[current, ] > 0)
    new_neighbors <- neighbors[!visited[neighbors]]

    visited[new_neighbors] <- TRUE
    queue <- c(queue, new_neighbors)
  }

  all(visited)
}

#' Gaussian Process spatial structure
#'
#' @description
#' Specify a Gaussian Process (GP) spatial random effect using coordinate-based
#' distances. Uses Nearest Neighbor Gaussian Process (NNGP) approximation for
#' computational efficiency with large datasets (scales to millions of observations).
#'
#' This provides a continuous spatial effect that captures smooth spatial
#' variation, unlike CAR/BYM2 which require discrete areal units.
#'
#' @param coords A one-sided formula specifying coordinate columns (e.g.,
#'   `~ lon + lat`), or a character vector of length 2 with column names.
#' @param cov Covariance function: `"exponential"` (default), `"matern"`,
#'   `"gaussian"`, or `"spherical"`.
#' @param nu Smoothness parameter for Matern covariance. Common values:
#'   - 0.5: Equivalent to exponential (rough, once mean-square differentiable)
#'   - 1.5: Once differentiable (moderate smoothness)
#'   - 2.5: Twice differentiable (smooth)
#'   Ignored for non-Matern covariance functions.
#' @param nn Number of nearest neighbors for NNGP approximation. Default 15.
#'   Larger values give better approximation but slower computation.
#' @param shared Logical; if TRUE (default), spatial effect enters both
#'   numerator and denominator. Set to FALSE for process-specific spatial
#'   effects (triggers warning about potential confounding).
#' @param scale_coords Logical; if TRUE (default), coordinates are scaled to
#'   unit variance before computing distances.
#' @param parameterization Parameterization of the latent field. One of
#'   `"centered"` (default), `"noncentered"` (Matt trick, better for diffuse
#'   prior regimes), or `"collapsed"` (marginalize spatial variance for
#'   improved mixing). Centered is robust for informative priors.
#'
#' @return A `ratiod_gp` object
#'
#' @details
#' The GP spatial model adds a spatially-correlated random effect to the
#' linear predictor:
#'
#' \deqn{\eta(s) = X\beta + w(s)}
#'
#' where \eqn{w(s)} follows a Gaussian process:
#' \deqn{w(s) \sim GP(0, \sigma^2 C(d; \phi))}
#'
#' The correlation function \eqn{C(d; \phi)} depends on distance \eqn{d} and
#' range parameter \eqn{\phi}:
#'
#' - **Exponential**: \eqn{C(d) = \exp(-d/\phi)}
#' - **Matern**: \eqn{C(d) = \frac{2^{1-\nu}}{\Gamma(\nu)} (\sqrt{2\nu} d/\phi)^\nu K_\nu(\sqrt{2\nu} d/\phi)}
#' - **Gaussian**: \eqn{C(d) = \exp(-(d/\phi)^2)}
#' - **Spherical**: \eqn{C(d) = 1 - 1.5(d/\phi) + 0.5(d/\phi)^3} for \eqn{d < \phi}
#'
#' **NNGP approximation**: For computational tractability, we use the
#' Nearest Neighbor Gaussian Process (Datta et al., 2016), which conditions
#' each location on its k nearest neighbors. This reduces complexity from
#' O(n^3) to O(n*k^2), enabling models with millions of observations.
#'
#' @examples
#' # Create GP spatial structure
#' gp <- spatial_gp(~ lon + lat)
#' print(gp)
#'
#' \donttest{
#' # Generate synthetic spatial data
#' set.seed(789)
#' n <- 50
#' df <- data.frame(
#'   lon = runif(n, 0, 10),
#'   lat = runif(n, 0, 10),
#'   depth = rnorm(n),
#'   temp = rnorm(n),
#'   count = rpois(n, 25),
#'   effort = rgamma(n, shape = 4, rate = 1)
#' )
#'
#' # Continuous spatial effect with exponential covariance
#' fit <- tratio(
#'   count | effort ~ depth + temp,
#'   data = df,
#'   family = ratiod_poisson_gamma(),
#'   spatial = spatial_gp(~ lon + lat),
#'   mode = "hmc",
#'   control = list(iter = 200, warmup = 100, chains = 1)
#' )
#' summary(fit)
#'
#' # Smoother spatial field with Matern covariance
#' fit2 <- tratio(
#'   count | effort ~ depth + temp,
#'   data = df,
#'   family = ratiod_poisson_gamma(),
#'   spatial = spatial_gp(~ lon + lat, cov = "matern", nu = 1.5),
#'   mode = "hmc",
#'   control = list(iter = 200, warmup = 100, chains = 1)
#' )
#' }
#'
#' @references
#' Datta, A., Banerjee, S., Finley, A. O., & Gelfand, A. E. (2016).
#' Hierarchical nearest-neighbor Gaussian process models for large
#' geostatistical datasets. Journal of the American Statistical Association,
#' 111(514), 800-812.
#'
#' @seealso [spatial_car()], [spatial_bym2()] for areal spatial effects,
#'   [spatial_svc()] for spatially-varying coefficients,
#'   [spatial_multiscale()] for multi-scale spatial effects
#'
#' @export
spatial_gp <- function(coords,
                       cov = c("exponential", "matern", "gaussian", "spherical"),
                       nu = 1.5,
                       nn = 15,
                       shared = TRUE,
                       scale_coords = TRUE,
                       parameterization = c("centered", "noncentered", "collapsed")) {

  cov <- match.arg(cov)
  parameterization <- match.arg(parameterization)

  # Parse coordinate specification
  if (inherits(coords, "formula")) {
    coord_vars <- all.vars(coords)
    if (length(coord_vars) != 2) {
      stop("`coords` formula must specify exactly 2 coordinate variables",
           call. = FALSE)
    }
  } else if (is.character(coords) && length(coords) == 2) {
    coord_vars <- coords
  } else {
    stop("`coords` must be a formula (~ lon + lat) or character vector of length 2",
         call. = FALSE)
  }

  # Validate nu for Matern
  if (cov == "matern") {
    if (!is.numeric(nu) || length(nu) != 1 || nu <= 0) {
      stop("`nu` must be a positive number for Matern covariance", call. = FALSE)
    }
  }

  # Validate nn
  if (!is.numeric(nn) || length(nn) != 1 || nn < 1) {
    stop("`nn` must be a positive integer", call. = FALSE)
  }
  nn <- as.integer(nn)

  # Warning for non-shared spatial effects
  if (!shared) {
    warning(
      "Non-shared spatial effects (shared = FALSE) may lead to confounded ratio estimates.\n",
      "Consider whether spatial effects should be shared between\n",
      "numerator and denominator to prevent spurious spatial patterns in ratios.",
      call. = FALSE
    )
  }

  structure(
    list(
      type = "gp",
      coord_vars = coord_vars,
      cov = cov,
      nu = if (cov == "matern") nu else NULL,
      nn = nn,
      shared = shared,
      scale_coords = scale_coords,
      parameterization = parameterization,
      # Filled in during validation
      n_obs = NULL,
      n_spatial = NULL,
      coords_matrix = NULL,
      neighbor_info = NULL
    ),
    class = c("ratiod_gp", "ratiod_spatial", "list")
  )
}


#' Print method for ratiod_gp
#'
#' @param x A ratiod_gp object
#' @param ... Ignored
#'
#' @export
print.ratiod_gp <- function(x, ...) {
  cat("tulpaRatio Gaussian Process spatial specification\n")
  cat("=============================================\n\n")

  cat("Coordinates:", paste(x$coord_vars, collapse = ", "), "\n")
  cov_str <- x$cov
  if (x$cov == "matern" && !is.null(x$nu)) {
    cov_str <- sprintf("%s (nu = %.1f)", x$cov, x$nu)
  }
  cat("Covariance:", cov_str, "\n")
  cat("Neighbors (NNGP):", x$nn, "\n")

  cat("Shared:", if (x$shared) "Yes (enters both processes)" else "No", "\n")

  if (!is.null(x$n_obs)) {
    cat("\nObservations:", x$n_obs, "\n")
  }

  invisible(x)
}


#' Hilbert Space Gaussian Process (HSGP) spatial structure
#'
#' @description
#' Specify a Hilbert Space Gaussian Process approximation for spatial effects.
#' HSGP approximates a GP using Laplacian eigenfunctions, providing O(N*M^2)
#' complexity with analytical gradients instead of O(N*k^2) with numerical
#' gradients for NNGP.
#'
#' This gives approximately 50x speedup over standard NNGP while maintaining
#' high accuracy for smooth spatial fields.
#'
#' @param coords A one-sided formula specifying coordinate columns (e.g.,
#'   `~ lon + lat`), or a character vector of length 2 with column names.
#' @param m Number of basis functions per dimension. Total basis functions
#'   will be m^2. Default 6. Higher values give better approximation but
#'   slower computation. Recommended range: 5-15.
#' @param c Boundary factor controlling domain extension beyond data range.
#'   Default 1.5. The domain is extended to \eqn{[-cL, cL]} where \eqn{L} is
#'   half the data range. Larger values improve approximation at boundaries.
#' @param shared Logical; if TRUE (default), spatial effect enters both
#'   numerator and denominator. Set to FALSE for process-specific spatial
#'   effects (triggers warning about potential confounding).
#' @param scale_coords Logical; if TRUE (default), coordinates are scaled to
#'   unit variance before computing basis functions.
#'
#' @return A `ratiod_hsgp` object
#'
#' @details
#' HSGP approximates a GP as:
#'
#' \deqn{f(x) = \sum_{j=1}^{M^2} \phi_j(x) \sqrt{S(\lambda_j)} \beta_j}
#'
#' where:
#' - \eqn{\phi_j(x)} are Laplacian eigenfunctions (products of sines)
#' - \eqn{S(\lambda_j)} is the spectral density of the squared exponential kernel
#' - \eqn{\beta_j \sim N(0, 1)} are basis coefficients
#'
#' The hyperparameters are:
#' - \eqn{\sigma^2}: marginal variance (PC prior: P(sigma > 1) = 0.01)
#' - \eqn{\ell}: lengthscale (LogNormal(0, 1) prior)
#'
#' **Advantages over NNGP**:
#' - Analytical gradients enable ~50x speedup
#' - Simple parameter interpretation
#' - Works well for smooth spatial fields
#'
#' **Limitations**:
#' - Assumes squared exponential (smooth) covariance
#' - Less accurate for rough fields (Matern with low nu)
#' - Approximation quality depends on m and c choices
#'
#' @examples
#' # Create HSGP spatial structure
#' hsgp <- spatial_hsgp(~ lon + lat)
#' print(hsgp)
#'
#' \donttest{
#' # Generate synthetic spatial data
#' set.seed(42)
#' n <- 100
#' df <- data.frame(
#'   lon = runif(n, 0, 10),
#'   lat = runif(n, 0, 10),
#'   x = rnorm(n),
#'   count = rpois(n, 25),
#'   effort = rgamma(n, shape = 4, rate = 1)
#' )
#'
#' # Fast spatial effect with HSGP (much faster than spatial_gp)
#' fit <- tratio(
#'   count | effort ~ x,
#'   data = df,
#'   family = ratiod_poisson_gamma(),
#'   spatial = spatial_hsgp(~ lon + lat, m = 6),
#'   mode = "hmc",
#'   control = list(iter = 500, warmup = 250, chains = 2)
#' )
#' summary(fit)
#' }
#'
#' @references
#' Riutort-Mayol, G., Bürkner, P. C., Andersen, M. R., Solin, A., & Vehtari, A.
#' (2023). Practical Hilbert space approximate Bayesian Gaussian processes for
#' probabilistic programming. Statistics and Computing, 33(1), 17.
#'
#' @seealso [spatial_gp()] for NNGP-based GP, [spatial_car()] for areal effects
#'
#' @export
spatial_hsgp <- function(coords,
                         m = 6,
                         c = 1.5,
                         shared = TRUE,
                         scale_coords = TRUE) {

  # Parse coordinate specification
  if (inherits(coords, "formula")) {
    coord_vars <- all.vars(coords)
    if (length(coord_vars) != 2) {
      stop("`coords` formula must specify exactly 2 coordinate variables",
           call. = FALSE)
    }
  } else if (is.character(coords) && length(coords) == 2) {
    coord_vars <- coords
  } else {
    stop("`coords` must be a formula (~ lon + lat) or character vector of length 2",
         call. = FALSE)
  }

  # Validate m
  if (!is.numeric(m) || length(m) != 1 || m < 3 || m > 50) {
    stop("`m` must be an integer between 3 and 50", call. = FALSE)
  }
  m <- as.integer(m)

  # Validate c
  if (!is.numeric(c) || length(c) != 1 || c < 1) {
    stop("`c` (boundary factor) must be >= 1", call. = FALSE)
  }

  # Warning for non-shared spatial effects
  if (!shared) {
    warning(
      "Non-shared spatial effects (shared = FALSE) may lead to confounded ratio estimates.\n",
      "Consider whether spatial effects should be shared between\n",
      "numerator and denominator to prevent spurious spatial patterns in ratios.",
      call. = FALSE
    )
  }

  structure(
    list(
      type = "hsgp",
      coord_vars = coord_vars,
      m = m,
      c = c,
      shared = shared,
      scale_coords = scale_coords,
      # Filled in during validation
      n_obs = NULL,
      coords_matrix = NULL
    ),
    class = c("ratiod_hsgp", "ratiod_spatial", "list")
  )
}


#' Print method for ratiod_hsgp
#'
#' @param x A ratiod_hsgp object
#' @param ... Ignored
#'
#' @export
print.ratiod_hsgp <- function(x, ...) {
  cat("tulpaRatio Hilbert Space GP (HSGP) spatial specification\n")
  cat("=====================================================\n\n")

  cat("Coordinates:", paste(x$coord_vars, collapse = ", "), "\n")
  cat("Basis functions:", x$m, "per dim (", x$m^2, "total )\n")
  cat("Boundary factor:", x$c, "\n")
  cat("Shared:", if (x$shared) "Yes (enters both processes)" else "No", "\n")

  if (!is.null(x$n_obs)) {
    cat("\nObservations:", x$n_obs, "\n")
  }

  invisible(x)
}


#' Validate HSGP spatial structure
#'
#' @param spatial A ratiod_hsgp object
#' @param data The data frame
#'
#' @return A validated ratiod_hsgp object with coords_matrix filled in
#'
#' @keywords internal
validate_hsgp <- function(spatial, data) {
  # Check coordinate columns exist
  missing_cols <- setdiff(spatial$coord_vars, names(data))
  if (length(missing_cols) > 0) {
    stop("Coordinate columns not found in data: ",
         paste(missing_cols, collapse = ", "), call. = FALSE)
  }

  # Extract coordinates
  coords <- as.matrix(data[, spatial$coord_vars, drop = FALSE])

  # Check for missing values
  if (any(is.na(coords))) {
    stop("Coordinate columns contain missing values", call. = FALSE)
  }

  # Scale if requested
  if (spatial$scale_coords) {
    coords <- scale(coords)
  }

  # Store validated data
  spatial$n_obs <- nrow(coords)
  spatial$coords_matrix <- coords

  spatial
}


#' Multi-Scale Gaussian Process spatial structure
#'
#' @description
#' Specify a multi-scale spatial random effect that decomposes spatial
#' variation into local (fine-scale) and regional (broad-scale) components.
#' Each scale has its own range and variance parameters.
#'
#' This is particularly useful for large datasets (>100k observations) where
#' spatial patterns exist at multiple scales.
#'
#' @param coords A one-sided formula specifying coordinate columns (e.g.,
#'   `~ lon + lat`), or a character vector of length 2 with column names.
#' @param scales Character vector specifying scale names. Default: `c("local", "regional")`.
#' @param range_local Prior range for local scale as `c(lower, upper)` in
#'   coordinate units. Default: `c(0.01, 1)` (after scaling).
#' @param range_regional Prior range for regional scale as `c(lower, upper)`.
#'   Default: `c(1, 10)` (after scaling).
#' @param cov Covariance function: `"exponential"` (default), `"matern"`,
#'   `"gaussian"`, or `"spherical"`.
#' @param nu Smoothness parameter for Matern covariance.
#' @param approx Approximation for the two fields: `"nngp"` (default), a
#'   nearest-neighbour Gaussian process per scale, or `"hsgp"`, a Hilbert-space
#'   basis expansion shared by both scales. The basis costs `m^2` coefficients
#'   per scale regardless of how many locations there are, so it is the cheaper
#'   of the two once locations outnumber basis functions; the neighbour sets
#'   follow the data more closely at short ranges.
#' @param nn_local Number of nearest neighbors for local scale. Default 10.
#'   Used when `approx = "nngp"`.
#' @param nn_regional Number of nearest neighbors for regional scale. Default 30.
#'   Used when `approx = "nngp"`.
#' @param m Basis functions per dimension when `approx = "hsgp"`, giving `m^2`
#'   coefficients per scale. Default 10.
#' @param c_boundary Boundary factor when `approx = "hsgp"`: the domain is
#'   extended by this multiple before the basis is built. Default 1.5.
#' @param shared Logical; if TRUE (default), spatial effects enter both
#'   numerator and denominator.
#' @param scale_coords Logical; if TRUE (default), coordinates are scaled to
#'   unit variance before computing distances.
#' @param sampler Warmup mass-matrix adaptation for the two GP blocks:
#'   `"auto"` (default) leaves the sampler's own adaptation in place, `"lbfgs"`
#'   runs an L-BFGS pass during warmup to initialize it.
#' @param parameterization Coordinate the two fields are sampled in. One of
#'   `"centered"` (default), where each block holds the effects themselves, or
#'   `"noncentered"`, where each block holds \eqn{z \sim N(0, I)} and its scale's
#'   effects are \eqn{w = L(\sigma^2, \phi) z}. Both scales carry a variance
#'   parameter over the same locations, so the non-centred coordinate is worth
#'   trying when the variances are weakly informed by the data. Ignored for
#'   `approx = "hsgp"`, whose basis coefficients are already that coordinate.
#'
#' @return A `ratiod_multiscale` object
#'
#' @details
#' The multi-scale model decomposes spatial variation additively:
#'
#' \deqn{\eta(s) = X\beta + w_{local}(s) + w_{regional}(s)}
#'
#' where each component follows an independent Gaussian process:
#' \deqn{w_{local}(s) \sim GP(0, \sigma^2_{local} C(\phi_{local}))}
#' \deqn{w_{regional}(s) \sim GP(0, \sigma^2_{regional} C(\phi_{regional}))}
#'
#' **Identifiability**: With sufficient data (>500 locations), the two scales
#' are typically well-identified when prior ranges are non-overlapping.
#' PC priors on variance components help prevent overfitting.
#'
#' **Computational cost**: Approximately 1.5-2x the cost of single-scale GP,
#' as two NNGP likelihoods must be evaluated.
#'
#' @examples
#' # Create multi-scale spatial structure
#' ms <- spatial_multiscale(
#'   ~ lon + lat,
#'   range_local = c(0.1, 0.5),
#'   range_regional = c(1, 5)
#' )
#' print(ms)
#'
#' # On a shared spectral basis rather than two neighbour sets
#' spatial_multiscale(~ lon + lat, approx = "hsgp", m = 8)
#'
#' \donttest{
#' set.seed(101)
#' n <- 60
#' df <- data.frame(
#'   lon = runif(n, 0, 10),
#'   lat = runif(n, 0, 10),
#'   depth = rnorm(n),
#'   temp = rnorm(n),
#'   count = rpois(n, 30),
#'   effort = rgamma(n, shape = 5, rate = 1)
#' )
#'
#' # Multi-scale spatial with local and regional components
#' fit <- tratio(
#'   count | effort ~ depth + temp,
#'   data = df,
#'   family = ratiod_poisson_gamma(),
#'   spatial = spatial_multiscale(
#'     ~ lon + lat,
#'     range_local = c(0.1, 0.5),
#'     range_regional = c(1, 5)
#'   ),
#'   control = list(iter = 200, warmup = 100, chains = 1)
#' )
#' summary(fit)
#' }
#'
#' @seealso [spatial_gp()] for single-scale GP, [temporal_multiscale()] for
#'   multi-scale temporal effects
#'
#' @export
spatial_multiscale <- function(coords,
                               scales = c("local", "regional"),
                               range_local = c(0.01, 1),
                               range_regional = c(1, 10),
                               cov = c("exponential", "matern", "gaussian", "spherical"),
                               nu = 1.5,
                               approx = c("nngp", "hsgp"),
                               nn_local = 10,
                               nn_regional = 30,
                               m = 10,
                               c_boundary = 1.5,
                               shared = TRUE,
                               scale_coords = TRUE,
                               sampler = c("auto", "lbfgs"),
                               parameterization = c("centered", "noncentered")) {

  cov <- match.arg(cov)
  approx <- match.arg(approx)
  sampler <- match.arg(sampler)
  parameterization <- match.arg(parameterization)

  # Parse coordinate specification
  if (inherits(coords, "formula")) {
    coord_vars <- all.vars(coords)
    if (length(coord_vars) != 2) {
      stop("`coords` formula must specify exactly 2 coordinate variables",
           call. = FALSE)
    }
  } else if (is.character(coords) && length(coords) == 2) {
    coord_vars <- coords
  } else {
    stop("`coords` must be a formula (~ lon + lat) or character vector of length 2",
         call. = FALSE)
  }

  # Validate scales
  if (length(scales) != 2) {
    stop("Currently only 2 scales are supported", call. = FALSE)
  }

  # Validate range specifications
  if (length(range_local) != 2 || range_local[1] >= range_local[2]) {
    stop("`range_local` must be c(lower, upper) with lower < upper", call. = FALSE)
  }
  if (length(range_regional) != 2 || range_regional[1] >= range_regional[2]) {
    stop("`range_regional` must be c(lower, upper) with lower < upper", call. = FALSE)
  }

  # Check range separation
  if (range_local[2] > range_regional[1]) {
    warning(
      "Local and regional range priors overlap.\n",
      "This may cause identifiability issues. Consider separating the ranges:\n",
      sprintf("  Local: [%.2f, %.2f], Regional: [%.2f, %.2f]",
              range_local[1], range_local[2], range_regional[1], range_regional[2]),
      call. = FALSE
    )
  }

  # Validate nn
  if (!is.numeric(nn_local) || nn_local < 1) {
    stop("`nn_local` must be a positive integer", call. = FALSE)
  }
  if (!is.numeric(nn_regional) || nn_regional < 1) {
    stop("`nn_regional` must be a positive integer", call. = FALSE)
  }

  # Validate the HSGP basis
  if (!is.numeric(m) || length(m) != 1 || m < 1) {
    stop("`m` must be a positive integer", call. = FALSE)
  }
  m <- as.integer(m)
  if (!is.numeric(c_boundary) || length(c_boundary) != 1 || c_boundary < 1) {
    stop("`c_boundary` must be >= 1", call. = FALSE)
  }

  # Warning for non-shared
  if (!shared) {
    warning(
      "Non-shared multi-scale spatial effects (shared = FALSE) may lead to confounded ratio estimates.",
      call. = FALSE
    )
  }

  structure(
    list(
      type = "multiscale",
      coord_vars = coord_vars,
      scales = scales,
      range_local = range_local,
      range_regional = range_regional,
      cov = cov,
      nu = if (cov == "matern") nu else NULL,
      approx = approx,
      nn_local = as.integer(nn_local),
      nn_regional = as.integer(nn_regional),
      m = m,
      c_boundary = c_boundary,
      shared = shared,
      scale_coords = scale_coords,
      sampler = sampler,
      parameterization = parameterization,
      # Filled in during validation
      n_obs = NULL,
      n_spatial = NULL,
      coords_matrix = NULL,
      neighbor_info_local = NULL,
      neighbor_info_regional = NULL
    ),
    class = c("ratiod_multiscale", "ratiod_spatial", "list")
  )
}


#' Print method for ratiod_multiscale
#'
#' @param x A ratiod_multiscale object
#' @param ... Ignored
#'
#' @export
print.ratiod_multiscale <- function(x, ...) {
  cat("tulpaRatio Multi-Scale spatial specification\n")
  cat("========================================\n\n")

  cat("Coordinates:", paste(x$coord_vars, collapse = ", "), "\n")
  cat("Scales:", paste(x$scales, collapse = " + "), "\n\n")

  is_hsgp <- identical(x$approx %||% "nngp", "hsgp")

  cat("Local scale:\n")
  cat("  Range prior: [", x$range_local[1], ", ", x$range_local[2], "]\n", sep = "")
  if (!is_hsgp) cat("  Neighbors:", x$nn_local, "\n")

  cat("\nRegional scale:\n")
  cat("  Range prior: [", x$range_regional[1], ", ", x$range_regional[2], "]\n", sep = "")
  if (!is_hsgp) cat("  Neighbors:", x$nn_regional, "\n")

  if (is_hsgp) {
    cat("\nApproximation: HSGP,", x$m, "basis functions per dim (",
        x$m^2, "per scale ), boundary factor", x$c_boundary, "\n")
  } else {
    cat("\nApproximation: NNGP\n")
  }

  cov_str <- x$cov
  if (x$cov == "matern" && !is.null(x$nu)) {
    cov_str <- sprintf("%s (nu = %.1f)", x$cov, x$nu)
  }
  cat("\nCovariance:", cov_str, "\n")
  cat("Shared:", if (x$shared) "Yes (enters both processes)" else "No", "\n")

  if (!is.null(x$n_obs)) {
    cat("\nObservations:", x$n_obs, "\n")
  }

  invisible(x)
}


#' Validate GP spatial specification against data
#'
#' @param gp ratiod_gp or ratiod_multiscale object
#' @param data Data frame
#'
#' @return Updated spatial object with computed neighbor structure
#' @keywords internal
validate_gp <- function(gp, data) {
  if (is.null(gp)) return(NULL)
  if (!inherits(gp, c("ratiod_gp", "ratiod_multiscale"))) return(gp)

  N <- nrow(data)

 # Check coordinate columns exist
  for (cv in gp$coord_vars) {
    if (!(cv %in% names(data))) {
      stop(sprintf("Coordinate variable '%s' not found in data", cv),
           call. = FALSE)
    }
  }

  # Extract coordinates
  coords <- cbind(
    data[[gp$coord_vars[1]]],
    data[[gp$coord_vars[2]]]
  )

  # Check for missing coordinates
  if (any(is.na(coords))) {
    stop("Coordinate columns contain missing values", call. = FALSE)
  }

  # Scale coordinates if requested. The centre and scale are kept: a fit made
  # on scaled coordinates has to read prediction coordinates in the same units,
  # and scale() puts them on an attribute that subsetting the matrix drops.
  gp$coord_center <- NULL
  gp$coord_scale <- NULL
  if (gp$scale_coords) {
    coords <- scale(coords)
    gp$coord_center <- as.numeric(attr(coords, "scaled:center"))
    gp$coord_scale <- as.numeric(attr(coords, "scaled:scale"))
    coords <- matrix(as.numeric(coords), nrow(coords), ncol(coords))
  }

  # Detect unique coordinates (NNGP requires unique locations)
  coord_key <- paste(coords[, 1], coords[, 2], sep = ",")
  unique_keys <- unique(coord_key)
  obs_to_loc <- match(coord_key, unique_keys)
  unique_coords <- coords[match(unique_keys, coord_key), , drop = FALSE]
  n_unique <- nrow(unique_coords)

  if (n_unique < N) {
    message(sprintf("GP: %d unique locations from %d observations", n_unique, N))
  }

  gp$n_obs <- n_unique
  gp$n_spatial <- n_unique
  gp$n_unique <- n_unique
  gp$obs_to_loc <- as.integer(obs_to_loc)
  gp$unique_coords <- unique_coords
  gp$coords_matrix <- coords  # Keep full coords for reference

  # Compute neighbors on unique coordinates only
  # Store clamped nn values back so prepare_gp_for_hmc passes correct sizes to C++
  if (inherits(gp, "ratiod_gp")) {
    # Single-scale GP
    nn <- min(gp$nn, n_unique - 1)
    gp$nn <- nn
    gp$neighbor_info <- compute_nngp_neighbors(unique_coords, nn)

  } else if (inherits(gp, "ratiod_multiscale")) {
    # Multi-scale: separate neighbor structures for each scale
    nn_local <- min(gp$nn_local, n_unique - 1)
    nn_regional <- min(gp$nn_regional, n_unique - 1)
    gp$nn_local <- nn_local
    gp$nn_regional <- nn_regional

    gp$neighbor_info_local <- compute_nngp_neighbors(unique_coords, nn_local)
    gp$neighbor_info_regional <- compute_nngp_neighbors(unique_coords, nn_regional)
  }

  gp
}


#' Default range prior for an SVC term
#'
#' The one place the bounds live. `spatial_svc()` takes them as its `range`
#' default and `prepare_svc_for_hmc()` reads them off the spec, so a fit's
#' range prior is the one the spec prints.
#'
#' @return Numeric `c(lower, upper)`.
#' @keywords internal
svc_default_range <- function() c(0.3, 30)


#' Spatially-Varying Coefficients (SVC)
#'
#' @description
#' Specify spatially-varying coefficients for ratio models. SVCs allow
#' regression coefficients to vary smoothly across space using a Gaussian
#' process prior. This captures local effects that may differ from global
#' relationships.
#'
#' Uses Nearest Neighbor Gaussian Process (NNGP) approximation for
#' computational efficiency with large datasets.
#'
#' @param coords A one-sided formula specifying coordinate columns (e.g.,
#'   `~ lon + lat`), or a character vector of length 2 with column names.
#' @param terms Which coefficients should vary spatially. Options:
#'   - Integer vector: Column indices of design matrix (1 = intercept)
#'   - Character vector: Coefficient names (e.g., `"(Intercept)"`, `"depth"`)
#'   - Formula: `~ 1 + depth` for intercept and depth
#' @param cov Covariance function: `"exponential"` (default), `"matern"`,
#'   `"gaussian"`, or `"spherical"`.
#' @param nn Number of nearest neighbors for NNGP approximation. Default 15.
#'   Larger values give better approximation but slower computation.
#' @param shared Logical; if TRUE (default), SVC effects enter both
#'   numerator and denominator. Set to FALSE for process-specific SVCs
#'   (triggers warning about potential confounding).
#' @param scale_coords Logical; if TRUE (default), coordinates are scaled to
#'   unit variance before computing distances.
#' @param approx Spatial GP approximation. One of `"nngp"` (default,
#'   Nearest Neighbor GP) or `"hsgp"` (Hilbert Space GP, basis-function
#'   approximation suitable for moderate N with strong smoothness).
#' @param m Number of basis functions per coordinate for HSGP (used only
#'   when `approx = "hsgp"`). Default 6.
#' @param c_boundary Boundary scaling factor for HSGP. The HSGP domain is
#'   extended to `c_boundary` times the half-range of scaled coordinates.
#'   Default 1.5.
#' @param range Prior range for the SVC range parameter \eqn{\phi} as
#'   `c(lower, upper)`, in the units of the scaled coordinates. The prior is
#'   uniform on that interval and the density is `-Inf` outside it, so a
#'   posterior that piles up against either end is reporting that the data
#'   want a range the interval excludes. Default `c(0.3, 30)`. Applies to
#'   `approx = "nngp"`; the HSGP arm carries a LogNormal(0, 1) prior on its
#'   lengthscale instead and rejects a `range` set here.
#' @param parameterization Coordinate each NNGP term is sampled in.
#'   `"noncentered"` (default) samples \eqn{z_j \sim N(0, I)} and builds
#'   \eqn{w_j = L(\sigma^2_j, \phi_j) z_j} through the NNGP autoregression, so
#'   the variance reaches the field through the transform rather than through
#'   the prior's quadratic form. `"centered"` samples the field \eqn{w_j}
#'   directly under its NNGP prior, where the field's level and
#'   \eqn{\sigma^2_j} form a funnel that mixes slowly on small data. Both place
#'   the same prior on \eqn{w_j}.
#'   Applies to `approx = "nngp"`; the HSGP arm already samples standardised
#'   basis coefficients.
#'
#' @return A `ratiod_svc` object
#'
#' @details
#' The SVC model extends the linear predictor:
#'
#' \deqn{\eta(s) = X\beta + \tilde{X}(s)w(s)}
#'
#' where:
#' - \eqn{\beta} are global (non-spatial) coefficients
#' - \eqn{\tilde{X}(s)} is the subset of covariates with SVCs

#' - \eqn{w(s)} are spatially-varying adjustments at location s
#'
#' Each SVC follows an independent Gaussian process:
#' \deqn{w_j(s) \sim GP(0, \sigma^2_j \cdot C(\phi_j))}
#'
#' where \eqn{C(\phi)} is the correlation function with range parameter
#' \eqn{\phi}.
#'
#' **NNGP approximation**: For computational tractability, we use the
#' Nearest Neighbor Gaussian Process (Datta et al., 2016), which conditions
#' each location on its k nearest neighbors. This reduces complexity from
#' O(n^3) to O(n*k^2), enabling models with thousands of locations.
#'
#' **Interpretation**: A positive SVC for depth at location s means the
#' depth effect is stronger at s than the global average. The spatial
#' variance \eqn{\sigma^2_j} quantifies how much the effect varies across
#' space.
#'
#' @examples
#' # Create SVC specification
#' svc_spec <- spatial_svc(~ lon + lat, terms = 1)
#' print(svc_spec)
#'
#' \dontrun{
#' # Generate synthetic spatial data (not run - SVC not fully supported)
#' set.seed(202)
#' n <- 40
#' df <- data.frame(
#'   lon = runif(n, 0, 10),
#'   lat = runif(n, 0, 10),
#'   depth = rnorm(n),
#'   temp = rnorm(n),
#'   count = rpois(n, 20),
#'   effort = rgamma(n, shape = 4, rate = 1)
#' )
#'
#' # Spatially-varying intercept (random spatial field)
#' fit <- tratio(
#'   count | effort ~ depth,
#'   data = df,
#'   family = ratiod_poisson_gamma(),
#'   spatial = spatial_svc(~ lon + lat, terms = 1),
#'   control = list(iter = 200, warmup = 100, chains = 1)
#' )
#' summary(fit)
#'
#' # Extract and plot the spatially-varying coefficients
#' svc_effects <- svc(fit)
#' plot(svc_effects, "depth")
#' }
#'
#' @references
#' Datta, A., Banerjee, S., Finley, A. O., & Gelfand, A. E. (
#' 2016). Hierarchical nearest-neighbor Gaussian process models for large
#' geostatistical datasets. Journal of the American Statistical Association,
#' 111(514), 800-812.
#'
#' @seealso [spatial_car()], [spatial_bym2()] for areal spatial effects,
#'   [svc()] for extracting SVC posteriors
#'
#' @export
spatial_svc <- function(coords,
                        terms = 1,
                        cov = c("exponential", "matern", "gaussian", "spherical"),
                        nn = 15,
                        shared = TRUE,
                        scale_coords = TRUE,
                        approx = c("nngp", "hsgp"),
                        m = 6,
                        c_boundary = 1.5,
                        range = svc_default_range(),
                        parameterization = c("noncentered", "centered")) {

  cov <- match.arg(cov)
  approx <- match.arg(approx)
  parameterization_set <- !missing(parameterization)
  parameterization <- match.arg(parameterization)
  if (approx == "hsgp" && parameterization_set) {
    stop("`parameterization` applies to `approx = \"nngp\"`. The HSGP arm\n",
         "samples standardised basis coefficients already.", call. = FALSE)
  }

  # Parse coordinate specification
  if (inherits(coords, "formula")) {
    coord_vars <- all.vars(coords)
    if (length(coord_vars) != 2) {
      stop("`coords` formula must specify exactly 2 coordinate variables",
           call. = FALSE)
    }
  } else if (is.character(coords) && length(coords) == 2) {
    coord_vars <- coords
  } else {
    stop("`coords` must be a formula (~ lon + lat) or character vector of length 2",
         call. = FALSE)
  }

  # Parse terms specification
  if (inherits(terms, "formula")) {
    # Will be resolved against design matrix later
    terms_spec <- list(type = "formula", formula = terms)
  } else if (is.numeric(terms)) {
    terms_spec <- list(type = "index", indices = as.integer(terms))
  } else if (is.character(terms)) {
    terms_spec <- list(type = "names", names = terms)
  } else {
    stop("`terms` must be a formula, integer vector, or character vector",
         call. = FALSE)
  }

  # Validate nn (only relevant for NNGP)
  if (approx == "nngp") {
    if (!is.numeric(nn) || length(nn) != 1 || nn < 1) {
      stop("`nn` must be a positive integer", call. = FALSE)
    }
    nn <- as.integer(nn)
  }

  # Validate HSGP parameters
  if (approx == "hsgp") {
    if (!is.numeric(m) || length(m) != 1 || m < 3 || m > 50) {
      stop("`m` must be an integer between 3 and 50", call. = FALSE)
    }
    m <- as.integer(m)
    if (!is.numeric(c_boundary) || length(c_boundary) != 1 || c_boundary < 1.0) {
      stop("`c_boundary` must be a number >= 1.0", call. = FALSE)
    }
  }

  # Validate the range prior
  if (!is.numeric(range) || length(range) != 2 || anyNA(range) ||
      range[1] <= 0 || range[1] >= range[2]) {
    stop("`range` must be c(lower, upper) with 0 < lower < upper", call. = FALSE)
  }
  range <- as.numeric(range)
  # The HSGP arm parameterises a lengthscale under LogNormal(0, 1), not a
  # range under this uniform, so a value set here would be read by nothing.
  if (approx == "hsgp" && !identical(range, svc_default_range())) {
    stop("`range` applies to `approx = \"nngp\"`. The HSGP arm carries a\n",
         "LogNormal(0, 1) prior on its lengthscale, which `range` does not set.",
         call. = FALSE)
  }

  # Warning for non-shared SVCs
  if (!shared) {
    warning(
      "Non-shared SVCs (shared = FALSE) may lead to confounded ratio estimates.\n",
      "Consider whether spatially-varying effects should be shared between\n",
      "numerator and denominator to prevent spurious spatial patterns in ratios.",
      call. = FALSE
    )
  }

  structure(
    list(
      type = "svc",
      coord_vars = coord_vars,
      terms_spec = terms_spec,
      cov = cov,
      nn = nn,
      shared = shared,
      scale_coords = scale_coords,
      approx = approx,
      m = m,
      c_boundary = c_boundary,
      range = range,
      parameterization = parameterization,
      # Filled in during validation
      n_obs = NULL,
      n_svc = NULL,
      svc_indices = NULL,
      svc_names = NULL,
      coords_matrix = NULL,
      neighbor_info = NULL
    ),
    class = c("ratiod_svc", "ratiod_spatial", "list")
  )
}


#' Print method for ratiod_svc
#'
#' @param x A ratiod_svc object
#' @param ... Ignored
#'
#' @export
print.ratiod_svc <- function(x, ...) {
  cat("tulpaRatio spatially-varying coefficients\n")
  cat("=====================================\n\n")

  cat("Coordinates:", paste(x$coord_vars, collapse = ", "), "\n")
  cat("Covariance:", x$cov, "\n")
  cat("Neighbors (NNGP):", x$nn, "\n")
  cat("Shared:", if (x$shared) "Yes (enters both processes)" else "No", "\n")
  if (identical(x$approx %||% "nngp", "hsgp")) {
    cat("Lengthscale prior: LogNormal(0, 1)\n")
  } else {
    rng <- x$range %||% svc_default_range()
    cat("Range prior: [", rng[1], ", ", rng[2], "]\n", sep = "")
  }

  if (!is.null(x$n_svc)) {
    cat("\nSVC terms:", x$n_svc, "\n")
    if (!is.null(x$svc_names)) {
      cat("  ", paste(x$svc_names, collapse = ", "), "\n")
    }
  } else {
    cat("\nTerms: ")
    if (x$terms_spec$type == "formula") {
      cat(deparse(x$terms_spec$formula), "\n")
    } else if (x$terms_spec$type == "index") {
      cat("columns ", paste(x$terms_spec$indices, collapse = ", "), "\n")
    } else {
      cat(paste(x$terms_spec$names, collapse = ", "), "\n")
    }
  }

  if (!is.null(x$n_obs)) {
    cat("Observations:", x$n_obs, "\n")
  }

  invisible(x)
}


#' Validate SVC specification against data and design matrix
#'
#' @param svc ratiod_svc object
#' @param data Data frame
#' @param X Design matrix (to resolve term names)
#'
#' @return Updated ratiod_svc object with computed neighbor structure
#' @keywords internal
validate_svc <- function(svc, data, X) {
  if (is.null(svc)) return(NULL)
  if (!inherits(svc, "ratiod_svc")) return(svc)  # Other spatial types

  N <- nrow(data)
  p <- ncol(X)

  # Check coordinate columns exist
  for (cv in svc$coord_vars) {
    if (!(cv %in% names(data))) {
      stop(sprintf("Coordinate variable '%s' not found in data", cv),
           call. = FALSE)
    }
  }

  # Extract coordinates
  coords <- cbind(
    data[[svc$coord_vars[1]]],
    data[[svc$coord_vars[2]]]
  )

  # Check for missing coordinates
  if (any(is.na(coords))) {
    stop("Coordinate columns contain missing values", call. = FALSE)
  }

  # Scale coordinates if requested
  if (svc$scale_coords) {
    coords <- scale(coords)
  }

  # Resolve SVC terms
  coef_names <- colnames(X)
  if (is.null(coef_names)) {
    coef_names <- paste0("V", seq_len(p))
  }

  if (svc$terms_spec$type == "index") {
    svc_indices <- svc$terms_spec$indices
    if (any(svc_indices < 1 | svc_indices > p)) {
      stop(sprintf("SVC term indices must be between 1 and %d", p),
           call. = FALSE)
    }
    svc_names <- coef_names[svc_indices]

  } else if (svc$terms_spec$type == "names") {
    svc_names <- svc$terms_spec$names
    svc_indices <- match(svc_names, coef_names)
    if (any(is.na(svc_indices))) {
      missing <- svc_names[is.na(svc_indices)]
      stop(sprintf("SVC terms not found in design matrix: %s",
                   paste(missing, collapse = ", ")), call. = FALSE)
    }

  } else if (svc$terms_spec$type == "formula") {
    # Parse formula to get terms
    fmla <- svc$terms_spec$formula
    tt <- terms(fmla)
    term_labels <- attr(tt, "term.labels")
    has_intercept <- attr(tt, "intercept") == 1

    svc_names <- character(0)
    if (has_intercept) {
      svc_names <- c(svc_names, "(Intercept)")
    }
    svc_names <- c(svc_names, term_labels)

    svc_indices <- match(svc_names, coef_names)
    if (any(is.na(svc_indices))) {
      missing <- svc_names[is.na(svc_indices)]
      stop(sprintf("SVC terms not found in design matrix: %s",
                   paste(missing, collapse = ", ")), call. = FALSE)
    }
  }

  # Compute nearest neighbors (NNGP only)
  approx <- svc$approx %||% "nngp"
  if (approx == "nngp") {
    nn <- min(svc$nn, N - 1)
    neighbor_info <- compute_nngp_neighbors(coords, nn)
  } else {
    neighbor_info <- NULL
  }

  # Update SVC object
  svc$n_obs <- N
  svc$n_svc <- length(svc_indices)
  svc$svc_indices <- svc_indices
  svc$svc_names <- svc_names
  svc$coords_matrix <- coords
  svc$neighbor_info <- neighbor_info

  # Set spatial parameters for parameter layout
  if (approx == "hsgp") {
    m <- svc$m %||% 6L
    svc$n_spatial <- length(svc_indices) * as.integer(m)^2  # m^2 basis per SVC term
  } else {
    svc$n_spatial <- svc$n_obs * svc$n_svc  # N effects per SVC term
  }

  svc
}


#' Covariance code for the compiled kernel family
#'
#' @description
#' The integer the C++ covariance family is selected by, from the name a
#' `cov = ` argument carries. One mapping, read by every path that hands a
#' covariance choice to compiled code, so a fit and a prediction cannot be made
#' under different kernels. A name the family does not know is an error rather
#' than a silent exponential.
#'
#' @param cov One of "exponential", "matern", "gaussian", "spherical"
#'
#' @return Integer code in 0:3
#'
#' @keywords internal
cov_type_code <- function(cov) {
  known <- c("exponential", "matern", "gaussian", "spherical")
  cov <- cov %||% "exponential"
  code <- match(cov, known)
  if (length(code) != 1L || is.na(code)) {
    stop(sprintf("Unknown covariance '%s'. One of %s.",
                 paste(cov, collapse = ", "),
                 paste(sQuote(known), collapse = ", ")), call. = FALSE)
  }
  as.integer(code - 1L)
}


#' Put new coordinates in the units a GP fit was made in
#'
#' @description
#' `spatial_gp(scale_coords = TRUE)` centres and scales the coordinates before
#' the neighbour structure is built, so the fitted field lives in those units.
#' Prediction coordinates arrive raw and are put in the same units here. A fit
#' made without scaling returns them unchanged.
#'
#' @param gp A validated spatial object
#' @param coords Matrix of new coordinates
#'
#' @return `coords` in the fit's own units
#'
#' @keywords internal
gp_scale_new_coords <- function(gp, coords) {
  coords <- as.matrix(coords)
  center <- gp$coord_center
  scale <- gp$coord_scale
  if (is.null(center) || is.null(scale)) return(coords)
  if (length(center) != ncol(coords) || length(scale) != ncol(coords)) {
    stop(sprintf(paste("The fit was made on %d scaled coordinate columns and",
                       "these have %d."),
                 length(center), ncol(coords)), call. = FALSE)
  }
  sweep(sweep(coords, 2L, center, "-"), 2L, scale, "/")
}


#' Compute nearest neighbors for NNGP
#'
#' @description
#' Compute the k nearest neighbors for each observation using Euclidean
#' distance. Returns in a format suitable for the NNGP likelihood.
#'
#' @param coords N x 2 matrix of coordinates
#' @param k Number of nearest neighbors
#'
#' @return List with:
#'   - `nn_idx`: N x k matrix of neighbor indices (0 for obs with fewer neighbors)
#'   - `nn_dist`: N x k matrix of distances to neighbors
#'   - `nn_order`: Ordering of observations for NNGP (by coordinate)
#'
#' @keywords internal
compute_nngp_neighbors <- function(coords, k) {
  N <- nrow(coords)

  # Order observations (improves NNGP conditioning)
  # Use maximum-minimum distance ordering for better numerical properties
  order_idx <- order(coords[, 1], coords[, 2])

  # Reorder coordinates
  coords_ordered <- coords[order_idx, , drop = FALSE]

  # Compute neighbors for each observation
  nn_idx <- matrix(0L, nrow = N, ncol = k)
  nn_dist <- matrix(Inf, nrow = N, ncol = k)

  # Phase 1.3: Precompute pairwise distances among neighbors
  nn_neighbor_dist <- array(0, dim = c(N, k, k))

  for (i in 2:N) {
    # Only consider previous observations (in ordering) as potential neighbors
    n_candidates <- min(i - 1, k)

    if (n_candidates > 0) {
      # Compute distances to all previous observations
      dists <- sqrt(
        (coords_ordered[1:(i-1), 1] - coords_ordered[i, 1])^2 +
        (coords_ordered[1:(i-1), 2] - coords_ordered[i, 2])^2
      )

      # Find k nearest
      if (length(dists) <= k) {
        nn_order <- order(dists)
        nn_idx[i, seq_len(length(dists))] <- nn_order
        nn_dist[i, seq_len(length(dists))] <- dists[nn_order]
      } else {
        nn_order <- order(dists)[1:k]
        nn_idx[i, ] <- nn_order
        nn_dist[i, ] <- dists[nn_order]
      }

      # Phase 1.3: Compute pairwise distances among neighbors
      n_neighbors <- sum(nn_idx[i, ] > 0)
      if (n_neighbors > 1) {
        neighbor_indices <- nn_idx[i, 1:n_neighbors]
        neighbor_coords <- coords_ordered[neighbor_indices, , drop = FALSE]
        for (j1 in 1:n_neighbors) {
          for (j2 in 1:n_neighbors) {
            if (j1 == j2) {
              nn_neighbor_dist[i, j1, j2] <- 0
            } else {
              nn_neighbor_dist[i, j1, j2] <- sqrt(
                (neighbor_coords[j1, 1] - neighbor_coords[j2, 1])^2 +
                (neighbor_coords[j1, 2] - neighbor_coords[j2, 2])^2
              )
            }
          }
        }
      }
    }
  }

  list(
    nn_idx = nn_idx,
    nn_dist = nn_dist,
    nn_neighbor_dist = nn_neighbor_dist,  # Phase 1.3: cached pairwise distances
    nn_order = order_idx,
    nn_order_inv = order(order_idx),  # Inverse permutation
    k = k
  )
}


#' Extract spatially-varying coefficients from a fitted model
#'
#' @description
#' Extract posterior distributions of spatially-varying coefficients (SVCs)
#' from a fitted ratio model with SVC specification.
#'
#' @param object A `ratiod_fit` object fitted with `svc` argument
#' @param terms Which SVC terms to extract. If NULL (default), extracts all.
#' @param summary Logical; if TRUE, return summary statistics instead of
#'   full posterior draws.
#' @param probs Quantiles to compute if `summary = TRUE`.
#' @param ... Ignored
#'
#' @return A `ratiod_svc_posterior` object containing:
#' - `draws`: Array of posterior draws (draws x locations x terms)
#' - `coords`: Coordinate matrix
#' - `term_names`: Names of SVC terms
#'
#' @examples
#' \dontrun{
#' # Generate synthetic spatial data (not run - SVC not fully supported)
#' set.seed(303)
#' n <- 40
#' df <- data.frame(
#'   lon = runif(n, 0, 10),
#'   lat = runif(n, 0, 10),
#'   depth = rnorm(n),
#'   count = rpois(n, 20),
#'   effort = rgamma(n, shape = 4, rate = 1)
#' )
#'
#' # Fit model with SVC
#' fit <- tratio(
#'   count | effort ~ depth,
#'   data = df,
#'   family = ratiod_poisson_gamma(),
#'   spatial = spatial_svc(~ lon + lat, terms = c(1, 2)),
#'   control = list(iter = 200, warmup = 100, chains = 1)
#' )
#'
#' # Extract SVC posteriors
#' svc_post <- svc(fit)
#' summary(svc_post)
#' }
#'
#' @seealso [spatial_svc()], [plot.ratiod_svc_posterior()]
#'
#' @export
svc <- function(object, terms = NULL, summary = FALSE,
                probs = c(0.025, 0.5, 0.975), ...) {
  UseMethod("svc")
}


#' @rdname svc
#' @export
svc.ratiod_fit <- function(object, terms = NULL, summary = FALSE,
                           probs = c(0.025, 0.5, 0.975), ...) {

  # Check if model has SVCs
  if (is.null(object$svc) || !inherits(object$svc, "ratiod_svc")) {
    stop("Model was not fitted with spatially-varying coefficients.\n",
         "Pass `spatial = spatial_svc(...)` to tratio() to specify SVCs.",
         call. = FALSE)
  }

  svc_info <- object$svc
  n_svc <- svc_info$n_svc
  svc_names <- svc_info$svc_names

  # Get SVC draws from model
  svc_draws <- object$.internal$svc_draws

  if (is.null(svc_draws)) {
    stop("SVC draws not found in model output", call. = FALSE)
  }

  # Subset terms if requested
  if (!is.null(terms)) {
    if (is.numeric(terms)) {
      term_idx <- terms
    } else if (is.character(terms)) {
      term_idx <- match(terms, svc_names)
      if (any(is.na(term_idx))) {
        stop("Terms not found: ", paste(terms[is.na(term_idx)], collapse = ", "),
             call. = FALSE)
      }
    } else {
      stop("`terms` must be numeric or character", call. = FALSE)
    }
    svc_draws <- svc_draws[, , term_idx, drop = FALSE]
    svc_names <- svc_names[term_idx]
    n_svc <- length(term_idx)
  }

  n_obs <- dim(svc_draws)[2]

  result <- structure(
    list(
      draws = svc_draws,
      coords = svc_info$coords_matrix,
      term_names = svc_names,
      n_obs = n_obs,
      n_svc = n_svc,
      n_draws = dim(svc_draws)[1],
      cov = svc_info$cov
    ),
    class = "ratiod_svc_posterior"
  )

  if (summary) {
    return(summary(result, probs = probs))
  }

  result
}


#' Summary method for ratiod_svc_posterior
#'
#' @param object A ratiod_svc_posterior object
#' @param probs Quantiles to compute
#' @param ... Ignored
#'
#' @export
summary.ratiod_svc_posterior <- function(object, probs = c(0.025, 0.5, 0.975), ...) {

  n_obs <- object$n_obs
  n_svc <- object$n_svc
  draws <- object$draws

  results <- list()

  for (j in seq_len(n_svc)) {
    term_draws <- draws[, , j]

    summaries <- data.frame(
      obs = seq_len(n_obs),
      term = object$term_names[j],
      coord_1 = object$coords[, 1],
      coord_2 = object$coords[, 2],
      mean = colMeans(term_draws),
      sd = apply(term_draws, 2, sd),
      t(apply(term_draws, 2, quantile, probs = probs))
    )
    names(summaries)[7:ncol(summaries)] <- paste0("q", probs * 100)
    rownames(summaries) <- NULL

    results[[j]] <- summaries
  }

  result <- do.call(rbind, results)

  structure(
    result,
    n_draws = object$n_draws,
    term_names = object$term_names,
    class = c("ratiod_svc_summary", "data.frame")
  )
}


#' Print method for ratiod_svc_posterior
#'
#' @param x A ratiod_svc_posterior object
#' @param ... Ignored
#'
#' @export
print.ratiod_svc_posterior <- function(x, ...) {
  cat("Spatially-varying coefficient posterior\n")
  cat("=======================================\n\n")
  cat("Terms:", paste(x$term_names, collapse = ", "), "\n")
  cat("Locations:", x$n_obs, "\n")
  cat("Posterior draws:", x$n_draws, "\n")
  cat("Covariance function:", x$cov, "\n")
  cat("\nUse summary() for posterior summaries\n")
  cat("Use plot() for spatial visualization\n")
  invisible(x)
}


#' Plot method for ratiod_svc_posterior
#'
#' @param x A ratiod_svc_posterior object
#' @param term Which term to plot (name or index). Default: first term.
#' @param type Plot type: "mean" (default), "sd", or quantile (e.g., "q50")
#' @param ... Additional arguments passed to plotting functions
#'
#' @export
plot.ratiod_svc_posterior <- function(x, term = 1, type = "mean", ...) {

  if (is.character(term)) {
    term_idx <- match(term, x$term_names)
    if (is.na(term_idx)) {
      stop("Term not found: ", term, call. = FALSE)
    }
  } else {
    term_idx <- term
  }

  term_name <- x$term_names[term_idx]
  draws <- x$draws[, , term_idx]

  # Compute summary to plot
  if (type == "mean") {
    values <- colMeans(draws)
    title <- paste("SVC:", term_name, "(posterior mean)")
  } else if (type == "sd") {
    values <- apply(draws, 2, sd)
    title <- paste("SVC:", term_name, "(posterior SD)")
  } else if (grepl("^q[0-9]+", type)) {
    prob <- as.numeric(gsub("q", "", type)) / 100
    values <- apply(draws, 2, quantile, probs = prob)
    title <- paste0("SVC: ", term_name, " (", type, ")")
  } else {
    stop("Unknown plot type: ", type, call. = FALSE)
  }

  coords <- x$coords

  # Use ggplot2 if available
  if (requireNamespace("ggplot2", quietly = TRUE)) {
    df <- data.frame(
      x = coords[, 1],
      y = coords[, 2],
      value = values
    )

    p <- ggplot2::ggplot(df, ggplot2::aes(x = .data$x, y = .data$y, color = .data$value)) +
      ggplot2::geom_point(size = 2, ...) +
      ggplot2::scale_color_viridis_c() +
      ggplot2::labs(
        title = title,
        x = "Coordinate 1",
        y = "Coordinate 2",
        color = "Effect"
      ) +
      ggplot2::coord_fixed() +
      theme_ratiod()

    return(p)
  }

  # Base R fallback
  col_ramp <- colorRampPalette(c("blue", "white", "red"))
  n_colors <- 100
  colors <- col_ramp(n_colors)

  val_range <- range(values)
  val_scaled <- (values - val_range[1]) / diff(val_range)
  val_scaled[is.na(val_scaled)] <- 0.5
  point_colors <- colors[pmax(1, pmin(n_colors, ceiling(val_scaled * n_colors)))]

  plot(coords[, 1], coords[, 2],
       col = point_colors,
       pch = 19,
       xlab = "Coordinate 1",
       ylab = "Coordinate 2",
       main = title,
       asp = 1,
       ...)

  invisible(NULL)
}


#' Validate spatial specification against data
#'
#' @param spatial ratiod_spatial object
#' @param data Data frame
#'
#' @return NULL (invisibly); errors if validation fails
#' @keywords internal
validate_spatial <- function(spatial, data) {
  if (is.null(spatial)) return(invisible(NULL))

  # SVC validation is handled separately via validate_svc()
  if (inherits(spatial, "ratiod_svc")) {
    return(invisible(NULL))
  }

  # Check group variable exists
  if (spatial$level == "group") {
    if (!(spatial$group_var %in% names(data))) {
      stop(sprintf("Spatial group variable '%s' not found in data",
                   spatial$group_var), call. = FALSE)
    }

    # Check number of groups matches adjacency matrix
    n_groups <- length(unique(data[[spatial$group_var]]))
    if (n_groups != spatial$n_spatial) {
      stop(sprintf(
        "Number of groups in data (%d) does not match adjacency matrix (%d)",
        n_groups, spatial$n_spatial
      ), call. = FALSE)
    }
  } else {
    # Observation-level spatial
    if (nrow(data) != spatial$n_spatial) {
      stop(sprintf(
        "Number of observations (%d) does not match adjacency matrix (%d)",
        nrow(data), spatial$n_spatial
      ), call. = FALSE)
    }
  }

  # Check connectivity
  if (!is_connected(spatial$adjacency)) {
    warning(
      "Spatial adjacency graph is not fully connected.\n",
      "This may cause identifiability issues. Consider:\n",
      "  - Adding edges to connect isolated components\n",
      "  - Fitting separate models for each connected component",
      call. = FALSE
    )
  }

  invisible(NULL)
}


# =============================================================================
# Spatial Confounding Mitigation
# =============================================================================

#' Restricted Spatial Regression (RSR)
#'
#' @description
#' Apply Restricted Spatial Regression to mitigate spatial confounding.
#' RSR orthogonalizes the spatial effect to the covariate space, preventing
#' the spatial random effect from absorbing covariate information.
#'
#' This is important when covariates are spatially smooth (e.g., climate
#' variables, elevation) because the spatial random effect can "steal"
#' variance from these covariates, leading to biased coefficient estimates.
#'
#' @param spatial A spatial specification (`spatial_gp`, `spatial_car`, etc.)
#' @param restrict_to Formula specifying which covariates to orthogonalize
#'   against (e.g., `~ depth + temp`). The spatial effect will be constrained
#'   to be orthogonal to the column space of these covariates.
#'
#' @return A modified spatial specification with RSR enabled
#'
#' @details
#' The RSR approach (Reich et al., 2006; Hodges & Reich, 2010) modifies the
#' spatial random effect to be orthogonal to the fixed effect design matrix:
#'
#' \deqn{w_{RSR} = (I - P_X) w}
#'
#' where \eqn{P_X = X(X'X)^{-1}X'} is the projection matrix onto the column
#' space of X.
#'
#' **When to use RSR:**
#' - Covariates are spatially smooth (environmental gradients)
#' - Interested in causal interpretation of covariate effects
#' - Coefficients appear attenuated toward zero
#'
#' **When NOT to use RSR:**
#' - Covariates are spatially uncorrelated
#' - Spatial effect is the primary quantity of interest
#' - Prediction is the main goal (not causal inference)
#'
#' @examples
#' # Create RSR spatial structure
#' rsr <- spatial_rsr(
#'   spatial_gp(~ lon + lat),
#'   restrict_to = ~ depth + temp
#' )
#' print(rsr)
#'
#' \donttest{
#' # Generate synthetic spatial data with confounding
#' set.seed(404)
#' n <- 50
#' lon <- runif(n, 0, 10)
#' lat <- runif(n, 0, 10)
#' # Make depth and temp spatially correlated
#' df <- data.frame(
#'   lon = lon,
#'   lat = lat,
#'   depth = lon/5 + rnorm(n, 0, 0.5),  # Correlated with lon
#'   temp = lat/5 + rnorm(n, 0, 0.5),   # Correlated with lat
#'   count = rpois(n, 25),
#'   effort = rgamma(n, shape = 4, rate = 1)
#' )
#'
#' # Standard GP (may have spatial confounding)
#' fit1 <- tratio(
#'   count | effort ~ depth + temp,
#'   data = df,
#'   spatial = spatial_gp(~ lon + lat),
#'   mode = "hmc",
#'   control = list(iter = 200, warmup = 100, chains = 1)
#' )
#'
#' # RSR to protect depth and temp coefficients
#' fit2 <- tratio(
#'   count | effort ~ depth + temp,
#'   data = df,
#'   spatial = spatial_rsr(
#'     spatial_gp(~ lon + lat),
#'     restrict_to = ~ depth + temp
#'   ),
#'   mode = "hmc",
#'   control = list(iter = 200, warmup = 100, chains = 1)
#' )
#'
#' # Compare coefficient estimates
#' summary(fit1)  # May be attenuated
#' summary(fit2)  # Protected from spatial confounding
#' }
#'
#' @references
#' Reich, B. J., Hodges, J. S., & Zadnik, V. (2006). Effects of residual
#' smoothing on the posterior of the fixed effects in disease-mapping models.
#' Biometrics, 62(4), 1197-1206.
#'
#' Hodges, J. S., & Reich, B. J. (2010). Adding spatially-correlated errors
#' can mess up the fixed effect you love. The American Statistician, 64(4),
#' 325-334.
#'
#' @seealso [spatial_gp()], [spatial_car()]
#'
#' @export
spatial_rsr <- function(spatial, restrict_to) {

  if (!inherits(spatial, "ratiod_spatial")) {
    stop("`spatial` must be a tulpaRatio spatial specification", call. = FALSE)
  }

  if (!inherits(restrict_to, "formula")) {
    stop("`restrict_to` must be a formula", call. = FALSE)
  }

  # Store RSR information in the spatial object
  spatial$rsr <- TRUE
  spatial$rsr_formula <- restrict_to

  # Add RSR class for dispatch
  class(spatial) <- c("ratiod_rsr", class(spatial))

  spatial
}


#' Print method for ratiod_rsr
#'
#' @param x A ratiod_rsr object
#' @param ... Passed to underlying print method
#'
#' @export
print.ratiod_rsr <- function(x, ...) {
  # Print underlying spatial type
  NextMethod()

  cat("\nRestricted Spatial Regression (RSR):\n")
  cat("  Orthogonal to:", deparse(x$rsr_formula), "\n")
  cat("  (Spatial effect constrained to be orthogonal to covariate space)\n")

  invisible(x)
}


#' Compute RSR projection matrix
#'
#' @description
#' Compute the orthogonal projection matrix P_perp = I - P_X that projects
#' the spatial effect into the space orthogonal to the covariates.
#'
#' @param X Design matrix of covariates to orthogonalize against
#'
#' @return Projection matrix (n x n)
#' @keywords internal
compute_rsr_projection <- function(X) {
  Q <- rsr_basis_of(X)
  diag(nrow(X)) - Q %*% t(Q)
}


#' Orthonormal basis of the restricted covariate space
#'
#' @description
#' The columns of `Q` span the column space of `X`, one column per unit of its
#' numerical rank, so `I - Q Q'` is the projection onto the orthogonal
#' complement.
#'
#' @param X Design matrix of covariates to orthogonalize against
#' @return An `n x rank(X)` matrix with orthonormal columns
#' @keywords internal
rsr_basis_of <- function(X) {
  if (ncol(X) >= nrow(X)) {
    warning("More covariates than observations; RSR may not be effective",
            call. = FALSE)
  }
  qr_X <- qr(X)
  qr.Q(qr_X)[, seq_len(qr_X$rank), drop = FALSE]
}


#' Restricted-covariate basis of a spatial specification
#'
#' @param spatial A spatial specification
#' @param data Data frame
#' @return The basis from [rsr_basis_of()] for a `spatial_rsr()`
#'   specification, `NULL` otherwise
#' @keywords internal
rsr_basis <- function(spatial, data) {
  if (!inherits(spatial, "ratiod_rsr")) return(NULL)
  rsr_vars <- all.vars(spatial$rsr_formula)
  missing_vars <- setdiff(rsr_vars, names(data))
  if (length(missing_vars) > 0) {
    stop(sprintf("RSR variables not found in data: %s",
                 paste(missing_vars, collapse = ", ")), call. = FALSE)
  }
  rsr_basis_of(model.matrix(spatial$rsr_formula, data = data))
}


#' Validate RSR specification
#'
#' @param spatial ratiod_rsr object
#' @param data Data frame
#' @param formula Model formula (to extract design matrix)
#'
#' @return Updated spatial object with projection matrix
#' @keywords internal
validate_rsr <- function(spatial, data, formula) {
  if (is.null(spatial) || !inherits(spatial, "ratiod_rsr")) {
    return(spatial)
  }

  Q <- rsr_basis(spatial, data)
  spatial$rsr_projection <- diag(nrow(Q)) - Q %*% t(Q)
  spatial$rsr_vars <- all.vars(spatial$rsr_formula)

  spatial
}


#' Apply RSR projection to spatial effect
#'
#' @description
#' Project spatial effect into the space orthogonal to covariates.
#' Called during posterior computation.
#'
#' @param w Spatial effect vector (length n)
#' @param P_perp Projection matrix from compute_rsr_projection
#'
#' @return Projected spatial effect (length n)
#' @keywords internal
apply_rsr_projection <- function(w, P_perp) {
  as.vector(P_perp %*% w)
}
