#' Temporal structure specifications for tulpaRatio
#'
#' @description
#' Functions to specify temporal random effects for ratio models.
#' Temporal effects are shared between numerator and denominator by default,
#' which helps prevent spurious ratio effects from temporally-structured
#' unmeasured confounders.
#'
#' @name ratiod_temporal
NULL


#' The single-component temporal specification every constructor returns
#'
#' @description
#' `temporal_rw1()`, `temporal_rw2()`, `temporal_ar1()` and `temporal_iid()`
#' differ in the `type` string the sampler dispatches on and in the extra
#' fields their structure carries. Everything else -- how a time or group
#' variable is read from a formula or a name, the warning a non-shared effect
#' earns, and the shape of the returned object -- is settled here, so a new
#' structure is a `type` string and its own fields.
#'
#' @param type The structure's name, as the C++ temporal parser reads it
#' @param time_var Time variable, formula or name
#' @param group_var Optional grouping variable, formula or name
#' @param cyclic Whether the random walk wraps around
#' @param shared Whether the effect enters both linear predictors
#' @param extra Named list of fields specific to this structure
#'
#' @return A `ratiod_temporal` object
#' @keywords internal
new_ratiod_temporal <- function(type, time_var, group_var = NULL,
                                cyclic = FALSE, shared = TRUE,
                                extra = list()) {

  time_var <- temporal_var_name(time_var, "time_var")
  if (!is.null(group_var)) {
    group_var <- temporal_var_name(group_var, "group_var")
  }

  if (!shared) {
    warning(
      "Non-shared temporal effects (shared = FALSE) may lead to confounded ratio estimates.\n",
      "Consider whether temporal effects should be shared between\n",
      "numerator and denominator to prevent spurious temporal patterns in ratios.",
      call. = FALSE
    )
  }

  structure(
    c(
      list(
        type = type,
        time_var = time_var,
        group_var = group_var,
        cyclic = cyclic,
        shared = shared
      ),
      extra,
      list(
        # Filled in when validated against data
        n_times = NULL,
        n_groups = NULL,
        time_index = NULL,
        group_index = NULL,
        precision_structure = NULL
      )
    ),
    class = c("ratiod_temporal", "list")
  )
}


#' Read a variable name from a one-sided formula or a character scalar
#'
#' @param x A formula (`~ year`) or a single name
#' @param arg The argument name, for the error message
#'
#' @return A single character string
#' @keywords internal
temporal_var_name <- function(x, arg) {
  if (inherits(x, "formula")) {
    v <- all.vars(x)
    if (length(v) != 1) {
      stop(sprintf("`%s` formula must specify exactly 1 variable", arg),
           call. = FALSE)
    }
    return(v)
  }
  if (!is.character(x) || length(x) != 1) {
    stop(sprintf("`%s` must be a formula (~ %s) or single character string",
                 arg, arg), call. = FALSE)
  }
  x
}


#' RW1 temporal structure (First-order Random Walk)
#'
#' @description
#' Specify a first-order random walk temporal random effect.
#' RW1 assumes that the difference between consecutive time points is
#' normally distributed: `phi[t] - phi[t-1] ~ N(0, sigma^2)`.
#'
#' This is an intrinsic model (improper prior) that captures smooth
#' temporal trends. It is equivalent to an ICAR model on a 1D chain.
#'
#' @param time_var Name of the time variable in data. Can be a formula
#'   (e.g., `~ year`) or a character string (e.g., `"year"`).
#' @param group_var Optional name of grouping variable for panel data.
#'   If provided, separate random walks are estimated for each group.
#' @param cyclic Logical; if TRUE, the random walk wraps around (first and
#'   last time points are neighbors). Useful for seasonal data.
#' @param shared Logical; if TRUE (default), temporal effect enters both
#'   numerator and denominator linear predictors identically.
#' @param parameterization Coordinate the walk is sampled in. `"centered"`
#'   (default) samples the effects themselves; `"noncentered"` samples
#'   `z ~ N(0, I)` and reaches the effects through `sigma * A^-1 z`, where `A`
#'   stacks the walk's difference operator on `1'/sqrt(n)`, so the variance
#'   enters the linear predictor through the transform rather than through a
#'   quadratic form. The two describe the same model and differ in the geometry
#'   the sampler sees: non-centred helps where the variance is shrunk toward
#'   zero and leaves a funnel the centred coordinate cannot descend, and can be
#'   worse where the data pin the field. Measured on a 500-observation fixture
#'   with no temporal signal, an RW1's bulk ESS for `tau_temporal` goes 409 to
#'   805, while an RW2 picks up 23 divergences of 2000 for no gain, which is
#'   why this is an argument rather than a default. With `group_var` the prior
#'   still augments one global constant, so the transform carries the group
#'   contrasts as free directions beside the scaled ones and the group levels
#'   stay unshrunk, as they are in the centred coordinate.
#'
#' @return A `ratiod_temporal` object
#'
#' @details
#' The RW1 precision matrix Q has the form (for T time points):
#' \deqn{Q[t,t] = 1 \text{ for } t = 1, T}
#' \deqn{Q[t,t] = 2 \text{ for } 1 < t < T}
#' \deqn{Q[t,t-1] = Q[t-1,t] = -1}
#'
#' This is a rank-deficient matrix (rank T-1), so a sum-to-zero constraint
#' is applied for identifiability.
#'
#' @examples
#' # Create temporal RW1 specification
#' temporal_rw1("year")
#' temporal_rw1("month", cyclic = TRUE)
#'
#' \donttest{
#' # Simple temporal trend
#' set.seed(123)
#' df <- data.frame(
#'   year = rep(2010:2021, each = 4),
#'   x = rnorm(48),
#'   count = rpois(48, lambda = 20),
#'   effort = rgamma(48, shape = 5, rate = 1)
#' )
#'
#' fit <- tratio(
#'   count | effort ~ x,
#'   data = df,
#'   family = ratiod_poisson_gamma(),
#'   temporal = temporal_rw1("year"),
#'   mode = "hmc",
#'   control = list(iter = 200, warmup = 100, chains = 1)
#' )
#'
#' # Panel data: separate trends per site
#' set.seed(124)
#' df_panel <- data.frame(
#'   year = rep(2015:2026, each = 5),
#'   site = rep(paste0("site", 1:5), times = 12),
#'   x = rnorm(60),
#'   count = rpois(60, lambda = 15),
#'   effort = rgamma(60, shape = 3, rate = 1)
#' )
#'
#' fit2 <- tratio(
#'   count | effort ~ x + (1 | site),
#'   data = df_panel,
#'   family = ratiod_poisson_gamma(),
#'   temporal = temporal_rw1("year", group_var = "site"),
#'   mode = "hmc",
#'   control = list(iter = 200, warmup = 100, chains = 1)
#' )
#'
#' # Cyclic for monthly seasonal data
#' set.seed(125)
#' df_monthly <- data.frame(
#'   month = rep(1:12, times = 4),
#'   x = rnorm(48),
#'   count = rpois(48, lambda = 10),
#'   effort = rgamma(48, shape = 2, rate = 1)
#' )
#'
#' fit3 <- tratio(
#'   count | effort ~ x,
#'   data = df_monthly,
#'   family = ratiod_poisson_gamma(),
#'   temporal = temporal_rw1("month", cyclic = TRUE),
#'   mode = "hmc",
#'   control = list(iter = 200, warmup = 100, chains = 1)
#' )
#' }
#'
#' @export
temporal_rw1 <- function(time_var, group_var = NULL, cyclic = FALSE,
                         shared = TRUE,
                         parameterization = c("centered", "noncentered")) {
  parameterization <- match.arg(parameterization)
  new_ratiod_temporal("rw1", time_var, group_var, cyclic = cyclic,
                      shared = shared,
                      extra = list(parameterization = parameterization))
}


#' RW2 temporal structure (Second-order Random Walk)
#'
#' @description
#' Specify a second-order random walk temporal random effect.
#' RW2 penalizes deviations from linearity, resulting in smoother trends
#' than RW1: `phi[t] - 2*phi[t-1] + phi[t-2] ~ N(0, sigma^2)`.
#'
#' @inheritParams temporal_rw1
#'
#' @return A `ratiod_temporal` object
#'
#' @details
#' RW2 produces smoother curves than RW1 because it penalizes the second
#' derivative (curvature) rather than the first derivative (slope).
#' It requires at least 3 time points.
#'
#' The precision matrix is rank T-2 (two constraints needed).
#'
#' @examples
#' # Create temporal RW2 specification
#' temporal_rw2("year")
#'
#' \donttest{
#' # Smooth temporal trend
#' set.seed(126)
#' df <- data.frame(
#'   year = rep(2008:2027, each = 2),
#'   x = rnorm(40),
#'   count = rpois(40, lambda = 25),
#'   effort = rgamma(40, shape = 4, rate = 1)
#' )
#'
#' fit <- tratio(
#'   count | effort ~ x,
#'   data = df,
#'   family = ratiod_poisson_gamma(),
#'   temporal = temporal_rw2("year"),
#'   mode = "hmc",
#'   control = list(iter = 200, warmup = 100, chains = 1)
#' )
#' }
#'
#' @export
temporal_rw2 <- function(time_var, group_var = NULL, cyclic = FALSE,
                         shared = TRUE,
                         parameterization = c("centered", "noncentered")) {
  parameterization <- match.arg(parameterization)
  new_ratiod_temporal("rw2", time_var, group_var, cyclic = cyclic,
                      shared = shared,
                      extra = list(parameterization = parameterization))
}


#' AR1 temporal structure (First-order Autoregressive)
#'
#' @description
#' Specify a first-order autoregressive temporal random effect.
#' AR1 models temporal correlation where each time point depends on
#' the previous one: `phi[t] = rho * phi[t-1] + epsilon[t]`.
#'
#' Unlike RW1/RW2, AR1 is a proper (stationary) model with an estimated
#' autocorrelation parameter rho.
#'
#' @inheritParams temporal_rw1
#' @param parameterization Coordinate the field is sampled in. `"centered"`
#'   (default) samples the effects themselves; `"noncentered"` samples the
#'   `N(0, I)` innovations the AR1 recursion builds the effects from, so both
#'   the precision and rho reach the linear predictor through the transform.
#'   AR1 is proper and stationary, so this is a different construction from the
#'   intrinsic walks' `sigma * A^-1 z`.
#' @param rho_prior Prior on the autocorrelation, as a [prior_beta()] on
#'   `(rho + 1) / 2`. `NULL` takes the model-wide
#'   `ratiod_priors(rho_temporal = )`, which defaults to `Beta(2, 2)`.
#'
#' @return A `ratiod_temporal` object
#'
#' @details
#' The AR1 process has marginal variance sigma^2 / (1 - rho^2) and
#' correlation between time points t and s of rho^|t-s|.
#'
#' The precision matrix is tridiagonal and full rank, so no constraints
#' are needed.
#'
#' `rho` is estimated on (-1, 1), the range every AR1 correlation in the
#' package is sampled on, so an alternating series is reachable. The sampled
#' coordinate is `logit((rho + 1) / 2)`, which is what `rho_prior` places its
#' Beta on.
#'
#' @examples
#' # Create temporal AR1 specification
#' temporal_ar1("year")
#' temporal_ar1("year", group = "site")
#'
#' \donttest{
#' # AR1 temporal correlation
#' set.seed(127)
#' df <- data.frame(
#'   year = rep(2005:2024, each = 3),
#'   x = rnorm(60),
#'   count = rpois(60, lambda = 18),
#'   effort = rgamma(60, shape = 3.5, rate = 1)
#' )
#'
#' fit <- tratio(
#'   count | effort ~ x,
#'   data = df,
#'   family = ratiod_poisson_gamma(),
#'   temporal = temporal_ar1("year"),
#'   mode = "hmc",
#'   control = list(iter = 200, warmup = 100, chains = 1)
#' )
#'
#' # Panel data with group-specific AR1
#' set.seed(128)
#' df_panel <- data.frame(
#'   year = rep(2012:2027, each = 3),
#'   site = rep(paste0("site", 1:3), times = 16),
#'   x = rnorm(48),
#'   count = rpois(48, lambda = 12),
#'   effort = rgamma(48, shape = 2.5, rate = 1)
#' )
#'
#' fit2 <- tratio(
#'   count | effort ~ x,
#'   data = df_panel,
#'   family = ratiod_poisson_gamma(),
#'   temporal = temporal_ar1("year", group_var = "site"),
#'   mode = "hmc",
#'   control = list(iter = 200, warmup = 100, chains = 1)
#' )
#' }
#'
#' @export
temporal_ar1 <- function(time_var, group_var = NULL, shared = TRUE,
                         rho_prior = NULL,
                         parameterization = c("centered", "noncentered")) {

  parameterization <- match.arg(parameterization)

  new_ratiod_temporal(
    "ar1", time_var, group_var,
    cyclic = FALSE,  # AR1 is not cyclic
    shared = shared,
    extra = list(
      rho_prior = validate_rho_prior(rho_prior),
      parameterization = parameterization
    )
  )
}


#' IID temporal structure (Independent Effects)
#'
#' @description
#' Specify an independent temporal random effect: each time point gets its own
#' coefficient with `phi[t] ~ N(0, sigma^2)`, no dependence between them.
#'
#' Use it for a year or season effect that varies without a trend, and as the
#' unstructured comparison against which `temporal_rw1()`, `temporal_rw2()`
#' and `temporal_ar1()` add smoothness.
#'
#' @inheritParams temporal_rw1
#'
#' @return A `ratiod_temporal` object
#'
#' @details
#' The precision matrix is `tau * I`: diagonal and full rank, so no constraint
#' is needed and no ordering of the time points is used. A model whose time
#' points carry no meaningful order fits the same way here as one whose do.
#'
#' @examples
#' # Create temporal IID specification
#' temporal_iid("year")
#'
#' \donttest{
#' # Unstructured year effects
#' set.seed(129)
#' df <- data.frame(
#'   year = rep(2010:2029, each = 2),
#'   x = rnorm(40),
#'   count = rpois(40, lambda = 22),
#'   effort = rgamma(40, shape = 4, rate = 1)
#' )
#'
#' fit <- tratio(
#'   count | effort ~ x,
#'   data = df,
#'   family = ratiod_poisson_gamma(),
#'   temporal = temporal_iid("year"),
#'   mode = "hmc",
#'   control = list(iter = 200, warmup = 100, chains = 1)
#' )
#' }
#'
#' @export
temporal_iid <- function(time_var, group_var = NULL, shared = TRUE) {
  new_ratiod_temporal("iid", time_var, group_var, cyclic = FALSE,
                      shared = shared)
}


#' Print method for ratiod_temporal
#'
#' @param x A ratiod_temporal object
#' @param ... Ignored
#'
#' @export
print.ratiod_temporal <- function(x, ...) {
 cat("tulpaRatio temporal specification\n")
 cat("============================\n\n")

 type_name <- switch(x$type,
   rw1 = "RW1 (First-order Random Walk)",
   rw2 = "RW2 (Second-order Random Walk)",
   ar1 = "AR1 (First-order Autoregressive)",
   iid = "IID (Independent Effects)",
   x$type
 )

 cat("Type:", type_name, "\n")
 cat("Time variable:", x$time_var, "\n")

 if (!is.null(x$group_var)) {
   cat("Group variable:", x$group_var, "\n")
 }

 if (x$type %in% c("rw1", "rw2") && x$cyclic) {
   cat("Cyclic: Yes\n")
 }

 if (identical(x$type, "ar1")) {
   cat("Parameterization:", x$parameterization %||% "centered", "\n")
 }

 cat("Shared:", if (x$shared) "Yes (enters both processes)" else "No", "\n")

 if (!is.null(x$n_times)) {
   cat("Time points:", x$n_times, "\n")
 }
 if (!is.null(x$n_groups) && x$n_groups > 1) {
   cat("Groups:", x$n_groups, "\n")
 }

 invisible(x)
}


#' Validate temporal specification against data
#'
#' @param temporal ratiod_temporal object
#' @param data Data frame
#'
#' @return Updated ratiod_temporal object with indices computed
#' @keywords internal
validate_temporal <- function(temporal, data) {
 if (is.null(temporal)) return(NULL)

 # Check time variable exists
 if (!(temporal$time_var %in% names(data))) {
   stop(sprintf("Temporal variable '%s' not found in data",
                temporal$time_var), call. = FALSE)
 }

 # Check group variable exists if specified
 if (!is.null(temporal$group_var)) {
   if (!(temporal$group_var %in% names(data))) {
     stop(sprintf("Temporal group variable '%s' not found in data",
                  temporal$group_var), call. = FALSE)
   }
 }

 # Get time values and create indices
 time_vals <- data[[temporal$time_var]]

 # Convert to factor to get consistent indexing
 if (is.factor(time_vals)) {
   time_factor <- time_vals
 } else {
   # Sort unique values to ensure temporal ordering
   unique_times <- sort(unique(time_vals))
   time_factor <- factor(time_vals, levels = unique_times)
 }

 temporal$n_times <- nlevels(time_factor)
 temporal$time_index <- as.integer(time_factor)
 temporal$time_levels <- levels(time_factor)

 # Check minimum time points for RW2
 if (temporal$type == "rw2" && temporal$n_times < 3) {
   stop("RW2 requires at least 3 time points", call. = FALSE)
 }

 # Handle grouping
 if (!is.null(temporal$group_var)) {
   group_vals <- data[[temporal$group_var]]
   group_factor <- as.factor(group_vals)
   temporal$n_groups <- nlevels(group_factor)
   temporal$group_index <- as.integer(group_factor)
   temporal$group_levels <- levels(group_factor)

   # Total temporal parameters = n_times * n_groups
   temporal$n_temporal_params <- temporal$n_times * temporal$n_groups
 } else {
   temporal$n_groups <- 1L
   temporal$group_index <- rep(1L, nrow(data))
   temporal$n_temporal_params <- temporal$n_times
 }

 # Build precision structure
 temporal$precision_structure <- build_temporal_precision_structure(temporal)

 temporal
}


#' Build temporal precision matrix structure
#'
#' @param temporal Validated ratiod_temporal object
#'
#' @return List with precision matrix information
#' @keywords internal
build_temporal_precision_structure <- function(temporal) {
 T <- temporal$n_times
 cyclic <- temporal$cyclic

 if (temporal$type == "rw1") {
   # RW1 precision matrix (tridiagonal)
   # Q[t,t] = 2 (interior), 1 (boundary)
   # Q[t,t-1] = Q[t,t+1] = -1

   if (cyclic) {
     # Cyclic: all diagonal = 2, wrap around
     diag_vals <- rep(2, T)
     off_diag <- rep(-1, T)  # Including wrap-around
   } else {
     # Non-cyclic: boundary diagonal = 1
     diag_vals <- c(1, rep(2, T - 2), 1)
     off_diag <- rep(-1, T - 1)
   }

   list(
     type = "rw1",
     T = T,
     cyclic = cyclic,
     diag = diag_vals,
     off_diag = off_diag,
     rank_deficiency = if (cyclic) 1 else 1  # Always rank T-1
   )

 } else if (temporal$type == "rw2") {
   # RW2 precision matrix (pentadiagonal)
   # Second differences

   if (cyclic) {
     diag_vals <- rep(6, T)
     off_diag_1 <- rep(-4, T)  # First off-diagonal
     off_diag_2 <- rep(1, T)   # Second off-diagonal
   } else {
     # Non-cyclic boundary handling
     diag_vals <- c(1, 5, rep(6, T - 4), 5, 1)
     if (T == 3) diag_vals <- c(1, 2, 1)
     if (T == 4) diag_vals <- c(1, 5, 5, 1)
     off_diag_1 <- c(-2, rep(-4, T - 3), -2)
     if (T == 3) off_diag_1 <- c(-2, -2)
     off_diag_2 <- rep(1, T - 2)
   }

   list(
     type = "rw2",
     T = T,
     cyclic = cyclic,
     diag = diag_vals,
     off_diag_1 = off_diag_1,
     off_diag_2 = off_diag_2,
     rank_deficiency = if (cyclic) 2 else 2  # Rank T-2
   )

 } else if (temporal$type == "ar1") {
   # AR1 precision matrix (tridiagonal, full rank)
   # Depends on rho, so just store structure

   list(
     type = "ar1",
     T = T,
     # Full precision matrix constructed at runtime with rho
     rank_deficiency = 0  # Full rank
   )

 } else if (temporal$type == "iid") {
   # IID precision matrix: tau * I, diagonal and full rank

   list(
     type = "iid",
     T = T,
     cyclic = FALSE,
     diag = rep(1, T),
     off_diag = numeric(0),
     rank_deficiency = 0  # Full rank
   )
 }
}


#' Build RW1 precision matrix
#'
#' @param T Number of time points
#' @param tau Precision parameter (1/sigma^2)
#' @param cyclic Whether to use cyclic boundary
#'
#' @return Sparse precision matrix
#' @keywords internal
build_rw1_precision <- function(T, tau = 1, cyclic = FALSE) {
 if (T < 2) {
   stop("RW1 requires at least 2 time points", call. = FALSE)
 }

 # Build as dense then convert (small matrices anyway)
 Q <- matrix(0, T, T)

 if (cyclic) {
   for (t in 1:T) {
     Q[t, t] <- 2
     next_t <- if (t == T) 1 else t + 1
     prev_t <- if (t == 1) T else t - 1
     Q[t, next_t] <- -1
     Q[t, prev_t] <- -1
   }
 } else {
   # Non-cyclic
   Q[1, 1] <- 1
   Q[1, 2] <- -1
   Q[T, T] <- 1
   Q[T, T - 1] <- -1

   if (T > 2) {
     for (t in 2:(T - 1)) {
       Q[t, t] <- 2
       Q[t, t - 1] <- -1
       Q[t, t + 1] <- -1
     }
   }
 }

 Q * tau
}


#' Build AR1 precision matrix
#'
#' @param T Number of time points
#' @param rho Autocorrelation parameter (-1 < rho < 1)
#' @param tau Marginal precision (1/sigma^2)
#'
#' @return Precision matrix
#' @keywords internal
build_ar1_precision <- function(T, rho, tau = 1) {
 if (T < 2) {
   stop("AR1 requires at least 2 time points", call. = FALSE)
 }

 if (abs(rho) >= 1) {
   stop("rho must be in (-1, 1) for stationarity", call. = FALSE)
 }

 # AR1 precision matrix
 # Q = tau * (1 - rho^2)^(-1) * tridiag structure
 # Diagonal: 1 + rho^2 (interior), 1 (boundary)
 # Off-diagonal: -rho

 Q <- matrix(0, T, T)

 # Boundary
 Q[1, 1] <- 1
 Q[T, T] <- 1

 # Interior
 if (T > 2) {
   for (t in 2:(T - 1)) {
     Q[t, t] <- 1 + rho^2
   }
 }

 # Off-diagonal
 for (t in 1:(T - 1)) {
   Q[t, t + 1] <- -rho
   Q[t + 1, t] <- -rho
 }

 # Scale by precision / (1 - rho^2)
 # The conditional variance is sigma^2 * (1 - rho^2) for interior
 # So precision matrix should give marginal precision tau
 Q * tau / (1 - rho^2)
}


#' Multi-Scale temporal structure
#'
#' @description
#' Specify a multi-scale temporal random effect that decomposes temporal
#' variation into trend, seasonal, and short-term components. Each component
#' has its own variance parameter and structure.
#'
#' This is particularly useful for long time series where patterns exist at
#' multiple temporal scales (e.g., annual cycles, decadal trends).
#'
#' @param time_var Name of the time variable in data. Can be a formula
#'   (e.g., `~ year`) or a character string (e.g., `"year"`).
#' @param trend Type of trend component: `"rw2"` (default, smooth),
#'   `"rw1"` (less smooth), or `"none"`.
#' @param seasonal Period for seasonal component (integer). Set to `NULL`
#'   or `0` for no seasonal component. Common values: 12 (monthly), 52 (weekly).
#' @param short_term Type of short-term residual component: `"ar1"` (default,
#'   correlated), `"iid"` (independent), or `"none"`.
#' @param group_var Optional name of grouping variable for panel data.
#' @param shared Logical; if TRUE (default), temporal effects enter both
#'   numerator and denominator.
#' @param parameterization Coordinate the intrinsic trend and seasonal arms are
#'   sampled in. `"centered"` (default) samples the effects themselves.
#'   `"noncentered"` samples `z ~ N(0, I)` and reaches the effects through
#'   `sigma * A^-1 z`, where `A` stacks the arm's difference operator on
#'   `1'/sqrt(n)`, so the variance enters the linear predictor through the
#'   transform rather than through a quadratic form. The two describe the same
#'   model; what differs is the geometry the sampler sees. Prefer
#'   `"noncentered"` when an arm's variance is shrunk toward zero, where a
#'   centred coordinate leaves a funnel it cannot descend: on a 500-observation
#'   fixture whose true trend is zero, the divergence count goes 30 of 2000 to
#'   3, the bulk ESS of `sigma2_trend` 21 to 310 and its rhat 1.13 to 1.01, and
#'   the centred run reports a posterior median four times larger because it
#'   never reaches the small end. The short-term arm is proper and is the same
#'   block either way.
#'
#' @return A `ratiod_temporal_multiscale` object
#'
#' @details
#' The multi-scale temporal model decomposes variation additively:
#'
#' \deqn{\eta(t) = trend(t) + seasonal(t) + short(t)}
#'
#' where:
#' - **trend(t)**: Long-term smooth change via RW1 or RW2
#' - **seasonal(t)**: Repeating pattern with period P via cyclic RW1
#' - **short(t)**: Residual temporal correlation via AR(1) or IID
#'
#' Each component has a separate variance parameter with PC priors to
#' favor simpler models (fewer active components).
#'
#' **Sum-to-zero constraints** are applied to trend and seasonal components
#' for identifiability with the intercept.
#'
#' @examples
#' # Create multi-scale temporal specification
#' temporal_multiscale("month_id", trend = "rw2", seasonal = 12, short_term = "ar1")
#' temporal_multiscale("year", trend = "rw2", seasonal = NULL, short_term = "none")
#'
#' \dontrun{
#' # Decompose into trend + seasonal + short-term (not run - experimental)
#' set.seed(129)
#' df <- data.frame(
#'   month_id = 1:48,
#'   x = rnorm(48),
#'   count = rpois(48, lambda = 20),
#'   effort = rgamma(48, shape = 4, rate = 1)
#' )
#'
#' fit <- tratio(
#'   count | effort ~ x,
#'   data = df,
#'   family = ratiod_poisson_gamma(),
#'   temporal = temporal_multiscale(
#'     time_var = "month_id",
#'     trend = "rw2",
#'     seasonal = 12,
#'     short_term = "ar1"
#'   ),
#'   control = list(iter = 200, warmup = 100, chains = 1)
#' )
#'
#' # Extract and plot components
#' temporal_effects <- temporal(fit)
#' plot(temporal_effects, component = "trend")
#' }
#'
#' @seealso [temporal_rw1()], [temporal_rw2()], [temporal_ar1()] for
#'   single-component temporal effects, [spatial_multiscale()] for multi-scale
#'   spatial effects
#'
#' @export
temporal_multiscale <- function(time_var,
                                trend = c("rw2", "rw1", "none"),
                                seasonal = NULL,
                                short_term = c("ar1", "iid", "none"),
                                group_var = NULL,
                                shared = TRUE,
                                parameterization = c("centered", "noncentered")) {

  trend <- match.arg(trend)
  short_term <- match.arg(short_term)
  parameterization <- match.arg(parameterization)

  if (!is.character(time_var) || length(time_var) != 1) {
    stop("`time_var` must be a single character string", call. = FALSE)
  }

  if (!is.null(group_var)) {
    if (!is.character(group_var) || length(group_var) != 1) {
      stop("`group_var` must be a single character string", call. = FALSE)
    }
  }

  # Validate seasonal
  if (!is.null(seasonal)) {
    if (!is.numeric(seasonal) || length(seasonal) != 1 || seasonal < 2) {
      stop("`seasonal` must be an integer >= 2 (the period)", call. = FALSE)
    }
    seasonal <- as.integer(seasonal)
  }

  # Check at least one component is active
  if (trend == "none" && is.null(seasonal) && short_term == "none") {
    stop("At least one temporal component must be active", call. = FALSE)
  }

  # Determine which components are active
  components <- character(0)
  if (trend != "none") components <- c(components, "trend")
  if (!is.null(seasonal) && seasonal >= 2) components <- c(components, "seasonal")
  if (short_term != "none") components <- c(components, "short_term")

  structure(
    list(
      type = "multiscale",
      time_var = time_var,
      group_var = group_var,
      shared = shared,
      # Component specifications
      trend = trend,
      seasonal = seasonal,
      short_term = short_term,
      parameterization = parameterization,
      components = components,
      # Filled in during validation
      n_times = NULL,
      n_groups = NULL,
      time_index = NULL,
      group_index = NULL,
      precision_structures = NULL
    ),
    class = c("ratiod_temporal_multiscale", "ratiod_temporal", "list")
  )
}


#' Print method for ratiod_temporal_multiscale
#'
#' @param x A ratiod_temporal_multiscale object
#' @param ... Ignored
#'
#' @export
print.ratiod_temporal_multiscale <- function(x, ...) {
  cat("tulpaRatio Multi-Scale temporal specification\n")
  cat("=========================================\n\n")

  cat("Time variable:", x$time_var, "\n")
  if (!is.null(x$group_var)) {
    cat("Group variable:", x$group_var, "\n")
  }

  cat("\nComponents:\n")

  if (x$trend != "none") {
    trend_name <- switch(x$trend,
      rw1 = "RW1 (first-order random walk)",
      rw2 = "RW2 (second-order random walk, smooth)"
    )
    cat("  Trend:", trend_name, "\n")
  }

  if (!is.null(x$seasonal) && x$seasonal >= 2) {
    cat("  Seasonal: Period =", x$seasonal, "(cyclic RW1)\n")
  }

  if (x$short_term != "none") {
    short_name <- switch(x$short_term,
      ar1 = "AR(1) (autocorrelated residuals)",
      iid = "IID (independent residuals)"
    )
    cat("  Short-term:", short_name, "\n")
  }

  cat("\nShared:", if (x$shared) "Yes (enters both processes)" else "No", "\n")

  if (!is.null(x$n_times)) {
    cat("Time points:", x$n_times, "\n")
  }
  if (!is.null(x$n_groups) && x$n_groups > 1) {
    cat("Groups:", x$n_groups, "\n")
  }

  invisible(x)
}


#' Validate multi-scale temporal specification
#'
#' @param temporal ratiod_temporal_multiscale object
#' @param data Data frame
#'
#' @return Updated object with indices computed
#' @keywords internal
validate_temporal_multiscale <- function(temporal, data) {
  if (is.null(temporal)) return(NULL)
  if (!inherits(temporal, "ratiod_temporal_multiscale")) {
    # Use regular validation for single-component temporal
    return(validate_temporal(temporal, data))
  }

  # Check time variable exists
  if (!(temporal$time_var %in% names(data))) {
    stop(sprintf("Temporal variable '%s' not found in data",
                 temporal$time_var), call. = FALSE)
  }

  # Check group variable exists if specified
  if (!is.null(temporal$group_var)) {
    if (!(temporal$group_var %in% names(data))) {
      stop(sprintf("Temporal group variable '%s' not found in data",
                   temporal$group_var), call. = FALSE)
    }
  }

  # Get time values and create indices
  time_vals <- data[[temporal$time_var]]

  if (is.factor(time_vals)) {
    time_factor <- time_vals
  } else {
    unique_times <- sort(unique(time_vals))
    time_factor <- factor(time_vals, levels = unique_times)
  }

  temporal$n_times <- nlevels(time_factor)
  temporal$time_index <- as.integer(time_factor)
  temporal$time_levels <- levels(time_factor)

  # Validate minimum time points
  if (temporal$trend == "rw2" && temporal$n_times < 3) {
    stop("RW2 trend requires at least 3 time points", call. = FALSE)
  }

  if (!is.null(temporal$seasonal) && temporal$n_times < temporal$seasonal) {
    warning(sprintf(
      "Number of time points (%d) is less than seasonal period (%d).\n",
      temporal$n_times, temporal$seasonal
    ), "Seasonal component may not be well-identified.", call. = FALSE)
  }

  # Handle grouping
  if (!is.null(temporal$group_var)) {
    group_vals <- data[[temporal$group_var]]
    group_factor <- as.factor(group_vals)
    temporal$n_groups <- nlevels(group_factor)
    temporal$group_index <- as.integer(group_factor)
    temporal$group_levels <- levels(group_factor)
  } else {
    temporal$n_groups <- 1L
    temporal$group_index <- rep(1L, nrow(data))
  }

  # Build precision structures for each component
  temporal$precision_structures <- list()

  if (temporal$trend != "none") {
    temporal$precision_structures$trend <- list(
      type = temporal$trend,
      T = temporal$n_times,
      cyclic = FALSE,
      rank_deficiency = if (temporal$trend == "rw1") 1 else 2
    )
  }

  if (!is.null(temporal$seasonal) && temporal$seasonal >= 2) {
    temporal$precision_structures$seasonal <- list(
      type = "rw1",
      T = temporal$seasonal,
      cyclic = TRUE,
      rank_deficiency = 1
    )
  }

  if (temporal$short_term == "ar1") {
    temporal$precision_structures$short_term <- list(
      type = "ar1",
      T = temporal$n_times,
      rank_deficiency = 0
    )
  } else if (temporal$short_term == "iid") {
    temporal$precision_structures$short_term <- list(
      type = "iid",
      T = temporal$n_times,
      rank_deficiency = 0
    )
  }

  # Calculate total number of temporal parameters
  n_params <- 0
  if (temporal$trend != "none") {
    n_params <- n_params + temporal$n_times
  }
  if (!is.null(temporal$seasonal) && temporal$seasonal >= 2) {
    n_params <- n_params + temporal$seasonal
  }
  if (temporal$short_term != "none") {
    n_params <- n_params + temporal$n_times
  }

  temporal$n_temporal_params <- n_params * temporal$n_groups

  temporal
}


# =============================================================================
# Gaussian Process Temporal Structure
# =============================================================================

#' Gaussian Process temporal structure
#'
#' @description
#' Specify a Gaussian Process (GP) temporal random effect for irregularly-spaced
#' or continuous time points. Unlike RW1/RW2/AR1 which assume equally-spaced
#' observations, GP temporal effects model correlation as a function of time
#' distance.
#'
#' This is particularly useful for:
#' - Irregularly-spaced time series
#' - Continuous time (e.g., exact timestamps)
#' - Smooth temporal trends with uncertainty
#'
#' @param time_var Name of the time variable in data. Can be a formula
#'   (e.g., `~ year`) or a character string (e.g., `"year"`). Should be
#'   numeric (continuous time) or convertible to numeric.
#' @param cov Covariance function: `"exponential"` (default, rough),
#'   `"matern"` (tunable smoothness), `"gaussian"` (very smooth), or
#'   `"periodic"` (for seasonal patterns).
#' @param nu Smoothness parameter for Matern covariance. Common values:
#'   - 0.5: Equivalent to exponential (rough)
#'   - 1.5: Once differentiable (moderate smoothness)
#'   - 2.5: Twice differentiable (smooth)
#'   Ignored for non-Matern covariance functions.
#' @param period Period for periodic covariance (e.g., 12 for monthly, 365 for
#'   daily data with annual cycle). Only used when `cov = "periodic"`.
#' @param group_var Optional name of grouping variable for panel data.
#'   If provided, separate GPs are estimated for each group.
#' @param shared Logical; if TRUE (default), temporal effect enters both
#'   numerator and denominator.
#' @param scale_coords Logical; if TRUE (default), time values are scaled to
#'   unit variance before computing distances.
#' @param parameterization Parameterization for GP effects:
#'   `"noncentered"` (default) stores z ~ N(0,1) and scales by covariance
#'   (better for weakly-informed effects);
#'   `"centered"` stores effects directly (better for strongly-informed effects).
#'
#' @return A `ratiod_temporal_gp` object
#'
#' @details
#' The GP temporal model adds a time-correlated random effect:
#'
#' \deqn{\eta(t) = X\beta + f(t)}
#'
#' where \eqn{f(t)} follows a Gaussian process:
#' \deqn{f(t) \sim GP(0, \sigma^2 C(|t - t'|; \phi))}
#'
#' The correlation function \eqn{C(d; \phi)} depends on time distance \eqn{d}:
#'
#' - **Exponential**: \eqn{C(d) = \exp(-d/\phi)} - continuous but not differentiable
#' - **Matern**: Smooth with tunable roughness via \eqn{\nu}
#' - **Gaussian**: \eqn{C(d) = \exp(-(d/\phi)^2)} - infinitely differentiable
#' - **Periodic**: \eqn{C(d) = \exp(-2\sin^2(\pi d/p)/\phi^2)} - for seasonal data
#'
#' **Implementation**: Uses a state-space representation for O(n) computational
#' complexity when possible (exponential, Matern with half-integer nu).
#'
#' @examples
#' # Create GP temporal specification
#' temporal_gp("timestamp")
#' temporal_gp("day", cov = "matern", nu = 1.5)
#' temporal_gp("month", cov = "periodic", period = 12)
#'
#' \dontrun{
#' # Irregularly-spaced time series (not run - GP temporal experimental)
#' set.seed(140)
#' times <- sort(runif(30, 0, 100))
#' df <- data.frame(
#'   time = times,
#'   x = rnorm(30),
#'   count = rpois(30, lambda = 20),
#'   effort = rgamma(30, shape = 4, rate = 1)
#' )
#'
#' fit <- tratio(
#'   count | effort ~ x,
#'   data = df,
#'   family = ratiod_poisson_gamma(),
#'   temporal = temporal_gp("time"),
#'   control = list(iter = 200, warmup = 100, chains = 1)
#' )
#' }
#'
#' @seealso [temporal_rw1()], [temporal_ar1()] for equally-spaced temporal effects,
#'   [spatial_gp()] for spatial GP effects
#'
#' @export
temporal_gp <- function(time_var,
                        cov = c("exponential", "matern", "gaussian", "periodic"),
                        nu = 1.5,
                        period = NULL,
                        group_var = NULL,
                        shared = TRUE,
                        scale_coords = TRUE,
                        parameterization = c("noncentered", "centered")) {

  cov <- match.arg(cov)
  parameterization <- match.arg(parameterization)

  if (!is.character(time_var) || length(time_var) != 1) {
    stop("`time_var` must be a single character string", call. = FALSE)
  }

  if (!is.null(group_var)) {
    if (!is.character(group_var) || length(group_var) != 1) {
      stop("`group_var` must be a single character string", call. = FALSE)
    }
  }

  # Validate nu for Matern
  if (cov == "matern") {
    if (!is.numeric(nu) || length(nu) != 1 || nu <= 0) {
      stop("`nu` must be a positive number for Matern covariance", call. = FALSE)
    }
  }

  # Validate period for periodic covariance
  if (cov == "periodic") {
    if (is.null(period) || !is.numeric(period) || length(period) != 1 || period <= 0) {
      stop("`period` must be a positive number when using periodic covariance",
           call. = FALSE)
    }
  }

  # Warning for non-shared temporal effects
  if (!shared) {
    warning(
      "Non-shared temporal GP effects (shared = FALSE) may lead to confounded ratio estimates.\n",
      "Consider whether temporal effects should be shared between\n",
      "numerator and denominator to prevent spurious temporal patterns in ratios.",
      call. = FALSE
    )
  }

  structure(
    list(
      type = "gp",
      time_var = time_var,
      group_var = group_var,
      cov = cov,
      nu = if (cov == "matern") nu else NULL,
      period = if (cov == "periodic") period else NULL,
      shared = shared,
      scale_coords = scale_coords,
      parameterization = parameterization,
      cyclic = FALSE,
      # Filled in during validation
      n_times = NULL,
      n_groups = NULL,
      n_obs = NULL,
      time_values = NULL,
      time_index = NULL,
      group_index = NULL,
      time_distances = NULL
    ),
    class = c("ratiod_temporal_gp", "ratiod_temporal", "list")
  )
}


#' Print method for ratiod_temporal_gp
#'
#' @param x A ratiod_temporal_gp object
#' @param ... Ignored
#'
#' @export
print.ratiod_temporal_gp <- function(x, ...) {
  cat("tulpaRatio Gaussian Process temporal specification\n")
  cat("==============================================\n\n")

  cat("Time variable:", x$time_var, "\n")
  if (!is.null(x$group_var)) {
    cat("Group variable:", x$group_var, "\n")
  }

  cov_str <- x$cov
  if (x$cov == "matern" && !is.null(x$nu)) {
    cov_str <- sprintf("%s (nu = %.1f)", x$cov, x$nu)
  } else if (x$cov == "periodic" && !is.null(x$period)) {
    cov_str <- sprintf("%s (period = %.1f)", x$cov, x$period)
  }
  cat("Covariance:", cov_str, "\n")
  cat("Shared:", if (x$shared) "Yes (enters both processes)" else "No", "\n")

  if (!is.null(x$n_obs)) {
    cat("\nObservations:", x$n_obs, "\n")
    cat("Unique time points:", x$n_times, "\n")
  }

  invisible(x)
}


#' Validate temporal GP specification against data
#'
#' @param temporal ratiod_temporal_gp object
#' @param data Data frame
#'
#' @return Updated object with computed time structure
#' @keywords internal
validate_temporal_gp <- function(temporal, data) {
  if (is.null(temporal)) return(NULL)
  if (!inherits(temporal, "ratiod_temporal_gp")) return(temporal)

  N <- nrow(data)

  # Check time variable exists
  if (!(temporal$time_var %in% names(data))) {
    stop(sprintf("Time variable '%s' not found in data", temporal$time_var),
         call. = FALSE)
  }

  # Extract time values
  time_vals <- data[[temporal$time_var]]

  # Convert to numeric
  if (!is.numeric(time_vals)) {
    if (inherits(time_vals, c("Date", "POSIXt"))) {
      time_vals <- as.numeric(time_vals)
    } else {
      stop("Time variable must be numeric or a date/time type", call. = FALSE)
    }
  }

  # Check for missing values
  if (any(is.na(time_vals))) {
    stop("Time variable contains missing values", call. = FALSE)
  }

  # Scale time values if requested
  if (temporal$scale_coords) {
    time_vals <- as.vector(scale(time_vals))
  }

  temporal$n_obs <- N

  # Extract unique time values (sorted) and create time index
  # GP uses one effect per unique time, not per observation
  unique_times <- sort(unique(time_vals))
  temporal$unique_time_values <- unique_times  # Unique times for GP cov matrix
  temporal$n_times <- length(unique_times)

  # Create time_index: maps each obs to its unique time point (1-based)
  temporal$time_index <- match(time_vals, unique_times)

  # Keep original time_values for backward compatibility
  temporal$time_values <- unique_times  # Now unique times, not per-obs

  # Handle grouping
  if (!is.null(temporal$group_var)) {
    if (!(temporal$group_var %in% names(data))) {
      stop(sprintf("Group variable '%s' not found in data", temporal$group_var),
           call. = FALSE)
    }
    group_vals <- data[[temporal$group_var]]
    group_factor <- as.factor(group_vals)
    temporal$n_groups <- nlevels(group_factor)
    temporal$group_index <- as.integer(group_factor)
    temporal$group_levels <- levels(group_factor)
  } else {
    temporal$n_groups <- 1L
    temporal$group_index <- rep(1L, N)
  }

  # Compute time distances for GP on unique times (not N x N)
  n_unique <- temporal$n_times
  if (n_unique <= 500) {
    temporal$time_distances <- outer(unique_times, unique_times,
                                     FUN = function(x, y) abs(x - y))
  } else {
    # For large data, we'll compute distances on-the-fly in C++
    temporal$time_distances <- NULL
  }

  # Total temporal parameters = n_times (one effect per unique time), not N
  temporal$n_temporal_params <- temporal$n_times * temporal$n_groups

  temporal
}


# =============================================================================
# Temporally-Varying Coefficients (TVC)
# =============================================================================

#' Temporally-Varying Coefficients (TVC)
#'
#' @description
#' Specify temporally-varying coefficients for ratio models. TVCs allow
#' regression coefficients to change smoothly over time using a Gaussian
#' process or random walk prior. This captures how effects evolve temporally.
#'
#' @param time_var Name of the time variable in data. Can be a formula
#'   (e.g., `~ year`) or a character string (e.g., `"year"`).
#' @param terms Which coefficients should vary over time. Options:
#'   - Integer vector: Column indices of design matrix (1 = intercept)
#'   - Character vector: Coefficient names (e.g., `"(Intercept)"`, `"depth"`)
#'   - Formula: `~ 1 + depth` for intercept and depth
#' @param structure Temporal structure for the varying coefficients:
#'   - `"rw1"`: First-order random walk (default)
#'   - `"rw2"`: Second-order random walk (smoother)
#'   - `"ar1"`: First-order autoregressive
#'   - `"iid"`: Independent draws at each time point (no temporal smoothing)
#'   - `"gp"`: Gaussian process (for irregular time spacing). Not yet
#'     implemented; `temporal_tvc()` errors if selected.
#' @param group_var Optional name of grouping variable for panel data.
#' @param shared Logical; if TRUE (default), TVC effects enter both
#'   numerator and denominator. Set to FALSE for process-specific TVCs
#'   (triggers warning about potential confounding).
#'
#' @return A `ratiod_tvc` object
#'
#' @details
#' The TVC model extends the linear predictor:
#'
#' \deqn{\eta(t) = X\beta + \tilde{X}(t)w(t)}
#'
#' where:
#' - \eqn{\beta} are global (non-time-varying) coefficients
#' - \eqn{\tilde{X}(t)} is the subset of covariates with TVCs
#' - \eqn{w(t)} are time-varying adjustments
#'
#' Each TVC follows a temporal process:
#' \deqn{w_j(t) \sim RW1(\sigma_j^2) \text{ or other temporal structure}}
#'
#' **Interpretation**: A positive TVC for depth at time t means the depth
#' effect is stronger at t than the global average. The temporal variance
#' \eqn{\sigma^2_j} quantifies how much the effect varies over time.
#'
#' **Identifiability**: TVCs are centered (sum-to-zero constraint) to separate
#' the time-varying component from the global effect.
#'
#' @examples
#' # Create TVC specification
#' tvc <- temporal_tvc("year", terms = 1)
#' print(tvc)
#'
#' \dontrun{
#' # Generate synthetic data with time-varying effect (not run - TVC experimental)
#' set.seed(150)
#' df <- data.frame(
#'   year = rep(2010:2025, each = 5),
#'   x = rnorm(80),
#'   count = rpois(80, lambda = 20),
#'   effort = rgamma(80, shape = 4, rate = 1)
#' )
#'
#' # Time-varying intercept (random temporal field)
#' fit <- tratio(
#'   count | effort ~ x,
#'   data = df,
#'   family = ratiod_poisson_gamma(),
#'   temporal = temporal_tvc("year", terms = 1),
#'   mode = "hmc",
#'   control = list(iter = 200, warmup = 100, chains = 1)
#' )
#' summary(fit)
#'
#' # Time-varying effect of x
#' fit2 <- tratio(
#'   count | effort ~ x,
#'   data = df,
#'   family = ratiod_poisson_gamma(),
#'   temporal = temporal_tvc("year", terms = c("(Intercept)", "x"), structure = "rw2"),
#'   mode = "hmc",
#'   control = list(iter = 200, warmup = 100, chains = 1)
#' )
#'
#' # Extract and plot the time-varying coefficients
#' tvc_effects <- tvc(fit2)
#' plot(tvc_effects, "x")
#' }
#'
#' @seealso [spatial_svc()] for spatially-varying coefficients,
#'   [temporal_rw1()] for temporal random effects,
#'   [tvc()] for extracting TVC posteriors
#'
#' @export
temporal_tvc <- function(time_var,
                         terms = 1,
                         structure = c("rw1", "rw2", "ar1", "iid", "gp"),
                         group_var = NULL,
                         shared = TRUE) {

  structure_type <- match.arg(structure)

  if (structure_type == "gp") {
    stop("`structure = \"gp\"` is not implemented for temporal_tvc(). ",
         "Use \"rw1\", \"rw2\", \"ar1\", or \"iid\".", call. = FALSE)
  }

  if (!is.character(time_var) || length(time_var) != 1) {
    stop("`time_var` must be a single character string", call. = FALSE)
  }

  if (!is.null(group_var)) {
    if (!is.character(group_var) || length(group_var) != 1) {
      stop("`group_var` must be a single character string", call. = FALSE)
    }
  }

  # Parse terms specification
  if (inherits(terms, "formula")) {
    terms_spec <- list(type = "formula", formula = terms)
  } else if (is.numeric(terms)) {
    terms_spec <- list(type = "index", indices = as.integer(terms))
  } else if (is.character(terms)) {
    terms_spec <- list(type = "names", names = terms)
  } else {
    stop("`terms` must be a formula, integer vector, or character vector",
         call. = FALSE)
  }

  # Warning for non-shared TVCs
  if (!shared) {
    warning(
      "Non-shared TVCs (shared = FALSE) may lead to confounded ratio estimates.\n",
      "Consider whether time-varying effects should be shared between\n",
      "numerator and denominator to prevent spurious temporal patterns in ratios.",
      call. = FALSE
    )
  }

  structure(
    list(
      type = "tvc",
      time_var = time_var,
      group_var = group_var,
      terms_spec = terms_spec,
      structure = structure_type,
      shared = shared,
      # Filled in during validation
      n_times = NULL,
      n_groups = NULL,
      n_tvc = NULL,
      tvc_indices = NULL,
      tvc_names = NULL,
      time_index = NULL,
      group_index = NULL
    ),
    class = c("ratiod_tvc", "ratiod_temporal", "list")
  )
}


#' Print method for ratiod_tvc
#'
#' @param x A ratiod_tvc object
#' @param ... Ignored
#'
#' @export
print.ratiod_tvc <- function(x, ...) {
  cat("tulpaRatio temporally-varying coefficients\n")
  cat("======================================\n\n")

  cat("Time variable:", x$time_var, "\n")
  if (!is.null(x$group_var)) {
    cat("Group variable:", x$group_var, "\n")
  }

  struct_name <- switch(x$structure,
    rw1 = "RW1 (first-order random walk)",
    rw2 = "RW2 (second-order random walk)",
    ar1 = "AR(1) (autoregressive)",
    iid = "IID (independent at each time point)",
    gp = "GP (Gaussian process)"
  )
  cat("Structure:", struct_name, "\n")
  cat("Shared:", if (x$shared) "Yes (enters both processes)" else "No", "\n")

  if (!is.null(x$n_tvc)) {
    cat("\nTVC terms:", x$n_tvc, "\n")
    if (!is.null(x$tvc_names)) {
      cat("  ", paste(x$tvc_names, collapse = ", "), "\n")
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

  if (!is.null(x$n_times)) {
    cat("Time points:", x$n_times, "\n")
  }

  invisible(x)
}


#' Validate TVC specification against data and design matrix
#'
#' @param tvc ratiod_tvc object
#' @param data Data frame
#' @param X Design matrix (to resolve term names)
#'
#' @return Updated ratiod_tvc object with computed structure
#' @keywords internal
validate_tvc <- function(tvc, data, X) {
  if (is.null(tvc)) return(NULL)
  if (!inherits(tvc, "ratiod_tvc")) return(tvc)

  N <- nrow(data)
  p <- ncol(X)

  # Check time variable exists
  if (!(tvc$time_var %in% names(data))) {
    stop(sprintf("Time variable '%s' not found in data", tvc$time_var),
         call. = FALSE)
  }

  # Get time values and create indices
  time_vals <- data[[tvc$time_var]]
  if (is.factor(time_vals)) {
    time_factor <- time_vals
  } else {
    unique_times <- sort(unique(time_vals))
    time_factor <- factor(time_vals, levels = unique_times)
  }

  tvc$n_times <- nlevels(time_factor)
  tvc$time_index <- as.integer(time_factor)
  tvc$time_levels <- levels(time_factor)

  # Handle grouping
  if (!is.null(tvc$group_var)) {
    if (!(tvc$group_var %in% names(data))) {
      stop(sprintf("Group variable '%s' not found in data", tvc$group_var),
           call. = FALSE)
    }
    group_vals <- data[[tvc$group_var]]
    group_factor <- as.factor(group_vals)
    tvc$n_groups <- nlevels(group_factor)
    tvc$group_index <- as.integer(group_factor)
    tvc$group_levels <- levels(group_factor)
  } else {
    tvc$n_groups <- 1L
    tvc$group_index <- rep(1L, N)
  }

  # Resolve TVC terms against design matrix
  coef_names <- colnames(X)
  if (is.null(coef_names)) {
    coef_names <- paste0("V", seq_len(p))
  }

  if (tvc$terms_spec$type == "index") {
    tvc_indices <- tvc$terms_spec$indices
    if (any(tvc_indices < 1 | tvc_indices > p)) {
      stop(sprintf("TVC term indices must be between 1 and %d", p),
           call. = FALSE)
    }
    tvc_names <- coef_names[tvc_indices]

  } else if (tvc$terms_spec$type == "names") {
    tvc_names <- tvc$terms_spec$names
    tvc_indices <- match(tvc_names, coef_names)
    if (any(is.na(tvc_indices))) {
      missing <- tvc_names[is.na(tvc_indices)]
      stop(sprintf("TVC terms not found in design matrix: %s",
                   paste(missing, collapse = ", ")), call. = FALSE)
    }

  } else if (tvc$terms_spec$type == "formula") {
    fmla <- tvc$terms_spec$formula
    tt <- terms(fmla)
    term_labels <- attr(tt, "term.labels")
    has_intercept <- attr(tt, "intercept") == 1

    tvc_names <- character(0)
    if (has_intercept) {
      tvc_names <- c(tvc_names, "(Intercept)")
    }
    tvc_names <- c(tvc_names, term_labels)

    tvc_indices <- match(tvc_names, coef_names)
    if (any(is.na(tvc_indices))) {
      missing <- tvc_names[is.na(tvc_indices)]
      stop(sprintf("TVC terms not found in design matrix: %s",
                   paste(missing, collapse = ", ")), call. = FALSE)
    }
  }

  tvc$n_tvc <- length(tvc_indices)
  tvc$tvc_indices <- tvc_indices
  tvc$tvc_names <- tvc_names

  # Store design matrix subset for TVC terms
  tvc$X_tvc <- X[, tvc_indices, drop = FALSE]

  # Total TVC parameters = n_times * n_tvc * n_groups
  tvc$n_temporal_params <- tvc$n_times * tvc$n_tvc * tvc$n_groups

  tvc
}


#' Extract temporally-varying coefficients from a fitted model
#'
#' @description
#' Extract posterior distributions of temporally-varying coefficients (TVCs)
#' from a fitted ratio model with TVC specification.
#'
#' @param object A `ratiod_fit` object fitted with `tvc` argument
#' @param terms Which TVC terms to extract. If NULL (default), extracts all.
#' @param group Which level of the temporal grouping variable to report, as an
#'   index into its levels; a model without temporal groups has the one level 1.
#' @param summary Logical; if TRUE, return summary statistics instead of
#'   full posterior draws.
#' @param probs Quantiles to compute if `summary = TRUE`.
#' @param ... Ignored
#'
#' @return A `ratiod_tvc_posterior` object containing:
#' - `draws`: Array of posterior draws (draws x times x terms)
#' - `time_levels`: Time point labels
#' - `term_names`: Names of TVC terms
#'
#' @examples
#' \dontrun{
#' # Generate synthetic data (not run - TVC experimental)
#' set.seed(160)
#' df <- data.frame(
#'   year = rep(2015:2024, each = 5),
#'   x = rnorm(50),
#'   count = rpois(50, lambda = 18),
#'   effort = rgamma(50, shape = 3.5, rate = 1)
#' )
#'
#' # Fit model with TVC
#' fit <- tratio(
#'   count | effort ~ x,
#'   data = df,
#'   family = ratiod_poisson_gamma(),
#'   temporal = temporal_tvc("year", terms = c(1, 2)),
#'   mode = "hmc",
#'   control = list(iter = 200, warmup = 100, chains = 1)
#' )
#'
#' # Extract TVC posteriors
#' tvc_post <- tvc(fit)
#' summary(tvc_post)
#'
#' # Plot temporal evolution
#' plot(tvc_post, "x")
#' }
#'
#' @seealso [temporal_tvc()], [plot.ratiod_tvc_posterior()]
#'
#' @export
tvc <- function(object, terms = NULL, group = 1L, summary = FALSE,
                probs = c(0.025, 0.5, 0.975), ...) {
  UseMethod("tvc")
}


#' @rdname tvc
#' @export
tvc.ratiod_fit <- function(object, terms = NULL, group = 1L, summary = FALSE,
                           probs = c(0.025, 0.5, 0.975), ...) {

  # Check if model has TVCs
  if (is.null(object$tvc) || !inherits(object$tvc, "ratiod_tvc")) {
    stop("Model was not fitted with temporally-varying coefficients.\n",
         "Pass `temporal = temporal_tvc(...)` to tratio() to specify TVCs.",
         call. = FALSE)
  }

  tvc_info <- object$tvc
  n_times <- tvc_info$n_times
  n_tvc <- tvc_info$n_tvc
  tvc_names <- tvc_info$tvc_names

  # Get TVC draws from model
  tvc_draws <- object$.internal$tvc_draws

  if (is.null(tvc_draws)) {
    stop("TVC draws not found in model output", call. = FALSE)
  }

  # Draws are [draw, time, term, group]; report one grouping level at a
  # time. A three-dimensional array is the single-group case already.
  n_groups <- if (length(dim(tvc_draws)) >= 4L) dim(tvc_draws)[4] else 1L
  if (length(group) != 1L || is.na(group) || group < 1L || group > n_groups) {
    stop("`group` must be one of 1:", n_groups, ".", call. = FALSE)
  }
  if (length(dim(tvc_draws)) >= 4L) {
    tvc_draws <- tvc_draws[, , , group, drop = FALSE]
    dim(tvc_draws) <- dim(tvc_draws)[1:3]
  }

  # Subset terms if requested
  if (!is.null(terms)) {
    if (is.numeric(terms)) {
      term_idx <- terms
    } else if (is.character(terms)) {
      term_idx <- match(terms, tvc_names)
      if (any(is.na(term_idx))) {
        stop("Terms not found: ", paste(terms[is.na(term_idx)], collapse = ", "),
             call. = FALSE)
      }
    } else {
      stop("`terms` must be numeric or character", call. = FALSE)
    }
    tvc_draws <- tvc_draws[, , term_idx, drop = FALSE]
    tvc_names <- tvc_names[term_idx]
    n_tvc <- length(term_idx)
  }

  result <- structure(
    list(
      draws = tvc_draws,
      time_levels = tvc_info$time_levels,
      term_names = tvc_names,
      n_times = n_times,
      n_tvc = n_tvc,
      n_draws = dim(tvc_draws)[1],
      structure = tvc_info$structure
    ),
    class = "ratiod_tvc_posterior"
  )

  if (summary) {
    return(summary(result, probs = probs))
  }

  result
}


#' Print method for ratiod_tvc_posterior
#'
#' @param x A ratiod_tvc_posterior object
#' @param ... Ignored
#'
#' @export
print.ratiod_tvc_posterior <- function(x, ...) {
  cat("Temporally-varying coefficient posterior\n")
  cat("========================================\n\n")
  cat("Terms:", paste(x$term_names, collapse = ", "), "\n")
  cat("Time points:", x$n_times, "\n")
  cat("Posterior draws:", x$n_draws, "\n")
  cat("Structure:", x$structure, "\n")
  cat("\nUse summary() for posterior summaries\n")
  cat("Use plot() for temporal visualization\n")
  invisible(x)
}


#' Summary method for ratiod_tvc_posterior
#'
#' @param object A ratiod_tvc_posterior object
#' @param probs Quantiles to compute
#' @param ... Ignored
#'
#' @export
summary.ratiod_tvc_posterior <- function(object, probs = c(0.025, 0.5, 0.975), ...) {

  n_times <- object$n_times
  n_tvc <- object$n_tvc
  draws <- object$draws

  results <- list()

  for (j in seq_len(n_tvc)) {
    term_draws <- draws[, , j]

    summaries <- data.frame(
      time_idx = seq_len(n_times),
      time = object$time_levels,
      term = object$term_names[j],
      mean = colMeans(term_draws),
      sd = apply(term_draws, 2, sd),
      t(apply(term_draws, 2, quantile, probs = probs))
    )
    names(summaries)[6:ncol(summaries)] <- paste0("q", probs * 100)
    rownames(summaries) <- NULL

    results[[j]] <- summaries
  }

  result <- do.call(rbind, results)

  structure(
    result,
    n_draws = object$n_draws,
    term_names = object$term_names,
    class = c("ratiod_tvc_summary", "data.frame")
  )
}


#' Plot method for ratiod_tvc_posterior
#'
#' @param x A ratiod_tvc_posterior object
#' @param term Which term to plot (name or index). Default: first term.
#' @param type Plot type: "ribbon" (default) or "line"
#' @param ... Additional arguments passed to plotting functions
#'
#' @export
plot.ratiod_tvc_posterior <- function(x, term = 1, type = "ribbon", ...) {

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

  n_times <- x$n_times
  times <- as.numeric(x$time_levels)

  # Compute summaries
  means <- colMeans(draws)
  lower <- apply(draws, 2, quantile, probs = 0.025)
  upper <- apply(draws, 2, quantile, probs = 0.975)

  title <- paste("TVC:", term_name)

  # Use ggplot2 if available
  if (requireNamespace("ggplot2", quietly = TRUE)) {
    df <- data.frame(
      time = times,
      mean = means,
      lower = lower,
      upper = upper
    )

    p <- ggplot2::ggplot(df, ggplot2::aes(x = .data$time, y = .data$mean)) +
      ggplot2::geom_ribbon(
        ggplot2::aes(ymin = .data$lower, ymax = .data$upper),
        alpha = 0.3, fill = "steelblue"
      ) +
      ggplot2::geom_line(color = "steelblue", linewidth = 1) +
      ggplot2::geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
      ggplot2::labs(
        title = title,
        x = "Time",
        y = "Time-Varying Effect"
      ) +
      theme_ratiod()

    return(p)
  }

  # Base R fallback
  ylim <- range(c(lower, upper))

  plot(times, means, type = "l", col = "steelblue", lwd = 2,
       ylim = ylim, xlab = "Time", ylab = "Time-Varying Effect",
       main = title, ...)

  polygon(c(times, rev(times)), c(lower, rev(upper)),
          col = adjustcolor("steelblue", alpha.f = 0.3), border = NA)

  abline(h = 0, lty = 2, col = "gray50")
  lines(times, means, col = "steelblue", lwd = 2)

  invisible(NULL)
}


# =============================================================================
# Restricted Temporal Regression (RTR)
# =============================================================================

#' Restricted Temporal Regression (RTR)
#'
#' @description
#' Apply Restricted Temporal Regression to mitigate temporal confounding.
#' RTR orthogonalizes the temporal effect to the covariate space, preventing
#' the temporal random effect from absorbing covariate information.
#'
#' This is important when covariates have temporal trends (e.g., increasing
#' over time) because the temporal random effect can "steal" variance from
#' these covariates, leading to biased coefficient estimates.
#'
#' @param temporal A temporal specification (`temporal_rw1`, `temporal_gp`, etc.)
#' @param restrict_to Formula specifying which covariates to orthogonalize
#'   against (e.g., `~ year_effect + trend`). The temporal effect will be
#'   constrained to be orthogonal to the column space of these covariates.
#'
#' @return A modified temporal specification with RTR enabled
#'
#' @details
#' The RTR approach (analogous to Restricted Spatial Regression) modifies the
#' temporal random effect to be orthogonal to the fixed effect design matrix:
#'
#' \deqn{f_{RTR}(t) = (I - P_X) f(t)}
#'
#' where \eqn{P_X = X(X'X)^{-1}X'} is the projection matrix onto the column
#' space of X.
#'
#' **When to use RTR:**
#' - Covariates have temporal trends (e.g., increasing/decreasing over time)
#' - Interested in causal interpretation of covariate effects
#' - Coefficients appear attenuated toward zero
#' - Time and covariates are correlated
#'
#' **When NOT to use RTR:**
#' - Covariates are temporally uncorrelated (random sampling over time)
#' - Temporal effect is the primary quantity of interest
#' - Prediction is the main goal (not causal inference)
#'
#' @examples
#' # Create RTR temporal structure
#' rtr <- temporal_rtr(
#'   temporal_rw2("year"),
#'   restrict_to = ~ temperature + precipitation
#' )
#' print(rtr)
#'
#' \dontrun{
#' # Generate data with temporal confounding (not run - RTR experimental)
#' set.seed(170)
#' years <- 2000:2023
#' n_per_year <- 4
#' df <- data.frame(
#'   year = rep(years, each = n_per_year),
#'   # Temperature increasing over time (confounded with year)
#'   temperature = rep(seq(15, 18, length.out = length(years)), each = n_per_year) +
#'                 rnorm(length(years) * n_per_year, 0, 0.5),
#'   count = rpois(length(years) * n_per_year, lambda = 20),
#'   effort = rgamma(length(years) * n_per_year, shape = 4, rate = 1)
#' )
#'
#' # Standard RW2 (may have temporal confounding)
#' fit1 <- tratio(
#'   count | effort ~ temperature,
#'   data = df,
#'   family = ratiod_poisson_gamma(),
#'   temporal = temporal_rw2("year"),
#'   mode = "hmc",
#'   control = list(iter = 200, warmup = 100, chains = 1)
#' )
#'
#' # RTR to protect temperature coefficient
#' fit2 <- tratio(
#'   count | effort ~ temperature,
#'   data = df,
#'   family = ratiod_poisson_gamma(),
#'   temporal = temporal_rtr(
#'     temporal_rw2("year"),
#'     restrict_to = ~ temperature
#'   ),
#'   mode = "hmc",
#'   control = list(iter = 200, warmup = 100, chains = 1)
#' )
#'
#' # Compare coefficient estimates
#' summary(fit1)  # May be attenuated
#' summary(fit2)  # Protected from temporal confounding
#' }
#'
#' @references
#' Hanks, E. M., Schliep, E. M., Hooten, M. B., & Hoeting, J. A. (2015).
#' Restricted spatial regression in practice: geostatistical models, confounding,
#' and robustness under model misspecification. Environmetrics, 26(4), 243-254.
#'
#' @seealso [temporal_rw1()], [temporal_rw2()], [temporal_gp()],
#'   [spatial_rsr()] for the spatial analog
#'
#' @export
temporal_rtr <- function(temporal, restrict_to) {

  if (!inherits(temporal, "ratiod_temporal")) {
    stop("`temporal` must be a tulpaRatio temporal specification", call. = FALSE)
  }

  if (!inherits(restrict_to, "formula")) {
    stop("`restrict_to` must be a formula", call. = FALSE)
  }

  # Store RTR information in the temporal object
  temporal$rtr <- TRUE
  temporal$rtr_formula <- restrict_to

  # Add RTR class for dispatch
  class(temporal) <- c("ratiod_rtr", class(temporal))

  temporal
}


#' Print method for ratiod_rtr (temporal)
#'
#' @param x A ratiod_rtr object
#' @param ... Passed to underlying print method
#'
#' @export
print.ratiod_rtr <- function(x, ...) {
  # Print underlying temporal type
  NextMethod()

  cat("\nRestricted Temporal Regression (RTR):\n")
  cat("  Orthogonal to:", deparse(x$rtr_formula), "\n")
  cat("  (Temporal effect constrained to be orthogonal to covariate space)\n")

  invisible(x)
}


#' Validate RTR specification
#'
#' @param temporal ratiod_rtr object
#' @param data Data frame
#' @param formula Model formula (to extract design matrix)
#'
#' @return Updated temporal object with projection matrix
#' @keywords internal
validate_rtr <- function(temporal, data, formula) {
  if (is.null(temporal) || !inherits(temporal, "ratiod_rtr")) {
    return(temporal)
  }

  # Build design matrix for RTR covariates
  rtr_formula <- temporal$rtr_formula

  # Check if terms exist in data
  rtr_vars <- all.vars(rtr_formula)
  missing_vars <- setdiff(rtr_vars, names(data))
  if (length(missing_vars) > 0) {
    stop(sprintf("RTR variables not found in data: %s",
                 paste(missing_vars, collapse = ", ")), call. = FALSE)
  }

  # Build design matrix
  X_rtr <- model.matrix(rtr_formula, data = data)

  # Compute projection matrix (using QR for numerical stability)
  temporal$rtr_projection <- compute_rtr_projection(X_rtr)
  temporal$rtr_vars <- rtr_vars

  temporal
}


#' Compute RTR projection matrix
#'
#' @description
#' Compute the orthogonal projection matrix P_perp = I - P_X that projects
#' the temporal effect into the space orthogonal to the covariates.
#'
#' @param X Design matrix of covariates to orthogonalize against
#'
#' @return Projection matrix (n x n)
#' @keywords internal
compute_rtr_projection <- function(X) {
  n <- nrow(X)
  p <- ncol(X)

  if (p >= n) {
    warning("More covariates than observations; RTR may not be effective",
            call. = FALSE)
  }

  # QR decomposition is more numerically stable than direct inverse
  qr_X <- qr(X)

  # P_X = Q %*% Q' where Q is orthonormal basis for col(X)
  Q <- qr.Q(qr_X)

  # P_perp = I - Q %*% Q'
  P_perp <- diag(n) - Q %*% t(Q)

  P_perp
}


#' Apply RTR projection to temporal effect
#'
#' @description
#' Project temporal effect into the space orthogonal to covariates.
#' Called during posterior computation.
#'
#' @param f Temporal effect vector (length n)
#' @param P_perp Projection matrix from compute_rtr_projection
#'
#' @return Projected temporal effect (length n)
#' @keywords internal
apply_rtr_projection <- function(f, P_perp) {
  as.vector(P_perp %*% f)
}


#' Extract temporal effects from a fitted model
#'
#' @description
#' Extract posterior distributions of temporal effects from a fitted tulpaRatio
#' model with temporal specification.
#'
#' @param object A `ratiod_fit` object fitted with `temporal` argument
#' @param component Which component to extract for multi-scale models:
#'   `"all"` (default), `"trend"`, `"seasonal"`, or `"short_term"`.
#' @param summary Logical; if TRUE, return summary statistics instead of
#'   full posterior draws.
#' @param probs Quantiles to compute if `summary = TRUE`.
#' @param ... Ignored
#'
#' @return A `ratiod_temporal_posterior` object
#'
#' @examples
#' \dontrun{
#' # Fit model with multi-scale temporal effects (not run - experimental)
#' set.seed(131)
#' df <- data.frame(
#'   year = 1:40,
#'   x = rnorm(40),
#'   count = rpois(40, lambda = 22),
#'   effort = rgamma(40, shape = 4.5, rate = 1)
#' )
#'
#' fit <- tratio(
#'   count | effort ~ x,
#'   data = df,
#'   family = ratiod_poisson_gamma(),
#'   temporal = temporal_multiscale("year", trend = "rw2", seasonal = 12),
#'   control = list(iter = 200, warmup = 100, chains = 1)
#' )
#'
#' # Extract all temporal effects
#' temp_post <- temporal(fit)
#' summary(temp_post)
#' }
#'
#' @seealso [temporal_multiscale()], [temporal_rw1()]
#'
#' @export
temporal <- function(object, component = "all", summary = FALSE,
                     probs = c(0.025, 0.5, 0.975), ...) {
  UseMethod("temporal")
}


#' @rdname temporal
#' @export
temporal.ratiod_fit <- function(object, component = "all", summary = FALSE,
                                probs = c(0.025, 0.5, 0.975), ...) {

  # Check if model has temporal effects
  if (is.null(object$temporal)) {
    stop("Model was not fitted with temporal effects.\n",
         "Use `temporal` argument in tratio() to specify temporal structure.",
         call. = FALSE)
  }

  temp_info <- object$temporal

  # Get temporal draws from model
  temp_draws <- object$.internal$temporal_draws

  if (is.null(temp_draws)) {
    stop("Temporal draws not found in model output", call. = FALSE)
  }

  # Handle multi-scale vs single-component
  if (identical(temp_info$type, "multiscale")) {
    available_components <- temp_info$components

    if (component != "all" && !(component %in% available_components)) {
      stop(sprintf("Component '%s' not in model. Available: %s",
                   component, paste(available_components, collapse = ", ")),
           call. = FALSE)
    }

    if (component != "all") {
      # Subset to requested component
      temp_draws <- temp_draws[[component]]
    }
  }

  result <- structure(
    list(
      draws = temp_draws,
      time_levels = temp_info$time_levels,
      n_times = temp_info$n_times,
      n_groups = temp_info$n_groups,
      n_draws = if (is.list(temp_draws)) dim(temp_draws[[1]])[1] else dim(temp_draws)[1],
      type = temp_info$type,
      components = if (identical(temp_info$type, "multiscale"))
        temp_info$components else temp_info$type,
      component_requested = component
    ),
    class = "ratiod_temporal_posterior"
  )

  if (summary) {
    return(summary(result, probs = probs))
  }

  result
}


#' Print method for ratiod_temporal_posterior
#'
#' @param x A ratiod_temporal_posterior object
#' @param ... Ignored
#'
#' @export
print.ratiod_temporal_posterior <- function(x, ...) {
  cat("Temporal effect posterior\n")
  cat("=========================\n\n")

  if (x$type == "multiscale") {
    cat("Type: Multi-scale decomposition\n")
    cat("Components:", paste(x$components, collapse = ", "), "\n")
    if (x$component_requested != "all") {
      cat("Extracted:", x$component_requested, "\n")
    }
  } else {
    cat("Type:", toupper(x$type), "\n")
  }

  cat("Time points:", x$n_times, "\n")
  if (x$n_groups > 1) {
    cat("Groups:", x$n_groups, "\n")
  }
  cat("Posterior draws:", x$n_draws, "\n")
  cat("\nUse summary() for posterior summaries\n")
  cat("Use plot() for visualization\n")
  invisible(x)
}


#' Summary method for ratiod_temporal_posterior
#'
#' @param object A ratiod_temporal_posterior object
#' @param probs Quantiles to compute
#' @param ... Ignored
#'
#' @export
summary.ratiod_temporal_posterior <- function(object, probs = c(0.025, 0.5, 0.975), ...) {

  # Handle multi-scale (list) vs single (matrix)
  if (is.list(object$draws) && !is.data.frame(object$draws)) {
    # Multi-scale: summarize each component
    results <- list()

    for (comp in names(object$draws)) {
      draws <- object$draws[[comp]]
      n_times <- ncol(draws)

      summaries <- data.frame(
        component = comp,
        time_idx = seq_len(n_times),
        time = if (!is.null(object$time_levels) && n_times == length(object$time_levels))
          object$time_levels else seq_len(n_times),
        mean = colMeans(draws),
        sd = apply(draws, 2, sd),
        t(apply(draws, 2, quantile, probs = probs))
      )
      names(summaries)[6:ncol(summaries)] <- paste0("q", probs * 100)
      rownames(summaries) <- NULL
      results[[comp]] <- summaries
    }

    result <- do.call(rbind, results)
  } else {
    # Single component
    draws <- object$draws
    n_times <- ncol(draws)

    result <- data.frame(
      time_idx = seq_len(n_times),
      time = if (!is.null(object$time_levels)) object$time_levels else seq_len(n_times),
      mean = colMeans(draws),
      sd = apply(draws, 2, sd),
      t(apply(draws, 2, quantile, probs = probs))
    )
    names(result)[5:ncol(result)] <- paste0("q", probs * 100)
    rownames(result) <- NULL
  }

  structure(
    result,
    n_draws = object$n_draws,
    class = c("ratiod_temporal_summary", "data.frame")
  )
}


#' Plot method for ratiod_temporal_posterior
#'
#' @param x A ratiod_temporal_posterior object
#' @param component Which component to plot (for multi-scale). Default: first.
#' @param type Plot type: "ribbon" (default) or "line"
#' @param ... Additional arguments passed to plotting functions
#'
#' @export
plot.ratiod_temporal_posterior <- function(x, component = NULL, type = "ribbon", ...) {

  # Get draws to plot
  if (is.list(x$draws) && !is.data.frame(x$draws)) {
    if (is.null(component)) {
      component <- names(x$draws)[1]
    }
    draws <- x$draws[[component]]
    title <- paste("Temporal effect:", component)
  } else {
    draws <- x$draws
    title <- paste("Temporal effect:", x$type)
  }

  n_times <- ncol(draws)
  times <- if (!is.null(x$time_levels) && n_times == length(x$time_levels)) {
    as.numeric(x$time_levels)
  } else {
    seq_len(n_times)
  }

  # Compute summaries
  means <- colMeans(draws)
  lower <- apply(draws, 2, quantile, probs = 0.025)
  upper <- apply(draws, 2, quantile, probs = 0.975)

  # Use ggplot2 if available
  if (requireNamespace("ggplot2", quietly = TRUE)) {
    df <- data.frame(
      time = times,
      mean = means,
      lower = lower,
      upper = upper
    )

    p <- ggplot2::ggplot(df, ggplot2::aes(x = .data$time, y = .data$mean)) +
      ggplot2::geom_ribbon(
        ggplot2::aes(ymin = .data$lower, ymax = .data$upper),
        alpha = 0.3, fill = "steelblue"
      ) +
      ggplot2::geom_line(color = "steelblue", linewidth = 1) +
      ggplot2::geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
      ggplot2::labs(
        title = title,
        x = "Time",
        y = "Effect"
      ) +
      theme_ratiod()

    return(p)
  }

  # Base R fallback
  ylim <- range(c(lower, upper))

  plot(times, means, type = "l", col = "steelblue", lwd = 2,
       ylim = ylim, xlab = "Time", ylab = "Effect",
       main = title, ...)

  polygon(c(times, rev(times)), c(lower, rev(upper)),
          col = adjustcolor("steelblue", alpha.f = 0.3), border = NA)

  abline(h = 0, lty = 2, col = "gray50")
  lines(times, means, col = "steelblue", lwd = 2)

  invisible(NULL)
}
