# Response validation --------------------------------------------------------
#
# Checked once at the front door, on the parsed responses, before any backend
# sees them. The families that coerce with as.integer() are the reason the
# check has to come first: a non-integer count is truncated silently by the
# coercion, and a numerator above its denominator reaches the likelihood,
# where portable_lchoose() returns -Inf for k > n and the log posterior is
# flat everywhere rather than the fit stopping with a message.

#' Validate a response vector against the distribution it is modelled by
#'
#' @param y Response vector
#' @param dist Distribution name carried by the family arm
#' @param name Arm name for error messages
#' @return `NULL`, invisibly; called for the error it may raise
#' @keywords internal
validate_response <- function(y, dist, name) {
  if (dist %in% c("poisson", "neg_binomial_2", "binomial", "beta_binomial")) {
    # Must be non-negative integers
    if (!is.numeric(y)) {
      stop(sprintf("%s response must be numeric", name), call. = FALSE)
    }
    if (any(y < 0, na.rm = TRUE)) {
      stop(sprintf("%s response must be non-negative", name), call. = FALSE)
    }
    if (any(y != floor(y), na.rm = TRUE)) {
      stop(sprintf("%s response must be integer counts", name), call. = FALSE)
    }
    # the samplers hold counts as 32-bit integers
    n_big <- sum(y > .Machine$integer.max, na.rm = TRUE)
    if (n_big > 0) {
      stop(sprintf("%s response has %d count(s) above %d, the largest the samplers hold.",
                   name, n_big, .Machine$integer.max), call. = FALSE)
    }
  } else if (dist %in% c("gamma", "lognormal")) {
    # Must be positive
    if (!is.numeric(y)) {
      stop(sprintf("%s response must be numeric", name), call. = FALSE)
    }
    if (any(y <= 0, na.rm = TRUE)) {
      stop(sprintf("%s response must be positive", name), call. = FALSE)
    }
  }
  invisible(NULL)
}

#' Is a family arm discrete
#' @keywords internal
response_is_discrete <- function(dist) {
  !is.null(dist) &&
    dist %in% c("poisson", "neg_binomial_2", "binomial", "beta_binomial")
}

#' Validate both responses of a parsed ratio formula against its family
#'
#' Applies the full per-arm check to the discrete arms, where a bad value has
#' no recovery: `as.integer()` truncates a non-integer silently, and a
#' numerator above its denominator makes the binomial coefficient `-Inf` at
#' every parameter value. A continuous arm is checked for type only, because
#' the backends clamp a non-positive gamma or lognormal value with a warning
#' and that is the behaviour those families are meant to have.
#'
#' @param formula A parsed `ratiod_formula`
#' @param family A `ratiod_family`
#' @return `NULL`, invisibly; called for the error it may raise
#' @keywords internal
validate_ratio_responses <- function(formula, family) {
  if (!inherits(family, "ratiod_family")) return(invisible(NULL))

  y_num <- formula$numerator$response
  y_denom <- formula$denominator$response

  num_dist <- family$numerator$distribution
  denom_dist <- family$denominator$distribution

  check_arm <- function(y, dist, name) {
    if (is.null(dist) || identical(dist, "fixed")) return(invisible(NULL))
    if (response_is_discrete(dist)) {
      validate_response(y, dist, name)
    } else if (!is.numeric(y)) {
      stop(sprintf("%s response must be numeric", name), call. = FALSE)
    }
    invisible(NULL)
  }

  check_arm(y_num, num_dist, "Numerator")
  check_arm(y_denom, denom_dist, "Denominator")

  # The trial-based families read the denominator as the trial count whether
  # or not they also model it, so it is checked in both cases.
  if (response_is_discrete(num_dist) &&
      num_dist %in% c("binomial", "beta_binomial")) {
    validate_response(y_denom, "binomial", "Denominator (trials)")
    if (any(y_num > y_denom, na.rm = TRUE)) {
      n_bad <- sum(y_num > y_denom, na.rm = TRUE)
      stop(sprintf(
        "Numerator (successes) cannot exceed denominator (trials): %d of %d observations do.",
        n_bad, length(y_num)), call. = FALSE)
    }
  }

  invisible(NULL)
}
