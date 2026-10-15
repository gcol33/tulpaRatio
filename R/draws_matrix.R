#' Draw matrices of predictive and simulated quantities
#'
#' @description
#' Prior predictive draws ([prior_predict()]), posterior predictive replicates
#' ([posterior_predict()]) and predictive draws at new data
#' (`predict(fit, summary = FALSE)`) each hold one draws-by-observation matrix
#' per component. `as.matrix()` returns the one named by `component`.
#'
#' @param x A `ratiod_priorpred`, `ratiod_ppred` or `ratiod_prediction_draws`
#'   object.
#' @param component Which quantity: `"ratio"`, `"numerator"` or
#'   `"denominator"`. Posterior predictive replicates carry the numerator and the
#'   denominator only; predictive draws at new data carry the components
#'   `predict()` was asked for. The default is the first available.
#' @param ... Ignored.
#'
#' @return A numeric matrix, one row per draw and one column per observation.
#'
#' @examples
#' set.seed(1)
#' df <- data.frame(count = 0L, total = 0L, x = rnorm(20))
#' pp <- prior_predict(count | total ~ x, family = ratiod_negbin_negbin(),
#'                     data = df, n = 30)
#' dim(as.matrix(pp))
#' range(as.matrix(pp, component = "denominator"))
#'
#' @name as.matrix.ratiod_priorpred
NULL

draws_component <- function(x, component, available, fields) {
  component <- if (is.null(component)) available[1L] else
    match.arg(component, c("ratio", "numerator", "denominator"))
  if (!component %in% available) {
    stop(sprintf("`component = \"%s\"` is not available; this object holds %s.",
                 component, paste0("\"", available, "\"", collapse = ", ")),
         call. = FALSE)
  }
  unclass(x)[[fields[[component]]]]
}

#' @rdname as.matrix.ratiod_priorpred
#' @export
as.matrix.ratiod_priorpred <- function(x, component = NULL, ...) {
  draws_component(x, component, c("ratio", "numerator", "denominator"),
                  c(ratio = "ratio", numerator = "y_num", denominator = "y_denom"))
}

#' @rdname as.matrix.ratiod_priorpred
#' @export
as.matrix.ratiod_ppred <- function(x, component = NULL, ...) {
  draws_component(x, component, c("numerator", "denominator"),
                  c(numerator = "y_num_rep", denominator = "y_denom_rep"))
}

#' @rdname as.matrix.ratiod_priorpred
#' @export
as.matrix.ratiod_prediction_draws <- function(x, component = NULL, ...) {
  available <- names(unclass(x))
  draws_component(x, component, available, stats::setNames(available, available))
}

#' @rdname as.matrix.ratiod_priorpred
#' @export
print.ratiod_ppred <- function(x, ...) {
  cat("Posterior predictive replicates\n")
  cat("  Draws:       ", nrow(x$y_num_rep), "\n", sep = "")
  cat("  Observations:", ncol(x$y_num_rep), "\n")
  invisible(x)
}

#' @rdname as.matrix.ratiod_priorpred
#' @export
print.ratiod_prediction_draws <- function(x, ...) {
  cat("Predictive draws at new data\n")
  cat("  Components:  ", paste(names(unclass(x)), collapse = ", "), "\n", sep = "")
  cat("  Draws:       ", attr(x, "n_draws"), "\n", sep = "")
  cat("  Observations:", attr(x, "n_obs"), "\n")
  invisible(x)
}
