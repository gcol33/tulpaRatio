fixed_effect_variables <- function(X, arm = "num") {
  if (is.null(X) || ncol(X) == 0L) return(character(0))
  sprintf("beta_%s[%d]", arm, seq_len(ncol(X)))
}

draw_variables <- function(draws) {
  dimnames(draws)[[length(dim(draws))]]
}

draw_matrix <- function(draws) {
  dims <- dim(draws)
  matrix(as.numeric(draws), nrow = prod(dims[-length(dims)]),
         ncol = dims[length(dims)], dimnames = list(NULL, draw_variables(draws)))
}

#' @importFrom tulpa fixef
#' @export
tulpa::fixef

#' @importFrom generics tidy
#' @export
generics::tidy

#' @rdname summary.ratiod_fit
#' @export
fixef.ratiod_fit <- function(object, ...) {
  coef(object, ...)
}

#' @rdname summary.ratiod_fit
#' @param conf.int Include central posterior intervals in the tidy table.
#' @param conf.level Probability of the central posterior interval.
#' @export
tidy.ratiod_fit <- function(x, conf.int = FALSE, conf.level = 0.95, ...) {
  draws <- fixed_effect_draws(x)
  out <- data.frame(term = colnames(draws), estimate = colMeans(draws),
                    std.error = apply(draws, 2L, stats::sd), row.names = NULL)
  if (conf.int) {
    interval <- stats::confint(x, level = conf.level)
    out$conf.low <- interval[, 1L]
    out$conf.high <- interval[, 2L]
  }
  out
}
