#' Null-coalescing operator
#'
#' @description
#' Returns `a` unless it is `NULL`, in which case it returns `b`. Used
#' throughout the package to fall back on a default when an optional list
#' element is absent. Defined here because the base R version arrived in
#' 4.4.0 and the package supports R (>= 4.1.0).
#'
#' @param a Value to return when it is not `NULL`.
#' @param b Fallback returned when `a` is `NULL`.
#'
#' @return `a` if it is not `NULL`, otherwise `b`.
#'
#' @keywords internal
#' @noRd
`%||%` <- function(a, b) if (is.null(a)) b else a

cat_title <- function(title, blank = TRUE) {
  cat(title, "\n", strrep("=", nchar(title, type = "width")),
      if (blank) "\n\n" else "\n", sep = "")
}

cat_shared <- function(shared, yes = "Yes (enters both processes)", no = "No",
                       prefix = "") {
  cat(paste0(prefix, "Shared:"), if (shared) yes else no, "\n")
}

print_point_fit <- function(x, title, extra, means, sds, digits = 3,
                            parameters = names(means)) {
  cat(title, "\n", sep = "")
  cat("Family:", x$family$name, "\n")
  cat("Observations:", x$n_obs, "\n")
  cat("Parameters:", x$n_params)
  extra()
  cat("Posterior means:\n")
  coef_summary <- data.frame(parameter = parameters, mean = means, sd = sds)
  n_show <- min(10, nrow(coef_summary))
  print(coef_summary[seq_len(n_show), ], digits = digits, row.names = FALSE)
  if (nrow(coef_summary) > n_show) {
    cat("... and", nrow(coef_summary) - n_show, "more parameters\n")
  }
  invisible(x)
}

print_stochastic_fit <- function(x, title, digits, epsilon = FALSE) {
  print_point_fit(x, title, extra = function() {
    cat(" \n")
    cat("Batch size:", x$batch_size, "(", round(100 * x$batch_size / x$n_obs, 1), "%)\n\n")
    cat("Sampling:\n")
    cat("  Iterations:", x$iter, "(warmup:", x$warmup, ")\n")
    if (epsilon) cat("  Final epsilon:", round(x$final_epsilon, 6), "\n")
    cat("\n")
  }, means = colMeans(x$draws), sds = apply(x$draws, 2, sd), digits = digits,
  parameters = colnames(x$draws))
}

varying_coef_summary <- function(draws, id_cols, probs) {
  results <- lapply(seq_len(dim(draws)[3]), function(j) {
    term_draws <- draws[, , j]
    quantiles <- t(apply(term_draws, 2, quantile, probs = probs))
    summaries <- data.frame(id_cols[[j]], mean = colMeans(term_draws),
                            sd = apply(term_draws, 2, sd), quantiles)
    quantile_cols <- seq.int(ncol(id_cols[[j]]) + 3L, ncol(summaries))
    names(summaries)[quantile_cols] <- paste0("q", probs * 100)
    rownames(summaries) <- NULL
    summaries
  })
  do.call(rbind, results)
}

print_varying_coef <- function(x, title, locations, detail, visualization) {
  cat_title(title)
  cat("Terms:", paste(x$term_names, collapse = ", "), "\n")
  cat(paste0(locations[1], ":"), locations[2], "\n")
  cat("Posterior draws:", x$n_draws, "\n")
  cat(paste0(detail[1], ":"), detail[2], "\n")
  cat("\nUse summary() for posterior summaries\n")
  cat("Use plot() for", visualization, "visualization\n")
  invisible(x)
}

varying_coef_term <- function(x, term) {
  if (is.character(term)) {
    index <- match(term, x$term_names)
    if (is.na(index)) stop("Term not found: ", term, call. = FALSE)
    return(index)
  }
  term
}

varying_coef_terms <- function(terms, term_names) {
  if (is.numeric(terms)) return(terms)
  if (!is.character(terms)) {
    stop("`terms` must be numeric or character", call. = FALSE)
  }
  indices <- match(terms, term_names)
  if (anyNA(indices)) {
    stop("Terms not found: ", paste(terms[is.na(indices)], collapse = ", "),
         call. = FALSE)
  }
  indices
}


#' Ridged precision mass matrix
#'
#' Adds `ridge * I` to a (possibly rank-deficient) precision matrix and
#' factors it once. The HMC mass matrix uses both the inverse, the prior
#' covariance, and the lower Cholesky factor.
#'
#' @param Q Symmetric positive semi-definite precision matrix.
#' @param ridge Diagonal ridge making `Q` positive definite.
#' @return List with `Q_inv` and `L` (lower Cholesky factor, `L %*% t(L)`
#'   equals the ridged `Q`), each flattened column-major.
#' @keywords internal
#' @noRd
ridged_precision_factors <- function(Q, ridge = 0.01) {
  R <- chol(Q + ridge * diag(nrow(Q)))
  list(Q_inv = as.numeric(chol2inv(R)), L = as.numeric(t(R)))
}
# Logistic CDF with log probabilities and upper-tail probabilities.
inv_logit <- function(x, lower.tail = TRUE, log.p = FALSE) {
  stats::plogis(x, lower.tail = lower.tail, log.p = log.p)
}
