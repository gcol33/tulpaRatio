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
