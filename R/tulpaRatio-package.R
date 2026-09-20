#' @keywords internal
"_PACKAGE"

#' @useDynLib tulpaRatio, .registration = TRUE
#' @importFrom Rcpp sourceCpp
#' @importFrom stats as.formula model.frame model.matrix na.pass quantile sd optim fitted predict rnorm rnbinom var ar cor pnorm qnorm reshape qlogis
#' @importFrom utils head
#' @importFrom grDevices adjustcolor
#' @importFrom graphics abline lines polygon
NULL

# The diagnostic front door is tulpa's generic; tulpaRatio contributes the
# `ratiod_fit` method (see diagnostics.ratiod_fit). Re-exported so
# `library(tulpaRatio)` puts the same object on the search path rather than a
# same-named function that would mask the engine's.
#' @importFrom tulpa diagnostics
#' @export
tulpa::diagnostics

#' tulpaRatio: Bayesian Hierarchical Models for Ratios, Rates, and Proportions
#'
#' @description
#' An opinionated framework for Bayesian hierarchical modelling of ratios,
#' rates, and proportions. The core premise is non-negotiable:
#'
#' **Ratios are not data. They are derived quantities.**
#'
#' All inference is performed on the latent processes generating the numerator
#' and the denominator, never on their quotient. Ratios, rates, and proportions
#' are computed post hoc with full uncertainty propagation.
#'
#' @section Model families:
#' \describe{
#'   \item{ratiod_binomial()}{Trial-based ratios (successes/trials)}
#'   \item{ratiod_negbin_negbin()}{Two-process ratios (count/count)}
#'   \item{ratiod_poisson_gamma()}{Count/effort ratios (e.g., CPUE)}
#' }
#'
#' @section Key features:
#' \itemize{
#'   \item Shared random effects between numerator and denominator (default)
#'   \item Spatial structure via CAR/BYM2
#'   \item Full posterior inference on derived ratios
#'   \item Explicit rejection of offset-based approaches
#' }
#'
#' @name tulpaRatio-package
#' @aliases tulpaRatio-package
NULL

# Suppress R CMD check notes about global variables
utils::globalVariables(c(".", "value", "variable", ".data"))
