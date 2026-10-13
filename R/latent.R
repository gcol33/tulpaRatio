#' Latent Factors for Unmeasured Confounders
#'
#' @description
#' Specify latent factors to capture shared unmeasured confounders between
#' numerator and denominator processes. Latent factors are particularly useful
#' when you suspect that both processes are driven by common unmeasured variables.
#'
#' @examples
#' latent_factor(n_factors = 2)
#' @name ratiod_latent
NULL


#' Latent Factor Specification
#'
#' @description
#' Define latent factors for capturing unmeasured structure shared between
#' numerator and denominator. Each observation carries factor scores, and
#' each process loads on them with its own loading, so the factors model the
#' residual covariance of an observation's two linear predictors.
#'
#' @param n_factors Integer; number of latent factors. Default is 1. With
#'   `shared = TRUE` at most 2 factors are identified, with `shared = FALSE`
#'   at most 1 (see Details).
#' @param prior Prior for the diagonal loadings. Default is a PC prior with
#'   P(loading > 1) = 0.01, which shrinks toward simpler models.
#' @param shared Logical; if TRUE (default), the factors load on both the
#'   numerator and the denominator linear predictor. If FALSE, they load on
#'   the numerator only.
#' @param constraint Identifiability constraint for the factor scores:
#'   - `"sum_to_zero"` (default): each factor's scores sum to zero across
#'     observations
#'   - `"first_zero"`: the first observation's scores are fixed at zero
#'
#' @return A `ratiod_latent` object for use in [tratio()].
#'
#' @details
#'
#' ## Why Use Latent Factors?
#'
#' When modeling ratios, the numerator and denominator often share unmeasured
#' drivers. For example, in relative abundance data, both the focal species
#' count and total count might be affected by:
#' - Observer skill (unmeasured)
#' - Local microhabitat conditions (unmeasured)
#' - Weather on sampling day (unmeasured)
#'
#' Latent factors capture this shared structure without requiring the
#' drivers to be measured.
#'
#' ## Mathematical Model
#'
#' For observation i with K latent factors:
#'
#' \deqn{\eta^{num}_i = X^{num}_i \beta^{num} + \sum_{k=1}^{K} \lambda_{1k} f_{ik} + \ldots}
#' \deqn{\eta^{denom}_i = X^{denom}_i \beta^{denom} + \sum_{k=1}^{K} \lambda_{2k} f_{ik} + \ldots}
#'
#' where:
#' - \eqn{f_{ik} \sim N(0, 1)} are factor scores, with
#'   \eqn{\sum_i f_{ik} = 0} for each k
#' - \eqn{\Lambda = (\lambda_{pk})} is the loading matrix, lower triangular
#'   with a positive diagonal
#'
#' The factors add the covariance \eqn{\Lambda \Lambda^\top} between the two
#' linear predictors of an observation. A 2 x 2 covariance has three free
#' entries: one factor (two loadings) fills it up to rank one and two factors
#' (three loadings) fill it completely, so a third factor would not be
#' identified. Fixing \eqn{\Lambda} lower triangular with a positive diagonal
#' makes it the Cholesky factor of that covariance, which is unique; without
#' it the loadings and scores rotate freely.
#'
#' The ratio's linear predictor gains
#' \eqn{\sum_k (\lambda_{1k} - \lambda_{2k}) f_{ik}}: a factor that loads
#' equally on both processes cancels in the ratio, one that loads unequally
#' moves it.
#'
#' ## Priors
#'
#' Each diagonal loading carries `prior` (a PC prior by default). Each
#' off-diagonal loading carries a normal prior centred at zero, with the
#' standard deviation that gives \eqn{|\lambda|} the same 99% quantile as the
#' diagonal prior.
#'
#' ## Choosing n_factors
#'
#' - `n_factors = 1`: one unmeasured driver acting on both processes, with
#'   process-specific strength
#' - `n_factors = 2`: an unrestricted residual covariance between the two
#'   processes
#'
#' ## Relationship to Random Effects
#'
#' Latent factors differ from random effects in several ways:
#' - Random effects are grouped (e.g., site-level), factors are observation-level
#' - Random effects require grouping structure, factors don't
#' - Factors capture residual correlation not explained by observed predictors
#'
#' You can use both together: random effects for known grouping, factors for
#' residual unmeasured structure.
#'
#' @examples
#' # Basic latent factor (single factor on both processes)
#' latent_factor()
#'
#' # Two latent factors: an unrestricted residual covariance
#' latent_factor(n_factors = 2)
#'
#' # Custom prior (more regularization)
#' latent_factor(n_factors = 1, prior = prior_pc(U = 0.5, alpha = 0.01))
#'
#' # Numerator-only factor (not shared)
#' latent_factor(n_factors = 1, shared = FALSE)
#'
#' \donttest{
#' # Use in model fitting
#' set.seed(1)
#' n <- 60
#' df <- data.frame(
#'   species_count = rnbinom(n, size = 5, mu = 15),
#'   total_count = rnbinom(n, size = 8, mu = 100),
#'   habitat = factor(rep(c("forest", "grassland"), each = n / 2)),
#'   site = factor(rep(1:10, each = n / 10))
#' )
#' fit <- tratio(
#'   species_count | total_count ~ habitat + (1 | site),
#'   data = df,
#'   family = ratiod_negbin_negbin(),
#'   latent = latent_factor(n_factors = 1),
#'   control = list(iter = 400, chains = 1, seed = 1)
#' )
#' head(latent_factors(fit))
#' }
#'
#' @seealso [tratio()] for model fitting, [prior_pc()] for prior specification
#'
#' @export
latent_factor <- function(
    n_factors = 1L,
    prior = NULL,
    shared = TRUE,
    constraint = c("sum_to_zero", "first_zero")
) {

  # Validate n_factors
  if (length(n_factors) != 1 || !is.numeric(n_factors) || n_factors < 1 ||
      n_factors != as.integer(n_factors)) {
    stop("`n_factors` must be a positive integer", call. = FALSE)
  }
  n_factors <- as.integer(n_factors)

  # Default prior
  if (is.null(prior)) {
    prior <- prior_pc(U = 1.0, alpha = 0.01)
  }

  # Validate prior
  if (!inherits(prior, "ratiod_prior")) {
    stop("`prior` must be a prior object (use prior_*() functions)", call. = FALSE)
  }

  # Validate shared
  if (!is.logical(shared) || length(shared) != 1) {
    stop("`shared` must be TRUE or FALSE", call. = FALSE)
  }

  n_proc <- if (shared) 2L else 1L
  if (n_factors > n_proc) {
    stop(n_factors, " latent factors on ", n_proc, " process(es): at most ",
         n_proc, " are identified.", call. = FALSE)
  }

  # Validate constraint
  constraint <- match.arg(constraint)

  structure(
    list(
      n_factors = n_factors,
      prior = prior,
      shared = shared,
      constraint = constraint
    ),
    class = c("ratiod_latent", "list")
  )
}


#' @rdname latent_factor
#' @param x A ratiod_latent object
#' @param ... Ignored
#'
#' @export
print.ratiod_latent <- function(x, ...) {
  cat("Latent factor specification\n")
  cat("===========================\n\n")

  cat("Number of factors:", x$n_factors, "\n")
  cat("Shared:", ifelse(x$shared, "Yes (loads on both num and denom)", "No (numerator only)"), "\n")
  cat("Constraint:", x$constraint, "\n\n")

  cat("Diagonal loading prior:\n")
  print_prior(x$prior, indent = "  ")

  invisible(x)
}


#' Validate latent factor specification against data
#'
#' @param latent A ratiod_latent object
#' @param N Number of observations
#'
#' @return Validated latent specification with additional computed fields
#' @keywords internal
validate_latent <- function(latent, N) {

  if (is.null(latent)) {
    return(NULL)
  }

  if (!inherits(latent, "ratiod_latent")) {
    stop("`latent` must be a ratiod_latent object from latent_factor()",
         call. = FALSE)
  }

  if (latent$n_factors >= N) {
    stop("Number of latent factors (", latent$n_factors,
         ") must be less than number of observations (", N, ")",
         call. = FALSE)
  }

  latent$n_obs <- N

  latent
}


#' Prepare latent factor data for HMC
#'
#' @param latent A validated ratiod_latent object (or NULL)
#' @param N Number of observations
#'
#' @return List with latent factor data for C++
#' @keywords internal
prepare_latent_for_hmc <- function(latent, N) {

  if (is.null(latent)) {
    return(list(
      type = "none",
      n_factors = 0L,
      n_obs = as.integer(N),
      shared = FALSE,
      constraint = "sum_to_zero",
      sigma_prior_rate = 0.0,
      offdiag_sd = 1.0
    ))
  }

  # Validate
  latent <- validate_latent(latent, N)

  # Extract prior rate for PC prior (exponential prior on sigma)
  sigma_prior_rate <- if (latent$prior$distribution == "pc") {
    latent$prior$rate
  } else if (latent$prior$distribution == "exponential") {
    latent$prior$rate
  } else if (latent$prior$distribution == "half_normal") {
    # Convert half-normal to approximate exponential rate
    # E[X] = sd * sqrt(2/pi), so rate ~= sqrt(pi/2) / sd
    sqrt(pi / 2) / latent$prior$sd
  } else if (latent$prior$distribution == "half_cauchy") {
    # Use scale as rough guide
    1.0 / latent$prior$scale
  } else {
    # Default
    -log(0.01) / 1.0  # PC prior default
  }

  # An off-diagonal loading is signed: N(0, sd^2) with the sd that gives
  # |lambda| the diagonal prior's 99% quantile
  offdiag_sd <- (-log(0.01) / sigma_prior_rate) / stats::qnorm(0.995)

  list(
    type = "latent",
    n_factors = as.integer(latent$n_factors),
    n_obs = as.integer(N),
    shared = latent$shared,
    constraint = latent$constraint,
    sigma_prior_rate = sigma_prior_rate,
    offdiag_sd = offdiag_sd
  )
}


#' @rdname latent_factor
#' @param fit A `ratiod_fit` sampled with `latent = latent_factor(...)`.
#' @param summary Logical; `TRUE` (default) returns one row per observation
#'   and factor with the posterior mean, sd and `probs` quantiles of its score,
#'   `FALSE` the draws matrix with columns `factor[observation,factor]`.
#' @param probs Quantiles for the summary.
#' @exportS3Method tulpa::latent_factors
latent_factors.ratiod_fit <- function(fit, summary = TRUE,
                                      probs = c(0.025, 0.5, 0.975), ...) {
  scores <- fit$.internal$latent_draws
  info <- fit$.internal$latent_info
  if (is.null(scores) || is.null(info) || identical(info$type, "none")) {
    stop("Model was not fit with latent factors", call. = FALSE)
  }
  n_obs <- dim(scores)[2]
  n_factors <- dim(scores)[3]
  draws <- matrix(scores, nrow = dim(scores)[1])
  colnames(draws) <- sprintf("factor[%d,%d]", rep(seq_len(n_obs), n_factors),
                             rep(seq_len(n_factors), each = n_obs))
  tulpa::latent_factors(list(draws = draws, .internal = list(latent_info = info)),
                        summary = summary, probs = probs)
}
