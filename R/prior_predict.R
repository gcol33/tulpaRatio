# Prior predictive simulation ------------------------------------------------
#
# Draws parameters from the prior, assembles the two linear predictors on the
# design the formula and data imply, and draws a numerator and denominator
# response from the family. The three pieces -- prior sampler, inverse link,
# response sampler -- are each keyed on the name the corresponding object
# carries, so a new prior, link or family arm is one table entry rather than a
# new branch in the driver.

#' Draw from a prior specification
#'
#' @param prior A `ratiod_prior` object
#' @param n Number of draws
#' @return Numeric vector of length `n`
#' @keywords internal
pp_draw_prior <- function(prior, n) {
  switch(
    prior$distribution,
    normal       = stats::rnorm(n, prior$mean, prior$sd),
    half_normal  = abs(stats::rnorm(n, 0, prior$sd)),
    half_cauchy  = abs(stats::rcauchy(n, 0, prior$scale)),
    gamma        = stats::rgamma(n, shape = prior$shape, rate = prior$rate),
    exponential  = stats::rexp(n, rate = prior$rate),
    beta         = stats::rbeta(n, prior$alpha, prior$beta),
    # A PC prior on a standard deviation is exponential on that scale, with
    # rate -log(alpha) / U -- the calibration prior_pc() already stores.
    pc           = stats::rexp(n, rate = prior$rate),
    stop(sprintf("Cannot simulate from prior distribution '%s'",
                 prior$distribution), call. = FALSE)
  )
}

#' Inverse link function by name
#'
#' @param link Link name
#' @return A function mapping the linear predictor to the mean
#' @keywords internal
pp_linkinv <- function(link) {
  switch(
    link,
    log     = exp,
    identity = function(eta) eta,
    inverse = function(eta) 1 / eta,
    logit   = stats::plogis,
    probit  = stats::pnorm,
    cloglog = function(eta) -expm1(-exp(eta)),
    stop(sprintf("Unsupported link '%s'", link), call. = FALSE)
  )
}

#' Link function by name
#'
#' @param link Link name
#' @return A function mapping a mean (or probability) to the linear predictor
#' @keywords internal
pp_linkfun <- function(link) {
  switch(
    link,
    log      = log,
    identity = function(mu) mu,
    inverse  = function(mu) 1 / mu,
    logit    = stats::qlogis,
    probit   = stats::qnorm,
    cloglog  = function(mu) log(-log1p(-mu)),
    stop(sprintf("Unsupported link '%s'", link), call. = FALSE)
  )
}

#' Log probability and log complement under a probability link
#'
#' @param eta Linear predictor
#' @param link Link name, one of `LINKS_PROBABILITY`
#' @param lower `TRUE` for log p, `FALSE` for log(1 - p)
#' @return Array the shape of `eta`
#' @keywords internal
pp_log_prob <- function(eta, link, lower = TRUE) {
  switch(
    link,
    logit   = stats::plogis(eta, lower.tail = lower, log.p = TRUE),
    probit  = stats::pnorm(eta, lower.tail = lower, log.p = TRUE),
    cloglog = if (lower) log(-expm1(-exp(eta))) else -exp(eta),
    stop(sprintf("Unsupported link '%s'", link), call. = FALSE)
  )
}

#' Draw a response vector from one arm of a family
#'
#' @param distribution Distribution name carried by the family arm
#' @param mu Mean vector
#' @param disp Dispersion parameter for the arm (size, shape, or sdlog)
#' @param trials Trial counts, for the binomial arms
#' @return Numeric vector the length of `mu`
#' @keywords internal
pp_rdist <- function(distribution, mu, disp, trials = NULL) {
  N <- length(mu)
  switch(
    distribution,
    poisson        = stats::rpois(N, mu),
    neg_binomial_2 = stats::rnbinom(N, mu = mu, size = disp),
    # Mean mu at shape `disp`, so rate = shape / mu.
    gamma          = stats::rgamma(N, shape = disp, rate = disp / mu),
    # Median mu on the response scale: eta is the log-mean of the underlying
    # normal, so meanlog = log(mu) is eta itself under the log link.
    lognormal      = stats::rlnorm(N, meanlog = log(mu), sdlog = disp),
    binomial       = stats::rbinom(N, size = trials, prob = mu),
    beta_binomial  = {
      # disp is the beta-binomial precision: a = mu * disp, b = (1 - mu) * disp.
      p <- stats::rbeta(N, mu * disp, (1 - mu) * disp)
      stats::rbinom(N, size = trials, prob = p)
    },
    stop(sprintf("Cannot simulate from family arm '%s'", distribution),
         call. = FALSE)
  )
}

#' Random-effect contribution to a linear predictor
#'
#' Draws one standard deviation per term from `sigma_prior`, then the group
#' effects at that scale, and accumulates their contribution. Slope terms get
#' one standard deviation per column; the correlation between slopes is not
#' drawn, so a correlated term is simulated at its marginal scales.
#'
#' @param re_terms The `random_effects` list from a parsed formula
#' @param sigma_prior Prior on the random-effect standard deviation
#' @param N Number of observations
#' @param prefix Arm label the drawn scales are named under
#' @return A list with the contribution `eta` and the drawn scales `sigma`
#' @keywords internal
pp_draw_re <- function(re_terms, sigma_prior, N, prefix) {
  eta <- numeric(N)
  sigmas <- numeric(0)

  for (term in re_terms) {
    grp <- term$group
    G <- term$n_groups

    if (isTRUE(term$has_intercept)) {
      s <- pp_draw_prior(sigma_prior, 1L)
      b <- stats::rnorm(G, 0, s)
      eta <- eta + b[grp]
      sigmas <- c(sigmas, stats::setNames(
        s, paste0(prefix, ".", term$group_var, ".sigma")))
    }

    if (!is.null(term$slope_matrix) && ncol(term$slope_matrix) > 0) {
      Z <- term$slope_matrix
      for (k in seq_len(ncol(Z))) {
        s <- pp_draw_prior(sigma_prior, 1L)
        b <- stats::rnorm(G, 0, s)
        eta <- eta + b[grp] * Z[, k]
        nm <- paste0(prefix, ".", term$group_var, ".", colnames(Z)[k], ".sigma")
        sigmas <- c(sigmas, stats::setNames(s, nm))
      }
    }
  }

  list(eta = eta, sigma = sigmas)
}

#' Does a family arm carry a dispersion parameter
#' @keywords internal
pp_has_dispersion <- function(distribution) {
  distribution %in% c("neg_binomial_2", "gamma", "lognormal", "beta_binomial")
}

#' Prior Predictive Simulation
#'
#' @description
#' Simulate numerator and denominator responses from the prior predictive
#' distribution: parameters are drawn from `priors`, the two linear predictors
#' are assembled on the design `formula` and `data` imply, and a response is
#' drawn from `family` at each draw. Comparing the resulting ratios against
#' what is plausible for the system under study is the check on whether the
#' priors are reasonable before any data are fitted.
#'
#' Random-effect terms contribute at a standard deviation drawn from
#' `priors$sigma`, one per intercept and one per slope column. Correlation
#' between slopes within a term is not drawn, so a correlated term simulates
#' at its marginal scales.
#'
#' @param formula A model formula, using the combined `num | denom ~ x` syntax
#'   or a single response with `formula_denom` supplied through
#'   [ratiod_formula()].
#' @param family A `ratiod_family` object.
#' @param data Data frame supplying the predictors, the grouping factors and,
#'   for the binomial families, the trial counts. Response columns need not be
#'   present; where one is missing a placeholder is used, since nothing about
#'   the prior predictive distribution depends on the observed response.
#' @param priors A `ratiod_priors` object. Defaults to `ratiod_priors()`.
#' @param n Number of prior predictive draws.
#'
#' @return An object of class `ratiod_priorpred` holding `n` draws of the
#'   ratio, the numerator and the denominator at each of the `N` rows of
#'   `data`, which [as.matrix()][as.matrix.ratiod_priorpred] returns as
#'   `n` x `N` matrices, together with the parameter values each draw was
#'   generated from.
#'
#' @examples
#' set.seed(1)
#' df <- data.frame(
#'   count = rep(0L, 40),
#'   total = rep(0L, 40),
#'   x = rnorm(40),
#'   site = factor(rep(1:8, each = 5))
#' )
#'
#' pp <- prior_predict(
#'   count | total ~ x + (1 | site),
#'   family = ratiod_negbin_negbin(),
#'   data = df,
#'   n = 50
#' )
#' pp
#' dim(as.matrix(pp))
#'
#' # A tighter coefficient prior narrows the predicted ratios
#' pp_tight <- prior_predict(
#'   count | total ~ x + (1 | site),
#'   family = ratiod_negbin_negbin(),
#'   data = df,
#'   priors = ratiod_priors(beta = prior_normal(0, 0.5)),
#'   n = 50
#' )
#' pp_tight
#'
#' @seealso [sim_ratiod()] for simulating from a known parameterisation,
#'   [ratiod_priors()] for the prior specification
#'
#' @export
prior_predict <- function(formula, family, data, priors = NULL, n = 100) {

  if (!inherits(family, "ratiod_family")) {
    stop("`family` must be a ratiod_family object", call. = FALSE)
  }
  if (!is.data.frame(data)) {
    stop("`data` must be a data frame", call. = FALSE)
  }
  n <- as.integer(n)
  if (length(n) != 1L || is.na(n) || n < 1L) {
    stop("`n` must be a single positive integer", call. = FALSE)
  }
  if (!is.null(priors) && !inherits(priors, "ratiod_priors")) {
    stop("`priors` must be a ratiod_priors object (see ratiod_priors())",
         call. = FALSE)
  }
  priors <- fill_priors(priors, get_hmc_model_type(family))

  # The parser reads the response columns to size and name things. Nothing
  # about the prior predictive distribution depends on their values, so a
  # missing one is filled with a placeholder rather than demanded of the user.
  work <- data
  for (v in all.vars(formula[[2]])) {
    if (!v %in% names(work)) work[[v]] <- 0
  }

  parsed <- ratiod_formula(formula, data = work)
  X_num <- parsed$numerator$X
  X_denom <- parsed$denominator$X
  N <- nrow(X_num)

  num_dist <- family$numerator$distribution
  denom_dist <- family$denominator$distribution
  num_linkinv <- pp_linkinv(family$numerator$link)
  denom_fixed <- identical(denom_dist, "fixed")
  denom_linkinv <- if (denom_fixed) NULL else pp_linkinv(family$denominator$link)

  # The binomial arms need trial counts, which are data rather than a drawn
  # quantity. They come from the denominator response column.
  trials <- NULL
  if (num_dist %in% c("binomial", "beta_binomial")) {
    tv <- parsed$denominator$response_var
    if (!tv %in% names(data)) {
      stop(sprintf(
        "Trial counts are needed for a %s numerator; column '%s' is not in `data`.",
        num_dist, tv), call. = FALSE)
    }
    trials <- as.integer(data[[tv]])
    if (anyNA(trials) || any(trials < 0)) {
      stop(sprintf("Trial counts in '%s' must be non-negative integers", tv),
           call. = FALSE)
    }
  }

  # A fixed denominator is data: the trial counts for the binomial arms, a
  # known exposure (the observed column) otherwise, which scales the
  # numerator mean
  fixed_denom <- NULL
  exposure <- NULL
  if (denom_fixed) {
    fixed_denom <- if (!is.null(trials)) trials else {
      dv <- parsed$denominator$response_var
      if (dv %in% names(data)) as.numeric(data[[dv]]) else rep(NA_real_, N)
    }
    if (is.null(trials)) exposure <- fixed_denom
  }

  p_num <- ncol(X_num)
  p_denom <- if (denom_fixed) 0L else ncol(X_denom)

  beta_num <- matrix(pp_draw_prior(priors$beta, n * p_num), n, p_num)
  colnames(beta_num) <- colnames(X_num)
  beta_denom <- matrix(numeric(0), n, 0)
  if (p_denom > 0) {
    beta_denom <- matrix(pp_draw_prior(priors$beta, n * p_denom), n, p_denom)
    colnames(beta_denom) <- colnames(X_denom)
  }

  phi_num <- if (pp_has_dispersion(num_dist)) pp_draw_prior(priors$phi, n) else
    rep(NA_real_, n)
  phi_denom <- if (!denom_fixed && pp_has_dispersion(denom_dist))
    pp_draw_prior(priors$phi, n) else rep(NA_real_, n)

  mu_num <- matrix(NA_real_, n, N)
  mu_denom <- matrix(NA_real_, n, N)
  y_num <- matrix(NA_real_, n, N)
  y_denom <- matrix(NA_real_, n, N)
  sigma_draws <- vector("list", n)

  re_num <- parsed$numerator$random_effects
  re_denom <- parsed$denominator$random_effects

  for (s in seq_len(n)) {
    eta_num <- as.numeric(X_num %*% beta_num[s, ])
    re_n <- pp_draw_re(re_num, priors$sigma, N, "num")
    eta_num <- eta_num + re_n$eta
    sigma_s <- re_n$sigma

    mu_num[s, ] <- num_linkinv(eta_num)
    if (!is.null(exposure)) mu_num[s, ] <- mu_num[s, ] * exposure
    y_num[s, ] <- pp_rdist(num_dist, mu_num[s, ], phi_num[s], trials)

    if (denom_fixed) {
      mu_denom[s, ] <- fixed_denom
      y_denom[s, ] <- fixed_denom
    } else {
      eta_denom <- as.numeric(X_denom %*% beta_denom[s, ])
      re_d <- pp_draw_re(re_denom, priors$sigma, N, "denom")
      eta_denom <- eta_denom + re_d$eta
      sigma_s <- c(sigma_s, re_d$sigma)

      mu_denom[s, ] <- denom_linkinv(eta_denom)
      y_denom[s, ] <- pp_rdist(denom_dist, mu_denom[s, ], phi_denom[s], trials)
    }

    sigma_draws[[s]] <- sigma_s
  }

  sigma_names <- if (length(sigma_draws[[1]])) names(sigma_draws[[1]]) else
    character(0)
  sigma <- matrix(unlist(sigma_draws), nrow = n, byrow = TRUE,
                  dimnames = list(NULL, sigma_names))

  # Over trial counts the modelled quantity is a probability, and the
  # reported ratio is that probability, the convention fitted() uses; over a
  # known exposure it is the rate, the numerator mean over the exposure.
  ratio <- if (!is.null(trials)) mu_num else mu_num / mu_denom

  structure(
    list(
      y_num = y_num,
      y_denom = y_denom,
      ratio = ratio,
      mu_num = mu_num,
      mu_denom = mu_denom,
      beta_num = beta_num,
      beta_denom = beta_denom,
      sigma = sigma,
      phi_num = phi_num,
      phi_denom = phi_denom,
      family = family,
      priors = priors,
      data = data,
      n = n,
      n_obs = N
    ),
    class = "ratiod_priorpred"
  )
}

#' @rdname prior_predict
#' @param x A `ratiod_priorpred` object
#' @param ... Ignored
#' @return `x`, invisibly
#' @export
print.ratiod_priorpred <- function(x, ...) {
  cat("Prior predictive draws\n")
  cat("  Family:      ", x$family$name, "\n", sep = "")
  cat("  Draws:       ", x$n, "\n", sep = "")
  cat("  Observations:", x$n_obs, "\n")

  qs <- stats::quantile(as.vector(x$ratio), c(0.025, 0.5, 0.975),
                        na.rm = TRUE, names = FALSE)
  cat("\n  Implied ratio, 2.5% / 50% / 97.5%:\n    ",
      paste(format(qs, digits = 3), collapse = "  "), "\n", sep = "")

  finite <- is.finite(as.vector(x$ratio))
  if (any(!finite)) {
    cat("  ", sum(!finite), " of ", length(finite),
        " ratios are not finite\n", sep = "")
  }
  invisible(x)
}
