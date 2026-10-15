#' Posterior Predictive Checks for Ratio Models
#'
#' @description
#' Visual and numerical checks comparing observed data to posterior
#' predictive distributions. Essential for assessing model fit.
#'
#' @examples
#' \donttest{
#' sim <- sim_ratiod(n = 100, family = ratiod_poisson_gamma(), seed = 1)
#' fit <- tratio(y_num | y_denom ~ x1 + (1 | group), data = as.data.frame(sim),
#'               family = ratiod_poisson_gamma(),
#'               control = list(iter = 1000, chains = 2, seed = 1))
#' if (requireNamespace("ggplot2", quietly = TRUE)) {
#'   pp_check(fit, type = "dens_overlay")
#' }
#' loo(fit)
#' }
#' @name ratiod_validate
NULL

#' Posterior Predictive Check
#'
#' @description
#' Generate posterior predictive checks for a fitted ratio model.
#' Compares observed data to replicated data from the posterior predictive
#' distribution.
#'
#' @param object A `ratiod_fit` object
#' @param type Type of check: "dens_overlay", "scatter", "intervals", "stat"
#' @param component Which component: "numerator", "denominator", or "both"
#' @param stat Function for "stat" type (default: mean)
#' @param ndraws Number of posterior draws to replicate (default: 50)
#' @param ... Additional arguments passed to bayesplot functions
#'
#' @return A ggplot object
#'
#' @examples
#' \donttest{
#' # Simulate data and fit model
#' set.seed(123)
#' n <- 50
#' df <- data.frame(
#'   count = rnbinom(n, size = 5, mu = 15),
#'   total = rnbinom(n, size = 8, mu = 100),
#'   x = rnorm(n),
#'   site = factor(rep(1:10, each = 5))
#' )
#' fit <- tratio(
#'   count | total ~ x + (1 | site),
#'   data = df,
#'   family = ratiod_negbin_negbin(),
#'   control = list(iter = 400, chains = 1, seed = 1)
#' )
#' # Density overlay of replicated against observed numerators
#' pp_check(fit, type = "dens_overlay")
#' }
#'
#' @export
#' @rdname pp_check
pp_check.ratiod_fit <- function(object, type = c("dens_overlay", "scatter", "intervals", "stat"),
                                component = c("numerator", "denominator", "both"),
                                stat = mean, ndraws = 50, ...) {

  type <- match.arg(type)
  component <- match.arg(component)

  # Replicate the response at each posterior draw
  replicates <- posterior_predict(object, ndraws = ndraws)
  y_rep_num <- replicates$y_num_rep
  y_rep_denom <- replicates$y_denom_rep
  y_num <- replicates$y_num
  y_denom <- replicates$y_denom

  # Select which to plot
  if (component == "numerator") {
    y <- y_num
    yrep <- y_rep_num
    title_suffix <- " (numerator)"
  } else if (component == "denominator") {
    y <- y_denom
    yrep <- y_rep_denom
    title_suffix <- " (denominator)"
  } else {
    # Both - create side-by-side plots
    if (!requireNamespace("patchwork", quietly = TRUE)) {
      message("Install 'patchwork' for side-by-side plots. Showing numerator only.")
      y <- y_num
      yrep <- y_rep_num
      title_suffix <- " (numerator)"
    } else {
      p1 <- pp_check_single(y_num, y_rep_num, type, stat, ndraws, " (numerator)", ...)
      p2 <- pp_check_single(y_denom, y_rep_denom, type, stat, ndraws, " (denominator)", ...)
      return(patchwork::wrap_plots(p1, p2, ncol = 2))
    }
  }

  pp_check_single(y, yrep, type, stat, ndraws, title_suffix, ...)
}

#' Single component pp_check
#'
#' @keywords internal
pp_check_single <- function(y, yrep, type, stat, ndraws, title_suffix, ...) {

  # Subsample draws for visualization
  if (nrow(yrep) > ndraws) {
    idx <- sample(nrow(yrep), ndraws)
    yrep <- yrep[idx, , drop = FALSE]
  }

  # Create plot based on type
  if (type == "dens_overlay") {
    p <- bayesplot::ppc_dens_overlay(y, yrep, ...)
  } else if (type == "scatter") {
    # Use mean of yrep for scatter
    yrep_mean <- colMeans(yrep)
    p <- bayesplot::ppc_scatter_avg(y, yrep, ...)
  } else if (type == "intervals") {
    p <- bayesplot::ppc_intervals(y, yrep, ...)
  } else if (type == "stat") {
    p <- bayesplot::ppc_stat(y, yrep, stat = stat, ...)
  }

  p + ggplot2::ggtitle(paste0("Posterior predictive check", title_suffix))
}


# ============================================================================
# Pointwise log-likelihood
# ----------------------------------------------------------------------------
# WAIC, PSIS-LOO, LOO stacking, and pseudo-BMA all consume the same
# [n_draws x n_obs] pointwise log-likelihood matrix, so it is reconstructed in
# one place from the posterior draws, the design matrices, and the family --
# no backend is required to store it. This is the tulpaRatio analogue of
# tulpaObs::.tobs_pointwise_loglik().
#
# The per-observation densities use the mean / shape / precision
# parameterizations of the C++ likelihood kernels (src/lik_specs/lik_helpers.h),
# evaluated with R's fully-normalized `d*()` functions so the resulting WAIC /
# elpd are on the standard scale.
#
# Fidelity note: the linear predictors are evaluated from the fixed-effect
# (population-level) coefficient draws -- the `beta_num[]` / `beta_denom[]`
# columns. Structured-term contributions (random effects, spatial, temporal
# fields) are not added to the predictor, so for models with those components
# the score is conditional on the fixed-effect predictor. This mirrors the
# documented behaviour of the sibling occupancy package.
# ============================================================================

# Model types this engine supports: every pair the samplers fit.
.ratiod_supported_model_types <- names(MODEL_TYPE_RESPONSES)

# Resolve a fit's model type regardless of which backend produced it (hmc /
# laplace store it in `.internal`, others at the top level; fall back to the
# family map).
.ratiod_model_type <- function(fit) {
  fit$model_type %||% fit$.internal$model_type %||% get_hmc_model_type(fit$family)
}

# Fixed-effect linear-predictor draws for both processes.
# Returns a list(eta_num, eta_denom) of [n_draws x n_obs] matrices; eta_denom is
# NULL for single-process families (binomial, beta_binomial).
.ratiod_linpred_fixed <- function(draws, hmc_data) {
  cn <- colnames(draws)

  pick <- function(prefix) {
    cols <- grep(paste0("^", prefix, "\\[[0-9]+\\]$"), cn)
    if (length(cols) == 0L) return(NULL)
    # Order by the bracketed index so eta is X %*% beta with matching columns.
    idx <- as.integer(sub(paste0("^", prefix, "\\[([0-9]+)\\]$"), "\\1", cn[cols]))
    draws[, cols[order(idx)], drop = FALSE]
  }

  beta_num <- pick("beta_num")
  if (is.null(beta_num)) {
    stop("Could not locate `beta_num[]` coefficient draws in the fit.",
         call. = FALSE)
  }
  if (ncol(beta_num) != hmc_data$p_num) {
    stop("Numerator coefficient draws (", ncol(beta_num),
         ") do not match the design matrix (", hmc_data$p_num, ").",
         call. = FALSE)
  }
  eta_num <- beta_num %*% t(hmc_data$X_num)

  eta_denom <- NULL
  if (hmc_data$p_denom > 0L) {
    beta_denom <- pick("beta_denom")
    if (is.null(beta_denom) || ncol(beta_denom) != hmc_data$p_denom) {
      stop("Could not locate matching `beta_denom[]` coefficient draws.",
           call. = FALSE)
    }
    eta_denom <- beta_denom %*% t(hmc_data$X_denom)
  }

  list(eta_num = eta_num, eta_denom = eta_denom)
}

# The two linear predictors each observation's likelihood is evaluated at,
# per draw: fixed effects plus every random and structured effect the fit
# carries. An HMC fit rebuilds them as fitted() does; the other backends that
# store a draw matrix carry fixed effects only, so a model with any other
# term is refused rather than scored on a likelihood missing it.
#' @keywords internal
.ratiod_pointwise_eta <- function(fit, draws, hmc_data) {
  if (has_hmc_layout(fit)) {
    eta <- hmc_eta_draws(hmc_fit_unpack(fit), hmc_fit_design(fit))
    if (length(eta$dropped)) {
      stop("Pointwise log-likelihood needs every term of the linear ",
           "predictor; ", paste(eta$dropped, collapse = ", "),
           " could not be reconstructed.", call. = FALSE)
    }
    return(eta[c("eta_num", "eta_denom")])
  }
  if (identical(fit$backend, "gibbs")) {
    # The sweep records the full linear predictor, spatial field included
    eta_num <- fit$.internal$eta_num
    return(list(eta_num = eta_num,
                eta_denom = fit$.internal$eta_denom %||%
                  matrix(0, nrow(eta_num), ncol(eta_num))))
  }
  structured <- (hmc_data$n_re_groups %||% 0L) > 0L ||
    !identical(fit$spatial$type %||% "none", "none") ||
    !is.null(fit$temporal) || !is.null(fit$latent) ||
    !is.null(fit$spatiotemporal) || !is.null(fit$svc) || !is.null(fit$tvc)
  if (structured) {
    stop("Pointwise log-likelihood on a '", fit$backend, "' fit is available ",
         "for fixed-effect models only; refit with mode = \"hmc\" for WAIC / ",
         "LOO with random or structured effects.", call. = FALSE)
  }
  .ratiod_linpred_fixed(draws, hmc_data)
}

# Per-draw dispersion / shape / scale parameters on the natural scale.
# Backends differ in both the column name and whether the parameter is stored
# on the log scale, so each quantity is resolved from an ordered list of
# candidate names; a `log_`-prefixed match is exponentiated.
.ratiod_dispersion <- function(draws, model_type) {
  cn <- colnames(draws)
  get_param <- function(natural, logged, what) {
    for (nm in natural) if (nm %in% cn) return(draws[, nm])
    for (nm in logged)  if (nm %in% cn) return(exp(draws[, nm]))
    stop("Could not locate the ", what, " parameter draws ",
         "(looked for: ", paste(c(natural, logged), collapse = ", "), ").",
         call. = FALSE)
  }

  switch(
    model_type,
    binomial = list(),
    beta_binomial = list(
      phi = get_param("phi", c("log_phi", "log_phi_num"), "beta-binomial precision")
    ),
    poisson_gamma = list(
      shape = get_param(c("shape", "shape_denom"),
                        c("log_shape", "log_phi_denom"), "Gamma shape")
    ),
    negbin_negbin = list(
      phi_num   = get_param("phi_num",   "log_phi_num",   "numerator overdispersion"),
      phi_denom = get_param("phi_denom", "log_phi_denom", "denominator overdispersion")
    ),
    negbin_gamma = list(
      phi_num   = get_param("phi_num",   "log_phi_num",   "numerator overdispersion"),
      phi_denom = get_param("phi_denom", "log_phi_denom", "Gamma shape")
    ),
    gamma_gamma = list(
      shape_num   = get_param("shape_num",   c("log_shape_num",   "log_phi_num"),   "numerator Gamma shape"),
      shape_denom = get_param("shape_denom", c("log_shape_denom", "log_phi_denom"), "denominator Gamma shape")
    ),
    lognormal = list(
      sigma_num   = get_param("sigma_num",   c("log_sigma_num",   "log_phi_num"),   "numerator log-scale SD"),
      sigma_denom = get_param("sigma_denom", c("log_sigma_denom", "log_phi_denom"), "denominator log-scale SD")
    ),
    poisson_fixed = list(),
    negbin_fixed = list(
      phi_num = get_param("phi_num", "log_phi_num", "numerator overdispersion")
    ),
    lognormal_fixed = list(
      sigma_num = get_param("sigma_num", c("log_sigma_num", "log_phi_num"),
                            "numerator log-scale SD")
    ),
    poisson_negbin = list(
      phi_denom = get_param("phi_denom", "log_phi_denom", "denominator overdispersion")
    ),
    stop("Pointwise log-likelihood is not implemented for model type '",
         model_type, "'.", call. = FALSE)
  )
}

# Pointwise log-likelihood matrix [n_draws x n_obs] for a fitted ratio model.
#' @keywords internal
.ratiod_pointwise_loglik <- function(fit) {
  draws <- fit$draws
  if (!is.matrix(draws)) {
    stop("Pointwise log-likelihood needs a posterior draw matrix; the '",
         fit$backend %||% "unknown", "' backend does not store one. WAIC / LOO ",
         "are supported on the hmc, ess, sghmc, and vi backends.",
         call. = FALSE)
  }

  model_type <- .ratiod_model_type(fit)
  if (!model_type %in% .ratiod_supported_model_types) {
    stop("Pointwise log-likelihood is not implemented for model type '",
         model_type, "'.", call. = FALSE)
  }

  hmc_data <- fit$.internal$hmc_data
  if (is.null(hmc_data)) {
    hmc_data <- suppressWarnings(
      prepare_hmc_data(fit$formula, fit$data, fit$family, model_type)
    )
  }

  lp  <- .ratiod_pointwise_eta(fit, draws, hmc_data)
  disp <- .ratiod_dispersion(draws, model_type)
  links <- family_links(fit$family, model_type)
  mu_num <- pp_linkinv(links$num)(lp$eta_num)
  mu_denom <- if (!links$probability && !links$exposure) {
    pp_linkinv(links$denom)(lp$eta_denom)
  }

  S <- nrow(lp$eta_num)
  N <- ncol(lp$eta_num)
  # Observed responses broadcast to [n_draws x n_obs] (constant down each column).
  ymat <- function(y) matrix(as.numeric(y), nrow = S, ncol = N, byrow = TRUE)

  y_num <- ymat(hmc_data$y_num)
  # A known exposure scales the numerator mean (and the log-normal median)
  exposure <- if (links$exposure) ymat(hmc_data$y_denom_cont)
  if (links$exposure) mu_num <- mu_num * exposure
  ll_num <- switch(
    model_type,
    binomial = stats::dbinom(y_num, size = ymat(hmc_data$y_denom),
                             prob = mu_num, log = TRUE),
    beta_binomial = {
      p <- mu_num
      n <- ymat(hmc_data$y_denom)
      alpha <- p * disp$phi          # disp$phi length S -> per-draw (per-row)
      beta  <- (1 - p) * disp$phi
      lchoose(n, y_num) + lbeta(y_num + alpha, n - y_num + beta) - lbeta(alpha, beta)
    },
    poisson_gamma = stats::dpois(y_num, lambda = mu_num, log = TRUE),
    negbin_negbin = ,
    negbin_gamma = stats::dnbinom(y_num, size = disp$phi_num,
                                  mu = mu_num, log = TRUE),
    gamma_gamma = stats::dgamma(ymat(hmc_data$y_num_cont), shape = disp$shape_num,
                                rate = disp$shape_num / mu_num, log = TRUE),
    lognormal = stats::dlnorm(ymat(hmc_data$y_num_cont), meanlog = lp$eta_num,
                              sdlog = disp$sigma_num, log = TRUE),
    poisson_fixed = ,
    poisson_negbin = stats::dpois(y_num, lambda = mu_num, log = TRUE),
    negbin_fixed = stats::dnbinom(y_num, size = disp$phi_num, mu = mu_num, log = TRUE),
    lognormal_fixed = stats::dlnorm(ymat(hmc_data$y_num_cont),
                                    meanlog = lp$eta_num + log(exposure),
                                    sdlog = disp$sigma_num, log = TRUE)
  )
  zi_type <- fit$zi$type %||% "none"
  if (!identical(zi_type, "none")) {
    ll_num <- .ratiod_inflated_num_loglik(fit, draws, mu_num, disp,
                                          hmc_data, zi_type, links)
  }

  ll_den <- switch(
    model_type,
    binomial = ,
    beta_binomial = 0,
    poisson_gamma = stats::dgamma(ymat(hmc_data$y_denom_cont), shape = disp$shape,
                                  rate = disp$shape / mu_denom, log = TRUE),
    negbin_negbin = stats::dnbinom(ymat(hmc_data$y_denom), size = disp$phi_denom,
                                   mu = mu_denom, log = TRUE),
    negbin_gamma = stats::dgamma(ymat(hmc_data$y_denom_cont), shape = disp$phi_denom,
                                 rate = disp$phi_denom / mu_denom, log = TRUE),
    gamma_gamma = stats::dgamma(ymat(hmc_data$y_denom_cont), shape = disp$shape_denom,
                                rate = disp$shape_denom / mu_denom, log = TRUE),
    lognormal = stats::dlnorm(ymat(hmc_data$y_denom_cont), meanlog = lp$eta_denom,
                              sdlog = disp$sigma_denom, log = TRUE),
    poisson_negbin = stats::dnbinom(ymat(hmc_data$y_denom), size = disp$phi_denom,
                                    mu = mu_denom, log = TRUE),
    poisson_fixed = ,
    negbin_fixed = ,
    lognormal_fixed = 0
  )

  matrix(ll_num + ll_den, S, N)
}

# Draws x observations linear predictor of an inflation component: the
# `prefix[j]` coefficient draws, in index order, times the design `X`.
#' @keywords internal
.ratiod_inflation_linpred <- function(draws, prefix, X) {
  cols <- grep(paste0("^", prefix, "\\[[0-9]+\\]$"), colnames(draws))
  idx <- as.integer(sub(paste0("^", prefix, "\\[([0-9]+)\\]$"), "\\1",
                        colnames(draws)[cols]))
  draws[, cols[order(idx)], drop = FALSE] %*% t(X)
}

# Numerator log-likelihood of a zero-inflated, one-inflated, zero-and-one-
# inflated or hurdle family, [n_draws x n_obs], as src/hmc_zi.h writes it:
# zi_* mix a structural zero (probability pi = linkinv(X_zi beta_zi)) with the
# count; hurdle_* send y = 0 to 1 - theta (theta = linkinv(X_zi beta_zi))
# and positive counts to theta times the zero-truncated count; oi_binomial
# mixes a structural y = n with probability linkinv(X_oi beta_oi); zoib draws
# a structural zero with pi_0, otherwise a structural y = n with pi_1,
# otherwise the binomial, which can itself give 0 or n. `mu_num` is the count
# mean or success probability, `links` the family's links.
#' @keywords internal
.ratiod_inflated_num_loglik <- function(fit, draws, mu_num, disp, hmc_data,
                                        zi_type, links) {
  S <- nrow(mu_num)
  N <- ncol(mu_num)
  ymat <- function(y) matrix(as.numeric(y), nrow = S, ncol = N, byrow = TRUE)
  linpred <- function(prefix, X) .ratiod_inflation_linpred(draws, prefix, X)
  log_add <- function(a, b) pmax(a, b) + log1p(exp(-abs(a - b)))

  y <- ymat(hmc_data$y_num)
  zero <- y == 0
  n <- ymat(hmc_data$y_denom)
  eta_zi <- if (!is.null(fit$zi$X_zi) && ncol(fit$zi$X_zi) > 0L) {
    linpred("beta_zi", fit$zi$X_zi)
  }
  log_pos <- function(x, link = links$zi) pp_log_prob(x, link, lower = TRUE)
  log_neg <- function(x, link = links$zi) pp_log_prob(x, link, lower = FALSE)

  if (zi_type %in% c("zi_poisson", "zi_negbin", "hurdle_poisson", "hurdle_negbin")) {
    mu <- mu_num
    if (zi_type %in% c("zi_poisson", "hurdle_poisson")) {
      lp_y <- stats::dpois(y, mu, log = TRUE)
      lp_0 <- -mu
    } else {
      size <- matrix(disp$phi_num, S, N)
      lp_y <- stats::dnbinom(y, size = size, mu = mu, log = TRUE)
      lp_0 <- stats::dnbinom(0, size = size, mu = mu, log = TRUE)
    }
    if (startsWith(zi_type, "zi_")) {
      return(ifelse(zero, log_add(log_pos(eta_zi), log_neg(eta_zi) + lp_0),
                    log_neg(eta_zi) + lp_y))
    }
    return(ifelse(zero, log_neg(eta_zi),
                  log_pos(eta_zi) + lp_y - log(-expm1(lp_0))))
  }

  p <- mu_num
  lp_y <- stats::dbinom(y, size = n, prob = p, log = TRUE)
  lp_0 <- n * log1p(-p)
  lp_n <- n * log(p)
  switch(
    zi_type,
    zi_binomial = ifelse(zero, log_add(log_pos(eta_zi), log_neg(eta_zi) + lp_0),
                         log_neg(eta_zi) + lp_y),
    hurdle_binomial = ifelse(zero, log_neg(eta_zi),
                             log_pos(eta_zi) + lp_y - log(-expm1(lp_0))),
    oi_binomial = {
      eta_oi <- linpred("beta_oi", fit$zi$X_oi)
      ifelse(y == n, log_add(log_pos(eta_oi, links$oi), log_neg(eta_oi, links$oi) + lp_n),
             log_neg(eta_oi, links$oi) + lp_y)
    },
    zoib = {
      eta_oi <- linpred("beta_oi", fit$zi$X_oi)
      both <- log_neg(eta_zi) + log_neg(eta_oi, links$oi)
      ifelse(zero, log_add(log_pos(eta_zi), both + lp_0),
             ifelse(y == n,
                    log_neg(eta_zi) + log_add(log_pos(eta_oi, links$oi), log_neg(eta_oi, links$oi) + lp_n),
                    both + lp_y))
    },
    stop(sprintf("Unknown inflation type '%s'", zi_type), call. = FALSE)
  )
}

#' Leave-One-Out Cross-Validation
#'
#' @description
#' Compute leave-one-out cross-validation using Pareto-smoothed
#' importance sampling (PSIS-LOO).
#'
#' @details
#' The pointwise log-likelihood is reconstructed from the posterior draws and
#' the model (see [ratiod_compare()]); it works on any fit produced by the
#' hmc, ess, sghmc, or vi backend, for every family including the zero-inflated,
#' one-inflated and hurdle ones. Each observation's likelihood is evaluated at
#' the full linear predictor: fixed, random and structured effects.
#'
#' @param x A `ratiod_fit` object
#' @param ... Additional arguments passed to loo::loo
#'
#' @return A loo object
#'
#' @importFrom loo loo
#' @examples
#' \donttest{
#' sim <- sim_ratiod(n = 100, family = ratiod_poisson_gamma(), seed = 1)
#' fit <- tratio(y_num | y_denom ~ x1 + (1 | group), data = as.data.frame(sim),
#'               family = ratiod_poisson_gamma(),
#'               control = list(iter = 1000, chains = 2, seed = 1))
#' loo(fit)
#' }
#'
#' @exportS3Method loo::loo
loo.ratiod_fit <- function(x, ...) {
  if (!requireNamespace("loo", quietly = TRUE)) {
    stop("Package 'loo' is required. Install with:\n",
         "  install.packages('loo')", call. = FALSE)
  }
  loo::loo(.ratiod_pointwise_loglik(x), ...)
}

#' Widely Applicable Information Criterion (WAIC)
#'
#' @description
#' Compute Widely Applicable Information Criterion (WAIC).
#'
#' @details
#' Uses the same reconstructed pointwise log-likelihood as [loo.ratiod_fit()];
#' supported on the hmc, ess, sghmc, and vi backends and conditional on the
#' fixed-effect linear predictor.
#'
#' @param x A `ratiod_fit` object
#' @param ... Additional arguments passed to loo::waic
#'
#' @return A waic object
#'
#' @importFrom loo waic
#' @examples
#' \donttest{
#' sim <- sim_ratiod(n = 100, family = ratiod_poisson_gamma(), seed = 1)
#' fit <- tratio(y_num | y_denom ~ x1 + (1 | group), data = as.data.frame(sim),
#'               family = ratiod_poisson_gamma(),
#'               control = list(iter = 1000, chains = 2, seed = 1))
#' # waic() warns when an observation's p_waic exceeds 0.4; loo() is then
#' # the estimate to report
#' waic(fit)
#' loo(fit)
#' }
#'
#' @exportS3Method loo::waic
waic.ratiod_fit <- function(x, ...) {
  if (!requireNamespace("loo", quietly = TRUE)) {
    stop("Package 'loo' is required. Install with:\n",
         "  install.packages('loo')", call. = FALSE)
  }
  loo::waic(.ratiod_pointwise_loglik(x), ...)
}

#' Compare Ratio Models
#'
#' @description
#' Compare multiple ratio models using LOO-CV or WAIC.
#'
#' @param ... Multiple `ratiod_fit` objects to compare
#' @param criterion Comparison criterion: "loo" (default) or "waic"
#'
#' @return A loo_compare object
#'
#' @examples
#' \donttest{
#' sim <- sim_ratiod(n = 100, family = ratiod_poisson_gamma(), seed = 1)
#' fit <- tratio(y_num | y_denom ~ x1 + (1 | group), data = as.data.frame(sim),
#'               family = ratiod_poisson_gamma(),
#'               control = list(iter = 1000, chains = 2, seed = 1))
#' fit0 <- tratio(y_num | y_denom ~ 1 + (1 | group), data = as.data.frame(sim),
#'                family = ratiod_poisson_gamma(),
#'                control = list(iter = 1000, chains = 2, seed = 1))
#' ratiod_compare(fit0, fit)
#' }
#'
#' @export
ratiod_compare <- function(..., criterion = c("loo", "waic")) {
  criterion <- match.arg(criterion)

  if (!requireNamespace("loo", quietly = TRUE)) {
    stop("Package 'loo' is required. Install with:\n",
         "  install.packages('loo')", call. = FALSE)
  }

  models <- list(...)

  if (length(models) < 2) {
    stop("At least two models required for comparison", call. = FALSE)
  }

  # Check all are ratiod_fit objects
  if (!all(vapply(models, inherits, logical(1), "ratiod_fit"))) {
    stop("All objects must be ratiod_fit models", call. = FALSE)
  }

  # Compute criterion for each model
  if (criterion == "loo") {
    results <- lapply(models, loo.ratiod_fit)
  } else {
    results <- lapply(models, waic.ratiod_fit)
  }

  # Get model names from call
  model_names <- as.character(substitute(list(...)))[-1]
  names(results) <- model_names

  # Compare
  loo::loo_compare(results)
}


#' Model Averaging for Ratio Models
#'
#' @description
#' Compute model-averaged predictions using stacking or pseudo-BMA weights.
#' Model weights are derived from LOO-CV or WAIC, and predictions are
#' combined accordingly.
#'
#' @param ... Multiple `ratiod_fit` objects to average
#' @param weights Method for computing weights: "loo" (stacking, default),
#'   "waic", "pbma" (pseudo-BMA), or "pbma+" (pseudo-BMA+ with Bayesian bootstrap).
#' @param newdata Optional new data for predictions. If NULL, uses fitted values.
#' @param type Type of prediction: "ratio" (default), "numerator", or "denominator".
#' @param summary Logical; if TRUE (default), return summary statistics. If FALSE,
#'   return full posterior draws.
#'
#' @return A `ratiod_average` object containing:
#' - `weights`: Model weights
#' - `predictions`: Model-averaged predictions
#' - `models`: List of model names
#'
#' @details
#' Model averaging accounts for model uncertainty by combining predictions
#' from multiple candidate models. The weighting methods are:
#'
#' - **stacking** (`weights = "loo"`): Optimal linear combination minimizing
#'   leave-one-out prediction error. Recommended default.
#' - **pseudo-BMA** (`weights = "pbma"`): Weights proportional to exp(-ELPD).
#'   Can be unstable with similar models.
#' - **pseudo-BMA+** (`weights = "pbma+"`): Bayesian bootstrap variant that
#'   accounts for estimation uncertainty.
#'
#' For ratio predictions, averaging is performed on the log scale to respect
#' the multiplicative nature of ratios.
#'
#' @examples
#' \donttest{
#' sim <- sim_ratiod(n = 100, family = ratiod_poisson_gamma(), seed = 1)
#' fit <- tratio(y_num | y_denom ~ x1 + (1 | group), data = as.data.frame(sim),
#'               family = ratiod_poisson_gamma(),
#'               control = list(iter = 1000, chains = 2, seed = 1))
#' fit0 <- tratio(y_num | y_denom ~ 1 + (1 | group), data = as.data.frame(sim),
#'                family = ratiod_poisson_gamma(),
#'                control = list(iter = 1000, chains = 2, seed = 1))
#' avg <- ratiod_average(fit0, fit)
#' avg
#' weights(avg)
#' }
#'
#' @seealso [ratiod_compare()] for model comparison, [loo::stacking_weights()]
#'
#' @export
ratiod_average <- function(..., weights = c("loo", "waic", "pbma", "pbma+"),
                          newdata = NULL, type = c("ratio", "numerator", "denominator"),
                          summary = TRUE) {

  weights_method <- match.arg(weights)
  type <- match.arg(type)

  if (!requireNamespace("loo", quietly = TRUE)) {
    stop("Package 'loo' is required. Install with:\n",
         "  install.packages('loo')", call. = FALSE)
  }

  models <- list(...)

  if (length(models) < 2) {
    stop("At least two models required for averaging", call. = FALSE)
  }

  # Check all are ratiod_fit objects
  if (!all(vapply(models, inherits, logical(1), "ratiod_fit"))) {
    stop("All objects must be ratiod_fit models", call. = FALSE)
  }

  # Get model names from call
  model_names <- as.character(substitute(list(...)))[-1]
  if (length(model_names) != length(models)) {
    model_names <- paste0("model", seq_along(models))
  }
  names(models) <- model_names

  # Check data compatibility (number of observations). `.internal$hmc_data` is
  # only populated by some backends, so fall back to the stored data frame.
  n_obs <- vapply(models, function(m) {
    as.integer(m$.internal$hmc_data$N %||% nrow(m$data))
  }, integer(1))
  if (length(unique(n_obs)) > 1) {
    stop("All models must be fitted to the same data", call. = FALSE)
  }

  # Compute weights
  model_weights <- compute_model_weights(models, weights_method)

  # Get predictions from each model
  predictions <- lapply(models, function(m) {
    get_predictions(m, newdata = newdata, type = type, summary = FALSE)
  })

  # Model-average the predictions
  avg_pred <- average_predictions(predictions, model_weights, type, summary)

  structure(
    list(
      weights = model_weights,
      predictions = avg_pred,
      models = model_names,
      type = type,
      weights_method = weights_method,
      n_models = length(models)
    ),
    class = "ratiod_average"
  )
}


#' Compute model weights
#'
#' @param models List of ratiod_fit objects
#' @param method Weighting method
#' @return Named numeric vector of weights
#' @keywords internal
compute_model_weights <- function(models, method) {

  # Compute LOO for each model
  loo_list <- lapply(models, function(m) {
    tryCatch(
      loo.ratiod_fit(m),
      error = function(e) {
        stop("Failed to compute LOO for model: ", e$message, call. = FALSE)
      }
    )
  })

  if (method == "loo") {
    # Stacking weights (optimal for prediction)
    lpd_matrix <- sapply(loo_list, function(x) x$pointwise[, "elpd_loo"])
    weights <- loo::stacking_weights(lpd_matrix)

  } else if (method == "waic") {
    # WAIC-based weights
    waic_list <- lapply(models, waic.ratiod_fit)
    lpd_matrix <- sapply(waic_list, function(x) x$pointwise[, "elpd_waic"])
    weights <- loo::stacking_weights(lpd_matrix)

  } else if (method == "pbma") {
    # Pseudo-BMA weights
    lpd_matrix <- sapply(loo_list, function(x) x$pointwise[, "elpd_loo"])
    weights <- loo::pseudobma_weights(lpd_matrix, BB = FALSE)

  } else if (method == "pbma+") {
    # Pseudo-BMA+ with Bayesian bootstrap
    lpd_matrix <- sapply(loo_list, function(x) x$pointwise[, "elpd_loo"])
    weights <- loo::pseudobma_weights(lpd_matrix, BB = TRUE)
  }

  weights <- as.numeric(weights)
  names(weights) <- names(models)
  weights
}


# Response-scale fitted draws [n_draws x n_obs] for one component, reconstructed
# from the fixed-effect linear predictors (the same machinery as the pointwise
# log-likelihood, so the two stay consistent). `type` is one of "ratio",
# "numerator", "denominator".
#' @keywords internal
.ratiod_component_draws <- function(fit, type) {
  model_type <- .ratiod_model_type(fit)
  hmc_data <- fit$.internal$hmc_data
  if (is.null(hmc_data)) {
    hmc_data <- suppressWarnings(
      prepare_hmc_data(fit$formula, fit$data, fit$family, model_type)
    )
  }
  lp <- .ratiod_linpred_fixed(fit$draws, hmc_data)
  draws <- hmc_response_draws(lp, family_links(fit$family, model_type),
                              exposure = hmc_exposure(hmc_data, model_type))

  switch(type, ratio = draws$ratio, numerator = draws$numerator,
         denominator = draws$denominator)
}

#' Get predictions from a model
#'
#' @param model A ratiod_fit object
#' @param newdata Optional new data
#' @param type Prediction type
#' @param summary Return summary or draws
#' @return Matrix of predictions (draws x observations)
#' @keywords internal
get_predictions <- function(model, newdata, type, summary) {

  if (!is.null(newdata)) {
    # Use predict method for new data
    pred <- predict(model, newdata = newdata, type = type, summary = FALSE)
    return(pred)
  }

  # Matrix-draw backends (hmc, ess, sghmc, vi): reconstruct fitted draws from
  # the fixed-effect linear predictors.
  if (is.matrix(model$draws)) {
    return(.ratiod_component_draws(model, type))
  }

  # Legacy list-shaped draws: read the stored component if present.
  if (type == "ratio") {
    if (!is.null(model$draws$ratio)) {
      pred <- model$draws$ratio
    } else if (!is.null(model$draws$eta_num) && !is.null(model$draws$eta_denom)) {
      pred <- exp(model$draws$eta_num - model$draws$eta_denom)
    } else {
      stop("Cannot extract ratio predictions from model", call. = FALSE)
    }
  } else if (type == "numerator") {
    pred <- model$draws$mu_num
    if (is.null(pred) && !is.null(model$draws$eta_num)) {
      pred <- exp(model$draws$eta_num)
    }
  } else {
    pred <- model$draws$mu_denom
    if (is.null(pred) && !is.null(model$draws$eta_denom)) {
      pred <- exp(model$draws$eta_denom)
    }
  }

  if (is.null(pred)) {
    stop("Cannot extract predictions for type '", type, "'", call. = FALSE)
  }

  pred
}


#' Average predictions across models
#'
#' @param predictions List of prediction matrices
#' @param weights Model weights
#' @param type Prediction type
#' @param summary Return summary or draws
#' @return Averaged predictions
#' @keywords internal
average_predictions <- function(predictions, weights, type, summary) {

  n_models <- length(predictions)
  n_draws <- nrow(predictions[[1]])
  n_obs <- ncol(predictions[[1]])

  # For ratios, average on log scale
  if (type == "ratio") {
    # Convert to log scale
    log_preds <- lapply(predictions, log)

    # Weighted average on log scale
    avg_log <- matrix(0, nrow = n_draws, ncol = n_obs)
    for (i in seq_len(n_models)) {
      avg_log <- avg_log + weights[i] * log_preds[[i]]
    }

    # Back-transform
    avg_pred <- exp(avg_log)

  } else {
    # Simple weighted average
    avg_pred <- matrix(0, nrow = n_draws, ncol = n_obs)
    for (i in seq_len(n_models)) {
      avg_pred <- avg_pred + weights[i] * predictions[[i]]
    }
  }

  if (summary) {
    # Return summary statistics
    data.frame(
      mean = colMeans(avg_pred),
      sd = apply(avg_pred, 2, sd),
      q2.5 = apply(avg_pred, 2, quantile, probs = 0.025),
      q25 = apply(avg_pred, 2, quantile, probs = 0.25),
      q50 = apply(avg_pred, 2, quantile, probs = 0.50),
      q75 = apply(avg_pred, 2, quantile, probs = 0.75),
      q97.5 = apply(avg_pred, 2, quantile, probs = 0.975)
    )
  } else {
    avg_pred
  }
}


#' @rdname ratiod_average
#' @param x A ratiod_average object
#' @param digits Number of digits for printing
#'
#' @export
print.ratiod_average <- function(x, digits = 3, ...) {
  cat("ratio model averaging\n")
  cat("=====================\n\n")
  cat("Method:", x$weights_method, "\n")
  cat("Models:", x$n_models, "\n")
  cat("Prediction type:", x$type, "\n\n")

  cat("Model weights:\n")
  for (i in seq_along(x$weights)) {
    cat(sprintf("  %s: %.3f\n", x$models[i], x$weights[i]))
  }

  cat("\n")
  if (is.data.frame(x$predictions)) {
    cat("Predictions summary (first 6 rows):\n")
    print(head(x$predictions), digits = digits)
    if (nrow(x$predictions) > 6) {
      cat(sprintf("  ... %d more rows\n", nrow(x$predictions) - 6))
    }
  } else {
    cat("Predictions:", nrow(x$predictions), "draws x",
        ncol(x$predictions), "observations\n")
  }

  invisible(x)
}


#' @rdname ratiod_average
#' @param object A ratiod_average object
#'
#' @return Predictions data frame or matrix
#'
#' @export
fitted.ratiod_average <- function(object, ...) {
  object$predictions
}


#' @rdname ratiod_average
#'
#' @return Named numeric vector of weights
#'
#' @importFrom stats weights
#' @export
weights.ratiod_average <- function(object, ...) {
  object$weights
}
