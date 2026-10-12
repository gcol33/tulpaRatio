#' Posterior Predictive Draws
#'
#' @description
#' Draw replicated numerator and denominator responses from the posterior
#' predictive distribution: at each posterior draw the two fitted means are
#' assembled from that draw's parameters, and a response is drawn from the
#' family at those means. The replicates are what [pp_check()] compares the
#' observed data against.
#'
#' @param object A `ratiod_fit` object
#' @param ndraws Number of posterior draws to replicate. `NULL` (the default)
#'   uses every draw; a smaller number takes a random subsample.
#' @param ... Unused.
#'
#' @return A list with `y_num_rep` and `y_denom_rep`, each a draws-by-observation
#'   matrix, plus `y_num` and `y_denom`, the observed responses. A family with a
#'   fixed denominator (the binomial arms, whose denominator is the trial count)
#'   reports the trials as `y_denom_rep`.
#'
#' @examples
#' \donttest{
#' set.seed(4)
#' n <- 40
#' df <- data.frame(
#'   count = rpois(n, 12),
#'   total = rpois(n, 80),
#'   x = rnorm(n)
#' )
#' fit <- tratio(count | total ~ x, data = df,
#'               family = ratiod_poisson_gamma(),
#'               control = list(iter = 400, chains = 1, seed = 1))
#' rep <- posterior_predict(fit, ndraws = 50)
#' dim(rep$y_num_rep)
#' }
#'
#' @seealso [pp_check()]
#'
#' @method posterior_predict ratiod_fit
#' @export
posterior_predict.ratiod_fit <- function(object, ndraws = NULL, ...) {
  family <- object$family
  num_dist <- family$numerator$distribution
  denom_dist <- family$denominator$distribution

  zi_type <- object$zi$type %||% "none"
  if (!identical(zi_type, "none")) {
    num_dist <- inflation_base_distribution(zi_type)
    draws_mat <- as.matrix(object$draws)
    links <- fit_links(object)
    pi_zi <- if (!is.null(object$zi$X_zi) && ncol(object$zi$X_zi) > 0L) {
      pp_linkinv(links$zi)(
        .ratiod_inflation_linpred(draws_mat, "beta_zi", object$zi$X_zi))
    }
    pi_oi <- if (zi_type %in% c("oi_binomial", "zoib")) {
      pp_linkinv(links$oi)(
        .ratiod_inflation_linpred(draws_mat, "beta_oi", object$zi$X_oi))
    }
  }

  fitted_vals <- compute_fitted_values(object)
  mu_num <- fitted_vals$numerator
  mu_denom <- fitted_vals$denominator
  n_draws <- nrow(mu_num)
  N <- ncol(mu_num)

  idx <- seq_len(n_draws)
  if (!is.null(ndraws) && ndraws < n_draws) {
    idx <- sort(sample.int(n_draws, ndraws))
  }

  observed <- observed_response(object)
  trials <- if (num_dist %in% c("binomial", "beta_binomial")) {
    observed$y_denom
  } else {
    NULL
  }

  disp_num <- dispersion_draws(object, "num")
  disp_denom <- dispersion_draws(object, "denom")
  denom_fixed <- identical(denom_dist, "fixed")

  y_num_rep <- matrix(NA_real_, length(idx), N)
  y_denom_rep <- matrix(NA_real_, length(idx), N)

  for (k in seq_along(idx)) {
    s <- idx[k]
    y_num_rep[k, ] <- pp_rdist(num_dist, mu_num[s, ], disp_num[s], trials)
    if (!identical(zi_type, "none")) {
      y_num_rep[k, ] <- pp_inflate(zi_type, y_num_rep[k, ], mu_num[s, ],
                                   disp_num[s], trials, pi_zi[s, ],
                                   if (!is.null(pi_oi)) pi_oi[s, ])
    }
    y_denom_rep[k, ] <- if (denom_fixed) {
      observed$y_denom
    } else {
      pp_rdist(denom_dist, mu_denom[s, ], disp_denom[s])
    }
  }

  list(
    y_num_rep = y_num_rep,
    y_denom_rep = y_denom_rep,
    y_num = observed$y_num,
    y_denom = observed$y_denom,
    draw_ids = idx
  )
}


#' Count or binomial distribution under an inflation component
#' @keywords internal
inflation_base_distribution <- function(zi_type) {
  switch(zi_type,
         zi_poisson = , hurdle_poisson = "poisson",
         zi_negbin = , hurdle_negbin = "neg_binomial_2",
         zi_binomial = , hurdle_binomial = , oi_binomial = , zoib = "binomial",
         stop(sprintf("Unknown inflation type '%s'", zi_type), call. = FALSE))
}

#' Replicate draws under an inflation component
#'
#' Applies the inflation of `src/hmc_zi.h` to draws `y` from the count or
#' binomial distribution: structural zeros (zi), a zero hurdle whose positive
#' outcomes come from the zero-truncated distribution, structural ones at the
#' trial count (oi), or both (zoib, whose binomial draws can also be 0 or the
#' trial count).
#'
#' @keywords internal
pp_inflate <- function(zi_type, y, mu, disp, trials, pi_zi, pi_oi) {
  N <- length(y)
  base <- inflation_base_distribution(zi_type)
  cdf <- function(q) switch(base,
    poisson = stats::ppois(q, mu),
    neg_binomial_2 = stats::pnbinom(q, size = disp, mu = mu),
    binomial = stats::pbinom(q, trials, mu))
  quant <- function(u) switch(base,
    poisson = stats::qpois(u, mu),
    neg_binomial_2 = stats::qnbinom(u, size = disp, mu = mu),
    binomial = stats::qbinom(u, trials, mu))
  between <- function(lo, hi) {
    a <- cdf(lo)
    quant(a + stats::runif(N) * (cdf(hi) - a))
  }
  switch(
    zi_type,
    zi_poisson = , zi_negbin = , zi_binomial = {
      y[stats::runif(N) < pi_zi] <- 0
      y
    },
    hurdle_poisson = , hurdle_negbin = , hurdle_binomial = {
      positive <- stats::runif(N) < pi_zi
      upper <- if (identical(base, "binomial")) trials else Inf
      ifelse(positive, between(0, upper), 0)
    },
    oi_binomial = {
      one <- stats::runif(N) < pi_oi
      y[one] <- trials[one]
      y
    },
    zoib = {
      zero <- stats::runif(N) < pi_zi
      one <- !zero & stats::runif(N) < pi_oi
      ifelse(zero, 0, ifelse(one, trials, y))
    }
  )
}


#' Observed responses of a fitted object
#' @keywords internal
observed_response <- function(object) {
  hmc_data <- object$.internal$hmc_data
  if (!is.null(hmc_data)) {
    y_num <- if (length(hmc_data$y_num_cont) && any(hmc_data$y_num_cont != 0)) {
      hmc_data$y_num_cont
    } else {
      hmc_data$y_num
    }
    y_denom <- if (length(hmc_data$y_denom_cont) &&
                   any(hmc_data$y_denom_cont != 0)) {
      hmc_data$y_denom_cont
    } else {
      hmc_data$y_denom
    }
    return(list(y_num = as.numeric(y_num), y_denom = as.numeric(y_denom)))
  }

  formula <- object$formula
  list(y_num = as.numeric(formula$numerator$response),
       y_denom = as.numeric(formula$denominator$response))
}


#' Dispersion draws of one arm
#'
#' The parameter is named for what it is in each family, so the draw is looked
#' up under the name that family reports it by. A family arm without a
#' dispersion parameter gets `NA`, which `pp_rdist()` does not read.
#'
#' @keywords internal
dispersion_draws <- function(object, arm = c("num", "denom")) {
  arm <- match.arg(arm)
  model_type <- fit_model_type(object)
  n_draws <- object$n_save %||% nrow(object$draws)

  names_for_type <- MODEL_DISPERSION_NAMES[[model_type]]
  name <- if (!is.null(names_for_type)) unname(names_for_type[arm]) else NULL
  if (is.null(name) || is.na(name)) return(rep(NA_real_, n_draws))

  draws <- object$draws
  if (is.null(draws) || !(name %in% colnames(draws))) {
    stop("The ", model_type, " family needs its `", name, "` draws to ",
         "replicate a response, and this ", object$backend,
         " fit does not report them.", call. = FALSE)
  }
  draws[, name]
}
