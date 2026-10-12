#' @importFrom stats rbinom rpois rgamma rexp terms
NULL

#' Simulate Data from Ratio Models
#'
#' @description
#' Generate simulated datasets from prior or posterior predictive distributions.
#' Useful for prior predictive checks, simulation-based calibration, and
#' understanding model behavior.
#'
#' @param n Number of observations to simulate.
#' @param family A tulpaRatio family object specifying the distributions.
#' @param formula Optional formula for generating covariates and structure.
#'   If NULL, generates intercept-only data.
#' @param beta_num Numeric vector of true numerator coefficients.
#'   If NULL, drawn from prior.
#' @param beta_denom Numeric vector of true denominator coefficients.
#'   If NULL, drawn from prior.
#' @param sigma_re Standard deviation of random effects. Default 0.5.
#' @param phi_num Overdispersion for numerator (negbin/gamma families). Default 5.
#' @param phi_denom Overdispersion for denominator (negbin/gamma families). Default 5.
#' @param n_groups Number of random effect groups. Default 10.
#' @param n_per_group Observations per group. If NULL, groups are unbalanced.
#' @param spatial Optional spatial structure specification.
#' @param seed Random seed for reproducibility.
#'
#' @return A list with class `ratiod_simdata` containing:
#' \describe{
#'   \item{data}{Data frame with simulated observations}
#'   \item{true_params}{List of true parameter values used}
#'   \item{family}{Family specification}
#'   \item{n}{Number of observations}
#' }
#'
#' @details
#' The simulation generates data according to the ratio model structure:
#'
#' For **negbin_negbin**:
#' - `y_num ~ NegBin(mu_num, phi_num)`
#' - `y_denom ~ NegBin(mu_denom, phi_denom)`
#' - `log(mu_num) = X %*% beta_num + re`
#' - `log(mu_denom) = X %*% beta_denom + re` (shared RE)
#'
#' For **binomial**:
#' - `y_num ~ Binomial(y_denom, p)`
#' - `logit(p) = X %*% beta + re`
#'
#' For **poisson_gamma**:
#' - `y_num ~ Poisson(mu_num)`
#' - `y_denom ~ Gamma(shape, rate)` where `rate = shape / mu_denom`
#'
#' @examples
#' # Simulate CPUE-type data
#' sim <- sim_ratiod(
#'   n = 200,
#'   family = ratiod_poisson_gamma(),
#'   beta_num = c(2, 0.5),      # intercept + depth effect
#'   beta_denom = c(1, 0.2),    # effort model
#'   sigma_re = 0.3,
#'   n_groups = 20,
#'   seed = 123
#' )
#'
#' # View simulated data
#' head(sim$data)
#'
#' # Check true ratio distribution
#' true_ratio <- sim$data$y_num / sim$data$y_denom
#' hist(true_ratio, main = "Simulated ratios")
#'
#' # Simulate binomial proportion data
#' sim_binom <- sim_ratiod(
#'   n = 100,
#'   family = ratiod_binomial(),
#'   beta_num = c(-0.5, 1),     # logit-scale coefficients
#'   n_groups = 10,
#'   seed = 456
#' )
#'
#' \donttest{
#' # Fit model to simulated data
#' fit <- tratio(
#'   y_num | y_denom ~ x1 + (1 | group),
#'   data = sim$data,
#'   family = ratiod_poisson_gamma(),
#'   control = list(iter = 400, chains = 1, seed = 1)
#' )
#' # Compare estimated vs. true parameters
#' # summary(fit)
#' # sim$true_params
#' }
#'
#' @seealso [tratio()] for model fitting, [prior_predict()] for prior checks
#'
#' @export
sim_ratiod <- function(n = 100,
                       family = ratiod_negbin_negbin(),
                       formula = NULL,
                       beta_num = NULL,
                       beta_denom = NULL,
                       sigma_re = 0.5,
                       phi_num = 5,
                       phi_denom = 5,
                       n_groups = 10,
                       n_per_group = NULL,
                       spatial = NULL,
                       seed = NULL) {

  if (!is.null(seed)) set.seed(seed)

  family_name <- family$name

  # Generate group assignments
  if (!is.null(n_per_group)) {
    if (n_per_group * n_groups != n) {
      n <- n_per_group * n_groups
      message(sprintf("Adjusted n to %d (n_groups * n_per_group)", n))
    }
    group <- rep(seq_len(n_groups), each = n_per_group)
  } else {
    group <- sample(seq_len(n_groups), n, replace = TRUE)
  }

  # Generate covariates
  x1 <- rnorm(n)
  x2 <- rbinom(n, 1, 0.5)  # Binary covariate

  # Set default coefficients
  p_fixed <- 2  # intercept + x1

  if (is.null(beta_num)) {
    if (identical(family_name, "binomial_fixed")) {
      beta_num <- c(0, 0.5)  # logit scale
    } else {
      beta_num <- c(2, 0.3)  # log scale
    }
  }

  if (is.null(beta_denom)) {
    if (identical(family_name, "binomial_fixed")) {
      beta_denom <- NULL  # Not used for binomial
    } else {
      beta_denom <- c(1.5, 0.2)  # log scale
    }
  }

  # Build design matrix
  X <- cbind(1, x1)
  colnames(X) <- c("(Intercept)", "x1")

  # Generate random effects (shared between num and denom)
  re <- rnorm(n_groups, 0, sigma_re)

  # Compute linear predictors
  eta_num <- as.numeric(X %*% beta_num) + re[group]

  if (!is.null(beta_denom)) {
    eta_denom <- as.numeric(X %*% beta_denom) + re[group]
  }

  # Generate responses based on family
  if (family_name == "negbin_negbin") {
    mu_num <- exp(eta_num)
    mu_denom <- exp(eta_denom)

    y_num <- rnbinom(n, size = phi_num, mu = mu_num)
    y_denom <- rnbinom(n, size = phi_denom, mu = mu_denom)

    # Ensure denominator > 0 for ratio
    y_denom[y_denom == 0] <- 1

  } else if (identical(family_name, "binomial_fixed")) {
    p <- 1 / (1 + exp(-eta_num))

    # Generate trials (denominator)
    if (family_name == "binomial_fixed") {
      y_denom <- sample(10:50, n, replace = TRUE)
    } else {
      y_denom <- rpois(n, 30) + 10
    }

    y_num <- rbinom(n, size = y_denom, prob = p)

  } else if (family_name == "poisson_gamma") {
    mu_num <- exp(eta_num)
    mu_denom <- exp(eta_denom)

    y_num <- rpois(n, mu_num)

    # Gamma: shape = phi, rate = phi / mu
    y_denom <- rgamma(n, shape = phi_denom, rate = phi_denom / mu_denom)
    y_denom[y_denom < 0.01] <- 0.01  # Avoid division by zero

  } else if (family_name == "negbin_gamma") {
    mu_num <- exp(eta_num)
    mu_denom <- exp(eta_denom)

    y_num <- rnbinom(n, size = phi_num, mu = mu_num)

    # Gamma: shape = phi, rate = phi / mu
    y_denom <- rgamma(n, shape = phi_denom, rate = phi_denom / mu_denom)
    y_denom[y_denom < 0.01] <- 0.01  # Avoid division by zero

  } else if (family_name == "gamma_gamma") {
    mu_num <- exp(eta_num)
    mu_denom <- exp(eta_denom)

    y_num <- rgamma(n, shape = phi_num, rate = phi_num / mu_num)
    y_denom <- rgamma(n, shape = phi_denom, rate = phi_denom / mu_denom)

    y_num[y_num < 0.001] <- 0.001
    y_denom[y_denom < 0.001] <- 0.001

  } else if (family_name %in% c("lognormal_fixed", "lognormal_lognormal")) {
    sigma_obs <- 0.5  # Observation SD

    y_num <- exp(eta_num + rnorm(n, 0, sigma_obs))

    if (family_name == "lognormal_lognormal") {
      y_denom <- exp(eta_denom + rnorm(n, 0, sigma_obs))
    } else {
      y_denom <- rep(1, n)
    }

  } else {
    stop("Simulation not implemented for family: ", family_name, call. = FALSE)
  }

  # Create data frame
  data <- data.frame(
    y_num = y_num,
    y_denom = y_denom,
    x1 = x1,
    x2 = x2,
    group = factor(group)
  )

  # Store true parameters
  true_params <- list(
    beta_num = beta_num,
    beta_denom = beta_denom,
    sigma_re = sigma_re,
    re = re
  )

  if (family_name %in% c("negbin_negbin", "negbin_gamma", "poisson_gamma", "gamma_gamma")) {
    true_params$phi_num <- phi_num
    true_params$phi_denom <- phi_denom
  }

  structure(
    list(
      data = data,
      true_params = true_params,
      family = family,
      n = n,
      n_groups = n_groups,
      seed = seed
    ),
    class = "ratiod_simdata"
  )
}


#' @rdname sim_ratiod
#' @param x A ratiod_simdata object
#' @param ... Ignored
#'
#' @export
print.ratiod_simdata <- function(x, ...) {
  cat("Simulated tulpaRatio data\n")
  cat("====================\n\n")

  cat("Family:", x$family$name, "\n")
  cat("Observations:", x$n, "\n")
  cat("Groups:", x$n_groups, "\n")
  if (!is.null(x$seed)) cat("Seed:", x$seed, "\n")
  cat("\n")

  cat("True parameters:\n")
  cat("  beta_num:", paste(round(x$true_params$beta_num, 3), collapse = ", "), "\n")
  if (!is.null(x$true_params$beta_denom)) {
    cat("  beta_denom:", paste(round(x$true_params$beta_denom, 3), collapse = ", "), "\n")
  }
  cat("  sigma_re:", round(x$true_params$sigma_re, 3), "\n")

  if (!is.null(x$true_params$phi_num)) {
    cat("  phi_num:", round(x$true_params$phi_num, 3), "\n")
    cat("  phi_denom:", round(x$true_params$phi_denom, 3), "\n")
  }

  cat("\nData columns:", paste(names(x$data), collapse = ", "), "\n")
  cat("\nUse x$data to access the simulated data frame\n")
  cat("Use x$true_params to access true parameter values\n")

  invisible(x)
}


#' Simulate Datasets for Simulation-Based Calibration
#'
#' @description
#' Generate multiple simulated datasets for simulation-based calibration (SBC).
#' Each dataset is generated with parameters drawn from the prior.
#'
#' @param n_sims Number of simulations to run.
#' @param n Number of observations per simulation.
#' @param family A tulpaRatio family object.
#' @param priors Prior specification for parameter generation.
#' @param n_groups Number of random effect groups.
#' @param seed Random seed.
#'
#' @return A list of `ratiod_simdata` objects.
#'
#' @examples
#' # Generate datasets for SBC (returns immediately)
#' sims <- sim_ratiod_sbc(
#'   n_sims = 3,
#'   n = 30,
#'   family = ratiod_negbin_negbin(),
#'   seed = 42
#' )
#' print(sims)
#'
#' \donttest{
#' # Fit model to each simulation
#' # ranks <- lapply(sims, function(sim) {
#' #   fit <- tratio(y_num | y_denom ~ x1 + (1 | group),
#' #                data = sim$data,
#' #                family = ratiod_negbin_negbin(),
#' #                control = list(iter = 400, chains = 1, seed = 1))
#' #   sum(as.matrix(fit$draws)[, "beta_num[1]"] < sim$true_params$beta_num[1])
#' # })
#' }
#'
#' @seealso [sim_ratiod()] for single dataset simulation
#'
#' @export
sim_ratiod_sbc <- function(n_sims = 100,
                           n = 100,
                           family = ratiod_negbin_negbin(),
                           priors = NULL,
                           n_groups = 10,
                           seed = NULL) {

  if (!is.null(seed)) set.seed(seed)

  priors <- fill_priors(priors, get_hmc_model_type(family))

  family_name <- family$name

  sims <- vector("list", n_sims)

  for (i in seq_len(n_sims)) {
    # Draw parameters from the priors the fit uses
    beta_num <- pp_draw_prior(priors$beta, 2)
    beta_denom <- pp_draw_prior(priors$beta, 2)
    sigma_re <- pp_draw_prior(priors$sigma, 1)

    # Overdispersion
    if (family_name %in% c("negbin_negbin", "negbin_gamma", "poisson_gamma", "gamma_gamma")) {
      phi_num <- pp_draw_prior(priors$phi, 1)
      phi_denom <- pp_draw_prior(priors$phi, 1)
    } else {
      phi_num <- 5
      phi_denom <- 5
    }

    # Simulate data with these parameters
    sims[[i]] <- sim_ratiod(
      n = n,
      family = family,
      beta_num = beta_num,
      beta_denom = beta_denom,
      sigma_re = sigma_re,
      phi_num = phi_num,
      phi_denom = phi_denom,
      n_groups = n_groups
    )
  }

  structure(
    sims,
    n_sims = n_sims,
    family = family,
    class = c("ratiod_sbc_sims", "list")
  )
}


#' @rdname sim_ratiod_sbc
#' @param x A ratiod_sbc_sims object
#' @param ... Ignored
#'
#' @export
print.ratiod_sbc_sims <- function(x, ...) {
  cat("tulpaRatio SBC simulations\n")
  cat("=====================\n\n")

  cat("Simulations:", attr(x, "n_sims"), "\n")
  cat("Family:", attr(x, "family")$name, "\n")
  cat("Observations per sim:", x[[1]]$n, "\n")
  cat("Groups per sim:", x[[1]]$n_groups, "\n")

  cat("\nUse lapply() to fit models to each simulation\n")

  invisible(x)
}
