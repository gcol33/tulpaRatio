#' Prior Specification for Ratio Models
#'
#' @description
#' Specify priors for model parameters. Supports both PC (penalized complexity)
#' priors for variance components and standard distributions for other parameters.
#'
#' @param beta Prior for fixed effects. Default: `prior_normal(0, 10)`.
#' @param sigma Prior for random effect SDs. Default: `prior_half_cauchy(2.5)`
#'   (the Laplace backend: `prior_pc(U = 1, alpha = 0.01)`).
#' @param phi Prior for the dispersion parameter. Default: a Gamma prior set
#'   by family: `prior_gamma(2, 2)` for lognormal, `prior_gamma(2, 0.5)` for
#'   gamma_gamma and poisson_gamma, `prior_gamma(2, 0.1)` otherwise.
#' @param rho_temporal Prior for a temporal autocorrelation, placed on
#'   `(rho + 1) / 2` so `rho` lives on (-1, 1). Default: `prior_beta(2, 2)`,
#'   symmetric about `rho = 0`. Applies to `temporal_ar1()`, a
#'   `spatiotemporal()` interaction's AR1 time margin, the TVC arm and the
#'   multi-scale short-term arm; each block's own `rho_prior` overrides it.
#' @param rho_spatial Prior for spatial proportion (BYM2). Default:
#'   `prior_beta(1, 1)` (uniform).
#'
#' @return A `ratiod_priors` object. A component left `NULL` takes its
#'   default, the same for every sampler and for [prior_predict()].
#'
#' @details
#' The MCMC samplers (`mode = "hmc"`, `"ess"`, `"pg"`, `"gibbs"`, `"vi"`,
#' `"sghmc"`, `"sgld"`) evaluate a Normal prior with mean 0 on fixed effects,
#' a Half-Cauchy prior on random-effect SDs and a Gamma prior on the dispersion
#' parameter; a component specified with another family is refused when the
#' model is fitted. The Laplace backend takes `prior_pc()` or
#' `prior_half_normal()` on `sigma`. The constructors `prior_normal()`,
#' `prior_half_normal()`, `prior_half_cauchy()`, `prior_gamma()`,
#' `prior_beta()`, `prior_exponential()` and `prior_pc()` build each
#' component; [prior_predict()] draws from any of them.
#'
#' @references
#' Simpson, D., Rue, H., Riebler, A., Martins, T. G., & Sorbye, S. H. (2017).
#' Penalising model component complexity: A principled, practical approach to
#' constructing priors. Statistical Science, 32(1), 1-28.
#'
#' @examples
#' # Default priors
#' ratiod_priors()
#'
#' # Custom fixed effect prior
#' ratiod_priors(beta = prior_normal(0, 1))
#'
#' # Tighter random effect prior
#' ratiod_priors(sigma = prior_half_cauchy(0.5))
#'
#' # PC prior on the random effect SD (Laplace backend, prior_predict())
#' ratiod_priors(sigma = prior_pc(U = 0.5, alpha = 0.01))
#'
#' # Informative prior for temporal correlation
#' ratiod_priors(rho_temporal = prior_beta(5, 2))  # Prior mode at ~0.8
#'
#' @export
ratiod_priors <- function(
    beta = NULL,
    sigma = NULL,
    phi = NULL,
    rho_temporal = NULL,
    rho_spatial = NULL
) {

  # Validate each prior given; NULL keeps the default
  if (!is.null(beta)) validate_prior(beta, "beta")
  if (!is.null(sigma)) validate_prior(sigma, "sigma")
  if (!is.null(phi)) validate_prior(phi, "phi")
  if (!is.null(rho_temporal)) validate_prior(rho_temporal, "rho_temporal")
  if (!is.null(rho_spatial)) validate_prior(rho_spatial, "rho_spatial")

  structure(
    list(
      beta = beta,
      sigma = sigma,
      phi = phi,
      rho_temporal = rho_temporal,
      rho_spatial = rho_spatial
    ),
    class = "ratiod_priors"
  )
}


#' Validate a prior specification
#'
#' @param prior A prior object
#' @param name Parameter name for error messages
#' @keywords internal
validate_prior <- function(prior, name) {
  if (!inherits(prior, "ratiod_prior")) {
    stop(sprintf("'%s' must be a prior object (use prior_*() functions)", name),
         call. = FALSE)
  }
}


#' Normal Prior
#'
#' @description
#' Specify a normal prior for a parameter.
#'
#' @param mean Prior mean. Default 0.
#' @param sd Prior standard deviation. Must be positive.
#'
#' @return A `ratiod_prior` object
#'
#' @examples
#' prior_normal(0, 2.5)
#' prior_normal(0, 1)
#'
#' @export
prior_normal <- function(mean = 0, sd = 2.5) {
  if (sd <= 0) stop("sd must be positive", call. = FALSE)
  structure(
    list(
      distribution = "normal",
      mean = mean,
      sd = sd
    ),
    class = "ratiod_prior"
  )
}


#' Half-Normal Prior
#'
#' @description
#' Specify a half-normal (truncated at 0) prior for a positive parameter.
#'
#' @param sd Scale parameter. Must be positive.
#'
#' @return A `ratiod_prior` object
#'
#' @examples
#' prior_half_normal(1)
#'
#' @export
prior_half_normal <- function(sd = 1) {
  if (sd <= 0) stop("sd must be positive", call. = FALSE)
  structure(
    list(
      distribution = "half_normal",
      sd = sd
    ),
    class = "ratiod_prior"
  )
}


#' Half-Cauchy Prior
#'
#' @description
#' Specify a half-Cauchy prior for a positive parameter.
#' Has heavier tails than half-normal, often used for variance parameters.
#'
#' @param scale Scale parameter. Must be positive.
#'
#' @return A `ratiod_prior` object
#'
#' @examples
#' prior_half_cauchy(2.5)
#'
#' @export
prior_half_cauchy <- function(scale = 2.5) {
  if (scale <= 0) stop("scale must be positive", call. = FALSE)
  structure(
    list(
      distribution = "half_cauchy",
      scale = scale
    ),
    class = "ratiod_prior"
  )
}


#' Gamma Prior
#'
#' @description
#' Specify a gamma prior for a positive parameter.
#'
#' @param shape Shape parameter (alpha). Must be positive.
#' @param rate Rate parameter (beta). Must be positive.
#'
#' @return A `ratiod_prior` object
#'
#' @details
#' Mean = shape/rate, Variance = shape/rate^2.
#'
#' @examples
#' prior_gamma(2, 0.1)  # Mean = 20, weakly informative
#' prior_gamma(1, 1)    # Exponential(1)
#'
#' @export
prior_gamma <- function(shape = 2, rate = 0.1) {
  if (shape <= 0) stop("shape must be positive", call. = FALSE)
  if (rate <= 0) stop("rate must be positive", call. = FALSE)
  structure(
    list(
      distribution = "gamma",
      shape = shape,
      rate = rate
    ),
    class = "ratiod_prior"
  )
}


#' Exponential Prior
#'
#' @description
#' Specify an exponential prior for a positive parameter.
#'
#' @param rate Rate parameter (lambda). Must be positive.
#'
#' @return A `ratiod_prior` object
#'
#' @examples
#' prior_exponential(1)
#'
#' @export
prior_exponential <- function(rate = 1) {
  if (rate <= 0) stop("rate must be positive", call. = FALSE)
  structure(
    list(
      distribution = "exponential",
      rate = rate
    ),
    class = "ratiod_prior"
  )
}


#' Beta Prior
#'
#' @description
#' Specify a beta prior for a parameter bounded in (0, 1).
#'
#' @param alpha First shape parameter. Must be positive.
#' @param beta Second shape parameter. Must be positive.
#'
#' @return A `ratiod_prior` object
#'
#' @details
#' - alpha = beta = 1: Uniform(0, 1)
#' - alpha = beta = 2: Symmetric, peaked at 0.5
#' - alpha > beta: Skewed toward 1
#' - alpha < beta: Skewed toward 0
#'
#' @examples
#' prior_beta(1, 1)   # Uniform
#' prior_beta(2, 2)   # Symmetric, centered at 0.5
#' prior_beta(5, 2)   # Skewed toward 1 (for high autocorrelation)
#'
#' @export
prior_beta <- function(alpha = 1, beta = 1) {
  if (alpha <= 0) stop("alpha must be positive", call. = FALSE)
  if (beta <= 0) stop("beta must be positive", call. = FALSE)
  structure(
    list(
      distribution = "beta",
      alpha = alpha,
      beta = beta
    ),
    class = "ratiod_prior"
  )
}


#' Validate a Beta prior on an AR1 correlation
#'
#' The correlation is sampled as `logit((rho + 1) / 2)`, so its prior is a Beta
#' on `(rho + 1) / 2`. `NULL` defers to the model-wide `rho_temporal` default.
#'
#' @param rho_prior A `prior_beta()` object, or `NULL`.
#' @param arg Name to report in an error message.
#'
#' @return `rho_prior`, unchanged.
#' @keywords internal
validate_rho_prior <- function(rho_prior, arg = "rho_prior") {
  if (is.null(rho_prior)) return(NULL)
  if (!inherits(rho_prior, "ratiod_prior") ||
      !identical(rho_prior$distribution, "beta")) {
    stop("`", arg, "` must be a Beta prior from prior_beta(alpha, beta).",
         call. = FALSE)
  }
  rho_prior
}


#' Beta anchors for an AR1 correlation
#'
#' Resolves the block's own `rho_prior` against the model-wide
#' `ratiod_priors(rho_temporal = )`, falling back to Beta(2, 2).
#'
#' @param block_prior The block's `rho_prior`, or `NULL`.
#' @param default_prior The model-wide prior, or `NULL`.
#'
#' @return Numeric vector with elements `a` and `b`.
#' @keywords internal
rho_prior_anchors <- function(block_prior, default_prior = NULL) {
  p <- validate_rho_prior(block_prior) %||%
       validate_rho_prior(default_prior, "rho_temporal") %||%
       prior_beta(2, 2)
  c(a = p$alpha, b = p$beta)
}


#' Penalized Complexity (PC) Prior
#'
#' @description
#' Specify a PC prior for a positive parameter (typically a standard deviation).
#' PC priors shrink toward simpler models by penalizing deviation from a base model.
#'
#' @param U Upper bound. P(x > U) = alpha.
#' @param alpha Tail probability. Default 0.01.
#'
#' @return A `ratiod_prior` object
#'
#' @details
#' The PC prior is specified via: P(sigma > U) = alpha
#' - U is the upper bound you consider "large"
#' - alpha is the probability of exceeding it
#'
#' This implies an exponential prior with rate = -log(alpha) / U.
#'
#' @references
#' Simpson, D., Rue, H., Riebler, A., Martins, T. G., & Sorbye, S. H. (2017).
#' Penalising model component complexity: A principled, practical approach to
#' constructing priors. Statistical Science, 32(1), 1-28.
#'
#' @examples
#' prior_pc(U = 1, alpha = 0.01)    # P(sigma > 1) = 0.01
#' prior_pc(U = 0.5, alpha = 0.05)  # Tighter, P(sigma > 0.5) = 0.05
#'
#' @export
prior_pc <- function(U = 1, alpha = 0.01) {
  if (U <= 0) stop("U must be positive", call. = FALSE)
  if (alpha <= 0 || alpha >= 1) stop("alpha must be in (0, 1)", call. = FALSE)

  rate <- -log(alpha) / U

  structure(
    list(
      distribution = "pc",
      U = U,
      alpha = alpha,
      rate = rate
    ),
    class = "ratiod_prior"
  )
}


#' Legacy prior specification (deprecated)
#'
#' @description
#' Default prior of a model component
#'
#' The prior every sampler, [prior_predict()] and [sim_ratiod_sbc()] applies
#' when [ratiod_priors()] leaves the component `NULL`. The dispersion default
#' depends on what the parameter is in the family: a log-scale SD (lognormal),
#' a Gamma shape (gamma_gamma, poisson_gamma) or an overdispersion / precision
#' (the negative binomial and beta-binomial families).
#'
#' @param component One of `"beta"`, `"sigma"`, `"phi"`, `"rho_temporal"`,
#'   `"rho_spatial"`.
#' @param model_type The family's model type, as `get_hmc_model_type()`
#'   returns it.
#' @return A `ratiod_prior` object.
#' @keywords internal
default_prior <- function(component, model_type = NULL) {
  switch(
    component,
    beta = prior_normal(0, 10),
    sigma = prior_half_cauchy(2.5),
    phi = switch(
      model_type %||% "",
      lognormal = prior_gamma(2, 2),
      gamma_gamma = ,
      poisson_gamma = prior_gamma(2, 0.5),
      prior_gamma(2, 0.1)
    ),
    rho_temporal = prior_beta(2, 2),
    rho_spatial = prior_beta(1, 1),
    stop(sprintf("Unknown prior component '%s'", component), call. = FALSE)
  )
}

#' A prior specification with every component filled
#'
#' @param priors A `ratiod_priors` object or `NULL`.
#' @param model_type The family's model type.
#' @return A `ratiod_priors` object whose `NULL` components hold
#'   `default_prior()`.
#' @keywords internal
fill_priors <- function(priors, model_type = NULL) {
  priors <- priors %||% ratiod_priors()
  for (nm in c("beta", "sigma", "phi", "rho_temporal", "rho_spatial")) {
    if (is.null(priors[[nm]])) priors[[nm]] <- default_prior(nm, model_type)
  }
  priors
}

#' Prior parameters the samplers evaluate
#'
#' Translates a [ratiod_priors()] specification into the scalars the MCMC
#' samplers take: the Normal SD on fixed effects (mean 0), the Half-Cauchy
#' scale on random-effect SDs and the Gamma shape and rate on the dispersion
#' parameter, plus the fixed precision and zero-inflation priors. A component
#' in a family the samplers do not implement is refused, since fitting under
#' another prior than the one asked for would go unnoticed.
#'
#' @param priors A `ratiod_priors` object or `NULL`.
#' @param model_type The family's model type.
#' @return A named list of prior parameters.
#' @keywords internal
sampler_priors <- function(priors, model_type) {
  p <- fill_priors(priors, model_type)
  refuse <- function(component, accepted) {
    stop(sprintf(paste0("The MCMC samplers take %s on `%s`; the %s prior given ",
                        "is not one. Use it with mode = \"laplace\" or ",
                        "prior_predict(), or choose %s."),
                 accepted, component, p[[component]]$distribution, accepted),
         call. = FALSE)
  }
  if (!identical(p$beta$distribution, "normal") || p$beta$mean != 0) {
    refuse("beta", "prior_normal(0, sd)")
  }
  if (!identical(p$sigma$distribution, "half_cauchy")) {
    refuse("sigma", "prior_half_cauchy(scale)")
  }
  if (!identical(p$phi$distribution, "gamma")) {
    refuse("phi", "prior_gamma(shape, rate)")
  }
  if (!identical(p$rho_spatial$distribution, "beta") ||
      p$rho_spatial$alpha != 1 || p$rho_spatial$beta != 1) {
    refuse("rho_spatial", "prior_beta(1, 1)")
  }
  list(
    sigma_beta = p$beta$sd,
    sigma_re_scale = p$sigma$scale,
    phi_shape = p$phi$shape,
    phi_rate = p$phi$rate,
    rho_temporal = p$rho_temporal,
    tau_spatial_shape = 1.0,
    tau_spatial_rate = 0.01,
    tau_temporal_shape = 1.0,
    tau_temporal_rate = 0.01,
    zi_prior_sd = 10.0
  )
}

#' @rdname ratiod_priors
#' @param x A `ratiod_priors` object from [ratiod_priors()], or a single
#'   `ratiod_prior` from a `prior_*()` constructor.
#' @param ... Ignored
#'
#' @export
print.ratiod_priors <- function(x, ...) {
  cat("tulpaRatio prior specification\n")
  cat("=========================\n\n")

  cat("Fixed effects (beta):\n")
  print_prior(x$beta, indent = "  ")
  cat("\n")

  cat("Random effect SD (sigma):\n")
  print_prior(x$sigma, indent = "  ")
  cat("\n")

  cat("Overdispersion (phi):\n")
  print_prior(x$phi, indent = "  ")
  cat("\n")

  cat("Temporal autocorrelation (rho_temporal):\n")
  print_prior(x$rho_temporal, indent = "  ")
  cat("\n")

  cat("Spatial proportion (rho_spatial):\n")
  print_prior(x$rho_spatial, indent = "  ")
  cat("\nComponents marked default take default_prior() when fitted.\n")

  invisible(x)
}


#' @rdname ratiod_priors
#'
#' @export
print.ratiod_prior <- function(x, ...) {
  print_prior(x, indent = "")
  invisible(x)
}


#' Format a single prior for printing
#'
#' @param prior A ratiod_prior object
#' @param indent Indentation string
#' @keywords internal
print_prior <- function(prior, indent = "") {
  if (is.null(prior)) {
    cat(sprintf("%sdefault\n", indent))
    return(invisible(NULL))
  }

  dist <- prior$distribution

  if (dist == "normal") {
    cat(sprintf("%sNormal(%.2f, %.2f)\n", indent, prior$mean, prior$sd))
  } else if (dist == "half_normal") {
    cat(sprintf("%sHalf-Normal(%.2f)\n", indent, prior$sd))
  } else if (dist == "half_cauchy") {
    cat(sprintf("%sHalf-Cauchy(%.2f)\n", indent, prior$scale))
  } else if (dist == "gamma") {
    cat(sprintf("%sGamma(%.2f, %.2f)  [mean = %.2f]\n",
                indent, prior$shape, prior$rate, prior$shape / prior$rate))
  } else if (dist == "exponential") {
    cat(sprintf("%sExponential(%.2f)  [mean = %.2f]\n",
                indent, prior$rate, 1 / prior$rate))
  } else if (dist == "beta") {
    cat(sprintf("%sBeta(%.2f, %.2f)  [mean = %.2f]\n",
                indent, prior$alpha, prior$beta,
                prior$alpha / (prior$alpha + prior$beta)))
  } else if (dist == "pc") {
    cat(sprintf("%sPC prior: P(x > %.2f) = %.3f\n", indent, prior$U, prior$alpha))
    cat(sprintf("%s  => Exponential(%.3f)\n", indent, prior$rate))
  } else {
    cat(sprintf("%s%s\n", indent, dist))
  }
}


#' Default Priors for a Model Family
#'
#' @description
#' Display the priors a fit uses for every component [ratiod_priors()] leaves
#' unspecified, as a starting point for customization.
#'
#' @param family A tulpaRatio family object (e.g., `ratiod_negbin_negbin()`).
#'   If NULL (default), shows the dispersion default of every family.
#' @param spatial Logical; if TRUE, include the BYM2 mixing prior.
#' @param temporal Logical; if TRUE, include the AR1 correlation prior.
#'
#' @return Invisibly, a `ratiod_priors` object holding the defaults (for
#'   `family = NULL`, with the negative binomial dispersion default).
#'
#' @details
#' - **Fixed effects (beta)**: Normal(0, 10) on the link scale.
#' - **Random effect SD (sigma)**: Half-Cauchy(2.5); the Laplace backend uses
#'   a PC prior with P(sigma > 1) = 0.01.
#' - **Dispersion (phi)**: Gamma(2, 2) for the lognormal log-scale SD,
#'   Gamma(2, 0.5) for a Gamma shape (gamma_gamma, poisson_gamma), Gamma(2, 0.1)
#'   for a negative binomial size or beta-binomial precision.
#' - **Temporal correlation (rho)**: Beta(2, 2) on `(rho + 1) / 2`, symmetric
#'   about `rho = 0`.
#' - **Spatial mixing (rho_spatial)**: Beta(1, 1), uniform on (0, 1).
#'
#' @examples
#' priors_default(ratiod_negbin_negbin())
#' priors_default(ratiod_poisson_gamma(), spatial = TRUE, temporal = TRUE)
#' priors_default()
#'
#' @seealso [ratiod_priors()] for creating custom priors
#'
#' @export
priors_default <- function(family = NULL, spatial = FALSE, temporal = FALSE) {
  model_type <- NULL
  if (!is.null(family)) {
    if (!inherits(family, "ratiod_family")) {
      stop("`family` must be a ratiod_family object", call. = FALSE)
    }
    model_type <- get_hmc_model_type(family)
  }
  defaults <- fill_priors(NULL, model_type %||% "negbin_negbin")

  title <- if (is.null(family)) "Default priors" else
    sprintf("Default priors for the %s family", family$name)
  cat(title, "\n", strrep("=", nchar(title)), "\n\n", sep = "")

  cat("Fixed effects (beta):\n")
  print_prior(defaults$beta, indent = "  ")
  cat("Random effect SD (sigma):\n")
  print_prior(defaults$sigma, indent = "  ")
  cat("  (Laplace backend: PC prior, P(sigma > 1) = 0.01)\n")

  cat("Dispersion (phi):\n")
  if (is.null(family)) {
    groups <- list(
      "lognormal" = "lognormal",
      "gamma_gamma, poisson_gamma" = "poisson_gamma",
      "negbin_negbin, negbin_gamma, beta_binomial" = "negbin_negbin")
    for (nm in names(groups)) {
      cat(sprintf("  %s: ", nm))
      print_prior(default_prior("phi", groups[[nm]]))
    }
  } else if (is.null(MODEL_DISPERSION_NAMES[[model_type]])) {
    cat(sprintf("  none (%s)\n", family$name))
  } else {
    print_prior(defaults$phi, indent = "  ")
  }

  if (temporal) {
    cat("Temporal autocorrelation (rho_temporal):\n")
    print_prior(defaults$rho_temporal, indent = "  ")
  }
  if (spatial) {
    cat("Spatial mixing (rho_spatial):\n")
    print_prior(defaults$rho_spatial, indent = "  ")
  }

  invisible(defaults)
}
