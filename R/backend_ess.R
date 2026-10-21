#' Elliptical Slice Sampling Backend for ratio Models
#'
#' @description
#' Elliptical Slice Sampling (ESS) backend for models with Gaussian priors.
#' ESS is particularly efficient for:
#' - Random effects (which have N(0, sigma^2) priors)
#' - Spatial effects (which have GMRF priors)
#' - GP effects (which have multivariate normal priors)
#'
#' @details
#' ESS has several advantages over HMC for models with Gaussian priors:
#' - **No tuning parameters**: Unlike HMC step size, ESS adapts automatically
#' - **Efficient for high-dimensional Gaussians**: Exploits prior structure
#' - **Exact transitions**: No discretization error (unlike leapfrog)
#'
#' ESS works by:
#' 1. Drawing a proposal from the prior (Gaussian)
#' 2. Defining an ellipse through current state and proposal
#' 3. Slice sampling along this ellipse
#'
#' Non-Gaussian parameters (variance components, correlations) are updated
#' using adaptive random-walk Metropolis-Hastings within each ESS iteration.
#'
#' @references
#' Murray, I., Adams, R., & MacKay, D. (2010). Elliptical slice sampling.
#' Proceedings of the 13th International Conference on Artificial Intelligence
#' and Statistics (AISTATS).
#'
#' @name ess_backend
#' @keywords internal
NULL


#' Fit ratio model using Elliptical Slice Sampling
#'
#' @param formula A ratiod_formula object
#' @param data Data frame
#' @param family Model family
#' @param spatial Optional spatial structure
#' @param temporal Optional temporal structure
#' @param spatiotemporal Optional spatiotemporal structure
#' @param zi Optional zero-inflation specification
#' @param latent Optional latent factor specification
#' @param priors Prior specification
#' @param iter Total iterations
#' @param warmup Warmup iterations
#' @param chains Number of chains; chain c is seeded with `seed + c - 1`
#' @param seed Random seed
#' @param verbose Print progress
#'
#' @return A `ratiod_fit` whose draws are unpacked to the natural scale
#'   exactly as an HMC fit's are, with `backend = "ess"`.
#' @keywords internal
fit_ess <- function(formula,
                    data,
                    family,
                    spatial = NULL,
                    temporal = NULL,
                    spatiotemporal = NULL,
                    zi = NULL,
                    latent = NULL,
                    priors = NULL,
                    iter = 2000,
                    warmup = 1000,
                    chains = 1,
                    seed = NULL,
                    verbose = TRUE) {

  # Set seed
  if (is.null(seed)) {
    seed <- sample.int(.Machine$integer.max, 1)
  }

  assert_backend_fits_structures(
    "ess",
    list(spatial = spatial, temporal = temporal,
         spatiotemporal = spatiotemporal, zi = zi, latent = latent)
  )

  inp <- sampler_inputs(formula, data, family, spatial, temporal,
                        spatiotemporal, zi, latent, priors, re_param = "centered")
  hmc_data <- inp$hmc_data
  if (identical(inp$spatial_info$type, "car_proper")) {
    stop("Proper CAR (`spatial_car(..., proper = TRUE)`) is not supported by ",
         "the ESS backend. Use `mode = \"hmc\"`.", call. = FALSE)
  }
  assert_single_random_intercept(hmc_data, "ess")

  # The sampler starts every chain at zero on the layout it computes
  q_init <- numeric(0)

  if (verbose) {
    message("Elliptical Slice Sampling")
    message("  Iterations: ", iter, " (warmup: ", warmup, "), chains: ", chains)
  }

  runs <- lapply(seq_len(chains), function(c) {
    do.call(cpp_ess_fit, c(
      list(q_init = q_init),
      model_data_args(inp, family),
      list(n_iter = as.integer(iter), n_warmup = as.integer(warmup),
           seed = as.integer(seed + c - 1L), verbose = verbose)
    ))
  })

  # ESS samples the parameter vector of the HMC sampler (one layout,
  # compute_param_layout()), with random effects in the centred coordinate,
  # so its draws go through the same unpacking. It has no trajectories: no
  # divergences, tree depth or energy.
  per_chain <- function(field) {
    out <- lapply(runs, function(r) r[[field]])
    if (chains == 1L) out[[1L]] else out
  }
  n_saved <- nrow(runs[[1L]]$samples)
  fill <- function(value) {
    out <- rep(list(rep(value, n_saved)), chains)
    if (chains == 1L) out[[1L]] else out
  }
  fit_raw <- list(
    samples = per_chain("samples"),
    log_prob = per_chain("log_lik"),
    accept_prob = fill(1),
    n_leapfrog = fill(0L),
    treedepth = fill(0L),
    divergent = fill(FALSE),
    epsilon = rep(NA_real_, chains),
    sampler = "ESS"
  )

  fit <- convert_hmc_to_ratiod_fit_full(
    fit_raw = fit_raw,
    hmc_data = hmc_data,
    spatial_info = inp$spatial_info,
    temporal_info = inp$temporal_info,
    zi_info = inp$zi_info,
    formula = formula,
    data = data,
    family = family,
    model_type = inp$model_type,
    iter = iter,
    warmup = warmup,
    chains = chains,
    re_param = "centered"
  )
  fit$backend <- "ess"
  fit$seed <- seed
  fit$diagnostics$avg_slice_evals <- mean(vapply(runs, function(r) r$avg_slice_evals, numeric(1)))

  if (verbose) {
    message("ESS complete. Avg slice evaluations per step: ",
            round(fit$diagnostics$avg_slice_evals, 2))
  }

  fit
}
