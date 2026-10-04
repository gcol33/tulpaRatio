#' Fit a ratio, rate, or proportion model
#'
#' @description
#' Fit a Bayesian hierarchical model for ratios, rates, or proportions.
#' The model jointly estimates the numerator and denominator processes
#' with optional shared latent structure.
#'
#' `tratio()` is tulpaRatio's front door, the ratio-model member of the
#' `tulpa*` family of fitting verbs ([tulpa::tulpa()] for the engine,
#' `tulpaObs::tobs()` for observation processes).
#'
#' **Core principle:** Ratios are derived quantities, computed post hoc
#' from the posterior of the joint model. The ratio is never modelled directly.
#'
#' @param formula Model formula specifying the response and predictors.
#'   Two syntax options: **Combined** (recommended) `num | denom ~ predictors + (1|group)`
#'   where both processes share the same predictors and random effects;
#'   **Separate** `num ~ predictors` with the `formula_denom` argument.
#' @param data Data frame containing all variables.
#' @param family A tulpaRatio family object specifying the distributions:
#'   [ratiod_negbin_negbin()] for both count processes (default),
#'   [ratiod_binomial()] for successes/trials,
#'   [ratiod_poisson_gamma()] for count/continuous effort (CPUE).
#' @param formula_num Optional one-sided formula for additional numerator
#'   predictors: `~ extra_terms`. Added to the main formula.
#' @param formula_denom Optional formula for denominator. Required if main
#'   formula has single response. Can be `denom ~ predictors`.
#' @param shared Formula specifying shared random effects structure:
#'   `NULL` (default) infers from matching random effects in both processes;
#'   `~ (1 | group)` for explicit shared structure;
#'   `~ 0` for independence assumption (triggers warning).
#' @param spatial Optional spatial structure specification.
#'   See [spatial_car()], [spatial_bym2()].
#' @param temporal Optional temporal structure specification.
#'   See [temporal_rw1()], [temporal_rw2()], [temporal_ar1()].
#' @param spatiotemporal Optional spatiotemporal interaction specification.
#'   See [spatiotemporal()].
#' @param zi Optional zero-inflation specification.
#'   See [zi_poisson()], [zi_negbin()], [hurdle_poisson()], [hurdle_negbin()].
#' @param latent Optional latent factor specification for unmeasured confounders.
#'   See [latent_factor()].
#' @param priors Prior specification. See [ratiod_priors()].
#' @param mode Inference mode or backend. Accepts either:
#'
#'   **Tier names** (epistemic guarantees):
#'   - `"auto"` (default): Selects between Exact and Structured based on model.
#'     **Never** silently chooses Optimized.
#'   - `"exact"` (Tier 1): Asymptotically correct posterior inference.
#'     Credible intervals are interpretable as posterior uncertainty.
#'   - `"structured"` (Tier 2): Accurate inference conditional on structural
#'     assumptions (latent Gaussian, conditional independence).
#'   - `"optimized"` (Tier 3): No general correctness guarantee.
#'     Point estimates usually good, uncertainty often underestimated.
#'     **Requires explicit opt-in.**
#'
#'   **Backend names** (specific implementation):
#'   - `"hmc"` (Tier 1): Native HMC/NUTS sampler
#'   - `"ess"` (Tier 1): Elliptical Slice Sampling
#'   - `"pg"` (Tier 1): Pólya-Gamma Gibbs (binomial only)
#'   - `"sghmc"` (Tier 1): Stochastic Gradient HMC (large datasets)
#'   - `"sgld"` (Tier 1): Stochastic Gradient Langevin Dynamics (large datasets)
#'   - `"laplace"` (Tier 2): Laplace approximation
#'   - `"vi"` (Tier 3): Variational Inference
#'
#'   **Structures each backend fits.** `"hmc"` fits every structure. The rest
#'   carry a subset, and a model naming a structure its backend cannot fit is
#'   refused with that structure named:
#'   - `"ess"`, `"sghmc"`, `"sgld"`, `"vi"`: `spatial`, `temporal`, `zi`
#'   - `"gibbs"`: `spatial`, and `temporal` as [temporal_tvc()]
#'   - `"pg"`, `"laplace"`: `spatial`, or `temporal` as
#'     [temporal_multiscale()], one of the two per model
#'
#'   `"auto"` selects a backend that fits the whole model, so it reaches
#'   `"laplace"` or `"gibbs"` only for models those carry.
#'
#'   See [inference_mode_info()] for details on the tier system.
#' @param control Named list of perf / numerical / tuning knobs. Statistical
#'   arguments stay in the signature above; everything that tunes *how* the
#'   fit runs lives here. Unknown names are an error, not a silent no-op.
#'
#'   Sampling budget (all backends):
#'   - `chains`: number of MCMC chains (default 4).
#'   - `iter`: total iterations per chain (default 2000).
#'   - `warmup`: warmup iterations per chain (default `floor(iter / 2)`).
#'   - `thin`: thinning interval (default 1).
#'   - `cores`: cores for parallel chains (default `getOption("mc.cores", chains)`).
#'   - `seed`: random seed for reproducibility.
#'   - `verbose`: print progress messages during fitting (default `TRUE`).
#'
#'   NUTS / HMC (`mode = "hmc"`):
#'   - `adapt_delta`: target average acceptance probability for step-size
#'     adaptation. Higher values give smaller steps, reducing divergences but
#'     slowing sampling. `NULL` (default) selects on model complexity: 0.80
#'     simple, 0.85 ICAR, 0.90 BYM2 and correlated random slopes. Range
#'     0.80--0.99.
#'   - `max_treedepth`: maximum NUTS tree depth. `NULL` (default) uses 10.
#'     Raise if you see many max-treedepth warnings; costs computation per
#'     iteration.
#'   - `metric`: mass matrix. `"auto"` (default) picks on model complexity;
#'     `"dense"` full matrix with Ledoit-Wolf shrinkage (correlated posteriors:
#'     random slopes, BYM2, HSGP, TVC); `"diag"` diagonal (faster per step, may
#'     need deeper trees); `"block_diag"` per parameter group. Auto falls back
#'     to diagonal when p > 2000.
#'   - `walnuts`: run each NUTS trajectory as a WALNUTS transition
#'     (Bou-Rabee et al. 2025, tulpa's `<tulpa/walnuts.h>`), which integrates
#'     each macro step at the coarsest leapfrog subdivision that keeps the
#'     joint log density within tolerance and keeps it only if the reversed
#'     step picks the same one. The kernel is exact, and its step follows the
#'     local curvature, which is what a funnel-shaped posterior (a hierarchical
#'     scale near zero) needs. `FALSE` (default) runs NUTS.
#'   - `gradient_mode`: `"auto"` (default) selects the fastest available;
#'     `"H"` hand-coded analytical (fastest); `"A_r"` arena reverse-mode
#'     autodiff (fast, O(N)); `"A"` forward-mode (O(p*N), thread-safe);
#'     `"A_t"` tape-based reverse-mode (slow, legacy); `"N"` numerical finite
#'     differences (slowest, always works).
#'   - `re_param`: random-effect parameterization. `"noncentered"` (default)
#'     stores z ~ N(0, 1) and computes re = sigma * z (or diag(sigma) * L * z
#'     when correlated), suiting weakly-informed effects or small groups;
#'     `"centered"` stores re directly under re ~ N(0, sigma^2), suiting
#'     strongly-informed effects or large groups.
#'   - `L`: fixed leapfrog steps. Default 0 leaves NUTS to choose.
#'
#'   Variational inference (`mode = "vi"`):
#'   - `vi_variant`: `"auto"` (default) selects on parameter count;
#'     `"meanfield"` diagonal covariance (fastest); `"lowrank"` low-rank plus
#'     diagonal (balanced); `"fullrank"` full Cholesky (best for small models).
#'
#'   Stochastic gradient backends (`mode = "sghmc"` / `"sgld"`):
#'   - `batch_size`: minibatch size (default `ceiling(sqrt(n))`).
#'   - `epsilon`: step size.
#'   - `alpha`: friction coefficient (`"sghmc"` only).
#'   - `schedule_a`, `schedule_b`, `schedule_gamma`, `use_schedule`: step-size
#'     decay schedule (`"sgld"` only).
#'
#' @return A `ratiod_fit` object containing:
#' \describe{
#'   \item{draws}{Posterior draws matrix}
#'   \item{formula}{Parsed formula specification}
#'   \item{family}{Model family}
#'   \item{data}{Original data}
#'   \item{backend}{Inference backend used}
#'   \item{diagnostics}{MCMC diagnostics (if applicable)}
#' }
#'
#' @examples
#' # Quick example: create model object (no fitting)
#' set.seed(123)
#' df <- data.frame(
#'   catch = rpois(50, 10),
#'   effort = rgamma(50, 2, 0.5),
#'   depth = rnorm(50),
#'   season = factor(sample(c("spring", "summer"), 50, replace = TRUE)),
#'   site = factor(sample(1:5, 50, replace = TRUE))
#' )
#'
#' \donttest{
#' # CPUE example with combined formula
#' fit <- tratio(
#'   catch | effort ~ depth + season + (1 | site),
#'   data = df,
#'   family = ratiod_poisson_gamma(),
#'   control = list(iter = 200, warmup = 100, chains = 1)
#' )
#'
#' # Different predictors for each process
#' fit2 <- tratio(
#'   catch | effort ~ (1 | site),
#'   formula_num = ~ depth + season,
#'   formula_denom = ~ season,
#'   data = df,
#'   family = ratiod_poisson_gamma(),
#'   control = list(iter = 200, warmup = 100, chains = 1)
#' )
#'
#' # Extract ratio posteriors
#' cpue <- ratio(fit)
#' summary(cpue)
#'
#' # Compare ratios between groups
#' ratio_contrast(fit, ~ season)
#' }
#'
#' @seealso
#' - [ratio()] for extracting ratio posteriors
#' - [ratio_contrast()] for comparing ratios between groups
#' - [pp_check()] for posterior predictive checks
#' - [loo::loo()] and [loo::waic()] for model comparison
#'
#' @export
tratio <- function(formula,
                   data,
                   family = ratiod_negbin_negbin(),
                   formula_num = NULL,
                   formula_denom = NULL,
                   shared = NULL,
                   spatial = NULL,
                   temporal = NULL,
                   spatiotemporal = NULL,
                   zi = NULL,
                   latent = NULL,
                   priors = NULL,
                   mode = c("auto", "exact", "structured", "optimized",
                            "hmc", "ess", "pg", "gibbs", "sghmc", "sgld",
                            "laplace", "vi"),
                   control = list()) {

  mode <- match.arg(mode)
  ctrl <- .tratio_control(control)

  chains        <- ctrl$chains
  iter          <- ctrl$iter
  warmup        <- ctrl$warmup
  thin          <- ctrl$thin
  cores         <- ctrl$cores
  seed          <- ctrl$seed
  verbose       <- ctrl$verbose
  vi_variant    <- ctrl$vi_variant
  gradient_mode <- ctrl$gradient_mode
  re_param      <- ctrl$re_param
  metric        <- ctrl$metric
  adapt_delta   <- ctrl$adapt_delta
  max_treedepth <- ctrl$max_treedepth
  walnuts       <- ctrl$walnuts
  L             <- as.integer(ctrl$L %||% 0L)

  # Test suites silence fit chatter to keep it out of the output buffer; an
  # explicit control$verbose still wins.
  if (isTRUE(getOption("tulpaRatio.test_mode")) &&
      !("verbose" %in% names(control))) {
    verbose <- FALSE
  }

  # Validate temporal specification
  if (!is.null(temporal)) {
    if (!inherits(temporal, "ratiod_temporal")) {
      stop(
        "`temporal` must be a ratiod_temporal object.\n",
        "Options: temporal_rw1(), temporal_rw2(), temporal_ar1(), temporal_iid(), temporal_tvc()",
        call. = FALSE
      )
    }
    # TVC needs special validation with design matrix
    if (inherits(temporal, "ratiod_tvc")) {
      # Parse formula to get design matrix for TVC validation
      parsed_formula <- ratiod_formula(formula, formula_num = formula_num,
                                       formula_denom = formula_denom,
                                       shared = shared, data = data)
      X_num <- parsed_formula$numerator$X
      temporal <- validate_tvc(temporal, data, X_num)
    } else if (inherits(temporal, "ratiod_temporal_gp")) {
      # Temporal GP needs its own validation to populate GP-specific fields
      temporal <- validate_temporal_gp(temporal, data)
    } else if (inherits(temporal, "ratiod_temporal_multiscale")) {
      # Multiscale temporal needs its own validation
      temporal <- validate_temporal_multiscale(temporal, data)
    } else {
      temporal <- validate_temporal(temporal, data)
    }
  }

  # Validate spatial specification
  # (SVC is now handled in fit_hmc() with prepare_svc_for_hmc())

  # Validate zero-inflation specification
  if (!is.null(zi)) {
    zi <- validate_zi(zi, data)
  }

  # Validate latent factor specification
  if (!is.null(latent)) {
    if (!inherits(latent, "ratiod_latent")) {
      stop(
        "`latent` must be a ratiod_latent object from latent_factor()",
        call. = FALSE
      )
    }
    latent <- validate_latent(latent, nrow(data))
  }

  # Validate spatiotemporal specification
  if (!is.null(spatiotemporal)) {
    if (!inherits(spatiotemporal, "ratiod_spatiotemporal")) {
      stop(
        "`spatiotemporal` must be a ratiod_spatiotemporal object.\n",
        "Build one with spatiotemporal().",
        call. = FALSE
      )
    }
    # Validate and compute dimensions
    spatiotemporal <- validate_spatiotemporal(spatiotemporal, data)

    # Auto-extract main spatial/temporal effects from spatiotemporal if not
    # provided separately. The Knorr-Held decomposition requires main effects
    # (phi_s, phi_t) alongside the interaction (delta_st).
    if (is.null(spatial) && !is.null(spatiotemporal$spatial)) {
      spatial <- spatiotemporal$spatial
      # validate_spatial() just checks, returns invisible(NULL) — don't reassign
      if (!is.null(spatial) && !inherits(spatial, c("ratiod_gp", "ratiod_multiscale"))) {
        validate_spatial(spatial, data)
      }
    }
    if (is.null(temporal) && !is.null(spatiotemporal$temporal)) {
      temporal <- spatiotemporal$temporal
      # validate_temporal() returns the validated object — reassign
      if (!is.null(temporal)) {
        temporal <- validate_temporal(temporal, data)
      }
    }
  }

  # ===========================================================================
  # Mode and Backend Selection
  # ===========================================================================

  # Everything the model asks a backend to carry, read by mode selection and
  # by the capability check below.
  structures <- list(
    spatial = spatial,
    temporal = temporal,
    spatiotemporal = spatiotemporal,
    zi = zi,
    latent = latent
  )

  # Select inference mode and backend
  mode_selection <- select_inference_mode(
    mode = mode,
    family = family,
    n_obs = nrow(data),
    structures = structures
  )

  # Extract selected backend
  selected_backend <- mode_selection$backend
  selected_mode <- mode_selection$mode
  selected_tier <- mode_selection$tier
  selected_tier_name <- mode_selection$tier_name

  # Display mode selection
  if (verbose) {
    message(sprintf("Inference: %s (Tier %d)", selected_tier_name, selected_tier))
    message(sprintf("  Backend: %s", selected_backend))
    if (mode == "auto") {
      message(sprintf("  Reason: %s", mode_selection$reason))
    }
  }

  # Refuse a model the selected backend would fit without part of its
  # structure. `auto` never reaches this, having filtered on the same table.
  assert_backend_fits_structures(selected_backend, structures)

  # PG backend only works for binomial family (currently experimental)
  if (selected_backend == "pg") {
    if (!can_use_pg_backend(family)) {
      stop(
        "PG backend only supports ratiod_binomial() family.\n",
        "For other families, use backend = 'hmc' or mode = 'exact'.",
        call. = FALSE
      )
    }
  }

  # Validate inputs
  if (missing(formula)) {
    stop("`formula` is required. Use: num | denom ~ predictors", call. = FALSE)
  }
  if (missing(data)) {
    stop("`data` is required", call. = FALSE)
  }
  if (!is.data.frame(data)) {
    stop("`data` must be a data frame", call. = FALSE)
  }
  if (!inherits(family, "ratiod_family")) {
    stop(
      "`family` must be a ratiod_family object.\n",
      "Options: ratiod_negbin_negbin(), ratiod_binomial(), ratiod_poisson_gamma()",
      call. = FALSE
    )
  }

  # Suggest alternative modes based on data characteristics (only for auto mode)
  if (mode == "auto") {
    suggest_mode(
      n_obs = nrow(data),
      structures = structures,
      current_mode = selected_mode
    )
  }

  # Parse formula
  formula_spec <- ratiod_formula(
    formula = formula,
    formula_num = formula_num,
    formula_denom = formula_denom,
    shared = shared,
    data = data
  )

  # Every backend reads its responses off this object, so the family's
  # requirements on them are checked here rather than once per backend.
  validate_ratio_responses(formula_spec, family)

  # Set default priors
  if (is.null(priors)) {
    priors <- ratiod_priors()
  }

  # -------------------------------------------------------------------------
  # Dispatch to appropriate backend
  # -------------------------------------------------------------------------

  # Common message elements
  if (verbose) {
    message(sprintf("Fitting ratio model..."))
    message(sprintf("  Family: %s", family$name))
    message(sprintf("  Observations: %d", nrow(data)))
  }

  # Store mode info for the result
  mode_info <- list(
    mode = selected_mode,
    tier = selected_tier,
    tier_name = selected_tier_name,
    backend = selected_backend,
    reason = mode_selection$reason
  )

  # -------------------------------------------------------------------------
  # Dispatch to Gibbs backend (Tier 1: Exact)
  # -------------------------------------------------------------------------
  if (selected_backend == "gibbs") {
    if (verbose) {
      message(sprintf("  Iterations: %d (warmup: %d)", iter, warmup))
    }

    result <- fit_gibbs(
      formula = formula_spec,
      data = data,
      family = family,
      spatial = spatial,
      temporal = temporal,
      priors = priors,
      iter = iter,
      warmup = warmup,
      thin = thin,
      chains = chains,
      cores = cores,
      seed = seed,
      verbose = verbose
    )

    # Add mode information
    result$inference_mode <- selected_mode
    result$inference_tier <- selected_tier
    result$inference_tier_name <- selected_tier_name
    result$mode_info <- mode_info

    return(result)
  }

  # -------------------------------------------------------------------------
  # Dispatch to PG backend (Tier 1: Exact)
  # -------------------------------------------------------------------------
  if (selected_backend == "pg") {
    if (verbose) {
      message(sprintf("  Iterations: %d (warmup: %d)", iter, warmup))
      message(sprintf("  Cores: %d", cores))
    }

    result <- fit_pg_binomial(
      formula = formula_spec,
      data = data,
      family = family,
      spatial = spatial,
      temporal = temporal,
      iter = iter,
      warmup = warmup,
      chains = chains,
      thin = thin,
      cores = cores,
      seed = seed,
      prior_beta_sd = priors$beta_sd %||% 10,
      prior_sigma_scale = priors$sigma_scale %||% 2.5,
      prior_tau_shape = priors$tau_shape %||% 1,
      prior_tau_rate = priors$tau_rate %||% 0.01,
      verbose = verbose
    )

    # Add mode information
    result$inference_mode <- selected_mode
    result$inference_tier <- selected_tier
    result$inference_tier_name <- selected_tier_name
    result$mode_info <- mode_info

    return(result)
  }

  # -------------------------------------------------------------------------
  # Dispatch to Laplace backend (Tier 2: Structured)
  # -------------------------------------------------------------------------
  if (selected_backend == "laplace") {
    if (verbose) {
      message(sprintf("  Posterior samples: %d", iter - warmup))
      message(sprintf("  Cores: %d", cores))
    }

    result <- fit_laplace(
      formula = formula_spec,
      data = data,
      family = family,
      spatial = spatial,
      temporal = temporal,
      priors = priors,
      n_samples = iter - warmup,
      cores = cores,
      verbose = verbose
    )

    # Add mode information
    result$inference_mode <- selected_mode
    result$inference_tier <- selected_tier
    result$inference_tier_name <- selected_tier_name
    result$mode_info <- mode_info

    return(result)
  }

  # -------------------------------------------------------------------------
  # Dispatch to HMC/NUTS backend (Tier 1: Exact)
  # -------------------------------------------------------------------------
  if (selected_backend == "hmc") {
    if (verbose) {
      message(sprintf("  Iterations: %d (warmup: %d)", iter, warmup))
      if (!is.null(temporal)) {
        message(sprintf("  Temporal: %s (%d time points)", temporal$type, temporal$n_times))
      }
      if (!is.null(spatiotemporal)) {
        message(sprintf("  Spatiotemporal: %s (%d x %d)",
                        spatiotemporal$type, spatiotemporal$n_spatial, spatiotemporal$n_times))
      }
    }

    result <- fit_hmc(
      formula = formula_spec,
      data = data,
      family = family,
      spatial = spatial,
      temporal = temporal,
      spatiotemporal = spatiotemporal,
      zi = zi,
      latent = latent,
      priors = priors,
      iter = iter,
      warmup = warmup,
      chains = chains,
      cores = cores,
      L = L,
      adapt_delta = adapt_delta,
      max_treedepth = max_treedepth,
      seed = seed,
      verbose = verbose,
      gradient_mode = gradient_mode,
      re_param = re_param,
      metric = metric,
      walnuts = walnuts
    )

    # Add mode information
    result$inference_mode <- selected_mode
    result$inference_tier <- selected_tier
    result$inference_tier_name <- selected_tier_name
    result$mode_info <- mode_info

    return(result)
  }

  # -------------------------------------------------------------------------
  # Dispatch to VI backend (Tier 3: Optimized)
  # -------------------------------------------------------------------------
  if (selected_backend == "vi") {
    if (verbose) {
      message(sprintf("  VI variant: %s", vi_variant))
      message("  Note: Tier 3 (Optimized) - uncertainty may be underestimated")
    }

    result <- fit_vi(
      formula = formula_spec,
      data = data,
      family = family,
      spatial = spatial,
      temporal = temporal,
      spatiotemporal = spatiotemporal,
      zi = zi,
      latent = latent,
      priors = priors,
      variant = vi_variant,
      max_iter = iter,
      seed = seed,
      verbose = verbose
    )

    # Add mode information
    result$inference_mode <- selected_mode
    result$inference_tier <- selected_tier
    result$inference_tier_name <- selected_tier_name
    result$mode_info <- mode_info

    return(result)
  }

  # -------------------------------------------------------------------------
  # Dispatch to ESS backend (Tier 1: Exact)
  # -------------------------------------------------------------------------
  if (selected_backend == "ess") {
    if (verbose) {
      message(sprintf("  Iterations: %d (warmup: %d)", iter, warmup))
    }

    result <- fit_ess(
      formula = formula_spec,
      data = data,
      family = family,
      spatial = spatial,
      temporal = temporal,
      spatiotemporal = spatiotemporal,
      zi = zi,
      latent = latent,
      priors = priors,
      iter = iter,
      warmup = warmup,
      seed = seed,
      verbose = verbose
    )

    # Add mode information
    result$inference_mode <- selected_mode
    result$inference_tier <- selected_tier
    result$inference_tier_name <- selected_tier_name
    result$mode_info <- mode_info

    return(result)
  }

  # -------------------------------------------------------------------------
  # Dispatch to SGHMC backend (Tier 1: Exact, large-scale)
  # -------------------------------------------------------------------------
  if (selected_backend == "sghmc") {
    # Default batch size: sqrt(n) for good variance/speed tradeoff
    batch_size <- ctrl$batch_size %||% ceiling(sqrt(nrow(data)))
    extra_args <- .tratio_sg_args(ctrl, "sghmc")

    if (verbose) {
      message(sprintf("  Iterations: %d (warmup: %d)", iter, warmup))
      message(sprintf("  Batch size: %d (%.1f%% of data)", batch_size,
                      100 * batch_size / nrow(data)))
      message("  Note: SGHMC uses minibatch gradients for scalability")
    }

    result <- do.call(fit_sghmc, c(list(
      formula = formula_spec,
      data = data,
      family = family,
      spatial = spatial,
      temporal = temporal,
      zi = zi,
      priors = priors,
      iter = iter,
      warmup = warmup,
      batch_size = batch_size,
      seed = seed,
      verbose = verbose
    ), extra_args))

    # Add mode information
    result$inference_mode <- selected_mode
    result$inference_tier <- selected_tier
    result$inference_tier_name <- selected_tier_name
    result$mode_info <- mode_info

    return(result)
  }

  # -------------------------------------------------------------------------
  # Dispatch to SGLD backend (Tier 1: Exact, large-scale)
  # -------------------------------------------------------------------------
  if (selected_backend == "sgld") {
    # Default batch size: sqrt(n) for good variance/speed tradeoff
    batch_size <- ctrl$batch_size %||% ceiling(sqrt(nrow(data)))
    extra_args <- .tratio_sg_args(ctrl, "sgld")

    if (verbose) {
      message(sprintf("  Iterations: %d (warmup: %d)", iter, warmup))
      message(sprintf("  Batch size: %d (%.1f%% of data)", batch_size,
                      100 * batch_size / nrow(data)))
      message("  Note: SGLD uses Langevin dynamics with minibatch gradients")
    }

    result <- do.call(fit_sgld, c(list(
      formula = formula_spec,
      data = data,
      family = family,
      spatial = spatial,
      temporal = temporal,
      zi = zi,
      priors = priors,
      iter = iter,
      warmup = warmup,
      batch_size = batch_size,
      seed = seed,
      verbose = verbose
    ), extra_args))

    # Add mode information
    result$inference_mode <- selected_mode
    result$inference_tier <- selected_tier
    result$inference_tier_name <- selected_tier_name
    result$mode_info <- mode_info

    return(result)
  }

  # If we get here, no valid backend was selected
  stop(
    sprintf("Unknown backend: '%s'", selected_backend),
    call. = FALSE
  )
}


#' Suggest alternative modes based on data characteristics
#'
#' @description
#' Only backends that fit every structure in the model are suggested, so a
#' suggestion taken up gives the same model under a faster backend.
#'
#' @param n_obs Number of observations
#' @param structures Named list of structure specifications, `NULL` where the
#'   structure is absent
#' @param current_mode Currently selected mode
#' @keywords internal
suggest_mode <- function(n_obs, structures, current_mode) {

  # Thresholds for suggestions
  LARGE_DATA_THRESHOLD <- 10000
  VERY_LARGE_DATA_THRESHOLD <- 50000

  if (current_mode != "exact" || n_obs < LARGE_DATA_THRESHOLD) {
    return(invisible(NULL))
  }

  alternatives <- c(
    sghmc = "  - mode = 'sghmc' for minibatch MCMC (Tier 1, scales better)",
    sgld = "  - mode = 'sgld' for stochastic gradient Langevin dynamics (Tier 1)",
    laplace = "  - mode = 'structured' for Tier 2 (Laplace approximation)"
  )
  alternatives <- alternatives[vapply(
    names(alternatives), backend_fits_structures, logical(1),
    structures = structures
  )]

  if (n_obs >= VERY_LARGE_DATA_THRESHOLD) {
    header <- c(
      sprintf("Large dataset (%s observations).", format(n_obs, big.mark = ",")),
      "Exact inference (Tier 1) may be slow. Consider:"
    )
    tail_lines <- "  - Reducing data via stratified sampling"
  } else {
    header <- c(
      sprintf("Moderately large dataset (%s observations).", format(n_obs, big.mark = ",")),
      "If fitting is slow, consider:"
    )
    tail_lines <- "  - Fewer iterations (iter = 1000 for initial exploration)"
  }

  suggestions <- c(header, unname(alternatives), tail_lines)

  # Print suggestions as a message (not warning - it's informational)
  message("\n", cli_rule("Mode suggestion"))
  for (s in suggestions) {
    message("  ", s)
  }
  message(cli_rule(), "\n")

  invisible(NULL)
}

#' Simple CLI rule for messages
#' @keywords internal
cli_rule <- function(title = NULL) {
  width <- getOption("width", 80)
  if (is.null(title)) {
    paste(rep("-", min(width, 60)), collapse = "")
  } else {
    title_len <- nchar(title) + 2
    side_len <- (min(width, 60) - title_len) / 2
    paste0(
      paste(rep("-", floor(side_len)), collapse = ""),
      " ", title, " ",
      paste(rep("-", ceiling(side_len)), collapse = "")
    )
  }
}

#' Select optimal backend based on model characteristics
#'
#' @description
#' Automatically chooses the best backend for a given model based on
#' the family, data size, and presence of spatial/temporal effects.
#'
#' @param family tulpaRatio family object
#' @param n_obs Number of observations
#' @param has_spatial Whether spatial effects are specified
#' @param has_temporal Whether temporal effects are specified
#'
#' @return Character string: "pg", "hmc", or "laplace"
#'
#' @details
#' Selection logic:
#' - **Binomial family**: Use PG (Polya-Gamma) - fastest for binomial
#' - **Spatial/temporal models**: Use HMC (only HMC supports all structure types)
#' - **Very large data (N > 50,000)**: Use Laplace (approximate but fast)
#' - **Default**: Use HMC (full MCMC, works for all families)
#'
#' @keywords internal
select_backend <- function(family, n_obs, has_spatial = FALSE, has_temporal = FALSE) {

  # Thresholds
  VERY_LARGE <- 50000
  LARGE <- 10000

  family_name <- family$name

  # Spatial/temporal models: HMC is most complete
  if (has_spatial || has_temporal) {
    return("hmc")
  }

  # Very large data: Laplace for speed (approximate but fast)
  if (n_obs > VERY_LARGE) {
    return("laplace")
  }

  # Default: HMC (full MCMC, works for everything)
  # Note: PG backend exists for binomial but is experimental
  return("hmc")
}


#' Print method for ratiod_fit
#'
#' @param x A ratiod_fit object
#' @param ... Ignored
#' @export
print.ratiod_fit <- function(x, ...) {
  cat("ratio model fit\n")
  cat("===============\n\n")

  # Inference mode information (always visible)
  tier <- x$inference_tier %||% NA
  tier_name <- x$inference_tier_name %||% "Unknown"
  backend <- x$backend %||% "unknown"

  if (!is.na(tier)) {
    cat(sprintf("Inference: %s (Tier %d)\n", tier_name, tier))
  } else {
    cat(sprintf("Backend: %s\n", backend))
  }

  cat("Family:", x$family$name, "\n")
  cat("  ", x$family$description, "\n\n")

  cat("Data:\n")
  cat("  Observations:", nrow(x$data), "\n")
  cat("  Numerator:", x$formula$numerator$response_var, "\n")
  cat("  Denominator:", x$formula$denominator$response_var, "\n\n")

  cat("Structure:\n")
  cat("  Num fixed effects:", ncol(x$formula$numerator$X), "\n")
  cat("  Denom fixed effects:", ncol(x$formula$denominator$X), "\n")

  n_re <- x$stan_data$n_re_groups %||% 0L
  if (length(n_re) > 0 && n_re > 0) {
    cat("  Random effect groups:", n_re, "\n")
    if (x$formula$shared$type %in% c("inferred", "explicit")) {
      cat("  Shared structure:", x$formula$shared$type, "\n")
    }
  }

  if (isTRUE(x$stan_data$use_spatial)) {
    cat("  Spatial:", "Yes\n")
  }

  # Tier-specific notes
  if (!is.na(tier)) {
    cat("\n")
    if (tier == 1) {
      cat("Tier 1: Credible intervals are posterior uncertainty\n")
    } else if (tier == 2) {
      cat("Tier 2: Inference conditional on structural assumptions\n")
    } else if (tier == 3) {
      cat("Tier 3: Uncertainty may be underestimated\n")
    }
  }

  cat("\nUse summary() for parameter estimates\n")
  cat("Use ratio() to extract ratio posteriors\n")

  invisible(x)
}

#' Summary method for ratiod_fit
#'
#' @description
#' Provides a summary of the fitted model including parameter estimates,
#' credible intervals, and MCMC diagnostics. Works with all backends
#' (HMC, PG, Laplace).
#'
#' @param object A ratiod_fit object
#' @param prob Probability mass for credible intervals (default 0.95)
#' @param ... Ignored
#' @export
summary.ratiod_fit <- function(object, prob = 0.95, ...) {
  cat("ratio model summary\n")
  cat("===================\n\n")

  # Inference mode information
  tier <- object$inference_tier %||% NA
  tier_name <- object$inference_tier_name %||% "Unknown"
  backend <- object$backend %||% "unknown"

  if (!is.na(tier)) {
    cat(sprintf("Inference: %s (Tier %d) via %s\n", tier_name, tier, backend))
  } else {
    cat("Backend:", backend, "\n")
  }
  cat("Family:", object$family$name, "\n\n")

  # Calculate quantile probabilities
  alpha <- (1 - prob) / 2
  probs <- c(alpha, 0.5, 1 - alpha)
  ci_names <- paste0(c(alpha * 100, 50, (1 - alpha) * 100), "%")

  # Get draws
  draws <- object$draws
  if (is.null(draws)) {
    cat("No draws available\n")
    return(invisible(object))
  }

  # Compute summary statistics
  summ <- compute_param_summary(draws, probs)
  if (identical(backend, "laplace")) {
    summ <- apply_laplace_engine_summary(summ, object, level = prob)
  }

  # Add diagnostics for MCMC backends
  if (backend %in% c("hmc", "pg", "sghmc", "sgld", "ess")) {
    diag <- tryCatch(
      diagnostics(object),
      error = function(e) NULL
    )
    if (!is.null(diag)) {
      summ <- merge(summ, diag[, c("parameter", "rhat", "ess_bulk")],
                    by = "parameter", all.x = TRUE)
    }
  }

  # Select main parameters (exclude high-dimensional)
  main_pars <- select_main_params(summ$parameter)
  summ_main <- summ[summ$parameter %in% main_pars, ]

  # Categorize and print parameters
  print_param_summary(summ_main, object$family$name, ci_names,
                      fixed = object$fixed_columns)

  # Print diagnostics summary
  print_diagnostics_summary(object)

  invisible(object)
}


#' Compute parameter summary statistics
#' @keywords internal
compute_param_summary <- function(draws, probs) {
  par_names <- colnames(draws)
  n_pars <- ncol(draws)

  means <- colMeans(draws)
  sds <- apply(draws, 2, stats::sd)
  quants <- apply(draws, 2, stats::quantile, probs = probs)

  data.frame(
    parameter = par_names,
    mean = means,
    sd = sds,
    q_lower = quants[1, ],
    q_median = quants[2, ],
    q_upper = quants[3, ],
    stringsAsFactors = FALSE
  )
}


#' Read the fixed effects of a Laplace summary off the engine's mixture
#'
#' The engine integrates the hyperparameters over its outer grid, so the
#' fixed-effect posterior is a Gaussian mixture whose mean, covariance and
#' interval it reads exactly. The stored draws serve derived quantities; the
#' fixed-effect rows of the summary do not move with their seed.
#' @keywords internal
apply_laplace_engine_summary <- function(summ, fit, level) {
  engine <- fit$engine
  rows <- match(fit$fixed_columns, summ$parameter)
  interval <- stats::confint(engine, level = level)
  summ$mean[rows] <- stats::coef(engine)
  summ$sd[rows] <- sqrt(diag(stats::vcov(engine)))
  summ$q_lower[rows] <- interval[, 1L]
  summ$q_upper[rows] <- interval[, 2L]
  summ
}


#' Print categorized parameter summary
#' @keywords internal
print_param_summary <- function(summ, family_name, ci_names, fixed = character(0)) {
  # Format columns
  summ$mean <- round(summ$mean, 3)
  summ$sd <- round(summ$sd, 3)
  summ$q_lower <- round(summ$q_lower, 3)
  summ$q_upper <- round(summ$q_upper, 3)

  if ("rhat" %in% names(summ)) {
    summ$rhat <- round(summ$rhat, 3)
    summ$ess_bulk <- round(summ$ess_bulk, 0)
  }

  # Rename CI columns
  names(summ)[names(summ) == "q_lower"] <- ci_names[1]
  names(summ)[names(summ) == "q_upper"] <- ci_names[3]

  # Select columns for display
  display_cols <- c("parameter", "mean", "sd", ci_names[1], ci_names[3])
  if ("rhat" %in% names(summ)) {
    display_cols <- c(display_cols, "rhat", "ess_bulk")
  }

  # Fixed effects numerator
  beta_num <- summ[grepl("^beta_num", summ$parameter), ]
  if (nrow(beta_num) > 0) {
    cat("Fixed effects (numerator):\n")
    print(beta_num[, display_cols, drop = FALSE], row.names = FALSE)
    cat("\n")
  }

  # Fixed effects denominator
  beta_denom <- summ[grepl("^beta_denom", summ$parameter), ]
  if (nrow(beta_denom) > 0) {
    cat("Fixed effects (denominator):\n")
    print(beta_denom[, display_cols, drop = FALSE], row.names = FALSE)
    cat("\n")
  }

  # Fixed effects (binomial - single process)
  beta_single <- summ[grepl("^\\(Intercept\\)|^beta\\[", summ$parameter) |
                        summ$parameter %in% fixed, ]
  if (nrow(beta_single) > 0 && nrow(beta_num) == 0) {
    cat("Fixed effects:\n")
    print(beta_single[, display_cols, drop = FALSE], row.names = FALSE)
    cat("\n")
  }

  # Variance components
  var_pars <- summ[grepl("^(sigma|phi_num|phi_denom|tau|shape|rho)", summ$parameter), ]
  if (nrow(var_pars) > 0) {
    cat("Variance components:\n")
    print(var_pars[, display_cols, drop = FALSE], row.names = FALSE)
    cat("\n")
  }
}


#' Print diagnostics summary
#' @keywords internal
print_diagnostics_summary <- function(object) {
  backend <- object$backend %||% "unknown"

  cat("Diagnostics:\n")

  if (backend == "hmc") {
    # HMC-specific diagnostics
    diag <- object$diagnostics

    if (!is.null(diag)) {
      n_div <- diag$n_divergent %||% 0
      avg_accept <- diag$avg_accept_prob %||% NA

      cat("  Divergences:", n_div, "\n")
      if (!is.na(avg_accept)) {
        cat("  Avg. acceptance:", round(avg_accept, 3), "\n")
      }

      if (n_div > 0) {
        cat("  Warning: Divergent transitions detected.\n")
        cat("  Consider: reparameterization or increasing adapt_delta.\n")
      }
    }

    cat("  Chains:", object$chains %||% 1, "\n")
    cat("  Iterations:", object$iter %||% NA, "\n")
    cat("  Warmup:", object$warmup %||% NA, "\n")

  } else if (backend == "pg") {
    # Polya-Gamma diagnostics
    cat("  Iterations:", object$iter %||% NA, "\n")
    cat("  Warmup:", object$warmup %||% NA, "\n")
    cat("  Thin:", object$thin %||% 1, "\n")

  } else if (backend == "laplace") {
    # Laplace diagnostics
    result <- object$laplace_result %||% object$.internal
    if (!is.null(result)) {
      converged <- result$converged %||% NA
      log_marg <- result$log_marginal %||% NA

      cat("  Converged:", converged, "\n")
      if (!is.na(log_marg)) {
        cat("  Log marginal:", round(log_marg, 2), "\n")
      }
    }
    cat("  Posterior samples:", object$n_save %||% NA, "\n")
    cat("  Note: Laplace provides approximate inference.\n")

  } else if (backend %in% c("sghmc", "sgld")) {
    # Stochastic gradient MCMC diagnostics
    cat("  Iterations:", object$iter %||% NA, "\n")
    cat("  Warmup:", object$warmup %||% NA, "\n")
    cat("  Batch size:", object$batch_size %||% NA, "\n")
    if (backend == "sghmc") {
      cat("  Method: Stochastic Gradient HMC (Chen et al., 2014)\n")
    } else {
      cat("  Method: Stochastic Gradient Langevin Dynamics (Welling & Teh, 2011)\n")
    }
    cat("  Note: Uses minibatch gradients for scalability.\n")
  }
}

#' Extract posterior draws from ratiod_fit
#'
#' @param x A ratiod_fit object
#' @param ... Passed to posterior::as_draws_df
#' @return A draws_df object
#' @export
as.data.frame.ratiod_fit <- function(x, ...) {
  posterior::as_draws_df(x$fit$draws(...))
}


#' Plot method for ratiod_fit
#'
#' @description
#' Creates diagnostic plots for ratio model fits including trace plots
#' and optional density plots. Works with all backends (HMC, PG, Laplace).
#'
#' @param x A ratiod_fit object
#' @param pars Character vector of parameter names to plot. If NULL (default),
#'   plots main parameters (fixed effects, variance components).
#' @param type Type of plot: "trace" (default), "dens" (density), or "both".
#' @param n_col Number of columns in the plot grid (default: auto).
#' @param ... Additional arguments (currently ignored).
#'
#' @return Invisibly returns the plot object (base graphics or ggplot2).
#'
#' @examples
#' # Create simple dataset
#' set.seed(456)
#' df <- data.frame(
#'   count = rpois(40, 8),
#'   effort = rgamma(40, 2, 0.5),
#'   depth = rnorm(40),
#'   site = factor(sample(1:4, 40, replace = TRUE))
#' )
#'
#' \donttest{
#' fit <- tratio(
#'   count | effort ~ depth + (1|site),
#'   data = df,
#'   family = ratiod_poisson_gamma(),
#'   control = list(iter = 200, warmup = 100, chains = 2)
#' )
#' plot(fit)
#' plot(fit, pars = "beta_num", type = "both")
#' }
#'
#' @export
plot.ratiod_fit <- function(x, pars = NULL, type = c("trace", "dens", "both"),
                           n_col = NULL, ...) {
  type <- match.arg(type)

  # Get draws in array format for multi-chain support
  draws_info <- get_draws_array(x)
  draws_array <- draws_info$draws
  n_chains <- draws_info$n_chains

  # Select parameters to plot
  all_pars <- dimnames(draws_array)[[3]]
  if (is.null(pars)) {
    # Default: main parameters (exclude high-dimensional RE and spatial)
    pars <- select_main_params(all_pars)
  } else {
    # Handle regex-like selection
    pars <- grep_params(pars, all_pars)
  }

  if (length(pars) == 0) {
    stop("No parameters selected for plotting", call. = FALSE)
  }

  # Subset draws
  draws_subset <- draws_array[, , pars, drop = FALSE]

  # Use bayesplot if available, otherwise base R
  if (requireNamespace("bayesplot", quietly = TRUE)) {
    plot_with_bayesplot(draws_subset, type, n_chains)
  } else {
    plot_with_base(draws_subset, type, n_chains, n_col)
  }

  invisible(x)
}


#' Get draws in array format
#'
#' Converts draws to a 3D array with dimensions iterations x chains x parameters.
#'
#' @param fit A ratiod_fit object
#' @return A list with draws array and number of chains
#' @keywords internal
get_draws_array <- function(fit) {
  backend <- fit$backend %||% "unknown"
  chains <- fit$chains %||% 1

  if (backend == "hmc" && chains > 1) {
    # Multi-chain HMC: need to reconstruct chain structure
    # The raw samples per chain are stored in .internal or we split evenly
    draws <- fit$draws
    n_total <- nrow(draws)
    n_per_chain <- n_total / chains
    n_pars <- ncol(draws)

    draws_array <- array(
      dim = c(n_per_chain, chains, n_pars),
      dimnames = list(
        iteration = seq_len(n_per_chain),
        chain = seq_len(chains),
        parameter = colnames(draws)
      )
    )

    for (c in seq_len(chains)) {
      start_idx <- (c - 1) * n_per_chain + 1
      end_idx <- c * n_per_chain
      draws_array[, c, ] <- as.matrix(draws[start_idx:end_idx, ])
    }

    return(list(draws = draws_array, n_chains = chains))

  } else {
    # Single chain or non-HMC: treat as one chain
    draws <- fit$draws
    n_samples <- nrow(draws)
    n_pars <- ncol(draws)

    draws_array <- array(
      as.matrix(draws),
      dim = c(n_samples, 1, n_pars),
      dimnames = list(
        iteration = seq_len(n_samples),
        chain = 1,
        parameter = colnames(draws)
      )
    )

    return(list(draws = draws_array, n_chains = 1))
  }
}


#' Select main parameters (exclude high-dimensional)
#' @keywords internal
select_main_params <- function(all_pars) {
  # Keep: beta, sigma, phi, tau, rho
  # Exclude: re[...], spatial[...], phi_spatial[...], theta[...], phi_scaled[...]
  pattern <- "^(re|spatial|phi_spatial|theta|phi_scaled|ratio)\\["
  is_high_dim <- grepl(pattern, all_pars)
  main_pars <- all_pars[!is_high_dim]

  # Limit to reasonable number

  if (length(main_pars) > 12) {
    main_pars <- main_pars[1:12]
  }

  main_pars
}


#' Grep-style parameter selection
#' @keywords internal
grep_params <- function(pars, all_pars) {
  selected <- character(0)
  for (p in pars) {
    if (p %in% all_pars) {
      selected <- c(selected, p)
    } else {
      # Try as regex
      matches <- grep(p, all_pars, value = TRUE)
      selected <- c(selected, matches)
    }
  }
  unique(selected)
}


#' Plot using bayesplot
#' @keywords internal
plot_with_bayesplot <- function(draws_array, type, n_chains) {
  if (type == "trace") {
    print(bayesplot::mcmc_trace(draws_array))
  } else if (type == "dens") {
    if (n_chains > 1) {
      print(bayesplot::mcmc_dens_overlay(draws_array))
    } else {
      print(bayesplot::mcmc_dens(draws_array))
    }
  } else {
    # "both" - trace and density side by side
    print(bayesplot::mcmc_combo(draws_array, combo = c("trace", "dens_overlay")))
  }
}


#' Plot using base R graphics
#' @keywords internal
plot_with_base <- function(draws_array, type, n_chains, n_col = NULL) {
  n_pars <- dim(draws_array)[3]
  par_names <- dimnames(draws_array)[[3]]

  # Determine layout
  if (type == "both") {
    n_plots <- n_pars * 2
    if (is.null(n_col)) n_col <- 2
    n_row <- ceiling(n_pars / 1)
    graphics::par(mfrow = c(n_pars, 2), mar = c(2, 3, 2, 1), oma = c(0, 0, 2, 0))
  } else {
    n_plots <- n_pars
    if (is.null(n_col)) n_col <- min(3, ceiling(sqrt(n_pars)))
    n_row <- ceiling(n_pars / n_col)
    graphics::par(mfrow = c(n_row, n_col), mar = c(2, 3, 2, 1), oma = c(0, 0, 2, 0))
  }

  # Chain colors
  chain_cols <- c("#1b9e77", "#d95f02", "#7570b3", "#e7298a",
                  "#66a61e", "#e6ab02", "#a6761d", "#666666")

  for (i in seq_len(n_pars)) {
    par_name <- par_names[i]

    if (type %in% c("trace", "both")) {
      # Trace plot
      y_range <- range(draws_array[, , i])
      graphics::plot(
        NULL, NULL,
        xlim = c(1, dim(draws_array)[1]),
        ylim = y_range,
        xlab = "", ylab = "",
        main = par_name,
        type = "n"
      )
      for (c in seq_len(n_chains)) {
        graphics::lines(
          draws_array[, c, i],
          col = grDevices::adjustcolor(chain_cols[((c - 1) %% 8) + 1], alpha.f = 0.7),
          lwd = 0.5
        )
      }
    }

    if (type %in% c("dens", "both")) {
      # Density plot
      all_vals <- as.vector(draws_array[, , i])
      d <- stats::density(all_vals)

      if (n_chains > 1) {
        # Overlay densities per chain
        graphics::plot(d, main = par_name, xlab = "", ylab = "",
                       col = "gray", lwd = 2, type = "n")
        for (c in seq_len(n_chains)) {
          dc <- stats::density(draws_array[, c, i])
          graphics::lines(dc, col = chain_cols[((c - 1) %% 8) + 1], lwd = 1.5)
        }
      } else {
        graphics::plot(d, main = par_name, xlab = "", ylab = "",
                       col = chain_cols[1], lwd = 2)
        graphics::polygon(d, col = grDevices::adjustcolor(chain_cols[1], alpha.f = 0.3),
                          border = NA)
      }
    }
  }

  backend_label <- "ratio model MCMC diagnostics"
  graphics::mtext(backend_label, outer = TRUE, cex = 1.1)
}


#' Posterior diagnostics for a ratiod_fit
#'
#' @description
#' Computes per-parameter improved Rhat and bulk / tail effective sample size
#' by delegating to the engine's [tulpa::diagnostics()], the single source of
#' truth for convergence diagnostics in the tulpa ecosystem. Works with all
#' backends.
#'
#' @param fit A ratiod_fit object
#' @param pars Character vector of parameter names (default: all).
#'
#' @return A data frame with columns: parameter, rhat, ess_bulk, ess_tail.
#'
#' @details
#' Rhat is the improved value of Vehtari et al. (2021): the maximum of
#' rank-normalized split-Rhat and folded split-Rhat. Split-Rhat is defined for a
#' single chain (the chain is split in half), so single-chain fits also get a
#' value.
#'
#' @param ... Ignored.
#' @exportS3Method tulpa::diagnostics
diagnostics.ratiod_fit <- function(fit, pars = NULL, ...) {
  # Get draws in array format
  draws_info <- get_draws_array(fit)
  draws_array <- draws_info$draws

  # Select parameters
  all_pars <- dimnames(draws_array)[[3]]
  if (!is.null(pars)) {
    pars <- grep_params(pars, all_pars)
    draws_array <- draws_array[, , pars, drop = FALSE]
  }

  # The engine reads the [iteration, chain, parameter] layout directly and
  # computes each statistic per parameter.
  result <- tulpa::diagnostics(list(draws = draws_array))
  if (is.null(result)) return(NULL)

  class(result) <- c("ratiod_diagnostics", "data.frame")
  return(result)
}

#' Print method for ratiod_diagnostics
#'
#' @param x A ratiod_diagnostics object
#' @param ... Ignored
#'
#' @export
print.ratiod_diagnostics <- function(x, ...) {
  cat("MCMC Diagnostics\n")
  cat("================\n\n")

  # Format for printing. A column subset of a diagnostics table keeps this
  # class, so round only the columns that are present.
  if (!is.null(x$rhat))     x$rhat     <- round(x$rhat, 3)
  if (!is.null(x$ess_bulk)) x$ess_bulk <- round(x$ess_bulk, 0)
  if (!is.null(x$ess_tail)) x$ess_tail <- round(x$ess_tail, 0)

  print.data.frame(x, row.names = FALSE)

  # Warnings
  bad_rhat <- x$rhat > 1.01
  low_ess <- x$ess_bulk < 400 | x$ess_tail < 400

  if (any(bad_rhat, na.rm = TRUE)) {
    cat("\nWarning: ", sum(bad_rhat, na.rm = TRUE),
        " parameter(s) have Rhat > 1.01\n")
  }
  if (any(low_ess, na.rm = TRUE)) {
    cat("Warning: ", sum(low_ess, na.rm = TRUE),
        " parameter(s) have ESS < 400\n")
  }

  invisible(x)
}


#' Check MCMC diagnostics and print warnings
#'
#' @description
#' Checks MCMC diagnostics (Rhat, ESS, divergences) and prints actionable
#' warnings with suggestions for fixing issues.
#'
#' @param fit A ratiod_fit object
#' @param quiet Logical; if TRUE, suppress output and just return status.
#'
#' @return Invisibly returns a list with diagnostic status:
#' \describe{
#'   \item{ok}{Logical; TRUE if all diagnostics pass}
#'   \item{n_divergent}{Number of divergent transitions}
#'   \item{n_bad_rhat}{Number of parameters with Rhat > 1.01}
#'   \item{n_low_ess}{Number of parameters with ESS < 400}
#' }
#'
#' @export
check_diagnostics <- function(fit, quiet = FALSE) {
  if (!inherits(fit, "ratiod_fit")) {
    stop("fit must be a ratiod_fit object", call. = FALSE)
  }

  backend <- fit$backend %||% "unknown"
  issues <- list()

  # Check divergences (HMC only)
  n_divergent <- 0
  if (backend == "hmc") {
    diag <- fit$diagnostics
    if (!is.null(diag)) {
      n_divergent <- diag$n_divergent %||% 0
    }
  }

  # Check Rhat and ESS
  n_bad_rhat <- 0
  n_low_ess <- 0

  if (backend %in% c("hmc", "pg", "sghmc", "sgld", "ess")) {
    mcmc_diag <- tryCatch(
      diagnostics(fit, pars = NULL),
      error = function(e) NULL
    )

    if (!is.null(mcmc_diag)) {
      # Filter to main parameters only
      main_pars <- select_main_params(mcmc_diag$parameter)
      mcmc_diag <- mcmc_diag[mcmc_diag$parameter %in% main_pars, ]

      bad_rhat <- !is.na(mcmc_diag$rhat) & mcmc_diag$rhat > 1.01
      n_bad_rhat <- sum(bad_rhat)

      low_ess <- (!is.na(mcmc_diag$ess_bulk) & mcmc_diag$ess_bulk < 400) |
                 (!is.na(mcmc_diag$ess_tail) & mcmc_diag$ess_tail < 400)
      n_low_ess <- sum(low_ess)
    }
  }

  # Determine overall status
  ok <- (n_divergent == 0) && (n_bad_rhat == 0) && (n_low_ess == 0)

  # Print warnings if not quiet
  if (!quiet) {
    cat("Diagnostic Check\n")
    cat("================\n\n")

    if (ok) {
      cat("All diagnostics OK.\n")
    } else {
      if (n_divergent > 0) {
        cat("DIVERGENCES: ", n_divergent, " divergent transition(s) detected.\n")
        cat("  This indicates the sampler encountered regions of high curvature.\n")
        cat("  Suggestions:\n")
        cat("    - Increase iter and warmup\n")
        cat("    - Try different initialization (seed)\n")
        cat("    - Consider reparameterization\n")
        cat("    - For spatial models, check adjacency structure\n\n")
      }

      if (n_bad_rhat > 0) {
        cat("RHAT: ", n_bad_rhat, " parameter(s) have Rhat > 1.01.\n")
        cat("  This indicates chains have not converged.\n")
        cat("  Suggestions:\n")
        cat("    - Run more iterations\n")
        cat("    - Run more chains\n")
        cat("    - Check for multimodality with plot(fit)\n\n")
      }

      if (n_low_ess > 0) {
        cat("ESS: ", n_low_ess, " parameter(s) have ESS < 400.\n")
        cat("  This indicates high autocorrelation in samples.\n")
        cat("  Suggestions:\n")
        cat("    - Run more iterations\n")
        cat("    - Increase thinning interval\n")
        cat("    - Consider reparameterization\n\n")
      }
    }

    if (backend == "laplace") {
      cat("Note: Laplace backend uses approximate inference.\n")
      cat("      Consider HMC backend for full uncertainty quantification.\n")
    }
  }

  result <- list(
    ok = ok,
    n_divergent = n_divergent,
    n_bad_rhat = n_bad_rhat,
    n_low_ess = n_low_ess
  )

  invisible(result)
}


#' Get number of divergent transitions
#'
#' @param fit A ratiod_fit object
#' @return Integer count of divergent transitions
#' @export
n_divergent <- function(fit) {
  if (!inherits(fit, "ratiod_fit")) {
    stop("fit must be a ratiod_fit object", call. = FALSE)
  }

  if (fit$backend == "hmc") {
    diag <- fit$diagnostics
    if (!is.null(diag)) {
      return(diag$n_divergent %||% 0)
    }
  }

  return(0)
}
