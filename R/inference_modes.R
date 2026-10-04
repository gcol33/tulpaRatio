#' Inference Mode System
#'
#' @description
#' tulpaRatio uses an explicit tier system for inference that encodes
#' **epistemic guarantees**, not just runtime characteristics.
#'
#' This design makes the difference between inference methods **first-class
#' and unavoidable**, rather than hiding them as implementation details.
#'
#' @section The Three Tiers:
#'
#' **Tier 1 - Exact:**
#' Asymptotically correct posterior inference (up to Monte Carlo error).
#' Credible intervals are interpretable as posterior uncertainty.
#' This is the reference standard.
#'
#' **Tier 2 - Structured:**
#' Accurate inference *conditional on explicit structural assumptions*.
#' Typically requires latent Gaussian structure, conditional independence,
#' smooth posteriors. Very fast for the right model class, but can be wrong
#' outside that class. Failure modes are predictable and explainable.
#'
#' **Tier 3 - Optimized:**
#' No general correctness guarantee beyond empirical usefulness.
#' Point estimates usually good, but uncertainty often underestimated.
#' Tails and correlations unreliable. Failure is usually silent.
#' This is optimization, not sampling.
#'
#' @section Auto Mode:
#'
#' `mode = "auto"` chooses between Tier 1 and Tier 2 only.
#' It will **never** silently choose Tier 3 (Optimized).
#'
#' The contract: "Use the most reliable method that is expected to finish
#' for this model."
#'
#' Auto decisions are deterministic, explainable, and overrideable.
#'
#' @section Implementation Rules:
#'
#' 1. Modes change semantics, not just runtime.
#'    Intervals from Optimized do not mean the same as Exact.
#'
#' 2. The mode must always be visible in output.
#'
#' 3. No silent upgrading or downgrading.
#'    If Exact fails, we error - we do not switch to Structured.
#'
#' 4. Backends slot into tiers.
#'    Adding a backend never introduces a new epistemic promise.
#'
#' @name inference_modes
#' @keywords internal
NULL


# ==============================================================================
# Tier Definitions
# ==============================================================================

#' Inference tiers with their properties
#' @keywords internal
INFERENCE_TIERS <- list(
  exact = list(
    tier = 1L,
    name = "Exact",
    description = "Asymptotically correct posterior inference",
    guarantee = "Credible intervals interpretable as posterior uncertainty",
    backends = c("hmc", "ess", "pg", "gibbs", "sghmc", "sgld"),
    note = "Reference standard"
  ),
  structured = list(
    tier = 2L,
    name = "Structured",
    description = "Accurate inference conditional on structural assumptions",
    guarantee = "Correct if model meets structural assumptions",
    backends = c("laplace"),
    note = "Controlled approximation, not heuristics"
  ),
  optimized = list(
    tier = 3L,
    name = "Optimized",
    description = "Optimization-based approximate inference",
    guarantee = "No general correctness guarantee",
    backends = c("vi"),
    note = "Uncertainty often underestimated; requires explicit opt-in"
  )
)


#' Get tier for a backend
#' @param backend Character string naming the backend
#' @return List with tier information
#' @keywords internal
get_backend_tier <- function(backend) {
  for (tier_name in names(INFERENCE_TIERS)) {
    tier <- INFERENCE_TIERS[[tier_name]]
    if (backend %in% tier$backends) {
      return(list(
        tier = tier$tier,
        name = tier$name,
        mode = tier_name,
        description = tier$description,
        guarantee = tier$guarantee,
        note = tier$note
      ))
    }
  }
  stop(sprintf("Unknown backend: '%s'", backend), call. = FALSE)
}


#' Map mode to valid backends
#' @param mode Character: "auto", "exact", "structured", or "optimized"
#' @return Character vector of valid backends for this mode
#' @keywords internal
get_mode_backends <- function(mode) {
  mode <- tolower(mode)

  if (mode == "auto") {
    # Auto can choose Tier 1 or Tier 2, never Tier 3
    return(c(INFERENCE_TIERS$exact$backends, INFERENCE_TIERS$structured$backends))
  }

  if (mode %in% names(INFERENCE_TIERS)) {
    return(INFERENCE_TIERS[[mode]]$backends)
  }

  stop(sprintf(
    "Unknown mode: '%s'. Use one of: auto, exact, structured, optimized",
    mode
  ), call. = FALSE)
}


# ==============================================================================
# Backend Capabilities
# ==============================================================================

#' Model structures a fit can carry
#'
#' @description
#' The `tratio()` arguments that put structure into a model, in the order
#' `unsupported_structures()` reports them.
#'
#' @keywords internal
MODEL_STRUCTURES <- c("spatial", "temporal", "spatiotemporal", "zi", "latent")


#' Temporal support rule for the Gibbs sampler
#'
#' @param temporal The temporal specification
#' @param structures Named list of all structure specifications, for rules that
#'   depend on what else is in the model
#' @return `TRUE`, or a phrase naming the restriction
#' @keywords internal
tvc_temporal_support <- function(temporal, structures) {
  if (inherits(temporal, "ratiod_tvc")) return(TRUE)
  paste0("only time-varying coefficients are implemented; ",
         "build the term with temporal_tvc()")
}


#' Temporal support rule for the Polya-Gamma sampler
#'
#' @description
#' The sampler holds one multi-scale temporal fit and dispatches on spatial
#' structure ahead of temporal, so the temporal fit is out of reach once a
#' spatial field is in the model.
#'
#' @inheritParams tvc_temporal_support
#' @return `TRUE`, or a phrase naming the restriction
#' @keywords internal
multiscale_temporal_support <- function(temporal, structures) {
  if (!inherits(temporal, "ratiod_temporal_multiscale")) {
    return(paste0("only multi-scale temporal is implemented; ",
                  "build the term with temporal_multiscale()"))
  }
  if (!is.null(structures$spatial)) {
    return("a temporal field and a spatial field cannot share one fit")
  }
  TRUE
}


#' Temporal support rule for the Laplace backend
#'
#' @description
#' The trend, seasonal and short-term components of a multi-scale temporal term
#' are latent blocks of the nested-Laplace engine, and they sit beside an areal
#' or Gaussian-process field in one fit.
#'
#' @inheritParams tvc_temporal_support
#' @return `TRUE`, or a phrase naming the restriction
#' @keywords internal
laplace_temporal_support <- function(temporal, structures) {
  if (!inherits(temporal, "ratiod_temporal_multiscale")) {
    return(paste0("only multi-scale temporal is implemented; ",
                  "build the term with temporal_multiscale()"))
  }
  TRUE
}


#' Spatial support rule for the Laplace backend
#'
#' @description
#' The Laplace backend hands every field to the nested-Laplace engine as a
#' latent block: ICAR, BYM2, proper CAR, and Gaussian-process fields on a
#' Hilbert-space basis (single scale, two scales), each also under restricted
#' spatial regression. Spatially varying coefficients carry no block and name
#' themselves rather than reaching a dispatch that would fit the model without
#' the field.
#'
#' @inheritParams tvc_temporal_support
#' @param spatial The spatial specification
#' @return `TRUE`, or a phrase naming the restriction
#' @keywords internal
laplace_spatial_support <- function(spatial, structures) {
  if (inherits(spatial, "ratiod_svc") || identical(spatial$type, "svc")) {
    return("spatially varying coefficients have no latent block on the nested-Laplace engine")
  }
  TRUE
}


#' Which structures each backend can fit
#'
#' @description
#' One entry per backend, one field per name in `MODEL_STRUCTURES`. A field is
#' `TRUE` (any specification of that structure), `FALSE` (none), or a rule
#' function of `(structure, structures)` returning `TRUE` or a phrase naming
#' the restriction.
#'
#' `auto_select_mode()` and the check `tratio()` runs before dispatch both read
#' this table, so a backend gains or loses a structure in one place.
#'
#' @keywords internal
BACKEND_STRUCTURE_SUPPORT <- list(
  hmc = list(
    spatial = TRUE, temporal = TRUE, spatiotemporal = TRUE,
    zi = TRUE, latent = TRUE
  ),
  ess = list(
    spatial = TRUE, temporal = TRUE, spatiotemporal = FALSE,
    zi = TRUE, latent = FALSE
  ),
  gibbs = list(
    spatial = TRUE, temporal = tvc_temporal_support, spatiotemporal = FALSE,
    zi = FALSE, latent = FALSE
  ),
  pg = list(
    spatial = TRUE, temporal = multiscale_temporal_support,
    spatiotemporal = FALSE, zi = FALSE, latent = FALSE
  ),
  sghmc = list(
    spatial = TRUE, temporal = TRUE, spatiotemporal = FALSE,
    zi = TRUE, latent = FALSE
  ),
  sgld = list(
    spatial = TRUE, temporal = TRUE, spatiotemporal = FALSE,
    zi = TRUE, latent = FALSE
  ),
  laplace = list(
    spatial = laplace_spatial_support, temporal = laplace_temporal_support,
    spatiotemporal = FALSE, zi = FALSE, latent = FALSE
  ),
  vi = list(
    spatial = TRUE, temporal = TRUE, spatiotemporal = FALSE,
    zi = TRUE, latent = FALSE
  )
)


#' Structures a backend cannot fit
#'
#' @param backend Character string naming the backend
#' @param structures Named list of structure specifications, `NULL` where the
#'   structure is absent
#' @return A named character vector holding one reason per structure the
#'   backend cannot carry, empty when the backend fits the whole model
#' @keywords internal
unsupported_structures <- function(backend, structures) {
  support <- BACKEND_STRUCTURE_SUPPORT[[backend]]
  if (is.null(support)) {
    stop(sprintf("Unknown backend: '%s'", backend), call. = FALSE)
  }

  reasons <- character(0)
  for (nm in MODEL_STRUCTURES) {
    if (is.null(structures[[nm]])) next
    rule <- support[[nm]]
    verdict <- if (is.function(rule)) rule(structures[[nm]], structures) else rule
    if (isTRUE(verdict)) next
    reasons[[nm]] <- if (is.character(verdict)) verdict else "not implemented"
  }

  reasons
}


#' Whether a backend can fit every structure in a model
#'
#' @inheritParams unsupported_structures
#' @return `TRUE` or `FALSE`
#' @keywords internal
backend_fits_structures <- function(backend, structures) {
  length(unsupported_structures(backend, structures)) == 0L
}


#' Stop unless the backend can fit every structure in the model
#'
#' @description
#' A backend that cannot carry a structure fits the model without it and
#' reports success, so the model is refused here and the structure named.
#'
#' @inheritParams unsupported_structures
#' @return `invisible(TRUE)`, or an error naming each structure
#' @keywords internal
assert_backend_fits_structures <- function(backend, structures) {
  reasons <- unsupported_structures(backend, structures)
  if (length(reasons) == 0L) return(invisible(TRUE))

  named <- sprintf("`%s`", names(reasons))

  stop(sprintf(
    "The %s backend cannot fit this model:\n%s\nUse `mode = \"hmc\"`, which fits every structure, or drop %s.",
    backend,
    paste(sprintf("  - %s: %s", named, reasons), collapse = "\n"),
    paste(named, collapse = " and ")
  ), call. = FALSE)
}


# ==============================================================================
# Mode Selection Logic
# ==============================================================================

#' Select inference mode and backend
#'
#' @description
#' Implements the mode selection logic for tulpaRatio. Accepts either tier names
#' (auto, exact, structured, optimized) or backend names (hmc, ess, pg, laplace, vi).
#'
#' When mode is "auto", selects between Tier 1 (Exact) and Tier 2 (Structured)
#' based on model characteristics. Never selects Tier 3 (Optimized) automatically.
#'
#' @param mode User-specified mode or backend name
#' @param family Model family object
#' @param n_obs Number of observations
#' @param structures Named list of structure specifications, `NULL` where the
#'   structure is absent. See `MODEL_STRUCTURES`.
#'
#' @return List with:
#'   - mode: The selected mode name
#'   - backend: The selected backend
#'   - tier: The tier number (1, 2, or 3)
#'   - tier_name: The tier name
#'   - reason: Explanation for the selection
#'
#' @keywords internal
select_inference_mode <- function(mode,
                                  family,
                                  n_obs,
                                  structures = list()) {

  mode <- tolower(mode)

  # Check if mode is actually a backend name
  all_backends <- c("hmc", "ess", "pg", "gibbs", "sghmc", "sgld", "laplace", "vi")
  if (mode %in% all_backends) {
    # User specified a backend directly
    backend <- mode
    tier_info <- get_backend_tier(backend)

    return(list(
      mode = tier_info$mode,
      backend = backend,
      tier = tier_info$tier,
      tier_name = tier_info$name,
      reason = sprintf("User-specified backend: %s", backend)
    ))
  }

  # Validate tier name

  valid_modes <- c("auto", "exact", "structured", "optimized")
  if (!mode %in% valid_modes) {
    stop(sprintf(
      "Invalid mode: '%s'. Use one of: %s, or a backend: %s",
      mode,
      paste(valid_modes, collapse = ", "),
      paste(all_backends, collapse = ", ")
    ), call. = FALSE)
  }

  # Auto-selection logic
  if (mode == "auto") {
    return(auto_select_mode(family, n_obs, structures))
  }

  # Explicit tier mode: select best backend within that tier
  backend <- select_backend_for_mode(mode)
  tier_info <- get_backend_tier(backend)

  return(list(
    mode = mode,
    backend = backend,
    tier = tier_info$tier,
    tier_name = tier_info$name,
    reason = sprintf("User-specified mode: %s", mode)
  ))
}


#' Auto-select mode (Tier 1 or Tier 2 only)
#'
#' @description
#' Implements the "auto" mode selection. Chooses the most reliable method
#' that is expected to finish for the given model.
#'
#' **Critical rule**: Auto never selects Tier 3 (Optimized).
#'
#' A backend is a candidate only when it fits every structure in the model.
#' Structure a backend cannot carry would otherwise be dropped from the fit.
#'
#' @inheritParams select_inference_mode
#' @keywords internal
auto_select_mode <- function(family, n_obs, structures = list()) {

  # Thresholds
  VERY_LARGE <- 50000
  LARGE <- 10000

  has_spatial <- !is.null(structures$spatial)
  has_latent <- !is.null(structures$latent)
  spatial_type <- if (has_spatial) structures$spatial$type else NULL

  # Decision logic
  reason_parts <- character(0)

  # Check if Laplace (Tier 2) is appropriate
  use_structured <- FALSE

  if (n_obs > VERY_LARGE) {
    use_structured <- TRUE
    reason_parts <- c(reason_parts, sprintf("large dataset (n=%s)", format(n_obs, big.mark = ",")))
  }

  # Latent Gaussian models are ideal for Laplace
  if (has_spatial && !has_latent && n_obs > LARGE) {
    use_structured <- TRUE
    reason_parts <- c(reason_parts, "spatial model with large n")
  }

  if (use_structured && backend_fits_structures("laplace", structures)) {
    return(list(
      mode = "structured",
      backend = "laplace",
      tier = 2L,
      tier_name = "Structured",
      reason = paste(reason_parts, collapse = "; ")
    ))
  }

  # Gibbs for ICAR spatial models
  # Gibbs is ~18x faster than HMC for ICAR because it updates phi
  # component-wise, avoiding HMC's curse of dimensionality
  is_gibbs_spatial <- has_spatial && !is.null(spatial_type) &&
                      spatial_type %in% c("icar", "car", "bym2")
  if (is_gibbs_spatial && backend_fits_structures("gibbs", structures)) {
    # Check family is supported by Gibbs
    gibbs_families <- c("poisson_gamma", "negbin_negbin", "binomial",
                        "negbin_gamma", "poisson", "neg_binomial_2",
                        "negative_binomial", "negbin")
    fam_name <- family$name %||% ""
    fam_dist <- family$numerator$distribution %||% ""
    if (fam_name %in% gibbs_families || fam_dist %in% gibbs_families ||
        fam_dist == "binomial") {
      return(list(
        mode = "exact",
        backend = "gibbs",
        tier = 1L,
        tier_name = "Exact",
        reason = sprintf("ICAR/BYM2 spatial model (Gibbs %s)", spatial_type)
      ))
    }
  }

  # Default: Tier 1 (Exact) with HMC
  # HMC is the most general and robust
  backend <- "hmc"
  reason <- "default (full MCMC)"

  return(list(
    mode = "exact",
    backend = backend,
    tier = 1L,
    tier_name = "Exact",
    reason = reason
  ))
}


#' Select best backend within a mode
#' @inheritParams select_inference_mode
#' @keywords internal
select_backend_for_mode <- function(mode) {

  if (mode == "exact") {
    # HMC is the default for Exact - most general
    return("hmc")
  }

  if (mode == "structured") {
    # Laplace is currently the only Tier 2 backend
    return("laplace")
  }

  if (mode == "optimized") {
    # VI is currently the only Tier 3 backend
    return("vi")
  }

  # Fallback
  return("hmc")
}


# ==============================================================================
# Mode Validation
# ==============================================================================

#' Validate that a fit used the expected mode
#'
#' @description
#' Ensures the fit object was created with the expected inference mode.
#' Useful for enforcing mode requirements in downstream analysis.
#'
#' @param fit A ratiod_fit object
#' @param expected_mode Mode that should have been used
#' @param error If TRUE (default), error on mismatch. If FALSE, return logical.
#'
#' @return If error = FALSE, returns TRUE/FALSE. Otherwise errors on mismatch.
#'
#' @export
validate_mode <- function(fit, expected_mode, error = TRUE) {
  if (!inherits(fit, "ratiod_fit")) {
    stop("fit must be a ratiod_fit object", call. = FALSE)
  }

  actual_mode <- fit$inference_mode %||% "unknown"
  actual_tier <- fit$inference_tier %||% NA

  expected_mode <- tolower(expected_mode)

  # Check if modes match
  match <- (actual_mode == expected_mode)

  # Also accept tier-based matching
 if (!match && expected_mode %in% names(INFERENCE_TIERS)) {
    expected_tier <- INFERENCE_TIERS[[expected_mode]]$tier
    match <- !is.na(actual_tier) && (actual_tier == expected_tier)
  }

  if (!match && error) {
    stop(sprintf(
      "Mode mismatch: fit used '%s' (Tier %s) but expected '%s'.\n%s",
      actual_mode,
      ifelse(is.na(actual_tier), "?", actual_tier),
      expected_mode,
      "Refit the model with the correct mode, or adjust your expectations."
    ), call. = FALSE)
  }

  return(match)
}


# ==============================================================================
# User-Facing Information
# ==============================================================================
#' Print inference mode information
#'
#' @description
#' Displays information about the inference modes available in tulpaRatio,
#' including their tiers, guarantees, and appropriate use cases.
#'
#' @export
inference_mode_info <- function() {
  cat("tulpaRatio Inference Modes\n")
  cat("========================\n\n")

  cat("Modes encode EPISTEMIC GUARANTEES, not just runtime characteristics.\n\n")

  for (tier_name in c("exact", "structured", "optimized")) {
    tier <- INFERENCE_TIERS[[tier_name]]

    cat(sprintf("TIER %d: %s\n", tier$tier, toupper(tier$name)))
    cat(sprintf("  %s\n", tier$description))
    cat(sprintf("  Guarantee: %s\n", tier$guarantee))
    cat(sprintf("  Backends: %s\n", paste(tier$backends, collapse = ", ")))
    cat(sprintf("  Note: %s\n", tier$note))
    cat("\n")
  }

  cat("AUTO MODE\n")
  cat("  Selects between Tier 1 and Tier 2 only.\n")
  cat("  NEVER silently chooses Tier 3 (Optimized).\n")
  cat("  Contract: 'Use the most reliable method expected to finish.'\n\n")

  cat("Usage (by tier):\n")
  cat("  tratio(..., mode = 'auto')       # Tier 1 or 2 (default)\n")
  cat("  tratio(..., mode = 'exact')      # Tier 1\n")
  cat("  tratio(..., mode = 'structured') # Tier 2\n")
  cat("  tratio(..., mode = 'optimized')  # Tier 3 (explicit opt-in)\n\n")

  cat("Usage (by backend):\n")
  cat("  tratio(..., mode = 'hmc')        # HMC/NUTS (Tier 1)\n")
  cat("  tratio(..., mode = 'ess')        # Elliptical Slice Sampling (Tier 1)\n")
  cat("  tratio(..., mode = 'sghmc')      # Stochastic Gradient HMC (Tier 1, large N)\n")
  cat("  tratio(..., mode = 'sgld')       # Stochastic Gradient Langevin (Tier 1, large N)\n")
  cat("  tratio(..., mode = 'laplace')    # Laplace approximation (Tier 2)\n")
  cat("  tratio(..., mode = 'vi')         # Variational Inference (Tier 3)\n")

  invisible(NULL)
}


#' Format tier information for display
#'
#' @param tier_info List from get_backend_tier()
#' @param verbose Include full description
#' @return Character string
#' @keywords internal
format_tier_info <- function(tier_info, verbose = FALSE) {
  if (verbose) {
    sprintf(
      "Tier %d (%s): %s",
      tier_info$tier,
      tier_info$name,
      tier_info$description
    )
  } else {
    sprintf("Tier %d (%s)", tier_info$tier, tier_info$name)
  }
}


#' Format mode selection message
#'
#' @param selection List from select_inference_mode()
#' @return Character string for display
#' @keywords internal
format_mode_selection <- function(selection) {
  lines <- c(
    sprintf("Mode: %s", selection$tier_name),
    sprintf("Backend: %s", selection$backend),
    sprintf("Reason: %s", selection$reason)
  )
  paste(lines, collapse = "\n")
}
