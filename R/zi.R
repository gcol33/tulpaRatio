#' Zero-Inflation and Hurdle Model Specifications
#'
#' @description
#' Functions to specify zero-inflation or hurdle model components for ratio models.
#' These handle excess zeros in count data where the standard count distribution
#' (Poisson, negative binomial) doesn't adequately capture the zero proportion.
#'
#' **Note on naming convention:** These functions (`zi_poisson()`, `zi_negbin()`,
#' `hurdle_poisson()`, `hurdle_negbin()`) are **ZI specification** functions for
#' the `zi` argument in [tratio()]. They are distinct from the **family functions**
#' (`ratiod_zipois()`, `ratiod_zinegbin()`, `ratiod_hurdle_pois()`, `ratiod_hurdle_negbin()`)
#' which specify the complete model family. Use ZI specifications when you want to
#' add zero-inflation to an existing family; use ZI family functions when you want
#' a pre-configured zero-inflated family.
#'
#' @details
#' **Zero-Inflated Models:**
#' Zero-inflated models assume two data-generating processes:
#' 1. A process that generates "structural zeros" with probability \eqn{\pi}
#' 2. A count process that generates counts (including zeros) with probability \eqn{1-\pi}
#'
#' The probability of observing y is:
#' \deqn{P(Y=0) = \pi + (1-\pi) \cdot P_{count}(0)}
#' \deqn{P(Y=y|y>0) = (1-\pi) \cdot P_{count}(y)}
#'
#' **Hurdle Models:**
#' Hurdle models treat the process as:
#' 1. A binary process determining zero vs non-zero
#' 2. A truncated count process (conditional on y > 0)
#'
#' \deqn{P(Y=0) = 1 - \theta}
#' \deqn{P(Y=y|y>0) = \theta \cdot P_{count}(y|y>0)}
#'
#' **When to use:**
#' - `zi_poisson()`: Excess zeros with equidispersed positive counts
#' - `zi_negbin()`: Excess zeros with overdispersed positive counts
#' - `hurdle_poisson()`: Zero hurdle with equidispersed positive counts
#' - `hurdle_negbin()`: Zero hurdle with overdispersed positive counts
#'
#' @examples
#' zi_poisson()
#' hurdle_negbin()
#' @name zi_specification
#' @keywords models
NULL


#' Zero-Inflated Poisson Specification
#'
#' @param formula Optional formula for the zero-inflation probability.
#'   If NULL, uses intercept-only model for P(structural zero).
#'
#' @return A `ratiod_zi` object
#'
#' @examples
#' # Intercept-only ZI
#' zi_poisson()
#'
#' # ZI probability depends on a covariate
#' zi_poisson(~ habitat)
#'
#' @export
zi_poisson <- function(formula = NULL) {
  structure(
    list(
      type = "zi_poisson",
      formula = formula,
      distribution = "poisson"
    ),
    class = "ratiod_zi"
  )
}


#' Zero-Inflated Negative Binomial Specification
#'
#' @param formula Optional formula for the zero-inflation probability.
#'   If NULL, uses intercept-only model for P(structural zero).
#'
#' @return A `ratiod_zi` object
#'
#' @examples
#' # Intercept-only ZI
#' zi_negbin()
#'
#' # ZI probability depends on multiple covariates
#' zi_negbin(~ habitat + season)
#'
#' @export
zi_negbin <- function(formula = NULL) {
  structure(
    list(
      type = "zi_negbin",
      formula = formula,
      distribution = "negbin"
    ),
    class = "ratiod_zi"
  )
}


#' Hurdle Poisson Specification
#'
#' @param formula Optional formula for the hurdle probability (P(Y > 0)).
#'   If NULL, uses intercept-only model.
#'
#' @return A `ratiod_zi` object
#'
#' @examples
#' # Intercept-only hurdle
#' hurdle_poisson()
#'
#' # Hurdle probability depends on effort
#' hurdle_poisson(~ effort)
#'
#' @export
hurdle_poisson <- function(formula = NULL) {
  structure(
    list(
      type = "hurdle_poisson",
      formula = formula,
      distribution = "poisson"
    ),
    class = "ratiod_zi"
  )
}


#' Zero-Inflated Binomial Specification
#'
#' @param formula Optional formula for the zero-inflation probability.
#'   If NULL, uses intercept-only model for P(structural zero).
#'
#' @return A `ratiod_zi` object
#'
#' @examples
#' # Intercept-only ZI
#' zi_binomial()
#'
#' # ZI probability depends on a covariate
#' zi_binomial(~ habitat)
#'
#' @export
zi_binomial <- function(formula = NULL) {
  structure(
    list(
      type = "zi_binomial",
      formula = formula,
      distribution = "binomial"
    ),
    class = "ratiod_zi"
  )
}


#' Hurdle Binomial Specification
#'
#' @param formula Optional formula for the hurdle probability (P(Y > 0)).
#'   If NULL, uses intercept-only model.
#'
#' @return A `ratiod_zi` object
#'
#' @examples
#' # Intercept-only hurdle
#' hurdle_binomial()
#'
#' @export
hurdle_binomial <- function(formula = NULL) {
  structure(
    list(
      type = "hurdle_binomial",
      formula = formula,
      distribution = "binomial"
    ),
    class = "ratiod_zi"
  )
}


#' One-Inflated Binomial Specification
#'
#' @param formula Optional formula for the one-inflation probability
#'   (excess at the upper boundary, y = n). If NULL, uses intercept-only model.
#'
#' @return A `ratiod_zi` object
#'
#' @examples
#' # Intercept-only OI
#' oi_binomial()
#'
#' @export
oi_binomial <- function(formula = NULL) {
  structure(
    list(
      type = "oi_binomial",
      formula = formula,
      distribution = "binomial"
    ),
    class = "ratiod_zi"
  )
}


#' Zero-and-One-Inflated Binomial (ZOIB) Specification
#'
#' @param zi_formula Optional formula for the zero-inflation probability.
#'   If NULL, uses intercept-only model.
#' @param oi_formula Optional formula for the one-inflation probability.
#'   If NULL, uses intercept-only model.
#'
#' @return A `ratiod_zi` object
#'
#' @examples
#' # Intercept-only for both zero and one inflation
#' zoib()
#'
#' @export
zoib <- function(zi_formula = NULL, oi_formula = NULL) {
  structure(
    list(
      type = "zoib",
      formula = zi_formula,
      oi_formula = oi_formula,
      distribution = "binomial"
    ),
    class = "ratiod_zi"
  )
}


#' Hurdle Negative Binomial Specification
#'
#' @param formula Optional formula for the hurdle probability (P(Y > 0)).
#'   If NULL, uses intercept-only model.
#'
#' @return A `ratiod_zi` object
#'
#' @examples
#' # Intercept-only hurdle
#' hurdle_negbin()
#'
#' # Hurdle probability depends on sampling effort
#' hurdle_negbin(~ log_effort)
#'
#' @export
hurdle_negbin <- function(formula = NULL) {
  structure(
    list(
      type = "hurdle_negbin",
      formula = formula,
      distribution = "negbin"
    ),
    class = "ratiod_zi"
  )
}


#' @rdname zi_specification
#' @param x A ratiod_zi object
#' @param ... Additional arguments (ignored)
#'
#' @return Invisibly returns the input
#' @export
print.ratiod_zi <- function(x, ...) {
  type_label <- switch(x$type,
    "zi_poisson" = "Zero-Inflated Poisson",
    "zi_negbin" = "Zero-Inflated Negative Binomial",
    "hurdle_poisson" = "Hurdle Poisson",
    "hurdle_negbin" = "Hurdle Negative Binomial",
    "zi_binomial" = "Zero-Inflated Binomial",
    "hurdle_binomial" = "Hurdle Binomial",
    "oi_binomial" = "One-Inflated Binomial",
    "zoib" = "Zero-and-One-Inflated Binomial",
    x$type
  )

  cat("tulpaRatio Zero-Inflation Specification\n")
  cat("==================================\n")
  cat("Type:", type_label, "\n")

  if (identical(x$type, "zoib")) {
    if (is.null(x$formula)) {
      cat("Zero-inflation formula: ~ 1 (intercept only)\n")
    } else {
      cat("Zero-inflation formula:", deparse(x$formula), "\n")
    }
    if (is.null(x$oi_formula)) {
      cat("One-inflation formula: ~ 1 (intercept only)\n")
    } else {
      cat("One-inflation formula:", deparse(x$oi_formula), "\n")
    }
  } else if (is.null(x$formula)) {
    cat("Formula: ~ 1 (intercept only)\n")
  } else {
    cat("Formula:", deparse(x$formula), "\n")
  }

  invisible(x)
}


#' Validate ZI specification
#'
#' @param zi A ratiod_zi object or NULL
#' @param data Data frame for checking formula variables
#'
#' @return Validated ratiod_zi object or NULL
#' @keywords internal
validate_zi <- function(zi, data = NULL) {
  if (is.null(zi)) {
    return(NULL)
  }

  if (!inherits(zi, "ratiod_zi")) {
    stop("zi must be a ratiod_zi object (created by zi_poisson(), zi_negbin(), ",
         "hurdle_poisson(), hurdle_negbin(), zi_binomial(), hurdle_binomial(), ",
         "oi_binomial(), or zoib())", call. = FALSE)
  }

  valid_types <- setdiff(cpp_enum_names("zi"), "none")
  if (!zi$type %in% valid_types) {
    stop("Unknown ZI type: ", zi$type, call. = FALSE)
  }

  # Validate formula variables exist in data
  if (!is.null(zi$formula) && !is.null(data)) {
    formula_vars <- all.vars(zi$formula)
    missing_vars <- setdiff(formula_vars, names(data))
    if (length(missing_vars) > 0) {
      stop("ZI formula contains variables not in data: ",
           paste(missing_vars, collapse = ", "), call. = FALSE)
    }
  }

  if (identical(zi$type, "zoib") && !is.null(zi$oi_formula) && !is.null(data)) {
    formula_vars <- all.vars(zi$oi_formula)
    missing_vars <- setdiff(formula_vars, names(data))
    if (length(missing_vars) > 0) {
      stop("ZOIB one-inflation formula contains variables not in data: ",
           paste(missing_vars, collapse = ", "), call. = FALSE)
    }
  }

  zi
}


#' Determine the ZI/hurdle/OI/ZOIB type string and design-matrix shape
#' implied by a ZI/hurdle *family* (e.g. `ratiod_zinegbin()`), independent
#' of any explicit `zi = ` argument.
#'
#' This is the single source of truth for family-based ZI auto-detection,
#' shared by every backend (HMC, VI, ESS, SGHMC) so the specific
#' `"zi_poisson"`/`"zi_negbin"`/`"zi_binomial"`/etc. string always reaches
#' the C++ dispatch, instead of a bare `"zi"`/`"hurdle"` collapsing every
#' variant to the same (or wrong) likelihood.
#'
#' @param family A `ratiod_family` object with `zero_inflated`/`one_inflated`
#'   set (checked by the caller before calling this)
#' @param N Number of observations, for building placeholder design matrices
#'
#' @return A list with `type`, `X_zi`, `p_zi`, `coef_names`, `X_oi`, `p_oi`,
#'   `coef_names_oi`
#' @keywords internal
family_zi_info <- function(family, N) {
  dist <- family$numerator$distribution

  zi_type <- if (family$zi_type == "hurdle") {
    switch(dist,
      "hurdle_binomial" = "hurdle_binomial",
      "hurdle_poisson" = "hurdle_poisson",
      "hurdle_neg_binomial" = "hurdle_negbin"
    )
  } else if (family$zi_type == "one_inflated") {
    # One-inflated models (excess at upper boundary)
    "oi_binomial"
  } else if (family$zi_type == "zoib") {
    # Zero-and-one-inflated binomial
    "zoib"
  } else {
    # Zero-inflated mixture models
    switch(dist,
      "zero_inflated_binomial" = "zi_binomial",
      "zero_inflated_poisson" = "zi_poisson",
      "zero_inflated_neg_binomial" = "zi_negbin"
    )
  }
  if (is.null(zi_type)) {
    stop(sprintf("No zero/one inflation structure for a %s numerator.", dist),
         call. = FALSE)
  }

  if (family$zi_type == "one_inflated") {
    # OI-binomial: only OI coefficient, no ZI
    list(
      type = zi_type,
      X_zi = matrix(0, nrow = N, ncol = 1),  # Placeholder (not used)
      p_zi = 0L,  # No ZI coefficient for OI-only models
      coef_names = NULL,
      X_oi = matrix(1, nrow = N, ncol = 1),  # Intercept-only
      p_oi = 1L,
      coef_names_oi = "(Intercept)_oi"
    )
  } else if (family$zi_type == "zoib") {
    # ZOIB: both ZI and OI coefficients
    list(
      type = zi_type,
      X_zi = matrix(1, nrow = N, ncol = 1),  # Intercept-only
      p_zi = 1L,
      coef_names = "(Intercept)_zi",
      X_oi = matrix(1, nrow = N, ncol = 1),  # Intercept-only
      p_oi = 1L,
      coef_names_oi = "(Intercept)_oi"
    )
  } else {
    # ZI/Hurdle models: only ZI coefficient
    list(
      type = zi_type,
      X_zi = matrix(1, nrow = N, ncol = 1),  # Intercept-only
      p_zi = 1L,
      coef_names = "(Intercept)_zi",
      X_oi = NULL,
      p_oi = 0L,
      coef_names_oi = NULL
    )
  }
}


#' Build a design matrix for a ZI/OI component from an optional formula
#' @keywords internal
build_zi_design_matrix <- function(formula, data, N, suffix) {
  if (is.null(formula) || identical(formula, ~1)) {
    X <- matrix(1, nrow = N, ncol = 1)
    colnames(X) <- paste0("(Intercept)", suffix)
  } else {
    X <- model.matrix(formula, data = data)
  }
  X
}
