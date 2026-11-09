#' Inflation specification implied by a model family
#' @keywords internal
family_inflation_spec <- function(family) {
  if (!isTRUE(family$zero_inflated) && !isTRUE(family$one_inflated)) return(NULL)
  type <- switch(family$numerator$distribution,
    zero_inflated_poisson = "zi_poisson",
    zero_inflated_neg_binomial = "zi_negbin",
    zero_inflated_binomial = "zi_binomial",
    hurdle_poisson = "hurdle_poisson",
    hurdle_neg_binomial = "hurdle_negbin",
    hurdle_binomial = "hurdle_binomial",
    one_inflated_binomial = "oi_binomial",
    zero_one_inflated_binomial = "zoib"
  )
  if (is.null(type)) {
    stop("Unknown inflated numerator: ", family$numerator$distribution, call. = FALSE)
  }
  structure(list(type = type,
                 formula = if (type == "oi_binomial") family$oi_formula else family$zi_formula,
                 oi_formula = family$oi_formula,
                 distribution = switch(family$numerator$base_distribution,
                                       neg_binomial_2 = "negbin",
                                       family$numerator$base_distribution)),
            class = "ratiod_zi")
}


#' Print an inflation specification
#' @keywords internal
#' @noRd
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

  cat_title("tulpaRatio Zero-Inflation Specification", blank = FALSE)
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
    stop("`zi` must be a ratiod_zi object derived from an inflated family", call. = FALSE)
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
