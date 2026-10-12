#' Parse and validate tulpaRatio formulas
#'
#' @description
#' Internal functions for parsing tulpaRatio formulas into a structured
#' specification for Stan model building.
#'
#' tulpaRatio supports a combined formula syntax:
#' - `num | denom ~ x + (1|group)` for shared predictors
#' - Separate `formula_num` and `formula_denom` for different predictors
#'
#' @name ratiod_formula
#' @keywords internal
NULL

#' Parse a tulpaRatio formula
#'
#' @description
#' Parses the main formula and optional process-specific formulas into
#' a structured specification. Supports both combined syntax
#' (`num | denom ~ x`) and separate formulas.
#'
#' @param formula Main formula. Can be:
#'   - Combined: `num | denom ~ predictors`
#'   - Numerator only: `num ~ predictors` (requires formula_denom)
#' @param formula_num Optional formula for numerator-specific predictors.
#'   Added to those from the main formula.
#' @param formula_denom Optional formula for denominator-specific predictors.
#'   Required if main formula has single response.
#' @param shared Formula for shared random effects. NULL = infer from formulas,
#'   `~ 0` = independence (triggers warning).
#' @param data Data frame containing all variables.
#'
#' @return A `ratiod_formula` object
#' @keywords internal
ratiod_formula <- function(formula,
                          formula_num = NULL,
                          formula_denom = NULL,
                          shared = NULL,
                          data) {

  # Parse the main formula
  parsed <- parse_main_formula(formula, data)

  # Handle combined vs separate formula specifications

  if (parsed$type == "combined") {
    # Combined syntax: num | denom ~ predictors
    num_response <- parsed$num_var
    denom_response <- parsed$denom_var
    base_rhs <- parsed$rhs

    # Build numerator formula
    num_formula <- build_process_formula(
      response = num_response,
      base_rhs = base_rhs,
      additional = formula_num,
      data = data
    )

    # Build denominator formula
    denom_formula <- build_process_formula(
      response = denom_response,
      base_rhs = base_rhs,
      additional = formula_denom,
      data = data
    )

  } else {
    # Single response: require formula_denom
    if (is.null(formula_denom))

      stop(
        "When using single-response formula, `formula_denom` is required.\n",
        "Either use combined syntax: num | denom ~ x\n",
        "Or provide both formula and formula_denom.",
        call. = FALSE
      )

    num_response <- parsed$response
    base_rhs <- parsed$rhs

    num_formula <- build_process_formula(
      response = num_response,
      base_rhs = base_rhs,
      additional = formula_num,
      data = data
    )

    denom_formula <- parse_denom_formula(formula_denom, data)
    denom_response <- denom_formula$response_var
  }

  # Validate - no offsets allowed
  check_no_offset(num_formula$full_formula, "numerator")
  check_no_offset(denom_formula$full_formula, "denominator")

  # Handle shared structure
  shared_parsed <- parse_shared(shared, num_formula, denom_formula, data)

  structure(
    list(
      numerator = num_formula,
      denominator = denom_formula,
      shared = shared_parsed,
      original_formula = formula
    ),
    class = "ratiod_formula"
  )
}

#' Parse main formula to detect combined vs single syntax
#'
#' @param formula The formula to parse
#' @param data Data frame
#' @return List with type, response(s), and RHS
#' @keywords internal
parse_main_formula <- function(formula, data) {
  if (!inherits(formula, "formula")) {
    stop("`formula` must be a formula object", call. = FALSE)
  }

  if (length(formula) < 3) {
    stop("Formula must have response and predictors: response ~ predictors",
         call. = FALSE)
  }

  lhs <- formula[[2]]
  rhs <- formula[[3]]

  # Check for combined syntax: num | denom
  lhs_text <- deparse(lhs, width.cutoff = 500)
  lhs_text <- paste(lhs_text, collapse = " ")

  if (grepl("\\|", lhs_text)) {
    # Combined syntax
    parts <- strsplit(lhs_text, "\\|")[[1]]
    if (length(parts) != 2) {
      stop("Combined formula must have exactly two responses: num | denom ~ x",
           call. = FALSE)
    }

    num_var <- trimws(parts[1])
    denom_var <- trimws(parts[2])

    # Validate variables exist
    if (!num_var %in% names(data)) {
      stop(sprintf("Numerator variable '%s' not found in data", num_var),
           call. = FALSE)
    }
    if (!denom_var %in% names(data)) {
      stop(sprintf("Denominator variable '%s' not found in data", denom_var),
           call. = FALSE)
    }

    list(
      type = "combined",
      num_var = num_var,
      denom_var = denom_var,
      rhs = rhs
    )
  } else {
    # Single response
    response <- as.character(lhs)
    if (!response %in% names(data)) {
      stop(sprintf("Response variable '%s' not found in data", response),
           call. = FALSE)
    }

    list(
      type = "single",
      response = response,
      rhs = rhs
    )
  }
}

#' Build a process-specific formula
#'
#' @param response Response variable name
#' @param base_rhs RHS from main formula
#' @param additional Optional additional formula
#' @param data Data frame
#' @return Parsed formula specification
#' @keywords internal
build_process_formula <- function(response, base_rhs, additional, data) {
  # Start with base formula
  base_rhs_text <- deparse(base_rhs, width.cutoff = 500)
  base_rhs_text <- paste(base_rhs_text, collapse = " ")

  # Add additional terms if provided
  if (!is.null(additional)) {
    if (!inherits(additional, "formula")) {
      stop("Additional formula must be a formula object", call. = FALSE)
    }
    # One-sided formula: ~ extra_terms
    if (length(additional) == 2) {
      add_rhs <- deparse(additional[[2]], width.cutoff = 500)
      add_rhs <- paste(add_rhs, collapse = " ")
      base_rhs_text <- paste(base_rhs_text, "+", add_rhs)
    }
  }

  # Build full formula
  full_formula <- as.formula(
    paste(response, "~", base_rhs_text),
    env = parent.frame(2)
  )

  # Parse into components
  parse_formula_components(full_formula, data)
}

#' Parse denominator-only formula
#'
#' @param formula Formula for denominator (response ~ predictors or ~ predictors)
#' @param data Data frame
#' @return Parsed formula specification
#' @keywords internal
parse_denom_formula <- function(formula, data) {
  if (!inherits(formula, "formula")) {
    stop("`formula_denom` must be a formula", call. = FALSE)
  }

  if (length(formula) == 3) {
    # Two-sided: response ~ predictors
    parse_formula_components(formula, data)
  } else {
    stop("`formula_denom` must include response: denom ~ predictors",
         call. = FALSE)
  }
}

#' Parse formula into fixed and random effect components
#'
#' @param f Formula with response
#' @param data Data frame
#' @return List with response, fixed effects matrix, random effects
#' @keywords internal
parse_formula_components <- function(f, data) {
  response_var <- as.character(f[[2]])

  if (!response_var %in% names(data)) {
    stop(sprintf("Variable '%s' not found in data", response_var),
         call. = FALSE)
  }

  response <- data[[response_var]]

  rhs_text <- deparse(f[[3]], width.cutoff = 500)
  rhs_text <- paste(rhs_text, collapse = " ")

  # Extract random effects
  random_effects <- extract_random_effects(rhs_text, data)

  # Build fixed effects matrix
  fixed_formula <- remove_random_effects(f)
  mm <- build_model_matrix(fixed_formula, data)

  list(
    response_var = response_var,
    response = response,
    full_formula = f,
    fixed_formula = fixed_formula,
    X = mm$X,
    design = mm$design,
    random_effects = random_effects
  )
}

#' Check formula does not contain offset()
#'
#' @description
#' The tulpaRatio philosophy explicitly forbids offset terms. Offsets encode
#' the assumption that the ratio is the quantity of interest, which is
#' exactly the modelling approach this package rejects.
#'
#' @param f Formula to check
#' @param name Name for error messages
#' @keywords internal
check_no_offset <- function(f, name) {
  f_text <- deparse(f, width.cutoff = 500)
  f_text <- paste(f_text, collapse = " ")

  if (grepl("\\boffset\\s*\\(", f_text, ignore.case = TRUE)) {
    stop(
      sprintf("offset() is not allowed in the %s formula.\n\n", name),
      "tulpaRatio does not model ratios directly. Instead, it jointly models the\n",
      "numerator and denominator processes with shared latent structure.\n",
      "Offsets encode the assumption that the ratio is the quantity of interest,\n",
      "which is exactly the modelling approach this package rejects.\n\n",
      "If you have exposure/effort data, include it as the denominator response.",
      call. = FALSE
    )
  }
}

#' Extract RE terms with balanced parenthesis matching
#'
#' @description
#' Extracts random effect terms from formula text, properly handling nested
#' parentheses in expressions like `(1 + I(x^2) | group)` or `(1 + poly(x, 2) | group)`.
#'
#' @param rhs_text Right-hand side of formula as text
#' @return Character vector of RE terms (with outer parentheses)
#' @keywords internal
extract_re_terms_balanced <- function(rhs_text) {
  re_terms <- character(0)

  # Find all potential RE starts: opening paren followed eventually by |
  i <- 1
  n <- nchar(rhs_text)

  while (i <= n) {
    # Look for opening parenthesis
    if (substr(rhs_text, i, i) == "(") {
      # Track parenthesis depth
      depth <- 1
      j <- i + 1
      has_bar <- FALSE
      bar_at_depth_1 <- FALSE

      while (j <= n && depth > 0) {
        char <- substr(rhs_text, j, j)
        if (char == "(") {
          depth <- depth + 1
        } else if (char == ")") {
          depth <- depth - 1
        } else if (char == "|" && depth == 1) {
          # Found | at depth 1 - this is a valid RE term
          has_bar <- TRUE
          bar_at_depth_1 <- TRUE
        }
        j <- j + 1
      }

      # If we found a | at depth 1 and balanced parens, this is an RE term
      if (bar_at_depth_1 && depth == 0) {
        re_term <- substr(rhs_text, i, j - 1)
        re_terms <- c(re_terms, re_term)
        i <- j  # Continue after this term
        next
      }
    }
    i <- i + 1
  }

  re_terms
}

#' Split RE specification on | or || respecting parentheses
#'
#' @description
#' Splits an RE inner specification (without outer parens) on the `|` or `||` bar,
#' properly handling nested parentheses. Finds the rightmost `|` or `||` at depth 0.
#'
#' @param re_inner RE specification without outer parentheses (e.g., "1 + poly(x, 2) | group")
#' @return List with `parts` (character vector of LHS and RHS) and `is_uncorrelated` (logical)
#' @keywords internal
split_re_on_bar <- function(re_inner) {
  n <- nchar(re_inner)
  depth <- 0
  bar_pos <- NULL
  is_double_bar <- FALSE

  # Scan from left to find the | or || at depth 0
  i <- 1
  while (i <= n) {
    char <- substr(re_inner, i, i)
    if (char == "(") {
      depth <- depth + 1
    } else if (char == ")") {
      depth <- depth - 1
    } else if (char == "|" && depth == 0) {
      # Check if it's ||
      next_char <- if (i < n) substr(re_inner, i + 1, i + 1) else ""
      if (next_char == "|") {
        bar_pos <- i
        is_double_bar <- TRUE
        i <- i + 1  # Skip the second |
      } else {
        bar_pos <- i
        is_double_bar <- FALSE
      }
    }
    i <- i + 1
  }

  if (is.null(bar_pos)) {
    return(list(parts = c(re_inner), is_uncorrelated = FALSE))
  }

  # Split at bar position
  if (is_double_bar) {
    lhs <- substr(re_inner, 1, bar_pos - 1)
    rhs <- substr(re_inner, bar_pos + 2, n)
  } else {
    lhs <- substr(re_inner, 1, bar_pos - 1)
    rhs <- substr(re_inner, bar_pos + 1, n)
  }

  list(parts = c(lhs, rhs), is_uncorrelated = is_double_bar)
}


#' Split RE left-hand side into terms respecting parentheses
#'
#' @description
#' Splits a formula LHS on `+` signs that are not inside parentheses.
#' E.g., `1 + poly(x, 2) + I(x^2)` splits to `c("1", "poly(x, 2)", "I(x^2)")`
#'
#' @param lhs_text LHS of RE specification (e.g., "1 + x + poly(x, 2)")
#' @return Character vector of terms
#' @keywords internal
split_re_lhs_terms <- function(lhs_text) {
  terms <- character(0)
  current <- ""
  depth <- 0

  for (i in seq_len(nchar(lhs_text))) {
    char <- substr(lhs_text, i, i)
    if (char == "(") {
      depth <- depth + 1
      current <- paste0(current, char)
    } else if (char == ")") {
      depth <- depth - 1
      current <- paste0(current, char)
    } else if (char == "+" && depth == 0) {
      # Split here
      if (nchar(trimws(current)) > 0) {
        terms <- c(terms, trimws(current))
      }
      current <- ""
    } else {
      current <- paste0(current, char)
    }
  }

  # Don't forget the last term
  if (nchar(trimws(current)) > 0) {
    terms <- c(terms, trimws(current))
  }

  terms
}

#' Extract random effect specifications from formula text
#'
#' @description
#' Parses random effect terms from formula text, supporting:
#' - Random intercepts: `(1 | group)`
#' - Correlated random slopes: `(1 + x | group)` - estimates intercept-slope correlation
#' - Uncorrelated random slopes: `(1 + x || group)` - no correlation estimated
#' - Nested groups: `(1 | a/b)` expands to `(1|a) + (1|a:b)`
#' - Interactions: `(1 + x*z | group)` expands `x*z` to `x + z + x:z`
#' - Polynomials: `(1 + poly(x,2) | group)` creates columns for each poly term
#' - Inline transforms: `(1 + I(x^2) | group)` evaluates the transformation
#'
#' The `||` syntax is equivalent to `(1 | group) + (0 + x | group)` but more concise.
#'
#' @param rhs_text Right-hand side of formula as text
#' @param data Data frame
#' @return List of random effect specifications, each containing:
#'   - `group_var`: Name of grouping variable
#'   - `group`: Integer vector of group indices (1-based)
#'   - `n_groups`: Number of unique groups
#'   - `has_intercept`: Whether term includes random intercept
#'   - `slope_vars`: Character vector of slope variable names (NULL if none)
#'   - `slope_vars_raw`: Original slope terms before expansion (for reference)
#'   - `slope_vars_clean`: Cleaned slope names for parameter output
#'
#'   - `slope_matrix`: Matrix of slope values (n_obs x n_slopes) if slopes present
#'   - `correlated`: Whether slopes are correlated with intercept (TRUE for `|`, FALSE for `||`)
#'   - `original`: Original formula text
#' @keywords internal
extract_random_effects <- function(rhs_text, data) {
  # Extract RE terms using balanced parenthesis matching
  # This handles nested parens like (1 + I(x^2) | group)
  re_texts <- extract_re_terms_balanced(rhs_text)

  if (length(re_texts) == 0) {
    return(list())
  }
  random_effects <- list()
  re_idx <- 1

  for (i in seq_along(re_texts)) {
    re_text <- re_texts[i]
    # Remove outer parentheses
    re_inner <- gsub("^\\(|\\)$", "", re_text)

    # Detect correlated vs uncorrelated syntax and split on | or ||
    # Must handle nested parentheses - find the last | or || at depth 0
    split_result <- split_re_on_bar(re_inner)
    is_uncorrelated <- split_result$is_uncorrelated
    parts <- split_result$parts

    if (length(parts) < 2) {
      stop(sprintf("Invalid random effect specification: '%s'", re_text),
           call. = FALSE)
    }

    lhs <- trimws(parts[1])
    group_spec <- trimws(parts[2])

    # Parse LHS terms using balanced parenthesis splitting
    lhs_terms <- split_re_lhs_terms(lhs)

    # Determine if intercept is present
    # Following lme4 convention: intercept is included by default UNLESS "0" is explicitly specified
    # - (1 | g) -> intercept only
    # - (x | g) -> intercept AND slope (implicit 1)
    # - (1 + x | g) -> intercept AND slope (explicit 1)
    # - (0 + x | g) -> slope only (explicit 0 suppresses intercept)
    has_intercept <- !("0" %in% lhs_terms)

    # Extract slope terms (everything that isn't "1" or "0")
    slope_terms <- lhs_terms[!lhs_terms %in% c("1", "0")]

    slope_vars_raw <- NULL
    slope_vars <- NULL
    slope_vars_clean <- NULL
    slope_matrix <- NULL

    if (length(slope_terms) > 0) {
      # Store raw slope terms (before expansion) - join with + for formula syntax
      slope_vars_raw <- paste(slope_terms, collapse = " + ")

      # Use expand_re_slopes() to handle interactions, polynomials, etc.
      # Pass individual slope terms (not joined) so base variables can be extracted
      expanded <- expand_re_slopes(slope_terms, data, include_intercept = FALSE)

      slope_vars <- expanded$slope_vars
      slope_vars_clean <- clean_slope_names(slope_vars)
      slope_matrix <- expanded$slope_matrix

      # Note: validation is done inside expand_re_slopes via model.matrix()
      # which will throw an error if variables don't exist
    }

    # Grouping terms: (1 | a/b) is (1 | a) + (1 | a:b), and a:b groups by
    # the interaction of a and b
    group_vars_all <- trimws(strsplit(group_spec, "[/:]")[[1]])
    for (v in group_vars_all) {
      if (!v %in% names(data)) {
        stop(sprintf("Grouping variable '%s' not found in data", v),
             call. = FALSE)
      }
    }
    term_vars <- if (grepl("/", group_spec)) {
      lapply(seq_along(group_vars_all), function(j) group_vars_all[seq_len(j)])
    } else {
      list(group_vars_all)
    }

    for (tv in term_vars) {
      group_factor <- if (length(tv) == 1L) {
        as.factor(data[[tv]])
      } else {
        interaction(data[tv], drop = TRUE)
      }

      random_effects[[re_idx]] <- list(
        group_var = paste(tv, collapse = ":"),
        group = as.integer(group_factor),
        n_groups = nlevels(group_factor),
        has_intercept = has_intercept,
        slope_vars = slope_vars,
        slope_vars_raw = slope_vars_raw,
        slope_vars_clean = slope_vars_clean,
        slope_matrix = slope_matrix,
        correlated = !is_uncorrelated,  # || means uncorrelated
        original = re_text,
        nested_vars = tv
      )
      re_idx <- re_idx + 1
    }
  }

  random_effects
}

#' Extract base variable name from transformation
#'
#' @description
#' Extracts the base variable name from transformations like `poly(x, 2)` or `ns(x, 3)`.
#' Returns the variable name as-is if no transformation is detected.
#'
#' @param var_spec Variable specification (possibly with transformation)
#' @return Base variable name, or NULL if cannot be determined
#' @keywords internal
extract_base_variable <- function(var_spec) {
  # Check for function calls like poly(x, 2), ns(x, 3), scale(x), etc.
  if (grepl("^[a-zA-Z_][a-zA-Z0-9_]*\\s*\\(", var_spec)) {
    # Extract first argument
    match <- regmatches(var_spec, regexec("\\(\\s*([^,)]+)", var_spec))
    if (length(match[[1]]) >= 2) {
      return(trimws(match[[1]][2]))
    }
    return(NULL)
  }
  # Return as-is for simple variables
  return(var_spec)
}

#' Expand random effect slopes using formula semantics
#'
#' @description
#' Expands slope terms in random effects using R's formula machinery to handle:
#' - Interactions: `x*z` expands to `x + z + x:z`
#' - Polynomials: `poly(x, 2)` creates 2 columns (poly.1, poly.2)
#' - Inline transforms: `I(x^2)` evaluates the transformation
#' - Standard variables: `x` remains as-is
#'
#' Uses `model.matrix()` internally for proper formula expansion, ensuring
#' consistency with lme4/glmmTMB behavior.
#'
#' @param slope_terms Character vector of slope terms (may include interactions, etc.)
#' @param data Data frame containing the variables
#' @param include_intercept Logical; should intercept be included in expansion?
#' @return List with:
#'   - `slope_vars`: Character vector of expanded slope variable names
#'   - `slope_matrix`: Numeric matrix of slope values (n_obs x n_slopes)
#'   - `base_vars`: Character vector of base variable names (for validation)
#' @keywords internal
expand_re_slopes <- function(slope_terms, data, include_intercept = FALSE) {
  if (is.null(slope_terms) || length(slope_terms) == 0 ||
      (length(slope_terms) == 1 && nchar(trimws(slope_terms)) == 0)) {
    return(list(
      slope_vars = character(0),
      slope_matrix = matrix(nrow = nrow(data), ncol = 0),
      base_vars = character(0)
    ))
  }

  # Ensure slope_terms is a character vector
  if (!is.character(slope_terms)) {
    slope_terms <- as.character(slope_terms)
  }

  # Build a formula from the slope terms
  # Combine all terms with +
  combined_terms <- paste(slope_terms, collapse = " + ")

  # Create the formula
  if (include_intercept) {
    slope_formula <- as.formula(paste("~ 1 +", combined_terms))
  } else {
    # Use -1 or 0 to exclude intercept from model.matrix
    slope_formula <- as.formula(paste("~ 0 +", combined_terms))
  }

  # Use model.matrix to expand - this handles:
  # - x*z -> x + z + x:z
  # - poly(x, 2) -> poly.1, poly.2
  # - I(x^2) -> I(x^2)
  # - Factors -> dummy variables
  tryCatch({
    slope_matrix <- model.matrix(slope_formula, data = data)

    # Get column names (these are the expanded slope variable names)
    slope_vars <- colnames(slope_matrix)

    # Remove intercept column if present (shouldn't be with ~ 0, but be safe)
    if ("(Intercept)" %in% slope_vars) {
      intercept_idx <- which(slope_vars == "(Intercept)")
      slope_vars <- slope_vars[-intercept_idx]
      slope_matrix <- slope_matrix[, -intercept_idx, drop = FALSE]
    }

    # Extract base variables for reference (model.matrix already validates)
    base_vars <- unique(unlist(lapply(slope_terms, extract_base_variable)))
    base_vars <- base_vars[!is.na(base_vars) & !is.null(base_vars)]

    list(
      slope_vars = slope_vars,
      slope_matrix = slope_matrix,
      base_vars = base_vars
    )
  }, error = function(e) {
    stop(sprintf(
      "Failed to expand random slope terms '%s': %s",
      combined_terms, e$message
    ), call. = FALSE)
  })
}

#' Clean up slope variable names for output
#'
#' @description
#' Converts model.matrix column names to cleaner versions for parameter naming.
#' E.g., "I(x^2)" becomes "x_sq", "x:z" becomes "x_z", etc.
#'
#' @param slope_vars Character vector of slope variable names
#' @return Character vector of cleaned names
#' @keywords internal
clean_slope_names <- function(slope_vars) {
  cleaned <- slope_vars

  # Handle I() wrapper - extract content
  cleaned <- gsub("^I\\((.+)\\)$", "\\1", cleaned)

  # Handle power notation: x^2 -> x_pow2

  cleaned <- gsub("\\^(\\d+)", "_pow\\1", cleaned)

  # Handle interactions: x:z -> x_z
  cleaned <- gsub(":", "_", cleaned)

  # Handle poly() columns: poly(x, 2)1 -> x_poly1
  # model.matrix names them like "poly(x, 2)1", "poly(x, 2)2"
  cleaned <- gsub("poly\\(([^,]+),\\s*\\d+\\)(\\d+)", "\\1_poly\\2", cleaned)

  # Handle ns() columns similarly
  cleaned <- gsub("ns\\(([^,]+),\\s*\\d+\\)(\\d+)", "\\1_ns\\2", cleaned)

  # Handle scale() - just extract variable name
  cleaned <- gsub("scale\\(([^)]+)\\)", "\\1_scaled", cleaned)

  # Handle log() - extract variable name
  cleaned <- gsub("log\\(([^)]+)\\)", "log_\\1", cleaned)

  # Handle sqrt()
  cleaned <- gsub("sqrt\\(([^)]+)\\)", "sqrt_\\1", cleaned)

  # Replace any remaining special characters with underscore

  cleaned <- gsub("[^a-zA-Z0-9_]", "_", cleaned)

  # Remove consecutive underscores
  cleaned <- gsub("_+", "_", cleaned)

  # Remove leading/trailing underscores
  cleaned <- gsub("^_|_$", "", cleaned)

  cleaned
}

#' Remove random effects from formula to get fixed effects only
#'
#' @param f Formula with potential random effects
#' @return Formula with only fixed effects
#' @keywords internal
remove_random_effects <- function(f) {
  fixed <- Filter(Negate(is_random_effect_term), additive_terms(f[[3]]))
  f[[3]] <- if (length(fixed)) {
    Reduce(function(a, b) call("+", a, b), fixed)
  } else {
    1
  }
  f
}

#' Additive terms of a formula's right-hand side
#'
#' Splits an expression on binary `+` into the terms it adds, leaving every
#' other call (`-`, `:`, `*`, `I()`, `poly()`, ...) whole.
#'
#' @param e A language object.
#' @return A list of language objects.
#' @keywords internal
additive_terms <- function(e) {
  if (is.call(e) && identical(e[[1]], as.name("+")) && length(e) == 3L) {
    c(additive_terms(e[[2]]), additive_terms(e[[3]]))
  } else {
    list(e)
  }
}

#' Is a formula term a random-effect term
#'
#' A parenthesised `|` or `||` call, as in `(1 | g)` or `(1 + x || g)`.
#'
#' @param e A language object.
#' @return Logical.
#' @keywords internal
is_random_effect_term <- function(e) {
  is.call(e) && identical(e[[1]], as.name("(")) && is.call(e[[2]]) &&
    (identical(e[[2]][[1]], as.name("|")) || identical(e[[2]][[1]], as.name("||")))
}

#' Build model matrix from fixed effects formula
#'
#' @param f Formula (response ~ fixed effects)
#' @param data Data frame
#' @return List with the model matrix `X` and its `design`: the terms without
#'   response, the factor levels and the contrasts it was built with, which
#'   `design_matrix_newdata()` reuses so new data get the same columns.
#' @keywords internal
build_model_matrix <- function(f, data) {
  mf <- model.frame(f, data = data, na.action = na.pass)
  tt <- terms(mf)
  X <- model.matrix(tt, data = mf)
  list(X = X,
       design = list(terms = delete.response(tt),
                     xlevels = .getXlevels(tt, mf),
                     contrasts = attr(X, "contrasts")))
}

#' Fixed-effects design matrix for new data
#'
#' Rebuilds the columns the fit was estimated with: factors take the training
#' levels and contrasts, so a level absent from `newdata` still has its column
#' and an unseen level is an error.
#'
#' @param design The `design` list `build_model_matrix()` recorded.
#' @param newdata Data frame holding the predictor variables.
#' @return Model matrix with the training column layout.
#' @keywords internal
design_matrix_newdata <- function(design, newdata) {
  mf <- model.frame(design$terms, newdata, xlev = design$xlevels,
                    na.action = na.pass)
  model.matrix(design$terms, mf, contrasts.arg = design$contrasts)
}

#' Parse shared formula specification
#'
#' @param shared Shared formula or NULL
#' @param num_parsed Parsed numerator formula
#' @param denom_parsed Parsed denominator formula
#' @param data Data frame
#' @return Shared structure specification
#' @keywords internal
parse_shared <- function(shared, num_parsed, denom_parsed, data) {
  if (is.null(shared)) {
    # Default: infer from matching random effects
    return(infer_shared_structure(num_parsed, denom_parsed))
  }

  if (inherits(shared, "formula")) {
    shared_text <- deparse(shared, width.cutoff = 500)
    shared_text <- paste(shared_text, collapse = " ")

    # ~ 0 or ~ -1 means no shared structure
    if (grepl("^~\\s*(0|-1)\\s*$", shared_text)) {
      warn_independence()
      return(list(
        type = "independent",
        random_effects = list(),
        warned = TRUE
      ))
    }

    # Parse shared random effects (one-sided formula)
    if (length(shared) == 2) {
      rhs_text <- deparse(shared[[2]], width.cutoff = 500)
      rhs_text <- paste(rhs_text, collapse = " ")
    } else {
      stop("shared formula should be one-sided: ~ (1 | group)", call. = FALSE)
    }

    shared_re <- extract_random_effects(rhs_text, data)

    return(list(
      type = "explicit",
      random_effects = shared_re,
      warned = FALSE
    ))
  }

  stop("`shared` must be a formula or NULL", call. = FALSE)
}

#' Infer shared structure from numerator and denominator formulas
#'
#' @param num_parsed Parsed numerator
#' @param denom_parsed Parsed denominator
#' @return Shared structure specification
#' @keywords internal
infer_shared_structure <- function(num_parsed, denom_parsed) {
  num_groups <- vapply(num_parsed$random_effects,
                       function(x) x$group_var, character(1))
  denom_groups <- vapply(denom_parsed$random_effects,
                         function(x) x$group_var, character(1))

  if (length(num_groups) == 0) num_groups <- character(0)
  if (length(denom_groups) == 0) denom_groups <- character(0)

  shared_groups <- intersect(num_groups, denom_groups)

  if (length(shared_groups) == 0) {
    return(list(
      type = "none",
      random_effects = list(),
      warned = FALSE
    ))
  }

  # Use RE specs from numerator for shared groups
  shared_re <- num_parsed$random_effects[num_groups %in% shared_groups]

  list(
    type = "inferred",
    random_effects = shared_re,
    warned = FALSE
  )
}

#' Warn about independence assumption
#' @keywords internal
warn_independence <- function() {
  warning(
    "You have specified `shared = ~ 0`, assuming independence between ",
    "numerator and denominator processes.\n\n",
    "This assumption is often violated in practice and can lead to spurious ",
    "ratio effects when both processes share unmeasured drivers.\n\n",
    "Consider whether shared random effects would be more appropriate.",
    call. = FALSE,
    immediate. = TRUE
  )
}

#' Print method for ratiod_formula
#'
#' @param x A ratiod_formula object
#' @param ... Ignored
#' @export
#' @keywords internal
print.ratiod_formula <- function(x, ...) {
  cat("tulpaRatio formula specification\n")
  cat("===========================\n\n")

  cat("Numerator:", x$numerator$response_var, "\n")
  cat("  Fixed effects:", ncol(x$numerator$X), "coefficients\n")
  cat("  Random effects:", length(x$numerator$random_effects), "terms\n")

  cat("\nDenominator:", x$denominator$response_var, "\n")
  cat("  Fixed effects:", ncol(x$denominator$X), "coefficients\n")
  cat("  Random effects:", length(x$denominator$random_effects), "terms\n")

  cat("\nShared structure:", x$shared$type, "\n")
  if (length(x$shared$random_effects) > 0) {
    groups <- vapply(x$shared$random_effects,
                     function(r) r$group_var, character(1))
    cat("  Groups:", paste(groups, collapse = ", "), "\n")
  }

  invisible(x)
}
