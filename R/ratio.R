#' @importFrom stats median
#' @importFrom grDevices colorRampPalette
NULL

#' Ratio Posteriors from a Ratio Model
#'
#' @description
#' Compute posterior distributions of ratios from a fitted ratio model.
#' Ratios are derived quantities: E\[numerator\] / E\[denominator\], computed
#' from each posterior draw, so they carry the full posterior uncertainty.
#'
#' @param object A `ratiod_fit` object
#' @param newdata Optional data frame for prediction. If NULL, uses
#'   original data.
#' @param type Scale for ratio: "response" (default), "log", or "logit".
#' @param by Optional grouping variable name to aggregate ratios by group.
#' @param summary Logical; if TRUE, return summary statistics instead
#'   of full posterior draws.
#' @param probs Quantiles to compute if `summary = TRUE`.
#' @param ... Ignored
#'
#' @return A `ratiod_ratio` object containing posterior draws or summaries.
#'
#' @examples
#' # Create a simple family
#' fam <- ratiod_poisson_gamma()
#'
#' \donttest{
#' # Generate synthetic data: a site effect shared by count and effort
#' set.seed(123)
#' n <- 60
#' site <- rep(letters[1:6], each = 10)
#' u <- rnorm(6, sd = 0.5)[match(site, letters[1:6])]
#' depth <- rnorm(n)
#' df <- data.frame(
#'   count = rpois(n, exp(3 + 0.3 * depth + u)),
#'   effort = rgamma(n, shape = 10, rate = exp(-u)),
#'   depth = depth,
#'   site = site
#' )
#'
#' # Fit model
#' fit <- tratio(
#'   count | effort ~ depth + (1 | site),
#'   data = df,
#'   family = ratiod_poisson_gamma(),
#'   mode = "hmc",
#'   control = list(iter = 400, chains = 1, seed = 1)
#' )
#'
#' # Extract all observation-level ratios
#' r <- ratio(fit)
#' summary(r)
#'
#' # Log-ratio scale
#' r_log <- ratio(fit, type = "log")
#'
#' # Group-level ratios
#' r_site <- ratio(fit, by = "site")
#' }
#'
#' @export
ratio <- function(object, newdata = NULL, type = c("response", "log", "logit"),
                  by = NULL, summary = FALSE, probs = c(0.025, 0.5, 0.975), ...) {
  UseMethod("ratio")
}

#' @export
ratio.ratiod_fit <- function(object, newdata = NULL, type = c("response", "log", "logit"),
                            by = NULL, summary = FALSE, probs = c(0.025, 0.5, 0.975), ...) {

  type <- match.arg(type)

  # Handle newdata via predict()
  if (!is.null(newdata)) {
    pred <- predict(object, newdata = newdata, component = "ratio", summary = FALSE)
    ratio_draws <- pred$ratio
  } else {
    # Get ratio draws from fitted values or stored ratio_draws
    if (!is.null(object$ratio_draws)) {
      # HMC backend stores pre-computed ratio draws
      ratio_draws <- object$ratio_draws
    } else {
      # Compute from fitted values
      fitted_vals <- compute_fitted_values(object)
      ratio_draws <- fitted_vals$ratio
    }
  }

  # Ensure column names
  if (is.null(colnames(ratio_draws))) {
    colnames(ratio_draws) <- paste0("ratio[", seq_len(ncol(ratio_draws)), "]")
  }

  # Aggregate by group if requested. The group value is a geometric mean of
  # ratios, which is defined on the natural scale, so aggregation happens
  # before the requested transform is applied to the result.
  obs_data <- newdata %||% object$data
  if (!is.null(by)) {
    ratio_draws <- aggregate_by_group(ratio_draws, obs_data, by)
  }

  # Transform to requested scale
  if (type == "log") {
    ratio_draws <- log(ratio_draws)
  } else if (type == "logit") {
    # Logit only makes sense if ratio is a proportion (0-1)
    if (any(ratio_draws > 1, na.rm = TRUE) || any(ratio_draws < 0, na.rm = TRUE)) {
      warning("Logit transform requested but ratios are not in (0, 1). ",
              "Results may include Inf/-Inf.", call. = FALSE)
    }
    ratio_draws <- log(ratio_draws / (1 - ratio_draws))
  }

  # Create result object
  result <- structure(
    list(
      draws = ratio_draws,
      type = type,
      by = by,
      n_obs = ncol(ratio_draws),
      n_draws = nrow(ratio_draws),
      data = obs_data
    ),
    class = "ratiod_ratio"
  )

  if (summary) {
    return(summary(result, probs = probs))
  }

  result
}

#' Aggregate ratio draws by grouping variable
#'
#' @param draws Matrix of posterior draws
#' @param data Original data frame
#' @param by Grouping variable name
#'
#' @return Matrix with columns for each group level
#' @keywords internal
aggregate_by_group <- function(draws, data, by) {
  if (!(by %in% names(data))) {
    stop(sprintf("Grouping variable '%s' not found in data", by), call. = FALSE)
  }

  groups <- unique(data[[by]])
  n_groups <- length(groups)
  n_draws <- nrow(draws)

  # Average ratios within each group
  result <- matrix(NA_real_, nrow = n_draws, ncol = n_groups)
  colnames(result) <- paste0("ratio[", groups, "]")

  for (g in seq_along(groups)) {
    idx <- which(data[[by]] == groups[g])
    if (length(idx) == 1) {
      result[, g] <- draws[, idx]
    } else {
      # Geometric mean for ratios (arithmetic mean on log scale)
      result[, g] <- exp(rowMeans(log(draws[, idx, drop = FALSE])))
    }
  }

  result
}

#' @rdname ratio
#'
#' @export
summary.ratiod_ratio <- function(object, probs = c(0.025, 0.5, 0.975), ...) {
  draws <- object$draws

  # Compute summaries
  summaries <- data.frame(
    mean = colMeans(draws),
    sd = apply(draws, 2, sd),
    t(apply(draws, 2, quantile, probs = probs))
  )

  # Clean up column names for quantiles
  names(summaries)[3:ncol(summaries)] <- paste0("q", probs * 100)

  # Add observation/group index
  if (is.null(object$by)) {
    summaries$obs <- seq_len(object$n_obs)
    summaries <- summaries[, c("obs", "mean", "sd", names(summaries)[3:(ncol(summaries)-1)])]
  } else {
    summaries$group <- gsub("^ratio\\[|\\]$", "", rownames(summaries))
    summaries <- summaries[, c("group", "mean", "sd", names(summaries)[3:(ncol(summaries)-1)])]
  }

  rownames(summaries) <- NULL

  structure(
    summaries,
    type = object$type,
    n_draws = object$n_draws,
    class = c("ratiod_ratio_summary", "data.frame")
  )
}

#' @rdname ratio
#' @param x A `ratiod_ratio` object, or the `ratiod_ratio_summary` its
#'   `summary()` returns.
#'
#' @export
print.ratiod_ratio <- function(x, ...) {
  cat("tulpaRatio ratio posterior\n")
  cat("=====================\n\n")
  cat("Scale:", x$type, "\n")
  cat("Observations:", x$n_obs, "\n")
  cat("Posterior draws:", x$n_draws, "\n")
  if (!is.null(x$by)) {
    cat("Grouped by:", x$by, "\n")
  }
  cat("\nUse summary() for posterior summaries\n")
  cat("Use plot() for visualization\n")
  invisible(x)
}

#' @rdname ratio
#' @param n Number of rows to print
#'
#' @export
print.ratiod_ratio_summary <- function(x, n = 10, ...) {
  cat("tulpaRatio ratio summary (", attr(x, "type"), " scale)\n", sep = "")
  cat("===========================================\n\n")

  if (nrow(x) > n) {
    print.data.frame(head(x, n), row.names = FALSE, ...)
    cat(sprintf("\n... and %d more rows\n", nrow(x) - n))
  } else {
    print.data.frame(x, row.names = FALSE, ...)
  }

  invisible(x)
}

#' Contrasts Between Ratio Conditions
#'
#' @description
#' Compute posterior distributions of contrasts (differences or ratios)
#' between conditions for the derived ratio quantity.
#'
#' @param object A `ratiod_fit` object
#' @param contrast A formula specifying the contrast, e.g., `~ treatment`
#' @param type Type of contrast: "difference" (default) or "ratio"
#' @param ref Reference level for factor contrasts (default: first level)
#' @param ... Ignored
#'
#' @return A data frame with posterior summaries of contrasts
#'
#' @examples
#' \donttest{
#' # Generate synthetic data with grouping variable
#' set.seed(456)
#' n <- 60
#' df <- data.frame(
#'   count = rpois(n, lambda = 25),
#'   effort = rgamma(n, shape = 12, rate = 1),
#'   season = sample(c("summer", "winter"), n, replace = TRUE)
#' )
#'
#' # Fit model
#' fit <- tratio(
#'   count | effort ~ season,
#'   data = df,
#'   family = ratiod_poisson_gamma(),
#'   mode = "hmc",
#'   control = list(iter = 400, chains = 1, seed = 1)
#' )
#'
#' # Compare CPUE between seasons
#' ratio_contrast(fit, ~ season)
#'
#' # Ratio of ratios (multiplicative comparison)
#' ratio_contrast(fit, ~ season, type = "ratio")
#' }
#'
#' @export
ratio_contrast <- function(object, contrast, type = c("difference", "ratio"),
                           ref = NULL, ...) {

  type <- match.arg(type)

  # Extract contrast variable from formula
  if (!inherits(contrast, "formula")) {
    stop("`contrast` must be a formula, e.g., ~ treatment", call. = FALSE)
  }

  contrast_var <- all.vars(contrast)
  if (length(contrast_var) != 1) {
    stop("contrast formula must contain exactly one variable", call. = FALSE)
  }

  if (!(contrast_var %in% names(object$data))) {
    stop(sprintf("Variable '%s' not found in data", contrast_var), call. = FALSE)
  }

  # Get ratio draws on log scale
  ratio_obj <- ratio(object, type = "log")
  draws <- ratio_obj$draws

  # Handle case where draws might be a list (from predict)
  if (is.list(draws) && !is.matrix(draws)) {
    draws <- draws$ratio
  }

  # Levels of the grouping variable, in factor order
  group <- object$data[[contrast_var]]
  if (!(is.factor(group) || is.character(group) || is.logical(group))) {
    stop(sprintf(paste0("`%s` is %s; ratio_contrast() compares the levels of a ",
                        "grouping variable. Convert it with factor() first."),
                 contrast_var, class(group)[1]), call. = FALSE)
  }
  levels <- if (is.factor(group)) levels(droplevels(group)) else sort(unique(group))
  if (is.null(ref)) {
    ref <- levels[1]
  }
  if (!(ref %in% levels)) {
    stop(sprintf("Reference level '%s' not found", ref), call. = FALSE)
  }

  # Compute contrasts
  ref_idx <- which(object$data[[contrast_var]] == ref)
  ref_mean <- rowMeans(draws[, ref_idx, drop = FALSE])

  results <- list()
  for (lvl in levels) {
    if (lvl == ref) next

    lvl_idx <- which(object$data[[contrast_var]] == lvl)
    lvl_mean <- rowMeans(draws[, lvl_idx, drop = FALSE])

    if (type == "difference") {
      # Difference on log scale = log ratio
      contrast_draws <- lvl_mean - ref_mean
    } else {
      # Ratio = exp(difference on log scale)
      contrast_draws <- exp(lvl_mean - ref_mean)
    }

    results[[as.character(lvl)]] <- data.frame(
      contrast = paste(lvl, "vs", ref),
      mean = mean(contrast_draws),
      sd = sd(contrast_draws),
      q2.5 = quantile(contrast_draws, 0.025),
      q50 = quantile(contrast_draws, 0.5),
      q97.5 = quantile(contrast_draws, 0.975),
      prob_positive = mean(contrast_draws > 0)
    )
  }

  result <- do.call(rbind, results)
  rownames(result) <- NULL

  structure(
    result,
    type = type,
    contrast_var = contrast_var,
    ref = ref,
    class = c("ratiod_contrast", "data.frame")
  )
}

#' @rdname ratio_contrast
#' @param x A ratiod_contrast object
#'
#' @export
print.ratiod_contrast <- function(x, ...) {
  cat("tulpaRatio ratio contrasts\n")
  cat("=====================\n\n")
  cat("Type:", attr(x, "type"), "\n")
  cat("Variable:", attr(x, "contrast_var"), "\n")
  cat("Reference:", attr(x, "ref"), "\n\n")
  print.data.frame(x, row.names = FALSE, ...)
  invisible(x)
}


#' Fitted Values of a Ratio Model
#'
#' @description
#' Compute posterior distributions of fitted values (expected values) for both
#' the numerator and denominator processes from a fitted ratio model.
#'
#' @param object A `ratiod_fit` object
#' @param component Which component to return: "numerator", "denominator",
#'   "ratio", or "all" (default).
#' @param summary Logical; if TRUE, return summary statistics instead of
#'   full posterior draws (default TRUE).
#' @param probs Quantiles to compute if `summary = TRUE`.
#' @param ... Ignored
#'
#' @return If `summary = TRUE`, a data frame with posterior summaries.
#'   If `summary = FALSE`, a list with matrices of posterior draws.
#'
#' @examples
#' \donttest{
#' # Generate synthetic data
#' set.seed(789)
#' n <- 40
#' df <- data.frame(
#'   count = rpois(n, lambda = 15),
#'   effort = rgamma(n, shape = 8, rate = 1),
#'   depth = rnorm(n)
#' )
#'
#' # Fit model
#' fit <- tratio(
#'   count | effort ~ depth,
#'   data = df,
#'   family = ratiod_poisson_gamma(),
#'   mode = "hmc",
#'   control = list(iter = 400, chains = 1, seed = 1)
#' )
#'
#' # Get fitted value summaries
#' fitted(fit)
#'
#' # Get just numerator fitted values
#' fitted(fit, component = "numerator")
#'
#' # Get full posterior draws
#' fitted(fit, summary = FALSE)
#' }
#'
#' @method fitted ratiod_fit
#' @export
fitted.ratiod_fit <- function(object, component = c("all", "numerator", "denominator", "ratio"),
                              summary = TRUE, probs = c(0.025, 0.5, 0.975), ...) {

  component <- match.arg(component)

  # Compute fitted values based on backend
  fitted_vals <- compute_fitted_values(object)

  # Select requested component
  if (component == "all") {
    draws_list <- fitted_vals
  } else if (component == "ratio") {
    draws_list <- list(ratio = fitted_vals$ratio)
  } else {
    draws_list <- list()
    draws_list[[component]] <- fitted_vals[[component]]
  }

  if (summary) {
    # Compute summaries for each component
    result_list <- list()
    for (comp_name in names(draws_list)) {
      draws <- draws_list[[comp_name]]
      summaries <- data.frame(
        obs = seq_len(ncol(draws)),
        component = comp_name,
        mean = colMeans(draws),
        sd = apply(draws, 2, sd),
        t(apply(draws, 2, quantile, probs = probs))
      )
      names(summaries)[5:ncol(summaries)] <- paste0("q", probs * 100)
      rownames(summaries) <- NULL
      result_list[[comp_name]] <- summaries
    }

    result <- do.call(rbind, result_list)
    rownames(result) <- NULL

    structure(
      result,
      n_draws = nrow(draws_list[[1]]),
      class = c("ratiod_fitted", "data.frame")
    )
  } else {
    structure(
      draws_list,
      n_draws = nrow(draws_list[[1]]),
      n_obs = ncol(draws_list[[1]]),
      class = "ratiod_fitted_draws"
    )
  }
}


#' Compute fitted values from ratiod_fit
#' @keywords internal
compute_fitted_values <- function(object) {
  backend <- object$backend

  if (has_hmc_layout(object)) {
    return(compute_fitted_hmc(object))
  } else if (backend %in% c("pg", "gibbs")) {
    return(compute_fitted_pg(object))
  } else if (backend == "laplace") {
    return(compute_fitted_laplace(object))
  } else {
    stop("Unknown backend: ", backend, call. = FALSE)
  }
}


#' Model type of a fitted object
#'
#' The HMC converter records it; every other backend carries the family it was
#' fitted with, which is what the response transform is taken from.
#'
#' @keywords internal
fit_model_type <- function(object) {
  object$.internal$model_type %||% get_hmc_model_type(object$family)
}


#' Links of a fitted object
#' @keywords internal
fit_links <- function(object) {
  family_links(object$family, fit_model_type(object))
}


#' Compute fitted values for HMC backend
#' @keywords internal
compute_fitted_hmc <- function(object) {
  unpacked <- hmc_fit_unpack(object)
  eta <- hmc_eta_draws(unpacked, hmc_fit_design(object))
  warn_dropped_structures(eta$dropped)
  hmc_response_draws(eta, fit_links(object),
                     exposure = hmc_exposure(object$.internal$hmc_data,
                                             fit_model_type(object)))
}


#' Compute fitted values for the PG and Gibbs backends
#'
#' Their sweeps record the linear predictor they sampled, so the fitted
#' values are that predictor on the response scale of the family.
#'
#' @keywords internal
compute_fitted_pg <- function(object) {
  internal <- object$.internal
  eta_num <- internal$eta_num %||% internal$eta
  if (is.null(eta_num)) {
    stop("This fit stores no linear predictor, so fitted values ",
         "cannot be computed from it.", call. = FALSE)
  }
  eta_denom <- internal$eta_denom %||% matrix(0, nrow(eta_num), ncol(eta_num))
  hmc_response_draws(list(eta_num = eta_num, eta_denom = eta_denom),
                     fit_links(object))
}


#' Linear predictor draws of a Laplace fit
#'
#' The stored sample matrix holds the fixed effects of each process followed by
#' the effect of every latent block on the linear predictor. A block's effect is
#' read at the unit each observation belongs to, and enters both processes.
#'
#' @keywords internal
laplace_eta_draws <- function(object) {
  internal <- object$.internal
  samples <- internal$samples
  layout <- internal$layout
  if (is.null(samples) || is.null(layout)) {
    stop("This Laplace fit stores no posterior samples.", call. = FALSE)
  }

  N <- nrow(internal$X_num)
  eta_num <- samples[, layout$beta$num, drop = FALSE] %*% t(internal$X_num)
  eta_denom <- if (!is.null(layout$beta$denom)) {
    samples[, layout$beta$denom, drop = FALSE] %*% t(internal$X_denom)
  } else {
    matrix(0, nrow(eta_num), ncol(eta_num))
  }

  for (part in layout$parts) {
    effect <- spread_effect(samples[, part$cols, drop = FALSE], part$index, N)
    eta_num <- eta_num + effect
    eta_denom <- eta_denom + effect
  }

  list(eta_num = eta_num, eta_denom = eta_denom)
}


#' Compute fitted values for Laplace backend
#' @keywords internal
compute_fitted_laplace <- function(object) {
  hmc_response_draws(laplace_eta_draws(object), fit_links(object))
}


#' Predictions from a Ratio Model
#'
#' @description
#' Compute posterior predictions for new data from a fitted ratio model.
#'
#' @param object A `ratiod_fit` object
#' @param newdata Data frame with new observations. Must contain all
#'   predictor variables used in the original model.
#' @param type Type of prediction: "response" for fitted values on
#'   the response scale, "link" for linear predictor scale.
#' @param component Which component to predict: "ratio" (default),
#'   "numerator", "denominator", or "all".
#' @param summary Logical; if TRUE, return summary statistics instead of
#'   full posterior draws (default TRUE).
#' @param probs Quantiles to compute if `summary = TRUE`.
#' @param re_formula Formula for random effects in prediction.
#'   Use `NULL` to include all random effects (if new levels match),
#'   `NA` or `~ 0` to exclude random effects (population-level prediction).
#' @param allow_new_levels Logical; if TRUE, allows new group levels not
#'   in the original data (predictions use population mean).
#' @param coords.0 Matrix of spatial coordinates for new prediction locations
#'   (required for GP/HSGP spatial models). Should be N x 2 matrix with same
#'   coordinate system as training data.
#' @param include_spatial Logical; if TRUE (default), include spatial random
#'   effects in predictions. Set to FALSE for fixed-effects-only predictions.
#' @param return_spatial Logical; if TRUE, return spatial effect samples at
#'   new locations as `w.0.samples` in the output (similar to spOccupancy).
#' @param n_samples Number of posterior samples to draw for backends that

#'   don't store samples directly (e.g., Laplace, VI). Default uses all
#'   available samples for HMC/PG.
#' @param ... Ignored
#'
#' @return If `summary = TRUE`, a data frame with posterior summaries.
#'   If `summary = FALSE`, a matrix of posterior draws (rows = draws, cols = obs).
#'   If `return_spatial = TRUE`, returns a list with `predictions` and `w.0.samples`.
#'
#' @examples
#' \donttest{
#' # A smooth spatial field w shared by count and effort
#' set.seed(321)
#' n <- 60
#' x <- runif(n)
#' y <- runif(n)
#' w <- 0.6 * sin(2 * pi * x) + 0.6 * cos(2 * pi * y)
#' depth <- rnorm(n)
#' season <- factor(sample(c("summer", "winter"), n, replace = TRUE))
#' df <- data.frame(
#'   count = rpois(n, exp(2.5 + 0.3 * depth + 0.4 * (season == "winter") + w)),
#'   effort = rgamma(n, shape = 9, rate = exp(-w)),
#'   depth = depth,
#'   season = season,
#'   x = x,
#'   y = y
#' )
#' fit <- tratio(
#'   count | effort ~ depth + season,
#'   data = df,
#'   family = ratiod_poisson_gamma(),
#'   control = list(iter = 400, chains = 1, seed = 1)
#' )
#'
#' # Predict for new data; levels of `season` follow the training data
#' new_df <- data.frame(depth = c(-1, 0, 1), season = "summer")
#' predict(fit, newdata = new_df)
#'
#' # Full posterior draws of the predicted ratio (draws x rows)
#' dim(predict(fit, newdata = new_df, summary = FALSE)$ratio)
#'
#' # Spatial prediction at new locations (kriging) from a GP fit
#' fit_gp <- tratio(
#'   count | effort ~ depth,
#'   data = df,
#'   spatial = spatial_gp(coords = c("x", "y")),
#'   family = ratiod_poisson_gamma(),
#'   control = list(iter = 400, chains = 1, seed = 1)
#' )
#' new_coords <- matrix(runif(10), ncol = 2)
#' predict(fit_gp, newdata = data.frame(depth = rnorm(5)),
#'         coords.0 = new_coords)
#' }
#'
#' @method predict ratiod_fit
#' @export
predict.ratiod_fit <- function(object, newdata, type = c("response", "link"),
                               component = c("ratio", "numerator", "denominator", "all"),
                               summary = TRUE, probs = c(0.025, 0.5, 0.975),
                               re_formula = NULL, allow_new_levels = FALSE,
                               coords.0 = NULL, include_spatial = TRUE,
                               return_spatial = FALSE, n_samples = NULL, ...) {

  type <- match.arg(type)
  component <- match.arg(component)

  if (missing(newdata) || is.null(newdata)) {
    # Use fitted values for original data
    return(fitted(object, component = component, summary = summary, probs = probs))
  }

  # Check backend and dispatch

  backend <- object$backend

  # Dispatch to appropriate backend

  if (has_hmc_layout(object)) {
    pred_result <- predict_hmc(object, newdata, type, re_formula, allow_new_levels,
                               coords.0, include_spatial, return_spatial)
  } else if (backend == "pg") {
    pred_result <- predict_pg(object, newdata, type, re_formula, allow_new_levels,
                              coords.0, include_spatial, return_spatial)
  } else if (backend == "gibbs") {
    pred_result <- predict_gibbs(object, newdata, type, re_formula, allow_new_levels,
                                 include_spatial, return_spatial)
  } else if (backend == "laplace") {
    pred_result <- predict_laplace(object, newdata, type, re_formula, allow_new_levels,
                                   coords.0, include_spatial, return_spatial,
                                   n_samples = n_samples %||% 1000L)
  } else if (backend == "vi") {
    pred_result <- predict_vi(object, newdata, type, re_formula, allow_new_levels,
                              coords.0, include_spatial, return_spatial,
                              n_samples = n_samples %||% 1000L)
  } else {
    stop("predict() not yet supported for backend '", backend, "'.\n",
         "Use fitted() for in-sample fitted values.", call. = FALSE)
  }

  pred_draws <- pred_result$draws
  w_samples <- pred_result$w_samples  # NULL if no spatial or return_spatial = FALSE

  # Select component
  if (component == "all") {
    draws_list <- pred_draws
  } else {
    draws_list <- list()
    draws_list[[component]] <- pred_draws[[component]]
  }

  if (summary) {
    result_list <- list()
    for (comp_name in names(draws_list)) {
      draws <- draws_list[[comp_name]]
      summaries <- data.frame(
        obs = seq_len(ncol(draws)),
        component = comp_name,
        mean = colMeans(draws),
        sd = apply(draws, 2, sd),
        t(apply(draws, 2, quantile, probs = probs))
      )
      names(summaries)[5:ncol(summaries)] <- paste0("q", probs * 100)
      rownames(summaries) <- NULL
      result_list[[comp_name]] <- summaries
    }

    result <- do.call(rbind, result_list)
    rownames(result) <- NULL

    predictions <- structure(
      result,
      n_draws = nrow(draws_list[[1]]),
      newdata = newdata,
      class = c("ratiod_prediction", "data.frame")
    )
  } else {
    predictions <- structure(
      draws_list,
      n_draws = nrow(draws_list[[1]]),
      n_obs = ncol(draws_list[[1]]),
      newdata = newdata,
      class = "ratiod_prediction_draws"
    )
  }

  # Return with spatial samples if requested
  if (return_spatial && !is.null(w_samples)) {
    return(list(
      predictions = predictions,
      w.0.samples = w_samples
    ))
  }

  predictions
}


#' Build prediction data structures
#' @keywords internal
build_prediction_data <- function(object, newdata, re_formula, allow_new_levels) {
  formula <- object$formula

  design_info <- prediction_design_info(object)
  X_num <- design_matrix_newdata(formula$numerator$design, newdata)

  # For binomial/beta_binomial, denominator is fixed trials - no design matrix
  if (design_info$p_denom == 0) {
    X_denom <- matrix(numeric(0), nrow = nrow(newdata), ncol = 0)
  } else {
    X_denom <- design_matrix_newdata(formula$denominator$design, newdata)
  }

  # Random effects, remapped onto the new data
  include_re <- !isTRUE(is.na(re_formula)) && !identical(re_formula, ~ 0)
  re_design <- prediction_re_design(object, newdata, include_re, allow_new_levels)

  # A known exposure, when the new data carry the denominator column: the rate
  # needs none, the numerator mean does
  exposure <- NULL
  if (fit_links(object)$exposure) {
    var <- formula$denominator$response_var
    if (!is.null(var) && var %in% names(newdata)) exposure <- as.numeric(newdata[[var]])
  }

  list(
    X_num = X_num,
    X_denom = X_denom,
    re_group = re_design$re_group,
    re_group_matrix = re_design$re_group_matrix,
    slope_matrices = re_design$slope_matrices,
    N = nrow(newdata),
    include_re = include_re,
    exposure = exposure
  )
}


#' Design matrices a fit was made with
#'
#' The HMC converter keeps them on `hmc_data`, the Laplace converter beside its
#' sample matrix. A fit without a denominator process carries a matrix of no
#' columns for it.
#'
#' @keywords internal
prediction_design_info <- function(object) {
  internal <- object$.internal
  if (!is.null(internal$hmc_data)) {
    return(list(X_num = internal$hmc_data$X_num,
                X_denom = internal$hmc_data$X_denom,
                p_denom = internal$hmc_data$p_denom))
  }
  X_num <- internal$X_num
  X_denom <- internal$X_denom %||% matrix(numeric(0), nrow(X_num), 0L)
  list(X_num = X_num, X_denom = X_denom, p_denom = ncol(X_denom))
}


#' Random-effect design of a prediction
#'
#' Rebuilds each term's grouping factor the way the formula parser built it, so
#' a level maps to the group the fit gave it, and rebuilds the slope covariates
#' from the new data.
#'
#' @keywords internal
prediction_re_design <- function(object, newdata, include_re, allow_new_levels) {
  n <- nrow(newdata)
  re_terms <- object$formula$numerator$random_effects
  n_terms <- length(re_terms %||% list())

  empty <- list(re_group = rep(0L, n), re_group_matrix = NULL,
                slope_matrices = NULL)
  if (!include_re || n_terms == 0L) return(empty)

  group_matrix <- matrix(0L, n, n_terms)
  slope_matrices <- vector("list", n_terms)

  for (t in seq_len(n_terms)) {
    term <- re_terms[[t]]
    group_matrix[, t] <- prediction_group_index(term, object$data, newdata,
                                                allow_new_levels)
    if (length(term$slope_vars %||% character(0)) > 0L) {
      expanded <- expand_re_slopes(
        strsplit(term$slope_vars_raw, " + ", fixed = TRUE)[[1]],
        newdata, include_intercept = FALSE
      )
      slope_matrices[[t]] <- expanded$slope_matrix[, term$slope_vars, drop = FALSE]
    }
  }

  list(re_group = group_matrix[, 1],
       re_group_matrix = group_matrix,
       slope_matrices = slope_matrices)
}


#' Group index of one random-effect term on new data
#' @keywords internal
prediction_group_index <- function(term, orig_data, newdata, allow_new_levels) {
  nested_vars <- term$nested_vars
  if (!is.null(nested_vars) && length(nested_vars) > 1L) {
    orig <- interaction(orig_data[nested_vars], drop = TRUE)
    new_levels <- interaction(newdata[nested_vars], drop = TRUE)
    levs <- levels(orig)
    labels <- as.character(new_levels)
  } else {
    var <- if (!is.null(nested_vars)) nested_vars[1] else term$group_var
    if (!var %in% names(newdata)) {
      stop("Grouping variable '", var, "' not found in newdata.", call. = FALSE)
    }
    levs <- levels(as.factor(orig_data[[var]]))
    labels <- as.character(newdata[[var]])
  }

  idx <- match(labels, levs)
  unseen <- is.na(idx)
  if (any(unseen) && !allow_new_levels) {
    stop("New level '", labels[which(unseen)[1]], "' found in '",
         term$group_var, "'. Set allow_new_levels = TRUE to predict at the ",
         "population level.", call. = FALSE)
  }
  idx[unseen] <- 0L
  as.integer(idx)
}


#' Predict for HMC backend
#' @keywords internal
predict_hmc <- function(object, newdata, type, re_formula, allow_new_levels,
                        coords.0, include_spatial, return_spatial) {

  # Build design matrices for new data
  pred_data <- build_prediction_data(object, newdata, re_formula, allow_new_levels)

  # Check for spatial model
  spatial <- object$spatial
  spatial_type <- detect_spatial_type(object)
  has_spatial <- !is.null(spatial_type) && include_spatial

  # Validate coords.0 for GP models
  if (has_spatial && spatial_type %in% c("gp", "hsgp", "msgp")) {
    if (is.null(coords.0)) {
      stop("coords.0 required for spatial prediction with GP/HSGP models.\n",
           "Provide a matrix of coordinates for new prediction locations.",
           call. = FALSE)
    }
    coords.0 <- as.matrix(coords.0)
    if (ncol(coords.0) != 2) {
      stop("coords.0 must be an N x 2 matrix of spatial coordinates.", call. = FALSE)
    }
    if (nrow(coords.0) != nrow(newdata)) {
      stop("coords.0 must have the same number of rows as newdata.", call. = FALSE)
    }
  }

  # The field at the new rows, on the link scale
  w_pred <- NULL
  w_samples <- NULL
  if (has_spatial) {
    spatial_result <- predict_spatial_hmc(object, newdata, coords.0, spatial_type,
                                          return_spatial)
    w_pred <- spatial_result$w_pred
    w_samples <- spatial_result$w_samples
  }

  pred_draws <- compute_predictions_hmc(object, pred_data, type, w_pred,
                                        w_shared = spatial$shared %||% TRUE)

  list(draws = pred_draws, w_samples = w_samples)
}


#' Detect spatial type from fitted object
#' @keywords internal
detect_spatial_type <- function(object) {
  spatial <- object$spatial
  if (is.null(spatial)) return(NULL)
  # The Laplace backend records the type alone
  if (is.character(spatial)) spatial <- list(type = spatial)

  # Check hmc_data for spatial type (string values)
  hmc_data <- object$.internal$hmc_data
  if (!is.null(hmc_data$gp_type) && is.character(hmc_data$gp_type)) {
    gp_type <- hmc_data$gp_type
    if (gp_type == "gp") return("gp")
    if (gp_type == "hsgp") return("hsgp")
    if (gp_type %in% c("msgp", "multiscale_gp")) return("msgp")
  }

  # Check spatial object class
  if (inherits(spatial, "ratiod_gp")) return("gp")
  if (inherits(spatial, "ratiod_hsgp")) return("hsgp")
  if (inherits(spatial, "ratiod_msgp")) return("msgp")

  # Check spatial$type field
  if (!is.null(spatial$type)) {
    if (spatial$type %in% c("icar", "car", "car_proper")) return("icar")
    if (spatial$type == "bym2") return("bym2")
    if (spatial$type == "gp") return("gp")
    if (spatial$type == "hsgp") return("hsgp")
    if (spatial$type %in% c("msgp", "multiscale_gp")) return("msgp")
  }

  NULL
}


#' Predict spatial effects for HMC backend
#' @keywords internal
predict_spatial_hmc <- function(object, newdata, coords.0, spatial_type, return_spatial) {
  refuse_restricted_field_prediction(object)
  hmc_data <- object$.internal$hmc_data
  samples <- object$.internal$samples

  if (spatial_type == "gp") {
    return(predict_spatial_gp(object, coords.0, return_spatial))
  } else if (spatial_type == "hsgp") {
    return(predict_spatial_hsgp(object, coords.0, return_spatial))
  } else if (spatial_type %in% c("icar", "bym2", "car", "car_proper")) {
    return(predict_spatial_areal(object, newdata, return_spatial))
  }

  list(w_pred = NULL, w_samples = NULL)
}


#' Predict GP spatial effects at new locations (kriging)
#'
#' Krigs each posterior draw of the field at the training locations onto
#' `coords.0`, put in the units the fit was made in, under that draw's
#' variance and range.
#'
#' @keywords internal
predict_spatial_gp <- function(object, coords.0, return_spatial) {
  gp <- object$.internal$spatial_spec
  w_train <- object$.internal$spatial_draws
  if (is.null(gp$unique_coords) || is.null(w_train)) {
    stop("This fit carries no GP field draws to krig from.", call. = FALSE)
  }
  draws <- object$draws
  w_pred <- cpp_kriging_predict(
    coords_train = gp$unique_coords,
    coords_new = gp_scale_new_coords(gp, coords.0),
    w_train = w_train,
    sigma2 = draws[, "sigma2_gp"],
    phi = draws[, "phi_gp"],
    cov_type = cov_type_code(gp$cov),
    nn = min(gp$nn, nrow(gp$unique_coords))
  )

  list(
    w_pred = w_pred,
    w_samples = if (return_spatial) w_pred else NULL
  )
}


#' Predict HSGP spatial effects at new locations (basis evaluation)
#'
#' Evaluates the fit's Laplacian eigenbasis, on the training rectangle, at
#' `coords.0` and applies each draw's spectral weights to it.
#'
#' @keywords internal
predict_spatial_hsgp <- function(object, coords.0, return_spatial) {
  gp <- object$.internal$spatial_spec
  lay <- object$.internal$layout$hsgp
  samples <- object$.internal$samples
  if (is.null(gp$coords_matrix) || is.null(lay) || identical(lay$type, "none")) {
    stop("This fit carries no HSGP field to evaluate.", call. = FALSE)
  }
  domain <- hsgp_domain_2d(gp$coords_matrix, gp$c)
  basis <- hsgp_basis_2d(gp_scale_new_coords(gp, coords.0), gp$m, gp$c, domain)
  w_pred <- hsgp_field(block_draws(samples, lay$beta),
                       exp(samples[, lay$log_sigma2]),
                       exp(samples[, lay$log_lengthscale]),
                       basis$phi, basis$eigenvalues)

  list(
    w_pred = w_pred,
    w_samples = if (return_spatial) w_pred else NULL
  )
}


#' Predict areal spatial effects (ICAR/BYM2 lookup)
#' @keywords internal
predict_spatial_areal <- function(object, newdata, return_spatial) {
  spatial <- object$spatial
  hmc_data <- object$.internal$hmc_data
  samples <- object$.internal$samples
  n_samples <- nrow(samples)

  # Get spatial grouping variable from newdata
  spatial_var <- spatial$group_var
  if (is.null(spatial_var) || !spatial_var %in% names(newdata)) {
    warning("Spatial group variable '", spatial_var, "' not found in newdata. ",
            "Spatial effects will be excluded from predictions.", call. = FALSE)
    return(list(w_pred = NULL, w_samples = NULL))
  }

  # Map new data spatial groups onto the units the fit numbered by factor level
  orig_data <- object$data
  orig_levels <- levels(as.factor(orig_data[[spatial_var]]))
  new_levels <- newdata[[spatial_var]]

  n_new <- nrow(newdata)
  spatial_group <- integer(n_new)
  for (i in seq_len(n_new)) {
    match_idx <- match(new_levels[i], orig_levels)
    if (!is.na(match_idx)) {
      spatial_group[i] <- match_idx
    }
    # Unmatched levels stay at 0 (no spatial effect)
  }

  # Find spatial effects in samples
  # For ICAR: phi_spatial[1:n_units]
  # For BYM2: sigma_spatial, rho, phi_scaled[1:n_units], theta[1:n_units]
  n_units <- length(orig_levels)
  idx <- hmc_data$p_num + hmc_data$p_denom
  if (hmc_data$n_re_groups > 0) {
    idx <- idx + 1 + hmc_data$n_re_groups
  }

  spatial_type <- spatial$type %||% "icar"

  # ICAR/car use the ICAR parameterization (log_tau, phi); car_proper adds a
  # logit_rho scalar between them.
  if (spatial_type %in% c("icar", "car")) {
    idx <- idx + 1  # skip tau_spatial
    phi_spatial <- samples[, (idx + 1):(idx + n_units), drop = FALSE]
    spatial_effect <- phi_spatial
  } else if (spatial_type == "car_proper") {
    idx <- idx + 2  # skip tau_spatial, logit_rho
    phi_spatial <- samples[, (idx + 1):(idx + n_units), drop = FALSE]
    spatial_effect <- phi_spatial
  } else if (spatial_type == "bym2") {
    sigma_spatial <- exp(samples[, idx + 1])
    rho <- 1 / (1 + exp(-samples[, idx + 2]))
    phi_scaled <- samples[, (idx + 3):(idx + 2 + n_units), drop = FALSE]
    theta <- samples[, (idx + 3 + n_units):(idx + 2 + 2 * n_units), drop = FALSE]

    bym2_scale <- spatial$scale_factor %||% 1.0

    spatial_effect <- matrix(0, n_samples, n_units)
    for (s in seq_len(n_samples)) {
      spatial_effect[s, ] <- sigma_spatial[s] * (
        sqrt(rho[s]) * phi_scaled[s, ] * bym2_scale +
        sqrt(1 - rho[s]) * theta[s, ]
      )
    }
  }

  # Look up spatial effects for new data
  w_pred <- matrix(0, n_samples, n_new)
  for (i in seq_len(n_new)) {
    g <- spatial_group[i]
    if (g > 0) {
      w_pred[, i] <- spatial_effect[, g]
    }
  }

  list(
    w_pred = w_pred,
    w_samples = if (return_spatial) w_pred else NULL
  )
}


#' Refuse a spatial prediction from a restricted field
#'
#' A restricted field (`spatial_rsr()`) is the projection of the field across
#' the fitted observations, so it has no value at a new row.
#'
#' @param object A fitted model
#' @keywords internal
refuse_restricted_field_prediction <- function(object) {
  if (is.null(object$spatial$rsr_basis)) return(invisible(NULL))
  stop("A restricted spatial field (`spatial_rsr()`) is defined through a ",
       "projection over the fitted observations and has no value at a new ",
       "row. Use `include_spatial = FALSE`, or fitted() for the in-sample fit.",
       call. = FALSE)
}


#' Predict for PG backend
#' @keywords internal
predict_pg <- function(object, newdata, type, re_formula, allow_new_levels,
                       coords.0, include_spatial, return_spatial) {
  if (include_spatial) refuse_restricted_field_prediction(object)
  # PG backend stores samples like HMC
  # Build design matrices for new data
  pred_data <- build_prediction_data_pg(object, newdata, re_formula, allow_new_levels)

  # Compute predictions from PG samples
  pred_draws <- compute_predictions_pg(object, pred_data, type)

  # Spatial effects (PG supports ICAR, BYM2, GP)
  w_samples <- NULL
  if (include_spatial && !is.null(object$spatial)) {
    spatial_type <- object$spatial_type %||% object$spatial$type
    if (!is.null(spatial_type)) {
      if (spatial_type == "gp" && !is.null(coords.0)) {
        spatial_result <- predict_spatial_gp_pg(object, coords.0, return_spatial)
      } else if (spatial_type %in% c("icar", "bym2", "car", "car_proper")) {
        spatial_result <- predict_spatial_areal_pg(object, newdata, return_spatial)
      } else {
        spatial_result <- list(w_pred = NULL, w_samples = NULL)
      }

      w_samples <- spatial_result$w_samples
      if (!is.null(spatial_result$w_pred)) {
        pred_draws <- add_spatial_to_predictions_pg(pred_draws, spatial_result$w_pred, type)
      }
    }
  }

  list(draws = pred_draws, w_samples = w_samples)
}


#' Build prediction data for PG backend
#' @keywords internal
build_prediction_data_pg <- function(object, newdata, re_formula, allow_new_levels) {
  formula <- object$formula

  X <- design_matrix_newdata(formula$numerator$design, newdata)

  # Handle random effects
  include_re <- !isTRUE(is.na(re_formula)) && !identical(re_formula, ~ 0)
  re_group <- rep(0L, nrow(newdata))

  if (include_re) {
    re_info <- object$.internal$re_info
    if (!is.null(re_info) && re_info$n_groups > 0) {
      re_var <- re_info$group_var
      if (!is.null(re_var) && re_var %in% names(newdata)) {
        orig_data <- object$data
        orig_levels <- levels(as.factor(orig_data[[re_var]]))
        for (i in seq_len(nrow(newdata))) {
          match_idx <- match(newdata[[re_var]][i], orig_levels)
          if (!is.na(match_idx)) {
            re_group[i] <- match_idx
          } else if (!allow_new_levels) {
            stop("New level found in '", re_var, "'. Set allow_new_levels = TRUE.",
                 call. = FALSE)
          }
        }
      }
    }
  }

  list(X = X, re_group = re_group, N = nrow(newdata), include_re = include_re)
}


#' Compute predictions for PG backend
#' @keywords internal
compute_predictions_pg <- function(object, pred_data, type) {
  beta <- object$.internal$beta
  re <- object$.internal$re
  n_samples <- nrow(beta)
  N <- pred_data$N

  # Compute linear predictor
  eta <- matrix(NA_real_, n_samples, N)
  for (s in seq_len(n_samples)) {
    eta[s, ] <- as.numeric(pred_data$X %*% beta[s, ])
    if (pred_data$include_re && !is.null(re)) {
      for (i in seq_len(N)) {
        g <- pred_data$re_group[i]
        if (g > 0 && g <= ncol(re)) {
          eta[s, i] <- eta[s, i] + re[s, g]
        }
      }
    }
  }

  if (type == "link") {
    return(list(
      numerator = eta,
      denominator = matrix(0, n_samples, N),
      ratio = eta
    ))
  }

  # Response scale (binomial -> probability)
  prob <- 1 / (1 + exp(-eta))
  list(
    numerator = prob,
    denominator = matrix(1, n_samples, N),
    ratio = prob
  )
}


#' Predict spatial effects for PG GP models
#' @keywords internal
predict_spatial_gp_pg <- function(object, coords.0, return_spatial) {
  internal <- object$.internal
  w_gp <- internal$w_gp
  if (is.null(w_gp)) return(list(w_pred = NULL, w_samples = NULL))

  # The unique locations the field is indexed by, not the observation-order
  # matrix: a design with several observations per location has more rows of
  # data than it has field entries.
  coords_train <- internal$gp_coords %||% object$spatial$unique_coords
  if (is.null(coords_train)) {
    warning("Training coordinates not found. Cannot predict spatial effects.",
            call. = FALSE)
    return(list(w_pred = NULL, w_samples = NULL))
  }

  hyper <- internal$gp_hyper
  if (is.null(hyper) || nrow(hyper) != nrow(w_gp)) {
    warning("The GP hyperparameter draws are not aligned with the field ",
            "draws. Cannot predict spatial effects.", call. = FALSE)
    return(list(w_pred = NULL, w_samples = NULL))
  }

  w_pred <- cpp_kriging_predict(
    coords_train = as.matrix(coords_train),
    coords_new = gp_scale_new_coords(object$spatial, coords.0),
    w_train = w_gp,
    sigma2 = hyper$sigma2,
    phi = hyper$phi,
    cov_type = cov_type_code(object$spatial$cov),
    nn = object$spatial$nn %||% 15L
  )

  list(w_pred = w_pred, w_samples = if (return_spatial) w_pred else NULL)
}


#' Predict areal spatial effects for PG backend
#' @keywords internal
predict_spatial_areal_pg <- function(object, newdata, return_spatial) {
  spatial_draws <- object$.internal$spatial
  if (is.null(spatial_draws)) return(list(w_pred = NULL, w_samples = NULL))

  spatial <- object$spatial
  spatial_var <- spatial$group_var

  if (is.null(spatial_var) || !spatial_var %in% names(newdata)) {
    return(list(w_pred = NULL, w_samples = NULL))
  }

  # Map groups onto the units the fit numbered by factor level
  orig_data <- object$data
  orig_levels <- levels(as.factor(orig_data[[spatial_var]]))
  n_new <- nrow(newdata)
  n_samples <- nrow(spatial_draws)

  spatial_group <- integer(n_new)
  for (i in seq_len(n_new)) {
    match_idx <- match(newdata[[spatial_var]][i], orig_levels)
    if (!is.na(match_idx)) spatial_group[i] <- match_idx
  }

  # Look up
  w_pred <- matrix(0, n_samples, n_new)
  for (i in seq_len(n_new)) {
    g <- spatial_group[i]
    if (g > 0 && g <= ncol(spatial_draws)) {
      w_pred[, i] <- spatial_draws[, g]
    }
  }

  list(w_pred = w_pred, w_samples = if (return_spatial) w_pred else NULL)
}


#' Add spatial effects to PG predictions
#' @keywords internal
add_spatial_to_predictions_pg <- function(pred_draws, w_pred, type) {
  if (type == "link") {
    pred_draws$numerator <- pred_draws$numerator + w_pred
    pred_draws$ratio <- pred_draws$numerator
  } else {
    eta <- log(pred_draws$numerator / (1 - pred_draws$numerator)) + w_pred
    pred_draws$numerator <- 1 / (1 + exp(-eta))
    pred_draws$ratio <- pred_draws$numerator
  }
  pred_draws
}


#' Predict for Gibbs backend
#'
#' @description
#' Posterior predictive draws for the Gibbs ICAR/BYM2 spatial sampler.
#' Mirrors `predict_pg`: builds the design matrix for `newdata`, looks up
#' each new row's spatial group against the levels stored in the fit, and
#' assembles `eta = X beta + phi[group]` per draw. For binomial families
#' (the supported case under the gibbs backend), the response-scale
#' prediction is `plogis(eta)`. Continuous-denominator families
#' (poisson_gamma, negbin_gamma) and count-denominator families
#' (negbin_negbin) add `eta_denom = X_denom beta_denom + phi[group]` and
#' return `mu_num / mu_denom` on the response scale.
#'
#' @keywords internal
predict_gibbs <- function(object, newdata, type, re_formula, allow_new_levels,
                          include_spatial, return_spatial) {
  family_str <- get_gibbs_family_string(object$family)

  # Build design matrices for newdata. Reuse the PG builder for the
  # numerator (same model.matrix discipline). For two-process families
  # we also need a denominator design matrix.
  pred_data <- build_prediction_data_pg(object, newdata, re_formula, allow_new_levels)
  X_new <- pred_data$X
  n_new <- nrow(newdata)

  draws_mat <- object$draws
  if (is.array(draws_mat) && length(dim(draws_mat)) == 3L) {
    # [iter, chain, var] -> [iter, var]
    draws_mat <- matrix(draws_mat, nrow = dim(draws_mat)[1L],
                        ncol = dim(draws_mat)[3L],
                        dimnames = list(NULL, dimnames(draws_mat)[[3L]]))
  }
  n_samples <- nrow(draws_mat)

  p_num <- ncol(object$.internal$X)
  beta_num_cols <- seq_len(p_num)
  beta_num <- draws_mat[, beta_num_cols, drop = FALSE]

  has_denom <- !(family_str == "binomial")
  if (has_denom) {
    X_denom_new <- design_matrix_newdata(object$formula$denominator$design,
                                         newdata)
    p_denom <- ncol(X_denom_new)
    beta_denom_cols <- p_num + seq_len(p_denom)
    beta_denom <- draws_mat[, beta_denom_cols, drop = FALSE]
  } else {
    X_denom_new <- NULL
    beta_denom <- NULL
  }

  # Map newdata rows to original spatial group indices.
  spatial <- object$spatial
  spatial_var <- spatial$group_var
  phi_draws <- object$.internal$phi_draws  # [n_save, S]
  if (is.null(phi_draws) || is.null(spatial_var) ||
      !spatial_var %in% names(newdata)) {
    stop("Gibbs predict requires spatial group_var to be present in newdata.",
         call. = FALSE)
  }

  group_factor_orig <- as.factor(object$data[[spatial_var]])
  orig_levels <- levels(group_factor_orig)
  new_levels <- as.character(newdata[[spatial_var]])
  group_idx <- match(new_levels, orig_levels)

  if (any(is.na(group_idx))) {
    if (!isTRUE(allow_new_levels)) {
      missing <- unique(new_levels[is.na(group_idx)])
      stop("New level(s) '", paste(missing, collapse = "', '"),
           "' found in '", spatial_var,
           "'. Set allow_new_levels = TRUE to predict at population level.",
           call. = FALSE)
    }
    group_idx[is.na(group_idx)] <- 0L
  }

  # Per-draw linear predictor: eta_num[s, i] = X_new[i,] %*% beta_num[s,] +
  # (include_spatial ? phi[s, group_idx[i]] : 0). w_samples = phi columns
  # corresponding to newdata rows (n_samples x n_new) when caller asks.
  w_samples <- matrix(0, n_samples, n_new)
  if (isTRUE(include_spatial)) {
    for (i in seq_len(n_new)) {
      g <- group_idx[i]
      if (g >= 1L && g <= ncol(phi_draws)) {
        w_samples[, i] <- phi_draws[, g]
      }
    }
  }

  eta_num <- X_new %*% t(beta_num) + t(w_samples)
  # eta_num is [n_new, n_samples]; transpose to [n_samples, n_new]
  eta_num <- t(eta_num)

  if (has_denom) {
    eta_denom <- X_denom_new %*% t(beta_denom) + t(w_samples)
    eta_denom <- t(eta_denom)
  }

  if (identical(type, "link")) {
    if (has_denom) {
      pred_draws <- list(
        numerator = eta_num,
        denominator = eta_denom,
        ratio = eta_num - eta_denom
      )
    } else {
      pred_draws <- list(
        numerator = eta_num,
        denominator = matrix(0, n_samples, n_new),
        ratio = eta_num
      )
    }
  } else {
    if (family_str == "binomial") {
      prob <- 1 / (1 + exp(-eta_num))
      pred_draws <- list(
        numerator = prob,
        denominator = matrix(1, n_samples, n_new),
        ratio = prob
      )
    } else {
      mu_num <- exp(eta_num)
      mu_denom <- exp(eta_denom)
      pred_draws <- list(
        numerator = mu_num,
        denominator = mu_denom,
        ratio = mu_num / mu_denom
      )
    }
  }

  list(
    draws = pred_draws,
    w_samples = if (isTRUE(return_spatial)) w_samples else NULL
  )
}


#' Predict for Laplace backend
#' @keywords internal
predict_laplace <- function(object, newdata, type, re_formula, allow_new_levels,
                            coords.0, include_spatial, return_spatial, n_samples) {
  internal <- object$.internal
  layout <- internal$layout
  if (is.null(internal$samples) || is.null(layout)) {
    stop("This Laplace fit stores no posterior samples. Cannot generate predictions.",
         call. = FALSE)
  }

  structured <- Filter(function(p) !identical(p$role, "re"), layout$parts)
  if (length(structured) > 0L) {
    stop("predict() on a Laplace fit that carries a spatial or temporal field ",
         "is not implemented, since the field has no value at a new row. ",
         "Use fitted() for the in-sample fit.", call. = FALSE)
  }

  pred_data <- build_prediction_data(object, newdata, re_formula, allow_new_levels)
  N <- pred_data$N
  samples <- internal$samples

  eta_num <- samples[, layout$beta$num, drop = FALSE] %*% t(pred_data$X_num)
  eta_denom <- if (!is.null(layout$beta$denom)) {
    samples[, layout$beta$denom, drop = FALSE] %*% t(pred_data$X_denom)
  } else {
    matrix(0, nrow(eta_num), ncol(eta_num))
  }

  if (isTRUE(pred_data$include_re)) {
    re_parts <- Filter(function(p) identical(p$role, "re"), layout$parts)
    for (t in seq_along(re_parts)) {
      g <- pred_data$re_group_matrix[, t]
      effect <- spread_effect(samples[, re_parts[[t]]$cols, drop = FALSE], g, N)
      eta_num <- eta_num + effect
      eta_denom <- eta_denom + effect
    }
  }

  pred_draws <- hmc_response_draws(list(eta_num = eta_num, eta_denom = eta_denom),
                                   fit_links(object), type)
  list(draws = pred_draws, w_samples = NULL)
}


#' Simple MVN sampler (avoid mvtnorm dependency)
#' @keywords internal
mvtnorm_rmvnorm <- function(n, mean, sigma) {
  p <- length(mean)
  # Cholesky decomposition
  L <- tryCatch(chol(sigma), error = function(e) {
    # Add jitter if not positive definite
    chol(sigma + diag(1e-6, p))
  })
  Z <- matrix(rnorm(n * p), n, p)
  sweep(Z %*% L, 2, mean, "+")
}


#' Predict for VI backend
#' @keywords internal
predict_vi <- function(object, newdata, type, re_formula, allow_new_levels,
                       coords.0, include_spatial, return_spatial, n_samples) {
  # VI stores variational parameters
  # Sample from variational posterior q(theta)

  vi_params <- object$.internal$vi_params
  if (is.null(vi_params)) {
    stop("VI fit missing variational parameters.", call. = FALSE)
  }

  # Sample from variational posterior
  samples <- sample_vi_posterior(vi_params, n_samples)

  # Build prediction data
  pred_data <- build_prediction_data(object, newdata, re_formula, allow_new_levels)

  # Create temporary object
  temp_object <- object
  temp_object$.internal$samples <- samples

  pred_draws <- compute_predictions_hmc(temp_object, pred_data, type)

  list(draws = pred_draws, w_samples = NULL)
}


#' Sample from VI variational posterior
#' @keywords internal
sample_vi_posterior <- function(vi_params, n_samples) {
  # Mean-field: q(theta) = prod N(mu_i, sigma_i^2)
  mu <- vi_params$mu
  sigma <- vi_params$sigma

  if (is.null(mu) || is.null(sigma)) {
    stop("VI parameters missing mu or sigma.", call. = FALSE)
  }

  p <- length(mu)
  Z <- matrix(rnorm(n_samples * p), n_samples, p)
  sweep(Z * matrix(sigma, n_samples, p, byrow = TRUE), 2, mu, "+")
}


#' Compute predictions for HMC backend
#'
#' Fixed effects and random effects, remapped onto the new data, plus the
#' spatial field at the new rows when `predict_spatial_hmc()` supplies one. The
#' field enters the linear predictors, and the response is taken once from
#' them, as `fitted()` takes it.
#'
#' @param w_pred Draws of the field at the new rows (draws by rows), or `NULL`
#' @param w_shared Whether the field enters the denominator's predictor too
#' @keywords internal
compute_predictions_hmc <- function(object, pred_data, type, w_pred = NULL,
                                    w_shared = TRUE) {
  unpacked <- hmc_fit_unpack(object)
  design <- hmc_eta_design(
    X_num = pred_data$X_num,
    X_denom = pred_data$X_denom,
    re_group = pred_data$re_group,
    re_group_matrix = pred_data$re_group_matrix,
    slope_matrices = pred_data$slope_matrices,
    include_re = isTRUE(pred_data$include_re),
    structures = FALSE
  )
  eta <- hmc_eta_draws(unpacked, design)
  if (!is.null(w_pred)) {
    eta$eta_num <- eta$eta_num + w_pred
    if (isTRUE(w_shared)) eta$eta_denom <- eta$eta_denom + w_pred
  }
  hmc_response_draws(eta, fit_links(object), type, exposure = pred_data$exposure)
}


#' @rdname fitted.ratiod_fit
#' @param x A ratiod_fitted object
#' @param n Number of rows to print
#'
#' @export
print.ratiod_fitted <- function(x, n = 10, ...) {
  cat("tulpaRatio fitted values\n")
  cat("===================\n\n")
  cat("Posterior draws:", attr(x, "n_draws"), "\n\n")

  if (nrow(x) > n) {
    print.data.frame(head(x, n), row.names = FALSE, ...)
    cat(sprintf("\n... and %d more rows\n", nrow(x) - n))
  } else {
    print.data.frame(x, row.names = FALSE, ...)
  }
  invisible(x)
}


#' @rdname predict.ratiod_fit
#' @param x A ratiod_prediction object
#' @param n Number of rows to print
#'
#' @export
print.ratiod_prediction <- function(x, n = 10, ...) {
  cat("tulpaRatio predictions\n")
  cat("=================\n\n")
  cat("Posterior draws:", attr(x, "n_draws"), "\n")
  cat("New observations:", nrow(attr(x, "newdata")), "\n\n")

  if (nrow(x) > n) {
    print.data.frame(head(x, n), row.names = FALSE, ...)
    cat(sprintf("\n... and %d more rows\n", nrow(x) - n))
  } else {
    print.data.frame(x, row.names = FALSE, ...)
  }
  invisible(x)
}


# -----------------------------------------------------------------------------
# Tidy draws
# -----------------------------------------------------------------------------

#' Convert a Fitted Ratio Model to a Draws Object
#'
#' @description
#' Convert a fitted ratio model to a format compatible with the
#' posterior and tidybayes packages. Returns draws in a tidy format
#' suitable for further analysis and visualization.
#'
#' @param x A `ratiod_fit` object
#' @param ... Additional arguments passed to `posterior::as_draws_df`
#'
#' @return A `draws_df` object from the posterior package.
#'
#' @details
#' The returned draws object is compatible with:
#' - **posterior** package functions (`summarise_draws`, `rvar`, etc.)
#' - **tidybayes** package functions (`spread_draws`, `gather_draws`, etc.)
#' - **bayesplot** package visualization functions
#'
#' To work in tidybayes, extract the draws with this function and pass them on
#' to its verbs.
#'
#' @examples
#' \donttest{
#' # Generate synthetic data
#' set.seed(111)
#' n <- 48
#' site <- rep(letters[1:12], each = 4)
#' u <- rnorm(12, sd = 0.5)[match(site, letters[1:12])]
#' depth <- rnorm(n)
#' df <- data.frame(
#'   count = rpois(n, exp(3 + 0.3 * depth + u)),
#'   effort = rgamma(n, shape = 11, rate = exp(-u)),
#'   depth = depth,
#'   site = site
#' )
#'
#' # Fit model
#' fit <- tratio(
#'   count | effort ~ depth + (1 | site),
#'   data = df,
#'   family = ratiod_poisson_gamma(),
#'   mode = "hmc",
#'   control = list(iter = 400, chains = 1, seed = 1)
#' )
#'
#' # Convert to draws format
#' draws <- as_draws(fit)
#'
#' # Use with posterior package
#' if (requireNamespace("posterior", quietly = TRUE)) {
#'   posterior::summarise_draws(draws)
#' }
#' }
#'
#' @seealso [posterior::as_draws_df()], [ratio()] for ratio-specific extraction
#'
#' @rdname as_draws
#' @export
as_draws.ratiod_fit <- function(x, ...) {
  if (!requireNamespace("posterior", quietly = TRUE)) {
    stop("Package 'posterior' is required. Install with:\n",
         "  install.packages('posterior')", call. = FALSE)
  }

  draws <- x$draws
  if (is.null(draws)) {
    stop("No draws available in the fitted model", call. = FALSE)
  }

  # Get chain information if available
  n_chains <- x$chains %||% 1
  n_total <- nrow(draws)
  n_per_chain <- n_total / n_chains

  if (n_chains > 1) {
    # Reconstruct chain structure
    draws_array <- array(
      as.matrix(draws),
      dim = c(n_per_chain, n_chains, ncol(draws)),
      dimnames = list(
        iteration = seq_len(n_per_chain),
        chain = seq_len(n_chains),
        variable = colnames(draws)
      )
    )
    posterior::as_draws_df(posterior::as_draws_array(draws_array), ...)
  } else {
    posterior::as_draws_df(draws, ...)
  }
}


#' Draws for Selected Parameters in Wide Format
#'
#' @description
#' Extract posterior draws for the named parameters into a wide data frame,
#' one row per draw and one column per parameter, alongside the `.chain`,
#' `.iteration` and `.draw` bookkeeping columns.
#'
#' Parameters are named unquoted, and a name matches every indexed element it
#' prefixes, so `beta_num` reaches `beta_num[1]`, `beta_num[2]` and so on.
#'
#' @param object A `ratiod_fit` object
#' @param ... Parameter names to extract (unquoted). Supports patterns like
#'   `beta_num` or `beta_num[i]` for indexed parameters.
#' @param regex Logical; if TRUE, treat parameter names as regex patterns.
#' @param ndraws Number of draws to return. If NULL, returns all.
#'
#' @return A data frame with columns `.chain`, `.iteration`, `.draw`, and
#'   one column per requested parameter.
#'
#' @examples
#' \donttest{
#' # Generate synthetic data
#' set.seed(222)
#' n <- 48
#' site <- rep(letters[1:12], each = 4)
#' u <- rnorm(12, sd = 0.5)[match(site, letters[1:12])]
#' depth <- rnorm(n)
#' df <- data.frame(
#'   count = rpois(n, exp(3 + 0.3 * depth + u)),
#'   effort = rgamma(n, shape = 10, rate = exp(-u)),
#'   depth = depth,
#'   site = site
#' )
#'
#' # Fit model
#' fit <- tratio(
#'   count | effort ~ depth + (1 | site),
#'   data = df,
#'   family = ratiod_poisson_gamma(),
#'   mode = "hmc",
#'   control = list(iter = 400, chains = 1, seed = 1)
#' )
#'
#' # Extract fixed effects
#' draws_wide(fit, beta_num, beta_denom)
#'
#' # Subsample draws
#' draws_wide(fit, sigma_re, ndraws = 100)
#' }
#'
#' @export
draws_wide <- function(object, ..., regex = FALSE, ndraws = NULL) {
  UseMethod("draws_wide")
}


#' @rdname draws_wide
#' @export
draws_wide.ratiod_fit <- function(object, ..., regex = FALSE, ndraws = NULL) {

  # Get all parameter names
  all_pars <- colnames(object$draws)

  # Parse requested parameters
  dots <- as.character(substitute(list(...)))[-1]

  if (length(dots) == 0) {
    # No parameters specified - return all main parameters
    pars <- select_main_params(all_pars)
  } else {
    pars <- character(0)
    for (pattern in dots) {
      # Remove backticks and index notation
      pattern_clean <- gsub("`", "", pattern)
      pattern_clean <- gsub("\\[.*\\]$", "", pattern_clean)

      if (regex) {
        matches <- grep(pattern_clean, all_pars, value = TRUE)
      } else {
        # Match exact or with indices
        matches <- grep(paste0("^", pattern_clean, "(\\[|$)"), all_pars, value = TRUE)
      }
      pars <- c(pars, matches)
    }
    pars <- unique(pars)
  }

  if (length(pars) == 0) {
    stop("No parameters matched the specified patterns", call. = FALSE)
  }

  # Extract draws
  draws <- object$draws[, pars, drop = FALSE]

  # Add metadata columns
  n_chains <- object$chains %||% 1
  n_total <- nrow(draws)
  n_per_chain <- n_total / n_chains

  result <- data.frame(
    .chain = rep(seq_len(n_chains), each = n_per_chain),
    .iteration = rep(seq_len(n_per_chain), times = n_chains),
    .draw = seq_len(n_total),
    draws,
    check.names = FALSE
  )

  # Subsample if requested
  if (!is.null(ndraws) && ndraws < n_total) {
    idx <- sample(n_total, ndraws)
    result <- result[idx, ]
    result$.draw <- seq_len(ndraws)
  }

  class(result) <- c("ratiod_draws", "data.frame")
  result
}


#' Draws for Selected Parameters in Long Format
#'
#' @description
#' The long counterpart of [draws_wide()]: one row per draw per parameter,
#' with the parameter in `.variable` and its value in `.value`.
#'
#' @param object A `ratiod_fit` object
#' @param ... Parameter names to extract (unquoted).
#' @param regex Logical; if TRUE, treat parameter names as regex patterns.
#' @param ndraws Number of draws to return. If NULL, returns all.
#'
#' @return A data frame with columns `.chain`, `.iteration`, `.draw`,
#'   `.variable`, and `.value`.
#'
#' @examples
#' \donttest{
#' # Generate synthetic data
#' set.seed(333)
#' n <- 48
#' site <- rep(letters[1:12], each = 4)
#' u <- rnorm(12, sd = 0.5)[match(site, letters[1:12])]
#' depth <- rnorm(n)
#' df <- data.frame(
#'   count = rpois(n, exp(3 + 0.3 * depth + u)),
#'   effort = rgamma(n, shape = 12, rate = exp(-u)),
#'   depth = depth,
#'   site = site
#' )
#'
#' # Fit model
#' fit <- tratio(
#'   count | effort ~ depth + (1 | site),
#'   data = df,
#'   family = ratiod_poisson_gamma(),
#'   mode = "hmc",
#'   control = list(iter = 400, chains = 1, seed = 1)
#' )
#'
#' # Gather all fixed effects
#' draws_long(fit, beta_num, beta_denom)
#'
#' # Use for plotting
#' if (requireNamespace("ggplot2", quietly = TRUE)) {
#'   library(ggplot2)
#'   draws_long(fit, beta_num) |>
#'     ggplot(aes(x = .value)) +
#'     geom_histogram(bins = 40) +
#'     facet_wrap(~ .variable)
#' }
#' }
#'
#' @export
draws_long <- function(object, ..., regex = FALSE, ndraws = NULL) {
  UseMethod("draws_long")
}


#' @rdname draws_long
#' @export
draws_long.ratiod_fit <- function(object, ..., regex = FALSE, ndraws = NULL) {

  # Get wide format first
  wide <- draws_wide(object, ..., regex = regex, ndraws = ndraws)

  # Get parameter columns (exclude metadata)
  meta_cols <- c(".chain", ".iteration", ".draw")
  par_cols <- setdiff(names(wide), meta_cols)

  # Reshape to long format
  long <- reshape(
    wide,
    direction = "long",
    varying = par_cols,
    v.names = ".value",
    timevar = ".variable",
    times = par_cols
  )

  # Clean up
  long <- long[, c(".chain", ".iteration", ".draw", ".variable", ".value")]
  rownames(long) <- NULL

  class(long) <- c("ratiod_draws_long", "data.frame")
  long
}


#' Point Estimates and Credible Intervals for Selected Parameters
#'
#' @description
#' Summarize the draws of the named parameters as a point estimate (median or
#' mean) and a credible interval (quantile or highest-density), one row per
#' parameter per interval width.
#'
#' @param object A `ratiod_fit` object
#' @param ... Parameter names to summarize (unquoted).
#' @param .width Width(s) of credible intervals. Default 0.95.
#' @param .point Point estimate function: "median" (default) or "mean".
#' @param .interval Interval type: "qi" for quantile interval (default),
#'   "hdi" for highest density interval.
#'
#' @return A data frame with columns for the variable name, point estimate,
#'   and interval bounds.
#'
#' @examples
#' \donttest{
#' # Generate synthetic data
#' set.seed(444)
#' n <- 48
#' site <- rep(letters[1:12], each = 4)
#' u <- rnorm(12, sd = 0.5)[match(site, letters[1:12])]
#' depth <- rnorm(n)
#' df <- data.frame(
#'   count = rpois(n, exp(3 + 0.3 * depth + u)),
#'   effort = rgamma(n, shape = 9, rate = exp(-u)),
#'   depth = depth,
#'   site = site
#' )
#'
#' # Fit model
#' fit <- tratio(
#'   count | effort ~ depth + (1 | site),
#'   data = df,
#'   family = ratiod_poisson_gamma(),
#'   mode = "hmc",
#'   control = list(iter = 400, chains = 1, seed = 1)
#' )
#'
#' # Get point estimates and 95% intervals
#' draws_interval(fit, beta_num, beta_denom)
#'
#' # Multiple interval widths
#' draws_interval(fit, beta_num, .width = c(0.5, 0.9, 0.95))
#' }
#'
#' @export
draws_interval <- function(object, ..., .width = 0.95,
                           .point = c("median", "mean"),
                           .interval = c("qi", "hdi")) {
  UseMethod("draws_interval")
}


#' @rdname draws_interval
#' @export
draws_interval.ratiod_fit <- function(object, ..., .width = 0.95,
                                      .point = c("median", "mean"),
                                      .interval = c("qi", "hdi")) {

  .point <- match.arg(.point)
  .interval <- match.arg(.interval)

  # Get draws
  draws_wide <- draws_wide(object, ...)
  meta_cols <- c(".chain", ".iteration", ".draw")
  par_cols <- setdiff(names(draws_wide), meta_cols)

  if (length(par_cols) == 0) {
    stop("No parameters matched", call. = FALSE)
  }

  # Compute summaries for each parameter and interval width
  results <- list()

  for (par in par_cols) {
    x <- draws_wide[[par]]

    for (w in .width) {
      point_val <- if (.point == "median") median(x) else mean(x)

      if (.interval == "qi") {
        # Quantile interval
        alpha <- (1 - w) / 2
        lower <- quantile(x, alpha)
        upper <- quantile(x, 1 - alpha)
      } else {
        # HDI - simple implementation
        sorted <- sort(x)
        n <- length(sorted)
        ci_n <- ceiling(w * n)
        widths <- sorted[(ci_n + 1):n] - sorted[1:(n - ci_n)]
        min_idx <- which.min(widths)
        lower <- sorted[min_idx]
        upper <- sorted[min_idx + ci_n]
      }

      results[[length(results) + 1]] <- data.frame(
        .variable = par,
        .value = point_val,
        .lower = lower,
        .upper = upper,
        .width = w,
        .point = .point,
        .interval = .interval,
        stringsAsFactors = FALSE
      )
    }
  }

  result <- do.call(rbind, results)
  rownames(result) <- NULL
  class(result) <- c("ratiod_draws_interval", "data.frame")
  result
}
