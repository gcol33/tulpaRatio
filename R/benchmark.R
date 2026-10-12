#' Benchmark Sampler Performance
#'
#' @description
#' Utilities for benchmarking ratio model fitting performance across
#' different dataset sizes and configurations.
#'
#' @name ratiod_benchmark
NULL


#' Run a simple benchmark
#'
#' @description
#' Fits a model on simulated data and reports timing and performance metrics.
#'
#' @param N Number of observations
#' @param p Number of predictors
#' @param n_groups Number of random effect groups (0 for none)
#' @param family Model family (default: ratiod_negbin_negbin())
#' @param n_iter Total iterations
#' @param n_warmup Warmup iterations
#' @param n_chains Number of chains
#' @param n_threads Number of threads per chain
#' @param verbose Print progress (default: FALSE)
#'
#' @return A list with timing and diagnostic information
#'
#' @examples
#' \donttest{
#' # Quick benchmark
#' bench <- ratiod_benchmark(N = 100, p = 3, n_iter = 200, n_warmup = 100)
#' print(bench)
#' }
#'
#' @export
ratiod_benchmark <- function(
    N = 500,
    p = 3,
    n_groups = 0,
    family = ratiod_negbin_negbin(),
    n_iter = 1000,
    n_warmup = 500,
    n_chains = 1,
    n_threads = 1,
    verbose = FALSE
) {

  # Generate simulated data
  set.seed(42)
  X <- cbind(1, matrix(rnorm(N * (p - 1)), nrow = N))

  # True parameters
  true_beta_num <- c(2.0, rep(0.3, p - 1))
  true_beta_denom <- c(2.5, rep(-0.2, p - 1))
  true_phi <- 5.0

  # Linear predictors
  eta_num <- X %*% true_beta_num
  eta_denom <- X %*% true_beta_denom

  # Add random effects if requested
  if (n_groups > 0) {
    group <- sample(1:n_groups, N, replace = TRUE)
    sigma_re <- 0.5
    re_true <- rnorm(n_groups, 0, sigma_re)
    eta_num <- eta_num + re_true[group]
    eta_denom <- eta_denom + re_true[group]
  } else {
    group <- rep(NA, N)
  }

  # Generate response
  mu_num <- exp(eta_num)
  mu_denom <- exp(eta_denom)
  y_num <- rnbinom(N, size = true_phi, mu = mu_num)
  y_denom <- rnbinom(N, size = true_phi, mu = mu_denom)

  # Create data frame
  df <- data.frame(
    y_num = y_num,
    y_denom = y_denom,
    X[, -1, drop = FALSE]
  )
  colnames(df)[3:ncol(df)] <- paste0("x", 1:(p - 1))

  if (n_groups > 0) {
    df$group <- factor(group)
  }

  # Build formula
  pred_terms <- paste(paste0("x", 1:(p - 1)), collapse = " + ")
  if (n_groups > 0) {
    formula_str <- sprintf("y_num | y_denom ~ %s + (1 | group)", pred_terms)
  } else {
    formula_str <- sprintf("y_num | y_denom ~ %s", pred_terms)
  }
  formula <- as.formula(formula_str)

  # Warm up JIT/compilation
  invisible(gc())

  # Time the fit
  start_time <- Sys.time()

  fit <- tratio(
    formula,
    data = df,
    family = family,
    control = list(iter = n_iter, warmup = n_warmup, chains = n_chains)
  )

  end_time <- Sys.time()
  elapsed <- as.numeric(difftime(end_time, start_time, units = "secs"))

  # Compute metrics
  n_samples <- (n_iter - n_warmup) * n_chains
  samples_per_sec <- n_samples / elapsed
  n_params <- ncol(fit$draws)

  # Divergent transitions
  n_divergent <- if (!is.null(fit$diagnostics$divergent)) {
    sum(fit$diagnostics$divergent)
  } else {
    NA
  }

  result <- list(
    # Configuration
    config = list(
      N = N,
      p = p,
      n_groups = n_groups,
      n_iter = n_iter,
      n_warmup = n_warmup,
      n_chains = n_chains,
      n_threads = n_threads,
      n_params = n_params
    ),

    # Timing
    timing = list(
      elapsed_seconds = elapsed,
      samples_per_second = samples_per_sec,
      iterations_per_second = n_iter * n_chains / elapsed
    ),

    # Diagnostics
    diagnostics = list(
      n_divergent = n_divergent,
      divergent_pct = if (!is.na(n_divergent)) 100 * n_divergent / n_samples else NA
    )
  )

  class(result) <- c("ratiod_benchmark", "list")
  result
}


#' @rdname ratiod_benchmark
#' @param x A ratiod_benchmark object
#' @param ... Ignored
#'
#' @export
print.ratiod_benchmark <- function(x, ...) {
  cat("tulpaRatio Benchmark Results\n")
  cat("=======================\n\n")

  cat("Configuration:\n")
  cat(sprintf("  Observations:   %d\n", x$config$N))
  cat(sprintf("  Predictors:     %d\n", x$config$p))
  cat(sprintf("  RE groups:      %d\n", x$config$n_groups))
  cat(sprintf("  Parameters:     %d\n", x$config$n_params))
  cat(sprintf("  Iterations:     %d (%d warmup)\n", x$config$n_iter, x$config$n_warmup))
  cat(sprintf("  Chains:         %d\n", x$config$n_chains))
  cat(sprintf("  Threads:        %d\n", x$config$n_threads))

  cat("\nTiming:\n")
  cat(sprintf("  Total time:     %.1f seconds\n", x$timing$elapsed_seconds))
  cat(sprintf("  Samples/sec:    %.1f\n", x$timing$samples_per_second))
  cat(sprintf("  Iter/sec:       %.1f\n", x$timing$iterations_per_second))

  if (!is.na(x$diagnostics$n_divergent)) {
    cat("\nDiagnostics:\n")
    cat(sprintf("  Divergent:      %d (%.2f%%)\n",
                x$diagnostics$n_divergent, x$diagnostics$divergent_pct))
  }

  invisible(x)
}


#' Compare Benchmarks Across Configurations
#'
#' @description
#' Run benchmarks across multiple configurations to compare performance.
#'
#' @param configs A list of configuration lists, each with N, p, n_groups
#' @param n_iter Iterations per benchmark
#' @param n_warmup Warmup iterations
#' @param n_threads Number of threads
#'
#' @return A data frame with benchmark results
#'
#' @examples
#' \donttest{
#' # Compare benchmarks
#' configs <- list(
#'   list(N = 50, p = 2, n_groups = 0),
#'   list(N = 100, p = 2, n_groups = 0)
#' )
#' comparison <- ratiod_benchmark_compare(configs, n_iter = 200, n_warmup = 100)
#' print(comparison)
#' }
#'
#' @export
ratiod_benchmark_compare <- function(
    configs,
    n_iter = 500,
    n_warmup = 250,
    n_threads = 1
) {

  results <- lapply(configs, function(cfg) {
    bench <- ratiod_benchmark(
      N = cfg$N,
      p = cfg$p,
      n_groups = cfg$n_groups %||% 0,
      n_iter = n_iter,
      n_warmup = n_warmup,
      n_threads = n_threads,
      verbose = FALSE
    )

    data.frame(
      N = cfg$N,
      p = cfg$p,
      n_groups = cfg$n_groups %||% 0,
      n_params = bench$config$n_params,
      elapsed_sec = bench$timing$elapsed_seconds,
      samples_per_sec = bench$timing$samples_per_second,
      iter_per_sec = bench$timing$iterations_per_second,
      n_divergent = bench$diagnostics$n_divergent
    )
  })

  do.call(rbind, results)
}


#' OpenMP Thread Count
#'
#' @description
#' Returns the maximum number of OpenMP threads available.
#'
#' @return Integer number of threads
#'
#' @examples
#' ratiod_threads()
#'
#' @export
ratiod_threads <- function() {
  cpp_get_max_threads()
}
