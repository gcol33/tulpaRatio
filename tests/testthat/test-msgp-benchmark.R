# Benchmark tests for MSGP sampler strategies
# Tests correctness and compares performance of different samplers

# Helper function to generate spatial data
generate_msgp_data <- function(n = 50, seed = 123) {
  set.seed(seed)

  # Generate coordinates on unit square
  coords <- data.frame(
    x = runif(n),
    y = runif(n)
  )

  # Generate true spatial effects (local + regional patterns)
  dist_mat <- as.matrix(dist(coords))

  # Local effect (short range)
  sigma_local <- 0.5
  phi_local <- 0.1
  K_local <- sigma_local^2 * exp(-dist_mat / phi_local)
  diag(K_local) <- diag(K_local) + 1e-6
  L_local <- chol(K_local)
  w_local <- as.numeric(t(L_local) %*% rnorm(n))

  # Regional effect (long range)
  sigma_regional <- 0.3
  phi_regional <- 0.5
  K_regional <- sigma_regional^2 * exp(-dist_mat / phi_regional)
  diag(K_regional) <- diag(K_regional) + 1e-6
  L_regional <- chol(K_regional)
  w_regional <- as.numeric(t(L_regional) %*% rnorm(n))

  # Combined spatial effect
  w_total <- w_local + w_regional

  # Generate counts
  denom <- rpois(n, 100)
  num <- rpois(n, exp(log(denom) + 0.5 + w_total))

  data.frame(
    x = coords$x,
    y = coords$y,
    num = num,
    denom = denom
  )
}

# `sampler` selects warmup mass-matrix adaptation for the two GP blocks. It
# used to offer five parameterization names beside those two, all of which
# produced byte-identical sampling because the field is centred wherever the
# density or the gradient reads it (gcol33/tulpaRatio#36); a name that changes
# nothing has to be rejected rather than accepted.
test_that("all MSGP sampler options run without error", {
  skip_on_cran()

  df <- generate_msgp_data(n = 30, seed = 42)

  samplers <- c("auto", "lbfgs")

  for (gone in c("noncentered", "centered", "interweaved", "adaptive",
                 "riemannian")) {
    expect_error(
      spatial_multiscale(~ x + y, sampler = gone),
      "should be one of",
      label = sprintf("sampler = %s", gone)
    )
  }

  for (sampler in samplers) {
    # Create spatial specification with sampler option
    ms <- spatial_multiscale(
      ~ x + y,
      m = 6,
      sampler = sampler
    )

    expect_equal(ms$sampler, sampler)

    # Fit model with minimal iterations (just test it runs)
    fit <- suppressWarnings(
      tratio(
        num | denom ~ 1,
        data = df,
        spatial = ms,
        family = ratiod_poisson_gamma(),
        mode = "hmc",
        control = list(chains = 1, iter = 100, warmup = 50, verbose = FALSE)
      )
    )

    expect_s3_class(fit, "ratiod_fit")

    # Check fit succeeded (has n_eff or similar diagnostic)
    expect_true(!is.null(fit$diagnostics) || !is.null(fit$fit))
  }
})

# Benchmark test for divergence rates
test_that("MSGP samplers have acceptable divergence rates", {
  skip_on_cran()
  skip_unless_slow()

  df <- generate_msgp_data(n = 40, seed = 123)

  ms_nc <- spatial_multiscale(~ x + y, m = 6, sampler = "auto")

  fit_nc <- suppressWarnings(
    tratio(
      num | denom ~ 1,
      data = df,
      spatial = ms_nc,
      family = ratiod_poisson_gamma(),
      mode = "hmc",
      control = list(chains = 1, iter = 1250, warmup = 500, seed = 123,
                     verbose = FALSE)
    )
  )

  divergent <- fit_nc$diagnostics$divergent
  n_divergent <- sum(divergent)
  n_samples <- nrow(as.matrix(fit_nc$draws))
  expect_length(divergent, n_samples)
  expect_true(all(divergent %in% c(0L, 1L)))
  expect_equal(n_samples, 750L)
  expect_equal(n_divergent, fit_nc$diagnostics$n_divergent)
  expect_true(n_divergent / n_samples < 0.05)  # Less than 5% divergence rate
})

# Speed benchmark
test_that("MSGP sampler speed comparison", {
  skip_on_cran()
  skip_unless_slow()

  df <- generate_msgp_data(n = 50, seed = 456)

  samplers_to_test <- c("auto", "lbfgs")
  results <- list()

  for (sampler in samplers_to_test) {
    ms <- spatial_multiscale(~ x + y, m = 6, sampler = sampler)

    start_time <- Sys.time()

    fit <- suppressWarnings(
      tratio(
        num | denom ~ 1,
        data = df,
        spatial = ms,
        family = ratiod_poisson_gamma(),
        mode = "hmc",
        control = list(chains = 1, iter = 800, warmup = 400, seed = 456,
                       verbose = FALSE)
      )
    )

    end_time <- Sys.time()
    elapsed <- as.numeric(difftime(end_time, start_time, units = "secs"))

    n_samples <- nrow(as.matrix(fit$draws))
    n_divergent <- sum(fit$diagnostics$divergent)
    expect_equal(n_samples, 400L)
    expect_length(fit$diagnostics$divergent, n_samples)
    expect_equal(n_divergent, fit$diagnostics$n_divergent)
    expect_true(is.finite(elapsed) && elapsed > 0)

    results[[sampler]] <- list(
      time = elapsed,
      n_samples = n_samples,
      n_divergent = n_divergent,
      samples_per_sec = n_samples / elapsed
    )
  }

  # Print results for manual review
  cat("\n\n=== MSGP Sampler Benchmark Results ===\n")
  for (sampler in names(results)) {
    r <- results[[sampler]]
    div_pct <- 100 * r$n_divergent / r$n_samples
    cat(sprintf("%s: %.1f sec, %d samples, %.1f samples/sec, %d div (%.1f%%)\n",
                sampler, r$time, r$n_samples, r$samples_per_sec, r$n_divergent, div_pct))
  }
  cat("=====================================\n\n")

  expect_true(all(vapply(results, function(x) {
    is.finite(x$samples_per_sec) && x$samples_per_sec > 0
  }, logical(1))))
})

# Stress test with multiple seeds
test_that("MSGP is robust across seeds", {
  skip_on_cran()
  skip_unless_slow()

  seeds <- c(111, 222, 333, 444, 555)
  all_ok <- TRUE

  for (seed in seeds) {
    df <- generate_msgp_data(n = 35, seed = seed)

    ms <- spatial_multiscale(~ x + y, m = 6, sampler = "auto")

    fit <- tryCatch({
      suppressWarnings(
        tratio(
          num | denom ~ 1,
          data = df,
          spatial = ms,
          family = ratiod_poisson_gamma(),
          mode = "hmc",
          control = list(chains = 1, iter = 200, warmup = 100, verbose = FALSE)
        )
      )
    }, error = function(e) {
      NULL
    })

    if (is.null(fit)) {
      all_ok <- FALSE
      cat(sprintf("Seed %d: FAILED (error)\n", seed))
    } else {
      n_samples <- nrow(as.matrix(fit$draws))
      n_divergent <- sum(fit$diagnostics$divergent)
      expect_equal(n_samples, 100L)
      expect_length(fit$diagnostics$divergent, n_samples)
      expect_equal(n_divergent, fit$diagnostics$n_divergent)
      div_pct <- 100 * n_divergent / n_samples

      if (div_pct > 50) {
        all_ok <- FALSE
        cat(sprintf("Seed %d: FAILED (%d/%d divergent, %.1f%%)\n",
                    seed, n_divergent, n_samples, div_pct))
      } else {
        cat(sprintf("Seed %d: OK (%d samples, %d div, %.1f%%)\n",
                    seed, n_samples, n_divergent, div_pct))
      }
    }
  }

  expect_true(all_ok)
})
