# test_ms_temporal_gradient.R — Verify H gradient matches A for multiscale temporal
library(tulpaRatio)
set.seed(42)
N <- 200
T_times <- 24
df <- data.frame(
  x = rnorm(N),
  time = rep(1:T_times, length.out = N),
  site = factor(sample(1:10, N, replace = TRUE))
)

eta <- 1.5 + 0.3 * df$x
df$y_num <- rnbinom(N, size = 5, mu = exp(eta))
df$y_denom <- rnbinom(N, size = 5, mu = exp(eta + 0.5))

cat("=== Multiscale Temporal Gradient Verification ===\n\n")

# Test 1: Full multiscale (RW1 trend + seasonal(12) + AR1 short)
cat("--- Test 1: Full multiscale (RW1 trend + seasonal + AR1 short) ---\n")
tryCatch({
  set.seed(123)
  fit_A <- tratio(y_num | y_denom ~ x, data = df, family = ratiod_negbin_negbin(),
    temporal = temporal_multiscale("time", trend = "rw1", seasonal = 12, short_term = "ar1"),
    control = list(iter = 20, warmup = 10, chains = 1, gradient_mode = "A", verbose = FALSE))
  cat("  A mode: OK (no NaN)\n")

  set.seed(123)
  fit_H <- tratio(y_num | y_denom ~ x, data = df, family = ratiod_negbin_negbin(),
    temporal = temporal_multiscale("time", trend = "rw1", seasonal = 12, short_term = "ar1"),
    control = list(iter = 20, warmup = 10, chains = 1, gradient_mode = "H", verbose = FALSE))
  cat("  H mode: OK (no NaN)\n")

  draws_A <- as.matrix(fit_A$draws)
  draws_H <- as.matrix(fit_H$draws)
  nan_A <- sum(is.nan(draws_A)); nan_H <- sum(is.nan(draws_H))
  cat(sprintf("  A mode NaN: %d, H mode NaN: %d\n", nan_A, nan_H))

  if (nan_A == 0 && nan_H == 0) {
    means_A <- colMeans(draws_A)
    means_H <- colMeans(draws_H)
    max_diff <- max(abs(means_A - means_H))
    cat(sprintf("  Max |mean_A - mean_H| = %.6f\n", max_diff))
    if (max_diff < 0.5) cat("  PASS: H matches A\n") else cat("  CHECK: means differ\n")
  }
}, error = function(e) cat(sprintf("  ERROR: %s\n", e$message)))

# Test 2: Trend only (RW1)
cat("\n--- Test 2: Trend only (RW1) ---\n")
tryCatch({
  set.seed(123)
  fit_A2 <- tratio(y_num | y_denom ~ x, data = df, family = ratiod_negbin_negbin(),
    temporal = temporal_multiscale("time", trend = "rw1", short_term = "none"),
    control = list(iter = 20, warmup = 10, chains = 1, gradient_mode = "A", verbose = FALSE))
  cat("  A mode: OK\n")

  set.seed(123)
  fit_H2 <- tratio(y_num | y_denom ~ x, data = df, family = ratiod_negbin_negbin(),
    temporal = temporal_multiscale("time", trend = "rw1", short_term = "none"),
    control = list(iter = 20, warmup = 10, chains = 1, gradient_mode = "H", verbose = FALSE))
  cat("  H mode: OK\n")

  draws_A2 <- as.matrix(fit_A2$draws)
  draws_H2 <- as.matrix(fit_H2$draws)
  nan_A2 <- sum(is.nan(draws_A2)); nan_H2 <- sum(is.nan(draws_H2))
  cat(sprintf("  A mode NaN: %d, H mode NaN: %d\n", nan_A2, nan_H2))
  if (nan_A2 == 0 && nan_H2 == 0) {
    max_diff2 <- max(abs(colMeans(draws_A2) - colMeans(draws_H2)))
    cat(sprintf("  Max |mean_A - mean_H| = %.6f\n", max_diff2))
    if (max_diff2 < 0.5) cat("  PASS\n") else cat("  CHECK\n")
  }
}, error = function(e) cat(sprintf("  ERROR: %s\n", e$message)))

# Test 3: RW2 trend + IID short
cat("\n--- Test 3: RW2 trend + IID short ---\n")
tryCatch({
  set.seed(123)
  fit_A3 <- tratio(y_num | y_denom ~ x, data = df, family = ratiod_negbin_negbin(),
    temporal = temporal_multiscale("time", trend = "rw2", short_term = "iid"),
    control = list(iter = 20, warmup = 10, chains = 1, gradient_mode = "A", verbose = FALSE))
  cat("  A mode: OK\n")

  set.seed(123)
  fit_H3 <- tratio(y_num | y_denom ~ x, data = df, family = ratiod_negbin_negbin(),
    temporal = temporal_multiscale("time", trend = "rw2", short_term = "iid"),
    control = list(iter = 20, warmup = 10, chains = 1, gradient_mode = "H", verbose = FALSE))
  cat("  H mode: OK\n")

  draws_A3 <- as.matrix(fit_A3$draws)
  draws_H3 <- as.matrix(fit_H3$draws)
  nan_A3 <- sum(is.nan(draws_A3)); nan_H3 <- sum(is.nan(draws_H3))
  cat(sprintf("  A mode NaN: %d, H mode NaN: %d\n", nan_A3, nan_H3))
  if (nan_A3 == 0 && nan_H3 == 0) {
    max_diff3 <- max(abs(colMeans(draws_A3) - colMeans(draws_H3)))
    cat(sprintf("  Max |mean_A - mean_H| = %.6f\n", max_diff3))
    if (max_diff3 < 0.5) cat("  PASS\n") else cat("  CHECK\n")
  }
}, error = function(e) cat(sprintf("  ERROR: %s\n", e$message)))

# Test 4: Timing comparison
cat("\n--- Test 4: Timing (500 iter, full multiscale) ---\n")
tryCatch({
  t_A <- system.time({
    set.seed(42)
    tratio(y_num | y_denom ~ x, data = df, family = ratiod_negbin_negbin(),
      temporal = temporal_multiscale("time", trend = "rw1", seasonal = 12, short_term = "ar1"),
      control = list(iter = 500, warmup = 250, chains = 1, gradient_mode = "A", verbose = FALSE))
  })["elapsed"]

  t_H <- system.time({
    set.seed(42)
    tratio(y_num | y_denom ~ x, data = df, family = ratiod_negbin_negbin(),
      temporal = temporal_multiscale("time", trend = "rw1", seasonal = 12, short_term = "ar1"),
      control = list(iter = 500, warmup = 250, chains = 1, gradient_mode = "H", verbose = FALSE))
  })["elapsed"]

  cat(sprintf("  A mode: %.1fs\n  H mode: %.1fs\n  Speedup: %.1fx\n", t_A, t_H, t_A / t_H))
}, error = function(e) cat(sprintf("  ERROR: %s\n", e$message)))

cat("\n=== Done ===\n")
