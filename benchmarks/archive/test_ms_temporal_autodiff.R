# test_ms_temporal_autodiff.R — Verify A_t/A_r/A modes work for multiscale temporal
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

cat("=== Multiscale Temporal Autodiff Verification ===\n\n")

# Test 1: H mode (reference — already validated)
cat("--- Test 1: H mode (reference) ---\n")
tryCatch({
  set.seed(123)
  fit_H <- tratio(y_num | y_denom ~ x, data = df, family = ratiod_negbin_negbin(),
    temporal = temporal_multiscale("time", trend = "rw1", seasonal = 12, short_term = "ar1"),
    control = list(iter = 50, warmup = 25, chains = 1, gradient_mode = "H", verbose = FALSE))
  draws_H <- as.matrix(fit_H$draws)
  nan_H <- sum(is.nan(draws_H))
  cat(sprintf("  H mode: OK, NaN: %d, params: %d\n", nan_H, ncol(draws_H)))
}, error = function(e) cat(sprintf("  ERROR: %s\n", e$message)))

# Test 2: A mode (forward autodiff)
cat("\n--- Test 2: A mode (forward autodiff) ---\n")
tryCatch({
  set.seed(123)
  fit_A <- tratio(y_num | y_denom ~ x, data = df, family = ratiod_negbin_negbin(),
    temporal = temporal_multiscale("time", trend = "rw1", seasonal = 12, short_term = "ar1"),
    control = list(iter = 50, warmup = 25, chains = 1, gradient_mode = "A", verbose = FALSE))
  draws_A <- as.matrix(fit_A$draws)
  nan_A <- sum(is.nan(draws_A))
  cat(sprintf("  A mode: OK, NaN: %d\n", nan_A))

  if (nan_A == 0 && nan_H == 0) {
    max_diff <- max(abs(colMeans(draws_A) - colMeans(draws_H)))
    cat(sprintf("  Max |mean_A - mean_H| = %.6f\n", max_diff))
    if (max_diff < 0.5) cat("  PASS: A matches H\n") else cat("  CHECK: means differ\n")
  }
}, error = function(e) cat(sprintf("  ERROR: %s\n", e$message)))

# Test 3: A_t mode (tape autodiff — was segfaulting)
cat("\n--- Test 3: A_t mode (tape autodiff — previously segfaulted) ---\n")
tryCatch({
  set.seed(123)
  fit_At <- tratio(y_num | y_denom ~ x, data = df, family = ratiod_negbin_negbin(),
    temporal = temporal_multiscale("time", trend = "rw1", seasonal = 12, short_term = "ar1"),
    control = list(iter = 50, warmup = 25, chains = 1, gradient_mode = "A_t", verbose = FALSE))
  draws_At <- as.matrix(fit_At$draws)
  nan_At <- sum(is.nan(draws_At))
  cat(sprintf("  A_t mode: OK, NaN: %d\n", nan_At))

  if (nan_At == 0 && nan_H == 0) {
    max_diff <- max(abs(colMeans(draws_At) - colMeans(draws_H)))
    cat(sprintf("  Max |mean_At - mean_H| = %.6f\n", max_diff))
    if (max_diff < 0.5) cat("  PASS: A_t matches H\n") else cat("  CHECK: means differ\n")
  }
}, error = function(e) cat(sprintf("  ERROR: %s\n", e$message)))

# Test 4: A_r mode (arena autodiff)
cat("\n--- Test 4: A_r mode (arena autodiff) ---\n")
tryCatch({
  set.seed(123)
  fit_Ar <- tratio(y_num | y_denom ~ x, data = df, family = ratiod_negbin_negbin(),
    temporal = temporal_multiscale("time", trend = "rw1", seasonal = 12, short_term = "ar1"),
    control = list(iter = 50, warmup = 25, chains = 1, gradient_mode = "A_r", verbose = FALSE))
  draws_Ar <- as.matrix(fit_Ar$draws)
  nan_Ar <- sum(is.nan(draws_Ar))
  cat(sprintf("  A_r mode: OK, NaN: %d\n", nan_Ar))

  if (nan_Ar == 0 && nan_H == 0) {
    max_diff <- max(abs(colMeans(draws_Ar) - colMeans(draws_H)))
    cat(sprintf("  Max |mean_Ar - mean_H| = %.6f\n", max_diff))
    if (max_diff < 0.5) cat("  PASS: A_r matches H\n") else cat("  CHECK: means differ\n")
  }
}, error = function(e) cat(sprintf("  ERROR: %s\n", e$message)))

# Test 5: Trend only (simpler case)
cat("\n--- Test 5: Trend only (RW1), A_t mode ---\n")
tryCatch({
  set.seed(123)
  fit_t5 <- tratio(y_num | y_denom ~ x, data = df, family = ratiod_negbin_negbin(),
    temporal = temporal_multiscale("time", trend = "rw1", short_term = "none"),
    control = list(iter = 50, warmup = 25, chains = 1, gradient_mode = "A_t", verbose = FALSE))
  draws_t5 <- as.matrix(fit_t5$draws)
  nan_t5 <- sum(is.nan(draws_t5))
  cat(sprintf("  A_t mode (trend only): OK, NaN: %d\n", nan_t5))
}, error = function(e) cat(sprintf("  ERROR: %s\n", e$message)))

# Test 6: RW2 + IID, A_t mode
cat("\n--- Test 6: RW2 trend + IID short, A_t mode ---\n")
tryCatch({
  set.seed(123)
  fit_t6 <- tratio(y_num | y_denom ~ x, data = df, family = ratiod_negbin_negbin(),
    temporal = temporal_multiscale("time", trend = "rw2", short_term = "iid"),
    control = list(iter = 50, warmup = 25, chains = 1, gradient_mode = "A_t", verbose = FALSE))
  draws_t6 <- as.matrix(fit_t6$draws)
  nan_t6 <- sum(is.nan(draws_t6))
  cat(sprintf("  A_t mode (rw2+iid): OK, NaN: %d\n", nan_t6))
}, error = function(e) cat(sprintf("  ERROR: %s\n", e$message)))

cat("\n=== Done ===\n")
