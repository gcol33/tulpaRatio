# test_gp_hsgp_consistency.R — Verify GP/HSGP prior consistency (H vs A_r)
library(tulpaRatio)
cat("=== GP/HSGP Prior Consistency: H vs A_r ===\n\n")

# Setup data with spatial structure
set.seed(42)
N <- 100
coords <- cbind(runif(N), runif(N))
eta <- 1.0 + 0.3 * rnorm(N)

# NegBin data
df <- data.frame(
  x = rnorm(N),
  coord_x = coords[, 1],
  coord_y = coords[, 2]
)
df$y_num <- rnbinom(N, size = 5, mu = exp(eta))
df$y_denom <- rnbinom(N, size = 5, mu = exp(eta + 0.3))

# Test 1: GP spatial — H vs A_r
cat("--- Test 1: spatial_gp() — H vs A_r ---\n")
tryCatch({
  set.seed(123)
  fit_H <- tratio(y_num | y_denom ~ x, data = df, family = ratiod_negbin_negbin(),
    spatial = spatial_gp(~ coord_x + coord_y),
    control = list(iter = 100, warmup = 50, chains = 1, gradient_mode = "H", verbose = FALSE))

  set.seed(123)
  fit_Ar <- tratio(y_num | y_denom ~ x, data = df, family = ratiod_negbin_negbin(),
    spatial = spatial_gp(~ coord_x + coord_y),
    control = list(iter = 100, warmup = 50, chains = 1, gradient_mode = "A_r", verbose = FALSE))

  draws_H <- as.matrix(fit_H$draws)
  draws_Ar <- as.matrix(fit_Ar$draws)
  nan_H <- sum(is.nan(draws_H)); nan_Ar <- sum(is.nan(draws_Ar))
  cat(sprintf("  H: NaN=%d, A_r: NaN=%d\n", nan_H, nan_Ar))

  if (nan_H == 0 && nan_Ar == 0) {
    max_diff <- max(abs(draws_H - draws_Ar))
    cat(sprintf("  Max |draws_H - draws_Ar| = %.10f\n", max_diff))
    if (max_diff < 1e-6) cat("  PASS: Exact match\n")
    else if (max_diff < 0.01) cat("  PASS: Close match (< 0.01)\n")
    else cat("  FAIL: Draws differ\n")
  }
}, error = function(e) cat(sprintf("  ERROR: %s\n", e$message)))

# Test 2: HSGP spatial — H vs A_r
cat("\n--- Test 2: spatial_hsgp() — H vs A_r ---\n")
tryCatch({
  set.seed(123)
  fit_H2 <- tratio(y_num | y_denom ~ x, data = df, family = ratiod_negbin_negbin(),
    spatial = spatial_hsgp(~ coord_x + coord_y),
    control = list(iter = 100, warmup = 50, chains = 1, gradient_mode = "H", verbose = FALSE))

  set.seed(123)
  fit_Ar2 <- tratio(y_num | y_denom ~ x, data = df, family = ratiod_negbin_negbin(),
    spatial = spatial_hsgp(~ coord_x + coord_y),
    control = list(iter = 100, warmup = 50, chains = 1, gradient_mode = "A_r", verbose = FALSE))

  draws_H2 <- as.matrix(fit_H2$draws)
  draws_Ar2 <- as.matrix(fit_Ar2$draws)
  nan_H2 <- sum(is.nan(draws_H2)); nan_Ar2 <- sum(is.nan(draws_Ar2))
  cat(sprintf("  H: NaN=%d, A_r: NaN=%d\n", nan_H2, nan_Ar2))

  if (nan_H2 == 0 && nan_Ar2 == 0) {
    max_diff2 <- max(abs(draws_H2 - draws_Ar2))
    cat(sprintf("  Max |draws_H - draws_Ar| = %.10f\n", max_diff2))
    if (max_diff2 < 1e-6) cat("  PASS: Exact match\n")
    else if (max_diff2 < 0.01) cat("  PASS: Close match (< 0.01)\n")
    else cat("  FAIL: Draws differ\n")
  }
}, error = function(e) cat(sprintf("  ERROR: %s\n", e$message)))

# Test 3: Multiscale GP — H vs A_r
cat("\n--- Test 3: spatial_multiscale() — H vs A_r ---\n")
tryCatch({
  set.seed(123)
  fit_H3 <- tratio(y_num | y_denom ~ x, data = df, family = ratiod_negbin_negbin(),
    spatial = spatial_multiscale(~ coord_x + coord_y),
    control = list(iter = 100, warmup = 50, chains = 1, gradient_mode = "H", verbose = FALSE))

  set.seed(123)
  fit_Ar3 <- tratio(y_num | y_denom ~ x, data = df, family = ratiod_negbin_negbin(),
    spatial = spatial_multiscale(~ coord_x + coord_y),
    control = list(iter = 100, warmup = 50, chains = 1, gradient_mode = "A_r", verbose = FALSE))

  draws_H3 <- as.matrix(fit_H3$draws)
  draws_Ar3 <- as.matrix(fit_Ar3$draws)
  nan_H3 <- sum(is.nan(draws_H3)); nan_Ar3 <- sum(is.nan(draws_Ar3))
  cat(sprintf("  H: NaN=%d, A_r: NaN=%d\n", nan_H3, nan_Ar3))

  if (nan_H3 == 0 && nan_Ar3 == 0) {
    max_diff3 <- max(abs(draws_H3 - draws_Ar3))
    cat(sprintf("  Max |draws_H - draws_Ar| = %.10f\n", max_diff3))
    if (max_diff3 < 1e-6) cat("  PASS: Exact match\n")
    else if (max_diff3 < 0.01) cat("  PASS: Close match (< 0.01)\n")
    else cat("  FAIL: Draws differ\n")
  }
}, error = function(e) cat(sprintf("  ERROR: %s\n", e$message)))

# Test 4: GP + negbin — verify parameter recovery
cat("\n--- Test 4: GP parameter recovery (longer chain) ---\n")
tryCatch({
  set.seed(42)
  N2 <- 200
  coords2 <- cbind(runif(N2, 0, 10), runif(N2, 0, 10))

  # Simulate GP field
  dist_mat <- as.matrix(dist(coords2))
  true_sigma2 <- 1.0
  true_phi <- 2.0
  Sigma <- true_sigma2 * exp(-dist_mat / true_phi) + diag(1e-6, N2)
  gp_field <- MASS::mvrnorm(1, rep(0, N2), Sigma)

  df2 <- data.frame(
    x = rnorm(N2),
    coord_x = coords2[, 1],
    coord_y = coords2[, 2]
  )
  true_beta <- c(1.0, 0.3)
  eta2 <- true_beta[1] + true_beta[2] * df2$x + gp_field
  df2$y_num <- rnbinom(N2, size = 5, mu = exp(eta2))
  df2$y_denom <- rnbinom(N2, size = 5, mu = exp(eta2 + 0.3))

  t4 <- system.time({
    set.seed(123)
    fit4 <- tratio(y_num | y_denom ~ x, data = df2, family = ratiod_negbin_negbin(),
      spatial = spatial_gp(~ coord_x + coord_y),
      control = list(iter = 1000, warmup = 500, chains = 2, verbose = FALSE))
  })["elapsed"]

  draws4 <- fit4$draws
  summ4 <- posterior::summarise_draws(draws4, "mean", "sd", "ess_bulk", "rhat")
  key4 <- summ4[grepl("beta_num|sigma2_gp|phi_gp|phi_num|phi_denom", summ4$variable), ]

  cat(sprintf("  Time: %.1fs\n", t4))
  cat("  Parameter recovery:\n")
  cat(sprintf("    %-20s mean=%.3f (true=%.1f)\n", "beta_num[1]",
    key4$mean[key4$variable == "beta_num[1]"], true_beta[1]))
  cat(sprintf("    %-20s mean=%.3f (true=%.1f)\n", "beta_num[2]",
    key4$mean[key4$variable == "beta_num[2]"], true_beta[2]))

  # Check GP hyperparameters if available
  if ("sigma2_gp" %in% key4$variable) {
    cat(sprintf("    %-20s mean=%.3f (true=%.1f)\n", "sigma2_gp",
      key4$mean[key4$variable == "sigma2_gp"], true_sigma2))
  }
  if ("phi_gp" %in% key4$variable) {
    cat(sprintf("    %-20s mean=%.3f (true=%.1f)\n", "phi_gp",
      key4$mean[key4$variable == "phi_gp"], true_phi))
  }

  # ESS check
  min_ess4 <- min(key4$ess_bulk, na.rm = TRUE)
  max_rhat4 <- max(key4$rhat, na.rm = TRUE)
  cat(sprintf("\n  Min ESS: %.0f, Max Rhat: %.4f\n", min_ess4, max_rhat4))
}, error = function(e) cat(sprintf("  ERROR: %s\n", e$message)))

cat("\n=== Done ===\n")
